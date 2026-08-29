/* src/stoke.c -- the public ABI, implemented as a thin composition.
 *
 * ▶ THIN ON PURPOSE. Every function here is argument validation plus a call
 * into sched.c / tok.c / model.c. Nothing in this file decides anything about
 * inference. If a function here grows a policy, that policy belongs in the
 * layer below and this file is the wrong place to have noticed it.
 *
 * ▶▶ THE ONE PIECE OF REAL WORK IS `stoke_open`, AND IT IS THE TRAP.
 * Opening touches five subsystems, any of which fatal()s on malformed input.
 * A CLI is entitled to die there; a library is not. So the whole construction
 * runs under a StokeTrap (see common.h) and the failure path tears down
 * whatever was built. Every pointer the teardown reads is `volatile`, because a
 * non-volatile local modified between `setjmp` and `longjmp` is indeterminate
 * afterwards -- and it would be indeterminate ONLY on the failure path, which
 * is the path least likely to be exercised and most likely to be trusted.
 */
#include "common.h"

#include "gguf.h"
#include "kv.h"
#include "model.h"
#include "pool.h"
#include "sched.h"
#include "serve.h"
#include "stoke.h"
#include "tok.h"

#include <string.h>

#define STOKE_DEFAULT_SEQS 8u

struct stoke_engine {
    GgufFile*  gf;
    Tok*       tok;
    Model*     model;
    KvStore*   store;
    ServeHost* host;
    Sched*     sched;
    u32        max_ctx;
};

/* ---- errors ------------------------------------------------------------
 *
 * The message buffer is common.h's `stoke_errmsg`, the same one fatal() writes
 * through when a trap is armed. Sharing it is the point: a trapped load
 * failure and an argument-validation failure are read by the caller in exactly
 * one place, so a binding never has to ask which kind it got. */
static stoke_status fail(stoke_status code, const char* msg) {
    snprintf(stoke_errmsg, sizeof stoke_errmsg, "%s", msg);
    return code;
}

const char* stoke_last_error(void) { return stoke_errmsg; }

int32_t stoke_abi_version(void) { return STOKE_ABI_VERSION; }

/* ---- open / close ------------------------------------------------------ */

/* Reads a possibly-older caller's opts struct. A caller compiled against a
 * future header may pass a LARGER struct (fields we ignore); one compiled
 * against an older header passes a SMALLER one, and every field past its `size`
 * must take the default rather than be read out of bounds. That is the entire
 * job of the `size` field and it is worth nothing unless it is checked here. */
static uint32_t opt_u32(const stoke_open_opts* o, size_t off, uint32_t dflt) {
    if (!o || o->size < off + sizeof(uint32_t)) return dflt;
    uint32_t v;
    memcpy(&v, (const char*)o + off, sizeof v);
    return v ? v : dflt;
}
static int32_t opt_i32(const stoke_open_opts* o, size_t off, int32_t dflt) {
    if (!o || o->size < off + sizeof(int32_t)) return dflt;
    int32_t v;
    memcpy(&v, (const char*)o + off, sizeof v);
    return v;
}

stoke_engine* stoke_open_abi(const char* gguf_path, const stoke_open_opts* opts,
                             int32_t caller_abi_version) {
    if (caller_abi_version != STOKE_ABI_VERSION) {
        snprintf(stoke_errmsg, sizeof stoke_errmsg,
                 "stoke_open: caller built against ABI %d, library is ABI %d",
                 (int)caller_abi_version, (int)STOKE_ABI_VERSION);
        return NULL;
    }
    if (!gguf_path || !gguf_path[0]) {
        snprintf(stoke_errmsg, sizeof stoke_errmsg,
                 "stoke_open: gguf_path must be a non-empty path");
        return NULL;
    }
    if (opts && opts->size < sizeof(size_t)) {
        snprintf(stoke_errmsg, sizeof stoke_errmsg,
                 "stoke_open: opts->size is %zu -- set it to "
                 "sizeof(stoke_open_opts)",
                 opts->size);
        return NULL;
    }

    /* ⚠ EVERY ONE OF THESE IS `volatile`. They are written between the setjmp
     * and a possible longjmp and read by the teardown afterwards; without
     * volatile their values there are indeterminate, and the compiler is
     * entitled to have kept them in registers the longjmp restored. The bug
     * that would produce -- a teardown freeing a stale pointer, only on the
     * error path -- is exactly the kind that ships. */
    GgufFile* volatile  gf    = NULL;
    Tok* volatile       tok   = NULL;
    Model* volatile     model = NULL;
    KvStore* volatile   store = NULL;
    ServeHost* volatile host  = NULL;
    Sched* volatile     sched = NULL;
    stoke_engine* volatile e  = NULL;

    StokeTrap trap;
    stoke_trap_push(&trap);
    if (STOKE_TRAP_ARMED(trap)) {
        uint32_t want_ctx  = opt_u32(opts, offsetof(stoke_open_opts, max_ctx), 0);
        uint32_t max_seqs  = opt_u32(opts, offsetof(stoke_open_opts, max_seqs),
                                     STOKE_DEFAULT_SEQS);
        uint32_t n_threads = opt_u32(opts, offsetof(stoke_open_opts, n_threads), 0);
        int32_t  kv_q8     = opt_i32(opts, offsetof(stoke_open_opts, kv_q8), 0);
        int32_t  dedup     = opt_i32(opts, offsetof(stoke_open_opts, dedup_prefix), 0);

        /* Same order main.c uses for `bench --serve`, deliberately: this is a
         * second caller of a stack that is already gated, not a new one. */
        pool_start(n_threads);
        gf    = gguf_open(gguf_path);
        tok   = tok_from_gguf(gf);
        model = model_open(gguf_path);

        u32 ctx = want_ctx ? want_ctx : model_max_ctx(model);
        if (ctx > model_max_ctx(model)) {
            fatal("stoke_open: max_ctx %u exceeds the model's %u", ctx,
                  model_max_ctx(model));
        }

        store = kv_store_open(model_n_layer(model),
                              max_seqs * kv_blocks_for_ctx(ctx),
                              model_n_head_kv(model), model_head_dim(model),
                              kv_q8 ? KV_Q8_0 : KV_F32);
        host  = serve_host_open(model);
        SchedHost sh = serve_host(host);
        sched = sched_open(store, ctx, &sh, dedup ? true : false);

        e = (stoke_engine*)xcalloc(1, sizeof(stoke_engine));
        e->gf = gf; e->tok = tok; e->model = model;
        e->store = store; e->host = host; e->sched = sched;
        e->max_ctx = ctx;

        stoke_trap_pop(&trap);
        return (stoke_engine*)e;
    }

    /* Trapped. `stoke_errmsg` already holds fatal()'s message. Tear down in
     * reverse, skipping whatever was never built -- each handle is NULL until
     * its constructor returned, and every constructor here either returns a
     * valid pointer or fatal()s, so non-NULL means fully built. */
    if (sched) sched_close((Sched*)sched);
    if (host)  serve_host_close((ServeHost*)host);
    if (store) kv_store_close((KvStore*)store);
    if (model) model_close((Model*)model);
    if (tok)   tok_free((Tok*)tok);
    if (gf)    gguf_close((GgufFile*)gf);
    if (e)     free((void*)e);
    return NULL;
}

void stoke_close(stoke_engine* e) {
    if (!e) return;
    sched_close(e->sched);
    serve_host_close(e->host);
    kv_store_close(e->store);
    model_close(e->model);
    tok_free(e->tok);
    gguf_close(e->gf);
    free(e);
}

/* ---- model facts ------------------------------------------------------- */

uint32_t stoke_n_vocab(const stoke_engine* e) {
    return e ? model_n_vocab(e->model) : 0u;
}
uint32_t stoke_max_ctx(const stoke_engine* e) { return e ? e->max_ctx : 0u; }
uint32_t stoke_eos_id(const stoke_engine* e) {
    return e ? tok_eos(e->tok) : 0u;
}

/* ---- requests ---------------------------------------------------------- */

stoke_status stoke_submit(stoke_engine* e, const stoke_request* req) {
    if (!e || !req) return fail(STOKE_ERR_INVALID_ARG,
                                "stoke_submit: engine and request must be non-NULL");
    if (req->size < sizeof(stoke_request)) {
        return fail(STOKE_ERR_INVALID_ARG,
                    "stoke_submit: req->size is smaller than this library's "
                    "stoke_request -- set it to sizeof(stoke_request)");
    }
    if (!req->prompt_ids || req->n_prompt == 0) {
        return fail(STOKE_ERR_INVALID_ARG,
                    "stoke_submit: prompt_ids must be non-NULL and n_prompt > 0");
    }
    if (req->max_tokens == 0) {
        return fail(STOKE_ERR_INVALID_ARG, "stoke_submit: max_tokens must be > 0");
    }
    /* Checked HERE as well as in sched_submit so the caller gets a distinct,
     * actionable code. sched_submit reports one boolean for several causes;
     * "your prompt does not fit" and "you reused a live id" are different bugs
     * with different fixes, and collapsing them into one code would make the
     * ABI worse than the internal call it wraps. */
    if ((u64)req->n_prompt + (u64)req->max_tokens > (u64)e->max_ctx) {
        return fail(STOKE_ERR_TOO_LONG,
                    "stoke_submit: n_prompt + max_tokens exceeds max_ctx");
    }

    SchedRequest sr;
    sr.id         = req->id;
    sr.prompt_ids = req->prompt_ids;
    sr.n_prompt   = req->n_prompt;
    sr.max_tokens = req->max_tokens;
    sr.eos_id     = req->eos_id;
    if (!sched_submit(e->sched, &sr)) {
        return fail(STOKE_ERR_DUPLICATE_ID,
                    "stoke_submit: an id already live was submitted again");
    }
    return STOKE_OK;
}

stoke_status stoke_step(stoke_engine* e, uint32_t* out_rows) {
    if (!e) return fail(STOKE_ERR_INVALID_ARG, "stoke_step: engine is NULL");
    u32 rows = sched_step(e->sched);
    if (out_rows) *out_rows = rows;
    return STOKE_OK;
}

int32_t stoke_poll(stoke_engine* e, uint64_t id, uint32_t* out, int32_t cap,
                   int32_t* finish) {
    if (!e || !out) return fail(STOKE_ERR_INVALID_ARG,
                                "stoke_poll: engine and out must be non-NULL");
    if (cap <= 0) return fail(STOKE_ERR_INVALID_ARG, "stoke_poll: cap must be > 0");
    SchedFinish f = SCHED_FINISH_NONE;
    u32 n = sched_poll(e->sched, id, out, (u32)cap, &f);
    if (finish) *finish = (int32_t)f;
    return (int32_t)n;
}

stoke_status stoke_cancel(stoke_engine* e, uint64_t id) {
    if (!e) return fail(STOKE_ERR_INVALID_ARG, "stoke_cancel: engine is NULL");
    if (!sched_cancel(e->sched, id)) {
        return fail(STOKE_ERR_UNKNOWN_ID, "stoke_cancel: no live request with that id");
    }
    return STOKE_OK;
}

/* ---- text -------------------------------------------------------------- */

int32_t stoke_tokenize(const stoke_engine* e, const char* text, int64_t len,
                       int32_t add_special, uint32_t* out, int32_t cap) {
    if (!e || !text || !out) {
        return fail(STOKE_ERR_INVALID_ARG,
                    "stoke_tokenize: engine, text and out must be non-NULL");
    }
    if (cap < 0) return fail(STOKE_ERR_INVALID_ARG, "stoke_tokenize: cap must be >= 0");
    u64 n = (len < 0) ? (u64)strlen(text) : (u64)len;
    return (int32_t)tok_encode(e->tok, text, n, out, (u32)cap,
                               add_special ? true : false);
}

int32_t stoke_detokenize(const stoke_engine* e, uint32_t id, char* buf,
                         int32_t cap) {
    if (!e || !buf) {
        return fail(STOKE_ERR_INVALID_ARG,
                    "stoke_detokenize: engine and buf must be non-NULL");
    }
    if (cap < 0) return fail(STOKE_ERR_INVALID_ARG, "stoke_detokenize: cap must be >= 0");
    return (int32_t)tok_decode(e->tok, id, buf, (u32)cap);
}

/* ---- chat ---------------------------------------------------------------
 *
 * ▶▶ THE WHOLE DESIGN OF THIS SECTION IS "NEVER REACH tok_apply_chatml's
 * fatal()". That function answers three questions by exiting the process: the
 * model has no ChatML specials, the cap is too small, and a length overflows.
 * Every one of those is correct for a CLI and is a REMOTE PROCESS KILL behind
 * an HTTP endpoint -- the second one is reachable by any client willing to send
 * a long enough prompt.
 *
 * ⚠ AND TRAPPING IT IS NOT THE ANSWER, WHICH IS WHY IT IS NOT USED HERE.
 * tok_apply_chatml xmalloc()s a scratch buffer per message and frees it at the
 * end of the loop body; a longjmp out of the encode would skip that free. The
 * M6 trap would have turned a process kill into a MEMORY LEAK PER FAILED
 * REQUEST, which is quieter and worse. So each condition is excluded BEFORE
 * the call instead: the specials are checked through tok_has_chatml, and the
 * cap is checked against a bound that provably covers the encoder's own
 * worst case.
 */

/* The encoder's stated worst case is 3 ids per byte of `{role}
{content}`.
 * Around that, per message: the <|im_start|> and <|im_end|> ids, and the "
"
 * separator which is itself encoded and so may cost up to 3. That is 5, and
 * this allows 8. The fixed tail is one id plus an encode of "assistant
" (10
 * bytes, so up to 30) = 31, and this allows 64.
 *
 * ⚠ The slack is deliberate and was NOT free to get wrong: an earlier version
 * of this allowed +4 per message, which is one short of the 5 actually needed,
 * and the shortfall would have surfaced as a fatal() on one particular input
 * rather than as a refusal. */
int64_t stoke_chat_template_bound(const stoke_chat_message* msgs, int32_t n) {
    if (!msgs || n <= 0) {
        return fail(STOKE_ERR_INVALID_ARG,
                    "stoke_chat_template_bound: msgs must be non-NULL and n > 0");
    }
    u64 total = 64;
    for (int32_t i = 0; i < n; i++) {
        if (!msgs[i].role || !msgs[i].content) {
            return fail(STOKE_ERR_INVALID_ARG,
                        "stoke_chat_template_bound: a message has a NULL role "
                        "or content");
        }
        u64 per = (u64)msgs[i].role_len + 1u + (u64)msgs[i].content_len;
        if (per > (UINT64_MAX - 8u) / 3u) {
            return fail(STOKE_ERR_TOO_LONG, "stoke_chat_template_bound: overflow");
        }
        per = per * 3u + 8u;
        if (total > UINT64_MAX - per) {
            return fail(STOKE_ERR_TOO_LONG, "stoke_chat_template_bound: overflow");
        }
        total += per;
    }
    if (total > (u64)INT32_MAX) {
        return fail(STOKE_ERR_TOO_LONG,
                    "stoke_chat_template_bound: bound exceeds INT32_MAX");
    }
    return (int64_t)total;
}

int32_t stoke_apply_chat_template(const stoke_engine* e,
                                  const stoke_chat_message* msgs, int32_t n,
                                  uint32_t* out, int32_t cap) {
    if (!e || !msgs || n <= 0 || !out || cap <= 0) {
        return fail(STOKE_ERR_INVALID_ARG,
                    "stoke_apply_chat_template: engine, msgs and out must be "
                    "non-NULL with n > 0 and cap > 0");
    }
    if (!tok_has_chatml(e->tok)) {
        return fail(STOKE_ERR_NO_TEMPLATE,
                    "stoke_apply_chat_template: this model's vocab has no "
                    "'<|im_start|>'/'<|im_end|>' specials");
    }

    int64_t bound = stoke_chat_template_bound(msgs, n);
    if (bound < 0) return (int32_t)bound;
    if ((int64_t)cap < bound) {
        return fail(STOKE_ERR_TOO_LONG,
                    "stoke_apply_chat_template: cap is below "
                    "stoke_chat_template_bound() for these messages");
    }

    /* Converted field by field rather than cast. The two structs happen to have
     * identical layout wherever size_t is 64 bits, and would silently stop
     * having it where it is not -- a cast here would work on this box and
     * misread every length on another. */
    StokeMsg* m = (StokeMsg*)xmalloc((size_t)n * sizeof(StokeMsg));
    for (int32_t i = 0; i < n; i++) {
        m[i].role        = msgs[i].role;
        m[i].role_len    = (u64)msgs[i].role_len;
        m[i].content     = msgs[i].content;
        m[i].content_len = (u64)msgs[i].content_len;
    }

    /* No fatal() is reachable from here -- see the section note. That is what
     * makes this allocation safe to hold across the call. */
    u32 got = tok_apply_chatml(e->tok, m, (u32)n, out, (u32)cap);
    free(m);
    return (int32_t)got;
}

int32_t stoke_has_chat_template(const stoke_engine* e) {
    return (e && tok_has_chatml(e->tok)) ? 1 : 0;
}
