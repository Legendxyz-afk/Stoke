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
#include "sample.h"
#include "serve.h"
#include "stoke.h"
#include "tok.h"

#include <pthread.h>
#include <stddef.h>
#include <string.h>

#define STOKE_DEFAULT_SEQS 8u

/* ---- sampling state ----------------------------------------------------
 *
 * One slot per LIVE request. Small and fixed: the scheduler admits at most
 * max_seqs sequences, and a request that is not admitted is not sampling.
 *
 * ⚠ THE HISTORY IS INDEXED BY POSITION, NOT APPENDED. A preempted sequence
 * is REPLAYED, so an appending history would grow a second copy of the same
 * tokens and the repeat penalty a replay computed would differ from the one
 * the original did -- silently, since the stream stays fluent either way.
 * Writing each token to its own slot makes a replay idempotent. */
typedef struct {
    int          live;
    u64          req_id;
    SampleParams p;
    u32*         hist;    /* [max_ctx] token at each position, UNSET_TOKEN
                           * wherever nothing has been generated yet */
    u32          n_hist;  /* highest position written, + 1    */
} SampleSlot;

/* ⚠ NOT ZERO. The history is indexed by ABSOLUTE position, so everything below
 * a request's first generated position is never written -- and zero is a real
 * vocabulary id, so a zero-filled prefix would penalise token 0 on every
 * request that used a repeat penalty. This value is >= any n_vocab, which is
 * the condition sample.c already skips on. */
#define UNSET_TOKEN 0xFFFFFFFFu

/* What stoke_submit hands to the step thread. See sample_prologue: the slot
 * table itself is touched by ONE thread, and this queue is the only place the
 * two meet. */
typedef struct {
    u64          req_id;
    SampleParams p;
} SamplePend;

/* Deep enough that a caller submitting a burst before its next step does not
 * lose params; overflow degrades to greedy rather than to an error, exactly as
 * running out of slots does. 256 * 40 bytes is not worth a growable buffer. */
#define STOKE_MAX_PENDING 256u

struct stoke_engine {
    GgufFile*  gf;
    Tok*       tok;
    Model*     model;
    KvStore*   store;
    ServeHost* host;
    Sched*     sched;
    u32        max_ctx;
    u32        n_vocab;

    /* ▶▶ TOUCHED ONLY BY THE THREAD THAT CALLS stoke_step. That is what makes
     * the whole sampling path free of locks and free of races: stoke_submit is
     * documented SAFE from any thread, so it may NOT write these. */
    SampleSlot* slots;
    u32         n_slots;
    f32*        sample_scratch; /* [n_vocab], reused: no malloc per token */
    int         any_sampling;   /* the hook is installed only if a request
                                 * has ever asked for it, so a greedy-only
                                 * engine keeps serve_step's original path */

    /* ...and this is the crossing. Written by whichever thread submits, drained
     * by the step thread. One plain mutex, taken once per submit and once per
     * step -- never per token. */
    pthread_mutex_t pend_mu;
    SamplePend      pend[STOKE_MAX_PENDING];
    u32             n_pend;
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
        e->n_vocab = model_n_vocab(model);

        /* One slot per admissible sequence, each with a position-indexed
         * history, plus ONE shared logits scratch -- the sampler is called from
         * the single scheduler thread, so one buffer is enough and a per-token
         * malloc is exactly what M2's op-spec forbids. Allocated at open rather
         * than on first use so the failure lands on the trapped open path. */
        e->n_slots = max_seqs ? max_seqs : STOKE_DEFAULT_SEQS;
        e->slots   = (SampleSlot*)xcalloc(e->n_slots, sizeof(SampleSlot));
        for (u32 i = 0; i < e->n_slots; i++) {
            e->slots[i].hist = (u32*)xcalloc(ctx, sizeof(u32));
        }
        e->sample_scratch = (f32*)xcalloc(e->n_vocab, sizeof(f32));
        pthread_mutex_init(&e->pend_mu, NULL);

#ifdef STOKE_MUT_NO_SLOT_RELEASE
        fprintf(stderr, "stoke: MUTANT src/stoke.c -- STOKE_MUT_NO_SLOT_RELEASE "
                        "(sampling slots are never reclaimed). NEVER use this "
                        "binary for anything but the mutation leg.\n");
#endif
#ifdef STOKE_MUT_STALE_HISTORY
        fprintf(stderr, "stoke: MUTANT src/stoke.c -- STOKE_MUT_STALE_HISTORY "
                        "(a reused slot keeps its last occupant's history). "
                        "NEVER use this binary for anything but the mutation "
                        "leg.\n");
#endif
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
    if (e->slots) {
        for (u32 i = 0; i < e->n_slots; i++) free(e->slots[i].hist);
        free(e->slots);
    }
    free(e->sample_scratch);
    pthread_mutex_destroy(&e->pend_mu);
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

/* The same job opt_u32 does for open opts, for the request's sampling tail: a
 * caller compiled against the older header passes a SMALLER struct, and every
 * field past its `size` must take its default instead of being read out of
 * bounds. That is what makes adding these fields a compatible change rather
 * than an ABI break. */
static f32 req_f32(const stoke_request* r, size_t off, f32 dflt) {
    if (r->size < off + sizeof(float)) return dflt;
    float v;
    memcpy(&v, (const char*)r + off, sizeof v);
    return (f32)v;
}
static u32 req_u32(const stoke_request* r, size_t off, u32 dflt) {
    if (r->size < off + sizeof(uint32_t)) return dflt;
    uint32_t v;
    memcpy(&v, (const char*)r + off, sizeof v);
    return v;
}
static u64 req_u64(const stoke_request* r, size_t off, u64 dflt) {
    if (r->size < off + sizeof(uint64_t)) return dflt;
    uint64_t v;
    memcpy(&v, (const char*)r + off, sizeof v);
    return v;
}

/* ---- the sampler, as serve.h's hook ------------------------------------ */

static SampleSlot* slot_for(stoke_engine* e, u64 req_id) {
    for (u32 i = 0; i < e->n_slots; i++) {
        if (e->slots[i].live && e->slots[i].req_id == req_id) return &e->slots[i];
    }
    return NULL;
}

/* Installed on the host only once some request has actually asked to sample --
 * see stoke_engine.any_sampling. A greedy-only engine therefore runs exactly
 * the serve_step path M5 measured, with not even the null check taken. */
static u32 stoke_sample_cb(void* ctx, u64 req_id, u32 pos, const f32* logits,
                           u32 n_vocab) {
    stoke_engine* e = (stoke_engine*)ctx;
    SampleSlot*   sl = slot_for(e, req_id);
    if (!sl) {
        /* No slot means greedy: either the request never asked to sample, or it
         * finished between the forward pass and here. Falling back rather than
         * failing is right -- a token is owed either way. */
        SampleParams none;
        memset(&none, 0, sizeof none);
        return sample_token(&none, pos, logits, n_vocab, NULL, 0, NULL);
    }
    u32 id = sample_token(&sl->p, pos, logits, n_vocab, sl->hist, sl->n_hist,
                          e->sample_scratch);
    /* Recorded at its own position, so a replay of this position overwrites
     * rather than appends -- see SampleSlot. */
    if (sl->hist && pos < e->max_ctx) {
        sl->hist[pos] = id;
        if (pos + 1u > sl->n_hist) sl->n_hist = pos + 1u;
    }
    return id;
}

static void slot_release(stoke_engine* e, u64 req_id) {
    SampleSlot* sl = slot_for(e, req_id);
    if (sl) { sl->live = 0; sl->n_hist = 0; }
}

/* ---- the step prologue, and the whole concurrency argument --------------
 *
 * ▶▶▶ WHY THE SLOTS ARE NOT CLAIMED IN stoke_submit, WHICH IS WHERE THEY
 * OBVIOUSLY BELONG. stoke.h promises stoke_submit, stoke_poll and stoke_cancel
 * are safe from ANY thread while stoke_step runs on one. The sampler callback
 * runs INSIDE stoke_step. So a slot table written by submit and read by the
 * callback is a data race across that promise -- and so is installing the hook
 * on a host another thread is already stepping. Both would be invisible on x86
 * and both would be real.
 *
 * Guarding the table with a mutex would fix the first and not the second. So
 * instead the table belongs to the step thread outright: submit leaves params
 * in a small queue, and everything that touches a slot happens here.
 *
 * ⚠ THE ORDER OF THE TWO HALVES IS LOAD-BEARING. Sweeping BEFORE draining
 * means a request that finished last step has already returned its slot by the
 * time this step's new requests look for a free one; the other order makes a
 * burst of short requests run greedily for no reason, and it would look like a
 * capacity limit rather than like an ordering mistake.
 *
 * ⚠⚠ AND THE SWEEP ASKS THE SCHEDULER, NOT THE CALLER. Reclaiming on
 * stoke_poll would leak a slot for every request whose caller stopped polling,
 * cancelled, or simply walked away -- and would put a slot write back on an
 * any-thread function. sched_is_live answers for cancelled, retired, abandoned
 * and never-admitted alike. */
static void sample_prologue(stoke_engine* e) {
    if (!e->slots) return;

    for (u32 i = 0; i < e->n_slots; i++) {
#ifdef STOKE_MUT_NO_SLOT_RELEASE
        /* MUTANT: slots are never reclaimed. Every sampled request past the
         * n_slots'th then runs greedy -- fluent text, no error. */
        break;
#endif
        if (e->slots[i].live && !sched_is_live(e->sched, e->slots[i].req_id)) {
            slot_release(e, e->slots[i].req_id);
        }
    }

    /* ⚠ THE EMPTY CHECK IS INSIDE THE LOCK, AND THE OBVIOUS OPTIMISATION IS A
     * DATA RACE. Reading e->n_pend here to skip the acquire is the classic
     * double-check: stoke_submit writes it under this mutex from what stoke.h
     * promises may be ANY thread, so an unsynchronised read of it is UB by the
     * C11 model -- benign on x86, invisible everywhere, and exactly the class
     * TSan exists to find (it cannot be run against this: TSan OOMs on anything
     * model-heavy on this box). The acquire it saves is one uncontended lock
     * against a step that runs a full forward pass, so the trade was never
     * real. Found 2026-08-30 while asking what A6 added to the M5 bars' path;
     * the rate half now measures the code with this in it. */
    SamplePend batch[STOKE_MAX_PENDING];
    u32        n;
    pthread_mutex_lock(&e->pend_mu);
    n = e->n_pend;
    if (n) {
        memcpy(batch, e->pend, (size_t)n * sizeof(SamplePend));
        e->n_pend = 0;
    }
    pthread_mutex_unlock(&e->pend_mu);
    if (!n) return;

    for (u32 k = 0; k < n; k++) {
        /* An id submitted twice cannot be live twice -- the scheduler refuses
         * that -- but a REUSED id whose predecessor has retired can still hold
         * a slot the sweep has not reached. Take it back rather than claim a
         * second one, or the old request's history would be this one's. */
        slot_release(e, batch[k].req_id);

        SampleSlot* sl = NULL;
        for (u32 i = 0; i < e->n_slots; i++) {
            if (!e->slots[i].live) { sl = &e->slots[i]; break; }
        }
        /* No free slot: the request still runs, greedily. Refusing it would
         * turn a resource limit into a client-visible error for something the
         * scheduler was willing to admit. */
        if (!sl) continue;

        sl->live   = 1;
        sl->req_id = batch[k].req_id;
        sl->p      = batch[k].p;
        sl->n_hist = 0;
        /* ⚠ CLEARED, NOT JUST COUNTED DOWN. n_hist alone would leave the
         * previous occupant's tokens visible to a request whose prompt is
         * shorter, and the unwritten prefix below any request's first position
         * has to read as "no token" rather than as id 0. */
#ifndef STOKE_MUT_STALE_HISTORY
        if (sl->hist) {
            memset(sl->hist, 0xFF, (size_t)e->max_ctx * sizeof(u32));
        }
#endif
        /* MUTANT STOKE_MUT_STALE_HISTORY: the clear above is skipped, so a slot
         * hands its last occupant's tokens to the next request at every
         * position the new one does not itself write. Observable ONLY across
         * differing prompt lengths -- see abi_sample.c case [G]. */

        /* ⚠ INSTALLED LAZILY, AND ONLY EVER ONCE -- and from the step thread,
         * which is the thread that reads it. An engine that is never asked to
         * sample runs serve_step's original argmax path, which is the path
         * every M5 number was measured through. */
        if (!e->any_sampling) {
            e->any_sampling = 1;
            serve_host_set_sampler(e->host, stoke_sample_cb, e);
        }
    }
}

stoke_status stoke_submit(stoke_engine* e, const stoke_request* req) {
    if (!e || !req) return fail(STOKE_ERR_INVALID_ARG,
                                "stoke_submit: engine and request must be non-NULL");
    /* ⚠ A SMALLER STRUCT IS ACCEPTED, AND THAT IS THE POINT OF `size`. This
     * used to demand at least sizeof(stoke_request), which meant the moment the
     * sampling tail was added every caller compiled against the older header
     * started failing -- an ABI break dressed as a validation. What must be
     * rejected is a struct too small to hold the fields this function actually
     * requires; everything past that takes its default. */
    if (req->size < offsetof(stoke_request, eos_id) + sizeof(int64_t)) {
        return fail(STOKE_ERR_INVALID_ARG,
                    "stoke_submit: req->size is too small to contain the "
                    "required fields -- set it to sizeof(stoke_request)");
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

    /* The sampling tail, defaulted for a caller who predates it. */
    SampleParams sp;
    memset(&sp, 0, sizeof sp);
    sp.temperature    = req_f32(req, offsetof(stoke_request, temperature), 0.0f);
    sp.top_k          = req_u32(req, offsetof(stoke_request, top_k), 0u);
    sp.top_p          = req_f32(req, offsetof(stoke_request, top_p), 0.0f);
    sp.repeat_penalty = req_f32(req, offsetof(stoke_request, repeat_penalty), 1.0f);
    sp.seed           = req_u64(req, offsetof(stoke_request, seed), 0u);

    int wants_sampling = (sp.temperature > 0.0f) ||
                         (sp.repeat_penalty > 1.0f);

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

    /* ▶ QUEUED ONLY AFTER THE SCHEDULER ACCEPTED IT, so a rejected submit
     * leaves nothing behind to be matched by a later request that happens to
     * reuse the id. The slot itself is claimed by the step thread -- see
     * sample_prologue for why this function must not touch the table. */
    if (wants_sampling) {
        pthread_mutex_lock(&e->pend_mu);
        if (e->n_pend < STOKE_MAX_PENDING) {
            e->pend[e->n_pend].req_id = req->id;
            e->pend[e->n_pend].p      = sp;
            e->n_pend++;
        }
        /* Queue full: the request still runs, greedily -- the same degradation
         * as running out of slots, and for the same reason. */
        pthread_mutex_unlock(&e->pend_mu);
    }
    return STOKE_OK;
}

stoke_status stoke_step(stoke_engine* e, uint32_t* out_rows) {
    if (!e) return fail(STOKE_ERR_INVALID_ARG, "stoke_step: engine is NULL");
    sample_prologue(e);
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
    /* ⚠ AN ID THIS SCHEDULER HAS NEVER SEEN IS AN ERROR, NOT SILENCE. Until the
     * A7 freeze review this returned 0/NONE for it -- byte for byte what a live
     * request that has not produced yet looks like -- so the documented loop
     * ("poll until finish != NONE") never ended for a caller who ignored a
     * failed stoke_submit or mistyped an id. stoke_cancel had always reported
     * it; poll had not.
     *
     * Asked ONLY when nothing was drained and nothing finished, so the hot path
     * pays nothing, and asked of sched_known rather than sched_is_live: a
     * RETIRED request is still pollable and must keep reporting its finish
     * reason, and a request between submit and the next step must keep
     * reporting 0/NONE. */
    if (n == 0 && f == SCHED_FINISH_NONE && !sched_known(e->sched, id)) {
        return fail(STOKE_ERR_UNKNOWN_ID, "stoke_poll: no request with that id");
    }
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
