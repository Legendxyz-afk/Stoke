/* src/server.c -- HTTP/1.1 + OpenAI-compatible endpoints. M6 rung A3.
 *
 * ▶▶▶ THE ENGINE IS REACHED ONLY THROUGH stoke.h. Not model.h, not sched.h, not
 * tok.h. That constraint is the reason the rung order puts the server before
 * the ABI freeze at A7: the only thing in this milestone that will exercise the
 * ABI hard enough to find out whether it is right is this file, and a server
 * that reached around the header would leave the ABI shipping having been used
 * by nothing. It has already paid for itself once -- `stoke_apply_chat_template`
 * exists because this file needed it and the header could not express a chat
 * request without it.
 *
 * tests/abi_isolation.sh enforces it mechanically, so the rule cannot decay
 * into a comment. http.h / json.h / net.h are NOT the engine -- they are this
 * server's own support code, built at A2 and A3 for exactly this caller.
 *
 * ▶▶ ONE THREAD, ONE LOOP, NOTHING BLOCKING. The loop owns both the sockets and
 * the model, so blocking on either starves the other: a recv() that waits stops
 * token generation for every other client, and so does a send() to one slow
 * reader. Every socket is non-blocking and net_wait() decides who is ready,
 * with a timeout the loop picks -- zero while the engine has work, longer while
 * it is idle so nothing burns a core waiting.
 *
 * ▶ THE MEMORY SHAPE, because it is the part with a sharp edge:
 *   - `prompt_ids` is BORROWED by the engine from submit until finish (stoke.h
 *     says so in capitals). Each connection therefore owns its own ids
 *     allocation for the whole life of its request. Nothing is shared.
 *   - Templating needs a buffer sized to stoke_chat_template_bound(), which is
 *     ~3x the message bytes. The HTTP layer caps a body at 1 MiB, so that is
 *     ~12 MB at worst -- and only ever ONE at a time, because templating is
 *     synchronous inside this single loop and the buffer is shrunk to the real
 *     id count immediately afterwards.
 */
/* ⚠ common.h FIRST, as everywhere in this project: on the Linux leg it defines
 * the feature-test macro that has to precede the first libc header, and it is
 * where va_list reaches buf_fmt below. Including it is NOT a breach of the
 * engine-isolation rule -- it carries fixed-width types and timing, not the
 * engine. tests/abi_isolation.sh states the rule exactly, and also checks that
 * this file never CALLS fatal(), which is the part that would matter. */
#include "common.h"

#include "stoke.h"

#include "http.h"
#include "json.h"
#include "net.h"

#include "server.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Well inside net.h's NET_WAIT_MAX, which is itself inside the Windows
 * FD_SETSIZE of 64. A local inference server does not need more, and a number
 * that cannot overflow select() is worth more than a number that looks big. */
#define MAX_CONNS 24

#define IN_CAP_INITIAL 4096
#define IN_CAP_MAX     (HTTP_MAX_HEAD_BYTES + HTTP_MAX_BODY_BYTES)

/* A connection that has neither sent nor received anything for this long is
 * closed. Without it, opening MAX_CONNS sockets and saying nothing takes the
 * server off the air permanently -- and costs the client nothing. */
#define IDLE_TIMEOUT_SECS 60

#define DEFAULT_MAX_TOKENS 128

/* OpenAI accepts up to four stop strings and so does this. A cap is a
 * refusal rather than an assertion: `stop` is client-supplied, and an
 * unbounded list is an unbounded per-token scan a client can ask for. */
#define STOP_MAX 4

/* An idle stream that says nothing gets closed by intermediaries -- 60 s is a
 * common proxy default, so the comment goes out well inside it. It is a
 * COMMENT (`: ping`) rather than an event precisely so no client can mistake
 * it for content: the SSE spec says a line starting with a colon is ignored. */
#define SSE_PING_MS_DEFAULT 15000

/* How much unread stream one connection may make this process hold before it
 * is disconnected. Sized so a normal reader never approaches it and a reader
 * that has stopped reading cannot buy memory indefinitely. */
#define SSE_MAX_PENDING_DEFAULT (256u * 1024u)

/* ---- a growable byte buffer, for building responses -------------------- */

typedef struct {
    char*  b;
    size_t len, cap;
    int    oom;
} Buf;

typedef enum {
    CONN_FREE = 0,
    CONN_READ,   /* accumulating a request                      */
    CONN_GEN,    /* submitted to the engine, generating         */
    CONN_WRITE,  /* whole response built, draining to the socket */
    CONN_STREAM, /* generating AND draining, one token at a time */
} ConnState;

typedef struct {
    ConnState state;
    NetSock   s;

    char*  in;
    size_t in_len, in_cap;

    /* ▶ ONE BUFFER, APPENDED TO, NOT REPLACED. A non-streamed response is
     * built once and drained once, which a fresh allocation per response
     * served fine. A stream appends after every token while the socket is
     * still draining the last one, so the buffer has to outlive any single
     * write and carry a read cursor into itself. */
    Buf    out;
    size_t out_sent;

    int keep_alive;

    /* the live request */
    uint64_t  req_id;
    uint32_t* prompt_ids; /* BORROWED by the engine while CONN_GEN/STREAM */
    uint32_t  n_prompt;
    uint32_t* gen;
    int32_t   n_gen, gen_cap;
    char      model[128]; /* echoed back; copied because `in` is reused */
    int64_t   created;    /* one timestamp for every chunk of one answer */

    /* ▶ /v1/completions and /v1/chat/completions are the SAME request with two
     * wire shapes: `text` against `message`/`delta`, `cmpl-` against
     * `chatcmpl-`, `text_completion` against `chat.completion`. A client that
     * reads delta.content off a completion chunk gets nothing, so the two are
     * asserted apart in the gate rather than assumed identical. */
    int is_completion;

    /* ---- stop strings (A6), consumer-side by design ----
     *
     * ▶▶ THE TAIL WINDOW IS A HOLD-BACK, NOT MERELY A SEARCH WINDOW, and that
     * is the whole difficulty. A stop string crosses token boundaries: "an I"
     * is the tail of one token and the whole of the next. Searching a window
     * finds it, but by then "an" has already gone out as a delta -- part of the
     * stop string leaked to the client. So emission is held back by
     * (longest stop string - 1) bytes while generating: those bytes cannot yet
     * be known not to begin a match. The bound is exact -- a match ending at
     * the newest byte starts at or after the previous emission limit -- and it
     * is why the window is that length and not a round number.
     *
     * `text` accumulates everything decoded, truncated at a match; the
     * non-streamed response is that buffer, and the streamed one is deltas cut
     * out of it. One accumulator for both, so a stop cannot behave differently
     * depending on how the client asked to be answered. */
    Buf    text;
    size_t text_emitted;
    char*  stops[STOP_MAX];
    size_t stop_len[STOP_MAX];
    int    n_stops;
    size_t hold;
    int    stopped;

    /* ---- streaming ---- */
    int  want_usage;  /* stream_options.include_usage                      */
    int  sent_role;   /* the role goes in the FIRST delta and nowhere else */
    int  stream_done; /* the [DONE] sentinel and the final chunk are in    */
    /* Bytes of a multi-byte character whose remaining bytes are still in a
     * token that has not been generated yet. See json_utf8_complete_prefix:
     * this vocabulary has all 256 single-byte ids, so one character really
     * can arrive as several tokens, and half a codepoint inside a JSON
     * string is a response the client cannot parse. Four bytes is the whole
     * of UTF-8; eight is room to be wrong in safely. */
    char   pend[8];
    size_t pend_len;
    double last_out_ms; /* for the heartbeat: when we last put bytes out */

    time_t last_active;
} Conn;

typedef struct {
    stoke_engine* e;
    Conn          conns[MAX_CONNS];
    uint64_t      next_req_id;
    int           completed;
    int           verbose;
    uint32_t      max_ctx;
    uint32_t      eos_id;
    int           ping_ms;
    size_t        max_pending;
    int           send_buffer;
    int           idle_secs;
    /* What /v1/models reports. Derived from the GGUF's filename, because that
     * is the only name this server actually has -- it serves one file and
     * never routes on the `model` field a client sends. */
    char          model_id[128];
    /* One reusable frame buffer for the whole server. Streaming builds a
     * complete SSE frame before it can chunk it (a chunk is prefixed by its
     * length), and a fresh allocation per token is a malloc on the hot path
     * for no reason -- the loop is single-threaded, so one is enough. */
    Buf           scratch;
} Server;

/* ---- the buffer helpers ------------------------------------------------ */

static void buf_free(Buf* d) {
    free(d->b);
    d->b   = NULL;
    d->len = d->cap = 0;
    d->oom = 0;
}

static int buf_reserve(Buf* d, size_t extra) {
    if (d->oom) return 0;
    if (d->len + extra <= d->cap) return 1;
    size_t want = d->cap ? d->cap * 2 : 1024;
    while (want < d->len + extra) want *= 2;
    char* nb = (char*)realloc(d->b, want);
    if (!nb) {
        d->oom = 1;
        return 0;
    }
    d->b   = nb;
    d->cap = want;
    return 1;
}

static void buf_add(Buf* d, const char* s, size_t n) {
    if (!buf_reserve(d, n)) return;
    memcpy(d->b + d->len, s, n);
    d->len += n;
}

static void buf_str(Buf* d, const char* s) { buf_add(d, s, strlen(s)); }

static void buf_fmt(Buf* d, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char    tmp[512];
    int     n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof tmp) {
        buf_add(d, tmp, (size_t)n);
        return;
    }
    /* Longer than the stack window: reserve exactly and format again rather
     * than truncating, because a truncated JSON body is not a smaller answer,
     * it is an unparseable one. */
    if (!buf_reserve(d, (size_t)n + 1)) return;
    va_start(ap, fmt);
    vsnprintf(d->b + d->len, (size_t)n + 1, fmt, ap);
    va_end(ap);
    d->len += (size_t)n;
}

/* Appends `len` bytes of `s` as the INSIDE of a JSON string. */
static void buf_json_str(Buf* d, const char* s, size_t len) {
    i64 need = json_escape(s, len, NULL, 0);
    if (need < 0 || !buf_reserve(d, (size_t)need)) return;
    json_escape(s, len, d->b + d->len, (size_t)need);
    d->len += (size_t)need;
}

/* ---- connections -------------------------------------------------------- */

static void conn_release_request(Server* sv, Conn* c) {
    /* Cancel before freeing: the engine BORROWS prompt_ids, so freeing them
     * while a request is still live is a use-after-free inside the scheduler,
     * on a path only a client disconnect reaches. */
    if (c->req_id) {
        stoke_cancel(sv->e, c->req_id);
        c->req_id = 0;
    }
    free(c->prompt_ids);
    c->prompt_ids = NULL;
    free(c->gen);
    c->gen    = NULL;
    c->n_gen  = 0;
    c->gen_cap = 0;
    c->n_prompt = 0;

    for (int i = 0; i < c->n_stops; i++) {
        free(c->stops[i]);
        c->stops[i] = NULL;
    }
    c->n_stops = 0;
    c->hold    = 0;
    c->stopped = 0;
    buf_free(&c->text);
    c->text_emitted = 0;
}

static void conn_close(Server* sv, Conn* c) {
    conn_release_request(sv, c);
    net_close(c->s);
    c->s = NET_INVALID_SOCK;
    free(c->in);
    buf_free(&c->out);
    memset(c, 0, sizeof *c);
    c->s     = NET_INVALID_SOCK;
    c->state = CONN_FREE;
}

/* Retires `n` bytes from the front of the input buffer, keeping whatever a
 * pipelining client already sent for the next request. */
static void conn_consume(Conn* c, size_t n) {
    if (n >= c->in_len) {
        c->in_len = 0;
        return;
    }
    memmove(c->in, c->in + n, c->in_len - n);
    c->in_len -= n;
}

/* ---- responses ---------------------------------------------------------- */

/* Drops whatever is already queued and starts the buffer over. Only a
 * complete-response path may do this: it is how a 500 replaces a half-built
 * body, and it is exactly wrong once a stream's head has gone out. */
static void conn_out_reset(Conn* c) {
    c->out.len  = 0;
    c->out.oom  = 0;
    c->out_sent = 0;
}

static void respond(Conn* c, int status, const char* content_type,
                    const char* body, size_t body_len) {
    conn_out_reset(c);
    Buf* h = &c->out;
    buf_fmt(h, "HTTP/1.1 %d %s\r\n", status, http_status_text(status));
    buf_fmt(h, "Content-Type: %s\r\n", content_type);
    buf_fmt(h, "Content-Length: %zu\r\n", body_len);
    buf_str(h, c->keep_alive ? "Connection: keep-alive\r\n"
                             : "Connection: close\r\n");
    buf_str(h, "\r\n");
    buf_add(h, body, body_len);

    if (h->oom) {
        /* Nothing can be sent, so send nothing and close. The request still
         * counts as answered when service_write drains this empty buffer --
         * see the note there; a client that gets a closed socket has had its
         * answer, badly. */
        conn_out_reset(c);
        c->keep_alive = 0;
    }
    c->state = CONN_WRITE;
}

/* The OpenAI error envelope. openai-python reads `error.message` and
 * `error.type` off a 4xx, so an error that is merely a status code with plain
 * text reaches the caller as an unhelpful exception. */
static void respond_error(Conn* c, int status, const char* type,
                          const char* message) {
    Buf b = {0};
    buf_str(&b, "{\"error\":{\"message\":\"");
    buf_json_str(&b, message, strlen(message));
    buf_str(&b, "\",\"type\":\"");
    buf_json_str(&b, type, strlen(type));
    buf_str(&b, "\",\"param\":null,\"code\":null}}\n");

    /* An error ends the connection. A client that sent something we could not
     * parse has, by definition, left us unsure where its next request begins. */
    c->keep_alive = 0;
    if (b.oom) {
        respond(c, status, "application/json", "{}", 2);
    } else {
        respond(c, status, "application/json", b.b, b.len);
    }
    buf_free(&b);
}

/* ---- server-sent events -------------------------------------------------
 *
 * ▶▶ CHUNKED, NOT Connection: close. A streamed body has no length when its
 * head goes out, and HTTP/1.1 offers exactly two ways to say so: chunked
 * framing, or close-delimited. Close-delimited is three lines shorter and
 * costs the connection -- every request would then pay a fresh TCP handshake,
 * including every request of the M0 benchmark harness, whose whole job is
 * measuring inter-token latency. Framing that distorts the instrument pointed
 * at it is not a saving. It is also what the real OpenAI API sends, and the
 * clients this milestone is graded against are its clients.
 *
 * ⚠ Inbound chunked is still refused with a 501 by src/http.c, and that is not
 * an inconsistency: a client sending it is a request we would have to buffer
 * and re-frame, while a server sending it is a body we already produce a piece
 * at a time.
 */

/* Rewinds the send cursor, moving anything unsent to the front. The common
 * case -- the socket kept up -- is a rewind with no copy at all. */
static void conn_out_compact(Conn* c) {
    if (c->out_sent == 0) return;
    if (c->out_sent >= c->out.len) {
        c->out.len  = 0;
        c->out_sent = 0;
        return;
    }
    memmove(c->out.b, c->out.b + c->out_sent, c->out.len - c->out_sent);
    c->out.len -= c->out_sent;
    c->out_sent = 0;
}

/* Queues one chunked-transfer chunk carrying `n` bytes. */
static void sse_chunk(Conn* c, const char* p, size_t n) {
    if (n == 0) return; /* a zero-length chunk ENDS the body; never by accident */
    conn_out_compact(c);
    buf_fmt(&c->out, "%zx\r\n", n);
    buf_add(&c->out, p, n);
    buf_str(&c->out, "\r\n");
    c->last_out_ms = now_ms();
}

static void sse_begin(Conn* c) {
    conn_out_reset(c);
    Buf* h = &c->out;
    buf_str(h, "HTTP/1.1 200 OK\r\n");
    buf_str(h, "Content-Type: text/event-stream\r\n");
    /* Both matter in front of a proxy: one stops it caching a stream, the
     * other stops it holding events back until it has "enough" of them. */
    buf_str(h, "Cache-Control: no-cache\r\n");
    buf_str(h, "X-Accel-Buffering: no\r\n");
    buf_str(h, "Transfer-Encoding: chunked\r\n");
    buf_str(h, c->keep_alive ? "Connection: keep-alive\r\n"
                             : "Connection: close\r\n");
    buf_str(h, "\r\n");
    c->last_out_ms = now_ms();
}

/* The terminating zero-length chunk. After this the body is over and the
 * connection is reusable. */
static void sse_finish_body(Conn* c) {
    conn_out_compact(c);
    buf_str(&c->out, "0\r\n\r\n");
    c->last_out_ms = now_ms();
}

/* ---- the chat endpoint -------------------------------------------------- */

/* ⚠ DROPPING THE END-OF-SEQUENCE ID IS THIS LAYER'S JOB AND tok.h SAYS SO.
 * stoke_poll reports every id the model generated, end-of-sequence included,
 * and detokenizing is the mechanical inverse of the vocab table -- so the stop
 * token comes back as the literal text "<|im_end|>". Whether a control token is
 * shown to an API client is a policy choice, and the policy for an
 * OpenAI-compatible endpoint is that it is not: `message.content` is the
 * assistant's words. consume_tokens() below is where that drop happens, for
 * both the streamed and the non-streamed path, because both now decode
 * incrementally -- see its note. */

static const char* finish_reason_of(int32_t finish) {
    switch (finish) {
        case STOKE_FINISH_MAX_TOKENS: return "length";
        case STOKE_FINISH_EOS: return "stop";
        /* Cancellation has no OpenAI spelling; "stop" is the honest one, since
         * from the client's side the answer simply ended. */
        default: return "stop";
    }
}

/* ---- the streamed shape of the same answer ------------------------------
 *
 * Every function here returns 1 to continue and 0 when the connection has been
 * closed under it -- the only failure any of them can hit is a buffer that
 * could not grow, and a stream missing a token is not a stream.
 */

/* The envelope every chunk of one response shares.
 *
 * ⚠ `created` is fixed when the request is accepted, not read per chunk. All
 * of OpenAI's chunks carry one timestamp, and a clock read per token would put
 * a field in the stream that changes for reasons the client cannot see. */
static void stream_envelope(Server* sv, Conn* c) {
    Buf* f = &sv->scratch;
    f->len = 0;
    f->oom = 0;
    buf_str(f, "data: ");
    buf_fmt(f, "{\"id\":\"%s-%llu\",\"object\":\"%s\",\"created\":%lld,\"model\":\"",
            c->is_completion ? "cmpl" : "chatcmpl",
            (unsigned long long)c->req_id,
            c->is_completion ? "text_completion" : "chat.completion.chunk",
            (long long)c->created);
    buf_json_str(f, c->model, strlen(c->model));
    buf_str(f, "\",");
}

static int stream_send_frame(Server* sv, Conn* c) {
    buf_str(&sv->scratch, "\n\n");
    if (sv->scratch.oom) {
        conn_close(sv, c);
        return 0;
    }
    sse_chunk(c, sv->scratch.b, sv->scratch.len);
    return 1;
}

/* One content delta.
 *
 * ▶ THE ROLE RIDES ON THE FIRST CONTENT CHUNK rather than in a chunk of its
 * own. openai-python accepts either -- it merges deltas -- but a role-only
 * chunk is one more SSE event, and the M0 harness counts events to derive
 * inter-token latency. A chunk carrying no token would land in that
 * measurement as a token, which is a worse error than a shorter stream. */
static int stream_delta(Server* sv, Conn* c, const char* text, size_t len) {
    stream_envelope(sv, c);
    if (c->is_completion) {
        /* No role and no delta object: a completion chunk's payload is a bare
         * `text` field, and there is no conversation for a role to belong to. */
        buf_str(&sv->scratch, "\"choices\":[{\"index\":0,\"text\":\"");
        buf_json_str(&sv->scratch, text, len);
        buf_str(&sv->scratch, "\",\"logprobs\":null,\"finish_reason\":null}]}");
        return stream_send_frame(sv, c);
    }
    buf_str(&sv->scratch, "\"choices\":[{\"index\":0,\"delta\":{");
    if (!c->sent_role) {
        buf_str(&sv->scratch, "\"role\":\"assistant\",");
        c->sent_role = 1;
    }
#ifdef STOKE_MUT_ROLE_EVERY_CHUNK
    /* ⚠ DELIBERATELY WRONG. A mutant for tests/serve_openai.py: the role must
     * ride on the FIRST delta and no other, and a suite that cannot see this
     * difference is not testing the trap it claims to. Built only by
     *   mingw32-make BUILDDIR=build/win-mut-openai-role \
     *                EXTRA_CFLAGS=-DSTOKE_MUT_ROLE_EVERY_CHUNK=1
     * and it announces itself on startup, because a build that silently
     * dropped its -D is a CLEAN binary and the leg would then report "the
     * suite did not catch it" about a mutant that was never there. */
    c->sent_role = 0;
#endif
    buf_str(&sv->scratch, "\"content\":\"");
    buf_json_str(&sv->scratch, text, len);
    buf_str(&sv->scratch, "\"},\"logprobs\":null,\"finish_reason\":null}]}");
    return stream_send_frame(sv, c);
}

/* Decodes the newly generated ids and emits them, holding back a trailing
 * partial character.
 *
 * ⚠⚠ THE HOLD-BACK IS NOT OPTIONAL. This vocabulary contains all 256
 * single-byte ids, so one character can be generated as several tokens, and a
 * server that emitted each token as it arrived would put half a codepoint
 * inside a JSON string -- an unparseable event, produced only by inputs that
 * make the model answer in a non-ASCII script. See
 * json_utf8_complete_prefix(). A partial character still pending when
 * generation ENDS is dropped; it was never a character. */
/* How much of `text` may be shown to the client right now.
 *
 * While generating, the last `hold` bytes are held back: they cannot yet be
 * known not to BEGIN a stop string, and a delta carrying half of one has
 * already leaked it. `hold` is (longest stop string - 1) and that bound is
 * exact -- a match ending at the newest byte starts at or after the previous
 * limit -- so nothing is held longer than it must be. With no stop strings
 * `hold` is zero and this is the whole buffer. */
static size_t emit_limit(const Conn* c, int final) {
    if (final || c->stopped) return c->text.len;
    return c->text.len > c->hold ? c->text.len - c->hold : 0;
}

/* Returns the offset of the first stop string at or after `from`, or (size_t)-1. */
static size_t find_stop(const Conn* c, size_t from) {
    size_t best = (size_t)-1;
    for (int i = 0; i < c->n_stops; i++) {
        size_t sl = c->stop_len[i];
        if (sl == 0 || c->text.len < sl) continue;
        for (size_t p = from; p + sl <= c->text.len; p++) {
            if (memcmp(c->text.b + p, c->stops[i], sl) == 0) {
                if (p < best) best = p;
                break;
            }
        }
    }
    return best;
}

/* Decodes newly generated ids into c->text, honouring the UTF-8 carry and the
 * stop strings, and (when streaming) emits whatever is now safe to show.
 *
 * ⚠⚠ THE HOLD-BACK IS NOT OPTIONAL AND NEITHER IS THE CARRY. This vocabulary
 * has all 256 single-byte ids, so one character can be generated as several
 * tokens -- half a codepoint inside a JSON string is unparseable. And a stop
 * string crosses token boundaries, so a server that checked each token in
 * isolation would miss "an I" split across " can" and " I" entirely. Both are
 * the same shape of bug: a decision made per token about a property that
 * belongs to the stream.
 *
 * Returns 1 normally, 0 if the connection was closed underneath. */
static int consume_tokens(Server* sv, Conn* c, const uint32_t* ids, int32_t n,
                          int streaming) {
    for (int32_t i = 0; i < n && !c->stopped; i++) {
        if (ids[i] == sv->eos_id) continue;

        char    piece[512];
        int32_t m = stoke_detokenize(sv->e, ids[i], piece, (int32_t)sizeof piece);
        if (m <= 0 || (size_t)m > sizeof piece) continue;

        char   whole[sizeof c->pend + sizeof piece];
        size_t len = c->pend_len;
        memcpy(whole, c->pend, c->pend_len);
        memcpy(whole + len, piece, (size_t)m);
        len += (size_t)m;

        size_t cut  = json_utf8_complete_prefix(whole, len);
        size_t tail = len - cut;
        if (tail > sizeof c->pend) { /* unreachable: a partial tail is < 4 */
            cut  = len;
            tail = 0;
        }
        memcpy(c->pend, whole + cut, tail);
        c->pend_len = tail;
        if (cut == 0) continue; /* the character has not finished arriving */

        /* The search starts `hold` bytes before what just arrived, because a
         * match may begin in text that is already here and end in text that is
         * not -- which is exactly the case a per-token check cannot see. */
        size_t before = c->text.len;
        buf_add(&c->text, whole, cut);
        if (c->text.oom) {
            conn_close(sv, c);
            return 0;
        }
        if (c->n_stops) {
            size_t from = before > c->hold ? before - c->hold : 0;
            size_t at   = find_stop(c, from);
            if (at != (size_t)-1) {
                c->text.len = at; /* not one byte of the match is kept */
                c->stopped  = 1;
                stoke_cancel(sv->e, c->req_id);
            }
        }

        /* ▶ EMITTED INSIDE THE LOOP, ONE DELTA PER TOKEN, and that placement is
         * load-bearing rather than stylistic. A poll can return several tokens
         * at once, and emitting once per BATCH would coalesce them into a
         * single SSE event -- which the M0 harness at gate [3] counts as ONE
         * token while `usage` reports several, turning an exact
         * count_vs_usage_diff of 0 into a silent mismatch and changing what its
         * inter-token latency measures. With no stop strings the hold is zero
         * and this emits exactly what the pre-A6 code did, token for token. */
        if (streaming) {
            size_t limit = emit_limit(c, 0);
            if (limit > c->text_emitted) {
                size_t off = c->text_emitted;
                size_t nb  = limit - off;
                c->text_emitted = limit;
                if (!stream_delta(sv, c, c->text.b + off, nb)) return 0;
            }
        }
    }
    return 1;
}

static int stream_final(Server* sv, Conn* c, int32_t finish) {
    /* ▶ THE HELD-BACK TAIL GOES OUT FIRST. While generating, the last
     * (longest stop string - 1) bytes are withheld in case they begin a match;
     * once generation is over nothing more can arrive to complete one, so they
     * are ordinary text and the client is owed them. Forgetting this would
     * silently truncate the last few characters of every answer that used a
     * stop string -- a defect that looks like the model being terse. */
    size_t limit = emit_limit(c, 1);
    if (limit > c->text_emitted) {
        size_t off = c->text_emitted;
        size_t nb  = limit - off;
        c->text_emitted = limit;
        if (!stream_delta(sv, c, c->text.b + off, nb)) return 0;
    }

    /* The finish chunk: an EMPTY delta carrying the reason. A client that
     * merges deltas gets nothing new, which is the point -- the reason is not
     * content. */
    stream_envelope(sv, c);
    buf_fmt(&sv->scratch,
            "\"choices\":[{\"index\":0,%s,\"logprobs\":null,"
            "\"finish_reason\":\"%s\"}]}",
            c->is_completion ? "\"text\":\"\"" : "\"delta\":{}",
            finish_reason_of(finish));
    if (!stream_send_frame(sv, c)) return 0;

    /* ▶ USAGE IS ITS OWN CHUNK, AFTER THE FINISH CHUNK, WITH EMPTY choices,
     * and only under stream_options.include_usage. Three separate traps in one
     * field: a client reading usage off the finish chunk finds nothing; one
     * that stops reading at finish_reason never sees it; and one that did not
     * ask for it and gets a chunk with no choices can trip over the empty
     * array. Non-streamed responses carry usage unconditionally -- that
     * asymmetry is OpenAI's, not ours. */
    if (c->want_usage) {
        stream_envelope(sv, c);
        buf_fmt(&sv->scratch,
                "\"choices\":[],\"usage\":{\"prompt_tokens\":%u,"
                "\"completion_tokens\":%d,\"total_tokens\":%u}}",
                c->n_prompt, (int)c->n_gen, c->n_prompt + (uint32_t)c->n_gen);
        if (!stream_send_frame(sv, c)) return 0;
    }

    sse_chunk(c, "data: [DONE]\n\n", sizeof "data: [DONE]\n\n" - 1);
    sse_finish_body(c);
    c->stream_done = 1;

    if (sv->verbose) {
        fprintf(stderr,
                "stoke serve: stream id=%llu prompt=%u gen=%d finish=%s\n",
                (unsigned long long)c->req_id, c->n_prompt, (int)c->n_gen,
                finish_reason_of(finish));
    }
    c->req_id = 0; /* the engine is done with it; nothing to cancel */
    conn_release_request(sv, c);
    return 1;
}

static void finish_chat(Server* sv, Conn* c, int32_t finish) {
    /* c->text was accumulated token by token by consume_tokens, for the same
     * reason the streamed path needs it: a stop string has to be able to
     * CANCEL generation when it is hit, which cannot be done by decoding
     * everything at the end. The two paths therefore share one accumulator
     * and a stop cannot behave differently depending on how the client
     * asked to be answered. */
    Buf text = c->text;

    Buf b = {0};
    buf_fmt(&b, "{\"id\":\"%s-%llu\",\"object\":\"%s\",\"created\":%lld,"
                "\"model\":\"",
            c->is_completion ? "cmpl" : "chatcmpl",
            (unsigned long long)c->req_id,
            c->is_completion ? "text_completion" : "chat.completion",
            (long long)c->created);
    buf_json_str(&b, c->model, strlen(c->model));
    if (c->is_completion) {
        buf_str(&b, "\",\"choices\":[{\"index\":0,\"text\":\"");
    } else {
        buf_str(&b, "\",\"choices\":[{\"index\":0,\"message\":{\"role\":"
                    "\"assistant\",\"content\":\"");
    }
    buf_json_str(&b, text.b ? text.b : "", text.len);
    buf_fmt(&b, "%s,\"logprobs\":null,\"finish_reason\":\"%s\"}],",
            c->is_completion ? "\"" : "\"}", finish_reason_of(finish));
    /* usage is ALWAYS present on a non-streamed response -- that is the
     * OpenAI contract, and clients read it unconditionally. */
    buf_fmt(&b, "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%d,"
                "\"total_tokens\":%u}}\n",
            c->n_prompt, (int)c->n_gen, c->n_prompt + (uint32_t)c->n_gen);

    if (sv->verbose) {
        fprintf(stderr, "stoke serve: chat id=%llu prompt=%u gen=%d finish=%s\n",
                (unsigned long long)c->req_id, c->n_prompt, (int)c->n_gen,
                finish_reason_of(finish));
    }

    if (b.oom) {
        respond_error(c, 500, "server_error", "response did not fit in memory");
    } else {
        respond(c, 200, "application/json", b.b, b.len);
    }
    buf_free(&b);

    c->req_id = 0; /* the engine is done with it; nothing to cancel */
    conn_release_request(sv, c);
}

/* Reads one message out of the JSON. Returns 0 and fills `m`, or non-zero with
 * `why` set. The pointers land INSIDE the request buffer, which is exactly
 * where they must stay until templating copies the ids out. */
static int read_message(const JsonDoc* d, const char* body, const JsonNode* n,
                        stoke_chat_message* m, const char** why) {
    if (!n || n->type != JSON_OBJ) {
        *why = "each element of 'messages' must be an object";
        return -1;
    }
    const JsonNode* role    = json_member(d, body, n, "role");
    const JsonNode* content = json_member(d, body, n, "content");
    if (!role || role->type != JSON_STR) {
        *why = "each message needs a string 'role'";
        return -1;
    }
    if (!content || content->type != JSON_STR) {
        /* OpenAI allows a content ARRAY of parts. Refusing it plainly beats
         * silently sending the model an empty prompt. */
        *why = "each message needs a string 'content'";
        return -1;
    }
    m->role        = body + role->start;
    m->role_len    = role->end - role->start;
    m->content     = body + content->start;
    m->content_len = content->end - content->start;

    /* ⚠ The template is applied to the RAW slice, escapes and all. Anything
     * else would mean decoding into a second buffer that must then outlive the
     * request; the escape sequences that survive are ordinary text to the
     * tokenizer, and no turn marker can be forged through them because
     * tok_apply_chatml encodes each message separately by construction. */
    return 0;
}

static void start_generation(Server* sv, Conn* c, uint32_t* ids, int32_t n_prompt,
                             uint32_t max_tokens, int want_stream, int want_usage);

/* Parses a request body and the fields BOTH endpoints share. Returns 0 with
 * `*root` set, or -1 having already answered with the right error. */
static int parse_body(Server* sv, Conn* c, const HttpRequest* r, JsonDoc* d,
                      const char** out_body, const JsonNode** out_root,
                      uint32_t* max_tokens, int* want_stream, int* want_usage) {
    const char* body     = c->in + r->body_off;
    size_t      body_len = r->body_len;
    if (body_len == 0) {
        respond_error(c, 400, "invalid_request_error", "request body is empty");
        return -1;
    }

    JsonError je;
    if (json_parse(body, body_len, d, &je) != 0) {
        char msg[160];
        snprintf(msg, sizeof msg, "body is not valid JSON (%s at byte %u)",
                 je.reason ? je.reason : "?", je.offset);
        respond_error(c, 400, "invalid_request_error", msg);
        return -1;
    }
    const JsonNode* root = json_root(d);
    if (!root || root->type != JSON_OBJ) {
        respond_error(c, 400, "invalid_request_error", "body must be an object");
        return -1;
    }

    const JsonNode* stream = json_member(d, body, root, "stream");
    *want_stream           = 0;
    if (stream) json_bool(body, stream, want_stream);

    /* ⚠ stream_options IS READ EVEN WHEN stream IS FALSE, and then ignored.
     * The alternative -- refusing the combination -- is a 400 for a request
     * that is merely redundant, and openai-python sends whatever the caller
     * set. A non-streamed response carries usage unconditionally anyway. */
    *want_usage           = 0;
    const JsonNode* sopts = json_member(d, body, root, "stream_options");
    if (sopts && sopts->type == JSON_OBJ) {
        const JsonNode* iu = json_member(d, body, sopts, "include_usage");
        if (iu) json_bool(body, iu, want_usage);
    }

    const JsonNode* model = json_member(d, body, root, "model");
    c->model[0]           = '\0';
    if (model && model->type == JSON_STR) {
        i64 mn = json_string(body, model, c->model, sizeof c->model - 1);
        if (mn >= 0 && (size_t)mn < sizeof c->model) c->model[mn] = '\0';
    }
    if (c->model[0] == '\0') snprintf(c->model, sizeof c->model, "stoke");

    /* ▶▶▶ THE SAMPLING SURFACE, ACCEPTED AT ITS NO-OP VALUE AND REFUSED
     * OTHERWISE. This build decodes greedily and has no sampler, and every
     * parameter below has a value at which "no sampler" is the CORRECT answer:
     * temperature 0 IS greedy, top_p 1 truncates nothing, top_k 0 is disabled,
     * the penalties at their identity, n=1 is one choice. At those values there
     * is nothing to implement and nothing to refuse -- which is why the M0
     * harness's `"temperature": 0.0` on every request needs no special case and
     * gets none.
     *
     * ⚠ AT ANY OTHER VALUE THE REQUEST IS REFUSED RATHER THAN SILENTLY
     * ANSWERED GREEDILY. A client asking for top_p 0.9 and quietly receiving
     * the greedy distribution gets plausible text that is not what it asked
     * for -- the kind of wrong nobody notices for weeks, and the kind that
     * makes a benchmark comparing two engines meaningless. A 400 is noticed
     * immediately. The whole table goes when the sampler lands.
     *
     * `seed` is deliberately NOT here: greedy decoding is already
     * deterministic, so any seed produces the identical stream, which is
     * exactly what a seed promises. Accepting it is honest. */
    static const struct {
        const char* key;
        double      noop;
    } k_sampling[] = {
        {"temperature", 0.0},       {"top_p", 1.0},
        {"top_k", 0.0},             {"frequency_penalty", 0.0},
        {"presence_penalty", 0.0},  {"repeat_penalty", 1.0},
        {"n", 1.0},
    };
    for (size_t i = 0; i < sizeof k_sampling / sizeof k_sampling[0]; i++) {
        const JsonNode* pn = json_member(d, body, root, k_sampling[i].key);
        if (!pn || pn->type == JSON_NULL) continue;
        double v;
        if (json_number(body, pn, &v) != 0) {
            char msg[128];
            snprintf(msg, sizeof msg, "'%s' must be a number", k_sampling[i].key);
            respond_error(c, 400, "invalid_request_error", msg);
            return -1;
        }
        if (v != k_sampling[i].noop) {
            char msg[224];
            snprintf(msg, sizeof msg,
                     "this build decodes greedily and has no sampler, so it "
                     "implements '%s' only at %g -- %g is refused rather than "
                     "silently answered greedily",
                     k_sampling[i].key, k_sampling[i].noop, v);
            respond_error(c, 400, "invalid_request_error", msg);
            return -1;
        }
    }

    /* `stop` is a string or an array of up to four. Both spellings are ordinary
     * in the wild -- openai-python passes through whatever the caller wrote --
     * so refusing either would be refusing valid requests. An empty string is
     * dropped rather than stored: it matches at every position, which would end
     * every generation before its first token. */
    const JsonNode* stopn = json_member(d, body, root, "stop");
    if (stopn && stopn->type != JSON_NULL) {
        const JsonNode* items[STOP_MAX];
        uint32_t        n_items = 0;
        if (stopn->type == JSON_STR) {
            items[n_items++] = stopn;
        } else if (stopn->type == JSON_ARR) {
            if (stopn->n_children > STOP_MAX) {
                respond_error(c, 400, "invalid_request_error",
                              "'stop' takes at most 4 strings");
                return -1;
            }
            for (uint32_t i = 0; i < stopn->n_children; i++) {
                const JsonNode* it = json_at(d, stopn, i);
                if (!it || it->type != JSON_STR) {
                    respond_error(c, 400, "invalid_request_error",
                                  "every element of 'stop' must be a string");
                    return -1;
                }
                items[n_items++] = it;
            }
        } else {
            respond_error(c, 400, "invalid_request_error",
                          "'stop' must be a string or an array of strings");
            return -1;
        }

        for (uint32_t i = 0; i < n_items; i++) {
            i64 need = json_string(body, items[i], NULL, 0);
            if (need <= 0) continue; /* empty, or unreadable: nothing to match */
            char* sp = (char*)malloc((size_t)need);
            if (!sp) {
                respond_error(c, 500, "server_error", "out of memory");
                return -1;
            }
            json_string(body, items[i], sp, (size_t)need);
            c->stops[c->n_stops]    = sp;
            c->stop_len[c->n_stops] = (size_t)need;
            c->n_stops++;
            if ((size_t)need > c->hold + 1) c->hold = (size_t)need - 1;
        }
    }

    *max_tokens        = DEFAULT_MAX_TOKENS;
    const JsonNode* mt = json_member(d, body, root, "max_tokens");
    if (mt) {
        double v;
        if (json_number(body, mt, &v) != 0 || v < 1 || v > (double)sv->max_ctx) {
            respond_error(c, 400, "invalid_request_error",
                          "'max_tokens' must be a number between 1 and the "
                          "context length");
            return -1;
        }
        *max_tokens = (uint32_t)v;
    }

    *out_body = body;
    *out_root = root;
    return 0;
}

static void handle_chat(Server* sv, Conn* c, const HttpRequest* r) {
    JsonDoc         d;
    const char*     body;
    const JsonNode* root;
    uint32_t        max_tokens;
    int             want_stream, want_usage;

    c->is_completion = 0;
    if (parse_body(sv, c, r, &d, &body, &root, &max_tokens, &want_stream,
                   &want_usage) != 0) {
        return;
    }

    const JsonNode* msgs = json_member(&d, body, root, "messages");
    if (!msgs || msgs->type != JSON_ARR || msgs->n_children == 0) {
        respond_error(c, 400, "invalid_request_error",
                      "'messages' must be a non-empty array");
        return;
    }

    stoke_chat_message* cm =
        (stoke_chat_message*)malloc(msgs->n_children * sizeof *cm);
    if (!cm) {
        respond_error(c, 500, "server_error", "out of memory");
        return;
    }
    for (uint32_t i = 0; i < msgs->n_children; i++) {
        const char* why = NULL;
        if (read_message(&d, body, json_at(&d, msgs, i), &cm[i], &why) != 0) {
            respond_error(c, 400, "invalid_request_error", why);
            free(cm);
            return;
        }
    }

    int64_t bound = stoke_chat_template_bound(cm, (int32_t)msgs->n_children);
    if (bound <= 0) {
        respond_error(c, 400, "invalid_request_error",
                      "messages are too large to template");
        free(cm);
        return;
    }

    uint32_t* ids = (uint32_t*)malloc((size_t)bound * sizeof(uint32_t));
    if (!ids) {
        respond_error(c, 500, "server_error", "out of memory");
        free(cm);
        return;
    }
    int32_t n_prompt = stoke_apply_chat_template(sv->e, cm,
                                                 (int32_t)msgs->n_children, ids,
                                                 (int32_t)bound);
    free(cm);
    if (n_prompt <= 0) {
        free(ids);
        if (n_prompt == STOKE_ERR_NO_TEMPLATE) {
            respond_error(c, 501, "invalid_request_error",
                          "this model has no chat template; use /v1/completions");
        } else {
            respond_error(c, 400, "invalid_request_error", stoke_last_error());
        }
        return;
    }

    /* Shrunk to the real count the moment it is known. The bound is ~3x the
     * message bytes and is only ever needed for the duration of the call
     * above; holding it for the life of the request would multiply this
     * server's memory by the number of connections. */
    uint32_t* shrunk = (uint32_t*)realloc(ids, (size_t)n_prompt * sizeof(uint32_t));
    if (shrunk) ids = shrunk;

    start_generation(sv, c, ids, n_prompt, max_tokens, want_stream, want_usage);
}

/* The half of a request that is identical whichever endpoint asked: bounds
 * check, submit, and either wait for the whole answer or start streaming.
 * TAKES OWNERSHIP of `ids` -- on every path here it is either handed to the
 * connection or freed. */
static void start_generation(Server* sv, Conn* c, uint32_t* ids, int32_t n_prompt,
                             uint32_t max_tokens, int want_stream, int want_usage) {
    if ((uint32_t)n_prompt + max_tokens > sv->max_ctx) {
        free(ids);
        char msg[192];
        snprintf(msg, sizeof msg,
                 "prompt is %d tokens and max_tokens is %u, which exceeds the "
                 "context length of %u",
                 (int)n_prompt, max_tokens, sv->max_ctx);
        respond_error(c, 400, "context_length_exceeded", msg);
        return;
    }

    c->gen_cap = (int32_t)max_tokens + 4;
    c->gen     = (uint32_t*)malloc((size_t)c->gen_cap * sizeof(uint32_t));
    if (!c->gen) {
        free(ids);
        respond_error(c, 500, "server_error", "out of memory");
        return;
    }

    c->prompt_ids  = ids;
    c->n_prompt    = (uint32_t)n_prompt;
    c->n_gen       = 0;
    c->req_id      = sv->next_req_id++;
    c->created     = (int64_t)time(NULL);
    c->want_usage  = want_usage;
    c->sent_role   = 0;
    c->stream_done = 0;
    c->pend_len    = 0;

    stoke_request req;
    memset(&req, 0, sizeof req);
    req.size       = sizeof req;
    req.id         = c->req_id;
    req.prompt_ids = c->prompt_ids;
    req.n_prompt   = c->n_prompt;
    req.max_tokens = max_tokens;
    req.eos_id     = (int64_t)sv->eos_id;

    if (stoke_submit(sv->e, &req) != STOKE_OK) {
        char msg[256];
        snprintf(msg, sizeof msg, "%s", stoke_last_error());
        c->req_id = 0;
        conn_release_request(sv, c);
        respond_error(c, 503, "server_error", msg);
        return;
    }

    if (!want_stream) {
        c->state = CONN_GEN;
        return;
    }

    /* ▶ THE HEAD GOES OUT NOW, BEFORE THE FIRST TOKEN EXISTS. That is the
     * point of streaming: a client blocked on response headers is a client
     * that has not started its clock. Everything after this is appended, so
     * NOTHING on this connection may call respond() again -- a status line in
     * the middle of a chunked body is not an error message, it is corruption.
     * The error paths above all run before this line, deliberately. */
    c->state = CONN_STREAM;
    sse_begin(c);
}

/* ---- /v1/completions ----------------------------------------------------
 *
 * ▶▶ RAW. NO TEMPLATE. That is the entire difference from /v1/chat/completions
 * and it is the point of the endpoint: a prompt is bytes, and wrapping it in
 * turn markers would silently change what the model was asked. It is also the
 * shape the M2 parity path uses, so a prompt that reproduced a parity result
 * there reproduces it here.
 *
 * ⚠ NO BOS EITHER. `add_special` is 0 because the chat path does not add one
 * (the template's own <|im_start|> is the boundary), and a completions endpoint
 * that silently prepended one would answer a different question from the CLI
 * given the same bytes.
 */
static void handle_completions(Server* sv, Conn* c, const HttpRequest* r) {
    JsonDoc         d;
    const char*     body;
    const JsonNode* root;
    uint32_t        max_tokens;
    int             want_stream, want_usage;

    c->is_completion = 1;
    if (parse_body(sv, c, r, &d, &body, &root, &max_tokens, &want_stream,
                   &want_usage) != 0) {
        return;
    }

    const JsonNode* pn = json_member(&d, body, root, "prompt");
    if (!pn || pn->type != JSON_STR) {
        /* OpenAI also allows an array of prompts (and of token ids), which
         * means N completions in one response. Refusing plainly beats
         * answering only the first and looking like it worked. */
        respond_error(c, 400, "invalid_request_error",
                      "'prompt' must be a string (an array of prompts is not "
                      "implemented)");
        return;
    }

    /* Decoded out of the JSON rather than used raw: unlike a chat message --
     * whose escapes are handed to the template as ordinary text -- a
     * completion's prompt IS the model's input verbatim, so `\n` must be a
     * newline here and not two characters. */
    i64 need = json_string(body, pn, NULL, 0);
    if (need < 0) {
        respond_error(c, 400, "invalid_request_error", "'prompt' is not a string");
        return;
    }
    char* text = (char*)malloc((size_t)need + 1);
    if (!text) {
        respond_error(c, 500, "server_error", "out of memory");
        return;
    }
    json_string(body, pn, text, (size_t)need);
    text[need] = '\0';

    /* tok.h's stated bound: at most 3 ids per byte, and +1 covers an empty
     * prompt asking for a zero-length allocation. */
    int64_t cap = (int64_t)need * 3 + 1;
    if (cap > INT32_MAX) {
        free(text);
        respond_error(c, 400, "invalid_request_error", "'prompt' is too long");
        return;
    }
    uint32_t* ids = (uint32_t*)malloc((size_t)cap * sizeof(uint32_t));
    if (!ids) {
        free(text);
        respond_error(c, 500, "server_error", "out of memory");
        return;
    }
    int32_t n_prompt = stoke_tokenize(sv->e, text, need, /*add_special=*/0, ids,
                                      (int32_t)cap);
    free(text);
    if (n_prompt <= 0) {
        free(ids);
        respond_error(c, 400, "invalid_request_error",
                      n_prompt == 0 ? "'prompt' encoded to zero tokens"
                                    : stoke_last_error());
        return;
    }

    start_generation(sv, c, ids, n_prompt, max_tokens, want_stream, want_usage);
}

/* ---- /v1/models ---------------------------------------------------------
 *
 * One entry, because this server has one file open. The id is that file's
 * name: it is the only name it actually has, and reporting something invented
 * would send a client looking for a model that does not exist. Nothing routes
 * on it -- the `model` field of a request is echoed back, never matched --
 * which is exactly what llama-server does with the same field. */
static void handle_models(Server* sv, Conn* c) {
    Buf b = {0};
    buf_str(&b, "{\"object\":\"list\",\"data\":[{\"id\":\"");
    buf_json_str(&b, sv->model_id, strlen(sv->model_id));
    buf_fmt(&b, "\",\"object\":\"model\",\"created\":%lld,\"owned_by\":\"stoke\"}]}\n",
            (long long)0);
    if (b.oom) {
        respond_error(c, 500, "server_error", "out of memory");
    } else {
        respond(c, 200, "application/json", b.b, b.len);
    }
    buf_free(&b);
}

/* ---- routing ------------------------------------------------------------ */

static int path_is(const HttpRequest* r, const char* buf, const char* want) {
    size_t n = strlen(want);
    return r->path_len == n && memcmp(buf + r->path_off, want, n) == 0;
}

static int method_is(const HttpRequest* r, const char* buf, const char* want) {
    size_t n = strlen(want);
    return r->method_len == n && memcmp(buf + r->method_off, want, n) == 0;
}

static void handle_request(Server* sv, Conn* c, const HttpRequest* r) {
    c->keep_alive = r->keep_alive;

    if (path_is(r, c->in, "/health")) {
        if (!method_is(r, c->in, "GET")) {
            respond_error(c, 405, "invalid_request_error", "use GET /health");
            return;
        }
        static const char ok[] = "{\"status\":\"ok\"}\n";
        respond(c, 200, "application/json", ok, sizeof ok - 1);
        return;
    }

    if (path_is(r, c->in, "/v1/chat/completions")) {
        if (!method_is(r, c->in, "POST")) {
            respond_error(c, 405, "invalid_request_error",
                          "use POST /v1/chat/completions");
            return;
        }
        handle_chat(sv, c, r);
        return;
    }

    if (path_is(r, c->in, "/v1/completions")) {
        if (!method_is(r, c->in, "POST")) {
            respond_error(c, 405, "invalid_request_error",
                          "use POST /v1/completions");
            return;
        }
        handle_completions(sv, c, r);
        return;
    }

    if (path_is(r, c->in, "/v1/models")) {
        if (!method_is(r, c->in, "GET")) {
            respond_error(c, 405, "invalid_request_error", "use GET /v1/models");
            return;
        }
        handle_models(sv, c);
        return;
    }

    respond_error(c, 404, "invalid_request_error", "unknown endpoint");
}

/* ---- the loop ----------------------------------------------------------- */

static void service_read(Server* sv, Conn* c) {
    if (c->in_cap == 0) {
        c->in = (char*)malloc(IN_CAP_INITIAL);
        if (!c->in) {
            conn_close(sv, c);
            return;
        }
        c->in_cap = IN_CAP_INITIAL;
    }

    if (c->in_len == c->in_cap) {
        if (c->in_cap >= IN_CAP_MAX) {
            /* The parser's own caps should have refused this first; reaching
             * here means the request is legal-so-far and simply enormous. */
            respond_error(c, 413, "invalid_request_error", "request is too large");
            return;
        }
        size_t want = c->in_cap * 2;
        if (want > IN_CAP_MAX) want = IN_CAP_MAX;
        char* nb = (char*)realloc(c->in, want);
        if (!nb) {
            conn_close(sv, c);
            return;
        }
        c->in     = nb;
        c->in_cap = want;
    }

    i64 n = net_recv(c->s, c->in + c->in_len, c->in_cap - c->in_len);
    if (n == NET_WOULD_BLOCK) {
        /* ▶▶ NOTHING NEW ON THE SOCKET IS NOT THE SAME AS NOTHING TO DO, and
         * this is the whole of pipelining. A client that wrote two requests in
         * one call has both sitting in `in`; the first recv() took them
         * together, so no further byte is coming and select() will never call
         * this socket readable again. Returning here answers request one and
         * leaves request two to sit until the idle reaper closes the
         * connection a minute later -- which is what this did until
         * tests/serve_pipeline.c was pointed at it (2026-08-29).
         *
         * With an empty buffer there genuinely is nothing to do. */
        if (c->in_len == 0) return;
    } else if (n <= 0) { /* NET_CLOSED or NET_ERR: the client is gone either way */
        conn_close(sv, c);
        return;
    } else {
        c->in_len += (size_t)n;
        c->last_active = time(NULL);
    }

    HttpRequest r;
    HttpError   he;
    HttpParseStatus st = http_parse(c->in, c->in_len, &r, &he);
    if (st == HTTP_PARSE_NEED_MORE) return;
    if (st == HTTP_PARSE_ERROR) {
        respond_error(c, he.status, "invalid_request_error",
                      he.reason ? he.reason : "malformed request");
        return;
    }

    if (sv->verbose) {
        fprintf(stderr, "stoke serve: %.*s %.*s\n", (int)r.method_len,
                c->in + r.method_off, (int)r.path_len, c->in + r.path_off);
    }

    size_t used = (size_t)r.head_len + r.body_len;
    handle_request(sv, c, &r);
    /* Retired AFTER handling, because every field of `r` and every JSON slice
     * points into these bytes. Retiring first would hand the handler a buffer
     * that had already moved under it. */
    conn_consume(c, used);
}

static void service_write(Server* sv, Conn* c) {
    while (c->out_sent < c->out.len) {
        i64 n = net_send(c->s, c->out.b + c->out_sent, c->out.len - c->out_sent);
        if (n == NET_WOULD_BLOCK) break;
        if (n < 0) {
            conn_close(sv, c);
            return;
        }
        c->out_sent += (size_t)n;
    }

    if (c->out_sent < c->out.len) {
        /* ▶▶ THE SLOW-CLIENT POLICY, and it is a policy rather than a bug
         * guard. A reader that stops reading mid-stream cannot be allowed to
         * make this process buy memory on its behalf for as long as it likes,
         * and it must never be allowed to make the SCHEDULER wait -- every
         * other client's tokens come off the same loop. So the backpressure
         * stops at this connection's own buffer: past the cap, the client is
         * disconnected and ITS request is cancelled, which returns the KV
         * blocks it was holding. Nobody else notices. */
        if (c->state == CONN_STREAM &&
            c->out.len - c->out_sent > sv->max_pending) {
            if (sv->verbose) {
                fprintf(stderr,
                        "stoke serve: stream id=%llu dropped -- %zu bytes "
                        "unread past the %zu-byte cap\n",
                        (unsigned long long)c->req_id,
                        c->out.len - c->out_sent, sv->max_pending);
            }
            conn_close(sv, c); /* cancels the request on the way out */
        }
        return;
    }

    /* Fully drained. Rewound rather than freed: a stream's next token reuses
     * the allocation instead of asking the allocator once per token. */
    c->out.len     = 0;
    c->out_sent    = 0;
    c->last_active = time(NULL);

    /* A stream that has not sent its sentinel is not a finished request, no
     * matter how empty its buffer is right now. */
    if (c->state == CONN_STREAM && !c->stream_done) return;

    /* ▶▶ COUNTED HERE, AND ONLY HERE, BECAUSE `completed` IS WHAT ENDS THE
     * LOOP. Counting a request when its response was BUILT is a whole answer
     * short: finish_chat() runs in service_generation(), which is the last
     * thing in a loop turn, so the count would reach max_requests with the
     * body still sitting in c->out and the next while-check would exit and
     * close the socket under it. That is not a hypothetical -- it is what this
     * server did the first time it was pointed at curl (2026-08-29): the
     * generation ran, the log said `finish=stop`, and the client got
     * `curl: (52) Empty reply from server`.
     *
     * It also makes the count mean the same thing for every response. An error
     * reply is a served request; when only the two success paths counted, a
     * session of 400s never reached max_requests at all and the server sat
     * there until something killed it -- a gate hang whose cause is invisible.
     *
     * Every response passes through here exactly once, including the
     * zero-length one respond() falls back to when it runs out of memory. */
    sv->completed++;

    if (!c->keep_alive) {
        conn_close(sv, c);
        return;
    }
    c->state = CONN_READ;

    /* A pipelining client may already have sent the next request; without this
     * it would sit in the buffer until the client wrote something more, which
     * for a client waiting on a response is never. */
    if (c->in_len > 0) service_read(sv, c);
}

/* An engine failure, reported to whichever kind of response is in flight.
 *
 * ⚠ A STREAM CANNOT BE TOLD. Its 200 and its headers left before the first
 * token existed, so there is no status code left to change -- writing an error
 * envelope into the middle of a chunked body produces neither an error nor a
 * response, just corruption the client parses as content. Dropping the
 * connection is the only honest signal, and it is the one SSE clients already
 * understand. */
static void generation_failed(Server* sv, Conn* c, const char* why) {
    if (c->state == CONN_STREAM) {
        conn_close(sv, c);
        return;
    }
    c->req_id = 0;
    conn_release_request(sv, c);
    respond_error(c, 500, "server_error", why);
}

static void service_generation(Server* sv) {
    int any = 0;
    for (int i = 0; i < MAX_CONNS; i++) {
        ConnState st = sv->conns[i].state;
        if (st == CONN_GEN || (st == CONN_STREAM && !sv->conns[i].stream_done)) {
            any = 1;
        }
    }
    if (!any) return;

    uint32_t rows = 0;
    if (stoke_step(sv->e, &rows) != STOKE_OK) {
        for (int i = 0; i < MAX_CONNS; i++) {
            Conn* c = &sv->conns[i];
            if (c->state != CONN_GEN && c->state != CONN_STREAM) continue;
            if (c->state == CONN_STREAM && c->stream_done) continue;
            generation_failed(sv, c, "the engine failed to step");
        }
        return;
    }

    for (int i = 0; i < MAX_CONNS; i++) {
        Conn* c = &sv->conns[i];
        if (c->state != CONN_GEN && c->state != CONN_STREAM) continue;
        if (c->state == CONN_STREAM && c->stream_done) continue;

        int     streaming = (c->state == CONN_STREAM);
        int32_t finish    = STOKE_FINISH_NONE;

        /* max_tokens bounds generation and gen_cap is max_tokens+4, so this
         * cannot legitimately fill -- but polling with a zero cap is argument
         * misuse, and an engine that over-produced would turn a bug of ours
         * into an error the client sees as ours. Cancel and finish instead. */
        if (c->n_gen >= c->gen_cap) {
            stoke_cancel(sv->e, c->req_id);
            if (streaming) {
                stream_final(sv, c, STOKE_FINISH_MAX_TOKENS);
            } else {
                finish_chat(sv, c, STOKE_FINISH_MAX_TOKENS);
            }
            continue;
        }

        int32_t before = c->n_gen;
        int32_t k      = stoke_poll(sv->e, c->req_id, c->gen + c->n_gen,
                                    c->gen_cap - c->n_gen, &finish);
        if (k < 0) {
            generation_failed(sv, c, "the engine failed to poll");
            continue;
        }
        c->n_gen += k;

        /* Decoded in BOTH modes and on every batch, because a stop string must
         * be able to CANCEL generation the moment it appears -- which is
         * impossible for a path that waits until the end to decode. */
        if (k > 0 && !consume_tokens(sv, c, c->gen + before, k, streaming)) {
            continue; /* the connection is gone; consume_tokens closed it */
        }

        if (!streaming) {
            if (finish != STOKE_FINISH_NONE) finish_chat(sv, c, finish);
            continue;
        }
        if (finish != STOKE_FINISH_NONE && !stream_final(sv, c, finish)) {
            continue;
        }

        /* ▶ PUSHED NOW RATHER THAN NEXT TURN. The send is non-blocking, so the
         * cost of trying is a syscall that either moves the token or says
         * WOULD_BLOCK -- and the alternative adds a whole loop turn of latency
         * to every token, which is the exact quantity a streaming client is
         * measuring. */
        if (c->state == CONN_STREAM && c->out_sent < c->out.len) {
            service_write(sv, c);
        }
    }
}

static void reap_idle(Server* sv) {
    time_t now = time(NULL);
    for (int i = 0; i < MAX_CONNS; i++) {
        Conn* c = &sv->conns[i];
        /* ▶▶▶ ONLY CONN_READ IS EVER IDLE, AND THE OTHERS ARE NOT A MATTER OF
         * DEGREE. This reaper exists for one thing: a client that opened a
         * socket and then said nothing, which would otherwise hold a slot
         * forever. That is CONN_READ and nothing else. Every other state is one
         * where WE owe the client bytes, and closing it discards an answer we
         * have already paid for.
         *
         * ⚠⚠ IT USED TO REAP CONN_WRITE, AND THAT DESTROYED FINISHED
         * RESPONSES. `last_active` is stamped when bytes move, so a request
         * whose GENERATION takes longer than the timeout arrives at CONN_WRITE
         * already stale -- and was closed on the very next turn, after the
         * engine had done all the work and the log had recorded
         * `finish=length`. The client saw `curl: (52) Empty reply from server`.
         * Found on the TSan leg, where a forward pass is slow enough to cross
         * sixty seconds; in production it is any long generation, any large
         * prompt, any busy server. A timeout that measures how long OUR work
         * took, and then punishes the client for it, is not a timeout. */
#ifdef STOKE_MUT_REAP_WRITE
        /* ⚠ DELIBERATELY WRONG -- the pre-fix reaper, restored. It also closes
         * CONN_WRITE, which throws away responses the engine has already
         * produced whenever generation outlasts the timeout. Built only by
         *   mingw32-make BUILDDIR=build/win-mut-reaper \
         *                EXTRA_CFLAGS=-DSTOKE_MUT_REAP_WRITE=1
         * to prove tests/serve_golden.sh's --idle-timeout guard can fail. It
         * announces itself, because a build that silently dropped its -D is a
         * CLEAN binary and the leg would then report "the gate did not catch
         * it" about a mutant that was never there. */
        if (c->state == CONN_FREE || c->state == CONN_GEN ||
            c->state == CONN_STREAM) {
            continue;
        }
#else
        if (c->state != CONN_READ) continue;
#endif
        if (now - c->last_active > sv->idle_secs) conn_close(sv, c);
    }
}

int server_run(const char* gguf_path, const stoke_open_opts* engine_opts,
               const ServerOpts* srv_opts) {
    ServerOpts so;
    memset(&so, 0, sizeof so);
    if (srv_opts) so = *srv_opts;

    Server sv;
    memset(&sv, 0, sizeof sv);
    sv.verbose     = so.verbose;
    sv.next_req_id = 1;
    sv.ping_ms     = so.sse_ping_ms > 0 ? so.sse_ping_ms : SSE_PING_MS_DEFAULT;
    sv.max_pending = so.sse_max_pending > 0 ? so.sse_max_pending
                                            : SSE_MAX_PENDING_DEFAULT;
    sv.send_buffer = so.sse_send_buffer;
    sv.idle_secs   = so.idle_timeout_secs > 0 ? so.idle_timeout_secs
                                             : IDLE_TIMEOUT_SECS;
    for (int i = 0; i < MAX_CONNS; i++) sv.conns[i].s = NET_INVALID_SOCK;

    sv.e = stoke_open(gguf_path, engine_opts);
    if (!sv.e) {
        fprintf(stderr, "stoke serve: %s\n", stoke_last_error());
        return 1;
    }
    sv.max_ctx = stoke_max_ctx(sv.e);
    sv.eos_id  = stoke_eos_id(sv.e);

    /* The model id /v1/models reports: the GGUF's basename with its extension
     * removed. Both separators are checked because a Windows path may carry
     * either, and this is a user-supplied argument rather than something the
     * program built. */
    {
        const char* base = gguf_path;
        for (const char* p = gguf_path; *p; p++) {
            if (*p == '/' || *p == '\\') base = p + 1;
        }
        snprintf(sv.model_id, sizeof sv.model_id, "%s", base);
        size_t n = strlen(sv.model_id);
        if (n > 5 && strcmp(sv.model_id + n - 5, ".gguf") == 0) {
            sv.model_id[n - 5] = '\0';
        }
        if (sv.model_id[0] == '\0') snprintf(sv.model_id, sizeof sv.model_id, "stoke");
    }

    /* Asked ONCE, at startup, exactly as stoke.h says to: it is a fact about
     * the model and cannot change between requests. */
    if (!stoke_has_chat_template(sv.e)) {
        fprintf(stderr, "stoke serve: warning -- this model has no chat "
                        "template; /v1/chat/completions will answer 501\n");
    }

    NetSock lis = net_listen(so.bind_addr, so.port, 16);
    if (lis == NET_INVALID_SOCK) {
        fprintf(stderr, "stoke serve: %s\n", net_error());
        stoke_close(sv.e);
        return 1;
    }
    net_set_nonblocking(lis);

#ifdef STOKE_MUT_REAP_WRITE
    fprintf(stderr, "stoke serve: MUTANT BUILD -- STOKE_MUT_REAP_WRITE "
                    "(the reaper closes CONN_WRITE, discarding finished "
                    "responses). NEVER use this binary for anything but "
                    "the mutation leg.\n");
#endif
#ifdef STOKE_MUT_ROLE_EVERY_CHUNK
    fprintf(stderr, "stoke serve: MUTANT BUILD -- STOKE_MUT_ROLE_EVERY_CHUNK "
                    "(role on every delta). NEVER use this binary for anything "
                    "but the mutation leg.\n");
#endif

    /* Printed BEFORE the loop and flushed, so a script can wait on this line;
     * with port 0 it is the only way to learn where the server actually is. */
    printf("stoke serve: listening on %s:%u  ctx=%u\n",
           so.bind_addr ? so.bind_addr : "127.0.0.1",
           (unsigned)net_port_of(lis), sv.max_ctx);
    fflush(stdout);

    /* ▶ A DEADLINE, NOT A KILL. See ServerOpts.max_seconds: the sanitizer soak
     * needs the process to END ITSELF, because LeakSanitizer reports at exit
     * and a killed server is one whose leaks were never looked at. */
    double stop_at_ms = so.max_seconds > 0
                            ? now_ms() + (double)so.max_seconds * 1000.0
                            : 0.0;

    while ((so.max_requests == 0 || sv.completed < so.max_requests) &&
           (stop_at_ms == 0.0 || now_ms() < stop_at_ms)) {
        NetWait w[MAX_CONNS + 1];
        /* ▶▶ WHICH CONNECTION EACH WAIT ENTRY BELONGS TO (-1 = the listener),
         * carried EXPLICITLY rather than re-derived by walking the connections
         * in the same order a second time.
         *
         * ⚠ The re-derivation is what this replaces, and it was wrong in a way
         * that is invisible from the outside: accept() runs BETWEEN the two
         * walks and fills the lowest free slot, so a connection that had no
         * entry when the set was built has a state by the time the set is
         * dispatched. Every later connection then reads its NEIGHBOUR's
         * readiness, and the last one reads w[n] -- past what net_wait() wrote,
         * so uninitialised stack. The visible cost is only a turn of latency,
         * because a wrongly-reported ready socket just answers WOULD_BLOCK and
         * a wrongly-reported idle one is polled again 50 ms later; that is
         * exactly why no black-box test in this gate can see it, and why the
         * fix is to remove the coupling rather than to test for it. */
        int     wconn[MAX_CONNS + 1];
        int     n = 0;
        int     generating = 0;

        /* The heartbeat, checked against each stream's own last write, BEFORE
         * the wait set is built so a ping queued here gets a slot in it rather
         * than waiting a turn. It fires only in the gap this server actually
         * has -- prefill of a long prompt, or a scheduler busy with other
         * sequences -- and never during steady generation, where the tokens
         * are the heartbeat. */
        for (int i = 0; i < MAX_CONNS; i++) {
            Conn* c = &sv.conns[i];
            if (c->state != CONN_STREAM || c->stream_done) continue;
            if (now_ms() - c->last_out_ms >= (double)sv.ping_ms) {
                sse_chunk(c, ": ping\n\n", sizeof ": ping\n\n" - 1);
            }
        }

        int free_slots = 0;
        for (int i = 0; i < MAX_CONNS; i++) {
            if (sv.conns[i].state == CONN_FREE) free_slots++;
        }
        if (free_slots > 0) {
            w[n].s          = lis;
            w[n].want_read  = 1;
            w[n].want_write = 0;
            wconn[n]        = -1;
            n++;
        }
        for (int i = 0; i < MAX_CONNS; i++) {
            Conn* c = &sv.conns[i];
            if (c->state == CONN_FREE) continue;
            if (c->state == CONN_GEN) {
                generating = 1;
                continue; /* nothing to do on its socket until it finishes */
            }
            if (c->state == CONN_STREAM) {
                /* ▶ A STREAM IS BOTH AT ONCE, which is the whole reason it
                 * needed a state of its own. It is still generating -- so the
                 * timeout below must stay at 0 for it -- AND it has bytes for
                 * a socket that may not be ready. With nothing queued there is
                 * nothing to wait on, so it takes no slot in the set and the
                 * engine simply steps again. */
                if (!c->stream_done) generating = 1;
                if (c->out_sent >= c->out.len) continue;
            }
            w[n].s          = c->s;
            w[n].want_read  = (c->state == CONN_READ);
            w[n].want_write = (c->state == CONN_WRITE ||
                               c->state == CONN_STREAM);
            w[n].can_read = w[n].can_write = 0;
            wconn[n]        = i;
            n++;
        }

        /* ▶ THE TIMEOUT IS THE WHOLE SCHEDULING POLICY. With a request in
         * flight it is 0: check the sockets, then immediately run a forward
         * pass. Idle, it is 50 ms, so a server with no work costs nothing. */
        int rc = net_wait(w, n, generating ? 0 : 50);
        if (rc == NET_ERR) {
            fprintf(stderr, "stoke serve: %s\n", net_error());
            break;
        }

        if (free_slots > 0) {
            /* w[0] is the listener exactly when one was added, which is the
             * same condition that put it there. */
            if (w[0].can_read) {
                NetSock cs = net_accept(lis);
                if (cs != NET_INVALID_SOCK) {
                    net_set_nonblocking(cs);
                    net_set_nodelay(cs);
                    if (sv.send_buffer > 0) {
                        net_set_send_buffer(cs, sv.send_buffer);
                    }
                    for (int i = 0; i < MAX_CONNS; i++) {
                        if (sv.conns[i].state != CONN_FREE) continue;
                        Conn* c = &sv.conns[i];
                        memset(c, 0, sizeof *c);
                        c->s           = cs;
                        c->state       = CONN_READ;
                        c->keep_alive  = 1;
                        c->last_active = time(NULL);
                        cs             = NET_INVALID_SOCK;
                        break;
                    }
                    /* No slot after all (a race with our own count): refuse by
                     * closing rather than leaking the descriptor. */
                    net_close(cs);
                }
            }
        }
        /* Driven by the wait set, not by the connection table: an entry knows
         * which connection it was built for, and a connection accepted just
         * above has no entry at all -- it is simply not serviced this turn,
         * which is correct, because nothing sampled its socket. */
        for (int k = 0; k < n; k++) {
            if (wconn[k] < 0) continue; /* the listener, handled above */
            Conn*    c = &sv.conns[wconn[k]];
            NetWait* e = &w[k];
            if (c->state == CONN_FREE || c->state == CONN_GEN) continue;
            if (c->state == CONN_READ && e->can_read) service_read(&sv, c);
            /* Re-checked, not else-if: service_read may have moved this
             * connection to CONN_WRITE with a complete response already built,
             * or to CONN_STREAM with the head queued, and waiting a whole loop
             * turn to start sending would add a scheduling delay to every
             * request that never blocked. */
            if ((c->state == CONN_WRITE || c->state == CONN_STREAM) &&
                (e->can_write || e->can_read)) {
                service_write(&sv, c);
            }
        }

        service_generation(&sv);
        reap_idle(&sv);
    }

    for (int i = 0; i < MAX_CONNS; i++) {
        if (sv.conns[i].state != CONN_FREE) conn_close(&sv, &sv.conns[i]);
    }
    buf_free(&sv.scratch);
    net_close(lis);
    stoke_close(sv.e);
    if (sv.verbose) {
        fprintf(stderr, "stoke serve: %d requests completed, exiting\n",
                sv.completed);
    }
    return 0;
}
