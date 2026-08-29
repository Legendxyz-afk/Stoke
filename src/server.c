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

typedef enum {
    CONN_FREE = 0,
    CONN_READ,  /* accumulating a request                    */
    CONN_GEN,   /* submitted to the engine, generating       */
    CONN_WRITE, /* response built, draining to the socket    */
} ConnState;

typedef struct {
    ConnState state;
    NetSock   s;

    char*  in;
    size_t in_len, in_cap;

    char*  out;
    size_t out_len, out_sent;

    int keep_alive;

    /* the live request */
    uint64_t  req_id;
    uint32_t* prompt_ids; /* BORROWED by the engine while CONN_GEN */
    uint32_t  n_prompt;
    uint32_t* gen;
    int32_t   n_gen, gen_cap;
    char      model[128]; /* echoed back; copied because `in` is reused */

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
} Server;

/* ---- a growable byte buffer, for building responses -------------------- */

typedef struct {
    char*  b;
    size_t len, cap;
    int    oom;
} Buf;

static void buf_free(Buf* d) {
    free(d->b);
    d->b   = NULL;
    d->len = d->cap = 0;
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
}

static void conn_close(Server* sv, Conn* c) {
    conn_release_request(sv, c);
    net_close(c->s);
    c->s = NET_INVALID_SOCK;
    free(c->in);
    free(c->out);
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

static void respond(Conn* c, int status, const char* content_type,
                    const char* body, size_t body_len) {
    Buf h = {0};
    buf_fmt(&h, "HTTP/1.1 %d %s\r\n", status, http_status_text(status));
    buf_fmt(&h, "Content-Type: %s\r\n", content_type);
    buf_fmt(&h, "Content-Length: %zu\r\n", body_len);
    buf_str(&h, c->keep_alive ? "Connection: keep-alive\r\n"
                              : "Connection: close\r\n");
    buf_str(&h, "\r\n");
    buf_add(&h, body, body_len);

    if (h.oom) {
        buf_free(&h);
        free(c->out); /* the previous response, if any -- not leaked here */
        c->keep_alive = 0;
        c->state      = CONN_WRITE;
        c->out        = NULL;
        c->out_len    = 0;
        c->out_sent   = 0;
        return;
    }
    free(c->out);
    c->out      = h.b;
    c->out_len  = h.len;
    c->out_sent = 0;
    c->state    = CONN_WRITE;
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

/* ---- the chat endpoint -------------------------------------------------- */

/* Decodes `n` ids to text, DROPPING the end-of-sequence id.
 *
 * ⚠ THE DROP IS THIS LAYER'S JOB AND tok.h SAYS SO. stoke_poll reports every id
 * the model generated, end-of-sequence included, and detokenizing is the
 * mechanical inverse of the vocab table -- so the stop token comes back as the
 * literal text "<|im_end|>". Deciding whether a control token is shown to an
 * API client is a policy choice, and the policy for an OpenAI-compatible
 * endpoint is that it is not: `message.content` is the assistant's words. */
static void decode_tokens(Server* sv, const uint32_t* ids, int32_t n, Buf* out) {
    for (int32_t i = 0; i < n; i++) {
        if (ids[i] == sv->eos_id) continue;
        char    piece[512];
        int32_t m = stoke_detokenize(sv->e, ids[i], piece, (int32_t)sizeof piece);
        if (m <= 0 || (size_t)m > sizeof piece) continue;
        buf_add(out, piece, (size_t)m);
    }
}

static const char* finish_reason_of(int32_t finish) {
    switch (finish) {
        case STOKE_FINISH_MAX_TOKENS: return "length";
        case STOKE_FINISH_EOS: return "stop";
        /* Cancellation has no OpenAI spelling; "stop" is the honest one, since
         * from the client's side the answer simply ended. */
        default: return "stop";
    }
}

static void finish_chat(Server* sv, Conn* c, int32_t finish) {
    Buf text = {0};
    decode_tokens(sv, c->gen, c->n_gen, &text);

    Buf b = {0};
    buf_fmt(&b, "{\"id\":\"chatcmpl-%llu\",\"object\":\"chat.completion\","
                "\"created\":%lld,\"model\":\"",
            (unsigned long long)c->req_id, (long long)time(NULL));
    buf_json_str(&b, c->model, strlen(c->model));
    buf_str(&b, "\",\"choices\":[{\"index\":0,\"message\":{\"role\":"
                "\"assistant\",\"content\":\"");
    buf_json_str(&b, text.b ? text.b : "", text.len);
    buf_fmt(&b, "\"},\"logprobs\":null,\"finish_reason\":\"%s\"}],",
            finish_reason_of(finish));
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
    buf_free(&text);

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

static void handle_chat(Server* sv, Conn* c, const HttpRequest* r) {
    const char* body     = c->in + r->body_off;
    size_t      body_len = r->body_len;
    if (body_len == 0) {
        respond_error(c, 400, "invalid_request_error", "request body is empty");
        return;
    }

    JsonDoc   d;
    JsonError je;
    if (json_parse(body, body_len, &d, &je) != 0) {
        char msg[160];
        snprintf(msg, sizeof msg, "body is not valid JSON (%s at byte %u)",
                 je.reason ? je.reason : "?", je.offset);
        respond_error(c, 400, "invalid_request_error", msg);
        return;
    }
    const JsonNode* root = json_root(&d);
    if (!root || root->type != JSON_OBJ) {
        respond_error(c, 400, "invalid_request_error", "body must be an object");
        return;
    }

    /* stream:true is refused rather than silently answered whole. A client that
     * asked for a stream and got one JSON blob would hang waiting for events
     * that never come -- a loud 501 is the kinder failure. */
    const JsonNode* stream = json_member(&d, body, root, "stream");
    int             want_stream = 0;
    if (stream && json_bool(body, stream, &want_stream) == 0 && want_stream) {
        respond_error(c, 501, "invalid_request_error",
                      "streaming responses are not implemented");
        return;
    }

    const JsonNode* model = json_member(&d, body, root, "model");
    c->model[0]           = '\0';
    if (model && model->type == JSON_STR) {
        i64 mn = json_string(body, model, c->model, sizeof c->model - 1);
        if (mn >= 0 && (size_t)mn < sizeof c->model) c->model[mn] = '\0';
    }
    if (c->model[0] == '\0') snprintf(c->model, sizeof c->model, "stoke");

    uint32_t        max_tokens = DEFAULT_MAX_TOKENS;
    const JsonNode* mt         = json_member(&d, body, root, "max_tokens");
    if (mt) {
        double v;
        if (json_number(body, mt, &v) != 0 || v < 1 || v > (double)sv->max_ctx) {
            respond_error(c, 400, "invalid_request_error",
                          "'max_tokens' must be a number between 1 and the "
                          "context length");
            return;
        }
        max_tokens = (uint32_t)v;
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

    c->prompt_ids = ids;
    c->n_prompt   = (uint32_t)n_prompt;
    c->n_gen      = 0;
    c->req_id     = sv->next_req_id++;

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
    c->state = CONN_GEN;
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
    while (c->out_sent < c->out_len) {
        i64 n = net_send(c->s, c->out + c->out_sent, c->out_len - c->out_sent);
        if (n == NET_WOULD_BLOCK) return;
        if (n < 0) {
            conn_close(sv, c);
            return;
        }
        c->out_sent += (size_t)n;
    }

    free(c->out);
    c->out      = NULL;
    c->out_len  = 0;
    c->out_sent = 0;
    c->last_active = time(NULL);

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

static void service_generation(Server* sv) {
    int any = 0;
    for (int i = 0; i < MAX_CONNS; i++) {
        if (sv->conns[i].state == CONN_GEN) any = 1;
    }
    if (!any) return;

    uint32_t rows = 0;
    if (stoke_step(sv->e, &rows) != STOKE_OK) {
        for (int i = 0; i < MAX_CONNS; i++) {
            Conn* c = &sv->conns[i];
            if (c->state != CONN_GEN) continue;
            c->req_id = 0;
            conn_release_request(sv, c);
            respond_error(c, 500, "server_error", "the engine failed to step");
        }
        return;
    }

    for (int i = 0; i < MAX_CONNS; i++) {
        Conn* c = &sv->conns[i];
        if (c->state != CONN_GEN) continue;

        int32_t finish = STOKE_FINISH_NONE;
        /* max_tokens bounds generation and gen_cap is max_tokens+4, so this
         * cannot legitimately fill -- but polling with a zero cap is argument
         * misuse, and an engine that over-produced would turn a bug of ours
         * into an error the client sees as ours. Cancel and finish instead. */
        if (c->n_gen >= c->gen_cap) {
            stoke_cancel(sv->e, c->req_id);
            finish_chat(sv, c, STOKE_FINISH_MAX_TOKENS);
            continue;
        }
        int32_t k = stoke_poll(sv->e, c->req_id, c->gen + c->n_gen,
                               c->gen_cap - c->n_gen, &finish);
        if (k < 0) {
            c->req_id = 0;
            conn_release_request(sv, c);
            respond_error(c, 500, "server_error", "the engine failed to poll");
            continue;
        }
        c->n_gen += k;
        if (finish != STOKE_FINISH_NONE) finish_chat(sv, c, finish);
    }
}

static void reap_idle(Server* sv) {
    time_t now = time(NULL);
    for (int i = 0; i < MAX_CONNS; i++) {
        Conn* c = &sv->conns[i];
        /* Never a generating connection: it is not idle, it is waiting on us. */
        if (c->state == CONN_FREE || c->state == CONN_GEN) continue;
        if (now - c->last_active > IDLE_TIMEOUT_SECS) conn_close(sv, c);
    }
}

int server_run(const char* gguf_path, const stoke_open_opts* engine_opts,
               const ServerOpts* srv_opts) {
    ServerOpts so = {NULL, 0, 0, 0};
    if (srv_opts) so = *srv_opts;

    Server sv;
    memset(&sv, 0, sizeof sv);
    sv.verbose     = so.verbose;
    sv.next_req_id = 1;
    for (int i = 0; i < MAX_CONNS; i++) sv.conns[i].s = NET_INVALID_SOCK;

    sv.e = stoke_open(gguf_path, engine_opts);
    if (!sv.e) {
        fprintf(stderr, "stoke serve: %s\n", stoke_last_error());
        return 1;
    }
    sv.max_ctx = stoke_max_ctx(sv.e);
    sv.eos_id  = stoke_eos_id(sv.e);

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

    /* Printed BEFORE the loop and flushed, so a script can wait on this line;
     * with port 0 it is the only way to learn where the server actually is. */
    printf("stoke serve: listening on %s:%u  ctx=%u\n",
           so.bind_addr ? so.bind_addr : "127.0.0.1",
           (unsigned)net_port_of(lis), sv.max_ctx);
    fflush(stdout);

    while (so.max_requests == 0 || sv.completed < so.max_requests) {
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
            w[n].s          = c->s;
            w[n].want_read  = (c->state == CONN_READ);
            w[n].want_write = (c->state == CONN_WRITE);
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
             * and waiting a whole loop turn to start sending it would add a
             * scheduling delay to every request that never blocked. */
            if (c->state == CONN_WRITE && (e->can_write || e->can_read)) {
                service_write(&sv, c);
            }
        }

        service_generation(&sv);
        reap_idle(&sv);
    }

    for (int i = 0; i < MAX_CONNS; i++) {
        if (sv.conns[i].state != CONN_FREE) conn_close(&sv, &sv.conns[i]);
    }
    net_close(lis);
    stoke_close(sv.e);
    if (sv.verbose) {
        fprintf(stderr, "stoke serve: %d requests completed, exiting\n",
                sv.completed);
    }
    return 0;
}
