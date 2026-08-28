/* src/http.h -- the HTTP/1.1 request parser. M6 rung A2.
 *
 * ▶▶ WHAT IT IS. One function over a caller-owned byte buffer. You hand it
 * everything you have read from a socket so far; it answers OK, NEED_MORE or
 * ERROR-with-a-status-code. It never allocates, never blocks, never touches a
 * socket and never calls fatal(). Those four are the whole design.
 *
 * ▶▶ WHY IT NEVER CALLS fatal(). Everything reaching this parser is untrusted
 * bytes from the network, and a malformed request is ORDINARY EXPECTED INPUT --
 * none of the M6 error-handling split's three classes. It is not a trapped
 * load-time failure, not API misuse by our host, and not a broken internal
 * invariant. It is Tuesday. A parser that reached for fatal() would hand every
 * client on the network the ability to kill the host process, which is the one
 * outcome a library must never make reachable from outside.
 *
 * ▶▶ WHY IT RE-PARSES FROM THE START EACH TIME rather than resuming from saved
 * state. The head is bounded at HTTP_MAX_HEAD_BYTES, so the worst case is
 * re-scanning 8 KiB per socket read -- unmeasurable next to a forward pass, and
 * it buys the property that matters here: there is no partially-advanced
 * machine to get wrong. Incremental HTTP parsers desynchronise; this one cannot
 * hold a state to desynchronise. Call it again with more bytes and it is
 * exactly as correct as the first time.
 *
 * ▶▶ THE OUTPUT IS OFFSETS, NOT COPIES. Every field is an (offset, length) pair
 * into YOUR buffer, so nothing is duplicated and nothing has to be freed. The
 * buffer must therefore stay alive and unmoved for as long as the HttpRequest
 * is read -- realloc'ing it while holding an HttpRequest invalidates every
 * field at once. Re-parse after a growth, which costs the same 8 KiB scan.
 *
 * ▶ WHAT IT DELIBERATELY DOES NOT DO, each with the reason attached:
 *   - INBOUND CHUNKED TRANSFER-ENCODING. Clients do not send it to a chat
 *     completions endpoint. Rather than half-support it, a request carrying
 *     `Transfer-Encoding` is answered 501 -- a LOUD refusal, never a silent
 *     ignore. Ignoring the header while honouring Content-Length is the exact
 *     shape of a request-smuggling desync.
 *   - ABSOLUTE-FORM TARGETS (`GET http://host/path`). That is what a client
 *     sends to a forward proxy. An origin server accepting it is how a request
 *     ends up routed somewhere its author did not choose. 400.
 *   - obs-fold (a header continued onto an indented next line). Deprecated by
 *     RFC 7230, and a recipient is explicitly permitted to reject it. 400.
 *   - URL-decoding, header-value unfolding, cookie parsing, ranges, upgrades.
 *     Not needed by the four endpoints M6 serves, so not present to be wrong.
 */
#ifndef STOKE_HTTP_H
#define STOKE_HTTP_H

#include "common.h"

/* ---- bounds ------------------------------------------------------------
 *
 * These are the reason a hostile client cannot make the server allocate. They
 * are compile-time constants rather than parameters because they are policy for
 * the whole server and a per-connection knob nobody sets is just a bigger
 * surface. Each maps to exactly one status code so the refusal is legible from
 * the client side.
 */

/* Request line + every header + the terminating blank line. Exceeded -> 431. */
#define HTTP_MAX_HEAD_BYTES 8192

/* Header COUNT, independent of the byte budget: 8 KiB of one-byte-named
 * headers is thousands of them, and a fixed array is what keeps this parser
 * allocation-free. Exceeded -> 431. */
#define HTTP_MAX_HEADERS 64

/* Declared Content-Length. Exceeded -> 413, decided the instant the headers are
 * complete, so a client declaring 10 GB is refused having sent none of it. */
#define HTTP_MAX_BODY_BYTES (1u << 20)

/* ---- results ------------------------------------------------------------ */

typedef enum {
    HTTP_PARSE_OK        = 0, /* a complete request, body included            */
    HTTP_PARSE_NEED_MORE = 1, /* well-formed so far; read more and call again */
    HTTP_PARSE_ERROR     = 2  /* malformed; answer HttpError.status and close */
} HttpParseStatus;

/* `reason` is a static string literal -- never allocated, never NULL after an
 * ERROR, valid forever. It is a kebab-case SLUG rather than a sentence, and
 * that is deliberate: it is logged, it is asserted by name in the committed
 * corpus, and several distinct defects share one status code. A test that
 * checked only the number would pass a parser that rejected the right request
 * for the wrong cause. */
typedef struct {
    i32         status; /* 400 / 413 / 431 / 501 / 505; 0 when not an error */
    const char* reason;
} HttpError;

typedef struct {
    u32 name_off, name_len;
    u32 value_off, value_len;
} HttpHeader;

typedef struct {
    u32 method_off, method_len;
    u32 target_off, target_len; /* the whole request-target, path?query      */
    u32 path_off, path_len;     /* target up to the first '?'                */
    u32 query_off, query_len;   /* after it; length 0 when there is no '?'   */

    i32 http_minor; /* 0 for HTTP/1.0, 1 for HTTP/1.1                        */

    u32        n_headers;
    HttpHeader headers[HTTP_MAX_HEADERS];

    u32 head_len; /* bytes through the blank line; body starts here          */
    u64 content_length;
    u32 body_off, body_len;

    /* HTTP/1.1 defaults to keep-alive and HTTP/1.0 does not; `Connection`
     * overrides either way. Resolved here because it is a fact about the
     * request and the caller should not re-derive a rule with two inputs. */
    i32 keep_alive;
} HttpRequest;

/* Parses `len` bytes at `buf`. On OK every field of `*out` is set; on ERROR
 * `*err` is set and `*out` holds nothing you may read; on NEED_MORE neither is
 * meaningful. `buf` may contain NUL bytes and is never treated as a C string.
 *
 * Thread-safe: no state outside the arguments. */
HttpParseStatus http_parse(const char* buf, size_t len, HttpRequest* out,
                           HttpError* err);

/* Case-insensitive lookup of the LAST header named `name` (ASCII only, which
 * is all a field name may legally contain). Returns a pointer into `buf` with
 * the length through `out_len`, or NULL when absent. The value has already had
 * leading and trailing optional whitespace stripped by the parser. */
const char* http_header(const HttpRequest* r, const char* buf, const char* name,
                        u32* out_len);

/* The standard reason phrase for `status` ("Not Found"), or "Error" for a code
 * this server never emits. Kept beside the parser so the status codes it
 * chooses and the phrases the server writes cannot drift apart. */
const char* http_status_text(i32 status);

#endif /* STOKE_HTTP_H */
