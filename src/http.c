/* src/http.c -- see http.h for the contract and for why it is shaped this way.
 *
 * The whole file is one downward pass over the buffer with no state kept
 * between calls, no allocation and no fatal(). Read http.h first; the reasons
 * live there.
 */
#include "common.h"

#include "http.h"

#include <string.h>

/* ---- deliberate defects, and why they live in the shipping file ----------
 *
 * Each block below breaks ONE claim the committed corpus rests on, selected by
 * a -D at build time into its own BUILDDIR. They are here rather than in an
 * edited copy of this file for the reason the M5 mutants are: a mutant built
 * from a tree that no longer exists cannot be re-run, and this project has
 * already paid for that once (2026-08-25). Built from the shipping source, they
 * stay valid as it changes -- and a mutation that stops compiling is itself a
 * finding about the code having moved underneath the gate.
 *
 * ⚠ EVERY MUTANT ANNOUNCES ITSELF on stderr the first time it parses anything.
 * These binaries are wrong on purpose and must never be shipped or measured.
 */
#if defined(STOKE_MUT_HTTP_BARE_CR)
#define STOKE_MUT_HTTP_NAME "BARE_CR (a lone CR is no longer rejected)"
#elif defined(STOKE_MUT_HTTP_TE)
#define STOKE_MUT_HTTP_NAME "TE (Transfer-Encoding silently ignored)"
#elif defined(STOKE_MUT_HTTP_DUP_CL)
#define STOKE_MUT_HTTP_NAME "DUP_CL (repeated Content-Length: last wins)"
#elif defined(STOKE_MUT_HTTP_HEAD_CAP)
#define STOKE_MUT_HTTP_NAME "HEAD_CAP (the head byte budget is not enforced)"
#elif defined(STOKE_MUT_HTTP_SPACE_COLON)
#define STOKE_MUT_HTTP_NAME "SPACE_COLON (right status, wrong reason)"
#elif defined(STOKE_MUT_HTTP_BODY_CAP)
#define STOKE_MUT_HTTP_NAME "BODY_CAP (a declared body is never refused)"
#elif defined(STOKE_MUT_HTTP_NAME_LEN)
#define STOKE_MUT_HTTP_NAME "NAME_LEN (header names are one byte too long)"
#endif

#ifdef STOKE_MUT_HTTP_NAME
static void stoke_mut_http_banner(void) {
    static int announced = 0;
    if (!announced) {
        announced = 1;
        fprintf(stderr, "*** MUTANT src/http.c: %s -- NEVER SHIP OR MEASURE "
                        "WITH THIS BINARY ***\n",
                STOKE_MUT_HTTP_NAME);
    }
}
#endif

/* ---- small helpers ------------------------------------------------------ */

static HttpParseStatus fail(HttpError* err, i32 status, const char* reason) {
    err->status = status;
    err->reason = reason;
    return HTTP_PARSE_ERROR;
}

static int is_digit(char c) { return c >= '0' && c <= '9'; }

/* RFC 7230's `tchar` -- the only bytes legal in a method or a field name.
 * Written as an explicit set rather than "printable minus separators" because
 * the separators are the whole point: a ':' or a space smuggled into a field
 * name is how two parsers end up disagreeing about where a header begins. */
static int is_tchar(unsigned char c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9')) {
        return 1;
    }
    return c == '!' || c == '#' || c == '$' || c == '%' || c == '&' ||
           c == '\'' || c == '*' || c == '+' || c == '-' || c == '.' ||
           c == '^' || c == '_' || c == '`' || c == '|' || c == '~';
}

static char lower(char c) { return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c; }

/* ASCII case-insensitive compare of a (ptr,len) slice against a NUL-terminated
 * name. ASCII is not a simplification here -- a field name may contain nothing
 * else, so a locale-aware fold would be answering a question nobody asked. */
static int ci_eq(const char* a, size_t alen, const char* b) {
    size_t i = 0;
    for (; i < alen; i++) {
        if (b[i] == '\0' || lower(a[i]) != lower(b[i])) return 0;
    }
    return b[i] == '\0';
}

/* Finds the terminator of the line starting at `i`.
 *
 * On success sets *end to the index one past the line's last CONTENT byte and
 * *next to the index of the following line, then returns 1. Returns 0 when no
 * terminator is present yet, which is NEED_MORE and not an error.
 *
 * ▶ A BARE LF IS ACCEPTED as a terminator alongside CRLF, and that is a
 * deliberate, bounded leniency. It is dangerous in a PROXY, where a lenient
 * origin behind a strict front end (or the reverse) is exactly the request-
 * smuggling desync -- but this parser refuses `Transfer-Encoding` outright
 * (http.h) and is the origin, so the desync it enables has nowhere to happen.
 * A bare CR is a different matter and is rejected by line_chars_ok().
 */
static int line_end(const char* b, size_t len, size_t i, size_t* end,
                    size_t* next) {
    for (size_t j = i; j < len; j++) {
        if (b[j] == '\n') {
            *next = j + 1;
            *end  = (j > i && b[j - 1] == '\r') ? j - 1 : j;
            return 1;
        }
    }
    return 0;
}

/* Every byte of every head line passes through here before anything tries to
 * interpret it, so no later rule has to re-ask whether a control character
 * slipped in. Returns 1 when clean, 0 with `err` set otherwise.
 *
 * A bare CR gets its own slug because it means something specific: someone is
 * probing for a parser that treats CR alone as a line break. */
static int line_chars_ok(const char* b, size_t i, size_t end, HttpError* err) {
    for (size_t j = i; j < end; j++) {
        unsigned char c = (unsigned char)b[j];
#ifdef STOKE_MUT_HTTP_BARE_CR
        /* ⚠ THE MUTANT SKIPS THE WHOLE ITERATION, not just the bare-cr branch.
         * The first version of it removed only the branch below -- and caught
         * nothing worth catching, because 0x0D is also `< 0x20 && != TAB` and
         * the control-char check underneath rejected it anyway. All that mutant
         * proved was that the two REASONS are distinguished, while the property
         * it claimed to test (a bare CR is never ACCEPTED) went untested and the
         * leg read green. A mutation masked by a second check that happens to
         * cover the same byte is a mutation that graded nothing. */
        if (c == '\r') continue;
#else
        if (c == '\r') {
            fail(err, 400, "bare-cr");
            return 0;
        }
#endif
        if ((c < 0x20 && c != '\t') || c == 0x7F) {
            fail(err, 400, "control-char");
            return 0;
        }
    }
    return 1;
}

/* ---- the request line ---------------------------------------------------
 *
 *     method SP request-target SP HTTP-version
 *
 * Exactly two spaces, both load-bearing. `end` is one past the last content
 * byte, so the terminator is already stripped.
 */
static HttpParseStatus parse_request_line(const char* b, size_t end,
                                          HttpRequest* r, HttpError* err) {
    if (end == 0) {
        /* RFC 7230 permits a server to skip a leading empty line; this one
         * refuses instead. Nothing legitimate sends it, and "skip some junk
         * before the request" is a capability with no caller and a history. */
        return fail(err, 400, "bad-request-line");
    }

    size_t sp1 = 0;
    while (sp1 < end && b[sp1] != ' ') sp1++;
    if (sp1 == 0 || sp1 >= end) return fail(err, 400, "bad-request-line");

    size_t sp2 = sp1 + 1;
    while (sp2 < end && b[sp2] != ' ') sp2++;
    if (sp2 == sp1 + 1 || sp2 >= end) return fail(err, 400, "bad-request-line");

    for (size_t j = 0; j < sp1; j++) {
        if (!is_tchar((unsigned char)b[j])) return fail(err, 400, "bad-method");
    }
    r->method_off = 0;
    r->method_len = (u32)sp1;

    size_t t_off = sp1 + 1;
    size_t t_len = sp2 - t_off;

    /* Origin-form only -- see http.h on absolute-form. */
    if (b[t_off] != '/') return fail(err, 400, "bad-target");

    r->target_off = (u32)t_off;
    r->target_len = (u32)t_len;

    /* Split at the FIRST '?': everything after it is query, including any
     * further '?' characters, which are legal there. */
    size_t q = t_off;
    while (q < t_off + t_len && b[q] != '?') q++;
    r->path_off  = (u32)t_off;
    r->path_len  = (u32)(q - t_off);
    r->query_off = (u32)(q < t_off + t_len ? q + 1 : q);
    r->query_len = (u32)(q < t_off + t_len ? (t_off + t_len) - (q + 1) : 0);

    const char* v     = b + sp2 + 1;
    size_t      v_len = end - (sp2 + 1);
    if (v_len != 8 || memcmp(v, "HTTP/", 5) != 0 || !is_digit(v[5]) ||
        v[6] != '.' || !is_digit(v[7])) {
        return fail(err, 400, "bad-version");
    }

    int major = v[5] - '0';
    int minor = v[7] - '0';
    /* Well-formed but not ours. 505 rather than 400 is the difference between
     * "I cannot read this" and "I read it and do not speak it" -- a client can
     * act on the second. */
    if (major != 1 || minor > 1) return fail(err, 505, "http-version");

    r->http_minor = minor;
    return HTTP_PARSE_OK;
}

/* ---- one header line ----------------------------------------------------
 *
 * Rule order is load-bearing and is the reverse of how it reads: the SPECIFIC
 * diagnosis has to be reached before the general one, or every malformed
 * header reports the last rule that happened to notice.
 */
static HttpParseStatus parse_header_line(const char* b, size_t i, size_t end,
                                         HttpRequest* r, HttpError* err) {
    if (b[i] == ' ' || b[i] == '\t') return fail(err, 400, "obs-fold");

    size_t c = i;
    while (c < end && b[c] != ':') c++;
    if (c >= end) return fail(err, 400, "header-no-colon");
    if (c == i) return fail(err, 400, "header-empty-name");

    for (size_t j = i; j < c; j++) {
#ifndef STOKE_MUT_HTTP_SPACE_COLON
        if (b[j] == ' ' || b[j] == '\t') {
            return fail(err, 400, "space-before-colon");
        }
#endif
        if (!is_tchar((unsigned char)b[j])) {
            return fail(err, 400, "bad-header-name");
        }
    }

    /* OWS is stripped from both ends of the value and nowhere else: a space
     * INSIDE a value is content, and trimming it would corrupt any header
     * carrying a phrase. */
    size_t vs = c + 1;
    while (vs < end && (b[vs] == ' ' || b[vs] == '\t')) vs++;
    size_t ve = end;
    while (ve > vs && (b[ve - 1] == ' ' || b[ve - 1] == '\t')) ve--;

    if (r->n_headers >= HTTP_MAX_HEADERS) {
        return fail(err, 431, "too-many-headers");
    }
    HttpHeader* h = &r->headers[r->n_headers++];
    h->name_off   = (u32)i;
#ifdef STOKE_MUT_HTTP_NAME_LEN
    h->name_len   = (u32)(c - i) + 1;
#else
    h->name_len   = (u32)(c - i);
#endif
    h->value_off  = (u32)vs;
    h->value_len  = (u32)(ve - vs);
    return HTTP_PARSE_OK;
}

/* `Connection` is a comma-separated TOKEN LIST, not one value: real clients
 * send "keep-alive, Upgrade". Matching the whole field would miss the `close`
 * in "close, TE" and leave us holding a socket the peer has already decided to
 * drop.
 *
 * Two rules, both deliberate. Matching is per TOKEN, so "notclose" is not
 * `close` -- a substring test here would let an unrelated extension token flip
 * the decision. And `close` WINS over `keep-alive` when a confused client sends
 * both, because the safe reading of an ambiguous instruction is the one that
 * closes the socket. An unknown token changes nothing and the version default
 * stands. */
static void apply_connection(const char* v, u32 vlen, i32* keep_alive) {
    int    saw_close = 0, saw_keep = 0;
    size_t i = 0;
    while (i < vlen) {
        while (i < vlen && (v[i] == ' ' || v[i] == '\t' || v[i] == ',')) i++;
        size_t s = i;
        while (i < vlen && v[i] != ',') i++;
        size_t e = i;
        while (e > s && (v[e - 1] == ' ' || v[e - 1] == '\t')) e--;
        if (e > s) {
            if (ci_eq(v + s, e - s, "close")) {
                saw_close = 1;
            } else if (ci_eq(v + s, e - s, "keep-alive")) {
                saw_keep = 1;
            }
        }
    }
    if (saw_close) {
        *keep_alive = 0;
    } else if (saw_keep) {
        *keep_alive = 1;
    }
}

/* ---- the entry point ---------------------------------------------------- */

HttpParseStatus http_parse(const char* buf, size_t len, HttpRequest* out,
                           HttpError* err) {
#ifdef STOKE_MUT_HTTP_NAME
    stoke_mut_http_banner();
#endif
    err->status = 0;
    err->reason = NULL;

    size_t end, next;

    /* The head budget is enforced in two places because it can be exceeded two
     * ways: a line that ENDS past the cap, and a client that sends past the cap
     * without ever ending a line. Only the second is slowloris, and only the
     * `len >` guard can see it -- without it that client waits forever on
     * NEED_MORE holding a connection. */
    if (!line_end(buf, len, 0, &end, &next)) {
        return len > HTTP_MAX_HEAD_BYTES ? fail(err, 431, "head-too-large")
                                         : HTTP_PARSE_NEED_MORE;
    }
#ifndef STOKE_MUT_HTTP_HEAD_CAP
    if (next > HTTP_MAX_HEAD_BYTES) return fail(err, 431, "head-too-large");
#endif
    if (!line_chars_ok(buf, 0, end, err)) return HTTP_PARSE_ERROR;

    HttpParseStatus st = parse_request_line(buf, end, out, err);
    if (st != HTTP_PARSE_OK) return st;

    out->n_headers = 0;
    size_t i       = next;
    for (;;) {
        if (!line_end(buf, len, i, &end, &next)) {
            return len > HTTP_MAX_HEAD_BYTES ? fail(err, 431, "head-too-large")
                                             : HTTP_PARSE_NEED_MORE;
        }
#ifndef STOKE_MUT_HTTP_HEAD_CAP
        if (next > HTTP_MAX_HEAD_BYTES) return fail(err, 431, "head-too-large");
#endif
        if (end == i) break; /* the blank line: head over */
        if (!line_chars_ok(buf, i, end, err)) return HTTP_PARSE_ERROR;
        st = parse_header_line(buf, i, end, out, err);
        if (st != HTTP_PARSE_OK) return st;
        i = next;
    }
    out->head_len = (u32)next;

    /* ---- framing, in the one order that is safe -------------------------
     *
     * Transfer-Encoding is decided FIRST and by its own pass, ahead of anything
     * Content-Length says. The hazard is not chunked decoding we lack; it is
     * answering a request framed one way by reading it another. Refusing 501
     * before Content-Length is even looked at makes that disagreement
     * unreachable rather than merely unlikely. */
#ifndef STOKE_MUT_HTTP_TE
    for (u32 k = 0; k < out->n_headers; k++) {
        const HttpHeader* h = &out->headers[k];
        if (ci_eq(buf + h->name_off, h->name_len, "transfer-encoding")) {
            return fail(err, 501, "transfer-encoding");
        }
    }
#endif

    u64 cl      = 0;
    u32 seen_cl = 0;
    for (u32 k = 0; k < out->n_headers; k++) {
        const HttpHeader* h = &out->headers[k];
        if (!ci_eq(buf + h->name_off, h->name_len, "content-length")) continue;

        /* Repeated Content-Length is refused even when the values agree.
         * Reconciling them is a rule another hop might apply differently;
         * refusing is a rule nobody can disagree with. */
#ifdef STOKE_MUT_HTTP_DUP_CL
        seen_cl++;
#else
        if (++seen_cl > 1) return fail(err, 400, "dup-content-length");
#endif
        if (h->value_len == 0) return fail(err, 400, "bad-content-length");

        cl = 0;
        for (u32 j = 0; j < h->value_len; j++) {
            char d = buf[h->value_off + j];
            if (!is_digit(d)) return fail(err, 400, "bad-content-length");
            /* Overflow is 400, not 413: 413 would be a claim about a size, and
             * we could not read this one. */
            u64 dig = (u64)(d - '0');
            if (cl > (UINT64_MAX - dig) / 10u) {
                return fail(err, 400, "bad-content-length");
            }
            cl = cl * 10u + dig;
        }
    }

    /* Decided the instant the head is complete, so a client declaring 10 GB is
     * refused having sent none of it. */
#ifndef STOKE_MUT_HTTP_BODY_CAP
    if (cl > HTTP_MAX_BODY_BYTES) return fail(err, 413, "body-too-large");
#endif

    out->content_length = cl;
    if (len - out->head_len < cl) return HTTP_PARSE_NEED_MORE;

    out->body_off = out->head_len;
    out->body_len = (u32)cl;

    out->keep_alive  = out->http_minor >= 1;
    u32         vlen = 0;
    const char* conn = http_header(out, buf, "connection", &vlen);
    if (conn) apply_connection(conn, vlen, &out->keep_alive);
    return HTTP_PARSE_OK;
}

const char* http_header(const HttpRequest* r, const char* buf, const char* name,
                        u32* out_len) {
    /* Backwards, so a repeated name resolves last-wins. Stated in http.h and
     * tested by name rather than left to whichever way this loop happened to
     * run. */
    for (i32 k = (i32)r->n_headers - 1; k >= 0; k--) {
        const HttpHeader* h = &r->headers[k];
        if (ci_eq(buf + h->name_off, h->name_len, name)) {
            if (out_len) *out_len = h->value_len;
            return buf + h->value_off;
        }
    }
    return NULL;
}

const char* http_status_text(i32 status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Content Too Large";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        case 505: return "HTTP Version Not Supported";
        default:  return "Error";
    }
}
