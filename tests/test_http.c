/* tests/test_http.c -- exercises src/http.c, the HTTP/1.1 request parser.
 *
 * ▶▶ THIS FILE LOADS NO MODEL AND OPENS NO SOCKET, deliberately. The M6 entry
 * note puts the parser at rung A2 precisely because it is the one piece of the
 * server that can be hardened for free: every input below is a byte string in
 * this file or a committed file under tests/data/http_corpus/, so the whole
 * hostile-input surface is testable in milliseconds with nothing warm.
 *
 * ▶ THE PARSER MUST NEVER fatal(). Everything here is untrusted network input;
 * a malformed request is ORDINARY EXPECTED INPUT, not one of the M6 failure
 * classes. It is neither a trapped load-time failure, nor API misuse by our
 * host, nor a broken internal invariant -- so it returns a status code and the
 * connection is answered. A parser that called fatal() would hand any client on
 * the network a remote process kill.
 *
 * All fixture paths are repo-root-relative; every test binary in this project
 * is built and run assuming CWD == repo root (tests/run_tests.sh never cd's).
 */
#include "common.h"

#include "http.h"

#include <string.h>

#include "test.h"

/* ---------------------------------------------------------------------------
 * Helpers. `S(lit)` passes a string literal with its length so an embedded NUL
 * in a hostile fixture is carried faithfully rather than truncated by strlen --
 * "the parser stops at a NUL" is one of the defects these tests exist to find,
 * so the harness must not commit it first.
 * ---------------------------------------------------------------------------
 */
#define S(lit_) (lit_), (sizeof(lit_) - 1)

static HttpParseStatus parse(const char* buf, size_t len, HttpRequest* r,
                             HttpError* e) {
    /* Poison both outputs first: a parser that writes nothing must not pass a
     * test by inheriting a zeroed struct that happens to read as correct. */
    memset(r, 0xAB, sizeof *r);
    e->status = -12345;
    e->reason = NULL;
    return http_parse(buf, len, r, e);
}

static int slice_is(const char* buf, u32 off, u32 len, const char* want) {
    return len == (u32)strlen(want) && memcmp(buf + off, want, len) == 0;
}

/* ===========================================================================
 * The request line.
 * ===========================================================================
 */

TEST(request_line_minimal_get) {
    const char* b = "GET /health HTTP/1.1\r\n\r\n";
    size_t      n = strlen(b);

    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, n, &r, &e), HTTP_PARSE_OK);
    ASSERT_TRUE(slice_is(b, r.method_off, r.method_len, "GET"));
    ASSERT_TRUE(slice_is(b, r.target_off, r.target_len, "/health"));
    ASSERT_EQ_I64(r.http_minor, 1);
    ASSERT_EQ_I64(r.n_headers, 0);
    ASSERT_EQ_I64(r.head_len, n);
    ASSERT_EQ_I64(r.content_length, 0);
    ASSERT_EQ_I64(r.body_len, 0);
}

/* The target splits at the FIRST '?'. A query string is not part of the path,
 * and routing that compared the whole target would fail to match /health the
 * moment a client appended one. */
TEST(request_line_splits_path_from_query) {
    const char* b = "GET /v1/models?limit=2&x=? HTTP/1.1\r\n\r\n";
    size_t      n = strlen(b);

    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, n, &r, &e), HTTP_PARSE_OK);
    ASSERT_TRUE(slice_is(b, r.path_off, r.path_len, "/v1/models"));
    ASSERT_TRUE(slice_is(b, r.query_off, r.query_len, "limit=2&x=?"));
}

TEST(request_line_no_query_gives_empty_query) {
    const char* b = "POST /v1/chat/completions HTTP/1.1\r\n\r\n";
    size_t      n = strlen(b);

    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, n, &r, &e), HTTP_PARSE_OK);
    ASSERT_TRUE(slice_is(b, r.method_off, r.method_len, "POST"));
    ASSERT_TRUE(slice_is(b, r.path_off, r.path_len, "/v1/chat/completions"));
    ASSERT_EQ_I64(r.query_len, 0);
}

TEST(request_line_http_1_0_is_accepted_and_recorded) {
    const char* b = "GET / HTTP/1.0\r\n\r\n";
    size_t      n = strlen(b);

    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, n, &r, &e), HTTP_PARSE_OK);
    ASSERT_EQ_I64(r.http_minor, 0);
}

/* ===========================================================================
 * Incompleteness. NEED_MORE is not an error and must never be confused with
 * one: a client whose first packet ends mid-header gets answered, not hung up
 * on. The split-inside-CRLF case is the one a hand-rolled scanner gets wrong.
 * ===========================================================================
 */

TEST(need_more_on_partial_request_line) {
    const char* b = "GET /heal";
    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &r, &e), HTTP_PARSE_NEED_MORE);
}

TEST(need_more_on_headers_without_blank_line) {
    const char* b = "GET / HTTP/1.1\r\nHost: x\r\n";
    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &r, &e), HTTP_PARSE_NEED_MORE);
}

TEST(need_more_when_split_between_cr_and_lf) {
    const char* b = "GET / HTTP/1.1\r\nHost: x\r\n\r";
    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &r, &e), HTTP_PARSE_NEED_MORE);
}

TEST(need_more_on_empty_buffer) {
    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse("", 0, &r, &e), HTTP_PARSE_NEED_MORE);
}

/* ===========================================================================
 * Request-line rejections. Each carries its own reason slug, and the slug is
 * asserted rather than merely the status: three different defects all answer
 * 400, so a test that checked only the number would pass while the parser
 * rejected the right request for the wrong cause.
 * ===========================================================================
 */

static void assert_rejected(const char* b, size_t n, int status,
                            const char* reason) {
    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, n, &r, &e), HTTP_PARSE_ERROR);
    ASSERT_EQ_I64(e.status, status);
    ASSERT_TRUE(e.reason != NULL);
    if (e.reason) {
        ASSERT_EQ_STR(e.reason, reason);
    }
}

TEST(reject_request_line_with_one_space) {
    assert_rejected(S("GET /health\r\n\r\n"), 400, "bad-request-line");
}

TEST(reject_empty_request_line) {
    assert_rejected(S("\r\nGET / HTTP/1.1\r\n\r\n"), 400, "bad-request-line");
}

TEST(reject_absolute_form_target) {
    /* Origin-form only. Absolute-form is what a client sends to a FORWARD
     * PROXY; accepting it at an origin server is how a request gets routed
     * somewhere its author did not intend. */
    assert_rejected(S("GET http://evil/ HTTP/1.1\r\n\r\n"), 400, "bad-target");
}

TEST(reject_unknown_protocol_name) {
    assert_rejected(S("GET / HTTPS/1.1\r\n\r\n"), 400, "bad-version");
}

TEST(reject_non_numeric_version) {
    assert_rejected(S("GET / HTTP/x.y\r\n\r\n"), 400, "bad-version");
}

/* Well-formed but unsupported major version is 505, not 400: the request is
 * legible, we simply do not speak it, and a client can act on the difference. */
TEST(reject_http_2_with_505) {
    assert_rejected(S("GET / HTTP/2.0\r\n\r\n"), 505, "http-version");
}

/* ===========================================================================
 * Headers.
 * ===========================================================================
 */

TEST(headers_are_recorded_with_ows_stripped) {
    const char* b = "GET / HTTP/1.1\r\n"
                    "Host: example.com\r\n"
                    "X-Trace:  \ta b \t \r\n"
                    "\r\n";
    size_t      n = strlen(b);

    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, n, &r, &e), HTTP_PARSE_OK);
    ASSERT_EQ_I64(r.n_headers, 2);
    ASSERT_TRUE(slice_is(b, r.headers[0].name_off, r.headers[0].name_len, "Host"));
    ASSERT_TRUE(slice_is(b, r.headers[0].value_off, r.headers[0].value_len,
                         "example.com"));
    ASSERT_TRUE(slice_is(b, r.headers[1].name_off, r.headers[1].name_len,
                         "X-Trace"));
    /* Leading and trailing OWS gone; the space INSIDE the value survives --
     * stripping that would silently corrupt any header carrying a phrase. */
    ASSERT_TRUE(slice_is(b, r.headers[1].value_off, r.headers[1].value_len,
                         "a b"));
    ASSERT_EQ_I64(r.head_len, n);
}

TEST(empty_header_value_is_legal) {
    const char* b = "GET / HTTP/1.1\r\nX-Empty:\r\nX-Ows:   \r\n\r\n";
    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &r, &e), HTTP_PARSE_OK);
    ASSERT_EQ_I64(r.n_headers, 2);
    ASSERT_EQ_I64(r.headers[0].value_len, 0);
    ASSERT_EQ_I64(r.headers[1].value_len, 0);
}

TEST(header_lookup_is_case_insensitive) {
    const char* b = "GET / HTTP/1.1\r\nContent-Type: application/json\r\n\r\n";
    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &r, &e), HTTP_PARSE_OK);

    u32         vlen = 0;
    const char* v    = http_header(&r, b, "content-type", &vlen);
    ASSERT_TRUE(v != NULL);
    if (v) {
        ASSERT_TRUE(vlen == strlen("application/json") &&
                    memcmp(v, "application/json", vlen) == 0);
    }
    ASSERT_TRUE(http_header(&r, b, "CONTENT-TYPE", &vlen) != NULL);
    ASSERT_TRUE(http_header(&r, b, "x-absent", &vlen) == NULL);
}

/* Last-wins, stated and tested rather than left to whichever way the loop
 * happens to run. It matters for exactly one header that reaches this server
 * (Connection), and a repeated Content-Length is rejected outright elsewhere
 * rather than resolved by this rule. */
TEST(header_lookup_returns_the_last_of_a_repeated_name) {
    const char* b = "GET / HTTP/1.1\r\nX-A: first\r\nX-A: second\r\n\r\n";
    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &r, &e), HTTP_PARSE_OK);

    u32         vlen = 0;
    const char* v    = http_header(&r, b, "x-a", &vlen);
    ASSERT_TRUE(v != NULL);
    if (v) {
        ASSERT_TRUE(vlen == 6 && memcmp(v, "second", 6) == 0);
    }
}

/* ===========================================================================
 * Content-Length bodies.
 * ===========================================================================
 */

TEST(body_is_complete_at_content_length) {
    const char* b = "POST /v1/chat/completions HTTP/1.1\r\n"
                    "Content-Length: 5\r\n"
                    "\r\n"
                    "HELLO";
    size_t      n = strlen(b);

    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, n, &r, &e), HTTP_PARSE_OK);
    ASSERT_EQ_I64(r.content_length, 5);
    ASSERT_EQ_I64(r.body_len, 5);
    ASSERT_TRUE(slice_is(b, r.body_off, r.body_len, "HELLO"));
    ASSERT_EQ_I64(r.head_len + r.body_len, n);
}

TEST(need_more_while_body_is_short) {
    const char* b = "POST /x HTTP/1.1\r\nContent-Length: 5\r\n\r\nHEL";
    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &r, &e), HTTP_PARSE_NEED_MORE);
}

/* A pipelined second request must not be swallowed into the first one's body.
 * head_len + body_len is the exact number of bytes to retire from the buffer. */
TEST(bytes_past_content_length_are_left_alone) {
    const char* b = "POST /x HTTP/1.1\r\nContent-Length: 5\r\n\r\nHELLOGET / HTTP/1.1\r\n\r\n";
    size_t      n = strlen(b);

    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, n, &r, &e), HTTP_PARSE_OK);
    ASSERT_EQ_I64(r.body_len, 5);
    ASSERT_TRUE(slice_is(b, r.body_off, r.body_len, "HELLO"));
    ASSERT_TRUE(r.head_len + r.body_len < n);
}

TEST(no_content_length_means_no_body) {
    const char* b = "GET / HTTP/1.1\r\nHost: x\r\n\r\n";
    size_t      n = strlen(b);

    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, n, &r, &e), HTTP_PARSE_OK);
    ASSERT_EQ_I64(r.content_length, 0);
    ASSERT_EQ_I64(r.body_len, 0);
    ASSERT_EQ_I64(r.head_len, n);
}

TEST(content_length_zero_is_a_complete_request) {
    const char* b = "POST /x HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &r, &e), HTTP_PARSE_OK);
    ASSERT_EQ_I64(r.body_len, 0);
}

/* ===========================================================================
 * Connection reuse. The default differs by version and `Connection` overrides
 * it in both directions, so all four combinations are asserted -- getting one
 * of them backwards leaks a socket or truncates a response, and neither shows
 * up in a single-request test.
 * ===========================================================================
 */

static i32 keep_alive_of(const char* b) {
    HttpRequest r;
    HttpError   e;
    if (parse(b, strlen(b), &r, &e) != HTTP_PARSE_OK) return -1;
    return r.keep_alive;
}

TEST(keep_alive_defaults_to_the_version) {
    ASSERT_EQ_I64(keep_alive_of("GET / HTTP/1.1\r\n\r\n"), 1);
    ASSERT_EQ_I64(keep_alive_of("GET / HTTP/1.0\r\n\r\n"), 0);
}

TEST(connection_header_overrides_both_defaults) {
    ASSERT_EQ_I64(keep_alive_of("GET / HTTP/1.1\r\nConnection: close\r\n\r\n"), 0);
    ASSERT_EQ_I64(
        keep_alive_of("GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n"), 1);
}

TEST(connection_value_is_matched_case_insensitively) {
    ASSERT_EQ_I64(keep_alive_of("GET / HTTP/1.1\r\nConnection: Close\r\n\r\n"), 0);
    ASSERT_EQ_I64(
        keep_alive_of("GET / HTTP/1.0\r\nConnection: Keep-Alive\r\n\r\n"), 1);
}

/* ===========================================================================
 * Reason phrases. They live beside the parser so a status the parser chooses
 * and a phrase the server writes cannot drift apart.
 * ===========================================================================
 */

TEST(status_text_covers_every_code_this_server_emits) {
    ASSERT_EQ_STR(http_status_text(200), "OK");
    ASSERT_EQ_STR(http_status_text(400), "Bad Request");
    ASSERT_EQ_STR(http_status_text(404), "Not Found");
    ASSERT_EQ_STR(http_status_text(405), "Method Not Allowed");
    ASSERT_EQ_STR(http_status_text(413), "Content Too Large");
    ASSERT_EQ_STR(http_status_text(431), "Request Header Fields Too Large");
    ASSERT_EQ_STR(http_status_text(500), "Internal Server Error");
    ASSERT_EQ_STR(http_status_text(501), "Not Implemented");
    ASSERT_EQ_STR(http_status_text(503), "Service Unavailable");
    ASSERT_EQ_STR(http_status_text(505), "HTTP Version Not Supported");
    ASSERT_EQ_STR(http_status_text(999), "Error");
}

/* ===========================================================================
 * Hostile heads. Everything below is a request a client can actually send, so
 * every one of them is answered rather than crashed on. The reason slugs are
 * asserted, not just the codes -- eleven of these are 400, and a test that
 * checked the number alone would accept a parser that rejected them all for
 * one wrong reason.
 * ===========================================================================
 */

TEST(reject_header_without_colon) {
    assert_rejected(S("GET / HTTP/1.1\r\nHostexample\r\n\r\n"), 400,
                    "header-no-colon");
}

TEST(reject_header_with_empty_name) {
    assert_rejected(S("GET / HTTP/1.1\r\n: value\r\n\r\n"), 400,
                    "header-empty-name");
}

/* RFC 7230 4.1.2 is explicit that this MUST be rejected rather than trimmed:
 * whitespace before the colon is a request-smuggling primitive, because an
 * intermediary that trims it and an origin that does not disagree about which
 * headers the message has. */
TEST(reject_space_before_colon) {
    assert_rejected(S("GET / HTTP/1.1\r\nHost : x\r\n\r\n"), 400,
                    "space-before-colon");
}

/* obs-fold: a header continued onto an indented following line. Deprecated,
 * and a recipient is permitted to reject it -- which is cheaper than being
 * right about it. */
TEST(reject_obs_fold) {
    assert_rejected(S("GET / HTTP/1.1\r\nHost: a\r\n  continued\r\n\r\n"), 400,
                    "obs-fold");
}

TEST(reject_bare_cr_in_a_header_value) {
    assert_rejected(S("GET / HTTP/1.1\r\nHost: a\rb\r\n\r\n"), 400, "bare-cr");
}

TEST(reject_nul_byte_in_the_head) {
    assert_rejected(S("GET / HTTP/1.1\r\nHost: a\0b\r\n\r\n"), 400,
                    "control-char");
}

TEST(reject_control_char_in_a_header_value) {
    assert_rejected(S("GET / HTTP/1.1\r\nHost: a\x01\x62\r\n\r\n"), 400,
                    "control-char");
}

TEST(reject_non_token_char_in_a_header_name) {
    assert_rejected(S("GET / HTTP/1.1\r\nHo(st): x\r\n\r\n"), 400,
                    "bad-header-name");
}

TEST(reject_control_char_in_the_method) {
    assert_rejected(S("GE\x01T / HTTP/1.1\r\n\r\n"), 400, "control-char");
}

TEST(reject_tab_in_the_method) {
    assert_rejected(S("GE\tT / HTTP/1.1\r\n\r\n"), 400, "bad-method");
}

TEST(reject_del_char_in_the_target) {
    assert_rejected(S("GET /a\x7f\x62 HTTP/1.1\r\n\r\n"), 400, "control-char");
}

/* 501, and LOUD, is the whole point: a request carrying Transfer-Encoding must
 * never be answered by quietly honouring its Content-Length instead. That
 * disagreement between what the sender framed and what we read is the CL.TE
 * desync. See http.h. */
TEST(reject_transfer_encoding_with_501) {
    assert_rejected(S("POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"),
                    501, "transfer-encoding");
}

TEST(reject_transfer_encoding_identity_too) {
    /* Not just `chunked`. Any framing directive we do not implement is a
     * refusal, because the danger is the disagreement, not the value. */
    assert_rejected(S("POST /x HTTP/1.1\r\nTransfer-Encoding: identity\r\n"
                      "Content-Length: 0\r\n\r\n"),
                    501, "transfer-encoding");
}

/* Repeated Content-Length is refused even when the two agree. Resolving it
 * "correctly" is a rule an intermediary might apply differently; refusing is a
 * rule nobody can disagree with. */
TEST(reject_duplicate_content_length_even_when_equal) {
    assert_rejected(S("POST /x HTTP/1.1\r\nContent-Length: 0\r\n"
                      "Content-Length: 0\r\n\r\n"),
                    400, "dup-content-length");
}

TEST(reject_non_numeric_content_length) {
    assert_rejected(S("POST /x HTTP/1.1\r\nContent-Length: 5x\r\n\r\n"), 400,
                    "bad-content-length");
}

TEST(reject_signed_content_length) {
    assert_rejected(S("POST /x HTTP/1.1\r\nContent-Length: +5\r\n\r\n"), 400,
                    "bad-content-length");
}

TEST(reject_empty_content_length) {
    assert_rejected(S("POST /x HTTP/1.1\r\nContent-Length:\r\n\r\n"), 400,
                    "bad-content-length");
}

/* A length no integer here can hold is malformed, not merely too big: 413
 * would be a claim about a number we could not read. */
TEST(reject_content_length_overflow) {
    assert_rejected(
        S("POST /x HTTP/1.1\r\nContent-Length: 999999999999999999999999\r\n\r\n"),
        400, "bad-content-length");
}

/* 413 the moment the headers are complete -- before a single body byte is
 * read, which is what makes a 10 GB declaration free to refuse. */
TEST(reject_body_over_the_cap_before_reading_it) {
    assert_rejected(S("POST /x HTTP/1.1\r\nContent-Length: 1048577\r\n\r\n"), 413,
                    "body-too-large");
}

TEST(accept_body_declaration_exactly_at_the_cap) {
    /* The other side of the same boundary. Declared, not sent: NEED_MORE means
     * the cap check passed and the parser is waiting for the bytes. */
    const char* b = "POST /x HTTP/1.1\r\nContent-Length: 1048576\r\n\r\n";
    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &r, &e), HTTP_PARSE_NEED_MORE);
}

/* ===========================================================================
 * The head budget, asserted on BOTH sides of both caps. An off-by-one in a
 * bound is the classic defect here and only a pair of tests can see it.
 * ===========================================================================
 */

static char g_big[HTTP_MAX_HEAD_BYTES * 2];

/* Builds "GET / HTTP/1.1CRLF" + one padded header + CRLF whose total head
 * length is exactly `want`. Returns the length written. */
static size_t build_head_of_length(size_t want) {
    const char* line = "GET / HTTP/1.1\r\n";
    size_t      n    = strlen(line);
    memcpy(g_big, line, n);
    size_t pad = want - n - strlen("X-Pad: ") - 2 - 2;
    memcpy(g_big + n, "X-Pad: ", strlen("X-Pad: "));
    n += strlen("X-Pad: ");
    memset(g_big + n, 'a', pad);
    n += pad;
    memcpy(g_big + n, "\r\n\r\n", 4);
    n += 4;
    return n;
}

TEST(head_of_exactly_the_cap_is_accepted) {
    size_t      n = build_head_of_length(HTTP_MAX_HEAD_BYTES);
    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(n, HTTP_MAX_HEAD_BYTES);
    ASSERT_EQ_I64(parse(g_big, n, &r, &e), HTTP_PARSE_OK);
    ASSERT_EQ_I64(r.head_len, HTTP_MAX_HEAD_BYTES);
}

TEST(head_one_byte_over_the_cap_is_431) {
    size_t n = build_head_of_length(HTTP_MAX_HEAD_BYTES + 1);
    ASSERT_EQ_I64(n, HTTP_MAX_HEAD_BYTES + 1);
    assert_rejected(g_big, n, 431, "head-too-large");
}

/* A client that never sends the blank line must be cut off at the cap rather
 * than kept on NEED_MORE forever. This is the slowloris shape. */
TEST(unterminated_head_past_the_cap_is_431) {
    const char* line = "GET / HTTP/1.1\r\n";
    size_t      n    = strlen(line);
    memcpy(g_big, line, n);
    memset(g_big + n, 'a', HTTP_MAX_HEAD_BYTES + 1);
    n += HTTP_MAX_HEAD_BYTES + 1;
    assert_rejected(g_big, n, 431, "head-too-large");
}

/* Builds a head carrying `count` short headers, well inside the byte budget,
 * so this pair isolates the COUNT cap from the BYTE cap. */
static size_t build_head_with_headers(size_t count) {
    const char* line = "GET / HTTP/1.1\r\n";
    size_t      n    = strlen(line);
    memcpy(g_big, line, n);
    for (size_t i = 0; i < count; i++) {
        char hdr[16];
        int  k = snprintf(hdr, sizeof hdr, "X-%02u: v\r\n", (unsigned)(i % 100));
        memcpy(g_big + n, hdr, (size_t)k);
        n += (size_t)k;
    }
    memcpy(g_big + n, "\r\n", 2);
    n += 2;
    return n;
}

TEST(exactly_max_headers_is_accepted) {
    size_t      n = build_head_with_headers(HTTP_MAX_HEADERS);
    HttpRequest r;
    HttpError   e;
    ASSERT_TRUE(n < HTTP_MAX_HEAD_BYTES); /* isolates count from bytes */
    ASSERT_EQ_I64(parse(g_big, n, &r, &e), HTTP_PARSE_OK);
    ASSERT_EQ_I64(r.n_headers, HTTP_MAX_HEADERS);
}

TEST(one_header_over_the_count_cap_is_431) {
    size_t n = build_head_with_headers(HTTP_MAX_HEADERS + 1);
    ASSERT_TRUE(n < HTTP_MAX_HEAD_BYTES);
    assert_rejected(g_big, n, 431, "too-many-headers");
}

/* ===========================================================================
 * Bare-LF line endings. Accepted deliberately -- see line_end() in http.c for
 * why that is safe HERE specifically and would not be in a proxy.
 * ===========================================================================
 */

TEST(bare_lf_line_endings_are_accepted) {
    const char* b = "POST /x HTTP/1.1\nContent-Length: 2\n\nhi";
    size_t      n = strlen(b);

    HttpRequest r;
    HttpError   e;
    ASSERT_EQ_I64(parse(b, n, &r, &e), HTTP_PARSE_OK);
    ASSERT_EQ_I64(r.n_headers, 1);
    ASSERT_TRUE(slice_is(b, r.body_off, r.body_len, "hi"));
    ASSERT_EQ_I64(r.head_len + r.body_len, n);
}

/* Connection is a comma-separated TOKEN LIST, not a single value. Browsers and
 * proxies send "keep-alive, Upgrade"; exact-matching the whole field misses the
 * "close" in "close, TE" and leaves us holding a socket the client has already
 * decided to drop. */

TEST(connection_close_inside_a_token_list_is_honoured) {
    ASSERT_EQ_I64(
        keep_alive_of("GET / HTTP/1.1\r\nConnection: keep-alive, close\r\n\r\n"),
        0);
    ASSERT_EQ_I64(keep_alive_of("GET / HTTP/1.1\r\nConnection: close, TE\r\n\r\n"),
                  0);
}

TEST(connection_keep_alive_inside_a_token_list_is_honoured) {
    ASSERT_EQ_I64(
        keep_alive_of("GET / HTTP/1.0\r\nConnection: Upgrade, Keep-Alive\r\n\r\n"),
        1);
}

/* close WINS over keep-alive when a confused client sends both: the safe
 * reading of an ambiguous instruction is the one that closes the socket. */
TEST(connection_close_wins_over_keep_alive_in_one_list) {
    ASSERT_EQ_I64(
        keep_alive_of("GET / HTTP/1.0\r\nConnection: keep-alive, close\r\n\r\n"),
        0);
}

TEST(connection_token_unknown_to_us_leaves_the_default) {
    ASSERT_EQ_I64(keep_alive_of("GET / HTTP/1.1\r\nConnection: Upgrade\r\n\r\n"),
                  1);
    ASSERT_EQ_I64(keep_alive_of("GET / HTTP/1.0\r\nConnection: Upgrade\r\n\r\n"),
                  0);
}

/* A token that merely CONTAINS one of ours is a different token. Substring
 * matching here would let "not-close" or "keep-alive-ish" flip the decision. */
TEST(connection_matching_is_by_token_not_by_substring) {
    ASSERT_EQ_I64(keep_alive_of("GET / HTTP/1.1\r\nConnection: notclose\r\n\r\n"),
                  1);
    ASSERT_EQ_I64(
        keep_alive_of("GET / HTTP/1.0\r\nConnection: keep-alive-ish\r\n\r\n"), 0);
}

/* ===========================================================================
 * M6 GATE ITEM [6] -- the committed garbage-input corpus.
 *
 * Every file in tests/data/http_corpus/ is fed to http_parse() as EXACTLY its
 * own bytes, and the expected outcome is carried in its FILENAME:
 *
 *     <expect>[.<reason-slug>].<note>.raw
 *
 * `<expect>` is `ok`, `more`, or the status code. For a status code the SECOND
 * field is the exact HttpError.reason the parser must give. That is not
 * decoration: eleven distinct defects here answer 400, and a corpus asserting
 * only the number would stay green against a parser that rejected every one of
 * them for one wrong cause. The filename is the expectation because a manifest
 * is a second thing to keep in sync, and a corpus whose expectations have
 * drifted from its files is worse than no corpus.
 *
 * ▶▶ THE BYTES ARE HANDED OVER IN A HEAP BUFFER OF EXACTLY THE FILE'S LENGTH,
 * never a padded stack array. A parser that reads one byte past the end is the
 * defect this corpus exists to find, and it is invisible in the middle of a
 * generous buffer -- the redzone has to be immediately after the last byte for
 * ASan to see the step off it.
 *
 * ▶ ANTI-VACUITY. A corpus driver that finds no files reports success at the
 * top of its loop, and a path bug is indistinguishable from a clean run. So the
 * count is asserted against a floor AND every expectation class must be
 * represented -- a glob that returned only the `ok` files would otherwise be a
 * green gate over a parser that rejects nothing.
 * ===========================================================================
 */

#include <dirent.h>
#include <stdlib.h>

#define CORPUS_DIR       "tests/data/http_corpus"
#define CORPUS_MIN_FILES 24

/* Reads the whole file into a buffer of EXACTLY its length -- see above. */
static char* read_all(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    char* buf = (char*)malloc((size_t)n ? (size_t)n : 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) {
        free(buf);
        return NULL;
    }
    *out_len = got;
    return buf;
}

/* Copies field `k` (dot-separated) of `name` into `out`. Returns 0 if absent. */
static int field(const char* name, int k, char* out, size_t cap) {
    const char* p = name;
    for (int i = 0; i < k; i++) {
        p = strchr(p, '.');
        if (!p) return 0;
        p++;
    }
    const char* e = strchr(p, '.');
    size_t      n = e ? (size_t)(e - p) : strlen(p);
    if (n + 1 > cap) return 0;
    memcpy(out, p, n);
    out[n] = '\0';
    return 1;
}

TEST(corpus_of_committed_garbage_parses_as_specified) {
    DIR* d = opendir(CORPUS_DIR);
    ASSERT_TRUE(d != NULL);
    if (!d) {
        fprintf(stderr, "  (corpus directory " CORPUS_DIR " is missing)\n");
        return;
    }

    int seen = 0;
    int seen_ok = 0, seen_more = 0, seen_400 = 0, seen_413 = 0, seen_431 = 0,
        seen_501 = 0, seen_505 = 0;

    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        const char* nm  = ent->d_name;
        size_t      nlen = strlen(nm);
        if (nlen < 5 || strcmp(nm + nlen - 4, ".raw") != 0) continue;

        char path[512];
        snprintf(path, sizeof path, "%s/%s", CORPUS_DIR, nm);

        size_t n   = 0;
        char*  buf = read_all(path, &n);
        ASSERT_TRUE(buf != NULL);
        if (!buf) continue;

        char expect[32];
        ASSERT_TRUE(field(nm, 0, expect, sizeof expect));

        HttpRequest r;
        HttpError   e;
        HttpParseStatus st = parse(buf, n, &r, &e);

        if (strcmp(expect, "ok") == 0) {
            seen_ok++;
            if (st != HTTP_PARSE_OK) {
                fprintf(stderr, "FAIL corpus %s: want OK, got %d (%d %s)\n", nm,
                        (int)st, e.status, e.reason ? e.reason : "(null)");
                g_test_case_failed = 1;
            } else {
                /* A parser that answered OK having written nothing would pass a
                 * bare status check. */
                ASSERT_TRUE(r.method_len > 0);
                ASSERT_TRUE(r.path_len > 0);
                ASSERT_TRUE((size_t)r.head_len + r.body_len <= n);
                ASSERT_TRUE(r.head_len <= HTTP_MAX_HEAD_BYTES);
            }
        } else if (strcmp(expect, "more") == 0) {
            seen_more++;
            if (st != HTTP_PARSE_NEED_MORE) {
                fprintf(stderr, "FAIL corpus %s: want NEED_MORE, got %d (%d %s)\n",
                        nm, (int)st, e.status, e.reason ? e.reason : "(null)");
                g_test_case_failed = 1;
            }
        } else {
            int  want_status = atoi(expect);
            char want_reason[64];
            ASSERT_TRUE(field(nm, 1, want_reason, sizeof want_reason));

            if (st != HTTP_PARSE_ERROR || e.status != want_status ||
                e.reason == NULL || strcmp(e.reason, want_reason) != 0) {
                fprintf(stderr,
                        "FAIL corpus %s: want ERROR %d %s, got st=%d %d %s\n", nm,
                        want_status, want_reason, (int)st, e.status,
                        e.reason ? e.reason : "(null)");
                g_test_case_failed = 1;
            }
            switch (want_status) {
                case 400: seen_400++; break;
                case 413: seen_413++; break;
                case 431: seen_431++; break;
                case 501: seen_501++; break;
                case 505: seen_505++; break;
                default:
                    fprintf(stderr, "FAIL corpus %s: status %d is not one this "
                                    "parser emits\n",
                            nm, want_status);
                    g_test_case_failed = 1;
            }
        }
        free(buf);
        seen++;
    }
    closedir(d);

    /* The guards. Each one is a way this test could report green having
     * measured nothing. */
    if (seen < CORPUS_MIN_FILES) {
        fprintf(stderr,
                "FAIL corpus: only %d files found in " CORPUS_DIR
                " (floor is %d) -- a corpus driver that finds nothing passes\n",
                seen, CORPUS_MIN_FILES);
        g_test_case_failed = 1;
    }
    if (!seen_ok || !seen_more || !seen_400 || !seen_413 || !seen_431 ||
        !seen_501 || !seen_505) {
        fprintf(stderr,
                "FAIL corpus: an expectation class is unrepresented "
                "(ok=%d more=%d 400=%d 413=%d 431=%d 501=%d 505=%d)\n",
                seen_ok, seen_more, seen_400, seen_413, seen_431, seen_501,
                seen_505);
        g_test_case_failed = 1;
    }
    printf("  corpus: %d files (ok=%d more=%d 400=%d 413=%d 431=%d 501=%d "
           "505=%d)\n",
           seen, seen_ok, seen_more, seen_400, seen_413, seen_431, seen_501,
           seen_505);
}

/* ===========================================================================
 * The fuzz-ish half. The corpus above is a SPECIFICATION -- every file states
 * what the parser must answer. This sweep is the opposite and is honest about
 * it: nobody can predict the right answer for a random byte string, so it
 * asserts only properties that must hold for EVERY input, and those properties
 * are what make it more than "it did not crash".
 *
 * Deterministic (splitmix64 from a fixed seed), so a failure is replayable from
 * the iteration number printed with it.
 * ===========================================================================
 */

static u64 sm64(u64* s) {
    u64 z = (*s += 0x9E3779B97F4A7C15ull);
    z     = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z     = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

TEST(random_and_mutated_inputs_always_answer_consistently) {
    static const char tmpl[] =
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost:8080\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: 17\r\n"
        "\r\n"
        "{\"model\":\"qwen\"}\n";
    const size_t tmpl_len = sizeof(tmpl) - 1;

    u64 s = 0xC0FFEEull;
    /* Sized so a mutation can push the head past its cap and exercise the 431
     * paths, not merely the short ones. */
    static char in[HTTP_MAX_HEAD_BYTES + 512];

    const int iters = 20000;
    for (int it = 0; it < iters; it++) {
        size_t n;
        if (sm64(&s) & 1) {
            /* Pure garbage. Reaches the early rejections. */
            n = (size_t)(sm64(&s) % 600);
            for (size_t j = 0; j < n; j++) in[j] = (char)(sm64(&s) & 0xFF);
        } else {
            /* A VALID request with a few bytes damaged. This is what reaches
             * deep code: pure random almost never forms a request line, so a
             * sweep made only of it would test the first twenty lines of the
             * parser forever. */
            n = tmpl_len;
            memcpy(in, tmpl, n);
            if (sm64(&s) % 4 == 0) { /* sometimes truncate */
                n = (size_t)(sm64(&s) % (tmpl_len + 1));
            }
            if (sm64(&s) % 8 == 0 && n > 0) { /* sometimes inflate a header */
                size_t grow = (size_t)(sm64(&s) % 900);
                size_t at   = 40 < n ? 40 : n;
                if (at + grow < sizeof in) {
                    memmove(in + at + grow, in + at, n - at);
                    memset(in + at, 'a', grow);
                    n += grow;
                }
            }
            int muts = (int)(sm64(&s) % 4) + 1;
            for (int m = 0; m < muts && n > 0; m++) {
                in[sm64(&s) % n] = (char)(sm64(&s) & 0xFF);
            }
        }

        /* Copied into a buffer of EXACTLY n bytes for the same redzone reason
         * as the corpus above. */
        char* exact = (char*)malloc(n ? n : 1);
        ASSERT_TRUE(exact != NULL);
        if (!exact) return;
        memcpy(exact, in, n);

        HttpRequest r;
        HttpError   e;
        HttpParseStatus st = parse(exact, n, &r, &e);

        int bad = 0;
        if (st == HTTP_PARSE_ERROR) {
            bad = (e.reason == NULL) ||
                  !(e.status == 400 || e.status == 413 || e.status == 431 ||
                    e.status == 501 || e.status == 505);
        } else if (st == HTTP_PARSE_OK) {
            /* Every slice must lie inside the buffer it points into, and the
             * head must respect its own cap. An offset bug that survives a
             * hand-written test shows up here as an out-of-range slice long
             * before it shows up as a crash. */
            bad = r.head_len > n || (size_t)r.head_len + r.body_len > n ||
                  r.head_len > HTTP_MAX_HEAD_BYTES ||
                  r.n_headers > HTTP_MAX_HEADERS ||
                  (size_t)r.method_off + r.method_len > n ||
                  (size_t)r.target_off + r.target_len > n ||
                  (size_t)r.path_off + r.path_len > n ||
                  (size_t)r.query_off + r.query_len > n ||
                  r.content_length != r.body_len;
            for (u32 k = 0; !bad && k < r.n_headers; k++) {
                bad = (size_t)r.headers[k].name_off + r.headers[k].name_len > n ||
                      (size_t)r.headers[k].value_off + r.headers[k].value_len > n;
                /* A name the parser ACCEPTED must be all-tchar by its own rule.
                 * The bounds check above cannot see a slice that is merely one
                 * byte too long, because an accepted head always has a blank
                 * line after it -- so the over-read stays inside the buffer and
                 * is invisible to a range test. This is the invariant that
                 * sees it. */
                for (u32 j = 0; !bad && j < r.headers[k].name_len; j++) {
                    unsigned char c =
                        (unsigned char)exact[r.headers[k].name_off + j];
                    bad = (c == ':' || c == ' ' || c == '\t' || c < 0x20 ||
                           c >= 0x7F);
                }
            }
            /* Inside a head the parser accepted, every CR is part of a CRLF.
             * A lone one means it let a bare CR through. */
            for (u32 j = 0; !bad && j < r.head_len; j++) {
                if (exact[j] == '\r') {
                    bad = (j + 1 >= r.head_len) || exact[j + 1] != '\n';
                }
            }
        } else if (st != HTTP_PARSE_NEED_MORE) {
            bad = 1;
        }

        if (bad) {
            fprintf(stderr, "FAIL sweep iteration %d: st=%d status=%d n=%zu\n",
                    it, (int)st, e.status, n);
            g_test_case_failed = 1;
            free(exact);
            return;
        }
        free(exact);
    }
    printf("  sweep: %d inputs, no inconsistent answer\n", iters);
}
