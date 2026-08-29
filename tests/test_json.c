/* tests/test_json.c -- exercises src/json.c, the JSON reader. M6 rung A3.
 *
 * ▶▶ WHY A PARSER AT ALL, WHEN THE SERVER ONLY WANTS THREE FIELDS. Because
 * "only three fields" is how you end up scanning for the literal bytes
 * `"content":` and getting it wrong on the first request whose PROMPT contains
 * that string, or whose messages nest, or whose quotes are escaped. A chat
 * completions body is attacker-controlled from end to end; the cheapest correct
 * thing is a real parser with real bounds.
 *
 * Same discipline as src/http.c and for the same reasons: no allocation, no
 * fatal(), results are offsets into the caller's buffer, and every rejection
 * carries a kebab-case reason slug that the tests assert BY NAME -- a dozen
 * different malformed documents are all "invalid JSON", and a test checking only
 * that a parse failed would accept a parser that failed for the wrong cause.
 */
#include "common.h"

#include "json.h"

#include <string.h>

#include "test.h"

#define S(lit_) (lit_), (sizeof(lit_) - 1)

static int parse(const char* b, size_t n, JsonDoc* d, JsonError* e) {
    memset(d, 0xAB, sizeof *d);
    e->reason = NULL;
    e->offset = (u32)-1;
    return json_parse(b, n, d, e);
}

/* NULL-tolerant accessors. A test that SEGFAULTS where it means to fail tells
 * you nothing about the other forty: these keep every assertion reportable when
 * a lookup legitimately returns NULL. */
static i64 ntype(const JsonNode* n) { return n ? n->type : -1; }
static i64 nkids(const JsonNode* n) { return n ? (i64)n->n_children : -1; }

static void assert_bad(const char* b, size_t n, const char* reason) {
    JsonDoc   d;
    JsonError e;
    ASSERT_TRUE(parse(b, n, &d, &e) != 0);
    ASSERT_TRUE(e.reason != NULL);
    if (e.reason) ASSERT_EQ_STR(e.reason, reason);
}

/* Reads a string member into `out`; returns its length or a negative error. */
static i64 member_str(const JsonDoc* d, const char* b, const JsonNode* obj,
                      const char* key, char* out, size_t cap) {
    const JsonNode* m = json_member(d, b, obj, key);
    if (!m) return -1;
    return json_string(b, m, out, cap);
}

/* ===========================================================================
 * Shape.
 * ===========================================================================
 */

TEST(flat_object_members_are_found_by_key) {
    const char* b = "{\"model\":\"qwen\",\"max_tokens\":16,\"stream\":false}";
    JsonDoc     d;
    JsonError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &d, &e), 0);

    const JsonNode* root = json_root(&d);
    ASSERT_TRUE(root != NULL);
    ASSERT_EQ_I64(ntype(root), JSON_OBJ);
    ASSERT_EQ_I64(nkids(root), 3);

    char s[32];
    ASSERT_EQ_I64(member_str(&d, b, root, "model", s, sizeof s), 4);
    ASSERT_TRUE(memcmp(s, "qwen", 4) == 0);

    double num = 0;
    ASSERT_EQ_I64(json_number(b, json_member(&d, b, root, "max_tokens"), &num), 0);
    ASSERT_EQ_I64((i64)num, 16);

    int bit = -1;
    ASSERT_EQ_I64(json_bool(b, json_member(&d, b, root, "stream"), &bit), 0);
    ASSERT_EQ_I64(bit, 0);

    /* An absent key is NULL, not a zero node -- the difference is whether the
     * server applies a default or reads a field that was never sent. */
    ASSERT_TRUE(json_member(&d, b, root, "temperature") == NULL);
}

TEST(nested_arrays_and_objects_walk_correctly) {
    const char* b =
        "{\"messages\":[{\"role\":\"system\",\"content\":\"a\"},"
        "{\"role\":\"user\",\"content\":\"b\"}]}";
    JsonDoc   d;
    JsonError e;
    ASSERT_EQ_I64(parse(b, strlen(b), &d, &e), 0);

    const JsonNode* msgs = json_member(&d, b, json_root(&d), "messages");
    ASSERT_TRUE(msgs != NULL);
    if (!msgs) return;
    ASSERT_EQ_I64(ntype(msgs), JSON_ARR);
    ASSERT_EQ_I64(nkids(msgs), 2);

    char s[32];
    ASSERT_EQ_I64(member_str(&d, b, json_at(&d, msgs, 0), "role", s, sizeof s), 6);
    ASSERT_TRUE(memcmp(s, "system", 6) == 0);
    ASSERT_EQ_I64(member_str(&d, b, json_at(&d, msgs, 1), "content", s, sizeof s), 1);
    ASSERT_TRUE(s[0] == 'b');

    ASSERT_TRUE(json_at(&d, msgs, 2) == NULL); /* past the end */
}

TEST(empty_object_and_array_are_legal) {
    JsonDoc   d;
    JsonError e;
    ASSERT_EQ_I64(parse(S("{}"), &d, &e), 0);
    ASSERT_EQ_I64(nkids(json_root(&d)), 0);
    ASSERT_EQ_I64(parse(S("[]"), &d, &e), 0);
    ASSERT_EQ_I64(ntype(json_root(&d)), JSON_ARR);
    ASSERT_EQ_I64(parse(S("{\"a\":{},\"b\":[]}"), &d, &e), 0);
    ASSERT_EQ_I64(nkids(json_root(&d)), 2);
}

/* Whitespace between every token, which is where a hand-rolled scanner that
 * skips it in only some positions comes apart. */
TEST(whitespace_is_allowed_everywhere_it_is_legal) {
    const char* b = " \t\r\n { \"a\" : [ 1 , 2 ] , \"b\" : null } \n ";
    JsonDoc     d;
    JsonError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &d, &e), 0);
    ASSERT_EQ_I64(nkids(json_root(&d)), 2);
    ASSERT_EQ_I64(nkids(json_member(&d, b, json_root(&d), "a")), 2);
    ASSERT_EQ_I64(ntype(json_member(&d, b, json_root(&d), "b")), JSON_NULL);
}

/* Last-wins, asserted rather than left to the loop. A client that sends the
 * same key twice must get one deterministic answer. */
TEST(a_repeated_key_resolves_to_the_last_one) {
    const char* b = "{\"n\":1,\"n\":2}";
    JsonDoc     d;
    JsonError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &d, &e), 0);
    double num = 0;
    ASSERT_EQ_I64(json_number(b, json_member(&d, b, json_root(&d), "n"), &num), 0);
    ASSERT_EQ_I64((i64)num, 2);
}

/* ===========================================================================
 * Strings. The escape handling is where an ordinary parser becomes a security
 * question: `content` is echoed back to the client and fed to a tokenizer.
 * ===========================================================================
 */

TEST(escapes_are_decoded) {
    const char* b = "{\"s\":\"a\\\"b\\\\c\\/d\\ne\\tf\\r\\b\\fg\"}";
    JsonDoc     d;
    JsonError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &d, &e), 0);

    char s[64];
    i64  n = member_str(&d, b, json_root(&d), "s", s, sizeof s);
    ASSERT_EQ_I64(n, 15);
    ASSERT_TRUE(n == 15 && memcmp(s, "a\"b\\c/d\ne\tf\r\b\fg", 15) == 0);
}

TEST(basic_multilingual_plane_escapes_become_utf8) {
    /* U+0041 'A' (1 byte), U+00E9 'e-acute' (2), U+20AC euro (3). */
    const char* b = "{\"s\":\"\\u0041\\u00e9\\u20AC\"}";
    JsonDoc     d;
    JsonError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &d, &e), 0);

    char s[32];
    i64  n = member_str(&d, b, json_root(&d), "s", s, sizeof s);
    ASSERT_EQ_I64(n, 6);
    ASSERT_TRUE(n == 6 && memcmp(s, "A\xc3\xa9\xe2\x82\xac", 6) == 0);
}

/* A surrogate PAIR is one codepoint in four UTF-8 bytes. openai-python sends
 * these for any emoji, so this is ordinary traffic, not an edge case. */
TEST(surrogate_pairs_become_one_four_byte_codepoint) {
    const char* b = "{\"s\":\"\\uD83D\\uDE00\"}"; /* U+1F600 */
    JsonDoc     d;
    JsonError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &d, &e), 0);

    char s[16];
    i64  n = member_str(&d, b, json_root(&d), "s", s, sizeof s);
    ASSERT_EQ_I64(n, 4);
    ASSERT_TRUE(n == 4 && memcmp(s, "\xf0\x9f\x98\x80", 4) == 0);
}

/* Raw UTF-8 passes through untouched -- clients send it far more often than
 * they send \u escapes. */
TEST(raw_utf8_passes_through_unchanged) {
    const char* b = "{\"s\":\"caf\xc3\xa9 \xe6\x97\xa5\"}";
    JsonDoc     d;
    JsonError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &d, &e), 0);

    char s[32];
    i64  n = member_str(&d, b, json_root(&d), "s", s, sizeof s);
    ASSERT_EQ_I64(n, 9);
    ASSERT_TRUE(n == 9 && memcmp(s, "caf\xc3\xa9 \xe6\x97\xa5", 9) == 0);
}

/* The caller-owns-buffer rule, same as tok.h and stoke.h: the return is the
 * length it WOULD have written, so a caller can size a buffer and retry. */
TEST(json_string_reports_the_length_it_would_have_written) {
    const char* b = "{\"s\":\"abcdefgh\"}";
    JsonDoc     d;
    JsonError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &d, &e), 0);

    char tiny[4];
    ASSERT_EQ_I64(member_str(&d, b, json_root(&d), "s", tiny, sizeof tiny), 8);
    char big[16];
    ASSERT_EQ_I64(member_str(&d, b, json_root(&d), "s", big, sizeof big), 8);
    ASSERT_TRUE(memcmp(big, "abcdefgh", 8) == 0);
}

TEST(empty_string_is_legal_and_not_an_absent_key) {
    const char* b = "{\"s\":\"\"}";
    JsonDoc     d;
    JsonError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &d, &e), 0);
    const JsonNode* m = json_member(&d, b, json_root(&d), "s");
    ASSERT_TRUE(m != NULL);
    if (m) {
        ASSERT_EQ_I64(ntype(m), JSON_STR);
        char s[4];
        ASSERT_EQ_I64(json_string(b, m, s, sizeof s), 0);
    }
}

/* ===========================================================================
 * Numbers.
 * ===========================================================================
 */

TEST(numbers_of_every_legal_shape_are_read) {
    const char* b = "[0,-1,3.5,-2.25e2,1E+2,1e-2,123456789]";
    JsonDoc     d;
    JsonError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &d, &e), 0);
    const JsonNode* a = json_root(&d);
    ASSERT_EQ_I64(nkids(a), 7);

    double v;
    ASSERT_EQ_I64(json_number(b, json_at(&d, a, 0), &v), 0);
    ASSERT_TRUE(v == 0.0);
    ASSERT_EQ_I64(json_number(b, json_at(&d, a, 1), &v), 0);
    ASSERT_TRUE(v == -1.0);
    ASSERT_EQ_I64(json_number(b, json_at(&d, a, 2), &v), 0);
    ASSERT_TRUE(v == 3.5);
    ASSERT_EQ_I64(json_number(b, json_at(&d, a, 3), &v), 0);
    ASSERT_TRUE(v == -225.0);
    ASSERT_EQ_I64(json_number(b, json_at(&d, a, 4), &v), 0);
    ASSERT_TRUE(v == 100.0);
    ASSERT_EQ_I64(json_number(b, json_at(&d, a, 6), &v), 0);
    ASSERT_TRUE(v == 123456789.0);
}

/* Type confusion is refused rather than coerced. `max_tokens: "16"` is a client
 * bug and answering it as 16 hides the bug in OUR log, not theirs. */
TEST(reading_a_value_as_the_wrong_type_fails) {
    const char* b = "{\"a\":\"16\",\"b\":16}";
    JsonDoc     d;
    JsonError   e;
    ASSERT_EQ_I64(parse(b, strlen(b), &d, &e), 0);

    double v;
    ASSERT_TRUE(json_number(b, json_member(&d, b, json_root(&d), "a"), &v) != 0);
    int bit;
    ASSERT_TRUE(json_bool(b, json_member(&d, b, json_root(&d), "b"), &bit) != 0);
    char s[8];
    ASSERT_TRUE(json_string(b, json_member(&d, b, json_root(&d), "b"), s,
                            sizeof s) < 0);
    /* And every accessor tolerates NULL, so an absent key does not have to be
     * checked twice at each call site. */
    ASSERT_TRUE(json_number(b, NULL, &v) != 0);
    ASSERT_TRUE(json_bool(b, NULL, &bit) != 0);
    ASSERT_TRUE(json_string(b, NULL, s, sizeof s) < 0);
}

/* ===========================================================================
 * Rejections. Each with its own slug.
 * ===========================================================================
 */

TEST(reject_empty_and_whitespace_only) {
    assert_bad(S(""), "empty");
    assert_bad(S("   \r\n\t "), "empty");
}

TEST(reject_trailing_content_after_the_value) {
    assert_bad(S("{} {}"), "trailing-content");
    assert_bad(S("1 2"), "trailing-content");
    /* A NUL after a complete value is still trailing content, not a
     * terminator -- this parser is never handed a C string. */
    assert_bad(S("{}\0x"), "trailing-content");
}

TEST(reject_unterminated_string) {
    assert_bad(S("{\"a\":\"unfinished}"), "unterminated-string");
}

TEST(reject_control_character_inside_a_string) {
    /* A raw newline inside a string is invalid JSON, and accepting it is how a
     * response gets a header injected into it later. */
    assert_bad(S("{\"a\":\"x\ny\"}"), "control-in-string");
    assert_bad(S("{\"a\":\"x\0y\"}"), "control-in-string");
}

TEST(reject_bad_escape) {
    assert_bad(S("{\"a\":\"x\\qy\"}"), "bad-escape");
    assert_bad(S("{\"a\":\"x\\u00zz\"}"), "bad-escape");
    assert_bad(S("{\"a\":\"x\\u12\"}"), "bad-escape");
}

/* A lone surrogate is not a codepoint and cannot be encoded as UTF-8. Emitting
 * the replacement character silently would put bytes in the prompt the client
 * never sent. */
TEST(reject_lone_surrogate) {
    assert_bad(S("{\"a\":\"\\uD83D\"}"), "lone-surrogate");
    assert_bad(S("{\"a\":\"\\uDE00\\uD83D\"}"), "lone-surrogate");
    assert_bad(S("{\"a\":\"\\uD83Dx\"}"), "lone-surrogate");
}

TEST(reject_structural_mistakes) {
    assert_bad(S("{\"a\" 1}"), "expected-colon");
    assert_bad(S("{\"a\":1,}"), "trailing-comma");
    assert_bad(S("[1,]"), "trailing-comma");
    assert_bad(S("{a:1}"), "expected-key");
    assert_bad(S("{\"a\":1"), "unexpected-end");
    assert_bad(S("[1,2"), "unexpected-end");
    assert_bad(S("[1 2]"), "expected-comma");
}

TEST(reject_javascript_that_is_not_json) {
    assert_bad(S("{'a':1}"), "expected-key");
    assert_bad(S("[NaN]"), "bad-value");
    assert_bad(S("[Infinity]"), "bad-value");
    assert_bad(S("[undefined]"), "bad-value");
    assert_bad(S("[tru]"), "bad-value");
}

/* Leading zeros and a bare `-` are the number shapes a hand-written scanner
 * accepts by accident. */
TEST(reject_malformed_numbers) {
    assert_bad(S("[01]"), "bad-number");
    assert_bad(S("[-]"), "bad-number");
    assert_bad(S("[1.]"), "bad-number");
    assert_bad(S("[.5]"), "bad-value");
    assert_bad(S("[1e]"), "bad-number");
    assert_bad(S("[1e+]"), "bad-number");
}

/* ===========================================================================
 * Bounds. A parser without these is a denial of service with a colon in it.
 * ===========================================================================
 */

static char g_deep[8192];

TEST(nesting_deeper_than_the_cap_is_refused_not_recursed) {
    /* Without a depth cap this is a stack overflow, which is a crash a client
     * can ask for in 70 bytes. */
    size_t n = 0;
    for (size_t i = 0; i < JSON_MAX_DEPTH + 5; i++) g_deep[n++] = '[';
    for (size_t i = 0; i < JSON_MAX_DEPTH + 5; i++) g_deep[n++] = ']';
    assert_bad(g_deep, n, "too-deep");
}

TEST(nesting_exactly_at_the_cap_is_accepted) {
    size_t n = 0;
    for (size_t i = 0; i < JSON_MAX_DEPTH; i++) g_deep[n++] = '[';
    for (size_t i = 0; i < JSON_MAX_DEPTH; i++) g_deep[n++] = ']';

    JsonDoc   d;
    JsonError e;
    ASSERT_EQ_I64(parse(g_deep, n, &d, &e), 0);
}

TEST(more_nodes_than_the_cap_is_refused) {
    size_t n = 0;
    g_deep[n++] = '[';
    for (size_t i = 0; i <= JSON_MAX_NODES; i++) {
        g_deep[n++] = '1';
        g_deep[n++] = ',';
    }
    g_deep[n - 1] = ']';
    assert_bad(g_deep, n, "too-many-nodes");
}

/* ===========================================================================
 * The document that actually matters.
 * ===========================================================================
 */

TEST(a_real_chat_completions_body_reads_end_to_end) {
    const char* b =
        "{\"model\": \"qwen3-4b\", \"messages\": ["
        "{\"role\": \"system\", \"content\": \"You are terse.\"},"
        "{\"role\": \"user\", \"content\": \"Say \\\"hi\\\" \\u0021\"}"
        "], \"max_tokens\": 24, \"stream\": false, \"temperature\": 0}";
    JsonDoc   d;
    JsonError e;
    ASSERT_EQ_I64(parse(b, strlen(b), &d, &e), 0);

    const JsonNode* root = json_root(&d);
    char            s[64];
    ASSERT_EQ_I64(member_str(&d, b, root, "model", s, sizeof s), 8);

    const JsonNode* msgs = json_member(&d, b, root, "messages");
    ASSERT_TRUE(msgs != NULL);
    if (!msgs) return;
    ASSERT_EQ_I64(nkids(msgs), 2);

    i64 n = member_str(&d, b, json_at(&d, msgs, 1), "content", s, sizeof s);
    ASSERT_EQ_I64(n, 10);
    ASSERT_TRUE(n == 10 && memcmp(s, "Say \"hi\" !", 10) == 0);

    double v;
    ASSERT_EQ_I64(json_number(b, json_member(&d, b, root, "max_tokens"), &v), 0);
    ASSERT_EQ_I64((i64)v, 24);
    int bit;
    ASSERT_EQ_I64(json_bool(b, json_member(&d, b, root, "stream"), &bit), 0);
    ASSERT_EQ_I64(bit, 0);
}

/* ===========================================================================
 * Writing. The response side is small enough to be a few helpers, but the
 * ESCAPING is not optional: a model can emit a quote or a newline in its very
 * first token, and an unescaped one makes the response unparseable by the
 * client that asked for it.
 * ===========================================================================
 */

TEST(json_escape_covers_everything_that_must_be_escaped) {
    /* 16 bytes in: the seven short escapes, a C0 control that has none, and
     * ordinary characters between them. 28 out. */
    static const char in[]   = "a\"b\\c\nd\te\rf\b\x0c" "g\x01h";
    static const char want[] = "a\\\"b\\\\c\\nd\\te\\rf\\b\\fg\\u0001h";
    char              out[128];

    ASSERT_EQ_I64(sizeof(in) - 1, 16);
    i64 n = json_escape(in, sizeof(in) - 1, out, sizeof out);
    ASSERT_EQ_I64(n, (i64)(sizeof(want) - 1));
    ASSERT_EQ_I64(n, 28);
    ASSERT_TRUE(n == 28 && memcmp(out, want, 28) == 0);
}

TEST(json_escape_reports_the_length_it_would_have_written) {
    char tiny[4];
    ASSERT_EQ_I64(json_escape("abcdefgh", 8, tiny, sizeof tiny), 8);
}

/* Raw UTF-8 is NOT escaped -- it is already valid JSON, and \u-escaping it
 * would triple the size of every non-English response for nothing. */
TEST(json_escape_leaves_utf8_alone) {
    char out[32];
    i64  n = json_escape("caf\xc3\xa9", 5, out, sizeof out);
    ASSERT_EQ_I64(n, 5);
    ASSERT_TRUE(n == 5 && memcmp(out, "caf\xc3\xa9", 5) == 0);
}

/* And the round trip, which is the only assertion that catches an escaper and
 * a parser that are wrong in the same direction. */
TEST(escape_then_parse_round_trips) {
    static const char nasty[] = "he said \"x\"\n\tand \\ then \x01 caf\xc3\xa9";
    const size_t      nlen    = sizeof(nasty) - 1;

    char body[256];
    char esc[192];
    i64  en = json_escape(nasty, nlen, esc, sizeof esc);
    ASSERT_TRUE(en > 0 && (size_t)en < sizeof esc);
    int blen = snprintf(body, sizeof body, "{\"c\":\"%.*s\"}", (int)en, esc);

    JsonDoc   d;
    JsonError e;
    ASSERT_EQ_I64(parse(body, (size_t)blen, &d, &e), 0);

    char back[128];
    i64  n = member_str(&d, body, json_root(&d), "c", back, sizeof back);
    ASSERT_EQ_I64(n, (i64)nlen);
    ASSERT_TRUE(n == (i64)nlen && memcmp(back, nasty, nlen) == 0);
}

/* ---- the streaming cut point --------------------------------------------
 *
 * ▶▶ WHY THIS LIVES BESIDE THE ESCAPER. json_escape's contract is "raw UTF-8 is
 * passed through unchanged, because it is already valid inside a JSON string".
 * That is only true if what you hand it IS valid UTF-8 -- and a streaming
 * producer does not get to choose where its input ends. This model's vocabulary
 * contains all 256 single-byte ids (see test_tok's
 * all_256_byte_ids_decode_to_their_own_byte), so one character can and does
 * arrive as several tokens, and a server that emitted each token as it came
 * would put half a codepoint inside a JSON string. This is the function that
 * says where it is safe to cut; the tail waits for the next token.
 */

static i64 upfx(const char* b, size_t n) {
    return (i64)json_utf8_complete_prefix(b, n);
}

TEST(utf8_prefix_of_empty_is_zero) { ASSERT_EQ_I64(upfx("", 0), 0); }

TEST(utf8_prefix_of_ascii_is_all_of_it) { ASSERT_EQ_I64(upfx(S("hello")), 5); }

TEST(utf8_prefix_keeps_a_complete_two_byte_sequence) {
    ASSERT_EQ_I64(upfx(S("caf\xc3\xa9")), 5);
}

TEST(utf8_prefix_cuts_a_lone_two_byte_lead) {
    /* "caf" + 0xC3 with its continuation byte still unsent. */
    ASSERT_EQ_I64(upfx(S("caf\xc3")), 3);
}

TEST(utf8_prefix_keeps_a_complete_three_byte_sequence) {
    ASSERT_EQ_I64(upfx(S("\xe2\x82\xac")), 3); /* U+20AC EURO SIGN */
}

TEST(utf8_prefix_cuts_a_three_byte_sequence_missing_its_last_byte) {
    ASSERT_EQ_I64(upfx(S("x\xe2\x82")), 1);
}

TEST(utf8_prefix_cuts_a_three_byte_sequence_missing_two_bytes) {
    ASSERT_EQ_I64(upfx(S("x\xe2")), 1);
}

TEST(utf8_prefix_keeps_a_complete_four_byte_sequence) {
    ASSERT_EQ_I64(upfx(S("\xf0\x9f\x98\x80")), 4); /* U+1F600 */
}

TEST(utf8_prefix_cuts_a_four_byte_sequence_missing_its_last_byte) {
    ASSERT_EQ_I64(upfx(S("hi\xf0\x9f\x98")), 2);
}

/* A continuation byte with no lead in front of it is not an unfinished
 * sequence, it is garbage -- and holding it back would stall the stream
 * forever waiting for a lead byte that is never coming. It goes out. */
TEST(utf8_prefix_passes_a_lone_continuation_byte_through) {
    ASSERT_EQ_I64(upfx(S("ok\x80")), 3);
}

/* The look-back is bounded: four continuation bytes after a 4-byte lead means
 * the sequence ENDED and the last byte is garbage, not an unfinished tail. */
TEST(utf8_prefix_does_not_look_back_past_a_full_sequence) {
    ASSERT_EQ_I64(upfx(S("\xf0\x9f\x98\x80\x80")), 5);
}

/* An overlong-looking 0xFF is not a lead byte at all under any UTF-8 rule, so
 * it can never become complete and must not hold the stream. */
TEST(utf8_prefix_passes_an_impossible_byte_through) {
    ASSERT_EQ_I64(upfx(S("a\xff")), 2);
}
