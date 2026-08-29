/* src/json.h -- the JSON reader and the escaper. M6 rung A3.
 *
 * ▶▶ WHY A PARSER, WHEN THE SERVER WANTS THREE FIELDS. Because "three fields"
 * is how you end up scanning for the bytes `"content":` and getting it wrong on
 * the first request whose PROMPT contains that string, or whose messages nest,
 * or whose quotes are escaped. A chat-completions body is attacker-controlled
 * end to end.
 *
 * ▶▶ THE SAME FOUR RULES AS src/http.c, for the same reasons: no allocation, no
 * blocking, results are offsets into YOUR buffer, and it never calls fatal().
 * Malformed JSON is ordinary input for a public endpoint, not one of the M6
 * failure classes, and a parser that reached for fatal() would hand any client
 * on the network a remote process kill.
 *
 * ▶ THE NODE POOL IS FIXED AND SO IS THE DEPTH. Both caps are refusals, not
 * assertions: a parser with neither is a denial of service with a colon in it
 * (`[[[[[...` is a stack overflow a client can ask for in seventy bytes). The
 * document is left in YOUR buffer and JsonDoc holds only structure, so a whole
 * request costs one stack object and no heap.
 *
 * ⚠ THE BUFFER MUST OUTLIVE THE JsonDoc. Every node points into it by offset,
 * so moving or freeing the bytes invalidates the whole document at once.
 */
#ifndef STOKE_JSON_H
#define STOKE_JSON_H

#include "common.h"

/* Sized for a chat-completions body: a few hundred nodes covers a long
 * conversation, and anything past it is refused with `too-many-nodes` rather
 * than allocated for. */
#define JSON_MAX_NODES 512

/* Recursion depth. The parser is recursive descent, so this is the stack
 * bound, and it is checked BEFORE descending rather than after. */
#define JSON_MAX_DEPTH 32

#define JSON_NONE ((u32)-1)

typedef enum {
    JSON_NULL = 0,
    JSON_BOOL = 1,
    JSON_NUM  = 2,
    JSON_STR  = 3,
    JSON_ARR  = 4,
    JSON_OBJ  = 5
} JsonType;

typedef struct {
    i32         type; /* a JsonType; i32 so the struct has no enum in it   */
    u32         start, end; /* the value's byte range in the source        */
    u32         key_start, key_end; /* the member's KEY range; 0,0 if none */
    u32         first_child, next_sibling; /* indices, or JSON_NONE        */
    u32         n_children;
} JsonNode;

/* `reason` is a static string, never NULL after a failure, and a kebab-case
 * slug for the same reason HttpError's is: a dozen malformed documents are all
 * "invalid JSON", and a test asserting only that a parse failed would accept a
 * parser that failed for the wrong cause. `offset` is where it gave up. */
typedef struct {
    const char* reason;
    u32         offset;
} JsonError;

typedef struct {
    JsonNode nodes[JSON_MAX_NODES];
    u32      n_nodes;
} JsonDoc;

/* 0 on success, non-zero with `*err` set otherwise. `buf` may contain NUL
 * bytes and is never treated as a C string -- a NUL inside a string literal is
 * a control character and is refused, and one after a complete value is
 * trailing content. Thread-safe: no state outside the arguments. */
int json_parse(const char* buf, size_t len, JsonDoc* d, JsonError* err);

/* The document's top-level value, or NULL if the document did not parse. */
const JsonNode* json_root(const JsonDoc* d);

/* The LAST member of `obj` named `key` (exact, case-sensitive, escapes in the
 * key resolved), or NULL. Last-wins is stated here and tested by name: a client
 * that sends a key twice must get one deterministic answer.
 *
 * Returns NULL, never a zero node, when the key is absent -- the difference is
 * whether the server applies a default or reads a field nobody sent. Tolerates
 * a NULL `obj`, so a chain of lookups needs one check at the end. */
const JsonNode* json_member(const JsonDoc* d, const char* buf,
                            const JsonNode* obj, const char* key);

/* Element `i` of `arr`, or NULL if absent or out of range. */
const JsonNode* json_at(const JsonDoc* d, const JsonNode* arr, u32 i);

/* Decodes a JSON_STR node's value -- escapes resolved, \uXXXX pairs folded to
 * UTF-8 -- into `out`. Returns the byte count it WOULD have written (the
 * tok.h/stoke.h caller-owns-with-a-cap rule, so a caller can size and retry),
 * or a negative value if `n` is NULL or is not a string. Not NUL-terminated.
 *
 * The escape VALIDITY was already decided at parse time, so this cannot fail on
 * the content -- json_parse refuses a bad escape or a lone surrogate outright
 * rather than leaving it for whoever reads the value later. */
i64 json_string(const char* buf, const JsonNode* n, char* out, size_t cap);

/* 0 on success. Non-zero if `n` is NULL or is not of that type -- a client
 * sending `"max_tokens": "16"` has a bug, and reading it as 16 puts their bug
 * in our log instead of theirs. */
int json_number(const char* buf, const JsonNode* n, double* out);
int json_bool(const char* buf, const JsonNode* n, int* out);

/* ---- writing ------------------------------------------------------------
 *
 * There is no serializer, only this: the response shapes are fixed and are
 * clearer written out with snprintf than assembled through a builder. What is
 * NOT optional is the escaping -- a model can emit a quote or a newline in its
 * first token, and one unescaped byte makes the response unparseable by the
 * client that asked for it.
 */

/* Escapes `len` bytes for placement inside a JSON string literal. Returns the
 * byte count it WOULD have written, same rule as json_string.
 *
 * Raw UTF-8 is passed through unchanged: it is already valid inside a JSON
 * string, and \u-escaping it would triple the size of every non-English
 * response to no purpose. Only the seven short escapes and C0 controls are
 * rewritten. */
i64 json_escape(const char* in, size_t len, char* out, size_t cap);

/* How many leading bytes of `buf` consist ONLY of complete UTF-8 sequences --
 * i.e. where a streaming producer may cut without splitting a character.
 *
 * ⚠ THIS IS THE OTHER HALF OF json_escape's CONTRACT. That function passes raw
 * UTF-8 through untouched because it is already valid inside a JSON string;
 * that reasoning holds only for input which IS valid UTF-8, and a server
 * emitting one token at a time does not choose where its bytes end. This model
 * has all 256 single-byte ids in its vocabulary, so one character can arrive as
 * several tokens. Cut here, hold the remainder, prepend it to the next piece.
 *
 * Only a TRAILING incomplete sequence is held back. A byte that cannot begin
 * one -- a stray continuation byte, 0xFE, 0xFF -- is passed through, because
 * waiting for it to complete is waiting forever. Never returns more than len.
 */
size_t json_utf8_complete_prefix(const char* buf, size_t len);

#endif /* STOKE_JSON_H */
