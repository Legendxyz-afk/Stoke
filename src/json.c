/* src/json.c -- the JSON reader and the escaper. See json.h for the contract.
 *
 * Recursive descent with an explicit depth cap, over a caller-owned buffer, into
 * a fixed node pool. No allocation, no fatal(), no state outside the arguments.
 */
#include "common.h"

#include "json.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* b;
    size_t      len;
    size_t      i;
    JsonDoc*    d;
    JsonError*  err;
    int         depth;
} P;

static int fail(P* p, const char* reason) {
    p->err->reason = reason;
    p->err->offset = (u32)p->i;
    return -1;
}

static void skip_ws(P* p) {
    while (p->i < p->len) {
        char c = p->b[p->i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            p->i++;
        } else {
            break;
        }
    }
}

static int at_end(const P* p) { return p->i >= p->len; }

static u32 new_node(P* p, i32 type) {
    if (p->d->n_nodes >= JSON_MAX_NODES) return JSON_NONE;
    u32       k = p->d->n_nodes++;
    JsonNode* n = &p->d->nodes[k];
    n->type         = type;
    n->start        = (u32)p->i;
    n->end          = (u32)p->i;
    n->key_start    = 0;
    n->key_end      = 0;
    n->first_child  = JSON_NONE;
    n->next_sibling = JSON_NONE;
    n->n_children   = 0;
    return k;
}

static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Reads four hex digits at `at`, or -1. */
static i32 hex4(const char* b, size_t len, size_t at) {
    if (at + 4 > len) return -1;
    i32 v = 0;
    for (int k = 0; k < 4; k++) {
        int h = hex_val(b[at + k]);
        if (h < 0) return -1;
        v = (v << 4) | h;
    }
    return v;
}

static int is_high_surrogate(i32 c) { return c >= 0xD800 && c <= 0xDBFF; }
static int is_low_surrogate(i32 c) { return c >= 0xDC00 && c <= 0xDFFF; }

/* ---- strings ------------------------------------------------------------
 *
 * VALIDATES ONLY -- nothing is decoded here and nothing is written. The value
 * stays in the caller's buffer and is decoded on demand by json_string().
 *
 * ▶ EVERY ESCAPE IS SETTLED AT PARSE TIME, including surrogate pairing. That is
 * what lets json_string() have no failure mode of its own: a decoder that could
 * still discover a lone surrogate would have to report it from somewhere with
 * no error channel, and the tempting answer there is to emit a replacement
 * character -- putting bytes into the prompt that the client never sent.
 *
 * On entry p->i is at the opening quote. On success p->i is past the closing
 * one, and the two out-params bracket the CONTENT.
 */
static int scan_string(P* p, u32* start, u32* end) {
    p->i++; /* the opening quote */
    *start = (u32)p->i;

    while (!at_end(p)) {
        unsigned char c = (unsigned char)p->b[p->i];
        if (c == '"') {
            *end = (u32)p->i;
            p->i++;
            return 0;
        }
        if (c < 0x20) {
            /* A raw newline or NUL inside a string is invalid JSON. Accepting
             * it is how a control character reaches a response body later. */
            return fail(p, "control-in-string");
        }
        if (c != '\\') {
            p->i++;
            continue;
        }

        p->i++; /* the backslash */
        if (at_end(p)) return fail(p, "unterminated-string");
        char e = p->b[p->i];
        if (e == '"' || e == '\\' || e == '/' || e == 'b' || e == 'f' ||
            e == 'n' || e == 'r' || e == 't') {
            p->i++;
            continue;
        }
        if (e != 'u') return fail(p, "bad-escape");

        i32 cp = hex4(p->b, p->len, p->i + 1);
        if (cp < 0) return fail(p, "bad-escape");
        p->i += 5; /* u + 4 hex */

        if (is_low_surrogate(cp)) {
            /* A trailing surrogate with no leading one is not a codepoint. */
            return fail(p, "lone-surrogate");
        }
        if (is_high_surrogate(cp)) {
            if (p->i + 1 >= p->len || p->b[p->i] != '\\' ||
                p->b[p->i + 1] != 'u') {
                return fail(p, "lone-surrogate");
            }
            i32 lo = hex4(p->b, p->len, p->i + 2);
            if (lo < 0) return fail(p, "bad-escape");
            if (!is_low_surrogate(lo)) return fail(p, "lone-surrogate");
            p->i += 6; /* \ u + 4 hex */
        }
    }
    return fail(p, "unterminated-string");
}

/* Writes one codepoint as UTF-8, returning its byte length and emitting only
 * what fits -- the caller-owns-with-a-cap rule applies per byte, so a truncated
 * call still reports the full length it would have needed. */
static size_t utf8_put(i32 cp, char* out, size_t cap, size_t at) {
    unsigned char tmp[4];
    size_t        n;
    if (cp < 0x80) {
        tmp[0] = (unsigned char)cp;
        n      = 1;
    } else if (cp < 0x800) {
        tmp[0] = (unsigned char)(0xC0 | (cp >> 6));
        tmp[1] = (unsigned char)(0x80 | (cp & 0x3F));
        n      = 2;
    } else if (cp < 0x10000) {
        tmp[0] = (unsigned char)(0xE0 | (cp >> 12));
        tmp[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[2] = (unsigned char)(0x80 | (cp & 0x3F));
        n      = 3;
    } else {
        tmp[0] = (unsigned char)(0xF0 | (cp >> 18));
        tmp[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
        tmp[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[3] = (unsigned char)(0x80 | (cp & 0x3F));
        n      = 4;
    }
    for (size_t k = 0; k < n; k++) {
        if (at + k < cap && out) out[at + k] = (char)tmp[k];
    }
    return n;
}

/* Decodes the escaped range [start,end) into `out`. Returns the length it WOULD
 * have written. Cannot fail: json_parse already settled every escape. */
static i64 decode_range(const char* buf, u32 start, u32 end, char* out,
                        size_t cap) {
    size_t w = 0;
    for (u32 k = start; k < end;) {
        if (buf[k] != '\\') {
            if (w < cap && out) out[w] = buf[k];
            w++;
            k++;
            continue;
        }
        char e = buf[k + 1];
        char lit;
        switch (e) {
            case 'n': lit = '\n'; break;
            case 't': lit = '\t'; break;
            case 'r': lit = '\r'; break;
            case 'b': lit = '\b'; break;
            case 'f': lit = '\f'; break;
            case '"': lit = '"'; break;
            case '\\': lit = '\\'; break;
            case '/': lit = '/'; break;
            default: lit = 0; break;
        }
        if (e != 'u') {
            if (w < cap && out) out[w] = lit;
            w++;
            k += 2;
            continue;
        }
        i32 cp = hex4(buf, (size_t)end, k + 2);
        k += 6;
        if (is_high_surrogate(cp)) {
            i32 lo = hex4(buf, (size_t)end, k + 2);
            cp     = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            k += 6;
        }
        w += utf8_put(cp, out, cap, w);
    }
    return (i64)w;
}

/* ---- numbers ------------------------------------------------------------
 *
 * The grammar exactly: -? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?
 * Leading zeros and a bare `-` are the shapes a hand-written scanner takes by
 * accident, and both are refused here.
 */
static int scan_number(P* p, u32* start, u32* end) {
    *start = (u32)p->i;
    if (!at_end(p) && p->b[p->i] == '-') p->i++;

    if (at_end(p)) return fail(p, "bad-number");
    if (p->b[p->i] == '0') {
        p->i++;
        if (!at_end(p) && p->b[p->i] >= '0' && p->b[p->i] <= '9') {
            return fail(p, "bad-number");
        }
    } else if (p->b[p->i] >= '1' && p->b[p->i] <= '9') {
        while (!at_end(p) && p->b[p->i] >= '0' && p->b[p->i] <= '9') p->i++;
    } else {
        return fail(p, "bad-number");
    }

    if (!at_end(p) && p->b[p->i] == '.') {
        p->i++;
        if (at_end(p) || p->b[p->i] < '0' || p->b[p->i] > '9') {
            return fail(p, "bad-number");
        }
        while (!at_end(p) && p->b[p->i] >= '0' && p->b[p->i] <= '9') p->i++;
    }

    if (!at_end(p) && (p->b[p->i] == 'e' || p->b[p->i] == 'E')) {
        p->i++;
        if (!at_end(p) && (p->b[p->i] == '+' || p->b[p->i] == '-')) p->i++;
        if (at_end(p) || p->b[p->i] < '0' || p->b[p->i] > '9') {
            return fail(p, "bad-number");
        }
        while (!at_end(p) && p->b[p->i] >= '0' && p->b[p->i] <= '9') p->i++;
    }

    *end = (u32)p->i;
    return 0;
}

/* ---- values ------------------------------------------------------------- */

static int parse_value(P* p, u32* out);

static int lit_is(const P* p, const char* s, size_t n) {
    return p->i + n <= p->len && memcmp(p->b + p->i, s, n) == 0;
}

/* Appends `child` to `parent`'s list, keeping source order. */
static void attach(JsonDoc* d, u32 parent, u32 child, u32* last) {
    if (*last == JSON_NONE) {
        d->nodes[parent].first_child = child;
    } else {
        d->nodes[*last].next_sibling = child;
    }
    *last = child;
    d->nodes[parent].n_children++;
}

static int parse_object(P* p, u32 self) {
    p->i++; /* '{' */
    u32 last = JSON_NONE;
    skip_ws(p);
    if (!at_end(p) && p->b[p->i] == '}') {
        p->i++;
        p->d->nodes[self].end = (u32)p->i;
        return 0;
    }
    for (;;) {
        skip_ws(p);
        if (at_end(p)) return fail(p, "unexpected-end");
        if (p->b[p->i] != '"') return fail(p, "expected-key");

        u32 ks, ke;
        if (scan_string(p, &ks, &ke) != 0) return -1;

        skip_ws(p);
        if (at_end(p)) return fail(p, "unexpected-end");
        if (p->b[p->i] != ':') return fail(p, "expected-colon");
        p->i++;

        u32 child;
        if (parse_value(p, &child) != 0) return -1;
        p->d->nodes[child].key_start = ks;
        p->d->nodes[child].key_end   = ke;
        attach(p->d, self, child, &last);

        skip_ws(p);
        if (at_end(p)) return fail(p, "unexpected-end");
        if (p->b[p->i] == ',') {
            p->i++;
            skip_ws(p);
            if (!at_end(p) && p->b[p->i] == '}') return fail(p, "trailing-comma");
            continue;
        }
        if (p->b[p->i] == '}') {
            p->i++;
            p->d->nodes[self].end = (u32)p->i;
            return 0;
        }
        return fail(p, "expected-comma");
    }
}

static int parse_array(P* p, u32 self) {
    p->i++; /* '[' */
    u32 last = JSON_NONE;
    skip_ws(p);
    if (!at_end(p) && p->b[p->i] == ']') {
        p->i++;
        p->d->nodes[self].end = (u32)p->i;
        return 0;
    }
    for (;;) {
        u32 child;
        if (parse_value(p, &child) != 0) return -1;
        attach(p->d, self, child, &last);

        skip_ws(p);
        if (at_end(p)) return fail(p, "unexpected-end");
        if (p->b[p->i] == ',') {
            p->i++;
            skip_ws(p);
            if (!at_end(p) && p->b[p->i] == ']') return fail(p, "trailing-comma");
            continue;
        }
        if (p->b[p->i] == ']') {
            p->i++;
            p->d->nodes[self].end = (u32)p->i;
            return 0;
        }
        return fail(p, "expected-comma");
    }
}

static int parse_value(P* p, u32* out) {
    skip_ws(p);
    if (at_end(p)) return fail(p, "unexpected-end");

    char c = p->b[p->i];
    i32  type;
    if (c == '{') {
        type = JSON_OBJ;
    } else if (c == '[') {
        type = JSON_ARR;
    } else if (c == '"') {
        type = JSON_STR;
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        type = JSON_NUM;
    } else if (lit_is(p, "true", 4) || lit_is(p, "false", 5)) {
        type = JSON_BOOL;
    } else if (lit_is(p, "null", 4)) {
        type = JSON_NULL;
    } else {
        return fail(p, "bad-value");
    }

    u32 self = new_node(p, type);
    if (self == JSON_NONE) return fail(p, "too-many-nodes");
    *out = self;

    switch (type) {
        case JSON_OBJ:
        case JSON_ARR: {
            /* Checked BEFORE descending: the cap is the stack bound, and a
             * check after the recursive call is a check that never runs. */
            if (++p->depth > JSON_MAX_DEPTH) return fail(p, "too-deep");
            int rc = (type == JSON_OBJ) ? parse_object(p, self)
                                        : parse_array(p, self);
            p->depth--;
            return rc;
        }
        case JSON_STR: {
            u32 s, e;
            if (scan_string(p, &s, &e) != 0) return -1;
            p->d->nodes[self].start = s;
            p->d->nodes[self].end   = e;
            return 0;
        }
        case JSON_NUM: {
            u32 s, e;
            if (scan_number(p, &s, &e) != 0) return -1;
            p->d->nodes[self].start = s;
            p->d->nodes[self].end   = e;
            return 0;
        }
        case JSON_BOOL: {
            size_t n = (p->b[p->i] == 't') ? 4 : 5;
            p->d->nodes[self].start = (u32)p->i;
            p->i += n;
            p->d->nodes[self].end = (u32)p->i;
            return 0;
        }
        default: {
            p->d->nodes[self].start = (u32)p->i;
            p->i += 4;
            p->d->nodes[self].end = (u32)p->i;
            return 0;
        }
    }
}

int json_parse(const char* buf, size_t len, JsonDoc* d, JsonError* err) {
    d->n_nodes  = 0;
    err->reason = NULL;
    err->offset = 0;

    P p = {buf, len, 0, d, err, 0};
    skip_ws(&p);
    if (at_end(&p)) return fail(&p, "empty");

    u32 root;
    if (parse_value(&p, &root) != 0) {
        d->n_nodes = 0; /* nothing here is readable after a failure */
        return -1;
    }
    skip_ws(&p);
    if (!at_end(&p)) {
        /* A NUL after a complete value is trailing content, not a terminator:
         * this parser is never handed a C string. */
        d->n_nodes = 0;
        return fail(&p, "trailing-content");
    }
    return 0;
}

/* ---- reading ------------------------------------------------------------ */

const JsonNode* json_root(const JsonDoc* d) {
    return d->n_nodes ? &d->nodes[0] : NULL;
}

const JsonNode* json_at(const JsonDoc* d, const JsonNode* arr, u32 i) {
    if (!arr || arr->type != JSON_ARR) return NULL;
    u32 k = arr->first_child;
    while (k != JSON_NONE && i > 0) {
        k = d->nodes[k].next_sibling;
        i--;
    }
    return k == JSON_NONE ? NULL : &d->nodes[k];
}

/* Keys are short ASCII literals in every call site this server has, so an
 * unescaped 128-byte window covers them; a key longer than that simply never
 * matches, which is correct because no lookup here uses one. */
#define JSON_KEY_MAX 128

const JsonNode* json_member(const JsonDoc* d, const char* buf,
                            const JsonNode* obj, const char* key) {
    if (!obj || obj->type != JSON_OBJ) return NULL;
    size_t klen = strlen(key);
    if (klen >= JSON_KEY_MAX) return NULL;

    const JsonNode* found = NULL;
    for (u32 k = obj->first_child; k != JSON_NONE; k = d->nodes[k].next_sibling) {
        const JsonNode* n = &d->nodes[k];
        char            tmp[JSON_KEY_MAX];
        i64 dn = decode_range(buf, n->key_start, n->key_end, tmp, sizeof tmp);
        if (dn == (i64)klen && memcmp(tmp, key, klen) == 0) {
            /* Keep going: LAST wins, so a client sending a key twice gets one
             * deterministic answer rather than whichever the loop saw first. */
            found = n;
        }
    }
    return found;
}

i64 json_string(const char* buf, const JsonNode* n, char* out, size_t cap) {
    if (!n || n->type != JSON_STR) return -1;
    return decode_range(buf, n->start, n->end, out, cap);
}

int json_number(const char* buf, const JsonNode* n, double* out) {
    if (!n || n->type != JSON_NUM) return -1;
    /* strtod needs a NUL-terminated string; a JSON number may legally be longer
     * than any double needs, so the window is generous and anything past it is
     * refused rather than silently truncated into a different value.
     *
     * The decimal point is locale-dependent in strtod. This project never calls
     * setlocale, so the C locale is in force and '.' is correct -- noted because
     * a future setlocale() anywhere in the process would break this quietly. */
    char   tmp[384];
    size_t len = n->end - n->start;
    if (len == 0 || len >= sizeof tmp) return -1;
    memcpy(tmp, buf + n->start, len);
    tmp[len] = '\0';

    char*  endp = NULL;
    double v    = strtod(tmp, &endp);
    if (endp != tmp + len) return -1;
    *out = v;
    return 0;
}

int json_bool(const char* buf, const JsonNode* n, int* out) {
    if (!n || n->type != JSON_BOOL) return -1;
    *out = buf[n->start] == 't';
    return 0;
}

/* ---- writing ------------------------------------------------------------ */

i64 json_escape(const char* in, size_t len, char* out, size_t cap) {
    static const char HEX[] = "0123456789abcdef";
    size_t            w     = 0;

    for (size_t k = 0; k < len; k++) {
        unsigned char c = (unsigned char)in[k];
        char          two = 0;
        switch (c) {
            case '"': two = '"'; break;
            case '\\': two = '\\'; break;
            case '\n': two = 'n'; break;
            case '\r': two = 'r'; break;
            case '\t': two = 't'; break;
            case '\b': two = 'b'; break;
            case '\f': two = 'f'; break;
            default: break;
        }
        if (two) {
            if (w < cap) out[w] = '\\';
            w++;
            if (w < cap) out[w] = two;
            w++;
        } else if (c < 0x20) {
            /* A C0 control with no short escape. Six bytes, and mandatory:
             * leaving it raw makes the response unparseable by the client that
             * asked for it. */
            const char esc[6] = {'\\', 'u', '0', '0', HEX[c >> 4], HEX[c & 0xF]};
            for (int j = 0; j < 6; j++) {
                if (w < cap) out[w] = esc[j];
                w++;
            }
        } else {
            /* Everything else, RAW UTF-8 included: it is already legal inside a
             * JSON string, and \u-escaping it would triple the size of every
             * non-English response for nothing. */
            if (w < cap) out[w] = (char)c;
            w++;
        }
    }
    return (i64)w;
}
