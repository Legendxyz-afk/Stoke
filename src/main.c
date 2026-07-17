/* src/main.c -- stoke CLI. Task 5 implements only the `info` subcommand
 * (task-5-brief.md Step 3 + appendix A10's golden-output rules); later
 * milestones add tokenize/run/serve/bench (see the plan's file structure
 * note for src/main.c).
 */
#include "common.h"

#include "gguf.h"

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

/* ===========================================================================
 * Name tables for `stoke info` output. Deliberately NOT part of gguf.h's
 * public API: the numeric type values are already exposed (GgufTensor.type,
 * gguf_meta_type()/gguf_meta_arr_elem_type()); mapping them to display
 * strings is presentation logic that belongs with the CLI that owns
 * appendix A10's formatting rules, not with the parser.
 * ===========================================================================
 */

static const char* gguf_type_name(u32 t) {
    switch (t) {
        case GGUF_TYPE_UINT8: return "UINT8";
        case GGUF_TYPE_INT8: return "INT8";
        case GGUF_TYPE_UINT16: return "UINT16";
        case GGUF_TYPE_INT16: return "INT16";
        case GGUF_TYPE_UINT32: return "UINT32";
        case GGUF_TYPE_INT32: return "INT32";
        case GGUF_TYPE_FLOAT32: return "FLOAT32";
        case GGUF_TYPE_BOOL: return "BOOL";
        case GGUF_TYPE_STRING: return "STRING";
        case GGUF_TYPE_ARRAY: return "ARRAY";
        case GGUF_TYPE_UINT64: return "UINT64";
        case GGUF_TYPE_INT64: return "INT64";
        case GGUF_TYPE_FLOAT64: return "FLOAT64";
        default:
            fatal("stoke info: internal -- unknown GGUF value type %u "
                  "(gguf_open should already have rejected this)",
                  t);
    }
    return NULL; /* unreachable */
}

static const char* ggml_type_name(u32 t) {
    switch (t) {
        case GGUF_GGML_TYPE_F32: return "F32";
        case GGUF_GGML_TYPE_Q4_K: return "Q4_K";
        case GGUF_GGML_TYPE_Q6_K: return "Q6_K";
        default:
            fatal("stoke info: internal -- unknown ggml type %u (gguf_open "
                  "should already have rejected this)",
                  t);
    }
    return NULL; /* unreachable */
}

/* ===========================================================================
 * STRING KV formatting (task-5-brief appendix A10):
 *   "print on one line with newlines/tabs escaped (\n, \t); if longer than
 *    96 bytes, truncate to 96 and append <ellipsis>(<N> bytes total)"
 *
 * This implementer's concrete reading of that rule (A10 explicitly defers
 * the exact behavior to "check what your code actually prints and golden
 * that" -- see task-5-report.md self-review for the full reasoning):
 *   - Escape backslash too (not just \n/\t), so the escaping is lossless/
 *     unambiguous -- without it, a literal "\n" in the source text and an
 *     actual newline byte would print identically.
 *   - The "96 bytes" / truncation length is measured on the ESCAPED text
 *     (what actually appears on the line), not the raw pre-escape bytes.
 *   - "<N> bytes total" reports the RAW (pre-escape) byte length.
 *   - Truncation never splits an escape pair: it stops before adding a
 *     character whose escaped form would cross the 96-byte mark, rather
 *     than hard-cutting at byte 96.
 * ===========================================================================
 */

/* Escapes `s` (raw_len bytes) into `out` (caller-provided, >= 97 bytes: up
 * to 96 content bytes + NUL), stopping before the 97th output byte. Sets
 * *raw_consumed to how many input bytes were actually escaped -- if that's
 * less than raw_len, the output was truncated. */
static void escape_capped(const char* s, size_t raw_len, char* out,
                           size_t* raw_consumed) {
    size_t w = 0;
    size_t i = 0;
    for (; i < raw_len; i++) {
        unsigned char c = (unsigned char)s[i];
        int           two = (c == '\\' || c == '\n' || c == '\t');
        size_t        need = two ? 2 : 1;
        if (w + need > 96) break;
        if (c == '\\') {
            out[w++] = '\\';
            out[w++] = '\\';
        } else if (c == '\n') {
            out[w++] = '\\';
            out[w++] = 'n';
        } else if (c == '\t') {
            out[w++] = '\\';
            out[w++] = 't';
        } else {
            out[w++] = (char)c;
        }
    }
    out[w] = '\0';
    *raw_consumed = i;
}

static void print_string_kv_value(const char* val) {
    size_t raw_len = strlen(val);
    char   buf[97];
    size_t consumed;
    escape_capped(val, raw_len, buf, &consumed);
    if (consumed < raw_len) {
        printf("%s\xe2\x80\xa6(%zu bytes total)", buf, raw_len);
    } else {
        printf("%s", buf);
    }
}

/* ===========================================================================
 * `stoke info` body
 * ===========================================================================
 */

static void print_metadata(GgufFile* f) {
    u64 n = gguf_meta_count(f);
    printf("== metadata (%llu kv) ==\n", (unsigned long long)n);
    for (u64 i = 0; i < n; i++) {
        const char* key = gguf_meta_key(f, i);
        u32         type = gguf_meta_type(f, key);
        printf("[%3llu] %s (%s) = ", (unsigned long long)i, key,
               gguf_type_name(type));
        if (type == GGUF_TYPE_STRING) {
            print_string_kv_value(gguf_meta_str(f, key));
        } else if (type == GGUF_TYPE_ARRAY) {
            u32 elem = gguf_meta_arr_elem_type(f, key);
            u64 len = gguf_meta_arr_len(f, key);
            printf("[%s x %llu]", gguf_type_name(elem),
                   (unsigned long long)len);
        } else if (type == GGUF_TYPE_FLOAT32) {
            printf("%g", (double)gguf_meta_f32(f, key));
        } else if (type == GGUF_TYPE_FLOAT64) {
            /* No scalar FLOAT64 getter exists (task-5-brief A8 defines
             * none, and no KV in the target model is FLOAT64) -- fatal
             * loudly rather than mis-printing or silently skipping. */
            fatal("stoke info: key '%s' is FLOAT64 -- no scalar getter "
                  "for this type exists yet (see src/gguf.h)",
                  key);
        } else {
            /* every remaining GGUF_TYPE_* is an integer or BOOL, which
             * gguf_meta_i64 widens (task-5-brief A8.4) */
            printf("%lld", (long long)gguf_meta_i64(f, key));
        }
        printf("\n");
    }
}

static void print_tensors(GgufFile* f) {
    u64 n = gguf_tensor_count(f);
    printf("== tensors (%llu, file order) ==\n", (unsigned long long)n);
    for (u64 i = 0; i < n; i++) {
        const GgufTensor* t = gguf_tensor(f, i);
        printf("[%3llu] %s %s n_dims=%u dims=[", (unsigned long long)i,
               t->name, ggml_type_name(t->type), t->n_dims);
        for (u32 d = 0; d < t->n_dims; d++) {
            printf("%llu%s", (unsigned long long)t->dims[d],
                   (d + 1 < t->n_dims) ? "," : "");
        }
        printf("] offset=%llu nbytes=%llu\n", (unsigned long long)t->offset,
               (unsigned long long)t->nbytes);
    }
}

static void cmd_info(const char* path) {
    GgufFile* f = gguf_open(path);
    print_metadata(f);
    print_tensors(f);
    gguf_close(f);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    /* Windows CRT stdout defaults to text mode, which translates '\n' to
     * "\r\n" on write. task-5-brief appendix A10 requires byte-identical
     * output (specifically '\n' line endings) across Windows and WSL for
     * the golden diff to work at all -- force binary mode so this
     * translation never happens. */
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    if (argc < 2) {
        fatal("usage: stoke <command> [args] (commands: info <path.gguf>)");
    }
    if (strcmp(argv[1], "info") == 0) {
        if (argc < 3) {
            fatal("usage: stoke info <path.gguf>");
        }
        cmd_info(argv[2]);
        return 0;
    }
    fatal("unknown command '%s' (commands: info)", argv[1]);
    return 1; /* unreachable */
}
