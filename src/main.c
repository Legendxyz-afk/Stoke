/* src/main.c -- stoke CLI. Implements the `info` subcommand
 * (tests/golden_info.sh runs it) and `tokenize` (tests/tok_golden.sh runs
 * it, mirroring how tests/golden_info.sh runs `stoke info`). Later
 * milestones add run/serve/bench (see the plan's file structure note for
 * src/main.c).
 */
#include "common.h"

#include "gguf.h"
#include "tok.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

/* ===========================================================================
 * Name tables for `stoke info` output. Deliberately NOT part of gguf.h's
 * public API: the numeric type values are already exposed (GgufTensor.type,
 * gguf_meta_type()/gguf_meta_arr_elem_type()); mapping them to display
 * strings is presentation logic that belongs with the CLI that owns the
 * golden-output formatting rules, not with the parser.
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
 * STRING KV formatting:
 *   "print on one line with newlines/tabs escaped (\n, \t); if longer than
 *    96 bytes, truncate to 96 and append <ellipsis>(<N> bytes total)"
 *
 * This implementer's concrete reading of that rule (the exact escape/
 * truncation behavior was left to "check what your code actually prints
 * and golden that", so what follows is the concrete behavior chosen and
 * golden):
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
            /* No scalar FLOAT64 getter exists (no KV in the target model
             * is FLOAT64, so none was implemented) -- fatal loudly rather
             * than mis-printing or silently skipping. */
            fatal("stoke info: key '%s' is FLOAT64 -- no scalar getter "
                  "for this type exists yet (see src/gguf.h)",
                  key);
        } else {
            /* every remaining GGUF_TYPE_* is an integer or BOOL, which
             * gguf_meta_i64 widens */
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

/* ===========================================================================
 * `stoke tokenize` (`--special` added alongside the special-token parse
 * path): opens the GGUF, builds a Tok, reads the whole input file as raw
 * bytes, encodes it (add_special=false -- no BOS), and prints the ids as a
 * Python list "[a, b, c]\n", byte-identical to
 * tests/data/tok_corpus.ids.golden's format: a single trailing '\n' (not
 * the oracle CLI's own "\r\n" console quirk), comma-space separated, no
 * other output on stdout. Plain `tokenize` calls
 * tok_encode() (matches the oracle's `--no-parse-special`); `tokenize
 * --special` calls tok_encode_special() (matches the oracle's default,
 * parse-special ON). This is the mechanism tests/tok_golden.sh and
 * tests/tok_special_golden.sh run, mirroring how tests/golden_info.sh runs
 * `stoke info`.
 * ===========================================================================
 */

/* Reads the whole file at `path` into an owned buffer; *out_len receives
 * the byte count. Fatal on any I/O failure -- matches this project's
 * "every failure path is fatal(), no silent fallback" convention. `who`
 * names the calling command (passed straight through from
 * tokenize_cmd_setup's own `who`), so e.g. a `stoke tokenize --roundtrip`
 * I/O failure doesn't misreport itself as plain `stoke tokenize`. */
static char* read_whole_file(const char* path, const char* who, u64* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fatal("%s: cannot open '%s' -- %s", who, path, strerror(errno));
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        int e = errno;
        fclose(f);
        fatal("%s: cannot seek to end of '%s' -- %s", who, path, strerror(e));
    }
    long size = ftell(f);
    if (size < 0) {
        int e = errno;
        fclose(f);
        fatal("%s: ftell failed for '%s' -- %s", who, path, strerror(e));
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        int e = errno;
        fclose(f);
        fatal("%s: cannot rewind '%s' -- %s", who, path, strerror(e));
    }

    char*  buf = (char*)xmalloc((size_t)size);
    size_t got = fread(buf, 1, (size_t)size, f);
    if (got != (size_t)size) {
        fatal("%s: short read on '%s' (%zu of %ld bytes)", who, path, got,
              size);
    }
    fclose(f);

    *out_len = (u64)size;
    return buf;
}

/* Shared preamble for both cmd_tokenize and cmd_tokenize_roundtrip (they
 * used to duplicate this open/read/size-guard sequence verbatim): opens
 * the GGUF model, builds a self-contained Tok, closing the GgufFile right
 * away (Tok never retains it -- src/tok.h), reads the whole text file into
 * an owned buffer, and computes the ids cap per tok_encode/
 * tok_encode_special's "cap >= 3*len" contract, fataling if 3*len would
 * overflow a u32. `who` names the calling command for the size-guard's
 * fatal message. Caller owns the returned *out_tok (tok_free) and
 * *out_buf (free). */
static void tokenize_cmd_setup(const char* model_path, const char* text_path,
                                const char* who, Tok** out_tok, char** out_buf,
                                u64* out_len, u32* out_cap) {
    GgufFile* f = gguf_open(model_path);
    Tok*      t = tok_from_gguf(f);
    gguf_close(f); /* Tok is self-contained after construction (src/tok.h) */

    u64   len;
    char* buf = read_whole_file(text_path, who, &len);

    /* src/tok.h's tok_encode/tok_encode_special contract: cap >= 3*len is
     * always sufficient (the 3x, not 1x, matters only for pathological
     * all-invalid-UTF-8 input -- see that comment and src/tok.c's block
     * comment above utf8_decode_one for why; the same bound holds with
     * special-token matching folded in). */
    if (len > (u64)UINT32_MAX / 3) {
        fatal("%s: '%s' is %llu bytes -- too large for this CLI (3x ids "
              "cap must fit in u32)",
              who, text_path, (unsigned long long)len);
    }

    *out_tok = t;
    *out_buf = buf;
    *out_len = len;
    *out_cap = (u32)(len * 3);
}

static void cmd_tokenize(const char* model_path, const char* text_path,
                          bool special) {
    Tok*  t;
    char* buf;
    u64   len;
    u32   cap;
    tokenize_cmd_setup(model_path, text_path, "stoke tokenize", &t, &buf,
                        &len, &cap);

    u32* ids = (u32*)xcalloc(cap > 0 ? cap : 1, sizeof(u32));

    u32 n = special
                ? tok_encode_special(t, buf, len, ids, cap, /*add_special=*/false)
                : tok_encode(t, buf, len, ids, cap, /*add_special=*/false);

    printf("[");
    for (u32 i = 0; i < n; i++) {
        if (i > 0) printf(", ");
        printf("%u", ids[i]);
    }
    printf("]\n");

    free(ids);
    free(buf);
    tok_free(t);
}

/* `stoke tokenize --roundtrip <model> <text-file>` -- this mode is
 * optional ("optionally also add a real-corpus roundtrip mode to `stoke
 * tokenize` for extra assurance on real data"). The MANDATORY
 * decode(encode(x))==x assertion lives in tests/test_tok.c on the
 * synthetic vocab (always-run, no model needed); this mode adds the same
 * check against the REAL ~152K-entry vocab, which the synthetic fixture
 * can't exercise. Splits the file on '\n' (matching
 * tests/tok_fuzz.py's one-line-per-entry corpus) and reports match/
 * mismatch counts per line. A mismatch is NOT automatically a bug: lines
 * containing invalid/overlong UTF-8 are expected to differ, since
 * canonicalization is not lossless there by design (src/tok.c's block
 * comment above utf8_decode_one) -- this prints line numbers so a caller
 * can cross-check them against known-invalid corpus lines by hand. */
static void cmd_tokenize_roundtrip(const char* model_path,
                                    const char* text_path) {
    Tok*  t;
    char* buf;
    u64   len;
    u32   cap;
    tokenize_cmd_setup(model_path, text_path, "stoke tokenize --roundtrip",
                        &t, &buf, &len, &cap);
    if (cap == 0) cap = 1;
    u32*  ids = (u32*)xcalloc(cap, sizeof(u32));
    char* out = (char*)xcalloc(cap, 4); /* generous: decode's total output
        for one line can't exceed that line's own canonical byte count,
        itself <= 3x the line's raw length <= 3x len */

    u64 line_no = 0, n_match = 0, n_mismatch = 0;
    u64 line_start = 0;
    for (u64 i = 0; i <= len; i++) {
        if (i == len || buf[i] == '\n') {
            u64 line_len = i - line_start;
            line_no++;
            u32 n = tok_encode(t, buf + line_start, line_len, ids, cap, false);
            u32 w = 0;
            for (u32 k = 0; k < n; k++) {
                w += tok_decode(t, ids[k], out + w, (u32)((u64)cap * 4 - w));
            }
            bool ok = (w == line_len) &&
                      (line_len == 0 ||
                       memcmp(out, buf + line_start, line_len) == 0);
            if (ok) {
                n_match++;
            } else {
                n_mismatch++;
                fprintf(stderr,
                        "roundtrip: line %llu MISMATCH (original %llu "
                        "bytes, decoded %u bytes)\n",
                        (unsigned long long)line_no,
                        (unsigned long long)line_len, w);
            }
            line_start = i + 1;
        }
    }

    printf("roundtrip: %llu/%llu lines match exactly, %llu mismatched\n",
           (unsigned long long)n_match, (unsigned long long)line_no,
           (unsigned long long)n_mismatch);

    free(out);
    free(ids);
    free(buf);
    tok_free(t);
}

/* Checked stdout finalization before a successful exit. Every subcommand
 * writes exclusively via stdio's own buffered printf; without this, `stoke
 * info > out.txt` on a full disk (or any other deferred-write failure)
 * would exit 0 with a silently truncated file, since a buffered printf's
 * return value is not checked anywhere else in this file and the actual
 * write only happens later, at flush/close time. This is the one place in
 * src/ that used to swallow an error: flush stdout and check its error
 * indicator, fataling loudly (house style) instead of exiting 0 over a
 * failed write. */
static void finish_stdout_or_fatal(void) {
    if (fflush(stdout) != 0) {
        fatal("stoke: failed to flush stdout -- %s", strerror(errno));
    }
    if (ferror(stdout)) {
        fatal("stoke: stdout reported a write error");
    }
}

int main(int argc, char** argv) {
#ifdef _WIN32
    /* Windows CRT stdout defaults to text mode, which translates '\n' to
     * "\r\n" on write. Byte-identical output (specifically '\n' line
     * endings) is required across Windows and WSL for the golden diff to
     * work at all -- force binary mode so this translation never
     * happens. */
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    if (argc < 2) {
        fatal("usage: stoke <command> [args] (commands: info <path.gguf>, "
              "tokenize [--special | --roundtrip] <path.gguf> <text-file>)");
    }
    if (strcmp(argv[1], "info") == 0) {
        if (argc < 3) {
            fatal("usage: stoke info <path.gguf>");
        }
        cmd_info(argv[2]);
        finish_stdout_or_fatal();
        return 0;
    }
    if (strcmp(argv[1], "tokenize") == 0) {
        if (argc >= 3 && strcmp(argv[2], "--roundtrip") == 0) {
            if (argc < 5) {
                fatal("usage: stoke tokenize --roundtrip <path.gguf> "
                      "<text-file>");
            }
            cmd_tokenize_roundtrip(argv[3], argv[4]);
            finish_stdout_or_fatal();
            return 0;
        }
        bool special = false;
        int  model_argi = 2;
        if (argc >= 3 && strcmp(argv[2], "--special") == 0) {
            special = true;
            model_argi = 3;
        }
        if (argc < model_argi + 2) {
            fatal("usage: stoke tokenize [--special] <path.gguf> "
                  "<text-file>");
        }
        cmd_tokenize(argv[model_argi], argv[model_argi + 1], special);
        finish_stdout_or_fatal();
        return 0;
    }
    fatal("unknown command '%s' (commands: info, tokenize)", argv[1]);
    return 1; /* unreachable */
}
