/* src/main.c -- stoke CLI. Implements the `info` subcommand
 * (tests/golden_info.sh runs it) and `tokenize` (tests/tok_golden.sh runs
 * it, mirroring how tests/golden_info.sh runs `stoke info`). Later
 * milestones add run/serve/bench (see the plan's file structure note for
 * src/main.c).
 */
#include "common.h"

#include "bench.h"
#include "gguf.h"
#include "kv.h"
#include "model.h"
#include "pool.h"
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
        case GGUF_GGML_TYPE_Q8_0: return "Q8_0";
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

/* ===========================================================================
 * `stoke run` -- M2 op-spec design note section 7 CLI contract:
 *   stoke run <model.gguf> [--raw] [-n N] [-p "text" | -f file]
 *             [--dump dir] [--probe-pos N] [--timing]
 * `--raw`: tokenize with add_special=false, parse-special OFF, no chat
 * template (the parity gate's mode). Without `--raw`: wraps the input as a
 * single ChatML user message via tok_apply_chatml (smoke-tested only, not
 * gated -- M1 already oracle-proved the template itself). `--dump dir`
 * writes the per-layer post-block hidden state of the LAST INPUT position
 * (model.c's model_generate/model_forward_position do the actual writing).
 * `--probe-pos N` (M2 W3 brief addition, for the W4 near-tie waiver
 * protocol): after generation, prints the top-5 (id, logit) at generated
 * position N to stderr, 5 lines, `probe pos=N: id=<u32> logit=<%.6f>`.
 * ===========================================================================
 */

static void cmd_run(int argc, char** argv) {
    if (argc < 3) {
        fatal("usage: stoke run <model.gguf> [--raw] [-n N] "
              "[-p \"text\" | -f file] [--dump dir] [--probe-pos N] "
              "[--timing] [--threads N]");
    }
    const char* model_path = argv[2];
    bool        raw = false;
    long long   n_long = 64;
    /* 0 means "the pool's measured default" (6 -- see pool.h; ten threads
     * measured 15%% WORSE than six on this box). Threading changes no
     * arithmetic: matvec rows are independent, so output is bit-identical at
     * any count, asserted in tests/test_pool.c. */
    long long   threads_long = 0;
    const char* prompt_text = NULL;
    const char* prompt_file = NULL;
    const char* dump_dir = NULL;
    long long   probe_pos_long = -1;
    bool        report_timing = false;

    /* strtoll (not strtol): on this project's Windows toolchain (MSYS2
     * ucrt64, LLP64 data model) `long` is only 32 bits, so `(long)
     * UINT32_MAX` silently overflows a signed 32-bit long (implementation-
     * defined, observed wrapping to -1 here) -- which broke `-n 1` outright
     * (n_long > (long)UINT32_MAX read as `1 > -1`, always true). `long
     * long` is guaranteed >= 64 bits on both toolchains, so every cast
     * below is exact. */
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--raw") == 0) {
            raw = true;
        } else if (strcmp(argv[i], "-n") == 0) {
            if (i + 1 >= argc) fatal("stoke run: -n needs an argument");
            n_long = strtoll(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-p") == 0) {
            if (i + 1 >= argc) fatal("stoke run: -p needs an argument");
            prompt_text = argv[++i];
        } else if (strcmp(argv[i], "-f") == 0) {
            if (i + 1 >= argc) fatal("stoke run: -f needs an argument");
            prompt_file = argv[++i];
        } else if (strcmp(argv[i], "--dump") == 0) {
            if (i + 1 >= argc) {
                fatal("stoke run: --dump needs a directory argument");
            }
            dump_dir = argv[++i];
        } else if (strcmp(argv[i], "--timing") == 0) {
            report_timing = true;
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (i + 1 >= argc) fatal("stoke run: --threads needs an argument");
            threads_long = strtoll(argv[++i], NULL, 10);
            if (threads_long < 0 || threads_long > 1024) {
                fatal("stoke run: --threads %lld out of range [0,1024] "
                      "(0 = the measured default)", threads_long);
            }
        } else if (strcmp(argv[i], "--probe-pos") == 0) {
            if (i + 1 >= argc) {
                fatal("stoke run: --probe-pos needs an argument");
            }
            probe_pos_long = strtoll(argv[++i], NULL, 10);
        } else {
            fatal("stoke run: unknown argument '%s'", argv[i]);
        }
    }
    if (prompt_text != NULL && prompt_file != NULL) {
        fatal("stoke run: specify only one of -p/-f, not both");
    }
    if (prompt_text == NULL && prompt_file == NULL) {
        fatal("stoke run: one of -p \"text\" or -f file is required");
    }
    if (n_long <= 0 || n_long > (long long)UINT32_MAX) {
        fatal("stoke run: -n %lld must be a positive integer", n_long);
    }
    if (probe_pos_long < -1 || probe_pos_long > (long long)INT32_MAX) {
        fatal("stoke run: --probe-pos %lld out of range", probe_pos_long);
    }
    u32 n = (u32)n_long;
    i32 probe_pos = (i32)probe_pos_long;

    char*       file_buf = NULL;
    const char* text;
    u64         text_len;
    if (prompt_file != NULL) {
        file_buf = read_whole_file(prompt_file, "stoke run", &text_len);
        text = file_buf;
    } else {
        text = prompt_text;
        text_len = (u64)strlen(prompt_text);
    }
    if (text_len > (u64)UINT32_MAX / 3) {
        fatal("stoke run: prompt is %llu bytes -- too large for this CLI",
              (unsigned long long)text_len);
    }

    double    t_load_start = now_ms();
    GgufFile* gf = gguf_open(model_path);
    Tok*      tok = tok_from_gguf(gf);
    gguf_close(gf); /* Tok is self-contained after construction (src/tok.h) */

    /* cap sizing: tok_encode/tok_encode_special need cap >= 3*len (raw,
     * add_special=false); tok_apply_chatml needs cap >= 8 + 3*(role_len +
     * content_len) + 31 for one message (role_len=4 here) -- both
     * comfortably covered by 3*len + 64. */
    u32  cap = (u32)(text_len * 3) + 64;
    u32* prompt_ids = (u32*)xcalloc(cap, sizeof(u32));
    u32  n_prompt;
    if (raw) {
        n_prompt = tok_encode(tok, text, text_len, prompt_ids, cap,
                               /*add_special=*/false);
    } else {
        StokeMsg msg;
        msg.role = "user";
        msg.role_len = 4;
        msg.content = text;
        msg.content_len = text_len;
        n_prompt = tok_apply_chatml(tok, &msg, 1, prompt_ids, cap);
    }
    if (n_prompt == 0) {
        fatal("stoke run: prompt tokenized to zero ids");
    }

    /* Start the pool before any forward pass. Threading is a pure speed
     * change here -- matvec rows are independent, so the token stream is
     * bit-identical at any count (tests/test_pool.c) -- which is why this
     * needs no flag-gated opt-in and no parity re-run. */
    pool_start((u32)threads_long);

    Model* model = model_open(model_path);
    Kv*    kv = kv_open(model_n_layer(model), model_max_ctx(model),
                         model_n_head_kv(model), model_head_dim(model));
    double load_ms = now_ms() - t_load_start;

    u32*            out_ids = (u32*)xcalloc(n, sizeof(u32));
    ModelProbeEntry probe[5];
    for (int i = 0; i < 5; i++) {
        probe[i].id = 0;
        probe[i].logit = 0.0f;
    }

    i64         eos_id = (i64)tok_eos(tok);
    ModelTiming timing;
    u32 count = model_generate(model, kv, prompt_ids, n_prompt, n, eos_id,
                                out_ids, dump_dir, probe_pos, probe, &timing);

    /* stderr, ALWAYS -- stdout is byte-compared against the goldens and
     * against ref_forward.py, so a rate printed there would break the one
     * property the parity gate rests on. */
    if (report_timing) {
        double dec_s = timing.decode_ms / 1000.0;
        fprintf(stderr,
                "timing: load %.1f ms | prefill %u tok in %.1f ms | "
                "decode %u tok in %.1f ms",
                load_ms, timing.n_prefill, timing.prefill_ms,
                timing.n_decode, timing.decode_ms);
        /* A rate needs both a nonzero numerator and a nonzero denominator.
         * `-n 1` legitimately yields zero decode tokens, and printing 0.000
         * tok/s for it would be a measurement of nothing presented as a
         * measurement of something. */
        if (timing.n_decode > 0u && dec_s > 0.0) {
            fprintf(stderr, " (%.3f tok/s)", (double)timing.n_decode / dec_s);
        } else {
            fprintf(stderr, " (no decode-loop tokens -- no rate)");
        }
        fprintf(stderr, "\n");
    }

    for (u32 i = 0; i < count; i++) {
        printf("%u\n", out_ids[i]);
    }
    printf("---\n");

    u32   decode_cap = count * 256u + 16u;
    char* decoded = (char*)xcalloc(decode_cap, 1);
    u32   w = 0;
    for (u32 i = 0; i < count; i++) {
        w += tok_decode(tok, out_ids[i], decoded + w, decode_cap - w);
    }
    fwrite(decoded, 1, w, stdout);
    printf("\n");
    free(decoded);

    if (probe_pos >= 0) {
        if ((u32)probe_pos >= count) {
            fatal("stoke run: --probe-pos %d requested but only %u ids "
                  "were generated (generation stopped early)",
                  probe_pos, count);
        }
        for (int i = 0; i < 5; i++) {
            fprintf(stderr, "probe pos=%d: id=%u logit=%.6f\n", probe_pos,
                    probe[i].id, (double)probe[i].logit);
        }
    }

    free(out_ids);
    free(prompt_ids);
    kv_close(kv);
    model_close(model);
    tok_free(tok);
    free(file_buf);
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
              "tokenize [--special | --roundtrip] <path.gguf> <text-file>, "
              "run <path.gguf> [--raw] [-n N] [-p \"text\" | -f file] "
              "[--dump dir] [--probe-pos N] [--timing], "
              "bench --ops <path.gguf> [--reps N] [--jsonl])");
    }
    if (strcmp(argv[1], "bench") == 0) {
        /* The `--ops` spelling is mandated by the M3 forward note; it is a
         * required mode rather than a default so that later bench modes
         * (end-to-end tok/s, scenario replay) can be added without ever
         * silently changing what a bare `stoke bench` measures. */
        if (argc < 4 || strcmp(argv[2], "--ops") != 0) {
            fatal("usage: stoke bench --ops <path.gguf> [--reps N] [--jsonl]");
        }
        u32 reps  = 0;   /* 0 = adaptive, see bench.c Decision 3 */
        int jsonl = 0;
        for (int i = 4; i < argc; i++) {
            if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
                long v = strtol(argv[++i], NULL, 10);
                if (v <= 0) {
                    fatal("stoke bench: --reps must be positive, got '%s'",
                          argv[i]);
                }
                reps = (u32)v;
            } else if (strcmp(argv[i], "--jsonl") == 0) {
                jsonl = 1;
            } else {
                fatal("stoke bench: unexpected argument '%s'", argv[i]);
            }
        }
        bench_ops(argv[3], reps, jsonl);
        finish_stdout_or_fatal();
        return 0;
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
    if (strcmp(argv[1], "run") == 0) {
        cmd_run(argc, argv);
        finish_stdout_or_fatal();
        return 0;
    }
    fatal("unknown command '%s' (commands: info, tokenize, run)", argv[1]);
    return 1; /* unreachable */
}
