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
#include "sched.h"
#include "serve.h"
#include "server.h"
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
              "[--timing] [--threads N] [--kv f32|q8]");
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
    /* M4 gate item 4 needs the KV-ON/KV-OFF pair to be gradeable by the
     * ruling's section 6 rule, which compares LOGITS at the divergence --
     * and --probe-pos is here, not in `batch`. So the dtype switch is here
     * too, rather than teaching `batch` a second probe implementation. */
    KvDtype     kv_dt = KV_F32;

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
        } else if (strcmp(argv[i], "--kv") == 0) {
            if (i + 1 >= argc) fatal("stoke run: --kv needs f32 or q8");
            const char* w = argv[++i];
            if (strcmp(w, "f32") == 0)     kv_dt = KV_F32;
            else if (strcmp(w, "q8") == 0) kv_dt = KV_Q8_0;
            else fatal("stoke run: --kv '%s' -- expected f32 or q8", w);
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
    /* One private store, exactly what kv_open builds -- spelled out here only
     * so the dtype can be chosen. At KV_F32 this is the same allocation and
     * the same arithmetic kv_open would have made, which is what keeps every
     * parity artifact produced through this path comparable (checked against
     * the archived m3 gate binary, not assumed). */
    KvStore* store = kv_store_open(model_n_layer(model),
                                   kv_blocks_for_ctx(model_max_ctx(model)),
                                   model_n_head_kv(model),
                                   model_head_dim(model), kv_dt);
    Kv* kv = kv_open_in(store, model_max_ctx(model));
    if (kv == NULL) {
        fatal("stoke run: could not admit a sequence of max_ctx %u",
              model_max_ctx(model));
    }
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
    kv_store_close(store); /* kv_open_in does not own the store */
    model_close(model);
    tok_free(tok);
    free(file_buf);
}

/* ===========================================================================
 * `stoke batch` -- M4's static-batch driver.
 *
 *   stoke batch <model.gguf> [--raw] [-n N] (-p "text" | -f file)...
 *               [--seqs N] [--ctx N] [--threads N] [--timing]
 *
 * WHY A SUBCOMMAND AND NOT A TEST. Gate items 2, 3 and 5 (ruling section 8.4)
 * all need N sequences driven to completion against the REAL model: token
 * identity across batch widths, prefix dedup firing, and the aggregate curve.
 * None of that belongs in the per-commit matrix -- a real-model batch at N=16
 * costs minutes, and under ASan far more -- so the harness has to be a
 * command the gate scripts run once and record.
 *
 * OUTPUT SHAPE IS THE COMPARISON. Each sequence prints a `== seq i ==` header
 * followed by EXACTLY what `stoke run` prints for that prompt: the ids one per
 * line, `---`, then the decoded text. So the identity gate is a byte-compare
 * of a block against a whole `stoke run` stdout, with no reformatting step in
 * between that could paper over a difference.
 *
 * `--seqs N` beyond the number of prompts given cycles through them, which is
 * how the curve reaches N=16 from a five-prompt corpus. Every sequence gets
 * the SAME per-slot ctx (playbook LOCK 2: "all N same ctx") so the pool is
 * sized by one multiplication and the curve's memory footprint is exactly
 * N x ctx.
 * ===========================================================================
 */
#define BATCH_MAX_PROMPTS 64
#define BATCH_MAX_SEQS    256

/* ---------------------------------------------------------------------------
 * `stoke bench --serve` -- M5's in-process serving driver.
 *
 * WHAT IT MEASURES, AND WHY THESE THREE. Gate item [2] is an AGGREGATE tok/s
 * bar over N concurrent requests; [3] is an inter-token-latency bound (no
 * decode stall > 250 ms while a prefill co-runs); and Fable's 08-02 addition
 * asks for p95 request completion, recorded not gated, because that is the
 * agent-shaped number the human-shaped ITL bar does not capture.
 *
 * > ITL IS TIMED AT THE CONSUMER, ONE STEP'S GRANULARITY. Every step, the
 * driver polls every request and stamps whatever ids arrived with the time it
 * saw them. That is exactly the granularity the bar is about -- a decode stall
 * IS a step that carried no row for that request -- and it needs nothing from
 * the scheduler except the ids it already emits. Timing inside sched.c would
 * measure the scheduler's opinion of when a token was ready rather than when a
 * consumer could have had it.
 *
 * > THIS PRINTS NUMBERS; IT DOES NOT GRADE THEM. The bars are met on a quiet
 * box under the interleaved protocol, against a referent measured in the same
 * sitting -- this box has two error sources (E-core migration, thermal drift)
 * and neither is visible from inside one run. A single invocation of this is an
 * observation, not a result.
 * ------------------------------------------------------------------------- */
static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

static double pct(const double* v, u32 n, double p) {
    if (n == 0) return 0.0;
    u32 i = (u32)(p * (double)(n - 1) + 0.5);
    return v[i];
}

static void cmd_bench_serve(int argc, char** argv) {
    if (argc < 4) {
        fatal("usage: stoke bench --serve <model.gguf> [--raw] [-n N] "
              "(-p text | -f file)... [--seqs N] [--ctx N] [--threads N] "
              "[--kv f32|q8] [--no-dedup] [--ids <file>]");
    }
    const char* model_path = argv[3];
    bool        raw = false;
    /* ▶ PREFIX DEDUP IS ON BY DEFAULT ON THE SERVING PATH (2026-08-26, Kabir's
     * call). It is bit-exact -- gated against a fixture that can SEE a dedup
     * defect, `run_m5_dedup_verify.ps1` green on four legs -- and on the S3
     * fanout it is worth 2.284x aggregate and 9.8x on ITL p95, removing 73% of
     * prefill rows. The engine that set M0's S3 bar has the equivalent on by
     * default (`--slot-prompt-similarity` 0.10), so OFF was the setting that
     * made stoke unlike its referent.
     *
     * `--dedup` is still ACCEPTED and still means on, so every banked runner,
     * report and command line from before the flip reproduces verbatim.
     * `--no-dedup` is how you turn it off, and the measurement runners that
     * mean OFF now say so out loud rather than relying on an absent flag. */
    bool        dedup = true;
    const char* ids_path = NULL;
    long long   n_long = 64, threads_long = 0, seqs_long = 0, ctx_long = 0;
    long long   stagger_long = 0, pool_blocks_long = 0;
    KvDtype     kv_dt = KV_F32;
    const char* src[BATCH_MAX_PROMPTS];
    bool        src_is_file[BATCH_MAX_PROMPTS];
    u32         n_src = 0;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--raw") == 0) {
            raw = true;
        } else if (strcmp(argv[i], "--dedup") == 0) {
            dedup = true;   /* the default since 2026-08-26; kept so that every
                             * pre-flip command line still means what it said */
        } else if (strcmp(argv[i], "--no-dedup") == 0) {
            dedup = false;
        } else if (strcmp(argv[i], "--pool-blocks") == 0) {
            /* ▶ WHY THIS EXISTS: gate item [6] wants preempt-losslessness shown
             * on a FORCED-PREEMPTION REAL-MODEL trace, and without this the
             * driver cannot produce one. The pool is `n_seq * blocks(ctx)`, so
             * every sequence can always reach its own context and decode-time
             * exhaustion is UNREACHABLE -- which is why every serve measurement
             * this project has banked reports preemptions 0. Squeezing the pool
             * below `n_seq * blocks(prompt + n_gen)` is what makes the
             * preemption path execute at all.
             *
             * A MEASUREMENT KNOB, like STOKE_SCHED_CHUNK: the default is the
             * derived capacity and the header line always prints the effective
             * value, so a squeezed run cannot be mistaken for a normal one. */
            if (i + 1 >= argc)
                fatal("stoke bench --serve: --pool-blocks needs a count");
            pool_blocks_long = strtoll(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--ids") == 0) {
            if (i + 1 >= argc)
                fatal("stoke bench --serve: --ids needs a file path");
            ids_path = argv[++i];
        } else if (strcmp(argv[i], "--kv") == 0) {
            if (i + 1 >= argc) fatal("stoke bench --serve: --kv needs f32 or q8");
            const char* w = argv[++i];
            if (strcmp(w, "f32") == 0)     kv_dt = KV_F32;
            else if (strcmp(w, "q8") == 0) kv_dt = KV_Q8_0;
            else fatal("stoke bench --serve: unknown --kv value, want f32 or q8");
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--seqs") == 0 ||
                   strcmp(argv[i], "--ctx") == 0 ||
                   strcmp(argv[i], "--stagger") == 0 ||
                   strcmp(argv[i], "--threads") == 0) {
            const char* flag = argv[i];
            if (i + 1 >= argc)
                fatal("stoke bench --serve: %s needs an argument", flag);
            long long v = strtoll(argv[++i], NULL, 10);
            if (strcmp(flag, "-n") == 0)             n_long = v;
            else if (strcmp(flag, "--seqs") == 0)    seqs_long = v;
            else if (strcmp(flag, "--ctx") == 0)     ctx_long = v;
            else if (strcmp(flag, "--stagger") == 0) stagger_long = v;
            else                                     threads_long = v;
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "-f") == 0) {
            bool is_file = (argv[i][1] == 'f');
            if (i + 1 >= argc)
                fatal("stoke bench --serve: %s needs an argument", argv[i]);
            if (n_src == BATCH_MAX_PROMPTS)
                fatal("stoke bench --serve: at most %d prompts", BATCH_MAX_PROMPTS);
            src[n_src] = argv[++i];
            src_is_file[n_src] = is_file;
            n_src++;
        } else {
            fatal("stoke bench --serve: unknown argument '%s'", argv[i]);
        }
    }
    if (n_src == 0)
        fatal("stoke bench --serve: at least one -p or -f is required");
    if (n_long <= 0 || n_long > 65536)
        fatal("stoke bench --serve: -n %lld out of range (1..65536)", n_long);
    if (seqs_long < 0 || seqs_long > BATCH_MAX_SEQS)
        fatal("stoke bench --serve: --seqs %lld out of range [0,%d]",
              seqs_long, BATCH_MAX_SEQS);

    u32 n_gen = (u32)n_long;
    u32 n_seq = (seqs_long > 0) ? (u32)seqs_long : n_src;

    GgufFile* gf = gguf_open(model_path);
    Tok*      tok = tok_from_gguf(gf);
    gguf_close(gf);

    u32* ids[BATCH_MAX_PROMPTS];
    u32  n_ids[BATCH_MAX_PROMPTS];
    u32  longest = 0;
    for (u32 p = 0; p < n_src; p++) {
        char*       file_buf = NULL;
        const char* text;
        u64         text_len;
        if (src_is_file[p]) {
            file_buf = read_whole_file(src[p], "stoke bench --serve", &text_len);
            text = file_buf;
        } else {
            text = src[p];
            text_len = (u64)strlen(src[p]);
        }
        u32 cap = (u32)(text_len * 3) + 64;
        ids[p] = (u32*)xcalloc(cap, sizeof(u32));
        if (raw) {
            n_ids[p] = tok_encode(tok, text, text_len, ids[p], cap, false);
        } else {
            StokeMsg msg;
            msg.role = "user";
            msg.role_len = 4;
            msg.content = text;
            msg.content_len = text_len;
            n_ids[p] = tok_apply_chatml(tok, &msg, 1, ids[p], cap);
        }
        if (n_ids[p] == 0)
            fatal("stoke bench --serve: prompt %u tokenized to zero ids", p);
        if (n_ids[p] > longest) longest = n_ids[p];
        free(file_buf);
    }

    pool_start((u32)threads_long);
    Model* model = model_open(model_path);

    u64 need_ctx = (u64)longest + (u64)n_gen;
    u32 ctx = (ctx_long > 0) ? (u32)ctx_long : (u32)need_ctx;
    if ((u64)ctx < need_ctx)
        fatal("stoke bench --serve: --ctx %u too small for %u + %u", ctx,
              longest, n_gen);
    if (ctx > model_max_ctx(model))
        fatal("stoke bench --serve: --ctx %u exceeds max_ctx %u", ctx,
              model_max_ctx(model));

    /* THE POOL IS SIZED AS THE STATIC BATCH PATH SIZES IT, deliberately: this
     * measurement is meant to be comparable with `stoke batch --seqs N`, and a
     * scheduler that won by being given more memory would not have won. */
    u32      capacity = n_seq * kv_blocks_for_ctx(ctx);
    bool     pool_squeezed = false;
    if (pool_blocks_long > 0) {
        if (pool_blocks_long > (long long)capacity) {
            fatal("stoke bench --serve: --pool-blocks %lld exceeds the derived "
                  "capacity %u -- a LARGER pool changes nothing, because no "
                  "sequence may pass --ctx",
                  pool_blocks_long, capacity);
        }
        pool_squeezed = ((u32)pool_blocks_long != capacity);
        capacity = (u32)pool_blocks_long;
    }
    KvStore* store = kv_store_open(model_n_layer(model), capacity,
                                   model_n_head_kv(model),
                                   model_head_dim(model), kv_dt);
    ServeHost* host = serve_host_open(model);
    SchedHost  sh = serve_host(host);
    Sched*     s = sched_open(store, ctx, &sh, dedup);

    double* t_tok = (double*)xcalloc((size_t)n_seq * n_gen, sizeof(double));
    /* Per token: was a prefill chunk co-running in the step that produced it?
     * This is what makes item [3] gradeable as written -- see the step loop. */
    unsigned char* tok_pf = (unsigned char*)xcalloc((size_t)n_seq * n_gen, 1);
    u64            prev_chunks = 0;
    u32*    got = (u32*)xcalloc(n_seq, sizeof(u32));
    u32*    out_ids = (u32*)xcalloc((size_t)n_seq * n_gen, sizeof(u32));
    double* t_done = (double*)xcalloc(n_seq, sizeof(double));

    /* ARRIVALS, AND WHY THEY ARE NOT ALL AT ZERO BY DEFAULT-OR-NEVER.
     *
     * --stagger 0 submits everything at once: that is the throughput shape,
     * and it is what gate item [2]'s aggregate bar is about. But it CANNOT
     * measure gate item [3], which bounds a decode stall "during co-running
     * prefill" -- with every request arriving at t=0 every prefill happens
     * before the decode set matures, so no prefill ever co-runs with a mature
     * one and the ITL max is measuring something else entirely. --stagger K
     * submits request i after K steps, so a prompt lands while decodes are in
     * flight. [3] needs a staggered run; [2] wants an unstaggered one. */
    u32    submitted = 0;
    double t0 = now_ms();

    u64 guard = 0;
    u64 guard_max = (u64)n_seq * n_gen * 64ull + 100000ull;
    for (;;) {
        while (submitted < n_seq &&
               (u64)submitted * (u64)stagger_long <= guard) {
            u32          p = submitted % n_src;
            SchedRequest r;
            r.id = 1000 + submitted;
            r.prompt_ids = ids[p];
            r.n_prompt = n_ids[p];
            r.max_tokens = n_gen;
            r.eos_id = -1;
            if (!sched_submit(s, &r))
                fatal("stoke bench --serve: request %u refused -- the pool "
                      "cannot hold prompt %u (%u tokens) plus -n %u",
                      submitted, p, n_ids[p], n_gen);
            submitted++;
        }
        if (!sched_pending(s) && submitted >= n_seq) break;
        if (guard++ >= guard_max) break;
        sched_step(s);
        /* ▶ DID THIS STEP CARRY PREFILL ROWS? Gate item [3] is worded "no decode
         * stall >250ms DURING CO-RUNNING PREFILL", and a pooled ITL distribution
         * cannot answer that: it mixes the steady-state cadence (a throughput
         * quantity, graded by item [2]) with the stall a decode suffers when a
         * prefill chunk shares its step (the quantity [3] names). The chunk
         * counter moving across a step is exactly "a prefill co-ran in it". */
        SchedStats st_now;
        sched_stats(s, &st_now);
        bool step_had_prefill = (st_now.prefill_chunks > prev_chunks);
        prev_chunks = st_now.prefill_chunks;
        double t = now_ms();
        for (u32 i = 0; i < n_seq; i++) {
            SchedFinish f;
            u32         before = got[i];
            got[i] += sched_poll(s, 1000 + i,
                                 out_ids + (size_t)i * n_gen + got[i],
                                 n_gen - got[i], &f);
            for (u32 k = before; k < got[i]; k++) {
                t_tok[(size_t)i * n_gen + k] = t;
                tok_pf[(size_t)i * n_gen + k] = step_had_prefill ? 1u : 0u;
            }
            if (f != SCHED_FINISH_NONE && t_done[i] == 0.0) t_done[i] = t;
        }
    }
    if (guard >= guard_max)
        fatal("stoke bench --serve: step budget exhausted -- livelock");
    double t1 = now_ms();

    SchedStats st;
    sched_stats(s, &st);

    u64 total_ids = 0;
    for (u32 i = 0; i < n_seq; i++) total_ids += got[i];

    /* ITL: gaps between CONSECUTIVE ids of ONE request, pooled. The first id is
     * excluded -- the gap before it is prefill, a different quantity, and
     * including it is what would make an ITL percentile look good by averaging
     * one long wait into many short ones. */
    u32     n_gap = 0;
    double* gaps = (double*)xcalloc((size_t)n_seq * n_gen, sizeof(double));
    /* THE SAME GAPS, SPLIT BY WHAT THE STEP WAS DOING. A gap is attributed to
     * the step that produced its ENDING token: that is the step the decode
     * waited through. `stall` is item [3]'s subject; `pure` is the cadence. */
    u32     n_stall = 0, n_pure = 0;
    double* stall = (double*)xcalloc((size_t)n_seq * n_gen, sizeof(double));
    double* pure  = (double*)xcalloc((size_t)n_seq * n_gen, sizeof(double));
    for (u32 i = 0; i < n_seq; i++) {
        for (u32 k = 1; k < got[i]; k++) {
            double g = t_tok[(size_t)i * n_gen + k] -
                       t_tok[(size_t)i * n_gen + k - 1];
            gaps[n_gap++] = g;
            if (tok_pf[(size_t)i * n_gen + k]) stall[n_stall++] = g;
            else                               pure[n_pure++]   = g;
        }
    }
    qsort(gaps, n_gap, sizeof(double), cmp_double);
    qsort(stall, n_stall, sizeof(double), cmp_double);
    qsort(pure, n_pure, sizeof(double), cmp_double);

    double* comp = (double*)xcalloc(n_seq, sizeof(double));
    for (u32 i = 0; i < n_seq; i++) comp[i] = t_done[i] - t0;
    qsort(comp, n_seq, sizeof(double), cmp_double);

    /* ===================================================================
     * THE GATE NUMBER IS M0'S NUMBER, COMPUTED M0'S WAY.
     *
     * Gate item [2]'s bars -- 17.7 / 14.2 / 10.8 / 16.8 -- are cells from
     * the M0 baseline table, and that table's `aggregate_decode_tps` is
     *
     *     sum over clients of (n_tokens - 1)
     *     -----------------------------------------------------------
     *     max(last-token wallclock) - min(first-token wallclock)
     *
     * (`bench/bench.py`): DECODE-only ids -- each client's first id is its
     * TTFT, a different quantity -- over the span from the earliest first
     * token ANYWHERE to the latest last token anywhere. It is not a
     * per-client average and it is not end-to-end throughput: the leading
     * prefill sits outside the span, while a later arrival's prefill sits
     * inside it and is charged in full.
     *
     * This driver used to print `total ids / total wall`, which is neither.
     * On S2's shape that is not a rounding difference -- the prompt is 512
     * tokens against 128 generated, so a metric that charges all prefill to
     * a decode bar is measuring mostly prefill. `stoke batch --timing`
     * already reports decode-only (`n_decode / decode_ms`), so the fixed
     * number here is also what makes the static referent comparable.
     *
     * BOTH are printed. End-to-end is the honest what-a-user-waited figure
     * and it is the one to read when TTFT is the story; it is simply not
     * the figure the bars were drawn from.
     * =================================================================== */
    double m0_first = 0.0, m0_last = 0.0;
    bool   m0_any = false;
    u64    decode_ids = 0;
    for (u32 i = 0; i < n_seq; i++) {
        if (got[i] == 0) continue;
        double f = t_tok[(size_t)i * n_gen];
        double l = t_tok[(size_t)i * n_gen + got[i] - 1];
        if (!m0_any || f < m0_first) m0_first = f;
        if (!m0_any || l > m0_last) m0_last = l;
        m0_any = true;
        decode_ids += got[i] - 1;
    }
    double m0_span_s = m0_any ? (m0_last - m0_first) / 1000.0 : 0.0;
    double m0_agg = (m0_span_s > 0.0) ? (double)decode_ids / m0_span_s : 0.0;

    /* TTFT, so a missed bar is attributable rather than merely observed: on
     * this shape prefill and decode are separable costs and the M0 table
     * carries both columns. */
    double* ttft = (double*)xcalloc(n_seq, sizeof(double));
    u32     n_ttft = 0;
    for (u32 i = 0; i < n_seq; i++)
        if (got[i] > 0) ttft[n_ttft++] = t_tok[(size_t)i * n_gen] - t0;
    qsort(ttft, n_ttft, sizeof(double), cmp_double);

    double wall_s = (t1 - t0) / 1000.0;
    printf("serve: %u seqs x %u gen over %u prompt(s), ctx %u, kv %s, "
           "%u pool blocks%s, dedup %s\n",
           n_seq, n_gen, n_src, ctx, (kv_dt == KV_Q8_0) ? "q8" : "f32",
           capacity, pool_squeezed ? " (SQUEEZED by --pool-blocks)" : "",
           dedup ? "on" : "OFF");
    printf("  AGGREGATE(M0) %.3f tok/s  = %llu decode ids / %.1f ms span "
           "(first token to last)\n",
           m0_agg, (unsigned long long)decode_ids, m0_last - m0_first);
    printf("  end-to-end %.3f tok/s  = %llu ids / %.1f ms wall "
           "(all prefill charged)\n",
           (wall_s > 0.0) ? (double)total_ids / wall_s : 0.0,
           (unsigned long long)total_ids, t1 - t0);
    printf("  TTFT p50 %.0f  p95 %.0f  max %.0f ms\n",
           pct(ttft, n_ttft, 0.50), pct(ttft, n_ttft, 0.95),
           n_ttft ? ttft[n_ttft - 1] : 0.0);
    printf("  ITL  p50 %.1f  p95 %.1f  max %.1f ms  (n=%u)\n",
           pct(gaps, n_gap, 0.50), pct(gaps, n_gap, 0.95),
           n_gap ? gaps[n_gap - 1] : 0.0, n_gap);
    /* ▶ THE SPLIT, AND WHICH LINE GRADES ITEM [3].
     *
     * The pooled line above is ONE distribution over two populations, and
     * reading a percentile off it answers neither question cleanly: its median
     * is the cadence, its tail is the stalls, and where the boundary falls
     * depends on HOW MANY prefill chunks the run happened to have. That is not
     * a bound being met or missed -- it is a mixing ratio.
     *
     *   stall   gaps whose step also carried a prefill chunk. THIS is item
     *           [3]'s "decode stall during co-running prefill".
     *   cadence gaps in pure-decode steps. A throughput quantity: it is
     *           N / decode-rate, and item [2] already grades that. */
    printf("  ITL  stall   p50 %.1f  p95 %.1f  max %.1f ms  (n=%u, %.1f%% of "
           "gaps)  <- gate [3]\n",
           pct(stall, n_stall, 0.50), pct(stall, n_stall, 0.95),
           n_stall ? stall[n_stall - 1] : 0.0, n_stall,
           n_gap ? 100.0 * (double)n_stall / (double)n_gap : 0.0);
    printf("  ITL  cadence p50 %.1f  p95 %.1f  max %.1f ms  (n=%u)\n",
           pct(pure, n_pure, 0.50), pct(pure, n_pure, 0.95),
           n_pure ? pure[n_pure - 1] : 0.0, n_pure);
    printf("  completion p50 %.0f  p95 %.0f  max %.0f ms\n",
           pct(comp, n_seq, 0.50), pct(comp, n_seq, 0.95),
           n_seq ? comp[n_seq - 1] : 0.0);
    printf("  steps %llu | rows %llu | prefill chunks %llu | preemptions %llu "
           "(%llu prefill yields)\n",
           (unsigned long long)st.steps,
           (unsigned long long)serve_host_rows(host),
           (unsigned long long)st.prefill_chunks,
           (unsigned long long)st.preemptions,
           (unsigned long long)st.prefill_yields);
    /* ▶ WHAT THE WORKLOAD ACTUALLY NEEDED, so a pressure claim stops being an
     * arithmetic argument. `--pool-blocks` forces preemption by squeezing the
     * pool below the workload's residency -- but "below" was, until this line,
     * DERIVED (blocks(prompt+n_gen) per sequence, less whatever adoption
     * shares), and a derivation is exactly the kind of thing that is wrong by
     * one block-rounding and reports a gate arm that never pressured anything.
     * Run the shape once at the derived pool, read the peak, and every squeezed
     * pool below it is forced to preempt BY MEASUREMENT. The peak is sampled
     * inside the step at maximum demand, not here -- see SchedStats. */
    printf("  pool: peak %u of %u blocks used (%.1f%%), %u free at end\n",
           st.peak_blocks_used, capacity,
           capacity ? 100.0 * (double)st.peak_blocks_used / (double)capacity
                    : 0.0,
           st.free_blocks);
    /* ALWAYS PRINTED WHEN --dedup IS ON, and for `stoke batch --dedup`'s
     * reason: "dedup was enabled" and "dedup fired" are different claims, and a
     * gate that cannot tell them apart passes while testing nothing. The
     * denominator is summed over ADMISSIONS, so a preempted request that
     * re-adopts counts in both -- which is what makes the ratio readable under
     * pressure instead of exceeding 100%. */
    /* ▶ PRINTED IN BOTH DIRECTIONS SINCE THE 2026-08-26 DEFAULT FLIP. While
     * dedup was opt-in, an absent line meant "off"; now an absent line would
     * mean nothing at all, and a log that cannot say which run it was is not a
     * measurement record. The OFF wording carries no "N of M" so the runners'
     * `dedup: (\d+) of` anti-vacuity regexes still match only the ON case. */
    if (dedup) {
        printf("  dedup: %llu of %llu admitted prompt positions adopted\n",
               (unsigned long long)st.adopted_positions,
               (unsigned long long)st.admitted_prompt_positions);
    } else {
        printf("  dedup: DISABLED by --no-dedup (%llu admitted prompt "
               "positions, none adopted)\n",
               (unsigned long long)st.admitted_prompt_positions);
    }
    /* ANTI-VACUITY, printed rather than assumed: a run where every request
     * generated nothing would still report a tidy aggregate of 0. */
    if (total_ids != (u64)n_seq * n_gen) {
        printf("  ! only %llu of %llu expected ids -- read nothing else here\n",
               (unsigned long long)total_ids,
               (unsigned long long)((u64)n_seq * n_gen));
    }

    /* ▶ THE STREAMS, TO A FILE, IN `stoke batch`'s EXACT OUTPUT SHAPE.
     *
     * A FILE rather than stdout because this driver's stats already own stdout
     * and run_m5_bars.ps1 parses them -- mixing 2,400 ids into that stream would
     * make the correctness artifact and the measurement artifact the same
     * artifact, and a regex is a poor place to keep them apart. The shape is
     * `batch`'s so the two drivers' outputs are directly comparable: same
     * header, ids one per line, `---`, then the decoded text. */
    if (ids_path) {
        FILE* fh = fopen(ids_path, "w");
        if (!fh) fatal("stoke bench --serve: cannot open --ids file '%s'", ids_path);
        for (u32 i = 0; i < n_seq; i++) {
            const u32* row = out_ids + (size_t)i * n_gen;
            fprintf(fh, "== seq %u ==\n", i);
            for (u32 k = 0; k < got[i]; k++) fprintf(fh, "%u\n", row[k]);
            fprintf(fh, "---\n");
            u32   dcap = got[i] * 256u + 16u;
            char* dec = (char*)xcalloc(dcap, 1);
            u32   w = 0;
            for (u32 k = 0; k < got[i]; k++)
                w += tok_decode(tok, row[k], dec + w, dcap - w);
            fwrite(dec, 1, w, fh);
            fprintf(fh, "\n");
            free(dec);
        }
        /* Checked, for finish_stdout_or_fatal's reason: a stream that fails to
         * flush would otherwise leave a SHORT file that compares as a DIFFER
         * and reads as an engine bug. */
        if (fclose(fh) != 0)
            fatal("stoke bench --serve: writing '%s' failed", ids_path);
    }

    free(ttft);
    free(stall);
    free(pure);
    free(tok_pf);
    free(gaps);
    free(comp);
    free(t_tok);
    free(got);
    free(out_ids);
    free(t_done);
    sched_close(s);
    serve_host_close(host);
    kv_store_close(store);
    model_close(model);
    pool_stop();
    for (u32 p = 0; p < n_src; p++) free(ids[p]);
    tok_free(tok);
}

static void cmd_batch(int argc, char** argv) {
    if (argc < 3) {
        fatal("usage: stoke batch <model.gguf> [--raw] [-n N] "
              "(-p \"text\" | -f file)... [--seqs N] [--ctx N] "
              "[--threads N] [--timing] [--dedup] [--kv f32|q8]");
    }
    const char* model_path = argv[2];
    bool        raw = false;
    long long   n_long = 64, threads_long = 0, seqs_long = 0, ctx_long = 0;
    bool        report_timing = false;
    bool        dedup = false;
    KvDtype     kv_dt = KV_F32;   /* M2's representation stays the default */

    const char* src[BATCH_MAX_PROMPTS];
    bool        src_is_file[BATCH_MAX_PROMPTS];
    u32         n_src = 0;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--raw") == 0) {
            raw = true;
        } else if (strcmp(argv[i], "--timing") == 0) {
            report_timing = true;
        } else if (strcmp(argv[i], "--dedup") == 0) {
            dedup = true;
        } else if (strcmp(argv[i], "--kv") == 0) {
            if (i + 1 >= argc) fatal("stoke batch: --kv needs f32 or q8");
            const char* w = argv[++i];
            if (strcmp(w, "f32") == 0)      kv_dt = KV_F32;
            else if (strcmp(w, "q8") == 0)  kv_dt = KV_Q8_0;
            else fatal("stoke batch: --kv '%s' -- expected f32 or q8", w);
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--seqs") == 0 ||
                   strcmp(argv[i], "--ctx") == 0 ||
                   strcmp(argv[i], "--threads") == 0) {
            const char* flag = argv[i];
            if (i + 1 >= argc) fatal("stoke batch: %s needs an argument", flag);
            long long v = strtoll(argv[++i], NULL, 10);
            if (strcmp(flag, "-n") == 0)             n_long = v;
            else if (strcmp(flag, "--seqs") == 0)    seqs_long = v;
            else if (strcmp(flag, "--ctx") == 0)     ctx_long = v;
            else                                     threads_long = v;
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "-f") == 0) {
            bool is_file = (argv[i][1] == 'f');
            if (i + 1 >= argc) {
                fatal("stoke batch: %s needs an argument", argv[i]);
            }
            if (n_src == BATCH_MAX_PROMPTS) {
                fatal("stoke batch: at most %d prompts", BATCH_MAX_PROMPTS);
            }
            src[n_src] = argv[++i];
            src_is_file[n_src] = is_file;
            n_src++;
        } else {
            fatal("stoke batch: unknown argument '%s'", argv[i]);
        }
    }

    if (n_src == 0) {
        fatal("stoke batch: at least one -p \"text\" or -f file is required");
    }
    if (n_long <= 0 || n_long > (long long)UINT32_MAX) {
        fatal("stoke batch: -n %lld must be a positive integer", n_long);
    }
    if (threads_long < 0 || threads_long > 1024) {
        fatal("stoke batch: --threads %lld out of range [0,1024] "
              "(0 = the measured default)", threads_long);
    }
    if (seqs_long < 0 || seqs_long > BATCH_MAX_SEQS) {
        fatal("stoke batch: --seqs %lld out of range [0,%d] (0 = one per "
              "prompt given)", seqs_long, BATCH_MAX_SEQS);
    }
    if (ctx_long < 0 || ctx_long > (long long)UINT32_MAX) {
        fatal("stoke batch: --ctx %lld out of range", ctx_long);
    }
    u32 n_gen = (u32)n_long;
    u32 n_seq = (seqs_long > 0) ? (u32)seqs_long : n_src;

    double t_load_start = now_ms();
    GgufFile* gf = gguf_open(model_path);
    Tok*      tok = tok_from_gguf(gf);
    gguf_close(gf);

    /* Tokenize each distinct prompt once. Sequences that share a prompt share
     * the id array too -- it is borrowed and read-only, and two slots running
     * the same prompt is exactly the workload prefix dedup exists for. */
    u32* ids[BATCH_MAX_PROMPTS];
    u32  n_ids[BATCH_MAX_PROMPTS];
    u32  longest = 0;
    for (u32 p = 0; p < n_src; p++) {
        char*       file_buf = NULL;
        const char* text;
        u64         text_len;
        if (src_is_file[p]) {
            file_buf = read_whole_file(src[p], "stoke batch", &text_len);
            text = file_buf;
        } else {
            text = src[p];
            text_len = (u64)strlen(src[p]);
        }
        if (text_len > (u64)UINT32_MAX / 3) {
            fatal("stoke batch: prompt %u is %llu bytes -- too large", p,
                  (unsigned long long)text_len);
        }
        u32 cap = (u32)(text_len * 3) + 64;
        ids[p] = (u32*)xcalloc(cap, sizeof(u32));
        if (raw) {
            n_ids[p] = tok_encode(tok, text, text_len, ids[p], cap,
                                  /*add_special=*/false);
        } else {
            StokeMsg msg;
            msg.role = "user";
            msg.role_len = 4;
            msg.content = text;
            msg.content_len = text_len;
            n_ids[p] = tok_apply_chatml(tok, &msg, 1, ids[p], cap);
        }
        if (n_ids[p] == 0) {
            fatal("stoke batch: prompt %u tokenized to zero ids", p);
        }
        if (n_ids[p] > longest) longest = n_ids[p];
        free(file_buf);
    }

    pool_start((u32)threads_long);
    Model* model = model_open(model_path);

    /* One ctx for every slot. The default is the smallest that fits the
     * longest prompt plus the full generation, so a run that does not pass
     * --ctx cannot be refused admission for a reason it never chose. */
    u64 need_ctx = (u64)longest + (u64)n_gen;
    u32 ctx = (ctx_long > 0) ? (u32)ctx_long : (u32)need_ctx;
    if ((u64)ctx < need_ctx) {
        fatal("stoke batch: --ctx %u is too small for the longest prompt "
              "(%u) plus -n %u = %llu", ctx, longest, n_gen,
              (unsigned long long)need_ctx);
    }
    if (ctx > model_max_ctx(model)) {
        fatal("stoke batch: --ctx %u exceeds the model's max_ctx %u", ctx,
              model_max_ctx(model));
    }

    u32      capacity = n_seq * kv_blocks_for_ctx(ctx);
    KvStore* store = kv_store_open(model_n_layer(model), capacity,
                                   model_n_head_kv(model),
                                   model_head_dim(model), kv_dt);

    Kv**           kv = (Kv**)xcalloc(n_seq, sizeof(Kv*));
    ModelBatchSeq* bs = (ModelBatchSeq*)xcalloc(n_seq, sizeof(ModelBatchSeq));
    for (u32 s = 0; s < n_seq; s++) {
        kv[s] = kv_open_in(store, ctx);
        if (kv[s] == NULL) {
            fatal("stoke batch: seq %u refused admission (pool has %u blocks, "
                  "%u free; each of %u sequences reserves %u)",
                  s, capacity, kv_store_free_blocks(store), n_seq,
                  kv_blocks_for_ctx(ctx));
        }
        u32 p = s % n_src;
        bs[s].prompt_ids = ids[p];
        bs[s].n_prompt   = n_ids[p];
        bs[s].n_gen      = n_gen;
        bs[s].out_ids    = (u32*)xcalloc(n_gen, sizeof(u32));
        bs[s].n_out      = 0;
        bs[s].n_adopted  = 0;
    }
    double load_ms = now_ms() - t_load_start;

    ModelTiming timing;
    model_generate_batch(model, kv, bs, n_seq, (i64)tok_eos(tok), dedup,
                         &timing);

    /* ALWAYS reported when --dedup is on, not only under --timing: "dedup was
     * enabled" and "dedup fired" are different claims, and a gate that cannot
     * distinguish them is the one that passes while testing nothing. Sum and
     * per-sequence both, because a single sequence adopting everything and
     * every sequence adopting a little are different workloads. */
    if (dedup) {
        u64 adopted_total = 0, prompt_total = 0;
        for (u32 s = 0; s < n_seq; s++) {
            adopted_total += bs[s].n_adopted;
            prompt_total  += bs[s].n_prompt;
        }
        fprintf(stderr, "dedup: %llu of %llu prompt positions adopted (",
                (unsigned long long)adopted_total,
                (unsigned long long)prompt_total);
        for (u32 s = 0; s < n_seq; s++) {
            fprintf(stderr, "%s%u", s ? " " : "", bs[s].n_adopted);
        }
        fprintf(stderr, ")\n");
    }

    /* stderr, always -- stdout carries the streams the gate compares. */
    if (report_timing) {
        double dec_s = timing.decode_ms / 1000.0;
        fprintf(stderr,
                "timing: load %.1f ms | %u seqs x ctx %u (%u blocks, kv %s) | "
                "prefill %u tok in %.1f ms | decode %u tok in %.1f ms",
                load_ms, n_seq, ctx, capacity,
                (kv_dt == KV_Q8_0) ? "q8" : "f32", timing.n_prefill,
                timing.prefill_ms, timing.n_decode, timing.decode_ms);
        if (timing.n_decode > 0u && dec_s > 0.0) {
            fprintf(stderr, " (%.3f tok/s aggregate)",
                    (double)timing.n_decode / dec_s);
        } else {
            fprintf(stderr, " (no decode-loop tokens -- no rate)");
        }
        fprintf(stderr, "\n");
    }

    for (u32 s = 0; s < n_seq; s++) {
        printf("== seq %u ==\n", s);
        for (u32 i = 0; i < bs[s].n_out; i++) printf("%u\n", bs[s].out_ids[i]);
        printf("---\n");

        u32   decode_cap = bs[s].n_out * 256u + 16u;
        char* decoded = (char*)xcalloc(decode_cap, 1);
        u32   w = 0;
        for (u32 i = 0; i < bs[s].n_out; i++) {
            w += tok_decode(tok, bs[s].out_ids[i], decoded + w, decode_cap - w);
        }
        fwrite(decoded, 1, w, stdout);
        printf("\n");
        free(decoded);
    }

    for (u32 s = 0; s < n_seq; s++) { kv_close(kv[s]); free(bs[s].out_ids); }
    kv_store_close(store);
    free(kv);
    free(bs);
    for (u32 p = 0; p < n_src; p++) free(ids[p]);
    model_close(model);
    tok_free(tok);
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

/* ---- `stoke serve` -- M6 rung A3 ----------------------------------------
 *
 * ▶ A THIN ARGUMENT PARSE AND NOTHING ELSE. Every decision the server makes
 * lives in src/server.c, which reaches the engine only through stoke.h; this
 * function exists so the CLI has a verb, and it must not grow policy. If
 * something here starts looking like a scheduling or protocol choice, it is in
 * the wrong file.
 *
 * It returns rather than fatal()ing on a bad argument for the same reason the
 * server does: `stoke serve` is the entry point a person runs by hand, and an
 * exit code with a printed usage line is more use than an abort.
 */
static int cmd_serve(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: stoke serve <model.gguf> [--host A] [--port P] "
                "[--ctx N] [--seqs N] [--threads N] [--kv q8] "
                "[--max-requests N] [-v]\n");
        return 2;
    }

    stoke_open_opts eo;
    memset(&eo, 0, sizeof eo);
    eo.size         = sizeof eo;
    eo.dedup_prefix = 1;

    ServerOpts so;
    memset(&so, 0, sizeof so);
    so.port = 8080;

    for (int i = 3; i < argc; i++) {
        const char* a = argv[i];
        int         has_next = (i + 1 < argc);
        if (strcmp(a, "-v") == 0) {
            so.verbose = 1;
        } else if (strcmp(a, "--host") == 0 && has_next) {
            so.bind_addr = argv[++i];
        } else if (strcmp(a, "--port") == 0 && has_next) {
            so.port = (u16)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(a, "--ctx") == 0 && has_next) {
            eo.max_ctx = (u32)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(a, "--seqs") == 0 && has_next) {
            eo.max_seqs = (u32)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(a, "--threads") == 0 && has_next) {
            eo.n_threads = (u32)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(a, "--kv") == 0 && has_next) {
            const char* v = argv[++i];
            if (strcmp(v, "q8") == 0) {
                eo.kv_q8 = 1;
            } else if (strcmp(v, "f32") != 0) {
                fprintf(stderr, "stoke serve: --kv takes f32 or q8\n");
                return 2;
            }
        } else if (strcmp(a, "--max-requests") == 0 && has_next) {
            so.max_requests = (int)strtol(argv[++i], NULL, 10);
        } else {
            fprintf(stderr, "stoke serve: unknown argument '%s'\n", a);
            return 2;
        }
    }

    return server_run(argv[2], &eo, &so);
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
              "batch <path.gguf> [--raw] [-n N] (-p \"text\" | -f file)... "
              "[--seqs N] [--ctx N] [--threads N] [--timing], "
              "bench --ops <path.gguf> [--reps N] [--jsonl], "
              "serve <path.gguf> [--host A] [--port P] [--ctx N] [--seqs N] "
              "[--threads N] [--kv q8] [--max-requests N] [-v])");
    }
    if (strcmp(argv[1], "bench") == 0) {
        /* The `--ops` spelling is mandated by the M3 forward note; it is a
         * required mode rather than a default so that later bench modes
         * (end-to-end tok/s, scenario replay) can be added without ever
         * silently changing what a bare `stoke bench` measures. */
        if (argc >= 3 && strcmp(argv[2], "--serve") == 0) {
            /* M5 in-process serving driver -- the mode this dispatch
             * was deliberately shaped to allow without ever changing
             * what a bare `stoke bench` or `--ops` measures. */
            cmd_bench_serve(argc, argv);
            finish_stdout_or_fatal();
            return 0;
        }
        if (argc >= 4 && strcmp(argv[2], "--matvec-batch") == 0) {
            /* The BATCHED K-quant kernel, timed directly. Its own mode for the
             * reason `--serve` has one: a bare `bench` and `--ops` must keep
             * measuring exactly what they always measured, and this one's
             * output is a RATIO between two arms rather than a table of rates
             * feeding `bench/history.jsonl`. */
            u32         cycles = 0, max_seqs = 0;
            const char* only = NULL;
            for (int i = 4; i < argc; i++) {
                if ((strcmp(argv[i], "--cycles") == 0 ||
                     strcmp(argv[i], "--seqs") == 0) && i + 1 < argc) {
                    const char* flag = argv[i];
                    long v = strtol(argv[++i], NULL, 10);
                    if (v <= 0) {
                        fatal("stoke bench --matvec-batch: %s must be "
                              "positive, got '%s'", flag, argv[i]);
                    }
                    if (strcmp(flag, "--cycles") == 0) cycles = (u32)v;
                    else                               max_seqs = (u32)v;
                } else if (strcmp(argv[i], "--tensor") == 0 && i + 1 < argc) {
                    only = argv[++i];
                } else {
                    fatal("stoke bench --matvec-batch: unexpected argument "
                          "'%s'", argv[i]);
                }
            }
            bench_matvec_batch(argv[3], cycles, max_seqs, only);
            finish_stdout_or_fatal();
            return 0;
        }
        if (argc < 4 || strcmp(argv[2], "--ops") != 0) {
            fatal("usage: stoke bench (--ops <path.gguf> [--reps N] [--jsonl] "
                  "| --matvec-batch <path.gguf> [--cycles N] [--seqs N])");
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
    if (strcmp(argv[1], "batch") == 0) {
        cmd_batch(argc, argv);
        finish_stdout_or_fatal();
        return 0;
    }
    if (strcmp(argv[1], "serve") == 0) {
        return cmd_serve(argc, argv);
    }
    fatal("unknown command '%s' (commands: info, tokenize, run, batch, serve)",
          argv[1]);
    return 1; /* unreachable */
}
