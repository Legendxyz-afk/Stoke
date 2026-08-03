/* tests/test_model.c -- exercises src/model.c + src/kv.c.
 *
 * File-ownership note: this worker's brief puts the tests/data directory
 * tree off-limits, so the usual "committed tiny .gguf fixture" idiom (see
 * tests/data/make_fixtures.py, used by test_gguf.c) isn't available here.
 * Instead, this file builds tiny SYNTHETIC qwen3-shaped GGUF v3 files
 * itself, at test run time (write_fixture() below), from a from-scratch
 * byte encoder written directly against the wire format src/gguf.c
 * parses (magic/version/counts, gguf_string = u64 len + raw bytes,
 * KV = key + u32 type + value, tensor_info = name + n_dims + dims[] +
 * ggml_type + rel_offset, general.alignment-padded data section).
 * general.alignment is pinned to FIX_ALIGNMENT (32, matching every real
 * GGUF file's default) with explicit padding inserted both between the
 * tensor-info section and the data section and before each tensor's own
 * bytes (pad_to_alignment()) -- an earlier version of this fixture used
 * alignment=1 to skip that bookkeeping, but that let tensor data start at
 * arbitrary non-4-byte-aligned offsets, which UBSan correctly flagged as
 * misaligned f32 reads once model.c cast t->data straight to `const
 * f32*` (always safe against real files, which never use alignment<32).
 * All fixture tensors are F32 (no Q8_0 encoding needed for these tests --
 * src/kernels/q8_0.c and its Q8_0 matvec dispatch path are already
 * covered by tests/test_kernels.c; this file is about model.c's config/
 * shape validation and the forward/generate control flow, not quant
 * kernels).
 *
 * REAL model files (models/Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf,
 * models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf) are used for two things
 * that need a genuine qwen3 GGUF rather than a synthetic one: (1) the
 * "doctored dims for a REAL tensor name" shape-validation proof the
 * brief calls for (model_check_tensor_shape_*), and (2) a fast (metadata-
 * only, no forward pass) sanity check that model_open() reads the real
 * production config exactly as probed. Both are gated by
 * require_real_model_or_skip(), mirroring tests/golden_lib.sh's
 * STOKE_REQUIRE_GOLDEN SKIP-vs-FAIL policy (models/ is gitignored, so a
 * fresh clone must SKIP cleanly; the reference box sets
 * STOKE_REQUIRE_GOLDEN=1 so a moved/missing model is a hard failure, not
 * a silent no-op) -- this project's shell gates already do this; this is
 * the same policy re-expressed in C since these are C unit tests, not
 * shell scripts. Every other test here builds its own fixture and never
 * touches models/, so it always runs regardless of what's on disk.
 *
 * Negative (fatal-expecting) tests use the spawn-self idiom from
 * tests/test_gguf.c / tests/test_kernels.c: fatal() calls exit(1)
 * directly, so the only way to observe "did this call fatal the way we
 * expect" from a test that must keep running afterwards is to run that
 * one call in a child process and inspect its exit code + stderr from
 * the outside.
 */
#include "common.h"

#include "gguf.h"
#include "kv.h"
#include "model.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

/* ===========================================================================
 * Tiny synthetic qwen3-shaped GGUF v3 fixture builder.
 * ===========================================================================
 */

#define MODEL_FIXTURE_PATH "test_model_fixture_tmp.gguf"
#define MODEL_DUMP_DIR "build/model_test_dump_tmp"

#define REAL_Q8_0_MODEL "models/Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf"
#define REAL_Q4KM_MODEL "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

/* Fixture dims: small, and chosen pairwise-distinct where it matters
 * (n_embd != n_head*head_dim != n_head_kv*head_dim != n_ff) so a
 * transposed/confused dimension would show up as a real mismatch rather
 * than an accidental match. gqa_group = 3 (3 query heads sharing 1 kv
 * head) exercises real GQA grouping, not just the group=1 degenerate
 * case. */
#define FIX_N_EMBD 6u
#define FIX_HEAD_DIM 4u
#define FIX_N_HEAD 3u
#define FIX_N_FF 10u
#define FIX_BLOCK_COUNT 2u
#define FIX_VOCAB 5u
#define FIX_CTX_LEN 64u

/* general.alignment used by the fixture -- 32, matching every real GGUF
 * file's default (NOT 1: an early version of this fixture used 1 to
 * simplify offset bookkeeping, but that let tensor data start at
 * arbitrary, non-4-byte-aligned byte offsets -- harmless for gguf.c's own
 * bounds checks, but UBSan correctly flagged the resulting misaligned f32
 * reads once model.c cast t->data straight to `const f32*` (safe against
 * every REAL file, which always pads to a multiple of 32). Padding to a
 * real value here instead of loosening model.c's casts keeps the fixture
 * representative of what model.c will ever actually be handed. */
#define FIX_ALIGNMENT 32u

typedef struct {
    u8*    data;
    size_t len;
    size_t cap;
} Buf;

static void buf_init(Buf* b) {
    b->data = NULL;
    b->len = 0;
    b->cap = 0;
}
static void buf_free(Buf* b) { free(b->data); }

static void buf_reserve(Buf* b, size_t extra) {
    if (b->len + extra <= b->cap) return;
    size_t newcap = b->cap ? b->cap * 2 : 256;
    while (newcap < b->len + extra) newcap *= 2;
    b->data = (u8*)xrealloc(b->data, newcap);
    b->cap = newcap;
}

static void buf_bytes(Buf* b, const void* p, size_t n) {
    buf_reserve(b, n);
    memcpy(b->data + b->len, p, n);
    b->len += n;
}

/* x86_64-only assumption (this project's two supported hosts, MSYS2
 * ucrt64 and WSL/Linux, are both little-endian x86_64): a raw memcpy of a
 * u32/u64/f32 host value IS already the correct GGUF little-endian byte
 * sequence, so no manual byte assembly is needed in this test-only
 * encoder (unlike src/gguf.c's rd_u32/rd_u16, which must be endian-
 * explicit because they decode arbitrary FILE bytes, not host values). */
static void buf_u32(Buf* b, u32 v) { buf_bytes(b, &v, 4); }
static void buf_u64(Buf* b, u64 v) { buf_bytes(b, &v, 8); }
static void buf_f32(Buf* b, f32 v) { buf_bytes(b, &v, 4); }

static void buf_gguf_string(Buf* b, const char* s) {
    u64 len = (u64)strlen(s);
    buf_u64(b, len);
    buf_bytes(b, s, (size_t)len);
}

static void kv_header(Buf* b, const char* key, u32 type) {
    buf_gguf_string(b, key);
    buf_u32(b, type);
}
static void kv_string(Buf* b, const char* key, const char* val) {
    kv_header(b, key, GGUF_TYPE_STRING);
    buf_gguf_string(b, val);
}
static void kv_u32(Buf* b, const char* key, u32 val) {
    kv_header(b, key, GGUF_TYPE_UINT32);
    buf_u32(b, val);
}
static void kv_f32(Buf* b, const char* key, f32 val) {
    kv_header(b, key, GGUF_TYPE_FLOAT32);
    buf_f32(b, val);
}

static void tensor_info(Buf* b, const char* name, u32 n_dims,
                         const u64* dims, u32 ggml_type, u64 rel_offset) {
    buf_gguf_string(b, name);
    buf_u32(b, n_dims);
    for (u32 i = 0; i < n_dims; i++) buf_u64(b, dims[i]);
    buf_u32(b, ggml_type);
    buf_u64(b, rel_offset);
}

/* Zero-pads `b` up to the next multiple of FIX_ALIGNMENT (no-op if already
 * aligned) -- shared by the tensor-data padding below and the
 * tensor-info-section-to-data-section padding in write_fixture, both of
 * which must reproduce gguf.c's own `pad = (alignment - (cur % alignment))
 * % alignment` exactly for the offsets this fixture records to mean what
 * gguf.c will think they mean. */
static void pad_to_alignment(Buf* b) {
    u64 rem = (u64)b->len % FIX_ALIGNMENT;
    if (rem == 0) return;
    u64 pad = FIX_ALIGNMENT - rem;
    u8  zeros[FIX_ALIGNMENT];
    memset(zeros, 0, sizeof zeros);
    buf_bytes(b, zeros, (size_t)pad);
}

/* Appends one F32 tensor's info (rel_offset = current cumulative length of
 * `data` AFTER padding it to FIX_ALIGNMENT -- gguf.c fatals unless every
 * rel_offset is itself a multiple of general.alignment) plus its actual
 * data bytes (small deterministic non-uniform pattern: distinct values, no
 * NaN/inf, exercises real arithmetic rather than degenerate all-zero
 * weights). */
static void add_tensor_f32(Buf* tensors, Buf* data, u64* tensor_count,
                            u64* fill_ctr, const char* name, u32 n_dims,
                            const u64* dims) {
    pad_to_alignment(data);
    u64 rel_offset = (u64)data->len;
    tensor_info(tensors, name, n_dims, dims, GGUF_GGML_TYPE_F32, rel_offset);
    u64 nelem = 1;
    for (u32 i = 0; i < n_dims; i++) nelem *= dims[i];
    for (u64 i = 0; i < nelem; i++) {
        f32 v = 0.05f * (f32)((int)(*fill_ctr % 7) - 3);
        buf_f32(data, v);
        (*fill_ctr)++;
    }
    (*tensor_count)++;
}

typedef enum {
    MUT_NONE = 0,
    MUT_MISSING_ARCH,
    MUT_WRONG_ARCH,
    MUT_MISSING_BLOCK_COUNT,
    MUT_VALUE_LEN_MISMATCH,
    MUT_HEAD_COUNT_NOT_DIVISIBLE,
    MUT_ODD_HEAD_DIM,
    MUT_BAD_ATTN_Q_DIMS
} Mutation;

/* Writes a tiny, otherwise-self-consistent qwen3-shaped GGUF v3 file to
 * `path`, applying exactly one deliberate mutation (MUT_NONE = fully
 * valid). See the file header comment for why this exists instead of a
 * committed tests/data/ fixture. */
static void write_fixture(const char* path, Mutation mut) {
    u32 n_head_kv = 1u;
    if (mut == MUT_HEAD_COUNT_NOT_DIVISIBLE) n_head_kv = 2u; /* 3 % 2 != 0 */
    u32 head_dim = FIX_HEAD_DIM;
    if (mut == MUT_ODD_HEAD_DIM) head_dim = 3u;
    u32 value_len = head_dim;
    if (mut == MUT_VALUE_LEN_MISMATCH) value_len = head_dim + 1u;

    Buf kvs;
    buf_init(&kvs);
    u32 kv_count = 0;

    if (mut != MUT_MISSING_ARCH) {
        kv_string(&kvs, "general.architecture",
                  mut == MUT_WRONG_ARCH ? "llama" : "qwen3");
        kv_count++;
    }
    kv_u32(&kvs, "general.alignment", FIX_ALIGNMENT);
    kv_count++;
    if (mut != MUT_MISSING_BLOCK_COUNT) {
        kv_u32(&kvs, "qwen3.block_count", FIX_BLOCK_COUNT);
        kv_count++;
    }
    kv_u32(&kvs, "qwen3.context_length", FIX_CTX_LEN);
    kv_count++;
    kv_u32(&kvs, "qwen3.embedding_length", FIX_N_EMBD);
    kv_count++;
    kv_u32(&kvs, "qwen3.feed_forward_length", FIX_N_FF);
    kv_count++;
    kv_u32(&kvs, "qwen3.attention.head_count", FIX_N_HEAD);
    kv_count++;
    kv_u32(&kvs, "qwen3.attention.head_count_kv", n_head_kv);
    kv_count++;
    kv_u32(&kvs, "qwen3.attention.key_length", head_dim);
    kv_count++;
    kv_u32(&kvs, "qwen3.attention.value_length", value_len);
    kv_count++;
    kv_f32(&kvs, "qwen3.rope.freq_base", 10000.0f);
    kv_count++;
    kv_f32(&kvs, "qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    kv_count++;

    Buf tensors;
    buf_init(&tensors);
    Buf data;
    buf_init(&data);
    u64 tensor_count = 0;
    u64 fill_ctr = 0;

    u32 q_dim = FIX_N_HEAD * head_dim;
    u32 kv_dim = n_head_kv * head_dim;
    u64 d1[1];
    u64 d2[2];
    char name[64];

    for (u32 L = 0; L < FIX_BLOCK_COUNT; L++) {
        snprintf(name, sizeof name, "blk.%u.attn_norm.weight", L);
        d1[0] = FIX_N_EMBD;
        add_tensor_f32(&tensors, &data, &tensor_count, &fill_ctr, name, 1, d1);

        snprintf(name, sizeof name, "blk.%u.attn_q.weight", L);
        d2[0] = FIX_N_EMBD;
        d2[1] = (mut == MUT_BAD_ATTN_Q_DIMS && L == 0) ? (u64)q_dim + 1
                                                        : (u64)q_dim;
        add_tensor_f32(&tensors, &data, &tensor_count, &fill_ctr, name, 2, d2);

        snprintf(name, sizeof name, "blk.%u.attn_q_norm.weight", L);
        d1[0] = head_dim;
        add_tensor_f32(&tensors, &data, &tensor_count, &fill_ctr, name, 1, d1);

        snprintf(name, sizeof name, "blk.%u.attn_k.weight", L);
        d2[0] = FIX_N_EMBD;
        d2[1] = kv_dim;
        add_tensor_f32(&tensors, &data, &tensor_count, &fill_ctr, name, 2, d2);

        snprintf(name, sizeof name, "blk.%u.attn_k_norm.weight", L);
        d1[0] = head_dim;
        add_tensor_f32(&tensors, &data, &tensor_count, &fill_ctr, name, 1, d1);

        snprintf(name, sizeof name, "blk.%u.attn_v.weight", L);
        d2[0] = FIX_N_EMBD;
        d2[1] = kv_dim;
        add_tensor_f32(&tensors, &data, &tensor_count, &fill_ctr, name, 2, d2);

        snprintf(name, sizeof name, "blk.%u.attn_output.weight", L);
        d2[0] = q_dim;
        d2[1] = FIX_N_EMBD;
        add_tensor_f32(&tensors, &data, &tensor_count, &fill_ctr, name, 2, d2);

        snprintf(name, sizeof name, "blk.%u.ffn_norm.weight", L);
        d1[0] = FIX_N_EMBD;
        add_tensor_f32(&tensors, &data, &tensor_count, &fill_ctr, name, 1, d1);

        snprintf(name, sizeof name, "blk.%u.ffn_gate.weight", L);
        d2[0] = FIX_N_EMBD;
        d2[1] = FIX_N_FF;
        add_tensor_f32(&tensors, &data, &tensor_count, &fill_ctr, name, 2, d2);

        snprintf(name, sizeof name, "blk.%u.ffn_up.weight", L);
        d2[0] = FIX_N_EMBD;
        d2[1] = FIX_N_FF;
        add_tensor_f32(&tensors, &data, &tensor_count, &fill_ctr, name, 2, d2);

        snprintf(name, sizeof name, "blk.%u.ffn_down.weight", L);
        d2[0] = FIX_N_FF;
        d2[1] = FIX_N_EMBD;
        add_tensor_f32(&tensors, &data, &tensor_count, &fill_ctr, name, 2, d2);
    }

    d2[0] = FIX_N_EMBD;
    d2[1] = FIX_VOCAB;
    add_tensor_f32(&tensors, &data, &tensor_count, &fill_ctr,
                   "token_embd.weight", 2, d2);

    d1[0] = FIX_N_EMBD;
    add_tensor_f32(&tensors, &data, &tensor_count, &fill_ctr,
                   "output_norm.weight", 1, d1);

    Buf out;
    buf_init(&out);
    buf_bytes(&out, "GGUF", 4);
    buf_u32(&out, 3); /* version */
    buf_u64(&out, tensor_count);
    buf_u64(&out, (u64)kv_count);
    buf_bytes(&out, kvs.data, kvs.len);
    buf_bytes(&out, tensors.data, tensors.len);
    /* Matches gguf.c's own data_start = align_up(cur, alignment): without
     * this, the data section here would start at an unaligned `out.len`,
     * but gguf_open() would still compute data_start as the ALIGNED
     * position -- silently misreading every tensor's bytes shifted by the
     * gap. */
    pad_to_alignment(&out);
    buf_bytes(&out, data.data, data.len);

    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fatal("test_model: cannot create fixture '%s' -- %s", path,
              strerror(errno));
    }
    size_t wrote = fwrite(out.data, 1, out.len, fp);
    if (wrote != out.len || fclose(fp) != 0) {
        fatal("test_model: short write / close error for fixture '%s'", path);
    }

    buf_free(&kvs);
    buf_free(&tensors);
    buf_free(&data);
    buf_free(&out);
}

/* ===========================================================================
 * Child-mode: every scenario that must observe a fatal().
 * ===========================================================================
 */
static int model_test_child_main(int argc, char** argv) {
    if (argc < 3 || strcmp(argv[1], "--scenario") != 0) {
        fprintf(stderr,
                "test_model child: bad invocation (want --scenario <name>)\n");
        return 2;
    }
    const char* s = argv[2];

    if (strcmp(s, "missing_arch") == 0) {
        write_fixture(MODEL_FIXTURE_PATH, MUT_MISSING_ARCH);
        (void)model_open(MODEL_FIXTURE_PATH);
    } else if (strcmp(s, "wrong_arch") == 0) {
        write_fixture(MODEL_FIXTURE_PATH, MUT_WRONG_ARCH);
        (void)model_open(MODEL_FIXTURE_PATH);
    } else if (strcmp(s, "missing_block_count") == 0) {
        write_fixture(MODEL_FIXTURE_PATH, MUT_MISSING_BLOCK_COUNT);
        (void)model_open(MODEL_FIXTURE_PATH);
    } else if (strcmp(s, "value_len_mismatch") == 0) {
        write_fixture(MODEL_FIXTURE_PATH, MUT_VALUE_LEN_MISMATCH);
        (void)model_open(MODEL_FIXTURE_PATH);
    } else if (strcmp(s, "head_count_not_divisible") == 0) {
        write_fixture(MODEL_FIXTURE_PATH, MUT_HEAD_COUNT_NOT_DIVISIBLE);
        (void)model_open(MODEL_FIXTURE_PATH);
    } else if (strcmp(s, "odd_head_dim") == 0) {
        write_fixture(MODEL_FIXTURE_PATH, MUT_ODD_HEAD_DIM);
        (void)model_open(MODEL_FIXTURE_PATH);
    } else if (strcmp(s, "bad_attn_q_dims") == 0) {
        write_fixture(MODEL_FIXTURE_PATH, MUT_BAD_ATTN_Q_DIMS);
        (void)model_open(MODEL_FIXTURE_PATH);
    } else if (strcmp(s, "check_tensor_shape_real_mismatch") == 0) {
        /* "doctored dims for a REAL tensor name FATALs naming the tensor"
         * (mandatory verification item 2), driven straight against the
         * real Q8_0 file's blk.0.attn_q.weight (dims [2560,4096]) with a
         * deliberately wrong want[1]. */
        GgufFile*         f = gguf_open(REAL_Q8_0_MODEL);
        const GgufTensor* t = gguf_tensor_by_name(f, "blk.0.attn_q.weight");
        u64               want[2];
        want[0] = 2560;
        want[1] = 9999; /* real dims[1] is 4096 */
        model_check_tensor_shape(t, 2, want);
        gguf_close(f);
    } else if (strcmp(s, "open_q4km_dev_file") == 0) {
        /* The Q4_K_M dev file's weight matrices are Q4_K/Q6_K -- model.c
         * restricts weight tensors to F32/Q8_0 at M2, so this must fatal
         * naming the first such tensor, not deep inside a later matvec. */
        (void)model_open(REAL_Q4KM_MODEL);
    } else if (strcmp(s, "generate_kv_too_small") == 0) {
        write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
        Model* m = model_open(MODEL_FIXTURE_PATH);
        Kv* kv = kv_open(model_n_layer(m), 2 /* too small for 3+2=5 */,
                          model_n_head_kv(m), model_head_dim(m));
        u32 prompt[3] = {0, 1, 2};
        u32 out[2];
        (void)model_generate(m, kv, prompt, 3, 2, -1, out, NULL, -1, NULL,
                              NULL);
    } else if (strcmp(s, "forward_token_id_oob") == 0) {
        write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
        Model* m = model_open(MODEL_FIXTURE_PATH);
        Kv*    kv = kv_open(model_n_layer(m), model_max_ctx(m),
                             model_n_head_kv(m), model_head_dim(m));
        f32    xout[FIX_N_EMBD];
        (void)model_forward_position(m, kv, 9999u /* out of vocab (5) */, 0,
                                      NULL, xout);
    } else if (strcmp(s, "forward_kv_max_ctx_too_large") == 0) {
        /* The oversized-cache case: legal for the Kv, past what the model's
         * s_scores can hold. Every OTHER dim must match the model exactly, or
         * the shape check above fatals first and this scenario would prove
         * nothing about the max_ctx clause. */
        write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
        Model* m  = model_open(MODEL_FIXTURE_PATH);
        Kv*    kv = kv_open(model_n_layer(m), model_max_ctx(m) + 1u,
                             model_n_head_kv(m), model_head_dim(m));
        f32    xout[FIX_N_EMBD];
        /* pos == the model's cap: inside the CACHE (so the pos guard passes)
         * and exactly one past the end of s_scores. This is the index that
         * would have written into the next scratch buffer. */
        (void)model_forward_position(m, kv, 0u, model_max_ctx(m), NULL, xout);
    } else if (strcmp(s, "kv_view_layer_oob") == 0) {
        Kv* kv = kv_open(2, 8, 1, 4);
        (void)kv_view(kv, 2 /* n_layer is 2 */, 0, 0);
    } else if (strcmp(s, "kv_view_kv_head_oob") == 0) {
        Kv* kv = kv_open(2, 8, 1, 4);
        (void)kv_view(kv, 0, 1 /* n_head_kv is 1 */, 0);
    } else if (strcmp(s, "kv_view_pos_oob") == 0) {
        Kv* kv = kv_open(2, 8, 1, 4);
        (void)kv_view(kv, 0, 0, 8 /* max_ctx is 8 */);
    } else {
        fprintf(stderr, "test_model child: unknown scenario '%s'\n", s);
        return 2;
    }

    printf("test_model child: scenario '%s' did NOT fatal\n", s);
    return 0;
}
#define TEST_CHILD_MAIN model_test_child_main
#include "test.h"

/* ===========================================================================
 * Spawn helper -- identical Windows cmd.exe outer-quote fix as
 * tests/test_gguf.c / tests/test_kernels.c's spawn_scenario (this repo
 * lives under a path with a space, "...\New Drive\...").
 * ===========================================================================
 */
static int spawn_scenario(const char* scenario, char* err_buf,
                           size_t err_buf_sz) {
#ifdef _WIN32
    char  exe_path_buf[4096];
    DWORD full_len = GetFullPathNameA(g_test_argv0, (DWORD)sizeof exe_path_buf,
                                       exe_path_buf, NULL);
    if (full_len == 0 || full_len >= sizeof exe_path_buf) {
        fatal("test_model: GetFullPathNameA failed for '%s'", g_test_argv0);
    }
    const char* exe_path = exe_path_buf;
#else
    const char* exe_path = g_test_argv0;
#endif

    char errfile[4200];
    char inner[8600];
    char cmd[8800];

    snprintf(errfile, sizeof errfile, "%s.%s.stderr", exe_path, scenario);
    snprintf(inner, sizeof inner, "\"%s\" --scenario %s 2>\"%s\"", exe_path,
             scenario, errfile);

#ifdef _WIN32
    snprintf(cmd, sizeof cmd, "\"%s\"", inner); /* extra outer quote pair */
#else
    snprintf(cmd, sizeof cmd, "%s", inner);
#endif

    int rc = system(cmd);

    err_buf[0] = '\0';
    FILE* ef = fopen(errfile, "rb");
    if (ef) {
        size_t n = fread(err_buf, 1, err_buf_sz - 1, ef);
        err_buf[n] = '\0';
        fclose(ef);
        remove(errfile);
    }
    return rc;
}

static void assert_scenario_fatals(const char* scenario,
                                    const char* msg_substr) {
    char errbuf[2048];
    int  rc = spawn_scenario(scenario, errbuf, sizeof errbuf);
    ASSERT_TRUE(rc != 0);
    ASSERT_TRUE(strncmp(errbuf, "fatal: ", 7) == 0);
    if (msg_substr != NULL && strstr(errbuf, msg_substr) == NULL) {
        fprintf(stderr,
                "FAIL %s:%d: scenario '%s': stderr did not contain \"%s\" "
                "(got: %s)\n",
                __FILE__, __LINE__, scenario, msg_substr, errbuf);
        g_test_case_failed = 1;
    }
}

/* ===========================================================================
 * Real-model presence policy -- mirrors tests/golden_lib.sh's
 * golden_lib_require_model exactly (models/ is gitignored: a fresh clone
 * must SKIP these tests cleanly; STOKE_REQUIRE_GOLDEN=1 on the reference
 * box turns a missing/moved model into a hard failure instead, so this
 * can never silently downgrade to a no-op there).
 * ===========================================================================
 */
static bool file_exists(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return false;
    fclose(fp);
    return true;
}

static bool require_real_model_or_skip(const char* path, const char* label) {
    if (file_exists(path)) return true;
    const char* req = getenv("STOKE_REQUIRE_GOLDEN");
    if (req != NULL && strcmp(req, "1") == 0) {
        fprintf(stderr,
                "FAIL %s: %s not present but STOKE_REQUIRE_GOLDEN=1 (this "
                "box is supposed to have the real model)\n",
                label, path);
        g_test_case_failed = 1;
        return false;
    }
    printf("SKIP %s: %s not present (models/ is gitignored; this test "
           "only runs on boxes with the real model)\n",
           label, path);
    return false;
}

/* ===========================================================================
 * Negative: config-key / shape-validation fatals (mandatory verification
 * item 2).
 * ===========================================================================
 */

TEST(model_open_missing_architecture_fatals) {
    assert_scenario_fatals("missing_arch", "general.architecture");
    remove(MODEL_FIXTURE_PATH);
}
TEST(model_open_wrong_architecture_fatals) {
    assert_scenario_fatals("wrong_arch", "expected exactly 'qwen3'");
    remove(MODEL_FIXTURE_PATH);
}
TEST(model_open_missing_block_count_fatals) {
    assert_scenario_fatals("missing_block_count", "qwen3.block_count");
    remove(MODEL_FIXTURE_PATH);
}
TEST(model_open_value_length_mismatch_fatals) {
    assert_scenario_fatals("value_len_mismatch", "value_length");
    remove(MODEL_FIXTURE_PATH);
}
TEST(model_open_head_count_not_divisible_fatals) {
    assert_scenario_fatals("head_count_not_divisible", "not divisible");
    remove(MODEL_FIXTURE_PATH);
}
TEST(model_open_odd_head_dim_fatals) {
    assert_scenario_fatals("odd_head_dim", "must be even");
    remove(MODEL_FIXTURE_PATH);
}
TEST(model_open_bad_attn_q_dims_fatals_naming_tensor) {
    assert_scenario_fatals("bad_attn_q_dims", "blk.0.attn_q.weight");
    remove(MODEL_FIXTURE_PATH);
}

TEST(model_check_tensor_shape_real_tensor_doctored_dims_fatals) {
    if (!require_real_model_or_skip(
            REAL_Q8_0_MODEL,
            "model_check_tensor_shape_real_tensor_doctored_dims_fatals")) {
        return;
    }
    assert_scenario_fatals("check_tensor_shape_real_mismatch",
                            "blk.0.attn_q.weight");
}

TEST(model_open_q4km_dev_file_unsupported_weight_type_fatals) {
    if (!require_real_model_or_skip(
            REAL_Q4KM_MODEL,
            "model_open_q4km_dev_file_unsupported_weight_type_fatals")) {
        return;
    }
    assert_scenario_fatals("open_q4km_dev_file", "are supported at M2");
}

/* ===========================================================================
 * Negative: model.c / kv.c runtime guards.
 * ===========================================================================
 */

TEST(model_generate_exceeding_kv_capacity_fatals) {
    assert_scenario_fatals("generate_kv_too_small", "exceeds kv max_ctx");
    remove(MODEL_FIXTURE_PATH);
}
TEST(model_forward_position_token_id_out_of_range_fatals) {
    assert_scenario_fatals("forward_token_id_oob", "out of vocab range");
    remove(MODEL_FIXTURE_PATH);
}
/* The too-LARGE cache direction. test_model already covered too-small
 * (model_generate_exceeding_kv_capacity_fatals, caught by total_cap), and
 * too-small is the SAFE direction -- nobody had tested the unsafe one, which
 * is how a buffer overrun sat latent behind a shape check that looked
 * complete. With this scenario in the suite, ASan now guards the path
 * permanently on every WSL run. */
TEST(model_forward_position_kv_max_ctx_larger_than_model_fatals) {
    assert_scenario_fatals("forward_kv_max_ctx_too_large",
                           "exceeds model max_ctx");
    remove(MODEL_FIXTURE_PATH);
}
TEST(kv_view_layer_out_of_range_fatals) {
    assert_scenario_fatals("kv_view_layer_oob", "layer");
}
TEST(kv_view_kv_head_out_of_range_fatals) {
    assert_scenario_fatals("kv_view_kv_head_oob", "kv_head");
}
TEST(kv_view_pos_out_of_range_fatals) {
    assert_scenario_fatals("kv_view_pos_oob", "pos");
}

/* ===========================================================================
 * Positive: kv_open/kv_view basic addressing sanity (in-process).
 * ===========================================================================
 */
TEST(kv_view_distinct_slots_do_not_alias) {
    Kv* kv = kv_open(2, 8, 2, 4);
    ASSERT_EQ_U32(kv_n_layer(kv), 2u);
    ASSERT_EQ_U32(kv_max_ctx(kv), 8u);
    ASSERT_EQ_U32(kv_n_head_kv(kv), 2u);
    ASSERT_EQ_U32(kv_head_dim(kv), 4u);

    KvSlot a = kv_view(kv, 0, 0, 0);
    KvSlot b = kv_view(kv, 0, 1, 0);
    KvSlot c = kv_view(kv, 1, 0, 0);
    KvSlot d = kv_view(kv, 0, 0, 1);

    a.k[0] = 1.0f;
    b.k[0] = 2.0f;
    c.k[0] = 3.0f;
    d.k[0] = 4.0f;
    ASSERT_TRUE(a.k[0] == 1.0f);
    ASSERT_TRUE(b.k[0] == 2.0f);
    ASSERT_TRUE(c.k[0] == 3.0f);
    ASSERT_TRUE(d.k[0] == 4.0f);

    /* zero-initialized (xcalloc): an untouched slot must read back 0. */
    KvSlot e = kv_view(kv, 1, 1, 7);
    ASSERT_TRUE(e.k[0] == 0.0f);
    ASSERT_TRUE(e.v[3] == 0.0f);

    kv_close(kv);
}

/* ===========================================================================
 * Positive: model_open on the tiny valid synthetic fixture, config
 * accessors, and a full coherence pass through generate() (embed, rmsnorm,
 * F32 matvec, per-head qk-norm+rope, kv append/attention with a real
 * gqa_group=3, swiglu, tied logits, argmax, --dump, --probe-pos,
 * determinism). This is the ONLY place src/model.c's generate loop runs
 * in the automated suite -- fast because every dim is tiny; the REAL
 * model's numeric correctness is proven separately by the manual L2/64-
 * token verification against ref_forward.py (this test's real job is
 * "does the whole pipeline run and stay internally consistent", not
 * "is the qwen3 math right").
 * ===========================================================================
 */
TEST(model_open_tiny_fixture_config_accessors) {
    write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
    Model* m = model_open(MODEL_FIXTURE_PATH);

    ASSERT_EQ_U32(model_n_layer(m), FIX_BLOCK_COUNT);
    ASSERT_EQ_U32(model_n_embd(m), FIX_N_EMBD);
    ASSERT_EQ_U32(model_n_ff(m), FIX_N_FF);
    ASSERT_EQ_U32(model_n_head(m), FIX_N_HEAD);
    ASSERT_EQ_U32(model_n_head_kv(m), 1u);
    ASSERT_EQ_U32(model_head_dim(m), FIX_HEAD_DIM);
    ASSERT_EQ_U32(model_max_ctx(m), FIX_CTX_LEN); /* 64 < 4096 cap */
    ASSERT_EQ_U32(model_n_vocab(m), FIX_VOCAB);
    ASSERT_TRUE(fabsf(model_freq_base(m) - 10000.0f) < 1e-3f);
    ASSERT_TRUE(fabsf(model_eps(m) - 1e-6f) < 1e-9f);

    model_close(m);
    remove(MODEL_FIXTURE_PATH);
}

TEST(model_generate_tiny_fixture_coherence_dump_probe_determinism) {
    write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
    Model* m = model_open(MODEL_FIXTURE_PATH);

    u32 prompt[3] = {0, 1, 2};
    u32 n_gen = 5;

    Kv* kv1 = kv_open(model_n_layer(m), model_max_ctx(m),
                       model_n_head_kv(m), model_head_dim(m));
    u32             out1[5];
    ModelProbeEntry probe[5];
    for (int i = 0; i < 5; i++) {
        probe[i].id = 0xFFFFFFFFu;
        probe[i].logit = 0.0f;
    }
    u32 count1 = model_generate(m, kv1, prompt, 3, n_gen, /*eos_id=*/-1,
                                 out1, MODEL_DUMP_DIR, /*probe_pos=*/2, probe,
                                 NULL);
    kv_close(kv1);

    /* eos disabled -> exactly n_gen ids, every one in [0, vocab) */
    ASSERT_EQ_U32(count1, n_gen);
    for (u32 i = 0; i < count1; i++) {
        ASSERT_TRUE(out1[i] < FIX_VOCAB);
    }

    /* --probe-pos 2 was reachable (count1=5 > 2): top-5 must be filled
     * (sentinel id cleared) and strictly-by-construction sorted
     * descending by logit (record_top5's own invariant, checked here
     * rather than just trusted). */
    ASSERT_TRUE(probe[0].id != 0xFFFFFFFFu);
    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(probe[i].logit >= probe[i + 1].logit);
    }

    /* --dump wrote one file per layer, each exactly n_embd f32's. */
    for (u32 L = 0; L < FIX_BLOCK_COUNT; L++) {
        char path[256];
        snprintf(path, sizeof path, "%s/L%02u.f32", MODEL_DUMP_DIR, L);
        FILE* fp = fopen(path, "rb");
        ASSERT_TRUE(fp != NULL);
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long sz = ftell(fp);
            ASSERT_EQ_I64(sz, (long)(FIX_N_EMBD * sizeof(f32)));
            fclose(fp);
        }
    }

    /* Determinism: same model, a FRESH kv, same prompt -> identical ids
     * (no hidden state survives across an independent kv/generate call --
     * the "no new file-scope mutable globals" contract's observable
     * consequence). */
    Kv* kv2 = kv_open(model_n_layer(m), model_max_ctx(m),
                       model_n_head_kv(m), model_head_dim(m));
    u32 out2[5];
    u32 count2 = model_generate(m, kv2, prompt, 3, n_gen, /*eos_id=*/-1,
                                 out2, NULL, -1, NULL, NULL);
    kv_close(kv2);
    ASSERT_EQ_U32(count2, count1);
    for (u32 i = 0; i < count1; i++) {
        ASSERT_EQ_U32(out1[i], out2[i]);
    }

    model_close(m);
    remove(MODEL_FIXTURE_PATH);
}

TEST(model_generate_eos_stops_early_and_is_included) {
    /* Force eos_id to equal whatever the FIRST generated id turns out to
     * be (queried on an untouched pass first), then confirm a second,
     * eos-armed run stops IMMEDIATELY after emitting that one id -- eos,
     * if generated, IS included in the output, and generation performs no
     * further forward_position call past it. Picking index 0 specifically
     * (rather than some later index) keeps this test's expectation
     * independent of whether this tiny fixture's arbitrary weights happen
     * to repeat a token later in the sequence. */
    write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
    Model* m = model_open(MODEL_FIXTURE_PATH);
    u32    prompt[2] = {0, 1};

    Kv* kv_probe = kv_open(model_n_layer(m), model_max_ctx(m),
                            model_n_head_kv(m), model_head_dim(m));
    u32 first_pass[4];
    u32 n_first = model_generate(m, kv_probe, prompt, 2, 4, /*eos_id=*/-1,
                                  first_pass, NULL, -1, NULL, NULL);
    kv_close(kv_probe);
    ASSERT_EQ_U32(n_first, 4u);

    i64 eos_id = (i64)first_pass[0];
    Kv* kv2 = kv_open(model_n_layer(m), model_max_ctx(m), model_n_head_kv(m),
                       model_head_dim(m));
    u32 out[4];
    u32 count = model_generate(m, kv2, prompt, 2, 4, eos_id, out, NULL, -1,
                                NULL, NULL);
    kv_close(kv2);

    ASSERT_EQ_U32(count, 1u);
    ASSERT_EQ_U32(out[0], first_pass[0]);
    ASSERT_TRUE((i64)out[0] == eos_id); /* eos IS included */

    model_close(m);
    remove(MODEL_FIXTURE_PATH);
}

/* ===========================================================================
 * Positive: model_open on the REAL Q8_0 parity model -- config values
 * exactly match the brief's probed constants table. Fast: model_open only
 * parses metadata + validates shapes (mmap is lazy), no forward pass.
 * ===========================================================================
 */
TEST(model_open_real_q8_0_file_matches_probed_config) {
    if (!require_real_model_or_skip(
            REAL_Q8_0_MODEL, "model_open_real_q8_0_file_matches_probed_config")) {
        return;
    }
    Model* m = model_open(REAL_Q8_0_MODEL);

    ASSERT_EQ_U32(model_n_layer(m), 36u);
    ASSERT_EQ_U32(model_n_embd(m), 2560u);
    ASSERT_EQ_U32(model_n_ff(m), 9728u);
    ASSERT_EQ_U32(model_n_head(m), 32u);
    ASSERT_EQ_U32(model_n_head_kv(m), 8u);
    ASSERT_EQ_U32(model_head_dim(m), 128u);
    ASSERT_EQ_U32(model_max_ctx(m), 4096u); /* 262144 clamped to the M2 cap */
    ASSERT_EQ_U32(model_n_vocab(m), 151936u);
    ASSERT_TRUE(fabsf(model_freq_base(m) - 5000000.0f) < 1.0f);
    ASSERT_TRUE(fabsf(model_eps(m) - 1e-6f) < 1e-9f);

    model_close(m);
}
