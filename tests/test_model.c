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

#include "blockpool.h"
#include "model.h"
#include "sched.h"
#include "serve.h"
#include "pool.h" /* L4: attention runs on the pool, so its tests drive it */

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
    MUT_BAD_ATTN_Q_DIMS,
    /* Not a defect -- a VALID fixture with gqa_group 4 instead of 3.
     *
     * It exists because M5's E2 attention loop takes a 4-wide chain block when
     * gqa_group >= 4 and a scalar tail otherwise, and the standing fixture's
     * group of 3 runs ONLY the tail. A suite that never builds a group of 4
     * would pass while leaving the block the real model actually uses (Qwen3
     * is 32 q-heads over 8 kv-heads) untested -- this project has already
     * recorded a test passing for a reason unrelated to its name four times. */
    MUT_GQA4
} Mutation;

/* Writes a tiny, otherwise-self-consistent qwen3-shaped GGUF v3 file to
 * `path`, applying exactly one deliberate mutation (MUT_NONE = fully
 * valid). See the file header comment for why this exists instead of a
 * committed tests/data/ fixture. */
static void write_fixture(const char* path, Mutation mut) {
    u32 n_head = FIX_N_HEAD;
    if (mut == MUT_GQA4) n_head = 4u; /* over n_head_kv 1 => gqa_group 4 */
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
    kv_u32(&kvs, "qwen3.attention.head_count", n_head);
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

    u32 q_dim = n_head * head_dim;
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
        /* Kept as a scenario name for continuity, but K4 INVERTED it: the
         * Q4_K_M dev file's weights are Q4_K/Q6_K, which model.c accepts as
         * of K4, so this must now OPEN. Exits non-zero only if it fatals. */
        Model* m = model_open(REAL_Q4KM_MODEL);
        model_close(m);
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

    /* Reaching here means the scenario ran to completion without fatalling.
     * For the many negative scenarios that is the failure, and the parent
     * catches it by rc == 0. But since K4, 'open_q4km_dev_file' is a
     * POSITIVE scenario -- completing IS its pass -- so it must not print a
     * line that reads like a fault in every green run. */
    if (strcmp(s, "open_q4km_dev_file") != 0) {
        printf("test_model child: scenario '%s' did NOT fatal\n", s);
    }
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

/* K4 INVERTED THIS TEST, and the inversion IS the deliverable.
 *
 * Until K4 this asserted that opening the shipping Q4_K_M file FAILS, because
 * model.c accepted only F32/Q8_0 weights. K-quants now load, so the same
 * scenario must succeed. The test was re-pointed rather than deleted: what it
 * has always protected is that the bar model's real weight types are handled
 * at OPEN time and named there, instead of the problem surfacing deep inside
 * a later matvec where the tensor name is long gone.
 *
 * It also checks the config the file actually reports, so "it opened" cannot
 * pass by opening something degenerate. */
TEST(model_open_q4km_dev_file_now_loads_k_quants) {
    if (!require_real_model_or_skip(
            REAL_Q4KM_MODEL,
            "model_open_q4km_dev_file_now_loads_k_quants")) {
        return;
    }
    char errbuf[2048];
    int rc = spawn_scenario("open_q4km_dev_file", errbuf, sizeof errbuf);
    if (rc != 0) {
        fprintf(stderr, "FAIL %s:%d: opening the Q4_K_M file failed: %s\n",
                __FILE__, __LINE__, errbuf);
        g_test_case_failed = 1;
        return;
    }
    Model* m = model_open(REAL_Q4KM_MODEL);
    ASSERT_EQ_U32(model_n_layer(m), 36u);
    ASSERT_EQ_U32(model_n_embd(m), 2560u);
    ASSERT_EQ_U32(model_head_dim(m), 128u);
    model_close(m);
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
/* ===========================================================================
 * M4: sequences sharing one paged KV store must not influence each other.
 *
 * The bar (static-batch design note section 5.4): a sequence's forward pass
 * must produce a BIT-IDENTICAL hidden state whether it runs alone or
 * interleaved with others, whatever else shares the pool.
 *
 * THIS IS NOT VACUOUS TODAY even though the forward path is sequential.
 * The sequences share a KvStore: they draw from one block pool, recycle each
 * other's freed blocks, and index through per-sequence block tables. Any of
 * that going wrong shows up here as one sequence reading another's KV --
 * which produces a plausible hidden state, not a crash.
 *
 * It also becomes the batch-of-N identity gate the moment the matvecs go
 * seqs-inner, which is why it lands BEFORE that change rather than after:
 * a test written against code you already believe works gets shaped to pass.
 *
 * POSITION IN THE ROUND-ROBIN IS VARIED deliberately. "First to step" is
 * exactly the case an off-by-one in a batched inner loop would leave working.
 *
 * ▶ THIS TEST IS NOW LOAD-BEARING, AND THAT WAS MEASURED, NOT ASSUMED.
 * Pointing every sequence at scratch_n[0] -- one shared SeqScratch for the
 * whole batch -- USED to pass here, because the body was sequence-outer and
 * each sequence finished its entire forward pass before the next began, so
 * none could observe another's scratch. That mutation was re-run against the
 * sequence-outer body immediately before the seqs-inner change (21/21, still
 * passing) and again immediately after, where it FAILS at all three
 * round-robin slots. Per-sequence scratch stopped being plumbing and started
 * being a correctness requirement at that commit.
 *
 * The width is fixed at 3 here. Independence across DIFFERENT N -- and the
 * scratch-table realloc that only a GROWING batch can reach -- is the
 * separate test below.
 * ===========================================================================
 */
/* ===========================================================================
 * M4.5 L4: attention runs on the worker pool, and must be BIT-IDENTICAL at any
 * thread count.
 *
 * This is test_pool.c's matvec argument one level up, and it needs its own
 * test because the two are not the same claim. Matvec's parallel unit writes
 * only its own output rows and reads nothing that is shared and mutable.
 * Attention's unit ALSO needs two scratch buffers -- a scores row and a KV
 * dequantization buffer -- which were one-per-SEQUENCE until this change and
 * are one-per-WORKER after it. If that move were wrong, two heads running at
 * once would share a scores row, and the result would still look like a
 * plausible number: no crash, no assertion, just a different token.
 *
 * THREE THINGS ARE ASSERTED, and the third is the one that keeps the first two
 * honest:
 *   1  6 threads reproduces 1 thread, bit for bit;
 *   2  6 threads reproduces ITSELF on a re-run (a race is intermittent, so a
 *      single agreeing pair is weak evidence);
 *   3  the pooled path ACTUALLY RAN -- via model_att_dispatch_count(). Without
 *      it, a build that silently fell back to the serial loop would pass every
 *      comparison above while testing nothing.
 * ===========================================================================
 */
TEST(l4_attention_on_the_pool_is_bit_identical_at_1_and_6_threads) {
    /* ENOUGH POSITIONS TO CROSS THE DISPATCH THRESHOLD, rather than an env var
     * that lowers it. The threshold is a speed heuristic -- a head at a short
     * context is not worth a broadcast -- and at the fixture's usual handful of
     * positions every forward would take the serial path, so the test would
     * pass without once running the code it names.
     *
     * The first attempt set STOKE_ATT_MIN_POS=0 through putenv(), which the
     * WSL leg rejected outright: glibc does not declare putenv under
     * -std=c11 (it is POSIX, not ISO), while MinGW does. Running the real
     * threshold on real positions is better anyway -- it tests what ships. */
    write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
    Model* m = model_open(MODEL_FIXTURE_PATH);

    const u32 n_embd = model_n_embd(m);
    const u32 ATT_MIN_POS_DEFAULT = 32u; /* mirrors model.c's g_att_min_pos */
    const u32 NTOK = 48;                 /* fixture max_ctx is 64 */
    u32 tokens[48];
    for (u32 t = 0; t < NTOK; t++) tokens[t] = t % 5u; /* vocab is 5 */

    f32* out1 = (f32*)xmalloc((size_t)NTOK * n_embd * sizeof(f32));
    f32* out6 = (f32*)xmalloc((size_t)NTOK * n_embd * sizeof(f32));
    f32* out6b = (f32*)xmalloc((size_t)NTOK * n_embd * sizeof(f32));

    u64 disp0 = model_att_dispatch_count();

    /* Serial reference: one thread, so the pool is inert and the heads run on
     * the caller exactly as they did before this change. */
    pool_stop();
    pool_start(1u);
    {
        Kv* kv = kv_open(model_n_layer(m), model_max_ctx(m),
                         model_n_head_kv(m), model_head_dim(m));
        for (u32 t = 0; t < NTOK; t++)
            model_forward_position(m, kv, tokens[t], t, NULL,
                                   out1 + (size_t)t * n_embd);
        kv_close(kv);
    }

    /* pool_start AFTER model_open on purpose: the per-worker scratch is sized
     * from the pool as it stands, and this is the ordering that would leave it
     * short if the re-check at the top of every forward were dropped. */
    pool_stop();
    pool_start(6u);
    f32* dst[2] = {out6, out6b};
    for (u32 rep = 0; rep < 2u; rep++) {
        Kv* kv = kv_open(model_n_layer(m), model_max_ctx(m),
                         model_n_head_kv(m), model_head_dim(m));
        for (u32 t = 0; t < NTOK; t++)
            model_forward_position(m, kv, tokens[t], t, NULL,
                                   dst[rep] + (size_t)t * n_embd);
        kv_close(kv);
    }
    pool_stop();

    size_t bytes = (size_t)NTOK * n_embd * sizeof(f32);
    ASSERT_TRUE(memcmp(out1, out6, bytes) == 0);
    ASSERT_TRUE(memcmp(out6, out6b, bytes) == 0);

    /* The anti-vacuity check: it counts the times the heads were HANDED to
     * pool_for_units, which is the decision the threshold could have skipped.
     * (At one thread the pool then runs them inline, which is the point of the
     * serial arm.) One per layer for every position at or past the threshold,
     * over three runs of NTOK positions.
     *
     * This encodes ATT_MIN_POS's DEFAULT. If that default moves, this fails
     * loudly with both numbers printed -- which is the intended behaviour: a
     * threshold change should have to come past a test that says what it was
     * relied on to be. */
    u64 want = (u64)model_n_layer(m) * (NTOK - ATT_MIN_POS_DEFAULT + 1u) * 3u;
    u64 dispatched = model_att_dispatch_count() - disp0;
    if (dispatched < want) {
        printf("  only %llu attention dispatches, expected %llu\n",
               (unsigned long long)dispatched, (unsigned long long)want);
    }
    ASSERT_TRUE(dispatched >= want);

    /* And the outputs must not be all-zero, or every memcmp above is vacuous
     * too -- the same guard test_pool.c puts on its matvec comparison. */
    u32 nonzero = 0;
    for (u32 i = 0; i < NTOK * n_embd; i++) {
        if (out1[i] != 0.0f) nonzero++;
    }
    ASSERT_TRUE(nonzero > 0u);

    free(out1);
    free(out6);
    free(out6b);
    model_close(m);
}

/* ===========================================================================
 * M5 E2 -- the kv-head-outer attention loop is thread-count invariant, on BOTH
 * of its shapes.
 *
 * ▶ THIS TEST USED TO COMPARE TWO ARMS. It ran E2's loop against M4.5's
 * q-head-outer one and asserted they were bit-identical; that arm was deleted
 * once the A/B that needed it had reported, so the comparison no longer exists
 * to be made. What must NOT be lost with it is the COVERAGE: E2's loop takes a
 * 4-wide chain block when gqa_group >= 4 and a scalar tail otherwise, and the
 * standing fixture's group of 3 exercises the TAIL ONLY. Deleting the arm test
 * without keeping MUT_GQA4 alive would have silently retired the only check on
 * the block the real model actually runs -- the shape of regression a deletion
 * commit is most likely to cause and least likely to be blamed for.
 *
 * WHAT IS ASSERTED NOW. The parallelism is over INDEPENDENT OUTPUTS, so the
 * result must not depend on how many workers claim them: 1 thread vs 6, on both
 * gqa shapes, compared over HIDDEN STATES (which diverge long before a sampled
 * token does). Plus the anti-vacuity check the L4 test established -- that the
 * pooled path actually ran -- because "same at 1 and 6 threads" is equally true
 * of a build that never dispatched at all.
 *
 * The stronger evidence lives elsewhere and is not duplicated here: the
 * PowerShell gate byte-compares the real 4B model's token streams against
 * archived binaries (gate-reports/2026-08-19-m5-entry-verify.txt, 10 cases x 4
 * rung arms, all SAME).
 * ===========================================================================
 */
TEST(m5_e2_kv_head_outer_attention_is_thread_invariant_on_both_gqa_shapes) {
    const u32      NTOK = 48; /* fixture max_ctx is 64; past ATT_MIN_POS 32 */
    const Mutation variants[2] = {MUT_NONE, MUT_GQA4};
    const u32      want_group[2] = {3u, 4u}; /* tail-only, then 4-wide block */

    for (u32 vi = 0; vi < 2u; vi++) {
        write_fixture(MODEL_FIXTURE_PATH, variants[vi]);
        Model*    m = model_open(MODEL_FIXTURE_PATH);
        const u32 n_embd = model_n_embd(m);

        /* The fixture must actually have the shape the case is named for --
         * otherwise "both shapes" is a comment, not a fact. */
        ASSERT_EQ_U32(model_n_head(m) / model_n_head_kv(m), want_group[vi]);

        u32 tokens[48];
        for (u32 t = 0; t < NTOK; t++) tokens[t] = t % 5u; /* vocab is 5 */

        size_t bytes = (size_t)NTOK * n_embd * sizeof(f32);
        f32*   out[2];
        out[0] = (f32*)xmalloc(bytes);
        out[1] = (f32*)xmalloc(bytes);

        u64       disp0 = model_att_dispatch_count();
        const u32 threads[2] = {1u, 6u};
        for (u32 w = 0; w < 2u; w++) {
            pool_stop();
            pool_start(threads[w]);
            Kv* kv = kv_open(model_n_layer(m), model_max_ctx(m),
                             model_n_head_kv(m), model_head_dim(m));
            for (u32 t = 0; t < NTOK; t++) {
                model_forward_position(m, kv, tokens[t], t, NULL,
                                       out[w] + (size_t)t * n_embd);
            }
            kv_close(kv);
        }
        pool_stop();

        ASSERT_TRUE(memcmp(out[0], out[1], bytes) == 0);

        /* Anti-vacuity: the heads must actually have been HANDED to the pool.
         * One dispatch per layer per position at or past ATT_MIN_POS's default,
         * over two runs. Encoding that default is deliberate -- if it moves,
         * this fails loudly rather than quietly testing less. */
        const u32 ATT_MIN_POS_DEFAULT = 32u; /* mirrors model.c */
        u64 want = (u64)model_n_layer(m) * (NTOK - ATT_MIN_POS_DEFAULT + 1u) * 2u;
        u64 got = model_att_dispatch_count() - disp0;
        if (got < want) {
            printf("  only %llu attention dispatches, expected %llu\n",
                   (unsigned long long)got, (unsigned long long)want);
        }
        ASSERT_TRUE(got >= want);

        /* And the states must not be all-zero, or every memcmp is vacuous. */
        u32 nonzero = 0;
        for (u32 i = 0; i < NTOK * n_embd; i++) {
            if (out[0][i] != 0.0f) nonzero++;
        }
        ASSERT_TRUE(nonzero > 0u);

        free(out[0]);
        free(out[1]);
        model_close(m);
    }
    /* Leave the shared fixture path holding the UNMUTATED file: a later test
     * that opens it without rewriting it must not silently inherit gqa 4. */
    write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
}

/* ===========================================================================
 * M5 -- the SCHEDULER's output equals SOLO generation, on the real forward.
 *
 * ▶ THIS IS THE PROPERTY EVERY THROUGHPUT NUMBER RESTS ON. Gate item [2] is a
 * tok/s bar measured through the scheduler; a bar met by a scheduler that
 * generates different text than `model_generate` measures nothing. So before
 * any of that is timed, the streams have to match -- and match PER REQUEST
 * while several run concurrently, since the interesting failure is a batch
 * where each sequence is individually fluent and one of them is somebody
 * else's continuation.
 *
 * What it exercises that the sim cannot: the real forward, the real batched
 * output projection, the real argmax, and the real chunked prefill -- the sim's
 * fake model computes none of those. What the SIM does that this cannot: a
 * million steps of invariant checking. They are complementary, and neither is
 * evidence for the other.
 *
 * Three requests with DIFFERENT prompts, so a host that fed one row's hidden
 * state to another row's projection produces a visibly wrong stream rather
 * than an accidentally-equal one.
 * ===========================================================================
 */
TEST(m5_scheduler_output_equals_solo_generation) {
    write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
    Model* m = model_open(MODEL_FIXTURE_PATH);

    const u32 NREQ = 3, NGEN = 6;
    const u32 nprompt[3] = {5, 9, 7}; /* distinct, and not chunk-aligned */
    u32       prompts[3][16];
    for (u32 i = 0; i < NREQ; i++)
        for (u32 p = 0; p < nprompt[i]; p++)
            prompts[i][p] = (u32)((i * 3 + p) % 5u); /* vocab is 5 */

    pool_stop();
    pool_start(2u);

    /* ---- solo reference: one sequence at a time, its own store. */
    u32 solo[3][8];
    u32 solo_n[3];
    for (u32 i = 0; i < NREQ; i++) {
        Kv* kv = kv_open(model_n_layer(m), model_max_ctx(m), model_n_head_kv(m),
                         model_head_dim(m));
        solo_n[i] = model_generate(m, kv, prompts[i], nprompt[i], NGEN,
                                   /*eos_id=*/-1, solo[i], NULL, -1, NULL, NULL);
        kv_close(kv);
    }

    /* ---- the same three through the scheduler, sharing ONE pool. */
    KvStore* store = kv_store_open(model_n_layer(m), 64u, model_n_head_kv(m),
                                   model_head_dim(m), KV_F32);
    ServeHost* host = serve_host_open(m);
    SchedHost  sh = serve_host(host);
    Sched*     s = sched_open(store, model_max_ctx(m), &sh);

    for (u32 i = 0; i < NREQ; i++) {
        SchedRequest r = {100 + i, prompts[i], nprompt[i], NGEN, -1};
        ASSERT_TRUE(sched_submit(s, &r));
    }

    u32 got[3][8];
    u32 got_n[3] = {0, 0, 0};
    u32 guard = 0;
    while (sched_pending(s) && guard++ < 500) {
        sched_step(s);
        for (u32 i = 0; i < NREQ; i++) {
            SchedFinish f;
            got_n[i] += sched_poll(s, 100 + i, got[i] + got_n[i],
                                   8 - got_n[i], &f);
        }
    }
    ASSERT_TRUE(guard < 500);

    /* The anti-vacuity pair: the host must actually have run forwards, and
     * they must have carried rows. "The streams matched" is also true of two
     * runs that both produced nothing. */
    ASSERT_TRUE(serve_host_steps(host) > 0);
    ASSERT_TRUE(serve_host_rows(host) > 0);

    for (u32 i = 0; i < NREQ; i++) {
        if (got_n[i] != solo_n[i]) {
            printf("  req %u: scheduler %u ids, solo %u\n", i, got_n[i],
                   solo_n[i]);
        }
        ASSERT_EQ_U32(got_n[i], solo_n[i]);
        for (u32 k = 0; k < solo_n[i]; k++) {
            if (got[i][k] != solo[i][k]) {
                printf("  req %u id %u: scheduler %u, solo %u\n", i, k,
                       got[i][k], solo[i][k]);
            }
            ASSERT_EQ_U32(got[i][k], solo[i][k]);
        }
    }

    /* And the three must not be identical to each other, or a host that mixed
     * up rows would satisfy every comparison above. */
    ASSERT_TRUE(memcmp(solo[0], solo[1], solo_n[0] * sizeof(u32)) != 0);

    sched_close(s);
    serve_host_close(host);
    kv_store_close(store);
    pool_stop();
    model_close(m);
}

/* ===========================================================================
 * M5 E1 -- a pool claim that STRADDLES two sequences maps to the right ones.
 *
 * E1 dispatches the whole batch's attention over a flattened (sequence, unit)
 * space, so one claim can begin inside sequence s and end inside s+1, and the
 * worker walks it as per-sequence segments. Get that arithmetic wrong and a
 * worker indexes past its sequence's arrays into the neighbour's -- valid
 * memory, plausible floats, silently different output.
 *
 * ▶ THE POINT OF THIS TEST IS THAT NOTHING ELSE REACHES THE PATH. A claim
 * straddles only when the claim size does not divide the units one sequence
 * contributes, and the shipping claim size is ONE, which divides everything.
 * So the straddling branch is unreachable in the default configuration and is
 * reachable for anyone who sets STOKE_ATT_CHUNK -- the worst combination:
 * untested and live. On this fixture n_head_kv is 1, so a sequence contributes
 * exactly one unit and a claim of 2 straddles by construction.
 *
 * Compared against claim size 1 -- the arrangement whose correctness the whole
 * rest of the suite already covers -- rather than against a fixed expectation.
 * ===========================================================================
 */
TEST(m5_e1_a_pool_claim_straddling_two_sequences_reads_the_right_sequence) {
    write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
    Model* m = model_open(MODEL_FIXTURE_PATH);

    const u32 n_embd = model_n_embd(m);
    const u32 NSEQ = 3;
    const u32 NTOK = 40; /* past ATT_MIN_POS so the dispatch actually happens */
    ASSERT_EQ_U32(model_n_head_kv(m), 1u); /* the premise of "2 straddles" */

    pool_stop();
    pool_start(4u);

    size_t bytes = (size_t)NSEQ * n_embd * sizeof(f32);
    f32*   outs[2];
    outs[0] = (f32*)xmalloc(bytes);
    outs[1] = (f32*)xmalloc(bytes);

    const u32 claims[2] = {1u, 2u};
    for (u32 c = 0; c < 2u; c++) {
        model_att_set_chunk(claims[c]);
        Kv*  kv[3];
        f32* xp[3];
        u32  tok[3], pos[3];
        for (u32 s = 0; s < NSEQ; s++) {
            kv[s] = kv_open(model_n_layer(m), model_max_ctx(m),
                            model_n_head_kv(m), model_head_dim(m));
            xp[s] = outs[c] + (size_t)s * n_embd;
        }
        /* Each sequence walks a DIFFERENT token pattern, so a worker that read
         * the neighbour's scratch would produce a different answer rather than
         * accidentally the same one. */
        for (u32 t = 0; t < NTOK; t++) {
            for (u32 s = 0; s < NSEQ; s++) {
                tok[s] = (t + s) % 5u; /* vocab is 5 */
                pos[s] = t;
            }
            model_forward_batch(m, kv, tok, pos, NSEQ, xp);
        }
        for (u32 s = 0; s < NSEQ; s++) kv_close(kv[s]);
    }
    model_att_set_chunk(1u); /* restore the shipping default */
    pool_stop();

    ASSERT_TRUE(memcmp(outs[0], outs[1], bytes) == 0);

    /* And the three sequences must not be identical to each other, or a worker
     * reading the wrong one would be invisible to the comparison above. */
    ASSERT_TRUE(memcmp(outs[0], outs[0] + n_embd,
                       (size_t)n_embd * sizeof(f32)) != 0);

    free(outs[0]);
    free(outs[1]);
    model_close(m);
}

/* ===========================================================================
 * M5 E3 -- chunked prefill writes the same KV cache as position-by-position.
 *
 * WHAT IS BEING CLAIMED. Prefill now runs a CHUNK of consecutive positions as
 * batch rows so each weight row is streamed once for the whole chunk instead of
 * once per position. The claim is that this changes no arithmetic: a chunk is a
 * batch whose rows happen to be consecutive positions of ONE sequence, and
 * forward_batch_body does not distinguish that from one position of several.
 *
 * ▶ THE COMPARISON IS THE WHOLE KV CACHE, NOT THE TOKEN STREAM. The fixture has
 * a 5-token vocabulary and an argmax dominated by recent positions, and a
 * measured mutation has already shown that a fixture-level token comparison
 * CANNOT see K/V that was written wrong (see the adoption check in
 * generate_batch). Every K and V element at every (layer, kv_head, position)
 * is compared instead -- that is where a chunking error would land, and it is
 * visible there long before it reaches a token.
 *
 * ▶ WHAT THIS TEST CANNOT SEE, stated rather than left implicit. The fixture's
 * weights are F32, so both widths take rows_f32_batch and the K-quant FUSED
 * batch kernel is never entered. On the real model n_seq=1 routes to the solo
 * kernel and n_seq>1 to the xn kernel -- genuinely different code. That
 * equivalence is M4 gate item [2]'s (31 streams byte-identical to solo over
 * N in {1,2,4,8,16}), and chunk 128 inherits it by tiling: the xn path tiles
 * sequences at KERNELS_BATCH_TILE = 16, so 128 rows are 8 tiles of exactly the
 * N=16 case that gate covered. The real-model byte-compare is what checks it
 * end to end; this test checks the plumbing the gate cannot reach cheaply.
 * ===========================================================================
 */
TEST(m5_e3_chunked_prefill_writes_the_same_kv_as_position_by_position) {
    write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
    Model* m = model_open(MODEL_FIXTURE_PATH);

    const u32 n_embd = model_n_embd(m);
    const u32 NPROMPT = 40; /* > 2 blocks of 16, and not a multiple of one */
    u32       ids[40];
    for (u32 t = 0; t < NPROMPT; t++) ids[t] = t % 5u; /* vocab is 5 */

    pool_stop();
    pool_start(4u);

    /* Chunk 16 rather than the shipping 128: the fixture's max_ctx is 64, so
     * 128 would run the whole prompt as ONE chunk and never cross a boundary --
     * and crossing boundaries is the entire mechanism under test. 40 positions
     * at 16 gives 16 + 16 + 8, i.e. two full chunks and a partial one. */
    const u32 widths[2] = {1u, 16u};
    const u32 NGEN = 4;
    /* Mirrors model.c's PREFILL_CHUNK, deliberately: if that default moves,
     * this restores the wrong value and the next test to depend on it says so.
     * Same reasoning as ATT_MIN_POS_DEFAULT in the L4 test above. */
    const u32 PREFILL_CHUNK_DEFAULT = 128u;
    Kv*       kvs[2];
    u32       out[2][4];
    u32       counts[2];

    u64 batched[2];
    for (u32 w = 0; w < 2u; w++) {
        u64 before = model_prefill_batch_rows();
        model_set_prefill_chunk(widths[w]);
        kvs[w] = kv_open(model_n_layer(m), model_max_ctx(m),
                         model_n_head_kv(m), model_head_dim(m));
        counts[w] = model_generate(m, kvs[w], ids, NPROMPT, NGEN,
                                   /*eos_id=*/-1, out[w], NULL, -1, NULL, NULL);
        batched[w] = model_prefill_batch_rows() - before;
    }
    model_set_prefill_chunk(PREFILL_CHUNK_DEFAULT);
    pool_stop();
    (void)n_embd;

    /* THE ANTI-VACUITY CHECK, and it is the reason this test is worth having.
     * Width 1 must run NO position through the batched path and width 16 must
     * run ALL of them (16 + 16 + 8). If the chunker silently never engaged --
     * clamped width, prompt shorter than a chunk, a dump forcing the solo path
     * -- every comparison below would be a code path against itself. */
    if (batched[0] != 0 || batched[1] != NPROMPT) {
        printf("  batched rows: width1=%llu (want 0), width16=%llu (want %u)\n",
               (unsigned long long)batched[0], (unsigned long long)batched[1],
               NPROMPT);
    }
    ASSERT_TRUE(batched[0] == 0);
    ASSERT_TRUE(batched[1] == (u64)NPROMPT);

    ASSERT_EQ_U32(counts[0], counts[1]);

    /* The token stream too -- weak evidence on a 5-token vocabulary, which is
     * exactly why it is the SECOND check and not the only one. */
    ASSERT_TRUE(memcmp(out[0], out[1], sizeof(out[0])) == 0);

    /* EVERY K and V element the prompt produced, compared elementwise. */
    u64 same = 0, diff = 0;
    for (u32 L = 0; L < model_n_layer(m); L++) {
        for (u32 h = 0; h < model_n_head_kv(m); h++) {
            for (u32 p = 0; p < NPROMPT; p++) {
                KvSlot a = kv_view(kvs[0], L, h, p);
                KvSlot b = kv_view(kvs[1], L, h, p);
                for (u32 d = 0; d < model_head_dim(m); d++) {
                    if (a.k[d] == b.k[d]) same++; else diff++;
                    if (a.v[d] == b.v[d]) same++; else diff++;
                }
            }
        }
    }
    if (diff != 0) {
        printf("  %llu of %llu KV elements differ between chunk 1 and 16\n",
               (unsigned long long)diff, (unsigned long long)(same + diff));
    }
    ASSERT_TRUE(diff == 0);
    /* Anti-vacuity: the loops above must actually have compared something, and
     * the cache must not be uniformly zero (which would compare equal while
     * meaning nothing was written). */
    ASSERT_TRUE(same > 0);
    u32 nonzero = 0;
    for (u32 p = 0; p < NPROMPT; p++) {
        KvSlot a = kv_view(kvs[0], 0, 0, p);
        for (u32 d = 0; d < model_head_dim(m); d++) {
            if (a.k[d] != 0.0f) nonzero++;
        }
    }
    ASSERT_TRUE(nonzero > 0u);

    kv_close(kvs[0]);
    kv_close(kvs[1]);
    model_close(m);
}

TEST(m4_interleaved_sequences_on_a_shared_store_are_bit_identical_to_solo) {
    write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
    Model* m = model_open(MODEL_FIXTURE_PATH);

    const u32 n_embd = model_n_embd(m);
    const u32 NTOK = 5;
    const u32 tokens_a[5] = {0, 1, 2, 3, 4};
    const u32 tokens_b[5] = {4, 3, 2, 1, 0};
    const u32 tokens_c[5] = {2, 2, 0, 4, 1};

    f32* solo = (f32*)xmalloc((size_t)NTOK * n_embd * sizeof(f32));
    f32* co   = (f32*)xmalloc((size_t)NTOK * n_embd * sizeof(f32));

    /* A alone, in its own store. */
    {
        Kv* kv = kv_open(model_n_layer(m), model_max_ctx(m),
                         model_n_head_kv(m), model_head_dim(m));
        for (u32 t = 0; t < NTOK; t++)
            model_forward_position(m, kv, tokens_a[t], t, NULL,
                                   solo + (size_t)t * n_embd);
        kv_close(kv);
    }

    /* A interleaved with B and C on ONE shared store, at three different
     * positions in the round-robin. */
    for (u32 slot = 0; slot < 3; slot++) {
        KvStore* st = kv_store_open(model_n_layer(m),
                                    3u * (model_max_ctx(m) / BP_BLOCK_TOKENS + 1u),
                                    model_n_head_kv(m), model_head_dim(m), KV_F32);
        Kv* kv[3];
        for (u32 i = 0; i < 3; i++) kv[i] = kv_open_in(st, model_max_ctx(m));
        ASSERT_TRUE(kv[0] && kv[1] && kv[2]);

        const u32* toks[3];
        toks[slot] = tokens_a;
        toks[(slot + 1u) % 3u] = tokens_b;
        toks[(slot + 2u) % 3u] = tokens_c;

        /* Driven through model_forward_batch, so the N-scratch plumbing is
         * what is under test -- each sequence gets its own SeqScratch and
         * none of them may reach another's. Stepping them by hand would
         * have exercised only the single m->scratch. */
        f32* scratchpad[3];
        for (u32 i = 0; i < 3; i++)
            scratchpad[i] = (f32*)xmalloc((size_t)n_embd * sizeof(f32));

        for (u32 t = 0; t < NTOK; t++) {
            u32 tok[3], posn[3];
            f32* dst[3];
            for (u32 i = 0; i < 3; i++) {
                tok[i]  = toks[i][t];
                posn[i] = t;
                dst[i]  = (i == slot) ? co + (size_t)t * n_embd : scratchpad[i];
            }
            model_forward_batch(m, kv, tok, posn, 3u, dst);
        }
        for (u32 i = 0; i < 3; i++) free(scratchpad[i]);

        ASSERT_TRUE(memcmp(solo, co, (size_t)NTOK * n_embd * sizeof(f32)) == 0);

        for (u32 i = 0; i < 3; i++) kv_close(kv[i]);
        kv_store_close(st);
    }

    free(solo);
    free(co);
    model_close(m);
}

/* ===========================================================================
 * M4: a sequence's output must not depend on the batch WIDTH -- including
 * when the width GROWS between calls.
 *
 * Two things are under test here, neither covered by the interleaved test
 * above, which only ever runs N=3.
 *
 * (1) BATCH-COMPOSITION INDEPENDENCE ACROSS DIFFERENT N. ROADMAP section M4
 *     wants A bit-identical whatever N is; a single fixed N cannot show that.
 *
 * (2) THE SCRATCH POINTER TABLE SURVIVING A REALLOC. model_forward_batch
 *     keeps a parallel array of &scratch_n[s] for the seqs-inner body to
 *     index. Growing the batch xreallocs scratch_n, which may MOVE it, so
 *     every entry of that array has to be rewritten -- not just the newly
 *     added ones. Refilling only the new range leaves the earlier entries
 *     pointing into freed memory: a use-after-free NO fixed-width batch can
 *     reach, because it needs one call to populate the table and a second,
 *     WIDER call to move it.
 *
 * Hence widths that ASCEND (2 -> 5), and A at index 0 -- the entry a partial
 * refill strands. Under ASan the stranded read is loud; without it, A's
 * hidden state simply stops matching solo.
 * ===========================================================================
 */
TEST(m4_a_sequence_is_bit_identical_across_batch_widths_including_growth) {
    write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
    Model* m = model_open(MODEL_FIXTURE_PATH);

    const u32 n_embd = model_n_embd(m);
    const u32 NTOK = 5;
    const u32 tokens_a[5] = {0, 1, 2, 3, 4};

    f32* solo = (f32*)xmalloc((size_t)NTOK * n_embd * sizeof(f32));
    {
        Kv* kv = kv_open(model_n_layer(m), model_max_ctx(m),
                         model_n_head_kv(m), model_head_dim(m));
        for (u32 t = 0; t < NTOK; t++)
            model_forward_position(m, kv, tokens_a[t], t, NULL,
                                   solo + (size_t)t * n_embd);
        kv_close(kv);
    }

    /* ASCENDING, in this order: the second call is the one that grows the
     * table. Reversing them would test nothing. */
    const u32 widths[2] = {2u, 5u};
    for (u32 w = 0; w < 2; w++) {
        const u32 n = widths[w];
        KvStore*  st =
            kv_store_open(model_n_layer(m),
                          n * (model_max_ctx(m) / BP_BLOCK_TOKENS + 1u),
                          model_n_head_kv(m), model_head_dim(m), KV_F32);

        Kv**  kv   = (Kv**)xmalloc((size_t)n * sizeof(Kv*));
        f32** dst  = (f32**)xmalloc((size_t)n * sizeof(f32*));
        u32*  tok  = (u32*)xmalloc((size_t)n * sizeof(u32));
        u32*  posn = (u32*)xmalloc((size_t)n * sizeof(u32));
        for (u32 i = 0; i < n; i++) {
            kv[i] = kv_open_in(st, model_max_ctx(m));
            ASSERT_TRUE(kv[i] != NULL);
            dst[i] = (f32*)xmalloc((size_t)n_embd * sizeof(f32));
        }

        f32* co = (f32*)xmalloc((size_t)NTOK * n_embd * sizeof(f32));
        for (u32 t = 0; t < NTOK; t++) {
            for (u32 i = 0; i < n; i++) {
                /* A at index 0; the others run unrelated streams, so a bleed
                 * between them is a WRONG value rather than a coincidence. */
                tok[i]  = (i == 0) ? tokens_a[t] : ((tokens_a[t] + i) % 5u);
                posn[i] = t;
            }
            model_forward_batch(m, kv, tok, posn, n, dst);
            memcpy(co + (size_t)t * n_embd, dst[0],
                   (size_t)n_embd * sizeof(f32));
        }

        ASSERT_TRUE(memcmp(solo, co, (size_t)NTOK * n_embd * sizeof(f32)) == 0);

        for (u32 i = 0; i < n; i++) { kv_close(kv[i]); free(dst[i]); }
        free(co); free(kv); free(dst); free(tok); free(posn);
        kv_store_close(st);
    }

    free(solo);
    model_close(m);
}

/* ===========================================================================
 * M4 gate item 2 (ruling section 8.4): a sequence's GENERATED TOKEN STREAM
 * must not depend on the batch it ran in.
 *
 * The two tests above compare HIDDEN STATES for a fixed number of hand-driven
 * positions. That is the arithmetic half. This is the orchestration half, and
 * it is a different failure surface: position bookkeeping per sequence, where
 * prefill ends and decode begins, which sequence's logits feed which argmax,
 * and -- once a sequence retires -- which batch slot everyone else moves to.
 * None of that is reachable by driving model_forward_batch by hand, because
 * the harness supplies the positions itself.
 *
 * PROMPT LENGTHS DELIBERATELY DIFFER (2 / 5 / 3). Equal-length prompts make
 * every sequence cross the prefill/decode boundary on the same step, which is
 * exactly the case a shared "are we still prefilling" flag would survive.
 * ===========================================================================
 */
static u32 gen_solo(Model* m, const u32* prompt, u32 n_prompt, u32 n_gen,
                    i64 eos_id, u32* out) {
    Kv* kv = kv_open(model_n_layer(m), model_max_ctx(m), model_n_head_kv(m),
                     model_head_dim(m));
    u32 n = model_generate(m, kv, prompt, n_prompt, n_gen, eos_id, out, NULL,
                           -1, NULL, NULL);
    kv_close(kv);
    return n;
}

#define BATCH_NSEQ 3u

TEST(m4_batched_generation_is_token_identical_to_solo_generation) {
    write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
    Model* m = model_open(MODEL_FIXTURE_PATH);

    const u32 p0[2] = {0, 3};
    const u32 p1[5] = {4, 1, 1, 2, 0};
    const u32 p2[3] = {2, 2, 4};
    const u32* prompts[BATCH_NSEQ]  = {p0, p1, p2};
    const u32  n_prompt[BATCH_NSEQ] = {2u, 5u, 3u};
    const u32  N_GEN = 6u;

    u32 solo[BATCH_NSEQ][6];
    for (u32 s = 0; s < BATCH_NSEQ; s++) {
        u32 n = gen_solo(m, prompts[s], n_prompt[s], N_GEN, -1, solo[s]);
        ASSERT_EQ_U32(n, N_GEN);
    }

    KvStore* st = kv_store_open(model_n_layer(m),
                                BATCH_NSEQ * (model_max_ctx(m) / BP_BLOCK_TOKENS + 1u),
                                model_n_head_kv(m), model_head_dim(m), KV_F32);
    Kv*           kv[BATCH_NSEQ];
    u32           got[BATCH_NSEQ][6];
    ModelBatchSeq bs[BATCH_NSEQ];
    for (u32 s = 0; s < BATCH_NSEQ; s++) {
        kv[s] = kv_open_in(st, model_max_ctx(m));
        ASSERT_TRUE(kv[s] != NULL);
        bs[s].prompt_ids = prompts[s];
        bs[s].n_prompt   = n_prompt[s];
        bs[s].n_gen      = N_GEN;
        bs[s].out_ids    = got[s];
        bs[s].n_out      = 0;
    }

    model_generate_batch(m, kv, bs, BATCH_NSEQ, -1, false, NULL);

    for (u32 s = 0; s < BATCH_NSEQ; s++) {
        ASSERT_EQ_U32(bs[s].n_out, N_GEN);
        ASSERT_TRUE(memcmp(got[s], solo[s], N_GEN * sizeof(u32)) == 0);
    }

    for (u32 s = 0; s < BATCH_NSEQ; s++) kv_close(kv[s]);
    kv_store_close(st);
    model_close(m);
    remove(MODEL_FIXTURE_PATH);
}

/* ===========================================================================
 * M4: a sequence that finishes while others are still running must not
 * disturb them.
 *
 * The batch is COMPACTED when a sequence retires -- the survivors move to
 * different slots, and a slot is what indexes the per-sequence scratch. A
 * driver that kept stepping a finished sequence, or that mixed up who moved
 * where, produces a plausible stream for the survivors rather than a crash.
 *
 * n_gen differs per sequence rather than using eos, so retirement order is
 * FIXED (0, then 2, then 1) instead of depending on what the fixture happens
 * to generate. Slot 0 retiring first is the case that shifts everyone.
 * ===========================================================================
 */
TEST(m4_a_sequence_retiring_early_does_not_disturb_the_rest_of_the_batch) {
    write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
    Model* m = model_open(MODEL_FIXTURE_PATH);

    const u32 p0[2] = {0, 3};
    const u32 p1[5] = {4, 1, 1, 2, 0};
    const u32 p2[3] = {2, 2, 4};
    const u32* prompts[BATCH_NSEQ]  = {p0, p1, p2};
    const u32  n_prompt[BATCH_NSEQ] = {2u, 5u, 3u};
    const u32  n_gen[BATCH_NSEQ]    = {2u, 7u, 4u};

    u32 solo[BATCH_NSEQ][7];
    for (u32 s = 0; s < BATCH_NSEQ; s++) {
        u32 n = gen_solo(m, prompts[s], n_prompt[s], n_gen[s], -1, solo[s]);
        ASSERT_EQ_U32(n, n_gen[s]);
    }

    KvStore* st = kv_store_open(model_n_layer(m),
                                BATCH_NSEQ * (model_max_ctx(m) / BP_BLOCK_TOKENS + 1u),
                                model_n_head_kv(m), model_head_dim(m), KV_F32);
    Kv*           kv[BATCH_NSEQ];
    u32           got[BATCH_NSEQ][7];
    ModelBatchSeq bs[BATCH_NSEQ];
    for (u32 s = 0; s < BATCH_NSEQ; s++) {
        kv[s] = kv_open_in(st, model_max_ctx(m));
        ASSERT_TRUE(kv[s] != NULL);
        bs[s].prompt_ids = prompts[s];
        bs[s].n_prompt   = n_prompt[s];
        bs[s].n_gen      = n_gen[s];
        bs[s].out_ids    = got[s];
        bs[s].n_out      = 0;
    }

    model_generate_batch(m, kv, bs, BATCH_NSEQ, -1, false, NULL);

    for (u32 s = 0; s < BATCH_NSEQ; s++) {
        ASSERT_EQ_U32(bs[s].n_out, n_gen[s]);
        ASSERT_TRUE(memcmp(got[s], solo[s], n_gen[s] * sizeof(u32)) == 0);
    }

    for (u32 s = 0; s < BATCH_NSEQ; s++) kv_close(kv[s]);
    kv_store_close(st);
    model_close(m);
    remove(MODEL_FIXTURE_PATH);
}

/* ===========================================================================
 * M4: eos stops a sequence inside a batch exactly where it stops it alone.
 *
 * The reference runs are made with the SAME eos_id, so the comparison holds
 * whichever sequences happen to emit it -- the fixture's vocabulary is 5 ids
 * wide and collisions are likely, and a test that depended on only ONE
 * sequence hitting eos would be testing the fixture, not the driver.
 * ===========================================================================
 */
TEST(m4_batched_generation_stops_a_sequence_at_eos) {
    write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
    Model* m = model_open(MODEL_FIXTURE_PATH);

    const u32 p0[2] = {0, 3};
    const u32 p1[5] = {4, 1, 1, 2, 0};
    const u32* prompts[2]  = {p0, p1};
    const u32  n_prompt[2] = {2u, 5u};
    const u32  N_GEN = 6u;

    /* Learn an id the first sequence actually emits, then make it eos. */
    u32 probe[6];
    ASSERT_EQ_U32(gen_solo(m, p0, 2u, N_GEN, -1, probe), N_GEN);
    const i64 eos_id = (i64)probe[2];

    u32 solo[2][6], n_solo[2];
    for (u32 s = 0; s < 2; s++) {
        n_solo[s] = gen_solo(m, prompts[s], n_prompt[s], N_GEN, eos_id, solo[s]);
    }
    ASSERT_TRUE(n_solo[0] < N_GEN); /* the eos id is reachable, or this is vacuous */

    KvStore* st = kv_store_open(model_n_layer(m),
                                2u * (model_max_ctx(m) / BP_BLOCK_TOKENS + 1u),
                                model_n_head_kv(m), model_head_dim(m), KV_F32);
    Kv*           kv[2];
    u32           got[2][6];
    ModelBatchSeq bs[2];
    for (u32 s = 0; s < 2; s++) {
        kv[s] = kv_open_in(st, model_max_ctx(m));
        ASSERT_TRUE(kv[s] != NULL);
        bs[s].prompt_ids = prompts[s];
        bs[s].n_prompt   = n_prompt[s];
        bs[s].n_gen      = N_GEN;
        bs[s].out_ids    = got[s];
        bs[s].n_out      = 0;
    }

    model_generate_batch(m, kv, bs, 2u, eos_id, false, NULL);

    for (u32 s = 0; s < 2; s++) {
        ASSERT_EQ_U32(bs[s].n_out, n_solo[s]);
        ASSERT_TRUE(memcmp(got[s], solo[s], n_solo[s] * sizeof(u32)) == 0);
    }

    for (u32 s = 0; s < 2; s++) kv_close(kv[s]);
    kv_store_close(st);
    model_close(m);
    remove(MODEL_FIXTURE_PATH);
}

/* ===========================================================================
 * M4 gate item 3 -- prefix dedup, at the level where it can be wrong about
 * TOKENS rather than about blocks.
 *
 * tests/test_kvshare.c proves the pool half: an adopted span is physically
 * the same storage, the last-token rule holds, blocks come back. What it
 * cannot show is the claim that matters here -- that skipping the forward
 * pass over an adopted prefix changes NOTHING about what comes out.
 *
 * The argument is that a chained-hash match certifies the whole preceding
 * prefix, so the shared block covers the same ABSOLUTE positions in both
 * sequences, and the forward pass depends only on (token id, position, prior
 * KV). That is a proof about the design; this is the check on the code.
 *
 * THE PROMPTS DIVERGE AFTER THE SHARED BLOCK ON PURPOSE. Two identical
 * prompts would produce identical streams whether or not dedup did anything,
 * so the test would pass with kv_adopt_prefix returning garbage.
 *
 * ⚠ AND HERE IS WHAT THIS TEST CANNOT DO, MEASURED RATHER THAN GUESSED.
 * Mutating kv_adopt_prefix to report positions as resident WITHOUT adopting
 * the block -- so the prefill skips 16 positions whose KV was never written
 * -- leaves this test at 26/26. The fixture's vocabulary is 5 ids wide and
 * its attention is dominated by recent positions, so an argmax comparison
 * simply cannot resolve it. tests/test_kvshare.c DOES catch that mutation
 * (two of its checks fail), and model_generate_batch now refuses the
 * inconsistency outright via kv_len. Recorded here so nobody reads this
 * test as covering more than it does: what it covers is that the dedup path
 * RUNS, reports what it did, and does not perturb the streams on a model
 * small enough to test cheaply. The claim on a real model is gate item 3's
 * spot check, and it is not optional for that reason.
 * ===========================================================================
 */
TEST(m4_prefix_dedup_reproduces_the_undeduped_streams_exactly) {
    write_fixture(MODEL_FIXTURE_PATH, MUT_NONE);
    Model* m = model_open(MODEL_FIXTURE_PATH);

    /* 20 ids: one whole block (16) plus a partial tail. A shares the first
     * 18 with B and diverges inside the tail, so exactly one block can be
     * adopted -- and the last-token rule is not what limits it. */
    u32 pa[20], pb[20];
    for (u32 i = 0; i < 20; i++) pa[i] = (i * 3u + 1u) % FIX_VOCAB;
    memcpy(pb, pa, sizeof(pa));
    pb[18] = (pa[18] + 1u) % FIX_VOCAB;
    pb[19] = (pa[19] + 2u) % FIX_VOCAB;

    const u32 N_GEN = 4u, CTX = 24u;
    u32 plain[2][4], dedup[2][4];
    u32 adopted[2] = {99u, 99u};

    for (u32 pass = 0; pass < 2; pass++) {
        bool     use_dedup = (pass == 1);
        KvStore* st = kv_store_open(model_n_layer(m),
                                    2u * kv_blocks_for_ctx(CTX),
                                    model_n_head_kv(m), model_head_dim(m), KV_F32);
        Kv*           kv[2];
        ModelBatchSeq bs[2];
        for (u32 s = 0; s < 2; s++) {
            kv[s] = kv_open_in(st, CTX);
            ASSERT_TRUE(kv[s] != NULL);
            bs[s].prompt_ids = (s == 0) ? pa : pb;
            bs[s].n_prompt   = 20u;
            bs[s].n_gen      = N_GEN;
            bs[s].out_ids    = use_dedup ? dedup[s] : plain[s];
            bs[s].n_out      = 0;
            bs[s].n_adopted  = 0;
        }
        model_generate_batch(m, kv, bs, 2u, -1, use_dedup, NULL);
        for (u32 s = 0; s < 2; s++) {
            ASSERT_EQ_U32(bs[s].n_out, N_GEN);
            if (use_dedup) adopted[s] = bs[s].n_adopted;
            kv_close(kv[s]);
        }
        kv_store_close(st);
    }

    /* Dedup FIRED -- asserted, not assumed. A run that silently adopted
     * nothing would satisfy every equality below while testing nothing, which
     * is exactly how the op storm reported dedup_hit=0 for a while. */
    ASSERT_EQ_U32(adopted[0], 0u);   /* nothing resident yet for the first */
    ASSERT_EQ_U32(adopted[1], 16u);  /* one whole block, and only one */

    /* And the streams are the ones it was supposed to reproduce. */
    ASSERT_TRUE(memcmp(dedup[0], plain[0], N_GEN * sizeof(u32)) == 0);
    ASSERT_TRUE(memcmp(dedup[1], plain[1], N_GEN * sizeof(u32)) == 0);

    /* Guard: if the two prompts happened to generate the same stream, the
     * comparison above would hold even with the sequences confused. */
    ASSERT_TRUE(memcmp(plain[0], plain[1], N_GEN * sizeof(u32)) != 0);

    model_close(m);
    remove(MODEL_FIXTURE_PATH);
}

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
