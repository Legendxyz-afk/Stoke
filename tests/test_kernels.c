/* tests/test_kernels.c -- exercises the .c files under src/kernels/ (M2
 * scalar forward-pass kernels) against hand-computed golden vectors (M2
 * op-spec design note §9, "L0"). No GGUF file is read anywhere in this
 * binary -- matvec's tests build tiny synthetic GgufTensor structs in
 * memory, matching the spawn-self/fake-tensor idiom the brief calls out
 * from tests/test_gguf.c.
 *
 * Every non-trivial golden here was independently computed by a python
 * transcript (double precision inputs, float32 throughout the actual
 * arithmetic via numpy.float32, matching this file's f32 arithmetic
 * step-for-step) pasted into .superpowers/sdd/m2-w2-report.md -- this file
 * only carries the resulting constants. Pure add/sub/mul/div/sqrt chains
 * (f16 bit patterns, the Q8_0 block, matvec, vec_dot_f32, rmsnorm) are
 * asserted exactly: those operations are IEEE754 correctly-rounded, so a
 * step-matched transcript reproduces the C result bit-for-bit. Anything
 * that calls a transcendental libm function (rope_neox's cosf/sinf/powf,
 * softmax/silu/swiglu's expf) is asserted with the tolerance the M2 W2
 * brief specifies for rope (1e-5 relative, extended here to the other
 * transcendental kernels too): IEEE754 does not mandate correctly-rounded
 * transcendentals, so a different libm can legitimately differ by a few
 * ULPs from numpy's.
 */
#include "common.h"

#include "kernels/kernels.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

/* ===========================================================================
 * Child-mode: the one scenario that must fatal (matvec on an unsupported
 * tensor type). Spawn-self pattern copied from tests/test_gguf.c's
 * gguf_test_child_main/spawn_scenario -- fatal() calls exit(1) directly, so
 * a test that must keep running afterwards can only observe it by running
 * that one call in a child process and inspecting exit code + stderr from
 * the outside.
 * ===========================================================================
 */
static int kernels_test_child_main(int argc, char** argv) {
    if (argc < 3 || strcmp(argv[1], "--scenario") != 0) {
        fprintf(stderr, "test_kernels child: bad invocation (want --scenario <name>)\n");
        return 2;
    }
    const char* s = argv[2];

    if (strcmp(s, "matvec_unsupported_type") == 0) {
        /* A real ggml type id (Q4_K) that is simply not one of the two
         * M2 dispatches -- no real file needed, a tiny fake GgufTensor is
         * enough (brief's own suggested idiom). */
        f32 fake_row[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        GgufTensor w;
        memset(&w, 0, sizeof w);
        w.name    = "fake.q4k.weight";
        w.type    = GGUF_GGML_TYPE_Q4_K;
        w.n_dims  = 2;
        w.dims[0] = 4;
        w.dims[1] = 1;
        w.data    = fake_row;
        w.nbytes  = sizeof fake_row;
        w.offset  = 0;

        f32 x[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        f32 y[1];
        matvec(&w, x, 4, y, 1, NULL);
    } else if (strcmp(s, "matvec_n_in_mismatch") == 0) {
        /* dims[0]=3 but the caller (wrongly) claims n_in=4 -- matvec must
         * fatal on the shape mismatch rather than trust the caller and
         * walk off the end of a 3-wide row. */
        f32 fake_row[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        GgufTensor w;
        memset(&w, 0, sizeof w);
        w.name    = "fake.shape.weight";
        w.type    = GGUF_GGML_TYPE_F32;
        w.n_dims  = 2;
        w.dims[0] = 3;
        w.dims[1] = 2;
        w.data    = fake_row;
        w.nbytes  = sizeof fake_row;
        w.offset  = 0;

        f32 x[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        f32 y[2];
        matvec(&w, x, 4, y, 2, NULL);
    } else if (strcmp(s, "matvec_q8_0_null_scratch") == 0) {
        /* A valid Q8_0 tensor in every respect EXCEPT that the caller passes
         * no activation scratch. Must fatal rather than fall back. */
        u8 blocks[34] = {0};
        blocks[0] = 0x00u;
        blocks[1] = 0x3Cu; /* d = 1.0 */
        GgufTensor w;
        memset(&w, 0, sizeof w);
        w.name    = "fake.q8_0.weight";
        w.type    = GGUF_GGML_TYPE_Q8_0;
        w.n_dims  = 2;
        w.dims[0] = 32;
        w.dims[1] = 1;
        w.data    = blocks;
        w.nbytes  = sizeof blocks;

        f32 x[32];
        for (int i = 0; i < 32; i++) {
            x[i] = 1.0f;
        }
        f32 y[1];
        matvec(&w, x, 32, y, 1, NULL);
    } else if (strcmp(s, "dequant_row_q8_0_bad_n") == 0) {
        u8 block[34] = {0};
        f32 out[32];
        dequant_row_q8_0(block, 31, out); /* 31 is not a multiple of 32 */
    } else {
        fprintf(stderr, "test_kernels child: unknown scenario '%s'\n", s);
        return 2;
    }

    printf("test_kernels child: scenario '%s' did NOT fatal\n", s);
    return 0;
}
#define TEST_CHILD_MAIN kernels_test_child_main
#include "test.h"

/* ===========================================================================
 * Spawn helper -- identical Windows cmd.exe outer-quote fix as
 * tests/test_gguf.c's spawn_scenario (this repo lives under a path with a
 * space, "...\New Drive\...", so the fix is required here too; see that
 * file's comment, or tests/test_common.c's, for the full quoting proof).
 * ===========================================================================
 */
static int spawn_scenario(const char* scenario, char* err_buf, size_t err_buf_sz) {
#ifdef _WIN32
    char exe_path_buf[4096];
    DWORD full_len = GetFullPathNameA(g_test_argv0, (DWORD)sizeof exe_path_buf,
                                       exe_path_buf, NULL);
    if (full_len == 0 || full_len >= sizeof exe_path_buf) {
        fatal("test_kernels: GetFullPathNameA failed for '%s'", g_test_argv0);
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

static void assert_scenario_fatals(const char* scenario, const char* msg_substr) {
    char errbuf[1024];
    int rc = spawn_scenario(scenario, errbuf, sizeof errbuf);
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
 * Test-only helpers: exact bit extraction + combined-tolerance comparison.
 * ===========================================================================
 */

static u32 f32_bits(f32 v) {
    u32 b;
    memcpy(&b, &v, sizeof b);
    return b;
}

/* |got - want| <= atol + rtol*|want| -- mirrors numpy.isclose's formula.
 * Needed for any golden that runs through a transcendental libm call
 * (expf/sinf/cosf/powf): those are IEEE754-recommended, not mandated, to
 * be correctly rounded, so a python-computed golden and the C runtime's
 * libm can legitimately differ by a handful of ULPs. atol keeps
 * near-zero goldens (e.g. silu at a large negative z) from demanding an
 * unreasonably tight relative bound. */
static int close_f32(f32 got, f32 want, f32 rtol, f32 atol) {
    f32 diff = fabsf(got - want);
    return diff <= atol + rtol * fabsf(want);
}

#define ASSERT_CLOSE_F32(got_, want_, rtol_, atol_) \
    do { \
        f32 stoke_got_  = (got_); \
        f32 stoke_want_ = (want_); \
        if (!close_f32(stoke_got_, stoke_want_, (rtol_), (atol_))) { \
            fprintf(stderr, \
                    "FAIL %s:%d: %s ~= %s (got %.9g, want %.9g)\n", \
                    __FILE__, __LINE__, #got_, #want_, \
                    (double)stoke_got_, (double)stoke_want_); \
            g_test_case_failed = 1; \
        } \
    } while (0)

/* ===========================================================================
 * Shared Q8_0 test fixtures.
 * ===========================================================================
 */

/* d = 0.75 (f16 bits 0x3A00u), 32 quants with mixed signs and the
 * requested ±127 extremes: {127,-127} then (k,-k) for k=1..15. Computed
 * and hand-annotated in the report. */
static const i8 QS_BLOCK_A[32] = {
    127, -127, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7,
    8, -8, 9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14, -14, 15, -15
};

/* d = 2.0 (f16 bits 0x4000u), all quants = 2 -- a second, arithmetically
 * trivial block used only to prove matvec's Q8_0 row-stride math walks
 * to the SECOND row correctly (its own dot product is 2*2*sum(x) =
 * 4*sum(x), independently checkable without re-deriving block A's
 * arithmetic). */
static const i8 QS_BLOCK_B[32] = {
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
};

static void build_q8_0_block(u8* out34, u16 d_bits, const i8* qs32) {
    out34[0] = (u8)(d_bits & 0xFFu);
    out34[1] = (u8)((d_bits >> 8) & 0xFFu);
    for (int i = 0; i < 32; i++) {
        out34[2 + i] = (u8)qs32[i];
    }
}

/* x[i] = (i+1)*0.5, i.e. 0.5, 1.0, 1.5, ..., 16.0 -- distinct per index
 * (catches index-order bugs) and every value exactly representable in
 * f32, so downstream products/sums stay exact too. */
static void fill_x32(f32* x) {
    for (int i = 0; i < 32; i++) {
        x[i] = (f32)(i + 1) * 0.5f;
    }
}

/* ===========================================================================
 * f16_to_f32
 * ===========================================================================
 */

TEST(f16_known_patterns) {
    ASSERT_EQ_U32(f32_bits(f16_to_f32(0x3C00u)), 0x3F800000u); /* 1.0 */
    ASSERT_EQ_U32(f32_bits(f16_to_f32(0x0001u)), 0x33800000u); /* subnormal */
    ASSERT_EQ_U32(f32_bits(f16_to_f32(0xC000u)), 0xC0000000u); /* -2.0 */
    ASSERT_EQ_U32(f32_bits(f16_to_f32(0x7BFFu)), 0x477FE000u); /* 65504.0 max normal */
    ASSERT_EQ_U32(f32_bits(f16_to_f32(0x8000u)), 0x80000000u); /* -0.0 */
    ASSERT_EQ_U32(f32_bits(f16_to_f32(0x7C00u)), 0x7F800000u); /* +inf */
    ASSERT_EQ_U32(f32_bits(f16_to_f32(0xFC00u)), 0xFF800000u); /* -inf */
    ASSERT_EQ_U32(f32_bits(f16_to_f32(0x7E00u)), 0x7FC00000u); /* nan */

    /* Semantic checks on top of the exact bit pins above: f16_to_f32
     * itself must NOT fatal on inf/nan (that check lives one layer up, in
     * dequant_row_q8_0/vec_dot_q8_0_f32) -- these prove the bare
     * converter really does hand back a usable inf/nan value rather than
     * some other bit pattern that merely happens to match by accident. */
    ASSERT_TRUE(isinf(f16_to_f32(0x7C00u)) && f16_to_f32(0x7C00u) > 0.0f);
    ASSERT_TRUE(isinf(f16_to_f32(0xFC00u)) && f16_to_f32(0xFC00u) < 0.0f);
    ASSERT_TRUE(isnan(f16_to_f32(0x7E00u)));
}

/* ===========================================================================
 * Q8_0: dequant_row_q8_0 + vec_dot_q8_0_f32
 * ===========================================================================
 */

TEST(q8_0_handcrafted_block_dequant_and_dot) {
    u8 block[34];
    build_q8_0_block(block, 0x3A00u, QS_BLOCK_A);

    f32 out[32];
    dequant_row_q8_0(block, 32, out);

    static const f32 want[32] = {
        95.25f, -95.25f, 0.75f, -0.75f, 1.5f, -1.5f, 2.25f, -2.25f,
        3.0f, -3.0f, 3.75f, -3.75f, 4.5f, -4.5f, 5.25f, -5.25f,
        6.0f, -6.0f, 6.75f, -6.75f, 7.5f, -7.5f, 8.25f, -8.25f,
        9.0f, -9.0f, 9.75f, -9.75f, 10.5f, -10.5f, 11.25f, -11.25f
    };
    for (int i = 0; i < 32; i++) {
        ASSERT_TRUE(out[i] == want[i]);
    }

    f32 x[32];
    fill_x32(x);
    f32 dot = vec_dot_q8_0_f32(block, x, 32);
    /* golden: -92.625, f32 bits 0xC2B94000 (report §2 has the full
       32-step accumulation transcript) */
    ASSERT_EQ_U32(f32_bits(dot), 0xC2B94000u);
    ASSERT_TRUE(dot == -92.625f);
}

TEST(q8_0_n_not_multiple_of_32_fatals) {
    assert_scenario_fatals("dequant_row_q8_0_bad_n",
                            "not a multiple of");
}

/* ===========================================================================
 * vec_dot_f32
 * ===========================================================================
 */

TEST(vec_dot_f32_known) {
    f32 a[3] = {1.0f, 2.0f, 3.0f};
    f32 b[3] = {4.0f, 5.0f, 6.0f};
    ASSERT_TRUE(vec_dot_f32(a, b, 3) == 32.0f);
}

/* ===========================================================================
 * matvec
 * ===========================================================================
 */

TEST(matvec_f32_dispatch) {
    /* row0 = [1,2,3], row1 = [4,5,6] (dims[0]=3 contraction, dims[1]=2 rows) */
    f32 wdata[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    GgufTensor w;
    memset(&w, 0, sizeof w);
    w.name    = "test.f32.weight";
    w.type    = GGUF_GGML_TYPE_F32;
    w.n_dims  = 2;
    w.dims[0] = 3;
    w.dims[1] = 2;
    w.data    = wdata;
    w.nbytes  = sizeof wdata;

    f32 x[3] = {1.0f, 2.0f, 3.0f};
    f32 y[2] = {0.0f, 0.0f};
    matvec(&w, x, 3, y, 2, NULL);

    ASSERT_TRUE(y[0] == 14.0f); /* 1*1+2*2+3*3 */
    ASSERT_TRUE(y[1] == 32.0f); /* 4*1+5*2+6*3 */
}

TEST(matvec_q8_0_dispatch_two_rows) {
    u8 blocks[68];
    build_q8_0_block(blocks, 0x3A00u, QS_BLOCK_A);      /* row0 = block A */
    build_q8_0_block(blocks + 34, 0x4000u, QS_BLOCK_B); /* row1 = block B */

    GgufTensor w;
    memset(&w, 0, sizeof w);
    w.name    = "test.q8_0.weight";
    w.type    = GGUF_GGML_TYPE_Q8_0;
    w.n_dims  = 2;
    w.dims[0] = 32;
    w.dims[1] = 2;
    w.data    = blocks;
    w.nbytes  = sizeof blocks;

    f32 x[32];
    fill_x32(x);
    f32 y[2];
    u8  aq[KERNELS_Q8_0_BLOCK_BYTES];
    matvec(&w, x, 32, y, 2, aq);

    /* ▶ THESE GOLDENS CHANGED AT M3, AND THE CHANGE IS THE POINT. Under M2
     * this read -92.625 and 1056.0, computed with a DENSE f32 activation.
     * The integer core quantizes x to Q8_0 first, so the arithmetic is now
     * (i32 sum of i8 products) * (d_w * d_a) -- a different, deliberately
     * lossy path that puts stoke in llama.cpp's numeric class.
     *
     * Independently derived, not read off this implementation: the same
     * python transcription of ggml's quantizer that produced the golden in
     * quantize_row_q8_0_matches_the_python_reference_byte_for_byte gives
     * d_a = 0.1259765625 (f16 0x3008), and integer sums of -988 and +4192.
     * Asserted EXACTLY -- the whole chain is integer accumulation followed
     * by two f32 multiplies, so a step-matched transcript reproduces it bit
     * for bit, exactly as this file's header requires for pure chains.
     *
     * The drift from M2 is small (-92.625 -> -93.349, 1056.0 -> 1056.1875),
     * which is the expected size of activation-quantization error and is
     * itself worth pinning: a LARGE drift here would mean a layout or
     * scale-ordering bug rather than quantization. */
    ASSERT_TRUE(y[0] == -93.3486328f);
    ASSERT_TRUE(y[1] == 1056.1875f);
}

/* The no-silent-fallback contract. A Q8_0 weight with no activation scratch
 * must FATAL, not quietly take the M2 f32 path -- two different arithmetic
 * classes reachable from the same call would make the oracle-parity
 * measurement meaningless and would be invisible in any output. */
TEST(matvec_q8_0_without_activation_scratch_fatals) {
    assert_scenario_fatals("matvec_q8_0_null_scratch",
                           "no activation scratch");
}

TEST(matvec_unsupported_type_fatals) {
    assert_scenario_fatals("matvec_unsupported_type",
                            "kernels: matvec -- tensor 'fake.q4k.weight' "
                            "has unsupported ggml type");
}

TEST(matvec_shape_mismatch_fatals) {
    assert_scenario_fatals("matvec_n_in_mismatch",
                            "kernels: matvec -- tensor 'fake.shape.weight' "
                            "dims[0]=3 but caller passed n_in=4");
}

/* ===========================================================================
 * rmsnorm
 * ===========================================================================
 */

TEST(rmsnorm_known_4vector) {
    f32 x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    f32 w[4] = {1.0f, 2.0f, 0.5f, -1.0f};
    f32 y[4];
    rmsnorm(x, w, 4, 1e-6f, y);

    /* mean_sq = 30/4 = 7.5; denom = sqrtf(7.500001) ~= 2.738613 (report §3) */
    ASSERT_CLOSE_F32(y[0], 0.36514837f, 1e-5f, 1e-6f);
    ASSERT_CLOSE_F32(y[1], 1.4605935f, 1e-5f, 1e-6f);
    ASSERT_CLOSE_F32(y[2], 0.5477225f, 1e-5f, 1e-6f);
    ASSERT_CLOSE_F32(y[3], -1.4605935f, 1e-5f, 1e-6f);
}

TEST(rmsnorm_in_place_aliasing_matches_out_of_place) {
    f32 x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    f32 w[4] = {1.0f, 2.0f, 0.5f, -1.0f};
    f32 y_out_of_place[4];
    rmsnorm(x, w, 4, 1e-6f, y_out_of_place);

    f32 buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    rmsnorm(buf, w, 4, 1e-6f, buf); /* y aliases x */

    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(buf[i] == y_out_of_place[i]);
    }
}

/* ===========================================================================
 * rope_neox
 * ===========================================================================
 */

#define ROPE_HEAD_DIM 128u
#define ROPE_FREQ_BASE 5000000.0f

static void fill_rope_vec(f32* v) {
    for (int i = 0; i < 128; i++) {
        v[i] = (f32)(i + 1); /* 1, 2, ..., 128 -- distinct at every index */
    }
}

TEST(rope_neox_pos0_identity) {
    f32 v[128];
    fill_rope_vec(v);
    f32 orig[128];
    memcpy(orig, v, sizeof v);

    rope_neox(v, ROPE_HEAD_DIM, 0, ROPE_FREQ_BASE);

    for (int i = 0; i < 128; i++) {
        ASSERT_TRUE(v[i] == orig[i]); /* exact: theta stays 0 for every k,
                                          cosf(0)=1/sinf(0)=0 exactly */
    }
}

TEST(rope_neox_pos1_known) {
    f32 v[128];
    fill_rope_vec(v);

    rope_neox(v, ROPE_HEAD_DIM, 1, ROPE_FREQ_BASE);

    /* goldens: report §4 (theta_scale=0.78582996f, theta_0=1.0 exact,
       theta_1=0.78582996f, theta_63=2.5450757e-07f) */
    ASSERT_CLOSE_F32(v[0],   -54.155315f, 1e-5f, 1e-4f);
    ASSERT_CLOSE_F32(v[64],   35.96112f,  1e-5f, 1e-4f);
    ASSERT_CLOSE_F32(v[1],   -45.275597f, 1e-5f, 1e-4f);
    ASSERT_CLOSE_F32(v[65],   48.063713f, 1e-5f, 1e-4f);
    ASSERT_CLOSE_F32(v[63],   63.999966f, 1e-5f, 1e-4f);
    ASSERT_CLOSE_F32(v[127], 128.00002f,  1e-5f, 1e-4f);
}

TEST(rope_neox_pos17_known) {
    f32 v[128];
    fill_rope_vec(v);

    rope_neox(v, ROPE_HEAD_DIM, 17, ROPE_FREQ_BASE);

    /* goldens: report §4 (theta_0=17.0 exact, theta_1=13.359109f,
       theta_63=4.3266296e-06f) */
    ASSERT_CLOSE_F32(v[0],    62.215675f, 1e-5f, 1e-4f);
    ASSERT_CLOSE_F32(v[64],  -18.847015f, 1e-5f, 1e-4f);
    ASSERT_CLOSE_F32(v[1],   -45.606556f, 1e-5f, 1e-4f);
    ASSERT_CLOSE_F32(v[65],   47.74979f,  1e-5f, 1e-4f);
    ASSERT_CLOSE_F32(v[63],   63.999447f, 1e-5f, 1e-4f);
    ASSERT_CLOSE_F32(v[127], 128.00027f,  1e-5f, 1e-4f);
}

/* ===========================================================================
 * softmax
 * ===========================================================================
 */

TEST(softmax_known_small) {
    f32 x[3] = {1.0f, 2.0f, 3.0f};
    softmax(x, 3);
    ASSERT_CLOSE_F32(x[0], 0.09003057f, 1e-5f, 1e-6f);
    ASSERT_CLOSE_F32(x[1], 0.24472846f, 1e-5f, 1e-6f);
    ASSERT_CLOSE_F32(x[2], 0.66524094f, 1e-5f, 1e-6f);
    ASSERT_CLOSE_F32(x[0] + x[1] + x[2], 1.0f, 1e-5f, 1e-6f);
}

TEST(softmax_large_magnitude_proves_max_subtract) {
    /* Without max-subtract, expf(1000.0f) alone already overflows f32
     * (e^1000 is astronomically past FLT_MAX) giving inf/inf = NaN. A
     * correct implementation gives EXACTLY the same result as the
     * shift-invariant small-number case above (softmax(x) ==
     * softmax(x+c) for any constant c). */
    f32 x[3] = {1000.0f, 1001.0f, 1002.0f};
    softmax(x, 3);
    ASSERT_TRUE(!isnan(x[0]) && !isnan(x[1]) && !isnan(x[2]));
    ASSERT_TRUE(!isinf(x[0]) && !isinf(x[1]) && !isinf(x[2]));
    ASSERT_CLOSE_F32(x[0], 0.09003057f, 1e-5f, 1e-6f);
    ASSERT_CLOSE_F32(x[1], 0.24472846f, 1e-5f, 1e-6f);
    ASSERT_CLOSE_F32(x[2], 0.66524094f, 1e-5f, 1e-6f);
}

TEST(softmax_n_zero_is_a_safe_no_op) {
    f32 x[1] = {42.0f}; /* untouched sentinel -- softmax(x, 0) must not
                            read or write index 0 */
    softmax(x, 0);
    ASSERT_TRUE(x[0] == 42.0f);
}

/* ===========================================================================
 * silu / swiglu
 * ===========================================================================
 */

TEST(silu_known) {
    ASSERT_TRUE(silu(0.0f) == 0.0f); /* 0/(1+e^0) = 0/2, exact */
    ASSERT_CLOSE_F32(silu(2.0f), 1.761594f, 1e-5f, 1e-6f);
    ASSERT_CLOSE_F32(silu(-2.0f), -0.23840587f, 1e-5f, 1e-6f);
    ASSERT_CLOSE_F32(silu(20.0f), 20.0f, 1e-5f, 1e-6f);
    ASSERT_CLOSE_F32(silu(-20.0f), -4.122307e-08f, 1e-5f, 1e-9f);
}

TEST(swiglu_known) {
    f32 g[4] = {1.0f, 2.0f, -1.0f, 0.0f};
    f32 u[4] = {2.0f, 0.5f, 3.0f, 10.0f};
    f32 h[4];
    swiglu(g, u, 4, h);
    ASSERT_CLOSE_F32(h[0], 1.4621172f, 1e-5f, 1e-6f);
    ASSERT_CLOSE_F32(h[1], 0.880797f, 1e-5f, 1e-6f);
    ASSERT_CLOSE_F32(h[2], -0.8068242f, 1e-5f, 1e-6f);
    ASSERT_TRUE(h[3] == 0.0f); /* silu(0)*10 = 0 exactly */
}

TEST(swiglu_aliased_output_matches_unaliased) {
    f32 g[4] = {1.0f, 2.0f, -1.0f, 0.0f};
    f32 u[4] = {2.0f, 0.5f, 3.0f, 10.0f};
    f32 h[4];
    swiglu(g, u, 4, h);

    f32 g2[4] = {1.0f, 2.0f, -1.0f, 0.0f};
    swiglu(g2, u, 4, g2); /* h aliases g */

    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(g2[i] == h[i]);
    }
}

/* ===========================================================================
 * argmax
 * ===========================================================================
 */

TEST(argmax_normal) {
    f32 logits[4] = {1.0f, 5.0f, 3.0f, -2.0f};
    ASSERT_EQ_U32(argmax(logits, 4), 1u);
}

TEST(argmax_tie_first_index_wins) {
    f32 logits[5] = {3.0f, 7.0f, 7.0f, 2.0f, 7.0f};
    ASSERT_EQ_U32(argmax(logits, 5), 1u);
}

TEST(argmax_single_element) {
    f32 logits[1] = {-3.5f};
    ASSERT_EQ_U32(argmax(logits, 1), 0u);
}

/* ===========================================================================
 * M3 integer core: f32_to_f16, quantize_row_q8_0, vec_dot_q8_0_q8_0
 * ===========================================================================
 */

/* EXHAUSTIVE, not sampled. f16 has only 65,536 bit patterns, so "widen then
 * narrow returns the original" can be PROVEN over the entire type rather than
 * spot-checked -- and it is the strongest available statement about
 * f32_to_f16 on the values this project actually stores, since every scale it
 * narrows was an f16 in a GGUF at some point.
 *
 * NaNs are excluded from the identity claim: their payload bits need not
 * survive a round trip, only their NaN-ness, asserted separately below. */
TEST(f16_roundtrip_is_identity_for_all_65536_bit_patterns) {
    u32 checked = 0, mismatches = 0, first_bad = 0;
    for (u32 h = 0; h < 65536u; h++) {
        u32 exp = (h >> 10) & 0x1Fu;
        u32 mant = h & 0x3FFu;
        if (exp == 0x1Fu && mant != 0u) {
            continue; /* NaN: checked separately */
        }
        f32 wide = f16_to_f32((u16)h);
        u16 back = f32_to_f16(wide);
        checked++;
        if (back != (u16)h) {
            if (mismatches == 0) {
                first_bad = h;
            }
            mismatches++;
        }
    }
    /* 65536 total - 2046 NaN patterns = 63490. NaN is exponent==0x1F with a
     * NONZERO mantissa, so it is 1023 mantissa values (1..1023, not 1024 --
     * mantissa 0 is infinity) times 2 signs. Deriving it here rather than
     * pasting a number is the point: the first version of this assertion
     * said 64512 from a miscount, and it failed on the COUNT while the
     * round-trip itself had zero mismatches. */
    ASSERT_EQ_U32(checked, 65536u - 2u * 1023u);
    if (mismatches != 0) {
        printf("  first mismatch at h=0x%04X\n", first_bad);
    }
    ASSERT_EQ_U32(mismatches, 0u);
}

TEST(f32_to_f16_nan_stays_nan_and_never_becomes_inf) {
    /* A payload living entirely in the low 13 bits truncates to zero, which
     * would turn a NaN into an INF -- the one silent corruption this
     * conversion can produce. The implementation forces such a payload to 1. */
    u32 bits = 0x7F800001u; /* NaN with the smallest possible payload */
    f32 nan_val;
    memcpy(&nan_val, &bits, sizeof nan_val);
    u16 h = f32_to_f16(nan_val);
    ASSERT_TRUE(((h >> 10) & 0x1Fu) == 0x1Fu); /* exponent all ones */
    ASSERT_TRUE((h & 0x3FFu) != 0u);           /* mantissa nonzero => NaN */
}

TEST(f32_to_f16_overflow_and_underflow_saturate) {
    ASSERT_EQ_U32((u32)f32_to_f16(1.0e30f), 0x7C00u);   /* +inf */
    ASSERT_EQ_U32((u32)f32_to_f16(-1.0e30f), 0xFC00u);  /* -inf */
    ASSERT_EQ_U32((u32)f32_to_f16(1.0e-30f), 0x0000u);  /* +0 */
    ASSERT_EQ_U32((u32)f32_to_f16(-1.0e-30f), 0x8000u); /* -0, sign kept */
    /* 65504 is the largest finite f16; 65520 is the tie that rounds UP to
     * inf under round-to-nearest-even -- correct IEEE behaviour, and the
     * boundary a naive clamp-to-max would get wrong. */
    ASSERT_EQ_U32((u32)f32_to_f16(65504.0f), 0x7BFFu);
    ASSERT_EQ_U32((u32)f32_to_f16(65520.0f), 0x7C00u);
}

TEST(f32_to_f16_ties_go_to_even_not_away_from_zero) {
    /* At f16 exponent 2^11 the representable step is 2. 2049 sits exactly
     * halfway between 2048 and 2050: round-half-even picks 2048 (even
     * mantissa), round-half-away would pick 2050. 2051 is halfway between
     * 2050 and 2052 and rounds to 2052, also even. */
    ASSERT_TRUE(f16_to_f32(f32_to_f16(2050.0f)) == 2050.0f);
    ASSERT_TRUE(f16_to_f32(f32_to_f16(2049.0f)) == 2048.0f);
    ASSERT_TRUE(f16_to_f32(f32_to_f16(2051.0f)) == 2052.0f);
}

TEST(f32_to_f16_subnormal_range_is_representable) {
    /* 2^-24 is the smallest positive f16 subnormal (bit pattern 0x0001), and
     * 2^-25 is exactly half of it -- a tie that rounds to even, i.e. to
     * ZERO. Both boundaries matter: the first proves the subnormal path
     * exists at all, the second proves it rounds rather than truncates. */
    ASSERT_EQ_U32((u32)f32_to_f16(5.9604645e-8f), 0x0001u);  /* 2^-24 */
    ASSERT_EQ_U32((u32)f32_to_f16(2.9802322e-8f), 0x0000u);  /* 2^-25, tie */
    ASSERT_EQ_U32((u32)f32_to_f16(6.1035156e-5f), 0x0400u);  /* smallest normal */
}

/* Quantize then dequantize must reproduce the block's largest-magnitude
 * element closely: that element maps to +-127, the extreme the scale is
 * defined by. */
TEST(quantize_row_q8_0_preserves_the_block_absmax) {
    f32 x[32];
    for (int i = 0; i < 32; i++) {
        x[i] = (f32)(i - 16) * 0.25f;
    }
    x[7] = -8.0f; /* the absmax */
    u8 blk[KERNELS_Q8_0_BLOCK_BYTES];
    quantize_row_q8_0(x, 32, blk);

    f32 out[32];
    dequant_row_q8_0(blk, 32, out);
    ASSERT_TRUE(fabsf(out[7] - (-8.0f)) < 0.05f);
    /* its quant is exactly -127, which is the definition of the scale */
    ASSERT_TRUE((i8)blk[2 + 7] == -127);
}

TEST(quantize_row_q8_0_all_zero_block_is_zero_not_nan) {
    /* The `d ? 1/d : 0` guard. Without it every quant is 0/0 = NaN, and an
     * all-zero activation block is entirely ordinary in a real forward pass. */
    f32 x[32];
    for (int i = 0; i < 32; i++) {
        x[i] = 0.0f;
    }
    u8 blk[KERNELS_Q8_0_BLOCK_BYTES];
    quantize_row_q8_0(x, 32, blk);
    ASSERT_EQ_U32((u32)blk[0], 0u);
    ASSERT_EQ_U32((u32)blk[1], 0u);
    for (int i = 0; i < 32; i++) {
        ASSERT_TRUE((i8)blk[2 + i] == 0);
    }
    f32 out[32];
    dequant_row_q8_0(blk, 32, out);
    for (int i = 0; i < 32; i++) {
        ASSERT_TRUE(out[i] == 0.0f);
    }
}

TEST(quantize_row_q8_0_rounds_half_away_from_zero_not_to_even) {
    /* ggml uses roundf, so a value landing exactly on .5 goes AWAY from zero.
     * An absmax of 127 fixes d = 1.0 exactly, making the intended quants
     * readable directly. This deliberately DIFFERS from f32_to_f16's tie
     * rule -- two separate contracts, one mirroring ggml's quantizer and one
     * mirroring F16C hardware. */
    f32 x[32];
    for (int i = 0; i < 32; i++) {
        x[i] = 0.0f;
    }
    x[0] = 127.0f; /* fixes d = 1.0 */
    x[1] = 0.5f;
    x[2] = 1.5f;
    x[3] = -0.5f;
    u8 blk[KERNELS_Q8_0_BLOCK_BYTES];
    quantize_row_q8_0(x, 32, blk);
    ASSERT_TRUE((i8)blk[2 + 0] == 127);
    ASSERT_TRUE((i8)blk[2 + 1] == 1);  /* half AWAY from zero, not to even */
    ASSERT_TRUE((i8)blk[2 + 2] == 2);
    ASSERT_TRUE((i8)blk[2 + 3] == -1); /* and away on the negative side */
}

/* The integer core against the f32 kernel it replaces. They are NOT expected
 * to agree bit for bit -- quantizing the activation loses information, which
 * is the entire point -- but on well-conditioned data they must agree
 * closely. A gross disagreement is a layout or sign error, not quant noise. */
TEST(vec_dot_q8_0_q8_0_tracks_the_f32_kernel_within_quant_error) {
    enum { N = 256 };
    f32 w[N], a[N];
    for (int i = 0; i < N; i++) {
        w[i] = sinf((f32)i * 0.37f) * 2.0f;
        a[i] = cosf((f32)i * 0.11f) * 1.5f;
    }
    u8 wq[(N / 32) * KERNELS_Q8_0_BLOCK_BYTES];
    u8 aq[(N / 32) * KERNELS_Q8_0_BLOCK_BYTES];
    quantize_row_q8_0(w, N, wq);
    quantize_row_q8_0(a, N, aq);

    f32 ref = vec_dot_q8_0_f32(wq, a, N); /* quantized W, dense activation */
    f32 got = vec_dot_q8_0_q8_0(wq, aq, N);
    f32 denom = fabsf(ref) > 1.0f ? fabsf(ref) : 1.0f;
    ASSERT_TRUE(fabsf(got - ref) / denom < 0.02f);
}

TEST(vec_dot_q8_0_q8_0_is_exact_on_a_hand_computable_case) {
    /* Both sides quantize to a scale of exactly 1.0 (absmax 127), so every
     * quant is its own integer and the dot is computable by hand:
     * 127*127 + 1*2 = 16131, scaled by 1.0*1.0. Any layout, sign, or
     * scale-ordering error moves this. */
    f32 w[32], a[32];
    for (int i = 0; i < 32; i++) {
        w[i] = 0.0f;
        a[i] = 0.0f;
    }
    w[0] = 127.0f; a[0] = 127.0f;
    w[1] = 1.0f;   a[1] = 2.0f;
    u8 wq[KERNELS_Q8_0_BLOCK_BYTES], aq[KERNELS_Q8_0_BLOCK_BYTES];
    quantize_row_q8_0(w, 32, wq);
    quantize_row_q8_0(a, 32, aq);
    ASSERT_TRUE(fabsf(vec_dot_q8_0_q8_0(wq, aq, 32) - 16131.0f) < 0.5f);
}

TEST(vec_dot_q8_0_q8_0_negative_products_accumulate_with_sign) {
    /* An unsigned-vs-signed byte confusion -- the exact bug dpbusd's sign
     * trick exists to avoid at K3 -- shows up here as a wrong sign or a wild
     * magnitude: -127*127 = -16129, not +16129 and not 129*127. */
    f32 w[32], a[32];
    for (int i = 0; i < 32; i++) {
        w[i] = 0.0f;
        a[i] = 0.0f;
    }
    w[0] = -127.0f; a[0] = 127.0f;
    u8 wq[KERNELS_Q8_0_BLOCK_BYTES], aq[KERNELS_Q8_0_BLOCK_BYTES];
    quantize_row_q8_0(w, 32, wq);
    quantize_row_q8_0(a, 32, aq);
    ASSERT_TRUE(fabsf(vec_dot_q8_0_q8_0(wq, aq, 32) - (-16129.0f)) < 0.5f);
}

/* The i32 accumulator cannot overflow BY CONSTRUCTION, and this pins the
 * bound the kernel's comment claims: 32 elements at the extreme +-127 on
 * both sides is 32*127*127 = 516,128, three orders of magnitude inside i32.
 * A future widening of the block size would trip this before it trips a
 * silent wraparound in production. */
TEST(vec_dot_q8_0_q8_0_saturated_block_stays_far_inside_i32) {
    f32 w[32], a[32];
    for (int i = 0; i < 32; i++) {
        w[i] = 127.0f;
        a[i] = 127.0f;
    }
    u8 wq[KERNELS_Q8_0_BLOCK_BYTES], aq[KERNELS_Q8_0_BLOCK_BYTES];
    quantize_row_q8_0(w, 32, wq);
    quantize_row_q8_0(a, 32, aq);
    f32 got = vec_dot_q8_0_q8_0(wq, aq, 32);
    ASSERT_TRUE(fabsf(got - 516128.0f) < 2.0f);
    ASSERT_TRUE(516128.0 < 2147483647.0 / 1000.0);
}

/* ▶ THE CROSS-IMPLEMENTATION CHECK, and the reason quant work in this project
 * is python-first (M3 forward note, K4: "python dequant == C scalar dequant,
 * byte-exact, on real rows, BEFORE the SIMD twin exists").
 *
 * The expected bytes below were produced by an INDEPENDENT python
 * transcription of ggml's quantize_row_q8_0_ref, written from the probed
 * source rather than from q8_0.c. That python reference was itself validated
 * against llama.cpp's own output: dequantizing a real Q8_0 row from the
 * Qwen3 GGUF and re-quantizing it reproduced the original bytes EXACTLY on
 * 12 of 12 rows across blk.0.attn_q, blk.0.ffn_down, blk.17.attn_v and
 * blk.35.ffn_up. So this golden traces back to llama.cpp's quantizer, not to
 * our own reasoning about it.
 *
 * The input is (i-16)/8.0, every value exactly representable in binary, so
 * C and python agree bit-for-bit on the input itself and any disagreement in
 * the output is genuinely the quantizer.
 *
 * absmax is 2.0 (from i=0), so d = 2/127 and the reciprocal is exactly 63.5 --
 * which puts x=+-1.0 precisely on +-63.5, a rounding boundary. */
TEST(quantize_row_q8_0_matches_the_python_reference_byte_for_byte) {
    f32 x[32];
    for (int i = 0; i < 32; i++) {
        x[i] = (f32)(i - 16) / 8.0f;
    }
    u8 blk[KERNELS_Q8_0_BLOCK_BYTES];
    quantize_row_q8_0(x, 32, blk);

    /* f16 scale, little-endian: 0x2408 == 2.0/127 narrowed */
    ASSERT_EQ_U32((u32)blk[0], 0x08u);
    ASSERT_EQ_U32((u32)blk[1], 0x24u);

    static const i8 want[32] = {
        -127, -119, -111, -103, -95, -87, -79, -71,
        -64,  -56,  -48,  -40,  -32, -24, -16,  -8,
          0,    8,   16,   24,    32,  40,  48,  56,
         64,   71,   79,   87,    95, 103, 111, 119
    };
    int bad = -1;
    for (int i = 0; i < 32; i++) {
        if ((i8)blk[2 + i] != want[i]) {
            bad = i;
            break;
        }
    }
    if (bad >= 0) {
        printf("  first mismatch at quant %d: got %d want %d\n", bad,
               (int)(i8)blk[2 + bad], (int)want[bad]);
    }
    ASSERT_TRUE(bad < 0);
}

/* ▶ THE GAP THE ROUND-TRIP TEST LEFT OPEN, and it is not a small one.
 *
 * f16_roundtrip_is_identity_for_all_65536_bit_patterns proves widen-then-
 * narrow is identity -- but every input it feeds f32_to_f16 came FROM an f16,
 * so it is exactly representable and no rounding ever happens. It therefore
 * exercises none of the rounding logic, which is the only part that can be
 * wrong. An exhaustive test can still be a shallow one.
 *
 * These are arbitrary f32 values that must genuinely round, with expectations
 * from numpy (np.float16), which is IEEE round-to-nearest-even. The first
 * entry is the case that actually broke: a Q8_0 block scale of 0.48/127, the
 * value the bench's activation pool produces. */
TEST(f32_to_f16_narrows_arbitrary_values_matching_numpy) {
    static const struct { f32 in; u16 want; const char* note; } cases[] = {
        { 0.48f / 127.0f, 0x1BBEu, "a real Q8_0 activation scale" },
        { 0.125976562f,   0x3008u, "the p00 activation scale" },
        { 1.0f,           0x3C00u, "exact" },
        { 0.5f,           0x3800u, "exact" },
        { 2.0f,           0x4000u, "exact" },
        { 0.75f,          0x3A00u, "exact" },
        { 0.01f,          0x211Fu, "rounds" },
        { 0.001f,         0x1419u, "rounds, small" },
        { 3.14159f,       0x4248u, "rounds" },
    };
    int bad = 0;
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        u16 got = f32_to_f16(cases[i].in);
        if (got != cases[i].want) {
            printf("  f32_to_f16(%.9g) = 0x%04X, want 0x%04X  (%s)\n",
                   (double)cases[i].in, got, cases[i].want, cases[i].note);
            bad++;
        }
    }
    ASSERT_EQ_U32((u32)bad, 0u);
}
