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
        matvec(&w, x, 4, y, 1);
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
        matvec(&w, x, 4, y, 2);
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
    matvec(&w, x, 3, y, 2);

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
    w.type    = KERNELS_GGML_TYPE_Q8_0;
    w.n_dims  = 2;
    w.dims[0] = 32;
    w.dims[1] = 2;
    w.data    = blocks;
    w.nbytes  = sizeof blocks;

    f32 x[32];
    fill_x32(x);
    f32 y[2];
    matvec(&w, x, 32, y, 2);

    ASSERT_TRUE(y[0] == -92.625f); /* same block+x as the standalone Q8_0 test */
    ASSERT_TRUE(y[1] == 1056.0f);  /* d=2, qs=2 everywhere: 4*sum(x) = 4*264 */
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
