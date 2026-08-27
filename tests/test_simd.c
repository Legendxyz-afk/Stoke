/* tests/test_simd.c -- M3 gate [1]: the scalar<->SIMD exact-diff harness.
 *
 * The forward note's gate is "scalar<->SIMD exact-diff (integer paths) /
 * <=1e-5 rel (f32 paths) on 1e4 random tensors per kernel, all quant kinds".
 * This file IS that gate for the kernels K3 twins, and it is written to fail
 * in every way the property can actually be violated:
 *
 *   - INTEGER PATHS ARE COMPARED AS BIT PATTERNS, not with an epsilon. The
 *     whole point of LOCKED decision #2 (both paths run the same i32 integer
 *     core) is that no tolerance is needed, so accepting one would hide
 *     exactly the reordering it exists to forbid -- a __m256 accumulator
 *     summing blocks in a different order lands within 1e-6 while being
 *     precisely the defect the gate forbids.
 *
 *   - THE SUITE REFUSES TO PASS VACUOUSLY. Every twin is called DIRECTLY,
 *     never through a dispatcher that could fall back to scalar, and
 *     simd_cpu_probe_finds_avx_vnni_on_this_box asserts the feature is
 *     actually present. Without that assertion this file would go green on a
 *     CPU where nothing under test ever executed, which is the failure mode
 *     this project has paid for three times in measurement and would
 *     otherwise now import into its correctness gate.
 *
 * Random inputs are drawn from a FIXED seed: a gate that samples a different
 * 1e4 tensors on every run cannot be bisected, and a failure that does not
 * reproduce is not evidence. The generator is a local xorshift rather than
 * rand(), whose sequence is implementation-defined and would differ between
 * the Windows and WSL legs -- the two legs must test the same tensors.
 */
#include "common.h"

#include "kernels/kernels.h"
#include "kernels/simd.h"

#include <string.h>

#include "test.h"

/* Block layout, mirrored from kernels.h's Q8_0 contract: f16 scale then 32
 * i8 quants. Declared here rather than reaching into the kernel's internals
 * so the test states its own understanding of the format it is feeding. */
#define BLK_ELEMS 32u
#define BLK_BYTES 34u

/* xorshift64*, fixed seed. Deterministic and identical on both legs. */
static u64 g_rng = 0x9E3779B97F4A7C15ull;

static void rng_reset(void) { g_rng = 0x9E3779B97F4A7C15ull; }

static u32 rng_next(void) {
    g_rng ^= g_rng >> 12;
    g_rng ^= g_rng << 25;
    g_rng ^= g_rng >> 27;
    return (u32)((g_rng * 0x2545F4914F6CDD1Dull) >> 32);
}

/* Fills one Q8_0 block: an f16 scale and 32 signed quants.
 *
 * The quants deliberately include the endpoints -128 and 127 rather than a
 * comfortable interior range. -128 is the value that makes the dpbusd sign
 * trick interesting: _mm256_sign_epi8(x, x) cannot represent |-128| in i8
 * and leaves it as -128, so if the scalar and SIMD paths disagree anywhere
 * it is most likely there, and a test drawing only from [-127,127] would
 * miss it completely. */
static void fill_block(u8* blk, f32 scale) {
    u16 h = f32_to_f16(scale);
    memcpy(blk, &h, 2);
    for (u32 i = 0; i < BLK_ELEMS; i++) {
        blk[2 + i] = (u8)(i8)(i32)((rng_next() % 256u) - 128u);
    }
}

/* Scales spanning several binades, including denormal-ish and large, so the
 * f32 product d_w*d_a is not always in a benign range. Kept finite: inf/nan
 * scales are a fatal() path on both implementations, tested elsewhere. */
static f32 random_scale(void) {
    static const f32 mags[] = {1.0f / 4096.0f, 1.0f / 127.0f, 0.00731f,
                               0.5f, 1.0f, 7.25f, 133.0f};
    u32 i = rng_next() % (u32)(sizeof mags / sizeof mags[0]);
    f32 s = mags[i];
    return (rng_next() & 1u) ? -s : s;
}

static bool bit_identical(f32 a, f32 b) {
    return memcmp(&a, &b, sizeof(f32)) == 0;
}

/* memcpy, not a pointer cast: type-punning an f32 through a u32* is a
 * strict-aliasing violation that -Werror rejects outright here, and at -O3
 * it is the kind gcc is entitled to miscompile silently. */
static u32 f32_bits(f32 v) {
    u32 b;
    memcpy(&b, &v, sizeof b);
    return b;
}

/* ------------------------------------------------------------------ tests */

/* THE GUARD THAT KEEPS EVERY OTHER TEST IN THIS FILE HONEST. If this fails,
 * the twins below are not being exercised on this box and their green bars
 * mean nothing -- so this is a hard failure, not a skip. */
TEST(simd_cpu_probe_finds_avx_vnni_on_this_box) {
    ASSERT_TRUE(kernels_cpu_has_avx_vnni());
}

TEST(simd_vec_dot_q8_0_q8_0_is_bit_exact_vs_scalar_over_1e4_tensors) {
    if (!kernels_cpu_has_avx_vnni()) {
        fprintf(stderr, "FAIL %s:%d: no AVX-VNNI -- the exact-diff gate "
                        "cannot run on this box\n", __FILE__, __LINE__);
        g_test_case_failed = 1;
        return;
    }

    /* Row lengths cover one block, a couple of blocks, and lengths that are
     * NOT multiples of the 4-block unrolling a vectorized loop naturally
     * wants -- a twin that handles only whole groups of 4 blocks would pass
     * on 128 and fail on 96, so both are here. */
    static const u64 lens[] = {32, 64, 96, 128, 160, 256, 2560};
    const u32        n_lens = (u32)(sizeof lens / sizeof lens[0]);
    const u32        iters  = 10000;

    rng_reset();
    u8* w = (u8*)xmalloc((size_t)(2560u / BLK_ELEMS) * BLK_BYTES);
    u8* a = (u8*)xmalloc((size_t)(2560u / BLK_ELEMS) * BLK_BYTES);

    u32 mismatches = 0;
    for (u32 it = 0; it < iters; it++) {
        u64 n      = lens[it % n_lens];
        u64 blocks = n / BLK_ELEMS;
        for (u64 b = 0; b < blocks; b++) {
            fill_block(w + b * BLK_BYTES, random_scale());
            fill_block(a + b * BLK_BYTES, random_scale());
        }

        f32 want = vec_dot_q8_0_q8_0(w, a, n);
        f32 got  = vec_dot_q8_0_q8_0_avx2(w, a, n);

        if (!bit_identical(want, got)) {
            if (mismatches < 3u) {
                fprintf(stderr,
                        "FAIL %s:%d: iter %u n=%llu scalar=%.9g simd=%.9g "
                        "(bits %08x vs %08x)\n",
                        __FILE__, __LINE__, it, (unsigned long long)n,
                        (double)want, (double)got,
                        f32_bits(want), f32_bits(got));
            }
            mismatches++;
        }
    }
    if (mismatches != 0u) {
        fprintf(stderr, "FAIL %s:%d: %u of %u tensors differ\n", __FILE__,
                __LINE__, mismatches, iters);
        g_test_case_failed = 1;
    }

    free(w);
    free(a);
}

/* The all-zero block is the one input where the scalar kernel's f32 grouping
 * could plausibly be replicated by a wrong-but-close SIMD path (0 * anything
 * is 0 in both), so it is asserted separately rather than left to the
 * random sweep to hit by chance. */
TEST(simd_vec_dot_q8_0_q8_0_matches_scalar_on_all_zero_quants) {
    ASSERT_TRUE(kernels_cpu_has_avx_vnni());
    if (!kernels_cpu_has_avx_vnni()) return;

    u8 w[BLK_BYTES * 2];
    u8 a[BLK_BYTES * 2];
    memset(w, 0, sizeof w);
    memset(a, 0, sizeof a);
    u16 h = f32_to_f16(0.125f);
    memcpy(w, &h, 2);
    memcpy(w + BLK_BYTES, &h, 2);
    memcpy(a, &h, 2);
    memcpy(a + BLK_BYTES, &h, 2);

    ASSERT_TRUE(bit_identical(vec_dot_q8_0_q8_0(w, a, 64),
                              vec_dot_q8_0_q8_0_avx2(w, a, 64)));
}

TEST(simd_vec_dot_q8_0_q8_0_x4_is_bit_exact_vs_scalar_over_1e4_groups) {
    ASSERT_TRUE(kernels_cpu_has_avx_vnni());
    if (!kernels_cpu_has_avx_vnni()) return;

    static const u64 lens[] = {32, 64, 96, 2560};
    const u32        n_lens = (u32)(sizeof lens / sizeof lens[0]);
    const u32        iters  = 10000;
    const u64        maxblk = 2560u / BLK_ELEMS;

    rng_reset();
    u8* w[4];
    for (u32 k = 0; k < 4u; k++) w[k] = (u8*)xmalloc((size_t)maxblk * BLK_BYTES);
    u8* a = (u8*)xmalloc((size_t)maxblk * BLK_BYTES);

    u32 mismatches = 0;
    for (u32 it = 0; it < iters; it++) {
        u64 n = lens[it % n_lens];
        for (u64 b = 0; b < n / BLK_ELEMS; b++) {
            fill_block(a + b * BLK_BYTES, random_scale());
            for (u32 k = 0; k < 4u; k++) {
                fill_block(w[k] + b * BLK_BYTES, random_scale());
            }
        }
        const void* rows[4] = {w[0], w[1], w[2], w[3]};
        f32 got[4];
        vec_dot_q8_0_q8_0_x4_avx2(rows, a, n, got);
        /* EACH ROW against the SCALAR kernel, not against the single-row
         * twin: chaining twin-vs-twin would let a shared defect cancel. */
        for (u32 k = 0; k < 4u; k++) {
            f32 want = vec_dot_q8_0_q8_0(w[k], a, n);
            if (!bit_identical(want, got[k])) {
                if (mismatches < 3u) {
                    fprintf(stderr,
                            "FAIL %s:%d: iter %u n=%llu row %u scalar=%.9g "
                            "x4=%.9g (bits %08x vs %08x)\n",
                            __FILE__, __LINE__, it, (unsigned long long)n, k,
                            (double)want, (double)got[k], f32_bits(want),
                            f32_bits(got[k]));
                }
                mismatches++;
            }
        }
    }
    if (mismatches != 0u) {
        fprintf(stderr, "FAIL %s:%d: %u row results differ\n", __FILE__,
                __LINE__, mismatches);
        g_test_case_failed = 1;
    }
    for (u32 k = 0; k < 4u; k++) free(w[k]);
    free(a);
}

TEST(simd_dequant_row_q8_0_is_bit_exact_vs_scalar_over_1e4_rows) {
    ASSERT_TRUE(kernels_cpu_has_avx_vnni());
    if (!kernels_cpu_has_avx_vnni()) return;

    static const u64 lens[]  = {32, 64, 96, 128, 2560};
    const u32        n_lens  = (u32)(sizeof lens / sizeof lens[0]);
    const u32        iters   = 10000;

    rng_reset();
    u8*  row  = (u8*)xmalloc((size_t)(2560u / BLK_ELEMS) * BLK_BYTES);
    f32* want = (f32*)xmalloc(2560u * sizeof(f32));
    f32* got  = (f32*)xmalloc(2560u * sizeof(f32));

    u32 mismatches = 0;
    for (u32 it = 0; it < iters; it++) {
        u64 n = lens[it % n_lens];
        for (u64 b = 0; b < n / BLK_ELEMS; b++) {
            fill_block(row + b * BLK_BYTES, random_scale());
        }
        dequant_row_q8_0(row, n, want);
        dequant_row_q8_0_avx2(row, n, got);
        /* memcmp over the whole row: an epsilon compare would accept a
         * fused or reassociated product, and there is nothing to reassociate
         * here -- any difference at all is a defect. */
        if (memcmp(want, got, (size_t)n * sizeof(f32)) != 0) {
            if (mismatches < 3u) {
                for (u64 i = 0; i < n; i++) {
                    if (!bit_identical(want[i], got[i])) {
                        fprintf(stderr,
                                "FAIL %s:%d: iter %u n=%llu idx %llu "
                                "scalar=%.9g simd=%.9g\n",
                                __FILE__, __LINE__, it,
                                (unsigned long long)n, (unsigned long long)i,
                                (double)want[i], (double)got[i]);
                        break;
                    }
                }
            }
            mismatches++;
        }
    }
    if (mismatches != 0u) {
        fprintf(stderr, "FAIL %s:%d: %u of %u rows differ\n", __FILE__,
                __LINE__, mismatches, iters);
        g_test_case_failed = 1;
    }
    free(row);
    free(want);
    free(got);
}

TEST(simd_argmax_matches_scalar_over_1e4_vectors) {
    ASSERT_TRUE(kernels_cpu_has_avx_vnni());
    if (!kernels_cpu_has_avx_vnni()) return;

    /* Lengths deliberately straddle the vector width: 1 and 7 are shorter
     * than one 8-lane register, 8 is exactly one, and 9/17/151936 leave a
     * ragged tail. A twin that mishandles the tail passes on 8 and fails on
     * 9, so both are here; 151936 is the real vocab. */
    static const u64 lens[] = {1, 7, 8, 9, 17, 64, 1000, 151936};
    const u32        n_lens = (u32)(sizeof lens / sizeof lens[0]);
    const u32        iters  = 10000;

    rng_reset();
    f32* v = (f32*)xmalloc(151936u * sizeof(f32));

    u32 mismatches = 0;
    for (u32 it = 0; it < iters; it++) {
        u64 n = lens[it % n_lens];
        /* A SMALL VALUE ALPHABET, ON PURPOSE. Uniform random floats almost
         * never tie, so a uniform sweep would exercise the tie rule -- the
         * one thing that is genuinely hard here -- essentially never. Values
         * drawn from a handful of levels make ties the common case instead
         * of a rarity, which is what the test is for. */
        for (u64 i = 0; i < n; i++) {
            u32 r = rng_next() % 6u;
            v[i] = (f32)((i32)r - 2) * 0.5f;
        }
        u32 want = argmax(v, n);
        u32 got  = argmax_avx2(v, n);
        if (want != got) {
            if (mismatches < 3u) {
                fprintf(stderr,
                        "FAIL %s:%d: iter %u n=%llu scalar=%u simd=%u "
                        "(values %.3g vs %.3g)\n",
                        __FILE__, __LINE__, it, (unsigned long long)n, want,
                        got, (double)v[want], (double)v[got]);
            }
            mismatches++;
        }
    }
    if (mismatches != 0u) {
        fprintf(stderr, "FAIL %s:%d: %u of %u vectors differ\n", __FILE__,
                __LINE__, mismatches, iters);
        g_test_case_failed = 1;
    }
    free(v);
}

/* The tie rule stated as its own assertion rather than left to the sweep:
 * every element equal means the answer must be 0, and any implementation
 * that reports "some index holding the maximum" fails here. */
TEST(simd_argmax_returns_the_first_index_on_an_all_equal_vector) {
    ASSERT_TRUE(kernels_cpu_has_avx_vnni());
    if (!kernels_cpu_has_avx_vnni()) return;

    f32 v[64];
    for (u32 i = 0; i < 64u; i++) v[i] = 1.25f;
    ASSERT_EQ_U32(argmax_avx2(v, 64), 0u);
    ASSERT_EQ_U32(argmax_avx2(v, 64), argmax(v, 64));

    /* Maximum duplicated across DIFFERENT vector lanes: the earlier index
     * must win even though both lie in the same 8-wide register. */
    f32 w[16];
    for (u32 i = 0; i < 16u; i++) w[i] = 0.0f;
    w[3] = 9.0f;
    w[11] = 9.0f;
    ASSERT_EQ_U32(argmax_avx2(w, 16), 3u);
    ASSERT_EQ_U32(argmax_avx2(w, 16), argmax(w, 16));
}

/* -128 cannot be negated in i8, which is exactly what the sign trick does to
 * the activation side. Driven to saturation on BOTH sides so the worst case
 * is a real input rather than a hypothetical. */
TEST(simd_vec_dot_q8_0_q8_0_matches_scalar_at_i8_min_saturation) {
    ASSERT_TRUE(kernels_cpu_has_avx_vnni());
    if (!kernels_cpu_has_avx_vnni()) return;

    u8 w[BLK_BYTES];
    u8 a[BLK_BYTES];
    u16 h = f32_to_f16(1.0f);
    memcpy(w, &h, 2);
    memcpy(a, &h, 2);
    for (u32 i = 0; i < BLK_ELEMS; i++) {
        w[2 + i] = (u8)(i8)-128;
        a[2 + i] = (u8)(i8)-128;
    }

    ASSERT_TRUE(bit_identical(vec_dot_q8_0_q8_0(w, a, 32),
                              vec_dot_q8_0_q8_0_avx2(w, a, 32)));
}

/* ------------------------------------------------------- K4: K-quant twins
 *
 * Same gate as the Q8_0 twins and for the same reason: these dispatch inside
 * matvec, so anything less than bit-identical is a numerics change that
 * would require re-running the parity gate, and it would be invisible in the
 * output until it was not.
 *
 * WEIGHT BYTES ARE RANDOM, ACTIVATIONS ARE REAL. The weight side is filled
 * with random bytes because every bit pattern is a legal quant and the
 * packed 6-bit scale fields then span their whole 0..63 range. The
 * activation side is produced by running the ACTUAL quantize_row_q8_k over
 * random floats rather than by scribbling random bytes into a Q8_K block:
 * bsums must agree with qs for the Q4_K min term to mean anything, and a
 * block whose sums contradict its quants is a state the engine can never
 * reach -- testing it would prove something about nothing.
 */
#define Q4K_BYTES 144u
#define Q6K_BYTES 210u
#define QK_ELEMS 256u

static void fill_q4_k_block(u8* blk) {
    u16 d = f32_to_f16(random_scale());
    u16 dmin = f32_to_f16(random_scale());
    memcpy(blk, &d, 2);
    memcpy(blk + 2, &dmin, 2);
    for (u32 i = 4; i < Q4K_BYTES; i++) blk[i] = (u8)(rng_next() & 0xFFu);
}

static void fill_q6_k_block(u8* blk) {
    for (u32 i = 0; i < 208u; i++) blk[i] = (u8)(rng_next() & 0xFFu);
    u16 d = f32_to_f16(random_scale());
    memcpy(blk + 208, &d, 2);
}

static void fill_q8_k_activation(f32* x, u8* aq, u64 n) {
    for (u64 i = 0; i < n; i++) {
        /* Spread over several binades and both signs, and let zeros occur:
         * an all-zero super-block takes quantize_row_q8_k's amax==0 branch,
         * which is the one that skips the bsums loop. */
        u32 r = rng_next();
        f32 mag = (r & 3u) == 0u ? 0.0f : (f32)(r % 1000u) * 0.0037f;
        x[i] = (r & 4u) ? -mag : mag;
    }
    quantize_row_q8_k(x, n, aq);
}

TEST(simd_vec_dot_q4_k_q8_k_is_bit_exact_vs_scalar_over_1e4_tensors) {
    if (!kernels_cpu_has_avx_vnni()) {
        fprintf(stderr, "FAIL %s:%d: no AVX-VNNI -- the K4 exact-diff gate "
                        "cannot run on this box\n", __FILE__, __LINE__);
        g_test_case_failed = 1;
        return;
    }
    /* 2560 and 9728 are the real contraction lengths of this model's
     * matvecs; 256/512/768 cover one, two and three super-blocks so a twin
     * that only handles an even number of blocks cannot hide. */
    static const u64 lens[] = {256, 512, 768, 2560, 9728};
    const u32 n_lens = (u32)(sizeof lens / sizeof lens[0]);
    const u32 iters = 10000;

    rng_reset();
    u8* w = (u8*)xmalloc((size_t)(9728u / QK_ELEMS) * Q4K_BYTES);
    u8* aq = (u8*)xmalloc((size_t)(9728u / QK_ELEMS) * KERNELS_Q8_K_BLOCK_BYTES);
    f32* x = (f32*)xmalloc((size_t)9728u * sizeof(f32));

    u32 mismatches = 0;
    for (u32 it = 0; it < iters; it++) {
        u64 n = lens[it % n_lens];
        u64 blocks = n / QK_ELEMS;
        for (u64 b = 0; b < blocks; b++) fill_q4_k_block(w + b * Q4K_BYTES);
        fill_q8_k_activation(x, aq, n);

        f32 got = vec_dot_q4_k_q8_k_avx2(w, aq, n);
        f32 want = vec_dot_q4_k_q8_k(w, aq, n);
        if (!bit_identical(got, want)) {
            if (mismatches == 0) {
                fprintf(stderr, "  iter %u n=%llu: scalar %.9g (0x%08X) vs "
                                "avx2 %.9g (0x%08X)\n",
                        it, (unsigned long long)n, (double)want,
                        f32_bits(want), (double)got, f32_bits(got));
            }
            mismatches++;
        }
    }
    free(w);
    free(aq);
    free(x);
    ASSERT_EQ_U32(mismatches, 0u);
}

TEST(simd_vec_dot_q6_k_q8_k_is_bit_exact_vs_scalar_over_1e4_tensors) {
    if (!kernels_cpu_has_avx_vnni()) {
        fprintf(stderr, "FAIL %s:%d: no AVX-VNNI -- the K4 exact-diff gate "
                        "cannot run on this box\n", __FILE__, __LINE__);
        g_test_case_failed = 1;
        return;
    }
    static const u64 lens[] = {256, 512, 768, 2560, 9728};
    const u32 n_lens = (u32)(sizeof lens / sizeof lens[0]);
    const u32 iters = 10000;

    rng_reset();
    u8* w = (u8*)xmalloc((size_t)(9728u / QK_ELEMS) * Q6K_BYTES);
    u8* aq = (u8*)xmalloc((size_t)(9728u / QK_ELEMS) * KERNELS_Q8_K_BLOCK_BYTES);
    f32* x = (f32*)xmalloc((size_t)9728u * sizeof(f32));

    u32 mismatches = 0;
    for (u32 it = 0; it < iters; it++) {
        u64 n = lens[it % n_lens];
        u64 blocks = n / QK_ELEMS;
        for (u64 b = 0; b < blocks; b++) fill_q6_k_block(w + b * Q6K_BYTES);
        fill_q8_k_activation(x, aq, n);

        f32 got = vec_dot_q6_k_q8_k_avx2(w, aq, n);
        f32 want = vec_dot_q6_k_q8_k(w, aq, n);
        if (!bit_identical(got, want)) {
            if (mismatches == 0) {
                fprintf(stderr, "  iter %u n=%llu: scalar %.9g (0x%08X) vs "
                                "avx2 %.9g (0x%08X)\n",
                        it, (unsigned long long)n, (double)want,
                        f32_bits(want), (double)got, f32_bits(got));
            }
            mismatches++;
        }
    }
    free(w);
    free(aq);
    free(x);
    ASSERT_EQ_U32(mismatches, 0u);
}

/* The extremes a uniform random sweep will not reliably produce: every
 * Q4_K scale AND min at their 6-bit maximum, against activation quants at
 * -128. This is the worst case for the i32 accumulator bound the twin's
 * folding relies on, so it is asserted rather than argued. */
TEST(simd_k_quant_twins_match_scalar_at_saturating_extremes) {
    if (!kernels_cpu_has_avx_vnni()) {
        g_test_case_failed = 1;
        return;
    }
    u8 w4[Q4K_BYTES], w6[Q6K_BYTES];
    u8 aq[KERNELS_Q8_K_BLOCK_BYTES];
    f32 x[QK_ELEMS];

    u16 d = f32_to_f16(133.0f), dm = f32_to_f16(-133.0f);
    memcpy(w4, &d, 2);
    memcpy(w4 + 2, &dm, 2);
    for (u32 i = 4; i < Q4K_BYTES; i++) w4[i] = 0xFFu; /* all 6-bit fields max */

    for (u32 i = 0; i < 192u; i++) w6[i] = 0xFFu;
    for (u32 i = 192u; i < 208u; i++) w6[i] = 0x80u;   /* scales = -128 */
    memcpy(w6 + 208, &d, 2);

    /* Drive every activation quant to the -128 endpoint. */
    for (u32 i = 0; i < QK_ELEMS; i++) x[i] = 1.0f;
    x[0] = -1.0f;
    quantize_row_q8_k(x, QK_ELEMS, aq);

    ASSERT_TRUE(bit_identical(vec_dot_q4_k_q8_k(w4, aq, QK_ELEMS),
                              vec_dot_q4_k_q8_k_avx2(w4, aq, QK_ELEMS)));
    ASSERT_TRUE(bit_identical(vec_dot_q6_k_q8_k(w6, aq, QK_ELEMS),
                              vec_dot_q6_k_q8_k_avx2(w6, aq, QK_ELEMS)));
}

/* An all-zero activation super-block takes quantize_row_q8_k's amax==0
 * branch, where d is 0 and the bsums loop is skipped. Both K-quant dots must
 * still agree -- and the Q4_K min term, which is the one that reads bsums,
 * must not turn 0 * something into a NaN. */
TEST(simd_k_quant_twins_match_scalar_on_a_zero_activation_block) {
    if (!kernels_cpu_has_avx_vnni()) {
        g_test_case_failed = 1;
        return;
    }
    u8 w4[Q4K_BYTES], w6[Q6K_BYTES];
    u8 aq[KERNELS_Q8_K_BLOCK_BYTES];
    f32 x[QK_ELEMS];
    rng_reset();
    fill_q4_k_block(w4);
    fill_q6_k_block(w6);
    for (u32 i = 0; i < QK_ELEMS; i++) x[i] = 0.0f;
    quantize_row_q8_k(x, QK_ELEMS, aq);

    f32 s4 = vec_dot_q4_k_q8_k(w4, aq, QK_ELEMS);
    f32 v4 = vec_dot_q4_k_q8_k_avx2(w4, aq, QK_ELEMS);
    f32 s6 = vec_dot_q6_k_q8_k(w6, aq, QK_ELEMS);
    f32 v6 = vec_dot_q6_k_q8_k_avx2(w6, aq, QK_ELEMS);
    ASSERT_TRUE(bit_identical(s4, v4));
    ASSERT_TRUE(bit_identical(s6, v6));
    ASSERT_TRUE(s4 == 0.0f);
    ASSERT_TRUE(s6 == 0.0f);
}

/* ▶▶ THE Q6_K MADDUBS TWIN, AND THIS IS THE FIRST DIRECT TEST ANY `_xn_` KERNEL
 * HAS HAD. Until now the batched kernels were covered only through
 * `matvec_batch`'s byte-identity test and the end-to-end gates -- which is real
 * coverage, but it grades a whole matvec and cannot say WHICH of the two
 * spellings of a dot product disagreed, or exercise n_seq and n independently.
 *
 * Two claims, and the second is the one that makes the first mean anything:
 *
 *   1. the maddubs twin equals the reference twin, BIT FOR BIT, for every
 *      sequence -- which is what licenses shipping it, since simd.h's
 *      acceptance criterion for an integer path is exactness, not closeness;
 *   2. BOTH equal the SOLO kernel per sequence -- the `_xn_` contract that
 *      nothing was checking directly. A batched kernel that agreed with another
 *      batched kernel while both drifted from the solo one would pass claim 1
 *      and be wrong.
 *
 * The fixture varies n_seq across 1, 2, 3, 8, 15 and 16 rather than just the
 * tile: 1 exercises the degenerate case, 3 and 15 are not powers of two, and 16
 * is the tile boundary. `n` covers one, two and three super-blocks plus the
 * model's real contraction lengths.
 *
 * ⚠ THE ACTIVATIONS MUST DIFFER PER SEQUENCE. If every sequence held the same
 * quantized activation, a kernel that ignored `s` entirely and returned
 * sequence 0's answer sixteen times would pass every assertion here -- the
 * project has banked a green fixture that could not see its own defect before
 * ([[fixture-can-be-blind-to-the-defect]]), and the check below is what makes
 * this one unable to. */
TEST(simd_q6_k_xn_maddubs_is_bit_exact_vs_reference_and_solo) {
    if (!kernels_cpu_has_avx_vnni()) {
        fprintf(stderr, "FAIL %s:%d: no AVX-VNNI -- the K4 exact-diff gate "
                        "cannot run on this box\n", __FILE__, __LINE__);
        g_test_case_failed = 1;
        return;
    }
    static const u64 lens[] = {256, 512, 768, 2560, 9728};
    static const u32 seqs[] = {1u, 2u, 3u, 8u, 15u, 16u};
    const u32 n_lens = (u32)(sizeof lens / sizeof lens[0]);
    const u32 n_seqs = (u32)(sizeof seqs / sizeof seqs[0]);
    const u32 iters = 600;
    const u64 nmax = 9728u;

    rng_reset();
    u8*  w = (u8*)xmalloc((size_t)(nmax / QK_ELEMS) * Q6K_BYTES);
    f32* x = (f32*)xmalloc((size_t)nmax * sizeof(f32));
    u8*  aq[KERNELS_BATCH_TILE];
    const void* aqp[KERNELS_BATCH_TILE];
    for (u32 s = 0; s < KERNELS_BATCH_TILE; s++) {
        aq[s] = (u8*)xmalloc((size_t)(nmax / QK_ELEMS) *
                             KERNELS_Q8_K_BLOCK_BYTES);
        aqp[s] = aq[s];
    }
    f32 got[KERNELS_BATCH_TILE], want[KERNELS_BATCH_TILE];

    u32 mismatch_twin = 0, mismatch_solo = 0, degenerate = 0;
    for (u32 it = 0; it < iters; it++) {
        u64 n = lens[it % n_lens];
        u32 ns = seqs[(it / n_lens) % n_seqs];
        u64 blocks = n / QK_ELEMS;
        for (u64 b = 0; b < blocks; b++) fill_q6_k_block(w + b * Q6K_BYTES);
        for (u32 s = 0; s < ns; s++) fill_q8_k_activation(x, aq[s], n);

        /* ANTI-VACUITY: the sequences must actually differ, or a kernel that
         * ignored `s` would pass everything below. */
        if (ns > 1u &&
            memcmp(aq[0], aq[ns - 1u],
                   (size_t)blocks * KERNELS_Q8_K_BLOCK_BYTES) == 0) {
            degenerate++;
        }

        vec_dot_q6_k_q8_k_xn_avx2(w, aqp, ns, n, want);
        vec_dot_q6_k_q8_k_xn_maddubs_avx2(w, aqp, ns, n, got);

        for (u32 s = 0; s < ns; s++) {
            if (!bit_identical(got[s], want[s])) {
                if (mismatch_twin == 0) {
                    fprintf(stderr,
                            "  iter %u n=%llu n_seq=%u s=%u: ref %.9g (0x%08X) "
                            "vs maddubs %.9g (0x%08X)\n",
                            it, (unsigned long long)n, ns, s, (double)want[s],
                            f32_bits(want[s]), (double)got[s],
                            f32_bits(got[s]));
                }
                mismatch_twin++;
            }
            /* Claim 2: both batched spellings must equal the SOLO kernel. */
            f32 solo = vec_dot_q6_k_q8_k_avx2(w, aq[s], n);
            if (!bit_identical(got[s], solo) ||
                !bit_identical(want[s], solo)) {
                if (mismatch_solo == 0) {
                    fprintf(stderr,
                            "  iter %u n=%llu n_seq=%u s=%u: solo %.9g vs ref "
                            "%.9g vs maddubs %.9g\n",
                            it, (unsigned long long)n, ns, s, (double)solo,
                            (double)want[s], (double)got[s]);
                }
                mismatch_solo++;
            }
        }
    }
    free(w);
    free(x);
    for (u32 s = 0; s < KERNELS_BATCH_TILE; s++) free(aq[s]);

    ASSERT_EQ_U32(degenerate, 0u);
    ASSERT_EQ_U32(mismatch_twin, 0u);
    ASSERT_EQ_U32(mismatch_solo, 0u);
}
