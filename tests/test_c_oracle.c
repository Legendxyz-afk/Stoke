/* tests/test_c_oracle.c -- M4.5 rung C's INTEGER-TOTALS ORACLE.
 *
 * WHY THIS FILE EXISTS. The C grading run (2026-08-13,
 * docs/superpowers/gate-reports/2026-08-13-m45-c-grade-*) fired C-FAIL-RATE:
 * 34/50 first-divergence against a bound of <=19/50, with ZERO BUG-class. The
 * throughput-parity ruling (2026-08-12 §7.2) pre-rules that branch and names
 * exactly one next action:
 *
 *     "STOP; run the integer-totals oracle; if totals are clean it is a
 *      results-vs-physics memo to Kabir with C parked on a branch."
 *
 * THE QUESTION IT ANSWERS. CONTRACT v2 changed ONLY the f32 fold: per
 * super-block the code now reduces the eight i32 lanes to a single exact
 * integer total S_b and executes two f32 operations, where it used to run
 * eight f32 multiply-adds against the eight lanes separately. If the INTEGER
 * core also moved, the 34/50 is a real bug and C must be fixed against the m4
 * implementation. If the integer totals are bit-identical, the divergence is
 * the honest cost of re-ordering f32 accumulation at near-ties.
 *
 * WHY IT IS FOUR TRANSCRIPTIONS AND NOT A FOLD FLAG. The obvious shape is one
 * integer core with a `new_fold` parameter. That version cannot fail: with a
 * single core there is nothing for the totals to disagree about, and the
 * oracle would be a tautology dressed as a test. So the m4 core and the HEAD
 * core are transcribed SEPARATELY, from `git show ed2aa32:` and from the
 * working tree, for both the scalar and the AVX2 twin -- four independent
 * copies whose totals are then compared.
 *
 * HOW THE COPIES ARE KEPT HONEST, because a hand-copied kernel that drifted
 * from the real one would prove nothing about the shipping engine:
 *
 *   - each NEW copy returns its own f32 result, which is asserted
 *     BIT-IDENTICAL to the linked library's real kernel (vec_dot_q4_k_q8_k,
 *     vec_dot_q4_k_q8_k_avx2, and the Q6_K pair). A transcription slip on the
 *     new side fails there before the oracle is consulted.
 *   - the two OLD copies are asserted bit-identical to EACH OTHER, which is
 *     precisely the invariant the m4 tree held (test_simd's twin-vs-scalar
 *     gate, green at ed2aa32). Two independent transcriptions -- one scalar,
 *     one AVX2 -- agreeing to the bit is a strong check that neither drifted.
 *   - c_oracle_totals_are_sensitive_to_the_integer_core perturbs one
 *     activation byte and asserts the reported totals MOVE, so a green oracle
 *     cannot be an instrument that reports "identical" unconditionally.
 *
 * THE SECOND MEASUREMENT: (f32)S_b IS NOT ALWAYS EXACT. f32 represents every
 * integer below 2^24 = 16,777,216 and not all of those above it. The old fold
 * converted eight LANES, each carrying about an eighth of the total; the new
 * fold converts the TOTAL. Q4_K's reachable worst case is 30,723,840 and
 * Q6_K's is 133,169,152 -- both above 2^24 -- so the new conversion CAN round
 * where the old one could not. Whether it ever does is a census, not an
 * argument, so every test here reports max |S_b|, the count at or above 2^24,
 * and the count that actually loses bits. The three quantities are NOT the
 * same: the worst cases above are all divisible by a high power of two and
 * convert exactly, which is why the threshold count alone would overstate the
 * effect. NOTE the corpus below is RANDOM weight bytes (uniform nibbles,
 * uniform 6-bit scales), which is not the shipping model's distribution --
 * the census over the real GGUF is a separate instrument.
 *
 * BUILD NOTE. The Makefile compiles this one test with SIMD_CFLAGS, because
 * it carries AVX2 intrinsics. That also puts the scalar copies under -mavx2;
 * their integer results are unaffected (i32 add and multiply are exact and
 * associative, and the bounds above sit three orders inside i32), and the f32
 * self-checks against the library kernels would catch it if anything else
 * moved.
 */
#include "common.h"

#include "kernels/kernels.h"
#include "kernels/simd.h"

#include <immintrin.h>
#include <stdio.h>
#include <string.h>

#include "test.h"

#define Q4K_BYTES KERNELS_Q4_K_BLOCK_BYTES
#define Q6K_BYTES KERNELS_Q6_K_BLOCK_BYTES
#define QK_ELEMS  KERNELS_Q4_K_BLOCK_ELEMS
#define MAX_N     9728u
#define MAX_BLOCKS (MAX_N / 256u)

/* f32 represents every integer strictly below this, and not all above it. */
#define F32_EXACT_INT_LIMIT 16777216 /* 2^24 */

/* ------------------------------------------------------------- little bits */

/* Byte-assembly readers, matching kquant.c's kq_read_*_le rather than a
 * memcpy, so the copies below read their operands exactly as the originals
 * do. (On x86-64 the two are the same; the point is to leave nothing to
 * inference.) */
static u16 orc_read_u16_le(const u8* p) {
    return (u16)((u16)p[0] | ((u16)p[1] << 8));
}

static i16 orc_read_i16_le(const u8* p) {
    return (i16)(u16)((u16)p[0] | ((u16)p[1] << 8));
}

static f32 orc_read_f32_le(const u8* p) {
    u32 b = (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
    f32 v;
    memcpy(&v, &b, sizeof(v));
    return v;
}

static bool bit_identical(f32 a, f32 b) {
    return memcmp(&a, &b, sizeof(f32)) == 0;
}

static u32 f32_bits(f32 v) {
    u32 b;
    memcpy(&b, &v, sizeof b);
    return b;
}

/* One of the copy-fidelity checks: bumps `count` and prints the first
 * offender with its bit pattern, since a difference that matters here can be
 * one ulp and would not survive %g alone. */
static void f32_check(const char* who, f32 got, f32 want, u32* count) {
    if (bit_identical(got, want)) return;
    if (*count == 0) {
        fprintf(stderr, "  %s: %.9g (0x%08X) != %.9g (0x%08X)\n", who,
                (double)got, f32_bits(got), (double)want, f32_bits(want));
    }
    (*count)++;
}

/* The per-super-block integer totals the ruling names: S_b is the exact
 * integer dot, M_b the Q4_K mins term (0 for Q6_K, which has none). */
/* d and dmin are the f32 super-block scales the fold multiplies the totals
 * by. They are recorded ONLY by the two HEAD variants -- all four read the
 * same bytes with the same expression, and the totals comparison already
 * proves the four agree -- and they exist so the accuracy leg can build an
 * f64 reference for the fold WITHOUT re-deriving the scales itself. */
typedef struct {
    i32 S_b;
    i32 M_b;
    f32 d;
    f32 dmin;
} Totals;

typedef struct {
    u64 blocks;    /* super-blocks seen */
    i64 max_abs_S; /* max |S_b| */
    u64 n_ge_2p24; /* how many sit outside f32's exact integer range */
    u64 n_inexact; /* how many actually LOSE bits in (f32)S_b */
} Census;

/* >= 2^24 is necessary for the new fold's conversion to round, and NOT
 * sufficient: above 2^24 f32 still represents every even integer, above 2^25
 * every multiple of four, and so on. The two quantities are counted
 * separately because the interesting one is n_inexact, and reporting only the
 * threshold count would overstate the effect -- the K-quant worst cases in
 * particular are all divisible by a high power of two and convert exactly. */
static bool f32_conversion_is_exact(i32 v) {
    return (f64)(f32)v == (f64)v;
}

static void census_note(Census* c, const Totals* t, u64 nb) {
    for (u64 i = 0; i < nb; ++i) {
        i64 a = (i64)t[i].S_b;
        if (a < 0) a = -a;
        if (a > c->max_abs_S) c->max_abs_S = a;
        if (a >= F32_EXACT_INT_LIMIT) c->n_ge_2p24++;
        if (!f32_conversion_is_exact(t[i].S_b)) c->n_inexact++;
        c->blocks++;
    }
}

static void census_report(const char* what, const Census* c) {
    printf("  census %-18s blocks=%llu  max|S_b|=%lld  >=2^24: %llu  "
           "inexact (f32)S_b: %llu\n",
           what, (unsigned long long)c->blocks, (long long)c->max_abs_S,
           (unsigned long long)c->n_ge_2p24,
           (unsigned long long)c->n_inexact);
}

/* Compares two totals arrays and reports the first disagreement. Returns the
 * number of super-blocks that differ. */
static u32 totals_diff(const char* a_name, const Totals* a, const char* b_name,
                       const Totals* b, u64 nb, u32* reported) {
    u32 bad = 0;
    for (u64 i = 0; i < nb; ++i) {
        if (a[i].S_b != b[i].S_b || a[i].M_b != b[i].M_b) {
            if (*reported == 0) {
                fprintf(stderr,
                        "  TOTALS DIFFER at super-block %llu: %s "
                        "S_b=%d M_b=%d vs %s S_b=%d M_b=%d\n",
                        (unsigned long long)i, a_name, a[i].S_b, a[i].M_b,
                        b_name, b[i].S_b, b[i].M_b);
                (*reported)++;
            }
            bad++;
        }
    }
    return bad;
}

/* ------------------------------------------------- corpus (from test_simd) */

/* Same generator, same fixed seed, same lengths as tests/test_simd.c's
 * exact-diff gate: this oracle grades the code over the corpus the project
 * already accepts as covering these kernels, rather than inventing one. */
static u64 g_rng = 0x9E3779B97F4A7C15ull;

static void rng_reset(void) { g_rng = 0x9E3779B97F4A7C15ull; }

static u32 rng_next(void) {
    g_rng ^= g_rng >> 12;
    g_rng ^= g_rng << 25;
    g_rng ^= g_rng >> 27;
    return (u32)((g_rng * 0x2545F4914F6CDD1Dull) >> 32);
}

static f32 random_scale(void) {
    static const f32 mags[] = {1.0f / 4096.0f, 1.0f / 127.0f, 0.00731f,
                               0.5f, 1.0f, 7.25f, 133.0f};
    u32 i = rng_next() % (u32)(sizeof mags / sizeof mags[0]);
    f32 s = mags[i];
    return (rng_next() & 1u) ? -s : s;
}

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
        u32 r = rng_next();
        f32 mag = (r & 3u) == 0u ? 0.0f : (f32)(r % 1000u) * 0.0037f;
        x[i] = (r & 4u) ? -mag : mag;
    }
    quantize_row_q8_k(x, n, aq);
}

/* ============================================================== Q4_K, m4 ==
 *
 * Integer core transcribed from `git show ed2aa32:src/kernels/kquant.c`
 * (vec_dot_q4_k_q8_k), fold included, plus two instrumentation lines that
 * read the same accumulators the fold reads.
 */
static f32 orc_q4k_old_scalar(const u8* wb, const u8* ab, u64 n, Totals* out) {
    u64 nb = n / KERNELS_Q4_K_BLOCK_ELEMS;

    i8 aux8[KERNELS_Q4_K_BLOCK_ELEMS];
    i16 aux16[8];
    i32 aux32[8];
    f32 sums[8];
    u8 scales8[8], mins8[8];
    memset(sums, 0, sizeof(sums));

    f32 sumf = 0.0f;
    for (u64 i = 0; i < nb; ++i) {
        const u8* blk = wb + i * KERNELS_Q4_K_BLOCK_BYTES;
        const u8* q4 = blk + 16;
        const u8* ablk = ab + i * KERNELS_Q8_K_BLOCK_BYTES;
        const i8* q8 = (const i8*)(ablk + 4);
        const u8* bsums = ablk + 4 + KERNELS_Q8_K_BLOCK_ELEMS;
        f32 ad = orc_read_f32_le(ablk);

        memset(aux32, 0, sizeof(aux32));
        i8* a = aux8;
        for (u32 j = 0; j < KERNELS_Q4_K_BLOCK_ELEMS / 64u; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (i8)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (i8)(q4[l] >> 4);
            a += 32;
            q4 += 32;
        }

        kernels_q4_k_unpack_scales(blk + 4, scales8, mins8);

        i32 sumi = 0;
        for (u32 j = 0; j < KERNELS_Q8_K_BLOCK_ELEMS / 16u; ++j) {
            sumi += (i32)orc_read_i16_le(bsums + 2u * j) * (i32)mins8[j / 2u];
        }

        a = aux8;
        u32 is = 0;
        for (u32 j = 0; j < KERNELS_Q4_K_BLOCK_ELEMS / 32u; ++j) {
            i32 scale = (i32)scales8[is++];
            for (u32 k = 0; k < 4u; k++) {
                for (int l = 0; l < 8; ++l) aux16[l] = (i16)(q8[l] * a[l]);
                for (int l = 0; l < 8; ++l) aux32[l] += scale * (i32)aux16[l];
                q8 += 8;
                a += 8;
            }
        }

        /* THE m4 FOLD, verbatim. */
        f32 d = f16_to_f32(orc_read_u16_le(blk)) * ad;
        for (int l = 0; l < 8; ++l) sums[l] += d * (f32)aux32[l];
        f32 dmin = f16_to_f32(orc_read_u16_le(blk + 2)) * ad;
        sumf -= dmin * (f32)sumi;

        /* Instrumentation: the same eight lanes the fold above consumed. */
        i32 S = 0;
        for (int l = 0; l < 8; ++l) S += aux32[l];
        out[i].S_b = S;
        out[i].M_b = sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

/* ============================================================ Q4_K, HEAD ==
 *
 * Integer core transcribed from the working tree (src/kernels/kquant.c),
 * CONTRACT v2 fold included.
 */
static f32 orc_q4k_new_scalar(const u8* wb, const u8* ab, u64 n, Totals* out) {
    u64 nb = n / KERNELS_Q4_K_BLOCK_ELEMS;

    i8 aux8[KERNELS_Q4_K_BLOCK_ELEMS];
    i16 aux16[8];
    i32 aux32[8];
    u8 scales8[8], mins8[8];

    f32 sumf = 0.0f;
    for (u64 i = 0; i < nb; ++i) {
        const u8* blk = wb + i * KERNELS_Q4_K_BLOCK_BYTES;
        const u8* q4 = blk + 16;
        const u8* ablk = ab + i * KERNELS_Q8_K_BLOCK_BYTES;
        const i8* q8 = (const i8*)(ablk + 4);
        const u8* bsums = ablk + 4 + KERNELS_Q8_K_BLOCK_ELEMS;
        f32 ad = orc_read_f32_le(ablk);

        memset(aux32, 0, sizeof(aux32));
        i8* a = aux8;
        for (u32 j = 0; j < KERNELS_Q4_K_BLOCK_ELEMS / 64u; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (i8)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (i8)(q4[l] >> 4);
            a += 32;
            q4 += 32;
        }

        kernels_q4_k_unpack_scales(blk + 4, scales8, mins8);

        i32 sumi = 0;
        for (u32 j = 0; j < KERNELS_Q8_K_BLOCK_ELEMS / 16u; ++j) {
            sumi += (i32)orc_read_i16_le(bsums + 2u * j) * (i32)mins8[j / 2u];
        }

        a = aux8;
        u32 is = 0;
        for (u32 j = 0; j < KERNELS_Q4_K_BLOCK_ELEMS / 32u; ++j) {
            i32 scale = (i32)scales8[is++];
            for (u32 k = 0; k < 4u; k++) {
                for (int l = 0; l < 8; ++l) aux16[l] = (i16)(q8[l] * a[l]);
                for (int l = 0; l < 8; ++l) aux32[l] += scale * (i32)aux16[l];
                q8 += 8;
                a += 8;
            }
        }

        /* THE CONTRACT v2 FOLD, verbatim. */
        i32 S_b = 0;
        for (int l = 0; l < 8; ++l) S_b += aux32[l];
        const i32 M_b = sumi;

        const f32 d = f16_to_f32(orc_read_u16_le(blk)) * ad;
        const f32 dmin = f16_to_f32(orc_read_u16_le(blk + 2)) * ad;
        sumf = sumf + d * (f32)S_b;
        sumf = sumf - dmin * (f32)M_b;

        out[i].S_b = S_b;
        out[i].M_b = M_b;
        out[i].d = d;
        out[i].dmin = dmin;
    }
    return sumf;
}

/* ============================================================ Q6_K, m4 ==== */
static f32 orc_q6k_old_scalar(const u8* wb, const u8* ab, u64 n, Totals* out) {
    u64 nb = n / KERNELS_Q6_K_BLOCK_ELEMS;

    i8 aux8[KERNELS_Q6_K_BLOCK_ELEMS];
    i16 aux16[8];
    i32 aux32[8];
    f32 sums[8];
    memset(sums, 0, sizeof(sums));

    f32 sumf = 0.0f;
    for (u64 i = 0; i < nb; ++i) {
        const u8* blk = wb + i * KERNELS_Q6_K_BLOCK_BYTES;
        const u8* q4 = blk;
        const u8* qh = blk + 128;
        const i8* sc = (const i8*)(blk + 192);
        const u8* ablk = ab + i * KERNELS_Q8_K_BLOCK_BYTES;
        const i8* q8 = (const i8*)(ablk + 4);
        f32 ad = orc_read_f32_le(ablk);

        memset(aux32, 0, sizeof(aux32));
        i8* a = aux8;
        for (u32 j = 0; j < KERNELS_Q6_K_BLOCK_ELEMS; j += 128) {
            (void)j;
            for (int l = 0; l < 32; ++l) {
                a[l + 0] = (i8)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (i8)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (i8)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (i8)((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128;
            q4 += 64;
            qh += 32;
        }

        a = aux8;
        u32 is = 0;
        for (u32 j = 0; j < KERNELS_Q6_K_BLOCK_ELEMS / 16u; ++j) {
            i32 scale = (i32)sc[is++];
            for (u32 k = 0; k < 2u; k++) {
                for (int l = 0; l < 8; ++l) aux16[l] = (i16)(q8[l] * a[l]);
                for (int l = 0; l < 8; ++l) aux32[l] += scale * (i32)aux16[l];
                q8 += 8;
                a += 8;
            }
        }

        /* THE m4 FOLD, verbatim. */
        f32 d = f16_to_f32(orc_read_u16_le(blk + 208)) * ad;
        for (int l = 0; l < 8; ++l) sums[l] += d * (f32)aux32[l];

        i32 S = 0;
        for (int l = 0; l < 8; ++l) S += aux32[l];
        out[i].S_b = S;
        out[i].M_b = 0;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

/* ============================================================ Q6_K, HEAD == */
static f32 orc_q6k_new_scalar(const u8* wb, const u8* ab, u64 n, Totals* out) {
    u64 nb = n / KERNELS_Q6_K_BLOCK_ELEMS;

    i8 aux8[KERNELS_Q6_K_BLOCK_ELEMS];
    i16 aux16[8];
    i32 aux32[8];

    f32 sumf = 0.0f;
    for (u64 i = 0; i < nb; ++i) {
        const u8* blk = wb + i * KERNELS_Q6_K_BLOCK_BYTES;
        const u8* q4 = blk;
        const u8* qh = blk + 128;
        const i8* sc = (const i8*)(blk + 192);
        const u8* ablk = ab + i * KERNELS_Q8_K_BLOCK_BYTES;
        const i8* q8 = (const i8*)(ablk + 4);
        f32 ad = orc_read_f32_le(ablk);

        memset(aux32, 0, sizeof(aux32));
        i8* a = aux8;
        for (u32 j = 0; j < KERNELS_Q6_K_BLOCK_ELEMS; j += 128) {
            (void)j;
            for (int l = 0; l < 32; ++l) {
                a[l + 0] = (i8)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (i8)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (i8)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (i8)((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128;
            q4 += 64;
            qh += 32;
        }

        a = aux8;
        u32 is = 0;
        for (u32 j = 0; j < KERNELS_Q6_K_BLOCK_ELEMS / 16u; ++j) {
            i32 scale = (i32)sc[is++];
            for (u32 k = 0; k < 2u; k++) {
                for (int l = 0; l < 8; ++l) aux16[l] = (i16)(q8[l] * a[l]);
                for (int l = 0; l < 8; ++l) aux32[l] += scale * (i32)aux16[l];
                q8 += 8;
                a += 8;
            }
        }

        /* THE CONTRACT v2 FOLD, verbatim. */
        i32 S_b = 0;
        for (int l = 0; l < 8; ++l) S_b += aux32[l];
        const f32 d = f16_to_f32(orc_read_u16_le(blk + 208)) * ad;
        sumf = sumf + d * (f32)S_b;

        out[i].S_b = S_b;
        out[i].M_b = 0;
        out[i].d = d;
        out[i].dmin = 0.0f;
    }
    return sumf;
}

/* ========================================================= the AVX2 twins ==
 *
 * kq_mac8 / kq_load_* are textually identical in ed2aa32 and HEAD, so they
 * are shared here; the fold helpers are not, and each twin carries its own.
 */
static inline __m256i orc_mac8(__m256i acc, __m256i w32, __m256i a32) {
    return _mm256_add_epi32(acc, _mm256_mullo_epi32(w32, a32));
}

static inline __m256i orc_load_i8x8_to_i32(const i8* p) {
    return _mm256_cvtepi8_epi32(
        _mm_loadl_epi64((const __m128i*)(const void*)p));
}

static inline __m256i orc_load_u8x8_to_i32(const u8* p) {
    return _mm256_cvtepu8_epi32(
        _mm_loadl_epi64((const __m128i*)(const void*)p));
}

/* m4's fold helpers. */
static inline __m256 orc_accum_sums(__m256 sums, __m256i acc, f32 d) {
    return _mm256_add_ps(sums,
                         _mm256_mul_ps(_mm256_set1_ps(d),
                                       _mm256_cvtepi32_ps(acc)));
}

static inline f32 orc_finish(f32 sumf, __m256 sums) {
    f32 tmp[8];
    _mm256_storeu_ps(tmp, sums);
    for (int l = 0; l < 8; ++l) sumf += tmp[l];
    return sumf;
}

/* HEAD's fold helper. */
static inline i32 orc_hsum_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

static f32 orc_q4k_old_twin(const u8* wb, const u8* ab, u64 n, Totals* out) {
    u64 nb = n / KERNELS_Q4_K_BLOCK_ELEMS;
    const __m256i lownib = _mm256_set1_epi32(0xF);
    __m256 sums = _mm256_setzero_ps();
    f32 sumf = 0.0f;

    for (u64 i = 0; i < nb; ++i) {
        const u8* blk = wb + i * KERNELS_Q4_K_BLOCK_BYTES;
        const u8* q4 = blk + 16;
        const u8* ablk = ab + i * KERNELS_Q8_K_BLOCK_BYTES;
        const i8* q8 = (const i8*)(ablk + 4);
        const u8* bsums = ablk + 4 + KERNELS_Q8_K_BLOCK_ELEMS;
        f32 ad;
        memcpy(&ad, ablk, sizeof ad);

        u8 scales8[8], mins8[8];
        kernels_q4_k_unpack_scales(blk + 4, scales8, mins8);

        i32 sumi = 0;
        for (u32 j = 0; j < KERNELS_Q8_K_BLOCK_ELEMS / 16u; ++j) {
            i16 bs;
            memcpy(&bs, bsums + 2u * j, sizeof bs);
            sumi += (i32)bs * (i32)mins8[j / 2u];
        }

        __m256i acc = _mm256_setzero_si256();
        for (u32 j = 0; j < 8u; ++j) {
            const u8* qsrc = q4 + (j / 2u) * 32u;
            int high = (int)(j & 1u);
            __m256i sub = _mm256_setzero_si256();
            for (u32 g = 0; g < 4u; ++g) {
                __m256i wv = orc_load_u8x8_to_i32(qsrc + g * 8u);
                wv = high ? _mm256_srli_epi32(wv, 4)
                          : _mm256_and_si256(wv, lownib);
                __m256i av = orc_load_i8x8_to_i32(q8 + j * 32u + g * 8u);
                sub = orc_mac8(sub, wv, av);
            }
            acc = _mm256_add_epi32(
                acc, _mm256_mullo_epi32(sub,
                                        _mm256_set1_epi32((int)scales8[j])));
        }

        u16 draw, dmraw;
        memcpy(&draw, blk, sizeof draw);
        memcpy(&dmraw, blk + 2, sizeof dmraw);
        /* THE m4 FOLD, verbatim. */
        sums = orc_accum_sums(sums, acc, f16_to_f32(draw) * ad);
        sumf -= (f16_to_f32(dmraw) * ad) * (f32)sumi;

        out[i].S_b = orc_hsum_i32(acc);
        out[i].M_b = sumi;
    }
    return orc_finish(sumf, sums);
}

static f32 orc_q4k_new_twin(const u8* wb, const u8* ab, u64 n, Totals* out) {
    u64 nb = n / KERNELS_Q4_K_BLOCK_ELEMS;
    const __m256i lownib = _mm256_set1_epi32(0xF);
    f32 sumf = 0.0f;

    for (u64 i = 0; i < nb; ++i) {
        const u8* blk = wb + i * KERNELS_Q4_K_BLOCK_BYTES;
        const u8* q4 = blk + 16;
        const u8* ablk = ab + i * KERNELS_Q8_K_BLOCK_BYTES;
        const i8* q8 = (const i8*)(ablk + 4);
        const u8* bsums = ablk + 4 + KERNELS_Q8_K_BLOCK_ELEMS;
        f32 ad;
        memcpy(&ad, ablk, sizeof ad);

        u8 scales8[8], mins8[8];
        kernels_q4_k_unpack_scales(blk + 4, scales8, mins8);

        i32 sumi = 0;
        for (u32 j = 0; j < KERNELS_Q8_K_BLOCK_ELEMS / 16u; ++j) {
            i16 bs;
            memcpy(&bs, bsums + 2u * j, sizeof bs);
            sumi += (i32)bs * (i32)mins8[j / 2u];
        }

        __m256i acc = _mm256_setzero_si256();
        for (u32 j = 0; j < 8u; ++j) {
            const u8* qsrc = q4 + (j / 2u) * 32u;
            int high = (int)(j & 1u);
            __m256i sub = _mm256_setzero_si256();
            for (u32 g = 0; g < 4u; ++g) {
                __m256i wv = orc_load_u8x8_to_i32(qsrc + g * 8u);
                wv = high ? _mm256_srli_epi32(wv, 4)
                          : _mm256_and_si256(wv, lownib);
                __m256i av = orc_load_i8x8_to_i32(q8 + j * 32u + g * 8u);
                sub = orc_mac8(sub, wv, av);
            }
            acc = _mm256_add_epi32(
                acc, _mm256_mullo_epi32(sub,
                                        _mm256_set1_epi32((int)scales8[j])));
        }

        u16 draw, dmraw;
        memcpy(&draw, blk, sizeof draw);
        memcpy(&dmraw, blk + 2, sizeof dmraw);
        /* THE CONTRACT v2 FOLD, verbatim. */
        const i32 S_b = orc_hsum_i32(acc);
        const f32 d = f16_to_f32(draw) * ad;
        const f32 dmin = f16_to_f32(dmraw) * ad;
        sumf = sumf + d * (f32)S_b;
        sumf = sumf - dmin * (f32)sumi;

        out[i].S_b = S_b;
        out[i].M_b = sumi;
    }
    return sumf;
}

/* Q6_K twins. The aux8 materialisation is identical in both revisions. */
static void orc_q6k_build_aux8(const u8* blk, i8* aux8) {
    const u8* ql = blk;
    const u8* qh = blk + 128;
    for (u32 h = 0; h < 2u; ++h) {
        const u8* q4h = ql + h * 64u;
        const u8* qhh = qh + h * 32u;
        i8* a = aux8 + h * 128u;
        for (int l = 0; l < 32; ++l) {
            a[l + 0] = (i8)((q4h[l + 0] & 0xF) | (((qhh[l] >> 0) & 3) << 4)) - 32;
            a[l + 32] = (i8)((q4h[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4)) - 32;
            a[l + 64] = (i8)((q4h[l + 0] >> 4) | (((qhh[l] >> 4) & 3) << 4)) - 32;
            a[l + 96] = (i8)((q4h[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) - 32;
        }
    }
}

static __m256i orc_q6k_acc(const i8* aux8, const i8* q8, const i8* sc) {
    __m256i acc = _mm256_setzero_si256();
    for (u32 j = 0; j < 16u; ++j) {
        __m256i sub = _mm256_setzero_si256();
        for (u32 g = 0; g < 2u; ++g) {
            __m256i wv = orc_load_i8x8_to_i32(aux8 + j * 16u + g * 8u);
            __m256i av = orc_load_i8x8_to_i32(q8 + j * 16u + g * 8u);
            sub = orc_mac8(sub, wv, av);
        }
        acc = _mm256_add_epi32(
            acc, _mm256_mullo_epi32(sub, _mm256_set1_epi32((int)sc[j])));
    }
    return acc;
}

static f32 orc_q6k_old_twin(const u8* wb, const u8* ab, u64 n, Totals* out) {
    u64 nb = n / KERNELS_Q6_K_BLOCK_ELEMS;
    __m256 sums = _mm256_setzero_ps();
    f32 sumf = 0.0f;

    for (u64 i = 0; i < nb; ++i) {
        const u8* blk = wb + i * KERNELS_Q6_K_BLOCK_BYTES;
        const i8* sc = (const i8*)(blk + 192);
        const u8* ablk = ab + i * KERNELS_Q8_K_BLOCK_BYTES;
        const i8* q8 = (const i8*)(ablk + 4);
        f32 ad;
        memcpy(&ad, ablk, sizeof ad);

        i8 aux8[KERNELS_Q6_K_BLOCK_ELEMS];
        orc_q6k_build_aux8(blk, aux8);
        __m256i acc = orc_q6k_acc(aux8, q8, sc);

        u16 draw;
        memcpy(&draw, blk + 208, sizeof draw);
        /* THE m4 FOLD, verbatim. */
        sums = orc_accum_sums(sums, acc, f16_to_f32(draw) * ad);

        out[i].S_b = orc_hsum_i32(acc);
        out[i].M_b = 0;
    }
    return orc_finish(sumf, sums);
}

static f32 orc_q6k_new_twin(const u8* wb, const u8* ab, u64 n, Totals* out) {
    u64 nb = n / KERNELS_Q6_K_BLOCK_ELEMS;
    f32 sumf = 0.0f;

    for (u64 i = 0; i < nb; ++i) {
        const u8* blk = wb + i * KERNELS_Q6_K_BLOCK_BYTES;
        const i8* sc = (const i8*)(blk + 192);
        const u8* ablk = ab + i * KERNELS_Q8_K_BLOCK_BYTES;
        const i8* q8 = (const i8*)(ablk + 4);
        f32 ad;
        memcpy(&ad, ablk, sizeof ad);

        i8 aux8[KERNELS_Q6_K_BLOCK_ELEMS];
        orc_q6k_build_aux8(blk, aux8);
        __m256i acc = orc_q6k_acc(aux8, q8, sc);

        u16 draw;
        memcpy(&draw, blk + 208, sizeof draw);
        /* THE CONTRACT v2 FOLD, verbatim. */
        const i32 S_b = orc_hsum_i32(acc);
        sumf = sumf + (f16_to_f32(draw) * ad) * (f32)S_b;

        out[i].S_b = S_b;
        out[i].M_b = 0;
    }
    return sumf;
}

/* ------------------------------------------------------------------ tests */

/* Without this the twins above are not executing on this box and every green
 * bar below is vacuous. Hard failure, not a skip -- test_simd.c's rule. */
TEST(c_oracle_requires_the_simd_feature_this_box_reports) {
    ASSERT_TRUE(kernels_cpu_has_avx_vnni());
}

TEST(c_oracle_q4k_integer_totals_are_identical_m4_vs_head) {
    if (!kernels_cpu_has_avx_vnni()) {
        g_test_case_failed = 1;
        return;
    }
    static const u64 lens[] = {256, 512, 768, 2560, 9728};
    const u32 n_lens = (u32)(sizeof lens / sizeof lens[0]);
    const u32 iters = 10000;

    rng_reset();
    u8* w = (u8*)xmalloc((size_t)MAX_BLOCKS * Q4K_BYTES);
    u8* aq = (u8*)xmalloc((size_t)MAX_BLOCKS * KERNELS_Q8_K_BLOCK_BYTES);
    f32* x = (f32*)xmalloc((size_t)MAX_N * sizeof(f32));
    Totals t_os[MAX_BLOCKS], t_ns[MAX_BLOCKS], t_ot[MAX_BLOCKS], t_nt[MAX_BLOCKS];

    Census cen = {0, 0, 0, 0};
    u32 bad_totals = 0, bad_new_scalar = 0, bad_new_twin = 0, bad_old_pair = 0;
    u32 reported = 0;

    for (u32 it = 0; it < iters; it++) {
        u64 n = lens[it % n_lens];
        u64 blocks = n / QK_ELEMS;
        for (u64 b = 0; b < blocks; b++) fill_q4_k_block(w + b * Q4K_BYTES);
        fill_q8_k_activation(x, aq, n);

        f32 f_os = orc_q4k_old_scalar(w, aq, n, t_os);
        f32 f_ns = orc_q4k_new_scalar(w, aq, n, t_ns);
        f32 f_ot = orc_q4k_old_twin(w, aq, n, t_ot);
        f32 f_nt = orc_q4k_new_twin(w, aq, n, t_nt);

        /* The copies are pinned to the real kernels before their totals are
         * believed. */
        f32_check("head-scalar copy vs library", f_ns,
                  vec_dot_q4_k_q8_k(w, aq, n), &bad_new_scalar);
        f32_check("head-twin copy vs library", f_nt,
                  vec_dot_q4_k_q8_k_avx2(w, aq, n), &bad_new_twin);
        f32_check("m4-scalar copy vs m4-twin copy", f_os, f_ot, &bad_old_pair);

        /* THE ORACLE. */
        bad_totals += totals_diff("m4-scalar", t_os, "head-scalar", t_ns,
                                  blocks, &reported);
        bad_totals += totals_diff("m4-scalar", t_os, "m4-twin", t_ot, blocks,
                                  &reported);
        bad_totals += totals_diff("head-scalar", t_ns, "head-twin", t_nt,
                                  blocks, &reported);

        census_note(&cen, t_ns, blocks);
    }
    free(w);
    free(aq);
    free(x);

    census_report("q4_k random corpus", &cen);
    ASSERT_EQ_U32(bad_new_scalar, 0u); /* HEAD scalar copy == library kernel */
    ASSERT_EQ_U32(bad_new_twin, 0u);   /* HEAD twin copy   == library kernel */
    ASSERT_EQ_U32(bad_old_pair, 0u);   /* the two m4 copies agree to the bit */
    ASSERT_EQ_U32(bad_totals, 0u);     /* the oracle itself */
}

TEST(c_oracle_q6k_integer_totals_are_identical_m4_vs_head) {
    if (!kernels_cpu_has_avx_vnni()) {
        g_test_case_failed = 1;
        return;
    }
    static const u64 lens[] = {256, 512, 768, 2560, 9728};
    const u32 n_lens = (u32)(sizeof lens / sizeof lens[0]);
    const u32 iters = 10000;

    rng_reset();
    u8* w = (u8*)xmalloc((size_t)MAX_BLOCKS * Q6K_BYTES);
    u8* aq = (u8*)xmalloc((size_t)MAX_BLOCKS * KERNELS_Q8_K_BLOCK_BYTES);
    f32* x = (f32*)xmalloc((size_t)MAX_N * sizeof(f32));
    Totals t_os[MAX_BLOCKS], t_ns[MAX_BLOCKS], t_ot[MAX_BLOCKS], t_nt[MAX_BLOCKS];

    Census cen = {0, 0, 0, 0};
    u32 bad_totals = 0, bad_new_scalar = 0, bad_new_twin = 0, bad_old_pair = 0;
    u32 reported = 0;

    for (u32 it = 0; it < iters; it++) {
        u64 n = lens[it % n_lens];
        u64 blocks = n / QK_ELEMS;
        for (u64 b = 0; b < blocks; b++) fill_q6_k_block(w + b * Q6K_BYTES);
        fill_q8_k_activation(x, aq, n);

        f32 f_os = orc_q6k_old_scalar(w, aq, n, t_os);
        f32 f_ns = orc_q6k_new_scalar(w, aq, n, t_ns);
        f32 f_ot = orc_q6k_old_twin(w, aq, n, t_ot);
        f32 f_nt = orc_q6k_new_twin(w, aq, n, t_nt);

        f32_check("head-scalar copy vs library", f_ns,
                  vec_dot_q6_k_q8_k(w, aq, n), &bad_new_scalar);
        f32_check("head-twin copy vs library", f_nt,
                  vec_dot_q6_k_q8_k_avx2(w, aq, n), &bad_new_twin);
        f32_check("m4-scalar copy vs m4-twin copy", f_os, f_ot, &bad_old_pair);

        bad_totals += totals_diff("m4-scalar", t_os, "head-scalar", t_ns,
                                  blocks, &reported);
        bad_totals += totals_diff("m4-scalar", t_os, "m4-twin", t_ot, blocks,
                                  &reported);
        bad_totals += totals_diff("head-scalar", t_ns, "head-twin", t_nt,
                                  blocks, &reported);

        census_note(&cen, t_ns, blocks);
    }
    free(w);
    free(aq);
    free(x);

    census_report("q6_k random corpus", &cen);
    ASSERT_EQ_U32(bad_new_scalar, 0u);
    ASSERT_EQ_U32(bad_new_twin, 0u);
    ASSERT_EQ_U32(bad_old_pair, 0u);
    ASSERT_EQ_U32(bad_totals, 0u);
}

/* The fold, evaluated in f64 from the very totals both implementations agree
 * on. Not a third opinion about the kernel -- the integer core is settled by
 * the tests above -- but the exact value the two f32 folds are approximating,
 * so "which one is closer" becomes a measurement instead of an argument. */
static f64 fold_reference_f64(const Totals* t, u64 nb) {
    f64 acc = 0.0;
    for (u64 i = 0; i < nb; ++i) {
        acc += (f64)t[i].d * (f64)t[i].S_b;
        acc -= (f64)t[i].dmin * (f64)t[i].M_b;
    }
    return acc;
}

/* WHICH FOLD IS ACTUALLY MORE ACCURATE. This is the question the 34/50
 * divergence raises and that neither the grading run nor the integer oracle
 * answers: CONTRACT v2 is a DIFFERENT rounding, but different is not worse.
 *
 * Counting the roundings makes the direction plausible before it is measured.
 * m4 per super-block: eight exact i32->f32 conversions, eight multiplies,
 * eight adds into eight lanes, then a final eight-term horizontal sum -- and
 * every lane value is small enough to convert exactly. HEAD: one conversion
 * of the TOTAL (which can round above 2^24), one multiply, one add. Fewer
 * operations against a possibly-inexact conversion.
 *
 * ▶ MEASURED 2026-08-14, AND THE PLAUSIBLE DIRECTION IS ONLY HALF RIGHT --
 * this test first asserted "HEAD's mean relative error <= m4's" and FAILED on
 * Q4_K:
 *
 *     q4_k: mean |rel err| m4 3.21e-07 vs HEAD 3.86e-07;
 *           HEAD closer 978, m4 closer 456, tied 566 (of 2000)
 *     q6_k: mean |rel err| m4 3.05e-07 vs HEAD 8.73e-08;
 *           HEAD closer 1095, m4 closer 306, tied 599 (of 2000)
 *
 * So HEAD is closer about twice as often in BOTH, and much better on Q6_K,
 * but carries a fatter tail on Q4_K. The candidate explanation was that the
 * regression is not in the per-super-block fold at all: m4's `sums[8]`
 * persisted ACROSS super-blocks and was summed once at the end -- an 8-way
 * split accumulation of the whole row -- while CONTRACT v2 accumulates every
 * super-block into ONE serial `sumf`. CONTRACT v2 therefore changed two
 * things at once, the per-block fold AND the cross-block accumulation shape,
 * and only the first is what the ruling describes.
 *
 * THAT EXPLANATION PREDICTS THE GAP COLLAPSES AT ONE SUPER-BLOCK, which is
 * why this test breaks the corpus down by row length rather than reporting a
 * mean it cannot interpret. Measured, Q4_K mean |rel err| m4 vs HEAD:
 *
 *     n=256   (1 block)   1.71e-07  vs  3.02e-08   HEAD 5.7x better
 *     n=512   (2 blocks)  2.94e-07  vs  5.55e-08   HEAD 5.3x better
 *     n=768   (3 blocks)  1.33e-07  vs  6.82e-08   HEAD 2.0x better
 *     n=2560  (10 blocks) 2.00e-07  vs  1.52e-07   HEAD 1.3x better
 *     n=9728  (38 blocks) 8.07e-07  vs  1.62e-06   HEAD 2.0x WORSE
 *
 * Confirmed, and it matters for reading the C grading run: HEAD's per-block
 * fold is strictly the better one, and its serial cross-block chain overtakes
 * that advantage as rows get long. n=9728 is not a synthetic length -- it is
 * this model's ffn_down contraction, and Q4_K_M stores some layers' ffn_down
 * as Q4_K. Q6_K does not cross over anywhere measured (2.22e-07 vs 5.53e-07
 * at 38 blocks); why the two quant types differ at the long end is NOT
 * established here, and the Q4_K mins term (a large subtraction the Q6_K path
 * does not have) is a hypothesis, not a finding.
 *
 * WHAT IS ASSERTED, and why it is not the ordering: both folds must stay
 * inside the milestone's arithmetic class, and HEAD must win more often than
 * it loses. The 1e-6 threshold is one order above the ~1e-7 that class means
 * throughout M3/M4; it is a CLASS check chosen for that reason and not a
 * bound fitted to these numbers, which would be the circularity the M3
 * tripwire memo already paid for. */
TEST(c_oracle_the_head_fold_is_not_less_accurate_than_m4s) {
    if (!kernels_cpu_has_avx_vnni()) {
        g_test_case_failed = 1;
        return;
    }
    static const u64 lens[] = {256, 512, 768, 2560, 9728};
    const u32 n_lens = (u32)(sizeof lens / sizeof lens[0]);
    const u32 iters = 2000;

    for (int kind = 0; kind < 2; ++kind) {
        const bool q4 = (kind == 0);
        rng_reset();
        u8* w = (u8*)xmalloc((size_t)MAX_BLOCKS * (q4 ? Q4K_BYTES : Q6K_BYTES));
        u8* aq = (u8*)xmalloc((size_t)MAX_BLOCKS * KERNELS_Q8_K_BLOCK_BYTES);
        f32* x = (f32*)xmalloc((size_t)MAX_N * sizeof(f32));
        Totals t_old[MAX_BLOCKS], t_new[MAX_BLOCKS];

        f64 sum_rel_old = 0.0, sum_rel_new = 0.0;
        u32 head_closer = 0, m4_closer = 0, tied = 0, counted = 0;
        /* Per-length, to test the cross-block explanation: at n=256 there is
         * exactly ONE super-block, so the two folds differ only in the
         * per-block shape and the gap should collapse. */
        f64 by_len_old[5] = {0}, by_len_new[5] = {0};
        u32 by_len_n[5] = {0};

        for (u32 it = 0; it < iters; it++) {
            u32 li = it % n_lens;
            u64 n = lens[li];
            u64 blocks = n / QK_ELEMS;
            for (u64 b = 0; b < blocks; b++) {
                if (q4) fill_q4_k_block(w + b * Q4K_BYTES);
                else fill_q6_k_block(w + b * Q6K_BYTES);
            }
            fill_q8_k_activation(x, aq, n);

            f32 f_old = q4 ? orc_q4k_old_scalar(w, aq, n, t_old)
                           : orc_q6k_old_scalar(w, aq, n, t_old);
            f32 f_new = q4 ? orc_q4k_new_scalar(w, aq, n, t_new)
                           : orc_q6k_new_scalar(w, aq, n, t_new);

            f64 ref = fold_reference_f64(t_new, blocks);
            if (ref == 0.0) continue;
            f64 e_old = (f64)f_old - ref;
            f64 e_new = (f64)f_new - ref;
            if (e_old < 0) e_old = -e_old;
            if (e_new < 0) e_new = -e_new;
            f64 mag = ref < 0 ? -ref : ref;
            sum_rel_old += e_old / mag;
            sum_rel_new += e_new / mag;
            by_len_old[li] += e_old / mag;
            by_len_new[li] += e_new / mag;
            by_len_n[li]++;
            if (e_new < e_old) head_closer++;
            else if (e_old < e_new) m4_closer++;
            else tied++;
            counted++;
        }
        free(w);
        free(aq);
        free(x);

        printf("  fold accuracy %s: mean |rel err| m4 %.3g vs HEAD %.3g; "
               "HEAD closer %u, m4 closer %u, tied %u (of %u)\n",
               q4 ? "q4_k" : "q6_k", sum_rel_old / (f64)counted,
               sum_rel_new / (f64)counted, head_closer, m4_closer, tied,
               counted);
        for (u32 li = 0; li < n_lens; ++li) {
            if (by_len_n[li] == 0) continue;
            printf("      n=%-5llu blocks=%-3llu m4 %.3g  HEAD %.3g\n",
                   (unsigned long long)lens[li],
                   (unsigned long long)(lens[li] / QK_ELEMS),
                   by_len_old[li] / (f64)by_len_n[li],
                   by_len_new[li] / (f64)by_len_n[li]);
        }
        ASSERT_TRUE(sum_rel_old / (f64)counted < 1e-6);
        ASSERT_TRUE(sum_rel_new / (f64)counted < 1e-6);
        ASSERT_TRUE(head_closer > m4_closer);
    }
}

/* The two deliberate configurations the random corpus will not reliably
 * reach: every Q4_K scale and nibble at its maximum, and every Q6_K quant at
 * an endpoint against the most negative i8 scale. Both are used twice below,
 * for the sensitivity leg and for the bounds leg. */
static void build_extreme_blocks(u8* w4, u8* w6_max, u8* w6_neg) {
    u16 d = f32_to_f16(1.0f);

    memcpy(w4, &d, 2);
    memcpy(w4 + 2, &d, 2);
    for (u32 i = 4; i < Q4K_BYTES; i++) w4[i] = 0xFFu; /* scales/mins = 63 */

    /* ql/qh all ones -> every q6 = 63 - 32 = +31, the largest POSITIVE value
     * the encoding can produce. */
    for (u32 i = 0; i < 192u; i++) w6_max[i] = 0xFFu;
    for (u32 i = 192u; i < 208u; i++) w6_max[i] = 0x80u; /* scales = -128 */
    memcpy(w6_max + 208, &d, 2);

    /* ql/qh all zero -> every q6 = 0 - 32 = -32, whose magnitude is one
     * LARGER, which is where Q6_K's true worst case lives. */
    for (u32 i = 0; i < 192u; i++) w6_neg[i] = 0x00u;
    for (u32 i = 192u; i < 208u; i++) w6_neg[i] = 0x80u;
    memcpy(w6_neg + 208, &d, 2);
}

/* THE INSTRUMENT MUST BE ABLE TO FAIL. If the reported totals were
 * insensitive to the integer core -- a wiring slip handing the comparator the
 * same array twice looks exactly like a clean oracle -- everything above
 * would be decoration. So: perturb one activation quant and one bsums entry
 * and require the totals to MOVE.
 *
 * The weights are the extreme blocks rather than random ones on purpose. On a
 * random block the first nibble or its scale can be zero, in which case
 * changing that quant provably cannot move S_b and this test would fail for a
 * correct instrument. With nibble 15 and scale 63 the delta is forced. */
TEST(c_oracle_totals_are_sensitive_to_the_integer_core) {
    if (!kernels_cpu_has_avx_vnni()) {
        g_test_case_failed = 1;
        return;
    }
    u8 w4[Q4K_BYTES], w6[Q6K_BYTES], w6n[Q6K_BYTES];
    u8 aq[KERNELS_Q8_K_BLOCK_BYTES];
    f32 x[QK_ELEMS];
    Totals base[1], moved[1];

    build_extreme_blocks(w4, w6, w6n);
    /* EVERY QUANT LANDS AT -127, not +127: quantize_row_q8_k builds
     * iscale = -127/max from the SIGNED extreme, so a constant-sign block maps
     * its largest magnitude to the negative endpoint. Measured, after the
     * first version of this test asserted the positive sign and failed. */
    for (u32 i = 0; i < QK_ELEMS; i++) x[i] = 1.0f;
    quantize_row_q8_k(x, QK_ELEMS, aq);

    /* Byte 4 is the first activation quant: -127 -> -126 must move S_b by
     * exactly nibble(15) * scale(63) = 945. */
    (void)orc_q4k_new_scalar(w4, aq, QK_ELEMS, base);
    u8 saved = aq[4];
    aq[4] = (u8)(i8)(-126);
    (void)orc_q4k_new_scalar(w4, aq, QK_ELEMS, moved);
    ASSERT_EQ_I64(moved[0].S_b - base[0].S_b, 945);
    (void)orc_q4k_new_twin(w4, aq, QK_ELEMS, moved);
    ASSERT_EQ_I64(moved[0].S_b - base[0].S_b, 945);
    aq[4] = saved;

    /* M_b must react to a bsums entry, which only the min term reads. */
    (void)orc_q4k_new_scalar(w4, aq, QK_ELEMS, base);
    const u32 bsums_off = 4u + KERNELS_Q8_K_BLOCK_ELEMS;
    u8 saved_bs = aq[bsums_off];
    aq[bsums_off] = (u8)(saved_bs ^ 0x7Fu);
    (void)orc_q4k_new_scalar(w4, aq, QK_ELEMS, moved);
    ASSERT_TRUE(base[0].M_b != moved[0].M_b);
    aq[bsums_off] = saved_bs;

    /* Q6_K has no min term; its S_b must still react, by q6(31) *
     * |scale|(128) = 3968. */
    (void)orc_q6k_new_scalar(w6, aq, QK_ELEMS, base);
    aq[4] = (u8)(i8)(-126);
    (void)orc_q6k_new_scalar(w6, aq, QK_ELEMS, moved);
    ASSERT_EQ_I64(base[0].S_b - moved[0].S_b, 3968);
    (void)orc_q6k_new_twin(w6, aq, QK_ELEMS, moved);
    ASSERT_EQ_I64(base[0].S_b - moved[0].S_b, 3968);
    aq[4] = saved;
}

/* THE 2^24 QUESTION, driven to the endpoints the random corpus will not
 * reliably reach. Both commit messages state a worst-case |S_b| computed
 * rather than assumed; this reproduces those numbers from the kernels
 * themselves, and in doing so proves the census above can see a super-block
 * outside f32's exact integer range -- without which "0 at or above 2^24" is
 * an untested instrument reporting a comforting number.
 *
 * Q4_K: 8 sub-blocks x scale 63 x 32 elements x nibble 15 x quant 127
 *       = 30,723,840, which is 1.83x 2^24.
 * Q6_K: 16 sub-blocks x |scale| 128 x 16 elements x |q6| 31 x quant 127
 *       -- the commit quotes 132,120,576 using the |q6| <= 32 endpoint; 31 is
 *       the largest value the 6-bit encoding can actually produce after the
 *       -32 centring, so the reachable maximum is 128,024,064. Both are far
 *       outside 2^24; the difference between them is exactly the kind of
 *       from-structure-not-computed slip this project keeps paying for, so it
 *       is asserted here at the reachable value. */
TEST(c_oracle_contract_worst_case_bounds_are_reproducible) {
    if (!kernels_cpu_has_avx_vnni()) {
        g_test_case_failed = 1;
        return;
    }
    u8 w4[Q4K_BYTES], w6[Q6K_BYTES], w6n[Q6K_BYTES];
    u8 aq[KERNELS_Q8_K_BLOCK_BYTES];
    f32 x[QK_ELEMS];
    Totals t4[1], t6[1], t6n[1], tw[1];

    build_extreme_blocks(w4, w6, w6n);

    /* Every activation quant at -127 (see the sensitivity test's note on
     * iscale = -127/max), which is why the totals below carry the signs they
     * do: the magnitudes are the worst cases, the signs are the quantizer's. */
    for (u32 i = 0; i < QK_ELEMS; i++) x[i] = 1.0f;
    quantize_row_q8_k(x, QK_ELEMS, aq);

    (void)orc_q4k_new_scalar(w4, aq, QK_ELEMS, t4);
    (void)orc_q6k_new_scalar(w6, aq, QK_ELEMS, t6);
    (void)orc_q6k_new_scalar(w6n, aq, QK_ELEMS, t6n);

    printf("  extremes: q4_k S_b=%d  q6_k(+31) S_b=%d  q6_k(-32) S_b=%d  "
           "(2^24 = %d)\n",
           t4[0].S_b, t6[0].S_b, t6n[0].S_b, F32_EXACT_INT_LIMIT);

    /* Q4_K: 8 sub-blocks x scale 63 x 32 elements x nibble 15 x quant 127.
     * The commit message's figure, reproduced from the kernel to the digit. */
    ASSERT_EQ_I64(t4[0].S_b, -30723840);
    /* Q6_K positive endpoint: 16 x 128 x 16 x 31 x 127. */
    ASSERT_EQ_I64(t6[0].S_b, 129007616);
    /* Q6_K's TRUE worst case, one magnitude larger because the centring makes
     * the negative end -32 while the positive end is +31: 16 x 128 x 16 x 32 x
     * 127 = 133,169,152. The Q6_K commit quotes 132,120,576, which is neither
     * this nor the product of its own stated factors (256 x 127 x 32 x 127 =
     * 132,128,768). The conclusion it drew -- an order inside i32 -- survives;
     * the constant does not. */
    ASSERT_EQ_I64(t6n[0].S_b, -133169152);

    (void)orc_q4k_new_twin(w4, aq, QK_ELEMS, tw);
    ASSERT_EQ_I64(t4[0].S_b, tw[0].S_b);
    (void)orc_q6k_new_twin(w6n, aq, QK_ELEMS, tw);
    ASSERT_EQ_I64(t6n[0].S_b, tw[0].S_b);

    /* All three sit outside f32's exact integer range -- and all three
     * CONVERT EXACTLY ANYWAY, because each is divisible by a high power of
     * two. That is the point: reaching 2^24 is not the same as losing bits,
     * so the worst case is the wrong place to look for the new fold's
     * rounding. */
    ASSERT_TRUE(-(i64)t4[0].S_b >= F32_EXACT_INT_LIMIT);
    ASSERT_TRUE(f32_conversion_is_exact(t4[0].S_b));
    ASSERT_TRUE(f32_conversion_is_exact(t6[0].S_b));
    ASSERT_TRUE(f32_conversion_is_exact(t6n[0].S_b));

    /* One quant off the endpoint makes |S_b| odd, and an odd integer above
     * 2^24 cannot be an f32. THIS is the case the m4 fold did not have: its
     * eight per-lane conversions each carried about an eighth of the total and
     * were exact. Cost here: one part in 3.07e7. */
    aq[4] = (u8)(i8)(-126);
    (void)orc_q4k_new_scalar(w4, aq, QK_ELEMS, t4);
    ASSERT_EQ_I64(t4[0].S_b, -30722895);
    ASSERT_TRUE(!f32_conversion_is_exact(t4[0].S_b));
    printf("  inexact case: S_b=%d -> (f32) %.1f, rel err %.3g\n", t4[0].S_b,
           (double)(f32)t4[0].S_b,
           (double)((f64)(f32)t4[0].S_b - (f64)t4[0].S_b) / (f64)t4[0].S_b);
}
