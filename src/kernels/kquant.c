/* src/kernels/kquant.c -- Q4_K / Q6_K dequant (see kernels.h for the full
 * contract). K4 owns this file exclusively, which is what lets it land
 * beside K3's SIMD work without either touching the other's sources.
 *
 * PROVENANCE, AND WHY IT IS NOT "TRANSCRIBED FROM THE C AND REVIEWED".
 * The 6-bit scale/min unpack below is the bug this milestone was warned
 * about by name. So the arithmetic here is pinned to a golden that traces
 * to llama.cpp's OWN dequant of the real model file (llama-quantize
 * --allow-requantize <Q4_K_M> <out> F32), not to anybody's reading of
 * ggml-quants.c: tests/data/k4_dequant.golden holds real block bytes
 * beside the f32 llama.cpp itself produced for them, and tests/test_kquant.c
 * requires this code to reproduce those values BIT FOR BIT.
 *
 * WHY BIT-EXACT RATHER THAN A TOLERANCE, measured rather than assumed: the
 * classic get_scale_min_k4 slip (taking the min's high bits from q[j-4]
 * instead of q[j]) moves only ~17% of a row's values and its largest error
 * is ~0.0138. It is SMALL and SPARSE, so any epsilon loose enough to be
 * called a tolerance accepts it. Dequantization is a fixed sequence of f32
 * operations with no reduction in it, so two correct implementations must
 * agree exactly -- exactness is the only criterion here that discriminates.
 *
 * ARITHMETIC GROUPING IS LOAD-BEARING. f32 multiply is not associative, so
 * `d1 * nibble - m1` (with d1 = d*sc, m1 = dmin*m) and Q6_K's left-to-right
 * `d * sc * q` are written the way ggml writes them. Matching costs nothing;
 * reasoning about which regroupings happen to be safe costs a gate.
 */
#include "common.h"

#include "kernels.h"

#include <math.h>
#include <string.h>

/* Little-endian u16 read. q8_0.c has its own copy of this and of the
 * inf/nan scale check; they are deliberately NOT shared. kernels.h's
 * contract keeps each kernel file self-contained so a milestone can own one
 * file outright, and exporting a 6-line helper to save six lines would widen
 * the kernel ABI to buy nothing. Both copies are alignment-safe by manual
 * byte assembly -- K-quant blocks are 144/210 bytes, so no field in them is
 * naturally aligned to its type at every block index. */
static u16 kq_read_u16_le(const u8* p) {
    return (u16)((u16)p[0] | ((u16)p[1] << 8));
}

/* Decodes an f16 super-block scale and fatals if it is inf/nan -- a sane
 * GGUF cannot contain one (M2 op-spec §4), and letting one through would
 * silently poison a whole 256-element super-block rather than 32. */
static f32 kq_scale_or_fatal(const u8* p, u64 b, const char* who,
                             const char* field) {
    f32 d = f16_to_f32(kq_read_u16_le(p));
    if (isinf(d) || isnan(d)) {
        fatal("kernels: %s -- block %llu %s decodes to inf/nan -- corrupt "
              "or hostile K-quant tensor (a sane GGUF cannot contain one)",
              who, (unsigned long long)b, field);
    }
    return d;
}

/* ggml's get_scale_min_k4, verbatim in structure.
 *
 * THE TRAP: in the j>=4 branch the MIN's high bits come from q[j] (ggml
 * writes it `q[j-0]`), while the SCALE's come from q[j-4]. The two lines are
 * near-mirror images, both indices are in range, and getting it wrong still
 * produces finite, plausible weights -- which is why this is pinned by a
 * golden rather than by review. */
static void get_scale_min_k4(int j, const u8* q, u8* d, u8* m) {
    if (j < 4) {
        *d = (u8)(q[j] & 63);
        *m = (u8)(q[j + 4] & 63);
    } else {
        *d = (u8)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (u8)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

void dequant_row_q4_k(const void* row, u64 n, f32* out) {
    if (n % KERNELS_Q4_K_BLOCK_ELEMS != 0) {
        fatal("kernels: dequant_row_q4_k -- n=%llu is not a multiple of "
              "%u -- Q4_K rows are whole super-blocks of %u elements",
              (unsigned long long)n, KERNELS_Q4_K_BLOCK_ELEMS,
              KERNELS_Q4_K_BLOCK_ELEMS);
    }
    const u8* base = (const u8*)row;
    u64 num_blocks = n / KERNELS_Q4_K_BLOCK_ELEMS;

    for (u64 b = 0; b < num_blocks; b++) {
        const u8* blk = base + b * KERNELS_Q4_K_BLOCK_BYTES;
        f32 d = kq_scale_or_fatal(blk, b, "dequant_row_q4_k", "d");
        f32 dmin = kq_scale_or_fatal(blk + 2, b, "dequant_row_q4_k", "dmin");
        const u8* scales = blk + 4;   /* 12 bytes of 6-bit scales+mins */
        const u8* q = blk + 16;       /* 128 bytes of 4-bit quants     */
        f32* y = out + b * KERNELS_Q4_K_BLOCK_ELEMS;

        int is = 0;
        for (u32 j = 0; j < KERNELS_Q4_K_BLOCK_ELEMS; j += 64) {
            u8 sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const f32 d1 = d * (f32)sc;
            const f32 m1 = dmin * (f32)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const f32 d2 = d * (f32)sc;
            const f32 m2 = dmin * (f32)m;
            /* Low nibbles of these 32 bytes, then the HIGH nibbles of the
             * SAME 32 bytes -- one 32-byte span feeds 64 outputs. */
            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * (f32)(q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * (f32)(q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

void dequant_row_q6_k(const void* row, u64 n, f32* out) {
    if (n % KERNELS_Q6_K_BLOCK_ELEMS != 0) {
        fatal("kernels: dequant_row_q6_k -- n=%llu is not a multiple of "
              "%u -- Q6_K rows are whole super-blocks of %u elements",
              (unsigned long long)n, KERNELS_Q6_K_BLOCK_ELEMS,
              KERNELS_Q6_K_BLOCK_ELEMS);
    }
    const u8* base = (const u8*)row;
    u64 num_blocks = n / KERNELS_Q6_K_BLOCK_ELEMS;

    for (u64 b = 0; b < num_blocks; b++) {
        const u8* blk = base + b * KERNELS_Q6_K_BLOCK_BYTES;
        const u8* ql = blk;             /* 128 B: low 4 bits          */
        const u8* qh = blk + 128;       /*  64 B: high 2 bits         */
        const i8* sc = (const i8*)(blk + 192); /* 16 B: i8 scales     */
        f32 d = kq_scale_or_fatal(blk + 208, b, "dequant_row_q6_k", "d");
        f32* y = out + b * KERNELS_Q6_K_BLOCK_ELEMS;

        for (u32 nn = 0; nn < KERNELS_Q6_K_BLOCK_ELEMS; nn += 128) {
            (void)nn;
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                /* The 6-bit value is 0..63, so the (i8) cast is the
                 * identity; the -32 is what makes it signed. */
                const i8 q1 = (i8)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const i8 q2 = (i8)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const i8 q3 = (i8)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const i8 q4 = (i8)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                /* Outputs are INTERLEAVED, not sequential: one l feeds four
                 * positions 32 apart, each with a different scale. */
                y[l + 0] = d * (f32)sc[is + 0] * (f32)q1;
                y[l + 32] = d * (f32)sc[is + 2] * (f32)q2;
                y[l + 64] = d * (f32)sc[is + 4] * (f32)q3;
                y[l + 96] = d * (f32)sc[is + 6] * (f32)q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* ===================================================================
 * The integer core: Q8_K activations + Q4_K/Q6_K integer dots.
 *
 * WHY Q8_K AND NOT Q8_0. K-quant weights are 256-element super-blocks with
 * per-sub-block scales AND, for Q4_K, per-sub-block MINS. The min term
 * contributes `dmin * sum(activation)` over each sub-block, so the dot needs
 * the activation's SUMS as well as its quants -- which is exactly what
 * block_q8_K carries and block_q8_0 does not. Q8_K is a different type, not
 * a widened Q8_0:
 *
 *     block_q8_K = { f32 d; i8 qs[256]; i16 bsums[16] } = 292 B
 *
 * This is the same move K2 made for Q8_0 and for the same reason: it puts
 * stoke in llama.cpp's numeric class rather than merely near it, which is
 * what the deferred oracle gate measures.
 * =================================================================== */

/* Little-endian u32 read, for the 12 packed scale bytes. */
static u32 kq_read_u32_le(const u8* p) {
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}

static void kq_write_f32_le(u8* p, f32 v) {
    u32 b;
    memcpy(&b, &v, sizeof(b));
    p[0] = (u8)(b & 0xFFu);
    p[1] = (u8)((b >> 8) & 0xFFu);
    p[2] = (u8)((b >> 16) & 0xFFu);
    p[3] = (u8)((b >> 24) & 0xFFu);
}

static f32 kq_read_f32_le(const u8* p) {
    u32 b = kq_read_u32_le(p);
    f32 v;
    memcpy(&v, &b, sizeof(v));
    return v;
}

static void kq_write_i16_le(u8* p, i16 v) {
    u16 u = (u16)v;
    p[0] = (u8)(u & 0xFFu);
    p[1] = (u8)(u >> 8);
}

static i16 kq_read_i16_le(const u8* p) {
    return (i16)(u16)((u16)p[0] | ((u16)p[1] << 8));
}

/* ggml's nearest_int, verbatim: add 1.5*2^23 so the mantissa's low bits ARE
 * the rounded integer, then subtract the bias. The FPU's own rounding does
 * the work, so this is round-to-nearest-EVEN -- which is NOT what roundf()
 * does (roundf is half-away-from-zero). Using roundf here would disagree
 * with ggml on every exact .5 and put us in a different numeric class on
 * those values, which is precisely what the integer core exists to avoid.
 *
 * ggml asserts |fval| <= 4194303. The only caller feeds `iscale*x[j]`, where
 * |iscale*x[j]| <= 127 by construction (iscale is -127/max and |x[j]| <=
 * |max|), so the precondition cannot be violated and is not re-checked in
 * this inner loop. */
static int kq_nearest_int(f32 fval) {
    f32 val = fval + 12582912.0f;
    int i;
    memcpy(&i, &val, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

void quantize_row_q8_k(const f32* x, u64 n, void* out) {
    if (n % KERNELS_Q8_K_BLOCK_ELEMS != 0) {
        fatal("kernels: quantize_row_q8_k -- n=%llu is not a multiple of %u",
              (unsigned long long)n, KERNELS_Q8_K_BLOCK_ELEMS);
    }
    u8* base = (u8*)out;
    u64 nb = n / KERNELS_Q8_K_BLOCK_ELEMS;

    for (u64 i = 0; i < nb; i++) {
        const f32* xb = x + i * KERNELS_Q8_K_BLOCK_ELEMS;
        u8* blk = base + i * KERNELS_Q8_K_BLOCK_BYTES;
        u8* qs = blk + 4;
        u8* bsums = blk + 4 + KERNELS_Q8_K_BLOCK_ELEMS;

        /* `max` is the SIGNED value at the largest magnitude, not the
         * magnitude itself -- iscale below is built from it, so its sign is
         * load-bearing and dropping it would flip every quant in the block. */
        f32 max = 0.0f, amax = 0.0f;
        for (u32 j = 0; j < KERNELS_Q8_K_BLOCK_ELEMS; ++j) {
            f32 ax = fabsf(xb[j]);
            if (ax > amax) { amax = ax; max = xb[j]; }
        }
        if (amax == 0.0f) {
            kq_write_f32_le(blk, 0.0f);
            memset(qs, 0, KERNELS_Q8_K_BLOCK_ELEMS);
            /* ggml `continue`s here and leaves bsums UNTOUCHED. We zero them
             * instead. That is output-identical -- the dot multiplies the
             * bsums term by dmin*ad, and ad is 0 here, so the term vanishes
             * whatever the bsums are -- but this scratch buffer is REUSED
             * across matvecs, so ggml's version would leave a previous row's
             * sums lying in it. Provably-dead stale data in a reused buffer
             * is still a hazard worth not having. */
            memset(bsums, 0, 2u * (KERNELS_Q8_K_BLOCK_ELEMS / 16u));
            continue;
        }
        const f32 iscale = -127.0f / max;
        for (u32 j = 0; j < KERNELS_Q8_K_BLOCK_ELEMS; ++j) {
            int v = kq_nearest_int(iscale * xb[j]);
            qs[j] = (u8)(i8)(v < 127 ? v : 127);
        }
        for (u32 j = 0; j < KERNELS_Q8_K_BLOCK_ELEMS / 16u; ++j) {
            int sum = 0;
            for (u32 ii = 0; ii < 16u; ++ii) {
                sum += (i8)qs[j * 16u + ii];
            }
            kq_write_i16_le(bsums + 2u * j, (i16)sum);
        }
        kq_write_f32_le(blk, 1.0f / iscale);
    }
}

/* Unpacks Q4_K's 12 packed bytes into 8 six-bit scales + 8 six-bit mins.
 *
 * This is ggml's utmp/kmask repacking from the DOT path -- a different
 * expression of the same thing get_scale_min_k4 computes one pair at a time
 * in the dequant path. Both are transcribed because both exist in ggml and
 * are used in different places; test_kquant.c asserts the two AGREE on real
 * blocks, so if either transcription is wrong the pair disagrees and says so.
 *
 * ggml does this by memcpy'ing into uint32_t[4] and aliasing the result as
 * bytes, which depends on little-endian layout. The bytes are extracted
 * explicitly here so the result does not depend on host byte order. */
static void kq_unpack_q4_k_scales(const u8* packed, u8* scales8, u8* mins8) {
    static const u32 kmask1 = 0x3f3f3f3fu;
    static const u32 kmask2 = 0x0f0f0f0fu;
    static const u32 kmask3 = 0x03030303u;
    u32 utmp[4];
    utmp[0] = kq_read_u32_le(packed + 0);
    utmp[1] = kq_read_u32_le(packed + 4);
    utmp[2] = kq_read_u32_le(packed + 8);

    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const u32 uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    for (u32 k = 0; k < 4u; k++) {
        scales8[k] = (u8)((utmp[0] >> (8u * k)) & 0xFFu);
        scales8[k + 4u] = (u8)((utmp[1] >> (8u * k)) & 0xFFu);
        mins8[k] = (u8)((utmp[2] >> (8u * k)) & 0xFFu);
        mins8[k + 4u] = (u8)((utmp[3] >> (8u * k)) & 0xFFu);
    }
}

/* Exposed wrappers over the two transcriptions -- see kernels.h. These are
 * the SAME code the kernels above call, not copies of it, so a test that
 * exercises them exercises what actually runs. */
void kernels_q4_k_scale_min_pair(const void* packed12, int j, u8* scale, u8* min) {
    if (j < 0 || j >= 8) {
        fatal("kernels: kernels_q4_k_scale_min_pair -- j=%d out of [0,8)", j);
    }
    get_scale_min_k4(j, (const u8*)packed12, scale, min);
}

void kernels_q4_k_unpack_scales(const void* packed12, u8* scales8, u8* mins8) {
    kq_unpack_q4_k_scales((const u8*)packed12, scales8, mins8);
}

f32 vec_dot_q4_k_q8_k(const void* wrow, const void* arow, u64 n) {
    if (n % KERNELS_Q4_K_BLOCK_ELEMS != 0) {
        fatal("kernels: vec_dot_q4_k_q8_k -- n=%llu is not a multiple of %u",
              (unsigned long long)n, KERNELS_Q4_K_BLOCK_ELEMS);
    }
    const u8* wb = (const u8*)wrow;
    const u8* ab = (const u8*)arow;
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
        f32 ad = kq_read_f32_le(ablk);

        memset(aux32, 0, sizeof(aux32));
        /* Widen the 4-bit quants to bytes first: low nibbles of a 32-byte
         * span, then high nibbles of the SAME span. */
        i8* a = aux8;
        for (u32 j = 0; j < KERNELS_Q4_K_BLOCK_ELEMS / 64u; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (i8)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (i8)(q4[l] >> 4);
            a += 32;
            q4 += 32;
        }

        kq_unpack_q4_k_scales(blk + 4, scales8, mins8);

        /* The MIN term rides on the activation's per-16 sums -- the whole
         * reason Q8_K exists. Note mins8[j/2]: one min per 32-element
         * sub-block, two bsums entries each. */
        i32 sumi = 0;
        for (u32 j = 0; j < KERNELS_Q8_K_BLOCK_ELEMS / 16u; ++j) {
            sumi += (i32)kq_read_i16_le(bsums + 2u * j) * (i32)mins8[j / 2u];
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

        /* ── CONTRACT v2 (ruling 2026-08-12 §3), and the eight lanes above are
         * now an IMPLEMENTATION DETAIL rather than the contract.
         *
         * S_b is this super-block's exact integer total. Summing the eight
         * accumulators is exact and freely regroupable BECAUSE IT IS INTEGER
         * -- that is the whole re-anchor: any implementation, at any SIMD
         * width, may reach S_b however it likes and must then execute the two
         * f32 operations below verbatim. Bit-exactness across widths follows
         * by construction instead of by pinning every twin to this file's
         * eight-lane partition, which was an ACCIDENT of transcribing ggml's
         * scalar and not a design value.
         *
         * Bound, computed not assumed: 256 elements x scale<=63 x q4<=15 x
         * |q8|<=127 = 30,723,840, three orders inside i32. */
        i32 S_b = 0;                                          /* CONTRACT */
        for (int l = 0; l < 8; ++l) S_b += aux32[l];          /* CONTRACT */
        const i32 M_b = sumi;                                 /* CONTRACT */

        const f32 d = f16_to_f32(kq_read_u16_le(blk)) * ad;
        const f32 dmin = f16_to_f32(kq_read_u16_le(blk + 2)) * ad;
        sumf = sumf + d * (f32)S_b;                           /* CONTRACT */
        sumf = sumf - dmin * (f32)M_b;                        /* CONTRACT */
    }
    return sumf;
}

f32 vec_dot_q6_k_q8_k(const void* wrow, const void* arow, u64 n) {
    if (n % KERNELS_Q6_K_BLOCK_ELEMS != 0) {
        fatal("kernels: vec_dot_q6_k_q8_k -- n=%llu is not a multiple of %u",
              (unsigned long long)n, KERNELS_Q6_K_BLOCK_ELEMS);
    }
    const u8* wb = (const u8*)wrow;
    const u8* ab = (const u8*)arow;
    u64 nb = n / KERNELS_Q6_K_BLOCK_ELEMS;

    i8 aux8[KERNELS_Q6_K_BLOCK_ELEMS];
    i16 aux16[8];
    i32 aux32[8];
    /* The old fold's `f32 sums[8]` lived here. CONTRACT v2 stopped reading it
     * in bf13034 but left the declaration and its memset behind, and the
     * memset is exactly why -Wunused-variable never fired. Removed. */

    f32 sumf = 0.0f;
    for (u64 i = 0; i < nb; ++i) {
        const u8* blk = wb + i * KERNELS_Q6_K_BLOCK_BYTES;
        const u8* q4 = blk;
        const u8* qh = blk + 128;
        const i8* sc = (const i8*)(blk + 192);
        const u8* ablk = ab + i * KERNELS_Q8_K_BLOCK_BYTES;
        const i8* q8 = (const i8*)(ablk + 4);
        f32 ad = kq_read_f32_le(ablk);

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

        /* Q6_K has NO min term -- its quants are already centred by the -32
         * above -- so unlike Q4_K it never touches the activation's bsums. */
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

        /* ── CONTRACT v2, the Q6_K shape: no mins term, so the canonical
         * fold is a single f32 operation per super-block against the exact
         * integer total. Same re-anchor as Q4_K -- the eight lanes are an
         * implementation detail, S_b is the contract.
         *
         * Bound, computed: 256 elements x |scale|<=127 x |q6|<=32 x |q8|<=127
         * = 132,120,576, still an order inside i32. */
        i32 S_b = 0;                                          /* CONTRACT */
        for (int l = 0; l < 8; ++l) S_b += aux32[l];          /* CONTRACT */
        const f32 d = f16_to_f32(kq_read_u16_le(blk + 208)) * ad;
        sumf = sumf + d * (f32)S_b;                           /* CONTRACT */
    }
    return sumf;
}
