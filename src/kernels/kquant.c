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
