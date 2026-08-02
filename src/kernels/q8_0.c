/* src/kernels/q8_0.c -- Q8_0 dequant + dot kernels (see kernels.h for the
 * full contract). Block layout is probed verbatim (m2-probe-notes.md
 * P3.1): 34 bytes = { f16 d; i8 qs[32] }, value_i = f16_to_f32(d) * qs[i].
 * Block geometry constants (KERNELS_Q8_0_BLOCK_ELEMS/_BYTES) live in
 * kernels.h -- vec.c's matvec needs the same numbers for its row-stride
 * arithmetic, so they are shared rather than duplicated as private magic
 * numbers in each file.
 */
#include "common.h"

#include "kernels.h"

#include <math.h>

/* Little-endian u16 read from a raw byte pointer -- host-endianness
 * independent (mirrors gguf.c's rd_u16: manual byte assembly, not a
 * pointer-cast-and-deref, so this is also alignment-safe regardless of
 * where a block happens to land -- Q8_0's 34-byte blocks do not keep
 * every field naturally aligned to its type's size in general). */
static u16 read_u16_le(const u8* p) {
    return (u16)((u16)p[0] | ((u16)p[1] << 8));
}

/* Decodes block b's f16 scale and fatals if it is inf/nan (a sane GGUF
 * cannot contain one -- M2 op-spec §4). `who` names the calling kernel
 * for the fatal message. */
static f32 block_scale_or_fatal(const u8* blk, u64 b, const char* who) {
    f32 d = f16_to_f32(read_u16_le(blk));
    if (isinf(d) || isnan(d)) {
        fatal("kernels: %s -- block %llu scale decodes to inf/nan -- "
              "corrupt or hostile Q8_0 tensor (a sane GGUF cannot contain "
              "one)",
              who, (unsigned long long)b);
    }
    return d;
}

void dequant_row_q8_0(const void* row, u64 n, f32* out) {
    if (n % KERNELS_Q8_0_BLOCK_ELEMS != 0) {
        fatal("kernels: dequant_row_q8_0 -- n=%llu is not a multiple of "
              "%u -- Q8_0 rows are whole blocks of %u elements",
              (unsigned long long)n, KERNELS_Q8_0_BLOCK_ELEMS,
              KERNELS_Q8_0_BLOCK_ELEMS);
    }
    const u8* base = (const u8*)row;
    u64 num_blocks = n / KERNELS_Q8_0_BLOCK_ELEMS;

    for (u64 b = 0; b < num_blocks; b++) {
        const u8* blk = base + b * KERNELS_Q8_0_BLOCK_BYTES;
        f32 d = block_scale_or_fatal(blk, b, "dequant_row_q8_0");
        const u8* qs = blk + 2;
        f32* out_blk = out + b * KERNELS_Q8_0_BLOCK_ELEMS;
        for (u64 i = 0; i < KERNELS_Q8_0_BLOCK_ELEMS; i++) {
            i8 q = (i8)qs[i]; /* stored two's-complement signed byte; C11
                                  leaves out-of-range unsigned->signed as
                                  implementation-defined (not UB), and GCC
                                  documents it as reinterpreting the bit
                                  pattern (i.e. the wraparound this cast
                                  needs) */
            out_blk[i] = d * (f32)q;
        }
    }
}

/* Little-endian u16 write, mirroring read_u16_le above. */
static void write_u16_le(u8* p, u16 v) {
    p[0] = (u8)(v & 0xFFu);
    p[1] = (u8)(v >> 8);
}

void quantize_row_q8_0(const f32* x, u64 n, void* out) {
    if (n % KERNELS_Q8_0_BLOCK_ELEMS != 0) {
        fatal("kernels: quantize_row_q8_0 -- n=%llu is not a multiple of %u",
              (unsigned long long)n, KERNELS_Q8_0_BLOCK_ELEMS);
    }
    u8* base = (u8*)out;
    u64 num_blocks = n / KERNELS_Q8_0_BLOCK_ELEMS;

    for (u64 b = 0; b < num_blocks; b++) {
        const f32* xb = x + b * KERNELS_Q8_0_BLOCK_ELEMS;
        u8*        blk = base + b * KERNELS_Q8_0_BLOCK_BYTES;

        f32 amax = 0.0f;
        for (u64 i = 0; i < KERNELS_Q8_0_BLOCK_ELEMS; i++) {
            f32 a = fabsf(xb[i]);
            if (a > amax) {
                amax = a;
            }
        }

        /* Transcribed from the probed ggml quantize_row_q8_0_ref
         * (m3-probe-notes.md P-A). Three details are load-bearing and each
         * is a silent wrongness if changed:
         *   - `id` is computed from the f32 `d`, BEFORE d is narrowed to
         *     f16. Quantizing with the narrowed value instead shifts quants
         *     by +-1 near block boundaries.
         *   - the zero guard is `d ? 1/d : 0`, so an all-zero block yields
         *     all-zero quants rather than 0/0 = NaN.
         *   - rounding is roundf (half away from zero), NOT rint/nearbyint
         *     (half to even). This differs from f32_to_f16's tie rule on
         *     purpose: that one mirrors F16C hardware, this one mirrors
         *     ggml's quantizer, and they are separate contracts. */
        const f32 d = amax / 127.0f;
        const f32 id = (d != 0.0f) ? (1.0f / d) : 0.0f;

        write_u16_le(blk, f32_to_f16(d));
        u8* qs = blk + 2;
        for (u64 i = 0; i < KERNELS_Q8_0_BLOCK_ELEMS; i++) {
            f32 v = xb[i] * id;
            qs[i] = (u8)(i8)roundf(v);
        }
    }
}

f32 vec_dot_q8_0_q8_0(const void* wrow, const void* arow, u64 n) {
    if (n % KERNELS_Q8_0_BLOCK_ELEMS != 0) {
        fatal("kernels: vec_dot_q8_0_q8_0 -- n=%llu is not a multiple of %u",
              (unsigned long long)n, KERNELS_Q8_0_BLOCK_ELEMS);
    }
    const u8* wbase = (const u8*)wrow;
    const u8* abase = (const u8*)arow;
    u64 num_blocks = n / KERNELS_Q8_0_BLOCK_ELEMS;
    f32 sum = 0.0f;

    for (u64 b = 0; b < num_blocks; b++) {
        const u8* wblk = wbase + b * KERNELS_Q8_0_BLOCK_BYTES;
        const u8* ablk = abase + b * KERNELS_Q8_0_BLOCK_BYTES;
        f32 dw = block_scale_or_fatal(wblk, b, "vec_dot_q8_0_q8_0 (weight)");
        f32 da = block_scale_or_fatal(ablk, b, "vec_dot_q8_0_q8_0 (activation)");
        const u8* wq = wblk + 2;
        const u8* aq = ablk + 2;

        /* THE INTEGER CORE. Products accumulate in i32 with NO rounding
         * whatsoever: each |q| <= 127, so a product fits in 15 bits and 32
         * of them cannot exceed 32*127*127 = 516,128 -- three orders of
         * magnitude inside i32. This exactness is what the K3 gate depends
         * on: a SIMD twin computing the same integer sum must agree with
         * this bit for bit, which no f32 accumulation could promise.
         *
         * The f32 grouping then matches ggml's scalar reference exactly:
         *   sumf += sumi * (d_w * d_a)
         * The two scales multiply EACH OTHER first, and only then multiply
         * the integer sum. Folding them in any other order (e.g.
         * (sumi*d_w)*d_a) is a different f32 rounding and would put stoke
         * back outside the oracle's numeric class for no gain. */
        i32 sumi = 0;
        for (u64 i = 0; i < KERNELS_Q8_0_BLOCK_ELEMS; i++) {
            sumi += (i32)(i8)wq[i] * (i32)(i8)aq[i];
        }
        sum += (f32)sumi * (dw * da);
    }
    return sum;
}

f32 vec_dot_q8_0_f32(const void* row, const f32* x, u64 n) {
    if (n % KERNELS_Q8_0_BLOCK_ELEMS != 0) {
        fatal("kernels: vec_dot_q8_0_f32 -- n=%llu is not a multiple of "
              "%u -- Q8_0 rows are whole blocks of %u elements",
              (unsigned long long)n, KERNELS_Q8_0_BLOCK_ELEMS,
              KERNELS_Q8_0_BLOCK_ELEMS);
    }
    const u8* base = (const u8*)row;
    u64 num_blocks = n / KERNELS_Q8_0_BLOCK_ELEMS;
    f32 sum = 0.0f;

    for (u64 b = 0; b < num_blocks; b++) {
        const u8* blk = base + b * KERNELS_Q8_0_BLOCK_BYTES;
        f32 d = block_scale_or_fatal(blk, b, "vec_dot_q8_0_f32");
        const u8* qs = blk + 2;
        const f32* xb = x + b * KERNELS_Q8_0_BLOCK_ELEMS;

        /* Block's own sum accumulates FIRST (f32, sequential over the 32
         * elements), then gets scaled by d and folded into the running
         * total (f32, sequential across blocks) -- this exact grouping
         * is the binding contract (kernels.h), not an equivalent
         * reordering such as accumulating d*qs[i]*x[i] per element. */
        f32 block_sum = 0.0f;
        for (u64 i = 0; i < KERNELS_Q8_0_BLOCK_ELEMS; i++) {
            i8 q = (i8)qs[i];
            block_sum += (f32)q * xb[i];
        }
        sum += d * block_sum;
    }
    return sum;
}
