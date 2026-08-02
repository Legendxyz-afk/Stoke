/* src/kernels/f16.c -- IEEE 754 binary16 -> f32 (see kernels.h for the full
 * contract). Standard widen-with-renormalization algorithm: normals and
 * zero/inf/nan just re-bias the exponent and shift the mantissa into f32's
 * wider fields; subnormals (f16 exponent field 0, nonzero mantissa) are
 * shifted left until the implicit leading bit appears, adjusting the
 * output exponent to match -- every f16 value is exactly representable in
 * f32, so no rounding ever actually occurs here, only bit repositioning.
 */
#include "common.h"

#include "kernels.h"

#include <string.h>

f32 f16_to_f32(u16 h) {
    u32 sign = (u32)(h & 0x8000u) << 16;
    u32 exp  = (u32)(h >> 10) & 0x1Fu;
    u32 mant = (u32)h & 0x3FFu;
    u32 bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign; /* +0 or -0 */
        } else {
            /* Subnormal: shift left until the implicit leading 1 lands in
             * bit 10, counting shifts so the f32 exponent can be derived
             * (each shift halves the represented value's exponent
             * contribution by one, relative to the smallest normal). */
            i32 shift = -1;
            u32 m = mant;
            do {
                shift++;
                m <<= 1;
            } while ((m & 0x400u) == 0);
            m &= 0x3FFu; /* drop the now-implicit leading bit */
            u32 f32_exp = (u32)(127 - 15 - shift);
            bits = sign | (f32_exp << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        /* inf (mant==0) or nan (mant!=0); f16's 10-bit mantissa widens
         * into f32's 23-bit field in the same top-aligned position as the
         * normal case below, preserving the quiet/signaling bit and any
         * payload. */
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        /* Normal: f16 bias 15 -> f32 bias 127, mantissa left-aligned into
         * the wider field. */
        u32 f32_exp = exp - 15u + 127u;
        bits = sign | (f32_exp << 23) | (mant << 13);
    }

    f32 result;
    memcpy(&result, &bits, sizeof result); /* well-defined reinterpretation,
                                               no aliasing/alignment UB --
                                               mirrors gguf.c's rd_f32 */
    return result;
}

/* f32 -> f16, round-to-nearest-EVEN (see kernels.h).
 *
 * Needed by M3's activation quantization: ggml stores a Q8_0 block's scale
 * as f16 (`y[i].d = GGML_FP32_TO_FP16(d)`), and stoke must store it the same
 * way or its dot products land in a different numeric class from the
 * oracle's -- which is the entire point of the integer core.
 *
 * Unlike f16_to_f32 this one genuinely ROUNDS: f32 has 23 mantissa bits and
 * f16 has 10, so 13 bits are discarded on every normal value. Ties go to
 * even, matching IEEE 754's default and what hardware F16C (_mm256_cvtps_ph
 * with _MM_FROUND_TO_NEAREST_INT) does -- K3 will substitute that
 * instruction for this function, so a different tie rule here would make the
 * scalar and SIMD paths disagree on exactly the values that fall on a tie.
 */
u16 f32_to_f16(f32 f) {
    u32 bits;
    memcpy(&bits, &f, sizeof bits);

    u32 sign = (bits >> 16) & 0x8000u;
    u32 raw_exp = (bits >> 23) & 0xFFu;
    u32 mant = bits & 0x7FFFFFu;

    if (raw_exp == 0xFFu) {
        if (mant == 0) {
            return (u16)(sign | 0x7C00u); /* +-inf */
        }
        /* NaN: keep the top payload bits, but never let a nonzero payload
         * truncate to zero -- that would silently turn a NaN into an inf. */
        u16 m = (u16)(mant >> 13);
        if (m == 0) {
            m = 1;
        }
        return (u16)(sign | 0x7C00u | m);
    }

    /* Rebias into f16's exponent range. Signed: <=0 means subnormal or
     * underflow, >=0x1F means overflow. */
    i32 exp = (i32)raw_exp - 127 + 15;

    if (exp >= 0x1F) {
        return (u16)(sign | 0x7C00u); /* overflows f16's range -> inf */
    }

    if (exp <= 0) {
        /* Subnormal output. The f32 value is (1.mant) * 2^(raw_exp-127); an
         * f16 subnormal is m * 2^-24, so m = (mant|implicit) >> (14-exp).
         * Below -10 even the rounding cannot reach 1, so it is a true
         * underflow to a signed zero. */
        if (exp < -10) {
            return (u16)sign;
        }
        mant |= 0x800000u; /* restore the implicit leading 1 */
        u32 shift = (u32)(14 - exp);   /* 24 at most, so 1u<<(shift-1) is
                                           always well-defined */
        u32 half = 1u << (shift - 1);
        u32 lower = mant & ((1u << shift) - 1u);
        u16 m = (u16)(mant >> shift);
        if (lower > half || (lower == half && (m & 1u) != 0u)) {
            m++; /* may carry into the exponent field, which is CORRECT:
                    0x400 is the smallest normal and its bit pattern is
                    exactly exponent 1, mantissa 0 */
        }
        return (u16)(sign | m);
    }

    /* Normal output: drop 13 mantissa bits, round to nearest even. */
    u16 m = (u16)(mant >> 13);
    u32 lower = mant & 0x1FFFu;
    if (lower > 0x1000u || (lower == 0x1000u && (m & 1u) != 0u)) {
        m++;
        if (m == 0x400u) {
            /* Mantissa carried out: bump the exponent and reset, which can
             * itself overflow to inf. */
            m = 0;
            exp++;
            if (exp >= 0x1F) {
                return (u16)(sign | 0x7C00u);
            }
        }
    }
    return (u16)(sign | ((u16)exp << 10) | m);
}
