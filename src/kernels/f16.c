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
