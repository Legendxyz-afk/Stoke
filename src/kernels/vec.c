/* src/kernels/vec.c -- dense f32 dot product + matvec (row-loop dispatch
 * over F32/Q8_0 tensors, the only quant-type dispatch point at M2). See
 * kernels.h for the full contract.
 */
#include "common.h"

#include "kernels.h"

f32 vec_dot_f32(const f32* a, const f32* b, u64 n) {
    f32 sum = 0.0f;
    for (u64 i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void matvec(const GgufTensor* w, const f32* x, u64 n_in, f32* y, u64 n_out) {
    if (w->n_dims != 2) {
        fatal("kernels: matvec -- tensor '%s' has n_dims=%u, want 2 -- "
              "matvec only supports 2D weight matrices",
              w->name, w->n_dims);
    }
    if (w->dims[0] != n_in) {
        fatal("kernels: matvec -- tensor '%s' dims[0]=%llu but caller "
              "passed n_in=%llu -- shape mismatch",
              w->name, (unsigned long long)w->dims[0],
              (unsigned long long)n_in);
    }
    if (w->dims[1] != n_out) {
        fatal("kernels: matvec -- tensor '%s' dims[1]=%llu but caller "
              "passed n_out=%llu -- shape mismatch",
              w->name, (unsigned long long)w->dims[1],
              (unsigned long long)n_out);
    }

    const u8* base = (const u8*)w->data;

    if (w->type == GGUF_GGML_TYPE_F32) {
        /* Row r starts at base + r*n_in*4 bytes. GGUF's data section
         * (and every general.alignment-padded tensor start within it) is
         * aligned to at least 4 bytes, and n_in*sizeof(f32) is itself a
         * multiple of 4, so every row start is a valid f32 alignment --
         * the direct typed cast below never reads a misaligned f32. */
        u64 row_bytes = n_in * (u64)sizeof(f32);
        for (u64 r = 0; r < n_out; r++) {
            const f32* row = (const f32*)(base + r * row_bytes);
            y[r] = vec_dot_f32(row, x, n_in);
        }
    } else if (w->type == KERNELS_GGML_TYPE_Q8_0) {
        /* n_in%KERNELS_Q8_0_BLOCK_ELEMS is NOT re-checked here:
         * vec_dot_q8_0_f32 enforces it on the very first row before any
         * row-stride arithmetic (below) is ever used to address memory,
         * so duplicating the check here would only be able to fire after
         * that one already did. */
        u64 row_bytes =
            (n_in / KERNELS_Q8_0_BLOCK_ELEMS) * KERNELS_Q8_0_BLOCK_BYTES;
        for (u64 r = 0; r < n_out; r++) {
            const void* row = base + r * row_bytes;
            y[r] = vec_dot_q8_0_f32(row, x, n_in);
        }
    } else {
        fatal("kernels: matvec -- tensor '%s' has unsupported ggml type "
              "%u -- only F32(%u) and Q8_0(%u) are dispatched at M2",
              w->name, w->type, (u32)GGUF_GGML_TYPE_F32,
              (u32)KERNELS_GGML_TYPE_Q8_0);
    }
}
