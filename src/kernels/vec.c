/* src/kernels/vec.c -- dense f32 dot product + matvec (row-loop dispatch
 * over F32/Q8_0 tensors, the only quant-type dispatch point at M2). See
 * kernels.h for the full contract.
 */
#include "common.h"

#include "kernels.h"
#include "simd.h"

f32 vec_dot_f32(const f32* a, const f32* b, u64 n) {
    f32 sum = 0.0f;
    for (u64 i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void matvec(const GgufTensor* w, const f32* x, u64 n_in, f32* y, u64 n_out,
            void* aq_scratch) {
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
    } else if (w->type == GGUF_GGML_TYPE_Q8_0) {
        /* n_in%KERNELS_Q8_0_BLOCK_ELEMS is NOT re-checked here: the dot
         * kernels enforce it on the very first row before any row-stride
         * arithmetic (below) is ever used to address memory, so duplicating
         * the check here could only fire after that one already did. */
        u64 row_bytes =
            (n_in / KERNELS_Q8_0_BLOCK_ELEMS) * KERNELS_Q8_0_BLOCK_BYTES;

        if (aq_scratch == NULL) {
            /* NO SILENT FALLBACK TO THE f32 PATH. The two paths are
             * different arithmetic classes -- that difference is precisely
             * what M3 exists to change and what the deferred oracle gate
             * measures -- so a missing scratch buffer must be a loud
             * programming error, never a quiet reversion to M2 numerics on
             * some call sites and not others. */
            fatal("kernels: matvec -- tensor '%s' is Q8_0 but no activation "
                  "scratch was supplied; the M3 integer core requires a "
                  "caller-owned buffer of (n_in/%u)*%u = %llu bytes",
                  w->name, KERNELS_Q8_0_BLOCK_ELEMS, KERNELS_Q8_0_BLOCK_BYTES,
                  (unsigned long long)row_bytes);
        }

        /* Quantize the activation ONCE for the whole matrix, not once per
         * row: x does not change across rows, and re-quantizing it n_out
         * times would cost 151,936 redundant passes on the logits matvec
         * alone. */
        quantize_row_q8_0(x, n_in, aq_scratch);

        /* K3 dispatch. The feature test is hoisted out of the row loop --
         * it is a cached load, but the logits matvec runs this loop 151,936
         * times and a branch on a global per row is pure noise in a profile.
         *
         * Choosing between them here is SAFE IN THE STRONGEST SENSE
         * available: the twin is bit-identical to the scalar kernel, proven
         * over 10,000 random tensors per leg (tests/test_simd.c), so which
         * branch runs cannot change a single generated token. That is the
         * whole return on LOCKED decision #2's integer core -- an f32 SIMD
         * path would have made this a numerics change requiring a parity
         * re-run, and instead it is a pure speed change. */
        if (kernels_cpu_has_avx_vnni()) {
            /* FOUR ROWS PER PASS. Measured 2026-08-03: the single-row twin
             * closed only 1.07x of the 1.997x gap to llama at one thread,
             * and the attribution ladder put the remainder outside the
             * kernel -- one row is one load stream feeding one serial f32
             * chain, which leaves the core waiting rather than computing.
             * Rows are independent dot products, so four at a time changes
             * no arithmetic and gate [1] is untouched. */
            u64 r = 0;
            for (; r + 4u <= n_out; r += 4u) {
                const void* rows[4] = {base + (r + 0u) * row_bytes,
                                       base + (r + 1u) * row_bytes,
                                       base + (r + 2u) * row_bytes,
                                       base + (r + 3u) * row_bytes};
                vec_dot_q8_0_q8_0_x4_avx2(rows, aq_scratch, n_in, &y[r]);
            }
            for (; r < n_out; r++) {
                y[r] = vec_dot_q8_0_q8_0_avx2(base + r * row_bytes,
                                              aq_scratch, n_in);
            }
        } else {
            for (u64 r = 0; r < n_out; r++) {
                y[r] = vec_dot_q8_0_q8_0(base + r * row_bytes, aq_scratch,
                                         n_in);
            }
        }
    } else if (w->type == GGUF_GGML_TYPE_Q4_K ||
               w->type == GGUF_GGML_TYPE_Q6_K) {
        /* K4. Same shape as the Q8_0 path above and for the same reasons,
         * but the activation quantizes to Q8_K rather than Q8_0 -- Q4_K's
         * per-sub-block MIN term is paid as `dmin * sum(activation)`, so the
         * dot needs the activation's per-16 sums, which only Q8_K carries.
         * Both K-quant types share that activation, so it is quantized once
         * here regardless of which branch runs below. */
        int is_q4 = (w->type == GGUF_GGML_TYPE_Q4_K);
        u32 blk_elems = is_q4 ? KERNELS_Q4_K_BLOCK_ELEMS
                              : KERNELS_Q6_K_BLOCK_ELEMS;
        u32 blk_bytes = is_q4 ? KERNELS_Q4_K_BLOCK_BYTES
                              : KERNELS_Q6_K_BLOCK_BYTES;
        u64 row_bytes = (n_in / blk_elems) * blk_bytes;

        if (aq_scratch == NULL) {
            /* Same rule as Q8_0: NO SILENT FALLBACK to a dequant-then-f32
             * path. That would be a second arithmetic class reachable from
             * one call site, invisible in the output and fatal to the parity
             * measurement the oracle gate exists to make. */
            fatal("kernels: matvec -- tensor '%s' is a K-quant but no "
                  "activation scratch was supplied; the integer core requires "
                  "a caller-owned buffer of (n_in/%u)*%u = %llu bytes",
                  w->name, KERNELS_Q8_K_BLOCK_ELEMS, KERNELS_Q8_K_BLOCK_BYTES,
                  (unsigned long long)((n_in / KERNELS_Q8_K_BLOCK_ELEMS)
                                       * KERNELS_Q8_K_BLOCK_BYTES));
        }

        /* Once for the whole matrix, not once per row -- the logits matvec
         * would otherwise re-quantize x 151,936 times. */
        quantize_row_q8_k(x, n_in, aq_scratch);

        if (is_q4) {
            for (u64 r = 0; r < n_out; r++) {
                y[r] = vec_dot_q4_k_q8_k(base + r * row_bytes, aq_scratch,
                                         n_in);
            }
        } else {
            for (u64 r = 0; r < n_out; r++) {
                y[r] = vec_dot_q6_k_q8_k(base + r * row_bytes, aq_scratch,
                                         n_in);
            }
        }
    } else {
        fatal("kernels: matvec -- tensor '%s' has unsupported ggml type "
              "%u -- only F32(%u), Q8_0(%u), Q4_K(%u) and Q6_K(%u) are "
              "dispatched",
              w->name, w->type, (u32)GGUF_GGML_TYPE_F32,
              (u32)GGUF_GGML_TYPE_Q8_0, (u32)GGUF_GGML_TYPE_Q4_K,
              (u32)GGUF_GGML_TYPE_Q6_K);
    }
}
