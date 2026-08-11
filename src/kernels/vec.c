/* src/kernels/vec.c -- dense f32 dot product + matvec (row-loop dispatch
 * over F32/Q8_0 tensors, the only quant-type dispatch point at M2). See
 * kernels.h for the full contract.
 */
#include "common.h"

#include "kernels.h"
#include "pool.h"
#include "simd.h"

f32 vec_dot_f32(const f32* a, const f32* b, u64 n) {
    f32 sum = 0.0f;
    for (u64 i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* ---------------------------------------------------------- K5: row workers
 *
 * Every worker below computes y[r] for r in [begin, end) and touches NOTHING
 * else -- no shared accumulator, no neighbouring output. That is what makes
 * the pool safe here without any locking in the hot path, and it is also why
 * the result is bit-identical at any thread count: each y[r] runs exactly
 * the same sequential kernel over exactly the same bytes whichever thread
 * happens to claim its chunk. The parallelism is over a dimension that
 * carries no arithmetic dependency, so it costs no gate.
 *
 * The activation quantization is done ONCE, before the parallel region, by
 * the calling thread -- it is shared read-only input to every row. */
typedef struct {
    const u8* base;
    u64 row_bytes;
    const f32* x;      /* dense activation (F32 path)      */
    const void* aq;    /* quantized activation (all others) */
    u64 n_in;
    f32* y;
} RowJob;

/* ▶ THE BATCHED JOB. Identical to RowJob except x and y are ARRAYS indexed
 * by sequence. The row loop is unchanged and the sequence loop goes INSIDE
 * it, which is the whole of "seqs-inner/weights-outer": the weight row is
 * read once and applied to every sequence.
 *
 * Bit-exactness falls out and the reason is worth being precise about: the
 * accumulation order INSIDE each dot product is untouched, and nothing sums
 * across sequences. y[s][r] is computed by the same kernel over the same
 * bytes in the same order as a solo matvec would compute it. That is the
 * same argument row-parallelism already rests on. */
typedef struct {
    const u8* base;
    u64 row_bytes;
    const f32* const* x;
    u64 n_in;
    f32* const* y;
    u32 n_seq;
    void* const* aq;                     /* quant paths: one per sequence  */
    void (*row_fn)(void*, u64, u64);     /* the validated 1-seq row worker */
} BatchRowJob;

/* ▶ QUANT PATHS FUSE BY REUSING THE VALIDATED ROW WORKER, NOT BY REWRITING
 * IT. rows_q8_0_avx2 processes FOUR rows per pass, and gate [1] holds it
 * bit-identical to the scalar kernel; a hand-written batched twin would have
 * to reproduce that grouping exactly, which is the 2026-08-03 trap (an
 * 8-lane f32 accumulator differed from per-block grouping on 4 of 4 real
 * rows). So instead the worker is called unchanged, once per (row chunk,
 * sequence) -- each call computes exactly what a solo matvec would compute
 * for those rows, so bit-exactness is structural rather than re-argued.
 *
 * The CHUNK is what buys the amortization. Looping sequences around the
 * worker's whole row range would re-read every row per sequence, and a
 * thread's range (n_out/threads rows, ~25k on the logits matvec) does not
 * fit in cache. A chunk does: its rows are read once from memory and then
 * hit warm for every remaining sequence. Multiple of 4 so the twin's
 * four-row stepping is never split across a chunk boundary. */
#define BATCH_ROW_CHUNK 64u

static void rows_delegate_batch(void* vctx, u64 begin, u64 end) {
    const BatchRowJob* j = (const BatchRowJob*)vctx;
    for (u64 r0 = begin; r0 < end; r0 += BATCH_ROW_CHUNK) {
        u64 r1 = r0 + BATCH_ROW_CHUNK;
        if (r1 > end) r1 = end;
        for (u32 s = 0; s < j->n_seq; s++) {
            RowJob rj = {j->base, j->row_bytes,
                         j->x ? j->x[s] : NULL,
                         j->aq ? j->aq[s] : NULL,
                         j->n_in, j->y[s]};
            j->row_fn(&rj, r0, r1);
        }
    }
}

static void rows_f32_batch(void* vctx, u64 begin, u64 end) {
    const BatchRowJob* j = (const BatchRowJob*)vctx;
    for (u64 r = begin; r < end; r++) {
        const f32* row = (const f32*)(const void*)(j->base + r * j->row_bytes);
        for (u32 s = 0; s < j->n_seq; s++) {
            j->y[s][r] = vec_dot_f32(row, j->x[s], j->n_in);
        }
    }
}

static void rows_f32(void* vctx, u64 begin, u64 end) {
    const RowJob* j = (const RowJob*)vctx;
    for (u64 r = begin; r < end; r++) {
        const f32* row = (const f32*)(const void*)(j->base + r * j->row_bytes);
        j->y[r] = vec_dot_f32(row, j->x, j->n_in);
    }
}

static void rows_q8_0_scalar(void* vctx, u64 begin, u64 end) {
    const RowJob* j = (const RowJob*)vctx;
    for (u64 r = begin; r < end; r++) {
        j->y[r] = vec_dot_q8_0_q8_0(j->base + r * j->row_bytes, j->aq,
                                    j->n_in);
    }
}

static void rows_q8_0_avx2(void* vctx, u64 begin, u64 end) {
    const RowJob* j = (const RowJob*)vctx;
    u64 r = begin;
    /* Four rows per pass within this slice; the tail is handled here rather
     * than left to another chunk, since chunk bounds are row bounds. */
    for (; r + 4u <= end; r += 4u) {
        const void* rows[4] = {j->base + (r + 0u) * j->row_bytes,
                               j->base + (r + 1u) * j->row_bytes,
                               j->base + (r + 2u) * j->row_bytes,
                               j->base + (r + 3u) * j->row_bytes};
        vec_dot_q8_0_q8_0_x4_avx2(rows, j->aq, j->n_in, &j->y[r]);
    }
    for (; r < end; r++) {
        j->y[r] = vec_dot_q8_0_q8_0_avx2(j->base + r * j->row_bytes, j->aq,
                                         j->n_in);
    }
}

static void rows_q4_k_scalar(void* vctx, u64 begin, u64 end) {
    const RowJob* j = (const RowJob*)vctx;
    for (u64 r = begin; r < end; r++) {
        j->y[r] = vec_dot_q4_k_q8_k(j->base + r * j->row_bytes, j->aq,
                                    j->n_in);
    }
}

static void rows_q4_k_avx2(void* vctx, u64 begin, u64 end) {
    const RowJob* j = (const RowJob*)vctx;
    for (u64 r = begin; r < end; r++) {
        j->y[r] = vec_dot_q4_k_q8_k_avx2(j->base + r * j->row_bytes, j->aq,
                                         j->n_in);
    }
}

static void rows_q6_k_scalar(void* vctx, u64 begin, u64 end) {
    const RowJob* j = (const RowJob*)vctx;
    for (u64 r = begin; r < end; r++) {
        j->y[r] = vec_dot_q6_k_q8_k(j->base + r * j->row_bytes, j->aq,
                                    j->n_in);
    }
}

static void rows_q6_k_avx2(void* vctx, u64 begin, u64 end) {
    const RowJob* j = (const RowJob*)vctx;
    for (u64 r = begin; r < end; r++) {
        j->y[r] = vec_dot_q6_k_q8_k_avx2(j->base + r * j->row_bytes, j->aq,
                                         j->n_in);
    }
}

/* Applies `w` to n_seq activations at once. For each s the result is
 * BYTE-IDENTICAL to matvec(w, x[s], ...) called alone -- that is the
 * contract, it is what ROADMAP section M4's batch-composition independence
 * requires, and tests/test_matvec_batch.c checks it rather than assuming it.
 *
 * ▶ THE F32 PATH IS FUSED; THE QUANT PATHS STILL DELEGATE, ONE CALL PER
 * SEQUENCE. That is deliberate and is not a stub: each quant path has a SIMD
 * twin under gate [1]'s bit-exactness requirement, and a batched twin that
 * changes accumulation grouping fails it -- the 2026-08-03 trap where an
 * 8-lane f32 accumulator differed from per-block grouping on 4 of 4 real
 * rows. Fusing them is worth doing one path at a time, against the test that
 * already exists. Delegation is exactly correct meanwhile; it is only slower
 * than it could be. */
void matvec_batch(const GgufTensor* w, const f32* const* x, u64 n_in,
                  f32* const* y, u64 n_out, void* const* aq_scratch,
                  u32 n_seq) {
    if (n_seq == 0) return;

    if (w->type == GGUF_GGML_TYPE_F32) {
        /* The fused path never enters matvec, so it must repeat matvec's
         * shape checks itself -- a wrong n_in here would silently address
         * memory with the wrong row stride rather than fatal. */
        if (w->n_dims != 2) {
            fatal("kernels: matvec_batch -- tensor '%s' has n_dims=%u, want 2",
                  w->name, w->n_dims);
        }
        if (w->dims[0] != n_in) {
            fatal("kernels: matvec_batch -- tensor '%s' dims[0]=%llu but "
                  "caller passed n_in=%llu -- shape mismatch",
                  w->name, (unsigned long long)w->dims[0],
                  (unsigned long long)n_in);
        }
        if (w->dims[1] != n_out) {
            fatal("kernels: matvec_batch -- tensor '%s' dims[1]=%llu but "
                  "caller passed n_out=%llu -- shape mismatch",
                  w->name, (unsigned long long)w->dims[1],
                  (unsigned long long)n_out);
        }
        BatchRowJob j = {(const u8*)w->data, n_in * (u64)sizeof(f32), x, n_in,
                         y, n_seq, NULL, NULL};
        pool_for_rows(n_out, rows_f32_batch, &j);
        return;
    }

    if (w->type == GGUF_GGML_TYPE_Q8_0) {
        u64 row_bytes =
            (n_in / KERNELS_Q8_0_BLOCK_ELEMS) * KERNELS_Q8_0_BLOCK_BYTES;
        for (u32 s = 0; s < n_seq; s++) {
            if (aq_scratch == NULL || aq_scratch[s] == NULL) {
                /* Same refusal as matvec, for the same reason: no silent
                 * fallback to the f32 path, because the two are different
                 * arithmetic classes. */
                fatal("kernels: matvec_batch -- tensor '%s' is Q8_0 but "
                      "sequence %u supplied no activation scratch "
                      "((n_in/%u)*%u = %llu bytes required)",
                      w->name, s, KERNELS_Q8_0_BLOCK_ELEMS,
                      KERNELS_Q8_0_BLOCK_BYTES,
                      (unsigned long long)row_bytes);
            }
            /* Once per sequence, outside the parallel region -- x differs
             * per sequence but not across rows, exactly as in matvec. */
            quantize_row_q8_0(x[s], n_in, aq_scratch[s]);
        }
        BatchRowJob j = {(const u8*)w->data, row_bytes, NULL, n_in, y, n_seq,
                         aq_scratch,
                         kernels_cpu_has_avx_vnni() ? rows_q8_0_avx2
                                                    : rows_q8_0_scalar};
        pool_for_rows(n_out, rows_delegate_batch, &j);
        return;
    }

    for (u32 s = 0; s < n_seq; s++) {
        matvec(w, x[s], n_in, y[s], n_out, aq_scratch ? aq_scratch[s] : NULL);
    }
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
        RowJob j = {base, n_in * (u64)sizeof(f32), x, NULL, n_in, y};
        pool_for_rows(n_out, rows_f32, &j);
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
        /* FOUR ROWS PER PASS in the twin. Measured 2026-08-03: the
         * single-row twin closed only 1.07x of the 1.997x gap to llama at
         * one thread, and the attribution ladder put the remainder outside
         * the kernel -- one row is one load stream feeding one serial f32
         * chain, which leaves the core waiting rather than computing. Rows
         * are independent dot products, so four at a time changes no
         * arithmetic and gate [1] is untouched -- and the same independence
         * is what lets K5 spread them over threads. */
        RowJob j = {base, row_bytes, x, aq_scratch, n_in, y};
        pool_for_rows(n_out,
                      kernels_cpu_has_avx_vnni() ? rows_q8_0_avx2
                                                 : rows_q8_0_scalar,
                      &j);
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

        /* K4 dispatch, hoisted out of the row loop for the same reason the
         * Q8_0 one is. Safe in the strongest sense available: the twins are
         * bit-identical to the scalar kernels over 1e4 random tensors each
         * (tests/test_simd.c), so which branch runs cannot change a token. */
        int simd = kernels_cpu_has_avx_vnni();
        void (*rows)(void*, u64, u64) =
            is_q4 ? (simd ? rows_q4_k_avx2 : rows_q4_k_scalar)
                  : (simd ? rows_q6_k_avx2 : rows_q6_k_scalar);
        RowJob j = {base, row_bytes, x, aq_scratch, n_in, y};
        pool_for_rows(n_out, rows, &j);
    } else {
        fatal("kernels: matvec -- tensor '%s' has unsupported ggml type "
              "%u -- only F32(%u), Q8_0(%u), Q4_K(%u) and Q6_K(%u) are "
              "dispatched",
              w->name, w->type, (u32)GGUF_GGML_TYPE_F32,
              (u32)GGUF_GGML_TYPE_Q8_0, (u32)GGUF_GGML_TYPE_Q4_K,
              (u32)GGUF_GGML_TYPE_Q6_K);
    }
}
