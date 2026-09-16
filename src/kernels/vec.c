/* src/kernels/vec.c -- dense f32 dot product + matvec (row-loop dispatch
 * over F32/Q8_0 tensors, the only quant-type dispatch point at M2). See
 * kernels.h for the full contract.
 */
#include "common.h"

#include "kernels.h"
#include "pool.h"
#include "simd.h"

#include <stdatomic.h> /* the once-only pad trace flag, cleared by whichever worker pads first */
#include <stdlib.h>    /* getenv -- STOKE_Q6K_KERNEL, STOKE_Q4K_KERNEL */
#include <string.h>    /* strcmp -- ditto */

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

/* The FUSED K-quant path's own row chunk, separate from the delegating one
 * because the two nests hold different things in cache: the delegate reuses a
 * chunk's ROWS across sequences, the fused path reuses a TILE's ACTIVATIONS
 * across rows while the chunk's weights stay warm. Overridable so the value
 * can be swept end-to-end rather than argued -- see the M7 prefill work. */
#ifndef KQUANT_ROW_CHUNK
#define KQUANT_ROW_CHUNK 64u
#endif

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

/* ▶ M4.5 RUNG B (ruling §7.1): the batched Q4_K worker. Same rows, same
 * sequences, same arithmetic -- the two innermost loops are inverted so the
 * row's weight-only unpack happens ONCE rather than once per sequence.
 *
 * This is the ONLY worker that is not `row_fn` called N times, so it hangs
 * off `row_fn_batch` and `rows_delegate_batch` keeps its structure for every
 * type that has not been converted yet. A type with no batched worker is not
 * broken, only unfused -- which is exactly the state every type was in
 * before this commit. */
/* ▶ THE PARTIAL GROUP IS PADDED TO A WHOLE ONE (2026-09-16). Every grouped
 * kernel below takes its sequences FOUR at a time and hands the remainder --
 * `n_seq % 4` of them -- to a per-sequence tail that runs the same integer dot
 * through the shared block dot and reduces ONE sequence where the group
 * reduces four. The tail is where the serving path spends more time than it
 * looks: the step trace of 2026-09-15 read the N=4 -> N=5 decode step at
 * +44 ms for ONE row against +17 ms per row inside a group, and a prefill
 * chunk's riding decodes (128 + 3, 256 + 3) are ALWAYS a partial group. So a
 * tile whose remainder qualifies is padded to the next multiple of the group
 * with copies of its LAST pointer, the kernel computes the padded sequences,
 * and their results are dropped.
 *
 * ⚠ BIT-EXACT BY THE PROPERTY test_matvec_batch ALREADY GRADES: a sequence's
 * result depends on its own bytes and the row's, never on which sequences
 * share the group (separate accumulators, separate lanes of the reduction,
 * its own scale in the fold). A padded copy is just another sequence whose
 * output nobody reads; the copies are written into a padded `out` and only
 * the real `cnt` are stored. `bench --matvec-batch --pad-ab` asserts the
 * padded and unpadded arms agree on every sequence before it times them.
 *
 * ▶ WHICH REMAINDERS, MEASURED NOT MODELLED (`bench --matvec-batch --pad-ab`,
 * one process, three interleaved arms, 32 cycles x 3 processes, pinned; the
 * null arm read 1.00 +- 0.02 in every cell; padded-over-unpadded time):
 *
 *   remainder  Q4_K attn_q / ffn_gate            Q6_K ffn_down (maddubs)
 *   3 alone    0.90-0.92 (N=3)                   0.92-0.95
 *   3 + group  0.76-0.81 (N=7)                   0.91-0.95
 *   2 alone    1.04-1.07 (N=2) -- SLOWER         1.09-1.11 -- SLOWER
 *   2 + group  0.90-0.93 (N=6)                   1.02-1.05 -- SLOWER
 *   1          not padded; forced on, 1.20 (N=5) -- SLOWER, as the uop model said
 *
 * So the default policy is per kernel: Q4_K pads a remainder of three always
 * and a remainder of two only when the tile also holds a full group (N=2 is
 * the one shape where four grouped sequences cost more than two on the tail --
 * left as a measured fact, not explained); Q6_K pads three only. A remainder
 * of one is never padded by default: one grouped call is four sequences' work
 * against one tail sequence's. The runtime knob is for the A/B --
 * STOKE_KQ_PAD_TAIL=0 never pads, =1/2/3 force that threshold on BOTH kernels
 * with no qualifier, unset = the policy above; read once, an unrecognised
 * value REFUSED like the kernel selectors'. The ungrouped Q6_K reference xn
 * kernel passes group=1 and is never padded: there is nothing to pad INTO. */
#define KQ_PAD_AUTO       0xFFFFFFFEu   /* the measured per-kernel policy */
#define KQ_PAD_UNRESOLVED 0xFFFFFFFFu
static u32 g_pad_tail_min = KQ_PAD_UNRESOLVED;
/* STOKE_KQ_PAD_TRACE=1: announce the FIRST padded group on stderr. The padded
 * and unpadded arms are bit-identical by design, so a measurement's log needs
 * this line to prove the arm it names was the arm that ran -- the same
 * non-vacuity rule the speed mutants follow. Off: one predictable branch per
 * padded tile. */
static atomic_int g_pad_trace = 0;

static void pad_select(void) {
    if (g_pad_tail_min != KQ_PAD_UNRESOLVED) return;
    const char* t = getenv("STOKE_KQ_PAD_TRACE");
    atomic_store_explicit(&g_pad_trace, (t != NULL && t[0] == '1') ? 1 : 0,
                          memory_order_relaxed);
    const char* e = getenv("STOKE_KQ_PAD_TAIL");
    if (e == NULL || e[0] == '\0') {
        g_pad_tail_min = KQ_PAD_AUTO;
    } else if (e[1] == '\0' && e[0] >= '0' && e[0] <= '3') {
        g_pad_tail_min = (u32)(e[0] - '0');
    } else {
        fatal("kernels: STOKE_KQ_PAD_TAIL=%s -- want 0 (never pad a partial "
              "group), 1, 2 or 3 (pad a remainder of at least that many); an "
              "unrecognised value is refused rather than ignored, because a "
              "typo that silently selected the default would make an A/B "
              "measure one arm twice",
              e);
    }
}

/* BENCH/TEST HOOK: overrides the threshold for the rest of the process and
 * marks it resolved, so pad_select will not read the env over it. `bench
 * --matvec-batch --pad-ab` flips it between arms inside one process. Call it
 * from the dispatching thread only, never inside a parallel region. */
void kernels_set_pad_tail(u32 min_remainder) {
    if (min_remainder > 3u && min_remainder != KERNELS_PAD_TAIL_AUTO) {
        fatal("kernels: kernels_set_pad_tail(%u) -- want 0..3 or "
              "KERNELS_PAD_TAIL_AUTO", min_remainder);
    }
    g_pad_tail_min = (min_remainder == KERNELS_PAD_TAIL_AUTO) ? KQ_PAD_AUTO
                                                             : min_remainder;
}

/* `group` is the kernel's grouping (4, or 1 for an ungrouped kernel);
 * `auto_min` is the kernel's measured threshold under the default policy (2
 * for Q4_K, 3 for Q6_K) -- see the table above. */
static void rows_kquant_batch_avx2(const BatchRowJob* j, u64 begin, u64 end,
                                   void (*dot_xn)(const void*, u64, u32,
                                                  const void* const*,
                                                  u32, u64, f32*),
                                   u32 group, u32 auto_min) {
    const int auto_policy = (g_pad_tail_min == KQ_PAD_AUTO);
    const u32 thr = auto_policy ? auto_min : g_pad_tail_min;
    for (u64 r0 = begin; r0 < end; r0 += KQUANT_ROW_CHUNK) {
        u64 r1 = r0 + KQUANT_ROW_CHUNK;
        if (r1 > end) r1 = end;
        /* ▶▶▶ TILE OUTSIDE ROW, WHICH IS THE OPPOSITE OF WHAT THIS LOOP USED
         * TO SAY, AND THE REASON IS THE ACTIVATION REUSE DISTANCE.
         *
         * With the row loop outside, ONE row touches every sequence's
         * activation before the next row starts: n_seq * (n_in/256) * 292 B,
         * which on a 128-token prefill chunk at n_in=2560 is 374 KiB. Every
         * row then re-reads all of it, so the reuse distance is 374 KiB and
         * nothing is ever in L1. With the tile outside, a tile's 16 sequences
         * (46.7 KiB) are reused across the whole row chunk, and the chunk's
         * weights (KQUANT_ROW_CHUNK * row_bytes) are read from memory once and
         * hit warm for every remaining tile.
         *
         * ⚠ That is exactly the argument BATCH_ROW_CHUNK's own note makes for
         * `rows_delegate_batch` above, which has had the sequence loop outside
         * the row loop all along. The fused path was written the other way and
         * the note did not carry over.
         *
         * ⚠⚠ WHY IT IS WORTH DOING NOW AND WAS NOT ON 2026-08-30:
         * `-DSTOKE_MUT_ALIAS_ACT` -- which collapses the activation tile to one
         * sequence's -- measured 5.8% against the kernel of that day and
         * measures 1.29x against this one. Removing the per-sequence
         * bookkeeping promoted the tile from noise to the largest remaining
         * cost. An ablation's verdict is scoped to the version it ran on.
         *
         * BIT-EXACT BY CONSTRUCTION, and that is why this is a loop swap
         * rather than a kernel change: every `dot_xn` call is byte-for-byte
         * the call it was, on the same row and the same sequences, computing
         * one (row, sequence) result from the same super-blocks in the same
         * ascending order. Only the ORDER OF INDEPENDENT CALLS changes. */
        for (u32 s0 = 0; s0 < j->n_seq; s0 += KERNELS_BATCH_TILE) {
            u32 cnt = j->n_seq - s0;
            if (cnt > KERNELS_BATCH_TILE) cnt = KERNELS_BATCH_TILE;
            /* The padded call: `cntp` sequences through `aqs`, of which only
             * the first `cnt` are real. KERNELS_BATCH_TILE is a multiple of
             * every group, so cntp never exceeds the tile. */
            const void* const* aqs  = (const void* const*)&j->aq[s0];
            u32                cntp = cnt;
            const void*        aqpad[KERNELS_BATCH_TILE];
            u32                rem  = (group > 1u) ? cnt % group : 0u;
            int pad = (rem != 0u && thr != 0u && rem >= thr);
            /* The measured exception (table above): a remainder of two with
             * no full group beside it (the tile IS the remainder) is slower
             * padded. */
            if (pad && auto_policy && rem == 2u && cnt == rem) pad = 0;
            if (pad) {
                cntp = cnt + (group - rem);
                if (atomic_load_explicit(&g_pad_trace, memory_order_relaxed) &&
                    atomic_exchange_explicit(&g_pad_trace, 0,
                                             memory_order_relaxed)) {
                    /* Once per process, from whichever worker pads first;
                     * the exchange makes it exactly once without a lock. */
                    fprintf(stderr, "kernels: pad trace -- a partial group of "
                                    "%u padded to %u (n_seq %u, tile at %u, "
                                    "threshold %u%s)\n",
                            rem, cntp, j->n_seq, s0, thr,
                            auto_policy ? " auto" : " forced");
                }
                for (u32 k = 0; k < cntp; k++) {
                    aqpad[k] = j->aq[s0 + (k < cnt ? k : cnt - 1u)];
                }
                aqs = aqpad;
            }
            /* ▶ AND THE ROWS GO IN TILES TOO, because the swap above only got
             * the reuse distance down to a tile's whole-row activation (46.7
             * KiB at n_in=2560, still the size of L1). The kernel takes
             * KERNELS_ROW_TILE rows and runs the row loop INSIDE its
             * super-block loop, so what is live is ONE super-block's
             * activations, 4.7 KiB. See simd.h. */
            for (u64 r = r0; r < r1; r += KERNELS_ROW_TILE) {
                u32 nr = (u32)(r1 - r);
                if (nr > KERNELS_ROW_TILE) nr = KERNELS_ROW_TILE;
                /* Row-major, nr * cnt: the kernel's accumulators stay a small
                 * stack array and each (row, sequence) is exact on its own. */
                f32 out[KERNELS_ROW_TILE * KERNELS_BATCH_TILE];
                dot_xn(j->base + r * j->row_bytes, j->row_bytes, nr,
                       aqs, cntp, j->n_in, out);
                for (u32 rr = 0; rr < nr; rr++) {
                    for (u32 k = 0; k < cnt; k++) {
                        j->y[s0 + k][r + rr] = out[rr * cntp + k];
                    }
                }
            }
        }
    }
}

static int g_q4k_vnni = -1;

static void rows_q4_k_batch_avx2(void* vctx, u64 begin, u64 end) {
    rows_kquant_batch_avx2((const BatchRowJob*)vctx, begin, end,
                           /* `== 1`, not truthiness: the unresolved state is
                            * -1, and a reordering that reached here before
                            * q4k_select would otherwise silently pick the
                            * kernel that is NOT the compiled-in default. */
                           g_q4k_vnni == 1 ? vec_dot_q4_k_q8_k_xn_vnni_avx2
                                           : vec_dot_q4_k_q8_k_xn_avx2,
                           4u /* both Q4_K kernels group by four */,
                           2u /* pad a remainder of 3, and of 2 beside a group */);
}

/* ▶ WHICH Q6_K BATCHED KERNEL, AND WHY THE CHOICE IS A FUNCTION POINTER TAKEN
 * ONCE PER MATVEC RATHER THAN A BRANCH IN THE LOOP. `rows_kquant_batch_avx2`
 * already takes the kernel as a parameter, so selecting here costs one call per
 * matvec and nothing per row, per super-block or per sequence.
 *
 * The env var exists so both arms come from ONE binary in ONE sitting -- see
 * simd.h. Read once into a static, the same shape `kernels_cpu_has_avx_vnni`
 * already uses, and resolved on the DISPATCHING thread before `pool_for_rows`
 * hands work to anyone else. An unrecognised value is refused: a typo that
 * silently selected the default would make an A/B measure one kernel twice. */
static int g_q6k_maddubs = -1;

static void q6k_select(void) {
    if (g_q6k_maddubs >= 0) return;
#ifdef STOKE_MUT_Q6K
    /* A MUTANT ANNOUNCES ITSELF, and it announces itself the FIRST time the
     * mutated kernel could possibly be reached rather than at startup -- so a
     * log that contains this line is a log where the defect was live. */
    fprintf(stderr, "!! STOKE_MUT_Q6K BUILD -- the Q6_K maddubs kernel's scale "
                    "lanes are DELIBERATELY swapped. Never ship this binary.\n");
#endif
    const char* e = getenv("STOKE_Q6K_KERNEL");
    if (e == NULL || e[0] == '\0') {
        g_q6k_maddubs = STOKE_Q6K_MADDUBS ? 1 : 0;
    } else if (strcmp(e, "maddubs") == 0) {
        g_q6k_maddubs = 1;
    } else if (strcmp(e, "ref") == 0) {
        g_q6k_maddubs = 0;
    } else {
        fatal("kernels: STOKE_Q6K_KERNEL=%s -- want 'ref' or 'maddubs'; an "
              "unrecognised value is refused rather than ignored, because a "
              "typo that silently selected the default would make an A/B "
              "measure one kernel twice",
              e);
    }
}

/* ▶ WHICH Q4_K BATCHED KERNEL -- the same shape as q6k_select above and for the
 * same reasons: resolved on the DISPATCHING thread before any worker exists,
 * one function-pointer choice per matvec and none per row or per sequence, and
 * an unrecognised value REFUSED rather than ignored, because a typo that
 * silently selected the default would make an A/B measure one kernel twice.
 *
 * It also announces the SPEED PROBES, the first time the mutated kernel could
 * be reached. A build that silently dropped its `-D` is a CLEAN binary, and a
 * leg graded without this line would report "the probe changed nothing" about
 * a probe that was never there.
 *
 * ⚠⚠ AND IT FORCES THE ARM A PROBE LIVES IN. Every `STOKE_MUT_` case below
 * except `STOKE_MUT_Q4K_VNNI` is cut into the MADDUBS twin only; with vnni as
 * the shipping default, building one of them and running normally would
 * measure an UNMUTATED kernel and report that the probe changed nothing. That
 * is the same vacuous pass the `fatal` above exists to prevent -- it bit this
 * function once already (see simd.h's STOKE_MINS_SIMD note) -- so the arm is
 * pinned to the kernel the mutation is in rather than left to the default. */
static void q4k_select(void) {
    if (g_q4k_vnni < 0) {
        const char* e = getenv("STOKE_Q4K_KERNEL");
        /* PARSE FIRST, ALWAYS -- the refusal below is this function's
         * anti-vacuity check and must not be skippable by another knob. An
         * earlier spelling put the STOKE_MINS_SIMD case in front of it and
         * silently accepted every value, which is how the guard was found to
         * be firing in every build. */
        if (e == NULL || e[0] == '\0') {
            g_q4k_vnni = STOKE_Q4K_VNNI ? 1 : 0;
        } else if (strcmp(e, "vnni") == 0) {
            g_q4k_vnni = 1;
        } else if (strcmp(e, "maddubs") == 0) {
            g_q4k_vnni = 0;
        } else {
            fatal("kernels: STOKE_Q4K_KERNEL=%s -- want 'maddubs' or 'vnni'; "
                  "an unrecognised value is refused rather than ignored, "
                  "because a typo that silently selected the default would "
                  "make an A/B measure one kernel twice",
                  e);
        }
#if !STOKE_MINS_SIMD
        /* ⚠ THE TWO KNOBS INTERACT, SO SAY SO RATHER THAN LET THEM. Building
         * `-DSTOKE_MINS_SIMD=0` selects the maddubs twin's PRE-SIMD mins arm by
         * disabling its grouped path entirely; the VNNI kernel has only the
         * SIMD spelling, so leaving it selectable here would make that A/B
         * measure the same mins term in both arms -- the failure the `fatal`
         * above exists to prevent, arriving by a different door. REFUSED, not
         * downgraded: silently handing back the other kernel is the same
         * vacuous pass in a quieter voice. */
        if (e != NULL && strcmp(e, "vnni") == 0) {
            fatal("kernels: STOKE_Q4K_KERNEL=vnni is not available in a "
                  "-DSTOKE_MINS_SIMD=0 build -- that arm exists to A/B the "
                  "mins term and the VNNI kernel has only the SIMD spelling, "
                  "so running it here would measure the same mins term twice");
        }
        /* An ASKED-FOR vnni is refused above; the COMPILED-IN default just
         * falls back, because `-DSTOKE_MINS_SIMD=0` is itself the request for
         * the kernel that has both mins arms. */
        g_q4k_vnni = 0;
#endif
#if defined(STOKE_MUT_NO_BOOKKEEP) || defined(STOKE_MUT_G4_DOTONLY) || \
    defined(STOKE_MUT_G4_VNNI_CEIL) || defined(STOKE_MUT_G4_VNNI_A) || \
    defined(STOKE_MUT_G4_VNNI_B)
        /* These five are cut into the MADDUBS twin. Pin the arm to it, and
         * refuse an explicit `vnni` rather than quietly overriding it. */
        if (e != NULL && strcmp(e, "vnni") == 0) {
            fatal("kernels: STOKE_Q4K_KERNEL=vnni in a build carrying a "
                  "maddubs-twin speed mutant -- the mutation is not in the "
                  "VNNI kernel, so this arm would measure an UNMUTATED kernel "
                  "and report that the probe changed nothing");
        }
        g_q4k_vnni = 0;
#endif
#ifdef STOKE_MUT_Q4K_VNNI
        /* And this one is cut into the VNNI kernel, so it pins the other way. */
        if (e != NULL && strcmp(e, "maddubs") == 0) {
            g_q4k_vnni = 0;   /* the deliberate control arm of the mutation test */
        } else {
            g_q4k_vnni = 1;
        }
#endif
    }
#if defined(STOKE_MUT_G4_VNNI_CEIL) || defined(STOKE_MUT_G4_VNNI_B) || \
    defined(STOKE_MUT_G4_VNNI_A) || defined(STOKE_MUT_Q4K_VNNI)
    static int announced = 0;
    if (!announced) {
        announced = 1;
        fprintf(stderr,
                "!! %s BUILD -- the Q4_K grouped kernel's per-sub-block scale "
                "is DELIBERATELY dropped or misapplied. Never ship this "
                "binary.\n",
#if defined(STOKE_MUT_G4_VNNI_CEIL)
                "STOKE_MUT_G4_VNNI_CEIL"
#elif defined(STOKE_MUT_G4_VNNI_A)
                "STOKE_MUT_G4_VNNI_A"
#elif defined(STOKE_MUT_Q4K_VNNI)
                "STOKE_MUT_Q4K_VNNI"
#else
                "STOKE_MUT_G4_VNNI_B"
#endif
        );
    }
#endif
}

static void rows_q6_k_batch_avx2(void* vctx, u64 begin, u64 end) {
    rows_kquant_batch_avx2((const BatchRowJob*)vctx, begin, end,
                           g_q6k_maddubs ? vec_dot_q6_k_q8_k_xn_maddubs_avx2
                                         : vec_dot_q6_k_q8_k_xn_avx2,
                           /* only the maddubs twin groups; the reference xn
                            * kernel is per-sequence and has nothing to pad
                            * into */
                           g_q6k_maddubs ? 4u : 1u,
                           3u /* Q6_K: a remainder of 3 only, measured */);
}

static void rows_q4_k_scalar(void* vctx, u64 begin, u64 end) {
    const RowJob* j = (const RowJob*)vctx;
    for (u64 r = begin; r < end; r++) {
        j->y[r] = vec_dot_q4_k_q8_k(j->base + r * j->row_bytes, j->aq,
                                    j->n_in);
    }
}

/* ▶ M4.5 RUNG A' WAS TRIED HERE AND REJECTED ON MEASUREMENT (2026-08-15).
 * Four rows per pass, sharing one activation load -- the Q8_0 x4 shape -- came
 * back at 0.907x, a 9.3% SLOWDOWN, consistent across three blocks with no
 * overlap in the arms' ranges, and with the anchor guard passing at +0.3% so
 * the measurement is sound. §7.3 permits keeping neither variant; the
 * single-row path stays.
 *
 * WHY IT LOST, hypothesis and unmeasured: the Q8_0 x4 win came when the inner
 * loop ran 8 products/uop and was latency-bound. maddubs changed the regime --
 * four concurrent weight streams, four accumulators and per-super-block
 * scale/min setup add register and cache pressure to a kernel that is already
 * dense. Row tiling solved a problem maddubs had already removed.
 *
 * The kernel, its bit-exactness test and the full reasoning are preserved in
 * history at 01184eb and in
 * gate-reports/2026-08-15-m45-aprime-measure-report.md. Not left in place
 * unused: dead code describing an abandoned approach is worse than no code. */
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

    if (w->type == GGUF_GGML_TYPE_Q4_K || w->type == GGUF_GGML_TYPE_Q6_K) {
        int is_q4 = (w->type == GGUF_GGML_TYPE_Q4_K);
        u32 blk_elems = is_q4 ? KERNELS_Q4_K_BLOCK_ELEMS
                              : KERNELS_Q6_K_BLOCK_ELEMS;
        u32 blk_bytes = is_q4 ? KERNELS_Q4_K_BLOCK_BYTES
                              : KERNELS_Q6_K_BLOCK_BYTES;
        u64 row_bytes = (n_in / blk_elems) * blk_bytes;

        for (u32 s = 0; s < n_seq; s++) {
            if (aq_scratch == NULL || aq_scratch[s] == NULL) {
                fatal("kernels: matvec_batch -- tensor '%s' is a K-quant but "
                      "sequence %u supplied no activation scratch "
                      "((n_in/%u)*%u = %llu bytes required)",
                      w->name, s, KERNELS_Q8_K_BLOCK_ELEMS,
                      KERNELS_Q8_K_BLOCK_BYTES,
                      (unsigned long long)((n_in / KERNELS_Q8_K_BLOCK_ELEMS)
                                           * KERNELS_Q8_K_BLOCK_BYTES));
            }
            quantize_row_q8_k(x[s], n_in, aq_scratch[s]);
        }

#ifdef STOKE_MUT_ALIAS_ACT
        /* MUTANT, SPEED PROBE ONLY -- THE OUTPUT IS DELIBERATELY WRONG.
         * Every sequence is pointed at sequence 0's quantized activation, so
         * the per-super-block activation tile collapses from n_seq*292 B to
         * 292 B and stays L1-resident however many rows sweep it. Nothing else
         * changes: same weights, same rows, same tiling, same kernel.
         *
         * It answers ONE question before a hard kernel rewrite is attempted:
         * is this matmul bound by RE-READING THE ACTIVATION TILE? If this arm
         * is much faster, it is, and an M-rows x N-columns microkernel is the
         * fix. If it is not, the tile is not the wall, and the rewrite would
         * have been a day spent on the wrong hypothesis.
         *
         * The single-threaded microbench answers a DIFFERENT question -- one
         * hot tensor on one core -- and says no. This asks it where the gate
         * measures: 8 threads, 253 cold tensors. */
        static const void* alias_buf[256];
        if (n_seq > 1u && n_seq <= 256u) {
            for (u32 s = 0; s < n_seq; s++) alias_buf[s] = aq_scratch[0];
            aq_scratch = (void* const*)alias_buf;
        }
#endif
        int simd = kernels_cpu_has_avx_vnni();
        BatchRowJob j = {(const u8*)w->data, row_bytes, NULL, n_in, y, n_seq,
                         aq_scratch,
                         is_q4 ? (simd ? rows_q4_k_avx2 : rows_q4_k_scalar)
                               : (simd ? rows_q6_k_avx2 : rows_q6_k_scalar)};
        /* Rung B is fused for both K-quants under AVX2. The scalar paths keep
         * the delegating worker, and n_seq==1 keeps it too: with one sequence
         * there is nothing to amortize and the solo worker is the kernel the
         * archive byte-compare and gate [1] already cover. The step model
         * predicts N=1 cannot move, so routing it here would risk a change
         * that could only be a bug. */
        if (simd && n_seq > 1u) {
            /* Resolved HERE, on the dispatching thread, before any worker
             * exists -- see q6k_select. */
            if (is_q4) q4k_select(); else q6k_select();
            pad_select();
            pool_for_rows(n_out, is_q4 ? rows_q4_k_batch_avx2
                                       : rows_q6_k_batch_avx2, &j);
        } else {
            pool_for_rows(n_out, rows_delegate_batch, &j);
        }
        return;
    }

    /* Unreachable for the four supported types; matvec's own else-branch
     * fatals on anything else, so delegating here keeps that one refusal in
     * one place rather than duplicating its message. */
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
