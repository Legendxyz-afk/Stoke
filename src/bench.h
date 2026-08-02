/* src/bench.h -- `stoke bench --ops`: the M3 per-op attribution instrument.
 *
 * Mandated by the M3-M7 forward design note (§M3, "Missing the >=50% bar"):
 * attribution FIRST, guessing never. This lands as M3's K1, BEFORE any SIMD
 * is written, because the whole milestone's miss-the-bar ladder runs on its
 * output and a ladder with no instrument is a guess with extra steps.
 *
 * The note's ranked culprit list is what this must be able to discriminate:
 *   (1) logits matvec not parallelized / embd rows dequanted per call
 *   (2) K/V layout breaking sequential streams in attention
 *   (3) f16->f32 conversions in the inner loop
 *   (4) barrier/pool overhead on small ops
 *   (5) page faults during timing
 * Culprit (5) is the one a naive harness silently BECOMES rather than
 * measures -- see bench.c's header for how that is handled.
 */
#ifndef STOKE_BENCH_H
#define STOKE_BENCH_H

#include "common.h"

/* Runs the per-op microbench over `model_path`'s real tensor shapes and
 * prints the table to stdout. `reps_override` == 0 selects the adaptive
 * rep count (see bench.c); nonzero pins it, which exists so a run can be
 * made cheap-and-noisy on purpose when you only want the table's shape.
 * Fatal on any file/shape problem -- never returns a partial table. */
void bench_ops(const char* model_path, u32 reps_override);

/* ---- pure helpers, EXPOSED FOR TESTS ----
 *
 * Same rationale as model.h's model_check_tensor_shape: these two functions
 * are the arithmetic that every number in the table is computed from, and a
 * microbench whose own statistics are wrong does not fail loudly -- it prints
 * a plausible table. They are pure and cheap to drive directly, so
 * tests/test_bench.c does exactly that rather than trying to infer their
 * correctness from formatted output. */

typedef struct {
    double min_ms;
    double med_ms;
    double max_ms;
    u32    reps;
} BenchTiming;

/* Sorts `samples[0..n)` IN PLACE (ascending) and returns its spread.
 * Median of an even count is the upper-middle element, not an interpolated
 * average: every reported figure must be a duration that was actually
 * observed, never a synthetic value no rep ever took. Fatal if n == 0. */
BenchTiming bench_spread(double* samples, u32 n);

/* bytes/sec expressed in GB/s (1e9 bytes). Returns 0.0 for a non-positive
 * duration rather than dividing -- a zero/negative interval means the timer
 * did not resolve the op, and 0.0 reads as "no measurement" where an inf
 * would read as "infinitely fast". */
double bench_gbs(u64 bytes, double ms);

/* Per-op timed budget. Long enough to swamp timer granularity and scheduler
 * jitter, short enough that a full table is not a coffee break. */
#define BENCH_TARGET_MS 250.0

/* Sample-count floor for ops too slow to fit BENCH_TARGET_MS.
 *
 * This was 3, and 3 was measurably wrong. `matvec LOGITS` costs ~180 ms, so
 * the sizer asked for 250/180 = 1.4 reps and floored to 3 -- roughly 1.4 s of
 * observation for the single op this table names culprit #1. Measured across
 * three runs minutes apart on a quiesced box (2026-08-02): 193.8 / 463.8 /
 * 176.8 ms, a 2.6x swing.
 *
 * The subtle part is WHY that is dangerous, and it is the reason this floor
 * is a SAMPLE COUNT and not a wall-clock budget. In the 463.8 ms run, two of
 * the three samples were slow. Median-of-3 tolerates exactly one outlier, so
 * it faithfully reported the bad value -- while max/med came to just 1.33,
 * which this table's own footer reads as a HEALTHY measurement. The run
 * looked clean while being 2.6x wrong. A spread heuristic detects TRANSIENT
 * interference and is blind to SUSTAINED interference, and sustained is
 * precisely the kind that moves a median.
 *
 * 11 samples require 6 bad ones to move the median rather than 2, and cost
 * ~2 s on the slowest op here. Odd, so the median remains a real observation
 * rather than a synthetic midpoint. */
#define BENCH_MIN_REPS 11u
#define BENCH_MAX_REPS 20000u

/* Adaptive rep count for an op whose single call took `pilot_ms`: enough reps
 * to fill BENCH_TARGET_MS, clamped to [BENCH_MIN_REPS, BENCH_MAX_REPS].
 *
 * A non-positive pilot returns BENCH_MAX_REPS -- the timer failed to resolve
 * one call, so the op is far faster than the clock and wants the most samples
 * available. NaN takes that branch too, deliberately: a NaN pilot must not
 * fall through to a comparison chain where every test is false. */
u32 bench_reps_for(double pilot_ms);

#endif /* STOKE_BENCH_H */
