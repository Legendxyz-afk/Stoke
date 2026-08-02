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

#endif /* STOKE_BENCH_H */
