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
 * Fatal on any file/shape problem -- never returns a partial table.
 *
 * `jsonl` nonzero additionally emits one machine-readable JSON object as the
 * FINAL line, for bench/append_history.py. It is purely additive: the human
 * table is byte-identical either way, so a consumer takes the last line and a
 * reader sees no difference. Measurements only -- provenance is the caller's
 * job, because this process cannot know its own git commit and a benchmark
 * that guesses provenance corrupts the history it is meant to inform. */
void bench_ops(const char* model_path, u32 reps_override, int jsonl);

/* `stoke bench --matvec-batch` -- the BATCHED K-quant kernel, timed directly,
 * with the activation-reuse ablation as its second arm.
 *
 * ▶ WHY IT IS SEPARATE FROM bench_ops RATHER THAN A ROW IN ITS TABLE. Two
 * reasons, both about not damaging an instrument that already works: the
 * standing table feeds `bench/history.jsonl`, whose only value is row-for-row
 * comparability across commits, and its capture array is bounded
 * (BENCH_MAX_CAPTURED) -- adding fifteen cells would either overflow it or
 * silently change what a historical row means. And this mode is an A/B, whose
 * output is a RATIO between two arms rather than a rate: it belongs beside the
 * ablations, not beside the projections.
 *
 * `cycles` is the interleaved A/B/C cycles per cell (0 = 8). `max_seqs` is the
 * largest n_seq swept, doubling from 1 (0 or > KERNELS_BATCH_TILE = the tile).
 * `only` restricts the run to one tensor by label (NULL = all four), which also
 * sizes the scratch buffers to THAT tensor -- two confounds, not one.
 * The full contract, the three power checks and the reason the alias lives in
 * the CALLER are in bench.c beside the code. */
void bench_matvec_batch(const char* model_path, u32 cycles, u32 max_seqs,
                        const char* only, const char* seqs_list,
                        int pad_ab);

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

/* The %ref denominator: the LARGER of the two fast reference variants.
 *
 * split8_f32 and xor_u64 trade places between runs on this box (13.9-16.6
 * GB/s, either one on top), so neither is "the" answer. The quantity being
 * estimated is "at least this much read bandwidth exists", and the larger of
 * two lower bounds is the better lower bound -- taking a mean or a fixed
 * favourite would understate it in exactly the direction that produced the
 * original error.
 *
 * Exposed rather than inlined for the same reason bench_spread and bench_gbs
 * are: this one line decides every %ref in the table, and the project has
 * already shipped one denominator that was wrong for months. A rule that
 * matters that much gets a test, not a reading. Non-positive inputs are
 * passed through by the same max, so an unresolvable reference yields <= 0
 * and row() falls back to printing 0.0% rather than dividing. */
double bench_ref_denominator(double split8_gbs, double xor_gbs);

/* The exact CFLAGS this translation unit was compiled with, injected by the
 * Makefile's explicit bench.o rule and emitted into --jsonl.
 *
 * WHY THIS DOES NOT VIOLATE "PROVENANCE IS NOT THE BENCHMARK'S JOB" (see
 * bench_ops' contract, which refuses to guess its own git commit): a commit
 * is EXTERNAL state the process would have to infer, and inferring it writes
 * confident falsehoods. Its own compile flags are an intrinsic property of
 * the binary, known at compile time, and impossible to get wrong. The
 * distinction is guessing versus knowing, not "metadata versus measurement".
 *
 * It became necessary on 2026-08-03: the build moved -O1 -> -O3, every
 * matvec got 3.1x faster, and history.jsonl recorded binary_sha256 but
 * nothing that said WHY two rows differed by 3x. A history file whose rows
 * cannot be compared is not a history file. */
#ifndef STOKE_BUILD_CFLAGS
#define STOKE_BUILD_CFLAGS "(unrecorded -- built outside the Makefile)"
#endif

/* Per-op timed budget. Long enough to swamp timer granularity and scheduler
 * jitter, short enough that a full table is not a coffee break.
 *
 * Raised 250 -> 1000 on 2026-08-02 against measurement, not taste. At 250 ms
 * the mid-size projections got 20-26 reps and their medians moved 20-27%
 * between runs on an otherwise quiet box -- enough to swamp any optimization
 * M3 is likely to produce, which would make the history file useless for the
 * regression-diffing it exists for.
 *
 * The cause was established two-directionally rather than assumed. Ops with
 * many reps were already stable (attn_k, 248 reps, 1.8% run-to-run); ops with
 * few were not (ffn_up, 20 reps, 26.5%). Pinning every op to 100 reps then
 * moved BOTH in the predicted direction: ffn_up 26.5 -> 9.6%, ffn_down
 * 10.1 -> 2.3%, attn_q 20.7 -> 6.8%, while attn_k -- which LOST reps, 248 ->
 * 100 -- degraded 1.8 -> 78.3%. Run-to-run spread is dominated by how well
 * the median is determined, i.e. by sampling, not by box state.
 *
 * 1000 ms costs ~12 s per run and leaves every projection above ~95 reps.
 * matvec LOGITS is the one op this cannot help: at ~176 ms it stays on
 * BENCH_MIN_REPS, and pinning it to 100 reps did NOT reduce its ~10% spread
 * (11.2% at 100 reps vs 10.0% at 11), so that residual is genuine box drift
 * over a multi-second op and is irreducible by sampling. */
#define BENCH_TARGET_MS 1000.0

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

/* Worst-case output bytes (excluding the NUL) for escaping `n` input bytes:
 * every byte could need the six-character \u00XX form. */
#define BENCH_JSON_ESCAPE_MAX(n) ((n) * 6u + 1u)

/* Escapes `in` for use inside a JSON string literal, writing at most `cap`
 * bytes to `out` including the NUL. Returns bytes written, excluding the NUL.
 * FATAL if `cap` is too small -- silently truncating would emit JSON that is
 * well-formed but WRONG, which is worse than the malformed JSON this fixes.
 *
 * THIS IS NOT DEFENSIVE PROGRAMMING, it is a bug that shipped. --jsonl
 * interpolated the model path straight into a JSON string, so the natural
 * Windows invocation
 *     stoke bench --ops models\foo.gguf --jsonl
 * emitted `"model":"models\foo.gguf"` -- an invalid \f escape, which made the
 * whole line unparseable and killed bench/append_history.py, the one consumer
 * the flag exists to serve. It went unnoticed because every recorded run so
 * far happened to be typed with forward slashes. */
u64 bench_json_escape(const char* in, char* out, u64 cap);

#endif /* STOKE_BENCH_H */
