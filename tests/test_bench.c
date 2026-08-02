/* tests/test_bench.c -- the arithmetic behind `stoke bench --ops`.
 *
 * WHY THIS FILE EXISTS. bench.c's output is a table of numbers that nobody
 * can eyeball for correctness: if the median is off by one index or GB/s is
 * scaled wrong, the table still looks entirely plausible and M3's whole
 * attribution ladder then runs on it. A benchmark's own statistics are
 * exactly the kind of code that fails toward "fine" -- it is never obviously
 * broken, it is just quietly wrong, and every conclusion drawn downstream
 * inherits the error.
 *
 * So bench_spread() and bench_gbs() are driven directly, with hand-computed
 * expectations, including the boundary cases a formatted table would never
 * reveal: even counts, all-equal samples, single samples, reverse-sorted
 * input, and a non-positive duration.
 */
#include "common.h"

#include "bench.h"

#include "../tests/test.h"

/* Doubles here are compared with a tolerance rather than == because the GB/s
 * path divides; the spread path is exact (it only copies samples around), so
 * those assertions use exact equality deliberately -- a spread function that
 * perturbs a value it was only supposed to select is itself a bug. */
static int close_enough(double a, double b) {
    double d = a - b;
    if (d < 0) {
        d = -d;
    }
    return d <= 1e-9 * (b < 0 ? -b : b) + 1e-12;
}

TEST(spread_sorts_and_selects_from_unsorted_input) {
    double s[5] = { 5.0, 1.0, 4.0, 2.0, 3.0 };
    BenchTiming t = bench_spread(s, 5);
    ASSERT_TRUE(t.min_ms == 1.0);
    ASSERT_TRUE(t.max_ms == 5.0);
    ASSERT_TRUE(t.med_ms == 3.0);
    ASSERT_EQ_U32(t.reps, 5u);
    /* sorted IN PLACE, as documented */
    ASSERT_TRUE(s[0] == 1.0 && s[4] == 5.0);
}

TEST(spread_handles_reverse_sorted_input) {
    double s[5] = { 9.0, 7.0, 5.0, 3.0, 1.0 };
    BenchTiming t = bench_spread(s, 5);
    ASSERT_TRUE(t.min_ms == 1.0);
    ASSERT_TRUE(t.med_ms == 5.0);
    ASSERT_TRUE(t.max_ms == 9.0);
}

/* Even n: the contract says UPPER-middle, not an interpolated average. This
 * is the assertion that pins "every reported figure was actually observed" --
 * an averaging median would return 2.5 here, a duration no rep ever took. */
TEST(spread_even_count_takes_upper_middle_never_an_average) {
    double s[4] = { 1.0, 2.0, 3.0, 4.0 };
    BenchTiming t = bench_spread(s, 4);
    ASSERT_TRUE(t.med_ms == 3.0);
    ASSERT_TRUE(t.med_ms != 2.5);
    ASSERT_TRUE(t.min_ms == 1.0);
    ASSERT_TRUE(t.max_ms == 4.0);
}

TEST(spread_single_sample_is_min_med_and_max) {
    double s[1] = { 42.5 };
    BenchTiming t = bench_spread(s, 1);
    ASSERT_TRUE(t.min_ms == 42.5);
    ASSERT_TRUE(t.med_ms == 42.5);
    ASSERT_TRUE(t.max_ms == 42.5);
    ASSERT_EQ_U32(t.reps, 1u);
}

TEST(spread_all_equal_samples_collapse) {
    double s[6] = { 7.0, 7.0, 7.0, 7.0, 7.0, 7.0 };
    BenchTiming t = bench_spread(s, 6);
    ASSERT_TRUE(t.min_ms == 7.0);
    ASSERT_TRUE(t.med_ms == 7.0);
    ASSERT_TRUE(t.max_ms == 7.0);
}

/* A single huge outlier must move max WITHOUT dragging the median -- this is
 * precisely the "interference during the run, not a slow kernel" signal the
 * table tells its reader to look for, so it needs to actually work. */
TEST(spread_outlier_moves_max_but_not_median) {
    double s[7] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0 };
    BenchTiming t = bench_spread(s, 7);
    ASSERT_TRUE(t.med_ms == 1.0);
    ASSERT_TRUE(t.max_ms == 1000.0);
    ASSERT_TRUE(t.min_ms == 1.0);
}

TEST(gbs_known_value) {
    /* 1e9 bytes in exactly 1000 ms == 1.0 GB/s. */
    ASSERT_TRUE(close_enough(bench_gbs(1000000000ull, 1000.0), 1.0));
    /* 1e9 bytes in 100 ms == 10 GB/s. */
    ASSERT_TRUE(close_enough(bench_gbs(1000000000ull, 100.0), 10.0));
    /* 256 MiB in 10 ms. */
    double want = ((double)(268435456ull) / 1e9) / 0.010;
    ASSERT_TRUE(close_enough(bench_gbs(268435456ull, 10.0), want));
}

/* Non-positive duration must read as "no measurement" (0.0), never inf --
 * an inf would print as the fastest op in the table and be believed. */
TEST(gbs_non_positive_duration_is_zero_not_infinity) {
    ASSERT_TRUE(bench_gbs(1000000000ull, 0.0) == 0.0);
    ASSERT_TRUE(bench_gbs(1000000000ull, -1.0) == 0.0);
}

TEST(gbs_zero_bytes_is_zero) {
    ASSERT_TRUE(bench_gbs(0ull, 5.0) == 0.0);
}

/* The %ref column is a ratio of two bench_gbs results over the SAME elapsed
 * time basis, so it must be invariant when the whole box slows down evenly.
 * That invariance is the entire argument for reporting %ref rather than
 * absolute GB/s (bench.c Decision 2) -- assert it rather than trust it. */
TEST(gbs_ratio_is_invariant_under_uniform_slowdown) {
    u64    op_bytes  = 400000000ull;
    u64    ref_bytes = 268435456ull;
    double op_ms     = 30.0;
    double ref_ms    = 8.0;

    double fast = bench_gbs(op_bytes, op_ms) / bench_gbs(ref_bytes, ref_ms);

    /* Everything takes 2.5x longer -- a loaded box. */
    double slow = bench_gbs(op_bytes, op_ms * 2.5)
                  / bench_gbs(ref_bytes, ref_ms * 2.5);

    ASSERT_TRUE(close_enough(fast, slow));
}

/* ---- bench_reps_for: the adaptive sample count ----
 *
 * These pin a lesson, not just an arithmetic contract. The rep floor was 3,
 * and 3 silently produced a 2.6x-wrong headline number (see bench.h's
 * BENCH_MIN_REPS comment). The sizer is pure arithmetic over one double, so
 * it is driven directly here rather than inferred from a printed table.
 */

/* THE REGRESSION TEST FOR THE 2026-08-02 DEFECT. A ~180 ms op asks for
 * 250/180 = 1.4 reps and lands on the floor. With the old floor of 3, the
 * median moved as soon as TWO samples were bad -- which is exactly what
 * happened: LOGITS read 193.8 / 463.8 / 176.8 ms across three runs while
 * max/med stayed at 1.33, so the table called a 2.6x-wrong run healthy. */
TEST(reps_for_slow_op_takes_a_floor_that_tolerates_more_than_one_outlier) {
    ASSERT_EQ_U32(bench_reps_for(180.0), BENCH_MIN_REPS);
    /* 3 was the bug: with 3 samples, 2 bad ones decide the median. */
    ASSERT_TRUE(BENCH_MIN_REPS >= 11u);
    /* An odd floor keeps the median a real observation, per bench_spread. */
    ASSERT_TRUE((BENCH_MIN_REPS % 2u) == 1u);
}

/* Between the clamps the count is just the budget divided by the cost. */
TEST(reps_for_scales_inversely_with_pilot_cost) {
    ASSERT_EQ_U32(bench_reps_for(10.0), 25u);   /* 250/10 */
    ASSERT_EQ_U32(bench_reps_for(5.0), 50u);    /* 250/5  */
}

/* A sub-microsecond op would otherwise ask for millions of reps. */
TEST(reps_for_fast_op_is_capped_at_max) {
    ASSERT_EQ_U32(bench_reps_for(0.001), BENCH_MAX_REPS);
}

/* Non-positive pilot = the timer never resolved a single call. */
TEST(reps_for_unresolved_timer_takes_max) {
    ASSERT_EQ_U32(bench_reps_for(0.0), BENCH_MAX_REPS);
    ASSERT_EQ_U32(bench_reps_for(-1.0), BENCH_MAX_REPS);
}

/* The case a plain `pilot <= 0.0` guard gets WRONG: every comparison against
 * NaN is false, so a NaN pilot would fall past both clamps and return a cast
 * of a NaN. Built at runtime through volatile so it cannot be constant-folded
 * into a compile-time diagnostic, and no libm dependency is added. */
TEST(reps_for_nan_pilot_takes_max_not_a_false_comparison_chain) {
    volatile double z = 0.0;
    double          nan_pilot = z / z;
    ASSERT_TRUE(nan_pilot != nan_pilot);   /* it really is a NaN */
    ASSERT_EQ_U32(bench_reps_for(nan_pilot), BENCH_MAX_REPS);
}

/* A cheaper op must never get FEWER samples than a costlier one. A sizer that
 * is not monotone would make two ops' spreads incomparable, which is the
 * whole basis of reading one row against another. */
TEST(reps_for_is_monotone_non_increasing_in_pilot_cost) {
    double pilots[7] = { 0.001, 0.1, 1.0, 10.0, 50.0, 180.0, 5000.0 };
    for (int i = 1; i < 7; i++) {
        ASSERT_TRUE(bench_reps_for(pilots[i]) <= bench_reps_for(pilots[i - 1]));
    }
}
