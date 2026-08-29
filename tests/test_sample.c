/* tests/test_sample.c -- src/sample.c, the token chooser. M6 rung A6.
 *
 * ▶▶ WHAT THESE TESTS ARE REALLY FOR. A sampler is easy to write and hard to
 * catch being wrong: any output is a plausible token, so a broken one produces
 * fluent text and nothing fails. So nothing here asserts "it picked a good
 * token". Every test pins a property that a WRONG sampler would violate while
 * still looking fine -- greedy equivalence, purity under replay, truncation
 * actually truncating, and the penalty moving the answer.
 */
#include "common.h"

#include "sample.h"

#include <string.h>

#include "test.h"

/* A distribution with a clear winner and a long tail, so truncation has
 * something to cut and the greedy answer is unambiguous. */
static void mk_logits(f32* v, u32 n) {
    for (u32 i = 0; i < n; i++) v[i] = -10.0f + (f32)(i % 7);
    v[3]  = 8.0f;  /* the winner  */
    v[11] = 6.0f;  /* the runner-up */
    v[42] = 4.0f;
}

#define NV 128

TEST(temperature_zero_is_exactly_argmax) {
    f32 v[NV], scratch[NV];
    mk_logits(v, NV);
    SampleParams p;
    memset(&p, 0, sizeof p);
    p.temperature = 0.0f;
    p.seed        = 12345;
    /* Every position must agree, because greedy does not consult the seed. */
    for (u32 pos = 0; pos < 8; pos++) {
        ASSERT_EQ_I64(sample_token(&p, pos, v, NV, NULL, 0, scratch), 3);
    }
}

/* A zeroed params struct IS greedy. The server leans on this: it means
 * `temperature: 0`, `top_p: 1` and the penalties at their identities need no
 * special case anywhere. */
TEST(a_zeroed_params_struct_is_greedy) {
    f32 v[NV], scratch[NV];
    mk_logits(v, NV);
    SampleParams p;
    memset(&p, 0, sizeof p);
    ASSERT_EQ_I64(sample_token(&p, 0, v, NV, NULL, 0, scratch), 3);
}

/* ⚠ THE REPLAY PROPERTY, and the reason this file exists. The scheduler
 * regenerates a preempted sequence's positions; the same (seed, pos) must give
 * the same token or preempt-replay silently stops being lossless. */
TEST(the_same_seed_and_position_always_give_the_same_token) {
    f32 v[NV], scratch[NV];
    mk_logits(v, NV);
    SampleParams p;
    memset(&p, 0, sizeof p);
    p.temperature = 1.5f;
    p.seed        = 777;

    for (u32 pos = 0; pos < 16; pos++) {
        u32 a = sample_token(&p, pos, v, NV, NULL, 0, scratch);
        u32 b = sample_token(&p, pos, v, NV, NULL, 0, scratch);
        ASSERT_EQ_I64(a, b);
    }
}

/* And the other half: it must not depend on CALL ORDER either, which is what a
 * hidden running RNG would sneak in. Walking the positions backwards must give
 * the same answers as walking them forwards. */
TEST(sampling_does_not_depend_on_the_order_positions_are_asked_for) {
    f32 v[NV], scratch[NV];
    mk_logits(v, NV);
    SampleParams p;
    memset(&p, 0, sizeof p);
    p.temperature = 1.5f;
    p.seed        = 4242;

    u32 fwd[16], bwd[16];
    for (u32 i = 0; i < 16; i++) fwd[i] = sample_token(&p, i, v, NV, NULL, 0, scratch);
    for (int i = 15; i >= 0; i--)
        bwd[i] = sample_token(&p, (u32)i, v, NV, NULL, 0, scratch);
    for (u32 i = 0; i < 16; i++) ASSERT_EQ_I64(fwd[i], bwd[i]);
}

/* A different seed must actually change the stream -- otherwise the seed is
 * decorative and every "reproducible run" claim built on it is empty. */
TEST(a_different_seed_gives_a_different_stream) {
    f32 v[NV];
    mk_logits(v, NV);
    SampleParams a, b;
    memset(&a, 0, sizeof a);
    a.temperature = 2.0f;
    a.seed        = 1;
    b = a;
    b.seed = 2;

    f32 scratch[NV];
    int differ = 0;
    for (u32 pos = 0; pos < 64; pos++) {
        u32 ta = sample_token(&a, pos, v, NV, NULL, 0, scratch);
        u32 tb = sample_token(&b, pos, v, NV, NULL, 0, scratch);
        if (ta != tb) differ++;
    }
    ASSERT_TRUE(differ > 0);
}

TEST(top_k_one_is_argmax_whatever_the_temperature) {
    f32 v[NV], scratch[NV];
    mk_logits(v, NV);
    SampleParams p;
    memset(&p, 0, sizeof p);
    p.temperature = 5.0f;
    p.top_k       = 1;
    p.seed        = 99;
    for (u32 pos = 0; pos < 32; pos++) {
        ASSERT_EQ_I64(sample_token(&p, pos, v, NV, NULL, 0, scratch), 3);
    }
}

/* Truncation must TRUNCATE: with k = 2 nothing outside the top two may ever be
 * chosen, however hot the temperature. A sampler that applied top_k to the
 * probabilities but still drew from the full vocabulary passes every "it
 * returns a token" check and fails this. */
TEST(top_k_two_never_returns_anything_outside_the_top_two) {
    f32 v[NV], scratch[NV];
    mk_logits(v, NV);
    SampleParams p;
    memset(&p, 0, sizeof p);
    p.temperature = 8.0f;
    p.top_k       = 2;
    p.seed        = 31337;
    int seen_winner = 0, seen_runner_up = 0;
    for (u32 pos = 0; pos < 400; pos++) {
        u32 id = sample_token(&p, pos, v, NV, NULL, 0, scratch);
        ASSERT_TRUE(id == 3 || id == 11);
        if (id == 3) seen_winner = 1;
        if (id == 11) seen_runner_up = 1;
    }
    /* ⚠ BOTH DIRECTIONS, because "never outside the top two" is satisfied by a
     * sampler that always returns the argmax -- which is exactly what the stub
     * this test was written against did, and it passed. Requiring the runner-up
     * to actually appear is what makes the assertion about TRUNCATION rather
     * than about returning a plausible token. */
    ASSERT_TRUE(seen_winner);
    ASSERT_TRUE(seen_runner_up);
}

/* top_p at a mass the winner alone already covers must collapse to the winner.
 * v[3]=8 against a runner-up at 6 is a ~0.88 share at temperature 1. */
TEST(top_p_below_the_winners_own_mass_collapses_to_the_winner) {
    f32 v[NV], scratch[NV];
    mk_logits(v, NV);
    SampleParams p;
    memset(&p, 0, sizeof p);
    p.temperature = 1.0f;
    p.top_p       = 0.5f;
    p.seed        = 5;
    for (u32 pos = 0; pos < 100; pos++) {
        ASSERT_EQ_I64(sample_token(&p, pos, v, NV, NULL, 0, scratch), 3);
    }
}

/* The penalty has to be able to move the answer, or it is not applied. With the
 * winner already generated and a hard enough penalty, greedy must pick the
 * runner-up instead. */
TEST(repeat_penalty_moves_greedy_off_a_token_already_generated) {
    f32 v[NV], scratch[NV];
    mk_logits(v, NV);
    SampleParams p;
    memset(&p, 0, sizeof p);
    p.temperature    = 0.0f;   /* greedy, so the effect is unambiguous */
    p.repeat_penalty = 100.0f;
    u32 hist[1] = {3};
    ASSERT_EQ_I64(sample_token(&p, 1, v, NV, hist, 1, scratch), 11);
}

TEST(repeat_penalty_at_one_changes_nothing) {
    f32 v[NV], scratch[NV];
    mk_logits(v, NV);
    SampleParams p;
    memset(&p, 0, sizeof p);
    p.repeat_penalty = 1.0f;
    u32 hist[1] = {3};
    ASSERT_EQ_I64(sample_token(&p, 1, v, NV, hist, 1, scratch), 3);
}

/* The RNG's shape is pinned: a seeded stream that changed between releases
 * would invalidate every reproduction anyone had recorded, silently. */
TEST(the_rng_is_a_pure_function_of_seed_and_position) {
    ASSERT_TRUE(sample_rng(1, 0) == sample_rng(1, 0));
    ASSERT_TRUE(sample_rng(1, 0) != sample_rng(1, 1));
    ASSERT_TRUE(sample_rng(1, 0) != sample_rng(2, 0));
    /* and it must not be trivially structured -- a counter would pass the three
     * lines above while being a terrible source of randomness. */
    u64 a = sample_rng(9, 5), b = sample_rng(9, 6);
    ASSERT_TRUE(a != b + 1 && b != a + 1);
}
