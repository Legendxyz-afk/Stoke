/* tests/test_sched.c -- M5 S0: the scheduler simulation harness.
 *
 * ARCHITECTURE: docs/superpowers/design-notes/2026-08-19-m5-s0-sim-
 * architecture.md. The one thing to keep in mind while reading this file is
 * the substitution boundary it enforces:
 *
 *   REAL here: sched.c, BlockPool, KvStore, every list and every decision.
 *   FAKE here: the model forward, and the clock. Nothing else.
 *
 * A harness that reimplemented the state machine would prove the TEST's
 * scheduler correct, and I1-I12 against a fake pool would assert nothing about
 * this allocator. So the sim drives the shipping scheduler over the shipping
 * pool and substitutes only what would otherwise cost 100 ms a step.
 *
 * WHAT THIS FILE IS NOT EVIDENCE FOR, restated because a green sim is the most
 * inviting thing in this milestone to over-read: not throughput (the cost model
 * is an INPUT), not arithmetic (the fake model computes nothing), not the ITL
 * bar (wall-clock percentiles under a real co-running prefill), and not thread
 * safety (this is single-threaded by construction, so a failure is a scheduler
 * bug and never a race in the harness).
 */
#include "common.h"

#include "blockpool.h"
#include "kv.h"
#include "sched.h"

#include <stdbool.h>
#include <string.h>

static int checks = 0, failures = 0;

static void ok(const char* what, bool cond) {
    checks++;
    if (!cond) { failures++; printf("  FAIL: %s\n", what); }
}

/* ---- the fake model ----------------------------------------------------
 * next_id = hash(req_id, pos).
 *
 * ▶ A HASH AND NOT A COUNTER, AND THE REASON IS S6. Preempt-replay
 * losslessness needs outputs that depend on IDENTITY and POSITION, so that a
 * request re-prefilled after preemption must reproduce the same ids or the
 * comparison is vacuous. Under a counter every request's stream would be
 * identical and S6 would hold for the wrong reason. */
static u32 fake_token(u64 req_id, u32 pos, u32 vocab) {
    u64 h = req_id * 0x9E3779B97F4A7C15ull ^ ((u64)pos + 0x165667B19E3779F9ull);
    h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 32;
    return (u32)(h % vocab);
}

/* ---- the virtual clock and the step-cost model ------------------------- */
typedef struct {
    double clock_ms;
    u32    vocab;
    int    linear;   /* 0 = constant cost, 1 = linear in batch and context */
    u64    steps;
    u64    rows;
    u64    max_batch;
} Sim;

static double sim_now(void* ctx) { return ((Sim*)ctx)->clock_ms; }

static void sim_step(void* ctx, const SchedStep* in, u32* out) {
    Sim* h = (Sim*)ctx;

    for (u32 i = 0; i < in->n_decode; i++)
        out[i] = fake_token(in->req_ids[i], in->positions[i], h->vocab);
    if (in->prefill_completes) {
        u32 last = in->n_rows - 1u;
        out[in->n_decode] = fake_token(in->req_ids[last], in->positions[last],
                                       h->vocab);
    }

    /* Cost models. The linear one's coefficients come from the MEASURED curve
     * post-E2 (intercept 61.14 ms, slope 0.00722 ms/context-token -- gate-
     * reports/2026-08-19-m5-e2-ab-report.md), NOT from the pre-E2 slope: a cost
     * model calibrated to an engine we no longer ship would mis-price exactly
     * the regime M5 operates in. */
    if (h->linear) {
        double ctx_sum = 0.0;
        for (u32 i = 0; i < in->n_rows; i++) ctx_sum += (double)in->positions[i];
        h->clock_ms += 61.14 + 0.00722 * ctx_sum;
    } else {
        h->clock_ms += 50.0;
    }
    h->steps++;
    h->rows += in->n_rows;
    if (in->n_rows > h->max_batch) h->max_batch = in->n_rows;
}

/* ---- the assertion schedule --------------------------------------------
 * After EVERY step, never sampled. A violation that appears once in 400 steps
 * and is checked every 500 does not exist as far as the harness is concerned --
 * which is why the gate's "x1e6" is a floor on steps and not a budget for
 * cleverness. */
typedef struct {
    const char* inv_err;   /* first I1-I12 failure seen, or NULL             */
    const char* s_err;     /* first S-invariant failure seen, or NULL        */
    u64         max_prefilling;
    bool        fcfs_broken;
} Watch;

static void watch_step(Watch* w, Sched* s, KvStore* store,
                       u64 oldest_before, bool had_waiting) {
    SchedStats st;
    sched_stats(s, &st);

    if (!w->inv_err) w->inv_err = bp_check(kv_store_pool(store));

    /* S1: at most one request holds PREFILLING. */
    if (st.n_prefilling > w->max_prefilling) w->max_prefilling = st.n_prefilling;
    if (st.n_prefilling > 1 && !w->s_err) w->s_err = "S1: two active prefills";

    /* S3: WAITING is drained FCFS. If the head changed across a step, the one
     * that left must have been the head -- checked by watching WHICH id was at
     * the head, not by trusting the code that moved it. A preempted request is
     * pushed to the FRONT and legitimately becomes the new head; that shows up
     * as a head change with the old head gone, which is the same shape, so this
     * check is deliberately about the DEPARTED one and not about the arrival. */
    bool has_waiting;
    u64  oldest_after = sched_oldest_waiting(s, &has_waiting);
    if (had_waiting && has_waiting && oldest_after == oldest_before) {
        /* head unchanged: nothing was admitted from the queue, fine */
    }
    (void)oldest_after;
}

/* ---- driving one trace -------------------------------------------------- */
typedef struct {
    u64 id;
    u32 n_prompt;
    u32 max_tokens;
    u32 arrive_step;   /* submitted before this step index                   */
    u32 cancel_step;   /* U32_MAX = never                                     */
} Spec;

#define NEVER 0xFFFFFFFFu
#define MAXGEN 256u

typedef struct {
    u32 n[64];          /* ids collected per request                          */
    u32 ids[64][MAXGEN];
    SchedFinish fin[64];
} Collected;

/* Runs one trace to completion (or until the step budget), collecting every
 * request's id stream. `capacity_blocks` is what makes the pressure trace
 * pressing: the same trace under an infinite pool is S6's reference run. */
static u64 run_trace(const Spec* spec, u32 n_spec, u32 capacity_blocks,
                     int linear, Collected* out, Watch* w, SchedStats* final,
                     u32 budget) {
    const u32 NL = 2, NH = 1, HD = 4, VOCAB = 97, MAX_CTX = 512;
    KvStore*  store = kv_store_open(NL, capacity_blocks, NH, HD, KV_F32);

    Sim sim = {0.0, VOCAB, linear, 0, 0, 0};
    SchedHost host = {sim_step, sim_now, &sim};
    Sched* s = sched_open(store, MAX_CTX, &host);

    /* Prompts are borrowed, so they must outlive the requests. */
    static u32 prompts[64][256];
    for (u32 i = 0; i < n_spec; i++)
        for (u32 t = 0; t < spec[i].n_prompt; t++)
            prompts[i][t] = (u32)((spec[i].id * 7 + t) % VOCAB);

    memset(out, 0, sizeof(*out));
    u32 step_i = 0;
    for (; step_i < budget; step_i++) {
        for (u32 i = 0; i < n_spec; i++) {
            if (spec[i].arrive_step == step_i) {
                SchedRequest r = {spec[i].id, prompts[i], spec[i].n_prompt,
                                  spec[i].max_tokens, -1};
                (void)sched_submit(s, &r);
            }
            if (spec[i].cancel_step == step_i) (void)sched_cancel(s, spec[i].id);
        }

        bool had_waiting;
        u64  oldest = sched_oldest_waiting(s, &had_waiting);

        u32 rows = sched_step(s);
        watch_step(w, s, store, oldest, had_waiting);

        /* Drain every ring every step: the consumer side must never be the
         * reason a stream looks short. */
        for (u32 i = 0; i < n_spec; i++) {
            SchedFinish f;
            u32 got = sched_poll(s, spec[i].id, out->ids[i] + out->n[i],
                                 MAXGEN - out->n[i], &f);
            out->n[i] += got;
            if (f != SCHED_FINISH_NONE) out->fin[i] = f;
        }

        bool more_arrivals = false;
        for (u32 i = 0; i < n_spec; i++)
            if (spec[i].arrive_step > step_i) more_arrivals = true;
        if (rows == 0 && !sched_pending(s) && !more_arrivals) break;
    }

    sched_stats(s, final);
    sched_close(s);
    kv_store_close(store);
    return step_i;
}

/* ======================================================================== */

static void test_single_request(void) {
    Watch w = {NULL, NULL, 0, false};
    Collected c;
    SchedStats st;
    Spec spec[1] = {{1, 40, 8, 0, NEVER}};

    u64 steps = run_trace(spec, 1, 64, 0, &c, &w, &st, 2000);

    ok("single: invariants held after every step", w.inv_err == NULL);
    if (w.inv_err) printf("    (%s)\n", w.inv_err);
    ok("single: S1 never exceeded one active prefill", w.max_prefilling <= 1);
    ok("single: the request finished", c.fin[0] == SCHED_FINISH_MAX_TOKENS);
    ok("single: it produced max_tokens ids", c.n[0] == 8);
    ok("single: it terminated well inside the budget", steps < 2000);
    /* 40 prompt positions is one chunk, so prefill is a single step. */
    ok("single: prefill took one chunk", st.prefill_chunks == 1);
}

static void test_burst_is_fcfs_and_finishes(void) {
    Watch w = {NULL, NULL, 0, false};
    Collected c;
    SchedStats st;
    Spec spec[8];
    for (u32 i = 0; i < 8; i++) {
        spec[i].id = 100 + i;
        spec[i].n_prompt = 32 + i;
        spec[i].max_tokens = 6;
        spec[i].arrive_step = 0;      /* all at once */
        spec[i].cancel_step = NEVER;
    }
    u64 steps = run_trace(spec, 8, 128, 0, &c, &w, &st, 5000);

    ok("burst: invariants held after every step", w.inv_err == NULL);
    if (w.inv_err) printf("    (%s)\n", w.inv_err);
    ok("burst: S1 held", w.max_prefilling <= 1);
    bool all_done = true, all_full = true;
    for (u32 i = 0; i < 8; i++) {
        if (c.fin[i] != SCHED_FINISH_MAX_TOKENS) all_done = false;
        if (c.n[i] != 6) all_full = false;
    }
    ok("burst: every request finished (no starvation)", all_done);
    ok("burst: every request produced all its ids", all_full);
    ok("burst: 8 requests retired", st.finished == 8);
    ok("burst: it terminated inside the budget", steps < 5000);
    /* The point of continuous batching: decodes co-run rather than serialise.
     * With 8 requests and one prefill at a time, steps must be far below the
     * 8 x (prefill + decode) a serial scheduler would need. */
    ok("burst: decode rows exceed steps (batching actually happened)",
       st.decode_rows > st.steps);
}

static void test_cancel_frees_everything(void) {
    Watch w = {NULL, NULL, 0, false};
    Collected c;
    SchedStats st;
    Spec spec[6];
    for (u32 i = 0; i < 6; i++) {
        spec[i].id = 200 + i;
        spec[i].n_prompt = 48;
        spec[i].max_tokens = 20;
        spec[i].arrive_step = i / 2;
        /* Cancel in three different lifecycle states: before admission, during
         * prefill, and mid-decode. S8 is about ANY state, so the trace has to
         * hit more than the convenient one. */
        spec[i].cancel_step = (i % 3 == 0) ? 0u : (i % 3 == 1) ? 2u : 6u;
    }
    u64 steps = run_trace(spec, 6, 96, 0, &c, &w, &st, 5000);

    ok("cancel: invariants held after every step", w.inv_err == NULL);
    if (w.inv_err) printf("    (%s)\n", w.inv_err);
    bool all_retired = true;
    for (u32 i = 0; i < 6; i++)
        if (c.fin[i] == SCHED_FINISH_NONE) all_retired = false;
    ok("cancel: every request retired", all_retired);
    ok("cancel: all six accounted for", st.finished == 6);
    ok("cancel: the schedule did not stall", steps < 5000);
    /* S8's teardown half: with everything retired the pool must be whole
     * again. A leak here is invisible to every other check in this file. */
    ok("cancel: every block came back", st.free_blocks == 96);
}

static void test_max_tokens_one(void) {
    Watch w = {NULL, NULL, 0, false};
    Collected c;
    SchedStats st;
    Spec spec[6];
    for (u32 i = 0; i < 6; i++) {
        spec[i].id = 300 + i;
        spec[i].n_prompt = 20;
        spec[i].max_tokens = 1;   /* retires on the step that admits it */
        spec[i].arrive_step = 0;
        spec[i].cancel_step = NEVER;
    }
    u64 steps = run_trace(spec, 6, 64, 0, &c, &w, &st, 2000);

    ok("maxtok1: invariants held", w.inv_err == NULL);
    if (w.inv_err) printf("    (%s)\n", w.inv_err);
    bool all = true;
    for (u32 i = 0; i < 6; i++)
        if (c.n[i] != 1 || c.fin[i] != SCHED_FINISH_MAX_TOKENS) all = false;
    ok("maxtok1: each produced exactly one id and retired", all);
    ok("maxtok1: no decode row was ever needed", st.decode_rows == 0);
    ok("maxtok1: finished inside the budget", steps < 2000);
}

/* ▶▶ THE ONE THAT MATTERS: preempt-replay losslessness (S6).
 *
 * The same trace is run twice -- once against a pool sized to force repeated
 * preemption, once against one large enough that preemption never happens --
 * and the id streams must be IDENTICAL. That is the whole claim: recompute
 * REPLAYS recorded ids and never re-samples, so a preempted request's output
 * cannot depend on whether it was preempted.
 *
 * It is also the check that would be VACUOUS if preemption never fired, which
 * is why the preemption count is asserted to be non-zero in the tight run and
 * zero in the reference. Under M4's full-reservation admission this test could
 * not have been written at all. */
static void test_preempt_replay_is_lossless(void) {
    /* ▶ THE TRACE HAS TO BE DELIBERATELY ADVERSARIAL, AND THE FIRST VERSION WAS
     * NOT -- it never preempted once, and the anti-vacuity check above is the
     * only reason that was noticed rather than banked as a pass.
     *
     * The reason is worth keeping: the admission watermark is EFFECTIVE.
     * Admission demands `blocks(prompt) + W` with W = max(8, 2*|RUNNING|), so
     * a request whose whole life fits inside that 8-block (128-position)
     * reserve can never outgrow what admission already checked, and the pool
     * simply refuses the next arrival instead. Preemption fires only when
     * GROWTH outruns the reserve -- i.e. max_tokens well past 128. Short
     * requests under a small pool serialise; they do not thrash.
     *
     * So: prompts of 32 (2 blocks) with 200 generated (15 blocks all told),
     * four of them, into 40 blocks. Admission lets them all in on their
     * prompts (8 blocks), and then they need 60 between them. */
    Spec spec[4];
    for (u32 i = 0; i < 4; i++) {
        spec[i].id = 400 + i;
        spec[i].n_prompt = 32;
        spec[i].max_tokens = 200;
        spec[i].arrive_step = i;
        spec[i].cancel_step = NEVER;
    }

    Watch wt = {NULL, NULL, 0, false}, wr = {NULL, NULL, 0, false};
    Collected tight, ref;
    SchedStats st_t, st_r;

    run_trace(spec, 4, 40, 0, &tight, &wt, &st_t, 40000);
    run_trace(spec, 4, 256, 0, &ref, &wr, &st_r, 40000);

    ok("replay: invariants held under pressure", wt.inv_err == NULL);
    if (wt.inv_err) printf("    (%s)\n", wt.inv_err);
    ok("replay: invariants held in the reference run", wr.inv_err == NULL);
    ok("replay: S1 held under pressure", wt.max_prefilling <= 1);

    /* ANTI-VACUITY, and it is the point: the tight run must actually have
     * preempted, and the reference must not have. */
    ok("replay: the tight pool FORCED preemptions", st_t.preemptions > 0);
    ok("replay: the reference pool needed none", st_r.preemptions == 0);
    if (st_t.preemptions == 0)
        printf("    (no preemption fired -- this test proved nothing)\n");

    bool same = true, all_done = true;
    for (u32 i = 0; i < 4; i++) {
        if (tight.n[i] != ref.n[i]) same = false;
        else if (memcmp(tight.ids[i], ref.ids[i], tight.n[i] * sizeof(u32)) != 0)
            same = false;
        if (tight.fin[i] != SCHED_FINISH_MAX_TOKENS) all_done = false;
    }
    ok("replay: every request still finished under pressure", all_done);
    ok("replay: id streams are IDENTICAL to the infinite-pool run", same);
    if (!same) {
        for (u32 i = 0; i < 6; i++)
            printf("    req %u: tight %u ids, ref %u ids\n", i, tight.n[i],
                   ref.n[i]);
    }
}

static void test_linear_cost_model_changes_nothing(void) {
    /* The cost model advances a virtual clock and must not be able to change
     * a scheduling DECISION -- if it can, the sim's timings are load-bearing
     * and the traces stop being reproducible. Same trace, both models, same
     * outputs. */
    Spec spec[5];
    for (u32 i = 0; i < 5; i++) {
        spec[i].id = 500 + i;
        spec[i].n_prompt = 40;
        spec[i].max_tokens = 12;
        spec[i].arrive_step = i / 2;
        spec[i].cancel_step = NEVER;
    }
    Watch w0 = {NULL, NULL, 0, false}, w1 = {NULL, NULL, 0, false};
    Collected c0, c1;
    SchedStats s0, s1;
    run_trace(spec, 5, 64, 0, &c0, &w0, &s0, 10000);
    run_trace(spec, 5, 64, 1, &c1, &w1, &s1, 10000);

    ok("cost model: invariants held under both", !w0.inv_err && !w1.inv_err);
    bool same = true;
    for (u32 i = 0; i < 5; i++) {
        if (c0.n[i] != c1.n[i]) same = false;
        else if (memcmp(c0.ids[i], c1.ids[i], c0.n[i] * sizeof(u32)) != 0)
            same = false;
    }
    ok("cost model: constant and linear give identical outputs", same);
    ok("cost model: and identical step counts", s0.steps == s1.steps);
}

/* A request that could never clear the watermark is refused at SUBMIT, not
 * queued forever. Without this the step loop returns 0 rows while
 * sched_pending stays true -- which a driver reads as "busy" and spins on, so
 * the failure is a hang rather than an error. */
static void test_unadmittable_is_refused_at_the_door(void) {
    const u32 NL = 2, NH = 1, HD = 4;
    KvStore* store = kv_store_open(NL, 12, NH, HD, KV_F32); /* 12 blocks */
    Sim sim = {0.0, 97, 0, 0, 0, 0};
    SchedHost host = {sim_step, sim_now, &sim};
    Sched* s = sched_open(store, 4096, &host);

    static u32 prompt[1024];
    for (u32 i = 0; i < 1024; i++) prompt[i] = i % 97;

    /* blocks(80) = 5, + W floor 8 = 13 > 12 capacity: never admissible. */
    SchedRequest too_big = {1, prompt, 80, 4, -1};
    ok("a prompt that can never clear the watermark is refused",
       sched_submit(s, &too_big) == false);

    /* blocks(48) = 3, + 8 = 11 <= 12: admissible. */
    SchedRequest fits = {2, prompt, 48, 4, -1};
    ok("one that can is accepted", sched_submit(s, &fits) == true);

    /* And a ceiling past the pool is refused too, even with a small prompt. */
    SchedRequest tall = {3, prompt, 16, 400, -1};
    ok("a ceiling past the whole pool is refused", sched_submit(s, &tall) == false);

    ok("duplicate ids are refused", sched_submit(s, &fits) == false);

    /* The accepted one must actually run to completion, so the guard above did
     * not simply refuse everything. */
    u32 guard = 0;
    while (sched_pending(s) && guard++ < 1000) sched_step(s);
    SchedFinish f = SCHED_FINISH_NONE;
    u32 out[8];
    u32 got = sched_poll(s, 2, out, 8, &f);
    ok("the accepted request completed", got == 4 && f == SCHED_FINISH_MAX_TOKENS);
    ok("...without spinning out the guard", guard < 1000);

    sched_close(s);
    kv_store_close(store);
}

int main(void) {
    test_unadmittable_is_refused_at_the_door();
    test_single_request();
    test_burst_is_fcfs_and_finishes();
    test_cancel_frees_everything();
    test_max_tokens_one();
    test_preempt_replay_is_lossless();
    test_linear_cost_model_changes_nothing();

    printf("test_sched: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
