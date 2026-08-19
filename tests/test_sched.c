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

/* xorshift, so every trace is reproducible from its seed. */
typedef struct { u64 s; } Rng;
static u32 rnd(Rng* r, u32 n) {
    r->s ^= r->s << 13; r->s ^= r->s >> 7; r->s ^= r->s << 17;
    return (u32)(r->s % (n ? n : 1u));
}

/* ---- the virtual clock and the step-cost model ------------------------- */
typedef struct {
    double clock_ms;
    u32    vocab;
    int    linear; /* 0 = constant cost, 1 = linear in batch and context */
    u64    steps;
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
}

/* ---- the assertion schedule --------------------------------------------
 * After EVERY step, never sampled. A violation that appears once in 400 steps
 * and is checked every 500 does not exist as far as the harness is concerned --
 * which is why the gate's "x1e6" is a floor on steps and not a budget for
 * cleverness. */
typedef struct {
    const char* err;        /* first failure of ANY invariant, or NULL       */
    u64         admissions; /* times a request left WAITING for PREFILLING   */
    u64         s2_checked;
    u64         s5_checked;
} Watch;

/* State captured BEFORE a step, so the checks compare ACROSS it rather than
 * trusting one snapshot. The watermark in particular is a statement about the
 * moment the decision was made, not about the state it produced. */
typedef struct {
    SchedStats st;
    u64        oldest_waiting;
    bool       had_waiting;
    u32        head_prompt_blocks;
} Before;

static void watch_after(Watch* w, Sched* s, KvStore* store, const Before* b) {
    SchedStats a;
    sched_stats(s, &a);

    if (!w->err) w->err = bp_check(kv_store_pool(store));

    /* S1 -- at most one active prefill. */
    if (!w->err && a.n_prefilling > 1) w->err = "S1: two active prefills";

    /* S2 -- ADMISSION ONLY ABOVE THE WATERMARK, checked directly rather than
     * inferred. An admission is visible as the PREFILLING slot going empty ->
     * occupied; the watermark it had to clear is computed from the state BEFORE
     * the step, which is why `Before` exists. */
    bool admitted = (b->st.n_prefilling == 0 && a.n_prefilling == 1);
    if (admitted) {
        w->admissions++;
        w->s2_checked++;
        u32 wm = SCHED_W_PER_RUNNING * b->st.n_running;
        if (wm < SCHED_W_FLOOR) wm = SCHED_W_FLOOR;
        if (!w->err && b->st.free_blocks < b->head_prompt_blocks + wm)
            w->err = "S2: admitted below the watermark";
    }

    /* S5 -- EVERY RUNNABLE DECODE DECODES EVERY STEP.
     *
     * Exactly, not as a bound: decode rows this step must equal the RUNNING
     * count at step top, less any of THOSE the step itself preempted. A yielded
     * PREFILL is also a preemption but costs no decode row, which is why
     * prefill_yields is reported separately -- with only the total this could
     * be an inequality at best, and an inequality stays true through a bug that
     * silently drops a runnable decode.
     *
     * ▶ AND A REQUEST CANCELLED AT THIS STEP'S TOP OWES NO ROW EITHER, because
     * it is no longer RUNNABLE -- which is what S5 quantifies over. The first
     * version of this assertion omitted that term and reported the scheduler
     * broken on every trace containing aborts. The assertion was wrong, not the
     * scheduler; the fix is the term, and the lesson is that "asserted
     * directly" is only worth more than "inferred" once the accounting is
     * complete. */
    if (a.steps != b->st.steps) {
        u64 d_rows = a.decode_rows - b->st.decode_rows;
        u64 d_pre = a.preemptions - b->st.preemptions;
        u64 d_yield = a.prefill_yields - b->st.prefill_yields;
        u64 d_cancel = a.cancelled_running - b->st.cancelled_running;
        w->s5_checked++;
        if (!w->err &&
            d_rows + (d_pre - d_yield) + d_cancel != (u64)b->st.n_running)
            w->err = "S5: a runnable decode did not decode";
    }

    /* S3 -- FCFS: whatever left WAITING must have been its HEAD. */
    if (admitted && b->had_waiting && !w->err) {
        bool still;
        u64  now_head = sched_oldest_waiting(s, &still);
        if (still && now_head == b->oldest_waiting)
            w->err = "S3: admitted something other than the queue head";
    }
}

/* ---- the trace catalogue -----------------------------------------------
 * Seeded generators. A trace name means the same thing across sessions because
 * its parameters are fixed here, and the same thing across runs because its
 * arrivals come from the seed. */
typedef enum {
    TR_POISSON_MIX = 0,
    TR_BURST,
    TR_GIANT_SMALL,
    TR_CANCEL_STORM,
    TR_MAXTOK1,
    TR_PRESSURE,
    TR_AGENTIC,
    TR__COUNT
} TraceKind;

static const char* trace_name(TraceKind k) {
    switch (k) {
        case TR_POISSON_MIX:  return "poisson-mix";
        case TR_BURST:        return "burst";
        case TR_GIANT_SMALL:  return "giant+small";
        case TR_CANCEL_STORM: return "cancel-storm";
        case TR_MAXTOK1:      return "maxtok1";
        case TR_PRESSURE:     return "pressure";
        case TR_AGENTIC:      return "agentic";
        default:              return "?";
    }
}

#define NEVER 0xFFFFFFFFu
#define MAXREQ 24u
#define MAXGEN 256u
#define MAXPROMPT 320u

typedef struct {
    u64 id;
    u32 n_prompt;
    u32 max_tokens;
    u32 arrive_step;
    u32 cancel_step;
} Spec;

typedef struct {
    Spec spec[MAXREQ];
    u32  n;
    u32  capacity_blocks;
    u32  budget;
} Trace;

static void gen_trace(TraceKind k, u64 seed, Trace* t) {
    Rng rng = {seed * 0x9E3779B97F4A7C15ull + 12345u};
    memset(t, 0, sizeof(*t));
    for (u32 i = 0; i < MAXREQ; i++) t->spec[i].cancel_step = NEVER;

    switch (k) {
        case TR_POISSON_MIX:
            t->n = 10; t->capacity_blocks = 192; t->budget = 20000;
            for (u32 i = 0; i < t->n; i++) {
                t->spec[i].id = 1000 + i;
                t->spec[i].n_prompt = 24 + rnd(&rng, 180);
                t->spec[i].max_tokens = 8 + rnd(&rng, 48);
                /* Geometric gaps, so bursts happen by chance rather than by
                 * construction. */
                t->spec[i].arrive_step =
                    (i == 0) ? 0 : t->spec[i - 1].arrive_step + rnd(&rng, 12);
            }
            break;
        case TR_BURST:
            t->n = 12; t->capacity_blocks = 192; t->budget = 20000;
            for (u32 i = 0; i < t->n; i++) {
                t->spec[i].id = 2000 + i;
                t->spec[i].n_prompt = 32 + rnd(&rng, 32);
                t->spec[i].max_tokens = 6 + rnd(&rng, 10);
                t->spec[i].arrive_step = 0;
            }
            break;
        case TR_GIANT_SMALL:
            t->n = 10; t->capacity_blocks = 192; t->budget = 30000;
            t->spec[0].id = 3000;
            t->spec[0].n_prompt = MAXPROMPT - 1u; /* the giant */
            t->spec[0].max_tokens = 16;
            t->spec[0].arrive_step = 0;
            for (u32 i = 1; i < t->n; i++) {
                t->spec[i].id = 3000 + i;
                t->spec[i].n_prompt = 16 + rnd(&rng, 16);
                t->spec[i].max_tokens = 4 + rnd(&rng, 8);
                t->spec[i].arrive_step = 1 + rnd(&rng, 4);
            }
            break;
        case TR_CANCEL_STORM:
            t->n = 12; t->capacity_blocks = 192; t->budget = 20000;
            for (u32 i = 0; i < t->n; i++) {
                t->spec[i].id = 4000 + i;
                t->spec[i].n_prompt = 32 + rnd(&rng, 64);
                t->spec[i].max_tokens = 16 + rnd(&rng, 32);
                t->spec[i].arrive_step = rnd(&rng, 8);
                /* Uniformly random cancel points hit WAITING, PREFILLING and
                 * RUNNING without the trace having to name which. */
                t->spec[i].cancel_step = t->spec[i].arrive_step + rnd(&rng, 40);
            }
            break;
        case TR_MAXTOK1:
            t->n = 12; t->capacity_blocks = 128; t->budget = 10000;
            for (u32 i = 0; i < t->n; i++) {
                t->spec[i].id = 5000 + i;
                t->spec[i].n_prompt = 16 + rnd(&rng, 32);
                t->spec[i].max_tokens = 1;
                t->spec[i].arrive_step = rnd(&rng, 3);
            }
            break;
        case TR_PRESSURE:
            /* ▶ DELIBERATELY ADVERSARIAL, AND IT HAS TO BE. The watermark is
             * effective: admission demands blocks(prompt) + W with W's floor at
             * 8 blocks (128 positions), so a request whose whole life fits
             * inside that reserve can never outgrow what admission already
             * checked -- the pool refuses the next arrival instead, and short
             * requests under a small pool SERIALISE rather than thrash. A first
             * version of this trace never preempted once and would have banked
             * as a passing S6 test proving nothing. Preemption needs GROWTH
             * past the reserve: small prompts, large max_tokens, and a pool
             * that cannot hold them all at full stretch. */
            t->n = 6; t->capacity_blocks = 48; t->budget = 60000;
            for (u32 i = 0; i < t->n; i++) {
                t->spec[i].id = 6000 + i;
                t->spec[i].n_prompt = 24 + rnd(&rng, 16);
                t->spec[i].max_tokens = 180 + rnd(&rng, 60);
                t->spec[i].arrive_step = i * (1 + rnd(&rng, 3));
            }
            break;
        case TR_AGENTIC:
        default:
            /* Fable 2026-08-02, Kabir-approved: on a consumer box N=8/16 is
             * agent swarms, not humans. Bursty arrivals, a long shared system
             * prefix, short decodes, 10-20% aborted mid-generation. This is the
             * trace M5's bars should be read against. */
            t->n = 16; t->capacity_blocks = 320; t->budget = 30000;
            for (u32 i = 0; i < t->n; i++) {
                t->spec[i].id = 7000 + i;
                t->spec[i].n_prompt = 192 + rnd(&rng, 48);
                t->spec[i].max_tokens = 8 + rnd(&rng, 24);
                t->spec[i].arrive_step = (i / 4) * (1 + rnd(&rng, 3));
                if (rnd(&rng, 100) < 15) /* inside Fable's 10-20% */
                    t->spec[i].cancel_step =
                        t->spec[i].arrive_step + 2 + rnd(&rng, 25);
            }
            break;
    }
}

typedef struct {
    u32         n[MAXREQ];
    u32         ids[MAXREQ][MAXGEN];
    SchedFinish fin[MAXREQ];
} Collected;

/* Runs one trace to completion or until its step budget. `capacity_override` of
 * 0 uses the trace's own pool size; anything else replaces it, which is how
 * S6's infinite-pool reference run comes from the identical trace. */
static u32 run_trace(const Trace* t, int linear, u32 capacity_override,
                     Collected* out, Watch* w, SchedStats* final) {
    const u32 NL = 2, NH = 1, HD = 4, VOCAB = 97;
    const u32 MAX_CTX = MAXPROMPT + MAXGEN + 8u;
    u32       cap = capacity_override ? capacity_override : t->capacity_blocks;
    KvStore*  store = kv_store_open(NL, cap, NH, HD, KV_F32);

    Sim sim = {0.0, VOCAB, linear, 0};
    SchedHost host = {sim_step, sim_now, &sim};
    Sched* s = sched_open(store, MAX_CTX, &host);

    /* Prompts are BORROWED by the scheduler, so they must outlive the run. */
    static u32 prompts[MAXREQ][MAXPROMPT];
    for (u32 i = 0; i < t->n; i++)
        for (u32 p = 0; p < t->spec[i].n_prompt; p++)
            prompts[i][p] = (u32)((t->spec[i].id * 7 + p) % VOCAB);

    memset(out, 0, sizeof(*out));
    u32 step_i = 0;
    for (; step_i < t->budget; step_i++) {
        for (u32 i = 0; i < t->n; i++) {
            if (t->spec[i].arrive_step == step_i) {
                SchedRequest r = {t->spec[i].id, prompts[i], t->spec[i].n_prompt,
                                  t->spec[i].max_tokens, -1};
                if (!sched_submit(s, &r) && !w->err)
                    w->err = "submit refused a request the trace sized to fit";
            }
            if (t->spec[i].cancel_step == step_i)
                (void)sched_cancel(s, t->spec[i].id);
        }

        Before b;
        sched_stats(s, &b.st);
        b.oldest_waiting = sched_oldest_waiting(s, &b.had_waiting);
        b.head_prompt_blocks = 0;
        if (b.had_waiting) {
            for (u32 i = 0; i < t->n; i++) {
                if (t->spec[i].id == b.oldest_waiting) {
                    b.head_prompt_blocks = kv_blocks_for_ctx(t->spec[i].n_prompt);
                    break;
                }
            }
        }

        u32 rows = sched_step(s);
        watch_after(w, s, store, &b);

        /* Drain every ring every step: the consumer side must never be the
         * reason a stream looks short. */
        for (u32 i = 0; i < t->n; i++) {
            SchedFinish f;
            u32 got = sched_poll(s, t->spec[i].id, out->ids[i] + out->n[i],
                                 MAXGEN - out->n[i], &f);
            out->n[i] += got;
            if (f != SCHED_FINISH_NONE) out->fin[i] = f;
        }

        bool more = false;
        for (u32 i = 0; i < t->n; i++)
            if (t->spec[i].arrive_step > step_i) more = true;
        if (rows == 0 && !sched_pending(s) && !more) break;
    }

    sched_stats(s, final);
    sched_close(s);
    kv_store_close(store);
    return step_i;
}

/* ======================================================================== */

/* GATE ITEM [1]: all traces green, all asserts, 3 seeds x {constant, linear}. */
static void test_trace_matrix(void) {
    u64  total_pre = 0, total_steps = 0, s2 = 0, s5 = 0;
    bool all_green = true, all_finished = true;

    for (u32 k = 0; k < TR__COUNT; k++) {
        for (u64 seed = 1; seed <= 3; seed++) {
            for (int lin = 0; lin < 2; lin++) {
                Trace t;
                gen_trace((TraceKind)k, seed, &t);
                Watch      w = {NULL, 0, 0, 0};
                Collected  c;
                SchedStats st;
                u32        steps = run_trace(&t, lin, 0, &c, &w, &st);

                if (w.err) {
                    all_green = false;
                    printf("  FAIL: %s seed=%llu %s -- %s\n", trace_name((TraceKind)k),
                           (unsigned long long)seed, lin ? "linear" : "constant",
                           w.err);
                }
                /* STARVATION: every request reaches a terminal state, and the
                 * run must not have been cut off by its budget -- a trace that
                 * hits the budget is a livelock this check would otherwise
                 * report as success. */
                if (steps >= t.budget) {
                    all_finished = false;
                    printf("  FAIL: %s seed=%llu hit the step budget\n",
                           trace_name((TraceKind)k), (unsigned long long)seed);
                }
                for (u32 i = 0; i < t.n; i++) {
                    if (c.fin[i] == SCHED_FINISH_NONE) {
                        all_finished = false;
                        printf("  FAIL: %s seed=%llu req %u never finished\n",
                               trace_name((TraceKind)k), (unsigned long long)seed, i);
                        break;
                    }
                }
                total_pre += st.preemptions;
                total_steps += st.steps;
                s2 += w.s2_checked;
                s5 += w.s5_checked;
            }
        }
    }

    ok("matrix: every trace x 3 seeds x 2 cost models held every invariant",
       all_green);
    ok("matrix: every request in every run reached a terminal state", all_finished);
    /* ANTI-VACUITY ON THE ASSERTIONS THEMSELVES: an invariant that was never
     * evaluated is not an invariant that held. */
    ok("matrix: S2 was actually evaluated", s2 > 0);
    ok("matrix: S5 was actually evaluated on thousands of steps", s5 > 1000);
    printf("  matrix: %llu steps, %llu preemptions, S2 checked %llu, S5 %llu\n",
           (unsigned long long)total_steps, (unsigned long long)total_pre,
           (unsigned long long)s2, (unsigned long long)s5);
}

/* ▶▶ THE ONE THAT MATTERS: preempt-replay losslessness (S6).
 *
 * The pressure trace is run twice -- once against the pool it was sized to
 * starve, once against one large enough that preemption never happens -- and
 * the id streams must be IDENTICAL. That is the whole claim: recompute REPLAYS
 * recorded ids and never re-samples, so a preempted request's output cannot
 * depend on whether it was preempted.
 *
 * It is also the check that would be VACUOUS if preemption never fired, which
 * is why the counts are asserted both ways. Under M4's full-reservation
 * admission this test could not have been written at all. */
static void test_preempt_replay_is_lossless(void) {
    bool all_same = true, forced_every_seed = true, ref_clean = true;
    u64  total_pre = 0;

    for (u64 seed = 1; seed <= 3; seed++) {
        Trace t;
        gen_trace(TR_PRESSURE, seed, &t);

        Watch      wt = {NULL, 0, 0, 0}, wr = {NULL, 0, 0, 0};
        Collected  tight, ref;
        SchedStats st_t, st_r;
        run_trace(&t, 0, 0, &tight, &wt, &st_t);
        run_trace(&t, 0, 1024, &ref, &wr, &st_r); /* effectively infinite */

        if (wt.err || wr.err) {
            all_same = false;
            printf("  FAIL: pressure seed=%llu invariant -- %s\n",
                   (unsigned long long)seed, wt.err ? wt.err : wr.err);
        }
        if (st_t.preemptions == 0) forced_every_seed = false;
        if (st_r.preemptions != 0) ref_clean = false;
        total_pre += st_t.preemptions;

        for (u32 i = 0; i < t.n; i++) {
            if (tight.n[i] != ref.n[i] ||
                memcmp(tight.ids[i], ref.ids[i], tight.n[i] * sizeof(u32)) != 0) {
                all_same = false;
                printf("  FAIL: seed=%llu req %u: tight %u ids, ref %u ids\n",
                       (unsigned long long)seed, i, tight.n[i], ref.n[i]);
            }
        }
    }

    ok("replay: the tight pool FORCED preemptions on every seed", forced_every_seed);
    ok("replay: the reference pool needed none", ref_clean);
    ok("replay: id streams IDENTICAL to the infinite-pool run", all_same);
    if (!forced_every_seed)
        printf("    (a seed with no preemption proved nothing)\n");
    printf("  replay: %llu preemptions forced across 3 seeds\n",
           (unsigned long long)total_pre);
}

static void test_cost_model_cannot_change_a_decision(void) {
    /* The cost model advances a virtual clock and must not be able to change a
     * scheduling DECISION -- if it can, the sim's timings are load-bearing and
     * its traces stop being reproducible. */
    bool same = true;
    for (u32 k = 0; k < TR__COUNT; k++) {
        Trace t;
        gen_trace((TraceKind)k, 2, &t);
        Watch      w0 = {NULL, 0, 0, 0}, w1 = {NULL, 0, 0, 0};
        Collected  c0, c1;
        SchedStats s0, s1;
        run_trace(&t, 0, 0, &c0, &w0, &s0);
        run_trace(&t, 1, 0, &c1, &w1, &s1);
        if (s0.steps != s1.steps || s0.preemptions != s1.preemptions) same = false;
        for (u32 i = 0; i < t.n; i++) {
            if (c0.n[i] != c1.n[i] ||
                memcmp(c0.ids[i], c1.ids[i], c0.n[i] * sizeof(u32)) != 0)
                same = false;
        }
    }
    ok("cost model: constant and linear give identical outputs and step counts",
       same);
}

static void test_cancel_returns_every_block(void) {
    /* S8's teardown half. A leak here is invisible to every other check in this
     * file, because the pool only runs out under pressure and the other traces
     * are not sized to notice. */
    bool whole = true, retired = true;
    for (u64 seed = 1; seed <= 3; seed++) {
        Trace t;
        gen_trace(TR_CANCEL_STORM, seed, &t);
        Watch      w = {NULL, 0, 0, 0};
        Collected  c;
        SchedStats st;
        run_trace(&t, 0, 0, &c, &w, &st);
        if (st.free_blocks != t.capacity_blocks) whole = false;
        if (st.finished != t.n) retired = false;
    }
    ok("cancel: every block came back once everything retired", whole);
    ok("cancel: every request retired", retired);
}

/* A request that could never clear the watermark is refused at SUBMIT, not
 * queued forever. Without this the step loop returns 0 rows while
 * sched_pending stays true -- which a driver reads as "busy" and spins on, so
 * the failure is a hang rather than an error. */
static void test_unadmittable_is_refused_at_the_door(void) {
    const u32 NL = 2, NH = 1, HD = 4;
    KvStore*  store = kv_store_open(NL, 12, NH, HD, KV_F32); /* 12 blocks */
    Sim       sim = {0.0, 97, 0, 0};
    SchedHost host = {sim_step, sim_now, &sim};
    Sched*    s = sched_open(store, 4096, &host);

    static u32 prompt[1024];
    for (u32 i = 0; i < 1024; i++) prompt[i] = i % 97;

    /* blocks(80) = 5, + W floor 8 = 13 > 12 capacity: never admissible. */
    SchedRequest too_big = {1, prompt, 80, 4, -1};
    ok("a prompt that can never clear the watermark is refused",
       sched_submit(s, &too_big) == false);

    SchedRequest fits = {2, prompt, 48, 4, -1}; /* blocks 3 + 8 = 11 <= 12 */
    ok("one that can is accepted", sched_submit(s, &fits) == true);

    SchedRequest tall = {3, prompt, 16, 400, -1};
    ok("a ceiling past the whole pool is refused", sched_submit(s, &tall) == false);
    ok("duplicate ids are refused", sched_submit(s, &fits) == false);

    u32 guard = 0;
    while (sched_pending(s) && guard++ < 1000) sched_step(s);
    SchedFinish f = SCHED_FINISH_NONE;
    u32         out[8];
    u32         got = sched_poll(s, 2, out, 8, &f);
    ok("the accepted request completed", got == 4 && f == SCHED_FINISH_MAX_TOKENS);
    ok("...without spinning out the guard", guard < 1000);

    sched_close(s);
    kv_store_close(store);
}

/* THE SOAK: the gate's x1e6 floor, as STEPS. Every invariant after every one of
 * them -- not sampled, because a violation that appears once in 400 steps and
 * is checked every 500 does not exist as far as this harness is concerned. */
static void soak(u64 want_steps) {
    u64  done = 0, seed = 1, preemptions = 0, traces = 0;
    bool green = true;

    while (done < want_steps && green) {
        TraceKind k = (TraceKind)(seed % TR__COUNT);
        Trace     t;
        gen_trace(k, seed, &t);
        Watch      w = {NULL, 0, 0, 0};
        Collected  c;
        SchedStats st;
        run_trace(&t, (int)(seed & 1u), 0, &c, &w, &st);
        if (w.err) {
            green = false;
            printf("  FAIL: soak seed=%llu %s -- %s\n",
                   (unsigned long long)seed, trace_name(k), w.err);
        }
        done += st.steps;
        preemptions += st.preemptions;
        traces++;
        seed++;
    }
    ok("soak: every invariant held after every step", green);
    printf("  soak: %llu steps over %llu traces, %llu preemptions\n",
           (unsigned long long)done, (unsigned long long)traces,
           (unsigned long long)preemptions);
}

int main(int argc, char** argv) {
    /* THE GATE'S FLOOR IS THE DEFAULT, not an opt-in. 1e6 steps with every
     * invariant checked after every one of them costs ~1.2 s here, which is
     * cheaper than the argument for running less would be -- and a gate item
     * that only holds when someone remembers to pass a flag is a gate item that
     * will eventually be reported green without having run. */
    u64 want = 1000000;
    if (argc > 1) want = strtoull(argv[1], NULL, 10);

    test_unadmittable_is_refused_at_the_door();
    test_trace_matrix();
    test_preempt_replay_is_lossless();
    test_cost_model_cannot_change_a_decision();
    test_cancel_returns_every_block();
    soak(want);

    printf("test_sched: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
