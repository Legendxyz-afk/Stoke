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

/* Trace bounds. UP HERE because the sim's position ledger is sized from them
 * and it is declared before the trace catalogue. */
#define NEVER 0xFFFFFFFFu
#define MAXREQ 24u
#define MAXGEN 256u
#define MAXPROMPT 320u

/* xorshift, so every trace is reproducible from its seed. */
typedef struct { u64 s; } Rng;
static u32 rnd(Rng* r, u32 n) {
    r->s ^= r->s << 13; r->s ^= r->s >> 7; r->s ^= r->s << 17;
    return (u32)(r->s % (n ? n : 1u));
}

/* ---- the virtual clock and the step-cost model -------------------------
 *
 * ▶▶ AND A POSITION LEDGER, WHICH EXISTS BECAUSE A MUTATION PROVED THE OBVIOUS
 * CHECK VACUOUS. The natural test of prefix adoption is "the id streams are
 * identical with and without it" -- and on 2026-08-25 a mutant that skipped ONE
 * BLOCK MORE than it adopted PASSED that test. It has to: `fake_token` depends
 * on (request id, position), a request's decode positions start at n_prompt
 * however its prefill got there, so NOTHING the prefill does can move a sim
 * stream. The comparison was measuring a quantity that cannot change.
 *
 * So the sim records WHICH POSITIONS ACTUALLY GOT A ROW. That is the statement
 * adoption is really making -- "these positions are already resident, do not
 * compute them" -- and its failure mode is a HOLE: a position no row ever
 * covered and no adoption ever claimed, whose K/V is therefore whatever was in
 * the block. Tracked only for traces that set `shared_prefix`, so every other
 * trace (and the soak) pays nothing. */
#define SIM_MAX_CTX (MAXPROMPT + MAXGEN + 8u)

typedef struct {
    double clock_ms;
    u32    vocab;
    int    linear; /* 0 = constant cost, 1 = linear in batch and context */
    u64    steps;
    int    track;  /* record the position ledger below                    */
    u32    n_rid;
    u64    rid[MAXREQ];
    u8     cov[MAXREQ][SIM_MAX_CTX];
} Sim;

static u32 sim_slot(Sim* h, u64 req_id) {
    for (u32 i = 0; i < h->n_rid; i++)
        if (h->rid[i] == req_id) return i;
    if (h->n_rid >= MAXREQ) return MAXREQ; /* caller ignores; MAXREQ is the trace cap */
    h->rid[h->n_rid] = req_id;
    return h->n_rid++;
}

static double sim_now(void* ctx) { return ((Sim*)ctx)->clock_ms; }

static void sim_step(void* ctx, const SchedStep* in, u32* out) {
    Sim* h = (Sim*)ctx;

    /* ▶ THE FAKE MODEL RECORDS WHICH TOKEN SITS AT EACH POSITION, exactly as
     * model_forward_batch does (model.c, `if (L == 0) kv_note_token(...)`).
     *
     * WITHOUT THIS THE SIM CANNOT SEE PREFIX DEDUP AT ALL: a block's ids are
     * what make it hashable, a hash is what makes it findable, and blocks grown
     * by bp_seq_ensure alone carry hash 0 and are storage-only. So a dedup test
     * driven by a host that never noted a token would report "nothing was
     * adopted" for a reason that has nothing to do with the scheduler.
     *
     * It is deliberately UNCONDITIONAL rather than a sim option: the real host
     * always notes, so a sim that only sometimes notes is a sim of a different
     * engine. With dedup off nothing ever looks the ids up, which is why every
     * trace banked before this line behaves identically after it. */
    for (u32 i = 0; i < in->n_rows; i++)
        kv_note_token(in->kv[i], in->positions[i], in->token_ids[i]);

    if (h->track) {
        for (u32 i = 0; i < in->n_rows; i++) {
            u32 sl = sim_slot(h, in->req_ids[i]);
            if (sl < MAXREQ && in->positions[i] < SIM_MAX_CTX)
                h->cov[sl][in->positions[i]] = 1u;
        }
    }

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
    u64         s4_checked;
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

    /* S4 -- PUSH-FRONT: a preemption victim goes to the HEAD of WAITING.
     *
     * That is the half of the victim rule an outside observer can check, and it
     * is the half the starvation argument actually rests on: the victim
     * re-admits before any newer arrival, so preemption costs it latency and
     * never its place in the queue. (The other half -- WHICH request is chosen
     * -- is not observable from here, and this file deliberately does not reach
     * into sched.c to find out.) */
    u64 d_pre_s4 = a.preemptions - b->st.preemptions;
    if (d_pre_s4 > 0 && !w->err) {
        bool had_v, has_head;
        u64  victim = sched_last_preempted(s, &had_v);
        u64  head = sched_oldest_waiting(s, &has_head);
        w->s4_checked++;
        if (had_v && has_head && head != victim)
            w->err = "S4: a preempted request is not at the head of WAITING";
    }

    /* S7 -- S-ALIGN, counted by the scheduler and asserted here. */
    if (!w->err && a.chunk_misaligned != 0)
        w->err = "S7: a prefill chunk ended off a block boundary";
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
    /* Leading prompt positions that are IDENTICAL across every request in the
     * trace -- the fanout shape gate item [4] measures. 0 for every trace
     * generated by gen_trace, so those keep the prompts they were banked with:
     * a shared prefix is a property of the WORKLOAD, and changing what an
     * existing trace's requests say would change what its banked result meant.
     * Must be a multiple of the KV block size to be adoptable at all. */
    u32  shared_prefix;
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

/* What the sim's position ledger is asked, per request: the positions that got a
 * row form a CONTIGUOUS run, and the uncovered prefix below it -- the `gap` --
 * is exactly what adoption claimed was already resident. An `interior_hole` is
 * the defect this exists to catch: a position no row computed and no adoption
 * covered, whose K/V on the real path is whatever the block happened to hold. */
typedef struct {
    u32  n;
    u64  rid[MAXREQ];
    u32  gap[MAXREQ];
    u32  last[MAXREQ];
    bool interior_hole;
} Cover;

/* Runs one trace to completion or until its step budget. `capacity_override` of
 * 0 uses the trace's own pool size; anything else replaces it, which is how
 * S6's infinite-pool reference run comes from the identical trace. */
static u32 run_trace(const Trace* t, int linear, u32 capacity_override, int dedup,
                     Collected* out, Watch* w, SchedStats* final, Cover* cov_out) {
    const u32 NL = 2, NH = 1, HD = 4, VOCAB = 97;
    const u32 MAX_CTX = SIM_MAX_CTX;
    u32       cap = capacity_override ? capacity_override : t->capacity_blocks;
    KvStore*  store = kv_store_open(NL, cap, NH, HD, KV_F32);

    /* static because the position ledger is 14 KB; one trace runs at a time.
     * ▶ AND THE LEDGER IS CLEARED ONLY WHEN IT IS USED. Clearing all 14 KB per
     * trace cost the 1e6-step soak 5x its wall clock (6.3 s against 1.2 s) --
     * the soak runs ~12,000 traces per million steps, and gate [5] is a
     * WALL-CLOCK budget, so a slower sim buys fewer traces for the same hour.
     * Statics start zeroed, and no trace without a shared prefix ever writes
     * here. */
    static Sim sim;
    sim.clock_ms = 0.0;
    sim.steps = 0;
    sim.n_rid = 0;
    sim.vocab = VOCAB;
    sim.linear = linear;
    sim.track = (t->shared_prefix != 0u);
    if (sim.track) memset(sim.cov, 0, sizeof(sim.cov));
    SchedHost host = {sim_step, sim_now, &sim};
    Sched* s = sched_open(store, MAX_CTX, &host, dedup != 0);

    /* Prompts are BORROWED by the scheduler, so they must outlive the run. */
    static u32 prompts[MAXREQ][MAXPROMPT];
    for (u32 i = 0; i < t->n; i++)
        for (u32 p = 0; p < t->spec[i].n_prompt; p++)
            prompts[i][p] = (p < t->shared_prefix)
                                ? (u32)((p * 31u + 7u) % VOCAB)
                                : (u32)((t->spec[i].id * 7 + p) % VOCAB);

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

    if (cov_out) {
        memset(cov_out, 0, sizeof(*cov_out));
        cov_out->n = sim.n_rid;
        for (u32 i = 0; i < sim.n_rid; i++) {
            cov_out->rid[i] = sim.rid[i];
            u32  first = 0, last = 0;
            bool any = false;
            for (u32 p = 0; p < SIM_MAX_CTX; p++) {
                if (!sim.cov[i][p]) continue;
                if (!any) { first = p; any = true; }
                last = p;
            }
            if (!any) continue;
            cov_out->gap[i] = first;
            cov_out->last[i] = last;
            for (u32 p = first; p <= last; p++)
                if (!sim.cov[i][p]) cov_out->interior_hole = true;
        }
    }

    sched_close(s);
    kv_store_close(store);
    return step_i;
}

/* ======================================================================== */

/* GATE ITEM [1]: all traces green, all asserts, 3 seeds x {constant, linear}. */
static void test_trace_matrix(void) {
    u64  total_pre = 0, total_steps = 0, s2 = 0, s4 = 0, s5 = 0;
    bool all_green = true, all_finished = true;

    for (u32 k = 0; k < TR__COUNT; k++) {
        for (u64 seed = 1; seed <= 3; seed++) {
            for (int lin = 0; lin < 2; lin++) {
                Trace t;
                gen_trace((TraceKind)k, seed, &t);
                Watch      w = {NULL, 0, 0, 0, 0};
                Collected  c;
                SchedStats st;
                u32        steps = run_trace(&t, lin, 0, 0, &c, &w, &st, NULL);

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
                s4 += w.s4_checked;
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
    ok("matrix: S4 was actually evaluated", s4 > 0);
    printf("  matrix: %llu steps, %llu preemptions; checked S2 %llu, S4 %llu, "
           "S5 %llu\n",
           (unsigned long long)total_steps, (unsigned long long)total_pre,
           (unsigned long long)s2, (unsigned long long)s4,
           (unsigned long long)s5);
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

        Watch      wt = {NULL, 0, 0, 0, 0}, wr = {NULL, 0, 0, 0, 0};
        Collected  tight, ref;
        SchedStats st_t, st_r;
        run_trace(&t, 0, 0, 0, &tight, &wt, &st_t, NULL);
        run_trace(&t, 0, 1024, 0, &ref, &wr, &st_r, NULL); /* effectively infinite */

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

/* ▶▶ GATE ITEM [4]'s PRECONDITION: PREFIX ADOPTION ON THE SERVE PATH.
 *
 * S3 is eight clients sharing a 1024-token system prompt. Without adoption the
 * scheduler prefills that prefix eight times, which is not what the 13.3 bar was
 * measured against -- llama-server's defaults have slot-prompt-similarity and
 * the prompt cache ON (memo 2026-08-24-m5-gate4-s3-status.md section 4). This is
 * that shape in miniature: 8 requests, a 128-token shared prefix (8 whole KV
 * blocks), 24 private tokens each.
 *
 * ▶ WHY THE TRACE IS NOT A TraceKind. The matrix, the cost-model test and the
 * soak all sweep `k < TR__COUNT`, so adding a member would change WHICH TRACES
 * gate [1] and gate [5] ran -- and both are banked results. This trace is
 * addressed by name from this test only; the counted set is untouched, and
 * test_sched's default path still reproduces gate [1] exactly.
 *
 * ▶▶ WHAT THE SIM CAN AND CANNOT SEE, MEASURED RATHER THAN ASSUMED. The first
 * version of this test asserted only "the id streams are identical with and
 * without adoption", and a mutant that skipped ONE BLOCK MORE than it adopted
 * PASSED it (2026-08-25, `build/win-dedup-mutant`). It had to: fake_token
 * depends on (request id, position), a request's decode positions start at
 * n_prompt however its prefill got there, so no prefill-side defect can move a
 * sim stream. The stream comparison here is therefore kept for what it does
 * cover -- a request ending its prefill at the wrong n_done -- and the load is
 * carried by the POSITION LEDGER (see Sim): every position must have been
 * covered by a row or claimed by adoption, with no hole between.
 *
 * What remains out of reach here is whether the ADOPTED positions hold the right
 * K/V, because the sim has no K/V at all. That is the real-model byte-compare's
 * job (tests/tooling/run_m5_dedup_verify.ps1), and the same mutant is the proof
 * of power there. */
#define FANOUT_N      8u
#define FANOUT_SHARED 128u  /* 8 whole blocks at BP_BLOCK_TOKENS = 16 */

static void gen_fanout_trace(Trace* t, u32 n, u32 capacity_blocks,
                             u32 max_tokens) {
    memset(t, 0, sizeof(*t));
    for (u32 i = 0; i < MAXREQ; i++) t->spec[i].cancel_step = NEVER;
    t->n = n;
    t->capacity_blocks = capacity_blocks;
    t->budget = 60000;
    t->shared_prefix = FANOUT_SHARED;
    for (u32 i = 0; i < t->n; i++) {
        t->spec[i].id = 8000 + i;
        t->spec[i].n_prompt = FANOUT_SHARED + 24u;
        t->spec[i].max_tokens = max_tokens;
        t->spec[i].arrive_step = 0; /* all at once: S3 is a fanout, not a queue */
    }
}

static bool streams_same(const Trace* t, const Collected* a, const Collected* b) {
    for (u32 i = 0; i < t->n; i++) {
        if (a->n[i] != b->n[i]) return false;
        if (memcmp(a->ids[i], b->ids[i], a->n[i] * sizeof(u32)) != 0) return false;
        if (a->fin[i] != b->fin[i]) return false;
    }
    return true;
}

static void test_prefix_adoption_on_the_serve_path(void) {
    Trace t;
    gen_fanout_trace(&t, FANOUT_N, 192u, 24u);

    Watch      woff = {NULL, 0, 0, 0, 0}, won = {NULL, 0, 0, 0, 0};
    Collected  coff, con;
    SchedStats soff, son;
    Cover      voff, von;
    run_trace(&t, 0, 0, 0, &coff, &woff, &soff, &voff);
    run_trace(&t, 0, 0, 1, &con, &won, &son, &von);

    ok("adopt: every invariant held with dedup OFF", woff.err == NULL);
    ok("adopt: every invariant held with dedup ON", won.err == NULL);
    if (woff.err) printf("    off: %s\n", woff.err);
    if (won.err) printf("    on:  %s\n", won.err);

    /* THE FLAG IS THE ONLY THING THAT DIFFERS, so the off arm is the control
     * that says the counter measures adoption rather than admissions. */
    ok("adopt: dedup OFF adopts nothing", soff.adopted_positions == 0);

    /* EXACT, not a bound. One prefill is active at a time (S1), so by the time
     * client k is admitted client k-1's whole prefix is written and hashed --
     * every client but the first adopts all 8 blocks. A bound here would stay
     * true if adoption silently degraded to one block. */
    ok("adopt: 7 of 8 clients adopted the whole 128-token prefix",
       son.adopted_positions == (u64)(FANOUT_N - 1u) * FANOUT_SHARED);
    printf("  adopt: %llu of %llu admitted prompt positions were resident\n",
           (unsigned long long)son.adopted_positions,
           (unsigned long long)son.admitted_prompt_positions);

    /* The point of the exercise: work not done. */
    ok("adopt: adoption cost strictly fewer prefill chunks",
       son.prefill_chunks < soff.prefill_chunks);

    /* THE CLAIM, HALF ONE: dedup does not move the stream. Kept, but see the
     * header -- this half cannot see a prefill-side skip. */
    ok("adopt: every id stream is IDENTICAL with and without adoption",
       streams_same(&t, &coff, &con));

    /* THE CLAIM, HALF TWO -- THE ONE WITH POWER OVER THIS CHANGE.
     *
     * Without adoption every position from 0 up must have been covered by a row.
     * With it, exactly the adopted prefix may be uncovered, and NOTHING ELSE:
     * one uncovered position above the prefix is a position the engine will
     * attend to and never wrote. Both are stated per request, and the gaps are
     * checked to sum to what the scheduler REPORTED adopting -- a counter that
     * agrees with itself is not evidence; a counter that agrees with the rows
     * actually stepped is. */
    bool cov_off_ok = (voff.interior_hole == false), cov_on_ok = !von.interior_hole;
    u64  gap_sum = 0;
    for (u32 i = 0; i < voff.n; i++)
        if (voff.gap[i] != 0) cov_off_ok = false;
    for (u32 i = 0; i < von.n; i++) {
        gap_sum += von.gap[i];
        if (von.gap[i] != 0 && von.gap[i] != FANOUT_SHARED) cov_on_ok = false;
    }
    ok("adopt: with dedup OFF every position from 0 was computed", cov_off_ok);
    ok("adopt: with dedup ON the uncovered run is exactly the adopted prefix",
       cov_on_ok);

    /* THE LEDGER SAW THE WHOLE RUN, not just its beginning. Every request's
     * highest covered position must be its LAST decode row: prefill computes
     * up to n_prompt-1 and its final row yields id 0, so max_tokens ids means a
     * last row at n_prompt + max_tokens - 2. Without this the two checks above
     * would still pass over a ledger that stopped recording halfway. */
    bool cov_end_ok = (von.n == FANOUT_N);
    u32  want_last = t.spec[0].n_prompt + t.spec[0].max_tokens - 2u;
    for (u32 i = 0; i < von.n; i++)
        if (von.last[i] != want_last) cov_end_ok = false;
    ok("adopt: every request's rows run to its last decode position", cov_end_ok);
    ok("adopt: no request has a HOLE between its adopted prefix and its rows",
       von.interior_hole == false);
    ok("adopt: the uncovered positions sum to what the scheduler reported",
       gap_sum == son.adopted_positions);
    if (gap_sum != son.adopted_positions)
        printf("    ledger says %llu uncovered, stats say %llu adopted\n",
               (unsigned long long)gap_sum,
               (unsigned long long)son.adopted_positions);

    /* ---- CONTROL: a trace with NO shared prefix must adopt nothing even with
     * the flag on, and must produce the same stream either way. This is what
     * separates "adoption fired" from "the flag perturbed the scheduler". */
    Trace p;
    gen_trace(TR_POISSON_MIX, 2, &p);
    Watch      wp0 = {NULL, 0, 0, 0, 0}, wp1 = {NULL, 0, 0, 0, 0};
    Collected  cp0, cp1;
    SchedStats sp0, sp1;
    run_trace(&p, 0, 0, 0, &cp0, &wp0, &sp0, NULL);
    run_trace(&p, 0, 0, 1, &cp1, &wp1, &sp1, NULL);
    ok("adopt: an unshared workload adopts nothing with the flag ON",
       sp1.adopted_positions == 0);
    ok("adopt: and its streams and step counts are unchanged",
       streams_same(&p, &cp0, &cp1) && sp0.steps == sp1.steps &&
           wp1.err == NULL);

    /* ---- UNDER PRESSURE: S6's shape with adoption as a second axis.
     *
     * Preemption frees a victim's blocks and resets its n_done to 0, and
     * kv_adopt_prefix is ADMISSION-TIME ONLY -- so re-adoption depends on the
     * readmission path opening a FRESH Kv. If it does not, this run either
     * adopts nothing after the first preemption or diverges.
     *
     * ▶ THE SHAPE HAD TO CHANGE TO STARVE THE POOL, and this is TR_PRESSURE's
     * recorded lesson applied a second time: the watermark admits on
     * blocks(prompt) + W, so requests whose whole life fits inside that reserve
     * SERIALISE instead of thrashing. The first version of this arm (24 gen, 64
     * blocks) preempted ZERO times and would have banked as a passing test of
     * nothing. Preemption needs GROWTH past the reserve: 180 generated against a
     * 48-block pool that cannot hold six of them at full stretch. */
    Trace tight;
    gen_fanout_trace(&tight, 6u, 48u, 180u);
    Watch      wt = {NULL, 0, 0, 0, 0}, wref = {NULL, 0, 0, 0, 0};
    Collected  ct, cref;
    SchedStats stt, stref;
    Cover      vt;
    u32        steps = run_trace(&tight, 0, 0, 1, &ct, &wt, &stt, &vt);
    run_trace(&tight, 0, 1024, 0, &cref, &wref, &stref, NULL); /* roomy, no dedup */
    ok("adopt: invariants held under a pool too small for the fanout",
       wt.err == NULL && wref.err == NULL);
    if (wt.err) printf("    tight: %s\n", wt.err);
    ok("adopt: the tight pool FORCED preemptions (else this proves nothing)",
       stt.preemptions > 0);
    ok("adopt: the reference pool needed none", stref.preemptions == 0);
    ok("adopt: it still finished inside its budget", steps < tight.budget);
    ok("adopt: adoption still fired after preemptions", stt.adopted_positions > 0);
    ok("adopt: and its streams match the roomy no-dedup reference",
       streams_same(&tight, &cref, &ct));
    /* THE LEDGER UNDER REPLAY. A preempted request recomputes from wherever
     * re-admission left it, so the union of its coverage across admissions may
     * start at 0 or at a re-adopted prefix -- but a HOLE is a defect in either
     * case, and this is the arm where re-adoption could leave one. */
    ok("adopt: no HOLE survives preemption and re-adoption", vt.interior_hole == false);
    printf("  adopt: tight pool -- %llu preemptions, %llu of %llu positions "
           "adopted\n",
           (unsigned long long)stt.preemptions,
           (unsigned long long)stt.adopted_positions,
           (unsigned long long)stt.admitted_prompt_positions);
}

static void test_cost_model_cannot_change_a_decision(void) {
    /* The cost model advances a virtual clock and must not be able to change a
     * scheduling DECISION -- if it can, the sim's timings are load-bearing and
     * its traces stop being reproducible. */
    bool same = true;
    for (u32 k = 0; k < TR__COUNT; k++) {
        Trace t;
        gen_trace((TraceKind)k, 2, &t);
        Watch      w0 = {NULL, 0, 0, 0, 0}, w1 = {NULL, 0, 0, 0, 0};
        Collected  c0, c1;
        SchedStats s0, s1;
        run_trace(&t, 0, 0, 0, &c0, &w0, &s0, NULL);
        run_trace(&t, 1, 0, 0, &c1, &w1, &s1, NULL);
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
        Watch      w = {NULL, 0, 0, 0, 0};
        Collected  c;
        SchedStats st;
        run_trace(&t, 0, 0, 0, &c, &w, &st, NULL);
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
    static Sim sim;
    memset(&sim, 0, sizeof(sim));
    sim.vocab = 97;
    SchedHost host = {sim_step, sim_now, &sim};
    Sched*    s = sched_open(store, 4096, &host, false);

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
/* `seed0` EXISTS SO A MULTI-HOUR SOAK CAN BANK AND RESUME. The seed walk is
 * deterministic (seed 1, 2, 3, ... each generating a different trace), so a
 * longer run explores STRICTLY MORE traces rather than repeating; but this
 * function prints only when it finishes, and gate item [5] is a 1 h run. An
 * unattended job that banks nothing until it completes is the shape that cost
 * this project 76 minutes of probe compute on 2026-08-12. The runner therefore
 * calls this in chunks and records the next seed after each one, which also
 * makes a failure REPRODUCIBLE IN SECONDS: the printed seed is the whole input.
 * Defaults are unchanged (seed0 = 1), so the per-commit suite and gate [1]'s
 * banked 1,000,130-step result are byte-for-byte the same run. */
static void soak(u64 want_steps, u64 seed0) {
    u64  done = 0, seed = seed0, preemptions = 0, traces = 0;
    bool green = true;

    while (done < want_steps && green) {
        TraceKind k = (TraceKind)(seed % TR__COUNT);
        Trace     t;
        gen_trace(k, seed, &t);
        Watch      w = {NULL, 0, 0, 0, 0};
        Collected  c;
        SchedStats st;
        run_trace(&t, (int)(seed & 1u), 0, 0, &c, &w, &st, NULL);
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
    /* The resume point, machine-readable for the chunked runner. Printed even
     * on failure: a red chunk's NEXT seed is what a re-run must skip past, and
     * the FAIL line above already carries the offending seed itself. */
    printf("  soak: seeds [%llu,%llu) next=%llu\n",
           (unsigned long long)seed0, (unsigned long long)seed,
           (unsigned long long)seed);
}

int main(int argc, char** argv) {
    /* THE GATE'S FLOOR IS THE DEFAULT, not an opt-in. 1e6 steps with every
     * invariant checked after every one of them costs ~1.2 s here, which is
     * cheaper than the argument for running less would be -- and a gate item
     * that only holds when someone remembers to pass a flag is a gate item that
     * will eventually be reported green without having run. */
    u64 want = 1000000;
    u64 seed0 = 1;
    if (argc > 1) want = strtoull(argv[1], NULL, 10);
    if (argc > 2) seed0 = strtoull(argv[2], NULL, 10);
    if (seed0 == 0) {
        /* seed 0 would make gen_trace's `seed % TR__COUNT` pick trace 0 with a
         * zero seed and, worse, would silently look like "resume from the
         * start" to a runner whose bank file was empty or unparsed. Refuse it
         * rather than run a soak that is not the one anyone asked for. */
        printf("test_sched: seed0 must be >= 1\n");
        return 1;
    }

    test_unadmittable_is_refused_at_the_door();
    test_trace_matrix();
    test_preempt_replay_is_lossless();
    test_prefix_adoption_on_the_serve_path();
    test_cost_model_cannot_change_a_decision();
    test_cancel_returns_every_block();
    soak(want, seed0);

    printf("test_sched: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
