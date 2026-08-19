# M5 S0 — the sim harness architecture: what it replaces, and what it must not

*2026-08-19, Opus 5. Written before `sched.c` exists, per the ROADMAP's M5
mandate that the simulation harness is built FIRST. Binds: the substitution
boundary (§1), the seam `sched.c` is written against (§2), the assertion
schedule and its cost budget (§5), and what the sim is not evidence for (§7).*

**The scheduler design itself is LOCKED and is not re-opened here** — states,
step loop, watermark `W = max(8, 2·|RUNNING|)`, preempt-youngest/push-front,
replay-recorded-ids, chunk 128 — see the forward note §M5 and the entry note
`2026-08-19-m5-entry.md` §2 for the asserted set S1–S8. What is *undecided*,
and what this note decides, is how a simulation drives that scheduler without
becoming a simulation of a different scheduler.

---

## 1. THE SUBSTITUTION BOUNDARY

**The sim replaces the MODEL and the CLOCK. It never replaces the SCHEDULER or
the BLOCK POOL.**

This is the whole architecture, and getting it wrong is the failure mode that
makes every number downstream worthless: a harness that reimplements the
scheduler's state machine in test code proves that the *test's* scheduler is
correct. M4's invariant sim already got this right by running ×1e6 ops against
the **real** `BlockPool`; S0 extends the same discipline one layer up.

| component | in the sim | why |
|---|---|---|
| `sched.c` | **REAL** | it is the thing under test |
| `BlockPool` / `Kv` | **REAL** | I1–I11 are statements about *this* allocator; against a fake pool they assert nothing |
| the model forward | **fake** | a step must cost microseconds, not 100 ms, or 1e6 steps is a week |
| the clock | **virtual** (and swappably real, §4) | a trace with 60 s of arrivals must run in milliseconds |
| arrival generation | seeded, deterministic | a trace that cannot be replayed cannot be bisected |

**Consequence, stated because it constrains `sched.c`'s design:** the scheduler
must not call `model_forward_batch` or `now_ms()` directly. Both arrive through
the seam in §2. That is a small amount of indirection bought for the ability to
assert eight scheduler invariants a million times.

## 2. THE SEAM

```c
typedef struct {
    /* Executes ONE fused step: the runnable decodes plus at most one prefill
     * chunk. Returns the sampled id per decode row. The real implementation
     * calls model_forward_batch and samples; the sim's returns
     * fake_token(req_id, pos) and touches no weights. */
    void (*step)(void* ctx, const StepBatch* in, u32* out_ids);
    /* Monotonic milliseconds. Real: now_ms(). Sim: the virtual clock. */
    double (*now_ms)(void* ctx);
    void* ctx;
} SchedHost;
```

**Two functions, and deliberately not more.** Every additional hook is another
thing the sim can get right while the real host gets it wrong. In particular
the pool is NOT behind this seam — the scheduler holds a real `KvStore` in both
modes, which is exactly what lets a sim step assert I1–I11.

**`step` takes the whole batch, not one sequence**, because the fused step *is*
the design (§M5: "extra rows in the same op walk, not a separate pass"). A
per-sequence hook would let a sim pass while the real implementation quietly
ran N separate forwards — the thing the co-execution requirement exists to
prevent.

## 3. THE FAKE MODEL

`next_id = hash(req_id, pos) % n_vocab`, deterministic, instant, no weights.

**Why a hash and not a counter:** the preempt-replay invariant (S6) needs
outputs that depend on *identity and position*, so that a request re-prefilled
after preemption must reproduce the same ids or the comparison is vacuous. A
counter would make every request's stream identical and S6 would hold for the
wrong reason.

**What the fake model cannot see, stated so nobody reads the sim as broader
evidence than it is:** it cannot catch an arithmetic error, a KV addressing
error, or a batch-composition dependence. Those are the byte-compare gates'
job (`run_m5_entry_verify.ps1`, `m4_batch_identity.py`). The sim's subject is
*scheduling*: admission, ordering, preemption, teardown, starvation.

## 4. THE CLOCK, AND THE ONE RUN THAT USES THE REAL ONE

Virtual by default: the step-cost model advances it, so a trace spanning
minutes of simulated arrivals runs in milliseconds.

Two cost models, both required by the gate ([1]: 3 seeds × {constant, linear}):
- **constant** — every step costs `C` ms. Finds ordering bugs without physics.
- **linear** — `C0 + a·|batch| + b·Σctx`, with `a` and `b` seeded from the
  measured curve (entry note §3: intercept 63.40 ms, slope 0.0289
  ms/context-token *pre-E2*, 0.00722 *post*). **Use the post-E2 slope**; a cost
  model calibrated to an engine we no longer ship would mis-price exactly the
  regime M5 operates in.

**And one run under the REAL clock**, per the forward note's attribution ladder
item (2): the fake model with `now_ms()` live, measuring the scheduler's own
bookkeeping. **Budget: <2% of a real step (~50 ms class), i.e. <1 ms of
bookkeeping per step over ≤16 requests.** It is bookkeeping over a handful of
lists; it has no excuse not to be. This is why `now_ms` is behind the seam
rather than compiled out.

## 5. THE ASSERTION SCHEDULE, AND WHY IT IS AFFORDABLE

After **every** step: M4's I1–I11 (the pool's structural set, `2026-08-11-m4-
invariants.md` §2.3) **plus** S1–S8 (entry note §2).

Cost: I1–I3 are O(pool) because they count references across all live tables.
At a sim pool of a few hundred blocks and ≥1e6 steps that is ~1e8–1e9 cheap
operations — minutes, not hours, and M4 already paid this bill at ×1e6. **The
gate's "×1e6" is a floor on steps, not a budget for cleverness:** do not
sample the invariants, and do not check them every N steps. A violation that
appears once in 400 steps and is checked every 500 is a violation that does not
exist as far as the harness is concerned.

**S6 (preempt-replay losslessness) is the exception and needs a second run,
not an assertion.** It compares a trace's outputs against *the same trace with
an infinite pool* — the reference run. Two runs, same seed, outputs compared
elementwise. Recorded per trace: the preemption count (a thrash metric, not a
gate) and p95 request-completion latency (Fable 08-02, measured not gated).

## 6. THE TRACE CATALOGUE

Seeded generators, each with its parameters fixed here so a trace name means
the same thing across sessions:

| trace | shape | what it is for |
|---|---|---|
| `poisson-mix` | λ sweep, prompt 64–512, gen 16–256 | the baseline; ordering under steady load |
| `burst64` | 64 requests at t=0 | admission watermark and FCFS under saturation |
| `giant+small` | one 4k prompt, then many 32-token ones | head-of-line blocking; the chunked-prefill co-run claim (S5) |
| `cancel-storm` | 10–20% cancelled at uniformly random points | S8, and the M6 disconnect path found here rather than there |
| `maxtok1` | every request `max_tokens=1` | retire-on-the-step-it-admits; the teardown path at maximum rate |
| `pressure` | arrivals sized to force repeated preemption | S3/S4/S6 and the thrash metric; the adversarial one |
| **`agentic`** | **bursty arrivals, shared 1–2k system prefix, short decodes, 10–20% mid-generation aborts** | **Fable 08-02, Kabir-approved: on a consumer box N=8/16 is agent swarms, not humans. This is M5's real operating point and the trace the bars should be read against.** |

**`agentic` is also the only trace that exercises prefix dedup under
scheduling**, which is where the entry note's deferred lever (§7,
prefix-shared attention) would be measured if it is ever taken.

## 7. WHAT THE SIM IS NOT EVIDENCE FOR

Written down because a green sim is the most inviting thing in this milestone
to over-read.

- **Not throughput.** Gate item [2]'s bars are real-model, M0 methodology. The
  sim's cost model is an input to the sim, not a measurement of the engine.
- **Not arithmetic.** §3.
- **Not the ITL bar.** Gate item [3] is a wall-clock percentile under a real
  co-running prefill. The sim can show that no runnable decode was *skipped*
  (S5) — that is the guarantee's skeleton, not the guarantee.
- **Not thread safety.** TSan on the real server is M5's gate item [5]. The sim
  is single-threaded by construction so that a failure is a scheduler bug and
  never a race in the harness.

---

*Next: `sched.c` against this seam, then the harness, then the traces. The
order matters — a harness written before the seam tends to grow the seam it
wanted rather than the one the engine needs.*
