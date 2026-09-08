# 2026-09-08 — clause 2 on end-to-end wall: the attempt, VOID, and what it found instead

## STATUS: ⛔ **NOT MEASURED. Both attempts are unreadable, for two DIFFERENT reasons — and the second reason is a lead worth more than the measurement.**

Protocol: `docs/superpowers/design-notes/2026-09-08-clause2-end-to-end-preregistration.md`
(committed `bfda9ec`, before either run).

| attempt | design | outcome |
|---|---|---|
| 1 — block-sequential | 5 cells × 3 runs, bracketed | ⛔ **VOID**: W2 fired, bracket drifted **18.4%** |
| 2 — interleaved | 1 run/unit, ABBA, cool-downs | ⛔ **NOT READABLE**: stoke's own spread **11.7%** vs a ~1% effect |

**Clause 2's disposition is therefore UNCHANGED and undecided.** Nothing here
re-grades it in either direction.

---

## 1. Attempt 1 — void, and the method was the defect

The same `llamacpp-tuned × s2-closed-8` cell read **67.6 s at t+0** and
**81.3 s at t+32.5** — 18.4% against W2's 5% threshold.

| t+min | cell | wall |
|---|---|---|
| 0.0 | llama @8 | 67.6 s |
| 4.7 | stoke @8 | 67.1 s |
| 9.6 | llama @16 | 171.1 s |
| 21.1 | stoke @16 | 164.9 s |
| 32.5 | llama @8 | **81.3 s** |

⚠ **The drift is a STEP, not a gradient**, and two bracket points cannot say
whether it began during p3 or p4 — which lands opposite ways for stoke. The @8
pair (p1, p2) sits 4.7 min apart and agrees to 0.7%, which is tempting;
**refused**, because "probably fine" is not what a pre-registered bracket is
for.

⇒ [[measurement-needs-interleaving]] had already ruled this out: *one unit per
PROCESS, counterbalance the order, detect drift from the arms' own repeats.* A
block-sequential matrix violates all three. The 2026-09-05 block got away with
sequential cells because it **swept the box between them**; this one reproduced
its shape without its care, under 35 minutes of sustained 8-thread load.

**Method finding, and it generalises:** *a ~35-minute sequential bench on this
box drifts ~20%.* Every future serving measurement here has to be interleaved.

## 2. Attempt 2 — the instrument works; the ENGINE is the variance

`tests/tooling/run_e2e_interleaved.py`: one measured run per unit, fresh server
each, arms alternating with the order flipping every pair, cool-down between
units, per-unit banking, and a refusal on any unit that cannot reconstruct its
own published aggregate. Five of six units banked before the session ended (the
sixth is missing because the sitting was stopped, not because it failed).

| t+min | arm | wall | agg | TTFT p50 | TTFT min |
|---|---|---|---|---|---|
| 3.0 | llamacpp-tuned | 88.0 s | 17.815 | 59.9 s | 31.0 s |
| 6.5 | stoke-tuned | **83.3 s** | 13.317 | 36.1 s | 7.0 s |
| 10.4 | stoke-tuned | **93.7 s** | 11.786 | 39.3 s | 7.5 s |
| 14.2 | llamacpp-tuned | 87.3 s | 17.613 | 58.4 s | 29.6 s |
| 17.9 | llamacpp-tuned | 87.9 s | 17.687 | 59.6 s | 30.5 s |

| arm | n | wall median | **own spread** |
|---|---|---|---|
| llamacpp-tuned | 3 | 87.9 s | **0.8%** |
| stoke-tuned | 2 | 88.5 s | **11.7%** |

`wall ratio = 0.9938×`, against a largest single-arm spread of **11.7%**
⇒ **NOT READABLE.** Reported as such.

⭐ **The box was NOT drifting this time** — llama repeated to 0.8% over 15
minutes. So the interleaving fixed attempt 1's problem, and exposed a different
one underneath it.

## 3. ⚠⚠ The lead: stoke is repeatable WITHIN a server lifetime and variable ACROSS server starts

| stoke measurement | spread | n |
|---|---|---|
| 3 runs inside ONE server lifetime (attempt 1, p2) | **0.2%** | 3 |
| 2 runs in SEPARATE server lifetimes (attempt 2) | **11.7%** | 2 |

The two attempt-2 units are **uniformly** slower, not tail events — wall 83.3 →
93.7, aggregate 13.317 → 11.786 (13% apart), while `TTFT min` barely moves (7.0
vs 7.5 s). That is the signature of a **per-process speed state fixed at
startup**, not of a workload effect.

⭐ **And there is a code-level candidate that matches it exactly.**
`pool_prefer_performance_here()` — the 2026-08-17 EcoQoS opt-out, which tells
Windows "this thread is performance work" rather than letting it classify a
condvar-sleeping thread as background and park it on an **E-core** — is called
from exactly one place in the tree:

```
src/bench.c:739:    pool_prefer_performance_here();
```

The pool's **worker** threads take it themselves at `pool.c:184`, so workers are
covered on every path. **The CALLING thread is not.** On the `bench` path it is
opted out at `bench.c:739`; on the **`serve`** path — the one every number in
this report was measured through, and the one that ships — **nothing calls it.**

⚠⚠ **This is a HYPOTHESIS, not a measurement.** What is established: the serve
path lacks a call the bench path has, and stoke's cross-process spread has the
shape a startup-fixed QoS decision would produce. What is NOT established: that
this is the cause. **n = 2.**

**If it is true it is not a bench nuisance — it is a defect in the shipping
server**, and it would mean an unknown fraction of `stoke serve` deployments run
their scheduler/sampling thread on an efficiency core.

### The named next check, cheapest first

1. **Read, don't run:** confirm no other call site opts the serve main thread
   out (done — one grep, one hit, in `bench.c`).
2. **A/B it on one binary:** `STOKE_NO_QOS=1` already exists to restore
   pre-opt-out behaviour, so the arms come from one binary in one interleaved
   sitting. Run ≥6 units per arm through `run_e2e_interleaved.py`.
3. **Only then** decide whether `serve` should call
   `pool_prefer_performance_here()`, and re-run clause 2's measurement — because
   until stoke's cross-process spread is under ~2%, a 1% ratio cannot be read no
   matter how many units are added.

## 4. What this does NOT establish

- **Clause 2, either way.** Undecided; the criterion change is pre-registered
  and **ungraded**.
- **That stoke is slower.** 0.9938× is inside the noise, not a result.
- **That stoke is faster.** The banked 1.025×/1.056× from 2026-09-05 remains
  in-process-vs-HTTP and is not gate-grade either.
- **That the EcoQoS gap is the cause of anything.** Named suspect, matching
  signature, n=2, unmeasured.
