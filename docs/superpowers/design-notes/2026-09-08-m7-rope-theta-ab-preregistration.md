# 2026-09-08 — PRE-REGISTRATION: is the τ rise into `[1200,∞)` the rope-theta accumulation?

**Committed BEFORE the two arms are compared.** Nothing below was written after
seeing a result; the thresholds are arithmetic on numbers already banked in
`docs/superpowers/gate-reports/2026-09-07-m7-entry-probe-tau.json`.

---

## 1. The question, and where it came from

The M7 entry probe's τ freeze (16/16 streams, `dfce716`) produced this
adjacent-band p95 growth:

| transition | growth |
|---|---|
| `[0,100)`→`[100,300)` | 0.730× |
| `[100,300)`→`[300,600)` | 1.059× |
| `[300,600)`→`[600,1200)` | 0.957× |
| **`[600,1200)`→`[1200,inf)`** | **1.322×** |

τ is flat from 100 to 1200 and then **rises** into the open-ended band. T4 does
not fire (1.322 < 1.5), so the freeze stands — but the gate report left the
cause open in as many words:

> whether that 1.322× is the theta accumulation or just a longer-context
> population. `ref_forward.py --rope-theta accum` exists for exactly this —
> re-running `[1200,∞)` under it is the named next probe.

This document is that probe.

## 2. Why the accumulation is a live suspect and not a formality

**The two sides genuinely differ in how the rope angles are built, and the
difference grows with position.**

- **stoke accumulates.** `src/kernels/ops.c:44,64,84,97`:
  `theta_scale = powf(freq_base, -2.0f/head_dim)`, `theta` starts at `(f32)pos`,
  and `theta *= theta_scale` once per k. This is llama.cpp's own shape
  (`ggml-cpu/ops.cpp`), documented at `src/kernels/kernels.h:279-286`.
- **The reference does not.** `ref_forward.py:111-114` default `theta_mode`
  is `"closed"` — `theta_k = pos * freq_base^(-2k/head_dim)` — M2's frozen
  closed form, chosen *deliberately* so the reference is independent of the
  implementation it grades.

So every probe in the banked grid compared an accumulating implementation
against a closed-form one. §8.9 of the M3 ruling names exactly this difference
as the first suspect when the per-band tolerance widens at long positions.

**The mechanism has a position signature, and it is worth writing down before
measuring it.** Accumulated relative error in `theta_k` grows with k (each f32
multiply, plus the compounding of `theta_scale`'s own rounding). The absolute
angle error is `pos · s^k · e(k)` with `s = 500000^(-2/128) = 0.8146`, which is
maximised near `k ≈ -1/ln s ≈ 4.9` and is **linear in `pos`**.

⇒ **PREDICTION, stated before the run:** if the accumulation is the cause, the
closed-vs-accum logit difference should grow roughly in proportion to absolute
position, so the `[1200,inf)` band (probed positions 1308–2913, median ≈ 1900)
should show roughly **2×** the `[600,1200)` band's (probed 670–1194, median
≈ 930). A closed-vs-accum difference that is flat in position, or orders of
magnitude below the τ envelope, falsifies the mechanism regardless of what the
band p95 does.

## 3. Design

### 3.1 The streams are not a new corpus — they are exactly the two bands' own

The banked grid's two upper bands are fed by seven streams and by nothing else:

| band | streams | probed absolute positions |
|---|---|---|
| `[600,1200)` | `L-660`, `L-900`, `L-1120` | 670–733, 925–988, 1131–1194 |
| `[1200,inf)` | `L-1300`, `L-1900`, `L-2500`, `L-2851` | 1308–1371, 1895–1958, 2508–2571, 2850–2913 |

Re-running those seven under each arm therefore reproduces **the same
population** the banked p95s were computed from — not a resampling of it. The
lower three bands are deliberately out of scope: the transition under
examination is the last one.

### 3.2 Two arms, one variable

| arm | dir | `--rope-theta` | everything else |
|---|---|---|---|
| **A — closed** | `arm-closed/` | `closed` | identical |
| **B — accum** | `arm-accum/` | `accum` | identical |

`m7_probe.py --phase a` runs stoke *and* the reference per stream, so the stoke
side is produced independently in each arm. That is deliberate: **R1 below turns
the redundancy into a null arm for stoke** rather than an economy.

### 3.3 What is NOT being done

- ⛔ **This does not re-freeze τ.** The banked freeze
  (`2026-09-07-m7-entry-probe-tau.json`) stands as the graded criterion, frozen
  under the ruled default (`closed`). A τ frozen from a diagnostic arm chosen
  after seeing which one is smaller is exactly the fit-to-result this project
  withdraws results for.
- ⛔ **No verdict changes.** The 14 ARITHMETIC-CLASS gradings are not re-run.
- ⛔ **No performance claim.** Correctness/numerics work is noise-insensitive:
  box state governs duration, not validity. No timing here is of record.

## 4. The readout, and the rule that grades it

Let `p95_X(band)` be the band p95 of the anchored Δ-profile over **agreeing**
positions under arm X — the same quantity `freeze_tau` already computes, via the
same imported code, never a second implementation of the formula.

```
growth_X = p95_X([1200,inf)) / p95_X([600,1200))
excess_removed = (growth_closed - growth_accum) / (growth_closed - 1.0)
```

`growth_closed` is **this run's own** closed arm, not the banked 1.322 — so the
rule survives the closed arm failing to reproduce (which R2 would catch
separately, and which would invalidate the comparison for a different reason).

| `excess_removed` | verdict |
|---|---|
| **≥ 0.50** | **EXPLAINED** — the accumulation accounts for at least half the rise |
| 0.10 – 0.50 | **PARTIAL** — a real contributor, not the whole story |
| **< 0.10** | **REFUTED** — the rise is the population, not the accumulation |

Secondary, and reported whether or not it is needed — the **bound**:

> `δ(band)` := the maximum over that band's probed positions, over the top-5 ids
> present on both arms, of `|logit_closed − logit_accum|`.

Because the anchored value is `|(x_i − x_a) − (y_i − y_a)|`, perturbing the ref
side by at most δ moves any anchored value by at most `|Δx_i| + |Δx_a| ≤ 2δ`,
hence moves the band p95 by at most `2δ` and τ by at most `4δ`. If
`2·δ([1200,inf)) < 0.0125` — a quarter of the banked rise of 0.049911 — the
switch **provably cannot** account for the rise, independently of the p95s.

## 5. Tripwires — all declared here, all reported whether or not they fire

| # | tripwire | fires when |
|---|---|---|
| **R1** | stoke's own probes must be identical between the arms | any top-5 entry differs at any probed position |
| **R2** | the closed arm must reproduce the banked p95s | either band's p95 differs from `2026-09-07-m7-entry-probe-tau.json` by > 1e-6, or its anchored-value count `n` differs at all |
| **R3** | null arm: a second `closed` run of one stream must be bit-identical | any probed logit differs |
| **R4** | **non-vacuity — the switch must actually change something** | `δ` is exactly 0.0 at every probed position |
| **R5** | each ref probe file's recorded `theta_mode` must match the arm it is filed under | any mismatch |

**R4 is the one that matters most and it is why this is not a bit-exact A/B
alone.** [[fixture-can-be-blind-to-the-defect]]: a bit-exact A/B cannot tell
"the two arms agree" from "one arm never ran". If `-RopeTheta accum` were
dropped anywhere between here and `rope_thetas`, both arms would be `closed`,
δ would be 0 everywhere, and the rule in §4 would return **REFUTED** — the
answer I would be least surprised by, produced by a defect in the probe. R4 and
R5 are the two independent ways that is caught.

R2 is the same class one level up: it is what licenses comparing this sitting's
accum arm to last sitting's freeze at all. ⚠ It is **not** guaranteed —
`build/win/stoke.exe` is a **different binary** from the one the grid ran
(`2f1a4611…` today vs `6a097f55…` in the gate report), rebuilt for the
template #2 commit `c3b4691`. That commit touches `src/main.c`, `src/stoke.c`,
`src/tok.c`, `src/tok.h` and **no** file on the forward path, so the probes
*should* be identical — R2 is what turns "should" into "did".

## 6. Cost, and why it is affordable

The reference's batched arm is licensed by C1/C3 (gate report §1) and is the
only way these positions are reachable at all. Seven streams at
T ∈ {735, 990, 1196, 1373, 1960, 2573, 2915}, two arms, plus one null re-run.
Sliced **one stream per invocation** per standing rule 2; `--phase a`
accumulates into its arm's directory, so an interrupted run loses only the
stream in flight.
