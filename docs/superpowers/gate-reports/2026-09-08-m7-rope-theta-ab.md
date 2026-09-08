# 2026-09-08 — ROPE-THETA A/B: gate report

## STATUS: **REFUTED. The τ rise into `[1200,∞)` is NOT the theta accumulation.**

The M7 entry probe's τ freeze rose **1.322×** from `[600,1200)` into
`[1200,∞)` and left one question open — the accumulation, or the population.
Answered here, and the answer is the one the mechanism's own size predicts once
it is measured rather than argued about.

**Both halves matter and they point opposite ways:**

- ⭐ **The mechanism is REAL and correctly identified.** The closed-vs-accum
  logit difference **grows linearly with absolute position** — per-stream
  `p95(|Δ|) / median position` is flat at **6.0e-08 → 1.3e-07** across a 4.1×
  span of position. That is exactly the signature the M3 ruling §8.9 and the
  ROADMAP's 2026-07-31 note predicted for it.
- ⛔ **And it is three orders of magnitude too small to be the cause.** At
  position 2882 it is `3.2e-04`, against a band p95 of `0.2047`. Switching the
  reference to stoke's own accumulation moves the band p95 by **+0.000146** —
  the wrong way, and by 0.07%. `excess_removed = -0.0025` against the
  pre-registered EXPLAINED threshold of 0.50.
- ⭐ **It never changed a token decision: 0 argmax flips in 448 probed
  positions**, out to absolute position 2913.

⇒ **The rise is a property of the positions and text sampled, not of the angle
arithmetic.** T4 did not fire (1.322 < 1.5), so nothing about the banked freeze
changes. ⚠ **"Population" is what is LEFT, not what was shown** — this probe
eliminated the one named suspect; it did not identify a replacement.

⭐ **A second result, not asked for and worth more than a by-product: the
batched reference is EXACTLY reproducible.** R3's null arm — two `closed`
runs of the same stream — agreed at **every probed logit, max |Δ| exactly
0.0**. C2 established that for the SERIAL path on 2026-09-07; these probes all
come from the BATCHED path, where it had never been checked and where C3 had
measured a ~2e-5 dependence on *batch shape*. Fixed T, fixed ids ⇒ bit-exact.
**That is what makes the 1.4e-3 bound below attributable to the switch and to
nothing else.**

⚠ **And R2 is the finding that licenses comparing this sitting to last
sitting's freeze at all.** The closed arm reproduced the banked table
**exactly** — `[600,1200)` p95 0.154794 over n=926, `[1200,inf)` p95 0.204705
over n=1232, and every stream's agree count — from a **different binary**
(`2f1a4611…` today against `6a097f55…` in the 09-07 report, rebuilt for the
template #2 commit `c3b4691`). The reasoning that it *should* be identical
(that commit touches `main.c`/`stoke.c`/`tok.c`/`tok.h` and nothing on the
forward path) was correct; R2 is what turned it into a measurement.

---

**Protocol:** `docs/superpowers/design-notes/2026-09-08-m7-rope-theta-ab-preregistration.md`,
committed in `d926809` **before the two arms were compared**.
**Question raised by:** `2026-09-07-m7-entry-probe.md` §STATUS.
**Named as the first suspect by:** M3 ruling §8.9 / §10.9, and
`docs/ROADMAP.md:614-618` (logged 2026-07-31).

⚠ The oracle is not consulted anywhere in this report. The two arms are
`ref_forward.py --rope-theta closed` and `--rope-theta accum`; stoke is the
third side and is unchanged between them.

---

## 0. Provenance

| | |
|---|---|
| model | `models/Llama-3.2-3B-Instruct-Q4_K_M.gguf`, 2,019,377,696 B |
| stoke binary | `build/win/stoke.exe`, sha256 **`2f1a461103e75b47729a1cafac3e46fe8c4626d2948601d09df66e04fcfd6866`**, 1,423,823 B, `make -n` reports nothing to be done and `git status` is clean at HEAD |
| reference | `tests/tooling/ref_forward.py`, batched arm (licensed by C1/C3, `2026-09-07-m7-entry-probe.md` §1) |
| pinning | every reference run: affinity `0xFFF`, `OPENBLAS_NUM_THREADS=6`, verified by the launcher, recorded in each probe file's `env` |
| streams | the seven that feed the two bands and nothing else — `L-660 L-900 L-1120` → `[600,1200)`; `L-1300 L-1900 L-2500 L-2851` → `[1200,inf)` |
| probes | 448 positions × 2 arms, absolute positions 670–2913 |
| artifact | `2026-09-08-m7-rope-theta-ab.json` (per-position Δ banked, so "it grows with position" is auditable from the file rather than from two aggregates) |

**Why these seven and not a new corpus:** the banked grid's two upper bands are
fed by exactly these streams, so re-running them reproduces *the same
population* the banked p95s were computed from — not a resample of it. R2 is
the check that this actually happened.

## 1. The mechanism, and why it was the right first suspect

The two sides genuinely differ in how the rope angles are built:

- **stoke accumulates** — `src/kernels/ops.c:44,64,84,97`:
  `theta_scale = powf(freq_base, -2.0f/head_dim)`, `theta` starts at `(f32)pos`,
  `theta *= theta_scale` once per k. llama.cpp's own shape.
- **the reference does not** — `ref_forward.py:111-114` default is the closed
  form `theta_k = pos · freq_base^(-2k/head_dim)`, chosen so the reference is
  independent of the implementation it grades.

The accumulated relative error grows with k while `theta_k` itself decays as
`s^k`, `s = 500000^(-2/128) = 0.8146`; the product peaks near
`k ≈ -1/ln s ≈ 4.9` and is **linear in `pos`**. That prediction was written
into the pre-registration §2 before the run.

## 2. Result

| band | arm | n | p95 | max |
|---|---|---:|---:|---:|
| `[600,1200)` | closed | 926 | **0.154794** | 0.984710 |
| `[600,1200)` | accum | 926 | **0.154811** | 0.985105 |
| `[1200,inf)` | closed | 1232 | **0.204705** | 0.398487 |
| `[1200,inf)` | accum | 1232 | **0.204851** | 0.398543 |

| | growth `[600,1200)`→`[1200,inf)` |
|---|---|
| closed (the ruled default, the freeze's arm) | **1.3224×** |
| accum (the diagnostic arm) | **1.3232×** |
| excess removed | **−0.0025** |
| **verdict** | **REFUTED** |

Reference closed-vs-accum logit Δ (the bound):

| band | positions | median position | max \|Δ\| | p95 \|Δ\| |
|---|---:|---:|---:|---:|
| `[600,1200)` | 192 | 957 | 4.138947e-04 | 7.438660e-05 |
| `[1200,inf)` | 256 | 2508 | 6.828308e-04 | 2.431870e-04 |

Reference argmax flips between the arms: **0 of 448** probed positions.

2 × max Δ in `[1200,inf)` = **1.365662e-03** against the 0.0125 refusal
threshold ⇒ **CANNOT account for the rise** — 9.2× inside it.

### The bound, stated as an inequality rather than an intuition

The anchored value is `|(x_i − x_a) − (y_i − y_a)|` with `x` the reference side.
Perturbing `x` by at most δ moves any anchored value by at most
`|Δx_i| + |Δx_a| ≤ 2δ`, hence the band p95 by at most `2δ` and τ by at most
`4δ`. With `δ = 6.83e-04`, the switch **cannot** move the `[1200,inf)` p95 by
more than `1.37e-03` — against an observed rise of `0.049911`. The direct
measurement above agrees: it moved it by `1.46e-04`.

## 3. ⭐ The position curve — the mechanism confirmed, per stream

Each stream probes a narrow window, so this is a seven-point curve in position
rather than two aggregates. Banked per position in the JSON.

| stream | median position | p95 \|Δ\| | max \|Δ\| | **p95 / position** |
|---|---:|---:|---:|---:|
| `L-660` | 702 | 4.196e-05 | 6.104e-05 | **5.98e-08** |
| `L-900` | 957 | 7.820e-05 | 1.183e-04 | **8.17e-08** |
| `L-1120` | 1163 | 7.820e-05 | 4.139e-04 | **6.72e-08** |
| `L-1300` | 1340 | 1.097e-04 | 2.003e-04 | **8.19e-08** |
| `L-1900` | 1927 | 2.174e-04 | 2.632e-04 | **1.13e-07** |
| `L-2500` | 2540 | 3.204e-04 | 6.828e-04 | **1.26e-07** |
| `L-2851` | 2882 | 3.176e-04 | 4.377e-04 | **1.10e-07** |

**δ/position is flat within ~2× across a 4.1× span of position** ⇒ the
difference is linear in position, as predicted. The residual upward drift in
the last column is real but small and is not modelled here.

⚠ **The pre-registration's numeric prediction was close in mechanism and wrong
in factor, and it is recorded rather than quietly dropped.** §2 predicted the
`[1200,inf)` band should show "roughly 2×" the `[600,1200)` band's Δ. Measured:
**3.27×** (p95). The mechanism (linear in position) is confirmed; the factor
was computed from a band median of ≈1900 that the actual sampling did not have
— the probed median is **2508**, because the four streams sit at 1340/1927/
2540/2882. 2508/957 = 2.62×, and δ/position itself drifts up ~1.5× over the
range. **The prediction was falsified as stated and confirmed as reasoning.**

⚠ **A resolution floor worth naming so nobody extrapolates through it.** A
shakedown pair on `S-a1` (positions 4–67) read max = p95 = **7.629395e-06**
exactly — `4 ulps` at logit magnitude ~16–32. At short context this quantity is
at the representation floor of an f32 logit and is not a measurement of the
mechanism. The curve above starts at position 670 for that reason.

## 4. What the rise actually is — and the limit of saying so

By elimination: **not the angle arithmetic.** Beyond that this probe is silent,
and "population" is the residual, not a finding.

One characterization *is* available from the table and is worth recording,
because it cuts against the shape a defect would make: `[600,1200)`'s **max**
is 0.9847 while `[1200,inf)`'s is only 0.3985. **The higher band's whole
distribution sits higher, it does not have a fatter tail** — the 1.322× is a
shift, not an outlier. (The 0.9847 is the `L-1120` position-1175 outlier the
09-07 report already named and declined to smooth.)

⇒ Nothing here asks for a change. T4 does not fire; §8.9 would permit
collapsing to one constant; per-band is kept as strictly more conservative,
exactly as the 09-07 report left it.

## 5. TRIPWIRES

| # | tripwire | measured | fired |
|---|---|---|---|
| **R1** | stoke's own probes identical between arms | 0 differences over 448 positions | no |
| **R2** | closed arm reproduces the banked p95 and n | p95 and n exact for both bands; every stream's agree count exact | no |
| **R3** | null arm bit-identical | 64 positions, 0 differing, max \|Δ\| **exactly 0.0** | no |
| **R4** | **non-vacuity — the switch changed something** | max \|Δ\| = 6.83e-04 > 0 | no |
| **R5** | each probe file's `theta_mode` is its arm's | enforced at write time in `ref_probe`, 14 files | no |

**R4 is the one that matters and it is why this is not a bit-exact A/B alone.**
[[fixture-can-be-blind-to-the-defect]]: a bit-exact A/B cannot distinguish "the
two arms agree" from "one arm never ran". Had `-RopeTheta` been dropped anywhere
between `m7_probe.py` and `rope_thetas()`, both arms would have been `closed`,
δ would have been 0 everywhere, and §2's rule would have returned **REFUTED** —
the same verdict, manufactured by a defect in the probe. R4 (δ must be non-zero)
and R5 (`ref_probe` refuses a probe file whose recorded `theta_mode` is not the
one asked for) are two independent ways that is caught, and the shakedown on
`S-a1` exercised both before the real run started.

⚠ **A structural hazard closed in the tooling rather than by discipline:**
`--phase a` **accumulates** into its out-dir — that is what makes a partial grid
completable without losing work — and it is also what would let two arms pile
into one folder and freeze τ from a **mixture**, which is not loud. Every grid
record now carries `ref_theta_mode`, read from the reference's own output rather
than from the argument that asked for it, and `load_grids` refuses a mixed or
untagged directory. Selftested both ways.

## 6. What is NOT established here

- **That the population explains the rise.** Only that the accumulation does
  not. No positive mechanism is offered.
- **Anything about τ.** The banked freeze stands, unchanged, frozen under the
  ruled default. Choosing whichever arm gives the smaller tolerance after seeing
  both is the fit-to-result this project withdraws results for — and it would
  not have helped here anyway: the accum arm's τ is very slightly *larger*.
- **Anything about performance.** No timing here is of record; correctness work
  is noise-insensitive and box state governed duration only.
- **That the accumulation is harmless in general.** It is harmless *on this
  model, at these positions, at this magnitude*. It is linear in position, so a
  context 100× longer is a different question — and the measured constant,
  ~1.1e-07 per position, is now on record to answer it with.

### Supersedes

The ROADMAP's 2026-07-31 note — "~1e-9 rounding difference that GROWS WITH
POSITION … if long-context parity wobbles, check this first" — is now measured
in the domain that matters. **The qualitative claim is CONFIRMED** (it grows,
linearly, at ~1.1e-07 logit per position). **The operational suspicion is
DISCHARGED**: parity did wobble at long context, this was checked first, and it
is not the cause.
