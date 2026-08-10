# M3 [2v2-iii] — the flip tripwire's base rate is measured, and it is ~7–10× the rate the threshold was derived from

**Status: FINDING, not a decision.** It records a measurement taken from
already-banked RUN A evidence. It deliberately does not change the tripwire,
does not touch τ, and does not pre-grade anything. The decision it sets up is
due at the stamp, when the full flip count exists.

**Date:** 2026-08-10, during the RUN A → continuations overnight window.
**Evidence:** the five banked RUN A probe legs (`manifest-A-*.json`).
**Binary:** unchanged, G11 freeze intact (`876ce53f…`).

## 1. What the tripwire assumes, in its own words

Ruling `2026-08-06-m3-probe-grid-RULING-FABLE.md` §5, derivation shown:

> the original ≥4 flips per 55-probe grid fires at ~0.015% false-alarm under
> the measured ~0.5%/token base (Poisson λ≈0.28). At ~384 covered positions
> λ≈1.9; the matching threshold is **≥8 graded flips ⇒ stop, anomaly memo**
> (false-alarm ~0.06%, same order).

`flip tripwire quiet` is part of [2v2]'s close criterion (08-04 ruling §2.2),
so this threshold gates the M3 stamp.

**The recalibration corrected the denominator (55 → 384 positions) and carried
the rate (~0.5%/token) across unchanged.** The rate is where the two files
differ.

## 2. Where ~0.5%/token came from

The Q8_0 leg: 19 divergences over ~3,200 positions (`d456213`). Two estimators:

- naive density — 19 / 3200 = **0.59%/token**
- prompt-level, which handles the conditioning correctly (a stream is only
  comparable up to its first divergence): 19 of 50 prompts diverged within 64
  tokens ⇒ (1−p)^64 = 31/50 ⇒ p = 1 − exp(ln(0.62)/64) = **0.74%/token**

Both sit at the ruling's stated ~0.5% base. The number is sound *for Q8_0*.

## 3. The same quantity, measured on Q4_K_M

Five RUN A legs, each run until its first divergence. First-divergence index
`j` (0-based, generation-relative):

| leg | prompt_len | j | agreeing positions (0..j−1) | flip |
|-----|-----------:|--:|----------------------------:|-----:|
| p00 | 23 | 9  | 9  | 1 |
| p10 | 24 | 19 | 19 | 1 |
| p20 | 64 | 21 | 21 | 1 |
| p30 | 53 | 18 | 18 | 1 |
| p45 | 14 | 26 | 26 | 1 |
| **total** | | | **93** | **5** |

Comparable positions = 93 + 5 = 98. The MLE for a Bernoulli flip process
observed as five geometric first-hitting times:

    p̂ = 5 / 98 = 5.10 % / token

Clopper-Pearson 95% CI = **[1.68%, 11.51%]**.

**Ratio to the assumed base: 10.2× against the ruling's 0.5%, 6.9× against
Q8_0's own 0.74%. The CI's lower bound (1.7%) is still 2.3× Q8_0's rate**, so
the direction of the finding does not depend on the small sample.

Mechanism, and it is the expected one: Q4_K_M is 4-bit where Q8_0 is 8-bit, so
the f32 reference and the integer core sit further apart per position and
resolve more near-ties differently. **Same τ family, same criterion machinery,
different file — and the file is the variable the tripwire's rate ignored.**

## 4. Consequence for the ≥8 threshold

At the tripwire's own denominator of ~384 covered positions:

| base rate | λ over 384 | P(≥8 flips) |
|-----------|-----------:|------------:|
| 0.5%/token (assumed) | 1.92 | **0.083%** ← the designed false-alarm |
| 1.68%/token (CI lower) | 6.44 | 31.9% |
| **5.10%/token (measured)** | **19.59** | **99.90%** |

(The ruling states ~0.06% for the first row; recomputed it is 0.083%. That
discrepancy was already recorded 2026-08-07, is ~6× *stricter* than
claimed-equal, and changes nothing here — quoted so this table is not a second
place the stated figure propagates unchecked.)

**A tripwire designed to fire falsely 0.06% of the time fires ~99.9% of the
time on this file.** At that rate it cannot separate "Q4_K_M is noisier than
Q8_0" from "there is a defect", which is the only thing it exists to do.

Predicted actual count (one continuation per leg is all §5 authorizes, so
positions after a *second* divergence stay uncovered): 6 first divergences,
plus a continuation flip with probability 1−(1−p̂)^(63−j) per leg (0.85–0.94),
summing to **5.39** ⇒ **≈11.4 flips over ≈223 covered positions** (4.9%/token,
reproducing p̂ as it must). Tripwire fires; coverage also falls well short of
the 384 the denominator assumes, because a leg goes dark after its *second*
divergence and §5 authorizes only one continuation.

## 5. What this does NOT establish

- **Not a BUG signal.** Zero-BUG-class is the hard criterion and is decided
  per divergence by §6, position by position, under τ_Q4KM. A high near-tie
  density is the arithmetic-class hypothesis, not the defect hypothesis.
- **Not a reason to change anything before the count exists.** The measurement
  above is provisional at n=5; p41 and five continuations are still running.
- **Not a τ input.** τ_Q4KM freezes from agreeing-position Δ-profiles under
  its own rule and is untouched by this.

## 6. The decision this sets up, and its trap

If the count lands ≥8, the ruling's literal instruction is **stop + anomaly
memo** — not "gate fails". The memo's content is then this measurement.

The tripwire threshold is a frozen constant, so changing it is §8.4-reserved:
under the 2026-08-07 governance change **Opus decides, records the reasoning
and a direction check, and Kabir overrules.** Two shapes are visible now:

- **(a) Keep ≥8 as written, let it fire, resolve by anomaly memo.** Follows the
  ruling exactly, changes no constant.
- **(b) Recalibrate λ from the measured Q4_K_M rate.** ⚠ **This has the
  circularity trap the whole τ discipline exists to prevent: calibrating a
  detector from the very data it must police.** Any recalibration must freeze
  its rate from a declared holdout the tripwire will never itself grade —
  the same rule as "every τ frozen from data it will never grade".

**Recommendation, not yet a decision: (a).** It needs no constant change, and
the anomaly memo is the ruling's own designed outlet for exactly this.

## 7. Lineage

Fifth instance of this milestone's recurring failure mode — a quantity
**carried across a boundary rather than recomputed at it** (08-03 D3, 08-04
§2.3 cost, 08-06 §5 bands, 08-07 §5 window arithmetic, now §5's tripwire
rate). Note the difference in kind: the 08-06 recalibration **showed its
derivation**, which is the only reason this was checkable at all. The failure
was not the arithmetic; it was recomputing `n` while inheriting `p`.
