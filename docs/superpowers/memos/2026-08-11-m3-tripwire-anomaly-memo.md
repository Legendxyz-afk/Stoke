# M3 — the flip tripwire fired at 9. Anomaly memo, and the decision.

**Required by ruling 2026-08-06 §5: "≥8 graded flips ⇒ stop, anomaly memo."
Nine were graded. This is that memo.**

Decided under the 2026-08-07 governance change: Opus decides §8.4-reserved
questions, records the reasoning and a direction check, **Kabir overrules.**

## 1. What fired

9 graded flips over 290 covered positions on the Q4_K_M grid. Threshold ≥8.
The close criterion (08-04 ruling §2.2) requires "flip tripwire quiet", so as
written **[2v2] does not close.**

## 2. Why it fired, and why that is not "the engine got worse"

The threshold's derivation is shown in the ruling, which is the only reason
any of this was checkable:

> the original ≥4 flips per 55-probe grid fires at ~0.015% false-alarm under
> the measured ~0.5%/token base (Poisson λ≈0.28). At ~384 covered positions
> λ≈1.9; the matching threshold is ≥8 graded flips.

The recalibration correctly recomputed the **denominator** (55 → 384
positions) and carried the **rate** across unchanged. The rate is the problem.

| pair / file | measured flip rate |
|---|---|
| Q8_0, M2↔HEAD (where ~0.5% came from) | 0.59 %/token naive; **0.74 %** by the prompt-level estimator |
| **Q4_K_M, ref↔HEAD (this grid)** | **3.10 %/token** (9/290) |

That is **4.2× Q8_0's own rate and 6.2× the ruling's stated base.** The
mechanism is the expected one and was predicted before the data was in: 4-bit
quantization puts the f32 reference and the integer core further apart than
8-bit does, so more near-ties resolve differently. Same τ family, same
machinery — the file is the variable.

**Under the ruling's assumed rate, 9 flips in 290 positions is a 1-in-46,800
event** (λ = 1.45, P(≥9) = 2.14 × 10⁻⁵). **Under the rate measured on the file
itself, the expectation is 290 × 0.0310 = 9.0.** We observed exactly what the
file's own density predicts.

*(That figure was first written here as "~1-in-5,000" from a mental estimate
and corrected to 1-in-46,800 on computing it — an error of ~9×, in a memo
whose subject is a quantity carried across a boundary instead of recomputed at
it. It is recorded rather than silently fixed because it is the same failure
mode one level down, and because the correction makes the argument stronger,
which is exactly when an error is least likely to be caught.)*

The denominator failed the same way: the ruling projected continuations would
restore coverage to ~384. Four of five continuations flipped again, and §5
authorizes only one per leg, so coverage stopped at 290. Both limbs of the
tripwire trace to the one inherited number.

## 3. The obvious objection, stated plainly

*"A detector fired, and the person it inconveniences has produced an
explanation. That is how gates become worthless."* The objection is correct in
general, so here is what distinguishes this case:

1. **The explanation is pre-registered, not post-hoc.** The base-rate mismatch
   was computed and committed in `6b484b9` when only **5** flips existed —
   before p41, before any continuation, before the count could be known. That
   commit predicted, in writing, that the tripwire would fire and why. It also
   predicted ≈11.4 flips over ≈223 covered; the actual 9 over 290 is close in
   conclusion and **wrong in central estimate**, which is recorded rather than
   quietly dropped.
2. **The independent detector passed.** Zero-BUG is the criterion that
   actually looks for defects, and it examines every divergence individually.
   All 9 (plus 1 supplementary) are ARITHMETIC-CLASS, worst case at 54 % of
   its τ, every top-5 set agreeing modulo the rank-5 tolerance, no foreign
   core candidate anywhere, no ESCALATE.
3. **The tripwire's input is demonstrably wrong for this file**, and that is a
   fact about the calibration, not about the run.

## 4. The limit of that argument — carried headroom, stated because it is real

τ_Q4KM is frozen from agreeing positions on the *same pair* it grades. **A
defect that uniformly inflated logit differences would inflate τ and the
divergence margins together, and the margin/τ ratio would stay comfortable.**
This criterion cannot see that class of defect.

What partially covers it, and what does not:
- `ref_forward.py` is independent of stoke's entire C lineage, so a defect in
  stoke's arithmetic is visible to the comparison. This is the main protection
  and it is real.
- τ landing at ~2× Q8_0's values is a *mechanistic prediction* of 4-bit
  quantization, not a parameter fitted to make the result pass.
- But a spec misreading shared by ref and HEAD would be invisible — the same
  blind spot the 07-28 ruling named for the M2 pair and deferred.

**Recorded as carried headroom at the M3 stamp, not as a blocker**, in the
same shape as the existing entries: number (3.10 %/token vs 0.74 % on Q8_0),
location (this memo, §4), date (2026-08-11).

## 5. Decision

**Keep the threshold at ≥8, exactly as written. Do not recalibrate λ.**

Recalibrating λ from the measured Q4_K_M rate would set a detector's threshold
from the very data it must police — the circularity the entire τ discipline
exists to prevent ("every τ frozen from data it will never itself grade"). If
a recalibration is ever wanted, it must freeze its rate from a declared
holdout. That is not needed here and is not proposed.

**The tripwire is therefore reported as FIRED, with this memo as its
disposition.** The substantive question the tripwire proxies — "are there more
class divergences than the arithmetic explains?" — is answered directly and
negatively by the file's own measured density, and the defect question is
answered independently by 10/10 ARITHMETIC-CLASS.

**Direction check.** This resolution does not weaken any bar: zero-BUG is
untouched and was met on its own terms; τ was frozen before grading, from
disjoint data; the added samples were excluded from every count. It weakens
nothing except the tripwire's ability to fire on *this file*, which it was
never calibrated for. The strictly-harder alternative — refusing to close M3
until the Q4_K_M grid produces ≤7 flips — is unreachable by construction: at
3.10 %/token no authorized grid of this size can produce it, so that
alternative is not "stricter", it is "never closes".

**What would change this decision:** any divergence classifying BUG; a
recomputation showing the Q8_0 rate is comparable to 3.10 % after all (which
would remove the file-difference explanation); or evidence that the flip
excess concentrates somewhere structural — a single layer, a single token
class, one prompt tier — rather than distributing as near-tie noise.

## 6. What this memo does not do

It does not stamp M3 and it does not tag. The evidence is complete and clean,
one precondition is met by judgement rather than mechanically, and the person
whose gate this is should get to say so before the tag exists.
