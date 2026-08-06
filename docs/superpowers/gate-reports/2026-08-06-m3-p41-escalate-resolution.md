# M3 [2v2-ii] — resolution of the single ESCALATE (p41)

**Verdict: ARITHMETIC-CLASS.** The L2/L3 ladder returned no STEP at the
divergent position or at any of three control positions, and the divergence's
layer curve sits inside the band that agreeing positions of the same pair
occupy. **[2v2-ii] therefore stands at 18 + 1 = 19 ARITHMETIC-CLASS, 0 BUG**,
which is the ruling's PASS condition (`§2.2`, "PASS = zero BUG-class").

τ was **not** re-frozen, moved, or re-derived. No threshold changed.

Adjudicated at HEAD `876ce53f…` (`build/win/stoke.exe`) against the archived
M2 gate binary `build/stoke-m2-gate-d2b17c53.exe`, model
`Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf` (4,280,405,216 bytes).

---

## 1. What escalated, and why

p41's first divergence is the token at abs 479, produced by the forward pass
at abs 478. Margins: **M2 0.037** (a knife-edge tie), **HEAD 0.460324**
(decided). `max(g_X, g_Y) = 0.460324` lands in `[τ, 2τ)` for band `[300,600)`
— `τ = 0.394898`, `2τ = 0.789796` — so `§6` classifies it ESCALATE and sends
it to the ladder. Top-5 sets were equal on both sides.

It escalates because `max()` takes the **confident** side. That is the same
asymmetry `§4` identified when it retired the oracle bar: confidence on either
side makes a divergence harder to clear. The ruling chose `max()` deliberately
and ratified it, so it is applied as written.

## 2. The ladder's magnitude leg does not apply to this pair, and never could

`§6` says: *"Ladder ≤1e-3 / no-STEP ⇒ ARITHMETIC-CLASS … a STEP ⇒ BUG."*

**The `1e-3` is inherited from a different comparison.** It is `l2_diff.py`'s
threshold, and `l2_diff.py` was built for M2's gate, which compared **C f32
against numpy f32** — two implementations of the *same* arithmetic, differing
only by float32 non-associativity. That gate's L2 leg passed 36/36 with a
worst layer of **6.104e-04** (`2026-08-02-m2-parity.md`), which is what a
`1e-3` bar was calibrated against.

The pair being graded here is **M2's f32 forward against HEAD's integer
core.** Since K2 (`cdea4f3`) HEAD quantizes activations to int8 per block.
The two sides differ **by construction**, not by reduction order. `36/36
layers over 1e-3` is the *expected* reading for this pair and carries no
information about defects — it is produced identically by a healthy engine and
a broken one. All four curves in this report report `L2: FAIL` for that
reason.

This is the same error the ruling itself corrected one level up: a bar
borrowed from one comparison and applied to a different one. It is recorded
here rather than repaired, because repairing it is not this document's to do
(see §6).

## 3. What the ladder actually decided

`§6`'s fail-safe paragraph is explicit about which part of the ladder carries
the verdict:

> *"Anything not affirmatively classifiable lands in ESCALATE, and **the
> ladder — not the constant — decides**."*

And `§6` names exactly one BUG trigger at this stage: **a STEP.** The
discriminator is structural, not absolute — `l2_diff.py:19` states its own
design intent the same way ("a structural bug in one op … appears as a STEP",
`JUMP_FACTOR = 10.0`, a layer diff ten times the running maximum).

**STEP was negative on the divergent position and on all three controls.**
Nothing anywhere in four 36-layer curves approached a 10× jump; the curves
grow smoothly and monotonically in the mean.

That is the ruling's stated criterion, met.

## 4. The magnitude question, answered by measurement instead of by the borrowed constant

A STEP test is a test of curve *shape* and cannot say whether a curve is
anomalously *large*. So the absolute bar was **measured for this pair** the
same way τ was: calibrate on positions where the two engines **agree**, then
judge the divergent position against that. The two sets are disjoint by
construction.

Control positions are each prompt's **last prefill position**, which is free
of the teacher-forcing risk the p41 dump had to take: `--dump` fires at
`t == n_prompt-1`, and during prefill both binaries consume identical given
tokens, so no divergence can have occurred yet.

| position | L35 max_abs_diff | L35 mean_abs_diff | STEP |
|---|---|---|---|
| **p41 @ abs 478 — DIVERGENT** | **2.687e+00** | **1.342e-01** | none |
| p41 @ abs 476 — control, same prompt | 6.983e-01 | 1.166e-01 | none |
| p42 @ abs 461 — control | 8.103e-01 | 1.094e-01 | none |
| p43 @ abs 470 — control | 1.355e+00 | 1.142e-01 | none |

| statistic | control range | control spread | divergent | vs highest control |
|---|---|---|---|---|
| `max_abs_diff` | 0.6983 – 1.355 | 1.94× | 2.687 | 1.98× |
| `mean_abs_diff` | 0.1094 – 0.1166 | 1.07× | 0.1342 | **1.15×** |

**The two statistics agree, and the mean is the one to read.** `max_abs_diff`
is the single worst of 2,560 values per layer — an extreme-value statistic,
and its 1.94× spread *among agreeing positions* is a direct measure of how
noisy it is. `mean_abs_diff` aggregates the whole layer: agreeing positions
sit within 1.07× of each other, and the divergent position is 1.15× the
highest of them.

So the divergent position's layer stack is slightly elevated and otherwise
indistinguishable from a position where the two engines chose the same token.
There is no discontinuity, and no layer at which the two engines part company.

## 5. The confound, stated plainly

**A divergence is selected for being a position where perturbation flipped the
decision**, so above-average noise there is close to the *definition* of the
event rather than evidence of a defect. Comparing a divergent position's noise
against an arbitrary position's is biased toward "looks worse."

This cuts against over-reading the 1.15×/1.98× elevation as a finding. It does
not weaken the STEP result, which is a within-curve shape test and is not
affected by the selection.

The confound was written down before the third control completed, not after.

## 6. Forward note — `§6`'s ladder clause carries a pair-specific constant

`§6` states `≤1e-3` as though it were pair-independent. It is not: it is the
C-f32-vs-numpy-f32 figure, and any ladder run on an f32-vs-integer-core pair
will exceed it by construction. **M7 re-runs ladders on Llama-3.2 and will hit
this immediately**, as will any M4+ escalation, so the clause wants restating
in the same per-pair, per-band form τ already has.

Per `§8.12` this is the **default** path, not the carve-out: the criterion is
being *applied as written* here — its only BUG trigger is a STEP, and no STEP
was found — rather than overridden. Nothing was banked under a criterion
believed wrong. Entry appended to the standing FOR-FABLE file for the M4-entry
cluster; no ruling is required before M3 stamps.

## 7. Provenance of the dump this rests on

From `61a086d`, verified rather than assumed:

- The ladder needed the prefill to **end** at abs 478, which required the
  prompt plus the two generated tokens both sides agree on (`first_diff = 2`).
  Tokenizing the constructed file returned **exactly p41's 477 ids followed by
  `[18611, 334]`** — without that check a BPE re-segmentation across the join
  would have silently dumped a different position.
- Both output cross-checks passed: **HEAD emitted 16141, M2 emitted 3966**,
  exactly what the banked streams record each choosing there.

Controls were produced by `tests/tooling/run_l2_controls.cmd`; p41 and p43 on
2026-08-06 14:54, p42 on 2026-08-06 20:25 after a box shutdown interrupted the
first run. The resumed run recomputed p41's and p43's curves from their banked
dumps and reproduced both to every printed digit.
