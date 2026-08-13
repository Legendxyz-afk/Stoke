# RESULTS-VS-PHYSICS — Q8-KV misses item [4]'s class-rate bound, and why the bound could not have been met

*2026-08-13. Opus 5. Filed under the 2026-08-04 ruling §8.1, which pre-ruled
this disposition: "if it measures worse than the base rate, that is not a
grind-until-green — it is a results-vs-physics memo (the quantizer itself is
then the finding: scale granularity, outlier channels)." Not blocking; the
decision it feeds is Kabir's.*

## 1. The measurement

| | result | bound | |
|---|---|---|---|
| BUG-class | **0** | 0 | ✅ **met** |
| first-divergence prompts | **38/50** | ≤ 19/50 | ❌ missed |
| flip rate | **1.1875 %/token** (38/3200) | ≤ 0.5 %/token | ❌ missed |
| classification | **38 ARITHMETIC-CLASS** (31 direct + 7 laddered) | — | complete |

τ frozen 2026-08-13T04:46:44Z from 9 agreeing positions of 15 probes, before
grading anything: **1.0103 / 1.2617 / 0.7259** across the three bands. Worst
direct margin **0.5660 = 56% of its τ**; worst ESCALATE margin 0.7652 = 76%.
All seven ESCALATEs cleared the ladder with **no STEP**, and their layer curves
sit at the *bottom* of the band that agreeing positions occupy
(`2026-08-13-m4-g4-escalate-resolution.md`).

**Nothing here is a defect signal.** Every divergence is a near-tie decided the
other way, every margin is comfortably inside threshold, and no layer curve
shows a discontinuity.

## 2. The physics — the bound compares two perturbations four orders of magnitude apart

§8.1's rationale: *"Q8-KV's noise should live at or below the arithmetic-class
noise this project already accepts between implementations."* The referent it
picks for "already accepts" is the M2 census — **19/50 prompts, recomputed
today as 19/3200 = 0.5938 %/token**.

But the M2 census measures a pair of binaries that differ **only in f32
reduction order**. Its perturbation is float32 non-associativity:

    relative perturbation ≈ ε_f32 ≈ 1.2 × 10⁻⁷

Q8-KV's pair differs by **requantising the KV cache to 8 bits**. Per 32-element
block, `d = max|x|/127` and values round to nearest, so the error is bounded by
`d/2` — the gate report's own measurement, **0.4836·d (K) and 0.4894·d (V)**,
confirms it. Against the block maximum:

    RMS relative perturbation ≈ d/√12 / max|x| = 1/(127·3.46) ≈ 2.3 × 10⁻³

**≈ 4 orders of magnitude larger than the perturbation the bound was
calibrated on.** A rate bound transferred across that gap is not a bar the
implementation can pass or fail; it is a category error in the calibration.

**The magnitudes line up with what was observed, which is the check that makes
this an explanation rather than a story.** Logits here run 16–35 in magnitude,
so a 2.3×10⁻³ relative perturbation lands ≈ 0.04–0.08 in absolute logit terms
— *the same order as the observed divergence margins*, which range 0.013 to
0.566 with most under 0.3. A perturbation comparable to the margin distribution
flips near-ties frequently and decisively-ranked tokens never. That is exactly
the observed pattern: 38 flips, zero BUG, every margin inside τ.

## 3. The instrument is also near saturation, which the bound's form hides

The metric counts **one first-divergence per prompt** over 3,200 compared
tokens, so its maximum possible value is **50/3200 = 1.5625 %/token**. The
measured 1.1875% is **76% of the largest value this instrument can report.**

A bound of 0.5%/token on a metric whose ceiling is 1.5625% is a bound on
"how many prompts diverge at all" wearing per-token clothing — it allows 16
prompts of 50. Once the perturbation is large enough to flip *most* prompts
somewhere in 64 tokens, the metric saturates and stops discriminating between
"slightly too noisy" and "catastrophically wrong". **Zero-BUG-class and the
margin distribution are what carry information at this perturbation size**, and
both are healthy.

*(Related, recorded in `2026-08-13-m45-ruling-analysis-OPUS.md` §A2: the 0.5%
figure is itself 0.5938% rounded down, so the two halves of §8.1's conjunction
are not the same constraint. It does not matter here — 38 misses every
reading — but it matters for rung C, where it is the entire width of C-GRAY.)*

## 4. What this says about Q8-KV itself

- **It is sound.** Zero BUG-class over 50 prompts, all 38 divergences
  classified, no STEP in ten 36-layer curves, and divergent positions no
  noisier than agreeing ones.
- **It is not free.** It genuinely moves token decisions at near-ties on ~76%
  of prompts within 64 tokens. Anyone expecting byte-identical streams with
  Q8-KV on will not get them, and no amount of tuning changes that — it is the
  representation, not a bug.
- **Its measured error bound is better than the guarantee**: 0.4836·d against
  the `d/2 + 127·d·2⁻¹¹ ≈ 0.562·d` that ggml's f32-then-narrow order actually
  guarantees.
- **The scale-granularity question §8.1 anticipated is real but not urgent.**
  A finer-grained scale (per-16 rather than per-32) would cut the perturbation
  roughly in half at a bytes cost; nothing measured here calls for it, and it
  belongs to M5's dtype decision if anywhere.

## 5. What it feeds — and what is Kabir's

Q8-KV's *footprint* argument is independent of this result and unchanged: at
f32 the KV pool is 4.50 MiB/block, and **N=16/ctx2048 does not fit on this
15.63 GB box** (9.66 GB pool + 2.32 GB model). A KV dtype reduction is required
to reach M5's target here, not merely a lever. This gate says the Q8 option is
numerically sound; it does not say it should be the default.

**Recommended disposition (Kabir's call):**

1. **M4 stamps with item [4] recorded as measured-and-dispositioned** — zero
   BUG-class met, class-rate bound missed by construction, this memo cited.
   §8.1 forbids grinding it green, and there is nothing to grind: the
   engine is not wrong.
2. **f32 KV stays the default through M4.** Nothing here changes it.
3. **M5's design note carries the dtype decision** with this evidence and the
   footprint arithmetic side by side.

**Forward note, criterion-shaped (§8.4-reserved, so recorded rather than
enacted):** the class-rate bound wants the same per-pair form τ already has —
calibrated against the *perturbation class* being graded, not inherited from a
reduction-order pair. M3 raised the identical shape for `l2_diff.py`'s `1e-3`;
this is the second constant in the same family and the third distinct pair to
meet it. A bound that cannot be met by a sound implementation is not protecting
anything.
