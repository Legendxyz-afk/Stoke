# M7 / B6 — T1 fired at 2.396%. Anomaly memo, and the decision.

**Required by the B6 pre-registration
(`2026-09-14-m7-b6-unsloth-probe-preregistration.md` §5): "T1 divergence rate
> 2% of probed positions ⇒ stop, anomaly memo, no stamp." 23 of 960 probed
positions diverged. This is that memo.**

Decided under the 2026-08-07 governance change: Opus decides §8.4-reserved
questions, records the reasoning and a direction check, **Kabir overrules.**
Report: `docs/superpowers/gate-reports/2026-09-15-m7-b6-unsloth-probe.md`.

## 1. What fired

23 divergent positions in 960 probed = **2.396%**, against T1's 2%. Every one
of the 23 classified **ARITHMETIC-CLASS** under a τ frozen before grading;
zero BUG, zero ESCALATE; worst margin/τ ratio 45% (`L-2851` 2896:
0.683926 / 1.520234). T2, T3, T5 and T6 are quiet. T4 fired separately and
has its own pre-registered test (report §7).

## 2. Why it fired, and why that is the carried expectation, not a surprise

The pre-registration wrote the expectation down before the run, so that it
could be falsified rather than fitted:

> the bartowski pair's M3 grid ran at 3.10 %/token class-flip base on Q4_K_M
> … against T1's 2%. If this file reads near 3% too, T1 fires and the memo
> says so — the threshold is not raised to fit a second Q4_K_M file. The B5
> Llama grid read 1.367%.

T1's 2% was set on 2026-09-07 for the Llama-3.2 entry probe and carried to
this Q4_K_M Qwen3 file unchanged. The three measured rates for the
`f32 reference ↔ stoke integer core` pair:

| pair / file | probed | divergent | rate |
|---|---|---|---|
| Q4_K_M bartowski Qwen3-4B, M3 grid (2026-08-11) | 290 | 9 | **3.10%** |
| Q4_K_M Llama-3.2-3B, B5 entry probe (2026-09-07) | 1024 | 14 | **1.37%** |
| **Q4_K_M unsloth Qwen3-4B, this grid** | 960 | 23 | **2.40%** |

The unsloth file sits between the two, below the only other Qwen3 Q4_K_M
rate on record. Under the bartowski rate the expectation for 960 positions
is 29.8 divergences and 23 is a *low* draw (P(X ≤ 23) = 0.12); under T1's own
2% line the expectation is 19.2 and 23 is unremarkable (P(X ≥ 23) = 0.22 —
and the line itself fires at ≥ 20, which a true 2% pair does 46% of the
time); only under the Llama rate is 23 improbable (P = 0.008), and the Llama
rate is a different model. The mechanism is the one M3's memo named: 4-bit
weights put the f32 reference and the integer core further apart than 8-bit
does, so more near-ties resolve differently; a second imatrix (79 chunks vs
497) does not change that.

## 3. Is the excess structural? — checked, not assumed

M3's memo listed what would change its decision: "evidence that the flip
excess concentrates somewhere structural — a single layer, a single token
class, one prompt tier — rather than distributing as near-tie noise."

- **By band:** 3/277, 6/160, 2/75, 6/192, 6/256 (1.1% / 3.8% / 2.7% / 3.1% /
  2.3%). The long bands `[600,1200)` and `[1200,∞)` — probed on a Qwen3 file
  for the first time — are at the rate, not above it.
- **By stream:** 0,0,1,1,1,1,2,2,2,2,3,3,4 and 0 divergences per 64; the
  largest, `L-900` at 4/64, is a P = 0.07 draw under 2.4%, and one such
  stream in fifteen is expected. Nothing concentrates.
- **By margin:** every divergent position's worst margin is < 45% of its τ;
  the top-5 sets agree modulo the rank-5 tolerance at all 23. These are
  near-ties resolving differently, which is what ARITHMETIC-CLASS means.

The one structured feature of this grid is *not* among the divergences: it is
the anchored-Δ tail at *agreeing* positions in the first generated tokens of
the three ≥1893-token streams (report §3 and §7). It widens τ; it does not
add divergences.

## 4. The limit of that argument — carried headroom, as M3 recorded it

τ is frozen from agreeing positions of the same pair it grades, so a defect
that inflated logit differences uniformly would inflate τ and the margins
together and stay invisible to this criterion. `ref_forward.py` is
independent of stoke's C lineage, which is the main protection; a spec
misreading shared by both sides is the blind spot the 07-28 ruling named. The
`[1200,∞)` τ of 1.520 is the widest ever frozen for this pair family, and
report §7 records what was done about that.

Carried headroom at the B6 mark, same shape as M3's entry: 2.40% divergence
on the unsloth Q4_K_M pair (vs 3.10% bartowski, 1.37% Llama), this memo, §4,
2026-09-15.

## 5. Decision

**Keep T1 at 2%, exactly as written. Do not recalibrate it from this grid.**
A threshold set from the data it polices is the circularity the τ discipline
exists to prevent; if a per-file T1 is ever wanted it freezes its rate from a
declared holdout, and none is proposed.

**T1 is therefore reported as FIRED, with this memo as its disposition.** The
question it proxies — are there more class divergences than the arithmetic
explains? — is answered by the file's own family rate (below bartowski's),
and the defect question is answered independently by 23/23 ARITHMETIC-CLASS
under a τ frozen first.

**Direction check.** Nothing is weakened: zero-BUG is untouched and met on its
own terms; τ was frozen before grading, from disjoint data; no probe was
discarded or added after the run; the threshold stands and will fire again
on the next Q4_K_M Qwen file, which is the correct behaviour for a detector
calibrated on Llama. The strictly-harder alternative — refusing the rung
until a Q4_K_M Qwen grid reads ≤ 2% — is unreachable at this pair's measured
density (the M3 memo's argument, unchanged), so it is not stricter, it never
closes.

**What would change this decision:** any divergence classifying BUG; a
recomputation showing the bartowski rate was not 3.10% after all; excess
concentrating in one band, one stream or one token class; or the §7 probes
in the report showing the boundary tail to be a stoke-internal inconsistency
(they did not — B and C both held).

## 6. What this memo does not do

It does not stamp gate item [2] and it does not tag. The grid is complete,
the pass rule is met, two tripwires fired and are dispositioned here and in
report §7 — and the person whose gate this is should say so before the
ROADMAP row reads ✅.
