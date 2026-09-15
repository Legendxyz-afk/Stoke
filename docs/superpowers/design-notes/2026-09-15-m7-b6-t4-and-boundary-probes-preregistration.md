# 2026-09-15 — B6 follow-ups: T4's named test, and two probes for the boundary tail

**Written after the B6 grid was frozen and graded (`build/b6/`, 2026-09-15
09:51) and BEFORE any of the three probes below was run or compared.**

## 0. What fired, and what this file is for

The B6 forward run (pre-registration
`2026-09-14-m7-b6-unsloth-probe-preregistration.md`) banked 15/15 streams,
960 probes, 23 divergent, **all 23 ARITHMETIC-CLASS, zero BUG ⇒ PASS on the
pass rule**. Two tripwires fired:

- **T1** — divergence rate 2.396% > 2%. Disposition: anomaly memo, no stamp,
  threshold not raised. That memo is a separate document; nothing here bears
  on it.
- **T4** — `[600,1200)`→`[1200,∞)` band p95 0.386129 → 0.760117 = **1.969×**
  > 1.5×. The pre-registered action: *"bands stand; the rope-theta suspect is
  tested with `--rope-theta accum` as on 09-08."* Probe A below is that test.

And one thing no tripwire names: the largest anchored |Δ| at an *agreeing*
position is **11.877 at `L-1900` position 1893** (stoke top-1 logit 37.27 vs
reference 50.03, both choosing id 387), fifteen times that band's p95. Read
by generated index, the tail is **localized**: the first ~10 generated
positions of the three streams with prompts ≥ 1893 tokens (`L-1900` gen 1 =
11.88, gen 3 = 1.89; `L-2851` gen 0–3 = 1.48 / 3.50 / 2.81 / 1.76; `L-2500`
gen 5 = 1.84), absent in `L-1300` (1307 tokens), and past gen ~10 the long
streams read like the short ones (rest-max 1.1–1.4 against 0.6–1.9). A
sensitivity read on the banked envelope — **not a re-freeze; the banked
freeze stands** — says dropping gen 0–11 of those three streams moves the
`[1200,∞)` p95 to 0.588 (growth 1.52×) and per-stream p95 still climbs
`L-1300` 0.43 → `L-1900` 0.77 → `L-2500` 0.75 → `L-2851` 0.95. So the rise is
two things: a boundary tail, and an envelope that widens past ~1500.

Probes B and C exist to say which side of the pair the boundary tail belongs
to, cheaply, before any expensive arm is spent on it.

## A. The named T4 test — rope-theta closed vs accum (pre-registered action)

Exactly the 09-08 protocol (`2026-09-08-m7-rope-theta-ab-preregistration.md`,
tool `tests/tooling/m7_theta_ab.py`), on this pair:

- streams: the seven that feed the two bands and nothing else — `L-660`,
  `L-900`, `L-1120`, `L-1300`, `L-1900`, `L-2500`, `L-2851`;
- arms: phase A re-run under `--rope-theta closed` into `build/b6/theta/closed`
  and under `--rope-theta accum` into `build/b6/theta/accum`, same binary
  (`57475862…`), same model, `--set unsloth`;
- R1 stoke sides identical between arms; R2 the closed arm reproduces the
  banked `build/b6/tau.json` p95 and n for both bands to 1e-6; R3 null arm =
  the banked `build/b6/grid/L-2851.ref.json` against the closed re-run's
  `L-2851.ref.json` (two closed batched runs at fixed T, must be
  bit-identical); R4 the switch must change something;
- the bound threshold is **a quarter of THIS pair's banked rise**, derived by
  the tool from `--banked-tau` (`--bound-threshold banked`), not the Llama
  literal 0.0125 — (0.760117 − 0.386129)/4 ≈ 0.0935.

**Prediction:** **REFUTED** — `excess_removed` < 0.10, 2 × max Δ in
`[1200,∞)` far below 0.0935, 0 reference argmax flips, R1/R2/R3 quiet. Reason:
the 09-08 A/B measured the accumulation's effect as linear in position at
~1e-7 logit/position, three orders below the envelope; the mechanism is the
same code on this graph (`rope_neox`, base 5e6 vs 5e5 changes `theta_scale`
from 0.815 to 0.786 and nothing else); and a rope error is smooth in position
where the observed tail is not. **FALSIFIED IF** the verdict is EXPLAINED or
PARTIAL. On falsification the memo says so and the 07-31 suspect is reopened.

## B. Stoke against itself across the prefill→decode boundary

`STOKE_PREFILL_CHUNK=1` re-run of `L-1900` (and `L-2851`) with `--probe-all`,
same binary, same prompt file (`build/b6/grid/work/L-1900.prompt.txt`),
compared to the banked stoke top-5 at every generated position. Under
chunk 1 every prompt position goes through `model_forward_position`, the
decode path; under the default 128 it goes through `model_forward_batch`
(`prefill_range`). E3's design claim is that the two write the same KV, tested
with an anti-vacuity counter — at fixture size, never at 1893 real tokens.

**Prediction:** the 64 top-5 lists are **bit-identical** to the banked ones
(ids and logits). **FALSIFIED IF** any logit differs. On falsification the
boundary tail is a stoke-internal inconsistency between two of its own paths
— ESCALATE by the ruling's ladder (L2: bisect the path), independent of any
reference.

## C. The reference against itself at the boundary (C3 on this pair)

The batched reference at **T = 1895** over the first 1895 ids of `L-1900`'s
stream (`build/b6/grid/work/L-1900.stream.ids` truncated), probing
positions 1892, 1893, 1894, against the banked T = 1957 run
(`build/b6/grid/L-1900.ref.json`). Causal attention makes position p depend
only on ids[0..p]; what may differ is the GEMM shape's reduction order — the
09-07 C3 measured that at ~2e-5 on Llama out to position 1199. It has never
been measured on the Qwen3 graph or at these positions.

**Prediction:** max |Δ| over the shared top-5 ids at each of the three
positions **< 1e-3** (the T5 scale), argmax unchanged. **FALSIFIED IF** any
position reads ≥ 1e-3 — then the reference's batched arm is shape-dependent
at this position at a size that matters, and the licence C1-Q earned at T=87
does not reach it.

## D. What the three outcomes mean together

| A | B | C | reading |
|---|---|---|---|
| REFUTED | identical | < 1e-3 | the boundary tail is a genuine difference between two self-consistent implementations at those positions; ARITHMETIC-class by the ruling (argmax agrees, τ frozen over it); recorded as a finding; the third arm (llama.cpp logits at those positions — the optional survey the 08-05 ruling permits) is the named follow-up and Kabir's call |
| — | differs | — | stoke-internal: ESCALATE, bisect |
| — | — | ≥ 1e-3 | reference-side: the batched licence is scoped to short T on this graph; the serial arm or a smaller-T batched arm re-grades the long bands |
| EXPLAINED/PARTIAL | — | — | the 07-31 rope suspect is live on this graph after all; `[1200,∞)` is re-examined under it |

None of the three re-freezes τ or changes a verdict already banked. The B6
report is written from `build/b6/` as graded; these probes are appended to it
as §7.
