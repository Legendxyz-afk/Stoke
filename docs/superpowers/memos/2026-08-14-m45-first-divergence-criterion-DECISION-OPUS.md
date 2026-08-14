# DECISION — the first-divergence count is retired as a gate criterion on Q4_K_M

*2026-08-14, Opus 5, max effort. §8.4-reserved (criterion text, frozen
constants) ⇒ under the 2026-08-07 governance change this is mine to decide,
with the reasoning and a direction check recorded, and Kabir overrules.*

## The decision

**On Q4_K_M, the 50-prompt first-divergence count stops being a bound and
becomes a reported statistic.** No replacement number is chosen. The gate on
that file is carried by what already carries it in substance:

1. **zero BUG-class** — unchanged, untouched, still absolute;
2. **every divergence classified** under §6, with τ frozen before grading;
3. **every ESCALATE laddered**, no sampling.

C-FAIL-RATE and the `≤19/50` conjunct do not apply to a Q4_K_M grading. A run
reports its count on its face and the count decides nothing by itself.

## Why not simply re-derive the number

It would have been legitimate to. The M3 tripwire memo §5 says a recalibration
"must freeze its rate from a **declared holdout**", and NULL-1 is exactly that:
a separately built, deliberately benign pair, measured independently of the
change it would judge, with its reading pre-registered before the run. My
earlier caution that using it would be "circular" was **wrong** — the
circularity that memo forbids is taking the threshold from the run it polices,
i.e. from C's own 34/50, which nobody proposed.

The reason not to re-derive a number is different and stronger: **no threshold
on this statistic can discriminate on this file.**

### Route 1 — the point estimates, across four orders of perturbation

| run | perturbation | N |
|---|---|---|
| NULL-1 | **zero semantic content** | 35/50 |
| rung C | ~1e-7 fold restructure | 34/50 |
| item [4] | 2.3e-3 KV requantisation | 38/50 |

Four prompts of total spread. The bound carried over from Q8_0 was 19.

### Route 2 — the paired test, which is the most powerful form available

All three runs used the same 50 prompts, so the comparison can be paired
(McNemar) rather than treated as independent samples. Only discordant prompts
carry information:

| pair | discordant (b / c) | exact two-sided p |
|---|---|---|
| NULL-1 vs rung C | 3 / 2 | ≈ 1.00 |
| NULL-1 vs item [4] | 4 / 7 | ≈ 0.55 |
| rung C vs item [4] | 3 / 7 | ≈ 0.34 |

**Nothing separates.** A change with no mathematical content is
indistinguishable from an 8-bit requantisation of the KV cache.

### Route 3 — the power calculation, which says how far short the corpus is

Taking the observed rates (p ≈ 0.70 null, 0.76 for the 2.3e-3 arm, Δ = 0.06),
the corpus size needed for a 3σ separation is

- **n ≥ 525** prompts if the null rate is treated as known exactly;
- **n ≥ 981** per arm if both arms are measured.

Against a corpus of **50**. The instrument is 10–20× short of being able to
resolve the difference it is being asked to police, and that is a property of
the statistic and the corpus, not of any threshold placed on it.

**A number chosen now would therefore be a threshold that cannot do its job.**
Set at the null point estimate (35) it rejects roughly half of all benign
changes, since binomial σ at p=0.7, n=50 is 3.24. Set with 2σ margin (≈41) it
sits above the 2.3e-3 arm and would pass a real requantisation defect. There is
no value that separates them, because they are not separated.

## Why this is not a weakening

The substantive question — *is this change worse than arithmetic-class?* — is
answered by the classification, not the count, and the classification is
untouched:

- **BUG-class remains zero-tolerance.** It is a per-divergence judgement
  against τ and the top-5 sets, not a census, so it does not inherit the
  count's power problem.
- **The ladder still adjudicates every ESCALATE.** Rung C's ten were laddered
  to no STEP beyond the calibrated L01 artifact, with every divergent curve
  below the second-lowest control.
- **τ discipline is unchanged**: frozen before grading, from positions it will
  not itself grade.

**This is the same move the project has already made once, for the same
reason.** The 2026-08-05 ruling retired item [2]'s 50/50 live-oracle bar as
*structurally unreachable* and replaced it with a classified-divergence
criterion — zero BUG-class plus ladder escalation — and M3 closed oracle-free
on that basis. This is that precedent applied to a second criterion that has
now been measured, rather than argued, to be unable to fire correctly.

## What is not decided here

- **Park-vs-revert on rung C is Kabir's**, and this memo does not pre-empt it.
  It removes the basis for calling C *failed*; it does not make C *adopted*.
- **Q8_0 is untouched.** The 19/50 census is a valid Q8_0 measurement and the
  criterion may still mean something on that file, where the near-tie density
  is roughly a quarter of Q4_K_M's (M3: 0.74 %/position vs 3.10 %).
- **M7's Llama-3.2 work inherits the question, not the answer.** Any new model
  file needs its own holdout measurement before a count criterion is applied to
  it, and this memo is the reason to check rather than assume.

## Carried headroom

The three-run comparison rests on **one** null measurement. A second holdout —
an independent benign pair, e.g. reversing the super-block loop order rather
than the lane order — would test whether 35/50 is stable or itself noisy. It is
not needed for this decision, because the decision is *not* to set a threshold;
it would be needed before anyone ever sets one.

## Direction check

This resolution weakens no bar that was doing work. Zero-BUG is untouched and
was met on its own terms in every run cited. τ was frozen before grading in
each. The ESCALATE ladders were run in full, not sampled. What it removes is a
count threshold that has now been shown — by a declared holdout, a paired test
and a power calculation — to be unable to distinguish a change that means
nothing from one four orders of magnitude larger.

The strictly-harder alternative, keeping `≤19/50` on Q4_K_M, is not
conservative but arbitrary: it fails a pair of binaries that compute identical
values in a different order, which the harness has now printed in its own words
(`GATE ITEM 4: NOT PASSED`, `BUG-class: 0`). Holding it would block the maddubs
twin — where rung C's throughput actually lives, and therefore P-BAR — on
evidence that has been measured to be meaningless.

**Kabir overrules.**
