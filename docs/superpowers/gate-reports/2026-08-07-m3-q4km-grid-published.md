# [2v2-iii] Q4_K_M grid — PUBLISHED BEFORE THE RUN, with one correction to the ruling's band arithmetic

**Status: decision of record, Opus 5 session 2026-08-07.** Authority for the
criterion remains `2026-08-06-m3-probe-grid-RULING-FABLE.md` (§13 governs);
this file discharges its §5 requirement that the grid be published before it
runs, records the per-prompt coverage denominator, and corrects one factual
assertion in §5 that arithmetic falsifies.

**Governance note.** Kabir instructed on 2026-08-07 that decisions no longer
route to a Fable session — Opus decides, records the reasoning and a direction
check, and Kabir overrules. The 08-04 ruling's §8.4 escalation target changes
accordingly; every ruling already issued stays binding exactly as written.
This file is the first decision made under that change.

---

## 1. THE CORRECTION — `[100,300)` is occupied, and τ_Q4KM gets three bands

Ruling §5 states:

> The G5 grid's positions lie in [0,100) ∪ [300,600); [100,300) hosts no G5
> position and gets no τ_Q4KM.

**That is false.** `[100,300)` hosts **45 of the 384 covered positions**.

**Why the ruling's reading fails.** `--probe-all` records a
*generation-relative* index, not an absolute position —
`tests/tooling/ref_forward.py:332` appends `{"index": len(out_ids), ...}`,
where `out_ids` holds only the generated tokens. Absolute position is therefore
`prompt_len + index`, and the ruling's band assignment omitted `prompt_len`.
Two of the five short prompts are long enough to cross 100: p20 is 64 tokens
and p30 is 53 (`tests/data/parity/MANIFEST.md`, counts produced by
`stoke tokenize`).

**Coverage table — the denominator §5 requires, published before the run:**

| prompt | tokens | absolute range | `[0,100)` | `[100,300)` | `[300,600)` | total |
|---|---:|---|---:|---:|---:|---:|
| p00 | 23 | 23–86 | 64 | — | — | 64 |
| p10 | 24 | 24–87 | 64 | — | — | 64 |
| p20 | 64 | 64–127 | 36 | **28** | — | 64 |
| p30 | 53 | 53–116 | 47 | **17** | — | 64 |
| p45 | 14 | 14–77 | 64 | — | — | 64 |
| p41 | 477 | 477–540 | — | — | 64 | 64 |
| **total** | | | **275** | **45** | **64** | **384** |

**Decision: τ_Q4KM is frozen for all three occupied bands.**

This applies §5's rule rather than amending it. The operative sentence is
"select **20 agreeing positions per occupied band** by rule"; *which* bands are
occupied is a fact the corpus determines, and the parenthetical asserted that
fact incorrectly. §8.4 reserves frozen constants and criterion text — not
arithmetic that the criterion's own inputs decide.

**Direction check.** Three τ bands instead of two means more positions are
gradable and none are graded under a borrowed constant: strictly more work,
strictly harder, no bar moved. The alternative is worse in both directions — a
divergence at p20:100–127 or p30:100–116 would either trip §5's fail-safe
mid-leg ("a graded position landing in a τ-less band stops grading") or, if the
fail-safe were read past, be graded under `[0,100)`'s τ, which is a borrowed
bar of exactly the kind this milestone has now retired twice.

**Why the ruling's own fail-safe is not sufficient on its own.** It stops
grading pending "a freeze from added samples of that band" — but no added
samples are needed here: the band carries 45 in-grid positions of its own, ~44
of which are expected to agree at the measured ~0.5%/token base, well above the
20 the rule asks for. Discovering this at grading time would cost a stopped leg
for a freeze that can be done in the same pass as the other two.

## 2. THE FROZEN SELECTION PROCEDURE — recorded before any position is graded

Per §5, τ_Q4KM(band) = `max(0.10, 2 × band-p95)` of the recorded §2.3/§3.6
Δ-profile, computed over rule-selected agreeing positions, with the stoke side
read via `stoke --probe-pos` on the frozen binary (G11).

Selection rule as written: every `⌈N_agree/20⌉`-th agreeing position in
(prompt-order, index-order). At the no-divergence coverage above this yields:

| band | N_agree | step `⌈N/20⌉` | selected |
|---|---:|---:|---:|
| `[0,100)` | 275 | 14 | 20 |
| `[100,300)` | 45 | 3 | 15 |
| `[300,600)` | 64 | 4 | 16 |

The rule is applied exactly as written; the counts below 20 are an artifact of
ceiling-then-dividing at small `N` and are **not** adjusted to reach 20 —
tuning a frozen selection procedure to hit a round number is the move this
project forbids. The counts are recorded here so the freeze is reproducible.
Filed forward for the M4-entry cluster: restate the rule as `step = max(1,
⌊N/20⌋)` for future pairs, where it delivers what it says.

`N_agree` is recomputed from the run's actual coverage before selection; the
table above is the zero-divergence case. Q8_0's τ values
(0.519674 / 0.413566 / 0.394898) **do not transfer** — G6, per-pair always.

## 3. UNCHANGED

Everything else in §5 stands as ruled: the six prompts and their selection
rule; RUN A's composition including the p00 bare-vs-`--probe-all` pair that
discharges §8A's OWED stream-invariance item (any mismatch stops the leg with a
tooling memo); RUN B; one required continuation per first divergence; §6
classification under §11's cross-class ladder reading (STEP decides, ≤1e-3 void
cross-class); tripwire ≥8 graded flips; zero BUG-class as PASS; the G11 freeze
to `git tag m3`; the ops constraints.

## 4. LAUNCH CONDITION — not met at the time of writing

Q4_K_M is 2,497,280,736 B (2.326 GB), so §5's condition is
`ullAvailPhys > 4.326 GB`. Measured this session: **4.34 GB** — a 14 MB
margin, which is not a margin. The recorded failure mode is a page-cache
collapse from an application opening *after* launch (cache faults 60 → 7,000,
~3× slowdown, observed twice), which would take RUN A past the ≤4 h window it
is authorized under. RUN A launches when the box has real headroom, announced
at launch per §8.1.

Tree clean at `d456213`. G11 freeze intact: `build/win/stoke.exe` unmodified
since 2026-08-04, no rebuild performed or planned.

## 5. CARRIED FORWARD — from this session's review of all five rulings

Two items for the M3 stamp's carried-headroom list, neither blocking:

1. **The Q8_0 leg's referent substitution narrows what [2v2-iii] catches, and
   the narrowing has a name.** `ref_forward.py` is independent of stoke's
   entire C lineage; the archived M2 binary shares everything predating K2
   (GGUF parse, rope, RMSNorm, attention structure). M2↔HEAD therefore catches
   a K1–K6 defect — which G2 correctly identifies as the leg's purpose — but is
   blind to a spec misreading M2 and HEAD hold in common. That is the exact
   limit the 2026-07-28 ruling named and expected "M3's gate" to retire; it is
   now carried at M2's density (24 prompts) rather than retired. Recorded with
   a location and a date, per §8.11.
2. **`1.4e-5` is a single-position measurement**, not a bound — p02:16, max Δ
   over the top-5 union (`2026-08-04-m3-gate-item-2-FOR-FABLE.md` §3.6). The
   08-04 ruling discounted that same probe's other row (0.0430) with a 0.10 τ
   floor "because 0.043 is a single-position measurement"; the 08-06 ruling
   then treated 1.4e-5 from the same row as a bound. The G2/G3 verdict survives
   comfortably — reaching the smallest frozen τ (0.394898) would require the
   figure to understate by ~4.5 orders — but it should be cited as robustness
   to a large underestimate, never as a measured bound.

Also noted, no action: §5's tripwire recalibration states a false-alarm rate of
~0.06% for ≥8 flips at λ≈1.92; the Poisson tail is ~0.083%, against the
original ≥4/55's ~0.013% (≥9 would have matched it). The recalibrated tripwire
is thus ~6× readier to fire than "same order" suggests — stricter, not looser,
so it stands as ruled.
