# M4.5 — how to read the overnight run, and what each outcome means

*Written 2026-08-14 overnight by Opus 5, before the numbers exist, so the
readings are fixed in advance rather than fitted to whatever arrives. Same
discipline as the NULL-1 pre-registration.*

## Read in this order

1. `gate-reports/2026-08-15-m45-overnight-pbar-report.md` — **the anchor guard
   section FIRST**, before any ratio.
2. `gate-reports/2026-08-15-m45-overnight-pbar-bank.jsonl` — every unit, if the
   report looks odd.
3. This note.

## Gate 0 — is the measurement admissible at all?

**Arm A is the archived pre-maddubs binary, and its quiesced N=1 rate is known:
10.01 tok/s.** The report prints A's deviation from it.

- **|deviation| ≤ 10%** → the baseline behaved. Every ratio is usable.
- **|deviation| > 10%** → **the B/A speedup is inadmissible.** A depressed
  baseline inflates it, which is exactly what happened at 22:5x on 2026-08-14
  (A came back 4.971/7.445/8.387 under Discord, and the script would have
  printed a flattering 1.476×). **Read B/L instead** — llama-bench is an
  independent arm and does not depend on A at all. If L also deviates far from
  its own 18.05 anchor, the whole night is contaminated: re-run, do not report.

Blocks are spaced 80 minutes apart specifically so this is checkable per block
rather than once. If one block is contaminated and three agree, drop the one
and say so.

## Gate 1 — P-BAR

P-BAR is **0.85 × llama measured in the same sitting**, not against the
historical 18.05. The report computes it from L's own median.

### If B ≥ P-BAR — the bar is MET

Then per the ruling: **M4.5 CLOSES, and D (repack) is SKIPPED** — record why,
which is that the bar was met without it. Order of work:

1. Re-anchor K6 and budget-N (§8.2/§8.3), interleaved and quiesced.
2. Decide A′ on its own measurement — it is *not* needed for the bar at that
   point, and §7.3 explicitly permits keeping neither variant.
3. Stamp M4.5, `git tag m4.5`, local, no push.
4. Kabir's park-vs-revert on C becomes moot in the "keep" direction: C is what
   delivered the bar.

### If B < P-BAR — A′ is load-bearing

The smoke put B/L at 0.822 against 0.85, so this is the likelier branch.

1. **Verify A′ first**: `tests\tooling\run_aprime_verify.ps1`. Correctness
   before speed, always — and A′ must not move the streams at all.
2. Measure A′ under the same interleaved protocol, as a fourth arm against the
   same llama median.
3. **§7.3's rule stands: under a 5% gain the single-row path stays** and A′ is
   recorded as tried, not kept. Reverting is one line in `rows_q4_k_avx2`.
4. If A′ closes the gap → close M4.5 as above, D skipped.
5. If A′ does not close it → **D (repack) unlocks, and it is RAM-gated**: its
   gate is residency, not speed — model + repacked copy + KV pool must hold
   with `Page Reads/sec` < 5. Measure that before writing it.

## If A′ also falls short — D's residency gate, costed in advance

§7.3 sends a post-A′ miss to **D (repack), and D's gate is RESIDENCY, not
speed**: model + repacked copy + KV pool must hold with `Page Reads/sec` < 5.
That is measurable before a line of D is written, and the arithmetic is
already available from figures this project has measured rather than guessed:

| item | size | source |
|---|---|---|
| Q4_K_M model | **2.326 GiB** | `model_bytes` = 2,497,280,736, from the gate run.json |
| repacked copy | **≈ 2.33 GiB** | an 8-row interleave is a re-layout, not a re-encode, so it is the same byte count |
| KV pool, N=1 ctx 2048 | **0.28 GiB** | 147,456 B/token measured from the GGUF geometry (36 layers, 8 KV heads GQA 4:1, head_dim 128) × 2048 |
| KV pool, N=16 ctx 2048 | **4.50 GiB** | the same × 16 |
| box total | **15.63 GiB** | measured |

So N=1 needs ≈ **4.9 GiB** resident and N=16 needs ≈ **9.2 GiB**, against
15.63 GiB total — and the M4 record already showed the f32 KV pool at
N=16/ctx2048 committing and running on this box at 9.05 GB private with
`Page Reads/sec` flat at 0.0. **D is therefore RAM-feasible at both ends of
the envelope on paper**, and the gate is a measurement of occupancy rather
than a question of whether it fits.

⚠ Two caveats that belong with that number. The mmap'd original and the
repacked copy are only both resident during load, so the steady-state figure
is lower than the table implies — but the load-time peak is the one the gate
must survive. And the M4 note's own honest caveat still stands: commit is not
occupancy, and full occupancy at N=16 has never been driven (carried headroom
(b), still open).

## What is already settled, and should not be re-litigated

- **Rung C is not a defect.** Integer oracle clean over 108,000 super-blocks
  per quant type; all ten ESCALATEs ARITHMETIC-CLASS with no STEP beyond the
  calibrated L01 artifact; and the bound that stopped it was disproven on
  Q4_K_M by a zero-semantic-content change scoring 35/50 against it.
- **The first-divergence count is retired as a gate criterion on Q4_K_M**
  (`memos/2026-08-14-m45-first-divergence-criterion-DECISION-OPUS.md`). Do not
  re-apply `≤19/50` to a Q4_K_M grading. Zero-BUG plus classification plus
  ladders carry the gate.
- **maddubs moved no bits.** Byte-identical to the pre-maddubs binary on p00
  and p41, both models, plus 31 batch-identity streams.
- **Both owed correctness re-runs passed** in phase 1 of this very run.

## The one thing to be suspicious of

**A speedup that looks too good.** The honest expectation from the smoke is
roughly 1.5× at N=1 from maddubs, landing just under the bar. If the overnight
medians show something dramatically larger, check the anchor guard and the
per-block spread before believing it — this box has produced a wrong +29.5%
and a wrongly-stashed kernel from exactly that kind of number, and the whole
block-and-rotation design exists because of it.
