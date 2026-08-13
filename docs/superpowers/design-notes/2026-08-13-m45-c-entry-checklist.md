# M4.5 rung C — entry checklist

*2026-08-13. Opus 5. Kabir authorized C explicitly ("authorization granted",
after "matching llama first according to the ruling"), which answers the
ruling's §8 direction check. **Authority for HOW is the 08-12 ruling §7.2; this
file only sequences it and folds in the corrections from
`memos/2026-08-13-m45-ruling-analysis-OPUS.md` so they are not re-derived.***

## Why this file exists

C is the milestone's one irreversible step and its atomic unit is large: the
new scalar IS the reference, so it cannot land without its matching twin, and
the K-quants cannot land half-converted. Starting it needs a session with room
to finish. This is the state to start from.

## Where C starts (state at `f91e44e`)

- M4 closed, tag `m4`. Rung B complete for both K-quants, +32.5% at N=16.
- **N=1 = 10.01 tok/s; llama.cpp b10037 = 18.05; ratio 0.555×.**
- **P-BAR = 0.85 × 18.05 = 15.34 tok/s ⇒ C must deliver 1.53×.**
- Referent binary for the graded run: `build/stoke-m4-gate-f4f9b667.exe`.
- `n_seq==1` routes to the UNFUSED worker (`vec.c`), so N=1 measures the
  single-row kernels — which is exactly what C changes.

## The order §7.2 fixes, with what to watch at each step

1. **Scalar rewrite = the definition.** `vec_dot_q4_k_q8_k` /
   `vec_dot_q6_k_q8_k` to the canonical fold: per superblock in file order,
   `acc += (f16_to_f32(d_b) * ad_b) * (f32)S_b` then
   `acc -= (f16_to_f32(dmin_b) * ad_b) * (f32)M_b`, one `acc`, ops in that
   order, `-ffp-contract=off` already pinned. **Mark it `/* CONTRACT */`** per
   the ruling's standing rule. Q6_K has no mins term.
2. **AVX2 twins, llama-shaped.** `maddubs` (32 products/uop) + `madd` fusing
   the scales, 32-byte loads, integer accumulate, **then reduce the 8 lanes to
   the single `S_b` and execute the scalar's two f32 ops verbatim.**
   Saturation bounds, both verified correct in the analysis pass: Q4_K
   `15·127·2 = 3,810`; Q6_K `63·127·2 = 16,002`; both < 32,767. Keep Q6_K
   quants unsigned through maddubs and subtract the 32-offset via `32·Σq8`
   from bsums.
3. **Verification, in the ruling's order.** (a) `test_kquant`/`test_simd`
   twins vs the NEW scalar over the existing 1e4 corpora — passes BY
   CONSTRUCTION or the reduction is wrong; (b) mutation: swap the two f32 ops
   in the twin, the test MUST fail, restore; (c) **Q8_0 anchor: p00 + p41
   byte-identical to the m3 archive** (Q8_0/F32 kernels untouched — a flip
   here is `C-ANCHOR-BREAKS`, a hard stop); (d) the graded old-vs-new
   50-corpus run; (e) matrix + WSL plain/asan/tsan; (f) archive the new gate
   binary and ledger the referent retirement.

## Corrections to carry in — do not re-derive these

- **The grading bound is ≤19/50, not the literal ≤0.5%/token.** Recomputed
  from `m2_sweep.json`, the M2 census is 19/3200 = **0.5938 %/token**; the
  "~0.5%" in both rulings is that same datum rounded down. (Analysis §A2.)
- **C-GRAY's whole width is that rounding** — the band is exactly 17/18/19
  prompts. Under the corrected bound it is empty.
- **C-FAIL-RATE's flip trigger (>3.10 %/token) is UNREACHABLE** on this
  instrument: one first-divergence per prompt over 3,200 tokens caps the
  metric at 50/3200 = **1.5625 %**. Only its ">19/50" half can fire.
- **The fold's dependency chain is NOT obviously noise.** ~40–60 cycles per
  n=2560 row against ~60–75 cycles of llama-shaped integer core — the same
  order, *because* C makes the integer work small. Rows are independent so an
  OoO core overlaps them, and A′ (row tiling) is the designed mitigation.
  **If C lands low, this is the first suspect and A′ is the answer** — which
  is what §7.2's C-SLOW branch already says. Refit `c`/`f` after C.
- **Expect 15.4–19.2 tok/s** (C's own 1.54–1.92× applied to the re-anchored
  10.01) against a 15.34 bar. A landing at 12–13 is *below plan*, not merely
  unimpressive.

## The debug oracle — build it before the first bug, not after

§7.2's decisive oracle: **old scalar and new scalar must produce BIT-IDENTICAL
per-superblock integer totals** (Σ old `aux32[]` == new `S_b`; old `sumi` ==
`M_b`), because C changes only the f32 fold. So any C bug bisects in one dump:

- totals differ old-vs-new ⇒ the new **integer core** is wrong (fix against
  the old);
- totals match but streams grade BUG ⇒ the **fold sequence** differs from the
  CONTRACT lines;
- twin-vs-scalar likewise: totals must match at every superblock.

## Measurement discipline, earned the hard way on 2026-08-13

**Interleave or the number is fiction.** This box loses 15–20 % over a couple
of hours of sustained load: the same binary measured 13.565 then
11.581/10.941/12.003 forty minutes apart. Worse, the drift is *load-dependent*
— the pre-B arm held a 1.0 % spread across cycles while both faster arms swung
~10 %, so **a sequential A-then-B reported the wrong ordering between two
arms**. P-core pinning (`0xFFF`) fixes short-term jitter and does nothing for
this. Every C number goes through `build/ab/run-interleaved.ps1`'s shape:
arms cycled inside one thermal state, 3 cycles, medians, raw kept.
