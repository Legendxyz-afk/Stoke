# PRE-REGISTRATION — the NULL-1 base rate, and how it will be read

*2026-08-14, Opus 5. Written and committed BEFORE the graded run, for the same
reason τ is frozen before it grades anything: a reading chosen after seeing the
number is a fitted constant. §8.4-reserved (how a ruling constant is read) ⇒
mine to decide with a direction check, Kabir overrules.*

## Why

M4.5 rung C was stopped by C-FAIL-RATE at 34/50 first-divergence against a
bound of ≤19/50. Three facts about that bound are now on record
(`memos/2026-08-14-m45-c-results-vs-physics.md`):

1. The 19/50 census is **`tests/tooling/m2_sweep.py`, which hardcodes the Q8_0
   model at line 20**. C was graded on **Q4_K_M**.
2. On Q4_K_M the same instrument scored **38/50** for item [4]'s Q8-KV — a
   **2.3e-3** perturbation — against C's 34/50 at **~1e-7**. Four orders of
   perturbation, four prompts of range.
3. Every one of C's ten ESCALATEs is two near-ties changing places, and **none**
   involves a token the other side never ranked.

What is missing is the obvious control: **what does this instrument report on
Q4_K_M for a change that is definitionally arithmetic-class?** Without it, 34
is a number with nothing to compare against.

## The instrument

Two binaries built from the **same worktree at `ed2aa32` (tag m4)**, with the
same compiler and flags, differing by exactly one thing:

    -    for (int l = 0; l < 8; ++l) sumf += sums[l];
    +    for (int l = 7; l >= 0; --l) sumf += sums[l];

applied to the Q4_K scalar, the Q6_K scalar, and the twin's `kq_finish` so
scalar↔twin bit-exactness is preserved on both sides. **The same eight addends
in the reverse order.** There is no mathematical difference between the two
binaries at all — only f32 rounding order. This is the smallest arithmetic-class
change that can exist in this kernel.

- `build/stoke-null0.exe` sha256 `45ad5855…` — unmodified `ed2aa32`
- `build/stoke-null1.exe` sha256 `a1eb6d26…` — the reversal

**Validated before use, not assumed:**

- `null0` is **stream-identical to the m4 gate archive** (`f4f9b667`) on p00
  `-n 32` ⇒ the worktree build reproduces m4's behaviour, so the graded pair's
  only difference is the reversal.
- `null1` passes `test_simd` 12/12 and `test_kquant` 8/8 ⇒ the twins still
  agree with the scalars bit-for-bit.
- `null0` vs `null1` on three prompts at `-n 32`: p23 identical, p08 identical,
  **p35 DIFFERS** ⇒ the perturbation is detectable by this instrument, so a
  0/50 result is already unlikely.

Graded with the identical harness, identical corpus and identical probe-spec
as the C run: `m4_q8kv_gate.py`, 50 prompts, `-n 64`, both sides `--kv f32`,
τ frozen from this pair's own agreeing positions before it grades anything.

## The reading, fixed now

Let **N** be null-1's first-divergence count over the 50 prompts.

- **N ≥ 19** — **the ≤19/50 bound is disproven as a Q4_K_M criterion.** A
  change with zero semantic content would have failed it. C's 34/50 then
  carries no defect signal on this instrument, C-FAIL-RATE fired on a constant
  imported from another model file, and the bound must be re-derived on Q4_K_M
  before any fold change is judged against it.
- **1 ≤ N ≤ 18** — the instrument **has** resolution on this file and the bound
  is not absurd. C's 34/50 is then genuinely larger than a floor reordering.
  ⚠ This does **not** establish a defect: C's perturbation is larger than a
  lane-order swap by construction, so a larger count is expected either way.
  It makes the **matched** null the next instrument — m4's per-block fold with
  only the cross-block accumulation collapsed to serial, which isolates the one
  change today's accuracy measurement identified as a regression (2.0× worse at
  38 super-blocks) and predicts what a CONTRACT v2.1 would score.
- **N = 0** — the minimal null sits below the instrument's detection floor and
  is uninformative as a calibrator; go straight to the matched null. Partly
  falsified already by p35 above.

**What no outcome licenses.** None of these three readings adopts C, and none
condemns it. The oracle already settled the integer core; this settles what the
*bound* means. Adoption remains Kabir's park-vs-revert call (memo §6a), and a
re-derived bound would be a criterion change requiring its own record.

## Cost and disposition

One graded run, ~2.5–3 h of banked box time (the C run's shape), fully
deterministic — contention changes only how long it takes. Registered as the
on-demand task `stoke-m45-null1-baserate`; it runs when Kabir starts it.
Both binaries are **throwaway** and live only in `build/` (gitignored); the
worktree that produced them is disposable and the working tree was never
touched.
