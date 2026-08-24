# The Q4_K mins term in SIMD: the verify gate, and the default is flipped

**2026-08-24, Opus 5, branch `m5-entry`, commit under test `72dbf84`.**
Raw log: `2026-08-24-m5-mins-verify.txt`. Runner:
`tests/tooling/run_m5_mins_verify.ps1`. Wall time 46 min.

This is the gate the 08-20 attribution session owed before flipping the default
(`memos/2026-08-20-m5-attribution-RESULTS.md` §3: *"a change on the M3/M4 parity
lineage deserves the full 10-case verify gate against the archives, not three
ad-hoc cases"*). It is green, and **`STOKE_MINS_SIMD` now defaults to 1.**

## 1. The result

**14 cases × 2 arms = 28 stream comparisons, every one bit-identical to an
ARCHIVED binary.** No arm was ever graded against another arm.

| leg | cases | referent | result |
|---|---|---|---|
| `run`, chunk 128, Q4_K_M | p00/p41 × kv f32/q8 × t 2/6 = 8 | `stoke-m45-maddubs-f3b7181a` | 8/8 SAME, both arms |
| `run`, chunk 128, Q8_0 | p41 × t 2/6 = 2 | `stoke-m3-gate-876ce53f` | 2/2 SAME, both arms |
| `batch`, p00 | seqs 2/8/16 kv f32, seqs 8 kv q8 = 4 | `stoke-m45-maddubs-f3b7181a` | 4/4 SAME, both arms |

Arms: `scalar` = HEAD at the old default, rebuilt today; `simd` =
`-DSTOKE_MINS_SIMD=1`, identical CFLAGS otherwise via the Makefile's new
`EXTRA_CFLAGS` hook. The `scalar` arm is what separates *"the define broke it"*
from *"the rebuild broke it"*; it also proves this build of the tree is sound,
which is why no separate referent-validation run was needed.

## 2. THE PART THAT MATTERS: the gate carries its own proof it could fail

A third binary was built from the same source with **one wrong lane in the
duplication mask** (`dup[5] = 3` instead of `2`) and run *inside* the gate on one
case per leg:

```
Q4_K_M p00  kv=f32 t=2   scalar=SAME  simd=SAME  mutant=DIFFER(good)
batch  seqs=2 kv=f32 t=6 scalar=SAME  simd=SAME  mutant=DIFFER(good)
```

Without this, 28 SAMEs would have been equally consistent with **a kernel that
never executed**. That was not a hypothetical worry:

> ⚠ **THE 08-20 MEMO BELIEVED `stoke run` COULD NOT REACH THIS KERNEL.** Its §3
> says the byte-compare *"had to go through `batch --seqs N`, not `stoke run`"*
> because the kernel only runs at `n_seq > 1`. That is true of **decode** and
> false of the whole command: E3 chunks **prefill** through
> `model_forward_batch` (`model.c` `prefill_range`, default
> `STOKE_PREFILL_CHUNK=128`), and `vec.c` tiles `n_seq` at
> `KERNELS_BATCH_TILE=16` — so `run` on p00 enters the kernel at n_seq=16 then
> 13, and on p41 at 16 eight times per chunk then the remainder.
>
> **This was not accepted as an argument.** The mutant's DIFFER on the run path
> is what makes leg 1 evidence rather than 10 vacuous SAMEs.

The runner also refuses to start if the mutant is byte-identical to either
graded arm, which would turn the power check into a tautology it passes.

## 3. What this gate does NOT prove, stated so it is not counted twice

- **The two Q8_0 cases are a CONTROL, not evidence for this change.** A Q8_0 file
  has no K-quant tensors, so `matvec_batch` never takes the branch this kernel
  lives on and the define cannot reach it. They prove the rebuild disturbed
  nothing else. The log labels them inline.
- **Speed is not measured here.** That is the interleaved A/B, already banked:
  `2026-08-20-m5-mins-ab-v2-report.md`, 1.063×/1.099× at t=1 and 1.079×/1.071×
  at t=8, B winning 6 of 6 pairings in each run.
- **There is deliberately no `chunk1` arm.** The define is textually scoped to
  two `#if` blocks inside one function that only executes at `n_seq > 1`, so a
  chunk1 arm is byte-identical to the scalar arm by construction —
  `run_m5_entry_verify.ps1`'s header names the failure that avoids: a gate whose
  arms are secretly the same arm still passes, and reads like twice the evidence.
- **p41 is not in leg 2.** The referent is pre-pool and prefills 1050 positions
  with single-threaded attention; leg 1 already grades p41 against the same
  archive at length.

## 4. What the flip changes, and what it cannot

- **It cannot move P-BAR.** M4.5's bar is N=1 decode, which never enters the
  batched kernel. No re-measurement is owed.
- **It cannot rescue gate [2].** ~7–10% on prefill against a required 2.75× is a
  few percent of the way in log terms. Gate [2] still misses, exactly as
  `memos/2026-08-20-m5-attribution-RESULTS.md` §4 said.
- **It does change `serve` and `batch` throughput**, and prefill inside `run`.

## 5. Instrument notes (three, all caught before they could publish anything)

1. **`H` is not a free function name.** A helper named `H` resolved to
   PowerShell's `h` alias for `Get-History`, which took the entire model stream
   as an `-Id`. Aliases beat functions in name resolution. Same family as
   `run_m5_e2_verify.ps1`'s `$args` trap; renamed `Hash12`.
2. **The staleness guard fired twice, once rightly and once wrongly.** Rightly on
   the arms (a `git checkout` had bumped `simd.c` past their build time — they
   were rebuilt). Wrongly on the **mutant**, which is built from a tree that was
   deliberately modified and then reverted, so no newest-source comparison can
   ever be satisfied by it. It is now exempt, and what protects it instead is
   stronger than a timestamp: a stale mutant would MATCH and fail the gate.
3. **`build/win/stoke.exe` was NOT "byte-identical" to `build/ab/mins-scalar.exe`**
   as `memos/2026-08-20-m5-attribution-RESULTS.md` states — 15,264 bytes differ.
   Their **`.text` sections are identical**, so the substantive claim holds; mingw
   embeds build paths, and PE builds from different directories differ in
   non-code sections and in the absolute addresses those shift. **Byte-identity
   is the wrong test for two builds out of different directories** — compare
   `.text`, or build both in the same directory.
