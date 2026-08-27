# PRE-REGISTERED — what are the activation RE-READS costing the batched Q4_K kernel?

**2026-08-27, Opus 5. Written and committed BEFORE the first cell was timed.**
Instrument: `stoke bench --matvec-batch` (new, `src/bench.c`). Runner:
`tests/tooling/run_m5_actreuse.ps1`.

## The question, and why it is the last named one

Gate [2] attributed the prefill gap to llama.cpp at **3.22× at every thread
count**, with stoke's thread scaling *better* than llama's — so the gap is
per-thread kernel efficiency, not the pool and not M5's scheduler. Two ablations
since have each come back **under their own count**:

| ablation | what was removed | bought |
|---|---|---|
| 2026-08-20 mins → `vpmaddwd` | ~40% of emitted instructions | ~7% |
| 2026-08-24 `vpmulld` → `vpaddd` | half the p01 µops in the inner loop | 12.1% |

Together they cap at **~1.232× ≈ 18%** of a 3.22× gap. The front-end theory and
the port-throughput theory are both dead, and what was left on the list is a
**memory** hypothesis, recorded but never measured:

> per super-block the kernel reads **144 B of weights** against
> **n_seq × 292 B of activations**, and re-reads that whole activation tile
> **for every output row** — a 28:1 byte ratio.

At `n_in` 2560 and `n_seq` 16 the re-read tile is **45.6 KiB per row**, right at
this box's L1d; at `n_in` 9728 (`ffn_down`) it is **173 KiB**, unambiguously L2.

## The arms

- **A** — `n_seq` distinct activation-scratch buffers. The real case.
- **B** — the same call with every activation pointer **aliased to one buffer**.
  Identical instruction stream, identical load count, identical arithmetic
  shape; the only difference is *which cache lines the loads touch*.

The alias is applied **in the caller**, not in the kernel, and that is
load-bearing: editing the kernel to read `arows[0]` would make the whole
per-sequence body loop-invariant, GCC would hoist it, and the ablation would
credit itself with 15/16 of the work removed — an ablation measuring its own
compiler. Sixteen pointers that happen to be equal cannot be proven equal at
compile time, so every load still issues. **One binary, one code path, two
inputs** — which also makes this immune to the code-layout artifact that forced
the `vpmulld` arms into equal-length build directories.

**A/B is an UPPER BOUND** on what perfect activation reuse could buy. A real
redesign (cache-blocking the row loop over a resident tile) still has to read
each activation at least once.

## The arithmetic this prediction comes from

Per (sequence, super-block) the inner loop issues, from the source and the
2026-08-24 disassembly: 8 × (`vpmaddubsw` 1 + `vpmaddwd` 1 + `vpmulld` 2 +
`vpaddd` 1) ≈ **40 p01-class µops**, plus the mins `vpmaddwd` and two horizontal
sums ≈ **42–46**. At 2 p01 ports that is **~21–23 cycles**. The same block issues
**10 loads / 292 B**, which at 2–3 loads per cycle is **~4 cycles** — about 19% of
the cycle budget, and from L1 they overlap the ALU work entirely.

Required activation bandwidth in arm A is therefore **292 B / 21 cycles ≈ 14
B/cycle**. L2 on this box sustains well above that.

⇒ **I expect the re-reads to be CHEAP, and I am saying so before measuring.**

## Predictions

| # | cell | prediction |
|---|---|---|
| **P1** | `attn_q`, N=16 (45.6 KiB tile) | **A/B ≤ 1.15** |
| **P2** | `ffn_down`, N=16 (173 KiB tile) | **A/B ≤ 1.25** — larger than P1, since the tile is 3.8× bigger and cannot be L1-resident, but still not the answer |
| **P3** | any tensor, N=1 | **A/B = 1.00 ± 0.03** — the CONTROL. `vec.c` routes `n_seq == 1` to the solo worker, so there is nothing to alias and a deviation here is harness bias, not physics |

## Discriminators, fixed now so the number cannot choose them

- **A/B ≥ 2.0 anywhere** ⇒ the memory hypothesis **IS** the gap. The fix is
  cache-blocking the row loop over a resident activation tile, and it becomes
  M5's lever.
- **A/B ∈ [1.25, 2.0)** ⇒ real but partial. It joins the ~18% of named compute
  levers and the 3.22× stays unexplained.
- **A/B ≤ 1.15 in every cell** ⇒ **the activation re-read hypothesis is DEAD.**
  With the front-end and port theories already dead, *nothing named remains*,
  and the honest reading is that the 3.22× is llama's **algorithm** — tinyBLAS
  register-blocked GEMM reusing weights AND activations out of registers —
  rather than a defect in this kernel. That is a different kind of answer and it
  changes what M5 should spend on. **This branch is written down because it is
  the one where I would otherwise be tempted to keep hunting.**

## Withdrawal conditions

1. **The power check fails** — arm B reproduces arm A. The alias never took
   effect and nothing was measured. (Asserted in the cell, which refuses to run.)
2. **The fixture is degenerate** — sequence 0 and sequence N−1 agree, so the
   activations did not differ from each other and both checks pass vacuously.
   (Also asserted; this is the failure mode ADDENDUM 13 recorded.)
3. **The N=1 control is not ~1.00** — the harness has a bias and no cell in the
   table is readable.
4. **A cell reports A/B > 1.15 while B wins fewer than half the pairings** — the
   median is not supported by the pairings and the cell is noise, not an effect.

## Protocol

Arms **alternate within each cycle** and the **order flips per cycle**, so a box
warming across the run cannot charge one arm the warmer half. Single-threaded
(`bench` never starts the pool), pinned to one P-core by the runner, because
[[measurement-needs-interleaving]] prices unpinned E-core migration at up to
5.8× and thermal drift at 22% on compute-dense work. The table sweeps
N ∈ {1,2,4,8,16} so the effect can be read against the tile size **as a curve**,
not as a single ratio — if the hypothesis is right, A/B must GROW with N and the
knee must sit where the tile crosses a cache capacity. A flat curve is itself
the answer.
