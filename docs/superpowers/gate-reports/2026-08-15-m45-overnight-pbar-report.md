# M4.5 overnight P-BAR evaluation

Run 2026-08-15-m45-overnight-pbar. Arms: A = pre-maddubs (`bc9dd3d6`), B = maddubs (HEAD), L = llama-bench b10037.
N=1 via `batch --seqs 1`, p00, -n 128, 6 threads, P-cores. Arm order rotates per cycle.

## Correctness (owed re-runs)

- **IDENTICAL** -- Q4_K_M p41 relinked vs pre-maddubs
- **IDENTICAL** -- Q8_0 p41 anchor vs m3 archive

## Rates by block

| block | A pre-maddubs | B maddubs | L llama | B/A | B/L | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| 1 | 8.262 | 12.523 | 18.11 | 1.516x | 0.691x | 0.19 âš  disk-bound | 12 |
| 2 | 8.319 | 12.609 | 18.26 | 1.516x | 0.691x | 8.08 | 0 |
| 3 | 8.171 | 12.536 | 17.78 | 1.534x | 0.705x | 7.49 | 7 |
| 4 | 7.823 | 12.044 | 17.66 | 1.540x | 0.682x | 7.65 | 23 |
| 5 | 8.264 | 12.641 | 18.2 | 1.530x | 0.695x | 7.45 | 17 |

A block whose standby cache is under 2.0 GB against a 2.33 GB model was reading the model from DISK and its rates are not comparable -- free RAM reads healthy while the model is entirely absent from cache. Drop such a block and say so rather than averaging it in.

## Overall medians

- A pre-maddubs : 8.262
- B maddubs     : 12.523
- L llama-bench : 18.1

## The anchor guard

Arm A measured **8.262** against its quiesced anchor of 10.01, a deviation of **-17.5%**.

âš  **A did not reproduce its anchor within 10%.** Every ratio computed against it is suspect: a depressed baseline inflates the speedup. Treat the B/A column as UNRELIABLE and read B/L instead, which does not depend on A.

## P-BAR

P-BAR is 0.85x llama measured in the same sitting: **15.39 tok/s**. B measured **12.523**.

**P-BAR NOT MET** -- short by 1.229x. A' (row tiling) is the next lever and is not optional.

## Stability across blocks

Arm B across every run: min 11.636, max 12.759, spread 1.10x over 15 runs spanning the night.

A spread materially wider than a few percent means the box drifted and the medians are the honest summary rather than any single figure.

Bank: `2026-08-15-m45-overnight-pbar-bank.jsonl`. Raw log: `2026-08-15-m45-overnight-pbar-raw.txt`.

---

## Addendum, written after the run by the session that set it up

Two annotations above are generated boilerplate and are WRONG for this run.
Both are corrected here by measurement rather than by argument.

**1. Block 1 is NOT disk-bound, and must not be dropped.** The generated line
says a block under 2.0 GB standby "was reading the model from DISK". Blocks 1
and 2 settle it: block 1 ran at **0.19 GB** standby and block 2 at **8.08 GB**,
and their ratios are **identical to three decimals** (B/A 1.516, B/L 0.691 in
both). The model is mmap'd, so it faults into the running process's working set
and stays mapped for that run, and `--timing` reports `load` SEPARATELY from
the tok/s. An empty standby list therefore costs the initial fault-in, not the
decode rate. Standby is a yellow flag on this instrument; the red flag is a
rate near 1.4 tok/s. **Block 1's numbers are good and are included.**

**2. The anchor guard misfired, and its conclusion is right for the wrong
reason.** It compares arm A against 10.01, a figure recorded at an UNKNOWN
generation length. Decode rate falls as the KV cache grows, so n=16 and n=128
are different workloads -- arm A measures 9.70-9.74 at n=16 in this session's
smokes and 8.2-8.3 at n=128. The -17.5% is a WORKLOAD MISMATCH IN THE GUARD,
not contamination. What certifies the box is that **llama-bench's tg128 is the
matched workload and reproduced its 18.05 anchor at 18.10 (+0.3%)**. Its advice
-- read B/L -- is correct regardless, because B/L depends on no anchor at all.

**The verdict is unaffected: P-BAR is NOT met, B/L = 0.692 against 0.85.**
Five blocks spanning five hours at different cache states and CPU loads agree
within 3.4% on B/L, so this is a settled result rather than a noisy one.

**maddubs is worth ~1.52x at N=1** -- large, real, and insufficient. The gap
needs a further 1.229x.
