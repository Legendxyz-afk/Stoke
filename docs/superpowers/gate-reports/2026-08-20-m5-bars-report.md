# M5 gate item [2] -- the bars, measured

Run `2026-08-20-m5-bars`. S2 = 512 prompt tokens (525 real, ChatML) x 128 generated,
M0's own prompts (`tests/data/bench_prompts/`), ctx 704, threads 8, unpinned,
3 interleaved blocks. Raw: `2026-08-20-m5-bars-raw.txt`, bank: `2026-08-20-m5-bars-bank.jsonl`.

**The gate number is AGGREGATE(M0)**: decode-only ids over the span from the first
token anywhere to the last token anywhere -- `bench/bench.py`'s `aggregate_decode_tps`,
which is how the 17.7 / 14.2 / 10.8 / 16.8 cells were computed.

**Thermal state.** llama `pp512` by block: 71.29 / 72.24 / 73.15 -- these are taken at each
block's END, so they date the plateau and say nothing about block 1's own cells.
Stoke's own repeats: block 1 runs at **1.32x** the median of the later blocks
-- **block 1 is a WARMING TRANSIENT and is not comparable to the rest.**
Read the later blocks against each other; they are the stable-clock replicates.

## Medians by cell

| cell | S serve AGG(M0) | S decode-only (N/ITL p50) | S TTFT p50 s | B batch decode | B prefill tok/s | bar | verdict |
|---|---|---|---|---|---|---|---|
| 8f32 | 4.938 | 18.37 | 109.0 | 17.367 | 23.90 | &gt;= 17.7 | MISS (0.28x) |
| 16f32 | 4.594 | 18.35 | 205.2 | 17.936 | 24.04 | &gt;= 16.8 | MISS (0.27x) |
| 8q8 | 4.545 | 17.89 | 115.5 | 17.390 | 24.87 | &gt;= 17.7 | MISS (0.26x) |
| 16q8 | 4.550 | 17.32 | 206.1 | 17.513 | 23.74 | &gt;= 16.8 | MISS (0.27x) |

## The same numbers by block (read column 1 against the rest)

| cell | arm | block 1 | block 2 | block 3 |
|---|---|---|---|---|
| 8f32 | S | 6.378 | 4.659 | 4.938 |
| 8f32 | B | 23.071 | 17.367 | 16.671 |
| 16f32 | S | 5.905 | 4.594 | 4.470 |
| 16f32 | B | 22.653 | 17.936 | 17.887 |
| 8q8 | S | 6.193 | 4.496 | 4.545 |
| 8q8 | B | 22.980 | 16.130 | 17.390 |
| 16q8 | S | 5.049 | 4.233 | 4.550 |
| 16q8 | B | 17.470 | 18.219 | 17.513 |

## The non-stoke referent (llama-bench b10037, t 8)

- tg128 (decode, N=1): **17.48** tok/s -- the box-health anchor
- pp512 (prefill): **72.24** tok/s -- the prefill referent

## Every unit

| id | arm | cell | gate tok/s | wall s | cpu s | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| b1-8f32-S | S | 8f32 | 6.378 | 178 | 1193.8 | 3.16 | 81 |
| b1-8f32-B | B | 8f32 | 23.071 | 175.4 | 1190.2 | 2.77 | 8 |
| b1-16f32-S | S | 16f32 | 5.905 | 361.1 | 2443.4 | 2.44 | 27 |
| b1-16f32-B | B | 16f32 | 22.653 | 367.2 | 2482.8 | 2.6 | 31 |
| b1-8q8-S | S | 8q8 | 6.193 | 181.8 | 1216.6 | 2.78 | 34 |
| b1-8q8-B | B | 8q8 | 22.98 | 180.7 | 1212.4 | 2.85 | 24 |
| b1-16q8-S | S | 16q8 | 5.049 | 420.5 | 2836.2 | 3.21 | 4 |
| b1-16q8-B | B | 16q8 | 17.47 | 476.9 | 3206.9 | 3.3 | 30 |
| b1-L | L | llama | 17.03 | 53.4 | 408.1 | 3.31 | 21 |
| b2-8f32-S | S | 8f32 | 4.659 | 241.9 | 1621.9 | 3.8 | 0 |
| b2-8f32-B | B | 8f32 | 17.367 | 235.4 | 1576.6 | 3.79 | 0 |
| b2-16f32-S | S | 16f32 | 4.594 | 464.7 | 3143 | 2.43 | 0 |
| b2-16f32-B | B | 16f32 | 17.936 | 464 | 3136.9 | 2.53 | 51 |
| b2-8q8-S | S | 8q8 | 4.496 | 248.7 | 1667.8 | 2.67 | 0 |
| b2-8q8-B | B | 8q8 | 16.13 | 250.9 | 1683 | 3.17 | 4 |
| b2-16q8-S | S | 16q8 | 4.233 | 504.2 | 3392 | 3.23 | 8 |
| b2-16q8-B | B | 16q8 | 18.219 | 463.8 | 3122.5 | 3.25 | 27 |
| b2-L | L | llama | 17.76 | 51.8 | 397.9 | 3.25 | 4 |
| b3-8f32-S | S | 8f32 | 4.938 | 226.9 | 1529 | 3.37 | 23 |
| b3-8f32-B | B | 8f32 | 16.671 | 245.7 | 1655.1 | 3.65 | 24 |
| b3-16f32-S | S | 16f32 | 4.47 | 477.6 | 3230.9 | 3.07 | 7 |
| b3-16f32-B | B | 16f32 | 17.887 | 467.1 | 3152.7 | 2.81 | 16 |
| b3-8q8-S | S | 8q8 | 4.545 | 245.6 | 1650.8 | 3.05 | 0 |
| b3-8q8-B | B | 8q8 | 17.39 | 228.2 | 1536.2 | 3.06 | 15 |
| b3-16q8-S | S | 16q8 | 4.55 | 468.6 | 3162.6 | 3.08 | 0 |
| b3-16q8-B | B | 16q8 | 17.513 | 471.3 | 3187.4 | 3.57 | 13 |
| b3-L | L | llama | 17.48 | 53.7 | 399.7 | 3.58 | 15 |

Any block whose standby cache summed under the 2.33 GB model was reading the
model from DISK and is not comparable; all three buckets are summed (the
Normal-only guard cried wolf on 3/3 good blocks on 2026-08-18). If the raw log
carries an ISLC warning, read the standby column as unavailable rather than low.
