# M5 gate item [2] -- the bars, measured

Run `overnight-f8b06a2`. S2 = 512 prompt tokens (525 real, ChatML) x 128 generated,
M0's own prompts (`tests/data/bench_prompts/`), ctx 704, threads 8, unpinned,
2 interleaved blocks. Raw: `overnight-f8b06a2-raw.txt`, bank: `overnight-f8b06a2-bank.jsonl`.

**The gate number is AGGREGATE(M0)**: decode-only ids over the span from the first
token anywhere to the last token anywhere -- `bench/bench.py`'s `aggregate_decode_tps`,
which is how the 17.7 / 14.2 / 10.8 / 16.8 cells were computed.

**Thermal state.** llama `pp512` by block: 78.17 / 76.52 -- these are taken at each
block's END, so they date the plateau and say nothing about block 1's own cells.
Stoke's own repeats: block 1 runs at **1.01x** the median of the later blocks
-- within 8%, so all blocks are at one clock and cross-block numbers compare.

## Medians by cell

| cell | S serve AGG(M0) | S decode-only (N/ITL p50) | S TTFT p50 s | B batch decode | B prefill tok/s | bar | verdict |
|---|---|---|---|---|---|---|---|
| 8f32 | 14.518 | 41.32 | 31.9 | - | - | &gt;= 17.7 | MISS (0.82x) |
| 16f32 | 14.746 | 49.44 | 59.4 | - | - | &gt;= 16.8 | MISS (0.88x) |

## The same numbers by block (read column 1 against the rest)

| cell | arm | block 1 | block 2 |
|---|---|---|---|
| 8f32 | S | 14.496 | 14.539 |
| 8f32 | B | - | - |
| 16f32 | S | 14.872 | 14.619 |
| 16f32 | B | - | - |

## The non-stoke referent (llama-bench b10037, t 8)

- tg128 (decode, N=1): **18.25** tok/s -- the box-health anchor
- pp512 (prefill): **77.345** tok/s -- the prefill referent

## Every unit

| id | arm | cell | gate tok/s | wall s | cpu s | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| b1-8f32-S | S | 8f32 | 14.496 | 78 | 487 | 5.29 | 1 |
| b1-16f32-S | S | 16f32 | 14.872 | 143.2 | 922.1 | 5.3 | 2 |
| b1-L | L | llama | 18.37 | 48.9 | 373 | 5.32 | 1 |
| b2-8f32-S | S | 8f32 | 14.539 | 76.3 | 484.3 | 5.33 | 1 |
| b2-16f32-S | S | 16f32 | 14.619 | 145.7 | 944.9 | 5.63 | 4 |
| b2-L | L | llama | 18.13 | 49.2 | 380.1 | 5.63 | 4 |

Any block whose standby cache summed under the 2.33 GB model was reading the
model from DISK and is not comparable; all three buckets are summed (the
Normal-only guard cried wolf on 3/3 good blocks on 2026-08-18). If the raw log
carries an ISLC warning, read the standby column as unavailable rather than low.
