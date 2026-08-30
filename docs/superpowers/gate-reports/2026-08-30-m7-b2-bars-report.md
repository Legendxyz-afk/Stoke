# M5 gate item [2] -- the bars, measured

Run `2026-08-30-m7-b2-bars`. S2 = 512 prompt tokens (525 real, ChatML) x 128 generated,
M0's own prompts (`tests/data/bench_prompts/`), ctx 704, threads 8, unpinned,
3 interleaved blocks. Raw: `2026-08-30-m7-b2-bars-raw.txt`, bank: `2026-08-30-m7-b2-bars-bank.jsonl`.

**The gate number is AGGREGATE(M0)**: decode-only ids over the span from the first
token anywhere to the last token anywhere -- `bench/bench.py`'s `aggregate_decode_tps`,
which is how the 17.7 / 14.2 / 10.8 / 16.8 cells were computed.

**Thermal state.** llama `pp512` by block: 78.34 / 74.45 / 75.66 -- these are taken at each
block's END, so they date the plateau and say nothing about block 1's own cells.
Stoke's own repeats: block 1 runs at **1.00x** the median of the later blocks
-- within 8%, so all blocks are at one clock and cross-block numbers compare.

## Medians by cell

| cell | S serve AGG(M0) | S decode-only (N/ITL p50) | S TTFT p50 s | B batch decode | B prefill tok/s | bar | verdict |
|---|---|---|---|---|---|---|---|
| 8f32 | 7.264 | 26.75 | 72.4 | - | - | &gt;= 17.7 | MISS (0.41x) |

## The same numbers by block (read column 1 against the rest)

| cell | arm | block 1 | block 2 | block 3 |
|---|---|---|---|---|
| 8f32 | S | 7.282 | 7.264 | 7.251 |
| 8f32 | B | - | - | - |

## The non-stoke referent (llama-bench b10037, t 8)

- tg128 (decode, N=1): **17.65** tok/s -- the box-health anchor
- pp512 (prefill): **75.66** tok/s -- the prefill referent

## Every unit

| id | arm | cell | gate tok/s | wall s | cpu s | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| b1-8f32-S | S | 8f32 | 7.282 | 155.2 | 1008.3 | 8.59 | 12 |
| b1-L | L | llama | 17.65 | 51 | 381.6 | 8.09 | 3 |
| b2-8f32-S | S | 8f32 | 7.264 | 153.7 | 1008.6 | 8.11 | 3 |
| b2-L | L | llama | 17.3 | 51.2 | 394.3 | 8.14 | 11 |
| b3-8f32-S | S | 8f32 | 7.251 | 154.2 | 1012.2 | 8.17 | 3 |
| b3-L | L | llama | 17.82 | 50.1 | 385.6 | 7.85 | 3 |

Any block whose standby cache summed under the 2.33 GB model was reading the
model from DISK and is not comparable; all three buckets are summed (the
Normal-only guard cried wolf on 3/3 good blocks on 2026-08-18). If the raw log
carries an ISLC warning, read the standby column as unavailable rather than low.
