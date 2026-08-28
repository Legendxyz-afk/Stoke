# M5 gate item [2] -- the bars, measured

Run `2026-08-28-m5-bars-q6k`. S2 = 512 prompt tokens (525 real, ChatML) x 128 generated,
M0's own prompts (`tests/data/bench_prompts/`), ctx 704, threads 8, unpinned,
3 interleaved blocks. Raw: `2026-08-28-m5-bars-q6k-raw.txt`, bank: `2026-08-28-m5-bars-q6k-bank.jsonl`.

**The gate number is AGGREGATE(M0)**: decode-only ids over the span from the first
token anywhere to the last token anywhere -- `bench/bench.py`'s `aggregate_decode_tps`,
which is how the 17.7 / 14.2 / 10.8 / 16.8 cells were computed.

**Thermal state.** llama `pp512` by block: 91.40 / 93.31 / 92.29 -- these are taken at each
block's END, so they date the plateau and say nothing about block 1's own cells.
Stoke's own repeats: block 1 runs at **1.00x** the median of the later blocks
-- within 8%, so all blocks are at one clock and cross-block numbers compare.

## Medians by cell

| cell | S serve AGG(M0) | S decode-only (N/ITL p50) | S TTFT p50 s | B batch decode | B prefill tok/s | bar | verdict |
|---|---|---|---|---|---|---|---|
| 8f32 | 8.960 | 32.52 | 59.7 | 32.572 | 45.60 | &gt;= 17.7 | MISS (0.51x) |
| 16f32 | 8.501 | 34.06 | 111.4 | 34.344 | 44.60 | &gt;= 16.8 | MISS (0.51x) |

## The same numbers by block (read column 1 against the rest)

| cell | arm | block 1 | block 2 | block 3 |
|---|---|---|---|---|
| 8f32 | S | 8.960 | 9.029 | 8.810 |
| 8f32 | B | 32.396 | 32.713 | 32.572 |
| 16f32 | S | 8.691 | 8.501 | 8.485 |
| 16f32 | B | 34.344 | 34.445 | 34.115 |

## The non-stoke referent (llama-bench b10037, t 8)

- tg128 (decode, N=1): **18.77** tok/s -- the box-health anchor
- pp512 (prefill): **92.29** tok/s -- the prefill referent

## Every unit

| id | arm | cell | gate tok/s | wall s | cpu s | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| b1-8f32-S | S | 8f32 | 8.96 | 126 | 827.7 | 4.82 | 12 |
| b1-8f32-B | B | 8f32 | 32.396 | 124.8 | 816.5 | 4.87 | 14 |
| b1-16f32-S | S | 16f32 | 8.691 | 246.3 | 1636.8 | 4.85 | 11 |
| b1-16f32-B | B | 16f32 | 34.344 | 249.8 | 1659.1 | 4.88 | 20 |
| b1-L | L | llama | 18.73 | 45.1 | 346.5 | 4.87 | 3 |
| b2-8f32-S | S | 8f32 | 9.029 | 125.3 | 834.4 | 4.89 | 10 |
| b2-8f32-B | B | 8f32 | 32.713 | 124.1 | 820.8 | 4.9 | 9 |
| b2-16f32-S | S | 16f32 | 8.501 | 251.8 | 1660.4 | 5.83 | 11 |
| b2-16f32-B | B | 16f32 | 34.445 | 248.3 | 1659.6 | 5.89 | 14 |
| b2-L | L | llama | 18.78 | 44.5 | 342.2 | 6.28 | 25 |
| b3-8f32-S | S | 8f32 | 8.81 | 128.4 | 852.6 | 6.33 | 10 |
| b3-8f32-B | B | 8f32 | 32.572 | 123.7 | 822.9 | 6.34 | 17 |
| b3-16f32-S | S | 16f32 | 8.485 | 252.3 | 1684.4 | 5.85 | 12 |
| b3-16f32-B | B | 16f32 | 34.115 | 248.4 | 1654.4 | 5.87 | 7 |
| b3-L | L | llama | 18.77 | 44.8 | 344.6 | 5.88 | 5 |

Any block whose standby cache summed under the 2.33 GB model was reading the
model from DISK and is not comparable; all three buckets are summed (the
Normal-only guard cried wolf on 3/3 good blocks on 2026-08-18). If the raw log
carries an ISLC warning, read the standby column as unavailable rather than low.

**THE QUIESCE GATE DID NOT PASS on 4 of 13 invocation(s).**
Units measured without it: `b1-8f32-B`, `b2-16f32-S`, `b3-8f32-S`, `b3-8f32-B`, `b3-16f32-S` -- their absolute rates are INDICATIVE.
