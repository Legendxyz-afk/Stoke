# M5 gate item [2] -- the bars, measured

Run `2026-08-29-m5-bars-presampler`. S2 = 512 prompt tokens (525 real, ChatML) x 128 generated,
M0's own prompts (`tests/data/bench_prompts/`), ctx 704, threads 8, unpinned,
1 interleaved blocks. Raw: `2026-08-29-m5-bars-presampler-raw.txt`, bank: `2026-08-29-m5-bars-presampler-bank.jsonl`.

**The gate number is AGGREGATE(M0)**: decode-only ids over the span from the first
token anywhere to the last token anywhere -- `bench/bench.py`'s `aggregate_decode_tps`,
which is how the 17.7 / 14.2 / 10.8 / 16.8 cells were computed.

## Medians by cell

| cell | S serve AGG(M0) | S decode-only (N/ITL p50) | S TTFT p50 s | B batch decode | B prefill tok/s | bar | verdict |
|---|---|---|---|---|---|---|---|
| 8f32 | 7.228 | 26.64 | 72.8 | - | - | &gt;= 17.7 | MISS (0.41x) |

## The same numbers by block (read column 1 against the rest)

| cell | arm | block 1 |
|---|---|---|
| 8f32 | S | 7.228 |
| 8f32 | B | - |

## The non-stoke referent (llama-bench b10037, t 8)

- tg128 (decode, N=1): **17.61** tok/s -- the box-health anchor
- pp512 (prefill): **78.46** tok/s -- the prefill referent

## Every unit

| id | arm | cell | gate tok/s | wall s | cpu s | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| b1-8f32-S | S | 8f32 | 7.228 | 154.6 | 1007.6 | 6 | 13 |
| b1-L | L | llama | 17.61 | 51.1 | 381.9 | 5.86 | 17 |

Any block whose standby cache summed under the 2.33 GB model was reading the
model from DISK and is not comparable; all three buckets are summed (the
Normal-only guard cried wolf on 3/3 good blocks on 2026-08-18). If the raw log
carries an ISLC warning, read the standby column as unavailable rather than low.
