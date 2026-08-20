# M5 gate item [2] -- the bars, measured

Run `PREFLIGHT-m5-bars`. S2 = 512 prompt tokens (525 real, ChatML) x 8 generated,
M0's own prompts (`tests/data/bench_prompts/`), ctx 544, threads 8, unpinned,
1 interleaved blocks on a quiesced box. Raw: `PREFLIGHT-m5-bars-raw.txt`, bank: `PREFLIGHT-m5-bars-bank.jsonl`.

**The gate number is AGGREGATE(M0)**: decode-only ids over the span from the first
token anywhere to the last token anywhere -- ench/bench.py's `aggregate_decode_tps`,
which is how the 17.7 / 14.2 / 10.8 / 16.8 cells were computed.

## Medians by cell

| cell | S serve AGG(M0) | S decode-only (N/ITL p50) | S TTFT p50 s | B batch decode | B prefill tok/s | bar | verdict |
|---|---|---|---|---|---|---|---|
| 2f32 | 0.839 | 28.82 | 34.8 | 18.218 | 30.93 | >=17.7 | MISS (0.05x) |

## The non-stoke referent (llama-bench b10037, t 8)

- tg8 (decode, N=1): **18.12** tok/s -- the box-health anchor
- pp512 (prefill): **91.3** tok/s -- the prefill referent

## Every unit

| id | arm | cell | gate tok/s | wall s | cpu s | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| b1-2f32-S | S | 2f32 | 0.839 | 37.9 | 0 | 0 | 23 |
| b1-2f32-B | B | 2f32 | 18.218 | 38.1 | 0 | 0 | 17 |
| b1-L | L | llama | 18.12 | 36.7 | 0 | 0 | 18 |

Any block whose standby cache summed under the 2.33 GB model was reading the
model from DISK and is not comparable; all three buckets are summed (the
Normal-only guard cried wolf on 3/3 good blocks on 2026-08-18).
