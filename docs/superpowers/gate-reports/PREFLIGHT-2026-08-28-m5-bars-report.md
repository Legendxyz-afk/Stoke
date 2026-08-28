# M5 gate item [2] -- the bars, measured

Run `PREFLIGHT-2026-08-28-m5-bars`. S2 = 512 prompt tokens (525 real, ChatML) x 128 generated,
M0's own prompts (`tests/data/bench_prompts/`), ctx 704, threads 8, unpinned,
1 interleaved blocks. Raw: `PREFLIGHT-2026-08-28-m5-bars-raw.txt`, bank: `PREFLIGHT-2026-08-28-m5-bars-bank.jsonl`.

**The gate number is AGGREGATE(M0)**: decode-only ids over the span from the first
token anywhere to the last token anywhere -- `bench/bench.py`'s `aggregate_decode_tps`,
which is how the 17.7 / 14.2 / 10.8 / 16.8 cells were computed.

## Medians by cell

| cell | S serve AGG(M0) | S decode-only (N/ITL p50) | S TTFT p50 s | B batch decode | B prefill tok/s | bar | verdict |
|---|---|---|---|---|---|---|---|
| 8f32 | - | - | - | - | - | &gt;= 17.7 | - |

## The same numbers by block (read column 1 against the rest)

| cell | arm | block 1 |
|---|---|---|
| 8f32 | S | - |
| 8f32 | B | - |

## The non-stoke referent (llama-bench b10037, t 8)

- tg128 (decode, N=1): **16.89** tok/s -- the box-health anchor
- pp512 (prefill): **92.29** tok/s -- the prefill referent

## Every unit

| id | arm | cell | gate tok/s | wall s | cpu s | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| b1-L | L | llama | 16.89 | 51.8 | 361.8 | 4.25 | 22 |

Any block whose standby cache summed under the 2.33 GB model was reading the
model from DISK and is not comparable; all three buckets are summed (the
Normal-only guard cried wolf on 3/3 good blocks on 2026-08-18). If the raw log
carries an ISLC warning, read the standby column as unavailable rather than low.

**THE QUIESCE GATE DID NOT PASS on 1 of 2 invocation(s).**
No banked unit carries `quiesced: false`, so the failures fell on invocations that measured nothing.
