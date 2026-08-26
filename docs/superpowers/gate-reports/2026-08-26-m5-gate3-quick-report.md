# M5 gate item [3] -- the decode stall during co-running prefill

Run `2026-08-26-m5-gate3-quick`. S3, 2 clients x 24 generated, ctx 1600, threads 8, kv f32.
Bound **250 ms**. Graded on the `ITL stall` line per the 2026-08-26 ruling
(`memos/2026-08-26-m5-gate3-criterion-RULING-OPUS.md`); predictions pre-registered
at `23fcb35` BEFORE this run. Raw `2026-08-26-m5-gate3-quick-raw.txt`.

| cell | dedup | chunk | stall p50 ms | stall max ms | stall n | stall % of gaps | cadence p50 ms | prefill chunks |
|---|---|---|---|---|---|---|---|---|
| Sd | on | 128 | 4,837.1 | 4,837 | 2 | 4.3% | 126.5 | 12 |
| Sn | OFF | 128 | 4,084.6 | 4,770 | 10 | 21.7% | 130.0 | 20 |
| C16 | on | 16 | 669.5 | 685 | 14 | 30.4% | 130.0 | 92 |
| Sn2 | OFF | 128 | 4,146.3 | 4,854 | 10 | 21.7% | 130.1 | 20 |
| Sd2 | on | 128 | 4,846.2 | 4,846 | 2 | 4.3% | 127.2 | 12 |

## The pre-registered predictions, graded

| # | prediction | band | measured | verdict |
|---|---|---|---|---|
| P1 | stall p50 at chunk 128 ~ 4.5 s | 3500-5500 ms | 4,837 | **HELD** |
| P2 | stall p50 same in both arms | within 20% | 4,842 vs 4,115 = 15.0% apart | **HELD** |
| P3 | stall share ~8% dedup / ~27% no-dedup | +-5 pp | 4.3% / 21.7% | **FALSIFIED** |
| P4 | cadence p50 ~403 ms, same in both arms | +-15% | 127 vs 130 | **FALSIFIED** |
| P5 | stall p50 at chunk 16 ~ 570 ms | 400-800 ms | 670 | **HELD** |

Discriminator: every cell's stall population is a proper subset of its gaps -- the conditioning works.

## VERDICT (gate [3], on the stall line): **MISS** -- the smallest stall p50 measured is 670 ms = 2.7x the 250 ms bound

Chunk 16 is the SMALLEST LEGAL width -- `BP_BLOCK_TOKENS` is 16 and M4 froze it --
so the chunk-width lever is exhausted at the C16 cell, not merely untried.

## Every cell, verbatim

```
--- Sd (dedup on, chunk 128)
serve: 2 seqs x 24 gen over 2 prompt(s), ctx 1600, kv f32, 200 pool blocks, dedup on
  AGGREGATE(M0) 4.170 tok/s  = 46 decode ids / 11030.7 ms span (first token to last)
  end-to-end 0.900 tok/s  = 48 ids / 53328.2 ms wall (all prefill charged)
  TTFT p50 50515  p95 50515  max 50515 ms
  ITL  p50 126.5  p95 128.4  max 4837.1 ms  (n=46)
  ITL  stall   p50 4837.1  p95 4837.1  max 4837.1 ms  (n=2, 4.3% of gaps)  <- gate [3]
  ITL  cadence p50 126.5  p95 127.7  max 128.4 ms  (n=44)
  completion p50 53328  p95 53328  max 53328 ms
  steps 35 | rows 1498 | prefill chunks 12 | preemptions 0 (0 prefill yields)
  dedup: 1024 of 2476 admitted prompt positions adopted
```

```
--- Sn (dedup OFF, chunk 128)
serve: 2 seqs x 24 gen over 2 prompt(s), ctx 1600, kv f32, 200 pool blocks, dedup OFF
  AGGREGATE(M0) 1.071 tok/s  = 46 decode ids / 42961.5 ms span (first token to last)
  end-to-end 0.563 tok/s  = 48 ids / 85221.9 ms wall (all prefill charged)
  TTFT p50 82757  p95 82757  max 82757 ms
  ITL  p50 130.4  p95 4424.9  max 4770.3 ms  (n=46)
  ITL  stall   p50 4084.6  p95 4770.3  max 4770.3 ms  (n=10, 21.7% of gaps)  <- gate [3]
  ITL  cadence p50 130.0  p95 130.7  max 130.7 ms  (n=36)
  completion p50 85222  p95 85222  max 85222 ms
  steps 43 | rows 2522 | prefill chunks 20 | preemptions 0 (0 prefill yields)
  dedup: DISABLED by --no-dedup (2476 admitted prompt positions, none adopted)
```

```
--- C16 (dedup on, chunk 16)
serve: 2 seqs x 24 gen over 2 prompt(s), ctx 1600, kv f32, 200 pool blocks, dedup on
  AGGREGATE(M0) 4.087 tok/s  = 46 decode ids / 11255.4 ms span (first token to last)
  end-to-end 0.881 tok/s  = 48 ids / 54468.7 ms wall (all prefill charged)
  TTFT p50 52210  p95 52210  max 52210 ms
  ITL  p50 131.3  p95 681.6  max 685.2 ms  (n=46)
  ITL  stall   p50 669.5  p95 681.6  max 685.2 ms  (n=14, 30.4% of gaps)  <- gate [3]
  ITL  cadence p50 130.0  p95 133.5  max 135.0 ms  (n=32)
  completion p50 54469  p95 54469  max 54469 ms
  steps 115 | rows 1498 | prefill chunks 92 | preemptions 0 (0 prefill yields)
  dedup: 1024 of 2476 admitted prompt positions adopted
```

```
--- Sn2 (dedup OFF, chunk 128)
serve: 2 seqs x 24 gen over 2 prompt(s), ctx 1600, kv f32, 200 pool blocks, dedup OFF
  AGGREGATE(M0) 1.053 tok/s  = 46 decode ids / 43696.1 ms span (first token to last)
  end-to-end 0.557 tok/s  = 48 ids / 86200.0 ms wall (all prefill charged)
  TTFT p50 83729  p95 83729  max 83729 ms
  ITL  p50 130.4  p95 4573.8  max 4853.7 ms  (n=46)
  ITL  stall   p50 4146.3  p95 4853.7  max 4853.7 ms  (n=10, 21.7% of gaps)  <- gate [3]
  ITL  cadence p50 130.1  p95 131.8  max 132.4 ms  (n=36)
  completion p50 86200  p95 86200  max 86200 ms
  steps 43 | rows 2522 | prefill chunks 20 | preemptions 0 (0 prefill yields)
  dedup: DISABLED by --no-dedup (2476 admitted prompt positions, none adopted)
```

```
--- Sd2 (dedup on, chunk 128)
serve: 2 seqs x 24 gen over 2 prompt(s), ctx 1600, kv f32, 200 pool blocks, dedup on
  AGGREGATE(M0) 4.171 tok/s  = 46 decode ids / 11028.8 ms span (first token to last)
  end-to-end 0.892 tok/s  = 48 ids / 53794.5 ms wall (all prefill charged)
  TTFT p50 50965  p95 50965  max 50965 ms
  ITL  p50 127.2  p95 128.7  max 4846.2 ms  (n=46)
  ITL  stall   p50 4846.2  p95 4846.2  max 4846.2 ms  (n=2, 4.3% of gaps)  <- gate [3]
  ITL  cadence p50 127.2  p95 128.7  max 128.7 ms  (n=44)
  completion p50 53794  p95 53794  max 53794 ms
  steps 35 | rows 1498 | prefill chunks 12 | preemptions 0 (0 prefill yields)
  dedup: 1024 of 2476 admitted prompt positions adopted
```

