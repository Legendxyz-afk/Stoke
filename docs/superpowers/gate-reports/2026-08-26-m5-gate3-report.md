# M5 gate item [3] -- the decode stall during co-running prefill

Run `2026-08-26-m5-gate3`. S3, 8 clients x 300 generated, ctx 1600, threads 8, kv f32.
Bound **250 ms**. Graded on the `ITL stall` line per the 2026-08-26 ruling
(`memos/2026-08-26-m5-gate3-criterion-RULING-OPUS.md`); predictions pre-registered
at `23fcb35` BEFORE this run. Raw `2026-08-26-m5-gate3-raw.txt`.

| cell | dedup | chunk | stall p50 ms | stall max ms | stall n | stall % of gaps | cadence p50 ms | prefill chunks |
|---|---|---|---|---|---|---|---|---|
| Sd | on | 128 | 4,843.3 | 5,126 | 56 | 2.3% | 401.9 | 24 |
| Sn | OFF | 128 | 4,383.2 | 5,251 | 280 | 11.7% | 430.0 | 80 |
| C16 | on | 16 | 1,108.2 | 1,272 | 392 | 16.4% | 495.2 | 176 |
| Sn2 | OFF | 128 | 5,463.7 | 6,665 | 280 | 11.7% | 528.6 | 80 |
| Sd2 | on | 128 | 5,835.4 | 6,561 | 56 | 2.3% | 515.7 | 24 |

## The pre-registered predictions, graded

| # | prediction | band | measured | verdict |
|---|---|---|---|---|
| P1 | stall p50 at chunk 128 ~ 4.5 s | 3500-5500 ms | 4,843 | **HELD** |
| P2 | stall p50 same in both arms | within 20% | 5,339 vs 4,923 = 7.8% apart | **HELD** |
| P3 | stall share ~8% dedup / ~27% no-dedup | +-5 pp | 2.3% / 11.7% | **FALSIFIED** |
| P4 | cadence p50 ~403 ms, same in both arms | +-15% | 459 vs 479 | **HELD** |
| P5 | stall p50 at chunk 16 ~ 570 ms | 400-800 ms | 1,108 | **FALSIFIED** |

## What the bound costs, per chunk width

| chunk | stall p50 ms | implied prefill tok/s | tok/s needed for 250 ms | shortfall |
|---|---|---|---|---|
| 128 | 5,339 | **24.0** | 512.0 | **21.4x** |
| 16 | 1,108 | **14.4** | 64.0 | **4.4x** |

Read the two rows against each other: a narrower chunk shortens the stall but
does NOT hold the prefill rate, so the shortfall does not fall with the width.

**Drift across the sitting, from the repeated cells: stall +20% (Sd) and +25% (Sn), cadence +28%.**
Monotonic and in the direction the box's thermal plateau predicts, which is why the
cell order is palindromic: each compared arm has one early and one late sample.
**P2 is gradeable only because of that** -- pairing Sd's early cell against Sn's late
one reads 11% apart, and Sd's late against Sn's early reads 25%, on either side of
P2's 20% band. Every verdict below clears its bound by more than 4x, so the drift
cannot flip any of them.

Discriminator: every cell's stall population is a proper subset of its gaps -- the conditioning works.

## VERDICT (gate [3], on the stall line): **MISS** -- the smallest stall p50 measured is 1,108 ms = 4.4x the 250 ms bound

Chunk 16 is the SMALLEST LEGAL width -- `BP_BLOCK_TOKENS` is 16 and M4 froze it --
so the chunk-width lever is exhausted at the C16 cell, not merely untried.

**Both falsified predictions failed in the same direction and for the same kind of
reason -- a per-stream or per-chunk quantity assumed to scale when it does not:**

- **P3.** Stalls are not `chunks x n_gen`. A chunk stalls only the streams ALREADY
  DECODING, so the count is the sum over chunks of the streams live at that moment:
  with dedup 2 chunks/client x (1+..+7) = **56**, without 10 x 28 = **280** -- both
  measured EXACTLY. Prefills run one at a time (S1), so client 0 stalls nobody.
- **P5.** A 16-token chunk does not cost 1/8th of a 128-token chunk. The implied
  prefill rate FALLS with the width (see the table above), so the stall shortens
  sub-linearly and the shortfall against the bound gets WORSE, not better.

## Every cell, verbatim

```
--- Sd (dedup on, chunk 128)
serve: 8 seqs x 300 gen over 8 prompt(s), ctx 1600, kv f32, 800 pool blocks, dedup on
  AGGREGATE(M0) 13.491 tok/s  = 2392 decode ids / 177302.8 ms span (first token to last)
  end-to-end 10.931 tok/s  = 2400 ids / 219563.1 ms wall (all prefill charged)
  TTFT p50 75482  p95 101330  max 101330 ms
  ITL  p50 402.8  p95 428.1  max 5125.9 ms  (n=2392)
  ITL  stall   p50 4843.3  p95 5125.9  max 5125.9 ms  (n=56, 2.3% of gaps)  <- gate [3]
  ITL  cadence p50 401.9  p95 423.4  max 462.5 ms  (n=2336)
  completion p50 218750  p95 219563  max 219563 ms
  steps 323 | rows 5128 | prefill chunks 24 | preemptions 0 (0 prefill yields)
  dedup: 7168 of 9904 admitted prompt positions adopted
```

```
--- Sn (dedup OFF, chunk 128)
serve: 8 seqs x 300 gen over 8 prompt(s), ctx 1600, kv f32, 800 pool blocks, dedup OFF
  AGGREGATE(M0) 5.700 tok/s  = 2392 decode ids / 419619.9 ms span (first token to last)
  end-to-end 5.184 tok/s  = 2400 ids / 462919.6 ms wall (all prefill charged)
  TTFT p50 213269  p95 346682  max 346682 ms
  ITL  p50 431.9  p95 4495.2  max 5251.4 ms  (n=2392)
  ITL  stall   p50 4383.2  p95 5133.2  max 5251.4 ms  (n=280, 11.7% of gaps)  <- gate [3]
  ITL  cadence p50 430.0  p95 467.3  max 481.9 ms  (n=2112)
  completion p50 458779  p95 462920  max 462920 ms
  steps 379 | rows 12296 | prefill chunks 80 | preemptions 0 (0 prefill yields)
  dedup: DISABLED by --no-dedup (9904 admitted prompt positions, none adopted)
```

```
--- C16 (dedup on, chunk 16)
serve: 8 seqs x 300 gen over 8 prompt(s), ctx 1600, kv f32, 800 pool blocks, dedup on
  AGGREGATE(M0) 10.653 tok/s  = 2392 decode ids / 224543.0 ms span (first token to last)
  end-to-end 8.961 tok/s  = 2400 ids / 267817.0 ms wall (all prefill charged)
  TTFT p50 90713  p95 137117  max 137117 ms
  ITL  p50 499.6  p95 1138.2  max 1272.0 ms  (n=2392)
  ITL  stall   p50 1108.2  p95 1190.8  max 1272.0 ms  (n=392, 16.4% of gaps)  <- gate [3]
  ITL  cadence p50 495.2  p95 549.7  max 614.8 ms  (n=2000)
  completion p50 260396  p95 267817  max 267817 ms
  steps 475 | rows 5128 | prefill chunks 176 | preemptions 0 (0 prefill yields)
  dedup: 7168 of 9904 admitted prompt positions adopted
```

```
--- Sn2 (dedup OFF, chunk 128)
serve: 8 seqs x 300 gen over 8 prompt(s), ctx 1600, kv f32, 800 pool blocks, dedup OFF
  AGGREGATE(M0) 4.601 tok/s  = 2392 decode ids / 519940.9 ms span (first token to last)
  end-to-end 4.190 tok/s  = 2400 ids / 572819.5 ms wall (all prefill charged)
  TTFT p50 263919  p95 428950  max 428950 ms
  ITL  p50 532.8  p95 5576.3  max 6664.7 ms  (n=2392)
  ITL  stall   p50 5463.7  p95 6338.2  max 6664.7 ms  (n=280, 11.7% of gaps)  <- gate [3]
  ITL  cadence p50 528.6  p95 575.3  max 655.1 ms  (n=2112)
  completion p50 567545  p95 572820  max 572820 ms
  steps 379 | rows 12296 | prefill chunks 80 | preemptions 0 (0 prefill yields)
  dedup: DISABLED by --no-dedup (9904 admitted prompt positions, none adopted)
```

```
--- Sd2 (dedup on, chunk 128)
serve: 8 seqs x 300 gen over 8 prompt(s), ctx 1600, kv f32, 800 pool blocks, dedup on
  AGGREGATE(M0) 10.548 tok/s  = 2392 decode ids / 226772.3 ms span (first token to last)
  end-to-end 8.615 tok/s  = 2400 ids / 278590.1 ms wall (all prefill charged)
  TTFT p50 93660  p95 126964  max 126964 ms
  ITL  p50 516.8  p95 571.0  max 6560.5 ms  (n=2392)
  ITL  stall   p50 5835.4  p95 6560.5  max 6560.5 ms  (n=56, 2.3% of gaps)  <- gate [3]
  ITL  cadence p50 515.7  p95 562.9  max 653.6 ms  (n=2336)
  completion p50 277523  p95 278590  max 278590 ms
  steps 323 | rows 5128 | prefill chunks 24 | preemptions 0 (0 prefill yields)
  dedup: 7168 of 9904 admitted prompt positions adopted
```

