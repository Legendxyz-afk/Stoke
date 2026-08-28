# M5 gate item [3] -- the decode stall during co-running prefill

Run `2026-08-28-m5-gate3`. S3, 8 clients x 300 generated, ctx 1600, threads 8, kv f32.
Binary `stoke.exe` sha256 `B1E99988B3C2`; the per-cell resume cache is keyed by it.
Bound **250 ms**. Graded on the `ITL stall` line per the 2026-08-26 ruling
(`memos/2026-08-26-m5-gate3-criterion-RULING-OPUS.md`); predictions pre-registered
at `23fcb35` BEFORE this run. Raw `2026-08-28-m5-gate3-raw.txt`.

| cell | dedup | chunk | stall p50 ms | stall max ms | stall n | stall % of gaps | cadence p50 ms | prefill chunks |
|---|---|---|---|---|---|---|---|---|
| Sd | on | 128 | 3,869.5 | 4,048 | 56 | 2.3% | 310.2 | 24 |
| Sn | OFF | 128 | 3,324.5 | 4,160 | 280 | 11.7% | 334.0 | 80 |
| C16 | on | 16 | 677.7 | 757 | 392 | 16.4% | 308.0 | 176 |
| Sn2 | OFF | 128 | 3,386.2 | 4,197 | 280 | 11.7% | 336.4 | 80 |
| Sd2 | on | 128 | 3,953.5 | 4,216 | 56 | 2.3% | 312.9 | 24 |

## The pre-registered predictions, graded

âš  **These were registered at `23fcb35`. If the binary under test is not the one they
were registered against (this run: `d482807`, exe `B1E99988B3C2`), a verdict below is a
RE-GRADE, not a fresh test**: a band calibrated on a different binary can be entered or
exited because the SYSTEM moved. Compare against the run that registered them before
reading any flip as the prediction being right or wrong.

| # | prediction | band | measured | verdict |
|---|---|---|---|---|
| P1 | stall p50 at chunk 128 ~ 4.5 s | 3500-5500 ms | 3,870 | **HELD** |
| P2 | stall p50 same in both arms | within 20% | 3,912 vs 3,355 = 14.2% apart | **HELD** |
| P3 | stall share ~8% dedup / ~27% no-dedup | +-5 pp | 2.3% / 11.7% | **FALSIFIED** |
| P4 | cadence p50 ~403 ms, same in both arms | +-15% | 312 vs 335 | **FALSIFIED** |
| P5 | stall p50 at chunk 16 ~ 570 ms | 400-800 ms | 678 | **HELD** |

## What the bound costs, per chunk width

| chunk | stall p50 ms | implied prefill tok/s | tok/s needed for 250 ms | shortfall |
|---|---|---|---|---|
| 128 | 3,912 | **32.7** | 512.0 | **15.6x** |
| 16 | 678 | **23.6** | 64.0 | **2.7x** |

Read the two rows against each other: a narrower chunk shortens the stall but
does NOT hold the prefill rate, so the shortfall does not fall with the width.

**Drift across the sitting, from the repeated cells: stall +2% (Sd) and +2% (Sn), cadence +1%.**
Monotonic and in the direction the box's thermal plateau predicts, which is why the
cell order is palindromic: each compared arm has one early and one late sample.
**P2 is gradeable only because of that** -- pairing Sd's early cell against Sn's late
one reads 11% apart, and Sd's late against Sn's early reads 25%, on either side of
P2's 20% band. Every verdict below clears its bound by more than 4x, so the drift
cannot flip any of them.

Discriminator: every cell's stall population is a proper subset of its gaps -- the conditioning works.

## VERDICT (gate [3], on the stall line): **MISS** -- the smallest stall p50 measured is 678 ms = 2.7x the 250 ms bound

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
  AGGREGATE(M0) 17.298 tok/s  = 2392 decode ids / 138284.9 ms span (first token to last)
  end-to-end 14.162 tok/s  = 2400 ids / 169467.5 ms wall (all prefill charged)
  TTFT p50 57846  p95 78459  max 78459 ms
  ITL  p50 310.4  p95 329.7  max 4047.8 ms  (n=2392)
  ITL  stall   p50 3869.5  p95 4047.8  max 4047.8 ms  (n=56, 2.3% of gaps)  <- gate [3]
  ITL  cadence p50 310.2  p95 328.7  max 349.9 ms  (n=2336)
  completion p50 168773  p95 169467  max 169467 ms
  steps 323 | rows 5128 | prefill chunks 24 | preemptions 0 (0 prefill yields)
  pool: peak 321 of 800 blocks used (40.1%), 800 free at end
  dedup: 7168 of 9904 admitted prompt positions adopted
```

```
--- Sn (dedup OFF, chunk 128)
serve: 8 seqs x 300 gen over 8 prompt(s), ctx 1600, kv f32, 800 pool blocks, dedup OFF
  AGGREGATE(M0) 7.399 tok/s  = 2392 decode ids / 323299.3 ms span (first token to last)
  end-to-end 6.770 tok/s  = 2400 ids / 354524.0 ms wall (all prefill charged)
  TTFT p50 161556  p95 263725  max 263725 ms
  ITL  p50 336.9  p95 3433.3  max 4160.2 ms  (n=2392)
  ITL  stall   p50 3324.5  p95 4047.5  max 4160.2 ms  (n=280, 11.7% of gaps)  <- gate [3]
  ITL  cadence p50 334.0  p95 348.2  max 358.0 ms  (n=2112)
  completion p50 350936  p95 354524  max 354524 ms
  steps 379 | rows 12296 | prefill chunks 80 | preemptions 0 (0 prefill yields)
  pool: peak 755 of 800 blocks used (94.4%), 800 free at end
  dedup: DISABLED by --no-dedup (9904 admitted prompt positions, none adopted)
```

```
--- C16 (dedup on, chunk 16)
serve: 8 seqs x 300 gen over 8 prompt(s), ctx 1600, kv f32, 800 pool blocks, dedup on
  AGGREGATE(M0) 16.890 tok/s  = 2392 decode ids / 141622.9 ms span (first token to last)
  end-to-end 13.865 tok/s  = 2400 ids / 173100.6 ms wall (all prefill charged)
  TTFT p50 63614  p95 92462  max 92462 ms
  ITL  p50 310.4  p95 714.6  max 757.1 ms  (n=2392)
  ITL  stall   p50 677.7  p95 752.6  max 757.1 ms  (n=392, 16.4% of gaps)  <- gate [3]
  ITL  cadence p50 308.0  p95 324.0  max 341.2 ms  (n=2000)
  completion p50 168262  p95 173101  max 173101 ms
  steps 475 | rows 5128 | prefill chunks 176 | preemptions 0 (0 prefill yields)
  pool: peak 300 of 800 blocks used (37.5%), 800 free at end
  dedup: 7168 of 9904 admitted prompt positions adopted
```

```
--- Sn2 (dedup OFF, chunk 128)
serve: 8 seqs x 300 gen over 8 prompt(s), ctx 1600, kv f32, 800 pool blocks, dedup OFF
  AGGREGATE(M0) 7.320 tok/s  = 2392 decode ids / 326773.5 ms span (first token to last)
  end-to-end 6.703 tok/s  = 2400 ids / 358068.2 ms wall (all prefill charged)
  TTFT p50 162976  p95 266576  max 266576 ms
  ITL  p50 338.9  p95 3483.5  max 4196.8 ms  (n=2392)
  ITL  stall   p50 3386.2  p95 4132.2  max 4196.8 ms  (n=280, 11.7% of gaps)  <- gate [3]
  ITL  cadence p50 336.4  p95 351.8  max 373.3 ms  (n=2112)
  completion p50 354453  p95 358068  max 358068 ms
  steps 379 | rows 12296 | prefill chunks 80 | preemptions 0 (0 prefill yields)
  pool: peak 755 of 800 blocks used (94.4%), 800 free at end
  dedup: DISABLED by --no-dedup (9904 admitted prompt positions, none adopted)
```

```
--- Sd2 (dedup on, chunk 128)
serve: 8 seqs x 300 gen over 8 prompt(s), ctx 1600, kv f32, 800 pool blocks, dedup on
  AGGREGATE(M0) 17.027 tok/s  = 2392 decode ids / 140481.1 ms span (first token to last)
  end-to-end 13.928 tok/s  = 2400 ids / 172309.1 ms wall (all prefill charged)
  TTFT p50 59065  p95 80451  max 80451 ms
  ITL  p50 313.0  p95 333.5  max 4215.5 ms  (n=2392)
  ITL  stall   p50 3953.5  p95 4215.5  max 4215.5 ms  (n=56, 2.3% of gaps)  <- gate [3]
  ITL  cadence p50 312.9  p95 331.7  max 351.8 ms  (n=2336)
  completion p50 171610  p95 172309  max 172309 ms
  steps 323 | rows 5128 | prefill chunks 24 | preemptions 0 (0 prefill yields)
  pool: peak 321 of 800 blocks used (40.1%), 800 free at end
  dedup: 7168 of 9904 admitted prompt positions adopted
```

