# M4.5 L4 -- attention on the pool, and the interleaved QK loop

3 cycles, point-order and arm-order both rotating, 8 threads, unpinned, n=128, `--kv f32`. All four stoke arms come from ONE binary via `STOKE_ATT_PAR` / `STOKE_ATT_LANES`; all four are byte-identical in output, so this is speed only. llama at matched depth (`-d`).

| point | avg ctx | par0/lanes1 | par0/lanes4 | par1/lanes1 | par1/lanes4 | best/base | llama | best/llama |
|---|---|---|---|---|---|---|---|---|
| p00 | 87 | 13.896 | 14.371 | 15.088 | 15.17 | **1.092x** | 18.32 | **0.828x** |
| p40 | 554 | 8.028 | 8.954 | 12.536 | 12.683 | **1.580x** | 16.28 | **0.779x** |
| long | 1500 | 3.185 | 4.012 | 9.251 | 9.368 | **2.941x** | 12.77 | **0.734x** |

## Per-token cost, which is where the attention term is visible

| point | avg ctx | base ms/tok | best ms/tok | saved ms/tok | llama ms/tok |
|---|---|---|---|---|---|
| p00 | 87 | 71.96 | 65.92 | 6.04 | 54.59 |
| p40 | 554 | 124.56 | 78.85 | 45.71 | 61.43 |
| long | 1500 | 313.97 | 106.75 | 207.22 | 78.31 |

## The slope, which is the M5-critical number

ms/token per CONTEXT-token, fitted between the extreme points. Before L4 stoke's slope was 0.1791 against llama's 0.0150 (2026-08-17, db85e0f) -- 11.9x worse -- and extrapolating it to ctx 1024 was what made M5's aggregate bars look unreachable. That extrapolation was measured against a SINGLE-THREADED attention loop, so this row is what replaces it.

| arm | ms/tok @ ctx 87 | ms/tok @ ctx 1500 | slope ms/context-token |
|---|---|---|---|
| par0lanes1 | 71.96 | 313.97 | **0.1713** |
| par0lanes4 | 69.58 | 249.25 | **0.1272** |
| par1lanes1 | 66.28 | 108.1 | **0.0296** |
| par1lanes4 | 65.92 | 106.75 | **0.0289** |
| llama | 54.59 | 78.31 | **0.0168** |

Two points fit a line and nothing more. The 2026-08-17 profile showed this cost is CONVEX -- the middle point sat 14% below the chord, consistent with the per-layer KV crossing out of L3 -- so a slope read off the extremes OVERSTATES the middle and UNDERSTATES the far end. Use it to compare arms in the same sitting, not as a law.

## Wrong-cores check (CPU-seconds per unit)

| point | arm | min | median | max |
|---|---|---|---|---|
| p00 | par0lanes1 | 60.5 | 61.4 | 61.5 |
| p00 | par0lanes4 | 61.3 | 61.4 | 61.4 |
| p00 | par1lanes1 | 0 | 63 | 66.4 |
| p00 | par1lanes4 | 62.4 | 62.7 | 64.5 |
| p40 | par0lanes1 | 228.5 | 228.7 | 239.5 |
| p40 | par0lanes4 | 0 | 228.8 | 230.9 |
| p40 | par1lanes1 | 0 | 259.6 | 260.2 |
| p40 | par1lanes4 | 256.2 | 256.7 | 260.3 |
| long | par0lanes1 | 678.9 | 682.8 | 714.6 |
| long | par0lanes4 | 637.3 | 648.8 | 653.3 |
| long | par1lanes1 | 0 | 0 | 0 |
| long | par1lanes4 | 0 | 797.2 | 797.6 |

A unit whose CPU-seconds sit far above its arm's median was doing the same work somewhere slower -- efficiency cores, thermal throttling, contention -- and should be discarded rather than averaged in. EcoQoS is fixed in the engine, but this check exists to catch the NEXT thing that quietly moves work.

Raw log: `2026-08-18-m45-att-par-ab-raw.txt`.
