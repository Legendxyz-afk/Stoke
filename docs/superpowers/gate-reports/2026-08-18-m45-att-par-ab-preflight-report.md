# M4.5 L4 -- attention on the pool, and the interleaved QK loop

1 cycles, point-order and arm-order both rotating, 8 threads, unpinned, n=8, `--kv f32`. All four stoke arms come from ONE binary via `STOKE_ATT_PAR` / `STOKE_ATT_LANES`; all four are byte-identical in output, so this is speed only. llama at matched depth (`-d`).

| point | avg ctx | par0/lanes1 | par0/lanes4 | par1/lanes1 | par1/lanes4 | best/base | llama | best/llama |
|---|---|---|---|---|---|---|---|---|
| p00 | 87 | 16.35 | 16.47 | 14.11 | 15.62 | **0.955x** | 17.81 | **0.877x** |
| p40 | 554 | 8.324 | 8.921 | 11.872 | 11.755 | **1.412x** | 15.25 | **0.771x** |

## Per-token cost, which is where the attention term is visible

| point | avg ctx | base ms/tok | best ms/tok | saved ms/tok | llama ms/tok |
|---|---|---|---|---|---|
| p00 | 87 | 61.16 | 64.02 | -2.86 | 56.15 |
| p40 | 554 | 120.13 | 85.07 | 35.06 | 65.57 |

## The slope, which is the M5-critical number

ms/token per CONTEXT-token, fitted between the extreme points. Before L4 stoke's slope was 0.1791 against llama's 0.0150 (2026-08-17, db85e0f) -- 11.9x worse -- and extrapolating it to ctx 1024 was what made M5's aggregate bars look unreachable. That extrapolation was measured against a SINGLE-THREADED attention loop, so this row is what replaces it.

| arm | ms/tok @ ctx 87 | ms/tok @ ctx 554 | slope ms/context-token |
|---|---|---|---|
| par0lanes1 | 61.16 | 120.13 | **0.1263** |
| par0lanes4 | 60.72 | 112.1 | **0.11** |
| par1lanes1 | 70.87 | 84.23 | **0.0286** |
| par1lanes4 | 64.02 | 85.07 | **0.0451** |
| llama | 56.15 | 65.57 | **0.0202** |

Two points fit a line and nothing more. The 2026-08-17 profile showed this cost is CONVEX -- the middle point sat 14% below the chord, consistent with the per-layer KV crossing out of L3 -- so a slope read off the extremes OVERSTATES the middle and UNDERSTATES the far end. Use it to compare arms in the same sitting, not as a law.

## Wrong-cores check (CPU-seconds per unit)

| point | arm | min | median | max |
|---|---|---|---|---|
| p00 | par0lanes1 | 0 | 0 | 0 |
| p00 | par0lanes4 | 13.2 | 13.2 | 13.2 |
| p00 | par1lanes1 | 0 | 0 | 0 |
| p00 | par1lanes4 | 0 | 0 | 0 |
| p40 | par0lanes1 | 183.4 | 183.4 | 183.4 |
| p40 | par0lanes4 | 192.3 | 192.3 | 192.3 |
| p40 | par1lanes1 | 207.3 | 207.3 | 207.3 |
| p40 | par1lanes4 | 214 | 214 | 214 |

A unit whose CPU-seconds sit far above its arm's median was doing the same work somewhere slower -- efficiency cores, thermal throttling, contention -- and should be discarded rather than averaged in. EcoQoS is fixed in the engine, but this check exists to catch the NEXT thing that quietly moves work.

Raw log: `2026-08-18-m45-att-par-ab-preflight-raw.txt`.
