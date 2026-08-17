# M4.5 -- attention share at ctx 1024-2048, the ruling's precondition for L4

3 cycles, point-order and engine-order both rotating, 8 threads, unpinned, n=128 generated per measurement. stoke reaches depth via a long prompt; llama via `-d/--n-depth` set to stoke's exact prompt-token count, so both generate 128 tokens over the same context window.

| point | prompt tok | avg ctx | stoke tok/s | stoke ms/tok | llama tok/s | llama ms/tok | stoke/llama |
|---|---|---|---|---|---|---|---|
| p00 | 23 | 87 | 15.542 | 64.34 | 18.58 | 53.82 | 0.836x |
| p40 | 490 | 554 | 7.713 | 129.65 | 16.48 | 60.68 | 0.468x |
| long | 1436 | 1500 | 3.151 | 317.36 | 13.32 | 75.08 | 0.237x |

## The fit: ms/token = intercept + slope * avg_ctx

- **stoke**: slope **0.1791** ms per context-token, intercept **48.76** ms
- **llama**: slope **0.0150** ms per context-token, intercept **52.51** ms
- stoke's attention cost per context-token is **11.9x** llama's
- context-independent work differs by **-3.75 ms/token** (-7.1%), which is the matvec/weight-streaming side

Linearity check at the middle point (avg ctx 554): predicted 147.96 ms/token, measured 129.65. A prediction that misses badly means the cost is not linear in context and the slope is not a single mechanism.

## THE RULING'S GATE: attention share of decode

| avg ctx | stoke ms/tok | attention ms/tok | **attention share** | >= 15%? |
|---|---|---|---|---|
| 87 | 64.34 | 15.58 | **24.2%** | YES |
| 554 | 129.65 | 99.20 | **76.5%** | YES |
| 1500 | 317.36 | 268.60 | **84.6%** | YES |

The ruling builds L4 **only** if attention is >= 15% of decode at ctx 1024-2048. The `long` row sits at avg ctx 1500, inside that range, and is the row that decides.

Raw log: `2026-08-17-m45-attention-share-raw.txt`.
