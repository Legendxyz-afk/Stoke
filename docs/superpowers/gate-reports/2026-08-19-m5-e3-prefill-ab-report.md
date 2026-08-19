# M5 E3 -- chunked prefill A/B

3 cycles, point-order and arm-order both rotating, 6 threads. One binary; `STOKE_PREFILL_CHUNK=1` is the pre-M5 position-by-position path. Both widths verified to write a bit-identical KV cache (`m5_e3_chunked_prefill_...` and the entry byte-compare), so this is speed only. Affinity **PINNED 0xFFF**.

| prompt | prefill tok | chunk 1 | chunk 128 | speedup | tok/s at 128 |
|---|---|---|---|---|---|
| p00 | 23 | 1621.5 ms | 990.1 ms | **1.64x** | 23.2 |
| p40 | 490 | 30240.3 ms | 18716 ms | **1.62x** | 26.2 |
| long | 1436 | 104081.4 ms | 62326.4 ms | **1.67x** | 23.0 |

## Against the pre-registered prediction

The weight term goes `P x 57.3 ms` to `ceil(P/128) x 57.3 ms`; attention and arithmetic are untouched, so the speedup is BOUNDED by what fraction of prefill was weight streaming. A figure anywhere near 128x means the measurement is wrong, not that the rung is brilliant.

**The falsifier:** prefill tok/s must RISE with prompt length under chunking and be roughly flat without it. p00 (23 tokens) spans a single chunk and must show almost nothing -- if it shows a large gain, the win is not coming from weight amortisation.

Raw log: `2026-08-19-m5-e3-prefill-ab-raw.txt`.
