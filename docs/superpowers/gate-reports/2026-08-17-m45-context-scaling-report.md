# M4.5 -- per-token cost vs generation length, one interleaved sitting

3 cycles, n-order and engine-order both rotating, 8 threads, unpinned. stoke = `batch --seqs 1 --kv f32`; llama = `llama-bench -p 0 -r 3`. Median per cell.

| n | stoke tok/s | stoke ms/tok | llama tok/s | llama ms/tok | stoke/llama |
|---|---|---|---|---|---|
| 64 | 16.626 | 60.15 | 18.51 | 54.02 | 0.898x |
| 128 | 15.557 | 64.28 | 18.48 | 54.11 | 0.842x |
| 256 | 13.752 | 72.72 | 18.27 | 54.73 | 0.753x |

## The slope, which is the whole question

| from -> to | stoke d(ms/tok) | llama d(ms/tok) | stoke excess |
|---|---|---|---|
| 64 -> 128 | 4.13 | 0.09 | 4.04 |
| 128 -> 256 | 8.44 | 0.62 | 7.82 |

**Reading it.** If stoke's ms/token climbs while llama's stays flat, the remaining P-BAR gap is stoke's cost per unit of CONTEXT -- attention over the KV cache -- and the next lever is sec 7.5 step (2)/L4, not rung D. If both climb together, the lead measured across sittings was an artefact and attribution returns to per-thread matvec work.

Every n here was measured in ONE sitting, interleaved. The lead that prompted this run came from two different sittings, which is the comparison this box has repeatedly punished.

Raw log: `2026-08-17-m45-context-scaling-raw.txt`.
