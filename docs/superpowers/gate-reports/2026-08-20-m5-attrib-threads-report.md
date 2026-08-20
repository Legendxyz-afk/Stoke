# Attributing the 2.83x: is it the kernel or the pool?

Run `2026-08-20-m5-attrib-threads`. PREFILL of a 525-token prompt, ctx 704, kv f32, 2 cycles with
the thread order reversed on even cycles. stoke's number is `--timing`'s `prefill`
term (load excluded); llama's is `llama-bench -p 512 -n 0` (warmed). GFLOP/s uses
8.46 GFLOP/token from `m5_compute_budget.py`.

| threads | stoke tok/s | stoke GFLOP/s | llama tok/s | llama GFLOP/s | gap | stoke scaling | llama scaling |
|---|---|---|---|---|---|---|---|
| 1 | 5.472 | 46.3 | 17.620 | 149.1 | 3.22x | 1.00x | 1.00x |
| 2 | 9.836 | 83.2 | 34.730 | 293.8 | 3.53x | 1.80x | 1.97x |
| 4 | 17.979 | 152.1 | 63.255 | 535.1 | 3.52x | 3.29x | 3.59x |
| 8 | 24.822 | 210.0 | 75.835 | 641.6 | 3.06x | 4.54x | 4.30x |

**The discriminator, pre-registered:** a single-thread gap near 2.8x means the gap is
INSTRUCTION EFFICIENCY and the kernel is the target; a single-thread gap under 1.8x
means the pool is, and the bit-exactness question is largely moot.

**The SCALING columns are the instrument for parallel efficiency, not `cores`.**
`cpu s` is whole-process -- it includes the model load and the short decode tail --
while the rate is prefill-only, so `cores` is a health tell (is anything running at
all?) and NOT prefill occupancy. Reading it as occupancy would price the pool from a
number that contains two phases the rate does not.

| unit | arm | t | tok/s | GFLOP/s | wall s | cpu s | cores (whole proc) | clock % |
|---|---|---|---|---|---|---|---|---|
| c1-t1-S | S | 1 | 5.184 | 43.9 | 107.2 | 98 | 0.91 | 117.6 |
| c1-t1-L | L | 1 | 17.46 | 147.7 | 95.4 | 90.3 | 0.95 | 122.4 |
| c1-t2-S | S | 2 | 9.86 | 83.4 | 57.1 | 93.6 | 1.64 | 176.5 |
| c1-t2-L | L | 2 | 35 | 296.1 | 52 | 91.7 | 1.76 | 96 |
| c1-t4-S | S | 4 | 17.925 | 151.6 | 30.9 | 98 | 3.17 | 180.1 |
| c1-t4-L | L | 4 | 65.22 | 551.8 | 31.1 | 97.8 | 3.14 | 99.7 |
| c1-t8-S | S | 8 | 24.706 | 209 | 22.6 | 131.3 | 5.81 | 151.9 |
| c1-t8-L | L | 8 | 75.84 | 641.6 | 26.1 | 160.6 | 6.15 | 94.6 |
| c2-t8-S | S | 8 | 24.939 | 211 | 22.3 | 130 | 5.83 | 157.5 |
| c2-t8-L | L | 8 | 75.83 | 641.5 | 26.8 | 161 | 6.01 | 95.2 |
| c2-t4-S | S | 4 | 18.033 | 152.6 | 30.6 | 97.5 | 3.19 | 179 |
| c2-t4-L | L | 4 | 61.29 | 518.5 | 31.6 | 102.5 | 3.24 | 96.1 |
| c2-t2-S | S | 2 | 9.811 | 83 | 55.4 | 94 | 1.70 | 130.9 |
| c2-t2-L | L | 2 | 34.46 | 291.5 | 51.9 | 92.1 | 1.77 | 104.9 |
| c2-t1-S | S | 1 | 5.76 | 48.7 | 96.6 | 92.1 | 0.95 | 157.6 |
| c2-t1-L | L | 1 | 17.78 | 150.4 | 93.2 | 89.1 | 0.96 | 104.2 |
