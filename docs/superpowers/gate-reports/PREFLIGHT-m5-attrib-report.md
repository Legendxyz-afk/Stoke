# Attributing the 2.83x: is it the kernel or the pool?

Run `PREFLIGHT-m5-attrib`. PREFILL of a 525-token prompt, ctx 704, kv f32, 1 cycles with
the thread order reversed on even cycles. stoke's number is `--timing`'s `prefill`
term (load excluded); llama's is `llama-bench -p 512 -n 0` (warmed). GFLOP/s uses
8.46 GFLOP/token from `m5_compute_budget.py`.

| threads | stoke tok/s | stoke GFLOP/s | llama tok/s | llama GFLOP/s | gap | stoke scaling | llama scaling |
|---|---|---|---|---|---|---|---|
| 8 | 23.621 | 199.8 | 69.640 | 589.2 | 2.95x | 1.00x | 1.00x |

**The discriminator, pre-registered:** a single-thread gap near 2.8x means the gap is
INSTRUCTION EFFICIENCY and the kernel is the target; a single-thread gap under 1.8x
means the pool is, and the bit-exactness question is largely moot.

| unit | arm | t | tok/s | GFLOP/s | wall s | cpu s | cores | clock % |
|---|---|---|---|---|---|---|---|---|
| c1-t8-S | S | 8 | 23.621 | 199.8 | 26 | 135.5 | 5.21 | 137.4 |
| c1-t8-L | L | 8 | 69.64 | 589.2 | 27.6 | 174.8 | 6.33 | 114.9 |
