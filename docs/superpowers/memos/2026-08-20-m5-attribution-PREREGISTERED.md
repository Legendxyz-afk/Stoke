# Attributing the 2.83× — prediction written before the first measurement

**2026-08-20, Opus 5. Committed before the runner produced a number.**

Gate [2]'s memo asks Kabir one question — keep bit-exactness or buy the
arithmetic — and then argues (§5b) that the question is premature, because the
2.83× prefill gap to llama.cpp is **measured and unattributed**. This is the
first attribution step.

## The split, and why it needs no new kernel code

stoke prefills at **273 GFLOP/s** and llama.cpp at **772** (same sitting, same
file, t=8). That end-to-end number contains at least three things:

1. the Q4_K dot kernel's own instruction efficiency,
2. the thread pool's parallelisation loss (barriers, dispatch, idle tails),
3. everything in the forward that is not `matvec_batch`.

**At t=1 there is no pool, no barrier and no dispatch.** So the single-thread
rate is (1) almost neat, and the ratio between t=1 and t=8 prices (2). Both
engines take `-t`/`--threads`, so both can be put through the identical split
with the shipping binaries and no instrumentation.

`stoke batch --timing` reports `prefill N tok in M ms` **separately from
`load`**, so page-cache effects — ISLC is running and purging standby — sit
outside the parsed number. llama-bench warms up before timing, likewise.

## The discriminator, stated in advance

| outcome | reading | what to work on |
|---|---|---|
| **single-thread gap ≈ 2.8×** | the gap is INSTRUCTION EFFICIENCY; threading is a bystander | the kernel — and §5b's integer-core argument applies in full |
| **single-thread gap ≤ 1.5×** | most of the gap is PARALLELISATION | the pool, and the bit-exactness question is largely moot |
| **stoke scales much worse than llama** | (2) dominates regardless of the ST gap | the pool |

## Predicted numbers

stoke's measured core occupancy at t=8 is 6.70 of 8, so if the pool were
perfectly efficient over that many cores the single-thread rate would be
≈ 273/6.70 = **41 GFLOP/s**; llama's would be ≈ 772/6.70 = **115**. Both are
upper bounds on the single-thread rate under a perfect-scaling assumption, and
this box is 6 P-cores + 4 E-cores, so at t=8 some threads land on cores worth
~40% of a P-core — **neither engine will scale 8×, and they face that equally.**

**I predict the single-thread gap lands near 2.8×**, i.e. threading is NOT the
story, because the flatness curve already showed thread occupancy contributing
only 1.155× of the 1.43× batching gain — a pool that is inefficient enough to
explain a 2.8× gap should have shown up there.

**Falsifier:** a single-thread gap under 1.8× means I have been pointing at the
wrong subsystem for a full session, and §5b's kernel argument is aimed at
something that is not the bottleneck.

**Second falsifier, on the method:** if stoke's t=1 prefill rate exceeds
273 GFLOP/s — i.e. one thread beats eight — the measurement is wrong, not the
engine, and nothing below should be read.
