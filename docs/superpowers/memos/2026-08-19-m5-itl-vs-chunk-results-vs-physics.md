# M5 gate item [3]: the 250 ms ITL bound and S-ALIGN's 16-token block are in tension

*2026-08-19, Opus 5. A **results-vs-physics memo**, which is the form the
forward note prescribes when a bar is not met and the attribution ladder has
been walked rather than ground against. **INDICATIVE, NOT GATE-GRADE**: the
numbers below come from an unquiesced box in a single sitting. The ratios and
the structural conclusion are robust; the absolutes are not, and none of them
should be quoted. A quiesced confirmation is the first thing this memo asks
for.*

---

## 1. What was measured

`stoke bench --serve` with **staggered arrivals** — request *i* submitted after
*i*×3 steps — so a 490-token prompt lands while decodes are in flight. Without
staggering every prefill completes before the decode set matures, no prefill
ever co-runs with one, and **gate item [3] cannot be measured at all**; that is
why the driver grew `--stagger` before this memo could be written.

4 sequences, 2 prompts (p00 at 23 tokens, p40 at 490), `-n 10`, 6 threads, f32
KV. `STOKE_SCHED_CHUNK` sweeps the prefill chunk from one binary in one sitting.

| chunk | aggregate tok/s | ITL p50 | ITL p95 | **ITL max** |
|---|---|---|---|---|
| **128** (locked) | 1.111 | 159 ms | 4367 ms | **4463 ms** |
| 32 | 1.125 | 1062 ms | 1128 ms | **1209 ms** |
| 16 (S-ALIGN floor) | 1.124 | 542 ms | 603 ms | **624 ms** |

## 2. The three things this says

**(a) The stall IS the chunk's own step.** ITL max falls almost exactly linearly
with chunk width — 8:2:1 in chunk gives 7.2:1.9:1 in stall. A decode co-running
with a prefill chunk waits for that whole step, because a step is atomic: the
fused forward carries all rows or none. That is the design working as specified,
not a defect. **Gate item [3] is therefore a statement about how long one step
may take, and only incidentally about prefill.**

**(b) Chunking does not remove the decodes' wait; it reshapes its
distribution.** The median moves the *other* way as the chunk shrinks (159 →
542 ms) while the max collapses (4463 → 624 ms). Total prefill work is
conserved, so a smaller chunk spreads the same delay over more, shorter stalls.
Anyone reading only p50 would conclude chunk 128 is the better setting for
latency; anyone reading only the max would conclude the opposite. **Both are
true, and the bar is written against the max.**

**(c) Aggregate throughput is unchanged across all three.** 1.111 / 1.125 /
1.124 — inside the noise of an unquiesced box. In this workload the chunk buys
no throughput at all, which is the surprise: chunked prefill's whole argument is
weight amortisation, and E3 measured that amortisation **saturating by ~23
rows** (report `2026-08-19-m5-e3-prefill-ab-report.md`). Above ~23 rows the
chunk is buying nothing and costing ITL linearly.

## 3. The tension, stated as arithmetic

Extrapolating (a) to the bound: 624 ms at chunk 16 ⇒ **250 ms needs chunk ≈ 6**.

**But a chunk must be a whole number of KV blocks (S-ALIGN), and a block is 16
tokens.** So the smallest legal chunk is 16, and it still exceeds the bound by
**2.5×**.

The forward note's own attribution ladder for this failure has been walked:

| ladder step | outcome |
|---|---|
| "drop chunk to 64" | tried, and further — 32 and 16 too |
| "if still, the fused step isn't actually sharing weight streams — verify the prefill rows ride the same op walk" | **they do.** E3 runs the chunk through `forward_batch_body`, the same op walk as the decodes, verified byte-identical by the entry gate |

So the ladder is exhausted and this is physics, not a bug: a step containing 16
prefill rows plus the live decodes costs ~600 ms on this hardware, and no
scheduling choice makes it shorter.

## 4. What could actually move it — options, not a recommendation

1. **Shrink `BP_BLOCK_TOKENS`** from 16. S-ALIGN's granularity is the block, so
   an 8- or 4-token block permits a legal chunk near the 250 ms figure. Cost:
   more blocks per sequence, more table entries, more pool bookkeeping — and it
   touches M4's closed invariant set, so it is not a small change.
2. **Make the step cheaper.** ~600 ms for 16 prefill rows + 3 decodes is
   dominated by compute, and the attention dot products are still **scalar f32
   by design** (kept that way for bit-exactness). The residual slope after E2 is
   0.0224 ms/context-token and SIMD there is already priced and waiting on
   Kabir. This is the option that helps every other bar too.
3. **Re-anchor the bound.** 250 ms is a human-interactivity figure. M5's own
   operating point is Fable's AGENTIC trace — agent swarms, not humans — where
   p95 *request completion* is the number that matters and is already recorded.
   **This is Kabir's call, not mine**: the bar is his headline.

## 5. What I am NOT claiming

- Not that the bar is wrong. It may simply need option 1 or 2 first.
- Not that these numbers are gate-grade. **Unquiesced box, one sitting, a
  4-sequence prefill-dominated workload whose 1.1 tok/s aggregate is nothing
  like the shape gate item [2] measures.** The next step is a quiesced,
  pinned, interleaved re-run at the bars' own shape before any of this is
  treated as measured.
- Not that chunk 16 should ship. The default stays the locked 128 until there is
  a quiesced measurement and a decision; `STOKE_SCHED_CHUNK` exists to measure,
  not to change what ships.

## 6. Direction check for Kabir

Recorded, not asked — nothing is blocked on an answer, since the default is
unchanged and the work continues at gate item [2].

**The one thing I would want him to know before [2] is measured:** if [3] ends
up needing option 2 (a cheaper step), that is the same SIMD-in-attention
decision already outstanding — and it would then be blocking two gate items
rather than being a throughput nicety.
