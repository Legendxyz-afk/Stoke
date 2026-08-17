# M4.5 L4 — attention was the last single-threaded work in the forward pass

**Date:** 2026-08-17 · **Branch:** `m45-aprime` · **Status:** written before the
speed measurement, deliberately — the predictions below are on the record so
the measurement can contradict them.

---

## 1. The finding

`pool_for_rows` is called from exactly one file:

```
src/kernels/vec.c:267  pool_for_rows(n_out, rows_f32_batch, &j);
src/kernels/vec.c:294  pool_for_rows(n_out, rows_delegate_batch, &j);
src/kernels/vec.c:331  pool_for_rows(n_out, is_q4 ? rows_q4_k_batch_avx2 ...
src/kernels/vec.c:334  pool_for_rows(n_out, rows_delegate_batch, &j);
src/kernels/vec.c:376  pool_for_rows(n_out, rows_f32, &j);
src/kernels/vec.c:425  pool_for_rows(n_out, ...
src/kernels/vec.c:470  pool_for_rows(n_out, rows, &j);
```

All seven are matvec. **The per-head attention loop in `model.c` ran on the
calling thread alone**, from K5 (which landed the pool) to today.

Put that next to the 2026-08-17 context profile, which measured attention at
**24.2% of decode at ctx 87, 76.5% at ctx 554 and 84.6% at ctx 1500**: at long
context roughly **six sevenths of a token's work was using one core out of
eight**, while the part that was already fast got all of them.

This was not in the 2026-08-12 ruling's field of view. §7.6 frames L4 entirely
as a SIMD problem — "`vec_dot_f32` is a SEQUENTIAL f32 chain … redefine the f32
scalar to an 8-lane partitioned canonical fold, twin mirrors it exactly" — and
prices it as a graded arithmetic change with a full old-vs-new run behind it.
The ruling is right that you cannot reassociate a reduction for free. It simply
did not ask whether the loop was running on one core.

## 2. Why it is bit-exact, and therefore free

The same argument `pool.h` makes for matvec, one level up:

> a q-head's output depends on that head's Q row, on the kv-head it reads, and
> on nothing else; it is written to its own slice of `s_attn_out`; and the
> arithmetic inside a head — the order of every add in every dot product, the
> softmax, the V accumulation — is character-for-character what it was when the
> loop was serial.

The parallelism is over **independent outputs**, not inside a reduction. Nothing
is reassociated, so there is no twin, no grading run, no re-anchor and no new
referent. CONTRACT v2 is not engaged at all.

The one thing that was not already per-head is the **scratch**: the scores row
and the KV dequantization buffer were one per SEQUENCE, which is exactly one too
few the moment two heads of one sequence run at once. They are now one per
WORKER (`Model::att_ws`, indexed by `pool_worker_index()`), which is the
tightest correct unit — a head runs on exactly one worker start to finish, and
the count is threads (8) rather than heads (32).

## 3. What had to change in the pool

`pool_for_rows` could not express this dispatch. `POOL_CHUNK_ROWS` is 64 and the
floor is two chunks, so **any parallel-for over fewer than 128 units silently
ran inline on the caller**. Matvec never noticed — its smallest output is 1024
rows. Attention has 32 units.

- `pool_for_units(n, chunk, fn, ctx)` — the chunk is the caller's choice;
  `pool_for_rows` is now this at chunk 64.
- `pool_worker_index()` — a `_Thread_local` set once per thread, 0 for the
  dispatching thread. Per-worker scratch needs it; matvec never did.

A parallel-for whose refusal to parallelise is invisible is a defect in its own
right, and `tests/test_pool.c` now covers the small-n, small-chunk regime that
`pool_for_rows` could never reach.

## 4. Prediction, on the record before the measurement

Model: the QK loop is latency-bound on a chain of FP adds; the V loop is
throughput-bound and already vectorised by the compiler (SSE2 — `model.c` is
built at the baseline ISA); `kv_read`'s address arithmetic is called twice per
(head, position).

Golden/Raptor Cove has **two FP-add ports at 2–3 cycle latency** (the fast-adder
bypass applies to back-to-back adds, which is precisely this loop), so
`ATT_LANES = 4` is latency × ports and saturates them. That is the same number
the lane loop already used, but for a different reason than its comment gave
("~4 cycles at ~1/cycle") — the arithmetic agreed by accident.

At ctx 1500, per token, taking attention's measured 268 ms:

| term | now | + lanes | + pool (8T) |
|---|---|---|---|
| QK   | ~250 ms | ~60–70 ms | ~8–9 ms |
| V    | ~18 ms  | ~18 ms    | ~2.3 ms |
| `kv_read` overhead | ~23 ms | ~23 ms | ~3 ms |
| **attention** | **268 ms** | **~100 ms** | **~14 ms** |
| decode total | 317 ms | ~150 ms | **~63 ms** |

against llama's 75 ms at the same depth. At ctx 87 the same model gives ~50
ms/token → ~20 tok/s against llama's 18.5.

**These are estimates and the gate is the measurement.** This project has twice
recorded a products-per-instruction argument overstating a real gain (rung A′
predicted a win and measured 0.907×; Q6_K predicted 1.20× "as a floor" and
measured 1.108×). The number that counts is what
`tests/tooling/run_att_par_ab.ps1` reports, four arms interleaved in one
sitting.

## 5. What this does NOT fix

- **Prefill is still position-by-position.** `model_generate` calls
  `model_forward_position` per prompt token, so prefill attention is O(n²) with
  a serial outer loop. llama batches prefill (ubatch 512) and turns it into a
  matmul, which is why its `pp` is an order faster than its `tg`. M5 work, not
  P-BAR work.
- **The 4× GQA read redundancy.** `kv_read` is called once per q-head, and the
  group is 4, so each KV slot is fetched four times (and twice more in the V
  loop). At ctx 1500 an L2-resident kv-head absorbs most of that, but the
  *call* overhead is real: ~3.5M `kv_vec_offset` calls per token, each doing
  bounds checks, a `bp_seq_ensure`, a divide and a multiply chain. The paged
  layout means a block-walking iterator could compute the base once per 16
  positions instead of once per position. Bit-exact, and the obvious next step.
- **Dispatch shape at N>1.** The dispatch is per (sequence, layer), so N=16
  costs 576 dispatches per token instead of 36. Dispatching over (s,h) pairs in
  one call per layer would fix it, and needs the KV write phase hoisted out of
  the sequence loop — safe (a write never lands in a shared prefix block, which
  is only shared once full) but a separate change with its own argument. M5
  entry.

## 6. Consequence for the record

The 2026-08-17 attribution — "the entire remaining llama gap is attention over
the KV cache, and matvec is already at parity (intercepts 48.76 vs 52.51 ms)" —
**stands, and is sharpened**: the gap is attention *running on one core*, not
attention being arithmetically slow. The intercept finding is untouched.

The M5 alarm is the part that changes most. "Extrapolated to ctx 1024 stoke pays
~187 ms/token vs llama ~61 = 0.33×, so M5's aggregate bars are unreachable on
this slope" was measured against a single-threaded attention loop. If the
prediction above survives contact, the slope falls by roughly an order and the
M5 bars come back into range — which would make this the most load-bearing
change in M4.5.
