# M4 §5 — static batch, and the test that must exist before the optimization

Design only. No code yet, deliberately: the change restructures the matvec
core, and §5.4 explains why its *test* should land first.

## 5.1 What "static batch" means here

N sequences step together, one token each per step. Static because the set is
fixed for the run — admission and eviction mid-flight is M5's continuous
batching, not this.

The pool already supports it: sequences share a `KvStore`, each holds its own
`Kv` handle, and `kv_view` resolves per handle. Nothing in M4's storage layer
needs to change. **What changes is the compute.**

## 5.2 seqs-inner / weights-outer, and why the order is the whole point

The ROADMAP fixes the structure: for each weight tile, loop over sequences.

    for each layer:
      for each weight W:
        load W                       <- once, amortized across the batch
        for each sequence s in batch:
          y[s] += W · x[s]           <- s is the INNER loop

The naive order (sequence-outer) re-reads every weight per sequence and
amortizes nothing — it is N independent forwards with extra bookkeeping. §1.4
makes the stakes concrete: weights are 2.4973 GB per step and the KV term is
`N × L × 294,912 B`, so amortizing the weight term is the entire reason
batching wins until the KV crossover at `N × L = 8,467`.

**And the order is also what makes batch-composition independence free.**
Each `y[s] += W · x[s]` is an independent dot product over the same bytes in
the same order, whichever other sequences share the batch. Nothing about
sequence *s*'s arithmetic depends on N or on who else is present. That is why
ROADMAP §M4 can demand EXACT (bit-level) independence and get it by
construction rather than by tolerance — the same argument K5's thread pool
already rests on (`tests/test_pool.c`: rows are independent, so spreading them
over threads reorders nothing).

## 5.3 What could break it, stated so the test knows what to look for

- **A shared accumulator or scratch buffer across sequences.** `Model` holds
  `s_k`, `s_v`, activation scratch — all currently sized for ONE sequence.
  Batching needs per-sequence scratch, and the failure mode of getting it
  wrong is one sequence's activations bleeding into another's.
- **A reduction whose order depends on N.** Any "sum across the batch" step
  would make sequence *s*'s result a function of the batch, which is exactly
  what the bar forbids.
- **Sequence-count-dependent tiling.** Choosing a different kernel path for
  N=2 vs N=9 reintroduces the device-GEMM problem (§8.1) on the CPU side,
  where we currently do not have it.

## 5.4 The identity test lands BEFORE the kernels change — and why that is not a vacuous test

**The bar:** the same sequence, same prompt, must produce a BYTE-IDENTICAL
token stream whether it runs alone or in a batch of N, for every N in the
tested range.

Written today, against a batch that steps sequences sequentially, this test
**passes trivially** — independent forwards cannot interfere. That looks like
the "test passes for a reason unrelated to what it names" trap this tree keeps
recording, so the distinction matters:

**It is a REGRESSION GUARD, and its value is entirely in existing before the
risky change.** The moment the matvecs go seqs-inner, every failure in §5.3
becomes reachable, and this test is what catches them. Writing it afterwards
means writing it against code you already believe works — which is how a test
gets shaped to pass rather than to bite.

What it must NOT do is claim to have verified independence while the
implementation is sequential. So the test **asserts its own preconditions**:
it records whether the batch path is sequential or fused, and says so in its
output, exactly as the storm now asserts `dedup_hit > 0` rather than assuming
its op mix produced hits.

**Test shape:**
1. Run prompt P alone, `-n 32`, record the ids.
2. Run P in a batch with 1, 3, 7 unrelated sequences, record P's ids each time.
3. All four streams must be byte-identical.
4. Repeat with P at a different batch *position*, since "first in the batch"
   is the case an off-by-one in the inner loop would leave working.

Point 4 is the one an obvious implementation of this test would miss.

## 5.5 Order of work

1. Per-sequence scratch in `Model` (the §5.3 hazard that is pure plumbing).
2. The identity test, against the sequential path — green, and honest about
   why it is green.
3. seqs-inner/weights-outer matvecs.
4. The identity test again, now load-bearing. **Plus the m3-archive
   byte-compare**, which stays the check that matters for any forward-path
   change: the golden gates cover tokenization only.
5. Measure N∈{1..16} for the monotone curve ROADMAP §M4 requires — on a warm
   cache with ISLC stopped, per the 2026-08-11 ops lesson.
