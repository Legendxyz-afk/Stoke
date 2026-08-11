# M4 §4 — the `kv_view` body swap

M2 froze the accessor with an explicit promise (`src/kv.h` header): *"M4's
paged block-table design replaces this file's body without touching any
caller."* This note settles how, and records one tension the freeze creates
that is worth knowing before code is written.

## 4.1 The signature problem

    KvSlot kv_view(Kv* kv, u32 layer, u32 kv_head, u32 pos);

**There is no sequence parameter.** M2 had exactly one sequence, so `pos` was
unambiguous. Under M4 a position means nothing without knowing whose it is,
and the frozen signature cannot carry it.

Three ways out:

| | approach | verdict |
|---|---|---|
| A | **one `Kv` handle per sequence**, all sharing a `BlockPool` | **chosen** |
| B | `Kv` holds a "current sequence" the scheduler sets before each seq | rejected |
| C | change the signature | forbidden by the freeze |

**A is chosen and it is not a close call.** `Kv*` is already per-run state in
M2; making it per-sequence keeps the accessor's meaning intact — a `Kv` is
*this sequence's view of the KV cache* — and the block pool underneath is
shared. Callers pass the `Kv` they already hold. Nothing in `model.c` changes.

**B is rejected because a hidden "current" is a mode**, and the 08-10 design
inputs note already records mode-shaped state as an anti-pattern not to
reproduce. Every call site would silently depend on a setter having run, and
the failure — reading another sequence's KV — is invisible and produces
plausible output, the same class as a bad dedup match.

So:

    struct Kv {
        BlockPool* pool;      /* shared, not owned */
        BpSeq*     seq;       /* this view's sequence */
        u32 n_layer, n_head_kv, head_dim, max_ctx;
    };

`kv_open` admits a sequence into the pool; `kv_close` releases it, which is
the teardown-at-any-point op the pool already supports.

## 4.2 The translation

    ord  = pos / BP_BLOCK_TOKENS         /* shift: block size is 16      */
    off  = pos % BP_BLOCK_TOKENS
    blk  = seq->table[ord]
    slot = pool->block[blk].arena_slot
    idx  = ((layer * capacity + slot) * 16 + off) * n_head_kv + kv_head
    k    = arena_k + idx * head_dim

Arena shape becomes `[n_layer][capacity][16][n_head_kv][head_dim]` per side,
which keeps a block's bytes contiguous per layer — the property that lets a
block be relocated host↔device by copying one contiguous run per layer.

Bounds behaviour is preserved exactly: fatal naming the offending argument,
no silent clamping. `ord >= seq->nblocks` becomes a fourth fatal case, and it
is a real one — it means attention asked for a position the sequence has not
written.

## 4.3 The tension the freeze creates, recorded before it bites

**The frozen accessor's granularity is wrong for paging, and this is
structural rather than a matter of tuning.**

`kv_view` is called per `(layer, kv_head, pos)`. The page translation —
`ord`, `blk`, `slot` — depends on **`pos` alone**. It is invariant across
layer and kv_head, so the frozen signature forces the same lookup to be
recomputed `n_layer × n_head_kv` times per position: **36 × 8 = 288× more
often than necessary.**

At a 2k context that is ~590,000 calls per token, each now carrying a
dependent load chain (`seq->table` → block descriptor → `arena_slot`) where
M2 had one multiply-add. M2's header says "scalar cost of the call is
irrelevant at M2" — that judgement was correct for M2 and does not carry.

**What this note does NOT do is pre-emptively add a batched accessor.** The
honest sequence is:

1. Swap the body as promised, callers untouched. Correctness first.
2. **Measure** the per-token cost against the K6 bar (≥7.8 tok/s, currently
   passing at 8.08–8.46, so the margin is ~4–8 %). The budget in §1 says M5 is
   not bandwidth-bound, which makes scalar overhead the thing most likely to
   eat that margin.
3. Only if measurement justifies it, add a *hoisted* accessor **alongside**
   `kv_view` — e.g. resolving a block once per position and returning a
   per-layer stride — and amend M2's "kv_view is the ONLY way anything
   computes a KV address" rule explicitly rather than quietly.

Step 3 amends a written M2 rule, so it is §8.4-reserved: Opus decides, records
the reasoning and a direction check, Kabir overrules. **Flagged now so it is a
decision when it arrives, not a surprise.**

## 4.4 What has to be true before the swap lands

- The pool's ×1e6 storm green under ASan — **done 2026-08-11, 22.8 s, zero
  findings.**
- `bp_append` must be driven by whatever advances a sequence's length, so a
  position is never read before it is written. In M2 that coupling was
  implicit in one contiguous array; under paging it becomes an ordering
  contract between the decode loop and the pool.
- Batch-composition independence stays EXACT and is unaffected: paging changes
  *where* a K/V vector lives, never the order of any arithmetic that reads it.
  Worth stating because it is the one gate a storage change could plausibly
  threaten, and it does not.
