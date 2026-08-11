# M4 §3 — the block pool

Design, with each decision traced to the constraint that forces it. Nothing
here is a preference; where a choice was genuinely open it says so.

## 3.1 The number that should shape this design

**stoke's KV is f32 today** — `src/kv.c:41` is `xcalloc(total, sizeof(f32))`,
read from the accessor rather than assumed (§1.3 carries the correction and
what it cost). So KV per token is 294,912 B and **a 16-token block holds
4.50 MiB** across all 36 layers, K and V.

| N | ctx | pool @ **f32 (today)** | @ f16 | @ Q8 | blocks |
|---:|---:|---:|---:|---:|---:|
| 8 | 2,048 | **4.83 GB** | 2.42 | 1.21 | 1,024 |
| **16** | **2,048** | **9.66 GB** | 4.83 | 2.42 | 2,048 |
| 16 | 1,024 | 4.83 GB | 2.42 | 1.21 | 1,024 |

**At M5's own target the f32 pool is 9.66 GB against a 2.50 GB model, on a
15.63 GB box — 12.16 GB before activations, the OS, or any page cache.** That
is not a tight fit; it does not fit. This box's recorded failure mode is a
page-cache collapse when a large mmap loses residency (~3× slowdown, measured
live twice), and 12.16 GB of 15.63 GB walks straight into it.

Two consequences, and the first is stronger than the earlier draft claimed:

1. **A KV dtype reduction is REQUIRED to reach M5's target on this hardware,
   not merely a lever.** At f16 the pool is 4.83 GB (7.33 GB with the model);
   at Q8, 2.42 GB (4.92 GB). The ROADMAP already frames Q8-KV as an M4/M5
   option whose default is Kabir's call at M5 — this says the *option* is load-
   bearing for the target, independent of the bandwidth argument in §1.4.
   Those two arguments are separate and both point the same way.
2. **Memory-elastic serving is not a Phase-2 nicety either.** Even at Q8, a
   2,048-block pool plus the model is ~4.9 GB of a box that also runs a
   desktop. ROADMAP §M4 constraint (1) — runtime-variable capacity, give
   blocks *back* under pressure — is what keeps that survivable, and it is why
   §3.2's index-not-pointer decision matters rather than being pedantry.

*The earlier version of this section sized the pool at f16 and reported 4.83 GB
at N=16. It was wrong by 2× in the same way §1.3 was, from the same cause.*

## 3.2 Indices, never pointers

vLLM's "free list == eviction order" input (§5 of the 08-10 note) says embed
`prev`/`next` in the block itself: O(1) everywhere, zero allocation. Correct,
and it transfers to C11 — **but not as pointers.**

Runtime-variable capacity means the descriptor array is reallocated when the
pool grows or shrinks, and **`realloc` invalidates every pointer into it.** A
pointer-linked free list would survive testing at fixed capacity and corrupt on
the first resize — which is exactly the shape ROADMAP §M4 constraint (1)
warns about: *no static-pool assumption baked into invariants or allocation.*

So: `uint32_t prev, next` block **indices**, with `BLOCK_NIL = UINT32_MAX`.
The free list is intrusive, O(1) at both ends, allocation-free, and survives
`realloc` untouched. Eviction order falls out for free: push freed blocks at
the tail, pop candidates from the head, and LRU is the list order itself.

*This is the first place where a borrowed idiom needed re-deriving rather than
copying — the Python original had no realloc to worry about.*

## 3.3 Descriptor and bytes are separate

Placement is a property of the block (08-10 amendment, Kabir), so a block must
be able to change residence without anything that references it changing.

    block descriptor  ->  { placement, arena_slot, refcount, prev, next,
                            n_filled, hash, token_ids[16] }
    arena             ->  [layer][K|V][slot][16][kv_heads][head_dim]

The block *table* of a sequence stores descriptor indices. Moving a block
host↔device rewrites `placement` and `arena_slot` in one descriptor; every
sequence referencing it is unaffected. **Preemption policy then becomes
per-placement** — pointer move on host, recompute-or-transfer on device — which
is what the 08-10 amendment requires and what M4 must not decide once for both.

Invariant **I11** (placement matches actual residence) is the guard that makes
this safe to touch.

## 3.4 Copy-on-write only ever hits the tail block

`fork` is in the sim's op set, so shared prefixes need COW, and **I7** forbids
writing to a block with `refcount > 1`. The naive reading is that any write may
have to copy.

**It cannot.** KV is append-only: once position *i*'s K and V are computed they
never change. A block that is *full* is therefore immutable for the rest of its
life, and the only mutable block a sequence owns is its **tail** — the partial
one currently being filled.

So COW reduces to a single rule: **on `fork`, if the parent's tail block is
partially filled, copy that one block; every full block is shared by refcount
alone.** One 4.50 MiB copy per fork, worst case at today's f32 (2.25 at f16,
1.12 at Q8), and zero copies when the fork point lands on a block boundary.

That is a large simplification and it is worth stating as a property rather
than leaving it implicit, because a later reader adding "COW check" to the
general write path would be adding a branch that can never fire — and a branch
that never fires is a branch nobody tests.

## 3.5 Verified dedup, and what I9 costs

**I9** requires that a dedup match compares stored token ids, not just hashes.
That means each block carries its 16 ids: **64 B against 4,718,592 B of KV, or
0.0014 %.** The strongest invariant in the set is essentially free.

Hash chaining follows vLLM: `hash(block) = H(hash(parent), token_ids)`, so a
match implies the whole prefix matches, not just this block's contents. The
hash *selects a candidate*; I9 asserts the *comparison happened*.

Two policy points inherited rather than invented:

- **The last-token rule.** A prefix hit caps at `prompt_len − 1`: with every
  prompt token cached there is still no logit vector for the next position, so
  at least one token must be recomputed. M4's token-id-keyed dedup LOCK walks
  straight into this.
- **Only full blocks are dedup candidates.** A partial tail has no stable hash
  — it changes on every append — so entering it in the index would mean
  invalidating and re-inserting per token.

## 3.6 Reservation and fencing

- **Full-reservation admission (I8).** Until preemption exists a sequence that
  cannot finish must not start: admit only if `ceil(max_len / 16)` blocks are
  free, and hold that reservation for its lifetime. vLLM reached the same
  constraint after shipping (08-10 note §4).
- **Reserved-against-livelock.** vLLM's stricter form — keep free blocks that
  *already-running* sequences will need — is recorded as an M5 input, not an
  M4 one: with full reservation and no preemption, a running sequence already
  owns everything it will ever need. It becomes load-bearing the moment
  preemption lands.
- **Fenced frees (I10).** A block freed mid-step cannot rejoin the free list
  until that step retires, or an in-flight reader sees another sequence's KV.
  Frees go to a pending list, drained at retirement. vLLM's own comment records
  that fences arrive only *near*-monotonically, so the drain must stop at the
  first pending one rather than scanning past it.

## 3.7 Open, and deliberately not decided here

- **Q8 KV default** is Kabir's at M5. §1.4 and §3.1 give both halves of the
  quantitative case (traffic and footprint); neither decides it.
- **Eviction policy beyond LRU.** Free-list order gives LRU for nothing. The
  agentic trace's shared prefix argues for pinning hot prefix blocks, but that
  is an M5 measurement, not an M4 structure choice — the structure supports it
  either way.
- **The §8.1 EXACT-vs-device collision** stays open and needs a device
  measurement. Nothing in this note prejudges it: the pool is placement-aware
  precisely so the numeric class can be decided per placement later.
