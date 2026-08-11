# M4 §2 — the invariant set, derived from failure modes rather than counted to a target

**The count was asserted before the set existed.** "11 sim invariants" appears
in ROADMAP §M4, the milestone table, and the playbook twice ("ALL 11
invariants", "11 M4 invariants") — and **no document lists them.** This note
derives them from what can actually go wrong, then reports how many there are.

**The hazard being avoided, stated up front:** deriving a set while knowing the
target is 11 makes 11 easy to reach by slicing finely in one place and coarsely
in another. So the failure modes come first — they are objective — and the
invariants are fitted to them afterwards. The number that falls out is
reported, not steered.

## 2.1 What the structure is

From the LOCKs: block = 16 tokens; a per-sequence block table; a shared pool
with a free list; refcounts, because `fork` is in the sim's op set
(`alloc/free/append/fork/release`, playbook §577); a token-id-keyed verified
prefix-dedup index; runtime-variable capacity; and — per the 08-10 amendment —
placement (host/device) as a property of the block.

## 2.2 The failure modes

Objective, and each is a way this data structure corrupts a token stream or
loses memory. Numbered F1–F12 so the mapping below is checkable.

| | failure mode | how it shows up |
|---|---|---|
| F1 | a block is reachable from a sequence *and* on the free list | reallocated under a live sequence; silent KV corruption |
| F2 | a block is on neither the free list nor any table | leak; pool exhausts under a long soak |
| F3 | refcount disagrees with actual references | premature free (too low) or permanent leak (too high) |
| F4 | table length disagrees with sequence length | reads past the written region, or drops the last token |
| F5 | a non-final block is partially filled | attention reads uninitialised KV in the middle of a sequence |
| F6 | the same block appears twice in one table | a write for position *i* corrupts position *j* |
| F7 | a write lands on a block with refcount > 1 | one sequence's generation mutates a forked sibling's prefix |
| F8 | allocated + free ≠ capacity | accounting drift; the pool cannot be resized safely |
| F9 | a sequence is admitted without enough blocks reserved for its worst case | mid-generation OOM with no preemption path (M4 has none) |
| F10 | a dedup match returns a block whose tokens differ from the request | wrong KV silently substituted — the worst failure here, because output stays plausible |
| F11 | a block freed during a step is reallocated before that step retires | in-flight reader sees another sequence's KV |
| F12 | a block's recorded placement disagrees with where its bytes are | reads host memory for a device-resident block, or vice versa |

## 2.3 The invariants

Each must be checkable after *every* op — that is the ROADMAP's requirement,
and it rules out anything needing a full-pool scan per op unless the sim
accepts O(pool) cost (it does: ≥1e6 ops against a small pool is affordable).

| | invariant | covers |
|---|---|---|
| **I1** | `refcount(b)` == number of (sequence, slot) references to `b` across all live tables | F3 |
| **I2** | `b` on the free list ⟺ `refcount(b) == 0` | F1, F2 |
| **I3** | `|free| + |{b : refcount(b) > 0}| == capacity`, and this survives a capacity change | F8 |
| **I4** | `len(table(s)) == ceil(len(s) / 16)` for every live sequence | F4 |
| **I5** | every block of `s` except the last holds exactly 16 tokens; the last holds `((len(s)−1) mod 16) + 1` | F5 |
| **I6** | `table(s)` contains no repeated block id | F6 |
| **I7** | no write is issued to `b` while `refcount(b) > 1` (fork must copy first) | F7 |
| **I8** | Σ reserved blocks over admitted sequences ≤ capacity | F9 |
| **I9** | for any dedup match of prefix `p` to block `b`, the token ids stored in `b` equal `p`'s | F10 |
| **I10** | no block in the pending-free set appears on the free list | F11 |
| **I11** | `placement(b)` equals the pool's actual residence for `b` | F12 |

**Eleven — and I do not claim that vindicates the number.** The set is
sensitive to slicing: I2 could be split into two directions, I4 and I5 could be
merged as one "table shape" invariant, and I11 is arguably premature while the
device path does not exist. Reasonable derivations land anywhere from 9 to 14.
**What is defensible is the coverage — every failure mode in §2.2 has at least
one invariant that catches it — and that is the property worth asserting, not
the cardinality.**

**Recommendation: the ROADMAP should cite the set, not the count.** "The M4
invariants (this note)" is checkable; "the 11 sim invariants" invites the next
session to pad or trim to match a number nobody derived.

## 2.4 What is deliberately not here

- **Teardown completeness is an operation, not an invariant.** "Teardown at any
  lifecycle point frees everything" is exactly I1–I3 continuing to hold after
  `release`, which the sim already asserts after every op. Making it a twelfth
  invariant would double-count.
- **Cancellation-safety belongs to M5** and joins the asserted set there
  (Fable 08-02). It is teardown plus "the remaining schedule does not stall" —
  the second half is a scheduler property with no meaning in M4.
- **The last-token rule** (a prefix hit caps at `prompt_len − 1`, since logits
  need one recomputed token) is a *dedup policy*, not a structural invariant.
  It belongs in the dedup path's tests. Recorded here because M4's
  token-id-keyed dedup LOCK walks straight into it.
- **Batch-composition independence** is a numerical property of the kernels
  (seqs-inner/weights-outer), asserted by the batch-of-N token-identity gate,
  not by the pool sim.

## 2.5 The one that is easy to get wrong

**I9 is the only invariant whose violation leaves the output plausible.** Every
other failure here either crashes, leaks, or produces visibly wrong tokens. A
bad dedup match substitutes a *different sequence's* KV for a prefix that
merely hashed the same, and generation continues fluently from it.

That is why the LOCK says token-id-keyed **verified** dedup rather than
hash-keyed: the hash selects a candidate, and the candidate's stored ids must
then be compared. **I9 asserts the comparison happened, not that the hash
matched.** A sim that only checks hash equality would pass while testing
nothing — the same shape as the selftest that passed for the wrong reason
during the M3 gate.
