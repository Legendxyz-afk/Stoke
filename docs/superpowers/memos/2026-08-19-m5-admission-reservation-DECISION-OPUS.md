# M5 admission vs M4's I8 — full reservation makes preemption unreachable

*2026-08-19, Opus 5. A DECISION under the 2026-08-07 governance change
(§8.4-reserved items route to Opus, who decides, records the reasoning and a
direction check; Kabir overrules). Found while writing `sched.h`, before any
scheduler code existed.*

---

## 1. The collision, in one paragraph

M5's locked scheduler admits a prefill when **`free_blocks ≥ blocks(prompt) +
W`**, `W = max(8, 2·|RUNNING|)` — the prompt's blocks plus a small decode
reserve. M4's pool admits differently: `bp_admit(pool, max_tokens)`
(`src/blockpool.c:144`) reserves **`blocks_for(max_tokens)` — the whole
context** — and refuses when `reserved_total + need > capacity`. That is
invariant **I8**, `Σ reserved ≤ capacity`.

**Under I8 as implemented, an admitted sequence can always grow to its full
context, so decode-time pool exhaustion is unreachable — and therefore so is
every preemption path M5 specifies.**

## 2. What that costs, concretely

Not a style question. It makes **four gate-bearing things unreachable**:

| | why it becomes unreachable |
|---|---|
| gate item **[6]** — "preempt-lossless demonstrated on a forced-preemption real-model trace" | nothing can force a preemption |
| the **`pressure` trace** (S0 note §6) — "arrivals sized to force preemption repeatedly" | same |
| **S4** (victim is youngest RUNNING) and **S6** (preempt-replay losslessness) | assertions over a branch that never executes: green, and about nothing |
| the **`preemptions` thrash metric** (Fable 08-02, recorded not gated) | constant zero |

And a fifth cost that is not a gate but is the product: **RAM**. Entry note §6
records that at N=16 / ctx 1500 / f32, full reservation demands **1,504 blocks =
7.1 GB of pool** against a 2.5 GB model — even though the agentic trace's
sequences share a 1–2k prefix and most are short. Full reservation prices every
sequence at its worst case simultaneously, which is exactly the pessimism paged
KV exists to remove.

## 3. Why this is a continuation of M4's design and not a reversal of it

**M4's own invariants note says so.** Failure mode F9 reads:

> *a sequence is admitted without enough blocks reserved for its worst case →
> mid-generation OOM **with no preemption path (M4 has none)***

The parenthetical is the whole argument. Full reservation was the right answer
**because M4 had no way to recover from exhaustion**. M5 builds that way. The
precondition F9 names as the reason for the strict form is precisely what this
milestone removes.

## 4. THE DECISION

> **⚠ §4 WAS AMENDED IN PLACE ON THE SAME DAY, BEFORE ANY CODE WAS WRITTEN.**
> The first version said: set `reserved = blocks_for(n_prompt) + W` and
> `cap_blocks = blocks_for(n_prompt + max_tokens)`, keeping the reserve as a
> *pool-level guarantee*. **That does not work, and the reason is worth more
> than the proposal was.** `Σ reserved ≤ capacity` does NOT protect any
> individual sequence's reserve once sequences may grow beyond their own:
> capacity 100, A and B each reserving 10 with a ceiling of 100 — A grows
> opportunistically to 95, and B's promised 10 blocks are gone while the
> invariant still holds. Making the promise real would require every
> opportunistic allocation to check *other* sequences' unspent reserves, which
> puts admission policy inside the allocator — where it stops being a tunable
> heuristic and becomes an invariant. **The locked design never asked for that:
> `W` is described as decode headroom checked AT ADMISSION, with preemption as
> the backstop.** Caught by working the allocation order through rather than by
> a test, which is the argument for writing the memo before the code.

**Partial admission reserves NOTHING in the pool's accounting. It raises the
CEILING and leaves the policy to the scheduler.**

| | `bp_admit` (unchanged) | `bp_admit_partial` (new) |
|---|---|---|
| `reserved` | `blocks_for(max_tokens)` | **0** |
| `cap_blocks` | `blocks_for(max_tokens)` | `blocks_for(n_prompt + max_tokens)` |
| growth beyond it | cannot fail | fails when the free list is dry ⇒ **preempt** |

- **I8 is untouched, in code and in meaning: `Σ reserved ≤ capacity`.** Sequences
  admitted through the new entry point contribute 0, so the invariant remains
  literally true and the full-reservation guarantee that `stoke run` and
  `stoke batch` depend on is unchanged — their sequences still reserve
  everything and still cannot fail.
- **The watermark lives in `sched.c`**, checked against
  `kv_store_free_blocks()`: `free_blocks ≥ blocks(prompt) + W`. That is where
  the forward note puts it, it is what makes `W` sweepable by the sim, and it
  keeps the allocator a dumb allocator.
- **F9 is answered by preemption instead of by reservation** at M5 — which is
  precisely what F9's own parenthetical anticipated (§3).

**`bp_seq_ensure` already supports the new regime and needs no change**
(`src/blockpool.c:233`): it fails on `need > cap_blocks` *and* on `block_alloc`
returning `BP_NIL`. The second branch is today unreachable and becomes the
pool-exhausted signal the scheduler preempts on. **A path that exists, is
correct, and has never executed is exactly what M5 turns on** — so it gets a
test that forces it, not an assumption that it works.

### 4.1 What must be added, and what must not

- **Add** `bp_admit_partial(pool, cap_tokens)`. Keep `bp_admit` as the
  full-reservation form: `stoke run`, `stoke batch` and every existing test use
  it, and their behaviour must not move under a scheduler change they do not
  participate in.
- **Do not** make `bp_admit` itself partial. That would silently convert every
  existing caller's "guaranteed" into "probably", and the failure would appear
  as a mid-generation fatal in tooling that has no preemption path — the exact
  F9 outcome, reintroduced in the one place nothing can catch it.
- **Add an invariant, do not weaken one.** **I12: for every live sequence,
  `nblocks ≤ cap_blocks` and `cap_blocks ≥ ceil(len/BP_BLOCK_TOKENS)`** — the
  ceiling is real and is honoured, which is the only structural promise a
  partially-admitted sequence gets. I8 continues to hold unchanged across
  admission, growth, preemption and release. The M4 note's standing advice
  applies — *cite the set, never the count*; this makes it twelve, and that note
  already says reasonable derivations land between 9 and 14.
- **And one the sim must assert rather than the pool enforce:** a
  scheduler-admitted sequence that is refused a block must be PREEMPTED, never
  fatal. `bp_seq_ensure` returning false is a normal scheduler event at M5 and a
  caller error everywhere else — so `kv_vec_offset`'s `fatal()` on an unbackable
  position stays exactly as it is, and `sched.c` is responsible for never
  letting a step reach it. **That asymmetry is deliberate and is the single
  most likely place for this decision to go wrong**: the scheduler must decide
  preemption BEFORE the forward, because by the time `kv_write` allocates, the
  step is already running.

### 4.2 The risk this takes on, named

**Livelock**: everything admitted, nothing able to take its next decode block.
That is what `W` is for, and it is why `W` scales with `|RUNNING|` rather than
being a constant. The design's answer is structural, not probabilistic —
preempt-youngest with push-front means the OLDEST request is never preempted,
so by induction on finite work it always completes, and S2/S3/S4 assert exactly
that. **The `pressure` trace exists to attack this and is now able to.**

## 5. Direction check for Kabir

Recorded, not asked — the work proceeds on this decision and reverses cheaply
if he disagrees, since `bp_admit` is untouched and the new entry point has no
other callers.

1. **The trade is honest:** M5 gets preemption, the `pressure` trace and gate
   item [6]; in exchange, "admitted ⇒ will finish without eviction" stops being
   true and becomes "admitted ⇒ will finish, possibly after being preempted and
   replayed". For an inference server that is the normal contract (it is
   vLLM's), and the replay rule makes it invisible in the output.
2. **If he prefers to keep full reservation**, then M5's gate item [6], the
   `pressure` trace and S4/S6 must be struck from the gate rather than
   asserted-and-green — a criterion change, and one I would rather he made
   knowingly than have a milestone close on four assertions about an
   unreachable branch.
3. **Not proposed and deliberately so:** making the reserve a tunable fraction.
   `W`'s FORM is locked by the forward note and changing it is a memo, not a
   knob; this decision changes only what `reserved` is measured over.

## 6. Status

Blocks `sched.c` — which is why it was written before it and not during it. No
code has been changed. The C entry point, I12, and the forced-exhaustion test go
in with S0.
