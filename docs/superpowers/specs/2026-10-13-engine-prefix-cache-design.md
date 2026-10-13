# Engine prefix cache — design

*2026-10-13, Opus 5.5, with Kabir. Brainstormed the same morning; every section
below was approved in that conversation before this was written. It lands after
tag `m7`: `src/` is frozen at `b1ee6b3` until then, so this is a design only.
Each phase gets its own implementation plan, written from this document.*

**Related:** `specs/2026-09-25-lorren-desktop-design.md` §3.2 and §8 (the limit
this removes); `design-notes/2026-08-11-m4-block-pool.md` and
`2026-08-11-m4-invariants.md` (the pool this extends);
`design-notes/2026-08-10-reference-sources-and-m4-design-inputs.md` §3 (vLLM's
prefix cache, which the pool already follows); `design-notes/2026-08-19-m5-entry.md`
§1 (S-ALIGN); `design-notes/2026-08-18-the-surface-web-first.md` §4
(`/internal/state` and the per-request prefix-hit record).

## 1. The problem

Lorren sends a chat's whole history with every message, and the engine computes
the KV for all of it again. Measured 2026-09-25 on Qwen3-4B-Instruct-2507 Q4_K_M
(`--ctx 4096 --seqs 2 --kv q8`), the time to the first token:

| chat so far | first token after |
|---|---|
| 1,107 tokens | 13.3 s (83 tokens/s) |
| 2,196 tokens | 36.5 s (60 tokens/s) |
| 3,318 tokens | 72.9 s (45 tokens/s) |

Ollama keeps its cache and starts in about a second. Lorren's spec calls this
the most valuable engine change for the app. Until it lands, the app sends only
the last ~1,500 tokens of a chat, so long chats forget their start.

## 2. Goal, constraints, decisions

- **Goal:** a long chat's next reply starts about as fast as a short chat's,
  about 1 s.
- **Constraints:**
  - The OpenAI API keeps its shape. The cache is automatic: no client flag.
  - **Replies are byte-identical with the cache on and off.** The engine's
    arithmetic does not depend on batch composition (a standing M4 rule), so
    reused KV equals recomputed KV bit for bit, and a gate can demand it.
  - The public C ABI (frozen at M6) changes only by addition. `stoke_open_opts`
    and `stoke_request` carry a `size` field for exactly this.
  - Running requests never lose memory to the cache.
- **Decided with Kabir, 2026-10-13:**
  - Design it for users, not for this PC: the cache may use **extra RAM that
    adjusts to the machine**, growing while memory is free and giving it back
    when other programs need it (approach A of three; the others were a size
    fixed at start, and llama.cpp's one remembered conversation per slot).
  - **Lorren asks the user before using any extra memory**, with that PC's
    numbers, and the user may enter any amount (§8.3).
  - An **opt-in disk tier** keeps chats' KV across sessions (§9).

## 3. What exists, and what is missing

The pool follows vLLM's design (`2026-08-10` note §3), and most of a prefix
cache is already built:

- 16-token blocks carry a **chained hash**: a block's hash covers every token
  before it, so a hit is always a contiguous run from position 0 (`blockpool.c`
  `hash_block`, `bp_seq_set_token`).
- A lookup **compares the 16 token ids** on every hash hit (invariant I9,
  `bp_dedup_find`).
- **Adoption at admission** (`kv_adopt_prefix`; `sched.c` admit): a request's
  leading whole blocks that are already resident are shared by refcount and
  skipped by prefill, and the scheduler counts them.
- **The last-token rule:** a hit never covers the whole prompt, so something
  always runs forward to give logits.
- **S-ALIGN:** prefill chunks end block-aligned, so every full block gets its
  hash.
- **The free list is the eviction order** (`blockpool.h` header): freed blocks
  join the tail and allocation takes the head.
- A mutant build, `STOKE_MUT_ADOPT_KV`, zeroes the adopted KV, so a gate can
  prove it would notice wrong reuse.

**Missing:** when a block's last reference goes, it is wiped (`block_deref` and
`bp_step_retire` clear its hash and fill count), and the lookup skips free
blocks (`bp_dedup_find` requires refcount > 0). So today's dedup shares only
between requests alive at the same time, and a finished turn's KV is lost.

## 4. Two things outside the cache decide whether it helps

1. **The escape bug (scan F4, queued after `m7`).** The server templates message
   content as the raw JSON slice (`server.c` ~776–785), so a reply's newline
   comes back to the model as the two characters `\n`. The KV holds the newline
   token the model actually generated, so the match breaks at the reply's first
   line break. The fix lands first (§6.6).
2. **Lorren's history window.** Past 1,500 tokens the app drops the oldest turn
   each time, the start of the prompt shifts, and only the system prompt would
   match. Phase 3 sends the whole chat and cuts rarely (§8.1). Until it ships,
   phase 1 helps Lorren only for chats under 1,500 tokens.

## 5. Overview

Four phases, in this order after tag `m7`, each with its own plan and gate:

| phase | what | where | ships alone? |
|---|---|---|---|
| 1 | the escape fix; freed blocks keep their KV; the lookup revives them; `cached_tokens` | engine: `blockpool`, `kv`, `sched`, `server`, `stoke.h` | yes: the 2–3 most recent chats stay warm with no extra RAM |
| 2 | block-major arena; a cache region beyond the working pool, sized against free RAM by a governor; hash index; guard pages; the `cache` state object | engine: `kv`, new `kvmem`, `sched`, `server` | yes |
| 3 | whole-chat history with half-cuts; cached-token and cache-size display; the consent dialog | Lorren | history and display right after phase 1; the dialog after phase 2 |
| 4 | opt-in disk tier: chats' KV saved per `prompt_cache_key` and restored next session | engine: new `kvdisk`, `sched`, `server`; Lorren | after phase 2 |

Before any phase-1 code, a probe (§11.0) measures the one assumption the whole
design rests on.

## 6. Phase 1 — the core cache

### 6.1 Block states

| state | refcount | on the free list | content | matchable |
|---|---|---|---|---|
| LIVE | ≥ 1 | no | in use | yes, when full and hashed |
| CACHED | 0 | yes, toward the tail | full, hashed, valid KV | yes |
| EMPTY | 0 | yes, at the head | nothing usable | no |
| PENDING | 0 | no (the pending list) | freed during an open step | no |

- LIVE → CACHED when the last reference goes and the block is full and hashed;
  LIVE → EMPTY otherwise (a partial tail, or a block whose ids were never
  recorded).
- PENDING → CACHED or EMPTY at `bp_step_retire`, by the same test.
- CACHED → LIVE on adoption (a **revive**).
- CACHED or EMPTY → LIVE on allocation, which wipes the old content. Eviction is
  lazy: a cached block's KV dies only when the block is handed out again.

### 6.2 Free-list order

- EMPTY blocks join the **head**: they hold nothing, so they are reused first.
- CACHED blocks join the **tail**: the least recently used go first.
- `bp_release` drops a sequence's blocks **last to first**, so a chat's first
  blocks enter the free list last and are evicted last. Losing a chat's first
  block would orphan every block after it, since each later hash depends on it.
- Allocation takes the head: an EMPTY block when there is one, otherwise the
  least recently used CACHED block.

### 6.3 Lookup and revive

- `bp_dedup_find` also considers CACHED blocks (never PENDING ones), still
  compares the 16 ids after a hash hit, and prefers a LIVE copy when both exist:
  sharing a LIVE block costs nothing, a revive takes a block off the free list.
- `bp_seq_adopt` on a CACHED block unlinks it from the free list and sets its
  refcount to 1; on a LIVE block it increments the refcount, as today.
- `kv_adopt_prefix`, admission and its accounting do not change: CACHED blocks
  still count as free, a request is admitted on its un-adopted size, and
  adopted positions are skipped by prefill.
- **Preemption gets cheaper for free:** a preempted request's blocks become
  CACHED, and its re-admission usually adopts them back.

### 6.4 Invariants

New, each checked by `bp_check` and each proven by a one-line mutant that must
fail before the rule is trusted:

- **C1** A CACHED block is full and has a non-zero hash.
- **C2** On the free list, every EMPTY block comes before every CACHED block.
- **C3** A lookup never returns a PENDING block.
- **C4** Allocation wipes the hash and fill count before the block is used, so
  a stale hash can never match new content.
- **S-NOTE** A position's token id is recorded only by the forward pass that
  writes its KV (`model_forward_position` → `kv_note_token`; true today).
  Otherwise a reply's last sampled token, never fed back and so never computed,
  could complete a block whose final slot was never written, and that block
  would be cached and reused. Its mutant records the id at sampling time.

Amended: I2's checker accepts a non-zero hash on a free block, which must then
satisfy C1.

### 6.5 Hashing

The existing 64-bit chained hash stays. A false match needs a different prefix
with the same 16 ids and a colliding hash, so its odds per lookup are about
(blocks in the pool) / 2⁶⁴: ~4 × 10⁻¹⁶ for an 8 GB Qwen3 cache (~7,000 blocks),
~1.5 × 10⁻¹⁵ for 30 GB, since users may enter any amount. That is acceptable on
one desktop; a multi-user server revisits it (§12).

### 6.6 The escape fix (F4)

The server decodes each message's JSON string (escapes, `\uXXXX`, surrogate
pairs) into a buffer that lives as long as the request, and templates the
decoded text, as `/v1/completions` already does. Content is still encoded one
message at a time with special-token parsing off, so a decoded `<|im_start|>`
in user text stays text and cannot forge a turn. Replies to multi-line messages
change from today's; that is intended, since today they answer the wrong prompt.

### 6.7 Reporting and the off switch

- **HTTP:** `usage.prompt_tokens_details.cached_tokens` (OpenAI's field), the
  request's adopted positions, in the streaming usage chunk and in
  non-streaming responses.
- **C API:** `int64_t stoke_request_cached(const stoke_engine*, uint64_t id)`
  returns the adopted positions once the request has been admitted, `-1`
  before that, and a negative status for an unknown id.
- **`--no-prefix-cache`** (server) sets `dedup_prefix = 0`: no adoption at all,
  every position computed. It is the reference arm of the byte-identical gate
  and a debugging switch.

## 7. Phase 2 — memory that adjusts to the machine

### 7.1 Block-major arena

Today the arena is two allocations laid out `[layer][block][16][kv_head]`, so a
block's bytes are spread over `n_layer` runs and no page belongs to one block.
Phase 2 lays it out **block-major**: one run per block, its K then its V, each
`[layer][16][kv_head]`, padded to whole 4 KiB pages.

| model | bytes per block | 4 KiB pages |
|---|---|---|
| Qwen3-4B, KV q8 (36 layers, 8 KV heads, head_dim 128) | 1,253,376 | 306 exactly |
| Llama-3.2-3B, KV q8 (28 layers) | 974,848 | 238 exactly |
| Qwen3-4B, KV f32 | 4,718,592 | 1,152 exactly |

A layer's piece of a block is the same size as today (16 × KV heads × vector),
so attention reads the same contiguous pieces; only the distance between a
sequence's blocks changes. `kv_vec_offset` is still the one function that
computes a KV address (M2's rule), so the change stays local. Outputs cannot
change; speed is re-measured (§11.2).

### 7.2 Working pool and cache region

- **Working pool:** `max_seqs × ctx / 16` blocks (512 blocks, 612 MiB for
  Lorren's Qwen3 at q8), committed when the engine opens, as today.
- **Cache region:** address space reserved up to `cache_max_bytes`, with memory
  committed one block at a time, only while a block holds cached KV. 3.0 GB
  holds 2,570 Qwen3 blocks: about 41,000 tokens, ten full-length chats.
- Blocks are interchangeable; the rules are on counts. **Live blocks never
  exceed the working size:** admission, the decode watermark and preemption use
  `working − live`, not the pool's free count, which now includes reserved
  blocks. Committed blocks never fall below the working size and never exceed
  working + the current budget.
- **Platform**, in `src/kvmem.c` behind four functions (reserve, commit,
  decommit, release): Windows `VirtualAlloc(MEM_RESERVE)`, then
  `VirtualAlloc(MEM_COMMIT)` and `VirtualFree(MEM_DECOMMIT)` per block; Linux one
  `mmap(PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE)` and
  `madvise(MADV_DONTNEED)` to give a block back (its next touch reads zeros).

### 7.3 Getting a block

1. A committed EMPTY block.
2. Otherwise, while committed < working + budget, commit a reserved block. This
   is how the cache grows.
3. Otherwise, evict the least recently used CACHED block.

Uncommitted reserved blocks are not on the free list; they wait on a separate
reserve list, so the free list still holds only committed EMPTY blocks (head)
and CACHED blocks (tail). A decommitted block returns to the reserve list.

A commit that fails (Windows refuses the commit charge) counts as "no room to
grow", and step 3 runs instead. Live work cannot be hurt: the working pool is
already committed.

### 7.4 The governor

- It reads the machine's available physical memory (Windows
  `GlobalMemoryStatusEx().ullAvailPhys`, Linux `MemAvailable`) and, on Windows,
  the commit headroom (`ullAvailPageFile`).
- **Floor:** `max(2 GB, 15% of RAM)`, or `cache_min_free_bytes` when set.
- When available memory or commit headroom is below the floor, it decommits the
  least recently used CACHED blocks (and committed EMPTY blocks above the
  working size) until both are above the floor again or the cache is empty. The
  memory goes back to the OS at once.
- The cache may grow again only once available memory is above **floor +
  256 MB** (the gap stops it flapping), and never past `cache_max_bytes`.
- It runs at a step boundary at most once a second, and before any growth
  (§7.3 step 2). It never touches LIVE blocks or the working pool.

### 7.5 Hash index

An open-addressing table from block hash to block indices (duplicates allowed)
replaces the linear scan in `bp_dedup_find`, whose cost grows with capacity
(thousands of blocks here). It is updated when a hash is finalized, when a block
is reallocated, and when it is decommitted. `bp_check` verifies that every LIVE
or CACHED hashed block is in it and nothing else is.

### 7.6 Debug guard pages

Debug and sanitizer builds reserve one inaccessible page after every block, so a
write past a block faults at once. Today the arena is one allocation and
AddressSanitizer cannot see an overrun inside it.

### 7.7 Visibility

`GET /internal/state` (the `2026-08-18` note, §4) carries a `cache` object:
`blocks_cached`, `bytes_committed`, `budget_bytes`, `floor_bytes`,
`tokens_from_cache`, `tokens_computed`, `evictions`, `shrinks`. The block map
gains a "cached" owner class. The note's per-request "prefix-hit blocks and
tokens" is `stoke_request_cached`.

## 8. Phase 3 — Lorren

### 8.1 History

- Send the system prompt, **the whole chat** and the new message, while the
  prompt plus the reply length fit in the context.
- When the next turn would not fit, drop the oldest turns until the system
  prompt plus the kept history is **at most half the context** (2,048 tokens). That turn re-reads about 2,000 tokens (about
  35 s at today's prefill speed), and the waiting line says "re-reading the chat
  after a cut". Every later turn is fast again until the next cut.
- Sizes come from the engine's own `usage` counts for the last turn, not from
  the app's estimate; the engine's 400 reply still corrects a miss.
- The cut marker stays: "N earlier turns not sent: the chat outgrew the
  4,096-token context".
- The 1,500-token history setting goes away, as the Lorren spec planned.

### 8.2 What the user sees

- **Reply panel:** `first token 0.9 s · 3,280 of 3,318 tokens from cache`.
- **CONTEXT panel:** Lorren keeps the chat in memory, so only new text is read;
  a chat that outgrows the context is cut to half.
- **Top bar:** `engine 2.4 GB · cache 1.1 GB`, from `/internal/state`.

### 8.3 Asking the user

Shown once, the first time Lorren starts with this feature, **before** the
engine starts with any extra cache:

> **Faster replies in long chats**
>
> Lorren can keep your recent chats in memory, so it doesn't re-read the whole
> chat before every reply. On this PC that uses up to **3.0 GB** of memory while
> it's free. When other apps need memory, Lorren gives it back at once and
> simply re-reads a chat the next time.
>
> ☐ **Also save it to disk, so long chats start fast next time.** Uses up to
> **10 GB** of disk. A chat's saved memory is deleted with the chat.
>
> **[ Use up to 3.0 GB ]  [ Choose amount… ]  [ No thanks ]**
>
> Unless you turn on saving to disk, the cache lives only in memory and is gone
> when Lorren closes. You can change this any time in Settings.

- **The suggestion:** 20% of the PC's RAM, rounded down to 0.5 GB, at most
  8 GB: 1.5 GB on an 8 GB PC, 3.0 GB on 16 GB, 6.0 GB on 32 GB.
- **Choose amount…** opens a free entry field in GB that accepts **any amount**.
  Above what the PC can spare it adds: "More than this PC usually has free.
  Lorren will use what's free and give it back when other apps need it."
- **No thanks** keeps the phase-1 cache only: no extra memory.
- **The disk checkbox** appears once phase 4 ships, unticked by default; its
  amount is set in Settings (any amount). A user who answered the dialog before
  phase 4 shipped is asked the disk question once, on its own, in the same
  words.
- The choice is saved, and the engine is started with `--cache-max-mb <choice>`
  (and phase 4's `--cache-disk-*` flags). Changing it later restarts the engine,
  which empties the RAM cache; the UI says so.

### 8.4 Settings (Lorren milestone 3)

The same controls, the current RAM and disk use, and **Clear saved cache**.

## 9. Phase 4 — the disk tier (opt-in)

### 9.1 What is saved, and when

- A request may carry **`prompt_cache_key`**, OpenAI's own request field;
  Lorren sends the chat id. Requests without one are never saved.
- After a keyed request finishes, a background writer **appends** the chat's new
  full blocks to that key's file. The blocks are copied out under the
  scheduler's lock (a few to a few dozen a turn), and the disk write never
  blocks a step.
- When a chat is cut, its prefix changes and its file is rewritten: a temp file,
  then an atomic rename.

### 9.2 The file

- One file per key: `<cache dir>/<model id>/<key hash>.skv`, where the key hash
  is a 64-bit hash of the key in hex; the key itself is stored in the header and
  checked, so two keys that collide never share a file. Lorren's cache dir is
  `%LOCALAPPDATA%\Lorren\kvcache` (per user, not roaming).
- **Header:** magic, format version, the engine's **KV-format version**, the
  model's identity (file size, modification time, a hash of the GGUF header),
  KV dtype, `n_layer`, `n_head_kv`, `head_dim`, block size, the key itself, and
  the **committed length**.
- **Per block:** the chained hash, the 16 token ids, a CRC32C, and the block's
  KV bytes.
- **Atomic append:** write the new records and flush, then update the committed
  length in the header and flush again. A reader trusts nothing past the
  committed length, so a crash mid-write leaves a shorter, valid file.

### 9.3 Restoring

- When a keyed request misses in RAM at block *k* and its key has a file, the
  request waits in a new **LOADING** state while a loader thread reads blocks
  *k* onward into a staging buffer, for as long as each block's hash and ids
  match the prompt's chain and its checksum holds.
- Between steps, the scheduler thread takes pool blocks for them (the
  governor's budget applies, and with no room it restores fewer), copies the KV
  in, and marks them CACHED with their hashes. The request is then admitted as
  usual and adopts them. Other requests keep running throughout.
- A full 4,096-token Qwen3 chat is 306 MiB: about 0.2 s to read from NVMe,
  0.6 s from a SATA SSD and a few seconds from a hard disk, against 73 s to
  recompute.

### 9.4 Never wrong, only slower

A file is ignored, and deleted, when anything in its header differs from the
running engine and model. A block that fails its checksum or its id comparison
ends the restore at that block. The **KV-format version** is a constant bumped
by any change that could alter KV bytes: kernels, the quantizer, rope, the block
layout. The byte-identical gate covers restored chats.

### 9.5 Limits and deletion

- Total size is capped (`--cache-disk-max-mb`, the user's amount); the least
  recently used files go first.
- `DELETE /internal/cache/keys/<key>` deletes one chat's file, and Lorren calls
  it when a chat is deleted. `DELETE /internal/cache/keys` clears them all
  (Settings → Clear saved cache).

## 10. API summary

| surface | addition | phase |
|---|---|---|
| HTTP response | `usage.prompt_tokens_details.cached_tokens` | 1 |
| server flag | `--no-prefix-cache` | 1 |
| C API | `stoke_request_cached(e, id)` | 1 |
| HTTP | `GET /internal/state` → `cache` object | 2 |
| server flags | `--cache-max-mb N` (default 0, phase-1 behaviour), `--cache-min-free-mb M` | 2 |
| C API | `stoke_open_opts` += `cache_max_bytes`, `cache_min_free_bytes` | 2 |
| HTTP request | `prompt_cache_key` (optional string) | 4 |
| HTTP | `DELETE /internal/cache/keys[/<key>]` | 4 |
| server flags | `--cache-disk-dir PATH` (off when absent), `--cache-disk-max-mb N` | 4 |
| C API | `stoke_open_opts` += `cache_disk_dir`, `cache_disk_max_bytes`; `stoke_request` += `cache_key`, `cache_key_len` | 4 |

Every C addition is a new function or a field appended to a size-versioned
struct. A caller built against today's header gets today's outputs byte for
byte, only sooner.

## 11. Gates (pre-registered)

### 11.0 Before phase 1: the re-tokenization probe

- **The assumption:** the next turn's prompt re-encodes the assistant's reply
  from its text, and the cache matches only if that gives the tokens the model
  sampled.
- **Method:** 100 replies (50 prompts × Nexus (Llama-3.2-3B) and Qwen3-4B, half
  greedy, half at Lorren's sampling with fixed seeds) through the C API: record
  the sampled ids, detokenize, re-encode with the chat template's assistant
  turn, compare. Report the share identical, the first-divergence positions, and
  the share of history a simulated 10-turn chat would get from cache.
- **Trigger:** if the median simulated chat gets **less than 90%** of its history
  from cache, this document is amended before any code is written.

### 11.1 Phase 1

1. **Byte-identical:** a scripted trace (10 chats × 8 turns, both models, greedy
   and seeded, Lorren's request shape) runs with the cache on and with
   `--no-prefix-cache`. Every stream must match byte for byte;
   `cached_tokens > 0` on every turn after the first (the cache fired, not just
   "was enabled"); the `STOKE_MUT_ADOPT_KV` build must fail the trace.
2. **Pool:** `tests/test_blockpool.c`'s targeted tests and the op storm extended
   with keep, revive and evict; C1–C4 and S-NOTE each with a one-line mutant
   that fails first; the I9 collision test extended to CACHED blocks.
3. **First token:** the §1 measurement re-run (Qwen3-4B, chats of 1,107, 2,196
   and 3,318 tokens, a new message of at most 32 tokens, the turn after a
   warming one). **Gate: ≤ 2 s at every size** (predicted 0.7–1.5 s), on a quiet
   box, against the same sitting's `--no-prefix-cache` run.
4. **No regression:** the M5-bar configuration against tag `m7` under the
   standing measurement rules, and one 6 h soak with the cache on.

### 11.2 Phase 2

1. The byte-identical trace again.
2. **Relayout speed:** a no-regression bar against the phase-1 binary, written
   and committed at phase-2 entry before any measurement, under the standing A/B
   rules (interleaved arms, repeats across process starts, a same-sitting
   referent).
3. **Pressure, on Windows and WSL:** fill the cache to ~2 GB, then a helper
   process eats memory until available memory is below the floor. The cache
   must shrink within 2 s by both the engine's counters and the OS working set;
   live requests keep streaming, byte-identical; after the helper exits, the
   cache grows back with new traffic.
4. A simulated commit refusal stops growth and touches no live work.
5. A debug build that writes one byte past a block faults.
6. **Soak:** the leak rule runs with `--cache-max-mb 0`, since a cache is meant
   to grow memory; a cache-on run checks that committed bytes never exceed
   working + budget.

### 11.3 Phase 3

Unit tests for the history policy, the dialog (shown once; every choice maps to
the right flags; free entry) and the cached-token display; the end-to-end run
with the fake engine taught `cached_tokens`; and a demo with Kabir: a long chat
keeps its first tokens ≤ 2 s except on the cut turn, and the cache shrinks when
a heavy app opens.

### 11.4 Phase 4

1. A long chat continued in a **fresh engine** from its file is byte-identical to
   `--no-prefix-cache`.
2. A file with one flipped byte, a different model or quantization, or a bumped
   KV-format version is ignored, and the output is still identical.
3. Killing the engine mid-write leaves a usable prefix or nothing.
4. Deleting a chat deletes its file; disk use stays under the cap.
5. **A reopened 3,318-token chat's first token ≤ 3 s on an SSD.**

## 12. Risks

| risk | what the design does |
|---|---|
| re-tokenization breaks matches | the §11.0 probe and its amendment trigger |
| the escape fix changes replies to multi-line messages | intended: today they answer the wrong prompt |
| a 64-bit hash false match | ~4 × 10⁻¹⁶ per lookup, recorded; revisited for multi-user |
| **timing leaks between users**: a fast first token tells one user that another sent the same prefix | nothing needed on a one-user desktop; the universal server adds a per-tenant cache salt before serving more than one user |
| Windows runs out of commit charge before RAM | the governor watches commit headroom too; a failed commit only stops growth |
| the relayout slows attention | a pre-registered bar (§11.2) |
| KV on disk is a copy of the conversation | opt-in and off by default, a per-user directory, deleted with the chat, a Clear button; the chat text already lives in Lorren's database |
| the 4,096 cap forces a cut every few thousand tokens | raising it (already queued) makes the cache worth more |
| the soak's RSS rule reads the cache as growth | phase 2's soak grades the leak rule with `--cache-max-mb 0` |

## 13. Out of scope

- Reusing the cache after a cut by shifting KV positions: it changes outputs.
- Sharing disk files between chats (content-addressed storage).
- A conversation or session API: not OpenAI-compatible.
- Raising the 4,096 cap and faster prefill kernels: separate queue items, and
  both make this better.
- Settings beyond the cache controls (Lorren milestone 3).

## 14. Order

After tag `m7`: the escape fix → the §11.0 probe → phase 1 → phase 2 → phase 4.
Phase 3's history and display follow phase 1 directly, its dialog follows
phase 2, and the disk checkbox follows phase 4. One implementation plan per
phase, each written after the previous phase's gate.
