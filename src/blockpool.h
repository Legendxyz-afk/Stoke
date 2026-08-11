/* src/blockpool.h -- M4's paged KV block pool (bookkeeping only).
 *
 * Design: docs/superpowers/design-notes/2026-08-11-m4-block-pool.md
 * Invariants: docs/superpowers/design-notes/2026-08-11-m4-invariants.md
 *
 * ▶ THIS FILE OWNS BLOCK BOOKKEEPING, NOT KV BYTES. A block descriptor
 * carries an `arena_slot` and a `placement`; what lives at that slot is the
 * arena's business, and kv.c's body swap wires it. Keeping them apart is
 * what lets a block change residence (host<->device) without any sequence
 * that references it changing -- the 2026-08-10 amendment's requirement that
 * "where a block lives" be a property of the BLOCK, not an assumption of the
 * allocator. It is also what makes the op storm testable with no model.
 *
 * ▶ INDICES, NEVER POINTERS. The descriptor array is realloc'd by
 * bp_resize (ROADMAP section M4 constraint 1: runtime-variable capacity, so
 * Phase-2 can give blocks BACK under desktop RAM pressure). realloc
 * invalidates every pointer into it, so the intrusive free list links with
 * u32 indices and BP_NIL. A pointer-linked list passes every fixed-capacity
 * test and corrupts on the first resize.
 *
 * ▶ THE FREE LIST IS THE EVICTION ORDER. Freed blocks push at the tail,
 * candidates pop from the head, so LRU falls out of list order at no cost.
 */
#ifndef STOKE_BLOCKPOOL_H
#define STOKE_BLOCKPOOL_H

#include "common.h"

#include <stdbool.h>

#define BP_BLOCK_TOKENS 16u
#define BP_NIL          UINT32_MAX

typedef enum { BP_HOST = 0, BP_DEVICE = 1 } BpPlacement;

typedef struct BlockPool BlockPool;

/* A sequence's block table. Opaque; sequences are owned by the pool so the
 * invariant checker can count references without the caller registering
 * anything (a checker that depends on the caller remembering to register is
 * a checker that silently covers less than it claims). */
typedef struct BpSeq BpSeq;

/* capacity = number of blocks. Never returns NULL. */
BlockPool* bp_open(u32 capacity);
void       bp_close(BlockPool* bp);

/* Runtime-variable capacity. Growing is always allowed. Shrinking is
 * refused (returns false, changes nothing) unless every block being dropped
 * is free -- giving away a block a sequence still holds is not a policy
 * choice, it is corruption. */
bool bp_resize(BlockPool* bp, u32 new_capacity);

/* ---- admission (I8) --------------------------------------------------
 * Full-reservation admission: until preemption exists, a sequence that
 * cannot finish must not start. Reserves ceil(max_tokens/16) blocks for the
 * sequence's whole life. Returns NULL if the reservation does not fit. */
BpSeq* bp_admit(BlockPool* bp, u32 max_tokens);

/* Frees every block the sequence holds and drops its reservation. Legal at
 * ANY point in the sequence's life -- teardown is a first-class op (ROADMAP
 * section M4 constraint 2), and it is what M5's cancellation invariant will
 * stand on. `seq` is invalid afterwards. */
void bp_release(BlockPool* bp, BpSeq* seq);

/* ---- the ops the sim storms ----------------------------------------- */

/* Appends one token id. Allocates a block when the tail is full. Returns
 * false only if the pool is exhausted, which full-reservation admission is
 * supposed to make impossible -- so a false here is a bug in admission, not
 * a condition callers are expected to handle gracefully. */
bool bp_append(BlockPool* bp, BpSeq* seq, u32 token_id);

/* Forks `seq` at its current length. Full blocks are shared by refcount; a
 * partially filled tail is COPIED, because KV is append-only so a full block
 * is immutable for life and the tail is the only mutable block a sequence
 * owns (design note section 3.4). Returns NULL if the child's reservation
 * does not fit. */
BpSeq* bp_fork(BlockPool* bp, BpSeq* seq, u32 max_tokens);

/* ---- step fencing (I10) ---------------------------------------------
 * A block freed mid-step must not be reallocated until that step retires,
 * or an in-flight reader sees another sequence's KV. Frees between
 * bp_step_begin and bp_step_retire land on a pending list. */
void bp_step_begin(BlockPool* bp);
void bp_step_retire(BlockPool* bp);

/* ---- prefix dedup (I9) ----------------------------------------------
 * Looks for a full block whose chained hash matches AND whose stored token
 * ids equal `ids` -- the hash only selects a candidate, the comparison is
 * what makes the match sound. Returns BP_NIL on miss. On a hit the block's
 * refcount is NOT touched; callers adopt it via bp_seq_adopt. */
u32  bp_dedup_find(const BlockPool* bp, u64 parent_hash, const u32* ids);
bool bp_seq_adopt(BlockPool* bp, BpSeq* seq, u32 block);

/* ---- introspection for the checker and the sim ---------------------- */
u32  bp_capacity(const BlockPool* bp);
u32  bp_free_count(const BlockPool* bp);
u32  bp_seq_len(const BpSeq* seq);
u32  bp_seq_nblocks(const BpSeq* seq);

/* Asserts every invariant in the design note. Returns NULL when they all
 * hold, else a static string naming the FIRST that failed. Cheap enough to
 * call after every op in the sim (O(capacity + live tokens)). */
const char* bp_check(const BlockPool* bp);

#endif /* STOKE_BLOCKPOOL_H */
