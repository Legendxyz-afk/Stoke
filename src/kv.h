/* src/kv.h -- M2 contiguous KV cache backing + the kv_view accessor.
 *
 * Contract (M2 op-spec design note, docs/superpowers/design-notes/
 * 2026-07-19-m2-op-spec-and-parity-protocol.md, section 6 "kv_view
 * accessor (plan mandate)"):
 *   - One contiguous allocation per side (K, V), shape
 *     [n_layer][max_ctx][n_head_kv][head_dim], f32. Allocated once at
 *     kv_open() (the x* wrappers), never per token/position -- M2's
 *     "no malloc inside the per-token loop" rule.
 *   - kv_view(kv, layer, kv_head, pos) is the ONLY way anything outside
 *     this file computes a KV address. Attention code (model.c) calls
 *     ONLY kv_view; M4's paged block-table design replaces this file's
 *     body without touching any caller.
 *   - Every out-of-range access (layer/kv_head/pos) is a fatal() naming
 *     which bound was violated -- no silent clamping, no UB.
 *
 * Layout note: for a fixed (layer, kv_head), positions are NOT contiguous
 * in memory (the note's dimension order is [n_layer][max_ctx][n_head_kv]
 * [head_dim], i.e. max_ctx varies faster than n_head_kv) -- attention
 * still just calls kv_view once per (layer, kv_head, position) pair, so
 * this costs nothing but a multiply-add per call, and "scalar cost of the
 * call is irrelevant at M2" per the design note.
 */
#ifndef STOKE_KV_H
#define STOKE_KV_H

#include "common.h"

#include <stdbool.h> /* kv_reserve returns one; common.h does not provide it */

typedef struct Kv Kv;

/* One position's K and V vectors for one (layer, kv_head): both point to
 * head_dim contiguous f32 elements inside Kv's own storage. Never
 * separately allocated/freed -- valid exactly as long as `kv` is open. */
typedef struct {
    f32* k;
    f32* v;
} KvSlot;

/* ---- M4: the shared store -------------------------------------------
 * A KvStore owns the block pool AND the arena those blocks index into.
 * Several Kv handles -- one per sequence, since the frozen kv_view carries
 * no sequence parameter -- share one store. That sharing is the entire
 * point of paged KV: sequences draw blocks from a common pool instead of
 * each reserving a whole contiguous max_ctx.
 *
 * kv_open() below is unchanged and still works: it wraps a private store,
 * so every existing caller keeps its own arena and never notices.
 *
 * `capacity_blocks` is the pool size, not a token count -- the pool is what
 * gets sized against RAM, and at M5's target it is the larger consumer
 * (design note section 3.1: 9.66 GB of f32 KV against a 2.50 GB model). */
typedef struct KvStore KvStore;
typedef struct BlockPool BlockPool;

/* ---- the KV representation (M4 gate item 4) --------------------------
 * KV_F32 is M2's representation and stays the default: exact, and the one
 * every parity artifact was produced against. KV_Q8_0 stores each head
 * vector as ggml Q8_0 blocks (32 elements in 34 bytes -- an f16 scale plus
 * 32 int8), which is 136 bytes against 512 for head_dim 128.
 *
 * The default is Kabir's call at M5, not this layer's; M4's job is to make
 * the option exist and to measure what it costs in divergence.
 *
 * ▶ Q8 requires head_dim % 32 == 0 (fatal otherwise) -- the block size is
 * the quantizer's, and a partial block has no defined encoding. */
typedef enum { KV_F32 = 0, KV_Q8_0 = 1 } KvDtype;

/* The store's pool, for admission decisions, prefix lookups and the sim.
 * Exposed because the pool IS the shared resource -- hiding it would mean
 * re-exporting each of its operations through kv.h one at a time. */
BlockPool* kv_store_pool(KvStore* st);

/* Records which token sits at `pos`. Storage and identity are separate
 * halves of a block: kv_view backs a position so K/V can be written, this
 * says WHICH token it is, and only the second makes a block eligible for
 * prefix dedup. model_forward_position calls it, since that is where both
 * the token id and the position are already in hand.
 *
 * Fatal if pos >= max_ctx, matching kv_view's bound exactly. */
void kv_note_token(Kv* kv, u32 pos, u32 token_id);

/* Blocks a sequence of `max_ctx` positions reserves -- what a caller sizing a
 * shared store multiplies by the batch width. Here rather than at the call
 * site so the block size stays this layer's business: a caller that computes
 * it itself has to know BP_BLOCK_TOKENS, and gets to be wrong about it. */
u32 kv_blocks_for_ctx(u32 max_ctx);

/* The block size in POSITIONS.
 *
 * ▶ WHY THIS EXISTS BESIDE kv_blocks_for_ctx, WHICH WAS ADDED SO CALLERS WOULD
 * NOT NEED IT. That function serves a caller SIZING a store, and such a caller
 * genuinely only needs a count. A caller that must ALIGN TO a block boundary
 * needs the granularity itself, and there is exactly one: M5's chunked prefill,
 * whose chunks must end block-aligned or the prefix cache cannot register what
 * the chunk produced (the S-ALIGN invariant, design note
 * 2026-08-19-m5-entry.md section 1). Deriving it by probing kv_blocks_for_ctx
 * would be the same coupling written obscurely. */
u32 kv_block_tokens(void);

/* ---- prefix dedup (M4 gate item 3) -----------------------------------
 * Adopts the longest run of leading WHOLE blocks of `ids[0..n_ids)` that the
 * store already holds with exactly those ids, sharing them by refcount, and
 * returns how many POSITIONS are now backed. The caller must not recompute
 * those positions -- their K/V is already resident and identical, because a
 * chained hash match means the entire preceding prefix matched too, so the
 * shared block covers the same absolute positions in both sequences and
 * carries K/V rope'd at the same positions.
 *
 * ▶ NEVER RETURNS n_ids. When every prompt token is already cached there is
 * still no logits vector until SOMETHING runs forward, so a fully-cached
 * prompt gives its last block back (vLLM caps a hit at prompt_len - 1 for
 * the same reason).
 *
 * ▶ ADMISSION-TIME ONLY: returns 0 for a sequence that has already been
 * written. Adoption splices shared blocks in from ordinal 0 upward; there is
 * no meaningful way to do that underneath positions a sequence already owns.
 *
 * The result is always a multiple of the block size. */
u32 kv_adopt_prefix(Kv* kv, const u32* ids, u32 n_ids);

KvStore* kv_store_open(u32 n_layer, u32 capacity_blocks, u32 n_head_kv,
                       u32 head_dim, KvDtype dtype);
void     kv_store_close(KvStore* st);
KvDtype  kv_dtype(const Kv* kv);

/* ---- the representation-independent accessor pair --------------------
 * ▶ WHY THIS EXISTS BESIDE kv_view, AND WHAT IT AMENDS.
 * kv_view hands back WRITABLE f32 POINTERS, and a pointer is a promise about
 * representation. A quantized store has no f32 address to give: the bytes
 * are int8 and a scale. M2 could make that promise because there was one
 * representation; the paged body swap kept it because paging changes WHERE
 * an f32 vector lives, not WHAT it is. Q8 changes what it is.
 *
 * So the engine reads and writes through this pair, and kv_view remains the
 * f32-mode accessor (tests use it; it is fatal on a Q8 store). M2's actual
 * rule -- ONE place owns KV address arithmetic -- is untouched: these live
 * in kv.c beside kv_view and share its translation. The signature moved,
 * the ownership did not. Design note:
 * docs/superpowers/design-notes/2026-08-12-m4-q8-kv.md section 3.
 *
 * kv_write writes this position's K and V for one (layer, kv_head), each
 * head_dim floats. kv_read reads them back: the returned pointers are
 * DIRECTLY INTO STORAGE in f32 mode (zero copy -- f32 stays exactly as fast
 * and exactly as exact as it was) and into `scratch` in Q8 mode, where they
 * stay valid until the next kv_read using the same scratch. `scratch` must
 * hold 2*head_dim floats: k at [0, head_dim), v at [head_dim, 2*head_dim).
 * It may be NULL only on an f32 store. */
void   kv_write(Kv* kv, u32 layer, u32 kv_head, u32 pos, const f32* k,
                const f32* v);
KvSlot kv_read(Kv* kv, u32 layer, u32 kv_head, u32 pos, f32* scratch);

/* One sequence's view onto a shared store. Admission is full-reservation
 * (I8), so this returns NULL when the store cannot guarantee max_ctx -- the
 * caller must not start a sequence it cannot finish. */
Kv*  kv_open_in(KvStore* st, u32 max_ctx);

/* M5: a view whose sequence RESERVES NOTHING and may grow opportunistically to
 * `cap_ctx`. Returns NULL only when `cap_ctx` exceeds the store's whole pool.
 *
 * ▶ THE CALLER TAKES ON AN OBLIGATION HERE, and it is not optional. A sequence
 * opened this way can be REFUSED A POSITION mid-generation when the pool runs
 * dry -- and the accessors below do not return that refusal, they fatal on it
 * (kv_vec_offset's "could not back position"), because for every other caller
 * in this program an unbackable position IS a bug. So a caller of this function
 * must know, BEFORE it steps a sequence, that the positions that step will
 * touch are already backed, and preempt something if they are not. That is
 * M5's scheduler; nothing else should call this.
 *
 * Why the accessors were not softened to return an error instead: it would push
 * a "cannot happen" branch into every KV read and write on the hot path, to be
 * ignored by all of them, so that one caller could avoid checking in advance --
 * and checking in advance is what that caller has to do anyway, since by the
 * time a forward pass is writing K/V the step is already running.
 *
 * Decision: superpowers/memos/2026-08-19-m5-admission-reservation-DECISION-
 * OPUS.md. */
Kv*  kv_open_in_partial(KvStore* st, u32 cap_ctx);

/* Grows this sequence's backing to cover `n_positions`, returning false if the
 * pool cannot. THE ONLY PLACE A CALLER CAN LEARN THAT A POSITION IS NOT
 * AVAILABLE WITHOUT DYING OF IT -- the read/write accessors fatal instead. M5's
 * scheduler calls this at step-build time for every row it intends to step, and
 * preempts when it returns false. */
bool kv_reserve(Kv* kv, u32 n_positions);

/* Blocks currently free in the store's pool -- for admission decisions and
 * for tests that need to see sharing actually happen. */
u32  kv_store_free_blocks(const KvStore* st);

/* Positions this sequence currently has storage for. Exposed so a caller
 * that was TOLD a prefix is resident can check that it actually is: the
 * failure mode of prefix dedup is skipping work over KV that was never
 * written, and unlike most bugs here it leaves the run alive and the output
 * merely wrong. */
u32  kv_len(const Kv* kv);

/* Allocates (xcalloc, zero-initialized) both the K and V backing arrays,
 * sized n_layer*max_ctx*n_head_kv*head_dim floats each. All four
 * dimensions must be > 0 (fatal otherwise -- a degenerate KV cache can
 * never be validly indexed). Never returns NULL. Always KV_F32: every
 * existing caller of this is a single-sequence path whose output is
 * compared byte-for-byte against artifacts made at f32. */
Kv* kv_open(u32 n_layer, u32 max_ctx, u32 n_head_kv, u32 head_dim);

/* Frees both backing arrays and `kv` itself. `kv` may be NULL (no-op). */
void kv_close(Kv* kv);

/* Returns pointers to the head_dim-length K/V slice for (layer, kv_head,
 * pos). Fatal (naming the offending argument and its bound) if
 * layer >= n_layer, kv_head >= n_head_kv, or pos >= max_ctx. This is the
 * ONLY function in the program that computes a KV address -- see file
 * header. */
KvSlot kv_view(Kv* kv, u32 layer, u32 kv_head, u32 pos);

/* ---- accessors (the dims kv_open was built with; model.c uses
 * kv_max_ctx to bound-check positions before generating) ---- */
u32 kv_n_layer(const Kv* kv);
u32 kv_max_ctx(const Kv* kv);
u32 kv_n_head_kv(const Kv* kv);
u32 kv_head_dim(const Kv* kv);

#endif /* STOKE_KV_H */
