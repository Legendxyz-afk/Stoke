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

KvStore* kv_store_open(u32 n_layer, u32 capacity_blocks, u32 n_head_kv,
                       u32 head_dim);
void     kv_store_close(KvStore* st);

/* One sequence's view onto a shared store. Admission is full-reservation
 * (I8), so this returns NULL when the store cannot guarantee max_ctx -- the
 * caller must not start a sequence it cannot finish. */
Kv*  kv_open_in(KvStore* st, u32 max_ctx);

/* Blocks currently free in the store's pool -- for admission decisions and
 * for tests that need to see sharing actually happen. */
u32  kv_store_free_blocks(const KvStore* st);

/* Allocates (xcalloc, zero-initialized) both the K and V backing arrays,
 * sized n_layer*max_ctx*n_head_kv*head_dim floats each. All four
 * dimensions must be > 0 (fatal otherwise -- a degenerate KV cache can
 * never be validly indexed). Never returns NULL. */
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
