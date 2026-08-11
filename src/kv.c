/* src/kv.c -- M4 PAGED KV backing behind M2's frozen accessor.
 *
 * See kv.h for the contract, which is UNCHANGED: kv_view is still the sole
 * KV address computation in the program, still fatal on every out-of-range
 * argument, still no silent clamping. M2's header promised "M4's paged
 * block-table design replaces this file's body without touching any
 * caller", and this is that swap -- main.c, model.c and test_model.c are
 * untouched.
 *
 * Design: docs/superpowers/design-notes/2026-08-11-m4-kv-view-body-swap.md
 *
 * ▶ ONE Kv HANDLE PER SEQUENCE. The frozen signature carries no sequence
 * parameter, so the sequence has to live in the handle. A Kv is "this
 * sequence's view of the KV cache"; the BlockPool underneath is shareable.
 * The rejected alternative was a "current sequence" set on the Kv before
 * each sequence's attention -- a mode, whose failure (reading another
 * sequence's KV) is invisible and produces plausible output.
 *
 * ▶ BLOCKS ARE ALLOCATED BY POSITION, LAZILY. model.c writes K/V through
 * kv_view(pos) and never announces an append, so kv_view grows the sequence
 * to cover `pos` itself. The reachable range is unchanged: pos < max_ctx is
 * valid, pos >= max_ctx is fatal, exactly as before -- so this adds no
 * silent tolerance, it only moves where the storage comes from.
 *
 * ▶ THE ZERO-INIT INSURANCE IS PRESERVED FOR THIS CONFIGURATION, AND ITS
 * LIMIT IS STATED. The arena is xcalloc'd, and a Kv that owns its pool
 * never recycles a block, so every reachable position still reads 0.0f
 * before it is written. Once pools are SHARED across sequences a recycled
 * block will carry the previous sequence's bytes, and the guarantee then
 * rests on the ordering contract (a position is written before it is read)
 * rather than on the allocator. Recorded rather than assumed away.
 */
#include "common.h"

#include "kv.h"

#include "blockpool.h"

struct Kv {
    u32 n_layer;
    u32 max_ctx;
    u32 n_head_kv;
    u32 head_dim;

    BlockPool* pool;
    bool       owns_pool;
    BpSeq*     seq;
    u32        capacity;   /* blocks in the arena                          */

    f32* k;   /* [n_layer][capacity][BP_BLOCK_TOKENS][n_head_kv][head_dim] */
    f32* v;   /* same shape                                                */
};

static u32 blocks_needed(u32 tokens) {
    return (tokens + BP_BLOCK_TOKENS - 1u) / BP_BLOCK_TOKENS;
}

Kv* kv_open(u32 n_layer, u32 max_ctx, u32 n_head_kv, u32 head_dim) {
    if (n_layer == 0 || max_ctx == 0 || n_head_kv == 0 || head_dim == 0) {
        fatal("kv: kv_open -- all dims must be > 0 (n_layer=%u max_ctx=%u "
              "n_head_kv=%u head_dim=%u)",
              n_layer, max_ctx, n_head_kv, head_dim);
    }

    Kv* kv = (Kv*)xmalloc(sizeof(Kv));
    kv->n_layer   = n_layer;
    kv->max_ctx   = max_ctx;
    kv->n_head_kv = n_head_kv;
    kv->head_dim  = head_dim;
    kv->capacity  = blocks_needed(max_ctx);

    kv->pool      = bp_open(kv->capacity);
    kv->owns_pool = true;
    kv->seq       = bp_admit(kv->pool, max_ctx);
    if (!kv->seq) fatal("kv: kv_open -- admission failed for max_ctx=%u", max_ctx);

    /* The arena rounds max_ctx up to a whole block, so the tail block of a
     * non-multiple context is fully backed rather than partially -- reading
     * the slack is impossible (pos is bounded by max_ctx) and allocating it
     * keeps every block the same size, which is what lets a block be moved
     * as one contiguous run per layer. */
    u64 total = (u64)n_layer * kv->capacity * BP_BLOCK_TOKENS *
                (u64)n_head_kv * (u64)head_dim;
    kv->k = (f32*)xcalloc((size_t)total, sizeof(f32));
    kv->v = (f32*)xcalloc((size_t)total, sizeof(f32));
    return kv;
}

void kv_close(Kv* kv) {
    if (!kv) return;
    if (kv->pool) {
        if (kv->seq) bp_release(kv->pool, kv->seq);
        if (kv->owns_pool) bp_close(kv->pool);
    }
    free(kv->k);
    free(kv->v);
    free(kv);
}

KvSlot kv_view(Kv* kv, u32 layer, u32 kv_head, u32 pos) {
    if (layer >= kv->n_layer) {
        fatal("kv: kv_view -- layer %u out of range [0,%u)", layer,
              kv->n_layer);
    }
    if (kv_head >= kv->n_head_kv) {
        fatal("kv: kv_view -- kv_head %u out of range [0,%u)", kv_head,
              kv->n_head_kv);
    }
    if (pos >= kv->max_ctx) {
        fatal("kv: kv_view -- pos %u out of range [0,%u)", pos,
              kv->max_ctx);
    }

    if (!bp_seq_ensure(kv->pool, kv->seq, pos + 1u)) {
        /* Full-reservation admission is supposed to make this unreachable,
         * so it is a bug in the reservation rather than a condition callers
         * handle -- fatal, naming what could not be satisfied. */
        fatal("kv: kv_view -- could not back position %u (max_ctx=%u)", pos,
              kv->max_ctx);
    }

    u32 ord  = pos / BP_BLOCK_TOKENS;
    u32 off  = pos % BP_BLOCK_TOKENS;
    u32 blk  = bp_seq_block(kv->seq, ord);
    u32 slot = bp_block_slot(kv->pool, blk);

    /* [n_layer][capacity][16][n_head_kv][head_dim]; u64 throughout so the
     * multiply chain cannot overflow on a larger future model. */
    u64 idx = ((((u64)layer * kv->capacity + slot) * BP_BLOCK_TOKENS + off) *
               kv->n_head_kv + kv_head) * kv->head_dim;

    KvSlot s;
    s.k = kv->k + idx;
    s.v = kv->v + idx;
    return s;
}

u32 kv_n_layer(const Kv* kv)   { return kv->n_layer; }
u32 kv_max_ctx(const Kv* kv)   { return kv->max_ctx; }
u32 kv_n_head_kv(const Kv* kv) { return kv->n_head_kv; }
u32 kv_head_dim(const Kv* kv)  { return kv->head_dim; }
