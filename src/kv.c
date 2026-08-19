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
#include "kernels/kernels.h"

#include <string.h>

struct KvStore {
    u32 n_layer, n_head_kv, head_dim, capacity;
    KvDtype dtype;
    /* Bytes per K (or V) vector. head_dim*4 at f32; (head_dim/32)*34 at Q8,
     * where 34 is an f16 scale plus 32 int8. Kept as a FIELD rather than
     * recomputed at each call so the address arithmetic below is the same
     * expression for both representations -- two parallel index formulas is
     * how the two modes drift apart. */
    u64 vec_bytes;
    BlockPool* pool;
    u8* k;   /* [n_layer][capacity][BP_BLOCK_TOKENS][n_head_kv] vectors */
    u8* v;   /* same shape                                             */
};

struct Kv {
    u32 n_layer;
    u32 max_ctx;
    u32 n_head_kv;
    u32 head_dim;

    KvStore* store;
    bool     owns_store;
    BpSeq*   seq;
};

static u32 blocks_needed(u32 tokens) {
    return (tokens + BP_BLOCK_TOKENS - 1u) / BP_BLOCK_TOKENS;
}

u32 kv_blocks_for_ctx(u32 max_ctx) { return blocks_needed(max_ctx); }

u32 kv_block_tokens(void) { return BP_BLOCK_TOKENS; }

KvStore* kv_store_open(u32 n_layer, u32 capacity_blocks, u32 n_head_kv,
                       u32 head_dim, KvDtype dtype) {
    if (n_layer == 0 || capacity_blocks == 0 || n_head_kv == 0 || head_dim == 0) {
        fatal("kv: kv_store_open -- all dims must be > 0 (n_layer=%u "
              "capacity_blocks=%u n_head_kv=%u head_dim=%u)",
              n_layer, capacity_blocks, n_head_kv, head_dim);
    }
    if (dtype != KV_F32 && dtype != KV_Q8_0) {
        fatal("kv: kv_store_open -- unknown dtype %d", (int)dtype);
    }
    if (dtype == KV_Q8_0 && head_dim % KERNELS_Q8_0_BLOCK_ELEMS != 0) {
        fatal("kv: kv_store_open -- Q8 needs head_dim (%u) a multiple of %u; "
              "a partial quantization block has no defined encoding",
              head_dim, KERNELS_Q8_0_BLOCK_ELEMS);
    }
    KvStore* st = (KvStore*)xmalloc(sizeof(KvStore));
    st->n_layer   = n_layer;
    st->n_head_kv = n_head_kv;
    st->head_dim  = head_dim;
    st->capacity  = capacity_blocks;
    st->dtype     = dtype;
    st->vec_bytes = (dtype == KV_F32)
                        ? (u64)head_dim * sizeof(f32)
                        : ((u64)head_dim / KERNELS_Q8_0_BLOCK_ELEMS) *
                              KERNELS_Q8_0_BLOCK_BYTES;
    st->pool      = bp_open(capacity_blocks);

    /* Whole blocks always, so a non-multiple context's tail block is fully
     * backed rather than partially -- the slack is unreachable (pos is
     * bounded by max_ctx) and uniform block size is what lets a block move
     * as one contiguous run per layer. */
    u64 total = (u64)n_layer * capacity_blocks * BP_BLOCK_TOKENS *
                (u64)n_head_kv * st->vec_bytes;
    st->k = (u8*)xcalloc((size_t)total, 1);
    st->v = (u8*)xcalloc((size_t)total, 1);
    return st;
}

void kv_store_close(KvStore* st) {
    if (!st) return;
    bp_close(st->pool);
    free(st->k);
    free(st->v);
    free(st);
}

u32 kv_store_free_blocks(const KvStore* st) { return bp_free_count(st->pool); }

u32 kv_len(const Kv* kv) { return bp_seq_len(kv->seq); }

BlockPool* kv_store_pool(KvStore* st) { return st->pool; }

void kv_note_token(Kv* kv, u32 pos, u32 token_id) {
    if (pos >= kv->max_ctx) {
        fatal("kv: kv_note_token -- pos %u out of range [0,%u)", pos,
              kv->max_ctx);
    }
    if (!bp_seq_set_token(kv->store->pool, kv->seq, pos, token_id)) {
        fatal("kv: kv_note_token -- could not back position %u (max_ctx=%u)",
              pos, kv->max_ctx);
    }
}

u32 kv_adopt_prefix(Kv* kv, const u32* ids, u32 n_ids) {
    BlockPool* bp = kv->store->pool;

    /* Admission-time only -- see kv.h. Checked on LENGTH rather than on block
     * alignment: a sequence 16 positions in is block-aligned and adopting
     * under it would still be wrong. */
    if (bp_seq_len(kv->seq) != 0) return 0;

    u32 max_blocks = n_ids / BP_BLOCK_TOKENS;
    /* The last-token rule. An exact multiple has to give back a whole block,
     * because adoption has no finer grain than one. */
    if (max_blocks > 0 && n_ids % BP_BLOCK_TOKENS == 0) max_blocks--;

    u64 parent  = 0;
    u32 adopted = 0;
    for (u32 k = 0; k < max_blocks; k++) {
        u32 blk = bp_dedup_find(bp, parent, ids + (size_t)k * BP_BLOCK_TOKENS);
        if (blk == BP_NIL) break;
        /* A refusal here is a reservation or aliasing bound, not a miss --
         * stopping is right either way: a prefix must be adopted CONTIGUOUSLY
         * from ordinal 0, since block k's positions are only [16k, 16k+16) if
         * every block before it is in place. */
        if (!bp_seq_adopt(bp, kv->seq, blk)) break;
        parent   = bp_block_hash(bp, blk);
        adopted += BP_BLOCK_TOKENS;
    }
    return adopted;
}

Kv* kv_open_in(KvStore* st, u32 max_ctx) {
    if (max_ctx == 0) fatal("kv: kv_open_in -- max_ctx must be > 0");
    BpSeq* seq = bp_admit(st->pool, max_ctx);
    if (!seq) return NULL;    /* full-reservation admission refused (I8) */

    Kv* kv = (Kv*)xmalloc(sizeof(Kv));
    kv->n_layer    = st->n_layer;
    kv->max_ctx    = max_ctx;
    kv->n_head_kv  = st->n_head_kv;
    kv->head_dim   = st->head_dim;
    kv->store      = st;
    kv->owns_store = false;
    kv->seq        = seq;
    return kv;
}

Kv* kv_open(u32 n_layer, u32 max_ctx, u32 n_head_kv, u32 head_dim) {
    if (n_layer == 0 || max_ctx == 0 || n_head_kv == 0 || head_dim == 0) {
        fatal("kv: kv_open -- all dims must be > 0 (n_layer=%u max_ctx=%u "
              "n_head_kv=%u head_dim=%u)",
              n_layer, max_ctx, n_head_kv, head_dim);
    }
    KvStore* st = kv_store_open(n_layer, blocks_needed(max_ctx), n_head_kv,
                                head_dim, KV_F32);
    Kv* kv = kv_open_in(st, max_ctx);
    if (!kv) fatal("kv: kv_open -- admission failed for max_ctx=%u", max_ctx);
    kv->owns_store = true;
    return kv;
}

void kv_close(Kv* kv) {
    if (!kv) return;
    if (kv->store) {
        if (kv->seq) bp_release(kv->store->pool, kv->seq);
        if (kv->owns_store) kv_store_close(kv->store);
    }
    free(kv);
}

/* THE ONE PLACE KV ADDRESSES ARE COMPUTED. kv_view, kv_write and kv_read all
 * come through here -- M2's rule was that one place owns this arithmetic, and
 * splitting it per representation is exactly how two modes stop meaning the
 * same thing. Returns the byte offset of the (layer, kv_head, pos) vector
 * within either the K or the V array; they have identical shape. */
static u64 kv_vec_offset(Kv* kv, u32 layer, u32 kv_head, u32 pos,
                         const char* who) {
    if (layer >= kv->n_layer) {
        fatal("kv: %s -- layer %u out of range [0,%u)", who, layer,
              kv->n_layer);
    }
    if (kv_head >= kv->n_head_kv) {
        fatal("kv: %s -- kv_head %u out of range [0,%u)", who, kv_head,
              kv->n_head_kv);
    }
    if (pos >= kv->max_ctx) {
        fatal("kv: %s -- pos %u out of range [0,%u)", who, pos, kv->max_ctx);
    }

    KvStore* st = kv->store;
    if (!bp_seq_ensure(st->pool, kv->seq, pos + 1u)) {
        fatal("kv: %s -- could not back position %u (max_ctx=%u)", who, pos,
              kv->max_ctx);
    }

    u32 ord  = pos / BP_BLOCK_TOKENS;
    u32 off  = pos % BP_BLOCK_TOKENS;
    u32 blk  = bp_seq_block(kv->seq, ord);
    u32 slot = bp_block_slot(st->pool, blk);

    /* [n_layer][capacity][16][n_head_kv] vectors; u64 throughout so the
     * multiply chain cannot overflow on a larger future model. */
    u64 vec = ((((u64)layer * st->capacity + slot) * BP_BLOCK_TOKENS + off) *
               st->n_head_kv + kv_head);
    return vec * st->vec_bytes;
}

KvDtype kv_dtype(const Kv* kv) { return kv->store->dtype; }

void kv_write(Kv* kv, u32 layer, u32 kv_head, u32 pos, const f32* k,
              const f32* v) {
    KvStore* st  = kv->store;
    u64      byt = kv_vec_offset(kv, layer, kv_head, pos, "kv_write");
    if (st->dtype == KV_F32) {
        /* Byte-for-byte the memcpy model.c used to do through kv_view. */
        memcpy(st->k + byt, k, (size_t)st->head_dim * sizeof(f32));
        memcpy(st->v + byt, v, (size_t)st->head_dim * sizeof(f32));
    } else {
        quantize_row_q8_0(k, st->head_dim, st->k + byt);
        quantize_row_q8_0(v, st->head_dim, st->v + byt);
    }
}

KvSlot kv_read(Kv* kv, u32 layer, u32 kv_head, u32 pos, f32* scratch) {
    KvStore* st  = kv->store;
    u64      byt = kv_vec_offset(kv, layer, kv_head, pos, "kv_read");
    KvSlot   s;
    if (st->dtype == KV_F32) {
        /* Zero copy: f32 costs exactly what it cost before this pair
         * existed, which is what keeps every parity artifact comparable. */
        s.k = (f32*)(void*)(st->k + byt);
        s.v = (f32*)(void*)(st->v + byt);
        return s;
    }
    if (scratch == NULL) {
        fatal("kv: kv_read -- a Q8 store needs a scratch buffer of "
              "2*head_dim (%u) floats", 2u * st->head_dim);
    }
    dequant_row_q8_0(st->k + byt, st->head_dim, scratch);
    dequant_row_q8_0(st->v + byt, st->head_dim, scratch + st->head_dim);
    s.k = scratch;
    s.v = scratch + st->head_dim;
    return s;
}

KvSlot kv_view(Kv* kv, u32 layer, u32 kv_head, u32 pos) {
    if (kv->store->dtype != KV_F32) {
        /* A pointer is a promise about representation and a quantized store
         * cannot keep it. Callers that need both use kv_read/kv_write. */
        fatal("kv: kv_view -- this store is Q8; there is no f32 address to "
              "return. Use kv_read/kv_write.");
    }
    /* Same arithmetic as kv_read's f32 path, from the same function -- the
     * "full-reservation admission makes this unreachable" fatal for an
     * unbackable position lives there too. */
    u64      byt = kv_vec_offset(kv, layer, kv_head, pos, "kv_view");
    KvStore* st  = kv->store;
    KvSlot   s;
    s.k = (f32*)(void*)(st->k + byt);
    s.v = (f32*)(void*)(st->v + byt);
    return s;
}

u32 kv_n_layer(const Kv* kv)   { return kv->n_layer; }
u32 kv_max_ctx(const Kv* kv)   { return kv->max_ctx; }
u32 kv_n_head_kv(const Kv* kv) { return kv->n_head_kv; }
u32 kv_head_dim(const Kv* kv)  { return kv->head_dim; }
