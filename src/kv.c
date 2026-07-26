/* src/kv.c -- M2 contiguous KV cache backing. See kv.h for the full
 * contract. One flat malloc'd array per side (K, V), dimension order
 * [n_layer][max_ctx][n_head_kv][head_dim] (M2 op-spec design note
 * section 6) -- ALL address arithmetic for KV lives here; nothing
 * outside this file may compute one (kv_view is the sole accessor).
 */
#include "common.h"

#include "kv.h"

struct Kv {
    u32 n_layer;
    u32 max_ctx;
    u32 n_head_kv;
    u32 head_dim;

    f32* k; /* [n_layer][max_ctx][n_head_kv][head_dim], flat */
    f32* v; /* same shape */
};

Kv* kv_open(u32 n_layer, u32 max_ctx, u32 n_head_kv, u32 head_dim) {
    if (n_layer == 0 || max_ctx == 0 || n_head_kv == 0 || head_dim == 0) {
        fatal("kv: kv_open -- all dims must be > 0 (n_layer=%u max_ctx=%u "
              "n_head_kv=%u head_dim=%u)",
              n_layer, max_ctx, n_head_kv, head_dim);
    }

    u64 total = (u64)n_layer * (u64)max_ctx * (u64)n_head_kv * (u64)head_dim;

    Kv* kv = (Kv*)xmalloc(sizeof(Kv));
    kv->n_layer = n_layer;
    kv->max_ctx = max_ctx;
    kv->n_head_kv = n_head_kv;
    kv->head_dim = head_dim;
    /* xcalloc, not xmalloc: zero-initialized so any position beyond the
     * ones actually written so far (there are none such reachable from
     * kv_view's own bound checks, but this is cheap insurance against an
     * uninitialized-read finding under ASan/valgrind if a future caller
     * ever mis-sequences writes vs reads) reads as a clean 0.0f rather
     * than whatever the allocator handed back. */
    kv->k = (f32*)xcalloc((size_t)total, sizeof(f32));
    kv->v = (f32*)xcalloc((size_t)total, sizeof(f32));
    return kv;
}

void kv_close(Kv* kv) {
    if (!kv) return;
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

    /* [n_layer][max_ctx][n_head_kv][head_dim], flat, u64 throughout so
     * the multiply chain cannot overflow u32 on a larger future model
     * (this model's own total is ~151M elements/side already, comfortably
     * inside u32 range, but the arithmetic costs nothing extra in u64). */
    u64 idx = (((u64)layer * kv->max_ctx + pos) * kv->n_head_kv + kv_head) *
              kv->head_dim;

    KvSlot slot;
    slot.k = kv->k + idx;
    slot.v = kv->v + idx;
    return slot;
}

u32 kv_n_layer(const Kv* kv) { return kv->n_layer; }
u32 kv_max_ctx(const Kv* kv) { return kv->max_ctx; }
u32 kv_n_head_kv(const Kv* kv) { return kv->n_head_kv; }
u32 kv_head_dim(const Kv* kv) { return kv->head_dim; }
