/* tests/test_kvshare.c -- several sequences over ONE shared KV store.
 *
 * WHAT IS BEING PROVED. Paging only earns its complexity if sequences draw
 * blocks from a COMMON pool; a per-sequence pool is just M2's contiguous
 * cache with extra indirection. So this checks the two things sharing has
 * to get right:
 *
 *   1. sequences get DISJOINT storage -- position p of sequence A and
 *      position p of sequence B must be different memory. The failure here
 *      is one sequence silently reading another's KV, which produces
 *      fluent, plausible, wrong output -- the same class as a bad prefix
 *      dedup, and just as invisible to a test that only checks liveness.
 *   2. blocks actually RETURN to the pool on teardown, at any point in a
 *      sequence's life, so a long-running server does not leak its way to
 *      exhaustion.
 *
 * Writes go through kv_view exactly as model.c's attention does, so this
 * exercises the real accessor rather than a parallel address computation.
 */
#include "common.h"

#include "kv.h"

#include "blockpool.h"

#include <stdbool.h>

static int checks = 0, failures = 0;

static void ok(const char* what, bool cond) {
    checks++;
    if (!cond) { failures++; printf("  FAIL: %s\n", what); }
}

/* Fill every (layer, head) at `pos` with a value derived from `tag`, so any
 * cross-sequence bleed shows up as the wrong tag rather than as a crash. */
static void write_pos(Kv* kv, u32 pos, f32 tag) {
    for (u32 L = 0; L < kv_n_layer(kv); L++) {
        for (u32 h = 0; h < kv_n_head_kv(kv); h++) {
            KvSlot s = kv_view(kv, L, h, pos);
            for (u32 d = 0; d < kv_head_dim(kv); d++) {
                s.k[d] = tag + (f32)d;
                s.v[d] = -tag - (f32)d;
            }
        }
    }
}

static bool check_pos(Kv* kv, u32 pos, f32 tag) {
    for (u32 L = 0; L < kv_n_layer(kv); L++) {
        for (u32 h = 0; h < kv_n_head_kv(kv); h++) {
            KvSlot s = kv_view(kv, L, h, pos);
            for (u32 d = 0; d < kv_head_dim(kv); d++) {
                if (s.k[d] != tag + (f32)d) return false;
                if (s.v[d] != -tag - (f32)d) return false;
            }
        }
    }
    return true;
}

/* Storage and identity are separate halves of a block, and only the second
 * makes it dedup-eligible. These check that the separation holds in BOTH
 * directions -- a noted block becomes findable, an un-noted one never does
 * -- because the dangerous failure is a block matching a prefix whose ids
 * were never actually recorded. */
static void test_token_ids(void) {
    /* 12 blocks, three 32-token sequences = 2 blocks reserved each. Sized
     * against the RESERVATION, not against what gets written: at max_ctx=64
     * each would reserve 4 and the third admission would be refused -- which
     * it was, and dereferencing that NULL is how this test first crashed
     * rather than reported. */
    KvStore* st = kv_store_open(2, 12, 2, 4);
    BlockPool* pool = kv_store_pool(st);
    Kv* a = kv_open_in(st, 32);
    if (!a) { ok("token-id setup: sequence a admitted", false); kv_store_close(st); return; }

    u32 ids[BP_BLOCK_TOKENS];
    for (u32 i = 0; i < BP_BLOCK_TOKENS; i++) {
        ids[i] = 300 + i;
        kv_note_token(a, i, ids[i]);
    }
    ok("a fully noted block becomes dedup-eligible",
       bp_dedup_find(pool, 0, ids) != BP_NIL);

    u32 other[BP_BLOCK_TOKENS];
    for (u32 i = 0; i < BP_BLOCK_TOKENS; i++) other[i] = 900 + i;
    ok("a different prefix does not match", bp_dedup_find(pool, 0, other) == BP_NIL);

    /* Storage without identity: back positions via kv_view only. */
    Kv* b = kv_open_in(st, 32);
    if (!b) { ok("token-id setup: sequence b admitted", false); kv_close(a); kv_store_close(st); return; }
    for (u32 i = 0; i < BP_BLOCK_TOKENS; i++) (void)kv_view(b, 0, 0, i);
    u32 blank[BP_BLOCK_TOKENS];
    for (u32 i = 0; i < BP_BLOCK_TOKENS; i++) blank[i] = 0;
    ok("a block backed but never noted is NOT dedup-eligible",
       bp_dedup_find(pool, 0, blank) == BP_NIL);

    /* THE CHAIN GUARD. Note block 1's ids while block 0's were never
     * recorded. Chaining from a zero parent would make this block
     * indistinguishable from a FIRST block holding the same 16 ids -- a
     * prefix match that silently skipped everything before it. */
    Kv* c = kv_open_in(st, 32);
    if (!c) { ok("token-id setup: sequence c admitted", false); kv_close(a); kv_close(b); kv_store_close(st); return; }
    for (u32 i = 0; i < BP_BLOCK_TOKENS; i++) (void)kv_view(c, 0, 0, i);
    u32 second[BP_BLOCK_TOKENS];
    for (u32 i = 0; i < BP_BLOCK_TOKENS; i++) {
        second[i] = 700 + i;
        kv_note_token(c, BP_BLOCK_TOKENS + i, second[i]);
    }
    ok("a block whose PREFIX was never recorded does not become eligible",
       bp_dedup_find(pool, 0, second) == BP_NIL);

    ok("the pool's invariants survive all of it", bp_check(pool) == NULL);

    kv_close(a); kv_close(b); kv_close(c);
    kv_store_close(st);
}

int main(void) {
    const u32 NL = 3, NH = 2, HD = 4;

    /* 8 blocks = 128 positions, shared by three sequences of 32 each. */
    KvStore* st = kv_store_open(NL, 8, NH, HD);
    ok("a fresh store is all free", kv_store_free_blocks(st) == 8);

    Kv* a = kv_open_in(st, 32);
    Kv* b = kv_open_in(st, 32);
    ok("two sequences share one store", a && b);

    /* Interleaved writes at the SAME positions, distinct tags. */
    for (u32 p = 0; p < 32; p++) {
        write_pos(a, p, 100.0f + (f32)p);
        write_pos(b, p, 900.0f + (f32)p);
    }

    bool a_ok = true, b_ok = true;
    for (u32 p = 0; p < 32; p++) {
        if (!check_pos(a, p, 100.0f + (f32)p)) a_ok = false;
        if (!check_pos(b, p, 900.0f + (f32)p)) b_ok = false;
    }
    ok("sequence A reads back its OWN values", a_ok);
    ok("sequence B reads back its OWN values", b_ok);
    ok("sharing actually consumed blocks from the common pool",
       kv_store_free_blocks(st) == 8 - 4);   /* 2 blocks each */

    /* Teardown mid-life returns storage. */
    kv_close(a);
    ok("closing one sequence returns its blocks", kv_store_free_blocks(st) == 8 - 2);
    ok("the survivor is untouched by the other's teardown",
       check_pos(b, 0, 900.0f) && check_pos(b, 31, 900.0f + 31.0f));

    /* A third sequence reuses the freed blocks. */
    Kv* c = kv_open_in(st, 32);
    ok("a new sequence reuses recycled blocks", c != NULL);
    for (u32 p = 0; p < 32; p++) write_pos(c, p, 500.0f + (f32)p);
    bool still_b = true;
    for (u32 p = 0; p < 32; p++)
        if (!check_pos(b, p, 900.0f + (f32)p)) still_b = false;
    ok("recycled blocks did NOT disturb the live sequence", still_b);

    /* I8: full-reservation admission refuses what it cannot guarantee.
     * Reservations are held for a sequence's LIFE, not sized to what it has
     * written, so the bound is on reserved blocks: b(2) + c(2) leaves 4 of
     * the 8, which is exactly two more 32-token sequences. The sixth is the
     * one that cannot be guaranteed. */
    Kv* d = kv_open_in(st, 32);
    ok("admission succeeds while the pool can guarantee it", d != NULL);
    Kv* e = kv_open_in(st, 32);
    ok("admission succeeds when the reservation fits EXACTLY", e != NULL);
    Kv* f = kv_open_in(st, 32);
    ok("admission is REFUSED once the pool cannot (I8)", f == NULL);

    kv_close(b); kv_close(c); kv_close(d); kv_close(e);
    ok("every block returns once all sequences are gone",
       kv_store_free_blocks(st) == 8);
    kv_store_close(st);

    /* The private-store path (kv_open) still behaves, since every existing
     * caller uses it. */
    Kv* solo = kv_open(NL, 40, NH, HD);
    write_pos(solo, 39, 7.0f);
    ok("kv_open still works and is self-contained", check_pos(solo, 39, 7.0f));
    kv_close(solo);

    test_token_ids();

    /* A caller sizing a shared store has to turn a context length into a
     * block count, and getting it wrong by one block per sequence is
     * invisible until the LAST sequence of a wide batch is refused admission
     * -- i.e. at N=16 and not before. Rounding is the whole content of the
     * function, so the cases are the boundaries. */
    ok("blocks_for_ctx(1) == 1", kv_blocks_for_ctx(1) == 1);
    ok("blocks_for_ctx(16) == 1 (exact fit)", kv_blocks_for_ctx(16) == 1);
    ok("blocks_for_ctx(17) == 2 (the remainder needs a block)",
       kv_blocks_for_ctx(17) == 2);
    ok("blocks_for_ctx(1024) == 64", kv_blocks_for_ctx(1024) == 64);
    {
        /* Not an arithmetic identity but the property the caller needs: a
         * store sized by this function admits exactly that many sequences. */
        const u32 CTX = 33; /* 3 blocks: deliberately not a multiple of 16 */
        KvStore*  s2 = kv_store_open(NL, 2u * kv_blocks_for_ctx(CTX), NH, HD);
        Kv* x = kv_open_in(s2, CTX);
        Kv* y = kv_open_in(s2, CTX);
        Kv* z = kv_open_in(s2, CTX);
        ok("a store sized for 2 admits 2", x != NULL && y != NULL);
        ok("a store sized for 2 refuses the 3rd", z == NULL);
        kv_close(x); kv_close(y);
        kv_store_close(s2);
    }

    printf("test_kvshare: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
