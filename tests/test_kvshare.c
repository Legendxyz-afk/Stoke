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

    printf("test_kvshare: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
