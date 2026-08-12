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
#include <string.h>  /* memcpy -- MSYS2 gcc pulls it in transitively and WSL
                      * gcc 13.3 does not, so this file built on Windows and
                      * failed the Linux leg with -Werror=implicit-function-
                      * declaration. That asymmetry is the whole reason the
                      * WSL leg is not optional. */

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
    KvStore* st = kv_store_open(2, 12, 2, 4, KV_F32);
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

/* ===========================================================================
 * M4 gate item 3 -- PREFIX ADOPTION. The half that makes dedup a feature
 * rather than a data structure.
 *
 * bp_dedup_find has existed since the pool landed and was called by NOTHING
 * outside its own tests: a sequence's blocks were always allocated fresh, so
 * the machinery could not fire in real inference no matter how correct it
 * was. kv_adopt_prefix is the call that closes that, and these are the
 * properties it has to have.
 *
 * THE BYTE-LEVEL CLAIM IS CHECKED AS POINTER IDENTITY, not as equal values.
 * The plan's wording is "identical physical block ids for the shared span";
 * comparing what kv_view returns is the same statement made where it bites,
 * and it cannot be satisfied by a copy that merely happens to agree.
 *
 * THE LAST-TOKEN RULE IS A CORRECTNESS BOUND, NOT A POLICY. Adopting every
 * position of a fully-cached prompt leaves NO position to run forward, and
 * therefore no logits and no next token. vLLM caps a hit at prompt_len - 1
 * for exactly this reason; since adoption moves in whole blocks, an
 * exactly-block-multiple prompt has to give back its entire last block.
 * ===========================================================================
 */
static void test_prefix_adoption(void) {
    /* 16 blocks. Reservations below total 13 (3+3+3+1+3), so admission is
     * never the thing under test -- what the count has to show is that
     * SHARING keeps allocation below reservation. */
    KvStore*   st   = kv_store_open(2, 16, 2, 4, KV_F32);
    BlockPool* pool = kv_store_pool(st);

    u32 shared[48];
    for (u32 i = 0; i < 48; i++) shared[i] = 1000 + i;

    Kv* a = kv_open_in(st, 48);
    if (!a) { ok("adoption setup: a admitted", false); kv_store_close(st); return; }
    for (u32 p = 0; p < 48; p++) {
        write_pos(a, p, 1.0f + (f32)p);
        kv_note_token(a, p, shared[p]);
    }
    u32 free_after_a = kv_store_free_blocks(st);
    ok("a occupies 3 blocks", free_after_a == 13);

    /* B shares the first 32 ids and then diverges, so adoption must stop at
     * the block boundary where they part -- not before, and not after. */
    u32 partly[48];
    memcpy(partly, shared, sizeof(partly));
    for (u32 i = 32; i < 48; i++) partly[i] = 5000 + i;

    Kv* b = kv_open_in(st, 48);
    if (!b) { ok("adoption setup: b admitted", false); kv_close(a); kv_store_close(st); return; }
    ok("adoption stops at the first block that differs",
       kv_adopt_prefix(b, partly, 48) == 32);
    ok("adoption allocated nothing", kv_store_free_blocks(st) == free_after_a);

    bool same = true;
    for (u32 p = 0; p < 32; p++) {
        KvSlot sa = kv_view(a, 0, 0, p), sb = kv_view(b, 0, 0, p);
        if (sa.k != sb.k || sa.v != sb.v) same = false;
    }
    ok("the adopted span IS the same storage, not a copy of it", same);

    bool distinct = true;
    for (u32 p = 32; p < 48; p++) {
        KvSlot sa = kv_view(a, 0, 0, p), sb = kv_view(b, 0, 0, p);
        if (sa.k == sb.k || sa.v == sb.v) distinct = false;
    }
    ok("past the match each sequence has private storage again", distinct);
    ok("b's adopted span still reads a's values",
       check_pos(b, 0, 1.0f) && check_pos(b, 31, 32.0f));

    /* The last-token rule: every block of this 32-id prompt is cached, and
     * adopting all of it would leave nothing to run forward -- so no logits,
     * and no next token. The whole last block goes back. */
    Kv* c = kv_open_in(st, 48);
    if (!c) ok("adoption setup: c admitted", false);
    else    ok("a fully cached prompt gives back its last block",
               kv_adopt_prefix(c, shared, 32) == 16);

    /* A prompt shorter than one block has no full block to match at all. */
    Kv* d = kv_open_in(st, 16);
    if (!d) ok("adoption setup: d admitted", false);
    else    ok("a sub-block prompt adopts nothing",
               kv_adopt_prefix(d, shared, 15) == 0);

    /* Adoption is an ADMISSION-TIME act: it splices shared blocks in under
     * ordinal 0, so a sequence that has already written anything cannot do
     * it. Checked on a sequence whose write is a single position -- the case
     * a length check catches and a block-alignment check would not. */
    Kv* e = kv_open_in(st, 48);
    if (!e) ok("adoption setup: e admitted", false);
    else {
        (void)kv_view(e, 0, 0, 0);
        ok("a sequence that has already started adopts nothing",
           kv_adopt_prefix(e, shared, 48) == 0);
    }

    ok("the pool's invariants survive adoption", bp_check(pool) == NULL);

    kv_close(a); kv_close(b);
    if (c) kv_close(c);
    if (d) kv_close(d);
    if (e) kv_close(e);
    ok("every adopted block comes back once its holders are gone",
       kv_store_free_blocks(st) == 16);
    kv_store_close(st);
}

/* ===========================================================================
 * M4 gate item 4 -- the Q8 KV representation.
 *
 * Two claims, and they are different claims.
 *
 * (1) f32 MODE IS UNCHANGED BY THE NEW ACCESSOR PAIR. kv_write must be the
 *     memcpy kv_view enabled, and kv_read must hand back the STORAGE pointer
 *     rather than a copy -- checked as pointer identity against kv_view, so
 *     "zero copy" is verified rather than asserted. If that ever stops
 *     holding, f32 stops being free and every parity artifact in the project
 *     is measuring a different thing.
 *
 * (2) Q8 MODE ROUND-TRIPS WITHIN THE BOUND THAT IS ACTUALLY TRUE OF IT.
 *     The playbook asks for |dq(q(x)) - x| <= d/2. ggml's reference computes
 *     d = amax/127 and quantizes with q = round(x/d), but STORES d as f16, so
 *     the reconstruction is f16(d)*q and the error can reach
 *     d/2 + 127*d*2^-11 ~ 0.562*d. The strict bound is therefore NOT expected
 *     to hold, and the reason is the scale's storage, not the quantizer.
 *     Asserting d/2 and widening the constant until it passed would turn a
 *     known property into a fitted one, so this asserts the f16-adjusted
 *     bound and PRINTS the worst case measured against both.
 * ===========================================================================
 */
static void test_kv_dtype(void) {
    const u32 NL = 2, NH = 2, HD = 32; /* Q8 needs head_dim % 32 == 0 */

    /* Values with a real dynamic range and one outlier per vector: a scale
     * chosen from an amax that is not representative is where quantization
     * error actually lives, and a flat test vector would hide it. */
    f32 k_in[32], v_in[32];
    for (u32 d = 0; d < HD; d++) {
        k_in[d] = (f32)((int)d - 16) * 0.37f;
        v_in[d] = (f32)((int)(d * 7u % 31u) - 15) * 0.11f;
    }
    k_in[9]  = 12.5f;   /* outlier: drags the scale for its whole block */
    v_in[20] = -9.75f;

    /* ---- f32: exact, and zero copy ---- */
    {
        KvStore* st = kv_store_open(NL, 4, NH, HD, KV_F32);
        Kv*      a  = kv_open_in(st, 32);
        f32      scratch[64];
        kv_write(a, 1, 1, 5, k_in, v_in);
        KvSlot got  = kv_read(a, 1, 1, 5, scratch);
        KvSlot direct = kv_view(a, 1, 1, 5);
        ok("f32 kv_read returns the storage pointer, not a copy",
           got.k == direct.k && got.v == direct.v);
        bool exact = true;
        for (u32 d = 0; d < HD; d++) {
            if (got.k[d] != k_in[d] || got.v[d] != v_in[d]) exact = false;
        }
        ok("f32 round-trips bit-exactly through kv_write/kv_read", exact);
        kv_close(a);
        kv_store_close(st);
    }

    /* ---- Q8: within the bound the scheme actually has ---- */
    {
        KvStore* st = kv_store_open(NL, 4, NH, HD, KV_Q8_0);
        Kv*      a  = kv_open_in(st, 32);
        f32      scratch[64];
        kv_write(a, 1, 1, 5, k_in, v_in);
        KvSlot got = kv_read(a, 1, 1, 5, scratch);
        ok("Q8 kv_read fills the caller's scratch",
           got.k == scratch && got.v == scratch + HD);

        /* d is per 32-element block; HD is exactly one block here. */
        f32 amax_k = 0.0f, amax_v = 0.0f;
        for (u32 d = 0; d < HD; d++) {
            f32 ak = k_in[d] < 0 ? -k_in[d] : k_in[d];
            f32 av = v_in[d] < 0 ? -v_in[d] : v_in[d];
            if (ak > amax_k) amax_k = ak;
            if (av > amax_v) amax_v = av;
        }
        f32 dk = amax_k / 127.0f, dv = amax_v / 127.0f;

        f32 worst_k = 0.0f, worst_v = 0.0f;
        for (u32 d = 0; d < HD; d++) {
            f32 ek = got.k[d] - k_in[d]; if (ek < 0) ek = -ek;
            f32 ev = got.v[d] - v_in[d]; if (ev < 0) ev = -ev;
            if (ek > worst_k) worst_k = ek;
            if (ev > worst_v) worst_v = ev;
        }
        /* 0.5 + 127*2^-11 = 0.5620. See the block comment. */
        const f32 BOUND = 0.5620f;
        printf("  q8 round-trip: worst |dq(q(x))-x| = %.6g (K, d=%.6g, "
               "%.4f d) / %.6g (V, d=%.6g, %.4f d); strict d/2 would be "
               "%.4f d\n",
               (double)worst_k, (double)dk, (double)(worst_k / dk),
               (double)worst_v, (double)dv, (double)(worst_v / dv), 0.5);
        ok("Q8 K round-trips within the f16-scale-adjusted bound",
           worst_k <= BOUND * dk);
        ok("Q8 V round-trips within the f16-scale-adjusted bound",
           worst_v <= BOUND * dv);
        ok("Q8 is lossy -- a bound nothing violates proves nothing",
           worst_k > 0.0f);

        ok("a Q8 store reports its dtype", kv_dtype(a) == KV_Q8_0);
        kv_close(a);
        kv_store_close(st);
    }
}

int main(void) {
    const u32 NL = 3, NH = 2, HD = 4;

    /* 8 blocks = 128 positions, shared by three sequences of 32 each. */
    KvStore* st = kv_store_open(NL, 8, NH, HD, KV_F32);
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
    test_prefix_adoption();
    test_kv_dtype();

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
        KvStore*  s2 = kv_store_open(NL, 2u * kv_blocks_for_ctx(CTX), NH, HD, KV_F32);
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
