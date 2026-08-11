/* tests/test_blockpool.c -- M4's invariant sim.
 *
 * WHAT IS BEING PROVED: that the enumerated invariants
 * (docs/superpowers/design-notes/2026-08-11-m4-invariants.md) hold after
 * EVERY op, under a randomized alloc/append/fork/release/resize storm --
 * not merely at the end of a tidy scripted sequence. The end-state of a
 * scripted test is the one place a broken structure is most likely to look
 * correct, because the script drove it there.
 *
 * The targeted cases below come first and exist to make FAILURES LEGIBLE:
 * when the storm trips an invariant at op 400,000 the message names which,
 * but a named small case tells you what shape of bug it is.
 */
#include "common.h"

#include "blockpool.h"

#include <string.h>

static int checks = 0, failures = 0;

static void ok(const char* what, bool cond) {
    checks++;
    if (!cond) { failures++; printf("  FAIL: %s\n", what); }
}

static void inv(const BlockPool* bp, const char* where) {
    const char* e = bp_check(bp);
    checks++;
    if (e) { failures++; printf("  FAIL: invariant after %s -- %s\n", where, e); }
}

/* xorshift, so the storm is reproducible from its seed. */
static u64 rng_s = 0x2545F4914F6CDD1Dull;
static u32 rnd(u32 n) {
    rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17;
    return (u32)(rng_s % (n ? n : 1));
}

static void test_basics(void) {
    BlockPool* bp = bp_open(8);
    inv(bp, "open");
    ok("a fresh pool is all free", bp_free_count(bp) == 8);

    BpSeq* s = bp_admit(bp, 32);
    ok("admission succeeds within capacity", s != NULL);
    inv(bp, "admit");

    for (u32 i = 0; i < 16; i++) { ok("append", bp_append(bp, s, 1000 + i)); inv(bp, "append"); }
    ok("16 tokens is exactly one block", bp_seq_nblocks(s) == 1);
    ok("length tracks appends", bp_seq_len(s) == 16);

    ok("append past the block boundary", bp_append(bp, s, 2000));
    ok("17 tokens is two blocks", bp_seq_nblocks(s) == 2);
    inv(bp, "boundary append");

    bp_release(bp, s);
    inv(bp, "release");
    ok("release returns every block", bp_free_count(bp) == 8);
    bp_close(bp);
}

static void test_admission_bound(void) {
    BlockPool* bp = bp_open(4);
    BpSeq* a = bp_admit(bp, 64);            /* 4 blocks: the whole pool */
    ok("a full-pool reservation is admitted", a != NULL);
    ok("a second reservation is REFUSED (I8)", bp_admit(bp, 16) == NULL);
    inv(bp, "refused admission");
    bp_release(bp, a);
    ok("releasing frees the reservation", bp_admit(bp, 16) != NULL);
    bp_close(bp);
}

static void test_fork_cow(void) {
    BlockPool* bp = bp_open(16);
    BpSeq* p = bp_admit(bp, 64);
    for (u32 i = 0; i < 20; i++) bp_append(bp, p, 500 + i);   /* 1 full + tail of 4 */
    inv(bp, "pre-fork");

    u32 free_before = bp_free_count(bp);
    BpSeq* c = bp_fork(bp, p, 64);
    ok("fork succeeds", c != NULL);
    inv(bp, "fork");
    ok("fork copies exactly ONE block (the partial tail)",
       bp_free_count(bp) == free_before - 1);
    ok("child length matches parent", bp_seq_len(c) == bp_seq_len(p));

    /* Both may now append without tripping I7, because each owns its tail. */
    ok("parent appends after fork", bp_append(bp, p, 9001));
    inv(bp, "parent append after fork");
    ok("child appends after fork", bp_append(bp, c, 9002));
    inv(bp, "child append after fork");

    bp_release(bp, p);
    inv(bp, "release parent, child still holds the shared block");
    bp_release(bp, c);
    ok("all blocks return once both are gone", bp_free_count(bp) == 16);
    bp_close(bp);
}

static void test_fork_on_boundary_copies_nothing(void) {
    BlockPool* bp = bp_open(16);
    BpSeq* p = bp_admit(bp, 64);
    for (u32 i = 0; i < 32; i++) bp_append(bp, p, 700 + i);   /* exactly 2 full */
    u32 before = bp_free_count(bp);
    BpSeq* c = bp_fork(bp, p, 64);
    ok("a boundary fork copies NOTHING", c && bp_free_count(bp) == before);
    inv(bp, "boundary fork");
    bp_release(bp, p); bp_release(bp, c);
    bp_close(bp);
}

static void test_fencing(void) {
    BlockPool* bp = bp_open(8);
    BpSeq* s = bp_admit(bp, 32);
    for (u32 i = 0; i < 16; i++) bp_append(bp, s, i);
    u32 free_before = bp_free_count(bp);

    bp_step_begin(bp);
    bp_release(bp, s);
    inv(bp, "release inside a step");
    ok("a block freed mid-step does NOT rejoin the free list (I10)",
       bp_free_count(bp) == free_before);
    bp_step_retire(bp);
    inv(bp, "step retire");
    ok("retirement releases the fence", bp_free_count(bp) == 8);
    bp_close(bp);
}

static void test_resize(void) {
    BlockPool* bp = bp_open(4);
    BpSeq* s = bp_admit(bp, 64);
    for (u32 i = 0; i < 40; i++) bp_append(bp, s, i);
    inv(bp, "pre-grow");

    ok("grow succeeds", bp_resize(bp, 32));
    inv(bp, "grow");
    ok("capacity grew", bp_capacity(bp) == 32);
    ok("the sequence survived the realloc", bp_seq_len(s) == 40);
    ok("appending still works after realloc", bp_append(bp, s, 12345));
    inv(bp, "append after grow");

    ok("shrink into held blocks is REFUSED", !bp_resize(bp, 1));
    inv(bp, "refused shrink");
    bp_release(bp, s);
    ok("shrink succeeds once the blocks are free", bp_resize(bp, 8));
    inv(bp, "shrink");
    bp_close(bp);
}

static void test_dedup(void) {
    BlockPool* bp = bp_open(16);
    BpSeq* s = bp_admit(bp, 64);
    u32 ids[BP_BLOCK_TOKENS];
    for (u32 i = 0; i < BP_BLOCK_TOKENS; i++) { ids[i] = 4000 + i; bp_append(bp, s, ids[i]); }
    inv(bp, "dedup fill");

    u32 hit = bp_dedup_find(bp, 0, ids);
    ok("an exact prefix matches", hit != BP_NIL);

    u32 other[BP_BLOCK_TOKENS];
    memcpy(other, ids, sizeof(other));
    other[7] ^= 1u;
    ok("a one-token difference does NOT match", bp_dedup_find(bp, 0, other) == BP_NIL);
    ok("a different parent hash does NOT match", bp_dedup_find(bp, 99, ids) == BP_NIL);

    BpSeq* t = bp_admit(bp, 64);
    ok("adopt succeeds", bp_seq_adopt(bp, t, hit));
    inv(bp, "adopt");
    ok("adopting shares, it does not copy", bp_seq_nblocks(t) == 1);
    bp_release(bp, s);
    inv(bp, "release the originator while the adopter holds the block");
    bp_release(bp, t);
    bp_close(bp);

}

/* ▶ I9's REAL PATH: a MANUFACTURED HASH COLLISION.
 *
 * Everything in test_dedup misses at the HASH, so it would pass unchanged
 * with the id comparison deleted -- full coverage of the wrong thing, the
 * shape this tree has been caught by four times. The comparison only runs
 * after a hash match, and a 64-bit mixer will not hand one over, so the
 * collision is manufactured: store block A, then force its recorded hash to
 * equal what a DIFFERENT id array B hashes to.
 *
 * A dedup that trusts the hash now returns A for a lookup of B -- silently
 * substituting another sequence's KV behind a prefix that merely hashed the
 * same, which is the one failure in this structure that leaves the output
 * plausible. I9 says it must return a miss. */
static void test_dedup_collision(void) {
    BlockPool* bp = bp_open(16);
    BpSeq* s = bp_admit(bp, 64);

    u32 a[BP_BLOCK_TOKENS], b[BP_BLOCK_TOKENS];
    for (u32 i = 0; i < BP_BLOCK_TOKENS; i++) {
        a[i] = 6000 + i;
        b[i] = 7000 + i;            /* every id differs */
        bp_append(bp, s, a[i]);
    }

    u32 blk = bp_dedup_find(bp, 0, a);
    ok("collision setup: A is findable by its own ids", blk != BP_NIL);
    ok("B does not match before the collision is forced",
       bp_dedup_find(bp, 0, b) == BP_NIL);

    bp_test_force_hash(bp, blk, bp_test_hash(0, b));

    /* The hash now matches for B while the stored ids are still A's. */
    ok("I9: a HASH MATCH with different ids is REFUSED",
       bp_dedup_find(bp, 0, b) == BP_NIL);
    ok("and A no longer matches, since its recorded hash was overwritten",
       bp_dedup_find(bp, 0, a) == BP_NIL);
    inv(bp, "forced collision");

    bp_release(bp, s);
    bp_close(bp);
}

static void storm(u32 ops) {
    enum { NSEQ = 12 };
    u32 n_settok = 0, n_adopt = 0, n_hit = 0;
    /* A SHARED PREFIX, because that is the workload. Looking up a RANDOM
     * 16-token prefix from a 6-symbol alphabet is 6^16 possibilities and
     * never hits -- the first version of this storm reported dedup_hit=0
     * and would have passed as green while testing none of the sharing it
     * exists to test. M5's own trace is a shared system prefix, so the
     * storm now models that. */
    u32 shared[BP_BLOCK_TOKENS];
    for (u32 t = 0; t < BP_BLOCK_TOKENS; t++) shared[t] = 40 + t;
    BlockPool* bp = bp_open(96);
    BpSeq* seq[NSEQ];
    u32 maxtok[NSEQ];
    for (u32 i = 0; i < NSEQ; i++) { seq[i] = NULL; maxtok[i] = 0; }

    u32 in_step = 0;
    for (u32 op = 0; op < ops; op++) {
        u32 i = rnd(NSEQ);
        switch (rnd(12)) {
        case 0: case 1:
            if (!seq[i]) {
                maxtok[i] = 16 * (1 + rnd(6));
                seq[i] = bp_admit(bp, maxtok[i]);
                /* Most sequences open with the shared prefix. */
                if (seq[i] && rnd(3) != 0)
                    for (u32 t = 0; t < BP_BLOCK_TOKENS; t++)
                        bp_seq_set_token(bp, seq[i], t, shared[t]);
            }
            break;
        case 2: case 3: case 4: case 5:
            if (seq[i] && bp_seq_len(seq[i]) < maxtok[i]) bp_append(bp, seq[i], rnd(50));
            break;
        case 6: {
            u32 j = rnd(NSEQ);
            if (seq[i] && !seq[j] && j != i) {
                maxtok[j] = maxtok[i];
                seq[j] = bp_fork(bp, seq[i], maxtok[j]);
            }
            break;
        }
        case 7:
            if (seq[i]) { bp_release(bp, seq[i]); seq[i] = NULL; }
            break;
        case 10:
            /* THE ID PATH. A small token alphabet makes prefixes collide
             * naturally, so dedup gets real hits instead of only misses --
             * a storm that never hits exercises the lookup and none of the
             * sharing it exists to enable. */
            if (seq[i] && bp_seq_len(seq[i]) < maxtok[i]) {
                bp_seq_set_token(bp, seq[i], bp_seq_len(seq[i]), rnd(6));
                n_settok++;
            }
            break;
        case 11: {
            /* Dedup + adopt: find a full block matching a known prefix and
             * share it. Exercises I1/I2/I6/I7 against genuinely shared
             * blocks, which nothing else in the storm produces. */
            u32 hit = bp_dedup_find(bp, 0, shared);
            if (hit != BP_NIL) n_hit++;
            if (hit != BP_NIL && seq[i] && bp_seq_adopt(bp, seq[i], hit)) n_adopt++;
            break;
        }
        case 8:
            if (!in_step) { bp_step_begin(bp); in_step = 1; }
            else { bp_step_retire(bp); in_step = 0; }
            break;
        case 9: {
            u32 cap = bp_capacity(bp);
            u32 want = (rnd(2) == 0) ? cap + 8 : (cap > 40 ? cap - 8 : cap + 8);
            bp_resize(bp, want);
            break;
        }
        default: break;
        }

        const char* e = bp_check(bp);
        if (e) {
            printf("  FAIL: storm op %u -- %s\n", op, e);
            failures++; checks++;
            break;
        }
    }
    checks++;   /* the storm counts as one check when it completes clean */

    if (in_step) bp_step_retire(bp);
    for (u32 i = 0; i < NSEQ; i++) if (seq[i]) bp_release(bp, seq[i]);
    inv(bp, "storm teardown");

    /* A storm that never reached these paths would pass while testing
     * nothing -- the shape this tree keeps getting caught by. Assert the
     * coverage rather than assuming the op mix produced it. */
    printf("  storm coverage: set_token=%u dedup_hit=%u adopt=%u\n",
           n_settok, n_hit, n_adopt);
    ok("the storm actually exercised the id path", n_settok > 0);
    ok("the storm actually HIT dedup, not just missed", n_hit > 0);
    ok("the storm actually adopted shared blocks", n_adopt > 0);
    bp_close(bp);
}

int main(int argc, char** argv) {
    u32 ops = 200000;
    if (argc > 1) ops = (u32)strtoul(argv[1], NULL, 10);

    test_basics();
    test_admission_bound();
    test_fork_cow();
    test_fork_on_boundary_copies_nothing();
    test_fencing();
    test_resize();
    test_dedup();
    test_dedup_collision();
    printf("targeted: %d checks, %d failures\n", checks, failures);

    storm(ops);
    printf("storm: %u ops\n", ops);

    printf("test_blockpool: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
