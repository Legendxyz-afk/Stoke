/* src/blockpool.c -- see blockpool.h for the design and its derivations. */
#include "common.h"

#include "blockpool.h"

#include <string.h>

typedef struct {
    u32 prev, next;              /* free-list links; BP_NIL when not listed */
    u32 refcount;                /* I1: == references from live tables      */
    u32 arena_slot;
    u64 hash;                    /* chained H(parent_hash, ids), full only  */
    u32 ids[BP_BLOCK_TOKENS];
    u16 n_filled;                /* 0..BP_BLOCK_TOKENS                      */
    u8  placement;
    u8  on_free;                 /* I2: must agree with refcount == 0       */
    u8  pending;                 /* I10: freed this step, not yet reusable  */
} BpBlock;

struct BpSeq {
    u32* table;                  /* block indices                           */
    u32  nblocks, cap_blocks;
    u32  len;                    /* tokens                                  */
    u32  reserved;               /* blocks held against I8                  */
    BpSeq* next;                 /* pool's live-sequence list               */
    BpSeq* prev;
};

struct BlockPool {
    BpBlock* b;
    u32 capacity;
    u32 free_head, free_tail, free_n;
    u32 pending_head;            /* singly linked via `next`                */
    u32 reserved_total;
    int step_open;
    BpSeq* seqs;                 /* live sequences                          */
};

/* ---- free list (indices, so realloc cannot break it) ---------------- */

static void fl_push_tail(BlockPool* bp, u32 i) {
    BpBlock* x = &bp->b[i];
    x->prev = bp->free_tail;
    x->next = BP_NIL;
    if (bp->free_tail != BP_NIL) bp->b[bp->free_tail].next = i;
    else bp->free_head = i;
    bp->free_tail = i;
    x->on_free = 1;
    bp->free_n++;
}

static u32 fl_pop_head(BlockPool* bp) {
    u32 i = bp->free_head;
    if (i == BP_NIL) return BP_NIL;
    BpBlock* x = &bp->b[i];
    bp->free_head = x->next;
    if (bp->free_head != BP_NIL) bp->b[bp->free_head].prev = BP_NIL;
    else bp->free_tail = BP_NIL;
    x->prev = x->next = BP_NIL;
    x->on_free = 0;
    bp->free_n--;
    return i;
}

static void fl_unlink(BlockPool* bp, u32 i) {
    BpBlock* x = &bp->b[i];
    if (!x->on_free) return;
    if (x->prev != BP_NIL) bp->b[x->prev].next = x->next; else bp->free_head = x->next;
    if (x->next != BP_NIL) bp->b[x->next].prev = x->prev; else bp->free_tail = x->prev;
    x->prev = x->next = BP_NIL;
    x->on_free = 0;
    bp->free_n--;
}

/* ---- lifecycle ------------------------------------------------------ */

static void block_init(BpBlock* x, u32 slot) {
    memset(x, 0, sizeof(*x));
    x->prev = x->next = BP_NIL;
    x->arena_slot = slot;
    x->placement = BP_HOST;
}

BlockPool* bp_open(u32 capacity) {
    if (capacity == 0) fatal("bp_open: capacity must be > 0");
    BlockPool* bp = (BlockPool*)xcalloc(1, sizeof(*bp));
    bp->b = (BpBlock*)xcalloc((size_t)capacity, sizeof(BpBlock));
    bp->capacity = capacity;
    bp->free_head = bp->free_tail = BP_NIL;
    bp->pending_head = BP_NIL;
    for (u32 i = 0; i < capacity; i++) { block_init(&bp->b[i], i); fl_push_tail(bp, i); }
    return bp;
}

void bp_close(BlockPool* bp) {
    if (!bp) return;
    BpSeq* s = bp->seqs;
    while (s) { BpSeq* n = s->next; free(s->table); free(s); s = n; }
    free(bp->b);
    free(bp);
}

bool bp_resize(BlockPool* bp, u32 new_capacity) {
    if (new_capacity == 0) return false;
    if (new_capacity == bp->capacity) return true;

    if (new_capacity < bp->capacity) {
        /* Only shrink into blocks that are free AND not pending. Refusing is
         * the fail-safe direction: handing back a block a sequence holds is
         * corruption, not a policy choice.
         *
         * ▶ EVERY REFUSAL IS DECIDED BEFORE ANY MUTATION. The first version
         * unlinked the doomed blocks and THEN checked the reservation bound,
         * so a refused shrink left them orphaned -- refcount 0, off the free
         * list, not pending -- which is I2's exact violation. The storm found
         * it at op 5,982; all 83 targeted checks had passed. A partially
         * applied "failure" is worse than a failure. */
        if (bp->reserved_total > new_capacity) return false;
        for (u32 i = new_capacity; i < bp->capacity; i++)
            if (bp->b[i].refcount != 0 || bp->b[i].pending) return false;
        for (u32 i = new_capacity; i < bp->capacity; i++) fl_unlink(bp, i);
    }

    BpBlock* nb = (BpBlock*)realloc(bp->b, (size_t)new_capacity * sizeof(BpBlock));
    if (!nb) fatal("bp_resize: out of memory for %u blocks", new_capacity);
    bp->b = nb;

    if (new_capacity > bp->capacity) {
        for (u32 i = bp->capacity; i < new_capacity; i++) {
            block_init(&bp->b[i], i);
            fl_push_tail(bp, i);
        }
    }
    bp->capacity = new_capacity;
    return true;
}

/* ---- admission ------------------------------------------------------ */

static u32 blocks_for(u32 tokens) {
    return (tokens + BP_BLOCK_TOKENS - 1u) / BP_BLOCK_TOKENS;
}

BpSeq* bp_admit(BlockPool* bp, u32 max_tokens) {
    u32 need = blocks_for(max_tokens);
    if (need == 0) need = 1;
    if (bp->reserved_total + need > bp->capacity) return NULL;

    BpSeq* s = (BpSeq*)xcalloc(1, sizeof(*s));
    s->cap_blocks = need;
    s->table = (u32*)xcalloc((size_t)need, sizeof(u32));
    s->reserved = need;
    bp->reserved_total += need;

    s->next = bp->seqs;
    if (bp->seqs) bp->seqs->prev = s;
    bp->seqs = s;
    return s;
}

static void block_deref(BlockPool* bp, u32 i) {
    BpBlock* x = &bp->b[i];
    if (x->refcount == 0) fatal("blockpool: deref of a free block %u", i);
    if (--x->refcount == 0) {
        if (bp->step_open) {
            /* I10: fenced until the step retires. */
            x->pending = 1;
            x->next = bp->pending_head;
            bp->pending_head = i;
        } else {
            x->n_filled = 0;
            x->hash = 0;
            fl_push_tail(bp, i);
        }
    }
}

void bp_release(BlockPool* bp, BpSeq* seq) {
    if (!seq) return;
    for (u32 k = 0; k < seq->nblocks; k++) block_deref(bp, seq->table[k]);
    bp->reserved_total -= seq->reserved;

    if (seq->prev) seq->prev->next = seq->next; else bp->seqs = seq->next;
    if (seq->next) seq->next->prev = seq->prev;
    free(seq->table);
    free(seq);
}

/* ---- the ops -------------------------------------------------------- */

static u32 block_alloc(BlockPool* bp) {
    u32 i = fl_pop_head(bp);
    if (i == BP_NIL) return BP_NIL;
    BpBlock* x = &bp->b[i];
    x->refcount = 1;
    x->n_filled = 0;
    x->hash = 0;
    return i;
}

static u64 hash_block(u64 parent, const u32* ids) {
    u64 h = parent ^ 0x9e3779b97f4a7c15ull;
    for (u32 t = 0; t < BP_BLOCK_TOKENS; t++) {
        h ^= (u64)ids[t] + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    }
    return h ? h : 1u;
}

bool bp_append(BlockPool* bp, BpSeq* seq, u32 token_id) {
    u32 tail_slot = seq->len % BP_BLOCK_TOKENS;

    if (tail_slot == 0) {
        if (seq->nblocks == seq->cap_blocks) return false;  /* past reservation */
        u32 nb = block_alloc(bp);
        if (nb == BP_NIL) return false;
        seq->table[seq->nblocks++] = nb;
    }

    u32 bi = seq->table[seq->nblocks - 1];
    BpBlock* x = &bp->b[bi];
    if (x->refcount != 1) fatal("bp_append: write to shared block %u (I7)", bi);

    x->ids[x->n_filled++] = token_id;
    seq->len++;

    if (x->n_filled == BP_BLOCK_TOKENS) {
        u64 parent = (seq->nblocks >= 2) ? bp->b[seq->table[seq->nblocks - 2]].hash : 0;
        x->hash = hash_block(parent, x->ids);
    }
    return true;
}

bool bp_seq_ensure(BlockPool* bp, BpSeq* seq, u32 n_tokens) {
    if (n_tokens <= seq->len) return true;
    u32 need = blocks_for(n_tokens);
    if (need > seq->cap_blocks) return false;

    while (seq->nblocks < need) {
        u32 nb = block_alloc(bp);
        if (nb == BP_NIL) return false;
        seq->table[seq->nblocks++] = nb;
    }
    /* Fill counts follow the length so I4/I5 hold: every block but the last
     * is full, the last carries the remainder. Ids and hash stay unset. */
    seq->len = n_tokens;
    for (u32 k = 0; k < seq->nblocks; k++) {
        u32 full = (k + 1 < seq->nblocks) ? BP_BLOCK_TOKENS
                                          : n_tokens - k * BP_BLOCK_TOKENS;
        bp->b[seq->table[k]].n_filled = (u16)full;
    }
    return true;
}

bool bp_seq_set_token(BlockPool* bp, BpSeq* seq, u32 pos, u32 token_id) {
    if (!bp_seq_ensure(bp, seq, pos + 1u)) return false;

    u32 ord = pos / BP_BLOCK_TOKENS;
    u32 off = pos % BP_BLOCK_TOKENS;
    BpBlock* x = &bp->b[seq->table[ord]];

    /* I7: never mutate a block someone else can see. A forked sequence
     * shares FULL blocks whose ids are already recorded, so there is
     * nothing to write -- skipping is correct, not a silent failure. */
    if (x->refcount != 1) return true;

    x->ids[off] = token_id;

    if (off == BP_BLOCK_TOKENS - 1u) {
        u64 parent = 0;
        if (ord > 0) {
            parent = bp->b[seq->table[ord - 1]].hash;
            /* ▶ A CHAIN IS ONLY FORMED WHEN THE WHOLE PREFIX IS RECORDED.
             * If the previous block has no hash its ids were never noted,
             * and chaining from 0 would make this block indistinguishable
             * from a FIRST block with the same 16 ids -- a prefix match
             * that skipped everything before it. Leaving the hash at 0
             * makes the block storage-only, which is the safe direction. */
            if (parent == 0) return true;
        }
        x->hash = hash_block(parent, x->ids);
    }
    return true;
}

BpSeq* bp_fork(BlockPool* bp, BpSeq* seq, u32 max_tokens) {
    u32 want = blocks_for(max_tokens);
    if (want < seq->nblocks) want = seq->nblocks;

    BpSeq* child = bp_admit(bp, want * BP_BLOCK_TOKENS);
    if (!child) return NULL;

    for (u32 k = 0; k < seq->nblocks; k++) {
        u32 bi = seq->table[k];
        BpBlock* x = &bp->b[bi];
        bool is_tail = (k + 1 == seq->nblocks);

        if (is_tail && x->n_filled < BP_BLOCK_TOKENS) {
            /* Section 3.4: the tail is the ONLY mutable block, so it is the
             * only one COW can ever need to copy. */
            u32 cp = block_alloc(bp);
            if (cp == BP_NIL) { bp_release(bp, child); return NULL; }
            BpBlock* y = &bp->b[cp];
            memcpy(y->ids, x->ids, sizeof(x->ids));
            y->n_filled = x->n_filled;
            y->hash = x->hash;
            y->placement = x->placement;
            child->table[k] = cp;
        } else {
            x->refcount++;
            child->table[k] = bi;
        }
        child->nblocks++;
    }
    child->len = seq->len;
    return child;
}

/* ---- step fencing --------------------------------------------------- */

void bp_step_begin(BlockPool* bp) { bp->step_open = 1; }

void bp_step_retire(BlockPool* bp) {
    bp->step_open = 0;
    u32 i = bp->pending_head;
    while (i != BP_NIL) {
        u32 nx = bp->b[i].next;
        BpBlock* x = &bp->b[i];
        x->pending = 0;
        x->n_filled = 0;
        x->hash = 0;
        x->prev = x->next = BP_NIL;
        fl_push_tail(bp, i);
        i = nx;
    }
    bp->pending_head = BP_NIL;
}

/* ---- dedup ---------------------------------------------------------- */

u32 bp_dedup_find(const BlockPool* bp, u64 parent_hash, const u32* ids) {
    u64 want = hash_block(parent_hash, ids);
    for (u32 i = 0; i < bp->capacity; i++) {
        const BpBlock* x = &bp->b[i];
        if (x->refcount == 0 || x->n_filled != BP_BLOCK_TOKENS) continue;
        if (x->hash != want) continue;
        /* I9: the hash SELECTS a candidate; this comparison is what makes
         * the match sound. A dedup that stops at the hash is the one failure
         * in this structure that leaves the output plausible. */
        if (memcmp(x->ids, ids, sizeof(x->ids)) == 0) return i;
    }
    return BP_NIL;
}

/* Test-only; see the header for why these exist and why they are not
 * behind a macro. */
u64 bp_test_hash(u64 parent_hash, const u32* ids) {
    return hash_block(parent_hash, ids);
}

void bp_test_force_hash(BlockPool* bp, u32 block, u64 hash) {
    if (block >= bp->capacity) fatal("bp_test_force_hash: block %u out of range", block);
    bp->b[block].hash = hash;
}

bool bp_seq_adopt(BlockPool* bp, BpSeq* seq, u32 block) {
    if (block >= bp->capacity) return false;
    if (seq->nblocks == seq->cap_blocks) return false;
    if (bp->b[block].refcount == 0) return false;
    bp->b[block].refcount++;
    seq->table[seq->nblocks++] = block;
    seq->len += bp->b[block].n_filled;
    return true;
}

/* ---- introspection -------------------------------------------------- */

u32 bp_seq_block(const BpSeq* seq, u32 ordinal) {
    if (ordinal >= seq->nblocks)
        fatal("bp_seq_block: ordinal %u out of range [0,%u)", ordinal, seq->nblocks);
    return seq->table[ordinal];
}

u32 bp_block_slot(const BlockPool* bp, u32 block) {
    if (block >= bp->capacity)
        fatal("bp_block_slot: block %u out of range [0,%u)", block, bp->capacity);
    return bp->b[block].arena_slot;
}

u32 bp_capacity(const BlockPool* bp)   { return bp->capacity; }
u32 bp_free_count(const BlockPool* bp) { return bp->free_n; }
u32 bp_seq_len(const BpSeq* s)         { return s->len; }
u32 bp_seq_nblocks(const BpSeq* s)     { return s->nblocks; }

/* ---- the invariant checker ------------------------------------------ */

const char* bp_check(const BlockPool* bp) {
    u32 cap = bp->capacity;
    u32* seen = (u32*)xcalloc((size_t)cap, sizeof(u32));
    const char* err = NULL;

    /* Count references from every live table (I1), and catch intra-table
     * aliasing (I6) while we are walking them anyway. */
    for (BpSeq* s = bp->seqs; s && !err; s = s->next) {
        for (u32 k = 0; k < s->nblocks && !err; k++) {
            u32 bi = s->table[k];
            if (bi >= cap) { err = "I6: table entry out of range"; break; }
            for (u32 j = 0; j < k; j++)
                if (s->table[j] == bi) { err = "I6: block repeated within one table"; break; }
            seen[bi]++;
        }
        if (err) break;

        /* I4 */
        if (!err && s->nblocks != blocks_for(s->len) && !(s->len == 0 && s->nblocks == 0))
            err = "I4: table length != ceil(len/16)";
        /* I5: every block but the last is full; the last holds the remainder */
        for (u32 k = 0; k + 1 < s->nblocks && !err; k++)
            if (bp->b[s->table[k]].n_filled != BP_BLOCK_TOKENS)
                err = "I5: a non-final block is partially filled";
        if (!err && s->nblocks > 0) {
            u32 want = s->len - (s->nblocks - 1) * BP_BLOCK_TOKENS;
            if (bp->b[s->table[s->nblocks - 1]].n_filled != want)
                err = "I5: tail fill != len mod 16";
        }
    }

    for (u32 i = 0; i < cap && !err; i++) {
        const BpBlock* x = &bp->b[i];
        if (x->refcount != seen[i]) err = "I1: refcount != actual references";
        else if (x->on_free && x->refcount != 0) err = "I2: free block has refcount > 0";
        else if (!x->on_free && x->refcount == 0 && !x->pending)
            err = "I2: unreferenced block is neither free nor pending";
        else if (x->on_free && x->pending) err = "I10: block is both free and pending";
        else if (x->n_filled > BP_BLOCK_TOKENS) err = "I5: n_filled out of range";
        else if (x->placement != BP_HOST && x->placement != BP_DEVICE)
            err = "I11: placement is neither host nor device";
    }

    /* I3: free + allocated + pending == capacity */
    if (!err) {
        u32 alloc_n = 0, pend_n = 0;
        for (u32 i = 0; i < cap; i++) {
            if (bp->b[i].refcount > 0) alloc_n++;
            else if (bp->b[i].pending) pend_n++;
        }
        if (bp->free_n + alloc_n + pend_n != cap) err = "I3: free + allocated + pending != capacity";
    }

    /* I3 again, structurally: the free list must contain exactly free_n
     * nodes and terminate. A count that agrees while the list is cyclic
     * would pass the arithmetic and hang the allocator. */
    if (!err) {
        u32 n = 0, i = bp->free_head, guard = cap + 1;
        while (i != BP_NIL && guard--) { n++; i = bp->b[i].next; }
        if (i != BP_NIL) err = "I3: free list is cyclic or longer than capacity";
        else if (n != bp->free_n) err = "I3: free list length != free_n";
    }

    /* I8 */
    if (!err) {
        u32 r = 0;
        for (BpSeq* s = bp->seqs; s; s = s->next) r += s->reserved;
        if (r != bp->reserved_total) err = "I8: reservation accounting drift";
        else if (bp->reserved_total > cap) err = "I8: reservations exceed capacity";
    }

    free(seen);
    return err;
}
