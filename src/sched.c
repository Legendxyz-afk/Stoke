/* src/sched.c -- M5 continuous batching: the scheduler.
 *
 * Contract and design authority: sched.h, the forward note's section M5
 * (LOCKED), and the S0 architecture note. This file implements them and
 * decides nothing they already decided.
 *
 * ▶▶ THE INVARIANT THAT SHAPES EVERY FUNCTION HERE: a position must be BACKED
 * before the batch carrying it reaches host->step. kv_reserve is the only
 * survivable refusal in the KV layer -- every accessor below it fatals on an
 * unbackable position, because for every caller except this one that IS a bug.
 * So preemption is decided at BATCH-BUILD time, never during a forward: by the
 * time a forward writes K/V the step is already running and there is nothing
 * left to decide. Reservation memo: superpowers/memos/
 * 2026-08-19-m5-admission-reservation-DECISION-OPUS.md.
 */
#include "common.h"

#include "sched.h"

#include "blockpool.h" /* bp_capacity, for the admissibility check at submit */

#include <pthread.h>
#include <stdlib.h> /* getenv, strtol -- STOKE_SCHED_CHUNK */
#include <stdatomic.h>
#include <string.h>

/* ---- one request ------------------------------------------------------
 *
 * ▶ POSITION BOOKKEEPING, STATED ONCE BECAUSE EVERYTHING ELSE READS IT.
 *   n_done  = positions COMPUTED and backed, i.e. the position the next
 *             forward row will occupy.
 *   n_gen   = ids generated so far, recorded in `gen`.
 *   target  = positions that must exist before this request may decode.
 *
 * The identity `n_done == n_prompt + n_gen - 1` holds for any RUNNING request:
 * prefill computes 0..n_prompt-1 and its last row yields id 0, so n_done is
 * n_prompt with n_gen 1; each decode adds one of each.
 *
 * ▶ PREFIX ADOPTION AMENDS THE DERIVATION AND NOT THE IDENTITY. With dedup on,
 * n_done starts at the number of positions found RESIDENT rather than at 0, so
 * prefill computes [adopted, n_prompt) and the positions below `adopted` are
 * shared rather than computed. n_done still reaches n_prompt at the end of the
 * prefill, so everything downstream reads the same. Worth stating because
 * "n_done counts positions this request computed" was true until 2026-08-25 and
 * is now false; what it counts is positions BACKED AND VALID, which is what
 * every use of it actually needs.
 *
 * ▶ AND THE ONE FORMULA THAT MAKES REPLAY FREE. The token at position p is
 * prompt_ids[p] while p < n_prompt and gen[p - n_prompt] after -- so a
 * re-prefill after preemption feeds exactly the recorded ids without a second
 * code path, and "replay, never re-sample" becomes a property of tok_at rather
 * than a rule someone has to remember. */
typedef struct Req Req;
struct Req {
    SchedRequest spec;
    SchedState   state;
    SchedFinish  finish;
    Kv*          kv;
    u32          n_done;
    u32          n_gen;
    u32          target; /* positions to recompute before decoding again */
    u32*         gen;    /* [spec.max_tokens] recorded ids -- REPLAYED     */
    bool         cancel;
    u64          arrival; /* FCFS order, and the youngest-victim key       */

    /* Per-request output ring. Sized to max_tokens (rounded to a power of two)
     * so it CANNOT overflow: a consumer that never polls costs memory, not
     * correctness, and the "what if full" branch stops existing. Head/tail are
     * atomic because M6's server consumes from another thread; at M5 there is
     * one thread and this costs nothing to have already got right. */
    u32*                 ring;
    u32                  ring_mask;
    _Atomic unsigned int ring_head; /* producer: scheduler */
    _Atomic unsigned int ring_tail; /* consumer: sched_poll */

    Req* next; /* intrusive: waiting list, running list, or the all-list */
    Req* all_next;
};

struct Sched {
    KvStore*  store;
    u32       max_ctx;
    SchedHost host;

    Req* waiting_head; /* FIFO; a preempted request is pushed to the FRONT */
    Req* waiting_tail;
    Req* running_head; /* arrival order, oldest first                      */
    Req* prefilling;   /* at most ONE (S1)                                 */
    Req* all;          /* every live request, for lookup by id             */

    u64 next_arrival;
    u32  chunk;   /* SCHED_CHUNK_TOKENS, or STOKE_SCHED_CHUNK */
    u64  last_preempted;
    bool ever_preempted;
    bool dedup;   /* prefix adoption at admission -- see sched.h */

    /* The submit queue: the ONE place a foreign thread touches this object.
     * Plain mutex, per the forward note's instruction not to get clever --
     * arrivals are rare relative to steps and TSan must stay quiet. */
    pthread_mutex_t mu;
    Req*            submitted; /* singly linked, newest first */

    /* Scratch for one step's rows, grown on demand and never shrunk. The step
     * loop must not allocate: it runs thousands of times a second in the sim
     * and once per token in production. */
    Kv** row_kv;
    u32* row_tok;
    u32* row_pos;
    u64* row_req;
    u32* row_out;
    u32  row_cap;

    SchedStats st;
};

/* ---- small helpers ----------------------------------------------------- */

static u32 blocks_for_tokens(u32 tokens) { return kv_blocks_for_ctx(tokens); }

static u32 tok_at(const Req* r, u32 pos) {
    if (pos < r->spec.n_prompt) return r->spec.prompt_ids[pos];
    return r->gen[pos - r->spec.n_prompt];
}

static u32 pow2_at_least(u32 n) {
    u32 p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void rows_ensure(Sched* s, u32 n) {
    if (n <= s->row_cap) return;
    s->row_kv = (Kv**)xrealloc(s->row_kv, (size_t)n * sizeof(Kv*));
    s->row_tok = (u32*)xrealloc(s->row_tok, (size_t)n * sizeof(u32));
    s->row_pos = (u32*)xrealloc(s->row_pos, (size_t)n * sizeof(u32));
    s->row_req = (u64*)xrealloc(s->row_req, (size_t)n * sizeof(u64));
    s->row_out = (u32*)xrealloc(s->row_out, (size_t)n * sizeof(u32));
    s->row_cap = n;
}

static void emit(Req* r, u32 id) {
    unsigned int h = atomic_load_explicit(&r->ring_head, memory_order_relaxed);
    r->ring[h & r->ring_mask] = id;
    atomic_store_explicit(&r->ring_head, h + 1u, memory_order_release);
}

static Req* find(Sched* s, u64 id) {
    for (Req* r = s->all; r; r = r->all_next)
        if (r->spec.id == id) return r;
    return NULL;
}

/* ---- list surgery ------------------------------------------------------
 * Explicit and unshared on purpose. A generic list would hide which of the
 * three memberships a request is in, and "in two lists at once" is the failure
 * this scheduler could have that no output would show. */

static void waiting_push_back(Sched* s, Req* r) {
    r->next = NULL;
    if (s->waiting_tail) s->waiting_tail->next = r;
    else s->waiting_head = r;
    s->waiting_tail = r;
    r->state = SCHED_WAITING;
}

static void waiting_push_front(Sched* s, Req* r) {
    r->next = s->waiting_head;
    s->waiting_head = r;
    if (!s->waiting_tail) s->waiting_tail = r;
    r->state = SCHED_WAITING;
}

static Req* waiting_pop_front(Sched* s) {
    Req* r = s->waiting_head;
    if (!r) return NULL;
    s->waiting_head = r->next;
    if (!s->waiting_head) s->waiting_tail = NULL;
    r->next = NULL;
    return r;
}

/* RUNNING is kept in arrival order, oldest at the head -- so "youngest" is the
 * tail and "oldest" is the head, and both of S3/S4's questions are answered by
 * a pointer instead of a scan. */
static void running_push(Sched* s, Req* r) {
    r->next = NULL;
    r->state = SCHED_RUNNING;
    if (!s->running_head) { s->running_head = r; return; }
    Req* p = s->running_head;
    while (p->next) p = p->next;
    p->next = r;
}

static void running_remove(Sched* s, Req* r) {
    Req** pp = &s->running_head;
    while (*pp && *pp != r) pp = &(*pp)->next;
    if (*pp) *pp = r->next;
    r->next = NULL;
}

static u32 running_count(const Sched* s) {
    u32 n = 0;
    for (const Req* r = s->running_head; r; r = r->next) n++;
    return n;
}

/* ---- lifecycle --------------------------------------------------------- */

Sched* sched_open(KvStore* store, u32 max_ctx, const SchedHost* host,
                  bool dedup_prefix) {
    if (!store) fatal("sched: sched_open -- store must be non-NULL");
    if (max_ctx == 0) fatal("sched: sched_open -- max_ctx must be > 0");
    if (!host || !host->step || !host->now_ms) {
        fatal("sched: sched_open -- host must supply both step and now_ms; a "
              "scheduler with no way to step is a caller error");
    }
#ifdef STOKE_MUT_PREEMPT
    /* A MUTANT ANNOUNCES ITSELF. The whole point of this build is to be WRONG,
     * so the one thing it must never do is look like the shipping binary in a
     * log somebody reads later. */
    fprintf(stderr, "!! STOKE_MUT_PREEMPT BUILD -- preempt replay is "
                    "DELIBERATELY off by one. Never ship this binary.\n");
#endif
#ifdef STOKE_MUT_ADOPT_KV
    fprintf(stderr, "!! STOKE_MUT_ADOPT_KV BUILD -- the adopted prefix's K/V is "
                    "DELIBERATELY zeroed. Never ship this binary.\n");
#endif
    Sched* s = (Sched*)xcalloc(1, sizeof(Sched));
    /* STOKE_SCHED_CHUNK overrides the locked 128, for MEASURING the co-run
     * tradeoff rather than for shipping a different value. A chunk amortises
     * the weight stream across its rows (throughput) and makes the step it
     * rides in longer (ITL) -- gate item [2] wants it large and gate item [3]
     * wants it small, and the forward note's attribution ladder says to try 64
     * if the co-run blows the ITL budget. An env var and not a rebuild, so both
     * arms come from ONE binary in ONE sitting, exactly as every other A/B on
     * this box has to.
     *
     * S-ALIGN still binds: a value that is not a whole number of KV blocks is
     * REFUSED rather than rounded, because a chunk ending mid-block leaves that
     * block unregistrable by the prefix cache and rounding would silently
     * measure a different width than the one asked for. */
    s->chunk = SCHED_CHUNK_TOKENS;
    {
        const char* e = getenv("STOKE_SCHED_CHUNK");
        if (e && e[0]) {
            long v = strtol(e, NULL, 10);
            if (v <= 0 || (u32)v % kv_block_tokens() != 0u) {
                fatal("sched: STOKE_SCHED_CHUNK=%s -- must be positive and a "
                      "multiple of the KV block size (%u); a chunk must end on "
                      "a block boundary (S-ALIGN)",
                      e, kv_block_tokens());
            }
            s->chunk = (u32)v;
        }
    }
    s->store = store;
    s->max_ctx = max_ctx;
    s->host = *host;
    s->dedup = dedup_prefix;
    pthread_mutex_init(&s->mu, NULL);
    return s;
}

static void req_free(Req* r) {
    if (!r) return;
    kv_close(r->kv);
    free(r->gen);
    free(r->ring);
    free(r);
}

void sched_close(Sched* s) {
    if (!s) return;
    Req* r = s->all;
    while (r) { Req* n = r->all_next; req_free(r); r = n; }
    pthread_mutex_lock(&s->mu);
    r = s->submitted;
    while (r) { Req* n = r->all_next; req_free(r); r = n; }
    pthread_mutex_unlock(&s->mu);
    pthread_mutex_destroy(&s->mu);
    free(s->row_kv); free(s->row_tok); free(s->row_pos);
    free(s->row_req); free(s->row_out);
    free(s);
}

bool sched_submit(Sched* s, const SchedRequest* req) {
    if (!req || req->n_prompt == 0 || req->max_tokens == 0) return false;
    if ((u64)req->n_prompt + req->max_tokens > (u64)s->max_ctx) return false;

    /* ▶ REFUSED AT THE DOOR IF IT COULD NEVER CLEAR THE WATERMARK ON AN EMPTY
     * POOL, and this is a LIVELOCK fix rather than tidiness. Admission needs
     * free >= blocks(prompt) + W with W's floor of 8, so a prompt bigger than
     * capacity - 8 is never admissible however long it waits -- and the step
     * loop would then return 0 rows forever while sched_pending stayed true,
     * which a driver reads as "busy" and spins on. A request that cannot run is
     * information the submitter can act on; a scheduler that never says so is
     * not. Same reasoning as kv_open_in_partial refusing an unreachable
     * ceiling: unreachable-and-loud beats unreachable-and-silent. */
    u32 cap = bp_capacity(kv_store_pool(s->store));
    if ((u64)blocks_for_tokens(req->n_prompt) + SCHED_W_FLOOR > (u64)cap)
        return false;
    if (blocks_for_tokens(req->n_prompt + req->max_tokens) > cap) return false;

    /* Duplicate ids are refused rather than tolerated: every lookup here is by
     * id, so two live requests sharing one would make cancel, poll and progress
     * all silently address whichever came first. */
    if (find(s, req->id)) return false;
    pthread_mutex_lock(&s->mu);
    for (Req* q = s->submitted; q; q = q->all_next) {
        if (q->spec.id == req->id) { pthread_mutex_unlock(&s->mu); return false; }
    }
    Req* r = (Req*)xcalloc(1, sizeof(Req));
    r->spec = *req;
    r->state = SCHED_WAITING;
    r->gen = (u32*)xcalloc(req->max_tokens, sizeof(u32));
    r->ring_mask = pow2_at_least(req->max_tokens) - 1u;
    r->ring = (u32*)xcalloc((size_t)r->ring_mask + 1u, sizeof(u32));
    r->all_next = s->submitted;
    s->submitted = r;
    pthread_mutex_unlock(&s->mu);
    return true;
}

bool sched_cancel(Sched* s, u64 id) {
    Req* r = find(s, id);
    if (r && r->state != SCHED_FINISHED) { r->cancel = true; return true; }
    pthread_mutex_lock(&s->mu);
    for (Req* q = s->submitted; q; q = q->all_next) {
        if (q->spec.id == id) { q->cancel = true; pthread_mutex_unlock(&s->mu); return true; }
    }
    pthread_mutex_unlock(&s->mu);
    return false;
}

/* ---- the step ---------------------------------------------------------- */

static void drain_submits(Sched* s) {
    pthread_mutex_lock(&s->mu);
    Req* q = s->submitted;
    s->submitted = NULL;
    pthread_mutex_unlock(&s->mu);

    /* The submit stack is newest-first; reverse it so arrival order survives
     * into WAITING. FCFS that silently depends on how a lock-free-ish stack
     * happens to be ordered is FCFS by accident. */
    Req* rev = NULL;
    while (q) { Req* n = q->all_next; q->all_next = rev; rev = q; q = n; }
    while (rev) {
        Req* n = rev->all_next;
        rev->arrival = s->next_arrival++;
        rev->all_next = s->all;
        s->all = rev;
        waiting_push_back(s, rev);
        rev = n;
    }
}

/* Frees a request's blocks and takes it out of every schedulable list. The
 * request object itself survives until sched_close so a late sched_poll can
 * still read its finish reason and drain its ring (S8's "teardown frees every
 * block" is about BLOCKS, not about forgetting the request existed). */
static void retire(Sched* s, Req* r, SchedFinish why) {
    if (r->state == SCHED_RUNNING) running_remove(s, r);
    else if (r->state == SCHED_PREFILLING) s->prefilling = NULL;
    else if (r->state == SCHED_WAITING) {
        Req** pp = &s->waiting_head;
        Req*  prev = NULL;
        while (*pp && *pp != r) { prev = *pp; pp = &(*pp)->next; }
        if (*pp) {
            *pp = r->next;
            if (s->waiting_tail == r) s->waiting_tail = prev;
        }
        r->next = NULL;
    }
    kv_close(r->kv);
    r->kv = NULL;
    r->state = SCHED_FINISHED;
    r->finish = why;
    s->st.finished++;
}

static void retire_cancelled(Sched* s) {
    for (Req* r = s->all; r; r = r->all_next) {
        if (r->cancel && r->state != SCHED_FINISHED) {
            /* Counted BEFORE retire() clears the state. S5 quantifies over
             * RUNNABLE decodes, and a request cancelled here stops being one
             * before any row is built -- so the sim needs to tell this apart
             * from a decode that was dropped. */
            if (r->state == SCHED_RUNNING) s->st.cancelled_running++;
            retire(s, r, SCHED_FINISH_CANCELLED);
        }
    }
}

/* PREEMPTION: victim is the YOUNGEST RUNNING, which with push-front readmission
 * gives starvation-freedom by induction -- the oldest request is never chosen,
 * so it always completes, and the argument then applies to the rest.
 *
 * Its blocks go back immediately and its recorded ids do not: `gen` is what a
 * re-prefill REPLAYS, which is why preempt-replay is lossless under greedy and
 * stays lossless under M6's seeded samplers. */
static void preempt(Sched* s, Req* victim) {
    if (victim->state == SCHED_RUNNING) running_remove(s, victim);
    else if (victim->state == SCHED_PREFILLING) s->prefilling = NULL;
    kv_close(victim->kv);
    victim->kv = NULL;
    victim->n_done = 0;
    /* Recompute prompt + everything generated except the last id, whose logits
     * the final replayed row reproduces. See Req's header. */
#ifdef STOKE_MUT_PREEMPT
    /* ▶▶ A DELIBERATE OFF-BY-ONE, BUILT ONLY VIA EXTRA_CFLAGS, AND IT EXISTS SO
     * GATE [6] CAN PROVE IT COULD FAIL. Replaying one row too few makes the
     * final replayed row reproduce the logits of the WRONG position, so the
     * stream continues fluently and is merely wrong -- the failure class kv.h
     * names as the dangerous one. A byte-compare that cannot distinguish this
     * from correct behaviour is testing nothing, and on this project a fixture
     * that could not tell has happened before ([[fixture-can-be-blind]]).
     *
     * It can never be in a shipping binary: nothing defines this macro, the
     * default build does not, and the banner below makes a mutant announce
     * itself on every run. */
    victim->target = victim->spec.n_prompt + victim->n_gen - 2u;
    if (victim->n_gen < 2u) victim->target = victim->spec.n_prompt;
#else
    victim->target = victim->spec.n_prompt + victim->n_gen - 1u;
#endif
    if (victim->target == 0) victim->target = victim->spec.n_prompt;
    waiting_push_front(s, victim);
    s->last_preempted = victim->spec.id;
    s->ever_preempted = true;
    s->st.preemptions++;
}

/* ▶▶ THE VICTIM MUST BE AT LEAST AS YOUNG AS THE ROW BEING BACKED, AND THAT IS
 * A MEMORY-SAFETY CONSTRAINT, NOT A POLICY ONE.
 *
 * Rows are built oldest-first, so by the time request `r` needs blocks, every
 * request older than it ALREADY HAS A ROW IN THIS BATCH pointing at its Kv.
 * Preempting one of those closes a handle the batch still references -- a
 * use-after-free that the byte-compare could never see, because the batch has
 * not been handed to the host yet and the corrupted read happens inside the
 * forward. The plain "youngest RUNNING" rule permits exactly that as soon as
 * two preemptions in one row-build walk the tail past `r`.
 *
 * So: youngest RUNNING with arrival >= r's. That is still "preempt-youngest"
 * for every case the design describes -- the excluded ones are strictly older
 * than a request that is itself failing to allocate, i.e. cases where the
 * design's own starvation argument says not to preempt anyway.
 *
 * ▶ AND THEN THE PREFILL, WHICH IS NOT A DESIGN CHANGE. If no eligible RUNNING
 * victim exists, the in-flight PREFILL yields instead: its rows are built after
 * every decode row, so nothing references its handle yet. That is S5's rule
 * ("a chunk co-runs, it never displaces a decode") applied to blocks rather
 * than to step slots -- a chunk that would starve a runnable decode gives way,
 * which is the same principle and not a new one. */
static bool preempt_for(Sched* s, const Req* r) {
    Req* victim = NULL;
    for (Req* q = s->running_head; q; q = q->next) {
        if (q->arrival >= r->arrival) victim = q; /* youngest eligible */
    }
    if (victim) { preempt(s, victim); return true; }
    if (s->prefilling && s->prefilling != r) {
        preempt(s, s->prefilling);
        s->st.prefill_yields++;
        return true;
    }
    return false;
}

static void admit_if_possible(Sched* s) {
    if (s->prefilling || !s->waiting_head) return;

    Req* r = s->waiting_head;
    u32  need = blocks_for_tokens(r->target ? r->target : r->spec.n_prompt);
    u32  w = SCHED_W_PER_RUNNING * running_count(s);
    if (w < SCHED_W_FLOOR) w = SCHED_W_FLOOR;
    /* THE WATERMARK LIVES HERE, not in the pool. A pool-level reserve cannot be
     * honoured once sequences grow past their own, so this is a policy checked
     * against the live free count, with preemption as the backstop. */
    if (kv_store_free_blocks(s->store) < need + w) return;

    Kv* kv = kv_open_in_partial(s->store,
                                r->spec.n_prompt + r->spec.max_tokens);
    if (!kv) return;

    (void)waiting_pop_front(s);
    r->kv = kv;
    r->state = SCHED_PREFILLING;
    if (r->target == 0) r->target = r->spec.n_prompt;

    /* ▶▶ PREFIX ADOPTION, AND IT BELONGS EXACTLY HERE.
     *
     * kv_adopt_prefix is ADMISSION-TIME ONLY -- it splices shared blocks in from
     * ordinal 0 and returns 0 for a sequence that has already been written -- so
     * the moment between opening the Kv and building the first row is the only
     * one at which it can fire. Starting n_done at what it returns is the whole
     * of the change downstream: the prefill loop already takes
     * `chunk = target - n_done`, so adopted positions are skipped with nothing
     * else to adjust, and `start` is a whole number of blocks so S-ALIGN still
     * holds by construction.
     *
     * ▶ A PREEMPTED REQUEST RE-ADOPTS, and that falls out rather than needing a
     * second path: preempt() closes the victim's Kv and resets n_done to 0, so
     * readmission arrives here with a FRESH sequence -- which is the one state
     * kv_adopt_prefix accepts. It re-adopts only what is still resident (a
     * donor may have retired meanwhile), and recomputing is always correct.
     *
     * ▶ AND IT NEVER SKIPS THE WHOLE PROMPT: kv_adopt_prefix gives back the last
     * block of an exact multiple, so adopted <= n_prompt - 1 <= target - 1 and
     * there is always at least one row left to compute. A request cannot be
     * admitted into a state where it can never complete its prefill.
     *
     * ▶ THE WATERMARK ABOVE IS DELIBERATELY NOT ADJUSTED. It is checked before
     * the Kv exists, so what adoption will find is not yet knowable; admitting
     * on the un-adopted size is conservative in the only direction that matters.
     *
     * ▶ THE SCHEDULER DOES NOT TAKE ADOPTION'S WORD FOR IT -- model.c's
     * generate_batch does not either, and for the reason kv.h names: skipping
     * positions that were REPORTED resident but are not BACKED is the one
     * failure on this path that leaves the run alive and merely wrong. Attention
     * reads whatever the blocks hold, the stream stays fluent, and nothing
     * crashes. A measured mutation on the batch path confirmed a fixture-level
     * token comparison cannot see it. */
    s->st.admitted_prompt_positions += r->spec.n_prompt;
    if (s->dedup) {
        u32 adopted = kv_adopt_prefix(kv, r->spec.prompt_ids, r->spec.n_prompt);
        if (kv_len(kv) != adopted) {
            fatal("sched: admit -- request %llu adoption reported %u resident "
                  "positions but the sequence is backed to %u",
                  (unsigned long long)r->spec.id, adopted, kv_len(kv));
        }
        r->n_done = adopted;
        s->st.adopted_positions += adopted;
#ifdef STOKE_MUT_ADOPT_KV
        /* ▶▶ A DELIBERATELY WRONG ADOPTION, BUILT ONLY VIA EXTRA_CFLAGS, AND IT
         * ASKS THE ONE QUESTION EVERY DEDUP GATE RESTS ON: if the adopted blocks
         * held the WRONG K/V, would a byte-compare of the id streams notice?
         *
         * A SAME from this arm means the comparison is powerless over adopted-KV
         * correctness and the gate around it is testing nothing -- which is not
         * hypothetical here: the `skip16` mutant (16 backed-but-unwritten
         * positions) was MEASURED SAME on 2026-08-25, because K = 0 scores
         * exp(0) against a thousand real keys and greedy argmax does not resolve
         * it. Zeroing the WHOLE adopted prefix is the version with power.
         *
         * It writes through the shared blocks on purpose: adoption is by
         * refcount, so this damages the donor too, which is exactly the shape of
         * the failure a preemption-plus-re-adoption path could have.
         *
         * It can never be in a shipping binary: nothing defines this macro, the
         * default build does not, and sched_open's banner makes a mutant
         * announce itself on every run. */
        if (adopted > 0) {
            u32  n_l = kv_n_layer(kv), n_h = kv_n_head_kv(kv);
            u32  hd = kv_head_dim(kv);
            f32* z = (f32*)xcalloc(hd, sizeof(f32));
            for (u32 l = 0; l < n_l; l++)
                for (u32 h = 0; h < n_h; h++)
                    for (u32 p = 0; p < adopted; p++)
                        kv_write(kv, l, h, p, z, z);
            free(z);
        }
#endif
    }
    s->prefilling = r;
}

/* Backs one decode row's position, preempting until it fits. Returns false when
 * `r` ITSELF was the victim (its row must then be skipped, not built) or when
 * nothing could be freed at all. The caller distinguishes the two by r->kv. */
static bool back_decode_row(Sched* s, Req* r) {
    while (r->kv && !kv_reserve(r->kv, r->n_done + 1u)) {
        if (!preempt_for(s, r)) return false;
    }
    return r->kv != NULL;
}

/* Backs one prefill row. NO preemption here: every decode row of this step is
 * already built and holds a Kv pointer, so there is nothing left that is safe
 * to free. A prefill that cannot grow simply takes a shorter chunk, or none,
 * and tries again next step -- which costs latency and never correctness. */
static bool back_prefill_row(Sched* s, Req* r, u32 pos) {
    (void)s;
    return r->kv && kv_reserve(r->kv, pos + 1u);
}

u32 sched_step(Sched* s) {
    drain_submits(s);
    retire_cancelled(s);
    admit_if_possible(s);

    u32 n_running = running_count(s);
    u32 chunk = 0;
    Req* pf = s->prefilling;
    if (pf) {
        chunk = pf->target - pf->n_done;
        if (chunk > s->chunk) chunk = s->chunk;
    }
    rows_ensure(s, n_running + chunk);

    /* ---- decode rows: EVERY runnable decode, every step (S5). A chunk
     * co-runs; it never displaces one. Built first so [0, n_decode) is the
     * decode span by construction. */
    u32 n = 0;
    Req* r = s->running_head;
    while (r) {
        Req* nxt = r->next; /* r may be preempted (and relinked) below */
        bool backed = back_decode_row(s, r);
        if (!backed && r->kv != NULL) {
            /* Nothing was eligible to preempt and `r` still holds its handle,
             * so `r` is the only block-holder that could have yielded. Its own
             * ceiling fits the pool by construction (kv_open_in_partial refuses
             * a larger one), so a sole holder can always grow: unreachable. */
            fatal("sched: could not back position %u for request %llu with "
                  "nothing eligible to preempt -- see preempt_for",
                  r->n_done, (unsigned long long)r->spec.id);
        }
        if (backed) {
            s->row_kv[n] = r->kv;
            s->row_pos[n] = r->n_done;
            s->row_tok[n] = tok_at(r, r->n_done);
            s->row_req[n] = r->spec.id;
            n++;
        }
        r = nxt;
    }
    u32 n_decode = n;

    /* ---- the prefill chunk. Re-read s->prefilling: a preemption above cannot
     * touch it (victims are RUNNING), but reading it once here rather than
     * trusting the earlier copy is what keeps that true if the victim rule ever
     * widens. */
    pf = s->prefilling;
    bool completes = false;
    if (pf && chunk > 0) {
        for (u32 i = 0; i < chunk; i++) {
            u32 pos = pf->n_done + i;
            if (!back_prefill_row(s, pf, pos)) { chunk = i; break; }
            s->row_kv[n] = pf->kv;
            s->row_pos[n] = pos;
            s->row_tok[n] = tok_at(pf, pos);
            s->row_req[n] = pf->spec.id;
            n++;
        }
        completes = (pf->n_done + chunk == pf->target) && (pf->n_gen == 0);
    }

    /* ---- the high-water mark, taken HERE and nowhere else. Every row of this
     * step is now backed and every preemption that backing forced has happened;
     * nothing has retired yet. That is the instant of maximum demand, and it is
     * the one a driver polling at step boundaries cannot see. See SchedStats. */
    {
        u32 cap  = bp_capacity(kv_store_pool(s->store));
        u32 used = cap - kv_store_free_blocks(s->store);
        if (used > s->st.peak_blocks_used) s->st.peak_blocks_used = used;
    }

    if (n == 0) return 0;

    SchedStep in;
    in.kv = s->row_kv;
    in.token_ids = s->row_tok;
    in.positions = s->row_pos;
    in.req_ids = s->row_req;
    in.n_rows = n;
    in.n_decode = n_decode;
    in.prefill_completes = completes;
    s->host.step(s->host.ctx, &in, s->row_out);

    /* ---- sample the decodes. Walk the SAME order the rows were built in. */
    u32 k = 0;
    r = s->running_head;
    while (r && k < n_decode) {
        Req* nxt = r->next;
        u32  id = s->row_out[k];
        r->n_done++;
        if (r->n_gen < r->spec.max_tokens) r->gen[r->n_gen++] = id;
        emit(r, id);
        if (r->spec.eos_id >= 0 && (i64)id == r->spec.eos_id) {
            retire(s, r, SCHED_FINISH_EOS);
        } else if (r->n_gen >= r->spec.max_tokens) {
            retire(s, r, SCHED_FINISH_MAX_TOKENS);
        }
        k++;
        r = nxt;
    }
    s->st.decode_rows += n_decode;

    /* ---- advance the prefill. */
    if (pf && chunk > 0) {
        pf->n_done += chunk;
        s->st.prefill_chunks++;
        /* S7 (S-ALIGN): a chunk must end on a block boundary, or the block it
         * stopped inside cannot be registered by the prefix cache until
         * something else fills it. Ending the prompt is the one allowed
         * exception. Chunk 128 over block 16 satisfies this by construction --
         * this is what notices if either constant moves without the other. */
        if (pf->n_done % kv_block_tokens() != 0u && pf->n_done != pf->target)
            s->st.chunk_misaligned++;
        if (pf->n_done >= pf->target) {
            s->prefilling = NULL;
            if (completes) {
                /* First prefill: its last row yields the first id. */
                u32 id = s->row_out[n_decode];
                pf->gen[pf->n_gen++] = id;
                emit(pf, id);
                if (pf->spec.eos_id >= 0 && (i64)id == pf->spec.eos_id) {
                    pf->state = SCHED_RUNNING; /* retire() expects a list state */
                    running_push(s, pf);
                    retire(s, pf, SCHED_FINISH_EOS);
                } else if (pf->n_gen >= pf->spec.max_tokens) {
                    pf->state = SCHED_RUNNING;
                    running_push(s, pf);
                    retire(s, pf, SCHED_FINISH_MAX_TOKENS);
                } else {
                    running_push(s, pf);
                }
            } else {
                /* A REPLAY completing: the ids already exist and are NOT
                 * re-sampled. That is the whole of the replay rule. */
                running_push(s, pf);
            }
        }
    }

    s->st.steps++;
    s->st.free_blocks = kv_store_free_blocks(s->store);
    return n;
}

bool sched_pending(const Sched* s) {
    if (s->waiting_head || s->prefilling || s->running_head) return true;
    bool any;
    pthread_mutex_lock(&((Sched*)s)->mu);
    any = (s->submitted != NULL);
    pthread_mutex_unlock(&((Sched*)s)->mu);
    return any;
}

u32 sched_poll(Sched* s, u64 id, u32* out, u32 cap, SchedFinish* finish) {
    Req* r = find(s, id);
    if (finish) *finish = SCHED_FINISH_NONE;
    if (!r) return 0;
    unsigned int h = atomic_load_explicit(&r->ring_head, memory_order_acquire);
    unsigned int t = atomic_load_explicit(&r->ring_tail, memory_order_relaxed);
    u32 got = 0;
    while (t != h && got < cap) out[got++] = r->ring[t++ & r->ring_mask];
    atomic_store_explicit(&r->ring_tail, t, memory_order_release);
    /* Reported only once the ring is DRAINED, so a consumer that stops at the
     * first non-NONE cannot lose a trailing id. */
    if (finish && r->state == SCHED_FINISHED && t == h) *finish = r->finish;
    return got;
}

void sched_stats(const Sched* s, SchedStats* out) {
    *out = s->st;
    out->n_waiting = 0;
    for (const Req* r = s->waiting_head; r; r = r->next) out->n_waiting++;
    out->n_running = running_count(s);
    out->n_prefilling = s->prefilling ? 1u : 0u;
    out->free_blocks = kv_store_free_blocks(s->store);
}

u64 sched_oldest_waiting(const Sched* s, bool* ok) {
    if (!s->waiting_head) { if (ok) *ok = false; return 0; }
    if (ok) *ok = true;
    return s->waiting_head->spec.id;
}

u64 sched_youngest_running(const Sched* s, bool* ok) {
    const Req* y = NULL;
    for (const Req* r = s->running_head; r; r = r->next) y = r;
    if (!y) { if (ok) *ok = false; return 0; }
    if (ok) *ok = true;
    return y->spec.id;
}

u64 sched_last_preempted(const Sched* s, bool* ok) {
    if (ok) *ok = s->ever_preempted;
    return s->last_preempted;
}

u32 sched_progress(const Sched* s, u64 id) {
    Req* r = find((Sched*)s, id);
    return r ? r->n_done : 0u;
}

bool sched_is_live(const Sched* s, u64 id) {
    Req* r = find((Sched*)s, id);
    if (r) return r->state != SCHED_FINISHED;
    /* Not on the lists yet. It may still be sitting in the submit queue, which
     * find() deliberately does not walk -- and a request cancelled from there
     * is retired without ever reaching the lists, so this half is what keeps a
     * caller's per-request state from outliving it. */
    bool queued = false;
    pthread_mutex_lock(&((Sched*)s)->mu);
    for (Req* q = s->submitted; q; q = q->all_next) {
        if (q->spec.id == id) { queued = !q->cancel; break; }
    }
    pthread_mutex_unlock(&((Sched*)s)->mu);
    return queued;
}
