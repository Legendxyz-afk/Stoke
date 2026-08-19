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

#include <pthread.h>
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

Sched* sched_open(KvStore* store, u32 max_ctx, const SchedHost* host) {
    if (!store) fatal("sched: sched_open -- store must be non-NULL");
    if (max_ctx == 0) fatal("sched: sched_open -- max_ctx must be > 0");
    if (!host || !host->step || !host->now_ms) {
        fatal("sched: sched_open -- host must supply both step and now_ms; a "
              "scheduler with no way to step is a caller error");
    }
    Sched* s = (Sched*)xcalloc(1, sizeof(Sched));
    s->store = store;
    s->max_ctx = max_ctx;
    s->host = *host;
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
        if (r->cancel && r->state != SCHED_FINISHED)
            retire(s, r, SCHED_FINISH_CANCELLED);
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
    victim->target = victim->spec.n_prompt + victim->n_gen - 1u;
    if (victim->target == 0) victim->target = victim->spec.n_prompt;
    waiting_push_front(s, victim);
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
    if (s->prefilling && s->prefilling != r) { preempt(s, s->prefilling); return true; }
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
        if (chunk > SCHED_CHUNK_TOKENS) chunk = SCHED_CHUNK_TOKENS;
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

u32 sched_progress(const Sched* s, u64 id) {
    Req* r = find((Sched*)s, id);
    return r ? r->n_done : 0u;
}
