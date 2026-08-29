/* src/sched.h -- M5 continuous batching: the scheduler's interface.
 *
 * DESIGN AUTHORITY: the forward note's section M5 (LOCKED -- states, step loop,
 * admission watermark, preempt-youngest/push-front, replay-recorded-ids, chunk
 * 128) and docs/superpowers/design-notes/2026-08-19-m5-s0-sim-architecture.md
 * (the substitution boundary and this seam). Neither is re-opened here; this
 * file is what they imply.
 *
 * ▶▶ THE ONE THING THIS HEADER EXISTS TO ENFORCE. The scheduler must be
 * drivable by the simulation harness WITHOUT the harness reimplementing it --
 * otherwise a green sim proves the TEST's scheduler is correct. So the two
 * things a sim must replace, and ONLY those two, arrive through SchedHost: the
 * model step and the clock. The block pool does NOT: the sim runs the REAL
 * BlockPool, because M4's invariants I1-I12 are statements about that allocator
 * and against a fake one they assert nothing.
 *
 * Everything else -- states, ordering, admission, preemption, retirement -- is
 * inside sched.c in both modes. That is the whole point.
 */
#ifndef STOKE_SCHED_H
#define STOKE_SCHED_H

#include "common.h"

#include <stdbool.h>

#include "kv.h"

typedef struct Sched Sched;

/* ---- what a step is ---------------------------------------------------
 * ONE fused forward: every runnable decode as one row each, plus at most one
 * chunk of the single active prefill. The prefill chunk RIDES THE SAME OP WALK
 * -- it is extra rows, not a second pass -- and that co-execution IS the
 * design, not an optimization of it.
 *
 * ▶ WHY THE HOST TAKES THE WHOLE BATCH AND NOT ONE SEQUENCE. A per-sequence
 * hook would let the sim pass while the real host quietly ran N separate
 * forwards, which is precisely the failure the co-execution requirement exists
 * to prevent. The shape of this struct is the requirement, made unavoidable. */
typedef struct {
    Kv**       kv;        /* [n_rows] one handle per ROW; a prefill chunk's
                           * rows share ONE handle, being one sequence      */
    const u32* token_ids; /* [n_rows]                                       */
    const u32* positions; /* [n_rows]                                       */
    const u64* req_ids;   /* [n_rows] whose request each row belongs to --
                           * the sim's fake model hashes it, and it is what
                           * makes preempt-replay checkable                 */
    u32        n_rows;
    /* Rows [0, n_decode) are decodes and each yields a sampled id. Rows
     * [n_decode, n_rows) are the prefill chunk; only its LAST row yields one,
     * and only when that chunk completes the prompt. Split rather than flagged
     * per row because the step loop builds them in this order by construction,
     * and a flag array is a second source of truth about the same fact. */
    u32  n_decode;
    bool prefill_completes;
} SchedStep;

/* The two substitutions, and deliberately not three. Each additional hook is
 * another thing the sim can get right while the real host gets wrong. */
typedef struct {
    /* Runs `in` and writes one id per row that yields one: out_ids[0..n_decode)
     * for the decodes, and out_ids[n_decode] for the prefill's last row when
     * `prefill_completes`. Real: model_forward_batch + sampling. Sim: a hash of
     * (request id, position) -- instant, deterministic, and dependent on
     * IDENTITY so preempt-replay losslessness cannot hold for the wrong
     * reason. */
    void (*step)(void* ctx, const SchedStep* in, u32* out_ids);
    /* Monotonic milliseconds. Real: now_ms(). Sim: the virtual clock, advanced
     * by the step-cost model. Behind the seam rather than compiled out because
     * ONE sim configuration runs the fake model under the REAL clock, to price
     * the scheduler's own bookkeeping against the forward note's <2%-of-a-step
     * budget. */
    double (*now_ms)(void* ctx);
    void* ctx;
} SchedHost;

/* ---- requests ---------------------------------------------------------- */

typedef enum {
    SCHED_WAITING = 0, /* FIFO arrival queue                                */
    SCHED_PREFILLING,  /* at most ONE request holds this at a time (S1)     */
    SCHED_RUNNING,     /* in the decode set                                 */
    SCHED_FINISHED
} SchedState;

typedef enum {
    SCHED_FINISH_NONE = 0,
    SCHED_FINISH_EOS,
    SCHED_FINISH_MAX_TOKENS,
    SCHED_FINISH_CANCELLED
} SchedFinish;

/* Submitted work. `prompt_ids` is BORROWED and must outlive the request -- the
 * scheduler never copies a prompt, because a preempted request re-prefills from
 * it and copying would double the memory the pool is already rationing. */
typedef struct {
    u64        id;         /* caller-assigned; unique among LIVE requests    */
    const u32* prompt_ids;
    u32        n_prompt;   /* > 0                                           */
    u32        max_tokens; /* > 0                                           */
    i64        eos_id;     /* -1 for none                                   */
} SchedRequest;

/* ---- lifecycle --------------------------------------------------------- */

/* `store` is BORROWED and shared: the scheduler draws every sequence's blocks
 * from it, which is the entire point of paged KV. `max_ctx` bounds any one
 * sequence. Fatal on a NULL host or either callback missing -- a scheduler with
 * no way to step is a caller error, not a runtime condition.
 *
 * ▶ ADMITS THROUGH kv_open_in_partial, NOT kv_open_in. A scheduler sequence
 * reserves nothing and only raises its own ceiling; the decode headroom `W` is
 * enforced HERE, against the live free count, and pool exhaustion is recovered
 * by preemption rather than prevented by reservation. Under full reservation an
 * admitted sequence can always grow, which makes preemption -- and gate item
 * [6], the memory-pressure trace, S4 and S6 -- unreachable. Decision:
 * superpowers/memos/2026-08-19-m5-admission-reservation-DECISION-OPUS.md.
 *
 * ▶ AND THE OBLIGATION THAT FOLLOWS: kv_reserve is the only survivable refusal;
 * every other KV accessor fatals on an unbackable position. So this file must
 * prove every row's position is backed BEFORE handing the batch to host->step.
 * Preemption is decided before the forward -- by the time a forward writes K/V,
 * the step is already running.
 *
 * ▶ `dedup_prefix` TURNS ON PREFIX ADOPTION AT ADMISSION (kv.h's
 * kv_adopt_prefix): a request whose leading whole blocks are already resident in
 * the pool shares them by refcount and starts its prefill past them. This is the
 * same capability `stoke batch --dedup` has had since M4 gate item [3]; M5's
 * serving driver is a NEW CALLER of it, which is why it is gated again rather
 * than inherited (memo 2026-08-24-m5-gate4-s3-status.md).
 *
 * A PARAMETER RATHER THAN A DEFAULT, in either direction: on, it changes how
 * much work a prompt costs and therefore every timing this scheduler produces;
 * off, gate item [4] is measured against an engine (llama-server, whose
 * slot-prompt-similarity and prompt cache are ON by default) that is doing it.
 * Every caller states which of those two runs it is. */
Sched* sched_open(KvStore* store, u32 max_ctx, const SchedHost* host,
                  bool dedup_prefix);
void   sched_close(Sched* s);

/* Queues a request. Returns false (and queues nothing) if `id` is already live,
 * `n_prompt` or `max_tokens` is 0, or prompt+max_tokens exceeds max_ctx.
 *
 * Thread-safe: the submit queue is the ONE place a foreign thread touches the
 * scheduler, and it is a plain mutex because arrivals are rare relative to
 * steps. The forward note's instruction not to get clever here is deliberate,
 * and TSan coming up clean is the reason. */
bool sched_submit(Sched* s, const SchedRequest* req);

/* Marks a request cancelled. Takes effect at the next step boundary, in ANY
 * state, and frees every block it holds (S8). Returns false for an id that is
 * not live -- which is information, not an error. */
bool sched_cancel(Sched* s, u64 id);

/* Runs ONE scheduler step: drain submits -> retire finished -> admit -> build
 * the batch -> host->step -> sample -> emit. Returns the number of ROWS the
 * step carried; 0 means nothing was runnable and nothing was admissible, which
 * a driver should treat as "idle", not as "done" (see sched_pending). */
u32 sched_step(Sched* s);

/* True while any request is WAITING, PREFILLING or RUNNING. A driver loop is
 * `while (sched_pending(s)) sched_step(s);` PLUS a step budget -- the budget is
 * what turns a livelock into a failed assertion instead of a hang. */
bool sched_pending(const Sched* s);

/* ---- output ------------------------------------------------------------
 * Per-request ring, ids and a finish reason ONLY. Detokenization and
 * stop-strings live on the CONSUMER side (M6) and must never run on the
 * scheduler thread -- that is one of the five items in the bar-miss attribution
 * ladder, and it is cheaper to make it structurally impossible here. */

/* Copies up to `cap` newly generated ids for `id` into `out`, returning how
 * many were written. `*finish` is set to the reason when the request has
 * finished AND its ring is drained, SCHED_FINISH_NONE otherwise -- so a
 * consumer that stops at the first non-NONE cannot lose a trailing id. */
u32 sched_poll(Sched* s, u64 id, u32* out, u32 cap, SchedFinish* finish);

/* ---- introspection, for the sim's assertions and for the bench ---------
 * These exist so the harness can check the invariants WITHOUT reaching into
 * sched.c's private state -- a test that pokes at internals has to be rewritten
 * every time the implementation moves, and quietly stops meaning what it says
 * when a field it reads changes purpose. */
typedef struct {
    u32 n_waiting, n_running;
    u32 n_prefilling;   /* 0 or 1 -- S1 asserts it is never 2                */
    u32 free_blocks;    /* at the last step boundary                         */
    /* HIGH-WATER MARK of blocks in use, sampled at the moment of MAXIMUM
     * DEMAND: after every row of a step is backed and after any preemption that
     * backing forced, and BEFORE the step's retirements hand blocks back.
     *
     * ▶ WHY IT IS TAKEN INSIDE sched_step AND NOT BY THE DRIVER. A driver can
     * only read free_blocks at a step BOUNDARY, which is after retirement -- so
     * the step that first retires a sequence hides the peak by that sequence's
     * whole residency, and the number a memory-pressure gate most needs is
     * exactly the one that sampling would miss. This is what makes "the pool was
     * squeezed BELOW what the workload needs" a measurement rather than an
     * arithmetic claim: run the same shape at the derived pool, read the peak,
     * and every squeezed pool below it is forced to preempt by construction. */
    u32 peak_blocks_used;
    u64 steps;          /* sched_step calls that carried >= 1 row            */
    u64 preemptions;    /* thrash metric: RECORDED, never gated              */
    /* Of those preemptions, the ones whose victim was the in-flight PREFILL
     * rather than a RUNNING request.
     *
     * ▶ SPLIT OUT SO S5 CAN BE ASSERTED EXACTLY RATHER THAN BOUNDED. "every
     * runnable decode decodes every step" means decode rows must equal the
     * RUNNING count at step top, less any of those the step itself preempted.
     * A yielded prefill is also a preemption but costs no decode row, so
     * without this the identity only holds as an inequality -- and an
     * inequality would stay true through a bug that silently dropped a
     * runnable decode. */
    u64 prefill_yields;
    /* RUNNING requests retired by cancellation at a step's top.
     *
     * ▶ ALSO SPLIT OUT FOR S5, AND FOR THE SAME REASON. A cancelled request is
     * retired before rows are built, so it owes no decode row -- it is no
     * longer RUNNABLE, which is what S5 quantifies over. Without this the
     * identity mistakes a correct cancellation for a dropped decode; the first
     * version of the assertion did exactly that and reported the scheduler
     * broken on every trace with aborts in it. */
    u64 cancelled_running;
    u64 prefill_chunks; /* chunks issued; with steps, the co-run rate        */
    /* S7 (S-ALIGN): chunk boundaries that were NOT block-aligned and were not
     * the end of the prompt. Must be 0. Counted rather than fatal because the
     * sim is where an invariant belongs: chunk 128 over block 16 satisfies this
     * by construction TODAY, and this is what notices if either constant moves
     * without the other. */
    u64 chunk_misaligned;
    u64 decode_rows;    /* total decode rows stepped                         */
    u64 finished;       /* requests retired, for any reason                  */
    /* ---- prefix adoption, and WHY BOTH NUMBERS.
     * "dedup was enabled" and "dedup fired" are different claims, and a gate
     * that cannot tell them apart is not a gate -- `stoke batch --dedup` has
     * reported the pair since M4 for exactly this reason. `adopted_positions`
     * counts positions found resident at admission; `admitted_prompt_positions`
     * is the denominator, summed over the SAME admissions (so a re-admitted
     * request counts again in both, which is what makes the ratio readable
     * under preemption). Both stay 0 when dedup is off. */
    u64 adopted_positions;
    u64 admitted_prompt_positions;
} SchedStats;

void sched_stats(const Sched* s, SchedStats* out);

/* The oldest WAITING request's id, with *ok false when none. FCFS admission
 * (S3) is checkable only against something that names the order the scheduler
 * believes it has. */
u64 sched_oldest_waiting(const Sched* s, bool* ok);

/* The youngest RUNNING request's id, with *ok false when none. S4's victim
 * rule is checkable only against the same. */
u64 sched_youngest_running(const Sched* s, bool* ok);

/* The id of the most recently preempted request, with *ok false if none ever
 * was. Exists so the sim can check S4's PUSH-FRONT half -- a victim goes to the
 * head of WAITING, which is what makes its re-admission immediate and is what
 * the starvation argument rests on. Checking the victim's identity from outside
 * is otherwise impossible: `preemptions` says one happened, not to whom. */
u64 sched_last_preempted(const Sched* s, bool* ok);

/* Positions a request currently has BACKED -- prompt plus generated, minus
 * whatever a preemption freed. The reference-run comparison needs it to tell a
 * request that is BEHIND from one that has DIVERGED. Returns 0 for an unknown
 * id, which is also what a request that has just been preempted reports. */
u32 sched_progress(const Sched* s, u64 id);

/* Whether an id still has work outstanding: submitted-and-not-yet-drained, or
 * WAITING, or RUNNING. False for an unknown id and for one that has retired --
 * both of which is the point, because the caller asking is trying to find out
 * whether it may reclaim per-request state, and "never existed" and "is over"
 * warrant the same answer. Covers the SUBMIT QUEUE as well as the lists: a
 * request cancelled before it was ever admitted never becomes a Req, and a
 * predicate blind to that would leak the caller's state for it forever. */
bool sched_is_live(const Sched* s, u64 id);

/* ---- the constants the design LOCKS ------------------------------------
 * Named here rather than hidden in sched.c because the sim sweeps them and the
 * bench may tune them -- but changing their FORM (not their value) is a design
 * change and goes to Kabir as a memo, per the forward note. */

/* Prefill chunk, in positions. 128 = 8 KV blocks, so a chunk boundary is block
 * aligned and S-ALIGN holds by construction (a chunk ending mid-block leaves
 * that block unregistrable by the prefix cache). It is the same 128 the engine
 * prefills at; a change to either must re-check the other. */
#define SCHED_CHUNK_TOKENS 128u

/* Admission reserve: admit a prefill only when
 * free_blocks >= blocks(prompt) + W, with W = max(8, 2 * |RUNNING|).
 * The reserve is decode headroom -- without it a batch admits everything and
 * then cannot take its next decode block, which is livelock rather than
 * slowness. */
#define SCHED_W_FLOOR 8u
#define SCHED_W_PER_RUNNING 2u

#endif /* STOKE_SCHED_H */
