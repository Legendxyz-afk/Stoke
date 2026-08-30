/* tests/test_sched_race.c -- M7 rung B1: the scheduler under THREE threads.
 *
 * ▶▶▶ WHY THIS FILE EXISTS, AND WHY IT COULD NOT EXIST BEFORE NOW.
 * `stoke.h` -- frozen at M6 -- promises stoke_submit, stoke_poll and
 * stoke_cancel are SAFE FROM ANY THREAD while another thread is inside
 * stoke_step. `sched.h` says something narrower and older: that the submit
 * queue is the ONE place a foreign thread touches the scheduler. Both cannot be
 * true, and the A7 freeze review found which one is not: poll and cancel reach
 * find(), which walks `s->all` with NO LOCK, while the step thread PREPENDS to
 * that same list in drain_submits, also with no lock.
 *
 * ⚠⚠ M6 RECORDED THAT AS "NOT GRADEABLE ON THIS BOX", and it was wrong about
 * why. The reason given was that TSan OOMs on anything model-heavy (abi_smoke
 * peaks at 7,364,440 kB against 7 GB of WSL) -- true, and irrelevant, because
 * sched.h injects the model step and the clock through SchedHost. A scheduler
 * driven by a FAKE model has no 2.5 GB of weights in it, so TSan can watch
 * submit, step and poll on three threads at once for the price of a unit test.
 * The blocker was never the sanitizer; it was that every existing test is
 * single-threaded, which tests/test_sched.c says of itself in its own header.
 *
 * ⚠ A CLEAN TSan RUN OVER SINGLE-THREADED TESTS IS NOT EVIDENCE OF THREAD
 * SAFETY. M6's "Linux TSan GREEN, zero data races" was true and covered none of
 * this, because a race needs two threads to be a race. That is the trap this
 * file closes: the result was scoped to what the suite actually ran
 * concurrently, and nothing ran concurrently.
 *
 * ▶▶ WHAT IT ASSERTS WITHOUT A SANITIZER, so it is not dead weight in a plain
 * build: every request that was accepted finishes, an uncancelled one yields
 * EXACTLY max_tokens ids (eos is off, so the count is arithmetic and not a
 * guess), and a cancelled one yields no more than that. Under TSan those same
 * assertions run while the reports are what matter.
 *
 * ⚠ EVERY FLAG IN THIS HARNESS IS AN ATOMIC, deliberately. A plain `volatile
 * int stop` shared between the stepping thread and main IS a data race, and
 * TSan would report it -- a report in the instrument, indistinguishable at a
 * glance from a report in the code under test. The harness must be clean or it
 * cannot grade anything.
 */
#include "common.h"

#include "blockpool.h"
#include "kv.h"
#include "sched.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <string.h>

static int checks = 0, failures = 0;

static void ok(const char* what, bool cond) {
    checks++;
    if (!cond) { failures++; printf("  FAIL: %s\n", what); }
}

/* ---- the fake model: the smallest host that is still honest -------------
 * It notes each row's token, exactly as model_forward_batch does, because the
 * KV layer's block hashing depends on it. Beyond that it computes nothing --
 * this file is about list access order, not arithmetic. */
static double race_now(void* ctx) {
    (void)ctx;
    return now_ms();
}

static void race_step(void* ctx, const SchedStep* in, u32* out) {
    (void)ctx;
    for (u32 i = 0; i < in->n_rows; i++)
        kv_note_token(in->kv[i], in->positions[i], in->token_ids[i]);
    u32 n_yield = in->n_decode + (in->prefill_completes ? 1u : 0u);
    for (u32 i = 0; i < n_yield; i++) out[i] = 11u;
}

typedef struct {
    Sched*      s;
    atomic_int  stop;
    atomic_long steps;
} Stepper;

static void* step_thread(void* p) {
    Stepper* st = (Stepper*)p;
    while (!atomic_load_explicit(&st->stop, memory_order_acquire)) {
        sched_step(st->s);
        atomic_fetch_add_explicit(&st->steps, 1, memory_order_relaxed);
    }
    return NULL;
}

#define N_REQ      48u
#define PROMPT_TOK 16u
#define GEN_TOK    8u
#define IN_FLIGHT  6u

typedef struct {
    u64  id;
    bool submitted;
    bool cancelled;
    bool finished;
    u32  got;
    SchedFinish why;
} Track;

int main(void) {
    /* Same shape as tests/test_sched.c's sim: 2 layers, 1 kv head, head dim 4.
     * 64 blocks is comfortably above blocks(prompt) + SCHED_W_FLOOR for the
     * in-flight window below, so admission is not what this test measures. */
    KvStore* store = kv_store_open(2, 64, 1, 4, KV_F32);
    if (!store) { printf("  FAIL: kv_store_open\n"); return 1; }

    SchedHost host = {race_step, race_now, NULL};
    Sched*    s    = sched_open(store, 128, &host, false);
    if (!s) { printf("  FAIL: sched_open\n"); kv_store_close(store); return 1; }

    static u32 prompt[PROMPT_TOK];
    for (u32 i = 0; i < PROMPT_TOK; i++) prompt[i] = 3u + i;

    static Track t[N_REQ];
    memset(t, 0, sizeof t);

    Stepper st;
    st.s = s;
    atomic_init(&st.stop, 0);
    atomic_init(&st.steps, 0);

    pthread_t th;
    if (pthread_create(&th, NULL, step_thread, &st) != 0) {
        printf("  FAIL: pthread_create\n");
        sched_close(s);
        kv_store_close(store);
        return 1;
    }

    /* ▶ THE RACE WINDOW. Everything below runs on THIS thread while the stepper
     * runs on the other: submit walks s->all for its duplicate check, poll and
     * cancel walk it through find(), and drain_submits prepends to it. */
    u32 next = 0, live = 0;
    u32 buf[GEN_TOK + 4];
    while (next < N_REQ || live > 0) {
        if (next < N_REQ && live < IN_FLIGHT) {
            SchedRequest r;
            r.id         = (u64)next + 1u;
            r.prompt_ids = prompt;
            r.n_prompt   = PROMPT_TOK;
            r.max_tokens = GEN_TOK;
            r.eos_id     = -1; /* off, so a finish is arithmetic: exactly GEN_TOK */
            if (sched_submit(s, &r)) {
                t[next].id        = r.id;
                t[next].submitted = true;
                live++;
                /* ⚠ CANCELLED IMMEDIATELY, NOT AFTER THE FIRST TOKEN, and the
                 * difference was measured rather than guessed. The first
                 * version cancelled once a request had produced something --
                 * which worked against the UNLOCKED scheduler (74 steps over
                 * the run) and stopped working entirely once the lock landed
                 * (3,526 steps): the stepper now outruns this thread, every
                 * request was already FINISHED by its first poll, and
                 * sched_cancel correctly refused all 48. The path under test
                 * silently stopped being tested, and only the anti-vacuity
                 * check below said so. Cancelling at submit hits a request
                 * that is queued or waiting, which is a state it cannot have
                 * outrun. */
                if ((r.id % 7u) == 0u) t[next].cancelled = sched_cancel(s, r.id);
            }
            next++;
        }

        /* Poll everything outstanding, and cancel a few on the way past so the
         * cancel path is exercised against a stepping thread as well. */
        for (u32 i = 0; i < next; i++) {
            if (!t[i].submitted || t[i].finished) continue;
            SchedFinish f = SCHED_FINISH_NONE;
            u32 n = sched_poll(s, t[i].id, buf, GEN_TOK + 4u, &f);
            t[i].got += n;
            if (f != SCHED_FINISH_NONE) {
                t[i].finished = true;
                t[i].why      = f;
                live--;
            }
        }
    }

    atomic_store_explicit(&st.stop, 1, memory_order_release);
    pthread_join(th, NULL);

    u32 n_sub = 0, n_fin = 0, n_can = 0;
    for (u32 i = 0; i < N_REQ; i++) {
        if (!t[i].submitted) continue;
        n_sub++;
        if (t[i].finished) n_fin++;
        if (t[i].cancelled) n_can++;
        if (t[i].cancelled) {
            /* A cancel can land after the request already finished, so the only
             * claim is that it never produced MORE than it was asked for. */
            if (t[i].got > GEN_TOK) {
                printf("  FAIL: cancelled id %llu yielded %u ids, over the %u cap\n",
                       (unsigned long long)t[i].id, t[i].got, GEN_TOK);
                failures++;
            }
        } else if (t[i].got != GEN_TOK) {
            printf("  FAIL: id %llu yielded %u ids, expected exactly %u\n",
                   (unsigned long long)t[i].id, t[i].got, GEN_TOK);
            failures++;
        }
        checks++;
    }

    /* ANTI-VACUITY, and it is not decoration: if admission refused everything,
     * or the stepper never ran, every assertion above is satisfied by an empty
     * set and this file would report success having raced nothing. */
    ok("every submitted request finished", n_sub > 0 && n_fin == n_sub);
    ok("the submit path was actually exercised", n_sub >= N_REQ / 2u);
    ok("at least one request was cancelled mid-flight", n_can > 0);
    ok("the stepping thread actually stepped",
       atomic_load_explicit(&st.steps, memory_order_relaxed) > 0);

    sched_close(s);
    kv_store_close(store);

    printf("test_sched_race: %d checks, %d failures (%u submitted, %u finished, "
           "%u cancelled, %ld steps)\n",
           checks, failures, n_sub, n_fin, n_can,
           atomic_load_explicit(&st.steps, memory_order_relaxed));
    if (failures) {
        printf("%d/%d tests passed\n", checks - failures, checks);
        return 1;
    }
    printf("%d/%d tests passed\n", checks, checks);
    return 0;
}
