/* tests/abi_firsttoken.c -- M7 rung B3: a request submitted MID-STEP must not
 * have its first token chosen greedily.
 *
 * ▶▶▶ THE DEFECT, WHICH `stoke.h` DOCUMENTS RATHER THAN HIDES. `stoke_step`
 * installs the sampling state for newly queued requests and THEN admits;
 * `stoke_submit` makes the request visible to admission and THEN queues its
 * params. A submit landing between those two points is admitted, prefilled and
 * yields its FIRST token in that same step -- with its params still in the
 * queue, so the sampler finds no slot and returns the argmax. One token,
 * greedily, in a stream that stays fluent and whose every later token is
 * sampled exactly as asked.
 *
 * ⚠⚠ IN A SHIPPING BUILD THAT WINDOW IS A FEW HUNDRED NANOSECONDS and no test
 * can hit it on purpose -- which is why M6 shipped this as a documented
 * ordering rather than a fixed bug. `-DSTOKE_MUT_WIDEN_SUBMIT_WINDOW` spins for
 * 20 ms between the two lines. The width is the ONLY difference between that
 * binary and the defect; hitting a real window on a loaded box is a matter of
 * luck, and a test that depends on luck reports "no bug" most days.
 *
 * ▶▶ HOW IT IS DETECTED, since a greedy token is a perfectly ordinary token.
 * Each request is run TWICE with the same prompt, the same params and the same
 * seed: once QUIESCENTLY (submit, then step, on this thread -- an ordering the
 * header says cannot hit the window) and once while a second thread steps. The
 * draws are a pure function of (seed, position), so the two id streams must be
 * IDENTICAL. If the raced one had its first token taken by the argmax instead,
 * they differ at index 0.
 *
 * ⚠ EIGHT PAIRS, NOT ONE, and the reason is that a coincidence is likely: at
 * any temperature the sampled token sometimes IS the argmax, and a single pair
 * would then agree while the defect was present. Eight independent seeds make
 * that vanishingly unlikely, and the run prints how many pairs actually raced
 * so a green result cannot mean "the window was never entered".
 */
#include "common.h"

#include "stoke.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#define N_PAIRS 8u
#define N_GEN   4
#define TEMP    5.0f /* high on purpose: the sampled id is rarely the argmax */

static stoke_engine* g_e;

static int fail(const char* what) {
    fprintf(stderr, "abi_firsttoken: FAIL %s -- %s\n", what, stoke_last_error());
    return 1;
}

/* Steps until told to stop. Every flag it shares with main is an atomic: a
 * plain int here would be a data race in the HARNESS, and this file exists to
 * talk about races in the engine. */
typedef struct {
    atomic_int  stop;
    atomic_long steps;
} Stepper;

static void* step_thread(void* p) {
    Stepper* st = (Stepper*)p;
    while (!atomic_load_explicit(&st->stop, memory_order_acquire)) {
        uint32_t rows = 0;
        stoke_step(g_e, &rows);
        atomic_fetch_add_explicit(&st->steps, 1, memory_order_relaxed);
    }
    return NULL;
}

/* Submits one request and drains it. `drive` non-zero means this thread also
 * steps (the quiescent reference); zero means someone else is stepping. */
static int run_one(uint64_t id, uint64_t seed, const uint32_t* prompt,
                   int32_t n_prompt, int drive, uint32_t* out, int32_t* n_out) {
    stoke_request req;
    memset(&req, 0, sizeof req);
    req.size        = sizeof req;
    req.id          = id;
    req.prompt_ids  = prompt;
    req.n_prompt    = (uint32_t)n_prompt;
    req.max_tokens  = N_GEN;
    req.eos_id      = -1; /* off: every request yields exactly N_GEN ids */
    req.temperature = TEMP;
    req.seed        = seed;
    if (stoke_submit(g_e, &req) != STOKE_OK) return -1;

    /* ⚠ A WALL-CLOCK DEADLINE, NOT AN ITERATION COUNT, and the first version got
     * this wrong: with `drive` zero this thread only polls, so a 20,000-turn
     * loop expired in milliseconds while the other thread was still spinning
     * 20 ms into its first step. The run reported a failure that was entirely
     * in the harness. A poll loop that does not wait is not waiting. */
    int32_t got = 0, finish = STOKE_FINISH_NONE;
    double  t0 = now_ms();
    while (finish == STOKE_FINISH_NONE) {
        if (drive) {
            uint32_t rows = 0;
            if (stoke_step(g_e, &rows) != STOKE_OK) return -1;
        } else {
            /* Give the stepping thread the CPU rather than starving it. */
            double s = now_ms();
            while (now_ms() - s < 1.0) { }
        }
        int32_t k = stoke_poll(g_e, id, out + got, N_GEN - got, &finish);
        if (k < 0) return -1;
        got += k;
        if (now_ms() - t0 > 120000.0) return -1;
    }
    *n_out = got;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: abi_firsttoken <model.gguf>\n");
        return 2;
    }

    stoke_open_opts opts;
    memset(&opts, 0, sizeof opts);
    opts.size     = sizeof opts;
    opts.max_ctx  = 256;
    opts.max_seqs = 4;
    g_e = stoke_open(argv[1], &opts);
    if (!g_e) return fail("stoke_open");

    uint32_t prompt[64];
    int32_t  n_prompt = stoke_tokenize(g_e, "The sea is", -1, 0, prompt, 64);
    if (n_prompt <= 0) { stoke_close(g_e); return fail("stoke_tokenize"); }

    /* ---- the references, taken quiescently on this thread ---- */
    uint32_t ref[N_PAIRS][N_GEN];
    int32_t  ref_n[N_PAIRS];
    for (uint32_t i = 0; i < N_PAIRS; i++) {
        if (run_one(1000u + i, 700u + i, prompt, n_prompt, 1, ref[i], &ref_n[i]) != 0) {
            stoke_close(g_e);
            return fail("reference run");
        }
    }

    /* ---- the same eight, submitted while another thread steps ---- */
    Stepper st;
    atomic_init(&st.stop, 0);
    atomic_init(&st.steps, 0);
    pthread_t th;
    if (pthread_create(&th, NULL, step_thread, &st) != 0) {
        stoke_close(g_e);
        return fail("pthread_create");
    }

    uint32_t raced[N_PAIRS][N_GEN];
    int32_t  raced_n[N_PAIRS];
    int      bad = 0;
    for (uint32_t i = 0; i < N_PAIRS; i++) {
        if (run_one(2000u + i, 700u + i, prompt, n_prompt, 0, raced[i], &raced_n[i]) != 0) {
            atomic_store_explicit(&st.stop, 1, memory_order_release);
            pthread_join(th, NULL);
            stoke_close(g_e);
            return fail("raced run");
        }
    }
    atomic_store_explicit(&st.stop, 1, memory_order_release);
    pthread_join(th, NULL);

    for (uint32_t i = 0; i < N_PAIRS; i++) {
        if (raced_n[i] != ref_n[i] ||
            memcmp(raced[i], ref[i], (size_t)ref_n[i] * sizeof(uint32_t)) != 0) {
            bad++;
            fprintf(stderr,
                    "abi_firsttoken: FAIL pair %u -- the raced stream differs "
                    "from the quiescent one for the SAME seed.\n"
                    "  quiescent:", i);
            for (int32_t k = 0; k < ref_n[i]; k++) fprintf(stderr, " %u", ref[i][k]);
            fprintf(stderr, "\n  raced:    ");
            for (int32_t k = 0; k < raced_n[i]; k++) fprintf(stderr, " %u", raced[i][k]);
            fprintf(stderr, "\n  (index 0 differing means the first token was "
                            "taken by the argmax: the request was admitted in a "
                            "step whose sampling prologue had already run)\n");
        }
    }

    long steps = atomic_load_explicit(&st.steps, memory_order_relaxed);
    /* ANTI-VACUITY: if the stepping thread never ran, or every request was
     * drained by this thread's own polling before the stepper touched it, then
     * nothing was raced and eight agreeing pairs mean nothing at all. */
    if (steps <= 0) {
        fprintf(stderr, "abi_firsttoken: FAIL the stepping thread never stepped "
                        "-- nothing was raced\n");
        bad++;
    }

    stoke_close(g_e);
    printf("abi_firsttoken: %u pairs, %d differing, %ld steps on the racing "
           "thread\n", N_PAIRS, bad, steps);
    if (bad) return 1;
    printf("abi_firsttoken: OK (a mid-step submit keeps its first token)\n");
    return 0;
}
