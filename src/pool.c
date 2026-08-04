/* src/pool.c -- K5: persistent workers + an atomic-counter parallel-for.
 * See pool.h for the design and for the independence argument that makes
 * parallelising matvec cost no gate.
 */
#include "common.h"

#include "pool.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>

/* The measured default -- see pool.h for the full reasoning. EIGHT, and
 * deliberately NOT the six that llama.cpp peaks at on this same box. */
#define POOL_DEFAULT_THREADS 8u

/* Below this many rows the dispatch (a broadcast, N wakeups, a join) costs
 * more than the work it hands out, so the caller just does it. Deliberately
 * expressed in CHUNKS rather than rows so it tracks POOL_CHUNK_ROWS: with
 * fewer than two chunks there is nothing to hand to a second thread anyway. */
#define POOL_MIN_CHUNKS 2u

typedef struct {
    void (*fn)(void*, u64, u64);
    void* ctx;
    u64 n_rows;
} Job;

static pthread_t* g_threads;
static u32 g_n_threads = 1u;      /* total, including the caller */
static int g_started;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_wake = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_done = PTHREAD_COND_INITIALIZER;

/* Bumped once per dispatch. A worker compares the generation it last served
 * against this one, which is what makes a missed or duplicated broadcast
 * impossible: a worker that was slow to wake still sees a NEW generation and
 * serves it exactly once, and one that wakes spuriously sees the SAME
 * generation and goes back to sleep. A bare "there is work" flag cannot
 * express either case. */
static u64 g_generation;
static u32 g_workers_done;        /* helpers finished with g_generation */
static int g_quit;
static Job g_job;

/* The claim counter. Sequentially consistent on purpose: this is claimed
 * once per 64 rows, so it is nowhere near hot enough to justify reasoning
 * about a weaker order, and a wrong memory order here is the kind of bug
 * that reproduces on one machine in a thousand runs. */
static atomic_ullong g_next_chunk;

static void run_chunks(void) {
    u64 n_rows = g_job.n_rows;
    for (;;) {
        u64 c = atomic_fetch_add(&g_next_chunk, 1ull);
        u64 begin = c * (u64)POOL_CHUNK_ROWS;
        if (begin >= n_rows) return;
        u64 end = begin + (u64)POOL_CHUNK_ROWS;
        if (end > n_rows) end = n_rows;
        g_job.fn(g_job.ctx, begin, end);
    }
}

static void* worker_main(void* arg) {
    (void)arg;
    u64 served = 0;
    for (;;) {
        pthread_mutex_lock(&g_mu);
        while (!g_quit && g_generation == served) {
            pthread_cond_wait(&g_wake, &g_mu);
        }
        if (g_quit) {
            pthread_mutex_unlock(&g_mu);
            return NULL;
        }
        served = g_generation;
        pthread_mutex_unlock(&g_mu);

        run_chunks();

        pthread_mutex_lock(&g_mu);
        g_workers_done++;
        pthread_cond_signal(&g_done);
        pthread_mutex_unlock(&g_mu);
    }
}

void pool_start(u32 n_threads) {
    if (g_started) pool_stop();
    if (n_threads == 0u) n_threads = POOL_DEFAULT_THREADS;
    g_n_threads = n_threads;
    g_generation = 0;
    g_workers_done = 0;
    g_quit = 0;
    if (n_threads <= 1u) {
        /* Inert on purpose: pool_for_rows runs inline and no thread is
         * created, so a single-threaded run has exactly the cost and exactly
         * the behaviour it had before this file existed. */
        g_started = 1;
        g_threads = NULL;
        return;
    }
    u32 helpers = n_threads - 1u;
    g_threads = (pthread_t*)xmalloc((size_t)helpers * sizeof(pthread_t));
    for (u32 i = 0; i < helpers; i++) {
        if (pthread_create(&g_threads[i], NULL, worker_main, NULL) != 0) {
            fatal("pool: could not create worker thread %u of %u", i + 1u,
                  helpers);
        }
    }
    g_started = 1;
}

void pool_stop(void) {
    if (!g_started) return;
    if (g_threads != NULL) {
        pthread_mutex_lock(&g_mu);
        g_quit = 1;
        pthread_cond_broadcast(&g_wake);
        pthread_mutex_unlock(&g_mu);
        for (u32 i = 0; i < g_n_threads - 1u; i++) {
            pthread_join(g_threads[i], NULL);
        }
        free(g_threads);
        g_threads = NULL;
    }
    g_started = 0;
    g_n_threads = 1u;
}

u32 pool_threads(void) { return g_started ? g_n_threads : 1u; }

void pool_for_rows(u64 n_rows, void (*fn)(void*, u64, u64), void* ctx) {
    if (n_rows == 0) return;
    u64 chunks = (n_rows + POOL_CHUNK_ROWS - 1u) / POOL_CHUNK_ROWS;
    if (!g_started || g_threads == NULL || g_n_threads <= 1u ||
        chunks < (u64)POOL_MIN_CHUNKS) {
        fn(ctx, 0, n_rows);
        return;
    }

    pthread_mutex_lock(&g_mu);
    g_job.fn = fn;
    g_job.ctx = ctx;
    g_job.n_rows = n_rows;
    atomic_store(&g_next_chunk, 0ull);
    g_workers_done = 0;
    g_generation++;
    pthread_cond_broadcast(&g_wake);
    pthread_mutex_unlock(&g_mu);

    /* The caller is a worker too, not a dispatcher that waits. On a 6-thread
     * pool that is a sixth of the machine that would otherwise sit idle
     * through every matvec. */
    run_chunks();

    pthread_mutex_lock(&g_mu);
    while (g_workers_done < g_n_threads - 1u) {
        pthread_cond_wait(&g_done, &g_mu);
    }
    pthread_mutex_unlock(&g_mu);
}
