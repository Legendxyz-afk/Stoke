/* src/pool.c -- K5: persistent workers + an atomic-counter parallel-for.
 * See pool.h for the design and for the independence argument that makes
 * parallelising matvec cost no gate.
 */
#include "common.h"

#include "pool.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
/* For SetThreadInformation -- see thread_prefer_performance() -- and for the
 * placement trace's GetCurrentProcessorNumber / GetProcessTimes /
 * GetLogicalProcessorInformationEx. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
/* ⚠ NOT <sched.h>, AND NOT _GNU_SOURCE. `-Isrc` makes every `<sched.h>` in
 * this tree -- including the one inside glibc's own <pthread.h> -- resolve to
 * OUR src/sched.h, the request scheduler. That has been silently true since
 * K5 and costs nothing under plain _POSIX_C_SOURCE, because pthread.h then
 * needs nothing from the real sched.h. Define _GNU_SOURCE and it does
 * (cpu_set_t), and the build breaks four lines into pthread.h. So the trace
 * asks the CPU number of the hardware directly (RDTSCP's TSC_AUX carries it
 * on Linux/x86-64) and falls back to the raw getcpu syscall elsewhere. */
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

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
    u32 chunk; /* units per claim; POOL_CHUNK_ROWS for every matvec */
} Job;

/* This thread's worker index -- 0 for the dispatching thread, 1..N-1 for the
 * helpers. Set once when the thread starts and never written again, so it
 * needs no synchronisation of any kind. See pool_worker_index() in pool.h for
 * why a parallel body wants this at all. */
static _Thread_local u32 g_worker_idx = 0u;

u32 pool_worker_index(void) { return g_worker_idx; }

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
/* ▶ ATOMIC, since 2026-08-17, so the spin below can read them WITHOUT the
 * mutex. The mutex and both condvars stay exactly as they were and remain the
 * blocking path; the atomics only make the fast path legal. Every store still
 * happens under g_mu while the waiter re-checks under g_mu, so a wakeup cannot
 * be lost -- the spin is an optimisation layered on the correct protocol, not
 * a replacement for it. */
static atomic_ullong g_generation;
static atomic_uint   g_workers_done; /* helpers finished with g_generation */
static int g_quit;
static Job g_job;

/* ▶▶ WHY WORKERS SPIN BEFORE THEY SLEEP. Measured 2026-08-17. A forward pass
 * issues ~250 matvec dispatches per token, so with a pure condvar wait each
 * helper slept and woke thousands of times a second. On this hybrid box
 * (6 P-cores + 4 E-cores) Windows' scheduler reads a thread that sleeps that
 * often as low-utilisation work and prefers an E-CORE for it -- so stoke's
 * whole pool drifts onto the slow cores, while llama.cpp's ggml workers spin
 * and therefore read as CPU-hungry and keep the P-cores.
 *
 * The cost was not subtle: unpinned 2.39/2.45 tok/s against 15.79/15.76 pinned
 * to P-cores, a 6.6x swing on the same binary minutes apart, with the sick runs
 * burning 74 CPU-seconds to the healthy runs' 26 (the same work takes more
 * processor-time on a slower core) at only ~45% duty cycle. Raising process
 * priority recovered part of it (2.39 -> 5.60), which is the tell that this is
 * a PLACEMENT decision and not the arithmetic.
 *
 * This also settles a claim the record had marked as not reproducing. The
 * ROADMAP's locked call for dynamic chunking was justified by an E-core penalty
 * that a later re-measurement could not find -- because that re-measurement was
 * taken on LLAMA, which spins and so cannot exhibit it. On stoke it reproduces
 * at 6.6x. The chunking was never the missing piece; the WAIT was.
 *
 * Tunable via STOKE_POOL_SPIN (iterations, 0 disables) so the sweep is a
 * property of the run rather than of the binary. */
/* ▶ DEFAULT 0 = OFF, deliberately. Spinning is only a PROXY for "look busy so
 * the scheduler gives me a P-core", and measured on a healthy box it buys ~2%
 * (inside the noise) for ~15% more CPU-seconds -- 30-33s against 26.5-27.5s at
 * n=64. On a box whose second documented error source is thermal drift under
 * sustained load, paying 15% more CPU forever for an unproven benefit is a bad
 * trade, and §7.3's "keep whichever is fastest, including neither" says drop it.
 * The code stays because it is verified and costs nothing switched off, and
 * because it is the fallback if the QoS opt-out below turns out insufficient. */
#define POOL_SPIN_DEFAULT 0u
static u32 g_spin = POOL_SPIN_DEFAULT;

/* ▶▶ THE ACTUAL FIX FOR THE E-CORE DRIFT: opt these threads out of EcoQoS.
 * Windows throttles threads it classifies as background work by scheduling them
 * on the efficiency cores, and a thread that sleeps on a condvar thousands of
 * times a second (≈250 matvec dispatches per token) looks exactly like
 * background work. SetThreadInformation(ThreadPowerThrottling) with StateMask=0
 * says "this thread is performance work" directly, which is the intent -- rather
 * than inferring it from a spin that burns power to look busy.
 *
 * Each thread sets its OWN state via GetCurrentThread(), so this needs no
 * native handle out of pthread_t and stays inside the portable pool structure.
 * Resolved dynamically and entirely best-effort: on a Windows without the entry
 * point, behaviour is exactly what it was before. */
#ifdef _WIN32
#ifndef THREAD_POWER_THROTTLING_EXECUTION_SPEED
#define THREAD_POWER_THROTTLING_EXECUTION_SPEED 0x1
#endif
#ifndef THREAD_POWER_THROTTLING_CURRENT_VERSION
#define THREAD_POWER_THROTTLING_CURRENT_VERSION 1
#endif
typedef struct {
    ULONG Version;
    ULONG ControlMask;
    ULONG StateMask;
} StokeThreadPowerThrottling;
typedef BOOL(WINAPI* SetThreadInformationFn)(HANDLE, int, LPVOID, DWORD);

static void thread_prefer_performance(void) {
    /* STOKE_NO_QOS=1 restores the pre-2026-08-17 behaviour, so the opt-out can
     * be A/B'd on one binary in one interleaved sitting. */
    const char* off = getenv("STOKE_NO_QOS");
    if (off && off[0] && !(off[0] == '0' && off[1] == '\0')) return;
    HANDLE k32 = GetModuleHandleA("kernel32.dll");
    if (!k32) return;
    SetThreadInformationFn set =
        (SetThreadInformationFn)(void (*)(void))GetProcAddress(
            (HMODULE)k32, "SetThreadInformation");
    if (!set) return;
    StokeThreadPowerThrottling s;
    s.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    s.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    s.StateMask = 0; /* 0 with the bit CONTROLLED = throttling OFF */
    /* ThreadPowerThrottling == 3 in the THREAD_INFORMATION_CLASS enum. */
    (void)set(GetCurrentThread(), 3, &s, (DWORD)sizeof(s));
}
#else
static void thread_prefer_performance(void) {}
#endif

void pool_prefer_performance_here(void) { thread_prefer_performance(); }

/* ▶▶ THE PLACEMENT TRACE: STOKE_POOL_TRACE=<seconds>. WHERE DID THE WORK RUN?
 *
 * Every claim this file makes about the scheduler -- E-core parking, the QoS
 * opt-out, "the caller is a worker too" -- has so far been argued from
 * throughput and CPU-seconds, never OBSERVED. On 2026-09-08 `stoke serve` read
 * 0.2% repeatable inside one server lifetime and 11.7% apart across two server
 * starts, the shape of a per-process state fixed at startup, and the first
 * suspect named for it (the caller missing the opt-out) was refuted by reading
 * pool_start. What was missing was not a hypothesis but an instrument.
 *
 * This is it. When enabled, every thread counts the chunks it claims BY THE
 * LOGICAL PROCESSOR IT WAS RUNNING ON when it claimed them, records the ideal
 * processor Windows assigned it at start, and the dispatching thread prints
 * the whole table every <seconds> together with the process's CPU times. Read
 * against the P/E map (printed once, from EfficiencyClass, not assumed) that
 * says directly whether a slow process is a process whose work sat on the
 * efficiency cores -- and, because it is periodic, whether that was true from
 * the first second (a startup state) or began partway (a transient).
 *
 * COST WHEN OFF: one predictable branch per chunk and one per dispatch.
 * COST WHEN ON: a GetCurrentProcessorNumber (RDPID/RDTSCP, ~tens of cycles)
 * per 64-row chunk of a matvec that costs tens of thousands, plus a clock
 * read every 64 dispatches. Each row is single-writer (its own thread) and
 * read racily-by-design by the reporter through relaxed atomics, so the trace
 * is TSan-clean and never takes the pool's mutex.
 *
 * ⚠ It counts CHUNKS CLAIMED THROUGH THE POOL. Work the pool runs inline on
 * the caller (n_threads <= 1, or fewer than POOL_MIN_CHUNKS chunks) is not a
 * placement question and is not counted. */
#define TRACE_MAX_CPUS 64u

typedef struct {
    _Alignas(64) atomic_ullong n[TRACE_MAX_CPUS]; /* chunks claimed on LP i */
    i64  ideal;                                   /* ideal LP at start; -1 unknown */
    u64  tid;
    char pad[48];                                 /* rows never share a line */
} TraceRow;

static double    g_trace_every_ms;  /* 0 = off */
static TraceRow* g_trace;           /* g_trace_rows rows, one per worker */
static u32       g_trace_rows;
static u64       g_trace_ecores;    /* bit i = LP i is an efficiency core; 0 = unknown/homogeneous */
static int       g_trace_ecores_known;
static u64       g_trace_dispatches;
static double    g_trace_t0_ms;
static double    g_trace_last_ms;

static u32 trace_cpu_now(void) {
#if defined(_WIN32)
    return (u32)GetCurrentProcessorNumber();
#elif defined(__linux__) && defined(__x86_64__)
    /* Linux writes (node << 12 | cpu) into IA32_TSC_AUX; RDTSCP reads it in
     * ~30 cycles with no syscall and no header the tree can shadow. */
    unsigned aux = 0;
    (void)__builtin_ia32_rdtscp(&aux);
    return (u32)(aux & 0xfffu);
#elif defined(__linux__)
    /* Declared here because <unistd.h> hides syscall() under strict POSIX. */
    extern long syscall(long number, ...);
    unsigned cpu = 0, node = 0;
    if (syscall(SYS_getcpu, &cpu, &node, (void*)0) != 0) return 0u;
    return (u32)cpu;
#else
    return 0u;
#endif
}

static i64 trace_ideal_here(void) {
#ifdef _WIN32
    typedef BOOL(WINAPI * GetIdealFn)(HANDLE, PPROCESSOR_NUMBER);
    HANDLE k32 = GetModuleHandleA("kernel32.dll");
    if (!k32) return -1;
    GetIdealFn get = (GetIdealFn)(void (*)(void))GetProcAddress(
        (HMODULE)k32, "GetThreadIdealProcessorEx");
    PROCESSOR_NUMBER pn;
    memset(&pn, 0, sizeof pn);
    if (!get || !get(GetCurrentThread(), &pn)) return -1;
    return (i64)pn.Number + 64 * (i64)pn.Group;
#else
    return -1;
#endif
}

static u64 trace_tid_here(void) {
#ifdef _WIN32
    return (u64)GetCurrentThreadId();
#else
    return (u64)(uintptr_t)pthread_self();
#endif
}

/* The P/E map, from the OS, once. Windows reports EfficiencyClass per core;
 * on a homogeneous box every core reads the same class and the mask stays 0
 * with `known` set, which the report distinguishes from "could not ask". */
static void trace_read_ecores(void) {
    g_trace_ecores       = 0;
    g_trace_ecores_known = 0;
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (len == 0) return;
    char* buf = (char*)xmalloc(len);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf,
            &len)) {
        free(buf);
        return;
    }
    u64 by_class[256];
    memset(by_class, 0, sizeof by_class);
    int max_class = 0;
    for (char* p = buf; p < buf + len;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX e =
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)p;
        if (e->Relationship == RelationProcessorCore &&
            e->Processor.GroupCount >= 1 && e->Processor.GroupMask[0].Group == 0) {
            int cls = (int)e->Processor.EfficiencyClass;
            by_class[cls] |= (u64)e->Processor.GroupMask[0].Mask;
            if (cls > max_class) max_class = cls;
        }
        p += e->Size;
    }
    free(buf);
    g_trace_ecores_known = 1;
    /* Windows: a HIGHER class is a MORE performant core, so on a hybrid part
     * the efficiency cores are every class below the top one. */
    for (int c = 0; c < max_class; c++) g_trace_ecores |= by_class[c];
#endif
}

static void trace_cpu_times(double* user_s, double* kernel_s) {
    *user_s = *kernel_s = 0.0;
#ifdef _WIN32
    FILETIME c, e, k, u;
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) {
        u64 ku = ((u64)k.dwHighDateTime << 32) | k.dwLowDateTime;
        u64 uu = ((u64)u.dwHighDateTime << 32) | u.dwLowDateTime;
        *kernel_s = (double)ku / 1e7;
        *user_s   = (double)uu / 1e7;
    }
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        *user_s   = (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1e6;
        *kernel_s = (double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec / 1e6;
    }
#endif
}

/* Called from pool_start, after g_n_threads is set and before any helper
 * exists, so the rows are in place before the first thread could count. */
static void trace_init(u32 n_threads) {
    g_trace_every_ms = 0.0;
    const char* s = getenv("STOKE_POOL_TRACE");
    if (!s || !s[0]) return;
    char*  end = NULL;
    double v   = strtod(s, &end);
    if (!end || *end != '\0' || !(v > 0.0)) return;
    g_trace_every_ms   = v * 1000.0;
    g_trace_rows       = n_threads;
    g_trace            = (TraceRow*)xcalloc((size_t)n_threads, sizeof(TraceRow));
    g_trace_dispatches = 0;
    g_trace_t0_ms = g_trace_last_ms = now_ms();
    for (u32 i = 0; i < n_threads; i++) g_trace[i].ideal = -1;
    trace_read_ecores();
    fprintf(stderr,
            "pool trace: on, every %.1fs, %u threads, ecores=%s0x%llx\n",
            v, n_threads, g_trace_ecores_known ? "" : "unknown/",
            (unsigned long long)g_trace_ecores);
}

/* Per thread, once, from the thread itself: its ideal processor and id. */
static void trace_thread_start(void) {
    if (!g_trace || g_worker_idx >= g_trace_rows) return;
    g_trace[g_worker_idx].ideal = trace_ideal_here();
    g_trace[g_worker_idx].tid   = trace_tid_here();
}

/* One chunk, claimed by this thread, on the LP it is running on right now.
 * Single writer per row, so a relaxed load+store is a plain increment. */
static inline void trace_count_chunk(void) {
    u32 c = trace_cpu_now();
    if (c >= TRACE_MAX_CPUS) c = TRACE_MAX_CPUS - 1u;
    atomic_ullong* slot = &g_trace[g_worker_idx].n[c];
    atomic_store_explicit(
        slot, atomic_load_explicit(slot, memory_order_relaxed) + 1ull,
        memory_order_relaxed);
}

u64 pool_trace_chunks(u32 worker) {
    if (!g_trace || worker >= g_trace_rows) return 0;
    u64 total = 0;
    for (u32 c = 0; c < TRACE_MAX_CPUS; c++) {
        total += atomic_load_explicit(&g_trace[worker].n[c], memory_order_relaxed);
    }
    return total;
}

u32 pool_trace_report(FILE* out) {
    if (!g_trace) return 0;
    double user_s, kernel_s;
    trace_cpu_times(&user_s, &kernel_s);
    u64 chunks = 0;
    for (u32 w = 0; w < g_trace_rows; w++) chunks += pool_trace_chunks(w);
    fprintf(out,
            "pool trace: t=+%.1fs dispatches=%llu chunks=%llu cpu_user=%.1fs "
            "cpu_kernel=%.1fs ecores=%s0x%llx\n",
            (now_ms() - g_trace_t0_ms) / 1000.0,
            (unsigned long long)g_trace_dispatches, (unsigned long long)chunks,
            user_s, kernel_s, g_trace_ecores_known ? "" : "unknown/",
            (unsigned long long)g_trace_ecores);
    for (u32 w = 0; w < g_trace_rows; w++) {
        u64 on_p = 0, on_e = 0;
        for (u32 c = 0; c < TRACE_MAX_CPUS; c++) {
            u64 n = atomic_load_explicit(&g_trace[w].n[c], memory_order_relaxed);
            if (g_trace_ecores >> c & 1ull) on_e += n; else on_p += n;
        }
        fprintf(out, "pool trace: thr %u tid=%llu ideal=%lld P=%llu E=%llu |", w,
                (unsigned long long)g_trace[w].tid, (long long)g_trace[w].ideal,
                (unsigned long long)on_p, (unsigned long long)on_e);
        for (u32 c = 0; c < TRACE_MAX_CPUS; c++) {
            u64 n = atomic_load_explicit(&g_trace[w].n[c], memory_order_relaxed);
            if (n) fprintf(out, " %u:%llu", c, (unsigned long long)n);
        }
        fputc('\n', out);
    }
    fflush(out);
    return g_trace_rows;
}

/* Dispatch thread, once per dispatch: the periodic report, clock read only
 * every 64th call so the trace never adds a syscall-rate cost to dispatch. */
static inline void trace_tick(void) {
    g_trace_dispatches++;
    if ((g_trace_dispatches & 63ull) != 0) return;
    double now = now_ms();
    if (now - g_trace_last_ms < g_trace_every_ms) return;
    g_trace_last_ms = now;
    pool_trace_report(stderr);
}

static void trace_stop(void) {
    if (!g_trace) return;
    pool_trace_report(stderr);
    free(g_trace);
    g_trace          = NULL;
    g_trace_rows     = 0;
    g_trace_every_ms = 0.0;
}

static inline void pool_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

/* The claim counter. Sequentially consistent on purpose: this is claimed
 * once per 64 rows, so it is nowhere near hot enough to justify reasoning
 * about a weaker order, and a wrong memory order here is the kind of bug
 * that reproduces on one machine in a thousand runs. */
static atomic_ullong g_next_chunk;

static void run_chunks(void) {
    u64 n_rows = g_job.n_rows;
    u64 chunk = (u64)g_job.chunk;
    for (;;) {
        u64 c = atomic_fetch_add(&g_next_chunk, 1ull);
        u64 begin = c * chunk;
        if (begin >= n_rows) return;
        u64 end = begin + chunk;
        if (end > n_rows) end = n_rows;
        g_job.fn(g_job.ctx, begin, end);
        if (g_trace) trace_count_chunk();
    }
}

static void* worker_main(void* arg) {
    /* The index is passed BY VALUE through the void*, not by pointer into
     * pool_start's frame: a pointer to a loop variable is the classic
     * pthread_create race, where the loop advances before the worker reads it
     * and two workers see the same number. */
    g_worker_idx = (u32)(uintptr_t)arg;
    thread_prefer_performance();
    trace_thread_start();
    u64 served = 0;
    for (;;) {
        /* Spin for a bounded budget, then fall back to the condvar. On quit the
         * generation never advances, so a spinning worker simply exhausts its
         * budget and then blocks and sees g_quit -- a bounded shutdown delay,
         * not a hang. */
        for (u32 s = 0; s < g_spin; s++) {
            if (atomic_load_explicit(&g_generation, memory_order_acquire) !=
                served) {
                break;
            }
            pool_relax();
        }
        pthread_mutex_lock(&g_mu);
        while (!g_quit &&
               atomic_load_explicit(&g_generation, memory_order_relaxed) ==
                   served) {
            pthread_cond_wait(&g_wake, &g_mu);
        }
        if (g_quit) {
            pthread_mutex_unlock(&g_mu);
            return NULL;
        }
        served = atomic_load_explicit(&g_generation, memory_order_relaxed);
        pthread_mutex_unlock(&g_mu);

        run_chunks();

        pthread_mutex_lock(&g_mu);
        atomic_fetch_add_explicit(&g_workers_done, 1u, memory_order_release);
        pthread_cond_signal(&g_done);
        pthread_mutex_unlock(&g_mu);
    }
}

void pool_start(u32 n_threads) {
    if (g_started) pool_stop();
    if (n_threads == 0u) n_threads = POOL_DEFAULT_THREADS;
    g_n_threads = n_threads;
    atomic_store(&g_generation, 0ull);
    atomic_store(&g_workers_done, 0u);
    g_quit = 0;
    /* The caller runs chunks too (see pool_for_rows), so it needs the same
     * treatment as the helpers -- otherwise one of the N workers is still
     * eligible for an efficiency core. */
    thread_prefer_performance();
    g_worker_idx = 0u; /* the dispatching thread is worker 0 */
    /* Rows first, then this thread's own row: helpers created below inherit
     * a table that already exists. */
    trace_init(n_threads);
    trace_thread_start();
    /* Read once here, never in the wait loop: a getenv() per dispatch would
     * cost more than the spin it configures. */
    g_spin = POOL_SPIN_DEFAULT;
    {
        const char* s = getenv("STOKE_POOL_SPIN");
        if (s && s[0]) {
            char*         end = NULL;
            unsigned long v = strtoul(s, &end, 10);
            if (end && *end == '\0' && v <= 10000000ul) g_spin = (u32)v;
        }
    }
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
        void* idx = (void*)(uintptr_t)(i + 1u);
        if (pthread_create(&g_threads[i], NULL, worker_main, idx) != 0) {
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
    trace_stop(); /* after the join: every row's writer is gone */
    g_started = 0;
    g_n_threads = 1u;
}

u32 pool_threads(void) { return g_started ? g_n_threads : 1u; }

void pool_for_rows(u64 n_rows, void (*fn)(void*, u64, u64), void* ctx) {
    pool_for_units(n_rows, POOL_CHUNK_ROWS, fn, ctx);
}

void pool_for_units(u64 n_rows, u32 chunk, void (*fn)(void*, u64, u64),
                    void* ctx) {
    if (n_rows == 0) return;
    if (chunk == 0u) chunk = POOL_CHUNK_ROWS;
    u64 chunks = (n_rows + chunk - 1u) / chunk;
    if (!g_started || g_threads == NULL || g_n_threads <= 1u ||
        chunks < (u64)POOL_MIN_CHUNKS) {
        fn(ctx, 0, n_rows);
        return;
    }

    pthread_mutex_lock(&g_mu);
    g_job.fn = fn;
    g_job.ctx = ctx;
    g_job.n_rows = n_rows;
    g_job.chunk = chunk;
    atomic_store(&g_next_chunk, 0ull);
    atomic_store_explicit(&g_workers_done, 0u, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_generation, 1ull, memory_order_release);
    pthread_cond_broadcast(&g_wake);
    pthread_mutex_unlock(&g_mu);

    /* The caller is a worker too, not a dispatcher that waits. On a 6-thread
     * pool that is a sixth of the machine that would otherwise sit idle
     * through every matvec. */
    run_chunks();

    /* Same shape on the join: the caller has just finished its own chunks and
     * the helpers are typically microseconds behind, so blocking here was
     * paying a context switch to wait out a delay shorter than the switch. */
    for (u32 s = 0; s < g_spin; s++) {
        if (atomic_load_explicit(&g_workers_done, memory_order_acquire) >=
            g_n_threads - 1u) {
            break;
        }
        pool_relax();
    }
    pthread_mutex_lock(&g_mu);
    while (atomic_load_explicit(&g_workers_done, memory_order_relaxed) <
           g_n_threads - 1u) {
        pthread_cond_wait(&g_done, &g_mu);
    }
    pthread_mutex_unlock(&g_mu);
    if (g_trace) trace_tick();
}
