/* tests/test_pool.c -- K5's gate: the pool must not change a single bit.
 *
 * WHAT IS BEING PROVED, and it is not "the locks work". Locking correctly is
 * necessary and TSan checks it on every WSL run. What THIS file checks is the
 * stronger property the design rests on: matvec rows are independent dot
 * products, so spreading them over threads reorders nothing and every y[r]
 * is computed by exactly the same sequential kernel over exactly the same
 * bytes. Therefore the output must be BIT-IDENTICAL at 1 thread and at N,
 * for every quant type, and identical again when run twice at N.
 *
 * A tolerance would be worse than useless here: the failure mode of a racy
 * or reordered parallel matvec is a handful of wrong values, which is
 * exactly what an epsilon hides. Bits or nothing.
 *
 * THE RUN-TWICE CHECK IS NOT REDUNDANT. Comparing N against 1 catches a
 * systematically wrong partition. Comparing N against N catches
 * NONDETERMINISM -- a race that resolves differently between runs -- which a
 * single N-vs-1 comparison can pass by luck. They fail on different bugs.
 */
#include "common.h"

#include "kernels/kernels.h"
#include "pool.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

/* Wide enough to span many 64-row chunks (so the counter is genuinely
 * contended) without making the test slow. */
#define N_IN 512u
#define N_OUT 1000u

static u64 g_rng = 0xD1B54A32D192ED03ull;

static u32 rng_next(void) {
    g_rng ^= g_rng >> 12;
    g_rng ^= g_rng << 25;
    g_rng ^= g_rng >> 27;
    return (u32)((g_rng * 0x2545F4914F6CDD1Dull) >> 32);
}

static void reset_rng(void) { g_rng = 0xD1B54A32D192ED03ull; }

/* Builds a fake 2D GgufTensor over caller-owned bytes -- the same idiom
 * test_kernels.c uses, so no GGUF file is needed. */
static void fake_tensor(GgufTensor* w, const char* name, u32 type,
                        const void* data, u64 nbytes) {
    memset(w, 0, sizeof *w);
    w->name = name;
    w->type = type;
    w->n_dims = 2;
    w->dims[0] = N_IN;
    w->dims[1] = N_OUT;
    w->data = (void*)(uintptr_t)data;
    w->nbytes = nbytes;
    w->offset = 0;
}

static u32 bits_differ(const f32* a, const f32* b, u64 n) {
    u32 bad = 0;
    for (u64 i = 0; i < n; i++) {
        if (memcmp(&a[i], &b[i], sizeof(f32)) != 0) bad++;
    }
    return bad;
}

/* Runs one matvec at `threads` and writes n_out results into `y`. */
static void run_at(u32 threads, const GgufTensor* w, const f32* x, f32* y,
                   void* aq) {
    pool_start(threads);
    matvec(w, x, N_IN, y, N_OUT, aq);
    pool_stop();
}

static void check_type(const char* label, u32 type, u64 block_elems,
                       u64 block_bytes) {
    u64 row_bytes = (N_IN / block_elems) * block_bytes;
    u64 wbytes = row_bytes * N_OUT;
    u8* wdata = (u8*)xmalloc((size_t)wbytes);
    reset_rng();
    for (u64 i = 0; i < wbytes; i++) wdata[i] = (u8)(rng_next() & 0xFFu);
    /* Keep every f16 scale finite: an inf/nan block scale is a fatal() path
     * on both implementations and is tested elsewhere, not here. */
    if (type == GGUF_GGML_TYPE_Q8_0) {
        for (u64 b = 0; b < wbytes / block_bytes; b++) {
            u16 h = f32_to_f16(0.0125f);
            memcpy(wdata + b * block_bytes, &h, 2);
        }
    } else if (type == GGUF_GGML_TYPE_Q4_K) {
        for (u64 b = 0; b < wbytes / block_bytes; b++) {
            u16 d = f32_to_f16(0.0125f), dm = f32_to_f16(0.003f);
            memcpy(wdata + b * block_bytes, &d, 2);
            memcpy(wdata + b * block_bytes + 2, &dm, 2);
        }
    } else if (type == GGUF_GGML_TYPE_Q6_K) {
        for (u64 b = 0; b < wbytes / block_bytes; b++) {
            u16 d = f32_to_f16(0.0125f);
            memcpy(wdata + b * block_bytes + 208, &d, 2);
        }
    }

    GgufTensor w;
    fake_tensor(&w, label, type, wdata, wbytes);

    f32* x = (f32*)xmalloc((size_t)N_IN * sizeof(f32));
    for (u32 i = 0; i < N_IN; i++) {
        x[i] = sinf((f32)i * 0.017f) * 1.25f;
    }

    /* One scratch, sized for the larger of the two activation layouts --
     * mirrors model.c's own sizing rule. */
    u64 aq_bytes = (N_IN / KERNELS_Q8_K_BLOCK_ELEMS) * KERNELS_Q8_K_BLOCK_BYTES;
    u64 q80 = (N_IN / KERNELS_Q8_0_BLOCK_ELEMS) * KERNELS_Q8_0_BLOCK_BYTES;
    if (q80 > aq_bytes) aq_bytes = q80;
    void* aq = xmalloc((size_t)aq_bytes);

    f32* y1 = (f32*)xmalloc((size_t)N_OUT * sizeof(f32));
    f32* yn = (f32*)xmalloc((size_t)N_OUT * sizeof(f32));
    f32* yn2 = (f32*)xmalloc((size_t)N_OUT * sizeof(f32));

    run_at(1u, &w, x, y1, aq);
    run_at(6u, &w, x, yn, aq);
    run_at(6u, &w, x, yn2, aq);

    u32 d1 = bits_differ(y1, yn, N_OUT);
    u32 d2 = bits_differ(yn, yn2, N_OUT);
    if (d1 || d2) {
        printf("  %s: %u of %u differ 1T-vs-6T, %u differ 6T-vs-6T\n", label,
               d1, (u32)N_OUT, d2);
    }
    ASSERT_EQ_U32(d1, 0u);
    ASSERT_EQ_U32(d2, 0u);

    /* Guard against a vacuous pass: if every output were 0 (a matvec that
     * silently did nothing) the comparisons above would all succeed. */
    u32 nonzero = 0;
    for (u32 i = 0; i < N_OUT; i++) {
        if (y1[i] != 0.0f) nonzero++;
    }
    ASSERT_EQ_U32(nonzero > 0u ? 1u : 0u, 1u);

    free(wdata);
    free(x);
    free(aq);
    free(y1);
    free(yn);
    free(yn2);
}

TEST(pool_matvec_f32_is_bit_identical_at_1_and_6_threads) {
    check_type("fake.f32.weight", GGUF_GGML_TYPE_F32, 1u, 4u);
}

TEST(pool_matvec_q8_0_is_bit_identical_at_1_and_6_threads) {
    check_type("fake.q8_0.weight", GGUF_GGML_TYPE_Q8_0,
               KERNELS_Q8_0_BLOCK_ELEMS, KERNELS_Q8_0_BLOCK_BYTES);
}

TEST(pool_matvec_q4_k_is_bit_identical_at_1_and_6_threads) {
    check_type("fake.q4_k.weight", GGUF_GGML_TYPE_Q4_K,
               KERNELS_Q4_K_BLOCK_ELEMS, KERNELS_Q4_K_BLOCK_BYTES);
}

TEST(pool_matvec_q6_k_is_bit_identical_at_1_and_6_threads) {
    check_type("fake.q6_k.weight", GGUF_GGML_TYPE_Q6_K,
               KERNELS_Q6_K_BLOCK_ELEMS, KERNELS_Q6_K_BLOCK_BYTES);
}

/* Every row must be written exactly once. A chunking bug that skipped the
 * tail, or one that let two threads claim the same chunk, shows here as a
 * poison value surviving or a count that is not n_rows. */
static u32 g_visits[N_OUT];

static void count_visit(void* ctx, u64 begin, u64 end) {
    (void)ctx;
    for (u64 r = begin; r < end; r++) {
        /* Not atomic on purpose: if two threads ever claimed the same chunk
         * this would be a data race AND would produce a wrong count, and
         * TSan would report the race on the WSL leg. Both signals are
         * wanted. */
        g_visits[r]++;
    }
}

TEST(pool_for_rows_visits_every_row_exactly_once) {
    static const u64 sizes[] = {1, 63, 64, 65, 127, 128, 1000, N_OUT};
    for (u32 s = 0; s < (u32)(sizeof sizes / sizeof sizes[0]); s++) {
        u64 n = sizes[s];
        if (n > N_OUT) continue;
        memset(g_visits, 0, sizeof g_visits);
        pool_start(6u);
        pool_for_rows(n, count_visit, NULL);
        pool_stop();
        u32 bad = 0;
        for (u64 r = 0; r < n; r++) {
            if (g_visits[r] != 1u) bad++;
        }
        for (u64 r = n; r < N_OUT; r++) {
            if (g_visits[r] != 0u) bad++;   /* wrote past the end */
        }
        if (bad) printf("  n=%llu: %u rows visited wrongly\n",
                        (unsigned long long)n, bad);
        ASSERT_EQ_U32(bad, 0u);
    }
}

/* The pool must be usable repeatedly: the generation counter, not a bare
 * "there is work" flag, is what makes a second dispatch safe after the first
 * has drained. Ten rounds because a missed wakeup is intermittent. */
TEST(pool_survives_repeated_dispatch) {
    pool_start(6u);
    u32 bad = 0;
    for (u32 round = 0; round < 10u; round++) {
        memset(g_visits, 0, sizeof g_visits);
        pool_for_rows(N_OUT, count_visit, NULL);
        for (u32 r = 0; r < N_OUT; r++) {
            if (g_visits[r] != 1u) bad++;
        }
    }
    pool_stop();
    ASSERT_EQ_U32(bad, 0u);
}

/* L4: pool_for_units at the SMALL sizes attention uses.
 *
 * pool_for_rows could never reach this regime -- its chunk is fixed at 64 and
 * its floor is two chunks, so anything under 128 units ran inline on the
 * caller and no test noticed, because matvec's smallest output is 1024 rows.
 * Attention dispatches 32 units at a chunk of 1, which is entirely below what
 * the old entry point could express. The sizes here bracket the chunk boundary
 * (n not a multiple of chunk, n smaller than chunk, n exactly one chunk) since
 * a tail that is dropped or double-claimed is the failure this catches. */
TEST(pool_for_units_visits_every_unit_exactly_once) {
    static const u64 sizes[] = {1, 2, 3, 7, 8, 31, 32, 33, 64, 128};
    static const u32 chunks[] = {1, 2, 4, 8};
    for (u32 ci = 0; ci < (u32)(sizeof chunks / sizeof chunks[0]); ci++) {
        for (u32 s = 0; s < (u32)(sizeof sizes / sizeof sizes[0]); s++) {
            u64 n = sizes[s];
            memset(g_visits, 0, sizeof g_visits);
            pool_start(6u);
            pool_for_units(n, chunks[ci], count_visit, NULL);
            pool_stop();
            u32 bad = 0;
            for (u64 r = 0; r < n; r++) {
                if (g_visits[r] != 1u) bad++;
            }
            for (u64 r = n; r < N_OUT; r++) {
                if (g_visits[r] != 0u) bad++; /* wrote past the end */
            }
            if (bad) {
                printf("  n=%llu chunk=%u: %u units visited wrongly\n",
                       (unsigned long long)n, chunks[ci], bad);
            }
            ASSERT_EQ_U32(bad, 0u);
        }
    }
}

/* Every worker must see a DISTINCT index inside [0, pool_threads()), because
 * per-worker scratch is indexed by it: two workers reading the same index
 * would share one scores row and one KV buffer, which is silent and lands in
 * the arithmetic. The caller must be 0 -- it runs chunks too. */
static u32 g_idx_seen[64];

static void record_worker_index(void* ctx, u64 begin, u64 end) {
    (void)ctx;
    u32 w = pool_worker_index();
    if (w < 64u) g_idx_seen[w]++;
    for (u64 r = begin; r < end; r++) g_visits[r]++;
}

TEST(pool_worker_index_is_in_range_and_the_caller_is_zero) {
    ASSERT_EQ_U32(pool_worker_index(), 0u); /* no pool: the caller is worker 0 */
    memset(g_idx_seen, 0, sizeof g_idx_seen);
    memset(g_visits, 0, sizeof g_visits);
    pool_start(6u);
    ASSERT_EQ_U32(pool_worker_index(), 0u); /* the dispatching thread, always 0 */
    /* Enough units, finely chunked, that every worker gets several claims --
     * one dispatch of 6 units could legitimately be swallowed by one fast
     * worker and prove nothing about the others. */
    for (u32 round = 0; round < 20u; round++) {
        pool_for_units(N_OUT, 1u, record_worker_index, NULL);
    }
    pool_stop();
    u32 out_of_range = 0, distinct = 0;
    for (u32 w = 0; w < 64u; w++) {
        if (g_idx_seen[w] == 0u) continue;
        distinct++;
        if (w >= 6u) out_of_range++;
    }
    ASSERT_EQ_U32(out_of_range, 0u);
    /* Not "exactly 6": the scheduler may leave a worker unclaimed on a busy
     * box, and asserting on that would make this test flaky for a reason that
     * has nothing to do with the property. More than one is what proves the
     * indices are not all colliding at zero. */
    ASSERT_TRUE(distinct > 1u);
}

/* The placement trace counts CHUNKS, and the count must be the dispatch's own
 * chunk count -- every chunk once, on some row, and nothing when the trace is
 * off. The per-LP split it prints cannot be asserted (it is the scheduler's
 * decision, which is the whole point of the instrument), but a table whose
 * total disagrees with the dispatch would be reporting placement of work that
 * did not happen. setenv is POSIX and MinGW lacks it; putenv is what MinGW
 * has and glibc hides it under -std=c11, hence the split. */
static void set_trace_env(const char* value) {
#ifdef _WIN32
    char buf[64];
    snprintf(buf, sizeof buf, "STOKE_POOL_TRACE=%s", value);
    _putenv(buf);
#else
    if (value[0]) setenv("STOKE_POOL_TRACE", value, 1);
    else          unsetenv("STOKE_POOL_TRACE");
#endif
}

TEST(pool_trace_counts_every_chunk_exactly_once) {
    pool_stop();
    set_trace_env("3600"); /* on, but the periodic report never fires */
    pool_start(4u);
    memset(g_visits, 0, sizeof g_visits);
    /* 1000 units at chunk 8 = 125 chunks, spread over four claimants. */
    pool_for_units(N_OUT, 8u, count_visit, NULL);
    u64 total = 0;
    for (u32 w = 0; w < 4u; w++) total += pool_trace_chunks(w);
    ASSERT_EQ_I64(total, 125);
    ASSERT_EQ_U32(pool_trace_report(stderr), 4u);
    pool_stop();
    ASSERT_EQ_I64(pool_trace_chunks(0), 0); /* freed with the pool */

    set_trace_env("");
    pool_start(4u);
    pool_for_units(N_OUT, 8u, count_visit, NULL);
    ASSERT_EQ_I64(pool_trace_chunks(0), 0);
    ASSERT_EQ_U32(pool_trace_report(stderr), 0u);
    pool_stop();
}

TEST(pool_thread_count_reports_and_defaults) {
    pool_stop();
    ASSERT_EQ_U32(pool_threads(), 1u);
    pool_start(1u);
    ASSERT_EQ_U32(pool_threads(), 1u);   /* inert, no helpers created */
    pool_stop();
    pool_start(4u);
    ASSERT_EQ_U32(pool_threads(), 4u);
    pool_stop();
    pool_start(0u);                       /* 0 asks for the measured default */
    /* Pins the DEFAULT, so moving it is a deliberate act with a measurement
     * behind it rather than a quiet edit. See pool.h: 8, from stoke's own
     * sweep, deliberately not llama's 6. */
    ASSERT_EQ_U32(pool_threads(), 8u);
    pool_stop();
    ASSERT_EQ_U32(pool_threads(), 1u);
}
