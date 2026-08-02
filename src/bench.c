/* src/bench.c -- `stoke bench --ops`, the M3 per-op attribution instrument.
 *
 * See bench.h for WHY this exists and lands first. This file documents HOW,
 * and in particular the three methodology decisions that make the numbers
 * mean anything. Every one of them is a lesson M2 already paid for; they are
 * written down here because a microbench that gets them wrong does not fail
 * loudly, it just prints confident nonsense.
 *
 * ---------------------------------------------------------------------------
 * DECISION 1 -- THE MMAP IS WARMED BEFORE ANY TIMED REGION.
 *
 * The forward note ranks "page faults during timing" as culprit #5, and on
 * this box it is not a small effect: M2 measured the SAME prompt at 12.32
 * s/position on a quiet box and 20.22 s/position on a loaded one, purely
 * because the 4.3 GB of weights were or were not resident. A cold matvec
 * measures the SSD. A warm one measures the kernel. Only one of those is the
 * thing M3 is trying to optimize.
 *
 * So every tensor this file will touch is streamed once, into a volatile
 * sink, before the first timer starts. The sink matters: without it the read
 * is dead code and -O1 is entitled to delete the entire warm-up, which would
 * leave the harness looking correct while measuring exactly the thing it
 * claims to have excluded.
 *
 * ---------------------------------------------------------------------------
 * DECISION 2 -- EVERY OP IS REPORTED RELATIVE TO A REFERENCE STREAM MEASURED
 * IN THE SAME RUN, not only against a hardcoded ceiling.
 *
 * This is the important one, and it is a direct answer to the single most
 * expensive recurring failure of M2: *the rate is a property of box state*.
 * An absolute "14.2 GB/s" is uninterpretable on its own -- it could be a slow
 * kernel, or it could be a browser eating the memory bus. M2 lost days to
 * exactly that ambiguity, twice concluding something about the code from a
 * number that was really about the machine.
 *
 * The fix is to measure, in the same run and under the same conditions, a
 * plain sequential f32 read over a buffer far larger than L3. That number is
 * this machine's practical streaming bandwidth *right now*. Every op is then
 * reported BOTH as absolute GB/s and as a percentage of that reference.
 *
 *   - The absolute number moves when the box is busy.
 *   - The RATIO does not (both numerator and denominator degrade together).
 *
 * So the ratio column is the one that survives being run on a machine someone
 * is using, and the ratio column is the one the attribution ladder should be
 * read from. An op at ~100% of reference is bandwidth-bound and no amount of
 * SIMD will help it; an op far below reference has headroom and is worth
 * attacking. That verdict is now robust to conditions rather than hostage to
 * them.
 *
 * The hardcoded ~60 GB/s from the forward note is still printed, but as
 * context only -- it is a spec figure for the platform, not a measurement of
 * this box on this afternoon.
 *
 * ---------------------------------------------------------------------------
 * DECISION 3 -- ADAPTIVE REPS, AND min/median/max IS REPORTED, NEVER A MEAN.
 *
 * Rep count is chosen per op from a pilot measurement so each op gets roughly
 * the same total timed duration regardless of whether it costs 2 us or 200
 * ms. A fixed rep count either drowns the fast ops in timer noise or makes
 * the slow ones take minutes.
 *
 * The spread is printed because a mean hides the thing you most need to see.
 * A scheduler preemption or an interrupt storm shows up as max >> median
 * while the median stays honest; a genuinely degraded box shows up as the
 * median itself moving. Those are different diagnoses and a single averaged
 * number cannot distinguish them. (M0's protocol already uses median-of-3
 * for the same reason; this is that rule applied per op.)
 *
 * ---------------------------------------------------------------------------
 * WHAT THE BYTE COUNTS MEAN. For the matvecs -- the ops that actually
 * dominate a token -- traffic is the weight matrix itself, streamed once per
 * call, and those are genuinely bandwidth-bound. For the small elementwise
 * ops (rmsnorm, rope, swiglu, softmax) the operands are L1/L2-resident and a
 * GB/s figure is close to meaningless; read the us column for those. Both are
 * printed rather than one being suppressed, because which one is meaningful
 * is a property of the op, and hiding a column invites the reader to assume
 * the other one is authoritative everywhere.
 */
#include "common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "gguf.h"
#include "kernels/kernels.h"

/* Practical streaming ceiling quoted by the forward note for this class of
 * box. Printed as CONTEXT ONLY -- Decision 2 explains why the same-run
 * reference, not this constant, is what the ratio column uses. */
#define BENCH_SPEC_CEILING_GBS 60.0

/* Reference buffer must be far larger than L3 (30 MB on this i5-13450HX) so
 * the read is genuinely a memory-bus measurement and not an L3 hit rate. */
#define BENCH_REF_BYTES (256u * 1024u * 1024u)

/* BENCH_TARGET_MS / BENCH_MIN_REPS / BENCH_MAX_REPS live in bench.h: they are
 * the contract of bench_reps_for(), not private tuning, and the sample-count
 * floor carries the reasoning for why it is 11 and not 3. */

/* Sink for reads whose results are otherwise unused. volatile so the
 * compiler must perform them -- see Decision 1. */
static volatile f32 g_sink;

typedef BenchTiming Timing;

static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a;
    double y = *(const double*)b;
    return (x > y) - (x < y);
}

BenchTiming bench_spread(double* samples, u32 n) {
    if (n == 0) {
        fatal("bench_spread: n == 0 -- no samples to summarize");
    }
    qsort(samples, (size_t)n, sizeof(double), cmp_double);
    BenchTiming t;
    t.min_ms = samples[0];
    t.med_ms = samples[n / 2];
    t.max_ms = samples[n - 1];
    t.reps   = n;
    return t;
}

double bench_gbs(u64 bytes, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    return ((double)bytes / 1e9) / (ms / 1000.0);
}

u32 bench_reps_for(double pilot_ms) {
    /* Negated comparison, not `pilot_ms <= 0.0`: this way a NaN pilot takes
     * the max-reps branch instead of falling through to a chain in which
     * every comparison against NaN is false. */
    if (!(pilot_ms > 0.0)) {
        return BENCH_MAX_REPS;
    }
    double want = BENCH_TARGET_MS / pilot_ms;
    if (want < (double)BENCH_MIN_REPS) {
        return BENCH_MIN_REPS;
    }
    if (want > (double)BENCH_MAX_REPS) {
        return BENCH_MAX_REPS;
    }
    return (u32)want;
}

/* Runs fn(ctx) enough times to fill BENCH_TARGET_MS (or exactly
 * `reps_override` times), returning the spread. One untimed warm-up call
 * happens first: it faults in this op's own working set and lets any
 * first-call branch prediction settle, so rep 1 is not systematically the
 * slowest sample. */
static Timing time_op(void (*fn)(void*), void* ctx, u32 reps_override) {
    fn(ctx);

    double t0 = now_ms();
    fn(ctx);
    double pilot_ms = now_ms() - t0;

    u32 reps = (reps_override != 0) ? reps_override : bench_reps_for(pilot_ms);

    double* samples = (double*)xmalloc((size_t)reps * sizeof(double));
    for (u32 i = 0; i < reps; i++) {
        double a = now_ms();
        fn(ctx);
        samples[i] = now_ms() - a;
    }
    Timing t = bench_spread(samples, reps);
    free(samples);
    return t;
}

/* ---- the reference stream (Decision 2) ---- */

typedef struct {
    const f32* buf;
    u64        n;
} RefCtx;

static void run_ref_stream(void* p) {
    RefCtx* c = (RefCtx*)p;
    f32 acc = 0.0f;
    for (u64 i = 0; i < c->n; i++) {
        acc += c->buf[i];
    }
    g_sink = acc;
}

/* ---- op contexts ---- */

typedef struct {
    const GgufTensor* w;
    const f32*        x;
    f32*              y;
    u64               n_in;
    u64               n_out;
} MatvecCtx;

static void run_matvec(void* p) {
    MatvecCtx* c = (MatvecCtx*)p;
    matvec(c->w, c->x, c->n_in, c->y, c->n_out);
}

typedef struct {
    const void* row;
    const f32*  x;
    u64         n;
} DotCtx;

static void run_dot_q8_0(void* p) {
    DotCtx* c = (DotCtx*)p;
    g_sink = vec_dot_q8_0_f32(c->row, c->x, c->n);
}

typedef struct {
    const void* row;
    f32*        out;
    u64         n;
} DequantCtx;

static void run_dequant_q8_0(void* p) {
    DequantCtx* c = (DequantCtx*)p;
    dequant_row_q8_0(c->row, c->n, c->out);
}

typedef struct {
    const f32* x;
    const f32* w;
    f32*       y;
    u64        n;
} RmsCtx;

static void run_rmsnorm(void* p) {
    RmsCtx* c = (RmsCtx*)p;
    rmsnorm(c->x, c->w, c->n, 1e-6f, c->y);
}

typedef struct {
    f32* v;
    u64  head_dim;
} RopeCtx;

static void run_rope(void* p) {
    RopeCtx* c = (RopeCtx*)p;
    rope_neox(c->v, c->head_dim, 1024, 1000000.0f);
}

typedef struct {
    f32* x;
    u64  n;
} SoftmaxCtx;

static void run_softmax(void* p) {
    SoftmaxCtx* c = (SoftmaxCtx*)p;
    softmax(c->x, c->n);
}

typedef struct {
    const f32* g;
    const f32* u;
    f32*       h;
    u64        n;
} SwigluCtx;

static void run_swiglu(void* p) {
    SwigluCtx* c = (SwigluCtx*)p;
    swiglu(c->g, c->u, c->n, c->h);
}

typedef struct {
    const f32* logits;
    u64        n;
} ArgmaxCtx;

static void run_argmax(void* p) {
    ArgmaxCtx* c = (ArgmaxCtx*)p;
    g_sink = (f32)argmax(c->logits, c->n);
}

/* ---- reporting ---- */

static double g_ref_gbs;   /* same-run reference stream, set before any row */

/* ---- machine-readable capture (--jsonl) ----
 *
 * ADDITIVE BY DESIGN: --jsonl changes not one character of the human table.
 * It captures each row as it is printed and emits a single JSON object as the
 * LAST line, so a consumer takes the final line and a human still reads the
 * same output they always did. The alternative -- guarding twenty printf
 * sites behind a mode flag -- would have churned every line of a freshly
 * verified reporting path to add a feature that needs none of them changed.
 *
 * This exists because `bench/history.jsonl` is a STANDING requirement (spec
 * "Perf regression tracking"; forward note "history.jsonl discipline"): every
 * green commit touching a hot path appends a row, and regressions are caught
 * by diffing history rather than by remembering. A format that a human must
 * retype into JSON would not survive that cadence. */
typedef struct {
    const char* op;
    char        shape[32];
    u64         bytes;
    Timing      t;
} CapturedRow;

#define BENCH_MAX_CAPTURED 32u

static int         g_jsonl;
static CapturedRow g_captured[BENCH_MAX_CAPTURED];
static u32         g_captured_n;

static void row(const char* op, const char* shape, u64 bytes, Timing t) {
    double g   = bench_gbs(bytes, t.med_ms);
    double pct = (g_ref_gbs > 0.0) ? (100.0 * g / g_ref_gbs) : 0.0;
    printf("  %-22s %-18s %10.1f %9.1f %9.1f %8.2f %7.1f%%  %6u\n",
           op, shape, t.med_ms * 1000.0, t.min_ms * 1000.0, t.max_ms * 1000.0,
           g, pct, t.reps);

    if (g_jsonl) {
        if (g_captured_n >= BENCH_MAX_CAPTURED) {
            /* Silently dropping rows would make history.jsonl quietly
             * incomplete, which is worse than not writing it at all. */
            fatal("bench: --jsonl capture overflow (%u rows, cap %u) -- raise "
                  "BENCH_MAX_CAPTURED",
                  g_captured_n + 1u, BENCH_MAX_CAPTURED);
        }
        CapturedRow* c = &g_captured[g_captured_n++];
        c->op    = op;
        c->bytes = bytes;
        c->t     = t;
        snprintf(c->shape, sizeof(c->shape), "%s", shape);
    }
}

/* One JSON object on one line: the raw MEASUREMENTS only. Provenance (commit,
 * binary hash, box state, cross-run medians) is deliberately NOT here -- this
 * process cannot know its own git commit, and a benchmark that guesses its
 * provenance writes confident falsehoods into a file whose entire purpose is
 * historical comparison. bench/append_history.py supplies that layer. */
static void emit_jsonl(const char* model_path, double ref_gbs, Timing rt) {
    printf("{\"kind\":\"ops\",\"model\":\"%s\",\"ref_gbs\":%.4f,"
           "\"ref_med_ms\":%.4f,\"ref_min_ms\":%.4f,\"ref_max_ms\":%.4f,"
           "\"ref_reps\":%u,\"ops\":[",
           model_path, ref_gbs, rt.med_ms, rt.min_ms, rt.max_ms, rt.reps);
    for (u32 i = 0; i < g_captured_n; i++) {
        const CapturedRow* c   = &g_captured[i];
        double             g   = bench_gbs(c->bytes, c->t.med_ms);
        double             pct = (ref_gbs > 0.0) ? (100.0 * g / ref_gbs) : 0.0;
        printf("%s{\"op\":\"%s\",\"shape\":\"%s\",\"bytes\":%llu,"
               "\"med_us\":%.3f,\"min_us\":%.3f,\"max_us\":%.3f,"
               "\"gbs\":%.4f,\"pct_ref\":%.3f,\"reps\":%u}",
               i ? "," : "", c->op, c->shape, (unsigned long long)c->bytes,
               c->t.med_ms * 1000.0, c->t.min_ms * 1000.0,
               c->t.max_ms * 1000.0, g, pct, c->t.reps);
    }
    printf("]}\n");
}

static void header(void) {
    printf("  %-22s %-18s %10s %9s %9s %8s %8s  %6s\n",
           "op", "shape", "med us", "min us", "max us", "GB/s", "%ref",
           "reps");
    printf("  %-22s %-18s %10s %9s %9s %8s %8s  %6s\n",
           "----------------------", "------------------", "----------",
           "---------", "---------", "--------", "--------", "------");
}

/* Faults every page of a tensor into residency. Decision 1.
 *
 * ONE TOUCH PER PAGE, not per byte. The goal is page-cache residency, and a
 * single read anywhere in a 4 KiB page faults the whole page in -- so reading
 * every byte does 4096x the work for identical effect. That is not merely
 * wasteful: token_embd.weight alone is ~400 MB, and a byte-at-a-time volatile
 * loop over it (volatile forbids vectorization, by design) turns the warm-up
 * into its own multi-second cost, which is a strange thing for the step whose
 * entire job is to stop slow I/O from contaminating a measurement.
 *
 * The final byte is touched explicitly because nbytes is not generally a
 * multiple of the page size, and the last partial page is exactly the one a
 * stride loop skips.
 *
 * volatile is load-bearing: without it this whole function is dead code and
 * -O1 may delete it, leaving a harness that still prints "warmed" while
 * measuring cold faults. */
#define BENCH_PAGE_BYTES 4096u

static void warm_tensor(const GgufTensor* t) {
    const volatile u8* p = (const volatile u8*)t->data;
    u32 acc = 0;
    if (t->nbytes == 0) {
        return;
    }
    for (u64 i = 0; i < t->nbytes; i += BENCH_PAGE_BYTES) {
        acc += p[i];
    }
    acc += p[t->nbytes - 1];
    g_sink = (f32)acc;
}

static const GgufTensor* need(const GgufFile* f, const char* name) {
    return gguf_tensor_by_name(f, name);   /* fatal inside if absent */
}

/* One projection row of the table: matvec over a real weight tensor. */
static void bench_proj(const GgufFile* f, const char* tensor_name,
                       const char* label, const f32* x_pool, f32* y_pool,
                       u32 reps_override) {
    const GgufTensor* w = need(f, tensor_name);
    if (w->n_dims < 2) {
        fatal("bench: %s has n_dims=%u, expected >= 2", tensor_name,
              w->n_dims);
    }
    u64 n_in  = w->dims[0];
    u64 n_out = w->dims[1];

    MatvecCtx c = { w, x_pool, y_pool, n_in, n_out };
    Timing    t = time_op(run_matvec, &c, reps_override);

    char shape[32];
    snprintf(shape, sizeof(shape), "%llu x %llu",
             (unsigned long long)n_in, (unsigned long long)n_out);
    /* Weight bytes dominate; the f32 operands are counted for honesty but
     * are three orders of magnitude smaller. */
    row(label, shape, w->nbytes + (n_in + n_out) * sizeof(f32), t);
}

void bench_ops(const char* model_path, u32 reps_override, int jsonl) {
    g_jsonl      = jsonl;
    g_captured_n = 0;

    GgufFile* f = gguf_open(model_path);

    /* Shapes come from the file, never from a constant: the whole point is
     * that these are the REAL model's dimensions. */
    const GgufTensor* embd = need(f, "token_embd.weight");
    const GgufTensor* q    = need(f, "blk.0.attn_q.weight");
    const GgufTensor* ffn  = need(f, "blk.0.ffn_gate.weight");
    u64 n_embd  = q->dims[0];
    u64 n_vocab = embd->dims[1];
    u64 n_ff    = ffn->dims[1];

    printf("stoke bench --ops\n");
    printf("  model    : %s\n", model_path);
    printf("  n_embd   : %llu\n", (unsigned long long)n_embd);
    printf("  n_ff     : %llu\n", (unsigned long long)n_ff);
    printf("  n_vocab  : %llu\n", (unsigned long long)n_vocab);
    printf("\n");

    /* ---- reference stream, BEFORE anything else (Decision 2) ---- */
    u64  ref_n   = BENCH_REF_BYTES / sizeof(f32);
    f32* ref_buf = (f32*)xmalloc((size_t)ref_n * sizeof(f32));
    for (u64 i = 0; i < ref_n; i++) {
        ref_buf[i] = (f32)(i & 0xFF);
    }
    RefCtx rc = { ref_buf, ref_n };
    Timing rt = time_op(run_ref_stream, &rc, reps_override ? 3u : 0u);
    g_ref_gbs = bench_gbs(ref_n * sizeof(f32), rt.med_ms);

    printf("  REFERENCE STREAM (this run, this box -- the ratio denominator)\n");
    printf("    sequential f32 read over %u MB : %.2f GB/s"
           "  (med %.2f ms, min %.2f, max %.2f)\n",
           BENCH_REF_BYTES / (1024u * 1024u), g_ref_gbs, rt.med_ms, rt.min_ms,
           rt.max_ms);
    printf("    forward-note spec ceiling      : %.1f GB/s"
           "  -- CONTEXT ONLY, not the ratio base\n", BENCH_SPEC_CEILING_GBS);
    printf("    read the %%ref column, not GB/s, when the box is not quiesced.\n");
    printf("\n");

    /* ---- warm every tensor we are about to time (Decision 1) ---- */
    warm_tensor(embd);
    for (u64 i = 0; i < gguf_tensor_count(f); i++) {
        const GgufTensor* t = gguf_tensor(f, i);
        if (strncmp(t->name, "blk.0.", 6) == 0) {
            warm_tensor(t);
        }
    }

    /* ---- scratch ---- */
    u64  x_len = (n_ff > n_embd ? n_ff : n_embd);
    u64  y_len = (n_vocab > n_ff ? n_vocab : n_ff);
    f32* x     = (f32*)xmalloc((size_t)x_len * sizeof(f32));
    f32* x2    = (f32*)xmalloc((size_t)x_len * sizeof(f32));
    f32* y     = (f32*)xmalloc((size_t)y_len * sizeof(f32));
    for (u64 i = 0; i < x_len; i++) {
        x[i]  = 0.01f * (f32)((i % 97) - 48);
        x2[i] = 0.01f * (f32)((i % 89) - 44);
    }
    for (u64 i = 0; i < y_len; i++) {
        y[i] = 0.0f;
    }

    printf("  PROJECTIONS (bandwidth-bound -- read GB/s and %%ref)\n");
    header();
    bench_proj(f, "blk.0.attn_q.weight",      "matvec attn_q",   x, y, reps_override);
    bench_proj(f, "blk.0.attn_k.weight",      "matvec attn_k",   x, y, reps_override);
    bench_proj(f, "blk.0.attn_v.weight",      "matvec attn_v",   x, y, reps_override);
    bench_proj(f, "blk.0.attn_output.weight", "matvec attn_out", x, y, reps_override);
    bench_proj(f, "blk.0.ffn_gate.weight",    "matvec ffn_gate", x, y, reps_override);
    bench_proj(f, "blk.0.ffn_up.weight",      "matvec ffn_up",   x, y, reps_override);
    bench_proj(f, "blk.0.ffn_down.weight",    "matvec ffn_down", x, y, reps_override);
    /* The logits matvec is culprit #1 on the forward note's ranked list and
     * is by far the largest single op in a token -- it gets its own line and
     * should be read first. */
    bench_proj(f, "token_embd.weight",        "matvec LOGITS",   x, y, reps_override);
    printf("\n");

    printf("  PRIMITIVES\n");
    header();
    {
        DotCtx c = { q->data, x, n_embd };
        Timing t = time_op(run_dot_q8_0, &c, reps_override);
        char s[32];
        snprintf(s, sizeof(s), "n=%llu", (unsigned long long)n_embd);
        row("vec_dot_q8_0_f32", s,
            (n_embd / KERNELS_Q8_0_BLOCK_ELEMS) * KERNELS_Q8_0_BLOCK_BYTES
                + n_embd * sizeof(f32), t);
    }
    {
        DequantCtx c = { q->data, y, n_embd };
        Timing     t = time_op(run_dequant_q8_0, &c, reps_override);
        char       s[32];
        snprintf(s, sizeof(s), "n=%llu", (unsigned long long)n_embd);
        row("dequant_row_q8_0", s,
            (n_embd / KERNELS_Q8_0_BLOCK_ELEMS) * KERNELS_Q8_0_BLOCK_BYTES
                + n_embd * sizeof(f32), t);
    }
    printf("\n");

    printf("  ELEMENTWISE (cache-resident -- read us, NOT GB/s)\n");
    header();
    {
        RmsCtx c = { x, x2, y, n_embd };
        Timing t = time_op(run_rmsnorm, &c, reps_override);
        char   s[32];
        snprintf(s, sizeof(s), "n=%llu", (unsigned long long)n_embd);
        row("rmsnorm", s, n_embd * 3u * sizeof(f32), t);
    }
    {
        u64    head_dim = 128;
        RopeCtx c = { y, head_dim };
        Timing  t = time_op(run_rope, &c, reps_override);
        char    s[32];
        snprintf(s, sizeof(s), "head_dim=%llu",
                 (unsigned long long)head_dim);
        row("rope_neox", s, head_dim * 2u * sizeof(f32), t);
    }
    {
        /* Softmax over a mid-length attention row: the shape it actually
         * sees in a decode step at a realistic context. */
        u64        n = 1024;
        SoftmaxCtx c = { y, n };
        Timing     t = time_op(run_softmax, &c, reps_override);
        char       s[32];
        snprintf(s, sizeof(s), "n=%llu (ctx row)", (unsigned long long)n);
        row("softmax", s, n * 2u * sizeof(f32), t);
    }
    {
        SwigluCtx c = { x, x2, y, n_ff };
        Timing    t = time_op(run_swiglu, &c, reps_override);
        char      s[32];
        snprintf(s, sizeof(s), "n=%llu", (unsigned long long)n_ff);
        row("swiglu", s, n_ff * 3u * sizeof(f32), t);
    }
    {
        ArgmaxCtx c = { y, n_vocab };
        Timing    t = time_op(run_argmax, &c, reps_override);
        char      s[32];
        snprintf(s, sizeof(s), "n=%llu (vocab)",
                 (unsigned long long)n_vocab);
        row("argmax", s, n_vocab * sizeof(f32), t);
    }
    printf("\n");

    printf("  Reading this table (forward note's ranked culprits):\n");
    printf("    * %%ref near 100  -> bandwidth-bound; SIMD will not help,\n");
    printf("                        parallelism and layout will.\n");
    printf("    * %%ref far below -> headroom; this op is compute or\n");
    printf("                        latency bound and is worth attacking.\n");
    printf("    * matvec LOGITS is culprit #1 -- largest single op per token.\n");
    printf("    * max >> med     -> interference during the run, not a slow\n");
    printf("                        kernel. Re-run on a quiesced box.\n");

    if (g_jsonl) {
        emit_jsonl(model_path, g_ref_gbs, rt);
    }

    free(x);
    free(x2);
    free(y);
    free(ref_buf);
    gguf_close(f);
}
