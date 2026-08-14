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
 * read over a buffer far larger than L3. That number is this machine's
 * practical streaming bandwidth *right now*. Every op is then reported BOTH
 * as absolute GB/s and as a percentage of that reference.
 *
 *   - The absolute number moves when the box is busy.
 *   - The RATIO does not (both numerator and denominator degrade together).
 *
 * ▶▶ THE DENOMINATOR WAS WRONG FROM K1 UNTIL 2026-08-03, AND EVERY %ref THIS
 * PROJECT RECORDED BEFORE THAT DATE IS AGAINST A FIGURE ~2.15x TOO LOW.
 *
 * The original reference stream was `f32 acc = 0; for(..) acc += buf[i];`.
 * That is a SERIAL FLOATING-POINT REDUCTION: every add depends on the
 * previous one, and gcc may not reassociate it without -ffast-math, so it
 * cannot be vectorized at ANY optimization level. Its rate is f32-add
 * LATENCY, not memory bandwidth. Measured on this box (tests/tooling/
 * bwprobe.c, same 256 MB buffer): serial 6.7-7.4 GB/s at both -O1 and -O3,
 * while split8_f32 and xor_u64 -- the same reads with the dependency chain
 * broken -- reach 13.9-16.6.
 *
 * So the denominator is now max(split8_f32, xor_u64), both measured in this
 * same run. They trade places between runs, and taking the larger is the
 * correct reading of a quantity that means "at least this much bandwidth
 * exists". serial_f32 is still measured and printed, but ONLY as a labelled
 * diagnostic: it is a decent FP-latency canary, and it is the conversion key
 * that makes pre-2026-08-03 rows re-interpretable. It must never again be a
 * denominator.
 *
 * ▶ HOW THE ERROR SURVIVED, which is the part worth carrying forward. The
 * serial stream's INVARIANCE across -O levels (7.63/7.77/7.84) was cited as
 * an internal control: "it did not speed up like everything else, therefore
 * it must already be at DRAM, therefore it is a legitimate denominator."
 * The invariance was the SYMPTOM, not the control. "It didn't change" is
 * produced equally by *this is a stable reference* and by *this is measuring
 * something the change cannot affect*. AN INVARIANT CITED AS A CONTROL MUST
 * HAVE ITS INVARIANCE EXPLAINED BEFORE IT IS TRUSTED, NOT MERELY OBSERVED.
 * Same family as BENCH_MIN_REPS' max/med = 1.33 reading as healthy while the
 * median was 2.6x wrong: a statistic is most dangerous where it is blind,
 * because that is where it looks calmest.
 *
 * ▶ AND %ref STILL DOES NOT MEAN "FRACTION OF WHAT THIS OP COULD ACHIEVE".
 * A read-bandwidth ceiling is an upper bound on the machine, not a target a
 * real kernel can reach. Measured 2026-08-03, interleaved and pinned:
 * llama.cpp's own production AVX2+VNNI decode reaches ~68% of this
 * denominator at one thread (2.46 tok/s), against ~32% for stoke's scalar
 * -O3 (1.12 tok/s). So a real kernel tops out well under 100%, and %ref
 * answers only "how much of the machine's read bandwidth is this op
 * moving" -- never "how much is left on the table".
 *
 * ▶ AND THE REFERENT ITSELF MUST BE RE-MEASURED, NOT CITED. The same llama
 * command read 1.26 tok/s earlier the same day and 2.46 a few hours later,
 * with the flag difference explicitly ruled out (`-r 3` gives 2.46 now,
 * `-r 1` gives 2.39). A referent taken from another session is not a
 * referent, it is a number. Compare against llama measured in the SAME
 * session as the stoke figure, and interleave the two if the ratio matters.
 *
 * ▶▶ THE COMPARABILITY RULE (Fable K3 ruling, 2026-08-03, section 4 --
 * `docs/superpowers/memos/2026-08-03-k3-simd-headroom-RULING-FABLE.md`,
 * which resolved the reopening and is the binding authority here).
 *
 *   Two performance figures may be compared -- ratio, difference, or
 *   percentage -- only if measured in the same session under the same
 *   recorded conditions, interleaved whenever both sides are runnable. A
 *   cross-session comparison is not evidence; it is a hypothesis, and it
 *   must be re-measured before it may be written into a memo, a ROADMAP
 *   line, a comment, or a ledger claim.
 *
 *   Corollary A. Rep-to-rep stability certifies precision WITHIN a window
 *   and says nothing about that window's validity. 1.26 +/- 0.01 and
 *   2.46 +/- 0.04 were both internally excellent.
 *   Corollary B. Subtraction is comparison. A difference of
 *   differently-conditioned measurements is a measurement of the conditions.
 *   Corollary C. A field name is not a basis. `ref_gbs` means the serial
 *   stream in schema-1 rows and max(split8, xor) in schema-2 rows; a
 *   percentage must name its denominator's stream AND its window.
 *
 * The rule is mechanised in two places, and this file is deliberately NOT
 * one of them: `bench/append_history.py` brackets each run with a box sample
 * and grades the row gate/ambient, and `bench/referent_interleave.py` runs
 * the A,B,A,B against llama and emits a paired record. This bench stays free
 * of any dependency on the vendor tree -- it is the self-instrument of an
 * embeddable engine and must run where no llama.cpp exists, and an
 * instrument that executes its own comparison target conflates measuring
 * with refereeing. The attribution ladder may cite paired records only.
 *
 * WHY THE NUMBERS TWO PARAGRAPHS UP WERE EVER IN DOUBT, recorded because it
 * is the concrete shape of the abstract rule: a game had been resident since
 * 09:57 that day, averaging 5.08 cores, spanning every contested measurement
 * window of two sessions. Nobody had recorded the process table. The box
 * state was not exotic and not thermal -- it was another program, and the
 * only reason it took forensics to find is that nothing was writing it down.
 *
 * The hardcoded ~60 GB/s from the forward note is still printed, but as
 * context only -- it is a spec figure for the platform, not a measurement of
 * this box on this afternoon. It is also wrong for this box by ~4x, which is
 * its own argument for measuring rather than quoting.
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

/* Integer sink, same rationale: the u64 reference variant must not have its
 * whole loop deleted as dead code. Separate from g_sink because routing a
 * u64 through an f32 would silently lose bits and invite a future reader to
 * think the value mattered. */
static volatile u64 g_sink_u;

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

double bench_ref_denominator(double split8_gbs, double xor_gbs) {
    return (split8_gbs > xor_gbs) ? split8_gbs : xor_gbs;
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
 * slowest sample.
 *
 * ▶ CONTRACT, AND IT IS A REAL CONSTRAINT ON WHAT YOU MAY BENCH HERE:
 * nothing is re-initialized between reps, so an IN-PLACE op measures its
 * own OUTPUT on every rep after the first. Two ops below are in-place --
 * `softmax` and `rope_neox`, both on the y pool -- and for those two it is
 * harmless: rope's cost comes from position-derived sin/cos and does not
 * depend on the data at all, and softmax's real input (an attention row)
 * and its degenerate one (a probability vector) both land on expf's fast
 * path. Checked 2026-08-03, and the figures stand.
 *
 * It is recorded because it is LATENT, not because it currently bites.
 * The 2026-08-02 K1 pool bug proved the same point from the other side:
 * swiglu's baseline figure was 2.9x too slow purely because its operands
 * were wrong, since expf has a magnitude-dependent slow path. Before
 * adding any in-place op whose cost can depend on its operand values,
 * either restore the input between reps outside the timed region or give
 * the op a scratch destination -- do not assume rep N sees rep 1's
 * distribution. */
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

/* ---- the reference streams (Decision 2) ----
 *
 * Three variants over the SAME buffer. The first two are candidates for the
 * denominator and the third is a diagnostic; see Decision 2 for why that
 * split exists and what it cost to discover.
 *
 * These are transcribed from tests/tooling/bwprobe.c, which is the
 * standalone probe that established the correction. Keeping them
 * structurally identical is deliberate: if this file's figures and the
 * probe's ever disagree, the difference is the harness and not the box. */

typedef struct {
    const f32* buf;
    u64        n;
} RefCtx;

/* Eight independent accumulators: the same f32 arithmetic as the old serial
 * stream with the dependency chain broken, so eight chains interleave and
 * the loop is bound by loads rather than by add latency. Still not
 * auto-vectorized (gcc may not reassociate f32 without -ffast-math), which
 * is the point -- this is an honest read rate, not a vectorization trick. */
static void run_ref_split8(void* p) {
    RefCtx* c = (RefCtx*)p;
    f32 a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (u64 i = 0; i + 8 <= c->n; i += 8) {
        a0 += c->buf[i];     a1 += c->buf[i + 1];
        a2 += c->buf[i + 2]; a3 += c->buf[i + 3];
        a4 += c->buf[i + 4]; a5 += c->buf[i + 5];
        a6 += c->buf[i + 6]; a7 += c->buf[i + 7];
    }
    g_sink = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
}

/* XOR over the same bytes read as u64: 1-cycle latency and associative, so
 * nothing binds but the load ports. This is the closest thing here to a pure
 * read-bandwidth measurement. */
static void run_ref_xor(void* p) {
    RefCtx*    c = (RefCtx*)p;
    const u64* q = (const u64*)(const void*)c->buf;
    u64        acc = 0;
    for (u64 i = 0; i < c->n / 2; i++) {
        acc ^= q[i];
    }
    g_sink_u = acc;
}

/* THE OLD REFERENCE STREAM. Kept, measured, printed -- and NOT a
 * denominator. It reports f32-add latency (~7 GB/s here, unmoved by -O),
 * which is roughly 2.15x under this box's real read bandwidth. Two reasons
 * it stays: it is a cheap canary for FP-latency changes, and it is the
 * conversion key that lets pre-2026-08-03 %ref figures be re-interpreted
 * instead of merely discarded. */
static void run_ref_serial(void* p) {
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
    void*             aq;   /* M3 activation-quant scratch, owned by caller */
} MatvecCtx;

static void run_matvec(void* p) {
    MatvecCtx* c = (MatvecCtx*)p;
    matvec(c->w, c->x, c->n_in, c->y, c->n_out, c->aq);
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
u64 bench_json_escape(const char* in, char* out, u64 cap) {
    static const char HEX[] = "0123456789abcdef";
    u64               w     = 0;

    /* Writes `n` bytes only if they fit WITH the NUL; otherwise fatal. The
     * check is done before every write rather than once up front, because
     * the expansion factor depends on the input's contents. */
#define PUT(n)                                                                \
    do {                                                                      \
        if (w + (n) + 1u > cap) {                                             \
            fatal("bench: bench_json_escape -- need more than %llu bytes to " \
                  "escape a %llu-byte string; use BENCH_JSON_ESCAPE_MAX",     \
                  (unsigned long long)cap, (unsigned long long)strlen(in));   \
        }                                                                     \
    } while (0)

    for (const u8* p = (const u8*)in; *p; p++) {
        const char* esc = NULL;
        switch (*p) {
        case '"':  esc = "\\\""; break;
        case '\\': esc = "\\\\"; break;
        case '\b': esc = "\\b";  break;
        case '\f': esc = "\\f";  break;
        case '\n': esc = "\\n";  break;
        case '\r': esc = "\\r";  break;
        case '\t': esc = "\\t";  break;
        default:   break;
        }
        if (esc != NULL) {
            PUT(2);
            out[w++] = esc[0];
            out[w++] = esc[1];
        } else if (*p < 0x20) {
            /* Every other control character. Bytes >= 0x80 are passed
             * through untouched: they are UTF-8 continuation bytes of a path
             * the shell already gave us, and re-encoding them would corrupt
             * a non-ASCII filename rather than protect anything. */
            PUT(6);
            out[w++] = '\\';
            out[w++] = 'u';
            out[w++] = '0';
            out[w++] = '0';
            out[w++] = HEX[(*p >> 4) & 0xF];
            out[w++] = HEX[*p & 0xF];
        } else {
            PUT(1);
            out[w++] = (char)*p;
        }
    }
#undef PUT
    out[w] = '\0';
    return w;
}

typedef struct {
    double gbs;
    Timing t;
} RefResult;

static void emit_jsonl(const char* model_path, double ref_gbs,
                       RefResult split8, RefResult xor_u64, RefResult serial) {
    /* The model path is the only externally-supplied string in this object;
     * `op` is an internal literal and `shape` is snprintf'd from integers.
     * Both are escaped anyway -- the cost is nothing, and "safe because of
     * where it comes from" is exactly the reasoning that let the unescaped
     * path ship in the first place. */
    u64   cap = BENCH_JSON_ESCAPE_MAX(strlen(model_path));
    char* esc = (char*)xmalloc((size_t)cap);
    bench_json_escape(model_path, esc, cap);

    /* The build flags are a compile-time literal, but they are escaped like
     * any other string: they contain no JSON metacharacters today, and
     * "safe because of where it comes from" is precisely the reasoning that
     * let the unescaped model path ship. */
    u64   fcap = BENCH_JSON_ESCAPE_MAX(strlen(STOKE_BUILD_CFLAGS));
    char* fesc = (char*)xmalloc((size_t)fcap);
    bench_json_escape(STOKE_BUILD_CFLAGS, fesc, fcap);

    /* ref_gbs is the denominator (max of the two fast variants); each
     * variant is also recorded whole so a future reader can re-derive the
     * choice, and serial_f32 is recorded so pre-2026-08-03 rows remain
     * convertible. ref_med_ms/min/max/reps describe the DENOMINATOR's
     * timing, which is what they meant before the split. */
    printf("{\"kind\":\"ops\",\"model\":\"%s\",\"build_cflags\":\"%s\","
           "\"ref_gbs\":%.4f,\"ref_gbs_split8\":%.4f,\"ref_gbs_xor\":%.4f,"
           "\"ref_gbs_serial\":%.4f,"
           "\"ref_med_ms\":%.4f,\"ref_min_ms\":%.4f,\"ref_max_ms\":%.4f,"
           "\"ref_reps\":%u,\"ops\":[",
           esc, fesc, ref_gbs, split8.gbs, xor_u64.gbs, serial.gbs,
           (split8.gbs >= xor_u64.gbs ? split8.t : xor_u64.t).med_ms,
           (split8.gbs >= xor_u64.gbs ? split8.t : xor_u64.t).min_ms,
           (split8.gbs >= xor_u64.gbs ? split8.t : xor_u64.t).max_ms,
           (split8.gbs >= xor_u64.gbs ? split8.t : xor_u64.t).reps);
    free(esc);
    free(fesc);

    for (u32 i = 0; i < g_captured_n; i++) {
        const CapturedRow* c   = &g_captured[i];
        double             g   = bench_gbs(c->bytes, c->t.med_ms);
        double             pct = (ref_gbs > 0.0) ? (100.0 * g / ref_gbs) : 0.0;
        /* `op` is a pointer, so it gets a fixed generous bound rather than a
         * sizeof (which would size the pointer); `shape` is a real array.
         * bench_json_escape fatals rather than truncates if an op name ever
         * outgrows this -- the longest today is 16 characters. */
        char op[BENCH_JSON_ESCAPE_MAX(48u)];
        char sh[BENCH_JSON_ESCAPE_MAX(sizeof c->shape)];
        bench_json_escape(c->op, op, sizeof op);
        bench_json_escape(c->shape, sh, sizeof sh);
        printf("%s{\"op\":\"%s\",\"shape\":\"%s\",\"bytes\":%llu,"
               "\"med_us\":%.3f,\"min_us\":%.3f,\"max_us\":%.3f,"
               "\"gbs\":%.4f,\"pct_ref\":%.3f,\"reps\":%u}",
               i ? "," : "", op, sh, (unsigned long long)c->bytes,
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

    /* The integer core quantizes the activation inside every matvec call, so
     * that cost is measured here rather than hidden -- it is genuinely part
     * of what a token pays. Allocated per projection because n_in varies and
     * this is setup code, not a timed region. */
    /* ▶▶ THIS BUFFER WAS SIZED FOR Q8_0 ONLY AND OVERFLOWED ON EVERY K-QUANT
     * PROJECTION, crashing `bench --ops` with STATUS_HEAP_CORRUPTION on the
     * Q4_K_M model. Found 2026-08-15; bisecting the archived binaries shows it
     * back to the m3 gate archive, so it has been broken for two milestones.
     *
     * A K-quant matvec quantizes the activation to Q8_K (256 elements -> 292
     * bytes), not Q8_0 (32 -> 34). At n_in=2560 that is 2,920 bytes written
     * into a 2,720-byte allocation -- a 200-byte overflow every call.
     *
     * It survived because `bench --ops` is not in the test matrix: `make test`
     * runs tests/test_*.c, and test_bench.c exercises this file's internals
     * rather than the end-to-end mode against a real model. A harness nobody
     * runs is a harness nobody notices is broken -- and this one is the
     * instrument the ruling's C-SLOW branch prescribes for attribution.
     *
     * Sized for the LARGER of the two formats rather than dispatched on the
     * weight type: the cost is a few hundred bytes of setup allocation, and it
     * cannot go stale the next time a quant format is added. */
    u64 aq_q8_0 = (n_in / KERNELS_Q8_0_BLOCK_ELEMS) * KERNELS_Q8_0_BLOCK_BYTES;
    u64 aq_q8_k = (n_in / KERNELS_Q8_K_BLOCK_ELEMS) * KERNELS_Q8_K_BLOCK_BYTES;
    u64 aq_bytes = aq_q8_0 > aq_q8_k ? aq_q8_0 : aq_q8_k;
    void* aq = xmalloc((size_t)aq_bytes);

    MatvecCtx c = { w, x_pool, y_pool, n_in, n_out, aq };
    Timing    t = time_op(run_matvec, &c, reps_override);

    char shape[32];
    snprintf(shape, sizeof(shape), "%llu x %llu",
             (unsigned long long)n_in, (unsigned long long)n_out);
    /* Weight bytes dominate; the f32 operands are counted for honesty but
     * are three orders of magnitude smaller. */
    row(label, shape, w->nbytes + (n_in + n_out) * sizeof(f32), t);
    free(aq);
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
    RefCtx rc       = { ref_buf, ref_n };
    u32    ref_reps = reps_override ? 3u : 0u;
    u64    ref_bytes = ref_n * sizeof(f32);

    Timing rt_split8 = time_op(run_ref_split8, &rc, ref_reps);
    Timing rt_xor    = time_op(run_ref_xor, &rc, ref_reps);
    Timing rt_serial = time_op(run_ref_serial, &rc, ref_reps);

    double split8_gbs = bench_gbs(ref_bytes, rt_split8.med_ms);
    double xor_gbs    = bench_gbs(ref_bytes, rt_xor.med_ms);
    double serial_gbs = bench_gbs(ref_bytes, rt_serial.med_ms);

    g_ref_gbs = bench_ref_denominator(split8_gbs, xor_gbs);

    printf("  REFERENCE STREAMS (this run, this box -- %u MB buffer)\n",
           BENCH_REF_BYTES / (1024u * 1024u));
    printf("    split8_f32  (8 indep. f32 chains) : %.2f GB/s"
           "  (med %.2f ms, min %.2f, max %.2f)\n",
           split8_gbs, rt_split8.med_ms, rt_split8.min_ms, rt_split8.max_ms);
    printf("    xor_u64     (load-port bound)     : %.2f GB/s"
           "  (med %.2f ms, min %.2f, max %.2f)\n",
           xor_gbs, rt_xor.med_ms, rt_xor.min_ms, rt_xor.max_ms);
    printf("    -> %%ref DENOMINATOR = max of those : %.2f GB/s\n", g_ref_gbs);
    printf("\n");
    printf("    serial_f32  (DIAGNOSTIC, NOT the denominator) : %.2f GB/s\n",
           serial_gbs);
    printf("      f32-add LATENCY, not bandwidth -- it cannot vectorize at any\n");
    printf("      -O level. It WAS the denominator until 2026-08-03, so every\n");
    printf("      %%ref recorded before that date is ~%.2fx too high a ratio.\n",
           (serial_gbs > 0.0) ? (g_ref_gbs / serial_gbs) : 0.0);
    printf("    forward-note spec ceiling : %.1f GB/s -- CONTEXT ONLY, and ~4x\n",
           BENCH_SPEC_CEILING_GBS);
    printf("      wrong for this box, which is why the denominator is measured.\n");
    printf("    read the %%ref column, not GB/s, when the box is not quiesced.\n");
    printf("    %%ref is share of READ BANDWIDTH, not of what the op could reach:\n");
    printf("      llama.cpp's own VNNI decode reaches ~68%% of it, 1 thread pinned;\n");
    printf("      stoke's scalar -O3 ~32%%. Re-measure that referent in THIS\n");
    printf("      session before citing it -- it read 1.26 and 2.46 tok/s hours\n");
    printf("      apart on 2026-08-03 under an identical command.\n");
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
    /* ▶ THE CASTS TO i64 ARE LOAD-BEARING, and their absence was a real bug
     * that shipped in K1 and survived until M3's integer core caught it.
     *
     * `i` is u64, so `(i % 97) - 48` is UNSIGNED arithmetic: whenever
     * i % 97 < 48 it wraps to ~1.8e19 instead of going negative, and the
     * pool filled with values around 1.8e17 rather than the intended
     * [-0.48, 0.48].
     *
     * M2's f32 matvec never noticed -- a benchmark times its kernel, it does
     * not check the numbers, and large-but-normal floats cost exactly what
     * small ones do. The integer core is what made it visible: quantizing an
     * activation block whose absmax is 1.8e17 gives a scale of 1.5e15, and
     * narrowing THAT to f16 saturates to inf, which the dot kernel refuses.
     *
     * Worth stating plainly: a harness that only measures time can carry a
     * value bug indefinitely, because nothing it reports depends on the
     * values being right. It took a kernel that cares about magnitude to
     * surface it. */
    for (u64 i = 0; i < x_len; i++) {
        x[i]  = 0.01f * (f32)((i64)(i % 97) - 48);
        x2[i] = 0.01f * (f32)((i64)(i % 89) - 44);
    }
    /* Cheap tripwire for exactly the class of mistake above: if a future
     * edit reintroduces unsigned wraparound, or any other way of filling
     * this pool with nonsense, fail here naming the value rather than deep
     * inside a kernel that can only report a symptom. */
    for (u64 i = 0; i < x_len; i++) {
        if (!(x[i] >= -1.0f && x[i] <= 1.0f) ||
            !(x2[i] >= -1.0f && x2[i] <= 1.0f)) {
            fatal("bench: activation pool value out of range at i=%llu "
                  "(x=%g, x2=%g) -- expected [-1,1]; check for unsigned "
                  "wraparound in the fill expression",
                  (unsigned long long)i, (double)x[i], (double)x2[i]);
        }
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
    printf("    * %%ref far below -> UNEXPLAINED, not automatically headroom.\n");
    printf("                        A real kernel does not reach a read ceiling:\n");
    printf("                        llama's VNNI decode sits at ~68%% (1 thread,\n");
    printf("                        pinned), stoke's scalar -O3 at ~32%%. Rank an\n");
    printf("                        op against a llama referent measured in THIS\n");
    printf("                        session, never against 100%% and never against\n");
    printf("                        a figure quoted from another day.\n");
    printf("    * matvec LOGITS is culprit #1 -- largest single op per token.\n");
    printf("    * max >> med     -> interference during the run, not a slow\n");
    printf("                        kernel. Re-run on a quiesced box.\n");

    if (g_jsonl) {
        RefResult r_split8 = { split8_gbs, rt_split8 };
        RefResult r_xor    = { xor_gbs, rt_xor };
        RefResult r_serial = { serial_gbs, rt_serial };
        emit_jsonl(model_path, g_ref_gbs, r_split8, r_xor, r_serial);
    }

    free(x);
    free(x2);
    free(y);
    free(ref_buf);
    gguf_close(f);
}
