/* src/model.c -- M2 Qwen3 scalar forward pass. See model.h for the full
 * contract. Op order is exactly ref_forward.py's Qwen3Ref.forward_position
 * (docs/superpowers/design-notes/2026-07-19-m2-op-spec-and-parity-protocol.md
 * section 6; NEOX rope math + graph order re-confirmed against b10037
 * source in .superpowers/sdd/m2-probe-notes.md P3.3/P3.4).
 */
#include "common.h"

#include "model.h"

#include "kernels/kernels.h"
#include "kernels/simd.h"
#include "pool.h" /* the attention heads run on the pool -- att_kvheads_run */
#include "sample.h" /* model_generate_sampled's optional chooser */

/* ▶ SSE2 ONLY, AND IT IS BASELINE ON x86-64 -- this file is deliberately
 * compiled WITHOUT -mavx2 (SIMD_CFLAGS is scoped to src/kernels/simd.c so gcc
 * cannot emit AVX2 into the scalar reference twins), and the attention score
 * pass needs exactly four lanes because gqa_group is 4. The guard is simd.c's,
 * and the scalar arm below it is what a non-x86 build takes. */
#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#define STOKE_ATT_SSE 1
#else
#define STOKE_ATT_SSE 0
#endif

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h> /* getenv, strtol -- see att_env_init */
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

/* M2 op-spec design note section 2: "M2 fixes max_ctx = 4096" -- the
 * config's own qwen3.context_length (262144 in the real file) is clamped
 * to this cap, never used directly. */
#define STOKE_M2_MAX_CTX 4096u

/* ===========================================================================
 * Model: private layout (opaque to model.h callers)
 * ===========================================================================
 */

typedef struct {
    const GgufTensor* attn_norm;
    const GgufTensor* attn_q;
    const GgufTensor* attn_q_norm;
    const GgufTensor* attn_k;
    const GgufTensor* attn_k_norm;
    const GgufTensor* attn_v;
    const GgufTensor* attn_output;
    const GgufTensor* ffn_norm;
    const GgufTensor* ffn_gate;
    const GgufTensor* ffn_up;
    const GgufTensor* ffn_down;
} Layer;

/* ▶ PER-SEQUENCE SCRATCH, GROUPED SO IT CAN BECOME PER-SEQUENCE.
 * Every buffer below is written and read within ONE sequence's forward
 * pass. Static batch (design note 2026-08-11-m4-static-batch.md section
 * 5.3) needs one of these per sequence in the batch; leaving them as
 * loose fields on Model would make that an edit at 67 call sites instead
 * of an array subscript. THIS COMMIT CHANGES NO BEHAVIOUR -- it is the
 * grouping only, verified byte-identical.
 *
 * Allocated once at open, reused by every forward_position/compute_logits/
 * generate call -- "no malloc inside the per-token loop" (M2 op-spec 6). */
/* ▶▶ ATT_LANES AND THE q-HEAD-OUTER ARM ARE GONE (M5, 2026-08-19), AND WHAT
 * THEY WERE FOR IS WORTH KEEPING.
 *
 * `vec_dot_f32` is `sum += a[i]*b[i]`, which carries a loop-carried dependency
 * on `sum` and therefore runs at FP-ADD LATENCY -- about 4 cycles per element,
 * however many multiply units are idle. Several INDEPENDENT dot products run at
 * once fill that pipeline instead of stalling on it. ATT_LANES got its four
 * independent chains from four POSITIONS of one q-head; E2's loop
 * (att_kvheads_run) gets them from the four Q-HEADS OF ONE GQA GROUP, which
 * fills the pipeline exactly as well AND fetches that kv-head's K and V once
 * for the group instead of once per q-head. E2 subsumed it, so keeping both
 * meant 16 live chains and 16 scratch slots for no argued reason.
 *
 * The lanes arm outlived its deletion only as the STOKE_ATT_GQA=0 measurement
 * anchor -- both A/B arms had to come from ONE binary in ONE sitting, since the
 * pre-`76dee7f` archive swings up to 10.9x between sittings. That A/B has
 * reported (gate-reports/2026-08-19-m5-e2-ab-report.md: 0.984x / 1.058x /
 * 1.120x at ctx 87 / 554 / 1500), so the anchor's job is discharged and keeping
 * it would be the "spent helper outliving its job" defect, recorded five times.
 *
 * ⚠ THE ONE THING THE DELETION MAKES PERMANENT, stated rather than buried: E2
 * is a 1.6% REGRESSION at ctx 87 (+1.75 ms at the intercept, the cost of
 * dropping the unit count from n_head to n_head_kv) and a 12% gain at ctx 1500.
 * Selecting the loop by context length at runtime was considered and REJECTED:
 * it doubles this file's attention code and its maintenance to recover <=1.6%
 * at short contexts, while M5's own operating point is an agentic trace with a
 * shared 1-2k prefix, i.e. the long end. */

/* Floats of kv scratch one attention loop needs: ONE live (K,V) pair. At Q8
 * kv_read dequantizes INTO this buffer and its pointers stay valid only "until
 * the next kv_read using the same scratch" (kv.h) -- E2's loop consumes both
 * halves of a slot within the iteration that fetched it, so one pair suffices.
 *
 * ▶ ONE EXPRESSION, USED BY EVERY SITE THAT ALLOCATES IT, and that is a scar.
 * There were two such sites, sized independently, and when the lane loop raised
 * the requirement from 1 pair to 4 only one of them was updated: the `run` path
 * then overran its buffer by 3072 bytes at --kv q8 and surfaced as
 * `bp_block_slot: block 3230404224 out of range`, a corrupted heap reported by
 * an innocent bystander. Caught by byte-comparing the q8 path against the
 * archived binary, which the f32-only check that preceded it could not have
 * caught: at f32 kv_read is zero-copy and never touches this buffer at all.
 * The requirement has now gone back DOWN to 1 pair, which is the direction that
 * cannot overrun -- but the expression stays single-sourced anyway. */
#define ATT_KV_FLOATS(head_dim) ((size_t)(head_dim) * 2u)

/* M5 E3: positions per prefill chunk. 128 = 8 KV blocks, so every chunk
 * boundary is block-aligned and S-ALIGN holds by construction; it also matches
 * the LOCKED scheduler chunk, so M5's co-running prefill inherits this width.
 * 1 turns E3 off. Declared up here with the other run-time knobs because
 * att_env_init validates it; the mechanism and its correctness argument live
 * at prefill_range. */
/* ---- the per-op prefill profile (M7, -DSTOKE_PROFILE_OPS) ---------------
 *
 * ▶▶▶ WHY THIS EXISTS. The 0.51x gap to the M5 bar is a PREFILL gap, and 95% of
 * prefill's FLOPs are matvec. But stoke's own batched Q4_K kernel measures ~70
 * GF/s single-threaded while the whole prefill extracts ~33 GF/s per thread --
 * so something between the kernel and the forward pass is losing more than the
 * kernel itself would explain. Every instrument this project had either times
 * ONE tensor in a hot loop or the whole prefill as a single number, and the gap
 * lives exactly between those two. This is the missing middle.
 *
 * ⚠ WALL CLOCK PER OP, INCLUDING THE POOL DISPATCH. The matmuls and attention
 * fan out to the worker pool, so timing around the call is what a caller
 * actually waits for -- which is the quantity that adds up to the prefill rate.
 * A per-thread CPU-time profile would hide exactly the dispatch and join costs
 * worth finding.
 *
 * ⚠ AND IT MUST COST NOTHING WHEN OFF. With the flag absent every macro below
 * expands to `((void)0)`: no counters, no timer calls, no variables to trip
 * -Werror. The shipping build is unchanged, which the byte-exact goldens check.
 * Roughly 1,800 timed regions in a 525-token prefill against ~20 ns per
 * QueryPerformanceCounter is under a millisecond of the ~17,000 it measures. */
/* ⚠⚠ FOUR OF THESE WERE NEVER WIRED, AND THE ENUM SAID OTHERWISE (fixed
 * 2026-08-31). PROF_QKNORM, PROF_SWIGLU and PROF_NORM had comments describing
 * exactly what they should measure, no PROF_ADD anywhere, and printed 0.0 ms
 * -- which reads as "free", not as "never timed". PROF_FFN_DOWN's comment said
 * "+ residual" and the region ended before the residual loop. Together they
 * hid 13% of prefill: at the end of the M7 kernel work the four matmul rows
 * plus attention accounted for 6835 ms of a 7870 ms prefill and the missing
 * 1035 ms was invisible.
 *
 * ⇒ A ZERO IN A PROFILE IS TWO DIFFERENT CLAIMS -- "this costs nothing" and
 * "nothing measured this" -- and they are indistinguishable in the output. The
 * table now prints a row for every enum entry AND its share of the caller's
 * wall clock, so an unwired bucket shows up as a gap rather than as a zero. */
typedef enum {
    PROF_QKV = 0,    /* attn_q + attn_k + attn_v matvecs            */
    PROF_QKNORM,     /* per-head QK-norm + rope (phase 1a)          */
    PROF_KVAPPEND,   /* kv_note_token + kv_write (phase 1b)         */
    PROF_ATT,        /* causal GQA attention (phase 2)              */
    PROF_ATTOUT,     /* attn_output matvec                          */
    PROF_FFN_GU,     /* ffn_gate + ffn_up matvecs                   */
    PROF_SWIGLU,     /* the SwiGLU elementwise                      */
    PROF_FFN_DOWN,   /* ffn_down matvec                             */
    PROF_RESID,      /* the two residual adds per layer             */
    PROF_NORM,       /* the two rmsnorms per layer                  */
    PROF_SETUP,      /* embed, views bind, validate, x_out copy     */
    PROF_N
} ProfOp;

#ifdef STOKE_PROFILE_OPS
static double g_prof[PROF_N];
static double g_prof_wall;   /* every forward_batch_body call, end to end */
static const char* const g_prof_name[PROF_N] = {
    "qkv matvec", "qk-norm+rope", "kv append", "attention",
    "attn_out matvec", "ffn gate+up matvec", "swiglu", "ffn_down matvec",
    "residual add", "rmsnorm", "embed+bind+copy"
};
#define PROF_DECL double prof_t0 = now_ms()
#define PROF_ADD(op) (g_prof[op] += now_ms() - prof_t0)
void model_profile_dump(void) {
    double tot = 0.0;
    for (int i = 0; i < PROF_N; i++) tot += g_prof[i];
    if (tot <= 0.0) return;
    fprintf(stderr, "\n  per-op profile (wall ms, pool dispatch included)\n");
    fprintf(stderr, "  %-22s %10s %8s\n", "op", "ms", "%");
    for (int i = 0; i < PROF_N; i++) {
        fprintf(stderr, "  %-22s %10.1f %7.1f%%\n", g_prof_name[i], g_prof[i],
                100.0 * g_prof[i] / tot);
    }
    fprintf(stderr, "  %-22s %10.1f\n", "TOTAL ACCOUNTED", tot);
    /* ▶ THE RESIDUE IS THE POINT OF THIS LINE. Summing the buckets and calling
     * it the total is what let four unwired regions hide for a milestone: the
     * percentages were of each other, so they always added to 100%. This
     * measures the same span the buckets sit inside -- every
     * forward_batch_body call, end to end -- so anything not in a bucket has
     * to appear here. */
    fprintf(stderr, "  %-22s %10.1f\n", "forward_batch_body wall", g_prof_wall);
    fprintf(stderr, "  %-22s %10.1f %7.1f%%\n", "UNACCOUNTED",
            g_prof_wall - tot,
            g_prof_wall > 0.0 ? 100.0 * (g_prof_wall - tot) / g_prof_wall
                              : 0.0);
}
#else
#define PROF_DECL ((void)0)
#define PROF_ADD(op) ((void)0)
#endif

#ifdef STOKE_PROFILE_OPS
#define PROF_WALL_DECL double prof_wall0 = now_ms()
#define PROF_WALL_ADD() (g_prof_wall += now_ms() - prof_wall0)
#else
#define PROF_WALL_DECL ((void)0)
#define PROF_WALL_ADD() ((void)0)
#endif

#define PREFILL_CHUNK 128u
static u32 g_prefill_chunk = PREFILL_CHUNK;

/* Run the heads on the pool (1) or serially on the calling thread (0). Both
 * arms are BIT-IDENTICAL -- see att_kvheads_run -- so this is a pure speed
 * switch, and it is an env var rather than a compile flag for the same reason
 * STOKE_NO_QOS and STOKE_NO_PIN are: the A/B then interleaves inside ONE
 * sitting from ONE binary, which is the only comparison this box supports. */
static u32 g_att_par = 1u;

/* Attention's AVX2 kernels, on unless STOKE_ATT_SIMD=0 -- and gated at the
 * call site by kernels_cpu_has_avx_vnni() as simd.h requires. */
static u32 g_att_simd = 1u;

/* KV-HEADS per claim (it was q-heads before E2 made the kv-head the unit). 1 by
 * default: the box is 6 P-cores + 4 E-cores, an E-core takes several times
 * longer on the same work, and the smallest claim is what keeps a slow core
 * from holding others hostage at the barrier. A claim is already gqa_group
 * q-heads wide, so raising this costs balance faster than it did before E2.
 *
 * ▶ ITS OLD HEADER PROMISED THE KV-CACHE REUSE THAT E2 NOW GETS STRUCTURALLY
 * ("a chunk of gqa_group reads that head's K and V once instead of four times
 * -- worth measuring, not assuming"). It was never true: chunking only put the
 * four q-heads on the same WORKER and left four separate passes over the same
 * K and V, so the reuse depended on cache residency rather than on the loop.
 * E2 made it a property of the loop nest. The knob survives for balance
 * tuning, and for exercising the sequence-straddling walk in att_batch_run,
 * which the default of 1 can never reach. */
static u32 g_att_chunk = 1u;

/* Below this many context positions a head is too small to be worth a
 * dispatch (a broadcast, N wakeups, a join). 32 is a starting value chosen
 * from the shape of the cost -- a head at context P costs roughly P*64 ns and
 * a dispatch a few us -- and is tunable so the crossover can be measured
 * rather than argued about. */
static u32 g_att_min_pos = 32u;

/* Counts the times the heads were actually handed to the pool.
 *
 * EXISTS SO A TEST CANNOT PASS VACUOUSLY. "Output is bit-identical at 1 and 6
 * threads" is also true of a build where the dispatch never fires -- because
 * the pool was not started, because g_att_par was 0, because the context sat
 * under g_att_min_pos, or because pool_for_units decided there were too few
 * chunks and quietly ran inline. That is four ways for the assertion to be
 * about nothing, and this project has already recorded a test passing for a
 * reason unrelated to its name. Written only by the dispatching thread. */
static u64 g_att_dispatches = 0;

u64 model_att_dispatch_count(void) { return g_att_dispatches; }

/* Positions that went through E3's BATCHED prefill path, as opposed to the
 * position-by-position one.
 *
 * THIRD ANTI-VACUITY COUNTER, GUARDING A THIRD NOTHING. "chunked prefill writes
 * the same KV as position-by-position" is also true of a build where the
 * chunker silently never engaged -- because the chunk width was clamped,
 * because the prompt was shorter than one chunk, because a dump forced the solo
 * path. That is three ways for the comparison to be between one code path and
 * itself. Written only by the prefilling thread. */
static u64 g_prefill_batch_rows = 0;

u64 model_prefill_batch_rows(void) { return g_prefill_batch_rows; }

void model_att_set_chunk(u32 units) {
    g_att_chunk = units ? units : 1u;
}

void model_set_prefill_chunk(u32 chunk) {
    if (chunk == 0u) chunk = 1u;
    if (chunk != 1u && chunk % kv_block_tokens() != 0u) {
        fatal("model: model_set_prefill_chunk(%u) -- not 1 and not a multiple "
              "of the KV block size (%u); a prefill chunk must end on a block "
              "boundary (S-ALIGN)",
              chunk, kv_block_tokens());
    }
    g_prefill_chunk = chunk;
}

static u32 env_u32(const char* name, u32 dflt, u32 lo, u32 hi) {
    const char* s = getenv(name);
    if (!s || !s[0]) return dflt;
    char*         end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (!end || *end != '\0' || v < (unsigned long)lo || v > (unsigned long)hi) {
        return dflt;
    }
    return (u32)v;
}

static void att_env_init(void) {
    /* STOKE_ATT_LANES and STOKE_ATT_GQA are GONE, and are deliberately not
     * accepted-and-ignored: a knob that silently does nothing is worse than one
     * that does not exist, because a later A/B would set it, see no difference,
     * and record that as a finding about the kernel. */
    g_att_par = env_u32("STOKE_ATT_PAR", 1u, 0u, 1u);

    /* STOKE_PREFILL_CHUNK: 1 turns E3 off (position-by-position, the pre-M5
     * path and the A/B's other arm); anything else must be a WHOLE NUMBER OF
     * BLOCKS or S-ALIGN is violated -- a chunk that ends mid-block leaves that
     * block unregistrable by the prefix cache until something else fills it.
     * A bad value is REFUSED rather than rounded: rounding would silently
     * measure a different chunk width than the one asked for, and this knob
     * exists to be measured. */
    u32 pc = env_u32("STOKE_PREFILL_CHUNK", PREFILL_CHUNK, 1u, 1u << 16);
    if (pc != 1u && pc % kv_block_tokens() != 0u) {
        fatal("model: STOKE_PREFILL_CHUNK=%u is not 1 and not a multiple of "
              "the KV block size (%u) -- a prefill chunk must end on a block "
              "boundary or the prefix cache cannot register it (S-ALIGN)",
              pc, kv_block_tokens());
    }
    g_prefill_chunk = pc;
    /* STOKE_ATT_SIMD=0 forces attention's SSE2/scalar arm even on a CPU that
     * has the AVX2 kernels. It exists so the two arms can be run against each
     * other on THIS box -- without it the SSE2 arm is dead code here and its
     * agreement with the AVX2 one would be asserted rather than tested, which
     * is the vacuous-pass trap simd.h's header names. */
    g_att_simd = env_u32("STOKE_ATT_SIMD", 1u, 0u, 1u);
    g_att_chunk = env_u32("STOKE_ATT_CHUNK", 1u, 1u, 1024u);
    g_att_min_pos = env_u32("STOKE_ATT_MIN_POS", 32u, 0u, 1u << 20);
}

typedef struct {
    f32* s_a;        /* [n_embd]              post-attn_norm            */
    f32* s_q;        /* [n_head*head_dim]     Q, all heads flattened    */
    f32* s_k;        /* [n_head_kv*head_dim]  K, all kv-heads flattened */
    f32* s_v;        /* [n_head_kv*head_dim]  V, all kv-heads flattened */
    f32* s_attn_out; /* [n_head*head_dim]     per-head attn output      */
    f32* s_proj;     /* [n_embd]              attn_output/ffn_down proj */
    f32* s_f;        /* [n_embd]              post-ffn_norm             */
    f32* s_gate;     /* [n_ff]                gate, then silu(gate)*up  */
    f32* s_up;       /* [n_ff]                up                        */
    f32* s_x;        /* [n_embd]              running residual stream   */
    f32* s_h;        /* [n_embd]              post-output_norm          */
    f32* s_xout;     /* [n_embd]              generate()'s x carry      */
    f32* s_logits;   /* [n_vocab]                                       */

    /* M3 integer core: the Q8_0-quantized activation handed to matvec.
     * Sized once at open for the LARGEST contraction length any projection
     * uses (n_ff dominates, via ffn_down), so no matvec can outgrow it and
     * the token loop still allocates nothing. */
    u8* s_aq;
    u64 s_aq_bytes;

    /* ▶ s_scores AND s_kv USED TO LIVE HERE, per sequence. They moved to
     * Model::att_ws, per WORKER -- see that field. Nothing else in this struct
     * is touched by more than one thread at a time, which is why the rest of
     * it stays sequence-owned. */
} SeqScratch;

/* Pointer views over the N scratch sets a batch is stepping.
 *
 * matvec_batch takes ARRAYS OF POINTERS -- one activation per sequence --
 * but the scratch sets are structs, so those pointers have to be gathered
 * somewhere. They are gathered ONCE per forward call rather than at each
 * call site: seven batched weights x n_layer means the same gather would
 * otherwise be redone 252 times per token on the real model, to no purpose.
 *
 * Grown on demand and never shrunk, exactly like scratch_n, and for the same
 * reason -- the batch width belongs to the caller and model_open cannot know
 * it. Kept as f32** (not const) because s_gate is a matvec_batch OUTPUT for
 * ffn_gate and then an INPUT for ffn_down; the const is added at the call
 * site, see forward_batch_body. */
typedef struct {
    f32**  a;        /* s_a        : x for attn_q/attn_k/attn_v       */
    f32**  q;        /* s_q        : y for attn_q                     */
    f32**  k;        /* s_k        : y for attn_k                     */
    f32**  v;        /* s_v        : y for attn_v                     */
    f32**  attn_out; /* s_attn_out : x for attn_output                */
    f32**  proj;     /* s_proj     : y for attn_output and ffn_down   */
    f32**  f;        /* s_f        : x for ffn_gate/ffn_up            */
    f32**  gate;     /* s_gate     : y for ffn_gate, then x for down  */
    f32**  up;       /* s_up       : y for ffn_up                     */
    f32**  h;        /* s_h        : x for the OUTPUT projection       */
    void** aq;       /* s_aq       : one quantized-activation buffer  */
    u32    n;        /* sets the arrays are sized for                 */
} BatchViews;

/* Which rotary pairing this model's architecture uses. Not a preference:
   see the note at the arch check in model_open, and kernels.h's rope_norm. */
typedef enum { ROPE_NEOX = 0, ROPE_NORM = 1 } RopeKind;

struct Model {
    GgufFile* f; /* owned; kept open for the model's whole lifetime -- tensor
                    data is borrowed from its mmap (gguf.h) */

    u32 n_layer, n_embd, n_ff, n_head, n_head_kv, head_dim, gqa_group;
    u32 max_ctx, n_vocab;
    f32 freq_base, eps;

    const GgufTensor* tok_embd;
    const GgufTensor* output_norm;
    const GgufTensor* output; /* NULL => tied to tok_embd (M2 note TRAP #2) */

    /* rope_freqs.weight (head_dim/2 f32, borrowed from the mmap) or NULL
       when the model ships no RoPE-scaling correction. qwen3 has none;
       llama3 does, and ignoring it is silent -- see kernels.h's rope_neox. */
    const f32* rope_freqs;
    RopeKind   rope_kind;
    bool       has_qk_norm; /* qwen3 yes, llama no -- all layers or none */

    Layer* layers; /* [n_layer], xmalloc'd */

    SeqScratch scratch;      /* the single-sequence set          */
    SeqScratch* scratch_n;   /* [n_scratch], grown by the batch  */
    SeqScratch** scratch_p;  /* [n_scratch], &scratch_n[s]       */
    u32 n_scratch;
    BatchViews views;        /* pointer arrays over N of them    */

    /* PER-WORKER attention scratch: [att_workers][att_stride], holding
     * gqa_group scores rows (one per q-head of the claim's group) then one KV
     * dequantization buffer (ATT_KV_FLOATS).
     *
     * ▶ WHY THIS IS PER WORKER AND NO LONGER PER SEQUENCE. Both regions are
     * private to ONE claim's arithmetic -- the scores rows are that group's
     * softmax rows, and the pair is where kv_read lands its dequantized K/V.
     * Per-sequence was the right unit while heads ran one after another; it is
     * the WRONG unit the moment two claims of the SAME sequence run at once,
     * which is exactly what the parallel attention loop does. Per worker is the
     * tightest correct unit: a claim runs on exactly one worker start to
     * finish, and the count is threads (8) rather than kv-heads (8 per
     * sequence, and n_seq times that in a batch).
     *
     * It also collapses two allocation sites into one. The model-level and
     * per-sequence copies were sized independently, and the lane loop's raised
     * requirement reached only one of them -- a 3072-byte heap overrun on the
     * `run` path (see ATT_KV_FLOATS). One buffer, one owner, one size.
     *
     * ▶▶ ONE ALLOCATION PER WORKER, NOT ONE BIG ONE SLICED BY WORKER, AND THAT
     * IS AN INSTRUMENT DECISION. The sliced version was written first. Then
     * the undersize bug was mutated back in and run under ASan on the real
     * model at --kv q8: ASan exited CLEAN. Of course it did -- an undersized
     * slice overruns into the NEXT WORKER'S slice, which is still inside the
     * same malloc block, so there is no boundary for ASan to see. The bug
     * stayed silent and landed in the arithmetic, which is exactly how it got
     * this far the first time.
     *
     * Separate allocations put a redzone between every worker, so the same
     * mistake becomes a loud ASan report instead of a wrong token. The cost is
     * one pointer indirection PER HEAD -- not per element -- against a bug
     * class this file has now demonstrated it can produce. */
    f32** att_ws;    /* [att_workers], each att_stride floats */
    u32   att_workers;
    u64   att_stride;
};

/* ===========================================================================
 * model_open helpers
 * ===========================================================================
 */

static u32 req_u32(GgufFile* f, const char* key) {
    i64 v = gguf_meta_i64(f, key); /* fatals on missing key / non-integer type */
    if (v < 0 || v > (i64)UINT32_MAX) {
        fatal("model: config key '%s' value %lld is out of u32 range", key,
              (long long)v);
    }
    return (u32)v;
}

static f32 req_f32(GgufFile* f, const char* key) {
    return gguf_meta_f32(f, key); /* fatals on missing key / non-FLOAT32 type */
}

static const GgufTensor* find_tensor_or_null(GgufFile* f, const char* name) {
    u64 n = gguf_tensor_count(f);
    for (u64 i = 0; i < n; i++) {
        const GgufTensor* t = gguf_tensor(f, i);
        if (strcmp(t->name, name) == 0) return t;
    }
    return NULL;
}

static const GgufTensor* layer_tensor(GgufFile* f, u32 L, const char* suffix) {
    char name[96];
    int  n = snprintf(name, sizeof name, "blk.%u.%s.weight", L, suffix);
    if (n < 0 || (size_t)n >= sizeof name) {
        fatal("model: internal -- tensor name buffer too small for "
              "blk.%u.%s.weight",
              L, suffix);
    }
    return gguf_tensor_by_name(f, name);
}

/* Same, but NULL when the layer does not carry that tensor -- for the
 * per-architecture optional ones (attn_q_norm/attn_k_norm: qwen3 has them,
 * llama has neither). The CALLER decides what absence means; see the
 * all-or-nothing check in model_open. */
static const GgufTensor* layer_tensor_opt(GgufFile* f, u32 L,
                                           const char* suffix) {
    char name[96];
    int  n = snprintf(name, sizeof name, "blk.%u.%s.weight", L, suffix);
    if (n < 0 || (size_t)n >= sizeof name) {
        fatal("model: internal -- tensor name buffer too small for "
              "blk.%u.%s.weight",
              L, suffix);
    }
    return gguf_tensor_by_name_opt(f, name);
}

/* Builds "<arch>.<suffix>" into `buf` and returns it. Every hyperparameter
 * key in a GGUF is namespaced by the architecture, and llama uses the SAME
 * suffixes qwen3 does (verified key by key against both files, not assumed
 * from the naming convention), so supporting a second architecture is a
 * prefix change rather than a second block of lookups. `buf` is reused
 * across calls: each returned key is consumed by its own lookup call
 * before the next one is built. */
static const char* arch_key(char* buf, size_t cap, const char* arch,
                             const char* suffix) {
    int n = snprintf(buf, cap, "%s.%s", arch, suffix);
    if (n < 0 || (size_t)n >= cap) {
        fatal("model: internal -- key buffer too small for '%s.%s'", arch,
              suffix);
    }
    return buf;
}

/* Norm vectors (attn_norm/attn_q_norm/attn_k_norm/ffn_norm/output_norm) are
 * always plain F32 in every known qwen3 GGUF (never quantized -- W2's real-
 * file histogram: 145 F32 = 36*4 + output_norm, exactly the norm count).
 * Requiring F32 here (rather than adding a dequant path this loader would
 * never otherwise exercise) means rmsnorm can read straight off the mmap
 * with zero copying: see the (const f32*)t->data casts in
 * model_forward_position/model_compute_logits below. */
static void check_norm_tensor(const GgufTensor* t, u64 want_len) {
    u64 want[1];
    want[0] = want_len;
    model_check_tensor_shape(t, 1, want);
    if (t->type != GGUF_GGML_TYPE_F32) {
        fatal("model: tensor '%s' has ggml type %u, expected F32(%u) -- "
              "norm vectors must be plain f32 in this loader",
              t->name, t->type, (u32)GGUF_GGML_TYPE_F32);
    }
}

static void check_weight_tensor(const GgufTensor* t, u64 want0, u64 want1) {
    u64 want[2];
    want[0] = want0;
    want[1] = want1;
    model_check_tensor_shape(t, 2, want);
    if (t->type != GGUF_GGML_TYPE_F32 && t->type != GGUF_GGML_TYPE_Q8_0 &&
        t->type != GGUF_GGML_TYPE_Q4_K && t->type != GGUF_GGML_TYPE_Q6_K) {
        fatal("model: tensor '%s' has ggml type %u -- only F32(%u)/Q8_0(%u)/"
              "Q4_K(%u)/Q6_K(%u) are supported",
              t->name, t->type, (u32)GGUF_GGML_TYPE_F32,
              (u32)GGUF_GGML_TYPE_Q8_0, (u32)GGUF_GGML_TYPE_Q4_K,
              (u32)GGUF_GGML_TYPE_Q6_K);
    }
}

/* ===========================================================================
 * model_open / model_close
 * ===========================================================================
 */

/* Allocates one SeqScratch from the model's dims. `s_aq_bytes` must already
 * be set on m->scratch -- model_open computes it from the widest contraction
 * any projection uses, which needs the layer table, so it stays there and is
 * copied rather than recomputed. One list, one place: a second copy of these
 * sizes is a second thing to keep in step with the kernels. */
static void scratch_alloc(Model* m, SeqScratch* sc) {
    sc->s_a        = (f32*)xmalloc((size_t)m->n_embd * sizeof(f32));
    sc->s_q        = (f32*)xmalloc((size_t)m->n_head * m->head_dim * sizeof(f32));
    sc->s_k        = (f32*)xmalloc((size_t)m->n_head_kv * m->head_dim * sizeof(f32));
    sc->s_v        = (f32*)xmalloc((size_t)m->n_head_kv * m->head_dim * sizeof(f32));
    sc->s_attn_out = (f32*)xmalloc((size_t)m->n_head * m->head_dim * sizeof(f32));
    sc->s_proj     = (f32*)xmalloc((size_t)m->n_embd * sizeof(f32));
    sc->s_f        = (f32*)xmalloc((size_t)m->n_embd * sizeof(f32));
    sc->s_gate     = (f32*)xmalloc((size_t)m->n_ff * sizeof(f32));
    sc->s_up       = (f32*)xmalloc((size_t)m->n_ff * sizeof(f32));
    sc->s_x        = (f32*)xmalloc((size_t)m->n_embd * sizeof(f32));
    sc->s_h        = (f32*)xmalloc((size_t)m->n_embd * sizeof(f32));
    sc->s_xout     = (f32*)xmalloc((size_t)m->n_embd * sizeof(f32));
    sc->s_logits   = (f32*)xmalloc((size_t)m->n_vocab * sizeof(f32));
    sc->s_aq_bytes = m->scratch.s_aq_bytes;
    sc->s_aq       = (u8*)xmalloc((size_t)sc->s_aq_bytes);
}

static void att_ws_free(Model* m) {
    if (m->att_ws == NULL) return;
    for (u32 w = 0; w < m->att_workers; w++) free(m->att_ws[w]);
    free(m->att_ws);
    m->att_ws = NULL;
    m->att_workers = 0;
}

/* Sizes the per-worker attention scratch for the pool as it stands NOW.
 *
 * Called from model_open and again at the top of every forward call. In the
 * normal order -- both of main.c's paths call pool_start BEFORE model_open --
 * the model_open call sizes it correctly and the per-forward call is a single
 * compare that never allocates, so the token loop still allocates nothing
 * (M2 op-spec 6). The per-forward check exists because "the pool is already
 * started" is a caller ordering, not an invariant this file can enforce, and
 * the failure mode if it were assumed is a worker indexing past the end of
 * this buffer -- silent, and in the arithmetic. A compare per token is the
 * cheapest way to make the assumption unnecessary. */
static void att_ws_ensure(Model* m) {
    u32 want = pool_threads();
    if (want < 1u) want = 1u;
    if (m->att_ws != NULL && want <= m->att_workers) return;
    /* Rounded to a whole 64-byte line. Separate allocations are already far
     * apart, so this is not about false sharing between workers -- it keeps
     * each buffer's own two regions (scores, then the KV pairs) line-aligned
     * relative to its start. */
    /* gqa_group scores rows, not one: E2's unit is a KV-HEAD, and its whole
     * group's softmax rows are live at once (the K pass fills all of them
     * before the first softmax runs). Then ONE (K,V) pair, which is all the
     * loop keeps alive at a time. */
    /* + gqa_group*head_dim for the TRANSPOSED Q, which the score pass builds
     * once per (sequence, kv-head) and then reads once per (j, d). Q is the
     * cheap side to transpose: G*head_dim is 2 KB against K's n*head_dim,
     * which is 269 KB at ctx 525. */
    u64 need = (u64)m->gqa_group * (u64)m->max_ctx +
               (u64)ATT_KV_FLOATS(m->head_dim) +
               (u64)m->gqa_group * (u64)m->head_dim;
    u64 stride = (need + 15u) & ~(u64)15u;
    f32** ws = (f32**)xmalloc((size_t)want * sizeof(f32*));
    for (u32 w = 0; w < want; w++) {
        ws[w] = (f32*)xmalloc((size_t)stride * sizeof(f32));
    }
    att_ws_free(m);
    m->att_ws = ws;
    m->att_stride = stride;
    m->att_workers = want;
}

/* Sizes the view arrays for at least `n` sequences and points them at `scs`.
 *
 * Grown independently of scratch_n because the two have different callers:
 * model_forward_position steps ONE sequence off m->scratch, which is not in
 * scratch_n at all, so n_scratch can legitimately be 0 while the views need
 * room for one. */
static void views_bind(BatchViews* vw, SeqScratch* const* scs, u32 n) {
    if (n > vw->n) {
        vw->a        = (f32**)xrealloc(vw->a, (size_t)n * sizeof(f32*));
        vw->q        = (f32**)xrealloc(vw->q, (size_t)n * sizeof(f32*));
        vw->k        = (f32**)xrealloc(vw->k, (size_t)n * sizeof(f32*));
        vw->v        = (f32**)xrealloc(vw->v, (size_t)n * sizeof(f32*));
        vw->attn_out = (f32**)xrealloc(vw->attn_out, (size_t)n * sizeof(f32*));
        vw->proj     = (f32**)xrealloc(vw->proj, (size_t)n * sizeof(f32*));
        vw->f        = (f32**)xrealloc(vw->f, (size_t)n * sizeof(f32*));
        vw->gate     = (f32**)xrealloc(vw->gate, (size_t)n * sizeof(f32*));
        vw->up       = (f32**)xrealloc(vw->up, (size_t)n * sizeof(f32*));
        vw->h        = (f32**)xrealloc(vw->h, (size_t)n * sizeof(f32*));
        vw->aq       = (void**)xrealloc(vw->aq, (size_t)n * sizeof(void*));
        vw->n        = n;
    }
    for (u32 s = 0; s < n; s++) {
        vw->a[s]        = scs[s]->s_a;
        vw->h[s]        = scs[s]->s_h;
        vw->q[s]        = scs[s]->s_q;
        vw->k[s]        = scs[s]->s_k;
        vw->v[s]        = scs[s]->s_v;
        vw->attn_out[s] = scs[s]->s_attn_out;
        vw->proj[s]     = scs[s]->s_proj;
        vw->f[s]        = scs[s]->s_f;
        vw->gate[s]     = scs[s]->s_gate;
        vw->up[s]       = scs[s]->s_up;
        vw->aq[s]       = scs[s]->s_aq;
    }
}

static void views_free(BatchViews* vw) {
    free(vw->a); free(vw->q); free(vw->k); free(vw->v);
    free(vw->attn_out); free(vw->proj); free(vw->f);
    free(vw->gate); free(vw->up); free(vw->aq); free(vw->h);
    memset(vw, 0, sizeof *vw);
}

static void scratch_free(SeqScratch* sc) {
    free(sc->s_a); free(sc->s_q); free(sc->s_k); free(sc->s_v);
    free(sc->s_attn_out); free(sc->s_aq);
    free(sc->s_proj); free(sc->s_f); free(sc->s_gate); free(sc->s_up);
    free(sc->s_x); free(sc->s_h); free(sc->s_xout); free(sc->s_logits);
}

Model* model_open(const char* path) {
    att_env_init();
    GgufFile* f = gguf_open(path);

    const char* arch = gguf_meta_str(f, "general.architecture");
    /* ⚠ The ROPE TYPE is a property of the architecture and choosing wrong
     * produces plausible-looking garbage rather than an error: llama.cpp's
     * llama_model_rope_type() is NORM for LLM_ARCH_LLAMA and NEOX for
     * LLM_ARCH_QWEN3 (llama-model.cpp:1329/1367 vs :1386/1439). Adding an
     * architecture here means reading that function for it, not assuming
     * it matches the one already implemented. */
    bool is_qwen3 = (strcmp(arch, "qwen3") == 0);
    bool is_llama = (strcmp(arch, "llama") == 0);
    if (!is_qwen3 && !is_llama) {
        char arch_buf[128];
        snprintf(arch_buf, sizeof arch_buf, "%s", arch);
        gguf_close(f);
        fatal("model: '%s' -- general.architecture is '%s', expected "
              "'qwen3' or 'llama' (this loader implements those two)",
              path, arch_buf);
    }
    RopeKind rope_kind = is_llama ? ROPE_NORM : ROPE_NEOX;

    char kb[96];
    u32 n_layer = req_u32(f, arch_key(kb, sizeof kb, arch, "block_count"));
    i64 ctx_len =
        gguf_meta_i64(f, arch_key(kb, sizeof kb, arch, "context_length"));
    u32 n_embd = req_u32(f, arch_key(kb, sizeof kb, arch, "embedding_length"));
    u32 n_ff =
        req_u32(f, arch_key(kb, sizeof kb, arch, "feed_forward_length"));
    u32 n_head =
        req_u32(f, arch_key(kb, sizeof kb, arch, "attention.head_count"));
    u32 n_head_kv =
        req_u32(f, arch_key(kb, sizeof kb, arch, "attention.head_count_kv"));
    u32 head_dim =
        req_u32(f, arch_key(kb, sizeof kb, arch, "attention.key_length"));
    u32 value_len =
        req_u32(f, arch_key(kb, sizeof kb, arch, "attention.value_length"));
    f32 freq_base = req_f32(f, arch_key(kb, sizeof kb, arch, "rope.freq_base"));
    f32 eps = req_f32(
        f, arch_key(kb, sizeof kb, arch, "attention.layer_norm_rms_epsilon"));

    if (n_layer == 0 || n_embd == 0 || n_ff == 0 || n_head == 0 ||
        n_head_kv == 0 || head_dim == 0) {
        gguf_close(f);
        fatal("model: '%s' -- one of block_count/embedding_length/"
              "feed_forward_length/head_count/head_count_kv/key_length is 0",
              path);
    }
    if (ctx_len <= 0) {
        gguf_close(f);
        fatal("model: '%s' -- %s.context_length=%lld must be positive",
              path, arch, (long long)ctx_len);
    }
    if (value_len != head_dim) {
        gguf_close(f);
        fatal("model: '%s' -- %s.attention.value_length=%u != "
              "%s.attention.key_length=%u (M2 requires them equal)",
              path, arch, value_len, arch, head_dim);
    }
    if (n_head % n_head_kv != 0) {
        gguf_close(f);
        fatal("model: '%s' -- %s.attention.head_count=%u not divisible "
              "by head_count_kv=%u (GQA group size must be exact)",
              path, arch, n_head, n_head_kv);
    }
    if (head_dim % 2 != 0) {
        gguf_close(f);
        fatal("model: '%s' -- %s.attention.key_length=%u must be even "
              "(both rope kinds rotate head_dim/2 pairs)",
              path, arch, head_dim);
    }
    if (!(isfinite(freq_base) && freq_base > 0.0f)) {
        gguf_close(f);
        fatal("model: '%s' -- %s.rope.freq_base=%g must be finite and "
              "positive",
              path, arch, (double)freq_base);
    }
    if (!(isfinite(eps) && eps >= 0.0f)) {
        gguf_close(f);
        fatal("model: '%s' -- %s.attention.layer_norm_rms_epsilon=%g "
              "must be finite and non-negative",
              path, arch, (double)eps);
    }

    u32 max_ctx = ((u64)ctx_len < (u64)STOKE_M2_MAX_CTX)
                      ? (u32)ctx_len
                      : STOKE_M2_MAX_CTX;

    const GgufTensor* tok_embd = gguf_tensor_by_name(f, "token_embd.weight");
    /* Copy every field the error paths below might report to LOCALS
     * BEFORE any gguf_close(f) -- tok_embd points into f->tensors, which
     * gguf_close() frees, so reading tok_embd->* in a fatal() call AFTER
     * closing would be a use-after-free (the exact bug class gguf.c's own
     * gguf_open() already guards against at its tensor-offset checks; W2/
     * ASan caught the same mistake here against the real Q4_K_M dev file,
     * whose weight tensors are the unsupported-type branch below). */
    u32 te_n_dims = tok_embd->n_dims;
    u64 te_dims0 = tok_embd->dims[0];
    u32 te_type = tok_embd->type;
    u32 n_vocab = (u32)tok_embd->dims[1];

    if (te_n_dims != 2 || te_dims0 != n_embd) {
        gguf_close(f);
        fatal("model: tensor 'token_embd.weight' has n_dims=%u dims[0]=%llu, "
              "expected n_dims=2 dims[0]=%u (embedding_length)",
              te_n_dims, (unsigned long long)te_dims0, n_embd);
    }
    if (te_type != GGUF_GGML_TYPE_F32 && te_type != GGUF_GGML_TYPE_Q8_0 &&
        te_type != GGUF_GGML_TYPE_Q4_K && te_type != GGUF_GGML_TYPE_Q6_K) {
        gguf_close(f);
        fatal("model: tensor 'token_embd.weight' has ggml type %u -- only "
              "F32(%u)/Q8_0(%u)/Q4_K(%u)/Q6_K(%u) are supported",
              te_type, (u32)GGUF_GGML_TYPE_F32, (u32)GGUF_GGML_TYPE_Q8_0,
              (u32)GGUF_GGML_TYPE_Q4_K, (u32)GGUF_GGML_TYPE_Q6_K);
    }
    if (n_vocab == 0) {
        gguf_close(f);
        fatal("model: tensor 'token_embd.weight' dims[1] (vocab size) is 0");
    }

    const GgufTensor* output_norm = gguf_tensor_by_name(f, "output_norm.weight");
    check_norm_tensor(output_norm, n_embd);

    /* TRAP #2 (M2 note section 3): output.weight is OPTIONAL -- use it if
     * present, else the LM head ties to token_embd.weight. */
    const GgufTensor* output = find_tensor_or_null(f, "output.weight");
    if (output != NULL) {
        check_weight_tensor(output, n_embd, n_vocab);
    }

    Model* m = (Model*)xcalloc(1, sizeof(Model));
    m->f = f;
    m->n_layer = n_layer;
    m->n_embd = n_embd;
    m->n_ff = n_ff;
    m->n_head = n_head;
    m->n_head_kv = n_head_kv;
    m->gqa_group = n_head / n_head_kv;
    m->head_dim = head_dim;
    m->freq_base = freq_base;
    m->eps = eps;
    m->max_ctx = max_ctx;
    m->n_vocab = n_vocab;
    m->tok_embd = tok_embd;
    m->output_norm = output_norm;
    m->output = output;
    m->rope_kind = rope_kind;

    /* rope_freqs.weight -- llama3's per-pair RoPE-scaling correction.
     * OPTIONAL: absent means no correction, which is qwen3 and also plenty
     * of llama models. Validated to head_dim/2 F32 when present, because
     * the alternative to a length check here is reading past the tensor in
     * the rope kernel.
     *
     * ⚠ NOT a long-context-only concern, which is the first thing assumed
     * about it: Llama-3.2-3B-Instruct's vector runs 1.0 -> 32.0 with 29 of
     * 64 entries at 1.0, so it changes the rotation at EVERY position. */
    m->rope_freqs = NULL;
    {
        const GgufTensor* rf = gguf_tensor_by_name_opt(f, "rope_freqs.weight");
        if (rf) {
            u64 want[1];
            want[0] = head_dim / 2u;
            model_check_tensor_shape(rf, 1, want);
            if (rf->type != GGUF_GGML_TYPE_F32) {
                fatal("model: tensor 'rope_freqs.weight' has ggml type %u, "
                      "expected F32(%u) -- the rope correction is read "
                      "straight off the mmap as f32",
                      rf->type, (u32)GGUF_GGML_TYPE_F32);
            }
            m->rope_freqs = (const f32*)rf->data;
        }
    }

    /* QK-norm is OPTIONAL and ALL-OR-NOTHING. qwen3 carries attn_q_norm/
     * attn_k_norm on every block; llama carries neither. Deciding from
     * block 0 and then REQUIRING every other block to agree is the point:
     * a file with the tensors on some layers and not others is malformed,
     * and silently norming only the layers that happen to have them would
     * be a wrong answer with no error -- exactly the failure mode this
     * whole rung is about. */
    m->has_qk_norm = (layer_tensor_opt(f, 0, "attn_q_norm") != NULL);

    m->layers = (Layer*)xmalloc((size_t)n_layer * sizeof(Layer));
    for (u32 L = 0; L < n_layer; L++) {
        Layer* ly = &m->layers[L];
        ly->attn_norm = layer_tensor(f, L, "attn_norm");
        ly->attn_q = layer_tensor(f, L, "attn_q");
        ly->attn_q_norm = layer_tensor_opt(f, L, "attn_q_norm");
        ly->attn_k = layer_tensor(f, L, "attn_k");
        ly->attn_k_norm = layer_tensor_opt(f, L, "attn_k_norm");
        ly->attn_v = layer_tensor(f, L, "attn_v");
        ly->attn_output = layer_tensor(f, L, "attn_output");
        ly->ffn_norm = layer_tensor(f, L, "ffn_norm");
        ly->ffn_gate = layer_tensor(f, L, "ffn_gate");
        ly->ffn_up = layer_tensor(f, L, "ffn_up");
        ly->ffn_down = layer_tensor(f, L, "ffn_down");

        if ((ly->attn_q_norm != NULL) != m->has_qk_norm ||
            (ly->attn_k_norm != NULL) != m->has_qk_norm) {
            fatal("model: '%s' -- block %u %s attn_q_norm/attn_k_norm while "
                  "block 0 %s them; QK-norm must be present on every block "
                  "or on none",
                  path, L, m->has_qk_norm ? "is missing" : "has",
                  m->has_qk_norm ? "has" : "is missing");
        }

        check_norm_tensor(ly->attn_norm, n_embd);
        if (m->has_qk_norm) {
            check_norm_tensor(ly->attn_q_norm, head_dim);
            check_norm_tensor(ly->attn_k_norm, head_dim);
        }
        check_norm_tensor(ly->ffn_norm, n_embd);

        check_weight_tensor(ly->attn_q, n_embd, (u64)n_head * head_dim);
        check_weight_tensor(ly->attn_k, n_embd, (u64)n_head_kv * head_dim);
        check_weight_tensor(ly->attn_v, n_embd, (u64)n_head_kv * head_dim);
        check_weight_tensor(ly->attn_output, (u64)n_head * head_dim, n_embd);
        check_weight_tensor(ly->ffn_gate, n_embd, n_ff);
        check_weight_tensor(ly->ffn_up, n_embd, n_ff);
        check_weight_tensor(ly->ffn_down, n_ff, n_embd);
    }

    m->scratch.s_a = (f32*)xmalloc((size_t)n_embd * sizeof(f32));
    m->scratch.s_q = (f32*)xmalloc((size_t)n_head * head_dim * sizeof(f32));
    m->scratch.s_k = (f32*)xmalloc((size_t)n_head_kv * head_dim * sizeof(f32));
    m->scratch.s_v = (f32*)xmalloc((size_t)n_head_kv * head_dim * sizeof(f32));
    m->scratch.s_attn_out = (f32*)xmalloc((size_t)n_head * head_dim * sizeof(f32));
    /* Largest contraction length over every matvec in the forward pass:
     * n_embd (Q/K/V, ffn_gate/up, logits), n_head*head_dim (attn_output),
     * and n_ff (ffn_down). Taking the max rather than assuming which one
     * wins keeps this correct for any config, including ones where n_ff is
     * not the largest. */
    {
        u64 max_n_in = n_embd;
        u64 attn_out_in = (u64)n_head * head_dim;
        if (attn_out_in > max_n_in) max_n_in = attn_out_in;
        if ((u64)n_ff > max_n_in) max_n_in = n_ff;
        /* ONE scratch buffer serves both activation quantizations, sized for
         * whichever needs more: Q8_0 for Q8_0 weights, Q8_K for K-quants.
         * Per 256 elements that is 8*34 = 272 B against 292 B, so Q8_K wins
         * -- but taking the max rather than hardcoding that keeps it correct
         * if either layout ever moves.
         *
         * Ceiling division, not floor: a contraction length that is not a
         * whole number of blocks is rejected loudly by the kernels
         * themselves, and until then this buffer must not be the thing that
         * is short. Rounding up can only over-allocate by one block. */
        u64 q8_0_bytes = ((max_n_in + KERNELS_Q8_0_BLOCK_ELEMS - 1u)
                          / KERNELS_Q8_0_BLOCK_ELEMS) * KERNELS_Q8_0_BLOCK_BYTES;
        u64 q8_k_bytes = ((max_n_in + KERNELS_Q8_K_BLOCK_ELEMS - 1u)
                          / KERNELS_Q8_K_BLOCK_ELEMS) * KERNELS_Q8_K_BLOCK_BYTES;
        m->scratch.s_aq_bytes = q8_0_bytes > q8_k_bytes ? q8_0_bytes : q8_k_bytes;
        m->scratch.s_aq = (u8*)xmalloc((size_t)m->scratch.s_aq_bytes);
    }
    m->scratch.s_proj = (f32*)xmalloc((size_t)n_embd * sizeof(f32));
    m->scratch.s_f = (f32*)xmalloc((size_t)n_embd * sizeof(f32));
    m->scratch.s_gate = (f32*)xmalloc((size_t)n_ff * sizeof(f32));
    m->scratch.s_up = (f32*)xmalloc((size_t)n_ff * sizeof(f32));
    m->scratch.s_x = (f32*)xmalloc((size_t)n_embd * sizeof(f32));
    m->scratch.s_h = (f32*)xmalloc((size_t)n_embd * sizeof(f32));
    m->scratch.s_xout = (f32*)xmalloc((size_t)n_embd * sizeof(f32));
    m->scratch.s_logits = (f32*)xmalloc((size_t)n_vocab * sizeof(f32));
    /* Both of main.c's paths call pool_start before this, so this is the call
     * that actually sizes it and the per-forward one never allocates. */
    att_ws_ensure(m);

    return m;
}

void model_close(Model* m) {
#ifdef STOKE_PROFILE_OPS
    model_profile_dump();
#endif
    if (m) {
        for (u32 i = 0; i < m->n_scratch; i++) scratch_free(&m->scratch_n[i]);
        free(m->scratch_n);
        free(m->scratch_p);
        m->scratch_n = NULL;
        m->scratch_p = NULL;
        m->n_scratch = 0;
        views_free(&m->views);
    }
    if (!m) return;
    free(m->layers);
    free(m->scratch.s_a);
    free(m->scratch.s_q);
    free(m->scratch.s_k);
    free(m->scratch.s_v);
    free(m->scratch.s_attn_out);
    free(m->scratch.s_aq);
    free(m->scratch.s_proj);
    free(m->scratch.s_f);
    free(m->scratch.s_gate);
    free(m->scratch.s_up);
    free(m->scratch.s_x);
    free(m->scratch.s_h);
    free(m->scratch.s_xout);
    free(m->scratch.s_logits);
    att_ws_free(m);
    gguf_close(m->f);
    free(m);
}

/* ===========================================================================
 * config accessors
 * ===========================================================================
 */

u32 model_n_layer(const Model* m) { return m->n_layer; }
u32 model_n_embd(const Model* m) { return m->n_embd; }
u32 model_n_ff(const Model* m) { return m->n_ff; }
u32 model_n_head(const Model* m) { return m->n_head; }
u32 model_n_head_kv(const Model* m) { return m->n_head_kv; }
u32 model_head_dim(const Model* m) { return m->head_dim; }
u32 model_max_ctx(const Model* m) { return m->max_ctx; }
u32 model_n_vocab(const Model* m) { return m->n_vocab; }
f32 model_freq_base(const Model* m) { return m->freq_base; }
f32 model_eps(const Model* m) { return m->eps; }

void model_check_tensor_shape(const GgufTensor* t, u32 n_dims_expected,
                               const u64* want) {
    if (t->n_dims != n_dims_expected) {
        fatal("model: tensor '%s' has n_dims=%u, expected %u", t->name,
              t->n_dims, n_dims_expected);
    }
    for (u32 i = 0; i < n_dims_expected; i++) {
        if (t->dims[i] != want[i]) {
            fatal("model: tensor '%s' dims[%u]=%llu, expected %llu", t->name,
                  i, (unsigned long long)t->dims[i],
                  (unsigned long long)want[i]);
        }
    }
}

/* ===========================================================================
 * embedding lookup (one row, never the full matrix -- M2 op-spec section 4:
 * "never materialize a dequantized tensor")
 * ===========================================================================
 */

static void embed_row(const Model* m, u32 token_id, f32* out) {
    const GgufTensor* t = m->tok_embd;
    u64                n_embd = m->n_embd;
    const u8*          base = (const u8*)t->data;

    if (t->type == GGUF_GGML_TYPE_F32) {
        memcpy(out, base + (u64)token_id * n_embd * sizeof(f32),
               (size_t)(n_embd * sizeof(f32)));
    } else if (t->type == GGUF_GGML_TYPE_Q8_0) {
        u64 row_bytes =
            (n_embd / KERNELS_Q8_0_BLOCK_ELEMS) * KERNELS_Q8_0_BLOCK_BYTES;
        /* K3: bit-identical twin, so the embedding row is unchanged. */
        if (kernels_cpu_has_avx_vnni()) {
            dequant_row_q8_0_avx2(base + (u64)token_id * row_bytes, n_embd,
                                   out);
        } else {
            dequant_row_q8_0(base + (u64)token_id * row_bytes, n_embd, out);
        }
    } else if (t->type == GGUF_GGML_TYPE_Q4_K) {
        u64 row_bytes =
            (n_embd / KERNELS_Q4_K_BLOCK_ELEMS) * KERNELS_Q4_K_BLOCK_BYTES;
        dequant_row_q4_k(base + (u64)token_id * row_bytes, n_embd, out);
    } else if (t->type == GGUF_GGML_TYPE_Q6_K) {
        /* The shipping Q4_K_M stores token_embd as Q6_K, so this is the LIVE
         * branch for the bar model, not a defensive one. */
        u64 row_bytes =
            (n_embd / KERNELS_Q6_K_BLOCK_ELEMS) * KERNELS_Q6_K_BLOCK_BYTES;
        dequant_row_q6_k(base + (u64)token_id * row_bytes, n_embd, out);
    } else {
        /* check_weight_tensor / the token_embd type check at model_open
         * already rejected everything else, so reaching here means those two
         * lists and this switch have drifted apart -- which is exactly the
         * kind of silent-wrong-output gap worth a loud fatal. */
        fatal("model: embed_row -- tensor '%s' has ggml type %u, which passed "
              "model_open's type check but has no dequant branch here",
              t->name, t->type);
    }
}

/* ===========================================================================
 * --dump support
 * ===========================================================================
 */

static void ensure_dir(const char* dir) {
#ifdef _WIN32
    int rc = _mkdir(dir);
#else
    int rc = mkdir(dir, 0755);
#endif
    if (rc != 0 && errno != EEXIST) {
        fatal("model: cannot create dump directory '%s' -- %s", dir,
              strerror(errno));
    }
}

static void write_dump_layer(const char* dir, u32 layer, const f32* x,
                              u32 n_embd) {
    char path[4096];
    int  n = snprintf(path, sizeof path, "%s/L%02u.f32", dir, layer);
    if (n < 0 || (size_t)n >= sizeof path) {
        fatal("model: dump path too long for directory '%s' layer %u", dir,
              layer);
    }
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fatal("model: cannot open dump file '%s' -- %s", path,
              strerror(errno));
    }
    size_t wrote = fwrite(x, sizeof(f32), n_embd, fp);
    if (wrote != n_embd) {
        fclose(fp);
        fatal("model: short write to dump file '%s' (%zu of %u floats)",
              path, wrote, n_embd);
    }
    if (fclose(fp) != 0) {
        fatal("model: error closing dump file '%s' -- %s", path,
              strerror(errno));
    }
}

/* ===========================================================================
 * model_forward_position -- one position, all n_layer blocks
 * ===========================================================================
 */

/* One sequence's admission checks, hoisted out of the forward body so the
 * batch can run ALL of them before stepping ANY sequence. The messages still
 * say "forward_position" because that is the public entry point a caller
 * recognises, and every one of these is a caller error. */
static void validate_seq(Model* m, Kv* kv, u32 token_id, u32 pos) {
    if (token_id >= m->n_vocab) {
        fatal("model: forward_position -- token id %u out of vocab range "
              "[0,%u)",
              token_id, m->n_vocab);
    }
    if (pos >= kv_max_ctx(kv)) {
        fatal("model: forward_position -- pos %u out of range [0,%u) (kv "
              "max_ctx)",
              pos, kv_max_ctx(kv));
    }
    if (kv_n_layer(kv) != m->n_layer || kv_n_head_kv(kv) != m->n_head_kv ||
        kv_head_dim(kv) != m->head_dim) {
        fatal("model: forward_position -- kv shape (n_layer=%u n_head_kv=%u "
              "head_dim=%u) does not match model (n_layer=%u n_head_kv=%u "
              "head_dim=%u)",
              kv_n_layer(kv), kv_n_head_kv(kv), kv_head_dim(kv), m->n_layer,
              m->n_head_kv, m->head_dim);
    }
    /* max_ctx is checked apart from the other three dims, and only in the
     * too-LARGE direction, because the two directions are not symmetric.
     *
     * `pos` is bounded against the CACHE's capacity just above, but
     * `s_scores` -- the per-head softmax row the attention loop below writes
     * at [0..pos] -- is sized from the MODEL's max_ctx at model_open. So a Kv
     * opened WIDER than the model admits a `pos` that clears the cache check
     * and still runs off the end of s_scores, into the scratch buffers
     * xmalloc'd after it (s_proj/s_f/s_gate). That is silent wrong tokens,
     * not a crash, which is the worst available failure mode.
     *
     * A NARROWER Kv stays legal on purpose: it only limits how far generation
     * can go, and model_generate already reports that against its own
     * total_cap guard (whose test depends on this remaining allowed).
     *
     * Unreachable from today's callers -- every one passes model_max_ctx() --
     * so this is latent rather than live. It is guarded now because M4
     * replaces kv.c's body with a paged block table, and "the cache's
     * capacity equals the model's" is precisely the convention that stops
     * holding then. Sanitizers cannot find it first: ASan only instruments
     * paths a test executes, and no test built an oversized cache until the
     * one that accompanies this check. */
    if (kv_max_ctx(kv) > m->max_ctx) {
        fatal("model: forward_position -- kv max_ctx %u exceeds model max_ctx "
              "%u (s_scores is sized from the model, so a wider cache would "
              "overrun it)",
              kv_max_ctx(kv), m->max_ctx);
    }
}

/* THE forward pass. One body, N sequences, SEQS-INNER/WEIGHTS-OUTER.
 *
 * ▶ THE LOOP ORDER IS THE WHOLE POINT (static-batch design note section 5.2).
 * Per layer, each weight is applied ONCE to all N activations via
 * matvec_batch, so the weight bytes are read once and amortized across the
 * batch. The naive order -- a whole forward per sequence -- re-reads every
 * weight N times and amortizes nothing, which is what this replaced.
 *
 * ▶ AND IT IS ALSO WHY BATCH-COMPOSITION INDEPENDENCE IS EXACT RATHER THAN
 * APPROXIMATE. Every value a sequence computes comes from its own SeqScratch
 * and its own Kv; nothing sums across sequences, and the accumulation order
 * inside each dot product is untouched by how many sequences share the call.
 * So sequence s's output is bit-identical whatever N is and wherever s sits
 * in the batch -- ROADMAP section M4's EXACT bar, earned by construction.
 * tests/test_model.c's interleaved-identity test checks it rather than
 * trusting this paragraph.
 *
 * ▶ THERE IS DELIBERATELY NO SEPARATE SINGLE-SEQUENCE BODY.
 * model_forward_position is this at n_seq=1 over m->scratch. A second body
 * would drift from this one, and -- worse -- the byte-compare against the
 * archived m3 gate binary runs through model_forward_position, so a split
 * would leave the batch path with no golden reference at all. One body means
 * that comparison covers both.
 *
 * The per-sequence work that remains sequence-outer (QK-norm, rope, the KV
 * append, attention) is exactly the work with no weight to amortize: it
 * reads that sequence's own cache and nothing else. */
/* One sequence's attention, at one layer, for a RANGE OF Q-HEADS.
 *
 * ▶▶ WHY THIS IS BIT-EXACT AT ANY THREAD COUNT, and it is the same argument
 * pool.h makes for matvec one level up. A q-head's output depends on that
 * head's Q row, the kv-head it reads, and nothing else; it is written to its
 * own slice of s_attn_out; and the arithmetic inside a head -- the order of
 * every add in every dot product, the softmax, the V accumulation -- is
 * character-for-character what it was when the loop was serial. The
 * parallelism is over INDEPENDENT OUTPUTS, so it reassociates nothing and
 * needs no twin, no grading run and no re-anchor. The byte-compare against the
 * archived binary is what checks that claim rather than this paragraph.
 *
 * The one thing that was NOT already per-head is the scratch: the scores row
 * and the KV dequantization buffer were one-per-sequence, which is exactly one
 * too few the moment two heads run at once. They come from Model::att_ws,
 * indexed by worker. */
typedef struct {
    Model*      m;
    SeqScratch* sc;
    Kv*         kvs;
    u32         L;
    u32         pos;
    f32         scale;
} AttHeads;

/* M5 E2 -- THE SAME ATTENTION, WALKED KV-HEAD-OUTER. One unit is one kv-head
 * and the whole GQA group rides its K and V.
 *
 * ▶▶ WHAT THIS BUYS, AND IT IS TRAFFIC, NOT ARITHMETIC. The loop above is
 * q-head-outer, so with GQA 4:1 each kv-head's K and V are streamed FOUR times
 * per layer per token -- once per q-head that reads them. At ctx 1500 that is
 * 1.77 GB of KV reads per token against 2.50 GB of weights, and decode there
 * measures 106.75 ms against a 97.9 ms pure-traffic floor, i.e. 92% of ceiling:
 * the traffic model is nearly the whole story and a traffic cut converts almost
 * one-for-one. Fetching once per GROUP takes that 1.77 GB to 0.44 GB. Numbers
 * and the M5 bar arithmetic they feed: design note 2026-08-19-m5-entry.md
 * section 3.
 *
 * ▶▶ WHY IT IS BIT-EXACT, stated as the thing that would make it not. A
 * reassociation would need some sum's operands to arrive in a new order. None
 * do: every score is still one q-head's dot product summed left to right over
 * d, character-for-character `vec_dot_f32`; every output element still
 * accumulates over j in ascending order with the same weights; the softmax sees
 * the same row it saw before. What changed is the LOOP NEST and how many times
 * a byte is fetched from the same address -- neither is arithmetic. As with L4,
 * the parallelism is across independent OUTPUTS, so no twin, no grading run and
 * no re-anchor; the byte-compare against the archived m45 binary is what checks
 * that claim rather than this paragraph.
 *
 * ▶ THE 4-WIDE BLOCK IS NOT A SPECIAL CASE FOR THIS MODEL. Any gqa_group runs:
 * groups of 4 take the 4-chain block for the FP-latency reason in this file's
 * header, and the remainder falls through to vec_dot_f32. gqa_group == 1 (a
 * plain-MHA model) takes the tail exclusively, giving ONE chain and no reuse --
 * correct, and slower there than the deleted lanes loop would have been. Stated
 * rather than hidden: with one q-head per kv-head there is no GQA redundancy to
 * remove, so the trade E2 makes has nothing to buy. M7's Llama-3.2 target is
 * also GQA, so this costs nothing on the roadmap as it stands; a true MHA model
 * would want the position-lane loop back, and its argument is preserved in the
 * file header and in git at the deletion commit. */
static void att_kvheads_run(void* vctx, u64 kh_begin, u64 kh_end) {
    const AttHeads*  a = (const AttHeads*)vctx;
    Model*           m = a->m;
    SeqScratch*      sc = a->sc;
    Kv*              kvs = a->kvs;
    const u32        L = a->L;
    const f32        scale = a->scale;
    const u32        hd = m->head_dim;
    const u32        G = m->gqa_group;
    const u32        n = a->pos + 1u; /* causal: positions 0..pos inclusive */
    const u64        row = (u64)m->max_ctx;

    const bool use_simd = (g_att_simd != 0u) && kernels_cpu_has_avx_vnni();

    f32* ws = m->att_ws[pool_worker_index()];
    f32* scores = ws;                       /* [G][max_ctx], row g per q-head */
    f32* kvbuf = ws + (u64)G * row;         /* ONE live (K,V) pair is enough:
                                             * the group shares the fetch, so
                                             * nothing outlives its iteration */
    f32* qt = kvbuf + ATT_KV_FLOATS(hd);    /* [head_dim][G], see below */

    for (u64 kk = kh_begin; kk < kh_end; kk++) {
        const u32  kvh = (u32)kk;
        const f32* qbase = sc->s_q + (u64)kvh * G * hd;
        f32*       obase = sc->s_attn_out + (u64)kvh * G * hd;

        /* ▶▶▶ Q TRANSPOSED ONCE PER UNIT, so the score pass gets LANES without
         * touching a summation order.
         *
         * The dot product `sum_d q[g][d] * K[j][d]` is a REDUCTION over d, and
         * a float reduction cannot be vectorized without reassociating it --
         * which is why no compiler flag has ever touched this loop and why
         * -mavx2 measured 1.06x. Put the FOUR q-heads of the group in the four
         * LANES instead: lane g accumulates q[g][d]*kj[d] over ascending d,
         * which is character for character the scalar chain it replaces. Each
         * lane is its own reduction, in its own order. Bit-exact by
         * construction, not by tolerance.
         *
         * ⚠ Q IS THE CHEAP SIDE TO TRANSPOSE and that is the whole trick: G*hd
         * is 2 KB per unit against K's n*hd, which is 269 KB at ctx 525. A
         * transposed K would also work and would cost 130x more to build.
         *
         * ⚠ SSE2 IS BASELINE ON x86-64, so this needs no TU move and no build
         * flag -- model.c is deliberately compiled without -mavx2 (SIMD_CFLAGS
         * is scoped to simd.c so gcc cannot emit AVX2 into the scalar reference
         * twins) and 128 bits is exactly G=4 lanes anyway.
         *
         * ⚠ G != 4 STILL WORKS AND GAINS NOTHING: the group blocks by four and
         * the remainder falls through to `vec_dot_f32`, exactly as the scalar
         * version did. Qwen3-4B and Llama-3.2's family are both G=4; a G=3
         * model takes the tail exclusively, which is what it did before. */
        for (u32 g = 0; g < G; g++) {
            const f32* qg = qbase + (u64)g * hd;
            for (u32 d = 0; d < hd; d++) qt[(u64)d * G + g] = qg[d];
        }

        /* ---- score pass: K[j] fetched once, scored against all G q-heads */
#ifdef STOKE_MUT_NO_SCORE
        /* MUTANT, SPEED PROBE ONLY -- OUTPUT DELIBERATELY WRONG, and its twin
         * is -DSTOKE_MUT_NO_VALUE. Attention is 18% of prefill for 4.9% of its
         * FLOPs and has two passes over the same bytes: the SCORE pass is a
         * scalar float reduction, the VALUE pass is a 4-wide AXPY. Whether
         * vectorizing the score pass is worth anything depends on which of the
         * two the 1296 ms is in -- and if they cost the SAME despite 4x the
         * lanes, attention is bound by streaming K and V, not by arithmetic,
         * and no amount of SIMD will help. Zeroed rather than left
         * uninitialised so the softmax below sees a clean uniform row instead
         * of whatever NaNs the stack held. */
        for (u32 g = 0; g < G; g++) {
            f32* srow = scores + (u64)g * row;
            for (u32 j = 0; j < n; j++) srow[j] = 0.0f;
        }
        const u32 n_score = 0u * n;   /* runtime zero: no -Wtype-limits */
        for (u32 j = 0; j < n_score; j++) {
#else
        for (u32 j = 0; j < n; j++) {
#endif
#ifdef STOKE_MUT_ALIAS_KV
            /* MUTANT, SPEED PROBE ONLY -- OUTPUT DELIBERATELY WRONG. Every j
             * reads POSITION 0, so the K/V working set collapses from n*512 B
             * to 512 B and is trivially L1-resident, while the kv_read CALL
             * count, the arithmetic and the loop structure are untouched. It
             * separates the two candidate bottlenecks the pass split could not:
             * streaming K/V, versus the work done on it. */
            const f32* kj = kv_read(kvs, L, kvh, 0u, kvbuf).k;
#else
            const f32* kj = kv_read(kvs, L, kvh, j, kvbuf).k;
#endif
            u32        g = 0;
            for (; g + 4u <= G; g += 4u) {
#if STOKE_ATT_SSE
                f32 out4[4];
                if (use_simd) {
                    att_score4_avx2(qt + g, G, kj, hd, scale, out4);
                } else {
#endif
#if !STOKE_ATT_SSE
                const f32* q0 = qbase + (u64)(g + 0u) * hd;
                const f32* q1 = qbase + (u64)(g + 1u) * hd;
                const f32* q2 = qbase + (u64)(g + 2u) * hd;
                const f32* q3 = qbase + (u64)(g + 3u) * hd;
                f32        a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                for (u32 d = 0; d < hd; d++) {
                    a0 += q0[d] * kj[d];
                    a1 += q1[d] * kj[d];
                    a2 += q2[d] * kj[d];
                    a3 += q3[d] * kj[d];
                }
                const f32 out4[4] = {a0 * scale, a1 * scale, a2 * scale,
                                     a3 * scale};
#else
                __m128 acc = _mm_setzero_ps();
                for (u32 d = 0; d < hd; d++) {
                    acc = _mm_add_ps(
                        acc, _mm_mul_ps(
                                 _mm_loadu_ps(qt + (u64)d * G + g),
                                 _mm_load1_ps(kj + d)));
                }
                acc = _mm_mul_ps(acc, _mm_set1_ps(scale));
                _mm_storeu_ps(out4, acc);
                }
#endif
                scores[(u64)(g + 0u) * row + j] = out4[0];
                scores[(u64)(g + 1u) * row + j] = out4[1];
                scores[(u64)(g + 2u) * row + j] = out4[2];
                scores[(u64)(g + 3u) * row + j] = out4[3];
            }
            for (; g < G; g++) {
                scores[(u64)g * row + j] =
                    vec_dot_f32(qbase + (u64)g * hd, kj, hd) * scale;
            }
        }

#ifndef STOKE_MUT_NO_SOFTMAX
        for (u32 g = 0; g < G; g++) softmax(scores + (u64)g * row, n);
#else
        /* MUTANT, SPEED PROBE ONLY -- OUTPUT DELIBERATELY WRONG. The score /
         * value split leaves a REMAINDER that both arms keep, and after the
         * Qt and AVX2 work that remainder is 25% of attention. This prices the
         * part of it that is not bookkeeping: G softmaxes of length n per
         * unit, i.e. one expf per (query, key, q-head). */
        (void)0;
#endif

        for (u32 g = 0; g < G; g++) {
            f32* o = obase + (u64)g * hd;
            for (u32 d = 0; d < hd; d++) o[d] = 0.0f;
        }

#ifdef STOKE_MUT_HOIST_KV
        const f32* vhoist = kv_read(kvs, L, kvh, 0u, kvbuf).v;
#endif
        /* ---- value pass: V[j] fetched once, accumulated into all G outputs.
         * Ordered j-outer/g-inner so each output's adds still arrive in
         * ascending j -- g-outer would be the same fetch count but would put
         * the reuse a whole pass apart, which is the redundancy this rung is
         * here to remove. */
#ifdef STOKE_MUT_NO_VALUE
        const u32 n_score = 0u * n;   /* runtime zero: no -Wtype-limits */
        for (u32 j = 0; j < n_score; j++) {
#else
        for (u32 j = 0; j < n; j++) {
#endif
#if defined(STOKE_MUT_HOIST_KV)
            /* MUTANT, SPEED PROBE ONLY -- OUTPUT DELIBERATELY WRONG. The
             * kv_read CALL leaves the loop entirely (bounds checks,
             * bp_seq_ensure, bp_seq_block, bp_block_slot and a multiply chain,
             * once per j around a 128-element inner loop). Its twin
             * -DSTOKE_MUT_ALIAS_KV keeps the call and removes only the
             * TRAFFIC, so the pair separates the two. */
            const f32* vj = vhoist;
#elif defined(STOKE_MUT_ALIAS_KV)
            const f32* vj = kv_read(kvs, L, kvh, 0u, kvbuf).v;
#else
            const f32* vj = kv_read(kvs, L, kvh, j, kvbuf).v;
#endif
            u32 g = 0;
            /* Four outputs per call so the call cost stays under a tenth of the
             * work: one call per (j, group of four) against 4*head_dim MACs. */
            if (use_simd) {
                for (; g + 4u <= G; g += 4u) {
                    const f32 w4[4] = {scores[(u64)(g + 0u) * row + j],
                                       scores[(u64)(g + 1u) * row + j],
                                       scores[(u64)(g + 2u) * row + j],
                                       scores[(u64)(g + 3u) * row + j]};
                    att_accum4_avx2(obase + (u64)(g + 0u) * hd,
                                    obase + (u64)(g + 1u) * hd,
                                    obase + (u64)(g + 2u) * hd,
                                    obase + (u64)(g + 3u) * hd, vj, w4, hd);
                }
            }
            for (; g < G; g++) {
                const f32 wgt = scores[(u64)g * row + j];
                f32*      o = obase + (u64)g * hd;
                for (u32 d = 0; d < hd; d++) o[d] += wgt * vj[d];
            }
        }
    }
}

/* M5 E1 -- ONE dispatch per layer for the WHOLE batch, over the flattened
 * (sequence, attention-unit) space.
 *
 * ▶▶ WHAT IT REPLACES. The attention dispatch used to sit inside the sequence
 * loop, so a step cost n_seq broadcasts per layer: at N=16 that is 16 x 36 =
 * 576 broadcast-wakeup-join round trips per token where 36 would do. A dispatch
 * is a few microseconds and the saving is ~1% of an N=16 step -- which is why
 * E1 is NOT on this milestone's list for its own number. It is here because it
 * is what makes batched prefill (rung E3) correct BY CONSTRUCTION rather than
 * by loop order: with every sequence's K/V written before any attention runs, a
 * chunk position attending over 0..pos picks up its chunk-mates naturally, with
 * no mask and no special case.
 *
 * ▶ THE HAZARD IT HAS TO NOT HAVE, stated so it can be attacked: a write
 * landing in a block another sequence is concurrently reading. It cannot.
 * Adoption is admission-time only and whole-block only (kv.h, kv.c:142), so a
 * sequence writes only into blocks it alone owns -- its adopted prefix blocks
 * are full and never rewritten, and decode never revisits a position. If the
 * batch byte-compare ever moves, THIS is the claim to attack first.
 *
 * ▶ AND THE ONE THAT IS ALREADY DISCHARGED: bp_seq_ensure, which kv_read
 * reaches through kv_vec_offset, is a pure read once the position is backed
 * (blockpool.c:234) -- and phase 1 has backed every position the batch will
 * read before phase 2 starts. So the parallel region mutates no pool state,
 * which is what lets several sequences of ONE shared store run at once. */
typedef struct {
    Model*             m;
    SeqScratch* const* scs;
    Kv* const*         kv;
    const u32*         positions;
    u64                per_seq; /* attention units one sequence contributes */
    void (*run)(void*, u64, u64);
    u32 L;
    f32 scale;
} AttBatch;

static void att_batch_run(void* vctx, u64 u_begin, u64 u_end) {
    const AttBatch* b = (const AttBatch*)vctx;
    const u64       per = b->per_seq;

    /* A claim can straddle sequences, so it is walked as a run of per-sequence
     * segments rather than assumed to lie inside one. Getting this wrong would
     * hand a worker head indices past its sequence's own arrays -- silently, in
     * the arithmetic, since the neighbouring sequence's scratch is a valid
     * allocation full of plausible floats. */
    for (u64 u = u_begin; u < u_end;) {
        const u32 s = (u32)(u / per);
        const u64 seq_end = (u64)(s + 1u) * per;
        const u64 stop = (u_end < seq_end) ? u_end : seq_end;

        AttHeads aj;
        aj.m = b->m;
        aj.sc = b->scs[s];
        aj.kvs = b->kv[s];
        aj.L = b->L;
        aj.pos = b->positions[s];
        aj.scale = b->scale;
        b->run(&aj, u - (u64)s * per, stop - (u64)s * per);

        u = stop;
    }
}

/* ---- the two per-layer rmsnorms on the pool -------------------------------
 *
 * The same shape as the two above, and it was measured before it was written:
 * 70.6 ms serial against a `residual add` of 31.4, which is why the norms are
 * dispatched and the adds are not -- at 31 ms across 148 layer-calls a
 * dispatch would cost about what the work does. */
typedef struct {
    SeqScratch* const* scs;
    f32**              dst;
    const f32*         w;
    u32                n;
    f32                eps;
} NormJob;

static void norm_run(void* vctx, u64 s_begin, u64 s_end) {
    const NormJob* j = (const NormJob*)vctx;
    for (u64 s = s_begin; s < s_end; s++) {
        rmsnorm(j->scs[s]->s_x, j->w, j->n, j->eps, j->dst[s]);
    }
}

/* ---- phase 1a on the pool ------------------------------------------------
 *
 * ▶▶ WHY: the completed profile puts the per-head QK-norm and rope at 390 ms
 * of prefill, on the calling thread, beside a `kv append` that is 26.
 * Splitting the old combined bucket is what showed that -- 94% of phase 1 is
 * the half that touches nothing shared, so the block pool never has to become
 * thread-safe for this.
 *
 * ⚠ WHAT MAKES IT SAFE, stated because the OTHER half is not: this touches
 * `sc->s_q` and `sc->s_k`, which are per-sequence scratch, plus two read-only
 * norm weights. `kv_write` reaches `bp_seq_ensure` through `kv_vec_offset` and
 * that call is a pure read only ONCE THE POSITION IS BACKED -- backing it is
 * exactly what phase 1b does, so 1b stays serial.
 *
 * ⚠ Parallelism over INDEPENDENT OUTPUTS again, so no twin and no re-anchor:
 * every (sequence, head) computes what it computed, in the order it computed
 * it. The token stream coming out byte-identical is the test. */
typedef struct {
    SeqScratch* const* scs;
    const u32*         positions;
    const f32*         qn;
    const f32*         kn;
    u32                n_head;
    u32                n_head_kv;
    u32                head_dim;
    f32                eps;
    f32                freq_base;
    const f32*         freq_factors; /* rope_freqs.weight, or NULL */
    RopeKind           rope_kind;
} QkRopeJob;

static void qkrope_run(void* vctx, u64 s_begin, u64 s_end) {
    const QkRopeJob* j = (const QkRopeJob*)vctx;
    /* Both branches are resolved ONCE per dispatch, not per head: a model
       with QK-norm and NEOX rope must run the instruction sequence it ran
       before either became optional. `qn` is NULL exactly when the model
       has no QK-norm (model_open enforces all-or-nothing). */
    const bool do_qk_norm = (j->qn != NULL);
    for (u64 s = s_begin; s < s_end; s++) {
        SeqScratch* sc  = j->scs[s];
        const u32   pos = j->positions[s];
        for (u32 h = 0; h < j->n_head; h++) {
            f32* qh = sc->s_q + (u64)h * j->head_dim;
            if (do_qk_norm) rmsnorm(qh, j->qn, j->head_dim, j->eps, qh);
            if (j->rope_kind == ROPE_NORM) {
                rope_norm(qh, j->head_dim, (u64)pos, j->freq_base,
                          j->freq_factors);
            } else {
                rope_neox(qh, j->head_dim, (u64)pos, j->freq_base,
                          j->freq_factors);
            }
        }
        for (u32 h = 0; h < j->n_head_kv; h++) {
            f32* kh = sc->s_k + (u64)h * j->head_dim;
            if (do_qk_norm) rmsnorm(kh, j->kn, j->head_dim, j->eps, kh);
            if (j->rope_kind == ROPE_NORM) {
                rope_norm(kh, j->head_dim, (u64)pos, j->freq_base,
                          j->freq_factors);
            } else {
                rope_neox(kh, j->head_dim, (u64)pos, j->freq_base,
                          j->freq_factors);
            }
        }
    }
}

/* ---- SwiGLU on the pool ---------------------------------------------------
 *
 * ▶▶ WHY: the completed profile puts swiglu at 473 ms of a 7870 ms prefill --
 * 6% -- and every millisecond of it ran on the CALLING thread while seven
 * workers waited at the barrier. That is the shape L4 found in attention (84.6%
 * of decode on one core) and it is the same fix.
 *
 * ⚠ AND IT IS THE SAME EXEMPTION FROM GRADING, for the same reason L4 claimed
 * it: this is parallelism over INDEPENDENT OUTPUTS. Element i of sequence s
 * depends on `s_gate[i]` and `s_up[i]` of that sequence and nothing else, the
 * arithmetic per element is untouched, and the scratch is per sequence. No
 * twin, no re-anchor, no CONTRACT v2 -- the token stream must come out
 * byte-identical, which is the test rather than the argument.
 *
 * ⚠ THE UNIT IS A SEQUENCE, and n_seq==1 therefore runs INLINE: pool_for_units
 * falls back when the claim count is below POOL_MIN_CHUNKS, so decode -- which
 * is one sequence and where a dispatch would cost more than the work -- keeps
 * exactly the code path it had. A prefill chunk is 128 sequences. */
typedef struct {
    SeqScratch* const* scs;
    u64                n_ff;
} SwigluJob;

static void swiglu_run(void* vctx, u64 s_begin, u64 s_end) {
    const SwigluJob* j = (const SwigluJob*)vctx;
    for (u64 s = s_begin; s < s_end; s++) {
        swiglu(j->scs[s]->s_gate, j->scs[s]->s_up, j->n_ff, j->scs[s]->s_gate);
        /* in place: safe aliasing per kernels.h's swiglu contract */
    }
}

static void forward_batch_body(Model* m, SeqScratch* const* scs,
                               Kv* const* kv, const u32* token_ids,
                               const u32* positions, u32 n_seq,
                               const char* dump_dir, f32* const* x_out) {
    if (n_seq == 0) return;
    PROF_WALL_DECL;

    /* EVERY sequence is validated before ANY of them is stepped.
     *
     * Not a style preference: this body writes KV as it goes, so a fatal
     * raised halfway through would leave the sequences already stepped with
     * a K/V row appended for a position the batch never completed. Decide
     * every refusal before any mutation -- the same rule bp_resize's shrink
     * path had to learn when a refused shrink left blocks orphaned. */
    for (u32 s = 0; s < n_seq; s++) {
        validate_seq(m, kv[s], token_ids[s], positions[s]);
    }

    if (dump_dir != NULL) {
        /* Refused rather than silently interleaved: write_dump_layer names
         * its files by LAYER only, so N sequences would overwrite each
         * other's L00.f32 and the survivor would be whichever ran last. The
         * dump exists to be diffed against ref_forward.py, and a file that
         * silently belongs to a different sequence than the reader assumes
         * is worse than no file. */
        if (n_seq != 1) {
            fatal("model: forward -- dump_dir is single-sequence only "
                  "(n_seq=%u); layer dumps are named by layer, so a batch "
                  "would overwrite them",
                  n_seq);
        }
        ensure_dir(dump_dir);
    }

    /* One compare per forward call; allocates only if the pool started after
     * model_open, which main.c's two paths never do. See att_ws_ensure. */
    BatchViews* vw = &m->views;
    { PROF_DECL;
    att_ws_ensure(m);
    views_bind(vw, scs, n_seq);
    for (u32 s = 0; s < n_seq; s++) embed_row(m, token_ids[s], scs[s]->s_x);
    PROF_ADD(PROF_SETUP); }

    f32 scale = 1.0f / sqrtf((f32)m->head_dim);

    for (u32 L = 0; L < m->n_layer; L++) {
        const Layer* ly = &m->layers[L];

        /* rmsnorm(attn_norm) -> Q/K/V matvec.
         *
         * The (const f32* const*) casts on every x argument add const, never
         * remove it: C will not implicitly convert f32** to const f32* const*
         * (the classic T** rule), and the arrays must stay f32** because
         * vw->gate is an OUTPUT here and an INPUT to ffn_down below. */
        { PROF_DECL;
        NormJob an = {scs, vw->a, (const f32*)ly->attn_norm->data, m->n_embd,
                      m->eps};
        pool_for_units(n_seq, 1u, norm_run, &an);
        PROF_ADD(PROF_NORM); }
        { PROF_DECL;
        matvec_batch(ly->attn_q, (const f32* const*)vw->a, m->n_embd, vw->q,
                     (u64)m->n_head * m->head_dim, vw->aq, n_seq);
        matvec_batch(ly->attn_k, (const f32* const*)vw->a, m->n_embd, vw->k,
                     (u64)m->n_head_kv * m->head_dim, vw->aq, n_seq);
        matvec_batch(ly->attn_v, (const f32* const*)vw->a, m->n_embd, vw->v,
                     (u64)m->n_head_kv * m->head_dim, vw->aq, n_seq);
        PROF_ADD(PROF_QKV); }

        /* NULL when this architecture has no QK-norm (llama). The job's
           branch is on qn, so this is the one place the distinction is
           made. */
        const f32* qn =
            ly->attn_q_norm ? (const f32*)ly->attn_q_norm->data : NULL;
        const f32* kn =
            ly->attn_k_norm ? (const f32*)ly->attn_k_norm->data : NULL;

        /* ▶▶ M5 E1: TWO PHASES, NOT ONE SEQUENCE-OUTER BLOCK. Phase 1 is every
         * sequence's QK-norm, rope, token note and KV append; phase 2 is ONE
         * dispatch over the whole batch's attention. None of it touches a
         * weight, so there is still nothing to amortize -- the split buys a
         * dispatch per layer instead of one per (sequence, layer), and buys E3
         * the property that all of a batch's K/V is resident before any of its
         * attention reads. See att_batch_run. */
        { PROF_DECL;
        /* per head: rmsnorm(q_norm/k_norm) THEN rope_neox (P3.4: QK-norm
           BEFORE rope) */
        QkRopeJob qj = {scs,       positions,    qn,          kn,
                        m->n_head, m->n_head_kv, m->head_dim, m->eps,
                        m->freq_base, m->rope_freqs, m->rope_kind};
        pool_for_units(n_seq, 1u, qkrope_run, &qj);
        PROF_ADD(PROF_QKNORM); }

        /* ---- phase 1b: the KV append, SEPARATE FROM 1a because only 1a is
         * safe to dispatch. `kv_write` reaches `bp_seq_ensure` through
         * `kv_vec_offset`, and that call is a pure read only ONCE THE POSITION
         * IS BACKED -- backing it is what this loop does. Phase 1a touches
         * nothing but per-sequence scratch and read-only weights. */
        { PROF_DECL;
        for (u32 s = 0; s < n_seq; s++) {
            SeqScratch* sc  = scs[s];
            Kv*         kvs = kv[s];
            const u32   pos = positions[s];

            /* M4: record WHICH token this position holds, once per position.
             * kv_view backs a position so K/V can be written; this says what
             * token it is, and only that makes the block eligible for prefix
             * dedup. Pure bookkeeping -- it touches no arithmetic, which is
             * why the token stream stays byte-identical. Done on the first
             * layer only because the answer does not vary by layer. */
            if (L == 0) kv_note_token(kvs, pos, token_ids[s]);

            /* kv append -- the ONLY place K/V ever get written.
             *
             * ▶ THROUGH kv_write/kv_read RATHER THAN kv_view SINCE M4's Q8
             * option. kv_view hands back a WRITABLE f32 POINTER, and a
             * pointer is a promise about representation that a quantized
             * store cannot keep. At f32 these are the same memcpy and the
             * same zero-copy read as before -- byte-identical, checked
             * against the archived m3 gate binary -- and at Q8 they are
             * where quantization happens. kv_write needs no scratch in either
             * mode (it memcpy's or quantizes straight into the store); the
             * dequantization buffer belongs to the READ side and lives per
             * WORKER in Model::att_ws, where L4 moved it when heads stopped
             * running one at a time. */
            for (u32 h = 0; h < m->n_head_kv; h++) {
                kv_write(kvs, L, h, pos, sc->s_k + (u64)h * m->head_dim,
                         sc->s_v + (u64)h * m->head_dim);
            }
        }
        PROF_ADD(PROF_KVAPPEND); }

        /* ---- phase 2: causal GQA attention for the WHOLE batch, ON THE POOL.
         *
         * The heads are dispatched rather than looped because attention was the
         * LAST single-threaded work in the forward pass: matvec has run on the
         * pool since K5, and at ctx 1500 attention is 84.6% of decode, so ~85%
         * of a long-context token was running on ONE core while the other seven
         * sat at the barrier.
         *
         * E2's unit is a KV-HEAD, so one sequence contributes n_head_kv units,
         * not n_head. At GQA 4:1 that is 8 claims instead of 32 -- fewer, and
         * each 4x larger, which trades scheduling balance for the traffic cut.
         * That trade was a real risk on a 6P+4E box (an E-core now holds four
         * q-heads at the barrier, not one) and it was MEASURED: +1.75 ms at the
         * intercept, i.e. 0.984x at ctx 87, against 1.120x at ctx 1500 --
         * gate-reports/2026-08-19-m5-e2-ab-report.md. Flattening the batch into
         * the same unit space is what keeps the claim count workable at N>1:
         * 8 units at N=1, but 128 at N=16, where the balance cost disappears
         * (N=8 measured 1.059x, no regression). */
        AttBatch ab;
        ab.m = m;
        ab.scs = scs;
        ab.kv = kv;
        ab.positions = positions;
        ab.per_seq = m->n_head_kv;
        ab.run = att_kvheads_run;
        ab.L = L;
        ab.scale = scale;

        /* The dispatch threshold is a SPEED heuristic about how much work a
         * unit carries, so the batch is judged by its LONGEST sequence -- the
         * one whose units are biggest. At n_seq=1 this is exactly the test that
         * was here before, which is what keeps the L4 dispatch-count test
         * meaning what it meant. */
        u32 pos_max = positions[0];
        for (u32 s = 1; s < n_seq; s++) {
            if (positions[s] > pos_max) pos_max = positions[s];
        }
        { PROF_DECL;
        if (g_att_par && (u64)pos_max + 1u >= (u64)g_att_min_pos) {
            g_att_dispatches++;
            pool_for_units((u64)n_seq * ab.per_seq, g_att_chunk, att_batch_run,
                           &ab);
        } else {
            /* Below the threshold the dispatch (a broadcast, N wakeups and a
             * join) costs more than the heads do, and this path is also
             * STOKE_ATT_PAR=0's arm: bit-identical, just serial. */
            att_batch_run(&ab, 0, (u64)n_seq * ab.per_seq);
        }
        PROF_ADD(PROF_ATT); }

        /* attn_output matvec -> residual add */
        { PROF_DECL;
        matvec_batch(ly->attn_output, (const f32* const*)vw->attn_out,
                     (u64)m->n_head * m->head_dim, vw->proj, m->n_embd,
                     vw->aq, n_seq);
        PROF_ADD(PROF_ATTOUT); }
        { PROF_DECL;
        for (u32 s = 0; s < n_seq; s++) {
            f32* x = scs[s]->s_x;
            for (u32 i = 0; i < m->n_embd; i++) x[i] += scs[s]->s_proj[i];
        }
        PROF_ADD(PROF_RESID); }

        /* rmsnorm(ffn_norm) -> SwiGLU -> ffn_down -> residual add */
        { PROF_DECL;
        NormJob fn = {scs, vw->f, (const f32*)ly->ffn_norm->data, m->n_embd,
                      m->eps};
        pool_for_units(n_seq, 1u, norm_run, &fn);
        PROF_ADD(PROF_NORM); }
        { PROF_DECL;
        matvec_batch(ly->ffn_gate, (const f32* const*)vw->f, m->n_embd,
                     vw->gate, m->n_ff, vw->aq, n_seq);
        matvec_batch(ly->ffn_up, (const f32* const*)vw->f, m->n_embd, vw->up,
                     m->n_ff, vw->aq, n_seq);
        PROF_ADD(PROF_FFN_GU); }
        { PROF_DECL;
        SwigluJob sj = {scs, m->n_ff};
        pool_for_units(n_seq, 1u, swiglu_run, &sj);
        PROF_ADD(PROF_SWIGLU); }
        /* ⚠ THE RESIDUALS ARE THEIR OWN BUCKET. The enum used to say
         * "ffn_down matvec + residual" while the region ended before the loop,
         * and PROF_ATTOUT's did include its own -- so the two rows were not
         * measuring the same shape. Both matvec rows are now pure matvec and
         * the adds are counted once, together, where their size can be read. */
        { PROF_DECL;
        matvec_batch(ly->ffn_down, (const f32* const*)vw->gate, m->n_ff,
                     vw->proj, m->n_embd, vw->aq, n_seq);
        PROF_ADD(PROF_FFN_DOWN); }
        { PROF_DECL;
        for (u32 s = 0; s < n_seq; s++) {
            f32* x = scs[s]->s_x;
            for (u32 i = 0; i < m->n_embd; i++) x[i] += scs[s]->s_proj[i];
        }
        PROF_ADD(PROF_RESID); }

        if (dump_dir != NULL) {
            /* n_seq == 1 here -- refused above otherwise. */
            write_dump_layer(dump_dir, L, scs[0]->s_x, m->n_embd);
        }
    }

    { PROF_DECL;
    for (u32 s = 0; s < n_seq; s++) {
        memcpy(x_out[s], scs[s]->s_x, (size_t)m->n_embd * sizeof(f32));
    }
    PROF_ADD(PROF_SETUP); }
    PROF_WALL_ADD();
}

/* ===========================================================================
 * M5 E3 -- CHUNKED BATCHED PREFILL
 *
 * ▶▶ WHAT IT FIXES. Prefill ran position by position, and a position streams
 * the WHOLE 2.4973 GB of weights: at 43.6 GB/s that is ~57 ms of pure traffic
 * each, so a 550-token prompt paid ~31 s before generating anything. llama
 * batches prefill at ubatch 512 for exactly this reason. Running a CHUNK of
 * positions as batch rows makes matvec_batch read each weight row once for the
 * whole chunk, so the weight term amortises by the chunk width and prefill
 * stops being bandwidth-bound on weights.
 *
 * ▶▶ IT IS THE SAME ARITHMETIC, and the property that says so is already
 * gated: batch-composition independence is EXACT (M4 gate item [2] -- 31
 * streams byte-identical to solo over N in {1,2,4,8,16}). A chunk is a batch
 * whose rows happen to be consecutive positions of ONE sequence rather than
 * one position of several, and nothing in forward_batch_body distinguishes
 * those cases: each row has its own scratch, and rows differ only in the
 * (token id, position) pair they were handed.
 *
 * ▶▶ INTRA-CHUNK CAUSALITY HOLDS BY CONSTRUCTION, AND THAT IS E1's DOING.
 * E1 split the layer body into "write every row's K/V" then "attend", so by
 * the time row i attends, its chunk-mates at positions < its own are already
 * resident and its chunk-mates at HIGHER positions are irrelevant -- attention
 * reads 0..pos and stops. No mask, no special case, and no dependence on the
 * order the rows happen to be visited. Under the pre-E1 interleaved order this
 * would have been a correctness property of a loop's direction, which is the
 * kind that holds until someone reorders it.
 *
 * ▶ ONE Kv HANDLE, REPEATED. kv.h's one-handle-per-sequence rule is about
 * DISTINCT sequences (the frozen kv_view carries no sequence parameter, so the
 * sequence has to live in the handle). A chunk's rows ARE one sequence, so
 * they share one handle by definition rather than by exception.
 * ===========================================================================
 */

/* Runs positions [from, to) of ONE sequence, in chunks, leaving x_out holding
 * the LAST position's hidden state -- which is all the caller wanted from the
 * loop this replaces. `dump_last` is passed to the final position only, and
 * that position always runs alone: write_dump_layer names its files by LAYER,
 * so a batched chunk would overwrite them and the survivor would be whichever
 * row ran last (forward_batch_body refuses n_seq != 1 with a dump for exactly
 * this reason). The dump exists to be diffed against ref_forward.py. */
static void prefill_range(Model* m, Kv* kv, const u32* ids, u32 from, u32 to,
                          const char* dump_last, f32* x_out) {
    if (from >= to) return;

    u32 chunk = g_prefill_chunk;
    u32 last = to;
    if (dump_last != NULL) last = to - 1u; /* the dumped position runs solo */

    if (chunk > 1u && last > from) {
        /* Buffers for one chunk, allocated ONCE per prefill and not per chunk.
         * M2 op-spec 6's rule is about the per-token loop; a prompt's prefill
         * is entered once per sequence per generate call, and sizing these from
         * Model would mean another grown-on-demand array to keep in step with
         * scratch_n for no measurable gain. */
        u32   width = chunk;
        Kv**  kvs = (Kv**)xmalloc((size_t)width * sizeof(Kv*));
        u32*  toks = (u32*)xmalloc((size_t)width * sizeof(u32));
        u32*  poss = (u32*)xmalloc((size_t)width * sizeof(u32));
        f32** xs = (f32**)xmalloc((size_t)width * sizeof(f32*));
        f32*  xbuf =
            (f32*)xmalloc((size_t)width * (size_t)m->n_embd * sizeof(f32));

        for (u32 t = from; t < last;) {
            u32 c = last - t;
            if (c > chunk) c = chunk;
            for (u32 i = 0; i < c; i++) {
                kvs[i] = kv;
                toks[i] = ids[t + i];
                poss[i] = t + i;
                xs[i] = xbuf + (size_t)i * m->n_embd;
            }
            model_forward_batch(m, kvs, toks, poss, c, xs);
            g_prefill_batch_rows += c;
            /* This chunk's LAST row, copied every chunk rather than the final
             * row computed once at the end. The closed form for "which slot
             * held the last position" is (last-1-from) % chunk, which is
             * correct and is exactly the kind of index that becomes wrong the
             * first time the loop's shape changes. One 10 KB copy per 128
             * positions is not a cost. */
            memcpy(x_out, xbuf + (size_t)(c - 1u) * m->n_embd,
                   (size_t)m->n_embd * sizeof(f32));
            t += c;
        }

        free(kvs);
        free(toks);
        free(poss);
        free(xs);
        free(xbuf);
    } else {
        for (u32 t = from; t < last; t++) {
            model_forward_position(m, kv, ids[t], t, NULL, x_out);
        }
    }

    if (last < to) {
        model_forward_position(m, kv, ids[last], last, dump_last, x_out);
    }
}

/* ===========================================================================
 * model_compute_logits -- output_norm -> tied matvec vs token_embd
 * ===========================================================================
 */


void model_forward_position(Model* m, Kv* kv, u32 token_id, u32 pos,
                            const char* dump_dir, f32* x_out) {
    /* The batch body at N=1, over the single-sequence scratch. Deliberately
     * not a second implementation: see forward_batch_body's header. */
    SeqScratch* scs[1] = {&m->scratch};
    Kv*         kvs[1] = {kv};
    f32*        out[1] = {x_out};
    forward_batch_body(m, scs, kvs, &token_id, &pos, 1u, dump_dir, out);
}

/* Grows the per-sequence scratch sets to cover `n`. EXTRACTED so the two
 * callers -- the forward body and the batched output projection -- cannot
 * drift: the use-after-free below is reachable only by a batch that GROWS, so
 * no fixed-width test would ever see a second copy get it wrong. */
static void scratch_n_ensure(Model* m, u32 n) {
    if (n <= m->n_scratch) return;
    /* Grown on demand rather than sized at open: the batch width is the
     * caller's, and model_open has no way to know it. Never shrunk -- a later
     * smaller batch reuses the same sets. */
    m->scratch_n = (SeqScratch*)xrealloc(m->scratch_n,
                                         (size_t)n * sizeof(SeqScratch));
    for (u32 s = m->n_scratch; s < n; s++) scratch_alloc(m, &m->scratch_n[s]);
    m->scratch_p = (SeqScratch**)xrealloc(m->scratch_p,
                                          (size_t)n * sizeof(SeqScratch*));
    m->n_scratch = n;

    /* EVERY entry is rewritten, not just the newly added ones. The xrealloc of
     * scratch_n above may have MOVED the array, which leaves every previously
     * stored &scratch_n[s] dangling; refilling only [old_n, n) would leave the
     * older entries pointing into freed memory. That is a use-after-free
     * reachable ONLY by a batch that grows -- so no fixed-width test would ever
     * see it. The block pool met this same hazard and answered it with indices
     * instead of pointers; here the array is model-owned and rebuilt in this
     * ONE place, which is what keeps it honest. */
    for (u32 s = 0; s < n; s++) m->scratch_p[s] = &m->scratch_n[s];
}

void model_forward_batch(Model* m, Kv* const* kv, const u32* token_ids,
                         const u32* positions, u32 n_seq, f32* const* x_out) {
    if (n_seq == 0) return;
    scratch_n_ensure(m, n_seq);

    /* One scratch set per sequence -- the array the seqs-inner body indexes.
     * Pointing several entries at the same set is exactly the failure the
     * interleaved-identity test now catches. */
    forward_batch_body(m, m->scratch_p, kv, token_ids, positions, n_seq, NULL,
                       x_out);
}

void model_compute_logits(Model* m, const f32* x, f32* logits_out) {
    rmsnorm(x, (const f32*)m->output_norm->data, m->n_embd, m->eps, m->scratch.s_h);
    const GgufTensor* w = (m->output != NULL) ? m->output : m->tok_embd;
    matvec(w, m->scratch.s_h, m->n_embd, logits_out, m->n_vocab, m->scratch.s_aq);
}

/* ▶▶ THE OUTPUT PROJECTION IS THE LAST UNBATCHED WEIGHT IN THE ENGINE, AND IT
 * IS NOT A SMALL ONE. `token_embd.weight` is 319 MB of the model's 2.497 GB --
 * 12.8% -- and the single-row form above streams ALL of it per row. The batch
 * decode loop called it once per active sequence, so at N=16 a step paid
 * 16 x 319 MB = 5.1 GB for logits against ~2.2 GB for the entire rest of the
 * forward: the output projection was TWICE the model.
 *
 * ⚠ AND IT SILENTLY INVALIDATED A NUMBER ALREADY IN THE ROADMAP. The M5 entry
 * note's aggregate ceilings (N=8 57.8 tok/s, N=16 72.9) were computed with the
 * weight term read ONCE per step, which is what forward_batch_body does and
 * what this function did not. Measured against the real traffic the N=16
 * ceiling was 48.6 tok/s, not 72.9. Batching restores the note's figure --
 * i.e. the note described the engine this commit makes true.
 *
 * Bit-exactness is M4 gate item [2]'s, unchanged and not re-argued: matvec_batch
 * computes each row with the identical single-row kernel, and the K-quant path
 * tiles sequences at KERNELS_BATCH_TILE = 16, so no row sees a shape the gate
 * did not already cover.
 *
 * `x` may alias nothing here: each row's normalised copy lands in that row's
 * OWN s_h, which is why this needs the scratch sets rather than a single
 * buffer -- the defect the single-row form would have had at N>1. */
void model_compute_logits_batch(Model* m, const f32* const* x, u32 n,
                                f32* const* logits_out) {
    if (n == 0) return;
    if (n == 1) {
        /* One row takes the solo kernel, exactly as before: matvec_batch's
         * K-quant path routes n_seq==1 to the same place anyway, and going
         * through here would only add a pointer gather. */
        model_compute_logits(m, x[0], logits_out[0]);
        return;
    }
    scratch_n_ensure(m, n);
    BatchViews* vw = &m->views;
    views_bind(vw, m->scratch_p, n);
    for (u32 s = 0; s < n; s++) {
        rmsnorm(x[s], (const f32*)m->output_norm->data, m->n_embd, m->eps,
                vw->h[s]);
    }
    const GgufTensor* w = (m->output != NULL) ? m->output : m->tok_embd;
    matvec_batch(w, (const f32* const*)vw->h, m->n_embd, logits_out, m->n_vocab,
                 vw->aq, n);
}

/* ===========================================================================
 * --probe-pos: top-5 (id, logit), descending, ties by lower id
 * ===========================================================================
 */

static void record_top5(const f32* logits, u32 n, ModelProbeEntry out[5]) {
    for (int i = 0; i < 5; i++) {
        out[i].id = 0xFFFFFFFFu;
        out[i].logit = -INFINITY;
    }
    for (u32 v = 0; v < n; v++) {
        f32 lv = logits[v];
        int ins = -1;
        for (int i = 0; i < 5; i++) {
            if (lv > out[i].logit || (lv == out[i].logit && v < out[i].id)) {
                ins = i;
                break;
            }
        }
        if (ins < 0) continue;
        for (int i = 4; i > ins; i--) out[i] = out[i - 1];
        out[ins].id = v;
        out[ins].logit = lv;
    }
}

/* ===========================================================================
 * model_generate -- prefill + greedy decode
 * ===========================================================================
 */

/* K3 dispatch for the one op whose result IS the generated token. The twin
 * reproduces the scalar's strict-`>` tie rule exactly (tests/test_simd.c
 * proves it over 10,000 tie-dense vectors), so this changes speed and
 * nothing else -- a twin returning any other valid maximum would rewrite
 * token streams on the first tie in a 151,936-wide vector. */
static u32 model_argmax(const f32* logits, u64 n) {
    return kernels_cpu_has_avx_vnni() ? argmax_avx2(logits, n)
                                      : argmax(logits, n);
}

/* ▶ THE GREEDY PATH IS UNTOUCHED WHEN `sp` IS NULL, and that is the whole
 * safety argument for adding sampling here: every existing caller -- including
 * every parity gate -- forwards NULL through model_generate below and takes the
 * identical `model_argmax` it always took. There is no shared mode flag to
 * leave set by accident. */
static u32 pick_next(const SampleParams* sp, const Model* m, u32 pos,
                     const u32* history, u32 n_history, f32* scratch) {
    if (!sp) return model_argmax(m->scratch.s_logits, m->n_vocab);
    return sample_token(sp, pos, m->scratch.s_logits, m->n_vocab, history,
                        n_history, scratch);
}

u32 model_generate(Model* m, Kv* kv, const u32* prompt_ids, u32 n_prompt,
                    u32 n_gen, i64 eos_id, u32* out_ids, const char* dump_dir,
                    i32 probe_pos, ModelProbeEntry probe_out[5],
                    ModelTiming* timing) {
    return model_generate_sampled(m, kv, prompt_ids, n_prompt, n_gen, eos_id,
                                  out_ids, dump_dir, probe_pos, probe_out,
                                  timing, NULL);
}

u32 model_generate_sampled(Model* m, Kv* kv, const u32* prompt_ids, u32 n_prompt,
                    u32 n_gen, i64 eos_id, u32* out_ids, const char* dump_dir,
                    i32 probe_pos, ModelProbeEntry probe_out[5],
                    ModelTiming* timing, const SampleParams* sp) {
    if (n_prompt == 0) {
        fatal("model: generate -- prompt_ids must be non-empty");
    }
    if (n_gen == 0) {
        fatal("model: generate -- n_gen must be positive");
    }
    u64 total_cap = (u64)n_prompt + (u64)n_gen;
    if (total_cap > (u64)kv_max_ctx(kv)) {
        fatal("model: generate -- prompt(%u)+n_gen(%u)=%llu exceeds kv "
              "max_ctx %u",
              n_prompt, n_gen, (unsigned long long)total_cap,
              kv_max_ctx(kv));
    }

    /* Timed unconditionally rather than under `if (timing)`: two now_ms()
     * calls per generation are unmeasurable against a token, and a timer
     * that only runs when someone is looking is a timer whose cost is not
     * in the numbers it reports. */
    double t_start = now_ms();

    /* M5 E3: chunked, with the dumped last position still run solo. Was a
     * position-by-position loop, each re-streaming all 2.4973 GB of weights. */
    /* ⚠ TWO BUFFERS, AND BOTH SHAPES MIRROR THE ENGINE'S so that this CLI path
     * and the server produce the same ids for the same (seed, params): the
     * scratch must be SEPARATE from s_logits because sample_token writes one
     * while reading the other, and the history is indexed by ABSOLUTE POSITION
     * and cleared to 0xFFFFFFFF -- 0 is a real vocabulary id, and a history
     * pre-filled with it would penalise that token at every position nothing
     * has written yet. Allocated once per generation, never per token. */
    f32* samp_scratch = NULL;
    u32* samp_hist    = NULL;
    if (sp) {
        samp_scratch = (f32*)xcalloc(m->n_vocab, sizeof(f32));
        samp_hist    = (u32*)xcalloc((size_t)n_prompt + n_gen, sizeof(u32));
        for (u64 i = 0; i < (u64)n_prompt + n_gen; i++) samp_hist[i] = 0xFFFFFFFFu;
    }

    prefill_range(m, kv, prompt_ids, 0u, n_prompt, dump_dir, m->scratch.s_xout);

    u32 pos = n_prompt; /* absolute position the next forward call will use */
    model_compute_logits(m, m->scratch.s_xout, m->scratch.s_logits);
    if (probe_pos == 0) {
        record_top5(m->scratch.s_logits, m->n_vocab, probe_out);
    }
    /* ⚠⚠ THE DRAW IS KEYED ON pos - 1, NOT pos, AND THE OFF-BY-ONE IS NOT
     * COSMETIC: it is what makes this path agree with the engine. `pos` is the
     * position the NEXT forward will occupy; the id being chosen right now came
     * out of the logits for the position just COMPUTED. In the scheduler the
     * same id is produced by the prefill's LAST row, whose position is
     * n_prompt - 1, and the sampler is keyed on that row's position. Keying on
     * `pos` here shifted every draw by one and the CLI produced a different
     * (perfectly fluent) continuation from the server for the same seed --
     * found by running both and diffing, not by reading. */
    u32 next_id = pick_next(sp, m, pos - 1u, samp_hist, pos, samp_scratch);
    if (samp_hist) samp_hist[pos - 1u] = next_id;
    out_ids[0] = next_id;
    u32 count = 1;

    /* Prefill ends with the first id in hand -- see ModelTiming's comment. */
    double t_decode_start = now_ms();

    while (count < n_gen) {
        if (eos_id >= 0 && (i64)next_id == eos_id) break;

        model_forward_position(m, kv, next_id, pos, NULL, m->scratch.s_xout);
        pos++;
        model_compute_logits(m, m->scratch.s_xout, m->scratch.s_logits);
        if (probe_pos >= 0 && (u32)probe_pos == count) {
            record_top5(m->scratch.s_logits, m->n_vocab, probe_out);
        }
        next_id = pick_next(sp, m, pos - 1u, samp_hist, pos, samp_scratch);
        if (samp_hist) samp_hist[pos - 1u] = next_id;
        out_ids[count] = next_id;
        count++;
    }
    free(samp_scratch);
    free(samp_hist);

    if (timing != NULL) {
        double t_end = now_ms();
        timing->prefill_ms = t_decode_start - t_start;
        timing->decode_ms  = t_end - t_decode_start;
        timing->n_prefill  = n_prompt;
        timing->n_decode   = count - 1u; /* count >= 1 always; see above */
    }
    return count;
}

/* ===========================================================================
 * model_generate_batch -- the static batch driver
 *
 * Structure mirrors model_generate deliberately: same prefill, same
 * first-id-ends-prefill boundary, same eos rule, same argmax. What differs is
 * only that N sequences step together during decode. Keeping the two loops
 * side by side is what makes a divergence between them visible to a reader;
 * what makes it visible to the BUILD is gate item 2, which compares their
 * token streams directly (tests/test_model.c, and the real-model gate).
 * ===========================================================================
 */
void model_generate_batch(Model* m, Kv* const* kv, ModelBatchSeq* seqs,
                          u32 n_seq, i64 eos_id, bool dedup_prefix,
                          ModelTiming* timing) {
    if (n_seq == 0) {
        fatal("model: generate_batch -- n_seq must be positive");
    }
    for (u32 s = 0; s < n_seq; s++) {
        if (seqs[s].n_prompt == 0) {
            fatal("model: generate_batch -- seq %u prompt_ids must be "
                  "non-empty", s);
        }
        if (seqs[s].n_gen == 0) {
            fatal("model: generate_batch -- seq %u n_gen must be positive", s);
        }
        u64 total_cap = (u64)seqs[s].n_prompt + (u64)seqs[s].n_gen;
        if (total_cap > (u64)kv_max_ctx(kv[s])) {
            fatal("model: generate_batch -- seq %u prompt(%u)+n_gen(%u)=%llu "
                  "exceeds kv max_ctx %u",
                  s, seqs[s].n_prompt, seqs[s].n_gen,
                  (unsigned long long)total_cap, kv_max_ctx(kv[s]));
        }
        /* Two slots sharing one handle is not a batch of two sequences -- it
         * is one sequence written twice, and every position after the first
         * step is overwritten by whichever slot ran last. Cheap to check once
         * here (n_seq is 16 at M4's widest) and impossible to see later: the
         * output stays plausible. */
        for (u32 t = 0; t < s; t++) {
            if (kv[t] == kv[s]) {
                fatal("model: generate_batch -- seqs %u and %u share one Kv "
                      "handle; each sequence needs its own", t, s);
            }
        }
    }

    double t_start = now_ms();

    /* Allocated ONCE per call, never inside the token loop -- M2's rule is
     * about the per-token loop, and the batch width is the caller's, so
     * model_open cannot have sized this. One contiguous block for the hidden
     * states, indexed by sequence. */
    f32* xbuf = (f32*)xmalloc((size_t)n_seq * m->n_embd * sizeof(f32));
    u32* pos  = (u32*)xmalloc((size_t)n_seq * sizeof(u32));
    u32* next = (u32*)xmalloc((size_t)n_seq * sizeof(u32));
    u32* act  = (u32*)xmalloc((size_t)n_seq * sizeof(u32));
    Kv**  kv_act  = (Kv**)xmalloc((size_t)n_seq * sizeof(Kv*));
    u32*  tok_act = (u32*)xmalloc((size_t)n_seq * sizeof(u32));
    u32*  pos_act = (u32*)xmalloc((size_t)n_seq * sizeof(u32));
    f32** x_act   = (f32**)xmalloc((size_t)n_seq * sizeof(f32*));
    /* One logits row per ACTIVE sequence. n_vocab is 151,936 on the 4B model,
     * so this is ~600 KB per sequence -- allocated once for the whole
     * generation rather than per step, since the decode loop must not
     * allocate (M2 op-spec 6). */
    f32*  logit_buf = (f32*)xmalloc((size_t)n_seq * (size_t)m->n_vocab *
                                    sizeof(f32));
    f32** logit_p = (f32**)xmalloc((size_t)n_seq * sizeof(f32*));

    u32 n_act = 0;
    u64 n_prefill_total = 0;

    /* Prefill: each sequence through its own prompt, then its first id. The
     * per-sequence loop is model_generate's prefill verbatim, except that it
     * may start past 0 when the prefix was already resident.
     *
     * ▶ ADOPTION HAPPENS HERE, PER SEQUENCE, IMMEDIATELY BEFORE ITS OWN
     * PREFILL -- not in a pass up front. A sequence can only adopt blocks
     * that have already been WRITTEN AND NOTED, so within one batch the
     * second sequence sharing a prompt is the first that can hit: doing it
     * up front would find an empty index and silently dedup nothing.
     *
     * ▶ WHY SKIPPING THOSE POSITIONS IS SOUND, not an approximation. A
     * chained-hash match certifies the whole preceding prefix, so an adopted
     * block covers the SAME absolute positions in both sequences; the forward
     * pass depends only on (token id, position, prior KV); so by induction the
     * K/V those positions would have produced is bit-identical to what is
     * already there. That is why gate item 3 can demand identical streams
     * rather than a tolerance. */
    for (u32 s = 0; s < n_seq; s++) {
        u32 start = 0;
        if (dedup_prefix) {
            start = kv_adopt_prefix(kv[s], seqs[s].prompt_ids,
                                    seqs[s].n_prompt);
            /* ▶ THE DRIVER DOES NOT TAKE ADOPTION'S WORD FOR IT. Skipping
             * positions that were reported resident but are not backed is
             * the one failure on this path that leaves the run alive and
             * merely wrong: attention reads whatever the block holds, the
             * stream stays fluent, and nothing crashes. A measured mutation
             * confirmed the fixture-level token comparison CANNOT see it
             * (5-token vocabulary, argmax dominated by recent positions), so
             * the check has to be here rather than in a test. */
            if (kv_len(kv[s]) != start) {
                fatal("model: generate_batch -- seq %u adoption reported %u "
                      "resident positions but the sequence is backed to %u",
                      s, start, kv_len(kv[s]));
            }
        }
        seqs[s].n_adopted = start;

        f32* x = xbuf + (size_t)s * m->n_embd;
        /* M5 E3: chunked. `start` is a multiple of the block size when it comes
         * from adoption (kv_adopt_prefix returns whole blocks) and 0 otherwise,
         * so chunking from it keeps every boundary block-aligned -- S-ALIGN
         * holds without the chunker having to re-derive it. */
        prefill_range(m, kv[s], seqs[s].prompt_ids, start, seqs[s].n_prompt,
                      NULL, x);
        model_compute_logits(m, x, m->scratch.s_logits);
        u32 id = model_argmax(m->scratch.s_logits, m->n_vocab);
        seqs[s].out_ids[0] = id;
        seqs[s].n_out      = 1;
        next[s]            = id;
        pos[s]             = seqs[s].n_prompt;
        n_prefill_total   += seqs[s].n_prompt - start;

        bool done = (seqs[s].n_out >= seqs[s].n_gen) ||
                    (eos_id >= 0 && (i64)id == eos_id);
        if (!done) act[n_act++] = s;
    }

    /* Prefill ends with every sequence's first id in hand -- the same
     * boundary ModelTiming documents for model_generate. */
    double t_decode_start = now_ms();

    while (n_act > 0) {
        for (u32 i = 0; i < n_act; i++) {
            u32 s      = act[i];
            kv_act[i]  = kv[s];
            tok_act[i] = next[s];
            pos_act[i] = pos[s];
            x_act[i]   = xbuf + (size_t)s * m->n_embd;
        }
        model_forward_batch(m, kv_act, tok_act, pos_act, n_act, x_act);

        /* Compacting in place while walking it: `keep` never overtakes `i`,
         * so a survivor is only ever written to a slot already read. */
        /* ONE batched output projection for the whole active set. Was one
         * per sequence, each streaming all 319 MB of token_embd -- 5.1 GB a
         * step at N=16, against ~2.2 GB for the entire rest of the forward. */
        for (u32 i = 0; i < n_act; i++)
            logit_p[i] = logit_buf + (size_t)i * m->n_vocab;
        model_compute_logits_batch(m, (const f32* const*)x_act, n_act, logit_p);

        u32 keep = 0;
        for (u32 i = 0; i < n_act; i++) {
            u32 s = act[i];
            pos[s]++;
            u32 id = model_argmax(logit_p[i], m->n_vocab);
            seqs[s].out_ids[seqs[s].n_out++] = id;
            next[s] = id;

            bool done = (seqs[s].n_out >= seqs[s].n_gen) ||
                        (eos_id >= 0 && (i64)id == eos_id);
            if (!done) act[keep++] = s;
        }
        n_act = keep;
    }

    if (timing != NULL) {
        double t_end = now_ms();
        u64 n_dec = 0;
        for (u32 s = 0; s < n_seq; s++) n_dec += seqs[s].n_out - 1u;
        timing->prefill_ms = t_decode_start - t_start;
        timing->decode_ms  = t_end - t_decode_start;
        timing->n_prefill  = (u32)n_prefill_total;
        timing->n_decode   = (u32)n_dec;
    }

    free(xbuf); free(pos); free(next); free(act);
    free(kv_act); free(tok_act); free(pos_act); free(x_act);
    free(logit_buf); free(logit_p);
}
