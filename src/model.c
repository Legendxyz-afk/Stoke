/* src/model.c -- M2 Qwen3 scalar forward pass. See model.h for the full
 * contract. Op order is exactly ref_forward.py's Qwen3Ref.forward_position
 * (docs/superpowers/design-notes/2026-07-19-m2-op-spec-and-parity-protocol.md
 * section 6; NEOX rope math + graph order re-confirmed against b10037
 * source in .superpowers/sdd/m2-probe-notes.md P3.3/P3.4).
 */
#include "common.h"

#include "model.h"

#include "kernels/kernels.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
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

struct Model {
    GgufFile* f; /* owned; kept open for the model's whole lifetime -- tensor
                    data is borrowed from its mmap (gguf.h) */

    u32 n_layer, n_embd, n_ff, n_head, n_head_kv, head_dim, gqa_group;
    u32 max_ctx, n_vocab;
    f32 freq_base, eps;

    const GgufTensor* tok_embd;
    const GgufTensor* output_norm;
    const GgufTensor* output; /* NULL => tied to tok_embd (M2 note TRAP #2) */

    Layer* layers; /* [n_layer], xmalloc'd */

    /* Scratch: allocated once here, reused by every forward_position/
     * compute_logits/generate call -- "no malloc inside the per-token
     * loop" (M2 op-spec section 6). */
    f32* s_a;        /* [n_embd]              post-attn_norm            */
    f32* s_q;        /* [n_head*head_dim]     Q, all heads flattened    */
    f32* s_k;        /* [n_head_kv*head_dim]  K, all kv-heads flattened */
    f32* s_v;        /* [n_head_kv*head_dim]  V, all kv-heads flattened */
    f32* s_attn_out; /* [n_head*head_dim]     per-head attn output      */
    f32* s_scores;   /* [max_ctx]             one head's softmax row    */
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
    if (t->type != GGUF_GGML_TYPE_F32 && t->type != GGUF_GGML_TYPE_Q8_0) {
        fatal("model: tensor '%s' has ggml type %u -- only F32(%u)/Q8_0(%u) "
              "are supported at M2",
              t->name, t->type, (u32)GGUF_GGML_TYPE_F32,
              (u32)GGUF_GGML_TYPE_Q8_0);
    }
}

/* ===========================================================================
 * model_open / model_close
 * ===========================================================================
 */

Model* model_open(const char* path) {
    GgufFile* f = gguf_open(path);

    const char* arch = gguf_meta_str(f, "general.architecture");
    if (strcmp(arch, "qwen3") != 0) {
        char arch_buf[128];
        snprintf(arch_buf, sizeof arch_buf, "%s", arch);
        gguf_close(f);
        fatal("model: '%s' -- general.architecture is '%s', expected "
              "exactly 'qwen3' (this loader implements qwen3 only)",
              path, arch_buf);
    }

    u32 n_layer = req_u32(f, "qwen3.block_count");
    i64 ctx_len = gguf_meta_i64(f, "qwen3.context_length");
    u32 n_embd = req_u32(f, "qwen3.embedding_length");
    u32 n_ff = req_u32(f, "qwen3.feed_forward_length");
    u32 n_head = req_u32(f, "qwen3.attention.head_count");
    u32 n_head_kv = req_u32(f, "qwen3.attention.head_count_kv");
    u32 head_dim = req_u32(f, "qwen3.attention.key_length");
    u32 value_len = req_u32(f, "qwen3.attention.value_length");
    f32 freq_base = req_f32(f, "qwen3.rope.freq_base");
    f32 eps = req_f32(f, "qwen3.attention.layer_norm_rms_epsilon");

    if (n_layer == 0 || n_embd == 0 || n_ff == 0 || n_head == 0 ||
        n_head_kv == 0 || head_dim == 0) {
        gguf_close(f);
        fatal("model: '%s' -- one of block_count/embedding_length/"
              "feed_forward_length/head_count/head_count_kv/key_length is 0",
              path);
    }
    if (ctx_len <= 0) {
        gguf_close(f);
        fatal("model: '%s' -- qwen3.context_length=%lld must be positive",
              path, (long long)ctx_len);
    }
    if (value_len != head_dim) {
        gguf_close(f);
        fatal("model: '%s' -- qwen3.attention.value_length=%u != "
              "qwen3.attention.key_length=%u (M2 requires them equal)",
              path, value_len, head_dim);
    }
    if (n_head % n_head_kv != 0) {
        gguf_close(f);
        fatal("model: '%s' -- qwen3.attention.head_count=%u not divisible "
              "by head_count_kv=%u (GQA group size must be exact)",
              path, n_head, n_head_kv);
    }
    if (head_dim % 2 != 0) {
        gguf_close(f);
        fatal("model: '%s' -- qwen3.attention.key_length=%u must be even "
              "(NEOX rope pairs (k, k+head_dim/2))",
              path, head_dim);
    }
    if (!(isfinite(freq_base) && freq_base > 0.0f)) {
        gguf_close(f);
        fatal("model: '%s' -- qwen3.rope.freq_base=%g must be finite and "
              "positive",
              path, (double)freq_base);
    }
    if (!(isfinite(eps) && eps >= 0.0f)) {
        gguf_close(f);
        fatal("model: '%s' -- qwen3.attention.layer_norm_rms_epsilon=%g "
              "must be finite and non-negative",
              path, (double)eps);
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
    if (te_type != GGUF_GGML_TYPE_F32 && te_type != GGUF_GGML_TYPE_Q8_0) {
        gguf_close(f);
        fatal("model: tensor 'token_embd.weight' has ggml type %u -- only "
              "F32(%u)/Q8_0(%u) are supported at M2",
              te_type, (u32)GGUF_GGML_TYPE_F32, (u32)GGUF_GGML_TYPE_Q8_0);
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

    m->layers = (Layer*)xmalloc((size_t)n_layer * sizeof(Layer));
    for (u32 L = 0; L < n_layer; L++) {
        Layer* ly = &m->layers[L];
        ly->attn_norm = layer_tensor(f, L, "attn_norm");
        ly->attn_q = layer_tensor(f, L, "attn_q");
        ly->attn_q_norm = layer_tensor(f, L, "attn_q_norm");
        ly->attn_k = layer_tensor(f, L, "attn_k");
        ly->attn_k_norm = layer_tensor(f, L, "attn_k_norm");
        ly->attn_v = layer_tensor(f, L, "attn_v");
        ly->attn_output = layer_tensor(f, L, "attn_output");
        ly->ffn_norm = layer_tensor(f, L, "ffn_norm");
        ly->ffn_gate = layer_tensor(f, L, "ffn_gate");
        ly->ffn_up = layer_tensor(f, L, "ffn_up");
        ly->ffn_down = layer_tensor(f, L, "ffn_down");

        check_norm_tensor(ly->attn_norm, n_embd);
        check_norm_tensor(ly->attn_q_norm, head_dim);
        check_norm_tensor(ly->attn_k_norm, head_dim);
        check_norm_tensor(ly->ffn_norm, n_embd);

        check_weight_tensor(ly->attn_q, n_embd, (u64)n_head * head_dim);
        check_weight_tensor(ly->attn_k, n_embd, (u64)n_head_kv * head_dim);
        check_weight_tensor(ly->attn_v, n_embd, (u64)n_head_kv * head_dim);
        check_weight_tensor(ly->attn_output, (u64)n_head * head_dim, n_embd);
        check_weight_tensor(ly->ffn_gate, n_embd, n_ff);
        check_weight_tensor(ly->ffn_up, n_embd, n_ff);
        check_weight_tensor(ly->ffn_down, n_ff, n_embd);
    }

    m->s_a = (f32*)xmalloc((size_t)n_embd * sizeof(f32));
    m->s_q = (f32*)xmalloc((size_t)n_head * head_dim * sizeof(f32));
    m->s_k = (f32*)xmalloc((size_t)n_head_kv * head_dim * sizeof(f32));
    m->s_v = (f32*)xmalloc((size_t)n_head_kv * head_dim * sizeof(f32));
    m->s_attn_out = (f32*)xmalloc((size_t)n_head * head_dim * sizeof(f32));
    m->s_scores = (f32*)xmalloc((size_t)max_ctx * sizeof(f32));
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
        m->s_aq_bytes = (max_n_in / KERNELS_Q8_0_BLOCK_ELEMS)
                        * KERNELS_Q8_0_BLOCK_BYTES;
        m->s_aq = (u8*)xmalloc((size_t)m->s_aq_bytes);
    }
    m->s_proj = (f32*)xmalloc((size_t)n_embd * sizeof(f32));
    m->s_f = (f32*)xmalloc((size_t)n_embd * sizeof(f32));
    m->s_gate = (f32*)xmalloc((size_t)n_ff * sizeof(f32));
    m->s_up = (f32*)xmalloc((size_t)n_ff * sizeof(f32));
    m->s_x = (f32*)xmalloc((size_t)n_embd * sizeof(f32));
    m->s_h = (f32*)xmalloc((size_t)n_embd * sizeof(f32));
    m->s_xout = (f32*)xmalloc((size_t)n_embd * sizeof(f32));
    m->s_logits = (f32*)xmalloc((size_t)n_vocab * sizeof(f32));

    return m;
}

void model_close(Model* m) {
    if (!m) return;
    free(m->layers);
    free(m->s_a);
    free(m->s_q);
    free(m->s_k);
    free(m->s_v);
    free(m->s_attn_out);
    free(m->s_scores);
    free(m->s_aq);
    free(m->s_proj);
    free(m->s_f);
    free(m->s_gate);
    free(m->s_up);
    free(m->s_x);
    free(m->s_h);
    free(m->s_xout);
    free(m->s_logits);
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
    } else {
        /* Q8_0: the only other type check_weight_tensor allows at model_open
           time, so this branch is exhaustive. */
        u64 row_bytes =
            (n_embd / KERNELS_Q8_0_BLOCK_ELEMS) * KERNELS_Q8_0_BLOCK_BYTES;
        dequant_row_q8_0(base + (u64)token_id * row_bytes, n_embd, out);
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

void model_forward_position(Model* m, Kv* kv, u32 token_id, u32 pos,
                             const char* dump_dir, f32* x_out) {
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

    if (dump_dir != NULL) {
        ensure_dir(dump_dir);
    }

    f32* x = m->s_x;
    embed_row(m, token_id, x);

    f32 scale = 1.0f / sqrtf((f32)m->head_dim);

    for (u32 L = 0; L < m->n_layer; L++) {
        const Layer* ly = &m->layers[L];

        /* rmsnorm(attn_norm) -> Q/K/V matvec */
        rmsnorm(x, (const f32*)ly->attn_norm->data, m->n_embd, m->eps,
                m->s_a);
        matvec(ly->attn_q, m->s_a, m->n_embd, m->s_q,
               (u64)m->n_head * m->head_dim, m->s_aq);
        matvec(ly->attn_k, m->s_a, m->n_embd, m->s_k,
               (u64)m->n_head_kv * m->head_dim, m->s_aq);
        matvec(ly->attn_v, m->s_a, m->n_embd, m->s_v,
               (u64)m->n_head_kv * m->head_dim, m->s_aq);

        /* per head: rmsnorm(q_norm/k_norm) THEN rope_neox (P3.4: QK-norm
           BEFORE rope) */
        const f32* qn = (const f32*)ly->attn_q_norm->data;
        const f32* kn = (const f32*)ly->attn_k_norm->data;
        for (u32 h = 0; h < m->n_head; h++) {
            f32* qh = m->s_q + (u64)h * m->head_dim;
            rmsnorm(qh, qn, m->head_dim, m->eps, qh);
            rope_neox(qh, m->head_dim, (u64)pos, m->freq_base);
        }
        for (u32 h = 0; h < m->n_head_kv; h++) {
            f32* kh = m->s_k + (u64)h * m->head_dim;
            rmsnorm(kh, kn, m->head_dim, m->eps, kh);
            rope_neox(kh, m->head_dim, (u64)pos, m->freq_base);
        }

        /* kv append via kv_view -- the ONLY place K/V ever get written */
        for (u32 h = 0; h < m->n_head_kv; h++) {
            KvSlot slot = kv_view(kv, L, h, pos);
            memcpy(slot.k, m->s_k + (u64)h * m->head_dim,
                   (size_t)m->head_dim * sizeof(f32));
            memcpy(slot.v, m->s_v + (u64)h * m->head_dim,
                   (size_t)m->head_dim * sizeof(f32));
        }

        /* causal GQA attention: q-head h reads kv-head h/gqa_group */
        for (u32 h = 0; h < m->n_head; h++) {
            u32        kvh = h / m->gqa_group;
            const f32* qh = m->s_q + (u64)h * m->head_dim;

            for (u32 j = 0; j <= pos; j++) {
                KvSlot slot = kv_view(kv, L, kvh, j);
                m->s_scores[j] = vec_dot_f32(qh, slot.k, m->head_dim) * scale;
            }
            softmax(m->s_scores, (u64)pos + 1u);

            f32* out_h = m->s_attn_out + (u64)h * m->head_dim;
            for (u32 d = 0; d < m->head_dim; d++) out_h[d] = 0.0f;
            for (u32 j = 0; j <= pos; j++) {
                KvSlot slot = kv_view(kv, L, kvh, j);
                f32    wgt = m->s_scores[j];
                for (u32 d = 0; d < m->head_dim; d++) {
                    out_h[d] += wgt * slot.v[d];
                }
            }
        }

        /* attn_output matvec -> residual add */
        matvec(ly->attn_output, m->s_attn_out, (u64)m->n_head * m->head_dim,
               m->s_proj, m->n_embd, m->s_aq);
        for (u32 i = 0; i < m->n_embd; i++) x[i] += m->s_proj[i];

        /* rmsnorm(ffn_norm) -> SwiGLU -> ffn_down -> residual add */
        rmsnorm(x, (const f32*)ly->ffn_norm->data, m->n_embd, m->eps, m->s_f);
        matvec(ly->ffn_gate, m->s_f, m->n_embd, m->s_gate, m->n_ff, m->s_aq);
        matvec(ly->ffn_up, m->s_f, m->n_embd, m->s_up, m->n_ff, m->s_aq);
        swiglu(m->s_gate, m->s_up, m->n_ff, m->s_gate); /* in place: safe
            aliasing per kernels.h's swiglu contract */
        matvec(ly->ffn_down, m->s_gate, m->n_ff, m->s_proj, m->n_embd, m->s_aq);
        for (u32 i = 0; i < m->n_embd; i++) x[i] += m->s_proj[i];

        if (dump_dir != NULL) {
            write_dump_layer(dump_dir, L, x, m->n_embd);
        }
    }

    memcpy(x_out, x, (size_t)m->n_embd * sizeof(f32));
}

/* ===========================================================================
 * model_compute_logits -- output_norm -> tied matvec vs token_embd
 * ===========================================================================
 */

void model_compute_logits(Model* m, const f32* x, f32* logits_out) {
    rmsnorm(x, (const f32*)m->output_norm->data, m->n_embd, m->eps, m->s_h);
    const GgufTensor* w = (m->output != NULL) ? m->output : m->tok_embd;
    matvec(w, m->s_h, m->n_embd, logits_out, m->n_vocab, m->s_aq);
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

u32 model_generate(Model* m, Kv* kv, const u32* prompt_ids, u32 n_prompt,
                    u32 n_gen, i64 eos_id, u32* out_ids, const char* dump_dir,
                    i32 probe_pos, ModelProbeEntry probe_out[5],
                    ModelTiming* timing) {
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

    for (u32 t = 0; t < n_prompt; t++) {
        bool last_prefill = (t == n_prompt - 1);
        model_forward_position(m, kv, prompt_ids[t], t,
                                last_prefill ? dump_dir : NULL, m->s_xout);
    }

    u32 pos = n_prompt; /* absolute position the next forward call will use */
    model_compute_logits(m, m->s_xout, m->s_logits);
    if (probe_pos == 0) {
        record_top5(m->s_logits, m->n_vocab, probe_out);
    }
    u32 next_id = argmax(m->s_logits, m->n_vocab);
    out_ids[0] = next_id;
    u32 count = 1;

    /* Prefill ends with the first id in hand -- see ModelTiming's comment. */
    double t_decode_start = now_ms();

    while (count < n_gen) {
        if (eos_id >= 0 && (i64)next_id == eos_id) break;

        model_forward_position(m, kv, next_id, pos, NULL, m->s_xout);
        pos++;
        model_compute_logits(m, m->s_xout, m->s_logits);
        if (probe_pos >= 0 && (u32)probe_pos == count) {
            record_top5(m->s_logits, m->n_vocab, probe_out);
        }
        next_id = argmax(m->s_logits, m->n_vocab);
        out_ids[count] = next_id;
        count++;
    }

    if (timing != NULL) {
        double t_end = now_ms();
        timing->prefill_ms = t_decode_start - t_start;
        timing->decode_ms  = t_end - t_decode_start;
        timing->n_prefill  = n_prompt;
        timing->n_decode   = count - 1u; /* count >= 1 always; see above */
    }
    return count;
}
