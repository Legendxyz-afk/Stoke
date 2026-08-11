/* src/model.h -- M2 Qwen3 scalar forward pass: config load, per-position
 * forward, tied-head logits, and greedy generate.
 *
 * Contract (M2 op-spec design note, docs/superpowers/design-notes/
 * 2026-07-19-m2-op-spec-and-parity-protocol.md, sections 3/4/6, and
 * .superpowers/sdd/m2-probe-notes.md P3.3/P3.4):
 *   - model_open() reads every REQUIRED qwen3.* config key (fatal if
 *     absent/wrong type/inconsistent -- e.g. value_length != key_length,
 *     head_count not divisible by head_count_kv), validates general.
 *     architecture == "qwen3", validates EVERY per-layer tensor's shape
 *     (and, for weight matrices, ggml type -- F32 or Q8_0 only) against
 *     the config-derived expectation, and resolves the tied-LM-head rule
 *     (use `output.weight` if present, else tie to `token_embd.weight`).
 *     Never returns NULL; any mismatch is a fatal() naming the tensor/key.
 *   - All activations f32; weights stay quantized in the mmap and are
 *     dequantized only inside kernels.h's matvec/dequant_row_q8_0 calls
 *     (or this file's own single-row embedding lookup, same rule) --
 *     never a materialized dequantized weight matrix.
 *   - Every scratch buffer the forward pass needs is allocated once, at
 *     model_open() time, and reused by every call -- no allocation in the
 *     per-token loop.
 *   - Per-position, per-layer op order is EXACTLY ref_forward.py's
 *     Qwen3Ref.forward_position (the trust-order authority per the M2
 *     note section 9): rmsnorm(attn_norm) -> Q/K/V matvec -> per-head
 *     rmsnorm(q_norm/k_norm) THEN rope_neox -> kv append via kv_view ->
 *     causal GQA attention (q-head h reads kv-head h/group; scores /
 *     sqrt(head_dim); softmax; weighted V sum) -> attn_output matvec ->
 *     residual add -> rmsnorm(ffn_norm) -> SwiGLU -> ffn_down -> residual
 *     add. After the last layer: output_norm rmsnorm -> tied matvec vs
 *     token_embd -> argmax (kernels.h's first-lowest-index tie rule).
 *   - Determinism: fixed accumulation order throughout (inherited from
 *     kernels.h's own contract; this file adds no reductions of its own
 *     beyond calling those kernels in a fixed sequence).
 *   - No new file-scope mutable globals -- all state lives in Model/Kv,
 *     passed explicitly. Every abort routes through fatal() (common.h).
 */
#ifndef STOKE_MODEL_H
#define STOKE_MODEL_H

#include "common.h"

#include "gguf.h"
#include "kv.h"

typedef struct Model Model;

/* Opens `path`, validates the full qwen3 config + tensor shape table, and
 * allocates every scratch buffer the forward pass will ever need. Keeps
 * its own GgufFile open for as long as the Model is (tensor data is
 * borrowed from the mmap -- see gguf.h's GgufTensor.data comment), so
 * model_close() is the only way to release it. Never returns NULL. */
Model* model_open(const char* path);

/* Closes the owned GgufFile and frees every allocation. `m` may be NULL
 * (no-op). */
void model_close(Model* m);

/* ---- config accessors (read-only view of what model_open() validated) ---- */
u32 model_n_layer(const Model* m);
u32 model_n_embd(const Model* m);
u32 model_n_ff(const Model* m);
u32 model_n_head(const Model* m);
u32 model_n_head_kv(const Model* m);
u32 model_head_dim(const Model* m);
u32 model_max_ctx(const Model* m);   /* qwen3.context_length, clamped to 4096 */
u32 model_n_vocab(const Model* m);   /* token_embd.weight's row count */
f32 model_freq_base(const Model* m);
f32 model_eps(const Model* m);

/* Fatal (naming t->name) unless t->n_dims == n_dims_expected and, for
 * each i in [0, n_dims_expected), t->dims[i] == want[i]. This is exactly
 * the check model_open() runs for every one of the ~398 tensors it loads
 * (M2 op-spec section 3's expectation table) -- exposed publicly so
 * tests/test_model.c can drive it directly against a REAL tensor (any
 * qwen3 GGUF has one) with a deliberately wrong `want`, proving the
 * shape-validation fatal path names the tensor correctly. (This worker's
 * file ownership excludes the tests/data directory tree, so a synthetic
 * qwen3-shaped fixture isn't available for this specific proof -- this
 * accessor is the alternative that needed no new fixture.) */
void model_check_tensor_shape(const GgufTensor* t, u32 n_dims_expected,
                               const u64* want);

/* Top-(id, logit) entry for the --probe-pos waiver-protocol hook. */
typedef struct {
    u32 id;
    f32 logit;
} ModelProbeEntry;

/* One generation's wall-clock split, filled by model_generate.
 *
 * THE BOUNDARY IS STATED HERE BECAUSE A FIELD NAME IS NOT A BASIS. The
 * logits+argmax that yields the FIRST id consumes prefill's final hidden
 * state, so it is counted in prefill_ms -- exactly as llama.cpp's pp phase
 * ends by producing that token. decode_ms therefore covers only the loop
 * whose every iteration is one forward + logits + argmax, and n_decode is
 * the number of ids that loop produced (returned count - 1, so 0 when
 * generation stopped at the first id).
 *
 * n_decode / decode_ms is thus the SAME unit as llama-bench's `tg` rate and
 * is the figure the referent interleave compares. Dividing the returned
 * count by the total would mix a prompt-length-dependent term into a rate
 * that is supposed to be prompt-independent. */
typedef struct {
    double prefill_ms;
    double decode_ms;
    u32    n_prefill;
    u32    n_decode;
} ModelTiming;

/* Runs all n_layer transformer blocks for ONE position `pos`, given input
 * token `token_id`. Reads the previous positions' K/V and writes this
 * position's own K/V via kv_view for every layer (fatal if pos >=
 * kv_max_ctx(kv) or token_id >= model_n_vocab(m)). Writes the final
 * (post-last-layer, PRE-output-norm) hidden state into `x_out`
 * (caller-owned, model_n_embd(m) floats).
 *
 * If `dump_dir` is non-NULL, creates the directory if needed and writes
 * L00.f32..L{n_layer-1}.f32 into it -- the post-block hidden state after
 * each layer, raw f32, byte-compatible with ref_forward.py's
 * --dump-layers output (same content, same naming). All scratch memory
 * used is internal to `m` (allocated once at model_open); this call
 * performs no allocation. */
void model_forward_position(Model* m, Kv* kv, u32 token_id, u32 pos,
                             const char* dump_dir, f32* x_out);

/* Steps `n_seq` sequences by one position each, one Kv handle per sequence.
 *
 * Each sequence gets its OWN SeqScratch, so nothing they compute can reach
 * each other -- that is what makes batch-composition independence possible
 * at all, and it is checked by test_model.c's interleaved-identity test.
 *
 * The bar (ROADMAP section M4): every sequence's x_out must be BIT-IDENTICAL
 * to what model_forward_position would have written for it alone, whatever N
 * is and wherever it sits in the batch.
 *
 * ▶ CURRENTLY SEQUENCE-OUTER. The weight-amortizing form calls matvec_batch
 * per weight with the sequence loop inside; matvec_batch exists and is fused
 * for all four weight types, but the forward BODY still steps one sequence at
 * a time, so this API is correct and not yet faster. Wiring the body is the
 * remaining step, and the identity test already guards it. */
void model_forward_batch(Model* m, Kv* const* kv, const u32* token_ids,
                         const u32* positions, u32 n_seq, f32* const* x_out);

/* output_norm rmsnorm(x) -> tied matvec (output.weight if present, else
 * token_embd.weight) -> logits_out[model_n_vocab(m)]. Caller-owned
 * logits_out buffer; no allocation. */
void model_compute_logits(Model* m, const f32* x, f32* logits_out);

/* Greedy-decodes up to n_gen ids following prompt_ids[0..n_prompt) (the
 * SAME per-position loop as ref_forward.py's generate(): prefill one
 * position at a time, dump_dir applies only to the LAST prompt position,
 * then repeatedly compute-logits+argmax+forward the next position).
 * `kv` must have kv_max_ctx(kv) >= n_prompt+n_gen (fatal otherwise, named
 * by size). Writes generated ids into out_ids (caller-owned, capacity >=
 * n_gen) and returns the count actually written -- < n_gen only if
 * eos_id was generated (eos_id < 0 disables early stop; eos, if
 * generated, IS included in the returned ids, matching the M2 run CLI
 * contract).
 *
 * probe_pos < 0 disables the probe. Otherwise, when generating the id at
 * generated-position-index probe_pos (0 = the first generated id), fills
 * probe_out[0..5) with the top-5 (id, logit) at that position, strictly
 * descending by logit, ties broken by lower id -- recorded at the exact
 * moment that position's logits are computed, independent of (and not
 * disturbing) the argmax that actually picks the next id. If generation
 * stops (eos) before reaching probe_pos, probe_out is left untouched;
 * callers must check probe_pos < (i32)returned_count themselves. */
u32 model_generate(Model* m, Kv* kv, const u32* prompt_ids, u32 n_prompt,
                    u32 n_gen, i64 eos_id, u32* out_ids, const char* dump_dir,
                    i32 probe_pos, ModelProbeEntry probe_out[5],
                    ModelTiming* timing);

#endif /* STOKE_MODEL_H */
