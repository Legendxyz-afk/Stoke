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

#include <stdbool.h>

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

/* How many times the attention heads have been handed to the worker pool.
 *
 * A TEST HOOK, and it is here rather than in the test because the condition it
 * reports is invisible from outside: "the same output at 1 and at 6 threads"
 * is equally true of a build that never dispatched at all. Process-global and
 * monotonic; a test takes the difference across the calls it cares about. */
u64 model_att_dispatch_count(void);

/* How many layer-steps ran the M5 E2 (kv-head-outer) arm rather than M4.5's.
 * The anti-vacuity counter for the arms' bit-identity test: without it, a
 * broken arm selector makes that test compare a code path against itself and
 * pass. Process-global and monotonic, like the dispatch counter. */
u64 model_att_gqa_step_count(void);

/* Selects the attention arm: 0 = M4.5's q-head-outer loop, 1 = M5 E2's
 * kv-head-outer loop (the default). Mirrors STOKE_ATT_GQA.
 *
 * A TEST HOOK, for the same reason the counter above is one. E2's entire claim
 * is that the two arms are bit-identical, so a test has to run both -- and it
 * cannot do that through the environment, because model_open reads
 * STOKE_ATT_GQA and glibc does not declare putenv under -std=c11 (the WSL leg
 * already rejected that approach once, see the L4 test). CALL IT AFTER
 * model_open: the open re-reads the environment and would overwrite this. */
void model_att_set_gqa(u32 on);

/* Prompt positions that went through E3's BATCHED prefill rather than the
 * position-by-position path. The anti-vacuity counter for the chunked-prefill
 * test: "both widths write the same KV" is also true when the chunker never
 * engaged. Process-global and monotonic. */
u64 model_prefill_batch_rows(void);

/* Positions per prefill chunk (M5 E3). 1 = position-by-position, the pre-M5
 * path. Mirrors STOKE_PREFILL_CHUNK, and like it, refuses a value that is
 * neither 1 nor a whole number of KV blocks -- a chunk must end block-aligned
 * or the prefix cache cannot register what it produced (S-ALIGN).
 *
 * A TEST HOOK for the same reason model_att_set_gqa is one: E3's claim is that
 * chunking changes no arithmetic, and a test has to run both widths. CALL IT
 * AFTER model_open, which re-reads the environment. */
void model_set_prefill_chunk(u32 chunk);

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
 * ▶ SEQS-INNER / WEIGHTS-OUTER. Per layer, each weight is applied once to
 * all N activations through matvec_batch, so the weight bytes are read once
 * for the whole batch instead of once per sequence -- which is the entire
 * reason batching wins until the KV crossover (design note section 1.4).
 * The work that has no weight to amortize (QK-norm, rope, the KV append,
 * attention) stays per-sequence and reads only that sequence's own cache.
 *
 * model_forward_position is this same body at n_seq=1; there is no second
 * implementation to drift from it. */
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

/* One sequence's slot in a static batch. */
typedef struct {
    const u32* prompt_ids;  /* borrowed; >= n_prompt entries, n_prompt > 0 */
    u32        n_prompt;
    u32        n_gen;       /* this sequence's own cap, > 0 */
    u32*       out_ids;     /* caller-owned, capacity >= n_gen */
    u32        n_out;       /* OUT: ids actually written */
    u32        n_adopted;   /* OUT: prompt positions taken from the prefix
                             * cache instead of being recomputed. Reported
                             * rather than inferred, because "dedup was
                             * enabled" and "dedup fired" are different
                             * claims and only the second is worth anything.
                             * 0 whenever dedup_prefix is false. */
} ModelBatchSeq;

/* Runs `n_seq` sequences to completion against one Kv handle each (which may
 * share a KvStore), greedily, and writes each one's ids into its own out_ids.
 *
 * THE BAR (ROADMAP section M4, ruling section 8.4 gate item 2): every
 * sequence's token stream must be EXACTLY what model_generate would have
 * produced for it alone, at any n_seq and wherever it sits in the batch.
 *
 * ▶ PREFILL IS PER-SEQUENCE, DECODE IS LOCKSTEP. That is what "static batch"
 * means at M4: the batch is fixed at entry, every member steps once per
 * decode step, and nothing is admitted mid-flight. Interleaving prefill
 * positions with other sequences' decode steps is chunked prefill, which is
 * M5's deliverable and needs the scheduler M4 does not have.
 *
 * ▶ A SEQUENCE THAT FINISHES LEAVES THE BATCH, and the survivors compact into
 * the lower slots. Composition independence is what makes that safe: the slot
 * a sequence occupies changes nothing about its arithmetic.
 *
 * ▶ `dedup_prefix` turns on prefix adoption (kv.h's kv_adopt_prefix): before
 * a sequence is prefilled, whole leading blocks of its prompt whose token ids
 * are already resident in the SHARED store are adopted by refcount and not
 * recomputed. It is opt-in rather than default for one reason: it removes
 * prefill work, so a scaling curve measured with it on would be reporting the
 * corpus's prompt overlap as though it were batching throughput.
 *
 * `eos_id` < 0 disables early stop; an emitted eos IS included in out_ids,
 * matching model_generate. `timing`, if given, reports the batch aggregate:
 * n_prefill summed over sequences, n_decode summed over the ids each sequence
 * produced after its first -- the same unit model_generate reports, so an
 * aggregate rate divides by the same denominator at every N. n_prefill counts
 * positions ACTUALLY FORWARDED, so adopted positions are excluded. */
void model_generate_batch(Model* m, Kv* const* kv, ModelBatchSeq* seqs,
                          u32 n_seq, i64 eos_id, bool dedup_prefix,
                          ModelTiming* timing);

#endif /* STOKE_MODEL_H */
