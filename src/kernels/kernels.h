/* src/kernels/kernels.h -- M2 scalar forward-pass kernels: pure math over
 * flat f32 buffers and GgufTensor weight matrices.
 *
 * Contract (M2 op-spec design note, docs/superpowers/design-notes/
 * 2026-07-19-m2-op-spec-and-parity-protocol.md, §4-5):
 *   - f32 accumulation, sequential, fixed order everywhere -- no
 *     pairwise/tree reduction tricks. Determinism (same input -> same
 *     output) is required; bit-matching llama.cpp's own reduction order is
 *     explicitly not.
 *   - Weights stay quantized in the mmap; dequantization happens ONLY
 *     inside the dot kernels (vec_dot_q8_0_f32) or the explicit
 *     dequant_row_q8_0 -- callers never materialize a dequantized tensor.
 *   - Every function here is a pure function of its arguments: no globals,
 *     no allocation (callers own all scratch space; at M2 none is needed).
 *
 * Split across this directory (implementation detail, not part of the
 * contract): f16.c (f16_to_f32), q8_0.c (dequant_row_q8_0,
 * vec_dot_q8_0_f32), vec.c (vec_dot_f32, matvec), ops.c (rmsnorm,
 * rope_neox, softmax, silu, swiglu, argmax). Every symbol this directory
 * exposes is declared here -- nothing else is public.
 */
#ifndef STOKE_KERNELS_H
#define STOKE_KERNELS_H

#include "common.h"

#include "gguf.h"

/* Q8_0's type id comes from gguf.h's GGUF_GGML_TYPE_* enum (Q8_0 was
 * added there together with gguf.c's open support; both sides trace to
 * the same b10037 source probe + real-bytes annotation). */

/* Q8_0 block geometry (m2-probe-notes.md P3.1, verbatim from b10037's
 * ggml-common.h): 32 elements per block, 34 bytes per block (2-byte f16
 * scale + 32 signed-byte quants). Shared between q8_0.c's own kernels and
 * vec.c's matvec row-stride arithmetic (`row_bytes = (n_in /
 * KERNELS_Q8_0_BLOCK_ELEMS) * KERNELS_Q8_0_BLOCK_BYTES`) so the block size
 * exists as one named constant, not two independently-typed magic
 * numbers. */
#define KERNELS_Q8_0_BLOCK_ELEMS 32u
#define KERNELS_Q8_0_BLOCK_BYTES 34u

/* K-quant super-block geometry (K4, b10037 ggml-common.h; the byte counts
 * are corroborated by the real Q4_K_M file's own tensor nbytes).
 *
 *   block_q4_K = { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] }  = 144 B
 *   block_q6_K = { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d } = 210 B
 *
 * Both hold 256 elements, NOT 32 -- a K-quant "block" is a super-block of
 * eight 32-element sub-blocks sharing one f16 scale pair, which is why the
 * scale/min unpack exists at all. Shared with vec.c's matvec row-stride
 * arithmetic for the same reason the Q8_0 pair is. */
#define KERNELS_Q4_K_BLOCK_ELEMS 256u
#define KERNELS_Q4_K_BLOCK_BYTES 144u
#define KERNELS_Q6_K_BLOCK_ELEMS 256u
#define KERNELS_Q6_K_BLOCK_BYTES 210u

/* ---- f16 ----
 * IEEE 754 binary16 -> f32, exact: every f16 value (normal, subnormal,
 * signed zero, inf, or nan) is exactly representable in f32, so this is a
 * correctly-rounded widen, not an approximation. Subnormal f16 inputs are
 * renormalized into f32's wider exponent range.
 *
 * Does NOT fatal on an inf/nan bit pattern -- it is a bare bit-level
 * converter. The "a sane GGUF cannot contain an inf/nan block scale"
 * fatal is layered in at dequant_row_q8_0/vec_dot_q8_0_f32, the one place
 * this project ever feeds a stored f16 into this function; a bare
 * converter that itself refused to convert inf/nan would make it
 * impossible to unit-test that those bit patterns decode correctly in
 * the first place. */
f32 f16_to_f32(u16 h);

/* f32 -> f16, round-to-nearest-EVEN, with subnormal/overflow/NaN handling.
 *
 * The inverse direction genuinely rounds (23 mantissa bits -> 10), so unlike
 * f16_to_f32 this is lossy by construction. Ties break to even to match IEEE
 * 754's default and hardware F16C (`_mm256_cvtps_ph` with
 * _MM_FROUND_TO_NEAREST_INT), which K3 will substitute for this function --
 * a different tie rule here would make the scalar and SIMD paths disagree on
 * precisely the values that land on a tie.
 *
 * Overflow saturates to +-inf and underflow to a signed zero rather than
 * fataling: this is a bare bit-level converter, exactly as f16_to_f32 is,
 * and the "a sane GGUF cannot contain inf/nan" policy is layered in at the
 * kernels that read stored scales. A NaN never degrades into an inf -- a
 * payload that would truncate to zero is forced to 1. */
u16 f32_to_f16(f32 f);

/* ---- Q8_0 ----
 * Block layout (probed b10037 ggml-common.h, m2-probe-notes.md P3.1):
 * 34 bytes = { f16 d; i8 qs[32] }, value_i = f16_to_f32(d) * qs[i]. A row
 * of n elements (n a multiple of 32) is n/32 consecutive 34-byte blocks,
 * laid out back to back with no padding.
 *
 * Both kernels below fatal if a block's f16 scale `d` converts to
 * inf/nan: a sane GGUF cannot contain one (M2 op-spec §4), so seeing one
 * here means a corrupt or hostile file.
 */

/* Dequantizes a full Q8_0 row (n/32 blocks) into out[0..n). Fatal if n is
 * not a multiple of 32, or if any block's scale is inf/nan. */
void dequant_row_q8_0(const void* row, u64 n, f32* out);

/* ---- K-quants (K4, src/kernels/kquant.c) ----
 * Dequantize a full Q4_K / Q6_K row (n/256 super-blocks) into out[0..n).
 * Fatal if n is not a multiple of 256, or if a super-block scale (d, or
 * Q4_K's dmin) is inf/nan.
 *
 * THE CONTRACT IS BIT-EXACTNESS, not closeness. These reproduce, value for
 * value and bit for bit, what llama.cpp's own dequant produces for the same
 * bytes -- pinned by tests/data/k4_dequant.golden, which was generated FROM
 * llama.cpp rather than by hand. The reason the bar is exactness and not a
 * tolerance is measured: the classic scale/min unpack slip perturbs only
 * ~17%% of a row by at most ~0.0138, so a tolerance accepts the one bug this
 * gate exists to catch. Do not "simplify" the arithmetic grouping in
 * kquant.c; f32 multiply is not associative and the grouping is part of the
 * contract. */
void dequant_row_q4_k(const void* row, u64 n, f32* out);
void dequant_row_q6_k(const void* row, u64 n, f32* out);

/* Q8_K activation block -- the K-quant integer core's activation type.
 *
 *     { f32 d; i8 qs[256]; i16 bsums[16] } = 4 + 256 + 32 = 292 B
 *
 * NOT a widened Q8_0. Q4_K carries a per-sub-block MIN whose contribution is
 * `dmin * sum(activation)`, so the dot needs the activation's SUMS as well
 * as its quants; `bsums` is what carries them. Q6_K uses the same activation
 * type but never reads bsums (its quants are pre-centred by -32). */
#define KERNELS_Q8_K_BLOCK_ELEMS 256u
#define KERNELS_Q8_K_BLOCK_BYTES 292u

/* Quantizes n f32 activations into n/256 Q8_K blocks at `out`, which must
 * hold (n/256)*292 bytes. Rounding is ggml's magic-constant nearest_int
 * (round-to-nearest-EVEN), deliberately NOT roundf (half-away-from-zero) --
 * a different tie rule here is a different numeric class. */
void quantize_row_q8_k(const f32* x, u64 n, void* out);

/* Integer dots: a Q4_K/Q6_K weight row against a Q8_K-quantized activation,
 * without materializing either side as f32. The accumulation grouping (i32
 * per 8-lane bucket, scaled per sub-block, summed into 8 f32 partials, and
 * for Q4_K the min term subtracted per super-block) is ggml's and is the
 * binding numeric contract -- it is what puts stoke in llama.cpp's numeric
 * class on Q4_K_M, which is what the deferred oracle gate measures. */
f32 vec_dot_q4_k_q8_k(const void* wrow, const void* arow, u64 n);
f32 vec_dot_q6_k_q8_k(const void* wrow, const void* arow, u64 n);

/* Q4_K's 12 packed bytes -> 8 six-bit scales + 8 six-bit mins, TWO WAYS.
 *
 * ggml contains both: the dequant path pulls one (scale, min) pair at a time
 * with get_scale_min_k4, the dot path repacks all sixteen at once with a
 * uint32 mask/shift dance. They must agree, and both are transcribed here
 * because both are used. Exposed (rather than left static) so tests can
 * assert that agreement directly on real blocks -- two independent
 * transcriptions of the same packing agreeing is real evidence about both,
 * where reviewing either alone is evidence about neither. This is the same
 * reason bench_reps_for() was hoisted out of its caller.
 *
 * `packed12` points at the 12 scale bytes (offset 4 of a Q4_K block); j is
 * the sub-block index in [0,8). */
void kernels_q4_k_scale_min_pair(const void* packed12, int j, u8* scale, u8* min);
void kernels_q4_k_unpack_scales(const void* packed12, u8* scales8, u8* mins8);

/* Dot product of a Q8_0-quantized row against a dense f32 vector x,
 * WITHOUT materializing the dequantized row. Per block: `d * sum(qs[i] *
 * x[i])` -- the block's own sum accumulates first (f32, sequential over
 * that block's 32 elements), THEN gets scaled by d and added into the
 * running total (f32, sequential across blocks). This exact grouping --
 * scale-after-block-sum, not scale-then-accumulate-per-element -- is the
 * binding numeric contract (M2 op-spec §5), not an equivalent reordering.
 * Fatal if n is not a multiple of 32, or if any block's scale is
 * inf/nan. */
f32 vec_dot_q8_0_f32(const void* row, const f32* x, u64 n);

/* ---- M3 integer core ----
 *
 * LOCKED decision #2 of the M3-M7 forward note: when weights are quantized,
 * activations are quantized to Q8_0 blocks FIRST and both the scalar and
 * SIMD kernels run the same i32 integer core. Two things follow, and both
 * are the reason this exists rather than being a micro-optimization:
 *
 *   - Scalar<->SIMD becomes EXACTLY comparable. The K3 gate demands a
 *     bit-identical diff over 1e4 random tensors per kernel, which an f32
 *     accumulation can never promise across two different summation orders,
 *     and an integer accumulation gives for free.
 *   - stoke moves INTO llama.cpp's numeric class. M2's oracle survey found
 *     19 of 50 prompts diverging, 18 of 18 classifiable ones a clean top-2
 *     swap at 0.001-0.204 nat gaps -- the diagnosis being that the oracle
 *     quantizes activations to Q8 for integer dots while M2's stoke was pure
 *     f32. The deferred oracle parity gate re-runs on THIS path, and if the
 *     diagnosis is right that divergence count should collapse.
 */

/* Quantizes a dense f32 vector into n/32 consecutive Q8_0 blocks written to
 * `out` (caller-owned, n/32 * KERNELS_Q8_0_BLOCK_BYTES bytes). Transcribed
 * from the probed ggml quantize_row_q8_0_ref: per block, d = absmax/127
 * stored as f16, quants = roundf(x * (1/d)) with 1/d taken from the f32 d
 * BEFORE narrowing, and an all-zero block yielding all-zero quants rather
 * than NaN. Fatal if n is not a multiple of 32. */
void quantize_row_q8_0(const f32* x, u64 n, void* out);

/* Dot product of a Q8_0 weight row against a Q8_0-quantized activation row.
 * Per block the i8 x i8 products accumulate in i32 with no rounding at all
 * (32 * 127 * 127 = 516,128, far inside i32), then the block contributes
 * `(f32)sumi * (d_w * d_a)` -- the two scales multiplying each other FIRST,
 * matching ggml's scalar reference grouping exactly. Fatal if n is not a
 * multiple of 32, or if either side's block scale is inf/nan. */
f32 vec_dot_q8_0_q8_0(const void* wrow, const void* arow, u64 n);

/* ---- dense f32 ---- */

/* a . b = sum(a[i]*b[i]), f32 accumulate, sequential, fixed left-to-right
 * order (§4). */
f32 vec_dot_f32(const f32* a, const f32* b, u64 n);

/* y = W @ x: one dot product per output row of the weight tensor `w`.
 * `w` supplies both the weight bytes and the shape -- dims[0] =
 * contraction length, dims[1] = row count (the GGUF dims convention, M2
 * op-spec §3: dims[0] is the fastest-varying / row-length-in-memory
 * axis). matvec cross-checks the caller-supplied n_in/n_out against
 * w->dims[0]/w->dims[1] and fatals (naming the tensor) on any mismatch,
 * rather than trusting a possibly-wrong shape to drive row-stride
 * arithmetic over a read-only mmap.
 *
 * w->type is the ONLY dispatch point on quant type at M2 (§5): F32 rows
 * go through vec_dot_f32, Q8_0 rows through vec_dot_q8_0_f32; any other
 * type is a fatal naming the tensor (both ids from gguf.h's probed
 * GGUF_GGML_TYPE_* enum).
 *
 * `aq_scratch` (M3) is a caller-owned buffer of
 * (n_in/KERNELS_Q8_0_BLOCK_ELEMS)*KERNELS_Q8_0_BLOCK_BYTES bytes, used to
 * hold the Q8_0-quantized activation. It is REQUIRED for a Q8_0 weight and
 * ignored (may be NULL) for an F32 one. A Q8_0 weight with a NULL scratch is
 * FATAL rather than a fallback to the f32-activation path: those are two
 * different arithmetic classes, and silently reverting to M2 numerics on
 * some call sites would be undetectable in the output and would invalidate
 * the very parity measurement M3 exists to make.
 *
 * The activation is quantized ONCE per call, not once per row -- x is
 * constant across rows, and re-quantizing per row would repeat the work
 * 151,936 times on the logits matvec alone. */
void matvec(const GgufTensor* w, const f32* x, u64 n_in, f32* y, u64 n_out,
            void* aq_scratch);

/* Applies `w` to `n_seq` activations at once -- seqs-inner/weights-outer, so
 * the weight row is read once and applied to every sequence. Design:
 * docs/superpowers/design-notes/2026-08-11-m4-static-batch.md section 5.5.
 *
 * THE CONTRACT IS BYTE-IDENTITY: for every s, y[s] must equal what
 * matvec(w, x[s], n_in, y[s], n_out, aq_scratch[s]) would have written,
 * bit for bit. ROADMAP section M4 requires batch-composition independence
 * EXACT, and this is where it is earned -- by construction, since the
 * accumulation order inside each dot product is untouched and nothing sums
 * across sequences. tests/test_matvec_batch.c checks it rather than
 * assuming it.
 *
 * `x` and `y` are arrays of n_seq pointers; `aq_scratch` likewise (one
 * quantized-activation buffer per sequence, because x is constant across
 * rows but NOT across sequences). aq_scratch may be NULL only when `w` is
 * F32, which needs none. */
void matvec_batch(const GgufTensor* w, const f32* const* x, u64 n_in,
                  f32* const* y, u64 n_out, void* const* aq_scratch,
                  u32 n_seq);

/* BENCH/TEST HOOK (2026-09-16): the partial-group padding threshold
 * (vec.c, STOKE_KQ_PAD_TAIL) set programmatically -- 0 never pads, k pads a
 * remainder of at least k sequences up to a whole group on both K-quant
 * kernels, KERNELS_PAD_TAIL_AUTO restores the measured per-kernel policy.
 * Overrides the env for the rest of the process. Dispatching thread only. */
#define KERNELS_PAD_TAIL_AUTO 0xFFFFFFFEu
void kernels_set_pad_tail(u32 min_remainder);

/* ---- normalization / positional / activation ---- */

/* y[i] = (x[i] / sqrt(mean(x^2) + eps)) * w[i], f32 mean, for i in
 * [0, n) (§5). x, w, y are each length n; y may safely alias x (every
 * read of x[i] happens before the one write to y[i] at that same index,
 * across both passes). */
void rmsnorm(const f32* x, const f32* w, u64 n, f32 eps, f32* y);

/* In-place NEOX rotary embedding over one head's vector (length
 * head_dim), applied AFTER QK-norm (M2 op-spec §6). NEOX pairs
 * (k, k+head_dim/2) for k in [0, head_dim/2) -- NOT adjacent (2k, 2k+1),
 * which is the GPT-J/original-RoPE style and produces plausible-looking
 * garbage here (TRAP #3, §5). Angle theta_k = pos *
 * freq_base^(-2k/head_dim), computed ITERATIVELY to match the oracle's
 * own accumulation exactly (m2-probe-notes.md P3.3: ggml_rope_cache_init
 * starts theta at the position and multiplies by a fixed theta_scale
 * once per step) rather than a per-k powf call:
 *   theta_scale = powf(freq_base, -2.0f/head_dim), theta starts (f32)pos
 *   for k in [0, head_dim/2): use theta as theta_k, THEN theta *=
 *   theta_scale for the next k
 *   v[k]'          = v[k]*cos(theta_k) - v[k+half]*sin(theta_k)
 *   v[k+half]'     = v[k]*sin(theta_k) + v[k+half]*cos(theta_k)
 * At pos 0, theta is identically 0 for every k (0 * theta_scale stays
 * 0), so cos=1/sin=0 throughout and the transform is the exact identity.
 *
 * freq_factors: NULL for models with no RoPE-scaling correction (qwen3),
 * or a head_dim/2-length f32 vector (llama3's `rope_freqs.weight`), in
 * which case pair k rotates by theta_k / freq_factors[k].
 * ⚠ THE ACCUMULATOR IS NOT DIVIDED -- the division applies per USE, so the
 * theta chain is bit-identical with and without factors. That is ggml's
 * own shape (`rope_yarn(theta/ff, ...)` then `theta *= theta_scale`,
 * ggml-cpu/ops.cpp:5846-5860, read rather than assumed); folding the
 * division into the accumulation would compound it across k and is wrong.
 * ⚠ AND IT IS NOT A LONG-CONTEXT-ONLY CORRECTION, which the B5 entry note
 * first assumed: Llama-3.2-3B-Instruct's vector runs 1.0 -> 32.0 with only
 * 29 of its 64 entries equal to 1.0 (read out of the file), so it changes
 * the rotation at EVERY position including pos 1 -- the magnitude just
 * grows with pos. The NULL branch is hoisted out of the loop so a
 * factor-free model executes the identical instruction stream it did
 * before this parameter existed. */
void rope_neox(f32* vec, u64 head_dim, u64 pos, f32 freq_base,
               const f32* freq_factors);

/* In-place NORM ("normal", GPT-J-style) rotary embedding -- the SAME angles
 * as rope_neox, over a DIFFERENT pairing: pair k is the ADJACENT elements
 * (2k, 2k+1) instead of (k, k+half).
 *
 * ⚠ WHICH ONE A MODEL WANTS IS A PROPERTY OF ITS ARCHITECTURE, NOT A STYLE
 * CHOICE, and picking the wrong one produces plausible-looking garbage
 * rather than an error (rope_neox's TRAP #3, from the other side).
 * llama.cpp's `llama_model_rope_type()` returns ROPE_TYPE_NORM for
 * LLM_ARCH_LLAMA and ROPE_TYPE_NEOX for LLM_ARCH_QWEN3
 * (llama-model.cpp:1329/1367 vs :1386/1439) -- read, not assumed. The
 * conversion script permutes Q/K on the way into a llama GGUF so that this
 * adjacent pairing is the correct one for the stored weights.
 *
 * theta_k, the accumulation, freq_factors and the pos-0 identity are all
 * exactly as documented for rope_neox above: ggml builds ONE cache indexed
 * by k = i0/2 and both rope types read it, differing only in which two
 * elements they rotate (`rotate_pairs(n_dims, 1, ..., scale=1)` for NORM
 * against `rotate_pairs(n_dims, n_dims/2, ...)` for NEOX,
 * ggml-cpu/ops.cpp:5934-5950, 6069-6077). */
void rope_norm(f32* vec, u64 head_dim, u64 pos, f32 freq_base,
               const f32* freq_factors);

/* In-place softmax over x[0..n): max-subtract (numerically stable for
 * large-magnitude inputs that would otherwise overflow exp), exp,
 * normalize (§5). Safe for n == 0 (no element of x is read or written,
 * so there is nothing to normalize and no division by an empty sum). */
void softmax(f32* x, u64 n);

/* silu(z) = z / (1 + e^(-z)) (§5) -- the scalar building block. */
f32 silu(f32 z);

/* SwiGLU combine (§5, used in §6's FFN: `silu(gate) * up` immediately
 * before the down projection): h[i] = silu(g[i]) * u[i], elementwise,
 * length n. g, u, h may alias in any combination (each h[i] is fully
 * computed from that index's g[i]/u[i] before anything is written). */
void swiglu(const f32* g, const f32* u, u64 n, f32* h);

/* Index of the maximum of logits[0..n). First-lowest-index wins ties: the
 * running best only moves on a STRICTLY greater value, so the first
 * occurrence of the max is what gets returned (§5). n == 0 returns 0
 * without dereferencing logits -- there is no element of a genuinely
 * empty array to compare. */
u32 argmax(const f32* logits, u64 n);

#endif /* STOKE_KERNELS_H */
