#!/usr/bin/env python3
"""tests/tooling/ref_forward.py -- float32 numpy reference of the M2 Qwen3
forward pass.

M2 W1 deliverable #2. Implements exactly
docs/superpowers/design-notes/2026-07-19-m2-op-spec-and-parity-protocol.md
section 6, using .superpowers/sdd/m2-probe-notes.md P3.3 (NEOX rope, source
-confirmed against llama.cpp b10037) and P3.4 (graph order: QK-norm BEFORE
rope; attention scale 1/sqrt(head_dim); tied LM head) verbatim. Tensors are
read via gguf_reader.py -- the same independent parser dump_tensor.py uses.

Per the trust order (design note section 9): llama-server (oracle) > this
ref > the C code under test. This tool's OWN correctness is established by
reproducing the oracle's greedy token ids on real prompts (m2-w1-report.md
records that run) before it is trusted for anything else (layer dumps for
L2/L3 bisection).

dtype=float32 EVERYWHERE, matmuls included. numpy 2.4.0 (the version this
was built and run against) verified empirically to preserve float32 through
every op used below (mean/sum with explicit dtype=, sqrt/exp/cos/sin, `@`/
np.dot, and NEP-50 scalar promotion when a bare python int/float combines
with a float32 array) -- see m2-w1-report.md's dtype-discipline section for
the exact probe commands and output. f32() below is a VALIDATOR, not a
coercer: it asserts its argument is already float32 and returns it
unchanged. It deliberately does NOT cast, because casting would silently
paper over exactly the class of bug this whole tool exists to catch --
`python ref_forward.py --selftest` includes a negative test proving a
float64 leak actually trips it (not a dead check).

CLI:
  python ref_forward.py <gguf> --ids 1,2,3 -n 64 [--dump-layers DIR]
      Greedy-decodes n token ids (fewer if EOS is generated first -- EOS,
      when generated, IS printed, matching the M2 note section 7 CLI
      contract stoke run will implement), one per line to stdout.
      --dump-layers DIR: also writes L00.f32..L35.f32 (raw float32, one
      file per layer) -- the post-block hidden state after each of the 36
      layers, for the LAST input (prompt) position only.
  python ref_forward.py --selftest
      Pure-function self-test (no GGUF needed): dtype discipline (positive
      + the negative/leak-catching test above) and the argmax tie rule.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from gguf_reader import GGUFFile, GgufError, fatal


# ============================================================================
# float32 discipline
# ============================================================================

def f32(x):
    """Assert `x` is ALREADY a float32 ndarray (or 0-d/scalar of that
    dtype) and return it unchanged -- see module docstring for why this
    validates rather than casts."""
    arr = x if isinstance(x, np.ndarray) else np.asarray(x)
    assert arr.dtype == np.float32, f"dtype leak: expected float32, got {arr.dtype!r}"
    return arr


# ============================================================================
# Kernels (M2 note section 5), pure functions, float32 in -> float32 out
# ============================================================================

def rmsnorm(x, w, eps):
    """y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i], f32 mean (explicit
    dtype= on the reduction -- belt and suspenders; numpy 2.4.0 already
    keeps a float32 array's mean in float32 by default, verified, but the
    M2 note calls out this exact reduction as the one to guard)."""
    x = f32(x)
    ms = np.mean(x * x, dtype=np.float32)
    denom = f32(np.sqrt(f32(ms + np.float32(eps))))
    return f32((x / denom) * w)


def rope_thetas(pos, freq_base, head_dim, freq_factors=None, theta_mode="closed"):
    """The half-length theta vector both pairings share. Split out of
    rope_neox when M7 added the second pairing, because the ANGLES are
    identical between NORM and NEOX and only the pairing differs
    (llama.cpp ggml-cpu/ops.cpp: one ggml_rope_cache_init feeds both
    rotate_pairs calls) -- so a bug in the angles must not be able to hide
    in one pairing and not the other.

    theta_mode:
      "closed" (DEFAULT, and what M2 froze) -- theta_k = pos *
          freq_base^(-2k/head_dim), the closed form derived in
          m2-probe-notes.md P3.3. Mathematically equal to llama.cpp's
          incremental loop, and DELIBERATELY not the same float sequence.
      "accum" -- theta_0 = pos, theta_{k+1} = theta_k * theta_scale with
          theta_scale = freq_base^(-2/head_dim), i.e. llama.cpp's and
          stoke's own accumulation. Provided as a DIAGNOSTIC, never the
          default: ROADMAP.md and the M3 ruling §8.9 both name this
          accumulation difference as the first suspect if the per-band
          tolerance grows at long positions, and a suspect you cannot
          switch off is not testable. Defaulting to it would instead HIDE
          the effect the band structure exists to detect.

    freq_factors (llama3 rope scaling, `rope_freqs.weight`): ggml divides
    AT USE and leaves the accumulator untouched --
    `rope_yarn(theta/ff[i0/2], ...)` then `theta *= theta_scale` -- so the
    division must not be folded into the recurrence, where it would
    compound across k. None (Qwen3) leaves the arithmetic bit-identical to
    the M2 form.
    """
    half = head_dim // 2
    if theta_mode == "closed":
        k = np.arange(half, dtype=np.float32)
        exponent = f32(np.float32(-2.0) * k / np.float32(head_dim))
        theta = f32(np.float32(pos) * np.power(np.float32(freq_base), exponent))
    elif theta_mode == "accum":
        theta_scale = np.float32(np.power(np.float32(freq_base),
                                          np.float32(-2.0) / np.float32(head_dim)))
        theta = np.empty(half, dtype=np.float32)
        t = np.float32(pos)
        for k in range(half):
            theta[k] = t
            t = np.float32(t * theta_scale)
        theta = f32(theta)
    else:
        fatal(f"rope theta_mode={theta_mode!r}, expected 'closed' or 'accum'")
    if freq_factors is not None:
        theta = f32(theta / f32(freq_factors))
    return theta


def rope_thetas_matrix(n_pos, freq_base, head_dim, freq_factors=None,
                       theta_mode="closed"):
    """rope_thetas for positions 0..n_pos-1 at once, [n_pos, head_dim/2].

    Every row MUST equal rope_thetas(pos, ...) bit for bit -- checked in
    --selftest, because the batched forward's whole justification is that it is
    the same computation reorganized, and an angle table that quietly drifted
    from the scalar one would be a second implementation wearing the first
    one's name."""
    half = head_dim // 2
    if theta_mode == "closed":
        pos_col = np.arange(n_pos, dtype=np.float32).reshape(n_pos, 1)
        k = np.arange(half, dtype=np.float32)
        exponent = f32(np.float32(-2.0) * k / np.float32(head_dim))
        th = f32(pos_col * np.power(np.float32(freq_base), exponent))
    else:
        # The accumulation is inherently serial in k; there is nothing to
        # vectorize across positions that would stay bit-identical, so this
        # branch calls the scalar builder per row on purpose.
        th = f32(np.stack([rope_thetas(p, freq_base, head_dim, None, theta_mode)
                           for p in range(n_pos)]))
    if freq_factors is not None:
        th = f32(th / f32(freq_factors))
    return th


def rope_neox(vec, pos, freq_base, head_dim, freq_factors=None,
              theta_mode="closed"):
    """NEOX pairing (k, k+half), k in [0,half). llama.cpp
    `rotate_pairs(n_dims, n_dims/2, ..., scale=2)`: element index ic =
    i0/2 = k, partner at ic + n_dims/2.

    P3.3 states the closed form as the contract stoke's C rope_neox must
    implement, so the ref matches that, not llama.cpp's own accumulation
    path -- token-level parity is the gate, bit-exact reduction order is
    explicitly not required (M2 note section 4)."""
    vec = f32(vec)
    half = head_dim // 2
    theta = rope_thetas(pos, freq_base, head_dim, freq_factors, theta_mode)
    cos = f32(np.cos(theta))
    sin = f32(np.sin(theta))
    v0, v1 = vec[:half], vec[half:]
    out = np.empty_like(vec)
    out[:half] = v0 * cos - v1 * sin
    out[half:] = v0 * sin + v1 * cos
    return f32(out)


def rope_norm(vec, pos, freq_base, head_dim, freq_factors=None,
              theta_mode="closed"):
    """NORM pairing (2k, 2k+1) -- ADJACENT, k in [0,half). llama.cpp
    `rotate_pairs(n_dims, 1, ..., scale=1)`: the `scale=1` hack makes
    ic = i0 (0,2,4,...) while the cache is still indexed by i0, so pair k
    uses cache[2k]/cache[2k+1] = the cos/sin of theta_k. Same angles as
    NEOX, different pairing.

    ⚠ THIS IS `kernels.h`'s TRAP #3 ARRIVING FROM THE OTHER SIDE. The two
    pairings are not distinguishable by "does it run" or by "is the output
    fluent" -- the wrong one produces plausible-looking garbage. The
    grader for that is the permutation equivalence in --selftest, not a
    reading of the text.

    llama.cpp picks the pairing per ARCHITECTURE (llama_model_rope_type():
    LLM_ARCH_LLAMA -> NORM, LLM_ARCH_QWEN3 -> NEOX), which is why this
    file's dispatch is keyed on general.architecture and not on anything
    in the tensor data."""
    vec = f32(vec)
    half = head_dim // 2
    theta = rope_thetas(pos, freq_base, head_dim, freq_factors, theta_mode)
    cos = f32(np.cos(theta))
    sin = f32(np.sin(theta))
    v0, v1 = vec[0::2], vec[1::2]
    out = np.empty_like(vec)
    out[0::2] = v0 * cos - v1 * sin
    out[1::2] = v0 * sin + v1 * cos
    return f32(out)


def softmax(x):
    x = f32(x)
    e = f32(np.exp(x - np.max(x)))
    return f32(e / np.sum(e, dtype=np.float32))


# --- row-wise twins of the three elementwise kernels, for the batched path ---
# Each is the SAME expression with an axis: elementwise ops and per-row
# reductions, so a row of the batched form is the serial form applied to that
# row, bit for bit. That is asserted by --selftest, not assumed -- the whole
# point of a batched instrument is that it must not become a second
# implementation.

def rmsnorm_rows(X, w, eps):
    X = f32(X)
    ms = np.mean(X * X, axis=-1, dtype=np.float32, keepdims=True)
    denom = f32(np.sqrt(f32(ms + np.float32(eps))))
    return f32((X / denom) * w)


def softmax_rows(S):
    S = f32(S)
    e = f32(np.exp(S - np.max(S, axis=-1, keepdims=True)))
    return f32(e / np.sum(e, axis=-1, dtype=np.float32, keepdims=True))


def top5_from_logits(logits):
    """Top-5 (id, logit), rank-0 highest first, ties broken by the LOWER id
    -- byte-for-byte the rule in src/model.c's record_top5(), whose
    insertion test is `lv > out[i].logit || (lv == out[i].logit && v <
    out[i].id)`. np.lexsort's LAST key is primary, so this sorts by
    -logit ascending (= logit descending) with the index ascending as the
    tiebreak, which is exactly that rule. Read-only exposure of logits the
    generate loop already computes -- it adds no arithmetic and cannot
    change a generated id."""
    order = np.lexsort((np.arange(logits.size), -logits))[:5]
    return [(int(i), float(logits[i])) for i in order]


def silu(z):
    z = f32(z)
    return f32(z / (np.float32(1.0) + np.exp(-z)))


# ============================================================================
# Model: config load (every value read from the GGUF header, section 3 key
# names, never hardcoded) + per-layer tensor lookup + forward pass
# ============================================================================

# llama.cpp's llama_model_rope_type() (src/llama-model.cpp), transcribed for
# the two architectures this file implements and NO others. It is a per-ARCH
# table upstream, not a property of the file's tensors, so it cannot be
# derived from the GGUF -- it has to be transcribed, and a wrong row here is
# silent (see rope_norm's docstring).
ARCH_ROPE_KIND = {
    "qwen3": "neox",
    "llama": "norm",
}


class ModelRef:
    """float32 numpy reference forward pass.

    ⚠ ACCEPTS ONLY ARCHITECTURES THAT ARE TESTED, by name. llama.cpp maps
    many arch strings onto the same graph; accepting one because it "should
    work" would be a claim, not a test."""

    def __init__(self, gguf_path: str, theta_mode: str = "closed"):
        self.gf = GGUFFile(gguf_path)
        kv = self.gf.kv
        self.theta_mode = theta_mode
        if theta_mode not in ("closed", "accum"):
            fatal(f"theta_mode={theta_mode!r}, expected 'closed' or 'accum'")

        arch = self.gf.kv.get("general.architecture")
        if arch not in ARCH_ROPE_KIND:
            fatal(f"general.architecture={arch!r}, expected one of "
                  f"{sorted(ARCH_ROPE_KIND)} (this ref implements only what is tested)")
        self.arch = arch
        self.rope_kind = ARCH_ROPE_KIND[arch]
        self.rope = rope_norm if self.rope_kind == "norm" else rope_neox

        def req(key):
            """Architecture-prefixed lookup. Every suffix below is shared by
            both architectures -- verified by dumping the metadata of both
            files, not assumed from the naming convention."""
            full = f"{arch}.{key}"
            if full not in kv:
                fatal(f"missing required config key {full!r}")
            return kv[full]

        self.n_layer = int(req("block_count"))
        self.n_embd = int(req("embedding_length"))
        self.n_ff = int(req("feed_forward_length"))
        self.n_head = int(req("attention.head_count"))
        self.n_head_kv = int(req("attention.head_count_kv"))
        self.head_dim = int(req("attention.key_length"))       # TRAP #1: own key, never n_embd//n_head
        v_len = int(req("attention.value_length"))
        if v_len != self.head_dim:
            fatal(f"value_length {v_len} != key_length {self.head_dim} (M2 requires equal)")
        self.freq_base = float(req("rope.freq_base"))
        self.eps = float(req("attention.layer_norm_rms_epsilon"))
        self.max_ctx = min(int(req("context_length")), 4096)    # M2 note section 2 cap
        if self.n_head % self.n_head_kv != 0:
            fatal(f"head_count {self.n_head} not divisible by head_count_kv {self.n_head_kv}")
        self.gqa_group = self.n_head // self.n_head_kv

        # PARTIAL rope (rope.dimension_count < head_dim) makes llama.cpp run a
        # THIRD rotate_pairs over the tail (ops.cpp:6079). Neither model on this
        # box uses it, so it is REFUSED rather than implemented untested.
        n_dims = kv.get(f"{arch}.rope.dimension_count")
        if n_dims is not None and int(n_dims) != self.head_dim:
            fatal(f"{arch}.rope.dimension_count={int(n_dims)} != key_length "
                  f"{self.head_dim}: partial rope is not implemented here")

        eos = kv.get("tokenizer.ggml.eos_token_id")
        self.eos_id = int(eos) if eos is not None else None

        self.tok_embd_t = self.gf.get_tensor("token_embd.weight")
        self.output_norm_t = self.gf.get_tensor("output_norm.weight")
        self.output_t = self.gf.tensors.get("output.weight")   # TRAP #2: tied if absent
        self.vocab_size = self.tok_embd_t.dims[1]

        # llama3 rope scaling. OPTIONAL, and length-checked because the
        # rotation indexes it by PAIR with no bound of its own: a short
        # factors vector would read past its end and a long one would be
        # silently half-used.
        self.freq_factors = None
        rf = self.gf.tensors.get("rope_freqs.weight")
        if rf is not None:
            self.freq_factors = f32(self.gf.tensor_dequant_full(rf).reshape(-1))
            if self.freq_factors.size != self.head_dim // 2:
                fatal(f"rope_freqs.weight has {self.freq_factors.size} entries, "
                      f"expected head_dim/2 = {self.head_dim // 2}")

        self.layers = []
        base = ("attn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
                "ffn_norm", "ffn_gate", "ffn_up", "ffn_down")
        qk_present = []
        for L in range(self.n_layer):
            layer = {k: self.gf.get_tensor(f"blk.{L}.{k}.weight") for k in base}
            for k in ("attn_q_norm", "attn_k_norm"):
                layer[k] = self.gf.tensors.get(f"blk.{L}.{k}.weight")
            qk_present.append(layer["attn_q_norm"] is not None
                              and layer["attn_k_norm"] is not None)
            if (layer["attn_q_norm"] is None) != (layer["attn_k_norm"] is None):
                fatal(f"layer {L} has exactly one of attn_q_norm/attn_k_norm; "
                      f"QK-norm is all-or-nothing")
            self._validate_layer_shapes(L, layer)
            self.layers.append(layer)
        # ALL-OR-NOTHING across layers as well as within one: a file with
        # QK-norm on some blocks would otherwise be silently half-normed,
        # which is the exact shape of a fluent wrong answer.
        if any(qk_present) and not all(qk_present):
            missing = [L for L, p in enumerate(qk_present) if not p]
            fatal(f"QK-norm present on some layers but not {missing}; "
                  f"it is all-or-nothing")
        self.qk_norm = bool(qk_present and qk_present[0])

        # Full dequant of token_embd is cached (bounded: one ~1.5-2GB f32
        # array) because it's reused EVERY position for both embedding
        # lookup and the tied lm_head. Every OTHER weight tensor is
        # deliberately re-dequantized on each matvec call with no cache:
        # caching all 36 layers' weights simultaneously would need
        # roughly (quantized bytes minus token_embd) * ~3.7 =~ 14GB+,
        # unsafe on a 16GB box. Measured cost of the uncached approach
        # (m2-w1-report.md): ~10-11s of per-layer dequant per position,
        # comfortably inside the "minutes per prompt is fine" / "30 min"
        # budget the design note gives this tool (section 9, section 10).
        self._embd_cache = None

    def _validate_layer_shapes(self, L, layer):
        E, F, H, HK, D = self.n_embd, self.n_ff, self.n_head, self.n_head_kv, self.head_dim
        expect = {
            "attn_norm": (E,), "ffn_norm": (E,),
            "attn_q_norm": (D,), "attn_k_norm": (D,),
            "attn_q": (E, H * D), "attn_k": (E, HK * D), "attn_v": (E, HK * D),
            "attn_output": (H * D, E),
            "ffn_gate": (E, F), "ffn_up": (E, F), "ffn_down": (F, E),
        }
        for key, dims in expect.items():
            t = layer[key]
            if t is None:          # optional (QK-norm); presence is checked by the caller
                continue
            if tuple(t.dims) != dims:
                fatal(f"layer {L} tensor {t.name} dims {t.dims} != expected {dims}")

    def embd_matrix(self):
        if self._embd_cache is None:
            self._embd_cache = self.gf.tensor_dequant_full(self.tok_embd_t)
        return self._embd_cache

    def embed(self, token_id: int):
        if not (0 <= token_id < self.vocab_size):
            fatal(f"token id {token_id} out of vocab range [0,{self.vocab_size})")
        return f32(self.embd_matrix()[token_id].copy())

    def matvec(self, t, x):
        W = self.gf.tensor_dequant_full(t)     # [n_out, n_in], fresh dequant every call
        x = f32(x)
        if W.shape[1] != x.shape[0]:
            fatal(f"matvec: {t.name} shape {W.shape} incompatible with input len {x.shape[0]}")
        return f32(W @ x)

    def logits(self, h):
        W = self.gf.tensor_dequant_full(self.output_t) if self.output_t is not None else self.embd_matrix()
        return f32(W @ f32(h))

    def forward_position(self, t_pos, x, kv_k, kv_v, dump_dir=None):
        """Runs all n_layer blocks for one position, updating kv_k/kv_v in
        place at index t_pos. Returns the post-final-layer hidden state
        (BEFORE output_norm). Implementation order per layer is P3.4,
        exactly: rmsnorm(attn_norm) -> Q/K/V matvec -> per-head
        rmsnorm(qk_norm) THEN rope_neox -> causal GQA attention -> attn_
        output matvec -> residual -> rmsnorm(ffn_norm) -> SwiGLU -> down ->
        residual."""
        x = f32(x)
        D = self.head_dim
        scale = np.float32(1.0 / np.sqrt(np.float32(D)))
        for L, layer in enumerate(self.layers):
            a = rmsnorm(x, self.gf.tensor_dequant_full(layer["attn_norm"]), self.eps)
            q = self.matvec(layer["attn_q"], a).reshape(self.n_head, D).copy()
            k = self.matvec(layer["attn_k"], a).reshape(self.n_head_kv, D).copy()
            v = self.matvec(layer["attn_v"], a).reshape(self.n_head_kv, D)

            # QK-norm is gated on TENSOR PRESENCE, never on the arch name:
            # graph order is rmsnorm(qk_norm) BEFORE rope (P3.4) when it is
            # there, and nothing at all when it is not. Llama-3.2 has neither
            # tensor; Qwen3 has both, and its branch is the one that executed
            # for all of M2/M3.
            rope, fb, D_, ff, tm = (self.rope, self.freq_base, D,
                                    self.freq_factors, self.theta_mode)
            if self.qk_norm:
                qn = self.gf.tensor_dequant_full(layer["attn_q_norm"])
                kn = self.gf.tensor_dequant_full(layer["attn_k_norm"])
                for h in range(self.n_head):
                    q[h] = rope(rmsnorm(q[h], qn, self.eps), t_pos, fb, D_, ff, tm)
                for h in range(self.n_head_kv):
                    k[h] = rope(rmsnorm(k[h], kn, self.eps), t_pos, fb, D_, ff, tm)
            else:
                for h in range(self.n_head):
                    q[h] = rope(q[h], t_pos, fb, D_, ff, tm)
                for h in range(self.n_head_kv):
                    k[h] = rope(k[h], t_pos, fb, D_, ff, tm)

            kv_k[L, :, t_pos, :] = k
            kv_v[L, :, t_pos, :] = v

            attn_out = np.empty((self.n_head, D), dtype=np.float32)
            for h in range(self.n_head):
                kvh = h // self.gqa_group                   # GQA: q-head h reads kv-head h//group
                Kc = kv_k[L, kvh, :t_pos + 1, :]             # [t+1, D]
                Vc = kv_v[L, kvh, :t_pos + 1, :]
                scores = f32((Kc @ q[h]) * scale)            # dot(q,K)/sqrt(head_dim)
                weights = softmax(scores)
                attn_out[h] = f32(weights @ Vc)
            attn_out_flat = f32(attn_out.reshape(self.n_head * D))
            x = f32(x + self.matvec(layer["attn_output"], attn_out_flat))

            f_ = rmsnorm(x, self.gf.tensor_dequant_full(layer["ffn_norm"]), self.eps)
            gate = self.matvec(layer["ffn_gate"], f_)
            up = self.matvec(layer["ffn_up"], f_)
            ffn_h = f32(silu(gate) * up)
            x = f32(x + self.matvec(layer["ffn_down"], ffn_h))

            if dump_dir is not None:
                x.tofile(os.path.join(dump_dir, f"L{L:02d}.f32"))
        return x

    def generate(self, ids, n, dump_dir=None, probe_pos=None, probe_all=None):
        """probe_all: an optional list. When given, the top-5 at EVERY generated
        position is APPENDED to it.

        WHY THIS IS ~FREE, and why it changes what a probe costs. The logits are
        ALREADY computed at every generated position -- the loop below needs
        them to pick next_id -- so `--probe-pos k` computes 64 distributions and
        reports one. Recording all of them adds one argpartition per position
        against ~18 s of forward pass, i.e. ~0.1%. A single run over prompt+64
        therefore yields 64 probes rather than 1, at the cost of the ONE prefix
        prefill that dominates everything.

        IT CANNOT MOVE THE TOKEN STREAM: probe_all is append-only, is read by
        nobody in this class, and next_id is computed from lg exactly as before.
        Comparability caveat for the CONSUMER, not for this method: these probes
        lie along THIS run's own stream, so they are comparable to another
        implementation's probes only up to the first position where the two
        streams diverge -- which is precisely the position the section 6 rule
        grades."""
        total_cap = len(ids) + n
        if total_cap > self.max_ctx:
            fatal(f"prompt+generation length {total_cap} exceeds max_ctx {self.max_ctx}")
        if dump_dir is not None:
            os.makedirs(dump_dir, exist_ok=True)

        kv_k = np.zeros((self.n_layer, self.n_head_kv, total_cap, self.head_dim), dtype=np.float32)
        kv_v = np.zeros((self.n_layer, self.n_head_kv, total_cap, self.head_dim), dtype=np.float32)

        x = None
        for t, tok in enumerate(ids):
            last_prefill = (t == len(ids) - 1)
            x = self.forward_position(t, self.embed(tok), kv_k, kv_v,
                                       dump_dir=(dump_dir if last_prefill else None))

        probe_out = None
        pos = len(ids) - 1
        h = rmsnorm(x, self.gf.tensor_dequant_full(self.output_norm_t), self.eps)
        lg = self.logits(h)
        if probe_pos == 0:
            probe_out = top5_from_logits(lg)
        if probe_all is not None:
            probe_all.append({"index": 0, "top5": top5_from_logits(lg)})
        next_id = int(np.argmax(lg))
        out_ids = [next_id]
        pos += 1
        while len(out_ids) < n:
            if self.eos_id is not None and next_id == self.eos_id:
                break
            x = self.forward_position(pos, self.embed(next_id), kv_k, kv_v)
            h = rmsnorm(x, self.gf.tensor_dequant_full(self.output_norm_t), self.eps)
            lg = self.logits(h)
            # len(out_ids) BEFORE the append is the index this new token will
            # occupy -- the same quantity src/model.c calls `count`, so
            # probe_pos means the identical generated position on both sides.
            if probe_pos is not None and probe_pos == len(out_ids):
                probe_out = top5_from_logits(lg)
            if probe_all is not None:
                probe_all.append({"index": len(out_ids), "top5": top5_from_logits(lg)})
            next_id = int(np.argmax(lg))
            out_ids.append(next_id)
            pos += 1
        if probe_pos is not None and probe_out is None:
            fatal(f"--probe-pos {probe_pos} was never reached: generation "
                  f"produced only {len(out_ids)} ids (EOS stops early). "
                  f"Refusing to report a probe for a position that does "
                  f"not exist.")
        return out_ids, probe_out

    def teacher_force(self, ids, probe_positions, progress=None, bank=None):
        """TEACHER-FORCED forward over EXACTLY `ids`, sampling nothing.

        This is the M7 entry-probe instrument, and teacher forcing is the
        property that makes it one. `generate()` follows the ref's OWN greedy
        stream, so its probes are comparable to another implementation only up
        to the first position where the two streams diverge -- which is fine
        for a first-divergence classifier and useless for measuring a Δ-profile
        at position 1400 on a stream that already parted at position 700.
        Forcing the ids makes EVERY probed position a comparison of the two
        implementations on the SAME prefix.

        `probe_positions`: absolute positions p at which to record the top-5 of
        the distribution computed AFTER consuming ids[p] -- i.e. the
        distribution that predicts ids[p+1]. That is the same object stoke's
        `--probe-pos k` reports for generated index k, at absolute position
        (n_prompt - 1 + k); the mapping is the consumer's job and is recorded
        in the output so it cannot be done from memory.

        Logits are computed ONLY at probed positions. The lm_head matvec is
        ~0.8 GFLOP against a whole forward, so this is a real saving on a long
        prefix and it changes nothing else: the hidden state carried forward is
        the same object either way."""
        total = len(ids)
        if total > self.max_ctx:
            fatal(f"forced length {total} exceeds max_ctx {self.max_ctx}")
        want = sorted(set(int(p) for p in probe_positions))
        for p in want:
            if not (0 <= p < total):
                fatal(f"probe position {p} outside the forced range [0,{total})")

        kv_k = np.zeros((self.n_layer, self.n_head_kv, total, self.head_dim), dtype=np.float32)
        kv_v = np.zeros((self.n_layer, self.n_head_kv, total, self.head_dim), dtype=np.float32)
        wanted = set(want)
        out = []
        for t, tok in enumerate(ids):
            x = self.forward_position(t, self.embed(tok), kv_k, kv_v)
            if t in wanted:
                h = rmsnorm(x, self.gf.tensor_dequant_full(self.output_norm_t), self.eps)
                lg = self.logits(h)
                rec = {"pos": t,
                       "argmax": int(np.argmax(lg)),
                       "next_id": int(ids[t + 1]) if t + 1 < total else None,
                       "top5": top5_from_logits(lg)}
                out.append(rec)
                # ⚠ BANKED AS IT LANDS, and this is a standing project rule, not
                # a nicety: a serial pass over a long prefix runs for HOURS, and
                # this box has killed long runs three times with no message. A
                # runner that banks only at the end converts every one of those
                # into a total loss. The JSONL is flushed per probe so a killed
                # run is still worth what it reached.
                if bank is not None:
                    import json as _j
                    bank.write(_j.dumps(rec) + "\n")
                    bank.flush()
            if progress is not None:
                progress(t, total)
        return out

    # ------------------------------------------------------------------
    # BATCHED teacher forcing
    # ------------------------------------------------------------------
    # ⚠⚠ WHY THIS EXISTS, AND WHY THE SERIAL PATH IS STILL THE ONE OF RECORD.
    # MEASURED on this box 2026-09-07: the serial path costs 45.7 s/position on
    # Llama-3.2-3B, because it re-dequantizes all 28 layers at EVERY position
    # (the deliberate M2 choice that keeps the working set bounded). The M3
    # ruling §8.9 requires M7 to freeze tau over position bands [600,1200) and
    # [1200,inf), so a probe at position 2851 needs 2851 forwards -- 32 hours
    # for ONE prompt. The instrument, not the engine, was the blocker.
    #
    # Batching moves the dequant OUT of the position loop: one dequant per
    # weight, then a [T, n_in] x [n_in, n_out] GEMM. Same graph, same order of
    # operations per row, one genuinely different thing: BLAS sums a dot
    # product in whatever order its blocking picks, so the batched logits are
    # NOT bit-identical to the serial ones.
    #
    # That difference is MEASURED, never assumed -- `--mode serial` still
    # exists and the equivalence control is a run of both over the same forced
    # stream. It has to be, because "three orders below tau" is exactly the
    # kind of claim this project has had to withdraw when it was argued
    # instead of measured.
    def teacher_force_batched(self, ids, probe_positions, progress=None):
        total = len(ids)
        if total > self.max_ctx:
            fatal(f"forced length {total} exceeds max_ctx {self.max_ctx}")
        want = sorted(set(int(p) for p in probe_positions))
        for p in want:
            if not (0 <= p < total):
                fatal(f"probe position {p} outside the forced range [0,{total})")

        D, H, HK = self.head_dim, self.n_head, self.n_head_kv
        half = D // 2
        scale = np.float32(1.0 / np.sqrt(np.float32(D)))

        # Angles for every position at once. Built with the SAME expression the
        # per-vector rope uses, just with a position column -- elementwise, no
        # reduction, so each row is bit-identical to rope_thetas(pos, ...).
        # Asserted in --selftest rather than reasoned about.
        th = rope_thetas_matrix(total, self.freq_base, D,
                                self.freq_factors, self.theta_mode)
        COS, SIN = f32(np.cos(th)), f32(np.sin(th))          # [T, half]

        def apply_rope(A):
            """A: [T, n_heads, D] -> roped in place-equivalent, same pairing as
            the scalar kernel."""
            out = np.empty_like(A)
            c = COS[:, None, :]
            s = SIN[:, None, :]
            if self.rope_kind == "norm":
                v0, v1 = A[:, :, 0::2], A[:, :, 1::2]
                out[:, :, 0::2] = v0 * c - v1 * s
                out[:, :, 1::2] = v0 * s + v1 * c
            else:
                v0, v1 = A[:, :, :half], A[:, :, half:]
                out[:, :, :half] = v0 * c - v1 * s
                out[:, :, half:] = v0 * s + v1 * c
            return f32(out)

        # Causal mask, built once. -inf before the softmax, so a masked score
        # contributes exactly zero weight and no row is ever all-masked (j=i is
        # always open).
        mask = np.triu(np.full((total, total), np.float32(-np.inf),
                               dtype=np.float32), k=1)

        X = f32(self.embd_matrix()[np.asarray(ids, dtype=np.int64)].copy())  # [T, E]

        for L, layer in enumerate(self.layers):
            A = rmsnorm_rows(X, self.gf.tensor_dequant_full(layer["attn_norm"]), self.eps)
            Q = f32(A @ self.gf.tensor_dequant_full(layer["attn_q"]).T).reshape(total, H, D)
            K = f32(A @ self.gf.tensor_dequant_full(layer["attn_k"]).T).reshape(total, HK, D)
            V = f32(A @ self.gf.tensor_dequant_full(layer["attn_v"]).T).reshape(total, HK, D)

            if self.qk_norm:
                Q = rmsnorm_rows(Q, self.gf.tensor_dequant_full(layer["attn_q_norm"]), self.eps)
                K = rmsnorm_rows(K, self.gf.tensor_dequant_full(layer["attn_k_norm"]), self.eps)
            Q = apply_rope(Q)
            K = apply_rope(K)

            O = np.empty((total, H, D), dtype=np.float32)
            for h in range(H):
                kvh = h // self.gqa_group
                S = f32(f32(Q[:, h, :] @ K[:, kvh, :].T) * scale)
                S = f32(S + mask)
                O[:, h, :] = f32(softmax_rows(S) @ V[:, kvh, :])
            X = f32(X + f32(O.reshape(total, H * D)
                            @ self.gf.tensor_dequant_full(layer["attn_output"]).T))

            F_ = rmsnorm_rows(X, self.gf.tensor_dequant_full(layer["ffn_norm"]), self.eps)
            gate = f32(F_ @ self.gf.tensor_dequant_full(layer["ffn_gate"]).T)
            up = f32(F_ @ self.gf.tensor_dequant_full(layer["ffn_up"]).T)
            X = f32(X + f32(f32(silu(gate) * up)
                            @ self.gf.tensor_dequant_full(layer["ffn_down"]).T))
            if progress is not None:
                progress(L, self.n_layer, "layers")

        onorm = self.gf.tensor_dequant_full(self.output_norm_t)
        out = []
        for p in want:
            h = rmsnorm(f32(X[p].copy()), onorm, self.eps)
            lg = self.logits(h)
            out.append({"pos": p,
                        "argmax": int(np.argmax(lg)),
                        "next_id": int(ids[p + 1]) if p + 1 < total else None,
                        "top5": top5_from_logits(lg)})
        return out


# Back-compat alias: q4km_grid.py / m2_classify_probes.py / ladder_teacher_force.py
# drive this file as a SUBPROCESS, so nothing imports the name -- but the M2/M3
# notes reference `Qwen3Ref` by name and the class is the same object.
Qwen3Ref = ModelRef


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("gguf", nargs="?")
    p.add_argument("--ids", help="comma-separated input token ids")
    p.add_argument("-n", type=int, default=64)
    p.add_argument("--dump-layers", metavar="DIR")
    p.add_argument("--probe-pos", type=int, default=None, metavar="N",
                   help="print the top-5 (id, logit) at generated position N "
                        "to stderr, in stoke's exact `probe pos=N: id=U "
                        "logit=F` format (src/main.c)")
    p.add_argument("--probe-all", metavar="FILE",
                   help="record the top-5 at EVERY generated position to FILE "
                        "as JSON. The logits are already computed at each "
                        "position, so this costs ~0.1% and turns one prefix "
                        "prefill into 64 probes instead of 1. Probes lie along "
                        "THIS run's stream, so a consumer may compare them to "
                        "another implementation only up to the first divergence.")
    p.add_argument("--force-ids", metavar="FILE",
                   help="TEACHER-FORCED mode (M7 entry probe): FILE holds the "
                        "full id stream, whitespace- or comma-separated. The "
                        "ref consumes exactly those ids and samples nothing, "
                        "so probes at every position compare the two "
                        "implementations on the SAME prefix. Requires "
                        "--probe-out; --ids/-n/--probe-pos/--probe-all do not "
                        "apply and are refused.")
    p.add_argument("--probe-at", metavar="LIST",
                   help="absolute positions to probe under --force-ids: "
                        "comma-separated, or 'all'. Default: the last position.")
    p.add_argument("--probe-out", metavar="FILE",
                   help="JSON output for --force-ids.")
    p.add_argument("--rope-theta", choices=("closed", "accum"), default="closed",
                   help="how the per-k rope angles are built. 'closed' (default) "
                        "is M2's frozen closed form and is INDEPENDENT of "
                        "llama.cpp/stoke's incremental theta*=theta_scale; "
                        "'accum' reproduces that accumulation. The M3 ruling "
                        "§8.9 names the difference as the first suspect if the "
                        "per-band tolerance grows at long positions, so it is a "
                        "switch and not a rewrite.")
    p.add_argument("--mode", choices=("serial", "batched"), default="serial",
                   help="how --force-ids runs the prefix. 'serial' (DEFAULT) is "
                        "M2's one-position-at-a-time path and is the definition "
                        "of record. 'batched' dequantizes each weight once and "
                        "GEMMs the whole prefix, which is ~500x faster and is "
                        "the only way the ruled [1200,inf) band is reachable at "
                        "all -- but BLAS chooses its own reduction order, so it "
                        "is NOT bit-identical and must be licensed by the "
                        "measured serial-vs-batched control, never by argument.")
    p.add_argument("--progress", action="store_true",
                   help="print a position counter to stderr (long prefixes "
                        "take minutes to hours; a silent run is "
                        "indistinguishable from a hung one).")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return run_selftest()

    if args.force_ids:
        return run_teacher_force(args)

    if args.probe_at or args.probe_out:
        print("fatal: --probe-at/--probe-out apply only to --force-ids",
              file=sys.stderr)
        return 1

    if not args.gguf or not args.ids:
        print("fatal: usage: ref_forward.py <gguf> --ids 1,2,3 -n 64 [--dump-layers DIR]", file=sys.stderr)
        return 1
    try:
        ids = [int(s) for s in args.ids.split(",")]
    except ValueError:
        print(f"fatal: --ids {args.ids!r} is not a comma-separated integer list", file=sys.stderr)
        return 1
    if not ids:
        print("fatal: --ids must be non-empty", file=sys.stderr)
        return 1
    if args.n <= 0:
        print(f"fatal: -n {args.n} must be positive", file=sys.stderr)
        return 1
    if args.probe_pos is not None and not (0 <= args.probe_pos < args.n):
        print(f"fatal: --probe-pos {args.probe_pos} outside the generated "
              f"range [0,{args.n})", file=sys.stderr)
        return 1

    all_probes = [] if args.probe_all else None
    try:
        model = ModelRef(args.gguf, theta_mode=args.rope_theta)
        out, probe = model.generate(ids, args.n, dump_dir=args.dump_layers,
                                     probe_pos=args.probe_pos,
                                     probe_all=all_probes)
    except GgufError as e:
        print(f"fatal: {e}", file=sys.stderr)
        return 1

    if all_probes is not None:
        # ids are written ALONGSIDE the probes deliberately: a probe set is
        # only interpretable against the stream it was taken along, and
        # separating them is how a later reader compares position k of one run
        # against position k of a different stream without noticing.
        import json as _json
        with open(args.probe_all, "w", encoding="utf-8") as fh:
            _json.dump({"ids": out, "n_input": len(ids), "probes": all_probes},
                       fh, indent=1)

    for tid in out:
        print(tid)
    if probe is not None:
        # Byte-identical to src/main.c: `probe pos=%d: id=%u logit=%.6f`,
        # 5 lines on stderr, rank-0 first -- so one comparator parses both.
        for tid, lg in probe:
            print(f"probe pos={args.probe_pos}: id={tid} logit={lg:.6f}",
                  file=sys.stderr)
    return 0


def run_environment() -> dict:
    """The instrument's own configuration, RECORDED (never set) here.

    ⚠⚠ THIS IS NOT BOOKKEEPING. numpy here is scipy-openblas 0.3.30, built
    DYNAMIC_ARCH / NO_AFFINITY: with no thread count set it sizes itself to
    every logical CPU and splits each reduction across them, so the LAST BITS
    OF EVERY LOGIT depend on how many threads it happened to pick and on which
    cores those threads landed. On this box 0-11 are the six P-cores' siblings
    and 12-15 are E-cores, so an unpinned run also mixes two core types.

    An instrument that cannot reproduce its own output cannot freeze a
    tolerance. Setting the thread count belongs to the LAUNCHER (OpenBLAS reads
    the env at load, which is before this file's first line runs); recording it
    belongs here, so a banked probe file says what produced it instead of
    leaving a later reader to assume."""
    env = {v: os.environ.get(v) for v in
           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    affinity = None
    try:                                   # Windows
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # ⚠ argtypes/restype are NOT optional here. Without them ctypes passes
        # the HANDLE as a 32-bit int on win64 and the call fails, so the
        # recorder wrote null while the launcher had already verified the mask
        # took -- an env block that silently records nothing.
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE,
                                               ctypes.POINTER(ctypes.c_size_t),
                                               ctypes.POINTER(ctypes.c_size_t)]
        k32.GetProcessAffinityMask.restype = wintypes.BOOL
        proc = ctypes.c_size_t(0)
        sysm = ctypes.c_size_t(0)
        if k32.GetProcessAffinityMask(k32.GetCurrentProcess(),
                                      ctypes.byref(proc), ctypes.byref(sysm)):
            affinity = hex(proc.value)
    except Exception:
        try:                               # POSIX
            affinity = sorted(os.sched_getaffinity(0))
        except Exception:
            affinity = None
    env["affinity_mask"] = affinity
    env["numpy"] = np.__version__
    return env


def run_teacher_force(args) -> int:
    """--force-ids driver. Kept out of main() so the flag-conflict refusals are
    in one place: a teacher-forced run that silently ignored --ids would grade
    a stream nobody asked for."""
    import json as _json
    import time as _time

    if not args.gguf:
        print("fatal: --force-ids needs the gguf path", file=sys.stderr)
        return 1
    if not args.probe_out:
        print("fatal: --force-ids requires --probe-out FILE", file=sys.stderr)
        return 1
    for name, val in (("--ids", args.ids), ("--probe-pos", args.probe_pos),
                      ("--probe-all", args.probe_all),
                      ("--dump-layers", args.dump_layers)):
        if val is not None:
            print(f"fatal: {name} does not apply to --force-ids", file=sys.stderr)
            return 1

    try:
        with open(args.force_ids, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        print(f"fatal: cannot read --force-ids {args.force_ids}: {e}", file=sys.stderr)
        return 1
    toks = [s for s in text.replace(",", " ").split() if s]
    try:
        ids = [int(s) for s in toks]
    except ValueError:
        print(f"fatal: --force-ids {args.force_ids} is not a list of integers",
              file=sys.stderr)
        return 1
    if not ids:
        print(f"fatal: --force-ids {args.force_ids} is empty", file=sys.stderr)
        return 1

    if args.probe_at is None:
        want = [len(ids) - 1]
    elif args.probe_at == "all":
        want = list(range(len(ids)))
    else:
        try:
            want = [int(s) for s in args.probe_at.split(",") if s.strip()]
        except ValueError:
            print(f"fatal: --probe-at {args.probe_at!r} is not 'all' or a "
                  f"comma-separated integer list", file=sys.stderr)
            return 1
        if not want:
            print("fatal: --probe-at selected no positions", file=sys.stderr)
            return 1

    t0 = _time.time()
    last = [t0]

    def progress(t, total, unit="positions"):
        now = _time.time()
        if now - last[0] >= 10.0 or t + 1 == total:
            done = t + 1
            rate = done / max(now - t0, 1e-9)
            # ⚠ The unit is a PARAMETER because the two modes count different
            # things: serial walks positions, batched walks layers. A batched
            # run that printed "28/28 positions" for a 2851-token stream is a
            # readout that contradicts the run it is describing.
            print(f"ref teacher-force: {done}/{total} {unit}, "
                  f"{rate:.3f} {unit[:3]}/s, "
                  f"eta {(total - done) / max(rate, 1e-9):.0f}s",
                  file=sys.stderr, flush=True)
            last[0] = now

    try:
        model = ModelRef(args.gguf, theta_mode=args.rope_theta)
        if args.mode == "batched":
            # The batched pass has no per-position boundary to bank at -- it
            # computes every position inside one layer sweep -- so incremental
            # banking is a SERIAL-path property and is not faked here. It does
            # not need one: it finishes in minutes where serial takes hours.
            probes = model.teacher_force_batched(
                ids, want, progress=progress if args.progress else None)
        else:
            bank_path = args.probe_out + ".partial.jsonl"
            with open(bank_path, "w", encoding="utf-8") as bank:
                probes = model.teacher_force(
                    ids, want, progress=progress if args.progress else None,
                    bank=bank)
    except GgufError as e:
        print(f"fatal: {e}", file=sys.stderr)
        return 1

    elapsed = _time.time() - t0
    # The ids are written WITH the probes for the reason --probe-all gives:
    # a probe set is interpretable only against the stream it was taken along.
    # sha256 of the id stream is the key that makes that checkable rather than
    # asserted -- a bank keyed by nothing is a bank that can go stale silently.
    import hashlib as _hashlib
    digest = _hashlib.sha256(
        (",".join(str(i) for i in ids)).encode("ascii")).hexdigest()
    with open(args.probe_out, "w", encoding="utf-8") as fh:
        _json.dump({"mode": f"teacher-force/{args.mode}",
                    "env": run_environment(),
                    "gguf": os.path.basename(args.gguf),
                    "arch": model.arch,
                    "rope_kind": model.rope_kind,
                    "qk_norm": model.qk_norm,
                    "freq_factors": model.freq_factors is not None,
                    "theta_mode": model.theta_mode,
                    "n_ids": len(ids),
                    "ids_sha256": digest,
                    "seconds": round(elapsed, 3),
                    "probes": probes}, fh, indent=1)
    print(f"ref teacher-force: {len(ids)} positions, {len(probes)} probe(s), "
          f"{elapsed:.1f}s -> {args.probe_out}", file=sys.stderr)
    return 0


def run_selftest() -> int:
    """No GGUF needed. Positive dtype checks on every kernel, a NEGATIVE
    check proving f32() actually catches a float64 leak (not a dead
    check), and the argmax tie-break rule."""
    failures = 0

    def check(label, ok):
        nonlocal failures
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            failures += 1

    x = np.array([1.0, -2.0, 3.0, -0.5], dtype=np.float32)
    w = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    check("rmsnorm output dtype float32", rmsnorm(x, w, 1e-6).dtype == np.float32)
    check("rope_neox output dtype float32 (head_dim=4)",
          rope_neox(x, 5, 5e6, 4).dtype == np.float32)
    check("softmax output dtype float32", softmax(x).dtype == np.float32)
    check("silu output dtype float32", silu(x).dtype == np.float32)

    # rope at pos 0 is the identity (P3.3: theta_k=0 for all k when pos=0)
    r0 = rope_neox(x, 0, 5e6, 4)
    check("rope_neox at pos=0 is identity", np.allclose(r0, x, atol=0))

    # negative test: f32() must REJECT a float64 leak, not silently cast it
    leaked = np.array([1.0, 2.0], dtype=np.float64)
    caught = False
    try:
        f32(leaked)
    except AssertionError:
        caught = True
    check("f32() raises AssertionError on a float64 leak (not a dead check)", caught)

    # argmax tie rule: first/lowest index wins
    tied = np.array([1.0, 3.0, 3.0, 2.0], dtype=np.float32)
    check("argmax tie -> first/lowest index", int(np.argmax(tied)) == 1)

    # top5_from_logits must reproduce src/model.c record_top5() exactly:
    # descending logit, ties broken by the LOWER id.
    lg = np.array([5.0, 9.0, 9.0, 1.0, 7.0, 9.0, 0.0], dtype=np.float32)
    t5 = top5_from_logits(lg)
    check("top5 ranks by descending logit with lower-id tiebreak",
          [i for i, _ in t5] == [1, 2, 5, 4, 0])
    check("top5 returns exactly 5 (id, logit) pairs",
          len(t5) == 5 and all(len(p) == 2 for p in t5))
    check("top5 logits are the values at those ids",
          [v for _, v in t5] == [9.0, 9.0, 9.0, 7.0, 5.0])
    # rank-0 must agree with argmax, or a probe would contradict the id
    # the same run generated.
    check("top5 rank-0 == argmax", t5[0][0] == int(np.argmax(lg)))

    # ---------------------------------------------------------------- M7 rope
    # The NORM/NEOX pairing is `kernels.h`'s TRAP #3 from the other side: the
    # wrong one is FLUENT, not broken, so none of these may be graded by
    # eyeballing output. Each is an exact algebraic identity.
    check("rope_norm output dtype float32 (head_dim=4)",
          rope_norm(x, 5, 5e6, 4).dtype == np.float32)
    check("rope_norm at pos=0 is identity",
          np.array_equal(rope_norm(x, 0, 5e6, 4), x))

    rng = np.random.default_rng(20260907)
    D = 128
    half = D // 2
    vec = rng.standard_normal(D).astype(np.float32)
    fb = 500000.0
    pos = 1234

    # (1) THE PAIRING GRADER: de-interleave -> NEOX -> re-interleave must
    # reproduce NORM BIT-FOR-BIT. No golden, no tolerance -- the two kernels
    # do the same multiplications on the same angles, so anything but equality
    # is a real difference. An element-indexed factors bug survives a UNIFORM
    # factors vector, so the factors here are deliberately non-uniform and the
    # identity is checked with and without them.
    for label, ff in (("no factors", None),
                      ("non-uniform factors",
                       (1.0 + 31.0 * rng.random(half)).astype(np.float32))):
        de = np.empty(D, dtype=np.float32)
        de[:half] = vec[0::2]
        de[half:] = vec[1::2]
        got_neox = rope_neox(de, pos, fb, D, ff)
        re = np.empty(D, dtype=np.float32)
        re[0::2] = got_neox[:half]
        re[1::2] = got_neox[half:]
        check(f"rope_norm == permuted rope_neox, bit-exact ({label})",
              np.array_equal(re, rope_norm(vec, pos, fb, D, ff)))

    # (1b) ⚠ (1) IS STRUCTURALLY BLIND TO THE ANGLES AND THAT IS PROVED BY
    # MUTATION: both pairings call the SAME rope_thetas, so any error inside it
    # moves both sides of the identity together and (1) still passes. Two
    # mutations sailed through it -- freq_factors indexed by element instead of
    # by pair, and the llama3 division folded into the accumulator. So the
    # angles get their own check against a formula written out here
    # INDEPENDENTLY, in plain Python, with NON-UNIFORM factors (a uniform
    # vector cannot distinguish per-element from per-pair indexing at all).
    ffn = (1.0 + 31.0 * rng.random(half)).astype(np.float32)
    want = np.array([pos * (fb ** (-2.0 * k / D)) / float(ffn[k])
                     for k in range(half)], dtype=np.float64)
    got = rope_thetas(pos, fb, D, ffn).astype(np.float64)
    check("rope_thetas matches an independently written closed form "
          "(non-uniform factors, indexed BY PAIR)",
          bool(np.max(np.abs(got - want) / np.abs(want)) < 1e-5))

    # (1c) The 'accum' path must handle factors the same way -- ggml divides AT
    # USE and leaves the accumulator alone, so folding the divide into the
    # recurrence compounds it across k. Without factors both paths agree
    # trivially, which is why the mutation survived until this check existed.
    ta_ff = rope_thetas(pos, fb, D, ffn, "accum").astype(np.float64)
    check("theta_mode='accum' with factors agrees with 'closed' to <1e-4 "
          "(the divide is at USE, not in the accumulator)",
          bool(np.max(np.abs(ta_ff - got) / np.abs(got)) < 1e-4))

    # (2) The two pairings must actually DIFFER on real data, or (1) would be
    # satisfied by a dispatch that returned the same kernel twice.
    check("rope_norm and rope_neox differ at pos>0 (dispatch is load-bearing)",
          not np.array_equal(rope_norm(vec, pos, fb, D),
                             rope_neox(vec, pos, fb, D)))

    # (3) freq_factors must change the answer, or the llama3 scaling could be
    # silently dropped and every short-context test would still pass.
    ffv = np.full(half, 8.0, dtype=np.float32)
    check("freq_factors change the rotation (not silently ignored)",
          not np.array_equal(rope_norm(vec, pos, fb, D, ffv),
                             rope_norm(vec, pos, fb, D)))
    # ... and they must be indexed BY PAIR: ff[k] scales theta_k, so a uniform
    # divisor is exactly a rescaled position. theta_k/c == theta_k(pos/c) only
    # for the closed form, which is what makes this checkable at all.
    check("uniform freq_factors == closed-form position scaling",
          np.allclose(rope_thetas(pos, fb, D, ffv),
                      rope_thetas(pos / 8.0, fb, D), rtol=1e-6, atol=0))

    # (4) The 'accum' diagnostic must be a DIFFERENT float sequence that agrees
    # to well inside any plausible tau -- if it were bit-identical the switch
    # would be dead, and if it disagreed materially one of the two is wrong.
    th_c = rope_thetas(pos, fb, D)
    th_a = rope_thetas(pos, fb, D, None, "accum")
    check("theta_mode='accum' is a genuinely different float sequence",
          not np.array_equal(th_c, th_a))
    check("theta_mode='accum' agrees with 'closed' to <1e-4 relative",
          bool(np.max(np.abs(th_a - th_c) / np.maximum(np.abs(th_c), 1e-30)) < 1e-4))

    # ------------------------------------------------- batched-path identity
    # The batched forward is justified ONLY as the same computation
    # reorganized. Everything in it that is elementwise or per-row must
    # therefore be BIT-IDENTICAL to the scalar kernel, and is checked to be.
    # What is left over -- the GEMM reduction order, which BLAS owns and no
    # test here can pin -- is the entire licensed difference, and it is what
    # the serial-vs-batched control measures on the real model.
    Xr = rng.standard_normal((7, 96)).astype(np.float32)
    wr = rng.standard_normal(96).astype(np.float32)
    check("rmsnorm_rows row == rmsnorm(row), bit-exact",
          all(np.array_equal(rmsnorm_rows(Xr, wr, 1e-5)[i],
                             rmsnorm(Xr[i], wr, 1e-5)) for i in range(7)))
    check("softmax_rows row == softmax(row), bit-exact",
          all(np.array_equal(softmax_rows(Xr)[i], softmax(Xr[i]))
              for i in range(7)))
    TH = rope_thetas_matrix(9, fb, D, ffn)
    check("rope_thetas_matrix row == rope_thetas(pos), bit-exact",
          all(np.array_equal(TH[p], rope_thetas(p, fb, D, ffn))
              for p in range(9)))
    TH_a = rope_thetas_matrix(4, fb, D, ffn, "accum")
    check("rope_thetas_matrix row == rope_thetas(pos), bit-exact (accum)",
          all(np.array_equal(TH_a[p], rope_thetas(p, fb, D, ffn, "accum"))
              for p in range(4)))

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
