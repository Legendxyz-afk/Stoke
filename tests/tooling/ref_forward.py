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


def rope_neox(vec, pos, freq_base, head_dim):
    """NEOX pairing (k, k+half), k in [0,half). theta_k = pos *
    freq_base^(-2k/head_dim) -- the CLOSED-FORM per-k contract derived in
    m2-probe-notes.md P3.3 from llama.cpp's incremental theta*=theta_scale
    loop (mathematically equivalent; P3.3 states this closed form as the
    contract stoke's C rope_neox must implement, so the ref matches that,
    not llama.cpp's own accumulation path -- token-level parity is the
    gate, bit-exact reduction order is explicitly not required, M2 note
    section 4)."""
    vec = f32(vec)
    half = head_dim // 2
    k = np.arange(half, dtype=np.float32)
    exponent = f32(np.float32(-2.0) * k / np.float32(head_dim))
    theta = f32(np.float32(pos) * np.power(np.float32(freq_base), exponent))
    cos = f32(np.cos(theta))
    sin = f32(np.sin(theta))
    v0, v1 = vec[:half], vec[half:]
    out = np.empty_like(vec)
    out[:half] = v0 * cos - v1 * sin
    out[half:] = v0 * sin + v1 * cos
    return f32(out)


def softmax(x):
    x = f32(x)
    e = f32(np.exp(x - np.max(x)))
    return f32(e / np.sum(e, dtype=np.float32))


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

class Qwen3Ref:
    def __init__(self, gguf_path: str):
        self.gf = GGUFFile(gguf_path)
        kv = self.gf.kv

        def req(key):
            if key not in kv:
                fatal(f"missing required config key {key!r}")
            return kv[key]

        arch = req("general.architecture")
        if arch != "qwen3":
            fatal(f"general.architecture={arch!r}, expected 'qwen3' (this ref only implements qwen3)")

        self.n_layer = int(req("qwen3.block_count"))
        self.n_embd = int(req("qwen3.embedding_length"))
        self.n_ff = int(req("qwen3.feed_forward_length"))
        self.n_head = int(req("qwen3.attention.head_count"))
        self.n_head_kv = int(req("qwen3.attention.head_count_kv"))
        self.head_dim = int(req("qwen3.attention.key_length"))       # TRAP #1: own key, never n_embd//n_head
        v_len = int(req("qwen3.attention.value_length"))
        if v_len != self.head_dim:
            fatal(f"value_length {v_len} != key_length {self.head_dim} (M2 requires equal)")
        self.freq_base = float(req("qwen3.rope.freq_base"))
        self.eps = float(req("qwen3.attention.layer_norm_rms_epsilon"))
        self.max_ctx = min(int(req("qwen3.context_length")), 4096)    # M2 note section 2 cap
        if self.n_head % self.n_head_kv != 0:
            fatal(f"head_count {self.n_head} not divisible by head_count_kv {self.n_head_kv}")
        self.gqa_group = self.n_head // self.n_head_kv

        eos = kv.get("tokenizer.ggml.eos_token_id")
        self.eos_id = int(eos) if eos is not None else None

        self.tok_embd_t = self.gf.get_tensor("token_embd.weight")
        self.output_norm_t = self.gf.get_tensor("output_norm.weight")
        self.output_t = self.gf.tensors.get("output.weight")   # TRAP #2: tied if absent
        self.vocab_size = self.tok_embd_t.dims[1]

        self.layers = []
        for L in range(self.n_layer):
            names = {k: f"blk.{L}.{k}.weight" for k in (
                "attn_norm", "attn_q", "attn_q_norm", "attn_k", "attn_k_norm",
                "attn_v", "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down")}
            layer = {k: self.gf.get_tensor(name) for k, name in names.items()}
            self._validate_layer_shapes(L, layer)
            self.layers.append(layer)

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

            qn = self.gf.tensor_dequant_full(layer["attn_q_norm"])
            kn = self.gf.tensor_dequant_full(layer["attn_k_norm"])
            for h in range(self.n_head):
                q[h] = rope_neox(rmsnorm(q[h], qn, self.eps), t_pos, self.freq_base, D)
            for h in range(self.n_head_kv):
                k[h] = rope_neox(rmsnorm(k[h], kn, self.eps), t_pos, self.freq_base, D)

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
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return run_selftest()

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
        model = Qwen3Ref(args.gguf)
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

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
