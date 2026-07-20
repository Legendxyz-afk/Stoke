# M2 entry — op-by-op spec + parity-debug protocol

*2026-07-19. Milestone-entry design note (model-allocation: Fable writes these;
Opus 4.8 drives the M2 sessions from it). Parent spec:
`../specs/2026-07-16-phase1-concurrent-cpu-core-design.md`; plan:
`../plans/2026-07-16-phase1-implementation.md`. Written at the `m1` tag.*

## 1. Scope and gate

M2 delivers the **scalar forward pass**: F32 + Q8_0 dequant-dot kernels,
`model.c` fixed Qwen3 pass, naive contiguous KV behind the `kv_view` accessor,
greedy sampling, and a `stoke run` subcommand.

**GATE:** greedy token parity vs llama.cpp over a 50-prompt committed corpus —
stoke's 64 generated ids match the oracle's token-exactly, per prompt, under
the waiver protocol of §8. Math proven before any perf work.

Explicit non-goals at M2: speed (scalar single-thread is fine; expect seconds
per token), SIMD, threads in kernels, batching, K-quants (M3), sampling beyond
greedy (M6), chat-template plumbing in the gate path (ChatML is already
oracle-proven at M1; the parity gate runs raw prompts).

## 2. Model file decision (binding)

The parity model is **Qwen3-4B-Instruct-2507 Q8_0** — a second GGUF, acquired
at M2 dispatch (verified present 2026-07-19: bartowski's HF repo publishes
`Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf`, 4.28GB — same repo as the dev file;
its Q4_0 will serve M3 the same way; magic-verify + record sha256 + `stoke
info` it before anything else). Rationale: the spec's quant
ladder is F32 → Q8_0 first; the Q4_K_M development file is unusable for M2
because its tensors are Q4_K/Q6_K super-block quants, which are M3 scope.
Q8_0 dequant is trivially exact (d × q), so M2's parity signal is pure
forward-pass math, not quant-kernel subtlety.

The Q4_K_M file stays exactly where it is — every M1 gate (info golden,
tokenizer goldens) keeps running against it unchanged. The tokenizer tables of
the two files should be identical; the M2 bring-up must assert that (`stoke
tokenize` a fixed line via both files, same ids) rather than assume it.

RAM budget on the 16GB box: 4.3GB mmap (shared page cache across processes) +
f32 KV at 288 KB/token (36 layers × 2 × 8 heads × 128 dim × 4B) → a 4096-token
context cap costs ~1.2GB. M2 fixes `max_ctx = 4096`.

## 3. Config-load contract

Values below were probed from the real Q4_K_M file (committed golden
`tests/data/qwen3-4b-info.golden`); the Q8_0 file must be re-probed at
dispatch and is expected to match on everything except tensor quant types.
REQUIRED = fatal if absent (loud-failure doctrine); no silent defaults except
where `_or` is stated.

| GGUF key | value | notes |
|---|---|---|
| `general.architecture` | `qwen3` | REQUIRED, exact match |
| `qwen3.block_count` | 36 | REQUIRED |
| `qwen3.context_length` | 262144 | read, then clamp to M2's 4096 cap |
| `qwen3.embedding_length` | 2560 | REQUIRED (`n_embd`) |
| `qwen3.feed_forward_length` | 9728 | REQUIRED (`n_ff`) |
| `qwen3.attention.head_count` | 32 | REQUIRED (`n_head`) |
| `qwen3.attention.head_count_kv` | 8 | REQUIRED (`n_head_kv`; GQA group = 4) |
| `qwen3.attention.key_length` | 128 | REQUIRED (`head_dim`) |
| `qwen3.attention.value_length` | 128 | REQUIRED (must equal key_length at M2) |
| `qwen3.rope.freq_base` | 5e6 | REQUIRED |
| `qwen3.attention.layer_norm_rms_epsilon` | 1e-6 | REQUIRED |

**TRAP #1 — head_dim is a first-class key, not a quotient.** For Qwen3-4B,
`head_dim = 128` while `n_embd / n_head = 2560/32 = 80`. Q/K/V projections map
2560 → `n_head*head_dim` = 4096 / 1024 / 1024, and `attn_output` maps
4096 → 2560. Any code that computes head_dim by division is wrong on this
model and will fail shape validation immediately — validate every tensor's
dims against the table below at load and fatal on mismatch.

**TRAP #2 — the LM head is tied.** The file has **no `output.weight`**
tensor: 398 tensors = 36 × 11 per-layer + `token_embd.weight` +
`output_norm.weight`. Logits = `token_embd.weight` (as a [vocab × n_embd]
matrix) dotted with the final hidden state. Rule: if `output.weight` exists,
use it; else tie to `token_embd.weight` (that is llama.cpp's semantic, and
Llama-3.2-3B at M7 is also tied).

Per-layer tensor set (names exact, from the golden): `blk.N.attn_norm`,
`attn_q`, `attn_q_norm`, `attn_k`, `attn_k_norm`, `attn_v`, `attn_output`,
`ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down` (all `.weight`; no biases in this
family). `attn_q_norm`/`attn_k_norm` are F32 `[128]` — per-head RMSNorm
weights, one vector shared by all heads of that layer (the Qwen3 delta).

**Dims convention (pin it once):** GGUF/ggml `dims[]` as printed by `stoke
info` lists `ne[0]` first = the contraction/input length = the row length in
memory (fastest-varying). `attn_q.weight dims=[2560,4096]` is 4096 rows of
2560 — `y[r] = Σ_c W[r][c] · x[c]`, `y` has 4096 entries. Rows are contiguous;
row r of a Q8_0 tensor starts at `data + r * (2560/32) * 34` bytes.

## 4. Numeric conventions (binding)

- All activations f32. Weights stay quantized in the mmap; dequant happens
  inside dot kernels only. Never materialize a dequantized tensor (token_embd
  alone would be 1.6GB f32).
- **Deliberate divergence from llama.cpp:** llama.cpp quantizes activations to
  Q8_0 and runs integer dot products; stoke M2 keeps activations f32 and
  dequantizes weights in the dot loop. Simpler, strictly more accurate, and
  the token-level gate absorbs the difference. M3 revisits activation
  quantization for VNNI.
- Accumulation: f32, sequential, fixed order. Determinism (same input → same
  ids) is required; bit-matching llama.cpp's reduction order is explicitly not.
- f16 → f32 (Q8_0 block scales are f16): one conversion helper, unit-tested
  against known bit patterns (0x3C00=1.0, 0x0001=subnormal, 0x7C00=inf —
  fatal on inf/nan scales, they cannot occur in a sane file).
- Q8_0 block layout (PROBE at dispatch before trusting): 32 elements per
  block, `{f16 d; i8 qs[32]}` = 34 bytes; value_i = d × qs[i]. Verify against
  a real tensor: `nbytes == ne0*ne1/32*34`, and the L1 dump-tool cross-check
  in §9 proves the byte interpretation independently.

## 5. Kernel inventory (scalar, exact contracts)

Each kernel: pure function, no globals, unit-tested against hand-computed
vectors (§9 L0) before model.c exists. Signatures are the M2 shape — M3 adds
SIMD twins behind the same contracts.

- `f16_to_f32(u16) -> f32`
- `dequant_row_q8_0(const void* row, u64 n, f32* out)` — n multiple of 32.
- `vec_dot_q8_0_f32(const void* row, const f32* x, u64 n) -> f32` — the
  workhorse: per block, `d * Σ qs[i]*x[i]` (f32 accumulate, sequential).
- `vec_dot_f32(const f32* a, const f32* b, u64 n) -> f32`
- `matvec(W_tensor, x[n_in], y[n_out])` — row loop over the right vec_dot per
  tensor type; the ONLY dispatch point on quant type at M2 (F32 and Q8_0;
  fatal on anything else, naming the tensor).
- `rmsnorm(const f32* x, const f32* w, u64 n, f32 eps, f32* y)` —
  `y[i] = x[i] / sqrt(mean(x²)+eps) * w[i]`, f32 mean.
- `rope_neox(f32* vec, u64 head_dim, u64 pos, f32 freq_base)` — applied
  per head, to Q and K, AFTER QK-norm. NEOX pairing: rotate `(v[i],
  v[i+64])` for i in [0,64), angle `θ_i = pos · freq_base^(−2i/head_dim)`.
  **TRAP #3:** NEOX pairs are (i, i+half), NOT adjacent (2i, 2i+1) — the
  latter is the GPT-J/original style and produces plausible-looking garbage.
  The exact formula and the norm→rope order MUST be re-confirmed from
  llama.cpp source at dispatch (§10 P3) — this note's formula is the
  hypothesis to check, not the authority.
- `softmax(f32* x, u64 n)` — max-subtract, exp, normalize.
- `silu(z) = z / (1 + e^(−z))`; SwiGLU combine `h = silu(g) * u` elementwise.
- `argmax(const f32* logits, u64 n) -> u32` — first-lowest-index wins ties.

## 6. Forward pass (per token position t, per layer)

```
x = embed_row(token[t])                       # dequant one token_embd row
for L in 0..35:
    a  = rmsnorm(x, attn_norm[L])
    q  = matvec(attn_q[L], a)                 # [4096] = 32 heads × 128
    k  = matvec(attn_k[L], a)                 # [1024] = 8 heads × 128
    v  = matvec(attn_v[L], a)                 # [1024]
    for h in 0..31: rmsnorm(q_head[h], attn_q_norm[L]); rope_neox(q_head[h], t)
    for h in 0..7:  rmsnorm(k_head[h], attn_k_norm[L]); rope_neox(k_head[h], t)
    kv_append(L, t, k, v)                     # via kv_view
    for h in 0..31:                           # GQA: q-head h reads kv-head h/4
        scores[j] = dot(q_head[h], K[L][h/4][j]) / sqrt(128)   for j in 0..t
        softmax(scores, t+1)
        attn_out_head[h] = Σ_j scores[j] * V[L][h/4][j]
    x += matvec(attn_output[L], attn_out)     # [4096] -> [2560]
    f  = rmsnorm(x, ffn_norm[L])
    x += matvec(ffn_down[L], silu(matvec(ffn_gate[L], f)) * matvec(ffn_up[L], f))
h = rmsnorm(x, output_norm)
logits[v] = vec_dot(token_embd_row[v], h)     for v in 0..151935   # tied head
next = argmax(logits)
```

Prefill = the same loop over prompt positions (no batching, no shortcuts —
one position at a time is acceptable at M2). The causal mask is implicit:
position t attends to [0..t].

**`kv_view` accessor (plan mandate — M4's paged tables must slot in without
touching attention math).** Concrete M2 contract:

```c
typedef struct { f32* k; f32* v; } KvSlot;    /* both [head_dim] */
/* slot for (layer, kv_head, pos); M2 backs this with one contiguous
 * malloc'd [n_layer][max_ctx][n_head_kv][head_dim] pair; M4 replaces the
 * body with block-table lookup. Attention code calls ONLY this. */
KvSlot kv_view(Kv* kv, u32 layer, u32 kv_head, u32 pos);
```

Attention iterates `pos` and calls `kv_view` per position — no code outside
`kv.c` may compute KV addresses. (Scalar cost of the call is irrelevant at
M2; M3/M4 may add a contiguous-run fast path INSIDE kv.c if profiling says
so.)

## 7. `stoke run` CLI contract

`stoke run <model.gguf> [--raw] [-n N] [-p "text" | -f file] [--dump dir]`

- `--raw`: no chat template — input text is tokenized (`add_special=false`,
  parse-special OFF) and fed as-is. **The parity gate runs raw**, eliminating
  template variance; the ChatML path needs no new gate (M1 proved it).
  Without `--raw`: wraps input as a single user message via `tok_apply_chatml`
  (smoke-tested, not parity-gated at M2).
- Output: generated ids one per line to stdout (`%u\n`), then the decoded
  continuation after a `---` separator line. Goldens/parity diff the id lines
  only (text is for humans).
- `-n` default 64. Generation stops at `-n` or on eos (151645) — eos, if
  generated, IS printed (the oracle prints its final token too; matching
  streams must match on it).
- `--dump dir`: writes per-layer activation dumps (§9 L2/L3) for the LAST
  position processed: raw f32 binary, one file per checkpoint, fixed names
  (`L07.post_attn.f32` etc.). Diagnostic only, no stability promise.

## 8. The parity gate (mechanics, corpus, waiver protocol)

**Oracle front-end: `llama-server` + `POST /completion`** (not llama-cli) —
machine-readable output, no prompt-echo parsing, and it returns top-n
logprobs in the same run, which the waiver protocol needs. Expected request
shape (PROBE the exact field names against b10037 at dispatch, two identical
runs required before trusting): `{"prompt": <str>, "n_predict": 64,
"temperature": 0, "n_probs": 5, "seed": 0}`, server started `-m <Q8_0> -t 1
--parallel 1`. Single-thread the oracle: determinism across its own runs is a
precondition the dispatch probe must demonstrate, not assume.

**PROBED (W1, 2026-07-19 — b10037's real response schema; supersedes the
expected shape above wherever they differ):**

- The request MUST add `"return_tokens": true`; the generated-id ground
  truth is the top-level `tokens` field (one id per raw generation step;
  without the flag it comes back `[]`).
- `completion_probabilities[*].id` is NOT that sequence: the server
  coalesces runs of tokens that individually encode incomplete UTF-8 into
  single entries (measured: 64 real steps → 54 entries on a multilingual
  prompt), so past any coalescing point its index no longer aligns with
  `tokens`. `return_tokens` is pure response-shaping — confirmed it does
  not alter generation.
- Per-token top-5 logprobs live at `completion_probabilities[i].
  top_logprobs[j] = {id, token, bytes, logprob}`; gap(top1,top2) =
  `top_logprobs[0].logprob − top_logprobs[1].logprob`. W4's harness must
  handle the `tokens`-vs-`completion_probabilities` index misalignment on
  coalescing prompts (align by consuming `tokens` ids against each
  entry's `bytes`, or equivalent).
- `tokens_predicted` echoes the configured `n_predict` even when
  `completion_probabilities` has fewer entries — never treat it as an
  output-length field; use `len(tokens)`.
- Determinism + no-chat-template on `/completion`: CONFIRMED (two
  identical greedy runs token-identical; `tokens_evaluated` equals the
  independently known prompt token count).
- PENDING (attribution experiment recorded in the ledger, 2026-07-20):
  whether this gate's server invocation gains `-ctk f32 -ctv f32` —
  ref-vs-oracle validation hit a non-waivable divergence consistent with
  the oracle's default f16 KV cache rather than a ref bug. Resolve (memo
  to Kabir if the invocation changes) BEFORE the W4 brief is written.

**Corpus: 50 prompts, committed** under `tests/data/parity/` (one file each,
UTF-8, ≤512 tokens): ~15 English prose/instruction, ~10 code (C/Python/JSON),
~10 multilingual (CJK, Devanagari, Arabic, emoji — reuse tok-corpus material),
~5 math/symbolic, ~5 long-context (400–512 tokens), ~5 adversarial-ish
(whitespace runs, mixed scripts, near-empty). Plus a committed **reserve pool
of 10** more for the waiver protocol. Committed BEFORE the C forward pass
lands (TDD ordering, and it keeps corpus selection uninfluenced by early
failures).

**Gate:** for every prompt, stoke's 64 ids (raw mode, greedy) == oracle's,
token-exact. All 50 must pass on ONE recorded run (per-prompt caching/resume
during development is fine; the gate run itself is clean-start).

**Near-tie waiver protocol** — the one escape hatch, for genuine argmax
coin-flips (different summation order → last-ulp logit differences can
legitimately swap a tied top-2). All conditions required:

1. First divergence at position p: oracle's top-5 logprobs at p (from
   `n_probs`) show gap(top1, top2) < ε; stoke's logits at p show
   gap(top1, top2) < ε; ε = **1e-3 nats**, and
2. the two candidates are the SAME pair, swapped, and
3. the divergent prompt is REPLACED by the next unused reserve prompt (never
   re-rolled, never trimmed), and the waiver is recorded in the gate report:
   prompt id, position, both gaps, both pairs.

Budget: **≤2 waivers per gate run.** A third, or any divergence failing
conditions 1–2, is GATE FAIL → §9. ε and the budget are gate parameters —
changing either is gate-loosening and follows the escalation protocol
(memo to Kabir; companion note
`2026-07-19-m3-m7-forward-designs-and-risk-playbook.md` §0). (Expectation to hold in mind: with correct math, zero
waivers is the likely outcome; the protocol exists so a real coin-flip
doesn't force a week of chasing ghosts.)

## 9. Parity-debug protocol (when the gate fails — cheapest rung first)

The vendored oracle is binaries-only: llama-cli/server can say THAT tokens
diverge, never WHERE the math went wrong. So M2 builds its own bisection
instrument as a first-class deliverable:

**The numpy reference (`tests/tooling/ref_forward.py`)** — a ~200-line
float32 numpy implementation of §6, reading tensors straight from the GGUF
via its own minimal parser (struct/numpy only, no gguf pip package — an
INDEPENDENT byte-interpretation that cross-checks gguf.c for free).
`dtype=float32` everywhere, matmuls included — numpy's default float64 would
hide exactly the class of bug being hunted.

**Trust order (establish before first use):** llama-server (end-to-end token
oracle) > numpy ref > C code under test. The ref must first REPRODUCE the
oracle's 64 ids on ≥3 corpus prompts (token-validated); only then is it
trusted as the layer-level authority. If ref and oracle disagree on tokens,
fix the ref first — its formulas were transcribed from the same probe pack
the C will be built from, so a shared misreading is caught HERE, before the C
exists. (The ref is slow — minutes per prompt is fine, it exists for 8-token
debug inputs, and its 3-prompt validation can run overnight once.)

The ladder:

- **L0 — kernel unit vectors** (exist before model.c): every §5 kernel tested
  against hand-computed golden arrays committed in the C tests (rmsnorm of a
  known 4-vector, rope at pos 0/1/17 with freq_base 5e6, softmax of knowns,
  silu of knowns, a handcrafted Q8_0 block, f16 edge patterns). No oracle
  involved — pure formula checks. Any L3 root-cause gains a permanent L0
  vector reproducing it.
- **L1 — tensor byte interpretation**: `tests/tooling/dump_tensor.py` (same
  independent parser) dequants any tensor row; compare stoke's
  `dequant_row_q8_0` byte-for-byte (f32-exact — same formula, same inputs).
  Run for token_embd row of token 9707 ("Hello"-ish; pick any fixed id) +
  one attn_q row + one ffn_down row, all layers 0/17/35.
- **L2 — layer bisection**: `stoke run --dump` on the FIXED 8-token debug
  prompt (committed) vs the ref's per-layer outputs. First layer whose
  output diverges (max-abs-diff > 1e-3, diagnostic threshold, not a gate)
  names the neighborhood.
- **L3 — op bisection inside the layer**: dumps at post-attn_norm, post-Q/K/V,
  post-qknorm, post-rope, post-attn-softmax-weighted-sum, post-attn_output,
  post-ffn_norm, post-gate/up, post-down. First divergent checkpoint names
  the op; fix; add the L0 vector; rerun the ladder from L0.
- **Escalation** (protocol in the companion note
  `2026-07-19-m3-m7-forward-designs-and-risk-playbook.md` §0): same-cause
  gate failure twice, or >1 day stuck on one divergence → stop and write
  Kabir the half-page memo. Never loosen ε, never swap prompts outside the
  waiver protocol, never "temporarily" skip a failing corpus entry.

## 10. Dispatch-time probe pack (Opus assembles before any brief goes out)

Per the manager/worker protocol (constants probed at dispatch, never
recalled — this note included):

- **P1**: acquire Q8_0 file; sha256; `stoke info` full dump → the M2 probe
  notes; diff config keys vs the table in §3 (expect: identical); record the
  tensor-type histogram (expect: Q8_0 matrices + F32 norms; token_embd type
  noted — whatever it is, matvec must support it).
- **P2**: Q8_0 byte layout — hex-dump the first 68 bytes of a known tensor,
  hand-annotate d/qs per block, verify against §4's claimed layout AND
  against dump_tensor.py's independent read.
- **P3**: fetch llama.cpp source (the vendored zip is binaries-only —
  WebFetch the b10037 tag on GitHub): `llm_build_qwen3` (QK-norm position
  relative to RoPE; attention scale; anything this note's §6 got wrong wins
  there, then this note gets amended), `ggml_rope` NEOX path (pairing +
  angle formula), Q8_0 struct in ggml-quants/ggml-common. Paste the relevant
  source lines into the probe notes verbatim.
- **P4**: llama-server /completion determinism + schema: two identical runs
  → byte-identical ids; capture exact response field names for ids and
  logprobs; confirm `-t 1` and greedy settings produce stable output; confirm
  the completion endpoint applies NO chat template.
- **P5**: tokenizer-table identity across the two files (§2's assert).

## 11. Worker decomposition (sketch — Opus refines into SDD briefs)

- **W1**: probe pack P1–P5 + `dump_tensor.py` + `ref_forward.py` +
  ref-vs-oracle 3-prompt token validation. (The riskiest knowledge work
  lands first, in Python where iteration is cheap.)
- **W2**: kernels (§5) + L0 unit vectors + f16 helper. Pure C, no model.c.
- **W3**: config load (§3, every shape validated) + model.c (§6) + kv.c M2
  body + `stoke run` (§7) + dump flags. Smoke: 8-token debug prompt runs L2
  clean vs the ref before any parity attempt.
- **W4**: parity corpus + reserve pool + gate harness (process-parallel
  across prompts — mmap page cache is shared, ~6 workers fit; kernels stay
  single-thread) + the recorded gate run + report.

Each worker Opus-reviewed per the SDD flow; W3's review is the deep one
(shapes, §6 order, kv_view discipline). W2||W1 can overlap; W3 needs both;
W4 needs W3.

**Parallel dispatch (Kabir 2026-07-19):** run the independent tier as
concurrent background workers, not a queue. Sequencing that makes it
sound: the manager starts the Q8_0 download (P1, wall-clock-long) FIRST,
then personally runs the two quick probes W2's brief depends on — P2
(Q8_0 byte layout) and P3 (rope/qwen3 source fetch; §5's rope formula is
only a hypothesis until P3) — then dispatches in parallel: **W1** (python
tooling + P4/P5 + ref-vs-oracle validation), **W2** (kernels + L0
vectors), and a third worker authoring the **§8 parity corpus + reserve**
(committed before any C lands, so it is parallel-safe by construction).
Rules: every brief pins an explicit file-ownership list (disjoint sets —
tooling under `tests/tooling/`, kernels under `src/kernels/` +
`tests/test_kernels.c`, corpus under `tests/data/parity/`); workers never
commit — the manager verifies each green bar and commits per-worker paths
sequentially. Same pattern forward: M3 kernel twins per quant kind, M4
sim harness ‖ kv.c.

## 12. Recorded at this gate (plan-amendment items owed by the gate reviewer)

- **`fatal()` vs the M6 embeddable ABI**: stoke currently has ~90 fatal/exit
  sites — correct for a CLI, impossible for a library that must not kill its
  host process. AMENDMENT (binding now, mechanism decided at M6 entry):
  (a) every abort keeps routing through the single `fatal()` choke point in
  common.h — no raw exit/abort anywhere else; (b) M2–M5 code keeps state in
  structs handed around explicitly (no new file-scope mutable globals), so a
  per-handle `stoke_open` can own everything later; (c) all allocation stays
  behind the x* wrappers (already true), making a tracked-allocation or trap
  based M6 mechanism a local change. The M6-entry note (Fable) picks the
  mechanism (leading candidate: thread-local trap context; alternatives
  weighed then, not now).
- **Windows-native `STOKE_REQUIRE_GOLDEN` half-gap** (carried from the M1
  review): on the reference box the Windows leg silently SKIPs golden gates
  if the model file is missing/moved; the WSL leg enforces via wsl-check.sh.
  Clean fix is a `scripts/win-check.cmd` counterpart — small, standalone;
  schedule as an early M2 chore (it will guard the new parity gate too, which
  makes it worth having before the gate exists).
- **Specials matching, known-unprovable revisit-condition (precision fix)**:
  the recorded condition said our leftmost-longest scan and llama.cpp's
  per-special sweep agree "unless a special is a substring of another." The
  correct condition is: unless one special is a substring of another OR two
  specials can overlap in input (a proper suffix of one equals a proper
  prefix of another). Qwen3's 26 and Llama-3.2's specials are all
  bracket-delimited with no suffix–prefix collisions, so both remain
  undiscriminable through M7 — but any future added-vocab check must test
  the two-sided condition, not the substring-only one.
