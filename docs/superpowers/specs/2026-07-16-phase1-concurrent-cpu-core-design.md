# stoke Phase 1 — the concurrent CPU serving core — Design

*2026-07-16. First spec. Parent record: `../../../MEMORY.md` (direction, wedge,
AySen alignment). Landscape background: `../../vllm-and-the-serving-landscape.md`.
Name locked 2026-07-16: **stoke** — stoking = feeding the fire to keep it
saturated, which is literally what continuous batching does.*

## What stoke is

**A small, embeddable, from-scratch LLM inference engine whose architecture is
concurrent-first** — vLLM's two core ideas (paged KV cache + continuous
batching) engineered for the bandwidth-bound consumer hardware that llama.cpp
and Ollama own. Light *and* concurrent: the empty quadrant.

**Library-first**: the product is `libstoke.a` + `stoke.h` (C ABI) — "the
SQLite of LLM inference." The `stoke` CLI binary (serve/run/bench) is a thin
in-tree frontend. The OpenAI-compatible server makes it useful to people who
have never heard of the rest of Kabir's stack; AySen's `local()` client links
it in-process at Phase 3.

## Phase-1 goal & success criteria (all measurable, on the reference box)

Reference box: i5-13450HX (6P+4E/16T, AVX2+AVX-VNNI, no AVX-512), 16GB
DDR5-4800 dual-channel (~77 GB/s theoretical), Windows 11 + WSL2. This is
deliberately the market's median machine.

1. **Correctness**: greedy-decode token parity vs llama.cpp (`llama-cli`) over
   a 50-prompt corpus — Qwen3-4B-Instruct-2507 at Q8_0, Q4_0, Q4_K_M, and
   Llama-3.2-3B-Instruct at Q4_K_M. Tokenizer: byte-exact vs `llama-tokenize`
   on a fuzz corpus.
2. **The headline**: aggregate decode throughput **≥2× Ollama** (stretch 3×) at
   8 and 16 concurrent streams (scenario S2), same GGUF file, measured by the
   in-repo harness. This is the launch number (defaults are what users run —
   `OLLAMA_NUM_PARALLEL` defaults to 1 per official docs). The honest
   *engineering* bar tracked alongside: **tuned llama-server**
   (`--parallel 8/16`) — beat or match it at 8–16 concurrent, win beyond 16
   where community consensus says it degrades.
3. **Sanity floor**: single-stream decode ≥50% of llama.cpp on the same
   file/box. (Single-stream parity is explicitly NOT a goal — the aggregate
   headline is.)
4. **Latency under load**: with chunked prefill working, no decode stall
   >250ms caused by another request's prefill; TTFT p95 reported per scenario.
5. **Footprint**: one static binary, zero runtime deps; `libstoke.a` +
   `stoke.h` embeddable; cold start (open model, mmap) <2s.
6. **API compat**: `/v1/chat/completions` (stream + non-stream),
   `/v1/completions`, `/v1/models` work against openai-python and curl.
7. **Discipline**: full suite green under ASan+TSan (WSL) at every commit;
   scheduler soak test; bench snapshot appended per green commit.

## Non-goals (Phase 1)

CUDA (Phase 1.5, onto the proven scheduler), NPU (Phase 2; note llama.cpp now
ships an OpenVINO build — the wasteland is being noticed), MoE models (wants
≥24GB boxes), LoRA, speculative decoding, multi-model serving, batch-invariant
determinism (documented, not promised), general Jinja chat templates (two
built-ins: ChatML for Qwen3, Llama-3 header format), prefix-cache persistence,
auth/multi-tenancy, non-x86-64 (ARM is Phase 1.5+ with CUDA).

## Physics budget (pre-M0 estimates; M0 re-anchors all numbers)

Decode is memory-bandwidth-bound: each token-step streams the weights once,
amortized across the whole batch, plus each sequence's KV.

- Weights: Qwen3-4B Q4_K_M ≈ 2.4GB. KV (Qwen3-4B GQA: 36 layers × 8 KV-heads
  × 128 dim): ~147 KB/token fp16, **~74 KB/token Q8_0**.
- Single stream @ ~60 GB/s real: ceiling ≈ 60/2.4 ≈ **25 tok/s** (llama.cpp
  real-world will be lower; M0 measures).
- Batch 8 @ ~1k ctx each: step bytes ≈ 2.4GB + 8×74MB ≈ 3.0GB → ~20 steps/s →
  **~160 tok/s aggregate ceiling**. Compute check (AVX2+VNNI, ~1.3 TFLOPS
  class): batch-8 step ≈ 64 GFLOP ≈ 30ms<50ms bandwidth time → bandwidth still
  binds. Real-world target: 80–120 aggregate.
- Ollama at 8 concurrent on this box: expected 15–30 aggregate (M0 measures).
- **Kill-switch**: if M0 shows Ollama/llama-server already near the physics
  ceiling at concurrency, the premise is wrong — stop and re-examine BEFORE
  writing kernels. Cheap insurance.

KV pool at 16GB RAM: model 2.4GB + pool ~6GB ≈ 85k pooled Q8 tokens ≈ 16
streams × 5k ctx. On CPU, KV capacity is comfortable; bandwidth is the fight.
(On GPU later it inverts — paged KV earns its keep twice.)

## Architecture

Units, each with one purpose, standalone-testable (AySen isolation
discipline):

```
src/gguf.c        GGUF v3 parser: header, metadata KV, tensor index, mmap weights
src/tok.c         byte-level BPE (tiktoken-style); chat templates: ChatML + Llama-3 (hardcoded pair)
src/kernels/      matmul/dot per quant (q8_0.c q4_0.c q4_k.c f16.c) + rmsnorm/rope/softmax/swiglu
                  EVERY kernel: scalar reference + AVX2/VNNI fast path, runtime-dispatched
src/model.c       fixed Llama-family forward pass; Qwen3 delta = per-head QK-norm. NOT a graph engine.
src/kv.c          paged KV: 16-token blocks (tunable), per-seq block tables, refcount + copy-on-write
                  tail blocks (shared-prefix reuse), Q8_0 KV quantization option
src/sched.c       continuous batcher: iteration-level loop; FCFS admission gated on free-block
                  watermark; chunked prefill; preempt-youngest + recompute on pool exhaustion
src/pool.c        C11-threads worker pool, row-partitioned kernel fan-out
src/sample.c      greedy / temperature / top-k / top-p / repeat-penalty / stop strings / seed
src/api.c+stoke.h C ABI: stoke_open/submit/poll/cancel/metrics/close; submit thread-safe
src/server.c      own minimal HTTP/1.1 + SSE (no deps); OpenAI endpoints; binds localhost by default
src/main.c        CLI: stoke serve | run | bench | tokenize | info
bench/            Python harness (offline tooling only — the runtime stays pure C):
                  drives ollama / llama-server / stoke with identical workloads
tests/            kernel units (vs scalar refs), tokenizer fuzz (vs llama-tokenize oracle),
                  greedy-parity corpus, scheduler simulation, soak; ASan/TSan matrix via WSL
```

Data flow: server threads (or embedding app threads) → MPSC submit queue →
**scheduler thread** owns the step loop — each step: admit waiting requests if
watermark allows; build the step batch = all runnable decodes + **at most one
prefill chunk**; fan out ops to the pool; sample; push tokens to per-request
output rings → SSE writers / `stoke_poll` / callbacks. One process, no IPC.

## Design decisions (binding)

1. **C11**, AySen discipline: sanitizer matrix every commit, loud failures,
   every access checked, goldens lock behavior. CUDA later lives in `.cu`
   behind `extern "C"` — the core never becomes C++.
2. **Fixed forward pass, not a tensor-graph engine.** Generality is ggml's
   trap; one architecture family, hand-scheduled, is faster to build, easier
   to reason about, and where all target models actually live.
3. **Quant ladder**: bring-up F32 → Q8_0 → Q4_0; **Q4_K_M before any published
   benchmark** (it is what people actually download). K-quant super-block
   layout gets probed from real files at implementation time, never recalled
   from memory (manager/worker rule).
4. **Paged KV is non-negotiable** — it is the wedge. Block = 16 tokens default
   (measure 8/32 at M4). Q8_0 KV option halves decode bytes; default ON for
   the benchmark once parity-proven.
5. **Chunked prefill is mandatory, not an optimization** — the CPU-specific
   insight. A 2k-token prefill on this class of CPU is seconds of compute; run
   whole, it stalls every other user's stream. Chunk = 128 tokens default, at
   most one chunk per step, so decode ITL stays bounded (criterion #4).
6. **Scheduler policy Phase 1**: FCFS admission + free-block watermark;
   preempt-youngest-then-recompute on exhaustion; no priority classes yet.
   Deterministic given fixed arrivals (simulation-testable).
7. **Prefix sharing**: block-granular refcounting; identical prefixes (system
   prompts, agent fan-out) share physical blocks; copy-on-write on the tail
   block at divergence.
8. **Own HTTP server** (~few hundred lines, HTTP/1.1 + SSE, localhost
   default). AySen's http module experience carries directly; zero deps.
9. **Tokenizer is oracle-tested, not spec-trusted**: fuzz corpus (random
   unicode, code, emoji, whitespace pathologies, CJK/Indic scripts) must match
   `llama-tokenize` byte-exactly per model.
10. **Windows + Linux from day 1** — MSYS2 ucrt64 + WSL gcc, the proven AySen
    matrix pattern (sanitizers live on the WSL side).
11. **Models**: Qwen3-4B-Instruct-2507 primary (Apache-2.0, ungated, exactly
    the class the box serves well); Llama-3.2-3B-Instruct secondary (proves
    family generality: different vocab, RoPE theta, no QK-norm).
12. **Benchmark fairness**: all engines serve the *same GGUF file* (Ollama via
    `Modelfile FROM`), same sampling, same workloads, 3 runs, median, quiesced
    machine, methodology + scripts published in-repo.
13. `stoke.h` is UNSTABLE until M6 freeze.

## Correctness strategy

- **Scalar reference = ground truth** for every kernel; SIMD paths exact-diffed
  (integer paths) or tolerance-diffed (float) against it, per-op unit tests.
- **Cross-engine parity**: greedy token-sequence match vs `llama-cli` over the
  corpus (logit-level comparison is too brittle across differing quant-dot
  implementations; token-level greedy over 50 diverse prompts is robust).
- **Scheduler simulation**: fake instant model; assert invariants — exact
  block accounting (alloc == free at drain), no starvation, watermarks
  honored, preempt/resume round-trip lossless, chunked-prefill interleave
  bounds decode gaps.
- **Determinism**: fixed seed + fixed batch composition → identical output.
  Batch-invariance (same output regardless of co-batched requests) is NOT
  promised — floating-point reduction order varies with batch shape; documented.
- **Sanitizers + soak**: ASan/TSan matrix per commit (WSL); N-hour soak with
  randomized arrivals before M5 sign-off.
- **Perf regression tracking**: harness snapshot appended to
  `bench/history.jsonl` at every green commit touching hot paths.

## Landscape (verified 2026-07-16, web research — supersedes memory)

The quadrant is confirmed empty, with sharper edges than the original scan:

- **mistral.rs** (v0.9.0, ~7.5k stars, very active): ships paged attention +
  continuous batching — but Rust/Python/server only (no C ABI, not
  embeddable), GPU-first; measured CPU datapoint: **17 tok/s vs llama.cpp's
  124** on the same settings (GH issue #763). Has the architecture, lacks CPU
  quality and embeddability.
- **exllamaV2/V3 + TabbyAPI**: real paged/dynamic batching — NVIDIA-only,
  Python-only. V2 archived; V3 active.
- **MLC-LLM** (~2k stars): continuous batching + paged KV via TVM-compiled
  kernels — but TVM compile pipeline per target, small community, mobile needs
  AOT builds. Has the architecture, lacks simplicity.
- **llama.cpp llama-server**: `--cont-batching` default-on, `--parallel`
  slots; no fundamental scheduler redesign found in 2025–26. Community
  consensus: workable to ~8–16 concurrent with manual tuning, degrades beyond,
  cited 2–10× behind vLLM at concurrency. Has the footprint, lacks the
  architecture.
- **Ollama** (0.30.x): `OLLAMA_NUM_PARALLEL` default **1**; now runs a
  three-engine strategy (own Go engine for multimodal, MLX on Apple, vendored
  llama.cpp for GGUF) — an existence proof that Ollama-class frontends swap
  engines underneath, which is exactly stoke's library-first insertion point.
- llama.cpp now ships an **OpenVINO build** — the NPU wasteland is being
  noticed; keeps Phase-2 urgency honest.

Positioning confirmed: every neighbor that has the concurrent architecture
gave up CPU quality, embeddability, or simplicity. Stoke's cell — concurrent-
first architecture, first-class CPU, C-ABI embeddable, zero-dep single binary
— has no occupant.

## Risks

- **Kernel gap vs llama.cpp's years of tuning** → aggregate-not-single-stream
  target absorbs it; VNNI; measure early (M3 gate at ≥50%).
- **Tokenizer rabbit hole** → oracle + fuzz + exactly two chat templates.
- **Scheduler complexity** → simulation-first TDD; the sim harness is written
  before the scheduler.
- **llama.cpp ships a real scheduler upstream** → speed, and the library-first
  category they can't easily pivot a stable single-user API into.
- **Premise risk** → M0 kill-switch (see Physics budget).
- **Solo bandwidth** → manager/worker delegation: Fable designs/reviews,
  cheaper models type mechanical code from surgical briefs with probed
  constants + mechanical verification steps.

## Milestones

- **M0 — measure the enemy.** Bench harness (scenarios S1 interactive-mix, S2
  closed-loop concurrency sweep N=1..16, S3 agent-fanout with shared 1k
  prefix, S4 4k-context sanity) against four baseline configs — Ollama
  default, Ollama tuned (`NUM_PARALLEL=8/16`), llama-server default,
  llama-server tuned (`--parallel 8/16`) — same GGUF file everywhere (Ollama
  via `Modelfile FROM`), report committed. GATE: anchors all targets; premise
  check.
- **M1 — gguf.c + tok.c**, oracle-gated (`stoke info` dumps metadata/tensors;
  `stoke tokenize` byte-exact vs llama-tokenize on fuzz corpus).
- **M2 — scalar forward pass** (F32 + Q8_0), single sequence, naive contiguous
  KV: greedy parity on the corpus. Math proven before perf work.
- **M3 — AVX2+VNNI kernels + thread pool**: ≥50% llama.cpp single-stream;
  exact-diff vs scalar refs. Q4_0 joins.
- **M4 — paged KV + static-batch decode** (fixed batch, no admission):
  aggregate scaling curve measured; prefix sharing on; physics validated
  before scheduler complexity.
- **M5 — continuous batching + chunked prefill**: THE gate — S2 aggregate
  ≥2× Ollama @ 8 and 16; TTFT/ITL bounds hold. Product moment.
- **M6 — `stoke.h` freeze + server**: OpenAI endpoints + SSE; openai-python
  smoke; `stoke serve` end-to-end.
- **M7 — harden + launch-ready**: Q4_K_M, Llama-3.2-3B, fuzz, soak, README
  with reproducible benchmark report.

## Open items (deliberate)

- Q4_K super-block dot kernel design — probe layout from real files at M3.
- Thread pinning (P-cores vs E-cores split) — measure at M3.
- Block size default — measure 8/16/32 at M4.
- Preemption policy refinement (recompute vs KV-spill-to-disk) — after M5 data.
- ARM NEON + CUDA backends — Phase 1.5 planning after M5.
- **Name collision watch**: PyPI `stoke` is taken (Fidelity ML training lib);
  StanfordPL/stoke is a known x86 superoptimizer; Stoke Space dominates search.
  GitHub/crates.io are clean. Rename cost ≈ 0 until public launch — revisit at
  M7 (Kabir's call).
- Benchmark file: we develop against bartowski's Q4_K_M; unsloth's is the
  most-downloaded (~87.5k/mo vs ~11k) — add an unsloth-file parity run at M7
  so the published numbers use the file people actually have.
