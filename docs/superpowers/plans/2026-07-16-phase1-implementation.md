# stoke Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the concurrent CPU serving core: measured baselines (M0), then GGUF+tokenizer (M1), scalar forward pass (M2), SIMD kernels (M3), paged KV (M4), continuous batching (M5), C API + server (M6), hardening (M7) — per the spec `../specs/2026-07-16-phase1-concurrent-cpu-core-design.md`.

**Architecture:** Fixed Llama-family forward pass in C11 (no tensor-graph engine); paged KV allocator + iteration-level scheduler as the wedge; scalar reference kernels as ground truth with AVX2/VNNI fast paths; own HTTP/SSE server; Python allowed only under `bench/`.

**Tech Stack:** C11 (MSYS2 ucrt64 gcc 15.2 on Windows, gcc via WSL for ASan/TSan), GNU Make, Python 3 for bench tooling only. Oracles: vendored llama.cpp b10037 (`bench/vendor/llama-b10037/`), Ollama 0.30.11.

## Global Constraints

- C11, zero runtime dependencies; Python appears ONLY under `bench/` and `tests/` tooling.
- Every access checked; loud failures (`fatal: <what> — <why> — <hint>` to stderr, nonzero exit); no silent fallbacks.
- Sanitizer matrix green at every commit: Windows native `mingw32-make test` + WSL `make SAN=asan test` + `make SAN=tsan test`.
- Scalar reference kernel exists for every SIMD kernel; SIMD is diffed against scalar in tests.
- No Claude attribution in commits (global setting enforces).
- Commit after every green task; never two makes in one BUILDDIR (AySen lesson: stale .o → bare "ld returned 5"; `rm -rf build/<plat>` on weird link errors).
- Reference model: `models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (bartowski). Reference box: i5-13450HX, 16GB DDR5-4800, Windows 11 + WSL2.
- Manager/worker protocol: Fable writes/reviews; mechanical implementation goes to sonnet workers with surgical briefs. Every brief includes (a) constants PROBED from real files/binaries at dispatch time — never recalled from memory, (b) a mechanical byte-level verification step the worker must run (this caught a shipped emitter bug in AySen pilot #3).

## File Structure (locked)

```
stoke/
  Makefile                    # plat= win|linux, SAN= asan|tsan, targets: all test bench clean
  src/common.h                # fixed-width types, fatal(), xmalloc, timing helpers
  src/gguf.{c,h}              # M1: GGUF v3 parse + mmap
  src/tok.{c,h}               # M1: byte-level BPE + ChatML/Llama-3 templates
  src/kernels/*.{c,h}         # M2/M3: per-quant dots + rmsnorm/rope/softmax/swiglu
  src/model.{c,h}             # M2: fixed forward pass
  src/kv.{c,h}                # M4: paged KV allocator
  src/sched.{c,h}             # M5: continuous batcher
  src/pool.{c,h}              # M3: C11-threads worker pool
  src/sample.{c,h}            # M2: greedy first, full set at M6
  src/api.c + src/stoke.h     # M6: C ABI
  src/server.{c,h}            # M6: HTTP/1.1 + SSE
  src/main.c                  # CLI: info|tokenize|run|serve|bench
  tests/                      # unit + golden + sim harnesses, run by make test
  bench/bench.py              # M0: harness core
  bench/adapters.py           # M0: ollama/llama-server/stoke launch+config
  bench/scenarios.py          # M0: S1-S4 definitions
  bench/results/              # committed reports (json+md); vendor/ gitignored
  scripts/wsl-check.sh        # full matrix, prints ALL GREEN
```

---

## M0 — measure the enemy (tasks 1–3, detailed)

### Task 1: bench harness core, verified against a mock server

**Files:**
- Create: `bench/bench.py`, `bench/scenarios.py`, `bench/mock_server.py`, `bench/test_harness.py`

**Interfaces (produces):**
- `run_scenario(base_url: str, scenario: Scenario, model: str) -> Report`
- `Scenario(name, n_clients, prompt_tokens, max_tokens, shared_prefix_tokens, mode)` — modes: `closed_loop` (S2) | `poisson` (S1)
- `Report` dict: `{aggregate_decode_tps, ttft_ms: {p50,p95}, itl_ms: {p50,p95}, per_client: [...], wall_s, completed, errors}`
- Scenarios locked: S1 interactive-mix (8 clients, ~512-tok prompts, 128 out, Poisson λ=0.5/s); S2 sweep N∈{1,2,4,8,16} closed-loop (512 in / 128 out); S3 agent-fanout (N=8, shared 1024-tok system prefix + 200-tok unique, 300 out); S4 single 4k-context sanity.

**Measurement core (the load-bearing code — timing happens at SSE-chunk receipt):**

```python
def timed_stream(base_url, model, messages, max_tokens):
    t0 = time.perf_counter(); token_times = []
    with httpx.stream("POST", f"{base_url}/v1/chat/completions", json={
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True}, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                token_times.append(time.perf_counter())
    if not token_times: raise RuntimeError("no tokens streamed")
    ttft = token_times[0] - t0
    itls = [b - a for a, b in zip(token_times, token_times[1:])]
    return ttft, itls, len(token_times), token_times[-1] - t0
```

Aggregate decode tok/s = Σ tokens after each client's first / (last token wall-time − first first-token wall-time) — decode-only, excludes prefill ramp; report also raw wall tok/s. Percentiles via `statistics.quantiles`. One SSE chunk ≈ one token for all three engines (verify in Task 2 against real servers; if an engine coalesces chunks, count via `usage`/`eval_count` fields instead — adapter concern).

- [ ] Step 1: write `mock_server.py` — stdlib `http.server`, SSE endpoint emitting T tokens at exactly D ms intervals after P ms initial delay (P, D, T from query params)
- [ ] Step 2: write `test_harness.py` — against the mock (P=200ms, D=50ms, T=20, N=4): asserts TTFT p50 ∈ [190, 280]ms, ITL p50 ∈ [45, 65]ms, token counts exact, aggregate math consistent (hand-computed expectation ±10%)
- [ ] Step 3: run `python bench/test_harness.py` → PASS (the measurement code is now itself tested — timing bugs here would poison every number we ever publish)
- [ ] Step 4: commit `bench: harness core with mock-verified timing`

### Task 2: engine adapters — four baseline configs, one GGUF

**Files:**
- Create: `bench/adapters.py`, `bench/Modelfile.qwen3-4b`

**Interfaces (produces):** `Adapter.start() -> base_url`, `.stop()`, `.label` for: `ollama-default`, `ollama-tuned` (`OLLAMA_NUM_PARALLEL=8` or 16 per run), `llamacpp-default`, `llamacpp-tuned` (`--parallel N -c <N*2048>`).

Key mechanics (verified at dispatch, expected forms):
- Ollama same-file import: `Modelfile` = `FROM ../models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`; `ollama create stoke-bench-qwen3 -f bench/Modelfile.qwen3-4b`; serve env: `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_KEEP_ALIVE=1h`; endpoint `http://127.0.0.1:11434/v1`.
- llama-server: `bench/vendor/llama-b10037/llama-server.exe -m models/…gguf --port 8089 -t <threads>` (+ tuned: `--parallel N -c N*2048`); endpoint `http://127.0.0.1:8089/v1`.
- Adapter healthcheck: poll `/v1/models` until 200 (60s timeout, loud fail).
- Warmup: 1 discarded request per client count before measuring; 3 measured runs; median reported; machine protocol: plugged in, High Performance power plan, close browsers (manual checklist printed by the harness).

- [ ] Step 1: write adapters + Modelfile; `ollama create` the bench model
- [ ] Step 2: smoke each adapter: 1 request through `timed_stream` per config, tokens > 0 (also validates the one-chunk≈one-token assumption per engine; fix counting per adapter if violated)
- [ ] Step 3: commit `bench: ollama + llama-server adapters, same-GGUF fairness`

### Task 3: run the baseline matrix, commit the report — M0 GATE

**Files:**
- Create: `bench/results/2026-07-XX-baseline-qwen3-4b.json` + `.md` (committed — results/ is un-gitignored for reports only, add `!results/*.md !results/*.json` to .gitignore)
- Modify: `MEMORY.md` (status log + anchored numbers)

- [ ] Step 1: full matrix: 4 configs × (S1, S2 sweep, S3, S4), 3 runs each, median
- [ ] Step 2: generate md report: table per scenario, aggregate-vs-N curve, TTFT/ITL percentiles
- [ ] Step 3: **premise check (kill-switch)**: compute each config's aggregate at N=8/16 as % of the ~160 tok/s physics ceiling. If any baseline ≥70% of ceiling, STOP — write findings to MEMORY.md and reconvene with Kabir before any kernel work. Expected per research: Ollama-default far below (serialized); tuned llama-server the real bar.
- [ ] Step 4: re-anchor spec targets from measured numbers (edit spec "Physics budget" with measured column; targets stay ≥2× Ollama-default, ≥1.0× llama-server-tuned at 8/16)
- [ ] Step 5: commit `bench: M0 baselines on reference box — premise confirmed` + MEMORY.md status update

## M1 — gguf.c + tok.c (tasks 4–7, detailed)

### Task 4: build skeleton — `make test` green on both platforms

**Files:**
- Create: `Makefile`, `src/common.h`, `tests/test_common.c`, `scripts/wsl-check.sh`, `tests/run_tests.sh`

**Interfaces (produces):** `fatal(fmt, ...)` (never returns, prints `fatal: …` to stderr, exit 1); `u8/u16/u32/u64/i8..i64/f32/f64` typedefs; `xmalloc/xcalloc/xrealloc` (fatal on OOM); `now_ms()` monotonic; `TEST(name)` + `ASSERT_*` macros in `tests/test.h`; Makefile pattern mirrors AySen: `BUILDDIR=build/$(PLAT)$(SAN_SUFFIX)`, `make test` builds+runs every `tests/test_*.c` binary plus script gates.

- [ ] Step 1: write test `tests/test_common.c` — xmalloc returns writable memory; fatal() exits nonzero with `fatal:` prefix (spawn self via system(), check exit code + stderr)
- [ ] Step 2: implement `common.h` + Makefile + runner; `mingw32-make test` → PASS
- [ ] Step 3: WSL: `make test`, `make SAN=asan test`, `make SAN=tsan test` → PASS (wsl-check.sh wraps all + prints ALL GREEN)
- [ ] Step 4: commit `build: C11 skeleton, test runner, sanitizer matrix`

### Task 5: gguf.c — parse + mmap + `stoke info`

**Files:**
- Create: `src/gguf.{c,h}`, `src/main.c` (info subcommand only), `tests/test_gguf.c`, `tests/data/` (tiny handcrafted GGUF fixtures, committed)

**Interfaces (produces):**
```c
typedef struct GgufFile GgufFile;             // opaque; owns mmap
GgufFile* gguf_open(const char* path);        // fatal on malformed; NULL never returned
void      gguf_close(GgufFile*);
// metadata: typed getters, fatal on wrong-type, _or variants for defaults
i64         gguf_meta_i64(const GgufFile*, const char* key);
const char* gguf_meta_str(const GgufFile*, const char* key);
bool        gguf_meta_has(const GgufFile*, const char* key);
// string-array + f32-array getters for tokenizer tables:
u64         gguf_meta_arr_len(const GgufFile*, const char* key);
const char* gguf_meta_arr_str(const GgufFile*, const char* key, u64 i);
f32         gguf_meta_arr_f32(const GgufFile*, const char* key, u64 i);
typedef struct { const char* name; u32 type; u32 n_dims; u64 dims[4];
                 const void* data; u64 nbytes; } GgufTensor;
u64               gguf_tensor_count(const GgufFile*);
const GgufTensor* gguf_tensor(const GgufFile*, u64 i);          // by index
const GgufTensor* gguf_tensor_by_name(const GgufFile*, const char* name); // fatal if missing
```
Worker brief carries: GGUF v3 layout probed from the real file via `xxd` (magic `GGUF`, version, counts, value-type enum incl. arrays, alignment key `general.alignment` default 32, tensor-data offset alignment) cross-checked against llama.cpp's `gguf.h` in the vendor zip source… vendor zip is binaries-only, so probe constants from bytes + the official GGUF spec fetched at dispatch; verification step: worker must dump first 256 bytes and annotate every field by hand in the brief reply.

- [ ] Step 1: tests first — fixtures: minimal valid GGUF (2 KV, 1 tensor) handcrafted by Python script `tests/data/make_fixtures.py`; negative fixtures: bad magic, truncated header, count overflow, misaligned tensor offset, string len > file size → each must `fatal` (spawn-self pattern from Task 4)
- [ ] Step 2: implement gguf.c until `mingw32-make test` green; every read bounds-checked against file size before dereference (mmap + explicit length checks; no trusting counts)
- [ ] Step 3: `stoke info models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` — golden-lock the output (arch, n_layers=36, heads, kv-heads=8, vocab size, rope theta, 398 tensors w/ shapes/types expected — exact values read from the real file, golden committed after hand-verification against unsloth/Qwen model card)
- [ ] Step 4: WSL matrix green; commit `gguf: v3 parser + mmap + stoke info, fuzz-negative gated`

### Task 6: tok.c — byte-level BPE, oracle-gated

**Files:**
- Create: `src/tok.{c,h}`, `tests/test_tok.c`, `tests/tok_fuzz.py`, `tests/data/tok_corpus.txt`

**Interfaces (produces):**
```c
typedef struct Tok Tok;
Tok* tok_from_gguf(const GgufFile*);      // reads tokenizer.ggml.* metadata tables
void tok_free(Tok*);
// encode: returns count, writes ids; fatal on cap overflow (caller sizes = bytes+8)
u32  tok_encode(const Tok*, const char* utf8, u64 len, u32* ids, u32 cap, bool add_special);
// decode one id into buf (token pieces can be partial UTF-8); returns bytes written
u32  tok_decode(const Tok*, u32 id, char* buf, u32 cap);
u32  tok_bos(const Tok*); u32 tok_eos(const Tok*);
// chat template: ChatML (Qwen3) hardcoded; Llama-3 header format at M7
u32  tok_apply_chatml(const Tok*, const StokeMsg* msgs, u32 n, u32* ids, u32 cap);
```
Algorithm: byte-level BPE with merge ranks from `tokenizer.ggml.merges`, token table `tokenizer.ggml.tokens`, special-token handling per `tokenizer.ggml.*` flags — exact key names probed from the real file at dispatch (Task 5's `stoke info` dumps them). GPT-2 byte↦unicode remap table generated at init, not hardcoded.

- [ ] Step 1: corpus: `tok_fuzz.py` writes ~2000 lines — ASCII, code, emoji+ZWJ, CJK, Devanagari, Arabic, whitespace runs, control chars, 10KB line, empty line, lone surrogates as bytes
- [ ] Step 2: oracle run: `bench/vendor/llama-b10037/llama-tokenize.exe -m models/…gguf -f tests/data/tok_corpus.txt --ids` → golden file (exact flag names verified at dispatch via `--help`)
- [ ] Step 3: test drives `tok_encode` over the corpus, byte-exact vs golden; decode(encode(x)) == x for every line
- [ ] Step 4: implement until green, Windows + WSL matrix
- [ ] Step 5: commit `tok: byte-level BPE + ChatML, byte-exact vs llama-tokenize on 2k-line fuzz corpus`

### Task 7: M1 close-out

- [ ] Step 1: `stoke tokenize <file>` CLI subcommand (prints ids; used by future debugging)
- [ ] Step 2: full matrix `scripts/wsl-check.sh` → ALL GREEN
- [ ] Step 3: MEMORY.md status log update; tag `m1`
- [ ] Step 4: commit `m1: front door complete — gguf + tokenizer oracle-gated`

---

## M2–M7 — roadmap level (detailed tasks written at each milestone entry, after the prior gate's data)

- **M2 scalar forward pass** (entry: M1 tag): f32/q8_0 dequant-dot scalar kernels; model.c fixed pass (RMSNorm→QKV(+QK-norm)→RoPE→attn→SwiGLU); naive contiguous KV; `stoke run` single prompt; GATE: greedy 50-prompt token parity vs `llama-cli -m …gguf --temp 0 -n 64`. Design note: attention over [1..t] written against an abstract `kv_view(seq, layer)` accessor from day 1 so M4's paged tables slot in without touching math.
- **M3 SIMD + pool** (entry: M2 parity): AVX2 q8_0/q4_0 dots (+VNNI path), pool.c row-partitioned matmul, thread pinning experiment (P vs E cores), q4_k probed+implemented; GATE: exact/tolerance diff vs scalar on random tensors ×1e4, single-stream ≥50% llama.cpp same file/box; bench snapshot into history.jsonl.
- **M4 paged KV + static batch** (entry: M3 gate): kv.c block pool + block tables + refcount/CoW + Q8_0 KV; batched decode over fixed N; GATE: block-accounting exact under randomized alloc/free sim ×1e6 (ASan), aggregate scaling curve N∈{1..16} measured and monotone, prefix-sharing dedup verified byte-level.
- **M5 continuous batching** (entry: M4 curve): sched.c admission/watermark/preempt-recompute + chunked prefill (128/step); scheduler SIMULATION harness first (fake instant model, deterministic seeds): starvation-free, watermarks honored, preempt lossless; then real: GATE = S2 ≥2× ollama-default AND ≥1.0× llama-server-tuned at N=8/16; ITL stall bound <250ms during co-running prefill; 1h randomized soak ASan+TSan.
- **M6 API freeze + server** (entry: M5 gate): stoke.h final (open/submit/poll/cancel/metrics/close, thread-safe submit), server.c HTTP/1.1+SSE, OpenAI endpoints, `stoke serve`; GATE: openai-python client suite + curl goldens; API doc in README.
- **M7 launch-ready** (entry: M6): Llama-3.2-3B + its template, unsloth-file parity run, tok fuzz expansion, 24h soak, README + reproducible benchmark report, name-collision decision (Kabir), Phase-1.5 planning doc (CUDA/ARM).

## Self-review notes (done at write time)

- Spec coverage: every spec criterion maps — #1→M2/T6, #2→M5+T3, #3→M3, #4→M5, #5→M6, #6→M6, #7→T4-onward gates. Landscape/premise → T3 kill-switch.
- No placeholders: dispatch-time probing is explicitly the protocol (constants named per task), not deferred content.
- Type consistency: GgufFile/GgufTensor/Tok signatures used by T6/M2 notes match T5 interfaces.
