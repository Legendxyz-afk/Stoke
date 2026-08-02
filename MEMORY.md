# The Local AI Engine — Direction & AySen Alignment

*Project memory / north-star doc — 2026-07-14; updated 2026-07-16. Read this
before writing any spec or code for the engine. This folder
(`Programming/stoke/` — renamed from `engine/`) is the home of the current
build: **the engine is named stoke** (locked 2026-07-16; collision caveats in
Status). This file is the complete master record: the decision,
the technical wedge, the AySen alignment, and — in the appendices — the full
asset inventory and the every-path business journey that led here. Ground-up
teaching lives in `docs/beginners-primer.md`; the vLLM teardown in
`docs/vllm-and-the-serving-landscape.md`.*

## The one-line

A **high-throughput, concurrent, local-first LLM inference engine for the
hardware vLLM abandons** — CPU, Apple Silicon, AMD, consumer NVIDIA today;
NPUs later. Open-source core, open-core company much later. Strategically:
**the execution layer that sits beneath AySen.**

---

## Why this (the decision, recorded)

Kabir's goal, stated 2026-07-14: **not money right now** — build something
*proper and useful*, low-level, and a bottomless place to learn. That single
constraint change flips the selection criteria away from "easy to sell" and
toward "worth building for years."

The filter every path was run through:

- **Moat = hard engineering.** Pick something whose defense is difficulty, not
  marketing — the one axis where a single elite engineer beats funded teams
  (llama.cpp's core is essentially one person). This is Kabir's edge.
- **Distribution = open source.** For deep dev/infra tools, OSS *is* the
  go-to-market. This deletes the distribution weakness instead of fighting it.
- **Fuel = learning.** A bottomless systems frontier (kernels, quantization,
  scheduling, memory) so "eager to learn" is the engine, not a footnote.
- **Horizon = long.** No burn, no deadline — spend that on something with a
  payoff most founders can't stomach waiting for.

Chosen frontier (Kabir, 2026-07-14): **low-level systems. Graphics secondary.**

What was explicitly *retracted*: a 3D/WebGL component shop (great money
business, but shallow for this person, small ceiling, moat = marketing = the
weak axis). Retired.

---

## What it is (technical)

**The problem it solves — the concurrency cliff on non-datacenter hardware:**

- **vLLM**: ~793 tok/s under concurrent load — the throughput king, but a
  **heavyweight Python+PyTorch, datacenter-GPU-shaped** system. Its V1 rewrite
  added CPU/AMD/Intel/TPU backends, yet it stays complex, VRAM/RAM-hungry, and
  **not embeddable or tuned for one-box consumer/edge serving.**
- **Ollama**: ~41 tok/s — a **19x** gap — and effectively single-user; poor at
  many simultaneous requests.
- **llama.cpp**: maximally portable across hardware, but a library with "no
  management layer" — single-user-shaped, every knob manual.

So the middle is empty: **no lightweight, embeddable engine serves many
users/requests *well* on the hardware most people actually own.** vLLM is heavy
and server-shaped ("stand up a GPU box"); Ollama is light but "one user at a
time." The space between — light *and* concurrent — is the wedge. (Full
teardown: `docs/vllm-and-the-serving-landscape.md`.)

**The hard core (the fun part):** continuous batching + a paged KV-cache + a
real request scheduler — the techniques that give vLLM its throughput — but
engineered for consumer/non-NVIDIA hardware instead of H100s.

**Who it's for:** anyone self-hosting AI for a team or an app on one good
machine — privacy/on-prem shops (health, legal, finance, government),
cost-sensitive builders (India / Global South / indie), edge boxes serving
multiple requests.

**Honest ceiling:** consumer hardware is **memory-bandwidth-bound**. Expect
~2–4x over Ollama at concurrency, not vLLM's 19x. Extracting maximum throughput
from bandwidth-limited hardware *is* the craft — it's the whole game, and 2–4x
is still transformative for the target user.

---

## The validated gap (2026 research, not memory)

- "The hard part is no longer making models run — it is choosing among
  excellent options... no single engine is optimal for all use cases; the same
  model needs a different engine when user count, latency, or hardware tier
  changes." (multiple 2026 comparisons)
- vLLM ~793 vs Ollama ~41 tok/s at peak concurrency (a 19x gap that "lives
  entirely at peak concurrency"). vLLM V1 *does* run on CPU/AMD/Intel/TPU now —
  but as a heavyweight, server-shaped port, not a lightweight embeddable engine.
- llama.cpp "trades production serving capabilities" for portability; "no
  management layer."
- **NPUs are a wasteland of idle silicon:** every 2026 Windows laptop ships a
  40–80 TOPS NPU, yet "most local LLM stacks still use GPU/CPU rather than
  NPUs"; "the NPU's killer use cases are still being defined"; memory bandwidth
  is the real bottleneck; Intel OpenVINO is the most mature vendor stack.

Sources: Red Hat (llamacpp-vs-vllm, 2026-06), BIZON inference-engine
comparison, local-llm.net 2026 guide, localaimaster NPU comparison 2026,
Microsoft Copilot+ NPU dev guide, daily.dev running-LLMs-locally.

---

## Honest risks (the stress-test)

- **"vLLM could just own the consumer niche."** It already added the backends
  (CPU/AMD/Intel/TPU) — but not a lightweight, embeddable, consumer-tuned
  engine; that cuts against its datacenter DNA and heavy Python/PyTorch core.
  The real risk is a funded team pivoting; the defense is footprint, focus, speed.
- **Physics ceiling.** Bandwidth-bound hardware caps the win (see above). Real,
  and named up front.
- **Incumbents are smart.** This is a bet on out-executing on a neglected axis.
  Not guaranteed — call it ~70% conviction, a genuine bet.
- **Focus risk.** Three big systems projects now exist (AySen, the router, this
  engine). Discipline: **the engine is the one current build.** AySen and the
  router are existing assets that *snap in later* (Phase 3), not parallel
  workstreams.

---

## Where it aligns with AySen (the reason this is coherent, not random)

The stack Kabir is actually building, top to bottom:

```
  AySen (.as)      — the command seat / orchestration layer      [EXISTS]
     |  client / train()  (typed contracts, retry, fallback, parallel fan-out)
     v
  Router (omni)    — routing, semantic cache, compression, cloud fallback [EXISTS]
     |
     v
  THE ENGINE       — continuous batching, paged KV-cache, scheduler, kernels [TO BUILD]
     |
     v
  Hardware         — CPU / GPU / NPU
```

1. **AySen's locked positioning already carves out exactly this layer.** The
   handbook says, verbatim: AySen is *"not replacing PyTorch/CUDA kernels. It
   is the command seat / orchestration layer."* The engine **is** the
   kernel/execution layer AySen deliberately delegates. Non-overlapping *by
   design* — Kabir's own locked vision anticipated this missing piece months
   ago. This bet is the bottom half of a stack whose top half already exists.

2. **The concrete integration point: a `local` client.** AySen clients today
   are `anthropic()`, `openai()`, `mock()`. Add `local(...)` backed by the
   engine → AySen programs run **fully offline, private, free**. A
   general-purpose language with AI orchestration in its *syntax*, running
   entirely on your own machine, is a genuinely novel combination nobody ships.

3. **AySen is the perfect dogfood — and stress test.** AySen's `parallel` /
   strands fan-out issues N concurrent agent calls at once. That is *exactly*
   the concurrent load the engine's continuous batching + scheduler is built to
   eat. The language **generates** the concurrency the engine is designed to
   serve. Best possible first workload; a built-in torture test.

4. **Reliability at two layers, reinforcing.** AySen enforces typed contracts +
   retry/fallback at the *language* level. The engine + router provide fallback
   at the *infrastructure* level (local → cloud spill). A local model whose
   output fails a contract → AySen retries; hardware that can't cope → router
   spills to cloud; the typed contract guarantees the final shape regardless.
   That combined behavior *is* "Phase 3 — the runtime," with AySen as the
   orchestration front-end.

5. **`train()`'s pluggable-engine protocol.** AySen's `train()` is already "a
   uniform front door over pluggable engines" (`AYSEN_TRAIN_BACKEND`). The
   engine registers as an inference/serving backend under the same philosophy —
   the plug already exists.

6. **Shared systems DNA.** Both are C-level. AySen is being self-hosted in
   **AS-core**, which now has fixed-width types, `*T` pointers, `[]T` slices,
   `alloc/free`, bit ops, and `extern "C" fn` FFI — a real systems language.
   Long-shot but tantalizing: the engine's *control plane* (scheduler, request
   queue, KV-cache bookkeeping) could one day be written in AS-core calling
   C/CUDA/SIMD kernels through `extern "C"` — dogfooding AySen as a systems
   language. Speculative, Phase-4+. The hot kernels stay CUDA/SIMD.

7. **Unified company thesis.** Language (AySen) + router + engine =
   **the local-first, reliable AI stack**: write reliable AI programs that run
   fast on your own hardware — private, offline, spilling to cloud only when you
   choose. **Nobody owns "local-first reliable AI" end-to-end.** That is the
   defensible, ambitious, coherent company story, assembled from parts that
   already mostly exist.

---

## Guardrails (so the alignment helps instead of trapping us)

- **The engine must stand alone.** It has to be useful to people who have never
  heard of AySen — an **OpenAI-compatible API**, like the router. AySen is *a*
  client, never *the* client, or the engine's market collapses to AySen's tiny
  userbase.
- **Don't let AySen's needs distort the engine's general design.** AySen is the
  flagship demo + dogfood, not the spec.
- **Sequence, don't parallelize.** Engine first (standalone, useful) → AySen
  `local` backend as the integration demo → the full runtime. AySen never
  blocks the engine.

---

## Phased roadmap

1. **Phase 1 — the concurrent serving core**, on Kabir's own hardware. Useful,
   standalone, the day it beats Ollama at concurrency. ← *spec this first.*
2. **Phase 2 — heterogeneous + NPU execution.** Schedule across CPU/GPU/NPU;
   light up the idle 40-TOPS silicon. The bleeding-edge moat, once the core
   exists (and once the hardware is on hand).
3. **Phase 3 — the runtime.** Fold in the router (cloud-spill fallback) + AySen
   (`local` client, contracts). Now it is not just serving — it is a *reliable,
   local-first* runtime.
4. **Phase 4 (optional, far off) — the company.** Open-core: hosted control
   plane, enterprise on-prem, support.

---

## Open questions — ALL ANSWERED 2026-07-16 (Kabir + probe)

- **First hardware target** → Kabir's box: i5-13450HX (6P+4E, AVX2+AVX-VNNI,
  no AVX-512), 16GB DDR5-4800 (~77 GB/s theor.), RTX 3050 Laptop 6GB (168 GB/s,
  CC 8.6) reserved for Phase 1.5. **No NPU on hand** (HX predates them) —
  Phase 2 needs future hardware. CPU-first, CUDA second.
- **Model / weight format** → GGUF. Qwen3-4B-Instruct-2507 primary (Apache,
  ungated), Llama-3.2-3B-Instruct secondary (family generality).
- **Engine language** → **C11, from scratch** (no ggml — own kernels, own
  paged-KV attention, own scheduler; single-stream parity NOT required, the
  aggregate concurrency headline is the target). CUDA later in `.cu` behind
  `extern "C"`. AS-core control plane stays a Phase-4 dream via the C seam.
- **Name** → **stoke** (Kabir, 2026-07-16). Caveats to revisit at M7: PyPI
  `stoke` taken (Fidelity ML lib), StanfordPL/stoke superoptimizer exists,
  Stoke Space dominates search; GitHub + crates.io clean.

---

## Appendix A — What Kabir brings (asset inventory)

The reason this direction fits: **building isn't the constraint** — Kabir is a
generalist who ships hard systems solo, fast. The constraints are *distribution*
and *focus*. This engine is chosen precisely so the moat is **hard engineering**
(Kabir's edge) and the distribution is **open source** (covering the weakness).

| Capability | Proof (project) | Rarity |
|---|---|---|
| Elite interactive 3D / WebGL | Cast Agents (`../../bot/Company/`) — R3F, custom GLSL, 262k-particle GPGPU, gaussian splats, GSAP | top ~1% |
| Systems / language design | AySen (`../loom/`) — general-purpose language in C11: lexer/parser/interp, mark-sweep GC, self-hosting (Stage 1 complete, Stage 2 in progress) | top ~1% |
| LLM infrastructure | universal-omni-router (`../../bot/TokenOptimizer/…`) — OpenAI-compatible gateway, FAISS+MiniLM semantic cache, LLMLingua-2 compression, multi-provider translation | uncommon |
| Full-stack AI product | job_copilot (`../../bot/Resume/`) — FastAPI, auth, Alembic, LLM pipeline, PDF/DOCX | common skill, done well |
| Sandboxed exec / dev tools | CompileX (Monaco + Anthropic + socket.io sandbox), Coding (online judge) | uncommon |
| Programmatic-SEO / monetized content | DataBot — scraper → processor → static-site builder with AdSense + affiliate wiring | — |
| Quant / data engineering | trading + silver — Silverbees ETF backtest → features → models → live | uncommon |
| OS / hardware systems | Hardware Accelerator — PID thermal controller, live sensors, throttle control (C++ + Python) | uncommon |

**The trifecta that makes this engine uniquely Kabir's:** bare-metal systems
(AySen runtime/GC) + hardware/kernel work (Hardware Accelerator) + LLM
infrastructure (the router). Almost nobody has all three at once.

---

## Appendix B — Every business path we ran (the decision journey)

**The ask (2026-07-14):** "invent me something I can sell as a business / make a
company." Two constraint updates then reshaped everything: **(1) no money
pressure — build something *proper and useful*; (2) low-level programming is what
energizes — graphics secondary.**

**Business models enumerated, with verdicts:**

| Model | Verdict for Kabir |
|---|---|
| Freelance (sell hours) | Funds runway; isn't a company. |
| Agency / white-label | Good #2 income; capped by time. |
| Productized service | Strong for 3D landing pages; still services. |
| Self-serve SaaS | High ceiling, wrong *first* solo move (distribution + support burden). |
| Micro-SaaS | Viable if the wedge is narrow. |
| Dev tool / API / infra | ✅ compute-heavy, but the moat + learning fit. |
| Open-core (OSS → paid cloud/enterprise) | ✅ strong — OSS *is* the distribution. |
| Digital products / templates | Scalable and self-marketing — but shallow for this person. |
| Content / audience flywheel | The distribution *engine*, not a company by itself. |
| Vertical AI app | Crowded, easy to copy. |
| Programmatic-SEO sites | Low-ceiling passive income. |
| Marketplace / two-sided | ❌ not solo, not now. |
| Sell trading signals | ❌ never — sell the engineering, not signals. |
| A language sold directly | ❌ languages don't monetize directly (AySen ≠ a product). |

**Three companies were invented, then narrowed:**

- **A — premium 3D/WebGL component shop.** First recommendation → **RETRACTED**
  the moment "no money, proper & useful" landed: shallow for Kabir, small ceiling,
  and its moat is *marketing* (the weak axis).
- **B — LLM cost-cutting gateway** (from the router). Kept as a cheap parallel
  experiment; category is crowded (OpenRouter / Portkey / Helicone).
- **C — build-in-public flywheel.** Not a company — the distribution engine; AySen
  is the credibility/marketing asset that powers it.

**Re-derived selection criteria (the filter that chose the engine):**
moat = hard engineering · distribution = open source · fuel = learning ·
horizon = long.

**Within "low-level," the forks considered:**

- **Concurrent serving for consumer / non-datacenter hardware — CHOSEN.**
  Validated gap, accessible on Kabir's own hardware, pure systems craft,
  OSS-distributable.
- **NPU / next-gen silicon** — highest moat + learning, but hardware access +
  immature vendor toolchains → **Phase 2**, not the start.
- **Agent-reliability runtime** — uses AySen/router more directly and ships
  sooner, but more contested (BAML / DSPy / Instructor). Becomes the Phase-3
  layer, via AySen.
- **Graphics-as-utility** (gaussian splats / WebGPU) — keeps graphics central;
  deprioritized since Kabir chose graphics secondary.
- **A from-scratch llama.cpp clone** — rejected: educational, not *useful*,
  without a real wedge.

---

## Status / build log

Direction + wedge + AySen alignment: **DECIDED (2026-07-14).**

- **2026-07-16 — project live.** Named **stoke**; folder renamed; git repo
  initialized (branch `main`, **local-only by directive — no remote/push
  until Kabir explicitly says so**). Decisions locked:
  from-scratch compute layer, CPU-first, C11, library-first (`libstoke.a` +
  `stoke.h`, server as thin frontend). Spec committed:
  `docs/superpowers/specs/2026-07-16-phase1-concurrent-cpu-core-design.md`.
  Plan committed: `docs/superpowers/plans/2026-07-16-phase1-implementation.md`
  (M0/M1 task detail, M2–M7 roadmap). Landscape re-verified by live web
  research — wedge confirmed; sharpest datapoints: mistral.rs CPU 17 tok/s vs
  llama.cpp 124 on same settings; exllamaV3 NVIDIA+Python-only; MLC tiny+TVM;
  no llama.cpp scheduler redesign found; Ollama `NUM_PARALLEL` default 1 and
  now runs a three-engine strategy (proof that frontends swap engines — the
  library-first insertion point). Assets staged: Qwen3-4B-Instruct-2507
  Q4_K_M GGUF (bartowski, 2.5GB, magic-verified, `models/`), llama.cpp b10037
  win-cpu vendored (`bench/vendor/`), Ollama 0.30.11 present. **NEXT: M0** —
  bench harness (mock-verified timing), 4-config baselines, premise
  kill-switch gate.

- **2026-07-17 — M0 COMPLETE: baselines banked, premise CONFIRMED, kernels
  are GO.** Full 32-cell matrix (4 configs × S1/S2-sweep/S3/S4, 3 runs/cell,
  medians, CPU-pinned, same GGUF) on the reference box; report committed:
  `bench/results/2026-07-17-baseline-qwen3-4b.{md,json}`. Kill-switch: best
  baseline at concurrency (llamacpp-tuned **17.7 @N=8**) = **11.0% of the
  ~160 tok/s physics ceiling** (threshold ≥70%). Key discovery: **every
  engine peaks at N=4 then collapses** (llamacpp-tuned 33.6 → 17.7 → 14.2) —
  nobody sustains scaling into N=8/16 on consumer silicon; the wedge is
  measured fact on this box. Findings kept: ollama-default inverts (8.4 @16 >
  5.4 @8, ITL faster than its own N=1 — unexplained, recorded not chased);
  out-of-box ollama hard-rejects a 4k prompt (fixed slot ctx 4096 → HTTP 400)
  while llamacpp-default serves it (b10037 default ctx follows the model) at
  ~77s TTFT — falsified the "both defaults reject" guess; llamacpp-default
  beats llamacpp-tuned on s3 fanout (13.3 vs 11.7) — static slot-splitting
  isn't free, a real scheduler must beat it. M5 bars now concrete (in spec
  "Physics budget", re-anchored with the measured table): ≥17.7 @8 / ≥14.2
  @16 (1× llamacpp-tuned) and ≥10.8 @8 / ≥16.8 @16 (2× ollama-default — the
  N=16 bar exceeds every measured baseline). **NEXT: M1 (owner: Opus 4.8) —
  Task 5 gguf.c**, worker-brief protocol in the plan.

- **2026-07-18/19 — M1 COMPLETE (tag `m1`) — front door built,
  oracle-proven, hardened.** Tasks 4–6b ALL COMPLETE + Opus-task-reviewed, ZERO
  Critical/Important each: skeleton `9f0fc7d`; gguf.c v3 parser + mmap +
  `stoke info` `0e89e10` (fuzz-negative gated, real-file golden byte-exact
  both platforms); tok.c 6a `d02a28f` — from-scratch Unicode byte-BPE
  (generated unicodedata category tables + qwen2 regex + GPT-2 byte-remap +
  merge-rank BPE), byte-exact vs vendored `llama-tokenize` b10037 on a
  2k-line multilingual fuzz corpus, Win+WSL asan/tsan green; the worker
  caught the brief's invalid-UTF-8 rule being wrong about the oracle (real
  behavior: structural validation + U+FFFD substitution, lossy) — ratified,
  proven by the passing gate; tok.c 6b `b314982` — special-token parse +
  `tok_apply_chatml`, goldens byte-exact in BOTH parse modes. 6b's dispatch
  probe found the oracle's match set is MODE-DEPENDENT (`--no-parse-special`
  still matches USER_DEFINED tokens, only suppresses CONTROL) ⇒ 6a's
  pure-byte `tok_encode` was subtly wrong in a way its own gate couldn't see
  — corrected in place; 6a's golden re-passed byte-identical as the
  regression net. ChatML emits structure as ids and encodes content with
  CONTROL unmatched — the render-then-parse path was rejected as a
  prompt-injection hole. Task 7 close-out: whole-branch Opus review ZERO
  Critical → fix wave #1 `b5e2f37` (unicode sweep gate: deterministic
  22,848-line codepoint sweep incl. unassigned cps, passed byte-exact ⇒ the
  feared Unicode-16-tables-vs-b10037 skew did not materialize; doc-integrity
  purge of git-ignored-path citations; fixture/assertion strengthening;
  green bar reproduced Win 98/98 + WSL plain/asan/tsan) → wave task-review 0
  Critical / 4 Important (incl. regen_goldens truncate-before-oracle hazard
  that could silently destroy the regression net) → **fix wave #2 LANDED
  `9a64d23`** (F1–F8: caller-attribution threading through
  `bpe_encode_span`/`bpe_encode_fragment` + a test pinning the ChatML-path
  attribution, bare-label purge [path-regex residual 0], regen-goldens
  temp-file/exit-status/shape-check safety [both failure modes demonstrated:
  oracle-fails and oracle-lies, golden untouched each time],
  `tests/golden_lib.sh` gate dedup [Kabir-decided; SKIP and
  STOKE_REQUIRE_GOLDEN=1 paths demonstrated], exact-fatal pins [worker left
  5 of 6 weak pins unfixed — completed at review with runtime-unique
  substrings, mechanically checked], honest F8 decomposition: ON 1707 / OFF
  2236 ids, 132 CONTROL + 42 USER_DEFINED-in-both; alignment identity
  1707−242+771=2236 closes EXACTLY; the brief's narrow arithmetic is short
  by 110 — all whitespace/punct neighbors {198×53, 220×49, 13×5, 30×1,
  271×2} absorbed across special seams by the qwen2 regex, matching 6b's
  seam-merge side-finding; both goldens stay oracle-byte-exact so this is a
  property, not a bug). The wave's worker died pre-report; Fable
  (2026-07-19 gate visit) reviewed the diff against the brief, completed
  F5, ran all verifications, and landed it. Incident caught at
  verification: the worker's editor rewrote the four gate scripts +
  regen_goldens.sh with CRLF — msys sh masks it, WSL dash fatals (`set:
  Illegal option -`), and Git-Bash-side reads HIDE the `\r` (text-mode
  mount; `od -c` from WSL is authoritative) — normalized to LF and pinned
  forever via `.gitattributes` (`*.sh text eol=lf`, in `9a64d23`). GREEN
  BAR at `9a64d23`: Windows `mingw32-make test` 9/9 + 43/43 + 47/47 + 4
  gates OK; WSL `wsl-check.sh` plain+asan+tsan ALL GREEN. **M1 GATE
  STAMPED (Fable, 2026-07-19).** Ledger: `.superpowers/sdd/progress.md`.

- **2026-07-19 — M2 entry designed (Fable).** Design note:
  `docs/superpowers/design-notes/2026-07-19-m2-op-spec-and-parity-protocol.md`
  — scalar forward pass op-by-op with the traps named (head_dim=128 is a
  first-class GGUF key, NOT n_embd/n_head=80; LM head is TIED to
  `token_embd.weight`, no `output.weight` tensor; RoPE is NEOX-paired
  (i,i+64), re-confirm from source at dispatch). Parity model = a SECOND
  GGUF, Qwen3-4B-Instruct-2507 **Q8_0** (bartowski, ~4.3GB, acquire + sha256
  at M2 dispatch — the Q4_K_M dev file's K-quants are M3 scope; it stays
  put, every M1 gate keeps running against it). GATE = greedy 64-id token
  parity vs llama-server `/completion` over a committed 50-prompt corpus,
  near-tie waiver protocol (ε=1e-3 nats, ≤2 waivers, committed reserve
  pool). Parity-debug ladder L0–L3 with an independent numpy
  `ref_forward.py` as first-class deliverable; dispatch probe pack P1–P5;
  worker sketch W1–W4. Plan amendments recorded at this gate (now in the
  plan's Global Constraints): fatal()-choke-point / no-new-globals /
  x*-alloc discipline binding M2 onward (keeps the M6 embeddable-ABI
  conversion local); `scripts/win-check.cmd` early-M2 chore (Windows
  golden-gate half-gap); specials-matching revisit-condition precision fix.

- **2026-07-19 — Fable exit: M3–M7 pre-designed, project goes Opus-only
  (Kabir's direction).** Design note
  `docs/superpowers/design-notes/2026-07-19-m3-m7-forward-designs-and-risk-playbook.md`
  front-loads every decision the model allocation reserved for Fable: M3
  locked decisions (Q6_K is REQUIRED for Q4_K_M support — the dev file's
  attn_v/ffn_down/token_embd are Q6_K; integer cores on both scalar+SIMD
  paths so the exact-diff gate is exact; AVX-VNNI VEX intrinsic trap;
  work-stealing pool; the ≥50% attribution ladder), M4 (11 sim invariants,
  batch-composition independence by construction, full-reservation
  admission, Q8-KV never token-parity-gated), M5 in full (state machine,
  step loop, watermark W=max(8,2·|RUNNING|), preempt-youngest + replay-rule,
  chunk co-run assertion, rings/queue design, sim-harness trace catalogue,
  bar-miss attribution ladder), M6 (fatal→library mechanism DECIDED:
  load-time trap via thread-local jmp_buf + API-misuse error returns +
  invariant aborts through a log callback; full ABI freeze checklist;
  server/SSE/OpenAI-compat traps; M0-harness-as-adapter is the compat
  smoke), M7 (Llama-3.2's four silent-garbage deltas incl. llama3 rope
  scaling + llama-bpe pre-tokenizer as a second regex entry; unsloth-file
  parity; 24h soak; name memo), cross-cutting rules, and per-milestone
  mechanical gate checklists Opus stamps itself. **Escalation now = a
  half-page memo to Kabir** (gate fails twice same cause / >1 day stuck /
  physics contradiction / any gate-loosening temptation); Kabir may summon
  Fable ad hoc, nothing depends on it. The companion M2 note's two Fable
  references were amended to this protocol. Allocation amendment recorded
  in the note (supersedes the plan's §Model allocation): **Opus 4.8 owns
  M2→M7 end-to-end — sessions, briefs, reviews, gate stamps.**

- **2026-08-02 — M2 COMPLETE (tag `m2`) — the scalar forward pass is
  correct, and it is proven against an independent implementation.** Gate
  report: `docs/superpowers/gate-reports/2026-08-02-m2-parity.md` (`b225737`).
  **Both legs PASS. Token leg 24/24 exact: 1,536 consecutive token-exact
  greedy decisions over 3,503 forward positions, 0 diverged, 0 of the 2
  available waivers drawn, 0 tooling errors** — adjudicated by
  `ref_gate.py --adjudicate` (`2372e98`), which sequences the ruling's
  committed decision functions under a hash guard and returned `verdict:
  PASS` with every reason array empty. **L2 leg 36/36 layers ≤1e-3, worst
  L34 6.104e-04, no STEP**, both sides emitting token 12095 — re-run at the
  gate HEAD rather than cited, and it reproduces the W3-landed figure
  exactly. Build chain: `stoke.exe d2b17c53…` + `ref_forward.py 4e293e4b…`,
  frozen throughout and re-verified unmoved; corpus `f5d99ee` proven an
  ANCESTOR of the forward pass `79a4b29`; no `tests/data/*.golden` changed
  since `m1`. Matrices green at close-out: Windows `win-check.cmd` and WSL
  `wsl-check.sh` plain+asan+tsan, 140 tests + 4 golden gates on every leg,
  all with `STOKE_REQUIRE_GOLDEN=1` so a moved model hard-fails instead of
  silently skipping. A3 tokenizer-table identity closed in-house (`stoke
  tokenize` over both ggufs → byte-identical 51-id list, both exit 0); L1
  spot-check re-run (`dump_tensor.py` dequantizes a real Q8_0 row, exit 0,
  2,560 values).
  **THE SCOPE CHANGE, and why it is not a loosened bar.** The original gate
  was 50/50 token-exact vs `llama-server`. The 07-28 survey measured **19 of
  50 prompts diverging from the oracle, with 18 of 18 classifiable hits a
  clean top-2 swap** at 0.00095–0.2039 nat gaps — because the oracle
  quantizes activations to Q8 for integer dot products while M2's stoke is
  pure f32. Different arithmetic classes rank near-ties differently; that is
  not a defect stoke can fix while it remains f32. **Kabir took memo option
  3:** re-scope M2's token gate to C-vs-ref (both pure f32) on a stratified
  subset + the L2 ladder, and **defer oracle token-parity to M3**, where the
  integer core puts stoke in the oracle's own numeric class. Crucially the
  substitute gate's **subset and pass criterion were fixed BEFORE the run**
  by Fable's binding ruling (`8e1e618`), so no bar could be written after
  seeing a result. **19 of the 24 subset prompts are ones that diverged from
  the oracle, and every one is token-exact against the f32 reference** —
  which is the evidence for the thesis, not merely consistent with it.
  **Carried to M3:** the deferred oracle gate, the required aligner fix
  (byte lengths from the GGUF vocab via `gguf_reader.py`), and the 07-28
  census as the M2-era baseline.
  **What M2 cost, honestly:** the forward pass itself landed in days; the
  gate took two weeks, almost entirely on measurement infrastructure —
  a cost model rebuilt twice (flat → context-aware → absolute-slack floor),
  timeout ceilings that killed six prompts before being derived properly,
  and an ops layer (Task Scheduler + watchdog + pauser) built so a 12-hour
  run survives sessions, reboots and gaming. The final run banked 16 prompts
  in 731.8 min unattended with **zero restarts**.

Related paths (from this folder):
- **THE OPERATIONAL MAP — read FIRST in any build session:** `docs/ROADMAP.md`
  (position, per-milestone gates + operative-doc pointers, session
  protocol, escalation; added 2026-07-19 at the `m1` tag)
- Beginner primer — every concept from scratch: `docs/beginners-primer.md`
- Background — what vLLM is, in depth: `docs/vllm-and-the-serving-landscape.md`
- AySen: `../loom/` (handbook: `../loom/docs/HANDBOOK.md`)
- Router: `../../bot/TokenOptimizer/universal-omni-router/`
- Earlier freelance/jobs plan (superseded as *company direction*, still valid as
  runway funding): `../../bot/GET-WORK-AND-MONEY.md`
