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
2. **Phase 2 — RE-ORDERED 2026-08-02** (Fable strategic review, Kabir-approved;
   binding form + rationale in `docs/ROADMAP.md` "Beyond Phase 1"). Phase 2 is
   now **the bridge layer — the Ollama half of the gap**: model alias registry
   + `stoke pull`, idle load/unload under an LRU residency budget, per-model
   chat templates, `/metrics` (M5's bench driver already computes the numbers),
   and memory-elastic serving on M4's runtime-variable block pool. Reasoning:
   Ollama's moat was never its engine, it is `ollama run <name>`; and the
   desktop-pressure behaviour M2's ops history measured live (Chrome returns,
   Windows trims the working set, throughput falls ~3×) is something neither
   vLLM nor llama.cpp handles, so graceful degradation is a differentiator the
   paged-KV machinery already almost pays for.
   **Heterogeneous + NPU execution moves BEHIND it** — schedule across
   CPU/GPU/NPU, light up the idle 40-TOPS silicon; still the bleeding-edge
   moat, but hardware-gated (see "Open questions": no NPU on this box), so it
   waits on silicon rather than blocking the product story.
3. **Phase 3 — the runtime.** Fold in the router (cloud-spill fallback) + AySen
   (`local` client, contracts). Now it is not just serving — it is a *reliable,
   local-first* runtime.
4. **Phase 4 (optional, far off) — the company.** Open-core: hosted control
   plane, enterprise on-prem, support.

---

## Open questions — ALL ANSWERED 2026-07-16 (Kabir + probe)

- **First hardware target** → Kabir's box: i5-13450HX (6P+4E, AVX2+AVX-VNNI,
  no AVX-512), 16GB DDR5-4800 (~77 GB/s theor.), RTX 3050 Laptop 6GB (168 GB/s,
  CC 8.6) reserved for Phase 1.5. **No NPU on hand** (HX predates them) — the
  NPU work needs future hardware, which is exactly why the 2026-08-02 re-order
  moved it behind the Phase-2 bridge layer. CPU-first, CUDA second.
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
  immature vendor toolchains → post-Phase-1, and since the 2026-08-02 re-order
  it sits BEHIND the Phase-2 bridge layer, not at the front of Phase 2.
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
  census as the M2-era baseline. **[RESOLVED AT M3, Fable 2026-08-05:
  the deferred gate ran 2026-08-04 and failed leg 1 at p02:16 — an
  M2-documented divergence where `ref_forward.py` sides with stoke — so
  the deferral's prediction (integer core ⇒ oracle's numeric class) is
  implemented and measured insufficient. The oracle bar is retired as
  structurally unreachable; item [2] is restated as the
  classified-divergence criterion and the oracle demoted to an optional
  survey. Binding:
  `docs/superpowers/memos/2026-08-04-m3-gate-item-2-RULING-FABLE.md`.]**
  **What M2 cost, honestly:** the forward pass itself landed in days; the
  gate took two weeks, almost entirely on measurement infrastructure —
  a cost model rebuilt twice (flat → context-aware → absolute-slack floor),
  timeout ceilings that killed six prompts before being derived properly,
  and an ops layer (Task Scheduler + watchdog + pauser) built so a 12-hour
  run survives sessions, reboots and gaming. The final run banked 16 prompts
  in 731.8 min unattended with **zero restarts**.

- **2026-08-11 — M3 COMPLETE (tag `m3`) — the SIMD kernels and the thread
  pool are in, and every divergence from an independent implementation is
  accounted for as arithmetic, not as a defect.** Gate report:
  `docs/superpowers/gate-reports/2026-08-11-m3-q4km-grid-gate-report.md`
  (`d145659`). **[2v2] ZERO BUG-CLASS: 9 divergences on the Q4_K_M
  reference grid, 9 ARITHMETIC-CLASS**, worst margin 0.555672 against a τ of
  1.023739 — 54 % of threshold, no ESCALATE, so the L2/L3 ladder was never
  needed; every top-5 set agreed modulo the rank-5 tolerance and no foreign
  core candidate appeared. **τ_Q4KM frozen BEFORE any grading** under the
  per-file key `f32_intcore_q4_k_m`: 1.023739 / 1.178094 / 0.811432 over
  bands [0,100) / [100,300) / [300,600), each from 20 rule-selected agreeing
  positions. Evidence: 6 probe legs + 5 required continuations + 2
  added-sample legs, **290 covered positions with per-prompt coverage stated
  so the tripwire's denominator is explicit**; p41 — the adversarial
  477-token leg — ran clean 64/64 and needed no continuation. Q8_0 was
  discharged by the 08-06 ruling against the archived M2 binary. Binary
  `stoke.exe 876ce53f…` frozen since 2026-04-08 and re-verified by sha256 at
  every leg and every probe; archived at the tag as
  `build/stoke-m3-gate-876ce53f.exe`. Matrix green at the last
  source-touching commit `5051878` (Windows 9/9 + 4 goldens; WSL
  plain/asan/tsan, zero sanitizer findings) with no `src/` file changed
  since. Other items: [1] test_simd 12/12 across all quant kinds; [3] the
  bar re-anchored HARDER to 8.08–8.46 and passed on both legs; [5]
  microbench table + `history.jsonl` GATE row.
  **THE ONE PRECONDITION MET BY JUDGEMENT, flagged rather than smoothed
  over.** The close criterion requires the flip tripwire quiet; **it fired,
  9 ≥ 8.** Its threshold was derived from a ~0.5 %/token near-tie base
  measured on **Q8_0**, and Q4_K_M measures **3.10 %/token** — 4.2× Q8_0's
  own rate, because 4-bit puts the f32 reference and the integer core
  further apart than 8-bit does. Under the assumed rate the observation is a
  1-in-46,800 event; under the file's own measured rate the expectation is
  exactly 9.0. **Kept at ≥8 rather than recalibrated**, because setting a
  detector's threshold from the data it must police is the circularity the
  whole τ discipline exists to prevent. What makes that an explanation
  rather than an excuse: the mismatch was computed and committed in
  `6b484b9` **when only 5 flips existed**, before p41 and before any
  continuation, and predicted the firing in writing. Disposition:
  `docs/superpowers/memos/2026-08-11-m3-tripwire-anomaly-memo.md`; Kabir
  confirmed 2026-08-11.
  **What M3 cost, honestly:** the kernels were the easy half. The gate
  consumed the milestone — an oracle bar that had to be retired as
  structurally unreachable after it failed on a divergence the reference
  itself sided with stoke on, a replacement criterion that took two Fable
  rulings to fix, and a τ-freeze pipeline that did not exist end-to-end
  until the final night. Three silent traps were found in inherited tooling
  while wiring it: a τ key that would have frozen Q4_K_M under Q8_0's name,
  a sample floor counting values where the ruling meant positions (20 → about
  4), and continuation divergences recorded in no manifest at all.

- **2026-08-13 — M4 COMPLETE (tag `m4`) — paged KV and static batching are
  in, sequences share one block pool, and the batch path is byte-identical to
  solo generation at every width.** Gate report:
  `docs/superpowers/gate-reports/2026-08-12-m4-gate-report.md`. All five §8.4
  items measured. **[1]** invariant sim ×1e6 clean under ASan+UBSan, the set
  enumerated (not counted) in the invariants note, with `dedup_hit=16633
  adopt=1187` proving the dedup half is exercised rather than merely present.
  **[2]** 31 sequence streams over N∈{1,2,4,8,16}, every one byte-identical
  to its solo `stoke run` reference. **[3]** prefix dedup fires in real
  inference: 192 of 287 prompt positions served from cache, prefill 287→95,
  streams unchanged. **[5]** the N∈{1..16} curve monotone within spread —
  **and its finding, not its verdict, is why M4.5 exists: batching buys
  1.116×, because the engine is compute-bound (~64 GFLOP/s flat) and the
  quant unpack repeats per sequence.**
  **[4] IS THE ONE THAT NEEDS READING IN FULL.** It passes the bound that
  gates correctness — **zero BUG-class over the frozen 50-corpus, all 38
  divergences classified ARITHMETIC-CLASS, and all seven ESCALATEs laddered
  rather than sampled, with no STEP in ten 36-layer curves** — and misses the
  class-rate bound at 38/50 and 1.1875 %/token against ≤19/50 and ≤0.5 %/token.
  τ frozen from 9 agreeing of 15 probes before grading (1.0103 / 1.2617 /
  0.7259, all three bands drawing real samples). Every ESCALATE turned out to
  be churn at the `rank5+τ` cut among tokens **both sides rank in their
  top-5** — none was "a candidate the other implementation never ranked",
  §6's own BUG language — and the divergent positions' layer curves sit
  *below the second-lowest control*, i.e. not elevated at all, which runs
  against the selection confound rather than with it.
  **§8.1 pre-ruled the disposition and it was followed: a results-vs-physics
  memo, not a grind-until-green** (`docs/superpowers/memos/2026-08-13-m4-q8kv-
  results-vs-physics.md`). The bound calibrates "arithmetic-class noise this
  project already accepts" against the M2 census — a pair differing only in
  **f32 reduction order, ~1.2×10⁻⁷** — while Q8-KV requantises the KV cache to
  8 bits, **~2.3×10⁻³ RMS, four orders larger** (measured 0.4836·d, better
  than its own 0.562·d guarantee). That lands ≈0.04–0.08 in absolute logit
  terms, the same order as the observed margins 0.013–0.566, so near-ties flip
  and decided tokens never do — exactly the measured pattern. The metric is
  near saturation too: one first-divergence per prompt over 3,200 tokens caps
  it at 1.5625 %/token, so 1.1875 % is 76 % of the largest value it can
  report. **f32 KV stays the default; M5's design note carries the dtype call**,
  where the footprint argument is independent — the f32 pool at N=16/ctx2048
  commits **9.04 GB, measured 2026-08-13** on a 15.63 GB box.
  Binary `stoke.exe f4f9b667…` archived as `build/stoke-m4-gate-f4f9b667.exe`;
  it differs from the [2]/[3] archive because item [4] needs `e2a20d9`'s
  `--kv`. **Two process lessons paid for themselves this milestone:** the G4
  harness banked nothing until a phase completed and lost 76 minutes when
  stopped, so it was made resumable *before* re-running (every unit fsync'd,
  `--resume`, cross-config resume refused) — and the smoke test then caught
  the fix's own defect, an append fusing onto a truncated tail. And **the
  E-cores were the measurement noise**: pinning to the P-cores took probe
  pairs from 95–146 s to a stable 29.4–30.1 s and collapsed a 5.8× run-to-run
  spread to ±0.5 s.

- **2026-08-17 — M4.5 catch-up, and L4.** This entry fills a gap: the M4.5
  ladder ran across several sessions that recorded themselves in
  `docs/superpowers/gate-reports/` and in the agent's own memory but never in
  this log. Ladder outcome to date: **rung B landed** (per-type batched row
  workers); **rung C / maddubs landed and is graded**, with **Q6_K on maddubs
  a 1.108× win that moves not one bit** (`cf0d8eb`); **rung A′ was tried and
  REJECTED ON MEASUREMENT at 0.907×** (`0ffee94`, removed in `3d39ae9`) —
  the whole ladder is **B/A = 1.54× in one interleaved sitting**. **Rung D was
  never run and is now ruled out on evidence, not on cost:** it repacks matvec,
  and matvec was measured at parity.
  **P-BAR was graded three times and missed all three at the same value —
  0.833 t=6 pinned, 0.827 re-graded post-EcoQoS, 0.828 at t=8 unpinned — short
  by ~2.7%, which discharges §7.5 step (1) (thread count) as a NEGATIVE.**
  Two findings dominate the milestone. **(1) A 10.9× intermittent collapse was
  Windows EcoQoS parking the pool on the E-cores** — helpers slept on a condvar
  ~250×/token, which is what background work looks like — **fixed by a
  per-thread `ThreadPowerThrottling` opt-out (`76dee7f`), with
  `tests/tooling/run_ecoqos_repro.ps1` kept as a regression test. Every stoke
  perf number taken before that commit may be depressed; llama's are not.**
  **(2) The remaining gap is attention over the KV cache, and matvec is
  already at parity** — fitted in one interleaved sitting, the two engines'
  context-independent intercepts agree (stoke 48.76 vs llama 52.51 ms/token)
  while stoke's slope is ~12× worse, and attention is 84.6% of decode at
  ctx 1500 (`774d625`, `db85e0f`).
  **▶▶ L4 THEN FOUND THE CAUSE, AND IT WAS NOT ARITHMETIC: ATTENTION HAD NEVER
  RUN ON THE POOL.** `pool_for_rows` is called from `src/kernels/vec.c` only —
  seven times, all matvec — so the per-head loop ran on the calling thread from
  K5 onward, i.e. ~6/7 of a long-context token on one core out of eight. The
  2026-08-12 ruling's §7.6 had framed L4 purely as SIMD and priced it as a
  graded arithmetic change; **parallelising over heads is parallelism over
  INDEPENDENT OUTPUTS, so the arithmetic inside a head is untouched — no twin,
  no grading run, no re-anchor, and §7.6's 8-lane canonical fold is not
  needed** (`ba4e140` pool, `755a43b` attention). Two pool additions were
  forced: `pool_for_units` (the old entry point had chunk 64 and a two-chunk
  floor, so **any parallel-for under 128 units silently ran inline** — matvec
  never noticed at ≥1024 rows, attention has 32) and `pool_worker_index`, since
  the scores row and KV scratch had to move from per-sequence to **per-worker**.
  **A real heap overrun was found and fixed on the way in** — `s_kv` had two
  allocation sites and the lane loop's raise from 1 to 4 (K,V) pairs reached
  only one, so `run` and all prefill overran by 3072 B at `--kv q8`; the
  f32-only check that preceded it could not have caught it, because at f32
  `kv_read` is zero-copy and never touches that buffer. **And the per-worker
  scratch is now one allocation PER WORKER rather than one block sliced by
  worker, because mutating the bug back in and running ASan on the real model
  returned CLEAN: a sliced arena has no boundary for a sanitizer to trip.**
  Correctness green on every leg — Windows 13/13, WSL plain 13/13, **WSL TSan
  13/13 with zero reports, which also discharges the TSan leg owed since
  `76dee7f`**, ASan clean at `--kv q8`, byte-compare 6/10 cases with all four
  arms identical (4 cases owed). The new test is mutation-verified twice.
  **SPEED IS NOT YET MEASURED and is the only open item in M4.5**; both changes
  are env-gated (`STOKE_ATT_PAR`, `STOKE_ATT_LANES`) so all four arms
  interleave from one binary. Design note:
  `docs/superpowers/design-notes/2026-08-17-m45-l4-attention-on-the-pool.md`.

- **2026-08-19 — M4.5 CLOSED, tag `m45`. P-BAR IS MET.** The overnight chain
  (byte-compare gate -> 4-arm A/B -> P-BAR at both graded configurations) ran
  unattended and finished clean. **P-BAR: 0.912x at t=6 pinned, 0.892x at t=8
  unpinned, against the 0.85 bar the milestone MISSED at 0.827-0.838 before
  L4** — three blocks each, spaced across the night from different thermal
  states, arm order rotating, llama re-measured inside the same sitting per
  SS8.2. Correctness closed the same day: **byte-compare 10/10 in ONE sitting**
  (`a48de9f`), discharging the four cases owed from the partial run.
  **What L4 is worth: 1.09x at ctx 87, 1.58x at 554, 2.94x at ctx 1500, and the
  slope falls 0.1713 -> 0.0289 ms/context-token** — stoke was 11.9x worse than
  llama on slope, now 1.72x. **THE M5 ALARM IS RETIRED**: ctx 1024 costs ~93
  ms/token, not the ~187 measured against a single-threaded attention loop, so
  the ratio goes 0.33x -> 0.76x and M5's aggregate bars are back in range; the
  08-11 budget's KV-traffic claim (crossover N x L ~ 16,936) is untouched and is
  now the live constraint. **THE POOL IS 99% OF THE WIN** — isolated at ctx
  1500, lanes alone 1.26x, pool alone 2.90x, both 2.94x, so `ATT_LANES` earns
  ~1.3% and is a simplification candidate rather than an optimisation.
  **THE PREDICTION MISSED HIGH (317 -> ~63 ms predicted, 106.75 measured) — the
  third instance after A' (0.907x) and Q6_K (1.108x vs 1.20x) ⇒ price a kernel
  argument at half its argued gain.** **ARM A IS UNFIT AS AN ANCHOR**: the
  pre-maddubs archive predates `76dee7f` and read 9.99 / 5.30 / 0.92 / 8.6
  across sittings while arm B and llama held to a few percent — stable within a
  sitting, up to 10.9x between them. A Task-Scheduler-classification
  explanation was TESTED AND REFUTED (scheduled 9.044 vs interactive 7.999), so
  what survives is the fact, not a mechanism; both reports grade on B/L and say
  so on their face. Two harness guards were fixed the same night: the standby
  check read Normal-Priority bytes alone and cried wolf on three of three good
  blocks (the model was in the RESERVE bucket), and `-AlwaysRestoreISLC` on a
  chained leg lets ISLC purge the cache the next leg depends on. Gate report:
  `docs/superpowers/gate-reports/2026-08-19-m45-gate-report.md` — its SS3 also
  corrects the record on rung C, which was NOT adopted (C-FAIL-RATE fired,
  `a53ee34`/`bc8bd00`); the criterion that stopped it was later retired as
  unsound (`e253d7e`), and what ships is the bit-exact maddubs twin (`650c47c`)
  plus Q6_K on maddubs (`cf0d8eb`).

- **2026-08-28 — M5 CLOSED, tag `m5`. Continuous batching works; the two bars it
  misses are not scheduler questions.** The scoreboard is SEVEN items, not five:
  **[1] cancel-safe sim MET · [2] the real bars MISSED, 0.51x · [3] ITL stall
  MISSED, 2.7x at the smallest legal chunk · [4] S3 fanout MET, 13.535 against
  13.3 · [5] 1 h ASan+TSan soak MET, 480,874,507 steps and zero findings ·
  [6] preempt-lossless on the real model MET, and re-MET on the dedup default ·
  [7] ledger/MEMORY close-out done today.** **[2] AND [3] ARE THE SAME ARITHMETIC
  WALL, and every scheduler-shaped explanation is dead**: preemptions read 0
  everywhere, the STATIC batch path clears decode, and serve's decode-only rate
  sits within 1.6% of it. **The wall is llama's ALGORITHM, not its instructions**
  — front-end (~40% of emitted instructions to ~7%), port throughput (half the
  p01 µops to 12.1%) and memory (activation re-reads, killed against a MEASURED
  7.0% noise floor) were each measured and each dead; what remains is that stoke
  runs a matrix-VECTOR product N times where tinyBLAS runs a register-blocked
  matrix-MATRIX product reusing BOTH operands. That is a redesign, M6/M7-sized.
  **THE ONE LIVE LEVER FOUND IN M5 WAS TAKEN AND PRICED HONESTLY.** An isolated
  per-tensor table showed the batched Q6_K kernel going BACKWARDS — every Q4_K
  tensor climbs with N (38→70 GF/s) while `ffn_down` read 33.42 at N=1 then
  17.64/19.27/20.44/10.89 — because Q4_K does 32 int8 products per `vpmaddubsw`
  where Q6_K widened i8→i32 for 8, and batching multiplies that body by N.
  `vec_dot_q6_k_q8_k_xn_maddubs_avx2` is **bit-exact by construction** (keep `u`
  raw in [0,63], carry −32 on `bsums`; a 32-B chunk spans TWO scale groups) and
  **7.68x at N=16 on the kernel** — but 7.68x on a kernel is not 7.68x on a
  token: **1.20x on a prefill token, and 1.2949x end-to-end on gate [2]'s own
  cell**, the latter from a same-sitting one-binary A/B (ref 6.993/6.962 vs
  maddubs 9.032/9.038, floor 0.45%, RESOLVED). **The bars went 0.28x/0.27x to
  0.51x/0.51x and the stall 4.4x to 2.7x, and neither is a pass.** ⚠ **The
  cross-sitting halves of those moves are NOT attribution and the referent said
  so**: llama-bench is a fixed third-party binary and ITS OWN numbers moved
  between sittings — `pp512` 72.24→92.29 (+27.8%) against `tg128` 17.48→18.77
  (+7.4%), the exact compute-dense-vs-memory-bound split this box's thermal
  drift produces. The kernel number above is the same-sitting one for that
  reason. **P1 was FALSIFIED and left standing rather than re-fitted** (the band
  was mine and wrong — an in-process null floor applied to a between-process
  ratio), and its successor was NOT SUPPORTED but FLAT: prefill fits
  `4447 ms/chunk + 4.68 ms/token`, so ~88.6% of it does not care how many tokens
  ride in a chunk — **which is exactly the term a matrix-MATRIX product
  amortises, making this rung a down payment on the wall rather than a crack in
  it.** ⚠ **THE INSTRUMENT FAILED MORE OFTEN THAN THE ENGINE, AGAIN.** A gate
  runner banked "written and desk-reviewed" turned out never to have PARSED;
  `[Parser]::ParseFile` is now the standing pre-flight for every .ps1, and it is
  not sufficient — `-f (if …)` parses clean and dies at runtime. A pre-flight
  shared the gate's BANK and would have resumed 2-seq units as an 8×300 gate
  result. A bank keyed by stamp alone would have reported 11/11 GREEN about
  binaries that no longer existed; **both binaries are now hashed into every
  bank key, and the same defect was found a third time on 2026-08-28 in the
  gate [3] cell cache, which still held all five 08-26 cells and would have
  published them under today's commit as a re-measure.** A null arm was reused
  one level out and manufactured a falsification: **a null bounds THE ARM IT IS
  BUILT FROM, must be read PER CELL (30% at the smallest, 1.0% at the largest),
  and a cell resolves only at 2x its floor.** And the M5 ledger row itself came
  out reporting a **fabricated 3x regression across all 15 ops** — `bench --ops`
  is the one path that never calls `pool_start` and so never gets the 2026-08-17
  EcoQoS opt-out, leaving the project's standing regression instrument the one
  thread Windows could park on an efficiency core. The opt-out recovers 3.25x
  and lands the ops ON their M3 values; the row that ships has `rmsnorm`,
  `rope_neox`, `softmax` and `argmax` **identical** to M3 as its own control.
  Gate reports: `docs/superpowers/gate-reports/2026-08-28-m5-bars-q6k-report.md`,
  `-m5-bars-q6k-ab-report.md`, `2026-08-28-m5-gate3-report.md`.

- **2026-08-28 — M6 STARTED: A1 and A2 are done and gate items [5] and [6] are
  banked.** M6 is `stoke.h` freeze + server. Rung order and the two claims it
  rests on: `docs/superpowers/design-notes/2026-08-28-m6-entry.md`; running
  narrative in `.superpowers/sdd/progress.md` ADDENDUM 24.
  **A1** (`39e7166`, `f69e224`): the `fatal()` trap, with the thread-local
  `jmp_buf` and error buffer given **EXTERNAL linkage** — `fatal()` is
  `static inline` in a header, so trap state kept `static` beside it would give
  every translation unit its own chain and whether an error was caught or killed
  the host would depend on which TU raised it. **All 322 call sites unchanged**
  (the forward note's "~90" was wrong by 3.5x, counted not estimated). `stoke.h`
  v0 is written, implemented and driven by a real generation in
  `tests/abi_smoke.c` — and is **NOT frozen**: A7 freezes it, after `server.c`
  has had the chance to prove it wrong, which is the whole reason the server is
  built as a CONSUMER of the header rather than a sibling reaching into
  `sched.h`. Gate [5] banked with a mutation whose signature was not the obvious
  one: with `extern "C"` removed the C TU still exits 0 and the C++ TU dies at
  link with **zero "undefined reference" lines on this toolchain** — a check
  grepping the log for that phrase would have missed it; the signal is the make
  exit code.
  **A2** (`e188e5b`, `053c746`): `src/http.c`, the HTTP/1.1 request parser —
  allocation-free, results are offsets into the caller's buffer, re-parses from
  the start every call (no partially-advanced machine to desynchronise), and
  **it never calls `fatal()`**, because everything reaching it is untrusted
  network bytes and a malformed request is ORDINARY INPUT, not one of the M6
  split's three classes; a parser that reached for `fatal()` would hand every
  client on the network a remote process kill. Three framing decisions all
  REFUSE rather than resolve: `Transfer-Encoding` → 501 by its own pass before
  Content-Length is even read; a repeated `Content-Length` refused **even when
  the values agree**; a bare LF accepted while a bare CR is not (safe HERE only
  because TE is refused outright, so the CL.TE desync has nowhere to happen).
  And `src/net.c`, the socket shim — the **only** file in the tree that knows
  whether these are Winsock or POSIX sockets, with `SO_REUSEADDR` set on POSIX
  and deliberately NOT on Windows (there it lets a *different process* take the
  port) and 127.0.0.1 as the default bind.
  **▶▶ GATE ITEM [6] IS BANKED AND IT HAS A MUTATION LEG.** 51 committed files
  under `tests/data/http_corpus/` (generator `tests/data/make_http_corpus.py`),
  each carrying its expected verdict **and its expected reason slug** in the
  FILENAME — sixteen distinct defects answer 400, so a corpus checking only the
  status would stay green against a parser that rejected all sixteen for one
  wrong cause. Each file is parsed from a heap buffer of **exactly its length**,
  because a padded array hides the one-byte over-read the corpus exists to find.
  All 51 verdicts were right first run — which is when a fixture is least
  trustworthy — so `tests/tooling/run_m6_http_mutants.sh` runs seven deliberate
  defects, one per claim, each in its own BUILDDIR: **all seven caught by the
  corpus**. The mutant's stderr banner **gates the verdict**, because a build
  that silently dropped the `-D` produces a clean binary and the leg would then
  report "the corpus did not catch it" about a mutant that was never there.
  ⚠⚠ **AND THE FIRST BARE_CR MUTANT GRADED NOTHING**: it removed only the
  bare-cr branch, but 0x0D is also `< 0x20 && != TAB` and the control-char check
  underneath rejected it anyway — so it proved only that the two REASONS differ
  while the property it claimed (a bare CR is never ACCEPTED) went untested and
  the table read GREEN. Caught by reading which corpus file caught it and what
  it actually said. **A mutation masked by a second check covering the same byte
  is a mutation that graded nothing.**
  ⚠ `.gitattributes` gained `tests/data/http_corpus/*.raw -text`: this repo has
  `core.autocrlf=true` and three fixtures exist to carry a bare LF or a CRLF
  split across a read boundary — without it git would rewrite those bytes on
  checkout and each fixture would become a different test under its old name,
  **on a fresh clone, never in the tree that wrote them**.
  Green: `mingw32-make test` rc 0 — **16 test binaries** (up from 14:
  `test_http` 60/60, `test_net` 7/7), `abi_tu`, `abi_tu_cxx`, `abi_smoke`, and
  all four goldens. **NEXT = A3**, the vertical slice: `/health` +
  non-streaming `/v1/chat/completions`, with `server.c` including `stoke.h` and
  `net.h` and nothing else from `src/`; its curl goldens are gate item [2].

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
