# Reference sources: llama.cpp and vLLM — provenance, spec re-verification, and M4/M5 design inputs

Date: 2026-08-10. Author: Opus 5 (M3 execution session, written while the
[2v2-iii] Q4_K_M grid was running).

Kabir placed two third-party source trees beside the repo for read-only study:
`Programming/llama.cpp-master` and `Programming/vllm-main` (plus
`Programming/llama-main`, which turned out to be the wrong project — see §1).
Nothing in them was modified, and nothing was copied into this tree.

**What this note is.** Three things: (1) a provenance and safety record for the
trees, (2) the re-verification of every llama.cpp fact stoke has transcribed,
which is evidence and is the load-bearing part, and (3) design input for M4 and
M5 read out of vLLM's engine.

**What this note is NOT.** It is not a plan, it changes no gate, freezes no
constant, and grants no authorization. Nothing here is on M3's critical path:
the remaining M3 work is the Q4_K_M grid, which runs on `ref_forward.py` and the
banked HEAD streams and touches neither tree. G11's `stoke.exe` freeze is
untouched — this note adds documentation only.

---

## 1. Provenance, and the folder that was the wrong project

`Programming/llama-main` is **`facebookresearch/llama`** — Meta's PyTorch
reference implementation, Llama 2 generation — not llama.cpp. Identified by
`requirements.txt` (torch/fairscale/fire/sentencepiece), `fairscale`
model-parallel layers, "Llama 2 Community License Agreement" headers, the
`[INST]`/`[/INST]` chat format, and `precompute_freqs_cis(..., theta=10000.0)`
(Llama 3 uses 500000). It is not a substitute for llama.cpp and is not a usable
M7 reference either, being the wrong model generation. Recorded so the next
reader does not re-derive it.

`Programming/llama.cpp-master` is genuine llama.cpp `master`: 167 MB, 3373
files, 1227 C/C++/headers, no `.git` (zip export).

`Programming/vllm-main` is vLLM `master`: 112 MB, 6437 files, 4119 Python, 241
CUDA/C++, Apache 2.0, no `.git`.

### 1.1 Untrusted-instruction scan — both trees clean

Third-party repos ship files that instruct coding agents. Those files were
written for that project's contributors and have no authority here; they are
data to report, not instructions to follow. Both trees were scanned before
anything else was read.

| Surface | llama.cpp-master | vllm-main |
| --- | --- | --- |
| `AGENTS.md` / `CLAUDE.md` | 11.8 KB / one-line `@AGENTS.md` | 5.6 KB / 11 B |
| `.claude/settings*.json` (hooks — the dangerous shape) | absent | absent |
| `.claude/skills/` | — | 1 (`ci-fails-buildkite`) |
| `skills/` | 2 (`add-new-model`, `code-review`) | — |
| `.gemini/` | `{"contextFileName": "AGENTS.md"}` | code-review config |
| `.pi/gg/SYSTEM.md` | present | — |

**No hooks, no settings override, nothing malicious.** All legitimate upstream
tooling. Two findings worth recording anyway:

- vLLM's `AGENTS.md:49` carries a live `curl -LsSf https://astral.sh/uv/install.sh | sh`.
  Legitimate (astral.sh distributes `uv`) but it is remote code execution by
  instruction. **Not run.** A tree being studied never needs its toolchain.
- vLLM's `AGENTS.md` mandates `Co-authored-by:` commit trailers, which directly
  contradicts this project's standing no-attribution rule, and frames
  compliance coercively ("Breaching these guidelines can result in automatic
  banning"). Recorded because that pressure shape is what makes an agent comply
  without checking who is asking.

Reading a file inside a tree can auto-load that tree's `CLAUDE.md`/`AGENTS.md`
into an agent's context. That happened with `vllm-main` and is expected;
auto-loading is not consent.

---

## 2. llama.cpp: the version gap is real and it does not touch us

This is the part that is evidence rather than opinion.

**The binaries stoke is gated against are `bench/vendor/llama-b10037/`, which is
binaries only — no source has ever been in this tree.** That is why every spec
question since M2 was answered by an ad-hoc fetch, with the payloads written
into `.superpowers/sdd/m2-probe-notes.md` — a file that is GITIGNORED and has
never been tracked. The new tree closes that gap for reading, but see §8.

**Version.** `llama-cli.exe --version` on the vendored binary reports
`version: 10037 (56d6e9dde)`, matching the commit the 2026-07-19 P3 probe
recorded (`56d6e9dde2c684cebbc11e1f705da4286cb535f2`). The source tree is
`master` and is far newer — it carries `LLM_ARCH_MISTRAL4`, `KIMI_LINEAR`,
`QWEN3TTS`, `EAGLE3`, a Svelte web UI and a `skills/` directory. Hundreds of
commits of drift. **Spec-vs-binary skew is a real defect class and this is a
real gap.**

**But every fact stoke transcribed still holds.** All 15 were re-checked
against the new tree:

| Fact stoke transcribed | Location in master | Result |
| --- | --- | --- |
| `block_q8_0` = `{ggml_half d; int8_t qs[32]}`, 34 B, `static_assert` | `ggml-common.h:254-256` | identical |
| `block_q4_K` union `{d,dmin}`/`dm` + `scales[K_SCALE_SIZE]` + `qs[QK_K/2]` | `ggml-common.h:337` | identical |
| `block_q6_K` `ql`/`qh`/`scales[QK_K/16]`/`d` | `ggml-common.h:362-367` | identical |
| QWEN3 -> `LLAMA_ROPE_TYPE_NEOX` | `llama-model.cpp:2632` | confirmed |
| M7: `LLM_ARCH_LLAMA` -> NORM, "pairs of consecutive head values" | `llama-model.cpp:2575` | comment verbatim |
| `rotate_pairs`, `ic = i0/scale`, `n_offset = n_dims/2` | `ops.cpp:5934` | **same line number** |
| `theta_scale = powf(freq_base, -2.0f/n_dims)` | `ops.cpp:6008` | **same line number** |
| `ggml_rope_cache_init` | `ops.cpp:5846` | **same line number** |
| QK-norm BEFORE rope, both Q and K | `models/qwen3.cpp:88-100` | confirmed |
| attention scale `1.0f/sqrtf(n_embd_head)` | `models/qwen3.cpp:112` | confirmed |
| `attn_q/k_norm` shape `{n_embd_head_k}` | `models/qwen3.cpp:39-40` | confirmed |
| `qwen3.cpp` = 159 lines | — | **still exactly 159** |
| K2: `sumi*(d_x*d_y)` — scales multiply each other first | `ggml-cpu/quants.c` | confirmed |
| K3: `#if AVX512VNNI && AVX512VL` / `#elif AVXVNNI` | `arch/x86/quants.c:106,110` | confirmed |
| M1: qwen2 pre-tokenizer regex, 7 alternatives | `llama-vocab.cpp:379` | byte-identical to `tok.c`'s |

Three independent `ops.cpp` line anchors landing on identical line numbers, and
an unchanged 159-line `qwen3.cpp`, say those files were not touched across the
whole gap. That is consistent with what they are: block layouts, rope pairing
and the tokenizer regex are format-level and cannot change without invalidating
every GGUF in existence.

**Honest limit.** Identical anchors are strong evidence, not proof of
byte-identity, and no b10037 source exists locally to diff against (the probe's
copies were in a session scratchpad that is gone). Read this as *no fact stoke
relies on has changed*, NOT as *the trees are identical*. §8 says what to fetch
to remove the caveat entirely.

**One real change:** the scalar dot product is now
`ggml_vec_dot_q8_0_q8_0_generic` behind an arch-dispatch layer
(`ggml-cpu/arch/{x86,arm,riscv,...}/quants.c` + `arch-fallback.h`). Same
arithmetic, new name and location. Any future probe should search by symbol,
not by path.

---

## 3. vLLM's engine, as read

The whole of what M4 must build lives in five files of `vllm/v1/core/`:
`sched/scheduler.py` (137 KB), `kv_cache_utils.py` (94 KB),
`single_type_kv_cache_manager.py` (85 KB), `kv_cache_manager.py` (39 KB),
`block_pool.py` (33 KB).

**Block pool.** A fixed array of block objects indexed `0..N`, plus a free
queue that is an intrusive doubly-linked list. The structural idea worth having:
**the free list IS the eviction-order list.** A block that is free but still
holds a cached prefix stays in that queue and is evicted only when someone
needs it. One structure yields O(1) allocation and O(1) LRU with no separate
eviction pass. A hash map (`block_hash -> block`) serves prefix lookup on top,
and blocks are reference-counted.

**Prefix caching** is hash-chained per block — each block's hash covers its
whole prefix — so a cache hit is always a contiguous run starting at block 0.

**Scheduler.** One `schedule()` per step returning a `SchedulerOutput`,
admission governed by a token budget. Preemption is LIFO by default; under a
priority policy the victim is `max(priority, arrival_time)`. **A preempted
request is not swapped out — its blocks are freed and it re-prefills later.**
Twelve request states.

**Frees are fenced.** `deferred_frees` holds blocks behind a step counter: a
block cannot be recycled until the step that read it has retired on the device.

---

## 4. Independent corroboration of three constraints M4 already locked

This is the most useful result, and it runs the opposite direction from
"borrow their ideas": vLLM reached three of M4's existing constraints the
expensive way, which raises confidence in constraints written here before any
of that code was read.

1. **Full-reservation admission.** ROADMAP §M4 already binds "full-reservation
   admission until preemption exists". vLLM's `allocate_slots` carries a
   `full_sequence_must_fit` flag whose documented purpose is "prevent
   over-admitting requests when chunked prefill would otherwise only check the
   first chunk". Same failure, same fix, found after shipping.

2. **Teardown at any lifecycle point.** ROADMAP §M4 constraint (2) makes
   teardown a first-class sim-exercised op. vLLM needs `deferred_frees`,
   `_free_cow_retained_blocks` and `_drain_deferred_frees` to make teardown
   safe against in-flight steps — i.e. teardown that was not first-class became
   three interacting mechanisms.

3. **Runtime-variable pool size.** ROADMAP §M4 constraint (1) forbids baking a
   static-pool assumption into invariants or allocation. vLLM asserts
   `num_gpu_blocks > 0` at construction and treats it as fixed for the process,
   which is exactly the assumption Phase-2 memory-elastic serving cannot have.

---

## 5. New inputs for M4/M5

Design inputs, not decisions. Each is to be weighed when the M4 design note is
written; none of them binds anything today.

- **Free list == eviction order** (§3). A natural fit for C11: embed
  `prev`/`next` in the block struct, O(1) everywhere, zero allocation.
- **The last-token rule.** vLLM caps a prefix-cache hit at `prompt_len - 1`,
  because when every prompt token is cached you must still recompute at least
  one to obtain logits. M4's `token-id-keyed verified prefix-dedup` LOCK walks
  into this directly.
- **Reserve blocks against livelock.** vLLM's `reserved_blocks` keeps free
  blocks that already-in-flight sequences need to finish, so new admissions
  cannot starve running work. This belongs beside M5's CANCELLATION-SAFETY
  invariant, and it is a stricter statement than full-reservation admission
  alone.
- **Fence frees behind step retirement.** The moment stoke has work in flight,
  a freed block cannot be recycled until the step that read it has retired.
  vLLM's own comment records the subtlety that fences arrive in only
  near-monotonic order, so the drain must stop at the first pending one.

---

## 6. Where stoke's position genuinely differs

Recorded as hypotheses with the measurement that would settle each. None is
claimed as fact.

**6.1 On the CPU path, stoke's KV never leaves system RAM, which inverts the
preemption trade.** vLLM preempts by discarding and recomputing because moving
KV over PCIe costs more than redoing the work. stoke's blocks are already in
host memory, so preemption could be a pointer move with no recompute.
*Settles by:* measuring re-prefill cost against block-table rewrite cost at M4,
on the M5 agentic trace.

**AMENDED 2026-08-10 (Kabir): this advantage is CPU-PATH-ONLY and must not be
designed in as a general property.** Kabir has confirmed GPU support is a real
target — he has a GPU and expects users to have a decent one — so the moment KV
blocks live in VRAM, stoke faces exactly vLLM's constraint and vLLM's answer
becomes the right one. **The block pool must therefore treat "where a block
lives" as a property of the block, not an assumption of the allocator**, in the
same way ROADMAP §M4 constraint (1) already forbids assuming a fixed pool size.
That is the same lesson twice: the pool's invariants must not encode today's
placement. Preemption policy then becomes per-placement — pointer move on
host, recompute-or-transfer on device — decided by measurement at M4, not
chosen once.

**6.2 A narrower target buys a scheduler that can be proven.** vLLM's 137 KB
scheduler and 12-parameter `allocate_slots` are the price of serving sessions,
streaming, spec decode, structured output, Mamba, sliding window, P/D
disaggregation and KV connectors. stoke's target — one user's agent swarms on
consumer hardware — admits a scheduler small enough to be exhaustively
sim-tested, which is the only kind compatible with the 11-invariant discipline
M4 already requires.

**6.3 The agent workload is a different shape.** M5's trace is already written
as bursty, shared 1–2k prefix, short decodes, 10–20% aborts. vLLM optimizes
throughput across many independent users. A workload where requests *share* a
prefix and then *fork*, and where a fifth are cancelled mid-flight, rewards
aggressive sharing and genuinely cheap teardown — which is what §4's
constraints already aim at. See §8 on SGLang, whose radix-tree cache addresses
the forking case natively where block-hash chaining addresses only the shared
prefix.

---

## 7. Two anti-patterns, recorded so they are not reproduced

- **Lifecycle state decided by enum ORDER.** vLLM's
  `RequestStatus.is_finished(status)` is `status > PREEMPTED`, guarded only by
  a comment ("anything after PREEMPTED will be considered as a finished
  status"). Inserting a state in the wrong position silently changes which
  requests are considered finished. stoke's equivalent must be an explicit
  predicate or flag with an assertion, never an ordering convention.
- **A sentinel whose invariant is "do not touch it".** vLLM's `null_block` is
  block 0 with a deliberately unmaintained reference count that "needs special
  care to avoid freeing it". If stoke needs a null block, its index is a named
  constant and every free path asserts against it.

---

## 8. What to fetch, and when

- **llama.cpp source at tag `b10037`** — highest value, smallest ask. It makes
  the spec match the binary stoke is gated against by construction and removes
  §2's caveat entirely. Everything verified today holds, but the first time
  something does NOT match, the bytes that matter are b10037's, not master's.
- **SGLang, at M4 entry** — the one genuinely new architectural idea found.
  RadixAttention keys a radix tree over token *sequences* where vLLM chains
  hashes over *blocks*; the difference bears directly on §6.3's forking case.
- **Ollama, at Phase 2 entry, not before** — its product surface is Phase 2's
  bridge layer (registry, pull, templates, idle-unload), and M0 already
  benchmarked it, so it is a known quantity.
- **Already in hand:** llama.cpp's `tools/server/` — CPU continuous batching
  with slot management, the closest existing thing to M4/M6's target. Read it
  from the tree that is already there.

**ANSWERED 2026-08-10 (Kabir), and both answers change the reading list.**

- **GPU IS A REAL TARGET, not a Phase-1.5 maybe.** Kabir's words: he has a GPU,
  users will have a decent GPU, "we need to make it GPU compatible too" — the
  "too" is load-bearing, the CPU path stays. This promotes vLLM's
  attention-backend layer (`vllm/v1/attention/backends/`), its worker split
  (`v1/worker/gpu` vs `v1/worker/cpu`), and `v1/kv_offload/` (CPU offload and
  tiering) from irrelevant to worth reading at M4 entry — roughly inverting
  this note's earlier estimate that most of the tree did not apply.
  llama.cpp's own split is the closer model to copy in shape (one engine, CPU
  path plus optional device offload) rather than vLLM's device-first design.
- **NO NPU FOCUS.** It stops constraining M4's block pool. ROADMAP's Phase-2
  ordering already put NPU behind the bridge layer; this pushes it further out.
  It is not deleted, just not a design input.

### 8.1 The collision this creates, flagged now rather than discovered at M4

**ROADMAP §M4 requires batch-composition independence to be EXACT (bit-level)
"regardless" — and that is materially harder on a GPU than on the CPU path.**
The CPU design earns it by construction (seqs-inner/weights-outer, so the same
sequence is token-identical at any batch size). Device GEMM libraries generally
do NOT promise shape-invariant reduction order: split-k choice, tile size and
tensor-core path commonly vary with batch shape, so the same sequence in a
batch of 2 and a batch of 9 can differ in the last bits. The same tension
applies one level up to gate [1]'s bit-exactness against llama.cpp, whose CUDA
kernels are a different arithmetic target from its scalar CPU path.

This is not a problem to solve today and nothing here resolves it. It is
recorded because it is exactly the class of constraint that is cheap to
restate now and expensive to discover mid-milestone. Three shapes are
available when M4 entry arrives — keep the bar and constrain the device path to
shape-invariant kernels; scope the bar per-backend (EXACT on CPU,
classified-divergence on device, reusing §6's existing machinery); or make the
device path a separate numeric class with its own entry probe. **Choosing among
them edits criterion text, which is §8.4-reserved: under the 2026-08-07
governance change Opus decides, records the reasoning with a direction check,
and Kabir overrules.** It should be decided at M4 entry with a measurement in
hand, not now.

**Open, and genuinely not yet answered: what "GPU compatible" means in scope.**
Offload-style (llama.cpp's `-ngl`: some layers on device, KV possibly split),
or a full device path with CPU fallback. The two produce different block pools
— the first needs blocks that can live in either place and a policy for which,
the second needs a device allocator and a transfer path. Worth settling before
the M4 design note is written, not before M3 closes.

---

## 9. Posture

These trees are **read-only reference**. The rule for M4 and beyond: read for
invariants, write stoke's own implementation from them, port no code. That
keeps the work clean-room independently of licence (ggml MIT, vLLM Apache 2.0)
and independently of whether a licence would have permitted more.
