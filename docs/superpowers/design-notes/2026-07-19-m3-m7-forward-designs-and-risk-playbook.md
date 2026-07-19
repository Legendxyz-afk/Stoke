# M3–M7 forward designs + risk playbook (Fable exit brief)

*2026-07-19. Written at the `m1` tag as the LAST Fable session on this project
(Kabir's direction: all subsequent work is Opus-driven — Fable's usage window
is closing). This document front-loads every design decision the model
allocation previously reserved for Fable at milestone entries: M4's sim
invariants, all of M5's scheduler/sim design, the M6 ABI + error-handling
mechanism, and a per-milestone catalogue of foreseeable failure modes with
their solutions. Companion: `2026-07-19-m2-op-spec-and-parity-protocol.md`
(M2, same authority). Parent spec + plan under `../specs/` and `../plans/`.*

## 0. How Opus uses this document

- Decisions marked **LOCKED** are settled — implement, don't re-litigate.
  Anything genuinely contradicted by probed reality follows the standing rule:
  the probe wins, the doc gets amended, the session ledger records why.
- Each milestone's **gate** is a mechanical checklist. Opus performs the gate
  stamp itself: run every box, record results in the session ledger and
  MEMORY.md, tag, move on. A gate passes on evidence, never on judgment.
- **Escalation (replaces "summon Fable"):** on (a) a gate failing twice for
  the same cause, (b) >1 day stuck on one divergence, (c) results
  contradicting the spec's physics reasoning, (d) ANY temptation to loosen a
  gate (ε, waiver budgets, bars, corpus trims) — STOP and write Kabir a
  half-page memo: what failed, the evidence, 2–3 options with one
  recommendation. Kabir decides (and may choose to open a Fable session; that
  is his call alone).
- Worker protocol unchanged: sonnet workers, surgical briefs, probed
  constants, mechanical byte-verification steps, Opus task-reviews.

---

## M3 — AVX2+VNNI kernels + thread pool

**Deliverables:** SIMD twins for every §M2 kernel; activation quantization;
Q4_0 + Q4_K + **Q6_K** dequant/dot; `pool.c`; per-op microbench; single-stream
≥50% of llama.cpp on the same file/box.

**LOCKED decisions:**

1. **Q6_K is in scope, not optional.** The published-benchmark file (Q4_K_M)
   stores `attn_v`/`ffn_down` on many layers AND `token_embd` as Q6_K (see
   `tests/data/qwen3-4b-info.golden`). "Q4_K_M support" = F32 + Q4_K + Q6_K.
   Plan text saying just "q4_k" undercounts; this supersedes it.
2. **Integer cores on both paths.** When weights are quantized, activations
   are quantized to Q8_0 blocks first (absmax/127 scaling), and BOTH the
   scalar and SIMD kernels run the same i32 integer core (i8×i8 products →
   i32 accumulate → scale by d_w·d_a per block). Scalar↔SIMD diff is then
   EXACT (integer), which is what the plan's "exact-diff ×1e4 random tensors"
   gate assumes. The f32-activation path from M2 remains only for F32
   tensors. Consequence: token streams may shift vs M2 — re-run the 50-prompt
   parity gate (Q8_0 file, then Q4_K_M once K-quants land) under the M2
   waiver protocol after the switch.
3. **Parallelize only the big ops.** Row-partition `matvec` for the seven
   large per-layer projections and the 151936-row logits matvec; attention
   parallelizes across heads. Small ops (norms, rope, 128-dim dots) stay
   single-thread — barrier cost exceeds the work.
4. **No malloc in the token loop.** All scratch preallocated in the model
   context at open. (Applies from M3 forever; M2 may allocate lazily but
   should already comply.)

**Foreseeable problems → solutions:**

- **The VNNI intrinsic trap.** This CPU (i5-13450HX) has AVX-VNNI (VEX
  256-bit), NOT AVX-512-VNNI. In GCC the VEX intrinsic is
  `_mm256_dpbusd_avx_epi32` under `-mavxvnni`; the identically-shaped
  `_mm256_dpbusd_epi32` requires AVX512VNNI+VL and will not compile/run here.
  Compile flags: `-mavx2 -mfma -mf16c -mavxvnni`. Runtime dispatch: check
  CPUID leaf 7 subleaf 1 EAX bit 4 (AVX-VNNI); scalar fallback always
  available (it is the reference).
- **dpbusd is unsigned×signed.** `dpbusd(a,b)` needs a unsigned u8, b signed
  i8; Q8_0 quants are signed on both sides. Use llama.cpp's sign trick —
  `dpbusd(_mm256_abs_epi8(x), _mm256_sign_epi8(y, x))` — or bias one side and
  subtract the correction. PROBE the exact idiom from fetched ggml source at
  dispatch (vendor zip is binaries-only; WebFetch the b10037 tree) and pin it
  with the exact-diff gate rather than trusting either memory or this note.
- **K-quant super-block layouts.** Expected (PROBE from ggml-quants source +
  verify on real tensor bytes + independent python dequant BEFORE any C):
  Q4_K = 256-elem super-block {f16 d, dmin; u8 scales[12] (6-bit packed
  scale/min pairs); u8 qs[128]} = 144B; Q6_K = {ql[128], qh[64], i8
  scales[16], f16 d} = 210B. The 6-bit scale/min packing in Q4_K
  (`get_scale_min_k4`) is the classic transcription bug — the mechanical
  check is: python dequant (from source formulas) == C scalar dequant,
  byte-exact f32, on real rows of all three K-quant tensor kinds, before the
  SIMD twin exists.
- **P/E-core straggling.** 6P+4E: a static even row split leaves P-cores
  idling at the barrier. Start with work-stealing chunks (chunk = 64 rows,
  atomic next-chunk counter — trivial and P/E-proof); measure a proportional
  static split only if the counter shows contention (it won't at 10 threads).
  Thread pinning experiment per plan: measure P-only (6T) / P+E (10T) /
  SMT (12–16T) on the microbench, pick per-op defaults, record in the ledger.
- **Missing the ≥50% bar.** Attribution FIRST, guessing never: mandate a
  `stoke bench --ops` microbench (W1 of M3) reporting per-op GB/s vs the
  ~60GB/s practical stream ceiling. The culprit ranks in this order in
  practice: (1) logits matvec not parallelized or embd rows dequanted
  per-call, (2) K/V layout breaking sequential streams in attention, (3)
  f16→f32 conversions in the inner loop (use `_mm256_cvtph_ps` via F16C),
  (4) barrier/pool overhead on small ops (don't parallelize them), (5) page
  faults during timing (warm the mmap: touch every weight page once before
  the timed region — M0 protocol already requires warmup requests).
- **Bench fairness.** The 50% compare is stoke-best vs llama.cpp-best (its
  own tuned threads), same file, same prompts, decode-only tok/s, median of
  3, quiesced box — reuse the M0 machine checklist verbatim.

**GATE (mechanical):** [1] scalar↔SIMD exact-diff (integer paths) / ≤1e-5
rel (f32 paths) on 1e4 random tensors per kernel, all quant kinds; [2]
50-prompt parity re-passes on Q8_0 AND Q4_K_M files (waiver protocol as M2);
[3] single-stream ≥7.8 tok/s (=50% of llama.cpp's 15.6 M0 number — if
llama.cpp is re-benched at gate time, the bar re-anchors to that number) on
Q4_K_M; [4] both-platform matrix green incl. tsan (pool lands here); [5]
microbench table recorded in ledger + `bench/history.jsonl` row appended.

---

## M4 — paged KV + static-batch decode

**Deliverables:** `kv.c` block pool + per-seq block tables + refcount/CoW +
prefix sharing + Q8_0-KV option; batched decode over fixed N; the alloc/free
randomized sim; scaling curve N∈{1..16}.

**LOCKED decisions:**

1. **Block = 16 tokens default** (measure 8/32 on the curve, keep 16 unless
   >5% aggregate delta). Physical block: per layer, per kv-head, 16×128 f32
   (or Q8_0 blocks when KV-quant on). `kv_view` (M2 note §6) keeps its
   signature; the M4 body resolves (layer, kv_head, pos) through the block
   table. Attention math does not change — that was the point of the
   accessor.
2. **Batch composition independence by construction.** Batched ops loop
   seqs-inner/weights-outer so each weight block streams once per step, but
   every sequence's arithmetic (order, operands) is IDENTICAL to its
   single-seq execution. Therefore: same seq + same prompt ⇒ token-identical
   output at any batch size. This is a hard correctness property, cheap to
   hold on CPU, that vLLM-class engines give up — and it makes M5 debugging
   sane (a scheduling bug can never masquerade as a numeric one).
3. **Full-reservation admission at M4.** No preemption exists yet, so a seq
   is admitted to the static batch only if worst-case blocks
   (ceil((prompt+max_new)/16)) fit. Loud refusal otherwise. Optimistic
   admission arrives WITH preemption at M5.
4. **Token-parity gates always run f32 KV.** Q8_0 KV changes outputs
   (quantization in the attention path), so it is benchmarked and
   property-tested, never token-parity-gated. Acceptance for Q8_0-KV =
   (a) property bound: |dequant(quant(x))−x| ≤ d/2 per element, (b) the
   50-prompt corpus run f32-KV vs q8-KV with the token-divergence count
   RECORDED (expect small drift; there is no exactness claim), (c) default
   ON/OFF for the published benchmark is **Kabir's call at M5** with those
   numbers in front of him.

**The sim invariant list** (the randomized alloc/free simulation, ≥1e6 ops
under ASan, seeded/deterministic — these are the checks, all asserted after
EVERY sim op):

1. Conservation: `free + Σ live unique physical = total` blocks, always.
2. `refcount(b)` == exact number of block-table entries referencing b.
3. No two seqs share b unless their logical prefixes over b's span are
   identical (sharing arises only from prefix match).
4. A block with refcount>1 is never written — every write to a shared block
   CoWs first (assert trap on violation).
5. CoW postcondition: old refcount decremented, new block refcount==1,
   contents byte-equal at the copy instant.
6. No freed block is reachable from any table; no allocated block is on the
   free list.
7. Allocation fails ⇔ free list is empty; never spuriously.
8. Per-seq logical positions are dense: full blocks then one partial tail;
   tail fill == seq_len mod 16.
9. On seq release: exclusive blocks return to the free list, shared blocks
   decrement only.
10. Determinism: same seed ⇒ same final state hash (catches iteration-order
    nondeterminism before TSan ever sees it).
11. (With KV-quant on) quant/dequant round-trip bound of decision 4a holds
    for every written block.

**Foreseeable problems → solutions:**

- **Scaling curve collapses past N=8 like the baselines.** Ranked causes:
  (1) KV walks lose locality — keep each (layer,head) block's 16×128 rows
  contiguous and iterate pos-runs within a block before chasing the table;
  (2) per-seq scratch false sharing — pad per-seq output rows to 64B; (3)
  the logits matvec dominating per-seq (it's per-token per-seq — at N=16
  that's 16 × 389M MACs/step; it parallelizes perfectly, confirm the pool
  saturates); (4) genuine bandwidth saturation — the curve may PLATEAU
  (fine; the gate demands monotone non-decreasing aggregate, not linear
  gain). If aggregate DROPS with N, it is software, not physics — find it.
- **Prefix-sharing correctness.** Byte-level dedup verify per plan: two seqs
  with a shared 1024-token prefix must show identical physical block ids for
  the shared span (assert in test), and outputs must equal the no-sharing
  run token-exactly (composition independence makes this exact).
- **CoW subtlety: the tail block.** Divergence happens mid-block; CoW copies
  the shared partial tail then writes diverge. The sim's invariant 5 covers
  the mechanics; the model-level test is: two seqs share prompt, generate
  different continuations (different sampling caps), both match their
  single-seq references.
- **Accounting drift under mixed alloc/free/CoW storms** is exactly what
  invariants 1/2/7 catch — do not weaken them to "eventually consistent";
  they hold after every op or the design is wrong.

**GATE (mechanical):** [1] sim ≥1e6 ops green under ASan, all 11 invariants,
3 seeds; [2] batch-of-N token-identity vs single-seq on 5 prompts × N∈{2,8};
[3] prefix-dedup byte-verify; [4] scaling curve measured N∈{1..16},
aggregate monotone non-decreasing, curve committed to `bench/history.jsonl`
+ ledger; [5] f32-vs-q8 KV divergence count recorded; [6] matrix green.

---

## M5 — continuous batching + chunked prefill (THE product gate)

**Deliverables:** `sched.c`; the scheduler SIMULATION harness (built FIRST);
MPSC submit queue; per-request output rings; preempt-recompute; chunked
prefill; the spec's headline gate.

**LOCKED scheduler design:**

- **States:** WAITING (FIFO arrival queue) → PREFILLING (at most ONE request
  holds prefill-active status at a time) → RUNNING (decode set) → FINISHED
  (eos | max_tokens | cancel | error). MPSC submit queue drained at step top.
- **Step loop (iteration-level):** (1) drain submits → WAITING; (2) retire
  finished, free blocks; (3) if no active prefill and WAITING nonempty and
  watermark holds → activate oldest (FCFS); (4) build step batch = ALL
  runnable decodes + ≤1 chunk (128 tokens) of the active prefill; (5) fused
  batched forward (prefill chunk rides the same weight streams as decodes —
  that co-execution IS the design; it is extra rows in the same op walk, not
  a separate pass); (6) sample decodes' next tokens; a prefill whose last
  chunk completed samples its first token and joins RUNNING; (7) emit ids to
  per-request SPSC rings.
- **Admission watermark:** admit a prefill only if
  `free_blocks ≥ blocks(prompt) + W`, W = max(8, 2·|RUNNING|) blocks
  (reserve decode headroom). W is a named constant; the sim sweeps it; the
  bench may tune it; changing its FORM (not value) is a design change → memo
  to Kabir.
- **Preemption (pool exhaustion at decode-alloc time):** victim = YOUNGEST
  RUNNING (LIFO), matching FCFS fairness — the oldest request is never
  preempted, which with finite work gives starvation-freedom by induction.
  Preempt = free all victim blocks, retain its prompt + generated-so-far ids,
  push to FRONT of WAITING (it re-admits before any newer arrival).
  Readmission = re-prefill prompt+generated verbatim (recompute), then
  continue. **Replay rule:** recompute REPLAYS recorded ids — it never
  re-samples. Under greedy this is provably identical; under M6's seeded
  samplers it stays identical BECAUSE we replay (write the rule into sched.c
  now; it is load-bearing for M6).
- **Chunked prefill:** chunk = 128 (constant, sim/bench may justify 64).
  Every runnable decode decodes EVERY step — a chunk co-runs, it never
  displaces; assert literally in the sim (no RUNNING seq skips a step while
  runnable). That assertion is the ITL guarantee's skeleton; the ≤250ms wall
  bound is then just step-time budget, measured at the bench.
- **Rings:** per-request SPSC (scheduler produces, server/API consumes),
  ids + finish reason only — detokenization/stop-strings live on the
  consumer side (M6), NEVER on the scheduler thread. Submit queue: plain
  mutex+cv (arrivals are rare relative to steps; do not get clever). Rings:
  C11 atomics, acquire/release on head/tail, power-of-2 capacity. Nothing
  fancier — TSan must come up clean and stay clean.
- **Cancel:** a flag polled at step boundaries; frees blocks next step-top.

**The sim harness (built BEFORE sched.c's real integration, plan mandate):**
event-driven, virtual clock, pluggable step-cost model (constant; then
linear in batch+ctx to emulate physics), fake model (next token =
hash(req_id, pos) — deterministic, instant), seeded arrival generators:
Poisson mixes, all-at-once bursts (64 reqs), one-giant-prompt + many-small,
cancel storms, max_tokens=1 spam, and an adversarial memory-pressure trace
(arrivals sized to force preemption repeatedly). After EVERY sim step assert:
all 11 M4 invariants; watermark honored; ≤1 active prefill; FCFS admission
order; victim-is-youngest; every-runnable-decodes-every-step; preempt-replay
losslessness (outputs == the same trace run with an infinite pool — the
reference run); all requests finish (starvation); preemption count reported
per trace (a thrash metric, not a gate — but record it).

**Foreseeable problems → solutions:**

- **The bar is missed (≥17.7 @8 / ≥14.2 @16 and 2×-ollama 10.8/16.8).**
  Attribution ladder, in order: (1) M4's static curve is the ceiling proof —
  if static N=8 already missed the bar, the scheduler was never the problem;
  return to M3/M4 with the microbench (this is why M4 measures the curve
  before sched.c exists). (2) Scheduler overhead: run the sim's fake model
  under the REAL clock — step bookkeeping must be <2% of a real step budget
  (~50ms-class); it is bookkeeping over ≤16 requests, it has no excuse not
  to be. (3) Chunk co-run blows ITL/decode budget: drop chunk to 64; if
  still, the fused step isn't actually sharing weight streams — verify the
  prefill rows ride the same op walk. (4) Preemption thrash under S3-style
  load: raise W; the sim's thrash metric localizes it. (5) Detok/stop-string
  work crept onto the scheduler thread: it must not (design above).
- **TSan noise vs real races.** Keep the concurrency surface exactly three
  objects (submit queue, rings, a stop flag). Any TSan report is REAL until
  proven otherwise (no suppressions committed, doctrine unchanged). The
  determinism invariant (same trace ⇒ same outputs) catches ordering bugs
  TSan can't see.
- **ollama-default's N=16 inversion** (M0's unexplained 16.8 bar): the 2×
  bar at N=16 exceeds every measured baseline — it is the honest but hard
  bar. If stoke clears 17.7@8 and 14.2@16 (llamacpp-tuned parity) but not
  16.8@16, that is a RESULTS-vs-physics finding: write the memo to Kabir
  with the curve rather than grinding blind — the 2× framing may need
  Kabir-level re-anchoring against the inversion anomaly (it is his
  headline, his call).
- **Soak instability** (1h randomized ASan+TSan): use the M0 ops lesson —
  detached/Task-Scheduler execution for multi-hour runs (session-tracked
  background tasks get reaped); resumable runner; two writers never share a
  log handle on Windows.

**GATE (mechanical):** [1] sim: all traces green, all asserts, 3 seeds ×
{constant, linear} cost models; [2] real: S2 aggregate ≥17.7 @8 AND ≥14.2
@16 AND ≥10.8 @8 / ≥16.8 @16 (both families of bars) on Q4_K_M, M0
methodology (medians, pinned, quiesced, same-file); [3] ITL: no decode stall
>250ms during co-running prefill (harness percentiles + max); [4] S3
fanout ≥ llamacpp-default's 13.3 (the static-slot-beating proof); [5] 1h
randomized soak ASan+TSan clean; [6] preempt-lossless demonstrated on a
forced-preemption real-model trace; [7] history.jsonl + ledger + MEMORY.md
updated. Q8-KV default decision memo → Kabir with §M4's divergence numbers.

---

## M6 — `stoke.h` freeze + server

**Deliverables:** final C ABI; the error-handling split (below); `server.c`
HTTP/1.1+SSE; OpenAI endpoints; `stoke serve`; openai-python + curl golden
gate; sampling set (temperature/top-k/top-p/repeat-penalty/stop/seed).

**LOCKED — the `fatal()` → library-error mechanism** (this supersedes the
"decide at M6 entry" deferral in the M2 note §12; decided now so no design
authority is needed later):

1. **Three classes of failure, three behaviors:**
   - **Load-time failures** (bad path, malformed GGUF, unsupported arch,
     OOM during open): TRAPPED. `stoke_open` arms a thread-local
     `jmp_buf`; `fatal()` checks it and `longjmp`s back; `stoke_open`
     returns NULL and the message is captured in a thread-local error
     buffer readable via `stoke_last_error()`. Partial-construction cleanup:
     the under-construction handle tracks its own allocations (everything
     already hangs off the GgufFile/Tok/Model structs); the trap path calls
     the ordinary `*_free` teardown on whatever is non-NULL. The x*
     allocation wrappers gain a trap-aware OOM path (they already route
     through one choke point).
   - **API misuse** (NULL handle, bad request id, zero cap, submit after
     close): explicit validated-argument error RETURNS at the ABI boundary
     — enumerable, no trap involved. Enum `STOKE_ERR_*` + message.
   - **Internal invariant violations** (broken block accounting, impossible
     states — genuine bugs): keep `fatal()`'s abort, but route the message
     through `stoke_set_log_callback()` first so a host application can
     capture it. Aborting a host on OUR bug is defensible (SQLite-style);
     silently corrupting their inference is not. Runtime malloc OOM during
     serving lands here for v1 (documented); KV exhaustion is NOT an error
     at all — the scheduler preempts, that's the M5 design.
2. **No `fatal()` call sites change.** The choke point in common.h gains the
   trap check; the ~90 sites stay as they are. CLI paths (no trap armed)
   behave exactly as today.
3. **ABI freeze checklist** (the review Fable would have done — run it as a
   mechanical pass): opaque pointers only, no structs by value across the
   boundary; every symbol `stoke_`-prefixed; `stoke_abi_version()` +
   compile-time macro, checked at open; caller-owns-buffer with explicit
   caps everywhere (the tok.h pattern); UTF-8 in/out; per-function
   thread-safety stated in the header (submit/poll/cancel: safe;
   open/close: external synchronization); no varargs, no enums in structs
   whose size matters, errors never renumbered after the tag; header
   compiles standalone as C11 AND C++ (extern "C" guard) — add a test TU
   compiled as C++ to the matrix; every public function documented in the
   header itself (the header is the API doc, README links it).

**Server design + foreseeable problems:**

- Own HTTP/1.1: bounded request line/headers (loud 431/413), only the three
  endpoints + `/health`, Content-Length bodies (no inbound chunked at v1 —
  clients don't send it), CRLF-robust parser with a garbage-input unit
  corpus (fuzz-ish, committed); winsock (WSAStartup, closesocket) vs POSIX
  behind a 50-line socket shim — both legs already in the test matrix.
- SSE: heartbeat comment `: ping` every 15s (idle streams die at proxies
  otherwise); `data: {...}\n\n` framing; final `data: [DONE]`; slow-client
  policy: per-conn bounded buffer, on overflow disconnect that client and
  cancel its request (never block the scheduler on a socket — the rings
  decouple; the server thread drains rings, sockets apply backpressure only
  to their own buffer).
- OpenAI compat traps (test against openai-python, not the docs): first
  stream delta carries `role`; `finish_reason` null-until-final then
  `stop`/`length`; `usage` only when `stream_options.include_usage` (and in
  non-stream responses always); model name echoed; ChatML application =
  existing `tok_apply_chatml`; `/v1/completions` is raw (no template) —
  matches the M2 parity path. **The definitive M6 smoke: the M0 bench
  harness runs against a `stoke` adapter UNCHANGED** — it already speaks
  OpenAI-SSE and counts tokens strictly; if it runs, the compat surface is
  real. Write the adapter, point the harness, that's the gate's core.
- Stop strings cross token boundaries: consumer-side (server) holds a
  tail window of decoded text per request (window = max stop-string bytes −
  1), checks after each token, cancels via the M5 flag on hit, drops
  overshoot tokens from the emitted stream. Sampling with seed: per-request
  RNG (splitmix64 seeded), sampled on the scheduler thread (cheap vs
  forward); preemption replay (M5 rule) keeps seeded streams identical.

**GATE (mechanical):** [1] openai-python suite (stream + non-stream chat,
completions, models) green; [2] curl goldens committed + matching; [3] M0
harness drives stoke via its adapter end-to-end on S1 without modification;
[4] ABI checklist above all-boxes; [5] C++-TU compile test green; [6] server
parser garbage-corpus green; [7] matrix + soak-smoke (10-min randomized
serve under ASan) green; then the header is TAGGED FROZEN.

---

## M7 — harden + launch-ready

**Deliverables:** Llama-3.2-3B end-to-end; unsloth-file parity run; fuzz
expansion; 24h soak; README + reproducible benchmark report; name decision.

**Foreseeable problems → solutions:**

- **Llama-3.2 is a different animal in four specific ways** (probe ALL from
  its GGUF + fetched source at dispatch; every one is a silent-garbage trap
  if assumed): (1) no QK-norm (skip those ops when tensors absent — make
  their presence conditional on config, not arch string matching); (2)
  rope differs TWICE (source-confirmed at the M2 probe, b10037
  `llama_model_rope_type`): pairing is `LLAMA_ROPE_TYPE_NORM` — ADJACENT
  pairs (2k, 2k+1), NOT qwen3's NEOX (i, i+half); convert_hf permutes Q/K
  at GGUF-conversion so NORM-on-GGUF ≡ HF rotate_half, and stoke reads the
  permuted file — AND it is `llama3`-scaled: freq_base 500000 PLUS the
  llama3 smooth frequency-scaling function over `rope.scaling.*` keys
  (factor 32, low/high freq factors, original context length) — fetch the
  exact freq-factors formula at dispatch; wrong pairing or missing scaling
  produces coherent-then-degrading output that WILL pass short tests and
  fail long ones, so the parity corpus for this model must include
  ≥1k-token contexts; (3) tokenizer
  `pre` is llama-bpe, a DIFFERENT regex (case-insensitive contractions,
  `\p{N}{1,3}` digit runs, different punctuation classes) — tok.c grows a
  per-`pre` regex table (second entry), the unicode table may need
  additional categories; oracle-gate byte-exact against llama-tokenize with
  THAT model exactly as M1 did (corpus reuse + template-specific lines);
  its specials/template are the Llama-3 header format (`<|start_header_id|>`
  etc.) — a second template function beside ChatML, probed ids, golden'd;
  (4) tied embeddings again but DIFFERENT head_dim derivation (3B: 3072/24
  = 128 — the quotient works there; keep reading key_length anyway, it is
  present and authoritative).
- **unsloth-file parity**: same weights, different GGUF writer — the risk is
  metadata deltas (key presence, token_type differences, added tokens).
  Run `stoke info` diff first, then the full parity corpus. Any delta is a
  finding to record, not to paper over.
- **24h soak**: Task Scheduler detached runner (the reaper lesson is
  recorded in MEMORY.md — session-tracked background tasks are killed);
  resumable; single log writer; capture RSS over time (leak watch: RSS
  slope ~0 after warmup; ASan-leak leg on WSL for a shorter 2h window).
- **README + report**: the reproducible-benchmark bar is "a stranger with
  the repo + the two GGUFs reproduces the table with one command" — the
  matrix runner already exists (`bench/run_matrix.cmd` + harness); the
  report generator formats it; pin exact versions of ollama/llama.cpp
  compared against, and RE-RUN baselines at M7 (their versions will have
  moved since M0; the claim must be against current, dated versions).
- **Name decision (Kabir)**: present the M7-current collision facts
  (PyPI/StanfordPL/Stoke-Space, GitHub/crates availability re-checked) in
  one short memo before README work begins — renaming after README/report
  polish wastes it.

**GATE:** [1] Llama-3.2 tokenizer byte-exact + 50-prompt parity (its own
corpus incl. long-context) + template goldens; [2] unsloth-file parity run
recorded; [3] 24h soak clean (no crash, no leak slope, no stall); [4]
benchmark report reproduced from clean checkout by script; [5] README
complete (build, embed-as-library example, server quickstart, honest
numbers vs dated versions); [6] Kabir sign-offs: name, Q8-KV default,
launch.

---

## Cross-cutting (applies every milestone)

- **RAM ceilings on the 16GB box:** Q8_0 parity file (4.3GB) is for
  single/low-N correctness runs; ALL concurrency benching stays on Q4_K_M
  (2.4GB) — at N=16 with f32 KV the Q8_0 file + pool would not fit. KV pool
  sizing: (RAM − model − ~4GB OS/headroom); spec's ~6GB pool ≈ 85k Q8-KV
  tokens stands.
- **Two GGUFs now exist** (Q4_K_M dev + Q8_0 parity; M7 adds Llama-3.2 and
  unsloth twins). Every gate script names its file EXPLICITLY; nothing
  defaults to "the model".
- **Never commit a model; never cite a git-ignored path from committed
  files** (the M1 fix waves enforced this — keep it true; the two finder
  regexes live in the fix-wave ledger and can be re-run any time).
- **history.jsonl discipline:** every green commit touching a hot path
  appends a row (op microbench or scenario snapshot). Regressions get
  caught by diffing history, not by memory.
- **Windows/WSL matrix is the floor, not the ceiling:** anything
  concurrency-touching (M3 pool onward) runs tsan on every commit, no
  exceptions, even when slow.
- **Shell gates are LF-pinned** (`.gitattributes: *.sh text eol=lf`, added
  at the M1 fix wave after a Windows-side edit shipped CRLF gates): WSL
  dash fatals on CRLF (`set: Illegal option -`) while msys sh masks it, and
  Git-Bash-side `grep`/`cat` can HIDE the `\r` (text-mode mount) — verify
  line endings from the WSL side (`od -c`) when a gate script misbehaves
  only on Linux. New scripts: POSIX sh, LF, no bashisms.
- **Probe-don't-recall applies to THIS document.** Where it states layouts,
  intrinsics, formulas, or protocol shapes, they are hypotheses of record —
  cheap to verify at dispatch, expensive to trust blindly. The probe wins
  conflicts, and the amendment gets one line in the ledger.

## Allocation amendment (supersedes prior model-allocation text)

Effective at the `m1` tag (2026-07-19, Kabir's direction): **Opus 4.8 owns
all remaining milestones M2→M7 end-to-end** — sessions, briefs, reviews,
gate stamps — driving from this document + the M2 note. Fable is not in the
loop; open questions that previously escalated to Fable become written memos
to Kabir (§0). Kabir may open a Fable session ad hoc; nothing in the plan
depends on one existing.
