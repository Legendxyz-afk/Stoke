# stoke — Phase 1 roadmap (the operational map)

*Updated 2026-08-11, M3 CLOSED (tag `m3`) — M4 is now. M3 closed with the
flip tripwire FLAGGED rather than quiet: see its row and the anomaly memo;
the zero-BUG criterion passed on its own terms. Previously updated
2026-08-02, M2 CLOSED (tag `m2`). Amended 2026-08-02
by a Kabir-launched Fable strategic review (additions marked "Fable
2026-08-02"; all tightenings, no gate loosened). This is the FIRST document a working
session reads. It is a map, not the territory: every milestone's binding
detail lives in the linked spec/plan/design-note sections, and on any
conflict the linked document wins. Master record + history:
`../MEMORY.md`. Exact current position + in-flight state: the final
NEXT-SESSION block of `.superpowers/sdd/progress.md`.*

**What stoke is:** a small, embeddable, from-scratch C11 LLM inference
engine whose architecture is concurrent-first — paged KV + continuous
batching + a real scheduler for the bandwidth-bound consumer hardware
vLLM abandons and llama.cpp/Ollama serve single-user. Library-first:
`libstoke.a` + `stoke.h`; the OpenAI-compatible server is a thin frontend.
The concurrency is FOR agents (Fable 08-02): on a consumer box N=8/16 is
one user's swarms, tool chains, and fan-outs (AySen `local()` at
Phase 3), not eight humans — so shared prefixes, bursty arrivals, and
cheap cancellation are architecture, not server QoL.

**Phase 1 is done when** (spec "Phase-1 goal", measured on the reference
box): tokenizer + greedy parity proven against llama.cpp; S2 aggregate
≥2× ollama-default AND ≥1.0× llama-server-tuned at N=8/16 on the same
GGUF; single-stream ≥50% of llama.cpp; no >250ms decode stall from a
co-running prefill; one static binary; OpenAI endpoints pass
openai-python; ASan/TSan green throughout; README with a reproducible
benchmark report.

## Governance (Kabir, 2026-07-19)

- **Every session is Opus (currently Opus 5)** — sessions, probes,
  briefs, worker dispatch, task-reviews, and GATE STAMPS, M2 through M7.
  The stamps are mechanical checklists; a gate passes on recorded
  evidence, never on judgment. **Model check is the first action of every
  session** — as of 2026-07-27 Kabir set the user-settings default to
  Opus 5 (max effort), superseding the 07-19 Opus 4.8 owner line, so a
  fresh session should open on the owner model; if it did not, run
  `/model claude-opus-5` before any stoke work. Owner is a FLOOR, not a
  pin: a newer Opus is a valid owner, anything below Opus is not. A
  non-Opus session is valid only on Kabir's explicit in-session
  override, and its job is manager-tier state work, not milestone
  execution.
- **Escalation (ROUTING CHANGED, Kabir 2026-07-28): major/design decisions
  go to FABLE, not a memo to Kabir.** The manager STOPS at the decision
  point, writes the decision brief (problem, PROBED constants, real
  trade-offs, a recommendation Fable may overrule, what is already Kabir's
  and not revisitable), parks it in the repo/ledger — and **Kabir launches
  the Fable session himself; NEVER self-dispatch** (a `model:"fable"` Agent
  override was silently ignored once and produced Sonnet rulings under
  Fable's name; verify from transcripts whenever it matters). Triggers
  unchanged: a gate failing twice for the same cause; >1 day stuck on one
  divergence; results contradicting the spec's physics; ANY temptation to
  loosen a gate (ε, waiver budget, bars, corpus trims) — and a bar fixed
  AFTER seeing results counts, so gate definitions are referred BEFORE the
  run. Kabir still owns: push policy, the model/owner line, and anything he
  asks for directly.
- **Workers**: sonnet, surgical briefs, constants PROBED at dispatch
  (never recalled — applies to the design notes themselves), a mechanical
  verification step in every brief, Opus task-review per task. Background
  workers die at session boundaries — recovery = check `git log` for the
  expected commit, else re-dispatch from the brief.

## The map

| # | Milestone | Status | Gate (short form) |
|---|---|---|---|
| M0 | Measure the enemy | ✅ 2026-07-17 `8ce4fd8` | Premise: best baseline 11% of ceiling (kill ≥70%) — GO |
| M1 | gguf.c + tok.c | ✅ 2026-07-19 tag `m1` = `cbf1e20` | Byte-exact vs llama-tokenize: fuzz corpus, specials both modes, 22,848-line unicode sweep |
| M2 | Scalar forward pass | ✅ 2026-08-02 tag `m2` | **PASS both legs: 24/24 token-exact (1,536 greedy decisions, 0 diverged, 0 of 2 waivers drawn, 0 tooling errors) + L2 36/36 ≤1e-3, worst L34 6.104e-04, no STEP.** Re-scoped 07-28 (Kabir, memo option 3) to C-vs-ref on a stratified subset + the 36-layer ladder; definition fixed beforehand by the Fable ruling. Report: `superpowers/gate-reports/2026-08-02-m2-parity.md` |
| M3 | AVX2+VNNI kernels + pool | ✅ 2026-08-11 tag `m3` | **[2v2] ZERO BUG-CLASS: 9/9 divergences ARITHMETIC-CLASS on the Q4_K_M grid (worst margin 0.555672 vs τ 1.023739 = 54% of threshold, no ESCALATE); τ_Q4KM frozen BEFORE grading under key `f32_intcore_q4_k_m` = 1.023739 / 1.178094 / 0.811432; 290 covered positions, per-prompt stated; p41 clean 64/64. Q8_0 discharged by the 08-06 ruling.** [1] test_simd 12/12 ✅; [3] re-anchored 8.08–8.46 both legs ✅; [4] matrix green at `5051878` ✅; [5] ✅. **⚠ FLAGGED, NOT QUIET — the flip tripwire FIRED (9 ≥ 8).** Its λ inherited a ~0.5%/token base measured on **Q8_0**; Q4_K_M measures **3.10%/token**, so the threshold fires by construction on this file. Kept at ≥8 rather than recalibrated from the data it polices (circular); the mismatch was committed in `6b484b9` at 5 flips, before the count could be known. Disposition: `superpowers/memos/2026-08-11-m3-tripwire-anomaly-memo.md`. Report: `superpowers/gate-reports/2026-08-11-m3-q4km-grid-gate-report.md` |
| **M4** | **Paged KV + static batch** | **✅ 2026-08-13 tag `m4`** | **THE GATE LIST IS RULING §8.4's FIVE — cite it, never a summary (a four-item paraphrase of this row cost a session):** [1] invariant sim ×1e6 ASan ✅ (set ENUMERATED in `superpowers/design-notes/2026-08-11-m4-invariants.md`); [2] batch identity N∈{1,2,4,8,16} real model ✅ 08-12; [3] prefix-dedup ✅ 08-12; [4] Q8-KV ON/OFF per §8.1 — **zero BUG-class ✅, class-rate bound MISSED (38/50, 1.1875%/token) and DISPOSITIONED per §8.1 as a results-vs-physics memo, not ground green**; [5] monotone N∈{1..16} curve ✅ 08-12 **PASS at 1.116× — see M4.5** |
| **M4.5** | **Kernel ladder → llama-band** | **← NOW** | **Authority: `superpowers/memos/2026-08-12-throughput-parity-RULING-FABLE.md`.** Order B → C → A′ → D(conditional) → L4(profiled). **P-BAR: N=1 decode ≥0.85× llama-bench tg** (same box/file/threads, referent re-anchored per §8.2; 0.95× iff D lands). C = CONTRACT v2 re-anchor (canonical f32 fold; integer core freely regroupable), graded old-vs-new by §6, zero BUG-class, **Q8_0 file streams BYTE-IDENTICAL through C or it is a bug** |
| M5 | Continuous batching | queued | **THE bars: ≥17.7 @8 / ≥14.2 @16 AND ≥10.8 @8 / ≥16.8 @16**; ITL <250ms; S3 ≥13.3; 1h soak; cancel-safe sim (Fable 08-02). **Bars are reachable through M4.5's kernels, not batching — measured 08-12 (curve flat at 8.7, compute-bound at ~64 GFLOP/s)** |
| M6 | `stoke.h` freeze + server | queued | openai-python suite; M0 harness drives stoke unmodified; ABI checklist; C++ TU |
| M7 | Harden + launch-ready | queued | Llama-3.2 parity incl. long-ctx; unsloth-file run; 24h soak; reproducible report; Kabir sign-offs |

Milestone entry = prior gate stamped. Full gate checklists are mechanical
lists in the design notes — run every box, record in the ledger +
MEMORY.md, tag if the milestone says so, move on.

## Per-milestone: where the truth lives

**M2 — scalar forward pass (✅ CLOSED 2026-08-02, tag `m2`).** Gate report:
`superpowers/gate-reports/2026-08-02-m2-parity.md`. Operative doc:
`superpowers/design-notes/2026-07-19-m2-op-spec-and-parity-protocol.md`
(op-by-op spec §5–6, config contract §3 with TRAP #1 head_dim=128-not-80 /
TRAP #2 tied-LM-head / TRAP #3 NEOX-pairing, `stoke run` CLI §7, gate +
waiver protocol §8, L0–L3 parity-debug ladder + numpy `ref_forward.py` §9,
dispatch probe pack P1–P5 §10, workers W1–W4 §11). Position 2026-07-27 (W3 landed, W4 in flight):
entry chores + P2/P3 probes (`6f9109b`); parity corpus (`f5d99ee` +
`af18af8`); W2 kernels + L0 (`7fbdd2b`); Q8_0 gguf support verified +
committed incl. the `stoke info` type-name completion (`5f97583`); W1
tooling committed with its oracle/ref validation captures (`21d3842`);
P1/P5 stoke-side legs closed. §8's oracle question SETTLED by
experiment: f16-KV hypothesis refuted, **invocation stays default**; the
one ref-vs-oracle divergence is a characterized 3-way near-tie (both
sides' evidence in the ledger) whose gate-run policy is Kabir's call —
memo `superpowers/memos/2026-07-20-m2-oracle-near-tie-policy.md`
[BLOCKING: the recorded gate run only]. W3 model.c LANDED `79a4b29`
(forward pass + kv + `stoke run`, L2 36-layer <1e-3, greedy 64/64 on
v1–v3, both platforms green, Opus-reviewed). W4 gate harness LANDED
`6f54801` (`tests/tooling/parity_gate.py` + runbook; selftest/dry-run/
matrix all green, Opus-reviewed, one manager fix: a cp437-unmappable `§`
in a normal print path aborted runs under a Windows console codepage
while `--selftest` passed under all of them). Oracle context CAPPED at
`-c 2048` — Kabir-approved `a03748a` on a byte-identical proof (v1/v2/v3
64/64 ids unchanged; 11,038 MB → 501 MB private, health 69 s → 8 s),
memo `superpowers/memos/2026-07-27-m2-oracle-context-cap.md`; without it
the survey needed ~20 GB on a 16 GB box. The 07-27 survey then died with
`p40-p43` recorded as tooling ERRORs — **not a stoke fault**: the harness's
flat `STOKE_TIMEOUT_S = 1800 s` sat below the long-context tier's own
documented ~49 min cost, so it killed exactly the four longest prompts at
the ceiling (the run's separate death was a hand shutdown, per the event
log). Fixed `5e0d484`: the ceiling is now DERIVED per prompt
(`max(1800 s, (bytes/2.5 + n_predict) × 12 s)`) off a fresh measurement —
**5.20 s/forward at `--jobs 1`, 6.07 s at `--jobs 4`** (contention only
1.17×), so p40 legitimately needs 56 min; `--selftest` §5 now asserts every
corpus prompt clears ≥2× its measured cost and that the old constant would
have failed. Same commit made the top-5 probe cached + pooled (it is a full
re-generation, ran serially, and a kill discarded it entirely). Nothing
gate-affecting: ε, waiver budget, oracle invocation and the compared ids
are untouched. **THE SURVEY COMPLETED 2026-07-28 AND THE GATE WAS RE-SCOPED.** Generate
phase 50/50, zero tooling errors (the derived-ceiling fix held). Census:
**19 of 50 prompts diverge from the oracle**, 31 exact, 74.9% of ids match
before first divergence, and **18 of 18 classifiable divergences are a
clean top-2 swap** (stoke picks exactly the oracle's rank-2 candidate and
vice versa) at oracle gaps of 0.001–0.204 nats, only one under ε. The 19th
(p28) is a coalesced-entry read-out limit, not a stoke result — a coalesced
`completion_probabilities` entry carries only its LAST swallowed step's
distribution, verified across all 2,867 entries. Divergence appears in
EVERY tier, so it is not an adversarial-tier artifact. Cause: the oracle
quantizes activations to Q8 blocks for integer dot products, M2's stoke is
pure f32 — two arithmetic classes ranking near-ties differently, not a bug
on either side (L2 puts all 36 layers within 1e-3 of the f32 reference).
**KABIR TOOK MEMO OPTION 3:** M2's token gate is now **C-vs-ref on a
stratified subset + the 36-layer L2 ladder**, and oracle token-parity is
**deferred to M3** (see the M3 entry — it inherits the gate AND the
mandatory aligner fix). **THE GATE DEFINITION IS FIXED — Fable ruled
2026-07-28** (Kabir-launched session, per the routing):
`superpowers/memos/2026-07-28-m2-substitute-gate-RULING-FABLE.md` is the
binding authority. Short form: subset = the 24 as launched (all 19
oracle-diverging + one oracle-exact control per short tier; p04 excluded by
recorded transitivity and cited in the report; no long-context control on
makespan grounds); pass = token-exact incl. stop index on every prompt with
≤2 waivers under the four as-built `ref_waiver_decision` conditions
(evaluated at the FIRST divergence only; ordered reserve replacement,
replacements non-waivable) AND the L2 ladder re-run ≤1e-3/no-STEP AND zero
unresolved tooling errors; a FAIL is immutable in-session and escalates to
Fable. **⏸ EVERYTHING IS PAUSED (Kabir 2026-07-28 ~15:45) — nothing
restarts without his word.** Remaining, in order, all post-resume: (1) the
recorded run — `Start-ScheduledTask -TaskName 'stoke-m2-ref-gate'`
(~4.5 h, subset already baked in, banked ZERO before the pause so it starts
over); (2) L2 ladder re-run after it; (3) adjudication per the ruling §3;
(4) gate report `superpowers/gate-reports/2026-07-28-m2-ref-parity.md`;
(5) checklist + stamp + tag `m2` — playbook §5's box 1 is superseded by the
ruling, the rest stand. FREEZES until the report is stamped: no `stoke.exe`
rebuild, no `ref_forward.py` edit (both hash-pinned into the evidence
chain). The ledger NEXT-SESSION block has the exact resume point.

**M4 — ✅ CLOSED 2026-08-13, tag `m4`.** All five §8.4 gate items measured.
Four pass outright; **item [4] passes the bound that gates correctness (zero
BUG-class over 50 prompts, all 38 divergences classified ARITHMETIC-CLASS,
all seven ESCALATEs laddered rather than sampled — no STEP anywhere) and
misses the class-rate bound (38/50, 1.1875%/token against ≤19/50 and
≤0.5%/token).** §8.1 pre-ruled that disposition: a results-vs-physics memo,
never a grind-until-green — `superpowers/memos/2026-08-13-m4-q8kv-results-vs-
physics.md`. The finding is that the bound calibrates against the M2 census, a
pair differing only in **f32 reduction order (~1.2×10⁻⁷)**, while Q8-KV
requantises the KV cache to 8 bits (**~2.3×10⁻³ RMS**) — four orders apart,
and the resulting logit perturbation (≈0.04–0.08) is the same order as the
observed margins (0.013–0.566), which is why near-ties flip and decided tokens
never do. **f32 KV remains the default; M5's design note carries the dtype
call**, where the footprint argument is independent and load-bearing (the f32
pool at N=16/ctx2048 commits 9.04 GB, measured 2026-08-13).

Four entries ride into M4.5/M5 as carried headroom: **(a)** τ for item [4] is
frozen from agreeing positions of the same pair it grades — same shape as M3's
(a), restated not newly found; **(b)** the curve's pool is reserved but barely
touched, so it measured SCALING, not residency at depth; **(c)** `kv_view` is
now a partial function (fatal on a Q8 store), the f32-mode accessor for tests
while the engine uses `kv_write`/`kv_read`; **(d)** the class-rate bound and
`l2_diff.py`'s `1e-3` are both constants inherited across perturbation classes
and both now want τ's per-pair form — M3 raised the shape, M4 met it twice.

**M3 — ✅ CLOSED 2026-08-11, tag `m3`.** Closed on the classified-divergence
criterion: 9/9 ARITHMETIC-CLASS, zero BUG, τ_Q4KM frozen before grading.
**One precondition was met by judgement, not measurement — the flip tripwire
fired (9 ≥ 8) and is FLAGGED, not quiet** (`superpowers/memos/2026-08-11-m3-
tripwire-anomaly-memo.md`; Kabir confirmed 2026-08-11). Two entries ride into
M4 as carried headroom: **(a)** τ is frozen from agreeing positions on the
same pair it grades, so a defect uniformly inflating logit differences would
inflate τ and the margins together and this criterion cannot see it —
`ref_forward.py`'s independence from stoke's C lineage is the actual
protection, and a spec misreading shared by ref and HEAD stays invisible (the
07-28 blind spot, still deferred); **(b)** gate [1]'s bit-exactness price of
1.235× and K3's ~1.16× residual, permanent for Phase 1 per the 08-05 ruling
and ratified by Kabir 2026-08-10. The text below is the milestone's original
scope, kept for its reasoning; the oracle gate it describes was RETIRED by
the 08-05 ruling and M3 closed oracle-free.

**M3 — SIMD + pool. NOW ALSO CARRIES M2'S DEFERRED ORACLE GATE.** Inherited
from the 07-28 option-3 decision, and M3 does not close without them:
(1) the **50-prompt greedy oracle parity gate** under the §8 waiver
protocol, run AFTER the integer-core switch — which is the point, since
LOCKED decision #2 puts stoke in llama.cpp's own numeric class (Q8_0
activations, i32 core) and the M2-era census (19/50 diverging, 18/18 a
clean top-2 swap at 0.001–0.204 nats) is the baseline it is compared
against; (2) **the aligner fix is a PREREQUISITE, not an option** — on 4 of
50 prompts (p26, p27, p31, p34) `align_tokens_to_entries` currently fatals
because a ground-truth id never resurfaces as a `top_logprobs` candidate,
so byte lengths must instead be resolved from the GGUF vocab via
`tests/tooling/gguf_reader.py`; without it the gate can hard-fail on
tooling rather than return a verdict. Both are recorded in the near-tie
memo's DECISION section.
Forward note §M3: Q6_K is REQUIRED (Q4_K_M stores
attn_v/ffn_down/token_embd as Q6_K); integer cores on BOTH scalar+SIMD
paths (exact-diff gate depends on it; re-run parity after the activation
switch); AVX-VNNI VEX intrinsic (`_mm256_dpbusd_avx_epi32`, NOT the
AVX-512 name) + dpbusd sign trick — probe idiom from fetched ggml source;
K-quant layouts python-verified before C; work-stealing 64-row chunks
(P/E-proof); miss-the-bar attribution ladder runs on the mandated
`stoke bench --ops` microbench.

**Fable 2026-08-02 — three M3 amendments (strategic review; all
tightenings):**
(1) **Entry chore, and M3 does not close without it: the `model.c`
kv-cap guard.** Found by inspection 2026-07-31: `model_forward_position`
(`src/model.c:464` as of tag `m2`) validates the Kv against the model on
`n_layer`/`n_head_kv`/`head_dim` but NOT `max_ctx`, while `pos` is
bounded by the CACHE's cap and `s_scores` is sized from the MODEL's — a
Kv opened larger than the model's cap lets the attention loop write past
`s_scores` into adjacent scratch: silent wrong tokens, not a crash. Not
reachable today (every call site passes `model_max_ctx()`), but M4's
paged KV dismantles exactly that convention, and ASan cannot catch it
until a test exercises it. Fix = add the `max_ctx` clause to the shape
check + a regression test that opens an oversized Kv and forwards past
the model cap (ASan then guards it permanently). Unblocked since tag
`m2` lifted the binary freeze; land it before K2 changes `model.c`.
(2) **K0 probe findings amend the forward note's §M3 traps (2026-08-02
probes, ledger):** the wrong-spelling `_mm256_dpbusd_epi32` COMPILES AND
RUNS on Windows gcc 15.2 — the promised compile error does not exist, so
K3 guards by ggml-style `#if defined(__AVXVNNI__)` discipline + a review
grep or an own `#error`, not by trusting rejection. **BOTH LEGS NOW
PROBED (2026-08-02) AND THEY AGREE COMPLETELY:** WSL gcc 13.3 also
compiles AND runs the wrong spelling, and both spellings emit the
identical `{vex} vpdpbusd` on both toolchains, so there is no
compile-time protection anywhere in the matrix and nothing about the
intrinsic choice is platform-conditional. `__AVXVNNI__` is defined and
`__AVX512VNNI__` is not, so ggml's `#elif defined(__AVXVNNI__)` branch
selects correctly. AVX-VNNI-INT8 = 0 on this box, so the dpbusd sign
trick is mandatory (measured; do not re-ask in K3).
(3) **Sequencing pin:** the aligner fix lands BEFORE the K2 oracle run
(4/50 prompts fatal the aligner — the fix is what makes the run
readable), and K2's 50-prompt gate runs on the single-threaded integer
core, before the pool (K5) exists, so the arithmetic-class question is
answered in isolation; K6 re-passes both files with the pool on.
Playbook §6's K2 text predates the 07-28 inheritance and omits the
aligner; where they differ, this section + the ledger govern.

**Fable 2026-08-03 — the -O ruling (binding:
`superpowers/memos/2026-08-03-m3-optimization-level-RULING-FABLE.md`;
amends the forward note §M3 where they differ; bar/ε/waivers/corpus
untouched):**
(1) **Build default is `-O3`** + `-ffp-contract=off`. Sanitizer legs stay
`-O1` explicitly (more sensitive there, not less; the un-instrumented
`-O3` plain legs are what verify the shipped artifact). **LANDED `d0f8a6b`
— and the ruling's stated REASON for the fp-contract pin was refuted at
landing, while the action stood.** The ruling argued GCC 14 flipped the
ISO-mode contract default so Win gcc 15.2 and WSL gcc 13.3 would disagree
inside `-mfma` files. Measured from codegen instead: under `-std=c11 -O3
-mfma` **both** toolchains emit `vmulss`+`vaddss` and neither contracts,
while the same source in gnu17 mode emits `vfmadd132ss` on both — probe
sensitive, predicted divergence absent. Pinned anyway because contraction
becomes possible under K3's `-mfma` and a future toolchain flip would
break scalar↔SIMD f32 exactness silently. **Also: `gcc -Q --help=common`
— the check the ruling prescribed — reports `fast` on BOTH legs and
contradicts codegen; it prints the raw option default, not the
language-adjusted value. Do not use it to settle this question.**
(2) **`bench --ops`' reference stream is replaced before any new `%ref`
row**: serial_f32 measured f32-add latency, ~2.15× under the real
~15 GB/s ceiling; new `ref_gbs` = max(split8_f32, xor_u64), both
recorded. history.jsonl → **schema 2** with compile flags embedded in
the binary; the two `-O1` rows stay untouched (`schema:1` IS the -O1-era
marker); the pre-SIMD baseline row is re-taken at `-O3` before K3's
first C.
(3) **✅ RESOLVED — Fable ruled same day (binding:
`superpowers/memos/2026-08-03-k3-simd-headroom-RULING-FABLE.md`). D3 is
RETRACTED AND REPLACED, and the box-state mystery has a named suspect.**
The single-thread SIMD lever is REAL: working band **~1.7–2.5× over
scalar `-O3`** (interleaved cycles 1.96×/2.52×; plan on ~2×); the
brief's ~2.3× is restored. **K3 exit expectation = llama-1T PARITY as a
RATIO (median within-cycle stoke/llama 0.9–1.1, ≥3 interleaved cycles,
pinned, same session) — never an absolute tok/s; absolutes are
quarantined project-wide** (only quiesced-checklist sessions or
interleaved pairs produce comparison-grade numbers). **"The bar is won
in K5" is withdrawn — and so is the reopening's saturation story:**
forensics found `FactoryGameSteam` averaging **5.08 cores since
09:57:55**, spanning EVERY contested window of both sessions (ruling's
runs ~10:45–11:15, reopening's ~12:10–13:35); the 6T/10T "wall" of
26–34 GB/s was measured against a running game on hardware that is
**2× DDR5-4800 dual-channel (76.8 GB/s theoretical)**, with M0's
quiesced llama already demonstrating ~39 GB/s effective. Attribution is
now a BUDGET: bar needs ~19.5 GB/s effective aggregate (≥2× slack);
K3 buys ~2× per-core; K5 needs only ~1.9× from a parity core; K4's
K-quant compute penalty is the open factor; levers compose as
min(wall, N×per-core), never a free product. **✅ THE QUIESCED
K3-ENTRY SESSION RAN 2026-08-03 (grade GATE, records under
`bench/results/2026-08-03-k3-entry-*.json`) AND THE ATTRIBUTION IS NOW
MEASURED, NOT ARGUED:** the interleaved 1T referent is **llama/stoke =
1.997× (median of 4 within-cycle ratios, spread 9.4%)**, landing on the
ruling's ~2× point estimate; and the quiet-box scaling curve is llama
**1T 3.574 / 6T 12.645 / 10T 10.802 tok/s** = **15.3 / 54.1 / 46.2 GB/s
effective**. Three consequences. (a) **The "26–34 GB/s wall" is dead
twice over** — 54.1 GB/s is 70% of the 76.8 theoretical and well above
even M0's ~39; the bar's 19.5 GB/s aggregate has **2.78× slack against
a measured figure**. (b) **10 THREADS IS SLOWER THAN 6** (46.2 vs 54.1
GB/s): adding the 4 E-cores COSTS 15%, which is the forward note's
P/E-straggling trap measured on the opponent's own binary — direct
evidence for LOCKED decision "work-stealing 64-row chunks", and K5 must
tune thread count rather than assume more is better. **[⚠ THE 10T DATUM
DID NOT REPRODUCE, 2026-08-11 M4 entry, quiesced, same box and binary
family — `docs/superpowers/design-notes/2026-08-11-m4-bandwidth-budget.md`
§1.5. 1T and 6T reproduce to −2.4% and −0.2%; 10T measures 13.47 against
10.802, i.e. +24.7%, so E-cores GAINED 6.7% (Q8_0) and 2.8% (Q4_K_M)
rather than costing 15%. Topology confirmed correct (i5-13450HX, 6P+4E).
A single depressed point in an otherwise-reproducing triple is a
box-state signature, and `38b4f5a` reopened D3 that same week for exactly
that — hypothesis, not finding. THE LOCKED DECISION SURVIVES, ITS CITED
EVIDENCE DOES NOT: 4 extra cores (+67% count) buy only +6.7%, which is
still straggling, just the non-negative kind, and static equal
partitioning across heterogeneous cores strands work either way. Stop
quoting a 15% penalty.]** (c) 1T→6T is
**3.54× on six cores (59% of linear)**: sublinear, but the game-window
2.49× understated it. **Projection, recorded before the work so it
cannot be fitted afterwards:** stoke 1.716 → ×2 (K3) → ×3.54 (K5, 6T)
= 12.1 tok/s Q8_0 ⇒ **~20.8 tok/s Q4_K_M before the K-quant compute
penalty, ~2.7× the 7.8 bar.** The penalty is the one open factor and
K4 entry measures it.
**⚠ K3 CAME IN AT ×1.390, NOT ×2 — the projection is re-run below with
the measured value, and the pre-registered version above is left
standing so the two can be compared.** K3 exit measured GATE-grade at
**llama/stoke = 1.437× (spread 5.2%)**, from 1.997× at entry; per-op
matvec time 600,865 → 442,178 µs/token. Parity (0.9–1.1) is NOT
reached. Re-projected on measured numbers: 1.716 → ×1.390 (K3) = 2.385
→ ×3.54 (K5, 6T) = 8.44 tok/s Q8_0 ⇒ **~14.5 tok/s Q4_K_M before the
K-quant penalty, ~1.86× the bar** (was ~2.7×). **The bar still clears,
with less room.**
**⚠ THE K-QUANT PENALTY IS NOW MEASURED (2026-08-04, K4 entry, GATE both
legs) AND IT IS THREAD-DEPENDENT — the ~1.86× above assumed the full
1.714× size ratio and does NOT survive.** llama-vs-llama interleaved,
same engine, quant file the variable (`bench/results/
2026-08-04-k4-entry-kquant-{1t,6t}.json`): **1T = 1.637× of a 1.714×
size ratio = 95.5% retained; 6T = 1.356× = 79.1% retained.** The GB/s
column carries the mechanism: at 1T neither file saturates anything
(15.0 vs 14.2 GB/s under a ~20 ceiling), but at 6T Q8_0 sits AT the
ceiling (57.4 GB/s, cf. the 54.13 entry sweep) while **Q4_K_M reaches
only 46.0 GB/s** — the superblock unpack, not bandwidth, is binding.
**So the penalty is worst exactly where the bar is measured, and a
single-thread probe would have reported a reassuring 95.5%.**
**Re-projected: 8.44 × 1.356 = 11.45 tok/s Q4_K_M ≈ ~1.47× the bar.**
**This is an OPTIMISTIC bound for stoke** — it is llama's SIMD K-quant
kernels; stoke's scalar Q4_K/Q6_K retains less until K4's twins land. Of K3's residual gap, **1.235× is measured as the
price of gate [1]'s bit-exactness** (three-way interleaved: bit-exact
8.29 GB/s, ggml-style 10.23, llama 15.76) leaving ~1.16× unexplained;
spending gate [1] is Kabir's/Fable's call and is priced, not proposed.
**DECIDED BY KABIR 2026-08-10: "keep exactness."** Gate [1] is NOT spent;
the 1.235× stays carried headroom, ratifying the 08-05 ruling's
"permanent for Phase 1" rather than changing it. The two named triggers
survive unchanged (a bar miss this would cover; Phase-1.5 CUDA) — and
note the CUDA trigger is now answered by scope rather than by spending:
GPU is offload-style with the CPU path as the REFERENCE (design note
2026-08-10 §8.2), so the CPU path keeps EXACT and the device path, which
could never be bit-exact against a different kernel family anyway, takes
its own numeric class (§8.1 shape #2, still to be decided at M4 entry).
Rationale of record: exactness costs SINGLE-STREAM speed and costs
NOTHING on concurrency, and M0 measured the concurrency opportunity at
~9× against this 1.24×. The ~1.16× residual needs no decision — K4's
SIMD twins are already scoped to answer it.
**Where the remaining per-core work is, from the per-op row:** four-row
batching gave attn_* 1.43–1.53× but ffn_up only 1.26× and **LOGITS
1.18×, and LOGITS alone is 10% of the token** — narrow-output matvecs
benefit least from row batching, which a uniform "memory-bound" story
does not predict. **STANDING
COMPARABILITY RULE (ruling §4, binds gate [3]'s referent, bench.c, the
playbook):** two perf figures compare only if same-session, interleaved
where runnable, cycles cited, sidecar attached; rep-±σ never certifies
a window; subtraction is comparison; a field name is not a basis.
Pinning stays mandatory AND pinned numbers carry a co-load sidecar
(pinning converts SMT-sibling occupancy into a hidden multiplier —
the unpinned bwprobe read 14.4 GB/s beside pinned llama 1.26 in the
same game-loaded window).
(4) **K3-entry chores (re-scoped by the 08-03 K3 ruling §7):** the §2.4
fork is SETTLED (`d6e02d1`) — neither branch; the "residue" was two box
states subtracted (paging refuted 2000:1 by `Pages Input/sec`; bench and
e2e agree to 1.12× within one window, ~89% coverage). Owed, blocking,
ONE quiesced session: pre-SIMD baseline row + llama 1T referent
interleaved + llama 6T/10T mini-sweep, sidecar proving quiesced
(game/Discord closed, all-LP busy <5% for 60 s, free RAM > model+2 GB).
Also K3-entry: `stoke run` in-binary timing report (differencing retired
— valid only within one window), bench.c sidecar fields + referent
driver script per ruling §4–5. K4 entry keeps the mandatory K-quant
bandwidth-vs-compute measurement **plus a llama Q4_K_M 1T/6T pinned
interleaved referent sweep**. **✅ BOTH DONE 2026-08-04, GATE on both
legs — see the re-projection above; the tool grew a `llama-vs-llama`
mode (same protocol, quant file as the variable) rather than the
comparison being written ad hoc.** **File census probed on the real
Q4_K_M: 216 Q4_K + 37 Q6_K (18 `attn_v`, 18 `ffn_down`, `token_embd`)
+ 145 F32, and ZERO Q4_0 ⇒ playbook P-D's Q4_0 download is DEAD SCOPE;
K4's real-bytes verification is Q4_K + Q6_K only.**
(5) **The deferred oracle gate may run at `-O3`** (same verdict ~3×
cheaper) under a mechanical precondition: rebuild the archived intcore
source at `-O3`, archive + sha alongside
`build/stoke-k2-intcore-2ed3d1e8.exe`, byte-identity re-verified on 2–3
gate-subset prompts first; any byte differs ⇒ run `-O1` and escalate.
Launch stays Kabir's; the gate is never blocked on the -O work.

**Fable 2026-08-05 — ITEM [2] RESTATED; the oracle bar is RETIRED as
structurally unreachable (binding:
`superpowers/memos/2026-08-04-m3-gate-item-2-RULING-FABLE.md`; the
inherited-gate text above stands as history of what was inherited, this
block governs).** The 2026-08-04 run failed leg 1 at p02:16 on an
M2-documented oracle divergence where `ref_forward.py` sides with stoke
(ref↔oracle 0.143 vs ref↔HEAD 0.043); the deferral's stated mechanism
(integer core ⇒ oracle's numeric class) was implemented at `vec.c:162`
and measured insufficient — matmuls converge, the f32 remainder does
not, and the waiver requires MUTUAL near-ties so stoke's confidence
disqualifies it by construction. **Item [2] v2 = the
classified-divergence criterion:** (i) within-stoke exactness (gate [1],
determinism, thread-invariance — ✅); (ii) M2-vs-HEAD sweep, all 50, every
divergence classified by the ruling §6 rule, ZERO BUG-class; (iii) ref
probe grids on BOTH files (τ frozen from the probe BEFORE it grades:
f32↔intcore max(0.10, 2×band-p95) per band [0,100)/[100,300)/[300,600);
stoke↔oracle 0.25 fixed), zero BUG-class, tripwires (sweep >30 divergent
prompts; ≥4 class flips/file-grid) ⇒ anomaly memo. **The oracle is a
SURVEY when Kabir authorizes it — divergence-counted vs the M2 census,
never pass/fail, never blocking. M3 closes oracle-free.** ε=0.001/
budget-2 waiver machinery RETIRED for cross-class token comparison.
**Also ruled same session: K3 exits as-is** (gate [3] passed
re-anchored-harder; ~1.16× residual + gate [1]'s 1.235× price carried as
numbered headroom, price permanent for Phase 1 with named triggers);
**f32-reduction twins parked dead-by-default** (≤1.32% combined, several
change the token stream, which is now criterion substrate); **standing
rules:** bar re-anchoring standing with asymmetric guard (easier
re-anchors need sidecar+memo), bar-miss ⇒ results-vs-physics memo
protocol generalized, headroom carried-vs-owed ratified (number+location+
date; ledger is the first shopping list on any miss), forward-note
doubts ⇒ proceed + FOR-FABLE entry except never run a gate leg the doubt
defines. Full stamp list + M4–M7 forward rulings in the ruling §7.5/§8.

**M4 — paged KV + static batch.** Forward note §M4: block=16; `kv_view`
body swap only (M2 froze the accessor); **batch-composition independence
by construction** (seqs-inner/weights-outer — same seq ⇒ token-identical
at any batch size; this is what keeps M5 debuggable); full-reservation
admission until preemption exists; **the sim invariants — ENUMERATED
2026-08-11 in `superpowers/design-notes/2026-08-11-m4-invariants.md`,
derived from 12 failure modes; the count was asserted in three places
before the set existed anywhere, so cite the SET** — asserted after
EVERY op; Q8-KV is property-tested + divergence-counted, never
token-parity-gated (default ON/OFF is Kabir's call at M5).
**Fable 2026-08-02 design constraints (Kabir-approved; bind the M4
design note when it is written):** (1) the block table keeps the pool
size **runtime-variable capable** — no static-pool assumption baked into
invariants or allocation — so Phase-2 memory-elastic serving (shrink and
give blocks BACK under desktop RAM pressure, the 3× failure mode M2's
ops history measured live) stays a feature, not a rewrite; (2)
**sequence teardown at any lifecycle point is a first-class op the sim
exercises** (all blocks freed, refcounts consistent) — it is the
foundation M5's cancellation invariant stands on; (3) **the design note
OPENS with a bandwidth budget** (08-03 K3 ruling §6): W measured
quiesced in-session at M4 entry (**floor is now ~54 GB/s effective —
measured GATE-grade 2026-08-03 at llama 6T, superseding the ~39 from
M0**; theoretical 76.8; the game-window 34 may not be quoted as W, and
note W PEAKS AT 6 P-THREADS — 10 threads measured 15% WORSE, so the
budget's N is not the core count **[SUPERSEDED BY MEASUREMENT 2026-08-11
at M4 entry, quiesced with ISLC stopped —
`docs/superpowers/design-notes/2026-08-11-m4-bandwidth-budget.md`. The 54
GB/s floor REPRODUCES exactly (Q8_0 6T = 54.0 vs the 54.13 entry sweep),
but it is a Q8_0 number: Q4_K_M tops out at ~44.8 GB/s because superblock
unpack binds before bandwidth does. THE THREAD-PEAK CLAIM DOES NOT
REPRODUCE — neither file peaks at 6; both rise monotonically through 10T
and 10T is BETTER (Q4_K_M +2.8%, Q8_0 +6.7%), not 15% worse; variance is
±0.02–0.27, too tight for noise. What survives is the conclusion, not the
mechanism: the budget's N is still not the core count, because returns
past 4–6 threads are small (Q4_K_M +63.3% over 2→4, then only +11.6%
across all of 4→10). USE N=6, which captures 97.3% of achievable W.]**);
per-step
traffic model where batching amortizes ONLY the weight term; the
KV-dominance crossover formula in N and ctx; Q8-KV's halving as the
input to Kabir's M5 default call; M5's ITL and 16.8@16 bars checked as
budget lines BEFORE paged KV is scoped.
**Fable 2026-08-05 (binding, same ruling as §M3):** Q8-KV's
divergence-count now HAS its bound — KV-ON vs KV-OFF within-stoke,
classified by the ruling §6 rule with its own entry-probe τ, zero
BUG-class, class divergences ≤ the M2 census rate (19/50 prompts AND
≤0.5%/token); exceeding it is a results-vs-physics memo about the
quantizer, not a grind. M4's gate-items-vs-constraints split and exact
stamp list: ruling §8.4. Batch-composition independence stays EXACT
(bit-level) regardless; W and the budget's N are re-measured at M4 entry
(re-anchoring is now a STANDING rule, ruling §8.2–8.3).
**Design inputs from the reference engines (2026-08-10, non-binding):**
`docs/superpowers/design-notes/2026-08-10-reference-sources-and-m4-design-inputs.md`
— read at M4 entry, before the M4 design note is written. It records that
vLLM independently reached three constraints M4 already binds (full-
reservation admission, first-class teardown, runtime-variable pool), adds
four inputs (free-list-as-eviction-order; the prefix-cache last-token
recompute rule; block reservation against livelock; frees fenced behind
step retirement), and names two anti-patterns not to reproduce. It also
carries the llama.cpp b10037-vs-master re-verification (15/15 transcribed
facts unchanged) that keeps M2's kernel and M7's rope transcriptions
trustworthy. It binds nothing and changes no gate.

**M5 — continuous batching (THE product gate).** Forward note §M5: the
locked scheduler (WAITING→PREFILLING→RUNNING→FINISHED; ≤1 active prefill;
128-token chunk co-runs and never displaces a decode — asserted; admission
watermark W=max(8, 2·|RUNNING|); preempt-youngest, push-front, and
**replay recorded ids, never re-sample**); the sim harness is built FIRST
(virtual clock, fake model, trace catalogue incl. the memory-pressure
trace; reference run = same trace with infinite pool); bar-miss
attribution starts from M4's static curve — **RESOLVED 2026-08-12: static
N=8 DID miss (plateau 8.7 vs 17.7), attribution is COMPLETE, and the gap
is per-core kernel arithmetic, not the scheduler — M4.5 closes it before
M5 opens; authority
`superpowers/memos/2026-08-12-throughput-parity-RULING-FABLE.md`.** M5's
design note MUST record the chunk-alignment invariant (chunk ends
block-aligned or the prefix cache cannot register the entry — vLLM v1
scheduler, mined 08-12; chunk 128 / block 16 satisfies it by
construction, a future chunk change must re-check). Soak uses the
detached-runner ops lesson. If llamacpp-tuned parity is cleared but the 16.8@16
inversion-derived bar is not: that is a results-vs-physics memo to Kabir,
not a grind.
**Fable 2026-08-02 additions (Kabir-approved; tightenings):** the trace
catalogue gains an **AGENTIC trace** — bursty arrivals, a shared
1–2k-token system prefix, short decodes, 10–20% mid-generation aborts —
because on a consumer box N=8/16 is agent swarms, not humans; and a
**CANCELLATION-SAFETY invariant** joins the asserted set — abort in ANY
scheduler state frees every block, refcounts stay consistent, and the
remaining schedule does not stall (M6 needs this exact path for client
disconnects — find its bugs here in the sim, not there). Also RECORD,
measured not gated: **p95 request-completion latency** under the mixed
trace — the agent-shaped number the human-shaped ITL bar does not
capture.

**M6 — freeze + server.** Forward note §M6: error-handling mechanism is
DECIDED (load-time trap via thread-local jmp_buf in the `fatal()` choke
point; API misuse returns `STOKE_ERR_*`; internal invariants abort through
`stoke_set_log_callback`); run the ABI freeze checklist as a mechanical
pass (incl. C++-TU compile test); server = own HTTP/1.1 + SSE with the
listed compat traps; **the definitive smoke: the M0 bench harness drives a
`stoke` adapter UNCHANGED**; stop-strings live consumer-side with a
tail-window; seeded sampling stays replay-safe via the M5 rule.

**M7 — harden + launch.** Forward note §M7: Llama-3.2's four
silent-garbage deltas (no QK-norm — gate ops on tensor presence; llama3
rope SCALING formula fetched exactly — plain NEOX passes short tests and
fails long ones, so its corpus includes ≥1k-token contexts; llama-bpe =
second pre-tokenizer regex entry, oracle-gated like M1; header-format
template #2); unsloth-file `stoke info` diff then parity; 24h detached
soak with RSS slope watch; re-run ALL baselines at current dated versions
before the README table; name-collision memo to Kabir BEFORE README work.
Also at long-ctx parity (Fable 08-02): stoke's `rope_neox` accumulates
`theta *= theta_scale` (llama.cpp-matching) while `ref_forward.py`
computes per-k powers directly — same math, ~1e-9 rounding difference
that GROWS WITH POSITION (logged 2026-07-31); if long-context parity
wobbles, check this first.
**Fable 2026-08-05 (binding, same ruling as §M3): M7's parity criterion
is ruled NOW, conditional as asked.** The tokenizer gate stays M1-style
byte-exact via `llama-tokenize` (discrete domain — unchanged). The
FORWARD-PASS parity is the §M3 v2 SHAPE re-instantiated for Llama-3.2 —
twins + ref probe grid + targeted tests for the four deltas — **never a
50/50 live-oracle token bar for any model**; τ is re-frozen from M7's
own entry probe with new position bands [600,1200)/[1200,∞) BEFORE any
parity evidence is graded (the one named DEFER trigger of the ruling).
"Re-run ALL baselines" resolved: PERF referents refresh at gate time,
the PARITY oracle stays pinned b10037 (per-model pin if Llama-3.2 needs
a newer llama); banked comparisons are never re-graded. Ruling §8.8–8.10.

**Cross-cutting (every milestone):** forward note's closing section —
16GB RAM ceilings (concurrency benching stays on Q4_K_M), every gate
names its GGUF explicitly, never commit a model / cite a git-ignored path,
history.jsonl row per hot-path-touching green commit, tsan every commit
from M3 on, `.gitattributes` keeps `*.sh` LF (WSL dash fatals on CRLF;
verify endings from WSL with `od -c`, Git-Bash reads can hide `\r`).

## Session protocol (every session, unchanged M2→M7)

1. Model gate: session model must be **Opus, at or above the owner line**
   in memory `engine-project.md` (currently Opus 5). Opened on anything
   else — below the line, OR on Fable because the saved default drifted
   there (it did on 2026-08-02, when Kabir summoned Fable and the CLI
   saved it as the new default)? STOP unless Kabir deliberately opened
   this model: run `/model claude-opus-5`, then start over from step 2.
   A deliberate Fable session does manager-tier state work and rulings,
   not milestone execution.
2. Read: this file's map → ledger NEXT-SESSION block
   (`.superpowers/sdd/progress.md`) → the current milestone's design note
   → the session playbook
   (`superpowers/plans/2026-07-20-opus-session-playbook-m2-m7.md`) —
   session orchestration, recovery matrices, and the operational problem
   catalogue live THERE.
3. Work the SDD loop: probe → brief → dispatch sonnet worker → verify
   green bar YOURSELF (Windows `mingw32-make test` + WSL
   `scripts/wsl-check.sh` plain/asan/tsan — judge by its printed ALL
   GREEN, not pipeline exit) → Opus task-review → commit (msg style:
   `area: what -- detail`; no attribution trailers) → ledger update.
   **Parallelize the independent tier** (Kabir 2026-07-19): dispatch
   independent workers concurrently as background agents — at M2 that is
   W1 ‖ W2 ‖ corpus-authoring after the manager's quick P2+P3 probes (M2
   note §11 has the exact sequencing); every brief pins a disjoint
   file-ownership list, workers never commit, the manager verifies and
   commits per-worker paths sequentially. Cross-MILESTONE work is never
   parallelized (gates sequence learning). Cross-PROJECT sessions (AySen)
   are Kabir's call, best placed in attention-light windows (M5/M7 soaks,
   long bench runs).
4. Milestone gate: run the checklist, record evidence in ledger +
   MEMORY.md build log, tag if specified, update the memory owner line,
   enter the next milestone's note.
5. Commit LOCAL-only. A remote now exists (`github.com/Legendxyz-afk/Stoke`,
   pushed by Kabir himself ~07-26, superseding the 07-16 no-remote rule) but
   pushing is HIS call — the manager never pushes. `.superpowers/` stays
   untracked.

## Beyond Phase 1 (context, not tasks)

Phase 1.5: CUDA (`.cu` behind `extern "C"`, onto the proven scheduler) +
ARM NEON — planning doc is an M7 deliverable. Phase 2 (RE-ORDERED,
Fable 08-02 Kabir-approved — supersedes the NPU-first line here and in
`../MEMORY.md`'s rationale): **the BRIDGE LAYER, the Ollama half of the
gap** — model alias registry + `stoke pull`, idle load/unload under an
LRU residency budget, per-model chat templates, `/metrics` (M5's driver
already computes the numbers), memory-elastic serving on M4's
variable-capable pool; NPU / heterogeneous execution moves BEHIND it
(hardware-gated anyway). Phase 3: the runtime —
router fold-in + AySen `local()` client in-process. Phase 4 (dream):
AS-core control plane; open-core company. Full rationale: `../MEMORY.md`.

Candidates parked by the Fable 2026-08-02 review for the M7 planning doc
(none touches a Phase-1 gate; listed so they are weighed then, not
forgotten): **speculative decoding** — trades the spare compute this box
has for the bandwidth it lacks, the same physics M0 measured, and the M5
replay-not-resample rule composes with a verify step; **cross-request
prefix reuse** — agent workloads resend system prompts every call, and
it falls out of M4's refcounted dedup if blocks outlive requests;
**grammar/JSON-constrained sampling** — sampler-layer, the AySen
`local()` runtime will want it at Phase 3.
