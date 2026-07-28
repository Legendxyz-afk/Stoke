# stoke — Phase 1 roadmap (the operational map)

*Updated 2026-07-28, late-M2. This is the FIRST document a working
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
| **M2** | **Scalar forward pass** | **← NOW (⏸ paused)** | **RE-SCOPED 07-28 (Kabir, memo option 3): greedy 64-id C-vs-ref parity on a stratified subset + the 36-layer L2 ladder.** Oracle token-parity DEFERRED to M3. **Definition FIXED by Fable ruling 07-28** (`superpowers/memos/2026-07-28-m2-substitute-gate-RULING-FABLE.md`) |
| M3 | AVX2+VNNI kernels + pool | queued | Scalar↔SIMD exact-diff; **the deferred 50-prompt ORACLE parity gate (+ the required aligner fix)**; parity re-pass Q8_0+Q4_K_M; ≥7.8 tok/s single-stream; tsan |
| M4 | Paged KV + static batch | queued | 11-invariant sim ×1e6 ASan; batch-of-N token-identity; prefix-dedup; monotone N∈{1..16} curve |
| M5 | Continuous batching | queued | **THE bars: ≥17.7 @8 / ≥14.2 @16 AND ≥10.8 @8 / ≥16.8 @16**; ITL <250ms; S3 ≥13.3; 1h soak |
| M6 | `stoke.h` freeze + server | queued | openai-python suite; M0 harness drives stoke unmodified; ABI checklist; C++ TU |
| M7 | Harden + launch-ready | queued | Llama-3.2 parity incl. long-ctx; unsloth-file run; 24h soak; reproducible report; Kabir sign-offs |

Milestone entry = prior gate stamped. Full gate checklists are mechanical
lists in the design notes — run every box, record in the ledger +
MEMORY.md, tag if the milestone says so, move on.

## Per-milestone: where the truth lives

**M2 — scalar forward pass (NOW).** Operative doc:
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

**M4 — paged KV + static batch.** Forward note §M4: block=16; `kv_view`
body swap only (M2 froze the accessor); **batch-composition independence
by construction** (seqs-inner/weights-outer — same seq ⇒ token-identical
at any batch size; this is what keeps M5 debuggable); full-reservation
admission until preemption exists; the 11 sim invariants asserted after
EVERY op; Q8-KV is property-tested + divergence-counted, never
token-parity-gated (default ON/OFF is Kabir's call at M5).

**M5 — continuous batching (THE product gate).** Forward note §M5: the
locked scheduler (WAITING→PREFILLING→RUNNING→FINISHED; ≤1 active prefill;
128-token chunk co-runs and never displaces a decode — asserted; admission
watermark W=max(8, 2·|RUNNING|); preempt-youngest, push-front, and
**replay recorded ids, never re-sample**); the sim harness is built FIRST
(virtual clock, fake model, trace catalogue incl. the memory-pressure
trace; reference run = same trace with infinite pool); bar-miss
attribution starts from M4's static curve (if static N=8 already missed,
the scheduler was never the problem). Soak uses the detached-runner ops
lesson. If llamacpp-tuned parity is cleared but the 16.8@16
inversion-derived bar is not: that is a results-vs-physics memo to Kabir,
not a grind.

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

**Cross-cutting (every milestone):** forward note's closing section —
16GB RAM ceilings (concurrency benching stays on Q4_K_M), every gate
names its GGUF explicitly, never commit a model / cite a git-ignored path,
history.jsonl row per hot-path-touching green commit, tsan every commit
from M3 on, `.gitattributes` keeps `*.sh` LF (WSL dash fatals on CRLF;
verify endings from WSL with `od -c`, Git-Bash reads can hide `\r`).

## Session protocol (every session, unchanged M2→M7)

1. Model gate: session model must be **Opus, at or above the owner line**
   in memory `engine-project.md` (currently Opus 5). Opened below it (the
   saved default has drifted to Fable before)? STOP — run
   `/model claude-opus-5`, then start over from step 2. No stoke work on
   a mismatched model without Kabir's explicit in-session override.
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
ARM NEON — planning doc is an M7 deliverable. Phase 2: NPU /
heterogeneous execution (needs future hardware). Phase 3: the runtime —
router fold-in + AySen `local()` client in-process. Phase 4 (dream):
AS-core control plane; open-core company. Full rationale: `../MEMORY.md`.
