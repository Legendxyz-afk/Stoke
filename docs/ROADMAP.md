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
| **M4.5** | **Kernel ladder → llama-band** | **✅ 2026-08-19 tag `m45`** | **Authority: `superpowers/memos/2026-08-12-throughput-parity-RULING-FABLE.md`.** Order B → C → A′ → D(conditional) → L4(profiled). **P-BAR: N=1 decode ≥0.85× llama-bench tg** (same box/file/threads, referent re-anchored per §8.2; 0.95× iff D lands). C = CONTRACT v2 re-anchor (canonical f32 fold; integer core freely regroupable), graded old-vs-new by §6, zero BUG-class, **Q8_0 file streams BYTE-IDENTICAL through C or it is a bug**. **▶▶ L4 EXECUTED 2026-08-17 (`ba4e140`, `755a43b`) AND IT COST NO GATE, WHICH THE RULING DID NOT ANTICIPATE: the remaining gap was not arithmetic — ATTENTION HAD NEVER RUN ON THE POOL. `pool_for_rows` is called from `vec.c` only (seven times, all matvec), so the per-head loop ran on the calling thread from K5 onward while being 84.6% of decode at ctx 1500. Parallelising over heads is parallelism over INDEPENDENT OUTPUTS, so the arithmetic inside a head is untouched ⇒ no twin, no grading run, no re-anchor, CONTRACT v2 never engaged. §7.6's 8-lane canonical fold is NOT NEEDED and is not done. Correctness green on every leg (Windows 13/13 · WSL plain 13/13 · **WSL TSan 13/13 zero reports**, which also discharges the TSan leg owed since `76dee7f` · ASan clean on the real model at `--kv q8` · byte-compare 6/10 cases all-arms-identical, 4 owed). **▶▶▶ P-BAR IS RE-GRADED 2026-08-18/19 AND IS ***MET AT BOTH GRADED CONFIGURATIONS***: 0.912× at t=6 pinned (`0e27a75`) and 0.892× at t=8 unpinned (`0b405aa`), against the 0.85 bar it MISSED at 0.827–0.838 before L4. Three blocks each, spanning the night from different thermal states; graded on B/L, llama measured in the same sitting.** The 4-arm A/B (`1ac09f8`) prices L4 at **1.09× at ctx 87, 1.58× at 554, 2.94× at 1500**, and **the slope falls 0.1713 → 0.0289 ms/context-token** — stoke was 11.9× worse than llama on slope, now 1.72×. **⚠ THE POOL IS 99% OF THE WIN: isolated at ctx 1500, lanes alone 1.26×, pool alone 2.90×, both 2.94× ⇒ `ATT_LANES` earns ~1.3% and is now a SIMPLIFICATION candidate, not an optimisation.** The L4 note's prediction (317 → ~63 ms) **MISSED HIGH** at 106.75 ms measured — third instance of a kernel argument running ~1.5–2× hot (A′ 0.907×, Q6_K 1.108× vs 1.20×), so **price future rungs at half their argued gain**. **⚠ ARM A IS UNFIT AS AN ANCHOR** — the pre-maddubs archive predates `76dee7f` and swung 9.99 / 5.30 / 0.92 / 8.6 across sittings while B and llama held to a few percent; a Task-Scheduler explanation was tested and REFUTED (scheduled 9.044 vs interactive 7.999). B/A is void; both reports grade on B/L and say so. Byte-compare is **10/10 in one sitting** (`a48de9f`), discharging the 4 owed cases. |
| M5 | Continuous batching | **← NOW** | **THE bars: ≥17.7 @8 / ≥14.2 @16 AND ≥10.8 @8 / ≥16.8 @16**; ITL <250ms; S3 ≥13.3; 1h soak; cancel-safe sim (Fable 08-02). **Bars are reachable through M4.5's kernels, not batching — measured 08-12 (curve flat at 8.7, compute-bound at ~64 GFLOP/s)**. **▶ THE 08-17 ALARM IS RETIRED 2026-08-19: "extrapolated to ctx 1024 stoke pays ~187 ms/token (~5.4 tok/s) vs llama ~61 ⇒ 0.33×, M5's aggregate bars unreachable" was measured against a SINGLE-THREADED attention loop. On L4's slope (0.0289 ms/context-token) ctx 1024 costs ~93 ms/token (~10.8 tok/s) against llama's ~70 (~14.2) = 0.76×, and batching still amortises the weight term ⇒ THE BARS ARE BACK IN RANGE. The 08-11 budget's structural claim is untouched and is now the live constraint: M5 is a KV-TRAFFIC problem, crossover N×L ≈ 16,936 token-positions.** **▶▶ AMENDED IN PLACE 2026-08-19 AT M5 ENTRY — THE CROSSOVER 16,936 IS ARITHMETICALLY RIGHT AND IS ABOUT AN ENGINE STOKE IS NOT.** It prices f16 KV read ONCE per position (2,497,280,736 / 147,456 = 16,936, reproduced exactly). stoke stores **f32** and, because attention was q-head-outer, read each kv-head's K and V **four times** per layer per token (GQA 4:1) — 1,179,648 B per token-position, so **stoke's real crossover was 2,117**, and at N=8 / ctx 1500 (12,000 token-positions) it was already **5.7× KV-dominated**, not approaching a crossover. Checked against measurement, not just arithmetic: the traffic model predicts 97.9 ms/token at N=1 ctx 1500 and L4 measured **106.75 ⇒ decode runs at 92% of its pure-traffic ceiling**, so a traffic cut converts nearly one-for-one. **⇒ THE BARS WERE AT RISK FOR A SECOND, INDEPENDENT REASON: at the old traffic the N=16 aggregate ceiling is 22.6 tok/s against the 16.8 bar (74% of a pure-DRAM ceiling — a physics claim, not a bar), and N=8 is 20.9 against 17.7.** Rung **E2 (kv-head-outer attention, landed `4056be9`)** removes the 4× amplification: crossover 2,117 → 8,468, N=8 ceiling → 57.8 (3.27× slack), N=16 → 72.9 (4.34× slack). **E2 is therefore not an optimisation on M5's list — it is what puts the bars inside the envelope.** Entry rungs, gates, the chunk-alignment invariant, the M5 asserted set S1–S8, the `ATT_LANES` disposition and the Q8-KV default call all live in `superpowers/design-notes/2026-08-19-m5-entry.md`; **the sim harness's architecture — what it substitutes and what it must not — is `superpowers/design-notes/2026-08-19-m5-s0-sim-architecture.md`, and it constrains `sched.c` before that file exists (no direct `model_forward_batch`, no direct `now_ms`).** **▶▶▶ GATE ITEM [3] IS AT RISK, AND IT IS STRUCTURAL — memo `superpowers/memos/2026-08-19-m5-itl-vs-chunk-results-vs-physics.md` (INDICATIVE, unquiesced; ratios robust, absolutes not). A decode co-running with a prefill waits the CHUNK'S WHOLE STEP, so ITL max falls linearly with chunk width: 4463 / 1209 / 624 ms at chunk 128 / 32 / 16. The 250 ms bound extrapolates to chunk ≈ 6 — BELOW THE 16-TOKEN KV BLOCK S-ALIGN REQUIRES, so the smallest LEGAL chunk still exceeds the bound by 2.5×. The forward note's attribution ladder is EXHAUSTED (64 tried, and 32 and 16; the prefill rows DO ride the same op walk — E3 uses `forward_batch_body`, byte-identical by the entry gate). Two findings from the sweep worth more than the headline: the MEDIAN moves the OTHER WAY (159 → 542 ms as the chunk shrinks) because total prefill work is conserved — chunking reshapes the wait's distribution rather than removing it, and p50 and max point at opposite settings; and AGGREGATE IS UNCHANGED across all three widths, because E3 measured amortisation SATURATING BY ~23 ROWS — so the locked 128 buys no throughput and costs ITL linearly. Options, no recommendation: shrink `BP_BLOCK_TOKENS` (touches M4's closed set) · make the STEP cheaper (the attention dots are still scalar f32 by design; SIMD there is priced and waiting on Kabir, and would then block TWO gate items) · re-anchor 250 ms, which is a human-interactivity figure while M5's operating point is agent swarms — Kabir's call, his headline. `STOKE_SCHED_CHUNK` exists to MEASURE this; the default stays the locked 128.** **▶▶▶ GATE ITEM [1] IS MET 2026-08-19 (`c229f27`): `src/sched.c` + `tests/test_sched.c`. All seven S0 traces (poisson-mix · burst · giant+small · cancel-storm · maxtok1 · pressure · AGENTIC) × 3 seeds × {constant, linear} = 42 runs green, plus a 1,000,130-step soak over 11,870 traces with 5,142 preemptions — every invariant after EVERY step, never sampled. Asserted: M4's I1–I12 via `bp_check`, and S1 · S2 (watermark, from the state BEFORE the step) · S3 · S4 · S5 (exact identity, not a bound) · S6 (pressure run vs infinite-pool reference, streams byte-identical, preemptions asserted >0 tight and ==0 reference) · S7 · S8. ⚠ **S4 IS ASSERTED AS PUSH-FRONT ONLY** — that the victim heads WAITING, which is what the starvation argument rests on; WHICH request is chosen is not observable from outside and the implementation deliberately refines the rule (victim's arrival ≥ the triggering row's, a MEMORY-SAFETY constraint: rows are built oldest-first, so plain youngest-RUNNING can close a `Kv` the batch already references). ⚠ **THE WATERMARK IS EFFECTIVE ENOUGH THAT `pressure` MUST BE DELIBERATELY ADVERSARIAL** — a request whose whole life fits inside W's 8-block reserve cannot outgrow what admission checked, so short requests under a small pool SERIALISE rather than thrash; the first pressure trace never preempted once and would have banked as a passing S6 test proving nothing. **[1] does NOT cover [2] the bars, [3] ITL, [4] S3 fanout or [5] the soak under ASan/TSan — the sim's cost model is an INPUT, not a measurement.** ▶▶ **BLOCKING DECISION, taken 2026-08-19 before any scheduler code: `superpowers/memos/2026-08-19-m5-admission-reservation-DECISION-OPUS.md`. M5 admits on `blocks(prompt) + W`; M4's `bp_admit` reserves the WHOLE context (I8), under which decode-time exhaustion is UNREACHABLE and with it gate item [6], the `pressure` trace, S4 and S6 — four things that would assert green over a branch that never executes. Resolved by splitting `reserved` from `cap_blocks` (I8 stays literally true; its *implication* that growth always succeeds was never an invariant), adding **I12**, and leaving `bp_admit` full-reservation for every existing caller. M4's F9 already named the precondition — full reservation was right "with no preemption path (M4 has none)" — and M5 builds that path. Direction check recorded for Kabir; if he prefers full reservation, [6]/`pressure`/S4/S6 must be STRUCK, not asserted.** **▶▶▶ GATE ITEM [2] IS MEASURED 2026-08-20 AND IT MISSES: serve AGGREGATE(M0) 4.938 @8 vs 17.7 and 4.594 @16 vs 16.8 (0.28x / 0.27x on a thermally plateaued box; 6.378 / 5.905 = 0.36x / 0.35x cool). Report `gate-reports/2026-08-20-m5-bars-report.md`, memo `memos/2026-08-20-m5-gate2-results-vs-physics-FOR-KABIR.md`, prediction pre-registered BEFORE the first cell at `0b7f260`. ⚠ THE CAUSE IS ARITHMETIC, NOT BATCHING AND NOT BANDWIDTH. A step at N rows streams the weights once but does N x 8.46 GFLOP, which amortises not at all. MEASURED (4 counterbalanced passes, `gate-reports/2026-08-20-m5-flatness-report.md`): N=1/2/4/8/16 -> 14.73 / 15.52 / 18.68 / 20.50 / 21.08 tok/s, a 1.43x gain across the whole range where traffic-only reasoning predicts 12.0x, and N=8->N=16 buys 2.9%. **⇒ THE AGGREGATE TRAFFIC CEILINGS IN THE ENTRY NOTE (57.8 @8, 72.9 @16) DESCRIBE A MACHINE THAT DOES NOT EXIST FROM N=2 UP** -- `m5_traffic_budget.py` ends its table with 'traffic only, no compute' and that caveat was never cashed; `tests/tooling/m5_compute_budget.py` now cashes it. This does NOT retract E2 (its measured 1.12x at ctx 1500 is real) -- it says the slack E2 reported was not there to spend. **THE SCHEDULER IS EXONERATED**: preemptions 0 in every cell, and the STATIC path clears both bars on decode (17.4-17.9 at the plateau, 23.07/22.65 cool), which is the playbook's attribution ladder rung (1) answering NO. The serve path is charged for prefill INSIDE the M0 span -- llama-server is charged identically and still read 17.7, because it prefills at 91.3 tok/s against stoke's 32.2. **NEEDED: a uniform 2.75x on arithmetic; the same-sitting prefill-vs-prefill gap to llama is 2.83x** (772 vs 273 GFLOP/s), so the headroom exists on this silicon but with no slack. ⚠ CORRECTED IN FLIGHT: an earlier 3.96x compared llama's PREFILL to stoke's DECODE. ⚠ AND A PRE-REGISTERED FALSIFIER FIRED -- 'flat in N' was WRONG (N=8 landed 32% over N=2 against a 20% falsifier) because the compute ceiling is a function of TILE AND THREAD OCCUPANCY, not a constant: 120 GFLOP/s at N=1 rising to 172 at N=16, of which 1.155x is thread occupancy alone (cores busy 5.80 -> 6.70). 'Saturating, and already saturated where the gate measures' is what the data supports. **GATE [3] ALSO MISSES ON THE REAL S2 SHAPE**: ITL max 4713 ms @8 and 5454 ms @16 against the 250 ms bound, confirming the 08-19 memo on a workload it called indicative. **OPEN FOR KABIR, AND IT IS ONE QUESTION NOT THREE** (SIMD in the attention dots, a repacked GEMM, and [3]'s bound are all downstream of it): keep bit-exactness with the M3/M4 parity lineage, or buy the arithmetic? ⚠ BUT SEE MEMO 5b -- THAT QUESTION IS PREMATURE. Inside a super-block the kernel is ENTIRELY INTEGER (floats only at the boundary, twice), integer addition is associative, so regrouping there is bit-exact BY CONSTRUCTION and only the cross-super-block f32 accumulation is pinned -- `simd.c:75`'s own founding principle. The 2.83x is MEASURED and UNATTRIBUTED; attribute it before trading anything for it.** **▶▶▶ THE 2.83x IS ATTRIBUTED 2026-08-20 (memo `memos/2026-08-20-m5-attribution-RESULTS.md`, prediction pre-registered at `b6de2b1`): IT IS THE KERNEL, AND IT IS NOT INSTRUCTION COUNT.** Thread sweep, both engines, order counterbalanced (`gate-reports/2026-08-20-m5-attrib-threads-report.md`): single-thread gap **3.22x**, and ~3x at EVERY thread count -- a per-thread difference, not a parallelisation one. **At t=8 stoke scales 4.54x against llama's 4.30x, so the pool is not merely exonerated, it is a STRENGTH**, and none of this gap is the scheduler or M5. ▶▶ `objdump` then found the per-sequence mins term fully unrolled and ENTIRELY SCALAR -- ~40 uops beside the vector dot's ~57, i.e. ~40-45% of emitted per-sequence work that is not the dot product -- and replacing it with `vpmaddwd` is **bit-exact BY CONSTRUCTION** (all integer, associative, overflow bound derived from `quantize_row_q8_k`: |bs| <= 2048, pairs <= 4096 < 32767, x mins <= 63 -> 258,048, madd -> 516,096). Measured interleaved with the arm order alternating: **B/A = 1.063x at t=1 and 1.079x at t=8, B winning 6 of 6 pairings.** ⇒ **REMOVING ~40% OF THE EMITTED INSTRUCTIONS BOUGHT ~7% OF THE TIME. The uop-count theory of the gap is DEAD** -- the kernel runs at ~2.5 uops/cycle against a 6-wide front end, so it was never front-end bound and the scalar `imul` chain overlapped with the vector work on separate ports. What holds it at 46 GFLOP/s single-threaded against llama's 149 is port pressure, dependency latency or memory. Candidates named but UNMEASURED: `vpmulld` on the critical path (2 uops, ~10-cycle latency, 8x per super-block per sequence, purely to apply a 6-bit scale -- llama defers scales to a float FMA, which is NOT bit-exact); activation re-reads; `wnib[8]` holding half the register file live. ⚠ **A SINGLE UNPAIRED PRE-FLIGHT SAMPLE READ 0.957x -- B SLOWER** -- and the interleaved run reversed it; the cheap reading would have discarded a change that wins. ⚠ **THE DEFAULT STAYS `STOKE_MINS_SIMD=0`**: byte-compare green (p00x8 f32/q8, p41x2 f32, all SAME, **mutation-verified** -- one wrong mask lane makes it DIFFER) and suite 14/14 + 4 goldens, but a change on the M3/M4 parity lineage owes the full 10-case verify gate against the archives, which runs 1-2 h. **NEXT: run that gate, then flip the default; then price `vpmulld` by ablation, which is the number Kabir's bit-exactness decision actually needs.** ⚠ The byte-compare had to run through `batch --seqs N`, NOT `stoke run` -- the kernel only executes at n_seq > 1, so an N=1 compare would have been vacuous (the same hole E1 fell into). |
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
(P/E-proof) **[⚠ 2026-08-17: "P/E-proof" IS FALSE AS WRITTEN. K5's
atomic-counter chunking was in place and stoke still lost up to 10.9x to
the E-cores, because Windows EcoQoS parked the whole pool there — chunking
cannot rescue work if every worker is on a slow core. Dynamic chunking is
necessary and was never sufficient; the scheduler had to be told these are
performance threads. See the §M2 bandwidth row and `76dee7f`.]**;
miss-the-bar attribution ladder runs on the mandated
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
quoting a 15% penalty.]**
**[⚠⚠ AMENDED AGAIN 2026-08-17, AND THIS TIME THE PENALTY IS REAL — BUT IT
IS STOKE'S, NOT THE HARDWARE'S, AND NEITHER EARLIER ENTRY COULD HAVE SEEN
IT. Both the original claim and its 08-11 retraction were measured on
LLAMA, whose ggml workers spin and therefore keep the P-cores; llama
cannot exhibit this defect, so no llama number is evidence either way.
Measured on STOKE (`76dee7f`, gate-report `2026-08-17-m45-spin-probe` and
`tests/tooling/run_ecoqos_repro.ps1`): Windows EcoQoS classified stoke's
pool as background work — helpers slept on a condvar ~250x per token —
and parked it on the E-cores, costing **up to 10.9x**, intermittently,
for stretches of tens of minutes. Deterministic reproduction via
process-level EcoQoS: 16.13-16.21 tok/s baseline, **1.40-1.50 throttled**,
15.84-16.16 with the per-thread opt-out. It corrupted P-BAR block 2 on
2026-08-17 and is the likeliest cause of the 08-15 ladder's arm A reading
8.262 where the same binary now gives 9.989 at 0.7% spread.
⇒ THE LOCKED DECISION'S *EVIDENCE* IS RESTORED, BUT ITS *REMEDY* WAS
MISDIAGNOSED: work-stealing was never the missing piece — `pool.c` has
chunked dynamically via an atomic counter since K5 and still collapsed.
THE WAIT WAS THE PROBLEM, and the fix is
`SetThreadInformation(ThreadPowerThrottling, StateMask=0)` per worker.
⇒ AND THE STANDING RULE FOR THIS FILE: a claim about how STOKE behaves on
heterogeneous cores can only be tested on STOKE. Twice now this row has
been rewritten from llama data.]** (c) 1T→6T is
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
`superpowers/memos/2026-08-12-throughput-parity-RULING-FABLE.md`.**
**⚠⚠ "PER-CORE KERNEL ARITHMETIC" IS SUPERSEDED TWICE, AMENDED IN PLACE
2026-08-17. First: the context-scaling sitting put stoke's matvec/
weight-streaming INTERCEPT at parity with llama (48.76 vs 52.51 ms/token)
and located the whole remaining deficit in attention over the KV cache.
Then L4 found that attention was not arithmetically slow — IT WAS RUNNING
ON ONE CORE, having never been dispatched to the pool. So the M5 alarm
built on the old slope ("~187 ms/token at ctx 1024 ⇒ M5's aggregate bars
are unreachable") was measured against a single-threaded attention loop
and MUST NOT be quoted until re-measured. The 08-11 budget's structural
claim is untouched and still governs: M5's bars are a KV-TRAFFIC problem,
not a weight-streaming one (crossover at N×L ≈ 16,936 token-positions).**
**▶▶ AND THAT CROSSOVER IS AMENDED AT M5 ENTRY (2026-08-19), see the M5 row:
16,936 prices f16 read once; stoke stored f32 and read each kv-head FOUR times
per layer per token, so its real crossover was 2,117 and the N=16 bar sat at
74% of a pure-traffic ceiling. Rung E2 (`4056be9`) removes the amplification
and takes that to 23%.** M5's
design note — `superpowers/design-notes/2026-08-19-m5-entry.md`, WRITTEN, and
the authority for the entry rungs, the asserted set S1–S8, the `ATT_LANES`
disposition and the Q8-KV default call — records the chunk-alignment invariant
(chunk ends block-aligned or the prefix cache cannot register the entry — vLLM
v1 scheduler, mined 08-12; chunk 128 / block 16 satisfies it by construction,
a future chunk change must re-check). Soak uses the
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
