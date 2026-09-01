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
| M5 | Continuous batching | **✅ 2026-08-28 tag `m5` — CLOSED WITH TWO DOCUMENTED MISSES, [2] at 0.51x and [3] at 2.7x; both are the same arithmetic wall and neither is a scheduler question** | **THE bars: ≥17.7 @8 / ≥14.2 @16 AND ≥10.8 @8 / ≥16.8 @16**; ITL <250ms; S3 ≥13.3; 1h soak; cancel-safe sim (Fable 08-02). **Bars are reachable through M4.5's kernels, not batching — measured 08-12 (curve flat at 8.7, compute-bound at ~64 GFLOP/s)**. **▶ THE 08-17 ALARM IS RETIRED 2026-08-19: "extrapolated to ctx 1024 stoke pays ~187 ms/token (~5.4 tok/s) vs llama ~61 ⇒ 0.33×, M5's aggregate bars unreachable" was measured against a SINGLE-THREADED attention loop. On L4's slope (0.0289 ms/context-token) ctx 1024 costs ~93 ms/token (~10.8 tok/s) against llama's ~70 (~14.2) = 0.76×, and batching still amortises the weight term ⇒ THE BARS ARE BACK IN RANGE. The 08-11 budget's structural claim is untouched and is now the live constraint: M5 is a KV-TRAFFIC problem, crossover N×L ≈ 16,936 token-positions.** **▶▶ AMENDED IN PLACE 2026-08-19 AT M5 ENTRY — THE CROSSOVER 16,936 IS ARITHMETICALLY RIGHT AND IS ABOUT AN ENGINE STOKE IS NOT.** It prices f16 KV read ONCE per position (2,497,280,736 / 147,456 = 16,936, reproduced exactly). stoke stores **f32** and, because attention was q-head-outer, read each kv-head's K and V **four times** per layer per token (GQA 4:1) — 1,179,648 B per token-position, so **stoke's real crossover was 2,117**, and at N=8 / ctx 1500 (12,000 token-positions) it was already **5.7× KV-dominated**, not approaching a crossover. Checked against measurement, not just arithmetic: the traffic model predicts 97.9 ms/token at N=1 ctx 1500 and L4 measured **106.75 ⇒ decode runs at 92% of its pure-traffic ceiling**, so a traffic cut converts nearly one-for-one. **⇒ THE BARS WERE AT RISK FOR A SECOND, INDEPENDENT REASON: at the old traffic the N=16 aggregate ceiling is 22.6 tok/s against the 16.8 bar (74% of a pure-DRAM ceiling — a physics claim, not a bar), and N=8 is 20.9 against 17.7.** Rung **E2 (kv-head-outer attention, landed `4056be9`)** removes the 4× amplification: crossover 2,117 → 8,468, N=8 ceiling → 57.8 (3.27× slack), N=16 → 72.9 (4.34× slack). **E2 is therefore not an optimisation on M5's list — it is what puts the bars inside the envelope.** Entry rungs, gates, the chunk-alignment invariant, the M5 asserted set S1–S8, the `ATT_LANES` disposition and the Q8-KV default call all live in `superpowers/design-notes/2026-08-19-m5-entry.md`; **the sim harness's architecture — what it substitutes and what it must not — is `superpowers/design-notes/2026-08-19-m5-s0-sim-architecture.md`, and it constrains `sched.c` before that file exists (no direct `model_forward_batch`, no direct `now_ms`).** **▶▶▶ GATE ITEM [3] IS AT RISK, AND IT IS STRUCTURAL — memo `superpowers/memos/2026-08-19-m5-itl-vs-chunk-results-vs-physics.md` (INDICATIVE, unquiesced; ratios robust, absolutes not). A decode co-running with a prefill waits the CHUNK'S WHOLE STEP, so ITL max falls linearly with chunk width: 4463 / 1209 / 624 ms at chunk 128 / 32 / 16. The 250 ms bound extrapolates to chunk ≈ 6 — BELOW THE 16-TOKEN KV BLOCK S-ALIGN REQUIRES, so the smallest LEGAL chunk still exceeds the bound by 2.5×. The forward note's attribution ladder is EXHAUSTED (64 tried, and 32 and 16; the prefill rows DO ride the same op walk — E3 uses `forward_batch_body`, byte-identical by the entry gate). Two findings from the sweep worth more than the headline: the MEDIAN moves the OTHER WAY (159 → 542 ms as the chunk shrinks) because total prefill work is conserved — chunking reshapes the wait's distribution rather than removing it, and p50 and max point at opposite settings; and AGGREGATE IS UNCHANGED across all three widths, because E3 measured amortisation SATURATING BY ~23 ROWS — so the locked 128 buys no throughput and costs ITL linearly. Options, no recommendation: shrink `BP_BLOCK_TOKENS` (touches M4's closed set) · make the STEP cheaper (the attention dots are still scalar f32 by design; SIMD there is priced and waiting on Kabir, and would then block TWO gate items) · re-anchor 250 ms, which is a human-interactivity figure while M5's operating point is agent swarms — Kabir's call, his headline. `STOKE_SCHED_CHUNK` exists to MEASURE this; the default stays the locked 128.** **▶▶▶ GATE ITEM [1] IS MET 2026-08-19 (`c229f27`): `src/sched.c` + `tests/test_sched.c`. All seven S0 traces (poisson-mix · burst · giant+small · cancel-storm · maxtok1 · pressure · AGENTIC) × 3 seeds × {constant, linear} = 42 runs green, plus a 1,000,130-step soak over 11,870 traces with 5,142 preemptions — every invariant after EVERY step, never sampled. Asserted: M4's I1–I12 via `bp_check`, and S1 · S2 (watermark, from the state BEFORE the step) · S3 · S4 · S5 (exact identity, not a bound) · S6 (pressure run vs infinite-pool reference, streams byte-identical, preemptions asserted >0 tight and ==0 reference) · S7 · S8. ⚠ **S4 IS ASSERTED AS PUSH-FRONT ONLY** — that the victim heads WAITING, which is what the starvation argument rests on; WHICH request is chosen is not observable from outside and the implementation deliberately refines the rule (victim's arrival ≥ the triggering row's, a MEMORY-SAFETY constraint: rows are built oldest-first, so plain youngest-RUNNING can close a `Kv` the batch already references). ⚠ **THE WATERMARK IS EFFECTIVE ENOUGH THAT `pressure` MUST BE DELIBERATELY ADVERSARIAL** — a request whose whole life fits inside W's 8-block reserve cannot outgrow what admission checked, so short requests under a small pool SERIALISE rather than thrash; the first pressure trace never preempted once and would have banked as a passing S6 test proving nothing. **[1] does NOT cover [2] the bars, [3] ITL, [4] S3 fanout or [5] the soak under ASan/TSan — the sim's cost model is an INPUT, not a measurement.** ▶▶ **BLOCKING DECISION, taken 2026-08-19 before any scheduler code: `superpowers/memos/2026-08-19-m5-admission-reservation-DECISION-OPUS.md`. M5 admits on `blocks(prompt) + W`; M4's `bp_admit` reserves the WHOLE context (I8), under which decode-time exhaustion is UNREACHABLE and with it gate item [6], the `pressure` trace, S4 and S6 — four things that would assert green over a branch that never executes. Resolved by splitting `reserved` from `cap_blocks` (I8 stays literally true; its *implication* that growth always succeeds was never an invariant), adding **I12**, and leaving `bp_admit` full-reservation for every existing caller. M4's F9 already named the precondition — full reservation was right "with no preemption path (M4 has none)" — and M5 builds that path. Direction check recorded for Kabir; if he prefers full reservation, [6]/`pressure`/S4/S6 must be STRUCK, not asserted.** **▶▶▶ GATE ITEM [2] IS MEASURED 2026-08-20 AND IT MISSES: serve AGGREGATE(M0) 4.938 @8 vs 17.7 and 4.594 @16 vs 16.8 (0.28x / 0.27x on a thermally plateaued box; 6.378 / 5.905 = 0.36x / 0.35x cool). Report `gate-reports/2026-08-20-m5-bars-report.md`, memo `memos/2026-08-20-m5-gate2-results-vs-physics-FOR-KABIR.md`, prediction pre-registered BEFORE the first cell at `0b7f260`. ⚠ THE CAUSE IS ARITHMETIC, NOT BATCHING AND NOT BANDWIDTH. A step at N rows streams the weights once but does N x 8.46 GFLOP, which amortises not at all. MEASURED (4 counterbalanced passes, `gate-reports/2026-08-20-m5-flatness-report.md`): N=1/2/4/8/16 -> 14.73 / 15.52 / 18.68 / 20.50 / 21.08 tok/s, a 1.43x gain across the whole range where traffic-only reasoning predicts 12.0x, and N=8->N=16 buys 2.9%. **⇒ THE AGGREGATE TRAFFIC CEILINGS IN THE ENTRY NOTE (57.8 @8, 72.9 @16) DESCRIBE A MACHINE THAT DOES NOT EXIST FROM N=2 UP** -- `m5_traffic_budget.py` ends its table with 'traffic only, no compute' and that caveat was never cashed; `tests/tooling/m5_compute_budget.py` now cashes it. This does NOT retract E2 (its measured 1.12x at ctx 1500 is real) -- it says the slack E2 reported was not there to spend. **THE SCHEDULER IS EXONERATED**: preemptions 0 in every cell, and the STATIC path clears both bars on decode (17.4-17.9 at the plateau, 23.07/22.65 cool), which is the playbook's attribution ladder rung (1) answering NO. The serve path is charged for prefill INSIDE the M0 span -- llama-server is charged identically and still read 17.7, because it prefills at 91.3 tok/s against stoke's 32.2. **NEEDED: a uniform 2.75x on arithmetic; the same-sitting prefill-vs-prefill gap to llama is 2.83x** (772 vs 273 GFLOP/s), so the headroom exists on this silicon but with no slack. ⚠ CORRECTED IN FLIGHT: an earlier 3.96x compared llama's PREFILL to stoke's DECODE. ⚠ AND A PRE-REGISTERED FALSIFIER FIRED -- 'flat in N' was WRONG (N=8 landed 32% over N=2 against a 20% falsifier) because the compute ceiling is a function of TILE AND THREAD OCCUPANCY, not a constant: 120 GFLOP/s at N=1 rising to 172 at N=16, of which 1.155x is thread occupancy alone (cores busy 5.80 -> 6.70). 'Saturating, and already saturated where the gate measures' is what the data supports. **GATE [3] ALSO MISSES ON THE REAL S2 SHAPE**: ITL max 4713 ms @8 and 5454 ms @16 against the 250 ms bound, confirming the 08-19 memo on a workload it called indicative. **OPEN FOR KABIR, AND IT IS ONE QUESTION NOT THREE** (SIMD in the attention dots, a repacked GEMM, and [3]'s bound are all downstream of it): keep bit-exactness with the M3/M4 parity lineage, or buy the arithmetic? ⚠ BUT SEE MEMO 5b -- THAT QUESTION IS PREMATURE. Inside a super-block the kernel is ENTIRELY INTEGER (floats only at the boundary, twice), integer addition is associative, so regrouping there is bit-exact BY CONSTRUCTION and only the cross-super-block f32 accumulation is pinned -- `simd.c:75`'s own founding principle. The 2.83x is MEASURED and UNATTRIBUTED; attribute it before trading anything for it.** **▶▶▶ THE 2.83x IS ATTRIBUTED 2026-08-20 (memo `memos/2026-08-20-m5-attribution-RESULTS.md`, prediction pre-registered at `b6de2b1`): IT IS THE KERNEL, AND IT IS NOT INSTRUCTION COUNT.** Thread sweep, both engines, order counterbalanced (`gate-reports/2026-08-20-m5-attrib-threads-report.md`): single-thread gap **3.22x**, and ~3x at EVERY thread count -- a per-thread difference, not a parallelisation one. **At t=8 stoke scales 4.54x against llama's 4.30x, so the pool is not merely exonerated, it is a STRENGTH**, and none of this gap is the scheduler or M5. ▶▶ `objdump` then found the per-sequence mins term fully unrolled and ENTIRELY SCALAR -- ~40 uops beside the vector dot's ~57, i.e. ~40-45% of emitted per-sequence work that is not the dot product -- and replacing it with `vpmaddwd` is **bit-exact BY CONSTRUCTION** (all integer, associative, overflow bound derived from `quantize_row_q8_k`: |bs| <= 2048, pairs <= 4096 < 32767, x mins <= 63 -> 258,048, madd -> 516,096). Measured interleaved with the arm order alternating: **B/A = 1.063x at t=1 and 1.079x at t=8, B winning 6 of 6 pairings.** ⇒ **REMOVING ~40% OF THE EMITTED INSTRUCTIONS BOUGHT ~7% OF THE TIME. The uop-count theory of the gap is DEAD** -- the kernel runs at ~2.5 uops/cycle against a 6-wide front end, so it was never front-end bound and the scalar `imul` chain overlapped with the vector work on separate ports. What holds it at 46 GFLOP/s single-threaded against llama's 149 is port pressure, dependency latency or memory. Candidates named but UNMEASURED: `vpmulld` on the critical path (2 uops, ~10-cycle latency, 8x per super-block per sequence, purely to apply a 6-bit scale -- llama defers scales to a float FMA, which is NOT bit-exact); activation re-reads; `wnib[8]` holding half the register file live. ⚠ **A SINGLE UNPAIRED PRE-FLIGHT SAMPLE READ 0.957x -- B SLOWER** -- and the interleaved run reversed it; the cheap reading would have discarded a change that wins. ▶▶▶ **THE GATE IS RUN AND THE DEFAULT IS FLIPPED 2026-08-24: `STOKE_MINS_SIMD` now defaults to 1.** Report `gate-reports/2026-08-24-m5-mins-verify-report.md`, raw log `gate-reports/2026-08-24-m5-mins-verify.txt`, runner `tests/tooling/run_m5_mins_verify.ps1`, 46 min. **14 cases x 2 arms = 28 stream comparisons, every one bit-identical to an ARCHIVED binary** (never to another arm): 8 Q4_K_M `run` cases (p00/p41 x kv f32/q8 x t 2/6) + 2 Q8_0 `run` cases vs the m3 anchor + 4 `batch` cases (seqs 2/8/16). Suite 14/14 + 4 goldens green on the flipped default, and `test_c_oracle` is an INDEPENDENT check because it carries its own scalar mins loop and cross-checks it against the library kernel. ⚠ **AND THE GATE CARRIES ITS OWN PROOF IT COULD FAIL**: a third binary with ONE WRONG LANE in `dup` ran inside the gate on one case per leg and DIFFERED on both -- without it, 28 SAMEs would have been equally consistent with a kernel that never executed. ⚠ **THAT WAS NOT HYPOTHETICAL, AND THE 08-20 CLAIM ABOVE IS CORRECTED: `stoke run` DOES reach this kernel.** "The byte-compare had to run through `batch --seqs N`, not `stoke run`" is true of DECODE and false of the command: E3 chunks PREFILL through `model_forward_batch` (default `STOKE_PREFILL_CHUNK=128`) and `vec.c` tiles at `KERNELS_BATCH_TILE=16`, so `run` enters at n_seq=16 then the chunk remainder -- which is why the 10-case lineage grid is gradeable on the run path at all. **⇒ The flip cannot move P-BAR (N=1 decode never enters the batched kernel) and cannot rescue gate [2] (~7-10% on prefill against a required 2.75x).** ▶▶▶ **`vpmulld` IS PRICED 2026-08-24 AND THE SCALE IS NOT THE LEVER EITHER: 1.121x at t=1, B winning 4/4** (memo `memos/2026-08-24-m5-mullo-ablation-RESULTS.md`, data `gate-reports/2026-08-24-m5-mullo-ablation-report.md`, prediction pre-registered at `memos/2026-08-24-m5-mullo-ablation-PREREGISTERED.md` = 1.15x accept 1.10-1.25x, **inside the band, and NEITHER discriminator fired**). Deleting `vpmulld` outright -- **2 of the 4 p01 uops in the inner loop, half the vector-ALU pressure** -- buys 12.1%. **⇒ THAT IS THE SECOND COMPUTE ARGUMENT TO COME IN AN ORDER UNDER ITS COUNT: ~40% of emitted instructions bought 1.07-1.10x on 08-20, and ~50% of p01 uops buys 1.121x now. The port-throughput theory dies with the front-end one -- this kernel is bound by nothing that can be counted statically.** In log terms against the 3.22x kernel gap that is **9.8% of the way**, and **both named compute levers TOGETHER cap at 1.232x = ~18%**. ⇒ **ANSWER TO §5 FOR THE SCALE SPECIFICALLY: crossing the bit-exactness line to buy it is not worth it** -- llama's deferred-float-scale design is capped at 12.1% and would collect less, since a float FMA is cheaper than `vpmulld`, not free. The BIT-EXACT alternative (pack two sub-blocks' `sub` to i16 -- bounded by 7680 -- and apply both scales with ONE `vpmaddwd`, trading 6 p01 uops for 2 p01 + 2 p5) is capped by the same ceiling and would collect ~5-8%: free in exactness terms, worth doing eventually, NOT the lever. **⇒ WHAT IS LEFT IS A MEMORY HYPOTHESIS, and both negative results point at it: activation re-reads, 16 x 256 B = 4 KB of activations against 144 B of weights per super-block, a 28:1 byte ratio re-read for every one of `n_out` rows.** ▶▶▶ **THAT HYPOTHESIS IS DEAD, MEASURED 2026-08-27 (`bf9155f`; instrument `1b879a7`, pre-registration `memos/2026-08-27-m5-activation-reuse-PREREGISTERED.md`, report `gate-reports/2026-08-27-m5-actreuse-report.md`).** New instrument `stoke bench --matvec-batch` — the batched K-quant kernel timed DIRECTLY, closing the recorded gap that made every published kernel ratio an end-to-end prefill ratio diluted by an unmeasured factor. Arms: **A** = N distinct activation buffers · **B** = every activation pointer ALIASED to one buffer (same loads, same arithmetic, one cache-resident working set) · **C** = a NULL arm, a second set of distinct buffers. **Largest A/B over every batched cell 1.111 against a MEASURED harness floor (A/C) of 7.0%, and the decisive cell — `ffn_down` N=16, the LARGEST tile at 173.4 KiB and deepest into L2 — reads 0.996 with 33 of 72 pairings.** Pre-registered P1 (≤1.15) HELD at 1.088, P2 (≤1.25) HELD at 0.996. **THE ALIAS IS APPLIED IN THE CALLER**, because editing the kernel to read `arows[0]` makes the per-sequence body loop-invariant and GCC hoists it — an ablation measuring its own compiler. ⚠ **ARM C EXISTS BECAUSE THE FIRST RUN'S N=1 CONTROL FIRED (1.038 where nothing can be aliased); the withdrawal condition said no cell is readable until that is EXPLAINED, so the null arm holds the physics constant and varies only addresses — whatever it reports is what the harness cannot resolve.** It also checks BOTH directions: C must AGREE with A byte for byte while B must DIFFER. ⇒ **WITH THE FRONT-END (~40% of instructions → ~7%) AND PORT-THROUGHPUT (half the p01 µops → 12.1%) THEORIES ALREADY DEAD, NO NAMED LEVER REMAINS FOR THE 3.22× KERNEL GAP.** The pre-registration named this branch before the run: the honest reading is that the gap is llama's **ALGORITHM** — tinyBLAS register-blocked GEMM reusing weights AND activations out of registers — not a defect in this kernel. **stoke's kernel is a matrix-VECTOR product run N times; llama's is a matrix-MATRIX product. That is a REDESIGN, not a lever, and it is M6/M7-sized.** ▶ **POST-HOC, FLAGGED AND NOT GRADED: the new `A GF/s` column collapses with `n_in` at IDENTICAL work** — `attn_q`/`attn_out` are transposes (same FLOP, same weight bytes, same body count) and read 65.84 vs 36.94 GF/s; `ffn_gate`/`ffn_down` 70.19 vs 10.83. Arm B rules out the activation side as the cause. ⚠ **But the ABSOLUTE rate moved 1.8× on a harness change alone (an earlier 3-tensor/2-arm run read `ffn_down` at 19.76), so the DIRECTION reproduces and the SIZE does not — nothing is built on it.** Next: sweep `n_out` at fixed `n_in`, one tensor per process. ▶▶▶ **DONE SAME DAY (`9e19895`, report `gate-reports/2026-08-27-m5-transpose-report.md`) AND THE OBSERVATION DIES — BUT IT HANDS OVER A REAL DEFECT.** ⚠ **First, one row was NEVER a comparison: `blk.0.ffn_down.weight` is Q6_K, not Q4_K** (Q4_K_M upgrades it — 210 B/super-block against 144, 20.4 MB against 14.0, and a DIFFERENT kernel), so the `ffn_gate` vs `ffn_down` 6.5× is WITHDRAWN. **A fixture drawn BY NAME from a real model can differ in exactly the property the comparison rests on, and nothing in the table said so — the instrument now prints the QUANT TYPE and weight bytes on every row.** ⇒ **The surviving pair was clean and dies too: `attn_q` and `attn_output` are exact TRANSPOSES (both Q4_K, both 5,898,240 bytes, same FLOP, same 40,960 unpacks) and read 65.84/36.94 = 1.78× contaminated but 71.75/67.88 = 1.057× ISOLATED** (one tensor per PROCESS, 4 blocks, order reversed, pinned). **H-ARTIFACT fired against its pre-registered <1.30× band; all three Q4_K tensors land within 5% of each other at N=16 regardless of shape.** Two contaminants, both removed by `--tensor`: earlier tensors' weights in L3, and **the activation scratch sized to `n_in_max` over the whole SET, spacing 16 regions 11 KB apart while measuring a tensor touching 2.9 KB of each** — invisible, and the more instructive. **The activation-reuse conclusion is untouched: it is a RATIO between arms inside ONE timing loop, the one quantity this contamination could not reach.** ▶▶▶ **AND THE ISOLATED TABLE FOUND A REAL DEFECT: THE BATCHED Q6_K KERNEL GOES BACKWARDS.** Every Q4_K tensor CLIMBS with N (38→70 GF/s); `ffn_down` (Q6_K) reads **33.42 at N=1 then 17.64 / 19.27 / 20.44 / 10.89** — **the SOLO kernel is fine and BATCHING makes it ~2× WORSE where it makes Q4_K ~2× better.** Cause legible in `simd.c`: per sequence per super-block Q4_K runs 8 × `vpmaddubsw`+`vpmaddwd` at **32 int8 products per instruction** while Q6_K runs 32 × `kq_load_i8x8_to_i32`+`kq_mac8`, **WIDENING i8→i32 for 8 products per operation** — same 256 elements, ~4× the µops, plus the widened `aux8` re-read per sequence, **and the per-sequence body is exactly what batching multiplies by N**. ⇒ **THE PRIZE IS ARITHMETIC: `ffn_down` is Q6_K in 18 of 36 layers and 24.7% of per-layer matvec MACs, so moving 0.897 GFLOP/token from 10.9 to 70 GF/s takes prefill's matvec portion from 0.173 to 0.104 = ~1.67×. Against the ~2.75× gate [2] needs, that is the LARGEST LIVE LEVER this project has and the first in weeks that is a MEASURED DEFECT rather than a hypothesis.** ▶ **A BIT-EXACT FIX EXISTS ON PAPER**: `vpmaddubsw` takes its first operand UNSIGNED, so keep the 6-bit weight raw in [0,63] and carry the −32 offset separately — the shape Q4_K already uses for `mins` — `SUM (w−32)·a·s == SUM w·a·s − 32·SUM a·s`, and `SUM a` is what Q8_K's `bsums` already stores. Bounds hold (63×127×2 = 16,002 vs i16; ×127 ×2 = 4,064,508 vs i32), and a 32-byte `vpmaddubsw` spanning TWO of Q6_K's 16-element scale groups is applied exactly by a following `vpmaddwd` against `[s_j ×8, s_{j+1} ×8]` since `(a+b)·s == a·s + b·s` in integers. ⚠ **NOT ATTEMPTED — a shipping kernel with a bit-exactness acceptance criterion needs its own mutation-backed gate. KABIR'S GO/NO-GO.** ▶▶▶ **BUILT SAME DAY ON HIS WORD (`8155244`, report `gate-reports/2026-08-27-m5-q6k-ab-report.md`, ADDENDUM 20): `vec_dot_q6_k_q8_k_xn_maddubs_avx2` IS BIT-EXACT AND 7.68× AT N=16.** One binary, `STOKE_Q6K_KERNEL` selecting at runtime (so NO code-layout confound — the thing the `vpmulld` arms had to engineer around), 4 blocks × 24 cycles, `ffn_down`, one tensor per process, pinned, order reversed on even blocks, and an unrecognised env value REFUSED with the runner checking that first. **ref 36.62 / 17.17 / 19.13 / 18.66 / 10.90 GF/s at N = 1/2/4/8/16 · maddubs 31.56 / 42.11 / 56.80 / 75.36 / 83.68 ⇒ B/A 0.86× / 2.45× / 2.97× / 4.04× / 7.68×, and maddubs CLIMBS with N like every Q4_K tensor where ref goes backwards.** ▶ **EXACT BY CONSTRUCTION**: `vpmaddubsw` is unsigned-first, so keep the 6-bit `u` raw in [0,63] and carry the −32 offset on `bsums` — `SUM (u−32)·a·s == SUM u·a·s − 32·SUM a·s` — with every intermediate inside its type (16,002 vs i16; 4,064,508 and 260,128,512 vs i32), and a 32-byte `vpmaddubsw` spanning TWO 16-element scale groups applied exactly by `vpmaddwd` against `[s_2c ×8, s_2c+1 ×8]`. ▶▶ **PROVEN, NOT ARGUED, AND IT CLOSES A COVERAGE GAP: `tests/test_simd.c` asserts the twin bit-identical to the reference twin AND to the SOLO kernel, per sequence, 600 iterations over n_seq 1/2/3/8/15/16 and n 256..9728 — THE FIRST DIRECT TEST ANY `_xn_` KERNEL HAS HAD** (they were covered only through `matvec_batch`'s whole-matvec byte-identity), **with an anti-vacuity check that the sequences differ, since a kernel ignoring `s` would otherwise pass everything.** ⚠ **P1 FALSIFIED AND THE BAND WAS MINE AND WRONG: I predicted the N=1 control (where both arms run IDENTICAL code) within ±7%, but 7% is an IN-PROCESS null-arm floor and this is a BETWEEN-PROCESS ratio.** Left standing as falsified rather than re-fitted; a `ref2` arm (same kernel, same env, different process) measures the right floor and it must be read PER CELL — **N=1 nulls at 30% (first cell measured, absorbs load/warm/page placement), N=16 at 1.0%. Against 1.0%, 7.68× is not a noise question.** ⏹ **DEFAULT STILL `ref` (`STOKE_Q6K_MADDUBS` 0) — the tree ships what it shipped this morning. Suite 14/14 + 4 goldens. NEXT: the end-to-end STREAM gate against the ARCHIVES with a mutant that must differ (an arm is graded against an archive because agreement among new arms is also consistent with all of them being wrong together), then the flip, then re-price prefill end-to-end — 7.68× on one kernel is NOT 7.68× on a token.** ⚠ **INSTRUMENT GAP: `bench --ops` has NO K-quant entry** (only rmsnorm/softmax/swiglu/rope/argmax/q8_0), so every kernel ratio this project has published is an END-TO-END prefill ratio diluted by an unmeasured factor -- at f~0.8 the kernel-level figure behind 1.121x is ~1.16x. ⚠ **AND A MECHANISM I PROPOSED WAS MEASURED AND KILLED**: B trended +6.6% at t=8 while A held flat, I attributed it to B drawing less power and holding a higher clock, and `% Processor Performance` says both arms reach ~185 -- it was noise (the noisy arm is A at t=1 and B at t=8, so it is not an arm property). **▶▶▶ GATE ITEM [4] IS RE-DIAGNOSED 2026-08-24 AND IT IS BLOCKED ON DEDUP, NOT ON PROMPTS** (memo `memos/2026-08-24-m5-gate4-s3-status.md`). **DONE: S3 is exported and its shared prefix VERIFIED** — `export_bench_prompts.py --scenario s3 --verify` writes `tests/data/bench_prompts/s3-fanout-c{00..07}.txt` from the same `bench/scenarios.py` the baseline was served, flattened `system + "\n" + user` so the shared text is a byte-identical PREFIX; `stoke tokenize` then confirms **common token prefix = 1028 tokens = 64 full 16-token blocks adoptable by dedup** (the exporter REFUSES to bless a set whose prefix is shorter than the scenario's, and that guard fired immediately — `stoke tokenize` prints a bracketed comma list, my first parser split on whitespace and produced ZERO ids, which a lenient comparison would have reported as success). **⇒ "no system-message seam" was NOT the obstacle.** ⚠ **THE REAL BLOCKER: `--dedup` exists only on `cmd_batch`; `serve.c` and `sched.c` contain the string nowhere, so the M5 serving driver cannot exercise the one capability S3 exists to measure.** Worth ~2x on this scenario (no dedup: 8x1224 = 9,792 prompt tokens prefilled, est. ~0.42x of the bar; with dedup: 1,224 + 7x200 = 2,624, est. ~0.89x). ⚠⚠ **AND THE BAR ITSELF SETTLES IT: 13.3 WAS NOT MEASURED WITH PREFILL CHARGED FRESH.** `llamacpp-default` passes no flags, and b10037's own `--help` shows **`--slot-prompt-similarity` default 0.10 (enabled), `--cache-ram` default 8192 MiB (enabled), `--cache-idle-slots` default enabled** — S3's clients share ~83% of their tokens and closed-loop re-sends match their own slot at ~100%. **⇒ A no-dedup stoke reading is NOT like-for-like and MUST NOT be published as gate [4]; dedup-on-serve is the precondition for the comparison being fair, not stoke's optional advantage.** The seam is `admit_if_possible` (call `kv_adopt_prefix`, start `n_done` there; the prefill loop's `chunk = target - n_done` then skips them) ≈ 15 lines, but four hazards make it a day: **S5 is asserted as an EXACT identity** and adoption makes `n_done` start above 0; **preemption resets `n_done = 0`** while `kv_adopt_prefix` is ADMISSION-TIME ONLY; the sim's cost model is an INPUT so preemption counts will move; and it needs its own byte-compare gate with a mutation, since "same claim, new caller" is what the entry gate exists for. â–¶â–¶â–¶ **DEDUP ON THE SERVE PATH SHIPS 2026-08-25, AND THE FIXTURE GATE [4] WAS TO BE GRADED ON IS BLIND.** `sched.c`'s `admit_if_possible` now calls `kv_adopt_prefix` after `kv_open_in_partial` and starts `n_done` there, with `model.c`'s residency guard (`kv_len(kv) != adopted` â‡’ fatal) carried over; `sched_open` takes `bool dedup_prefix` so all four call sites state which run they are; `bench --serve` gains `--dedup` and `--ids <file>` (streams in `stoke batch`'s exact shape, to a file, so the correctness artifact and the measurement artifact are not the same artifact). **The four hazards the 08-24 memo sized: S5's exact identity was NEVER at risk** (adoption moves where prefill STARTS, not where it ends â€” the DERIVATION needed amending, not the invariant); **re-adoption after preemption falls out** (preempt closes the victim's `Kv`, so readmission opens a fresh one, which is the one state `kv_adopt_prefix` accepts â€” measured: the tight-pool arm adopts 1024 over 9 admissions with 3 preemptions); **the sim's traces did not move**, because the new fanout trace is addressed by name and is NOT in `TR__COUNT` â€” `test_sched` with no arguments still prints **1,000,130 steps over 11,870 traces, 5,142 preemptions**, gate [1]'s banked numbers unchanged. âš  **THE SIM CAN ONLY SEE ADOPTION BECAUSE THE FAKE HOST NOW NOTES TOKENS** (`kv_note_token` per row, as `model_forward_batch` does): without it a block's hash stays 0 and `bp_dedup_find` can never match, so a dedup test would report "nothing adopted" for a reason unrelated to the scheduler. Cost, measured: the 1e6-step soak went **1.2 s â†’ 2.4 s**, so a future gate [5] leg buys ~half the steps per hour and each step exercises `bp_seq_set_token` on every row.  âš âš  **THE SESSION'S REAL FINDING, AND IT IS ABOUT THE DATA, NOT THE CODE: M0's BENCH PROMPTS CANNOT SEE A DEDUP DEFECT.** Two mutants were built to prove the byte-compare could fail â€” `skip16` (skips one block MORE than it adopted, over positions that are backed but never written: the failure `kv.h` names as the one that leaves the run alive and the output merely wrong) and `kv` (zeroes the K/V under the WHOLE adopted prefix, all 36 layers, all 8 kv heads). **At pre-flight size on S3 (2 clients Ã— 4 generated) BOTH produced byte-identical ids.** Two independent reasons, neither in the scheduler: **(1) M0's clients differ by ONE DIGIT** â€” diff `s3-fanout-c00.txt` against `c01.txt` and the only difference in 5,957 bytes is `[client 0]` vs `[client 1]`; S2 is built the same way â€” so all eight clients generate the SAME ids and a cross-client adoption error is indistinguishable from correct behaviour **at any N** (measured at 8: **1 distinct stream over 8 clients**); and **(2) the continuation does not depend on the prefix** â€” a zeroed key scores qÂ·k = 0, weighted `exp(0)` against `exp(large)` for every real position, and a zeroed value adds nothing to the sum. âš  **Reason (2) is SIZE-DEPENDENT and the honest statement says so**: at the gate's own size (8 Ã— 24) the `kv` mutant DOES differ on S3 while `skip16` still does not. â‡’ **A gate over S3 alone would have reported three SAMEs and proved nothing.** Fixed by a fixture, not a disclaimer: `tests/data/bench_prompts/fanout-hetero-c{00..07}.txt` keeps S3's real shared system text (same `build_messages`, same adoptable run of blocks), appends eight records to it, and gives every client a different question that needs a different record â€” so the streams are pairwise distinct (asserted by the exporter AND re-asserted by the runner) and every answer needs the shared prefix. On it, both mutants DIFFER.  âš  **AND A SECOND INSTRUMENT BUG THE GATE CAUGHT ON ITSELF: `(n-1) Ã— shared_prefix` IS THE WRONG EXPECTATION.** A green run failed the exact-adoption check at 1168 measured against 1152 predicted. **Adoption is PAIRWISE**: one prefill runs at a time (S1), so by the time client k is admitted every earlier client's blocks are resident and `bp_dedup_find` matches the longest run k shares with ANY of them â€” at least, and usually more than, what all n share. The 8-way common prefix is a LOWER BOUND per client. Fixed by having `export_bench_prompts.py` emit a MANIFEST carrying `expected_adopted[n]` computed with the engine's own tokenizer, which the runner READS â€” a runner that recomputes the expectation in its own arithmetic is a second implementation of the thing under test. Same family: the arms moved to `--raw` because `stoke tokenize` (which writes the manifest) is raw while `bench --serve` applies ChatML by default.  âš  **THE SIM'S OWN HEADLINE ASSERTION WAS VACUOUS AND A MUTANT PROVED IT.** "The id streams are identical with and without adoption" PASSED under `skip16`, and had to: `fake_token` depends on (request id, position), a request's decode positions start at `n_prompt` however its prefill got there, so no prefill-side defect can move a sim stream. Replaced by a POSITION LEDGER â€” the sim records which positions actually got a row, and the test asserts the uncovered run is exactly the adopted prefix, that no HOLE sits between it and the computed rows, that the uncovered total equals what the scheduler REPORTED adopting, and that the rows run to each request's last decode position. The mutant now fails it (ledger 1008 uncovered against 896 reported). â‡’ **Coverage stated honestly: the sim catches wrong POSITIONS, the real-model byte-compare catches wrong CONTENT, and on M0's own prompts neither catches anything.** ▶▶▶ **GATE ITEM [4] IS MEASURED 2026-08-26 AND IT IS MET: S3 AGGREGATE(M0) median 13.535 against the 13.3 bar (1.018×), and ALL THREE counterbalanced blocks clear it on their own (13.757 / 13.535 / 13.457) — the median is a plateau reading, not the cold block.** Report `gate-reports/2026-08-26-m5-gate4-report.md`, 12 units, ~50 min, `tests/tooling/run_m5_gate4.ps1`. The gate arm is `bench --serve --dedup` because the bar's referent (llama-server in its defaults) has prompt-KV reuse ON; **the no-dedup FLOOR is 5.926 = 0.446×**, so **dedup is worth 2.284×** and is the difference between missing by 2.2× and clearing. The 08-24 memo's pre-registered estimates (~0.42× / ~0.89×) both landed. **Four self-checks close exactly:** adoption 7,168 = the manifest's `expected_adopted[8]`; rows not run 12,296→5,128 = 7,168 EXACTLY; prefill chunks 80→24 = 56 × 128 = 7,168, steps 379→323; **preemptions 0** in all six stoke units. Streams byte-identical across all six (ids AND decoded text) — ⚠ a same-computation check for the rates, **NOT an independent correctness gate**, since S3 is the fixture ADDENDUM 13 proved blind; the dedup correctness gate stays the HETERO fixture at `4dd7e2f`. **The scheduler is exonerated tighter than before: Sd's decode-only 19.85 against the static path's 20.180 — within 1.6% while also carrying every arrival's prefill inside its span.** ⚠⚠ **[4] IS THROUGHPUT AND [3] STILL MISSES ON THIS SAME DATA — ITL p50 403 ms, max 5,118 ms, TTFT p50 75.1 s.** ▶▶ **THE SAME UNITS SPLIT [3] INTO THREE MECHANISMS THAT HAD BEEN READ AS ONE NUMBER: dedup moves ITL p95 4,290 → 438 ms (9.8×) and leaves the max unchanged (5,050 → 5,118).** p95 = prefill-chunk blocking (56 of 80 chunks gone); max = the FIRST prefill, irreducible by dedup; **p50 = the step time itself — and the evidence is B's INDEPENDENTLY measured static decode aggregate 20.180 agreeing to 1.6% with the 19.85 implied by the p50, NOT the `N / ITL p50` arithmetic, which merely restates it. ⇒ p50 ≤ 250 ms needs the STEP ~1.6× faster (decode ≥ 32 tok/s) = the same ~2.75× arithmetic wall, and NO chunk width or scheduling change reaches it.** ⚠ **THE RUNNER HAD NEVER BEEN EXECUTABLE**: seven defects found by a parse check and one 6-min `-Quick`, two of them fatal — `"$id:"` is a SCOPE reference so the file did not parse at all, and `-Quick` shared the gate's BANK, so a pre-flight would have been RESUMED AS the gate result and published. Fixed with shape-keyed unit ids, a `-quick` stamp, and a report that refuses to median across shapes. **A desk review is not a pre-flight.** ▶▶▶ **SAME DAY — PREFIX DEDUP IS NOW THE SERVE DEFAULT, ON KABIR'S WORD.** `bench --serve` defaults to dedup ON; `--dedup` is still accepted and still means on (so every banked command line reproduces verbatim), and **`--no-dedup` is how you turn it off**. **The flip's own proof: the new default's stream is BYTE-IDENTICAL to the PRE-FLIP binary's `--dedup` artifact AND to its no-dedup artifact** (one hash across five runs: new default, new `--dedup`, new `--no-dedup`, old `--dedup`, old no-flag), suite **14/14 + 4 goldens**, and `test_sched` still prints gate [1]'s exact **1,000,130 steps / 11,870 traces / 5,142 preemptions**. ⚠ **THE FLIP'S REAL HAZARD WAS THE MEASUREMENT ESTATE, NOT THE ENGINE: three runners meant "off" by OMITTING the flag**, so the flip would have silently turned `run_m5_gate4.ps1`'s FLOOR arm into a second gate arm (reporting "dedup bought 1.00×"), `run_m5_dedup_verify.ps1`'s arm A into a copy of arm B, and — worst — **`run_m5_bars.ps1` into a re-measurement of gate [2]'s banked bars on a workload those bars never described** (S2's clients differ by one digit, so dedup would adopt nearly everything). All three now say `--no-dedup` out loud. **The serve driver also prints `dedup on|OFF` in its header and a dedup line in BOTH directions**, because while dedup was opt-in an absent line meant "off" and after the flip it would mean nothing. `stoke batch --dedup` is UNCHANGED and stays opt-in: it is an attribution instrument, not a serving path. ▶▶▶ **SAME DAY — GATE ITEM [3] IS RULED, INSTRUMENTED AND MEASURED, AND IT MISSES BY 4.4×.** Ruling `memos/2026-08-26-m5-gate3-criterion-RULING-OPUS.md` committed at `23fcb35` BEFORE the run; report `gate-reports/2026-08-26-m5-gate3-report.md`; runner `tests/tooling/run_m5_gate3.ps1`. ⚠ **THE CRITERION SAYS SOMETHING NARROWER THAN THIS FILE'S OWN SHORTHAND.** The playbook's text is "no decode stall **>250ms during co-running prefill**", not `ITL <250ms` — the subject is the wait a decode suffers BECAUSE a prefill chunk shares its step, not the steady-state cadence. **Ruled: that reading governs** (the 08-19 memo already analysed [3] that way and swept CHUNK WIDTH against it, which is coherent only under it; and the alternative makes [3] a second unlabelled bar over the quantity [2] grades). ⚠ **NOT a re-anchoring — the bound stays 250 ms and [3] still misses.** ⚠ **THE INSTRUMENT COULD NOT REPORT IT**: `bench --serve` pooled every gap into ONE distribution over TWO populations, so a percentile off it tracked how many prefill chunks the run happened to have. Each token now records whether its step carried a prefill chunk, and the driver prints `ITL stall` and `ITL cadence` beside the pooled line. **MEASURED (5 cells, palindromic order): stall p50 4,843 ms at chunk 128 and 1,108 ms at chunk 16 — the SMALLEST LEGAL width, since `BP_BLOCK_TOKENS` is 16 and M4 froze it, so the chunk lever is EXHAUSTED, not untried. ⇒ [3] MISSES at 4.4×, which is LARGER than [2]'s 2.75×: [3] is the STRICTEST item in the M5 set and is downstream of the same arithmetic wall.** ⚠⚠ **AND THE 9.8× ITL p95 GAIN PUBLISHED THIS MORNING IS NOT A LATENCY GAIN — WITHDRAWN AND CORRECTED.** Pre-registered P2 ("stall p50 is the SAME in both arms") **HELD at 7.8% apart** (5,339 dedup vs 4,923 no-dedup): **dedup does not shorten one stall, it makes them rarer — 56 against 280, 2.3% of gaps against 11.7% — and at 2.3% the pooled p95 no longer reaches the stall population, so the dedup arm's 428 ms p95 is a CADENCE number.** "ITL p95 4,290 → 438" was a percentile crossing a population boundary. Also withdrawn: "[3] misses on the median as well as the max" — the median is the cadence and was never [3]'s subject; the gate [4] report now refuses to be read as grading [3]. ▶ **TWO PREDICTIONS FALSIFIED, BOTH BY ASSUMING A QUANTITY SCALES WHEN IT DOES NOT, BOTH WITH EXACT CORRECTED MODELS: P3** — stalls are not `chunks × n_gen`; a chunk stalls only the streams ALREADY DECODING and prefills run one at a time (S1), so client 0 stalls nobody ⇒ **2 chunks/client × (1+…+7) = 56 with dedup, 10 × 28 = 280 without, both measured EXACTLY**; **P5** — a 16-token chunk does not cost ⅛ of a 128-token one, because **the implied prefill rate FALLS with the width (24.0 → 14.4 tok/s)**, so the stall shortens sub-linearly and the shortfall WORSENS relative to width (21.4× at 128, 4.4× at 16), against the 08-19 extrapolation's ~2.5×. **P1, P2 and P4 HELD** (cadence 401.9 against a predicted 403), so the model — a decode waits its step and the step is one chunk of prefill — is confirmed; only its scaling assumptions were wrong. ⚠ Drift **+20% to +28% across 31 min**, monotonic; **the palindromic cell order is what makes P2 gradeable at all** (Sd-early vs Sn-late reads 11%, the reverse 25%, on either side of the 20% band), and every verdict clears its bound by >4× so drift cannot flip one. **⇒ [3]'S OPTIONS ARE NOW FOUR PRICED ONES, NOT A QUESTION ABOUT A NUMBER: (a) 4.4× on prefill at chunk 16 — the arithmetic wall, and the named compute levers cap at ~18%; (b) `BP_BLOCK_TOKENS` < 16, reopening M4's closed set, and P5 says the return is sub-linear; (c) a scheduler that does not hold decode rows behind a prefill chunk's step — undesigned, and it trades against the amortisation E3 measured saturating by ~23 rows; (d) accept [3] as a documented miss with a named cause and a measured price. KABIR'S CHOICE.** ▶▶▶ **SAME DAY — GATE ITEM [6] IS MET (`abf98d2`), AND MY OWN SCOREBOARD WAS WRONG: "[3] IS THE LAST UNMET ITEM" IGNORED [6] AND [7].** [6] is "preempt-lossless demonstrated on a forced-preemption REAL-MODEL trace" and **it had never been run** — no gate report covered it and no banked serve measurement had a nonzero preemption count. ([1] asserts losslessness in the SIM, whose cost model is an INPUT — which is why [6] is a separate item.) **RESULT on `fanout-hetero` (4 clients, pairwise-distinct streams over a load-bearing prefix, 300 gen, dedup OFF): R (derived pool) preemptions 0, rows 5,877 · P (pool 170) preemptions 3, rows 9,796, stream BYTE-IDENTICAL to R · M (mutant) preemptions 3 and DIFFERS.** Verified independently of the runner: R and P share `B38F02B1D247FFE72CA98DBF` over 1,294 lines; the mutant diverges at line 542 and is 19 lines shorter. **Four claims, each able to fail alone, all PASS ⇒ preemption on the real model is LOSSLESS.** ⚠⚠ **WHY IT HAD NEVER BEEN RUN — THE DRIVER COULD NOT PRODUCE THE TRACE, for three guards each measured on the way: (1) the pool is `n_seq × blocks(ctx)`, so every sequence can always reach its own context and decode-time exhaustion is UNREACHABLE (`--pool-blocks` added for this gate, and the header prints "SQUEEZED" so such a run is never mistaken for a normal one); (2) admission approves `blocks(prompt) + W`, so a request whose whole GROWTH fits in W's 8-block reserve can never outgrow it — at n 32 a squeezed pool made the clients SERIALISE instead (rows 1,349 → 4,805, preemptions still 0), so growth must exceed W×16 = 128 tokens; (3) admission refuses to overcommit, so a merely small pool just admits fewer sequences (at pool 200, 2 of 4 admitted and both fit).** ⇒ **preemption fires ONLY for a pool inside `[k·(blocks(prompt)+W), k·blocks(prompt+n_gen))` — enough for k to be ADMITTED, not enough to FINISH; here [164,186) at k=2.** **This is the real-model form of gate [1]'s own recorded lesson that `pressure` must be DELIBERATELY ADVERSARIAL, and it is why every serve measurement this project has published reads preemptions 0.** The mutant is the shipping source built via `EXTRA_CFLAGS` with `-DSTOKE_MUT_PREEMPT` (replays one row too FEW, so the output stays fluent and is merely wrong — `kv.h`'s dangerous class); it cannot ship and prints a banner every run. ⚠ **NOT COVERED THEN, PAID 2026-08-27: THE DEDUP PATH, which is the serve DEFAULT.** ▶▶ **PREEMPT-LOSSLESSNESS IS NOW DEMONSTRATED ON THE SHIPPING DEFAULT** (report `2026-08-27-m5-gate6-dedup-report.md`, runner `run_m5_gate6_dedup.ps1`): 8 hetero clients × n 300, ctx 1600, kv f32, **dedup ON**, squeezed pool 200 — **R (derived pool) preemptions 0 and peak residency 232 blocks · P (pool 200) preemptions 2, adoption 8,064 → 10,368 so each readmission RE-ADOPTED 1,152 positions, stream BYTE-IDENTICAL to R over all 2,400 ids · MP (`-DSTOKE_MUT_PREEMPT`) diverges at id #2,068 · MK (`-DSTOKE_MUT_ADOPT_KV`, NEW — zeroes the adopted prefix's K/V THROUGH THE SHARED BLOCKS, so it damages the donor too) diverges at id #2.** Eight claims and four pre-registered predictions, all held. ⇒ **WHY 08-26's ATTEMPT FAILED, AND IT WAS NOT THE POOL VALUE: under dedup, ADMISSION AND RESIDENCY MEASURE DIFFERENT THINGS** — admission checks `blocks(target)+W` before the Kv exists (blind to adoption, deliberately) while the 2nd..kth clients adopt 72 of their 74 blocks by refcount and cost ~2 each, **so a pool below the RESIDENCY is also below the ADMISSION ceiling and the clients merely SERIALISE. The lever is PRIVATE GROWTH — the shared prefix is paid once, every generated token is private — so widen k and n, never shrink the pool.** That is guard (2) generalised. ⇒ **NEW INSTRUMENT `peak blocks used`, sampled INSIDE `sched_step` at maximum demand (after every row is backed and after any preemption it forced, BEFORE the step's retirements): a driver polling at step boundaries misses the peak by the whole residency of the first sequence to retire. It turns "the pool is below what the workload needs" from ARITHMETIC into a MEASUREMENT — 232 > 200, and every pool under R's peak is forced to preempt by construction.** ⇒ **AND A FINDING FOR FREE: adoption makes PREEMPTION cheap, not just admission** — decode rows are identical in both arms, so P's extra rows are all re-prefill: 529 over 2 preemptions = **~265 recomputed against a replay target of ~1,417**, because `preempt()` resets `n_done` to 0 and readmission reaches `kv_adopt_prefix` as a FRESH sequence, the one state it accepts. ⚠ **STILL NOT COVERED: Q8 KV, staggered arrivals, and the admission asymmetry itself — UNFIXED, deliberate, and an M6 question.** ⚠ **[7] (history.jsonl + ledger + MEMORY.md) CHECKED AND DELIBERATELY NOT DONE**: `bench/history.jsonl` is a MICROBENCH ledger (6 rows, last at M3) whose own label says a row "stays comparable to the K3 rows" — appending an M5 row now would assert a close that has not happened. It stays for M5 close-out. **⇒ CORRECTED M5 SCOREBOARD: [1] MET · [2] MISSED · [3] MISSED · [4] MET · [5] MET · [6] MET · [7] close-out. THE TWO OPEN GATE ITEMS ARE [2] AND [3], AND THEY ARE THE SAME ARITHMETIC WALL — nothing left in M5 is a scheduler question.** |
| M6 | `stoke.h` freeze + server | **CLOSED 2026-08-30, tag `m6` (local, not pushed). ALL SEVEN GATE ITEMS BANKED: [1][2][3][4][5][6][7]. Rungs A1–A7 done — `stoke.h` v0 IS FROZEN: error codes are never renumbered, fields are only appended, nothing is removed.** ⚠⚠ **THE FREEZE REVIEW EARNED ITS PLACE: `stoke_poll` answered an id it had never seen with 0/NONE — byte for byte what a live-but-quiet request looks like — so a caller who ignored a failed `stoke_submit` waited FOREVER, while `stoke_cancel` had always reported UNKNOWN_ID for the same mistake. Fixed, with a test PAIR (never-submitted ⇒ error; submitted-but-unstepped ⇒ 0/NONE, the control).** ⚠⚠ **AND ONE ITEM LEFT AS WORK, OWNER M7: `sched.h`'s "the submit queue is the ONE place a foreign thread touches the scheduler" is FALSE — `stoke.h` promises poll/cancel are any-thread and they reach `find()`, which walks `s->all` unlocked while the step thread prepends to it unlocked. Benign on x86, real on weak memory, and NOT covered by TSan here because no test drives submit/step/poll from three threads at once.** Narrative: ADDENDUM 24–34. Earlier detail: **Rungs A1–A5 done; A6 was CODE-COMPLETE — stop strings, the chooser (`84cf791`), its engine wiring (`29a1622`) and the server surface that reaches it (`b3cf7ff`), with a 5/5 mutation leg proving every parameter's DELIVERY can fail and does (`48e23b4`). **A6 IS COMPLETE — the rate half is discharged (ADDENDUM 33): a 6-block, 18-unit A/B of the tag-`m5` binary against HEAD with a NULL ARM, median 5.846 → 5.853 (0.12%) against a null-arm floor of 2.6%, and the A6-vs-M5 spread NARROWER than the A6-vs-itself spread. Pre-flights: byte-identical `--ids` output and `steps`=167 in all 18 units. ⚠ The BARS GATE'S ABSOLUTES ARE STILL M5's and were deliberately not re-taken — this box fails its own quiesce bar (median 21.5%, 14.5% after stopping ISLC, against two consecutive reads ≤12%), so an absolutes run is its own sitting. ⚠⚠ And the finding that shaped the design: `bench --serve` NEVER ENTERS `stoke.c`, so `sample_prologue` is not on the bars path at all — following "re-measure the bars" to the letter would have measured a path the change cannot reach.** What remains before the FREEZE is A7.** ⚠ **The byte-exact serve golden is MEASURED blind to a dropped `temperature`/`top_k`/`top_p`: it can only pin configurations whose answer is deterministic across libms, and every one of those is the greedy answer. `tests/serve_openai.py` is the sole grader for four of the five parameters.** Rung order: `docs/superpowers/design-notes/2026-08-28-m6-entry.md`; narrative in `progress.md` ADDENDUM 24–32. ⚠⚠ **THE MATRIX WAS RE-RUN ON A6 (ADDENDUM 32) and the re-blessed 25-request `serve.golden` REPRODUCES BYTE-FOR-BYTE on Linux/gcc 13.3 and again under ASan at -O1 — the deterministic-only pinning rule tested rather than asserted, in three environments. Gate [7]'s soak is RE-BANKED against `stoke sha256 879d8cb8…` (600 s, 4 clients, 219 actions / 118 answered / 0 refused, no reports) — and it found that the soak had NEVER ENTERED THE SAMPLER: its builder hardcoded `"temperature":0`, while a 400 counts as an answered exchange, so a soak of pure refusals passed every check it had. The driver now reports what its traffic contained (55 sampled / 17 penalized on the re-bank) and the gate requires the server's own log to show ≥10 completed generations.** **Gate [7] = the matrix + a 10-min randomized ASan serve smoke, and it earned its place THREE TIMES over.** Soak: 600 s, 4 clients, 51 corpus files, 298 actions / 187 answered / 0 refused, no sanitizer reports, clean exit (`stoke sha256 eac4d2b3…`). Matrix: **Windows plain GREEN; Linux plain GREEN** (the byte-exact serve golden built on Windows/gcc 15.2 REPRODUCES on Linux/gcc 13.3); **Linux ASan GREEN** (full suite, no reports); **Linux TSan GREEN, ZERO DATA RACES** (model-heavy gates skipped there with a MEASURED reason: abi_smoke peaks at 7,364,440 kB RSS under TSan against 7 GB of WSL, so it fits alone and is OOM-killed inside `make test`; every skipped gate runs and passes on the ASan leg). ⚠⚠ **THREE REAL DEFECTS ONLY THE MATRIX COULD FIND: `net.c` had not COMPILED on Linux since A3; `stoke serve` DIED WITH SIGPIPE on the first client disconnect there — Windows has no SIGPIPE, so all four serve gates were green against it; and THE IDLE REAPER DESTROYED FINISHED RESPONSES, closing CONN_WRITE whenever generation outlasted the timeout, which is ANY long generation on ANY loaded box and is silent server-side. All three fixed; the reaper has a mutation-verified regression guard (`serve_golden` runs at `--idle-timeout 2`).** | openai-python suite; M0 harness drives stoke unmodified; ABI checklist; C++ TU |
| M7 | Harden + launch-ready | **← NOW.** Entry note: `docs/superpowers/design-notes/2026-08-30-m7-entry.md`; rungs B1–B8. **B1 DONE (ADDENDUM 35): the cross-thread surface is locked. `tests/test_sched_race.c` drives the REAL scheduler over a FAKE model with a stepping thread beside a submitting/polling/cancelling one — TSan found SIX races where reading the code had found ONE, and reports zero after the fix.** ⚠⚠ **M6 called this ungradeable because TSan OOMs on model-heavy work; the real blocker was that every test was single-threaded, and "TSan GREEN, zero races" was true while covering none of it — a sanitizer result is scoped to what the suite runs CONCURRENTLY.** The lock is held for bookkeeping and RELEASED across the forward pass (holding it there is a stall, not thread safety). Cost, measured against tag `m6` with a null arm: 8f32 0.09% against a 0.25% floor, 16f32 0.03% against 3.0%. Still inherited from M6: ~~(1) the cross-thread surface~~ — ~~(2) the first-token ordering~~ — **B3 DONE (ADDENDUM 38). M6 shipped it as documented behaviour because the window is a few hundred nanoseconds; B1 had just retired that reasoning, so `-DSTOKE_MUT_WIDEN_SUBMIT_WINDOW` spins 20 ms in it and `tests/abi_firsttoken.c` compares each request's raced stream against its own quiescent one at the same seed. Unfixed: 7 of 8 pairs differed, and EVERY raced stream began with the same token (264) whatever its seed — the argmax, seen from outside. Fixed by draining the pending queue on a callback MISS, which closes it by construction; 8 pairs, 0 differing with the window still widened. ALL THREE of M6's correctness items are now closed;** ~~(3) the M5 bars' absolutes~~ — **B2 DONE (ADDENDUM 37). ⚠⚠ The gate's QUIESCE INSTRUMENT was reading noise: a one-shot `Get-CimInstance` on a FORMATTED WMI counter, returning 7-38% on a box three other instruments put at ~3%. Wrong in BOTH directions — it voided the 08-29 attempt and could equally have PASSED a loaded box, so any `quiesced: true` banked before 2026-08-30 is weak evidence. Fixed (two raw samples, delta computed here, gate-FAILING on an unusable sample); the gate now passes at 2-3% and the S blocks span 0.4% where a noisy box spanned 10%.** Bars today at 8f32: **stoke 7.264, llama-bench pp512 75.66, tg128 17.65.** ⚠⚠ **THE ABSOLUTE SWINGS 1.8× ACROSS SITTINGS WITH NO CODE CHANGE (08-20 4.7-6.4, 08-28 8.5-9.0, today 7.25-7.28) — and the tag-`m5` BINARY run today gives 7.077, so it is the BOX, not a regression. The L arm proves it: 08-28→today stoke −18.2%, pp512 −18.0%, normalised −0.3%. ⇒ THE REPRODUCIBLE NUMBER IS THE RATIO, NOT THE RATE — gate item [5] publishes stoke/pp512 with both raw numbers, never a bare tok/s;** **(4) the Linux C++ TU box** (`apt install g++`, Kabir's); ~~(5) `stoke run` has no sampling flags~~ — **B4 DONE (ADDENDUM 36): five flags, same ranges and refusals as the server. ⚠⚠ And the header's claim that the CLI reproduces the SERVER for a given seed was FALSE when tested — the CLI keyed each draw on the position the next forward will occupy, the engine on the position whose logits produced the id (`n_prompt-1` for the first token), so every draw was shifted by one. Invisible from either side alone: both sampled, both reproduced their own seed, both passed every relation. A claim written in a header is a test that has not been run yet. Now byte-identical, and the comparison is kept as a case in `tests/run_sampling.sh`.** ▶▶▶▶ **PERFORMANCE (Kabir's standing priority: the gap outranks the remaining rungs). 2026-08-31, commits `5ded2fc`/`fe87b1f`/`5f2788f`, ADDENDUM 39 — THE BOOKKEEPING WAS 32 OF THE 55 INSTRUCTIONS PER (SUPER-BLOCK, SEQUENCE), AND 1.12× OF IT IS TAKEN, BIT-EXACT.** ⚠ The recorded next probe ("strip the per-column scale") was retired by ten minutes of `objdump`: gcc already hoists the eight `vpbroadcastw` per super-block and folds the reload into `vpmaddwd`'s memory operand, so it would have measured a cost nobody pays and read ~1.00 as "the dot is the wall". The real split is 23 dot / 32 bookkeeping — two 7-instruction horizontal sums, a 10-instruction scalar float epilogue, the mins reload, the pointer arithmetic. `-DSTOKE_MUT_NO_BOOKKEEP` priced the ceiling (pinned, single thread, ffn_gate): **N=2 1.85×, N=4 1.70×, N=8 1.60×, N=16 1.32×** — the instruction-count prediction holds where memory is not in the way and decays as the tile grows. ⚠⚠ **AND THE ABLATION REQUALIFIED THE 08-30 REFUTATION: the instrument's aliasing arm reads A/B 1.02 on the clean kernel and 1.30 on the stripped one, so the activation tile is worth 2% before this change and ~30% after. An ablation's verdict is scoped to the VERSION it ran on — removing the dominant cost promotes the next one; the M×N/tiling work is back on the table.** ▶ **THE FIX: four sequences reduced at once** (`kq_hsum4_i32`, 11 instructions for four totals where four calls are 28, exact by CONTRACT v2 because the totals are integers; plus a four-wide float fold that is not reassociated at all — same operations, same order, `-ffp-contract=off` pinned). ⚠⚠ **THE LOOP NEST IS THE ENGINEERING AND THE FIRST TWO SPELLINGS WERE WORSE THAN WHAT THEY REPLACED: 55.0 instr/seq before → 57.0 with four accumulators live across `wnib[8]` (spilled, 66 `vmovdqu`, measured 1.06× when 1.32× was available) → 63.0 rolled → 53.8 rolled by 2 → 43.5 SHIPPED with the sub-block loop OUTSIDE the sequence loop.** Q6_K took the same treatment, 59 → 43.0. **MEASURED, three arms interleaved on a box at 1.9-7.3%: matmul total 7940.7 → 6846.7 = 1.160×, prefill 10273.6 → 9173.9 = 1.120×, within-arm spreads 0.27-1.3%.** ⚠ `ffn_down` is a MIXED row (Q4_K_M leaves some ffn_down layers Q4_K), so only the change between the last two arms isolates Q6_K. **RATIO FROM ONE SITTING, per gate item [5]'s rule: llama-bench pp512 97.15 tok/s; stoke's 525-token prefill 51.1 → 57.2 tok/s ⇒ stoke/pp512 on the prefill axis 0.526 → 0.589.** Green throughout: 19 test binaries, four goldens, `serve_golden` 25/25 byte-identical, `test_matvec_batch` 180 checks bit-identical to the untouched solo kernel, id stream `cmp`-equal to the pre-change binary. ⚠ **INSTRUMENT: `bench --matvec-batch` never calls `pool_start`, so it runs inline and does NOT take the EcoQoS opt-out — unpinned it read 19.81 and 33.37 GF/s on two binaries whose N=1 path is byte-identical. Pin it, and use a same-binary repeat as the control, not the N=1 cell.** ▶▶▶ **AND THE TILE WAS THEN TAKEN, IN TWO STEPS, SAME SITTING. Commits `093eaeb` (the loop swap) and `c6089c7` (the row tile).** `rows_kquant_batch_avx2` had the ROW loop outside the TILE loop, so one output row touched all 374 KiB of the chunk's activations before the next row started — while `rows_delegate_batch`, six lines above, had had the right nest and the right comment all along and it never carried over. Swapping is **1.043× matmul, bit-exact, three lines** (only the order of INDEPENDENT calls changes). That leaves 46.7 KiB — still L1-sized — so the three xn kernels then took `(wrows, row_bytes, n_rows)` with the ROW loop INSIDE the super-block loop: the live activation becomes one super-block's 4.7 KiB, and the weight prep stays once per (row, super-block), which is exactly what shrinking `KERNELS_BATCH_TILE` cannot do (**=8 measured 1.7% WORSE, =4 8% worse — the 08-30 tile-8 result reproduced with a much smaller penalty**). **`KERNELS_ROW_TILE=32` was SWEPT, not chosen: 8→6014 ms, 16→5649, 32→5434, 64→5545.** ⚠⚠ **THE GRADER COULD NOT HAVE SEEN THIS CHANGE AND WAS WIDENED FIRST: `test_matvec_batch` ran at N_IN=256 = ONE super-block, where `row_bytes == block_bytes`, so a kernel that SWAPPED the super-block and row strides computes the right answer. Now N_IN=1024 (four) and N_OUT=100 (a short final row tile), and it was MUTATION-TESTED — swapping the strides gives 42 failures of 180, reverting gives 0.** ⏹⏹ **SESSION TOTAL, one block, 3 reps per arm, plus a same-block referent: matmul 7916.0 → 5655.3 = 1.400×, prefill 10242.3 → 7982.2 = 1.283×; llama-bench pp512 93.43 t/s (it read 97.15 earlier the same day — the box moved 3.8%, which is the whole reason for same-block). ⇒ stoke/pp512 on the prefill axis 0.549 → 0.704.** ⏹ **THE TILE IS NOW EXHAUSTED — the aliasing probe re-priced against the shipped kernel reads 1.008× (5.8% on 08-30, 29% mid-sitting, 0.8% now).** ⏹ **WHERE THE TIME IS NOW (525-token prefill, 7869.9 ms): matmuls 5526.0 (70%), attention 1309.1 (17%), and 1034.8 ms (13%) UNACCOUNTED — rmsnorm, rope, kv append, swiglu and the residuals, none of which is instrumented (`PROF_QKNORM`/`PROF_SWIGLU`/`PROF_NORM` exist and are never used, and all three read 0.0 ms). NEXT: instrument those three, then attention.** ▶▶▶▶ **AND THAT WAS DONE, SAME SITTING — commits `155fe5a` (the instrument) and `b56b4ee` (the fix), ADDENDUM 41.** ⚠⚠ **A ZERO IN A PROFILE IS TWO DIFFERENT CLAIMS — "this costs nothing" and "nothing measured this" — and the output cannot tell them apart. Three buckets had comments naming what they should measure and NO `PROF_ADD` anywhere; a fourth (`PROF_FFN_DOWN`) documented a residual its region excluded while its twin `PROF_ATTOUT` included one. AND THE PERCENTAGES CONCEALED IT BY CONSTRUCTION: the dump summed its own buckets and called that the total, so the shares always added to 100% however much sat outside. It now times the span the buckets live inside and prints the residue — UNACCOUNTED is 0.1 ms of 7936.7.** ▶▶ **The 13% was: swiglu 473.4, qk-norm+rope+kv 411.1, rmsnorm 71.8, the stray residual ~61 — ALL ON THE CALLING THREAD while seven workers waited, the shape L4 found in attention.** ▶▶▶ **THE SPLIT DECIDED THE DESIGN AND WAS MEASURED FIRST: phase 1 was one bucket; splitting it gives qk-norm+rope 390.3 and kv append 26.4. Only the append touches shared state (`kv_write` → `bp_seq_ensure` is a pure read only ONCE THE POSITION IS BACKED, which the append itself does), so 94% of phase 1 is the safe half and THE BLOCK POOL NEVER HAD TO BECOME THREAD-SAFE — which is exactly what the obvious move would have been.** ▶ **Pooled (parallelism over INDEPENDENT OUTPUTS, so L4's exemption: no twin, no re-anchor): swiglu 473.4→78.9 (6.0×), qk-norm+rope 390.3→67.4 (5.8×), rmsnorm 71.3→44.3 (1.63×). ⚠ The residual adds got their own bucket and were LEFT SERIAL at 31.4 ms — 0.2 ms per layer-call against a dispatch that costs about that; rmsnorm was worth it and only just, which is why both numbers are in the profile rather than assumed.** ⏹ **Decode untouched, structurally (`pool_for_units` falls back inline below POOL_MIN_CHUNKS at n_seq==1) and checked: 17.169/17.158 serial vs 17.177 pooled.** ⏹⏹ **THE SITTING, counterbalanced, same-block referent: prefill 10913.7 → 7538.6 ms = 1.448×; 48.11 → 69.64 tok/s against llama pp512 94.48 ⇒ stoke/pp512 on the prefill axis 0.509 → 0.737.** ⏹ **NEXT, read straight off a now-complete profile (7113 ms): ffn gate+up 2427.6 (34%), ffn_down 1482.5 (21%), attention 1296.3 (18%), qkv 1022.3 (14%), attn_out 640.1 (9%), everything else 190. Matmuls are 78%; ATTENTION is the only non-matmul item left.** ⏹⏹ **ATTENTION IS NOW CHARACTERISED (ADDENDUM 41 §7): 1296 ms = 18% of prefill against 4.9% of its FLOPs. ⚠ `att_kvheads_run` lives in `model.c`, which the Makefile deliberately compiles WITHOUT `-mavx2` (SIMD_CFLAGS is scoped to `simd.c` so gcc cannot emit AVX2 into the scalar reference twins). `objdump`: the SCORE pass is scalar `mulss`/`addss` because it is a REDUCTION over d and no flag can vectorize a float reduction; the VALUE pass got 4-wide `mulps` because it is an AXPY. Same MAC count, so the score pass is ~4× the value pass — **~1040 ms of the 1296.** ⏹ **PROBED: `-mavx2 -mfma` is worth only 1.06×** (1289/1298 → 1201/1223 interleaved) — gcc keeps the value pass 4-wide even given ymm, so the compile flag is not the fix. ▶▶▶ **THE PRIZE, BIT-EXACT: TRANSPOSE THE SCORE PASS INTO AN AXPY.** `score[g][j] = Σ_d q[g][d]·K[j][d]` written d-outer — `score[g][0..n-1] += q[g][d] * Kt[d][0..n-1]` — keeps every (g,j) summing over d in ASCENDING d, so it is bit-exact by the same construction the batched K-quant kernels use, and it vectorizes 8 wide because j becomes the lane. It needs K transposed to [d][j], which is a SECOND win: today each of the 128 sequences in a chunk re-reads the same K, so the buffer is built once per (layer, kv-head) per chunk and read by all 128 (269 KiB per kv-head, 2.15 MiB per layer). ▶▶▶▶ **AND ATTENTION WAS PROBED THEN FIXED, SAME SITTING — commits `52a86ac` (three probes) + `1e2ce6e`, ADDENDUM 42.** ⚠⚠ **THE PROBES KILLED BOTH CANDIDATE PLANS BEFORE EITHER WAS BUILT.** `-DSTOKE_MUT_NO_SCORE` / `-DSTOKE_MUT_NO_VALUE` split it (score ≈592 ms, value ≈540, softmax ≈146) and **PLAN 1 died on the spot: the score pass is a scalar float REDUCTION and the value pass a 4-wide AXPY, same MAC count, four times the lanes — AND THEY COST THE SAME**, so it is not short of arithmetic throughput. `-DSTOKE_MUT_ALIAS_KV` (whole K/V working set L1-resident) is worth **1.15×**, killing PLAN 2 — the 128 sequences of a chunk DO re-read the same K/V and it costs 15%, not the bulk. And a plain `-mavx2` rebuild was 1.06×. What survived: the score pass is at the **scalar FP-PORT CEILING** (40.7 GFLOP in 592 ms = 68.7 GF/s), i.e. short of LANES — and a float reduction cannot be given lanes over its own summation index. ▶▶▶ **SO GIVE IT LANES OVER THE OTHER INDEX: the four q-heads of the GQA group become the four LANES, each accumulating over ascending d — lane g is character-for-character the scalar chain it replaces, so BIT-EXACT BY CONSTRUCTION.** ⚠ **Q is the cheap side to transpose and that is the whole trick** (G·head_dim = 2 KB per unit against K's n·head_dim = 269 KB, 130× less to build); ⚠ **SSE2 is baseline on x86-64** so no TU move and no build-flag change, and 128 bits is exactly the four lanes G asks for (scalar four-chain kept under `#if !STOKE_ATT_SSE`). **MEASURED: attention 1276.8 → 1027.3 = 1.243×.** ⚠⚠ **AND THE ABLATION REQUALIFIED ITSELF A THIRD TIME THIS SITTING: `-mavx2` reads 1.06× against the scalar score pass and 1.218× against the vectorized one, because `vbroadcastss` is one uop where `_mm_load1_ps` is two and the loop is now three-quarters broadcast overhead.** ▶▶▶▶ **AND THE AVX2 KERNELS LANDED, SAME SITTING — commit `b08f9af`.** `att_score4_avx2` + `att_accum4_avx2` moved into simd.c (AVX2-compiled, already carrying the non-x86 fallbacks), gated at the call site by `kernels_cpu_has_avx_vnni()` as simd.h requires. ⚠ **The SCORE kernel stays 128-BIT on purpose** — its four lanes ARE the GQA group's q-heads, so widening means either eight q-heads (there are four) or two d per lane group, and the latter sums even and odd d separately and combines at the end, i.e. a reassociation; only the ENCODING changes, VEX for the one-uop broadcast. The ACCUMULATE kernel does widen to 256 bits because its lanes are independent d's. ⚠ **`-ffp-contract=off` is load bearing**: simd.c is built with `-mfma` and without that global flag gcc would fuse a mul+add here into an FMA the SSE2 arm cannot spell. **MEASURED, six units counterbalanced: attention 1024.8 → 849.9 = 1.206×.** ⚠ **AND ONE THING DID NOT PAY: widening the VALUE pass from the compiler's 4 lanes to 8 bought nothing measurable** (whole-file `-mavx2`, which left it 4-wide, read 1.218×; these kernels, 8-wide, read 1.206×) — the gain in both is the BROADCAST, not the width, which is a THIRD piece of evidence that the value pass is not arithmetic-limited. ⏹ **THE GRADER IS AN A/B, NOT AN ASSERTION: `STOKE_ATT_SIMD=0` forces the SSE2 arm on a CPU that has the kernels** — without it that arm is dead code here and its agreement would be asserted rather than tested, the vacuous-pass trap simd.h's own header names. Id streams BIT-IDENTICAL between arms and both `cmp`-equal to the pre-change binary; non-vacuous because the switch moves the clock; `serve_golden` 25/25 with the switch ON and OFF. ⏹⏹ **SITTING TOTAL: prefill 10278.6 → 6791.8 ms = 1.513×; 51.08 → 77.30 tok/s vs llama pp512 95.76 same-block ⇒ stoke/pp512 on the prefill axis 0.533 → 0.807.** ▶▶▶▶ **AND IT WAS DECOMPOSED TO THE LAST 15%, SAME SITTING — commit `4f28650`.** Re-run against the SHIPPED kernel the split moved again (fourth requalification this sitting): score ≈236 (was 592), value ≈400 (was 540), remainder ≈214, and **traffic is now worth 1.21× where it was 1.15× because the arithmetic shrank underneath it.** Two new probes finish it: `-DSTOKE_MUT_NO_SOFTMAX` → softmax **59 ms**; `-DSTOKE_MUT_NO_ACC` (store instead of accumulate) → the o[] read-modify-write **73 ms**; `-DSTOKE_MUT_HOIST_KV` (the call out of the j loop) → per-j `kv_read` **117 ms**. ⚠ **THE REGISTER-BLOCKING REWRITE IS NOT THE PRIZE**: holding a d-block of the output in registers removes the o[] load+store (8 of the 17 uops per j) but forces the four weight broadcasts back INSIDE the j loop and makes vj re-read hd/blocksize times — 73 ms is the whole prize and the rewrite does not collect all of it. ⚠ **THE LARGER ITEM IS THE PER-j LOOKUP AND IT IS AMORTIZABLE 16 WAYS**: both passes call `kv_read` per position (a cross-TU call into `kv_vec_offset` — three bounds checks, `bp_seq_ensure`, `bp_seq_block`, `bp_block_slot`, a multiply chain) around a 128-element inner loop, but `BP_BLOCK_TOKENS` is 16 and positions inside a block share block and slot, consecutive ones being exactly `n_head_kv * vec_bytes` apart. Bit-exact by construction. ▶▶▶▶▶ **THE GATE NUMBER ITSELF, MEASURED RATHER THAN DERIVED (2026-08-31, end of sitting).** Everything above is on the PREFILL axis against llama pp512; Kabir's "0.51×" names the AGGREGATE serving rate against the **≥17.7 tok/s @8 bar**, which is `bench --serve` and had not been re-run. Run it: the tag-`m6` BINARY (the state the 0.51× was said of) against HEAD, same block, counterbalanced M6/NOW/NOW/M6, the exact bars command (`--seqs 8 --ctx 704 -n 128 --no-dedup --threads 8 --kv f32`, eight s2-512 prompts): **AGGREGATE(M0) 8.788 / 8.709 → 13.923 / 14.000; medians 8.749 → 13.962 = 1.596×. Against the 17.7 bar that is 0.494× → 0.789×.** Within-arm spreads 0.9% and 0.55% against a 60% effect. ⭐⭐ **AND THEN CLOSED FURTHER TO 0.845× — commits `fda089a`/`4cd6e12` (the reassociation barriers) + `6a577c8` (the decomposition), ADDENDUM 43. AGGREGATE(M0) 8.770 → 14.954 = 1.705×.** The last two wins: an empty `__asm__("" : "+x"(v))` claims each accumulator and stops gcc turning the four integer chains into trees — Q4_K spills 34→10 (matmul 1.070×), Q6_K 40→21 (ffn_down 1.046×), both bit-exact because integer addition is associative so the tree and the serial chain are the SAME SUM. ⚠ `#pragma GCC unroll` was tried first and made it WORSE (63.0 instr/seq rolled vs 47.5 unrolled); the barrier keeps the unrolled schedule AND the serial chains. ⭐ **THE REFERENT ON THE SAME WORKLOAD, TODAY (`llama-batched-bench -npp 512 -ntg 128 -npl 8 -t 8`, a one-shot CLI): llama prompt-processing 44.063 s (92.96 t/s) vs stoke 52.66 s (79.8 t/s) = 0.858×; token-generation 21.648 s (47.30 t/s) vs 22.03 s (46.1 t/s) = 0.975×; total 65.711 vs 74.69 s. DECODE IS AT PARITY — THE ENTIRE REMAINING GAP IS PREFILL, and 0.858× against a bar ratio of 0.845× is the same number twice, so THE 17.7 BAR IS APPROXIMATELY "llama on this box" AND IS REACHABLE RATHER THAN STALE.** ⏹ **AND THE KERNEL'S REMAINING BIT-EXACT HEADROOM IS BOUNDED: `-DSTOKE_MUT_G4_DOTONLY` (dot only, everything else in the group path deleted) reads 1.120× on the matmuls — ~1.10× on prefill against the 1.17× prefill needs, and most of that 12% is work correctness requires. The remaining gap is NOT in this kernel's bookkeeping.** ⏹ **THREE CANDIDATES CLOSED BY MEASUREMENT so nobody re-opens them: (1) PREFIX DEDUP is worth NOTHING here — run with it ON, S2 adopts 0 of 4200 admitted prompt positions because the eight prompts diverge inside the first 16-token block, so `--no-dedup` costs nothing; (2) `KERNELS_ROW_TILE=32` is right for DECODE too (32→56.56 t/s, 16→56.11, 8→54.64 on a decode-dominated run), retiring the worry that 32 interleaved weight streams defeat the prefetcher at n_seq=8; (3) `STOKE_PREFILL_CHUNK` is worth 1.1% at 256 and WORSE at 512.** ⚠ The old binary reads **0.494× on today's box** where the note recorded 0.51× — which is why the OLD BINARY was run rather than the old number reused. ⚠ And the aggregate gained MORE than prefill did (1.596× vs 1.513×) because at `--seqs 8` the pooled elementwise dispatches clear POOL_MIN_CHUNKS, so batched DECODE takes them too — the n_seq==1 inline fallback only spares single-stream decode. **STILL SHORT OF THE BAR: 13.96 against 17.7 needs another 1.27×.** ⏹⏹ **WHERE THIS STOPS AND WHY: attention is 845 ms of a 6790 ms prefill (12.5%), and the three remaining items are 1.7%, 1.1% and 0.9% OF PREFILL each — the first needing a `kv.h` API that must keep working for the Q8 store, which dequantizes into scratch and has no contiguous-stride view to hand back. A design, not a tweak, for 1.7%. The numbers are banked so a later sitting can decide.** — superseded: **NEXT: the VALUE pass, which is most of attention's remaining ~850 ms. Three findings say what it is NOT** — not arithmetic-limited, not width-limited, not mostly K/V bandwidth. **What is left and unprobed: the `o[] += wgt*vj[d]` READ-MODIFY-WRITE (the whole 512-byte output row re-read and re-written for every j, 2.1 MB of L1 traffic per unit) and the PER-j `kv_read` call overhead around a 128-element inner loop. The probe that separates them hoists `kv_read` out of the j loop while leaving the arithmetic and the o[] traffic alone; the fix for the RMW is holding a d-block of the output in registers across j.** — superseded plan below: **NEXT: a runtime-gated AVX2 kernel in simd.c beside the others** — it cannot just be switched on because `vec.c` gates its AVX2 kernels behind `kernels_cpu_has_avx_vnni()` and this engine still runs without it. One call per j for the score row, one per j for all four value AXPYs so call cost stays under a tenth of the work; **both passes want it, the value pass being broadcast-bound in SSE2 for the same reason.** ⏹⏹ **THE SITTING END TO END: prefill 10313.9 → 6781.8 ms = 1.521×; 50.90 → 77.41 tok/s against llama pp512 96.73 same-block ⇒ stoke/pp512 on the prefill axis 0.526 → 0.800.** Within-arm spreads 1.2% against a 52% effect. Final profile (6785 ms): ffn gate+up 2391.5 (35%), ffn_down 1474.2 (22%), attention 1034.4 (15%), qkv 1019.9 (15%), attn_out 616.0 (9%), rest 249; matmuls 81%, UNACCOUNTED 0.1 ms. ▶▶▶▶▶ **THE DOT ITSELF, ON AVX-VNNI — 2026-09-01, ADDENDUM 44. ADDENDUM 43 left this direction SECOND, behind a recorded objection that was checked on paper and never measured: `vpdpbusd` retires 32 MACs in one uop where `vpmaddubsw`+`vpmaddwd` take two, but it accumulates four byte-products per i32 lane and has nowhere to put Q4_K's per-sub-block scale, so applying the scale would need a `vpmulld` costing more than it saves. THE ARGUMENT IS TRUE OF THE NATURAL LAYOUT AND FALSE OF A PERMUTED ONE.** `objdump` first, as the standing rule requires: 157 instructions per (super-block, row, FOUR sequences), of which 68 are on ports 0-1 ⇒ 34 cycles against a front end wanting 26, so it is PORT-BOUND ON THE MULTIPLIERS. ▶ **THREE wrong-output arms, each carrying the exact instruction mix of a design: `_CEIL` (8 dpbusd, no scale) 1.270×, `_B` (8 dpbusd + one pack + one maddwd, the permuted-layout mix) **1.272×**, `_A` (sub-blocks in PAIRS, the design needing NO layout change) **only 1.090×**.** ⭐ **`_B` EQUALS `_CEIL`, so applying all eight scales is FREE once the lanes span sub-blocks** — and ⚠⚠ **`_A` IS THE ARM THAT MATTERED: without it the cheap variant was the obvious thing to build, and it would have delivered a third of the available gain and looked like a success.** ▶▶ **THE KERNEL: `kq_perm4` puts four DIFFERENT sub-blocks in each accumulator's lanes in eight `vpunpck`s with NO cross-lane permute, applied identically to the nibbles and the activations; the eight scales then arrive in ONE `vpmaddwd` per super-block. 9 p01 uops where the maddubs twin spends 16.** ⚠ **The lane-to-sub-block map is OURS TO CHOOSE, which is the whole saving** — a full 8x8 dword transpose needs eight `vpermd` more and buys nothing, because `scales8` is reordered to match for two instructions. ⚠ **Bit-exact by CONTRACT v2** (a permutation, then integer sums, and no float anywhere), with the ladder computed at every step: a `vpdpbusd` lane holds 16 elements ⇒ ≤30,720 against the i16 ceiling `vpackssdw` saturates at, so TWO accumulators per sequence is the BOUND, not a convenience. ⏹ **GRADED, then MUTATION-TESTED: `-DSTOKE_MUT_Q4K_VNNI` swaps two sub-blocks on the WEIGHT SIDE ONLY — 27 of 180 fail on the vnni arm, 0 on maddubs. It has to be ONE-SIDED: permuting both sides is a relabelling and stays exact, which is the mutant that would pass and prove nothing.** Green: 19 binaries, four goldens, `serve_golden` 25/25 byte-identical, id streams `cmp`-equal at n_seq=1 and 4-way batch. ⏹⏹ **MEASURED, ONE BINARY, TWO ARMS BY ENV, 20 UNITS ABBA — and Kabir flagged the box NOISY for the first half, so the protocol carries its own proof: prefill 6359.7 → 5443.6 ms = 1.168×, Q4_K matmuls 1.258×, and THE ARMS' RANGES DO NOT OVERLAP while the untouched `attention` bucket reads 873.7 vs 874.1 = 1.000×. A disjoint-range result and a null arm at 1.000× are what make a noisy box readable — not the medians.** ⭐ **The 1.258× is the WHOLE of the 1.272× the probe promised: both permutations and the extra 4 KiB of activation tile cost nothing measurable.** ⏹ Decode untouched, structurally (n_seq==1 never reaches an xn kernel) and checked. ⭐⭐ **THE GATE NUMBER, on a box Kabir then QUIESCED: `bench --serve` AGGREGATE(M0) 14.855 → 16.934 = 1.140×, i.e. **0.839× → 0.957× of the ≥17.7 @8 bar**, 12 units in three ABBA blocks, ranges DISJOINT, TTFT p50 32.9 → 27.6 s (1.19×, matching prefill, which is the internal consistency check).** ⭐⭐ **AND THE SAME-SITTING REFERENT SAYS STOKE'S PREFILL HAS PASSED llama's: `llama-batched-bench -npp 512 -ntg 128 -npl 8 -t 8` run BEFORE and AFTER the stoke arms to bracket them (44.897 s and 44.872 s, 0.06% apart, which is what says the box was quiet) — prompt processing llama 91.26 t/s vs stoke 84.28 (0.924×) → **99.41 (1.089×)**; token generation 46.92 vs 43.48 → 46.31 (0.987×, parity).** ⚠ **THE BASELINE IS TODAY'S maddubs ARM, NOT YESTERDAY'S 0.858×** — the same code read 0.858× on 08-31 and 0.924× today; that is the box, which is exactly what the same-sitting rule is for. ⚠⚠ **AND THE GUARD ADDED TO PROTECT THIS KERNEL SILENTLY DISABLED IT.** `STOKE_MINS_SIMD` was `#define`d in simd.c, so in vec.c the name is UNDEFINED and `#if !STOKE_MINS_SIMD` is `#if !0` — ALWAYS TRUE — so the guard fired in every build and forced the kernel off for twenty minutes while every functional test still read 180/180, because the twin computes the same answer. **What caught it was the ANTI-VACUITY REFUSAL, not a test**: the guard sat in front of the `strcmp` chain, so `STOKE_Q4K_KERNEL=bogus` stopped being refused. ⇒ the knob moved to simd.h (a per-TU default for a build-wide knob is two different knobs), the parse now happens FIRST and unconditionally, an asked-for `vnni` under `-DSTOKE_MINS_SIMD=0` is REFUSED rather than downgraded, and every maddubs-twin-only speed mutant now PINS the arm it lives in. ⚠ Related, found the same way: `-DSTOKE_MINS_SIMD=0` had not COMPILED since `fda089a` (`KQ_KEEP`'s `#define` went inside a `#if STOKE_MINS_SIMD` block while the Q6_K kernel uses it unconditionally); hoisted to file scope, CODE-NEUTRAL (simd `.text` byte-identical). **A knob that no longer compiles is not a knob, it is a claim.** ⏹⏹ **NEXT: (1) Q6_K wants the same treatment** — `ffn_down` is 22% of prefill and moved only 1.101× because Q4_K_M leaves some of its rows Q6_K; the centring distributes the way `block_acc` already does for Q8_0, but its sub-blocks are 16 elements not 32, so the permutation and the overflow ladder must be RECOMPUTED, not inherited. **(2) THE TILES WERE SWEPT AGAINST THE OLD KERNEL** — `KERNELS_ROW_TILE=32` / `KERNELS_BATCH_TILE=16` were chosen when the dot cost 16 p01 uops; it now costs 9 and carries a 4 KiB permuted-activation buffer, so re-run the sweep rather than inherit it (the fifth time this milestone that removing a dominant cost repriced an earlier measurement). **(3) Hoisting the activation permute to `matvec_batch`** would amortise it over all n_out rows instead of the tile's 32 — priced at ~3% and deliberately not taken. ⭐⭐ **AND THE ANSWER TO "BETTER FROM 0.51×?", MEASURED AGAINST THE BINARY RATHER THAN CHAINED: the tag-`m6` binary against HEAD, ONE block, four units per arm, ABBA, box quiesced — AGGREGATE(M0) **8.760 → 16.907 = 1.930×**, i.e. **0.495× → 0.955× of the ≥17.7 @8 bar**, ranges disjoint, TTFT p50 61.4 → 27.9 s (2.20×). Short by 1.047×.** ⚠ **"0.51×" WAS ALWAYS SLIGHTLY GENEROUS**: the binary reads 0.495× today and 0.496× on 08-31, and those agree to 0.1% — which is what licenses comparing the sittings at all. ⚠⚠ **A BLOCK TAKEN BEFORE THE BOX WAS QUIESCED READ 0.898× FOR THE SAME BINARY, WHILE THE RATIO HELD AT 1.904× with 0.1% spreads on both arms.** A same-block llama referent read 89.08 t/s against 91.26 earlier — the box was down 2.4% by the referent and 6% by `bench --serve`, so the SERVING AGGREGATE IS MORE SENSITIVE TO BOX STATE THAN pp512 IS. **The ratio travels; the bar fraction does not — publish both and name the block.** ⚠ And one arm behaves differently: across every block today the maddubs arm spread 5.8-8.2% while m6 and vnni spread 0.1-2.0%, same binary, different kernel. Unexplained; do not use maddubs as a reference arm without repeats. ⭐⭐ **AND ON llama's OWN DENOMINATOR, ONE BLOCK, ALL THREE: prefill m6 43.79 t/s = **0.491× of llama** → HEAD 100.82 = **1.131×**; decode m6 32.32 = 0.680× → HEAD 47.19 = 0.992× (llama 89.14 / 47.56).** So the "0.51×" was against the BAR, but at the m6 end the two denominators agree — m6 really was ~0.49× of llama on prefill — and **stoke's prefill is now 13% FASTER than llama's.** (An earlier block read 1.089×; the box moved 2.3% by llama's own measure, so the ratio sits at 1.09-1.13×.) ⚠⚠ **AND THAT BREAKS AN IDENTITY ADDENDUM 43 RELIED ON. Prefill is 1.13× of llama and decode is parity, yet `bench --serve` AGGREGATE(M0) is only 0.955× OF THE BAR** — where 43 could write "0.858 against 0.845 is the same number twice, so the 17.7 bar IS ≈llama-on-this-box". The kernel work moved one side of that identity and not the other. ⇒ **THE REMAINING 1.047× IS NO LONGER IN THE KERNELS**: both component rates now match or beat the referent while the SERVING aggregate does not, so it is in what `bench --serve` measures and `batch` does not — continuous-batching scheduling, the prefill-chunk interleave against running decodes, TTFT distribution, or the bar's own provenance (17.7 came from M0 against llama-SERVER, a different harness). ⚠ **THIS RE-ORDERS THE NEXT RUNGS: Q6_K was ranked first because ffn_down is 22% of prefill, but PREFILL IS ALREADY AHEAD OF THE REFERENT.** Before spending a sitting on it, decompose AGGREGATE(M0) against the same workload run through `batch`, and re-derive what 17.7 actually was. ⏹ **AND AT n=1 (`llama-bench -p 512 -n 128` vs `stoke run`, one block): prefill m6 44.85 → HEAD 97.56 t/s against llama 92.81 = **0.483× → 1.051×**; decode 17.13 → 17.62 against 18.33 = 0.934× → **0.961×** (UNTOUCHED — n_seq==1 routes to the solo kernel; only prefill takes the new one, because it chunks 128 positions through the batched path even for one sequence). The prefill margin is SMALLER at n=1 than at n=8 (1.051× vs 1.131×) because a bigger batch amortizes the weight stream further.** ⚠⚠ **AND THE FIRST DECODE READING WAS WRONG BY A CONTEXT CONFOUND, 0.834×, and would have been published: `llama-bench`'s tg128 starts from an EMPTY context while `stoke run -f <525-token prompt>` decodes at ctx 525..652, so stoke was reading ~5× the KV per token and being graded against a number that was not. Re-run near ctx 0: 15.285 → 17.617 t/s = 0.961×.** ⇒ **A DECODE RATE IS COMPARABLE ONLY AT THE SAME CONTEXT LENGTH** — same binary, same kernel, same box reads 17.6 t/s at ctx ~30 and 15.3 at ctx ~590, a 13% spread that is physics and attaches to whichever arm carries the longer prompt. Two rates are a ratio only if their denominators match; here the hidden denominator was the context. ⭐⭐⭐ **AND AT n=16 THE THESIS HOLDS: llama SATURATES AFTER B=4 AND STOKE DOES NOT.** `llama-batched-bench -npl 1,2,4,8,16` against `stoke batch`, one block — DECODE t/s llama 16.10 / 25.77 / 44.68 / 46.82 / 48.00 against stoke 15.30 / 20.36 / 33.52 / 44.22 / **57.69**. Per doubling llama goes 1.60×, 1.73×, **1.05×, 1.02×** — it front-loads its batching gain and hits a WALL at B=4 — while stoke goes 1.33×, 1.65×, 1.32×, **1.31×**, smooth and still climbing at 16. **B=4→16: llama 1.07× for 4× the batch, stoke 1.72×. At n=16 stoke's decode is 1.202× of llama's**; PREFILL is 1.03-1.14× of llama at every point on the curve. Bracketed: stoke decode 57.69/57.88 and llama 48.00/47.87, both 0.3% apart. ⚠ **stoke is BEHIND at B=2 and B=4 (0.790×, 0.750×) and it is NOT explained** — the obvious hypothesis was CHECKED and is wrong (`POOL_MIN_CHUNKS` is 2, so the pooled elementwise dispatches already fire at n_seq=2). What is true is that those dispatches parallelize over SEQUENCES, so at B=2/4 they have 2/4 units for 8 threads while the matmuls stay row-parallel across all 8 — a hypothesis wanting a probe, not a plan. ⭐⭐ **AND THE BAR CELLS RE-READ — THERE ARE **TWO** GATE BARS, NOT FOUR.** `run_m5_bars.ps1` is the authority: `$GATE = @{ 8 = 17.7; 16 = 16.8 }` — **the binding bar at each width is the HIGHER of the two families** (llamacpp-tuned's 17.7 at n=8; 2×-ollama's 16.8 at n=16, "where the inversion makes it the harder bar"), while 14.2/10.8 are the other family's values at those widths and every other N is a `ref` label, not a bar. **@8 f32: m6 8.760 (0.495×) → HEAD 16.907 = MISS at 0.955×. @16 f32: m6 8.473 (0.504×) → HEAD 17.181 (3 units, 1.9% spread) = **MET at 1.023×**, an improvement of 2.028×. ONE OF THE TWO GATE BARS IS NOW MET, where at M5 close BOTH missed at ≈0.50×.** ⭐ **AND THE PROVENANCE OF 17.7 CHANGES HOW THE MISS SHOULD BE READ.** M0's own file has llamacpp-tuned at 15.571 / 23.513 / **33.639** / **17.664** / 14.163 for n=1/2/4/8/16 — so 17.7 is its S2 @8 cell from **2026-07-17, 46 days old** — and its SHAPE is this project's whole argument: it peaks at n=4 and then COLLAPSES, while stoke goes the other way (16.907 at 8 → 17.181 at 16). The inversion the second bar family was built around is visible in M0's own numbers, and it is the same wall today's batched-bench curve found at B=4. ⚠⚠ **THE 4.5% MISS THEREFORE CANNOT BE ATTRIBUTED WITHOUT A SAME-SITTING REFERENT**: this box's between-sitting range is far wider than 4.5% (llama-bench pp512 alone: 75.66 on 08-30, 93.43-97.15 on 08-31, 92.81 today — a 28% span with no code change), so 0.955× against a fixed 46-day-old absolute is inside the noise. Reproducing it needs **llama-SERVER**, since the 17.7 cell was produced by llama-server through the M0 harness; `llama-batched-bench` is NOT a substitute, being lockstep, so its M0-equivalent span excludes the interleaved prefills a continuous server is charged for. **Asked of Kabir rather than done — a long-lived local model server is never started without asking.** ⚠⚠ **MEASURED, NOT GATE-STAMPED: gate item [2]'s protocol is `run_m5_bars.ps1` (medians over three blocks, pinned, its OWN quiesce instrument gating each unit, and the q8 cells). This is 4 units @8 and 3 @16 on a hand-quiesced box, f32 only; 8q8 and 16q8 are UNMEASURED. Anyone quoting a gate pass has to run the runner.** | Llama-3.2 parity incl. long-ctx; unsloth-file run; 24h soak; reproducible report; Kabir sign-offs |

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
