# stoke — Phase 1 close-out sequence (the execution queue)

*Written 2026-09-01 (Opus 5, max effort), at HEAD `ff1949b` on branch
`m5-entry`, the day the AVX-VNNI Q4_K dot landed (ADDENDUM 44) and the day
llama-server was re-measured on this box for the first time since M0.*

**This file is a SECOND TYPE of document, not a replacement for anything.**
`ROADMAP.md` is the MAP — what each milestone is, what its gate says, and the
full narrative of how each one closed. This file is the QUEUE — what is left,
in what order, blocked on what, done when. Neither compresses the other and
neither is a summary of the other; they answer different questions.

**Authority order, unchanged:** the linked ruling memos and gate reports win
over `ROADMAP.md`, and `ROADMAP.md` wins over this file. **This file never
re-specifies a gate criterion.** Where an item below restates a bar or a bound,
the restatement is a convenience and the linked document governs. If they ever
disagree, the linked document is right and this file has a bug.

**How to use it:** read `ROADMAP.md` first (it is still the first document a
working session reads), then this file for position and order, then
`.superpowers/sdd/progress.md`'s final NEXT block for the exact in-flight
resume point. This file is updated at the END of a stage, not during one.

---

# PART I — POSITION AS OF 2026-09-01

## I.1 The repository, mechanically

| fact | value | how it was checked |
|---|---|---|
| branch | `m5-entry` | `git status -sb` |
| HEAD | `ff1949b` — *"ROW_TILE 64: check the axis the gate measures, not just the ones swept"* | `git rev-parse` |
| commits on HEAD | **420** | `git rev-list --count HEAD` |
| first commit | `5866e2b`, **2026-07-16** — *"docs: pre-code project record"* | `git log --reverse` |
| elapsed | **47 days** | 07-16 → 09-01 |
| engine source | **22,511 lines** of `.c`/`.h` under `src/` | `wc -l` |
| test source | **13,773 lines** of `.c` at `tests/` top level | `wc -l` |
| shipped binary | `build/win/stoke.exe`, **1,384,255 bytes**, `LDFLAGS` empty | `ls -la`, `Makefile:53` |
| models present | Qwen3-4B-Instruct-2507 **Q4_K_M** and **Q8_0** only | `ls models/` |
| remote | private `Legendxyz-afk/Stoke` | §0 of `engine-project.md` |
| `origin/m5-entry` | `ff1949b` — **exactly HEAD**, plus all seven tags | `git branch -r -v` |
| `origin/main` | `71e84aa` — *"First gear: Initial massive commit"* (M1) | `git branch -r -v` |

### Tag → commit map (checked by dereferencing, not by memory)

| tag | tag object | commit | date | kind |
|---|---|---|---|---|
| `m1` | `cbf1e20` | `cbf1e20` | 2026-07-19 | lightweight |
| `m2` | `f730a64` | `f730a64` | 2026-08-02 | lightweight |
| `m3` | `d3c1c24` | **`b00272a`** | 2026-08-11 | annotated |
| `m4` | `47aadea` | **`ed2aa32`** | 2026-08-13 | annotated |
| `m45` | `969bb8a` | **`5f6ced8`** | 2026-08-19 | annotated |
| `m5` | `c3f7002` | **`bc969b9`** | 2026-08-28 | annotated |
| `m6` | `bea7171` | **`e97845f`** | 2026-08-30 | annotated |

⚠ **TWO CORRECTIONS TO THE RECORD, both found by running the check rather than
trusting the note:**

1. **`m6` is an ANNOTATED tag and the record quotes its TAG OBJECT.** Every
   note in this project says "tag `m6` = `bea7171`". `bea7171` is the tag
   object; the **commit is `e97845f`**. The same is true of `m4` and `m45`,
   which have never had their commits written down at all. Only `m3` and `m5`
   were recorded by commit (`b00272a`, `bc969b9`, both confirmed correct).
   ⇒ **Anything that resolves `m6` as a commit sha — an archived gate binary,
   a `git diff bea7171..HEAD`, a bank key — must use `m6^{commit}` or
   `e97845f`.** `git checkout bea7171` happens to work (git dereferences), so
   this has never bitten; a script doing string comparison on shas would.
2. **`main` is 177 commits behind HEAD, not "~230".** The M7 entry note's
   figure was approximate and has since drifted. Measured:
   `git rev-list --count main..HEAD` = **177**, and
   `git rev-list --count origin/main..main` = **214**. Local `main` sits at
   `5f6ced8`, which is exactly the `m45` commit — so the M7 entry note's
   "main is still at `m45`" is CORRECT, only its distance was wrong.

## I.2 Milestones

| # | milestone | closed | tag → commit | note |
|---|---|---|---|---|
| M0 | Measure the enemy | 2026-07-17 | `8ce4fd8` (no tag) | premise: best baseline 11% of ceiling — GO |
| M1 | `gguf.c` + `tok.c` | 2026-07-19 | `m1` = `cbf1e20` | byte-exact vs `llama-tokenize`, 22,848-line unicode sweep |
| M2 | Scalar forward pass | 2026-08-02 | `m2` = `f730a64` | 24/24 token-exact, 1,536 greedy decisions, 0 diverged |
| M3 | AVX2+VNNI kernels + pool | 2026-08-11 | `m3` = `b00272a` | zero BUG-class on the Q4_K_M grid; tripwire FLAGGED not quiet |
| M4 | Paged KV + static batch | 2026-08-13 | `m4` = `ed2aa32` | five ruling-§8.4 items; [4] class-rate MISSED and dispositioned |
| M4.5 | Kernel ladder → llama-band | 2026-08-19 | `m45` = `5f6ced8` | P-BAR MET at both configs (0.912× t=6 pinned, 0.892× t=8 unpinned) |
| M5 | Continuous batching | 2026-08-28 | `m5` = `bc969b9` | **closed with TWO documented misses: [2] at 0.51×, [3] at 2.7×** |
| M6 | `stoke.h` freeze + server | 2026-08-30 | `m6` = `e97845f` | **all seven gate items banked; `stoke.h` v0 FROZEN** |
| **M7** | **Harden + launch-ready** | **← NOW** | — | **rungs B1–B8; B1 B2 B3 done, B4 half done** |

**Seven of eight milestones closed in 47 days.** M7 is the last one in Phase 1.

## I.3 The Phase-1 goal, clause by clause

`ROADMAP.md` line 25: *"Phase 1 is done when (spec 'Phase-1 goal', measured on
the reference box): tokenizer + greedy parity proven against llama.cpp; S2
aggregate ≥2× ollama-default AND ≥1.0× llama-server-tuned at N=8/16 on the same
GGUF; single-stream ≥50% of llama.cpp; no >250ms decode stall from a co-running
prefill; one static binary; OpenAI endpoints pass openai-python; ASan/TSan green
throughout; README with a reproducible benchmark report."*

| # | clause | verdict | evidence |
|---|---|---|---|
| 1 | tokenizer + greedy parity vs llama.cpp | ✅ **MET** | M1 byte-exact tokenizer (fuzz corpus, specials both modes, 22,848-line unicode sweep); M2 24/24 token-exact over 1,536 greedy decisions, 0 diverged, plus the 36-layer L2 ladder at ≤1e-3 worst 6.104e-04; M3 zero BUG-class on the Q4_K_M grid (9/9 ARITHMETIC-CLASS, worst margin 54% of a τ frozen before grading) |
| 2 | S2 aggregate ≥2× ollama-default **AND** ≥1.0× llama-server-tuned @ N=8/16 | **2026-09-24: DISPOSITIONED.** llama-server half ✅ **PASSED**; ollama half ⛔ **a DOCUMENTED MISS, ACCEPTED BY KABIR 2026-09-24 — the referent is not re-specified, `memos/2026-09-24-clause2-ollama-DISPOSITION-KABIR.md`** (ollama-default serves ONE request at a time on a CPU box). *09-23 read: HALF MET, the ollama half awaiting Kabir's call* *09-01 read: ❌ OPEN, denominator in dispute — settled by the 09-08 re-spec (§2.1)* | graded as a same-sitting END-TO-END WALL ratio (re-spec 2026-09-08, `design-notes/2026-09-08-clause2-end-to-end-preregistration.md`): **1.181× @8** (5/5 pairs, 09-15), **1.195× @16** (3/3), **1.281× @32** beyond the clause (3/3); ollama **1.735× @8, 1.836× @16** — `gate-reports/2026-09-18-clause2-e2e-16-32-ollama.md`, `docs/BENCHMARKS.md`. *(09-01: 0.955× / 1.023× against the frozen M0 bars; 0.687× against llama-server that day)* |
| 3 | single-stream ≥50% of llama.cpp | ✅ **MET, and far past it** | n=1 prefill **1.051×** of `llama-bench` (was 0.483× at m6); n=1 decode **0.961×**; M4.5's P-BAR (≥0.85× decode) met at 0.912× / 0.892× |
| 4 | no >250 ms decode stall from a co-running prefill | ❌ **MISSED, STRUCTURAL — DISPOSITIONED 2026-09-16 (Kabir): a DOCUMENTED MISS, `memos/2026-09-16-clause4-itl-stall-DISPOSITION-KABIR.md`; the overlap redesign is Phase 1.5's first scheduler item** | stall p50 **738 ms at chunk 64** — the smallest S-ALIGN-legal width — 1287 ms at 128, 2450 ms at 256 (09-08 sweep); **09-15 same-sitting screen: 1.84 s at 128, 3.34 s at 256, and 256 is the SHIPPED width since the 09-15 evening (Kabir's ruling; wall −3.4%, TTFT −4/−6%)**. `cadence` p50 ~171 ms is comfortably inside 250 at every width, so this is the *stall* population only, exactly as the 08-26 ruling scoped it |
| 5 | one static binary | ✅ **MET** | single 1.38 MB `stoke.exe`, `LDFLAGS` empty, no runtime deps beyond platform sockets |
| 6 | OpenAI endpoints pass openai-python | ✅ **MET** | M6 gate; `tests/serve_openai.py` is the sole grader for four of the five sampling parameters, backed by a 5/5 mutation leg proving each parameter's DELIVERY can fail and does |
| 7 | ASan/TSan green throughout | ✅ **MET** | M6 matrix: Windows plain · Linux plain · Linux ASan · Linux TSan all GREEN. **B1 then closed the coverage hole that made the TSan result narrower than it read** — every test was single-threaded, so "zero races" covered none of the cross-thread surface; a sim-driven test found SIX and reports zero after the fix |
| 8 | README + reproducible benchmark report | 🟡 **FIRST PASS 2026-09-18** (`5e56039`) — the 24 h soak's row is the one line owed | README rewritten on the measured position; `docs/BENCHMARKS.md` reproducible from its printed commands, absolutes + ratios + the block named, both misses stated. *(09-01: not started)* |

**2026-09-23: five met (1 3 5 6 7); clause 2 half met — the llama-server
half passed, the ollama half a documented miss (Kabir accepted it 2026-09-24);
clause 4 a documented miss (Kabir, 09-16); clause 8 first pass, the soak row
owed.** *(09-01 read: "five met, one not started but cheap, two open —
neither an engineering unknown any more, both decisions (§ Stage 2)" — and
both were decided: 2.1 on 09-08, 2.2 on 09-16.)*

## I.4 The performance position, with all four denominators

This is the single most misread number in the project, because **the answer
depends entirely on what you divide by, and three of the four denominators are
defensible.** Publish more than one or the number is not interpretable.

### The measurement (2026-09-01, box quiesced by Kabir, one block, ABBA)

| arm | AGGREGATE(M0) @8 f32 | TTFT p50 |
|---|---|---|
| tag-`m6` binary | 8.760 | 61.4 s |
| **HEAD** | **16.907** | **27.9 s** |
| ratio | **1.930×** | 2.20× |

Four units per arm, ranges DISJOINT. A within-binary A/B the same sitting (one
binary, both arms selected by `STOKE_Q4K_KERNEL`, 12 units in three ABBA blocks)
reads 14.855 → 16.934 = 1.140×, TTFT p50 32.9 → 27.6 s.

### The four denominators

| # | denominator | reading | what it is worth |
|---|---|---|---|
| 1 | the frozen M0 bar, **17.7 @8** | **0.955×** — MISS by 1.047× | the gate as written. ⚠ **46 days old and provably stale** — see below |
| 1b | the frozen M0 bar, **16.8 @16** | **1.023×** — nominal MET, up **2.028×** from m6's 8.473 | ⚠ **MUST NOT BE QUOTED AS A PASS.** 16.8 was never re-measured; only @8's referent was. An equally stale target |
| 2 | **llama-server measured TODAY: 24.608** | **0.687×** | the honest live number. m6 was 0.356× |
| 3 | llama's component rates, same sitting | prefill **1.131×** · decode **0.992×** @8, **1.202×** @16 | **stoke's kernels are AHEAD.** The gap is not arithmetic |
| 4 | **total wall on the identical job** | **0.945×** (65.39 s vs 61.76 s) — and **TTFT p50 1.50× BETTER** (27.6 s vs 41.3 s) | the work actually done, and the latency a user actually feels |

### Why the 17.7 bar is stale, measured not asserted

`run_matrix.py --only-config llamacpp-tuned --only-scenario s2-closed-8 --runs 3
--fresh` — **M0's own harness, M0's own tuned config** (`--parallel 8 -c 16384`),
server tree-killed after — reads **24.608 tok/s today** (24.722 / 24.608 /
24.488, 0.95% spread, 8/8 completed, 0 errors) where M0 recorded **17.664 on
2026-07-17**. Same machine, same RAM, both under the High-performance plan.
**39% of that is box state.** The bar is 0.72× of what llama does here now.

Corroborating: this box's between-sitting range on `llama-bench pp512` alone is
75.66 (08-30) → 93.43–97.15 (08-31) → 92.81 (09-01) = **a 28% span with no code
change.** A fixed 46-day-old absolute cannot survive that.

### Where the remaining 31% actually is — measured, not inferred

**It is not the kernels, and it is barely work at all. It is where the clock
starts.**

AGGREGATE(M0) spans *earliest-first-token* → *latest-last-token*. On the
identical job:

- stoke's earliest first token lands at **5.3 s**; llama's at **20.7 s**.
- ⇒ stoke is charged **60.1 s of span for 65.4 s of work**; llama is charged
  **41.1 s for 61.8 s**.
- llama interleaves its eight prefills so no client finishes one early — its
  last tokens then land within 0.17 s of each other. stoke runs **≤1 active
  prefill** and is already emitting at 5.3 s.

⇒ **stoke is penalized by this metric precisely for being responsive early, and
the responsiveness is measured (TTFT p50 1.50× better), not asserted.**

The residual real gap is **5.5%**, and it is in the serving layer, not the
kernels: same job, four measurements — stoke LOCKSTEP (`batch`) 63.21 s **beats**
llama LOCKSTEP (`batched-bench`) 65.32 s = 1.033×; going CONTINUOUS, llama
**gains 5.5%** (server 61.76 s) while stoke **loses 3.4%** (serve wall 65.39 s).
That 9-point swing is the 0.945×.

The co-run was then probed and **is already at its knee**: `STOKE_SCHED_CHUNK`
64 → AGG 16.241 / wall 68,128 ms / stall p50 738 ms; **128 (shipping) → 17.019 /
65,056 / 1,287**; 256 → 17.043 / 65,018 / 2,450. Halving costs 4.8% end-to-end;
doubling buys 0.06% while doubling stall. `cadence` p50 is FLAT at ~171 ms
across all three while `stall` scales LINEARLY ⇒ **co-running does not make the
decode slower, it makes the step it rides in longer by exactly the prefill added
to it. The decode rows wait; they are not harmed.**

### The thesis, validated — this is the actual achievement

`llama-batched-bench` vs `stoke batch` across `-npl 1,2,4,8,16`, one block,
bracketed (stoke 57.69/57.88 and llama 48.00/47.87, both 0.3% apart):

| N | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| llama decode t/s | 16.10 | 25.77 | 44.68 | 46.82 | **48.00** |
| stoke decode t/s | 15.30 | 20.36 | 33.52 | 44.22 | **57.69** |
| llama per doubling | — | 1.60× | 1.73× | **1.05×** | **1.02×** |
| stoke per doubling | — | 1.33× | 1.65× | 1.32× | **1.31×** |

**llama hits a WALL at B=4 — 1.07× for 4× the batch from 4→16 — while stoke
delivers 1.72× over the same range and is still climbing at 16. At n=16 stoke's
decode is 1.202× of llama's.** Prefill is 1.03–1.14× of llama at *every* point
on the curve.

And the same shape is visible in **M0's own 46-day-old baseline**:
llamacpp-tuned reads 15.571 / 23.513 / **33.639** / **17.664** / 14.163 at
n=1/2/4/8/16 — **it peaks at n=4 and then collapses**, while stoke goes the
other way (16.907 @8 → 17.181 @16). *The inversion the second bar family was
built around is in the enemy's own numbers.*

⇒ **"concurrent-first for the consumer hardware vLLM abandons and llama.cpp
serves single-user" is now a measured claim, not an argument.** That is the
deliverable. Everything remaining is packaging, porting and paperwork.

## I.5 The M5 scoreboard as it stands (M5 is CLOSED; these are carried facts)

| item | verdict | detail |
|---|---|---|
| [1] scheduler invariants in sim | ✅ MET | 7 traces × 3 seeds × 2 cost models = 42 green, plus a 1,000,130-step soak over 11,870 traces with 5,142 preemptions, every invariant after EVERY step |
| [2] the bars | ❌ **MISSED at 0.51× when M5 closed; now 0.955× @8 / 1.023× @16 against stale bars, 0.687× against a live referent** | **measured, NOT gate-stamped** — see Stage 1.3 |
| clause 2 (Phase 1), llama half @8, on the pre-registered END-TO-END WALL | ✅ **PASS WITH MARGIN 1.181×, 5/5 pairs (2026-09-15 afternoon)** after the multi-key attention kernels (bit-identical, −6.9%), 12 threads and chunk 256; the morning's default-config sitting read 0.970× (parity) | `gate-reports/2026-09-15-clause2-e2e-8-graded.md`; 12T IS the shipping default (`b389674`) and chunk 256 SHIPS (Kabir's ruling, 09-15 evening, `memos/2026-09-15-prefill-chunk-width-DECISION-OPUS.md`) — the graded configuration is the shipped one. **2026-09-18 (`gate-reports/2026-09-18-clause2-e2e-16-32-ollama.md`): llama half @16 ✅ PASS 1.195× (3/3 pairs; H1's point was 1.20×), @32 1.281× (3/3, above @16 — the mechanism bet holds); ollama half ⛔ MISS ≥2× at both widths, 1.735× @8 and 1.836× @16 (2/2 pairs each, spreads 0.2%), as pre-registered — ollama-default serves ONE request at a time on a CPU box, so the bar is against a serial server and stoke's prefill cost misses it: a DOCUMENTED MISS. Found on the way: the listen backlog (16) refused 13 of 32 clients — `b1ee6b3`** |
| [3] ITL <250 ms under co-running prefill | ❌ MISSED, structural — **documented miss by Kabir's ruling 2026-09-16** (`memos/2026-09-16-clause4-itl-stall-DISPOSITION-KABIR.md`) | 738 ms at the smallest legal chunk; 4.4× at M5 close; **stall p50 3.34 s @8 at the SHIPPED chunk 256 (1.84 s at 128, same sitting, 09-15) — ~13× the bound, priced and accepted with the width; retired by §IV.6 row 1, not by a width** |
| [4] S3 fanout ≥13.3 | ✅ MET | median 13.535 (1.018×), all three counterbalanced blocks clear on their own. Dedup is worth **2.284×** here and is the difference between missing by 2.2× and clearing |
| [5] 1 h soak | ✅ MET | — |
| [6] preempt-lossless on a real model | ✅ MET, twice | once on the no-dedup path, then **again on the shipping dedup default**: R preempt 0 · P preempt 2, stream BYTE-IDENTICAL over all 2,400 ids · two mutants diverge |
| [7] close-out ledger | ⚠ carried | see the `history.jsonl` item, Stage 2.3 |

## I.6 The M7 gate (six items) and rung state

| # | gate item | source | needs | state |
|---|---|---|---|---|
| [1] | Llama-3.2-3B parity, incl. long context | forward note | ~~a file that is not on this box~~ | ✅ **MET 2026-09-08** — rung B5 closed, `gate-reports/2026-09-08-m7-b5-closeout.md` |
| [2] | unsloth-file parity run on Q4_K_M | forward note | ~~a file that is not on this box~~ | ✅ **MET 2026-09-15, PASS on the pass rule, signed off by Kabir** — 960 probes, zero BUG, T1 + T4 fired and dispositioned: `gate-reports/2026-09-15-m7-b6-unsloth-probe.md`, memo `memos/2026-09-15-m7-b6-tripwire-anomaly-memo.md` |
| [3] | 24 h soak, **re-specified 2026-10-12 as four 6 h runs** | ROADMAP; Kabir 2026-10-12 | four 6 h windows, the PC free between them | 🟡 rung B7, **1 of 4 runs counted** — **RE-SPECIFIED 2026-10-12 (Kabir), before any run: four separate 6 h runs (seeds 20261012/18/24/30, amended 2026-10-13 before run 2 from 20261012–15 so no chunk seed repeats), each graded over its final 3 h at the same rate (one KV block per 6 h of window), ASan quarantine 16 MiB; the single-lifetime 24 h is OWED before any server-style release (`design-notes/2026-10-12-m7-soak-4x6h-rule.md`, launcher `build/wslmatrix/launch_soak6h.sh <k>`).** **2026-10-13: run 1 VOID — a rig fault, not the engine (the server's budget and the fast guest clock; note §10); the rig fixed and tested 2026-10-15; run 1 re-runs with seed 20261012 in a window Kabir names.** **2026-10-19: run 1's re-run OK** (`gate-reports/2026-10-18-1842-m7-soak.*`, committed `62d8580`; the note's §12): 6 chunks, 0 lost, 4457 generations, no sanitizer reports, 0 OOM/balloon lines, RSS FLAT ⇒ **1 of 4 runs counted; runs 2–4 (seeds 20261018/24/30) each in a window Kabir names.** Before that: **rig shaken down and a 1 h BASELINE banked 2026-09-18 (`gate-reports/2026-09-18-0844-m7-soak.*`: OK, 719 generations, LSan clean); the 24 h itself waits for the window.** ⚠ **Corrected 2026-09-23: the baseline's non-arena anon is NOT flat** (as banked) — re-read at 240 s it climbs a steady ~18–20 MiB/h with VmData on the ASan build after a 5-min warm-up (+ one +21 MiB step), and soak mode graded no RSS at all. ⇒ **the 24 h's RSS rule is PRE-REGISTERED** (`design-notes/2026-09-23-m7-soak-rss-rule.md`): over the final 6 h, VmData and non-arena anon each grow ≤ one KV block (4.5 MiB) by least squares, else the soak FAILS; `tests/tooling/soak_rss_rule.sh`, its 17-case test (8/8 mutants caught) run in the soak's pre-flight; ASan quarantine set to 64 MiB; launcher `build/wslmatrix/launch_soak24h.sh` (shakedown first) |
| [4] | fuzz | forward note | corpora exist (A2's 51 files, `serve_fuzz`) | ✅ **MET 2026-09-18 on the frozen source** — `gate-reports/2026-09-18-m7-fuzz-gate.txt`: `serve_asan` 600 s / 4 clients / 51 files / seed 20260918 on the frozen ASan pair, OK (no sanitizer reports, clean exit, the ≥10-generations floor cleared); the `-DSTOKE_MUT_FUZZ_BAD_BODY` arm built from the same source FAILED the floor at 0 generations ⇒ the grader can fail. (First closed 2026-09-05 with seed 20260829, never banked in the repo.) |
| [5] | README + a REPRODUCIBLE benchmark report | forward note | a box that passes the quiesce bar | 🟡 rung B8 **first pass 2026-09-18 (`5e56039`): README rewritten on the measured position, `docs/BENCHMARKS.md` reproducible from its commands; the soak row is the one line still owed** |
| [6] | Kabir sign-offs | ROADMAP | Kabir | ⬜ last |

| rung | what | state |
|---|---|---|
| B1 | the cross-thread surface | ✅ DONE — TSan found **six** races where reading the code found one; zero after the fix. Cost against `m6` with a null arm: 8f32 0.09% against a 0.25% floor, 16f32 0.03% against 3.0% |
| B2 | the bars' absolutes on a quiet box | ✅ DONE — **and it found the quiesce instrument was reading noise**: a one-shot `Get-CimInstance` on a FORMATTED WMI counter returned 7–38% where three other instruments read ~3%. Wrong in BOTH directions ⇒ **any `quiesced: true` banked before 2026-08-30 is weak evidence** |
| B3 | the first-token ordering | ✅ DONE — 7 of 8 pairs differed unfixed, and every raced stream began with token 264 whatever its seed (the argmax, seen from outside). Fixed by draining the pending queue on a callback MISS; 8 pairs, 0 differing with the window still widened |
| B4 | the small debts | ✅ **DONE** — `stoke run` sampling flags ✅ (and the header's claim that the CLI reproduces the SERVER for a given seed was FALSE when tested — a one-position shift, invisible from either side alone). **The Linux C++ TU ✅ — g++ 13.3.0 is installed in WSL and the matrices print `abi_tu_cxx: stoke.h C++ compile+link OK` (`build/wslmatrix/wsl_matrix_b1ee6b3.log`, also on `9eca59e` / `d56e496`).** *(Recorded 2026-09-23: this row said OPEN for weeks after it was not.)* |
| B5 | Llama-3.2-3B incl. long context | ✅ **DONE 2026-09-08** — all six ruled elements discharged and cited; the grid 16/16 streams, 1024 probes, zero BUG; the fourth targeted delta (the Llama-3 header template) did not exist until `c3b4691` and was found by reading a different source than the parity items |
| B6 | the unsloth Q4_K_M parity run | ✅ **RUN 2026-09-14/15** — 15/15 streams, 960 probes to position 2933, 23 divergent, **all ARITHMETIC-class, zero BUG**; C1-Q re-earned the batched licence on the Qwen3 graph (5.2e-05), T6 bit-identical. **T1 fired at 2.40%** (the carried expectation: bartowski 3.10%, Llama 1.37%; threshold kept, memo) and **T4 fired at 1.97×** into `[1200,∞)` — part a localized tail at the first generated positions of the ≥1893-token streams (an 11.88 anchored Δ at an *agreeing* position, stoke bit-identical across its prefill paths there, the reference stable across batch shapes to 7e-5), part a real widening past ~1500. Survey ordered by Kabir at sign-off and RUN (report §8): llama.cpp is as far from the f32 reference as stoke is; the boundary shrink is the engine class (llama.cpp sides with stoke); the 11.88 position is a knife-edge (stoke's `--kv q8` arm, llama.cpp and the reference in one basin, the graded arm in the other; smooth 1000× amplification through layers 13–25, gone under a one-token substitution 1500 tokens earlier) — not a defect |
| B7 | the 24 h soak | 🟡 freeze done (`dbe16db`), rig shaken down (`fd7d26f`), 1 h baseline OK (`968cc45`); **2026-09-23: the RSS rule pre-registered, ASan quarantine 64 MiB, the launcher written — READY; the 24 h needs a window Kabir names**; **2026-10-12: re-specified as four 6 h runs (Kabir), the rule and the launcher ready, no run yet**; **2026-10-13: run 1 VOID (rig, not engine — note §10); the rig fixed and tested 2026-10-15, and the fast clock's cause found and stopped (systemd-timesyncd fighting WSL's own sync — note §11); 0 of 4 runs counted**; **2026-10-19: run 1's re-run OK (note §12; `gate-reports/2026-10-18-1842-m7-soak.*`) — 1 of 4 runs counted; runs 2–4 wait for windows Kabir names** |
| B8 | README + reproducible benchmark report | 🟡 first pass `5e56039`; finishes after B7 |

---

# PART II — THE SEQUENCE

Six stages. **Stage 0 is three things Kabir does that cost nothing and unblock
the longest pole in the project.** Stages 1 and 2 run in parallel with Stage 0's
downloads. Stage 3 is the only real remaining engineering. Stages 4 and 5 are
freeze, ship and housekeeping.

Every item states: **what · why it is here · blocked on · done when · hazards.**

---

## STAGE 0 — UNBLOCK (today; ~zero effort; all three are Kabir's)

> These three cost a click each and every one of them gates a whole rung. Doing
> them first converts dead wall-clock into parallel wall-clock. **Nothing else in
> this file should start before these are started.**

### 0.1 — Start both model downloads

- **What:** fetch **Llama-3.2-3B** (GGUF) and the **unsloth Qwen3-4B-Instruct
  Q4_K_M**.
- **Why here:** they gate B5 and B6, and **B5 is the longest pole in Phase 1**
  (§3.1). They download unattended while Stages 1–2 run on the box.
- **Blocked on:** Kabir. **Owner:** Kabir.
- **Done when:** both files sit in `models/` and `stoke info` reads each one.
- **Why the unsloth file specifically:** the published numbers should use the
  file people actually download — **~87.5k/month against bartowski's ~11k**,
  and bartowski's is what *everything so far* was developed against. If the two
  quantizations differ anywhere, the README is quoting the wrong file.
- **Hazard:** `models/` currently holds only the two Qwen3-4B files. Do not
  overwrite either — the Q4_K_M there is the parity lineage for M2/M3/M4/M5 and
  every archived gate binary is graded against streams from it.

### 0.2 — `apt install g++` on the WSL side

- ✅ **DONE** (recorded 2026-09-23): g++ 13.3.0 is in WSL and every WSL
  matrix since `9eca59e` prints `abi_tu_cxx: stoke.h C++ compile+link OK`.
- **What:** install a C++ compiler in WSL.
- **Why here:** it closes the open half of B4 and **the last of the five items
  M6 handed to M7**. It blocks nothing, which is exactly why it should be
  cleared now rather than discovered at freeze time.
- **Blocked on:** Kabir. **Owner:** Kabir (it is a system install).
- **Done when:** the Linux C++ TU compiles `stoke.h` clean and the matrix's C++
  leg runs.
- **Why it matters at all:** `stoke.h` is **FROZEN**. A frozen C ABI that has
  never been included from a C++ translation unit on the platform most
  integrators use is a promise that has not been tested. Windows already covers
  the C++ TU; Linux does not.

### 0.3 — Approve one `llama-server` sitting

- **What:** permission to run llama-server for one measurement block, covering
  **@16 and @8 bracketed in the same block**.
- **Why here:** Stage 1 cannot start without it, and Stage 1 gates Stage 2,
  which gates the README.
- **Blocked on:** Kabir. **Owner:** Kabir.
- **Standing rule:** `[[ask-before-running-servers]]` — a long-lived local model
  server is never started unasked. The 09-01 @8 run was approved as a one-off;
  **that approval does not extend to @16.**
- **Hazard:** the server must be **tree-killed** after, as the @8 run was.

---

## STAGE 1 — SETTLE THE DENOMINATOR

> **Nothing else in performance is worth doing until this is done.** Every
> downstream question — whether clause 2 is met, whether the prefill policy is
> worth changing, what the README claims — divides by a number that is currently
> known to be wrong by 39%.

### 1.1 — Measure llama-server at N=16, today, same sitting as @8

- **What:** M0's own harness and tuned config against `s2-closed-16`, 3 runs,
  `--fresh`, with the @8 cell re-taken in the same block to bracket it.
- ⚠⚠ **AND THE OLLAMA HALF, WHICH THIS FILE ORIGINALLY MISSED.** Clause 2 is
  **"≥2× ollama-default AND ≥1.0× llama-server-tuned"**, and `MEMORY.md` makes
  beating Ollama at concurrency the definition of Phase 1 being *"useful,
  standalone"* — it is the marketing headline, not a footnote. **The
  ollama-default referent is exactly as stale as the llama one was (2026-07-17),
  and the llama one moved 39% when re-measured.** So the 2×-ollama bar is
  currently of unknown difficulty in an unknown direction, and the sentence the
  project most wants to say in public rests on it. **Measure ollama-default @8
  and @16 in the same sitting.** It is the same server-approval, the same
  harness, and a third config on an already-scheduled run.
- **Why here — and why this is the single highest-value measurement left:**
  **@16 is where the entire thesis lives.** stoke *rises* across 8→16
  (16.907 → 17.181) while M0's own llama *collapses* (17.664 → 14.163), and
  today's `batched-bench` curve found the same wall at B=4. @8 has been
  re-measured (24.608); **@16 has not, and it is the cell that could turn
  "0.687× and short" into "clause 2 genuinely MET on the axis the project was
  founded to win".**
- **Blocked on:** 0.3. **Owner:** the session, on Kabir's approval.
- **Done when:** an @16 llama-server figure exists from a named block on a
  quiesced box, with @8 taken in the same block.
- ⭐ **PRE-REGISTERED 2026-09-16, awaiting Kabir's go:**
  `superpowers/design-notes/2026-09-16-clause2-n16-n32-ollama-preregistration.md`
  — four blocks (llama @16 · llama @32 · ollama-default @8 · @16, each vs the
  @8-graded stoke arm), bands frozen, ~2 h 15 min, launcher with the banks.
  The pre-flight corrected the plan: no prompt files or `--pool-blocks` on
  the HTTP path (the pool derives, 1,408 blocks @32); llama @32 needs
  `llamacpp-tuned@ctx=22528` because the policy's `-c 65536` predicts ~14 GB;
  and ⚠⚠ **`src/server.c` `MAX_CONNS 24` makes @32 a 24-wide server plus a
  queue — raising it (48 proposed) is a source change and Kabir's call**
  before block 2. Blocks 1, 3, 4 can run on `d56e496`.
- ✅ **DONE 2026-09-18** — Kabir: yes to 48, go. Four blocks, 09:46–13:21,
  `gate-reports/2026-09-18-clause2-e2e-16-32-ollama.md`. llama @16 1.195×
  PASS, @32 1.281× (above @16), ollama @8 / @16 1.735× / 1.836× (≥2× MISS,
  documented; ollama-default is a one-slot server here). ⚠ And @32 found
  what the MAX_CONNS pre-flight had not: the listen backlog of 16 refused 13
  of 32 clients (`b1ee6b3`; the pre-flight read the connection table and not
  the listen queue). Referent bracketed every block within 2.5%.
- **Hazards:**
  - **Bracket it.** Run the referent BEFORE and AFTER the stoke arms; if the two
    readings are not within ~1% the block is not quiet and nothing in it is
    quotable. (The 09-01 prefill block bracketed at 44.897 / 44.872 s = 0.06%,
    which is what licensed its conclusion.)
  - **Name the block in every number published from it.** The ratio travels
    between sittings; the bar fraction does not.
  - **The serving aggregate is MORE box-sensitive than `pp512`** — a block
    before quiescing read 0.898× where the same binaries read 0.955× after,
    while the *ratio* held at 1.904×. That is why both must be published.

> **1.2 and 1.3 are SUPERSEDED (recorded 2026-09-18).** Clause 2 was re-specified on
> 2026-09-08 (Kabir-approved) to a same-sitting ratio on END-TO-END WALL and has been
> graded that way at @8, @16 and @32; the M5 aggregate bars stay what the M5 row says
> they are — a documented miss on a metric that bills stoke's early first tokens — and
> are not re-anchored or re-graded. B8 quotes the wall ratios and states the aggregate
> miss beside them (`docs/BENCHMARKS.md` §1). The hazards below still apply to any
> future bars run.

### 1.2 — Re-anchor both bars, in the hard direction, with the block named

- **What:** amend the bars in `run_m5_bars.ps1` and `ROADMAP.md`: @8
  **17.7 → ~24.6**; @16 → whatever 1.1 reads.
- **Why here:** a 46-day-old absolute on a box with a 28% between-sitting range
  cannot govern a gate. Grading against a number you already know is 0.72× of
  reality is not a pass, it is an accounting error.
- **Blocked on:** 1.1. **Owner:** the session; Kabir overrules.
- **Done when:** `$GATE = @{ 8 = …; 16 = … }` carries the new values, every
  affected row in `ROADMAP.md` says which block produced them, and a memo
  records the re-anchor.
- **Hazards:**
  - **@8's move is the HARD direction (17.7 → 24.6) and is therefore SAFE**
    under the standing asymmetric guard — the guard exists to stop bars being
    made *easier*.
  - ⚠⚠ **@16 may go either way.** If today's llama-server @16 reads **below
    16.8**, that is an **EASIER** re-anchor and the **asymmetric guard plus the
    bar-miss memo protocol apply in full.** Do not quietly lower it.
  - **There are TWO gate bars, not four.** `run_m5_bars.ps1` is the authority:
    the binding bar at each width is the **HIGHER** of the two families
    (llamacpp-tuned 17.7 @8; 2×-ollama 16.8 @16, "where the inversion makes it
    the harder bar"). **14.2 and 10.8 are the other family's values at those
    widths, NOT cells to pass**, and every other N is a `ref` label.
  - The **2× ollama-default** half of clause 2 has its own referent and it has
    not been re-measured either. Re-anchoring only the llama half leaves clause
    2 half-graded.

### 1.3 — Re-grade gate [2] properly with `run_m5_bars.ps1`

- **What:** run the actual gate runner against the re-anchored bars.
- **Why here:** **everything currently quoted is MEASURED, NOT GATE-STAMPED.**
  The 09-01 numbers are 4 units @8 and 3 @16 on a hand-quiesced box, **f32
  only**. Gate [2]'s protocol is three blocks, medians, pinned, its **own**
  quiesce instrument gating each unit, **and the q8 cells**.
- **Blocked on:** 1.2, and a box that passes the quiesce bar.
- **Done when:** a gate report exists covering **8f32, 16f32, 8q8 and 16q8**,
  banked per-unit with per-unit quiesce flags, against the re-anchored bars.
- **Hazards — this is the item with the most ways to publish a wrong pass:**
  - ⚠⚠ **`8q8` and `16q8` are UNMEASURED.** They are not a formality; the Q8-KV
    path has its own kernel behaviour and has never been through the bars.
  - **Slicing moves the quiesce gate to PER INVOCATION.** Every unit must carry
    its OWN `quiesced` flag and the report must read **EVERY** quiesce record —
    not `$q[0]`, or a failure in slice 7 publishes under a PASSED banner.
  - **Every bank keyed by the BINARY.** Three separate sightings of stale banks
    in this project, each of which would have published old numbers under a new
    commit. *A stale binary has an mtime you can look at; a stale bank does not.*
  - **A sliced runner must never truncate its own bank.** On 09-01 a runner did
    `: > "$OUT"` and was called once per slice, so it banked **4 of 12 units**
    while printing all twelve correctly.
  - **`[Parser]::ParseFile` before running it — and that is not sufficient.**
    A parse check proves a file is executable, not that its statements resolve.
  - **Run it FOREGROUND, sliced one unit per invocation.** Long background jobs
    on this box were killed three times mid-run with no user message.
  - **ISLC:** `tasklist | grep ISLC` does not detect it; use `Get-Process`.
    The runner THROWS on it; `-AllowISLC` is the deliberate override.
  - **`--no-dedup` must be explicit.** Dedup is the serve default now; three
    runners once meant "off" by omitting the flag, and the flip would have
    turned the bars into a measurement of a workload the bars never described
    (S2's clients differ by one digit, so dedup would adopt nearly everything).

---

## STAGE 2 — THE THREE CALLS ONLY KABIR CAN MAKE

> All three must land **before** B8, because the README has to state a
> disposition for each. All three are informed by Stage 1 and none of them is a
> session's call.

### 2.1 — The prefill policy (clause 2's real content)

- **The question:** accept the serving aggregate as a **documented miss with a
  measured cause**, or **redesign to batch prefill across sequences**.
- **What each costs, priced:**
  - *Batched prefill* would raise AGGREGATE(M0) — it is exactly what llama does
    to score 24.6 — and would **worsen first-token latency**, which M5 chose
    deliberately. stoke's TTFT p50 is currently **1.50× better** than llama's
    (27.6 s vs 41.3 s) and its earliest first token lands at 5.3 s against
    llama's 20.7 s.
  - *Documented miss* costs a launch talking point and gains the property the
    project exists to deliver.
- **The session's recommendation, on record:** **do not chase it.** It is the
  only remaining change that would make the engine *worse at its stated purpose*
  in order to score better on a borrowed metric — and the metric is the part of
  M0 that has aged worst. The total-work gap is **5.5%**; the aggregate gap is
  **31%**; the difference between those two numbers is entirely *where the clock
  starts*.
- ⚠ **The honest counter, stated so it is not buried:** Kabir's standing
  priority is *"achieve the goal first, 0.51 bar is unacceptable"*, and clause 2
  as written says **≥1.0× llama-server-tuned**. 0.687× is not 1.0×. **Re-specify
  the criterion BEFORE re-grading it, never after seeing the number** — the
  legitimate version of this is 1.1 (measure @16 live, where the thesis
  predicts stoke wins), not a redefinition chosen because @8 missed.
- **Done when:** the disposition is written into `ROADMAP.md`'s M7 row, with the
  measured mechanism attached.

### 2.2 — Gate [3], the 250 ms bound

- **The question:** **redesign / re-anchor / accept as a documented miss.**
- **What is known:** 738 ms stall p50 at chunk 64, the smallest S-ALIGN-legal
  width (`BP_BLOCK_TOKENS` is 16 and M4 froze it). `cadence` p50 ~171 ms is
  inside 250 at every width. ⇒ **the miss is structural to putting a whole
  prefill chunk in one scheduler step, not a tuning failure.** Getting under 250
  ms is a **scheduler redesign**, and the chunk lever is exhausted in both
  directions (halving costs 4.8% end-to-end, doubling buys 0.06%).
- **The re-anchor argument, on record:** 250 ms is a **human-interactivity**
  figure while M5's operating point is **agent swarms** — one user's fan-outs
  and tool chains, not eight humans. That was flagged as Kabir's call and his
  headline as early as 2026-08-19 and has never been decided.
- **The options, all four already priced** (from the M5 row): (a) 4.4× on
  prefill at chunk 16 — the arithmetic wall, named compute levers cap at ~18%;
  (b) `BP_BLOCK_TOKENS` < 16, reopening M4's closed set, and P5 says the return
  is sub-linear; (c) a scheduler that does not hold decode rows behind a prefill
  chunk's step — **undesigned**, and it trades against the amortisation E3
  measured saturating by ~23 rows; (d) accept as a documented miss.
- **Done when:** the choice is recorded; if (d), with the named cause and the
  measured price beside it.
- ✅ **DECIDED 2026-09-16 — (d), Kabir's ruling on Opus's recommendation:**
  `memos/2026-09-16-clause4-itl-stall-DISPOSITION-KABIR.md` — the cause (one
  step carries the chunk; the stall IS the chunk's forward time), the price
  (3.34 s p50 @8 at the shipped 256; 738 ms at the smallest legal width; 678–1,108
  ms even at an off-spec 16), and the four reasons (c) is deferred to Phase 1.5:
  the referent stalls the same way (to be recorded in the @16/@32 sitting), the
  redesign is architectural near the freeze, its payoff is bounded on a 6-core
  box (decode on 4 threads ≈ 250–300 ms/step), and the throughput half of its
  case weakened once the 09-16 sitting found decode ~60% compute at N=8.

### 2.3 — The `history.jsonl` rule

- **The question:** **narrow the rule, or give the file a second `kind`.**
- **Why it is here:** the standing rule is *"a row per hot-path-touching green
  commit"*, and it has been **answered by OMISSION for all of ADDENDA 39–44** —
  six addenda covering every kernel win of this milestone. The reason is
  structural, not laziness: **`bench --ops` is single-thread Q8_0 with the pool
  inert and cannot see the batched K-quant path at all.** Following the rule to
  the letter would append rows that measure a path the changes cannot reach.
- **Why it must be settled before B8:** `bench/history.jsonl` is a ledger whose
  own label says a row "stays comparable to the K3 rows". If the README cites a
  ledger with a six-addendum hole in it, the hole is the story.
- **Done when:** either the rule's scope is narrowed in writing, or the file
  gains a second `kind` and the missing rows are added under it.
- **Small, and the smallest of the three — but it is a standing rule currently
  being broken silently, which is the category this project treats most
  seriously.**
- ✅ **SETTLED 2026-09-23 (Opus decides and records; Kabir overrules) —
  NARROWED, not a second `kind`.** The rule now reads: *a green commit that
  changes code on the path `bench --ops` measures — the single-thread Q8_0
  matvecs, `vec_dot_q8_0_f32`, `dequant_row_q8_0`, and `rmsnorm`, `rope_neox`,
  `softmax`, `swiglu`, `argmax` — appends an `ops` row.* The batched K-quant
  kernels, attention, the scheduler and the server are not ledgered there;
  their record is the same-sitting gate reports and `docs/BENCHMARKS.md`.
  - **Why not a second `kind`:** its rows would be serving absolutes from
    different sittings, and on this box those moved 1.8× with no code change
    (standing rule 5; BENCHMARKS.md's own header) — a ledger of them invites
    exactly the cross-sitting diff this project forbids. A ratio from ONE
    sitting is the serving number that travels, and it already has a home.
  - **Why not honour the old wording:** `bench --ops` is single-thread Q8_0
    with the pool inert, so rows for ADDENDA 39–44 would have measured a path
    those changes do not reach — numbers that look like evidence and are not.
  - ⚠ **ONE ROW IS OWED under the narrowed rule.** Since the last row
    (`ec9e8cb`, 2026-08-28) one commit touched the measured path: `0c2da0c`
    changed `rope_neox` (the Llama-3 rope work; `ops.c` has no other change
    in that range, and no commit touched the Q8_0 matvec or dot). One row on
    the M7 freeze binary (`b1ee6b3`), taken on a quiesced box and NOT during
    the soak, closes it; it is a precondition of the M7 stamp (4.5).
  - Written into `ROADMAP.md`'s cross-cutting line in place, dated.

---

## STAGE 3 — THE REMAINING ENGINEERING, RISK-ORDERED

### 3.1 — B5: Llama-3.2-3B parity, including long context ← THE BIG ONE

- **What:** run, grade and gate the first non-Qwen model this engine has ever
  seen, including at long context.
- **Why here:** it starts the moment the file lands (0.1) and **it is the
  largest unknown left in Phase 1.** Every architecture assumption in
  `model.c`, `gguf.c` and the tokenizer has only ever been exercised against
  Qwen3-4B. The M7 entry note's own expectation: *"expect the architecture
  assumptions to be where the work is, not the kernels."*
- ⚠⚠ **Treat this as MILESTONE-risk, not rung-risk.** It is the one item in
  this file that could reasonably take several sittings and turn up structural
  work. It is sequenced early for exactly that reason — a surprise here after
  the soak would invalidate the soak.
- **Blocked on:** 0.1. **Owner:** the session.
- **Done when:** greedy parity is proven against llama.cpp on the new file at
  short AND long context, to the standard M2/M3 set (token-exact streams with
  a τ frozen before grading, zero BUG-class), and the run is banked.
- **Hazards:**
  - **Different GQA ratio, different rope, different vocab, different chat
    template.** The attention kernels' four-lane score pass is *specifically*
    the GQA group's four q-heads — `att_score4` is 128-bit **because G is 4**.
    A model with a different G does not just run slower, it needs the fallback
    path, and that path must be proved live rather than assumed
    (`STOKE_ATT_SIMD=0` forces the SSE2 arm — use it).
  - **A tokenizer that passes on one vocab is not proven on another.** M1's
    22,848-line unicode sweep is a Qwen artefact.
  - **Long context is where M4.5's L4 slope claim lives** (0.0289
    ms/context-token). It has never been measured on another model.

### 3.2 — B6: the unsloth Q4_K_M parity run

- **What:** the same parity treatment against the unsloth quantization.
- **Why here:** cheap once the file exists, and it re-points every published
  number at the file people actually download (~87.5k/mo vs ~11k).
- **Blocked on:** 0.1. **Owner:** the session.
- **Done when:** the run is banked and the README's file provenance is decided.
- **Hazard:** if the two quantizations differ in tensor types anywhere — as
  Q4_K_M already does internally, leaving **18 of 36 `ffn_down` layers Q6_K**
  (model-wide **216 Q4_K / 37 Q6_K / 145 F32**) — then the kernel mix changes
  and the perf numbers move with it. **`stoke info` prints the quant type; read
  it before comparing anything.** A fixture drawn BY NAME from a real model can
  differ in exactly the property the comparison rests on — this project has
  already withdrawn one 6.5× result to that mistake.

### 3.3 — Fuzz (M7 gate item [4])

- **What:** run and stamp the fuzz leg.
- **Why here:** the corpora already exist — A2's 51 files and `serve_fuzz` —
  so this is a run-and-bank, not a build. It has simply never been done.
- **Blocked on:** nothing. **Owner:** the session. **Can fill any blocked window.**
- ✅ **DONE 2026-09-18** — `gate-reports/2026-09-18-m7-fuzz-gate.txt`, both arms, on the frozen source.
- **Done when:** a gate record exists with the corpus named and the build
  identified by binary.
- **Hazard:** ⚠ **`build/{win,linux}-mut-fuzz-badbody`** (`-DSTOKE_MUT_FUZZ_BAD_BODY`)
  exists precisely so the fuzz leg's own checks can be proved able to fail.
  Use it. A green fuzz run whose grader cannot fail proves nothing — and this
  project has caught that exact shape before (the soak that had **never entered
  the sampler** because its builder hardcoded `"temperature":0`, while a 400
  counted as an answered exchange, so a soak of pure refusals passed every
  check it had).

### 3.4 — `KERNELS_BATCH_TILE` re-sweep against VNNI *(optional, cheap)*

- **What:** re-sweep the one tiling constant still inherited from the old kernel.
- **Why here:** `KERNELS_ROW_TILE` was re-swept after the VNNI dot landed and
  **the optimum MOVED, 32 → 64** (against maddubs the curve read 8→6014 /
  16→5649 / **32→5434** / 64→5545 ms; against VNNI it reads 16→4487.6 /
  32→4325.1 / **64→4255.9** / 128→4356.3). `KERNELS_BATCH_TILE=16` was chosen
  against maddubs (=8 was 1.7% worse, =4 8% worse) and **has not been re-swept**.
  The dot now costs **9 p01 uops where it cost 16**, and carries a new 4 KiB
  permuted-activation buffer.
- **Why it is optional:** it is a small percentage on a metric that is already
  ahead of the referent. But it is the last inherited constant, and *"a swept
  constant is scoped to the KERNEL it was swept against"* is now a five-time
  lesson in this milestone.
- **Blocked on:** nothing. **Owner:** the session.
- **Done when:** swept, with a null arm, ranges disjoint, and either changed or
  explicitly confirmed at 16.
- **STATUS 2026-09-16 (K3 of `docs/superpowers/design-notes/2026-09-16-m7-kquant-partial-group-preregistration.md`):** tiles 32 and 64 built and graded exact (`test_matvec_batch` 2280/0, `test_simd` 15/15, knob proved live by three `simd.o` hashes + `-dM`); measured in situ only (the kernel bench is capped at the tile), S2 @8 12T, two ABAB pairs each: **t16/t32 1.017 · 0.945 (inconclusive), t16/t64 0.953 · 0.929 (64 slower in both orders)** — ⚠ INDICATIVE, NOT A RESULT: Chrome came on mid-block (Kabir, 11:0x), walls jumped 70 → 80 s. **16 stays; 32 owes one quiet re-run of 3 pairs; 64 is out.** And the finding that reframes this item: **the single-thread kernel bench over-predicts an instruction-count change's in-situ value by 5–10× on this box** (K2: 1.26× at N=8 alone, 1.10× SMT-paired on one core, 1.01–1.05× in situ at 12T) — every kernel A/B now runs SMT-paired first (`build/kpad/run_smt_ab.ps1`), and the transferable levers are port-0/1 uop cuts, of which the bit-exact contract leaves ~5% (the 8-wide float fold).
- ⚠ **Hazard, learned the hard way on ROW_TILE:** *a constant validated on the
  axes you SWEPT is not validated on the axis you are GRADED by.* ROW_TILE was
  swept on `run` and `batch` and shipped while `bench --serve` — the gate
  metric — went unmeasured. It turned out fine, **by luck**. Sweep against the
  graded axis too.
- ⚠ **Non-vacuity matters here:** the three `stoke.exe` sizes were nearly
  identical when ROW_TILE moved, so the knob was proved live by three different
  `simd.o`/`vec.o` hashes plus `gcc -E -dM`, not by the binary differing.

### 3.5 — *Conditional on 2.1:* the scheduler prefill-policy work

- Only if Kabir chooses to chase the aggregate. **Undesigned.** It would touch
  `sched.c`'s admission and step construction, trade against the amortisation
  E3 measured saturating by ~23 rows, and worsen TTFT by construction.
- If taken, it must land **before** the freeze (Stage 4), and B1's cross-thread
  locking and B3's drain-on-miss both sit in the code it would move.

---

## STAGE 4 — FREEZE → SOAK → PUBLISH

### 4.1 — Code freeze for M7

- **What:** declare the tree frozen for the soak.
- **Why here:** **anything committed after this invalidates B7.** The soak must
  exercise the code that ships.
- **Done when:** the freeze is recorded with the commit it freezes at, and the
  binary is archived and hashed (`stoke sha256`), the way M6's soak binary was.
- **Precedent:** `stoke.exe` was frozen the same way from M3 until tag `m3`,
  with a single stated exception (a BUG fix-cycle). Use the same wording.
- ✅ **DONE 2026-09-18 — `memos/2026-09-18-m7-code-freeze.md`: `src/` frozen
  at `b1ee6b3` until tag `m7`, sole exception a BUG fix-cycle (which restarts
  B7). Archived by hash: `build/stoke-m7-gate-fbc64f8c.exe` (Windows),
  `stoke-m7-gate-linux-c922098f`, and the soak pair
  `stoke-m7-soak-linux-asan-337bbc1b` / `…-serve_fuzz-ab4299ca`; WSL matrix
  on `b1ee6b3` green on all three legs the same afternoon.**

### 4.2 — B7: the 24 h soak

- ⚠ **Re-specified 2026-10-12 (Kabir): four separate 6 h runs**, each graded
  over its final 3 h at the same rate, ASan quarantine 16 MiB, seeds
  20261012/18/24/30 (amended 2026-10-13 before run 2 so the 24 chunks never
  repeat a seed — the note's §8); launcher `build/wslmatrix/launch_soak6h.sh <k>`. The rule,
  and what the change gives up (a single-lifetime 24 h, now owed before any
  server-style release), are in `design-notes/2026-10-12-m7-soak-4x6h-rule.md`.
  The text below describes the original 24 h shape; every hazard in it
  applies to each 6 h run. **Run 1 (2026-10-13) was VOID:** the server's
  budget did not cover the driver and the guest clock ran fast, a rig fault
  with the engine clean; the fixes and their tests are the note's §10, and the clock's cause (this distro's systemd-timesyncd fighting WSL's own time sync, stopped for every run) is §11.
  **Run 1's re-run (2026-10-18/19, seed 20261012) was OK:** 6 chunks, 0 lost,
  4457 generations, RSS FLAT (`gate-reports/2026-10-18-1842-m7-soak.*`, the
  note's §12). Runs 2–4 remain.
- **What:** a full day of randomized serving traffic on the frozen tree.
- **Why here:** after B1 and B3 so it soaks code without a known race in it,
  and after 3.1–3.4 so it soaks the shipping kernels.
- **Blocked on:** 4.1, and **a box that can be left alone for a day.**
- **Done when:** 24 h clean, with the driver reporting **what its traffic
  actually contained** and the server's own log showing the completed
  generations — **and, pre-registered 2026-09-23 before the run, the RSS rule
  reading FLAT**: over the final 6 h, VmData and non-arena anon each grow ≤ one
  KV block (4.5 MiB) by least squares; CLIMBING or VOID fails the soak
  (`design-notes/2026-09-23-m7-soak-rss-rule.md`, `tests/tooling/soak_rss_rule.sh`).
  Launch: `build/wslmatrix/launch_soak24h.sh` as root (the 6-min shakedown
  first, the 24 h only on its OK; ~24 h 15 min wall). Verdict: the
  `=== M7 soak:` line, which now carries `RSS FLAT` / `RSS NOT FLAT`.
- **Hazards — every one of these is a scar, not a precaution:**
  - ⚠⚠ **A soak that never enters the sampler passes every check it has.** M6's
    did: its builder hardcoded `"temperature":0`, and a 400 counted as an
    "answered" exchange, so a soak of pure refusals looked green. The re-bank
    reports its traffic mix (55 sampled / 17 penalized) and the gate requires
    ≥10 completed generations **from the server's own log**.
  - `serve_golden.sh`'s **`--idle-timeout 2` is a REGRESSION GUARD, not
    tuning** — do not "clean it up". The idle reaper used to destroy finished
    responses whenever generation outlasted the timeout, which is *any* long
    generation on *any* loaded box, and it is silent server-side (`curl: (52)
    Empty reply` while the server logs `finish=stop`).
  - **This is 24 h of box time in which nothing else is measurable.** Plan 4.3
    into it.
  - ⚠⚠ **The VM can end before the soak does, and it ends the leak check with
    it** (2026-09-14: the 1 h probe's server died FIRST at 2.7 GB RSS, the
    sidecar outlived it, the VM shut itself down two minutes later — host
    pressure → hv_balloon → guest OOM kill of the largest process → empty
    distro → idle timer; Hyper-V event 18508 "by the guest OS"). The rig now
    (`28761aa`) puts the server and a `dmesg -W` witness at `oom_score_adj
    -1000` (root: `wsl.exe -u root`), banks the witness to `/mnt/m`, refuses
    the soak without an explicit `.wslconfig` (written 2026-09-18: `memory=8GB
    swap=4GB`), and reports a killed chunk instead of hiding it. **Still
    Kabir's: a quiet host for the day** — the protections make a kill land on
    a chunk, they do not make the box free.

### 4.3 — The M7 planning doc *(write it DURING the soak)*

- **What:** the Phase-1.5 planning document, which `ROADMAP.md`'s "Beyond Phase
  1" section names as **an M7 deliverable**.
- **Must scope:** **ARM NEON** and **CUDA** (`.cu` behind `extern "C"`, onto the
  proven scheduler) — ⭐ **IN THAT ORDER. DECIDED BY KABIR 2026-09-01: "arm
  first, we'll make it embed on most of the hardware."** That reframes Phase 1.5
  from a GPU phase into an **EMBEDDABILITY** phase, and the planning doc must be
  written to that goal: the target hardware matrix, the ARMv8.2 `SDOT`/`SMMLA`
  question (the direct counterpart to `vpdpbusd` — the Q4_K lane-permutation
  insight is expected to transfer but **the overflow ladder must be recomputed,
  not inherited**), cross-compilation and toolchains, and the size budget.
  Full reasoning and the matrix: **`COMPANY.md` §7.**
- **Must deliver a verdict on the three candidates parked by the Fable 08-02
  review specifically for this document** — *none of them touches a Phase-1
  gate; they were listed so they are weighed here, not forgotten*:
  1. **Speculative decoding** — trades the spare compute this box has for the
     bandwidth it lacks, which is the same physics M0 measured, and the M5
     **replay-not-resample** rule composes with a verify step.
  2. **Cross-request prefix reuse** — agent workloads resend system prompts
     every call, and it falls out of M4's refcounted dedup **if blocks outlive
     requests**.
  3. **Grammar / JSON-constrained sampling** — sampler-layer; the AySen
     `local()` runtime will want it at Phase 3.
- **Why here:** it needs **no box**, and the soak is 24 h during which the box
  is unavailable. This is the one piece of Phase-1 work that fits perfectly into
  dead measurement time.
- **Blocked on:** nothing. **Owner:** the session.

### 4.4 — B8: README + the reproducible benchmark report

- **What:** the public-facing README and a benchmark report someone else can
  reproduce.
- **Why here:** last, because it quotes everything above.
- **Blocked on:** 1.3 (the stamped bars), all of Stage 2 (each needs a
  disposition in the text), 3.1/3.2 (the model provenance), 4.2 (the soak
  result). **Needs a box that passes the quiesce bar.**
- **Done when:** README exists, the report reproduces from the commands it
  prints, and every number in it names its block.
- **Publication rules — non-negotiable, and each has a withdrawn claim behind
  it:**
  - **Publish the RATIO and the ABSOLUTES, never a bare tok/s.** Gate item [5]'s
    own rule. This box's absolute swings **1.8× across sittings with no code
    change**; the reproducible number is the ratio.
  - **Name the block.** *The ratio travels; the bar fraction does not.*
  - **Publish TTFT and total wall beside the aggregate.** The aggregate alone
    reports 0.687× for an engine that does the same work in 0.945× the time and
    delivers first tokens **1.50× sooner**. Reporting one of those three is
    choosing a story.
  - **A decode rate is comparable only at the SAME CONTEXT LENGTH.** The same
    binary reads 17.6 t/s at ctx ~30 and 15.3 at ctx ~590 — 13% that is physics
    and attaches to whichever arm carries the longer prompt. One published claim
    was nearly made on a 5× context mismatch.
  - **Reconstruct every published aggregate from the raw fields before printing
    it.** `1016 / (max(last_token_abs) − min(first_token_abs)) = 1016/41.10 =
    24.722`, the reported value to three decimals. **This is the only check that
    catches a unit error** — reading `first_token_abs` as ms instead of seconds
    produced two withdrawn claims in one day and survived every other sanity
    check.

### 4.5 — Stamp M7 → tag `m7`, then Kabir's sign-offs

- **What:** the gate stamp (a mechanical checklist over recorded evidence, never
  judgment) and gate item [6].
- **Done when:** the stamp passes on banked evidence, `m7` is tagged, and Kabir
  signs off. **Phase 1 closes here.**
- **Hazard:** tag `m7` **annotated**, and record **both** the tag object sha and
  the commit sha — see the §I.1 correction. Also archive the gate binary as part
  of the stamp (the M3 ruling added that step for a reason).
- **Precondition added 2026-09-23 (2.3):** the one `history.jsonl` `ops` row
  owed on the freeze binary, taken on a quiesced box, not during the soak.

---

## STAGE 5 — PRE-LAUNCH HOUSEKEEPING

> Independent of Stages 1–4 and can be done in any blocked window, but **all of
> it must be done before anything is public.**

### 5.0 — The LICENCE ← more consequential than everything else in Stage 5, and it has a clock

- ✅ **DECIDED 2026-09-23 (Kabir): AGPL-3.0 + a CLA on outside contributions.**
  Consequences and reasoning: `COMPANY.md` §5. Before the repo goes public:
  the CLA (lawyer-reviewed) and a CLA bot live, a clean public snapshot with
  the internal documents left out, and 5.1's name settled.

- **What:** choose stoke's licence before the repo goes public.
- **Why it is numbered 5.0:** it is **the only irreversible decision in this
  file**, and it is not recorded anywhere in `MEMORY.md`'s business appendix,
  which enumerates thirteen business models and never names a licence.
- ⏳ **WHY THERE IS A CLOCK ON IT — this is the actionable part.** Kabir is the
  **sole author of all 420 commits**, so he holds **100% of the copyright** and
  can still licence the work any way he likes, including dual-licensing later.
  **That option closes the first time a substantial outside contribution is
  merged without a CLA** — after which relicensing needs every contributor's
  consent, which in practice means it never happens. ⇒ **Decide the licence, or
  adopt a CLA, BEFORE the repo is public and accepting PRs.** Doing neither is
  itself a decision, and it is the one that forecloses the most.
- **Why it interacts with the product shape:** stoke's selling point to an ISV
  is a **1.38 MB statically-linked binary with a frozen ABI and no
  dependencies** — and static linking is precisely what copyleft restricts.
  **The technical property and the monetisation wedge are the same property.**
  A permissive licence gives that away for free and forever; a Qt-style
  split (copyleft + a commercial licence for static linking / modification)
  keeps it, at some cost in adoption. Both are defensible. **Choosing by
  default is not.**
- **Blocked on:** nothing. **Owner:** Kabir. **Decide before 5.1 and before any
  public push.**

### 5.1 — The rename decision ✅ DECIDED 2026-09-24: **Lorren**

- **The problem (as it stood):** **PyPI `stoke` is taken** (Fidelity's ML
  training library), **StanfordPL/stoke** is a well-known x86 superoptimizer,
  and **Stoke Space** dominates search. ⚠ Re-checked 2026-09-24: **crates.io
  `stoke` is taken now too**, so "GitHub and crates.io are clean" no longer
  holds.
- **The economics:** rename cost is **~0 until public launch and rises steeply
  after.** This is the last cheap moment.
- **▶ DECIDED (Kabir, 2026-09-24): the product is LORREN, and the company is
  castagents** ("lorren for my app and my company name is castagents").
  "Rename everything" was chosen explicitly, and that option named the app, the
  repo, the command **and the API**. Lorren was Kabir's own pick after ~200
  checked candidates. He turned down fire words ("more related to what we do",
  then "clean and brief"), descriptive llm/mux compounds ("sounds ai") and
  real-word metaphors, and settled on a short, clean invented name ("dont make
  it mean too much about the project").
- **Reconsidered and confirmed the same day.** Kabir asked whether to keep
  "stoke" and launch it from the company site. The USPTO register answered
  it: 93 live STOKE marks, several of them for software and held by other
  companies. They are STOKE, SPC (SaaS, class 42), Stoke LLC
  (business-intelligence services, 35/42), Stoke Games, Stoke Social and THE
  STOKE (downloadable software, class 9), and Stoke Space (class 42). A STOKE
  mark for our software would likely be refused, and a holder could object.
  ⇒ **Lorren confirmed, launched from castagents.com**, which is the front
  door and also hosts the update manifest. ⚠ **castagents.com and
  castagent.com are NOT owned yet** (corrected 2026-10-19: another registrant
  holds both at NameBright, with terms ending 2026-11-25 and 2026-11-07; Kabir
  will not buy them before the product earns money). **The updater's manifest
  needs a domain Kabir controls before any public release.**
- **Availability, checked 2026-09-24** (RDAP from the IANA bootstrap, the
  registry APIs, `gh api`, and USPTO tmsearch; every sweep carried a
  known-taken and a known-free control):
  - **Unregistered: lorren.ai, lorren.dev, lorren.run.** ⇒ Kabir registers
    lorren.ai + lorren.dev; the lookups were public.
  - **Taken:** lorren.com (since 2000, Dynadot, a parked "this domain may be
    for sale" page, so buyable at an unknown price); lorren.app (registered
    2026-06).
  - **Packages:** PyPI and crates `lorren` are free. npm `lorren` is an
    abandoned 2016 style-analyzer with one version ⇒ `lorren-ai` or
    `@lorren/…`.
  - **GitHub:** `lorren` is a person's account; `lorren-ai` and `lorrenhq` are
    free for the organisation.
  - **Trademark, US only:** 4 live LORREN marks (jewelry, real estate,
    dishware, spices); none of the 40 live LOREN marks covers software.
    **EU, India and WIPO are not checked; the CLA lawyer runs the clearance.**
- **Execution:** after tag `m7` (`src/` is frozen) and before the clean public
  snapshot (5.0): `stoke_` → `lorren_` in `stoke.h` and every symbol, the
  binary and library names, the docs, README and BENCHMARKS. It is one
  mechanical commit plus the full suite. The v0 freeze breaks no one, because
  the header has never had an outside user.
- **Owner:** Kabir (decided). **Unblocks:** the website, the domains, the
  public repo, the CLA text.

### 5.2 — Land `main`

- **The state, measured today:** local `main` = `5f6ced8` = the `m45` commit.
  **HEAD is 177 commits ahead of `main`**, and **`main` is 214 commits ahead of
  `origin/main`**, which sits at `71e84aa` (M1) *deliberately*.
- **What:** fast-forward `main` to HEAD and push. **It is a clean
  fast-forward** — HEAD's ancestor chain contains `origin/main`, so **no force
  is needed.**
- **Why it is not done already:** moving the default branch was never asked for
  and is a separate call from the 09-01 push. **Standing rule: the manager never
  pushes unasked.**
- **Owner:** Kabir's call, session executes.
- **Also parked here:** the `m45-aprime` branch (`5f6ced8`) — three
  unverified-by-design commits from the 08-14/15 overnight. Decide keep or
  delete before launch.

### 5.3 — `gh auth` — restore the account if needed

- **The state:** `gh` has **two accounts on this box**. The active one was
  `liora4u`; it was switched to **`Legendxyz-afk`** on 09-01 to reach the
  private repo, and **that is global state left changed.**
- **Why it is in this file:** a private repo read as **"Repository not found"**
  from both `gh repo view` and `git ls-remote` — **indistinguishable from "it
  does not exist"**. On any GitHub 404 here, check `gh auth status` **before**
  concluding anything about the repo.
- **Owner:** Kabir, if `liora4u` is wanted for other work.

---

# PART III — CRITICAL PATH, PARALLELISM, AND WHAT IS PARKED

## III.1 The critical path

```
0.1 downloads ──► 3.1 B5 Llama-3.2-3B ──► 4.1 freeze ──► 4.2 24h soak ──► 4.4 README ──► 4.5 tag m7
     (hours,            (UNKNOWN —            (instant)      (24 h wall,      (1 sitting,     (1 sitting)
   unattended)       largest risk left)                     box unusable)    quiesced box)
```

**Everything else is off the critical path and runs beside it:**

| runs in parallel with | items |
|---|---|
| the downloads (0.1) | **all of Stage 1** (1.1 → 1.2 → 1.3), **all of Stage 2**, 3.3 fuzz, 3.4 tile sweep, 5.1 rename decision |
| the 24 h soak (4.2) | **4.3 the M7 planning doc** — needs no box, and this is the only 24 h window where nothing else can be measured |
| any blocked window | 0.2 `g++`, 3.3 fuzz, 5.1, 5.2, 5.3 |

**Two observations that decide the calendar:**

1. **The only irreducible wall-clock is the 24 h soak.** Everything else is
   sittings, and sittings are compressible.
2. **Stage 0 decides whether that clock starts today or next week.** Three
   clicks — two downloads, one `apt install`, one approval — gate the longest
   pole and the entire measurement chain. **If nothing else in this file gets
   done today, do Stage 0.**

## III.2 Parked by measurement — do not reopen without a new reason

Each of these was closed by a number, not by an opinion. **They are listed so
they are not re-derived; re-deriving them has already cost sittings.**

| item | why it is parked | the number |
|---|---|---|
| **Q6_K on VNNI** | priced and deprioritized | `ffn_down` is **18 Q4_K + 18 Q6_K across 36 layers** (exactly half); it is 23.2% of prefill ⇒ the Q6_K half is 11.6% ⇒ at 1.09× that saves **~1% of prefill** for a whole kernel + permutation + ladder + mutation-tested grader. **The overflow ladder is 4× TIGHTER and that decides the shape:** a `vpdpbusd` lane must stay inside i16 for `vpackssdw`, and Q6_K quants reach **63** where Q4_K's reach **15**, so a lane holds **4** elements (4×63×127 = 32,004) where Q4_K holds **16** (16×15×127 = 30,480). ⇒ 8 accumulators + 4 packs + 4 `vpmaddwd` = **12 p01 uops = exactly the `_A` shape that MEASURED 1.090×**, not the 9-uop `_B` shape Q4_K got. ⭐ **NAMED FALSIFIER: if a built version reads materially better than 1.09× on Q6_K-only tensors, the overflow analysis is wrong somewhere — re-derive it, do not explain it away.** |
| **Attention's last ~15%** | three items, all small, one needs an API redesign | softmax **59 ms**, the `o[]` read-modify-write **73 ms**, the per-`j` `kv_read` **117 ms** — **1.7% / 1.1% / 0.9% of prefill**. The largest needs a `kv.h` API that must keep working for the Q8 store, which dequantizes into scratch and has no contiguous-stride view to hand back. *A design, not a tweak, for 1.7%.* ⚠ The register-blocking rewrite is **not** the prize: it removes 8 of 17 uops per `j` but forces the four weight broadcasts back inside the `j` loop |
| **Hoisting the activation permute** to `matvec_batch` | priced, declined | ~**3%** |
| **The B=2 / B=4 decode dip** | unexplained, and not on a gate width | stoke is 0.790× / 0.750× of llama there. **The obvious hypothesis was CHECKED and is WRONG** — `POOL_MIN_CHUNKS` is 2, so pooled elementwise already fires at n_seq=2. Live hypothesis: those dispatches parallelize over **sequences**, so at B=2/4 they have 2/4 units for 8 threads while matmuls stay row-parallel over all 8. Probed: `rmsnorm` costs **more absolute time at B=4 (601.2 ms) than at B=16 (547.6 ms)** on a quarter of the work — real, but ~3 points of share, not the 60% relative swing. **B=4 is not a gate width.** *Probe before plan.* |
| **Prefix dedup on S2** | worth nothing there | S2 adopts **0 of 4200** admitted prompt positions — the eight prompts diverge inside the first 16-token block. `--no-dedup` costs nothing on the bars workload. (Dedup is worth **2.284×** on S3, where it is the difference between missing and clearing gate [4].) |
| **`STOKE_PREFILL_CHUNK`** | swept | 1.1% at 256, **worse** at 512 |
| **`STOKE_SCHED_CHUNK`** | swept, at its knee | 64 / 128 / 256 → AGG 16.241 / 17.019 / 17.043. **Halving costs 4.8%, doubling buys 0.06% while doubling stall.** The locked 128 is right |
| **`KERNELS_ROW_TILE`** | re-swept 09-01, **moved 32 → 64** | against VNNI: 16→4487.6 / 32→4325.1 / **64→4255.9** / 128→4356.3 ms. Plateaus at ≥64; 64 is the safer end (`out[]`/`sumf[]` are ROW_TILE×BATCH_TILE stack arrays). **Not** a prefill-for-decode trade — batched decode at 8 seqs went 63.638 → 65.240 tok/s |
| **The uop-count theory of the kernel gap** | dead, twice | ~40% of emitted instructions removed bought **~7%**; ~50% of p01 uops removed bought **12.1%**. *This kernel is bound by nothing that can be counted statically.* |
| **The activation re-read hypothesis** | dead, measured | largest A/B over every batched cell **1.111** against a measured harness floor of 7.0%, and the decisive cell (`ffn_down` N=16, largest tile, deepest into L2) reads **0.996** |

**The one genuinely open perf item is 3.4** — `KERNELS_BATCH_TILE`, never
re-swept against VNNI.

## III.3 Standing constraints that apply to every item above

Carried unchanged from M6/M7 entry. **These are not style; each one has a cost
attached.**

**Building and running**
- **Build from the BASH tool, never PowerShell** — the Makefile uses `mkdir -p`
  and `sh`.
- **Gate sittings FOREGROUND, sliced one unit per invocation**; never write a
  runner that banks only at the end.
- **On any WSL build over `/mnt/m`, check every artifact is an ELF before
  believing a red.** `make -j8` there has emitted corrupt binaries that `file`
  calls `data` and that the runner reports as plain test FAILURES. Prefer `-j4`.
- **`wsl.exe -- bash -c '...'` eats `$var`, not just `$?`.** Put the script in a
  file and run it. ⚠ And a bare `/tmp/x.sh` as a wsl.exe *argument* is
  path-converted by MSYS and fails with 127; inside the `-c '…'` string it is
  left alone.
- ⚠⚠ **NEVER JUDGE A LEG BY ITS EXIT CODE.** `wsl.exe -- bash -c 'bash
  /tmp/x.sh'` on a **missing** script printed `No such file or directory` and
  still reported **exit 0**. Take the verdict from a line the leg PRINTS.

**Measuring**
- **Any perf claim needs a same-sitting, one-binary A/B with a null arm** — and
  **a null arm is only a denominator at its OWN level** (an in-process floor is
  not a between-process floor; N=1 nulls at 30%, N=16 at 1.0%).
- **Never quote a cross-sitting ratio as an attribution.**
- **Report `engine / referent` from ONE sitting, never a bare tok/s.**
- **Counterbalance the order** (ABBA / palindromic), one unit per **process**,
  pinned to P-cores.
- **A disjoint-range result plus a null arm at 1.000× is what makes a noisy box
  readable — medians alone are not.**
- **`bench --matvec-batch` is UNPINNED and un-QoS'd** — it never calls
  `pool_start`, so it does not take the EcoQoS opt-out. Unpinned it read 19.81
  and 33.37 GF/s **on two binaries whose N=1 path is byte-identical.** Pin it,
  and use a same-binary repeat as the control, not the N=1 cell.
- **`objdump` BEFORE designing any probe about instruction selection.** Ten
  minutes of disassembly retired a whole planned probe that would have measured
  a cost nobody pays and read ~1.00 as "the dot is the wall".

**Grading**
- **`stoke.h` IS FROZEN.** Error codes are never renumbered, fields are only
  appended, nothing is removed. A change a caller could notice bumps
  `STOKE_ABI_VERSION`.
- **Deliberately WRONG binaries — never measure with them:**
  `build/win-mut-{preempt,adoptkv,q6k}`, the seven `win-mut-http-*`,
  `win-mut-openai-role`, the five `win-mut-srv-*`, `win-mut-widen`,
  `{win,linux}-mut-fuzz-badbody`, `win-p43-{ceil,a,b}`, `win-mut-q4k-vnni`.
  Each announces itself, **and that banner GATES the leg's verdict** — a build
  that silently dropped its `-D` is a CLEAN binary, and the leg would then
  report "the corpus did not catch it" about a mutant that was never there.
  Also **not wrong, just OLD or different**: `build/win-m5/stoke.exe` (the tag-`m5`
  reference arm) and `build/win-mins0` (the pre-SIMD mins arm).
- ⚠⚠ **A BIT-EXACT A/B CANNOT DETECT THAT ONE ARM NEVER RAN.** Arms required to
  produce identical output are invisible to any test that grades output. **The
  switch must REFUSE something, and that refusal must be tested.** On 09-01 a
  guard silently forced the new VNNI kernel off in every build for twenty
  minutes while every functional test read 180/180 — **what caught it was
  `STOKE_Q4K_KERNEL=bogus` no longer being refused.**
- **A per-TU `#define` default for a build-wide knob is two different knobs**,
  and the TU that does not include it silently gets zero.
- **A knob that no longer compiles is not a knob, it is a claim.**
  `-DSTOKE_MINS_SIMD=0` had not compiled since `fda089a` and nobody knew.

## III.4 Open questions carried for Kabir, consolidated

Everything in this file that is **not** a session's call, in one list:

| # | question | stage | cost of deferring |
|---|---|---|---|
| 1 | the two downloads | 0.1 | ✅ done — B5 and B6 ran on them |
| 2 | `apt install g++` | 0.2 | ✅ done — the Linux C++ TU is green (recorded 2026-09-23) |
| 3 | approve the llama-server @16 sitting | 0.3 | ✅ done — the 09-18 @16/@32/ollama sitting |
| 4 | the prefill policy — chase the aggregate or document the miss | 2.1 | ✅ decided 09-08 — clause 2 re-specified as end-to-end wall |
| 5 | gate [3]'s 250 ms — redesign / re-anchor / documented miss | 2.2 | ✅ decided 09-16 — a documented miss, the redesign is Phase 1.5's |
| 6 | the `history.jsonl` rule — narrow it or add a second `kind` | 2.3 | ✅ settled 09-23 (Opus; Kabir overrules) — narrowed; one row owed at the freeze |
| 6a | the ollama half of clause 2 — accept the documented miss, or re-specify the referent | I.3 | ✅ **decided 2026-09-24 — accepted as a documented miss, referent not re-specified** (`memos/2026-09-24-clause2-ollama-DISPOSITION-KABIR.md`). *Was:* the README states it as a miss until he rules; the stamp needs a disposition. ⚠ IV.9: on S2 the reachable ceiling against a one-slot server is ≈1.93× (derived), just under the bar |
| 6b | **the LICENCE** | 5.0 | ⏳ the only irreversible one — closes the first time an outside PR is merged without a CLA |
| 6c | **the 24 h window** | 4.2 | ✅ **decided 2026-10-12: four 6 h runs** (`design-notes/2026-10-12-m7-soak-4x6h-rule.md`); the four windows are Kabir's to name. *Was:* the critical path: nothing after B7 can start |
| 6d | push `m5-entry` | — | ✅ **pushed 2026-09-24 (Kabir: "push it to my repo")** — the 71 commits 09-06 → 09-24. *Was:* a disk failure loses the whole clause-2 sitting, the freeze and gate [4] |
| 7 | the rename | 5.1 | ✅ **decided 2026-09-24: Lorren** (company: castagents); executes after tag `m7` |
| 8 | fast-forward `main` (177 commits) and whether to keep `m45-aprime` | 5.2 | cannot launch from a default branch at M1 |
| 9 | restore `gh auth` to `liora4u`? | 5.3 | global state left changed |

## III.5 Beyond Phase 1 — pointer, not tasks

Full text in `ROADMAP.md` § "Beyond Phase 1"; reproduced here only so the queue
above has an endpoint that is not the horizon.

- **Phase 1.5** — CUDA (`.cu` behind `extern "C"`, onto the proven scheduler) +
  ARM NEON. **Its planning doc is an M7 deliverable — item 4.3 above.**
- **Phase 2 — the BRIDGE LAYER, the Ollama half of the gap.** *(Re-ordered by
  the Fable 08-02 review, Kabir-approved; this SUPERSEDES the NPU-first line.)*
  Model alias registry + `stoke pull`; idle load/unload under an LRU residency
  budget; per-model chat templates; `/metrics` (M5's driver already computes the
  numbers); memory-elastic serving on M4's variable-capable pool. **NPU /
  heterogeneous execution moves BEHIND this** and is hardware-gated anyway.
- **Phase 3 — the runtime.** Router fold-in + AySen `local()` client in-process.
  *(This is why the concurrency is FOR agents: on a consumer box N=8/16 is one
  user's swarms, tool chains and fan-outs — not eight humans — so shared
  prefixes, bursty arrivals and cheap cancellation are architecture, not server
  QoL.)*
- **Phase 4 (dream)** — AS-core control plane; open-core company.

---

# PART IV — WILL FINISHING THE SEQUENCE REACH THE GOAL?

*Added 2026-09-01 in answer to Kabir's question: "will we be able to reach our
goal after we finish all those tasks? stoke faster than llama on n > 4 and
efficient?" It is a FORECAST. Everything in it is labelled measured, derived or
hypothesised, and the derived parts are claims — this project has already had a
derivation come back wrong by 45% in exactly the direction its own caveat named.*

## IV.1 The blunt answer first

**On "faster than llama at n > 4": already true on prefill at every n, and true
on decode from about n ≈ 10. NOT yet true on decode at n = 8, where it is
parity.**

**On "efficient": no — there is a real 5.5% deficit on total work, it is NOT a
metric artifact, and NOTHING IN THE SEQUENCE CLOSES IT.** The sequence is a
*close-out* plan: it proves, grades, ports and publishes what already exists. It
contains almost no performance work. Finishing every task in Parts II–III leaves
the engine **exactly as fast as it is today**.

⇒ **The sequence reaches the LAUNCH goal. It does not, by itself, reach the
PERFORMANCE goal at N=8.** Whether that matters is decision 2.1.

## IV.2 Where stoke actually stands, n by n

Decode t/s, `llama-batched-bench` vs `stoke batch`, one block, bracketed
(stoke 57.69/57.88 and llama 48.00/47.87, both 0.3% apart):

| N | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| llama | 16.10 | 25.77 | 44.68 | 46.82 | 48.00 |
| stoke | 15.30 | 20.36 | 33.52 | 44.22 | **57.69** |
| **stoke / llama** | 0.950 | **0.790** | **0.750** | 0.944 | **1.202** |
| llama per doubling | — | 1.60× | 1.73× | **1.05×** | **1.02×** |
| stoke per doubling | — | 1.33× | 1.65× | 1.32× | **1.31×** |

**Prefill is 1.03–1.14× of llama at EVERY point on this curve**, n=1 included.

**The decode crossover sits between n=9 and n=10.** *(Derived — a log-linear
interpolation between the two measured points n=8 and n=16, not a measurement.
The curve could be non-monotonic in there; nothing was sampled between 8 and 16.)*

⚠ **Read the n=2 and n=4 cells honestly: stoke is 0.790× and 0.750×, and it is
UNEXPLAINED.** The obvious hypothesis was checked and is wrong
(`POOL_MIN_CHUNKS` is 2, so pooled elementwise already fires at n_seq=2). So
"faster at n>4" is precisely: **behind at 4 · parity at 8 · ahead from ~10 ·
1.202× at 16.** Neither n=2 nor n=4 is a gate width, but neither is a number to
quote past.

## IV.3 Three different questions wear the word "efficient"

| metric | reading | what it means |
|---|---|---|
| **lockstep total wall** (`batch` vs `batched-bench`, same job) | stoke **63.21 s** vs llama **65.32 s** = **1.033× — stoke AHEAD** | the kernels and the batching are *already better* |
| **continuous total wall** (serve vs llama-server, same job) | stoke **65.39 s** vs llama **61.76 s** = **0.945× — stoke BEHIND** | the real, non-artifact deficit |
| **AGGREGATE(M0)** @8 | stoke **16.907** vs llama-server **24.608** = **0.687×** | mostly artifact — see IV.5 |
| **TTFT p50** | stoke **27.6 s** vs llama **41.3 s** = **stoke 1.50× BETTER** | what an agent workload actually feels |

**The decomposition is unusually clean, and it is the single most useful fact in
this file:**

> **In lockstep stoke WINS by 3.3%. Going continuous, llama GAINS 5.5% while
> stoke LOSES 3.4%. That nine-point swing is the entire 0.945×.**

⇒ **stoke's deficit is not in its kernels, not in its arithmetic, and not in its
batching. It is that llama's continuous scheduler extracts a gain from
overlapping prefill with decode and stoke's extracts a penalty.**

## IV.4 Why — measured, not guessed

`STOKE_SCHED_CHUNK` swept on one binary in one block: `cadence` p50 is **FLAT at
~171 ms** across 64/128/256 while `stall` p50 scales **LINEARLY** (738 / 1,287 /
2,450 ms). Cross-checked: `cadence` p50 171 ms against `batch`'s width-8 decode
step of 8/44.22 = 181 ms — the same step in two harnesses, 5% apart.

⇒ **Co-running does not make the decode slower. It makes the step the decode
rides in LONGER, by exactly the prefill added to it. The decode rows WAIT; they
are not harmed.**

That is the mechanism in one sentence: **`sched.c` builds `n_decode + chunk`
rows into a single `SchedStep` and runs them as one forward pass. stoke's
continuous batching SHARES A STEP; it does not OVERLAP RESOURCES.** llama gets
5.5% from the overlap; stoke gets −3.4% from the concatenation.

▶ **HYPOTHESIS, and it is labelled as one:** prefill is compute-bound and decode
is memory-bandwidth-bound, so genuinely overlapping them fills two different
resources and a continuous scheduler *should* beat lockstep. **llama's does,
which is an existence proof that the 5.5% is available on this silicon.** stoke's
co-runs a 128-token chunk with decodes and still loses.
⚠ The llama half of that comparison **crosses harnesses** (`batched-bench` and
the server are different programs), so it is suggestive; **stoke's half is
clean** — same binary, same block, same box.

## IV.5 What metric-chasing alone would buy — and why it still misses

If stoke adopted llama's prefill policy exactly, it would hide the same fraction
of its work outside the measurement window that llama hides:

- stoke today: **60.1 s of span for 65.4 s of work** = 0.919 charged
- llama today: **41.1 s of span for 61.8 s of work** = 0.665 charged
- stoke at llama's ratio: 65.4 × 0.665 = **43.5 s span → 1016/43.5 ≈ 23.4**
- **⇒ 0.687× → ≈0.945× of llama-server — which is exactly the total-work ratio,
  as it must be.**

⚠⚠ **DERIVED, THEREFORE A CLAIM** — [[derived-quantity-is-a-claim]]. It assumes
the policy change costs nothing in throughput, which is the assumption most
likely to be wrong, and it is the same *shape* of reasoning that produced a
45%-wrong answer on this exact metric three days ago. **Do not quote it as a
prediction of a build.**

**But the shape of the conclusion survives the uncertainty, and it is the point:**

> **Even a maximal, TTFT-sacrificing redesign lands at ≈0.945×, NOT ≥1.0×.
> Clause 2 cannot be met by chasing the metric. It can only be met by closing
> the 5.5% of real work — or at N=16, where the physics is different.**

That is the strongest argument against decision 2.1's "chase it" branch: **it
costs the project's best property and still does not clear the bar.**

## IV.6 So what WOULD reach it

| # | lever | worth | state |
|---|---|---|---|
| 1 | **The scheduler overlap redesign** — stop putting the prefill chunk in the same step the decode rides in; overlap the compute-bound and bandwidth-bound work instead of concatenating it | **up to ~9 points** (llama's +5.5% and stoke's −3.4%), taking total wall from 0.945× to ~1.03× — i.e. **stoke's lockstep lead, preserved through continuous batching** | **UNDESIGNED.** The largest real lever left in the project |
| 2 | **N=16 rather than N=8** | the curves have opposite slopes; at 16 stoke's decode is already 1.202× against a lockstep referent | **UNMEASURED against a live server — this is task 1.1** |
| 3 | metric disposition (2.1) | ~26 points of the *number*, **0 points of real efficiency** | Kabir's call |
| 4 | `KERNELS_BATCH_TILE` re-sweep (3.4) | low single digits on prefill, which is already ahead | cheap, optional |

⭐ **Lever 1 and gate item [3] ARE THE SAME WORK.** [3]'s <250 ms ITL is missed
at **every** chunk width (738 ms even at the smallest legal 64) precisely
*because* a whole chunk sits in one step, and `cadence` p50 ~171 ms is inside
250 ms at every width. **One redesign addresses the 5.5%, gate [3], and the
"efficient" question together.** Nothing else in this project has that property,
and it is why 2.2 should not be dispositioned as "documented miss" before lever
1 is at least *designed*.

⚠ **It is not free money.** The chunk knob is already at its knee in both
directions (halving costs 4.8%, doubling buys 0.06%), so the gain has to come
from restructuring the step, not from tuning it — and it trades against the
amortisation E3 measured **saturating by ~23 rows**.

## IV.7 The caveat that could reverse the headline

**Nobody has measured llama-SERVER at N=16.** The 1.202× is against
`llama-batched-bench`, which is **lockstep**.

⚠⚠ **And the @8 lesson says lockstep FLATTERS stoke.** On 2026-09-01 a
derivation from lockstep `batched-bench` put llama-server at 16.97 and concluded
parity; the server then measured **24.608 — wrong by 45%, in exactly the
direction the caveat named**, because a real continuous server overlaps better
than lockstep. **If the same correction has any force at N=16, llama-server
there could be materially above 48 t/s and stoke's 1.202× lead could shrink or
vanish.**

⇒ **The single decisive unknown in this entire document is one cell**, and it is
task 1.1. Until it is run, "stoke beats llama at high concurrency" is
**established against a lockstep referent and unproven against the thing we are
graded on.**

## IV.8 Verdict, with confidence

| claim | verdict | confidence |
|---|---|---|
| stoke's prefill beats llama's at every n∈[1,16] | **YES** | **high** — measured, bracketed, disjoint ranges, multiple sittings |
| stoke's decode beats llama's at n=16 *against lockstep* | **YES** | **high** — measured, bracketed 0.3% |
| stoke's decode beats llama's at n=16 *against a live server* | **UNKNOWN** | ⚠ **untested — task 1.1, and the @8 precedent cuts against us** |
| stoke's decode beats llama's at n=8 | **NO — parity** (0.944–0.992×) | high |
| llama saturates after B=4 and stoke does not | **YES** | **high** — 1.07× vs 1.72× over 4→16, and M0's own 46-day-old baseline shows the same collapse |
| stoke is more *efficient* (total work, continuous) | **NO — 0.945×**, a real 5.5% | high |
| stoke is more efficient in *lockstep* | **YES — 1.033×** | high |
| stoke is more *responsive* | **YES — TTFT 1.50× better**, first token at 5.3 s vs 20.7 s | high |
| **finishing the Part II sequence closes the 5.5%** | **NO** | **high — nothing in it is that work** |
| **finishing the sequence closes clause 2 at N=8** | **NO**, and re-anchoring (1.2) makes the gap officially larger | high |
| **finishing the sequence closes clause 2 at N=16** | **PLAUSIBLE, and task 1.1 decides it** | ⚠ unknown |
| lever 1 would close the 5.5% | **PLAUSIBLE — llama proves the 5.5% exists on this silicon** | ⚠ hypothesis, undesigned |

**Summary in one line: the sequence ships a defensible Phase 1 and proves the
thesis at high concurrency; it does not make the engine faster, and the one
piece of work that would — the scheduler overlap redesign — is not in it, is
undesigned, and is the same work gate [3] needs.**

## IV.9 2026-09-23 — why the margin is ~1.3× and not 2×, and what N=128 would show

*Kabir: "only 1.28? what about n = 128? why isnt it double". DERIVED from the
09-18 sitting's banked walls (`docs/BENCHMARKS.md` §1), not a new measurement —
[[derived-quantity-is-a-claim]] applies. The @8 row is left out on purpose: it
is a different sitting in the slow box state, and absolutes do not cross
sittings (standing rule 5).*

- **The job is mostly prompt.** S2 is ~525 prompt tokens + 128 generated per
  client, so ~80% of the forward work is prefill — which batching cannot make
  cheaper: every engine processes every prompt token.
- **By N=32 stoke runs the whole job at its prefill ceiling.** Work rate =
  N × 653 tokens ÷ wall: **stoke 74.3 → 78.1 t/s** at N = 16 → 32;
  **llama-server 62.2 → 60.9**; the same-block `llama-bench pp512` referent
  **~70–72**. stoke's 78.1 ≈ **1.09×** the referent — its own prefill-kernel
  lead (09-01: 1.089×). The decode rows ride inside the prefill chunk's step
  (IV.4), so at N=32 they are processed at prefill efficiency: that is the
  mechanism behind "the margin grows with N", and it is also where it ends.
  llama-server runs at ~85% of llama.cpp's own kernels: its batched decode
  stops scaling past ~8 (IV.2: 1.02–1.05× per doubling against stoke's 1.31×)
  and its 2,048-token prompt batches stop every stream for 29–30 s.
- ⇒ **1.28× ≈ 1.09 (kernels) × ~1.17 (what llama-server loses).** More clients
  cannot lift stoke past its ceiling; **2× over llama-server would need
  llama-server to waste half the CPU.** The kernel levers priced ~null on
  09-16, so the ceiling does not move cheaply either.
- **N=128 cannot be measured on this box as built.** 128 × ~650-token contexts
  need ~26 GB of stoke's f32 KV (the @32 arm used 6.5 GB) and ~13 GB of
  llama's f16 KV next to a 2.5 GB model on a 15.6 GB box, and the frozen
  server accepts 48 connections (`MAX_CONNS`). It needs q8 KV on both arms and
  a `MAX_CONNS` change — a `src/` change, so after tag `m7`. Expected from the
  above: **~1.3×**, higher only if llama-server degrades further.
- **The ollama bar sits just above what this workload allows.** ollama is
  serial at 16.2 s per request; stoke at its ceiling needs 653/78.1 ≈ 8.4 s ⇒
  **≈1.93× is the most S2 permits** against a one-slot server on this box (1.84×
  measured @16). Clearing ≥2× needs ~3% more prefill speed (80.6 t/s), or a workload with a
  smaller prompt share. Input to III.4 6a.
- **Where more than 2× is real: work stoke does not do.** Shared prompts —
  agents with a common system prompt — are what prefix reuse removes: measured
  2.3–4.2× within stoke on S3 fan-out (M5 gate 4; PHASE-1.5-PLAN §6), never yet
  against llama-server, and S2 salts every prompt so none is shared.
  Decode-heavy work (short prompts, long answers) widens the gap against a
  serial server the same way.

---

# PART V — WHO WOULD USE THIS, GIVEN llama.cpp EXISTS

*Added 2026-09-01 in answer to Kabir's question: "will ppl use stoke if llama is
still good and better in some places?" ⚠ **This part is JUDGMENT, not
measurement** — unlike Parts I–IV, almost nothing here is a number. It is here
because it decides three things already in this file: what 4.4's README leads
with, whether 5.1's rename is worth doing, and — unexpectedly — **decision 2.1.***

## V.1 The blunt answer

**Not as a llama.cpp replacement, and it should not try to be.** A 1.202× decode
win at n=16, on one CPU architecture, with one verified model family, does not
move anyone off llama.cpp. That is not a close call, and the reason is
instructive: **llama.cpp did not win on speed either.** It won on running
everywhere, on GGUF, and on being embeddable enough that Ollama, LM Studio,
koboldcpp, jan and llamafile could all be built on top of it. Performance parity
±20% has never been why anyone adopts an inference engine.

⇒ **Any pitch of the form "faster than llama.cpp" loses, and deserves to.**

## V.2 What llama.cpp is better at, conceded explicitly

This list should appear in the README, in stoke's own words, before anyone else
writes it:

- **Every platform stoke does not have**: CUDA, Metal, ROCm, Vulkan, ARM.
  stoke is x86-64 AVX2/AVX-VNNI CPU **only**.
- **Every model architecture.** stoke has verified **one family** (Qwen3-4B)
  until B5 lands.
- Quantization variety, tooling, conversion scripts, the GGUF ecosystem it owns.
- Four years of maturity, hundreds of contributors, a known bug surface.
- **Total throughput on a single stream and on small batches** — and, today,
  **total wall on the continuous 8-client job by 5.5%**.

**A project that claims to beat llama.cpp generally gets dismissed by one
benchmark. A project that concedes the matrix and names one cell it owns is
credible.** Concede loudly.

## V.3 The one cell stoke actually owns

Narrow, real, and genuinely underserved — it is the founding premise, and
`docs/vllm-and-the-serving-landscape.md` is where it was argued:

> **x86-64 CPU · ONE user issuing MANY concurrent requests · where FIRST-TOKEN
> latency matters · embedded in-process rather than behind an HTTP daemon.**

That is an **agent** workload — swarms, tool chains, fan-outs — not eight humans.
llama.cpp and Ollama are tuned for one human chatting; vLLM is tuned for many
humans on a GPU. **Nobody is tuned for one machine running sixteen agents.**

What stoke has there, measured:

| property | stoke | llama |
|---|---|---|
| decode scaling B=4 → 16 | **1.72×** | **1.07×** (a wall) |
| first token on an 8-client job | **5.3 s** | 20.7 s |
| TTFT p50 | **27.6 s** | 41.3 s (**1.50× better**) |
| prefix dedup on the serving path | **2.284×** on S3 fanout | has its own (`--slot-prompt-similarity`, on by default) |
| embeddable | **frozen `stoke.h` v0**, `libstoke.a`, 1.38 MB static, no deps | C API exists; ABI is not frozen and integrators pin commits |

⚠ **The saturation claim rests on two legs and one is weak.** It reproduces in
today's lockstep `batched-bench` AND in M0's own llama-**server** S2 curve
(33.639 @4 → 17.664 @8 → 14.163 @16) — but **that M0 curve was taken on a box
state we have since proven was 39% different**, and @16 has never been
re-measured. **Task 1.1 is not just the gate's decisive cell; it is the PRODUCT
CLAIM's decisive cell.** Until it runs, the central marketing sentence is
unproven.

## V.4 The inversion — and it settles decision 2.1

**If the product is "does not fall over when your agent fans out", then TTFT and
the shape of the scaling curve ARE the product, and aggregate throughput is the
thing stoke should not lead with** — it is llama's home turf and stoke is 5.5%
behind on it.

⇒ **Decision 2.1 answers itself.** Batching prefill to raise AGGREGATE(M0) would
trade **the single best property stoke has** (first token at 5.3 s against 20.7 s)
for a better score on **the one axis where it cannot win anyway** — Part IV.5
derives the maximal payoff at ≈0.945×, still short of ≥1.0×.

**Selling the thing you are worse at, by destroying the thing you are better at,
is the whole decision in one sentence.** Record it that way.

## V.5 The honest adoption answer: there is a first user, and it is not "the market"

`ROADMAP.md` already says the concurrency is **FOR agents** — **AySen `local()`
at Phase 3**. That is the strategically important fact in this whole section:

> **stoke has an anchor tenant by construction.** The question "will people use
> it" is the wrong question for a 47-day-old engine. The right one is "does it
> have a first user who needs exactly this, and would that user pick it over
> llama.cpp?" — and the answer is yes, because that user is
> [[aysen-language-project]] and needs an **in-process, embeddable, cancel-cheap,
> many-concurrent-request** engine, which is a shape llama.cpp does not offer.

An engine that is genuinely the best choice for **one** demanding consumer is a
real project. An engine that is 20% faster for **nobody in particular** is a
benchmark.

⚠⚠ **AMENDED 2026-09-01 — KABIR: "aysen is gonna be secondary, stoke is gonna be
first." THIS ARGUMENT IS NOW DEFERRED, AND THE PARAGRAPH ABOVE MUST NOT BE
QUOTED WITHOUT THIS ONE.** The anchor-tenant reasoning depended on AySen being
built alongside; with AySen secondary, **stoke has no captive first user and must
find real external ones instead.** That is the right call for capacity reasons —
and it aligns the project with Paths 1 and 2, the only two that are solo-viable
and neither of which needs AySen — but it **promotes "nobody installs it" from a
background worry to THE primary commercial risk.** Full disposition:
`COMPANY.md` §0.1 and §4.2.

## V.6 What actually decides broader adoption — in order

| # | gate on adoption | status | why it dominates speed |
|---|---|---|---|
| 1 | **More than one model architecture** | **B5, blocked on a download** | Below roughly three or four families, adoption is ~zero *regardless of speed*. This is the single largest blocker and it is a rung, not a research problem |
| 2 | **Proof the concurrency lead is real against a live server** | **task 1.1, unmeasured** | The whole pitch rests on it |
| 3 | **Phase 2, the bridge layer** | planned, post-Phase-1 | `stoke pull`, alias registry, idle load/unload, per-model templates, `/metrics`. **The Fable 08-02 review re-ordered this AHEAD of NPU work precisely because the engine alone is not adoptable** — this is the Ollama half of the gap and it is what makes stoke usable by someone who is not Kabir |
| 4 | **A name people can find** | **5.1 ✅ Lorren (2026-09-24)** | was: PyPI taken, StanfordPL/stoke is a known superoptimizer, Stoke Space dominates search |
| 5 | **A README that concedes V.2 and claims only V.3** | **4.4** | Overclaiming is how a good engine gets dismissed in one thread |
| 6 | closing the 5.5% | Part IV lever 1, undesigned | removes the one number that kills the pitch on sight |

## V.7 The risk, stated plainly

**The lead is a lead, not a moat.** llama.cpp is actively developed by people who
are very good at this. Nothing in stoke's continuous-batching advantage is
patentable, secret, or hard to copy once someone decides to — it is a scheduler
design, and the saturation stoke exploits is a fixable property of somebody
else's code, one PR away from disappearing.

⇒ **Two consequences worth acting on:** (a) the durable asset is the
**embeddable frozen ABI plus the agent-runtime integration**, not the tok/s —
a scheduler win can be copied, a first-class in-process API with a real consumer
on top is a position; (b) **there is a clock on the benchmark claims**, which is
an argument for finishing Phase 1 rather than extending it.

---

# PART VI — THE ONE-PARAGRAPH ANSWER

**Seven of eight milestones are closed in 47 days, 420 commits and 36k lines.
Five of Phase 1's eight goal clauses are met, one is unstarted and cheap, and
the two that are open are decisions rather than engineering unknowns. The
kernels are finished and ahead: stoke's prefill is 1.13× of llama.cpp's, its
decode is at parity at N=8 and 1.202× at N=16, and llama.cpp's decode saturates
after B=4 while stoke's keeps climbing — which is the project's founding thesis,
now measured rather than argued. The one number that still misses is a serving
aggregate whose bar is 46 days old and provably 0.72× of what the referent does
on this box today, and whose remaining 31% is almost entirely *where the clock
starts*: the engine does the same work in 0.945× the time and delivers first
tokens 1.50× sooner. What is left is one blocked model port, a fuzz run, a 24 h
soak, a write-up — and three decisions that are Kabir's, not the code's. And one
thing that is NOT on that list and should be understood before it is chosen
against: the sequence ships Phase 1 but makes the engine no faster, and the real
5.5% deficit — the whole of it in the continuous scheduler, none of it in the
kernels — needs an overlap redesign that is undesigned, is the same work gate
[3] needs, and is the largest lever this project has left (Part IV).**

---

*Companion documents: `ROADMAP.md` (the map, authoritative on every gate) ·
`docs/superpowers/design-notes/2026-08-30-m7-entry.md` (M7's rung order and its
open items) · `.superpowers/sdd/progress.md` (the exact in-flight resume point,
final NEXT block) · `../MEMORY.md` (master record and history).*
