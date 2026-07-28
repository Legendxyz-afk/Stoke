# FOR FABLE — 2026-07-28 — define M2's substitute gate    [ANSWERED 2026-07-28: Fable ruled — see `2026-07-28-m2-substitute-gate-RULING-FABLE.md`, which is the authority; this file remains as the question of record]

**How to use this file:** Kabir opens a Fable session and points it here.
Fable delivers the five items in §3. Opus implements them. Opus does **not**
decide any of them — under the 2026-07-28 routing, major decisions go to
Fable, and Kabir launches that session himself.

---

## 1. WHAT IS ALREADY DECIDED — not Fable's to revisit

M2's gate was greedy 64-id parity of stoke against `llama-server` (the
"oracle"), 50 prompts, ε=1e-3 nats, ≤2 waivers. The full survey completed
2026-07-28 and **19 of 50 prompts diverge** (details in §2).

**Kabir took memo option 3** (`2026-07-20-m2-oracle-near-tie-policy.md`,
DECISION section): M2's token gate is re-scoped to **C-vs-ref** — stoke's C
forward pass against `tests/tooling/ref_forward.py`, the numpy f32
reference, both pure f32 — and **oracle token-parity is deferred to M3**,
where LOCKED decision #2's Q8_0 activation quantization + i32 integer core
put stoke in the oracle's own numeric class and the forward note already
mandates re-running the 50-prompt oracle gate. Kabir also chose the
**shape**: a **stratified subset + the existing 36-layer L2 ladder**, not
all 50 prompts.

M3 additionally inherits the **aligner fix** as a prerequisite: on 4 of 50
prompts (p26, p27, p31, p34) `align_tokens_to_entries` fatals because a
ground-truth id never resurfaces as a `top_logprobs` candidate; byte lengths
must instead come from the GGUF vocab via `gguf_reader.py`.

## 2. THE EVIDENCE FABLE IS RULING ON

- **19 diverged / 31 exact.** 2,398 of 3,200 ids (**74.9%**) match before
  first divergence. Median first-divergence position 16 of 64.
- **18 of 18 classifiable divergences are a clean top-2 swap** — stoke picks
  exactly the oracle's rank-2 candidate and the oracle picks stoke's; same
  candidate set, ranking permuted. Oracle top1−top2 gaps: min **0.000950**,
  median **0.067308**, max **0.203868**. Only one under ε. Seven ≥0.1 nats.
- The 19th (p28) is **unclassifiable, not a counterexample**: its divergence
  falls inside a coalesced `completion_probabilities` entry, and such an
  entry carries only the LAST swallowed step's distribution — verified across
  all 2,867 entries (`entry.id` == token at last covered position in 2,866;
  `top_logprobs[0].id == entry.id` in 2,867/2,867).
- **Every tier diverges**: prose 6/15, code 2/10, multilingual 3/10, math
  1/5, long-context 3/5, adversarial 4/5.
- **Cause, established:** the oracle quantizes activations to Q8 blocks for
  integer dot products; M2 stoke is pure f32. Two arithmetic classes ranking
  near-ties differently — not a bug on either side. stoke's L2 ladder puts
  all 36 layers within 1e-3 of the f32 reference (worst 6.1e-4).
- **First C-vs-ref data point:** p48 — a prompt that diverged from the
  oracle — is **C-vs-ref EXACT, 64/64**.

## 3. WHAT FABLE MUST DECIDE

Fixed **before adjudication**, because a bar written after seeing results is
exactly the gate-loosening the governance forbids:

1. **SUBSET** — exact prompt ids, and the rule behind them.
2. **PASS CRITERION** — mechanically checkable; state what FAILS.
3. **IF A FLIP OCCURS** — the pre-committed handling, what evidence is
   required, who adjudicates.
4. **PREREQUISITE TOOLING** — stated as concrete requirements.
5. **WHAT THIS GATE DOES AND DOES NOT PROVE** — one paragraph for the M2
   close-out record.

**(2) is the hard one.** "Both pure f32 ⇒ token-exact" is an **expectation,
not a measured fact**: numpy's BLAS summation order differs from stoke's
scalar C loops, and the L2 ladder already shows f32 non-associativity drift
to 6.1e-4 in hidden state. Demand 100% and a ~1e-5 tie-flip stalls M2 on
something that proves nothing; permit flips without saying **exactly** what
excuses one, in advance, and the gate cannot fail.

## 4. PROBED CONSTANTS — measured this session, do not re-estimate

- **ref cost 13.8 s/position** (W1 benchmark); one live prompt measured
  **12.32 s/position**. Cost/prompt = (tokens + 64) × 13.8 s. ref
  re-dequantizes every weight per matvec by design — caching all 36 layers
  would need 14–16 GB on a 16 GB box.
- **Parallelism is RAM-bound at 3 ref processes.** One measured at **2,292 MB**
  private; three running leaves ~3.2 GB free.
- **stoke's 64 ids for all 50 prompts are already cached and hash-verified**
  (`stoke.exe` sha256 `d2b17c53…` == current binary). The run is **ref-only**.
  Reserve prompts **r00–r09 are NOT cached** — any reserve replacement needs a
  stoke generation as well.
- Gate file `Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf`; `gguf_reader.py`
  dequantizes F32 + Q8_0 only.
- **HARD CONSTRAINT: `stoke run` accepts NO token-id input.** Its CLI is
  `[--raw] [-n N] [-p "text" | -f file] [--dump dir] [--probe-pos N]` — read
  from the arg parser in `src/main.c`. Any design that needs stoke fed a raw
  id prefix is not implementable without rebuilding `stoke.exe`, which
  changes its sha256 and invalidates the cached id streams that this
  milestone's whole evidentiary chain is pinned to.
- **Both sides emit probes identically:** `probe pos=N: id=%u logit=%.6f`,
  5 lines to stderr, rank-0 first, ties broken by the LOWER id. stoke already
  had `--probe-pos`; the matching one was added to `ref_forward.py`
  (selftest green). **Either side FATALS if the position was never generated**
  (early EOS).
- **EOS semantics match:** 151643 is Qwen3's BOS, not EOS
  (`eos_token_id` = **151645**). stoke and ref read the same key and their
  generate loops are structurally identical, so they stop on the same id at
  the same index.
- `l2_diff.py` is committed: per-layer max/mean abs diff over L00..L35 against
  a threshold, and it separately flags a **STEP** (any layer jumping >10× the
  running trend), on the reasoning that smooth accumulation is f32
  non-associativity while a step is a structural single-op bug, and a bare
  max-abs-diff conflates the two.
- **Corpus token counts** — prose p00 23, p01 25, p02 23, p03 26, p04 18,
  p05 21, p06 24, p07 27, p08 21, p09 21, p10 24, p11 22, p12 22, p13 26,
  p14 22; code p15 40, p16 22, p17 47, p18 49, p19 41, p20 64, p21 63,
  p22 88, p23 39, p24 51; multilingual p25 16, p26 19, p27 26, p28 82,
  p29 24, p30 53, p31 25, p32 104, p33 36, p34 34; math p35 53, p36 45,
  p37 26, p38 36, p39 31; long-context p40 490, p41 477, p42 462, p43 471,
  p44 447; adversarial p45 14, p46 22, p47 26, p48 2, p49 18.
- **The 19 diverging:** p01 p02 p05 p08 p09 p12 p20 p24 p25 p28 p29 p39 p41
  p43 p44 p46 p47 p48 p49. Long-context prompts cost **~2 h each**.

## 5. PRIOR DRAFT — INPUT ONLY, NOT AUTHORITY

This decision was dispatched once and, through an Opus dispatch error
(`model: "fable"` was silently ignored on a `general-purpose` agent), was
answered by **Sonnet**, not Fable. Reproduced so the work isn't wasted.
**Fable may accept, modify, or replace any of it.**

- **Subset, 24:** `p01,p02,p05,p08,p09,p11,p12,p16,p20,p24,p25,p26,p28,p29,p37,p39,p41,p43,p44,p45,p46,p47,p48,p49`
  — all 19 diverging + one oracle-exact control per tier (p11 prose, p16 code,
  p26 multilingual, p37 math, p45 adversarial), **no long-context control**
  because 3 long jobs fill exactly 3 slots in one round, so a 4th ~120-min job
  would force a second round for one data point. Cost independently re-derived
  and exact: 3,503 positions, **13.4 h serial, ~4.5 h wall at 3 jobs**,
  makespan floor p41 at 124 min. It chose p11 over the cheaper p04 because
  p04's C-vs-ref outcome is already forced by transitivity — W1 recorded ref
  reproducing the oracle 64/64 on p04, and p04 is oracle-exact for stoke.
  **That transitivity claim was independently verified and holds.**
- **Pass criterion:** all subset prompts token-exact **AND** the L2 ladder
  re-run as part of this evidence — 36 layers ≤1e-3, and no STEP flagged.
- **Waiver, ≤2, reserve r00–r09, replacement must itself pass.** Eligible iff
  (2) top-2 are the same pair swapped; (3) BOTH sides' top1−top2 gap < ε=1e-3;
  (4) both sides' top-5 have identical ids in identical rank order at ranks
  3–5. Probe unavailable on either side ⇒ FAIL, never a free pass.
- **Fails on:** >2 diverging; any flip missing a condition; any tooling
  error/timeout; any ladder layer >1e-3 or a STEP; any reserve replacement
  itself failing.

## 6. STATE OF PLAY

The 24-prompt measurement run is **already executing** under the draft subset
(scheduled task `stoke-m2-ref-gate`, `--jobs 3`, ~4.5 h). It **only generates
ids and records comparisons — it performs no adjudication**, so no bar has
been applied and none will be until Fable rules. Every prompt result is cached
atomically, so any prompt Fable keeps is already paid for and any it drops
costs only box time. If Fable changes the subset, Opus runs the difference; if
it changes the criterion, Opus applies Fable's.

**No results have been shown to any decision-maker, and none will be before
the criterion is fixed.**
