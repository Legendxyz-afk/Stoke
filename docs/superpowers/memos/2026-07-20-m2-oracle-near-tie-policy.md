# MEMO — 2026-07-20 — M2: oracle near-tie policy for the recorded parity run    [BLOCKING: the recorded gate run + any protocol change. NOT blocking: W3, W4 harness construction + dev-mode measurement]

**WHAT:** The M2 parity oracle (llama-server) computes in a different
numeric class than stoke's pure-f32 scalar pass, and at least one
characterized divergence exists that the §8 waiver protocol cannot absorb.
Decision needed on how the recorded 50-prompt gate run treats this class.
The KV-cache hypothesis from 07-20 is REFUTED by experiment; the gate's
oracle invocation stays DEFAULT (no `-ctk/-ctv` flags) — that question is
closed without you; this memo is about the divergence class it exposed.

**EVIDENCE (all mechanically recorded; artifacts committed under
`tests/tooling/validation/`, details in the session ledger):**

- f32-KV experiment: server rerun with `-ctk f32 -ctv f32` on all 3
  validation prompts, twice on v1 → deterministic, and **byte-identical
  to the default-KV captures everywhere**. KV type contributes zero
  token difference; the numeric gap is the oracle's activation
  quantization (it quantizes activations to Q8 blocks for integer dot
  products; not switchable by any flag).
- The one divergence: v1 ("Explain how photosynthesis works in one
  sentence."), position 32 of 64 — a genuine 3-way near-tie ("one/two/
  three sentences" continuation). Teacher-forcing the numpy ref on the
  oracle's own 42-token prefix reproduces it exactly:

  | rank | oracle id (logprob) | ref id (logit) |
  |---|---|---|
  | 1 | 2326 (−1.4927) | 825 (22.9451) |
  | 2 | 1378 (−1.5191) | 2326 (22.8749) |
  | 3 | 825 (−1.5204) | 1378 (22.8518) |
  | 4 | 264 (−1.9065) | 264 (22.4663) |
  | 5 | 4236 (−3.0839) | 4236 (21.3111) |

  Same five candidates, tail order identical, top-3 cluster permuted.
  Gaps(top1−top2): oracle **0.0264 nats**, ref **0.0702 nats** — both
  26–70× larger than the waiver ε (1e-3), and the top-2 pairs are not a
  clean swap, so §8 conditions 1 AND 2 both fail. This is not a bug on
  either side: v2 (code) and v3 (multilingual) reproduce the oracle
  **64/64 token-exact**, and every layer-level check agrees; the two
  implementations legitimately rank a ~2%-relative tie differently.
- Rate: 1 flip in 192 validation tokens (~0.5%/token as a single
  observation). Over the 50×64 = 3,200-token gate that extrapolates to
  anywhere from 0 to ~15 affected prompts — too uncertain to decide on;
  see the plan below, which measures it before you must choose.

**ALREADY IN MOTION (reversible; proceeding unless you object):** W3
(model.c) is dispatched — unaffected by this question. W4's harness will
be built and run in **dev mode only** across all 50 prompts to measure
the REAL affected-prompt count; you get that number appended here before
any recorded run. The recorded run itself waits for your call.

**OPTIONS for the recorded run:**
1. **Strict as written.** Any >ε divergence = gate fail → I compile the
   per-hit evidence package (both sides' top-5, L2 ladder proof that
   C == ref at every checkpoint for that prompt) and return to you.
   Zero protocol change; costs one extra round-trip if hits > 0.
2. **Bounded near-tie recognition (protocol change, M2 only):** a
   divergence counts as waiver-class iff (a) both sides' top-3 SETS are
   identical, (b) both sides' top1–top2 gaps < 0.1 nats, (c) the prompt
   is replaced from the reserve exactly like a §8 waiver, sharing the
   same ≤2 budget, every use fully recorded. Dies at M2 close.
3. **Re-scope M2's token gate to C-vs-ref (both pure f32, 50/50
   token-exact) + defer oracle token-parity to M3**, where the integer
   core puts stoke in the oracle's numeric class and the forward note
   already mandates a full parity re-pass. Keeps an absolute bar now;
   trades away oracle-independence for one milestone.

**RECOMMENDATION:** hold the recorded run until the dev-mode count is
appended here (hours, not days). Then: count = 0 → option 1 and done;
count 1–2 → option 2 (the reserve absorbs it, evidence recorded);
count ≥3 → option 3 (the class is systematic at M2's numeric distance;
grinding 50 prompts against a different-class oracle proves nothing the
M3 re-pass won't prove properly).

**IF DEFERRED:** M2 close waits on the recorded run; W3/W4 construction
and everything else proceeds. M3 is unaffected either way — its integer
core re-runs this exact gate in the oracle's own numeric class.

---

# APPENDED 2026-07-28 — THE DEV-MODE COUNT. **19 of 50.**

Source: the `--survey` run over all 50 corpus prompts (scheduled task
`stoke-m2-survey`, relaunched 10:17 after the harness timeout fix
`5e0d484`; generate phase 50/50 complete in 239.9 min, **zero tooling
errors** — the four `p40–p43` ERRORs of 07-27 do not recur). Numbers below
are computed from the survey's own per-prompt cache using the harness's own
`compare_ids` / `align_tokens_to_entries` / `top5_for_position`, so this
analysis and the survey report cannot disagree about alignment or
first-diff position. Cross-check: the three prompts measured independently
on 07-28 (p46, p48, p49) reproduce their recorded gaps to the last digit.

**COUNT = 19 diverged / 31 exact.** This is above the top of this memo's
own 0–15 extrapolation, and the decision rule above maps **≥3 → option 3**.

**Token-level:** 2,398 of 3,200 compared ids match before first divergence
(**74.9%**). Median first-divergence position is 16 of 64; 5 of the 19
diverge in the first 5 positions, 6 of them only after position 32.

**THE MECHANISM IS UNIFORM — 18 of 18 classifiable hits are the same
shape.** At every divergence whose oracle distribution is directly
readable, stoke picks **exactly the oracle's rank-2 candidate** and the
oracle picks stoke's — a clean top-2 swap, same candidate set, ranking
permuted. Not one hit shows stoke choosing a token the oracle ranked low.

| prompt | tier | pos | oracle id | stoke id | oracle top1−top2 gap (nats) | class |
|---|---|---|---|---|---|---|
| p01 | prose | 26 | 6821 | 66192 | 0.131350 | top-2 swap |
| p02 | prose | 16 | 714 | 438 | 0.000950 | top-2 swap |
| p05 | prose | 51 | 56981 | 4680 | 0.055361 | top-2 swap |
| p08 | prose | 56 | 264 | 1293 | 0.128040 | top-2 swap |
| p09 | prose | 4 | 83752 | 1033 | 0.101929 | top-2 swap |
| p12 | prose | 7 | 21571 | 65 | 0.004536 | top-2 swap |
| p20 | code | 25 | 429 | 323 | 0.006651 | top-2 swap |
| p24 | code | 13 | 52847 | 57203 | 0.001179 | top-2 swap |
| p25 | multilingual | 54 | 99652 | 101912 | 0.033733 | top-2 swap |
| p28 | multilingual | 23 | 14925 | 83636 | (1.389990) | **UNCLASSIFIABLE — see below** |
| p29 | multilingual | 42 | 23224 | 124074 | 0.111189 | top-2 swap |
| p39 | math | 0 | 2014 | 6771 | 0.003998 | top-2 swap |
| p41 | long-context | 2 | 16141 | 3966 | 0.040001 | top-2 swap |
| p43 | long-context | 36 | 14250 | 47609 | 0.077057 | top-2 swap |
| p44 | long-context | 6 | 323 | 25352 | 0.020979 | top-2 swap |
| p46 | adversarial | 0 | 1283 | 1304 | 0.067308 | top-2 swap |
| p47 | adversarial | 36 | 701 | 8 | 0.130280 | top-2 swap |
| p48 | adversarial | 2 | 1172 | 1879 | 0.157925 | top-2 swap |
| p49 | adversarial | 15 | 6932 | 279 | 0.203868 | top-2 swap |

Gap distribution over the 18: min **0.000950**, median **0.067308**, max
**0.203868**. Under ε=1e-3: **1** (p02). Under 0.1 nats: 11. At or above
0.1 nats: 7.

**p28 IS NOT A COUNTEREXAMPLE — IT IS A READ-OUT LIMIT, AND I CHECKED
RATHER THAN ASSUMED.** Its headline "1.39 nats, stoke's pick not even in
the oracle's top-5" is an artifact. p28 is Devanagari; its divergence at
position 23 falls **inside a coalesced `completion_probabilities` entry**
(entry 17 covers positions 23–24), and a coalesced entry carries only the
**last** swallowed step's distribution. Verified mechanically over all
2,867 entries in the 50 responses: `entry.id` equals the token at the last
covered position in **2,866**, and `top_logprobs[0].id == entry.id` in
**2,867 of 2,867**. (The single exception is p46's terminal EOS 151643,
which contributes zero bytes and so merges into its predecessor.) So the
top-5 shown at p28 position 23 belongs to position 24, and no gap or rank
read off it means anything. The oracle's response format exposes no
finer-grained distribution — this is a limit of the oracle, not a stoke
result. p28 is the only one of the 19 so affected; the other 18 sit at
1:1 entries where the read-out is exact.

**DIVERGENCE IS PRESENT IN EVERY TIER — the 07-27 hope that this was an
adversarial-tier artifact is dead.** prose 6/15 · code 2/10 ·
multilingual 3/10 · math 1/5 · long-context 3/5 · adversarial 4/5.

**SEPARATE LIVE TOOLING GAP FOUND WHILE DOING THIS COUNT (not a math
result, records the W4 flag as now-materialized).** On **4 of 50** prompts
— p26, p27, p31, p34 — `align_tokens_to_entries` **fatals**: a
ground-truth token id never resurfaces as a `top_logprobs` candidate
anywhere in its own response, so its byte length cannot be resolved and
the harness refuses to guess (exactly as W4's report flagged, and it is
right to refuse). All four are EXACT prompts, so the census is unaffected
and no number above depends on them. But had any of the four diverged, the
gate would have hard-failed with a tooling error instead of a verdict.
**Any future oracle gate run needs the already-identified fix first**:
resolve byte lengths from the GGUF vocab via `tests/tooling/gguf_reader.py`
instead of from the response's own candidates. Cheap, and it removes the
last "refuses to guess" hole. This is M3 work under option 3.

## What the count does to each option

- **Option 1 (strict).** Fails decisively, not marginally: 19 divergences
  against a waiver budget of 2, and only **one** of them (p02, 0.000950)
  even has an oracle-side gap under ε — its waiver would additionally
  require stoke's own gap under ε, still unmeasured. Grinding this
  produces a 19-item evidence package proving 19 times over what the L2
  ladder already proves once.
- **Option 2 (bounded near-tie recognition).** Also dead, and by a wider
  margin than the 07-27 analysis anticipated. Its 0.1-nat bar was
  calibrated on the v1 case (0.0264); **7 of the 18 exceed it**, up to
  0.2039. Making option 2 absorb this census would mean moving the bar
  past 0.21 nats *and* raising the ≤2 budget to ≥19 — that is not a
  parameter tweak, it is abandoning the gate, and per governance I will
  not propose it. Stated here only so the option space is complete.
- **Option 3 (re-scope to C-vs-ref, defer oracle parity to M3).** The
  count, the uniform top-2-swap signature, and the all-tier spread are
  exactly the "systematic at M2's numeric distance" condition this memo
  pre-committed to option 3. It is also already designed for: the forward
  note's M3 **LOCKED decision #2** quantizes activations to Q8_0 blocks
  and runs an i32 integer core on both scalar and SIMD paths — the
  oracle's own numeric class — and *already* mandates "re-run the
  50-prompt parity gate under the M2 waiver protocol after the switch",
  anticipating that token streams shift there. Deferring costs no new
  work; it moves the check to the milestone where both sides finally do
  the same arithmetic.

**COST OF OPTION 3, MEASURED, SO THE CHOICE IS HONEST.** Option 3's
substitute gate is C-vs-ref token-exactness, and `ref_forward.py` is a
debug tool, not a gate runner: it re-dequantizes every weight tensor on
every matvec (deliberately — caching all 36 layers would need 14–16 GB on
a 16 GB box) and measured **13.8 s/position** end-to-end in W1's own
benchmark. A full 50×64 C-vs-ref run is 7,038 positions ≈ **27 h
serial**, and parallelism is again RAM-bound (~1.6 GB f32 `token_embd`
cache per process plus transient dequants), so realistically ~7–10 h wall
at 3–4 processes. **Recommended shape if option 3 is chosen:** a
stratified subset — all 6 tiers represented, and specifically including
the prompts that diverged against the oracle, since those are precisely
where stoke's f32 result most needs independent corroboration — plus the
existing full 36-layer L2 ladder for depth. That is a few hours, not a
day, and it is strictly more informative per hour than 50 uniform prompts.
Note also that "both pure f32 ⇒ 50/50 token-exact" is an **expectation,
not a measured fact**: numpy's BLAS summation order differs from stoke's
scalar loops, and W3's L2 ladder already shows f32 non-associativity drift
up to 6.1e-4 in hidden state. Ties tighter than that could flip C-vs-ref
too — a much smaller class than the 0.001–0.204 nat spread seen against
the oracle, but it must be measured, not assumed.

**STILL PENDING (does not change the option mapping):** the survey's
probe phase is computing stoke's own top-5 at each of the 19 divergences
(19 probes, ~2 h). Those numbers complete the both-sides evidence for the
record. They cannot change the recommendation: option 1 needs ≤2
divergences and there are 19; option 2 needs every gap under 0.1 and 7
oracle-side gaps already exceed it.

**RECOMMENDATION: option 3**, per this memo's own pre-committed rule,
with the stratified-subset shape above. The forward pass is not in
question — 74.9% of ids match exactly, 31 prompts match 64/64, every
readable divergence is a top-2 swap of the same candidate pair, and the
L2 ladder puts all 36 layers within 1e-3 of the f32 reference. What the
oracle gate is currently measuring is the distance between two arithmetic
classes, and M3 closes that distance by construction.

---

## DECISION — KABIR, 2026-07-28. **OPTION 3, STRATIFIED SUBSET + L2 LADDER.**

Taken on the census above, per this memo's own pre-committed rule (≥3 →
option 3). Two parts, both his:

1. **Option 3.** M2's token gate is re-scoped to **C-vs-ref** (stoke's C
   against `ref_forward.py`, both pure f32). **Oracle token-parity is
   DEFERRED TO M3**, where LOCKED decision #2's Q8_0 activation
   quantization + i32 integer core put stoke in the oracle's own numeric
   class, and where the forward note already mandates re-running this
   exact 50-prompt oracle gate under the §8 waiver protocol.
2. **Shape: stratified subset + the existing 36-layer L2 ladder**, not
   all 50 prompts — on the measured cost above (13.8 s/position ⇒ ~27 h
   serial for a full 50×64 run).

**CARRIED FORWARD TO M3, so it is not lost:** (a) the deferred oracle
parity gate itself; (b) the aligner fix — resolve token byte lengths from
the GGUF vocab via `gguf_reader.py` rather than from the response's own
`top_logprobs` candidates — **required before any future oracle gate run**,
since 4 of 50 prompts currently make the aligner fatal; (c) this census as
the M2-era baseline the M3 re-pass is compared against.

**ESCALATION ROUTING CHANGED (Kabir, same session):** major decisions now
go to **Fable**, not to Kabir. This memo's own question was the last one he
took personally. Accordingly the *definition* of the substitute gate —
exact subset composition and the pass criterion, both fixed BEFORE the run
so that no bar is written after seeing results — was referred to Fable, not
chosen by the executing Opus session. Its ruling is recorded below.
