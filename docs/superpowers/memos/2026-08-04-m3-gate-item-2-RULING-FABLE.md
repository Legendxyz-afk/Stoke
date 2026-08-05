# RULING — M3 gate item [2] and every dependent: the oracle bar is retired as structurally unreachable, the criterion is restated on evidence the project owns, and M4–M7 are ruled forward

**Status: binding ruling. Fable 5 session, 2026-08-05 ~13:12 IST, Kabir-launched
per the routing.** Question of record:
`2026-08-04-m3-gate-item-2-FOR-FABLE.md` (unmodified except its ANSWERED stamp).
Read before ruling: the full memo, `docs/ROADMAP.md`, the ledger through
`878307f` (tree clean), the K3 ruling (`2026-08-03-k3-simd-headroom-RULING-FABLE.md`),
and the M2 substitute-gate terms as summarized in the memo §1 — which are not
revisited here.

**The filled §13 checklist is §11 of this file. The prose sections are the
reasoning; the checklist is the mechanical extract. Where they could ever be
read to differ, the checklist governs — it is the part Opus executes.**

---

## 0. THE RULING IN ONE PARAGRAPH

Item [2] as written asked for something no correctness work can deliver — §4 of
the memo is correct, and the memo's own calibration (~0.5%/token near-tie flips
× ~3,200 gate decisions) shows the failure was *predicted*, not unlucky. The
criterion is therefore **restated, not waived**: M3's correctness exit becomes a
**classified-divergence criterion on evidence the project owns** — the
M2-vs-HEAD full-corpus sweep plus a reference probe grid on both quant files,
every divergence graded by the classification rule ruled in §6, **zero
BUG-class divergences tolerated anywhere**. The live oracle is **demoted from
gate to survey**: valuable, optional, Kabir-launched, recorded
divergence-counted, never again pass/fail. This is option (d) wearing (a)'s
machinery, and it closes M3 without `llama-server` (§9.8: yes, explicitly).
K3 exits as-is (§3). The f32-reduction twins are parked dead-by-default (§4).
M4–M7 are ruled forward in §8; the roadmap DAG is unchanged.

---

## 1. THE STRATEGIC FRAME — what the criterion is for, so the restatement is an alignment and not a retreat

stoke's goal (ROADMAP, one-line): llama.cpp's footprint × vLLM's concurrency
for agent workloads on consumer hardware. **Correctness, for that goal, means:
stoke faithfully computes the model — deterministic, self-consistent, within a
measured numeric envelope of a trusted reference — not that stoke reproduces
llama.cpp's arithmetic bit-for-bit.** llama.cpp is a *calibration referent*,
one point in the space of legitimate arithmetic classes; the project's own
trust order already puts `ref_forward.py` above it for adjudication, and at
p02:16 the reference sides with stoke by 3× (memo §3.6). A criterion that
fails stoke for llama.cpp's own near-tie coin-flips measures the referent's
arithmetic, not stoke's correctness.

**The durable gate semantics for the rest of Phase 1, in one line each:**

1. **EXACT where stoke owns both sides** — scalar↔SIMD twins, thread-count
   invariance, determinism, M4's batch-composition identity, M5's replay.
   Bit-for-bit, no waivers, hard fail. These are the properties the
   architecture stands on and they cost nothing to demand.
2. **CLASSIFIED AND COUNTED where an arithmetic-class boundary is crossed** —
   stoke-integer vs f32 reference, stoke vs oracle, KV-quantized vs not.
   Divergences are graded by §6's rule; BUG-class = zero tolerance;
   arithmetic-class = counted, recorded, tripwired.
3. **RE-ANCHORED where the referent moves** — perf bars re-anchor at gate time
   (§8.2), parity referents stay pinned (§7.7). Two referent kinds, two rules.

The milestone DAG M3→M7 is unchanged — this ruling changes what gates *mean*,
not what gets built or in what order. Risk ranking forward, for where effort
goes: (1) M4/M5 is where the product thesis is proven or dies, and the
binding physics is the 6T superblock-unpack ceiling K4-entry measured — **the
~52% K-quant size-ratio retention (llama: 95.5% at 1T) is the highest-value
optimization target left, because it feeds the aggregate bars directly**;
(2) K3's residual and the f32 twins are not on that path (§3, §4); (3) M7's
long-context parity risk is pre-armed now by making the probe grid measure τ's
position dependence (§8.9).

---

## 2. Q1 — M3's EXIT CRITERION: item [2] v2, the classified-divergence criterion

### 2.1 Why not (a), (b), or (c)

- **(c) keep it literally** — honest only as a permanent block. §4's argument
  is structural: the waiver forgives *mutual* uncertainty, stoke's *confidence*
  is what disqualifies it, and more accuracy makes stoke more confident. A
  criterion that punishes convergence is not measuring correctness. Rejected.
- **(b) restrict to the 31** — rejected on both halves: it deletes exactly the
  near-tie-dense prompts where a real C bug hides best (M2's own words), and it
  keeps the structural bet at reduced odds — the census's "31 exact" records
  one draw of the coin, not the absence of near-ties; ~2,000 retained decisions
  at the measured ~0.5%/token near-tie density still expect order-ten flips
  eventually. Same unreachable shape, less bug-detection power.
- **(a) full C-vs-ref** — right referent, wrong dose. It is a token gate across
  a numeric-class gap that is now 0.043 nats (three orders above the 1.4e-5 the
  ε=0.001 machinery was calibrated on), so it needs exactly the classification
  machinery (d) builds — and at 9.7 s/position it spends ~9.5 h re-deriving
  what M2's ruled gate plus the M2-vs-HEAD sweep already compose to
  (M2 ≡ ref on the subset; HEAD ≡ M2 on stable tokens ⇒ HEAD ≈ ref on stable
  tokens). Build the machinery, spend a tenth of the compute (§2.3).

### 2.2 The ruled criterion — item [2] v2

M3's correctness exit is the conjunction of:

- **[2v2-i] Within-stoke exactness — already ✅, cited not re-run.** Gate [1]
  (twins bit-exact, 1e4 tensors/kernel, all quant kinds), determinism on
  re-probe, thread-invariance at 1/2/8/16 (memo §3.3), K5's stream hashes at
  1/6/10/16.
- **[2v2-ii] The M2-vs-HEAD sweep, all 50 prompts, Q8_0 (`m2_sweep.py`,
  resume from 6/50).** Every first-divergence classified per §6; **PASS = zero
  BUG-class**; the long-context tier p40–p44 explicitly included. Tripwire:
  >30 divergent prompts ⇒ stop, anomaly memo (expected: on the order of the
  19). REQUIRED evidence — it is the K-work's token-level alibi and it needs
  no oracle.
- **[2v2-iii] The reference probe grid, BOTH files (spec §2.3).** Teacher-
  forced `ref_forward.py` checks of HEAD's banked streams at sampled
  positions; **PASS = zero BUG-class at every probed position**; class flips
  counted with tripwire ≥4 per file-grid (expected ≈0–1). This is also the
  measurement that freezes τ and answers §8.9's position question. REQUIRED —
  it is the only *independent* check the K-quant path has (no M2 referent
  exists: `4623441` is the first commit where stoke runs Q4_K_M at all), and
  on Q8_0 it bounds the class envelope end-to-end.
- **[2v2-iv] Tripwires quiet, report written.** Gate report at
  `docs/superpowers/gate-reports/2026-08-XX-m3-parity-v2.md`, binaries named
  by sha256, models by name+size, τ values recorded as frozen BEFORE the
  classifications they grade.

**The oracle is REMOVED from the criterion.** When and only when Kabir
authorizes `llama-server`, either leg may be run as a **survey**: recorded
divergence-counted per §6, reported beside the M2 census (19/50, 18/18 top-2
swaps, 0.001–0.204 nats), never pass/fail, never blocking, never stamped as a
gate. Its value — comparison against an implementation that is not ours — is
real, which is why it is kept; its bar form is what §4 killed.

### 2.3 The probe protocol — mechanical, so Opus executes without judgment

- **Streams:** leg 1's banked Q8_0 ids (all 50, verified under HEAD's sha256);
  for Q4_K_M, HEAD generates and banks 50×64 first (~17 min at the measured
  0.127–0.151 s/forward; no oracle involved).
- **Grid, per file:** every prompt at its final generated index (63), plus the
  mid index (32) on p40–p44. RECOMMENDED if wall-clock permits: mid+last on
  all 50. Minimum grid = 55 probes/file.
- **Each probe:** one teacher-forced ref forward over prompt + banked
  ids[0..k−1]; record ref's argmax, both sides' top-5, and the Δ-profile over
  the union of top-5s anchored on HEAD's chosen id (the memo §3.6 method).
  Mismatch ⇒ classify per §6.
- **Outputs, per file:** per-band max and p95 |Δ| where bands are absolute
  position [0,100) / [100,300) / [300,600); mismatch count; class of each.
- **FREEZE τ before classification is finalized:**
  τ(f32↔intcore, file, band) = **max(0.10, 2 × that band's p95)**, recorded in
  the gate report. The 0.10 floor exists because 0.043 is a single-position
  measurement and a knife-edge τ would classify on noise. Ordering rule: the
  probe grid runs and freezes τ **before** [2v2-ii]'s divergences are graded
  (same τ family — M2↔HEAD and ref↔HEAD are both f32↔intcore pairs; §3.6
  measured both at 0.0430 on the same position).
- **Cost:** ~10–20 min/file typical, worst case ~45–90 min/file if long-tier
  forwards run slow. Classification follow-ups on [2v2-ii]'s divergent prompts
  need the M2 binary's top-5 at the divergent position (~2.5–3.5 min/prompt,
  full re-generation) — order an hour for ~19 prompts. Everything local,
  nothing Kabir-blocked.

### 2.4 The sub-decisions (a1–a4, b1–b3, d) — answered even where the option was not taken

- **(a1) subset:** all 50 for both the sweep and the probe grid. M2's 24 was
  built around oracle-divergence structure this criterion no longer rests on.
- **(a2) waiver policy:** **the ε=0.001 / budget-2 waiver machinery is RETIRED
  for every cross-class token comparison** — it was calibrated for a 1.4e-5
  same-class gap that no longer exists anywhere it would apply. It survives
  untouched in the M2 record. Its replacement is not a recalibrated waiver but
  §6's classification: class divergences are not "forgiven", they are counted
  and tripwired; BUG divergences are never forgiven at any budget.
- **(a3) the ~9.5 h ref run:** NOT REQUIRED and not part of the criterion. If
  Kabir wants the fuller record it is authorized as a background evidence run
  (it would enrich §8.9's τ corpus), and it never blocks the stamp.
- **(a4) must Q4_K_M use ref too:** YES — [2v2-iii] runs on both files. The
  K-quant leg is exactly where an independent referent earns its keep: gate
  [1] proves twin-equivalence (both sides ours), the K4 goldens prove dequant
  transcription against llama.cpp's own output, and the ref probe is the only
  end-to-end check of the composed path.
- **(b1)** n/a — (b) not chosen; the 19 need no special evidence because they
  are no longer a gate category (see b3).
- **(b2)** n/a as asked; ruled anyway because §9.2-C needs it: **the Q4_K_M
  leg's corpus is the same 50; no diverging list is required**, because §6
  grades each divergence on its own margins and candidate sets — the
  historical 19 becomes corroborating context, never a gate input. If an
  oracle survey ever runs on Q4_K_M, its diverging list is *derived* from that
  survey as a recorded fact, not built as a prerequisite.
- **(b3)** yes — handled by §6, and this generalizes: **membership in the 19
  carries no exculpatory or incriminating weight by itself**, on either file,
  in either direction. The rule is the test; the list is history.
- **(d) the bound:** for gate purposes the bound is **zero BUG-class**, with
  class divergences bounded only by tripwires ([2v2-ii] >30 prompts;
  [2v2-iii] ≥4 flips/file-grid) calibrated against the measured base rate
  (~0.5%/token, 19/50 prompts). For oracle *surveys*, the recorded comparison
  is against that same M2 census, with "investigate before citing" at >2× the
  census rate. A divergence-rate bar tighter than zero-BUG is refused: it
  would re-import the structural problem at a different constant.

---

## 3. Q2 — K3's EXIT: exits as-is; the residual is carried headroom with a number on it

**K3 exits as-is. Nothing about it blocks the M3 stamp.** Grounds:

1. **The band it missed was a scoping aid; the bar it feeds passed.** The K3
   ruling said explicitly: "This expectation is a scoping aid for K3-the-work.
   Gate [3] … is judged at M3 close." Gate [3] was judged — re-anchored
   HARDER (7.80 → 8.08–8.46 against a gate-time llama) and passed both legs at
   GATE grade (1.888× / 1.630× against the ≤2.0 criterion, margins 5.6% and
   18.5%). Under §8.11's headroom rule this is the textbook carried case: the
   milestone's own bar passed without the residual, and the residual is
   recorded with numbers.
2. **The residual is priced and located, not mysterious.** Of the 1.437×:
   1.235× is the measured price of gate [1]'s bit-exactness (three-way
   interleaved, GATE-grade), ~1.16× is unexplained but bounded, and the per-op
   row names where it lives (LOGITS 1.18×, narrow-output matvecs, ffn_up
   1.26×). An "unexplained" with a magnitude, a location, and a GATE-grade
   measurement behind it is headroom, not debt.
3. **The marginal session is worth more elsewhere.** The binding risk is now
   M4/M5's aggregate bars under the 6T unpack ceiling, where our K-quant
   kernels retain ~52% of the size ratio against llama's 95.5% at 1T. Q8_0
   four-row polishing does not feed that; K-quant unpack quality does.

- **Q2-2 — the ~1.16×:** NOT owed before the stamp. Carried, with the standing
  condition of §8.11: it enters the headroom ledger with number + location,
  and it is the first shopping list opened if a later bar misses.
- **Q2-3 — gate [1]'s 1.235× price: PERMANENT for Phase 1.** Bit-exactness is
  not a luxury; it is the property that made K4's twins land gate-free, K5's
  pool land parity-re-run-free, and every exact-diff gate cheap. Selling it
  buys ~1.2× per-core and costs the verification architecture of every
  remaining milestone. **Named revisit triggers (DEFER-shaped, not declined):**
  (i) an M4–M7 bar miss where the then-measured bit-exactness price ≥ the
  miss — then it is priced against that bar with Kabir deciding; (ii) Phase
  1.5's CUDA work, where the twin methodology is re-architected anyway. Until
  a trigger fires, no session spends time on ggml-style kernel strategy.

---

## 4. Q3 — f32-REDUCTION TWINS: out of M3, and parked DEAD-BY-DEFAULT, not deferred-as-task

- **Q3-1:** OUT of M3 — and not "deferred to M4" as if it were queued work.
  The measured facts: the remaining twins are worth **≤1.32% combined**, and
  several of them **change the token stream** (reordered f32 reductions flip
  near-ties). After this ruling the token stream is load-bearing evidence —
  [2v2-ii]'s banked ids, M4's batch-composition identity, M5's replay rule all
  stand on it — so a stream-changing ≤1.32% "optimization" has **negative
  expected value**: it invalidates every banked id and every cached gate
  entry (the cache keys on `stoke_hash`) for less than measurement noise.
  **Reopening trigger:** a profiled bar miss at M4+ that names an
  f32-reduction op at ≥5% of token time. Absent that, they are never picked
  up.
- **Q3-2:** yes, Q1's ruling *hardens* this — it is why the parking is
  dead-by-default rather than a deferral. (The direction of the interaction:
  Q1 made stream stability part of the criterion substrate, which raises the
  price of every stream-changing edit project-wide. That is intended.)

---

## 5. Q4 — STANDING AUTHORIZATIONS

- **8.1 — compute:** the ~9.5 h run is not required (§2.4-a3). Standing rule:
  **evidence-only unattended runs** (no source change; writes confined to
  caches, `bench/results/`, reports) **up to ~4 h wall are Opus's discretion,
  announced to Kabir at launch** — it is his box, and GATE-grade needs it
  quiet anyway. Longer than ~4 h, RAM-pressuring (working set > ~8 GB), or
  oracle-involving: Kabir per-run. **No spend of any size ever needs a fresh
  Fable ruling once the evidence it produces has been ruled on** — rulings are
  about meaning, not wattage.
- **8.2 — re-runs:** YES, pre-authorized, unasked, after any change Opus
  believes gate-relevant — PROVIDED the criterion text and frozen constants
  are untouched and every run is recorded (report + ledger line). Note the
  economics inverted: the stoke side is ~19/~17 min per leg now, so
  fix-then-re-gate is cheap; the expensive resource was only ever the oracle,
  which is out of the criterion.
- **8.3 — corpus:** the M2-committed 50+10 is **FROZEN as the gate
  denominator** (its value is longitudinal comparability). Opus MAY add
  prompts any time for diagnosis in an additive, clearly-labeled tier
  (`x00…` prefix, never renumbered, never retired); additions enter a gate
  denominator only by a future ruling. The Q4_K_M diverging list is not a
  corpus question and is not needed (§2.4-b2).
- **8.4 — what Opus must not decide alone:** confirmed as read (the criterion,
  the stamp, anything that moves a bar), PLUS: any frozen constant a gate
  reads (τ, envelopes, tripwire counts), any standing-rule text (§8's rules),
  and anything that changes what a banked number *means* (schema semantics,
  referent identity). Explicitly WITHIN Opus's authority: tooling defect
  fixes with selftests + failable mutations (house standard), running and
  re-running ruled evidence, applying §6 mechanically, resuming interrupted
  runs.
- **8.5:** acknowledged — `llama-server` starts are Kabir's, here and in every
  §9/§10 branch; rulings below say what happens *when* he authorizes, never
  whether.

---

## 6. THE CLASSIFICATION RULE (§9.4) — ratified with corrections; this is the load-bearing section

Opus's two-condition rule is ratified in structure and corrected in three
places: the margin condition is made two-sided, the top-5 condition gets a
boundary tolerance (rank-5 churn under tiny noise is not evidence of
anything), and the rule gets a third outcome so a knife-edge constant never
decides a gate alone — the project already owns the right escalation
instrument (the L2/L3 ladder).

**THE RULE. At the first divergent position, sides X and Y, X chose x, Y chose
y, x ≠ y. Compute both self-margins: g_X = logit_X(x) − logit_X(y) and
g_Y = logit_Y(y) − logit_Y(x) (each ≥ 0 by construction). With τ = τ(pair,
band) from the frozen table:**

- **ARITHMETIC-CLASS** iff **max(g_X, g_Y) < τ** AND the top-5 sets are equal
  after discarding, on each side, any candidate within τ of that side's rank-5
  logit (order free; boundary churn within τ is not a set difference).
- **BUG** iff **max(g_X, g_Y) ≥ 2τ**, OR either side's chosen token sits more
  than τ below the other side's rank-5 logit (a candidate the other
  implementation never ranked is not class noise).
- **ESCALATE** otherwise (the band τ ≤ max(g_X, g_Y) < 2τ, or a set difference
  beyond the boundary tolerance): run the L2/L3 ladder over the shared prefix
  at that position. Ladder ≤1e-3 / no-STEP ⇒ ARITHMETIC-CLASS, margin
  recorded; a STEP ⇒ BUG.

**Fail-safe direction, stated because the memo asked about the unmeasured
"smallest genuine bug signal": the rule never needs it.** Anything not
affirmatively classifiable lands in ESCALATE, and the ladder — not the
constant — decides. An unmeasured bug class therefore fails toward
investigation, never toward a pass.

**τ TABLE (9.4-2), frozen by this ruling:**

| pair | τ | basis |
|---|---|---|
| f32 ↔ integer-core (ref↔HEAD, M2↔HEAD), per file, per band | **max(0.10, 2 × band p95)** — frozen by the §2.3 probe before it grades anything | 0.0430 measured at one position (memo §3.6); the floor covers single-point uncertainty |
| stoke ↔ oracle (any survey) | **0.25 nats, fixed now** | set from PRE-EXISTING data it will never itself grade: the M2 census divergence gaps 0.001–0.204 (all layer-ladder-cleared as class effects) and the 0.1427 max ref↔oracle Δ |

τ is **per implementation-pair and per position band**, never universal
(§8.9). Any desire to move a frozen τ after seeing gate data is the
fix-the-bar move: memo to Fable, never an adjustment.

**9.4-3 — who applies it:** Opus, mechanically, including the ESCALATE
ladder runs, without further rulings. A BUG verdict blocks the affected leg
and Opus bisects under the pre-authorizations of §7 — the verdict itself
never needs Fable; only criterion or τ changes do.

---

## 7. CONDITIONAL RULINGS (§9) — every branch, ruled now

### 7.1 M2-vs-HEAD sweep outcomes

- **(A) all 50 identical:** discharges the K-correctness concern at token
  level in full — no ref evidence owed *for that concern*. [2v2-iii] still
  runs: it serves the class-envelope measurement, the τ freeze, and the
  K-quant path's independent check, which (A) does not touch.
- **(B) differences all inside the 19:** treated the same as (A) **after**
  each divergence classifies as ARITHMETIC-CLASS per §6. Expected outcome
  (6/6 partial already matches it). "Inside the 19" is corroboration, not the
  test.
- **(C) differences outside the 19:** does NOT auto-block. Grade by §6 first:
  BUG ⇒ hard block + bisect (pre-authorized: binary bisection across the K
  commits, `--dump-layers` L2/L3, twin harness); ARITHMETIC-CLASS ⇒ record
  and proceed — the 19 was an *oracle*-divergence census, not a near-tie
  census of the M2↔HEAD pair, and the two sets have no reason to coincide
  exactly.
- **(D) sweep cannot complete:** it is REQUIRED evidence ([2v2-ii]). Fix the
  tooling and re-run — it is cheap, oracle-free, and resumable. M3 does not
  stamp on a partial sweep.

### 7.2 Leg 2 (Q4_K_M) oracle outcomes — all survey-grade now, whenever Kabir authorizes

- **(A) 50/50 pass:** recorded; does NOT stamp item [2] by itself — the
  criterion is [2v2], and a pass on the file with *more* class distance while
  Q8_0 flipped would be evidence about near-tie luck, not about correctness.
- **(B) fails inside the 19:** inherited class — confirmed, record it.
  (With the caveat that the 19 is a Q8_0-measured list; the classification,
  not the list, is what grades it.)
- **(C) fails outside the 19:** grade directly by §6. No list-building
  prerequisite — the missing Q4_K_M diverging list is derived from the survey
  itself, afterward, as a recorded fact.
- **(D) K-quant-only divergence:** YES — Opus is pre-authorized to bisect
  scalar-vs-SIMD twin and Q4_K-vs-Q6_K unasked. A twin split contradicts gate
  [1]: re-run the gate-[1] harness on the failing shape first (it is the
  cheaper, sharper instrument), and a confirmed split is a FINDING that goes
  straight into a memo regardless of what else the divergence grades as.

### 7.3 Reference-comparison outcomes (the probe grid, or any fuller ref run Kabir opts into)

- **(A) passes (zero BUG):** the correctness leg closes, subject to §7.5's
  stamp list. Confirmed.
- **(B) BUG-class failure:** HARD BLOCK on the stamp. Opus is pre-authorized
  to bisect via `--dump-layers` L2/L3 without asking — both sides are ours
  and the memo is right that this is the strongest bug signal available. The
  block lifts when the bug is fixed and the affected evidence re-run.
- **(C) failures inside the class gap:** these are ARITHMETIC-CLASS
  divergences by definition, not waivable events — counted, recorded,
  tripwired (≥4 per file-grid ⇒ stop, anomaly memo; expected ≈0–1 at the
  measured near-tie density). ε/budget machinery: retired (§2.4-a2).

### 7.4 — see §6 (the classification rule).

### 7.5 M3 stamp and tag (9.5)

**The exact precondition list — all of it mechanical:**

1. [1] ✅ (test_simd 12/12, 1e4/kernel, all quant kinds — `e26ff0d` et al.).
2. [2v2] per §2.2: sweep complete + all-CLASS (+tripwire quiet); probe grids
   complete on BOTH files, τ frozen first, zero BUG, flip tripwire quiet;
   Q4_K_M streams banked under HEAD's sha256; gate report written naming
   binaries by sha256, models by name+size, τ table included.
3. [3] ✅ re-anchored 8.08–8.46, both legs GATE-grade (`86fd2fa`).
4. [4] full matrix green (Win + WSL plain/asan/tsan) at the last
   source-touching commit; re-run if any `src/` file changed since.
5. [5] ✅ microbench table + `history.jsonl` GATE row.
6. Entry chores ✅ on record: kv-cap guard `d9b633f`, aligner `aa9d6ff`,
   `-O3` + fp-contract pin `d0f8a6b`.

**Sequence CONFIRMED as Opus assumed, with the memo stamps added:** gate
report → ROADMAP row → MEMORY.md build log → ledger → owner line M3→M4 →
`git tag m3` — all local, **no push** (Kabir's). The FOR-FABLE ANSWERED stamp
is applied by this session; the RULING file needs no stamp.

**9.5-3:** Q2 and Q3 do NOT block the stamp — both ride into M4's entry
NEXT-block as recorded carried headroom (with §8.11's number+location
requirement).

### 7.6 Forward guards (9.6)

- **9.6-1:** Q1 does not *set* M4's KV criterion by precedent — M4's was
  already (d)-shaped in writing. §8.1 below gives it the bound it lacked.
  Independent rulings that happen to agree, which is the healthy direction.
- **9.6-2:** CONFIRMED — batch-composition independence stays an EXACT,
  bit-level criterion regardless of anything in this ruling. It is
  within-stoke, same arithmetic class, zero excuse for divergence, and it is
  the property M5's debuggability stands on.
- **9.6-3:** CONFIRMED — M5's "replay recorded ids, never re-sample" is
  untouched (and Q3's ruling exists partly to protect it).

### 7.7 Referent drift (9.7)

**The parity/classification oracle is FROZEN at b10037 (sha-pinned) for the
life of Phase 1.** Every banked oracle comparison keeps its meaning. If a
future model forces a newer llama.cpp (see §8.10), the new binary enters as a
**second pinned referent for that model only** — old banked data is never
re-graded against a new referent, and no upgrade is ever silent.

### 7.8 If the oracle is never authorized again (9.8)

**YES — M3 closes on oracle-free evidence alone, by construction: [2v2] is
oracle-free.** So do M4–M6 (their criteria are within-stoke or vs-ref). M7's
forward-pass parity is likewise ruled oracle-free (§8.8); only its tokenizer
gate wants `llama-tokenize`, which is a CLI oracle, not a server. No path to
any remaining stamp passes through a resource that may not be forthcoming.

---

## 8. FORWARD RULINGS — M4 THROUGH M7 (§10)

### 8.1 Q8-KV divergence bound at M4

The criterion is **KV-ON vs KV-OFF, within-stoke, same engine, only the KV
representation differing** — graded by §6 with its own τ (frozen from an M4
entry probe of the ON↔OFF pair), **zero BUG-class**, and class divergences
bounded by: **first-divergence prompt rate ≤ the M2 census rate (19/50) AND
flip rate ≤ 0.5%/token**, both on the frozen 50-corpus. Rationale: Q8-KV's
noise should live at or below the arithmetic-class noise this project already
accepts between implementations; if it measures worse than the base rate,
that is not a grind-until-green — it is a **results-vs-physics memo** (the
quantizer itself is then the finding: scale granularity, outlier channels).
**10.1b: YES — M4's KV criterion stays (d)-shaped regardless of what Q1 had
ruled.** Perplexity-style instruments are not introduced; the project grades
token decisions, one instrument family, end to end.

### 8.2 Bar re-anchoring is a STANDING RULE

**RULED STANDING for every remaining milestone: any bar expressed relative to
a moving referent (llama.cpp's speed, measured W, measured N) re-anchors to
the referent re-measured at gate time, in-session, interleaved, per the
Comparability Rule.** One asymmetric guard: a re-anchor that makes a bar
HARDER applies silently (K6 did exactly this, honestly); a re-anchor that
would make a bar EASIER (the referent measured *slower* than its recorded
figure) requires the referent's sidecar attached and a one-paragraph memo
BEFORE the gate is judged against it — an aging or mis-measured referent must
never quietly clear a bar. That asymmetry is the whole point of the rule.

### 8.3 The bandwidth budget's N

**Re-measured per milestone** — K5's own data settle this: the optimum moved
6→8 when the kernels' bottleneck moved, because "the optimal thread count is
a property of the kernel's bottleneck, not of the machine." A frozen N would
embed a stale fact about superseded kernels. The re-measurement is one sweep,
in-session, and the budget re-anchors with it per §8.2 (both directions
guarded the same way).

### 8.4 M4 — gate items vs design constraints, and the stamp list

**GATE ITEMS (stamp preconditions):**

1. **Sim invariants:** the 11 invariants asserted after EVERY op, ×1e6-op
   randomized runs, ASan-clean — with **sequence teardown-at-any-point and
   full-reservation admission exercised in the trace mix**, and the pool size
   **runtime-variable in the sim** (the 08-02 constraints are verified here,
   not alongside).
2. **Batch-composition independence:** same seq ⇒ token-identical at batch
   sizes {1,2,4,8,16}, real model, EXACT (bit-level ids). Hard fail.
3. **Prefix-dedup correctness:** dedup'd shared prefix produces streams
   identical to un-dedup'd, refcounts prove out in sim and one real-model
   spot check.
4. **Q8-KV:** ON↔OFF per §8.1 — zero BUG, bounds held (default ON/OFF stays
   Kabir's at M5).
5. **The static N∈{1..16} curve, GATE-grade quiesced:** monotone
   non-decreasing aggregate within within-cycle spread through N=8;
   **any genuine inversion (beyond spread) = results-vs-physics memo before
   the stamp, not a silent recording** — a static-batch inversion is the M0
   disease stoke exists to not have, and it would indict the design, not the
   tuning.

**DESIGN CONSTRAINTS (verified inside the items above, not separate gates):**
block=16; `kv_view` body-swap only; the budget-note-first entry requirement
(the note OPENS with W re-measured at entry per §8.2/§8.3, M5's bars checked
as budget lines before paged KV is scoped — already bound by the K3 ruling).

**Stamp sequence: identical shape to §7.5, tag `m4`.**

### 8.5 Bar-miss protocol — generalized, standing, and retroactively satisfied for K3

**RULED STANDING for every remaining milestone:** a missed bar or band ⇒
(1) stop grinding by default; (2) run the attribution ladder to locate the
gap; (3) write the results-vs-physics memo — expected, measured, mechanism,
priced options; (4) route per §8.12 (build continues on non-dependent work);
the decision is Kabir's/Fable's. **Retroactive application to K3: §3 IS that
memo's disposition** — and the process that actually happened
(measure → attribute → price → escalate priced-not-proposed) is exactly this
protocol; it is now written down as the rule it was behaving as.

### 8.6 Acknowledged

Q8-KV's default ON/OFF at M5 is Kabir's call. Not covered, not constrained,
by anything here — except that §8.1 exists to hand him a measured input.

### 8.7 M6's gate — enumerated

The unchanged-M0-harness smoke is **the definitive item, not the only item**.
M6's stamp list: (1) **the smoke: M0's bench harness drives a `stoke` adapter
UNCHANGED** — this is the "it is actually embeddable" proof and the one that
cannot be argued around; (2) openai-python suite green against the server;
(3) ABI freeze checklist as a mechanical pass, including the C++ TU compile
test; (4) error-model conformance tests (the DECIDED mechanism: load-time
jmp_buf trap, `STOKE_ERR_*` API returns, invariant aborts through the log
callback); (5) the server compat traps checked off the forward note's list;
(6) matrix + tsan green (cross-cutting, as everywhere). Same stamp sequence,
tag `m6`.

### 8.8 M7's parity criterion for Llama-3.2 — ruled now, conditional on Q1 as asked

- **The tokenizer gate is UNTOUCHED: M1-style, byte-exact, oracle-gated via
  `llama-tokenize`** — a discrete domain with no arithmetic class; the §4
  argument does not apply to it, and it is cheap (CLI, not server).
- **The forward-pass parity criterion is [2v2]'s SHAPE re-instantiated for
  the new model — NEVER a 50/50 live-oracle token bar.** Concretely, on the
  M7 corpus (which includes ≥1k contexts): gate-[1] twins on every kernel the
  model exercises; a ref-vs-stoke probe grid with the long bands of §8.9;
  τ(pair, band) re-frozen from M7's own entry probe — Llama-3.2 has its own
  near-tie population and class gap, so Qwen3's constants do not transfer;
  zero BUG-class anywhere; targeted tests for the four silent-garbage deltas
  (QK-norm presence gating, the llama3 rope scaling formula probed at long
  positions, llama-bpe regex, template #2). There is no old-binary sweep for
  a new architecture — the ref probe carries the end-to-end weight, which is
  why its grid is denser here (§8.9).
- **The oracle for Llama-3.2 = survey, same status as §2.2's**, when Kabir
  authorizes; its diverging list is derived then, not prerequisite.

### 8.9 τ is per-band, and M7 re-freezes it at long positions

**τ is a function of position band, realized as a per-band frozen constant —
not a global constant, not a fitted formula.** Bands now: [0,100), [100,300),
[300,600). The §2.3 probe grid measures all three on both files THIS
milestone, which is deliberate: it is §8.8's calibration rehearsal.
**Collapse to a single constant is permitted iff adjacent-band p95 growth is
≤1.5×;** otherwise the bands stand. **M7 trigger, named: the M7 entry probe
adds [600,1200) and [1200,∞) bands on the Llama-3.2 corpus and re-freezes τ
there before any parity evidence is graded** — the `rope_neox`
position-growing term is exactly why the answer cannot be assumed constant at
1k+. (If the growth is real, the first suspect is already named in the
ROADMAP: the theta accumulation difference, logged 2026-07-31.)

### 8.10 M7's "re-run ALL baselines" vs §7.7's pinning — resolved

Opus's guess is CONFIRMED as the rule: **two referent kinds, two rules.**
**Perf baselines refresh** — M7 re-runs every engine at current dated
versions for the README table, and §8.2's re-anchoring applies to any bar
those feed. **The parity/classification oracle stays pinned** (b10037 for
Qwen3, for the life of Phase 1). If Llama-3.2 requires a newer llama.cpp
build to run at all, that build is pinned as the Llama-3.2 parity referent
(per-model pinning, §7.7) while b10037 remains Qwen3's. A perf table cell and
a parity referent are different objects even when they name the same
upstream project — that is the sentence that resolves the tension.

### 8.11 Headroom: carried vs owed — ratified with two additions

Opus's rule is RATIFIED: *carried if the milestone's own bar passes without
it and it is recorded with a number; owed if the bar passes only because the
bar moved.* Two additions with teeth:

1. **A carried-headroom entry must carry a number, a location, and a date**
   (e.g. "~1.16×, Q8_0 four-row matvec residual, 2026-08-03") — an unnumbered
   headroom claim is not an entry.
2. **On any bar miss, the headroom ledger is the FIRST shopping list:**
   priced options already on it (gate [1]'s 1.235×, the K-quant retention
   gap, attention-across-heads threading) are evaluated before any new scope
   is invented. And each milestone's close restates the carried list in its
   NEXT block — headroom never silently expires.

Note under §8.2's asymmetric guard a bar cannot silently move easier, so the
"owed" branch in practice means: passed only via referent degradation or
tripwire silence ⇒ memo before stamp.

### 8.12 When Opus believes a forward-note item is wrong

**Adopted: (iii) with one carve-out.** Default: proceed under the note AND
open (or append to) a FOR-FABLE entry immediately, without stopping — the
build keeps moving and the ruling is guaranteed to happen. **Carve-out: if
the suspect item defines a gate criterion, a stamp precondition, or the
meaning of evidence about to be banked, do NOT run that gate leg or bank that
evidence until ruled — everything else proceeds.** (A gate run under a
criterion believed wrong either fabricates a pass or manufactures a fail;
item [2] is the live demonstration.) The live-accumulating FOR-FABLE file
mechanism this memo pioneered is RATIFIED as the standing vehicle: one file
per decision cluster, append-only, Kabir launches when he chooses.

---

## 9. WHAT THIS RULING DOES NOT TOUCH

M2's closed gate and its terms; the M5 bars (17.7/14.2/10.8/16.8 and ITL);
the ≥50%-of-llama meaning of gate [3]; ε=0.001 as the M2-era record; the
Comparability Rule (binds everything here); push policy; the owner line
(Opus 5 executes M3→M7); `llama-server` authorization (Kabir's);
Q8-KV's default (Kabir's, M5).

**Direction check, per house practice:** this ruling retires one bar as
unreachable and replaces it with a criterion that is *stricter where
strictness is meaningful* — zero BUG-class anywhere, on more evidence (all 50
prompts, both files, plus a probe tier M3 never had), with the fail-safe
pointing toward investigation. Every discretionary constant (τ floors,
tripwires) was set from pre-existing measurements and frozen before the data
it will grade exists. No gate got easier to pass by accident; one got
possible to pass at all, which is different.

---

## 10. WHO DOES WHAT — the execution extract

| # | action | owner | blocking M3? |
|---|---|---|---|
| 1 | Bank Q4_K_M streams (50×64, ~17 min) | Opus | yes ([2v2-iii]) |
| 2 | Probe grids both files (§2.3), freeze τ, classify | Opus | yes |
| 3 | Resume `m2_sweep.py` to 50/50; classify divergences per §6 | Opus | yes |
| 4 | Gate report + stamp sequence per §7.5, tag `m3`, local | Opus | — |
| 5 | Oracle surveys (either file), when authorized | Kabir launches | never |
| 6 | M4 entry: budget note first; W + N re-measured; ON↔OFF τ probe | Opus, M4 | no |
| 7 | Headroom ledger entries (1.16×; K-quant retention; attn-across-heads) | Opus, in stamp/NEXT | no |
| 8 | ROADMAP/MEMORY.md supersession edits + ledger entry for this ruling | Fable, this session | — |

---

## 11. THE ANSWER SHEET — §13, every row filled

```
RULING — 2026-08-05 — M3 gate item [2] and dependents
Authority: Fable 5 session, launched by Kabir 2026-08-05 ~13:12 IST

--- Q1  M3's exit criterion (§5) -------------------------------------
Q1      option chosen: (d), as item [2] v2 (§2.2): classified-divergence
        criterion on owned evidence — M2-vs-HEAD sweep + ref probe grids on
        BOTH files, zero BUG-class (rule §6); oracle demoted to optional
        recorded survey, never pass/fail.
Q1-a1   subset: all 50 (sweep and probes); M2's 24 superseded for this use.
Q1-a2   waiver policy: ε=0.001/budget-2 RETIRED for all cross-class token
        comparison (calibrated for a 1.4e-5 gap that no longer exists);
        replaced by §6 classification + tripwires, no waiver budget exists.
Q1-a3   ~9.5 h ref run: NOT required. Authorized only as Kabir-optional
        background evidence; never blocks.
Q1-a4   Q4_K_M uses ref: YES — probe grid on both files ([2v2-iii]); it is
        the K-quant path's only independent end-to-end check.
Q1-b1   n/a — (b) not chosen; the 19 carry no special evidentiary duty.
Q1-b2   Q4_K_M corpus = the same frozen 50; NO diverging list needed —
        §6 grades per-position; a list is derived from any future survey.
Q1-b3   divergence outside the 19 -> graded by §6 (9.4). yes.
Q1-d    bound: zero BUG-class; class divergences bounded by tripwires only
        (sweep >30 prompts; probes ≥4 flips/file) against base rate
        ~0.5%/token, 19/50. Surveys report against the M2 census; >2× census
        = investigate before citing.

--- Q2  K3's exit decision (§6) --------------------------------------
Q2-1    K3 exits as-is: YES. Its band was a scoping aid; gate [3] passed
        re-anchored-harder both legs (1.888×/1.630× vs ≤2.0, GATE-grade).
Q2-2    ~1.16x residual: NOT owed before stamp. Carried headroom with
        number+location per §8.11; first shopping list on any future miss.
Q2-3    gate [1]'s 1.235x price: PERMANENT for Phase 1. Revisit triggers
        (DEFER, named): an M4–M7 bar miss ≥ covered by that price; or
        Phase-1.5 CUDA re-architecture.

--- Q3  f32-reduction twins (§7) -------------------------------------
Q3-1    OUT of M3; parked DEAD-BY-DEFAULT (≤1.32% combined, several change
        the token stream). Reopen trigger: profiled M4+ miss naming an
        f32-reduction op ≥5% of token time.
Q3-2    does Q1 change that? YES — it hardens it: token streams are now
        criterion substrate, raising the price of stream-changing edits.

--- Q4  Standing authorizations (§8) ---------------------------------
Q4-8.1  evidence-only runs ≤~4 h: Opus discretion, announced at launch.
        Longer / >~8 GB / oracle: Kabir per-run. Ruled evidence never needs
        a fresh Fable ruling to re-run.
Q4-8.2  re-run ruled gates after gate-relevant changes, unasked: YES —
        criterion text + frozen constants untouched, every run recorded.
Q4-8.3  corpus 50+10 FROZEN as gate denominator; additive labeled x-tier
        prompts allowed for diagnosis; enter denominators only by ruling.
Q4-8.4  confirmed + additions: also not Opus's alone — frozen constants
        (τ/envelopes/tripwires), standing-rule text, banked-evidence
        semantics. Explicitly Opus's: tooling fixes w/ selftests+mutations,
        evidence runs/re-runs/resumes, mechanical §6 application.
Q4-8.5  acked: llama-server starts are Kabir's, everywhere in this ruling.

--- 9.1  M2-vs-HEAD sweep outcomes -----------------------------------
9.1-A   YES — discharges the K-correctness concern; probes still run (they
        serve the envelope/τ/K-quant checks, a different concern).
9.1-B   same as A after each divergence classifies CLASS per §6.
9.1-C   neither auto-block nor ignore: grade by §6. BUG ⇒ block + bisect
        (pre-authorized); CLASS ⇒ record and proceed.
9.1-D   REQUIRED evidence ([2v2-ii]). If it cannot complete, fix and re-run;
        no stamp on a partial sweep.

--- 9.2  Leg 2 (Q4_K_M) outcomes — survey-grade whenever authorized ---
9.2-A   NO — a Q4_K_M 50/50 does not stamp item [2]; the criterion is [2v2].
        Recorded as a survey result.
9.2-B   confirmed — inherited class; record (list is Q8_0-measured; the
        rule, not the list, grades it).
9.2-C   grade directly by §6; the Q4_K_M diverging list is derived from the
        survey afterward, never built as a prerequisite.
9.2-D   YES — bisect twins + Q4_K-vs-Q6_K unasked; a twin split re-runs the
        gate-[1] harness first and is a memo-grade finding regardless.

--- 9.3  Ref-gate outcomes (probe grid; any fuller run Kabir opts into) -
9.3-A   confirmed — zero-BUG probes close the correctness leg, subject to
        §7.5's stamp list.
9.3-B   BUG-class ref failure: HARD BLOCK; --dump-layers L2/L3 bisection
        pre-authorized, no asking.
9.3-C   in-gap failures are CLASS divergences: counted, never waived;
        tripwire ≥4/file-grid ⇒ stop + anomaly memo. ε/budget: retired.

--- 9.4  THE CLASSIFICATION RULE -------------------------------------
9.4-1   RATIFIED WITH CORRECTIONS (§6): two-sided margin max(g_X,g_Y);
        top-5 set equality with τ-at-the-rank-5-cut tolerance; third
        outcome ESCALATE ⇒ L2/L3 ladder decides (≤1e-3/no-STEP ⇒ CLASS;
        STEP ⇒ BUG). Fail-safe direction: unclassifiable ⇒ investigate.
9.4-2   τ(f32↔intcore, file, band) = max(0.10, 2×band-p95), frozen by the
        §2.3 probe BEFORE it grades anything (basis: 0.0430 single-position
        measurement). τ(stoke↔oracle) = 0.25 nats, fixed now from the M2
        census (0.001–0.204 gaps, ladder-cleared) + 0.1427 max ref↔oracle Δ
        — pre-existing data it will never itself grade. BUG line at 2τ.
9.4-3   Opus applies it mechanically, ladder runs included; BUG verdicts
        block + bisect without Fable; only criterion/τ changes return here.

--- 9.5  Stamp and tag ------------------------------------------------
9.5-1   list (§7.5): [1]✅; [2v2] = sweep 50/50 all-CLASS + probes both
        files zero-BUG w/ τ frozen first + Q4 streams banked + tripwires
        quiet + report written (binaries by sha256, models by name+size);
        [3]✅ re-anchored; [4] matrix green at last source-touching commit;
        [5]✅; entry chores ✅ (d9b633f, aa9d6ff, d0f8a6b).
9.5-2   confirmed: report -> ROADMAP -> MEMORY.md -> ledger -> owner line
        M3->M4 -> tag m3; all local, no push. FOR-FABLE stamp done this
        session.
9.5-3   Q2/Q3 do NOT block; both ride into M4's NEXT block as carried
        headroom with numbers.

--- 9.6  Forward guards ----------------------------------------------
9.6-1   No precedent needed — M4-KV was already (d)-shaped; §8.1 supplies
        the missing bound. Independent and agreeing.
9.6-2   confirmed: batch-composition independence stays EXACT regardless.
9.6-3   confirmed: M5 replay-recorded-ids untouched.

--- 9.7  Referent drift ----------------------------------------------
9.7-1   FREEZE b10037 for parity/classification, life of Phase 1. New model
        needing a newer llama ⇒ second pinned referent for that model only;
        banked data never re-graded. (Perf referents refresh per 10.2.)

--- 9.8  If the oracle is never authorized again ---------------------
9.8-1   YES — [2v2] is oracle-free by construction; M3 (and M4–M6) close
        without it. M7's forward parity likewise (§8.8); only its tokenizer
        gate uses llama-tokenize (CLI, not server).

--- 10  FORWARD RULINGS, M4-M7 ---------------------------------------
10.1    Q8-KV bound: ON↔OFF within-stoke per §6 w/ its own entry-probe τ;
        zero BUG; class ≤ 19/50 prompt rate AND ≤0.5%/token (the M2 census
        base). Exceeding = results-vs-physics memo, not a grind.
10.1b   YES — (d)-shaped regardless of Q1.
10.2    YES — STANDING for all milestones, with the asymmetric guard:
        harder re-anchors apply silently; easier ones need sidecar + memo
        BEFORE judging. (K6 executed the harder case honestly.)
10.3    N re-measured per milestone (K5 proved the optimum tracks the
        kernel's bottleneck, 6→8); budget re-anchors with it per 10.2.
10.4    gate items: 11-invariant sim ×1e6 ASan (teardown + admission +
        runtime-variable pool exercised IN it); batch-composition identity
        {1,2,4,8,16} EXACT; prefix-dedup correctness; Q8-KV per 10.1;
        static N-curve GATE-grade monotone-within-spread (genuine inversion
        = physics memo pre-stamp). Constraints (not gates): block=16,
        kv_view body-swap, budget-note-first entry. Stamp = §7.5 shape,
        tag m4.
10.5    YES — generalized standing (miss ⇒ attribute ⇒ physics memo ⇒
        route per 10.12; build continues on non-dependent work). K3
        retroactively satisfied: §3 is that memo's disposition.
10.6    acked — Q8-KV default at M5 is Kabir's.
10.7    the smoke is THE definitive item but one of several: smoke +
        openai-python + ABI checklist + C++ TU + error-model conformance +
        compat traps + matrix/tsan (§8.7). Tag m6.
10.8    M7: tokenizer gate M1-style byte-exact (UNCHANGED, llama-tokenize);
        forward parity = [2v2]'s shape re-instantiated (twins + ref probe
        grid incl. ≥1k bands + four-delta targeted tests + τ re-frozen from
        M7's own entry probe); oracle = survey only. NEVER a 50/50
        live-oracle bar for any model, this one or the next.
10.9    τ = per-band frozen constants, not global, not a formula. Bands now
        [0,100)/[100,300)/[300,600); collapse iff adjacent p95 growth
        ≤1.5×. M7 trigger (named): entry probe adds [600,1200)/[1200,∞) and
        re-freezes before grading; first suspect if growth appears =
        rope_neox theta accumulation (logged 2026-07-31).
10.10   resolved: perf baselines REFRESH (that is 10.2); parity oracle
        stays PINNED b10037; per-model pin if a new arch needs newer llama;
        two referent kinds, two rules.
10.11   RATIFIED + two additions: entries carry number+location+date; on
        any miss the headroom ledger is the FIRST shopping list; carried
        lists restated at every milestone close.
10.12   (iii) with carve-out: proceed + open/append FOR-FABLE without
        stopping, EXCEPT never run a gate leg or bank evidence whose
        meaning the suspect item defines until ruled. The accumulating
        FOR-FABLE file mechanism is ratified as standing.

--- EVIDENCE REQUESTED -----------------------------------------------
        measurement: the §2.3 probe grids (both files) and the completed
        §3.7 sweep — already REQUIRED as [2v2-ii]/[2v2-iii], protocol
        specified to execution grade in §2.3.
        conditional ruling per plausible result: pre-ruled in full — §7.1
        A–D, §7.3 A–C, plus tripwires (§2.2) and τ-freeze ordering (§2.3).
        Nothing else requested; no second launch is expected for M3, M4,
        M5, or M6, and M7's single named trigger is §8.9's long-band
        re-freeze.
```
