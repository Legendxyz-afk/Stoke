# RULING — [2v2-iii] is restated as a referent-per-file requirement: Q8_0 is DISCHARGED by the sweep, Q4_K_M gets a fixed budgeted grid, and numpy was never required by name

**Status: binding ruling. Fable 5 session, 2026-08-06, Kabir-launched (deliberate
`/model` switch, same day the second Opus session closed [2v2-ii]).** Question of
record: `2026-08-05-m3-probe-grid-cost-FOR-FABLE.md` including §8A/8B/8C and the
2026-08-06 log entries (unmodified except its ANSWERED stamp). Read before
ruling: the full question memo, the 2026-08-04 ruling
(`2026-08-04-m3-gate-item-2-RULING-FABLE.md`, "the ruling" below), the p41
resolution (`gate-reports/2026-08-06-m3-p41-escalate-resolution.md`), the ledger
NEXT block, and `ref_forward.py`'s probe implementation (flags verified against
source, not the memo's description).

**The filled checklist is §13. The prose is the reasoning; the checklist is the
mechanical extract. Where they could ever be read to differ, the checklist
governs — it is the part Opus executes.**

---

## 0. THE RULING IN ONE PARAGRAPH

[2v2-iii] never needed numpy; it needed an **independent, validated f32-class
referent per file** — the ruling's own τ table already binds ref↔HEAD and
M2↔HEAD as ONE f32↔intcore family, so §2.2's naming of `ref_forward.py` was
instrumental (the then-only such referent), not essential. For **Q8_0** that
referent is the archived M2 gate binary, and [2v2-ii]'s completed 50/50 sweep
plus the 50/50 probe pass **discharge [2v2-iii] in full** — every job the leg
had (bug check, τ freeze, §8.9's band answer) is done ~64× denser than the
specified grid, and the referent-substitution error is bounded at 1.4e-5, three
orders below the smallest frozen τ. For **Q4_K_M**, where no C f32 referent
exists, the referent stays `ref_forward.py` and the grid is **fixed herein**
(§5): six full-stream `--probe-all` runs, ~384 positions, two ≤4 h windows
inside §8.1's existing authorization (continuations, if any, under the same
cap), τ_Q4KM frozen from its own pair's agreeing positions before any
divergence is graded. The measured ~57–98 h cost
is accepted; the ruling's §2.3 cost line is retired. The stoke.exe freeze is
acked and **extended to the m3 tag**. G9/G10's classifier readings are
ratified, so the banked 19/19 needs no re-grade. The stamp list gains one step
(archive the gate binary) and otherwise stands.

---

## 1. THE OPERATIVE AMENDMENT — [2v2-iii] v2, the referent-per-file requirement

The ruling's §2.2 text for [2v2-iii] is **restated, not waived** (the same move
the ruling itself made on item [2], one level down):

> **[2v2-iii] v2.** For each gate file, HEAD's banked streams are checked
> end-to-end against an **independent, validated f32-class referent** — an
> implementation sharing none of the K-work's code, whose membership in the
> f32 reference class is on record at gate grade. Checks are
> position-by-position stream comparisons spanning the corpus's occupied τ
> bands — valid while the streams agree, with each first divergence graded
> (continuation runs may re-lock a stream past a divergence). Every
> divergence is classified per §6 under a τ frozen from the **same pair's**
> agreeing positions before any divergence is graded. **PASS = zero
> BUG-class.** Class flips are counted, with the tripwire scaled to the grid's
> position count at the original ≥4-per-55 false-fire calibration. The gate
> report names referents by sha256 and records each pair's τ table.
>
> **Referent of record, Q8_0:** `build/stoke-m2-gate-d2b17c53.exe` — validated
> by M2's own gate (ref probes + oracle survey + L2/L3 ladder) and agreeing
> with `ref_forward.py` at 1.4e-5 over top-5 (question memo §3.6).
> **Referent of record, Q4_K_M:** `ref_forward.py`, unmodified — the only
> f32-class implementation that reads K-quants (§2.4-a4's reasoning intact).

Unchanged and explicitly not reopened: zero-BUG-class; §6's classification
rule (except the two ratified readings of §9 and the cross-class ladder
restatement of §11, both of which are how it was actually applied); τ's
definition max(0.10, 2×band-p95); the oracle's survey status; the frozen
50+10 corpus; the push policy; the owner line.

## 2. G1 — the cost is ACCEPTED as measured; the ruling's §2.3 cost line is RETIRED

Two probed points on a disclosed window, a linear fit, the cause located in
`ref_forward.py:183-187` as a deliberate RAM-bound trade-off, and a quiet-box
corroboration from the source's own figure — with the verdict robust across
the entire window range. That is measurement done to house standard, and the
2026-08-06 entry's retraction discipline (§4's rate re-confirmed at 1.56–1.68
s/position after its own challenge was withdrawn) raises confidence rather
than lowering it. **~49 h/file loaded, ~29 h/file quiet, is accepted. The
ruling's "~10–20 min/file" line priced a probe as one forward and is retired
as a costing error of ~65–300×.** No re-measurement is owed. The full
specified grid is dead at every plausible window — including §8A's ~35 h
variant — and nothing below authorizes it.

## 3. G2, G3, AND THE SCOPE QUESTION — Q8_0 is discharged; numpy was never required by name

**G2: the Q8_0 referent is the M2 binary.** Three grounds, in strength order:

1. **The ruling already treats the two referents as one family.** Its τ table
   row reads "f32 ↔ integer-core **(ref↔HEAD, M2↔HEAD)**, per file, per band"
   — one row, one τ, both pairs — and §2.3's ordering rule says in terms
   "same τ family — M2↔HEAD and ref↔HEAD are both f32↔intcore pairs; §3.6
   measured both at 0.0430 on the same position." The restatement in §1 makes
   explicit what the frozen table already implied.
2. **Independence is what the requirement is FOR, and the M2 binary has it.**
   [2v2-iii] guards against a K1–K6 defect. The M2 binary predates the K-work
   entirely, is frozen on disk, and its f32-class membership was established
   by M2's full gate — ref probe grid, oracle survey, ladder — not by one row.
3. **The substitution error is quantitatively negligible.** Ref and the M2
   binary agree at 1.4e-5 over top-5. The smallest frozen τ is 0.394898.
   No §6 verdict at any position can flip on a 1.4e-5 referent difference —
   it is three-plus orders below the grading resolution.

**G3: YES — [2v2-ii]'s completed evidence DISCHARGES [2v2-iii] for Q8_0 in
full, and no numpy probe is owed on top, at any density.** The leg had three
jobs (§2.2): the independent end-to-end bug check — done at every generated
position of all 50 prompts (~3,200 positions vs the specified 55), 19/19
ARITHMETIC-CLASS, zero BUG; the τ freeze — done (G6); §8.9's position-band
answer — done, three bands frozen. Each job was completed **denser than the
criterion specified**, against a referent in the same ruled family. Running
numpy on Q8_0 now would re-measure a composition already bounded at 1.4e-5
and could not change any verdict: density owed = **zero**.

**The scope question is answered NO accordingly:** [2v2-iii] requires an
independent validated f32-class referent per file, not a tool by name. Q8_0's
is the M2 binary (discharged); Q4_K_M's is `ref_forward.py` (G4/G5, still
owed). This resolves the remaining work to **Q4_K_M alone**.

## 4. G4 — the Q4_K_M referent is option (i): budgeted `ref_forward.py` with `--probe-all`

- **(iv) Q8_0-only stays declined** — §2.4-a4's reasoning was never about
  cost and is unchanged: the K-quant leg is exactly where an independent
  referent earns its keep.
- **(ii) the existing chain is recorded as supporting context, not the
  check.** Twins, K4 goldens, and the shared post-dequant path are real
  evidence, but they compose an argument; [2v2-iii] exists to be the
  *measurement* that argument predicts. It keeps that role.
- **(iii) dequant-to-F32 + C referent is DECLINED as YAGNI.** It requires a
  third engine: the M2 binary cannot be extended without becoming a new,
  unvalidated binary (its value IS that it is frozen), HEAD as its own
  referent is circular, so an F32-GGUF loader + f32 matvec path would need
  building and validating — days of work whose own correctness argument
  would lean on ref and the twins, i.e. it is not evidentially above option
  (i), only faster. And the speed is a one-milestone rental: **the moment
  `m3` is tagged, G8's archive step gives the K-quant path its M2-binary
  analog for free** — M4 regressions diff HEAD against the archived M3
  binary on Q4_K_M at C speed, full corpus, exactly as this milestone
  diffed against M2's.
- **(i) is chosen because §8A changed what it buys.** At 64 probes per
  prefill, a two-window budget yields ~384 stream-locked positions — ~7×
  the density the criterion originally specified per file — using the
  already-validated referent, zero new code on the engine side, inside
  §8.1's standing authorization.

One recorded caveat, stated so it is priced rather than discovered: ref's
K-quant **dequant** has no golden of its own (K4's goldens pin stoke's).
This is safe in the direction that matters — a ref-side dequant defect
cannot produce a false PASS, only spurious divergences, which grade toward
ESCALATE/BUG and trigger the §7.3-B bisection that would exonerate stoke via
the goldens. Agreement co-validates; disagreement investigates. The fail
direction is investigation, never a silent pass.

## 5. G5 — THE GRID, FIXED BEFORE THE RUN — mechanical, so Opus executes without judgment

**Prompts (6 prefills, chosen by stated rule, not taste):** shorts **p00,
p10, p20, p30** (every 10th) and **p45** (first of the post-long block);
long **p41** — the adversarial pick: the corpus's one demonstrated ESCALATE
position class on Q8_0, biasing the grid **toward** finding trouble.

**Runs (each ≤4 h, §8.1 discretion, announced at launch, detached):**

- **RUN A (~2.7–3.2 h):** `p00` twice — once bare, once with
  `--probe-all` + `--probe-pos 63` in the same invocation — then p10, p20,
  p30, p45 with `--probe-all`. The p00 pair discharges §8A's OWED item
  empirically: **id-stream equality bare-vs-flag, and bit-equality of the
  probe-all index-63 record against the probe-pos printout of the same
  run.** Any mismatch stops the leg (tooling defect memo before any use).
- **RUN B (~2.8 h):** p41 with `--probe-all` (~541 positions of prefill+64).

**Comparison semantics (why continuations exist):** `--probe-all` records
along ref's OWN greedy stream, so positions are comparable to the banked
Q4_K_M HEAD ids only while the streams agree. Per prompt: compare ref's
recorded argmax at each index against the banked id; positions 0..j−1 agree
by construction, index j = first divergence, graded per §6. **After each
first divergence, ONE continuation run is REQUIRED** — `--ids <prompt +
banked ids[0..j]> -n 64−(j+1)` with `--probe-all` — re-locking the stream
past j so the grid's position coverage returns to ~384 and the tripwire
denominator holds (j = 63 needs none). A second divergence inside a
continuation is graded too; recovering its tail is at executing discretion
within the window budget.
Expected load at the measured ~0.5%/token base: ~2 divergences, ~1 h of
continuations; a p41 continuation, if needed, is its own ≤4 h window.

**τ_Q4KM — frozen BEFORE any divergence is graded, from the pair's own
agreeing positions:** after runs A+B (and continuations), select **20
agreeing positions per occupied band** by rule — every ⌈N_agree/20⌉-th
agreeing position in (prompt-order, index-order) — fetch stoke's top-5 at
each via `stoke --probe-pos` on the frozen binary, compute the recorded
§2.3/§3.6 Δ-profile, freeze τ_Q4KM(band) = max(0.10, 2×band-p95), record in
the gate report. **The G5 grid's positions lie in [0,100) ∪ [300,600);
[100,300) hosts no G5 position and gets no τ_Q4KM.** Fail-safe: a graded
position landing in a τ-less band stops grading pending a freeze from added
samples of that band; a band with zero agreeing positions is a bug-hunt,
not a grading exercise. **Q8_0's τ values do not transfer to this pair in
either direction.**

**Classification:** §6 with τ_Q4KM, under the G9/G10 readings and §11's
cross-class ladder restatement. Stoke-side top-5 at divergent positions via
`stoke --probe-pos` (frozen binary — G11). BUG ⇒ §7.3-B hard block +
pre-authorized ladder bisection, unchanged.

**Tripwire, recalibrated with its derivation shown:** the original ≥4 flips
per 55-probe grid fires at ~0.015% false-alarm under the measured
~0.5%/token base (Poisson λ≈0.28). At ~384 covered positions λ≈1.9; the
matching threshold is **≥8 graded flips ⇒ stop, anomaly memo** (false-alarm
~0.06%, same order). The report states per-prompt covered positions so the
denominator is explicit, and reports flips against it.

**Ops constraints carried from the record:** launch condition free RAM >
model + 2 GB by `ullAvailPhys` (the guard's own number — quote no other);
never concurrent with anything mapping the Q8_0 file (the page-cache
alternation lesson); Task Scheduler launchers at Normal priority; every run
recorded (report + ledger line). Expected total ~6.5 h box time across 2–3
windows; worst realistic ~9.5 h across 4 — every window ≤4 h and announced.

## 6. G6 — the executed τ freeze is RATIFIED as the τ of record

τ was frozen 2026-08-06 from the M2↔HEAD pair's **31 agreeing-position
probes** — 0.519674 / 0.413566 / 0.394898 for [0,100) / [100,300) /
[300,600) — before any divergence was graded. That is the correct source
under this ruling: the pair being graded was M2↔HEAD, the τ-table row covers
that pair by name, the ordering rule (freeze before grading) was honoured,
and 31 agreeing samples is the same order as the specified grid's 55 probes
would have yielded. **Ratified as frozen; no re-freeze, no re-grade, no
numpy-sourced τ owed.** The full-corpus-envelope phrasing of the question
memo's §6 describes a denser freeze than was executed; the executed freeze
is sufficient and stands — the density difference moves no band constant's
legitimacy, and §8.9's collapse question is answered by the frozen values
themselves (adjacent-band growth ≤1.5× would permit collapse; the bands
stand anyway, per-band being the ruled form).

## 7. G7 — NO for M3; a conditional YES for M4+, evaluated at M7 entry

**For M3: NO.** Nothing left in M3 needs the speed (Q8_0 is discharged; the
G5 budget fits the unmodified tool), and the memo's own §3 reasoning is
endorsed: a batched prefill changes the referent's arithmetic, and §8.4
puts referent identity outside Opus's authority.

**For M4+: authorized under these conditions, all of them:**

1. **New tool, new name** (e.g. `ref_forward_batched.py`).
   `ref_forward.py` is never edited in place for the life of Phase 1 — it
   is the referent of record for every banked M3 number.
2. **Dual validation before any of its numbers are bankable:** (a) on
   Q8_0, teacher-forced top-5 agreement with the M2 binary at the 1.4e-5
   class on at least the §3.6 position set; (b) against unmodified
   `ref_forward.py`, **teacher-forced** top-5 logit agreement within ≤1e-3
   at every position of a fixed cross-file set (≥3 prompts per file, full
   streams forced along the banked ids) — the C-f32-vs-numpy-f32 bar
   applied, for once, to the pair it was calibrated for. **Never an
   id-stream-equality bar:** reassociated f32 legitimately flips near-ties
   at the measured ~0.5%/token base, and a free-stream bar would re-import
   the exact structural error the 08-04 ruling retired.
3. **Evaluation trigger, named:** M7 entry. §8.8's Llama-3.2 grid must be
   denser and reach ≥1k-token bands; at 18.4 s/position a single 1k prefill
   is ~5 h and the unmodified tool cannot pay for that grid. Build-or-not
   is decided there, against M7's actual corpus, not now.

## 8. G8 — the stamp list stands, with one addition and one substitution

**Substantively NO — the §7.5 sequence, its items, and 9.5-3 are
unchanged.** Two mechanical deltas:

1. **Substitution (the point of this ruling):** stamp item 2's "[2v2] per
   §2.2" is read under §1's restated [2v2-iii] v2 — the gate report records
   the Q8_0 leg as discharged by the sweep + probe pass against the M2
   binary, the Q4_K_M leg per §5, both τ tables, referents by sha256, and
   cites this ruling as authority.
2. **Addition:** at stamp time, **archive the gate binary** as
   `build/stoke-m3-gate-<shorthash>.exe` beside the tag, exactly as M2's
   was. That archive is what made G2 possible this milestone; M4 inherits
   the same dense-regression referent for BOTH files the moment it exists.

Forward notes ride to the M4-entry cluster, NOT the stamp: §6's ladder
clause restated per-pair for future pairs (§11 covers M3's remaining
grading), the max→mean ladder-statistic finding, the priority-7 launcher
default. None blocks the stamp; none is forgotten — §8.11's carried-list
rule applies.

## 9. G9 / G10 — both classifier readings RATIFIED

- **G9 — ESCALATE for the unobservable margin: YES.** BUG must be
  affirmative; when the other side's top-5 does not contain the token, the
  quantity §6 names is a bound, not a measurement. BUG only where the bound
  alone proves ≥2τ; otherwise ESCALATE, where the ladder — not a guessed
  number — decides. Escalation cannot pass a real bug; a guessed BUG would
  block a stamp on a number nobody measured. Correct on both edges.
- **G10 — CONTAINMENT for the top-5 test: YES.** §6's own BUG rationale is
  "a candidate the other implementation never ranked" — a containment test
  by construction. Strict equality would manufacture set differences out of
  the rank-5 boundary churn the τ tolerance exists to forgive.

**Consequence, stated so nobody re-derives it:** these are the readings
[2v2-ii] was graded under. Ratification confirms the banked 19/19 verdict
clean as graded — **no re-grade owed.**

## 10. G11 — the freeze is ACKED and EXTENDED to the m3 tag

Acked, and the terminus moves: not "until [2v2-ii] is banked" (already
passed) but **until `git tag m3` is cut.** Grounds: the 50 banked Q4_K_M
streams, every G5 stoke-side probe, and stamp item [4] ("matrix green at
the last source-touching commit") all key on the current binary
`876ce53f…` — a rebuild for any reason would strand all three at once, and
would be discovered after the fact via the sweep guard, not before. The
recoverable cost (~19 min re-bank) is not the point; silent evidence
invalidation is.

**One exception, so the freeze cannot deadlock its own purpose: §7.3-B's
fix-and-re-run cycle.** A BUG verdict is the one thing that legitimately
rebuilds — the fix lands, affected streams re-bank, affected evidence
re-runs, and the freeze re-binds to the new hash. On the expected zero-BUG
path the stamp-time hash is `876ce53f…` by construction, and G8's archive
is named from whatever hash the evidence actually closed under. After the
tag, the archived copy carries the referent role and HEAD is free to move
into M4.

## 11. THE CROSS-CLASS LADDER READING — p41's resolution ratified as the operative rule

§6's ladder clause is restated, binding for all remaining M3 grading and
carried forward: **at ESCALATE, the ladder's verdict is STEP ⇒ BUG /
no-STEP ⇒ ARITHMETIC-CLASS. The ≤1e-3 magnitude clause binds only
same-arithmetic-class pairs — its calibration domain (C-f32 vs numpy-f32,
worst layer 6.104e-04) — and is VOID for cross-class pairs, where it
returns FAIL identically for healthy and broken engines and therefore
carries no information.** For cross-class ESCALATEs, magnitude is recorded
as context against ≥2 agreeing-position control curves from the same pair,
mean_abs_diff quoted first (the extreme-value statistic's control spread,
1.94×, is nearly the effect size it is asked to detect; the mean separates
the same controls to 1.07×). The p41 resolution
(`2026-08-06-m3-p41-escalate-resolution.md`) is **ratified as the precedent
application**: §6 applied as written, §8.12's default path, nothing banked
under a criterion believed wrong, and **no further ruling owed before the
stamp on its account.** The full per-pair restatement of the clause for
M4+/M7 pairs rides in the M4-entry cluster (G8).

## 12. WHAT THIS RULING DOES NOT TOUCH — and the direction check

Untouched: the 2026-08-04 ruling everywhere not named here; M2's closed
gate; the frozen corpus; τ's definition and the frozen Q8_0 values (G6
ratifies, nothing moves); the oracle's survey status and Kabir's
`llama-server` authority; the push policy; the owner line (Opus 5 executes
M3→M7); Q8-KV, M4–M7 forward rulings, all §8 standing rules.

**Direction check, per house practice:** this ruling makes the Q8_0 leg's
evidence **denser** than the criterion specified (~3,200 positions vs 55)
against a referent the frozen τ table already recognized; fixes the Q4_K_M
grid at ~7× the specified density with its tripwire recalibrated to the
same false-fire probability, its τ frozen from its own pair, and its grid
published before it runs; declines to let cost lower §2.4-a4's bar (the
K-quant leg keeps an independent end-to-end referent); and extends a freeze
rather than relaxing one. One item got cheaper only because a stronger
referent already existed; nothing got easier to pass.

---

## 13. THE ANSWER SHEET — every row filled; this section governs

```
RULING — 2026-08-06 — [2v2-iii] referent and grid density
Authority: Fable 5 session, launched by Kabir 2026-08-06 (deliberate /model
switch; question of record = 2026-08-05-m3-probe-grid-cost-FOR-FABLE.md)

G1   YES — accepted as measured (~49 h/file loaded, ~29 h quiet; §8A's ~35 h
     variant included). The 08-04 ruling's §2.3 "~10–20 min/file" cost line
     is RETIRED as a ~65–300× costing error. No re-measurement owed; the
     full numpy grid is dead at every window.
G2   M2 binary (build/stoke-m2-gate-d2b17c53.exe), full corpus, every
     generated position. The ruling's own τ-table row already binds ref↔HEAD
     and M2↔HEAD as ONE f32↔intcore family; substitution error bounded at
     1.4e-5 — three orders below the smallest frozen τ (0.394898).
G3   YES — [2v2-ii]'s 50/50 sweep + 50/50 probe pass DISCHARGE [2v2-iii]
     for Q8_0 IN FULL: bug check (~3,200 positions vs 55 specified, 19/19
     CLASS, 0 BUG), τ freeze, §8.9 band answer — all denser than specified.
     Numpy probes owed on Q8_0: NONE, at any density.
G4   (i) — budgeted ref_forward.py with --probe-all. (iii) DECLINED (YAGNI:
     needs a third engine + its own validation, is not evidentially above
     (i), and G8's archive step gives M4 the C-speed Q4_K_M referent free at
     the tag). (ii) recorded as supporting context, not the check. (iv)
     stays declined per §2.4-a4. Ref's K-dequant caveat recorded: fails
     toward investigation, never toward a pass.
G5   FIXED (§5, before any run): prompts p00, p10, p20, p30, p45 + p41
     (rule-stated; p41 = adversarial pick). RUN A ≈2.7–3.2 h (p00 bare +
     p00 --probe-all/--probe-pos-63 discharging §8A's OWED item, then
     p10/p20/p30/p45); RUN B ≈2.8 h (p41). ONE required continuation run
     per first divergence (--ids prompt+banked[0..j], re-locking the
     stream; ~384 covered positions total). τ_Q4KM frozen BEFORE grading
     from 20 agreeing positions per occupied band ([0,100), [300,600);
     [100,300) hosts no G5 position), rule-selected, stoke side via
     --probe-pos on the frozen binary. Tripwire ≥8 graded flips
     (recalibrated: λ≈1.9 at 384 positions, false-alarm ~0.06% ≈ the
     original ≥4/55's ~0.015%). Ops: free RAM > model+2 GB by ullAvailPhys;
     never concurrent with Q8_0 mappings; Normal priority; each window
     ≤4 h, announced, recorded.
G6   RATIFIED — τ of record = the 2026-08-06 freeze from the M2↔HEAD pair's
     31 agreeing probes (0.519674 / 0.413566 / 0.394898), frozen before
     grading. No re-freeze, no re-grade, no numpy-sourced τ owed. Q4_K_M
     freezes ITS OWN τ (G5); constants never transfer across pairs.
G7   NO for M3. For M4+: YES under conditions — new tool, new name;
     ref_forward.py never edited in place in Phase 1; dual validation
     before bankability (vs M2 binary at the 1.4e-5 class on ≥ the §3.6
     set; vs unmodified ref: TEACHER-FORCED top-5 within ≤1e-3 on a fixed
     cross-file set — never an id-stream-equality bar); build-or-not
     decided at M7 entry against M7's corpus.
G8   Substantively NO. Two mechanical deltas: (a) stamp item 2 read under
     §1's [2v2-iii] v2, gate report citing this ruling, both τ tables,
     referents by sha256; (b) ADD: archive the gate binary as
     build/stoke-m3-gate-<shorthash>.exe beside the tag. Forward notes
     (per-pair ladder restatement, max→mean statistic, priority-7 default)
     ride to the M4-entry cluster, never the stamp.
G9   YES — ESCALATE for the unobservable-margin case; BUG only where the
     bound alone proves ≥2τ. Ratified.
G10  YES — CONTAINMENT for the top-5 test. Ratified. (G9+G10 are the
     readings [2v2-ii] was graded under ⇒ the banked 19/19 stands as
     graded; no re-grade.)
G11  YES — acked and EXTENDED: stoke.exe is frozen until `git tag m3`.
     Grounds: banked Q4_K_M streams, every G5 stoke probe, and stamp item
     [4] all key on 876ce53f…. Sole exception: §7.3-B's BUG fix-and-re-run
     cycle (rebuild ⇒ re-bank + re-run affected evidence, freeze re-binds).
     The archived copy (G8) carries the referent role after the tag.
SQ   NO — [2v2-iii] never required numpy by name; it requires an
     independent validated f32-class referent per file (§1). Q8_0 = the M2
     binary, DISCHARGED (G3). Q4_K_M = ref_forward.py, grid per G5.
     Remaining [2v2-iii] work = the Q4_K_M grid alone.

--- ratified beyond the checklist ------------------------------------
p41  the ESCALATE resolution is ratified as precedent (§11); the
     cross-class ladder reading (STEP decides; ≤1e-3 void for cross-class
     pairs) is binding for all remaining M3 grading. No further ruling owed
     before the stamp.

--- EVIDENCE REQUESTED -----------------------------------------------
     The G5 grid (runs A+B, continuations, τ_Q4KM freeze, classification,
     coverage bookkeeping), then the §7.5 stamp with G8's two deltas.
     Nothing else. No further Fable launch is expected before the m3 tag.
```

---

## 14. WHO DOES WHAT — the execution extract

| # | action | owner | blocking M3? |
|---|---|---|---|
| 1 | RUN A + RUN B + continuations (§5), announced at launch | Opus | yes |
| 2 | τ_Q4KM freeze (20/band, rule-selected) BEFORE grading | Opus | yes |
| 3 | Classify divergences per §6 + §11; tripwire watch (≥8) | Opus | yes |
| 4 | Gate report per G8-a; stamp sequence §7.5 + G8-b archive; tag m3, local, no push | Opus | — |
| 5 | FOR-FABLE ANSWERED stamp; ledger + memory updates for this ruling | Fable, this session | — |
| 6 | M4-entry cluster: forward notes (ladder restatement, max→mean, priority-7), carried headroom restated | Opus, M4 | no |
