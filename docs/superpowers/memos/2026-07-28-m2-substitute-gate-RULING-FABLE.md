# FABLE RULING — 2026-07-28 — M2's substitute gate: subset + pass criterion

**Authority:** Fable 5 session, launched by Kabir himself 2026-07-28 (per the
07-28 routing: major decisions go to Fable; Kabir opens the session — this is
that session). Answers `2026-07-28-m2-substitute-gate-definition-FOR-FABLE.md`
(committed `f6300e6`). Opus implements; the bar below moves only on a fresh
Fable ruling. Nothing here restarts any run — **everything stays paused until
Kabir's word** (his 2026-07-28 ~15:45 pause stands).

**Disclosure, recorded for the integrity trail:** the ledger intended the
criterion to be fixed blind, but the FOR-FABLE brief's §2 disclosed one
outcome — p48 C-vs-ref EXACT 64/64. That datum is maximally strict-compatible:
knowing a prompt was *exact* can only bias toward demanding exactness, never
toward loosening. The criterion below is what the evidence-blind reasoning
yields; the disclosure changed nothing. (The banked p48 entry is orphaned under
a stale `ref_forward.py` hash anyway and re-runs with the rest.)

**Relation to the Sonnet draft (INPUT ONLY, per the brief):** the subset and
the four waiver conditions are **RATIFIED** — independently re-derived below,
not rubber-stamped; the cost arithmetic was re-checked exactly (3,503
positions; 13.4 h serial; 124-min makespan floor). Everything else is
sharpened or added: the calibration argument for ε and for the budget (the
draft asserted both numbers without deriving them), stop-index equality,
first-divergence-only adjudication, the no-chaining rule for replacements, the
pinned L2 inputs, the reserve banking path, the p04 transitivity citation, and
the provenance fix in `ref_gate.py`.

---

## 1. SUBSET — ratified: the 24 as launched

`p01,p02,p05,p08,p09,p11,p12,p16,p20,p24,p25,p26,p28,p29,p37,p39,p41,p43,p44,p45,p46,p47,p48,p49`

**The rule behind them:**
- **All 19 oracle-diverging prompts.** Two independent reasons. (a) They are
  the near-tie-dense set — the places where a real C bug could hide behind
  "it's just a tie", so they maximize bug-detection power per ref-second.
  (b) They are exactly where M2's close-out claim — *the oracle divergences
  are arithmetic-class effects, not stoke bugs* — needs direct evidence: if
  C-vs-ref is exact precisely where C-vs-oracle diverged, the divergence is
  attributable to the oracle's Q8-activation arithmetic, not to stoke's C.
  p28 stays in: its oracle-side record is unreadable (coalesced-entry
  read-out limit) but C-vs-ref is a clean, well-defined comparison — this
  gate is the only M2 evidence that prompt can get.
- **One oracle-exact control per short tier** (p11 prose, p16 code, p26
  multilingual, p37 math, p45 adversarial). Controls test ref as much as
  stoke — a divergence on a confident-distribution control would point at a
  systematic/tier problem rather than a near-tie, and that separation is
  worth ~1.6 h serial. p26's aligner fatal is oracle-tooling-side and cannot
  affect this gate (no aligner in the C-vs-ref path).
- **p04 excluded by recorded transitivity, and the gate report must cite
  it:** W1 recorded ref==oracle 64/64 on p04, and the survey recorded
  stoke==oracle on p04 ⇒ stoke==ref on p04 is already proven evidence.
  Re-running it adds zero information; citing it makes prose effectively
  two-control.
- **No long-context control**, ratified on the makespan argument: the three
  ~2 h diverging prompts (p41/p43/p44) fill exactly 3 RAM-bound slots in one
  round and already cover the tier; a 4th ~120-min job forces a second round
  for one data point M3's full 50-prompt oracle gate re-covers anyway.

**Cost (verified against the brief's probed constants, not re-estimated):**
1,967 prompt tokens + 24×64 generated = **3,503 ref positions** × 13.8 s =
13.4 h serial, **~4.5 h wall at `--jobs 3`**, makespan floor p41 ≈ 124 min.
stoke's side is fully cached and hash-verified (`d2b17c53…`) — the run is
ref-only. The launched task `stoke-m2-ref-gate` already carries exactly this
subset (`run_ref_gate_detached.cmd` line 56 checked verbatim), so **resume
requires no reconfiguration**.

## 2. PASS CRITERION — mechanically checkable

**PASS requires ALL of:**

1. **Zero unresolved tooling errors.** Any timeout at the derived ceiling,
   fatal, or cache-guard trip (`stoke_hash` mismatch) is a tooling event, not
   an adjudicable result: fix the tooling, re-run that prompt. The gate can
   never pass around an unresolved error.
2. **Token-exact on every subset prompt** — identical id sequence AND
   identical stop index (early EOS on one side only IS a divergence at that
   position) — **except at most 2 prompts waived** under §3, each replaced by
   the next reserve in order, **every replacement itself token-exact with NO
   waiver eligibility** (no chaining: a flip in a replacement fails the gate
   outright).
3. **The L2 ladder re-run at the gate HEAD as part of this evidence:** same
   procedure and prompt as the W3-landed evidence (ids
   `[785,6722,315,9625,374]`; `stoke run --dump` vs
   `ref_forward.py --dump-layers`, last prompt position), adjudicated by the
   committed `l2_diff.py`: **all 36 layers max-abs-diff ≤ 1e-3 AND no STEP
   flag.**

**A divergence is waivable iff ALL FOUR conditions hold at the
first-divergence position p — evaluated there and only there; the post-flip
suffix is definitionally incomparable (different contexts) and is never
evidence of anything:**

1. p is identified (trivial on this pair — one id per line, no coalescing).
2. Both sides' top-2 are the SAME PAIR, SWAPPED.
3. **BOTH** sides' top1−top2 gap < ε = 1e-3 nats (a raw logit difference IS
   the nats gap — the softmax normalizer cancels in the subtraction).
4. Ranks 3–5: identical ids in identical order on both sides.

These are exactly the conditions already implemented in
`ref_gate.py::ref_waiver_decision` — **ratified as built**, including the
substitution of the original hidden-state-diff condition (unimplementable:
`stoke run` accepts no token-id input, and rebuilding stoke.exe forks the
hash the evidentiary chain is pinned to).

**Calibration — why ε=1e-3 and why ≤2, on the record this time:**
- The only mechanism that can legitimately flip C-vs-ref is f32
  summation-order noise, and it is *measured*: the L2 ladder bounds it at
  6.1e-4 in hidden state (worst layer, L34). A flip at a gap ≥1e-3 on either
  side would require noise beyond the measured envelope — that is evidence of
  something more than non-associativity, and it must fail. ε=1e-3 is thus
  doubly anchored: it is §8's standing constant reused verbatim (no new magic
  number), and it sits just above the measured noise ceiling.
- Budget ≤2 is calibrated by measured near-tie density, not copied blindly:
  the survey saw exactly **1 of 3,200 positions** with a sub-ε gap. Over the
  subset's 1,536 generated positions that predicts ~0.5 sub-ε near-ties, each
  flipping only if the noise lands the wrong way — expected flips ≲ 0.25, so
  0 waivers is the likely outcome, 1–2 is plausible physics, and **3+ means
  something systematic and MUST fail** (a real C bug would not stop at two).

**FAIL on any of:** >2 flips (even individually benign — budget exhaustion
contradicts the rare-physics story); any flip failing any condition; any
probe unobtainable on either side at p (fatal or position never generated —
absence of evidence defaults to FAIL, never a free pass); any replacement
diverging at all; any L2 layer >1e-3 or any STEP; any unresolved tooling
error. **A FAIL does not reopen this definition in-session — the bar is
immutable to the executing session under any reasoning; see §3.4.**

## 3. IF A FLIP OCCURS — pre-committed handling

1. The run never adjudicates (as built — it only records comparisons). Other
   prompts continue; the cache is per-prompt.
2. **Evidence:** for each diverged prompt, after the run completes (RAM: a
   probe is a full ref process), the manager obtains both-side top-5 probes
   at the first-divergence position p: `stoke run … --probe-pos p` (pinned
   binary — `--probe-pos` is already in it, NO rebuild) and
   `ref_forward.py --ids … --probe-pos p`. Costs, so nobody is surprised:
   stoke probe re-generates fully (short prompt ~7 min, long ~47 min at
   5.2 s/fw); ref probe = (tokens+p)×13.8 s, worst ~2 h on a long prompt.
   Probes are deterministic; record both 5-line blocks verbatim in the gate
   report the moment they are obtained.
3. **Adjudicate mechanically** via the committed
   `ref_gate.py::ref_waiver_decision` — never a re-derivation — and record
   its full output dict. Eligible + budget available ⇒ WAIVED; draw the next
   reserve **in order** (r00 first, no selection), bank its stoke ids per
   §4.2, run it through the same gate; token-exact required, no waiver
   eligibility.
4. **Any condition fails, or budget exceeded ⇒ GATE FAIL.** No in-session
   adjustment of ε, budget, subset, or conditions under any reasoning — a bar
   rewritten after seeing results is the gate-loosening the governance
   forbids. Escalation per standing routing: STOP, write the decision brief
   (probe blocks verbatim, L2 state, census context, the failing condition),
   park it in the repo, **Kabir launches the Fable session himself — never
   self-dispatch** (the 07-28 `model:"fable"`-silently-ignored pitfall is on
   record).

## 4. PREREQUISITE TOOLING — concrete requirements

1. **Adjudication wiring.** `ref_waiver_decision`, `WaiverLedger`,
   `stoke_probe`/`ref_probe`/`parse_probe` exist in `ref_gate.py` but are not
   invoked from the run path (correct — measurement and adjudication are
   separate). Requirement: adjudication MUST execute the committed
   `ref_waiver_decision` (a small `--adjudicate` mode reading the report +
   cache is the clean shape; manager-driven invocation of the same functions
   is acceptable). Either way the gate report carries the full decision dict
   + both verbatim probe blocks per divergence.
2. **Reserve banking.** Reserves r00–r09 have no cached stoke ids. A
   replacement's stoke ids MUST be produced through the same hash-guarded
   `.gate_cache` path as the 50 (`parity_gate.py --only rXX`; if `--only`
   does not accept reserve ids, extend it — **never** an ad-hoc uncached
   run), so `ref_gate.py`'s `read_stoke_ids` guard applies to replacements
   identically. Note this path starts the oracle briefly; that is harmless
   surplus data, and it happens only under an actual waiver.
3. **Freezes until the gate report is stamped:** NO `stoke.exe` rebuild for
   any purpose (sha256 `d2b17c53…` is the evidentiary pin; `--probe-pos`
   already exists in it); NO edit to `ref_forward.py` (its sha256 is in the
   cache key — the misleading 10–11 s comment fix stays deferred, as the
   ledger already ruled). `ref_gate.py` itself is not cache-keyed; comment
   edits there are safe.
4. **Provenance fix, ordered now:** `ref_gate.py`'s waiver block header
   claims "Fable's 2026-07-28 ruling, transcribed" — written when the ruling
   was still the mis-attributed Sonnet draft. Point it at this memo. (Done in
   the same commit as this ruling.)
5. **L2 leg:** `l2_diff.py` committed; inputs pinned in §2.3. Run it AFTER
   the ref run completes (RAM: `--jobs 3` saturates the box; the L2 ref
   process needs its own 2.3 GB).
6. **The run itself:** scheduled task `stoke-m2-ref-gate` as registered —
   subset verified identical to §1, `--jobs 3`, resumable per-prompt cache.
   **Start it only on Kabir's explicit word.**

## 5. WHAT THIS GATE DOES AND DOES NOT PROVE (for the M2 close-out record)

PASSING proves stoke's C11 scalar forward pass is computation-for-computation
equivalent, at greedy-argmax granularity over 64-token generations, to an
independent pure-f32 implementation of the same spec (numpy ref_forward), on
the 24 most adversarial prompts of the M2 corpus — including every prompt
where stoke diverged from llama.cpp — with per-layer hidden-state agreement
within 1e-3 and no structural step, any residual token flips individually
certified as sub-ε f32 tie-noise. Combined with the recorded ref↔oracle
anchor (p04 64/64) and the census's 31/50 stoke↔oracle exact prompts, it
supports the attribution that all 18 classifiable oracle divergences are
arithmetic-class effects (Q8 activations vs f32), not implementation bugs. It
does NOT prove token parity with llama.cpp (deferred to M3's integer core,
where the 50-prompt oracle gate re-runs against this census as baseline); it
cannot exclude a spec misreading faithfully shared by both implementations
(mitigated only by the oracle anchor and retired by M3's gate); and it says
nothing about performance or about sampling beyond greedy argmax.

---

**Execution order for the Opus session (all of it AFTER Kabir's word):**
resume `Start-ScheduledTask -TaskName 'stoke-m2-ref-gate'` (~4.5 h; monitor
`.ref_gate_cache/ref-gate-console.log` below the run marker) → L2 ladder
re-run (§2.3) → adjudicate any divergence (§3) → gate report
`docs/superpowers/gate-reports/2026-07-28-m2-ref-parity.md` (identity block:
stoke sha, ref sha, subset, this memo as criterion authority; p04 transitivity
citation; L2 table; waiver records verbatim) → playbook §5 checklist — **box 1
is superseded by this memo; the other boxes stand**, incl. the 4 pre-banked on
07-27 → close-out commit + `git tag m2` → ROADMAP M2 ✅ + memory owner line to
M3 (M3 inherits: the deferred oracle gate, the mandatory aligner fix, this
census as baseline).
