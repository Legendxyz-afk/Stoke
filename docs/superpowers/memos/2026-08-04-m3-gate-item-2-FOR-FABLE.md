# FOR FABLE — 2026-08-04 — M3's exit, the parity criterion, and every ruling that follows from it

**STATUS: OPEN — LIVE, ACCUMULATING. ONE LAUNCH, MANY RULINGS.**

**How to use this file.** Kabir runs Fable sparingly and by hand. So this is
not one question — it is **the whole decision package**: the primary question
(§5), the two other open decisions that were already Fable's (§6, §7), the
standing authorizations that stop Opus needing to come back (§8), a
**conditional-ruling tree (§9) covering every outcome the pending work can
produce**, and **forward rulings for M4 through M7 (§10)** so the rest of the
roadmap is decided in this launch too.

**Please rule on §9 and §10 even where the triggering evidence has not landed
yet.** An unruled branch is a second Fable launch. A ruled branch that never
fires costs nothing. Where a forward question truly cannot be answered before
its milestone, answer `DEFER` **and name the trigger** — that is a different
answer from `DECLINED`, and it tells Opus a later launch is expected rather
than leaving it to guess.

Sections §1–§4 and §12 are evidence and settled context — read them, but
nothing there is being asked. **§5–§10 are the ask**; §11 is what Opus still
owes; §13 is the fill-in answer sheet.

Append-only below §14; do not rewrite earlier entries.

---

## WHERE TO WRITE YOUR RULING — read before answering

**Write the ruling to a NEW file:**

```
docs/superpowers/memos/2026-08-04-m3-gate-item-2-RULING-FABLE.md
```

That is the house convention, matching
`2026-07-28-m2-substitute-gate-RULING-FABLE.md`,
`2026-08-03-k3-simd-headroom-RULING-FABLE.md`, and
`2026-08-03-m3-optimization-level-RULING-FABLE.md`. **The RULING file is the
authority. THIS file stays as the question of record and must not be
overwritten with answers** — the two are kept separate so a later reader can
see what was asked *and* what was known when it was asked.

**Then stamp this file's title line** the way M2's was, so nobody re-launches
a settled question:

> `# FOR FABLE — 2026-08-04 — … [ANSWERED 2026-08-04: Fable ruled — see`
> `` `2026-08-04-m3-gate-item-2-RULING-FABLE.md`, which is the authority] ``

**Structure the ruling against §13's checklist.** It lists every question ID
in this memo. **Answer every ID.** Where you decline to rule, write
`DECLINED` and why — **silence is the one outcome Opus cannot act on**, since
it cannot distinguish "no ruling needed" from "overlooked", and the whole
point of this memo is that a second launch is expensive.

**If you need evidence Opus has not gathered:** do not rule provisionally and
leave it hanging. Write the measurement you want under §13's `EVIDENCE
REQUESTED` heading — precisely enough to execute — plus **what you would rule
for each plausible result**. Opus runs it and applies your conditional. That
keeps the loop at one launch even when the answer depends on data.

**Authority boundaries, so the ruling lands where it can be acted on:**
`llama-server` starts are **Kabir's**, not yours (§1) — rule what happens
*when* he authorizes, not whether. Pushing to the remote is **Kabir's**.
Everything else in §5–§10 is yours.

---

## 0. THE ASK IN ONE PARAGRAPH

M3's gate item [2] — *"50-prompt parity re-passes on Q8_0 AND Q4_K_M"* — was
run, and leg 1 failed on its third prompt. **The failure is not a stoke bug
and is not caused by the K-work.** It is the same oracle arithmetic-class
divergence M2 documented, on one of the 19 prompts M2 already recorded as
diverging, and stoke's own trusted numpy reference **agrees with stoke and
disagrees with the oracle** at exactly that position. Item [2] was a
deliberate M2→M3 deferral resting on a stated prediction — that the integer
core would put stoke in the oracle's numeric class — and **that prediction has
now been implemented and measured insufficient.** The question is what M3's
exit criterion should be, given its justification has been measured away.

---

## 1. WHAT IS ALREADY DECIDED — not Fable's to revisit

- **M2's exit criterion.** Ruled 2026-07-28
  (`2026-07-28-m2-substitute-gate-RULING-FABLE.md`): a **substitute** gate,
  **C-vs-ref**, on a **24-prompt subset**, ≤2 waivers, 0 drawn. M2 closed
  2026-08-02, tag `m2` = `f730a64`.
- **The four waiver conditions and ε = 0.001 nats.** Same memo. §4 below only
  draws out a consequence of them; it does not reopen them.
- **The oracle invocation is frozen:** `-t 1 --parallel 1 -c 2048`, default KV.
  The `-c 2048` cap is approved (`2026-07-27-m2-oracle-context-cap.md`), the
  f32-KV hypothesis is refuted by experiment, and `-t 1` is a control, not a
  performance knob — llama.cpp's reductions are thread-count dependent, so
  changing it changes the baseline itself.
- **M3 gate items [1], [3], [4], [5] — all ✅.** Item [3]'s bar was
  re-anchored against a gate-time llama (8.08–8.46, **harder** than the quoted
  7.80) and passed both legs. None of that is in question here.
- **`-O3` for M3** — `2026-08-03-m3-optimization-level-RULING-FABLE.md`.
- **The STANDING COMPARABILITY RULE** (K3 ruling §4) binds every perf figure
  quoted in §6.
- **Starting `llama-server` is KABIR'S call, not Fable's.** Where §9 depends
  on oracle data, the ruling should say what to do *when* Kabir authorizes it,
  not authorize it.

---

## 2. WHAT HAPPENED

Kabir launched item [2] on 2026-08-04 11:36:39.

**Leg 1 (Q8_0): GATE FAIL, exit 1, finished 14:55:42** (~3 h 20 m), at `p02`
generated position 16, **0 of 2 waivers drawn**. Report at
`tests/tooling/parity_gate_dev_report.md`.

All 50 prompts were generated on both sides; the comparison loop then fails
loud at the first non-waivable mismatch. **So `p00` and `p01` are the only
prompts with a recorded verdict — the other 47 are untested, not passing.**

**Leg 2 (Q4_K_M): reached oracle 3/50, then STOPPED** — not failed — ahead of
a planned laptop shutdown. A recorded run is `--clean` and cannot resume, so
nothing was lost.

---

## 3. THE EVIDENCE

### 3.1 The failing position

| | token 714 | token 438 | gap (nats) |
|---|---|---|---|
| oracle (llama.cpp b10037, `-t 1`) | **−1.1327481** | −1.1336979 | 0.00095 → picks **714** |
| stoke HEAD (logits) | 19.721512 | **19.785263** | 0.06375 → picks **438** |

Top-2 logprob differences equal logit differences (softmax is shift-invariant),
so both gaps are in nats and directly comparable.

### 3.2 The K-work did not cause it — bisected against binaries on disk

| build | picks | 438−714 gap | RMS err vs oracle, top-5 |
|---|---|---:|---:|
| `stoke-m2-gate-d2b17c53.exe` — **the binary that passed M2** | **438** | +0.096916 | 0.09514 |
| `stoke-k2-intcore-2ed3d1e8.exe` — integer core | **438** | +0.063751 | 0.06669 |
| `build/win/stoke.exe` HEAD `876ce53f…` — the gate's own binary | **438** | +0.063751 | 0.06669 |
| oracle | 714 | −0.000950 | — |

**M2's binary picks the same token and is FURTHER from the oracle than HEAD.**
The integer core cut the error ~30% (0.0951 → 0.0667) and **K3/K4/K5 changed
this path not at all** — K2 and HEAD are bit-identical here. Error signs are
mixed (+0.065, +0.045, +0.102, −0.033) with no common scale: accumulated
arithmetic-class difference, not a broken scale factor or a transcription bug.

### 3.3 Determinism and thread-invariance both hold

HEAD reproduces the gate's five logits **bit-exactly** on re-probe, and is
**bit-identical at 1, 2, 8 and 16 threads**. K5's row-parallel pool is exact
and is not a suspect.

### 3.4 p02 is a KNOWN, DOCUMENTED oracle divergence — from M2

`tests/data/parity/p02.txt` has exactly one commit (`f5d99ee`, corpus
creation): the prompt is unchanged since M2.

`2026-07-28-m2-substitute-gate-definition-FOR-FABLE.md:114` lists **"The 19
diverging": `p01 p02 p05 p08 p09 p12 p20 p24 p25 p28 p29 p39 p41 p43 p44 p46
p47 p48 p49`.** p02 is on it. The 24-prompt subset was built as *"All 19
oracle-diverging prompts"* plus controls — **p02 is in that subset precisely
because it already diverged from the oracle.** Its `✅ 64/64` in the M2 gate
report is a **C-vs-ref** result, not an oracle result.

### 3.5 The deferral's premise is implemented — and it was not enough

Item [2] is **not a drafting slip**. `ref_gate.py`'s header records a
deliberate deferral with a stated mechanism:

> *"M2's token gate is re-scoped to C-vs-ref, and ORACLE token-parity is
> **DEFERRED TO M3, where LOCKED decision #2's integer core puts stoke in the
> oracle's own numeric class** and the forward note already mandates
> re-running the 50-prompt oracle gate."*

`2026-07-20-m2-oracle-near-tie-policy.md` names the mechanism it bet on: *"the
numeric gap is the oracle's activation quantization (it quantizes activations
to Q8 blocks for integer dot products; not switchable by any flag)."*

**That mechanism is implemented.** `src/kernels/vec.c:162` calls
`quantize_row_q8_0(x, n_in, aq_scratch)` before the integer dot.

**And the divergence survived it** (§3.2). The bet was placed, the work was
done, and **the premise is measured false — not in principle, but in degree.**

The reason, once stated: the integer core puts the **matmuls** in the oracle's
class. RMSNorm, QK-norm, rope, attention scores, their softmax, and the
residual accumulation are all still f32 on both sides with different
association order. Near-ties at the ~1e-3 nat scale live below that residual
noise floor, and no amount of matmul agreement reaches them.

**Calibration, from the same memo — leg 1's failure is PREDICTED, not
surprising:** measured flip rate *"1 flip in 192 validation tokens
(~0.5%/token)"*, and **19 of 50 prompts diverged.** A 50×64 gate is ~3,200
token decisions; even after a 30% error reduction that is of order ten
expected flips. **Failing on the third prompt is what this project's own
recorded statistics predict.** A 50/50 pass needed the error at essentially
zero, not down 30%.

### 3.6 THE REFERENCE AGREES WITH STOKE, NOT WITH THE ORACLE

`ref_forward.py` run on p02's 23 tokens (from `stoke tokenize`; count matches
the M2 manifest) with `--probe-pos 16`:

| implementation | numeric class | picks | 438 | 714 |
|---|---|---|---:|---:|
| `ref_forward.py` (numpy f32) | pure f32 | **438** | 19.839981 | 19.743076 |
| `stoke-m2-gate` binary | pure f32 | **438** | 19.840002 | 19.743086 |
| HEAD | integer core | **438** | 19.785263 | 19.721512 |
| oracle | Q8-activation integer | **714** | (−1.1336979) | (−1.1327481) |

Max |Δ| between implementations across the top-5, relative to token 714:

| pair | max Δ (nats) |
|---|---:|
| ref vs M2 binary | **0.0000140** |
| ref vs HEAD | 0.0430110 |
| M2 binary vs HEAD | 0.0430160 |
| **ref vs ORACLE** | **0.1427390** |

**Read the last row against the ones above it.** The project's own reference —
which sits **above** stoke in the documented trust order, and whose
correctness was established by reproducing the oracle's greedy ids on real
prompts — **disagrees with the oracle by 3× more than it disagrees with
stoke's integer core.** All three implementations pick 438. **llama.cpp is
the outlier, by 0.00095 nats.**

This is M2's close-out claim reproduced on M3's binary, in M2's own words:
*the oracle divergences are arithmetic-class effects, not stoke bugs.*
**There is no stoke bug at p02:16 to fix.**

**ref vs M2 at 1.4e-5 validates the comparison itself** — it lands where
`ref_gate.py` says it must ("only float32 non-associativity"; W3's L2 ladder
bounds it at 6.1e-4), which is what makes the 0.143 row credible rather than
a tooling artefact.

**Also measured: HEAD's token stream matched ref's EXACTLY for all 17
generated positions of p02**, the 0.043 logit gap notwithstanding.

### 3.7 M2-vs-HEAD full-corpus token sweep — ⏳ RUNNING, result pending

Leg 1 banked HEAD's 64 ids for all 50 prompts (all verified under HEAD's
sha256). The M2 binary is on disk. So *"did K1–K5 change ANY token this
corpus produces?"* is answerable **with no oracle at all**.
**§9.1 pre-rules every outcome.**

**PARTIAL — 6 of 50, stopped at 15:51 for a planned shutdown.** The M2 binary
runs ~147–214 s/prompt (pre-K5, single-threaded f32), so the full sweep is
~2–3 h and will be resumed.

| prompt | on M2's 19-diverging list? | M2 vs HEAD |
|---|---|---|
| p00 | no | **IDENTICAL** |
| p01 | **yes** | DIFFERS at 26 |
| p02 | **yes** | DIFFERS at 41 |
| p03 | no | **IDENTICAL** |
| p04 | no | **IDENTICAL** |
| p05 | **yes** | DIFFERS at 14 |

**6 of 6 consistent with §9.1 branch (B):** every difference falls on a prompt
M2 had ALREADY recorded as oracle-diverging, and every prompt outside that
list is token-identical between the M2 binary and HEAD. **No prompt outside
the 19 has moved.**

⚠ **Six points is a pattern, not a result** — 44 prompts remain, the long-
context tier (p40–p44, 447–490 tokens) is untouched, and that tier is exactly
where the position-growing `rope_neox` term of §10.9 would show up first. **Do
not rule 9.1 as though (B) were established;** rule all four branches as asked.

Note p02 differs at position **41**, not the 16 the gate failed on — consistent
with §3.2/§3.6, where M2 and HEAD both choose 438 at position 16 and only
split later.

---

## 4. THE STRUCTURAL POINT — this class cannot be waived, ever

The waiver requires **both** engines' top-2 gaps < ε. At p02:16 the oracle's
gap is 0.00095 (a genuine near-tie, under ε) and stoke's is 0.06375 (67×ε).

So *"the oracle is at a coin flip, and stoke has ordinary arithmetic-class
error"* is **non-waivable by construction — and gets no more waivable as
stoke gets more accurate.** Shrinking the error 0.0951 → 0.0667 did not help
and shrinking it further will not: the waiver forgives **mutual** uncertainty,
and stoke's *confidence* is what disqualifies it.

**Any criterion requiring 50/50 live-oracle agreement is asking for something
no correctness work can deliver**, short of stoke reproducing llama.cpp's
arithmetic bit-for-bit — which is not a goal this project has ever held.

---

## 5. QUESTION 1 (PRIMARY) — what is M3's exit criterion?

**The question is not whether item [2] was written in error.** §3.5 shows it
was a reasoned deferral. **The deferral made a testable prediction; the
prediction has been tested; it failed.** Fable is ruling on what to do with a
criterion whose stated justification has been measured away — not on whether
to forgive a number that came up short.

Options as Opus sees them. **Not a recommendation; not exhaustive — rule
something else if something else is right.**

### (a) Restate item [2] in M2's own ruled terms: C-vs-ref, extended to K-quants

- **For:** continuous with M2's ruled criterion; needs no oracle at all;
  `gguf_reader` already supports Q4_K/Q6_K (`DEQUANT_SUPPORTED` at
  `gguf_reader.py:116`) so the K-quant leg is coverable.
- **Against, and Fable must see this:** `ref_gate.py`'s justification —
  *"Both sides are pure float32, so unlike the oracle gate there is no
  numeric-class gap between them"* — **is now FALSE of HEAD.** ref-vs-HEAD is
  0.043, three orders of magnitude above the 1.4e-5 that sentence describes.
  Option (a) is a token gate **across a numeric-class gap**, with its own
  near-tie flip risk. **The ≤2 waiver budget and ε were calibrated for a gap
  that no longer exists.**
- **Sub-decisions (a) forces — please rule these too, not just (a):**
  - **(a1) Subset:** all 50, or M2's 24, or something else?
  - **(a2) Waiver policy for ref-vs-C:** keep ε=0.001/budget 2, or recalibrate
    against the measured 0.043 class gap? Note ε=0.001 was set when the two
    sides agreed to 1.4e-5.
  - **(a3) Cost is real:** ref measures **9.7 s/position** this session. M2's
    24-prompt subset is 3,503 positions ⇒ **~9.5 h serial**; all 50 is
    materially more. Is that authorized (see §8.1)?
  - **(a4)** Must the Q4_K_M leg use ref too, or is Q8_0-only sufficient given
    gate item [1] already proves scalar↔SIMD bit-exactness on all quant kinds?

### (b) Keep the oracle, restrict the corpus to the 31 non-diverging prompts

- **For:** keeps a live-oracle criterion, which is the only criterion that
  tests against an *independent* implementation rather than one of ours.
- **Against:** the 19 are the near-tie-dense set — *"the places where a real C
  bug could hide behind 'it's just a tie'"* (M2's own words). Excluding them
  removes exactly the prompts with the most bug-detection power per second.
- **Sub-decisions:**
  - **(b1)** What evidence, if any, must the excluded 19 still produce? (M2's
    answer was C-vs-ref on them; is that still required?)
  - **(b2)** The 19 were measured **on Q8_0**. **There is no equivalent list
    for Q4_K_M** — the K-quant file's diverging set is unmeasured. How is the
    Q4_K_M leg's corpus defined? (See §9.2, which cannot be resolved without
    this.)
  - **(b3)** If a prompt **outside** the 19 diverges, that is a new signal —
    §9.4 asks for the rule that classifies it.

### (c) Keep item [2] literally

- Then **M3 cannot close**, and the honest record is that it is blocked on an
  **unreachable criterion**, not on work. §4 is the argument that this is
  structural. If Fable takes (c), please say explicitly that M3 is blocked
  rather than failing, so the ledger records the right thing.

### (d) Something else

Including: a divergence-**counting** criterion (bounded rate rather than zero
divergences), which is what the forward note already mandates for Q8-KV at M4
(*"property-tested + divergence-counted, never token-parity-gated"*). If that
shape is right for KV, Fable may judge it right here. If so: **what bound?**
The measured base rate is ~0.5%/token and 19/50 prompts.

---

## 6. QUESTION 2 (SEPARATE) — K3's exit decision

**Deliberately kept distinct from §5. §5 is correctness; this is performance.
Please rule them separately so neither ruling can be read as covering the
other.**

Standing since the K3 ruling: K3 shipped and **missed its parity band** —
exit **1.437×**, total **×1.390**, against an expected **×2**. Of K3's
residual gap, **1.235× is measured as the price of gate [1]'s bit-exactness**
(three-way interleaved: bit-exact 8.29 GB/s, ggml-style 10.23, llama 15.76),
leaving **~1.16× unexplained**. Spending gate [1] is priced, not proposed.

Newer evidence that bears on it:

- **K4 entry (2026-08-04):** the K-quant penalty is **thread-dependent** — 6T
  retains only **79.1%** of the 1.714× size ratio (1T retains 95.5%), because
  at 6T Q8_0 sits at the bandwidth ceiling (57.4 GB/s) while Q4_K_M reaches
  only 46.0 — **the superblock unpack, not bandwidth, is binding.** The bar
  margin fell ~1.86× → **~1.47×**. A single-thread probe would have reported a
  reassuring 95.5%.
- **Per-op:** four-row batching gave attn_* 1.43–1.53× but ffn_up only 1.26×
  and **LOGITS 1.18×, and LOGITS alone is 10% of the token** — narrow-output
  matvecs benefit least from row batching, which a uniform "memory-bound"
  story does not predict.

**Ask:** does K3 exit as-is, or is the ~1.16× unexplained residual owed work
before M3 stamps? And is gate [1]'s 1.235× bit-exactness price accepted as
permanent, or revisited now that its cost is measured?

---

## 7. QUESTION 3 — the f32-reduction twins (≤1.3%)

Standing as Kabir's/Fable's, unresolved. **Ask:** are they in scope for M3, or
deferred to M4? If deferred, does anything about §5's ruling change that?

---

## 8. QUESTION 4 — standing authorizations, so Opus need not come back

- **8.1 — Compute budget.** May Opus spend a **~9.5 h** unattended reference
  run (option (a)) on Kabir's say-so alone, or does each such spend need a
  fresh ruling? Same question for any re-run after a `stoke.exe` change.
- **8.2 — Re-run triggers.** **Any change to `stoke.exe` changes the cache key
  and invalidates BOTH legs.** So a fix-then-re-gate loop costs a full gate
  each time. Is Opus authorized to re-run the ruled gate after any change it
  believes gate-relevant, without asking?
- **8.3 — Corpus.** Is Opus authorized to ADD prompts (e.g. to build the
  missing Q4_K_M diverging list, §5-b2), or is the corpus frozen at the
  M2-committed 50+10?
- **8.4 — What Opus must NOT decide alone.** Opus's read is: the criterion
  (§5), the stamp (§9.5), and anything that moves a bar. Confirm or correct.
- **8.5 — Not Fable's:** starting `llama-server` is Kabir's. §9 branches that
  need oracle data should say what to do **when** he authorizes it.

---

## 9. CONDITIONAL RULINGS — please rule these now, before the evidence lands

**Each branch below is a potential second Fable launch. Ruling them now
collapses them all into this one.**

### 9.1 — Outcomes of the running M2-vs-HEAD sweep (§3.7)

- **(A) All 50 token-identical.** ⇒ K1–K5 are **token-neutral on this
  corpus**; leg 1's failure is inherited from M2 wholesale, not just at the
  bisected position. **Ask:** does (A) alone discharge the "is the K-work
  correct" concern for M3, or is ref evidence still required on top?
- **(B) Some differ, all inside the 19.** ⇒ the K-work perturbs near-ties that
  were *already* near-ties. **Ask:** treated the same as (A)?
- **(C) Some differ OUTSIDE the 19.** ⇒ the K-work changed a token where M2
  was stable. **This is the branch that could indicate a real bug** — see
  §9.4 for the classification rule Opus needs. **Ask:** does (C) block M3
  pending investigation, or is it graded by §9.4's rule first?
- **(D) Sweep errors or cannot complete.** **Ask:** is it required evidence,
  or nice-to-have? (It needs no oracle, so it is repeatable cheaply — but it
  costs ~3 h each time.)

### 9.2 — Outcomes of leg 2 (Q4_K_M), whenever Kabir authorizes the oracle

- **(A) Passes 50/50.** Then the criterion problem is Q8_0-only, which would
  be surprising given the K-quant path has *more* arithmetic-class distance
  from the oracle, not less. **Ask:** is a Q4_K_M pass sufficient to stamp
  item [2] on its own, or does a Q8_0 failure block regardless?
- **(B) Fails on a prompt in the 19.** ⇒ inherited class, same as leg 1.
- **(C) Fails on a prompt NOT in the 19.** ⇒ **cannot be classified**, because
  **the 19 were measured on Q8_0 and no diverging list exists for Q4_K_M**
  (§5-b2). **Ask:** does Opus build that list first (costs a full oracle
  survey), or is (C) graded directly by §9.4?
- **(D)** The K-quant path has kernels Q8_0 does not (`get_scale_min_k4`, the
  Q6_K unpack, the Q8_K activation quantizer). **Ask:** on any K-quant-only
  divergence, is Opus pre-authorized to bisect scalar-vs-SIMD twin and
  Q4_K-vs-Q6_K without a further ruling? (Gate [1] already proves the twins
  bit-exact on 1e4 random tensors per kernel, so a twin split would itself be
  a finding.)

### 9.3 — Outcomes if option (a) is chosen and the ref gate is run

- **(A) Ref gate passes.** M3 closes on it, subject to §9.5.
- **(B) Ref gate fails.** **Both sides are ours**, so this is a much stronger
  bug signal than any oracle divergence. **Ask:** does a ref-gate failure hard
  block M3 (Opus's assumption), and is Opus pre-authorized to bisect it via
  `--dump-layers` L2/L3 without asking?
- **(C) Ref gate fails only at positions whose margin is below the measured
  ref-vs-HEAD class gap (0.043 nats).** ⇒ that is the *new* near-tie class
  option (a) creates. **Ask:** waivable, and under what ε/budget? (This is
  §5-a2, but it is the branch where it actually bites.)

### 9.4 — THE CLASSIFICATION RULE Opus most needs

Several branches above reduce to one unanswered question: **what
distinguishes an "arithmetic-class divergence" from a bug?** M2 asserted the
distinction but never reduced it to a test Opus can apply alone.

**Opus proposes, for ratification or correction:** a divergence is
**arithmetic-class** iff **both** hold at the first divergent position —

1. the **losing margin** (winner minus the token the other side chose) is
   **below τ**, and
2. the two implementations' **top-5 candidate sets are equal** (order may
   differ).

…and is a **BUG** otherwise, because a perturbation of the measured size
cannot move a decision whose margin exceeds it, nor introduce a candidate the
other side never ranked.

**Ask: ratify or correct, and set τ.** Measured inputs for calibrating τ:
ref-vs-HEAD class gap **0.043**; M2-binary-vs-oracle RMS **0.0951**;
HEAD-vs-oracle RMS **0.0667**; ref-vs-oracle max **0.1427**; the M2 survey's
observed oracle gaps in the diverging class **0.001–0.204 nats**. Note τ must
exceed the class gap or every divergence reads as a bug, and must sit below
the smallest *genuine* bug signal — Opus has no measurement of the latter and
does not want to invent one.

### 9.5 — M3 stamp and tag conditions

**Ask for the exact list**, so Opus never stamps on a judgement call.
Currently: [1] ✅, [3] ✅, [4] ✅, [5] ✅, [2] ruled by §5. Opus's assumption
is that on the ruled criterion passing, the sequence is: **gate stamp →
ROADMAP row → MEMORY.md build log → owner line M3→M4 → `git tag m3`**, all
local, **no push** (pushing is Kabir's). **Confirm, and state explicitly
whether §6 (K3's exit) and §7 (f32 twins) block the stamp or ride into M4.**

### 9.6 — Forward implications, so this ruling does not leak into M4/M5

- **Q8-KV at M4** is already specified *"property-tested + divergence-counted,
  **never token-parity-gated**"* — which is the same shape as option (d).
  **Ask:** does §5's ruling set the precedent for M4's KV criterion, or are
  they independent?
- **M4's batch-composition independence** (*"same seq ⇒ token-identical at any
  batch size"*) is a **within-stoke** identity, unaffected by any oracle
  divergence. **Ask:** confirm it stays an exact criterion regardless of how
  §5 is ruled — Opus believes it must, since it is the property M5's
  debuggability stands on.
- **M5's "replay recorded ids, never re-sample"** likewise. Confirm untouched.

### 9.7 — Referent drift

The oracle is pinned to llama.cpp b10037 by sha256. If it is ever upgraded,
**the oracle's own near-tie choices move**, and any banked oracle comparison
silently changes meaning. **Ask for a standing rule:** on any oracle-binary
change, is the whole parity criterion re-derived from scratch, or is the
pinned b10037 frozen for the life of the project?

### 9.8 — If Kabir never authorizes the oracle again

Plausible: it is a 3 h wall-clock, RAM-heavy, single-threaded job on his only
box. **Ask:** is there a ruling that lets M3 close on oracle-free evidence
alone (options (a)/(d))? Opus does not want to discover at stamp time that
every path requires a resource that is not forthcoming.

---

## 10. FORWARD RULINGS — M4 THROUGH M7, asked now so they are never asked again

**Kabir's instruction, 2026-08-04: rule the rest of the roadmap now.** These
are not hypotheticals invented to fill space — each one is a decision the
forward note already leaves open, or a tension between two things it says.
Where a question genuinely cannot be answered before its milestone's evidence
exists, **say so with `DEFER` and name the trigger that should reopen it**;
that is different from `DECLINED` and tells Opus a future launch is expected.

### 10.1 — The §5 question RECURS at M4, already in writing

M4's forward note specifies Q8-KV as *"property-tested + divergence-counted,
**never token-parity-gated**"* — i.e. option (d)'s shape is **already the
ruled approach for KV**. But **no bound was ever set.** A divergence count
with no threshold is a measurement, not a gate.

**Ask:** what is the acceptable Q8-KV divergence bound, and is it derived from
the same base rate §5-(d) would use (~0.5%/token, 19/50 prompts)? If §5 is
ruled (a)/(b)/(c), does M4's KV criterion stay (d)-shaped anyway?

### 10.2 — Is bar RE-ANCHORING a standing rule or a per-clause accident?

Gate [3] carried an explicit clause: *"if llama.cpp is re-benched at gate
time, the bar RE-ANCHORS to that number."* It fired this session and **made
the gate harder** (7.80 → 8.08–8.46), which is the direction that proves it
was applied honestly. M4's bandwidth budget has the same shape: **W is
measured at M4 entry, floor now ~54 GB/s GATE-grade, superseding M0's ~39.**

**Ask:** is *"a bar expressed as a fraction of a moving referent re-anchors at
gate time"* a **STANDING rule for every remaining milestone**, or does it bind
only where a clause says so? Opus's read is that it must be standing —
otherwise an aging referent silently clears a bar — but that is a rule about
all future gates and is not Opus's to declare.

### 10.3 — The bandwidth budget's N is not the core count

Measured: **W peaks at 6 P-threads; 10 threads measured 15% WORSE.** So the
budget's N is a measured optimum, not a machine property.

**Ask:** for M4's budget, is N frozen at the measured 6, or re-measured per
milestone? And if a later measurement moves it, does the budget re-anchor per
10.2?

### 10.4 — M4's exit criteria, enumerated now

The forward note names: block=16; `kv_view` body-swap only; **batch-composition
independence** (same seq ⇒ token-identical at any batch size); full-reservation
admission; **the 11 sim invariants asserted after EVERY op**; Q8-KV
divergence-counted; plus the 08-02 constraints (runtime-variable pool size;
sequence teardown as a first-class sim op; design note OPENS with the
bandwidth budget).

**Ask:** which of those are **gate items** versus design constraints, and what
is M4's exact stamp precondition list — the §9.5 treatment, but for M4, so
Opus never stamps M4 on a judgement call either.

### 10.5 — Bar-miss policy: already ruled for M5, unruled for everything else

M5's note says: if llamacpp-tuned parity clears but the 16.8@16
inversion-derived bar does not, *"that is a results-vs-physics memo to Kabir,
not a grind."*

**K3 has ALREADY missed a bar** (§6: 1.437× vs an expected ×2) and there was
no such rule to apply.

**Ask:** does M5's bar-miss protocol **generalise to every milestone**,
including retroactively to §6's K3 decision? Opus would rather apply one
consistent rule than adjudicate each miss.

### 10.6 — Acknowledge what is NOT Fable's here

Q8-KV's **default ON/OFF at M5 is explicitly Kabir's call.** Opus lists it only
so the ruling does not accidentally cover it. **Ask: acknowledge.**

### 10.7 — M6's exit criterion

The note calls out *"the definitive smoke: the M0 bench harness drives a
`stoke` adapter UNCHANGED"*, plus the ABI freeze checklist as a mechanical
pass, error handling already DECIDED, and the server's compat traps.

**Ask:** is the unchanged-harness smoke **the** M6 gate, or one item among
several? If several, the enumerated list, as in 10.4.

### 10.8 — ⚠ THE BIG ONE: M7 plans to oracle-gate a NEW model

M7's note specifies Llama-3.2 support with *"llama-bpe = second pre-tokenizer
regex entry, **oracle-gated like M1**"*, plus *"unsloth-file `stoke info` diff
then parity"*.

**If §5 concludes that live-oracle token parity is structurally unreachable
for Qwen3, then M7 is currently planning to gate a second architecture on
exactly the criterion §5 just found unreachable.** Llama-3.2 will have its own
near-tie population and its own arithmetic-class gap; nothing about a new
model makes the §4 argument weaker.

**Ask:** rule M7's parity criterion **now, conditional on §5's answer** — e.g.
*"whatever §5 rules for Qwen3 binds M7's Llama-3.2 corpus, with the diverging
set re-measured for the new model."* Opus specifically does not want to
discover at M7 that the tokenizer gate is fine (M1-style, discrete and exact)
while the forward-pass parity gate inherits an unreachable bar.

### 10.9 — τ may not be a constant, and M7 is where that bites

§9.4 asks for a single τ. But the forward note already records a
position-dependent error: stoke's `rope_neox` accumulates
`theta *= theta_scale` (llama.cpp-matching) while `ref_forward.py` computes
per-k powers directly — *"same math, ~1e-9 rounding difference that **GROWS
WITH POSITION**"*. **M7's corpus deliberately includes ≥1k-token contexts**,
and M2's own corpus already has 447–490-token prompts.

**Ask:** is τ a constant, or a function of position? If constant, Opus needs
to know it is expected to hold at 1k+ positions where a known error term is
growing. If a function, its shape.

### 10.10 — A TENSION between M7's note and §9.7, please resolve

M7 requires *"re-run ALL baselines at current dated versions before the README
table."* §9.7 asks whether an oracle-binary change re-derives the parity
criterion. **These pull opposite ways:** M7 mandates refreshing the referent
exactly where §9.7 may mandate freezing it.

**Ask:** resolve explicitly. Opus's guess is that **perf baselines refresh**
(that is what 10.2's re-anchoring is for) while the **parity oracle stays
pinned at b10037** — two different referents with two different rules — but
guessing at a milestone boundary is how a criterion silently changes meaning.

### 10.11 — Standing rule for headroom: carried, or owed?

Recurring pattern, no rule. Currently open: attention-across-heads is still
single-threaded; the K-quant twins retain only ~half the size ratio (llama
1.63× ahead at 8T); §6's ~1.16× unexplained. Each was called "headroom,
neither blocks the gate" — by Opus, not by a ruling.

**Ask:** a standing rule for when measured headroom is **carried forward**
versus **owed before a stamp**. Opus proposes: *carried if the milestone's
own bar passes without it and it is recorded with a number; owed if the bar
passes only because the bar moved.* **Ratify or correct.**

### 10.12 — What should Opus do when it believes a forward-note item is wrong?

§3.5 is the live example: an item written in good faith, resting on a
prediction that later measurement contradicted. Opus found it by accident
while investigating something else.

**Ask:** the standing instruction. Options: (i) stop and memo Fable, (ii)
proceed under the note and record the contradiction, (iii) proceed under the
note **and** open a FOR-FABLE entry without stopping — Opus's preference,
since it keeps the build moving while guaranteeing the ruling happens.

---

## 11. WHAT OPUS STILL OWES — tracked openly

- **12.1 — §3.7 sweep** ⏳ running, ~2–3 h. No permission needed.
- **12.2 — Full C-vs-ref on HEAD.** ❌ Not run. **~9.5 h** for M2's 24-prompt
  subset at the measured 9.7 s/position. Blocked on §5-(a) and §8.1.
- **12.3 — Leg 2 verdict.** ❌ Blocked on `llama-server` (Kabir).
- **12.4 — The Q4_K_M diverging list.** ❌ Does not exist; blocked on the
  oracle. Required before §9.2-(C) can be graded.
- **12.5 — Q8_0 oracle payloads.** ❌ **All 50 destroyed** by §12.2's defect.
  Unrecoverable without the oracle.

---

## 12. PROBED CONSTANTS AND DEFECTS — measured this session, do not re-estimate

**11.1 — The gate is ~6× cheaper than leg 1 made it look.** Leg 1 ran ~100
s/prompt; the same prompt on a quiet box, same binary, is **16 s** (14–17 s at
1/2/8/16 threads). Leg 1 was measured against a box carrying an oracle plus,
for part of the run, a second leg's oracle. **A clean serial leg is plausibly
~40–60 min, not "multi-hour unattended".** Not yet confirmed end-to-end.
The M2 binary, by contrast, is **~147 s/prompt** — pre-K5, single-threaded.

**11.2 — DEFECT (FIXED): `parity_gate` destroyed its own oracle results under
`--clean`.** `_cache_key_parts()` has no term for which side produced an
entry, so a prompt's oracle and stoke results shared one **filename**; under
`--clean`, `_load_cache_data()` returned `{}`, so the stoke write **replaced**
the oracle entry.

> **CORRECTED, and the correction matters.** First recorded as "17
> oracle-shaped + 33 stoke-shaped", from classifying by **file size**. That
> proxy was wrong. Reading the actual `data` keys: **all 50 of leg 1's entries
> hold `['stoke_ids']` and NOT ONE holds `oracle`.** The only 3 surviving
> oracle payloads are **leg 2's**. **Every one of leg 1's 50 oracle payloads
> is gone.**

No verdict was ever wrong (run_gate holds both halves in memory), but
resumability was void and a 71-minute oracle phase is unrecoverable.
**Fixed** via `_SESSION_WRITES`: the two sides merge through an in-process
record of *this run's own writes*, so `--clean`'s guarantee is preserved
verbatim — the merge source is never the disk. **Verified both ways:** both
halves survive a store/store sequence, and a hand-tampered cache file is still
ignored by a `--clean` load. `--selftest` ALL GREEN.

**DURABLE, and it happened twice in one day:** a shape inferred from a
**proxy** (file size) was read as confidently as a shape read from the
**data**, and was wrong in the direction that made the damage look smaller.
The status line made the identical error, counting cache FILES as prompts
completed and showing 100% with half the leg left. **Read the field, not the
silhouette.**

**11.3 — DEFECT (FIXED): both legs shared one report path.**
`DEFAULT_REPORT_PATH` is a single `parity_gate_dev_report.md`; leg 2 would
have erased leg 1's evidence on completion. `run_m3_gate_leg.cmd` now passes
a per-leg `--report`.

**11.4 — DEFECT (FIXED): `write_report()` hardcoded `# M2 parity gate
report`**, mislabelling the M3 run as M2. Now names the model it graded.

**11.5 — Status-line and watcher defects, both of which HID a verdict rather
than changed one.** The bar counted cache files as prompt completions (100%
with half the leg left, because one file serves both phases); the leg-1 exit
watcher read `.ExitCode` without first touching `.Handle`, recorded `rc=`, and
wrote malformed JSON that `json.load` rejected — so the finished leg vanished
from the bar entirely. Both fixed. **Leg 1's rc=1 is verified from the `GATE
FAIL` line plus the report write, not from the watcher.**

---

## 13. ANSWER CHECKLIST — copy into the RULING file and fill every row

Every ID below is a question in this memo. **`DECLINED` is a valid answer;
blank is not.** Opus treats a missing ID as an oversight and will have to ask
again, which is the outcome this memo exists to prevent.

```
RULING — 2026-08-04 — M3 gate item [2] and dependents
Authority: Fable 5 session, launched by Kabir <date/time>

--- Q1  M3's exit criterion (§5) -------------------------------------
Q1      option chosen: (a) / (b) / (c) / (d) / other:
Q1-a1   subset if (a): all 50 / M2's 24 / other:
Q1-a2   waiver policy for ref-vs-C: keep eps=0.001,budget 2 / recalibrate to:
Q1-a3   ~9.5 h reference run authorized?  yes / no / conditions:
Q1-a4   must Q4_K_M use ref too?  yes / no / why:
Q1-b1   evidence still owed by the excluded 19, if (b):
Q1-b2   how the Q4_K_M leg's corpus is defined (no diverging list exists):
Q1-b3   prompt outside the 19 diverges -> handled by 9.4? yes / no / other:
Q1-d    if (d): the divergence bound, and measured against what base rate:

--- Q2  K3's exit decision (§6) --------------------------------------
Q2-1    K3 exits as-is?  yes / no / conditions:
Q2-2    is the ~1.16x unexplained residual owed work before M3 stamps?
Q2-3    gate [1]'s measured 1.235x bit-exactness price: permanent / revisit:

--- Q3  f32-reduction twins (§7) -------------------------------------
Q3-1    in scope for M3 / deferred to M4 / other:
Q3-2    does Q1's ruling change that?  yes / no:

--- Q4  Standing authorizations (§8) ---------------------------------
Q4-8.1  ~9.5 h unattended spend on Kabir's say-so alone?  yes / no:
Q4-8.2  re-run the ruled gate after any gate-relevant change, unasked?
Q4-8.3  may Opus ADD corpus prompts, or is 50+10 frozen?
Q4-8.4  confirm/correct what Opus must not decide alone:
Q4-8.5  ack: llama-server starts stay Kabir's:

--- 9.1  M2-vs-HEAD sweep outcomes -----------------------------------
9.1-A   all 50 identical -> discharges the K-correctness concern? yes/no:
9.1-B   differences all inside the 19 -> same as A?  yes / no:
9.1-C   differences OUTSIDE the 19 -> block M3 / grade by 9.4 / other:
9.1-D   sweep required evidence, or nice-to-have?

--- 9.2  Leg 2 (Q4_K_M) outcomes -------------------------------------
9.2-A   Q4_K_M passes 50/50 -> stamps item [2] alone?  yes / no:
9.2-B   fails inside the 19 -> inherited class?  confirm / correct:
9.2-C   fails outside the 19 -> build the list first / grade by 9.4:
9.2-D   K-quant-only divergence: bisect twins + Q4_K/Q6_K unasked? yes/no:

--- 9.3  Ref-gate outcomes (only if Q1 = a) --------------------------
9.3-A   ref gate passes -> M3 closes, subject to 9.5?  confirm:
9.3-B   ref gate FAILS -> hard block? and may Opus bisect via --dump-layers?
9.3-C   ref failure inside the 0.043 class gap -> waivable? eps/budget:

--- 9.4  THE CLASSIFICATION RULE (most needed) -----------------------
9.4-1   ratify Opus's two-condition rule?  ratified / corrected as:
9.4-2   tau =                       nats     (rationale:)
9.4-3   who may apply it without a further ruling:

--- 9.5  Stamp and tag ------------------------------------------------
9.5-1   exact stamp precondition list:
9.5-2   confirm sequence: stamp -> ROADMAP -> MEMORY.md -> owner M3->M4 -> tag m3, local, no push:
9.5-3   do Q2 / Q3 block the stamp, or ride into M4?

--- 9.6  Forward guards ----------------------------------------------
9.6-1   does Q1 set precedent for M4's Q8-KV criterion?  yes / no:
9.6-2   confirm M4 batch-composition independence stays EXACT regardless:
9.6-3   confirm M5 "replay recorded ids, never re-sample" untouched:

--- 9.7  Referent drift ----------------------------------------------
9.7-1   on any oracle-binary change: re-derive criterion / freeze b10037:

--- 9.8  If the oracle is never authorized again ---------------------
9.8-1   is there an oracle-free path to closing M3?  yes -> which / no:

--- 10  FORWARD RULINGS, M4-M7 (DEFER + trigger is valid here) -------
10.1    Q8-KV divergence BOUND at M4:              (and its base rate:)
10.1b   does M4's KV criterion stay (d)-shaped whatever Q1 rules? yes/no:
10.2    is bar RE-ANCHORING a STANDING rule for all milestones? yes/no:
10.3    bandwidth budget's N: frozen at measured 6 / re-measured per m/s:
10.4    M4 gate items vs design constraints, and M4's stamp list:
10.5    does M5's bar-miss protocol generalise to all milestones? incl. K3?
10.6    ack: Q8-KV default ON/OFF at M5 stays Kabir's:
10.7    is the unchanged-M0-harness smoke THE M6 gate, or one of several?
10.8    M7 Llama-3.2 parity criterion, CONDITIONAL on Q1:
10.9    is tau constant or position-dependent?  constant / f(pos) =
10.10   M7 "re-run ALL baselines" vs 9.7 pinning -- resolution:
10.11   headroom carried vs owed -- ratify Opus's rule / correct as:
10.12   standing instruction when a forward-note item looks wrong: i/ii/iii:

--- EVIDENCE REQUESTED (if any) --------------------------------------
        measurement:
        conditional ruling per plausible result:
```

---

## 14. APPEND-ONLY LOG

**2026-08-04 — memo opened.** Leg 1 GATE FAIL at p02:16; bisection across
M2/K2/HEAD; p02 identified as one of M2's documented 19. Leg 2 stopped at
oracle 3/50. Nothing tagged, nothing stamped.

**2026-08-04 (same session) — §3.5 added.** Found item [2] was a *deliberate*
deferral, not a slip; its stated mechanism is implemented at `vec.c:162` and
measured insufficient. Reframed §5 accordingly.

**2026-08-04 (same session) — §3.6 added; the C-vs-ref question answered.** `ref_forward.py`
sides with stoke against the oracle. ref-vs-M2 = 1.4e-5 validates the
comparison; ref-vs-oracle = 0.143 is 3× ref-vs-HEAD. **No stoke bug exists at
the failing position.**

**2026-08-04 (same session) — §12.2 corrected and fixed**; all 50 leg-1 oracle
payloads confirmed destroyed, not 33. §12.3/§12.4 fixed. §3.7 sweep launched.

**2026-08-04 (same session) — restructured into the full decision package**
at Kabir's instruction: one Fable launch must cover every question and every
branch. §5–§9 are now the complete ask; §9 pre-rules outcomes that have not
happened yet.

**2026-08-04 (same session) — routing added.** "WHERE TO WRITE YOUR RULING"
names the output path, forbids overwriting this file, requires the ANSWERED
stamp on the title line, and makes `DECLINED` mandatory over silence. §13 is
a fill-in checklist carrying every question ID, so nothing is answered by
omission; the EVIDENCE REQUESTED block lets a data-dependent answer stay a
one-launch answer.

**2026-08-04 (same session) — §10 added: FORWARD RULINGS for M4–M7**, at
Kabir's instruction that the rest of the roadmap be ruled now rather than one
milestone at a time. Twelve items, each grounded in something the forward note
leaves open or in a tension between two things it says — notably **10.8**
(M7 plans to oracle-gate Llama-3.2 on the very criterion §5 may find
unreachable), **10.9** (τ may not survive the position-growing `rope_neox`
error at M7's ≥1k contexts), and **10.10** (M7's "re-run ALL baselines" pulls
against §9.7's pinning). `DEFER` + a named trigger is a valid answer there,
and is distinguished from `DECLINED`. Sections renumbered: old §10–§13 became
§11–§14.

**2026-08-04 15:51 (same session) — §3.7 partial recorded, sweep STOPPED at
6/50 for a planned laptop shutdown.** 6 of 6 fall on §9.1's branch (B): every
M2↔HEAD difference is on a prompt already in M2's 19, every prompt outside it
is token-identical. **Recorded as a pattern, explicitly NOT as a result** —
44 prompts remain, including the whole long-context tier. Resume by re-running
`m2_sweep.py`; it needs no oracle and no permission.
