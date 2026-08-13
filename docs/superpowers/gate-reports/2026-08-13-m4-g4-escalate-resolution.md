# M4 gate item [4] — resolution of all seven ESCALATEs

**Verdict: ARITHMETIC-CLASS, all seven.** The L2/L3 ladder returned **no STEP**
at any of the seven divergent positions or at any of three control positions,
and every divergent position's layer curve sits *inside* the band that
agreeing positions of the same pair occupy. Item [4] therefore stands at
**31 + 7 = 38 ARITHMETIC-CLASS, 0 BUG-class.**

τ was **not** re-frozen, moved, or re-derived. No threshold changed. The
sample is not a sample: all seven were laddered.

Adjudicated at `build/win/stoke.exe` sha256 `F4F9B667…` (archived as
`build/stoke-m4-gate-f4f9b667.exe`), model
`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, pair `--kv f32` ↔ `--kv q8`.

---

## 1. What escalated, and why — every one is churn at the τ cut

**None of the seven is a margin breach.** §6 sends a divergence to ESCALATE
either for `max(g_X,g_Y) ∈ [τ, 2τ)` or for a top-5 set difference. All seven
have `max(g) < τ` — the largest is **0.7652 against τ=1.0103, 76% of
threshold** — so every one escalated on the *set* clause alone.

Re-probing both sides at each position (`tests/tooling/g4_escalate_probe.py`,
output `2026-08-13-m4-g4-escalate-characterisation.json`) shows what the set
difference consists of:

| position | τ | symmetric difference | is that token in BOTH top-5s? | nearest candidate to its own cut |
|---|---|---|---|---|
| p01 gp=22 | 1.0103 | {2293} | yes — f32 rank 1, q8 rank 2 | 0.0298 |
| p03 gp=4 | 1.0103 | {26244} | yes — f32 rank 2, q8 rank 1 | 0.0896 |
| p12 gp=17 | 1.0103 | {2293} | yes — rank 4 both sides | 0.2563 |
| p17 gp=9 | 1.0103 | {4030} | yes — rank 3 both sides | 0.0474 |
| p25 gp=5 | 1.0103 | {99486} | yes — rank 4 both sides | 0.0173 |
| p44 gp=3 | 0.7259 | {11379} | yes — f32 rank 1, q8 rank 2 | 0.0067 |
| p45 gp=0 | 1.0103 | {320, 4710} | yes — ranks 1–2 both sides | 0.0881 |

**In all seven, every token in the symmetric difference appears in BOTH sides'
top-5.** Not one is "a candidate the other implementation never ranked" —
which is the exact language §6 uses to define its BUG trigger. What differs is
only which side of the `rank5 + τ` cut a mutually-ranked token falls on, and in
six of seven a candidate sits within 0.09 of that cut.

⚠ **This is a property of the instrument, recorded rather than repaired.** §6's
set clause discards candidates within τ of rank-5 precisely so that "boundary
churn within τ is not a set difference" — but the discard is itself a hard cut,
so churn simply moves from rank-5 to `rank5+τ`. The rule's fail-safe handles
it correctly (unclassifiable ⇒ ESCALATE ⇒ the ladder decides, never a BUG on
the constant alone), so the instrument behaved as designed. It is noted here
for the same reason M3 noted the `1e-3` constant: a later reader will meet it
again. See the forward note in §4.

## 2. The ladder — STEP is the leg that carries the verdict

Per `l2_diff.py`'s design (`JUMP_FACTOR = 10.0`), a STEP is a layer whose diff
jumps ten times the running trend — the signature of a structural bug in one
op, as against the smooth compounding of a numeric difference.

Dumps are `stoke run --dump` at the **last prefill position** on each side.
p45's divergence is at gp=0, so its divergent forward pass *is* the last
prefill position and needed no construction at all. The other six were
teacher-forced (`tests/tooling/ladder_teacher_force.py`): prompt + the gp
shared tokens before the divergence, which both sides agree on by construction
since gp is the FIRST divergence.

**The teacher-forcing is verified, not assumed.** The constructed text is
re-tokenised and compared id-for-id against `prompt_ids + gen_ids[:gp]`, and a
mismatch refuses the position. This caught a real defect on the first attempt:
the captured continuation carried the print's trailing newline, which
tokenises to id 198, so all three prompts were one token long — they would
have dumped *the position after* the one under adjudication. All seven finals
round-trip EXACT.

| position | gp | L35 max_abs_diff | L35 mean_abs_diff | STEP |
|---|---|---|---|---|
| p01 | 22 | 1.117e+01 | 2.285e-01 | **none** |
| p03 | 4 | 1.512e+00 | 2.038e-01 | **none** |
| p12 | 17 | 3.130e+00 | 3.382e-01 | **none** |
| p17 | 9 | 2.661e+00 | 3.470e-01 | **none** |
| p25 | 5 | 1.146e+01 | 2.772e-01 | **none** |
| p44 | 3 | 6.471e+00 | 3.496e-01 | **none** |
| p45 | 0 | 2.554e+00 | 3.228e-01 | **none** |
| *control* p06 | — | 4.443e+00 | 4.401e-01 | none |
| *control* p15 | — | 4.019e+01 | 3.767e+00 | none |
| *control* p16 | — | 4.157e+00 | 2.856e-01 | none |

**No STEP anywhere in ten 36-layer curves.** §6: *"Ladder ≤1e-3 / no-STEP ⇒
ARITHMETIC-CLASS."*

## 3. The magnitude leg does not apply to this pair — and is reported anyway

`l2_diff.py` reports **`L2: FAIL`, 36/36 layers over 1e-3, for every pair
above, including the three controls where the two sides chose the SAME token.**
That is the reading the M3 p41 resolution predicted in its forward note: the
`1e-3` is `l2_diff.py`'s C-f32-vs-numpy-f32 constant, and a pair differing *by
construction* exceeds it identically whether the engine is healthy or broken.
Here the two sides differ by an 8-bit requantisation of the KV cache, so the
constant carries no information at all. **M3 predicted "any M4+ escalation will
hit this immediately"; this is that.**

So the absolute question was answered the way M3 answered it — calibrate on
positions where the two engines **agree**, then judge the divergent positions
against that:

| statistic | control range | divergent range | divergent vs controls |
|---|---|---|---|
| `mean_abs_diff` | 0.2856 – 3.767 | 0.2038 – 0.3496 | **every divergent position is below the SECOND-lowest control (p06, 0.4401)** |
| `max_abs_diff` | 4.157 – 40.19 | 1.512 – 11.46 | all inside; the largest divergent (11.46) is under a third of the largest control |

**The divergent positions are not elevated at all** — they sit at the bottom of
the control band on both statistics. This is a stronger result than M3's, where
the divergent position ran 1.15× the highest control.

⚠ **And it runs *against* the selection confound rather than with it.** A
divergence is selected for being a position where the perturbation flipped the
decision, so it should if anything look noisier than an arbitrary position.
These do not. The confound therefore cannot explain the result away; it makes
it more conservative.

⚠ **Honest limit on the magnitude leg:** the control band is wide — p15's mean
is 13× p16's — so this leg discriminates weakly. It is the STEP leg that
carries the verdict, exactly as §6 specifies, and the magnitude comparison is
recorded as corroboration rather than as the finding.

## 4. Forward note — two constants that will bite again

1. **`l2_diff.py`'s `1e-3`** wants restating in the per-pair, per-band form τ
   already has. M3 raised this; M4 has now hit it exactly as predicted, on a
   third distinct pair. M7's Llama-3.2 ladders will be the fourth.
2. **§6's set clause converts rank-5 churn into `rank5+τ` churn.** Every
   ESCALATE in this gate came from that, and none was a substantive
   disagreement. A future restatement could compare the *undiscarded* top-5
   sets and treat any mutually-ranked token as agreement, which is what the
   clause's parenthetical already says in words.
