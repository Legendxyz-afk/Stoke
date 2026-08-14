# M4.5 rung C — resolution of all ten ESCALATEs

**Verdict: ARITHMETIC-CLASS, all ten.** The L2/L3 ladder returned **no STEP
beyond the calibrated L01 artifact** at any of the ten divergent positions or
at any of three control positions, and every divergent position's layer curve
sits *below the second-lowest control*. C's grading therefore stands at
**24 + 10 = 34 ARITHMETIC-CLASS, 0 BUG-class.**

τ was **not** re-frozen, moved, or re-derived. No threshold changed. The sample
is not a sample: all ten were laddered, and none was refused.

Adjudicated between `build/stoke-m4-gate-f4f9b667.exe` (side A, pre-CONTRACT-v2)
and `build/stoke-m45-c-head-bc9dd3d6.exe` (side B, the binary that was actually
graded — **not** `build/win/stoke.exe`, which a rebuild from the unchanged tree
gave a different sha256 on 2026-08-14). Model `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`,
both sides `--kv f32`. Run 2026-08-14 15:47–16:06 on a quiet box, `exit=0`.

---

## 1. What escalated — one margin breach, nine boundary churn

§6 sends a divergence to ESCALATE either for `max(g_X,g_Y) ∈ [τ, 2τ)` or for a
top-5 set difference. Re-probing both binaries at all ten positions
(`c_escalate_probe.py`, output `2026-08-14-m45-c-escalate-characterisation.json`):

- **0 of 10** have a disputed token the other side **never ranked** — which is
  the exact language §6 uses to define its BUG trigger.
- **9 of 10** escalated on the *set* clause alone, where a third token crosses
  the hard `rank5 + τ` cut. Nearest candidate to a cut: **0.0094 – 0.19**.
- **1 of 10** is a margin breach — **p40**, `max(g)=0.6621` against τ=0.5389,
  123% of threshold and 61% of the way to the 2τ BUG line — and **its symmetric
  difference is empty**: identical survivors, identical order but for the
  top-two swap.

Every one is two near-ties changing places. p35's pair differs by **0.0039**.

⚠ **This corrects the 2026-08-13 end-state note**, which said these "include
genuine [τ, 2τ) margin breaches, not the top-5-set churn that dismissed all
seven of item [4]'s". Exactly one is a margin breach; the other nine have
`max(g) < τ` (largest 0.5301 against τ=0.7939, 67%). The plural overstated it —
this is item [4]'s churn class reproduced, not a harder set.

## 2. The STEP leg — calibrated before the run, because it would have lied

`l2_diff.py` flags a STEP when a layer exceeds `JUMP_FACTOR` (10×) the running
trend. **At L01 that trend is one sample — L00** — and for a pair differing
only by an f32 fold, L00 sits on the rounding floor (measured **2.4e-07 to
4.8e-07**). The first layer that compounds anything clears 10× it by
construction. The smaller the perturbation, the more certain the false STEP,
which is why item [4]'s pair (KV requantised to 8 bits, ~2.3e-3) never tripped
it and this one always does.

**Measured, not argued** (`2026-08-14-m45-c-ladder-calibration.txt`): the
control p15 — a prompt where the two binaries produced **byte-identical token
streams** — STEPs at L01 exactly as the divergent p23 does. A flag that fires
where the engines fully agree carries no information about where they disagree.

So the verdict leg is read as **a STEP at L02 or later, present in divergent
curves and absent from controls**. `l2_diff.py` is deliberately unchanged — it
is the instrument behind closed gate items — and the calibration lives in
`c_ladder_run.py`, which banks the full 36-layer curve per position.

**Result: 13 of 13 curves carry the L01 artifact. `STEP AT L02+ : 0 — NONE`.**

§6: *"Ladder ≤1e-3 / no-STEP ⇒ ARITHMETIC-CLASS."*

## 3. The magnitude leg — divergent positions are not elevated

| position | gp | L35 max_abs_diff | L35 mean_abs_diff | STEP |
|---|---|---|---|---|
| p08 | 54 | 5.178e+00 | 3.022e-01 | L01 only |
| p11 | 12 | 2.489e+00 | 2.192e-01 | L01 only |
| p18 | 24 | 4.795e+00 | 2.732e-01 | L01 only |
| p23 | 1 | 6.332e+00 | 2.398e-01 | L01 only |
| p25 | 9 | 1.206e+00 | 2.249e-01 | L01 only |
| p30 | 10 | 5.394e+00 | 2.662e-01 | L01 only |
| p35 | 18 | 5.462e+00 | 2.378e-01 | L01 only |
| **p40** | 22 | **9.652e-01** | **1.916e-01** | L01 only |
| p42 | 22 | 3.447e+00 | 2.977e-01 | L01 only |
| p49 | 30 | 1.936e+00 | 2.547e-01 | L01 only |
| *control* p16 | — | 1.797e+00 | 2.475e-01 | L01 only |
| *control* p32 | — | 2.316e+00 | 3.231e-01 | L01 only |
| *control* p15 | — | 9.454e+00 | 1.161e+00 | L01 only |

As in item [4], `l2_diff.py` reports **`L2: FAIL`, 35/36 layers over 1e-3, for
every pair including the controls where the two sides chose the same token** —
the 1e-3 is a C-vs-numpy constant and carries no information for a pair that
differs by construction. So the absolute question is answered the way M3
answered it: calibrate on positions where the engines **agree**.

| statistic | control range | divergent range | reading |
|---|---|---|---|
| `mean_abs_diff` | 0.2475 – 1.1610 | 0.1916 – 0.3022 | **every divergent position is below the second-lowest control (p32, 0.3231)** |
| `max_abs_diff` | 1.797 – 9.454 | 0.965 – 6.332 | all inside; the largest divergent is 67% of the largest control |

**The divergent positions are not elevated.** ⚠ And it runs *against* the
selection confound rather than with it: a divergence is selected for being a
position where the perturbation flipped the decision, so it should if anything
look noisier than an arbitrary position. These do not.

▶ **The single margin breach is the quietest curve in the set.** p40 is the one
position where both engines were moderately confident in *different* tokens,
and it has the lowest `mean_abs_diff` (0.1916) and the only `max_abs_diff`
under 1.0 of any divergent position — below both on every control too. A
structural defect driving that disagreement would elevate its curve. It is the
smallest.

⚠ **Honest limit, same as item [4]'s:** the control band is wide — p15's mean
is 4.7× p16's — so this leg discriminates weakly, and 5 of the 10 divergent
means sit modestly above the *lowest* control. It is the STEP leg that carries
the verdict, exactly as §6 specifies; the magnitude comparison is corroboration.

## 4. What this does and does not settle

**Settles:** C's ten ESCALATEs are ARITHMETIC-CLASS. Combined with the clean
integer-totals oracle (`2026-08-14-m45-c-oracle-run.txt`, zero `S_b`/`M_b`
disagreement over 108,000 super-blocks per quant type), **nothing in C's grading
is a defect signal.** The C verdict is now complete rather than partial.

**Does not settle:** whether 34/50 is *high*. That is a question about the
bound, not about C, and the bound was measured on Q8_0 (`m2_sweep.py`:20) while
C was graded on Q4_K_M. The NULL-1 base-rate run answers it, with its reading
pre-registered before the number exists
(`design-notes/2026-08-14-m45-null1-baserate-preregistration.md`). Park-vs-revert
remains Kabir's call (memo §6a).

## 5. Forward note

1. **`l2_diff.py`'s `1e-3`** wants restating in the per-pair, per-band form τ
   already has. M3 raised it, item [4] hit it, this is the third pair.
2. **`JUMP_FACTOR` against a one-sample trend is the second such constant.** An
   L01 STEP is uninformative for any small-perturbation pair, and the fix is
   structural: judge a step against a trend of at least two layers, or start at
   L02. M7's Llama-3.2 ladders will meet both constants again.
