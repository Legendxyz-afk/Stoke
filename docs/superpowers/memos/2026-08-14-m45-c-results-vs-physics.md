# RESULTS-VS-PHYSICS — rung C's integer core is clean, and the bound it missed was measured on a different file

*2026-08-14. Opus 5, max effort. Filed under the throughput-parity ruling
(2026-08-12) §7.2, which pre-ruled this disposition: "STOP; run the
integer-totals oracle; if totals are clean it is a results-vs-physics memo to
Kabir with C parked on a branch." The oracle has been run. Totals are clean.
Not blocking; the decision it feeds is Kabir's.*

## 1. The oracle — the question the ruling asked, answered

§7.2's decisive test: old and new must produce **bit-identical per-super-block
integer totals**, because C changes only the f32 fold. Anything else means the
integer core moved and C is a real bug to be fixed against m4.

Harness: `tests/test_c_oracle.c`, four **separately transcribed**
implementations — m4 scalar and m4 AVX2 twin (from `git show ed2aa32:`), HEAD
scalar and HEAD twin — compared per super-block over `test_simd.c`'s
fixed-seed corpus at lengths {256, 512, 768, 2560, 9728}.

| | Q4_K | Q6_K |
|---|---|---|
| super-blocks compared | **108,000** | **108,000** |
| `S_b` / `M_b` disagreements, all six pairings | **0** | **0** |
| HEAD scalar copy vs library kernel (f32, bitwise) | identical | identical |
| HEAD twin copy vs library twin (f32, bitwise) | identical | identical |
| m4 scalar copy vs m4 twin copy (f32, bitwise) | identical | identical |

**⇒ C's integer core is byte-for-byte m4's.** This is not merely a diff
observation (the accumulation lines *are* textually unchanged in both the
scalar and the twin — worth knowing, and not sufficient: two different `S_b`
can collapse to the same f32 product, which is the hole an f32-only test
cannot close).

Three things keep the result from being self-congratulatory:

- The copies are **pinned to the shipping kernels**: every HEAD copy's f32
  return is asserted bit-identical to the linked library's real kernel, so a
  transcription slip fails before the oracle is consulted. The m4 copies are
  asserted bit-identical to *each other*, which is exactly the invariant the
  m4 tree held.
- **The instrument is proven able to fail**: perturbing one activation quant
  moves `S_b` by exactly the predicted 945 (Q4_K) and 3968 (Q6_K), on weights
  chosen so the delta cannot be zero.
- A deliberate mutation of the fold is what `test_simd`/`test_kquant` already
  cover; C step 1 recorded that leg failing at 7153 mismatches when the scalar
  changed and the twins had not yet followed.

Artifact: `gate-reports/2026-08-14-m45-c-oracle-run.txt` (6/6, with
provenance). The full Windows matrix is green with the new test in it.

## 2. What C actually changed — TWO things, and the ruling describes one

CONTRACT v2 is stated per super-block: reduce the eight i32 lanes to `S_b`,
then two f32 operations. That is the change the ruling authorised, and per
super-block **it is the more accurate fold** — one conversion, one multiply,
one add, against m4's eight conversions, eight multiplies and eight adds.

But m4's `sums[8]` was declared **before** the super-block loop and summed
**after** it. It was not only a per-block partition — it was an **8-way split
accumulation of the whole row**. CONTRACT v2 replaced it with a single serial
`sumf`. At a 9728-element row that is 38 terms down one dependency chain
instead of 38 spread across eight.

Measured (mean |relative error| against an f64 evaluation of the same totals,
2000 rows per length; `test_c_oracle.c`'s accuracy leg):

| row length | super-blocks | m4 fold | HEAD fold | |
|---|---|---|---|---|
| 256 | 1 | 1.71e-07 | **3.02e-08** | HEAD 5.7× better |
| 512 | 2 | 2.94e-07 | **5.55e-08** | HEAD 5.3× better |
| 768 | 3 | 1.33e-07 | **6.82e-08** | HEAD 2.0× better |
| 2560 | 10 | 2.00e-07 | **1.52e-07** | HEAD 1.3× better |
| **9728** | **38** | **8.07e-07** | 1.62e-06 | **HEAD 2.0× worse** |

This was a prediction before it was a result: the explanation "the regression
is in the cross-block chain, not the per-block fold" predicts the gap collapses
at one super-block, and it does — HEAD is 5.7× better there and only loses once
the row is long. **n=9728 is this model's `ffn_down` contraction**, and Q4_K_M
stores some layers' `ffn_down` as Q4_K. Q6_K does not cross over anywhere
measured (2.22e-07 vs 5.53e-07 at 38 blocks); *why* the two types differ at the
long end is **not established** — the Q4_K mins term is a large subtraction the
Q6_K path does not have, which is a hypothesis, not a finding.

Overall HEAD is the closer of the two folds about twice as often in both types
(Q4_K 978 vs 456, Q6_K 1095 vs 306, of 2000). Both stay inside the milestone's
arithmetic class throughout.

## 3. The 2^24 question, and why it is not the story

The new fold converts the *total* where the old converted eight lanes, so
`(f32)S_b` can round above 2^24 = 16,777,216 where m4's per-lane conversions
could not. Censused over the real GGUF as a bound over **every possible**
activation (`tests/tooling/c_fold_exactness_census.py`, `|q8| ≤ 127`
guaranteed by `quantize_row_q8_k`'s `iscale = -127/max`):

| | super-blocks | worst possible \|S_b\| | can reach 2^24 |
|---|---|---|---|
| Q4_K | 12,257,280 | 17,527,143 (104% of 2^24) | **30** |
| Q6_K | 3,454,720 | 110,199,678 (657% of 2^24) | 3,379,896 |

So on Q4_K the conversion is effectively always exact — 30 super-blocks in 12.3
million could reach the threshold, and only under an activation with every
quant at ±127 sign-aligned. On Q6_K it genuinely can round. **But the cost when
it does is ≤ 1 ulp — one part in 3.07e7, measured** — i.e. the same order as
any single f32 rounding, and HEAD performs three of those per super-block where
m4 performed twenty-four. This mechanism is real, quantified, and **too small
to be the story**; §2's accumulation shape is the larger effect by an order.

## 4. Why 34/50 is not evidence of a defect

The graded result: zero BUG-class, 34/50 first-divergence against ≤19/50, flip
1.0625 %/token, 24 ARITHMETIC-CLASS, 10 ESCALATE.

**⚠ The bound and the grading are on different model files.** The ≤19/50
census is `tests/tooling/m2_sweep.py`, which hardcodes at line 20:

    MODEL = ROOT / "models" / "Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf"

C was graded on **Q4_K_M** (`2026-08-13-m45-c-grade-run.json`). This is the
same cross-file calibration import the M3 tripwire memo already documented for
λ — "its λ inherited a ~0.5%/token base measured on Q8_0 while Q4_K_M measures
3.10%" — recurring in the branch that decides whether C is adopted. M3 measured
the per-position divergence rate as ~4× higher on Q4_K_M than Q8_0 (3.10% vs
0.74%); that is a different instrument and does not convert, but it fixes the
direction and the rough size, and the direction is against the bound.

**⚠ And the instrument barely discriminates on this file.** Three runs, same
corpus (50 prompts, n=64, `tests/data/parity`), same first-divergence metric:

| pair | file | perturbation | result |
|---|---|---|---|
| M2 census (**the bound's source**) | **Q8_0** | f32 reduction order, ~1.2e-7 | 19/50 |
| gate item [4], Q8-KV | Q4_K_M | KV requantised to 8 bits, ~2.3e-3 | 38/50 |
| **rung C** | Q4_K_M | f32 fold order, ~1e-7 | **34/50** |

Perturbations **four orders of magnitude apart** produce 38 and 34 on the same
file and the same instrument. A metric whose entire dynamic range across 1e-7
and 2.3e-3 is four prompts cannot separate arithmetic-class changes on Q4_K_M,
and a threshold imported from a file where it sat at 19 is not a discriminator
here. (It also caps at 50/3200 = 1.5625 %/token, so C-FAIL-RATE's flip trigger
was unreachable by construction — analysis memo §A2, already recorded.)

**No Q4_K_M base rate for this instrument exists.** M3's Q4_K_M work was a
*probe grid* (per-position), not a 50-prompt sweep, and the only Q4_K_M number
from this instrument is item [4]'s, at a perturbation four orders too large.
Establishing one needs a deliberately-constructed f32-reordering pair on
Q4_K_M — a real experiment, named here, not run.

## 5. Corrections to recorded constants, computed not asserted

1. **`bf13034`'s Q6_K bound is wrong.** It states "256 elements x |scale|<=127 x
   |q6|<=32 x |q8|<=127 = 132,120,576". Its own factors multiply to
   **132,128,768**, and the reachable maximum is larger still — **133,169,152**
   — because the 6-bit encoding centres to [-32, +31], so the negative end has
   the larger magnitude, and the i8 scale reaches -128 not 127. Reproduced from
   the kernel itself (`c_oracle_contract_worst_case_bounds_are_reproducible`).
   The conclusion it drew (an order inside i32) survives; the constant does
   not. This is the sixth instance of the milestone's signature failure —
   a quantity asserted from structure rather than computed.
2. **`1f78ebe`'s Q4_K bound is right**: 30,723,840, reproduced to the digit.
3. **`src/kernels/simd.h`:140-149 is now stale and says the opposite of the
   code** — it still describes the eight-lane partition as the contract and
   "the final horizontal sum runs lane 0..7 in order because that is the
   scalar's order". If C is kept, this must be rewritten; if C is reverted, it
   becomes correct again. Flagged, not touched, because which fix applies is
   the decision below.

## 6. The decision — Kabir's

Per §7.2 the disposition is **C parked, not adopted**. Two questions follow.

**(a) Park on a branch, or revert `1f78ebe` + `bf13034`?** These are not
equivalent. Reverting restores the m4 streams and the m3 archive as K-quant
referent, and re-blocks the maddubs twin — the old eight-lane partition is
precisely what forbids a wide-ISA kernel, which is why CONTRACT v2 was
authorised. **Note what C has cost and bought so far: the step that changed the
streams is the step that buys no speed.** C's throughput lives in the maddubs
upgrade, which is not written, and which would be bit-exact to the new scalar
and need no second grading run.

**(b) §8.4-reserved — should CONTRACT v2 be amended to pin the CROSS-BLOCK
accumulation as well?** §2 shows the ruling's contract text specifies the
per-super-block fold and is silent on how super-blocks accumulate, and that the
silent half is where the measured regression is. A v2.1 that keeps the exact
integer `S_b` (strictly better per block) but restores a multi-lane cross-block
accumulator would plausibly beat m4 at *every* length. Under the 2026-08-07
governance change this is mine to decide with a direction check, and my
recommendation is: **do not amend anything while C is parked** — an amendment
is a third stream change needing its own graded run, and the right order is to
settle (a) first. Direction check to Kabir: *is closing the parity gap still
the priority such that C's ladder continues at all, or does the 34/50 on a
mis-calibrated bound mean the ladder pauses until a Q4_K_M base rate exists?*

## 7. Carried headroom

- **(a)** No Q4_K_M-measured base rate for the 50-prompt first-divergence
  instrument. Every C/item-[4] verdict on that file is graded against a Q8_0
  number until one exists. §4.
- **(b)** Why Q6_K does not cross over at 38 super-blocks while Q4_K does is
  unexplained; the mins-term hypothesis is untested. §2.
- **(c)** The 10 ESCALATEs from the C grading are **still owed their ladders**
  and are not resolved by this memo.

  ⚠ **Correction to the 08-13 end-state note, which said these "include genuine
  [τ, 2τ) margin breaches, not the top-5-set churn that dismissed all seven of
  item [4]'s".** Recomputed from the banked verdict against §6's actual rule
  (`ESCALATE ⟺ max(g_X,g_Y) ≥ τ OR the discarded top-5 sets differ`): **exactly
  one of the ten is a margin breach** — p40 gp=22, max(g)=0.6621 against
  τ=0.5389, 123% of threshold and 61% of the way to the 2τ BUG line. **The
  other nine all have max(g) < τ** (largest 0.5301 against τ=0.7939, 67%) and
  escalated on the *set* clause alone — which is exactly item [4]'s churn
  class. The plural overstated it.
- **(d)** `stoke-m45-c-grade` is still registered as a scheduled task, kept
  only because deleting it discards the resume state while C is unresolved.
  Seventh recording of "spent helper outliving its job".
