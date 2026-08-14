# NULL-1 base rate — the ≤19/50 bound is disproven as a Q4_K_M criterion

**Result: N = 35/50.** The pre-registered reading
(`design-notes/2026-08-14-m45-null1-baserate-preregistration.md`, committed
`3def4aa` before the run) fixed the interpretation in advance:

> **N ≥ 19** — the ≤19/50 bound is disproven as a Q4_K_M criterion. A change
> with zero semantic content would have failed it. C's 34/50 then carries no
> defect signal on this instrument, C-FAIL-RATE fired on a constant imported
> from another model file, and the bound must be re-derived on Q4_K_M before
> any fold change is judged against it.

N = 35. That branch is taken.

## What was compared

Two binaries built from the same worktree at `ed2aa32` (tag m4), same compiler,
same flags, differing by **one thing**: the eight f32 lane additions summed in
reverse order, applied to the Q4_K scalar, the Q6_K scalar and the twin's
`kq_finish` so scalar↔twin bit-exactness holds on both sides.

    -    for (int l = 0; l < 8; ++l) sumf += sums[l];
    +    for (int l = 7; l >= 0; --l) sumf += sums[l];

The same eight addends, the other way round. **There is no mathematical
difference between the two binaries.** `null0` was verified stream-identical to
the m4 gate archive before use; `null1` passes `test_simd` 12/12 and
`test_kquant` 8/8. Graded with the identical harness, corpus, `-n 64` and
probe-spec as the C run, both sides `--kv f32`.

## The numbers

| run | perturbation | file | N | flip rate | mean first-div position |
|---|---|---|---|---|---|
| M2 census — **the bound's source** | f32 reduction order | **Q8_0** | **19/50** | 0.5938 %/tok | — |
| **NULL-1** | **zero semantic content** | Q4_K_M | **35/50** | 1.0938 %/tok | 18.5 |
| rung C | ~1e-7 fold restructure | Q4_K_M | 34/50 | 1.0625 %/tok | 18.4 |
| item [4] | 2.3e-3 KV requantisation | Q4_K_M | 38/50 | 1.1875 %/tok | 19.0 |

**All three Q4_K_M runs land within four prompts of each other, against a bound
of 19.** The perturbations they represent span **four orders of magnitude**.

⚠ **35 vs 34 is not a finding.** At n=50 a one-prompt difference is noise, and
nothing here says null-1 perturbs *more* than C. The correct reading is that
the three are **indistinguishable on this instrument** — which is the point.

## The divergent sets coincide

| pair | overlap | expected if independent | σ | z |
|---|---|---|---|---|
| null-1 ∩ rung C | **32** | 23.8 | 1.53 | **5.37** |
| rung C ∩ item [4] | 31 | 25.8 | 1.42 | 3.63 |

Three perturbations spanning four orders of magnitude flip substantially the
same prompts at substantially the same positions (mean first divergence 18.5 /
18.4 / 19.0). **What this instrument measures is the corpus's near-tie density,
not the size of the perturbation.** Once a perturbation clears the near-tie
margin scale — which reversing eight additions does, after 36 layers of
compounding — the corpus picks which prompts flip.

Corroborating this from phase 1: the two runs freeze τ from **identically many
agreeing probe positions (12 of 15 each)**, and their p95 logit deltas among
those agreeing positions are the same size — 0.3195 / 0.3669 / 0.3212 for
null-1 against 0.3969 / 0.3969 / 0.2694 for C. **A pure lane-order reversal
perturbs this model's logits by within 24 % of what C's entire fold restructure
does, and by more in one band.**

## Why the comparison is sound

- **τ-free**, recorded before the result (`d7c7627`): `m4_q8kv_gate.py:423`
  computes `n_div` in phase 2 from stream comparison, and phase 2 never reads
  τ. τ is frozen in phase 1 and used only in phase 3 classification. The three
  runs are directly comparable on N despite each freezing its own τ.
- **Same instrument throughout**: same harness, same 50 prompts, same `-n 64`,
  same model file, same probe-spec.
- **Same-build pair**: null0 and null1 come from one worktree with one edit
  between them, so no build difference confounds the comparison — and null0 was
  checked stream-identical to the m4 archive first.

## What this does and does not license

**Does:** the ≤19/50 bound cannot be applied to Q4_K_M. It is the M2 census
(`m2_sweep.py:20` hardcodes the Q8_0 model), and on Q4_K_M the floor for *any*
arithmetic-class change measures ~35/50. C-FAIL-RATE therefore fired on an
imported constant, and **C's 34/50 is not a defect signal — it sits at or below
the null floor.** Any future fold change faces the same broken criterion, which
is why this blocked the maddubs twin and hence P-BAR.

**Does not:** adopt C. The pre-registration was explicit that no outcome adopts
or condemns it. Park-vs-revert remains Kabir's call (memo §6a). What this
removes is the only evidentiary basis for treating C as *failed*.

**Owed:** a re-derived bound is a criterion change and needs its own record.
The obvious form is this run's own number — a Q4_K_M null floor — but choosing
a bound from the measurement it will police is the circularity the M3 tripwire
memo already paid for, and it should be reasoned about rather than assumed.

## Phase 3 — the verdict profiles are indistinguishable too

Completed 18:37:38. The classification cannot change N, which was fixed at the
end of phase 2, but it turns out to say the same thing again:

| | N | BUG-class | ARITHMETIC-CLASS | ESCALATE |
|---|---|---|---|---|
| **NULL-1** — zero semantic content | 35/50 | **0** | 26 | 9 |
| rung C — ~1e-7 fold restructure | 34/50 | **0** | 24 | 10 |

**A pair of binaries with no mathematical difference between them produces a
verdict profile indistinguishable from C's on every axis.** Zero BUG-class
either way, and the ARITHMETIC/ESCALATE split matches within two.

The harness's own summary is the disproof stated in its own words:

    === GATE ITEM 4 ===
    first-divergence prompts : 35/50   (bound: <= 19/50)
    flip rate                : 1.094%/token  (bound: <= 0.5)
    BUG-class                : 0              (bound: 0)
    GATE ITEM 4: NOT PASSED

**The instrument reports NOT PASSED for a binary pair that computes the same
values in a different order.** The `exit=1` that followed is the harness
applying the inherited bound correctly; it is not a malfunction, and the queue
recorded `ladder=0 baserate=1` for exactly that reason.

⚠ **A prediction recorded before the run came out wrong.** The
pre-registration expected null-1's classification to look *worse* than C's —
its τ is smaller in both well-populated bands (0.639 / 0.734 vs 0.794 / 0.794),
and a smaller τ makes ARITHMETIC-CLASS harder and ESCALATE likelier. It
produced **9 ESCALATEs against C's 10**, marginally fewer. At ~9–10 of ~35 the
two are indistinguishable, so the reasoning was not refuted so much as swamped;
but the directional prediction was not borne out and is recorded as such.

The nine ESCALATEs are **not** laddered here. Unlike C's, they adjudicate
nothing: this pair is a throwaway calibration instrument, not a candidate for
adoption, and no decision depends on their classification.
