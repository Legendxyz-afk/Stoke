# M7 rung B2 — the bars' absolutes, and why an absolute is the wrong number to publish

**Date:** 2026-08-30 · **Stamp:** `2026-08-30-m7-b2-bars` (bank + runner report
beside this file) · **Cell:** 8f32 · **Arms:** S (stoke `bench --serve`),
L (`llama-bench`, the non-stoke referent) · **3 blocks.**

M7 owes a *reproducible benchmark report*. This rung set out to re-take the
absolutes, which were still M5's, and found that **the absolutes are not the
reproducible quantity.**

## 0. FIRST, THE GATE'S QUIESCE INSTRUMENT WAS READING NOISE

`CpuBusy()` was a one-shot `Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor`.
A FORMATTED WMI counter is a rate WMI derives from a two-sample window it manages
internally; a single query returns whatever its last internal refresh produced.
On an idle box it returned **7, 8, 10, 12, 14, 14, 16, 19, 25, 27, 38**.

Measured in the same seconds against a raw two-sample delta:

| instrument | readings |
|---|---|
| the gate's formatted counter | 27, 14, 16, 10, 12, 14, 19, 8 |
| raw `PercentIdleTime` delta | 2.4, 7.3, 2.5, 2.9, 4.4, 2.9, 4.0, 2.7 |
| `100 - % Idle Time` (PDH) | 2.7, 2.9, 3.9, 2.6 |
| per-process CPU accounting | 1.96% |

Three independent instruments agree at ~3%; the gate's was 5× high and jittered
by a factor of five on a quiet machine.

⚠⚠ **The gate was therefore wrong in BOTH directions.** It rejected quiet boxes —
the 2026-08-29 bars attempt voided itself over a "median 18%" that was this
artifact, and this session deferred B2 twice on the same reading — and, far
worse, **it could pass a loaded box** whenever two consecutive artifact reads
landed low. A `quiesced: true` banked before today is weaker evidence than it
looks. `CpuBusy()` now computes the delta itself and returns a gate-FAILING
number if the sample is unusable, never 0.

▶ With it fixed the gate passed at **2–3%**, and the effect is visible in the
data: the three S blocks span **0.4%**, where the same harness on a noisy box
this morning spanned 10%.

## 1. The numbers, today

| | b1 | b2 | b3 | median |
|---|---|---|---|---|
| stoke 8f32 S, `AGGREGATE(M0)` | 7.282 | 7.264 | 7.251 | **7.264** |
| `llama-bench` tg128 | 17.650 | 17.300 | 17.820 | 17.650 |
| `llama-bench` pp512 | 78.340 | 74.450 | 75.660 | 75.660 |

Binary: `stoke sha256 E4A3DEA95E7B`, `llama-bench 1AFC0B22E21F`.

## 2. ⚠⚠ THE ABSOLUTE SWINGS 1.8× ACROSS SITTINGS WITH NO CODE CHANGE

| sitting | stoke 8f32 S |
|---|---|
| 2026-08-20 (M5 gate) | 6.378, 4.659, 4.938 |
| 2026-08-28 | 8.548, 8.810, 8.960, 9.029 |
| 2026-08-29 (voided) | 7.223, 7.228 |
| 2026-08-30 (today) | 7.282, 7.264, 7.251 |

Today reads 0.81× the 8.960 banked on 08-28. **That is not a regression, and it
was settled by measurement rather than argument: the tag-`m5` BINARY, run today
in this sitting, gives 7.077** — it cannot reproduce 8.960 either. Two more
same-sitting A/Bs today agree (m5 vs HEAD 0.12%, m6 vs HEAD 0.09%).

## 3. THE REFERENT MOVED WITH IT, WHICH IS WHAT THE L ARM IS FOR

| 08-28 → today | change |
|---|---|
| stoke, absolute | **−18.2%** |
| `llama-bench` pp512 | **−18.0%** |
| **stoke normalised to pp512** | **−0.3%** |
| stoke normalised to tg128 | −13.1% |

The box lost 18% between those two days and took stoke with it. Normalised
against a referent measured in the SAME sitting, the engine is unchanged.

⇒ **The reproducible number is the ratio, not the rate.** A README that publishes
"7.26 tok/s" publishes a property of one afternoon; the same file measured on
08-28 would have said 8.96 and on 08-20 4.94, for the same code.

## 4. AND pp512 IS THE RIGHT REFERENT FOR THIS WORKLOAD, tg128 IS NOT

0.3% against 13.1% across the same pair of sittings. The mechanism says it
should be so: S2 is 512 prompt tokens against 128 generated, and `AGGREGATE(M0)`
charges the span with every later arrival's prefill — so this number is
prefill-dominated and should track a prefill referent.

⚠ **One cross-sitting pair is what supports this, not a law.** The 08-20 point
cannot test it, because real code landed between then and 08-28 (the Q6_K
kernel and the EcoQoS fix — visible as the S/pp ratio moving 0.0684 → 0.0963,
a genuine +41% relative gain). The claim is a well-motivated hypothesis with one
confirming pair, and it makes a prediction the next sitting can falsify: if the
box moves again, S/pp should hold and S/tg should not.

| sitting | S/pp512 | S/tg128 |
|---|---|---|
| 2026-08-20 | 0.0684 | 0.2825 |
| 2026-08-28 | 0.0963 | 0.4734 |
| 2026-08-30 | 0.0960 | 0.4116 |

## 5. What B2 hands to the report (gate item [5])

1. Publish **stoke / llama-bench pp512, both measured in the same sitting**, with
   both raw numbers beside it. Never a bare tok/s.
2. State the box, the thread count, the cell, and the referent's build.
3. Keep the L arm in every future bars run — it is the only arm that can tell
   "the engine changed" from "the machine changed", and today it did exactly that.

⏹ Not done: the 16f32 / q8 cells, and blocks beyond 3. The methodological answer
did not need them, and the S arm's 0.4% spread says more blocks would buy little
at this cell.
