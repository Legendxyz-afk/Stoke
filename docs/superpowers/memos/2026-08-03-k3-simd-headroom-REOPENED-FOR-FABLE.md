# K3 — the ruling's central measurement was taken in a depressed box state, and both conclusions it carried invert

**Status:** decision brief, parked for Kabir to hand to Fable per the 07-28
escalation routing. **Written by the Opus 5 manager session of 2026-08-03,
after landing the ruling's D1/D2/D4/D5.** Those landed and stand. **This brief
reopens D3 only.**

**Escalation trigger:** *"results contradict spec physics"* — and this time the
contradiction is with a ruling this project already adopted, so it is not mine
to re-decide.

> **THE RULING IS NOT AT FAULT FOR ITS REASONING; IT IS AT FAULT FOR ITS
> DENOMINATOR — AND IT IS THE SAME ERROR IT WAS WRITTEN TO EXPOSE.** The ruling
> corrected the project for comparing figures measured under different
> conditions. Its own central comparison then did exactly that: a llama number
> measured on this box during one window against a stoke number taken from the
> brief's earlier session. Re-measured together, both of its conclusions
> reverse. The lesson is now three-for-three (K1's %ref denominator, the §2.4
> residue, this) and should probably become a checklist item rather than a
> lesson.

---

## 1. What the ruling claimed, and what it rested on

Ruling §2.1–2.2 measured vendored llama b10037 at one thread, pinned and
resident, at **1.24–1.26 tok/s (±0.01)**, and built two conclusions on it:

1. **"SIMD collects nothing single-thread."** llama's 5.4 GB/s effective was
   *below* stoke's scalar `-O3` matvec rate of 6.92 GB/s, so the brief's ~2.3×
   SIMD headroom was declared dead and K3 re-based to "llama-1T-class,
   ~1.0–1.3× over scalar."
2. **"The bar is won in K5."** 6.86 tok/s at 6 P-cores ÷ 1.26 at one =
   5.5×/6 cores, read as near-linear, therefore per-core is compute-limited
   and threads are the lever.

**The 6.92 GB/s in (1) came from the brief's session, not from the ruling's
own.** The ruling labels it "(brief, interleaved medians)" — it was transparent
about the source; it just did not treat the basis mismatch as disqualifying.

## 2. The re-measurement

**Interleaved A,B,A,B** — the brief's own discipline, so box drift cannot pass
for a difference. Both single-threaded, both pinned to LP0, both mmap, both
timing exactly 32 generated tokens (llama `tg32`; stoke by differencing
`-n 36` against `-n 4` so load and prompt cancel).

| cycle | stoke scalar `-O3` | llama AVX2+VNNI | ratio |
|---|---:|---:|---:|
| 1 | 1.118 tok/s (4.79 GB/s) | 2.19 tok/s (9.38 GB/s) | **1.96×** |
| 2 | 0.922 tok/s (3.95 GB/s) | 2.32 tok/s (9.94 GB/s) | **2.52×** |

Absolutes swing (stoke 1.21×, llama 1.06×); the **within-cycle ratio** is the
figure, which is the entire reason for interleaving.

**The control that makes this a finding rather than a contradiction.** llama's
own number moved 1.26 → 2.19–2.46 between the ruling's window and now, which
had to be explained before anything was concluded from it. The candidate
explanation was the `-r` flag (the ruling used `-r 3`, the interleave `-r 1`).
**Refuted: re-run now, `-r 3` gives 2.46 ± 0.04 and `-r 1` gives 2.39** — the
identical command that produced 1.26 ± 0.01 in the ruling's window produces
2.46 today. The earlier figure was a box state, not a flag.

**Current same-session numbers, all pinned, memory conditions good:**

| | tok/s | GB/s eff. | % of ~15 GB/s ceiling |
|---|---:|---:|---:|
| stoke scalar `-O3`, 1T | 1.12 | 4.8 | 32% |
| llama VNNI, 1T | 2.46 | 10.5 | **68%** (ruling said 34%) |
| llama VNNI, 6 P-cores | 6.13 | 26.3 | — |
| llama VNNI, 10 threads | 7.97 | 34.1 | — |

## 3. Both conclusions invert, and both errors are the same error

**(1) The single-thread SIMD lever is ~2.2×, not "nothing".** llama's
production VNNI runs at 68% of the read ceiling where stoke's scalar sits at
32%. **This restores the brief's original ~2.3× estimate**, which the ruling
had dismissed as "landing near the right answer for the wrong reason." On this
evidence it landed near the right answer.

**(2) Thread scaling is SUBLINEAR, not near-linear — so K5 is a smaller lever
than the ruling assumed.** With the corrected 1T figure, 1T→6T is **2.49× on
six cores** and 1T→10T is **3.24× total**. The ruling's "5.5×/6 cores,
near-linear" was that same ratio divided by the depressed 1T.

**The two re-measurements are consistent with each other, which is what makes
this coherent rather than merely different.** Single-thread nearly doubled
(1.26 → 2.46) while 6-thread barely moved (6.86 → 6.13). That is exactly what
a DRAM-saturating workload does: at six threads the bus binds and the
single-core condition — whatever it was — stops mattering.

## 4. What I think follows — Fable may overrule all of it

1. **K3 is worth roughly as much as K5, not much less.** The plausible route to
   the bar is ~2.2× (SIMD) × ~3.2× (threads, saturating) × Q4_K_M's smaller
   weights — with SIMD no longer the junior partner. "The bar is won in K5"
   should come out of ROADMAP §M3.
2. **The K3 exit expectation should be llama-1T PARITY (~2.4 tok/s pinned,
   ~2.2× over scalar `-O3`)**, not the ruling's ~1.0–1.3×.
3. **Bandwidth saturation, not per-core throughput, is the ceiling that decides
   M4/M5.** If the bus saturates near ~34 GB/s at 10 threads on Q8_0, the
   concurrency architecture's headroom is a bandwidth budget, and that belongs
   in M4's design note before paged KV is scoped.
4. **The referent rows in ruling §2.3 must be RE-TAKEN, not merely re-cited.**
   Any figure from the ruling's measurement window is suspect by the same
   argument; the two llama rows should be re-measured immediately before each
   use, in the same session as the stoke figure they are compared against.
5. **A standing rule, if Fable wants one:** no comparison between two
   performance figures is admissible unless both were measured in the same
   session, and preferably interleaved. This project has now been bitten three
   times by the same thing.

## 5. What I did NOT change, because it is not mine

K3's scope, the ruling's D3, and ROADMAP §M3's strategic framing are untouched
pending Fable. **Landed and unaffected:** D1 (`-O3`, sanitizers `-O1`,
`-ffp-contract=off`), D2 (reference-stream replacement), D4 (schema 2 +
`build.cflags`), D5 (gate precondition satisfied on p48/p49). Those rest on
byte-identity and codegen, not on the depressed measurements.

**Corrected as plain fact, not as re-scoping:** the "~34% of ceiling / below
stoke's scalar rate" sentences I wrote into `src/bench.c` and ROADMAP §M3 on
the ruling's authority are measured-false and now say what was measured. The
inference drawn from them is left to Fable.

## 6. Not up for revisiting

The ≥7.8 tok/s bar, ε, the waiver budget, the corpus, the parity protocol,
push policy, the model/owner line. **Note this brief moves a bar-relevant
number in the HARDER direction for K5 and the EASIER one for K3; it does not
move the bar.** If any part reads as re-scoping a milestone after seeing
results, that reading wins and it waits for Fable.
