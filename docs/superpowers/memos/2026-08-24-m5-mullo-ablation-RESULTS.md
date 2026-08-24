# Pricing `vpmulld`: 1.121×, and the scale is not the lever either

**2026-08-24, Opus 5, branch `m5-entry`.** Prediction committed before the arm
was built: `2026-08-24-m5-mullo-ablation-PREREGISTERED.md`. Data:
`gate-reports/2026-08-24-m5-mullo-ablation-report.md` (+ `-raw.txt`, `-bank.jsonl`).
Runner: `tests/tooling/run_m5_mullo_ablation.ps1`.

## 1. The number

Prefill of a 525-token prompt, ctx 704, kv f32, arms interleaved with the order
alternating per cycle. **B = the shipping kernel with `_mm256_mullo_epi32(sub,
scale)` replaced by `_mm256_add_epi32` — B computes the WRONG ANSWER and exists
only to be timed.** The runner asserts B's stream DIFFERS from A's before timing
anything, because a 1.00× from a substitution that never took effect is
indistinguishable from "vpmulld is free".

| threads | A tok/s | B tok/s | **B/A** | A GFLOP/s | B GFLOP/s | pairings B won |
|---|---|---|---|---|---|---|
| **1** | 6.0950 | 6.8328 | **1.121×** | 51.6 | 57.8 | **4/4** |
| 8 | 30.4322 | 31.9998 | 1.052× | 257.5 | 270.7 | 4/4 |

t=1 pairings: 1.105 / 1.140 / 1.095 / 1.147. **The prediction was 1.15×, accept
1.10–1.25×. 1.121× is inside the band.**

**NEITHER DISCRIMINATOR FIRED.** ≥1.25× would have said "p01-throughput-bound,
the scale is the biggest lever"; ≤1.08× would have said "red herring". The answer
is in between and has to be stated as such rather than rounded to whichever
model I preferred.

## 2. What it means: the third compute argument to come in far under its count

`vpmulld` is **2 of the 4 p01 µops** in the per-sequence inner loop — half the
vector-ALU pressure, ~28% of all per-sequence µops. Deleting it entirely buys
**12.1%**.

| what was removed | share of the count | measured |
|---|---|---|
| the scalar mins loop (08-20) | ~40% of emitted instructions | **1.07–1.10×** |
| `vpmulld` (today) | ~50% of p01 µops | **1.121×** |

Two independent removals, two results an order below what a throughput model
predicts. **This kernel is not bound by anything we can count statically.** The
08-20 memo killed the front-end theory; this kills the port-throughput theory
with it — a p01-bound kernel would have handed back most of a 2× on the
per-sequence portion, and it handed back 12%.

**In log terms against the 3.22× single-thread kernel gap** (08-20, measured
in-sitting against llama): ln(1.121)/ln(3.22) = **9.8% of the way**. Together
with the mins term already banked, **both named compute levers combined cap at
1.099 × 1.121 = 1.232×, which is ~18% of the gap** — and one of them is free
while the other is an upper bound nobody can actually collect.

## 3. ⇒ THE ANSWER TO §5's QUESTION, for the scale specifically

The 08-20 memo framed llama's deferred-float-scale design as the thing on the
far side of the bit-exactness line. **This prices that trade: at most 12.1%, and
in practice less, because a real redesign still has to apply the scale — a float
FMA is cheaper than `vpmulld`, not free.**

⇒ **Crossing the bit-exactness line to buy the scale is not worth it.** That is
not a recommendation about the line in general; it is the number for this one
candidate, which is the number that was missing.

**And the bit-exact candidate from §4 of the pre-registration is capped by the
same 12.1%** — packing two sub-blocks' `sub` to i16 and applying both scales with
one `vpmaddwd` still issues 1 p01 + 2 p5 µops per pair, so it collects perhaps
half of the ceiling. **~5–8%, free in exactness terms, and NOT the lever.** Worth
doing eventually on the same grounds as the mins term (a loop-invariant rebuilt N
times is wrong regardless of the stopwatch); worth doing *next*, no.

## 4. What is left standing, and it is a MEMORY hypothesis

Both negative results point the same way: away from compute. The 08-20 candidate
list had one memory entry, and it is now the leading one.

**Activation re-reads.** Per super-block the kernel reads **16 × 256 B = 4 KB of
activations against 144 B of weights** — a **28:1 byte ratio** — and re-reads all
of it for every one of the `n_out` weight rows. The activations are L1/L2
resident, so this is not a DRAM claim; it is a claim about L1 bandwidth and load
throughput, which is exactly the class of thing that survives when instruction
count and port pressure have both been excluded.

That is the next thing to price, and it is testable by the same method: ablate
the activation load (reuse one sequence's block for all 16, computing the wrong
answer), measure, discard.

## 5. ⚠ INSTRUMENT: every ratio in this line of work is END-TO-END and therefore UNDERSTATED

`bench --ops` benchmarks `rmsnorm`, `softmax`, `swiglu`, `rope_neox`, `argmax`,
`vec_dot_q8_0_f32`, `dequant_row_q8_0` — **there is no microbenchmark for the
K-quant kernels at all.** So 08-20's 1.07× and today's 1.121× are both *prefill*
ratios, diluted by whatever fraction of prefill is not this kernel. At a plausible
f ≈ 0.8 the kernel-level figure behind 1.121× is ~1.16×.

**The dilution factor has never been measured**, which means no kernel ratio this
project has published is directly comparable to a kernel-level claim. It does not
change today's conclusion — 1.16× is no closer to 3× than 1.12× is — but a
`bench --ops` entry for `matvec_batch` would make every future ratio readable
directly, and it is cheap beside another end-to-end A/B.

## 6. ⚠ A STORY I TOLD MYSELF, MEASURED, AND KILLED

At t=8 arm B trended up across cycles (31.61 → 33.69, +6.6%) while A stayed flat
(1.2%). I proposed a mechanism: B issues fewer p01 µops, draws less power, and
holds a higher clock as conditions improve.

**Tested with `\Processor Information(_Total)\% Processor Performance` — the
counter that actually moves on this box — over four alternating runs. Both arms
reach ~185. There is no sustained clock difference.** The trend was noise: the
noisy arm is *A* at t=1 (5.7% spread vs B's 1.2%) and *B* at t=8 (6.6% vs A's
1.2%), which is not a property of either arm. Four cycles is not enough to
distinguish a 6% trend from 6% noise, and the mechanism I had ready would have
made a virtue of it. **t=8's 1.052× is therefore reported as ~1.05 with the
spread stated, and t=1 carries the finding.**

## 7. Disposition

- `build/win-mullo-b` is **deleted**. It computes the wrong answer and must never
  be mistaken for a candidate; the mutation is one line and is recorded here and
  in the runner.
- Nothing shipped. The default kernel is unchanged by this memo.
