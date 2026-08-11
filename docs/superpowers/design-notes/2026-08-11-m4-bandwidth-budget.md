# M4 §1 — the bandwidth budget

ROADMAP §M4 (Fable 2026-08-02 constraint 3) requires the M4 design note to
OPEN with this, and the 08-05 ruling requires **W and the budget's N to be
re-measured at M4 entry** — re-anchoring is a standing rule, so the 08-03
figures are re-derived here rather than cited.

**Measured 2026-08-11 ~05:20, quiesced box, ISLC stopped for the run and
restored after** (08-04 precedent: ISLC idles at 0.005 cores but lands
asymmetrically on the candidate side under page-cache churn). Tool:
`bench/vendor/llama-b10037/llama-bench.exe`, build `56d6e9dde (10037)`,
`-p 0 -n 64 -r 3`. Reported ± is llama-bench's own stddev over 3 reps.

## 1.1 W, re-measured

W is *effective* bandwidth: `tok/s × model_bytes`, since one decode step
streams every weight exactly once.

| threads | Q4_K_M tok/s | W (×2.4973 GB) | Q8_0 tok/s | W (×4.2804 GB) |
|---:|---:|---:|---:|---:|
| 1 | 4.98 ± 0.22 | 12.4 | 3.49 ± 0.05 | 14.9 |
| 2 | 9.84 ± 0.01 | 24.6 | 6.90 ± 0.11 | 29.5 |
| 4 | 16.07 ± 0.21 | 40.1 | 10.74 ± 0.01 | 46.0 |
| 6 | 17.45 ± 0.14 | 43.6 | 12.62 ± 0.27 | **54.0** |
| 8 | 17.85 ± 0.02 | 44.6 | 13.40 ± 0.08 | 57.4 |
| 10 | 17.93 ± 0.25 | **44.8** | 13.47 ± 0.15 | **57.7** |

**The 08-03 floor REPRODUCES.** Q8_0 at 6T measures **54.0 GB/s** against the
recorded 54.13 entry sweep — four months and a different milestone apart. The
K4-entry observation reproduces too: Q4_K_M tops out well below Q8_0 (44.8 vs
57.7) because the superblock unpack binds before bandwidth does, so **Q4_K_M's
ceiling is a compute ceiling wearing a bandwidth costume.** Quote 54 GB/s only
for Q8_0; **for Q4_K_M the number is ~44.8 GB/s.**

Theoretical remains 76.8 GB/s. The game-window 34 may not be quoted as W.

## 1.2 The thread-peak claim does NOT reproduce — correct the map

ROADMAP §M4 states: *"note W PEAKS AT 6 P-THREADS — 10 threads measured 15%
WORSE, so the budget's N is not the core count."*

**Neither file peaks at 6 today. Both rise monotonically through 10 threads**,
and 10T is *better* than 6T — Q4_K_M by 2.8 %, Q8_0 by 6.7 % — not 15 % worse.
Variance is too tight (±0.02–0.27) for this to be noise.

A hypothesis was tested and **rejected**: that Q8_0, being memory-bound, would
contend and peak while Q4_K_M, being unpack-bound, would keep scaling. Q8_0
does not peak either, so the difference is not memory-bound vs compute-bound.
The most likely remaining explanation is that the original measurement was
taken on a box that was not quiesced — high thread counts amplify contention —
but that is a hypothesis, not a finding, and nothing here needs it resolved.

**What survives of the original claim, and it is the part that matters:** the
budget's N is still not the core count. Returns are sharply diminishing past
4–6 threads — Q4_K_M gains **63.3 %** going 2→4, then only **11.6 %** over the
whole of 4→10. **Use N = 6 for budget lines**, which captures **97.3 %** of
achievable W, and stop treating 10 as a penalty.

*The ROADMAP line should be corrected. It is a measured constraint, not
criterion text, so it is not §8.4-reserved.*

## 1.3 The per-step traffic model

Geometry read from the Q4_K_M GGUF, not recalled: 36 layers, 32 heads, **8 KV
heads** (GQA 4:1), head_dim 128, embedding 2560.

    KV bytes per token = 2 (K,V) × 8 kv_heads × 128 × 36 layers × dtype
                       = 73,728 elements
                       = 147,456 B @ f16  (144.0 KiB; 294,912 @ f32, ~73,728 @ Q8)

One decode step over a batch of N sequences each at context L:

    traffic(N, L) = model_bytes  +  N × L × kv_bytes_per_token
                     └ read ONCE per step, amortized across all N
                                       └ per sequence, NOT amortized

**Batching amortizes only the weight term.** That is the whole reason
continuous batching wins, and the whole reason it stops winning.

## 1.4 The KV-dominance crossover

KV traffic overtakes weight traffic when `N × L × 147,456 = 2.4973e9`:

    N × L = 16,936 token-positions      (Q4_K_M, f16 KV)

| N | crossover L | M5's operating point (L ≈ 1–2k) |
|---:|---:|---|
| 4 | 4,234 | weights dominate |
| 8 | 2,117 | borderline at 2k |
| **16** | **1,059** | **KV dominates** |

**At M5's stated trace — bursty, shared 1–2k prefix, 16 concurrent — the KV
term is already the larger one.** M5's bars are therefore not a weight-
streaming problem; they are a KV-traffic problem, which is precisely what
paged KV exists to control. Q8 KV halves the KV term and moves the crossover
to `N × L = 33,872`, restoring weight-dominance across the whole M5 envelope —
that is the quantitative input to Kabir's M5 default call.

Throughput ceiling implied by the budget, at W = 43.6 GB/s (Q4_K_M, 6T):

| N | L | traffic/step | ms/step | aggregate tok/s |
|---:|---:|---:|---:|---:|
| 1 | 1,500 | 2.72 GB | 62.4 | 16.0 |
| 8 | 1,500 | 4.27 GB | 97.9 | 81.7 |
| 16 | 1,500 | 6.04 GB | 138.5 | 115.5 |
| 16 | 1,500 (Q8 KV) | 4.27 GB | 97.9 | 163.5 |

These are ceilings from traffic alone — no compute, no scheduling, no
overhead — so they are upper bounds on what M5 can reach, not predictions.

## 1.5 The E-core datum that did not reproduce, and what it costs

The thread-peak claim traces to one measurement (ROADMAP §M3, 08-03 K3-entry,
quiesced, grade GATE): llama Q8_0 **1T 3.574 / 6T 12.645 / 10T 10.802 tok/s**,
read as *"10 THREADS IS SLOWER THAN 6 … adding the 4 E-cores COSTS 15%, which
is the forward note's P/E-straggling trap measured on the opponent's own
binary — direct evidence for LOCKED decision 'work-stealing 64-row chunks'."*

Today, same box, same binary family:

| threads | 08-03 | 2026-08-11 | delta |
|---:|---:|---:|---:|
| 1 | 3.574 | 3.49 | −2.4 % |
| 6 | 12.645 | 12.62 | −0.2 % |
| **10** | **10.802** | **13.47** | **+24.7 %** |

**Two of the three points reproduce almost exactly; only 10T does not.** The
topology assumption behind the reading is correct — the CPU is an i5-13450HX,
6 P-cores + 4 E-cores — so `-t 10` really does span both core types. But
E-cores did not cost 15 % today; they *gained* 6.7 % on Q8_0 and 2.8 % on
Q4_K_M.

A single depressed datum in an otherwise-reproducing triple is a box-state
signature, and this milestone has the precedent on record: `38b4f5a`
(2026-08-03, the same week) reopened ruling D3 because *"its central
measurement was a depressed box state."* Today's power plan is "Turbo"; what
it was on 08-03 cannot be recovered. **Hypothesis, not finding.**

**What this costs the LOCKED decision: less than it first appears, and the
distinction is worth being precise about.** "Work-stealing 64-row chunks" was
justified by E-cores *subtracting*. They do not subtract — but 4 extra cores
(+67 % core count) buy only +6.7 %, which is still straggling, just the
non-negative kind. **The decision survives; its cited evidence does not.**
Static equal partitioning across heterogeneous cores strands work either way,
and that is the argument to keep. The ROADMAP's parenthetical should stop
claiming a 15 % penalty.

## 1.6 M5's bars checked as budget lines — the ROADMAP's required check

The bars are **aggregate** tok/s on Q4_K_M (forward-designs note: "S2
aggregate ≥17.7 @8 AND ≥14.2 @16 AND ≥10.8 @8 / ≥16.8 @16"). Two families:
llamacpp-tuned parity (17.7 / 14.2, from M0's measured baseline) and
2×-ollama (10.8 / 16.8; the 16.8 is M0's unexplained N=16 inversion).

Against the traffic ceilings of §1.4 (W = 43.6 GB/s, L = 1,500, f16 KV):

| N | budget ceiling | binding bar | slack |
|---:|---:|---:|---:|
| 8 | 81.7 tok/s | 17.7 | **4.6×** |
| 16 | 115.5 tok/s | 16.8 | **6.9×** |

**M5's bars are nowhere near the bandwidth ceiling.** Whatever makes M5 hard,
it is not weight streaming — it will be per-core compute efficiency and
scheduler behaviour. That is the answer the ROADMAP wanted before paged KV is
scoped, and it means paged KV should be designed for *correctness and
teardown cost*, not for bandwidth.

**One correction this forces on the headline premise.** M0's "best concurrent
baseline is 11 % of the ceiling" implies a ~9× opportunity, and that ceiling
amortizes weights while ignoring KV traffic: 8 × 43.6 / 2.4973 = 139.7 tok/s,
against which 17.7 is 12.7 % — M0's number, reproduced. But at **M5's own
agentic trace** (N=8, shared 1–2k prefix) the KV term is real and the ceiling
is 81.7, against which 17.7 is **21.7 %**. **The opportunity at M5's operating
point is ~4.6×, not ~9×.** Both are large and the premise stands; but the 9×
belongs to short contexts, and quoting it for the agentic trace overstates the
room by roughly half. Worth stating plainly before M5's bars are judged.

## 1.7 What this does and does not settle

- **Settles:** W for both files; N = 6 for budget lines; the crossover formula
  and its value; that Q8 KV is a *structural* lever at M5's operating point,
  not a micro-optimization.
- **Does not settle:** whether M5's bars are per-stream or aggregate — they
  must be read against these lines before paged KV is scoped (ROADMAP §M4
  requires exactly that check, and it is the next task).
- **Untouched:** the §8.1 EXACT-vs-device collision, which needs a device
  measurement and is M4-entry work with its own decision record.
