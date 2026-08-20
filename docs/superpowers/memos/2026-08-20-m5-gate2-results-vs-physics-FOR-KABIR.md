# M5 gate item [2] — the bars are MISSED, and the cause is arithmetic

**2026-08-20, Opus 5, branch `m5-entry`. Results-vs-physics memo, the category
the M5 playbook named in advance for exactly this outcome: "write the memo to
Kabir with the curve rather than grinding blind."**

Prediction and falsifiers were committed before the first cell was read
(`2026-08-20-m5-gate2-prediction-PREREGISTERED.md`, `0b7f260`). One falsifier
fired. The headline was right for a reason that was half wrong, and the half
that was wrong is the more important finding.

## 1. The verdict

| cell | AGGREGATE(M0) | bar | | at thermal equilibrium (§2b) |
|---|---|---|---|---|
| serve N=8, f32 KV | **6.378** | 17.7 | **MISS, 0.36×** | 4.659 → 0.26× |
| serve N=16, f32 KV | **5.905** | 16.8 | **MISS, 0.35×** | 4.594 → 0.27× |

The bar is missed by 2.8× on a cool box and by 3.8× on a box held under
sustained load — which is the condition a serving engine actually runs in, and
worth stating because the gate's methodology line says "quiesced" and a quiesced
box is the *kindest* state, not the operating one.

M0's counting throughout: decode-only ids over the span from the first token
anywhere to the last token anywhere, which is how the 17.7 / 14.2 / 10.8 / 16.8
cells were computed. `bench --serve` was printing `total ids / total wall`; that
was fixed on the way in and it changes the number by ~9%, not the verdict.

*(Block 1 of 3, f32 cells. Medians across all three blocks and the q8 cells
follow in the gate report; the calibration cell at q8 read 6.024, so these are
stable to ~5%.)*

## 2. What actually binds, and it is not what the budget said

**`stoke batch`, the static M4 path, decodes at 23.071 tok/s at N=8 and
22.653 at N=16.** Flat. The traffic model predicted a 1.23× rise.

Convert both to arithmetic rate: **185.5 and 182.1 GFLOP/s** — the same number
twice, 1.8% apart. Time scaling exactly with FLOPs while the row count doubles
is the signature of a compute ceiling, and nothing else produces it. Traffic
predicts 89 tok/s at N=8 and 131 at N=16: wrong in level and wrong in shape.

`tests/tooling/m5_compute_budget.py` prices this from the GGUF's own tensor
table plus **E3's 195 GFLOP/s, measured a day earlier on a different workload
(prefill)**:

```
  matvec       8.045 GFLOP/token   (253 two-dimensional weight tensors)
  attention    0.415 GFLOP/token   (at ctx 704)
  TOTAL        8.460 GFLOP/token
  => a COMPUTE ceiling of 23.1 tok/s aggregate, flat in N

   N   traffic ms   compute ms   binds        ceiling
   1        62.0         43.4   traffic         16.1
   2        66.8         86.8   COMPUTE         23.1
   8        95.4        347.1   COMPUTE         23.1
  16       133.5        694.1   COMPUTE         23.1

  traffic and compute cross at N = 1.48
```

Predicted 23.1, measured 23.071 and 22.653. **A decode step at N rows streams
the weights once but does N × 8.46 GFLOP, and arithmetic does not amortise.**

**A third control, and it fell out of the matrix rather than being designed
for.** The gate cells run f32 KV and Q8 KV. `--kv q8` cuts KV bytes per context
position from 512 to 136 — **3.76× less traffic through the term that grows
with N and ctx**, worth 1.41× if traffic bound and 1.00× if compute bound — and
the static decode at N=8 measured **23.071 f32 vs 22.980 q8, 0.4% apart**.

### 2a. ⚠ AND THEN THE BOX THROTTLED, WHICH COSTS ME TWO OF THE THREE

**Measured, not suspected.** `% Processor Performance` read 143.1 → 132.6 →
126.0 across three consecutive one-second samples 45 minutes into the run —
throttling in progress — while core occupancy held at 6.70 against block 1's
6.71. Same cores, same occupancy, fewer cycles. The llama arm, which is in the
matrix precisely to be a non-stoke thermometer, splits exactly as clock
throttling predicts and contention does not:

| | early | 50 min in | |
|---|---|---|---|
| llama `pp512` (compute-dense) | 91.30 | 71.29 | **−22%** |
| llama `tg128` (memory-bound) | 18.12 | 17.03 | −6% |
| stoke `8f32-S` b1 → b2 | 6.378 | 4.659 | **−27%** |

**Cells run sequentially, so every CROSS-CELL comparison in §2 spans 6–19
minutes of a monotonically heating box.** Correcting with llama's pp curve
(≈0.44%/min, provisional — two points, and a different workload):

| comparison | raw | drift-corrected | traffic says | compute says |
|---|---|---|---|---|
| N=16 ÷ N=8 | 0.982× | **≈1.04×** | 1.23× | 1.00× |
| q8 KV ÷ f32 KV | 0.996× | **≈1.10×** | 1.41× | 1.00× |

**Both still land far nearer compute than traffic, and neither is now clean
enough to quote as "flat" or "0.4%".** I wrote both of those before the
thermometer had two points; that was wrong, and the raw numbers are kept above
rather than edited away. What survives untouched is the comparison that needs
no cross-cell arithmetic: **the LEVEL**. `b1-8f32-B` was taken 15 minutes in at
≈6.6% drift, and 23.071 against a 23.1 ceiling predicted from a different day's
workload is not a ratio between two of my own cells.

**⇒ OWED: a counterbalanced re-run** — B arm only, short prompts so a cell is
under 90 s, cell order reversed on alternate passes so linear drift cancels
between a forward and a reverse pass rather than accumulating
(`tests/tooling/run_m5_flatness.ps1`, with the clock banked per unit).

### 2b. AND THEN THE BOX STOPPED DRIFTING, WHICH GIVES THE CLEAN ANSWER ANYWAY

The llama arm runs once per block, so it dates the drift as well as measuring
it: **`b1-L` pp 71.29 → `b2-L` pp 72.24, 49 minutes apart — 1.3%.** The box hit
thermal equilibrium during block 1 and stayed there. So block 1 is the warming
transient and **block 2 is a stable-clock replicate in which every cell is
comparable to every other**, which is precisely the design the sequential
matrix could not otherwise supply.

Block 2, decode-only aggregate, all four configurations at one clock:

| N | KV | measured | traffic predicts |
|---|---|---|---|
| 8 | f32 | **17.367** | 83.9 |
| 16 | f32 | **17.936** | 119.9 |
| 8 | q8 | **16.130** | 118.7 |
| 16 | q8 | **18.219** | 206.4 |

**Traffic predicts these four spread over 2.46×. They span 1.13×.** Doubling the
rows and cutting KV traffic 3.76× — which between them are worth 2.46× if
bandwidth binds — move the number by 13%, and not monotonically. At N=8, Q8 KV
is **7% SLOWER** than f32 despite 3.76× less traffic, which is what a compute
wall plus Q8's extra unpack work looks like and is not something a
bandwidth-bound engine can do at all.

The LEVEL agrees too, and by an independent route: stoke's own cool-vs-plateau
ratio is 17.367/23.071 = 0.753 and llama's is 72.24/91.30 = 0.791 — one cause,
both engines. Scaling the cool-box compute ceiling of 23.1 by that gives
**≈17.4–18.3 against a measured mean of 17.41**.

*(The entry note's §6 Q8-KV call — "the answer to RAM, not to speed" — is
confirmed outright: at N=8 it is 7% slower, and the note's justification was
capacity.)*

### 2c. The counterbalanced run, and its prediction written first

`run_m5_flatness.ps1`, N ∈ {1, 2, 4, 8, 16}, four counterbalanced passes,
ctx 192, launched at 12:08 on the plateau. **Written before the result.**

The compute model does not merely say "flat" — it puts a **knee at N = 1.48**,
because at N=1 the weight stream (57.3 ms at this ctx) still costs more than the
arithmetic (41.8 ms), and from N=2 up it never does again. So:

| N | compute model (plateau) | traffic model (plateau) |
|---|---|---|
| 1 | **≈13.1** (traffic-bound) | ≈13.1 |
| 2 | **≈17.9** | ≈25.0 |
| 4 | **≈17.9** | ≈46.8 |
| 8 | **≈17.9** | ≈81.0 |
| 16 | **≈17.9** | ≈126 |

**A rise from N=1 to N=2 and then dead flat.** The two models agree at N=1 and
diverge 7× by N=16, and the shape — a knee then a plateau, rather than a curve
that keeps climbing — is a thing only the compute account predicts.

**Falsifier:** N=4 or N=8 landing above ~21 (i.e. more than 20% over the N=2
value) means the flatness is not flat and the crossover is somewhere else.

**⇒ THE AGGREGATE TRAFFIC CEILINGS IN THE M5 ENTRY NOTE — 57.8 at N=8, 72.9 at
N=16 — DESCRIBE A MACHINE THAT DOES NOT EXIST FROM N=2 UP.** `m5_traffic_budget.py`
ends its ceiling table with the words "upper bounds: traffic only, no compute,
no scheduler". That caveat was honest and it was never cashed. The real ceiling
is 23.1, and the bars sit at **77% and 73% of it** — which is the same "a bar at
74% of a ceiling is a physics claim, not a bar" that made E2 structural at
entry, arriving a second time about a different ceiling.

**This does not retract E2.** E2's measured 1.12× at ctx 1500 is real and its
mechanism was verified by shape. It says the slack E2 reported was not there to
be spent.

## 3. Why the scheduler reads 6.4 when the static path reads 23.1

Not a scheduler defect, and `preemptions 0` in every cell says the pool was
never the constraint. It is the metric doing exactly what it did to
llama-server:

- A **lockstep batch** finishes all 8 prefills before any first token, so its
  M0 span contains **no prefill at all**.
- A **continuous batcher** — llama-server, and `sched.c` — has every later
  arrival's prefill land **inside** the span.

M0 measured llamacpp-tuned at 17.7 while its own ITL p50 (211.8 ms) implies
37.8 tok/s of decode. Same haircut, same reason. So the comparison that means
something is serve-vs-llama-server, both continuous: **6.378 against 17.7**.

Reconstructing the measured span from the two measured rates —
prefill 32.229 tok/s, decode 23.071 tok/s:

```
  span = 3675/32.229 + 1016/23.071 = 114 + 44 = 158 s  =>  6.43 tok/s
  measured: 6.378                                          (0.8% apart)
```

The model reproduces the measurement, so it can be used to ask what would fix it.

## 4. What it would take, and the number is 2.75×

Holding the workload and the metric fixed, and scaling both prefill and decode
by a factor *f*: the span is 158/*f*, and 17.7 needs span ≤ 57.4 s.

> **stoke needs a uniform 2.75× on arithmetic to meet the 17.7 bar.**

Two things follow, and they point the same way:

1. **The headroom exists on this box, and llama.cpp is standing in it.**
   llama-bench `pp512`, measured in the same sitting, is **91.3 tok/s = 772
   GFLOP/s effective — 3.96× stoke's 195.** The required 2.75× is inside a gap
   a competing engine demonstrates on the same silicon and the same file.
2. **Prefill and decode-at-N are one defect, not two.** stoke prefills at 259
   GFLOP/s and decodes at 185; llama does 772. Fixing one fixes both, and no
   scheduler change touches either.

**Where the 3.96× lives, as far as the evidence reaches:** llama.cpp repacks
Q4_K into `block_q4_Kx8` at load time and runs a register-tiled GEMM over it.
That is already on this project's books — the 08-12 parity ruling names it as
"step D (repack)", priced at ~15% at N=1 and ≈ +2.3 GB RSS, and the P-BAR was
deliberately set at 0.85× so it could be met *without* it. At N=1 that pricing
was right, because N=1 is bandwidth-bound (the table above: traffic 62.0 ms vs
compute 43.4). **From N=2 up the same change is worth up to the whole 3.96×,
because the term it attacks is the one that binds.** A repack was priced against
the regime where it mattered least.

## 5. Three open items are now one decision

Each of these was already open, and each is the same trade:

| item | worth | costs |
|---|---|---|
| SIMD in the attention dot products (open with you since 08-19) | the residual 0.0224 ms/context-token slope | bit-exactness, a re-anchor |
| a repacked / register-tiled GEMM | up to 3.96× on the term that binds both halves of gate [2] | bit-exactness, ≈ +2.3 GB RSS, a re-anchor |
| gate [3]'s 250 ms ITL bound | needs a cheaper step; the chunk ladder is exhausted at 16 | — |

Gate [3] is downstream of the same thing: a decode co-running with a prefill
waits the chunk's whole step, and this run measured **ITL max 4713 ms at N=8 and
5454 ms at N=16** against the 250 ms bound, on the real S2 shape with the
quiesce gate passing at 11%/7% — the 08-19 memo reached the same conclusion from
an indicative 4-seq run. Make the arithmetic 2.75× cheaper and the chunk's step
shrinks with it.

**⇒ These three collapse into one question: does stoke keep bit-exactness with
the M3/M4 parity lineage, or does it buy the arithmetic?** — but see §5b, which
I wrote after reading the kernel and which argues that question is **premature**.
The constraint binds a narrower region than the framing above assumes, and the
3.96× is measured but not yet attributed. I have not started either path.

### 5a. Two guesses I made about WHERE the gap lives, and both are wrong

I wrote them down, then read `vec_dot_q4_k_q8_k_xn_avx2` and
`matvec_batch`. Recording the corrections because they narrow the question.

**Guess 1 — "llama.cpp repacks to amortise the Q4_K unpack across eight rows;
stoke unpacks per row." Wrong: stoke already amortises it, across SEQUENCES.**
That is what M4.5's rung B did, and the kernel says so at the seam —
`WEIGHT-ONLY WORK, ONCE PER ROW instead of once per sequence`: the scales, mins
and nibbles are unpacked once per weight row and reused for up to 16
activations. At N ≥ 2 there is no unexploited unpack-amortisation lever on the
sequence axis. (llama's repack works the *output-row* axis, which stoke does
not, and which also helps at N=1 where stoke has only one sequence to amortise
over. M4.5's rung A′ tried 4-row blocking and measured **0.907×, a 9.3%
slowdown**, rejected on that measurement — taken at N=1, where traffic binds
and the regime is not this one.)

**Guess 2 — "run a Q8_0 file to separate unpack cost from multiply cost."
Invalid, and I nearly burned box time on it.** `matvec_batch`'s Q8_0 path still
delegates one call per sequence; only the K-quants got rung B's fused twin. A
Q8_0 batch at N=8 would re-read the weights eight times and I would have
measured traffic while believing I was measuring unpack.

### 5b. The constraint is narrower than "the arithmetic", and that may matter

Inside one super-block this kernel is **entirely integer**: `maddubs` →
`madd` → `mullo_epi32` → `add`, then one `hsum`. Floats appear only at the
super-block boundary, twice: `sumf += (dw·ad)·S_b` and `sumf -= (dm·ad)·sumi`.

Integer addition is associative and exact, so **any regrouping inside a
super-block is bit-exact by construction; only the f32 accumulation ACROSS
super-blocks is pinned.** This is not my observation — it is the codebase's own
founding principle for these twins, written at `simd.c:75`: *"Exact at any
order -- integer addition is associative, which is the entire reason the
integer core makes a bit-exact SIMD twin possible at all."*

So "keep bit-exactness or buy the arithmetic" — which is how I framed §5 — is
**too pessimistic as a dichotomy**. An unknown share of the 3.96× sits inside
the integer core, where it is reachable without giving up a bit. One visible
candidate, unmeasured and named only as a candidate: the per-sequence
per-super-block `sumi` loop is **16 scalar iterations** (`sumi += bs[j] *
mins8[j/2]`) sitting next to 8 vector ones, and it is a pure integer sum whose
operands cannot overflow i32 (|bs| ≤ 2032, mins ≤ 63, 16 terms ⇒ ≤ 2,048,256).

**⇒ REVISED RECOMMENDATION: do not make the bit-exactness decision yet.** It
should be made against an attribution of the 3.96×, and there isn't one — I
have the gap measured and its location unattributed. An attribution pass on the
kernel is a day's work with the existing bit-exactness tests as the guardrail,
and it would tell you how much of the 2.75× is available for free before you
are asked to trade anything for the rest.

## 6. What is NOT in doubt

- **N=1 is healthy and stays healthy.** Traffic binds there (62.0 ms vs 43.4),
  stoke measures 16.58 against a 16.1 traffic ceiling, and P-BAR passed at
  0.892× llama. None of this touches that.
- **The scheduler works.** Gate item [1] is met: 42 trace runs plus a
  1,000,130-step soak, every invariant after every step. In this run,
  `preemptions 0`, every request completed, and
  `m5_scheduler_output_equals_solo_generation` holds. **M5's batching is not
  what is missing.**
- **The static path already meets the aggregate bars on decode** — 23.071 ≥
  17.7 and 22.653 ≥ 16.8 — which is worth stating plainly, because it is the
  playbook's ladder rung (1) answering *no*: static did not miss, so the
  scheduler was never the problem.

## 7. What I would do next, if the call is "not yet, keep bit-exactness"

Nothing above needs deciding today for the milestone to keep moving:

1. Finish [2]'s medians and the q8 cells (running).
2. Measure the B arm at N ∈ {2, 4, 8, 16} in one sitting — the flat claim rests
   on two points plus a KV-traffic control, and four would draw the line.
3. **Attribute the 3.96×** (§5b) — the piece that would let you decide with
   numbers instead of with a principle.
4. Land [4] (S3 fanout ≥ 13.3) and [5] (the 1 h ASan/TSan soak), neither of
   which depends on any of this.
5. Re-anchor the bars, or don't — **that is your headline and your call.** The
   playbook already contemplated re-anchoring for the N=16 inversion; this is a
   larger version of the same question, and I am not going to move a bar to fit
   a measurement.
