# The prefill chunk width — 128 (locked at M5) or 256 (the graded configuration)? Analysis and recommendation

Kabir, 2026-09-15: *"analyze which chunk size is the best for us, pros and
cons, which aligns with our end goal."* Decided under the 2026-08-07
governance: Opus recommends and records the reasoning; **Kabir rules**,
because `SCHED_CHUNK_TOKENS = 128` is an M5 lock (S-ALIGN set; `sched.c:229`
calls the env override "for MEASURING the co-run tradeoff rather than for
shipping a different value"). Nothing in the tree changes with this memo.

## 1. What the chunk is

One fused step = every runnable decode row + at most ONE chunk of the single
active prefill (invariant S1). The chunk width sets how many prompt rows that
step carries. Wider: the weight stream and the per-step overheads are
amortised over more rows and the prefill phase takes fewer steps; but every
decode row riding in that step waits for the whole chunk's forward — that wait
IS the "decode stall from a co-running prefill" of gate [3].

## 2. The numbers — same sitting, in-process S2 @8, 10 threads, new kernels' predecessor (2026-09-15 screen; drift control 0.3%)

| | chunk 128 (locked) | chunk 256 | 512 (2026-09-08 sweep) |
|---|---|---|---|
| end-to-end wall | 86.4 s | **83.5 s (−3.4%)** | +1.2% vs 128 — not monotone |
| prefill steps | 40 | 24 | — |
| decode rows riding on prefill steps (cost ~40 ms each vs ~29 in the tail) | 140 | 84 | — |
| TTFT p50 / p95 | 35.7 / 59.9 s | **34.2 / 56.6 s (−4% / −6%)** | — |
| AGGREGATE(M0) | 12.72 | 13.16 (+3.5%) | — |
| ITL cadence p50 / p95 (decode-only steps) | 220 / 248 ms | 218 / 245 ms | — |
| **ITL stall p50 / p95 / max** (a decode gap that contains a chunk) | **1.84 / 2.17 / 2.31 s**, 140 stalls = 13.8% of gaps | **3.34 / 3.90 / 3.91 s**, 84 stalls = 8.3% of gaps | — |
| total stalled gap-time per run (n × p50, rough) | ~258 s | ~281 s | — |

Over HTTP, 12T + chunk 256 against the 10T/128 default read 1.048× (3/3 pairs,
drifting box), and the clause-2 grade that PASSED at 1.181× (five of five
pairs) ran with chunk 256. The 09-08 sweep already showed the curve turns
over at 512, so 256 is the width, not "wider is better".

## 3. Pros and cons

**For 256:**
- Every agent-shaped number improves in the same sitting: wall −3.4%, TTFT
  p50 −4% / p95 −6%, aggregate +3.5%. Fewer steps means fewer decode rows pay
  the dear ~40 ms "riding" rate and fewer per-step overheads (16 steps × 252
  dispatches × ~50 µs alone is ~0.2 s).
- It is the configuration that was graded. Shipping 128 would put the
  shipped default ~2–3% below the graded number — the 18% margin survives, but
  the ROADMAP row would describe a configuration users do not get.
- S-ALIGN holds (256 is a multiple of the 16-token KV block); the prefix cache
  still registers whole blocks at chunk ends; scratch grows by a few MB.

**Against 256:**
- The stall a co-running decode suffers is **1.8× longer** (1.84 → 3.34 s at
  p50; max 3.9 s). Gate [3]'s 250 ms bound is already a documented structural
  miss at EVERY width (chunk 16 measured 1.1 s on 08-26; even 64 read a 738 ms
  p50 on 09-08) — 256 does not change its status, it deepens it from ~7× to
  ~13×. The total stalled time per run barely moves (~258 → ~281 s); what
  changes is the shape: fewer, longer freezes in a live token stream.
- A request arriving mid-prefill-phase waits for the current step to end
  before its own chunk can start: up to ~3.4 s at 256 against ~1.8 s at 128, a
  tail cost on TTFT for bursty arrivals (not measured; S2 is a closed loop with
  all eight present at t=0).
- @16 is unmeasured under either width. The stall length does not depend on N
  (it is the chunk's forward time), but at N=16 twice as many decode rows ride
  in each prefill step.

## 4. Which aligns with the end goal

The end goal is stated twice in the ROADMAP: Phase 1's positioning clause is
graded on **wall against llama-server-tuned**, and the concurrency "is FOR
agents (Fable 08-02): on a consumer box N=8/16 is one user's swarms, tool
chains, and fan-outs … not eight humans". The same ruling asked for **p95
request completion, recorded not gated**, as the agent-shaped number, and
called the ITL bar "human-shaped". An agent waiting on a tool-call round trip
sees TTFT and completion; it does not watch the stream's cadence. Chunk 256
improves both of the numbers an agent sees and worsens the one a human
reader would see.

The stall class itself is not a chunk-width question. Its fix — a scheduler
that does not hold decode rows behind a chunk's whole step (the [3] options'
"(c) undesigned"; sub-steps of the chunk interleaved with decode-only steps,
or decode rows dispatched ahead of the chunk's tail) — is exactly the
scheduler work this project calls its moat, and once it exists the chunk
width becomes a pure throughput knob with no stall cost at all. Until then,
the width trades a human-shaped number for agent-shaped ones.

## 5. Recommendation

**Ship 256.** Move `SCHED_CHUNK_TOKENS` to 256 (S-ALIGN keeps binding;
`STOKE_SCHED_CHUNK` stays the measuring override), record in the ROADMAP's
gate [3] text that the stall p50 at the shipped width is ~3.3 s at @8, and
put "decode not held behind a chunk" on the scheduler queue as the item that
retires the trade. The gate-[3] bar is not re-anchored and not quietly met:
it stays a documented miss with a measured, now larger, price.

**If Kabir prefers the stream to stutter less**, keep 128: the predicted cost
is ~2–3% of an 18% margin, and the next clause-2 grade must then re-run at the
shipped width so the row describes what ships.

**What would change this:** an @16 sitting showing 256 losing its wall gain
(twice the riding rows per chunk step), or bursty-arrival TTFT tails growing
by more than the ~1.6 s the step length predicts.
