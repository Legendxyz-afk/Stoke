# Phase-1 clause 4 / M5 gate item [3] — "no >250 ms decode stall from a co-running prefill": DOCUMENTED MISS

**Ruling:** Kabir, 2026-09-16 ("mark it as a documented miss with ample proofs"),
on Opus's recommendation of the same day. **The clause is NOT met, the miss is
structural to the scheduler's one-step model, no tuning reaches the bound, and
the fix — the scheduler overlap redesign — is deferred to Phase 1.5 as its first
scheduler item.** This memo is the authority for the disposition; the numbers
below are citations of banked runs, not new measurements.

## 1. What the clause asks and what is measured

The clause (ROADMAP "Phase-1 goal"): no decode stall longer than 250 ms while
another request's prompt is being prefilled. The instrument (ruled 2026-08-26,
`memos/2026-08-26-m5-gate3-criterion-RULING-OPUS.md`): the `ITL stall` line of
`stoke bench --serve` — a decode gap that contains a prefill chunk — graded on
its p50, timed at the consumer, one step's granularity, with `cadence` (a
decode-only gap) reported beside it.

## 2. The proofs — every measurement of the stall this project has made

| date | run | chunk | stall p50 | cadence p50 | source |
|---|---|---|---|---|---|
| 2026-08-26 | M5 gate [3], first grading | 128 | ~3.9–4.5 s | — | `gate-reports/2026-08-26-m5-gate3-report.md` (P1 held) |
| 2026-08-26 | same, chunk 16 (the linear-law probe) | 16 | **1,108 ms** | — | same report, P5 FALSIFIED: a 16-token chunk does not cost 1/8 of a 128 one |
| 2026-08-28 | M5 gate [3], re-graded on a quiet box, pre-registered at `23fcb35` | 128 | 3,870 (dedup) / 3,325 (no dedup) | 310 / 334 ms | `gate-reports/2026-08-28-m5-gate3-report.md` |
| 2026-08-28 | same, chunk 16 | 16 | **678 ms** | 308 ms | same report, cell C16 |
| 2026-09-08 | `STOKE_SCHED_CHUNK` sweep, one binary, one block | 64 | **738 ms** | ~171 ms | SEQUENCE §I.4 (the knee): 64 → 738, 128 → 1,287, 256 → 2,450 ms; halving the chunk costs 4.8% end-to-end, doubling buys 0.06% |
| 2026-09-15 | same-sitting 128 vs 256 @8 (the width decision) | 128 / 256 | **1.84 s / 3.34 s** (p95 2.17 / 3.90) | 220 / 218 ms | `memos/2026-09-15-prefill-chunk-width-DECISION-OPUS.md` §table; 256 SHIPPED since `56ec57d` |

Three independent gradings, five widths from 16 to 256, two dedup arms, three
binaries a fortnight apart: **the stall is above 250 ms at every S-ALIGN-legal
width, and at the shipped width it is ~13× the bound.** The smallest legal
width (64: `BP_BLOCK_TOKENS` is 16 and M4 froze it; 16 was probed off-spec and
still read 678–1,108 ms) misses by 3×.

## 3. Why it is structural, not tuning

`sched.c` builds `n_decode + chunk` rows into ONE `SchedStep` and the host
runs them as ONE forward pass (`serve.c` → `model_forward_batch`). A decode
row that shares a step with a chunk is not slowed — `cadence` p50 sits at
171–220 ms at every width while `stall` scales linearly with the width — it
WAITS the whole step, and the step's length is the chunk's forward time. So
the stall equals the chunk's compute, and:

- **the chunk knob is at its knee in both directions** (09-08: halving costs
  4.8% wall, doubling buys 0.06%), and no legal width is under the bound;
- **a chunk small enough to fit 250 ms would collapse prefill** — the linear
  law does not even hold at 16 (P5 falsified: 678–1,108 ms, not 570), because
  a tiny chunk is bandwidth-bound like a decode step;
- **the four options were priced on 2026-08-19/26 and again on 09-01
  (SEQUENCE §2.2):** (a) 4.4× on prefill at chunk 16 — the arithmetic wall;
  (b) `BP_BLOCK_TOKENS` < 16 — reopens M4's closed set, sub-linear return;
  (c) a scheduler that does not hold decode rows behind a chunk's step —
  UNDESIGNED; (d) a documented miss. Only (c) can meet the bound.

## 4. Why (c) is deferred rather than built now (Opus, 2026-09-16; Kabir concurred)

1. **The bound is stoke's own aspiration, not a llama gap.** `llama-server`
   builds the same kind of batch — decode tokens ride with a prompt's chunk
   in one `llama_decode` — so its streams stall for seconds during a
   co-running prefill too. The @16/@32 sitting Kabir will run records the
   referent's ITL max beside stoke's; if it is also seconds, this clause is
   re-anchored to "≤ the referent's stall" and the miss is shared.
2. **True overlap is architectural, near the freeze.** Two concurrent
   forward passes need a pool that is not one global with a one-owner rule
   (`pool.c`), per-pass scratch instead of the model's single
   `views`/`att_ws` (`model.c`), a two-in-flight step model in `sched.c`,
   and every M5 gate re-run. Sized at 3–6 sittings plus re-gating.
3. **Its payoff on this box is bounded.** With 4 of 12 threads left to the
   decode while a chunk runs on the other 8, the N=8 decode step lands
   ~250–300 ms: the bound is met marginally at @8 and not at @16, where the
   plain decode step already exceeds 250 ms.
4. **The throughput half of the case weakened on 2026-09-16.** SEQUENCE
   §IV.6 row 1's "up to ~9 points" assumed decode is bandwidth-bound and
   prefill compute-bound, so they overlap on different resources. The
   09-16 kernel sitting found the matmul at 12T bound by the P-core's
   shared ports and the N=8 decode step ~60% compute (intercept ~74 ms +
   ~14.5 ms per row): the two phases would fight for the same ports.

## 5. What ships, what is owed, what comes next

- **Ships:** chunk 256 (Kabir's ruling 09-15, wall −3.4%, TTFT −4/−6%), with
  the stall priced and recorded at 3.34 s p50 / 3.90 s p95 @8. Streams are
  never starved — `cadence` is inside the bound — they pause for one chunk
  when a new request arrives.
- **Owed, in the @16/@32 sitting:** `llama-server`'s ITL max during a
  co-running prefill, same harness, same sitting — the re-anchor.
- **Next:** the overlap redesign is Phase 1.5's first scheduler item
  (SEQUENCE §IV.6 row 1, still UNDESIGNED), after the ARM-port decision —
  it is the moat's work and gets a full design cycle, not a pre-freeze rush.
  ROADMAP's Phase-1 paragraph and the M7 row carry this disposition; the
  README/report (B8) states the miss with this memo's table.
