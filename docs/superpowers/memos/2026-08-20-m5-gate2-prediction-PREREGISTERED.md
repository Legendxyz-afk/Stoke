# M5 gate item [2] — what I expect to measure, written before measuring

**2026-08-20, Opus 5, branch `m5-entry`. Written and committed BEFORE the
first bar cell was read.** The calibration cell was already running when this
was written and its output was deliberately not opened first.

Why pre-register at all: this session's own ledger records three separate
occasions in four days where a model of stoke's cost ran hot — the 08-11
crossover that described a different engine, E2's slope priced at half when
one third was still generous, and a 319 MB weight the traffic model did not
know it was reading. A prediction written afterwards cannot be wrong.

## The workload, fixed

M0's S2, unchanged: 512 nominal prompt tokens (**525 real** — `[client N] `
plus the ChatML template, measured through stoke's own tokenizer), 128
generated, N ∈ {8, 16}, no shared prefix. Prompts exported verbatim from
`bench/scenarios.py` into `tests/data/bench_prompts/`. ctx 704, threads 8,
unpinned, Q4_K_M, f32 KV (§6 of the entry note: f32 is the default and the
headline).

## The metric, and a defect fixed on the way in

The bars are M0 table cells and M0's `aggregate_decode_tps` is

```
sum over clients of (n_tokens - 1)
------------------------------------------------
max(last-token wallclock) - min(first-token wallclock)
```

— decode-only ids over the span from the first token *anywhere* to the last
token *anywhere*. `stoke bench --serve` was printing `total ids / total wall`
instead. On S2's shape that is not a rounding difference: the prompt is 512
tokens against 128 generated, so a metric that charges all prefill to a decode
bar is mostly measuring prefill. Fixed to print both, with the M0 one named as
the gate number. **It does not rescue anything** — on the numbers below the two
differ by ~12% and the verdict is the same either way — which is the reason it
is safe to fix at gate time.

## Inputs, all measured, all cited

| quantity | value | source |
|---|---|---|
| decode, N=1, post-E2 | `61.14 + 0.02242·ctx` ms/token | E2 A/B, 08-19 |
| prefill, chunk 128 | **~41 ms per prefilled token** (~24.4 tok/s), flat 87→1500 | E3 A/B, 08-19 |
| llama tg128, t=8 | 18.58 tok/s | P-BAR L4 t8, 08-19 |
| llamacpp-tuned N=8 | agg **17.7**, ITL p50 211.8 ms, TTFT p50 58.5 s | M0, 07-17 |
| llamacpp-tuned N=16 | agg **14.2**, ITL p50 403.5 ms, TTFT p50 89.7 s | M0, 07-17 |
| llamacpp-tuned N=1 | TTFT 7439 ms on 512 tokens ⇒ **~69 tok/s prefill** | M0, 07-17 |

## The prediction

Two phases. **A:** 5 prefill chunks per request (128×4 + 13), one active
prefill per step (S7), and a decode row co-running with a prefill chunk waits
that chunk's whole step (the 08-19 ITL memo). So 5N steps of ~5.3 s each, during
which request *i* emits one token per step from step 5(i+1) on.
**B:** pure decode, step ≈ `61.14 + 13.21·k` ms at k active rows.

| | N=8 | N=16 |
|---|---|---|
| phase A | ~179 s, 140 ids | ~366 s, 600 ids |
| phase B | ~15 s, 884 ids | ~21 s, 1448 ids |
| **AGGREGATE(M0)** | **~5.9** | **~5.6** |
| end-to-end | ~5.3 | ~5.3 |
| decode-only (N / ITL p50) | ~53 | ~65 |
| ITL p50 / max | ~150 ms / ~5.3 s | ~270 ms / ~5.3 s |
| TTFT p50 | ~96 s | ~172 s |
| `batch` decode aggregate | ~48 | ~58 |
| `batch` prefill | ~24 tok/s | ~24 tok/s |
| bar | 17.7 | 16.8 |
| **verdict** | **MISS, 0.33×** | **MISS, 0.33×** |

## The attribution I expect, stated as a claim that can be wrong

**Gate [2] misses, and prefill is why — not the scheduler and not decode.**
S2 puts 512 of every 640 tokens of work in prefill, stoke prefills at ~24 tok/s
against llama.cpp's ~69, and M0's span metric charges every arrival's prefill
after the first one to the decode bar. Decode itself should come out *ahead* of
the referent: ~48–53 tok/s at N=8 against llamacpp-tuned's implied 8 / 0.2118 s
= 37.8.

If that is what lands, M5's batching is not the thing to fix. The prefill path
is a tiled matvec running at ~195 GFLOP/s ≈ 22 % of this CPU's f32 peak, and
the gap to a register-blocked GEMM is the whole finding.

## Falsifiers, before the fact

1. **AGGREGATE(M0) ≥ 12 at N=8** ⇒ the prefill model is wrong by ≥2×; re-derive
   from measured TTFT rather than patching the story.
2. **ITL p50 > 1 s** ⇒ phase B is not amortising and the decode half of this
   model is wrong too.
3. **`batch` decode aggregate < 25 tok/s at N=8** ⇒ decode is the binding term,
   the attribution flips, and the playbook's ladder rung (1) fires: static
   already missed, so the scheduler was never the problem.
4. **`batch` prefill > 45 tok/s** ⇒ E3's 41 ms/token does not describe the
   static path; the prefill story needs re-deriving before it is told.
5. **llama `tg128` outside 16.7–20.4** (18.58 ±10 %) ⇒ the box, not the engine.
   Nothing in this run is quotable until that clears.

A miss on 1–4 is a finding to write up, not a number to argue down.
