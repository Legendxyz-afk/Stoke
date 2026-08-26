# M5 gate item [3] — the criterion says something narrower than we have been grading, and the instrument cannot report it

**2026-08-26, Opus 5. RULING + PRE-REGISTRATION, written and committed BEFORE
the measurement it predicts.** Authority: the 08-04 ruling's §8.4 as superseded
2026-08-07 — criterion TEXT resolves to Opus. Kabir's word today was "re-anchor
the 250 ms bound if you want to take it"; **this memo does not re-anchor it**,
and §3 says why that turned out to be the wrong thing to do.

---

## 1. What the criterion actually says

From `design-notes/2026-07-19-m3-m7-forward-designs-and-risk-playbook.md`, the
M5 gate line, verbatim:

> **[3] ITL: no decode stall >250ms during co-running prefill (harness
> percentiles + max)**

The ROADMAP's shorthand for this is `ITL <250ms`, and that shorthand is what
every later reading has actually used — including mine this morning, which
reported "[3] MISSES on the median as well as the max" from a **pooled** ITL
distribution.

**The full text names a narrower subject than the shorthand:** a decode *stall*,
*during co-running prefill*. Not the steady-state inter-token cadence.

## 2. The ruling: reading 1 governs

The text admits two readings.

- **Reading 1 — the stall attributable to a co-running prefill.** The subject is
  the wait a decode suffers *because* a prefill chunk shares its step.
- **Reading 2 — any inter-token gap in a run where prefills co-run.** "During
  co-running prefill" describes the scenario, and every gap is in scope.

**READING 1 GOVERNS.** Reasons, in order of weight:

1. **The project's own operative interpretation is reading 1, and predates the
   dispute.** The 08-19 memo analysed [3] as "a decode co-running with a prefill
   waits the CHUNK'S WHOLE STEP", and swept CHUNK WIDTH against it — 4463 /
   1209 / 624 ms at chunk 128 / 32 / 16. That analysis is only coherent under
   reading 1; under reading 2 the chunk sweep would not be the relevant knob.
2. **The words.** "Stall" denotes something anomalous. Under reading 2 the
   ordinary cadence of an 8-way batch is a "stall", which drains the word.
3. **Reading 2 makes [3] a second, unlabelled throughput bar.** In pure-decode
   steps the gap IS N / decode-rate; grading it is grading throughput, which
   item [2] already does with a bar of its own. A gate set should not grade one
   quantity twice under two different numbers.

⚠ **AN ARGUMENT I CONSIDERED AND REJECTED, recorded because it is seductive and
wrong.** I first reasoned: "[2] demands AGGREGATE(M0) ≥ 17.7 @8, which implies
per-stream spacing 8/17.7 = 452 ms, so under reading 2 the gate set demands
≤250 ms and ≥452 ms at once — self-contradictory, therefore reading 1." **That
does not hold.** AGGREGATE(M0) is span-based and its span *contains prefill*, so
an engine can read 17.7 aggregate while its pure-decode ITL is well under 250 ms
— the two are compatible if prefill occupies enough of the span. The
contradiction is imaginary and the ruling does not rest on it.

## 3. Why this is a clarification and NOT a re-anchoring

Under reading 1 the bound stays **250 ms**. Nothing here lowers a bar, and
nothing here makes [3] pass — §5 predicts it still misses, by ~2.3× at the
smallest legal chunk. Kabir offered the re-anchor; the honest finding is that
the criterion did not need re-anchoring, it needed reading. **A bound that is
missed for a well-understood reason is more useful than a bound moved until it
is met.**

What DOES change: **which number is compared to 250 ms.** The pooled ITL median
is the cadence and is out of [3]'s scope. My 08-26 statement that "[3] misses on
the median as well as the max" is **withdrawn** — the max leg was always in
scope and always missed; the median leg was never [3]'s subject.

## 4. ⚠ The instrument could not report the governing quantity, and its pooled
percentiles are a mixing ratio

`bench --serve` pooled every consecutive-token gap into one distribution. That
distribution spans two populations — stalls and cadence — so **a percentile read
off it depends on how many prefill chunks the run happened to contain**, not on
any bound being met or missed.

That is exactly what produced this morning's most quoted number. Dedup moved ITL
p95 from 4,290 ms to 438 ms (9.8×) and left the max at ~5,100. Under the
mixing-ratio account that is not a latency improvement at all: dedup cut prefill
chunks 80 → 24, so the stall population fell from ~27% of gaps to ~8% — from
above the 95th percentile to below it. **The stalls did not get shorter. They
got rarer, and the percentile crossed them.**

Fixed in `src/main.c`: each token records whether the step that produced it also
carried a prefill chunk (`SchedStats.prefill_chunks` moving across the step),
and the driver now prints three lines — the pooled one unchanged, plus
`ITL stall` (gate [3]'s subject) and `ITL cadence`.

## 5. PRE-REGISTERED PREDICTIONS — written before the first cell

Model under test: *a decode sharing a step with a prefill chunk waits that whole
step, so the stall length is set by CHUNK WIDTH ÷ PREFILL RATE and is
independent of how often it happens.*

| # | prediction | accept band | falsifies |
|---|---|---|---|
| **P1** | stall p50 at chunk 128 ≈ 128 / 28.2 tok/s ≈ **4.5 s** | 3.5–5.5 s | the whole "waits the chunk's whole step" model |
| **P2** | stall p50 is the **same in the dedup and no-dedup arms** | within 20% | "dedup changes frequency, not magnitude" |
| **P3** | stall share of gaps ≈ chunks / n_gen: **~8% with dedup, ~27% without** | ±5 pp | the mixing-ratio account of the 9.8× p95 move |
| **P4** | cadence p50 ≈ **403 ms**, and the same in both arms | ±15% | that cadence is prefill-independent |
| **P5** | at `STOKE_SCHED_CHUNK=16`, stall p50 ≈ 16 / 28.2 ≈ **570 ms** | 400–800 ms | the linear chunk-width law |

**DISCRIMINATOR (instrument, not finding):** if `n_stall` comes back ≈ 0 or
≈ `n_gap`, the conditioning is broken and NOTHING above may be reported.

**PREDICTED VERDICT: [3] MISSES, and it is the third independent measurement of
the kernel gap.** 250 ms at stoke's ~28 tok/s prefill needs a chunk of ~7
tokens, below the 16-token KV-block alignment M4 froze — so the smallest LEGAL
chunk misses by ~2.3×. ⚠ Stated as an estimate and not a measurement: the same
16-token chunk at the referent's prefill rate (llama-bench pp512 reads ~93
tok/s here) would be ~170 ms and would pass — but neither engine's rate was
measured AT chunk 16, and small chunks amortise worse for both, so this
comparison is indicative only.

## 6. What a green [3] would require, stated so nobody has to re-derive it

Either **(a)** prefill ~2.3× faster at the smallest legal chunk — the same
arithmetic wall as [2], already priced at ≤18% from both named compute levers
together, so not reachable through them; or **(b)** `BP_BLOCK_TOKENS` below 16,
which reopens M4's closed set; or **(c)** a scheduler that splits a prefill
chunk's step so decode rows are not held behind it — not designed, and it
trades against the amortisation E3 measured saturating by ~23 rows.

**Kabir's decision, if he wants one, is now a real choice between (a)/(b)/(c)
and accepting [3] as a documented miss with a named cause — not a choice about
where to put a number.**
