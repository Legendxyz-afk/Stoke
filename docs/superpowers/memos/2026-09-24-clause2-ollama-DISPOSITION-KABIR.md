# Phase-1 clause 2, ollama half — "≥2× ollama-default at N=8/16": DOCUMENTED MISS

**Ruling:** Kabir, 2026-09-24 ("accept the ollama as a miss"), on Opus's
recommendation of the same day. **The ollama half of clause 2 is NOT met; the
miss is accepted as documented, and the referent is NOT re-specified.** This
memo is the authority for the disposition; the numbers below are citations of
banked runs, not new measurements. Clause 2's other half (≥1.0×
llama-server-tuned) PASSED and is unaffected.

## 1. What the clause asks and what was measured

The clause (ROADMAP "Phase-1 goal"): S2 aggregate ≥2× ollama-default AND
≥1.0× llama-server-tuned at N=8/16 on the same GGUF, graded since 2026-09-08
as a same-sitting ratio on END-TO-END WALL through one HTTP harness
(`design-notes/2026-09-08-clause2-end-to-end-preregistration.md`, thresholds
unchanged). The ollama bands were pre-registered before the run
(`design-notes/2026-09-16-clause2-n16-n32-ollama-preregistration.md`).

| width | wall ollama / stoke | pairs | pre-registered | verdict | source |
|---|---|---|---|---|---|
| @8 | **1.735×** | 2/2 (1.736 · 1.734), spreads 0.2% / 0.3% | H3 1.6× [1.3, 1.9] — inside | ⛔ miss by 13% | `gate-reports/2026-09-18-clause2-e2e-16-32-ollama.md` §2 |
| @16 | **1.836×** | 2/2 (1.833 · 1.840), spreads 0.2% / 0.2% | H4 1.7× [1.3, 2.0] — inside | ⛔ miss by 8% | same report, §3 |

Both reads landed inside bands registered before the run: the miss was
predicted, not discovered.

## 2. Why it is a ceiling of the workload, not a defect

- **ollama-default serves ONE request at a time on this box.** Without a GPU,
  ollama 0.30.11 resolves `OLLAMA_NUM_PARALLEL` to 1: wall = N × 16.2 s,
  first-token times in lockstep, 68 ms between tokens with no co-run stalls
  (report §2). "2× ollama" is therefore "2× a serial server".
- **S2 is ~80% prompt, and batching cannot make prefill cheaper.** At its
  prefill ceiling stoke needs 653 / 78.1 ≈ 8.4 s of work per request against
  ollama's serial 16.2 s ⇒ **≈1.93× is the most S2 permits** against a one-slot
  server on this box (SEQUENCE §IV.9, derived from the 09-18 bank; 1.84×
  measured @16). Clearing 2× needs ~3% more prefill speed (80.6 t/s), and the
  kernel levers priced ~null on 2026-09-16.

## 3. Why the referent is not re-specified (Opus, 2026-09-24; Kabir concurred)

1. **Changing the referent after seeing the result is moving the goalposts.**
   The bands were registered, the reads fell inside them, and a different
   ollama configuration chosen now would be chosen because of the answer.
2. **The harder referent passed.** ollama with `OLLAMA_NUM_PARALLEL` set is
   llama-server underneath and would read like the llama rows, which PASSED:
   1.181× @8 (5/5 pairs), 1.195× @16 (3/3), 1.281× @32 beyond the clause.
3. **The published record already states it as a miss**, with the serial
   referent explained: README "What it does not win", `docs/BENCHMARKS.md` §2.

## 4. What ships, what comes next

- **Ships:** the README and BENCHMARKS lines as written on 2026-09-18 — 1.74× /
  1.84× ollama-default, "the target was 2×; missed".
- **For the M7 stamp:** clause 2 = llama half PASSED, ollama half a DOCUMENTED
  MISS (this memo). Phase 1 closes with two documented misses: this one and
  clause 4 (`memos/2026-09-16-clause4-itl-stall-DISPOSITION-KABIR.md`).
- **Where more than 2× is real (SEQUENCE §IV.9):** shared prompts — prefix
  reuse measured 2.3–4.2× within stoke on S3 fan-out (M5 gate 4) — and
  decode-heavy work against a serial server. Neither is S2; neither is re-run
  to rescue this clause.
