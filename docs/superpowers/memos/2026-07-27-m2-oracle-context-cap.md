# Memo to Kabir — capping the oracle's context (`-c`) for the M2 gate

**Date:** 2026-07-27 · **From:** Opus 5 session manager · **Status:** DECISION REQUESTED
**Blocks:** the M2 divergence-class survey, and therefore the recorded gate run.

## What happened

The M2 gate harness (W4) landed and was verified (`6f54801`). The first live
run of the survey was stopped 25 seconds in, when process measurement showed
`llama-server` holding **11.0 GB PRIVATE** — it sizes its KV cache from the
model's advertised default context, not from what this gate needs (max 554
positions: the 490-token longest corpus prompt plus 64 generated).

That makes the survey unrunnable as configured on a 16 GB box:

| configuration | oracle | 4 stoke jobs | shared model pages | total |
|---|---|---|---|---|
| default context | 11.0 GB | 4.7 GB | ~4.3 GB | **≈20.0 GB** |
| default, `--jobs 1` | 11.0 GB | 1.2 GB | ~4.3 GB | **≈16.5 GB** |

Lowering `--jobs` does not rescue it. The oracle is the problem, not the
parallelism.

## The evidence (run before asking you, so the call is not speculative)

Same settled invocation plus `-c 2048`, measured directly:

- **Private memory: 501 MB**, down from 11,038 MB — a 22× reduction, ~10.5 GB
  freed.
- **Health in 8 s**, down from 69 s (it is no longer allocating an 11 GB KV).
- **Greedy output unchanged, byte-for-byte.** The three committed validation
  captures (`oracle_v1_english`, `oracle_v2_code`, `oracle_v3_multilingual`)
  were re-issued under the pinned request body and returned **64/64 identical
  ids each**. This is the proof condition I set for myself before proposing
  the change — not an assumption that a context bound "shouldn't" matter.

Revised budget at `-c 2048`, `--jobs 4`: 0.5 + 4.7 + 4.3 ≈ **9.5 GB**.
Comfortable, with headroom to raise `--jobs` to 6 (≈11.9 GB) if wall clock
matters more than margin.

## Why this is your call and not mine

`-c` is a change to the oracle invocation that A5/outcome-E3 SETTLED, and the
2026-07-20 near-tie memo established the precedent that invocation changes are
gate-affecting and come to you. It does not loosen any gate parameter — ε, the
waiver budget, the bars, and the corpus are all untouched, and the change is
now demonstrated output-identical rather than argued to be harmless. But the
precedent is the precedent, so I stopped.

## Options

- **(a) Adopt `-c 2048`** on the strength of the byte-identical proof above.
  Survey runs at ~9.5 GB, no swapping, and the oracle starts 8× faster —
  which also shortens every future gate iteration. **My recommendation.**
- **(b) Freeze the invocation**, run at `--jobs 1`, accept ~16.5 GB (swapping)
  and a serial-ish wall clock in the 9–11 h range rather than 3–4 h.
- **(c) Free RAM another way** (close everything else on the box). Fragile,
  and it does not fix the 69 s startup paid on every run.

## If you pick (a), what I do next, in order

1. Add `-c 2048` to `oracle_argv()` in `parity_gate.py`, and record the
   byte-identical proof plus this memo in the gate report's identity block, so
   the recorded run's provenance shows exactly why the invocation differs from
   the A5 capture.
2. Re-run `--selftest` + `--dry-run`, commit.
3. Launch the detached survey (`Start-ScheduledTask -TaskName 'stoke-m2-survey'`).
4. Census → append the divergence-class count to
   `2026-07-20-m2-oracle-near-tie-policy.md` → bring you the near-tie policy
   call → recorded `--clean` gate run → M2 checklist → tag `m2`.

## One thing to know before the census

The 2-prompt live smoke already hit a non-waivable divergence: **p48 position
2**, oracle picks 1172 over 1879, stoke picks 1879 over 1172 — same pair,
swapped, but gaps 0.158 / 0.054 nats against ε=1e-3. That is a second instance
of the near-tie memo's class, at a much larger gap than the v1 case (0.026).
p48 is the 2-token prompt, the flattest and most adversarial distribution in
the corpus, so one prompt is not a verdict — but it does raise the odds that
the census comes back non-empty and that the near-tie policy call is a real
decision rather than a formality.
