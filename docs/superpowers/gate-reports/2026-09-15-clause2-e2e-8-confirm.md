# 2026-09-15 — clause 2's llama half @8, the confirmation sitting: MISS, 0.970×, every pair in llama's favour — the two 10T sittings disagree in sign

Same protocol as `2026-09-11-clause2-e2e-8-10t.md` (pre-registration
`2026-09-08-clause2-end-to-end-preregistration.md`, driver
`tests/tooling/run_e2e_interleaved.py`: one server lifetime per unit, arms
alternating with the order flipping every pair, 45 s cool-down, per-unit
quiesce gate, per-unit banking), now with the **shipping default** on stoke's
side (`stoke-tuned` with no suffix ⇒ `stoke_open` resolves 10 threads for
`max_seqs > 1`, `d35f8a2`; verified `trace.threads == 10` on every unit) and the
stopping rule the 09-11 report asked for: **five complete pairs.** Kabir:
*"continue with the clause 2 confirmation sitting, you can start llama-server."*

Bank: `bench/results/2026-09-15-clause2-e2e-8-confirm.json` (all 10 records
merged from the gitignored `bench/results/2026-09-15-confirm-10t/grade-8.jsonl`,
each unit verified from its own record: `completed 8/8`, no errors, quiesce
passed, affinity `0xffff`, power-throttling state 0, stoke `threads 10`
`spin 0`, stoke units show stoke's own pool trace and llama units llama-server's
slot timing). Binary `build/win/stoke.exe` `57475862…` (built 2026-09-11 21:32;
no `src/` change since); referent b10037. Launched detached as a script file
at 11:58, finished 12:40. **Box: no Chrome, no ISLC, no Discord, 8.35 GB free —
and still the slow state** (see §2).

## STATUS

| half of clause 2 (thresholds inherited verbatim) | reads | verdict |
|---|---|---|
| **≥ 1.0× llama-server-tuned**, end-to-end wall, @8, stoke at the shipping default (10T) | **0.970×** (llama/stoke wall medians, n = 5 + 5); **all five pairs < 1.0** (0.980, 0.978, 0.983, 0.973, 0.964; median 0.978) | ⛔ **MISS this sitting, by 3.0%** |
| same configuration, 2026-09-11 evening sitting | 1.013× (3 pairs, all > 1.0) | ✅ PASS, thin (that report stands as written) |
| **the two sittings together** | **the sign flips with no code change** | ⚠ **PARITY within the sitting-to-sitting variation (±2–3%); the ≥ 1.0× criterion is NOT met with any margin** |
| ≥ 2.0× ollama-default | not run | ⏳ unmeasured, predicted MISS |
| @16 | not run | ⏳ |

**How this is read.** The arm spreads (4.2% / 4.4%) exceed the 3.0% effect, so
by the medians-only rule the ratio would be "not readable" — but that spread is
**common-mode**: both arms moved together across the sitting (llama 85.2 → 88.8
→ 86.6 → 87.3 → 86.1, stoke 86.9 → 90.8 → 88.1 → 89.7 → 89.3) while every pair
ratio stayed inside 0.964–0.983. Interleaving exists to cancel exactly that, and
the pairwise reading is unambiguous: five of five in llama's favour, with the
order effect (~0.8%, second arm reads better) present and not changing a sign
either way. Reported as a MISS by the same pairwise criterion the 09-11 PASS
was reported by.

## 1. The sitting

| arm | n | wall median | own spread | walls in order | AGGREGATE(M0) | TTFT p50 | TTFT min | CPU-s / unit (user+kernel) | E-share |
|---|---|---|---|---|---|---|---|---|---|
| llamacpp-tuned (10T auto) | 5 | **86.6 s** | 4.2% | 85.2 88.8 86.6 87.3 86.1 | 17.68 | 57.6 s | 29.0 s | 1718 | — |
| stoke-tuned (default 10T) | 5 | **89.3 s** | 4.4% | 86.9 90.8 88.1 89.7 89.3 | 12.38 | **37.8 s** | **7.2 s** | 1337 | 0.123–0.128 |

| pair | order | llama | stoke | llama / stoke |
|---|---|---|---|---|
| 0 | llama first | 85.2 | 86.9 | **0.980** |
| 1 | stoke first | 88.8 | 90.8 | **0.978** |
| 2 | llama first | 86.6 | 88.1 | **0.983** |
| 3 | stoke first | 87.3 | 89.7 | **0.973** |
| 4 | llama first | 86.1 | 89.3 | **0.964** |

Ratios, same sitting, both arms over HTTP+SSE:

| quantity | reads | 09-11 evening read |
|---|---|---|
| end-to-end wall, `llama_wall / stoke_wall` (> 1 = stoke first) | **0.970×** | 1.013× |
| AGGREGATE(M0), stoke / llama | 0.701× | 0.727× |
| TTFT p50, llama / stoke | 1.53× | 1.61× |
| TTFT min, llama / stoke | 4.01× | 4.29× |
| CPU-seconds, stoke / llama | **0.778×** | 0.764× |

Everything that is not the wall ratio reproduces the 09-11 evening sitting to a
few percent: stoke still does the same work in 78% of llama's CPU-seconds and
answers its first client 4× sooner. The wall ratio moved 4.3% between the two
sittings, and that is the whole finding.

## 2. The box, and a withdrawn attribution

- **This is a slow-state sitting with nothing open.** llama 85–89 s and stoke
  87–91 s, where the 09-11 afternoon (the only fast-state 10T data, n = 2)
  read llama 63.2 s and stoke@10T 61.4 s, and the 09-11 evening (Chrome open)
  read 85.9 / 84.8. Both slow sittings on record had Chrome open and this one
  did not ⇒ **the 09-11 report's "both slow sittings had Chrome open" is no
  longer an attribution; the slow state has some other cause and it is not
  named here.** What is known: CPU-seconds per unit are ~35% above the fast
  state on both arms (llama 1718 vs ~1240, stoke 1337 vs ~960), power-throttling
  state 0 on every process, the quiesce gate passed every unit at 3–9% (two
  units waited one round on 17%), `Win32_Processor.CurrentClockSpeed` reads the
  base clock and says nothing.
- **In the fast state stoke@10T led (61.4 vs 63.2, n = 2, not a graded pair);
  in the slow state it reads 1.013× once and 0.970× once.** Whether the box
  state moves the two engines differently is a question, not a finding — but
  it is the first candidate for why the sign flips, and the next sitting should
  bank something that names the state (a same-sitting `llama-bench pp512`
  referent, as the bars runner does) before another ratio is taken.
- Kernel CPU time is large in the first two units (stoke 244 s and 191 s
  against ~31 s in the rest; llama 133 s against ~38 s) with page faults
  identical to five digits — the cost of the model's pages coming in from disk
  on a fresh box, paid once. It did not slow those units' walls.

## 3. Disposition

- **Clause 2's llama half @8 on end-to-end wall, at the shipping default: two
  sittings, one thin PASS (1.013×) and one MISS (0.970×), same binary, same
  protocol.** The honest number is **parity within ±2–3%, sign not
  reproducible** — and the criterion is ≥ 1.0×, so it is **not met** as a
  gate can be met. The 09-11 pass is not withdrawn (it was readable by its own
  rule); it is joined by a reading of the opposite sign under the same rule,
  and the pair of them is the finding.
- **`ROADMAP.md`'s clause-2 row therefore records both sittings and reads
  "parity, criterion not met"** — the criterion change was pre-registered and
  Kabir-approved, the grading is now done under it, and the row states the
  result rather than the hope.
- **What would move it, in order of cost:** (1) name the box state and grade
  only in one of them — a same-sitting `llama-bench pp512` referent banked
  beside every unit; (2) the one priced lever not yet measured on wall,
  `STOKE_SCHED_CHUNK=256` (+3.2% on AGGREGATE in-process on 09-08, not
  monotone, never measured over HTTP on the wall) — a `stoke-tuned@chunk=256`
  variant in the interleaved driver is a one-line harness change and a
  40-minute sitting; (3) 12 threads (+1.5% at +15% CPU, n = 1). Each is worth
  about the size of the gap; none is worth more than the gap.
- @16 and the ollama half remain unmeasured.
