# Block `2026-09-05-overnight-bars` — Stage 0.3, the authorized llama-server sitting

**Purpose.** Stage 0.3 had blocked Stages 1–2 for weeks: one `llama-server`
sitting covering **@16 and @8 bracketed in the same block**, plus the
`ollama-default` half of clause 2. Authorized 2026-09-05, run overnight.

**Method.** M0's own harness (`bench/run_matrix.py`), 3 runs per cell, S2 =
512 prompt tokens × 128 generated. Reproduces the 2026-09-01 argv exactly and
extends it. Turbo power scheme, box swept between cells. stoke's own arms come
from `run_m5_bars.ps1` (report:
`docs/superpowers/gate-reports/overnight-f8b06a2-report.md`), taken **inside**
this block.

⚠ **Every aggregate below was reconstructed from the raw per-client fields
before being written down** — `sum(n_tokens−1) / (max(last_token_abs) −
min(first_token_abs))`. All 17 runs matched to three decimals. This is the only
check that catches a unit error, and skipping it produced two withdrawn claims
in a single day.

## The referent, bracketed

| cell | config | scenario | median tok/s | spread | TTFT p50 |
|---|---|---|---|---|---|
| p1 | llamacpp-tuned | s2-closed-8 | 19.879 | 1.73% | 52.7 s |
| p2 | llamacpp-tuned | s2-closed-16 | **16.307** | 1.29% | 78.7 s |
| p3 | llamacpp-tuned | s2-closed-8 | 20.067 | 0.28% | 51.8 s |
| p4 | ollama-default | s2-closed-8 | 9.246 | 0.19% | 57.8 s |
| p5 | ollama-default | s2-closed-16 | **8.933** ⚠ | — | 116.8 s |
| p7 | llamacpp-tuned | s2-closed-8 | 19.444 | 3.12% | 54.2 s |

⚠ **p5 is ONE valid run, not a median.** The box ran out of memory during the
cell: run 0 completed 16/16 with zero errors, run 1 completed 9/16 with seven
HTTP 500s, run 2 completed 0/16. `ollama-default` at 16 parallel slots spawns
its own `llama-server` holding 6.69 GB and exhausts this 15.6 GB box.

⭐ **@16 had never been measured on this box before.** It is the reason 0.3
existed.

## stoke, same block

| cell | AGGREGATE(M0) | blocks | decode-only | TTFT p50 |
|---|---|---|---|---|
| 8f32 | 14.518 | 14.496 / 14.539 | 41.32 tok/s | 31.9 s |
| 16f32 | 14.745 | 14.872 / 14.619 | 49.44 tok/s | 59.4 s |

Box health, same block: `llama-bench` tg128 18.25, pp512 77.345. Thermal
control: block 1 reads 1.01× the later block, so one clock throughout. The
runner's own quiesce gate passed at 5%/1% CPU.

## The ratios — this is the reproducible part

| | stoke | llamacpp-tuned | ratio |
|---|---|---|---|
| @8 | 14.518 | 19.879 | **0.730×** |
| @16 | 14.745 | 16.307 | **0.904×** |

⭐ **The gap narrows as concurrency rises.** That is the project's founding
thesis — *llama saturates after B=4 and stoke does not* — measured against
llama-**server** in one bracketed sitting, rather than inferred from lockstep
`batched-bench` (a derivation the 2026-09-01 session itself found wrong by 45%).

And the trade that buys it, in the same block:

| | stoke | referent | stoke better by |
|---|---|---|---|
| TTFT p50 @8 | 31.9 s | 52.7 s (llamacpp) | **1.65×** |
| TTFT p50 @16 | 59.4 s | 78.7 s (llamacpp) | **1.33×** |
| TTFT p50 @16 | 59.4 s | 116.8 s (ollama) | **1.97×** |

**Clause 2's other half, measured for the first time since 2026-07-17:**
stoke/ollama-default is **1.570× @8** and **1.651× @16**, against a required
**≥2.0×**. **Both MISS.**

## What this block says about the BARS

| width | llamacpp-tuned | 2× ollama-default | binding | recorded bar | direction |
|---|---|---|---|---|---|
| @8 | 19.879 | 18.492 | **19.879** | 17.7 | harder |
| @16 | 16.307 | **17.866** | **17.866** | 16.8 | harder |

Both move **harder**, so the standing asymmetric guard is satisfied. The @16
binding bar comes from the 2×-ollama family — exactly the inversion that family
was created to capture.

⚠⚠ **But re-anchoring to any single sitting is not defensible.** This same
`llamacpp-tuned` @8 cell has now read:

| sitting | median |
|---|---|
| 2026-07-17 (M0 — the source of the 17.7 bar) | 17.664 |
| 2026-09-01 | 24.608 |
| 2026-09-05 (this block) | ~19.9 |

with **no code change**, while *within* a sitting it repeats to 1–2% and the
bracket p1-vs-p3 agrees to **0.95%**. Between 09-01 and tonight both absolutes
moved double digits (stoke −14.1%, llama −19.2%) and **the ratio moved 6.3%**.

⇒ **Publish the ratio, name the block.** The recommendation to Kabir is that
clause 2 stop being an absolute tok/s and become a same-sitting ratio — which
is already this project's rule for anything it publishes, and was simply never
applied to the gate criterion itself.

## Reproducing this

```sh
# referent cells (one engine at a time; the adapter tree-kills and frees the port)
python bench/run_matrix.py --only-config llamacpp-tuned --only-scenario s2-closed-8  \
    --runs 3 --fresh --out-dir bench/results/<block>/p1-llamacpp-8a
python bench/run_matrix.py --only-config llamacpp-tuned --only-scenario s2-closed-16 \
    --runs 3 --fresh --out-dir bench/results/<block>/p2-llamacpp-16
# ... then repeat @8 AFTER the others: the bracket is what separates
#     drift inside the block from drift between sittings.

# stoke's own arms, inside the same block
powershell -ExecutionPolicy Bypass -File tests/tooling/run_m5_bars.ps1 \
    -Cells 8f32,16f32 -Arms S,L -Blocks 2 -Threads 8 -Stamp <stamp>
```

⚠ Raw per-cell JSON is deliberately not committed (`bench/results/*` is
gitignored except top-level `*.json`/`*.md`, and the 2026-09-01 referent run is
not tracked either). This file and the `.json` beside it are the artifact.
