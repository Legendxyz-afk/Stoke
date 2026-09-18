# stoke — benchmarks, and how to reproduce them

Every number here is a **ratio measured in one sitting**, with the absolute
walls, the box state, and the block that produced it beside it. That is not
caution for its own sake: on the development box the same binary, workload and
harness read 1.8× apart on different days with no code change, and 4% apart
between two sittings an hour apart. A bare tok/s from this or any engine is a
property of an afternoon. A ratio taken with the arms interleaved in the same
hour is reproducible.

## 1. The claim, in one table

Workload S2: N closed-loop clients, each sending one chat request with a
~525-token prompt (512 synthetic words plus a per-client salt so no prefix is
shared) and reading a 128-token streamed answer; the clock runs from the first
request to the last token. Same GGUF on every arm (Qwen3-4B-Instruct-2507,
Q4_K_M, 2.5 GB), CPU only, over HTTP+SSE through the same client harness.

| clients | referent (as a competent operator would configure it) | wall ratio **referent / stoke** | pairs, all > 1 | stoke first token vs referent's | CPU-seconds stoke / referent | sitting |
|---|---|---|---|---|---|---|
| 8 | llama-server b10037, `--parallel 8 -c 16384` | **1.18×** | 5/5 (1.13–1.30) | 4.7× sooner (min), 1.9× (median) | 0.75× | 2026-09-15 [1] |
| 16 | llama-server, `--parallel 16 -c 32768` | **1.20×** | 3/3 (1.19–1.23) | 4.5× / 1.5× | 0.73× | 2026-09-18 [2] |
| 32 | llama-server, `--parallel 32 -c 22528` | **1.28×** | 3/3 (1.25–1.30) | 5.1× / 1.4× | 0.71× | 2026-09-18 [2] |
| 8 | ollama 0.30.11, default settings, CPU | **1.74×** | 2/2 (1.73–1.74) | 1.3× / 2.3× | n/a | 2026-09-18 [2] |
| 16 | ollama, default settings, CPU | **1.84×** | 2/2 (1.83–1.84) | 1.3× / 2.4× | n/a | 2026-09-18 [2] |

Read: at 16 clients stoke finishes the same job in 1/1.20 = 83% of
llama-server's wall, answers its first client 4.5× sooner, and uses 73% of
the CPU time. **The margin grows with concurrency** — 1.18 → 1.20 → 1.28× —
which is the design's claim: prompts are prefilled in chunks while every
already-admitted request keeps decoding, so the more requests are in flight,
the more of the prefill phase is shared.

[1] `docs/superpowers/gate-reports/2026-09-15-clause2-e2e-8-graded.md`
[2] `docs/superpowers/gate-reports/2026-09-18-clause2-e2e-16-32-ollama.md`
(raw records: `bench/results/2026-09-15-clause2-e2e-8-graded.json`,
`bench/results/2026-09-18-clause2-e2e-16-32-ollama.json`).

### The absolutes behind the ratios (so the ratios can be checked)

| clients | referent wall (median, s) | stoke wall (median, s) | referent AGGREGATE(M0) tok/s | stoke AGGREGATE(M0) | box state: `llama-bench pp512` same block |
|---|---|---|---|---|---|
| 8 vs llama-server | 86.7 | 73.4 | 17.8 | 15.1 | slow state (llama 86 s where the fast state reads 63 s) |
| 16 vs llama-server | 168.0 | 140.6 | 14.6 | 15.2 | 69.8 → 71.5 t/s @12T |
| 32 vs llama-server | 343.0 | 267.7 | 13.0 | 15.5 | 72.4 → 71.1 |
| 8 vs ollama | 129.4 | 74.6 | 8.4 | 14.9 | 69.4 → 70.1 |
| 16 vs ollama | 257.6 | 140.3 | 8.1 | 15.2 | 71.4 → 70.7 |

AGGREGATE(M0) = tokens generated ÷ (last token − first token) across all
clients. It is quoted because it is the number most engines publish, and it is
**not** the headline here because it starts its clock at the first token and so
credits an engine for answering nobody for a minute: at 8 clients llama-server
reads 17.8 against stoke's 15.1 on that metric while finishing the job 18%
later. Wall, first-token latency and CPU-seconds together describe the work;
any one of them alone is choosing a story.

## 2. What stoke does not win, stated where the wins are

- **≥ 2× ollama-default was the Phase-1 target and it is missed: 1.74× @8,
  1.84× @16.** Two things about the referent matter for reading that. On a
  CPU-only box ollama serves **one request at a time** (`OLLAMA_NUM_PARALLEL`
  resolves to 1 without a GPU; measured: wall = N × 16.2 s, first-token times
  in lockstep, 68 ms between tokens with no co-run stalls), so "2× ollama" is
  "2× a serial server", and what stops stoke at 1.8× is its own prefill cost,
  which is the same arithmetic on both sides. Ollama with `OLLAMA_NUM_PARALLEL`
  set is llama-server underneath and would read like the llama rows.
- **A request that arrives while others are decoding stalls their streams for
  one prefill chunk: ~3.3 s median, 3.5–3.8 s worst, at 8–32 clients.** The
  design bound was 250 ms and it is missed by an order of magnitude at every
  chunk width tried; the fix is a scheduler that does not hold decode rows
  behind a chunk's whole step, deferred to Phase 1.5. For scale: llama-server's
  streams stop for **29–30 s** at 16 and 32 clients (one whole 2,048-token
  prompt batch), and ollama's serial server has no co-run stall because it
  has no co-running. Between stalls stoke's cadence is 190 ms @8, 310 ms @16,
  560 ms @32 per token per client (llama-server: 390 / 810 ms at 16 / 32).
- **Single-stream decode is ~0.9× llama.cpp's** (M4.5, 2026-08-19, pinned
  and unpinned); the engine is built for many concurrent requests, not for
  one.
- **Absolute throughput on this box is what a 6P+4E laptop CPU gives**: ~15
  tokens/s aggregate at 8–32 clients on a 4B Q4_K_M model. The claim is the
  ratio and the shape of the curve, not the number.

## 3. Reproduce it

Prerequisites: the GGUF at `models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`,
llama.cpp b10037 under `bench/vendor/llama-b10037/` (`llama-server.exe`,
`llama-bench.exe`), ollama 0.30.11 installed with the model registered
CPU-only (`bench/adapters.py` does the `ollama create` from its Modelfile),
Python 3.12+, Windows (the harness's process instrumentation is Windows-only;
the engines are not). Build stoke: `mingw32-make` from a POSIX shell.

```sh
# one block: 3 pairs, arms alternating, the order flipping every pair,
# one server lifetime per unit, a quiesce gate before each unit
python tests/tooling/run_e2e_interleaved.py --scenario s2-closed-16 \
    --arm-a llamacpp-tuned --arm-b stoke-tuned@threads=12,chunk=256 \
    --pairs 3 --cooldown 45 --timeout-s 900 --out block16.jsonl

# the other rows
#   s2-closed-8   llamacpp-tuned
#   s2-closed-32  llamacpp-tuned@ctx=22528     (the policy's -c 65536 does not fit 15.6 GB)
#   s2-closed-8   ollama-default
#   s2-closed-16  ollama-default

# name the box state around each block (one-shot, no server):
bench/vendor/llama-b10037/llama-bench.exe -m models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf -p 512 -n 0 -t 12

# read a block against the pre-registered tripwires and band
python tests/tooling/e2e_block_summary.py block16.jsonl --log block16.log --band 1.10 1.35
```

The harness prints, per unit, the wall, AGGREGATE(M0), first-token p50, the
process's CPU-seconds / page faults / peak working set, the configuration the
server **announced** (stoke's thread count and chunk width; llama-server's
slots and per-slot context), and the client-observed inter-token latencies;
then per arm the median and its own spread, the ratio, and every pair. A unit
whose aggregate cannot be reconstructed from its own raw records is refused,
not banked. `stoke-tuned@threads=12,chunk=256` names the shipped defaults;
the label exists so the bank says so.

What "tuned" means for the referents, unchanged from the M0 matrix: slots
matched to the client count and per-slot context to the prompt+answer size
(`--parallel N -c N×2048`; at N = 32 the total context is capped by memory to
704 per slot, which llama-server pads to 768 — capacity, not work: every
request uses 653). ollama-default is ollama as installed, with nothing set.

### Rules this report keeps, each because a number was once withdrawn for breaking it

1. **Ratio and absolutes together, block named.** The ratio travels between
   sittings; the absolute does not.
2. **Wall, first-token latency and CPU-seconds beside any aggregate.**
3. **A decode rate is comparable only at the same context length** — the same
   binary reads 13% apart between ctx ~30 and ~590, and that difference attaches
   to whichever arm carries the longer prompt.
4. **Every aggregate reconstructed from the raw records before it is printed**
   (`Σ(n_tokens − 1) / (max last_token − min first_token)`); it is the one
   check that catches a unit error.
5. **The configuration is what the server said, never the label.**

## 4. The development box

13th-gen Intel i5-13450HX (6 P-cores with SMT + 4 E-cores = 16 logical), 15.6
GB, Windows 11 Home, MinGW gcc for stoke, llama.cpp b10037 (Clang 20, AVX2/
AVX-VNNI CPU build), ollama 0.30.11 pinned to CPU. Chrome and ISLC closed for
every block; the quiesce gate refuses a unit when CPU busy exceeds 12% twice or
ISLC is running.
