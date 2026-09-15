# 2026-09-15 — clause 2's llama half @8: PASS WITH MARGIN, 1.181×, every pair — and what changed to get there

Kabir, on the morning's parity reading: *"nah get stoke to llama."* This is
the sitting that answers it. Same protocol as the morning's confirmation
(`2026-09-15-clause2-e2e-8-confirm.md`; pre-registration
`2026-09-08-clause2-end-to-end-preregistration.md`; driver
`tests/tooling/run_e2e_interleaved.py`, one server lifetime per unit, arms
alternating with the order flipping every pair, per-unit quiesce gate and
banking), five pairs, launched detached at 15:04, finished 15:44.

Bank: `bench/results/2026-09-15-clause2-e2e-8-graded.json` (all 10 records
merged from the gitignored raw bank; every unit verified from its own record:
`completed 8/8`, quiesce passed, affinity `0xffff`, throttling 0, stoke
`threads 12`, `spin 0`, and **`sched_chunk 256` as announced by the scheduler
itself**, stoke units carrying stoke's pool trace and llama units llama-server's
slot timing). Binary `build/win/stoke.exe` `a7d63b6b…` = HEAD `6adf82e`;
referent b10037. **Box: nothing open, and still the slow state** (llama 86–87 s
where the fast state reads ~63 s) — the same state as this morning's 0.970×,
which is what makes the two sittings comparable.

## STATUS

| half of clause 2 (thresholds inherited verbatim) | reads | verdict |
|---|---|---|
| **≥ 1.0× llama-server-tuned**, end-to-end wall, @8 | **1.181×** (llama/stoke wall medians 86.7 / 73.4 s, n = 5 + 5); **all five pairs > 1.0: 1.164 · 1.302 · 1.174 · 1.175 · 1.129** | ✅ **PASS, with margin** |
| the same morning, before the changes | 0.970× (5/5 pairs < 1) | ⛔ was a miss |
| ≥ 2.0× ollama-default | not run | ⏳ unmeasured, predicted MISS |
| @16 | not run | ⏳ |

**Readable.** The 18% effect is twice the largest arm spread (9.8%), and each
spread comes from one unit — llama's 94.5 s (same CPU-seconds as its 86 s
units, so the box moved under it) and stoke's last 77.3 s; without those two
the remaining 4 + 4 units span 0.7% / 2.6% and read 1.18× again. The worst
pair, 1.129×, is itself bigger than the ~4% sitting-to-sitting spread that
made the morning's parity unreadable as a pass.

## 1. The sitting

| arm | n | wall median | own spread | walls in order | AGGREGATE(M0) | TTFT p50 | TTFT min | CPU-s / unit |
|---|---|---|---|---|---|---|---|---|
| llamacpp-tuned (10T auto) | 5 | **86.7 s** | 9.8% | 86.7 94.5 86.2 86.1 87.3 | 17.76 | 58.3 s | 29.0 s | 1687 |
| stoke @12T, chunk 256, new attention | 5 | **73.4 s** | 6.3% | 74.5 72.6 73.4 73.3 77.3 | 15.07 | **30.8 s** | **6.1 s** | 1266 |

| pair | order | llama | stoke | llama / stoke |
|---|---|---|---|---|
| 0 | llama first | 86.7 | 74.5 | **1.164** |
| 1 | stoke first | 94.5 | 72.6 | **1.302** |
| 2 | llama first | 86.2 | 73.4 | **1.174** |
| 3 | stoke first | 86.1 | 73.3 | **1.175** |
| 4 | llama first | 87.3 | 77.3 | **1.129** |

Same-sitting ratios, both arms over HTTP+SSE:

| quantity | this sitting | this morning (default stoke) |
|---|---|---|
| end-to-end wall, `llama / stoke` (> 1 = stoke first) | **1.181×** | 0.970× |
| AGGREGATE(M0), stoke / llama | 0.849× | 0.701× |
| TTFT p50, llama / stoke | 1.89× | 1.53× |
| TTFT min, llama / stoke | 4.74× | 4.01× |
| CPU-seconds, stoke / llama | **0.751×** | 0.778× |

Stoke now finishes the same eight-client job 15% sooner than llama-server,
answers its first client 4.7× sooner, and does it in three quarters of the
CPU time.

## 2. What changed — three things, in order of size

The morning's decomposition (`STOKE_STEP_TRACE=1`, new in `src/serve.c`;
`tests/tooling/step_trace_phases.py`) said where the 86.6 s of an @8 run
went: **59.5 s in the prefill phase** (4136 prompt rows in 128-row chunks at
12.7 ms/row, plus 140 decode rows riding along at ~40 ms each) and **27 s in
the decode tail** (N = 8 steps at 231 ms, ~30 ms per row). A second run with
20-token prompts put the N = 8 step at **146 ms** — so attention over ~600
keys was ~85 ms of every 231 ms step, a third of it, and the biggest single
target. Everything below was chosen from those numbers, not argued.

1. **The attention kernels — bit-identical, −6.9% wall on its own** (`6adf82e`,
   `src/kernels/simd.c`, `src/model.c`). The score pass computed one key per
   call as a single 128-long dependent chain (`acc = acc + q·k[d]`, one xmm
   accumulator, ~4 cycles of add latency per step with the key's 512-byte K
   row streaming from DRAM behind it); the value pass read-modify-wrote the
   four 512-byte output rows for every key. Now `att_score4x4_avx2` runs four
   keys' chains interleaved — one accumulator each, the same q vector, the same
   broadcast of `k[d]`, mul then add in the same order — and
   `att_accum4x8_avx2` keeps the four output rows in registers across eight
   keys with the same per-element add sequence. Every output is bit-identical
   to the one-key kernels: the twin gate (`tests/att_simd_twins.sh`, AVX2 vs
   scalar at G = 4), the byte-exact serve golden, the unit suite, and stoke's
   own banked 1893-token L-1900 probe (64/64 top-5 logits identical) all
   reproduce. Same-sitting ABAB in-process: old 83.0 / 83.2 s, new 77.3 / 77.7 s
   (`2026-09-15-att-multikey-ab.txt`); the N = 8 step 231 → 192 ms; the prefill
   chunk 12.0 → 11.1–11.4 ms/row (prefill rows attend too). Gated on an f32 KV
   store (a Q8 store dequantizes through one scratch pair) and compiled out
   under every attention speed mutant.
2. **12 threads instead of 10** for concurrent serving (`stoke-tuned@threads=12`;
   −4.0% in the in-process screen, all of it in the compute-bound prefill
   phase, which uses the P-cores' SMT siblings; the decode tail does not care
   about threads beyond 6). CPU-seconds +15% for that. Not yet the shipping
   default — see §3.
3. **Prefill chunk 256 instead of 128** (`STOKE_SCHED_CHUNK=256`;
   `stoke-tuned@chunk=256`; −3.4% in-process: 24 prefill steps instead of 40,
   84 riding decode rows instead of 140). ⚠ It is the one lever with a price:
   a decode that co-runs with a prefill chunk waits for the whole chunk, so
   the ITL stall gate [3] already misses doubles. Not the shipping default —
   see §3.

Over HTTP, 2 + 3 together vs the default read 1.048× (3/3 pairs) before the
kernels landed; 1 + 2 + 3 vs llama reads 1.181× here.

What was tried and did NOT help, so nobody retries it: `--kv q8` (+3.6%
slower — the dequantize costs more than the traffic it saves, so an f16 KV
was shelved unbuilt); `STOKE_POOL_SPIN` (+1.6%, as on 09-11); 6 threads
(+2.9%). The pool's dispatch fixed cost was measured at ~50 µs (10T) — one
step's 252 matvec dispatches cost 12.6 ms empty, ~5% of a step, so fusing
q/k/v dispatches was not worth doing first.

## 3. Disposition

- **Clause 2's llama half @8 on end-to-end wall: PASS, 1.181×, five of five
  pairs, margin 3× the sitting-to-sitting noise.** The ROADMAP row moves.
- **Shipping default vs graded configuration.** The kernels ship (they are the
  default path, bit-identical). Threads: 12 is a pure speed/CPU trade and
  should become the concurrent default (`STOKE_DEFAULT_THREADS_CONCURRENT`
  10 → 12; the N = 1 path keeps 8T, untouched). Chunk 256 trades ITL for
  wall and is **Kabir's call**; until it is made, the shipped default is 128
  and the graded configuration differs from it in that one knob — the
  predicted cost of shipping 128 is ~1–3% of the margin, not the margin.
- **Next in this line:** the decode tail is still ~30% of the wall at ~26 ms
  per row; the prefill GEMM runs at ~34% of the VNNI peak. Both are kernel
  work, both are bigger than any knob left.
- @16 and the ollama half remain unmeasured.
