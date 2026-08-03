# RULING — M3 optimization level, reference stream, and what K3 is for

**Status:** binding ruling. Fable session, 2026-08-03, Kabir-launched per the
07-28 escalation routing. Brief ruled on:
`2026-08-03-m3-optimization-level-FOR-FABLE.md`. The summons text reached me
with D2–D4 garbled in transmission; I reconstructed them from the brief's §4–§5
and rule on the full set below — if the reconstruction mismatches what the Opus
session meant to ask, the mismatch is itself a finding and comes back to me.

**This ruling adds evidence the brief did not have.** The brief's open question
— whether the matvec shortfall is scalar code (SIMD collects it) or something
SIMD does not fix — was settled this session by measurement, and the answer
**overturns the brief's central quantitative claim**: the ~2.3× is NOT
collectable by SIMD single-thread. §2 has the numbers; every command is
reproducible as written. Source changes remain the Opus session's; this memo
and the ROADMAP §M3 amendment are the only files I touched.

---

## 0. What this session measured (box under normal session load, not quiesced)

All llama.cpp runs use the vendored release build (`bench/vendor/llama-b10037/`,
`ggml-cpu-alderlake.dll` = the AVX2+AVX-VNNI arch this box dispatches to), on
the same `models/Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf` (4,280,405,216 B) every
stoke number uses. Box: i5-13450HX, 6P+4E (16 LP; logical 0–11 = P), free RAM
at measurement time **5.01 GB of 15.63**, M: = NVMe (Intel SSDPEKNW512G8).

| # | run | command (abbrev.) | result |
|---|---|---|---|
| 1 | bwprobe re-run, `-O3` | `gcc -O3 … bwprobe.c && ./bwprobe` | serial_f32 **6.73**, split8 13.91, sum_i64 10.45, xor_u64 **14.42** GB/s |
| 2 | llama 1T, unpinned | `llama-bench -m Q8_0 -t 1 -p 0 -n 32 -r 3` | **0.99 ± 0.14** tok/s |
| 3 | llama 1T, pinned | … `-C 0x1 --cpu-strict 1 --prio 2` | **1.26 ± 0.01** tok/s |
| 4 | llama 1T, pinned, resident | … `-mmp 0` (model malloc'd, no mmap) | **1.24 ± 0.01** tok/s |
| 5 | llama 6T on 6 P-cores, resident | … `-t 6 -C 0x555 --cpu-strict 1 --prio 2 -mmp 0` | **6.86 ± 0.78** tok/s |
| 6 | stoke `-O3` e2e 1T, pinned (affinity 0x1, High) | wall(`run -n 36`) − wall(`run -n 4`), same prompt | 42.7 s / 32 tok = **0.749 tok/s** |

Derived (÷ or × 4.2804 GB weights/token):

- **Real single-thread read ceiling:** 14.4 GB/s this run; 15.5–16.6 on the
  brief's quiet-box runs. Treat as ~15±1. Run 1 reproduces the brief's §3
  structure exactly: serial_f32 sits at 2.14× under the fast variants —
  the broken-denominator finding is CONFIRMED.
- **llama.cpp 1T effective streaming: 5.3–5.4 GB/s ≈ 34% of ceiling.**
- **llama.cpp 6T aggregate: ~29.4 GB/s**, scaling 5.5×/6 cores — near-linear,
  so one core is compute/latency-limited, not bandwidth-limited.
- stoke `-O3` matvec-only (brief, interleaved medians): 6.92 GB/s ≈ 43–48%.
- Bar arithmetic: ≥7.8 tok/s on Q4_K_M (2,497,280,736 B) ⇒ **~19.5 GB/s
  aggregate**; Q8_0→Q4_K_M size ratio 1.714×.

Runs 3 vs 4 (mmap vs resident identical within noise) say run 3–5 are
storage-clean. Run 6 is NOT certified storage-clean — see §2.4.

---

## 1. Rulings

### D1 — Default optimization level: **-O3. Granted, with two riders.**

The 3.1× is measured, interleaved, byte-identical, matrix-green, and the
opponent ships release binaries; `-O3`'s extra ~10% over `-O2` was measured at
no observed cost, and the bar is relative to that release-built opponent.
Nothing here is gate-loosening: the bar does not move, the code got faster.

**Rider 1 — sanitizer legs stay at `-O1`, and the reason is recorded, not
implied.** Sanitizers verify source-level properties; raising their `-O` makes
them *less* sensitive, because the optimizer deletes or hoists exactly the
dead/racy accesses instrumentation exists to catch, and diagnostics get harder
to read. "What ships is what is sanitized" is a false requirement — an
instrumented binary is never the shipped code. What verifies the shipped
artifact is the **un-instrumented plain legs (Windows + WSL plain) at `-O3`**
running the same 172-test matrix + 4 goldens. Implementation: the Makefile's
asan/tsan branches must explicitly override the optimization level to `-O1`
(they currently only append), with a one-line comment carrying this rationale.
The chosen default also gets a one-line defense in the Makefile pointing at
this memo — the brief's real complaint was that `-O1` was never defended
anywhere, and the cure is a note in the file, not lore.

**Rider 2 — `-ffp-contract=off` joins the same CFLAGS line, in the same
commit.** Today it is provably free: the build targets baseline x86-64, which
has no FMA instruction, so no contraction is possible at any `-O`. But K3 adds
per-file `-mfma` for SIMD twins, and **GCC 14 changed the ISO-mode default of
`-ffp-contract` from `off` to `on`** — so Windows gcc 15.2 would begin fusing
`a*b+c` inside `-mfma` files while WSL gcc 13.3 (default still `off`) would
not: cross-leg f32 divergence and a silent break of scalar↔SIMD f32-path
exactness (forward-note gate [1]), discovered at the worst possible time.
Pinning `off` now kills the trap before it exists. Verify each toolchain's
actual default at landing (`gcc -Q --help=common | grep fp-contract` on both
legs) — my GCC-14 claim is from knowledge and costs one command to confirm.

`-g` stays (no codegen effect). The gitignored `build/win-O2`/`build/win-O3`
probe trees may be deleted once the default moves.

### D2 — Reference stream (reconstructed): **replaced before any new `%ref` row is recorded; `ref_gbs` = max(split8_f32, xor_u64).**

`run_ref_stream`'s serial reduction is retired as a denominator. The bench runs
BOTH fast variants — they trade places run-to-run (brief's two runs + run 1
here), and a denominator meaning "at least this much bandwidth exists" is
correctly estimated by the max of the two medians. Each variant is recorded as
its own field (`ref_gbs_split8`, `ref_gbs_xor`, runs arrays), `ref_gbs` = the
max. Keeping `serial_f32` as a clearly-labeled non-denominator diagnostic field
is at Opus's discretion — it is a decent FP-latency canary and the conversion
key that makes old rows' `%ref` reinterpretable — but it must never again be
the denominator. The `%ref` column's documented meaning becomes "% of measured
single-thread read ceiling (lower bound)", and bench.c's Decision comment
carries the §3.1 lesson: **an invariant cited as a control must have its
invariance explained before it is trusted.** This lands in the same commit
series as D1; no `%ref` row may be recorded between the old and new
denominators.

### D3 — K3's justification (reconstructed): **stands, but its magnitude and target are re-based. The open question is settled — see §2.**

Short form: the ~2.3×-to-ceiling is real bandwidth arithmetic but is **not a
single-thread SIMD collectable** — the production VNNI implementation of this
exact block format, pinned and resident, collects ~34% of ceiling, *below*
stoke's scalar `-O3` matvec rate. K3's single-thread expectation is re-based to
**llama-1T-class parity (~1.0–1.3× over scalar `-O3`)**, and the ≥7.8 bar is
re-attributed as **K5-dominant** (aggregate bandwidth, confirmed available:
llama 6T does ~29 GB/s ≥ the ~19.5 the bar needs) with K3 as the per-core
multiplier. The bar itself is untouched.

### D4 — `bench/history.jsonl` (reconstructed): **schema 2; flags embedded in the binary; old rows stay, unedited.**

- `schema` bumps 1→2; the `build` object gains the optimization-relevant flags
  **embedded at compile time and read back from the binary** (a generated
  define carrying CFLAGS is fine) — never taken from the appender's
  environment. That honors the standing provenance principle exactly as the
  brief argued: a binary's own compile flags are a property of the binary.
- **The two `-O1` rows stay in the file, byte-untouched.** Recorded evidence is
  append-only; nobody edits history to reinterpret it. No "superseded" markers
  are written into the file: `schema:1` IS the `-O1`-era marker, mechanically
  true because schema 2 lands with the level change; the interpretation lives
  in the ledger and this memo.
- The **pre-SIMD baseline row is re-taken at `-O3`** (new row, label saying so)
  **before K3's first C lands** — that row exists to be K3's diff base.
- The K2 "+9.1% integer-core cost" figure is an O1-vs-O1 truth. It may not be
  quoted for an `-O3` binary without re-measurement (the re-baseline row
  supplies it for free).

### D5 — Sequencing vs the deferred 50-prompt oracle gate: **the -O change lands first, under a mechanical precondition. Kabir's launch authority is untouched.**

Grounds: (a) **semantics** — at baseline x86-64 (no FMA), `-std=c11`, no
fast-math, an `-O` change alters instruction scheduling only; identical FP
operations ⇒ identical bytes *by construction*, not merely by observation;
(b) **evidence** — full matrix + exact-f32 goldens + p48/p49 byte-identical
sha256 across O1/O2/O3; (c) **economics** — ~4–7 h → ~1.5–2.5 h, and the parity
re-pass at K6 buys the saving twice.

**Precondition (mechanical, before any `-O3` gate launch):** rebuild the
archived intcore source at `-O3`, archive it alongside
`build/stoke-k2-intcore-2ed3d1e8.exe` with its sha256 recorded, and re-verify
byte-identity on 2–3 prompts **from the 24-prompt gate subset** (shortest
diverging ones) between the two binaries. Any byte differs ⇒ the gate runs at
`-O1` and the difference comes back to me as a finding, because it would
falsify the semantics argument, which matters more than the gate's runtime.
If Kabir launches before the `-O` work lands, running against the `-O1`
archived binary remains fully valid — same verdict, more hours. The gate is
never blocked on this ruling.

---

## 2. The open question, settled: the shortfall is NOT scalar code, and SIMD does not collect 2.3×

### 2.1 The experiment the brief was missing

The brief assumed "matvecs at 43% of real bandwidth ⇒ ~2.3× of headroom that
SIMD collects" and flagged the assumption. The decisive test costs one command
and nobody had run it: **the vendored llama.cpp release build at one thread is
an existence proof of what production AVX2+VNNI kernels — the exact ISA, exact
Q8_0 block format, exact idiom K3 would copy — actually collect on this box.**

Result (runs 3–4): **1.24–1.26 tok/s, pinned P-core, high priority, resident;
±0.01.** That is **5.3–5.4 GB/s effective ≈ 34% of the ~15 GB/s ceiling — and
below stoke's scalar `-O3` matvec median of 6.92 GB/s.**

Even granting llama.cpp generous non-matvec overhead, its matvec-only streaming
rate is bounded well under the ceiling: at 794 ms/token (ctx ≤ 32), matvecs at
the 15.5 GB/s ceiling would take 276 ms and leave an implausible 518 ms of
per-token overhead at tiny context; the arithmetic only turns plausible with
matvecs in the ~6–8 GB/s range — i.e. **the same class as stoke's scalar
`-O3`.** If the single-thread shortfall were scalar instruction throughput,
VNNI would have cleared it. It does not. The per-core bind on this box is the
latency/MLP structure of real decode (34-byte blocks, per-block scale decode,
dependent accumulation), which SIMD does not remove.

### 2.2 Where the headroom actually is — and it is confirmed, not conjectured

Run 5: **6 P-cores, resident: 6.86 ± 0.78 tok/s ≈ 29.4 GB/s aggregate, 5.5×/6
cores.** Near-linear scaling proves per-core efficiency multiplies into
aggregate until DRAM saturates — and 29.4 ≥ 19.5, the aggregate the ≥7.8 bar
implies on Q4_K_M. **The bar is feasible on this box today, on the enemy's own
binary, and it is won in K5**, with K3 as the per-core multiplier and Q4_K_M's
1.714× size advantage as the third factor. (Implied Q4_K_M-equivalent of run 5:
~11.8 tok/s before K-quant compute penalty — consistent with M0's tuned 15.6 on
a quiesced box with more threads.)

### 2.3 What K3 is FOR (restated, replacing the brief's §4.3)

1. **Redeem the integer core.** K2's arithmetic-class alignment costs ~9%
   scalar (O1-measured); VNNI is where it pays. Without K3, K2 is a tax.
2. **K-quant compute.** The bar's file is Q4_K_M; scalar K-quant dequant is
   compute-heavy per byte in a way Q8_0 is not, and SIMD's largest single-
   thread win is likely there. Unmeasured — the brief's §4.5 K4-entry
   measurement is endorsed and stays mandatory.
3. **Fewer cores to saturate DRAM.** A more efficient core reaches the
   aggregate ceiling with fewer threads, freeing cores for M5's prefill co-run
   — this directly serves the <250 ms ITL bar, which is a product gate.
4. **Batched decode.** Arithmetic intensity rises with batch; SIMD's value
   compounds exactly where stoke's concurrent-first architecture lives.

What K3 is NOT for: collecting 2.3× single-thread. Expectation at K3 exit,
single-thread, Q8_0: **llama-1T-class (~1.2–1.6 tok/s e2e pinned), i.e.
~1.0–1.3× over scalar `-O3`** — measured against the referent protocol below,
not against the stream ceiling.

**Attribution-ladder amendment:** the miss-the-bar ladder gains two referent
rows, replacing the stream ceiling as "what should this achieve":
per-core = `llama-bench -m <Q8_0> -t 1 -C 0x1 --cpu-strict 1 --prio 2 -mmp 0`;
aggregate = same with `-t 6 -C 0x555`. **Pinning is mandatory for every
single-thread number on both sides from now on** — run 2 vs run 3 is a 27%
scheduling artifact on the same binary (0.99 ± 0.14 unpinned vs 1.26 ± 0.01
pinned) on this 6P+4E part, and an unpinned single-thread number is not
comparable to anything.

### 2.4 The one number left open, and the fork is named rather than argued

stoke `-O3` e2e pinned measured **0.749 tok/s** (run 6) against a matvec-only
implied 1.644 — a residue of ~0.73 s/token, while the ops bench's non-matvec
rows sum to ~5 ms. Two candidate explanations, evidence on both sides:

- **Paging.** Free RAM was 5.01 GB < the 4.28 GB mmap; file-backed pages are
  evicted before private-committed ones, so stoke's mmap can cycle through NVMe
  every token where llama's `-mmp 0` malloc held residency. ~1.1–1.4 GB
  re-read/token at this drive's ~1.5–2 GB/s fits the residue almost exactly.
- **Real unbenched cost.** llama's mmap run (3) equalled its resident run (4)
  on the same box-state, arguing the cache sufficed; if so, the residue is
  genuine per-token work the ops bench does not measure — a coverage gap of
  half the token, same family as the swiglu lesson.

The instrument that would have settled it cheaply is blind:
`Win32_Process.ReadTransferCount` reported **0 bytes** for a run that streamed
a 4.28 GB file — **mmap page-ins are memory-manager hard faults, not ReadFile
I/O, and do not appear in the process I/O counters.** Recorded as an ops
lesson; use hard-fault deltas or disk-level counters instead.

**K3-entry chore (mandatory, before K3's first C):** on a quiesced box per the
machine checklist, re-measure stoke e2e pinned (`-n 4` vs `-n 36` differencing
is fine). If it lands ~1.3–1.6 tok/s, the residue was paging and the 16 GB
constraint gets a sharper edge (a 4.28 GB Q8_0 mmap does not fit under session
load — relevant to M5 soak conditions). If it stays ~0.75, the ops bench is
missing ~half the token and the gap gets attributed op-by-op before K3 is
scoped against it. Measure, don't argue — either branch changes something
downstream, which is why the fork is recorded instead of guessed at.

---

## 3. Landing order (Opus session; nothing here needs me again)

1. **Commit A:** Makefile — `-O3`, `-ffp-contract=off`, sanitizer branches
   pinned `-O1` with rationale comments, default defended in one line. Verify:
   full Windows matrix + WSL plain/asan/tsan + 4 goldens green; p48/p49
   byte-identity re-check vs the `-O1` binary; fp-contract defaults confirmed
   on both toolchains.
2. **Commit B:** bench.c reference streams (D2) + history.jsonl schema 2 with
   embedded flags (D4). Tests updated; matrix green.
3. **Commit C:** re-take the pre-SIMD baseline row at `-O3` (quiesced,
   machine checklist) + the §2.4 e2e re-measure and residue attribution.
4. **Gate (Kabir's launch, any time):** `-O3` intcore rebuild + archive + the
   D5 byte-identity precondition, else run the archived `-O1` binary as-is.
5. **K3 C** starts only after 1–3, scoped per §2.3 with the pinned referent
   protocol in its brief.

## 4. Not revisited

The ≥7.8 bar (unchanged — the code got faster and the enemy got re-measured,
the bar did not move), ε, the waiver budget, the corpus, the parity protocol,
push policy, the model/owner line. The referent rows in §2.3 change where a
miss is attributed, never what counts as a pass. If any reading of this ruling
looks like moving a bar after seeing results, that reading wins and the change
does not happen — same clause the brief carried, now binding on my text too.
