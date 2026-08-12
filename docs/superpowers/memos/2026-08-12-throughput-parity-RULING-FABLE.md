# RULING — throughput parity with llama.cpp, and the kernel ladder (M4.5)

*2026-08-12. Fable 5, summoned deliberately by Kabir (the 2026-08-07 governance
change bars PROPOSING summons; it does not bar Kabir invoking one). This memo is
the authority for the M4.5 milestone; the ROADMAP row points here. Kabir
overrules any line of it by saying so.*

---

## 1. What Opus's M4 execution established, and what it lacks

The M4 record is strong where this project is strong: five gate items scoped
from §8.4, four closed with real-model evidence, every forward-path change
byte-compared against the archived m3 binary, the curve run on a quiesced box
with interleaved cycles and judged against its own spread. Nothing below is a
correctness criticism.

What the measurements exposed is STRATEGIC, and it is three facts:

- **F1 — the batch path amortizes traffic, not arithmetic.** `rows_delegate_batch`
  (`src/kernels/vec.c:78`) re-runs the full quant unpack+dot once per sequence;
  only F32 is truly fused. The design premise "batching amortizes the weight
  term" was implemented for BYTES (64-row chunks stay in L2) and not for
  COMPUTE. Disclosed honestly in `vec.c:174` — but the engine turned out
  compute-bound, so batching bought 1.116×, and the premise's payoff with it.
- **F2 — the inner loop wastes the machine.** `kq_mac8` = widen-to-i32 +
  `mullo` + `add`: 8 products per multiply, ~0.75 uops/product, 8-byte loads.
  llama.cpp's AVX2 Q4_K kernel (`arch/x86/quants.c:2038`): 32-byte loads,
  `maddubs` (32 products/uop), scales FUSED into the widening `madd`, ~10
  instructions per 64 elements ≈ 0.08–0.15 uops/product. That 4–6× inner-loop
  deficit is the measured 2.2× end-to-end gap (8.0 vs 17.45 tok/s, same box,
  same file). VNNI (`dpbusd`) narrows it further and the box has it.
- **F3 — the exactness contract is shaped wrong, and it is the only blocker.**
  Gate [1] as practiced pins SIMD twins to the scalar's **8-lane f32
  partition** (`aux32[8]`/`sums[8]`, `kquant.c:325`). That partition is an
  ACCIDENT of transcribing ggml's scalar, not a design value. The design value
  is determinism + cross-build bit-reproducibility. As written, the contract
  forbids every wide-ISA kernel forever — while llama.cpp, which we benchmark
  against, maintains no scalar↔SIMD exactness at all (its own AVX2 fold differs
  from its own generic scalar). stoke's edge is HAVING the contract; the fix is
  to restate it so it stops encoding one register width.

Also observed, ratified as standing process: **memory summaries drifted from
ruling text twice this milestone** (gate-item scope; "fused for all four
types"). The §8.4 list and the source file are the authorities; a resume block
is a pointer, never a substitute. Opus already recorded this; it is now a rule.

## 1b. M0–M4 retrospective — were they best-for-us? (asked by Kabir, ruled here)

**The architecture needs no reopening.** Paged KV with index-linked pool,
verified dedup, one forward body, the freeze-then-body-swap discipline, the
classified-divergence criterion — all correct, all earning their keep in the
record. What follows is per-milestone, verdict then improvement:

- **M0 — right milestone, wrong surviving artifact.** The harness survives;
  the NUMBERS did not: the 9× headline ignored KV traffic (→ ~3.3× at M5's
  operating point), the 6-thread peak and the E-core penalty both failed to
  reproduce — all taken pre-quiesce-protocol. And the deeper miss: M0 measured
  a BANDWIDTH ceiling and framed the whole plan with it, but never measured a
  COMPUTE ceiling (uops/product). One extra day there would have predicted
  2026-08-12's flat curve two milestones early and re-ordered the kernel work.
  That is the largest structural miss in the plan.
- **M1 — exactly right, change nothing.** Byte-exact oracle gate, zero rework
  since. It is the template M7's tokenizer gate already copies.
- **M2 — right gate, one silent inheritance.** Token-exact + 36-layer ladder
  worked; the kv_view freeze paid off at M4 precisely as promised. The miss:
  ggml's 8-lane f32 partition was transcribed as if normative. Nobody asked
  "contract or accident?" — and that accident is today's only blocker on
  kernel speed. **New standing rule: when transcribing a reference, mark every
  numeric choice CONTRACT or ACCIDENT at transcription time, in the source.**
- **M3 — the twin discipline is a keeper; it was pointed at the wrong
  invariant.** Bit-exact twins caught real bugs and enabled the archive
  byte-compare — keep forever. But the twins were pinned to M2's accidental
  partition (same root), and the integer-core unification stopped at Q8_0
  instead of covering the K-quants — the asymmetry C now fixes. Also: K6
  recorded absolute tok/s, so a 2.2× referent gap sat normalized for two
  milestones. **Bars against a referent must carry the RATIO on their face.**
- **M4 — the process worked; it worked at the end instead of the start.** The
  curve gate existed precisely to catch the M0 disease, and it did. But the
  premise "seqs-inner amortizes the weight term" was implemented for bytes and
  not decomposed into traffic vs compute; a 30-minute paper cost model (uops +
  bytes per step) would have predicted 1.1× before a line was written. **New
  standing rule: any "X amortizes Y" premise gets that paper model first.**

**The pattern, named once:** five instances of one shape — a number or
contract inherited without asking whether it was measured, normative, and for
THIS box and purpose (M0's numbers, M2's partition, M3's pinning, M4's
premise, the memory-summary drift). The correctness machinery is this
project's real asset and repeatedly caught the consequences; the two new
standing rules above close the remaining gap at the source.

## 2. THE PARITY QUESTION — ruled

**Yes: llama.cpp-band throughput is the target, and it already was.** M5's
committed bars (≥17.7 aggregate @8, ≥14.2 @16 …) are the llamacpp-tuned parity
family. The curve measured the plateau at 8.7 and batching at 1.116×, so those
bars are reachable through KERNELS or not at all. Refusing parity now would
mean deleting M5's bars; nothing in the record justifies that.

**The bar (new, frozen here):**

> **P-BAR.** stoke N=1 decode ≥ **0.85×** llama-bench `tg` — same box, same
> GGUF, same thread count, referent re-measured interleaved at gate time per
> §8.2 (asymmetric guard applies). If step D (repack) lands and survives its
> RAM test, the bar tightens to **0.95×**. Aggregate bars at M5 are unchanged.

Why 0.85 and not 1.0: the last ~15% of llama.cpp's rate comes from its
load-time repack GEMM (`block_q4_Kx8`, 4×8 register tiles), which duplicates
weights into non-mmap memory — ≈ +2.3 GB RSS on this 15.63 GB box, i.e. inside
the measured page-cache-collapse zone. That trade is a measurement (step D),
not an assumption, and the bar must be meetable without it.

**Explicitly NOT chased:** repack-level micro-tuning past the band, AVX-512
paths (this box has none), multi-ISA breadth, GPU.

## 3. THE NUMERIC-CONTRACT RE-ANCHOR — the load-bearing ruling

Gate [1]'s bit-exactness requirement is RESTATED, not weakened:

> **CONTRACT v2.** (i) The integer core of every quant kernel is EXACT and may
> be regrouped freely by any implementation — integer adds are associative, so
> any grouping of the same mathematical sum is the same bits. (ii) The scalar
> reference defines, per row, ONE canonical f32 fold: sequentially per
> superblock `b` in file order, `acc += d_b * (f32)S_b` then
> `acc -= dmin_b * ad_b * (f32)M_b`, where `S_b`/`M_b` are the superblock's
> exact integer totals; the exact op sequence is pinned by the scalar's source.
> (iii) Every twin reduces its lanes to the SAME `S_b`/`M_b` (exact, by (i))
> and executes the SAME f32 sequence. Scalar↔twin bit-exactness then holds BY
> CONSTRUCTION at any SIMD width, on any ISA, permanently.

This is a **unification, not an invention**: `vec_dot_q8_0_q8_0` already has
exactly this shape (integer block total → single f32 chain), which is WHY its
x4 twin passed gate [1] unmodified. The K-quant kernels are the outliers being
brought into line. The sequential fold's dependency chain costs ~4 cycles ×
(n/256) superblocks per row — noise against the integer work.

**What the re-anchor costs, and how it is graded — not assumed:**

- The K-quant SCALAR itself changes (8-lane partition → canonical fold), so
  K-quant-weighted token streams may flip at near-ties. This is an
  ARITHMETIC-CLASS change under §6 and is graded by the §6 instrument:
  OLD-binary vs NEW-binary on the frozen 50-corpus, τ(old↔new pair) frozen
  from its own entry probe of agreeing positions BEFORE grading (the
  `m4_q8kv_gate.py` phase structure is this instrument; reuse it), **zero
  BUG-class**, first-divergence ≤ 19/50, flip rate ≤ 0.5%/token.
- **The Q8_0 file is the free anchor: its streams must stay BYTE-IDENTICAL
  through C** (Q8_0 and F32 kernels are untouched). Assert it; a flip there is
  a bug, full stop.
- On pass: archive the NEW gate binary (G8 step), record in the ledger that
  the m3-archive byte-compare RETIRES for K-quant files at that commit (it
  remains binding for Q8_0 files and for tokenization), re-anchor K6 and the
  budget's N per §8.2/§8.3.

## 4. THE LADDER — Opus's command sequence

**L0 — close M4 first. No kernel work inside M4.** Finish the G4 sweep, grade
it, gate report, stamp, `git tag m4` (local, no push). Delete the spent
`stoke-m4-g4` task; restore ISLC. M4's scope is the §8.4 five and is done when
they are.

**L1 = B — the batch unpack hoist (bit-exact, no ruling needed).** Hoist the
Q4_K/Q6_K nibble-widen + scale-unpack (weight-only work) so it runs once per
row-chunk instead of once per sequence, inside `rows_delegate_batch`'s
chunking. No accumulation order changes. Existing `test_matvec_batch` identity
checks verify; re-run the CURVE post-B (quiesced protocol), commit to
`bench/history.jsonl`. This makes the "fused" claim true for arithmetic and is
the direct answer to the flat curve.

**L2 = C — the contract re-anchor, §3 above.** Rewrite `vec_dot_q4_k_q8_k` /
`vec_dot_q6_k_q8_k` scalars to the canonical fold; write the AVX2 twins
`maddubs`-shaped (fuse scales into `madd`, 32-byte loads; use `dpbusd` where
VNNI applies). Grade per §3. Expected N=1 movement: 8 → ~12–15 tok/s.
ESTIMATE, NOT A PROMISE — the gate is the measurement.

**A as originally scoped is CANCELLED.** An 8-lane x4 port under the OLD
contract would be verified, landed, and then deleted by C weeks later; the
isolation it buys does not pay for a discarded kernel. **A′ replaces it, after
C:** row-tiling (x2/x4) on top of the canonical fold, where it is trivially
exact by construction. Ordering is B → C → A′ → D.

**L3 = D — repack, CONDITIONAL and RAM-gated.** Only if post-A′ measurement
sits below P-BAR: prototype llama-style 8-row interleave at load into
malloc'd buffers. Its gate is not speed but RESIDENCY: model + repacked copy +
KV pool must hold the standby-cache health counters (`Page Reads/sec` < 5
sustained) at N=8/ctx1024 on this box. Expected verdict on 15.63 GB: reject —
write the number down either way, and on reject P-BAR stays 0.85×.

**L4 — attention f32 twin, PROFILED FIRST.** `vec_dot_f32` is a scalar
sequential chain, legal to twin under CONTRACT v2 (canonical fold = the
existing scalar order). Build it ONLY if a post-C profile shows attention ≥15%
of decode at ctx 1024–2048. At M5's contexts it will matter; let the profile
say when.

**Every step:** the standing recipe — byte-compare where exact (m3 archive
until C, new archive after; Q8_0 file always), classified re-anchor where not,
mutation re-anchored before each change, matrix + WSL plain/asan/tsan, perf
only on the quiesced protocol. One step per commit, measurements in the
message.

## 5. M5 notes touched by this ruling

- ROADMAP's "if static N=8 already missed, attribution starts from M4's static
  curve" is **RESOLVED**: attribution is complete, the gap is per-core kernel
  arithmetic. The line should say so and point here.
- From vLLM's v1 scheduler (data, mined 2026-08-12): **chunk ends must be
  block-aligned or the prefix cache can never register the entry** (their
  scheduler enforces `end = aligned_end` mid-prompt). stoke's chunk 128 is a
  multiple of block 16, so M5 inherits this by construction — RECORD the
  invariant in M5's design note so a future chunk-size change cannot silently
  break dedup registration. Non-binding design input, binding once M5's note
  adopts it.
- M5's scheduler work remains as locked by the forward note. This ruling
  changes the FLOOR it schedules on, nothing about its structure.

## 6. Direction check

Kabir: the one genuinely irreversible piece is C — after it, K-quant token
streams differ from M2/M3's at near-ties (graded, bounded, zero-BUG-gated),
and the m3 archive stops being the K-quant referent. Everything else is
bit-exact or conditional. If you want the old streams preserved as the
reference forever, say so and C is off; the cost is P-BAR (and with it M5's
committed bars), which cannot be met at 8 products per multiply.
