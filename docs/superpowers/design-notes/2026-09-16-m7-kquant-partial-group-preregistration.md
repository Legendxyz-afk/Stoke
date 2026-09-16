# M7 — the K-quant batched kernel's partial group and preamble: pre-registration

**Date:** 2026-09-16, written BEFORE any of it was measured.
**Owner:** Opus (the 08-07 governance: decide, record, Kabir overrules).
**Where it sits in the queue:** clause 2's llama half @8 is a PASS with margin
(1.181×, `80de9dd`); the 09-15 RESUME lists the next levers as the decode tail
(~27 s of an ~83 s @8 wall, ~30 ms/row, flat across threads / spin / KV dtype)
and the prefill GEMM (~34% of VNNI peak). Both run through ONE kernel,
`vec_dot_q4_k_q8_k_xn_vnni_avx2` (and its Q6_K sibling for `ffn_down`), so this
note is about that kernel and nothing else.

## 1. What was READ before anything was designed (standing rule 16: objdump first)

`objdump -d build/win/kernels/simd.o`, the VNNI kernel as shipped at `73b9246`:

- **The group loop** (four sequences × one row × one super-block) is ~98
  instructions for 32 `vpdpbusd`: 32 dpbusd with the permuted activations as
  memory operands, 4 `vpmaddwd` (mins) + 4 (scales), 4 `vpackssdw`, 6 `vphaddd`
  (two four-way horizontal sums), 3 `vmovdqu` reloads of `wperm[0..2]` from the
  stack (the same registers double as the second accumulator set), and a
  4-wide SSE float fold. Ports 0/1 carry ~56 of them per 1024 MACs, of which
  the 32 dpbusd are 57%.
- **The per-(row, super-block) preamble** carries THREE out-of-line calls:
  `kernels_q4_k_unpack_scales` (kquant.c) and `f16_to_f32` ×2 (f16.c), each
  preceded by `vzeroupper`, so the eight unpacked nibble vectors are stored to
  the stack before the calls and reloaded after them; the two `kq_perm4` run
  after. Roughly 150–200 instructions per (row, super-block) that are not the
  dot product. At N=8 a row has TWO groups, so the preamble is of the same
  order as the arithmetic; at a 16-wide tile it is ~30%.
- **The remainder path.** `n_seq % 4` sequences leave the grouped path and run
  one at a time through `kq4_dot_block` + a one-sequence reduction + a scalar
  fold. The 2026-09-15 step trace already priced it without naming it: in the
  decode-only tail the N=4→5 step costs **+44 ms for one row** where a row
  inside a group costs **~17 ms** (N=4→8: +68 ms for four), and N=7→8 costs
  **+1 ms** because the eighth row turns three remainder rows into a group.
  Every prefill chunk with riding decodes (128+3, 256+3) is a partial group,
  and so are the tail steps N ∈ {2, 3, 5, 6, 7} and every N ∈ 9..15 at @16.

## 2. The three changes, in order, each bit-exact by an ARGUMENT and a GRADER

| # | change | bit-exact because | grader |
|---|---|---|---|
| K1 | **Pad a partial group of 2 or 3 to 4** with copies of the last activation pointer; compute, discard the copies (`vec.c`, `rows_kquant_batch_avx2`). Threshold `STOKE_KQ_PAD_TAIL` = 0/1/2/3 (default 2) for the A/B; `STOKE_KQ_PAD_TRACE=1` announces the first padded group (non-vacuity). | a sequence's result depends on its own bytes and the row's only — separate accumulators, separate reduction lanes, its own scale in the fold; a copy is a sequence nobody reads | `test_matvec_batch`, MAX_SEQ 5→19 (every remainder mod 4 on both sides of the 16-tile boundary, rotated) |
| K2 | **Inline the preamble**: F16C `vcvtph2ps` for the two f16 scales, a `static inline` copy of the 12-byte scale unpack, in the Q4_K solo/xn/vnni kernels and the Q6_K kernels' f16 loads | f16→f32 is exact for every finite/inf/nan input (any correct converter gives the same bits; an sNaN differs only until the first arithmetic op quiets it in both arms); the unpack is the same integer code | `test_simd` twins, `test_matvec_batch`, `test_kquant` (unpack pair agreement), the archived-binary byte compare, serve_golden |
| K3 | **Re-sweep `KERNELS_BATCH_TILE`** ∈ {16, 32, 64, 128} — the last constant inherited from the maddubs kernel (SEQUENCE 3.4); the VNNI kernel's per-super-block live set is `n_seq × 256 B` (the permuted copy), not the 292-B original the 16 was chosen against | a tile boundary changes only the order of independent calls | same graders; the knob proved live by `simd.o`/`vec.o` hashes + `gcc -E -dM` |

K1 is in the tree as a parked patch (`build/kpad/`), applied after the
chunk-256 matrix commits. K2 and K3 follow, each its own commit.

## 3. Pre-registered predictions (so the result can say NO)

Kernel bench, single thread pinned to CPU 0, `bench --matvec-batch --tensor
ffn_gate --seqs-list 2,3,4,5,6,7,8,16 --cycles 8`, one binary, arms by env:

- **K1.a** at N=6 and N=7, `STOKE_KQ_PAD_TAIL=2` vs `=0`: the per-row cost
  moves to within 10% of N=8's per-row cost (today the remainder rows cost
  ~2.6× a grouped row). At N=5 the two arms are IDENTICAL by design (a
  remainder of one is not padded) — that cell is the null arm.
- **K1.b** `STOKE_KQ_PAD_TAIL=1` vs `=2` at N=5: padding a remainder of ONE is
  predicted SLOWER (four sequences' dpbusd for one result vs one slow
  sequence: ~54 vs ~35 port-0/1 uops per super-block). If it is faster, the
  default moves to 1; the note is wrong about the tail's price, not the design.
- **K1.c** in situ, S2 @8 12T chunk 256 (`bench --serve`, dedup on, ctx 704,
  step trace on), arms by env, ABAB in one sitting: wall −2% to −4%; the
  riding-decode cost in the prefill phase drops from ~40 ms/row toward the
  tail's ~27; tail steps at N ∈ {2, 3, 6, 7} shorten, N ∈ {1, 4, 5, 8} do not.
- **K2** kernel bench, two binaries in equal-length build dirs (the code-layout
  artifact of 2026-08-24): N=8 +10–20%, N=16 +5–10%; in situ the pure chunk
  cost (ms/row in the prefill phase) −5 to −10%, the N=8 tail step −5 to −10%.
  Non-vacuity: `objdump` shows ZERO `call` inside the row loop of the new
  kernel.
- **K3** no direction predicted; the curve is reported with a null arm (the
  same binary run twice) and the optimum adopted only if its range is
  disjoint from 16's.

## 4. Protocol and refusals

- Timing waits for the WSL matrix to finish (it is a correctness run and owns
  the box until it prints `MATRIX END`). Correctness legs may run beside it.
- One unit per process; arms interleaved ABAB; read the pair ratios, not the
  absolute rates; publish `engine / referent` from one sitting. A same-sitting
  `llama-bench pp512` (one-shot CLI, no server) names the box state beside the
  in-situ units.
- A K1 arm whose log lacks the pad-trace line is VOID (the arms are bit-identical,
  so only the trace proves which one ran). A K2 binary whose row loop still
  contains a `call` is VOID.
- Adopt a change only if: every grader green on Windows + WSL plain/ASan/TSan,
  AND the in-situ wall improves with the pairwise ratio consistent in sign
  across ≥3 pairs. A bit-exact change that is NOT faster stays out (K1's
  threshold can move instead).
- No server is started; nothing here needs one.

## 5. What this note does NOT claim

That the kernel is "port-bound", "front-end-bound" or "memory-bound". The
2026-08-20 attribution found instruction count buying 7% where it was expected
to buy 40%, and the 2026-08-30 tile swap found memory buying 1.29× where a
single-thread bench had said no — every model of this kernel has been wrong by
a factor of two in one direction or the other. The predictions above are
bands, and a result outside a band is a finding, not a failure of the run.

---

## 6. RESULTS — K1 (filled in after the runs; the predictions above stand as written)

**Kernel level, pinned, one process, three interleaved arms (`--pad-ab`), 32
cycles × 3 processes; null arm A/C = 1.00 ± 0.02 in every cell; the table is
padded-over-unpadded TIME (below 1 = padding faster):**

| remainder | Q4_K attn_q / ffn_gate | Q6_K ffn_down (maddubs) |
|---|---|---|
| 3 alone (N=3) | 0.86–0.92 | 0.91–0.95 |
| 3 beside a group (N=7, 11, 15) | 0.76–0.85 | 0.91–0.97 |
| 2 alone (N=2) | 1.04–1.07 — slower | 1.09–1.11 — slower |
| 2 beside a group (N=6, 10, 14) | 0.90–0.96 | 1.02–1.05 — slower |
| 1 (forced, N=5) | 1.20 — slower | — |

K1.a HELD (remainders 3 and, for Q4_K, 2-beside-a-group are cheaper padded);
K1.b HELD (a remainder of one is slower padded, 1.20×). Two things the
pre-registration did not predict: a Q4_K remainder of two is slower padded when
the tile IS the remainder (N=2) and faster beside a group (N=6+), and Q6_K's
maddubs kernel gains from three only. The shipped policy is therefore PER
KERNEL (Q4_K: 3 always, 2 beside a group; Q6_K: 3 only), with
`STOKE_KQ_PAD_TAIL` forcing a threshold on both for the A/B. The first
env-switched attempt (one process per arm, `run_kbench.ps1`) read up to 8%
apart on cells where the arms run IDENTICAL code — the per-process spread of a
1.4 s process is the size of the effect — which is why the in-process
`--pad-ab` mode exists. Banks: `build/kpad/k1b-*`, `k1c-*`, `k1d-*`, `k1e-*`.

**In situ (S2 @8, 12T, chunk 256, one binary, arms by env, ABAB):** see the
line appended below. **K1.c is FALSIFIED as pre-registered**: the padded steps
are 1.5% of a 259-row chunk step (3 of 259 rows) and the N∈{3,6,7} tail steps
are 9 of 127, so the whole effect on this workload is < 0.5% of wall — inside
the sitting's drift. The −2 to −4% band came from the 09-15 "riding decode rows
at ~40 ms/row" attribution, which this run shows to be an ARTIFACT of the
estimator: `(fwd − pure × prefill_rows) / riding_rows` charges the riding rows
with the second chunk's longer-context attention (the two 256-row chunks of
one prompt read 2689 and 2868 ms with ZERO riding rows). K1 stays in because it
is bit-exact, graded, and 10–30% on the matvec at every remainder it pads —
a serving load that sits at N ∈ {3, 6, 7, 10, 11, 14, 15} gets it whole — but
it is not an S2 @8 lever, and the note says so rather than re-fitting the band.

**In-situ result line (S2 @8, 12T, chunk 256, dedup on, `build/kpad/k1-*-p*.{out,err}`,
llama-bench pp512 @12T bracketing the block: 75.56 ± 4.30 before, 71.56 ± 1.59
after — the box slowed ~5% across the block):** pad0/auto wall = 0.989 · 1.021 ·
0.995, median 0.995, 1/3 pairs > 1 — a null. The pure chunk cost drifted
10.85 → 12.21 ms/row across the six units, three times the effect looked for.
