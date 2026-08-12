# M4 — paged KV + static batch: gate report

*2026-08-12. Opus 5. Criterion: the 2026-08-04 Fable ruling §8.4's FIVE gate
items. **Item [4] is pending the running sweep; this report is complete for
[1][2][3][5] and is finished by appending §5 when `stoke-m4-g4` lands.***

> **Scope warning, recorded because it cost real time.** The resume block
> carried into this session listed TWO remaining gate items (dedup, curve).
> §8.4 lists five, and items [2] and [4] were also outstanding — [2] had no
> harness at all. **The ruling text is the authority; a memory summary of it
> is a pointer, never a substitute.** The ROADMAP M4 row now spells the five
> out so the paraphrase cannot recur.

## 1. Referents and artifacts, named as §7.5 requires

| | |
|---|---|
| Model (all items) | `models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` |
| Corpus | `tests/data/parity/p00..p49` (the frozen 50) |
| Golden forward referent | `build/stoke-m3-gate-876ce53f.exe` (the m3 archive) |
| Gate binary, items [2]/[3] | `build/stoke-m4-gate-bde28da.exe`, sha256 `C096E3D6…8827A97` |
| Item [2]/[3] raw | `2026-08-12-m4-g2-g3-raw.txt` |
| Item [5] raw | `2026-08-12-m4-g5-curve.json` |
| Item [4] raw | `2026-08-12-m4-g4-raw.txt` + `-tau.json` / `-sweep.json` / `-verdict.json` |

**Standing recipe applied to every forward-path commit this milestone:** byte
compare vs the m3 archive on p00 `-n 48` and p41 `-n 64` (the golden gates
cover tokenization only), `test_simd` green, and the relevant mutation
re-anchored BEFORE the change. Run at `fa2ddd3` (batch driver), `54078ad`
(Q8 KV accessor pair) and `e2a20d9` (`stoke run --kv`) — identical each time.

## 2. Item [1] — sim invariants ×1e6, ASan

Green throughout. The invariant SET is `2026-08-11-m4-invariants.md`; cited as
the set, never a count. Storm coverage this milestone's final run:
`set_token=10453 dedup_hit=16633 adopt=1187` — the dedup/adopt half of the
structure is exercised, not merely available. `bp_check` after every op;
teardown-at-any-point, full-reservation admission and runtime-variable
capacity all in the trace mix per the 08-02 constraints.

WSL ASan+UBSan and TSan legs green (`scripts/wsl-check.sh` exit 0), 12 test
binaries + 4 golden gates on each of plain/asan/tsan, plus the Windows matrix.

## 3. Item [2] — batch-composition independence, real model — **PASS**

Bar: same seq ⇒ token-identical at batch sizes {1,2,4,8,16}, real model,
EXACT (bit-level ids). Hard fail.

**31 sequence streams over 5 widths, every one byte-identical to its solo
`stoke run` reference.** 5 prompts (p00/p10/p20/p30/p45), `-n 16`.

- The comparison is a **byte compare of a `stoke batch` block against a whole
  `stoke run` stdout** — the CLI prints each sequence as `== seq i ==`
  followed by exactly what `run` prints, so nothing reformats between the two
  sides and no difference can be absorbed.
- **N=1 is not redundant:** `run` drives `model_forward_position` over
  `m->scratch`; `batch --seqs 1` drives `model_forward_batch` over
  `m->scratch_n[0]` — same body, different scratch and different orchestration.
- At N=16 over 5 prompts, three or four slots run the SAME prompt; each is
  compared to the solo reference, not to its twin, so a slot-indexed bug
  cannot look self-consistent.

⚠ **A tooling failure is recorded here because it presented as an engine bug.**
The first run reported "N−1 of N slots differ at every width". Cause: the
harness split blocks by line-join and dropped the newline the following header
had consumed, so every block but the last was one byte short. Now sliced by
byte offset. **A gate whose failure mode mimics the defect it hunts is worse
than no gate**; the fix is in `m4_batch_identity.py`'s docstring.

## 4. Item [3] — prefix dedup — **PASS**

Bar: dedup'd shared prefix produces streams identical to un-dedup'd;
refcounts prove out in sim AND one real-model spot check.

**192 of 287 prompt positions served from the prefix cache; prefill fell
287 → 95 tokens; stdout byte-identical to the un-dedup'd run.** Four
sequences sharing p20 as a preamble and diverging after it; adopted
`[0, 64, 64, 64]` — zero for the first (nothing resident yet), four whole
blocks each for the rest.

- **`bp_dedup_find` was called by NOTHING in `src/` before this milestone.**
  The index was correct, tested, and unreachable: every sequence allocated its
  blocks fresh. `kv_adopt_prefix` is the call that closes it.
- **Three witnesses, because "dedup was enabled" ≠ "dedup fired":** the
  reported adoption count, the prefill drop matching it exactly, and the
  unchanged streams.
- **Prompts are BUILT, not picked.** The parity corpus shares no prefixes, so
  a dedup run over it would adopt nothing and pass while proving nothing —
  the exact shape that made the pool's first op storm report `dedup_hit=0`.
- Sim half: the ×1e6 storm's `adopt=1187` with `bp_check` after every op.
- Pool half: `test_kvshare` checks the adopted span is **the same memory**
  (kv_view POINTER identity, not equal values — a copy that agreed would pass
  a value check), that private storage resumes past the match, and the
  **last-token rule** (a fully-cached prompt gives its whole last block back,
  or there is no position left to forward and so no logits).

⚠ **Honest limit, measured not assumed.** The model-level fixture test cannot
carry this claim: mutating `kv_adopt_prefix` to report positions resident
WITHOUT adopting them left `test_model` at 26/26, because the fixture's
vocabulary is 5 ids wide and its attention is dominated by recent positions.
`test_kvshare` DOES catch that mutation. Rather than leave the gap,
`model_generate_batch` now refuses the inconsistency outright via `kv_len`,
and re-running the mutation fires that fatal by name.

## 5. Item [4] — Q8-KV ON/OFF divergence count — *pending*

Criterion §8.1: KV-ON vs KV-OFF within-stoke, graded by §6 with its own τ
frozen from an M4 entry probe **before it grades anything**, zero BUG-class,
first-divergence ≤ 19/50 AND flip rate ≤ 0.5%/token on the frozen 50-corpus;
plus the property bound.

**Property bound — done.** ggml computes `id` from the f32 `d` BEFORE
narrowing `d` to f16 (its own comment says so, and why), so the *guarantee* is
`d/2 + 127·d·2⁻¹¹ ≈ 0.562·d`, not `d/2`. Measured worst case **0.4836·d (K)
and 0.4894·d (V)** — under `d/2`. The test asserts the guarantee and prints
both, so neither the guarantee nor the observation is lost. *An earlier draft
of the design note claimed the strict bound would not hold; that was reasoning
from the worst case as though it were typical, and the measurement corrected
it.*

**Representation — computed, not recalled.** 136 B per head vector against
512 = **0.2656, a 3.77× reduction**; pool at N=16/ctx1024 goes 4.83 GB →
1.28 GB. ⚠ **"Q8-KV halves it" is an f16-era statement** surviving from the
bandwidth note that `346e0d0` corrected for dtype but not for that language.

**Sweep status:** running as Task Scheduler task `stoke-m4-g4`. Append the
verdict here, then stamp.

## 6. Item [5] — the static N∈{1..16} curve — **PASS**

Bar: aggregate monotone non-decreasing within within-cycle spread through
N=8; any genuine inversion ⇒ results-vs-physics memo before the stamp.

Config is playbook LOCK 2 (Q4_K_M, per-slot ctx 1024, all N the same ctx),
3 interleaved cycles, quiesced box. Medians N=1..8:
**7.811 · 8.118 · 8.398 · 8.349 · 8.482 · 8.426 · 8.726 · 8.706 tok/s** —
monotone non-decreasing within spread ⇒ **PASS**, gain 1.116×.

**The verdict is not an artifact of an inflated tolerance.** Cycle 0 competed
with 2.6 GB of EnterpriseDB java services (since stopped) and its outliers
inflate the spread at N=4/5/15. Re-judged on the two clean cycles alone the
verdict is unchanged: still PASS, with N=3→4 (drop 0.015 vs tol 0.090) and
N=7→8 (drop 0.029 vs tol 0.032) marginal but inside spread.

⚠⚠ **THE FINDING IS NOT THE VERDICT, and it is why M4.5 now exists.**
Batching buys **1.116×**, not a multiple, and the curve is flat from N≈7 to
N=16. Mechanism, read from the source rather than inferred:
`rows_delegate_batch` (`src/kernels/vec.c:78`) loops SEQUENCES OUTSIDE the row
worker, so for every quantized type — every weight in the real model — each
Q4_K superblock is nibble-widened and scale-unpacked ONCE PER SEQUENCE. Only
F32 is truly fused. The 64-row chunking keeps bytes in L2; the ARITHMETIC
still scales linearly with N. `vec.c:174` states this honestly; the memory
summary "fused for all four weight types" did not.

**Which makes the engine compute-bound, measured:** 8.04 GFLOP/token (36
layers × 100.9 M MAC + 389 M LM head) × 8 tok/s = **64 GFLOP/s at N=1 and
~68 at N=16 — flat** — while batching drops weight traffic 18.6 → 1.2 GB/s
against a measured 44.8 GB/s ceiling. Removing the bandwidth term entirely
does not move the rate. This **confirms by measurement** the M4-entry
conclusion that M5 is per-core compute + scheduler, not weight streaming.

**Consequence, recorded now rather than discovered at M5:** M5's headline bar
is ≥17.7 tok/s aggregate at N=8; this plateaus at 8.7. Batching will not close
that — kernel speed must. Disposition: the 2026-08-12 Fable ruling, milestone
**M4.5**, ladder B → C → A′ → D → L4 with every branch pre-ruled in its §7.

## 7. §7.5 stamp preconditions

- [x] Gate report (this file) — **completed on item [4]'s verdict**
- [ ] ROADMAP row: M4 CLOSED + M4.5 ← NOW + close block naming carried headroom
- [ ] `stoke/MEMORY.md` build log
- [ ] Ledger block appended (`.superpowers` is gitignored — verdicts live in
      commit messages)
- [ ] Owner line moved M4 → M4.5
- [ ] **G8: archive the gate binary** (`build/stoke-m4-gate-bde28da.exe` is the
      [2]/[3] binary; archive the item-[4] binary too if it differs)
- [ ] `git tag m4` — annotated, **local, NOT PUSHED**
- [ ] Delete the spent `stoke-m4-g4` scheduled task (the "spent helper
      outliving its job" defect, recorded four times)
- [ ] **Restore ISLC** (stopped for the quiesced runs)

## 8. Carried headroom (new this milestone)

1. **τ for item [4] is frozen from agreeing positions of the SAME pair it
   grades** — a defect uniformly inflating logit differences would inflate τ
   and the margins together. Same shape as M3's carried entry; restated, not
   newly discovered.
2. **The curve's pool is reserved but barely touched.** At ctx 1024 each
   sequence writes ~40 of 1024 positions, so the run does not exercise the
   memory pressure the block-pool note warns about at full occupancy. The
   curve measures SCALING, not residency behaviour at depth.
3. **`kv_view` is now a partial function** — fatal on a Q8 store. The engine
   uses `kv_write`/`kv_read`; `kv_view` remains the f32-mode accessor for
   tests. M2's actual rule (one place owns KV address arithmetic) holds via
   `kv_vec_offset`; the signature moved, the ownership did not.
