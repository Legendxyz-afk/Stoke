# M4.5 — kernel ladder to llama-band: gate report

**DRAFT, written unattended 2026-08-19 05:1x by the Opus 5 session that ran the
measurement chain. NOT STAMPED — §7.5's close sequence and the `m45` tag are
left for Kabir's word, as M3's close was.** Everything below is measured and
committed; §7 lists exactly what is outstanding.

Authority: `docs/superpowers/memos/2026-08-12-throughput-parity-RULING-FABLE.md`.

## 1. Referents and artifacts, named as §7.5 requires

| what | where |
|---|---|
| branch | `m45-aprime` (local, never pushed) |
| binary under test | `build/win/stoke.exe`, rebuilt 2026-08-17 21:20:58 |
| referent, correctness | `build/stoke-m45-maddubs-f3b7181a.exe` (Q4_K_M), `build/stoke-m3-gate-876ce53f.exe` (Q8_0 anchor) |
| referent, speed | `bench/vendor/llama-b10037/llama-bench.exe`, re-measured interleaved at gate time per §8.2 |
| model | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (2.33 GB), Q8_0 for the anchor legs |
| byte-compare gate | `2026-08-18-m45-att-par-verify.txt` — commit `a48de9f` |
| speed A/B | `2026-08-18-m45-att-par-ab-report.md` — commit `1ac09f8` |
| P-BAR, t=6 pinned | `2026-08-18-m45-pbar-l4-t6-pinned-report.md` — commit `0e27a75` |
| P-BAR, t=8 unpinned | `2026-08-19-m45-pbar-l4-t8-unpinned-report.md` — commit `0b405aa` |
| night's operating log | `2026-08-18-m45-overnight-chain.txt` (carries three corrections against myself) |

## 2. P-BAR — **MET, at both graded configurations**

The bar (§2, frozen by the ruling): *stoke N=1 decode ≥ 0.85× llama-bench `tg`,
same box, same GGUF, same thread count, referent re-measured interleaved at gate
time.* The 0.95× variant applies only if step D lands, and D never ran (§3).

| configuration | blocks | B median | llama median | **B/L** |
|---|---|---|---|---|
| t=6, pinned 0xFFF | 0.907 / 0.920 / 0.908 | 16.529 | 18.13 | **0.912×** |
| t=8, unpinned | 0.893 / 0.887 / 0.910 | 16.577 | 18.58 | **0.892×** |

Before L4 this bar was graded at **0.827–0.838** and missed. Both runs are three
blocks spaced across the night from different thermal states; each block's
arm order rotates; llama is measured inside the same sitting.

**Graded on B/L, not B/A** — see §6: arm A is unfit as an anchor, both reports
say so on their face, and no part of this verdict depends on it.

## 3. The ladder, as executed

| rung | disposition |
|---|---|
| B — batch hoist | **landed.** `1788b24` hoists the Q4_K unpack out of the sequence loop (N=16 +29.5%), `cb9d00a` retracts that figure as unreliable, `f91e44e` re-measures it properly at **+32.5%** with Q6_K joining the hoist. Per §7.1's step model B cannot move N=1, and it did not |
| C — CONTRACT v2 re-anchor | **NOT ADOPTED AS SPECIFIED — and this row is the reason the ledger check was owed.** v2 landed for Q4_K (`1f78ebe`) and Q6_K (`bf13034`) with grading owed; the graded run then **fired C-FAIL-RATE** (`a53ee34`) and C was **parked, not adopted** (`bc8bd00`), even though its integer-totals oracle came back clean. The criterion that stopped it was then **disproven on its own terms**: the NULL-1 base rate measured 35/50 for a change that means nothing (`20fb0c9`), a mathematically identical pair scored like C on every axis (`4883b25`), and first-divergence count was formally **retired as a gate criterion on Q4_K_M** (`e253d7e`). What actually ships from this rung is the **maddubs twin** (`650c47c`) — 32 products per instruction and **not one bit changes** — plus **Q6_K on maddubs** (`cf0d8eb`, 1.108× at N=1, also bit-identical). Both are bit-exact, so neither needed the grading that stopped C |
| A′ — row tiling under v2 | **DEAD — rejected on measurement at 0.907×**, not on argument |
| D — repack | **never triggered.** §7.4 conditions it on a P-BAR miss after the ladder; P-BAR is met. Independently, the 08-17 attribution showed matvec already at parity (intercepts 52.95 vs 53.61 ms), so D would repack the term that is not the problem, and it was expected to fail its RAM gate on a 15.63 GB box |
| L4 — attention | **executed `ba4e140` + `755a43b`, and cost NO gate** — the cause was not arithmetic: attention had never run on the pool. Parallelism over independent outputs, so no twin, no grading run, no re-anchor; §7.6's 8-lane canonical fold is not needed and not done |

## 4. Correctness

**Byte-compare 10/10 in ONE sitting** (`a48de9f`, 15:18→16:20 on 08-18): all
eight Q4_K_M cases identical to the archive across **all four arms** (par 0/1 ×
lanes 1/4) at kv f32 **and** q8, at 2 and 6 threads, plus both Q8_0 p41 anchors
against the m3 archive. The binary-vs-source mtime guard passed on its face, so
the gate graded the artifact it claims to grade. This discharges the 4 cases
owed from the partial run of 08-17.

Also green and on the record from 08-17: Windows matrix 13/13, WSL plain 13/13,
**WSL TSan 13/13 with zero reports** (which discharged the TSan leg owed since
`76dee7f`), ASan clean on the real model at `--kv q8`, and the new test is
mutation-verified twice.

The P-BAR runner's own PHASE 1 re-ran the two owed correctness legs against the
relinked binary during leg 3: both **IDENTICAL**.

## 5. What the measurement changed elsewhere

- **The M5 alarm is retired.** "Extrapolated to ctx 1024, ~187 ms/token ⇒ M5's
  aggregate bars are unreachable" was measured against a single-threaded
  attention loop. Slope 0.1713 → **0.0289** ms/context-token; ctx 1024 now costs
  ~93 ms/token (~10.8 tok/s) against llama's ~70 (~14.2) — **0.76×, not 0.33×**.
  ROADMAP's M5 row is amended in place.
- **`ATT_LANES` is a simplification candidate, not an optimisation.** Isolated at
  ctx 1500: lanes alone 1.26×, pool alone 2.90×, both 2.94× — the lanes earn
  ~1.3% once the heads are parallel, because both changes attacked the same
  stalled add chain.
- **Price kernel arguments at half their argued gain.** L4 predicted 317 → ~63 ms
  and measured 106.75. Third instance: A′ predicted a win, measured 0.907×; Q6_K
  predicted 1.20× "as a floor", measured 1.108×.

## 6. Carried headroom and known defects (new this milestone)

1. **Arm A is unfit as an anchor.** The pre-maddubs archive predates `76dee7f`
   and read 9.99 / 5.30 / 0.92 / 8.6 across sittings while arm B and llama held
   to a few percent — stable *within* a sitting, up to 10.9× *between* them. A
   Task-Scheduler-classification explanation was tested and **refuted**
   (scheduled 9.044 vs interactive 7.999; `2026-08-19-m45-launch-context-probe.txt`).
   Fix: use llama as the readiness reference, or re-archive the baseline rebuilt
   with the QoS opt-out.
2. **Two fixes still owed to `main` from this branch**: the `bench --ops` heap
   overflow (broken since m3) and the Makefile `src/kernels/*.h` dependency fix.
3. **Not fixed, in priority order** (from the L4 note): prefill is still
   position-by-position; the 4× GQA read redundancy and ~3.5M `kv_vec_offset`
   calls/token at ctx 1500; dispatch is per (sequence, layer), so N=16 costs 576
   dispatches/token instead of 36. All three are M5 entry work.

## 7. §7.5 stamp preconditions — what is outstanding

- [x] P-BAR graded, both configurations, interleaved referent
- [x] correctness gate 10/10, one sitting, on the current binary
- [x] ROADMAP M4.5 row updated; M5 row's alarm retired in place
- [x] measurement artifacts committed
- [x] **§3's B and C rows verified against the branch history (2026-08-19).**
      Worth doing: the draft said "C landed, graded per §6" and **that was
      wrong** — C fired C-FAIL-RATE and was parked; the criterion that stopped
      it was later retired as unsound, and what ships is the bit-exact maddubs
      twin plus Q6_K. §3 is corrected
- [x] `stoke/MEMORY.md` build-log entry for the M4.5 close
- [x] `.superpowers` ledger block appended
- [x] owner line moved M4.5 → M5
- [x] `git tag m45` (local, not pushed) — **Kabir's word**
- [x] the two fixes in §6.2 landed on `main` (`df035ae`, `20f65e7`, both
      verified ancestors of `main` at M5 entry)
- [x] seven spent `stoke-m45-*` scheduled tasks from earlier sessions deleted —
      **zero `stoke*` scheduled tasks exist** (checked at M5 entry, 2026-08-19)

**These six were ticked at M5 ENTRY, not at close — they were all done on
2026-08-19 and the report was never re-opened to record it.** Verified one by
one rather than assumed: the tag, the ancestry of both fixes, the ledger block,
the MEMORY.md entry, the owner line, and an empty `Get-ScheduledTask stoke*`.
**A checklist that outlives the work it tracked reads as work outstanding** —
the same defect class as the spent helper, one level up, and cheap to close.
