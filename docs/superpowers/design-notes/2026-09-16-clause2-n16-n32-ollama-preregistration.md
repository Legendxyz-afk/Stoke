# 2026-09-16 — PRE-REGISTRATION: clause 2 at @16 (the clause's own width), @32 (the "N > 16" question), and the ollama-default half — one sitting, Kabir starts it

Written BEFORE the first unit. Kabir, 09-16 11:3x, on "are we ahead of llama /
what about N > 16": *"append this in memory, i'll run it later."* This note is
the sitting, sized and banded, so that when he says go nothing is decided on
the day. Protocol inherited unchanged from
`2026-09-08-clause2-end-to-end-preregistration.md` and the two @8 sittings
(`gate-reports/2026-09-15-clause2-e2e-8-{confirm,graded}.md`): one server
lifetime per unit, arms alternating with the order flipping every pair, a
per-unit quiesce gate, every unit banked as it completes, the verdict a
same-sitting ratio of end-to-end walls.

## 0. What is being answered, and what is NOT

| question | where it lives | status before this sitting |
|---|---|---|
| clause 2, llama half, **@16** — the clause says "at N=8/16" | ROADMAP Phase-1 paragraph, M7 row | @8 PASS with margin 1.181× (09-15 evening); **@16 never run live** |
| clause 2, **ollama-default half**, ≥2.0× @8 and @16 | same | **never measured against this binary**; the only referent is 2026-09-05's cell, a different sitting |
| **@32** — "are we ahead past 16" | Kabir's question, not a clause | unmeasured on BOTH sides; N=16's 1.20× is vs the LOCKSTEP `llama-batched-bench`, which flattered stoke by ~45% at @8 |
| clause 4's **re-anchor**: llama-server's own ITL max behind its prompt batches | `memos/2026-09-16-clause4-itl-stall-DISPOSITION-KABIR.md` §5 "owed" | the @8 banks kept no ITLs; this sitting banks them for both arms |

NOT answered here: anything about a GPU (stoke has no GPU path; the claim is
CPU concurrency), and anything about a second box (a many-core cloud CPU VM is
the meaningful second box, later).

## 1. ⚠⚠ Two things the pre-flight found that the plan in memory had wrong

1. **`src/server.c:57` — `MAX_CONNS 24`.** stoke's HTTP server holds 24
   connections; the listener is not even polled while none is free. At N=32
   the harness opens 32 closed-loop clients at t=0, so **eight would sit in
   the listen backlog until a slot frees — the "@32" stoke arm would be a
   24-wide server plus a queue** while llama-server runs 32 slots. The engine
   itself is fine (`--seqs 32` derives a 1,408-block pool; `BATCH_MAX_SEQS`
   is 256). The constant's own comment sets its ceiling at `NET_WAIT_MAX`
   (60, inside Windows' `FD_SETSIZE` 64). ⇒ **DECISION FOR KABIR: raise
   `MAX_CONNS` to 48 (still 12 under the ceiling) before the @32 units, or
   accept that @32 is not measurable as stated.** It is a one-line source
   change to the server before the 4.1 freeze; it needs the Windows suite +
   `serve_golden` and the WSL matrix re-run on the new HEAD, and the @32
   units run on THAT binary. @16 (16 < 24) is unaffected and can run first
   on `d56e496`.
2. **"32 prompt files" and "`--pool-blocks` sized up" are NOT needed on this
   path.** Both belong to `stoke bench --serve` (in-process, reads files,
   takes `--pool-blocks`). The live sitting goes through
   `tests/tooling/run_e2e_interleaved.py` → `bench.run_scenario`, which
   synthesises every client's prompt (`scenarios.build_messages`) and drives
   `stoke serve --seqs N --ctx 704`, whose pool is `N × 44` blocks by
   construction (pre-flighted: 352 @8 — the @8 sittings' number — 704 @16,
   1,408 @32). Also corrected: the memory's "~1.9 GB with prefix dedup on
   the S2 prompts" — the per-client salt sits at the FRONT of every user
   message, so dedup adopts only the template header; **stoke @32 needs the
   full f32 KV, 32 × 704 × 288 KB = 6.5 GB.**

## 2. What changed in the harness for this sitting (bench Python only, `src/` untouched)

- `bench/scenarios.py`: `s2-closed-32` (same shape: 512 prompt words + salt
  ≈ 525 tokens, 128 generated, closed loop, no shared prefix) as `S2_WIDE`,
  selectable by name only — NOT in the locked `S2` family the M0 bars were
  drawn on and NOT in `run_matrix.py`'s default set (the first cut put it in
  `S2` and would have silently grown a default full-matrix run by a 32-client
  cell against every baseline; same rule as `EXTRA_CONFIG_LABELS`).
- `bench/run_matrix.py`: **`llamacpp-tuned@ctx=N`** — the tuned cell with its
  TOTAL context set explicitly. Not a tuning of the referent: the policy's
  `P × 2048` does not FIT at N=32 (§3).
- `bench/adapters.py`: `Adapter.startup_lines` — the engine's output up to
  the moment it turns healthy, snapshotted. The log ring is 500 lines and a
  ten-minute N=32 unit's pool trace alone is ~800, so the lines a bank is
  VERIFIED from (`sched: STOKE_SCHED_CHUNK=`, the pool's `threads/spin`
  header, llama-server's `n_slots` / `n_ctx_slot`) would have rotated out
  and the bank would have read `thr=None chunk=None` for a server that had
  said both. Proven with a fake child that bursts 2,000 lines after binding:
  the announcements survive only in the snapshot.
- `tests/tooling/run_e2e_interleaved.py`: reads those announcements from the
  snapshot; banks `server_slots` / `server_ctx_slot` for llama units; banks
  **`itl` = pooled client-observed inter-token latency p50 / p95 / max plus
  every client's own max** for BOTH arms (clause 4's re-anchor needs the
  referent's number and the @8 banks had dropped the lists); prints
  `chunk= slots= ctx/slot= peakws= itl50/95/max=` on every unit line.
- Pre-flighted on the mock: 32 clients through `run_scenario`, 32/32
  completed, `unit()`'s reconstruction exact (`Σ(n_tokens−1) / span` equals
  the published aggregate to 1e-9), token counts `{128}`, every label the
  sitting names resolves to the adapter shape it claims, malformed labels
  refused, `bench/test_harness.py` ALL PASS.

## 3. Memory, from the @8 bank's own peak working sets

`proc.peak_ws_mb` in `bench/results/2026-09-15-clause2-e2e-8-graded.json`:
stoke @8 **4,203 MB** (= 2.5 GB model + 8 × 704 × 288 KB f32 KV, as
predicted); llama-server @8 `-c 16384` **7,277 MB**, of which f16 KV is
16,384 × 144 KB = 2.36 GB, so ~4.9 GB is model + compute buffers.

| arm | N | ctx | predicted peak WS | fits 15.6 GB (Chrome closed, one server at a time)? |
|---|---|---|---|---|
| stoke-tuned | 16 | 704/seq | ~5.8 GB | yes |
| stoke-tuned | 32 | 704/seq | ~9.0 GB | yes, ~3 GB headroom |
| llamacpp-tuned (policy) | 16 | 32,768 total = 2048/slot | ~9.6 GB | yes, but 2.3 GB past anything measured — **first-unit tripwire, §6** |
| llamacpp-tuned@ctx=11264 (fallback) | 16 | 704/slot | ~6.5 GB | yes |
| llamacpp-tuned (policy) | 32 | 65,536 total | **~14.3 GB** | **NO** |
| **llamacpp-tuned@ctx=22528** | 32 | 704/slot | ~8.1 GB | yes |
| ollama-default | 8, 16 | ollama's own (`n_ctx_slot 4096`, `n_slots 4` in the 09-05 bank) | as 09-05 | yes |

So: **@16 llama runs the policy (`-c 32768`, method-identical to @8); @32
llama runs `@ctx=22528` by necessity**, 704 per slot = stoke's per-sequence
ctx, which also retires the asymmetry the stoke-tuned comment in
`run_matrix.py` records. Per-slot ctx is capacity, not work: attention
costs the positions used, and S2 uses 653 of either. The bank verifies the
server's `n_ctx_slot` either way.

## 4. The sitting

Chrome closed; nothing else open; ports 8089 / 8091 / 11434 free; quiesce
gate per unit (default 10 tries); cooldown 45 s; `--timeout-s 900` (the
longest single request at @32 is the last slot's, ~5 min); ABBA within each
pair. A same-sitting `llama-bench pp512 @12T` referent (one-shot, no server)
before and after each block names the box state.

| block | scenario | arm A | arm B | pairs | est. unit | est. block |
|---|---|---|---|---|---|---|
| 1 | s2-closed-16 | llamacpp-tuned | stoke-tuned@threads=12,chunk=256 | 3 | ~5 min | ~30 min |
| 2 | s2-closed-32 | llamacpp-tuned@ctx=22528 | stoke-tuned@threads=12,chunk=256 | 3 | ~10 min | ~60 min |
| 3 | s2-closed-8 | ollama-default | stoke-tuned@threads=12,chunk=256 | 2 | ~4 min | ~16 min |
| 4 | s2-closed-16 | ollama-default | stoke-tuned@threads=12,chunk=256 | 2 | ~7 min | ~28 min |

~2 h 15 min plus referents and cooldowns; the memory's "~1 h" covered
blocks 1–2 only. Block 2 waits for the `MAX_CONNS` decision (§1); blocks 1,
3, 4 can run on `d56e496`. Each block is its own bank under
`bench/results/2026-09-16-clause2-n16-n32/`, launched DETACHED as a script
file (the 09-15 way; a harness kill mid-block keeps every banked unit,
`--append` resumes): `launch_block.sh <1-4>` in that directory (untracked
with the banks, `bench/results/*` is gitignored) runs the referent, the
block, the referent, and appends to `block<N>.log` / `block<N>.jsonl`. The stoke arm label is the @8-graded one so the arm is
the same configuration the clause was passed at; `threads 12` and `chunk
256` are the shipped defaults and the label makes the bank say so.

## 5. PREDICTIONS, frozen here (Opus, before any unit)

Stated to Kabir before this note existed: *ahead at both N, by more at 32
than at 16.* The mechanism bet: both engines prefill at ~70 t/s aggregate
in this harness (llama's TTFT min/p50 29/58 s @8 = its prompt-batch rounds,
the first at 29 s and the second at 58 s; stoke's 6/31 s = one prompt per two 256-token steps with
every admitted row decoding beside it), and the wall gap at @8 came from
the scheduler — decode rows riding every prefill step — not the kernel. At
larger N the prefill phase is a larger share of the wall, so the overlap is
worth more.

| quantity | point | band (PASS if inside; a miss is recorded, not re-fitted) |
|---|---|---|
| H1 · wall `llama / stoke` @16 | **1.20×** | [1.10, 1.35]; every pair > 1.0 |
| H2 · wall `llama / stoke` @32 | **1.30×** | [1.15, 1.50]; every pair > 1.0; **and > H1's reading** — if @32 reads BELOW @16 the mechanism bet is falsified and that is the finding |
| H3 · wall `ollama-default / stoke` @8 | 1.6× | [1.3, 1.9] ⇒ **≥2.0× MISSES** (the 09-05 ollama cell read 116.7 s ×3 against the 09-15 stoke 73.4 s — cross-sitting, a prior only) |
| H4 · wall `ollama-default / stoke` @16 | 1.7× | [1.3, 2.0] ⇒ predicted MISS (ollama serves 4 slots, so 16 clients queue in four rounds; the 09-05 cell read 234.5 s) |
| H5 · clause-4 re-anchor: llama-server client-observed ITL **max** @16 | ≥ 20 s | [15, 45] s — its streams stop for a whole prompt-batch iteration (~2,048 tokens ÷ ~70 t/s). **If it reads under 5 s, the disposition memo's "llama-server's streams stall the same way" is WRONG and the memo is corrected, not the band** |
| H6 · stoke ITL max @16 / @32 | 3.5 s / 5 s | [2.5, 6] / [3, 9] s — the chunk's forward with 16 / 32 decode rows riding; cadence p50 stays inside 250 ms at both |
| H7 · CPU-seconds `stoke / llama` | 0.75× @16, 0.8× @32 | [0.65, 0.95] both — the @8 reading was 0.751× |
| H8 · TTFT p50 `llama / stoke` | 1.9× @16, 2.5× @32 | [1.4, 3.5] — grows with N because llama's slots wait whole rounds |

## 6. TRIPWIRES (a unit that trips one is VOID, not a data point)

- `completed != N`, any client error, token count ≠ 128, reconstruction
  failed — the harness already refuses these.
- **Paging on the @16 llama arm at the policy ctx**: `proc.page_faults` >
  2× `peak_ws_mb × 256` (first-touch faults are ~WS/4 KB; the @8 unit read
  1.9 M at 7.28 GB), or CPU-seconds / wall below 8 where the @8 unit read
  19. ⇒ void the block, re-run it as `llamacpp-tuned@ctx=11264`, and say so.
- Either arm's own spread across its 3 units > 10% (the @8 sitting's worst
  was 9.8%, and one unit each) — the ratio is reported but graded only
  against that spread.
- `server_slots != N` or `server_ctx_slot != 704` on an `@ctx=` unit;
  `sched_chunk != 256` or `trace.threads != 12` on a stoke unit — the
  configuration is what the server said, never the label.
- Quiesce gate refused, or the pp512 referent moves > 8% across a block.
- The WSL VM vanishing is not a hazard here (Windows binaries, Windows
  harness), but the harness-kill one is: Chrome closed, detached launch.

## 7. What must be true before the first unit

1. This note exists (it does).
2. `MAX_CONNS` decision taken (§1) — block 2 only.
3. The WSL matrix on `d56e496` green (running detached at the time of
   writing; plain leg rc=0), and the Windows suite green on the binary the
   sitting runs.
4. Kabir's go. ASK-before-servers stands; he starts it.

## 8. 2026-09-18 — the two answers, and one deviation from §4, recorded BEFORE the first unit

Kabir, 09-18 09:0x: **MAX_CONNS 24 → 48: yes** (§1 item 1), and **go** for the
sitting (§7 item 4). The change is commit `9eca59e` (`src/server.c`: the
constant, its comment, and `MAX_CONNS + 1 <= NET_WAIT_MAX` as a
`_Static_assert` — proved red at 60 before the commit). Windows suite on the
new binary: `run_tests` 19/19 binaries (57/57), four goldens OK,
`serve_golden` 25/25 OK. WSL matrix (plain / asan / tsan) launched detached
on `9eca59e` at 09:16 (`build/wslmatrix/wsl_matrix_9eca59e.{sh,log}`); its
verdict lines are in the log, and no block starts until it has ended (the
matrix loads the box).

**Deviation from §4, decided by Opus:** all four blocks run on the `9eca59e`
binary, not blocks 1/3/4 on `bee09e6` and block 2 on the new sha. Reason:
one binary for the whole sitting is the better record — it is the binary
that ships to B7 and B8 — and the change cannot move an N ≤ 24 unit
(`Conn` is ~350 bytes plus per-connection heap; the listener's poll set
grows from 25 to 49 entries of one `select()` call per turn). The launcher
prints the binary's sha256 into every block log, so the bank says which
binary it was.

Bands, tripwires and order are unchanged: §5 and §6 as frozen on 09-16;
blocks 1 → 3 → 4 → 2.
