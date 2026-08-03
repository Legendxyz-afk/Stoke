# Opus session playbook — M2 close through M7

*2026-07-20. Written by a Fable session Kabir explicitly opened ("fable
override") to plan all remaining sessions. Executor: **Opus 4.8**, every
session, per the governance in `docs/ROADMAP.md`. This document is the
session-level orchestration layer: WHAT each session does, in what order,
with which recovery path. Design content (op specs, gate definitions,
scheduler design, risk catalogues) stays in the two design notes —
`2026-07-19-m2-op-spec-and-parity-protocol.md` ("the M2 note") and
`2026-07-19-m3-m7-forward-designs-and-risk-playbook.md` ("the forward
note") — and on design conflicts the notes win. On orchestration/recovery
conflicts this playbook wins. Probe-don't-recall applies to THIS document
too: facts marked [verified 07-20] were mechanically verified while
writing it; everything else is hypothesis-of-record — cheap to re-probe,
expensive to trust.*

**Goal:** carry stoke from its current mid-M2 suspended state to the M7
launch gate with zero design authority needed from anyone but Kabir
(memos) and zero session that cannot die safely.

**Architecture of the plan:** every work unit ends at a checkpoint =
(green bar reproduced by the manager) + (commit) + (ledger entry). A
session may execute one unit or several, but NEVER ends mid-unit
voluntarily. Background sonnet workers die at session boundaries by
design; each unit's brief carries its own recovery clause.

---

## 0. Standing session frame (run this at the top of EVERY session)

1. **Model gate.** The stage owner is **Opus 5** (M2→M7), and Kabir set the
   saved default back to Opus 5 on 2026-08-03. The owner is a FLOOR: a newer
   Opus is valid; anything below Opus is a HARD STOP. A Fable session touches
   stoke only on Kabir's explicit in-session override, and only for
   manager-tier state work or a ruling. *(Superseded 2026-08-03: this item
   previously said the default was Fable 5 and told the session to run
   `/model claude-opus-4-8`. Both were stale — repaired per §0.3's trust
   order rather than left to mislead a fresh session.)*
2. **Read order:** `docs/ROADMAP.md` (map) → `.superpowers/sdd/progress.md`
   final NEXT-SESSION block (exact position) → the current milestone's
   design-note section → this playbook's section for the current session.
3. **Ledger discipline (tightened after the 07-20 incident — see §1):**
   - The ledger is the ONE file that must never lag. Update it in the
     same work unit as the commit it describes, before starting the next
     unit.
   - Write an `IN FLIGHT:` ledger line BEFORE starting any manager-owned
     src edit or dispatching any worker — the line is the recovery
     breadcrumb if the session dies mid-edit.
   - Doc edits (ROADMAP, notes) commit WITH the work they describe, never
     sit uncommitted across a session boundary.
   - On any disagreement between records, trust order: `git log` >
     working tree > ledger > ROADMAP > memory. Repair the lagging record,
     mark the repair `[RECONSTRUCTED]`, one ledger line says why.
4. **Definition of done for a work unit:** worker report adjudicated (or
   manager work self-checked) → green bar REPRODUCED BY THE MANAGER
   (Windows `mingw32-make test` + `scripts\win-check.cmd`; WSL
   `bash scripts/wsl-check.sh` judged by its PRINTED `ALL GREEN`, never
   by pipeline exit) → Opus task-review for worker-produced diffs →
   commit (msg style `area: what -- detail`, NO attribution trailers) →
   ledger entry.
5. **Measurement discipline — THE COMPARABILITY RULE** (binding: Fable K3
   ruling 2026-08-03 §4,
   `docs/superpowers/memos/2026-08-03-k3-simd-headroom-RULING-FABLE.md`).
   Before quoting any two performance numbers together, in a memo, a
   commit message, the ROADMAP, a code comment, or the ledger:

   > **Same session, interleaved where both sides are runnable, cycles
   > cited, sidecar attached — or don't quote them together.**

   - A cross-session comparison is a HYPOTHESIS, not evidence. Re-measure
     before writing it down.
   - Rep-to-rep stability (±σ) certifies precision within a window and says
     NOTHING about that window's validity. `1.26 ± 0.01` and `2.46 ± 0.04`
     were both internally excellent, and one of them was wrong.
   - **Subtraction is comparison.** A difference of differently-conditioned
     measurements measures the conditions.
   - A field name is not a basis: name the denominator's stream AND window.
   - Absolutes are quarantined. No absolute tok/s or GB/s is a target,
     expectation, or comparison operand unless it comes from a
     quiesced-checklist session or is one side of a same-session
     interleaved pair. Everything else is labelled decoration.
   - Tools that mechanise this, and the only admissible sources:
     `python bench/append_history.py …` grades every row `gate`/`ambient`
     from a box sample bracketing each run; `python
     bench/referent_interleave.py --model … --cycles 3` runs the A,B,A,B
     against vendored llama and emits the paired record. **The attribution
     ladder may cite paired records only.** An `ambient` row is never a
     diff base, a referent, or quotable against another row.
   - Quiesced, mechanically: foreign CPU < 5% of the box AND free RAM >
     model + 2 GB, for every run in the row, proven by the sidecar rather
     than asserted. Close games and chat clients FIRST — on 2026-08-03 a
     resident game averaging 5.08 cores silently spanned two sessions'
     worth of measurements and inverted two conclusions.
6. **Escalation** = STOP + memo to Kabir on: gate fails twice same cause;
   >1 day stuck on one divergence; results contradict spec physics; ANY
   gate-loosening temptation (ε, waiver budget, bars, corpus trims).
   Memo file: `docs/superpowers/memos/YYYY-MM-DD-<topic>.md` (create the
   dir on first use), committed, plus surface it verbatim in the
   session's final message and one ledger line. Template:

   ```
   # MEMO — <date> — <milestone>: <topic>    [BLOCKING: <what it blocks> | FYI]
   WHAT: <one sentence — what failed or what decision is needed>
   EVIDENCE: <numbers, commits, file paths — no adjectives>
   OPTIONS: 1) … 2) … 3) …
   RECOMMENDATION: <one of the options, one sentence of why>
   IF DEFERRED: <what happens / what stays blocked>
   ```
7. **Workers:** sonnet, background, surgical briefs, constants PROBED at
   dispatch and written INTO the brief, disjoint pinned file-ownership
   lists, a mandatory mechanical-verification section, workers never
   commit. Parallel dispatch only within the independent tier of ONE
   milestone; cross-milestone never. Never dispatch a worker without
   first writing the ledger's recovery clause for it.
8. **Git:** local only, branch `main`, NO remote/push (Kabir 2026-07-16,
   binding). Tag at milestone close (`m2`…`m7`). Models and
   `.superpowers/sdd/*` are git-ignored; committed files under
   `src/ tests/ scripts/ Makefile` never cite git-ignored paths.

---

## 1. Verified state snapshot (2026-07-20, Fable session) — Session A's input

All items below were mechanically verified while writing this playbook
[verified 07-20]. Background: the previous session (2026-07-20, Opus)
updated `docs/ROADMAP.md` and the M2 note §8, fixed gguf.c for Q8_0, and
adjudicated W1 — then died WITHOUT committing anything or writing its
ledger entry. The ledger's last entry (W2 landing) still says "W1 in
flight". This snapshot supersedes that.

- **HEAD = `7fbdd2b`** (W2 scalar kernels). Tags through `m1`. Committed
  M2 state: entry chores `6f9109b`, corpus `f5d99ee`+`af18af8`, W2
  kernels `7fbdd2b`.
- **Uncommitted src diff** (inspected line-by-line; it is exactly the
  manager fix owed from the W2 landing entry, nothing extra):
  `src/gguf.c` + `src/gguf.h` add `GGUF_GGML_TYPE_Q8_0 = 8` with table
  row `{Q8_0, 32, 34}` and updated fatal text; `src/kernels/kernels.h`
  drops the interim `KERNELS_GGML_TYPE_Q8_0` #define for the enum;
  `src/kernels/vec.c` + `tests/test_kernels.c` swap to the enum name.
  UNVERIFIED (no green bar run yet) — Session A task A2.
- **Uncommitted doc edits:** `docs/ROADMAP.md` (position refresh) and the
  M2 note (§8 "PROBED (W1…)" block + the PENDING f16-KV item). Checked
  against evidence: faithful. Committed by THIS Fable session together
  with this playbook — so by the time Opus reads this, `git status`
  should show ONLY the src diff + untracked `tests/tooling/`. If it
  shows more, stop and reconcile per §0.3.
- **Untracked `tests/tooling/`** — W1's complete delivery: 
  `gguf_reader.py`, `dump_tensor.py`, `ref_forward.py`, and
  `validation/` (v1/v2/v3 prompts, `ident.txt`, 4 oracle JSON captures,
  `ref_out_v{1,2,3}.txt`, `v3_coalescing_evidence.txt`, `dump_v1/`
  L00–L35 layer dumps). The worker's report
  (`.superpowers/sdd/m2-w1-report.md`) still contains placeholders — the
  worker died before finalizing it after its 3 background ref runs
  completed (outputs on disk are complete and later than the report).
  The report will NEVER be finalized; the ledger adjudication is the
  record. Do not wait for it, do not regenerate it.
- **W1 adjudication (banked; do not redo):** tokenizer tables of the two
  GGUFs are fully identical (all 151,936 tokens + 151,387 merges +
  token_type element-wise); oracle determinism confirmed (two identical
  greedy runs); `/completion` applies no chat template (prompt token
  count check); the id ground truth is the top-level `tokens` field with
  `"return_tokens": true` — `completion_probabilities[*].id` COALESCES
  multi-byte-UTF-8 runs (64 real steps → 54 entries on v3; evidence file
  committed with the tooling); Q8_0 config keys match the §3 table; Q8_0
  34-byte block layout hand-annotated digit-exact.
- **Ref-vs-oracle validation result (re-derived independently this
  session from the on-disk artifacts):** v2 (code) and v3 (multilingual)
  ref==oracle **64/64 token-exact**. v1 (english) diverges at **position
  32**: oracle picks 2326, ref picks 825, and the oracle's own top-5 at
  that position is a 3-way near-tie — 2326 (−1.4927), 1378 (−1.5191),
  825 (−1.5204): top1–top2 gap **0.0264 nats** ≫ ε=1e-3, so NOT
  waivable under §8. Clean agreement for 32 positions then a reorder
  inside a 0.028-nat cluster is the signature of accumulated numeric
  drift (oracle's default **f16 KV cache**), not a ref formula bug (a
  formula bug would corrupt v2/v3 too). This is exactly the M2 note §8
  PENDING item; Session A task A5 settles it by experiment.
- **`-ctk f32 -ctv f32` (`--cache-type-k/v`) exist on the vendored
  b10037 `llama-server.exe`** — the experiment is runnable as designed.
- **No stale `llama-server.exe` is running** (W1's report left its kill
  unconfirmed; checked — none). Check again at every session that runs
  the oracle: `Get-Process llama-server`.
- **Disk:** 147 GB free on M:. **Stash:** `stash@{0}` ("partial task-7
  fix wave from dead session") is superseded by the landed fix waves —
  drop it in A8.

---

## 2. SESSION A — land M2's suspended state

Order matters: A2 before A3 (the tooling's P1/P5 stoke-side legs need a
stoke.exe that opens Q8_0), A5 needs A2's binary too. All of A1–A8 fit
one session; if context runs short, every task below is individually
checkpointable — close the current one, write the NEXT block, stop.

- [ ] **A1. Frame.** Run §0. Confirm `git status` shows exactly: 5
  modified src/test files + untracked `tests/tooling/`. Anything else →
  §0.3 reconciliation first.
- [ ] **A2. Verify + commit the Q8_0 fix.**
  Run: `mingw32-make test` then `scripts\win-check.cmd` (repo root),
  then `wsl bash scripts/wsl-check.sh` — need printed `ALL GREEN`.
  Expected: 9/9 + 43/43 + 22/22 + 47/47 + 4 golden gates OK.
  KNOWN POSSIBLE RED: a fix-wave-era negative test pins the OLD
  unsupported-type fatal text (`"only F32=0, Q4_K=12, Q6_K=14"` — now
  reads `"only F32=0, Q8_0=8, Q4_K=12, Q6_K=14"`). If exactly that pin
  fails, update the pinned substring to the new text (that is the test
  tracking an intended message change, not a regression) and rerun all
  three legs. Then ALSO prove the fix does its job:
  `build\win\stoke.exe info models\Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf`
  must now succeed (it fataled before).
  Commit (src+tests only, NOT tooling/docs):
  `m2: gguf Q8_0 open support -- probed 34-byte blocks, kernels enum dedup`
- [ ] **A3. Commit W1's tree.** Add a `.gitignore` line
  `tests/tooling/validation/dump_*/` (the L*.f32 layer dumps are
  regenerable diagnostics; everything else is evidence and ships).
  `git add tests/tooling .gitignore` → verify `git status` staged list =
  3 .py + validation files WITHOUT dump_v1/ → commit:
  `m2: tooling -- independent gguf reader, dump_tensor, numpy ref_forward + oracle validation captures`
  Then run the two stoke-side legs W1 could not (its concern #1):
  P1-leg: the `stoke info` dump from A2 — diff config values vs the M2
  note §3 table (expect identical; tensor types differ by design);
  P5-leg: `stoke tokenize` BOTH ggufs over
  `tests/tooling/validation/ident.txt` → identical id lists (W1 proved
  it via server `/tokenize`; this is the in-house closure). Ledger line
  with both results.
- [ ] **A4. Ledger reconstruction.** Write the missing 07-20 entry
  `[RECONSTRUCTED]` (dead session: W1 adjudicated, gguf fix authored,
  docs edited, no commits) + Session A's own entry as tasks complete.
- [ ] **A5. The f16-KV attribution experiment.** Full spec §3 below.
  Outcome recorded in ledger + M2 note §8 PENDING block resolved
  (amend the note in the same commit as the gate-invocation decision).
- [ ] **A6. Trust-order closure.** Per §3's outcome table: ref must
  stand validated on ≥3 prompts under the SETTLED oracle invocation
  before W3's L2 smoke may cite it as authority.
- [ ] **A7. Memo to Kabir** iff the gate invocation changed (expected
  outcome E1 → it does): `docs/superpowers/memos/2026-07-XX-m2-oracle-kv-invocation.md`,
  header `[BLOCKING: W4 dispatch]`. W3 (Session B) proceeds regardless —
  model.c does not depend on the oracle invocation.
- [ ] **A8. Cleanup + docs.** `git stash drop stash@{0}` (superseded,
  §1). Refresh ROADMAP's M2 position paragraph (short — point here).
  Commit docs: `docs: M2 position -- W1 landed, oracle invocation settled, session playbook live`
- [ ] **A9. If context allows:** write + dispatch the W3 brief (§4).
  Else NEXT-SESSION block = "Session B, W3 dispatch".

---

## 3. The f16-KV attribution experiment (task A5, fully specified)

**Question:** is v1's position-32 divergence caused by the oracle's
default f16 KV cache (⇒ gate oracle should run f32 KV), or by something
in the ref (⇒ fix ref before any C exists)?

**Procedure** (RAM note: f32 KV ≈ 1.2 GB server-side at 4k ctx on top of
the 4.3 GB mmap — fine alone; do NOT run ref_forward concurrently):

1. Start:
   `bench\vendor\llama-b10037\llama-server.exe -m models\Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf -t 1 --parallel 1 --port 8090 -ctk f32 -ctv f32`
   Poll `http://127.0.0.1:8090/health` until ready.
2. POST `/completion` with the EXACT W1 body + return_tokens, prompt =
   the exact bytes of `tests/tooling/validation/v1_english.txt`:
   `{"prompt": <v1 text>, "n_predict": 64, "temperature": 0, "n_probs": 5, "seed": 0, "return_tokens": true}`
   Run it TWICE — the two `tokens` arrays must be identical
   (determinism precondition under the new flags; if not → E4).
3. Same for v2 and v3 (they must STAY matching — regression check).
4. Save each response as `f32kv_v{1,2,3}.json` in the repo root (or
   scratchpad), then compare (run from repo root; pattern of record):

   ```python
   import json, pathlib
   base = pathlib.Path(r"tests/tooling/validation")
   f32 = json.loads(pathlib.Path("f32kv_v1.json").read_text())["tokens"]
   ref = [int(x) for x in (base/"ref_out_v1.txt").read_text().split()]
   old = json.loads((base/"oracle_v1_english.json").read_text())["tokens"]
   div = next((i for i in range(64) if f32[i] != ref[i]), None)
   print("f32kv-vs-ref first divergence:", div)
   print("f32kv-vs-defaultkv first divergence:",
         next((i for i in range(64) if f32[i] != old[i]), None))
   ```
5. Kill the server. Confirm: `Get-Process llama-server` → none.

**Outcome table:**

- **E1 — f32-KV oracle == ref 64/64 on v1 (and v2/v3 still match).**
  Attribution CONFIRMED. The parity-gate oracle invocation permanently
  gains `-ctk f32 -ctv f32`. Amend the M2 note §8 (PENDING → resolved,
  invocation line updated). This is a gate-parameter change: memo to
  Kabir (A7) BEFORE W4 is dispatched. Content: the invocation makes the
  oracle compute the same numeric class the gate exists to test (f32 KV
  both sides); token-exactness, ε, waiver budget, corpus are all
  UNCHANGED — this is confound removal, not loosening. Ref is now
  validated 3/3 → trusted as layer-level authority (§9 trust order).
- **E2 — f32-KV oracle differs from ref somewhere NEW.** Partial
  attribution. At the new divergence position, pull both gaps: if
  oracle gap < 1e-3 AND ref gap < 1e-3 AND same swapped pair → genuine
  coin-flip; validation still needs 3 clean prompts: swap v1 for one
  SHORT committed corpus prompt (pick from p00–p49, ≤16 input tokens),
  ref-run it, require clean; record everything. If the new divergence
  is NOT coin-flip-shaped → treat as E3.
- **E3 — f32-KV oracle STILL says 2326 at pos 32 (identical to default
  KV).** KV type is not the cause. Remaining candidates: (a) llama.cpp's
  activation quantization (it quantizes activations to Q8-blocks for
  integer dots — not switchable by flag), (b) a ref numeric subtlety.
  Do NOT grind: this is results-contradicting-expectation → STOP +
  memo `[BLOCKING: W4 dispatch]` presenting the evidence and options:
  (1) proceed with default invocation, expect rare non-waivable
  near-tie hits, adjudicate each via the L0–L3 ladder + per-hit
  evidence, escalate only if >2 on the recorded run; (2) also acquire
  the F16 GGUF (~7.5 GB) as a third oracle datapoint before deciding;
  (3) any ε/protocol change — present as option, NEVER recommend.
  Recommend (1) unless the evidence says otherwise at that point.
  Note for later: M3's integer-core switch makes stoke's math MATCH the
  oracle's activation-quant class, so an E3-class residual should
  shrink at M3 — say so in the memo.
- **E4 — flags rejected / nondeterminism under flags.** Probe accepted
  type strings via `--help`/error text (expect lowercase `f32`). If
  determinism itself breaks under f32-KV: record runs, fall back to
  default-KV invocation + E3's memo path.

---

## 4. SESSION B — W3: model.c + config + kv + `stoke run`

Dispatch AFTER: A2/A3 committed; A5 settled (any outcome). Base = the
Session-A HEAD. One worker, background, sonnet.

**W3 brief must contain** (write it fresh at dispatch; probe-don't-recall):

- Ownership: `src/model.c`, `src/model.h`, `src/kv.c`, `src/kv.h`,
  `src/main.c` (the `run` subcommand only), `tests/test_model.c`,
  Makefile auto-picks new src (verify object list grows). NOTHING else —
  not kernels, not gguf, not tooling.
- Read list: M2 note §3 (config contract + TRAP #1/#2/#3 verbatim), §6
  (pass + kv_view contract verbatim), §7 (CLI contract), probe notes
  P3.3/P3.4 (rope math + graph order, verbatim payloads);
  `src/kernels/kernels.h` (the kernel contracts W2 froze).
- Constants pinned IN the brief (from §3 + A3's stoke-side Q8_0 info
  dump): all config values, per-layer tensor name list, dims table,
  `max_ctx=4096`, KV cost 288 KB/token, eos=151645.
- CLI additions beyond §7: `--probe-pos N` — after generation, print to
  stderr stoke's top-5 (id + logit, sorted) at generated position N.
  The W4 waiver protocol consumes this (stoke-side gap at a divergence
  position). Cheap: keep the last-step logits when N == current pos.
- kv_view: EXACTLY the §6 signature; all KV addressing inside kv.c.
- Mandatory verification (worker-run, manager-reproduced):
  1. All existing tests still green both platforms.
  2. Shape validation: every tensor's dims checked against the table at
     load; prove the fatal fires (doctored expectation test or a
     wrong-file negative test).
  3. **L2 smoke vs the ref:** author `tests/tooling/validation/debug8.txt`
     (~8 input tokens); ids via `stoke tokenize` (Q8_0 file);
     `build\win\stoke.exe run models\...Q8_0.gguf --raw -f ...debug8.txt -n 1 --dump <dir>`
     vs `python tests/tooling/ref_forward.py <gguf> --ids <same> --dump-layers <dir2>`;
     max-abs-diff per layer over all 36 layers < 1e-3 (diagnostic bound,
     §9), AND stoke's first generated id == ref's.
  4. 64-token greedy on v1/v2/v3 vs `ref_out_v*.txt`: v2/v3 must match
     64/64. EXPECTATION SET IN THE BRIEF: C-vs-ref may legitimately
     flip at a genuine near-tie (different summation order, both f32) —
     v1's pos-32 cluster is the known candidate. Agreement through
     pos 31 required; a pos-32 flip is RECORDED, not failed. Any OTHER
     divergence = report DONE_WITH_CONCERNS with position + both ids.
- Recovery clause: on session death, check for
  `.superpowers/sdd/m2-w3-report.md` + `src/model.c`; report present →
  verify per brief → review → commit; absent/partial →
  `git clean` the owned paths → re-dispatch.

**Manager verify + the DEEP task-review** (this is the milestone's
highest-risk diff; review checklist):
head_dim never computed by division (TRAP #1); tied head reads
token_embd (TRAP #2); rope pairs (k, k+64) with the P3.3 iterative theta
(TRAP #3); QK-norm BEFORE rope, per head, shared [128] weight; GQA
mapping q-head h → kv-head h/4; scale 1/√128; kv addressing only in
kv.c (grep proves it); fixed accumulation order (determinism); fatal()
choke point only; no new file-scope mutable globals (M6 ABI prep, note
§12); scratch allocation pattern sane (lazily is allowed at M2, no
per-token malloc churn).
Commit: `m2: scalar forward pass -- model.c + kv + stoke run, L2-clean vs ref`

---

## 5. SESSION C — W4: gate harness + THE M2 GATE

Prereq: W3 landed; oracle invocation settled + Kabir's ack of the A7
memo if one was sent (the memo says BLOCKING: W4 — honor it).

**W4 brief must contain:**

- Ownership: `tests/tooling/parity_gate.py` + `tests/tooling/README-gate.md`
  + its report. Nothing in src.
- Harness contract: manages the oracle lifecycle itself (start
  `llama-server.exe -m <Q8_0> -t 1 --parallel 1 --port 8090` + the
  settled KV flags; health-poll; kill on exit — assert no
  `llama-server` survives). Records the server build identity in the
  gate report (probe `/props` for the build/commit field at dispatch;
  guards against a PATH-shadowed binary). Per prompt: send the file's
  exact bytes as `prompt`, body pinned as §3.2 + `return_tokens: true`;
  ground truth = top-level `tokens`. Stoke side:
  `stoke run <Q8_0> --raw -f <prompt> -n 64` → ids. Compare FULL
  sequences (eos included if generated; lengths must agree).
  Process-parallel across prompts — **RAM-bound, not core-bound, and the
  ORACLE dominates** (amended 2026-07-27 by live measurement; supersedes
  BOTH this line's original "~6 workers" AND the same day's first
  amendment claiming `--jobs 4` ≈ 12.5 GB, which under-counted the
  server): **`llama-server` alone holds ~11.0 GB PRIVATE** — it sizes KV
  from the model's default context, not from the ~554 positions this
  gate needs — and each `stoke.exe` adds **1,181 MB PRIVATE** (eager
  `[36][4096][8][128]` f32 K+V at `model_open`). Budget = 11.0 +
  jobs×1.18 + ~4.3 shared ⇒ **≈20 GB at 4 jobs, ≈16.5 GB even at 1 job,
  on a 16 GB box.** Lowering `--jobs` does not rescue this; **capping the
  oracle's context (`-c`) does** — but that is a change to the SETTLED
  invocation, hence gate-affecting and KABIR'S CALL (pending), and if
  adopted must first be proven to leave the oracle's greedy ids
  byte-identical to the committed v1/v2/v3 captures. Oracle calls stay
  SERIALIZED (`--parallel 1`); `--jobs` parallelizes the stoke side only.
  Stoke itself is compute-bound, not bandwidth-bound (scalar streams only
  ~811 MB/s of weights against DDR5's tens of GB/s), so jobs would scale
  near-linearly if RAM allowed.
  Dev mode caches per-prompt results; `--clean` ignores the cache — THE
  gate run is `--clean`.
- Waiver automation (§8 protocol, mechanical): on first divergence at p,
  oracle top-5 from `completion_probabilities` ALIGNED BY CONSUMING
  `tokens` against entry `bytes` (the coalescing fix — unit-test the
  aligner against the committed
  `tests/tooling/validation/oracle_v3_multilingual.json`, 64 tokens /
  54 entries); stoke top-5 via `--probe-pos p` re-run. Waivable iff
  both gaps < 1e-3 nats AND same swapped pair → REPLACE prompt with
  next unused reserve (r00…r09 in order), record everything, budget ≤2.
  Anything else = GATE FAIL → stop, report, §9 ladder time.
- Recovery clause as W3's.

**Sizing the run (measured 2026-07-27, Q8_0 scalar, warm page cache —
supersedes the earlier "~8 s/token ⇒ 6–7 h", which counted only decodes):**
~**5.3 s per forward pass**, and the corpus is mean 77.8 / max 490 prompt
tokens, so a 50×64 run is **7,038 forwards ≈ 9–11.5 h SERIAL**, prefill
being 55% of it. At `--jobs 4` expect ~3–4 h wall. **Order the prompts
longest-first** — p40–p44 are ~553 forwards ≈ 49 min each and in corpus
order they strand the tail. Detached + resumable + single log writer (the
ops lesson); a session boundary must not cost the run.

**The recorded gate run (manager-driven):** quiesce the box (no bench
fairness needed — parity is not timing — but don't fight RAM: close
heavy apps), `--clean`, all 50, one run. Write
`docs/superpowers/gate-reports/2026-07-XX-m2-parity.md`: per-prompt
result, waivers with full evidence, oracle invocation + build identity,
corpus + reserve commit hashes, stoke commit hash.

**M2 GATE CHECKLIST (stamp = all boxes recorded in ledger + MEMORY.md):**
- [ ] 50/50 token-exact on the recorded `--clean` run, waivers ≤2 fully
      documented per §8 (expectation: 0).
- [ ] Windows `mingw32-make test` + `win-check.cmd` green at gate HEAD.
- [ ] WSL `wsl-check.sh` plain+asan+tsan printed ALL GREEN at gate HEAD.
- [ ] No committed `tests/data/*.golden` changed since `m1` (git diff).
- [ ] Corpus committed before forward pass (git log order: `f5d99ee` ≺
      model.c commit) — record the two hashes.
- [ ] Tokenizer-table identity assert banked (A3).
- [ ] Oracle invocation + build identity in the gate report; report
      committed.
- [ ] L0 net green (in matrix); L1 spot-check banked (W1's byte
      annotation); L2 smoke banked (W3).
- [ ] Ledger + MEMORY.md build-log entry; ROADMAP M2 row → ✅ with date
      + tag; memory `engine-project.md` owner line updated at transition.
- [ ] `git tag m2` at the close-out commit.

Close-out commit: `m2: scalar forward pass complete -- 50-prompt greedy parity vs llama-server`
Then write the M3 entry NEXT-SESSION block (→ §6 K0).

**If the gate FAILS:** §9 ladder from L0 upward, one divergence at a
time, cheapest rung first. Budget one session per divergence class;
same cause failing twice or >1 day → memo. NEVER: trim a prompt, retry
with a different corpus, adjust ε, or re-roll the oracle until it
agrees.

---

## 6. M3 — SIMD/VNNI kernels + thread pool (forward note §M3 is the design authority)

Work units (each = brief → worker → verify → review → commit → ledger):

- **K0 (manager, entry session):** probes into
  `.superpowers/sdd/m3-probe-notes.md`:
  P-A: fetch from b10037 raw sources — `ggml-quants.c`
  `get_scale_min_k4` + `block_q4_K`/`block_q6_K`/`block_q4_0` structs +
  the `vec_dot_q8_0_q8_0` / dpbusd sign-trick idiom +
  `quantize_row_q8_0` (activation-quant reference) — VERBATIM into the
  notes. P-B: toolchain — `gcc --version` on BOTH legs; compile a
  5-line `_mm256_dpbusd_avx_epi32` probe with
  `-mavx2 -mfma -mf16c -mavxvnni` on both; if the Windows MinGW gcc
  lacks AVX-VNNI support (needs GCC 11+), memo to Kabir
  `[BLOCKING: K3]` with the upgrade path (msys2 pacman / winget) —
  WSL-first development is the interim. P-C: 20-line CPUID probe
  (leaf 7 subleaf 1, EAX bit 4) run on the box, output recorded.
  P-D: start the Q4_0 GGUF download in background (bartowski, same
  repo; disk verified fine) — it serves the real-bytes leg of K4.
  P-E: append the M2-gate `history.jsonl` row if not already done.
- **K1: microbench** — `stoke bench --ops` (the forward note's mandated
  spelling): per-op GB/s + µs
  over the real model's shapes (matvec per tensor kind, attention,
  norms, rope, logits row), plus single-stream tok/s. This is the
  attribution instrument for the whole milestone — it lands FIRST
  (the forward note's miss-the-bar ladder runs on it).
- **K2: activation quantization + scalar integer core** for Q8_0
  weights (quantize activations per 32-block absmax/127 exactly as the
  probed `quantize_row_q8_0`; i8×i8→i32 accumulate → scale d_w·d_a).
  This CHANGES token streams → immediately re-run the 50-prompt parity
  gate (Q8_0 file, settled invocation, waiver protocol) in the same
  unit. Note: this moves stoke's math INTO the oracle's numeric class —
  if A5 ended in E3, re-examine v1 here and record.
- **K3: SIMD twins** for f32 + q8_0-int paths (AVX2+FMA+F16C+VNNI VEX;
  `loadu` everywhere — gguf mmap rows are unaligned) + runtime CPUID
  dispatch + scalar fallback. Gate piece: exact-diff harness — integer
  paths bit-exact scalar↔SIMD, f32 paths ≤1e-5 rel, 1e4 random tensors
  per kernel. Owns `src/kernels/simd*.c` only.
- **K4: K-quants** — dump_tensor.py grows Q4_K/Q6_K/Q4_0 dequant
  (python from P-A formulas) FIRST, verified against real rows of the
  Q4_K_M + Q4_0 files; then C scalar (integer core), byte-exact f32 vs
  python on real rows × {q4_k, q6_k, q4_0} × layers {0,17,35}; then
  SIMD twins + exact-diff. The 6-bit scale/min unpack
  (`get_scale_min_k4`) is the classic transcription bug — python-first
  exists to catch it before C. Owns `src/kernels/kquant*.c` (disjoint
  from K3 → K3 ‖ K4-python may run concurrently; K4-C follows K3's
  commit or stays file-disjoint; manager owns the Makefile at commit).
- **K5: pool.c** — work-stealing 64-row chunks (atomic next-chunk
  counter), parallel matvec for the 7 big projections + the logits
  matvec, attention across heads, small ops single-thread. TSAN ON
  EVERY COMMIT from here to M7 (cross-cutting rule). Thread-count
  experiment (6P / 10T / SMT) on K1's bench, per-op defaults recorded.
- **K6: the bar** — full parity re-pass on Q8_0 AND Q4_K_M (K-quants
  now live); single-stream ≥7.8 tok/s on Q4_K_M (re-anchor to a fresh
  llama.cpp-tuned run if one is taken; M0 machine checklist verbatim —
  quiesced, plugged in, performance power plan, closed apps); miss →
  K1 attribution ladder (forward note's ranked culprits), fix, repeat.
  GATE = forward note §M3 checklist + history.jsonl + tag `m3`.

M3 session shape: K0+K1 one session; K2 one; K3/K4 two-three (parallel
where file-disjoint); K5 one-two (tsan soak time); K6 one. Recovery per
unit as always.

---

## 7. M4 — paged KV + static batch (forward note §M4 authority)

**Two orchestration LOCKs the forward note left open (locked here, now,
so no session re-derives them):**

1. **Prefix-dedup mechanism:** full blocks are keyed in a map by the
   token-id sequence they cover (hash of ids, verified by comparing the
   actual ids on hit — hash collisions must be IMPOSSIBLE to act on);
   admission walks a new seq's full blocks and reuses on verified hit
   (refcount++). Partial tail blocks are never shared. This satisfies
   sim invariant 3 by construction and the plan's dedup byte-verify.
2. **Curve RAM math (16 GB box):** f32 KV on Qwen3-4B = 288 KB/token.
   16 seqs × 2048 ctx × 288 KB ≈ 9.4 GB — does NOT fit beside the
   model. The M4 curve therefore runs Q4_K_M, per-slot ctx **1024**,
   pool 5 GB (16×1024×288 KB ≈ 4.7 GB), all N same ctx. Consequence
   for M5 (recorded now): the HEADLINE bars at N=16 (S2, ctx 2048)
   cannot run f32-KV — they run **Q8-KV** (2.4 GB) — while N=8 fits
   both ways (4.7 GB). So M4's f32-vs-q8 divergence count is not a
   nice-to-have, it is the evidence Kabir needs to bless the N=16 bar
   configuration → send that memo AT M4 CLOSE, not during M5.

Work units:

- **N0 (manager):** write `src/kv.h`'s FULL M4 contract (kv_view
  unchanged + pool init/free, seq create/release, append, fork,
  block-table introspection for tests — exact signatures) into BOTH
  briefs. No probes needed (internal design; the note is the authority).
- **N1: the sim** — `tests/test_kv_sim.c`: seeded randomized
  alloc/free/append/fork/release storm, ≥1e6 ops, ALL 11 invariants
  asserted after EVERY op, 3 seeds, ASan. ‖ **N2: kv.c real body**
  (block pool, tables, refcount, CoW, dedup map per LOCK 1). N1 ‖ N2
  are file-disjoint against the frozen kv.h; dispatch together.
- **N3: batched decode** — model.c restructure to seqs-inner/
  weights-outer with IDENTICAL per-seq arithmetic (composition
  independence by construction); full-reservation admission; gates in
  the same unit: batch-of-N token-identity (5 prompts × N∈{2,8} vs
  single-seq) + M2 parity corpus re-run at N=1 (regression net —
  cheap now, ~minutes at M3 speeds).
- **N4: close** — Q8-KV option (property bound |dq(q(x))−x| ≤ d/2 per
  element + 50-corpus f32-vs-q8 divergence COUNT recorded); block-size
  16 vs 8/32 measured on the curve (keep 16 unless >5% aggregate);
  scaling curve N∈{1..16} per LOCK 2 config (monotone non-decreasing
  aggregate = gate; a DROP is software, find it — note's ranked
  causes); history.jsonl; forward-note gate checklist; tag `m4`;
  **memo to Kabir**: Q8-KV divergence numbers + the N=16 bar
  configuration recommendation `[BLOCKING: M5 bar run]`.

---

## 8. M5 — continuous batching (forward note §M5 authority; THE product gate)

The scheduler design is fully LOCKED in the note (states, step loop,
watermark W=max(8,2·|RUNNING|), preempt-youngest push-front,
replay-never-resample, chunk 128 co-run, SPSC rings, mutex+cv submit).
Orchestration adds ONE missing deliverable:

**The in-process bench driver** (`stoke bench --serve`): M6's server
doesn't exist yet, but the bars are M0-methodology numbers. The driver
loads committed prompt sets exported once from the M0 harness's
calibrated synth_prompt (S2/S3 shapes → `tests/data/bench_prompts/`,
generated by a small python exporter using the EXISTING bench harness
code, seeds recorded, committed), submits N concurrent requests
in-process, measures aggregate decode tok/s + per-token ITL
percentiles/max, warmup + 3 salted runs, median, fresh scheduler state
per run. Its report format mirrors the M0 table so the ≥17.7/≥14.2/
≥10.8/≥16.8 comparisons are apples-to-apples (same prompts, same
counting: arrivals-driven decode tokens).

Work units:

- **S0 (manager):** re-read note §M5 whole; write sched.h contract +
  the sim's trace-catalogue file list into briefs.
- **S1: sim harness** (virtual clock, pluggable cost model, fake model
  = hash(req_id,pos), arrival generators incl. memory-pressure trace;
  reference run = same trace, infinite pool). All asserts from the
  note (11 M4 invariants + watermark + ≤1 prefill + FCFS +
  victim-youngest + every-runnable-decodes-every-step + replay
  losslessness + all-finish + thrash metric reported).
- **S2: sched.c** against the sim until every trace × 3 seeds ×
  {constant, linear} cost models is green.
- **S3: rings + submit queue + cancel** (C11 atomics acquire/release,
  power-of-2 caps; mutex+cv submit; cancel flag at step boundaries).
  TSan target: the concurrency surface is exactly 3 objects; ANY
  report is real (no suppressions, ever).
- **S4: bench driver** as above + prompt-set export + a smoke S2@N=4.
- **S5: the bars** — real-model runs per M4's LOCK 2 + Kabir's memo
  ack: S2@8 f32-KV AND Q8-KV, S2@16 Q8-KV, S3 fanout, ITL
  percentiles (<250 ms max stall with co-running prefill), forced
  preemption real-model trace (lossless vs infinite-pool reference).
  Bar miss → the note's attribution ladder IN ORDER (static-curve
  ceiling first — if M4's curve already missed, the scheduler was
  never the problem). llamacpp-tuned bars met but 16.8@16 (the
  inversion-derived 2× bar) not → that is the RESULTS-VS-PHYSICS memo
  with the curve attached; do not grind.
- **S6: 1h randomized soak** ASan then TSan, DETACHED (Task Scheduler /
  `.cmd` runner — session-tracked processes get reaped; single log
  writer; resumable), then gate checklist + history.jsonl + tag `m5`.

---

## 9. M6 — `stoke.h` freeze + server (forward note §M6 authority)

Everything is decided in the note (trap mechanism, three failure
classes, ABI checklist, server design, compat traps, stop-string
tail-window, seeded sampling on the replay rule). Orchestration:

- **F0 (manager):** probes — `pip show openai` version pin for the
  suite; `/props`-style compat checks NOT needed here (own server);
  decide port default; **bind 127.0.0.1 by default** (avoids the
  Windows-Firewall first-listen prompt entirely; `--host` opt-in for
  LAN — document).
- **F1: ABI + trap** — common.h fatal() gains the thread-local trap
  check; `stoke_open/close/last_error/set_log_callback/abi_version`;
  partial-construction teardown via the ordinary *_free on non-NULL;
  x* wrappers' trap-aware OOM path; header doc pass; C++ TU compile
  test joins the matrix. Review focus: NO fatal call site changes; no
  new globals; every public symbol prefixed.
- **F2: sampling** — temperature/top-k/top-p/repeat-penalty/stop/seed;
  per-request splitmix64 on the scheduler thread; replay rule keeps
  seeded streams identical under preemption (sim gains a seeded-replay
  assert).
- **F3: server core** — socket shim (winsock/POSIX, ~50 lines), HTTP
  parser + committed garbage-input corpus, SSE framing + 15s `: ping`
  heartbeat + slow-client disconnect policy (bounded per-conn buffer;
  never block the scheduler on a socket).
- **F4: OpenAI endpoints** — chat/completions/models + `/health`;
  ChatML via existing tok_apply_chatml; `/v1/completions` raw;
  stop-strings consumer-side tail window; the compat-trap list from
  the note tested against openai-python, not docs.
- **F5: gates** — openai-python suite; curl goldens committed; **the
  definitive smoke: the M0 bench harness drives a `stoke` adapter
  UNCHANGED on S1**; ABI checklist as a recorded mechanical pass;
  10-min randomized ASan serve soak; garbage corpus green; matrix
  green; THEN tag `m6` = the header freeze (errors never renumbered
  after this tag).

---

## 10. M7 — harden + launch-ready (forward note §M7 authority)

- **T0 (manager, entry):** probes — download Llama-3.2-3B-Instruct
  Q8_0 (parity) + Q4_K_M (dev) + the unsloth Q4_K_M twin (writer-diff)
  [disk check first: ≥15 GB free]; fetch VERBATIM from b10037: the
  llama arch graph (llm_build_llama), rope NORM path + the llama3
  frequency-scaling application (does the GGUF carry a
  `rope_freqs.weight` tensor? gate ops on TENSOR PRESENCE, never arch
  strings), llama-bpe pre-tokenizer regex from llama-vocab.cpp, the
  header-format template + special ids from the actual GGUF; `stoke
  info` both new files, config tables into the probe notes.
- **T1: tokenizer** — per-`pre` regex table (second entry: llama-bpe;
  case-insensitive contractions, `\p{N}{1,3}`, its punctuation
  classes; unicode table categories extended if the regex needs them),
  oracle-gated byte-exact vs `llama-tokenize` with THAT model, M1
  method (corpus reuse + model-specific lines + a fresh
  codepoint-sweep leg if the regex touches new categories).
- **T2: template #2** — Llama-3 header format beside ChatML, probed
  ids, goldens both parse modes.
- **T3: model deltas + parity** — conditional QK-norm on tensor
  presence; ADJACENT (NORM) rope pairing + the fetched scaling
  formula; head_dim still read from key_length; its OWN 50-prompt
  corpus **including ≥5 prompts with ≥1k-token contexts** (wrong
  rope-scaling passes short and fails long — that is the trap the
  long prompts exist for); full parity gate vs the oracle with the
  llama Q8_0 file under the SAME settled invocation.
- **T4: unsloth twin** — `stoke info` structural diff FIRST (every
  metadata delta recorded as a finding), then the full parity corpus
  on the unsloth file.
- **T5: hardening** — gguf fuzz corpus expansion (both writers'
  header shapes × truncation/count/type garbage), tok fuzz on random
  bytes, ASan clean; **24h detached soak** (Task Scheduler runner,
  RSS-slope watch ~0 after warmup, single log writer) + 2h WSL
  ASan-leak leg.
- **T6: the report** — RE-RUN all baselines at current dated versions
  (ollama + llama.cpp latest release, versions pinned in the report;
  M0's numbers are stale by now and must not be republished);
  README (build, embed-as-library example, server quickstart, honest
  table); reproduce-from-clean-checkout by one command
  (run_matrix.cmd + report generator); **name-collision memo to Kabir
  BEFORE README work** (renaming after polish wastes it).
- **T7: gate + sign-offs** — forward-note §M7 gate checklist; memos:
  name, Q8-KV default (if still open), launch. Tag `m7`. Phase-1.5
  planning doc (CUDA/NEON) is an M7 deliverable — one session, after
  the gate.

---

## 11. Operational problem catalogue (symptom → action)

**Session/state:**
- Fresh session on wrong model → §0.1. Never work stoke on Fable
  without Kabir's explicit in-session override.
- Ledger tail ≠ ROADMAP ≠ git log → trust order §0.3; repair with
  `[RECONSTRUCTED]`; the 07-20 incident is the reference case.
- Session died mid-manager-edit → the `IN FLIGHT` ledger line names it;
  verify the half-edit against its stated intent or stash
  (label it) and redo from clean; never build on unverified half-work.
- Session died with workers in flight → per-worker recovery clause in
  the ledger: report+deliverables present → verify → review → commit;
  else `git clean` owned paths → re-dispatch from the brief (briefs
  survive: `.superpowers/sdd/*-brief.md`).
- Context running out mid-session → finish the current unit's
  commit+ledger, write the NEXT block, stop. Do not open new units or
  dispatch new workers on a shrinking budget.

**Workers:**
- Worker touched paths outside its pinned list → do not commit those
  paths; `git checkout --` them after confirming they're not
  load-bearing to its owned diff; note in the review.
- Worker report contradicts its brief's stated facts → probe wins:
  verify which is right; if the brief was wrong, ratify the worker
  (6a/B4 precedent), fix the brief text for reuse, ledger line.
- Worker returned DONE but the green bar fails for you → the
  reproduction IS the bar; adjudicate the delta, fix or bounce.
- Worker wrote CRLF into `.sh`/gate scripts → symptom: WSL dash
  `set: Illegal option -` with garbled first line while msys sh works;
  check FROM WSL with `od -c` (Git-Bash grep/cat hide `\r`);
  `.gitattributes` pins `*.sh` LF / `*.cmd` CRLF / parity `*.txt`
  `-text` — keep extending it for new file classes.

**Oracle / bench:**
- Stale `llama-server.exe` (holds 4.3 GB mmap, skews RAM + may answer
  a port) → `Get-Process llama-server` before/after every oracle use;
  kill leftovers before any gate or bench run.
- Port 8090 busy → find owner (`netstat -ano | findstr 8090`), kill if
  it's ours, else `--port 8091` + record in the gate report.
- Oracle output flaky across identical runs → precondition broken:
  confirm `-t 1 --parallel 1`, seed, same body; if still flaky the
  gate is INVALID, stop and memo — never gate against a nondeterministic
  oracle.
- `completion_probabilities` shorter than 64 → that's the UTF-8
  coalescing (v3 evidence file is the fixture); ids come from `tokens`,
  logprob entries align by byte-consumption. Never index them 1:1.
- `tokens_predicted` ≠ len(tokens) → it echoes n_predict; ignore it.
- Timing run on battery / thermal throttle / busy box → M0 machine
  checklist verbatim before EVERY timed run (plugged in, performance
  plan, quiesced, warmup). Parity runs don't need quiesce, only RAM.

**RAM/disk (16 GB box):**
- Never: oracle + ref_forward concurrently; concurrency benching on the
  Q8_0 file; f32-KV at 16×2048 (LOCK 2 math, §7).
- Before each M7 download: ≥15 GB free check. Models never committed.
- Windows Defender real-time scan can tax first-touch of a fresh 4 GB
  GGUF → expect a slow first mmap pass; warmup covers it; do not
  "fix" it by exclusions without Kabir.

**Build/toolchain:**
- New src dir → Makefile wildcard picks `.c` but object dirs need
  `mkdir -p $(dir $@)` (W2 lesson — already in place; keep the pattern).
- WSL link errors on math syms → `-lm` (already in LIBS; keep it).
- `-mavxvnni` unknown flag → GCC < 11 on that leg; K0's P-B probe +
  memo path. WSL-first in the interim.
- NEVER add `-ffast-math`/`-Ofast` (breaks inf/nan semantics softmax
  and the f16 fatal paths depend on; determinism doctrine).
- `wsl <cmd> | tail` exit status is tail's → judge WSL by printed
  ALL GREEN only.

**Numerics:**
- C vs numpy flip at a genuine near-tie (both f32, different summation
  order) → expected class; record position + both gaps; it is a
  finding only if the gap structure isn't tie-shaped (then L2/L3).
- New python tooling → f32() validator pattern + a float64-leak
  negative selftest (ref_forward.py precedent).
- Any candidate "fix" that changes committed goldens → goldens are the
  regression net; a golden changes ONLY when its generator's contract
  changed on purpose, via regen_goldens.sh (now failure-safe), with
  the diff explained in the commit message.

**Gate integrity (the temptations — each of these is the escalation
trigger (d), not a judgment call):**
- Raising ε, the waiver budget, or `max-abs-diff` thresholds.
- Swapping/trimming corpus prompts outside the written reserve
  protocol; re-rolling oracle runs until agreement.
- Skipping asan/tsan legs "just this commit"; committing suppressions.
- Narrowing a sweep/fuzz corpus to make it green (R2 precedent: the
  mismatch IS the divergence the gate exists to catch).
- Re-anchoring a bar to a friendlier baseline number without re-running
  the baseline at its current version, dated, on the checklist box.

---

## 12. The never-list (compact, absolute)

Never push / add a remote. Never attribution trailers. Never commit a
model file or cite a git-ignored path from committed `src/ tests/
scripts/ Makefile`. Never pop old stashes (drop `stash@{0}` at A8).
Never let the ledger lag a commit. Never dispatch without a recovery
clause. Never gate on judgment where the checklist says evidence. Never
loosen a gate without Kabir's written say-so via memo. Never leave a
`llama-server.exe` running. Never trust this playbook over a probe.

---

*End of playbook. First action for the next Opus session: §0 frame,
then §2 SESSION A. The ledger's NEXT-SESSION block points here.*
