# M2 parity gate -- runbook (`tests/tooling/parity_gate.py`)

Implements
`docs/superpowers/design-notes/2026-07-19-m2-op-spec-and-parity-protocol.md`
§8 (the parity gate + waiver protocol) against the SETTLED oracle
invocation (default KV, no `-ctk/-ctv`), and the near-tie policy of
`docs/superpowers/memos/2026-07-20-m2-oracle-near-tie-policy.md`.

This file is the manager's runbook: the exact commands and what each
output means. The harness itself (argparse `--help`, docstrings) is the
detailed reference; this file is the fast path.

**ORDER OF OPERATIONS FOR THE M2 CLOSE-OUT** (the sections below are
numbered by increasing commitment, NOT by run order — read this first):
Step 0 (selftest + dry-run) → a live smoke on 2 prompts at small `-n`
→ **Step 3, the `--survey` census** → classify the divergences and append
the count to the near-tie memo → **Kabir's oracle near-tie policy call,
if any non-waivable near-tie appears** → only then **Step 2, the recorded
`--clean` gate run** → gate report → the M2 gate checklist (session
playbook §5). The survey comes BEFORE the recorded run because its census
is what the policy call is made on, and the policy call is what the
recorded run's validity depends on. Step 1 (cached dev mode) is the
optional iteration loop, not part of the close-out path.

## Prerequisites (checked automatically; listed here so a failure is fast to diagnose)

- `build/win/stoke.exe` built and present.
- `models/Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf` present (4,280,405,216 bytes).
- `bench/vendor/llama-b10037/llama-server.exe` present (DLLs beside it).
- Nothing already listening on `--port` (default 8090) -- the harness
  FATALS rather than silently picking a different port.
- Plain Python 3, stdlib only (no pip packages to install).

## Step 0 -- before running anything live: selftest + dry-run

```
python tests/tooling/parity_gate.py --selftest
python tests/tooling/parity_gate.py --dry-run
```

Both are SERVER-FREE (no process spawned, no port touched, no cache
written). `--selftest` proves the aligner and the waiver-decision logic
against committed fixtures and synthetic data; `--dry-run` prints the
exact plan (server argv, resolved prompt list, stoke command templates)
for a plumbing sanity check. Run these first every time before a live
invocation below.

## Step 1 -- dev-mode measurement (iterative, cached)

```
python tests/tooling/parity_gate.py
```

- Uses the dev cache under `tests/tooling/.gate_cache/` (gitignored) so
  repeated runs during development skip already-computed oracle/stoke
  results for prompts that haven't changed.
- Same STRICT semantics as the recorded run: stops at the first
  non-waivable divergence, budget ≤2 waivers, reserve consumed in order.
  This is a cheaper, resumable rehearsal of the recorded run, not a
  looser mode.
- Iterate on a subset with `--only`:
  ```
  python tests/tooling/parity_gate.py --only p07,p12,p23
  ```

## Step 2 -- THE recorded gate run (official M2 gate result)

```
python tests/tooling/parity_gate.py --clean --report docs/superpowers/gate-reports/<name>.md
```

- `--clean` ignores and refreshes the dev cache. **This is the only
  invocation whose result should ever be treated as the recorded M2 gate
  outcome** -- per the brief, the manager runs this, verifies it, and
  places/commits the final report under `docs/superpowers/gate-reports/`.
- Exit 0 + `GATE PASS` on the terminal means all 50 prompts are
  token-exact or resolved via a recorded, in-budget waiver. Exit 1 +
  `GATE FAIL` (both to stderr and the report) means a non-waivable
  divergence or budget exhaustion -- full evidence (prompt id, position,
  both top-5 lists, both gaps) is in the report; see design note §9 for
  the debug ladder.

## Step 3 -- the divergence-class census (`--survey`) -- DIAGNOSTIC ONLY

```
python tests/tooling/parity_gate.py --survey --clean
```

**This produces no gate result of any kind.** It exists to answer the
near-tie memo's open question (how many of the 50 prompts actually hit
the oracle's numeric-class near-tie behavior) without paying for a
re-driven multi-hour run per divergence found:

- Runs every selected prompt to completion. Never stops early. Never
  enforces the §8 waiver budget. Never consumes a reserve prompt --
  `WaiverLedger` is not used at all in this code path.
- For every divergence, records full evidence: prompt id, first
  divergence position, BOTH top-5 lists, both gaps, and the verdict
  class the §8 waiver test WOULD have assigned:
  `waivable` / `non_waivable_gap` / `non_waivable_pair` / (if the
  evidence itself couldn't be gathered, e.g. stoke stopped early)
  `evidence_unavailable`.
- The report (`tests/tooling/parity_survey_dev_report.md` by default) is
  headed `DIAGNOSTIC ONLY, NOT A GATE RESULT` in the title itself and
  again in bold on the next line, and never contains PASS/FAIL
  vocabulary anywhere. The terminal banner is the same:
  `MODE: SURVEY -- DIAGNOSTIC ONLY, NOT A GATE RESULT`.
- Per-prompt fetch/evidence failures are caught and recorded as an
  `ERROR`/`evidence_unavailable` row rather than aborting the whole
  multi-hour survey -- unlike the strict gate, which fails loud
  immediately (correct for a gate authority, wrong for an unattended
  census).
- `--survey --clean` is the invocation to actually run (a fresh,
  uncached census). `--survey` combined with `--selftest` is rejected
  (fixture-only mode, unrelated to the corpus/oracle). There is no other
  CLI flag today whose presence specifically means "this is the recorded
  run" to guard against -- the separation is structural instead: survey
  is dispatched to its own function, writes its own report via a
  wholly separate writer, and never calls the gate's. See
  `.superpowers/sdd/m2-w4-report.md` for the full note on this.

## Operational constants (measured live by the manager, 2026-07-27 -- read before a full run)

- **`--jobs` is a RAM ceiling, not a core count.** Each `stoke.exe`
  holds ~1,181 MB PRIVATE (the `[36][4096][8][128]` f32 K+V cache is
  allocated eagerly at `model_open`, regardless of the prompt's actual
  length). At `--jobs 6`: 6 × 1,181 MB (~7.1 GB) + the 4.3 GB shared
  model mmap + the oracle server + the OS is ≈ 14.9 GB, which thrashes a
  16 GB box. **Use `--jobs 4` (the default) on a 16 GB machine** --
  raising it because CPU cores are idle will cost you swapping, not
  speed.
- **Run-cost math.** Measured ~5.3s per forward pass; corpus mean 77.8 /
  max 490 prompt tokens. A 50-prompt × 64-token run is ≈7,038 forward
  passes ≈ 9-11.5h serial, ≈3-4h wall at `--jobs 4`. Budget accordingly
  -- this is not a quick command.
- **Prompts are scheduled longest-first automatically, in both gate and
  survey modes** -- by corpus file byte size (a token-count proxy; see
  `_order_longest_first()` in `parity_gate.py`). The long-context tier
  (`p40`-`p44`, `r07`: 447-490 tokens, ~553 forward passes, ~49 min each
  at the measured per-pass cost) is submitted to the `--jobs` pool
  FIRST, so it starts immediately across multiple workers instead of
  landing in the tail after ~45 short prompts drain and stranding the
  run's makespan on stragglers with idle workers. This is automatic --
  no flag needed, nothing to remember to pass.
- **Cold start is slow.** The health-poll timeout defaults to 300s
  (measured cold start: 69s to `/health` reporting healthy, 4.3GB Q8_0
  model, `-t 1`) -- do not expect an instant response, and do not lower
  this timeout.
- **Build identity comes from `--version`, not `/props`.** Measured live:
  `/props` carries `model_path`/`n_ctx`/`total_slots`/`chat_template`
  etc but no build/commit field. The report header's build identity is
  parsed from `llama-server.exe --version` (stderr:
  `version: <build> (<commit>)`), which is instant and does not load the
  model.

## Output reference

- **Gate mode**: per-prompt `PASS`/`WAIVE`/`FAIL`; every waiver's full
  evidence (prompt id, position, both gaps, both top-2 pairs, which
  reserve prompt replaced it); oracle build identity + exact server argv
  + KV setting + oracle exe sha256; `stoke.exe` path + sha256; corpus +
  reserve identity.
- **Survey mode**: per-prompt `EXACT`/`DIVERGED`/`ERROR`; the divergence
  breakdown by verdict class; same identity block as gate mode.
- **Dev cache** (`tests/tooling/.gate_cache/`, gitignored): one JSON
  file per (prompt file hash, `stoke.exe` hash, `n_predict`, invocation
  signature) -- safe to delete entirely at any time, it only ever
  affects dev-loop speed, never a gate decision (the recorded run always
  passes `--clean` and never reads it).

## If the harness reports a survived server

`parity_gate.py` kills its own `llama-server` on every exit path
(try/finally, `atexit`, and a `SIGINT`/`SIGTERM`/`SIGBREAK` handler) and
verifies it's dead before exiting. If it ever prints
`oracle server ... survived cleanup`, the message includes the exact
command to finish the job by hand:

```
taskkill /PID <pid> /F
```
