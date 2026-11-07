# Lorren: as many chats at once as memory allows (2 to 8) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** v0.1 runs the engine with as many parallel replies as the PC's memory allows, from 2 to 8, chosen at each engine start.
That makes the website's "Many chats. One PC. All at once." true for v0.1.

**Architecture:** A pure function, `gguf::choose_seqs`, picks the count from the model's memory to run and the PC's memory. The
supervisor's memory check calls it before every start and restart, passes the count as `--seqs`, and puts it in the engine status.
The library gives each model's memory to run at 2 and at 8 replies. The Models tab shows both, and the status bar shows the running
engine's count.

**Tech Stack:** Rust (Tauri 2, tokio, ts-rs), React 18 + vitest.

**Spec:** `docs/superpowers/specs/2026-09-25-lorren-desktop-design.md` §3.1 (Task 3 amends it), and ghost's call of 2026-11-07:
option 1 of `company/docs/superpowers/audits/2026-11-07-site-claims-vs-v0.1.md` §"Ghost's call: two chats at once".

## Global Constraints

- Ghost's call (2026-11-07), verbatim from the audit: "Raise the app's parallel replies for v0.1. [...] the app would choose as many
  replies as the PC's memory allows, between 2 and 8."
- The rule. At each engine start, n is the largest count from 2 to 8 that meets two conditions:
  - The memory to run with n replies (`gguf::run_bytes`), plus the 1 GB spare (`min_free_bytes`), fits in the available physical
    memory.
  - For any n above 2, the memory to run is at most half of the PC's total physical memory.

  If 2 does not fit the available memory, the load is refused as it is today (`NotEnoughMemory`, with `need_mb` taken from 2
  replies). The half-of-total cap never refuses 2.
- The engine's other flags do not change: `--ctx 4096 --kv q8`, with threads unset (the engine then picks its default of 12 when
  `--seqs` is above 1). The engine (`stoke` `src/`) is frozen; nothing changes there.
- v0.1 has no setting for the count. The spec's Settings "Engine" section is not built, and this plan adds none.
- Gate before every cargo, vitest, build or e2e run:
  - none of `WardogsClient-Win64-Shipping.exe`, `sp26-cod.exe` or `codCrashHandler.exe` is running;
  - at least 4 GB of memory is free;
  - `CARGO_BUILD_JOBS=4`.
- Never start the real engine (`lorren-engine.exe`, `stoke serve`) or any model server. The tests use the fake engine
  (`--features fake-engine`).
- Commit messages follow `git log` (`desktop: ...`) and carry no Co-Authored-By or "Generated with" trailer. Do not push.

## Review Focus

1. A hostile or huge GGUF header (KV bytes `u64::MAX`, a file of `1 << 60`). No count fits, the load is refused, and nothing wraps
   into "fits". The `tests/supervisor.rs` tests at lines ~220 and ~238 keep passing, and `choose_seqs` gets a `u64::MAX` case.
2. A PC with little memory. Only the available memory can refuse 2, never the cap. This is `choose_seqs` case "the floor".
3. The edge. With available memory exactly the need for 8 plus the spare, the count is 8; with one byte less, it is 7.
4. Before any start, the status gives no count (0), and the status bar says nothing about parallel replies.
5. A restart chooses again, because it goes through the same `start_locked` and memory check. Code review covers this; there is no
   separate test.

---

### Task 1: The engine runs as many replies as memory allows

**Files:**
- Modify: `desktop/src-tauri/src/gguf.rs`: replace `ENGINE_SEQS` with `ENGINE_SEQS_MIN = 2` and `ENGINE_SEQS_MAX = 8`, and add
  `choose_seqs`.
- Modify: `desktop/src-tauri/src/engine/supervisor.rs`:
  - rename `EngineConfig.seqs` to `max_seqs` (default `ENGINE_SEQS_MAX`) and add `EngineConfig.memory`;
  - make `memory_check` return the count, and give `args` a `seqs` parameter;
  - set the status's `seqs` at each start, and leave it at 0 before the first.
- Modify: `desktop/src-tauri/src/library.rs`: compute `run_bytes` at `ENGINE_SEQS_MIN`, and add `run_bytes_max` at
  `ENGINE_SEQS_MAX`.
- Modify: every other `ModelInfo` literal the compiler names (`commands.rs` ~445, `library.rs` ~300).
- Regenerate: `desktop/src/ipc/bindings/ModelInfo.ts`, and `EngineStatus.ts` if its doc comment changes. ts-rs writes them when
  `cargo test` runs (`src-tauri/.cargo/config.toml`).
- Test: the `gguf.rs` and `library.rs` unit tests, and `desktop/src-tauri/tests/supervisor.rs`.

**Interfaces:**
- Produces, in `gguf.rs`: `pub fn choose_seqs(file_bytes: u64, kv_bytes_per_token: u64, ctx: u32, max_seqs: u32, available: u64,
  total: u64, spare: u64) -> Option<u32>`. It returns the rule's n, or `None` when 2 does not fit. A `max_seqs` below 2 is read as 2.
  Its arithmetic saturates, like `run_bytes`.
- Produces, in `engine/supervisor.rs`:
  - `#[derive(Debug, Clone, Copy)] pub struct MemoryFigures { pub available: u64, pub total: u64 }`.
  - `EngineConfig.memory: Option<MemoryFigures>`. `None` reads the PC's own figures (`memory::available_physical_bytes`,
    `memory::total_physical_bytes`); the tests set it.
  - `EngineConfig.max_seqs: u32`. With `check_memory` false, the engine runs `max_seqs`.
  - `EngineStatus.seqs`: the count the running (or last) engine was started with; 0 before the first start.
- Produces `ModelInfo.run_bytes_max: u64` with `#[ts(type = "number")]`. `run_bytes` keeps its name and becomes the figure at 2.
- Task 2 consumes `EngineStatus.seqs` (0 means none yet), `ModelInfo.run_bytes` and `ModelInfo.run_bytes_max`.

- [ ] **Step 1: Write the failing unit tests** in `gguf.rs`'s tests.
  - The fixture is Qwen3-4B: `file_bytes = 2_497_280_736`, `kv = 78_336`, `ctx = 4096`, `max = 8`, `spare = 1 << 30`. One GiB is
    `1 << 30`.
  - With those figures, `run_bytes(n)` is `2_765_716_192 + n * 320_864_256`: 3_407_444_704 at 2 and 5_332_630_240 at 8.

  | test name | available | total | other | expect |
  |---|---|---|---|---|
  | `plenty_of_memory_runs_eight` | 8 GiB | 16 GiB | | `Some(8)` |
  | `the_available_memory_limits_the_count` | 5 GiB | 16 GiB | | `Some(4)` |
  | `half_the_total_caps_the_count` | 6 GiB | 7 GiB | | `Some(3)` |
  | `the_cap_never_refuses_the_floor` | 4_831_838_208 | 6 GiB | | `Some(2)` (`run_bytes(2)` is above the 3 GiB cap) |
  | `no_room_for_two_refuses` | 4 GiB | 16 GiB | | `None` |
  | `the_edge_is_exact` | 6_406_372_064, then 6_406_372_063 | 16 GiB | | `Some(8)`, then `Some(7)` |
  | `a_hostile_header_fits_nowhere` | 16 GiB | 16 GiB | kv `u64::MAX` | `None` |
  | `the_ceiling_is_kept` | 16 GiB | 64 GiB | `max = 4`, then `max = 0` | `Some(4)`, then `Some(2)` |

  One more test, `the_counts_are_two_and_eight`, asserts `ENGINE_SEQS_MIN == 2 && ENGINE_SEQS_MAX == 8`. Its comment says that the
  Models tab's memory lines name these two numbers (`desktop/src/tabs/ModelsTab.tsx`).
- [ ] **Step 2:** run `cargo test --lib gguf` (gate first). Expected: a compile failure (`choose_seqs` and the constants do not
  exist).
- [ ] **Step 3:** implement the constants (remove `ENGINE_SEQS`) and `choose_seqs`. Its doc comment states the rule from Global
  Constraints. Scan down from the ceiling: `run_bytes` grows with n, so the first n that fits is the largest.
- [ ] **Step 4:** run `cargo test --lib gguf`. Expected: PASS.
- [ ] **Step 5: Write the failing supervisor tests** in `tests/supervisor.rs`.
  - `it_starts_the_engine_and_reports_serving`: expect `(4096, 8, "q8")`. The helper `cfg` turns `check_memory` off, so the engine
    runs the ceiling.
  - `a_model_path_with_spaces_arrives_as_one_argument`: expect `--seqs 8`.
  - New `the_engine_runs_as_many_replies_as_the_memory_allows`:
    - setup: `cfg` with `FAKE_ENGINE_EVENTS`, then `check_memory = true` and `memory = Some(MemoryFigures { available: 5 << 30,
      total: 16 << 30 })`; the model has `file_bytes: 2_497_280_736, kv_bytes_per_token: 78_336`;
    - expect: argv has `--seqs 4`, and `status().seqs == 4`.
  - New `a_pc_without_room_for_two_replies_refuses_the_load`:
    - setup: `memory = Some(MemoryFigures { available: 4 << 30, total: 16 << 30 })` and the same model;
    - expect: `ensure_model` returns `Err(EngineError::NotEnoughMemory { need_mb: 3249, free_mb: 4096, .. })` (3_407_444_704 >> 20
      is 3249), with `status().seqs == 0` and state `Failed`.
  - New `before_the_first_start_the_status_gives_no_count`: `Supervisor::new(cfg(&[]), ...).status().seqs == 0`.
- [ ] **Step 6:** run `cargo test --features fake-engine --test supervisor`. Expected: a compile failure, or the new tests FAIL.
- [ ] **Step 7: Implement the supervisor wiring.**
  - The count comes from `memory_check`, which returns `Result<u32, EngineError>`.
  - `start_locked` sets `s.seqs` in the same status update that sets Starting/Restarting, and passes the count to `args`.
  - The initial status has `seqs: 0`. The field's doc comment says what Interfaces says.
  - Each start logs one line, as `self.monitor.log("app", "info", ...)`:
    - with the check: `parallel replies: {n} (up to {max}; {available_mb} MB of {total_mb} MB free)`;
    - without it: `parallel replies: {n} (memory not checked)`.

  Leave the existing `starting the engine with {path}` line unchanged; the e2e may read it.
- [ ] **Step 8:** in `library.rs`, set `run_bytes` with `ENGINE_SEQS_MIN` and `run_bytes_max` with `ENGINE_SEQS_MAX`, and update
  both doc comments ("at the fewest replies at once, 2" / "at the most, 8").
  - Under the existing `nexus.run_bytes` assertion, add `assert_eq!(nexus.run_bytes_max, nexus.size_bytes + 1_996_488_704 +
    268_435_456);`, with the comment `60,928 B/token x 4,096 x 8`.
  - Update the old comment's "x 2" line so that it still reads correctly.
- [ ] **Step 9:** run the whole Rust suite, gated: `cargo test` and then `cargo test --features fake-engine`. Expected: all pass. Then
  check that `git status` shows `desktop/src/ipc/bindings/ModelInfo.ts` regenerated with `run_bytes_max: number`.
- [ ] **Step 10: Commit**
  `desktop: the engine runs as many replies at once as the PC's memory allows, 2 to 8, chosen at each start (ghost's call 11-07)`.

### Task 2: The status bar and the Models tab say how many

**Files:** `desktop/src/components/StatusBar.tsx`, `desktop/src/tabs/ModelsTab.tsx`, `desktop/src/__tests__/testApi.ts`,
`desktop/src/__tests__/tabs.test.tsx` (~line 40), `desktop/src/__tests__/App.test.tsx` (the StatusBar tests, ~line 200)

**Interfaces:** consumes Task 1's `EngineStatus.seqs`, `ModelInfo.run_bytes` and `ModelInfo.run_bytes_max`.

- [ ] **Step 1: Write the failing tests.**
  - In `testApi.ts`, the Qwen3-4B model gets `run_bytes_max: 5_332_630_240` and the Llama model (`run_bytes: 2_786_935_328`) gets
    `run_bytes_max: 4_284_301_856`.
  - In `tabs.test.tsx`, the memory test is renamed "shows the memory a model needs to run with 2 and with 8 chats at once" and
    expects `~3.2 GB` and `~5.0 GB`. Its comment gives the arithmetic: 5,332,630,240 = 2,497,280,736 + 2,566,914,048 (78,336 x 4,096
    x 8) + 268,435,456, and / 1,073,741,824 = 4.966.
  - In `App.test.tsx`:
    - with the engine status at `seqs: 8`, the status bar shows `/8 parallel/`;
    - with `seqs: 0`, `queryByText(/parallel/)` is null.

    Feed the status the way the file's existing StatusBar tests do.
- [ ] **Step 2:** run `npx vitest run src/__tests__/tabs.test.tsx src/__tests__/App.test.tsx` (gate first). Expected: the new
  assertions FAIL.
- [ ] **Step 3: Implement.**
  - `StatusBar.tsx` shows `engine frozen b1ee6b3 · ctx {ctx}`, followed by ` · {seqs} parallel` only when `seqs > 0`.
  - `ModelsTab.tsx` replaces the one memory fact with two, both `—` when `run_bytes` is 0:
    - key `memory to run, 2 chats at once (model + KV cache + runtime)`, value `~{gb(run_bytes)}`;
    - key `memory to run, 8 chats at once`, value `~{gb(run_bytes_max)}`.
- [ ] **Step 4:** run `npx vitest run` (all pass) and `npm run build` (tsc + vite: no type errors).
- [ ] **Step 5: Commit** `desktop: the status bar names the replies the engine runs at once, and the Models tab the memory for 2
  and for 8`.

### Task 3 (the manager's): the spec, the site audit, the sweep and the install

- [ ] **Step 1:** amend the desktop spec, with each change dated "(changed 2026-11-07, ghost's call)":
  - §3.1 "Always explicit sizes" and "Memory estimate before loading" state the rule;
  - §3.2 "Background work" no longer says 2.

  In `docs/superpowers/specs/2026-11-04-lorren-distribution-first-design.md`, the line "(milestone 1 runs two at once)" gains "; v0.1
  runs as many as memory allows, 2 to 8".
- [ ] **Step 2:** in `company`, the audit's rows 2 and 3 become **from the Discover release**, because the first release with Discover
  carries this change. The audit's "Ghost's call" section records "Decided 2026-11-07: option 1", with the stoke-lorren commits. Note
  it in the step-3 ledger.
- [ ] **Step 3:** run the step-2 sweep, gated:
  - `cargo test` and `cargo test --features fake-engine`;
  - `npx vitest run`;
  - `npm run build`;
  - `npm run test:release` (70);
  - the e2e (25).
- [ ] **Step 4:** make the release build, then install it on ghost's PC the way step 2 did:
  - no Lorren running;
  - the installed exe backed up beside the install;
  - the sha256 of the new and the installed exe compared.

  No schema change, so no database copy test. Then update the memory.
