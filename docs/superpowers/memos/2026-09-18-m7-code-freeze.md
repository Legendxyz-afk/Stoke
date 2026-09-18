# 2026-09-18 — M7 CODE FREEZE (SEQUENCE 4.1): `src/` is frozen at `b1ee6b3` until tag `m7`

**Frozen at:** source commit **`b1ee6b3`** (`server: listen backlog 16 ->
MAX_CONNS`), which is the last commit touching `src/`; HEAD at the freeze is
`2e24f55` plus this memo (docs only above `b1ee6b3`). Same wording as the M3
freeze (Fable 08-05 ruling, kept through M6): **`src/` and every shipped
binary built from it are FROZEN until tag `m7`. The sole exception is a
BUG fix-cycle — a defect found by the soak, the fuzz or a parity gate — and
a fix restarts B7 from zero, because the soak must exercise the code that
ships.** Docs, `bench/` and `tests/tooling/` may move; nothing under `src/`
may, and the Makefile's flags may not.

## What is frozen, by hash

| binary | sha256 (first 16) | built from | archived as |
|---|---|---|---|
| `build/win/stoke.exe` (the gate binary, the @32 block's binary) | `fbc64f8c188861f2` | `b1ee6b3`, MinGW gcc `-O3`, 2026-09-18 11:45 | `build/stoke-m7-gate-fbc64f8c.exe` |
| `build/linux/stoke` | `c922098fa48000bb` | `b1ee6b3`, WSL gcc, matrix plain leg 13:31 | `build/stoke-m7-gate-linux-c922098f` |
| `build/linux-asan/stoke` (**the soak's server**) | `337bbc1b501e111f` | `b1ee6b3`, `SAN=asan`, matrix asan leg 13:51 | `build/stoke-m7-soak-linux-asan-337bbc1b` |
| `build/linux-asan/serve_fuzz` (the soak's driver) | `ab4299ca0dbd2303` | same leg | `build/stoke-m7-soak-linux-asan-serve_fuzz-ab4299ca` |

`build/` is gitignored, as for every archived gate binary since M2; the
hashes above are the record, and `run_m7_soak.sh` now prints the hash of the
binary it runs so the soak log ties itself to this table.

## What the frozen source has passed

- Windows suite on `fbc64f8c`: `run_tests` 19/19 binaries (57/57), the four
  tokenizer/info goldens, `serve_golden` 25/25 (11:5x, before block 2).
- WSL matrix on `b1ee6b3` (`build/wslmatrix/wsl_matrix_b1ee6b3.log`,
  13:25–13:53): plain rc=0 (27/27 ELF), ASan+UBSan rc=0 (27/27 ELF, every
  model-backed gate OK, zero sanitizer reports), TSan rc=0 (27/27 ELF; the
  model-backed gates SKIP on TSan by design — measured: `abi_smoke` peaks at
  7.36 GB RSS under TSan on a 7.7 GB VM).
- The @32 block of the clause-2 sitting ran on `fbc64f8c` (3 stoke units,
  32/32 clients each, 1.281× vs llama-server), and blocks 1/3/4 on `9eca59e`,
  which differs from `b1ee6b3` only in the listen backlog — a constant that
  cannot act below 17 simultaneous connections.
- `burst_smoke.py`: 30/30 connections answered during a prefill (16/30 on the
  binary before the fix).

## What the freeze does NOT claim

- Gate [3] (the 24 h soak, rung B7) is not run yet; this freeze exists so
  that it can be. Its rig (`tests/tooling/run_m7_soak.sh`, `28761aa` + the
  hash line) has been shaken down only as far as the witness mechanics
  (root, `oom_score_adj -1000`, a `/dev/kmsg` line reaching `/mnt/m`); the
  6-minute `--shakedown` is next and needs Kabir's go, since it is a server.
- Two clause-2 misses are documented, not fixed, and are not bugs under the
  exception above: ≥2× ollama-default (1.735× / 1.836×) and gate [3]'s 250 ms
  stall (Kabir's 09-16 disposition). Neither reopens the freeze.
- The one-accept-per-turn loop in `server.c` is a known follow-up (a burst's
  last client is accepted ~N turns in); it is not a defect under the
  exception either, since it costs no throughput and refuses no one now.

## Two last `src` changes this week, so nobody wonders

`9eca59e` (`MAX_CONNS` 24 → 48, static-asserted against `NET_WAIT_MAX`) and
`b1ee6b3` (listen backlog 16 → `MAX_CONNS`): both Kabir's call, both proved
red first, both suite- and matrix-green before any unit ran on them. The
second was found by the first stoke @32 unit; the pre-flight that found the
first had read the connection table and not the listen queue.
