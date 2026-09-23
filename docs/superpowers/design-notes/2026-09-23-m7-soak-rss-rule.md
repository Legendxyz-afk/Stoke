# M7 gate [3] — the 24 h soak's RSS rule, pre-registered

*2026-09-23, Opus 5.5. Written and committed BEFORE the 24 h run. Kabir agreed
the rule the same day ("i agree with ur proposal"). Code:
`tests/tooling/soak_rss_rule.sh` (the grader), `tests/tooling/test_soak_rss_rule.sh`
(its test), `tests/tooling/run_m7_soak.sh` (the wiring). `src/` is untouched and
the freeze stands (the freeze memo lets `tests/tooling/` move).*

## 1. The gap

ROADMAP §M7 asks for a "24h detached soak with RSS slope watch", and the M7
stamp must be a mechanical checklist. But in soak mode `run_m7_soak.sh`
**reported** RSS and graded nothing:

> ⚠ RSS is REPORTED, not gated … The first full run establishes the baseline;
> the SECOND can gate it.

The first full run was the 2026-09-18 1 h baseline, and nobody wrote the rule
after it. The 24 h run's printed verdict would have said nothing about memory,
and any reading of its series made afterwards would have been a band fitted to
the result. This note closes that before the run.

## 2. What the 1 h baseline actually shows

It was banked as "non-arena anon flat at 131 → 158 → 162 → 167 MiB" and "VmData
+113 MiB decelerating 80/24/5/4 per chunk = allocator warm-up". Re-read at 240 s
resolution (`gate-reports/2026-09-18-0844-m7-soak.rss.csv`; non-arena =
`anon_mb − arena_mb`; VmData relative to t = 0):

| t (s) | gens | VmData + | non-arena anon |
|---:|---:|---:|---:|
| 287 | 48 | 76 | 126 |
| 1200 | 236 | 81 | 133 |
| 1443 | 285 | 102 | 156 |
| 2415 | 486 | 108 | 161 |
| 3455 | 719 | 113 | 167 |

- a **warm-up** of +76 MiB in the first 287 s;
- **one +21 MiB step** between t = 1200 and 1443 s, in both quantities;
- and otherwise a **steady ~18–20 MiB/h in both**, from minute 5 to the end
  (t = 287 → 1200: +5 MiB; t = 1443 → 3455: +11 MiB over 434 generations,
  ~26 KiB per generation). The last two chunks' +5 and +4 are the same rate,
  not the tail of a deceleration.

The "decelerating" reading came from folding the warm-up and the step into the
first two 15-min chunks. The series is **not flat**, and one hour cannot say
whether it stops.

The best-fit cause is **ASan's quarantine**. The 09-14 RSS probe ran the
**plain** build (`build/linux`, ctx 256). Its non-arena set was 37 MiB in every
chunk, with VmData 378 → 379 over 500 generations. The soak runs the **ASan**
build, and ASan holds every freed chunk back from reuse until its quarantine
fills. On this toolchain the quarantine defaults to 256 MiB: libasan prints
`quarantine_size_mb=256M` under `verbosity=1` when the flag is unset. That
explanation is **unproven**. The two runs also differ in ctx (256 vs 1024), so
the probe does not control for it.

## 3. The rule

Over the **final 6 h** of the series, fit the least-squares growth (slope × 6 h)
of

- **(a) VmData**: mapped memory. This is the half RssAnon cannot see: a leak
  that allocates without touching, or pages the kernel swapped out.
- **(b) RssAnon − arena**: resident anonymous memory outside the KV arena. The
  arena's own residency comes from pagemap (the sidecar's `arena_mb` column).

**FLAT** if both are ≤ one KV block (4,718,592 B = 4.50 MiB for the soak's
Qwen3-4B: 2 × 36 layers × 16 tokens × 8 kv-heads × 128 × 4 B). **CLIMBING**
if either is larger. **VOID** if the reading does not exist. Only FLAT passes.

**It grades only the gate run:** soak mode with ≥ 24 h. A shorter soak prints
the same reading, marked "REPORTED not graded", because the rule asks whether
growth has *stopped by hour 18*, which a 1 h run cannot answer.

**VOID** is printed when:
- there are fewer than 6 usable rows;
- the series ends more than 10 min before the run does (the sidecar or the
  server died, so the series' last 6 h are not the run's);
- the series is shorter than the window;
- the window holds fewer rows than half the sidecar's 5 s cadence;
- the header is not the sidecar's nine columns.

**Rows dropped rather than fitted:**
- **Malformed rows.** A sidecar killed mid-`printf` leaves a partial last line
  like `864`, which would otherwise "end" the series at t = 864 s.
- **Rows inside the window with arena 0 and swapped 0.** Two things write such
  a row:
  - `arena_rss` failing, which prints `0 0`. Fitted, it moves the whole arena
    into "non-arena" and a flat run CLIMBS.
  - The race at the sidecar's last `kill -0`, where the server is already gone
    and every column is 0. Fitted, it drags VmData ~20,000 MiB negative and a
    leak reads FLAT.

Both drops are counted and printed in the verdict line.

### Why these constants

- **The final window, not the whole run.** A bounded fill and a leak look
  identical while the fill lasts. Only a leak is still climbing at hour 18, so
  the rule asks the one question that separates them. A fill that has not
  finished by the window fails. That is the conservative side: "not shown
  bounded" is not "bounded".
- **6 h.** The window is the run's last quarter. It needs to be long enough
  that a 0.75 MiB/h leak crosses one block, and short enough that a bounded
  fill has finished before it opens (§4).
- **One block.** This is the unit the RSS probe's rule already uses, and it is
  derived from the model header, not from any result. The CSV holds integer
  MiB, so the estimate carries about ±1 MiB of quantization. A synthetic
  1 MiB/h ramp reads +5.8 MiB rather than +6.0.
- **Least squares rather than endpoints.** The fit uses all ~4,300 rows, so
  per-sample jitter cancels: ±3 MiB of noise reads −0.1. A permanent step
  inside the window reads as ~1.5× its height at mid-window, so it fails as
  growth should.
- **What it cannot see:**
  - a leak slower than one block per 6 h (~0.75 MiB/h);
  - anything below the quantization.
  - The ASan build is not the shipping build. The plain build's memory
    behaviour is what the probe measured, and that read flat.

## 4. ASan's quarantine is now set, not defaulted: 64 MiB

At the baseline's rate, the default 256 MiB fills at about 14 h. That is close
to the window, which opens at 18 h. A slower day of traffic would stretch the
fill into the window and fail a bounded fill as a leak. So `run_m7_soak.sh` now
passes `quarantine_size_mb=64`.

- **Fill time.** 64 MiB fills in ~3.5 h at the baseline's rate, and still
  before 18 h at down to ~20% of that rate.
- **What it still catches.** It still holds several hours of this server's
  frees, so a use-after-free has to reach back further than that to escape.
- **Verified.** libasan prints `quarantine_size_mb=64M` when the flag is set.
  A misspelled control prints `found 1 unrecognized flag(s)` and `256M`.

This also makes the run **diagnostic**. If the last 6 h still climb at
~18 MiB/h with a 64 MiB quarantine, the quarantine was not the cause, and the
verdict is a real finding rather than an argument.

## 5. Direction check (Opus decides and records; Kabir overrules)

- **Harder or easier?** Harder. Before this, RSS was not graded at all. Now the
  gate fails on CLIMBING and on VOID.
- **Fitted to a result?** No. The 24 h data does not exist yet. The threshold
  is the model's block size, and the window follows from the fill-time
  argument. The baseline set only two things: the *need* for a rule, and the
  quarantine setting.
- **The cost of being wrong.** A false CLIMBING costs another day of box time.
  A false FLAT would pass a leak into Phase 1's close. The rule leans toward
  the first.

## 6. How it was verified

- **Test.** `tests/tooling/test_soak_rss_rule.sh` passes 17/17. It passes under
  Git Bash (bash + gawk), under WSL (dash + gawk), and under WSL dash with `awk`
  resolved to mawk 1.3.4 (stock Ubuntu's default awk, so a plain cloud VM runs
  the same grader). The real 1 h baseline reads VOID ("the series spans
  3473 s").
- **RED first.** Before the grader existed, all 16 cases then in the test
  failed (`No such file`).
- **Mutation check.** Eight single-guard mutants of the grader were run, and
  every one turned the test red:

  | mutant | caught by |
  |---|---|
  | VmData branch removed | `mapped_leak`, `mapped_zero_tail` |
  | non-arena branch removed | `nonarena_leak` |
  | malformed-row check removed | `truncated_tail` |
  | arena-0 drop removed | `arena_readfail`, `mapped_zero_tail` |
  | sparse check removed | `sparse` |
  | early-end check removed | `died10` |
  | whole run fitted instead of the window | `fill14`, `step_early`, `step_late`, `sparse` |
  | header check removed | `badheader` |

- **A guard that was never load-bearing.** The first version also dropped
  rows with RSS or VmData equal to 0. Its mutant stayed **green**, because the
  arena-0 drop already removes every such row the sidecar can write. That guard
  was deleted rather than kept as an untested claim, and the truncated-line
  case was added for the malformed-row check.
- **Wiring.** The verdict block was extracted verbatim from `run_m7_soak.sh`
  and run under dash with stubs:
  - a 24 h flat run passes;
  - a 24 h run growing 2 MiB/h fails;
  - a 24 h run whose series stops early VOIDs and fails;
  - the real 1 h run and the shakedown print the reading "REPORTED not graded".
- **Syntax.** `dash -n` is clean on all three scripts.
- **Pre-flight.** The soak's own pre-flight now runs the test and ABORTS unless
  the printed line reads `test_soak_rss_rule: N/N OK`. The grader is graded in
  the VM that will use it, before the day starts.

## 7. After the run

- **The verdict line.** `=== M7 soak: OK (…, RSS FLAT) ===` is gate [3]
  evidence for the stamp. The full reading is on the `RSS rule (gate [3],
  final 6 h):` line.
- **CLIMBING.** The gate fails. Find the growth before a re-run. The first
  probe would be the plain build at ctx 1024, which separates "ASan" from
  "ctx".
- **VOID.** The reading does not exist. Re-run. Hyper-V event 18508 and the
  dmesg witness say why the series stopped.

The launcher is `build/wslmatrix/launch_soak24h.sh` (untracked, like its
siblings). It runs the 6-min shakedown first, because the rig has not served
since this change, and starts the 24 h only on `=== M7 soak: OK`.
