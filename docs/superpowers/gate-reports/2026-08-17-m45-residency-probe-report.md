# M4.5 -- residency probe: is the collapse a disk?

6 identical stoke runs, n=128, 6 threads, back to back. `Page Reads/sec` sampled at 4 Hz across each run -- the same counter the ruling specifies for D's residency gate.

| run | tok/s | wall s | page reads/s avg | max | standby GB before -> after | avail GB before -> after |
|---|---|---|---|---|---|---|
| 1 | 2.161 | 62.4 |  |  | 3.09 -> 3.12 | 6.07 -> 6.07 |
| 2 | 8.675 | 17.1 |  |  | 3.12 -> 3.12 | 6.05 -> 6.05 |
| 3 | 5.958 | 24.2 |  |  | 3.17 -> 3.16 | 6.1 -> 6.07 |
| 4 | 2.234 | 59.4 |  |  | 3.15 -> 3.12 | 6.04 -> 5.89 |
| 5 | 2.453 | 54.4 |  |  | 3.17 -> 3.18 | 6 -> 5.85 |
| 6 | 2.215 | 60.4 |  |  | 3.18 -> 3.18 | 5.85 -> 5.79 |

**Reading it.** A slow run with page reads near zero is NOT a residency problem and the disk hypothesis dies. A slow run with sustained page reads in the thousands is the model being streamed off storage every token, and the fix is residency, not kernels. The ruling's own threshold for 'resident' is < 5 sustained.

Raw log: `2026-08-17-m45-residency-probe-raw.txt`.
