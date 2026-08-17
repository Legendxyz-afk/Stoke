# M4.5 -- affinity mask SHAPE probe

Does the shape of the affinity mask, not its size, decide stoke's rate? 3 reps, mask order rotating, 6 threads, n=128, N=1.

| mask | meaning | stoke | llama | stoke/llama | stoke spread |
|---|---|---|---|---|---|
| `0xFFF` | 12 siblings of 6 P-cores (standing mask) | 12.265 | 17.21 | 0.713x | 59.2% |
| `0x555` | one sibling per P-core | 9.53 | 17.09 | 0.558x | 70.6% |
| `0xFFFF` | all 16 logical (unpinned) | 3.151 | 17.73 | 0.178x | 766.2% |

If `0x555` matches or beats `0xFFF` while llama barely moves, then stoke's exposure is hyperthread STACKING under a mask that permits it, and the engine's missing thread affinity -- `src/pool.c` creates workers with bare `pthread_create` -- is a real defect rather than a measurement artefact.

Bank: `2026-08-17-m45-affinity-probe-bank.jsonl`. Raw log: `2026-08-17-m45-affinity-probe-raw.txt`.
