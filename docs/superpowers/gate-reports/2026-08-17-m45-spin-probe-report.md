# M4.5 -- spin-before-block probe

3 passes, config order rotating, 6 threads, n=64, N=1. `STOKE_POOL_SPIN` sets the pool's spin budget; 0 is the pre-2026-08-17 pure-condvar wait.

| config | median tok/s | min | max | mean CPU-s |
|---|---|---|---|---|
| unpinned spin=0 | 16.318 | 16.098 | 16.33 | 27 |
| unpinned spin=2000 | 16.69 | 16.517 | 16.931 | 30.9 |
| unpinned spin=20000 | 16.743 | 16.103 | 17.365 | 32 |
| 0xFFF spin=2000 | 16.503 | 16.165 | 16.72 | 31 |
| llama unpinned | 18.82 | 18.66 | 19.02 | 0 |

**Reading it.** The `0xFFF spin=2000` row is the ceiling -- affinity already excludes the E-cores, so it shows what the machine can do. If `unpinned spin=2000` approaches that row while `unpinned spin=0` stays low, the wait shape is what was placing the pool on E-cores. If spin=0 and spin=2000 agree, the hypothesis is dead and the 6.6x has some other cause.

CPU-seconds is the second signal, and an independent one: the same work costs MORE processor-time on a slower core, so a sick run reads high there (74s vs 26s when this was first measured) while also finishing later.

Raw log: `2026-08-17-m45-spin-probe-raw.txt`.
