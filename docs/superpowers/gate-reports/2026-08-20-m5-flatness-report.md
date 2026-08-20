# Is decode throughput flat in N?

Run `2026-08-20-m5-flatness`. `stoke batch --timing` decode-only aggregate, 23-token prompt,
ctx 192, -n 128, threads 8, kv f32, 4 counterbalanced passes
(odd ascending, even descending) so monotonic thermal drift cancels between a
forward and a reverse pass instead of accumulating into the ratio.

| N | median tok/s | up-pass median | down-pass median | GFLOP/s | vs N=1 |
|---|---|---|---|---|---|
| 1 | 14.733 | 14.868 | 14.733 | 120 | 1.000x |
| 2 | 15.522 | 15.522 | 15.014 | 127 | 1.054x |
| 4 | 18.682 | 18.682 | 18.311 | 152 | 1.268x |
| 8 | 20.498 | 20.498 | 20.356 | 167 | 1.391x |
| 16 | 21.082 | 21.108 | 20.989 | 172 | 1.431x |

**Read the up and down columns against each other.** If they disagree by more
than the pass-to-pass spread, drift was still large inside a pass and the
counterbalancing is carrying more weight than it should -- shorten the cells
or let the box cool between passes rather than trusting the mean.

| unit | N | tok/s | wall s | cpu s | cores | clock % of nominal |
|---|---|---|---|---|---|---|
| p1-n1 | 1 | 15.258 | 10.6 | 58.2 | 5.49 | 98.5 |
| p1-n2 | 2 | 15.499 | 19.6 | 122.4 | 6.24 | 112.8 |
| p1-n4 | 4 | 18.617 | 32.7 | 209.9 | 6.42 | 101.3 |
| p1-n8 | 8 | 21.125 | 58.2 | 383.4 | 6.59 | 101.9 |
| p1-n16 | 16 | 21.82 | 113 | 760.2 | 6.73 | 96.9 |
| p2-n16 | 16 | 21.767 | 112.7 | 754.3 | 6.69 | 105.6 |
| p2-n8 | 8 | 21.193 | 57.8 | 383.9 | 6.64 | 99.6 |
| p2-n4 | 4 | 18.918 | 32.2 | 205.5 | 6.38 | 101 |
| p2-n2 | 2 | 15.626 | 19.7 | 122.5 | 6.22 | 97.2 |
| p2-n1 | 1 | 14.898 | 10.6 | 61.7 | 5.82 | 106 |
| p3-n1 | 1 | 14.479 | 10.8 | 63.5 | 5.88 | 98.4 |
| p3-n2 | 2 | 15.544 | 19.6 | 123.8 | 6.32 | 98.2 |
| p3-n4 | 4 | 18.748 | 32.7 | 209.6 | 6.41 | 111.7 |
| p3-n8 | 8 | 19.872 | 61.9 | 403.4 | 6.52 | 101.8 |
| p3-n16 | 16 | 20.396 | 120.2 | 800.9 | 6.66 | 109.2 |
| p4-n16 | 16 | 20.211 | 122.4 | 820.6 | 6.70 | 130.2 |
| p4-n8 | 8 | 19.518 | 62.4 | 414.6 | 6.64 | 116.8 |
| p4-n4 | 4 | 17.704 | 34.5 | 219.2 | 6.35 | 106.5 |
| p4-n2 | 2 | 14.401 | 20.9 | 130.5 | 6.24 | 105.7 |
| p4-n1 | 1 | 14.568 | 11.1 | 65 | 5.86 | 132.9 |

Clock is `\Processor Information(_Total)\% Processor Performance`, sampled
before and after each cell and averaged. This box boosts to ~175% of nominal
cool and was measured at 126% after 45 minutes of sustained load, so a unit's
clock is a covariate, not a constant.
