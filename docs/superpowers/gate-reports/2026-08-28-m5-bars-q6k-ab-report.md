# What the Q6_K maddubs kernel bought on the gate [2] cell

Run `2026-08-28-m5-bars-q6k-ab`. ONE binary, `stoke.exe` sha256 `B1E99988B3C2`, both arms
selected at runtime by `STOKE_Q6K_KERNEL`. Cell is gate [2]'s own: 8 clients on S2's
prompts, ctx 704, n 128, kv f32, `--no-dedup`, threads 8. Order ref/mad/mad/ref.

**The switch was proven live before any arm ran**: `STOKE_Q6K_KERNEL=not-a-kernel`
is refused by `vec.c` and the probe run exited nonzero. The two kernels are
bit-identical, so output cannot distinguish them, and the timing difference is the
thing being measured -- it cannot also be the evidence that the switch worked.

## Units

| id | kernel | AGGREGATE(M0) | wall s | cpu s | quiesced |
|---|---|---|---|---|---|
| u1-ref | ref | 6.993 | 161 | 1098.4 | True |
| u2-maddubs | maddubs | 9.032 | 124.8 | 829.9 | True |
| u3-maddubs | maddubs | 9.038 | 124.8 | 829.9 | True |
| u4-ref | ref | 6.962 | 161.4 | 1103.9 | True |

## The ratio, against a floor made of the arms' own spreads

- `ref` mean **6.978** tok/s over 2 runs, spread 0.45%
- `maddubs` mean **9.035** tok/s over 2 runs, spread 0.07%
- floor = max(spread) = 0.45%

**maddubs / ref = 1.2949x** -- gain 29.49% against a floor of 0.45%, so **RESOLVED**

A cell resolves only when the gain clears TWICE its floor. That rule exists because
`gain > floor` passed a cell on 2026-08-27 at 33.91% over 32.76% -- which is exactly
what a gain that IS the noise looks like.
