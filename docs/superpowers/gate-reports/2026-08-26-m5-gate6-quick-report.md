# M5 gate item [6] -- preempt-lossless on a forced-preemption real-model trace

Run `2026-08-26-m5-gate6-quick`. Fixture **fanout-hetero** (2 clients, pairwise-distinct streams over a
load-bearing shared prefix), 200 generated, ctx 1600, threads 8, kv f32, **dedup OFF**.
Raw `2026-08-26-m5-gate6-quick-raw.txt`.

| arm | binary | pool blocks | preemptions | steps | rows | ids sha256 |
|---|---|---|---|---|---|---|
| R (reference) | win | derived | 0 | 219 | 2,735 | `891A966EF5435595` |
| P (pressure) | win | 168 | 1 | 253 | 4,070 | `891A966EF5435595` |
| M (mutant) | win-mut-preempt | 168 | 1 | 253 | 4,069 | `23969E347210AD02` |

## The four claims, each able to fail on its own

| # | claim | result |
|---|---|---|
| 1 | the reference is genuinely unpressured (preemptions == 0) | **PASS** (0) |
| 2 | the pressure arm genuinely preempted (preemptions > 0) | **PASS** (1) |
| 3 | **preemption lost nothing**: P's stream == R's, byte for byte | **PASS** |
| 4 | the comparison can tell: the off-by-one mutant DIFFERS | **PASS** (mutant preempted 1 times and diverged) |

## VERDICT (gate [6]): **MET**

Claim 4 is why claims 1-3 mean anything. The mutant is the shipping source built
with `-DSTOKE_MUT_PREEMPT`, which replays one row too FEW on preemption, so the
final replayed row reproduces the logits of the wrong position: the run stays alive
and the output is merely wrong -- the failure class `kv.h` names as the dangerous
one. Two SAMEs without it would be equally consistent with a preemption path that
never executed, which is exactly what every previous serve measurement recorded.

## Why a forced-preemption trace needed a new flag

The serve driver could not produce one. The pool is `n_seq * blocks(ctx)`, so every
sequence can always reach its own context; `--pool-blocks` was added for this gate.
Two further guards had to be defeated, both measured on the way here:

- **admission's watermark.** It approves `blocks(prompt) + W`, so a request whose
  whole growth fits in W's 8-block reserve can never outgrow it. At n 32 a sequence
  grows 2 blocks: a squeezed pool made the clients SERIALISE instead (rows 1,349 ->
  4,805, preemptions still 0). Growth must exceed W*16 = 128 tokens.
- **admission's refusal to overcommit.** A merely small pool admits fewer sequences.
  At pool 200, 2 of 4 admitted and both fit at full length: preemptions 0.

Preemption fires only for a pool inside **[k*(blocks(prompt)+W), k*blocks(prompt+n_gen))**
-- enough for k requests to be ADMITTED, not enough for them to FINISH. Here that is
[164, 186) at k=2, and 168 sits inside it.

## What this gate does NOT cover

**The dedup path.** With dedup on, admission still reserves the whole prompt while
adoption makes the residency SHARED, so `reserved` exceeds what the sequences occupy
and the window above closes -- measured: at pools 84 and 88 the clients serialised
and re-prefilled (rows 1,349 -> 4,805) without one preemption. Re-adoption after
preemption is asserted in the sim (`test_sched`: 1024 adopted over 9 admissions with
3 preemptions). **On the real model it is OWED**, and since dedup is now the serve
DEFAULT that is the more important of the two paths.
