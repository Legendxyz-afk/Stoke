# Gate item [4] (S3 fanout ≥ 13.3): the prompts are ready, and the real blocker is dedup

**2026-08-24, Opus 5, branch `m5-entry`.** Status note, not a result. Written
because the resume point recorded the blocker as "S3's shape is not exported and
`bench --serve` has no system-message seam", and only the first half of that is
the problem.

## 1. DONE: S3 is exported, and its shared prefix is VERIFIED

`tests/tooling/export_bench_prompts.py --scenario s3 --verify <stoke.exe>` writes
`tests/data/bench_prompts/s3-fanout-c{00..07}.txt` from the SAME
`bench/scenarios.py` the 13.3 baseline was served — the prompts are not
re-invented, for the same reason S2's were not.

**The flattening.** `stoke bench --serve` reads one file as one prompt and
applies no chat template, so the two messages are joined `system + "\n" + user`.
That gives the property the scenario needs: the shared text is a byte-identical
PREFIX of all 8 files, hence a shared TOKEN prefix.

**Verified, not argued** — the exporter tokenizes the set with `stoke tokenize`
and refuses to bless it if the common prefix is short:

```
verify: 8 files, lengths 1230..1230 tokens
verify: common token prefix = 1028 tokens = 64 full 16-token blocks adoptable by dedup
verify: OK (>= 1024)
```

1028, not 1025, because `"\n[client "` is itself shared and only the digit
differs. Blocks are 16 tokens and dedup registers only FULL blocks, so **64
blocks = 1024 positions per client are adoptable** — the whole system prefix.

⚠ The verifier earned its keep immediately: `stoke tokenize` prints a bracketed
comma list (`[840, 20772, ...]`), my first parser split on whitespace, and it
produced **zero** ids. A lenient parser would have compared two empty lists and
reported a common prefix of 0 — or, written the other way round, reported
success. It is now a strict parse with a non-empty assertion.

## 2. THE ACTUAL BLOCKER: there is no prefix dedup on the serve path

`--dedup` exists only on `cmd_batch`. `src/serve.c` and `src/sched.c` contain the
string nowhere. **So the M5 serving driver cannot exercise the one capability S3
exists to measure.**

**Why that decides the gate, in arithmetic.** S3 is 8 clients × (1024 shared +
~200 private) prompt tokens, 300 generated each.

| | prompt tokens prefilled | at ~32 tok/s | aggregate est. | vs 13.3 |
|---|---|---|---|---|
| no dedup | 8 × 1224 = 9,792 | ~306 s | ~5.6 tok/s | ~0.42× |
| with dedup | 1,224 + 7 × 200 = 2,624 | ~82 s | ~11.9 tok/s | ~0.89× |

Estimates, not measurements — decode is taken at the ~20 tok/s aggregate gate [2]
measured at N=8, and both rows inherit that. The point is not the second decimal;
it is that **dedup is worth roughly 2× on this scenario**, which is the
difference between missing the bar by 3× and missing it by 10%. Gate [4] is
named "the static-slot-beating proof", and sharing the prefix is *how* a
continuous batcher beats static slots. Measuring S3 without it measures stoke
with one hand tied.

## 3. Where it goes, and the four hazards — sized, not hand-waved

The seam is `admit_if_possible()` (`src/sched.c`, around the
`kv_open_in_partial` call): after the `Kv` opens, call
`kv_adopt_prefix(kv, r->spec.prompt_ids, r->spec.n_prompt)` and start `n_done` at
what it returns. The prefill loop already computes `chunk = pf->target -
pf->n_done`, so adopted positions are skipped with no further change. That part
is ~15 lines.

**What makes it a day and not an hour:**

1. **S5 is asserted as an EXACT identity**, not a bound — `n_done == n_prompt +
   n_gen - 1` for any RUNNING request. Adoption makes `n_done` start above 0 at
   admission. The invariant is still true, but the sim's derivation of it is not,
   and `test_sched` asserts it after every step.
2. **Preemption resets `victim->n_done = 0`.** With adoption that is now
   *wrong-but-safe* (it recomputes what it could have re-adopted) — and
   `kv_adopt_prefix` is **admission-time only**, returning 0 for a sequence
   already written, so re-adoption requires the readmission path to hold a fresh
   `Kv`. Whether it does is the first thing to check.
3. **The cost model is an INPUT to the sim, not a measurement** (gate [1]'s own
   caveat). A sim whose requests suddenly skip 1024 positions will report
   different preemption counts; that is a change in what the traces exercise, and
   the `pressure` trace already had to be made deliberately adversarial once.
4. **It needs its own correctness gate.** M4 proved dedup does not move the
   stream on the `batch` path; the serve path is a new caller of the same claim.
   The precedent is exactly the mins gate just run: byte-compare against a
   no-dedup arm, with a mutation to prove the comparison has power.

## 4. ⚠ CHECKED, AND IT SETTLES THE QUESTION: 13.3 WAS **NOT** MEASURED WITH PREFILL CHARGED FRESH

I was about to leave this as "someone should check whether llamacpp-default
shares prompt KV". It is checkable in one command — `llama-server --help` prints
and exits — and the answer is unambiguous. From the vendored b10037 binary's own
help, **all defaults, since `llamacpp-default` passes no flags beyond
`-m/--port/--host`**:

```
-sps, --slot-prompt-similarity SIMILARITY
        how much the prompt of a request must match the prompt of a slot in
        order to use that slot (default: 0.10, 0.0 = disabled)
-cram, --cache-ram N        set the maximum cache size in MiB (default: 8192, 0 = disable)
--cache-idle-slots          save idle slots to the prompt cache on new task ... (default: enabled)
```

**Slot-prompt similarity routing is ON by default at 0.10, the prompt cache is ON
by default at 8 GiB, and idle-slot caching is ON by default.** S3's clients share
1024 of ~1230 tokens (~83% similarity), and in closed-loop mode each client
re-sends the *same* prompt, so a slot it has already occupied matches it at
~100%.

⇒ **The 13.3 bar was set by an engine doing prompt-KV reuse. A no-dedup stoke
reading is therefore NOT like-for-like against it and must not be published as a
gate [4] result.** This inverts §5's first bullet: the no-dedup number is worth
having as a floor and as the denominator for pricing the dedup work, but it
cannot be graded against 13.3.

⇒ And it removes the choice from §3: **dedup on the serve path is not stoke's
optional advantage, it is the precondition for the comparison being fair at
all.** Gate [4] is blocked on it.

*(Nothing here says the bar is wrong. llamacpp-default is a real engine in its
real default configuration, which is exactly what M0 chose to measure against —
the point is only that stoke must be allowed its equivalent capability before the
two numbers mean the same thing.)*

## 5. Recommendation

1. **Gate [4] is BLOCKED on dedup-on-serve**, per §4 — not on the prompts, which
   are now exported and verified, and not on a "system-message seam", which turned
   out not to be the obstacle.
2. **Measure the no-dedup S3 reading anyway**, banked explicitly as a FLOOR and
   NOT as a gate result. It costs one quiesced run, it prices the dedup work
   against §2's table, and it is the denominator the dedup change will be graded
   on.
3. **Then implement dedup-on-serve** with §3's four hazards as the estimate, and
   gate it the way the mins term was gated: byte-compare a dedup arm against a
   no-dedup arm on the real model, with a deliberate mutation proving the
   comparison can fail. M4 proved dedup does not move the stream on the `batch`
   path; the serve path is a new caller of the same claim, and "same claim, new
   caller" is exactly what `run_m5_entry_verify.ps1` was written for.
4. Only then is there a number that can be called gate [4].
