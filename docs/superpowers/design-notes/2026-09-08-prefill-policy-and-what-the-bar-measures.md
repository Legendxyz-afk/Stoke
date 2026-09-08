# 2026-09-08 — The prefill policy, what the bar actually measures, and what the change would buy

**Question (Kabir): what is the prefill policy, and what is best for us?**

Short answer: **the policy is one-prefill-at-a-time fused with decode; the 0.730×
gap is mostly an artifact of what `AGGREGATE(M0)` measures; and on total time to
finish the same work stoke already beats llama-server at both widths.** The
policy change would move the *bar* a lot and the *clock* about 3%.

Everything below is either read out of the code or reconstructed from data
already on disk. **One new measurement was taken** (§4) and it is a same-sitting,
counterbalanced, one-unit-per-process A/B.

---

## 1. What the policy IS

`src/sched.h` states it as the step contract:

> ONE fused forward: every runnable decode as one row each, **plus at most one
> chunk of the single active prefill**. The prefill chunk RIDES THE SAME OP WALK
> — it is extra rows, not a second pass — and that co-execution IS the design.

and `src/sched.c` carries it as invariant **S1**: `Req* prefilling; /* at most
ONE (S1) */`. The chunk is `SCHED_CHUNK_TOKENS = 128`, overridable at runtime by
`STOKE_SCHED_CHUNK` (S-ALIGN: must be a whole number of KV blocks, refused
otherwise).

So, concretely, per step:

| | stoke today |
|---|---|
| decode rows | every runnable sequence, one row each |
| prefill rows | up to 128 tokens **of one sequence** |
| passes | **one** — prefill and decode share the op walk |
| prefills in flight | **exactly one** |

**The behavioural consequence is the whole point:** a request starts decoding as
soon as *its own* prompt is done, not when the batch's prompts are done. First
tokens therefore come out **early and staggered**.

llama-server does the opposite: it packs prefill across sequences, so it
front-loads a large prefill batch and everyone's first token arrives **later and
together**.

## 2. Why that makes us look 0.730× when we are not

`AGGREGATE(M0)`, the metric the bars are cells of, is

```
        sum over clients of (n_tokens - 1)
  ----------------------------------------------------
  max(last-token wallclock) - min(first-token wallclock)
```

**`src/main.c:1090` already says the consequence in as many words:**

> It is not a per-client average and it is not end-to-end throughput: **the
> leading prefill sits outside the span, while a later arrival's prefill sits
> inside it and is charged in full.**

Measured, from the banked 2026-09-05 block (every row below reconstructed from
the raw per-client fields and checked against the published aggregate — all 15
referent runs matched to 12 digits):

| @8 | first token anywhere | span (the denominator) | prefill inside the span |
|---|---|---|---|
| llama-server | **~26.7 s** | 51.1 s | little — the bulk is *before* the span opens |
| stoke | **~5.7 s** | 70.1 s | nearly all of it, charged in full |

⇒ **The metric pays an engine for making its first client wait.** llama-server's
26.7 s of prefill is free; ours is billed. That is the 0.730×.

## 3. The number that has no such freedom: total time to finish the work

Both harnesses record it, and it was already on disk.

- llama/ollama: `bench/bench.py` `wall_s` = perf-counter across the client
  threads, server already up and warm.
- stoke: `end-to-end` = `total_ids / (t1 - t0)`, `t0` set immediately before the
  submit/step loop with the model already open (`src/main.c:991`/`:1040`).
  **Same semantics: run start → all clients done.**

Same block, same box, same sitting:

| workload | stoke | llama-server | ollama-default |
|---|---|---|---|
| @8 (8 × 512-tok prompt, 128 gen) | **75.7 s** | 77.6 s | 116.7 s |
| @16 | **143.4 s** | 151.5 s | — (OOMs this box) |

| ratio (referent ÷ stoke, >1 = stoke finishes first) | @8 | @16 |
|---|---|---|
| vs llama-server | **1.025×** | **1.056×** |
| vs ollama-default | **1.542×** | — |

**stoke already finishes first at both widths**, while the gate metric reads
0.730× and 0.904×. And the trade that buys it is real, not a rounding artifact:
TTFT p50 **31.9 s vs 52.0 s** @8 (1.65×) and **59.4 s vs 78.7 s** @16 (1.33×);
first-token-anywhere **5.7 s vs 26.7 s** (4.7×).

⚠ **Three honest limits on that table, all favouring stoke, none of them
measured:**
1. **stoke's arm is in-process; the referents go over HTTP/SSE.** ~1000 SSE
   chunks over ~76 s on localhost — probably ~1%, but it is *unmeasured*.
2. **The referent's prompts carried M0's per-run salt (~6 tokens); the exported
   stoke prompts do not** (`export_bench_prompts.py` documents this).
3. `wall_s` and `end-to-end` were built by different code for the same intent.

⇒ **This is directionally strong and NOT gate-grade.** Closing it is one run:
`bench/adapters.py` already has a **`StokeAdapter`** (M6 rung A5), so
`run_matrix.py --only-config stoke-tuned --only-scenario s2-closed-8` produces a
`wall_s` from the *same* code path as the referents. **That starts a server, so
it is Kabir's call.**

⭐ **Not metric-shopping:** clause 2's ollama half **misses on either metric** —
1.542× on wall against 1.570× on aggregate, both short of the required ≥2.0×.
Switching metric does not manufacture a pass.

## 4. What the policy change would actually buy — MEASURED

Batching prefill across sequences pulls exactly one arithmetic lever: **more
rows per weight load**. A GEMM does not care whether its rows came from one
sequence or four — the shape is `[n_rows × n_embd] × [n_embd × n_out]` either
way. So `STOKE_SCHED_CHUNK` is a valid proxy for that half, and it needs no
scheduler change. `sched.c:229` frames it the same way: *"A chunk amortises the
weight stream across its rows (throughput) and makes the step it rides in longer
(ITL) — gate item [2] wants it large and gate item [3] wants it small."*

Six units, @8, one process each, order **A B C / C B A** counterbalanced,
2026-09-08:

| chunk | units | median AGG(M0) | median end-to-end | within-arm spread |
|---|---|---|---|---|
| **128** (shipping) | 16.857, 16.709 | 16.783 | 15.532 | 0.88% |
| **256** | 17.438, 17.195 | **17.317** | **15.974** | 1.41% |
| **512** | 16.815, 17.151 | 16.983 | 15.691 | 2.00% |

| | vs chunk 128 |
|---|---|
| 256 | **+3.2%** |
| 512 | +1.2% — **not monotone** |

⇒ **Four times the prefill batch width is worth ~3% at best, and the trend
reverses at 512.** At 128 rows the weight stream is already amortised; the gain
is inside the same order as the arms' own spread.

⚠ **What this does NOT test, stated because it is the other half:** the chunk
sweep prices *arithmetic intensity* only. Cross-sequence batching also changes
**when a sequence becomes decodable**, and that is the half that moves
`AGGREGATE(M0)` — by pushing prefill *outside* the measured span. **So the
policy change would move the BAR substantially while moving the CLOCK ~3%.**
That asymmetry is the finding.

⚠ Today's absolutes (16.8–17.4) must **not** be compared to the 2026-09-05
block's (14.5). Same code, different sitting; this box has moved 1.8× across
sittings with no code change. The A-arm's own drift across this sitting was
−0.88% (first unit vs last), so the *ratios* above are sound and the *rates* are
not portable.

## 5. Recommendation

1. **Do not change the prefill policy to chase the bar.** It buys ~3% of real
   throughput, costs the number we are actually best at (TTFT, 1.65×/1.33×
   better), and reverses a choice M5 made deliberately.
2. **Change the bar.** The 2026-09-05 block already recommended clause 2 become
   a same-sitting *ratio* rather than an absolute tok/s. This note sharpens it:
   make it a ratio on **end-to-end wall for a fixed workload**. It is the number
   a user feels, it has no metric-gaming degree of freedom, and stoke passes it
   at both widths today.
3. **Close the one asymmetry** with a `StokeAdapter` run through `run_matrix.py`,
   so both sides are measured by the same code over the same transport. One
   server run; Kabir's word needed.
4. **Keep the ~3% in the headroom ledger, not on the critical path** — and note
   it is the *same* lever as the G≠4·k attention finding of the same day: both
   are "more rows per weight load", both are small, and neither is the gap.

⇒ **"0.51 bar is unacceptable" was right to be alarming and is now re-priced:
the engine is not half as fast as llama-server. It finishes the same work
sooner, at both widths, and gets the first token out 4.7× earlier. What was
half as fast was the metric's view of it.**
