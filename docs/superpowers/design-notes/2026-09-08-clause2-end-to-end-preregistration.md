# 2026-09-08 — PRE-REGISTRATION: clause 2 on END-TO-END WALL, and the same-block run that grades it

**Committed BEFORE the measurement it grades.** The reframing in
`2026-09-08-prefill-policy-and-what-the-bar-measures.md` was derived from data
already on disk; **the run below has not happened yet**, and its thresholds are
fixed here.

---

## 0. ⚠⚠ THE OBJECTION, STATED FIRST AND NOT BURIED

`docs/ROADMAP-SEQUENCE.md` §2.1 already warned against this exact move:

> ⚠ **The honest counter, stated so it is not buried:** Kabir's standing priority
> is *"achieve the goal first, 0.51 bar is unacceptable"*, and clause 2 as
> written says **≥1.0× llama-server-tuned**. 0.687× is not 1.0×. **Re-specify
> the criterion BEFORE re-grading it, never after seeing the number** — the
> legitimate version of this is 1.1 (measure @16 live, where the thesis predicts
> stoke wins), not a redefinition chosen because @8 missed.

**That warning applies to this document and it is not dismissed.** The
end-to-end reframing *was* worked out after seeing @8 miss on the aggregate.
Three things are offered against it, and a reader should weigh them rather than
take the change on trust:

1. **The metric's defect is structural and pre-dates the analysis.**
   `src/main.c:1090`, written weeks earlier, already says AGGREGATE(M0) leaves
   "the leading prefill outside the span" while charging "a later arrival's
   prefill in full". `ROADMAP-SEQUENCE.md` §2.1 itself says *"the difference
   between those two numbers is entirely where the clock starts."* The
   reframing is a restatement of something already written down, not a new idea
   invented to escape a number.
2. **The thresholds are inherited VERBATIM.** ≥1.0× llama-server-tuned and ≥2.0×
   ollama-default. Only the measured *quantity* changes. A change that also
   moved the thresholds would be indefensible; this one can only be argued about
   on the quantity.
3. **It does not manufacture a pass.** The ollama half is predicted to MISS on
   the new metric exactly as it missed on the old (§3).

⇒ **Recorded disposition: this is a POST-HOC re-specification of the measured
quantity, approved by Kabir on 2026-09-08 ("yea do it, i agree with your
recommendations") after the objection was put to him.** What is *not* post-hoc
is the grading: the thresholds and predictions below are frozen before the run.

## 1. The change

| | clause 2, as written | clause 2, proposed |
|---|---|---|
| quantity | `AGGREGATE(M0)` tok/s, absolute | **end-to-end wall** for a fixed workload, as a **same-sitting ratio** |
| vs llama-server-tuned | ≥ 1.0× | **≥ 1.0×** (unchanged) |
| vs ollama-default | ≥ 2.0× | **≥ 2.0×** (unchanged) |
| both engines measured by | different code, different transports | **the same harness cell, over HTTP+SSE** |

**Definition, so it cannot drift:** for one scenario cell,
`ratio = referent_wall_s / stoke_wall_s`, where `wall_s` is
`bench/bench.py`'s — the perf-counter span from just before the client threads
start to when the last one finishes, with the server already up and warm.
Higher is better; > 1.0 means stoke finishes the same work first.

**Why this quantity:** it is what a user waits, it is recorded identically for
every engine by code that already exists, and it has **no degree of freedom an
engine can game** — an engine cannot improve it by delaying its first token,
which is precisely what AGGREGATE(M0) rewards.

## 2. The run that grades it

The banked comparison (75.7 s vs 77.6 s @8) has stoke measured **in-process**
and the referents over **HTTP+SSE**. That asymmetry favours stoke and is
unmeasured. This run removes it: `bench/adapters.py`'s `StokeAdapter` (M6 rung
A5) makes stoke a cell of the same matrix, so both sides go through
`bench.run_scenario` unchanged.

**Block, one engine at a time, bracketed:**

| order | cell |
|---|---|
| p1 | `llamacpp-tuned` × `s2-closed-8` |
| p2 | `stoke-tuned` × `s2-closed-8` |
| p3 | `llamacpp-tuned` × `s2-closed-16` |
| p4 | `stoke-tuned` × `s2-closed-16` |
| p5 | `llamacpp-tuned` × `s2-closed-8` — **the bracket** |

3 runs per cell. p1-vs-p5 separates drift *inside* the block from drift
*between* sittings, exactly as the 2026-09-05 block did (it agreed to 0.95%).

**stoke's cell holds the banked arm's configuration** — ctx 704, threads 8,
seqs = N, prefix dedup off — so the only thing that changes against the
in-process number is the transport.

## 3. PREDICTIONS, frozen here

1. **stoke/llama-server ≥ 1.0× at BOTH widths.** The in-process arm says 1.025×
   @8 and 1.056× @16.
2. ⚠ **The HTTP transport will cost stoke something, so both same-block ratios
   should come in BELOW the in-process 1.025× / 1.056×.** A ratio that comes in
   *higher* is a signal something else changed and must be investigated, not
   banked.
3. ⚠⚠ **@8 is the one at risk and it is called now: 1.025× is a 2.5% margin, and
   if the HTTP path costs stoke more than ~2.5% the @8 ratio drops BELOW 1.0×
   and clause 2's llama half FAILS on the new metric too.** That outcome is to
   be reported as a fail, not explained away. The metric change is not a promise
   about the result.
4. **stoke/ollama-default will MISS the ≥2.0× half**, as it does on the
   aggregate (1.542× wall vs 1.570× aggregate). Not run in this block —
   ollama @16 OOMs this box — so this half stays where 2026-09-05 left it.

## 4. TRIPWIRES

| # | tripwire | fires when |
|---|---|---|
| **W1** | every cell completes | any run reports `errors` or `completed` < n_clients |
| **W2** | the bracket holds | p1-vs-p5 `wall_s` medians differ by > 5% ⇒ the block drifted and no cross-cell ratio from it is publishable |
| **W3** | reconstruction | any run's `aggregate_decode_tps` fails to reproduce from its own per-client fields |
| **W4** | the workload is the one on record | any client's `n_tokens` ≠ 128, or `n_arrivals` ≠ `n_tokens` |
| **W5** | stoke is actually being measured | the stoke cells' `adapter_log_tail` does not show stoke's own server, or the label is not `stoke-tuned` |

**W5 is the non-vacuity one.** A harness that silently fell back to a llama
adapter would produce a beautiful 1.00× ratio and it would mean nothing —
[[fixture-can-be-blind-to-the-defect]] #11, one arm never ran.

## 5. What this does NOT change

- **AGGREGATE(M0) is still recorded** in every cell, so every earlier number
  stays comparable and nothing already banked is re-graded.
- **Gate item [3]'s 250 ms ITL bound is untouched.**
- **The ≥2× ollama half is untouched** and still missing.
- **The bars' absolute values** (17.7 / 16.8 …) are not re-anchored here; the
  2026-09-05 recommendation that they become same-sitting ratios stands
  separately and is what this change instantiates.
