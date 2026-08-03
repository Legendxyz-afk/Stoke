# RULING — K3 SIMD headroom reopened: D3 is retracted and replaced, and the box-state question is answered by forensics, not physics

**Status:** binding ruling. Fable session, 2026-08-03 (~15:40 IST), Kabir-launched
per the routing. Brief ruled on:
`2026-08-03-k3-simd-headroom-REOPENED-FOR-FABLE.md`, which reopened D3 of my own
`2026-08-03-m3-optimization-level-RULING-FABLE.md` (§2.1–2.3). D1/D2/D4/D5
stand, landed, out of scope. The instruction to read my prior self skeptically
was followed; the outcome is below, and it is not kind to the prior ruling.

**This session took no interleave of its own, and that refusal is the rule
being applied:** at ruling time the box hosts `FactoryGameSteam-Win64-Shipping`
at 72–93% busy on LP0 (the pinned core) with free RAM at 1.79 GB against a
4.28 GB model. Any tok/s number taken in this window measures the game and the
NVMe. What this session took instead was **forensics** (§2), which turn out to
be worth more than another cycle of tok/s.

---

## 1. R1 — D3 is RETRACTED and REPLACED

**Retracted, as measured-false or built-on-measured-false:**

- §2.1's central comparison (llama 1T effective 5.4 GB/s vs stoke matvec
  6.92 GB/s) — cross-window, both operands now known to be from differently
  co-loaded windows (§2). The conclusion "SIMD collects nothing single-thread"
  is dead in every window: the reopening's interleaved cycles put llama VNNI at
  1.96× and 2.52× over stoke scalar `-O3` within the same minutes.
- §2.1's overhead arithmetic (794 ms/token ⇒ matvecs ~6–8 GB/s) — valid
  arithmetic on a poisoned premise; in the less-depressed window the same
  command gives ~407 ms/token and the argument dissolves.
- §2.1's mechanism claim ("the per-core bind is latency/MLP structure, which
  SIMD does not remove") — refuted as stated: the production VNNI kernel
  demonstrably runs ~2× stoke's scalar rate under identical conditions.
- §2.2 in full: "5.5×/6 cores, near-linear, the bar is won in K5" was the
  6T figure divided by a depressed 1T. Withdrawn (and see §3 — its replacement
  is NOT the reopening's saturation story either).
- §2.3's exit expectation "~1.0–1.3× over scalar `-O3` (~1.2–1.6 tok/s)".
- The "landing near the right answer for the wrong reason" dismissal of the
  brief's ~2.3×. **Withdrawn.** The brief's number is restored as directionally
  right; its original derivation (bandwidth-gap arithmetic) was incomplete
  rather than wrong, and what makes the figure solid now is the measured llama
  existence proof, taken interleaved.

**Survives:** §2.3's four purposes of K3 (redeem the integer core; K-quant
compute; fewer cores to saturate DRAM; batched decode) — all four strengthen
under the corrected numbers. The pinning mandate survives **amended** (§4:
pinning without a co-load sidecar converts sibling occupancy into a hidden
multiplier — pinning is necessary, no longer sufficient). §2.2's "the bar is
feasible on this box" survives on better evidence than it was written on.

**Replacement — the single-thread SIMD lever and K3's exit expectation:**

- The lever is REAL and the working band is **~1.7–2.5×** over scalar `-O3` on
  Q8_0, one thread. The reopening's interleaved cycles give 1.96–2.52× under
  that window's co-load; the cross-window quiet figures (brief's stoke 6.92
  GB/s vs the best llama 2.46 tok/s) imply ~1.7–1.9× — quoted as indicative
  only, being cross-window. The point estimate for planning is **~2×**.
- **The binding number is the quiesced K3-entry pair** (already owed, §5): the
  pre-SIMD baseline row and the llama 1T referent, taken interleaved in one
  quiesced session. That pair pins the entry-time lever; nothing measured today
  may substitute for it.
- **K3's exit expectation is llama-1T PARITY expressed as a ratio, never an
  absolute:** median within-cycle ratio stoke/llama ∈ **0.9–1.1**, ≥3
  interleaved cycles, both pinned per protocol, same session, sidecar recorded.
  The brief's "~2.4 tok/s pinned" phrasing is rejected: this box currently has
  no trustworthy absolutes (§2), and an absolute target would inherit whichever
  window measures it. Parity against a co-measured referent is box-state-robust
  by construction; that is the whole design.
- This expectation is a scoping aid for K3-the-work. Gate [3] (≥7.8 on Q4_K_M)
  is untouched and is judged at M3 close per §4's binding, not at K3 exit.

## 2. The box-state question — answered by forensics; the ratio was enough, and now we know why it had to be

The brief's most important paragraph was its last: both sessions' absolutes
swing up to 2× and no physical cause was found ("power plan is Performance, on
AC, no thermal signal"). **This session found the variable nobody recorded: the
process table.**

Forensic timeline (all 2026-08-03 IST, from process/boot metadata and commit
timestamps, gathered this session):

| when | event |
|---|---|
| 08:43 | box boots |
| 08:48 | VS Code (session host) starts |
| 09:52 | Discord starts (two processes; 4.6 + 1.3 CPU-hrs by 15:37) |
| **09:57:55** | **Satisfactory starts** (`FactoryGameSteam-Win64-Shipping`) |
| ~10:45–11:15 | prior ruling's runs 1–6 (ruling committed 11:17:57) |
| ~11:55–12:06 | fork-settlement measurements (d6e02d1 committed 12:06:31) |
| ~12:10–13:35 | reopening's interleave + re-measurements (38b4f5a at 13:38:16) |
| 15:37 | game at **28.75 CPU-hours over 5.66 wall-hours = 5.08 cores sustained average**; free RAM 1.79 GB; LP0 72–93% busy, LP4 >100% utility |

So: **every contested measurement of the day was taken with a ~5-core game
resident**, except plausibly the `-O` brief's own window (its session predates
the 10:31 brief commit; it recorded the highest ref streams, 15.5–16.6, and the
highest stoke matvec, 6.92 — self-consistent with genuine quiet, unverifiable
now, treated as indicative). No memo of either session mentions the game; the
box-state checks that were done (power plan, AC, thermals) were the right idea
aimed at the wrong subsystem.

**The mechanism, and it explains every anomaly of the day at once.** An
unpinned probe is migrated by the scheduler onto whatever core is idle — it
DODGES co-load and reads high. A pinned run cannot move; high priority preempts
same-core threads but not the SMT sibling's occupancy or the DRAM queue, so a
pinned number inherits the neighborhood of its moment — reproducibly within a
window (hence ±0.01), arbitrarily across windows (hence 2×). And the tax is
selective by kernel structure: the bench's reference streams (verified this
session: 256 MB buffer, 10× LLC — a genuine DRAM probe) run 8+ independent
chains and hide latency under load, while the scalar Q8_0 matvec's dependent
34-byte-block chain cannot, with llama's VNNI kernel in between. That one
sentence explains: the prior ruling's crossed signature (unpinned bwprobe 14.4
GB/s beside pinned llama 1.26 in the same session — runs taken after 09:57);
fast-ref/slow-matvec inside single bench invocations (the deleted baseline row:
ref 14.97/14.98/13.16 beside matvec implying 0.987 tok/s); the 1.26→2.46 llama
swing on an identical command (game load phase, not flag); and the reopening's
own cycle-2 wobble (stoke −21% while llama +6% — a co-load phase edge inside an
interleave).

**Ruled, on the open question:** the within-cycle ratio was sufficient to rule
on — every decision at stake is a direction/structure decision that does not
flip anywhere in the observed 1.7–2.5× band — and the box-state question did
NOT need settling first, because it is not a physics mystery to be solved
before work continues; it is a **recording gap**, closed by instrumentation
(§4). The candidate cause is now concrete and named. It is not PROVEN (there is
no retroactive per-LP history); the sidecar makes the next depressed window
self-explaining, which is what settling it costs and all it costs. **If** a
future window shows a ≥1.5× swing with a clean sidecar — co-load nil, sibling
idle, hard faults ~0, pinned-LP clock flat — that is a results-vs-physics memo
to Kabir naming the hardware, and I do not expect it.

**Absolutes are quarantined project-wide:** no absolute tok/s or GB/s may
appear as a target, expectation, or comparison operand unless it comes from a
quiesced-checklist session or is one side of a same-session interleaved pair.
Everything else is decoration and must be labeled so. The ≥7.8 bar is not
threatened by this: gate [3]'s own LOCKED text re-anchors the bar to llama
re-benched at gate time, so the gate comparison is a same-session ratio by
construction once §4 binds. The bar's number stays Kabir's, untouched.

**One note the project should keep:** a 5-core game is not a defect of this
box; it is stoke's actual operating environment — the product thesis is
inference on desktops that do other things. The discipline is to keep two
regimes distinct: **gate/attribution numbers** (quiesced checklist, sidecar
proving it) and **ambient numbers** (co-loaded, labeled, exactly what M5's soak
should care about). I recommend — Kabir's call, M5's note — that the M5 soak
add a deliberately co-loaded AMBIENT leg, since today accidentally demonstrated
it is the environment stoke will actually live in.

## 3. R2 — "the bar is won in K5" comes out, and its replacement is a budget, not a new hero

Out of ROADMAP §M3 it comes. But the reopening's replacement — "scaling
saturates at ~26–34 GB/s aggregate, K5 is a small lever" — is **also not
granted as a finding**, and this is where reading my successor skeptically
earned its keep:

- **New hardware fact (read this session):** the box is 2× 8 GB Samsung
  DDR5-4800, one DIMM per controller — **dual-channel, 76.8 GB/s theoretical**.
- The reopening's 6T/10T runs (6.13 / 7.97 tok/s ⇒ 26.3 / 34.1 GB/s effective)
  were taken against a ~5-core game. Sublinearity under those conditions is
  contention arithmetic, not a DRAM wall.
- M0's quiesced llama-tuned run already demonstrated **~39 GB/s effective**
  (15.6 tok/s × 2.4973 GB on Q4_K_M). The quiesced scaling curve on Q8_0 has
  never been measured. "Saturating ~26–34" may not be written anywhere as this
  box's ceiling.

**The attribution that replaces both stories — a budget:**

- The bar needs **~19.5 GB/s effective aggregate** (7.8 × 2.4973 GB, Q4_K_M).
- The box has demonstrated **≥39 GB/s effective quiesced** — ≥2× slack — on
  hardware whose theoretical is 76.8. The wall does not bind at the bar.
- **K3** buys per-core effective bandwidth: ~2×-class (§1), from ~4.8–6.9 to
  the ~10.5 GB/s the VNNI existence proof shows this box gives one core.
- **K5** must then multiply a parity-class core by only **~1.9×** to reach the
  bar's aggregate — two cores' worth against six P-cores available — versus the
  5.5× my ruling claimed it had and the 3.24× cap the reopening measured under
  a game.
- **K4** prices the token 1.714× cheaper with an unmeasured K-quant compute
  penalty — the one genuinely open factor.
- The levers compose as **min(wall, N_eff × per-core)**, not as a free product;
  the reopening's §4.1 chain (2.2 × 3.2 × 1.714) double-counts by multiplying
  through the very ceiling that caps it.

**Can the bar be attributed across K3/K4/K5 today? No — and here is what
settles it:** (a) the quiesced K3-entry session (§5) pins the per-core lever
and, extended by a llama 6T/10T mini-sweep in the same session, the true
scaling curve; (b) the K4-entry K-quant measurement (already mandatory) plus a
llama Q4_K_M 1T/6T pinned interleaved referent sweep pins the priced-file
per-core number. Until both exist, ROADMAP may say only: *both K3 and K5 are
real levers; the bar has ≥2× aggregate slack; the split is measured, not
argued.*

## 4. R4 — the STANDING RULE. Granted, named, and bound to artifacts

Three-for-three means the lesson format has failed; lessons inform authors,
rules bind artifacts. The rule:

> **THE COMPARABILITY RULE.** Two performance figures may be compared — ratio,
> difference, or percentage — only if measured in the same session under the
> same recorded conditions, interleaved whenever both sides are runnable. A
> comparison must cite its interleave (≥2 cycles) and quote the within-cycle
> ratio spread. A cross-session comparison is not evidence; it is a hypothesis,
> and it must be re-measured before it may be written into a memo, a ROADMAP
> line, a bench comment, or a ledger claim.
>
> **Corollary A.** Rep-to-rep stability (±σ) certifies precision within a
> window and says nothing about the window's validity. 1.26 ± 0.01 and
> 2.46 ± 0.04 were both internally excellent.
>
> **Corollary B.** Subtraction is comparison. A difference of
> differently-conditioned measurements is a measurement of the conditions
> (the §2.4 "residue", manufactured by subtracting a quiet-window bench from a
> loaded-window e2e).
>
> **Corollary C.** A field name is not a basis. `ref_gbs` means the serial
> stream in schema-1 rows and max(split8, xor) in schema-2 rows; a percentage
> must name its denominator's stream AND window. (Live demonstration: this
> session briefly read the schema-1 row's 7.64 as a depressed ceiling before
> checking the denominator's era. The error class survives even while
> adjudicating it.)

**Where it binds — all three, each at its artifact:**

1. **Gate condition.** K3 gate [3]'s llama referent is taken at gate time, in
   the gate session, interleaved with stoke's number, on the quiesced machine
   checklist — which the gate's own LOCKED re-anchor clause ("if llama.cpp is
   re-benched at gate time, the bar re-anchors") already provides for. This
   operationalizes existing text; the numeric bar and its 50% meaning do not
   move. If Kabir reads this operationalization as touching his bar, his
   reading wins and this clause severs — the rest of the rule stands.
2. **bench.c contract** (Opus lands): the Decision comment carries the rule;
   history rows' `box` object gains sidecar fields — co-load summary (top
   foreign process by CPU + count of processes >5%), per-LP busy for the pinned
   LP and its SMT sibling, hard-fault rate, pinned-LP effective MHz — additive
   optional fields, schema stays 2, documented in the comment. Rows missing a
   sidecar are ambient-grade by definition.
3. **Playbook + ledger template** (Opus lands): one checklist line — *"quoting
   two numbers together? same session, interleaved, cycles cited, sidecar
   attached — or don't."* — and the ledger template requires every recorded
   figure to carry its window timestamp and sidecar reference.

**Bound with it, same family:** e2e-by-differencing (`-n 36` − `-n 4`) is
valid only within one window and is retired as the standard method: **`stoke
run` gains an in-binary timing report** (decode-loop wall, tokens, tok/s,
load/prefill split) so every future e2e figure is single-window by
construction. Small observability change, no kernel code, K3-entry item, Opus
lands it.

## 5. R5 — no llama inside `bench --ops`; the referent lives at the protocol layer

**Denied as asked:** `bench --ops` does not grow a llama dependency. The bench
is the self-instrument of an embeddable engine — it must run on boxes with no
vendor tree, and an instrument that executes its own comparison target
conflates measurement with refereeing. The fork settlement already proved the
composition works without it: bench-implied and e2e agreed to 1.12× when taken
in one window.

**Granted at the layer where comparison actually happens:** a small driver
(`bench/referent-interleave.ps1` or `tests/tooling/`, Opus lands) that runs the
A,B,A,B cycle — llama-bench tg32 ↔ stoke (`run` timing or `bench --ops`
matvec) — captures the sidecar between cycles, and emits ONE paired record;
`bench/history.jsonl` rows gain an optional `referent` object (command, build
id b10037, per-cycle tok/s, within-cycle ratios) written by the driver. **The
attribution ladder may only cite paired records.** The reopening's §4.4
(referent rows re-taken, never re-cited) is subsumed: a referent row is valid
for exactly one session, the one that took it.

## 6. R3 — M4/M5: granted; the concurrency headroom is a bandwidth budget and the budget goes in the design note first

The frame is right even though the reopening's wall number was not. **M4's
design note opens with a bandwidth budget, before paged KV is scoped**, with
this skeleton:

- **W** = aggregate effective bandwidth, measured quiesced at M4 entry
  (in-session; llama multi-thread as the estimator; sidecar attached). Floor
  from evidence: ~39 GB/s effective (M0). Theoretical: 76.8. **Today's 34 may
  not be quoted as W.**
- **Per-composite-step traffic model:** weights(file) once per step, amortized
  ÷N across the batch — batching amortizes ONLY the weight term — plus
  Σ_seq KV(ctx_i) per step, un-amortized, plus activations (small). Include the
  crossover context beyond which the KV term dominates the amortized weight
  term, as a formula in N and ctx, so admission design sees it.
- **Q8-KV halves the KV term** — state it as the measured input Kabir's M5
  default-ON/OFF call consumes.
- **M5's bars as budget lines:** ITL <250 ms ⇒ per-step traffic ≤ W × 0.25 s;
  the 16.8@16 aggregate checked against W with the KV term at trace contexts.
  If the budget shows the 16.8 bar infeasible at measured W, that is a
  results-vs-physics memo to Kabir at design time — cheaper than discovering it
  in a soak.
- ROADMAP §M4's constraints block gains one line binding this (edited this
  session).

## 7. What was decided, who does what

| # | decision | owner |
|---|---|---|
| 1 | D3 retracted/replaced; lever ~2×-class (band 1.7–2.5×); K3 exit = interleaved parity ratio 0.9–1.1, no absolute targets | ruled here |
| 2 | "Bar won in K5" out of ROADMAP; budget attribution; saturation claim withdrawn; composition = min(wall, N×per-core) | ruled here; ROADMAP edited |
| 3 | Comparability rule + corollaries; absolutes quarantined | ruled here; bindings → Opus |
| 4 | bench.c Decision comment + `box` sidecar fields + optional `referent` field | Opus |
| 5 | Referent-interleave driver script; ladder cites paired records only | Opus |
| 6 | `stoke run` in-binary timing report (retire differencing) | Opus, K3-entry |
| 7 | Playbook checklist line + ledger template line | Opus |
| 8 | K3-entry (blocking, one quiesced session): baseline row + llama 1T referent interleaved + llama 6T/10T mini-sweep + sidecar; quiesced = game/Discord closed, all-LP busy <5% for 60 s, free RAM > model + 2 GB, sidecar proving it | Opus, Kabir's box |
| 9 | K4-entry adds llama Q4_K_M 1T/6T pinned interleaved sweep beside the K-quant measurement | Opus |
| 10 | M4 design note opens with the bandwidth budget; W measured at entry | Opus, M4 |
| 11 | AMBIENT soak leg recommendation | Kabir, M5 note |

**Files touched by this session:** this memo; `docs/ROADMAP.md` §M3 items (3)
and (4) and one line in §M4's constraints block. No source. Commit local-only.

## 8. Not revisited

The ≥7.8 bar, ε, the waiver budget, the corpus, the parity protocol, push
policy, the model/owner line. Direction check, per the brief's own clause: this
ruling makes K3's expectation HARDER (parity, up from 1.0–1.3×), removes an
unearned promise made on K5's behalf rather than shrinking K5's scope, and
moves no gate in any direction. The quiesced K3-entry session and the K4-entry
measurement are pre-registered as the numbers that settle the attribution —
recorded here, before their results exist, which is what makes using them later
legitimate.

**On the prior ruling, for the record:** its failure was not arithmetic. It
compared across windows in the act of ruling against comparing across bases;
it read ±0.01 as validity (Corollary A now exists because of it); and it
applied "measure, don't argue" to the kernel question while never measuring the
window the kernel ran in. All three correctives in this ruling are structural —
rule, sidecar, pre-registration — because the third recording of a lesson is
the last one that gets to be a lesson.
