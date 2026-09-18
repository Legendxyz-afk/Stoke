# stoke — Phase 1.5 planning document

**Status:** M7 deliverable (roadmap 4.3). Written 2026-09-06; **§6 added at the M7 code freeze, 2026-09-18** (the scheduler thesis measured, the one-slot ollama, prefix-reuse evidence, server hardening, the memory budget).
**Scope ruled by Kabir 2026-09-01:** *"arm first, we'll make it embed on most of
the hardware."* ⇒ **ARM NEON before CUDA**, and Phase 1.5 is therefore an
**embeddability** phase, not a GPU phase.

This document does not restate `COMPANY.md` §7, which carries the strategic
case and the verified port surface. It does three things §7 could not:

1. records a **measurement from 2026-09-06 that changes how the port can be
   validated at all**, and which contradicts a claim §7.5 makes;
2. turns the port into a **staged plan with gates**;
3. delivers the **verdict on the three parked candidates** the Fable 08-02
   review deferred to this document.

---

## 1. ⚠⚠ THE FINDING THAT CHANGES THE PLAN: determinism is per-build, not per-project

`COMPANY.md` §7.5 lists, as an asset to be marketed:

> **Determinism and bit-exactness** — already a project invariant, and it is
> exactly what regulated and embedded buyers require. It has been paid for; now
> it should be *marketed*.

**That claim is too strong, and it was measured on 2026-09-06.** stoke's own
Windows and Linux builds, from identical source on identical weights with greedy
decoding, produce **different text**:

| prompt | Windows (MSYS ucrt64 gcc 15.2) | Linux (Ubuntu gcc 13.3) |
|---|---|---|
| fibonacci | `if n <= 0:  return 0` | `if n <= 1:  return n` |
| Apollo | `The crew consisted of three astronauts` | `The crew consisted of astronauts` |

In the same run, **every discrete-domain golden passed on both platforms** —
`golden_info`, `tok_golden`, `tok_special_golden`, `unicode_sweep_golden`, and
all three legs of `tok_llama_golden`.

**Mechanism.** The forward pass runs f32 transcendentals — rope's
`cosf`/`sinf`/`powf` — through a different libm and a different compiler on each
leg. A sub-ulp difference in one angle propagates through 28 layers and
occasionally flips an argmax. Nothing is wrong with either build.

### 1.1 What this means for the port — three consequences

**(a) The ARM port cannot be validated by diffing text against the x86 build.**
This is the obvious plan and it is unavailable. If two x86 toolchains already
disagree, NEON will too, and a text diff would report "the port is broken" on
every run while telling you nothing.

⇒ **Decide the correctness instrument BEFORE writing NEON kernels.** Options,
in ascending order of what they cost and buy:

| instrument | what it proves | cost |
|---|---|---|
| **scalar twins, bit-exact** | each NEON kernel equals its tested scalar reference **on the same machine** | cheap, and §7.2 says the twins already exist and are tested — **this is the backbone** |
| **logit distance at fixed positions** | the forward pass agrees to a stated tolerance, per layer or at the logits | needs a τ, and τ must be frozen before grading |
| **top-k set agreement** | the ranking agrees even where the argmax is close | robust to exactly the flips seen above |
| **token-stream diff vs x86** | ✗ **not usable** — falsified 2026-09-06 | — |

⭐ **The scalar-twin route is the one this project already paid for.** §7.2's
most valuable line is that every SIMD kernel has a tested scalar twin, and a
twin comparison is *within one build on one machine*, so it is immune to the
cross-toolchain problem entirely. The port should be graded twin-by-twin, not
end-to-end against x86.

**(b) The README claim must be scoped (affects 4.4/B8).** "Deterministic" is
true and valuable, but the honest form is **"deterministic for a given build:
same binary, same weights, same seed ⇒ same tokens"** — which is what
`serve_golden` and `run_sampling` actually gate, and which is genuinely what a
regulated buyer needs (reproduce *this* deployment, not match a different
compiler). Claiming cross-platform bit-exactness would be falsified by anyone
who ran two builds.

**(c) It is a product question, not only a test question.** An ISV shipping the
same model to x86 and ARM users will see different outputs on the two. That is
normal for every inference engine, but it must be *documented*, because a buyer
who discovers it themselves will read it as a bug.

---

## 2. The port, staged with gates

§7.2's finding — only `simd.c`, `model.c:21` and `pool.c:154` gate on
`__x86_64__`; threading is already pthreads; `SIMD_CFLAGS` is scoped to
`simd.c`; runtime feature dispatch already exists — means the work is a port
with a **working, gradeable intermediate state**.

| stage | deliverable | gate |
|---|---|---|
| **1.5.0** | ARM64 builds and runs with **scalar kernels only** | the full test matrix green on ARM64, `tok_golden` byte-exact (discrete ⇒ portable), generation self-consistent |
| **1.5.1** | NEON for the f32 ops (rmsnorm, softmax, swiglu, rope) | each kernel **bit-exact against its scalar twin on the same machine** |
| **1.5.2** | `SDOT` K-quant dots (ARMv8.2) | twins again, plus a **recomputed overflow ladder** with a named falsifier |
| **1.5.3** | `SMMLA`/`i8mm` where available (ARMv8.6) | twins; runtime dispatch chooses |
| **1.5.4** | cross-compilation + size budget | Android NDK / iOS / buildroot builds; binary size tracked with a ceiling |

⚠ **The overflow ladder must be recomputed from first principles for each
(instruction, quant) pair, never inherited.** This is the trap Q6_K-on-VNNI
walked into: the ladder is a function of the quantization's value range and the
lane width, and getting it wrong produces *saturation*, not an error. The Q4_K
lane-permutation insight (put four different sub-blocks in the lanes so all
their scales apply at once) is expected to transfer to `SDOT` — but "expected to
transfer" is a hypothesis to test, and the ladder is the part that does not.

⚠ **Stage 1.5.0 is the one that proves the thesis.** `COMPANY.md` §1.2 claims
the *scheduler* is the asset and the kernels are commodity. An ARM64 build that
runs correctly on scalar kernels — no NEON at all — demonstrates exactly that,
and it is the cheapest possible demonstration. **Ship 1.5.0 before optimising.**

---

## 3. Verdict on the three parked candidates

The Fable 08-02 review parked these specifically so they would be weighed here.
None touches a Phase-1 gate.

### 3.1 Speculative decoding — **DEFER past 1.5, and the reason is measured**

The case for it is real: it trades spare compute for the bandwidth this box
lacks, which is the physics M0 measured, and M5's **replay-not-resample** rule
composes with a verify step.

**But it conflicts with the phase's own goal.** Speculative decoding needs a
draft model resident alongside the target — more memory, on the phase whose
entire point is *embeddability*, on hardware chosen because it is constrained.
And tonight's soak preparation is a reminder that memory is already the binding
constraint here: a 15.6 GB box could not run `ollama-default` at 16 clients at
all.

⇒ **Revisit when there is a target with memory headroom**, i.e. alongside CUDA
or on Apple Silicon's large unified memory — not as part of the embeddability
push.

### 3.2 Cross-request prefix reuse — **DO IT, and it is the strongest of the three**

- **It is nearly free**: it falls out of M4's refcounted dedup **if blocks
  outlive requests**, which is a lifetime change, not an algorithm.
- **It targets the measured weakness.** stoke's gap to llama-server is on
  AGGREGATE(M0), which is dominated by prefill; the 2026-09-06 block measured
  0.730× at 8 clients while TTFT was **1.65× better**. Prefix reuse removes
  prefill work outright rather than making it faster.
- **It matches the buyer.** Agent workloads resend the same system prompt every
  call — Path 1's ISVs and the AySen `local()` runtime both do exactly this.
- It strengthens the scheduler, which is the asset being claimed.

⇒ **Schedule it in 1.5, ahead of `SMMLA` work.** A cache hit beats a faster
kernel.

### 3.3 Grammar / JSON-constrained sampling — **DO IT, but late and small**

Sampler-layer, so it is contained and cannot destabilise the scheduler. The
demand is real (structured output is table stakes for tool use) and AySen's
`local()` runtime wants it at Phase 3. But it buys nothing for embeddability and
nothing for throughput.

⇒ **After 1.5.4**, or whenever a caller actually needs it. Sized in days.

---

## 4. What Phase 1.5 must add beyond kernels

From §7.5, with the planning detail it asks for:

- **Build system.** One Makefile assuming gcc + MSYS2/WSL does not survive
  embedding. Needs cross-compilation (Android NDK, iOS toolchain,
  buildroot/Yocto) and probably a CMake path, because that is what ISVs expect
  to vendor. ⚠ Note the Makefile already carries hard-won behaviour (`-j4` over
  `/mnt`, per-`(platform,sanitizer)` BUILDDIR, `EXTRA_CFLAGS` as the A/B arm
  hook) — a CMake path must not quietly drop those.
- **Size budget.** 1.38 MB is a feature. **Track it with a ceiling**, the way
  perf is tracked, or it erodes silently.
- **No-OS / constrained targets.** M4's block pool is already runtime-variable,
  which is most of the way there. The open questions are the filesystem
  dependency (model loading is `mmap`) and a fixed memory budget.
- **The ABI is the product.** `stoke.h` is frozen; that discipline is a
  commercial asset and should be stated in the README, not just practised.

---

## 5. Open for Kabir

1. **The correctness instrument for the port (§1.1a).** Scalar twins is the
   recommendation and it is nearly free. Confirm before NEON work starts, because
   retrofitting a grading protocol after kernels exist is how bands get fitted to
   results.
2. **How the determinism claim is worded in the README (§1.1b).** "Deterministic
   for a given build" is defensible and still valuable; anything stronger is
   falsifiable in one afternoon.
3. **Prefix reuse ahead of `SMMLA`?** (§3.2 recommends yes.)
4. **Which ARM64 machine is the reference?** There is none on this box. Tier 1
   correctness needs real hardware, and the choice (Apple Silicon vs Pi 5 vs a
   Graviton instance) decides the toolchain work in 1.5.4.

---

## 6. 2026-09-18 addendum — what the M7 close-out measured, and what it changes above

Written at the M7 code freeze (`memos/2026-09-18-m7-code-freeze.md`), from
the 09-16 kernel pricing and the 09-18 clause-2 sitting
(`gate-reports/2026-09-18-clause2-e2e-16-32-ollama.md`). Nothing above is
withdrawn; five things are now measured that were argued.

1. **§2's "1.5.0 proves the thesis" is no longer a bet — the thesis is
   measured.** Stoke's end-to-end margin over llama-server *grows* with
   concurrency: **1.18× @8 → 1.20× @16 → 1.28× @32**, in 0.75 → 0.73 → 0.71×
   the CPU-seconds, first client answered 4.5–5.1× sooner — with every
   pre-registered band met. In the same week the kernel levers priced
   **null in situ** (K1/K2: 1.26× on a single-thread bench, ~1.01–1.05× at
   12T, because the matmul is bound by the P-cores' shared ports). ⇒ The
   asset the port carries to ARM is the scheduler, and stage 1.5.0 (scalar
   kernels, the scheduler intact) is the demonstration that matters
   commercially. Tier-2/3 NEON work is where the *absolute* numbers come
   from, not where the *relative* claim lives. **Ship 1.5.0 first** stands,
   now with numbers behind it.

2. **The ollama comparison is a positioning fact, not a benchmark loss.**
   `ollama-default` on a CPU-only box serves **one request at a time**
   (`OLLAMA_NUM_PARALLEL` resolves to 1 without a GPU; measured: wall =
   N × 16.2 s, ITL 68 ms = a single-sequence decode). Stoke reads 1.74× @8
   and 1.84× @16 against it — the clause's ≥2× is a documented miss — but
   the sentence for `COMPANY.md` §3.2 and the README is sharper than "2×":
   **on CPU, the incumbent does not do concurrency out of the box at all.**
   That is the cell stoke owns, on exactly the hardware Phase 1.5 targets.

3. **§3.2 (prefix reuse) gets a caveat about its own evidence.** The
   clause-2 scenario (S2) salts each client's prompt at the FRONT of the
   user message, so M4's dedup adopts only the chat-template header there
   (measured 09-16: the @32 arm needs the full 6.5 GB f32 KV). The case for
   cross-request reuse is unchanged — agent traffic shares a *system prompt*,
   which S2 does not model — but the number that supports it is S3's
   (dedup worth 2.284× on fan-out), not anything in the clause-2 banks.
   When 1.5 builds it, the grader must be a scenario with a shared prefix
   and a per-client tail, and the reconstruction check must hold.

4. **Server hardening is an embeddability item, and it now has two scars.**
   `MAX_CONNS 24` and a listen backlog of 16 were both "a local server does
   not need more" until a 32-client burst arrived: the backlog refused 13
   clients because the server accepts **one connection per turn** and a turn
   is a whole prefill chunk (~2.8 s) once anything is running, while Windows
   drops a SYN that finds the queue full and the client gives up after ~2 s.
   Both constants are fixed (`9eca59e`, `b1ee6b3`), the loop is not. For an
   embedded server the right shape is: drain the whole accept queue per
   turn, size the backlog to the table, and keep `tests/tooling/burst_smoke.py`
   (which had to be made red twice before it could see the defect) as the
   gate. Add to **1.5.4** beside the size budget; a day, with its own A/B.

5. **Memory arithmetic for constrained targets, from the sitting's peak
   working sets.** Stoke's footprint is exactly `model (mmap) + N × ctx ×
   288 KB` at f32 KV: 4.2 / 5.8 / 9.0 GB at 8 / 16 / 32 clients × 704 ctx
   on a 2.5 GB Q4_K_M model. On a 4–8 GB ARM board the KV term is the whole
   budget. Two consequences for §4's "fixed memory budget" question:
   - the embeddable interface is a **memory budget that derives `seqs ×
     ctx`** (the pool already derives from those two; only the inversion is
     missing), not two knobs a host must size by hand;
   - **a KV quantization decision belongs to 1.5**, re-measured on ARM: on
     x86 `--kv q8` cost +3.6% because the dequantize outweighed the traffic
     it saved, but the balance is different where memory, not the port
     count, is the wall — and it halves the term that dominates.
   And from B7's rig: an engine that is *never* the OOM killer's pick
   (`oom_score_adj -1000`) is a deployment trick, not a property. The
   property an embedded host wants is admission that refuses a request the
   budget cannot hold, which is the same derivation as the first bullet.

Open for Kabir, added to §5: **(5)** does 1.5 carry a `--memory-budget` (or
`stoke.h` field) that derives the pool, and a KV-quant re-measurement on the
ARM reference, as stage-1.5.4 items? Recommendation: yes to both; they are
the two lines a board vendor will read first.
