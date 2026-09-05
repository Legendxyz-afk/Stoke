# stoke — the company analysis

*Written 2026-09-01 (Opus 5, max effort), at HEAD `ff1949b`, 47 days and 420
commits into the build, with Phase 1 seven-eighths closed.*

⚠⚠ **READ THIS FIRST: THIS DOCUMENT IS JUDGMENT, NOT MEASUREMENT.** Every other
strategic claim in this repo is anchored to a number taken on a named block.
This one is not, and it must never be quoted as though it were. Where a number
appears here it is cited from `ROADMAP-SEQUENCE.md` or `ROADMAP.md` and carries
their caveats with it. **The engineering documents outrank this file on every
fact; this file only arranges them.**

**What it extends, and does not replace:** `../MEMORY.md` §"Why this", §7
"Unified company thesis", §"Phased roadmap", and Appendix B "Every business path
we ran" hold the ORIGINAL decision journey (2026-07-14 onward) and are still
binding as a record of what was decided and why. This file is the **first
strategic re-examination since the engine actually existed**, and it exists
because building it for 47 days changed three of the inputs.

**Scope:** this spans **stoke AND AySen** (`Programming/loom`, private
`Legendxyz-afk/AySen`). The company was never "an inference engine company" — it
is the stack. stoke's repo hosts this file because stoke's `MEMORY.md` already
hosts the business record.

---

## 0. What is already decided — do not re-litigate

| decision | date | source |
|---|---|---|
| The goal is **not money right now** — build something proper and useful, low-level, a bottomless place to learn | 2026-07-14 | `MEMORY.md` §Why this |
| Selection filter: **moat = hard engineering · distribution = open source · fuel = learning · horizon = long** | 2026-07-14 | same |
| Frontier: **low-level systems; graphics secondary** | 2026-07-14 | same |
| **Retracted:** the 3D/WebGL component shop (moat = marketing = the weak axis) | 2026-07-14 | Appendix B |
| **The engine must stand alone.** OpenAI-compatible; AySen is *a* client, never *the* client, or the market collapses to AySen's userbase | — | §Guardrails |
| **Sequence, don't parallelize.** Engine first → AySen `local` backend → the full runtime | — | §Guardrails |
| **Phase 2 = the bridge layer**, NPU moves behind it | 2026-08-02 | Fable review, Kabir-approved |
| **Phase 1.5 = ARM NEON FIRST, then CUDA.** "we'll make it embed on most of the hardware" | **2026-09-01** | **Kabir, this session — see §7** |
| ⭐ **stoke is PRIMARY, AySen is SECONDARY.** "aysen is gonna be secondary, stoke is gonna be first" | **2026-09-01** | **Kabir, this session — see §0.1** |

### 0.1 stoke primary, AySen secondary — what it settles and what it costs

**This is not a new policy; it is the RECORDED guardrail finally being
enforced.** `MEMORY.md` §Guardrails already said *"Sequence, don't parallelize.
Engine first (standalone, useful) → AySen `local` backend as the integration demo
→ the full runtime. AySen never blocks the engine"* — and AySen was in Stage 2
anyway, which was Risk 4 in §8. **That risk is now dispositioned by decision
rather than left to drift.**

**⭐ It is also more coherent than it looks, for a reason worth stating: the two
solo-viable revenue paths do not need AySen at all.** Path 1 (ISVs embedding) and
Path 2 (silicon vendors) sell stoke's own properties — the frozen ABI, the static
binary, the scaling curve. **Paths 3 and 4 are the ones that need AySen, and they
are also the two that need a team.** So demoting AySen aligns the project with
exactly the paths that are reachable by one person.

**⚠ But it costs three things that were load-bearing, and they should be named
rather than discovered later:**

1. **The anchor tenant is gone (or deferred).** §4.1 and Part V.5 of
   `ROADMAP-SEQUENCE.md` both leaned on *"stoke has a first demanding user by
   construction."* With AySen secondary, **stoke must find real external users
   sooner**, which promotes the §9.3 falsifier — *"public release produces no
   inbound"* — from a background worry to **the primary commercial risk.**
2. **The unified company thesis is DEFERRED, not deleted.** "Language + router +
   engine = the local-first reliable AI stack" needs AySen. The near-term story is
   narrower and cleaner: **stoke alone — the embeddable concurrent inference
   engine.** That is a better fit for Paths 1 and 2 anyway, but it is a smaller
   claim and the record should say so out loud.
3. **The build-in-public credibility asset shifts.** Appendix B cast AySen as the
   marketing/credibility engine. That job now falls to stoke's own artefacts —
   and the **measurement discipline** (pre-registered predictions, null arms,
   mutation-tested graders, publicly withdrawn claims) is arguably stronger
   content than a language would have been. **Use it deliberately.**

⇒ **Consequences applied in this file:** §4.1's conclusion is amended, §5.4's
licence recommendation is sharpened, Risk 4 in §8 is closed, and question 3 in
§10 is answered.

⏭ **Operationally undecided and worth one line from Kabir:** does "secondary"
mean *paused*, *maintenance only*, or *worked on in stoke's blocked windows*?
⚠ Note the blocked windows in `ROADMAP-SEQUENCE.md` are already partly spoken
for — the 24 h soak window is allocated to the M7 planning doc (item 4.3).

Everything below is *analysis on top of these*, not a re-opening of them.

---

## 1. The thesis, pressure-tested

### 1.1 The thesis as recorded

> **Language (AySen) + router + engine = the local-first, reliable AI stack.**
> Write reliable AI programs that run fast on your own hardware — private,
> offline, spilling to cloud only when you choose. **Nobody owns "local-first
> reliable AI" end-to-end.**

**It survives.** Nothing in 47 days of building contradicts it, and the
engineering has made the "runs fast on your own hardware" half considerably more
credible than it was when written. Keep it.

### 1.2 What building it changed — three updates

**UPDATE 1 — The moat is the SCHEDULER, not the kernels.**

The record says *"moat = hard engineering"* without naming which engineering.
The build has now answered it, and the answer is not the obvious one:

- **NOT the moat: the AVX2/AVX-VNNI kernels.** They are the most impressive
  artefact and the least defensible one. llama.cpp already has broader kernel
  coverage on more architectures; a permutation trick is re-derivable by anyone
  who reads `objdump` for an afternoon; and every kernel gain in this project
  has a documented ceiling.
- **THE MOAT: the scheduler and its proof estate.** Paged KV + continuous
  batching + admission/reservation split + **preemption demonstrated LOSSLESS on
  a real model, twice** + the S1–S8 invariant set asserted after *every* step
  over a **1,000,130-step sim across 11,870 traces** + a **frozen ABI**. That
  took the milestones. It is portable to CUDA, NEON and NPU **unchanged**, and
  it is the part a competitor cannot copy in a weekend because most of its value
  is in the invariants that were proven, not the code that was written.

⇒ **The framing, internally and externally: stoke is a SCHEDULER WITH PLUGGABLE
KERNELS, not a kernel library.** Phase 1.5's own wording — *"CUDA onto the
**proven** scheduler"* — was already right. Adopt it everywhere.

⇒ **A corollary that matters commercially:** if the kernels commoditize, stoke
is not damaged. If someone lands a better AVX-VNNI dot in llama.cpp tomorrow,
stoke loses a benchmark row and keeps its asset.

**UPDATE 2 — The pitch narrowed, and sharpened.**

The recorded business case rested on the vLLM↔Ollama gap (~793 vs ~41 tok/s,
19×). What was actually built is **not** "multiples faster than everyone":

- prefill **1.09–1.13×** of llama.cpp, decode **parity at n=8** and **1.202× at
  n=16**;
- **total work 0.945×** — 5.5% *behind* on the continuous 8-client job;
- **TTFT p50 1.50× better**, first token at **5.3 s against 20.7 s**;
- **decode scaling B=4→16: stoke 1.72×, llama 1.07×** — a wall on their side.

⇒ **The sentence is not "faster." It is "the one whose curve goes UP where
theirs goes FLAT."** Narrower, and far more defensible — a speed claim invites a
benchmark war on someone else's turf; a *shape* claim is about architecture and
survives a 10% kernel improvement on either side.

⚠ **And the corollary for the README (`ROADMAP-SEQUENCE.md` 4.4): lead with
TTFT and the scaling curve, never with aggregate throughput**, which is llama's
home turf and where stoke is 5.5% behind. See §4 of `ROADMAP-SEQUENCE.md` Part V.

**UPDATE 3 — ⚠ The claim the whole story rests on is UNMEASURED.**

`MEMORY.md` makes Phase 1 *"useful, standalone, the day it beats Ollama at
concurrency."* **That referent is from 2026-07-17 and has never been
re-measured.** Its sibling — the llama-server bar — moved **39%** when it finally
was. So the single sentence this project most wants to say in public rests on a
46-day-old number of unknown accuracy in an unknown direction.

⇒ **Task 1.1 in `ROADMAP-SEQUENCE.md` now includes ollama-default at @8 and
@16.** Nothing in this document should be acted on commercially before that
number exists.

---

## 2. The asset — what is actually owned

| asset | strength | why |
|---|---|---|
| **The scheduler + its proof estate** | ⭐⭐⭐ | §1.2. Portable across every backend. The invariants are the value |
| **The frozen `stoke.h` v0 ABI** | ⭐⭐⭐ | Error codes never renumbered, fields only appended, nothing removed. **llama.cpp's C API is not frozen and integrators pin commits.** This is a real, rare, boring, valuable thing |
| **1.38 MB static binary, deps = `-lm -lpthread`** | ⭐⭐⭐ | The embeddability story, and (see §5) **also the licensing wedge** |
| **The measurement discipline** | ⭐⭐ | Unusual to the point of being a differentiator in itself: pre-registered predictions, null arms, mutation-tested graders, withdrawn claims left standing. It is why the benchmark report will be believed |
| **Portability posture** | ⭐⭐ | Verified this session: **only `simd.c`, `model.c:21` and `pool.c:154` gate on `__x86_64__`**; threading is already **pthreads**; runtime feature dispatch already exists. **And every SIMD kernel has a scalar reference twin** — so a new architecture is CORRECT before it is FAST |
| **AySen** | ⭐⭐ | The orchestration layer and the anchor tenant. v0.1.0 shipped, Stage 1 self-hosting complete. ⚠ Stage 2 in progress |
| **The AVX-VNNI kernels** | ⭐ | Impressive, valuable to users, **not defensible** |
| **Distribution** | ❌ **none** | Private repo, no README, no name that can be searched. §5, §6 |
| **A named buyer** | ❌ **none** | §4. This is the real gap |

---

## 3. The market, without flattery

### 3.1 What each incumbent owns

| | owns | does not own |
|---|---|---|
| **llama.cpp** | portability, GGUF, the ecosystem, single-stream, four years of maturity, hundreds of contributors | a frozen ABI; **concurrency shape past B=4**; first-token latency under fan-out |
| **Ollama** | `ollama run <name>` — **distribution, which was never the engine** | concurrency; embeddability; it is a wrapper |
| **vLLM** | datacenter-GPU throughput, the research mindshare | consumer/edge hardware, embeddability, small footprint |
| **LM Studio / jan / koboldcpp** | end-user GUIs | they are all *downstream of llama.cpp* |
| **Apple / Google / Microsoft on-device runtimes** | the OS-integrated slot | openness, model choice, cross-platform |

### 3.2 The cell stoke owns

> **x86-64 (and soon ARM) CPU · ONE user issuing MANY concurrent requests ·
> where FIRST-TOKEN latency matters · embedded in-process rather than behind an
> HTTP daemon.**

That is an **agent** workload — swarms, tool chains, fan-outs. llama.cpp and
Ollama are tuned for one human chatting; vLLM for many humans on a GPU.
**Nobody is tuned for one machine running sixteen agents**, and that is the gap
`docs/vllm-and-the-serving-landscape.md` identified before any code was written.
It has held up.

### 3.3 ⚠ The sobering comparable — adoption is not revenue

**llama.cpp has enormous adoption and no company. Ollama has enormous adoption
and no clear revenue model.** Those are the two closest comparables and both are
cautionary. Inference engines are commodity inputs: the value is captured
*above* them (orchestration, applications, managed services) or *below* them
(silicon). The historical pattern for companies built directly on inference
technology is **acquisition, not independence** — CPU-inference and
inference-compiler companies have consistently been absorbed by platform and
silicon vendors rather than scaling alone.

⇒ **Two conclusions, and they are the most important in this file:**
1. **Adoption is necessary and radically insufficient.** A plan that ends at
   "lots of people use it" is not a company plan.
2. **Acquisition is a legitimate outcome, not a failure**, and it should be
   planned for deliberately (it changes what you optimize and what you sign).

### 3.4 Is the niche growing or shrinking?

**Honest answer: the *substrate* is shrinking and the *workload* is growing, and
the second is bigger.**

- Against: GPUs get cheaper; **NPUs are shipping in every new laptop**; on-device
  runtimes are being pushed by OS vendors. Pure *CPU* inference matters somewhat
  less each year.
- For: **agent workloads are exploding**, and they are precisely many-concurrent,
  latency-sensitive, and privacy/cost-motivated toward local execution. Nobody is
  building for that shape.
- ⭐ **And the resolution is the same as §1.2's:** if the asset is the
  **scheduler**, then NPUs are not a threat, they are the next backend. The
  roadmap already has heterogeneous/NPU execution queued behind Phase 2, and
  Phase 1.5's ARM decision is the first proof that the scheduler ports.

⇒ **Bet on the workload, not the substrate.** Everything about the framing
should say "the runtime for local agent concurrency," never "a fast CPU engine."

---

## 4. Who writes a cheque — four paths, ranked

### PATH 1 — ISVs embedding local inference ⭐ best near-term fit

- **Buyer:** desktop/edge software vendors who need on-device LLM inference
  without shipping a GPU requirement or a 200 MB dependency — IDEs, creative and
  CAD tools, medical and industrial devices, POS, retail, robotics, defence.
- **What they pay for:** a **commercial licence** (escape from copyleft), support
  with an SLA, indemnity, **platform ports**, roadmap influence, long-term ABI
  stability guarantees.
- **Why stoke fits this better than anything else it could be:** every property
  the engineering already produced is on this buyer's checklist — **1.38 MB
  static, zero dependencies, a FROZEN ABI, readable in a sitting, no Python, no
  runtime, deterministic**. This was not designed for; it fell out of building it
  properly, and it is the closest thing to product-market fit the project has.
- **Deal size:** ~$10k–$100k/year. **Sales motion:** inbound from OSS usage,
  then a real conversation. **Solo-viable: YES.**
- **Requires:** multi-platform (⇒ §7 ARM), multi-model (⇒ B5), a licence that
  makes the commercial one necessary (⇒ §5), and public distribution.

### PATH 2 — Silicon vendors and OEMs ⭐ the acquisition path

- **Buyer:** Intel, AMD, Qualcomm, ARM, laptop OEMs, NPU vendors — anyone who
  needs a credible "your silicon runs agents well" story and does not have one.
- **What they pay for:** engineering contracts, co-marketing benchmarks,
  optimization work — **or they acquire the team.**
- **Why it is real here:** stoke's *entire premise* is extracting concurrency
  from hardware that is not a datacenter GPU. That is a vendor's marketing
  problem, and a scheduler that demonstrably beats the incumbent's scaling curve
  on their silicon is directly useful to them. **NPU support (already queued
  behind Phase 2) makes this path much stronger.**
- **Requires:** published, reproducible, *credible* benchmarks — which is exactly
  what the measurement discipline was built for — and hardware access.
- **Solo-viable: YES**, and it is the highest-ceiling outcome per unit of sales
  effort.

### PATH 3 — On-prem agent runtime for regulated enterprises

- **Buyer:** healthcare, legal, finance, defence, government — anyone who needs
  agents but cannot send data out.
- **What they pay for:** the whole stack (AySen orchestration + stoke inference +
  deployment + observability), support, compliance work.
- **Deal size:** $50k–$500k. **Highest revenue ceiling of the four.**
- ⚠ **Requires a sales motion, a team, certifications, and a support
  organisation. NOT solo-viable near-term.** Correctly a later phase.

### PATH 4 — Open-core control plane (the recorded Phase 4)

- **Buyer:** developers and teams running local agent fleets.
- **What they pay for:** hosted control plane, fleet management, model
  distribution, observability, policy, team features.
- ⚠ **Weakest near-term, and the record should be updated to say so:**
  local-first developer tools monetize poorly, and the buyer for "manage my local
  agent fleet" **barely exists yet**. This is a bet that the agent market matures
  into needing it.
- **It is still the right long-term shape** — just not the first revenue.

### 4.1 ⭐ The conclusion that ties the four together

> **Do not try to monetize stoke. Monetize what runs on it.**

An engine is a commodity input; a **runtime with an opinionated programming model
on top** is what people build businesses on and therefore pay for. This is
already the recorded thesis (§7 of `MEMORY.md`) — the update is that 47 days of
building have made the reason concrete: **stoke's job is to be free, excellent,
and embedded everywhere, so that AySen + the runtime is the thing with a price.**

⚠ **The one exception, and it is why §5 is a real decision rather than a
formality:** Path 1 monetizes stoke *directly*, and it is the best near-term
revenue. **Path 1 and "keep stoke free everywhere" pull the licence in opposite
directions.** That tension is the licence decision, and it cannot be deferred by
choosing both.

### 4.2 ⚠⚠ AMENDED 2026-09-01 BY THE AySen-SECONDARY DECISION — READ THIS BEFORE ACTING ON §4.1

**§4.1's conclusion assumed AySen would be the value-capture layer. Kabir has
made AySen secondary (§0.1), so that layer is now DEFERRED — possibly by
years — and the sentence must be re-read accordingly:**

> **"Monetize what runs on it" is still the right END STATE. It is no longer an
> available NEAR-TERM strategy, because the thing that was going to run on it is
> now the secondary project.**

⇒ **For the foreseeable future, stoke must stand on its own commercially, and
that means Path 1 and Path 2 are not fallbacks — they are THE plan.** Both sell
stoke's own properties; neither needs AySen; both are solo-viable.

⇒ **This makes the licence question (§5) MORE consequential, not less.** Under
the old reading, giving stoke away permissively cost nothing because value was
captured upstack. Under the new one, **a permissive licence with no CLA gives
away the only commercial asset that exists today** — and §5.1's clock is what
determines whether that is recoverable.

---

## 5. The licence — the only irreversible decision, and it has a clock

**`MEMORY.md`'s Appendix B enumerates thirteen business models and never names a
licence.** That is the largest gap in the strategic record.

### 5.1 ⏳ Why there is a clock

> **Kabir is the sole author of all 420 commits and therefore holds 100% of the
> copyright. He can still licence this work any way he likes — including
> dual-licensing, including changing his mind later.**
>
> **That option closes the first time a substantial outside contribution is
> merged without a CLA.** After that, relicensing requires every contributor's
> consent, which in practice means it never happens.

⇒ **Decide the licence, or adopt a CLA, BEFORE the repo goes public and starts
accepting PRs.** Doing neither is itself a decision, and it forecloses the most.
Filed as **item 5.0** in `ROADMAP-SEQUENCE.md`, ahead of the rename.

### 5.2 ⭐ The wedge, and why it is unusually clean here

stoke's pitch to an ISV is **a statically-linked 1.38 MB binary with a frozen ABI
and no dependencies.** Static linking is precisely what copyleft restricts.

> **The technical selling point and the monetisation wedge are THE SAME
> PROPERTY.** That is rare, and it is worth not giving away by accident.

### 5.3 The three structures

| structure | adoption | Path 1 revenue | notes |
|---|---|---|---|
| **Permissive (MIT / Apache-2.0)** | **maximum** | **none — gone forever** | Right if value capture is entirely upstack (§4.1). Apache-2.0 over MIT for the patent grant. ⚠ Irreversible in practice once contributions land |
| **LGPL-style split** (free for dynamic linking, commercial licence for static linking / modification) | high | **yes** | **The Qt model, and the closest fit to stoke's shape.** Dynamic linking stays free so hobbyists and OSS are unaffected; the ISV who needs the single static binary — which is the whole selling point — buys a licence. ⚠ Adds friction and a licensing-question inbox |
| **AGPL + commercial** | lower | yes, aggressively | Too blunt for an embeddable library: it scares away the OSS adoption that produces the inbound in the first place |

### 5.4 Recommendation

**If the answer to §4 is "Path 1 is the first revenue" ⇒ the LGPL-style split.
If the answer is "value capture is entirely upstack in AySen" ⇒ Apache-2.0.**

⭐ **SHARPENED 2026-09-01 BY THE AySen-SECONDARY DECISION (§0.1, §4.2): the
second branch is no longer available in the near term.** Value capture upstack
requires AySen, and AySen is now the secondary project. **So the expected
eventual answer has moved toward the split — which makes PRESERVING THE OPTION
the single highest-value cheap action in this entire document.**

⇒ **The recommendation below is unchanged in form and stronger in force. Do not
read "publish permissive" as "publish permissive and forget it" — the CLA is the
load-bearing half, and it is now the half that protects the only commercial
asset the project currently has.**

**And if the answer is genuinely undecided — which it reasonably is, because
task 1.1 has not run and nobody has seen who shows up — then take the option that
preserves both: publish under a permissive licence with a lightweight CLA on
inbound contributions.** The CLA costs contributors almost nothing, is standard
practice, and **keeps the dual-licence door open for the two-to-three years it
takes to find out which path is real.** That is the cheapest possible insurance
on the only irreversible decision in this document.

---

## 6. The staged path, with decision gates

### Stage A — finish and publish *(now → Phase 1 close)*

Execute `ROADMAP-SEQUENCE.md` Parts II–III. **Decide §5 and the rename before
the repo is public.** Publish with the README that concedes the matrix and claims
one cell (Part V of that file).

**Gate out of A — three questions, all answerable with evidence:**
1. Does stoke beat a **live** llama-server and ollama-default at n=16? *(task 1.1)*
2. Does it run a second model family? *(B5)*
3. Did anyone install it?

### Stage B — become installable *(Phase 1.5 + Phase 2)*

**ARM first (§7)**, then CUDA. Then the bridge layer — `stoke pull`, alias
registry, idle load/unload, per-model templates, `/metrics`. **This is what turns
an engine into a thing people install**, and the Fable review already re-ordered
it ahead of NPU work for exactly that reason.

**Instrument who shows up.** ISVs asking licensing questions ⇒ Path 1. Devs
filing feature requests ⇒ Path 4. Vendor engineers benchmarking ⇒ Path 2.
**Do not choose the business model in advance — read it off the inbox.**

**Gate out of B:** is there a repeat, non-Kabir user with a use case worth money?

### Stage C — pick one path and commit *(driven by B's evidence)*

Path 1 is the default expectation and the most solo-compatible. Path 2 is
opportunistic — take the meeting if it comes. Path 3 needs a team. Path 4 needs
the market to mature.

### Stage D — raise, bootstrap, or be acquired

Follows from C, and only from C. Note that Path 1 revenue **bootstraps** and does
not require raising; Paths 3 and 4 do.

### 6.1 The gate that outranks all of them

⚠ **`MEMORY.md`'s founding constraint is "not money right now — build something
proper and useful."** Every stage above is conditional on that constraint not
having changed. **If it has not changed, Stages C and D are optional and the
correct move is to keep building.** This document is a map of what becomes
possible, not an argument that it should be pursued now.

---

## 7. Phase 1.5 — ARM FIRST *(Kabir, 2026-09-01)*

> **"arm first, we'll make it embed on most of the hardware."**

### 7.1 Why this is the right call

1. **ARM is where embeddability lives.** Apple Silicon, Android, iOS, Raspberry
   Pi, Jetson, Snapdragon X laptops, Ampere/Graviton servers. Path 1's buyer is
   overwhelmingly shipping onto ARM.
2. **It is the largest population of "capable machine, no usable discrete GPU."**
   Apple Silicon Macs in particular: lots of fast unified memory, no CUDA, and a
   very high concentration of the developers who try local inference.
3. **CUDA is the incumbents' strongest ground** — llama.cpp and vLLM are both
   excellent there. Walking onto it first is fighting where they are strong.
4. **It contradicts the founding premise to go GPU-first.** The one-line is *"the
   hardware vLLM abandons."* CUDA is not that hardware.
5. **It proves the central claim of §1.2** — that the *scheduler* is the asset —
   by porting it to a different ISA without a GPU rewrite. That is a strictly more
   valuable demonstration than a CUDA backend, which proves only that kernels can
   be rewritten.

### 7.2 The port surface — verified 2026-09-01, and it is small

| finding | consequence |
|---|---|
| Only **`simd.c`**, **`model.c:21`** and **`pool.c:154`** gate on `__x86_64__` | the x86 assumption is **localized**, not diffuse |
| Threading is **pthreads** (`pthread_t`/`mutex`/`cond`), `LIBS := -lm -lpthread` | already portable to Linux, macOS, Android, iOS |
| `SIMD_CFLAGS` is scoped to `simd.c` alone | the rest of the tree is compiled without ISA flags **by design** |
| Runtime feature dispatch already exists (`kernels_cpu_has_avx_vnni()` gates the AVX2 call sites) | the pattern for NEON/`SDOT` detection is already in place |
| ⭐ **Every SIMD kernel has a scalar reference twin, and the twins are TESTED** | **an ARM build can be made CORRECT on day one and FAST incrementally** — the single most valuable fact in this section |

⇒ **Phase 1.5's ARM leg is a port, not a rewrite**, and the scalar twins mean it
has a working, gradeable intermediate state — which is exactly the position this
project's measurement discipline is built to exploit.

### 7.3 The technical question that decides the schedule

**ARMv8.2's `SDOT` is the direct counterpart to `vpdpbusd`** — it accumulates
four int8 products into an i32 lane, which is the same shape the Q4_K
lane-permutation kernel was built around. `SMMLA` (ARMv8.6 `i8mm`) goes further.

⭐ **So the ADDENDUM 44 insight is expected to transfer: put four different
sub-blocks in the lanes so the per-sub-block scales can all be applied at once.**

⚠⚠ **BUT THE OVERFLOW LADDER MUST BE RECOMPUTED, NOT INHERITED.** This is
exactly the trap that Q6_K-on-VNNI walked into: the ladder is a function of the
quantization's value range and the lane width, and getting it wrong produces
saturation rather than an error. **Recompute it from first principles for each
(instruction, quant) pair, and keep the named-falsifier discipline.**

### 7.4 The target matrix

| tier | targets | what is needed |
|---|---|---|
| **1 — correctness** | any ARM64: Apple Silicon, Pi 5, Android, Graviton | scalar twins + pthreads + a build. **Should be days, not weeks** |
| **2 — competitive** | ARMv8.2 `SDOT` (nearly all modern ARM64) | the K-quant dots in NEON, ladder recomputed |
| **3 — leading** | ARMv8.6 `i8mm`/`SMMLA`, Apple AMX-adjacent paths | matrix-multiply instructions; where the real headroom is |
| **4 — reach** | RISC-V vector, WASM SIMD128 | later; SIMD128 is interesting for browser embedding |

### 7.5 What "embed on most of the hardware" adds beyond kernels

Kabir's phrasing sets a broader goal than a NEON backend, and the planning doc
(`ROADMAP-SEQUENCE.md` 4.3) must cover all of it:

- **Build system.** Currently one Makefile assuming `gcc` and MSYS2/WSL. Embedding
  means **cross-compilation** (Android NDK, iOS toolchain, buildroot/Yocto) and
  probably a CMake path, because that is what ISVs expect to vendor.
- **Size budget.** 1.38 MB is a *feature*. **Make it a tracked number with a
  ceiling**, the way the perf numbers are tracked, or it will erode.
- **No-OS / constrained targets.** How much of the engine works without a
  filesystem or with a fixed memory budget? M4's block pool is already
  runtime-variable, which is most of the way there.
- **The ABI is the product here.** Path 1's buyer is buying `stoke.h`'s stability
  as much as its speed. The freeze discipline is a commercial asset — **say so in
  the README.**
- **Determinism and bit-exactness** — already a project invariant, and it is
  exactly what regulated and embedded buyers (Path 1, Path 3) require. It has
  been paid for; now it should be *marketed*.

---

## 8. Risks, ranked

| # | risk | severity | mitigation status |
|---|---|---|---|
| 1 | **No named buyer.** The plan ends at adoption, and adoption is not revenue (§3.3) | ⭐⭐⭐ | §4 names four; none validated. **Stage B's instrumentation is the mitigation** |
| 2 | **The lead is a lead, not a moat.** The scaling advantage is one good PR away from evaporating | ⭐⭐⭐ | Partly structural (§1.2 — the asset is the scheduler + ABI). **Argues for finishing Phase 1, not extending it** |
| 3 | **One model family, one ISA.** Below ~3–4 architectures, adoption is ~zero regardless of speed | ⭐⭐⭐ | B5 and §7. **Both are rungs, not research** |
| 4 | ~~**Solo capacity, and TWO unfinished projects**, the guardrail violated in practice~~ | ~~⭐⭐⭐~~ | ✅ **CLOSED 2026-09-01 by decision: stoke primary, AySen secondary (§0.1).** The guardrail is now enforced rather than aspirational. ⏭ Residual: what "secondary" means operationally is still one line of Kabir's to write |
| 4b | ⭐ **NEW, and it is Risk 4's replacement: with AySen secondary, stoke has NO ANCHOR TENANT.** The "there is a first demanding user by construction" argument (§4.1, and Part V.5 of `ROADMAP-SEQUENCE.md`) is deferred with it | ⭐⭐⭐ | **Unmitigated, and it PROMOTES §9.3 to the primary commercial risk.** stoke now needs real external users to validate the thesis, on a schedule nobody controls. Mitigations available: publish earlier, make §9.3 measurable *before* launch, and treat Path 1 inbound as the success metric rather than stars |
| 5 | **The headline claim is unmeasured** (§1.2 update 3) | ⭐⭐ | Task 1.1, scheduled |
| 6 | **The licence door closes silently** | ⭐⭐ | §5, filed as item 5.0. **Cheap to fix now, impossible later** |
| 7 | **The name is unsearchable** | ⭐⭐ | Item 5.1, undecided |
| 8 | **The CPU substrate is a shrinking share** | ⭐ | §3.4 — bet on the workload; NPUs are the next backend, not a threat |
| 9 | **OS vendors ship their own on-device runtimes** | ⭐ | They own the integrated slot; they do not offer openness, model choice, or cross-platform embedding |

---

## 9. What would falsify this plan

Stated in advance, in the project's own style, so that they are read as evidence
rather than explained away:

1. **Task 1.1 shows llama-server matching or beating stoke at n=16.** Then the
   scaling-shape claim — the entire product thesis in §3.2 — is wrong, and the
   honest move is to reposition on embeddability and the ABI alone.
2. **B5 shows the architecture assumptions are deeply Qwen-shaped**, making each
   new model family a milestone rather than a rung. Then the multi-model
   prerequisite in §8.3 becomes a multi-year problem and Path 1 dies.
3. **Public release produces no inbound after Stage B.** Then the underserved
   cell in §3.2 is underserved because nobody wants it, not because nobody built
   it — and that is the most likely quiet failure mode of the entire project.
4. **llama.cpp lands continuous-batching improvements that flatten the
   difference.** Then risk 2 has materialized and §4.1's "monetize what runs on
   it" becomes the only remaining strategy.

⭐ **Each of these is cheap to check and none is being checked yet.** Point 3 in
particular has no instrument at all — decide what "inbound" would be measured
with *before* publishing, or the result will be interpreted after the fact.

---

## 10. Open for Kabir

| # | question | why now |
|---|---|---|
| 1 | **The licence** (§5) — or at minimum, a CLA | **irreversible, and the clock starts at first public PR** |
| 2 | Is the **founding constraint still "not money right now"** (§6.1)? | Everything in §6 is conditional on the answer |
| 3 | ✅ ~~AySen Stage 2's timing~~ — **ANSWERED 2026-09-01: stoke primary, AySen secondary (§0.1).** ⏭ Residual: does "secondary" mean paused, maintenance-only, or worked in stoke's blocked windows? | One line. The blocked windows are already partly allocated (item 4.3) |
| 4 | **The name** (item 5.1) | Cheap now, steep after launch, and trademark matters if there is ever a company |
| 5 | Should the **acquisition path (Path 2) be planned for deliberately** rather than left as an accident? | It changes what gets optimized and what gets signed |
| 6 | What instrument measures **"did anyone install it"** (§9.3)? | Decide before publishing, not after |

---

*Companion documents: `ROADMAP.md` (the map — authoritative on every gate) ·
`ROADMAP-SEQUENCE.md` (the Phase-1 close-out queue; Parts IV and V hold the
performance forecast and the positioning analysis this file builds on) ·
`../MEMORY.md` (the master record — §7 the unified thesis, §Phased roadmap,
Appendix B the original decision journey) ·
`vllm-and-the-serving-landscape.md` (the pre-code market teardown that chose
this cell).*
