# MEMO — 2026-07-20 — M2: oracle near-tie policy for the recorded parity run    [BLOCKING: the recorded gate run + any protocol change. NOT blocking: W3, W4 harness construction + dev-mode measurement]

**WHAT:** The M2 parity oracle (llama-server) computes in a different
numeric class than stoke's pure-f32 scalar pass, and at least one
characterized divergence exists that the §8 waiver protocol cannot absorb.
Decision needed on how the recorded 50-prompt gate run treats this class.
The KV-cache hypothesis from 07-20 is REFUTED by experiment; the gate's
oracle invocation stays DEFAULT (no `-ctk/-ctv` flags) — that question is
closed without you; this memo is about the divergence class it exposed.

**EVIDENCE (all mechanically recorded; artifacts committed under
`tests/tooling/validation/`, details in the session ledger):**

- f32-KV experiment: server rerun with `-ctk f32 -ctv f32` on all 3
  validation prompts, twice on v1 → deterministic, and **byte-identical
  to the default-KV captures everywhere**. KV type contributes zero
  token difference; the numeric gap is the oracle's activation
  quantization (it quantizes activations to Q8 blocks for integer dot
  products; not switchable by any flag).
- The one divergence: v1 ("Explain how photosynthesis works in one
  sentence."), position 32 of 64 — a genuine 3-way near-tie ("one/two/
  three sentences" continuation). Teacher-forcing the numpy ref on the
  oracle's own 42-token prefix reproduces it exactly:

  | rank | oracle id (logprob) | ref id (logit) |
  |---|---|---|
  | 1 | 2326 (−1.4927) | 825 (22.9451) |
  | 2 | 1378 (−1.5191) | 2326 (22.8749) |
  | 3 | 825 (−1.5204) | 1378 (22.8518) |
  | 4 | 264 (−1.9065) | 264 (22.4663) |
  | 5 | 4236 (−3.0839) | 4236 (21.3111) |

  Same five candidates, tail order identical, top-3 cluster permuted.
  Gaps(top1−top2): oracle **0.0264 nats**, ref **0.0702 nats** — both
  26–70× larger than the waiver ε (1e-3), and the top-2 pairs are not a
  clean swap, so §8 conditions 1 AND 2 both fail. This is not a bug on
  either side: v2 (code) and v3 (multilingual) reproduce the oracle
  **64/64 token-exact**, and every layer-level check agrees; the two
  implementations legitimately rank a ~2%-relative tie differently.
- Rate: 1 flip in 192 validation tokens (~0.5%/token as a single
  observation). Over the 50×64 = 3,200-token gate that extrapolates to
  anywhere from 0 to ~15 affected prompts — too uncertain to decide on;
  see the plan below, which measures it before you must choose.

**ALREADY IN MOTION (reversible; proceeding unless you object):** W3
(model.c) is dispatched — unaffected by this question. W4's harness will
be built and run in **dev mode only** across all 50 prompts to measure
the REAL affected-prompt count; you get that number appended here before
any recorded run. The recorded run itself waits for your call.

**OPTIONS for the recorded run:**
1. **Strict as written.** Any >ε divergence = gate fail → I compile the
   per-hit evidence package (both sides' top-5, L2 ladder proof that
   C == ref at every checkpoint for that prompt) and return to you.
   Zero protocol change; costs one extra round-trip if hits > 0.
2. **Bounded near-tie recognition (protocol change, M2 only):** a
   divergence counts as waiver-class iff (a) both sides' top-3 SETS are
   identical, (b) both sides' top1–top2 gaps < 0.1 nats, (c) the prompt
   is replaced from the reserve exactly like a §8 waiver, sharing the
   same ≤2 budget, every use fully recorded. Dies at M2 close.
3. **Re-scope M2's token gate to C-vs-ref (both pure f32, 50/50
   token-exact) + defer oracle token-parity to M3**, where the integer
   core puts stoke in the oracle's numeric class and the forward note
   already mandates a full parity re-pass. Keeps an absolute bar now;
   trades away oracle-independence for one milestone.

**RECOMMENDATION:** hold the recorded run until the dev-mode count is
appended here (hours, not days). Then: count = 0 → option 1 and done;
count 1–2 → option 2 (the reserve absorbs it, evidence recorded);
count ≥3 → option 3 (the class is systematic at M2's numeric distance;
grinding 50 prompts against a different-class oracle proves nothing the
M3 re-pass won't prove properly).

**IF DEFERRED:** M2 close waits on the recorded run; W3/W4 construction
and everything else proceeds. M3 is unaffected either way — its integer
core re-runs this exact gate in the oracle's own numeric class.
