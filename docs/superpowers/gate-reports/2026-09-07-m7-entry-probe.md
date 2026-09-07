# 2026-09-07 — M7 ENTRY PROBE: gate report

## STATUS: **ALL FIVE RULED BANDS COVERED, ZERO BUG-CLASS — AND TWO THINGS ARE STILL OUTSTANDING**

| | |
|---|---|
| streams banked | **10 of 16 declared** |
| probes | **640** — 631 agreeing, 9 divergent (1.406%) |
| bands covered | **all five**: `[0,100)` `[100,300)` `[300,600)` `[600,1200)` `[1200,∞)` |
| τ frozen from | 268 / 163 / 76 / 62 / 62 agreeing positions — every band over T3's floor of 20 |
| verdicts | **9 ARITHMETIC-CLASS, 0 BUG** |
| tripwires | T1 no · T2 no · T3 no · T4 no · T5 no |

⭐ **The substantive result: τ does NOT grow with position.**
0.418 → 0.305 → 0.290 → **0.277** → 0.310. §8.9 required this re-freeze on the
premise that it *might* — the `rope` theta accumulation is a position-growing
term and is the ruling's own named first suspect. Measured, over 640 probes out
to position 1371, **it does not show up**: the smallest τ in the table is the
`[600,1200)` band, and adjacent-band p95 growth peaks at 1.116× against T4's
1.5×. This is the first evidence in the project that the named suspect is not
live on this model.

### Still outstanding — B5 is NOT closed by this report

1. **Six declared streams were not run** (`L-520`, `L-900`, `L-1120`, `L-1900`,
   `L-2500`, `L-2851`). The grid is *less dense* than pre-registered, and the
   consequence is specific: **`[1200,∞)` is sampled only at its low end
   (1308–1371)**. The band is open-ended and the corpus reaches 2914; nothing
   here probes past 1371, so "τ does not grow" is established **to 1371, not to
   2914**. `m7_probe.py --phase a --only …` accumulates into the same directory,
   so finishing costs only the missing streams.
2. ~~C2, the null arm~~ — **DONE, and it passed bit-identically** (§1). The
   pinned reference has no noise floor of its own.
3. ⚠ **THE FOURTH TARGETED DELTA IS NOT IMPLEMENTED AT ALL — "header-format
   template #2".** §8.8 lists four silent-garbage deltas needing targeted tests:
   QK-norm presence gating ✅, the llama3 rope scaling formula ✅, the llama-bpe
   regex ✅, and **the Llama-3 chat template ⬜**. `src/tok.h:115` says so in as
   many words — *"Llama-3 header format is M7"* — and nothing has been built.
   Llama-3.2's vocab has neither `<|im_start|>` nor `<|im_end|>`, so
   `tok_has_chatml()` is false, `stoke_apply_chat_template` returns
   `STOKE_ERR_NO_TEMPLATE`, and `stoke serve` warns at startup and answers
   **501** on `/v1/chat/completions`. Raw completions are unaffected.

   **This is a documented refusal, not a bug — and it is still an unmet part of
   the criterion.** It is recorded here because a forward-pass grid can be
   perfect and leave this untouched: I checked the parity items and would have
   called B5 closed without looking at the template. That is
   [[fixture-can-be-blind-to-the-defect]] #14 exactly — an enumeration of what
   remains is scoped by which source you read it from.

   ⚠ And `tok.h` already names one trap for whoever builds it: **that template
   must prepend BOS itself.** `tok_apply_chatml`'s "no BOS" is a property of the
   ChatML gate, not of chat prompts — llama-bpe models default `add_bos_token`
   to true, so a Llama-3 header built on the ChatML assumption would silently
   omit BOS and diverge at position 0. **The file confirms it: the template's
   first statement is `{{- bos_token }}`.**

   ⭐ **Read out of the GGUF rather than assumed — and it is NOT "ChatML with
   different tags".** `tokenizer.chat_template` is 3,827 bytes of Jinja2. Turns
   are `<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>`, but
   a system block is emitted **unconditionally**, even with no system message,
   and it carries a fixed preamble:

   ```
   Cutting Knowledge Date: December 2023
   Today Date: {date_string}
   ```

   ⚠⚠ **`date_string` is `strftime_now("%d %b %Y")` — TODAY'S DATE**, falling
   back to `"26 Jul 2024"` when the renderer has no clock. So a faithful
   implementation **emits a different prompt every day**, and any golden pinning
   its output breaks at midnight unless the date is pinned deliberately. That is
   a design decision to take *before* writing it, not a bug to find afterwards.
   (The template also has tool-calling and `tools_in_user_message` branches; the
   Qwen-equivalent subset is system + user/assistant turns.)

---

The ruled ref-vs-stoke probe grid for Llama-3.2-3B, and the τ freeze it exists
to produce.

**Protocol:** `docs/superpowers/design-notes/2026-09-07-m7-entry-probe-preregistration.md`,
committed in `9310932` **before any probe was compared**.
**Binding criterion:** `docs/ROADMAP.md:619-627` re-instantiating the M3 RULING
§2.3 / §6 / §8.8 / §8.9.

⚠ **The oracle is not consulted anywhere in this report.** The two sides are
`ref_forward.py` (f32 numpy) and the stoke binary.

---

## 0. Provenance

| | |
|---|---|
| model | `models/Llama-3.2-3B-Instruct-Q4_K_M.gguf`, 2,019,377,696 B, sha256-verified against HF's LFS oid on 2026-09-05 |
| arch read from the file | `llama`, 28 layers, n_embd 3072, n_ff 8192, 24 heads / 8 kv (**G=3**), head_dim 128, rope freq_base 500000, eps 1e-5, vocab 128256 |
| tensors | `rope_freqs.weight` F32[64] present; **no** `attn_q_norm`/`attn_k_norm`; **no** `output.weight` ⇒ tied lm_head |
| stoke binary | `build/win/stoke.exe`, sha256 **`6a097f55fcea5f716e29462cdcc26da4be1e6f6156ede080f6f995ea64172533`**, 1,408,461 B |
| ⚠ binary staleness, checked not assumed | `make -n` reports it relinkable, because `build/win/model.o` (09:47:44) is newer than the exe (09:45:29) — that object was **recompiled from the RESTORED source after the mutation cycle**, from the same clean text that produced the linked one. `git status src/` is **empty at HEAD**, so the source behind this binary is the committed source. Recorded rather than left as an unexplained flag: rule 4's class is a stale artifact that looks exactly like a fresh one |
| reference | `tests/tooling/ref_forward.py` @ `3685f13`, numpy 2.4.0 / scipy-openblas 0.3.30 |
| pinning | every reference run: affinity `0xFFF` (P-cores + siblings), `OPENBLAS_NUM_THREADS=6`, verified by the launcher and **recorded in each probe file's `env` block** |

---

## 1. THE CONTROLS — the instrument, before anything it measures

### C1 — serial vs batched, and it LICENSES the batched arm

`serial(250)` vs `batched(250)` over the same 250 ids, both pinned:

| | |
|---|---|
| common positions | 250 (0..249) |
| argmax disagreements | **0** |
| bit-identical top-5 | 0 / 250 |
| **anchored \|Δ\| max** | **1.907e-05** |
| anchored \|Δ\| p95 | 1.240e-05 |
| raw \|Δ\| max | 1.335e-05 |
| trend by fifth | 1.24e-5 · 1.34e-5 · 1.91e-5 · 1.34e-5 · 1.62e-5 — **flat** |
| cost | 9628.5 s serial vs **65.1 s** batched = **148×** |

**T5 refuses at ≥ 1e-3. Measured 1.907e-05 ⇒ PASS, 52× inside the threshold**,
and almost exactly the M3 precedent (ref vs the M2 binary agreed at 1.4e-5
against a smallest frozen τ of 0.394898).

0/250 bit-identical is the *expected* shape, not a warning: the graph is
identical and the GEMM reduction order is not.

⚠ **Deviation from the pre-registration, stated rather than absorbed:** C1 was
declared on `short1` (69 positions) and run on the 250-position `long2` prefix.
That is a strengthening — more positions, deeper into the text — but it is a
deviation.

### C3 — batch-independence, and MY PREDICTION FAILED

The pre-registration said batched(T=2915) and batched(T=250) **"MUST agree"** at
shared positions, because causal attention makes position *p* depend only on
ids[0..*p*].

**The dependence argument is right. The bit-identity conclusion was wrong** — the
one thing that had to differ, the GEMM shape, is exactly the freedom BLAS uses
to choose a reduction order.

| comparison | argmax diffs | bit-identical | anchored \|Δ\| max |
|---|---|---|---|
| T=2851 vs T=250, positions 0–249 | 0 | 0 / 250 | 2.003e-05 |
| T=2851 vs T=1200, positions 0–1199 | 0 | 3 / 1200 | 1.907e-05 |

trend to position 1199: 1.62e-5 · 1.72e-5 · 1.91e-5 · 1.53e-5 · 1.43e-5 — **flat**.

⇒ **the batched answer depends on the batch it was computed in, at ~2e-5.** The
pre-registration already ruled what that means: *"the size of that dependence is
the honest bound on the long bands and it goes in the report as such."* It is the
same envelope C1 measured, and it does not grow out to position 1199 — which
**extends** C1's licence rather than undermining it.

**The prediction stays falsified.** It is not re-stated as "agree to within
2e-5", which is what fitting the band to the result would look like.

⚠ C3 is a **self-consistency** check. It cannot detect an error the batched path
makes at every T; that is C1's job, and C1's reach — 250 positions — remains the
stated limit of the independent evidence.

### C2 — the null arm: **BIT-IDENTICAL, and it makes the other two readable**

A second pinned serial run, over a *different-length* stream (12 ids vs 250),
compared at the 12 shared positions with the id prefix verified element by
element:

| | |
|---|---|
| common positions | 12 (0..11), ids verified identical |
| argmax disagreements | **0** |
| bit-identical top-5 | **12 / 12** |
| anchored \|Δ\| max | **0.0 — exactly zero** |

**The pinned reference has NO noise floor of its own.** That is the result the
null arm exists to produce, and it is what makes C1 and C3 mean anything: their
~2e-5 is *entirely* attributable to the difference under test (serial-vs-batched,
and batch shape), with nothing hiding underneath from run-to-run variation.

⚠ It also settles the question Kabir's P-core catch raised. Unpinned, OpenBLAS
sizes itself to all 16 logical CPUs and splits every reduction across them, so
the last bits of every logit depended on how many threads it happened to pick.
Pinned to `0xFFF` with `OPENBLAS_NUM_THREADS=6`, the reference is **exactly
reproducible** — a property that had been asserted by configuration and is now
measured.

⚠ Scope, stated: 12 positions in `[0,100)`. Reproducibility is a property that
shows at every position if it fails at all, so this does not need many — but it
is 12, not 250, and it was run short deliberately for time.

---

## 2. THE GRID

**10 of 16 declared streams banked.**

| stream | target | realized n_prompt | probes | absolute positions | bands touched | agree | selected-on-success |
|---|---|---|---|---|---|---|---|
| `S-a1` | full | 5 | 64 | 4–67 | [0,100) | 64/64 | YES |
| `S-a2` | full | 12 | 64 | 11–74 | [0,100) | 63/64 | YES |
| `S-a3` | full | 17 | 64 | 16–79 | [0,100) | 64/64 | YES |
| `L-40` | 40 | 39 | 64 | 38–101 | [0,100), [100,300) | 62/64 | no |
| `L-80` | 80 | 84 | 64 | 83–146 | [0,100), [100,300) | 64/64 | no |
| `L-160` | 160 | 171 | 64 | 170–233 | [100,300) | 63/64 | no |
| `L-240` | 240 | 249 | 64 | 248–311 | [100,300), [300,600) | 63/64 | no |
| `L-380` | 380 | 386 | 64 | 385–448 | [300,600) | 64/64 | no |
| `L-660` | 660 | 671 | 64 | 670–733 | [600,1200) | 62/64 | no |
| `L-1300` | 1300 | 1309 | 64 | 1308–1371 | [1200,inf) | 62/64 | no |

**Total: 640 probes, 631 agreeing, 9 divergent (1.406%).**

Per band:

| band | probes | agreeing | divergent |
|---|---|---|---|
| `[0,100)` | 271 | 268 | 3 |
| `[100,300)` | 165 | 163 | 2 |
| `[300,600)` | 76 | 76 | 0 |
| `[600,1200)` | 64 | 62 | 2 |
| `[1200,inf)` | 64 | 62 | 2 |

---

## 3. THE τ FREEZE

Pair key: **`f32_intcore_llama32_q4km`** — its own, never a Qwen3 or Q8_0 key (`classify.py`'s fatal fires on an ABSENT key, never on a present and wrong one).

τ(band) = max(0.10, 2 × band p95 of the anchored Δ-profile over AGREEING positions only), nearest-rank.

| band | agreeing positions | frozen τ |
|---|---|---|
| `[0,100)` | 268 | **0.418357** |
| `[100,300)` | 163 | **0.305446** |
| `[300,600)` | 76 | **0.290104** |
| `[600,1200)` | 62 | **0.277461** |
| `[1200,inf)` | 62 | **0.309627** |

Freeze detail, as the tool printed it:

```
  [0,100): n=1315 max=0.526021 p95=0.209179 => tau=0.418357
  [100,300): n=797 max=0.306960 p95=0.152723 => tau=0.305446
  [1200,inf): n=299 max=0.330494 p95=0.154814 => tau=0.309627
  [300,600): n=374 max=0.421999 p95=0.145052 => tau=0.290104
  [600,1200): n=307 max=0.236189 p95=0.138730 => tau=0.277461
```

Adjacent-band p95 growth (T4's 1.5× collapse test):

| transition | growth |
|---|---|
| [0,100)->[100,300) | 0.730× |
| [100,300)->[300,600) | 0.950× |
| [300,600)->[600,1200) | 0.956× |
| [600,1200)->[1200,inf) | 1.116× |

---

## 4. THE GRADING

Divergent positions graded: **9**

| verdict | n |
|---|---|
| ARITHMETIC-CLASS | 9 |

| stream | position | band | verdict | reason |
|---|---|---|---|---|
| `L-1300` | 1310 | `[1200,inf)` | **ARITHMETIC-CLASS** | worst margin 0.027637 < tau (0.309627) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-1300` | 1371 | `[1200,inf)` | **ARITHMETIC-CLASS** | worst margin 0.034638 < tau (0.309627) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-160` | 223 | `[100,300)` | **ARITHMETIC-CLASS** | worst margin 0.051472 < tau (0.305446) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-240` | 271 | `[100,300)` | **ARITHMETIC-CLASS** | worst margin 0.125896 < tau (0.305446) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-40` | 66 | `[0,100)` | **ARITHMETIC-CLASS** | worst margin 0.103031 < tau (0.418357) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-40` | 75 | `[0,100)` | **ARITHMETIC-CLASS** | worst margin 0.049315 < tau (0.418357) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-660` | 704 | `[600,1200)` | **ARITHMETIC-CLASS** | worst margin 0.049232 < tau (0.277461) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-660` | 708 | `[600,1200)` | **ARITHMETIC-CLASS** | worst margin 0.106217 < tau (0.277461) and the top-5 sets agree modulo the rank-5 tolerance |
| `S-a2` | 54 | `[0,100)` | **ARITHMETIC-CLASS** | worst margin 0.074510 < tau (0.418357) and the top-5 sets agree modulo the rank-5 tolerance |

**PASS RULE — zero BUG-class: PASS**

---

## 5. TRIPWIRES

| # | tripwire | threshold | measured | fired |
|---|---|---|---|---|
| T1 | divergence rate | > 2% | **1.406%** | no |
| T2 | any BUG-class | ≥ 1 | **0** | no |
| T3 | band with < 20 agreeing positions | any | none | no |
| T4 | adjacent-band p95 growth | > 1.5× | max 1.116x | no |
| T5 | serial-vs-batched control | ≥ 1e-3 | **1.907e-05** | no |
| T6 | serial null arm not bit-identical | any | **bit-identical, max Δ = 0.0** | no |

---

## 6. What is NOT established here

- **Byte-exact agreement with llama.cpp.** Structurally unavailable, and *not*
  the criterion. 2026-09-06 measured why: stoke's own Windows and Linux builds
  produce different greedy text from identical source and weights, while every
  discrete-domain golden passes on both.
- **Anything about performance.** No perf number was taken here and none may be
  quoted from it.
- **The B6 (unsloth) rung.** Separate referent, separate rung.
- ~~C2, the null arm~~ — **closed**: bit-identical, max Δ exactly 0.0 (§1).
- **The `[600,1200)` and `[1200,∞)` bands, unless §2 shows streams in them.**
  §8.9 names those bands specifically; a grid that stops short of them has not
  met the criterion no matter how quiet the bands below are.

### The honest reading of a PASS here

That stoke's Llama-3.2-3B forward pass agrees with an *independent* f32
reference to within a tolerance measured from the pair's own agreeing behaviour
— and that where they differ, they differ the way two f32 implementations of one
graph differ, not the way a bug does.

⚠ And the standing limit on the reference itself: the batched arm is licensed by
C1 over 250 positions and by C3's flat ~2e-5 batch-dependence out to 1199. Both
are the *instrument's* envelope, three-to-four orders below every frozen τ. That
gap is the reason the classification is readable at all.
