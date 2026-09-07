# 2026-09-07 — M7 ENTRY PROBE: gate report

## ⛔ STATUS: **PARTIAL — B5 IS NOT CLOSED BY THIS REPORT**

Everything below is real and everything below passes. It is **not the ruled
criterion**, and the gap is precise:

| | |
|---|---|
| streams banked | **8 of 16** |
| bands covered | `[0,100)`, `[100,300)`, `[300,600)` |
| bands **NOT covered** | **`[600,1200)` and `[1200,∞)`** |

§8.9 names those two bands *specifically* — they are the whole reason M7
re-freezes τ at all, because the rope theta term grows with position. **A grid
that stops at 448 has not tested the thing the bands exist to test**, however
quiet everything below is. The streams that carry them (`L-660`…`L-2851`) are
the expensive ones and the session ran out of time; they were still running when
this was written, and `m7_probe.py --phase a --only …` accumulates into the same
directory, so completing it loses nothing already done.

**Re-run `--phase b` and `--phase c` after the remaining streams land.** The τ
table below is frozen from an 8-stream grid and is superseded by the full one —
it is banked so the partial freeze is auditable, not so it can be reused.

Also outstanding: **C2, the null arm** (§1).

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

### C2 — the null arm

**STATUS: see §6.** Its job is the instrument's own noise floor: a second serial
run must reproduce the first bit-for-bit, or no Δ below that floor is readable.

---

## 2. THE GRID

**8 of 16 declared streams banked.**

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

**Total: 512 probes, 507 agreeing, 5 divergent (0.977%).**

Per band:

| band | probes | agreeing | divergent |
|---|---|---|---|
| `[0,100)` | 271 | 268 | 3 |
| `[100,300)` | 165 | 163 | 2 |
| `[300,600)` | 76 | 76 | 0 |

---

## 3. THE τ FREEZE

Pair key: **`f32_intcore_llama32_q4km`** — its own, never a Qwen3 or Q8_0 key (`classify.py`'s fatal fires on an ABSENT key, never on a present and wrong one).

τ(band) = max(0.10, 2 × band p95 of the anchored Δ-profile over AGREEING positions only), nearest-rank.

| band | agreeing positions | frozen τ |
|---|---|---|
| `[0,100)` | 268 | **0.418357** |
| `[100,300)` | 163 | **0.305446** |
| `[300,600)` | 76 | **0.290104** |

Freeze detail, as the tool printed it:

```
  [0,100): n=1315 max=0.526021 p95=0.209179 => tau=0.418357
  [100,300): n=797 max=0.306960 p95=0.152723 => tau=0.305446
  [300,600): n=374 max=0.421999 p95=0.145052 => tau=0.290104
```

Adjacent-band p95 growth (T4's 1.5× collapse test):

| transition | growth |
|---|---|
| [0,100)->[100,300) | 0.730× |
| [100,300)->[300,600) | 0.950× |

---

## 4. THE GRADING

Divergent positions graded: **5**

| verdict | n |
|---|---|
| ARITHMETIC-CLASS | 5 |

| stream | position | band | verdict | reason |
|---|---|---|---|---|
| `L-160` | 223 | `[100,300)` | **ARITHMETIC-CLASS** | worst margin 0.051472 < tau (0.305446) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-240` | 271 | `[100,300)` | **ARITHMETIC-CLASS** | worst margin 0.125896 < tau (0.305446) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-40` | 66 | `[0,100)` | **ARITHMETIC-CLASS** | worst margin 0.103031 < tau (0.418357) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-40` | 75 | `[0,100)` | **ARITHMETIC-CLASS** | worst margin 0.049315 < tau (0.418357) and the top-5 sets agree modulo the rank-5 tolerance |
| `S-a2` | 54 | `[0,100)` | **ARITHMETIC-CLASS** | worst margin 0.074510 < tau (0.418357) and the top-5 sets agree modulo the rank-5 tolerance |

**PASS RULE — zero BUG-class: PASS**

---

## 5. TRIPWIRES

| # | tripwire | threshold | measured | fired |
|---|---|---|---|---|
| T1 | divergence rate | > 2% | **0.977%** | no |
| T2 | any BUG-class | ≥ 1 | **0** | no |
| T3 | band with < 20 agreeing positions | any | none | no |
| T4 | adjacent-band p95 growth | > 1.5× | max 0.950x | no |
| T5 | serial-vs-batched control | ≥ 1e-3 | **1.907e-05** | no |
| T6 | serial null arm not bit-identical | any | *(see §1)* | |

---

## 6. What is NOT established here

- **Byte-exact agreement with llama.cpp.** Structurally unavailable, and *not*
  the criterion. 2026-09-06 measured why: stoke's own Windows and Linux builds
  produce different greedy text from identical source and weights, while every
  discrete-domain golden passes on both.
- **Anything about performance.** No perf number was taken here and none may be
  quoted from it.
- **The B6 (unsloth) rung.** Separate referent, separate rung.
- **C2, the null arm** — see §1. Until a second pinned serial run reproduces the
  first bit-for-bit, the reference's own reproducibility is asserted by its
  configuration rather than measured. The C1 and C3 numbers (~2e-5) bound how
  much this could matter, but they do not replace it.
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
