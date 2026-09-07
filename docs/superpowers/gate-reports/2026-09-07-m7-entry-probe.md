# 2026-09-07 — M7 ENTRY PROBE: gate report

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
| stoke binary | `build/win/stoke.exe` @ `3685f13` — sha256 in §6 |
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

*(filled from `--phase a`)*

---

## 3. THE τ FREEZE

*(filled from `--phase b`, which writes the table BEFORE §4 grades anything)*

---

## 4. THE GRADING

*(filled from `--phase c`)*

---

## 5. TRIPWIRES

| # | tripwire | threshold | measured | fired |
|---|---|---|---|---|
| T1 | divergence rate | > 2% | | |
| T2 | any BUG-class | ≥ 1 | | |
| T3 | band with < 20 agreeing positions | any | | |
| T4 | adjacent-band p95 growth | > 1.5× | | |
| T5 | serial-vs-batched control | ≥ 1e-3 | **1.907e-05** | **no** |
| T6 | serial null arm not bit-identical | any | | |

---

## 6. What is NOT established here

*(filled at close)*
