# 2026-09-14/15 — B6, THE UNSLOTH FILE'S FORWARD HALF: gate report

## STATUS: **THE FULL GRID RAN. 15/15 STREAMS, ZERO BUG-CLASS, BOTH CONTROLS PASS — AND TWO TRIPWIRES FIRED. PASS ON THE PASS RULE; SIGNED OFF BY KABIR 2026-09-15 — M7 GATE ITEM [2] MET.**

| | |
|---|---|
| streams banked | **15 of 15 declared** |
| probes | **960** — 937 agreeing, 23 divergent (**2.396%**) |
| bands covered | **all five**, out to absolute position **2933** — the first Qwen3 probes ever in `[600,1200)` and `[1200,∞)` |
| τ frozen from | 274 / 154 / 73 / 186 / 250 agreeing positions — every band over T3's floor of 20 |
| verdicts | **23 ARITHMETIC-CLASS, 0 BUG, 0 ESCALATE** |
| tripwires | **T1 FIRED (2.396% > 2%)** · T2 no · T3 no · **T4 FIRED (1.969× > 1.5×)** · T5 no · T6 no |
| controls | C1-Q **PASS** (5.150e-05 vs 1e-3, 87/87 argmax) · T6 **BIT-IDENTICAL** 40/40 |
| T1 disposition | `docs/superpowers/memos/2026-09-15-m7-b6-tripwire-anomaly-memo.md` — threshold kept, FIRED reported, carried expectation confirmed (bartowski 3.10% · this file 2.40% · Llama 1.37%) |
| T4 disposition | §7 — the pre-registered rope-theta test **REFUTED** (excess removed 0.0007, 0 flips in 448), and two probes for the outlier that drives part of the rise, both **HELD** (stoke bit-identical across prefill paths; reference stable across batch shapes to 7e-5) |

### What a reader should take from this page

Stoke's forward pass on the file people actually download agrees with the
independent f32 reference at 937 of 960 probed positions out to 2,933 tokens,
and where the two chose different tokens they did so at near-ties, inside a
tolerance frozen from the pair's own agreeing behaviour before any of them
was graded. **No divergence is BUG-class.** That is the criterion, and it is
met.

Two detectors fired, and neither is a defect signal on its own terms:

- **T1** fired because this is a Q4_K_M Qwen file on the integer-core pair,
  which the pre-registration said would happen (§5: *"if this file reads near
  3% too, T1 fires and the memo says so"*). It read 2.40% — *below* the only
  other Qwen3 Q4_K_M rate on record (3.10%). The memo keeps the threshold.
- **T4** fired because the `[1200,∞)` envelope is 1.97× the band below it.
  Two things are in that number and §7 separates them: a **localized tail**
  at the first ~10 generated positions of the three ≥1893-token streams,
  carrying the largest anchored Δ ever seen on this pair family (**11.88 at
  `L-1900` position 1893**, at an *agreeing* position), and a genuine widening
  of the envelope past ~1500 tokens (per-stream p95 0.43 → 0.77 → 0.75 →
  0.95). The pre-registered rope-theta test runs against the second; two
  cheap probes were pre-registered against the first, and both **held**:
  stoke is bit-identical across its two prefill paths at every probed
  position, and the reference is stable across batch shapes to 7e-5 at the
  outlier. So the outlier is a real difference between two self-consistent
  implementations that agree on the token. It is recorded, not explained.

---

The ruled ref-vs-stoke probe grid for the unsloth Qwen3-4B-Instruct-2507
Q4_K_M file, and the τ freeze it exists to produce.

**Protocol:** `docs/superpowers/design-notes/2026-09-14-m7-b6-unsloth-probe-preregistration.md`,
committed in `b343ddd` **before any probe was compared**. Follow-ups:
`2026-09-15-m7-b6-t4-and-boundary-probes-preregistration.md` (`ec5451d`),
committed before any of them ran.
**Binding criterion:** M7 gate item [2] ("unsloth-file parity run recorded")
under the referent-per-file rule of the 08-06 ruling §1 and
`docs/ROADMAP.md:619-627` — never a 50/50 oracle bar.

⚠ **`llama.cpp` is not consulted anywhere in this report.** The two sides are
`ref_forward.py` (f32 numpy) and the stoke binary. ⚠ **And the bartowski file
is not compared to anywhere in it either** — the two Q4_K_M files carry
different weight bits (imatrix 497 vs 79 chunks); B6 is *stoke vs the
referent on the unsloth file*, never *do the two files agree*.

---

## 0. Provenance

| | |
|---|---|
| model | `models/unsloth-Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, 2,497,281,120 B, sha256 `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597` (banked in `preflight.json` on both launches) |
| arch read from the file | `qwen3`, 36 layers, n_embd 2560, n_ff 9728, 32 heads / 8 kv (**G=4** — the AVX2 attention arm runs, unlike on Llama-3.2), head_dim 128, rope freq_base **5e6**, eps 1e-6, vocab 151936, ctx 262144; `attn_q_norm`/`attn_k_norm` present (QK-norm), no `output.weight` ⇒ tied lm_head (`token_embd` Q6_K); `quantize.imatrix.chunks_count` 79; no `bos_token_id` key (the read that B5 fixed) |
| stoke binary | `build/win/stoke.exe`, sha256 **`57475862adf360fcbd5ae0713dd2df9b416d501d61ee2fc37008865752bea0e9`**, 1,442,711 B, built 2026-09-11 21:32:31 |
| binary vs source, checked not assumed | `mingw32-make -n` on HEAD `b343ddd`: *Nothing to be done* — every object is newer than its source; the only source commits after the build (`d35f8a2`) touched `stoke.c`/`pool.h` serve-default lines that were already in the tree when the binary was linked (`make test` ran on it before that commit). `git status src/` empty |
| reference | `tests/tooling/ref_forward.py` @ `dacbd6f` (unchanged since 09-07), numpy 2.4.0 / scipy-openblas 0.3.30; `qwen3` path = the M2/M3 referent of record (`rope_kind: neox`, `qk_norm: True`, `freq_factors: False`, `theta_mode: closed` in every probe file) |
| pinning | every reference run: affinity `0xFFF`, `OPENBLAS_NUM_THREADS=6`, verified by `run_ref_probe.ps1` and **recorded in each probe file's `env` block** |
| driver | `tests/tooling/run_b6_probe.sh build/b6`, launched as a script file (`b6_launch.sh`) twice: 2026-09-14 16:11 (steps 0–5, stopped on a step boundary at Kabir's request) and 2026-09-15 09:22 (steps 1–4 skipped on their banks, 5 re-run and reproduced `C1Q.json`/`T6.json` byte-for-byte, 6–8 run). Pre-flight: ullAvailPhys 9801 MB / 8783 MB against the ruled `model + 2 GB` = 4429 MB |
| banks | `build/b6/` (untracked): `grid/*.grid.json` ×15, `grid/*.ref.json` ×15, `tau.json`, `verdicts.json`, `C1Q.json`, `T6.json`, `P-00.{batched,serial}.all.json`, `P-00.serial.null40.json`, `driver.log`; copies beside this report as `2026-09-15-m7-b6-unsloth-probe-{tau,verdicts,C1Q,T6}.json` |

---

## 1. THE CONTROLS — the instrument, before anything it measures

### C1-Q — serial vs batched on the Qwen3 graph, and it LICENSES the batched arm

The 09-07 licence was earned on the `llama` architecture. The Qwen3 graph
differs (QK-norm before rope, NEOX rope), so the pre-registration required it
re-earned here: `serial(87)` vs `batched(87)` over `P-00`'s 23 prompt + 64
generated ids, every position probed, both pinned.

| | |
|---|---|
| common positions | 87 (0..86), ids verified identical |
| argmax disagreements | **0** |
| bit-identical top-5 | 0 / 87 (the expected shape: same graph, different GEMM reduction order) |
| **anchored \|Δ\| max** | **5.150e-05** |
| anchored \|Δ\| p95 | 2.670e-05 |
| raw \|Δ\| max | 3.624e-05 |
| trend by fifth | 1.53e-5 · 2.48e-5 · 2.29e-5 · 2.29e-5 · 5.15e-5 |
| cost | 3469.7 s serial vs **40.8 s** batched = **85×** |

**T5 refuses at ≥ 1e-3. Measured 5.150e-05 ⇒ PASS, 19× inside the
threshold** (Llama's C1 read 1.907e-05, 52× inside). The last fifth is 2×
the others — 87 positions is too few to call that a trend, and the
pre-registered rule is a maximum, which passes.

### T6 — the null arm: **BIT-IDENTICAL, and it makes C1-Q readable**

A second pinned serial run over positions 0..39 of the same stream:

| | |
|---|---|
| common positions | 40 (0..39), ids verified identical |
| argmax disagreements | **0** |
| bit-identical top-5 | **40 / 40** |
| anchored \|Δ\| max | **0.0 — exactly zero** |

**The pinned serial reference has NO noise floor on this pair.** C1-Q's
5e-5 is therefore entirely the serial-vs-batched difference.

⚠ Two facts about cost, recorded because the pre-registration priced them
wrong: the serial reference is effectively single-threaded (~1 CPU-s/s under
`OPENBLAS_NUM_THREADS=6`) and walks ALL 87 positions even when only 0..39
are probed, so the null arm cost the same 58 min as the full serial run —
"~1 h for two runs of p00" was ~2 h. And `P-00.serial.null40.json`'s
`seconds` (16154.44) includes a 3.5 h `NtSuspendProcess` pause taken at
Kabir's request; nothing graded depends on it (grading is logits), and the
resume picked up mid-position at the old pace.

---

## 2. THE GRID

**15 of 15 declared streams banked.**

| stream | target | realized n_prompt | probes | absolute positions | bands touched | agree | selected-on-success |
|---|---|---|---|---|---|---|---|
| `P-00` | full | 23 | 64 | 22–85 | [0,100) | 63/64 | no |
| `P-10` | full | 24 | 64 | 23–86 | [0,100) | 64/64 | no |
| `P-20` | full | 64 | 64 | 63–126 | [0,100), [100,300) | 64/64 | no |
| `P-30` | full | 53 | 64 | 52–115 | [0,100), [100,300) | 62/64 | no |
| `P-45` | full | 14 | 64 | 13–76 | [0,100) | 63/64 | no |
| `P-41` | full | 477 | 64 | 476–539 | [300,600) | 62/64 | no |
| `L-160` | 160 | 170 | 64 | 169–232 | [100,300) | 61/64 | no |
| `L-240` | 240 | 248 | 64 | 247–310 | [100,300), [300,600) | 62/64 | no |
| `L-660` | 660 | 667 | 64 | 666–729 | [600,1200) | 63/64 | no |
| `L-900` | 900 | 920 | 64 | 919–982 | [600,1200) | 60/64 | no |
| `L-1120` | 1120 | 1121 | 64 | 1120–1183 | [600,1200) | 63/64 | no |
| `L-1300` | 1300 | 1307 | 64 | 1306–1369 | [1200,inf) | 62/64 | no |
| `L-1900` | 1900 | 1893 | 64 | 1892–1955 | [1200,inf) | 63/64 | no |
| `L-2500` | 2500 | 2503 | 64 | 2502–2565 | [1200,inf) | 64/64 | no |
| `L-2851` | full | 2871 | 64 | 2870–2933 | [1200,inf) | 61/64 | no |

**Total: 960 probes, 937 agreeing, 23 divergent (2.396%).**

Per band:

| band | probes | agreeing | divergent |
|---|---|---|---|
| `[0,100)` | 277 | 274 | 3 |
| `[100,300)` | 160 | 154 | 6 |
| `[300,600)` | 75 | 73 | 2 |
| `[600,1200)` | 192 | 186 | 6 |
| `[1200,inf)` | 256 | 250 | 6 |

Every stream landed in the band it was declared for (§3.3 of the
pre-registration); Qwen's tokenizer put `L-2851` at 2871 tokens against
Llama's 2851, and `L-1900` at 1893. `[300,600)` is the thinnest band at 73
agreeing positions — over T3's 20, but it is fed only by `P-41` and the tail
of `L-240`; a fourth stream there would be cheap.

---

## 3. THE τ FREEZE

Pair key: **`f32_intcore_unsloth_q4km`** — its own, never another file's key (`classify.py`'s fatal fires on an ABSENT key, never on a present and wrong one).

τ(band) = max(0.10, 2 × band p95 of the anchored Δ-profile over AGREEING positions only), nearest-rank.

| band | agreeing positions | frozen τ |
|---|---|---|
| `[0,100)` | 274 | **0.990144** |
| `[100,300)` | 154 | **0.901143** |
| `[300,600)` | 73 | **0.883762** |
| `[600,1200)` | 186 | **0.772259** |
| `[1200,inf)` | 250 | **1.520234** |

The three M3 bands land where the bartowski pair's did (M3 froze 1.023739 /
1.178094 / 0.811432 for the same three bands on the other Q4_K_M file) — the
pair's noise is a property of the integer core against f32, not of which
imatrix quantized the weights. The two new bands are the first τ ever frozen
for a Qwen3 file past 600 tokens.

⚠ **The outlier, named rather than smoothed:** the largest anchored Δ
anywhere is **11.877 at `L-1900` position 1893**, an *agreeing* position
(both sides chose id 387; stoke's top logit 37.27, the reference's 50.03),
fifteen times that band's p95 of 0.760 — and twelve times the 0.98 outlier
the 09-07 report named on Llama. It is inside the envelope the freeze is
built from, so it widens τ rather than being hidden by it; every decision
in `[1200,∞)` is graded against a τ that this one position helped set. §7
records what was done about it, and what was not.

Freeze detail, as the tool printed it:

```
  [0,100): n=1334 max=1.183546 p95=0.495072 => tau=0.990144
  [100,300): n=750 max=1.901163 p95=0.450571 => tau=0.901143
  [1200,inf): n=1198 max=11.877276 p95=0.760117 => tau=1.520234
  [300,600): n=347 max=0.801913 p95=0.441881 => tau=0.883762
  [600,1200): n=901 max=0.864138 p95=0.386129 => tau=0.772259
```

Adjacent-band p95 growth (T4's 1.5× collapse test):

| transition | growth |
|---|---|
| [0,100)->[100,300) | 0.910× |
| [100,300)->[300,600) | 0.981× |
| [300,600)->[600,1200) | 0.874× |
| [600,1200)->[1200,inf) | 1.969× |

τ is **flat-to-falling from 0 to 1200** and then **rises 1.969× into
`[1200,∞)`**. T4 fires; per-band τ is kept (it is the more conservative
table and the one the grading used); the rise is examined in §7.

---

## 4. THE GRADING

Divergent positions graded: **23**

| verdict | n |
|---|---|
| ARITHMETIC-CLASS | 23 |

| stream | position | band | verdict | reason |
|---|---|---|---|---|
| `L-1120` | 1150 | `[600,1200)` | **ARITHMETIC-CLASS** | worst margin 0.165001 < tau (0.772259) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-1300` | 1320 | `[1200,inf)` | **ARITHMETIC-CLASS** | worst margin 0.571283 < tau (1.520234) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-1300` | 1323 | `[1200,inf)` | **ARITHMETIC-CLASS** | worst margin 0.617880 < tau (1.520234) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-160` | 202 | `[100,300)` | **ARITHMETIC-CLASS** | worst margin 0.244226 < tau (0.901143) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-160` | 222 | `[100,300)` | **ARITHMETIC-CLASS** | worst margin 0.222361 < tau (0.901143) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-160` | 229 | `[100,300)` | **ARITHMETIC-CLASS** | worst margin 0.116545 < tau (0.901143) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-1900` | 1925 | `[1200,inf)` | **ARITHMETIC-CLASS** | worst margin 0.146981 < tau (1.520234) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-240` | 252 | `[100,300)` | **ARITHMETIC-CLASS** | worst margin 0.288332 < tau (0.901143) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-240` | 277 | `[100,300)` | **ARITHMETIC-CLASS** | worst margin 0.364412 < tau (0.901143) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-2851` | 2896 | `[1200,inf)` | **ARITHMETIC-CLASS** | worst margin 0.683926 < tau (1.520234) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-2851` | 2912 | `[1200,inf)` | **ARITHMETIC-CLASS** | worst margin 0.454750 < tau (1.520234) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-2851` | 2921 | `[1200,inf)` | **ARITHMETIC-CLASS** | worst margin 0.208557 < tau (1.520234) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-660` | 723 | `[600,1200)` | **ARITHMETIC-CLASS** | worst margin 0.144648 < tau (0.772259) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-900` | 922 | `[600,1200)` | **ARITHMETIC-CLASS** | worst margin 0.043762 < tau (0.772259) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-900` | 961 | `[600,1200)` | **ARITHMETIC-CLASS** | worst margin 0.336329 < tau (0.772259) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-900` | 969 | `[600,1200)` | **ARITHMETIC-CLASS** | worst margin 0.155037 < tau (0.772259) and the top-5 sets agree modulo the rank-5 tolerance |
| `L-900` | 975 | `[600,1200)` | **ARITHMETIC-CLASS** | worst margin 0.098190 < tau (0.772259) and the top-5 sets agree modulo the rank-5 tolerance |
| `P-00` | 31 | `[0,100)` | **ARITHMETIC-CLASS** | worst margin 0.224592 < tau (0.990144) and the top-5 sets agree modulo the rank-5 tolerance |
| `P-30` | 52 | `[0,100)` | **ARITHMETIC-CLASS** | worst margin 0.193785 < tau (0.990144) and the top-5 sets agree modulo the rank-5 tolerance |
| `P-30` | 114 | `[100,300)` | **ARITHMETIC-CLASS** | worst margin 0.238165 < tau (0.901143) and the top-5 sets agree modulo the rank-5 tolerance |
| `P-41` | 485 | `[300,600)` | **ARITHMETIC-CLASS** | worst margin 0.126055 < tau (0.883762) and the top-5 sets agree modulo the rank-5 tolerance |
| `P-41` | 534 | `[300,600)` | **ARITHMETIC-CLASS** | worst margin 0.150946 < tau (0.883762) and the top-5 sets agree modulo the rank-5 tolerance |
| `P-45` | 29 | `[0,100)` | **ARITHMETIC-CLASS** | worst margin 0.310762 < tau (0.990144) and the top-5 sets agree modulo the rank-5 tolerance |

**PASS RULE — zero BUG-class: PASS**

The worst margin/τ ratio anywhere is 45% (`L-2851` 2896); M3's grid read 54%.

---

## 5. TRIPWIRES

| # | tripwire | threshold | measured | fired |
|---|---|---|---|---|
| T1 | divergence rate | > 2% | **2.396%** | **YES** — memo `2026-09-15-m7-b6-tripwire-anomaly-memo.md`: threshold kept, the carried expectation confirmed (bartowski 3.10% · this file 2.40% · Llama 1.37%), no concentration by band, stream or margin |
| T2 | any BUG-class | ≥ 1 | **0** | no |
| T3 | band with < 20 agreeing positions | any | none | no |
| T4 | adjacent-band p95 growth | > 1.5× | max 1.969x | **YES** — §7 |
| T5 | serial-vs-batched control | ≥ 0.001 anchored, or any argmax disagreement | **5.150e-05** over 87 positions, 0 argmax disagreements | no |
| T6 | serial null arm not bit-identical | any | 40/40 bit-identical, 0 argmax disagreements | no |

---

## 6. What is NOT established here

- **Byte-exact agreement with llama.cpp.** Structurally unavailable and not
  the criterion (2026-09-06).
- **Agreement between the two Q4_K_M files.** They cannot agree; their weights
  differ. Nothing here compares them.
- **Anything about performance.** No perf number was taken and none may be
  quoted from this run.
- **Why the `[1200,∞)` envelope is 1.97× the band below.** §7 says what it is
  *not* (a stoke prefill-path inconsistency; a reference batch-shape artifact;
  the rope-theta accumulation — see §7A) and names the arm that could say what
  it is.
- **The `[300,600)` band beyond 73 agreeing positions.** Over the floor, but
  thin; it grades two divergences here, both at < 18% of τ.

### The honest reading of a PASS here

That stoke's forward pass on the unsloth file agrees with an independent f32
reference of the same file, out past 2,900 tokens, to within a tolerance
measured from the pair's own agreeing behaviour — and that where the two
differ, they differ as two f32 implementations of one graph do, not as a bug
does. The batched arm is licensed by C1-Q over 87 positions on this graph and
by the T=1895-vs-1957 check at the outlier (§7C); both are the *instrument's*
envelope, four orders below every frozen τ.

---

## 7. THE FOLLOW-UPS — pre-registered 2026-09-15 (`ec5451d`), run after

Three probes, each with a prediction and a falsifier written down before it
ran. The full protocol is
`docs/superpowers/design-notes/2026-09-15-m7-b6-t4-and-boundary-probes-preregistration.md`.

### 7B. Stoke against itself across the prefill→decode boundary — **HELD**

`L-1900` re-run under `STOKE_PREFILL_CHUNK=1` (every prompt position through
`model_forward_position`, the decode path) against the banked run (chunks of
128 through `model_forward_batch`), same binary, same prompt file:

| | default (128), re-run | chunk 1 |
|---|---|---|
| prefill 1893 tok | 37,884 ms | **153,458 ms** (4.05× — the other path really ran) |
| generated ids vs banked | identical | identical |
| top-5 ids at 64/64 positions | identical | identical |
| top-5 logits at 64/64 positions | **max \|Δ\| 0.000000** | **max \|Δ\| 0.000000** |

**Prediction held: bit-identical.** E3's design claim — chunked prefill writes
the same KV as position-by-position — holds at 1893 real tokens on this
graph, and the default-path re-run doubles as a determinism null for stoke
(same binary, two runs, bit-identical). The 11.88 position is **not** a stoke
prefill/decode inconsistency. Banked: `build/b6/probes/L-1900.chunk{128,1}.*`.

### 7C. The reference against itself at the boundary (C3 on this pair) — **HELD**

The batched reference at **T = 1895** over the first 1895 ids of `L-1900`'s
stream, probing 1892–1894, against the banked T = 1957 run:

| position | argmax T=1957 / T=1895 | shared top-5 | max \|Δ\| |
|---|---|---|---|
| 1892 | 1410 / 1410 | 5/5 | 2.480e-05 |
| **1893** | 387 / 387 | 5/5 | **4.578e-05** (top logit 50.0333 / 50.0334) |
| 1894 | 8947 / 8947 | 5/5 | 7.248e-05 |

**Prediction held: < 1e-3.** The reference's answer at the outlier does not
depend on the batch it was computed in beyond the ~2e-5 GEMM-shape floor the
09-07 C3 measured on Llama. The 11.88 is **not** a reference batch-shape
artifact. Banked: `build/b6/probes/L-1900.T1895.*`.

### 7A. The named T4 test — rope-theta closed vs accum — **REFUTED**

The 09-08 protocol on this pair: phase A re-run on the seven streams that
feed the two bands, under both arms, same binary; graded by
`m7_theta_ab.py --set unsloth --bound-threshold banked` (artifact
`2026-09-15-m7-b6-rope-theta-ab.json` beside this report; raw arms under
`build/b6/theta/{closed,accum}`).

| band | arm | n | p95 | max |
|---|---|---:|---:|---:|
| `[600,1200)` | closed | 901 | **0.386129** | 0.864138 |
| `[600,1200)` | accum | 901 | **0.386173** | 0.864177 |
| `[1200,inf)` | closed | 1198 | **0.760117** | 11.877276 |
| `[1200,inf)` | accum | 1198 | **0.759936** | 11.877936 |

| | growth `[600,1200)`→`[1200,inf)` |
|---|---|
| closed (the ruled default, the freeze's arm) | **1.9686×** |
| accum (the diagnostic arm) | **1.9679×** |
| excess removed | **0.0007** |
| **verdict** | **REFUTED** (EXPLAINED ≥ 0.50, REFUTED < 0.10) |

Reference closed-vs-accum logit Δ (the bound):

| band | positions | median position | max \|Δ\| | p95 \|Δ\| | Δ/position, median … max |
|---|---:|---:|---:|---:|---|
| `[600,1200)` | 192 | 951 | 1.468658e-04 | 8.773804e-05 | 3.1e-08 … 1.9e-07 |
| `[1200,inf)` | 256 | 2502 | 1.342773e-03 | 4.463196e-04 | 4.3e-08 … 4.7e-07 |

Reference argmax flips between the arms: **0 of 448** probed positions.
2 × max Δ in `[1200,inf)` = **2.686e-03** against the **0.0935** refusal
threshold (a quarter of this pair's own banked rise, derived by the tool from
`tau.json`) ⇒ **CANNOT account for the rise**.

| # | tripwire | fired |
|---|---|---|
| R1 | stoke's own probes differ between arms | no — bit-identical across both arms' independent stoke runs |
| R2 | closed arm does not reproduce the banked p95/n | no — both bands reproduced exactly (n 901/1198, p95 to 1e-6) |
| R3 | null arm not bit-identical | no — `L-2851` banked vs re-run: 64 positions, 0 differing, max \|Δ\| **0.0** |
| R4 | the switch changed nothing | no — the accum arm moved every probed logit |
| R5 | a probe file's `theta_mode` is not its arm's | no |

**Prediction held.** The mechanism is real on this graph too — the
closed-vs-accum Δ is linear in position at ~3e-8 to 5e-7 logit/position,
about 3× Llama's — and it is **two orders of magnitude below the envelope**
(1.3e-3 against a p95 of 0.76). The 07-31 rope-theta suspect is discharged
for the Qwen3 file as it was for Llama; the 09-08 finding transfers.

R2 and R3 are worth more than their tripwire rows: the closed re-run
reproduced the banked freeze **exactly** from a second launch (stoke's side
bit-identical, the batched reference bit-identical at fixed T on the Qwen3
graph), which is what makes the 0.0007 attributable to the switch and
nothing else.

### 7D. What the three say together

Row 1 of the pre-registration's table: **stoke is self-consistent across its
two prefill paths (bit-identical), the reference is self-consistent across
batch shapes (7e-5), and at position 1893 they differ by 12.8 on the top
logit while choosing the same token.** By the ruling this is
ARITHMETIC-class — argmax agrees, and τ was frozen over it — and it is
recorded here as a finding, not explained.

One observation about its shape, offered as a hypothesis and nothing more:
at every outlier position stoke's large logits are *lower* than the
reference's, roughly in proportion to their size (1893: −12.8 / −5.3 / −3.9
/ −0.9 on 50 / 25 / 25 / 23; `L-2851` 2871: −4.4 / −3.7 / −2.2 / −0.9 on 34
/ 32 / 25 / 20). A rounding difference has no sign; a one-sided proportional
shrink is the shape block-wise activation quantization makes when one block
holds a massive activation and its 255 neighbours lose their resolution —
stoke's integer core quantizes activations to Q8_K, 256-wide blocks with one
scale, exactly as llama.cpp does. If that is the mechanism, llama.cpp's own
logit at position 1893 reads near stoke's 37, not the reference's 50, and
the gap is a property of the engine class rather than of stoke. **The
discriminator is llama.cpp's logits at these few positions** — the optional
survey the 08-05 ruling permits, a tooling task (`llama-perplexity`'s logit
dump or a small harness on `llama-cli`), and Kabir's call. It is the named
follow-up. Nothing in this report depends on its answer.
