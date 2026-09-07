# 2026-09-07 — M7 ENTRY PROBE: the pre-registration

**Written BEFORE any probe is compared, and that ordering is the point.** The
binding ruling requires τ to be frozen from this probe *before any parity
evidence is graded*; a corpus, a grid or a tripwire chosen after seeing results
is a bar fitted to the answer, which is the specific error this project has
already withdrawn results for.

Nothing in this file is a result. Results go in the gate report.

---

## 1. What binds this

`docs/ROADMAP.md:619-627` (Fable, 2026-08-05, binding), re-instantiating
`docs/superpowers/memos/2026-08-04-m3-gate-item-2-RULING-FABLE.md` §2.2/§2.3,
§6, §8.8 and §8.9:

> The tokenizer gate stays M1-style byte-exact via `llama-tokenize`. The
> FORWARD-PASS parity is the §M3 v2 SHAPE re-instantiated for Llama-3.2 —
> twins + ref probe grid + targeted tests for the four deltas — **never a 50/50
> live-oracle token bar for any model**; τ is re-frozen from M7's own entry
> probe with new position bands [600,1200)/[1200,∞) BEFORE any parity evidence
> is graded.

⚠ **The oracle does not appear anywhere in this protocol.** The two sides are
`ref_forward.py` (f32 numpy) and the stoke binary. `llama.cpp` is not consulted,
which is what keeps this out of the forbidden shape.

### Rung state going in

| half | ruled criterion | state |
|---|---|---|
| tokenizer | M1-style byte-exact via `llama-tokenize` | ✅ `tests/tok_llama_golden.sh`, 3 legs, landed 09-05 |
| twins | gate-[1] twins on every kernel the model exercises | ✅ both attention arms byte-identical at G=3; kernel twins carried from M3 |
| targeted delta tests | QK-norm gating, rope kind, `rope_freqs`, llama-bpe regex | ✅ landed 09-05, each mutation-tested |
| **ref probe grid** | §2.3 protocol over §8.9's bands | ⬜ **this document** |
| **τ freeze** | per band, from this probe, before grading | ⬜ **this document** |

`tests/llama_gen_golden.sh` is NOT part of this criterion and never was —
it is a regression detector, relabelled as one in its own header, and its
cases are selected on success.

---

## 2. The instrument, and the one thing about it that is not free

Landed in `835ba39`:

- `tests/tooling/ref_forward.py` implements `llama` as well as `qwen3` (rope
  kind dispatched on `general.architecture` from a transcription of llama.cpp's
  `llama_model_rope_type()`; QK-norm on tensor presence; optional
  length-checked `rope_freqs`; tied lm_head). 26 selftest checks, 5 mutations,
  5 caught.
- `--force-ids` gives it **teacher forcing**, which is what makes a probe at
  position 1400 mean anything on a stream that may have parted at 700.
- `stoke run --probe-all` reports the `--probe-pos` line format at every
  generated position. 3 mutations, 3 caught.
- `classify.py` gains bands `[600,1200)` and `[1200,∞)` under §8.9's named
  trigger. Adding the *range* adds no tolerance: no τ exists there until this
  probe freezes one, and `classify_records` still fatals on a band with no
  entry — asserted by a new selftest that hands it an M3-era table and a
  position at 1400.

### 2.1 ⚠⚠ The batched reference is a genuinely different arithmetic

MEASURED, on this box: the serial reference costs **~26–46 s per position** (it
re-dequantizes all 28 layers at every position — the deliberate M2 choice that
bounds the working set). §8.9 requires probes in `[1200,∞)`. At that rate one
2851-token stream is **≈32 hours**. The instrument, not the engine, was the
blocker.

`--mode batched` dequantizes each weight once and GEMMs the whole prefix.
Everything elementwise or per-row is asserted **bit-identical** to the scalar
kernel in `--selftest` (`rmsnorm_rows`, `softmax_rows`, `rope_thetas_matrix`).
What is left is the GEMM reduction order, which BLAS owns.

⚠ The timing figures above were taken **unpinned**, before the launcher existed.
They are a cost estimate that drove a design decision — a 2× error does not
change 32 h into anything usable — and they are **not** measurements of record.

**PRE-REGISTERED PREDICTION (serial-vs-batched control, `short1`, 69
positions):**

> Over every probed position: the argmax agrees on all 69, and the maximum
> anchored |Δ| between the serial and batched references is **< 1e-3**.
>
> **FALSIFIED IF** either fails. On falsification the batched arm is NOT
> licensed, this pre-registration is void, and the grid is redesigned — it is
> not re-run against a looser threshold.

1e-3 is two orders below the τ floor of 0.10. The M3 precedent for "the
instrument is not the signal" was three orders (ref vs the M2 binary agreed at
1.4e-5 against a smallest frozen τ of 0.394898), so ~1e-5 is what is expected
and 1e-3 is where refusal begins.

### 2.2 ⚠ Every run is pinned, and that is not bookkeeping

`numpy` here is `scipy-openblas 0.3.30`, `DYNAMIC_ARCH / NO_AFFINITY`. With no
thread count in the environment it sizes itself to all 16 logical CPUs and
splits each reduction across them, so **the last bits of every logit depend on
how many threads it picked** — and on this box an unpinned run also spreads
across two core types (0-11 are the six P-cores' siblings, 12-15 are E-cores).
A reference that cannot reproduce its own output cannot freeze a tolerance.

⇒ every probe in this protocol runs through
`tests/tooling/run_ref_probe.ps1`: affinity `0xFFF`, `OPENBLAS_NUM_THREADS=6`,
`OMP_NUM_THREADS=6`, and the launcher **throws** if the affinity request did
not take. `ref_forward.py` records what it observed into each probe file's
`env` block, so a banked file says what produced it.

**NULL ARM, REQUIRED:** the serial reference is run **twice** under identical
pinning before any comparison is believed. If the two runs are not
bit-identical, the instrument has a noise floor of its own and every Δ below
that floor is unreadable — that number goes in the gate report either way.

### 2.3 ⚠ The control does not reach the long bands, and a third arm covers why

**Stated because it is the weakest joint in this protocol.** The serial arm is
affordable to a few hundred positions, so C1 licenses the batched arm at
positions in `[0,300)` and says nothing directly about position 2900 — and the
mechanism by which a GEMM-order difference could grow with position is precisely
that the GEMM gets bigger.

So a third control, declared here with the others:

| # | control | what it measures |
|---|---|---|
| C1 | serial(250) vs batched(250), pinned | licenses the batched arm — T5's threshold applies to this |
| C2 | serial(40) vs serial(250) at positions 0–39 | the instrument's own noise floor (expect bit-identical) |
| C3 | batched(T=2915) vs batched(T=250) at positions 0–249, and vs batched(T=1200) at 0–1199 | whether the batched answer depends on the BATCH it was computed in |

C3 is the one that speaks to long positions, and it is cheap. Causal attention
means position *p*'s state depends only on ids[0..*p*], so both runs **must**
produce the same logits at *p* — the only thing that differs is the GEMM shape,
which is exactly the freedom BLAS uses. If C3 is bit-identical, the batched
arm's answer at a position does not depend on how much came after it, and C1's
licence extends. If C3 is not, the size of that dependence is the honest bound
on the long bands and it goes in the report as such.

⚠ C3 is a **self-consistency** check, not an independent one: it cannot detect
an error the batched path makes at every T. That is C1's job, and C1's reach is
the stated limit.

---

## 3. THE CORPUS — fixed here, before any grading

### 3.1 The rule that generated it

Probes must land in all five bands, and stoke exposes logits only at
**generated** positions, so a probe at absolute position *p* is reached by
running stoke on a *p*-token prefix and reading generated index 0..63
(absolute *p*-1 .. *p*+62). Every stream is therefore
**(a prefix of a committed text) + (stoke's own 64-token greedy continuation)**,
and the reference is teacher-forced on exactly that concatenation — so both
sides are on the identical prefix at every probed position by construction.

Prefix cut lengths were chosen to tile the bands, **before any of them was
run**.

⚠ **AMENDMENT, made before any probe was compared, for a mechanical reason and
not a result.** The first draft of this table said the cut is made on the ID
STREAM, because BPE is not prefix-stable and cutting text then re-tokenizing can
silently produce a different prefix (Phase H's off-by-one on 2026-09-06 was that
class). **`stoke run` accepts text only — it has no `--ids`** — so an id cut is
not reachable without changing the CLI.

The resolution keeps the property that actually matters and drops the one that
does not. What matters is that **both sides consume the identical id sequence**,
and that is guaranteed by construction: stoke tokenizes the cut text, and the
reference is teacher-forced on *the ids stoke produced*, never on a
re-tokenization of anything. What is given up is only that a cut prefix's ids
are a prefix of the FULL text's ids — a property nothing in this protocol uses.

⇒ each cut is a **byte offset**, snapped to the next space so a stream never
begins mid-word; the token count that falls out is **measured and reported**,
not assumed. The table's token counts are therefore TARGETS, and the gate report
carries the realized counts. A target that lands in a different band than
intended is recorded as such and the band is read from the realized position —
positions are what the bands are keyed on, and no probe is discarded for
landing where it landed.

### 3.2 ⚠ Four streams are inherited from a corpus that WAS selected on success

`short1/short2/short3` are the prompts that came back byte-identical to
llama.cpp, and `L-2851` is long2 at its full length, which is the fourth.
Using them re-imports that selection. (The count was wrong in the first
draft of this sentence -- it said three while the table flagged four -- and
`m7_probe.py --selftest` caught the disagreement between the prose and the
table it is supposed to mirror.) They are kept, because they are
the model's real short-context behaviour and dropping them would be its own
selection — but they are **flagged in every table**, and `[0,100)` also carries
two unselected streams (`L-40`, `L-80`) so no band rests only on selected
material.

The bias direction is stated so it can be checked rather than argued: selection
on agreement makes the *envelope* quieter, which makes τ **smaller**, which
makes grading **stricter**. It cannot manufacture a pass.

### 3.3 The streams

`long2.txt` is 2851 tokens of continuous prose (the printing-press essay). Its
*prefixes* were never selected on anything; only its full-length behaviour was.

| id | source | prefix (tok) | probed absolute positions | bands touched | selected-on-success |
|---|---|---|---|---|---|
| S-a1 | short1.txt | 5 (full) | 4–68 | [0,100) | ⚠ yes |
| S-a2 | short2.txt | 12 (full) | 11–75 | [0,100) | ⚠ yes |
| S-a3 | short3.txt | 17 (full) | 16–80 | [0,100) | ⚠ yes |
| L-40 | long2.txt | 40 | 39–103 | [0,100), [100,300) | no |
| L-80 | long2.txt | 80 | 79–143 | [0,100), [100,300) | no |
| L-160 | long2.txt | 160 | 159–223 | [100,300) | no |
| L-240 | long2.txt | 240 | 239–303 | [100,300), [300,600) | no |
| L-380 | long2.txt | 380 | 379–443 | [300,600) | no |
| L-520 | long2.txt | 520 | 519–583 | [300,600) | no |
| L-660 | long2.txt | 660 | 659–723 | [600,1200) | no |
| L-900 | long2.txt | 900 | 899–963 | [600,1200) | no |
| L-1120 | long2.txt | 1120 | 1119–1183 | [600,1200) | no |
| L-1300 | long2.txt | 1300 | 1299–1363 | [1200,∞) | no |
| L-1900 | long2.txt | 1900 | 1899–1963 | [1200,∞) | no |
| L-2500 | long2.txt | 2500 | 2499–2563 | [1200,∞) | no |
| L-2851 | long2.txt | 2851 (full) | 2850–2914 | [1200,∞) | ⚠ yes |

**16 streams × 65 probed positions = 1040 probes**, against the M3 grid's
minimum of 55 per file. §8.8 asks for a denser grid here and this is what that
means in positions.

### 3.4 What is deliberately NOT in the corpus

- The unsloth Qwen file. That is B6, a separate rung with its own referent.
- The Q8_0 Llama file. It isolates quantization from architecture and is a
  *survey*, not part of this criterion.
- Any prompt authored after this document. Adding one later is permitted only
  as a **named amendment with a reason that is not "the results".**

---

## 4. THE PROTOCOL — three phases, and they run in this order

The order is structural, not a convention: the τ freeze and the grading are
separate invocations with a file between them, exactly as `q4km_tau.py` and
`classify.py` already are, so no tool can both set a threshold and apply it.

### Phase A — bank the probes
For each stream: cut the id prefix, run `stoke run --probe-all` (banking
stoke's top-5 at all 64 generated positions **and** its generated ids), then run
the pinned reference teacher-forced over `prefix + stoke's ids` with
`--probe-at` on the same absolute positions. Bank both raw.

### Phase B — freeze τ, from AGREEING positions only
A position is *agreeing* when both sides' rank-0 id is the same. τ measures how
large a logit disagreement is ordinary for this pair; it is measured where they
agree and spent where they do not — disjoint sets, so the freeze cannot be
circular.

For each band: the anchored Δ-profile (`m2_classify_probes.delta_profile`,
anchored on stoke's chosen id) over the shared top-5 ids of every agreeing
position, then

  **τ(f32_intcore_llama32_q4km, band) = max(0.10, 2 × band p95)**

nearest-rank, per `m2_classify_probes.freeze_tau`. Written to a τ file.

- **A band with fewer than 20 AGREEING POSITIONS is REFUSED, not floored.**
  Flooring converts "not measured" into "measured quiet". A refused band means
  positions in it cannot be graded, and `classify_records` already fatals on a
  missing entry, so the refusal propagates.
- The pair key is `f32_intcore_llama32_q4km` and **must not** be any Q8_0 or
  Qwen3 key. A τ written under the wrong key is not loud: the fatal fires on an
  ABSENT key, never on a present and wrong one.

### Phase C — grade
Every **divergent** position (rank-0 ids differ) is classified by `classify.py`
against the frozen table, per ruling §6: ARITHMETIC-CLASS iff
max(g_X,g_Y) < τ and the top-5 cores contain each other; BUG iff
max(g_X,g_Y) ≥ 2τ or a chosen token was never ranked by the other side and the
bound alone proves it; ESCALATE otherwise → the L2/L3 ladder decides.

---

## 5. THE PASS RULE, and the tripwires

**PASS = zero BUG-class at every probed position**, with every occupied band's
τ frozen from ≥20 agreeing positions, and the gate report naming binaries by
sha256, the model by name+size, and each τ as frozen BEFORE the
classifications it grades.

Tripwires, declared now:

| # | tripwire | threshold | action |
|---|---|---|---|
| T1 | divergence rate | **> 2% of probed positions** | stop, anomaly memo, no stamp |
| T2 | any BUG-class | ≥ 1 | stop, bisect under §7's pre-authorizations |
| T3 | a band with < 20 agreeing positions | any | band REFUSED; positions in it ungradeable |
| T4 | adjacent-band p95 growth | **> 1.5×** | bands stand (no collapse to one τ), and §8.9's named suspect is tested with `--rope-theta accum` |
| T5 | serial-vs-batched control | max anchored \|Δ\| ≥ 1e-3 | batched arm not licensed; this document void |
| T6 | serial null arm not bit-identical | any | the instrument has its own noise floor; report it, and read no Δ below it |

T1's 2% is the M3 expectation carried over as a RATE rather than a count:
§2.2 expected ≈0–1 class flips in a 55-probe grid, which is ~1.8%. A grid 19×
larger cannot inherit the count.

---

## 6. What a PASS here does and does not establish

**Does:** that stoke's Llama-3.2-3B forward pass agrees with an independent f32
reference, at positions out to 2914, to within a tolerance measured from the
pair's own agreeing behaviour — and that where they differ, they differ the way
two f32 implementations of the same graph differ, not the way a bug does.

**Does not:** byte-exact agreement with llama.cpp. That is structurally
unavailable and is *not* the criterion — and 2026-09-06 measured why: stoke's
own Windows and Linux builds already produce different greedy text from
identical source and weights, while every discrete-domain golden passes on both.
A byte-exact generation bar is not portable across toolchains, let alone across
implementations.

**Also does not:** say anything about performance on this model. No perf number
is taken here and none should be quoted from it.
