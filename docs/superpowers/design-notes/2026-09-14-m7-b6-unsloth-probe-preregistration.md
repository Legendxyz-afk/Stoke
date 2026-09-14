# 2026-09-14 — B6, THE UNSLOTH FILE'S FORWARD HALF: the pre-registration

**Written BEFORE any probe on this file is compared.** Same reason as the 09-07
entry probe's: τ is frozen from the pair's own agreeing positions before any
divergence is graded, and a corpus, grid or tripwire chosen after seeing a
result is a bar fitted to the answer.

Nothing in this file is a result. Results go in the gate report.

---

## 1. What binds this

- **M7 gate item [2]** (`docs/superpowers/design-notes/2026-07-19-m3-m7-forward-designs-and-risk-playbook.md:444`):
  "unsloth-file parity run recorded". Its foreseen risk (`:424-427`): *same
  weights, different GGUF writer — metadata deltas. Run `stoke info` diff
  first, then the full parity corpus. Any delta is a finding to record, not
  to paper over.*
- **The referent-per-file requirement**, `docs/superpowers/memos/2026-08-06-m3-probe-grid-RULING-FABLE.md` §1
  ([2v2-iii] v2): *for EACH gate file, HEAD's banked streams are checked
  against an independent, validated f32-class referent … every divergence
  classified per §6 under a τ frozen from the SAME PAIR's agreeing positions
  before any divergence is graded. PASS = zero BUG-class.* The referent of
  record for Q4_K_M is `ref_forward.py` (§1, §4).
- **`docs/ROADMAP.md:619-627`** (Fable, 2026-08-05): *never a 50/50
  live-oracle token bar for any model.*
- **The 09-05 B5 entry note §3a** on what B6 is and is not: the two Q4_K_M
  files have identical tensor tables but **different weight bits** (imatrix
  497 vs 79 chunks), so B6 is *"does stoke match the referent ON the unsloth
  file"* — never *"do the two files agree"*. Getting that backwards would
  manufacture a BUG-class divergence out of a quantization difference.

⚠ `llama.cpp` is not consulted anywhere below. The 09-05 note's phrase "match
llama.cpp on the unsloth file" is read through the ROADMAP line above: the
comparison is stoke vs the f32 reference on this file, the shape every other
forward-pass gate in this project has.

### Rung state going in

| half | ruled criterion | state |
|---|---|---|
| metadata diff | `stoke info` diff, deltas recorded | ✅ 09-05 B5 entry note §3a: 398/398 tensor lines identical; deltas are `general.base_model.*`, `quantized_by`, `repo_url`, one tag, `padding_token_id`, and **`bos_token_id` absent** — which exposed `tok.c`'s stricter-than-its-own-use required read (fixed in the tokenizer work, `bfda9ec`) |
| tokenizer | M1-style byte-exact via `llama-tokenize` | ✅ `tests/tok_unsloth_golden.sh`, 4 legs + leg 0 (the metadata guard that makes the file identity checkable — the two files share a tokenizer, so the byte-exact legs alone cannot tell them apart; demonstrated) |
| twins | kernel twins on every kernel the file exercises | ✅ carried from M3 — same tensor types as the bartowski file, so the same kernels; `tests/att_simd_twins.sh` covers attention's SIMD arm at G=4 |
| **ref probe grid** | §2.3 protocol, τ per band from this pair | ⬜ **this document** |
| **τ freeze** | `f32_intcore_unsloth_q4km`, per band, before grading | ⬜ **this document** |

---

## 2. The instrument

`tests/tooling/m7_probe.py`, the three-phase tool the 09-07 entry probe ran,
gains a `--set` switch selecting a model set — `llama32` (its existing
constants, the default, byte-for-byte the B5 configuration) or `unsloth`
(this document's). The phase logic, the shared imports (`m2_classify_probes`,
`classify`) and the launcher are untouched; only MODEL, PAIR, the stream
table and the report name vary by set. The selftest keeps asserting the
Llama table equals the 09-07 pre-registration and additionally asserts the
unsloth table equals §3 below.

`ref_forward.py`'s `qwen3` path is the ORIGINAL M2 tool — the referent of
record for the bartowski file's M3 grid — and needs no change. Its
`--mode batched` arm was licensed on 2026-09-07 by C1 **on the `llama`
architecture**. The Qwen3 graph differs (QK-norm before rope, NEOX rope), so
the licence is re-earned here rather than inherited:

**C1-Q (pre-registered prediction, serial-vs-batched, `p00`, teacher-forced
along stoke's own 64-token stream, every generated position probed):**

> Over every probed position: the argmax agrees on all of them, and the
> maximum anchored |Δ| between the serial and batched references is < 1e-3.
> **FALSIFIED IF** either fails. On falsification the batched arm is not
> licensed for this pair, this document is void, and the grid is redesigned —
> not re-run against a looser threshold.

**T6 null arm, required:** the serial reference on `p00` is run twice under
identical pinning; the two probe files must be bit-identical over every
top-5. If not, the instrument's own noise floor is reported and no Δ below it
is read.

Every reference run goes through `tests/tooling/run_ref_probe.ps1` (affinity
`0xFFF`, `OPENBLAS_NUM_THREADS=6`, the launcher throws if affinity does not
take) and the probe file's `env` block records what the run observed.

**Cost, so it is priced before it is spent:** the 09-07 batched arm banked
70–181 s per stream on Llama-3B for 671–2851-token streams (`ref_seconds` in
the banked grids). Qwen3-4B is ~1.3× the parameters; expect ~2–5 min per
long stream, under 40 min for the whole grid. The serial control at the
measured ~26–46 s/position is ~1 h for two runs of `p00`.

---

## 3. THE CORPUS — fixed here

### 3.1 The rule

Two sources, each chosen by a rule that predates this file:

- **The M3 G5 grid, verbatim** (`2026-08-06-…-RULING-FABLE.md` §5, "fixed
  before the run — mechanical"): `p00 p10 p20 p30 p45` (every 10th short
  prompt and the first of the post-long block) and `p41` (the corpus's one
  demonstrated ESCALATE class on Q8_0 — the adversarial pick). This is the
  grid the bartowski file was gated on; "for each gate file" means the same
  grid on this one.
- **The M7 long-band cuts, verbatim** (`2026-09-07-m7-entry-probe-preregistration.md`
  §3.3): `long2.txt` at targets 660 / 900 / 1120 / 1300 / 1900 / 2500 and
  full length. The text is prose; nothing about it is Llama-specific. These
  exist because §8.9's bands `[600,1200)` and `[1200,∞)` have never been
  probed on ANY Qwen3 file — the M3 grid stops at 600 — and the 09-06 note
  records a 1426-token Qwen prompt diverging from llama.cpp on this very file
  with no grading behind it. The band is where the unmeasured claim is.
- **Two more of the same cuts, 160 and 240** (also from the 09-07 table),
  added in this document before any run, for a reason that is not a result:
  on the G5 grid `[100,300)` is reached only by the tails of `p20` (64
  tokens) and `p30` (53), which on the bartowski file gave it 45 positions —
  enough for a τ, but one bad stream from T3's floor of 20 agreeing
  positions. `L-160` lies wholly in the band and `L-240` bridges it to
  `[300,600)`.

Cuts are byte offsets snapped to the next space, exactly as §3.2 of the 09-07
document amended (stoke tokenizes the cut text; the reference is
teacher-forced on the ids stoke produced). Realized token counts are measured
and banked; targets are targets.

### 3.2 Selection on success — none in this corpus

No stream here was ever selected on this file's behaviour: the G5 prompts
were chosen by index rule on 2026-08-06 and the long cuts by band-tiling on
2026-09-07, both before any of them touched the unsloth file. `p41` was
chosen for being *adversarial* on Q8_0, which biases toward finding trouble.

### 3.3 The streams

| id | source | prefix (tok, target) | bands touched (expected) |
|---|---|---|---|
| P-00 | `tests/data/parity/p00.txt` | full (23 per MANIFEST) | [0,100) |
| P-10 | `tests/data/parity/p10.txt` | full (24) | [0,100) |
| P-20 | `tests/data/parity/p20.txt` | full (64) | [0,100), [100,300) |
| P-30 | `tests/data/parity/p30.txt` | full (53) | [0,100), [100,300) |
| P-45 | `tests/data/parity/p45.txt` | full (14) | [0,100) |
| P-41 | `tests/data/parity/p41.txt` | full (477) | [300,600) |
| L-160 | `tests/data/llama_gen/long2.txt` | 160 | [100,300) |
| L-240 | `long2.txt` | 240 | [100,300), [300,600) |
| L-660 | `long2.txt` | 660 | [600,1200) |
| L-900 | `long2.txt` | 900 | [600,1200) |
| L-1120 | `long2.txt` | 1120 | [600,1200) |
| L-1300 | `long2.txt` | 1300 | [1200,∞) |
| L-1900 | `long2.txt` | 1900 | [1200,∞) |
| L-2500 | `long2.txt` | 2500 | [1200,∞) |
| L-2851 | `long2.txt` | full (~2851 on Llama's tokenizer; Qwen's count is measured) | [1200,∞) |

**15 streams × ~64 probed positions ≈ 960 probes.** Bands are read from
realized positions; a stream that lands in a different band than expected is
recorded as such and no probe is discarded for landing where it landed.

### 3.4 Deliberately NOT in the corpus

- The bartowski file. Its grid is M3's record; re-running it here would
  invite the comparison §1 forbids.
- Any prompt authored after this document.

---

## 4. THE PROTOCOL — three phases, in this order

Identical to the 09-07 protocol, with the pair key changed:

- **Phase A** — per stream: `stoke run --probe-all` on the unsloth file (banks
  stoke's top-5 at every generated position and its ids), then the pinned
  batched reference teacher-forced over `prefix + stoke's ids` with
  `--probe-at` the same absolute positions. Bank both raw.
- **Phase B** — freeze **τ(f32_intcore_unsloth_q4km, band) = max(0.10, 2 × band p95)**
  from AGREEING positions only, per band, nearest-rank, via
  `m2_classify_probes.freeze_tau`. A band with fewer than 20 agreeing positions
  is REFUSED, not floored. Written to a τ file before phase C runs.
- **Phase C** — every divergent position classified by `classify.py` against
  the frozen table per ruling §6 (ARITHMETIC / BUG / ESCALATE → ladder).

⚠ **The pair key is `f32_intcore_unsloth_q4km`.** Not `f32_intcore_q4_k_m`
(the bartowski pair's frozen 1.023739 / 1.178094 / 0.811432) and not the
Llama key. `classify.py` fatals on an absent key, never on a present and
wrong one — a bartowski τ applied to this pair would be silent and, since
the weights differ, meaningless in either direction.

---

## 5. THE PASS RULE, and the tripwires

**PASS = zero BUG-class at every probed position**, every occupied band's τ
frozen from ≥20 agreeing positions, binaries named by sha256, the model by
name + size + sha256, each τ recorded as frozen before the classifications it
grades.

| # | tripwire | threshold | action |
|---|---|---|---|
| T1 | divergence rate | > 2% of probed positions | stop, anomaly memo, no stamp |
| T2 | any BUG-class | ≥ 1 | stop, bisect under the ruling's §7 pre-authorizations |
| T3 | a band with < 20 agreeing positions | any | band REFUSED; its positions ungradeable |
| T4 | adjacent-band p95 growth | > 1.5× | bands stand; the rope-theta suspect is tested with `--rope-theta accum` as on 09-08 |
| T5 | C1-Q serial-vs-batched | max anchored \|Δ\| ≥ 1e-3, or any argmax disagreement | batched arm not licensed for this pair; this document void |
| T6 | serial null arm not bit-identical | any | instrument noise floor reported; no Δ below it read |

**Carried expectation, stated so it can be falsified rather than fitted:** the
bartowski pair's M3 grid ran at 3.10 %/token class-flip base on Q4_K_M
(`2026-08-10-m3-flip-base-rate-finding.md`), against T1's 2%. If this file
reads near 3% too, T1 fires and the memo says so — the threshold is not
raised to fit a second Q4_K_M file. The B5 Llama grid read 1.367%.

---

## 6. What a PASS establishes

That stoke's forward pass on the file people actually download agrees with
an independent f32 reference of the same file, out past 2,800 tokens, to
within a tolerance measured from the pair's own agreeing behaviour — and that
where the two differ, they differ as two f32 implementations of one graph
do. It says nothing about agreement with llama.cpp's token stream, which is
structurally unavailable across toolchains (2026-09-06) and is not the
criterion.
