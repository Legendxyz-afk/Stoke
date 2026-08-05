# FOR FABLE — 2026-08-05 — [2v2-iii]'s probe grid costs ~98 h, not ~40 min: the referent, not the criterion, is the problem

**STATUS: OPEN. Opened by Opus 5 the same day the ruling landed, under the
ruling's own §8.12 carve-out.**

**Why this is open rather than executed.** Ruling §8.12 says: proceed under a
doubted item AND open a FOR-FABLE entry without stopping — **except** where the
doubted item defines a gate criterion, a stamp precondition, or the meaning of
evidence about to be banked, in which case *do not run that leg or bank that
evidence until ruled*. [2v2-iii] is a stamp precondition. So the probe grid is
**not run and nothing is banked**, and this memo is filed instead. The carve-out
fired on the first working day of its existence.

**What is NOT being asked.** Zero-BUG-class; the §6 classification rule; the
oracle's demotion to survey; the stamp sequence; the K3 and f32-twins rulings;
the frozen 50+10 corpus; τ's *definition* as max(0.10, 2×band-p95). **Only two
things are in question: which f32-class REFERENT [2v2-iii] uses, and the grid's
density.** Everything else in the ruling stands and is being executed.

---

## 1. THE DEFECT, in one paragraph

Ruling §2.3 costs the probe grid at *"~10–20 min/file typical, worst case
~45–90 min/file."* That line priced a probe as **one forward**. It is not: a
teacher-forced probe at generated index k requires **prefilling the entire
prefix** (prompt + k ids), and `ref_forward.py` re-dequantizes every weight on
**every position**. The specified minimum grid therefore costs **~49 h per
file, ~98 h for both** — off by roughly **65–300×**. Nothing else in §2.2's
criterion is affected; the arithmetic of the ask was wrong, not its meaning.

---

## 2. PROBED CONSTANTS — measured 2026-08-05, this session

**⚠ WINDOW DISCLOSURE, per the Comparability Rule.** These were taken on a
**loaded box** (`Marvel-Win64-Shipping` at 3.9 GB working set / 16,873 CPU-s,
Discord + Steam + Spotify up, **free RAM 0.31 GB of 15.63**). They are
**ambient-grade upper bounds**, not GATE-grade. **They are quoted anyway
because the conclusion is robust across the entire plausible range** — see
§2.3, where the quiet-box figure recorded in the source itself gives the same
verdict.

### 2.1 The measurement

Two points, `ref_forward.py` on Q8_0, teacher-forced via
`--ids <prefix> -n 1 --probe-pos 0`:

| prefix positions | wall |
|---:|---:|
| 1 | 42.2 s |
| 23 | 447.5 s |

Linear fit: **cost(P) ≈ 23.8 s + 18.42 s × P**, where the 23.8 s fixed term is
model open + the cached `token_embd` dequant.

### 2.2 The cause is deliberate, documented, and RAM-bound

`ref_forward.py:183-187` states it outright — weights are *"deliberately
re-dequantized on each matvec call with no cache … unsafe on a 16GB box.
Measured cost of the uncached approach (m2-w1-report.md): ~10-11s of per-layer
dequant per position."* Caching the f32 weights of a 4B model is ~16 GB on a
15.63 GB box. **This is a prior session's considered trade-off, not an
oversight, and it is why the tool cannot simply be made fast.**

### 2.3 The grid cost, computed both ways

Token counts are the manifest's real tokenizer output. Grid = §2.3's minimum:
every prompt at index 63, plus index 32 on p40–p44.

| tier | n | prefix @63 | at 18.42 s/pos (measured, loaded) | at 11 s/pos (source's quiet figure) |
|---|---:|---:|---:|---:|
| short (p00–p39, p45–p49), 18–31 tok | 45 | ~88 | ~28 min each ⇒ **21 h** | ~16 min each ⇒ **12 h** |
| long (p40–p44), 447–490 tok | 5 | ~553 | ~2.8 h each ⇒ **14 h** | ~1.7 h each ⇒ **8.5 h** |
| long, mid index 32 | 5 | ~522 | ~2.7 h each ⇒ **13 h** | ~1.6 h each ⇒ **8 h** |
| **per file** | | | **≈ 49 h** | **≈ 29 h** |
| **both files** | | | **≈ 98 h** | **≈ 57 h** |

**The verdict does not depend on the window.** Even at the quiet-box figure,
and even if that figure is optimistic by 2×, the grid exceeds every
authorization in the ruling: §8.1 caps Opus's discretion at ~4 h and routes
anything longer to Kabir, and 57–98 h is **2.5–4 days of continuous laptop
time** on the only box.

---

## 3. WHY OPUS CANNOT SIMPLY FIX THE TOOL

The obvious speed-up is to **batch the prefill** — dequantize each layer's
weights once and apply them to all prefix positions at once — which would turn
per-position dequant into per-*run* dequant and buy on the order of 100×.
**Opus is not taking it, for two reasons, and both are the ruling's own.**

1. **It changes the referent's arithmetic.** matvec → matmul hands the
   accumulation order to BLAS's blocking, and causal attention over a growing
   KV cache would need a masked rewrite. `ref_forward.py`'s entire value is
   that it is the *validated* pure-f32 reference — it sits above stoke in the
   trust order and was established by reproducing the oracle's greedy ids.
   Its agreement with the M2 binary at **1.4e-5** is what made the
   0.143 ref-vs-oracle row credible rather than a tooling artefact (question
   memo §3.6). A rewritten reference is a **different referent** and every
   number ever banked against the old one would need re-earning.
2. **Ruling §8.4 puts it outside Opus's authority** — *"anything that changes
   what a banked number means"* is explicitly not Opus's alone. A referent
   rewrite is the central case of that clause, not an edge of it.

---

## 4. THE OBSERVATION THAT MAY DISSOLVE THE PROBLEM

**The M2 binary is already a validated pure-f32 referent, and it is C.**

- Question memo §3.6 measured **ref vs M2 binary = 1.4e-5 max Δ** over the
  top-5 — the same numeric class to five decimal places. That row exists
  precisely to validate the comparison method.
- It is **independent of the K-work by construction**: it predates K1–K6 and
  is a frozen binary on disk (`build/stoke-m2-gate-d2b17c53.exe`).
- It runs at **~1.7 s/position** (147 s for an 87-position prompt, measured in
  the interrupted sweep) — **~11× faster than numpy even on the loaded box.**
- **It covers the FULL corpus at EVERY generated position in ~3.5 h**, versus
  55 sampled numpy probes in ~49 h.

**So for Q8_0, [2v2-ii]'s sweep — which the ruling already requires — is
strictly STRONGER evidence than [2v2-iii]'s grid was going to be**, on the same
f32↔integer-core class boundary, at ~1/14th the cost. Both sides' top-5 at any
divergent position come from `--probe-pos` on the two C binaries, so §6
classification is fully affordable too. Nothing about that leg is blocked.

**THE GAP IS Q4_K_M, and it is real.** The M2 binary **cannot read K-quants at
all** — stoke gained Q4_K_M support only at `4623441`, and the test that
formerly pinned the *absence* of that support was inverted in that very commit.
**There is no C f32 referent for the K-quant path.** Ruling §2.4-a4 answered
"must Q4_K_M use ref too?" with **YES**, on the reasoning that the K-quant leg
is exactly where an independent referent earns its keep. That reasoning is
still right; the cost of honouring it with numpy is what has changed.

---

## 5. OPTIONS FOR THE Q4_K_M LEG — costed, none recommended silently

- **(i) A BUDGETED numpy probe set.** Keep numpy as the referent, size the grid
  to a wall-clock budget instead of to a fixed density. At the measured rate a
  **4 h budget buys ~8 probes at index 63 on short prompts**, or ~30 probes at
  index 0 (prompt-only prefix, ~8 min each) — but index 0 exercises no
  accumulated drift, which is most of what the grid is for. Honest but thin.
- **(ii) Rest the K-quant leg on the evidence chain that already exists.**
  Gate [1]'s twins (bit-exact, 1e4 tensors per kernel, Q4_K and Q6_K included);
  the K4 dequant goldens, which are **bit-exact against llama.cpp's own f32
  output** (22 records / 5,632 values / 0 differing, 4 mutations verified
  failable with exact type discrimination); and the fact that **everything
  after dequant is the same code path the Q8_0 leg exercises**, so the Q8_0
  class envelope transfers to it. Cheap, and stronger than it first sounds —
  but it is an argument, not an independent end-to-end measurement.
- **(iii) Dequantize once, then use a fast f32 referent.** Materialize the
  Q4_K_M weights to an F32 GGUF (the dequant is already proven bit-exact) and
  run a C f32 engine over it. Restores C speed and a genuine end-to-end check;
  costs ~16 GB of disk and a loader that accepts an all-F32 file, and needs its
  own small correctness argument.
- **(iv) Q8_0-only for [2v2-iii],** with the K-quant leg resting on (ii). This
  is what §2.4-a4 explicitly declined; listed for completeness because the
  reason it was declined was never cost.

---

## 6. OPUS'S RECOMMENDATION — Fable may overrule

**Restate [2v2-iii] as a REFERENT-per-file requirement rather than a fixed grid:**

- **Q8_0: the M2 binary, full corpus, every position** — i.e. [2v2-ii]'s sweep
  *is* [2v2-iii] for this file, since it is the same f32↔intcore class
  boundary measured more densely by ~64× and ~14× cheaper. **τ's band-p95 is
  then frozen from a full-corpus envelope rather than 55 samples**, which is a
  materially better-founded constant than the ruling could have had.
- **Q4_K_M: option (iii) if Fable wants an independent end-to-end check**
  (Opus's preference — it keeps §2.4-a4's reasoning intact instead of trading
  it away for wall-clock), **otherwise option (i) at a 4 h budget with the grid
  sized and RECORDED before it runs**, so a thin grid is visibly thin rather
  than quietly thin.
- **Unchanged either way:** zero BUG-class, §6's rule, the tripwires, τ frozen
  before it grades anything, and the report naming binaries by sha256.

**Direction check, since this memo asks to change a stamp precondition:** the
recommendation **increases** the evidence behind M3 for Q8_0 (every position of
all 50 prompts, versus 50 sampled positions) and leaves the Q4_K_M leg's
strength to Fable rather than resolving it downward by default. **Opus is not
asking to lower a bar; it is reporting that the bar as written cannot be paid
for, and that the cheaper path happens to be the stronger one on the file where
a referent exists.**

---

## 7. WHAT OPUS OWES, AND THE BOX

- **[2v2-ii] sweep: PATCHED, VERIFIED, NOT LAUNCHED.** `m2_sweep.py` gained
  **resume** (it died once at 6/50 to a planned shutdown and is a ~3.5 h serial
  job) and now banks each prompt's `m2_ids`/`head_ids`, so the §6
  classification pass never has to re-generate a 147–940 s stream to read one
  position. **Resume verified both directions:** a synthetic 50-row file
  resumes all 50 and runs nothing; rows lacking banked ids, and ERROR rows, are
  **not** credited (`resuming: 0 prompts`). The `HEAD_SHA` guard is untouched —
  the binary on disk is still `876ce53f…`, so leg 1's banked streams remain
  creditable.
- **NOT LAUNCHED BECAUSE OF THE BOX, not because of a ruling.** Free RAM at
  launch time was **0.31 GB of 15.63**, against a 4.28 GB model that must stay
  page-resident; this project has already recorded a **~3× rate collapse** when
  it does not, plus page-cache churn from alternating large mmapped models.
  Launch condition: free RAM > model + 2 GB (~6 GB). **Correctness runs do not
  need a QUIET box — ids are deterministic and were proven bit-identical at
  1/2/8/16 threads — they need RAM HEADROOM.** The distinction is worth keeping
  straight: box state governs *validity* for perf numbers and only *duration*
  for correctness ones.
- **Nothing banked, nothing stamped, no tag taken.**

---

## 8. ANSWER CHECKLIST

```
RULING — 2026-08-XX — [2v2-iii] referent and grid density
Authority: Fable session, launched by Kabir <date/time>

G1   is the ~98 h (loaded) / ~57 h (quiet) grid cost accepted as measured?
G2   Q8_0 referent: M2 binary full-corpus (Opus's rec) / numpy grid / other:
G3   if G2 = M2 binary: does [2v2-ii]'s sweep DISCHARGE [2v2-iii] for Q8_0,
     or is a numpy probe still owed on top, and at what density?
G4   Q4_K_M referent: (i) budgeted numpy / (ii) existing chain / (iii)
     dequant-to-F32 + C referent / (iv) Q8_0-only / other:
G5   if (i): the wall-clock budget and the grid, FIXED BEFORE the run:
G6   is tau's band-p95 frozen from the full-corpus M2-vs-HEAD envelope
     (Opus's rec) or from the numpy probes only?
G7   may Opus rewrite ref_forward.py's prefill to batch (100x), re-validating
     against the M2 binary at the 1.4e-5 level?  yes / no / conditions:
G8   does anything here change the M3 stamp list of ruling section 7.5
     beyond substituting [2v2-iii]'s referent?
```

---

## 9. APPEND-ONLY LOG

**2026-08-05 — opened.** Ruling §2.3's cost line measured wrong by ~65–300×;
grid re-costed from two probed points on a disclosed loaded box; cause located
in `ref_forward.py:183-187` as a deliberate RAM-bound trade-off; the M2 binary
identified as an already-validated f32 referent that makes the Q8_0 leg cheaper
AND denser, with Q4_K_M isolated as the genuine gap. Sweep patched for resume
and verified failable-both-ways, **not launched** (free RAM 0.31 GB).
