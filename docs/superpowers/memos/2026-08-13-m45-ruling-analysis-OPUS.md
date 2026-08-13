# ANALYSIS — the 2026-08-12 Fable ruling, checked before M4.5 executes from it

*2026-08-13. Opus 5, at Kabir's instruction ("analyze fable's ruling").
Authority unchanged: the ruling stands as written; this memo records what its
checkable claims measure when recomputed, and the execution consequences.
Where a finding touches criterion text or a frozen constant it is
§8.4-reserved ⇒ decided here under the 2026-08-07 governance, Kabir overrules.*

**Why this pass exists.** The 08-07 analysis of the previous five rulings found
one recurring shape — *a quantity asserted from structure rather than
computed* — three times. The 08-12 ruling's §7 appendix pre-rules every M4.5
branch precisely so no further summons is needed, which means its constants
will be executed on without a second reader. That is the argument for reading
it once, hard, before the first line of B.

**Method.** Every numeric or source claim was re-derived: source lines opened,
tensor sizes read from the GGUF via `stoke info`, rates recomputed from the
banked JSON. Nothing below is inferred from a summary.

---

## 1. Verified exact — no action

- **F1** — `rows_delegate_batch` (`src/kernels/vec.c:78`) is chunk-outer,
  sequence-next, `row_fn` per sequence: the quant unpack does repeat per
  sequence. `vec.c:174`'s comment discloses it in those terms. Both citations
  land where the ruling says.
- **F2's kernel shape** — `kq_mac8` (`src/kernels/simd.c:377`) is exactly
  `_mm256_add_epi32(acc, _mm256_mullo_epi32(w,a))`, 8 i32 lanes = 8 products
  per multiply, fed by `_mm_loadl_epi64` — **8-byte loads**, as claimed.
- **F3** — `aux32[8]`/`aux16[8]` are in `vec_dot_q4_k_q8_k` (`kquant.c`, at
  ~335 rather than the cited 325). The 8-lane partition is real.
- **§3's unification claim** — `vec_dot_q8_0_q8_0` (`src/kernels/q8_0.c:115`)
  is already "integer block total → one sequential f32 chain"
  (`sumf += sumi * (d_w*d_a)`, scales multiplied together first). CONTRACT v2
  generalises an existing shape; it does not invent one. This is the ruling's
  load-bearing claim and it holds.
- **Both saturation bounds** — Q4_K `15·127·2 = 3,810` and Q6_K
  `63·127·2 = 16,002`, both < 32,767. Correct.
- **C's integer-totals debug oracle** — sound as stated: if the change is
  fold-only, old and new must agree bit-for-bit on per-superblock integer
  totals, so any C bug bisects to "integer core" or "fold" in one dump.

## 2. Findings

### A1 — D's RAM price is overstated ~2.4×, and §7.4 names a class the repack cannot touch

§2 prices the repack at **"≈ +2.3 GB RSS"** and uses that to justify P-BAR at
0.85× rather than 0.95×; §7.4 predicts **"Expected verdict on 15.63 GB:
reject"**. §7.4 also scopes D to *"the two largest weight classes only
(ffn_down/ffn_up)"*.

Measured from the file (`stoke info`, 398 tensors, total 2,491,323,904 B =
2.32 GB):

| class | GB | share | type |
|---|---|---|---|
| ffn_down | 0.577 | 24.9% | **Q6_K** |
| ffn_gate | 0.470 | 20.2% | Q4_K |
| ffn_up | 0.470 | 20.2% | Q4_K |
| token_embd | 0.297 | 12.8% | Q6_K |
| attn_q / attn_output | 0.198 each | 8.5% each | Q4_K |
| attn_v | 0.061 | 2.6% | Q6_K |
| attn_k | 0.049 | 2.1% | Q4_K |

Three consequences, none of which the ruling's numbers survive:

1. **+2.3 GB is the whole model.** It is the cost of duplicating everything,
   not of repacking. The named format `block_q4_Kx8` repacks **Q4_K only** —
   all Q4_K in this file totals **1.385 GB**, which is the ceiling for any
   llama-style repack here.
2. **§7.4's own scope is 0.94 GB, not 2.3.** The two largest *Q4_K* classes
   are ffn_gate + ffn_up = 0.940 GB.
3. **ffn_down is Q6_K** — the single largest class, 24.9% of weights, and
   `block_q4_Kx8` cannot repack it. §7.4 names the one big class outside D's
   reach. That also caps D's *speed* benefit: a quarter of the matvec work is
   untouched by it.

**Disposition.** The ORDER does not change and D stays conditional — the
ruling's machinery is self-correcting because D's gate is a measurement with
"write the number down either way". What changes is the prior: at ~0.94 GB
against model 2.32 GB + KV 2.25 GB (f32: 512 blocks × 4.50 MiB at
N=8/ctx1024) ≈ 5.5 GB on a box with
~12 GB available, **"expected reject" is not supported by the ruling's own
arithmetic**, and must not be carried into the measurement as an expectation.
§2's 0.85-vs-0.95 rationale rests on the inflated figure; if D is reached and
passes, the tightening to 0.95× is better founded than the memo assumed.
D's scope, if it runs, is **ffn_gate + ffn_up** — the two largest classes the
format can actually repack.

### A2 — the 0.5%/token bound is a rounded restatement of 0.594%, and that rounding is the entire width of C-GRAY

Both rulings state the bound as one calibration: "first-divergence ≤ the M2
census rate (19/50) **AND** flip rate ≤ 0.5%/token", the 08-04 memo writing it
"the measured base rate (**~0.5%/token**, 19/50 prompts)".

Recomputed from `tests/tooling/.gate_cache/m2_sweep.json`: the census is
19 diverging prompts over **3,200 compared tokens = 0.5938%/token**. No
definition of flip rate on that census yields 0.5% (all differing positions =
20.4%; positions up to and including the first divergence = 0.748%). The
"~0.5%" is 0.5938% rounded down, and §8.1 then hardened it into a strict bound
**stricter than the base rate it is explicitly calibrated to** — which §5(d)
of the same memo refuses in principle ("a divergence-rate bar tighter than
zero-BUG is refused").

At n=64 × 50 prompts the two conjuncts are not the same constraint: 19/50
passes the first and fails the second (0.5% allows only 16).

**And it propagates into the C branches, where it is the whole story:**

- **C-GRAY** = zero BUG, ≤19/50, flip ∈ (0.5%, 3.10%] ⇒ **exactly 17, 18 or 19
  prompts**. Under the corrected 0.5938% the band is **empty**.
- **C-FAIL-RATE's flip trigger (>3.10%/token) is unreachable by construction.**
  The instrument counts one first-divergence per prompt over 3,200 tokens, so
  its maximum possible value is **50/3200 = 1.5625%**. Only the ">19/50" half
  of C-FAIL-RATE can ever fire. The 3.10% ceiling is a *probe-grid* rate
  (9/290 covered positions, M3) — a per-position divergence probability, not
  the same quantity as a per-prompt-first-divergence rate, and importing it as
  a ceiling compares two different instruments.

**Disposition (§8.4-reserved, mine, Kabir overrules).** The operative bound is
**≤19/50** — the census's own measured 0.5938%/token — for gate item [4] and
for C alike, with the measured rate reported on its face both ways. The stakes
are lower than they look in either direction: the 08-04 ruling's disposition
for a miss is *a results-vs-physics memo, not a grind-until-green*, and the
08-12 ruling independently pre-ruled 17–19 as **pass-with-memo** for C. A
result in that range is a memo either way; only a >19 result is a stop.

### A3 — "the fold's dependency chain is noise" compares against a quantity nobody computed

§3/§7.2: *"The sequential fold's dependency chain costs ~4 cycles × (n/256)
superblocks per row — noise against the integer work."*

The chain figure itself is defensible: the fold is **two dependent f32 ops per
superblock** (`acc += d·S_b` then `acc -= dmin·ad·M_b`), and at Golden Cove's
2-cycle FP-add latency that is the quoted ~4 cycles per superblock. *I do not
challenge the 4.* What was never computed is **the thing it is called noise
against**:

- n_in = 2560 (most matvecs) ⇒ 10 superblocks ⇒ **~40–60 cycles** of chain
  (2 dependent adds × 2–3 cycles × 10).
- n_in = 9728 (ffn_down) ⇒ 38 superblocks ⇒ ~150–230 cycles.
- The llama-shaped integer core it is compared against: 2560 products at 32
  per `maddubs` ⇒ ~300 uops ⇒ **~60–75 cycles** at 4–5 uops/cycle.

So the chain is **the same order as the integer work, not noise** — and that
is precisely because C succeeds: C's value is making the integer work small,
which is what promotes the fold from negligible to comparable. The claim was
true of the OLD kernel and is asserted about the NEW one.

**The mitigant the ruling also does not state:** rows are independent, so an
out-of-order core overlaps row *r*'s fold chain with row *r+1*'s integer work;
the chain only binds with one row in flight. **A′ (row tiling) is exactly
that**, and §7.2's C-SLOW branch already routes there ("proceed to A′
regardless — row-tiling is the designed next lever").

**Disposition.** No change to the ladder. But if C lands at the low end of its
12–15 estimate, the fold chain is the first suspect and A′ is the answer — do
not tune blind, which C-SLOW already forbids. Refit the step model after C
before attributing anything.

### A4 — P-BAR needs ~1.90×, and C's own estimate spans a miss

- llama-bench Q4_K_M **17.45 tok/s** @6T (2026-08-11 budget note); stoke N=1 =
  **7.811** (item [5] curve median). Gap **2.234×**.
- P-BAR = 0.85 × 17.45 = **14.83 tok/s** ⇒ stoke must gain **1.90×**.
- §4's estimate for C is "8 → ~12–15 tok/s" = **1.54–1.92×**. The midpoint
  misses the bar; only the top edge meets it.

**Consequence:** A′ is on P-BAR's critical path, not polish. The ruling already
orders A′ before D, so nothing re-orders — but C's result must be read against
14.83, not against "did it improve", and a C landing at 12–13 is *on plan*,
not a failure.

⚠ **P-BAR's referent path needs pinning before it is measured.** "stoke N=1
decode" does not name a harness, and `stoke run` (via `model_forward_position`,
`m->scratch`) and `stoke batch --seqs 1` (via `model_forward_batch`,
`m->scratch_n[0]`) are the same body through different orchestration — proven
byte-identical in output by gate item [2], never compared for *speed*. The
curve's 7.811 came from the batch harness. **Decision: P-BAR is measured on the
same harness as the curve (`batch --seqs 1`), so the bar and its baseline share
an instrument; record the `run` number beside it.**

### A5 — two unsourced constants

- **§7.5(4): "K6 already bounded it [`kv_vec_offset`] ~0.9%, confirm still."**
  The figure 0.9% appears **nowhere in the project's documents** except that
  line. K6 bounded the translation cost by its *bar margin* (8.08–8.46 against
  ≥7.8), not with a percentage. ⇒ treat §7.5(4) as a measurement to **take**,
  not a number to confirm.
- **§1 F2's "~0.75 uops/product"** for `kq_mac8`. Reading the loop: 2×
  `loadl_epi64` + 2× widening convert + 1 mask/shift + `vpmulld` (2 uops) +
  `vpaddd` ≈ **8 uops per 8 products ≈ 1.0/product**. The 4–6× deficit
  conclusion is unaffected — if anything it is understated.

### A6 — §4 and §7.1 describe B differently; §7.1 governs

§4 (L1) hoists the unpack "once per **row-chunk**"; §7.1 ("exactly") says "for
each **ROW**, unpack the weight once, then loop the N sequences' activations
against that unpacked row". Per-chunk hoisting would mean holding 64 unpacked
rows live; per-row is the implementable shape and the one whose arithmetic is
unchanged by construction. **Implement §7.1.**

### A7 — §5's chunk-alignment reassurance is premature, though harmless

"stoke's chunk 128 is a multiple of block 16, so M5 inherits this by
construction" — there is **no prefill chunking in `src/`**. 128 is a *planned*
constant in the 07-19 forward-designs note, which explicitly leaves 64 open
(also a multiple of 16). The property is a choice M5 must preserve, not an
inheritance. The ruling's action — record the invariant in M5's design note —
is exactly right; only the reassurance is early.

## 3. What does not change

The ladder order (B → C → A′ → D → L4), CONTRACT v2's substance and its
by-construction bit-exactness argument, zero-BUG gating, the branch structure
including every disposition, the standing rules from §1b, and §7.7's process
constraints. Five of the ruling's checkable claims verified exact, including
the load-bearing one (§3's Q8_0 shape). **The findings above change three
priors and one execution detail; they do not reopen the ruling.**

## 4. Carried into execution

1. B: implement §7.1's per-row shape; N=1 must not move (step model) — any
   N=1 change is a dispatch bug.
2. C: grade against ≤19/50; expect 12–15 tok/s against a 14.83 bar; refit
   `c`/`f` before attributing.
3. A′: on P-BAR's critical path, not optional.
4. D, if reached: scope is **ffn_gate + ffn_up (0.94 GB)**; ffn_down is Q6_K
   and out of reach. Take the residency measurement without the "reject"
   prior.
5. §7.5(4): measure `kv_vec_offset`'s share; do not confirm 0.9%.
