# Attributing the 3.2× — it is the kernel, and it is NOT instruction count

**2026-08-20, Opus 5, branch `m5-entry`.** Follows
`2026-08-20-m5-gate2-results-vs-physics-FOR-KABIR.md` §5b, which argued the
bit-exactness question was premature because the gap was measured and
unattributed. This attributes it. Prediction committed first
(`2026-08-20-m5-attribution-PREREGISTERED.md`, `b6de2b1`).

## 1. It is the kernel. The pool is not merely exonerated — it is a strength.

`gate-reports/2026-08-20-m5-attrib-threads-report.md`. Prefill of a 525-token
prompt, both engines, thread order reversed on even cycles.

| threads | stoke GFLOP/s | llama GFLOP/s | gap | stoke scaling | llama scaling |
|---|---|---|---|---|---|
| 1 | 46.3 | 149.1 | **3.22×** | 1.00× | 1.00× |
| 2 | 83.2 | 293.8 | 3.53× | 1.80× | 1.97× |
| 4 | 152.1 | 535.1 | 3.52× | 3.29× | 3.59× |
| 8 | 210.0 | 641.6 | 3.06× | **4.54×** | 4.30× |

The pre-registered discriminator was "≈2.8× ⇒ instruction efficiency, the kernel
is the target; ≤1.8× ⇒ the pool is". **The single-thread gap is 3.22×, and the
gap is ~3× at every thread count** — the signature of a per-thread difference,
not a parallelisation one. At t=8 **stoke scales 4.54× against llama's 4.30×**.

⇒ **Nothing about this gap is the thread pool, the scheduler, or M5.** §5b's
argument applies in full.

## 2. Where it is not: instruction count. A negative result, and the useful one.

`objdump` on `vec_dot_q4_k_q8_k_xn_avx2` found the per-sequence mins term
**fully unrolled and entirely scalar** — 16 `movswl` + 8 `add` + 8 `imul` +
8 `add`, roughly **40 µops beside the vector dot's ~57**. So ~40–45% of the
emitted per-sequence work was a loop that is not the dot product.

GCC had already applied the one algebraic regrouping that matters — the
disassembly shows `add %edx,%r15d` feeding a *single* `imul`, i.e. it folds
`bs[2k]+bs[2k+1]` before multiplying by the shared `mins8[k]`, which is exactly
what `vpmaddwd` does. Replacing the loop with `vpmaddwd` is therefore **bit-exact
by construction**: all terms integer, addition associative, and nothing
overflows (|bs| ≤ 2048 from 16 quants clamped to ±128; a pair ≤ 4096 against
i16's 32767; × a 6-bit min ≤ 258,048; `vpmaddwd` sums two to ≤ 516,096 against
i32). The bound is derived from `quantize_row_q8_k`, not assumed.

**Measured, interleaved, order alternating per cycle:**

| threads | A (scalar) | B v1 | B/A | B v2 (hoisted) | B/A |
|---|---|---|---|---|---|
| 1 | 5.4639 | 5.8081 | **1.063×** | 5.7574 | **1.099×** |
| 8 | 24.1512 | 26.0538 | **1.079×** | 26.4741 | **1.071×** |

B won **6 of 6** pairings in each run. The two variants are equivalent within
noise, so the change is worth **~7–10%**.

> **⇒ REMOVING ~40% OF THE EMITTED INSTRUCTIONS BOUGHT ~7% OF THE TIME.**
> The µop-count theory of the gap is dead. stoke's kernel runs at roughly
> **2.5 µops/cycle against a 6-wide front end** — it was never front-end bound,
> and the scalar `imul` chain was evidently overlapping with the vector work on
> separate ports. Whatever holds this kernel at 46 GFLOP/s single-threaded
> against llama's 149 is **port pressure, dependency latency, or memory
> traffic — not how many instructions there are.**

**This is the finding worth more than the 7%**, because it redirects the next
attribution step away from counting instructions. Candidates it leaves standing,
none measured:

- **`vpmulld` on the critical vector path** — 2 µops, ~10-cycle latency, issued
  once per sub-block (8× per super-block per sequence) purely to apply a 6-bit
  scale. llama.cpp does not do this: it defers scales to a float FMA at the
  super-block boundary. That change is **not** bit-exact — it moves work across
  the integer/float boundary — and is exactly the kind of thing §5b's line
  separates from the free ones.
- **Activation re-reads.** Each sequence re-reads its 256-byte q8 block per
  super-block; at a 16-wide tile that is 4 KB per weight row.
- **Register pressure.** `wnib[8]` holds 8 of 16 ymm registers live across the
  whole per-sequence loop.

## 3. What is safe to take, and what I did not do

The SIMD mins term is **bit-exact on the real model through the batch path** —
`p00x8` f32, `p00x8` q8, `p41x2` f32, all SAME — and **that comparison is
mutation-verified**: one wrong lane in the duplication mask makes it DIFFER.
The suite (14 binaries) and all four goldens pass against it.

⚠ **The byte-compare had to go through `batch --seqs N`, not `stoke run`.** The
modified kernel only executes at `n_seq > 1`, so an N=1 comparison would have
been vacuous — the same hole E1 fell into, recorded on 08-19.

**THE DEFAULT IS STILL `STOKE_MINS_SIMD=0`.** My stated bar was byte-compare
green *and* an interleaved A/B win, and both are met — but a change on the
M3/M4 parity lineage deserves the full 10-case verify gate against the
archives, not three ad-hoc cases, and that gate runs 1–2 h. **Flipping the
default is the next session's first action, after that gate.** The code, both
arms and the evidence are committed; nothing ships on three cases.

*(v2 hoists the loop-invariant `m16` out of the per-sequence loop — it was
being rebuilt N times inside a loop already holding `wnib[8]`, half the ymm
register file, live. It measures the same as v1 within noise. Kept anyway,
because rebuilding a loop-invariant N times is wrong regardless of the
stopwatch; that it changed nothing suggests GCC was already hoisting it, which
is one more datum against the front-end theory. `build/win/stoke.exe` has been
rebuilt with the DEFAULT flag and verified byte-identical to
`build/ab/mins-scalar.exe`, so the shipping artifact matches the default and no
future gate can grade the experimental arm by accident.)*

**⚠ AND THE PRE-FLIGHT SAID THE OPPOSITE.** A single unpaired sample read
**0.957× — B slower**. The interleaved run with alternating arm order reversed
it, B winning every pairing. **The cheap reading would have discarded a change
that wins** — one clean instance of the interleaving protocol paying for itself,
in the same session where it also caught a 27% thermal drift.

## 4. What this does and does not do to gate [2]

**Does not rescue it.** A bit-exact 1.07× against a required **2.75×** is ~7% of
the way in log terms. Gate [2] still misses.

**Does sharpen your decision.** The question in §5 was "keep bit-exactness or
buy the arithmetic". The honest state now:

- the gap is **entirely** in the kernel's per-thread efficiency,
- the free, bit-exact instruction-count win is **~7% and now essentially spent**,
- the remaining ~3× is in mechanisms I have **named but not measured**, and at
  least one of them (deferring scales to float) is squarely on the far side of
  the bit-exactness line.

So the dichotomy in §5 is **less false than I claimed in §5b, but for a reason
§5b got right**: the integer core was indeed reachable for free — it just did
not contain much. I would still not decide today. The next cheap step is to
price `vpmulld`'s contribution by ablation (compute the wrong answer
deliberately, measure, discard) — that costs an afternoon and tells you what
the bit-exactness line is actually worth in tok/s.
