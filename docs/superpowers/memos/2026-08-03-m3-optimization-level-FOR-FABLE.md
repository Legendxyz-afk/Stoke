# M3 — the build is `-O1`, and the reference stream is not a bandwidth ceiling

**Status:** decision brief. Written by the Opus 5 manager session of 2026-08-03,
parked for Kabir to hand to Fable per the 07-28 escalation routing. **Nothing
has been changed in the build on the strength of this.**

**Escalation trigger:** *"results contradict spec physics."* Two of this
project's recorded quantitative conclusions do not survive re-measurement. The
qualitative one they were used to support does survive — but only after the
instrument underneath it is replaced.

> **This brief was rewritten once, before it was ever sent.** Its first version
> concluded that the matvecs are at the memory-bandwidth ceiling and that K3
> therefore has no headroom to collect. That conclusion was an artifact of the
> same broken denominator it was written to expose. §3 is the correction and
> §3.1 is how it was caught, because the way it was caught is the transferable
> part.

---

## 1. The build has always been `-O1`

One `CFLAGS` line, no release variant:

```
CFLAGS := -std=c11 -Wall -Wextra -Werror -g -O1 -Isrc
```

Every number this project has recorded — M0's premise work, K1's microbench,
both `bench/history.jsonl` rows, the 13.3× and 14.5× gaps to the bar — comes
from an `-O1` binary. The vendored llama.cpp it is measured against
(`bench/vendor/llama-b10037/`) ships as arch-dispatched **release** DLLs.

It appears to be inherited rather than chosen: the Makefile describes itself as
mirroring `../loom/Makefile`, `-g -O1` is a debugging default, and no note
anywhere defends it.

## 2. Raising it is a large, free speedup

Three builds, identical source, runs **interleaved** (`O1,O2,O3, O1,O2,O3, …`)
so box drift cannot masquerade as a flag effect. Medians of 3, quiet box.

| op | `-O1` µs | `-O2` µs | `-O3` µs | O3/O1 |
|---|---:|---:|---:|---:|
| matvec attn_q | 4681.6 | 1720.9 | 1447.8 | **0.309** |
| matvec attn_k | 1129.3 | 398.7 | 352.5 | **0.312** |
| matvec attn_v | 1162.9 | 401.5 | 352.3 | **0.303** |
| matvec attn_out | 4664.5 | 1746.8 | 1453.9 | **0.312** |
| matvec ffn_gate | 12107.4 | 4362.9 | 3882.8 | **0.321** |
| matvec ffn_up | 11543.2 | 4405.5 | 3934.8 | **0.341** |
| matvec ffn_down | 11660.3 | 4364.4 | 3847.0 | **0.330** |
| matvec LOGITS | 182833.0 | 70533.0 | 58396.4 | **0.319** |

| | implied s/token | tok/s | gap to ≥7.8 |
|---|---:|---:|---:|
| `-O1` | 1.873 | 0.534 | **14.6×** |
| `-O2` | 0.697 | 1.435 | **5.4×** |
| `-O3` | 0.608 | 1.644 | **4.7×** |

**Numerics are unchanged, and this was checked rather than argued.** GCC does
not reassociate floating point without `-ffast-math`, which `-O2`/`-O3` do not
imply. Evidence: the **full test matrix is green at both levels** (172 unit
tests, 6/6 binaries, all 4 golden gates — including `test_kernels`' exact-f32
goldens `-93.3486328` / `1056.1875`, precisely what an FP-semantics change would
break), and **two real prompts are byte-identical across O1/O2/O3** (p48, 553 B;
p49, 565 B; same sha256; 64 generated tokens each). A third (p45) was started
and exceeded the 600 s call ceiling — it is not evidence either way and is not
counted.

## 3. `stoke bench --ops`' reference stream measures the wrong thing

This is the finding that matters more than the flag.

```c
static void run_ref_stream(void* p) {
    f32 acc = 0.0f;
    for (u64 i = 0; i < c->n; i++) acc += c->buf[i];
    g_sink = acc;
}
```

That is a **serial floating-point reduction**: every add depends on the previous
one, and GCC may not reassociate it without `-ffast-math`, so it cannot be
vectorized at any optimization level. **Its rate is f32-add latency, not memory
bandwidth.**

Measured on the same 256 MB buffer, median of 7, same box:

| variant | `-O1` GB/s | `-O3` GB/s | |
|---|---:|---:|---|
| `serial_f32` | 7.43 | **7.40** | ← bench.c's reference stream |
| `sum_i64` | 9.43 | 12.86 | integer add is associative, so it vectorizes |
| `split8_f32` | 12.53 | 15.56 | same f32 arithmetic, dependency chain broken |
| `xor_u64` | 13.57 | **16.15** | 1-cycle latency; only the load ports bind |

**This box's single-thread read bandwidth is at least ~16 GB/s. The reference
stream reports 7.4.** Every `%ref` figure this project has recorded is against
a denominator roughly **2.15× too low**, and `%ref` has never meant "fraction of
the memory ceiling".

**Corrected picture** (matvec median GB/s against ~16.15 GB/s, not against the
reference stream):

| | matvec GB/s | % of reference stream | **% of real bandwidth** | **headroom** |
|---|---:|---:|---:|---:|
| `-O1` | 2.31 | 30% | **14%** | **~7.0×** |
| `-O2` | 6.12 | 79% | **38%** | **~2.6×** |
| `-O3` | 6.92 | 88% | **43%** | **~2.3×** |

So the total single-thread headroom at `-O1` was ~7×, of which **~3.1× is the
build flag and ~2.3× remains for K3 to collect at `-O3`** (3.1 × 2.3 ≈ 7, which
is the consistency check).

**K1's qualitative conclusion survives: the matvecs are not bandwidth-bound and
SIMD has real headroom to collect. Its magnitude did not — "~2.8×" was measured
against a broken denominator and happened to land near the right answer for the
wrong reason.** The `-O3` figure is a lower bound on headroom in the useful
direction: `xor_u64` is not necessarily the true ceiling either, and anything
faster only increases the headroom.

### 3.1 How this was nearly missed, which is the transferable part

The first version of this brief cited the reference stream's **invariance across
`-O` levels** — 7.63 / 7.77 / 7.84 GB/s — as an *internal control*: "it is a
plain `acc += buf[i]` loop, so if it were code-limited it would have sped up
like everything else; it did not, because it is already at DRAM." On that basis
it concluded the matvecs at 82–102% of it were at the ceiling, and that K3 had
nothing left to collect.

**The invariance was the symptom, not the control.** "It didn't change" is
produced equally by *this is a stable reference* and by *this is measuring
something the change cannot affect*. The reference stream is invariant to `-O`
because a serial FP dependency chain is invariant to `-O`.

**Durable: an invariant cited as a control must have its invariance EXPLAINED
before it is trusted, not merely observed.** Same family as the 08-02 lesson
where `max/med = 1.33` read as healthy while the median was 2.6× wrong — a
statistic that looks reassuring precisely when it is blind.

## 4. What I think follows — Fable may overrule any of it

1. **Raise the default optimization level.** `-O2` captures 89% of the gain;
   `-O3` adds ~10% more at no observed cost. Recommend `-O3`. Open sub-question
   I do not think I should settle alone: whether the **sanitizer legs** stay at
   `-O1` (ASan/TSan diagnostics are conventionally read at low `-O`, and the
   Makefile's "one BUILDDIR per (platform, sanitizer)" discipline would extend
   naturally to an optimization axis), or whether the whole matrix moves
   together so that what ships is what is sanitized.
2. **Fix the reference stream before any further `%ref` is recorded**, and treat
   it as the higher-priority repair. It is the denominator of the attribution
   ladder that M3's miss-the-bar procedure depends on. `split8_f32` is the
   natural replacement — same f32 arithmetic, dependency chain broken — or
   `xor_u64` if the intent is explicitly "load-port ceiling". Whichever is
   chosen, the existing rows' `%ref` column becomes non-comparable and should be
   marked rather than silently reinterpreted.
3. **K3 keeps its justification.** ~2.3× of single-thread headroom at `-O3` is
   real and is what SIMD collects. The brief's earlier draft argued the
   opposite; it was wrong. What *is* worth restating is that **batched decode is
   where SIMD pays most** — at batch > 1 the same weight bytes serve many
   tokens, arithmetic intensity rises, and the kernel moves further from
   memory-bound. Single-stream is the workload where SIMD matters *least*, and
   it is the one the M3 bar measures.
4. **The arithmetic to the bar now closes plausibly.** 4.7× remaining at `-O3`,
   against ~2.3× (SIMD) × ~1.7× (Q4_K_M's smaller weights) ≈ 3.9× before the
   pool contributes anything at all. K5 supplies the rest. Under the `-O1`
   numbers this needed 14.6× and looked marginal; it no longer does.
5. **K-quants are unmeasured in this respect.** Whether Q4_K/Q6_K dequant is
   bandwidth-bound at `-O3` is not known and should be measured before K4's SIMD
   is scoped, rather than assumed to behave like Q8_0.

## 5. Consequences worth knowing before deciding

- **`history.jsonl` records `binary_sha256` but NOT the compile flags.** The two
  existing rows are `-O1` rows and nothing in the file says so, so they cannot be
  compared against any future `-O2`/`-O3` row. The schema needs a build-flags
  field *before* the level is raised. This does not violate the standing
  "provenance is not in the C" principle: a git commit is external state a
  process would have to guess, whereas its own compile flags are a property of
  the binary and can be embedded at compile time.
- **The pre-SIMD baseline row will need re-taking at whatever level is chosen**,
  since it exists to be the thing K3's result is diffed against.
- **The deferred 50-prompt oracle gate gets ~3× cheaper at `-O3`** (~4–7 h →
  ~1.5–2.5 h) with a **byte-identical verdict**, so if the level is being raised
  anyway, raising it before the gate run saves most of a night. The gate is on
  hold and is Kabir's to launch.
- **The sequencing pin** ("K2's gate runs on the single-threaded integer core,
  before the pool") is unaffected by an `-O` change, since output is identical.
  The `-O1` integer-core binary is archived at
  `build/stoke-k2-intcore-2ed3d1e8.exe` either way.

## 6. Not up for revisiting

Kabir owns these and this brief does not touch them: the **≥7.8 tok/s bar**
(unchanged — the code got faster, the bar did not move), ε, the waiver budget,
the corpus, the parity protocol, push policy, and the model/owner line. Raising
an optimization level is not gate-loosening; if any part of this reads as moving
a bar after seeing results, that reading should win and the change should not
happen.

## 7. Reproducing this

```
mingw32-make BUILDDIR=build/win-O3 \
  CFLAGS="-std=c11 -Wall -Wextra -Werror -g -O3 -Isrc" all test
build/win-O3/stoke.exe bench --ops models/<Q8_0>.gguf --jsonl
```

The bandwidth probe of §3 is `tests/tooling/bwprobe.c` — standalone, not part of
the build:

```
gcc -std=c11 -Wall -Wextra -Werror -g -O3 -o bwprobe tests/tooling/bwprobe.c
./bwprobe
```

A second run taken while writing this gave `serial_f32` 7.31, `split8_f32`
16.58, `sum_i64` 13.58, `xor_u64` 14.56 — **`split8_f32` and `xor_u64` trade
places between runs at ~15–16.5 GB/s, so treat the ceiling as "at least ~15,
around 16" rather than a fixed figure; `serial_f32` sits at 7.3–7.4 in every
run and never moves.** The conclusion does not depend on which of the two fast
variants is highest, only on both being roughly 2.1× the reference stream.

Both variant build trees are left in place at `build/win-O2` and `build/win-O3`
(gitignored).
