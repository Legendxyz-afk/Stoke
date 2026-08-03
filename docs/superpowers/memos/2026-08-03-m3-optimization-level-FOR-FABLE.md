# M3 — the build is `-O1`, and that changes what K3 is for

**Status:** decision brief. Written by the Opus 5 manager session of 2026-08-03,
parked for Kabir to hand to Fable per the 07-28 escalation routing. **Nothing
has been changed in the build on the strength of this.** The measurements below
are landed and reproducible; the re-plan they imply is not mine to make.

**Escalation trigger:** *"results contradict spec physics."* K1's central
finding — the one M3's whole shape rests on — does not survive a change of
compiler flag.

---

## 1. What was measured

`stoke` has always been built with a single `CFLAGS` line in the Makefile:

```
CFLAGS := -std=c11 -Wall -Wextra -Werror -g -O1 -Isrc
```

There is no optimized or release variant. Every number this project has ever
recorded — M0's premise work, K1's microbench, both `bench/history.jsonl` rows,
the 13.3× and 14.5× gaps to the bar — comes from an `-O1` binary.

The vendored llama.cpp it is benchmarked against (`bench/vendor/llama-b10037/`)
ships as arch-dispatched release DLLs (`ggml-cpu-alderlake.dll` and siblings).

Three builds, identical source, interleaved runs (`O1,O2,O3, O1,O2,O3, …`) so
box drift cannot masquerade as a flag effect. Medians of 3, quiet box.

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

## 2. The part that changes the plan

K1's headline, repeated through the ledger and into M3's design, is that the
projections run at **32–37% of the reference stream**, therefore are **not
bandwidth-bound**, therefore carry **~2.8× of headroom that SIMD exists to
collect**. Re-measured across the three builds:

| op | `-O1` %ref | `-O2` %ref | `-O3` %ref |
|---|---:|---:|---:|
| matvec attn_q | 30.8 | 85.6 | **94.4** |
| matvec attn_k | 31.4 | 90.4 | **101.3** |
| matvec attn_v | 31.6 | 89.8 | **101.8** |
| matvec attn_out | 30.7 | 85.5 | **93.9** |
| matvec ffn_gate | 29.3 | 78.4 | **85.0** |
| matvec ffn_up | 29.6 | 76.3 | **84.8** |
| matvec ffn_down | 29.4 | 74.9 | **85.6** |
| matvec LOGITS | 28.0 | 74.5 | **81.6** |

**At `-O3` the Q8_0 matvecs run at 82–102% of the single-thread reference
stream. They are at the single-thread memory-bandwidth ceiling. The ~2.8×
"scalar signature" K3 was scoped to collect is an artifact of `-O1`.**

Two internal controls say this is real rather than a measurement accident:

- **The reference stream itself does not move** — 7.63 / 7.77 / 7.84 GB/s across
  O1/O2/O3. It is a plain `acc += buf[i]` loop, so if it were code-limited it
  would have sped up like everything else. It did not, because it is already at
  DRAM. That is what makes it a legitimate denominator, and it is why the
  matvecs converging on it means what it appears to mean.
- **attn_k and attn_v exceed 100%** (101.3, 101.8). Those are the smallest
  weight matrices and partly fit in cache, so beating a DRAM-resident stream is
  the expected behaviour, not an error — but it does mean %ref should be read as
  "at the ceiling", not as a precise ratio, for those two rows.

## 3. Numerics are unchanged — checked, not assumed

GCC does not reassociate floating point without `-ffast-math`, which is not
implied by `-O2`/`-O3`. Verified rather than argued:

- **Full test matrix green at both `-O2` and `-O3`**: 172 unit tests, 6/6
  binaries, all 4 golden gates. This includes `test_kernels`' exact-f32 goldens
  (`matvec_q8_0_dispatch_two_rows` reads `-93.3486328` / `1056.1875`), which are
  precisely what a change in FP semantics would break.
- **Real prompts byte-identical across O1/O2/O3**: p48 (553 bytes of output) and
  p49 (565 bytes), 64 generated tokens each, same sha256. A third (p45) was
  started and exceeded the 600 s call ceiling; it is not evidence either way and
  is not counted.

So this is a free speedup, not a trade.

## 4. What I think follows — Fable may overrule any of it

1. **Raise the default optimization level.** `-O2` captures 89% of the available
   gain; `-O3` adds ~10% more with no observed cost. Recommend `-O3`, or `-O2`
   if there is a reason to prefer the more conservative pass set that I am not
   seeing.
2. **K3's justification changes and should be restated before it is built.**
   SIMD cannot collect headroom that is not there. What survives, and I think
   genuinely does survive:
   - **Batched decode is compute-bound, and stoke is a batching engine.** At
     batch > 1 the same weight bytes serve many tokens, arithmetic intensity
     rises, and the matvec stops being bandwidth-bound. M4/M5 are where SIMD
     pays, and they are the milestones the whole project is actually for. The
     single-stream bar is the one workload where SIMD matters least.
   - **K-quants (K4) are not the same shape.** Q4_K/Q6_K dequant does real work
     per byte; whether it is bandwidth-bound at `-O3` is unmeasured and should
     be measured before K4's SIMD is scoped.
   - **The exact-diff gate is still worth having** regardless, as the mechanism
     that lets a SIMD path be adopted without re-running parity.
3. **K5 (the pool) is now the single-stream lever, not K3.** More cores means
   more aggregate bandwidth. The remaining 4.7× at `-O3` against Q4_K_M's ~1.7×
   smaller weights leaves ~2.8× to find in threads — which is a very different
   proposition from the 14.6× that the `-O1` numbers implied, and it is
   plausible on 6P+4E.
4. **Sequencing question I cannot answer alone:** if K3 is demoted or deferred,
   does M3 still close on the forward note's gate [1] (scalar↔SIMD exact-diff),
   or does that gate move to whichever milestone actually lands SIMD?

## 5. Consequences worth knowing before deciding

- **The two existing `history.jsonl` rows are `-O1` rows and cannot be compared
  against any future `-O2`/`-O3` row.** The schema records `binary_sha256` but
  **not the compile flags**, so nothing in the file distinguishes the regimes.
  If the level is raised, the schema needs a build-flags field first, or the
  history becomes silently un-interpretable at exactly the point it starts
  mattering. (The existing "provenance is not in the C" principle does not
  forbid this: a git commit is external state a process would have to guess,
  whereas its own compile flags can be embedded at compile time and are a
  property of the binary.)
- **The deferred 50-prompt oracle gate gets ~3× cheaper.** It is Kabir's to
  launch and is currently on hold. At `-O1` it sizes at ~4–7 h; at `-O3`,
  ~1.5–2.5 h. Output is byte-identical, so the gate's verdict is unaffected by
  which binary runs it. If the level is being raised anyway, raising it *before*
  the gate run saves most of a night.
- **`-O1` was almost certainly never a decision.** It appears once, in the
  original Makefile, described as mirroring `../loom/Makefile`. There is no
  note anywhere defending it, and `-g -O1` is a debugging default. This looks
  like an inherited convenience rather than a considered choice — which is
  itself worth saying out loud, because it means there is probably no
  constraint behind it that I am failing to see.

## 6. Not up for revisiting

Kabir owns these and this brief does not touch them: the **≥7.8 tok/s bar**
(unchanged — the code got faster, the bar did not move), ε, the waiver budget,
the corpus, the parity protocol, push policy, and the model/owner line. Raising
an optimization level is not gate-loosening; if any part of this reads as
moving a bar after seeing results, that reading should win and the change
should not happen.

## 7. Reproducing this

```
mingw32-make BUILDDIR=build/win-O3 \
  CFLAGS="-std=c11 -Wall -Wextra -Werror -g -O3 -Isrc" all
mingw32-make BUILDDIR=build/win-O3 \
  CFLAGS="-std=c11 -Wall -Wextra -Werror -g -O3 -Isrc" test
build/win-O3/stoke.exe bench --ops models/<Q8_0>.gguf --jsonl
```

Both variant build trees are left in place at `build/win-O2` and `build/win-O3`
(gitignored). The `-O1` integer-core binary the oracle gate should run against
is archived at `build/stoke-k2-intcore-2ed3d1e8.exe`.
