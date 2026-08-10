# M3 [2v2-iii] — Q4_K_M reference grid: gate report

**Verdict: the evidence is complete and ZERO BUG-CLASS. One precondition of
the §7.5 stamp is NOT met — the flip tripwire fired (9 ≥ 8) — and its
disposition is the separate anomaly memo
`2026-08-11-m3-tripwire-anomaly-memo.md`.**

## 1. Referents, named as §7.5 requires

| thing | identity |
|---|---|
| engine under test | `build/win/stoke.exe` sha256 `876ce53fd4a943fdd820f0bf9e8328b8aa751fa1269df9fd20f4fedbfb6390ff` |
| reference | `tests/tooling/ref_forward.py` sha256 `d4a038e5ad0412dd76f7f84b…` (numpy f32) |
| model | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, 2,497,280,736 bytes |
| pair key | `f32_intcore_q4_k_m` (never Q8_0's; τ does not transfer — G6) |
| tree | `2d98c58`, working tree clean |

G11 freeze intact: the binary above is unchanged since 2026-04-08 and was
verified by sha256 at the start of every leg and every probe.

## 2. The grid as the ruling fixed it

p00 / p10 / p20 / p30 / p45 + p41, `--probe-all`, one required continuation
per first divergence. All six probe legs and all five continuations ran;
p41 needed no continuation.

| leg | prompt_len | covered / 64 | flips | first divergence | continuation |
|---|---:|---:|---:|---|---|
| p00 | 23 | 64 | 1 | idx 9 (abs 32) | clean through 63 |
| p10 | 24 | 45 | 2 | idx 19 (abs 43) | flipped idx 44 (abs 68) |
| p20 | 64 | 25 | 2 | idx 21 (abs 85) | flipped idx 24 (abs 88) |
| p30 | 53 | 53 | 2 | idx 18 (abs 71) | flipped idx 52 (abs 105) |
| p45 | 14 | 39 | 2 | idx 26 (abs 40) | flipped idx 38 (abs 52) |
| p41 | 477 | 64 | 0 | **none — clean 64/64** | not required |
| **total** | | **290** | **9** | | |

**Per-prompt covered positions are stated above because the ruling requires
the tripwire's denominator to be explicit.** Coverage is 290, not the ~384 the
ruling projected: a leg goes dark after its SECOND divergence and §5
authorizes only one continuation, so four of the five continuations closed
their leg rather than restoring it.

Measured flip rate: **9 / 290 = 3.10 %/token.**

## 3. τ_Q4KM — frozen BEFORE anything was graded

Selected by §5's rule (every ⌈N/20⌉-th agreeing position in prompt-order,
index-order), stoke's side fetched with `--probe-pos` on the frozen binary,
Δ-profile anchored on HEAD's chosen id per §3.6.

| band | agreeing available | positions used | anchored values | max | p95 | **τ_Q4KM** |
|---|---:|---:|---:|---:|---:|---:|
| `[0,100)` | 213 | 20 | 97 | 0.893974 | 0.511869 | **1.0237387687988289** |
| `[100,300)` | 69 | 20 | 93 | 0.884807 | 0.589047 | **1.1780941621093746** |
| `[300,600)` | 64 | 20 | 98 | 0.499857 | 0.405716 | **0.8114320996093838** |

Frozen 04:12:50, before the divergence records were assembled (04:15:30) and
before any grading. The tool refuses to freeze and build records in one
invocation, so the ordering is structural rather than remembered.

τ is ~2× Q8_0's frozen values (0.519674 / 0.413566 / 0.394898) — the expected
direction for 4-bit against 8-bit, and a concrete reason τ may never transfer
between files.

**Band `[100,300)` required added samples.** Only p20 and p30 can reach it at
all; p20 died at index 24 and p30's ceiling there is 17 positions, so 20 was
unreachable from the fixed grid however the run had gone — and p30's
continuation put a divergence there at abs 105. Per §5's fail-safe ("pending a
freeze from added samples of that band"), `p32` (104 tokens, all 64 generated
positions at abs 104–167) was run; it agreed with HEAD at every position and
took the band from 5 to 69. `p22` was run in parallel and diverged at index 1,
contributing nothing to the band. **Added samples fed the envelope only — never
coverage, flips, or the tripwire** (see `e458094`); the grid totals in §2 are
unchanged by their presence, which was verified rather than asserted.

## 4. Classification — §6 with τ_Q4KM, G9/G10 readings, §11 ladder

| prompt | idx | abs | band | verdict | worst margin | τ | margin/τ |
|---|---:|---:|---|---|---:|---:|---:|
| p00 | 9 | 32 | `[0,100)` | ARITHMETIC-CLASS | 0.092613 | 1.023739 | 0.09 |
| p10 | 19 | 43 | `[0,100)` | ARITHMETIC-CLASS | 0.306267 | 1.023739 | 0.30 |
| p10 | 44 | 68 | `[0,100)` | ARITHMETIC-CLASS | 0.158798 | 1.023739 | 0.16 |
| p20 | 21 | 85 | `[0,100)` | ARITHMETIC-CLASS | 0.056314 | 1.023739 | 0.06 |
| p20 | 24 | 88 | `[0,100)` | ARITHMETIC-CLASS | 0.086805 | 1.023739 | 0.08 |
| p30 | 18 | 71 | `[0,100)` | ARITHMETIC-CLASS | 0.410904 | 1.023739 | 0.40 |
| p30 | 52 | 105 | `[100,300)` | ARITHMETIC-CLASS | 0.098709 | 1.178094 | 0.08 |
| p45 | 26 | 40 | `[0,100)` | ARITHMETIC-CLASS | 0.555672 | 1.023739 | 0.54 |
| p45 | 38 | 52 | `[0,100)` | ARITHMETIC-CLASS | 0.103886 | 1.023739 | 0.10 |

**9/9 ARITHMETIC-CLASS. ZERO BUG-CLASS. No ESCALATE, so the L2/L3 ladder was
not needed.** The worst case sits at 54 % of its τ; these are not knife-edge
calls. Every top-5 set agreed modulo the rank-5 tolerance, so no foreign core
candidate appeared anywhere.

**Supplementary, outside the criterion:** p22's divergence at abs 89 also
grades ARITHMETIC-CLASS (worst 0.034645 vs τ 1.023739). It is reported because
discarding an observation merely because the sample was added at my own
initiative would be too convenient — but it is excluded from the counts above,
from the tripwire, and from the zero-BUG criterion, which are the fixed grid's.

## 5. §7.5 stamp preconditions

| # | item | state |
|---|---|---|
| 1 | [1] test_simd 12/12, all quant kinds | ✅ `e26ff0d` |
| 2 | [2v2] probe grids both files, τ frozen first, zero BUG | ✅ this report |
| 2 | [2v2] **flip tripwire quiet** | ❌ **FIRED at 9 ≥ 8** — see anomaly memo |
| 2 | [2v2] Q4_K_M streams banked under HEAD's sha256 | ✅ 50/50 |
| 3 | [3] bar re-anchored 8.08–8.46, both legs GATE-grade | ✅ `86fd2fa` |
| 4 | [4] full matrix green at last source-touching commit | ✅ ledger: Windows 9/9 + 4 goldens, WSL plain/asan/tsan all green, zero findings, at `5051878`; no `src/` file has changed since |
| 5 | [5] microbench table + `history.jsonl` GATE row | ✅ |
| 6 | entry chores | ✅ `d9b633f`, `aa9d6ff`, `d0f8a6b` |

**Everything mechanical is satisfied. The stamp waits on one judgement, and
that judgement is the anomaly memo's.**
