# Q6_K `vpmaddubsw` kernel -- the stream gate

Run `2026-08-27-m5-q6k-gate-postflip`, 11 cases, every one byte-compared against
`stoke-m45-gate-fddd12cd.exe` -- a binary built BEFORE this kernel existed, whose
streams the M3/M4/M4.5 gates already certified. **An arm is graded against an
ARCHIVE, never against another arm of the same build**, because agreement among
new arms is also consistent with all of them being wrong together.

Binaries under test: `stoke.exe` / `stoke.exe` (bank key `B1E99979A25D`), commit `1000e37`. **The bank is
keyed by both binaries' content**, so a rebuild starts a new bank instead of
resuming units measured against code that no longer exists.

| case | ref vs archive | maddubs vs archive | mutant | kernel dispatched | archive sha |
|---|---|---|---|---|---|
| `run-p00-f32-t2` | **MATCH** | **MATCH** | DIFFERS (good) | yes (banner) | `25894278F83D` |
| `run-p00-f32-t6` | **MATCH** | **MATCH** | DIFFERS (good) | yes (banner) | `25894278F83D` |
| `run-p00-q8-t2` | **MATCH** | **MATCH** | DIFFERS (good) | yes (banner) | `CB29BF73B70C` |
| `run-p00-q8-t6` | **MATCH** | **MATCH** | DIFFERS (good) | yes (banner) | `CB29BF73B70C` |
| `run-p41-f32-t2` | **MATCH** | **MATCH** | DIFFERS (good) | yes (banner) | `13E71FB4126F` |
| `run-p41-f32-t6` | **MATCH** | **MATCH** | DIFFERS (good) | yes (banner) | `13E71FB4126F` |
| `run-p41-q8-t2` | **MATCH** | **MATCH** | DIFFERS (good) | yes (banner) | `AB7028BE94D0` |
| `run-p41-q8-t6` | **MATCH** | **MATCH** | DIFFERS (good) | yes (banner) | `AB7028BE94D0` |
| `batch-s2` | **MATCH** | **MATCH** | DIFFERS (good) | yes (banner) | `2CFC3CAA1233` |
| `batch-s8` | **MATCH** | **MATCH** | DIFFERS (good) | yes (banner) | `07F802D8579A` |
| `batch-s16` | **MATCH** | **MATCH** | DIFFERS (good) | yes (banner) | `D831F81C014C` |

## The claims

| # | claim | result |
|---|---|---|
| 1 | the OLD kernel still reproduces the archive (nothing else moved) | **PASS** |
| 2 | the NEW kernel reproduces the archive, byte for byte | **PASS** |
| 3 | the mutant DIFFERS on every case -- the error reaches the STREAM | **PASS** (11/11) |
| 4 | the batched Q6_K kernel was DISPATCHED in every case (mutant banner) | **PASS** (11/11) |

**Claims 3 and 4 are different questions and the draft of this runner asked only
the first.** Claim 4 reads the mutant's stderr banner, which `q6k_select()` emits
from exactly one call site -- `vec.c:374`, inside `simd && n_seq > 1` and
`!is_q4` -- so it means the batched Q6_K AVX2 kernel was *dispatched*, whatever
the output did afterwards. Without it, a `SAME` mutant cell has two causes that
the table cannot separate and the report would have announced the wrong one:
the kernel was never reached (the cell grades nothing), or the kernel ran WRONG
and the stream did not care (**the fixture is blind**). The second is not
hypothetical here -- zeroing 1024 positions of adopted K/V across 36 layers once
produced byte-identical ids on this project. 3 PASS + 4 PASS is the only
combination in which claims 1 and 2 carry their full weight.

## VERDICT: **the default may flip**

**Claim 3 is why 1 and 2 mean anything.** A case that never reaches the Q6_K
batched kernel compares EQUAL to the archive whatever the kernel does -- a green
cell that tests nothing. The mutant swaps the two scale-group halves of
`scv[c]`, which is the mistake this formulation invites: Q4_K's eight scales line
up with its eight 32-byte chunks exactly, while Q6_K has SIXTEEN, so a chunk
spans TWO groups and the halves must match the halves `vpmaddubsw` produced.
It leaves the run alive and merely wrong -- the class `kv.h` names as dangerous.

`stoke run` reaches a BATCHED kernel because E3 chunks PREFILL through
`model_forward_batch` and `vec.c` tiles at 16, so `run` enters `matvec_batch`
at n_seq=16 then the remainder -- the 2026-08-24 correction, and what makes the
archives (which only speak `run`) gradeable for this at all.
