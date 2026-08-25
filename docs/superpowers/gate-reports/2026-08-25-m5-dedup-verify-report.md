# M5 -- prefix dedup on the serve path: the correctness gate

Run `2026-08-25-m5-dedup-verify`, 8 seqs x 24 generated, ctx 1600, threads 8, Q4_K_M.
Raw log `2026-08-25-m5-dedup-verify.txt`. Arms are ONE binary differing in one runtime flag.

| leg | prompts | adopted | rows A -> B | dedup vs no-dedup | mutant kv | mutant skip16 | client streams |
|---|---|---|---|---|---|---|---|
| s3 | 8 clients | 7168 of 9840 | 10024 -> 2856 | **SAME** | DIFFERS | same | 1 distinct of 8 |
| hetero | 8 clients | 8064 of 9357 | 9541 -> 1477 | **SAME** | DIFFERS | aborted | 8 distinct of 8 |

**Leg 4 (lineage).** serve, 1 seq, no dedup, vs the m4.5 archive's `stoke run` on p00: **SAME**.

**What leg `s3` cannot see.** M0's S3 clients differ by ONE DIGIT
(`[client 0]` vs `[client 1]`) inside an otherwise byte-identical word list,
so all eight generate the same ids and a cross-client adoption error is
indistinguishable from correct behaviour there. That is why leg `hetero` exists:
S3's real shared system prefix (same 64 adoptable blocks) with eight genuinely
different bodies, whose reference streams are asserted pairwise distinct.

**The mutants.** `kv` zeroes the K/V under the whole adopted prefix at admission
(every layer, every kv head) and asks the question the gate rests on: would this
comparison notice if the adopted blocks held the wrong K/V? It MUST differ.
`skip16` skips one block MORE than it adopted, over positions that are backed but
never written -- the failure `kv.h` names as the one that leaves the run alive and
the output merely wrong. **It is INVISIBLE here** (K = 0 gets softmax weight exp(0)
against exp(q.k) for 1,222 real positions), so it is recorded rather than required:
what covers that failure is the `kv_len(kv) != adopted` guard in `sched.c` and the
sim's position ledger, both of which DO catch it.

## VERDICT: GREEN -- adoption on the serve path does not move the stream.
