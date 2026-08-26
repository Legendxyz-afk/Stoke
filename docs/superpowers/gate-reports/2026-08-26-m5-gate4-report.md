# M5 gate item [4] -- S3 fanout, measured

Run `2026-08-26-m5-gate4`. S3 = 8 clients x ~200 user tokens over a 1024-token SHARED system
prefix, 300 generated, ctx 1600, threads 8, kv f32, unpinned. Bar **13.3**
(M0's llamacpp-default cell). Raw `2026-08-26-m5-gate4-raw.txt`, bank `2026-08-26-m5-gate4-bank.jsonl`.

**The gate arm is Sd (`--dedup`).** The bar was set by llama-server in its
defaults, where `--slot-prompt-similarity` (0.10), `--cache-ram` (8192 MiB) and
`--cache-idle-slots` are all ON, so the referent was never charged eight fresh
prefills of the shared prefix. **Sn is the FLOOR, not a gate result.**

| arm | AGGREGATE(M0) tok/s | vs bar 13.3 | TTFT p50 s | ITL p50 ms | ITL max ms | rows | adopted |
|---|---|---|---|---|---|---|---|
| Sd (dedup) -- GATE | 13.535 | 1.02x | 75.1 | 403.0 | 5,118 | 5128 | 7168 |
| Sn (no dedup) -- floor | 5.926 | 0.45x | 204.4 | 423.0 | 5,050 | 12296 | 0 |

**What dedup bought: 2.28x** (5.926 -> 13.535 tok/s aggregate).

**Rows the gate arm did not run: 7,168 (12,296 -> 5,128), against 7,168 positions reported adopted -- matches exactly.**
Prefill chunks 80 -> 24: 56 chunks removed at 128 tokens each = 7,168.

## Attribution -- the static path on the same prompts (`batch --dedup`)

- B decode aggregate: **20.180** tok/s; B prefill: **28.21** tok/s
- Sd decode-only (N / ITL p50): **19.85** tok/s

Read B's decode against Sd's decode-only, NOT against Sd's AGGREGATE(M0):
a lockstep batch's span carries no prefill at all, while the scheduler's carries
every later arrival's. If B's decode already misses the bar, the scheduler was
never the problem -- that is the playbook ladder's first rung.

## VERDICT (gate arm Sd): **MET** at 13.535 >= 13.3

Gate-arm blocks, worst to best: 13.457, 13.535, 13.757. **3 of 3 clear the bar on their own.**

**Streams across every stoke unit: IDENTICAL.**
Both arms wrote their ids; a dedup arm that moved the stream would make every
rate here a rate for a different computation. Each file carries per-sequence
ids AND the decoded text, so the compare covers both.

!! **THIS IS NOT AN INDEPENDENT CORRECTNESS GATE, AND MUST NOT BE QUOTED AS ONE.**
S3's eight clients differ by ONE DIGIT, so all eight streams inside a file are
the same stream, and the continuation is a reply to a filler word list that does
not depend on the shared prefix. Measured 2026-08-25: on S3 a mutant that zeroes
the whole adopted K/V is visible only from 8x24 up, and a mutant that skips 16
unwritten positions is NEVER visible here. The dedup correctness gate is the
HETERO fixture (`run_m5_dedup_verify.ps1`, green on all four legs at `4dd7e2f`),
where both mutants differ. What this check buys is that the two arms above priced
the same computation.

**And what the same units say about gate [3], which is a different gate:** the
gate arm's ITL p50 is **403 ms** and its max **5,118 ms** against a 250 ms bound,
with TTFT p50 **75 s**. [4] is throughput; on this same data [3] MISSES on the
median as well as the max. Item [4] being met says the fanout is SERVED, not that
it is served interactively.

## By block (column 1 against the rest: block 1 is the warming transient)

| arm | block 1 | block 2 | block 3 |
|---|---|---|---|
| Sd | 13.757 | 13.535 | 13.457 |
| Sn | 5.915 | 5.926 | 5.928 |
| L | 18.300 | 18.210 | 18.400 |

**Box health (llama-bench b10037, t 8):** tg128 median 18.3, pp512 median 93.74 tok/s.
A stoke-only collapse would move both stoke arms and leave this one alone.

## Every unit

| id | arm | AGG(M0) | wall s | cpu s | standby GB | cpu % | adopted |
|---|---|---|---|---|---|---|---|
| b1-Sd-s8n300 | Sd | 13.757 | 218.6 | 1489.5 | 0.02 | 30 | 7168 |
| b1-Sn-s8n300 | Sn | 5.915 | 450 | 3063.6 | 0 | 2 |  |
| b1-B-s8n300 | B | 20.299 | 217.1 | 1473.3 | 0.07 | 31 | 7168 |
| b1-L-s8n300 | L | 18.3 | 52.6 | 347 | 1.21 | 37 |  |
| b2-Sn-s8n300 | Sn | 5.926 | 449.4 | 3061.3 | 0.08 | 36 |  |
| b2-B-s8n300 | B | 19.945 | 223 | 1502.4 | 0.01 | 15 | 7168 |
| b2-Sd-s8n300 | Sd | 13.535 | 222.3 | 1496.5 | 0 | 4 | 7168 |
| b2-L-s8n300 | L | 18.21 | 52.4 | 347.9 | 0 | 17 |  |
| b3-B-s8n300 | B | 20.18 | 218.8 | 1483.2 | 0 | 17 | 7168 |
| b3-Sd-s8n300 | Sd | 13.457 | 222.8 | 1498.3 | 0.01 | 61 | 7168 |
| b3-Sn-s8n300 | Sn | 5.928 | 449.4 | 3066 | 0 | 10 |  |
| b3-L-s8n300 | L | 18.4 | 50.3 | 348.2 | 0.4 | 9 |  |
