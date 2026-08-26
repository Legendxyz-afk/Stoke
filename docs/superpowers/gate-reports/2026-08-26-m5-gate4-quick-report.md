# M5 gate item [4] -- S3 fanout, measured

Run `2026-08-26-m5-gate4-quick`. S3 = 8 clients x ~200 user tokens over a 1024-token SHARED system
prefix, 16 generated, ctx 1600, threads 8, kv f32, unpinned. Bar **13.3**
(M0's llamacpp-default cell). Raw `2026-08-26-m5-gate4-quick-raw.txt`, bank `2026-08-26-m5-gate4-quick-bank.jsonl`.

**The gate arm is Sd (`--dedup`).** The bar was set by llama-server in its
defaults, where `--slot-prompt-similarity` (0.10), `--cache-ram` (8192 MiB) and
`--cache-idle-slots` are all ON, so the referent was never charged eight fresh
prefills of the shared prefix. **Sn is the FLOOR, not a gate result.**

| arm | AGGREGATE(M0) tok/s | vs bar 13.3 | TTFT p50 s | ITL p50 ms | ITL max ms | rows | adopted |
|---|---|---|---|---|---|---|---|
| Sd (dedup) -- GATE | 3.028 | 0.23x | 50.8 | 125.8 | 4,812 | 1482 | 1024 |
| Sn (no dedup) -- floor | 0.724 | 0.05x | 82.0 | 129.5 | 4,723 | 2506 | 0 |

**What dedup bought: 4.18x** (0.724 -> 3.028 tok/s aggregate).

## Attribution -- the static path on the same prompts (`batch --dedup`)

- B decode aggregate: **15.708** tok/s; B prefill: **29.13** tok/s
- Sd decode-only (N / ITL p50): **15.9** tok/s

Read B's decode against Sd's decode-only, NOT against Sd's AGGREGATE(M0):
a lockstep batch's span carries no prefill at all, while the scheduler's carries
every later arrival's. If B's decode already misses the bar, the scheduler was
never the problem -- that is the playbook ladder's first rung.

## VERDICT (gate arm Sd): **MISS** at 3.028 = 0.23x of 13.3

**Streams across every stoke unit: IDENTICAL.**
Both arms wrote their ids; a dedup arm that moved the stream would make every
rate here a rate for a different computation.

**Box health (llama-bench b10037, t 8):** tg128 median 18.24, pp512 median 93.4 tok/s.
A stoke-only collapse would move both stoke arms and leave this one alone.

## Every unit

| id | arm | AGG(M0) | wall s | cpu s | standby GB | cpu % | adopted |
|---|---|---|---|---|---|---|---|
| b1-Sd-s2n16 | Sd | 3.028 | 56.6 | 340.5 | 0.55 | 3 | 1024 |
| b1-Sn-s2n16 | Sn | 0.724 | 87.4 | 550.8 | 0.01 | 16 |  |
| b1-B-s2n16 | B | 15.708 | 55.1 | 335.2 | 0.48 | 21 | 1024 |
| b1-L-s2n16 | L | 18.24 | 53.6 | 346.9 | 0.93 | 18 |  |
