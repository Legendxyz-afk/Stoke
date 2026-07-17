# stoke M0 baselines — Qwen3-4B-Instruct-2507 Q4_K_M — 2026-07-17

All four baseline configs (ollama-default, ollama-tuned, llamacpp-default, llamacpp-tuned) serve the same GGUF over the same locked scenario set, and the engine process is restarted between every cell so no cache or KV state leaks across configs or scenarios. Each cell runs 1 discarded warmup request followed by N measured runs, and the numbers below are the median run by aggregate decode tok/s. Prompts are salted per run so engines with prefix/prompt caching cannot skip prefill on runs 2..N and silently inflate them, while S3's shared system prefix stays identical across clients within a single run (the capability S3 measures). Every request carries a per-request socket timeout. The published aggregate decode tok/s is decode-only -- it excludes each client's own first (TTFT/prefill) token -- and token counts come from the engine's own reported usage.completion_tokens (authoritative even when an engine coalesces multiple decode steps into one SSE chunk), while chunk arrival times still drive TTFT/ITL -- so ITL here is inter-arrival latency, and a coalesced multi-token chunk contributes a single gap. The underlying timing/measurement code (bench.run_scenario) is mock-verified against a known-exact server in test_harness.py before ever touching a real engine. All four primary configs execute on CPU ONLY, matching stoke Phase 1's target silicon and the premise gate's CPU-bandwidth ceiling: the vendored llama-server build has no GPU backends (--list-devices: empty), and ollama is pinned via a num_gpu=0 model registration of the same GGUF -- its GPU discovery ignores CUDA_VISIBLE_DEVICES on this box, and unpinned it silently offloads the whole model to the RTX 3050 (53 tok/s "single-stream", double the CPU physics ceiling -- how the pin's necessity was caught). Ollama's unpinned GPU default is reported separately in the appendix.

## Environment

| field | value |
| --- | --- |
| CPU | 13th Gen Intel(R) Core(TM) i5-13450HX |
| RAM | 15.6 GB |
| Platform | Windows-11-10.0.26200-SP0 |
| Power scheme | Power Scheme GUID: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  (High performance) |
| Ollama version | Warning: client version is 0.30.11 |
| llama.cpp version | version: 10037 (56d6e9dde) |
| Model file | Qwen3-4B-Instruct-2507-Q4_K_M.gguf (2,497,280,736 bytes (2.50 GB)) |
| Git rev | ee95db78f850253223cb54f9f5bf8e55d035cca7 |

## Tuned-config policy

Let P(scenario): `s2-closed-N` → N; `s1-interactive` and `s3-agent-fanout` → 8;
`s4-longctx` → 1.

| config            | construction                                                                     |
|-------------------|-----------------------------------------------------------------------------------|
| ollama-default    | `adapters.OllamaAdapter()`                                                        |
| ollama-tuned      | `adapters.OllamaAdapter(num_parallel=P, context_length=(8192 if s4 else 2048))`   |
| llamacpp-default  | `adapters.LlamaCppAdapter()`                                                      |
| llamacpp-tuned    | `adapters.LlamaCppAdapter(parallel=P, ctx=(8192 if s4 else P*2048))`              |

Rationale: "tuned" = what a competent operator sets for the known workload -- slots matched to
concurrency, per-slot context matched to prompt+output size (2048 covers every scenario except
S4's 4096+128, which gets 8192 and P=1). Per-slot ctx also kept equal across the two tuned engines
for fairness. A fresh adapter instance is constructed per cell; the engine restarts per cell so no
cache/KV state leaks between cells.


## s1-interactive

| config | aggregate tok/s | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | ITL p95 ms | completed | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ollama-default | 5.7 | 90236.5 | 183969.7 | 91.8 | 97.1 | 8/8 |  |
| ollama-tuned | 12.5 | 40405.0 | 51550.5 | 225.3 | 276.1 | 8/8 |  |
| llamacpp-default | 11.1 | 41385.4 | 76322.7 | 122.9 | 165.9 | 8/8 |  |
| llamacpp-tuned | 13.2 | 33026.7 | 51258.9 | 202.8 | 220.0 | 8/8 |  |

run-to-run aggregates:
- ollama-default: 5.6, 5.7, 5.7
- ollama-tuned: 12.5, 12.7, 12.5
- llamacpp-default: 11.0, 11.1, 11.8
- llamacpp-tuned: 13.2, 13.2, 13.3

## s2-closed-1

| config | aggregate tok/s | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | ITL p95 ms | completed | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ollama-default | 11.0 | 12540.4 | 12540.4 | 91.1 | 94.9 | 1/1 | +1 |
| ollama-tuned | 15.1 | 7323.1 | 7323.1 | 66.2 | 68.6 | 1/1 |  |
| llamacpp-default | 14.9 | 8235.1 | 8235.1 | 66.8 | 72.2 | 1/1 |  |
| llamacpp-tuned | 15.6 | 7439.4 | 7439.4 | 63.8 | 65.9 | 1/1 |  |

run-to-run aggregates:
- ollama-default: 11.1, 11.0, 10.9
- ollama-tuned: 15.1, 15.1, 15.2
- llamacpp-default: 15.9, 14.9, 14.8
- llamacpp-tuned: 15.6, 15.2, 15.6

## s2-closed-2

| config | aggregate tok/s | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | ITL p95 ms | completed | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ollama-default | 7.2 | 24609.1 | 56616.0 | 92.7 | 97.5 | 2/2 |  |
| ollama-tuned | 22.5 | 14340.5 | 14923.6 | 85.3 | 89.7 | 2/2 |  |
| llamacpp-default | 23.3 | 15135.7 | 15138.7 | 85.9 | 89.1 | 2/2 |  |
| llamacpp-tuned | 23.5 | 14641.6 | 14643.2 | 84.4 | 90.2 | 2/2 |  |

run-to-run aggregates:
- ollama-default: 7.2, 7.1, 7.2
- ollama-tuned: 22.6, 22.5, 22.3
- llamacpp-default: 22.6, 23.3, 23.3
- llamacpp-tuned: 23.6, 23.5, 23.5

## s2-closed-4

| config | aggregate tok/s | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | ITL p95 ms | completed | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ollama-default | 6.1 | 48480.3 | 102421.4 | 92.6 | 97.2 | 4/4 |  |
| ollama-tuned | 17.3 | 28114.5 | 29540.2 | 111.8 | 131.1 | 4/4 |  |
| llamacpp-default | 30.2 | 31119.8 | 33224.7 | 123.0 | 129.0 | 4/4 |  |
| llamacpp-tuned | 33.6 | 29485.7 | 30934.8 | 111.8 | 126.0 | 4/4 |  |

run-to-run aggregates:
- ollama-default: 6.1, 6.1, 6.1
- ollama-tuned: 17.2, 17.3, 17.3
- llamacpp-default: 29.8, 30.2, 32.1
- llamacpp-tuned: 33.9, 33.6, 33.4

## s2-closed-8

| config | aggregate tok/s | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | ITL p95 ms | completed | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ollama-default | 5.4 | 102593.0 | 203126.0 | 95.3 | 105.3 | 8/8 |  |
| ollama-tuned | 14.3 | 42721.6 | 59268.0 | 214.6 | 229.7 | 8/8 | +1 |
| llamacpp-default | 16.2 | 49624.7 | 81018.4 | 116.7 | 141.9 | 8/8 |  |
| llamacpp-tuned | 17.7 | 58546.4 | 60839.6 | 211.8 | 240.8 | 8/8 |  |

run-to-run aggregates:
- ollama-default: 5.4, 5.5, 5.4
- ollama-tuned: 14.4, 14.3, 14.2
- llamacpp-default: 16.2, 15.5, 17.0
- llamacpp-tuned: 17.4, 17.7, 17.8

## s2-closed-16

| config | aggregate tok/s | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | ITL p95 ms | completed | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ollama-default | 8.4 | 124499.6 | 243558.5 | 66.3 | 69.1 | 16/16 | +1 |
| ollama-tuned | 13.3 | 71374.5 | 117510.6 | 412.0 | 463.2 | 16/16 |  |
| llamacpp-default | 13.5 | 93861.3 | 166383.1 | 112.9 | 120.7 | 16/16 |  |
| llamacpp-tuned | 14.2 | 89684.3 | 123095.7 | 403.5 | 512.7 | 16/16 |  |

run-to-run aggregates:
- ollama-default: 8.4, 8.4, 8.4
- ollama-tuned: 13.3, 13.3, 13.3
- llamacpp-default: 13.5, 13.6, 12.8
- llamacpp-tuned: 14.2, 14.2, 13.9

## s3-agent-fanout

| config | aggregate tok/s | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | ITL p95 ms | completed | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ollama-default | 10.6 | 125249.6 | 238914.9 | 83.1 | 96.5 | 8/8 | +5 |
| ollama-tuned | 11.5 | 106186.2 | 159167.6 | 329.6 | 387.5 | 8/8 |  |
| llamacpp-default | 13.3 | 108944.7 | 154271.7 | 187.0 | 246.4 | 8/8 | +2 |
| llamacpp-tuned | 11.7 | 107956.2 | 149747.0 | 293.3 | 323.8 | 8/8 | +1 |

run-to-run aggregates:
- ollama-default: 10.9, 10.6, 10.1
- ollama-tuned: 11.5, 11.5, 11.7
- llamacpp-default: 13.4, 13.3, 13.1
- llamacpp-tuned: 11.8, 11.7, 11.7

## s4-longctx

| config | aggregate tok/s | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | ITL p95 ms | completed | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ollama-default | FAILED | - | - | - | - | - | no successful run (all measured runs failed) |
| ollama-tuned | 6.4 | 73301.5 | 73301.5 | 152.7 | 183.8 | 1/1 |  |
| llamacpp-default | 7.8 | 77384.1 | 77384.1 | 127.0 | 131.7 | 1/1 |  |
| llamacpp-tuned | 7.8 | 74996.4 | 74996.4 | 127.7 | 130.7 | 1/1 |  |

run-to-run aggregates:
- ollama-default: 0.0, 0.0, 0.0
- ollama-tuned: 6.4, 6.9, 5.9
- llamacpp-default: 7.8, 7.9, 7.8
- llamacpp-tuned: 7.9, 7.8, 7.8

## S2 concurrency sweep — aggregate decode tok/s vs N

| N | ollama-default | ollama-tuned | llamacpp-default | llamacpp-tuned |
| --- | --- | --- | --- | --- |
| 1 | 11.0 | 15.1 | 14.9 | 15.6 |
| 2 | 7.2 | 22.5 | 23.3 | 23.5 |
| 4 | 6.1 | 17.3 | 30.2 | 33.6 |
| 8 | 5.4 | 14.3 | 16.2 | 17.7 |
| 16 | 8.4 | 13.3 | 13.5 | 14.2 |

Scaling, normalized to each config's N=1 (×, 2dp) -- the scaling curve is the wedge evidence:

| N | ollama-default | ollama-tuned | llamacpp-default | llamacpp-tuned |
| --- | --- | --- | --- | --- |
| 1 | 1.00× | 1.00× | 1.00× | 1.00× |
| 2 | 0.65× | 1.49× | 1.56× | 1.51× |
| 4 | 0.55× | 1.14× | 2.03× | 2.16× |
| 8 | 0.49× | 0.95× | 1.09× | 1.13× |
| 16 | 0.77× | 0.88× | 0.91× | 0.91× |

## Premise gate (kill-switch)

Spec anchor: physics ceiling ≈ **160 tok/s aggregate** (spec `docs/superpowers/specs/2026-07-16-phase1-concurrent-cpu-core-design.md`, "Physics budget": batch-8 step ≈ 2.4GB weights + 8×74KB/tok×~1k KV ≈ 3.0GB @ ~60GB/s real → ~20 steps/s → ~160 tok/s; threshold 70% = 112 tok/s).

| scenario | config | median aggregate tok/s | % of 160 | verdict |
| --- | --- | --- | --- | --- |
| s2-closed-8 | ollama-default | 5.4 | 3.4% | below |
| s2-closed-8 | ollama-tuned | 14.3 | 9.0% | below |
| s2-closed-8 | llamacpp-default | 16.2 | 10.1% | below |
| s2-closed-8 | llamacpp-tuned | 17.7 | 11.0% | below |
| s2-closed-16 | ollama-default | 8.4 | 5.2% | below |
| s2-closed-16 | ollama-tuned | 13.3 | 8.3% | below |
| s2-closed-16 | llamacpp-default | 13.5 | 8.5% | below |
| s2-closed-16 | llamacpp-tuned | 14.2 | 8.9% | below |

**PREMISE GATE: CONFIRMED — all baselines below 70% of ceiling**

## Failures & anomalies

### ollama-default--s4-longctx (no successful run)

0/3 measured runs succeeded. Per-run errors: [['client 0: http://127.0.0.1:11434/v1/chat/completions returned HTTP 400'], ['client 0: http://127.0.0.1:11434/v1/chat/completions returned HTTP 400'], ['client 0: http://127.0.0.1:11434/v1/chat/completions returned HTTP 400']]

```
srv    operator(): Chat format: peg-native
slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 4304625
srv  get_availabl: updating prompt cache
srv          load:  - looking for better prompt, base f_keep = -1.000, sim = 0.000
srv        update:  - cache state: 0 prompts, 0.000 MiB (limits: 8192.000 MiB, 4096 tokens, 8589934592 est)
srv  get_availabl: prompt cache update took 0.01 ms
slot launch_slot_: id  0 | task -1 | sampler chain: logits -> penalties -> ?dry -> ?top-n-sigma -> top-k -> ?typical -> ?top-p -> ?min-p -> ?xtc -> temp-ext -> dist 
slot launch_slot_: id  0 | task -1 | sampler params: 
	repeat_last_n = 64, repeat_penalty = 1.100, frequency_penalty = 0.000, presence_penalty = 0.000
	dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 4096
	top_k = 40, top_p = 1.000, min_p = 0.000, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 0.000
	mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000, adaptive_target = -1.000, adaptive_decay = 0.900
slot launch_slot_: id  0 | task 6 | processing task, is_child = 0
slot   operator(): id  0 | task 6 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 4112
srv    send_error: task id = 6, error: request (4112 tokens) exceeds the available context size (4096 tokens), try increasing it
slot      release: id  0 | task 6 | stop processing: n_tokens = 0, truncated = 0
srv          stop: cancel task, id_task = 6
srv  update_slots: all slots are idle
srv  update_slots: all slots are idle
[GIN] 2026/07/17 - 10:49:55 | 400 |    293.4052ms |       127.0.0.1 | POST     "/v1/chat/completions"
srv    operator(): Chat format: peg-native
slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 4611564
srv  get_availabl: updating prompt cache
srv          load:  - looking for better prompt, base f_keep = -1.000, sim = 0.000
srv        update:  - cache state: 0 prompts, 0.000 MiB (limits: 8192.000 MiB, 4096 tokens, 8589934592 est)
srv  get_availabl: prompt cache update took 0.01 ms
slot launch_slot_: id  0 | task -1 | sampler chain: logits -> penalties -> ?dry -> ?top-n-sigma -> top-k -> ?typical -> ?top-p -> ?min-p -> ?xtc -> temp-ext -> dist 
slot launch_slot_: id  0 | task -1 | sampler params: 
	repeat_last_n = 64, repeat_penalty = 1.100, frequency_penalty = 0.000, presence_penalty = 0.000
	dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 4096
	top_k = 40, top_p = 1.000, min_p = 0.000, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 0.000
	mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000, adaptive_target = -1.000, adaptive_decay = 0.900
slot launch_slot_: id  0 | task 9 | processing task, is_child = 0
slot   operator(): id  0 | task 9 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 4112
srv    send_error: task id = 9, error: request (4112 tokens) exceeds the available context size (4096 tokens), try increasing it
slot      release: id  0 | task 9 | stop processing: n_tokens = 0, truncated = 0
srv  update_slots: all slots are idle
srv          stop: cancel task, id_task = 9
srv  update_slots: all slots are idle
[GIN] 2026/07/17 - 10:49:55 | 400 |    391.1585ms |       127.0.0.1 | POST     "/v1/chat/completions"
```


## Appendix — ollama GPU default (unpinned), sidebar

Unpinned `ollama serve` on this box offloads the whole model to the RTX 3050 Laptop 6GB. These cells were measured with the identical protocol before the CPU pin existed and are kept as honest context for what ollama's out-of-the-box default delivers here. They are NOT comparable to the CPU matrix above (different silicon, ~2x the memory bandwidth) and play no part in the premise gate.

| scenario | aggregate tok/s | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | ITL p95 ms | completed |
| --- | --- | --- | --- | --- | --- | --- |
| s1-interactive | 32.9 | 10621.6 | 14239.5 | 27.3 | 31.5 | 8/8 |
| s2-closed-1 | 53.0 | 424.6 | 424.6 | 18.8 | 19.2 | 1/1 |
| s2-closed-2 | 50.0 | 1772.8 | 5398.0 | 18.8 | 19.2 | 2/2 |
| s2-closed-4 | 48.6 | 4470.8 | 10514.6 | 18.8 | 19.2 | 4/4 |
| s2-closed-8 | 48.0 | 9862.6 | 20733.3 | 18.8 | 19.1 | 8/8 |
| s2-closed-16 | 47.7 | 20750.9 | 41272.0 | 18.8 | 19.1 | 16/16 |
| s3-agent-fanout | 50.2 | 21775.0 | 45928.6 | 19.5 | 19.9 | 8/8 |
| s4-longctx | FAILED |  |  |  |  |  |