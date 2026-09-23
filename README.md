# stoke

An embeddable inference engine for local LLMs on CPU, written in C11 with no
dependencies. It runs many concurrent requests against one model using paged KV
and continuous batching, and it ships an OpenAI-compatible HTTP server built on
its own frozen C ABI.

```c
stoke_engine* e = stoke_open("model.gguf", NULL);

stoke_request req = { .size = sizeof req, .id = 1, .prompt_ids = ids,
                      .n_prompt = n, .max_tokens = 128, .eos_id = stoke_eos_id(e),
                      .temperature = 0.8f, .seed = 42 };
stoke_submit(e, &req);

while (/* not finished */) {
    stoke_step(e, NULL);                      /* you own the loop */
    int32_t n = stoke_poll(e, 1, out, cap, &finish);
}
```

## Status, honestly

**This is not released software.** It is a working engine with a frozen ABI, a
large test suite, and a set of measured limitations that are listed here rather
than discovered later.

| | |
|---|---|
| Models | **Qwen3 and Llama** (`general.architecture` = `qwen3` or `llama`; anything else is refused at load). Parity against llama.cpp is gated on Qwen3-4B and Llama-3.2-3B, including a 2,900-token context and an unsloth-quantized file. |
| Weights | F32, Q8_0, Q4_K, Q6_K (Q4_K_M files work, including their Q6_K `ffn_down`) |
| Platforms | Linux and Windows, x86-64 with AVX2 (AVX-VNNI is detected at runtime and used when present). **macOS and ARM are untested** — the port surface is small (`simd.c` and two `__x86_64__` guards; threading is pthreads; every SIMD kernel has a tested scalar twin) and is Phase 1.5's first item. |
| Throughput | **Finishes the 8/16/32-client benchmark in 0.85 / 0.83 / 0.78× the wall of llama.cpp's server** (same GGUF, same harness, same hour), first token 4.5–5× sooner, 0.71–0.75× the CPU time; **1.74× / 1.84× ollama-default** @8/@16 (the target was 2×; missed, and said so below). `docs/BENCHMARKS.md` has the method and the raw numbers. |
| C ABI | `src/stoke.h`, **frozen at v0.** Error codes are never renumbered, struct fields are only appended, nothing is removed. |

## Build

No dependencies beyond a C11 compiler and pthreads.

```sh
make                      # build everything
make test                 # the full gate suite
make SAN=asan test        # under AddressSanitizer
make SAN=tsan test        # under ThreadSanitizer
```

On Windows use MSYS2/mingw-w64 and run `mingw32-make` **from a POSIX shell** —
the Makefile uses `mkdir -p` and `sh`.

## Use

### As a server

```sh
stoke serve model.gguf --port 8080 --ctx 4096 --seqs 8 --threads 8
```

Speaks `/health`, `/v1/models`, `/v1/completions` and `/v1/chat/completions`,
with SSE streaming, `stop` strings, and `stream_options.include_usage`. The
`openai` Python client works against it unmodified.

⚠ **It refuses what it does not implement rather than answering it wrongly.**
`n != 1` is a 400, and so are `frequency_penalty` / `presence_penalty` — those
are OpenAI's *additive* penalties, and the one this engine has is llama.cpp's
*multiplicative* `repeat_penalty`. Quietly substituting it would return a
distribution nobody asked for, in a way that reads as the model being odd.

### As a CLI

```sh
stoke run model.gguf -p "Write one sentence about the sea." -n 64 \
    --temp 1.2 --top-k 40 --seed 42
stoke info model.gguf          # the GGUF's metadata and tensor table
stoke tokenize model.gguf prompt.txt   # ids for a text FILE
stoke batch model.gguf -f a.txt -f b.txt --seqs 8   # static batch
stoke bench --serve model.gguf -f a.txt -f b.txt --seqs 8   # concurrent bench
```

`stoke run` and the server use the same sampler and the same `(seed, position)`
keying, so **the CLI reproduces the server's output byte for byte for the same
prompt, parameters and seed.** That is checked by `tests/run_sampling.sh`, not
assumed — it was false when first written.

### As a library

`src/stoke.h` is the whole API and the whole documentation. The engine owns no
thread: you call `stoke_step()` in your own loop, and `stoke_submit`,
`stoke_poll` and `stoke_cancel` are safe from any thread while you do.

⚠ One sharp edge, stated in capitals in the header too: **`prompt_ids` is
borrowed, not copied**, and must outlive the request. A preempted request
re-prefills from its prompt, so copying would double exactly the memory the KV
pool is rationing.

## Benchmarks — and how to read one

**Do not trust a bare tok/s from this or any other engine, including anything
here.** On the development box the same binary, workload and harness read
**1.8× apart on different days with no code change** (4.7–6.4 tok/s on
2026-08-20, 8.6–9.0 on 08-28, 7.3 on 08-30) and ~4% apart between two sittings
an hour apart. Inside one quiet sitting the spread is under 1%. So every
number published here is a **ratio against a referent measured in the same
hour, with the absolutes and the block beside it** — that is the reproducible
quantity. The full method, the raw records and the commands that reproduce
each row are in **`docs/BENCHMARKS.md`**.

Workload: N closed-loop clients, each one chat request of ~525 prompt tokens
and 128 streamed answer tokens, no shared prefix, clock from first request to
last token, same GGUF (Qwen3-4B Q4_K_M) on every arm, CPU only, all over
HTTP+SSE through one client harness.

| clients | referent, configured for the workload | wall **referent / stoke** | pairs > 1 | first token, stoke sooner by (min / median) | CPU-s stoke / referent |
|---|---|---|---|---|---|
| 8 | llama-server b10037 `--parallel 8 -c 16384` | **1.18×** | 5/5 | 4.7× / 1.9× | 0.75× |
| 16 | llama-server `--parallel 16 -c 32768` | **1.20×** | 3/3 | 4.5× / 1.5× | 0.73× |
| 32 | llama-server `--parallel 32 -c 22528` | **1.28×** | 3/3 | 5.1× / 1.4× | 0.71× |
| 8 | ollama 0.30.11, defaults, CPU | **1.74×** | 2/2 | 1.3× / 2.3× | — |
| 16 | ollama, defaults, CPU | **1.84×** | 2/2 | 1.3× / 2.4× | — |

The margin over llama-server **grows with concurrency**, which is the design's
whole claim: a new prompt is prefilled in 256-token chunks while every request
already admitted keeps decoding in the same step, so the more requests are in
flight, the more of the prefill phase is shared. The kernels are within a few
percent of llama.cpp's on the same instruction mix — measured, not assumed —
so the ratio is the scheduler's.

**What it does not win, in the same table's terms.** The Phase-1 target
against ollama was 2× and it reads 1.74× / 1.84×: on a CPU-only box ollama
serves one request at a time, so "2× ollama" is "2× a serial server", and what
stops stoke at 1.8× is its own prefill cost. And a request arriving while
others are decoding stalls their streams for one prefill chunk — ~3.3 s
median, up to 3.8 s at 32 clients — against a design bound of 250 ms: missed
by an order of magnitude at every chunk width, fixable only by a scheduler
change that is deferred to Phase 1.5. For scale, llama-server's streams stop
for 29–30 s under the same traffic (one whole prompt batch), and ollama's
serial server has no co-running to stall.

Single-stream decode is ~0.9× llama.cpp's (M4.5). The engine is built for
many concurrent requests on hardware without a GPU, not for one stream.

Method, in one line: `tests/tooling/run_e2e_interleaved.py` runs the two arms
alternating with the order flipping every pair, one server lifetime per unit,
a CPU-quiesce gate before each unit, every unit banked as it lands and refused
if its own aggregate cannot be reconstructed from its raw records; a
`llama-bench pp512` referent brackets each block and names the box state.

## Testing

`make test` runs 19 unit binaries plus the gate scripts: byte-exact tokenizer
and GGUF goldens (Qwen3 and Llama-3, an unsloth file, a 22,848-line unicode
sweep), a 25-request byte-exact HTTP transcript reproduced across
Windows/gcc-15 and Linux/gcc-13 (and again under ASan), the `openai` Python
client suite, a randomized 10-minute sanitized serve fuzz with a mutant arm
that proves the fuzz can fail, an ABI checklist with its own mutation leg and
a C++ translation-unit leg, and a scheduler race test that drives the real
scheduler over a fake model from two threads (TSan found six races where
reading the code had found one). Greedy-parity probes against llama.cpp's own
logits run to 2,900 tokens on both models with every divergence classified.

**Determinism is per build:** the same binary, weights, seed and parameters
give the same tokens, and that is gated (`serve_golden`, `run_sampling`).
Two builds of the same source with different compilers can differ in a
greedy token after a few hundred positions — a sub-ulp difference in a
transcendental, propagated through 36 layers — and so can any engine's; the
tokenizer and everything discrete is byte-exact everywhere.

Several of those gates exist because they caught something: a server that died
on the first client disconnect on Linux, an idle reaper that discarded finished
responses, a `stoke_poll` that answered an unknown id with silence, and a
sampler whose first token was greedy for requests submitted mid-step. The
newest one: a 32-client burst found a listen backlog of 16 behind a server
that accepts one connection per turn — 13 clients refused — and the smoke
written to prove the fix had to be made to fail twice before it could see it.

**Not yet run:** the 24 h sanitized serving soak (one server, a day of
randomized traffic under ASan+LSan, and a memory rule fixed before the run:
over the last 6 h, mapped and resident memory outside the KV cache must each
grow by at most one KV block). Its rig is shaken down; the day of box time is
the last gate before this is tagged.

## What this is not

- Not a training or fine-tuning tool.
- Not a GPU engine. CPU only, by design, for now.
- Not multi-model: one process serves one model.
- Not tuned for single-stream latency — the design target is many concurrent
  requests on hardware that has no GPU.

## License

Not yet chosen.
