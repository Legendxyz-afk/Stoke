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
| Models | **Qwen3 only.** `general.architecture` must be `qwen3`; anything else is refused at load. |
| Weights | F32, Q8_0, Q4_K, Q6_K (Q4_K_M files work, including their Q6_K `ffn_down`) |
| Platforms | Linux and Windows, x86-64 with AVX2 (AVX-VNNI is detected at runtime and used when present). **macOS is untested** — the server's `MSG_NOSIGNAL` needs a `SO_NOSIGPIPE` equivalent there. |
| Throughput | **~0.5× llama.cpp's server** on the 8-client concurrent benchmark. The cause is known and named below. |
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

**Do not trust a bare tok/s from this or any other engine, including the numbers
below.** On the development box the same binary, the same workload and the same
harness produced these three sittings:

| date | 8-client aggregate |
|---|---|
| 2026-08-20 | 4.66 – 6.38 tok/s |
| 2026-08-28 | 8.55 – 9.03 tok/s |
| 2026-08-30 | 7.25 – 7.28 tok/s |

**A 1.8× spread with no code change.** Within the quiet sitting the three blocks
span 0.4%; the variance is *between* days, not inside them. A published absolute
is a property of one afternoon.

So every measurement here carries a referent measured in the **same sitting** —
`llama-bench` on the same file and thread count — and the reproducible quantity
is the ratio. Between 08-28 and 08-30 the absolute fell 18.2% and the referent
fell 18.0%; normalised, the engine moved 0.3%.

**Where stoke actually stands:** its own M5 gate set a bar of **≥ 17.7 tok/s**
aggregate on the S2 workload (8 concurrent clients, 512-token prompts, 128
generated), derived from a reference engine measured with the same harness.
stoke closed that milestone at **0.51× the bar** — a documented miss, recorded
as one rather than quietly restated.

It is not a mystery, either: llama.cpp batches concurrent decode into a
matrix-**matrix** product while stoke still issues one matrix-**vector** product
per sequence, measured at **3.22× at every thread count**. Continuous batching,
paged KV and prefix sharing are all in and working — the arithmetic underneath
them is the gap, and closing it is a redesign, not a tuning pass.

Method, if you want to reproduce it: `tests/tooling/run_m5_bars.ps1` runs the
arms interleaved, counterbalanced, with a per-unit quiesce check, and banks each
unit keyed by the binary's sha256.

## Testing

`make test` runs 19 unit binaries plus the gate scripts: byte-exact tokenizer
and GGUF goldens, a 25-request byte-exact HTTP transcript reproduced across
Windows/gcc-15 and Linux/gcc-13 (and again under ASan), the `openai` Python
client suite, a randomized 10-minute sanitized serve soak, an ABI checklist with
its own mutation leg, and a scheduler race test that drives the real scheduler
over a fake model from two threads.

Several of those gates exist because they caught something: a server that died
on the first client disconnect on Linux, an idle reaper that discarded finished
responses, a `stoke_poll` that answered an unknown id with silence, and a
sampler whose first token was greedy for requests submitted mid-step.

## What this is not

- Not a training or fine-tuning tool.
- Not a GPU engine. CPU only, by design, for now.
- Not multi-model: one process serves one model.
- Not tuned for single-stream latency — the design target is many concurrent
  requests on hardware that has no GPU.

## License

Not yet chosen.
