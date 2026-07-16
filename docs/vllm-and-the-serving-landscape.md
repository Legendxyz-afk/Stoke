# vLLM and the Inference-Serving Landscape — deep background

*2026-07-14. Written to understand the incumbent we're building next to. To beat
someone in a niche you must understand them cold. This explains what LLM
inference serving actually is, what vLLM invented, what it's great at, and —
precisely — where the gap we're targeting lives. Read alongside `../MEMORY.md`.*

---

## 0. TL;DR

**vLLM is the open-source, high-throughput LLM inference & serving engine** out
of UC Berkeley's Sky Computing Lab (2023). It won by solving the *memory* problem
in LLM serving with two ideas: **PagedAttention** (treat the KV cache like OS
virtual memory) and **continuous batching** (schedule work one token-step at a
time so the GPU never idles). Together they gave up to a **24× throughput**
jump over naive serving on the same hardware.

It is the throughput king — but it is a **heavyweight, Python + PyTorch,
datacenter-GPU-shaped** system. Even though its V1 rewrite added CPU / AMD /
Intel / TPU backends, it is not lightweight, not embeddable, and not tuned for
one-box consumer/edge serving. **That** is our opening — not "vLLM can't do CPU"
(it can), but "no one brings vLLM's concurrency tricks to the lightweight,
embeddable, runs-on-your-hardware world that llama.cpp/Ollama own."

---

## 1. What "LLM inference serving" actually is

An LLM generates text **autoregressively** — one token at a time, each new token
conditioned on everything before it. A serving engine's job is to run many such
generations for many users at once, fast. Two phases per request:

- **Prefill** — the model ingests your whole prompt in one parallel pass. This is
  **compute-bound** (one big matrix-multiply over all prompt tokens). It produces
  the first output token and populates the KV cache (below). Prefill time ≈
  **time-to-first-token (TTFT)**.
- **Decode** — the model then emits output tokens one by one. Each step reads the
  *entire* model's weights plus the growing KV cache to produce a single token, so
  decode is **memory-bandwidth-bound** (you're streaming gigabytes per token, not
  doing much math). Per-token decode time ≈ **inter-token latency (ITL)**, a.k.a.
  time-per-output-token (TPOT).

Understanding "prefill = compute-bound, decode = bandwidth-bound" explains
almost every design decision in this whole field, including ours.

### The KV cache — the thing everything revolves around

Attention at token *t* needs the **K**ey and **V**alue vectors of every earlier
token. Recomputing them each step would be O(n²) wasted work, so the engine
**caches K and V for every token** — the *KV cache*. Problem: it is **large**
(gigabytes for long contexts × many users) and **dynamic** — you don't know how
long a response will be, so you don't know how much cache a request will need.

### Why memory ≈ throughput

Decode reads the model weights once per step and **amortizes that read across
every sequence in the batch**. So the more sequences you can hold in memory at
once, the bigger your batch, the more tokens/sec you get for the same weight
reads. **KV cache is what fills memory → memory efficiency directly sets your max
batch size → batch size sets throughput.** This single chain is why a "memory
management" trick (PagedAttention) produced a *throughput* revolution.

### The two metrics, always in tension

- **Latency** — TTFT and ITL, what a single user feels.
- **Throughput** — total tokens/sec across all concurrent users, what an operator
  pays for.

Bigger batches raise throughput but can hurt latency; the whole art of a serving
engine is managing this trade-off well. Our wedge lives specifically at
**throughput under concurrency on hardware vLLM ignores.**

---

## 2. The old way, and why it wasted 60–80% of memory

Naive/"static" batching pre-allocates a **contiguous, max-length** KV buffer per
request and runs the batch until the *longest* sequence finishes. Two failures:

- **Fragmentation.** *Internal*: a request reserved 2048 tokens of cache but used
  400 — the rest is reserved-but-idle. *External*: variable-sized buffers leave
  unusable gaps between them. Measured waste: **60–80% of KV memory sat idle.**
  Idle memory = smaller batches = low throughput.
- **Convoy stalls.** The batch can't release until the slowest sequence ends, so
  as short requests finish their GPU slots sit empty. The GPU is underused.

vLLM attacked both — memory with PagedAttention, scheduling with continuous
batching.

---

## 3. Innovation #1 — PagedAttention (the memory fix)

**The analogy:** an operating system doesn't give a process one giant contiguous
block of RAM; it hands out fixed-size **pages** and keeps a **page table** mapping
logical → physical addresses. PagedAttention does exactly this to the KV cache.

**Mechanics:**
- The KV cache is cut into fixed-size **blocks** (default **16 tokens** each,
  `--block-size`).
- Each sequence has a **block table** mapping its logical blocks → arbitrary,
  **non-contiguous** physical blocks anywhere in the GPU pool.
- Attention is computed **blockwise**: to attend at position *t*, gather the
  blocks holding positions [1..t], wherever they physically live.

**What it buys:**
- External fragmentation → **gone**; internal fragmentation → only the last,
  partially-filled block of each sequence. Waste drops from 60–80% to **under 4%**.
- **Sharing via copy-on-write.** Multiple sequences can point at the *same*
  physical blocks until one diverges — this makes parallel sampling, beam search,
  and **shared-prefix reuse** (system prompts, few-shot examples) nearly free.
- More free memory → bigger batches → **up to 24× throughput** vs naive
  (HuggingFace Transformers) on the same GPU.

Source: Kwon et al., *"Efficient Memory Management for Large Language Model
Serving with PagedAttention,"* SOSP 2023 (UC Berkeley Sky Computing Lab).

---

## 4. Innovation #2 — Continuous batching (the scheduling fix)

Also called **iteration-level scheduling** or **in-flight batching**; lineage is
the *Orca* paper (OSDI 2022).

Instead of scheduling a fixed batch and waiting for it to finish, the scheduler
runs at the granularity of **a single forward pass (one token step)**. At **every
step** it can:

- **admit** newly arrived requests into the running batch,
- **preempt/evict** running ones if memory is tight (swapping their KV blocks out
  to CPU RAM, to be resumed later),
- **resume** previously waiting/evicted requests.

The instant any sequence emits its stop token, its slot is refilled the same step.
The GPU therefore stays **saturated** — no convoy stall. This is the direct cause
of the headline gap you keep seeing: **~793 tok/s (vLLM) vs ~41 tok/s (Ollama)**
at peak concurrency — a 19× difference that "lives entirely at peak concurrency."

PagedAttention makes continuous batching *practical*: you can only keep swapping
sequences in and out cheaply if their memory is paged rather than contiguous.

---

## 5. The rest of vLLM's toolbox (so you know the full surface)

- **Chunked prefill** (default in V1): split a long prefill into chunks and
  interleave them with ongoing decode, so one user's huge prompt doesn't block
  everyone else's token stream ("pipeline bubbles").
- **Automatic prefix caching (APC):** detect shared prompt prefixes across
  requests and reuse their cached KV — big win for repeated system prompts.
- **Tensor / pipeline parallelism:** shard one model across multiple GPUs.
- **Quantization:** GPTQ, AWQ, FP8, etc. — smaller weights leave more room for KV
  cache / bigger batches; can quantize at server startup.
- **Speculative decoding:** a cheap **draft** predicts several tokens ahead; the
  real model **verifies** them in one pass, cutting latency (methods: n-gram/
  suffix = modest; EAGLE / MTP / draft-model = best).
- **The V1 re-architecture (2025):** an **async scheduler** that separates
  CPU-side work (tokenization, detokenization, image preprocessing) from GPU
  inference so they overlap instead of blocking; cleaner engine core; and
  **multi-backend support — NVIDIA (CUDA), AMD (ROCm), Intel (XPU), Google TPU
  (XLA), and CPU.**
- **An OpenAI-compatible HTTP server**, which is why so many tools point at it by
  just swapping `base_url`.

---

## 6. What vLLM is genuinely great at — and what it is *not* built for

**Great at:**
- **Maximum throughput at high concurrency** on datacenter GPUs. It is the
  reference implementation of paged KV + continuous batching.
- **Production serving at scale**: multi-GPU, big models, many simultaneous users,
  an ops-grade feature set.

**Not built for (this is the honest gap, post-V1):**
- It is a **heavyweight Python + PyTorch** stack with a large dependency
  footprint. It is **not embeddable** into a native/desktop app, not a single
  small binary, not a `.so`/`.dll` you drop into a game or a Tauri/Electron app.
- Its **design center is datacenter GPUs.** The CPU/ROCm/XPU/TPU backends are
  real but are *ports around a server-shaped core* — not a system engineered from
  scratch for a **single consumer box serving a handful of users**.
- It is **complex to deploy and tune**, and **VRAM/RAM-hungry** — overkill for one
  machine, a small team, or an app that just wants to embed inference.
- Its user is a **server operator**, not a **desktop/edge/embedded developer**.

**And the other side of the market:**
- **llama.cpp / Ollama** own "**lightweight, runs anywhere, embeddable**" — a
  small C/C++ core, a single binary, CPU/Metal/Vulkan/CUDA, trivial to ship. But
  they are **single-user-shaped**: Ollama ~41 tok/s under concurrent load;
  llama.cpp has "**no management layer**," every knob manual.

---

## 7. The wedge, stated in technical terms

The gap is **not** "vLLM can't run on a CPU" — V1 can. The gap is:

> **No one has brought vLLM's concurrency techniques — paged KV cache +
> continuous batching — to the lightweight, embeddable, runs-on-your-own-hardware
> world that llama.cpp and Ollama own.**

So the engine we're building is, in one line:

> **llama.cpp's footprint and portability × vLLM's continuous-batching scheduler
> and paged KV cache — engineered for bandwidth-bound consumer/edge hardware.**

Small. Embeddable. Multi-user. OpenAI-compatible. It sits exactly in the empty
middle: too concurrent for Ollama, too light and consumer-shaped for vLLM.

**The honest physics caveat (repeat from MEMORY.md):** on consumer hardware,
decode is memory-bandwidth-bound, so the realistic prize is **~2–4× over Ollama
at concurrency**, not vLLM's 19× (which assumes fat datacenter GPUs and
bandwidth). Extracting that 2–4× from a bandwidth wall *is* the low-level craft
this whole project is about — paged KV so you can batch at all, a scheduler tuned
for few-but-real concurrency, quantization, and cache-aware kernels.

---

## 8. Glossary (fast reference)

- **Autoregressive** — generate one token at a time, each conditioned on all prior.
- **Prefill** — parallel pass over the prompt; compute-bound; sets **TTFT**.
- **Decode** — one-token-at-a-time generation; memory-bandwidth-bound; sets **ITL/TPOT**.
- **KV cache** — stored Key/Value vectors of past tokens so attention isn't
  recomputed; large and dynamically-sized; the thing that fills memory.
- **TTFT / ITL (TPOT)** — time-to-first-token / inter-token latency (time per
  output token).
- **Throughput vs latency** — total tokens/sec across users vs what one user feels.
- **Static / dynamic / continuous batching** — fixed batch to completion / regroup
  between requests / regroup **every token step**.
- **PagedAttention** — OS-paging applied to the KV cache: fixed blocks + a block
  table; kills fragmentation, enables sharing.
- **Block table** — per-sequence map from logical KV blocks → physical blocks.
- **Prefix caching (APC)** — reuse KV for shared prompt prefixes across requests.
- **Chunked prefill** — split long prefills to interleave with decode.
- **Tensor/pipeline parallelism** — shard one model across GPUs.
- **Quantization** — store weights in fewer bits (INT4/INT8/FP8) to save memory/bandwidth.
- **Speculative decoding** — cheap draft proposes tokens; real model verifies in
  one pass; lowers latency.
- **Memory-bandwidth-bound vs compute-bound** — limited by how fast you can move
  bytes vs how fast you can do math. Decode is the former; this is *the* constraint
  on consumer hardware.

---

## Sources

- Kwon et al., PagedAttention / vLLM — SOSP 2023 (UC Berkeley Sky Computing Lab).
- RunPod, *"vLLM Explained: PagedAttention and Continuous Batching."*
- Swfte AI, *"vLLM Continuous Batching Deep Dive."*
- codepointer, *"PagedAttention: Saving Millions on Wasted GPU Memory."*
- Medium (Serah Tapia), *"How PagedAttention Solved the Memory Crisis in LLM Serving."*
- vLLM V1 / DeepWiki engine-architecture; ROCm Blogs, *"vLLM V1 Meets AMD Instinct GPUs."*
- DigitalOcean, *"Speculative Decoding on vLLM."*
- Orca (Yu et al.), OSDI 2022 — iteration-level (continuous) batching.
- Earlier landscape sources in `../MEMORY.md` (Red Hat, BIZON, local-llm.net, etc.).
