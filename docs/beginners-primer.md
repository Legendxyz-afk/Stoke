# Beginner's Primer — LLM inference from absolute zero

*2026-07-14. Every concept behind the engine, built from the ground up, no prior
AI knowledge assumed. Programming analogies used throughout (OS paging,
memoization, allocators) because that's the fastest bridge. If a step is fuzzy,
that's the one to re-read. Companion to `vllm-and-the-serving-landscape.md`.*

---

## 1. What an "LLM" / "model" really is

A **model** is a giant math function with billions of adjustable numbers inside,
called **parameters** or **weights**. "A 7B model" = **7 billion** numbers.
"Llama / Mistral / Qwen" are just specific sets of these numbers that companies
trained and released. Picture a spreadsheet with 7 billion cells, or a machine
with 7 billion tiny knobs — the knob settings *are* the intelligence.

Two separate activities:

- **Training** — the one-time, enormously expensive process of *setting* the
  knobs by showing the model huge amounts of text. Someone else already did this;
  you **download the finished file of numbers.**
- **Inference** — actually *using* the finished model to produce answers.

> Training = writing the book. Inference = reading it aloud to people.
> **A serving engine only does inference.** We never train anything.

Those numbers are big: even compressed, a small model is ~**4 GB** you must store
and — crucially — **read over and over.**

## 2. Text becomes "tokens"

Computers don't handle words directly. Text is chopped into **tokens** — chunks
that are usually a word or part of a word ("cat" = 1 token; "unbelievable" ≈ "un"
+ "believ" + "able"). Rule of thumb: **1 token ≈ ¾ of a word.** Each token is just
an integer ID. Converting text ↔ tokens is **tokenization**. To the model, your
sentence is a list of numbers.

## 3. The model's only trick: predict the next token

Given the tokens so far, the model outputs a **probability for every possible next
token** — the world's most sophisticated autocomplete. For "The cat sat on the":
`mat 60%`, `floor 20%`, `roof 5%`, … (a probability over all ~100,000 tokens).
We then **pick one** (the top, or a weighted random draw — that randomness is
"temperature").

## 4. Autoregressive generation (the loop)

1. Predict one token → "mat"
2. **Append** it → "The cat sat on the mat"
3. Feed the whole thing back in → predict the next token → repeat.

**Autoregressive** = each new token depends on all tokens so far. A 500-token
answer = **500 trips through the model.** That's why chatbots look like they're
"typing" — they genuinely emit one token at a time and **cannot skip ahead**
(token 5 needs token 4 first). Generation is fundamentally **sequential**.

## 5. Attention: how the model looks back

To predict well, the model focuses on relevant earlier tokens (in "…the mat
because **it** was tired," resolving "it" means looking back at "cat"). That
mechanism is **attention**, the heart of the **Transformer** architecture every
LLM uses. The mechanical detail that matters later: each token produces three
things —

- **Query** — "what am I looking for right now?"
- **Key** — "here's a label describing me"
- **Value** — "here's my actual content"

The new token matches its **Query** against every previous token's **Key** to
decide relevance, then pulls in their **Values**. (A lookup: Query = search term,
Keys = labels, Values = the content behind each label.)

## 6. The KV cache — the most important concept

When generating token #400, you compare against the Keys/Values of tokens
#1–399 — but token #3's Key/Value **never change.** Recomputing them every step
is huge wasted work, so: compute each token's Key and Value **once and store
them.** That store is the **KV cache** — plain **memoization.**

Two properties drive the whole field:

- It **grows** by one Key + one Value **every token.** Long chats → big cache.
- It's **huge** (gigabytes across many users) and **dynamic** — you don't know how
  long an answer will be, so you can't know the cache size in advance.

> **The KV cache is the thing that fills your memory.** Everything else follows.

## 7. The two phases: prefill and decode

- **Prefill** — the model ingests your whole prompt (say 100 tokens) **in one
  parallel pass**, computing their Keys/Values (filling the cache) and producing
  the **first** output token. Lots of simultaneous math → **compute-heavy.** Its
  duration = **time-to-first-token (TTFT).**
- **Decode** — the one-at-a-time loop. Each step reads the **entire model** (4 GB)
  + the whole KV cache, does a *little* math, emits **one** token. The bottleneck
  is **moving the bytes**, not the math. Per-token time = **inter-token latency
  (ITL).**

## 8. Compute-bound vs memory-bandwidth-bound

Every chip has two speeds: **compute** (math rate — FLOPS/TOPS) and **memory
bandwidth** (how fast it moves data between memory and compute units — GB/s). A
task is **compute-bound** if math is the limit, **memory-bandwidth-bound** if
moving data is the limit.

> Kitchen analogy: compute-bound = chef chopping at max hand-speed;
> bandwidth-bound = a fast chef standing idle waiting for ingredients carried from
> a distant pantry. The limit is the **hallway**, not the chef.

**Prefill is compute-bound; decode is memory-bandwidth-bound.** To make **one**
token you stream the **entire** model through the chip. At 400 GB/s bandwidth and
a 4 GB model, that's ~100 tokens/sec max for a single stream — **no matter how
fast the math unit is.** This is why bandwidth is the real bottleneck for local
LLMs, and why a high-TOPS laptop NPU can still be slow (it's **starved for
bandwidth**, not math).

## 9. Batching — why serving many users is efficient

To make one token for **one** user you stream all 4 GB of weights. But if **10
users** generate at the same step, you stream those weights **once** and reuse
them for all 10. The expensive weight-read is **amortized** across the whole
**batch** — 10 users cost barely more bandwidth than 1.

So **throughput** (total tokens/sec across everyone) climbs sharply as you add
users to the batch… until you run out of memory to hold everyone's **KV cache.**

> The whole game of serving: **cram as many users into the batch as memory
> allows.** More batch = more throughput; the limit is KV-cache memory. A memory
> problem is secretly a speed problem.

## 10. Latency vs throughput

- **Latency** — how fast **one** user is served (TTFT + per-token speed).
- **Throughput** — total tokens/sec across **all** users.

They fight: bigger batches raise throughput but can slow any individual user
slightly. Serving is the art of balancing them. **Our engine chases
throughput-under-concurrency on cheap hardware.**

## 11. The problem: wasted memory

Because cache size is unknown up front, old engines reserved one big
**contiguous** block per request for the maximum length. Waste:

- **Internal fragmentation** — a request that used 400 of 2048 reserved tokens
  locks up the other 1648.
- **External fragmentation** — different-sized blocks leave unusable gaps.

Measured: **60–80% of KV-cache memory sat idle.** Since memory = batch size =
throughput, wasting 70% of memory throws away most of your speed.

> Restaurant analogy: reserving a 12-seat table for every couple "in case friends
> show up," then turning away crowds because every table *looks* full.

## 12. PagedAttention — the fix (borrowed from operating systems)

An OS doesn't give each program one contiguous slab of RAM; it hands out
fixed-size **pages** scattered across physical memory and keeps a **page table**
mapping the program's view → physical location. Fragmentation vanishes.

**PagedAttention applies this to the KV cache:**

- Chop the cache into fixed-size **blocks** (default **16 tokens**).
- Each request gets a **block table** mapping its tokens → whatever physical
  blocks are free, anywhere.
- A request grabs a new block **only when it needs one** — never over-reserves.

Waste drops from 60–80% to **under 4%.** More users fit → bigger batches → **up to
24× throughput** on the same GPU. Bonus: requests sharing the same starting text
(e.g. an identical system prompt) can point their block tables at the **same**
physical blocks — copy-on-write, like `fork()`.

> For a systems programmer: this is a **slab allocator + page table** for
> attention memory. The "AI" is a thin layer over allocator/GC problems.

## 13. Continuous batching — the scheduling fix

Users arrive and finish at different times.

- **Static batching (old):** run a group until the **longest** one finishes; a
  user who wanted 10 tokens sits idle waiting for the user who wanted 500. Wasted
  GPU.
- **Continuous batching (vLLM):** the scheduler decides **every single
  token-step** — the instant any user finishes, a waiting user takes the slot; new
  arrivals join mid-flight. The GPU stays ~100% busy.

> Analogy: not a bus that waits to fill and drives everyone to the last stop, but
> a taxi rank where the moment a seat frees, the next person hops in.

This is the source of **~793 tok/s (vLLM) vs ~41 (Ollama)** under concurrent load
— a **19× gap that appears only when many users hit at once.** PagedAttention is
what makes it feasible (you can only cheaply swap sequences in/out if memory is
paged, not contiguous).

## 14. What vLLM *is*, in one paragraph

A serving engine that (1) manages KV-cache memory like an OS manages RAM
(**PagedAttention**), so it can (2) pack many users into big **batches** and (3)
keep the GPU perfectly busy with **continuous batching**. Those three together
make it the throughput champion on datacenter GPUs.

## 15. What *we* build

vLLM is a **heavy** Python/PyTorch system for **big datacenter GPUs** and
professional operators — not embeddable, not tuned for "one normal PC serving my
small team." The easy home tools (llama.cpp, Ollama) run anywhere and are light
but serve basically **one user at a time.**

> **Our engine takes vLLM's two ideas — paged KV cache + continuous batching — and
> puts them into a small, embeddable engine for the hardware normal people own.**
> Light *and* concurrent. That empty middle is the opportunity, and it's built
> from exactly the low-level systems work (allocators, scheduling, memory layout,
> squeezing a bandwidth-limited machine) that this project is about.

Honest ceiling (see `../MEMORY.md`): on bandwidth-bound consumer hardware the
realistic prize is **~2–4× over Ollama** at concurrency, not vLLM's 19×. Getting
that 2–4× *is* the craft.
