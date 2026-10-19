# stoke universal server — design

*2026-09-23, Opus 5.5 with Kabir. Brainstormed and approved section by section
in conversation; this is the written spec for his review before an
implementation plan is written. **Nothing here starts before tag `m7`:**
`src/` is frozen until then (`superpowers/memos/2026-09-18-m7-code-freeze.md`).*

> **Name decided 2026-09-24 (Kabir): the product is Lorren, and the company is
> castagents** (`ROADMAP-SEQUENCE.md` 5.1). Read "stoke" below as Lorren. This
> text is renamed together with the code, after tag `m7`.

> **v0.1 is the Lorren desktop app (Kabir, 2026-10-12;
> `2026-09-25-lorren-desktop-design.md`).** The two specs had each defined an
> updater, a model catalog, a config, an OpenAI endpoint, file hosting and a
> v0.1. From now on the desktop app owns updates, the model catalog and its
> downloads, and settings, cloud keys included. This spec keeps the engine side
> that the app sits on: the router, the scheduler front door, the cloud lanes
> and adapters, and the model families (§3 items 1 and 3–5, §4–§7). Where §2,
> §3 items 2, 6 and 7, §8 and §11 say otherwise, this note wins. The full
> rewrite waits for this spec's review, paused at part 1 of 4.

## 1. What it is

One app, `stoke`, with one OpenAI-compatible API in front of:

- **local open models**, served by stoke's own engine (paged KV + continuous
  batching), and
- **cloud models** (Claude, GPT), called with the user's own API keys.

Every request, local or cloud, enters **stoke's scheduler**: one queue, with
priorities, cancellation, pacing and caching. Local requests then run on the
engine. Cloud requests run on the provider, with stoke's speed-ups applied
where they fit (§5). Claude and GPT cannot run *on* the engine itself: their
weights are not available, and they generate on the providers' servers.

**Who it is for:** people who want big LLMs (Claude, GPT) *and* fast
concurrent serving of local models, from one endpoint on their own machine.
The typical workload is agents, tool chains and fan-outs.

**Success looks like this:**
- a tester installs one binary and points any OpenAI-speaking tool at it;
- by changing only the model name, they get an answer from a local model or
  from Claude or GPT;
- with many parallel requests, the local ones batch and the cloud ones do not
  stall on rate limits.

## 2. Decisions (Kabir, 2026-09-23)

| topic | decision |
|---|---|
| Shape | Built into stoke's own app: one binary, not a separate router process. *(2026-10-12: the app is the Lorren desktop app, and this router lives in the engine underneath it; see the note at the top.)* |
| Build approach | A router in C inside stoke. The Python router in the private `tokenoptimizer` repo (`universal-omni-router`) is **a source to port from, not a component.** Its semantic cache and prompt compression stay out, and its provider translation is ported. |
| Routing | **The user chooses per request, or by a configured default.** Manual: the model name decides. Automatic: difficulty rules, local first. |
| Cloud-when-busy | Kept, **off by default**. |
| Local models | stoke's own engine only; no bundled llama.cpp. Model families are added one at a time through updates, most-downloaded first (§7). |
| Client API | The OpenAI API now; Anthropic's Messages API (for Claude Code) later. |
| Cloud speed-ups | Prompt-cache shaping, rate-limit pacing, a response cache and warm connections. Hedging and prompt compression come later. |
| Updates | The app receives updates Kabir publishes: **app releases and a model catalog, both signed.** *(2026-10-12: delivered by the desktop app's updater and catalog, §3.7 of its spec.)* |
| Order | After tag `m7`, **this comes before the ARM port.** That reverses the 2026-09-01 "ARM NEON first" order (`COMPANY.md` §0, §7). |
| First release | ~~**v0.1, a tester preview with manual routing only** (§8).~~ **2026-10-12: v0.1 is the Lorren desktop app.** This spec's first release becomes the app's cloud-routing release (Claude and GPT, manual routing), after `m7`. |
| Licence | **AGPL-3.0 + a CLA on outside contributions** (`COMPANY.md` §5). This means **v0.1 ships with public source**, not binaries only. |
| Downloads | **castagents.com, the company site, is the front door** (Kabir, 2026-09-24). The files are on GitHub Releases of the public repo. The signed update manifest lives at a stable URL on castagents.com, e.g. `updates.castagents.com` (§11). *(2026-10-12: where the files themselves are hosted, GitHub Releases or the desktop spec's R2 bucket, is settled in this spec's review; the front door and the manifest URL stand.)* |

## 3. Components

```
 tools (OpenAI SDK, Aider, Continue, Open WebUI, ...)
             |  OpenAI API (localhost by default)
             v
 +------------------------------ stoke app ------------------------------+
 | router: manual = the model name decides | auto = difficulty, local 1st |
 | scheduler front door: one queue for local and cloud                   |
 |   +- local: model manager -> stoke engine (existing scheduler)        |
 |   +- cloud: per-provider lanes (pacing) -> adapters -> HTTPS          |
 | updater: signed app releases + signed model catalog                   |
 +-----------------------------------------------------------------------+
```

1. **Engine (`libstoke`).** Its role is unchanged. It gains model families over
   releases and stays dependency-free.
   - **Code change:** today `LIB_SRC` compiles every `src/*.c` except `main.c`
     into the library, the server included.
   - **All new app code lives in `src/app/`** and links only into the `stoke`
     binary. TLS and the updater must never enter `libstoke.a`, the
     embeddable product.
2. **Model manager.**
   - **The catalog** maps a friendly name (`llama3.1:8b`) to a download URL,
     sha256, size, family, minimum engine version and chat format.
   - **Commands:** `stoke pull | list | rm`.
   - **Storage:** `~/.stoke/models/`.
   - **Resident models:** one local model at a time in v0.1. From v1, several
     within a memory budget, with the least recently used unloaded first.
   - **Chat formats are built in per family.** There is no general Jinja
     template engine.
3. **Router.** It resolves every request (§4). Response headers report the
   route: `X-Stoke-Route` (local | cloud), `X-Stoke-Model`, `X-Stoke-Reason`.
4. **Scheduler front door.** One admission queue for local and cloud.
   - **Local requests** go to the engine's existing continuous-batching
     scheduler.
   - **Cloud requests** go to per-provider lanes, paced to rate limits
     (§5.2).
   - **Cloud I/O runs on worker threads,** so it never blocks the server's
     event loop.
5. **Cloud adapters.**
   - **OpenAI:** passthrough.
   - **Anthropic:** translation between OpenAI chat and Anthropic Messages,
     covering the system prompt, tools and streaming events. Ported from
     tokenoptimizer's `app/translation`.
   - **Keys** are stored in `~/.stoke/config.json`, with environment
     overrides.
   - **HTTPS uses the operating system's TLS:** WinHTTP on Windows; on Linux,
     the system OpenSSL loaded at runtime. If OpenSSL is absent, cloud is
     disabled with a clear message. The plan confirms this choice (§13).
   - **A "local only" switch** turns cloud off entirely.
6. **Updater.** Two channels Kabir publishes to: (a) app releases, and (b) the
   model catalog.
   - **Signing:** both are signed with an **ed25519 key kept offline**. The
     public key is compiled into the app, and nothing unsigned is ever
     applied.
   - **v0.1:** checks a signed version file and tells the user a new version
     exists. The tester downloads it themselves.
   - **v1:** downloads to a temp file, verifies the signature and sha256, and
     swaps on restart. The previous version is kept for rollback. User
     setting: auto | notify | off.
7. **CLI and config.**
   - `stoke serve` becomes the gateway.
   - New commands: `stoke pull | list | rm | update | config`.
   - One config file, `~/.stoke/config.json` (stoke already has a JSON
     parser).
   - The server already binds to `127.0.0.1` by default and keeps doing so.
     Binding to a network interface becomes opt-in, and needs a local access
     key once the app holds cloud keys.

## 4. Request flow and routing

**Flow for a chat request:**
1. An OpenAI-format request arrives with a `model` name.
2. The router decides where it goes:
   - a local catalog name runs locally;
   - a cloud name (`gpt-*`, `o*`, `claude-*`, or `openai/…`, `anthropic/…`)
     goes to that provider;
   - `auto` goes to automatic mode (v0.2 and later).
3. **Local path:** make sure the model is loaded, apply the chat format,
   tokenize, submit to the engine's scheduler, and stream SSE back on the
   existing path.
4. **Cloud path:** translate if needed, pace, send over HTTPS, and stream back
   as OpenAI SSE.

**Automatic mode (v0.2):**
- **Hard rules first:**
  - images, a context longer than the local model's, or tool calls the local
    family does not support → cloud;
  - no cloud key, or local-only on → local.
- **Difficulty score.** It is built from request features: prompt tokens,
  turns, code, tools, math, explicit reasoning requests (`reasoning_effort`,
  "step by step") and the requested `max_tokens`.
  - Below the user's threshold → the local model set for automatic mode
    (`auto_local_model`; default: the largest installed chat model that fits
    the memory budget).
  - Above it → the cloud model set for automatic mode (`auto_cloud_model`).
  - The default threshold favours local.
  - **The weights are set from a labelled prompt sample before v0.2 ships,
    and recorded with the release.**
- **Per-request override:** the header `X-Stoke-Prefer: local|cloud`.
- **Decision log.** Each entry records the features, score, route, latency
  and tokens, **never the prompt text**. It is a local JSONL file, and the
  training data for a later learned judge.
- **Cloud-when-busy** (v1, off by default): when the predicted local wait
  exceeds a configurable limit (`busy_spill_seconds`), an automatic request
  goes to the cloud.

## 5. Cloud speed-ups

stoke cannot make Claude or GPT generate faster. What it can remove is the
waiting *around* generation.

1. **Prompt-cache shaping (v1).** This is stoke's prefix-reuse idea applied to
   the cloud.
   - It marks the stable prefix (system prompt, tool definitions) with
     Anthropic's `cache_control`.
   - It keeps the prefix byte-stable, and uses OpenAI's cache-routing key
     where the API offers one, so automatic caching hits.
   - The payoff is a much faster first token and a lower cost when agents
     resend the same long prompt.
   - It is verified on the exact bytes sent.
2. **Rate-limit pacing (v0.1).** Token buckets per provider and key, fed by
   the provider's limits and `retry-after`. There are no thundering-herd
   retries, so many parallel agents stop stalling on 429s.
3. **Response cache (v1).** An exact match on the normalized request answers
   instantly. It only caches a request whose output is deterministic
   (temperature 0, or an explicit seed), unless the user opts in for sampled
   requests. A semantic cache follows the embedding engine (update 4).
4. **Warm connections (v0.1).** Keep-alive HTTPS per provider saves the setup
   time on every request.

Later: **hedging**, off by default, which sends a request to two providers and
takes the first answer at double the cost. Also later: **prompt compression on
the local engine**, off by default, once the embedding engine exists.

## 6. Failure handling

| what fails | manual mode (the user named the model) | automatic mode |
|---|---|---|
| Cloud rate limit, server error or timeout | Bounded retries with back-off, then the provider's error, in OpenAI format | The same retries, then fall back to a local model |
| Cloud stream breaks mid-answer | An error event, and the stream ends | The same, with no silent restart: part of the answer was already delivered |
| Local model not downloaded | An error with a `stoke pull <name>` hint | Another installed local model, or the cloud |
| Not enough memory to load a model | Unload the least recently used; if it is still too big, an error | Unload the least recently used; if it is still too big, the cloud |
| Missing or invalid key | A clear error naming the setting; keys are never logged or returned | The same |
| An update fails | Never a broken app: temp file, signature + sha256 check, swap on restart, previous version kept | The same |

## 7. Local models: most-downloaded first

The source is Ollama's library sorted by popularity, read 2026-09-23. The top
30 models total 816.5M all-time pulls. All-time counts over-weight older
models; newer ones (qwen3, gemma3, gpt-oss) matter more going forward.

| group | models | share | status | release |
|---|---|---|---|---|
| Llama 3.x | llama3.1, llama3.2, llama3 | 28% | runs today | v0.1 |
| Qwen3 (dense) | qwen3 | 5% | runs today | v0.1 |
| Qwen2 architecture | qwen2.5, qwen2.5-coder, qwen2, deepseek-r1 local sizes | 20% | near-copy of Qwen3 | v1 |
| SentencePiece tokenizer | mistral, llama2, codellama, tinyllama (+ mistral-nemo's tekken) | 7% | Llama architecture + a new tokenizer type | update 1 |
| Gemma | gemma3, gemma2, gemma | 10% | new graph + SentencePiece | update 2 |
| Phi | phi3, phi4 | 3% | new graph | update 3 |
| Embeddings | nomic-embed-text, mxbai-embed-large, bge-m3 | 13% | new encoder path + `/v1/embeddings` | update 4 |
| Mixture-of-experts | gpt-oss-20b, qwen3 30B-A3B, qwen3-coder | 3% | expert routing + the MXFP4 format | update 5 |
| Not planned | vision models, gemma4, qwen3.5/3.6, 70B+ | 11% | — | — |

Weight formats arrive with the families that need them: Q4_0, Q5_K, F16/BF16
and MXFP4. stoke has Q8_0, Q4_K and Q6_K today.

**The bar for shipping a family** is the same correctness method Llama-3.2
passed (rung B5):
- a tokenizer byte-exact against `llama-tokenize`;
- output parity against the llama.cpp reference;
- graders that are themselves mutation-tested;
- the result recorded in a gate report.

## 8. Releases

| release | contents | rough timing (after tag `m7`) |
|---|---|---|
| **v0.1 — tester preview** *(2026-10-12: superseded. v0.1 is the Lorren desktop app, and this row becomes the app's cloud-routing release)* | Manual routing. Local Llama 3.x + Qwen3, one model resident. Cloud OpenAI + Anthropic with the user's keys. Rate-limit pacing and warm connections. `stoke pull`. A signed "update available" notice. Windows and Linux x86-64, **open source under AGPL-3.0 + CLA** (§11). | ~3–4 weeks |
| **v0.2** | Automatic routing (difficulty rules, local first) and the decision log. | ~+2 weeks |
| **v1** | Qwen2.5 and DeepSeek-R1 local sizes. Prompt-cache shaping and the response cache. Cloud-when-busy (off by default). Several models resident. Self-installing signed updates. | ~2–2.5 months |
| **updates 1–5** | The model families of §7, in that order. | ~3–4 more months |
| **later** | The Anthropic client API (Claude Code), hedging, prompt compression, a semantic cache, a learned judge. | — |

The timings are **derived estimates**, from the Llama-3.2 bring-up (about a week
including its correctness checks) and the size of the router. The plan refines
them.

**One plan per release.** This spec is too big for a single implementation
plan. The first plan covers **v0.1 only**. v0.2, v1, and each model family
(each with its own correctness gate) get their own plans, written when they
start.

## 9. Testing

- **Router:** a test for each difficulty feature, the thresholds, the hard
  rules and the override header. Plus a decision-log test that the prompt
  text never appears.
- **Cloud:**
  - A fake OpenAI/Anthropic server gives golden translation tests: request,
    response, streaming, and tool calls.
  - The same fake server injects failures: 429 with `retry-after`, 5xx,
    timeout, a mid-stream drop and a bad key.
  - Prompt-cache shaping is checked on the exact bytes sent.
  - An optional manual live check uses Kabir's own keys (it costs cents).
- **Pacing:** tests run on a fake clock.
- **Updater:**
  - **A tampered package must be refused.** Flip one byte of a signed release
    and prove the install fails; the grader must be able to fail.
  - Rollback and an interrupted download are also tested.
- **Model manager:** a checksum mismatch on pull is refused, and least recently
  used models are evicted under a memory budget.
- **Existing gates keep running:** the unit suite, ASan/TSan, and the serving
  soak against the gateway.

## 10. Security and privacy

- **Keys** stay on the user's machine. They are never logged, never returned,
  and never sent anywhere except their own provider.
- **Cloud is opt-in.** It needs a key, and "local only" turns it off.
- **The decision log** stores features, never prompts.
- **The gateway** stays on localhost by default. Binding to a network
  interface needs a local access key, since anyone who can reach the gateway
  can spend the user's cloud keys.
- **Updates** are signed with an offline key, and the verification is
  mutation-tested. An auto-updater is a remote-code channel into every user's
  machine, so the signing key is the most sensitive secret in the project.

## 11. Blocked on Kabir

The licence is decided: AGPL-3.0 + a CLA (Kabir, 2026-09-23). Everyone who
receives AGPL binaries is entitled to the source, so **v0.1 ships as open
source.** That puts all of the following before v0.1:

1. ✅ **The name is DECIDED (2026-09-24): Lorren. The company is castagents**
   (`ROADMAP-SEQUENCE.md` 5.1 has the availability evidence). Still owed:
   register lorren.ai + lorren.dev, and the lawyer's EU/India trademark check.
2. **The CLA**, lawyer-reviewed, granting to the company's legal entity
   (castagents), or to Kabir with a right to assign to it. A CLA bot on the
   public repo blocks every outside PR until it is signed.
3. **A clean public repo.** It starts from a snapshot of code, tests, README,
   BENCHMARKS, LICENSE, CLA and CONTRIBUTING. The internal documents
   (`COMPANY.md`, the memos, `MEMORY.md`) and the private history stay in the
   private repo.
4. **Downloads.**
   - **castagents.com, the company site, is the front door** (Kabir,
     2026-09-24): download page and docs, e.g. `castagents.com/lorren`. A
     static site is enough. ⚠ castagents.com and castagent.com are not
     owned yet (corrected 2026-10-19: another registrant holds both, and Kabir
     will not buy them before the product earns money). The updater depends
     on the domain, so one Kabir controls is needed before any public release.
   - **GitHub Releases on the public repo** hold the files: free bandwidth,
     plus checksums.
   - **The signed update manifest** lives at a stable URL on castagents.com
     (e.g. `updates.castagents.com`), so file hosting can move later without
     breaking installed apps.
5. **The offline update-signing key**, generated and stored safely.

Optional: a Windows code-signing certificate. Without one, testers see
SmartScreen warnings.

**Licence compatibility of the planned parts:**
- OpenSSL 3 and mbedTLS are Apache-2.0, which AGPL-3.0 can include.
- WinHTTP is a system library.
- Any code ported from llama.cpp is MIT and keeps its notice.
- Model weights keep their own licences. The app downloads them from their
  original hosts and shows each model's licence at `stoke pull`.

## 12. Out of scope for now

Vision and multimodal models; gemma4 and qwen3.5/3.6; 70B+ models; the
Anthropic client API (later); hedging; prompt compression; the semantic
cache; a learned judge; GPU backends; macOS.

## 13. Open questions for the implementation plan

1. Linux TLS: the system OpenSSL loaded at runtime, or a vendored mbedTLS.
2. Multi-model serving in `server.c`: one step loop per resident model, or one
   engine thread each. The server holds one engine today.
3. Where releases and the catalog are hosted (Kabir, §11).
4. The labelled prompt sample for the difficulty weights: its source and size.
5. Which local families support tool calls (this feeds the hard rule).
6. How the response cache normalizes a request into its key.
