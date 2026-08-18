# The surface: what a human actually touches

2026-08-18, Opus 5 session. **Decision ratified by Kabir in-session** ("so webapp
it is"), so this note records reasoning and consequences rather than proposing.

Canvas of the design: `design/stoke-swarm-console.html` (published as an
Artifact; four artboards — three directions plus the block-map anatomy).
Working files are `design/*.dc.html`, and the canvas re-seeds from them.

## 1. The gap this closes

`docs/ROADMAP.md` has no human-facing surface anywhere in it. Phase 1 ships
`libstoke.a` + `stoke.h` + the CLI, with "the OpenAI-compatible server is a thin
frontend"; Phase 2 is the bridge layer (alias registry, `stoke pull`, LRU
residency, `/metrics`); Phase 3 folds into AySen `local()`. Every one of those
is an interface for *programs*. Nothing in the plan is a thing a person opens.

That is not an oversight to paper over — an engine can be finished and correct
without a GUI — but the question "what does this run on" had never been asked,
and the answer turns out to be nearly free, so it is worth writing down before
M5/M6 harden the server's shape without it.

## 2. The decision

**The UI is a web page served by stoke itself**, on `127.0.0.1:8080`, out of the
same static binary. Not Electron, not Tauri, not a separate app.

The reasoning, in order of weight:

1. **It is nearly free.** M6 already owes an HTTP/1.1 + SSE server with
   OpenAI-compatible endpoints, and the Phase-1 goal already demands one static
   binary. A console is then a page that binary already knows how to serve. No
   second runtime, no second language, no second release pipeline.
2. **A native app is not free.** Electron is LM Studio's ~300 MB idle footprint.
   Tauri is lighter (Jan and Msty both use it) but costs a Rust toolchain,
   WebView2, per-platform installers, an update channel, Windows code signing —
   an EV certificate, or SmartScreen warning off every early installer — and AV
   false positives on a binary that opens sockets. None of that makes inference
   faster.
3. **It gets the network for free.** Watching a swarm from a phone while the box
   works is a real use of this specific product, and a native app cannot do it
   without *also* building the server.
4. **It gets the platforms for free.** Phase 1.5 is CUDA + ARM NEON; those users
   are on Linux and macOS. One frontend covers all of them.
5. **It is additive, not exclusive.** If a tray icon, launch-at-login and a
   model-folder picker ever earn their place, Tauri wraps *this exact page* with
   `stoke.exe` as an `externalBin` sidecar. Same frontend, no rewrite.

### What web-first gives up

No tray icon, no launch-at-login, no OS file picker beyond the browser's, and
the user must open a tab (mitigate with `stoke serve --open`). Accepted.

### The one requirement it creates

"Watch it from your phone" means binding something other than `127.0.0.1`, and
an unauthenticated inference server on a LAN is a real exposure. **Default stays
localhost; any non-loopback bind must require a token.** That is a server-side
decision, and M6 is where it lands.

## 3. Why not another chat window

Researched 2026-08-18. The chat UI is a commodity and is being given away:

- **LM Studio** — Electron, ~300 MB baseline, model hub and GPU wizards.
- **Jan** — Tauri, MIT, wraps llama.cpp, OpenAI-compatible on `:1337`.
- **Msty** — Tauri, split-pane, multi-model compare, paid tiers.
- **llama.cpp itself** now bundles a SvelteKit web UI with multimodal input,
  MCP integration, conversation persistence and a router mode that switches
  models mid-conversation. Free, in the box, from the project stoke is measured
  against.

All of them run the same backends, so they land within a few percent of each
other on speed and compete on interface. **Building a fifth chat window is
competing where we have nothing to offer.**

Meanwhile the observability that exists for what stoke actually does is
datacenter-shaped: Prometheus counters (`vllm:gpu_cache_usage_perc`,
e2e-latency histograms) rendered on Grafana, aimed at cluster operators. Nobody
surfaces block-level KV state, prefix sharing or scheduling decisions to the one
person whose laptop is doing the work.

**So the console shows concurrency, and the chat box is the least important
thing on it** — present only because you need a way to create load.

## 4. What the four tabs oblige the engine to expose

This is the part that matters for M5/M6 planning: designing the UI first turns
it into a specification, instead of the UI being whatever the engine happened to
leave lying around. None of the following exists today.

| Tab | Needs | Where it lands |
|---|---|---|
| Swarm | per-sequence state, ctx length, rate, TTFT, last text fragment; block-map ownership; a ring of recent step durations flagged prefill/decode | M5 (the scheduler already computes most of it for `bench --serve`) |
| Models | registry with geometry, file, size, ctx max, measured single-stream rate, LRU rank, state; load/unload; `pull` | Phase 2 bridge layer, already planned |
| Requests | per-request record: queue → prefill → decode phases, prefix-hit blocks and tokens, status, client | M6, new — nothing times a *request* today |
| Logs | structured ring buffer: timestamp, source, level, message | M6, new |

Two mechanisms carry all of it, and both reuse what M6 already owes:

- `GET /internal/state` — one snapshot for first paint.
- `GET /internal/events` — **SSE**, the same streaming primitive the OpenAI
  endpoints already require. No WebSocket, no second protocol, no polling.

**Encoding note, worth getting right the first time:** the block map is 384
entries today and scales with pool size, so do not send an object per block per
frame. Run-length encode ownership (`[owner, count]` pairs) and emit only on
change; the map is overwhelmingly long runs of one owner.

## 5. Still open

- **Frontend stack.** Vanilla + a tiny renderer, or Svelte compiled to static
  files. Recommendation: vanilla first — the page is one document, and a build
  toolchain in a from-scratch C project is a dependency with a maintenance bill.
- **How the page ships.** Phase-1 says one static binary, so the assets get
  embedded as a C array at build time (an `xxd -i`-shaped step, no new
  dependency). A `--ui-dir` override for development.
- **Which direction.** B ("Control room") is chosen and on its second pass.
  A ("Instrument") and C ("Blueprint") stay on the canvas as the record of what
  was rejected and why.

## 6. What this does not change

Nothing about M4.5. The engine work in flight — L4's attention-on-the-pool, the
owed A/B, the P-BAR re-grade — is untouched by any of this, and the surface
question does not become urgent until M6.
