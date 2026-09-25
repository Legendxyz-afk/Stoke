# Lorren desktop app — design

*2026-09-25, Opus 5.5 with Kabir. Brainstormed on 09-24 and 09-25 and approved
section by section in conversation: part 1 (architecture and updates), part 2
(screens, failures, tests, scope), the shell (Tauri, chosen after a live
Tauri-versus-native demo), and the UI (the Control Room design plus a Chat
tab). This is the written spec for his review before an implementation plan is
written. **`src/` is not touched:** the engine is frozen until tag `m7`
(`superpowers/memos/2026-09-18-m7-code-freeze.md`), and the app wraps it as it
is.*

> **Names (decided 2026-09-24):** the product is Lorren and the company is
> castagents. The engine is still called `stoke` in this repo until the rename
> after `m7`; the app ships it as `lorren-engine.exe`.

## 1. What it is

A Windows desktop app, **Lorren**, for using your own local models: Nexus
(`M:\New Drive\ai_models\nexus.gguf`, your Llama-3.2-3B fine-tune),
Qwen3-4B-Instruct-2507 and Llama-3.2-3B-Instruct. Chat is the main use. The
same window is the console for the engine: the Control Room design from
2026-08-18. The app runs today's `stoke serve` unmodified as a child process,
and updates itself from releases you publish and sign.

**Who it is for:** you first, as the consumer; then the Lorren v0.1 testers.

**Success looks like this:**
- One installer. Lorren opens with your models already listed, and Nexus
  arrives with its trader system prompt.
- Chat with streaming replies; stop, regenerate and edit; chats survive a
  restart.
- Switch models from the app. A crashed engine comes back without losing the
  chat.
- A version you publish installs itself on the next restart. A tampered one is
  refused.
- The console tabs show what the engine is doing, as far as the frozen engine
  lets the app see (§8).

## 2. Decisions (Kabir, 2026-09-24 and 09-25)

| topic | decision |
|---|---|
| main use | a chat window, chosen over "an API for my code" and "the full v0.1 with Claude/GPT" |
| scope | the full product at enterprise grade, not an MVP. The build is phased; the ambition is not |
| shell | **Tauri 2** (09-25). Electron was picked on 09-24; a C++ host with WebView2 and a fully native Win32 app were also weighed. The deciding demo is below this table |
| UI | **the Control Room design** (`design/ControlRoom.dc.html`, chosen 2026-08-18) **plus a Chat tab** in its visual language, approved from the demo on 09-25 |
| theme | dark only in v1: the finalized design has one palette. This replaces part 2's "light / dark / system" |
| engine | today's `stoke serve`, unmodified, in a child process |
| updates | "through me": two ed25519-signed (minisign) channels you publish, app releases and a model catalog, hosted on Cloudflare R2 behind `updates.castagents.com` |
| models | local only. Claude and GPT arrive with the universal server (`2026-09-23-stoke-universal-server-design.md`) |
| platform | Windows first |
| first milestone | a working demo you verify before the rest is built (§11) |

**The shell demo (09-25).** The Control Room design and the new Chat tab were
built twice: as a web page in WebView2, which is what a Tauri app shows on
Windows, and as a native C++ window drawn with Direct2D and DirectWrite. The
results:

- They rendered almost pixel for pixel alike, and equally sharp. Both draw
  text through DirectWrite with subpixel smoothing: the mean edge step was
  80.3 native against 81.4 web.
- Native was far lighter: 1 process and 61 MB, against 8 processes and 291 MB
  for the web page in a full Edge window. A real Tauri app runs lighter than
  full Edge (estimated 100–200 MB).
- Native cost more to build: about 870 lines for 2 tabs, against about 480 for
  all 5 in the web version.
- The web version got text selection, scrolling, hover states and typeset
  maths for free. Native would have to hand-build each, and its maths was a
  one-line approximation.

The web version of the demo is kept as `design/lorren-console-demo.html`: the
reference for §4.

**This is the path the 08-18 note already named.** "The surface is a web
console" (`superpowers/design-notes/2026-08-18-the-surface-web-first.md`)
said: "Tauri wraps *this exact page* with `stoke.exe` as an `externalBin`
sidecar. Same frontend, no rewrite." The engine-served console it chose, the
same page on `127.0.0.1` and viewable from a phone, stays possible later from
the same frontend.

## 3. Architecture

```
Lorren.exe (Tauri 2)
 ├─ host (Rust)
 │   ├─ engine supervisor ──spawns──► lorren-engine.exe  (stoke serve, frozen source,
 │   │                                                       127.0.0.1, random port)
 │   ├─ chat service        SSE client; saves while streaming; Stop = close the connection
 │   ├─ local API proxy     optional OpenAI endpoint for your own code (off by default)
 │   ├─ request monitor     one record per request, from chat and proxy → Swarm, Requests
 │   ├─ store               SQLite: chats and versions, search, settings, library, request log
 │   ├─ model library       your folders + Ollama import + Modelfile sidecar + GGUF check
 │   └─ updater + catalog   minisign-verified app updates and model catalog
 ├─ UI (React + TypeScript in WebView2) ⇄ typed commands + event channels
 └─ engine: child process, inside a Windows job object so it dies with the app
```

Each unit below says what it does, how it is used, and what it depends on.

### 3.1 Engine supervisor

Starts, watches and restarts the engine. Used by everything that talks to the
engine; depends on the sidecar binary and the settings.

- **Launch:** `lorren-engine.exe serve <model.gguf> --host 127.0.0.1 --port 0
  --ctx <C> --seqs <S> --threads <T> --kv q8`. It waits for the banner
  `stoke serve: listening on 127.0.0.1:<port>  ctx=<N>`, which the engine
  prints and flushes before serving. The load timeout is 120 s.
- **The context is capped at 4,096 tokens.** The engine clamps every model to
  4,096, whatever its header says (`STOKE_M2_MAX_CTX`, `model.c:43`, a
  milestone-2 decision). A larger `--ctx` is refused at startup: on 09-25
  `--ctx 8192` failed with "max_ctx 8192 exceeds the model's 4096". *(Corrected
  09-25: the first version of this spec read the 131,072 in the header as the
  engine's limit.)*
- **Always explicit sizes.** Without `--seqs`, the engine reserves KV for 8
  full-length sequences (`stoke.c` ~198–226). Defaults: context 4,096;
  parallel replies 2; KV `q8`; threads "auto", meaning the engine's default of
  12 when `--seqs` is above 1.
- **Memory estimate before loading:** the file size (it is memory-mapped)
  plus seqs × ctx × KV bytes per token. The KV cost comes from the GGUF's
  layers × KV heads × head dimension × 2 × 34/32 bytes at `q8`. That is
  59.5 KiB per token for Llama-3.2-3B and 76.5 KiB for Qwen3-4B, so the
  defaults cost about 0.47 GiB and 0.6 GiB. A load that would leave less than
  1 GB of RAM free is not started; the app offers smaller settings instead.
- **Health:** `GET /health` every 5 s. Three misses in a row means the engine
  is hung.
- **Crash or hang:** restart after 1 s, then 5 s, then 30 s. After three
  failures in 10 minutes it stops, shows the log, and offers a manual restart.
- **Output:** the engine's stderr goes into a ring buffer and the log file. A
  load failure shows the engine's own last error line, for example an
  unsupported architecture.
- **Lifetime:** the engine runs inside a job object with
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, so it never outlives the app, even when
  the app crashes.
- **Model switch:** stop the engine and start it with the other model. If a
  reply is streaming, the app asks first: switching cancels it.

### 3.2 Chat service

Turns a chat into engine requests and streams the reply into the UI and the
store. Depends on the supervisor, the store and the request monitor.

- **Request:** `POST /v1/chat/completions` with `stream: true`,
  `stream_options.include_usage: true`, the chat's sampling parameters, and
  **always** a `max_tokens`. Left out, the engine answers only 128 tokens
  (`server.c:71`).
- **Sampling the engine accepts:** `temperature` 0–100, `top_p`, `top_k`,
  `repeat_penalty` (multiplicative), `seed`, `stop` (up to 4) and
  `max_tokens` (1 to the context). It refuses a non-zero `frequency_penalty` or
  `presence_penalty`, so the UI does not offer them.
- **Context budget:** `max_tokens` is the smaller of the chat's reply length
  and the context minus the prompt.
  - The app has no tokenizer. It learns prompt sizes from each reply's `usage`,
    and from the engine's `400 context_length_exceeded` reply, whose message
    states the prompt's token count: "prompt is N tokens and max_tokens is M,
    which exceeds the context length of C".
  - On that error it drops the oldest turns and retries once. It never drops
    the system prompt or the latest message. The chat marks where the history
    stopped being sent.
- **Streaming:** tokens reach the UI over a Tauri channel. The message is
  saved every 500 ms and when it finishes. Stop aborts the HTTP request:
  closing the connection cancels the generation inside the engine (verified in
  `server.c` on 09-24).
- **Liveness:** while a request waits, the engine sends an SSE ping every 15 s
  (`--sse-ping-ms`, default 15,000). Its idle reaper closes only connections
  that never sent a request, so a long prompt read is safe. If no bytes arrive
  for 45 s, the app treats the engine as hung.
- **Limits:** a request body over 1 MiB is refused with 413; the context runs
  out long before that.
- **Background work:** with 2 parallel replies, one chat reply can run next to
  one background job. After a chat's first reply, the background job asks the
  model for a 3–6 word title.

### 3.3 Local OpenAI API (optional)

A fixed-port endpoint for your own programs. Depends on the supervisor and the
request monitor.

- **Off by default.** When on, it listens on `127.0.0.1:<port>` (the default
  port is picked in the plan) and forwards `/v1/chat/completions`,
  `/v1/completions` and `/v1/models` to the running engine.
- **Only the loaded model.** A request that names another model gets a `404
  model_not_found`; v1 never switches models on an API request's behalf.
- **Visible in the console:** every proxied request goes through the request
  monitor, so other programs' traffic shows up in Swarm and Requests too.
- Localhost only. LAN access with an API key is out of scope.

### 3.4 Request monitor

Keeps one record per request, from the chat and from the proxy, and feeds the
Swarm and Requests tabs. It depends on nothing but the requests passing
through it.

- **Record fields:** id, time, source (chat or API), endpoint, prompt tokens,
  generated tokens, time to first token, tokens per second, total time, status
  (200, cancel or error), the last text fragment, and the gaps between arriving
  tokens.
- **Kept:** the recent records in memory, plus a capped `request_log` table.
  The cap is set in the plan.

### 3.5 Store

SQLite through `rusqlite` with SQLite bundled and FTS5 enabled. The file is
`%APPDATA%\Lorren\lorren.db`, in WAL mode.

- **Messages form a tree.** Editing a message or regenerating a reply adds a
  sibling; the chat shows one path, and ‹ 1/2 › switches between siblings.
- **Tables:**
  - `chats`: title, model, system prompt, parameters, pinned, timestamps;
  - `messages`: parent, role, content, token counts;
  - `messages_fts`: FTS5 over message text and chat titles;
  - `models`, `presets`, `settings`, `catalog`, `downloads`, `request_log`.
- **Safety:** `PRAGMA quick_check` at startup, and a daily backup with `VACUUM
  INTO`, keeping 7 days. Restore is in Settings.
- **Other folders:** logs rotate in `%LOCALAPPDATA%\Lorren\logs` (the app log
  plus the engine's stderr). Downloaded models go to
  `%USERPROFILE%\Lorren\models`.

### 3.6 Model library

Finds models, checks whether they can run, and holds per-model presets.
Depends on the file system and the store.

- **Watched folders:** by default `M:\New Drive\ai_models`,
  `M:\New Drive\Desktop\Programming\stoke\models` and the downloads folder.
- **Ollama import:**
  - Manifests are read from
    `%USERPROFILE%\.ollama\models\manifests\registry.ollama.ai\library\<name>\<tag>`.
  - The layer with media type `application/vnd.ollama.image.model` names the
    GGUF blob (`blobs\sha256-<hex>`), which is used where it is and never
    copied.
  - `…image.system` supplies the system prompt, and `…image.params` the
    parameters.
- **Modelfile sidecar:** a `Modelfile` in the same folder as a GGUF applies to
  it when its `FROM` names that file.
  - This is how Nexus gets "You are Nexus AI, a professional quantitative
    trader. You give concise, decisive trading verdicts."
  - Its `FROM` still points at the old `C:/Users/Kabir/ai_models/`, so it is
    matched by file name, not by path.
- **Compatibility check,** from the header only, without loading:
  - the architecture is `llama` or `qwen3`;
  - every tensor is F32, Q8_0, Q4_K or Q6_K;
  - the vocabulary has the tokens of one of the engine's two chat templates,
    ChatML (`<|im_start|>`, `<|im_end|>`) or the Llama-3 header tokens.

  A failure shows CAN'T RUN with the reason (for example, `qwen2.5vl:3b` is a
  vision architecture). The engine would stop with an error anyway; the check
  says why before trying.
- **Facts shown** in the Models tab: layers, query and KV heads, head
  dimension, vocabulary size, file, size, maximum context, KV cost per token,
  chat template, where the system prompt came from, and the sha256 (computed
  in the background and cached).
- **Presets:** each model gets a system prompt and default sampling
  parameters. Nexus starts from its Modelfile.

### 3.7 Updates and the model catalog

**App updates** use `tauri-plugin-updater`.
- **Your side:** the release script (`npm run release`) builds the installer,
  signs it with your minisign (ed25519) key, and uploads the installer, its
  `.sig` and `latest.json` to R2 under
  `updates.castagents.com/app/<channel>/`.
- **App side:**
  - checks at startup and every 6 hours;
  - the setting is auto, notify or off (default auto), on a stable or beta
    channel (default stable);
  - downloads in the background and verifies the signature against the
    public key compiled into the app;
  - installs on the next restart, with a per-user NSIS installer (no admin
    prompt).
- **A bad update:** an unsigned or tampered one is thrown away, and the app
  keeps running.

**The model catalog** is `catalog.json` plus `catalog.json.minisig` under
`updates.castagents.com/catalog/`.
- **A separate offline key signs it,** so a leaked catalog key cannot ship
  code.
- **Each entry has:** id, name, version, description, architecture,
  quantization, size, sha256, a URL on R2, an optional system prompt and
  parameters, and the minimum app version.
- **In the app:** a Discover panel in the Models tab lists the entries.
  Downloads resume (HTTP Range) and are checked against their sha256 before
  they join the library. A newer version of an installed model shows "update
  available".

**Hosting:** an R2 bucket with `updates.castagents.com` as its custom domain.
castagents.com's DNS is already on Cloudflare, and R2 charges nothing for
downloads.

**Keys:** two minisign key pairs, one for the app and one for the catalog.
You generate them, keep them offline (USB plus a password), and give me the
public keys. Rotating a key means shipping an update, signed with the old key,
that trusts the new one.

### 3.8 The UI shell

- **Stack:** React + TypeScript + Vite, bundled into the app. No remote
  content. A strict CSP (`default-src 'self'`, no `eval`). Tauri 2 capabilities
  let the UI call only the app's own commands.
- **IPC:** typed commands (list chats, send, stop, switch model, …) plus
  channels for token streams and engine and monitor events.
- **Fonts:** the design's two faces, Archivo (400/500/600) and JetBrains Mono,
  are bundled under the OFL, never loaded from Google Fonts.
- **Markdown:** `react-markdown` with `remark-gfm`, a code highlighter and
  KaTeX. Raw HTML in model output is never rendered.
- **Streaming redraws are throttled** to at most 30 per second, so the UI does
  not compete with the engine for CPU.
- **Tauri plugins provide** a single instance, the tray icon, launch at login,
  notifications, and remembered window size and position.

## 4. The screens

The visual language comes from `design/ControlRoom.dc.html`:
- a dark ground, `oklch(0.15 0.006 255)`;
- Archivo for text and JetBrains Mono for numbers and labels;
- square panels with 1 px borders, and caps mono labels tracked at 0.14 em;
- three accents: amber `oklch(0.76 0.09 70)`, a "shared" blue
  `oklch(0.70 0.07 250)` and a "flame" red `oklch(0.66 0.09 30)`;
- a 54 px top bar with tabs, a 396 px right-hand detail panel and a 30 px
  status bar.

Every screen except Settings has a working reference in
`design/lorren-console-demo.html`.

**Top bar:**
- the `lorren` wordmark and the version;
- the tabs Chat · Swarm · Models · Requests · Logs · Settings;
- on the right, the engine state (serving, loading, stopped or crashed), the
  model, the threads, the KV mode, and RAM used out of the total.

**Chat** (new; approved from the demo):
- **Left (260 px):** the chats list, with search, pin and `+ NEW`. Chats are
  titled automatically, and each row shows the model and the time.
- **Centre:**
  - a header with the title and the model's facts;
  - the messages: `YOU` and the model's name as caps mono labels, and replies
    rendered as Markdown;
  - under each reply, its token count, tokens per second and time to first
    token, plus REGENERATE · EDIT · COPY · ‹ 1/2 ›;
  - the composer: a model picker, the input (with text-file attachments), and
    SEND, which becomes STOP while a reply streams.
- **Right: "This reply":**
  - LIFECYCLE bars for queue, prefill (time to first token) and decode;
  - a CONTEXT gauge (tokens used out of the context), with the trimming note;
  - MODEL facts;
  - the chat's SAMPLING values, editable per chat.

**Swarm** (the design, fed by the request monitor):
- **KPI strip:**
  - aggregate tokens per second with a 60 s sparkline;
  - active requests;
  - TTFT p50/p99;
  - memory;
  - the worst stall, measured at the app as the longest gap between arriving
    tokens across the active streams.
- **Agents table:** one row per active request, from the chat or the API.
  States are QUEUED, PREFILL (waiting for the first token) and DECODE
  (streaming), with the context, rate, time to first token and the last text.
- **Decode interval chart:** built from the gaps between arriving tokens,
  measured at the app.
- **KV pool block map and PREFIX REUSE panels:** they need engine data (§8).
  Until then they show "arrives with the engine telemetry update".
- **RESIDENT MODELS** works as designed.

**Models:**
- the library table: MODEL, SIZE, CTX MAX, SINGLE-STREAM (measured from real
  single-reply rates), LAST USED, and STATE (SERVING, ON DISK, CAN'T RUN or
  DOWNLOADING);
- a residency bar comparing the estimate with available RAM;
- the right panel: facts, KV cost, chat template and system prompt, plus a
  per-model preset editor;
- the design's `stoke pull … FETCH` row becomes Discover: search your catalog
  and fetch.

**Requests:**
- filters ALL · CHAT · API · COMPLETIONS, and the request table;
- the right panel shows LIFECYCLE and CLIENT (source, stream, model,
  max_tokens);
- the design's PREFIX section says "no prefix cache in this engine version"
  until the engine has one (§8).

**Logs:**
- level filters over the engine's stderr and the app's own log, with a source
  column (engine, app, http or updater);
- the right panel holds the session counters and SAVE SESSION LOG.

**Settings** (new: it is in neither the design nor the demo, so review this
part). The same panel language: sections down the left, fields on the right.
- **Engine:** threads, context, parallel replies, KV mode (`q8` by default, or
  `f32`), and the memory estimate.
- **Local API:** on or off, and the port.
- **Models:** watched folders, Ollama import on or off, the downloads folder.
- **Updates:** auto, notify or off; stable or beta; check now.
- **Data:** export chats as Markdown or JSON, back up now, restore.
- **App:** launch at login, close to tray, notifications.
- **About:** the version and the third-party licences, listed at build time.

**Status bar:** session time, requests, cancelled, errors, and the app and
engine versions.

**Behaviour:**
- a second launch focuses the running window;
- closing quits the app and frees the model's memory; hiding to the tray
  instead is a setting, off by default, as is launch at login;
- a notification appears when a reply finishes while the window is hidden;
- shortcuts: Ctrl+N new chat, Ctrl+K search, Esc stop, Enter send,
  Shift+Enter new line, Ctrl+1…6 switch tabs.

## 5. Failure handling

- **A model won't load:** its row turns CAN'T RUN with the engine's own message
  (from stderr), and the previous model is started again.
- **The engine crashes or hangs:** it restarts after 1, 5 and 30 s (§3.1). The
  chat keeps the partial reply, marked "interrupted". After three failures in
  10 minutes the app stops, shows the log, and offers a restart.
- **Not enough memory:** it estimates before loading (§3.1) and never starts a
  load that would leave less than 1 GB free. It offers a smaller context or
  fewer parallel replies instead.
- **The context overflows:** it trims and retries once (§3.2). A single message
  longer than the whole context is refused in the composer, with its token
  count.
- **The database:** `quick_check` runs at startup. On a failure the app offers
  to restore the latest backup. Writes are transactions, and WAL is on.
- **An update is bad or tampered with:** it is discarded and logged, the app
  keeps running, and the next check tries again.
- **A download fails:** it resumes. On a sha256 mismatch the file is deleted
  and an error is shown.
- **The local API's port is taken:** Settings says so. The engine uses port 0,
  so it never collides.
- **The app itself crashes** (a Rust panic): the backtrace goes to the log
  folder, the engine dies with the app (the job object), and the next startup
  runs the integrity checks.

## 6. Security and privacy

- **Nothing leaves the machine except updates.** There is no telemetry and no
  account. The only outside traffic is update checks and downloads from
  `updates.castagents.com`, and update checks can be turned off.
- **Loopback only:** the engine and the local API bind to `127.0.0.1`.
- **The UI:** no remote content, a strict CSP, and IPC limited by
  capabilities. Model output is rendered as Markdown without raw HTML. Links
  open in the default browser, and only when clicked.
- **Signatures:** updates and the catalog are verified against minisign public
  keys compiled into the app. The private keys stay offline.
- **Files:** models are read where they are. The app writes only to its own
  data, log and download folders.
- **No code-signing certificate in v1:** Windows SmartScreen warns once at
  install. That is acceptable for you and the testers (§12).

## 7. Performance budgets

These are measured, not assumed, and a miss is recorded.
- **Cold start** to a window: under 1.5 s, not counting the model load.
- **Idle memory** of the host plus WebView2: 200 MB at most, first measured in
  milestone 1.
- **UI CPU while streaming:** under 5% of one core, because redraws are
  throttled.
- **Model switch:** the GGUF is memory-mapped (`gguf.c` ~262), so the restart
  takes seconds. It is timed again in milestone 1.

**First measurements, 2026-09-25.** The frozen `build/win/stoke.exe` on this
PC, run by hand with `--ctx 4096 --seqs 2 --kv q8` and the default threads:

| what | Nexus (Llama-3.2-3B, Q4_K_M) | Qwen3-4B-Instruct-2507 (Q4_K_M) |
|---|---|---|
| load, banner printed | 0.61 s (file already cached) | 2.2 s |
| model switch (stop Nexus, Qwen3 serving) | 2.36 s in total | — |
| first token, short prompt | 0.48–0.68 s | 0.53 s cold, 0.21 s warm |
| decode, one reply | 20.0–21.1 tok/s | 16.6–16.7 tok/s |
| engine memory after use | 2.06 GB working set, 647 MB private | 2.44 GB working set, 676 MB private |
| Stop mid-reply | generation ended at once: 0.00 CPU-s in the next 3 s | — |

## 8. Known limits (the frozen engine)

- **A chat holds at most 4,096 tokens,** prompt and reply together (§3.1).
  Raising the cap is engine work after `m7`.
- **No KV reuse across requests.** A freed block loses its hash
  (`blockpool.c` ~184–197), and prefix lookup skips free blocks (~363). So
  every new message re-reads the whole chat, and the re-read slows down as the
  chat grows. Measured on 09-25 with Qwen3-4B, as the time before the first
  token:

  | chat so far | first token after |
  |---|---|
  | 1,107 tokens | 13.3 s (83 tokens/s) |
  | 2,196 tokens | 36.5 s (60 tokens/s) |
  | 3,318 tokens | 72.9 s (45 tokens/s) |

  Ollama keeps its cache and starts in about a second. The fix is an engine
  prefix cache after `m7`, and for this app it is the most valuable engine
  change there is. Until it lands, the CONTEXT panel says why a long chat is
  slow to start, and §14 asks for a default history budget.
- **One model at a time.** Switching restarts the engine.
- **Architectures:** `llama` and `qwen3` only; tensor types F32, Q8_0, Q4_K and
  Q6_K; no vision.
- **No telemetry endpoints.** The block map, prefix reuse and engine-side stall
  panels need `/internal/state` and `/internal/events`, as specified in §4 of
  the 08-18 note. That is engine work after `m7`.

## 9. Testing

- **Rust (`cargo test`):**
  - the GGUF header reader, on fixtures the tests write, and on your five real
    files behind an opt-in environment variable;
  - Modelfile and Ollama-manifest parsing;
  - the context budget and trimming, including the parser for the 400 message;
  - the SSE parser: chunk boundaries, pings, `[DONE]` and the `usage` chunk;
  - the store: version trees, search, backup and restore, and an integrity
    failure;
  - the supervisor's state machine, run against a **fake engine** binary that
    prints the banner and streams, and can crash, hang, refuse to load or load
    slowly.
- **Tamper tests, written to fail first:** for the updater and the catalog,
  flip a byte in the installer, in `latest.json` and in `catalog.json`; use
  the wrong key; drop the `.sig`.
- **Frontend (vitest + Testing Library):**
  - Markdown rendering never emits raw HTML;
  - streaming updates, version switching and the context gauge work;
  - every tab renders from fixture data.
- **End to end:** WebDriver (`tauri-driver` with Edge WebDriver) or Playwright
  over WebView2's CDP drives the real app against the fake engine. The flows:
  - send, stream, stop, regenerate and edit;
  - switch the model;
  - crash the engine and recover;
  - offer an update: one signed build served locally, and one tampered build.
- **The real-model smoke test** is opt-in and needs your OK, because it starts
  the engine. It loads Nexus, runs one chat round-trip, stops a reply midway,
  and switches to Qwen3.
- **Packaging:** install, upgrade and uninstall the NSIS installer on this PC,
  and check the engine sidecar is present and runs.
- **Standing rule:** no model server or other long-lived server is started
  without asking.

## 10. Repository and build

- **Branch:** `lorren-desktop`, cut from `m5-entry`, in its own git worktree.
  The app lives in `desktop/`: `desktop/src-tauri` (Rust) and `desktop/ui`
  (React). `src/` is not touched.
- **The engine sidecar** is built from the frozen source with static linking,
  because today's `stoke.exe` imports `libwinpthread-1.dll`. The output is
  Tauri's sidecar name,
  `desktop/src-tauri/binaries/lorren-engine-x86_64-pc-windows-msvc.exe`.
  - The link flag lives in the app branch's build script, never in `src/`.
  - At build time, the sidecar's banner is checked against the frozen commit.
- **Toolchain:**
  - already on this PC: Node 24; MSVC 14.44 Build Tools and Windows SDK
    10.0.26100; MinGW g++ 15.2 for the engine;
  - still to install: Rust through rustup (with your OK, §12).
- **Licence:** AGPL-3.0, like the engine (decided 09-23). The bundled fonts
  are OFL. Third-party licences are listed in About.

## 11. Milestones

The first implementation plan covers milestone 1 only. Each later milestone
gets its own plan, written after you have used the one before it.

1. **The working demo.** You verify it before anything else is built.
   - The installer runs and Lorren opens: Control Room plus Chat, dark.
   - Models lists Nexus (with its Modelfile prompt), Qwen3-4B and Llama-3.2-3B,
     and marks `qwen2.5vl:3b` CAN'T RUN.
   - A chat with Nexus streams, stops and regenerates.
   - Switching the chat to Qwen3 visibly restarts the engine.
   - Killing the engine in Task Manager brings it back, and the chat carries
     on.
   - After closing and reopening, the chats are still there.
   - Requests and Logs show the traffic, in basic form. Milestone 3 completes
     them.
   - A signed 0.0.2 from a local test feed installs on restart, and a tampered
     one is refused.
   - Measured and shown: the model-switch time, time to first token, and the
     memory of the app plus the engine.
   - The demo's engine runs never overlap the 24 h soak.
2. **The complete chat:** search, pin, export, edit and versions, attachments,
   automatic titles, presets, context trimming and shortcuts.
3. **The complete console:** Swarm (the app's view), Requests, Logs, Settings,
   the tray, notifications and launch at login.
4. **Distribution:**
   - the R2 bucket and `updates.castagents.com`;
   - the release and catalog scripts;
   - Discover and downloads;
   - the real signing keys;
   - the packaging tests.
5. **After `m7`**, as engine work with its own specs: the telemetry endpoints
   for the block-map, prefix and stall panels; the prefix cache; then the
   universal server (Claude and GPT) plugging into this same app.

## 12. Blocked on Kabir

- An OK to install Rust (rustup, about 1 GB).
- An OK to run the engine for the demo and the opt-in smoke test. These are
  model servers, and the standing rule is to ask first.
- Cloudflare: create the R2 bucket and attach `updates.castagents.com`.
- Generate the two minisign key pairs, store them offline, and hand over the
  public keys.
- Renew castagents.com (it expires 2026-11-25; the updater depends on it) and
  castagent.com (2026-11-07).
- Later: a code-signing certificate, which removes the SmartScreen warning. v1
  ships without one.

## 13. Out of scope for this version

- Claude, GPT and routing: the universal-server spec.
- Access from other PCs (LAN plus an API key).
- Images and vision, voice, document search (RAG), plugins and tools.
- macOS and Linux builds. Tauri keeps them possible.
- Several models loaded at once.
- A light theme.
- Any engine change, including the telemetry endpoints and the prefix cache.
  Those are separate engine specs after `m7`.

## 14. Open questions for the implementation plan

- Which Tauri and plugin versions to pin; WebDriver or Playwright over CDP for
  the end-to-end tests.
- Whether the sidecar is built by the app's build script (the engine Makefile
  with a static-link flag) or checked in per release as a prebuilt binary.
- The local API's default port, and the cap on the request log (by count or by
  age).
- How the fake engine is built: a Rust test binary or a small script.
- The exact layout of Discover inside the Models tab.
- Where Settings' sections sit, if Kabir wants them somewhere other than a
  sixth tab.
- **A default history budget (Kabir to decide).** Sending only the most recent
  ~1,500 tokens of a chat would keep the wait under ~20 s (§8), but the model
  would forget more of the chat. The alternative is to send everything up to
  the 4,096 cap and accept waits of a minute or more. My recommendation is
  1,500 tokens, adjustable in Settings, until the prefix cache lands.

## 15. Where this came from

- **09-24:**
  - main use: chat;
  - the full product;
  - Electron picked, with part 1 (architecture and updates) presented.
- **09-25:**
  - part 1 approved;
  - native against Electron, then native against Tauri, then a live demo of the
    Control Room as a Tauri-style web view and as native C++;
  - Tauri chosen;
  - the Control Room plus the Chat tab approved;
  - part 2 approved.
- **Design sources:**
  - `design/ControlRoom.dc.html`;
  - `superpowers/design-notes/2026-08-18-the-surface-web-first.md`;
  - `design/lorren-console-demo.html` (the demo's web version, committed with
    this spec).
- **Engine facts** were read from `server.c`, `stoke.c`, `blockpool.c`,
  `gguf.c` and `main.c` at `5fa0385` (`src/` at `b1ee6b3`).
