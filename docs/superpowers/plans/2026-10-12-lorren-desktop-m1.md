# Lorren desktop — milestone 1 (the working demo) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Tauri 2 desktop app, Lorren, that shows the Control Room console with a working Chat tab on top of the frozen `stoke serve` engine: chat with your local models, switch models, survive an engine crash, keep chats across restarts, and take a signed update from a local test feed while refusing a tampered one.

**Architecture:** A Rust host (Tauri 2) owns every service: an engine supervisor that runs `lorren-engine.exe` (the frozen engine, statically linked) as a child process inside a Windows job object; a chat service that streams replies over SSE and saves them to SQLite; a model library that reads GGUF headers, Modelfiles and Ollama manifests; a monitor that keeps request records and logs; and the updater plugin. The UI is React + TypeScript in WebView2 and talks to the host only through typed Tauri commands and channels. Nothing in the engine's `src/` changes.

**Tech Stack:** Tauri 2 (Rust, WebView2), tokio, reqwest (streaming), rusqlite (bundled SQLite), ts-rs (TypeScript bindings), the `windows` crate (job objects, memory), React 19 + TypeScript + Vite, react-markdown + remark-gfm + remark-math + rehype-katex + rehype-highlight, Vitest + Testing Library, playwright-core over WebView2's CDP, `tauri-plugin-updater` (minisign/ed25519), `tauri-plugin-single-instance`, `tauri-plugin-process`.

**Spec:** `docs/superpowers/specs/2026-09-25-lorren-desktop-design.md` (approved 2026-09-25; commits `12d754a`, `203d75d`, `f543c7f`; the 2026-10-12 row "Lorren v0.1 is this app", `3e633d4`). This plan implements its §11 milestone 1 only. UI reference: `design/lorren-console-demo.html`.

**Written:** Tasks 1–12 on 2026-09-25 (that session ended before it finished; the text was recovered from its transcript on 2026-10-12 with both of its later edits applied). Tasks 13–17 and this header's updates on 2026-10-12. The updater facts in Task 15 were read from `tauri-plugin-updater`'s source that day.

## Global Constraints

- `src/` is frozen until tag `m7` (`docs/superpowers/memos/2026-09-18-m7-code-freeze.md`). No task edits `src/`, `tests/` or the engine `Makefile`. The sidecar is built from `src/` exactly as committed at `b1ee6b3`.
- All work happens on branch `lorren-desktop` in its own worktree (`M:\New Drive\Desktop\Programming\stoke-lorren`), cut from `m5-entry`. The app lives in `desktop/` (standard Tauri layout: web UI at `desktop/`, Rust at `desktop/src-tauri/`).
- Product name `Lorren`, identifier `com.castagents.lorren`, wordmark `lorren`, version `0.0.1` (the update test publishes `0.0.2`).
- Windows 11 x64 only. Rust through rustup with the MSVC toolchain (MSVC 14.44 Build Tools + Windows SDK 10.0.26100 are already installed). Node 24. The engine builds with MinGW g++ 15.2 from `C:/msys64/ucrt64/bin`, via `mingw32-make` run from Git Bash, never PowerShell.
- Engine launch: `lorren-engine.exe serve <model.gguf> --host 127.0.0.1 --port 0 --ctx 4096 --seqs 2 --kv q8` (+ `--threads N` only when set). Context is capped at 4,096 by the engine (`STOKE_M2_MAX_CTX`); never pass more.
- Every chat request sends `stream: true`, `stream_options.include_usage: true` and an explicit `max_tokens` (the engine's default is 128).
- Sampling parameters the engine accepts: `temperature` (0–100), `top_p` (0–1), `top_k`, `repeat_penalty`, `seed`, `stop` (≤4), `max_tokens`. Never send a non-zero `frequency_penalty` or `presence_penalty`.
- Defaults: default model = the first runnable file whose name starts with `Qwen3-4B-Instruct-2507-Q4_K_M`; history budget = 1,500 tokens; reply cap = 1,024 tokens; temperature 0.7, top_p 0.9, top_k 40, repeat_penalty 1.1.
- Engine and all app sockets bind `127.0.0.1` only. No telemetry. No remote content in the UI. Model output is rendered as Markdown with raw HTML disabled.
- Dark only: colours, fonts and layout follow `design/ControlRoom.dc.html` / `design/lorren-console-demo.html` (Archivo + JetBrains Mono, bundled under the OFL).
- Standing rule: no model server or other long-lived server is started without Kabir's OK. He gave it for this milestone's real-model steps (2026-09-25), and he names the time.
- **The M7 soak (four separate 6 h runs, Kabir 2026-10-12) owns the PC while a run is in progress:** no real-engine run, no `cargo build`/`tauri build` and no e2e run during one. Check first: `MSYS_NO_PATHCONV=1 wsl.exe -- pgrep -fa run_m7_soak.sh` must print nothing. (The soak runs inside WSL, so `tasklist` cannot see it.) Unit tests and plan or doc work are fine.
- The updater's public key in `tauri.conf.json` is a TEST key (Task 15) until milestone 4 swaps in Kabir's offline production key. `tauri.updater-test.conf.json` (an `http://127.0.0.1` feed, `dangerousInsecureTransportProtocol`) is for test builds only and must never reach a release.
- The version the app shows and the updater compares is `app.package_info().version` (from `tauri.conf.json`, which a 0.0.2 build overrides), never `env!("CARGO_PKG_VERSION")`.
- The model library's default folders are this PC's (Task 4) until Settings arrives in milestone 3.
- No `Co-Authored-By` or "Generated with" trailers in commits. Never push unless Kabir says so.

## Review Focus

1. **Model paths with spaces** (every one of Kabir's models lives under `M:\New Drive\...`): the engine must receive the whole path as one argument. Pinned by a supervisor test that launches the fake engine with a path containing spaces and checks the argv it recorded (Task 8).
2. **A multi-byte character split across two SSE chunks** (₹, emoji, accented names in a reply): the stream must not produce U+FFFD or drop bytes. Pinned by an SSE parser test that splits `₹` and an emoji at every byte boundary (Task 9) and a chat test that streams them end to end (Task 10).
3. **Sending while the engine is switching or restarting:** the send must wait for the engine and then succeed, not fail with "connection refused". Pinned by a chat test that starts a model switch and a send together (Task 10).
4. **Closing or killing the app while a reply streams:** on the next start the half-written reply must show as interrupted, never as still streaming, and no orphan engine may survive. Pinned by a store test (streaming → interrupted on open, Task 5) and a job-object test (Task 8).
5. **One message longer than the whole context** (a pasted document): the app must refuse it with its token estimate instead of sending a request the engine rejects. Pinned by a budget test (Task 9) and a chat test (Task 10).

---

## File structure

```
desktop/                                 (web UI root = Tauri project root)
  package.json, tsconfig.json, vite.config.ts, index.html
  scripts/
    build-engine.sh        static sidecar from the frozen src/ (Task 2)
    make-icon.py           source icon for `tauri icon` (Task 1)
    fetch-fonts.py         Archivo + JetBrains Mono + OFL texts (Task 12)
    update-feed.py         test keys, signed 0.0.1/0.0.2 installers, the four feeds, a local server (Task 15)
  .keys/                   the TEST signing keys (git-ignored, Task 15)
  e2e/run-demo-e2e.mjs     milestone checklist against the fake engine (Task 16)
  src/
    main.tsx, App.tsx, theme.css, fonts.css, fonts/, format.ts
    ipc/bindings/          generated by ts-rs (Task 11) — never hand-edited
    ipc/api.ts             typed wrappers over invoke/Channel/listen (Task 12; update calls in Task 15)
    components/TopBar.tsx, StatusBar.tsx, Placeholder.tsx (Task 12), UpdateBadge.tsx (Task 15)
    chat/stream.ts, ChatTab.tsx, ChatList.tsx, MessageView.tsx, Markdown.tsx, Composer.tsx, ReplyPanel.tsx, chat.css (Task 13)
    tabs/ModelsTab.tsx, RequestsTab.tsx, LogsTab.tsx, tabs.css (Task 14)
    __tests__/*.test.ts(x), testApi.ts
  src-tauri/
    Cargo.toml, build.rs, tauri.conf.json, capabilities/default.json, icons/, .cargo/config.toml
    tauri.updater-test.conf.json   the test feed overlay, never in a release (Task 15)
    binaries/lorren-engine-x86_64-pc-windows-msvc.exe   (built by Task 2, git-ignored)
    src/
      main.rs              calls lorren_lib::run()
      lib.rs               Tauri builder, state, plugins, startup (Task 11)
      paths.rs             data/log dirs, sidecar path, env overrides (Task 11)
      gguf.rs              GGUF header reader, facts, runnable check, fixture writer (Task 3)
      library.rs           folders, Modelfile, Ollama, default model (Task 4)
      store.rs             SQLite: chats, message tree, statuses (Task 5)
      monitor.rs           request records, log ring, log file (Task 6)
      bin/fake_engine.rs   test double of `stoke serve` (Task 7, feature `fake-engine`)
      engine/mod.rs, engine/supervisor.rs, engine/job.rs, engine/memory.rs (Task 8)
      sse.rs, budget.rs    pure stream + budget logic (Task 9)
      chat.rs              send / stop / regenerate (Task 10)
      commands.rs          Tauri commands + DTOs (Task 11)
      updater.rs           check, download, install (Task 15)
    tests/supervisor.rs, tests/chat_flow.rs  (integration, fake engine)
```

## Tasks

1. Worktree, toolchain and the Tauri skeleton
2. The engine sidecar (static, frozen source)
3. GGUF header reader
4. Model library
5. Store
6. Monitor
7. Fake engine
8. Engine supervisor
9. SSE parser and budget
10. Chat service
11. Commands, bindings and app wiring
12. UI foundation: theme, fonts, top bar, status bar
13. UI: the Chat tab
14. UI: Models, Requests and Logs
15. Updater and the test feed
16. End-to-end run against the fake engine
17. Installer and the real-model demo (with Kabir)

---

### Task 1: Worktree, toolchain and the Tauri skeleton

**Files:**
- Create: `desktop/package.json`, `desktop/tsconfig.json`, `desktop/vite.config.ts`, `desktop/index.html`, `desktop/src/main.tsx`, `desktop/src/App.tsx`, `desktop/src/__tests__/App.test.tsx`, `desktop/.gitignore`
- Create: `desktop/src-tauri/Cargo.toml`, `desktop/src-tauri/build.rs`, `desktop/src-tauri/tauri.conf.json`, `desktop/src-tauri/capabilities/default.json`, `desktop/src-tauri/src/main.rs`, `desktop/src-tauri/src/lib.rs`
- Create: `desktop/scripts/make-icon.py`, `desktop/src-tauri/icons/*` (generated)

**Interfaces:**
- Consumes: nothing.
- Produces: the crate `lorren` with library name `lorren_lib` and `pub fn run()`; the npm scripts `dev`, `build`, `test`, `tauri`; `App` exported from `desktop/src/App.tsx`.

- [ ] **Step 1: Create the worktree**

Use superpowers:using-git-worktrees. Concretely:

```bash
cd "/m/New Drive/Desktop/Programming/stoke"
git worktree add -b lorren-desktop "/m/New Drive/Desktop/Programming/stoke-lorren" m5-entry
cd "/m/New Drive/Desktop/Programming/stoke-lorren"
git log -1 --format='%h %s'
git diff --quiet b1ee6b3 -- src/ && echo "src/ is the frozen source"
```

Expected: a commit at or after `f543c7f`, then `src/ is the frozen source`.

- [ ] **Step 2: Install Rust (approved by Kabir 2026-09-25)**

```powershell
Invoke-WebRequest https://win.rustup.rs/x86_64 -OutFile "$env:TEMP\rustup-init.exe"
& "$env:TEMP\rustup-init.exe" -y --default-toolchain stable --profile minimal
& "$env:USERPROFILE\.cargo\bin\rustc.exe" --version
& "$env:USERPROFILE\.cargo\bin\cargo.exe" --version
```

Expected: two version lines, host `x86_64-pc-windows-msvc`. New shells get `%USERPROFILE%\.cargo\bin` on PATH; in the current one, prepend it.

- [ ] **Step 3: Write the failing UI test**

`desktop/src/__tests__/App.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "../App";

describe("App", () => {
  it("shows the lorren wordmark", () => {
    render(<App />);
    expect(screen.getByText("lorren")).toBeTruthy();
  });
});
```

- [ ] **Step 4: Create the web project**

`desktop/package.json`:

```json
{
  "name": "lorren-desktop",
  "private": true,
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "test": "vitest run",
    "tauri": "tauri",
    "e2e": "node e2e/run-demo-e2e.mjs"
  }
}
```

`desktop/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noEmit": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "resolveJsonModule": true,
    "types": ["vite/client"]
  },
  "include": ["src"]
}
```

`desktop/vite.config.ts`:

```ts
/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 5173, strictPort: true },
  build: { target: "es2022", outDir: "dist" },
  test: { environment: "jsdom" },
});
```

`desktop/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Lorren</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`desktop/src/main.tsx`:

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

`desktop/.gitignore`:

```
node_modules/
dist/
src-tauri/target/
src-tauri/binaries/*.exe
src-tauri/gen/
.keys/
e2e/.work/
```

Install the dependencies:

```bash
cd "/m/New Drive/Desktop/Programming/stoke-lorren/desktop"
npm install react react-dom @tauri-apps/api
npm install -D typescript vite @vitejs/plugin-react vitest jsdom @testing-library/react @types/react @types/react-dom @tauri-apps/cli
```

- [ ] **Step 5: Run the test to see it fail**

Run: `npx vitest run`
Expected: FAIL — `Failed to resolve import "../App"`.

- [ ] **Step 6: Add the minimal App**

`desktop/src/App.tsx`:

```tsx
export function App() {
  return <span className="wordmark">lorren</span>;
}
```

Run: `npx vitest run` — Expected: PASS (1 test).

- [ ] **Step 7: Create the Rust side**

`desktop/src-tauri/Cargo.toml`:

```toml
[package]
name = "lorren"
version = "0.0.1"
description = "Lorren desktop app"
edition = "2021"
default-run = "lorren"

[lib]
name = "lorren_lib"
crate-type = ["staticlib", "cdylib", "rlib"]

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
tauri = { version = "2", features = [] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

`desktop/src-tauri/build.rs`:

```rust
fn main() {
    tauri_build::build()
}
```

`desktop/src-tauri/src/main.rs`:

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    lorren_lib::run()
}
```

`desktop/src-tauri/src/lib.rs`:

```rust
#[tauri::command]
fn ping() -> String {
    "pong".into()
}

pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![ping])
        .run(tauri::generate_context!())
        .expect("error while running Lorren");
}

#[cfg(test)]
mod tests {
    #[test]
    fn ping_answers_pong() {
        assert_eq!(super::ping(), "pong");
    }
}
```

`desktop/src-tauri/tauri.conf.json`:

```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "Lorren",
  "version": "0.0.1",
  "identifier": "com.castagents.lorren",
  "build": {
    "beforeDevCommand": "npm run dev",
    "devUrl": "http://localhost:5173",
    "beforeBuildCommand": "npm run build",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      {
        "label": "main",
        "title": "Lorren",
        "width": 1440,
        "height": 900,
        "minWidth": 1100,
        "minHeight": 680
      }
    ],
    "security": {
      "csp": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; font-src 'self' data:; connect-src ipc: http://ipc.localhost"
    }
  },
  "bundle": {
    "active": true,
    "targets": ["nsis"],
    "icon": ["icons/32x32.png", "icons/128x128.png", "icons/128x128@2x.png", "icons/icon.ico"],
    "windows": { "nsis": { "installMode": "currentUser" } }
  }
}
```

`desktop/src-tauri/capabilities/default.json`:

```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "default",
  "description": "The main Lorren window",
  "windows": ["main"],
  "permissions": ["core:default"]
}
```

- [ ] **Step 8: Generate the icons**

`desktop/scripts/make-icon.py`:

```python
"""Writes the 1024x1024 source icon for `tauri icon`: an amber square 'l' on the console's ground."""
from PIL import Image, ImageDraw

SIZE = 1024
GROUND = (9, 11, 14)      # oklch(0.15 0.006 255)
ACCENT = (214, 167, 113)  # oklch(0.76 0.09 70)
img = Image.new("RGB", (SIZE, SIZE), GROUND)
d = ImageDraw.Draw(img)
d.rectangle([96, 96, SIZE - 96, SIZE - 96], outline=ACCENT, width=40)
d.rectangle([420, 250, 540, 700], fill=ACCENT)   # the stem of the 'l'
d.rectangle([420, 640, 640, 760], fill=ACCENT)   # its foot
img.save("scripts/icon-1024.png")
print("wrote scripts/icon-1024.png")
```

```bash
cd "/m/New Drive/Desktop/Programming/stoke-lorren/desktop"
python scripts/make-icon.py
npx tauri icon scripts/icon-1024.png
ls src-tauri/icons | head
```

Expected: `32x32.png`, `128x128.png`, `128x128@2x.png`, `icon.ico` among the files.

- [ ] **Step 9: Run both test suites and the dev window**

```bash
cd "/m/New Drive/Desktop/Programming/stoke-lorren/desktop"
npx vitest run
cd src-tauri && cargo test && cd ..
npx tauri dev
```

Expected: Vitest PASS (1), `cargo test` PASS (`ping_answers_pong`), and a window titled "Lorren" showing `lorren`. Close it.

- [ ] **Step 10: Commit**

```bash
cd "/m/New Drive/Desktop/Programming/stoke-lorren"
git add desktop
git commit -m "desktop: Tauri 2 skeleton for Lorren (milestone 1, task 1)"
```

---

### Task 2: The engine sidecar (static, frozen source)

**Files:**
- Create: `desktop/scripts/build-engine.sh`
- Modify: `desktop/src-tauri/tauri.conf.json` (add `bundle.externalBin`)

**Interfaces:**
- Consumes: the repo's `Makefile` (unchanged) with `BUILDDIR` and `LDFLAGS` set on the command line; `src/` at `b1ee6b3`.
- Produces: `desktop/src-tauri/binaries/lorren-engine-x86_64-pc-windows-msvc.exe`, which Tauri copies next to `lorren.exe` as `lorren-engine.exe` in dev and bundled builds.

- [ ] **Step 1: Write the check that must fail first**

Nothing exists yet, so the check is the script's own guard. Run:

```bash
cd "/m/New Drive/Desktop/Programming/stoke-lorren/desktop"
test -f src-tauri/binaries/lorren-engine-x86_64-pc-windows-msvc.exe && echo present || echo "missing (expected)"
```

Expected: `missing (expected)`.

- [ ] **Step 2: Write the build script**

`desktop/scripts/build-engine.sh`:

```sh
#!/usr/bin/env sh
# Builds lorren-engine.exe: the FROZEN engine (src/ exactly as at b1ee6b3), statically linked.
# Run from Git Bash (the Makefile needs sh + mkdir -p). Never edits src/ or the Makefile:
# BUILDDIR and LDFLAGS are command-line overrides.
set -eu
FREEZE=b1ee6b3
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
if ! git diff --quiet "$FREEZE" -- src/; then
    echo "build-engine: src/ differs from the frozen $FREEZE -- refusing" >&2
    exit 1
fi
PATH="/c/msys64/ucrt64/bin:$PATH" mingw32-make -j4 BUILDDIR=build/lorren-engine LDFLAGS=-static build/lorren-engine/stoke.exe
OUT="$ROOT/desktop/src-tauri/binaries/lorren-engine-x86_64-pc-windows-msvc.exe"
mkdir -p "$(dirname "$OUT")"
cp build/lorren-engine/stoke.exe "$OUT"
if PATH="/c/msys64/ucrt64/bin:$PATH" objdump -p "$OUT" | grep -qiE 'DLL Name: (libwinpthread|libgcc|libstdc)'; then
    echo "build-engine: still imports a MinGW runtime DLL -- not static" >&2
    exit 1
fi
case "$(file -b "$OUT")" in
    PE32+*) ;;
    *) echo "build-engine: $OUT is not a PE32+ executable" >&2; exit 1 ;;
esac
echo "build-engine: OK $OUT"
```

- [ ] **Step 3: Build it**

```bash
cd "/m/New Drive/Desktop/Programming/stoke-lorren/desktop"
sh scripts/build-engine.sh
```

Expected: the last line is `build-engine: OK .../lorren-engine-x86_64-pc-windows-msvc.exe`.

- [ ] **Step 4: Prove it is the same engine as the frozen build**

`stoke info` loads only the header and exits (a one-shot tool, allowed without asking). Compare the static sidecar with the frozen `build/win/stoke.exe` of the main tree:

```bash
M="/m/New Drive/Desktop/Programming/stoke/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
./src-tauri/binaries/lorren-engine-x86_64-pc-windows-msvc.exe info "$M" > /tmp/lorren-info.txt
"/m/New Drive/Desktop/Programming/stoke/build/win/stoke.exe" info "$M" > /tmp/frozen-info.txt
cmp /tmp/lorren-info.txt /tmp/frozen-info.txt && echo "identical info output"
head -2 /tmp/lorren-info.txt
```

Expected: `identical info output`, then `== metadata (...) ==` and `general.architecture (STRING) = qwen3`.

- [ ] **Step 5: Register the sidecar with Tauri**

In `desktop/src-tauri/tauri.conf.json`, add to `"bundle"`:

```json
    "externalBin": ["binaries/lorren-engine"],
```

Run: `cd src-tauri && cargo build && ls target/debug/lorren-engine.exe`
Expected: the file exists (Tauri copies the sidecar next to the app binary without the target triple).

- [ ] **Step 6: Commit**

```bash
cd "/m/New Drive/Desktop/Programming/stoke-lorren"
git add desktop/scripts/build-engine.sh desktop/src-tauri/tauri.conf.json
git commit -m "desktop: build the frozen engine as a static lorren-engine sidecar (task 2)"
```

---

### Task 3: GGUF header reader

**Files:**
- Create: `desktop/src-tauri/src/gguf.rs`
- Modify: `desktop/src-tauri/src/lib.rs` (add `pub mod gguf;`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub const ENGINE_MAX_CTX: u32 = 4096`
  - `pub fn read_header(path: &Path) -> io::Result<Header>` — never panics on a bad file
  - `pub fn facts(h: &Header) -> Result<Facts, String>`
  - `pub fn check_runnable(h: &Header, f: &Facts) -> Result<(), String>` — `Err` carries the user-facing reason
  - `pub fn file_type_name(ft: u32) -> &'static str`, `pub fn ggml_type_name(t: u32) -> &'static str`
  - `Facts { arch, name, size_label, file_type, n_layer, n_head, n_head_kv, head_dim, vocab, ctx_file, template }` with `engine_ctx()` and `kv_bytes_per_token_q8()`
  - `Template::{ChatMl, Llama3, None}` with `as_str()` → `"chatml" | "llama3" | "none"`
  - `pub mod fixture { pub struct Spec; Spec::llama(name), Spec::qwen3(name); pub fn write(path, &Spec) }` — used by later tests and by the fake engine

- [ ] **Step 1: Write the failing tests**

Create `desktop/src-tauri/src/gguf.rs` containing only the test module for now, and add `pub mod gguf;` to `lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::fixture::{self, Spec};
    use super::*;
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("lorren-gguf-{}-{}", std::process::id(), name))
    }

    #[test]
    fn reads_a_llama_fixture() {
        let p = tmp("llama.gguf");
        fixture::write(&p, &Spec::llama("Nexus_Model")).unwrap();
        let h = read_header(&p).unwrap();
        let f = facts(&h).unwrap();
        assert_eq!(f.arch, "llama");
        assert_eq!(f.name.as_deref(), Some("Nexus_Model"));
        assert_eq!((f.n_layer, f.n_head, f.n_head_kv, f.head_dim), (28, 24, 8, 128));
        assert_eq!(f.template, Template::Llama3);
        assert_eq!(f.ctx_file, 131_072);
        assert_eq!(f.engine_ctx(), 4096);
        assert_eq!(f.kv_bytes_per_token_q8(), 60_928); // 59.5 KiB, spec §3.1
        assert!(check_runnable(&h, &f).is_ok());
    }

    #[test]
    fn reads_a_qwen3_fixture() {
        let p = tmp("qwen3.gguf");
        fixture::write(&p, &Spec::qwen3("Qwen3-4B")).unwrap();
        let h = read_header(&p).unwrap();
        let f = facts(&h).unwrap();
        assert_eq!(f.template, Template::ChatMl);
        assert_eq!(f.kv_bytes_per_token_q8(), 78_336); // 76.5 KiB, spec §3.1
        assert!(check_runnable(&h, &f).is_ok());
    }

    #[test]
    fn a_vision_architecture_is_refused_with_its_name() {
        let p = tmp("vl.gguf");
        let spec = Spec { arch: "qwen2vl", ..Spec::qwen3("qwen2.5vl") };
        fixture::write(&p, &spec).unwrap();
        let h = read_header(&p).unwrap();
        let err = check_runnable(&h, &facts(&h).unwrap()).unwrap_err();
        assert!(err.contains("qwen2vl"), "{err}");
    }

    #[test]
    fn an_unsupported_tensor_type_is_refused() {
        let p = tmp("q5k.gguf");
        let spec = Spec { tensor_types: &[0, 13], ..Spec::llama("x") };
        fixture::write(&p, &spec).unwrap();
        let h = read_header(&p).unwrap();
        let err = check_runnable(&h, &facts(&h).unwrap()).unwrap_err();
        assert!(err.contains("Q5_K"), "{err}");
    }

    #[test]
    fn missing_template_tokens_are_refused() {
        let p = tmp("notpl.gguf");
        let spec = Spec { tokens: &["hello", "world"], ..Spec::llama("x") };
        fixture::write(&p, &spec).unwrap();
        let h = read_header(&p).unwrap();
        let err = check_runnable(&h, &facts(&h).unwrap()).unwrap_err();
        assert!(err.contains("chat template"), "{err}");
    }

    #[test]
    fn a_truncated_file_is_an_error_not_a_panic() {
        let p = tmp("cut.gguf");
        fixture::write(&p, &Spec::llama("x")).unwrap();
        let len = std::fs::metadata(&p).unwrap().len();
        std::fs::OpenOptions::new().write(true).open(&p).unwrap().set_len(len / 2).unwrap();
        assert!(read_header(&p).is_err());
    }

    #[test]
    fn a_file_that_is_not_gguf_is_an_error() {
        let p = tmp("junk.gguf");
        std::fs::write(&p, b"GGML this is not a gguf file").unwrap();
        assert!(read_header(&p).is_err());
    }

    /// Kabir's real files. Run with: cargo test real_models -- --ignored
    #[test]
    #[ignore]
    fn real_models() {
        let nexus = std::path::Path::new("M:/New Drive/ai_models/nexus.gguf");
        let h = read_header(nexus).unwrap();
        let f = facts(&h).unwrap();
        assert_eq!((f.arch.as_str(), f.n_layer, f.n_head_kv), ("llama", 28, 8));
        assert_eq!(f.template, Template::Llama3);
        assert_eq!(f.kv_bytes_per_token_q8(), 60_928);
        assert!(check_runnable(&h, &f).is_ok());
        let qwen = std::path::Path::new(
            "M:/New Drive/Desktop/Programming/stoke/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        );
        let h = read_header(qwen).unwrap();
        let f = facts(&h).unwrap();
        assert_eq!((f.arch.as_str(), f.n_layer), ("qwen3", 36));
        assert_eq!(f.template, Template::ChatMl);
        assert!(check_runnable(&h, &f).is_ok());
    }
}
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `cd desktop/src-tauri && cargo test gguf`
Expected: compile errors — `read_header`, `facts`, `fixture` not found.

- [ ] **Step 3: Implement the reader, the facts and the fixture writer**

Put this above the test module in `gguf.rs`:

```rust
//! Reads a GGUF file's header -- metadata and tensor table -- without touching the
//! weights, and decides whether the frozen engine (src/ at b1ee6b3) can run it (spec §3.6).
//! The engine would stop with an error on an unsupported file anyway; this says why first.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// The engine clamps every model to this context (STOKE_M2_MAX_CTX, model.c:43).
pub const ENGINE_MAX_CTX: u32 = 4096;

const SUPPORTED_ARCH: [&str; 2] = ["llama", "qwen3"];
/// ggml tensor type ids the engine loads: F32, Q8_0, Q4_K, Q6_K.
const SUPPORTED_TYPES: [u32; 4] = [0, 8, 12, 14];
/// String arrays worth keeping; every other array is skipped (merges alone are ~280k strings).
const KEEP_ARRAYS: [&str; 2] = ["tokenizer.ggml.tokens", "general.tags"];

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    StrArray(Vec<String>),
    /// An array we did not keep: its length only.
    Array(u64),
}

#[derive(Debug, Clone)]
pub struct Header {
    pub version: u32,
    pub meta: HashMap<String, Value>,
    pub tensor_types: Vec<u32>,
}

impl Header {
    pub fn str(&self, k: &str) -> Option<&str> {
        match self.meta.get(k) {
            Some(Value::Str(s)) => Some(s),
            _ => None,
        }
    }
    pub fn uint(&self, k: &str) -> Option<u64> {
        match self.meta.get(k) {
            Some(Value::U64(v)) => Some(*v),
            Some(Value::I64(v)) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn tokens(&self) -> Option<&[String]> {
        match self.meta.get("tokenizer.ggml.tokens") {
            Some(Value::StrArray(v)) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    ChatMl,
    Llama3,
    None,
}

impl Template {
    pub fn as_str(self) -> &'static str {
        match self {
            Template::ChatMl => "chatml",
            Template::Llama3 => "llama3",
            Template::None => "none",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Facts {
    pub arch: String,
    pub name: Option<String>,
    pub size_label: Option<String>,
    pub file_type: Option<u32>,
    pub n_layer: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub ctx_file: u32,
    pub template: Template,
}

impl Facts {
    pub fn engine_ctx(&self) -> u32 {
        self.ctx_file.min(ENGINE_MAX_CTX)
    }
    /// KV bytes per token at `--kv q8`: layers x (K+V) x KV heads x head dim x 34/32 (Q8_0 blocks).
    pub fn kv_bytes_per_token_q8(&self) -> u64 {
        self.n_layer as u64 * 2 * self.n_head_kv as u64 * self.head_dim as u64 * 34 / 32
    }
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

struct Rd<T: Read> {
    r: T,
}

impl<T: Read> Rd<T> {
    fn bytes<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b)?;
        Ok(b)
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes::<4>()?))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes::<8>()?))
    }
    fn string(&mut self) -> io::Result<String> {
        let n = self.u64()?;
        if n > 1 << 24 {
            return Err(bad("string longer than 16 MiB"));
        }
        let mut b = vec![0u8; n as usize];
        self.r.read_exact(&mut b)?;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }
    fn skip(&mut self, n: u64) -> io::Result<()> {
        let copied = io::copy(&mut (&mut self.r).take(n), &mut io::sink())?;
        if copied != n {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated"));
        }
        Ok(())
    }
    fn scalar(&mut self, t: u32) -> io::Result<Value> {
        Ok(match t {
            0 => Value::U64(self.bytes::<1>()?[0] as u64),
            1 => Value::I64(self.bytes::<1>()?[0] as i8 as i64),
            2 => Value::U64(u16::from_le_bytes(self.bytes::<2>()?) as u64),
            3 => Value::I64(i16::from_le_bytes(self.bytes::<2>()?) as i64),
            4 => Value::U64(self.u32()? as u64),
            5 => Value::I64(self.u32()? as i32 as i64),
            6 => Value::F64(f32::from_bits(self.u32()?) as f64),
            7 => Value::Bool(self.bytes::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => return Err(bad(&format!("unknown metadata type {t}"))),
        })
    }
}

fn scalar_size(t: u32) -> Option<u64> {
    match t {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4 | 5 | 6 => Some(4),
        10 | 11 | 12 => Some(8),
        _ => None,
    }
}

pub fn read_header(path: &Path) -> io::Result<Header> {
    let mut r = Rd { r: BufReader::with_capacity(1 << 20, File::open(path)?) };
    if &r.bytes::<4>()? != b"GGUF" {
        return Err(bad("not a GGUF file"));
    }
    let version = r.u32()?;
    if version < 2 {
        return Err(bad("GGUF v1 is not supported"));
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_kv > 1 << 20 || n_tensors > 1 << 24 {
        return Err(bad("implausible header counts"));
    }
    let mut meta = HashMap::new();
    for _ in 0..n_kv {
        let key = r.string()?;
        let t = r.u32()?;
        let value = if t == 9 {
            let et = r.u32()?;
            let n = r.u64()?;
            if et == 8 {
                if KEEP_ARRAYS.contains(&key.as_str()) {
                    let mut v = Vec::with_capacity(n.min(1 << 20) as usize);
                    for _ in 0..n {
                        v.push(r.string()?);
                    }
                    Value::StrArray(v)
                } else {
                    for _ in 0..n {
                        let len = r.u64()?;
                        r.skip(len)?;
                    }
                    Value::Array(n)
                }
            } else {
                let size = scalar_size(et).ok_or_else(|| bad("nested or unknown array type"))?;
                r.skip(size * n)?;
                Value::Array(n)
            }
        } else {
            r.scalar(t)?
        };
        meta.insert(key, value);
    }
    let mut tensor_types = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let _name = r.string()?;
        let n_dims = r.u32()?;
        if n_dims > 8 {
            return Err(bad("tensor with more than 8 dimensions"));
        }
        for _ in 0..n_dims {
            r.u64()?;
        }
        tensor_types.push(r.u32()?);
        r.u64()?; // data offset
    }
    Ok(Header { version, meta, tensor_types })
}

pub fn facts(h: &Header) -> Result<Facts, String> {
    let arch = h.str("general.architecture").ok_or("no general.architecture")?.to_string();
    let key = |k: &str| format!("{arch}.{k}");
    let need = |k: &str| h.uint(&key(k)).map(|v| v as u32).ok_or_else(|| format!("missing {}", key(k)));
    let n_layer = need("block_count")?;
    let n_head = need("attention.head_count")?;
    let n_head_kv = h.uint(&key("attention.head_count_kv")).map(|v| v as u32).unwrap_or(n_head);
    let emb = need("embedding_length")?;
    let head_dim = h
        .uint(&key("attention.key_length"))
        .map(|v| v as u32)
        .unwrap_or(if n_head > 0 { emb / n_head } else { 0 });
    let ctx_file = need("context_length")?;
    let tokens = h.tokens();
    let vocab = tokens
        .map(|t| t.len() as u32)
        .or_else(|| h.uint(&key("vocab_size")).map(|v| v as u32))
        .unwrap_or(0);
    let has = |s: &str| tokens.map_or(false, |t| t.iter().any(|x| x == s));
    let template = if has("<|im_start|>") && has("<|im_end|>") {
        Template::ChatMl
    } else if has("<|start_header_id|>") && has("<|end_header_id|>") && has("<|eot_id|>") {
        Template::Llama3
    } else {
        Template::None
    };
    Ok(Facts {
        name: h.str("general.name").map(String::from),
        size_label: h.str("general.size_label").map(String::from),
        file_type: h.uint("general.file_type").map(|v| v as u32),
        n_layer,
        n_head,
        n_head_kv,
        head_dim,
        vocab,
        ctx_file,
        template,
        arch,
    })
}

pub fn check_runnable(h: &Header, f: &Facts) -> Result<(), String> {
    if !SUPPORTED_ARCH.contains(&f.arch.as_str()) {
        return Err(format!(
            "architecture '{}' is not supported (the engine runs llama and qwen3)",
            f.arch
        ));
    }
    if let Some(t) = h.tensor_types.iter().find(|t| !SUPPORTED_TYPES.contains(t)) {
        return Err(format!(
            "tensor type {} is not supported (F32, Q8_0, Q4_K and Q6_K only)",
            ggml_type_name(*t)
        ));
    }
    if f.template == Template::None {
        return Err("no chat template tokens (needs ChatML or Llama-3 header tokens)".into());
    }
    Ok(())
}

pub fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1",
        8 => "Q8_0", 9 => "Q8_1", 10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K",
        14 => "Q6_K", 15 => "Q8_K", 30 => "BF16", _ => "an unknown type",
    }
}

pub fn file_type_name(ft: u32) -> &'static str {
    match ft {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 7 => "Q8_0", 10 => "Q2_K", 12 => "Q3_K_M",
        14 => "Q4_K_S", 15 => "Q4_K_M", 16 => "Q5_K_S", 17 => "Q5_K_M", 18 => "Q6_K",
        _ => "other",
    }
}

/// Writes small but valid GGUF v3 headers for tests and the fake engine. No weights.
pub mod fixture {
    use super::*;

    pub const LLAMA3_TOKENS: [&str; 5] =
        ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "hello"];
    pub const CHATML_TOKENS: [&str; 4] = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "hello"];

    pub struct Spec<'a> {
        pub arch: &'a str,
        pub name: &'a str,
        pub n_layer: u32,
        pub n_head: u32,
        pub n_head_kv: u32,
        pub emb: u32,
        pub ctx: u32,
        pub file_type: u32,
        pub tokens: &'a [&'a str],
        pub tensor_types: &'a [u32],
    }

    impl<'a> Spec<'a> {
        pub fn llama(name: &'a str) -> Self {
            Spec { arch: "llama", name, n_layer: 28, n_head: 24, n_head_kv: 8, emb: 3072,
                   ctx: 131_072, file_type: 15, tokens: &LLAMA3_TOKENS, tensor_types: &[0, 12, 14] }
        }
        pub fn qwen3(name: &'a str) -> Self {
            Spec { arch: "qwen3", name, n_layer: 36, n_head: 32, n_head_kv: 8, emb: 2560,
                   ctx: 262_144, file_type: 15, tokens: &CHATML_TOKENS, tensor_types: &[0, 12, 14] }
        }
    }

    enum Kv {
        U32(u32),
        Str(String),
        Strs(Vec<String>),
    }

    fn wstr(w: &mut impl Write, s: &str) -> io::Result<()> {
        w.write_all(&(s.len() as u64).to_le_bytes())?;
        w.write_all(s.as_bytes())
    }

    pub fn write(path: &Path, s: &Spec) -> io::Result<()> {
        let a = s.arch;
        let kv = vec![
            ("general.architecture".to_string(), Kv::Str(a.to_string())),
            ("general.name".to_string(), Kv::Str(s.name.to_string())),
            ("general.file_type".to_string(), Kv::U32(s.file_type)),
            (format!("{a}.block_count"), Kv::U32(s.n_layer)),
            (format!("{a}.attention.head_count"), Kv::U32(s.n_head)),
            (format!("{a}.attention.head_count_kv"), Kv::U32(s.n_head_kv)),
            (format!("{a}.embedding_length"), Kv::U32(s.emb)),
            (format!("{a}.attention.key_length"), Kv::U32(128)),
            (format!("{a}.context_length"), Kv::U32(s.ctx)),
            ("tokenizer.ggml.tokens".to_string(), Kv::Strs(s.tokens.iter().map(|t| t.to_string()).collect())),
            ("tokenizer.ggml.merges".to_string(), Kv::Strs(vec!["h e".into(), "l l".into()])),
        ];
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"GGUF")?;
        w.write_all(&3u32.to_le_bytes())?;
        w.write_all(&(s.tensor_types.len() as u64).to_le_bytes())?;
        w.write_all(&(kv.len() as u64).to_le_bytes())?;
        for (k, v) in &kv {
            wstr(&mut w, k)?;
            match v {
                Kv::U32(x) => {
                    w.write_all(&4u32.to_le_bytes())?;
                    w.write_all(&x.to_le_bytes())?;
                }
                Kv::Str(x) => {
                    w.write_all(&8u32.to_le_bytes())?;
                    wstr(&mut w, x)?;
                }
                Kv::Strs(xs) => {
                    w.write_all(&9u32.to_le_bytes())?;
                    w.write_all(&8u32.to_le_bytes())?;
                    w.write_all(&(xs.len() as u64).to_le_bytes())?;
                    for x in xs {
                        wstr(&mut w, x)?;
                    }
                }
            }
        }
        for (i, t) in s.tensor_types.iter().enumerate() {
            wstr(&mut w, &format!("blk.{i}.weight"))?;
            w.write_all(&1u32.to_le_bytes())?; // one dimension
            w.write_all(&256u64.to_le_bytes())?;
            w.write_all(&t.to_le_bytes())?;
            w.write_all(&0u64.to_le_bytes())?; // data offset (no data)
        }
        w.flush()
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test gguf`
Expected: 7 passed, 1 ignored.

Run: `cargo test real_models -- --ignored`
Expected: PASS (reads Kabir's `nexus.gguf` and the Qwen3-4B file).

- [ ] **Step 5: Commit**

```bash
git add desktop/src-tauri/src/gguf.rs desktop/src-tauri/src/lib.rs
git commit -m "desktop: GGUF header reader and the engine's runnable check (task 3)"
```

---

### Task 4: Model library

**Files:**
- Create: `desktop/src-tauri/src/library.rs`, `desktop/src-tauri/.cargo/config.toml`
- Modify: `desktop/src-tauri/src/lib.rs` (add `pub mod library;`), `desktop/src-tauri/Cargo.toml` (ts-rs)

**Interfaces:**
- Consumes: `gguf::{read_header, facts, check_runnable, file_type_name, Template}` and `gguf::fixture` (Task 3).
- Produces:
  - `ModelInfo` (serde + ts-rs; fields listed in Step 3), `ModelInfo.id` = `"file:<lower-case path>"` or `"ollama:<name>:<tag>"`
  - `pub struct LibraryConfig { folders, ollama_models_dir, default_prefix }` with `LibraryConfig::from_env(downloads: PathBuf)`
  - `pub fn scan(cfg: &LibraryConfig) -> Vec<ModelInfo>` — runnable models first, exactly one `is_default` when anything is runnable
  - `pub fn parse_modelfile(text: &str) -> Modelfile { from, system, params }`

- [ ] **Step 1: Add ts-rs and its export directory**

```bash
cd desktop/src-tauri
cargo add ts-rs
```

`desktop/src-tauri/.cargo/config.toml`:

```toml
# ts-rs writes TypeScript bindings here when `cargo test` runs (the export_bindings_* tests).
[env]
TS_RS_EXPORT_DIR = { value = "../src/ipc/bindings", relative = true }
```

- [ ] **Step 2: Write the failing tests**

Create `desktop/src-tauri/src/library.rs` with only the tests, and add `pub mod library;` to `lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::fixture::{self, Spec};
    use std::fs;

    /// A library laid out like Kabir's: a folder with nexus.gguf + its Modelfile, a folder
    /// with Qwen3, a vision model, and an Ollama store. The root has a space in it on purpose.
    fn layout(tag: &str) -> (PathBuf, LibraryConfig) {
        let root = std::env::temp_dir().join(format!("lorren lib {}-{}", std::process::id(), tag));
        let _ = fs::remove_dir_all(&root);
        let ai = root.join("ai models");
        let stoke = root.join("stoke models");
        let ollama = root.join("ollama");
        fs::create_dir_all(&ai).unwrap();
        fs::create_dir_all(&stoke).unwrap();
        fixture::write(&ai.join("nexus.gguf"), &Spec::llama("Nexus_Model")).unwrap();
        fs::write(
            ai.join("Modelfile"),
            "FROM C:/Users/Kabir/ai_models/nexus.gguf\nSYSTEM \"You are Nexus AI, a professional quantitative trader. You give concise, decisive trading verdicts.\"\n",
        )
        .unwrap();
        fixture::write(&stoke.join("Qwen3-4B-Instruct-2507-Q4_K_M.gguf"), &Spec::qwen3("Qwen3-4B")).unwrap();
        fixture::write(&stoke.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"), &Spec::llama("Llama-3.2-3B")).unwrap();
        let man = ollama.join("manifests/registry.ollama.ai/library/qwen2.5vl");
        fs::create_dir_all(&man).unwrap();
        fs::create_dir_all(ollama.join("blobs")).unwrap();
        fixture::write(&ollama.join("blobs/sha256-aaa"), &Spec { arch: "qwen2vl", ..Spec::qwen3("qwen2.5vl") }).unwrap();
        fs::write(ollama.join("blobs/sha256-bbb"), "You are a helpful vision assistant.").unwrap();
        fs::write(
            man.join("3b"),
            r#"{"schemaVersion":2,"layers":[
                {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:aaa","size":1},
                {"mediaType":"application/vnd.ollama.image.system","digest":"sha256:bbb","size":1}]}"#,
        )
        .unwrap();
        let cfg = LibraryConfig {
            folders: vec![ai, stoke, root.join("missing folder")],
            ollama_models_dir: Some(ollama),
            default_prefix: "qwen3-4b-instruct-2507-q4_k_m".into(),
        };
        (root, cfg)
    }

    fn by_name<'a>(v: &'a [ModelInfo], name: &str) -> &'a ModelInfo {
        v.iter().find(|m| m.name == name).unwrap_or_else(|| panic!("no model {name} in {v:?}"))
    }

    #[test]
    fn nexus_gets_its_system_prompt_from_the_modelfile_matched_by_file_name() {
        let (_root, cfg) = layout("nexus");
        let models = scan(&cfg);
        let nexus = by_name(&models, "nexus");
        assert!(nexus.runnable, "{:?}", nexus.reason);
        assert_eq!(nexus.system_source.as_deref(), Some("Modelfile"));
        assert!(nexus.system_prompt.as_deref().unwrap().starts_with("You are Nexus AI"));
        assert!(nexus.path.contains("lorren lib"), "the path keeps its spaces: {}", nexus.path);
        assert_eq!(nexus.template, "llama3");
        assert_eq!(nexus.ctx_engine, 4096);
    }

    #[test]
    fn qwen3_4b_is_the_default_and_only_one_model_is() {
        let (_root, cfg) = layout("default");
        let models = scan(&cfg);
        let defaults: Vec<_> = models.iter().filter(|m| m.is_default).collect();
        assert_eq!(defaults.len(), 1);
        assert_eq!(defaults[0].name, "Qwen3-4B-Instruct-2507-Q4_K_M");
    }

    #[test]
    fn the_ollama_vision_model_is_listed_but_cannot_run() {
        let (_root, cfg) = layout("ollama");
        let models = scan(&cfg);
        let vl = by_name(&models, "qwen2.5vl:3b");
        assert_eq!(vl.id, "ollama:qwen2.5vl:3b");
        assert!(!vl.runnable);
        assert!(vl.reason.as_deref().unwrap().contains("qwen2vl"));
        assert_eq!(vl.system_source.as_deref(), Some("Ollama"));
        assert!(models.first().unwrap().runnable, "runnable models come first");
    }

    #[test]
    fn a_missing_folder_is_skipped_quietly() {
        let cfg = LibraryConfig {
            folders: vec![PathBuf::from("Z:/does/not/exist")],
            ollama_models_dir: None,
            default_prefix: "x".into(),
        };
        assert!(scan(&cfg).is_empty());
    }

    #[test]
    fn modelfile_parsing_handles_quotes_triple_quotes_and_parameters() {
        let m = parse_modelfile("from ./a.gguf\nSYSTEM \"\"\"line one\nline two\"\"\"\nPARAMETER temperature 0.4\nPARAMETER stop <|eot_id|>\n");
        assert_eq!(m.from.as_deref(), Some("./a.gguf"));
        assert_eq!(m.system.as_deref(), Some("line one\nline two"));
        assert_eq!(m.params, vec![("temperature".into(), "0.4".into()), ("stop".into(), "<|eot_id|>".into())]);
        let plain = parse_modelfile("FROM x.gguf\nSYSTEM be brief\n");
        assert_eq!(plain.system.as_deref(), Some("be brief"));
    }
}
```

Run: `cargo test library`
Expected: compile errors — `scan`, `LibraryConfig`, `ModelInfo`, `parse_modelfile` not found.

- [ ] **Step 3: Implement the library**

Above the tests in `library.rs`:

```rust
//! The model library (spec §3.6): GGUF files in watched folders, Ollama's manifests, and a
//! Modelfile sidecar next to a GGUF. Every entry says whether the engine can run it and why not.

use crate::gguf;
use serde::Serialize;
use std::path::{Path, PathBuf};
use ts_rs::TS;

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub path: String,
    /// "folder" or "ollama"
    pub source: String,
    #[ts(type = "number")]
    pub size_bytes: u64,
    pub arch: String,
    pub quant: String,
    pub size_label: String,
    pub n_layer: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub ctx_file: u32,
    pub ctx_engine: u32,
    #[ts(type = "number")]
    pub kv_bytes_per_token: u64,
    /// "chatml", "llama3" or "none"
    pub template: String,
    pub runnable: bool,
    pub reason: Option<String>,
    pub system_prompt: Option<String>,
    /// "Modelfile" or "Ollama"
    pub system_source: Option<String>,
    pub is_default: bool,
}

pub struct LibraryConfig {
    pub folders: Vec<PathBuf>,
    pub ollama_models_dir: Option<PathBuf>,
    /// Lower-case file-name prefix of the default model (spec §2: Qwen3-4B-Instruct-2507).
    pub default_prefix: String,
}

impl LibraryConfig {
    /// Spec §3.6 defaults. `LORREN_MODEL_DIRS` (semicolon-separated) and `LORREN_OLLAMA_DIR`
    /// (`none` disables Ollama) override them for tests and the end-to-end run.
    pub fn from_env(downloads: PathBuf) -> Self {
        let folders = match std::env::var("LORREN_MODEL_DIRS") {
            Ok(v) => v.split(';').filter(|s| !s.trim().is_empty()).map(PathBuf::from).collect(),
            Err(_) => vec![
                PathBuf::from(r"M:\New Drive\ai_models"),
                PathBuf::from(r"M:\New Drive\Desktop\Programming\stoke\models"),
                downloads,
            ],
        };
        let ollama_models_dir = match std::env::var("LORREN_OLLAMA_DIR") {
            Ok(v) if v == "none" => None,
            Ok(v) => Some(PathBuf::from(v)),
            Err(_) => std::env::var("OLLAMA_MODELS").ok().map(PathBuf::from).or_else(|| {
                std::env::var("USERPROFILE").ok().map(|h| PathBuf::from(h).join(".ollama").join("models"))
            }),
        };
        LibraryConfig { folders, ollama_models_dir, default_prefix: "qwen3-4b-instruct-2507-q4_k_m".into() }
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct Modelfile {
    pub from: Option<String>,
    pub system: Option<String>,
    pub params: Vec<(String, String)>,
}

pub fn parse_modelfile(text: &str) -> Modelfile {
    let mut m = Modelfile::default();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let Some((kw, rest)) = line.trim().split_once(char::is_whitespace) else { continue };
        let rest = rest.trim();
        match kw.to_ascii_uppercase().as_str() {
            "FROM" => m.from = Some(rest.to_string()),
            "SYSTEM" => {
                let value = if let Some(body) = rest.strip_prefix("\"\"\"") {
                    if let Some(end) = body.find("\"\"\"") {
                        body[..end].to_string()
                    } else {
                        let mut acc = body.to_string();
                        for l in lines.by_ref() {
                            acc.push('\n');
                            if let Some(end) = l.find("\"\"\"") {
                                acc.push_str(&l[..end]);
                                break;
                            }
                            acc.push_str(l);
                        }
                        acc
                    }
                } else if rest.len() >= 2 && rest.starts_with('"') && rest.ends_with('"') {
                    rest[1..rest.len() - 1].to_string()
                } else {
                    rest.to_string()
                };
                m.system = Some(value.trim().to_string());
            }
            "PARAMETER" => {
                if let Some((k, v)) = rest.split_once(char::is_whitespace) {
                    m.params.push((k.to_string(), v.trim().to_string()));
                }
            }
            _ => {}
        }
    }
    m
}

fn file_name_lower(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_lowercase()
}

pub fn scan(cfg: &LibraryConfig) -> Vec<ModelInfo> {
    let mut out = Vec::new();
    for dir in &cfg.folders {
        scan_folder(dir, &mut out);
    }
    if let Some(root) = &cfg.ollama_models_dir {
        scan_ollama(root, &mut out);
    }
    out.sort_by(|a, b| {
        b.runnable.cmp(&a.runnable).then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    out.dedup_by(|a, b| a.id == b.id);
    let default = out
        .iter()
        .position(|m| m.runnable && file_name_lower(&m.path).starts_with(&cfg.default_prefix))
        .or_else(|| out.iter().position(|m| m.runnable));
    if let Some(i) = default {
        out[i].is_default = true;
    }
    out
}

fn scan_folder(dir: &Path, out: &mut Vec<ModelInfo>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut ggufs = Vec::new();
    let mut modelfiles = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if name.to_lowercase().ends_with(".gguf") && p.is_file() {
            ggufs.push(p);
        } else if name == "Modelfile" || name.to_lowercase().ends_with(".modelfile") {
            if let Ok(text) = std::fs::read_to_string(&p) {
                modelfiles.push(parse_modelfile(&text));
            }
        }
    }
    for p in ggufs {
        let own = file_name_lower(&p.to_string_lossy());
        let system = modelfiles
            .iter()
            .find(|m| m.from.as_deref().map_or(false, |f| file_name_lower(f) == own))
            .and_then(|m| m.system.clone());
        let source = system.as_ref().map(|_| "Modelfile".to_string());
        let name = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let id = format!("file:{}", p.to_string_lossy().to_lowercase());
        out.push(describe(&p, "folder", id, name, system, source));
    }
}

fn scan_ollama(root: &Path, out: &mut Vec<ModelInfo>) {
    let library = root.join("manifests").join("registry.ollama.ai").join("library");
    let Ok(names) = std::fs::read_dir(&library) else { return };
    for n in names.flatten() {
        let Ok(tags) = std::fs::read_dir(n.path()) else { continue };
        for t in tags.flatten() {
            let Ok(text) = std::fs::read_to_string(t.path()) else { continue };
            let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&text) else { continue };
            let layers = manifest["layers"].as_array().cloned().unwrap_or_default();
            let blob = |media: &str| {
                layers
                    .iter()
                    .find(|l| l["mediaType"] == media)
                    .and_then(|l| l["digest"].as_str())
                    .map(|d| root.join("blobs").join(d.replace(':', "-")))
            };
            let Some(model) = blob("application/vnd.ollama.image.model") else { continue };
            if !model.is_file() {
                continue;
            }
            let system = blob("application/vnd.ollama.image.system")
                .and_then(|p| std::fs::read_to_string(p).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let name = format!("{}:{}", n.file_name().to_string_lossy(), t.file_name().to_string_lossy());
            let source = system.as_ref().map(|_| "Ollama".to_string());
            out.push(describe(&model, "ollama", format!("ollama:{name}"), name, system, source));
        }
    }
}

fn describe(
    path: &Path,
    source: &str,
    id: String,
    name: String,
    system_prompt: Option<String>,
    system_source: Option<String>,
) -> ModelInfo {
    let mut m = ModelInfo {
        id,
        name,
        path: path.to_string_lossy().to_string(),
        source: source.into(),
        size_bytes: std::fs::metadata(path).map(|x| x.len()).unwrap_or(0),
        arch: String::new(),
        quant: String::new(),
        size_label: String::new(),
        n_layer: 0,
        n_head: 0,
        n_head_kv: 0,
        head_dim: 0,
        vocab: 0,
        ctx_file: 0,
        ctx_engine: 0,
        kv_bytes_per_token: 0,
        template: "none".into(),
        runnable: false,
        reason: None,
        system_prompt,
        system_source,
        is_default: false,
    };
    match gguf::read_header(path) {
        Err(e) => m.reason = Some(format!("cannot read the file header: {e}")),
        Ok(h) => match gguf::facts(&h) {
            Err(e) => m.reason = Some(e),
            Ok(f) => {
                m.arch = f.arch.clone();
                m.quant = f.file_type.map(gguf::file_type_name).unwrap_or("other").into();
                m.size_label = f.size_label.clone().unwrap_or_default();
                m.n_layer = f.n_layer;
                m.n_head = f.n_head;
                m.n_head_kv = f.n_head_kv;
                m.head_dim = f.head_dim;
                m.vocab = f.vocab;
                m.ctx_file = f.ctx_file;
                m.ctx_engine = f.engine_ctx();
                m.kv_bytes_per_token = f.kv_bytes_per_token_q8();
                m.template = f.template.as_str().into();
                match gguf::check_runnable(&h, &f) {
                    Ok(()) => m.runnable = true,
                    Err(e) => m.reason = Some(e),
                }
            }
        },
    }
    m
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test library`
Expected: 5 passed. `cargo test` also writes `desktop/src/ipc/bindings/ModelInfo.ts`.

- [ ] **Step 5: Commit**

```bash
git add desktop/src-tauri desktop/src/ipc/bindings
git commit -m "desktop: model library -- folders, Modelfile sidecar, Ollama import (task 4)"
```

---

### Task 5: Store

**Files:**
- Create: `desktop/src-tauri/src/store.rs`
- Modify: `desktop/src-tauri/src/lib.rs` (add `pub mod store;`), `Cargo.toml` (rusqlite, uuid)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub fn now_ms() -> i64`
  - `Role::{User, Assistant}`, `MsgStatus::{Done, Streaming, Stopped, Interrupted, Error}` (serde snake_case, ts-rs)
  - `Chat { id, title, model_id, system_prompt, active_leaf_id, created_ms, updated_ms }`
  - `MsgStats { prompt_tokens, completion_tokens, ttft_ms, total_ms, error }`, `Message { id, chat_id, parent_id, role, content, status, model_id, stats, created_ms }`
  - `Store::open(path)`, `Store::open_in_memory()`, `create_chat(model_id, system_prompt)`, `get_chat(id)`, `list_chats()`, `set_chat_model(id, model_id)`, `append_message(chat_id, parent_id, role, content, status, model_id) -> Message` (also moves the chat's active leaf, and titles a new chat from its first user message), `get_message(id)`, `update_content(id, content)`, `finish_message(id, status, &MsgStats)`, `active_path(chat_id) -> Vec<Message>` (root first), `siblings(message_id) -> Vec<Message>` (oldest first), `set_active_leaf(chat_id, message_id)`

- [ ] **Step 1: Add the crates**

```bash
cd desktop/src-tauri
cargo add rusqlite --features bundled
cargo add uuid --features v4
```

- [ ] **Step 2: Write the failing tests**

Create `desktop/src-tauri/src/store.rs` with the tests and add `pub mod store;` to `lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn chat_with_reply(s: &Store) -> (Chat, Message, Message) {
        let chat = s.create_chat("file:q", None).unwrap();
        let user = s.append_message(&chat.id, None, Role::User, "BTC just broke 68,400. Long or wait?", MsgStatus::Done, None).unwrap();
        let reply = s.append_message(&chat.id, Some(&user.id), Role::Assistant, "Verdict: HOLD.", MsgStatus::Done, Some("file:q")).unwrap();
        (s.get_chat(&chat.id).unwrap(), user, reply)
    }

    #[test]
    fn a_chat_keeps_its_path_and_is_titled_from_the_first_message() {
        let s = Store::open_in_memory().unwrap();
        let (chat, user, reply) = chat_with_reply(&s);
        assert_eq!(chat.title, "BTC just broke 68,400. Long or wait?");
        assert_eq!(chat.active_leaf_id.as_deref(), Some(reply.id.as_str()));
        let path = s.active_path(&chat.id).unwrap();
        assert_eq!(path.iter().map(|m| m.id.clone()).collect::<Vec<_>>(), vec![user.id, reply.id]);
    }

    #[test]
    fn a_long_first_message_gives_a_short_title() {
        let s = Store::open_in_memory().unwrap();
        let chat = s.create_chat("m", None).unwrap();
        s.append_message(&chat.id, None, Role::User, &"word ".repeat(40), MsgStatus::Done, None).unwrap();
        let title = s.get_chat(&chat.id).unwrap().title;
        assert!(title.chars().count() <= 41 && title.ends_with('…'), "{title}");
    }

    #[test]
    fn regenerating_adds_a_sibling_and_versions_can_be_switched() {
        let s = Store::open_in_memory().unwrap();
        let (chat, user, first) = chat_with_reply(&s);
        let second = s.append_message(&chat.id, Some(&user.id), Role::Assistant, "Verdict: SELL.", MsgStatus::Done, None).unwrap();
        let sibs = s.siblings(&second.id).unwrap();
        assert_eq!(sibs.iter().map(|m| m.id.clone()).collect::<Vec<_>>(), vec![first.id.clone(), second.id.clone()]);
        assert_eq!(s.active_path(&chat.id).unwrap().last().unwrap().id, second.id);
        s.set_active_leaf(&chat.id, &first.id).unwrap();
        assert_eq!(s.active_path(&chat.id).unwrap().last().unwrap().id, first.id);
    }

    #[test]
    fn streaming_content_and_final_stats_are_saved() {
        let s = Store::open_in_memory().unwrap();
        let chat = s.create_chat("m", None).unwrap();
        let m = s.append_message(&chat.id, None, Role::Assistant, "", MsgStatus::Streaming, None).unwrap();
        s.update_content(&m.id, "Buy at ₹68,400 🚀").unwrap();
        let stats = MsgStats { prompt_tokens: Some(69), completion_tokens: Some(6), ttft_ms: Some(640), total_ms: Some(890), error: None };
        s.finish_message(&m.id, MsgStatus::Done, &stats).unwrap();
        let got = s.get_message(&m.id).unwrap();
        assert_eq!(got.content, "Buy at ₹68,400 🚀");
        assert_eq!(got.status, MsgStatus::Done);
        assert_eq!(got.stats.completion_tokens, Some(6));
        assert_eq!(got.stats.ttft_ms, Some(640));
    }

    /// Review Focus 4: the app died while a reply was streaming.
    #[test]
    fn a_reply_left_streaming_is_interrupted_after_reopening() {
        let dir = std::env::temp_dir().join(format!("lorren-store-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lorren.db");
        let _ = std::fs::remove_file(&path);
        let (chat_id, msg_id) = {
            let s = Store::open(&path).unwrap();
            let chat = s.create_chat("m", None).unwrap();
            let m = s.append_message(&chat.id, None, Role::Assistant, "half a rep", MsgStatus::Streaming, None).unwrap();
            (chat.id, m.id)
        };
        let s = Store::open(&path).unwrap();
        let m = s.get_message(&msg_id).unwrap();
        assert_eq!(m.status, MsgStatus::Interrupted);
        assert_eq!(m.content, "half a rep");
        assert_eq!(s.list_chats().unwrap()[0].id, chat_id);
    }

    #[test]
    fn chats_are_listed_newest_first() {
        let s = Store::open_in_memory().unwrap();
        let a = s.create_chat("m", None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let b = s.create_chat("m", None).unwrap();
        let ids: Vec<_> = s.list_chats().unwrap().into_iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![b.id, a.id]);
    }
}
```

Run: `cargo test store`
Expected: compile errors — `Store`, `Role`, `MsgStatus`, `MsgStats` not found.

- [ ] **Step 3: Implement the store**

Above the tests in `store.rs`:

```rust
//! Chats and their message trees in SQLite (spec §3.5). Editing or regenerating adds a
//! sibling; a chat shows one path, from the root to its `active_leaf_id`.

use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::Serialize;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use ts_rs::TS;

pub fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
    fn parse(s: &str) -> Role {
        if s == "user" { Role::User } else { Role::Assistant }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum MsgStatus {
    Done,
    Streaming,
    Stopped,
    Interrupted,
    Error,
}

impl MsgStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            MsgStatus::Done => "done",
            MsgStatus::Streaming => "streaming",
            MsgStatus::Stopped => "stopped",
            MsgStatus::Interrupted => "interrupted",
            MsgStatus::Error => "error",
        }
    }
    fn parse(s: &str) -> MsgStatus {
        match s {
            "done" => MsgStatus::Done,
            "streaming" => MsgStatus::Streaming,
            "stopped" => MsgStatus::Stopped,
            "interrupted" => MsgStatus::Interrupted,
            _ => MsgStatus::Error,
        }
    }
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct Chat {
    pub id: String,
    pub title: String,
    pub model_id: String,
    pub system_prompt: Option<String>,
    pub active_leaf_id: Option<String>,
    #[ts(type = "number")]
    pub created_ms: i64,
    #[ts(type = "number")]
    pub updated_ms: i64,
}

#[derive(Debug, Clone, Default, Serialize, TS)]
#[ts(export)]
pub struct MsgStats {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    #[ts(type = "number | null")]
    pub ttft_ms: Option<u64>,
    #[ts(type = "number | null")]
    pub total_ms: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct Message {
    pub id: String,
    pub chat_id: String,
    pub parent_id: Option<String>,
    pub role: Role,
    pub content: String,
    pub status: MsgStatus,
    pub model_id: Option<String>,
    pub stats: MsgStats,
    #[ts(type = "number")]
    pub created_ms: i64,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS chats (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, model_id TEXT NOT NULL, system_prompt TEXT,
  active_leaf_id TEXT, created_ms INTEGER NOT NULL, updated_ms INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
  parent_id TEXT REFERENCES messages(id),
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL, status TEXT NOT NULL, model_id TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, ttft_ms INTEGER, total_ms INTEGER, error TEXT,
  created_ms INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS messages_by_chat ON messages(chat_id, created_ms);
CREATE INDEX IF NOT EXISTS messages_by_parent ON messages(parent_id);
";

const CHAT_COLS: &str = "id, title, model_id, system_prompt, active_leaf_id, created_ms, updated_ms";
const MSG_COLS: &str = "id, chat_id, parent_id, role, content, status, model_id, \
                        prompt_tokens, completion_tokens, ttft_ms, total_ms, error, created_ms";

fn chat_row(r: &Row) -> rusqlite::Result<Chat> {
    Ok(Chat {
        id: r.get(0)?,
        title: r.get(1)?,
        model_id: r.get(2)?,
        system_prompt: r.get(3)?,
        active_leaf_id: r.get(4)?,
        created_ms: r.get(5)?,
        updated_ms: r.get(6)?,
    })
}

fn msg_row(r: &Row) -> rusqlite::Result<Message> {
    let role: String = r.get(3)?;
    let status: String = r.get(5)?;
    Ok(Message {
        id: r.get(0)?,
        chat_id: r.get(1)?,
        parent_id: r.get(2)?,
        role: Role::parse(&role),
        content: r.get(4)?,
        status: MsgStatus::parse(&status),
        model_id: r.get(6)?,
        stats: MsgStats {
            prompt_tokens: r.get(7)?,
            completion_tokens: r.get(8)?,
            ttft_ms: r.get::<_, Option<i64>>(9)?.map(|v| v as u64),
            total_ms: r.get::<_, Option<i64>>(10)?.map(|v| v as u64),
            error: r.get(11)?,
        },
        created_ms: r.get(12)?,
    })
}

fn short_title(text: &str) -> String {
    let line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
    if line.chars().count() <= 40 {
        line.to_string()
    } else {
        format!("{}…", line.chars().take(40).collect::<String>().trim_end())
    }
}

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(path: &Path) -> rusqlite::Result<Store> {
        let conn = Connection::open(path)?;
        conn.pragma_update_and_check(None, "journal_mode", "WAL", |r| r.get::<_, String>(0))?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> rusqlite::Result<Store> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> rusqlite::Result<Store> {
        conn.pragma_update(None, "foreign_keys", "ON")?;
        let check: String = conn.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
        if check != "ok" {
            return Err(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CORRUPT),
                Some(format!("quick_check: {check}")),
            ));
        }
        conn.execute_batch(SCHEMA)?;
        conn.pragma_update(None, "user_version", 1)?;
        // A reply that was streaming when the app died is interrupted, never "still streaming".
        conn.execute("UPDATE messages SET status = 'interrupted' WHERE status = 'streaming'", [])?;
        Ok(Store { conn })
    }

    pub fn create_chat(&self, model_id: &str, system_prompt: Option<&str>) -> rusqlite::Result<Chat> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = now_ms();
        self.conn.execute(
            "INSERT INTO chats (id, title, model_id, system_prompt, active_leaf_id, created_ms, updated_ms) \
             VALUES (?1, 'New chat', ?2, ?3, NULL, ?4, ?4)",
            params![id, model_id, system_prompt, now],
        )?;
        self.get_chat(&id)
    }

    pub fn get_chat(&self, id: &str) -> rusqlite::Result<Chat> {
        self.conn.query_row(&format!("SELECT {CHAT_COLS} FROM chats WHERE id = ?1"), [id], chat_row)
    }

    pub fn list_chats(&self) -> rusqlite::Result<Vec<Chat>> {
        let mut st = self.conn.prepare(&format!("SELECT {CHAT_COLS} FROM chats ORDER BY updated_ms DESC, rowid DESC"))?;
        let rows = st.query_map([], chat_row)?;
        rows.collect()
    }

    pub fn set_chat_model(&self, id: &str, model_id: &str) -> rusqlite::Result<()> {
        self.conn.execute(
            "UPDATE chats SET model_id = ?2, updated_ms = ?3 WHERE id = ?1",
            params![id, model_id, now_ms()],
        )?;
        Ok(())
    }

    pub fn append_message(
        &self,
        chat_id: &str,
        parent_id: Option<&str>,
        role: Role,
        content: &str,
        status: MsgStatus,
        model_id: Option<&str>,
    ) -> rusqlite::Result<Message> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = now_ms();
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO messages (id, chat_id, parent_id, role, content, status, model_id, created_ms) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![id, chat_id, parent_id, role.as_str(), content, status.as_str(), model_id, now],
        )?;
        tx.execute(
            "UPDATE chats SET active_leaf_id = ?2, updated_ms = ?3 WHERE id = ?1",
            params![chat_id, id, now],
        )?;
        if role == Role::User {
            tx.execute(
                "UPDATE chats SET title = ?2 WHERE id = ?1 AND title = 'New chat'",
                params![chat_id, short_title(content)],
            )?;
        }
        tx.commit()?;
        self.get_message(&id)
    }

    pub fn get_message(&self, id: &str) -> rusqlite::Result<Message> {
        self.conn.query_row(&format!("SELECT {MSG_COLS} FROM messages WHERE id = ?1"), [id], msg_row)
    }

    pub fn update_content(&self, id: &str, content: &str) -> rusqlite::Result<()> {
        self.conn.execute("UPDATE messages SET content = ?2 WHERE id = ?1", params![id, content])?;
        Ok(())
    }

    pub fn finish_message(&self, id: &str, status: MsgStatus, s: &MsgStats) -> rusqlite::Result<()> {
        self.conn.execute(
            "UPDATE messages SET status = ?2, prompt_tokens = ?3, completion_tokens = ?4, \
             ttft_ms = ?5, total_ms = ?6, error = ?7 WHERE id = ?1",
            params![
                id,
                status.as_str(),
                s.prompt_tokens,
                s.completion_tokens,
                s.ttft_ms.map(|v| v as i64),
                s.total_ms.map(|v| v as i64),
                s.error
            ],
        )?;
        Ok(())
    }

    pub fn active_path(&self, chat_id: &str) -> rusqlite::Result<Vec<Message>> {
        let mut out = Vec::new();
        let mut cur = self.get_chat(chat_id)?.active_leaf_id;
        while let Some(id) = cur {
            let m = self.get_message(&id)?;
            cur = m.parent_id.clone();
            out.push(m);
        }
        out.reverse();
        Ok(out)
    }

    pub fn siblings(&self, message_id: &str) -> rusqlite::Result<Vec<Message>> {
        let m = self.get_message(message_id)?;
        let mut st = self.conn.prepare(&format!(
            "SELECT {MSG_COLS} FROM messages WHERE chat_id = ?1 AND parent_id IS ?2 AND role = ?3 \
             ORDER BY created_ms, rowid"
        ))?;
        let rows = st.query_map(params![m.chat_id, m.parent_id, m.role.as_str()], msg_row)?;
        rows.collect()
    }

    /// Shows `message_id`'s branch: follows its newest children down to a leaf.
    pub fn set_active_leaf(&self, chat_id: &str, message_id: &str) -> rusqlite::Result<()> {
        let mut leaf = message_id.to_string();
        loop {
            let next: Option<String> = self
                .conn
                .query_row(
                    "SELECT id FROM messages WHERE parent_id = ?1 ORDER BY created_ms DESC, rowid DESC LIMIT 1",
                    [&leaf],
                    |r| r.get(0),
                )
                .optional()?;
            match next {
                Some(n) => leaf = n,
                None => break,
            }
        }
        self.conn.execute(
            "UPDATE chats SET active_leaf_id = ?2, updated_ms = ?3 WHERE id = ?1",
            params![chat_id, leaf, now_ms()],
        )?;
        Ok(())
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test store`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add desktop/src-tauri desktop/src/ipc/bindings
git commit -m "desktop: SQLite store -- chats, message tree, interrupted-on-reopen (task 5)"
```

---

### Task 6: Monitor

**Files:**
- Create: `desktop/src-tauri/src/monitor.rs`
- Modify: `desktop/src-tauri/src/lib.rs` (add `pub mod monitor;`)

**Interfaces:**
- Consumes: `store::now_ms()`.
- Produces:
  - `pub const MAX_REQUESTS: usize = 500`, `pub const MAX_LOGS: usize = 5_000`
  - `LogLine { seq, time_ms, source, level, text }`, `RequestRecord { id, started_ms, source, endpoint, model, prompt_tokens, completion_tokens, ttft_ms, total_ms, tokens_per_s, status, dropped_turns }` (ts-rs)
  - `Monitor::new(log_file: Option<PathBuf>) -> Arc<Monitor>`; `log(&self, source, level, text)`; `logs_after(&self, seq) -> Vec<LogLine>`; `start_request(&self, source, endpoint, model) -> String` (id `req_xxxxxxxx`, status `"streaming"`); `update_request(&self, id, FnOnce(&mut RequestRecord))`; `requests(&self) -> Vec<RequestRecord>` (newest first)

- [ ] **Step 1: Write the failing tests**

Create `monitor.rs` with the tests and add `pub mod monitor;` to `lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logs_are_numbered_and_can_be_read_after_a_sequence_number() {
        let m = Monitor::new(None);
        m.log("app", "info", "one");
        m.log("engine", "warn", "two");
        m.log("app", "info", "three");
        let after = m.logs_after(1);
        assert_eq!(after.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(), vec!["two", "three"]);
        assert_eq!(after[0].source, "engine");
        assert_eq!(after[0].level, "warn");
    }

    #[test]
    fn the_log_ring_is_capped() {
        let m = Monitor::new(None);
        for i in 0..MAX_LOGS + 10 {
            m.log("app", "info", format!("line {i}"));
        }
        let all = m.logs_after(0);
        assert_eq!(all.len(), MAX_LOGS);
        assert_eq!(all[0].text, "line 10");
    }

    #[test]
    fn requests_are_listed_newest_first_and_can_be_updated() {
        let m = Monitor::new(None);
        let a = m.start_request("chat", "/v1/chat/completions", "nexus");
        let b = m.start_request("chat", "/v1/chat/completions", "qwen");
        m.update_request(&a, |r| {
            r.status = "ok".into();
            r.ttft_ms = Some(640);
        });
        let list = m.requests();
        assert_eq!(list[0].id, b);
        assert_eq!(list[1].status, "ok");
        assert_eq!(list[1].ttft_ms, Some(640));
        assert!(a.starts_with("req_") && a.len() == 12);
    }

    #[test]
    fn logs_also_go_to_the_file() {
        let path = std::env::temp_dir().join(format!("lorren-monitor-{}.log", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let m = Monitor::new(Some(path.clone()));
        m.log("app", "info", "written to disk");
        drop(m);
        assert!(std::fs::read_to_string(&path).unwrap().contains("written to disk"));
    }
}
```

Run: `cargo test monitor` — Expected: compile errors (`Monitor` not found).

- [ ] **Step 2: Implement the monitor**

Above the tests:

```rust
//! Request records and the log ring (spec §3.4; the Requests and Logs tabs). In memory and
//! capped; log lines also go to a plain file so a crash leaves a trail.

use crate::store::now_ms;
use serde::Serialize;
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use ts_rs::TS;

pub const MAX_REQUESTS: usize = 500;
pub const MAX_LOGS: usize = 5_000;

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct LogLine {
    #[ts(type = "number")]
    pub seq: u64,
    #[ts(type = "number")]
    pub time_ms: i64,
    /// "app", "engine" or "updater"
    pub source: String,
    /// "info", "warn" or "error"
    pub level: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct RequestRecord {
    pub id: String,
    #[ts(type = "number")]
    pub started_ms: i64,
    pub source: String,
    pub endpoint: String,
    pub model: String,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    #[ts(type = "number | null")]
    pub ttft_ms: Option<u64>,
    #[ts(type = "number | null")]
    pub total_ms: Option<u64>,
    pub tokens_per_s: Option<f32>,
    /// "streaming", "ok", "stopped", "interrupted" or "error"
    pub status: String,
    pub dropped_turns: u32,
}

pub struct Monitor {
    seq: AtomicU64,
    logs: Mutex<VecDeque<LogLine>>,
    requests: Mutex<VecDeque<RequestRecord>>,
    file: Option<Mutex<File>>,
}

impl Monitor {
    pub fn new(log_file: Option<PathBuf>) -> Arc<Monitor> {
        let file = log_file
            .and_then(|p| {
                if let Some(d) = p.parent() {
                    let _ = std::fs::create_dir_all(d);
                }
                OpenOptions::new().create(true).append(true).open(p).ok()
            })
            .map(Mutex::new);
        Arc::new(Monitor {
            seq: AtomicU64::new(0),
            logs: Mutex::new(VecDeque::new()),
            requests: Mutex::new(VecDeque::new()),
            file,
        })
    }

    pub fn log(&self, source: &str, level: &str, text: impl Into<String>) {
        let line = LogLine {
            seq: self.seq.fetch_add(1, Ordering::SeqCst) + 1,
            time_ms: now_ms(),
            source: source.into(),
            level: level.into(),
            text: text.into(),
        };
        if let Some(f) = &self.file {
            if let Ok(mut f) = f.lock() {
                let _ = writeln!(f, "{} {:<7} {:<5} {}", line.time_ms, line.source, line.level, line.text);
            }
        }
        let mut logs = self.logs.lock().unwrap();
        logs.push_back(line);
        while logs.len() > MAX_LOGS {
            logs.pop_front();
        }
    }

    pub fn logs_after(&self, seq: u64) -> Vec<LogLine> {
        self.logs.lock().unwrap().iter().filter(|l| l.seq > seq).cloned().collect()
    }

    pub fn start_request(&self, source: &str, endpoint: &str, model: &str) -> String {
        let id = format!("req_{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        let record = RequestRecord {
            id: id.clone(),
            started_ms: now_ms(),
            source: source.into(),
            endpoint: endpoint.into(),
            model: model.into(),
            prompt_tokens: None,
            completion_tokens: None,
            ttft_ms: None,
            total_ms: None,
            tokens_per_s: None,
            status: "streaming".into(),
            dropped_turns: 0,
        };
        let mut q = self.requests.lock().unwrap();
        q.push_back(record);
        while q.len() > MAX_REQUESTS {
            q.pop_front();
        }
        id
    }

    pub fn update_request(&self, id: &str, f: impl FnOnce(&mut RequestRecord)) {
        if let Some(r) = self.requests.lock().unwrap().iter_mut().rev().find(|r| r.id == id) {
            f(r);
        }
    }

    pub fn requests(&self) -> Vec<RequestRecord> {
        self.requests.lock().unwrap().iter().rev().cloned().collect()
    }
}
```

- [ ] **Step 3: Run the tests**

Run: `cargo test monitor` — Expected: 4 passed.

- [ ] **Step 4: Commit**

```bash
git add desktop/src-tauri desktop/src/ipc/bindings
git commit -m "desktop: monitor -- request records, capped log ring, log file (task 6)"
```

---

### Task 7: Fake engine

**Files:**
- Create: `desktop/src-tauri/src/bin/fake_engine.rs`, `desktop/src-tauri/tests/fake_engine.rs`
- Modify: `desktop/src-tauri/Cargo.toml` (feature + `[[bin]]`)

**Interfaces:**
- Consumes: `lorren_lib::gguf::fixture` (Task 3).
- Produces: the binary `fake-engine` (built only with `--features fake-engine`; integration tests reach it as `env!("CARGO_BIN_EXE_fake-engine")`). It mirrors `stoke serve`'s command line, banner, `/health`, `/v1/models`, streaming `/v1/chat/completions` (SSE `data:` chunks, `: ping` comments, a finish chunk, a usage chunk, `data: [DONE]`) and the 400 `context_length_exceeded` body, byte for byte in shape. Behaviour switches (environment variables):
  - `FAKE_ENGINE_MODE` = `ok` (default) | `refuse` | `hang` | `crash_after_ms:<N>`
  - `FAKE_ENGINE_CRASH_ONCE` = a file path: with `crash_after_ms`, crash only if the file does not exist yet (and create it)
  - `FAKE_ENGINE_LOAD_MS` delay before the banner; `FAKE_ENGINE_TOKEN_MS` delay between streamed words (default 20)
  - `FAKE_ENGINE_REPLY` the reply (default `Hello from the fake engine.`)
  - `FAKE_ENGINE_EVENTS` a file that gets one line per event: `argv <json>`, `request <json>`, `context_length_exceeded`, `cancelled`, `done`
  - Like the real engine: refuses `--ctx` above 4096 at startup, and answers 400 when prompt tokens (characters / 4) + `max_tokens` exceed `--ctx`.
- `fake-engine write-fixtures <dir>` writes the fixture library used by the end-to-end run (Task 16).

- [ ] **Step 1: Declare the binary**

Append to `desktop/src-tauri/Cargo.toml`:

```toml
[features]
fake-engine = []

[[bin]]
name = "fake-engine"
path = "src/bin/fake_engine.rs"
required-features = ["fake-engine"]
```

- [ ] **Step 2: Write the failing test**

`desktop/src-tauri/tests/fake_engine.rs`:

```rust
#![cfg(feature = "fake-engine")]
//! The fake engine is the ground every supervisor and chat test stands on, so it is tested first.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};

fn exe() -> &'static str {
    env!("CARGO_BIN_EXE_fake-engine")
}

#[test]
fn it_prints_the_engine_banner_and_answers_health() {
    let mut child = Command::new(exe())
        .args(["serve", "M:/New Drive/x.gguf", "--host", "127.0.0.1", "--port", "0", "--ctx", "4096"])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
    assert!(line.starts_with("stoke serve: listening on 127.0.0.1:"), "{line}");
    assert!(line.trim_end().ends_with("ctx=4096"), "{line}");
    let port: u16 = line.split("127.0.0.1:").nth(1).unwrap().split_whitespace().next().unwrap().parse().unwrap();
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    assert!(resp.starts_with("HTTP/1.1 200"), "{resp}");
    assert!(resp.contains("{\"status\":\"ok\"}"));
    child.kill().unwrap();
}

#[test]
fn it_refuses_like_the_engine_does() {
    let out = Command::new(exe())
        .args(["serve", "x.gguf", "--port", "0"])
        .env("FAKE_ENGINE_MODE", "refuse")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("is not supported"));
    let big = Command::new(exe()).args(["serve", "x.gguf", "--port", "0", "--ctx", "8192"]).output().unwrap();
    assert!(String::from_utf8_lossy(&big.stderr).contains("max_ctx 8192 exceeds the model's 4096"));
}
```

Run: `cargo test --features fake-engine --test fake_engine`
Expected: FAIL — the bin source `src/bin/fake_engine.rs` does not exist.

- [ ] **Step 3: Write the fake engine**

`desktop/src-tauri/src/bin/fake_engine.rs`:

```rust
//! A test double of `stoke serve` (feature "fake-engine"; never shipped). See Task 7 of the
//! milestone-1 plan for the switches. Formats follow the real server (src/server.c):
//! `data: {json}\n\n` chunks, `: ping\n\n` comments, `data: [DONE]\n\n`, and errors shaped
//! {"error":{"message":..,"type":..,"param":null,"code":null}}.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn event(line: &str) {
    if let Some(p) = env("FAKE_ENGINE_EVENTS") {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let _ = writeln!(f, "{line}");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("write-fixtures") {
        write_fixtures(args.get(2).expect("write-fixtures <dir>"));
        return;
    }
    if args.get(1).map(String::as_str) != Some("serve") {
        eprintln!("fatal: usage: stoke serve <path.gguf> [--host A] [--port P] [--ctx N] [--seqs N] [--kv q8]");
        std::process::exit(2);
    }
    event(&format!("argv {}", serde_json::to_string(&args).unwrap()));
    let model = args.get(2).cloned().unwrap_or_default();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let host = flag("--host").unwrap_or_else(|| "127.0.0.1".into());
    let port: u16 = flag("--port").and_then(|p| p.parse().ok()).unwrap_or(8080);
    let ctx: u32 = flag("--ctx").and_then(|p| p.parse().ok()).unwrap_or(4096);
    if ctx > 4096 {
        eprintln!("stoke serve: stoke_open: max_ctx {ctx} exceeds the model's 4096");
        std::process::exit(1);
    }
    let mode = env("FAKE_ENGINE_MODE").unwrap_or_else(|| "ok".into());
    if mode == "refuse" {
        eprintln!("stoke serve: model: '{model}' -- architecture 'qwen2vl' is not supported");
        std::process::exit(1);
    }
    if let Some(ms) = env("FAKE_ENGINE_LOAD_MS").and_then(|v| v.parse().ok()) {
        std::thread::sleep(Duration::from_millis(ms));
    }
    let listener = TcpListener::bind((host.as_str(), port)).expect("bind");
    let actual = listener.local_addr().unwrap().port();
    println!("stoke serve: listening on {host}:{actual}  ctx={ctx}");
    std::io::stdout().flush().unwrap();
    eprintln!("fake engine: serving {model}");
    if let Some(ms) = mode.strip_prefix("crash_after_ms:").and_then(|v| v.parse::<u64>().ok()) {
        let armed = match env("FAKE_ENGINE_CRASH_ONCE") {
            Some(marker) if std::path::Path::new(&marker).exists() => false,
            Some(marker) => {
                let _ = std::fs::write(marker, "crashed");
                true
            }
            None => true,
        };
        if armed {
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(ms));
                eprintln!("fake engine: crashing on purpose");
                std::process::exit(3);
            });
        }
    }
    let hang = mode == "hang";
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        if hang {
            std::mem::forget(stream); // accept, then never answer
            continue;
        }
        let model = model.clone();
        std::thread::spawn(move || {
            let _ = handle(stream, &model, ctx);
        });
    }
}

fn handle(mut s: TcpStream, model: &str, ctx: u32) -> std::io::Result<()> {
    let mut r = BufReader::new(s.try_clone()?);
    let mut first = String::new();
    r.read_line(&mut first)?;
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        r.read_line(&mut h)?;
        let h = h.trim_end().to_ascii_lowercase();
        if h.is_empty() {
            break;
        }
        if let Some(v) = h.strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body)?;
    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => respond(&mut s, 200, "{\"status\":\"ok\"}\n"),
        ("GET", "/v1/models") => {
            let id = std::path::Path::new(model).file_stem().map(|x| x.to_string_lossy().to_string()).unwrap_or_default();
            let body = serde_json::json!({"object":"list","data":[{"id":id,"object":"model","created":0,"owned_by":"stoke"}]});
            respond(&mut s, 200, &format!("{body}\n"))
        }
        ("POST", "/v1/chat/completions") => chat(&mut s, &body, ctx),
        _ => respond(&mut s, 404, &error_body("not found", "invalid_request_error")),
    }
}

fn error_body(message: &str, kind: &str) -> String {
    format!("{}\n", serde_json::json!({"error":{"message":message,"type":kind,"param":null,"code":null}}))
}

fn respond(s: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        _ => "Not Found",
    };
    write!(
        s,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    s.flush()
}

fn chat(s: &mut TcpStream, body: &[u8], ctx: u32) -> std::io::Result<()> {
    event(&format!("request {}", String::from_utf8_lossy(body)));
    let v: serde_json::Value = serde_json::from_slice(body).unwrap_or_default();
    let chars: usize = v["messages"]
        .as_array()
        .map(|m| m.iter().map(|x| x["content"].as_str().unwrap_or("").chars().count()).sum())
        .unwrap_or(0);
    let prompt_tokens = (chars / 4).max(1) as u64;
    let max_tokens = v["max_tokens"].as_u64().unwrap_or(128);
    if prompt_tokens + max_tokens > ctx as u64 {
        event("context_length_exceeded");
        let msg = format!(
            "prompt is {prompt_tokens} tokens and max_tokens is {max_tokens}, which exceeds the context length of {ctx}"
        );
        return respond(s, 400, &error_body(&msg, "context_length_exceeded"));
    }
    let reply = env("FAKE_ENGINE_REPLY").unwrap_or_else(|| "Hello from the fake engine.".into());
    let token_ms: u64 = env("FAKE_ENGINE_TOKEN_MS").and_then(|x| x.parse().ok()).unwrap_or(20);
    write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
    s.write_all(b": ping\n\n")?;
    let words: Vec<&str> = reply.split_inclusive(' ').collect();
    let mut n = 0u64;
    for w in &words {
        if n >= max_tokens {
            break;
        }
        let chunk = serde_json::json!({"id":"chatcmpl-fake","object":"chat.completion.chunk",
            "choices":[{"index":0,"delta":{"content":w},"finish_reason":null}]});
        if s.write_all(format!("data: {chunk}\n\n").as_bytes()).and_then(|_| s.flush()).is_err() {
            event("cancelled");
            return Ok(());
        }
        n += 1;
        std::thread::sleep(Duration::from_millis(token_ms));
    }
    let finish = if (n as usize) < words.len() { "length" } else { "stop" };
    let done = serde_json::json!({"id":"chatcmpl-fake","object":"chat.completion.chunk",
        "choices":[{"index":0,"delta":{},"finish_reason":finish}]});
    let usage = serde_json::json!({"id":"chatcmpl-fake","object":"chat.completion.chunk","choices":[],
        "usage":{"prompt_tokens":prompt_tokens,"completion_tokens":n,"total_tokens":prompt_tokens + n}});
    if write!(s, "data: {done}\n\ndata: {usage}\n\ndata: [DONE]\n\n").and_then(|_| s.flush()).is_err() {
        event("cancelled");
        return Ok(());
    }
    event("done");
    Ok(())
}

fn write_fixtures(dir: &str) {
    use lorren_lib::gguf::fixture::{write, Spec};
    let d = std::path::Path::new(dir);
    std::fs::create_dir_all(d).unwrap();
    write(&d.join("nexus.gguf"), &Spec::llama("Nexus_Model")).unwrap();
    std::fs::write(
        d.join("Modelfile"),
        "FROM C:/Users/Kabir/ai_models/nexus.gguf\nSYSTEM \"You are Nexus AI, a professional quantitative trader. You give concise, decisive trading verdicts.\"\n",
    )
    .unwrap();
    write(&d.join("Qwen3-4B-Instruct-2507-Q4_K_M.gguf"), &Spec::qwen3("Qwen3-4B")).unwrap();
    write(&d.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"), &Spec::llama("Llama-3.2-3B")).unwrap();
    write(&d.join("qwen2.5vl-3b.gguf"), &Spec { arch: "qwen2vl", ..Spec::qwen3("qwen2.5vl") }).unwrap();
    println!("fixtures written to {}", d.display());
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test --features fake-engine --test fake_engine`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add desktop/src-tauri
git commit -m "desktop: fake engine -- a test double of stoke serve (task 7)"
```

---

### Task 8: Engine supervisor

**Files:**
- Create: `desktop/src-tauri/src/engine/mod.rs`, `engine/job.rs`, `engine/memory.rs`, `engine/supervisor.rs`, `desktop/src-tauri/tests/supervisor.rs`
- Modify: `desktop/src-tauri/src/lib.rs` (add `pub mod engine;`), `Cargo.toml` (tokio, reqwest, windows, thiserror)

**Interfaces:**
- Consumes: `Monitor` (Task 6); the fake engine for tests (Task 7).
- Produces:
  - `engine::job::Job` — `Job::kill_on_close() -> io::Result<Job>`, `job.assign(&Child) -> io::Result<()>`; dropping the last `Job` kills its processes
  - `engine::memory::{available_physical_bytes() -> u64, total_physical_bytes() -> u64, process_memory(pid) -> Option<(u64 /*working set*/, u64 /*private*/)>, tree_working_set(root_pid, exclude_pid: Option<u32>) -> u64}`
  - `EngineState::{Stopped, Starting, Serving, Restarting, Failed}` (serde snake_case), `EngineStatus { state, model_id, model_path, port, pid, ctx, seqs, threads, kv, load_ms, restarts, last_error }` (ts-rs)
  - `EngineConfig { exe, ctx, seqs, threads, kv, load_timeout, health_interval, health_timeout, health_misses, backoff, failure_window, max_failures, env, min_free_bytes, check_memory }` with `EngineConfig::defaults(exe)`
  - `ModelRef { id, path, file_bytes, kv_bytes_per_token }`
  - `EngineError::{NotEnoughMemory, LoadFailed(String), LoadTimeout(u64), GaveUp(usize), Spawn(String)}` (Display = user-facing text)
  - `Supervisor::new(cfg, monitor) -> Arc<Supervisor>`; `ensure_model(self: &Arc<Self>, &ModelRef) -> Result<u16, EngineError>` (serving port); `status() -> EngineStatus`; `subscribe() -> watch::Receiver<EngineStatus>`; `stop()`
  - `pub fn parse_banner(line: &str) -> Option<u16>`

- [ ] **Step 1: Add the crates**

```bash
cd desktop/src-tauri
cargo add tokio --features rt-multi-thread,macros,sync,time
cargo add reqwest --no-default-features --features json,stream,rustls-tls
cargo add futures-util thiserror
cargo add windows --features Win32_Foundation,Win32_Security,Win32_System_JobObjects,Win32_System_Threading,Win32_System_ProcessStatus,Win32_System_SystemInformation,Win32_System_Diagnostics_ToolHelp
```

- [ ] **Step 2: Write the failing tests**

`desktop/src-tauri/tests/supervisor.rs`:

```rust
#![cfg(feature = "fake-engine")]

use lorren_lib::engine::job::Job;
use lorren_lib::engine::supervisor::*;
use lorren_lib::monitor::Monitor;
use std::path::PathBuf;
use std::time::Duration;

fn cfg(env: &[(&str, String)]) -> EngineConfig {
    let mut c = EngineConfig::defaults(PathBuf::from(env!("CARGO_BIN_EXE_fake-engine")));
    c.health_interval = Duration::from_millis(100);
    c.health_timeout = Duration::from_millis(300);
    c.backoff = vec![Duration::from_millis(50); 3];
    c.load_timeout = Duration::from_secs(10);
    c.check_memory = false;
    c.env = env.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
    c
}

fn model(id: &str, path: &str) -> ModelRef {
    ModelRef { id: id.into(), path: PathBuf::from(path), file_bytes: 0, kv_bytes_per_token: 0 }
}

fn events_file(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("lorren-sup-{}-{}.log", std::process::id(), tag));
    let _ = std::fs::remove_file(&p);
    p
}

async fn wait_for(sup: &Supervisor, want: EngineState, secs: u64) -> EngineStatus {
    let mut rx = sup.subscribe();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
    loop {
        let now = rx.borrow().clone();
        if now.state == want {
            return now;
        }
        if tokio::time::timeout_at(deadline, rx.changed()).await.is_err() {
            panic!("timed out waiting for {want:?}; status is {:?}", sup.status());
        }
    }
}

#[tokio::test]
async fn it_starts_the_engine_and_reports_serving() {
    let sup = Supervisor::new(cfg(&[]), Monitor::new(None));
    let port = sup.ensure_model(&model("a", "a.gguf")).await.unwrap();
    let s = sup.status();
    assert_eq!(s.state, EngineState::Serving);
    assert_eq!(s.port, Some(port));
    assert!(s.pid.is_some() && s.load_ms.is_some());
    assert_eq!((s.ctx, s.seqs, s.kv.as_str()), (4096, 2, "q8"));
    sup.stop().await;
}

/// Review Focus 1: every one of Kabir's model paths has a space in it.
#[tokio::test]
async fn a_model_path_with_spaces_arrives_as_one_argument() {
    let ev = events_file("spaces");
    let sup = Supervisor::new(cfg(&[("FAKE_ENGINE_EVENTS", ev.display().to_string())]), Monitor::new(None));
    sup.ensure_model(&model("n", "M:/New Drive/ai models/nexus.gguf")).await.unwrap();
    let text = std::fs::read_to_string(&ev).unwrap();
    let argv: Vec<String> = serde_json::from_str(text.lines().next().unwrap().strip_prefix("argv ").unwrap()).unwrap();
    assert_eq!(argv[1], "serve");
    assert_eq!(argv[2], "M:/New Drive/ai models/nexus.gguf");
    assert!(argv.windows(2).any(|w| w[0] == "--ctx" && w[1] == "4096"));
    assert!(argv.windows(2).any(|w| w[0] == "--seqs" && w[1] == "2"));
    assert!(argv.windows(2).any(|w| w[0] == "--kv" && w[1] == "q8"));
    sup.stop().await;
}

#[tokio::test]
async fn a_refused_model_reports_the_engines_own_reason() {
    let sup = Supervisor::new(cfg(&[("FAKE_ENGINE_MODE", "refuse".into())]), Monitor::new(None));
    let err = sup.ensure_model(&model("vl", "vl.gguf")).await.unwrap_err();
    assert!(matches!(err, EngineError::LoadFailed(ref m) if m.contains("is not supported")), "{err}");
    assert_eq!(sup.status().state, EngineState::Failed);
}

#[tokio::test]
async fn a_crash_is_restarted_and_three_crashes_give_up() {
    let sup = Supervisor::new(cfg(&[("FAKE_ENGINE_MODE", "crash_after_ms:300".into())]), Monitor::new(None));
    sup.ensure_model(&model("a", "a.gguf")).await.unwrap();
    let first = sup.status().pid;
    wait_for(&sup, EngineState::Restarting, 5).await;
    let again = wait_for(&sup, EngineState::Serving, 5).await;
    assert_ne!(again.pid, first, "a new process");
    assert!(again.restarts >= 1);
    let failed = wait_for(&sup, EngineState::Failed, 10).await;
    assert!(failed.last_error.unwrap().contains("failed 3 times"));
}

#[tokio::test]
async fn a_hung_engine_is_detected_and_replaced() {
    let sup = Supervisor::new(cfg(&[("FAKE_ENGINE_MODE", "hang".into())]), Monitor::new(None));
    sup.ensure_model(&model("a", "a.gguf")).await.unwrap();
    let s = wait_for(&sup, EngineState::Restarting, 5).await;
    assert!(s.last_error.unwrap().contains("/health"));
    sup.stop().await;
}

#[tokio::test]
async fn switching_models_restarts_and_the_same_model_does_not() {
    let sup = Supervisor::new(cfg(&[]), Monitor::new(None));
    sup.ensure_model(&model("a", "a.gguf")).await.unwrap();
    let pid_a = sup.status().pid;
    sup.ensure_model(&model("b", "b.gguf")).await.unwrap();
    let s = sup.status();
    assert_eq!(s.model_id.as_deref(), Some("b"));
    assert_ne!(s.pid, pid_a);
    sup.ensure_model(&model("b", "b.gguf")).await.unwrap();
    assert_eq!(sup.status().pid, s.pid, "no restart for the model already serving");
    sup.stop().await;
}

/// Review Focus 4: no orphan engine when the app dies.
#[test]
fn closing_the_job_kills_the_engine() {
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_fake-engine"))
        .args(["serve", "x.gguf", "--port", "0"])
        .stdout(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let job = Job::kill_on_close().unwrap();
    job.assign(&child).unwrap();
    drop(job);
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while child.try_wait().unwrap().is_none() {
        assert!(std::time::Instant::now() < deadline, "the engine outlived its job");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn the_banner_gives_the_port() {
    assert_eq!(parse_banner("stoke serve: listening on 127.0.0.1:52341  ctx=4096"), Some(52341));
    assert_eq!(parse_banner("fake engine: serving x"), None);
}
```

Create `desktop/src-tauri/src/engine/mod.rs`:

```rust
pub mod job;
pub mod memory;
pub mod supervisor;
```

and add `pub mod engine;` to `lib.rs`. Run: `cargo test --features fake-engine --test supervisor`
Expected: compile errors — `job`, `memory`, `supervisor` modules missing.

- [ ] **Step 3: The job object**

`desktop/src-tauri/src/engine/job.rs`:

```rust
//! A Windows job object that kills every process in it when its last handle closes, so the
//! engine never outlives the app, even when the app crashes (spec §3.1).

use std::os::windows::io::AsRawHandle;
use windows::core::PCWSTR;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

pub struct Job(HANDLE);

// The handle is only ever used through the Win32 calls below, which are thread-safe.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

fn io(e: windows::core::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
}

impl Job {
    pub fn kill_on_close() -> std::io::Result<Job> {
        unsafe {
            let h = CreateJobObjectW(None, PCWSTR::null()).map_err(io)?;
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(
                h,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
            .map_err(io)?;
            Ok(Job(h))
        }
    }

    pub fn assign(&self, child: &std::process::Child) -> std::io::Result<()> {
        unsafe { AssignProcessToJobObject(self.0, HANDLE(child.as_raw_handle() as _)).map_err(io) }
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}
```

If the resolved `windows` version's signatures differ (older versions return `BOOL` instead of `Result`, or take `HANDLE(isize)`), adapt these calls — the behaviour pinned by `closing_the_job_kills_the_engine` is what matters.

- [ ] **Step 4: Memory**

`desktop/src-tauri/src/engine/memory.rs`:

```rust
//! Physical and per-process memory, for the load check (spec §3.1) and the top bar (§11).

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
};
use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX};
use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

fn status() -> MEMORYSTATUSEX {
    let mut s = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
    unsafe {
        let _ = GlobalMemoryStatusEx(&mut s);
    }
    s
}

pub fn available_physical_bytes() -> u64 {
    status().ullAvailPhys
}

pub fn total_physical_bytes() -> u64 {
    status().ullTotalPhys
}

/// (working set, private bytes) of one process.
pub fn process_memory(pid: u32) -> Option<(u64, u64)> {
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let mut c = PROCESS_MEMORY_COUNTERS_EX {
            cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
            ..Default::default()
        };
        let ok = GetProcessMemoryInfo(h, &mut c as *mut _ as *mut PROCESS_MEMORY_COUNTERS, c.cb);
        let _ = CloseHandle(h);
        ok.ok()?;
        Some((c.WorkingSetSize as u64, c.PrivateUsage as u64))
    }
}

/// Working set of `root` and every process below it (the WebView2 processes), except `exclude`
/// (the engine, reported on its own).
pub fn tree_working_set(root: u32, exclude: Option<u32>) -> u64 {
    let mut parent_of = Vec::new();
    unsafe {
        let Ok(snap) = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) else { return 0 };
        let mut e = PROCESSENTRY32W { dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32, ..Default::default() };
        if Process32FirstW(snap, &mut e).is_ok() {
            loop {
                parent_of.push((e.th32ProcessID, e.th32ParentProcessID));
                if Process32NextW(snap, &mut e).is_err() {
                    break;
                }
            }
        }
        let _ = CloseHandle(snap);
    }
    let mut tree = vec![root];
    let mut i = 0;
    while i < tree.len() {
        let p = tree[i];
        for &(child, parent) in &parent_of {
            if parent == p && child != p && !tree.contains(&child) && Some(child) != exclude {
                tree.push(child);
            }
        }
        i += 1;
    }
    tree.iter().filter_map(|&pid| process_memory(pid)).map(|(ws, _)| ws).sum()
}

#[cfg(test)]
mod tests {
    #[test]
    fn this_process_has_memory_and_the_machine_has_more() {
        let (ws, private) = super::process_memory(std::process::id()).unwrap();
        assert!(ws > 0 && private > 0);
        assert!(super::total_physical_bytes() > super::available_physical_bytes());
        assert!(super::tree_working_set(std::process::id(), None) >= ws);
    }
}
```

- [ ] **Step 5: The supervisor**

`desktop/src-tauri/src/engine/supervisor.rs`:

```rust
//! Starts, watches and restarts the engine (spec §3.1). One engine process at a time; a model
//! switch is a restart. Status changes go out on a watch channel for the UI.

use crate::engine::job::Job;
use crate::engine::memory;
use crate::monitor::Monitor;
use serde::Serialize;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};
use tokio::sync::{watch, Mutex};
use ts_rs::TS;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum EngineState {
    Stopped,
    Starting,
    Serving,
    Restarting,
    Failed,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct EngineStatus {
    pub state: EngineState,
    pub model_id: Option<String>,
    pub model_path: Option<String>,
    pub port: Option<u16>,
    pub pid: Option<u32>,
    pub ctx: u32,
    pub seqs: u32,
    pub threads: Option<u32>,
    pub kv: String,
    #[ts(type = "number | null")]
    pub load_ms: Option<u64>,
    pub restarts: u32,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub exe: PathBuf,
    pub ctx: u32,
    pub seqs: u32,
    pub threads: Option<u32>,
    pub kv: String,
    pub load_timeout: Duration,
    pub health_interval: Duration,
    pub health_timeout: Duration,
    pub health_misses: u32,
    pub backoff: Vec<Duration>,
    pub failure_window: Duration,
    pub max_failures: usize,
    pub env: Vec<(String, String)>,
    pub min_free_bytes: u64,
    pub check_memory: bool,
}

impl EngineConfig {
    pub fn defaults(exe: PathBuf) -> Self {
        EngineConfig {
            exe,
            ctx: 4096,
            seqs: 2,
            threads: None,
            kv: "q8".into(),
            load_timeout: Duration::from_secs(120),
            health_interval: Duration::from_secs(5),
            health_timeout: Duration::from_secs(3),
            health_misses: 3,
            backoff: vec![Duration::from_secs(1), Duration::from_secs(5), Duration::from_secs(30)],
            failure_window: Duration::from_secs(600),
            max_failures: 3,
            env: vec![],
            min_free_bytes: 1 << 30,
            check_memory: true,
        }
    }

    fn args(&self, model: &Path) -> Vec<String> {
        let mut a: Vec<String> = vec![
            "serve".into(),
            model.to_string_lossy().to_string(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            "0".into(),
            "--ctx".into(),
            self.ctx.to_string(),
            "--seqs".into(),
            self.seqs.to_string(),
            "--kv".into(),
            self.kv.clone(),
        ];
        if let Some(t) = self.threads {
            a.push("--threads".into());
            a.push(t.to_string());
        }
        a
    }
}

#[derive(Debug, Clone)]
pub struct ModelRef {
    pub id: String,
    pub path: PathBuf,
    pub file_bytes: u64,
    pub kv_bytes_per_token: u64,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum EngineError {
    #[error("not enough memory to load {model}: it needs about {need_mb} MB and {free_mb} MB is free (1 GB is kept spare)")]
    NotEnoughMemory { model: String, need_mb: u64, free_mb: u64 },
    #[error("the engine stopped while loading: {0}")]
    LoadFailed(String),
    #[error("the engine did not start within {0} s")]
    LoadTimeout(u64),
    #[error("the engine has failed {0} times in 10 minutes; see Logs")]
    GaveUp(usize),
    #[error("could not start the engine: {0}")]
    Spawn(String),
}

pub fn parse_banner(line: &str) -> Option<u16> {
    let rest = line.split("listening on ").nth(1)?;
    rest.split_whitespace().next()?.rsplit(':').next()?.parse().ok()
}

struct Running {
    child: Child,
    _job: Option<Job>,
    model: ModelRef,
    port: u16,
    stderr_tail: Arc<StdMutex<Vec<String>>>,
}

struct Inner {
    running: Option<Running>,
    generation: u64,
    failures: Vec<Instant>,
    restarts: u32,
}

pub struct Supervisor {
    cfg: EngineConfig,
    monitor: Arc<Monitor>,
    tx: watch::Sender<EngineStatus>,
    inner: Mutex<Inner>,
    http: reqwest::Client,
}

impl Supervisor {
    pub fn new(cfg: EngineConfig, monitor: Arc<Monitor>) -> Arc<Supervisor> {
        let status = EngineStatus {
            state: EngineState::Stopped,
            model_id: None,
            model_path: None,
            port: None,
            pid: None,
            ctx: cfg.ctx,
            seqs: cfg.seqs,
            threads: cfg.threads,
            kv: cfg.kv.clone(),
            load_ms: None,
            restarts: 0,
            last_error: None,
        };
        let (tx, _rx) = watch::channel(status);
        Arc::new(Supervisor {
            cfg,
            monitor,
            tx,
            inner: Mutex::new(Inner { running: None, generation: 0, failures: vec![], restarts: 0 }),
            http: reqwest::Client::builder().no_proxy().build().expect("http client"),
        })
    }

    pub fn status(&self) -> EngineStatus {
        self.tx.borrow().clone()
    }

    pub fn subscribe(&self) -> watch::Receiver<EngineStatus> {
        self.tx.subscribe()
    }

    fn set(&self, f: impl FnOnce(&mut EngineStatus)) {
        self.tx.send_modify(f);
    }

    /// Makes sure the engine is serving `model`, starting or switching it if needed.
    /// Waits for any start already in progress. Returns the port.
    pub async fn ensure_model(self: &Arc<Self>, model: &ModelRef) -> Result<u16, EngineError> {
        let mut inner = self.inner.lock().await;
        let status = self.status();
        if let Some(r) = inner.running.as_mut() {
            if r.model.id == model.id && status.state == EngineState::Serving && matches!(r.child.try_wait(), Ok(None)) {
                return Ok(r.port);
            }
        }
        if status.state == EngineState::Failed {
            inner.failures.clear(); // an explicit request retries after giving up
        }
        self.stop_locked(&mut inner);
        self.start_locked(&mut inner, model.clone(), false).await
    }

    pub async fn stop(&self) {
        let mut inner = self.inner.lock().await;
        self.stop_locked(&mut inner);
    }

    fn stop_locked(&self, inner: &mut Inner) {
        inner.generation += 1; // retires any watcher of the old process
        if let Some(mut r) = inner.running.take() {
            let _ = r.child.kill();
            let _ = r.child.wait();
            self.monitor.log("app", "info", format!("stopped the engine ({})", r.model.id));
        }
        self.set(|s| {
            s.state = EngineState::Stopped;
            s.port = None;
            s.pid = None;
        });
    }

    fn memory_check(&self, model: &ModelRef) -> Result<(), EngineError> {
        if !self.cfg.check_memory {
            return Ok(());
        }
        let kv = model.kv_bytes_per_token * self.cfg.ctx as u64 * self.cfg.seqs as u64;
        let need = model.file_bytes + kv + (256 << 20);
        let free = memory::available_physical_bytes();
        if free < need + self.cfg.min_free_bytes {
            return Err(EngineError::NotEnoughMemory { model: model.id.clone(), need_mb: need >> 20, free_mb: free >> 20 });
        }
        Ok(())
    }

    async fn start_locked(self: &Arc<Self>, inner: &mut Inner, model: ModelRef, restart: bool) -> Result<u16, EngineError> {
        if let Err(e) = self.memory_check(&model) {
            self.set(|s| {
                s.state = EngineState::Failed;
                s.last_error = Some(e.to_string());
            });
            self.monitor.log("app", "error", e.to_string());
            return Err(e);
        }
        inner.generation += 1;
        let gen = inner.generation;
        let path = model.path.to_string_lossy().to_string();
        self.set(|s| {
            s.state = if restart { EngineState::Restarting } else { EngineState::Starting };
            s.model_id = Some(model.id.clone());
            s.model_path = Some(path.clone());
            s.port = None;
            s.pid = None;
            s.last_error = None;
        });
        self.monitor.log("app", "info", format!("starting the engine with {path}"));
        let t0 = Instant::now();
        let mut cmd = Command::new(&self.cfg.exe);
        cmd.args(self.cfg.args(&model.path))
            .envs(self.cfg.env.iter().cloned())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let mut child = cmd.spawn().map_err(|e| EngineError::Spawn(e.to_string()))?;
        let job = Job::kill_on_close().ok();
        if let Some(j) = &job {
            let _ = j.assign(&child);
        }
        let (banner_tx, banner_rx) = mpsc::channel::<u16>();
        let stdout = child.stdout.take().expect("piped stdout");
        let mon = self.monitor.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Some(p) = parse_banner(&line) {
                    let _ = banner_tx.send(p);
                }
                mon.log("engine", "info", line);
            }
        });
        let tail = Arc::new(StdMutex::new(Vec::<String>::new()));
        let (tail_w, mon) = (tail.clone(), self.monitor.clone());
        let stderr = child.stderr.take().expect("piped stderr");
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let level = if line.contains("fatal") || line.contains("not supported") { "error" } else { "info" };
                mon.log("engine", level, line.clone());
                let mut t = tail_w.lock().unwrap();
                t.push(line);
                if t.len() > 20 {
                    t.remove(0);
                }
            }
        });
        let timeout = self.cfg.load_timeout;
        // Ok(port) | Err(true) = exited before the banner | Err(false) = timed out
        let waited = tokio::task::spawn_blocking(move || match banner_rx.recv_timeout(timeout) {
            Ok(p) => Ok(p),
            Err(mpsc::RecvTimeoutError::Timeout) => Err(false),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(true),
        })
        .await
        .unwrap_or(Err(true));
        match waited {
            Ok(port) => {
                let load_ms = t0.elapsed().as_millis() as u64;
                let pid = child.id();
                let restarts = inner.restarts;
                inner.running = Some(Running { child, _job: job, model: model.clone(), port, stderr_tail: tail });
                self.set(|s| {
                    s.state = EngineState::Serving;
                    s.port = Some(port);
                    s.pid = Some(pid);
                    s.load_ms = Some(load_ms);
                    s.restarts = restarts;
                });
                self.monitor.log(
                    "app",
                    "info",
                    format!("engine serving {} on port {port} after {:.2} s", model.id, load_ms as f64 / 1000.0),
                );
                self.spawn_watch(gen);
                Ok(port)
            }
            Err(exited) => {
                let _ = child.kill();
                let _ = child.wait();
                tokio::time::sleep(Duration::from_millis(100)).await; // let stderr drain its last line
                let reason = tail
                    .lock()
                    .unwrap()
                    .iter()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .cloned()
                    .unwrap_or_else(|| "no output".into());
                let err = if exited { EngineError::LoadFailed(reason) } else { EngineError::LoadTimeout(timeout.as_secs()) };
                self.set(|s| {
                    s.state = EngineState::Failed;
                    s.port = None;
                    s.pid = None;
                    s.last_error = Some(err.to_string());
                });
                self.monitor.log("app", "error", err.to_string());
                Err(err)
            }
        }
    }

    fn spawn_watch(self: &Arc<Self>, gen: u64) {
        let me = self.clone();
        tokio::spawn(async move {
            let mut misses = 0u32;
            loop {
                tokio::time::sleep(me.cfg.health_interval).await;
                let (port, exited) = {
                    let mut inner = me.inner.lock().await;
                    if inner.generation != gen {
                        return;
                    }
                    let Some(r) = inner.running.as_mut() else { return };
                    (r.port, !matches!(r.child.try_wait(), Ok(None)))
                };
                if !exited && me.healthy(port).await {
                    misses = 0;
                    continue;
                }
                if !exited {
                    misses += 1;
                    if misses < me.cfg.health_misses {
                        continue;
                    }
                }
                let why = if exited { "the engine exited" } else { "the engine stopped answering /health" };
                me.recover(gen, why).await;
                return;
            }
        });
    }

    async fn healthy(&self, port: u16) -> bool {
        matches!(
            self.http.get(format!("http://127.0.0.1:{port}/health")).timeout(self.cfg.health_timeout).send().await,
            Ok(r) if r.status().is_success()
        )
    }

    /// A crash or a hang: restart after a backoff, or give up after `max_failures` in `failure_window`.
    async fn recover(self: &Arc<Self>, gen: u64, why: &str) {
        let (model, delay) = {
            let mut inner = self.inner.lock().await;
            if inner.generation != gen {
                return;
            }
            let Some(mut r) = inner.running.take() else { return };
            let _ = r.child.kill();
            let _ = r.child.wait();
            let last = r.stderr_tail.lock().unwrap().last().cloned();
            self.monitor.log(
                "app",
                "warn",
                format!("{why}{}", last.map(|l| format!(" (last engine line: {l})")).unwrap_or_default()),
            );
            let now = Instant::now();
            let window = self.cfg.failure_window;
            inner.failures.retain(|t| now.duration_since(*t) < window);
            inner.failures.push(now);
            if inner.failures.len() >= self.cfg.max_failures {
                let e = EngineError::GaveUp(inner.failures.len());
                self.set(|s| {
                    s.state = EngineState::Failed;
                    s.port = None;
                    s.pid = None;
                    s.last_error = Some(e.to_string());
                });
                self.monitor.log("app", "error", e.to_string());
                return;
            }
            inner.restarts += 1;
            let delay = self.cfg.backoff.get(inner.failures.len() - 1).copied().unwrap_or(Duration::from_secs(30));
            self.set(|s| {
                s.state = EngineState::Restarting;
                s.port = None;
                s.pid = None;
                s.last_error = Some(why.to_string());
            });
            (r.model, delay)
        };
        tokio::time::sleep(delay).await;
        let mut inner = self.inner.lock().await;
        if inner.generation != gen || inner.running.is_some() {
            return; // someone started an engine meanwhile
        }
        let _ = self.start_locked(&mut inner, model, true).await;
    }
}
```

- [ ] **Step 6: Run the tests**

```bash
cargo test --features fake-engine --test supervisor
cargo test memory
```

Expected: 8 passed, then 1 passed.

- [ ] **Step 7: Commit**

```bash
git add desktop/src-tauri desktop/src/ipc/bindings
git commit -m "desktop: engine supervisor -- start, health, backoff restarts, job object (task 8)"
```

---

### Task 9: SSE parser and budget

**Files:**
- Create: `desktop/src-tauri/src/sse.rs`, `desktop/src-tauri/src/budget.rs`
- Modify: `desktop/src-tauri/src/lib.rs` (add `pub mod sse; pub mod budget;`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `sse::SseEvent::{Data(String), Comment, Done}`; `sse::SseParser::new()`, `push(&mut self, chunk: &[u8]) -> Vec<SseEvent>`
  - `sse::ChatChunk { choices: Vec<Choice>, usage: Option<Usage> }`, `Choice { delta: Delta, finish_reason: Option<String> }`, `Delta { content: Option<String> }`, `Usage { prompt_tokens: u32, completion_tokens: u32 }` (serde Deserialize)
  - `budget::{HISTORY_BUDGET = 1_500, REPLY_CAP = 1_024, MIN_REPLY = 64}`
  - `budget::Turn { role: &'static str, content: String }`
  - `budget::estimate_tokens(&str) -> u32`, `select_history(&[Turn], budget: u32) -> (Vec<Turn>, u32 /*dropped*/)`, `max_tokens_for(ctx, prompt_tokens, cap) -> Option<u32>`, `message_too_long(system: Option<&str>, message: &str, ctx: u32) -> Option<u32 /*estimate*/>`, `parse_context_error(msg: &str) -> Option<(u32 /*prompt*/, u32 /*max_tokens*/, u32 /*ctx*/)>`

- [ ] **Step 1: Write the failing tests**

`desktop/src-tauri/src/sse.rs` (tests only for now):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Review Focus 2: a character split across network chunks must survive.
    #[test]
    fn multibyte_characters_survive_every_split_point() {
        let line = "data: {\"content\":\"Buy at ₹68,400 🚀\"}\n\n".as_bytes();
        for cut in 1..line.len() {
            let mut p = SseParser::new();
            let mut events = p.push(&line[..cut]);
            events.extend(p.push(&line[cut..]));
            assert_eq!(events, vec![SseEvent::Data("{\"content\":\"Buy at ₹68,400 🚀\"}".into())], "cut at byte {cut}");
        }
    }

    #[test]
    fn pings_done_crlf_and_several_events_in_one_chunk() {
        let mut p = SseParser::new();
        let ev = p.push(b": ping\n\ndata: {\"a\":1}\r\n\r\ndata: [DONE]\n\n");
        assert_eq!(ev, vec![SseEvent::Comment, SseEvent::Data("{\"a\":1}".into()), SseEvent::Done]);
    }

    #[test]
    fn a_partial_line_waits_for_the_rest() {
        let mut p = SseParser::new();
        assert!(p.push(b"data: {\"a\"").is_empty());
        assert_eq!(p.push(b":2}\n"), vec![SseEvent::Data("{\"a\":2}".into())]);
    }

    #[test]
    fn engine_chunks_parse() {
        let c: ChatChunk = serde_json::from_str(
            r#"{"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#,
        )
        .unwrap();
        assert_eq!(c.choices[0].delta.content.as_deref(), Some("Hi"));
        let u: ChatChunk = serde_json::from_str(
            r#"{"choices":[],"usage":{"prompt_tokens":69,"completion_tokens":6,"total_tokens":75}}"#,
        )
        .unwrap();
        assert_eq!(u.usage.unwrap().completion_tokens, 6);
    }
}
```

`desktop/src-tauri/src/budget.rs` (tests only for now):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: &'static str, n: usize) -> Turn {
        Turn { role, content: "x".repeat(n) }
    }

    #[test]
    fn the_most_recent_turns_that_fit_are_kept_whole() {
        let h = vec![turn("user", 700), turn("assistant", 700), turn("user", 700), turn("assistant", 700)];
        let per = estimate_tokens(&h[0].content);
        let (kept, dropped) = select_history(&h, per * 2);
        assert_eq!(kept.len(), 2);
        assert_eq!(dropped, 2);
        assert_eq!(kept[0].role, "user");
    }

    #[test]
    fn history_never_starts_on_an_assistant_turn() {
        let h = vec![turn("user", 700), turn("assistant", 70), turn("user", 70), turn("assistant", 70)];
        let (kept, _) = select_history(&h, estimate_tokens(&"x".repeat(70)) * 3);
        assert_eq!(kept.first().unwrap().role, "user");
    }

    #[test]
    fn max_tokens_is_capped_and_leaves_room_for_the_prompt() {
        assert_eq!(max_tokens_for(4096, 500, REPLY_CAP), Some(1024));
        assert_eq!(max_tokens_for(4096, 3500, REPLY_CAP), Some(4096 - 3500 - 8));
        assert_eq!(max_tokens_for(4096, 4060, REPLY_CAP), None);
    }

    /// Review Focus 5: a pasted document longer than the whole context.
    #[test]
    fn a_message_longer_than_the_context_is_caught_before_sending() {
        let doc = "word ".repeat(4_000);
        let est = message_too_long(None, &doc, 4096).expect("refused");
        assert!(est > 4096);
        assert_eq!(message_too_long(Some("be brief"), "hello", 4096), None);
    }

    #[test]
    fn the_engines_context_message_is_read() {
        let msg = "prompt is 4210 tokens and max_tokens is 1024, which exceeds the context length of 4096";
        assert_eq!(parse_context_error(msg), Some((4210, 1024, 4096)));
        assert_eq!(parse_context_error("something else"), None);
    }
}
```

Add `pub mod sse; pub mod budget;` to `lib.rs`. Run: `cargo test sse budget` — Expected: compile errors.

- [ ] **Step 2: Implement the SSE parser**

Above the tests in `sse.rs`:

```rust
//! Splits an SSE byte stream into events (spec §3.2). Bytes are buffered until a whole line
//! has arrived, so a UTF-8 character split across network chunks is never decoded in halves
//! (a 0x0A byte never occurs inside a multi-byte UTF-8 sequence).

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq)]
pub enum SseEvent {
    Data(String),
    Comment,
    Done,
}

#[derive(Default)]
pub struct SseParser {
    buf: Vec<u8>,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
            let mut line: Vec<u8> = self.buf.drain(..=nl).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if line.is_empty() {
                continue;
            }
            if line[0] == b':' {
                out.push(SseEvent::Comment);
            } else if let Some(rest) = line.strip_prefix(b"data:") {
                let text = String::from_utf8_lossy(rest).trim_start().to_string();
                out.push(if text == "[DONE]" { SseEvent::Done } else { SseEvent::Data(text) });
            }
        }
        out
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct ChatChunk {
    #[serde(default)]
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    #[serde(default)]
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct Delta {
    pub content: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}
```

- [ ] **Step 3: Implement the budget**

Above the tests in `budget.rs`:

```rust
//! Which part of a chat to send, and how long a reply may be (spec §3.2).
//! The app has no tokenizer: it estimates conservatively, and the engine's 400 reply corrects it.

/// Earlier turns sent with each message, in estimated tokens (decided 2026-09-25).
pub const HISTORY_BUDGET: u32 = 1_500;
pub const REPLY_CAP: u32 = 1_024;
pub const MIN_REPLY: u32 = 64;
/// Template tokens around each message (role headers, end markers).
const PER_MESSAGE: u32 = 6;
const MARGIN: u32 = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    /// "system", "user" or "assistant"
    pub role: &'static str,
    pub content: String,
}

/// About 3.5 characters per token, rounded up, plus the template's per-message cost.
pub fn estimate_tokens(text: &str) -> u32 {
    let chars = text.chars().count() as u32;
    (chars * 2 + 6) / 7 + PER_MESSAGE
}

/// Keeps the most recent earlier turns that fit in `budget`, whole messages only, and never
/// starts on an assistant turn. Returns the kept turns and how many were dropped.
pub fn select_history(history: &[Turn], budget: u32) -> (Vec<Turn>, u32) {
    let mut used = 0u32;
    let mut start = history.len();
    for (i, t) in history.iter().enumerate().rev() {
        let cost = estimate_tokens(&t.content);
        if used + cost > budget {
            break;
        }
        used += cost;
        start = i;
    }
    while start < history.len() && history[start].role != "user" {
        start += 1;
    }
    (history[start..].to_vec(), start as u32)
}

/// The reply length to ask for: at most `cap`, and what the context leaves after the prompt.
/// `None` when the prompt leaves less than MIN_REPLY tokens.
pub fn max_tokens_for(ctx: u32, prompt_tokens: u32, cap: u32) -> Option<u32> {
    let room = ctx.checked_sub(prompt_tokens + MARGIN)?;
    if room < MIN_REPLY { None } else { Some(room.min(cap)) }
}

/// Spec §5: a message that cannot fit even with no history is refused before sending.
pub fn message_too_long(system: Option<&str>, message: &str, ctx: u32) -> Option<u32> {
    let est = system.map(estimate_tokens).unwrap_or(0) + estimate_tokens(message);
    max_tokens_for(ctx, est, REPLY_CAP).is_none().then_some(est)
}

/// Reads the engine's 400 `context_length_exceeded` message:
/// "prompt is N tokens and max_tokens is M, which exceeds the context length of C".
pub fn parse_context_error(msg: &str) -> Option<(u32, u32, u32)> {
    let after = |key: &str| -> Option<u32> {
        let i = msg.find(key)? + key.len();
        msg[i..].trim_start().split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
    };
    Some((after("prompt is")?, after("max_tokens is")?, after("context length of")?))
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test sse` then `cargo test budget` — Expected: 4 passed, then 5 passed.

- [ ] **Step 5: Commit**

```bash
git add desktop/src-tauri
git commit -m "desktop: SSE parser (byte-safe) and the history/context budget (task 9)"
```

---

### Task 10: Chat service

**Files:**
- Create: `desktop/src-tauri/src/chat.rs`, `desktop/src-tauri/tests/chat_flow.rs`
- Modify: `desktop/src-tauri/src/lib.rs` (add `pub mod chat;`), `Cargo.toml` (tokio-util)

**Interfaces:**
- Consumes: `Store` (Task 5), `Monitor` (Task 6), `Supervisor`, `ModelRef` (Task 8), `SseParser`, `ChatChunk` (Task 9), `budget::*` (Task 9), `ModelInfo` (Task 4).
- Produces:
  - `ReplyStats { prompt_tokens, completion_tokens, ttft_ms, total_ms, tokens_per_s, finish_reason }` (ts-rs)
  - `StreamEvent` (serde `tag = "kind"`, snake_case; ts-rs): `Started { message_id, user_message_id, dropped_turns, request_id }`, `Token { text }`, `Finished { stats }`, `Stopped { stats }`, `Interrupted { reason }`, `Error { message }`
  - `ChatSettings { history_budget, reply_cap, temperature, top_p, top_k, repeat_penalty, liveness, save_every }` with `Default` (1,500 / 1,024 / 0.7 / 0.9 / 40 / 1.1 / 45 s / 500 ms)
  - `ChatError::{Busy, UnknownModel, NotRunnable, TooLong(est, ctx), NothingToRegenerate, Store}`
  - `ChatService::new(store: Arc<Mutex<Store>>, supervisor: Arc<Supervisor>, library: Arc<RwLock<Vec<ModelInfo>>>, monitor: Arc<Monitor>, settings) -> Arc<ChatService>`
  - `send(self: &Arc<Self>, chat_id, text, sink: impl Fn(StreamEvent) + Send + Sync) -> Result<(), ChatError>`; `regenerate(self: &Arc<Self>, chat_id, sink)`; `stop(&self, chat_id)`
  - `pub fn model_ref(m: &ModelInfo) -> ModelRef`

- [ ] **Step 1: Add the crate**

```bash
cd desktop/src-tauri && cargo add tokio-util
```

- [ ] **Step 2: Write the failing tests**

`desktop/src-tauri/tests/chat_flow.rs`:

```rust
#![cfg(feature = "fake-engine")]

use lorren_lib::chat::*;
use lorren_lib::engine::supervisor::{EngineConfig, EngineState, Supervisor};
use lorren_lib::gguf::fixture::{self, Spec};
use lorren_lib::library::{self, LibraryConfig};
use lorren_lib::monitor::Monitor;
use lorren_lib::store::{MsgStatus, Role, Store};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

struct World {
    svc: Arc<ChatService>,
    events_file: PathBuf,
    qwen: String,
    nexus: String,
}

fn world(tag: &str, env: &[(&str, &str)], ctx: u32, settings: ChatSettings) -> World {
    let dir = std::env::temp_dir().join(format!("lorren chat {}-{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    fixture::write(&dir.join("Qwen3-4B-Instruct-2507-Q4_K_M.gguf"), &Spec::qwen3("Qwen3-4B")).unwrap();
    fixture::write(&dir.join("nexus.gguf"), &Spec::llama("Nexus_Model")).unwrap();
    let models = library::scan(&LibraryConfig {
        folders: vec![dir.clone()],
        ollama_models_dir: None,
        default_prefix: "qwen3-4b-instruct-2507-q4_k_m".into(),
    });
    let qwen = models.iter().find(|m| m.is_default).unwrap().id.clone();
    let nexus = models.iter().find(|m| m.name == "nexus").unwrap().id.clone();
    let events_file = dir.join("events.log");
    let mut cfg = EngineConfig::defaults(PathBuf::from(env!("CARGO_BIN_EXE_fake-engine")));
    cfg.ctx = ctx;
    cfg.check_memory = false;
    cfg.health_interval = Duration::from_millis(100);
    cfg.backoff = vec![Duration::from_millis(50); 3];
    cfg.env = env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    cfg.env.push(("FAKE_ENGINE_EVENTS".into(), events_file.display().to_string()));
    let monitor = Monitor::new(None);
    let sup = Supervisor::new(cfg, monitor.clone());
    let store = Arc::new(Mutex::new(Store::open_in_memory().unwrap()));
    let svc = ChatService::new(store, sup, Arc::new(RwLock::new(models)), monitor, settings);
    World { svc, events_file, qwen, nexus }
}

fn fast() -> ChatSettings {
    ChatSettings { liveness: Duration::from_secs(5), save_every: Duration::from_millis(50), ..ChatSettings::default() }
}

fn collector() -> (Arc<Mutex<Vec<StreamEvent>>>, impl Fn(StreamEvent) + Send + Sync + Clone) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s2 = seen.clone();
    (seen, move |e| s2.lock().unwrap().push(e))
}

fn events(w: &World) -> String {
    std::fs::read_to_string(&w.events_file).unwrap_or_default()
}

fn new_chat(w: &World, model: &str) -> String {
    w.svc.store.lock().unwrap().create_chat(model, None).unwrap().id
}

fn last_reply(w: &World, chat: &str) -> lorren_lib::store::Message {
    w.svc.store.lock().unwrap().active_path(chat).unwrap().pop().unwrap()
}

#[tokio::test]
async fn a_reply_streams_into_the_ui_and_the_store() {
    let w = world("stream", &[("FAKE_ENGINE_REPLY", "Verdict: HOLD."), ("FAKE_ENGINE_TOKEN_MS", "5")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = collector();
    w.svc.send(&chat, "Long or wait?", sink).await.unwrap();
    let seen = seen.lock().unwrap();
    assert!(matches!(seen.first(), Some(StreamEvent::Started { dropped_turns: 0, .. })));
    let text: String = seen.iter().filter_map(|e| match e { StreamEvent::Token { text } => Some(text.clone()), _ => None }).collect();
    assert_eq!(text, "Verdict: HOLD.");
    assert!(matches!(seen.last(), Some(StreamEvent::Finished { stats }) if stats.completion_tokens == Some(2)));
    let reply = last_reply(&w, &chat);
    assert_eq!((reply.role, reply.status, reply.content.as_str()), (Role::Assistant, MsgStatus::Done, "Verdict: HOLD."));
    let req = &w.svc.monitor.requests()[0];
    assert_eq!(req.status, "ok");
    assert!(req.ttft_ms.is_some());
    let body = events(&w);
    assert!(body.contains("\"max_tokens\":1024") && body.contains("\"include_usage\":true"), "{body}");
}

/// Review Focus 2, end to end.
#[tokio::test]
async fn multibyte_text_arrives_intact() {
    let w = world("utf8", &[("FAKE_ENGINE_REPLY", "Buy at ₹68,400 🚀 now"), ("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (_seen, sink) = collector();
    w.svc.send(&chat, "₹?", sink).await.unwrap();
    assert_eq!(last_reply(&w, &chat).content, "Buy at ₹68,400 🚀 now");
}

#[tokio::test]
async fn stop_cancels_the_generation_in_the_engine() {
    let long = "word ".repeat(60);
    let w = world("stop", &[("FAKE_ENGINE_REPLY", long.as_str()), ("FAKE_ENGINE_TOKEN_MS", "100")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = collector();
    let svc = w.svc.clone();
    let c2 = chat.clone();
    let task = tokio::spawn(async move { svc.send(&c2, "go", sink).await });
    for _ in 0..100 {
        if seen.lock().unwrap().iter().any(|e| matches!(e, StreamEvent::Token { .. })) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    w.svc.stop(&chat);
    task.await.unwrap().unwrap();
    assert!(matches!(seen.lock().unwrap().last(), Some(StreamEvent::Stopped { .. })));
    let reply = last_reply(&w, &chat);
    assert_eq!(reply.status, MsgStatus::Stopped);
    assert!(!reply.content.is_empty() && reply.content.len() < long.len());
    for _ in 0..30 {
        if events(&w).contains("cancelled") {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("the engine never saw the connection close:\n{}", events(&w));
}

#[tokio::test]
async fn an_engine_crash_marks_the_reply_interrupted_and_the_next_send_works() {
    let marker = std::env::temp_dir().join(format!("lorren-crash-once-{}", std::process::id()));
    let _ = std::fs::remove_file(&marker);
    let long = "word ".repeat(40);
    let m = marker.display().to_string();
    let w = world(
        "crash",
        &[("FAKE_ENGINE_MODE", "crash_after_ms:400"), ("FAKE_ENGINE_CRASH_ONCE", m.as_str()),
          ("FAKE_ENGINE_REPLY", long.as_str()), ("FAKE_ENGINE_TOKEN_MS", "50")],
        4096,
        fast(),
    );
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = collector();
    w.svc.send(&chat, "go", sink.clone()).await.unwrap();
    assert!(matches!(seen.lock().unwrap().last(), Some(StreamEvent::Interrupted { .. })));
    assert_eq!(last_reply(&w, &chat).status, MsgStatus::Interrupted);
    for _ in 0..100 {
        if w.svc.supervisor.status().state == EngineState::Serving {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    w.svc.send(&chat, "again", sink).await.unwrap();
    assert!(matches!(seen.lock().unwrap().last(), Some(StreamEvent::Finished { .. })));
}

#[tokio::test]
async fn a_full_context_is_trimmed_and_retried_once() {
    let settings = ChatSettings { history_budget: 100_000, ..fast() };
    let w = world("ctx", &[("FAKE_ENGINE_REPLY", "ok"), ("FAKE_ENGINE_TOKEN_MS", "1")], 512, settings);
    let chat = new_chat(&w, &w.qwen);
    {
        let s = w.svc.store.lock().unwrap();
        let mut parent: Option<String> = None;
        for i in 0..6 {
            let role = if i % 2 == 0 { Role::User } else { Role::Assistant };
            let m = s.append_message(&chat, parent.as_deref(), role, &"y".repeat(400), MsgStatus::Done, None).unwrap();
            parent = Some(m.id);
        }
    }
    let (seen, sink) = collector();
    w.svc.send(&chat, "and now?", sink).await.unwrap();
    let seen = seen.lock().unwrap();
    assert!(matches!(seen.first(), Some(StreamEvent::Started { dropped_turns, .. }) if *dropped_turns > 0), "{seen:?}");
    assert!(matches!(seen.last(), Some(StreamEvent::Finished { .. })), "{seen:?}");
    assert_eq!(events(&w).matches("context_length_exceeded").count(), 1);
}

#[tokio::test]
async fn the_history_budget_drops_the_oldest_turns() {
    let settings = ChatSettings { history_budget: 60, ..fast() };
    let w = world("budget", &[("FAKE_ENGINE_REPLY", "ok"), ("FAKE_ENGINE_TOKEN_MS", "1")], 4096, settings);
    let chat = new_chat(&w, &w.qwen);
    {
        let s = w.svc.store.lock().unwrap();
        let mut parent: Option<String> = None;
        for i in 0..4 {
            let role = if i % 2 == 0 { Role::User } else { Role::Assistant };
            let m = s.append_message(&chat, parent.as_deref(), role, &format!("turn {i} {}", "z".repeat(100)), MsgStatus::Done, None).unwrap();
            parent = Some(m.id);
        }
    }
    let (seen, sink) = collector();
    w.svc.send(&chat, "latest", sink).await.unwrap();
    assert!(matches!(seen.lock().unwrap().first(), Some(StreamEvent::Started { dropped_turns, .. }) if *dropped_turns >= 2));
    let body = events(&w);
    assert!(!body.contains("turn 0") && body.contains("latest"), "{body}");
}

#[tokio::test]
async fn regenerate_adds_a_sibling_and_shows_it() {
    let w = world("regen", &[("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (_seen, sink) = collector();
    w.svc.send(&chat, "hi", sink.clone()).await.unwrap();
    let first = last_reply(&w, &chat);
    w.svc.regenerate(&chat, sink).await.unwrap();
    let second = last_reply(&w, &chat);
    assert_ne!(first.id, second.id);
    assert_eq!(second.parent_id, first.parent_id);
    assert_eq!(w.svc.store.lock().unwrap().siblings(&second.id).unwrap().len(), 2);
}

/// Review Focus 3: a send while the engine is switching models waits instead of failing.
#[tokio::test]
async fn a_send_during_a_model_switch_waits_for_the_engine() {
    let w = world("switch", &[("FAKE_ENGINE_LOAD_MS", "400"), ("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let chat = new_chat(&w, &w.nexus);
    let lib = w.svc.library.read().unwrap().clone();
    let qwen = lib.iter().find(|m| m.id == w.qwen).unwrap().clone();
    let sup = w.svc.supervisor.clone();
    let switching = tokio::spawn(async move { sup.ensure_model(&model_ref(&qwen)).await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    let (seen, sink) = collector();
    w.svc.send(&chat, "hello", sink).await.unwrap();
    switching.await.unwrap().unwrap();
    assert!(matches!(seen.lock().unwrap().last(), Some(StreamEvent::Finished { .. })));
    assert_eq!(w.svc.supervisor.status().model_id.as_deref(), Some(w.nexus.as_str()));
}

/// Review Focus 5: refused before any request is sent.
#[tokio::test]
async fn a_message_longer_than_the_context_is_refused_up_front() {
    let w = world("toolong", &[], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (_seen, sink) = collector();
    let err = w.svc.send(&chat, &"word ".repeat(5_000), sink).await.unwrap_err();
    assert!(matches!(err, ChatError::TooLong(est, 4096) if est > 4096), "{err}");
    assert!(!events(&w).contains("request"));
    assert!(w.svc.store.lock().unwrap().active_path(&chat).unwrap().is_empty());
}
```

Add `pub mod chat;` to `lib.rs`. Run: `cargo test --features fake-engine --test chat_flow`
Expected: compile errors — `chat` module missing.

- [ ] **Step 3: Implement the chat service**

`desktop/src-tauri/src/chat.rs`:

```rust
//! The chat service (spec §3.2): turns a chat into engine requests, streams the reply into
//! the UI and the store, and handles Stop, crashes, the context limit and regenerate.

use crate::budget::{self, Turn};
use crate::engine::supervisor::{ModelRef, Supervisor};
use crate::library::ModelInfo;
use crate::monitor::Monitor;
use crate::sse::{ChatChunk, SseEvent, SseParser, Usage};
use crate::store::{Message, MsgStats, MsgStatus, Role, Store};
use futures_util::StreamExt;
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use ts_rs::TS;

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct ReplyStats {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    #[ts(type = "number | null")]
    pub ttft_ms: Option<u64>,
    #[ts(type = "number")]
    pub total_ms: u64,
    pub tokens_per_s: Option<f32>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StreamEvent {
    Started { message_id: String, user_message_id: String, dropped_turns: u32, request_id: String },
    Token { text: String },
    Finished { stats: ReplyStats },
    Stopped { stats: ReplyStats },
    Interrupted { reason: String },
    Error { message: String },
}

pub struct ChatSettings {
    pub history_budget: u32,
    pub reply_cap: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
    /// No bytes for this long means the engine is not responding (it pings every 15 s).
    pub liveness: Duration,
    pub save_every: Duration,
}

impl Default for ChatSettings {
    fn default() -> Self {
        ChatSettings {
            history_budget: budget::HISTORY_BUDGET,
            reply_cap: budget::REPLY_CAP,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            liveness: Duration::from_secs(45),
            save_every: Duration::from_millis(500),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    #[error("a reply is already being written in this chat")]
    Busy,
    #[error("no such model: {0}")]
    UnknownModel(String),
    #[error("{0} cannot run: {1}")]
    NotRunnable(String, String),
    #[error("this message is about {0} tokens, more than a {1}-token context can take")]
    TooLong(u32, u32),
    #[error("there is no reply to regenerate")]
    NothingToRegenerate,
    #[error(transparent)]
    Store(#[from] rusqlite::Error),
}

pub fn model_ref(m: &ModelInfo) -> ModelRef {
    ModelRef {
        id: m.id.clone(),
        path: PathBuf::from(&m.path),
        file_bytes: m.size_bytes,
        kv_bytes_per_token: m.kv_bytes_per_token,
    }
}

enum Outcome {
    Done,
    Stopped,
    Interrupted(String),
}

pub struct ChatService {
    pub store: Arc<Mutex<Store>>,
    pub supervisor: Arc<Supervisor>,
    pub library: Arc<RwLock<Vec<ModelInfo>>>,
    pub monitor: Arc<Monitor>,
    pub settings: ChatSettings,
    http: reqwest::Client,
    active: Mutex<HashMap<String, CancellationToken>>,
}

struct ActiveGuard<'a> {
    svc: &'a ChatService,
    chat_id: String,
}

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        self.svc.active.lock().unwrap().remove(&self.chat_id);
    }
}

fn role_str(r: Role) -> &'static str {
    match r {
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

fn turns(path: &[Message]) -> Vec<Turn> {
    path.iter()
        .filter(|m| m.status != MsgStatus::Error && !m.content.is_empty())
        .map(|m| Turn { role: role_str(m.role), content: m.content.clone() })
        .collect()
}

impl ChatService {
    pub fn new(
        store: Arc<Mutex<Store>>,
        supervisor: Arc<Supervisor>,
        library: Arc<RwLock<Vec<ModelInfo>>>,
        monitor: Arc<Monitor>,
        settings: ChatSettings,
    ) -> Arc<ChatService> {
        Arc::new(ChatService {
            store,
            supervisor,
            library,
            monitor,
            settings,
            http: reqwest::Client::builder().no_proxy().build().expect("http client"),
            active: Mutex::new(HashMap::new()),
        })
    }

    pub fn stop(&self, chat_id: &str) {
        if let Some(t) = self.active.lock().unwrap().get(chat_id) {
            t.cancel();
        }
    }

    fn begin(&self, chat_id: &str) -> Result<(ActiveGuard<'_>, CancellationToken), ChatError> {
        let mut a = self.active.lock().unwrap();
        if a.contains_key(chat_id) {
            return Err(ChatError::Busy);
        }
        let t = CancellationToken::new();
        a.insert(chat_id.to_string(), t.clone());
        Ok((ActiveGuard { svc: self, chat_id: chat_id.to_string() }, t))
    }

    fn model(&self, id: &str) -> Result<ModelInfo, ChatError> {
        let lib = self.library.read().unwrap();
        let m = lib.iter().find(|m| m.id == id).cloned().ok_or_else(|| ChatError::UnknownModel(id.into()))?;
        if !m.runnable {
            return Err(ChatError::NotRunnable(m.name.clone(), m.reason.clone().unwrap_or_default()));
        }
        Ok(m)
    }

    pub async fn send(
        self: &Arc<Self>,
        chat_id: &str,
        text: &str,
        sink: impl Fn(StreamEvent) + Send + Sync,
    ) -> Result<(), ChatError> {
        let (_guard, cancel) = self.begin(chat_id)?;
        let (chat, path) = {
            let s = self.store.lock().unwrap();
            (s.get_chat(chat_id)?, s.active_path(chat_id)?)
        };
        let model = self.model(&chat.model_id)?;
        let system = chat.system_prompt.clone().or_else(|| model.system_prompt.clone());
        if let Some(est) = budget::message_too_long(system.as_deref(), text, model.ctx_engine) {
            return Err(ChatError::TooLong(est, model.ctx_engine));
        }
        let parent = path.last().map(|m| m.id.clone());
        let user = self
            .store
            .lock()
            .unwrap()
            .append_message(chat_id, parent.as_deref(), Role::User, text, MsgStatus::Done, None)?;
        self.stream(chat_id, &model, system, turns(&path), text.to_string(), &user, &sink, &cancel).await
    }

    pub async fn regenerate(self: &Arc<Self>, chat_id: &str, sink: impl Fn(StreamEvent) + Send + Sync) -> Result<(), ChatError> {
        let (_guard, cancel) = self.begin(chat_id)?;
        let (chat, path) = {
            let s = self.store.lock().unwrap();
            (s.get_chat(chat_id)?, s.active_path(chat_id)?)
        };
        let pos = path.iter().rposition(|m| m.role == Role::User).ok_or(ChatError::NothingToRegenerate)?;
        let user = path[pos].clone();
        let model = self.model(&chat.model_id)?;
        let system = chat.system_prompt.clone().or_else(|| model.system_prompt.clone());
        self.stream(chat_id, &model, system, turns(&path[..pos]), user.content.clone(), &user, &sink, &cancel).await
    }

    fn request_body(&self, messages: &[(&str, String)], max_tokens: u32) -> serde_json::Value {
        let msgs: Vec<_> = messages.iter().map(|(r, c)| serde_json::json!({"role": r, "content": c})).collect();
        serde_json::json!({
            "model": "local",
            "messages": msgs,
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": max_tokens,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "top_k": self.settings.top_k,
            "repeat_penalty": self.settings.repeat_penalty,
        })
    }

    fn finish_early(&self, msg: &str, req: &str, sink: &(impl Fn(StreamEvent) + Send + Sync), status: MsgStatus, text: String) -> Result<(), ChatError> {
        let stats = MsgStats { error: (status == MsgStatus::Error).then(|| text.clone()), ..Default::default() };
        self.store.lock().unwrap().finish_message(msg, status, &stats)?;
        let label = if status == MsgStatus::Stopped { "stopped" } else { "error" };
        self.monitor.update_request(req, |r| r.status = label.into());
        if status == MsgStatus::Stopped {
            sink(StreamEvent::Stopped {
                stats: ReplyStats { prompt_tokens: None, completion_tokens: None, ttft_ms: None, total_ms: 0, tokens_per_s: None, finish_reason: None },
            });
        } else {
            self.monitor.log("app", "error", text.clone());
            sink(StreamEvent::Error { message: text });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn stream(
        &self,
        chat_id: &str,
        model: &ModelInfo,
        system: Option<String>,
        history: Vec<Turn>,
        user_text: String,
        user_msg: &Message,
        sink: &(impl Fn(StreamEvent) + Send + Sync),
        cancel: &CancellationToken,
    ) -> Result<(), ChatError> {
        let asst = self.store.lock().unwrap().append_message(
            chat_id,
            Some(&user_msg.id),
            Role::Assistant,
            "",
            MsgStatus::Streaming,
            Some(&model.id),
        )?;
        let req = self.monitor.start_request("chat", "/v1/chat/completions", &model.name);

        // 1. The engine: start or switch it, or wait for a start already under way.
        let mref = model_ref(model);
        let port = tokio::select! {
            r = self.supervisor.ensure_model(&mref) => match r {
                Ok(p) => p,
                Err(e) => return self.finish_early(&asst.id, &req, sink, MsgStatus::Error, e.to_string()),
            },
            _ = cancel.cancelled() => return self.finish_early(&asst.id, &req, sink, MsgStatus::Stopped, String::new()),
        };

        // 2. The request, with one retry when the engine says the context is full.
        let (mut kept, mut dropped) = budget::select_history(&history, self.settings.history_budget);
        let mut forced_max: Option<u32> = None;
        let mut retried = false;
        let t0 = Instant::now();
        let resp = loop {
            let mut messages: Vec<(&str, String)> = Vec::new();
            if let Some(s) = &system {
                messages.push(("system", s.clone()));
            }
            messages.extend(kept.iter().map(|t| (t.role, t.content.clone())));
            messages.push(("user", user_text.clone()));
            let estimate: u32 = messages.iter().map(|(_, c)| budget::estimate_tokens(c)).sum();
            let max_tokens = match forced_max.or_else(|| budget::max_tokens_for(model.ctx_engine, estimate, self.settings.reply_cap)) {
                Some(m) => m,
                None => {
                    return self.finish_early(&asst.id, &req, sink, MsgStatus::Error, "this chat no longer fits the context; start a new chat".into())
                }
            };
            let send = self.http.post(format!("http://127.0.0.1:{port}/v1/chat/completions")).json(&self.request_body(&messages, max_tokens)).send();
            let resp = tokio::select! {
                r = send => r,
                _ = cancel.cancelled() => return self.finish_early(&asst.id, &req, sink, MsgStatus::Stopped, String::new()),
            };
            let resp = match resp {
                Ok(r) => r,
                Err(e) => return self.finish_early(&asst.id, &req, sink, MsgStatus::Error, format!("could not reach the engine: {e}")),
            };
            if resp.status().is_success() {
                break resp;
            }
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let parsed: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
            let is_ctx = parsed["error"]["type"] == "context_length_exceeded";
            let counts = parsed["error"]["message"].as_str().and_then(budget::parse_context_error);
            match (is_ctx, counts, retried) {
                (true, Some((n, _m, c)), false) => {
                    retried = true;
                    // The engine counted n real tokens where we estimated `estimate`; scale our
                    // estimates by that ratio so the retry is sized in the engine's units.
                    let scale = n as f32 / estimate.max(1) as f32;
                    let real = |est: u32| (est as f32 * scale).ceil() as u32;
                    let mut est_after = estimate;
                    if c.saturating_sub(n + 8) < budget::MIN_REPLY {
                        while !kept.is_empty() && real(est_after) + budget::MIN_REPLY + 8 > c {
                            est_after = est_after.saturating_sub(budget::estimate_tokens(&kept.remove(0).content));
                            dropped += 1;
                        }
                        while !kept.is_empty() && kept[0].role != "user" {
                            est_after = est_after.saturating_sub(budget::estimate_tokens(&kept.remove(0).content));
                            dropped += 1;
                        }
                    }
                    let n_after = real(est_after).min(n);
                    forced_max = Some(c.saturating_sub(n_after + 8).clamp(budget::MIN_REPLY, self.settings.reply_cap));
                    self.monitor.log("app", "warn", format!("context full ({n} of {c} tokens); trimmed and retried once"));
                }
                _ => {
                    return self.finish_early(&asst.id, &req, sink, MsgStatus::Error, format!("the engine refused the request ({status}): {body}"))
                }
            }
        };

        // 3. The stream.
        sink(StreamEvent::Started {
            message_id: asst.id.clone(),
            user_message_id: user_msg.id.clone(),
            dropped_turns: dropped,
            request_id: req.clone(),
        });
        self.monitor.update_request(&req, |r| r.dropped_turns = dropped);
        let mut body = resp.bytes_stream();
        let mut parser = SseParser::new();
        let mut content = String::new();
        let mut first: Option<Instant> = None;
        let mut usage: Option<Usage> = None;
        let mut finish: Option<String> = None;
        let mut last_save = Instant::now();
        let outcome = loop {
            let next = tokio::select! {
                n = tokio::time::timeout(self.settings.liveness, body.next()) => n,
                _ = cancel.cancelled() => break Outcome::Stopped,
            };
            match next {
                Err(_) => break Outcome::Interrupted("the engine stopped responding".into()),
                Ok(None) => {
                    break if finish.is_some() { Outcome::Done } else { Outcome::Interrupted("the engine closed the stream before finishing".into()) }
                }
                Ok(Some(Err(e))) => break Outcome::Interrupted(format!("the stream broke: {e}")),
                Ok(Some(Ok(bytes))) => {
                    let mut done = false;
                    for ev in parser.push(&bytes) {
                        match ev {
                            SseEvent::Comment => {}
                            SseEvent::Done => done = true,
                            SseEvent::Data(json) => {
                                let Ok(c) = serde_json::from_str::<ChatChunk>(&json) else { continue };
                                if c.usage.is_some() {
                                    usage = c.usage;
                                }
                                for ch in c.choices {
                                    if let Some(t) = ch.delta.content.filter(|t| !t.is_empty()) {
                                        first.get_or_insert_with(Instant::now);
                                        content.push_str(&t);
                                        sink(StreamEvent::Token { text: t });
                                    }
                                    if ch.finish_reason.is_some() {
                                        finish = ch.finish_reason;
                                    }
                                }
                            }
                        }
                    }
                    if last_save.elapsed() >= self.settings.save_every {
                        let _ = self.store.lock().unwrap().update_content(&asst.id, &content);
                        last_save = Instant::now();
                    }
                    if done {
                        break Outcome::Done;
                    }
                }
            }
        };
        drop(body); // closing the connection is what cancels generation in the engine

        // 4. Save and report.
        let total_ms = t0.elapsed().as_millis() as u64;
        let ttft_ms = first.map(|f| f.duration_since(t0).as_millis() as u64);
        let completion = usage.map(|u| u.completion_tokens);
        let tokens_per_s = match (first, completion) {
            (Some(f), Some(n)) if n > 1 => Some((n - 1) as f32 / f.elapsed().as_secs_f32().max(0.001)),
            _ => None,
        };
        let stats = ReplyStats { prompt_tokens: usage.map(|u| u.prompt_tokens), completion_tokens: completion, ttft_ms, total_ms, tokens_per_s, finish_reason: finish };
        let (status, event, label) = match outcome {
            Outcome::Done => (MsgStatus::Done, StreamEvent::Finished { stats: stats.clone() }, "ok"),
            Outcome::Stopped => (MsgStatus::Stopped, StreamEvent::Stopped { stats: stats.clone() }, "stopped"),
            Outcome::Interrupted(reason) => {
                self.monitor.log("app", "warn", format!("reply interrupted: {reason}"));
                (MsgStatus::Interrupted, StreamEvent::Interrupted { reason }, "interrupted")
            }
        };
        {
            let s = self.store.lock().unwrap();
            s.update_content(&asst.id, &content)?;
            s.finish_message(
                &asst.id,
                status,
                &MsgStats { prompt_tokens: stats.prompt_tokens, completion_tokens: completion, ttft_ms, total_ms: Some(total_ms), error: None },
            )?;
        }
        self.monitor.update_request(&req, |r| {
            r.prompt_tokens = stats.prompt_tokens;
            r.completion_tokens = completion;
            r.ttft_ms = ttft_ms;
            r.total_ms = Some(total_ms);
            r.tokens_per_s = tokens_per_s;
            r.status = label.into();
        });
        sink(event);
        Ok(())
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test --features fake-engine --test chat_flow`
Expected: 9 passed. Then the whole suite: `cargo test --features fake-engine` — all green.

- [ ] **Step 5: Commit**

```bash
git add desktop/src-tauri desktop/src/ipc/bindings
git commit -m "desktop: chat service -- stream, stop, regenerate, crash and context handling (task 10)"
```

---

### Task 11: Commands, bindings and app wiring

**Files:**
- Create: `desktop/src-tauri/src/paths.rs`, `desktop/src-tauri/src/commands.rs`
- Modify: `desktop/src-tauri/src/lib.rs` (replace), `desktop/src-tauri/Cargo.toml` (single-instance plugin)
- Generated: `desktop/src/ipc/bindings/*.ts`

**Interfaces:**
- Consumes: every service from Tasks 3–10.
- Produces:
  - `paths::{data_dir(), log_dir(), downloads_dir(), engine_exe()}`; overrides: `LORREN_DATA_DIR`, `LORREN_ENGINE_EXE`
  - `commands::AppState { store, library, monitor, supervisor, chat, started_ms }`
  - DTOs (ts-rs): `MemoryInfo { app_bytes, engine_bytes, used_bytes, total_bytes }`, `Overview { engine, memory, app_version, started_ms, request_count }`, `MessageView { message, version, versions, sibling_ids }`, `ChatView { chat, messages }`
  - Commands (JS name → args): `overview()`, `models()`, `chats()`, `chat({chatId})`, `new_chat({modelId})`, `send({chatId, text, onEvent: Channel<StreamEvent>})`, `regenerate({chatId, onEvent})`, `stop({chatId})`, `set_chat_model({chatId, modelId})`, `select_version({chatId, messageId})`, `requests()`, `logs({after})`
  - Event `engine-status` with an `EngineStatus` payload on every engine state change
  - `commands::chat_view(&Store, chat_id) -> rusqlite::Result<ChatView>`, `commands::memory_info(engine_pid) -> MemoryInfo`

- [ ] **Step 1: Add the plugin**

```bash
cd desktop/src-tauri && cargo add tauri-plugin-single-instance
```

- [ ] **Step 2: Write the failing tests**

`desktop/src-tauri/src/paths.rs`, tests first:

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn overrides_win_and_defaults_are_under_appdata() {
        std::env::set_var("LORREN_ENGINE_EXE", r"C:\x\fake-engine.exe");
        assert_eq!(super::engine_exe(), std::path::PathBuf::from(r"C:\x\fake-engine.exe"));
        std::env::remove_var("LORREN_ENGINE_EXE");
        assert!(super::engine_exe().ends_with("lorren-engine.exe"));
        assert!(super::data_dir().ends_with("Lorren") || std::env::var("LORREN_DATA_DIR").is_ok());
    }
}
```

`desktop/src-tauri/src/commands.rs`, tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{MsgStatus, Role};

    #[test]
    fn a_regenerated_reply_shows_as_version_two_of_two() {
        let s = Store::open_in_memory().unwrap();
        let chat = s.create_chat("m", None).unwrap();
        let u = s.append_message(&chat.id, None, Role::User, "hi", MsgStatus::Done, None).unwrap();
        let a = s.append_message(&chat.id, Some(&u.id), Role::Assistant, "one", MsgStatus::Done, None).unwrap();
        let b = s.append_message(&chat.id, Some(&u.id), Role::Assistant, "two", MsgStatus::Done, None).unwrap();
        let v = chat_view(&s, &chat.id).unwrap();
        assert_eq!(v.messages.len(), 2);
        let last = &v.messages[1];
        assert_eq!((last.version, last.versions), (2, 2));
        assert_eq!(last.sibling_ids, vec![a.id, b.id]);
        assert_eq!(v.messages[0].versions, 1);
    }

    #[test]
    fn memory_info_counts_this_process() {
        let m = memory_info(None);
        assert!(m.app_bytes > 0 && m.total_bytes > m.used_bytes && m.engine_bytes == 0);
    }
}
```

Run: `cargo test paths commands` — Expected: compile errors.

- [ ] **Step 3: Implement the paths**

Above the tests in `paths.rs`:

```rust
//! Where things live (spec §3.5), with environment overrides for tests and the e2e run.

use std::path::PathBuf;

fn env_path(k: &str) -> Option<PathBuf> {
    std::env::var(k).ok().filter(|v| !v.is_empty()).map(PathBuf::from)
}

fn known(k: &str) -> PathBuf {
    env_path(k).unwrap_or_else(std::env::temp_dir)
}

/// %APPDATA%\Lorren: the database.
pub fn data_dir() -> PathBuf {
    env_path("LORREN_DATA_DIR").unwrap_or_else(|| known("APPDATA").join("Lorren"))
}

/// %LOCALAPPDATA%\Lorren\logs, or <LORREN_DATA_DIR>\logs.
pub fn log_dir() -> PathBuf {
    match env_path("LORREN_DATA_DIR") {
        Some(d) => d.join("logs"),
        None => known("LOCALAPPDATA").join("Lorren").join("logs"),
    }
}

/// %USERPROFILE%\Lorren\models: downloaded models (milestone 4 writes here).
pub fn downloads_dir() -> PathBuf {
    known("USERPROFILE").join("Lorren").join("models")
}

/// The engine sidecar: next to lorren.exe, where Tauri copies `externalBin`.
pub fn engine_exe() -> PathBuf {
    env_path("LORREN_ENGINE_EXE").unwrap_or_else(|| {
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("lorren-engine.exe")))
            .unwrap_or_else(|| PathBuf::from("lorren-engine.exe"))
    })
}
```

- [ ] **Step 4: Implement the commands**

Above the tests in `commands.rs`:

```rust
//! The Tauri commands the UI calls (spec §3.8): thin shells over the services.

use crate::chat::{model_ref, ChatService, StreamEvent};
use crate::engine::memory;
use crate::engine::supervisor::{EngineStatus, Supervisor};
use crate::library::ModelInfo;
use crate::monitor::{LogLine, Monitor, RequestRecord};
use crate::store::{Chat, Message, Store};
use serde::Serialize;
use std::sync::{Arc, Mutex, RwLock};
use tauri::ipc::Channel;
use tauri::State;
use ts_rs::TS;

pub struct AppState {
    pub store: Arc<Mutex<Store>>,
    pub library: Arc<RwLock<Vec<ModelInfo>>>,
    pub monitor: Arc<Monitor>,
    pub supervisor: Arc<Supervisor>,
    pub chat: Arc<ChatService>,
    pub started_ms: i64,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct MemoryInfo {
    /// Lorren plus its WebView2 processes (working set).
    #[ts(type = "number")]
    pub app_bytes: u64,
    #[ts(type = "number")]
    pub engine_bytes: u64,
    #[ts(type = "number")]
    pub used_bytes: u64,
    #[ts(type = "number")]
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct Overview {
    pub engine: EngineStatus,
    pub memory: MemoryInfo,
    pub app_version: String,
    #[ts(type = "number")]
    pub started_ms: i64,
    pub request_count: u32,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct MessageView {
    pub message: Message,
    /// 1-based position among its siblings (‹ version/versions ›).
    pub version: u32,
    pub versions: u32,
    pub sibling_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct ChatView {
    pub chat: Chat,
    pub messages: Vec<MessageView>,
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

pub fn memory_info(engine_pid: Option<u32>) -> MemoryInfo {
    let total = memory::total_physical_bytes();
    MemoryInfo {
        app_bytes: memory::tree_working_set(std::process::id(), engine_pid),
        engine_bytes: engine_pid.and_then(memory::process_memory).map(|(ws, _)| ws).unwrap_or(0),
        used_bytes: total.saturating_sub(memory::available_physical_bytes()),
        total_bytes: total,
    }
}

pub fn chat_view(store: &Store, chat_id: &str) -> rusqlite::Result<ChatView> {
    let chat = store.get_chat(chat_id)?;
    let mut messages = Vec::new();
    for m in store.active_path(chat_id)? {
        let sibs = store.siblings(&m.id)?;
        let version = sibs.iter().position(|s| s.id == m.id).map(|i| i as u32 + 1).unwrap_or(1);
        messages.push(MessageView {
            versions: sibs.len() as u32,
            version,
            sibling_ids: sibs.into_iter().map(|s| s.id).collect(),
            message: m,
        });
    }
    Ok(ChatView { chat, messages })
}

#[tauri::command]
pub fn overview(state: State<'_, AppState>) -> Overview {
    let engine = state.supervisor.status();
    Overview {
        memory: memory_info(engine.pid),
        engine,
        app_version: env!("CARGO_PKG_VERSION").into(),
        started_ms: state.started_ms,
        request_count: state.monitor.requests().len() as u32,
    }
}

#[tauri::command]
pub fn models(state: State<'_, AppState>) -> Vec<ModelInfo> {
    state.library.read().unwrap().clone()
}

#[tauri::command]
pub fn chats(state: State<'_, AppState>) -> Result<Vec<Chat>, String> {
    state.store.lock().unwrap().list_chats().map_err(err)
}

#[tauri::command]
pub fn chat(state: State<'_, AppState>, chat_id: String) -> Result<ChatView, String> {
    chat_view(&state.store.lock().unwrap(), &chat_id).map_err(err)
}

#[tauri::command]
pub fn new_chat(state: State<'_, AppState>, model_id: Option<String>) -> Result<Chat, String> {
    let model = {
        let lib = state.library.read().unwrap();
        model_id
            .and_then(|id| lib.iter().find(|m| m.id == id && m.runnable).cloned())
            .or_else(|| lib.iter().find(|m| m.is_default).cloned())
    }
    .ok_or("there is no runnable model in the library")?;
    state.store.lock().unwrap().create_chat(&model.id, None).map_err(err)
}

#[tauri::command]
pub async fn send(state: State<'_, AppState>, chat_id: String, text: String, on_event: Channel<StreamEvent>) -> Result<(), String> {
    state.chat.send(&chat_id, &text, move |e| {
        let _ = on_event.send(e);
    })
    .await
    .map_err(err)
}

#[tauri::command]
pub async fn regenerate(state: State<'_, AppState>, chat_id: String, on_event: Channel<StreamEvent>) -> Result<(), String> {
    state.chat.regenerate(&chat_id, move |e| {
        let _ = on_event.send(e);
    })
    .await
    .map_err(err)
}

#[tauri::command]
pub fn stop(state: State<'_, AppState>, chat_id: String) {
    state.chat.stop(&chat_id);
}

/// Switches the chat's model and starts the engine on it right away, so the switch is visible.
#[tauri::command]
pub async fn set_chat_model(state: State<'_, AppState>, chat_id: String, model_id: String) -> Result<(), String> {
    let model = state.library.read().unwrap().iter().find(|m| m.id == model_id).cloned().ok_or("no such model")?;
    if !model.runnable {
        return Err(format!("{} cannot run: {}", model.name, model.reason.clone().unwrap_or_default()));
    }
    state.store.lock().unwrap().set_chat_model(&chat_id, &model_id).map_err(err)?;
    let sup = state.supervisor.clone();
    tauri::async_runtime::spawn(async move {
        let _ = sup.ensure_model(&model_ref(&model)).await;
    });
    Ok(())
}

#[tauri::command]
pub fn select_version(state: State<'_, AppState>, chat_id: String, message_id: String) -> Result<(), String> {
    state.store.lock().unwrap().set_active_leaf(&chat_id, &message_id).map_err(err)
}

#[tauri::command]
pub fn requests(state: State<'_, AppState>) -> Vec<RequestRecord> {
    state.monitor.requests()
}

#[tauri::command]
pub fn logs(state: State<'_, AppState>, after: u64) -> Vec<LogLine> {
    state.monitor.logs_after(after)
}
```

- [ ] **Step 5: Wire the app**

Replace `desktop/src-tauri/src/lib.rs`:

```rust
pub mod budget;
pub mod chat;
pub mod commands;
pub mod engine;
pub mod gguf;
pub mod library;
pub mod monitor;
pub mod paths;
pub mod sse;
pub mod store;

use commands::AppState;
use std::sync::{Arc, Mutex, RwLock};
use tauri::{Emitter, Manager};

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.set_focus();
            }
        }))
        .setup(|app| {
            let monitor = monitor::Monitor::new(Some(paths::log_dir().join("lorren.log")));
            monitor.log("app", "info", format!("Lorren {} starting", env!("CARGO_PKG_VERSION")));
            std::fs::create_dir_all(paths::data_dir())?;
            let store = store::Store::open(&paths::data_dir().join("lorren.db"))?;
            let models = library::scan(&library::LibraryConfig::from_env(paths::downloads_dir()));
            monitor.log(
                "app",
                "info",
                format!("library: {} models, {} runnable", models.len(), models.iter().filter(|m| m.runnable).count()),
            );
            let default_model = models.iter().find(|m| m.is_default).cloned();
            let library = Arc::new(RwLock::new(models));
            let supervisor = engine::supervisor::Supervisor::new(
                engine::supervisor::EngineConfig::defaults(paths::engine_exe()),
                monitor.clone(),
            );
            let store = Arc::new(Mutex::new(store));
            let chat = chat::ChatService::new(store.clone(), supervisor.clone(), library.clone(), monitor.clone(), chat::ChatSettings::default());
            app.manage(AppState { store, library, monitor, supervisor: supervisor.clone(), chat, started_ms: store::now_ms() });

            let handle = app.handle().clone();
            let mut rx = supervisor.subscribe();
            tauri::async_runtime::spawn(async move {
                while rx.changed().await.is_ok() {
                    let status = rx.borrow().clone();
                    let _ = handle.emit("engine-status", status);
                }
            });
            // Start the engine on the default model so the first reply is quick.
            if let Some(m) = default_model {
                tauri::async_runtime::spawn(async move {
                    let _ = supervisor.ensure_model(&chat::model_ref(&m)).await;
                });
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::overview,
            commands::models,
            commands::chats,
            commands::chat,
            commands::new_chat,
            commands::send,
            commands::regenerate,
            commands::stop,
            commands::set_chat_model,
            commands::select_version,
            commands::requests,
            commands::logs
        ])
        .run(tauri::generate_context!())
        .expect("error while running Lorren");
}
```

- [ ] **Step 6: Run the tests and generate the bindings**

```bash
cargo test --features fake-engine
ls ../src/ipc/bindings
```

Expected: all tests pass; the bindings directory holds `ChatView.ts`, `EngineState.ts`, `EngineStatus.ts`, `LogLine.ts`, `MemoryInfo.ts`, `Message.ts`, `MessageView.ts`, `ModelInfo.ts`, `MsgStats.ts`, `MsgStatus.ts`, `Overview.ts`, `ReplyStats.ts`, `RequestRecord.ts`, `Role.ts`, `StreamEvent.ts`, `Chat.ts`.

- [ ] **Step 7: Smoke-run the app against the fake engine**

```bash
cd "/m/New Drive/Desktop/Programming/stoke-lorren/desktop/src-tauri"
cargo build --features fake-engine --bin fake-engine
./target/debug/fake-engine.exe write-fixtures ../e2e/.work/models
export LORREN_ENGINE_EXE="$PWD/target/debug/fake-engine.exe"
export LORREN_MODEL_DIRS="$PWD/../e2e/.work/models"
export LORREN_OLLAMA_DIR=none
export LORREN_DATA_DIR="$PWD/../e2e/.work/data"
cd .. && timeout 60 npx tauri dev
```

Close the window after it opens. Expected in `e2e/.work/data/logs/lorren.log`: `library: 4 models, 3 runnable`, then `engine serving file:...qwen3-4b-instruct-2507-q4_k_m.gguf on port ... after ... s`.

- [ ] **Step 8: Commit**

```bash
git add desktop/src-tauri desktop/src/ipc/bindings
git commit -m "desktop: commands, TypeScript bindings and app wiring (task 11)"
```

---

### Task 12: UI foundation — theme, fonts, top bar, status bar

**Files:**
- Create: `desktop/scripts/fetch-fonts.py`, `desktop/src/fonts/*` (downloaded), `desktop/src/fonts.css`, `desktop/src/theme.css`, `desktop/src/format.ts`, `desktop/src/ipc/api.ts`, `desktop/src/components/TopBar.tsx`, `desktop/src/components/StatusBar.tsx`, `desktop/src/components/Placeholder.tsx`, `desktop/src/__tests__/testApi.ts`
- Modify: `desktop/src/App.tsx` (replace), `desktop/src/__tests__/App.test.tsx` (replace)

**Interfaces:**
- Consumes: the bindings from Task 11 (`Overview`, `EngineStatus`, `ModelInfo`, `Chat`, `ChatView`, `StreamEvent`, `RequestRecord`, `LogLine`).
- Produces:
  - `api` (in `src/ipc/api.ts`): `overview()`, `models()`, `chats()`, `chat(chatId)`, `newChat(modelId | null)`, `send(chatId, text, onEvent)`, `regenerate(chatId, onEvent)`, `stop(chatId)`, `setChatModel(chatId, modelId)`, `selectVersion(chatId, messageId)`, `requests()`, `logs(after)`, `onEngineStatus(cb) -> Promise<UnlistenFn>`
  - `TABS`, `type Tab`, `TopBar({tab, onTab, overview})`, `modelName(path)` from `components/TopBar.tsx`
  - `StatusBar({overview, children?})`, `Placeholder({tab})`
  - `format.ts`: `mb(bytes)`, `gb(bytes)`, `secs(ms)`, `clock(ms)`, `since(startMs)`, `commas(n)`
  - CSS classes in `theme.css` used by Tasks 13–15: `app content col left center right head title section fact k v btn accent link pill bar scroll placeholder table-head table-row selected mono num cap ell dim dot`
  - `testApi.ts`: `overview` fixture and `mockApi()` for component tests

- [ ] **Step 1: Fetch the fonts**

`desktop/scripts/fetch-fonts.py`:

```python
"""Downloads the design's fonts (OFL) into src/fonts: Archivo 400/500/600 and JetBrains Mono
400/500 as the static TrueType files Google Fonts serves, plus each family's OFL text."""
import pathlib
import re
import urllib.request

OUT = pathlib.Path("src/fonts")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "curl/8.0"}  # a non-browser agent is served TrueType URLs


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


for family, stem, weights in (("Archivo", "Archivo", (400, 500, 600)), ("JetBrains+Mono", "JetBrainsMono", (400, 500))):
    css = get(f"https://fonts.googleapis.com/css2?family={family}:wght@{';'.join(map(str, weights))}").decode()
    found = re.findall(r"font-weight:\s*(\d+);.*?url\((https://[^)]+\.ttf)\)", css, re.S)
    assert len(found) == len(weights), (family, found)
    for w, url in found:
        (OUT / f"{stem}-{w}.ttf").write_bytes(get(url))
        print("font", stem, w)
for name, path in (("Archivo", "ofl/archivo/OFL.txt"), ("JetBrainsMono", "ofl/jetbrainsmono/OFL.txt")):
    (OUT / f"OFL-{name}.txt").write_bytes(get(f"https://raw.githubusercontent.com/google/fonts/main/{path}"))
    print("licence", name)
```

```bash
cd "/m/New Drive/Desktop/Programming/stoke-lorren/desktop"
python scripts/fetch-fonts.py
ls src/fonts
```

Expected: five `.ttf` files and two `OFL-*.txt` files.

- [ ] **Step 2: Write the failing tests**

`desktop/src/__tests__/testApi.ts`:

```ts
import { vi } from "vitest";
import type { Overview } from "../ipc/bindings/Overview";

export const overview: Overview = {
  engine: {
    state: "serving",
    model_id: "file:q",
    model_path: "M:\\New Drive\\Desktop\\Programming\\stoke\\models\\Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    port: 50123,
    pid: 42,
    ctx: 4096,
    seqs: 2,
    threads: null,
    kv: "q8",
    load_ms: 2160,
    restarts: 0,
    last_error: null,
  },
  memory: { app_bytes: 150 * 1048576, engine_bytes: 2.4 * 1073741824, used_bytes: 10 * 1073741824, total_bytes: 15.6 * 1073741824 },
  app_version: "0.0.1",
  started_ms: Date.now() - 65 * 60000,
  request_count: 3,
};

export function mockApi() {
  return {
    overview: vi.fn(async () => overview),
    models: vi.fn(async () => []),
    chats: vi.fn(async () => []),
    chat: vi.fn(),
    newChat: vi.fn(),
    send: vi.fn(async () => {}),
    regenerate: vi.fn(async () => {}),
    stop: vi.fn(async () => {}),
    setChatModel: vi.fn(async () => {}),
    selectVersion: vi.fn(async () => {}),
    requests: vi.fn(async () => []),
    logs: vi.fn(async () => []),
    onEngineStatus: vi.fn(async () => () => {}),
  };
}
```

Replace `desktop/src/__tests__/App.test.tsx`:

```tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { App } from "../App";

describe("App", () => {
  it("shows the wordmark, the six tabs and the engine state", async () => {
    render(<App />);
    expect(screen.getByText("lorren")).toBeTruthy();
    expect(screen.getAllByRole("tab").map((t) => t.textContent)).toEqual(["Chat", "Swarm", "Models", "Requests", "Logs", "Settings"]);
    expect(await screen.findByText("serving")).toBeTruthy();
    expect(screen.getByText("Qwen3-4B-Instruct-2507-Q4_K_M")).toBeTruthy();
    expect(screen.getByText(/engine 2\.4 GB/)).toBeTruthy();
  });

  it("switches tabs", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Swarm" }));
    expect(screen.getByRole("tab", { name: "Swarm" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getByText(/milestone 3/)).toBeTruthy();
  });
});
```

Run: `npx vitest run` — Expected: FAIL (`../ipc/api`, `./components/TopBar` do not exist).

- [ ] **Step 3: The typed API**

`desktop/src/ipc/api.ts`:

```ts
import { Channel, invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { Chat } from "./bindings/Chat";
import type { ChatView } from "./bindings/ChatView";
import type { EngineStatus } from "./bindings/EngineStatus";
import type { LogLine } from "./bindings/LogLine";
import type { ModelInfo } from "./bindings/ModelInfo";
import type { Overview } from "./bindings/Overview";
import type { RequestRecord } from "./bindings/RequestRecord";
import type { StreamEvent } from "./bindings/StreamEvent";

function channel(onEvent: (e: StreamEvent) => void): Channel<StreamEvent> {
  const ch = new Channel<StreamEvent>();
  ch.onmessage = onEvent;
  return ch;
}

export const api = {
  overview: () => invoke<Overview>("overview"),
  models: () => invoke<ModelInfo[]>("models"),
  chats: () => invoke<Chat[]>("chats"),
  chat: (chatId: string) => invoke<ChatView>("chat", { chatId }),
  newChat: (modelId: string | null) => invoke<Chat>("new_chat", { modelId }),
  send: (chatId: string, text: string, onEvent: (e: StreamEvent) => void) =>
    invoke<void>("send", { chatId, text, onEvent: channel(onEvent) }),
  regenerate: (chatId: string, onEvent: (e: StreamEvent) => void) =>
    invoke<void>("regenerate", { chatId, onEvent: channel(onEvent) }),
  stop: (chatId: string) => invoke<void>("stop", { chatId }),
  setChatModel: (chatId: string, modelId: string) => invoke<void>("set_chat_model", { chatId, modelId }),
  selectVersion: (chatId: string, messageId: string) => invoke<void>("select_version", { chatId, messageId }),
  requests: () => invoke<RequestRecord[]>("requests"),
  logs: (after: number) => invoke<LogLine[]>("logs", { after }),
  onEngineStatus: (cb: (s: EngineStatus) => void): Promise<UnlistenFn> =>
    listen<EngineStatus>("engine-status", (e) => cb(e.payload)),
};

export type Api = typeof api;
```

- [ ] **Step 4: Formatting helpers, styles and fonts**

`desktop/src/format.ts`:

```ts
export const mb = (b: number) => `${Math.round(b / 1048576).toLocaleString("en-US")} MB`;
export const gb = (b: number) => `${(b / 1073741824).toFixed(1)} GB`;
export const secs = (ms: number | null | undefined) =>
  ms == null ? "—" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
export const clock = (ms: number) => new Date(ms).toLocaleTimeString("en-GB", { hour12: false });
export const since = (startMs: number, now = Date.now()) => {
  const m = Math.max(0, Math.floor((now - startMs) / 60000));
  return m < 60 ? `${m}m` : `${Math.floor(m / 60)}h ${m % 60}m`;
};
export const commas = (n: number) => n.toLocaleString("en-US");
```

`desktop/src/fonts.css`:

```css
@font-face { font-family: "Archivo"; src: url("./fonts/Archivo-400.ttf") format("truetype"); font-weight: 400; }
@font-face { font-family: "Archivo"; src: url("./fonts/Archivo-500.ttf") format("truetype"); font-weight: 500; }
@font-face { font-family: "Archivo"; src: url("./fonts/Archivo-600.ttf") format("truetype"); font-weight: 600; }
@font-face { font-family: "JetBrains Mono"; src: url("./fonts/JetBrainsMono-400.ttf") format("truetype"); font-weight: 400; }
@font-face { font-family: "JetBrains Mono"; src: url("./fonts/JetBrainsMono-500.ttf") format("truetype"); font-weight: 500; }
```

`desktop/src/theme.css` (tokens and shared pieces from `design/ControlRoom.dc.html`):

```css
:root {
  color-scheme: dark;
  --bg: oklch(0.15 0.006 255); --text: oklch(0.95 0.004 255);
  --l27: oklch(0.27 0.008 255); --l22: oklch(0.22 0.008 255); --l20: oklch(0.20 0.008 255);
  --l24: oklch(0.24 0.008 255); --l30: oklch(0.30 0.008 255); --t21: oklch(0.21 0.008 255);
  --sel: oklch(0.19 0.008 255); --field: oklch(0.185 0.006 255); --code: oklch(0.175 0.006 255);
  --d50: oklch(0.50 0.008 255); --d52: oklch(0.52 0.008 255); --d55: oklch(0.55 0.008 255);
  --d58: oklch(0.58 0.008 255); --d60: oklch(0.60 0.008 255); --d66: oklch(0.66 0.008 255);
  --d68: oklch(0.68 0.008 255); --d70: oklch(0.70 0.008 255); --d72: oklch(0.72 0.008 255);
  --d74: oklch(0.74 0.008 255); --d80: oklch(0.80 0.008 255); --d86: oklch(0.86 0.008 255);
  --accent: oklch(0.76 0.09 70); --shared: oklch(0.70 0.07 250); --flame: oklch(0.66 0.09 30);
  --sans: "Archivo", ui-sans-serif, system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, monospace;
}
html, body, #root { margin: 0; height: 100%; background: var(--bg); color: var(--text); }
body { font-family: var(--sans); font-size: 13px; overflow: hidden; }
button, input, textarea, select { font-family: inherit; }
::selection { background: oklch(0.70 0.07 250 / 0.35); }

.app { display: flex; flex-direction: column; height: 100vh; }
.content { flex: 1; min-height: 0; display: flex; }
.mono { font-family: var(--mono); }
.num { font-variant-numeric: tabular-nums; }
.cap { font-family: var(--mono); font-size: 10px; letter-spacing: 0.14em; color: var(--d60); }
.ell { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dim { color: var(--d55); }

.topbar { height: 54px; flex-shrink: 0; border-bottom: 1px solid var(--l27); display: flex; align-items: stretch; }
.brand { display: flex; align-items: center; gap: 10px; padding: 0 22px; border-right: 1px solid var(--l22); }
.wordmark { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
.version { font-family: var(--mono); font-size: 10px; color: var(--d60); border: 1px solid var(--l30); padding: 2px 5px; }
.tabs { display: flex; align-items: stretch; }
.tab { display: flex; align-items: center; padding: 0 18px; font-size: 12.5px; cursor: pointer; background: none;
       border: 0; border-bottom: 2px solid transparent; color: var(--d55); }
.tab:hover { color: var(--d80); }
.tab[aria-selected="true"] { color: var(--text); border-bottom-color: var(--accent); }
.engine { margin-left: auto; display: flex; align-items: center; gap: 22px; padding: 0 22px;
          font-family: var(--mono); font-size: 11px; color: var(--d68); min-width: 0; }
.dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin-right: 7px; background: var(--d50); }
.dot.serving { background: var(--accent); }
.dot.starting, .dot.restarting { background: var(--shared); }
.dot.failed { background: var(--flame); }

.statusbar { height: 30px; flex-shrink: 0; border-top: 1px solid var(--l27); display: flex; align-items: center;
             gap: 20px; padding: 0 22px; font-family: var(--mono); font-size: 10px; color: var(--d52); }
.statusbar .spacer { flex: 1; }

.col { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.left { width: 260px; flex-shrink: 0; border-right: 1px solid var(--l22); }
.center { flex: 1; border-right: 1px solid var(--l27); }
.right { width: 396px; flex-shrink: 0; overflow-y: auto; }
.head { padding: 13px 20px 11px; border-bottom: 1px solid var(--l27); display: flex; align-items: baseline; gap: 10px; }
.title { font-size: 13px; font-weight: 600; }
.section { padding: 16px 20px; border-bottom: 1px solid var(--l22); display: flex; flex-direction: column; gap: 10px; }
.fact { display: flex; justify-content: space-between; gap: 14px; font-family: var(--mono); font-size: 11px; }
.fact .k { color: var(--d58); flex-shrink: 0; }
.fact .v { text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.btn { font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em; cursor: pointer; background: none;
       border: 1px solid var(--l30); color: var(--d72); padding: 9px 14px; }
.btn.accent { border-color: var(--accent); color: var(--accent); }
.btn.accent:hover { background: oklch(0.76 0.09 70 / 0.12); }
.btn:disabled { opacity: 0.4; cursor: default; }
.link { background: none; border: 0; padding: 0; cursor: pointer; color: var(--d55); font-family: var(--mono);
        font-size: 10px; letter-spacing: 0.08em; }
.link:hover { color: var(--text); }
.pill { font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.08em; border: 1px solid currentColor; padding: 3px 6px; text-align: center; }
.bar { height: 4px; background: var(--l24); position: relative; }
.bar > i { position: absolute; left: 0; top: 0; bottom: 0; background: var(--accent); }
.scroll { overflow-y: auto; scrollbar-width: thin; scrollbar-color: var(--l30) transparent; }
.placeholder { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; }
.table-head { display: grid; gap: 14px; padding: 14px 22px 8px; font-family: var(--mono); font-size: 10px;
              letter-spacing: 0.1em; color: var(--d55); border-bottom: 1px solid var(--l27); }
.table-row { display: grid; gap: 14px; align-items: center; padding: 0 22px; border-bottom: 1px solid var(--l20); }
.table-row.selected { background: var(--sel); }
```

- [ ] **Step 5: The components and the App**

`desktop/src/components/TopBar.tsx`:

```tsx
import { gb, mb } from "../format";
import type { Overview } from "../ipc/bindings/Overview";

export const TABS = ["Chat", "Swarm", "Models", "Requests", "Logs", "Settings"] as const;
export type Tab = (typeof TABS)[number];

export function modelName(path: string | null | undefined): string {
  if (!path) return "no model";
  const file = path.split(/[\\/]/).pop() ?? path;
  return file.replace(/\.gguf$/i, "");
}

export function TopBar({ tab, onTab, overview }: { tab: Tab; onTab: (t: Tab) => void; overview: Overview | null }) {
  const e = overview?.engine;
  return (
    <header className="topbar">
      <div className="brand">
        <span className="wordmark">lorren</span>
        <span className="version">{overview?.app_version ?? "…"}</span>
      </div>
      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t} role="tab" className="tab" aria-selected={t === tab} onClick={() => onTab(t)}>
            {t}
          </button>
        ))}
      </nav>
      <div className="engine" aria-label="engine">
        <span>
          <i className={`dot ${e?.state ?? "stopped"}`} />
          {e?.state ?? "starting"}
        </span>
        <span className="ell" style={{ maxWidth: 260 }}>{modelName(e?.model_path)}</span>
        <span className="dim">t={e?.threads ?? "auto"}</span>
        <span className="dim">kv {e?.kv ?? "q8"}</span>
        {overview && (
          <span className="dim">
            app {mb(overview.memory.app_bytes)} · engine {gb(overview.memory.engine_bytes)} ·{" "}
            {gb(overview.memory.used_bytes)}/{gb(overview.memory.total_bytes)}
          </span>
        )}
      </div>
    </header>
  );
}
```

`desktop/src/components/StatusBar.tsx`:

```tsx
import type { ReactNode } from "react";
import { secs, since } from "../format";
import type { Overview } from "../ipc/bindings/Overview";

export function StatusBar({ overview, children }: { overview: Overview | null; children?: ReactNode }) {
  const e = overview?.engine;
  return (
    <footer className="statusbar">
      <span>session {overview ? since(overview.started_ms) : "—"}</span>
      <span>{overview?.request_count ?? 0} requests</span>
      <span>{e?.restarts ?? 0} engine restarts</span>
      {e?.load_ms != null && <span>last load {secs(e.load_ms)}</span>}
      {e?.last_error && (
        <span className="ell" style={{ color: "var(--flame)", maxWidth: 520 }}>{e.last_error}</span>
      )}
      <span className="spacer" />
      {children}
      <span>engine frozen b1ee6b3 · ctx {e?.ctx ?? 4096} · {e?.seqs ?? 2} parallel</span>
    </footer>
  );
}
```

`desktop/src/components/Placeholder.tsx`:

```tsx
export function Placeholder({ tab }: { tab: string }) {
  const text =
    tab === "Swarm"
      ? "The Swarm view arrives in milestone 3. Its engine panels need telemetry the engine adds after m7."
      : "Settings arrive in milestone 3. This version runs on the spec's defaults.";
  return (
    <div className="placeholder">
      <span style={{ fontSize: 15, fontWeight: 600 }}>{tab}</span>
      <span className="mono dim" style={{ fontSize: 11 }}>{text}</span>
    </div>
  );
}
```

Replace `desktop/src/App.tsx`:

```tsx
import { useEffect, useState, type ReactNode } from "react";
import "./fonts.css";
import "./theme.css";
import { Placeholder } from "./components/Placeholder";
import { StatusBar } from "./components/StatusBar";
import { TopBar, type Tab } from "./components/TopBar";
import { api } from "./ipc/api";
import type { Overview } from "./ipc/bindings/Overview";

export function App() {
  const [tab, setTab] = useState<Tab>("Chat");
  const [overview, setOverview] = useState<Overview | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = () => api.overview().then((o) => alive && setOverview(o)).catch(() => {});
    tick();
    const timer = setInterval(tick, 2000);
    const unlisten = api.onEngineStatus((engine) => setOverview((o) => (o ? { ...o, engine } : o)));
    return () => {
      alive = false;
      clearInterval(timer);
      unlisten.then((f) => f());
    };
  }, []);

  return (
    <div className="app">
      <TopBar tab={tab} onTab={setTab} overview={overview} />
      <main className="content">{renderTab(tab, overview)}</main>
      <StatusBar overview={overview} />
    </div>
  );
}

function renderTab(tab: Tab, _overview: Overview | null): ReactNode {
  return <Placeholder tab={tab} />;
}
```

- [ ] **Step 6: Run the tests and the type check**

```bash
npx vitest run
npx tsc --noEmit
```

Expected: 2 passed; no type errors (the bindings resolve).

- [ ] **Step 7: Commit**

```bash
git add desktop/scripts/fetch-fonts.py desktop/src
git commit -m "desktop: Control Room theme, bundled fonts, top bar and status bar (task 12)"
```

---

### Task 13: UI — the Chat tab

**Files:**
- Create: `desktop/src/chat/stream.ts`, `desktop/src/chat/Markdown.tsx`, `desktop/src/chat/ChatList.tsx`, `desktop/src/chat/MessageView.tsx`, `desktop/src/chat/Composer.tsx`, `desktop/src/chat/ReplyPanel.tsx`, `desktop/src/chat/ChatTab.tsx`, `desktop/src/chat/chat.css`
- Create: `desktop/src/__tests__/stream.test.ts`, `desktop/src/__tests__/Markdown.test.tsx`, `desktop/src/__tests__/ChatTab.test.tsx`
- Modify: `desktop/src/__tests__/testApi.ts` (fixtures), `desktop/src/App.tsx` (render the Chat tab), `desktop/package.json` (Markdown packages)

**Interfaces:**
- Consumes: `api` (Task 12: `chats`, `chat`, `models`, `newChat`, `send`, `regenerate`, `stop`, `setChatModel`, `selectVersion`); the bindings `Chat`, `ChatView`, `MessageView`, `Message`, `ModelInfo`, `Overview`, `ReplyStats`, `StreamEvent` (Tasks 5, 10, 11); `format.ts` (`secs`, `commas`); the CSS classes of `theme.css` (Task 12). `StreamEvent` is tagged by `kind`: `started | token | finished | stopped | interrupted | error` (Task 10). A failure before the reply starts arrives as `error` or `stopped` with no `started` before it. `api.send` resolves only when the reply has ended, and rejects with the host's message for `ChatError` (busy, too long, unknown or unrunnable model).
- Produces:
  - `stream.ts`: `type Live = { busy, messageId, text, dropped, stats, ended, error }`, `idle`, `begin()`, `applyEvent(s: Live, e: StreamEvent): Live`, `statsOf(m: Message): ReplyStats`
  - `Markdown({ text })`: GFM + highlighted code + KaTeX; raw HTML dropped; links and images never load anything
  - `ChatTab({ overview })`, and inside it `ChatList`, `MessageRow` (in `MessageView.tsx`), `Composer`, `ReplyPanel`
  - `testApi.ts`: `model(partial)`, `models`, `chat`, `msg(partial)`, `view`, `withTurn(view, userText, reply)`

- [ ] **Step 1: Add the Markdown packages**

```bash
cd "/m/New Drive/Desktop/Programming/stoke-lorren/desktop"
npm install react-markdown remark-gfm remark-math rehype-katex rehype-highlight katex highlight.js
```

- [ ] **Step 2: Add the fixtures**

Append to `desktop/src/__tests__/testApi.ts`:

```ts
import type { Chat } from "../ipc/bindings/Chat";
import type { ChatView } from "../ipc/bindings/ChatView";
import type { Message } from "../ipc/bindings/Message";
import type { ModelInfo } from "../ipc/bindings/ModelInfo";

export function model(p: Partial<ModelInfo>): ModelInfo {
  return {
    id: "file:x", name: "x", path: "M:\\New Drive\\ai_models\\x.gguf", source: "folder",
    size_bytes: 2_019_377_696, arch: "llama", quant: "Q4_K_M", size_label: "3B",
    n_layer: 28, n_head: 24, n_head_kv: 8, head_dim: 128, vocab: 128256,
    ctx_file: 131072, ctx_engine: 4096, kv_bytes_per_token: 60928, template: "llama3",
    runnable: true, reason: null, system_prompt: null, system_source: null, is_default: false,
    ...p,
  };
}

export const models: ModelInfo[] = [
  model({ id: "file:q", name: "Qwen3-4B-Instruct-2507-Q4_K_M", arch: "qwen3", template: "chatml", n_layer: 36,
          kv_bytes_per_token: 78336, size_bytes: 2_497_280_736, is_default: true }),
  model({ id: "file:n", name: "nexus", system_source: "Modelfile",
          system_prompt: "You are Nexus AI, a professional quantitative trader. You give concise, decisive trading verdicts." }),
  model({ id: "ollama:qwen2.5vl:3b", name: "qwen2.5vl:3b", source: "ollama", arch: "qwen2vl", runnable: false,
          reason: "architecture qwen2vl is not supported by the engine (llama and qwen3 only)" }),
];

const noStats = { prompt_tokens: null, completion_tokens: null, ttft_ms: null, total_ms: null, error: null };

export function msg(p: Partial<Message> & Pick<Message, "id" | "role" | "content">): Message {
  return { chat_id: "c1", parent_id: null, status: "done", model_id: null, stats: noStats, created_ms: 1, ...p };
}

export const chat: Chat = {
  id: "c1", title: "Long or wait?", model_id: "file:n", system_prompt: null,
  active_leaf_id: "a2", created_ms: 1, updated_ms: 2,
};

/** A chat whose reply was regenerated once: version 2 of 2 is showing. */
export const view: ChatView = {
  chat,
  messages: [
    { message: msg({ id: "u1", role: "user", content: "Long or wait?" }), version: 1, versions: 1, sibling_ids: ["u1"] },
    {
      message: msg({ id: "a2", role: "assistant", parent_id: "u1", model_id: "file:n", content: "**Verdict: HOLD.**",
                     stats: { prompt_tokens: 41, completion_tokens: 5, ttft_ms: 480, total_ms: 730, error: null } }),
      version: 2, versions: 2, sibling_ids: ["a1", "a2"],
    },
  ],
};

export function withTurn(base: ChatView, userText: string, reply: Message): ChatView {
  return {
    ...base,
    messages: [
      ...base.messages,
      { message: msg({ id: "u3", role: "user", parent_id: "a2", content: userText }), version: 1, versions: 1, sibling_ids: ["u3"] },
      { message: reply, version: 1, versions: 1, sibling_ids: [reply.id] },
    ],
  };
}
```

- [ ] **Step 3: Write the failing tests**

`desktop/src/__tests__/stream.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { applyEvent, begin, idle, statsOf } from "../chat/stream";
import { msg } from "./testApi";

const stats = { prompt_tokens: 50, completion_tokens: 4, ttft_ms: 212, total_ms: 600, tokens_per_s: 7.7, finish_reason: "stop" };

describe("applyEvent", () => {
  it("collects a reply from started to finished", () => {
    let s = begin();
    s = applyEvent(s, { kind: "started", message_id: "a3", user_message_id: "u3", dropped_turns: 2, request_id: "req_1" });
    s = applyEvent(s, { kind: "token", text: "Buy " });
    s = applyEvent(s, { kind: "token", text: "at ₹68,400" });
    expect(s).toMatchObject({ busy: true, messageId: "a3", text: "Buy at ₹68,400", dropped: 2 });
    s = applyEvent(s, { kind: "finished", stats });
    expect(s).toMatchObject({ busy: false, ended: "finished", stats });
  });

  it("keeps a refusal that arrives before the reply starts", () => {
    const s = applyEvent(begin(), { kind: "error", message: "the engine refused the request (400 Bad Request)" });
    expect(s).toMatchObject({ busy: false, messageId: null, ended: "error", error: "the engine refused the request (400 Bad Request)" });
  });

  it("ends on stopped and interrupted", () => {
    expect(applyEvent(begin(), { kind: "stopped", stats }).ended).toBe("stopped");
    const i = applyEvent(begin(), { kind: "interrupted", reason: "the engine stopped responding" });
    expect(i).toMatchObject({ busy: false, ended: "interrupted", error: "the engine stopped responding" });
    expect(idle.busy).toBe(false);
  });
});

describe("statsOf", () => {
  it("derives the decode speed from a saved reply", () => {
    const m = msg({ id: "a", role: "assistant", content: "x",
                    stats: { prompt_tokens: 41, completion_tokens: 21, ttft_ms: 500, total_ms: 1500, error: null } });
    expect(statsOf(m)).toMatchObject({ prompt_tokens: 41, completion_tokens: 21, ttft_ms: 500, total_ms: 1500, tokens_per_s: 20 });
  });
});
```

`desktop/src/__tests__/Markdown.test.tsx`:

```tsx
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Markdown } from "../chat/Markdown";

describe("Markdown", () => {
  it("renders GFM tables, highlighted code and maths", () => {
    const text = "**HOLD** at ₹68,400\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```rust\nfn main() {}\n```\n\n$x^2$";
    const { container } = render(<Markdown text={text} />);
    expect(container.querySelector("strong")?.textContent).toBe("HOLD");
    expect(container.querySelector("table")).not.toBeNull();
    expect(container.querySelector("pre code.hljs")).not.toBeNull();
    expect(container.querySelector(".katex")).not.toBeNull();
    expect(container.textContent).toContain("₹68,400");
  });

  it("never renders raw HTML, fetches an image or makes a link that navigates", () => {
    const text = '<img src=x onerror="alert(1)"><script>alert(2)</script>\n\n![chart](https://example.com/x.png) and [site](https://example.com)';
    const { container } = render(<Markdown text={text} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("a")).toBeNull();
    expect(container.textContent).toContain("[image: chart]");
    expect(container.textContent).toContain("site");
  });
});
```

`desktop/src/__tests__/ChatTab.test.tsx`:

```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { ChatTab } from "../chat/ChatTab";
import { api } from "../ipc/api";
import type { StreamEvent } from "../ipc/bindings/StreamEvent";
import { chat, models, msg, overview, view, withTurn } from "./testApi";

const started: StreamEvent = { kind: "started", message_id: "a3", user_message_id: "u3", dropped_turns: 0, request_id: "req_1" };
const done = { prompt_tokens: 60, completion_tokens: 4, ttft_ms: 212, total_ms: 650, tokens_per_s: 7.0, finish_reason: "stop" };

function type(text: string) {
  fireEvent.change(screen.getByLabelText("message"), { target: { value: text } });
  fireEvent.click(screen.getByText("SEND"));
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.models).mockResolvedValue(models);
  vi.mocked(api.chats).mockResolvedValue([chat]);
  vi.mocked(api.chat).mockResolvedValue(view);
});

describe("ChatTab", () => {
  it("opens the latest chat and renders its reply as Markdown, with its versions and stats", async () => {
    render(<ChatTab overview={overview} />);
    const verdict = await screen.findByText("Verdict: HOLD.");
    expect(verdict.tagName).toBe("STRONG");
    expect(screen.getByText("2/2")).toBeTruthy();
    expect(screen.getByText("480 ms")).toBeTruthy();
  });

  it("streams a reply into the window and then shows its stats", async () => {
    let release!: () => void;
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      onEvent(started);
      onEvent({ kind: "token", text: "Buy " });
      onEvent({ kind: "token", text: "at ₹68,400" });
      await new Promise<void>((r) => (release = r));
      onEvent({ kind: "finished", stats: done });
    });
    vi.mocked(api.chat)
      .mockResolvedValueOnce(view)
      .mockResolvedValueOnce(withTurn(view, "and now?", msg({ id: "a3", role: "assistant", parent_id: "u3", content: "", status: "streaming" })))
      .mockResolvedValue(withTurn(view, "and now?", msg({ id: "a3", role: "assistant", parent_id: "u3", content: "Buy at ₹68,400" })));
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    type("and now?");
    expect(await screen.findByText("Buy at ₹68,400")).toBeTruthy();
    expect(screen.getByText("STOP")).toBeTruthy();
    expect(api.send).toHaveBeenCalledWith("c1", "and now?", expect.any(Function));
    release();
    expect(await screen.findByText("212 ms")).toBeTruthy();
    expect(screen.getByText("SEND")).toBeTruthy();
  });

  it("asks the host to stop the reply", async () => {
    let end!: () => void;
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      onEvent(started);
      await new Promise<void>((r) => (end = r));
      onEvent({ kind: "stopped", stats: done });
    });
    vi.mocked(api.stop).mockImplementation(async () => end());
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    type("go");
    fireEvent.click(await screen.findByText("STOP"));
    await waitFor(() => expect(api.stop).toHaveBeenCalledWith("c1"));
    expect(await screen.findByText("SEND")).toBeTruthy();
  });

  it("shows that it is waiting for the engine before the reply starts", async () => {
    let go!: () => void;
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      await new Promise<void>((r) => (go = r));
      onEvent(started);
      onEvent({ kind: "finished", stats: done });
    });
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    type("hello");
    expect(await screen.findByText(/waiting for the engine/)).toBeTruthy();
    go();
    await waitFor(() => expect(screen.queryByText(/waiting for the engine/)).toBeNull());
  });

  it("shows a refusal that arrives before the reply starts", async () => {
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      onEvent({ kind: "error", message: "the engine refused the request (400 Bad Request)" });
    });
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    type("hello");
    expect((await screen.findByRole("alert")).textContent).toContain("the engine refused the request");
  });

  it("shows the host's refusal of a message longer than the context (Review Focus 5)", async () => {
    vi.mocked(api.send).mockRejectedValue("this message is about 6250 tokens, more than a 4096-token context can take");
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    type("word ".repeat(5000));
    expect((await screen.findByRole("alert")).textContent).toContain("more than a 4096-token context");
    expect(screen.getByText("SEND")).toBeTruthy();
  });

  it("marks where the history stops being sent", async () => {
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      onEvent({ ...started, dropped_turns: 1 } as StreamEvent);
      onEvent({ kind: "token", text: "ok" });
    });
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    type("next");
    expect(await screen.findByText(/1 earlier turn not sent/)).toBeTruthy();
  });

  it("switches reply versions and regenerates", async () => {
    render(<ChatTab overview={overview} />);
    await screen.findByText("2/2");
    fireEvent.click(screen.getByLabelText("previous version"));
    await waitFor(() => expect(api.selectVersion).toHaveBeenCalledWith("c1", "a1"));
    fireEvent.click(screen.getByText("REGENERATE"));
    await waitFor(() => expect(api.regenerate).toHaveBeenCalledWith("c1", expect.any(Function)));
  });

  it("switches the chat's model, and a model that can't run cannot be picked", async () => {
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    const option = screen.getByRole("option", { name: /qwen2\.5vl:3b/ }) as HTMLOptionElement;
    expect(option.disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("model"), { target: { value: "file:q" } });
    await waitFor(() => expect(api.setChatModel).toHaveBeenCalledWith("c1", "file:q"));
  });

  it("starts a new chat on the default model", async () => {
    vi.mocked(api.newChat).mockResolvedValue({ ...chat, id: "c2", title: "" });
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    fireEvent.click(screen.getByText("+ NEW"));
    await waitFor(() => expect(api.newChat).toHaveBeenCalledWith(null));
  });
});
```

Run: `npx vitest run` — Expected: FAIL, the `../chat/*` modules do not exist.

- [ ] **Step 4: The stream state**

`desktop/src/chat/stream.ts`:

```ts
import type { Message } from "../ipc/bindings/Message";
import type { ReplyStats } from "../ipc/bindings/ReplyStats";
import type { StreamEvent } from "../ipc/bindings/StreamEvent";

/** The reply being written in the open chat, as the UI sees it. */
export type Live = {
  /** From the moment Send is pressed until the reply ends. */
  busy: boolean;
  /** The assistant message, once the engine has accepted the request. */
  messageId: string | null;
  text: string;
  /** How many of the oldest turns were left out by the history budget. */
  dropped: number;
  stats: ReplyStats | null;
  ended: "finished" | "stopped" | "interrupted" | "error" | null;
  error: string | null;
};

export const idle: Live = { busy: false, messageId: null, text: "", dropped: 0, stats: null, ended: null, error: null };

export function begin(): Live {
  return { ...idle, busy: true };
}

export function applyEvent(s: Live, e: StreamEvent): Live {
  switch (e.kind) {
    case "started":
      return { ...s, busy: true, messageId: e.message_id, text: "", dropped: e.dropped_turns };
    case "token":
      return { ...s, text: s.text + e.text };
    case "finished":
      return { ...s, busy: false, stats: e.stats, ended: "finished" };
    case "stopped":
      return { ...s, busy: false, stats: e.stats, ended: "stopped" };
    case "interrupted":
      return { ...s, busy: false, ended: "interrupted", error: e.reason };
    case "error":
      return { ...s, busy: false, ended: "error", error: e.message };
  }
}

/** A saved reply's numbers in the shape of a live one. Speed counts the tokens after the first. */
export function statsOf(m: Message): ReplyStats {
  const s = m.stats;
  const gen = s.total_ms != null && s.ttft_ms != null ? s.total_ms - s.ttft_ms : null;
  const n = s.completion_tokens;
  return {
    prompt_tokens: s.prompt_tokens,
    completion_tokens: n,
    ttft_ms: s.ttft_ms,
    total_ms: s.total_ms ?? 0,
    tokens_per_s: n != null && n > 1 && gen != null && gen > 0 ? (n - 1) / (gen / 1000) : null,
    finish_reason: null,
  };
}
```

- [ ] **Step 5: Markdown**

`desktop/src/chat/Markdown.tsx`:

```tsx
import "highlight.js/styles/github-dark.css";
import "katex/dist/katex.min.css";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

/** Model output. Raw HTML is dropped (`skipHtml`), a link never navigates the app window, and an
 *  image is never fetched: the app shows no remote content (spec §3.8). */
export function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        skipHtml
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={{
          a: ({ children, href }) => (
            <span className="md-link" title={href ?? ""}>
              {children}
            </span>
          ),
          img: ({ alt }) => <span className="md-img">[image: {alt ?? ""}]</span>,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
```

- [ ] **Step 6: The pieces**

`desktop/src/chat/ChatList.tsx`:

```tsx
import type { Chat } from "../ipc/bindings/Chat";
import type { ModelInfo } from "../ipc/bindings/ModelInfo";

export function ChatList({ chats, models, activeId, onSelect, onNew }: {
  chats: Chat[];
  models: ModelInfo[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
}) {
  return (
    <nav className="col left">
      <div className="head">
        <span className="title">Chats</span>
        <button className="link" style={{ marginLeft: "auto" }} onClick={onNew}>
          + NEW
        </button>
      </div>
      <div className="scroll">
        {chats.length === 0 && <div className="dim mono" style={{ padding: 20, fontSize: 11 }}>No chats yet.</div>}
        {chats.map((c) => (
          <button
            key={c.id}
            className={`chat-item ${c.id === activeId ? "selected" : ""}`}
            aria-current={c.id === activeId}
            onClick={() => onSelect(c.id)}
          >
            <span className="ell">{c.title || "New chat"}</span>
            <span className="cap ell">{models.find((m) => m.id === c.model_id)?.name ?? "missing model"}</span>
          </button>
        ))}
      </div>
    </nav>
  );
}
```

`desktop/src/chat/MessageView.tsx`:

```tsx
import type { MessageView } from "../ipc/bindings/MessageView";
import { Markdown } from "./Markdown";

export function MessageRow({ mv, liveText, onVersion }: {
  mv: MessageView;
  /** The streaming text, while this is the reply being written. */
  liveText: string | null;
  onVersion: (messageId: string) => void;
}) {
  const m = mv.message;
  const text = liveText ?? m.content;
  return (
    <div className={`msg ${m.role}`} data-status={m.status}>
      <div className="msg-head cap">
        <span>{m.role === "user" ? "YOU" : "MODEL"}</span>
        {mv.versions > 1 && (
          <span className="versions">
            <button className="link" aria-label="previous version" disabled={mv.version <= 1}
                    onClick={() => onVersion(mv.sibling_ids[mv.version - 2])}>‹</button>
            <span>{mv.version}/{mv.versions}</span>
            <button className="link" aria-label="next version" disabled={mv.version >= mv.versions}
                    onClick={() => onVersion(mv.sibling_ids[mv.version])}>›</button>
          </span>
        )}
        {(m.status === "stopped" || m.status === "interrupted" || m.status === "error") && (
          <span className={`pill status-${m.status}`}>{m.status.toUpperCase()}</span>
        )}
      </div>
      {m.role === "user" ? <div className="msg-user">{text}</div> : <Markdown text={text} />}
      {m.status === "error" && m.stats.error && <div className="msg-error">{m.stats.error}</div>}
    </div>
  );
}
```

`desktop/src/chat/Composer.tsx`:

```tsx
import { useState } from "react";

export function Composer({ busy, disabled, canRegenerate, onSend, onStop, onRegenerate }: {
  busy: boolean;
  disabled: boolean;
  canRegenerate: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onRegenerate: () => void;
}) {
  const [text, setText] = useState("");
  const submit = () => {
    const t = text.trim();
    if (!t || busy || disabled) return;
    onSend(t);
    setText("");
  };
  return (
    <div className="composer">
      <textarea
        aria-label="message"
        placeholder="Message (Enter sends, Shift+Enter is a new line)"
        value={text}
        disabled={disabled}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
      />
      <div className="composer-actions">
        {canRegenerate && !busy && (
          <button className="btn" onClick={onRegenerate} disabled={disabled}>REGENERATE</button>
        )}
        {busy ? (
          <button className="btn accent" onClick={onStop}>STOP</button>
        ) : (
          <button className="btn accent" onClick={submit} disabled={disabled || !text.trim()}>SEND</button>
        )}
      </div>
    </div>
  );
}
```

`desktop/src/chat/ReplyPanel.tsx`:

```tsx
import { commas, secs } from "../format";
import type { Chat } from "../ipc/bindings/Chat";
import type { ModelInfo } from "../ipc/bindings/ModelInfo";
import type { Overview } from "../ipc/bindings/Overview";
import type { ReplyStats } from "../ipc/bindings/ReplyStats";
import type { Live } from "./stream";

export function ReplyPanel({ chat, models, overview, live, last, onModel }: {
  chat: Chat | null;
  models: ModelInfo[];
  overview: Overview | null;
  live: Live;
  last: ReplyStats | null;
  onModel: (modelId: string) => void;
}) {
  const current = models.find((m) => m.id === chat?.model_id);
  const e = overview?.engine;
  const engine = !chat ? "—" : e?.model_id === chat.model_id ? (e?.state ?? "—") : e?.state === "starting" ? "switching" : "loads on the next send";
  const stats = live.stats ?? last;
  const ctx = current?.ctx_engine ?? 4096;
  const used = stats?.prompt_tokens != null ? stats.prompt_tokens + (stats.completion_tokens ?? 0) : null;
  return (
    <aside className="col right">
      <div className="section">
        <span className="cap">MODEL</span>
        <select aria-label="model" value={chat?.model_id ?? ""} disabled={!chat || live.busy}
                onChange={(ev) => onModel(ev.target.value)}>
          {models.map((m) => (
            <option key={m.id} value={m.id} disabled={!m.runnable} title={m.reason ?? ""}>
              {m.name}{m.runnable ? "" : " (can't run)"}
            </option>
          ))}
        </select>
        <div className="fact"><span className="k">engine</span><span className="v">{engine}</span></div>
        {current?.system_source && (
          <div className="fact"><span className="k">system prompt</span><span className="v">from {current.system_source}</span></div>
        )}
      </div>
      <div className="section">
        <span className="cap">LAST REPLY</span>
        <div className="fact"><span className="k">first token</span><span className="v num">{secs(stats?.ttft_ms)}</span></div>
        <div className="fact"><span className="k">speed</span>
          <span className="v num">{stats?.tokens_per_s != null ? `${stats.tokens_per_s.toFixed(1)} tok/s` : "—"}</span></div>
        <div className="fact"><span className="k">tokens in · out</span>
          <span className="v num">{stats?.prompt_tokens ?? "—"} · {stats?.completion_tokens ?? "—"}</span></div>
        <div className="fact"><span className="k">total</span><span className="v num">{stats ? secs(stats.total_ms) : "—"}</span></div>
      </div>
      <div className="section">
        <span className="cap">CONTEXT</span>
        <div className="bar" aria-label="context used">
          <i style={{ width: `${Math.min(100, ((used ?? 0) / ctx) * 100)}%` }} />
        </div>
        <div className="fact"><span className="k">last request</span><span className="v num">{used ?? "—"} / {commas(ctx)}</span></div>
        <span className="mono dim" style={{ fontSize: 10.5, lineHeight: 1.5 }}>
          Each turn sends the last ~1,500 tokens of the chat. The engine re-reads them on every turn until it keeps
          a cache between turns, so more history means a longer wait for the first word.
        </span>
      </div>
    </aside>
  );
}
```

`desktop/src/chat/chat.css`:

```css
.chat-item { display: flex; flex-direction: column; gap: 4px; width: 100%; text-align: left; background: none; border: 0;
             border-bottom: 1px solid var(--l20); padding: 12px 20px; color: var(--d80); cursor: pointer; font-size: 12.5px; }
.chat-item.selected { background: var(--sel); color: var(--text); box-shadow: inset 2px 0 0 var(--accent); }
.messages { flex: 1; padding: 18px 28px; display: flex; flex-direction: column; gap: 18px; }
.msg { display: flex; flex-direction: column; gap: 6px; max-width: 860px; }
.msg-head { display: flex; gap: 12px; align-items: center; }
.msg-user { white-space: pre-wrap; background: var(--field); border: 1px solid var(--l24); padding: 10px 14px; }
.msg-error { color: var(--flame); font-family: var(--mono); font-size: 11px; }
.versions { display: inline-flex; gap: 6px; align-items: center; }
.status-stopped, .status-interrupted, .status-streaming { color: var(--shared); }
.status-error { color: var(--flame); }
.status-ok { color: var(--d72); }
.history-cut { border-top: 1px dashed var(--l30); padding-top: 6px; }
.waiting { color: var(--d60); font-family: var(--mono); font-size: 11px; }
.notice { color: var(--flame); font-family: var(--mono); font-size: 11px; padding: 0 20px 8px; }
.composer { border-top: 1px solid var(--l27); padding: 12px 20px; display: flex; gap: 12px; align-items: flex-end; }
.composer textarea { flex: 1; min-height: 44px; max-height: 200px; resize: vertical; background: var(--field); color: var(--text);
                     border: 1px solid var(--l30); padding: 10px 12px; font-size: 13px; }
.composer-actions { display: flex; gap: 8px; }
.md { line-height: 1.55; overflow-wrap: anywhere; }
.md pre { background: var(--code); border: 1px solid var(--l24); padding: 10px 12px; overflow-x: auto; }
.md code { font-family: var(--mono); font-size: 12px; }
.md table { border-collapse: collapse; }
.md th, .md td { border: 1px solid var(--l27); padding: 4px 8px; }
.md-link { color: var(--shared); text-decoration: underline dotted; }
.right select { background: var(--field); color: var(--text); border: 1px solid var(--l30); padding: 7px 8px;
                font-family: var(--mono); font-size: 11px; }
```

- [ ] **Step 7: The tab**

`desktop/src/chat/ChatTab.tsx`:

```tsx
import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { api } from "../ipc/api";
import type { Chat } from "../ipc/bindings/Chat";
import type { ChatView } from "../ipc/bindings/ChatView";
import type { ModelInfo } from "../ipc/bindings/ModelInfo";
import type { Overview } from "../ipc/bindings/Overview";
import type { StreamEvent } from "../ipc/bindings/StreamEvent";
import "./chat.css";
import { ChatList } from "./ChatList";
import { Composer } from "./Composer";
import { MessageRow } from "./MessageView";
import { ReplyPanel } from "./ReplyPanel";
import { applyEvent, begin, idle, statsOf, type Live } from "./stream";

/** At most ~30 redraws a second while tokens stream, so the UI does not compete with the engine (spec §3.8). */
const FLUSH_MS = 33;

export function ChatTab({ overview }: { overview: Overview | null }) {
  const [chats, setChats] = useState<Chat[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [view, setView] = useState<ChatView | null>(null);
  const [live, setLive] = useState<Live>(idle);
  const [notice, setNotice] = useState<string | null>(null);
  const active = useRef<string | null>(null);
  const pending = useRef("");
  const timer = useRef<number | null>(null);

  const refreshChats = useCallback(async () => {
    const cs = await api.chats();
    setChats(cs);
    return cs;
  }, []);

  const reload = useCallback(async (chatId: string) => {
    const v = await api.chat(chatId);
    if (active.current === chatId) setView(v);
  }, []);

  useEffect(() => {
    api.models().then(setModels).catch(() => {});
    refreshChats()
      .then((cs) => {
        if (cs.length > 0) setActiveId((id) => id ?? cs[0].id);
      })
      .catch((e) => setNotice(String(e)));
  }, [refreshChats]);

  useEffect(() => {
    active.current = activeId;
    setLive(idle);
    setNotice(null);
    if (activeId) reload(activeId).catch((e) => setNotice(String(e)));
    else setView(null);
  }, [activeId, reload]);

  const flush = () => {
    timer.current = null;
    const text = pending.current;
    pending.current = "";
    if (text) setLive((s) => applyEvent(s, { kind: "token", text }));
  };

  const onEvent = (chatId: string) => (e: StreamEvent) => {
    if (active.current !== chatId) return; // the reply still lands in the store; reopening the chat shows it
    if (e.kind === "token") {
      pending.current += e.text;
      if (timer.current == null) timer.current = window.setTimeout(flush, FLUSH_MS);
      return;
    }
    if (timer.current != null) window.clearTimeout(timer.current);
    flush();
    setLive((s) => applyEvent(s, e));
    reload(chatId).catch(() => {});
    if (e.kind !== "started") refreshChats().catch(() => {});
  };

  const run = async (chatId: string, call: (cb: (e: StreamEvent) => void) => Promise<void>) => {
    setNotice(null);
    setLive(begin());
    try {
      await call(onEvent(chatId));
      // The host always ends a reply with an event; never leave the composer stuck on STOP if one is lost.
      if (active.current === chatId) setLive((s) => (s.busy ? { ...s, busy: false } : s));
    } catch (e) {
      if (active.current === chatId) {
        setLive(idle);
        setNotice(String(e));
      }
    }
  };

  const send = (text: string) => {
    const id = activeId;
    if (id) void run(id, (cb) => api.send(id, text, cb));
  };
  const regenerate = () => {
    const id = activeId;
    if (id) void run(id, (cb) => api.regenerate(id, cb));
  };
  const stop = () => {
    if (activeId) api.stop(activeId).catch(() => {});
  };
  const newChat = async () => {
    try {
      const c = await api.newChat(null);
      await refreshChats();
      setActiveId(c.id);
    } catch (e) {
      setNotice(String(e));
    }
  };
  const setModel = async (modelId: string) => {
    const id = activeId;
    if (!id) return;
    try {
      await api.setChatModel(id, modelId);
      await reload(id);
      await refreshChats();
    } catch (e) {
      setNotice(String(e));
    }
  };
  const selectVersion = async (messageId: string) => {
    const id = activeId;
    if (!id) return;
    await api.selectVersion(id, messageId);
    await reload(id);
  };

  const messages = view?.messages ?? [];
  const chat = view?.chat ?? null;
  const lastAssistant = [...messages].reverse().find((m) => m.message.role === "assistant");
  const busy = live.busy || messages.some((m) => m.message.status === "streaming");
  const cut = live.dropped;

  return (
    <>
      <ChatList chats={chats} models={models} activeId={activeId} onSelect={setActiveId} onNew={newChat} />
      <section className="col center">
        <div className="head">
          <span className="title ell">{chat?.title || "Chat"}</span>
          <span className="cap">{models.find((m) => m.id === chat?.model_id)?.name ?? ""}</span>
        </div>
        <div className="scroll messages" aria-label="messages">
          {!chat && <div className="dim mono" style={{ fontSize: 11 }}>Start a chat with + NEW.</div>}
          {messages.map((mv, i) => (
            <Fragment key={mv.message.id}>
              <MessageRow
                mv={mv}
                liveText={mv.message.id === live.messageId && (live.busy || mv.message.status === "streaming") ? live.text : null}
                onVersion={selectVersion}
              />
              {cut > 0 && i === cut - 1 && (
                <div className="history-cut cap">
                  {cut} earlier turn{cut === 1 ? "" : "s"} not sent: outside the ~1,500-token history budget
                </div>
              )}
            </Fragment>
          ))}
          {live.busy && live.messageId == null && (
            <div className="waiting">waiting for the engine ({overview?.engine.state ?? "starting"})…</div>
          )}
        </div>
        {(notice ?? live.error) && <div className="notice" role="alert">{notice ?? live.error}</div>}
        <Composer busy={busy} disabled={!chat} canRegenerate={!!lastAssistant}
                  onSend={send} onStop={stop} onRegenerate={regenerate} />
      </section>
      <ReplyPanel chat={chat} models={models} overview={overview} live={live}
                  last={lastAssistant ? statsOf(lastAssistant.message) : null} onModel={setModel} />
    </>
  );
}
```

In `desktop/src/App.tsx`, import `ChatTab` and replace `renderTab`:

```tsx
import { ChatTab } from "./chat/ChatTab";

function renderTab(tab: Tab, overview: Overview | null): ReactNode {
  switch (tab) {
    case "Chat":
      return <ChatTab overview={overview} />;
    default:
      return <Placeholder tab={tab} />;
  }
}
```

- [ ] **Step 8: Run the tests and the type check**

```bash
npx vitest run
npx tsc --noEmit
```

Expected: every test passes (App 2, stream 4, Markdown 2, ChatTab 10), and no type errors. If `App.test.tsx`'s `findByText("serving")` now finds two elements, the Reply panel is printing the engine state for a chat that is not open; it must print "—" when no chat is open.

- [ ] **Step 9: Look at it**

Run the app against the fake engine as in Task 11 Step 7 (same environment variables), then:
- `+ NEW`, send "Long or wait?": the reply streams in, the Reply panel shows the first token and speed.
- REGENERATE: `‹ 2/2 ›` appears; `‹` shows the first version.
- Switch the model to `nexus`: the top bar shows the engine starting, then serving `nexus`.

Close the window.

- [ ] **Step 10: Commit**

```bash
git add desktop/package.json desktop/package-lock.json desktop/src
git commit -m "desktop: the Chat tab -- streaming Markdown replies, stop, regenerate, versions, model switch (task 13)"
```

---

### Task 14: UI — Models, Requests and Logs

**Files:**
- Create: `desktop/src/tabs/ModelsTab.tsx`, `desktop/src/tabs/RequestsTab.tsx`, `desktop/src/tabs/LogsTab.tsx`, `desktop/src/tabs/tabs.css`, `desktop/src/__tests__/tabs.test.tsx`
- Modify: `desktop/src/App.tsx` (render the three tabs)

**Interfaces:**
- Consumes: `api.models()`, `api.requests()`, `api.logs(after)` (Task 12); the bindings `ModelInfo` (Task 4), `RequestRecord`, `LogLine` (Task 6), `Overview` (Task 11); `format.ts` (`gb`, `secs`, `clock`, `commas`); the `testApi.ts` fixtures `models`, `overview` (Tasks 12–13). `Monitor::logs_after(seq)` returns the lines with a larger `seq`, and numbering starts at 1, so `logs(0)` returns everything (Task 6). `requests()` is newest first.
- Produces: `ModelsTab({ overview })`, `RequestsTab()`, `LogsTab()`; App renders them for "Models", "Requests" and "Logs" (Swarm and Settings stay placeholders until milestone 3).

- [ ] **Step 1: Write the failing tests**

`desktop/src/__tests__/tabs.test.tsx`:

```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { api } from "../ipc/api";
import type { LogLine } from "../ipc/bindings/LogLine";
import { LogsTab } from "../tabs/LogsTab";
import { ModelsTab } from "../tabs/ModelsTab";
import { RequestsTab } from "../tabs/RequestsTab";
import { models, overview } from "./testApi";

beforeEach(() => vi.clearAllMocks());

describe("ModelsTab", () => {
  it("lists every model, marks the default and the loaded one, and the one that can't run", async () => {
    vi.mocked(api.models).mockResolvedValue(models);
    render(<ModelsTab overview={overview} />);
    expect(await screen.findByText("qwen2.5vl:3b")).toBeTruthy();
    expect(screen.getAllByRole("row")).toHaveLength(3);
    expect(screen.getByText("DEFAULT")).toBeTruthy();
    expect(screen.getByText("LOADED")).toBeTruthy();
    expect(screen.getAllByText("CAN'T RUN")).toHaveLength(1);
    expect(screen.getByText("3 IN LIBRARY · 2 RUNNABLE")).toBeTruthy();
  });

  it("shows a model's Modelfile prompt, and why another cannot run", async () => {
    vi.mocked(api.models).mockResolvedValue(models);
    render(<ModelsTab overview={overview} />);
    fireEvent.click(await screen.findByText("nexus"));
    expect(screen.getByText(/You are Nexus AI/)).toBeTruthy();
    expect(screen.getByText(/FROM MODELFILE/)).toBeTruthy();
    fireEvent.click(screen.getByText("qwen2.5vl:3b"));
    expect(screen.getAllByText("CAN'T RUN")).toHaveLength(2);
    expect(screen.getByText(/architecture qwen2vl/)).toBeTruthy();
  });

  it("estimates the memory a model needs: the file plus its KV cache for 2 x 4,096 tokens", async () => {
    vi.mocked(api.models).mockResolvedValue(models);
    render(<ModelsTab overview={overview} />);
    await screen.findByText("qwen2.5vl:3b");
    // Qwen3-4B: 2,497,280,736 B + 78,336 B/token x 4,096 x 2 (641,728,512 B) = 3,139,009,248 B = 2.9 GB
    expect(screen.getByText("~2.9 GB")).toBeTruthy();
  });
});

describe("RequestsTab", () => {
  it("shows each request with its timings and outcome", async () => {
    vi.mocked(api.requests).mockResolvedValue([
      { id: "req_1", started_ms: Date.now(), source: "chat", endpoint: "/v1/chat/completions", model: "nexus",
        prompt_tokens: 41, completion_tokens: 5, ttft_ms: 480, total_ms: 730, tokens_per_s: 16.0, status: "ok", dropped_turns: 0 },
      { id: "req_2", started_ms: Date.now(), source: "chat", endpoint: "/v1/chat/completions", model: "nexus",
        prompt_tokens: null, completion_tokens: null, ttft_ms: null, total_ms: 90, tokens_per_s: null, status: "stopped", dropped_turns: 2 },
    ]);
    render(<RequestsTab />);
    expect(await screen.findByText("480 ms")).toBeTruthy();
    expect(screen.getByText("730 ms")).toBeTruthy();
    expect(screen.getByText("16.0")).toBeTruthy();
    expect(screen.getByText("OK")).toBeTruthy();
    expect(screen.getByText("STOPPED · 2 CUT")).toBeTruthy();
  });

  it("says when there is nothing yet", async () => {
    vi.mocked(api.requests).mockResolvedValue([]);
    render(<RequestsTab />);
    expect(await screen.findByText("No requests yet.")).toBeTruthy();
  });
});

describe("LogsTab", () => {
  const line = (seq: number, source: string, text: string, level = "info"): LogLine =>
    ({ seq, time_ms: Date.now(), source, level, text });

  it("asks only for lines after the last one it has, and filters by source", async () => {
    vi.mocked(api.logs)
      .mockResolvedValueOnce([line(1, "app", "Lorren 0.0.1 starting"), line(2, "engine", "stoke serve: listening on 127.0.0.1:50123")])
      .mockResolvedValue([]);
    render(<LogsTab />);
    expect(await screen.findByText(/listening on 127\.0\.0\.1:50123/)).toBeTruthy();
    expect(api.logs).toHaveBeenNthCalledWith(1, 0);
    await waitFor(() => expect(api.logs).toHaveBeenLastCalledWith(2), { timeout: 2500 });
    fireEvent.click(screen.getByText("ENGINE"));
    expect(screen.queryByText(/Lorren 0\.0\.1 starting/)).toBeNull();
    expect(screen.getByText(/listening on/)).toBeTruthy();
  });
});
```

Run: `npx vitest run src/__tests__/tabs.test.tsx` — Expected: FAIL, the `../tabs/*` modules do not exist.

- [ ] **Step 2: The Models tab**

`desktop/src/tabs/ModelsTab.tsx`:

```tsx
import { useEffect, useState } from "react";
import { commas, gb } from "../format";
import { api } from "../ipc/api";
import type { ModelInfo } from "../ipc/bindings/ModelInfo";
import type { Overview } from "../ipc/bindings/Overview";
import "./tabs.css";

/** The engine runs with --seqs 2 at the 4,096-token cap (Global Constraints). */
const SEQS = 2;
const COLS = "minmax(240px, 2fr) 80px 76px 80px 76px 84px 110px";

export function ModelsTab({ overview }: { overview: Overview | null }) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  useEffect(() => {
    api.models()
      .then((ms) => {
        setModels(ms);
        setSelected((s) => s ?? ms[0]?.id ?? null);
      })
      .catch(() => {});
  }, []);
  const m = models.find((x) => x.id === selected) ?? null;
  const loaded = overview?.engine.model_id;
  return (
    <>
      <section className="col center">
        <div className="head">
          <span className="title">Models</span>
          <span className="cap">{models.length} IN LIBRARY · {models.filter((x) => x.runnable).length} RUNNABLE</span>
        </div>
        <div className="table-head" style={{ gridTemplateColumns: COLS }}>
          <span>NAME</span><span>SOURCE</span><span>ARCH</span><span>QUANT</span><span>SIZE</span><span>TEMPLATE</span><span>STATUS</span>
        </div>
        <div className="scroll">
          {models.map((x) => (
            <div key={x.id} role="row" className={`table-row clickable ${x.id === selected ? "selected" : ""}`}
                 style={{ gridTemplateColumns: COLS, height: 44 }} onClick={() => setSelected(x.id)}>
              <span className="ell">
                <span>{x.name}</span>
                {x.is_default && <span className="pill tag accent-text">DEFAULT</span>}
                {x.id === loaded && <span className="pill tag shared-text">LOADED</span>}
              </span>
              <span className="mono dim">{x.source}</span>
              <span className="mono">{x.arch}</span>
              <span className="mono">{x.quant}</span>
              <span className="mono num">{gb(x.size_bytes)}</span>
              <span className="mono">{x.template}</span>
              <span className={`pill ${x.runnable ? "" : "flame-text"}`}>{x.runnable ? "RUNNABLE" : "CAN'T RUN"}</span>
            </div>
          ))}
        </div>
      </section>
      <aside className="col right">{m && <ModelDetails m={m} />}</aside>
    </>
  );
}

function ModelDetails({ m }: { m: ModelInfo }) {
  const kv = m.kv_bytes_per_token * m.ctx_engine * SEQS;
  return (
    <>
      <div className="section">
        <span className="cap">MODEL</span>
        <span className="title">{m.name}</span>
        <span className="mono dim ell" title={m.path} style={{ fontSize: 10.5 }}>{m.path}</span>
      </div>
      {!m.runnable && (
        <div className="section">
          <span className="cap flame-text">CAN'T RUN</span>
          <span>{m.reason}</span>
        </div>
      )}
      <div className="section">
        <div className="fact"><span className="k">size</span><span className="v">{m.size_label || "—"} · {m.quant}</span></div>
        <div className="fact"><span className="k">layers · heads · KV heads</span><span className="v num">{m.n_layer} · {m.n_head} · {m.n_head_kv}</span></div>
        <div className="fact"><span className="k">context (file / engine)</span><span className="v num">{commas(m.ctx_file)} / {commas(m.ctx_engine)}</span></div>
        <div className="fact"><span className="k">KV per token (q8)</span><span className="v num">{(m.kv_bytes_per_token / 1024).toFixed(1)} KiB</span></div>
        <div className="fact"><span className="k">memory to run</span><span className="v num">~{gb(m.size_bytes + kv)}</span></div>
      </div>
      <div className="section">
        <span className="cap">SYSTEM PROMPT{m.system_source ? ` · FROM ${m.system_source.toUpperCase()}` : ""}</span>
        <span className="mono" style={{ fontSize: 11.5, whiteSpace: "pre-wrap" }}>{m.system_prompt ?? "none"}</span>
      </div>
    </>
  );
}
```

- [ ] **Step 3: The Requests and Logs tabs**

`desktop/src/tabs/RequestsTab.tsx`:

```tsx
import { useEffect, useState } from "react";
import { clock, secs } from "../format";
import { api } from "../ipc/api";
import type { RequestRecord } from "../ipc/bindings/RequestRecord";
import "./tabs.css";

const COLS = "86px 64px minmax(200px, 2fr) 64px 64px 96px 86px 70px 150px";

export function RequestsTab() {
  const [rows, setRows] = useState<RequestRecord[]>([]);
  useEffect(() => {
    let alive = true;
    const tick = () => api.requests().then((r) => alive && setRows(r)).catch(() => {});
    tick();
    const t = setInterval(tick, 1000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);
  return (
    <section className="col center wide">
      <div className="head">
        <span className="title">Requests</span>
        <span className="cap">{rows.length} THIS SESSION · NEWEST FIRST</span>
      </div>
      <div className="table-head" style={{ gridTemplateColumns: COLS }}>
        <span>TIME</span><span>FROM</span><span>MODEL</span><span>IN</span><span>OUT</span><span>FIRST TOKEN</span><span>TOTAL</span><span>TOK/S</span><span>STATUS</span>
      </div>
      <div className="scroll">
        {rows.length === 0 && <div className="dim mono" style={{ padding: 22, fontSize: 11 }}>No requests yet.</div>}
        {rows.map((r) => (
          <div key={r.id} role="row" className="table-row" style={{ gridTemplateColumns: COLS, height: 36 }}>
            <span className="mono num">{clock(r.started_ms)}</span>
            <span className="mono dim">{r.source}</span>
            <span className="ell">{r.model}</span>
            <span className="mono num">{r.prompt_tokens ?? "—"}</span>
            <span className="mono num">{r.completion_tokens ?? "—"}</span>
            <span className="mono num">{secs(r.ttft_ms)}</span>
            <span className="mono num">{secs(r.total_ms)}</span>
            <span className="mono num">{r.tokens_per_s != null ? r.tokens_per_s.toFixed(1) : "—"}</span>
            <span className={`pill status-${r.status}`}>
              {r.status.toUpperCase()}{r.dropped_turns > 0 ? ` · ${r.dropped_turns} CUT` : ""}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
```

`desktop/src/tabs/LogsTab.tsx`:

```tsx
import { useEffect, useRef, useState } from "react";
import { clock } from "../format";
import { api } from "../ipc/api";
import type { LogLine } from "../ipc/bindings/LogLine";
import "./tabs.css";

const SOURCES = ["all", "app", "engine", "updater"] as const;
type Source = (typeof SOURCES)[number];
/** The host keeps 5,000 lines (Task 6); the tab keeps the same. */
const KEEP = 5000;

export function LogsTab() {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [filter, setFilter] = useState<Source>("all");
  const after = useRef(0);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let alive = true;
    const tick = () =>
      api.logs(after.current)
        .then((ls) => {
          if (!alive || ls.length === 0) return;
          after.current = ls[ls.length - 1].seq;
          setLines((prev) => [...prev, ...ls].slice(-KEEP));
        })
        .catch(() => {});
    tick();
    const t = setInterval(tick, 1000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  useEffect(() => {
    const b = box.current;
    if (b) b.scrollTop = b.scrollHeight;
  }, [lines, filter]);

  const shown = filter === "all" ? lines : lines.filter((l) => l.source === filter);
  return (
    <section className="col center wide">
      <div className="head">
        <span className="title">Logs</span>
        <span className="filters">
          {SOURCES.map((f) => (
            <button key={f} className={`link ${f === filter ? "on" : ""}`} aria-pressed={f === filter} onClick={() => setFilter(f)}>
              {f.toUpperCase()}
            </button>
          ))}
        </span>
      </div>
      <div className="scroll mono logbox" ref={box}>
        {shown.map((l) => (
          <div key={l.seq} className={`log-${l.level}`}>
            <span className="dim">{clock(l.time_ms)}</span> <span className="dim">{l.source.padEnd(8, " ")}</span>
            <span>{l.text}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
```

`desktop/src/tabs/tabs.css`:

```css
.wide { border-right: 0; }
.table-row.clickable { cursor: pointer; }
.pill.tag { margin-left: 8px; }
.accent-text { color: var(--accent); }
.shared-text { color: var(--shared); }
.flame-text { color: var(--flame); }
.filters { margin-left: auto; display: flex; gap: 14px; }
.link.on { color: var(--text); }
.logbox { flex: 1; padding: 10px 22px; font-size: 11px; line-height: 1.7; white-space: pre-wrap; }
.log-warn { color: var(--accent); }
.log-error { color: var(--flame); }
```

- [ ] **Step 4: Show them**

In `desktop/src/App.tsx`, import the three tabs and extend `renderTab`:

```tsx
import { LogsTab } from "./tabs/LogsTab";
import { ModelsTab } from "./tabs/ModelsTab";
import { RequestsTab } from "./tabs/RequestsTab";

function renderTab(tab: Tab, overview: Overview | null): ReactNode {
  switch (tab) {
    case "Chat":
      return <ChatTab overview={overview} />;
    case "Models":
      return <ModelsTab overview={overview} />;
    case "Requests":
      return <RequestsTab />;
    case "Logs":
      return <LogsTab />;
    default:
      return <Placeholder tab={tab} />;
  }
}
```

- [ ] **Step 5: Run the tests and the type check**

```bash
npx vitest run
npx tsc --noEmit
```

Expected: every test passes (this task adds 6), no type errors.

- [ ] **Step 6: Look at it against the fake engine**

Run the app as in Task 11 Step 7. Models: four rows, `qwen2.5vl-3b` CAN'T RUN, Qwen3 marked DEFAULT and LOADED, `nexus` shows the trader prompt FROM MODELFILE. Send one chat message, then Requests shows it with a first-token time, and Logs shows `engine serving …`. Close the window.

- [ ] **Step 7: Commit**

```bash
git add desktop/src
git commit -m "desktop: Models, Requests and Logs tabs (task 14)"
```

---

### Task 15: Updater and the test feed

**Files:**
- Create: `desktop/src-tauri/src/updater.rs`, `desktop/src-tauri/tauri.updater-test.conf.json`, `desktop/scripts/update-feed.py`, `desktop/src/components/UpdateBadge.tsx`, `desktop/src/__tests__/UpdateBadge.test.tsx`
- Modify: `desktop/src-tauri/Cargo.toml` (plugin; `minisign-verify` for tests), `desktop/src-tauri/tauri.conf.json` (`createUpdaterArtifacts`, `plugins.updater`), `desktop/src-tauri/src/lib.rs` (plugin, service, startup check, commands, version from the config), `desktop/src-tauri/src/commands.rs` (`AppState.updates`, three commands, `overview` takes the version from the config), `desktop/src/ipc/api.ts`, `desktop/src/__tests__/testApi.ts`, `desktop/src/App.tsx` (the badge in the status bar)

**Interfaces:**
- Consumes: `Monitor` (Task 6); `AppState`, `overview` (Task 11); `StatusBar({overview, children})` (Task 12).
- Produces:
  - `UpdateState` (serde `tag = "state"`, snake_case; ts-rs): `NotConfigured`, `Idle`, `Checking`, `UpToDate { version }`, `Downloading { version, done, total }`, `Ready { version }`, `Refused { version, reason }`, `Failed { message }`
  - `Verdict::{NotConfigured, Refused(String), Failed(String)}`, `pub fn classify(&tauri_plugin_updater::Error) -> Verdict`
  - `UpdateService::new(monitor) -> Arc<UpdateService>`; `state()`; `check(self: &Arc<Self>, &AppHandle)` (check, then download and verify, never install); `install()` (on Windows the plugin starts the NSIS installer and exits the app)
  - Commands `update_state()`, `check_update()`, `install_update()`; event `update-state`
  - `api.updateState()`, `api.checkUpdate()`, `api.installUpdate()`, `api.onUpdateState(cb)`; `UpdateBadge()`
  - `scripts/update-feed.py keys | build <version> | feeds | serve <feed>`; feeds `good`, `tampered`, `wrongkey`, `nosig` under `e2e/.work/feed/`, served on `http://127.0.0.1:8765/`

**Facts this task relies on** (read from `tauri-plugin-updater` v2's source on 2026-10-12; check them against the version `cargo add` installs, in `~/.cargo/registry/src/*/tauri-plugin-updater-2.*/src/`):
- `Update::download(on_chunk, on_finish)` verifies the minisign signature itself (`self.verify(&buffer)?`) before it returns the bytes. `Update::install(bytes)` on Windows starts the NSIS installer with `/UPDATE` and calls `std::process::exit(0)`.
- `Error` has `EmptyEndpoints`, `Minisign(minisign_verify::Error)`, `Base64(..)`, `SignatureUtf8(String)` and `Network(String)`, among others.
- `pubkey` is required in `plugins.updater`. A plain `http://` endpoint is refused unless `plugins.updater.dangerousInsecureTransportProtocol` is `true`, and there is no builder method for it. **So the `http://127.0.0.1` test feed works only through `tauri.updater-test.conf.json`, which no release build may use** (milestone 4's release script must refuse it).
- `tauri build --config` may be given more than once (later files win); `tauri signer generate -w <file> -p <password> --ci`; `tauri signer sign -f <key file> -p <password> <file>`; signing during `tauri build` reads `TAURI_SIGNING_PRIVATE_KEY_PATH` and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`.

- [ ] **Step 1: The test keys**

`desktop/scripts/update-feed.py`:

```python
"""The updater's test kit (milestone 1 only; the real keys and R2 arrive in milestone 4).

  python scripts/update-feed.py keys            two TEST key pairs in .keys/ (git-ignored); the first
                                                one's public key goes into tauri.conf.json
  python scripts/update-feed.py build <version> a signed NSIS installer of that version, built with
                                                the test overlay (http feed on 127.0.0.1)
  python scripts/update-feed.py feeds           e2e/.work/feed/{good,tampered,wrongkey,nosig} from 0.0.2
  python scripts/update-feed.py serve <feed>    serves one feed on http://127.0.0.1:8765/ until Ctrl+C

Run from desktop/. Uses the project's own Tauri CLI through node, never a global one.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path.cwd()
KEYS = HERE / ".keys"
WORK = HERE / "e2e" / ".work"
CONF = HERE / "src-tauri" / "tauri.conf.json"
OVERLAY = HERE / "src-tauri" / "tauri.updater-test.conf.json"
PASSWORD = "lorren-test"
PORT = 8765
TAURI = [shutil.which("node") or "node", str(HERE / "node_modules" / "@tauri-apps" / "cli" / "tauri.js")]


def run(args, env=None):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run(args, check=True, env=env)


def keys():
    KEYS.mkdir(exist_ok=True)
    for name in ("test", "wrong"):
        if not (KEYS / f"{name}.key").exists():
            run(TAURI + ["signer", "generate", "--ci", "-w", str(KEYS / f"{name}.key"), "-p", PASSWORD])
    pub = (KEYS / "test.key.pub").read_text().strip()
    conf = json.loads(CONF.read_text(encoding="utf-8"))
    conf.setdefault("plugins", {}).setdefault("updater", {})["pubkey"] = pub
    CONF.write_text(json.dumps(conf, indent=2) + "\n", encoding="utf-8")
    print("test public key written to", CONF)


def build(version):
    env = dict(os.environ)
    env["TAURI_SIGNING_PRIVATE_KEY_PATH"] = str(KEYS / "test.key")
    env["TAURI_SIGNING_PRIVATE_KEY_PASSWORD"] = PASSWORD
    run(TAURI + ["build", "--config", str(OVERLAY), "--config", json.dumps({"version": version})], env=env)
    nsis = HERE / "src-tauri" / "target" / "release" / "bundle" / "nsis"
    exe = nsis / f"Lorren_{version}_x64-setup.exe"
    sig = pathlib.Path(str(exe) + ".sig")
    assert exe.exists() and sig.exists(), f"missing {exe} or its .sig"
    out = WORK / "builds" / version
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exe, out / exe.name)
    shutil.copy2(sig, out / sig.name)
    print("built", out / exe.name)


def write_feed(name, installer, signature, note):
    d = WORK / "feed" / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "Lorren_0.0.2_x64-setup.exe").write_bytes(installer)
    latest = {
        "version": "0.0.2",
        "notes": f"Lorren 0.0.2 (test feed: {note})",
        "pub_date": "2026-10-12T00:00:00Z",
        "platforms": {
            "windows-x86_64": {
                "signature": signature,
                "url": f"http://127.0.0.1:{PORT}/Lorren_0.0.2_x64-setup.exe",
            }
        },
    }
    (d / "latest.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")
    print("feed", name, "->", d)


def feeds():
    src = WORK / "builds" / "0.0.2" / "Lorren_0.0.2_x64-setup.exe"
    installer = src.read_bytes()
    good_sig = pathlib.Path(str(src) + ".sig").read_text().strip()
    write_feed("good", installer, good_sig, "good")

    flipped = bytearray(installer)
    flipped[len(flipped) // 2] ^= 0x01
    assert bytes(flipped) != installer
    write_feed("tampered", bytes(flipped), good_sig, "one byte flipped, original signature")

    tmp = WORK / "wrongkey-sign" / src.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    run(TAURI + ["signer", "sign", "-f", str(KEYS / "wrong.key"), "-p", PASSWORD, str(tmp)])
    wrong_sig = pathlib.Path(str(tmp) + ".sig").read_text().strip()
    assert wrong_sig != good_sig
    write_feed("wrongkey", installer, wrong_sig, "signed with a key the app does not trust")

    write_feed("nosig", installer, "", "no signature")


def serve(name):
    d = WORK / "feed" / name
    assert (d / "latest.json").exists(), f"no feed {name}; run `feeds` first"
    print(f"serving {d} on http://127.0.0.1:{PORT}/ (Ctrl+C stops)", flush=True)
    run([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1", "--directory", str(d)])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "keys":
        keys()
    elif cmd == "build" and len(sys.argv) == 3:
        build(sys.argv[2])
    elif cmd == "feeds":
        feeds()
    elif cmd == "serve" and len(sys.argv) == 3:
        serve(sys.argv[2])
    else:
        sys.exit(__doc__)
```

```bash
cd "/m/New Drive/Desktop/Programming/stoke-lorren/desktop"
python scripts/update-feed.py keys
ls .keys
git check-ignore -q .keys/test.key && echo "the private keys are git-ignored"
grep -c '"pubkey"' src-tauri/tauri.conf.json
```

Expected: `test.key`, `test.key.pub`, `wrong.key`, `wrong.key.pub`; `the private keys are git-ignored`; `1`.

- [ ] **Step 2: Configure the plugin**

```bash
cd src-tauri
cargo add tauri-plugin-updater
cargo tree -i minisign-verify --depth 0
```

Note the version that `cargo tree` prints, then add exactly that version for the tests (a different version would be a different `Error` type):

```bash
cargo add --dev minisign-verify@=<the version cargo tree printed>
```

In `desktop/src-tauri/tauri.conf.json`, add `"createUpdaterArtifacts": true` inside `"bundle"`, and complete `"plugins"` (the `pubkey` is already there from Step 1):

```json
  "plugins": {
    "updater": {
      "pubkey": "<written by update-feed.py keys>",
      "endpoints": [],
      "windows": { "installMode": "passive" }
    }
  }
```

No endpoint is the honest milestone-1 state: a normal run reports "updates not configured". Milestone 4 sets the real endpoint and the production key.

`desktop/src-tauri/tauri.updater-test.conf.json`:

```json
{
  "plugins": {
    "updater": {
      "endpoints": ["http://127.0.0.1:8765/latest.json"],
      "dangerousInsecureTransportProtocol": true
    }
  }
}
```

- [ ] **Step 3: Write the failing tests**

`desktop/src-tauri/src/updater.rs`, tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tauri_plugin_updater::Error as E;

    #[test]
    fn a_signature_that_does_not_verify_is_a_refusal() {
        assert!(matches!(classify(&E::Minisign(minisign_verify::Error::InvalidSignature)), Verdict::Refused(_)));
        assert!(matches!(classify(&E::SignatureUtf8("not base64".into())), Verdict::Refused(_)));
    }

    #[test]
    fn a_network_failure_is_not_a_refusal_and_no_endpoint_is_not_configured() {
        assert!(matches!(classify(&E::Network("connection refused".into())), Verdict::Failed(_)));
        assert_eq!(classify(&E::EmptyEndpoints), Verdict::NotConfigured);
    }

    #[test]
    fn the_ui_reads_the_state_by_its_tag() {
        let j = serde_json::to_value(UpdateState::Ready { version: "0.0.2".into() }).unwrap();
        assert_eq!(j, serde_json::json!({"state": "ready", "version": "0.0.2"}));
        assert_eq!(serde_json::to_value(UpdateState::NotConfigured).unwrap(), serde_json::json!({"state": "not_configured"}));
    }
}
```

Add `pub mod updater;` to `lib.rs`. Run: `cargo test updater` — Expected: compile errors (`classify`, `Verdict`, `UpdateState` missing).

`desktop/src/__tests__/UpdateBadge.test.tsx`:

```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { UpdateBadge } from "../components/UpdateBadge";
import { api } from "../ipc/api";

beforeEach(() => vi.clearAllMocks());

describe("UpdateBadge", () => {
  it("offers a verified update for install on restart", async () => {
    vi.mocked(api.updateState).mockResolvedValue({ state: "ready", version: "0.0.2" });
    render(<UpdateBadge />);
    fireEvent.click(await screen.findByText(/0\.0\.2 ready/));
    await waitFor(() => expect(api.installUpdate).toHaveBeenCalled());
  });

  it("says a refused update was refused, and can check again", async () => {
    vi.mocked(api.updateState).mockResolvedValue({ state: "refused", version: "0.0.2", reason: "the signature did not verify" });
    render(<UpdateBadge />);
    fireEvent.click(await screen.findByText(/0\.0\.2 refused/));
    await waitFor(() => expect(api.checkUpdate).toHaveBeenCalled());
  });

  it("shows nothing when updates are not configured", async () => {
    vi.mocked(api.updateState).mockResolvedValue({ state: "not_configured" });
    const { container } = render(<UpdateBadge />);
    await waitFor(() => expect(api.updateState).toHaveBeenCalled());
    expect(container.textContent).toBe("");
  });
});
```

Run: `npx vitest run src/__tests__/UpdateBadge.test.tsx` — Expected: FAIL (`../components/UpdateBadge` missing).

- [ ] **Step 4: The update service**

Above the tests in `updater.rs`:

```rust
//! App updates (spec §3.7) through tauri-plugin-updater: check, then download and verify in the
//! background; install only when the user restarts. The plugin checks the minisign signature
//! against the public key in tauri.conf.json; this module decides what a failure means and logs it.

use crate::monitor::Monitor;
use serde::Serialize;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Runtime};
use tauri_plugin_updater::{Update, UpdaterExt};
use ts_rs::TS;

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum UpdateState {
    /// No update endpoint is configured (every milestone-1 build except the test-feed ones).
    NotConfigured,
    Idle,
    Checking,
    UpToDate { version: String },
    Downloading {
        version: String,
        #[ts(type = "number")]
        done: u64,
        #[ts(type = "number | null")]
        total: Option<u64>,
    },
    /// Downloaded and its signature verified: installs when the user restarts.
    Ready { version: String },
    /// Downloaded, but its signature did not verify: thrown away, nothing installed.
    Refused { version: String, reason: String },
    Failed { message: String },
}

#[derive(Debug, PartialEq)]
pub enum Verdict {
    NotConfigured,
    Refused(String),
    Failed(String),
}

pub fn classify(e: &tauri_plugin_updater::Error) -> Verdict {
    use tauri_plugin_updater::Error as E;
    match e {
        E::EmptyEndpoints => Verdict::NotConfigured,
        E::Minisign(_) | E::Base64(_) | E::SignatureUtf8(_) => Verdict::Refused(format!("the signature did not verify ({e})")),
        _ => Verdict::Failed(e.to_string()),
    }
}

pub struct UpdateService {
    state: Mutex<UpdateState>,
    pending: Mutex<Option<(Update, Vec<u8>)>>,
    monitor: Arc<Monitor>,
}

impl UpdateService {
    pub fn new(monitor: Arc<Monitor>) -> Arc<UpdateService> {
        Arc::new(UpdateService { state: Mutex::new(UpdateState::Idle), pending: Mutex::new(None), monitor })
    }

    pub fn state(&self) -> UpdateState {
        self.state.lock().unwrap().clone()
    }

    fn set<R: Runtime>(&self, app: &AppHandle<R>, s: UpdateState) {
        *self.state.lock().unwrap() = s.clone();
        let _ = app.emit("update-state", s);
    }

    fn fail<R: Runtime>(&self, app: &AppHandle<R>, version: Option<String>, e: &tauri_plugin_updater::Error) {
        match classify(e) {
            Verdict::NotConfigured => self.set(app, UpdateState::NotConfigured),
            Verdict::Refused(reason) => {
                let v = version.unwrap_or_default();
                self.monitor.log("updater", "error", format!("update {v} REFUSED: {reason}; nothing was installed"));
                self.set(app, UpdateState::Refused { version: v, reason });
            }
            Verdict::Failed(message) => {
                self.monitor.log("updater", "warn", format!("update check failed: {message}"));
                self.set(app, UpdateState::Failed { message });
            }
        }
    }

    /// Checks the feed and, when it offers a newer version, downloads and verifies it. Never installs.
    pub async fn check<R: Runtime>(self: &Arc<Self>, app: &AppHandle<R>) {
        if matches!(self.state(), UpdateState::Checking | UpdateState::Downloading { .. } | UpdateState::Ready { .. }) {
            return;
        }
        self.set(app, UpdateState::Checking);
        let updater = match app.updater() {
            Ok(u) => u,
            Err(e) => return self.fail(app, None, &e),
        };
        let update = match updater.check().await {
            Ok(Some(u)) => u,
            Ok(None) => {
                let v = app.package_info().version.to_string();
                self.monitor.log("updater", "info", format!("up to date ({v})"));
                return self.set(app, UpdateState::UpToDate { version: v });
            }
            Err(e) => return self.fail(app, None, &e),
        };
        let version = update.version.clone();
        self.monitor.log("updater", "info", format!("update {version} offered at {}", update.download_url));
        self.set(app, UpdateState::Downloading { version: version.clone(), done: 0, total: None });
        let (me, progress_app, v) = (self.clone(), app.clone(), version.clone());
        let mut done = 0u64;
        let mut last = Instant::now();
        let result = update
            .download(
                move |chunk, total| {
                    done += chunk as u64;
                    if last.elapsed().as_millis() >= 250 {
                        last = Instant::now();
                        me.set(&progress_app, UpdateState::Downloading { version: v.clone(), done, total });
                    }
                },
                || {},
            )
            .await;
        match result {
            Ok(bytes) => {
                self.monitor.log(
                    "updater",
                    "info",
                    format!("update {version}: {} bytes downloaded and the signature verified; installs on restart", bytes.len()),
                );
                *self.pending.lock().unwrap() = Some((update, bytes));
                self.set(app, UpdateState::Ready { version });
            }
            Err(e) => self.fail(app, Some(version), &e),
        }
    }

    /// Installs the verified update. On Windows the plugin starts the NSIS installer and exits the
    /// app; the engine dies with it (job object, Task 8).
    pub fn install(&self) -> Result<(), String> {
        let (update, bytes) = self.pending.lock().unwrap().take().ok_or("no verified update is waiting")?;
        self.monitor.log("updater", "info", format!("installing {}", update.version));
        update.install(bytes).map_err(|e| e.to_string())
    }
}
```

Run: `cargo test updater` — Expected: 3 passed.

- [ ] **Step 5: Wire it in**

In `commands.rs`: add `pub updates: Arc<crate::updater::UpdateService>` to `AppState`. Make `overview` read the version from the config, which is what a 0.0.2 build changes (`CARGO_PKG_VERSION` stays 0.0.1):

```rust
#[tauri::command]
pub fn overview(app: tauri::AppHandle, state: State<'_, AppState>) -> Overview {
    let engine = state.supervisor.status();
    Overview {
        memory: memory_info(engine.pid),
        engine,
        app_version: app.package_info().version.to_string(),
        started_ms: state.started_ms,
        request_count: state.monitor.requests().len() as u32,
    }
}

#[tauri::command]
pub fn update_state(state: State<'_, AppState>) -> crate::updater::UpdateState {
    state.updates.state()
}

#[tauri::command]
pub async fn check_update(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let updates = state.updates.clone();
    updates.check(&app).await;
    Ok(())
}

#[tauri::command]
pub fn install_update(state: State<'_, AppState>) -> Result<(), String> {
    state.updates.install()
}
```

In `lib.rs`:
- add `pub mod updater;` and `.plugin(tauri_plugin_updater::Builder::new().build())` after the single-instance plugin;
- log the config's version: `monitor.log("app", "info", format!("Lorren {} starting", app.package_info().version));`
- create the service and put it in the state: `let updates = updater::UpdateService::new(monitor.clone());` and `updates: updates.clone()` in `AppState { .. }`;
- after `app.manage(..)`, check 5 s after start and every 6 h (spec §3.7):

```rust
            let (updates_bg, handle_bg) = (updates.clone(), app.handle().clone());
            tauri::async_runtime::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                loop {
                    updates_bg.check(&handle_bg).await;
                    if updates_bg.state() == updater::UpdateState::NotConfigured {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(6 * 3600)).await;
                }
            });
```

- add `commands::update_state, commands::check_update, commands::install_update` to `generate_handler!`.

In `desktop/src/ipc/api.ts`, import the generated `UpdateState` and add:

```ts
  updateState: () => invoke<UpdateState>("update_state"),
  checkUpdate: () => invoke<void>("check_update"),
  installUpdate: () => invoke<void>("install_update"),
  onUpdateState: (cb: (s: UpdateState) => void): Promise<UnlistenFn> =>
    listen<UpdateState>("update-state", (e) => cb(e.payload)),
```

In `mockApi()` (`testApi.ts`) add:

```ts
    updateState: vi.fn(async (): Promise<UpdateState> => ({ state: "not_configured" })),
    checkUpdate: vi.fn(async () => {}),
    installUpdate: vi.fn(async () => {}),
    onUpdateState: vi.fn(async () => () => {}),
```

(with `import type { UpdateState } from "../ipc/bindings/UpdateState";` at the top).

- [ ] **Step 6: The badge**

`desktop/src/components/UpdateBadge.tsx`:

```tsx
import { useEffect, useState } from "react";
import { api } from "../ipc/api";
import type { UpdateState } from "../ipc/bindings/UpdateState";

/** The update state, in the status bar. Nothing at all when updates are not configured. */
export function UpdateBadge() {
  const [s, setS] = useState<UpdateState | null>(null);
  useEffect(() => {
    api.updateState().then(setS).catch(() => {});
    const un = api.onUpdateState(setS);
    return () => {
      un.then((f) => f());
    };
  }, []);
  const check = () => api.checkUpdate().catch(() => {});
  if (!s || s.state === "not_configured") return null;
  switch (s.state) {
    case "idle":
      return <button className="link" onClick={check}>check for updates</button>;
    case "checking":
      return <span>checking for updates…</span>;
    case "up_to_date":
      return <button className="link" onClick={check}>up to date ({s.version})</button>;
    case "downloading":
      return <span>downloading {s.version}{s.total ? ` ${Math.round((s.done / s.total) * 100)}%` : ""}</span>;
    case "ready":
      return (
        <button className="link accent-text" onClick={() => api.installUpdate().catch(() => {})}>
          update {s.version} ready · RESTART TO INSTALL
        </button>
      );
    case "refused":
      return (
        <button className="link flame-text" title={s.reason} onClick={check}>
          update {s.version} refused: signature did not verify · check again
        </button>
      );
    case "failed":
      return <button className="link" title={s.message} onClick={check}>update check failed · retry</button>;
  }
}
```

In `App.tsx`, render it inside the status bar: `<StatusBar overview={overview}><UpdateBadge /></StatusBar>`. The `accent-text` and `flame-text` classes come from `tabs.css` (Task 14).

- [ ] **Step 7: Run every test**

```bash
cd src-tauri && cargo test --features fake-engine && cd ..
npx vitest run && npx tsc --noEmit
```

Expected: all green; `src/ipc/bindings/UpdateState.ts` exists.

- [ ] **Step 8: Build 0.0.2 and the four feeds**

This takes several minutes and a lot of RAM. **Do not run it while a soak run is in progress** (Global Constraints).

```bash
python scripts/update-feed.py build 0.0.2
python scripts/update-feed.py feeds
ls e2e/.work/feed
```

Expected: `built …/Lorren_0.0.2_x64-setup.exe`, then the four feeds `good`, `nosig`, `tampered`, `wrongkey`.

- [ ] **Step 9: The tamper checks, positive control last**

Each feed is the good feed with exactly one thing changed, so the good feed's success is what makes the three refusals mean something. Run the 0.0.1 app (dev build) with the test overlay against the fake engine (environment as in Task 11 Step 7):

```bash
npx tauri dev --config src-tauri/tauri.updater-test.conf.json
```

In a second terminal, serve each feed in turn (Ctrl+C before the next) and press the badge to check again:

| feed | the badge must read | the Logs tab must show (source `updater`) |
|---|---|---|
| `tampered` | `update 0.0.2 refused: signature did not verify` | `update 0.0.2 REFUSED: the signature did not verify (…)` |
| `wrongkey` | the same | the same |
| `nosig` | the same | the same |
| `good` | `update 0.0.2 ready · RESTART TO INSTALL` | `… bytes downloaded and the signature verified` |

If `good` does not reach "ready", the three refusals prove nothing: find out why (the URL, the JSON, the platform key `windows-x86_64`) before going on. If a refusal reads "update check failed" instead, the failure was not the signature, and the check has not passed. Do not press RESTART TO INSTALL in dev; installing is Task 17's, on the real installer. Close the app.

- [ ] **Step 10: Commit**

```bash
git add desktop/scripts/update-feed.py desktop/src-tauri desktop/src
git commit -m "desktop: signed updates through tauri-plugin-updater, the test feed kit and the badge (task 15)"
```

The commit carries the TEST public key in `tauri.conf.json`. The private keys stay in `desktop/.keys/` (git-ignored). Milestone 4 replaces the test public key with Kabir's offline production key before any build leaves this PC.

---

### Task 16: End-to-end run against the fake engine

**Files:**
- Create: `desktop/e2e/run-demo-e2e.mjs`
- Modify: `desktop/package.json` (`playwright-core`)

**Interfaces:**
- Consumes: the app as built by `tauri build --debug --no-bundle` with the test overlay (Task 15); the fake engine and `fake-engine write-fixtures` (Task 7), whose library is `nexus.gguf` + `Modelfile`, `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (the default), `Llama-3.2-3B-Instruct-Q4_K_M.gguf` and `qwen2.5vl-3b.gguf` (CAN'T RUN); the environment overrides `LORREN_ENGINE_EXE`, `LORREN_MODEL_DIRS`, `LORREN_OLLAMA_DIR`, `LORREN_DATA_DIR` (Tasks 4, 11); the fake engine's `FAKE_ENGINE_*` switches, which it reads from the environment the app passes on (Task 7); the feeds of `update-feed.py` (Task 15); the UI's labels: tab names, `+ NEW`, the `model` select, the `message` textarea, `SEND`, `STOP`, `REGENERATE`, `previous version`, the status pills, the update badge texts (Tasks 12–15).
- Produces: `npm run e2e`, which prints one `ok <step>` line per checklist item, then `e2e: <n>/<n> OK` or `e2e: FAILED at <step>: <reason>`, and writes `e2e/.work/measurements.json`.

- [ ] **Step 1: Add the driver**

```bash
cd "/m/New Drive/Desktop/Programming/stoke-lorren/desktop"
npm install -D playwright-core
```

WebView2 opens a DevTools port when the app's environment has `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9222`, and `playwright-core` drives that page through `chromium.connectOverCDP`. No browser download is needed.

- [ ] **Step 2: Write the run**

`desktop/e2e/run-demo-e2e.mjs`:

```js
// Milestone 1's checklist (spec §11.1), end to end against the FAKE engine: the real app window,
// driven through WebView2's DevTools port. Prints one line per step; the verdict is the last line.
//   npm run e2e            build, then run
//   npm run e2e -- --no-build
import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const DESKTOP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const TAURI_DIR = path.join(DESKTOP, "src-tauri");
const WORK = path.join(DESKTOP, "e2e", ".work");
const APP = path.join(TAURI_DIR, "target", "debug", "lorren.exe");
const FAKE = path.join(TAURI_DIR, "target", "debug", "fake-engine.exe");
const CDP = 9222;
const REPLY = "Verdict: HOLD. The trend is intact, so wait for a close above the range high before adding to the position today.";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
const measurements = {};
let app = null;
let feed = null;
let page = null;
let browser = null;

function sh(cmd, args, opts = {}) {
  console.log("+", cmd, args.join(" "));
  execFileSync(cmd, args, { stdio: "inherit", cwd: DESKTOP, ...opts });
}

function pids(image) {
  const out = execFileSync("tasklist", ["/FI", `IMAGENAME eq ${image}`, "/FO", "CSV", "/NH"], { encoding: "utf8" });
  return out.split(/\r?\n/).filter((l) => l.startsWith(`"${image}"`)).map((l) => Number(l.split('","')[1]));
}

function portOpen(port) {
  return new Promise((resolve) => {
    const s = net.connect(port, "127.0.0.1");
    s.once("connect", () => { s.destroy(); resolve(true); });
    s.once("error", () => resolve(false));
  });
}

async function until(what, fn, ms = 30000) {
  const end = Date.now() + ms;
  for (;;) {
    const v = await fn().catch(() => false);
    if (v) return v;
    if (Date.now() > end) throw new Error(`timed out after ${ms} ms waiting for ${what}`);
    await sleep(200);
  }
}

const topbar = () => page.locator(".topbar .engine").innerText();
const badge = () => page.locator(".statusbar").innerText();
const overview = () => page.evaluate(() => window.__TAURI_INTERNALS__.invoke("overview"));
const updaterLines = async () =>
  (await page.evaluate(() => window.__TAURI_INTERNALS__.invoke("logs", { after: 0 })))
    .filter((l) => l.source === "updater")
    .map((l) => l.text);

function env() {
  return {
    ...process.env,
    LORREN_ENGINE_EXE: FAKE,
    LORREN_MODEL_DIRS: path.join(WORK, "models"),
    LORREN_OLLAMA_DIR: "none",
    LORREN_DATA_DIR: path.join(WORK, "data"),
    WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${CDP}`,
    FAKE_ENGINE_TOKEN_MS: "60",
    FAKE_ENGINE_REPLY: REPLY,
    FAKE_ENGINE_EVENTS: path.join(WORK, "engine-events.log"),
  };
}

async function launch() {
  app = spawn(APP, [], { env: env(), stdio: "ignore" });
  await until("the DevTools port", () => portOpen(CDP), 30000);
  browser = await until("a CDP connection", () => chromium.connectOverCDP(`http://127.0.0.1:${CDP}`), 30000);
  page = await until("the app page", async () => browser.contexts()[0]?.pages()[0], 30000);
  await page.getByText("lorren", { exact: true }).waitFor({ timeout: 30000 });
}

async function close(graceful = true) {
  if (!app) return;
  const pid = app.pid;
  await browser?.close().catch(() => {});
  try { execFileSync("taskkill", graceful ? ["/PID", String(pid)] : ["/F", "/PID", String(pid)], { stdio: "ignore" }); } catch {}
  await until("the app to exit", async () => !pids("lorren.exe").includes(pid), 15000).catch(() => {
    execFileSync("taskkill", ["/F", "/PID", String(pid)], { stdio: "ignore" });
  });
  app = null;
}

async function serveFeed(name) {
  stopFeed();
  await until("port 8765 to be free", async () => !(await portOpen(8765)), 15000);
  feed = spawn(process.platform === "win32" ? "python" : "python3", ["scripts/update-feed.py", "serve", name], { cwd: DESKTOP, stdio: "ignore" });
  await until(`the ${name} feed on :8765`, () => portOpen(8765), 15000);
}

function stopFeed() {
  if (feed) { try { execFileSync("taskkill", ["/F", "/T", "/PID", String(feed.pid)], { stdio: "ignore" }); } catch {} feed = null; }
}

async function sendMessage(text) {
  await page.getByLabel("message").fill(text);
  await page.getByLabel("message").press("Enter");
  await page.getByRole("button", { name: "STOP" }).waitFor({ timeout: 15000 });
}

async function waitIdle(ms = 30000) {
  await page.getByRole("button", { name: "SEND" }).waitFor({ timeout: ms });
}

async function step(name, fn) {
  await fn();
  results.push(name);
  console.log(`ok   ${name}`);
}

const steps = [
  ["the window: wordmark, six tabs, the engine serving the default model", async () => {
    const tabs = await page.getByRole("tab").allInnerTexts();
    if (tabs.join(",") !== "Chat,Swarm,Models,Requests,Logs,Settings") throw new Error(`tabs: ${tabs}`);
    await until("serving Qwen3-4B", async () => { const t = await topbar(); return t.includes("serving") && t.includes("Qwen3-4B-Instruct-2507-Q4_K_M"); });
  }],
  ["Models: four models, qwen2.5vl CAN'T RUN, nexus has its Modelfile prompt", async () => {
    await page.getByRole("tab", { name: "Models" }).click();
    await until("4 rows", async () => (await page.getByRole("row").count()) === 4);
    const vl = page.getByRole("row").filter({ hasText: "qwen2.5vl" });
    if (!(await vl.innerText()).includes("CAN'T RUN")) throw new Error("qwen2.5vl is not marked CAN'T RUN");
    await page.getByRole("row").filter({ hasText: "nexus" }).click();
    await page.getByText(/FROM MODELFILE/).waitFor();
    await page.getByText(/You are Nexus AI/).waitFor();
  }],
  ["a new chat switched to Nexus restarts the engine on it", async () => {
    await page.getByRole("tab", { name: "Chat" }).click();
    await page.getByText("+ NEW").click();
    const t0 = Date.now();
    await page.locator("select[aria-label=model]").selectOption({ label: "nexus" });
    await until("serving nexus", async () => { const t = await topbar(); return t.includes("serving") && t.includes("nexus"); });
    measurements.switch_ms = Date.now() - t0;
  }],
  ["a reply streams, and its first-token time is shown", async () => {
    await sendMessage("Long or wait?");
    await page.locator(".messages").getByText("Verdict: HOLD.").first().waitFor({ timeout: 15000 });
    await waitIdle();
    const ttft = await page.locator(".right .fact").filter({ hasText: "first token" }).locator(".v").innerText();
    if (ttft === "—") throw new Error("no first-token time shown");
    measurements.ttft_shown = ttft;
  }],
  ["Stop ends a reply midway", async () => {
    await sendMessage("Explain.");
    await sleep(700);
    await page.getByRole("button", { name: "STOP" }).click();
    await waitIdle();
    await until("the STOPPED pill", async () => (await page.locator(".messages .pill").allInnerTexts()).includes("STOPPED"));
  }],
  ["Regenerate makes version 2/2, and the first version is still there", async () => {
    await page.getByRole("button", { name: "REGENERATE" }).click();
    await waitIdle();
    await page.getByText("2/2").waitFor();
    await page.getByLabel("previous version").last().click();
    await page.getByText("1/2").waitFor();
  }],
  ["killing the engine brings it back, and the chat carries on", async () => {
    const before = pids("fake-engine.exe");
    if (before.length !== 1) throw new Error(`expected one engine, found ${before.length}`);
    execFileSync("taskkill", ["/F", "/PID", String(before[0])], { stdio: "ignore" });
    await until("a new engine serving", async () => {
      const now = pids("fake-engine.exe");
      return now.length === 1 && now[0] !== before[0] && (await topbar()).includes("serving");
    });
    await until("the restart counted", async () => (await page.locator(".statusbar").innerText()).includes("1 engine restarts"));
    await sendMessage("Still there?");
    await waitIdle();
  }],
  ["Requests and Logs show the traffic", async () => {
    await page.getByRole("tab", { name: "Requests" }).click();
    await until("4 requests", async () => (await page.getByRole("row").count()) >= 4);
    const statuses = await page.locator(".pill").allInnerTexts();
    if (!statuses.some((s) => s.startsWith("OK")) || !statuses.some((s) => s.startsWith("STOPPED"))) throw new Error(`statuses: ${statuses}`);
    await page.getByRole("tab", { name: "Logs" }).click();
    await page.getByText(/engine serving/).first().waitFor();
  }],
  ["a tampered, a wrongly signed and an unsigned update are refused; the signed one is ready", async () => {
    // Each refusal is read from its own REFUSED log line, never from the badge alone: the badge
    // still shows the previous feed's identical text until the new check lands, and a check that
    // failed for another reason (no server, bad JSON) logs a warning, not a refusal.
    let refused = 0;
    for (const name of ["tampered", "wrongkey", "nosig", "good"]) {
      if (!fs.existsSync(path.join(WORK, "feed", name, "latest.json"))) throw new Error("no feeds: run `python scripts/update-feed.py build 0.0.2` then `feeds` (Task 15)");
      await serveFeed(name);
      await page.locator(".statusbar button").last().click();
      if (name === "good") {
        await until("the good update verified", async () => (await updaterLines()).some((t) => t.includes("signature verified")), 60000);
        await until("the ready badge", async () => (await badge()).includes("0.0.2 ready"));
      } else {
        refused += 1;
        await until(`the ${name} update refused`, async () => (await updaterLines()).filter((t) => t.includes("REFUSED")).length === refused, 60000);
        if (!(await badge()).includes("0.0.2 refused: signature did not verify")) throw new Error(`badge after ${name}: ${await badge()}`);
      }
      stopFeed();
    }
    const v = (await overview()).app_version;
    if (v !== "0.0.1") throw new Error(`the app is ${v}; nothing may install in this run`);
  }],
  ["closing the app leaves no engine behind, and the chats come back", async () => {
    await close(true);
    await until("no orphan engine", async () => pids("fake-engine.exe").length === 0, 15000);
    await launch();
    await page.getByRole("tab", { name: "Chat" }).click();
    await page.getByText("Long or wait?").first().click();
    await page.locator(".messages").getByText("Verdict: HOLD.").first().waitFor({ timeout: 15000 });
  }],
  ["memory is measured and shown", async () => {
    const o = await overview();
    Object.assign(measurements, { app_bytes: o.memory.app_bytes, engine_bytes: o.memory.engine_bytes, load_ms: o.engine.load_ms });
    if (!(o.memory.app_bytes > 0)) throw new Error("app memory not measured");
    if (!(await topbar()).includes("app ")) throw new Error("memory not shown in the top bar");
  }],
];

async function main() {
  if (!process.argv.includes("--no-build")) {
    sh(process.execPath, ["node_modules/@tauri-apps/cli/tauri.js", "build", "--debug", "--no-bundle", "--config", "src-tauri/tauri.updater-test.conf.json"]);
    sh("cargo", ["build", "--features", "fake-engine", "--bin", "fake-engine"], { cwd: TAURI_DIR });
  }
  if (pids("lorren.exe").length || pids("fake-engine.exe").length) throw new Error("close every running Lorren and fake engine first");
  fs.rmSync(WORK + "/data", { recursive: true, force: true });
  fs.rmSync(WORK + "/models", { recursive: true, force: true });
  fs.rmSync(WORK + "/engine-events.log", { force: true });
  sh(FAKE, ["write-fixtures", path.join(WORK, "models")]);
  await launch();
  for (const [name, fn] of steps) {
    try {
      await step(name, fn);
    } catch (e) {
      console.log(`e2e: FAILED at ${name}: ${e.message}`);
      return 1;
    }
  }
  fs.writeFileSync(path.join(WORK, "measurements.json"), JSON.stringify(measurements, null, 2));
  console.log("measurements:", JSON.stringify(measurements));
  console.log(`e2e: ${results.length}/${steps.length} OK`);
  return 0;
}

let code = 1;
try {
  code = await main();
} catch (e) {
  console.log(`e2e: FAILED before the steps: ${e.message}`);
} finally {
  stopFeed();
  await close(false);
}
process.exit(code);
```

- [ ] **Step 3: Run it red first**

Point the run at a state it must reject before trusting its green: start it with the feeds missing (rename `e2e/.work/feed` to `e2e/.work/feed.off`):

```bash
npm run e2e
```

Expected: every step before the update step passes, then `e2e: FAILED at a tampered, a wrongly signed and an unsigned update are refused; the signed one is ready: no feeds: …`. That proves the run can fail, and fails where it should. Rename the folder back.

- [ ] **Step 4: Run it green**

```bash
npm run e2e -- --no-build
```

Expected: eleven `ok` lines, then `e2e: 11/11 OK`, and `e2e/.work/measurements.json` with `switch_ms`, `ttft_shown`, `app_bytes`, `engine_bytes`, `load_ms`. Against the fake engine these numbers check the plumbing, not the speed; the real ones come from Task 17. If a step fails, the line names it: fix the app, never the expectation, unless the expectation contradicts the spec.

- [ ] **Step 5: Commit**

```bash
git add desktop/e2e/run-demo-e2e.mjs desktop/package.json desktop/package-lock.json
git commit -m "desktop: end-to-end run of the milestone-1 checklist against the fake engine (task 16)"
```

---

### Task 17: Installer and the real-model demo (with Kabir)

**Files:**
- Create: `docs/superpowers/gate-reports/<date>-lorren-m1-demo.md` (the record of the demo, on `lorren-desktop`)
- No planned code changes. A defect found here goes back to the task that owns it: a failing test there first, then the fix, then this task again from Step 3.

**Interfaces:**
- Consumes: everything above. The real library comes from Task 4's defaults (`M:\New Drive\ai_models`, `M:\New Drive\Desktop\Programming\stoke\models`, the downloads folder, and Ollama's manifests under `%USERPROFILE%\.ollama\models`). The installers and feeds come from `update-feed.py` (Task 15); the sidecar from `build-engine.sh` (Task 2).
- Produces: Lorren 0.0.2 installed on this PC for Kabir to use; the demo record with the measurements against spec §7; Kabir's verdict, which decides milestone 2.

**This task runs the real engine, a model server.** Kabir OK'd it for this milestone on 2026-09-25, but he names the time. **Never while a soak run is in progress.**

- [ ] **Step 1: Preconditions (ask, then check)**

Ask Kabir for the time. Then check:

```bash
MSYS_NO_PATHCONV=1 wsl.exe -- pgrep -fa run_m7_soak.sh || echo "no soak running"
powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"
```

Expected: `no soak running`, and at least 4,000 (MB free; the engine takes 2–2.5 GB). If a soak run is in progress, stop here: the demo waits for it to finish.

- [ ] **Step 2: Build the release installers**

Both versions are built from the same code now, so the update installs exactly what was tested:

```bash
cd "/m/New Drive/Desktop/Programming/stoke-lorren/desktop"
sh scripts/build-engine.sh
python scripts/update-feed.py build 0.0.1
python scripts/update-feed.py build 0.0.2
python scripts/update-feed.py feeds
ls -la e2e/.work/builds/*/
```

Expected: `build-engine: OK`, two `Lorren_<v>_x64-setup.exe` files each with a `.sig`, and the four feeds. These are TEST builds: they trust the test key and look for updates on `http://127.0.0.1:8765`.

- [ ] **Step 3: Install 0.0.1 and check the package**

Run `e2e/.work/builds/0.0.1/Lorren_0.0.1_x64-setup.exe` (a per-user install: no admin prompt). Then:

```bash
ls "$LOCALAPPDATA/Lorren"
"$LOCALAPPDATA/Lorren/lorren-engine.exe" info "M:/New Drive/Desktop/Programming/stoke/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" | head -2
```

Expected: `lorren.exe`, `lorren-engine.exe` and the uninstaller; the engine prints the model's metadata (a one-shot command, not a server). If the install folder differs, record the real one.

- [ ] **Step 4: The demo, with Kabir at the keyboard**

Start Lorren from the Start menu (no `LORREN_*` variables: the real library). Walk through spec §11.1 and note, for each item, what happened and the time it took:

1. Lorren opens: the Control Room plus Chat, dark. Time it from the click to the window (spec §7: under 1.5 s, not counting the model load).
2. Models lists Nexus with its trader prompt FROM MODELFILE, Qwen3-4B-Instruct-2507 Q4_K_M (DEFAULT), its Q8_0 and unsloth twins, Llama-3.2-3B-Instruct Q4_K_M and Q8_0, and marks `qwen2.5vl:3b` CAN'T RUN.
3. `+ NEW`, switch the chat to Nexus: the top bar shows the engine restart. Read the switch time from Logs (`engine serving … after N s`).
4. Ask Nexus something: the reply streams. Stop one reply midway. Regenerate one: `‹ 2/2 ›`.
5. Switch the chat to Qwen3: a visible restart; note the time again.
6. Task Manager → end `lorren-engine.exe`: the engine comes back by itself, the status bar counts one restart, and the next message works.
7. Close Lorren and open it again: the chats are there. Then check that no `lorren-engine.exe` is left in Task Manager after closing.
8. Requests and Logs show the traffic.
9. Updates, refusal first: `python scripts/update-feed.py serve tampered`, then press the badge to check again. It must read "update 0.0.2 refused", and Logs must show `REFUSED`. Then stop the server, run `serve good`, and check again: "update 0.0.2 ready · RESTART TO INSTALL". Press it: the installer runs and Lorren comes back (or start it again), and the top bar reads `0.0.2`. The chats are still there. Note whether Lorren restarted by itself.
10. Measurements for spec §7, with the numbers written down:
    - first token and speed of a short Nexus reply and a short Qwen3 reply (Reply panel);
    - the app and engine memory (top bar), idle, with one model loaded (budget: app plus WebView2 at most 200 MB);
    - the UI's CPU while a reply streams: in PowerShell, `Get-Process lorren,msedgewebview2 | Measure-Object CPU -Sum` before and after a 20 s stream; the difference ÷ 20 s (budget: under 5% of one core);
    - the model switch time (Logs).

- [ ] **Step 5: Uninstall, then reinstall for Kabir**

Run `"$LOCALAPPDATA/Lorren/uninstall.exe"`. Check that the install folder and the Start menu entry are gone and that no `lorren-engine.exe` remains, and note whether `%APPDATA%\Lorren` (the chats) was kept. Then install `e2e/.work/builds/0.0.2/Lorren_0.0.2_x64-setup.exe` again so Kabir keeps the app.

- [ ] **Step 6: Write the record**

`docs/superpowers/gate-reports/<date>-lorren-m1-demo.md`:

```markdown
# Lorren milestone 1 — the demo, <date>

Builds: 0.0.1 and 0.0.2 from `lorren-desktop` at <sha>, the engine sidecar from `src/` at `b1ee6b3`
(static). TEST builds: test key, test feed on 127.0.0.1.

| # | spec §11.1 item | result | notes |
|---|---|---|---|
| 1 | opens: Control Room + Chat, dark | | cold start <n> s |
| 2 | Models: Nexus (Modelfile), Qwen3 (default), Llama, qwen2.5vl CAN'T RUN | | |
| 3 | chat with Nexus streams, stops, regenerates | | |
| 4 | switch to Qwen3 visibly restarts the engine | | <n> s |
| 5 | killed engine comes back, the chat carries on | | |
| 6 | chats survive close and reopen; no orphan engine | | |
| 7 | Requests and Logs show the traffic | | |
| 8 | tampered 0.0.2 refused; signed 0.0.2 installs on restart | | restarted by itself: yes/no |
| 9 | install, upgrade, uninstall | | data kept: yes/no |

| spec §7 budget | measured | verdict |
|---|---|---|
| cold start < 1.5 s | | |
| idle app + WebView2 <= 200 MB | | |
| UI CPU while streaming < 5% of a core | | |
| model switch: seconds | Nexus <n> s, Qwen3 <n> s | |
| first token, short prompt | Nexus <n> s, Qwen3 <n> s | (09-25 by hand: 0.48–0.68 / 0.21–0.53 s) |

Known limits shown, not fixed here (spec §8): no KV reuse between turns, so a long chat waits longer for its first word;
the 4,096-token cap; one model at a time; the library's folders are this PC's until Settings (milestone 3).

Kabir's verdict: <his words>.
```

A missed budget is recorded as a miss, with the number. Do not round a miss into a pass.

- [ ] **Step 7: Commit**

```bash
cd "/m/New Drive/Desktop/Programming/stoke-lorren"
git add docs/superpowers/gate-reports/*-lorren-m1-demo.md
git commit -m "Lorren milestone 1: the demo with Kabir -- record, measurements against spec 7, his verdict"
```

Do not push. Kabir's verdict decides what milestone 2's plan contains.
