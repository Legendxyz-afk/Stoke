# Lorren distribution step 2: the catalog, Discover, downloads and the update setting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A stranger who installs Lorren can get a model inside the app: Discover lists a signed catalog of the two models milestone 1 runs, a download shows its size, licence and disk room before it starts, pauses and resumes, survives a dropped connection, is checked by sha256 as it arrives and then joins the library and the chat; and Settings has the update choice (auto, notify, off) the website's privacy chapter promises.

**Architecture:** A Node catalog script (beside the step-1 release script) builds `catalog.json` from a committed source list using Hugging Face's API for each file's size and sha256, signs it with the catalog key and writes it into the feed's master folder, which the release script now checks and compiles into the app (the catalog's address and public key ride in the build overlay as a `lorren-catalog` plugin section). In the app, `catalog.rs` fetches and verifies the catalog only when Discover is opened, `downloads.rs` runs the downloads (HTTP Range, backoff, sha256 as the bytes arrive, `.part` files, rows in the store) and adds the finished file to the library, and the update service obeys a mode kept in a new `settings` table. The UI (Discover in the Models tab, a Settings tab with one section, the empty-library chat) is built only after ghost has used and approved a demo of it (Task 1).

**Tech Stack:** Node 24 ESM with no dependencies (node:test), Rust (Tauri 2, reqwest 0.13 streaming, rusqlite, minisign-verify =0.3.0, sha2 0.10, tokio-util CancellationToken, tauri-plugin-opener), React 19 + TypeScript (vitest, jsdom), the M1 CDP e2e (`e2e/run-demo-e2e.mjs`, fake engine).

**Spec:** `docs/superpowers/specs/2026-11-04-lorren-distribution-first-design.md` (step 2 of its Order, the catalog script, the update setting) and `docs/superpowers/specs/2026-09-25-lorren-desktop-design.md` §3.5–§3.7, §4 (Models, Settings), §5, §6 — where they differ the distribution design is newer. Step 1's plan and ledger: `docs/superpowers/plans/2026-11-04-lorren-distribution-step1-release.md`, `.superpowers/sdd/2026-11-04-lorren-distribution-step1-release/progress.md`.

## Global Constraints

- Downloads (distribution design, Order 2): "no account, no Hugging Face page, size and licence and disk room shown before, progress with pause and resume, retry with backoff, sha256 as the bytes arrive".
- §3.7: "Downloads resume (HTTP Range) and are checked against their sha256 before they join the library. A newer version of an installed model shows "update available"."
- §5: "A download fails: it resumes. On a sha256 mismatch the file is deleted and an error is shown."
- §3.7: "A separate offline key signs it, so a leaked catalog key cannot ship code." The app verifies the catalog against a public key compiled into the build; the app's update key must not be able to sign a catalog.
- §3.7 entry fields: "id, name, version, description, architecture, quantization, size, sha256, a URL [on R2 → now a Hugging Face file pinned to a commit], an optional system prompt and parameters, and the minimum app version", plus the licence (distribution design: "each with its licence").
- §3.5: "Downloaded models go to `%USERPROFILE%\Lorren\models`."
- §6: "The UI: no remote content, a strict CSP, and IPC limited by capabilities … Links open in the default browser, and only when clicked." "The app writes only to its own data, log and download folders."
- The website's privacy page (`company/src/site/legal.js`): "Its only outside traffic is update checks (which you can turn off), update downloads, and the model downloads you start, from Hugging Face." ⇒ the app reads the catalog ONLY when the user opens Discover (never at startup, never on a timer), and never downloads a model the user did not start. (Naming the catalog read on that page is step 3's copy work, in `company`.)
- The update setting (distribution design): "a Settings section with only that choice, kept in the store, which the update service obeys (off: no check at all; notify: check, download nothing)"; default auto (§3.7).
- §4 Settings: "The same panel language: sections down the left, fields on the right." The design language of §4 (dark ground, Archivo + JetBrains Mono, square 1 px panels, caps mono labels tracked 0.14 em, the amber/blue/flame accents).
- §3.8: streaming redraws at most 30 per second ⇒ download progress events at most 4 per second (the updater's 250 ms).
- Private keys never enter the repository; in this step the catalog is signed with a TEST catalog key made by the M1 kit (`scripts/update-feed.py keys`); ghost's real catalog key is step 4. `--deploy` stays refused (step 4).
- Ghost's box: no compile, `cargo test` or e2e while he games; every heavy run behind the gate (no Wardogs/CoD process, ≥ 4 GB free); never start a model server; nothing is pushed.
- No Claude attribution trailers in commits.

## The catalog's two models (probed 2026-11-04)

The bytes the engine was validated on (`stoke/models/`, sha256 computed locally) are these Hugging Face files, so the catalog points at them:

| id | name | file (as saved) | repo @ commit | path in repo | size | sha256 | licence |
|---|---|---|---|---|---|---|---|
| `qwen3-4b-instruct-2507-q4_k_m` | Qwen3-4B-Instruct-2507 | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF` @ `ae44f08e1392f39c0e474af10c3ff8355c8b6688` | `Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2497280736 | `2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e` | Apache-2.0, `https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE` |
| `llama-3.2-3b-instruct-q4_k_m` | Llama-3.2-3B-Instruct | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | `bartowski/Llama-3.2-3B-Instruct-GGUF` @ `5ab33fa94d1d04e903623ae72c95d1696f09f9e8` | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | 2019377696 | `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff` | Llama 3.2 Community License, `https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt` |

Descriptions (copy for the demo; ghost may change them there): Qwen — "Qwen3 4B Instruct, July 2025: Lorren's default, quick and capable on the processor."; Llama — "Meta's Llama 3.2 3B Instruct: smaller and quicker, for everyday chats." Sizes as the app shows them (`format.ts` `gb`, binary): 2.3 GB and 1.9 GB.

Both repos are ungated; `resolve/<commit>/<path>` answers 302 to a signed CDN address with `Accept-Ranges: bytes`, `X-Linked-Size` and `X-Linked-ETag` (the sha256). The Qwen file is saved under the name WITHOUT bartowski's `Qwen_` prefix: the library's default rule is the file-name prefix `qwen3-4b-instruct-2507-q4_k_m` (`library.rs:70`), so the downloaded Qwen becomes the default.

## The catalog document (format 1)

```json
{ "format": 1, "generated": "<ISO time>", "models": [ {
  "id": "...", "name": "...", "version": 1, "description": "...", "architecture": "qwen3" | "llama", "quantization": "Q4_K_M",
  "file": "<name as saved>.gguf", "size": 2497280736, "sha256": "<64 lowercase hex>",
  "url": "https://huggingface.co/<repo>/resolve/<commit>/<path>",
  "licence": { "name": "...", "url": "https://..." }, "source": { "repo": "...", "commit": "...", "path": "..." },
  "system_prompt": null, "min_app_version": "0.1.0" } ] }
```

Entry rules — the SAME list in `scripts/release/catalog.mjs` `validateEntry` and `src-tauri/src/catalog.rs` `validate_entry` (one test table each, the same cases):
id `^[a-z0-9][a-z0-9._-]{0,63}$` and unique; name 1–80 chars; description ≤ 400; version a positive safe integer; architecture `qwen3` or `llama`; quantization 1–16 chars; file `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.gguf$` (case-insensitive extension), no `..`, unique; size an integer 1 ≤ size ≤ 68719476736; sha256 `^[0-9a-f]{64}$`; url `https://` (the app also takes `http://` when its catalog source is `insecure`, a test build; the catalog script requires `https://huggingface.co/`); licence name 1–80 and url `https://`; min_app_version SemVer; system_prompt null or ≤ 4000 chars. `parameters` is NOT in format 1: the app has nowhere to apply them yet (Ruling to ledger: a later format adds it with the app that uses it). The app DROPS a bad entry (logged with the rule it broke) and keeps the rest; the catalog script REFUSES to write a catalog with any problem.

## Review Focus

1. **A paused or retried download whose redirect target has expired.** Hugging Face's CDN address is signed and short-lived: every request, the first and every resume, starts again from the catalog's `url` and follows the redirect afresh; nothing stores the CDN address. Test: Task 5 `a_resume_asks_the_catalog_url_again_not_the_redirect_target`.
2. **GET pressed twice (a double click, or two windows of state).** One worker per entry: `start` on an entry that is running returns its view and starts nothing; there are never two writers on one `.part`. Test: Task 5 `starting_a_running_download_again_starts_nothing`.
3. **A file of that name is already in the downloads folder** (copied there by hand, or a done row lost). It is never overwritten: Discover shows ON DISK (no GET), and `start` refuses. Test: Task 5 `an_existing_file_is_never_overwritten`.
4. **The disk fills during the download.** A write that fails with ERROR_DISK_FULL (os error 112) or ERROR_HANDLE_DISK_FULL (39) is FAILED "the disk is full" with the `.part` kept and resumable — not a retry loop that hammers the network. Test: Task 5 `a_full_disk_fails_once_and_keeps_the_part`.
5. **Discover opened offline.** The last VERIFIED catalog (kept in the store and verified again when read) is shown, marked stale with the time it was read and the reason; an unverified or tampered cache is never shown. Test: Task 4 `offline_discover_shows_the_cached_catalog_verified_again`.

---

## Execution order and the demo gate

Task 1 (the demo) goes first and is handed to ghost: he verifies a design by USING a demo. Tasks 2–6 (no UI) do not wait for his verdict. **Tasks 7 and 8 start only after ghost has approved the demo**, and the approved demo is their specification for layout and words; any change he asks for is folded into the demo first. Task 9 is last. Heavy steps (`cargo test`, `cargo build`, the e2e) are gated.

## File map

| file | change | responsibility |
|---|---|---|
| `design/lorren-console-demo.html` | modify | Task 1: Discover, the empty-library chat, the Settings tab, the badge's new states, as a clickable demo |
| `desktop/release/catalog.sources.json` | create | the two sources (the table above) |
| `desktop/scripts/release/catalog.mjs` | create | `npm run catalog`: build, validate, sign, write into the master folder |
| `desktop/scripts/release/check.mjs` | modify | the feed check verifies the catalog's signature and entries |
| `desktop/scripts/release/release.mjs` | modify | `catalogPublicKey` in the config; the `lorren-catalog` overlay section |
| `desktop/release/release.config.json`, `release.test.json`, `README.md` | modify | the catalog key's place; the catalog section of the README |
| `desktop/scripts/update-feed.py` | modify | `keys` also makes `.keys/catalog.key`; writes `src-tauri/tauri.catalog-test.conf.json` |
| `desktop/src-tauri/src/store.rs` | modify | schema 2: `settings`, `catalog`, `downloads` |
| `desktop/src-tauri/src/paths.rs` | modify | `downloads_dir()` overridable for tests |
| `desktop/src-tauri/src/library.rs` | modify | the downloads folder is always watched; `add_file` |
| `desktop/src-tauri/src/catalog.rs` | create | the catalog's source, verification, entry rules, fetch, cache, view |
| `desktop/src-tauri/src/downloads.rs` | create | the download manager |
| `desktop/src-tauri/src/updater.rs` | modify | `UpdateMode`; `Off` and `Available`; `download_offered` |
| `desktop/src-tauri/src/commands.rs`, `lib.rs` | modify | the new commands, services, startup reconciliation, the mode-aware update loop |
| `desktop/src-tauri/Cargo.toml` | modify | `sha2 = "0.10.9"`, `semver = "1.0.28"`, `tauri-plugin-opener = "2"`, windows feature `Win32_Storage_FileSystem` |
| `desktop/src-tauri/tests/fixtures/catalog/*` | create | signed catalog fixtures (Task 4) |
| `desktop/src/tabs/Discover.tsx`, `ModelsTab.tsx`, `SettingsTab.tsx`, `chat/NoModel.tsx`, `chat/ChatTab.tsx`, `components/UpdateBadge.tsx`, `App.tsx`, `ipc/api.ts` | create/modify | Tasks 7 and 8 |
| `desktop/e2e/catalog-server.mjs` | create | Task 9: a local catalog and file server with Range, a pace, one drop, a request log |
| `desktop/e2e/run-demo-e2e.mjs` | modify | Task 9: the stranger's first run and the update modes |

---

### Task 1: The demo (Discover, the empty-library chat, Settings) for ghost

**Files:** Modify `design/lorren-console-demo.html` (the M1 reference demo: `TABS`, `render()`, `modelsTab()`; plain HTML + JS, no build).

**Interfaces:** Produces the approved look and words for Tasks 7 and 8 (they cite it).

The demo starts as a stranger's first run (an empty library) and must show, with the real values of the table above and in the M1 demo's own style:
- [ ] **Step 1: The empty-library chat.** Chat with no model: the transcript says there is no model yet and how to get one, with a button to Discover; the composer is disabled with that reason.
- [ ] **Step 2: Discover in the Models tab.** The design's bottom bar (`stoke pull … FETCH`, `design/ControlRoom.dc.html:250-255`) becomes Discover's search; the list shows each entry's name, architecture, quantization, size and licence; the right panel shows the selected entry's description, licence (a link), where it comes from (the Hugging Face repo at its commit, as text), the disk room ("needs 2.3 GB · 412.0 GB free", the app's binary `gb`) and GET. States to click through: GET → DOWNLOADING (bar, done/total, speed, time left, PAUSE, CANCEL) → a simulated drop at 40 % ("connection lost · retrying in 2 s", then it continues from 40 %) → PAUSED (RESUME) → VERIFYING → IN LIBRARY (the row appears in the library table; Qwen gets DEFAULT); a toggle shows the two failures: the checksum one ("the file did not match its checksum and was deleted") and not enough room (GET disabled, the reason shown).
- [ ] **Step 3: The Settings tab.** A sixth tab: sections down the left (UPDATES only), fields on the right: AUTO / NOTIFY / OFF with one line each saying what it does (auto: downloads updates in the background, installs when you restart; notify: tells you when there is one and downloads only when you say; off: never checks), the default AUTO.
- [ ] **Step 4: The status bar's update badge** reflects the choice: off → "updates off"; notify → "update 0.1.1 available · DOWNLOAD", then the existing download/ready path.
- [ ] **Step 5: Hand it to ghost.** Give him the file's full path to open in his browser (a local file, no server), list the states to try in two lines, and record his verdict and every change he asks for in the ledger. Commit the demo (`design: Discover, the empty-library chat and the update setting, as a demo for ghost`).

---

### Task 2: The catalog script, and the release tooling that checks and compiles in the catalog

**Files:**
- Create: `desktop/release/catalog.sources.json`, `desktop/scripts/release/catalog.mjs`, `desktop/scripts/release/test/catalog.test.mjs`
- Modify: `desktop/scripts/release/check.mjs` (`checkFeed`), `desktop/scripts/release/release.mjs` (`readConfig`, the overlay at :141-143), `desktop/scripts/release/test/check.test.mjs`, `release.test.mjs`, `desktop/release/release.config.json`, `release.test.json`, `README.md`, `desktop/scripts/update-feed.py` (`keys`), `desktop/package.json` (`"catalog": "node scripts/release/catalog.mjs"`)

**Interfaces:**
- Consumes: `verify(data, signatureText, publicKeyText)`, `parsePublicKey` (minisign.mjs), `isVersion` (semver.mjs), `checkFeed`, `tauriOpenKey` pattern (password through `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`), `DEFAULT_MASTER` (release.mjs).
- Produces: `validateEntry(entry) -> string[]`; `async buildCatalog(sources, { fetchJson, now }) -> object` (throws one Error listing every problem); `catalogText(catalog) -> string` (`JSON.stringify(catalog, null, 2) + '\n'`); `signCatalog({ path, keyPath, password }) -> string` (the minisign TEXT, decoded from the Tauri CLI's base64 `.sig`, which it deletes); `async catalog(options, deps)`; `parseCatalogArgs(argv) -> { key, config, master, deploy }`; `checkCatalog({ site, catalogPublicKey }) -> string[]` (the catalog's own problems, each prefixed `catalog: `; the catalog script runs THIS after writing, since a fresh master folder has no channels yet) and `checkFeed({ site, feedUrl, appPublicKey, catalogPublicKey })` (which includes it; a config with no `catalogPublicKey` neither requires nor compiles in a catalog — the step-1 tests keep passing unchanged; README: run `npm run catalog` before the first release); the overlay's `plugins['lorren-catalog'] = { url: '<feedUrl>/catalog/catalog.json', pubkey: <catalog public key, Tauri form>, insecure: <true only with insecureHttp> }`. Site files: `catalog/catalog.json` and `catalog/catalog.json.minisig` (minisign text).

- [ ] **Step 1: Write the failing tests** in `catalog.test.mjs` (keys from `test/keys.mjs`; `fetchJson` faked with the tree listing `[{ path, size, lfs: { oid, size } }]` Hugging Face returns for `GET /api/models/<repo>/tree/<commit>`):
  - `the two sources build the catalog the table pins` — every field of both entries equals the plan's table; `url` is `https://huggingface.co/<repo>/resolve/<commit>/<path>`; `format` 1; `generated` = the fake clock's ISO time; the fetched URLs are exactly the two tree URLs.
  - `a file that is not in the tree, or not an LFS file, or whose LFS size disagrees, is refused with its source named`.
  - `validateEntry: each rule` — a table: one good entry, then one bad case per rule of "Entry rules" (an id with a capital, a file with `..`, a file with a `/`, `size` 0 and 68719476737, a 63-hex sha256, an `http://` url, a licence url on http, a non-SemVer min_app_version, a 4001-char system prompt, an architecture `qwen2vl`, duplicate ids, duplicate files), each giving exactly one problem naming the field.
  - `the catalog is signed, verified with the catalog key and written; the app key's signature is refused` — `catalog()` with deps faking `sign` writes `site/catalog/catalog.json` + `.minisig`, and a catalog signed with the APP key fails the write with `the catalog's signature does not verify with the catalog key (key id)` and leaves the previous catalog files as they were.
  - `--deploy is refused` with `deploying arrives with the Cloudflare project (step 4)`; `a missing --key is refused` with `no --key: the catalog is signed with the catalog key, read from its file`.
  - in `check.test.mjs`: `a feed whose catalog fails its signature, or has a bad entry, or has no .minisig, is not deployable` (three sentences, prefixed `catalog: `) and `with a catalog key configured, a feed without a catalog is not deployable` (`catalog: catalog/catalog.json is missing`).
  - in `release.test.mjs`: the overlay test gains `plugins['lorren-catalog']` (url from the feed url, pubkey from the config, `insecure: true` for the http test config and absent for https).
- [ ] **Step 2: Run them; they fail.** Run: `node --test --test-reporter=tap "scripts/release/test/*.test.mjs"` (in `desktop/`). Expected: the new tests fail (`catalog.mjs` does not exist; checkFeed has no catalog key); the 53 step-1 tests still pass.
- [ ] **Step 3: Implement** `catalog.mjs` (the sources file is the table above with its two descriptions, `min_app_version` `0.1.0`, `system_prompt` null), the `check.mjs` catalog check (signature with `verify`, then `validateEntry` per entry, then that every `file` is unique), the release config's `catalogPublicKey` (placeholder `release/keys/catalog.pub`; test `.keys/catalog.key.pub`; refused as a placeholder like `feedUrl`), the overlay section, `update-feed.py keys` making `catalog` beside `test` and `wrong` and writing `src-tauri/tauri.catalog-test.conf.json` = `{ "plugins": { "lorren-catalog": { "url": "http://127.0.0.1:8766/catalog/catalog.json", "pubkey": "<.keys/catalog.key.pub>", "insecure": true } } }`, and the README's catalog section (making the catalog key: the same `tauri signer generate` command as the app key, its own file and password, kept offline; `npm run catalog -- --key <path>`; what the check refuses).
- [ ] **Step 4: Run them; they pass.** Same command. Expected: every test passes (53 + the new ones); then run `python scripts/update-feed.py keys` once and confirm `.keys/catalog.key` exists and `src-tauri/tauri.catalog-test.conf.json` holds its public key (commit that file: a test public key is public).
- [ ] **Step 5: A real catalog from Hugging Face, signed with the test catalog key** (network reads only): `npm run catalog -- --key .keys/catalog.key --config release/release.test.json --master <scratch>` and read the written `catalog.json`: both entries' size and sha256 equal the table. Expected: written, verified, and the feed check passes. Commit (`release: the catalog script, the catalog in the feed check, and the catalog's address and key in the build`).

---

### Task 3: The store's second schema, the downloads folder, and adding one file to the library

**Files:** Modify `desktop/src-tauri/src/store.rs` (SCHEMA :107-121, `init` :192-206), `paths.rs` (`downloads_dir` :32-34), `library.rs` (`LibraryConfig::from_env` :54-71; new `add_file`), their `mod tests`.

**Interfaces:**
- Produces (store): tables
  ```sql
  CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
  CREATE TABLE IF NOT EXISTS catalog (id INTEGER PRIMARY KEY CHECK (id = 1), json BLOB NOT NULL, minisig TEXT NOT NULL, fetched_ms INTEGER NOT NULL);
  CREATE TABLE IF NOT EXISTS downloads (entry_id TEXT PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL, url TEXT NOT NULL,
    file TEXT NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL, system_prompt TEXT,
    state TEXT NOT NULL CHECK (state IN ('downloading', 'paused', 'failed', 'done')), error TEXT,
    created_ms INTEGER NOT NULL, updated_ms INTEGER NOT NULL);
  ```
  `user_version` 2; methods `get_setting(&self, key) -> rusqlite::Result<Option<String>>`, `set_setting(&self, key, value)`, `save_catalog(&self, json: &[u8], minisig: &str, fetched_ms: i64)`, `load_catalog(&self) -> rusqlite::Result<Option<(Vec<u8>, String, i64)>>`, `upsert_download(&self, row: &DownloadRow)`, `set_download_state(&self, entry_id, state: &str, error: Option<&str>)`, `downloads(&self) -> rusqlite::Result<Vec<DownloadRow>>`, `delete_download(&self, entry_id)`, `pause_running_downloads(&self) -> rusqlite::Result<usize>` (every `downloading` row becomes `paused`). `pub struct DownloadRow { entry_id, name, version: i64, url, file, size: u64, sha256, system_prompt: Option<String>, state: String, error: Option<String>, created_ms, updated_ms }`.
- Produces (paths): `downloads_dir()` = `LORREN_DOWNLOADS_DIR`, else `<LORREN_DATA_DIR>\models` when that is set, else `%USERPROFILE%\Lorren\models`; pure `downloads_dir_from(over: Option<PathBuf>, data_over: Option<PathBuf>, userprofile: PathBuf) -> PathBuf`.
- Produces (library): `from_env` always watches the downloads folder (appended when the folder list, the default or `LORREN_MODEL_DIRS`, does not already hold it); `pub fn add_file(lib: &mut Vec<ModelInfo>, path: &Path, default_prefix: &str) -> ModelInfo` — describes the file exactly as `scan_folder` would (the same `file:<lower-case path>` id, a sidecar Modelfile from its folder), inserts it in the library's order (runnable first, then by name), marks it default when no entry is, returns the entry; an id already present is returned unchanged and nothing is added.

- [ ] **Step 1: Write the failing tests:** `a_version_1_database_opens_as_version_2_and_keeps_its_chats` (make a v1 file with the M1 SCHEMA, write a chat, reopen: the chat is there, `user_version` is 2, the three tables exist); `settings_round_trip_and_a_missing_key_is_none`; `the_catalog_cache_keeps_one_row`; `downloads_rows_round_trip_and_running_ones_pause_at_start`; `downloads_dir_precedence` (the three cases); `the_downloads_folder_is_watched_even_with_model_dirs_set` (on `from_env` with the env set, through a helper that takes the env values as arguments); `add_file_gives_the_id_a_scan_gives` (write a fixture into a folder named with a space, `add_file`, then `scan` the folder: same id, same facts); `add_file_into_an_empty_library_makes_the_default`; `add_file_keeps_an_existing_default`; `add_file_twice_adds_once`; `add_file_reads_a_sidecar_modelfile`.
- [ ] **Step 2: Run them; they fail.** Run (gated): `cargo test --lib store:: paths:: library::` in `desktop/src-tauri`. Expected: compile errors for the missing items, the M1 tests unaffected.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them; they pass,** then the whole crate: `cargo test --lib`. Expected: all pass (the M1 count + the new ones).
- [ ] **Step 5: Commit** (`desktop: the store's second schema (settings, the catalog, downloads), the downloads folder always watched, one file added to the library`).

---

### Task 4: The catalog in the app

**Files:** Create `desktop/src-tauri/src/catalog.rs`, `desktop/src-tauri/tests/fixtures/catalog/` (`catalog.json`, `catalog.json.minisig` signed with the kit's `.keys/catalog.key`, `catalog.json.appkey.minisig` signed with `.keys/test.key` — the app's update test key —, `catalog.json.wrongkey.minisig` with `.keys/wrong.key`, `catalog-test.pub`); Modify `lib.rs` (the service), `commands.rs` (`catalog`, `refresh_catalog`, `open_licence`), `Cargo.toml` (`semver`, `tauri-plugin-opener`).

**Interfaces:**
- Consumes: `Store::save_catalog`/`load_catalog` (Task 3); the overlay section of Task 2.
- Produces: `pub const CATALOG_PLUGIN: &str = "lorren-catalog"`; `pub struct CatalogSource { pub url: String, pub public_key: String, pub insecure: bool }`; `pub fn source_from_config(section: Option<&serde_json::Value>) -> Option<CatalogSource>` (None without both `url` and `pubkey`); `#[ts(export)] pub struct CatalogEntry` (the document's fields, `licence: Licence { name, url }`, `source: EntrySource { repo, commit, path }`, `system_prompt: Option<String>`); `pub fn validate_entry(e: &CatalogEntry, insecure: bool) -> Result<(), String>`; `pub fn verify(json: &[u8], minisig: &str, public_key: &str, insecure: bool) -> Result<Catalog, CatalogError>` with `pub struct Catalog { pub generated: String, pub entries: Vec<CatalogEntry>, pub dropped: Vec<String> }` (`minisign_verify::PublicKey::decode` on the key text, either form, `Signature::decode`, `verify(json, &sig, false)`; refuse `format != 1`; drop and name bad entries and duplicates); `pub struct CatalogService` with `new(source: Option<CatalogSource>, store, monitor, app_version: semver::Version) -> Arc<Self>`, `pub fn view(&self) -> CatalogView` (from memory, else the store's cache verified again; no network), `pub async fn refresh(&self) -> CatalogView` (GET `url` and `url + ".minisig"`, 20 s timeout, http refused unless `insecure`; verified ⇒ cached and shown; a failure ⇒ the last verified catalog marked `stale` with `error`), `pub fn entry(&self, id) -> Option<CatalogEntry>`; `#[ts(export)] pub struct CatalogView { pub configured: bool, pub entries: Vec<CatalogEntry>, pub needs_app: Vec<String> /* ids whose min_app_version is newer than this app */, pub fetched_ms: Option<i64>, pub stale: bool, pub error: Option<String> }`. Commands: `catalog() -> CatalogView`, `async refresh_catalog() -> CatalogView`, `open_licence(entry_id) -> Result<(), String>` (opens the VERIFIED entry's `licence.url` with `tauri_plugin_opener::OpenerExt::opener().open_url`; the plugin's JS API is NOT given to the UI: `capabilities/default.json` stays `core:default`).

- [ ] **Step 1: Make the fixtures** with Task 2's `signCatalog` (password `lorren-test`, the M1 kit's): a three-entry catalog in format 1 — the plan's two entries, then a third whose `file` breaks the file rule (`../evil.gguf`) — signed once with each of the three keys. Record the command in a comment at the top of the tests.
- [ ] **Step 2: Write the failing tests:** `a_signed_catalog_verifies_and_the_bad_entry_is_dropped_by_name`; `one_changed_byte_is_refused`; `the_apps_update_key_cannot_sign_a_catalog` (the appkey fixture: refused, key id); `another_key_is_refused`; `format_2_is_refused`; `validate_entry_each_rule` (the same table as Task 2's `validateEntry`, ported); `http_is_taken_only_from_an_insecure_source`; `an_entry_for_a_newer_app_is_listed_in_needs_app` (app 0.0.1, entry `min_app_version` 0.1.0); `source_from_config_reads_the_overlay_and_none_without_it`; `offline_discover_shows_the_cached_catalog_verified_again` (Review Focus 5: a refresh against a closed port returns the cached entries, `stale: true`, the error; then a cache whose json was altered in the store gives `entries: []`, `error` set, `stale: false`); `the_catalog_is_read_only_by_refresh` (`CatalogService::new` and `view()` make no request: a test listener on the port accepts nothing).
- [ ] **Step 3: Run them; they fail.** `cargo test --lib catalog::` (gated). Expected: compile errors for the missing module.
- [ ] **Step 4: Implement,** register the commands, `.plugin(tauri_plugin_opener::init())`, the service in `AppState` (`catalog: Arc<CatalogService>`), built from `app.config().plugins.0.get(catalog::CATALOG_PLUGIN)`.
- [ ] **Step 5: Run them; they pass,** then `cargo test --lib`. Commit (`desktop: the model catalog, read only when Discover opens, verified with its own key, kept for offline`).

---

### Task 5: The download manager

**Files:** Create `desktop/src-tauri/src/downloads.rs` (with a `#[cfg(test)] mod server`: a tokio `TcpListener` HTTP/1.1 server that serves one body with Range, and can drop the connection once after N bytes, answer 500 once, ignore Range, stall, redirect, and records each request's path and Range header); Modify `commands.rs`, `lib.rs`, `Cargo.toml` (`sha2`, windows `Win32_Storage_FileSystem`).

**Interfaces:**
- Consumes: `CatalogEntry`, `validate_entry` (Task 4); `DownloadRow` and its store methods, `library::add_file`, `downloads_dir` (Task 3); `ChatService::start_model` (chat.rs:191).
- Produces:
  - `pub struct RetryPolicy { pub first: Duration, pub cap: Duration, pub stall: Duration }`, `RetryPolicy::DEFAULT` = 1 s, 30 s, 30 s; `pub fn backoff(p: &RetryPolicy, attempt: u32) -> Duration` = `min(cap, first × 2^attempt)` (1, 2, 4, 8, 16, 30, 30 … s); a `Retry-After` of a 429/503 is obeyed up to 300 s.
  - `pub const MARGIN_BYTES: u64 = 1 << 30` (a download never leaves less than 1 GiB free, as §5 keeps 1 GB of memory); `pub fn free_bytes(dir: &Path) -> std::io::Result<u64>` (`GetDiskFreeSpaceExW`); `pub fn room_check(free: u64, remaining: u64) -> Result<(), String>` — `"needs {remaining} with 1 GB to spare; {free} free"` in the app's `gb` format.
  - `#[ts(export)] pub struct DownloadView { pub entry_id: String, pub name: String, pub file: String, pub version: u32, pub state: String /* downloading | paused | verifying | done | failed */, pub done: u64, pub total: u64, pub bytes_per_sec: u64, pub retry_in_ms: Option<u64>, pub error: Option<String> }`; `#[ts(export)] pub struct DiskRoom { pub dir: String, pub free_bytes: u64 }`.
  - `pub struct DownloadManager`; `new(cfg: DownloadConfig { dir, insecure, policy, default_prefix }, store, library, emit: Arc<dyn Fn(&DownloadView) + Send + Sync>, on_model: Arc<dyn Fn(ModelInfo, bool /* the library had no runnable model */) + Send + Sync>) -> Arc<Self>`; `list(&self) -> Vec<DownloadView>`; `start(self: &Arc<Self>, entry: &CatalogEntry) -> Result<DownloadView, String>`; `pause(&self, id) -> Result<(), String>`; `resume(self: &Arc<Self>, id) -> Result<(), String>`; `cancel(&self, id) -> Result<(), String>` (the `.part` and the row go); `reconcile_at_start(&self)` (running rows → paused; a `.part` with no row is deleted; a row whose `.part` is gone restarts from 0 on resume).
  - Files: `<dir>/<file>.part` while it runs; `<dir>/<file>` once verified (a rename); a `<stem>.modelfile` beside it (`FROM <file>` and `SYSTEM """…"""`) when the entry has a system prompt. An UPDATE (the entry's done row has an older version) is saved as `<stem>.v<version>.gguf` beside the old file, which stays (the engine may have it loaded); the row becomes the new version's. A file of that name that is not an older version of this entry is never overwritten: `start` refuses.
  - Events: `"download-state"` (a `DownloadView`, at most every 250 ms and at every state change), `"library-changed"` (no payload) after `add_file`.
  - Commands: `downloads() -> Vec<DownloadView>`, `disk_room() -> DiskRoom`, `start_download(entry_id)`, `pause_download(entry_id)`, `resume_download(entry_id)`, `cancel_download(entry_id)` — `start_download` takes the entry from `CatalogService::entry` (only a verified entry can be downloaded) and refuses an id in `needs_app`.

The worker (the algorithm the signatures leave open):
```
have = len(.part)  (0 when absent); hasher = in memory if this session kept it, else sha256 of the .part's bytes read back
loop:
  stop if paused or cancelled (a CancellationToken; every await below selects on it)
  GET entry.url (ALWAYS the catalog's url: the redirect is followed afresh), Range: bytes=<have>- when have > 0
  206 with Content-Range starting at <have> -> append | 200 -> truncate the .part, reset the hasher, write from 0
  416 with have == size -> verify | 416 otherwise -> truncate, reset, retry at once
  408, 429, 5xx, a connection error, no byte for policy.stall -> retryable | any other status -> failed "<status> from the server"
  each chunk: write, hash, have += n; have > size -> failed "the server sent more than the catalog's size"
  have == size -> state verifying; digest != sha256 -> delete the .part, failed "the file did not match its checksum and was deleted"
               -> rename, add_file, row done, on_model(entry, library_had_no_runnable_model)
  retryable -> state downloading with retry_in_ms = backoff(attempt) (or Retry-After); sleep; attempt += 1 (reset to 0 after a chunk arrives)
  a write error 112 or 39 -> failed "the disk is full", the .part kept
```
`on_model` in `lib.rs`: emit `library-changed`; when the library had no runnable model before, `chat.start_model(&m)` (as startup starts the default: the stranger's first chat is then quick). Ledger it as a Ruling.

- [ ] **Step 1: Write the failing tests** (policy in milliseconds; the server's body is a fixture GGUF from `gguf::fixture::write` padded with zeros to 64 KiB, so that the library can describe the result): `a_download_completes_verifies_and_joins_the_library` (the final file, no `.part`, the row done, `library` holds it with `add_file`'s id, `on_model` called once with `true`); `a_dropped_connection_resumes_from_where_it_stopped` (the server's second request has `Range: bytes=<n>-` where n is what the first sent; the digest matches); `a_500_is_retried_after_the_backoff` (the second request comes no sooner than `first`); `backoff_doubles_to_the_cap`; `a_404_fails_without_a_retry` (one request); `a_server_that_ignores_range_restarts_from_zero_and_still_verifies`; `a_checksum_mismatch_deletes_the_file_and_says_so` (no `.part`, no final file, the message); `pause_stops_the_requests_and_resume_finishes` (no request while paused for 3 × stall); `cancel_removes_the_part_and_the_row`; `a_stall_is_a_dropped_connection`; `more_bytes_than_the_size_fail`; `not_enough_room_is_refused_before_any_request` (an injected `free_bytes` of size − 1 + MARGIN; zero requests; the room message); `a_restart_pauses_a_running_download_and_resume_rehashes_the_part` (a new manager over the same store and folder: paused; resume sends `Range: bytes=<len>-`, the digest still matches); `a_part_with_no_row_is_deleted_at_start`; `an_entry_with_a_system_prompt_gets_a_modelfile_the_library_reads`; `an_update_downloads_beside_the_old_version` (a done row at version 1 and its file; the entry at version 2: saved as `<stem>.v2.gguf`, the old file's bytes unchanged, the library holds both, the row at 2); and the Review Focus tests `a_resume_asks_the_catalog_url_again_not_the_redirect_target` (the server redirects `/file` to `/cdn?once=<n>` that answers only once: every request after the first is for `/file` again), `starting_a_running_download_again_starts_nothing` (two `start`s, one request stream, one view), `an_existing_file_is_never_overwritten` (`start` refuses with `"<file> is already in <dir>"`; the file's bytes unchanged), `a_full_disk_fails_once_and_keeps_the_part` (an injected writer that fails with raw os error 112 after 16 KiB: FAILED "the disk is full", one request, the `.part` kept).
- [ ] **Step 2: Run them; they fail.** `cargo test --lib downloads::` (gated). Expected: compile errors.
- [ ] **Step 3: Implement** (the writer and `free_bytes` are injectable through `DownloadConfig` for the tests; production uses `std::fs::File` and `GetDiskFreeSpaceExW`), register the commands, build the manager in `lib.rs` and call `reconcile_at_start` before the window is told anything.
- [ ] **Step 4: Run them; they pass,** then `cargo test --lib` and `cargo build` (gated). Commit (`desktop: downloads that resume, retry with backoff, check sha256 as the bytes arrive, and join the library`).

---

### Task 6: The update setting in the update service

**Files:** Modify `desktop/src-tauri/src/updater.rs`, `lib.rs` (the loop :78-92), `commands.rs`.

**Interfaces:**
- Consumes: `Store::get_setting`/`set_setting` (Task 3).
- Produces: `#[ts(export)] #[serde(rename_all = "lowercase")] pub enum UpdateMode { Auto, Notify, Off }`; `pub const MODE_KEY: &str = "updates.mode"`; `UpdateMode::parse(&str) -> UpdateMode` (anything unknown is Auto, the default); new states `UpdateState::Off` and `UpdateState::Available { version }`; `UpdateService::new(monitor, configured, mode)`; `mode(&self)`, `set_mode<R>(&self, app, mode)` (stored, then the state: Off ⇒ `Off`; from Off to another ⇒ `Idle` and the loop is woken to check now); `pub fn wants_check(mode: UpdateMode) -> bool`; in `check()`: Off ⇒ returns before any request with the state `Off`; Notify ⇒ an offered update is kept (`offered: Mutex<Option<Update>>`) and the state is `Available { version }`, nothing downloaded; Auto ⇒ as today; `pub async fn download_offered<R>(self: &Arc<Self>, app) -> Result<(), String>` (the kept update downloaded and verified exactly as Auto does, ⇒ `Ready`; `"no update is waiting to be downloaded"` when there is none); a download already running when the mode changes runs to its end. The loop: `tokio::sync::Notify` woken by `set_mode`; each turn checks only when `wants_check(mode)`. Commands: `update_mode() -> UpdateMode`, `set_update_mode(mode)`, `async download_update()`.

- [ ] **Step 1: Write the failing tests:** `the_mode_is_auto_unless_the_store_says_otherwise` (`parse` of `auto`, `notify`, `off`, `""`, `"OFF?"`); `wants_check_is_false_only_for_off`; `off_makes_no_request` (mode Off, on a `tauri::test::mock_app` with NO updater plugin: `check()` ⇒ state `Off`; had it asked the plugin, the state would be `Failed`); `turning_off_back_on_wakes_the_loop` (the `Notify` is notified on Off → Auto, not on Auto → Notify); `download_offered_with_nothing_kept_says_so`; the ts-rs export of `UpdateState` carries `off` and `available`.
- [ ] **Step 2: Run them; they fail.** `cargo test --lib updater::` (gated).
- [ ] **Step 3: Implement;** read the mode from the store in `lib.rs` before the service is built.
- [ ] **Step 4: Run them; they pass,** then `cargo test --lib`. Commit (`desktop: the update setting (auto, notify, off), kept in the store, obeyed by the update service`).

---

### Task 7: Discover and the empty-library chat (after ghost's verdict on the demo)

**Files:** Create `desktop/src/tabs/Discover.tsx`, `desktop/src/chat/NoModel.tsx`, `desktop/src/__tests__/Discover.test.tsx`, `NoModel.test.tsx`; Modify `desktop/src/tabs/ModelsTab.tsx`, `desktop/src/chat/ChatTab.tsx`, `desktop/src/App.tsx`, `desktop/src/ipc/api.ts`, `desktop/src/__tests__/testApi.ts`, `desktop/src/tabs/tabs.css`.

**Interfaces:**
- Consumes: the commands and events of Tasks 4 and 5; the generated bindings `CatalogView`, `CatalogEntry`, `DownloadView`, `DiskRoom`.
- Produces: `api.catalog()`, `api.refreshCatalog()`, `api.downloads()`, `api.diskRoom()`, `api.startDownload(id)`, `api.pauseDownload(id)`, `api.resumeDownload(id)`, `api.cancelDownload(id)`, `api.openLicence(id)`, `api.onDownloadState(cb)`, `api.onLibraryChanged(cb)`; `ModelsTab({ overview, view, onView })` with `view: "library" | "discover"`; `ChatTab` gains `onDiscover: () => void`; App holds the Models view so the chat's button can open Discover.

The layout and words are the approved demo's (Task 1). The behaviour the tests pin:
- [ ] **Step 1: Write the failing tests** (vitest + the `testApi` fake): `opening_discover_reads_the_catalog_once_and_the_library_tab_never_does` (Global Constraints: no network unless Discover is shown); `each_entry_shows_size_licence_and_disk_room_before_GET` (the Qwen row: "2.3 GB", "Apache-2.0", the room line from `diskRoom`); `GET_starts_and_the_row_follows_download_state_events` (DOWNLOADING with done/total and speed; PAUSE ⇒ `pauseDownload`; RESUME; CANCEL); `retry_in_ms_shows_the_retry_line`; `a_failed_download_shows_its_error_and_offers_GET_again`; `an_entry_on_disk_or_in_the_library_has_no_GET` (the file name already a library entry's file name ⇒ ON DISK; a done download of the same version ⇒ IN LIBRARY; an older version ⇒ UPDATE AVAILABLE with GET); `an_entry_for_a_newer_app_says_so_and_cannot_be_started`; `a_stale_catalog_says_when_it_was_read_and_why`; `the_licence_link_calls_openLicence_and_navigates_nowhere`; `library_changed_reloads_the_library_table`; `the_library_table_shows_DOWNLOADING_for_a_running_download`; `with_no_runnable_model_the_chat_says_so_and_its_button_opens_discover` (the composer disabled with the reason); `the_models_tab_opens_on_discover_when_the_library_is_empty`.
- [ ] **Step 2: Run them; they fail.** `npx vitest run src/__tests__/Discover.test.tsx src/__tests__/NoModel.test.tsx` (in `desktop/`).
- [ ] **Step 3: Implement** to the demo; progress redraws throttled by the 250 ms events (no timer of the UI's own).
- [ ] **Step 4: Run them; they pass,** then `npm test` (every frontend test) and `npm run build` (tsc). Commit (`desktop: Discover in the Models tab, and the chat's way to it when there is no model yet`).

---

### Task 8: The Settings tab and the update badge (after ghost's verdict on the demo)

**Files:** Create `desktop/src/tabs/SettingsTab.tsx`, `desktop/src/__tests__/SettingsTab.test.tsx`; Modify `desktop/src/App.tsx` (`renderTab`: Settings ⇒ `SettingsTab`; Swarm keeps its Placeholder), `desktop/src/components/Placeholder.tsx` (Settings' sentence goes), `desktop/src/components/UpdateBadge.tsx`, `desktop/src/__tests__/UpdateBadge.test.tsx`, `desktop/src/ipc/api.ts`.

**Interfaces:** Consumes `update_mode`, `set_update_mode`, `download_update` (Task 6), the bindings `UpdateMode`, `UpdateState`. Produces `api.updateMode()`, `api.setUpdateMode(mode)`, `api.downloadUpdate()`.

- [ ] **Step 1: Write the failing tests:** `settings_shows_the_updates_section_with_the_stored_mode` (AUTO selected by default; the three lines of the demo); `choosing_off_calls_setUpdateMode_once_and_shows_it`; `a_failed_set_is_shown_and_the_choice_goes_back`; `the_badge_says_updates_off_in_off_and_offers_nothing`; `the_badge_offers_DOWNLOAD_for_available_and_calls_downloadUpdate`; the existing UpdateBadge tests still pass.
- [ ] **Step 2: Run them; they fail.** `npx vitest run src/__tests__/SettingsTab.test.tsx src/__tests__/UpdateBadge.test.tsx`.
- [ ] **Step 3: Implement** to the demo.
- [ ] **Step 4: Run them; they pass,** then `npm test` and `npm run build`. Commit (`desktop: Settings with the update choice, and the badge's off and available states`).

---

### Task 9: The stranger's first run, end to end

**Files:** Create `desktop/e2e/catalog-server.mjs`; Modify `desktop/e2e/run-demo-e2e.mjs` (the build at :702 adds `--config src-tauri/tauri.catalog-test.conf.json`; new steps after the M1 ones), `desktop/package.json` if a script is added.

**Interfaces:** Consumes everything above; `signCatalog` and `catalogText` (Task 2) to sign the test catalog with `.keys/catalog.key`.

`catalog-server.mjs <root> --port 8766 [--pace <bytes per second>] [--drop <file>:<bytes>]` serves `<root>/catalog/*` and `<root>/files/*` with Range, paces bodies, drops the first request of `<file>` after `<bytes>`, and appends every request (`{ t, method, path, range }`) as a JSON line to `<root>/requests.log`. The e2e writes `<work>/catalog-site/` before the launch, four entries, each a fake-engine fixture padded with zeros to 3 MiB, `min_app_version` `0.0.1`, urls `http://127.0.0.1:8766/files/<file>`, signed: the Qwen and Llama fixtures under the library's file names, `broken-test` (`Broken-Test-Q4_K_M.gguf`, its sha256 that of other bytes) and `restart-test` (`Restart-Test-Q4_K_M.gguf`, a llama fixture). The server's pace is 1 MB/s.

- [ ] **Step 1: Write the steps** (each prints one line, as the M1 steps do):
  1. `a stranger's first run: no model, the chat says how to get one, and nothing has gone online` — relaunch with `LORREN_MODEL_DIRS` an empty folder and `LORREN_DOWNLOADS_DIR` `<work>/downloads`; the chat's no-model text and button; after 6 s `requests.log` is empty.
  2. `Discover lists the catalog with size, licence and room` — the button opens Discover; four entries; `requests.log` has the catalog and its `.minisig`, nothing else.
  3. `a download survives a dropped connection` — the server started with `--drop Qwen3-4B-Instruct-2507-Q4_K_M.gguf:1200000`; GET; the row reaches IN LIBRARY; `requests.log` has a second request for the file with `range` `bytes=1200000-`; the file in `<work>/downloads` has the fixture's sha256; the library table lists it as DEFAULT; the top bar reaches serving it.
  4. `a chat streams on the downloaded model` — send; the reply streams (the fake engine).
  5. `pause stops the bytes and resume finishes` — GET Llama; PAUSE at > 0 %; for 3 s no new request and the file's `.part` size is unchanged; RESUME; IN LIBRARY.
  6. `a checksum mismatch deletes the file and says so` — GET broken-test; its error text; neither `broken`'s file nor its `.part` in `<work>/downloads`.
  7. `a download running at close is paused at the next start and resumes` — GET restart-test, close the app at its first progress (> 0 %), relaunch: PAUSED, and no request for it until RESUME; RESUME; IN LIBRARY with the right sha256.
  8. `updates off makes no request; notify checks and downloads nothing until asked` — serve the M1 kit's good feed with `catalog-server.mjs` on 8765 (its request log); Settings ⇒ OFF; relaunch; after 8 s the feed's log is empty and the badge says updates off; Settings ⇒ NOTIFY; relaunch; the feed's log has `latest.json` and not the installer; the badge offers DOWNLOAD; press it: the installer is requested and the badge reaches ready (no install).
- [ ] **Step 2: Run the whole e2e** (gated; builds the e2e app): `npm run e2e` in `desktop/`. Expected: every M1 step and the eight new ones OK, the last line `e2e: <n>/<n> OK`. A failing step is debugged at its root (superpowers:systematic-debugging), never loosened.
- [ ] **Step 3: Commit** (`desktop: the stranger's first run end to end: the catalog, a download through a dropped connection, pause and resume, a bad checksum, a restart mid-download, the update modes`).

---

## After this plan (not in it)

- Step 3's website half in `company`: name the catalog read in the privacy sentence (`src/site/legal.js:25` — "update checks …, update downloads, the model list when you open Discover, and the model downloads you start, from Hugging Face"), check the badges against the v0.1 build, `TODAY_CONFIRMED_FOR = '0.1.0'`.
- Step 4 (ghost's Cloudflare account, the feed name, the two real key pairs): the real catalog key's public half replaces the placeholder in `release/release.config.json`, and the first real `npm run catalog` signs the real catalog.
