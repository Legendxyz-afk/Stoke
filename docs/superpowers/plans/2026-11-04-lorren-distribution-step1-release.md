# Lorren distribution, step 1: the release command and the feed - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `npm run release -- <version>` turns a version and its notes into a signed installer laid out in the update feed's master
folder, checks the whole feed, shows the deploy it would make, and proves the result by fetching it from a local server: everything
the real release needs except the Cloudflare project and the real keys (step 4).

**Architecture:** Node ES modules in `desktop/scripts/release/`, no dependencies (Node 24's `crypto` does Ed25519 and BLAKE2b-512):
`minisign.mjs` verifies, `semver.mjs` orders versions, `feed.mjs` lays a release out, `check.mjs` checks the whole feed and keeps the
deploy manifest, `release.mjs` is the command. The Tauri build is a step the command calls through an injectable builder, so every
other step is tested without compiling the app; the real build runs once, in the sweep (Task 5).

**Tech Stack:** Node 24 (`crypto`, `node:test`, `node:http`, `child_process.spawnSync` with argument arrays), the project's Tauri
CLI 2.12 through node (`node_modules/@tauri-apps/cli/tauri.js`), git.

**Spec:** `docs/superpowers/specs/2026-11-04-lorren-distribution-first-design.md` (step 1 of its Order), with
`docs/superpowers/specs/2026-09-25-lorren-desktop-design.md` §3.7.

## Global Constraints

- The deployed folder is `<master>/site/`; `<master>/state/` holds what is not deployed (the last deploy's manifest). The master
  folder is `M:\New Drive\Desktop\Programming\lorren-feed` unless `--master` says otherwise; tests use temp folders.
- Layout under `site/`: `app/<channel>/Lorren_<version>_x64-setup.exe` and `.sig`; `app/<channel>/latest.json`;
  `app/<channel>/releases.json`; `app/stable/Lorren-setup.exe` (a byte copy of the newest stable installer, stable only);
  `catalog/` (step 2); `_headers`; `index.html`. Channels: `stable` (default) and `beta`.
- `latest.json` is the M1 test kit's shape, which the updater accepted (`scripts/update-feed.py` `write_feed`):
  `{ version, notes, pub_date, platforms: { "windows-x86_64": { signature, url } } }`; `notes` = the notes joined by `\n`;
  `pub_date` = the release moment, `Date#toISOString()`; `url` = `<feedUrl>/app/<channel>/Lorren_<version>_x64-setup.exe`;
  `signature` = the `.sig` file's text, trimmed (Tauri writes it as base64 of the minisign text).
- `releases.json` = `[{ version, date, notes }]`, newest first, `date` = the releasing machine's LOCAL date `YYYY-MM-DD`, `notes` a
  list of non-empty strings: the contract `company/src/site/releases.js` reads. It keeps every release; only installers are pruned.
- Each channel keeps the 2 newest installer pairs; older pairs are removed.
- `_headers` (exact): `/app/stable/Lorren-setup.exe` then `  Content-Disposition: attachment; filename="Lorren-setup.exe"`. Pages
  already revalidates every file by default (`max-age=0, must-revalidate`), so the spec's revalidate rules add nothing, and an
  immutable rule for the versioned files would overlap the others and have its values combined (builder's call, refining the spec).
- `index.html` (exact): `<!doctype html><meta charset="utf-8"><title>Lorren updates</title><p>This is Lorren's update feed: there is
  nothing to read here. Lorren comes from the castagents website.</p>` (one line).
- The release configuration is committed as JSON and turned into the Tauri overlay at build time (passed as `--config <json>`, as
  the test kit passes the version), refining the spec's committed `tauri.release.conf.json`: `desktop/release/release.config.json`
  = `{ "feedUrl": "https://<feed>.pages.dev", "appPublicKey": "release/keys/app.pub", "pagesProject": "<feed>" }` (placeholders
  until ghost names the feed: the command refuses them), `desktop/release/release.test.json` = `{ "feedUrl":
  "http://127.0.0.1:8765", "appPublicKey": ".keys/test.key.pub", "pagesProject": "lorren-test", "insecureHttp": true }`. Paths in a
  config are relative to `desktop/`.
- The private key is a path given at run time (`--key`), never copied into the repository or the master folder; its password comes
  from `LORREN_KEY_PASSWORD`.
- Every subprocess is `spawnSync(file, [args...])`: the repository is under `M:\New Drive\`, a path with a space.
- The deploy is a plan in this step: the command prints `npx wrangler pages deploy <master>/site --project-name <pagesProject>
  --branch main` and runs nothing; `--deploy` is refused until step 4. The tag `lorren-v<version>` is local; nothing is pushed.

## Review Focus

1. A release run twice, or again after an interrupted run: the same version is refused ("0.1.0 is already in the stable feed") and
   changes nothing (Task 2). (Reporting a laid-out release that was never deployed needs deploys, so it is step 4's, on Task 3's
   `state/` manifest.)
2. Paths with spaces in every command: the integration tests' temp master folder has a space in its name.
3. Windows line ends and BOMs in key and signature files: a CRLF or BOM `.pub`/`.sig` still verifies (Task 1).
4. Release notes as people write them (`- ` and `* ` bullets, blank lines, CRLF, trailing spaces): normalized to non-empty strings;
   a notes file with none is refused, because the website drops a release whose notes are empty (Task 2).
5. A feed URL written with a trailing slash: no `//` in `latest.json`'s url (Task 2).

---

### Task 1: Minisign verification

**Files:**
- Create: `desktop/scripts/release/minisign.mjs`
- Test: `desktop/scripts/release/test/minisign.test.mjs`

**Interfaces:**
- Produces: `parsePublicKey(text: string) -> { keyId: Buffer(8), key: Buffer(32) }`; `parseSignature(text: string) -> { algorithm:
  'Ed'|'ED', keyId: Buffer(8), signature: Buffer(64), trustedComment: string, globalSignature: Buffer(64) }`; `verify(data:
  Buffer, signatureText: string, publicKeyText: string) -> { ok: true } | { ok: false, reason: 'format'|'key id'|'signature'|
  'trusted comment' }`. Each text may be the minisign text itself or Tauri's base64 of it (a text that does not start with
  `untrusted comment:` after a BOM and whitespace is base64-decoded first).
- Format: a public key is `untrusted comment: ...` then base64(`Ed` (2 bytes) | key id (8) | key (32)); a signature is `untrusted
  comment: ...`, base64(algorithm (2) | key id (8) | signature (64)), `trusted comment: <text>`, base64(global signature (64)).
  `ED` signs BLAKE2b-512 of the data, `Ed` the data itself; the global signature signs signature | utf8(text). Ed25519 through
  `crypto.verify(null, message, createPublicKey({ key: Buffer.concat([Buffer.from('302a300506032b6570032100', 'hex'), key]),
  format: 'der', type: 'spki' }), signature)`.

- [ ] **Step 1: Write the failing tests.** In the test file, a helper `makeMinisign(data, { prehashed = true, keyPair, keyId,
  trusted = 'timestamp:1\tfile:x' })` returns `{ publicKeyText, signatureText }` from an `crypto.generateKeyPairSync('ed25519')`
  pair (raw key: the last 32 bytes of the SPKI DER export). Tests:
  - `a signature made over the file verifies` -> `{ ok: true }`;
  - `one flipped byte in the file fails on the signature` -> `{ ok: false, reason: 'signature' }`;
  - `another key fails on the key id` -> reason `'key id'`;
  - `an edited trusted comment fails` -> reason `'trusted comment'`;
  - `a legacy Ed signature (the data unhashed) verifies` -> ok;
  - `CRLF line ends and a BOM in either text still verify` (Review Focus 3) -> ok;
  - `Tauri's base64 form of both texts verifies` -> ok;
  - `the M1 kit's 0.0.2 installer verifies against its test key, and not against the wrong one`: reads
    `desktop/e2e/.work/builds/0.0.2/Lorren_0.0.2_x64-setup.exe` and its `.sig`, `desktop/.keys/test.key.pub` -> ok;
    `desktop/.keys/wrong.key.pub` -> reason `'key id'`; skips with `t.skip('the M1 test kit has not been built here')` when the
    files are absent.
- [ ] **Step 2: Run them to see them fail.** `node --test --test-reporter=tap "desktop/scripts/release/test/*.test.mjs"` (from `stoke-lorren/`). Expected: FAIL,
  `Cannot find module .../minisign.mjs`.
- [ ] **Step 3: Implement `minisign.mjs`** as the Interfaces say. A text that is not a well-formed key or signature is
  `{ ok: false, reason: 'format' }`, never a throw.
- [ ] **Step 4: Run them to see them pass.** Same command: all pass (the M1 one may skip).
- [ ] **Step 5: Commit** `desktop/scripts/release/minisign.mjs` and its test: `release: verify a minisign signature in Node (no
  dependency), as Tauri signs and as the updater checks`.

### Task 2: Versions and the feed layout

**Files:**
- Create: `desktop/scripts/release/semver.mjs`, `desktop/scripts/release/feed.mjs`
- Test: `desktop/scripts/release/test/semver.test.mjs`, `desktop/scripts/release/test/feed.test.mjs`

**Interfaces:**
- Produces (`semver.mjs`): `isVersion(v: string) -> boolean` (`x.y.z` with an optional `-prerelease`, no `+build`, no `v`);
  `compareVersions(a, b) -> number` (SemVer 2.0.0 precedence, section 11).
- Produces (`feed.mjs`): `INSTALLER = (version) => \`Lorren_${version}_x64-setup.exe\``; `STABLE_COPY = 'Lorren-setup.exe'`;
  `KEEP = 2`; `HEADERS` and `INDEX_HTML` (the exact texts of Global Constraints); `parseNotes(text: string) -> string[]`;
  `channelState(site: string, channel: string) -> { latest: object|null, releases: object[] }`; `layoutRelease({ site, channel,
  version, notes: string[], installer: string (a path), signature: string, feedUrl: string, now: Date }) -> { written: string[],
  pruned: string[] }`.
- `layoutRelease` throws, before writing anything: `no release notes` (empty `notes`); `<version> is not a version`;
  `<version> is already in the <channel> feed`; `<version> is not newer than <current>, the <channel> feed's newest`.

- [ ] **Step 1: Write the failing tests.**
  - semver, as a table: `0.1.0 < 0.1.1`, `0.1.0-beta.1 < 0.1.0`, `0.1.0-beta.2 < 0.1.0-beta.10`, `0.1.0-alpha < 0.1.0-beta`,
    `0.9.0 < 0.10.0`, `1.0.0 == 1.0.0`; `isVersion` false for `0.1`, `v0.1.0`, `0.1.0+build`, `01.0.0`.
  - `parseNotes("- First\r\n\r\n* Second  \n   \n")` deep-equals `['First', 'Second']` (Review Focus 4).
  - `a stable release is laid out exactly`: a temp site under a folder named `lorren feed` (Review Focus 2), a fake installer
    (`Buffer.from('MZ fake 0.1.0')`), signature `'c2lnLTAuMS4w'`, feedUrl `'http://127.0.0.1:8765/'` (trailing slash: Review
    Focus 5), `now` = local noon on 2026-11-20 (`new Date(2026, 10, 20, 12)`). `latest.json` deep-equals `{ version: '0.1.0',
    notes: 'First\nSecond', pub_date: now.toISOString(), platforms: { 'windows-x86_64': { signature: 'c2lnLTAuMS4w', url:
    'http://127.0.0.1:8765/app/stable/Lorren_0.1.0_x64-setup.exe' } } }`; `releases.json` deep-equals `[{ version: '0.1.0', date:
    '2026-11-20', notes: ['First', 'Second'] }]`; the `.sig` reads `c2lnLTAuMS4w`; `Lorren-setup.exe` has the installer's bytes;
    `_headers` and `index.html` equal `HEADERS` and `INDEX_HTML`.
  - `the third release prunes the first installer and keeps every release note`: 0.1.0, 0.1.1, 0.1.2 -> only the 0.1.1 and 0.1.2
    pairs remain, `releases.json` lists 0.1.2, 0.1.1, 0.1.0, `latest.json` is 0.1.2, `Lorren-setup.exe` has 0.1.2's bytes,
    `pruned` lists the 0.1.0 pair.
  - `the same version twice is refused and changes nothing` (Review Focus 1): every file's bytes before equal those after.
  - `an older version is refused`: 0.1.2 then 0.1.1 throws `0.1.1 is not newer than 0.1.2, the stable feed's newest`.
  - `a beta release has no stable copy and leaves stable alone`: `app/stable` does not exist after a `0.1.0-beta.1` beta release.
  - `notes with nothing in them are refused`: `notes: []` throws `no release notes`.
- [ ] **Step 2: Run them to see them fail.** `node --test --test-reporter=tap "desktop/scripts/release/test/*.test.mjs"`: FAIL, the modules are missing.
- [ ] **Step 3: Implement `semver.mjs` and `feed.mjs`.** The feedUrl loses its trailing slashes before joining; `layoutRelease`
  copies the installer (`copyFileSync`), writes the JSON with two-space indentation and a final newline, prunes by version order
  (never by file date), and writes `_headers` and `index.html` on every release.
- [ ] **Step 4: Run them to see them pass.**
- [ ] **Step 5: Commit** the two modules and their tests: `release: the feed's layout (versioned pair, latest.json, releases.json,
  the stable copy, two kept per channel)`.

### Task 3: The whole-feed check and the deploy manifest

**Files:**
- Create: `desktop/scripts/release/check.mjs`
- Test: `desktop/scripts/release/test/check.test.mjs`

**Interfaces:**
- Consumes: `verify` (Task 1); `INSTALLER`, `STABLE_COPY`, `KEEP`, `channelState`, `layoutRelease` (Task 2, the last for the
  tests' fixtures).
- Produces: `checkFeed({ site, feedUrl, appPublicKey: string (its text) }) -> string[]` (the problems; empty is a good feed);
  `manifest(site) -> { [path: string]: string }` (each file under `site/`, `/`-separated, to its sha256 hex); `readDeployed(state)
  -> manifest|null`; `writeDeployed(state, manifest)`; `dropped(previous, current, pruned: string[]) -> string[]`.
- `checkFeed` reports, for each channel folder present: `app/<ch>/latest.json does not parse` (and the same for releases.json);
  `latest.json names <a> but releases.json starts at <b>`; `latest.json's url is not <expected>`; `the installer latest.json
  points at is not in the feed: <path>`; `latest.json's signature is not the .sig file's`; `<path> fails its signature (<reason>)`;
  `Lorren-setup.exe is not the newest stable installer`; `app/<ch> keeps <n> installers, more than 2`; and `_headers is missing` /
  `index.html is missing`. A `catalog/catalog.json` that does not parse is reported too (its signature check is step 2's).

- [ ] **Step 1: Write the failing tests.** A fixture `goodFeed()` lays out 0.1.0 then 0.1.1 stable with Task 2 and a real test
  signature (Task 1's `makeMinisign` moved to `test/keys.mjs` so both test files share it). Tests: `a good feed has no problems`
  -> `[]`; then one test per break, each asserting `checkFeed` returns exactly its message: releases.json edited to start at 0.1.0;
  the 0.1.1 installer deleted; latest.json's signature changed; one byte of the 0.1.1 installer flipped (`... fails its signature
  (signature)`); `Lorren-setup.exe` replaced by the 0.1.0 bytes; latest.json truncated; `_headers` deleted. `dropped`: a path in
  the previous manifest that is gone now and not in `pruned` is listed; a pruned one is not; an added one is not.
- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement `check.mjs`.**
- [ ] **Step 4: Run them to see them pass,** with Tasks 1 and 2's tests still passing.
- [ ] **Step 5: Commit**: `release: check the whole feed before a deploy, and never deploy one that drops a file`.

### Task 4: The release command

**Files:**
- Create: `desktop/scripts/release/release.mjs`, `desktop/release/release.config.json`, `desktop/release/release.test.json`,
  `desktop/release/README.md`
- Modify: `desktop/package.json` (`"release": "node scripts/release/release.mjs"` in `scripts`)
- Test: `desktop/scripts/release/test/release.test.mjs`

**Interfaces:**
- Consumes: Tasks 1 to 3.
- Produces: `parseArgs(argv: string[]) -> { version, channel, notes, key, config, master, deploy }` (`<version>` positional;
  `--channel stable|beta`, default stable; `--notes <file>`; `--key <file>`; `--config <file>`, default
  `release/release.config.json`; `--master <dir>`, default `M:\New Drive\Desktop\Programming\lorren-feed`; `--deploy`);
  `release(options, deps) -> { written, deployCommand: string[], fetched: string[] }` with `deps = { git(args) -> string,
  openKey(keyPath, password) -> void (throws if the key does not open), build({ version, overlay, keyPath, password }) -> {
  installer: string, signature: string }, now() -> Date, password: string }`; `tauriBuild` (the real `build`) and `tauriOpenKey`
  (the real `openKey`) exported for the command's own use.
- The order, each refusal changing nothing (the message is the test's expectation):
  1. the config: `release.config.json still has the placeholder feed name: name the feed first (step 4)` for a `<feed>` in it;
  2. `git status --porcelain` empty, else `the tree is not clean`;
  3. the version (Task 2's messages, checked against `channelState` before the build);
  4. the notes file: `no release notes`;
  5. `deps.openKey`: `the key does not open: <message>` (the real one signs a temp file with `tauri signer sign -f <key> -p
     <password>`);
  6. `deps.build` with the overlay `{ version, plugins: { updater: { pubkey: <the .pub file's text, trimmed>, endpoints:
     [<feedUrl>/app/<channel>/latest.json] } } }`, plus `dangerousInsecureTransportProtocol: true` only when the config says
     `insecureHttp`;
  7. `verify` of the built installer: `the build's signature does not verify (<reason>): nothing was laid out`;
  8. a copy of `site/` is taken, then `layoutRelease`;
  9. `checkFeed`: on any problem, `site/` is put back from the copy and the problems are thrown;
  10. `dropped` against `readDeployed(state)`: `this deploy would drop <paths>` (the copy is put back);
  11. `--deploy`: `deploying arrives with the Cloudflare project (step 4)`; else `deployCommand` = `['npx', 'wrangler', 'pages',
      'deploy', <master>/site, '--project-name', <pagesProject>, '--branch', 'main']`, printed;
  12. the local live check: `site/` served on a free 127.0.0.1 port (`node:http`, files only), `app/<channel>/latest.json`
      fetched (its version must be the release's) and, for stable, `app/stable/Lorren-setup.exe` (its sha256 must be the built
      installer's): `the served feed does not have <what>`; `fetched` lists the two paths;
  13. `git tag lorren-v<version>`.
- The real `tauriBuild` runs `node <desktop>/node_modules/@tauri-apps/cli/tauri.js build --config <JSON.stringify(overlay)>` with
  `TAURI_SIGNING_PRIVATE_KEY=<key path>` and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD=<password>` (the test kit's finding: the build
  reads only these two), and returns `src-tauri/target/release/bundle/nsis/Lorren_<version>_x64-setup.exe` and its `.sig` text.

- [ ] **Step 1: Write the failing tests** (deps faked: `git` records its calls and answers `''`; `build` writes a fake installer
  and signs it with a test pair from `test/keys.mjs`; the master folder is a temp folder named `lorren feed`):
  - `a stable release lays out, checks, plans the deploy, serves and tags`: `written` includes the five channel files;
    `deployCommand` deep-equals the list above with `pagesProject` `lorren-test`; `fetched` deep-equals
    `['app/stable/latest.json', 'app/stable/Lorren-setup.exe']`; git's last call is `['tag', 'lorren-v0.1.0']`;
  - `a dirty tree is refused before anything is built`: git answers ` M src/x.ts`; throws `the tree is not clean`; `build` never
    called;
  - `the placeholder feed name is refused`: a config with `https://<feed>.pages.dev`;
  - `a key that does not open is refused before the build`;
  - `a build whose signature does not verify lays out nothing`: `build` returns a signature from another key; the site's files
    are unchanged;
  - `a failed check puts the site back`: before the release, plant `app/beta/latest.json` = `{}` with no `releases.json` (a
    channel that fails the check); the release throws `checkFeed`'s problems, and every file of the site before equals after;
  - `--deploy is refused in this step`;
  - `parseArgs` defaults and a missing version (`usage: npm run release -- <version> --notes <file> --key <file> [--channel
    beta] [--config <file>] [--master <dir>]`).
- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement `release.mjs`, the two configs, the README and the npm script.** The README covers: making the two key
  pairs (`npm run tauri signer generate -- -w <USB>\lorren-app.key`, the password kept with the USB; the `.pub` copied to
  `desktop/release/keys/app.pub`; the catalog pair the same way, step 2), the master folder (`site/` deployed, `state/` not; back
  it up after every release: it holds installers no later build reproduces), a test release
  (`LORREN_KEY_PASSWORD=lorren-test npm run release -- 0.0.3 --channel beta --config release/release.test.json --notes notes.txt
  --key .keys/test.key --master <a temp folder>`), and what each refusal means.
- [ ] **Step 4: Run them to see them pass,** all four test files.
- [ ] **Step 5: Commit**: `release: npm run release -- <version>: preconditions, build, verify, layout, whole-feed check, the
  deploy it would make, a local live check, the tag`.

### Task 5: The sweep (on a free box: it compiles the app in release mode)

- [ ] **Step 1:** gate (no game, >= 6 GB free, the owner's Lorren closed), then the README's test release for real, with a
  temp master folder: the Tauri release build runs; `verify` passes on the real signature; the layout, the check, the printed
  deploy and the local live check pass; the tag `lorren-v0.0.3` is made, then deleted (`git tag -d`: a test release is not a
  release).
- [ ] **Step 2:** the four test files once more, and `npx vitest run` + `npx tsc --noEmit -p .` in `desktop/` (nothing under
  `src/` changed: both as before, 156 and 0).
- [ ] **Step 3:** the ledger: what ran, the counts, the build's time and installer size.
