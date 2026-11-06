// Milestone 1's checklist (spec §11.1), end to end against the FAKE engine: the real app window,
// driven through WebView2's DevTools port. Prints one line per step; the verdict is the last line.
//   npm run e2e            build, then run
//   npm run e2e -- --no-build
// The layout step ASSERTS the defects the 1100 px screenshots showed (R70: one-line top-bar items, tables that fit their
// column, uncut tags, a hanging Logs indent, a 300 px composer). The last step is a MEASUREMENT (R66): it fails only when a
// reading could not be taken, never on its value.
import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { chromium } from "playwright-core";

const DESKTOP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const TAURI_DIR = path.join(DESKTOP, "src-tauri");
const WORK = path.join(DESKTOP, "e2e", ".work");
const APP = path.join(TAURI_DIR, "target", "debug", "lorren.exe");
const FAKE = path.join(TAURI_DIR, "target", "debug", "fake-engine.exe");
const CDP = 9222;
// The stream step (demo D5): the fake engine's pace (55 ms a token is ~18 tok/s, a real engine's), where in the stream the UI's
// CPU is read and for how long, how long the same launch's idle is read first, and the CPU OVER THAT IDLE, as a percentage of one
// core, above which the step fails (R82; the idle is read at rest just after the reply. The idle baseline moves between sittings, 1.6-3.0% on 10-21 and 4.0% on 11-04, mostly the
// WebView2 browser process at rest; over it the streaming cost read 5.6% then and 6.2% now, and the code before D5 about 21%).
const STREAM_TOKEN_MS = 55;
const UI_CPU_SKIP_S = 10;
const UI_CPU_WINDOW_S = 20;
const UI_CPU_IDLE_S = 10;
const UI_CPU_LIMIT_PCT = 10;
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

// The CPU the UI uses (demo D5): lorren.exe and every process under it -- its WebView2 browser, renderer and GPU processes --
// with the engine left out. A snapshot lists every process with its parent and the CPU time it has used (user plus kernel, in
// 100 ns); two snapshots a window apart give the CPU used in between, as a percentage of one core.
function processSnapshot() {
  const script =
    "Get-CimInstance Win32_Process | ForEach-Object { [pscustomobject]@{ pid = [int]$_.ProcessId; ppid = [int]$_.ParentProcessId; " +
    "name = $_.Name; cmd = $_.CommandLine; cpu = [double]($_.UserModeTime + $_.KernelModeTime) } } | ConvertTo-Json -Compress";
  const began = Date.now();
  const out = execFileSync("powershell", ["-NoProfile", "-Command", script], { encoding: "utf8", maxBuffer: 64 << 20, timeout: 60000 });
  // The readings were taken somewhere inside the query: its middle is the moment they belong to.
  return { at: (began + Date.now()) / 2, procs: JSON.parse(out) };
}

// The UI's processes in a snapshot: `rootPid` and everything below it, but not an engine.
function uiProcesses(snapshot, rootPid) {
  const children = new Map();
  for (const p of snapshot.procs) children.set(p.ppid, [...(children.get(p.ppid) ?? []), p]);
  const found = new Map();
  const root = snapshot.procs.find((p) => p.pid === rootPid);
  if (root) found.set(root.pid, root);
  const walk = (pid) => {
    for (const c of children.get(pid) ?? []) {
      if (/engine/i.test(c.name) || found.has(c.pid)) continue;
      found.set(c.pid, c);
      walk(c.pid);
    }
  };
  walk(rootPid);
  return found;
}

// A process's part in the UI, for the breakdown: the app itself, or the kind of WebView2 process (renderer, gpu-process, ...).
const uiKind = (p) => (p.name === "lorren.exe" ? "app" : (/--type=([\w-]+)/.exec(p.cmd ?? "")?.[1] ?? "webview2-browser"));

// The CPU the UI used between two snapshots. A process that was born in between counts from its start.
function uiCpu(from, to, rootPid) {
  const before = uiProcesses(from, rootPid);
  const seconds = (to.at - from.at) / 1000;
  const used = {};
  let total = 0;
  for (const [pid, p] of uiProcesses(to, rootPid)) {
    const s = (p.cpu - (before.get(pid)?.cpu ?? 0)) / 1e7;
    total += s;
    used[uiKind(p)] = (used[uiKind(p)] ?? 0) + s;
  }
  const pct = (s) => Math.round((s / seconds) * 1000) / 10;
  return { window_s: Math.round(seconds * 10) / 10, pct: pct(total), by_kind_pct: Object.fromEntries(Object.entries(used).map(([k, s]) => [k, pct(s)])) };
}

function portOpen(port) {
  return new Promise((resolve) => {
    const s = net.connect(port, "127.0.0.1");
    s.once("connect", () => { s.destroy(); resolve(true); });
    s.once("error", () => resolve(false));
  });
}

// What WebView2 was actually started with: the command line of Lorren's WebView2 browser process (the one that names
// its host, `--webview-exe-name=lorren.exe`, and has no --type=). A DevTools port that never opens then says whether
// the flag reached WebView2 at all.
function webviewCommandLine() {
  const script =
    "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'msedgewebview2.exe' -and $_.CommandLine -like '*--webview-exe-name=lorren.exe*' " +
    "-and $_.CommandLine -notlike '*--type=*' } | ForEach-Object { '{0} {1}' -f $_.ProcessId, $_.CommandLine }";
  try {
    const out = execFileSync("powershell", ["-NoProfile", "-Command", script], { encoding: "utf8", timeout: 20000 }).trim();
    return out || "no WebView2 browser process of lorren.exe is running";
  } catch (e) {
    return `its command line could not be read (${e.message})`;
  }
}

// R68: the DevTools flag is carried by the window's additionalBrowserArgs in src-tauri/tauri.e2e.conf.json, passed to
// the e2e's own debug build only (never to the test installers: an installed build must not open a CDP port). A JSON
// merge replaces the whole `windows` array, so that file restates the main window; this keeps it from drifting.
function checkE2eOverlay() {
  const read = (f) => JSON.parse(fs.readFileSync(path.join(TAURI_DIR, f), "utf8")).app.windows;
  const base = read("tauri.conf.json");
  const over = read("tauri.e2e.conf.json");
  const { additionalBrowserArgs, ...restated } = over[0] ?? {};
  if (base.length !== 1 || over.length !== 1 || !isDeepStrictEqual(restated, base[0])) {
    throw new Error("src-tauri/tauri.e2e.conf.json no longer restates the main window of tauri.conf.json: update it");
  }
  if (!String(additionalBrowserArgs).includes(`--remote-debugging-port=${CDP}`)) {
    throw new Error(`src-tauri/tauri.e2e.conf.json does not carry --remote-debugging-port=${CDP}`);
  }
}

// `guard`, when given, runs before every attempt and is NOT swallowed: it throws when waiting is pointless (R65).
async function until(what, fn, ms = 30000, guard = null) {
  const end = Date.now() + ms;
  for (;;) {
    if (guard) guard();
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
    // The e2e's own WebView2 profile (R5): a run never reads or leaves the user's.
    WEBVIEW2_USER_DATA_FOLDER: path.join(WORK, "webview"),
    // The DevTools port is NOT opened from here (R68). WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS did not reach Lorren's WebView2
    // browser: probed on runtime 155.0.4283.62 (alone, with logging flags, and with a fresh WEBVIEW2_USER_DATA_FOLDER, which
    // WAS honoured) its command line carried only the arguments wry sets itself through AdditionalBrowserArguments. The flag
    // is built into the e2e's own debug exe instead, by src-tauri/tauri.e2e.conf.json (see main()).
    FAKE_ENGINE_TOKEN_MS: "60",
    FAKE_ENGINE_REPLY: REPLY,
    FAKE_ENGINE_EVENTS: path.join(WORK, "engine-events.log"),
  };
}

// A ~900-word Markdown reply, the same every run: eight blocks of a paragraph, a bulleted list, a fenced code block,
// a line with inline code and a table. The fake engine streams one space-split word per token and the app cuts a
// reply at 1,024 tokens (REPLY_CAP), so the step checks the count before it uses this.
function longReply() {
  const vocab = ("the engine reads each prompt once and then writes one token after another while the window " +
    "redraws the transcript so the text that arrives stays smooth to read even when a table or a block of code " +
    "grows in the middle of the answer and the reader keeps scrolling to follow it").split(" ");
  const pick = (b, i) => vocab[(b * 7 + i) % vocab.length];
  const blocks = [];
  for (let b = 0; b < 8; b++) {
    const paragraph = Array.from({ length: 60 }, (_, i) => pick(b, i)).join(" ") + ".";
    const list = [0, 1, 2, 3, 4].map((j) => `- ${pick(b, 61 + j)} ${pick(b, 70 + j)}`).join("\n");
    const code = ["```js", `const n = ${b + 3};`, "let s = 0;", "for (let i = 0; i < n; i++) {", "  s += i * 2;", "}", "console.log(s);", "return s;", "```"].join("\n");
    const inline = `Call \`run(${b})\` and read \`total\` once.`;
    const table = ["| k | v |", "|---|---|", `| a | ${b} |`, `| b | ${b + 1} |`, `| c | ${b + 2} |`].join("\n");
    blocks.push([paragraph, list, code, inline, table].join("\n\n"));
  }
  return blocks.join("\n\n");
}

let lastLaunch = null;

async function launch(overrides = {}, { keepOpening = false } = {}) {
  let exited = null;
  const spawnedAt = Date.now();
  const child = spawn(APP, [], { env: { ...env(), ...overrides }, stdio: "ignore" });
  app = child;
  child.once("exit", (code) => { exited = { code }; });
  child.once("error", (e) => { exited = { code: `spawn failed: ${e.message}` }; });
  // tauri-plugin-single-instance keys on the app identifier alone: a launch while ANY Lorren process lives exits 0
  // before its setup, and the DevTools port never opens. Say so at once rather than time out 30 s later (R65).
  const guard = () => {
    if (exited) throw new Error(`the app exited (code ${exited.code}) before its window appeared -- is another Lorren running?`);
  };
  await until("the DevTools port", () => portOpen(CDP), 30000, guard).catch((e) => {
    if (exited) throw e; // the early-exit message says it all
    const cl = webviewCommandLine();
    const hint = cl.includes(`--remote-debugging-port=${CDP}`) ? "" : "; the flag is not on it: was this lorren.exe built with src-tauri/tauri.e2e.conf.json? (npm run e2e without --no-build builds it)";
    throw new Error(`${e.message} (the app is running; its WebView2 browser process: ${cl}${hint})`);
  });
  browser = await until("a CDP connection", () => chromium.connectOverCDP(`http://127.0.0.1:${CDP}`), 30000, guard);
  page = await until("the app page", async () => browser.contexts()[0]?.pages()[0], 30000, guard);
  await page.waitForFunction(() => !!document.getElementById("root"), null, { timeout: 30000 });
  const firstFrame = await page.evaluate(openingSnapshot);
  lastLaunch = { spawnedAt, firstFrame };
  fs.mkdirSync(path.join(WORK, "shots"), { recursive: true });
  await page.screenshot({ path: path.join(WORK, "shots", "opening-first.png") }).catch(() => {});
  if (!keepOpening) {
    await page.waitForFunction(() => window.__lorrenOpening?.phase === "done", null, { timeout: 20000 });
    await page.locator(".topbar .wordmark").waitFor({ state: "visible", timeout: 30000 });
  }
}

async function close(graceful = true) {
  if (!app) return;
  const pid = app.pid;
  await browser?.close().catch(() => {});
  try { execFileSync("taskkill", graceful ? ["/PID", String(pid)] : ["/F", "/PID", String(pid)], { stdio: "ignore" }); } catch {}
  await until("the app to exit", async () => !pids("lorren.exe").includes(pid), 15000).catch(() => {
    execFileSync("taskkill", ["/F", "/PID", String(pid)], { stdio: "ignore" });
  });
  // The WebView2 browser process outlives its host for a moment and keeps the DevTools port: a relaunch must not find it open.
  await until("the DevTools port to close", async () => !(await portOpen(CDP)), 15000).catch(() => {});
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
  await page.getByRole("textbox", { name: "message", exact: true }).fill(text);
  await page.getByRole("textbox", { name: "message", exact: true }).press("Enter");
  await page.getByRole("button", { name: "STOP" }).waitFor({ timeout: 15000 });
}

async function waitIdle(ms = 30000) {
  await page.getByRole("button", { name: "SEND" }).waitFor({ timeout: ms });
}

// "+ NEW" makes the chat in the host and then opens it, two round trips after the click, and until then the chat that was open
// before is still the one on screen. A step that picks a model or types at once reaches THAT chat: on 2026-11-04 the Nexus step
// picked nexus for the previous chat, three runs in three, and its first key then asked for the new chat's default model, which
// the engine already had. So this clicks "+ NEW" and waits until the list has one more chat and the open one is the new, empty one.
async function newChat() {
  const before = await page.locator(".chat-item").count();
  await page.getByRole("button", { name: "+ NEW", exact: true }).click();
  await until("the new chat to open", async () =>
    (await page.locator(".chat-item").count()) === before + 1 &&
    (await page.locator(".chat-item.selected").innerText().catch(() => "")).startsWith("New chat"));
}

async function step(name, fn) {
  await fn();
  results.push(name);
  console.log(`ok   ${name}`);
}

// <in-page>
// The checks of the layout step (R70). This runs IN THE PAGE (Playwright serialises it), so it uses nothing from this
// file. `tab` is the tab on show and `width` the window's width in CSS px. It returns the defects it found as sentences
// (one per kind of defect, naming the first element that has it and how many more do), the controls that failed (a
// reading that cannot be trusted, because what it is about is not on screen), and the numbers it read.
function layoutDefects({ tab, width }) {
  const found = new Map();
  const controls = [];
  const facts = {};
  const add = (kind, message) => {
    const known = found.get(kind);
    if (known) known.more += 1;
    else found.set(kind, { message, more: 0 });
  };
  const rect = (el) => el.getBoundingClientRect();
  const shown = (el) => el.getClientRects().length > 0; // the Chat tab stays in the DOM, hidden, while another tab shows
  const px = (n) => `${Math.round(n * 10) / 10} px`;
  const name = (el) => {
    const cls = typeof el.className === "string" && el.className.trim() ? "." + el.className.trim().split(/\s+/).join(".") : "";
    const text = (el.textContent ?? "").trim().replace(/\s+/g, " ");
    return `<${el.tagName.toLowerCase()}${cls}> "${text.length > 28 ? text.slice(0, 28) + "..." : text}"`;
  };

  // (a) The top bar: each item of the engine cluster on one line, and the cluster within its box.
  const engine = document.querySelector(".topbar .engine");
  const items = [...engine.children];
  facts.engine_items = items.length;
  const engineText = engine.textContent ?? "";
  if (items.length < 5 || !engineText.includes("Qwen3-4B-Instruct-2507-Q4_K_M") || !engineText.includes("app ")) {
    controls.push(`the engine cluster holds ${items.length} items and reads "${engineText.slice(0, 80)}": it has to show the long default model name and the memory`);
  }
  for (const item of items) {
    if (!shown(item)) continue; // an item the layout leaves out on purpose
    const fontSize = parseFloat(getComputedStyle(item).fontSize);
    if (rect(item).height > 1.6 * fontSize) add("topbar-lines", `top bar item ${name(item)} is ${px(rect(item).height)} tall: more than one line of ${px(fontSize)} text`);
    if (rect(item).right > rect(engine).right + 0.5) add("topbar-edge", `top bar item ${name(item)} ends at ${px(rect(item).right)}, past the engine cluster's edge at ${px(rect(engine).right)}`);
  }
  if (engine.scrollWidth > engine.clientWidth) add("topbar-overflow", `.topbar .engine overflows sideways: scrollWidth ${engine.scrollWidth} px > clientWidth ${engine.clientWidth} px`);

  // (b) A table fits the centre column: no row wider than itself, no cell past the column's right edge (the header is not
  // inside a scroll box, so it would be drawn over the right panel), no scroll box that scrolls sideways.
  const centre = [...document.querySelectorAll("section.center")].find(shown);
  if (tab === "Models" || tab === "Requests") {
    if (!centre) controls.push("no centre column is showing");
    else {
      const edge = rect(centre).right;
      facts.table_rows = centre.querySelectorAll(".table-row").length;
      if (facts.table_rows < 1) controls.push(`the ${tab} table has no row to measure`);
      for (const row of centre.querySelectorAll(".table-head, .table-row")) {
        if (row.scrollWidth > row.clientWidth) add("row-overflow", `${name(row)} is wider than its box: scrollWidth ${row.scrollWidth} px > clientWidth ${row.clientWidth} px`);
        for (const cell of row.children) {
          if (rect(cell).right > edge + 0.5) add("cell-edge", `in ${name(row)} the cell ${name(cell)} ends at ${px(rect(cell).right)}, past the centre column's right edge at ${px(edge)}`);
        }
      }
      for (const box of centre.querySelectorAll(".scroll")) {
        if (box.scrollWidth > box.clientWidth) add("scroll-sideways", `${name(box)} scrolls sideways: scrollWidth ${box.scrollWidth} px > clientWidth ${box.clientWidth} px`);
      }
    }
  }

  // (c) A DEFAULT / LOADED tag is whole: not past its cell's right edge, not wider than itself, not cut at the top or bottom.
  const tags = [...document.querySelectorAll(".pill.tag")].filter(shown);
  facts.tags = tags.length;
  if (tab === "Models" && tags.length < 2) controls.push(`the Models table shows ${tags.length} DEFAULT/LOADED tags; the default model is the loaded one, so there are two`);
  for (const tag of tags) {
    const t = rect(tag);
    const c = rect(tag.parentElement);
    if (t.right > c.right + 0.5 || tag.scrollWidth > tag.clientWidth + 1) {
      add("tag-cut-right", `the tag ${name(tag)} is cut: it ends at ${px(t.right)} where its cell ends at ${px(c.right)} (scrollWidth ${tag.scrollWidth} px, clientWidth ${tag.clientWidth} px)`);
    }
    if (t.top < c.top - 1 || t.bottom > c.bottom + 1) {
      add("tag-cut-vertical", `the tag ${name(tag)} is cut at the top or bottom: it spans ${px(t.top)} to ${px(t.bottom)} in a cell that spans ${px(c.top)} to ${px(c.bottom)}`);
    }
  }

  // (d) Logs: a line that wraps continues under its message, not under the time column.
  if (tab === "Logs") {
    const lines = [...document.querySelectorAll(".logbox > div")];
    facts.log_lines = lines.length;
    facts.log_lines_wrapped = 0;
    for (const line of lines) {
      const message = line.lastElementChild;
      const range = document.createRange();
      range.selectNodeContents(message);
      const rects = [...range.getClientRects()].filter((r) => r.width > 0);
      if (rects.length < 2) continue;
      facts.log_lines_wrapped += 1;
      const first = rects[0].left;
      const leftmost = Math.min(...rects.map((r) => r.left));
      if (leftmost < first - 1) add("log-hang", `the log line ${name(line)} wraps to ${px(leftmost)}, left of where its message starts (${px(first)}): under the time column`);
    }
    if (facts.log_lines < 1) controls.push("the Logs tab shows no line");
    if (width === 1100 && facts.log_lines_wrapped < 1) controls.push("no log line wraps at 1100 px, so the hanging indent had nothing to hold");
  }

  // (e) Chat at 1100: the composer's text box is at least 300 px wide beside its buttons.
  if (tab === "Chat") {
    const box = document.querySelector(".composer textarea");
    if (!box || !shown(box)) controls.push("the composer's text box is not showing");
    else {
      facts.composer_textarea_px = Math.round(rect(box).width * 10) / 10;
      if (width === 1100 && rect(box).width < 300) add("composer", `the composer's text box ${name(box)} is ${px(rect(box).width)} wide: under 300 px at 1100 px`);
    }
  }

  // (f) The right panel (Chat's reply panel, Models' details): each value whole, on one line, clear of its label.
  for (const fact of document.querySelectorAll(".right .fact")) {
    if (!shown(fact)) continue;
    const label = fact.querySelector(".k");
    const value = fact.querySelector(".v");
    if (!label || !value) continue;
    const fontSize = parseFloat(getComputedStyle(value).fontSize);
    if (value.scrollWidth > value.clientWidth + 1) add("panel-cut", `the panel value ${name(value)} is cut: scrollWidth ${value.scrollWidth} px > clientWidth ${value.clientWidth} px`);
    if (rect(value).height > 1.6 * fontSize) add("panel-lines", `the panel value ${name(value)} runs onto a second line`);
    const l = rect(label);
    const v = rect(value);
    if (v.top < l.bottom && l.top < v.bottom && v.left < l.right - 0.5) add("panel-overlap", `the panel label ${name(label)} overlaps its value ${name(value)}`);
  }

  const defects = [...found.values()].map(({ message, more }) => (more ? `${message} (and ${more} more like it)` : message));
  return { defects, controls, facts };
}
// What the opening shows right now (R5). Runs IN THE PAGE.
function openingSnapshot() {
  const el = document.getElementById("opening");
  const w = el?.querySelector(".op-word");
  const r = w?.getBoundingClientRect();
  const tab = [...document.querySelectorAll("[role=tab]")].find((t) => t.textContent.trim() === "Models");
  const tr = tab?.getBoundingClientRect();
  let hitsOpening = null;
  if (tr && tr.width > 0 && tr.height > 0) {
    const hit = document.elementFromPoint(tr.left + tr.width / 2, tr.top + tr.height / 2);
    hitsOpening = !!(hit && el && el.contains(hit));
  }
  return {
    present: !!el,
    phase: window.__lorrenOpening?.phase ?? null,
    word: r ? { left: r.left, top: r.top, width: r.width, height: r.height } : null,
    vw: innerWidth,
    vh: innerHeight,
    hitsOpening,
  };
}
// </in-page>

const centredOff = (s) => {
  if (!s.word) return Infinity;
  return Math.max(Math.abs(s.word.left + s.word.width / 2 - s.vw / 2), Math.abs(s.word.top + s.word.height / 2 - s.vh / 2));
};
const openingLines = async () =>
  (await page.evaluate(() => window.__TAURI_INTERNALS__.invoke("logs", { after: 0 }))).map((l) => l.text).filter((t) => t.startsWith("opening: "));
const waitDone = (ms = 20000) => page.waitForFunction(() => window.__lorrenOpening?.phase === "done", null, { timeout: ms });
const getReport = () => page.evaluate(() => window.__lorrenOpening.report);
const openingMeasure = (r) => ({
  first_frame_ms: r.first_frame_ms - r.navigation_ms,
  ready_ms: r.ready_ms === null ? null : r.ready_ms - r.first_frame_ms,
  top_bar_ms: r.handover_end_ms - r.first_frame_ms,
  longest_gap_ms: r.longest_gap_ms,
  longest_task_ms: r.longest_task_ms,
  landing_error_px: r.landing_error_px,
  plan_ms: r.plan_ms,
});

const steps = [
  ["the opening: on screen at the first frame, lit by 2.0 s, handed over onto the wordmark, logged", async () => {
    const f = lastLaunch.firstFrame;
    if (!f.present || f.phase === "done") throw new Error(`INSTRUMENT FAILURE: the harness connected after the opening ended (${Date.now() - lastLaunch.spawnedAt} ms after launch at the latest)`);
    if (centredOff(f) > 1) throw new Error(`the word is not centred at the first frame (off by ${centredOff(f).toFixed(2)} px): ${JSON.stringify(f)}`);
    const r = await getReport();
    if (!r) throw new Error("no report");
    if (r.ready_ms === null || r.valve || r.reduced_motion || r.error !== null) throw new Error(`report: ${JSON.stringify(r)}`);
    if (r.finished_ms - r.first_frame_ms < 1983) throw new Error(`lit for ${r.finished_ms - r.first_frame_ms} ms, under 1983`);
    if (r.handover_start_ms - Math.max(r.finished_ms, r.ready_ms) < 298) throw new Error(`hand-over began ${r.handover_start_ms - Math.max(r.finished_ms, r.ready_ms)} ms after the later of finished/ready, under 298`);
    if (!(r.landing_error_px <= 0.5)) throw new Error(`landing error ${r.landing_error_px} px`);
    const after = await page.evaluate(() => ({
      el: !!document.getElementById("opening"),
      cls: document.documentElement.classList.contains("opening"),
      vis: getComputedStyle(document.querySelector(".topbar .wordmark")).visibility,
    }));
    if (after.el || after.cls || after.vis !== "visible") throw new Error(`after the hand-over: ${JSON.stringify(after)}`);
    const lines = await until("the opening log line", async () => { const l = await openingLines(); return l.length ? l : false; }, 10000);
    console.log("     " + lines.join("\n     "));
    if (lines.length !== 1 || !lines[0].includes("landed within")) throw new Error(`log: ${JSON.stringify(lines)}`);
    measurements.opening = openingMeasure(r);
  }],
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
  // Demo D1 + D4. Nothing has been typed and no chat exists yet. The composer is enabled and the transcript says to type; the
  // picker is enabled and shows the model a new chat will get, which is the default and says so; a message typed here makes
  // the chat on that model and is sent in it.
  ["type first: with no chat open, a typed message makes the chat on the default model", async () => {
    await page.getByRole("tab", { name: "Chat" }).click();
    await page.getByText("Type a message to start a chat.").waitFor({ timeout: 5000 });
    if (!(await page.getByRole("textbox", { name: "message", exact: true }).isEnabled())) throw new Error("the composer is disabled with no chat open");
    const picker = page.locator("select[aria-label=model]");
    if (!(await picker.isEnabled())) throw new Error("the model picker is disabled with no chat open");
    const shown = await picker.evaluate((el) => el.selectedOptions[0]?.textContent ?? "");
    if (shown !== "Qwen3-4B-Instruct-2507-Q4_K_M (default)") throw new Error(`the picker shows "${shown}", not the default model marked (default)`);
    await sendMessage("Hello, first message.");
    await page.locator(".messages").getByText("Verdict: HOLD.").first().waitFor({ timeout: 15000 });
    await waitIdle();
    const chats = await page.locator(".chat-item").allInnerTexts();
    if (chats.length !== 1 || !chats[0].includes("Hello, first message.") || !chats[0].includes("Qwen3-4B-Instruct-2507-Q4_K_M")) {
      throw new Error(`the chat list after the first message: ${JSON.stringify(chats)}`);
    }
  }],
  // The one step that still makes its chat with "+ NEW". Demo D9: picking a model for a chat only stores it; the engine
  // restarts on the model at the FIRST KEY typed in the chat (and on Send), not on the pick.
  ["a new chat switched to Nexus restarts the engine on it when the user starts typing", async () => {
    await page.getByRole("tab", { name: "Chat" }).click();
    await newChat();
    await page.locator("select[aria-label=model]").selectOption({ label: "nexus" });
    // Picking is not loading: a while later the engine still serves the default model.
    await sleep(1500);
    const after_pick = await topbar();
    if (!after_pick.includes("serving") || !after_pick.includes("Qwen3-4B-Instruct-2507-Q4_K_M") || after_pick.includes("nexus")) {
      throw new Error(`picking a model moved the engine: the top bar says "${after_pick}"`);
    }
    if ((await page.locator("select[aria-label=model]").inputValue()) === "") throw new Error("the picker lost the model that was picked");
    const t0 = Date.now();
    await page.getByRole("textbox", { name: "message", exact: true }).pressSequentially("L"); // the first key
    await until("serving nexus", async () => { const t = await topbar(); return t.includes("serving") && t.includes("nexus"); });
    measurements.switch_ms = Date.now() - t0;
    const log = (await page.evaluate(() => window.__TAURI_INTERNALS__.invoke("logs", { after: 0 }))).map((l) => l.text);
    if (!log.some((t) => t.startsWith("preparing nexus: the user is typing"))) throw new Error("the log has no 'preparing nexus: the user is typing' line");
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
    await page.getByText("2/2", { exact: true }).waitFor();
    await page.getByLabel("previous version").last().click();
    await page.getByText("1/2", { exact: true }).waitFor();
  }],
  ["killing the engine brings it back, and the chat carries on", async () => {
    const before = pids("fake-engine.exe");
    if (before.length !== 1) throw new Error(`expected one engine, found ${before.length}`);
    execFileSync("taskkill", ["/F", "/PID", String(before[0])], { stdio: "ignore" });
    await until("a new engine serving", async () => {
      const now = pids("fake-engine.exe");
      return now.length === 1 && now[0] !== before[0] && (await topbar()).includes("serving");
    });
    // Demo D6: one restart reads "1 engine restart" (and "1 engine restarts" would not).
    await until("the restart counted", async () => /(^|\D)1 engine restart(?!s)/.test(await page.locator(".statusbar").innerText()));
    await sendMessage("Still there?");
    await waitIdle();
  }],
  ["Requests and Logs show the traffic", async () => {
    await page.getByRole("tab", { name: "Requests" }).click();
    await until("4 requests", async () => (await page.getByRole("row").count()) >= 4);
    // `:visible`, not every `.pill`: the Chat tab stays in the DOM while hidden (R36a) and its STOPPED pill still has text (R37).
    const statuses = await page.locator(".pill:visible").allInnerTexts();
    if (!statuses.some((s) => s.startsWith("OK")) || !statuses.some((s) => s.startsWith("STOPPED"))) throw new Error(`statuses: ${statuses}`);
    await page.getByRole("tab", { name: "Logs" }).click();
    await page.getByText(/engine serving/).first().waitFor();
  }],
  ["a tampered, a wrongly signed and an unsigned update are refused; the signed one is ready", async () => {
    // Each refusal is read from its own REFUSED log line, never from the badge alone: the badge
    // still shows the previous feed's identical text until the new check lands, and a check that
    // failed for another reason (no server, bad JSON) logs a warning, not a refusal.
    // Demo D7: the words each refusal gets in the badge (a few words) and in the log (one sentence and the code for support).
    const refusals = {
      tampered: { label: "installer altered or damaged", sentence: "does not match its signature", code: "[Minisign(InvalidSignature)]" },
      wrongkey: { label: "untrusted signing key", sentence: "a key this Lorren does not trust", code: "[Minisign(UnexpectedKeyId)]" },
      nosig: { label: "unsigned", sentence: "the update has no signature", code: "[EmptySignature]" },
    };
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
        const want = refusals[name];
        // The log line is written just before the state event reaches the badge, and the previous feed's badge reads
        // differently (its own label), so the badge is waited for and not read once.
        await until(`the ${name} badge`, async () => (await badge()).includes(`update 0.0.2 refused · ${want.label} · check again`), 10000).catch(async (e) => {
          throw new Error(`${e.message}; the bar says: ${await badge()}`);
        });
        const title = await page.locator(".statusbar button").last().getAttribute("title");
        if (!title?.includes(want.sentence) || !title.endsWith("; nothing was installed")) throw new Error(`the badge's tooltip after ${name}: ${title}`);
        const line = (await updaterLines()).filter((t) => t.includes("REFUSED")).at(-1) ?? "";
        if (!line.includes(want.sentence) || !line.endsWith(`; nothing was installed ${want.code}`)) throw new Error(`the log line after ${name}: ${line}`);
        if (line.includes("The signature verification failed") || line.includes("did not verify")) throw new Error(`the log line repeats the plugin's text: ${line}`);
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
  // R66 + R70. Each tab is shown at 1100 px (the window's minimum) and at 1440 px (its default), a screenshot is saved,
  // three numbers are read (the top bar's height, how far the document overflows sideways, and how far <main> does: the
  // values are for the controller to read), and `layoutDefects` (above) looks for the defects the 1100 px screenshots
  // showed. The step fails on those, naming the width, the tab and the element, and on a reading that could not be taken.
  ["the layout fits at 1100 and 1440 px", async () => {
    // The app was relaunched two steps back, so the Requests table is empty and the top bar has only what the default model
    // gives it. One message to the default model gives the table a row to measure and the bar the long model name.
    await page.getByRole("tab", { name: "Chat", exact: true }).click();
    await newChat();
    await until("the default model serving", async () => { const t = await topbar(); return t.includes("serving") && t.includes("Qwen3-4B-Instruct-2507-Q4_K_M"); });
    await sendMessage("Layout check.");
    await waitIdle();
    const shots = path.join(WORK, "shots");
    fs.mkdirSync(shots, { recursive: true });
    const layout = (measurements.layout = {});
    const defects = [];
    const cdp = await page.context().newCDPSession(page);
    try {
      for (const [width, height] of [[1100, 800], [1440, 900]]) {
        await cdp.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 0, mobile: false });
        // CONTROL: an override that silently did nothing must not turn into "no overflow" readings.
        let got = null;
        await until(`innerWidth to be ${width}`, async () => (got = await page.evaluate(() => window.innerWidth)) === width, 3000)
          .catch(() => { throw new Error(`the ${width} px override did not take: innerWidth is ${got}`); });
        layout[width] = {};
        for (const tab of ["Chat", "Models", "Requests", "Logs"]) {
          await page.getByRole("tab", { name: tab, exact: true }).click();
          await sleep(400);
          // CONTROL: the reading belongs to the tab it is filed under.
          if ((await page.getByRole("tab", { name: tab, exact: true }).getAttribute("aria-selected")) !== "true") throw new Error(`the ${tab} tab is not selected at ${width} px`);
          const file = path.join(shots, `${width}-${tab}.png`);
          await page.screenshot({ path: file });
          if (!(fs.statSync(file).size > 1000)) throw new Error(`the screenshot ${file} is empty`);
          layout[width][tab] = await page.evaluate(() => {
            const main = document.querySelector("main.content");
            return {
              topbar_h: document.querySelector(".topbar").getBoundingClientRect().height,
              doc_overflow_px: document.documentElement.scrollWidth - window.innerWidth,
              main_overflow_px: main.scrollWidth - main.clientWidth,
            };
          });
          const read = await page.evaluate(layoutDefects, { tab, width });
          // CONTROL: a reading of something that is not on screen would pass for any layout.
          if (read.controls.length) throw new Error(`${width} px, ${tab} tab: the reading is not valid: ${read.controls.join("; ")}`);
          Object.assign(layout[width][tab], read.facts);
          for (const d of read.defects) defects.push(`${width} px, ${tab} tab: ${d}`);
        }
      }
    } finally {
      await cdp.send("Emulation.clearDeviceMetricsOverride").catch(() => {});
      await cdp.detach().catch(() => {});
    }
    if (defects.length) throw new Error(`${defects.length} layout defect(s): ${defects.join(" | ")}`);
  }],
  // R66, measurement 2, and demo D5. A ~900-word Markdown reply streams at 55 ms a token (~18 tok/s, a real engine's pace; it
  // was 10 ms a token, 100 tok/s, before D5). A requestAnimationFrame recorder runs from just before SEND is pressed until the
  // CPU window begins, and again from its end until the reply is done; the gaps between frames, within a segment, are the
  // reading. A hidden window throttles rAF, so the reading is marked invalid then (and the step throws only if no frame came at
  // all). The UI's CPU is read over UI_CPU_WINDOW_S seconds in the middle of the stream, with the recorder off (it keeps the page
  // drawing at the display's rate, which would be most of the reading); spec §7 budget: under 5% of one core while streaming.
  // The step FAILS above UI_CPU_LIMIT_PCT OVER the same launch's idle (R82), a regression guard with headroom: the number to read against the budget is
  // measurements.stream.ui_cpu_pct, of THIS build (the app is a debug build with a DevTools port, so the figure is higher than
  // the release build's).
  ["a long streaming reply's frame gaps and UI CPU are measured", async () => {
    const reply = longReply();
    const words = reply.split(" ").length; // = the tokens the fake engine streams
    if (words >= 1000) throw new Error(`the generated reply is ${words} tokens; the app cuts a reply at 1,024`);
    await close(true);
    await until("no orphan engine", async () => pids("fake-engine.exe").length === 0, 15000);
    await launch({ FAKE_ENGINE_TOKEN_MS: String(STREAM_TOKEN_MS), FAKE_ENGINE_REPLY: reply });
    await page.getByRole("tab", { name: "Chat", exact: true }).click();
    await newChat();
    await until("serving", async () => (await topbar()).includes("serving"));
    // The frame recorder runs in two segments, before the CPU window and after it, and is OFF inside it: a requestAnimationFrame
    // loop keeps the page drawing at the display's rate (300 Hz on the box this was written on), which is CPU of its own.
    const visibility_start = await page.evaluate(() => {
      window.__segments = [];
      window.__record = (on) => {
        window.__recording = on;
        if (!on) return;
        const segment = [];
        window.__segments.push(segment);
        const tick = (t) => {
          if (!window.__recording || window.__segments.at(-1) !== segment) return;
          segment.push(t);
          requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      };
      window.__record(true);
      return document.visibilityState;
    });
    await sendMessage("Write the long reply.");
    const sent = Date.now();
    // The UI's CPU over a window in the middle of the stream. Nothing of this script touches the page while the window runs.
    await sleep(UI_CPU_SKIP_S * 1000);
    await page.evaluate(() => window.__record(false));
    const cpuFrom = processSnapshot();
    await sleep(UI_CPU_WINDOW_S * 1000);
    const cpuTo = processSnapshot();
    const streaming_at_window_end = await page.evaluate(() => [...document.querySelectorAll(".composer-actions button")].some((b) => b.textContent === "STOP"));
    await page.evaluate(() => window.__record(true));
    // Waiting for SEND is polled from here and not with waitIdle(): Playwright's waitFor re-runs its locator on every
    // animation frame, which would add work to the very frames being timed.
    await until("SEND to return", () => page.evaluate(() => [...document.querySelectorAll(".composer-actions button")].some((b) => b.textContent === "SEND")), 90000);
    const stream_ms = Date.now() - sent;
    const { segments, visibility_end } = await page.evaluate(() => {
      window.__record(false);
      return { segments: window.__segments, visibility_end: document.visibilityState };
    });
    // The same launch at rest, just after the reply: the recorder is off, nothing of this script touches the page, and the start-up
    // work is long over (read just after a relaunch, the idle took in that work: 7% once, against 4% settled). The streaming window is
    // judged over this (R82).
    const idleFrom = processSnapshot();
    await sleep(UI_CPU_IDLE_S * 1000);
    const idle = uiCpu(idleFrom, processSnapshot(), app.pid);
    const frames = segments.flat();
    if (frames.length === 0) throw new Error(`no animation frame was recorded (page ${visibility_start} at the start, ${visibility_end} at the end)`);
    const chars_shown = (await page.locator(".messages .msg").last().locator(".md").innerText()).length;
    // Gaps are taken inside a segment: the stretch with the recorder off is not a gap.
    const gaps = segments.flatMap((segment) => segment.slice(1).map((t, i) => t - segment[i]));
    const sorted = [...gaps].sort((a, b) => a - b);
    const r1 = (x) => Math.round(x * 10) / 10;
    const cpu = uiCpu(cpuFrom, cpuTo, app.pid);
    const stream = {
      words,
      chars_shown,
      duration_ms: r1(segments.reduce((sum, segment) => sum + (segment.length ? segment[segment.length - 1] - segment[0] : 0), 0)),
      stream_ms,
      tokens_per_s: r1((words * 1000) / stream_ms),
      ui_cpu_pct: cpu.pct,
      ui_cpu_window_s: cpu.window_s,
      ui_cpu_by_kind_pct: cpu.by_kind_pct,
      ui_cpu_idle_pct: idle.pct,
      ui_cpu_idle_window_s: idle.window_s,
      ui_cpu_over_idle_pct: r1(cpu.pct - idle.pct),
      ui_cpu_limit_pct: UI_CPU_LIMIT_PCT,
      frames: frames.length,
      gap_max_ms: r1(sorted.length ? sorted[sorted.length - 1] : 0),
      gap_p95_ms: r1(sorted.length ? sorted[Math.ceil(0.95 * sorted.length) - 1] : 0),
      gaps_over_50ms: gaps.filter((g) => g > 50).length,
      visibility_start,
      visibility_end,
      valid: visibility_start === "visible" && visibility_end === "visible",
    };
    if (!stream.valid) stream.reason = `the page was not visible (${visibility_start} at the start, ${visibility_end} at the end): a hidden window throttles requestAnimationFrame, so these gaps mean nothing`;
    measurements.stream = stream;
    // CONTROLS: a reading of a window that was not inside the stream, or of no process, is not a reading of the stream.
    if (!streaming_at_window_end) throw new Error(`the reply had ended before the ${UI_CPU_WINDOW_S} s window did: the UI's CPU (${cpu.pct}%) is not a reading of a streaming reply`);
    if (Object.keys(cpu.by_kind_pct).length < 3) throw new Error(`the UI's processes were not found (${JSON.stringify(cpu.by_kind_pct)}): lorren.exe, the WebView2 browser and a renderer are the least there is`);
    if (stream.ui_cpu_over_idle_pct > UI_CPU_LIMIT_PCT) {
      throw new Error(`the UI used ${cpu.pct}% of one core over ${cpu.window_s} s of streaming, ${stream.ui_cpu_over_idle_pct}% over the ${idle.pct}% it used at rest just before (${JSON.stringify(cpu.by_kind_pct)}): the limit of this step is ${UI_CPU_LIMIT_PCT}% over idle, the spec's budget 5%`);
    }
  }],
  ["the opening with the host held back 6 s: centred and taking the clicks until ready, then lands", async () => {
    await close();
    await launch({ LORREN_DEBUG_HOLD_MODELS_MS: "6000" }, { keepOpening: true });
    const cap = Date.now() + 20000;
    let playing = 0, first = null;
    for (;;) {
      const s = await page.evaluate(openingSnapshot);
      if (s.phase === "handover" || s.phase === "done") { first = { s, at: Date.now() }; break; }
      if (s.phase === "playing") {
        if (centredOff(s) > 1) throw new Error(`held: the word is off centre by ${centredOff(s).toFixed(2)} px while playing`);
        if (s.hitsOpening !== true) throw new Error(`held: the Models tab is not under the opening while playing (hitsOpening=${s.hitsOpening})`);
        playing++;
      }
      if (Date.now() > cap) throw new Error("held: the opening never left 'playing' in 20 s");
      await sleep(200);
    }
    if (playing < 10) throw new Error(`held: only ${playing} playing snapshots (need 10): the sampling did not run`);
    const since = first.at - lastLaunch.spawnedAt;
    if (since < 6000) throw new Error(`held: the first snapshot no longer playing came ${since} ms after spawn, under 6000`);
    await waitDone();
    const r = await getReport();
    if (r.ready_ms - lastLaunch.spawnedAt < 6000) throw new Error(`held: ready ${r.ready_ms - lastLaunch.spawnedAt} ms after spawn, under 6000`);
    if (r.valve) throw new Error("held: the valve ended it");
    if (r.handover_start_ms < r.ready_ms) throw new Error("held: the hand-over began before ready");
    if (r.finished_ms < r.ready_ms - 20) throw new Error(`held: it finished ${r.ready_ms - r.finished_ms} ms before ready`);
    if (!(r.landing_error_px <= 0.5)) throw new Error(`held: landing error ${r.landing_error_px} px`);
    const lines = await openingLines();
    console.log(`     playing snapshots: ${playing}; first not playing at ${since} ms \n     ` + lines.join("\n     "));
    if (lines.length < 1) throw new Error("held: no opening log line");
    measurements.opening_held = openingMeasure(r);
  }],
  ["reduced motion: the lit word fades in, and out after ready", async () => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.reload();
    await waitDone();
    const r = await getReport();
    if (r.reduced_motion !== true || r.landing_error_px !== null || r.ready_ms === null || r.handover_start_ms < r.ready_ms) throw new Error(`reduced-motion report: ${JSON.stringify(r)}`);
    await page.locator(".topbar .wordmark").waitFor({ state: "visible", timeout: 10000 });
    const lines = await until("the reduced-motion log line", async () => { const l = await openingLines(); return l.some((x) => x.includes("reduced motion: faded")) ? l : false; }, 10000);
    console.log("     " + lines[lines.length - 1]);
    if (!lines[lines.length - 1].includes("reduced motion: faded")) throw new Error(`newest opening line: ${lines[lines.length - 1]}`);
  }],
];

async function main() {
  if (!process.argv.includes("--no-build")) {
    checkE2eOverlay();
    sh(process.execPath, ["node_modules/@tauri-apps/cli/tauri.js", "build", "--debug", "--no-bundle", "--config", "src-tauri/tauri.updater-test.conf.json", "--config", "src-tauri/tauri.e2e.conf.json"]);
    sh("cargo", ["build", "--features", "fake-engine", "--bin", "fake-engine"], { cwd: TAURI_DIR });
  }
  if (pids("lorren.exe").length || pids("fake-engine.exe").length) throw new Error("close every running Lorren and fake engine first");
  if (await portOpen(CDP)) throw new Error(`close whatever holds port ${CDP} (the e2e drives the app through it)`);
  fs.rmSync(WORK + "/data", { recursive: true, force: true });
  fs.rmSync(WORK + "/models", { recursive: true, force: true });
  fs.rmSync(WORK + "/engine-events.log", { force: true });
  // A failed run must not leave the last good run's numbers and pictures lying next to its verdict.
  fs.rmSync(WORK + "/shots", { recursive: true, force: true });
  fs.rmSync(WORK + "/webview", { recursive: true, force: true });
  fs.rmSync(WORK + "/measurements.json", { force: true });
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
