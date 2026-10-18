// Milestone 1's checklist (spec §11.1), end to end against the FAKE engine: the real app window,
// driven through WebView2's DevTools port. Prints one line per step; the verdict is the last line.
//   npm run e2e            build, then run
//   npm run e2e -- --no-build
// The last two steps are MEASUREMENTS (R66): they fail only when a reading could not be taken, never on its value.
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

async function launch(overrides = {}) {
  let exited = null;
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
    await page.getByRole("button", { name: "+ NEW", exact: true }).click();
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
    await until("the restart counted", async () => (await page.locator(".statusbar").innerText()).includes("1 engine restarts"));
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
  // R66, measurement 1. Each tab is shown at 1100 px (the window's minimum) and at 1440 px (its default), a screenshot
  // is saved, and three numbers are read: the top bar's height, how far the document overflows sideways, and how far
  // <main> does. The values are for the controller to read; the step fails only if one could not be taken.
  ["the layout at 1100 and 1440 px is captured", async () => {
    const shots = path.join(WORK, "shots");
    fs.mkdirSync(shots, { recursive: true });
    const layout = (measurements.layout = {});
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
        }
      }
    } finally {
      await cdp.send("Emulation.clearDeviceMetricsOverride").catch(() => {});
      await cdp.detach().catch(() => {});
    }
  }],
  // R66, measurement 2. A ~900-word Markdown reply streams at 10 ms a token; a requestAnimationFrame recorder runs
  // from just before SEND is pressed until it returns, and the gaps between frames are the reading. A hidden window
  // throttles rAF, so the reading is marked invalid then (and the step throws only if no frame came at all).
  ["a long streaming reply's frame gaps are measured", async () => {
    const reply = longReply();
    const words = reply.split(" ").length; // = the tokens the fake engine streams
    if (words >= 1000) throw new Error(`the generated reply is ${words} tokens; the app cuts a reply at 1,024`);
    await close(true);
    await until("no orphan engine", async () => pids("fake-engine.exe").length === 0, 15000);
    await launch({ FAKE_ENGINE_TOKEN_MS: "10", FAKE_ENGINE_REPLY: reply });
    await page.getByRole("tab", { name: "Chat", exact: true }).click();
    await page.getByRole("button", { name: "+ NEW", exact: true }).click();
    await until("serving", async () => (await topbar()).includes("serving"));
    const visibility_start = await page.evaluate(() => {
      window.__frames = [];
      window.__recording = true;
      const tick = (t) => {
        if (!window.__recording) return;
        window.__frames.push(t);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
      return document.visibilityState;
    });
    await sendMessage("Write the long reply.");
    // Waiting for SEND is polled from here and not with waitIdle(): Playwright's waitFor re-runs its locator on every
    // animation frame, which would add work to the very frames being timed.
    await until("SEND to return", () => page.evaluate(() => [...document.querySelectorAll(".composer-actions button")].some((b) => b.textContent === "SEND")), 90000);
    const { frames, visibility_end } = await page.evaluate(() => {
      window.__recording = false;
      return { frames: window.__frames, visibility_end: document.visibilityState };
    });
    if (frames.length === 0) throw new Error(`no animation frame was recorded (page ${visibility_start} at the start, ${visibility_end} at the end)`);
    const chars_shown = (await page.locator(".messages .msg").last().locator(".md").innerText()).length;
    const gaps = frames.slice(1).map((t, i) => t - frames[i]);
    const sorted = [...gaps].sort((a, b) => a - b);
    const r1 = (x) => Math.round(x * 10) / 10;
    const stream = {
      words,
      chars_shown,
      duration_ms: r1(frames[frames.length - 1] - frames[0]),
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
