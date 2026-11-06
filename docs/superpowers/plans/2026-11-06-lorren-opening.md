# Lorren's opening ("drawn, then lit") — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every launch of Lorren opens on the theme's background with the word "lorren" drawn, then lit, in the centre of the window, paced to how long the app takes to load, and handed over into the top bar's wordmark only once everything visible behind it is in place.

**Architecture:** The opening's markup sits in `index.html` and its look in `src/opening/opening.css`, which `index.html` links and Vite bundles into the main stylesheet. That stylesheet is render-blocking, so the opening is on screen at the webview's first paint. A small classic script, `public/opening.js`, is loaded before the app bundle. It holds the pacing as pure functions (unit-tested) and a runner that builds the animations, drives them every frame, waits for the app's `lorren:ready` event, does the hand-over and reports one log line to the host. The app fires `lorren:ready` from `src/opening/ready.ts` once App and ChatTab have reported their parts. The host gets the window's background colour, an `opening_report` command that writes the line, and (debug builds only) a hold on `models` so the e2e can play a slow host.

**Tech Stack:** Tauri 2.12 (Rust, serde), React 19 + TypeScript, Vite 8, vitest 5 (jsdom), plain ES2022 JavaScript for the classic script, the M1 CDP e2e (`desktop/e2e/run-demo-e2e.mjs`, playwright-core, fake engine).

**Spec:** `docs/superpowers/specs/2026-10-23-lorren-opening-design.md` (APPROVED `d0b7dcf`). Reference mockup (values, not structure): `.superpowers/brainstorm/1092-1792727290/content/opening-v10.html` (git-ignored; lines 139-264 are the timeline and the pacing). Parent: `2026-09-25-lorren-desktop-design.md`.

All paths below are relative to `stoke-lorren/desktop/` unless they start with `docs/` or `.superpowers/`.

## Global Constraints

- The sequence (spec §2): drawn 0 to 0.95 s (all six letters, a 1.2 px light line); from 0.55 s the letters settle in, embossed (`oklch(0.205 0.007 255)`, a dark top edge, a faint light bottom edge); lit 0.6 to 2.0 s (the light crosses the letters at an even pace, 0.79 to 1.8 s; the last letter is fully lit at 2.0 s; the outline fades as the light passes).
- The word: lowercase "lorren", Archivo 600, **112 CSS px at every window size**, letter-spacing -0.01em, centred in the window's content area. Background `--bg` = `oklch(0.15 0.006 255)` = `#090b0e` (computed: rgb 9.47 11.30 13.69). Lit text = `--text` `oklch(0.95 0.004 255)`.
- The hand-over (§2.4): the glide lasts 0.76 s, ease in-out (`cubic-bezier(.65,0,.35,1)`), and the word shrinks onto the top bar's wordmark exactly. The app fades in over 0.52 s, starting 0.26 s into the glide. The real wordmark takes over at the end, with no visible swap.
- Ready (§3) means: the first overview has arrived, OR it failed and the top bar shows "host unreachable"; the chat list is loaded; the open chat's messages are rendered (or the empty state is); the models list is loaded; the fonts are loaded; and one frame has been painted after those renders. It does **not** include the model warming up.
- The word never leaves the centre before ready. The valve: if ready has not come 12 s after the window appeared, the hand-over happens anyway.
- Pacing (§4): 2 s by default and never shorter; plans for the last start's time, clamped to 2-5 s (a first start plans 2 s); corrects itself every frame and eases every change. While loading it slows from 0.6 s before the planned end and never finishes; the lit word breathes a warm glow while it waits. Ready before 2 s lands exactly on 2.0 s. Ready after 2 s finishes briskly (up to 1.6x), and the hand-over starts 0.3 s after the finish.
- Mockup measurements the pacing must reproduce (§4): ready at 1 s → finished 2.0 s. First start, ready at 4 s → finished 4.0 s. Learned 4 s → 0.5x throughout, finished 4.1 s. Learned 4 s, ready at 1 s → up to 1.6x, finished 2.1 s. Ready at 7 s → finished 7.0 s.
- When it plays (§5): every launch, including the restart after an update; not when an already running Lorren is opened again. Reduced motion (`prefers-reduced-motion: reduce`): no drawing, no light, no glide. The lit word fades in, then fades out into the app once ready. Never before ready, and the 12 s valve still applies.
- §6: the CSP forbids inline scripts — keep it so. No font flash: Archivo 600 is preloaded with `font-display: block`. The window's `backgroundColor`, `html` and the webview all use the theme background. No frame of the opening may be more than 50 ms late on Kabir's PC; measure it, and split the app's first render if it breaks the budget, never shorten the opening. It must never trap the user. It lands exactly at any window size. It is `aria-hidden` and takes no keyboard focus. No new dependencies.
- §7: one log line per start (source `app`), giving: when the window appeared, when the opening first painted, when Lorren was ready, when the hand-over ended, the plan, and the longest frame gap.
- Ghost's box: no compile, `cargo test`, build or e2e while he games. Every heavy step runs behind the gate: no `WardogsClient-Win64-Shipping.exe` / `sp26-cod.exe` / `codCrashHandler.exe` process, and at least 4 GB free. Never start a real model server: the measurements use the fake engine. Nothing is pushed. No Claude attribution in commits. Build from Git Bash, not PowerShell.

## Rulings made while planning (each with its reason)

- **R1 — the opening's CSS is a linked stylesheet, not an inline `<style>`.** Tauri gives every `<style>` element a nonce (`tauri-utils-2.10.2/src/html.rs:156`, `inject_nonce(document, "style", …)`) and adds it to `style-src` (`tauri-2.12.3/src/manager/mod.rs:96`). Once `style-src` carries a nonce, Chromium ignores `'unsafe-inline'`, which would block every style the app sets through an attribute. A `<link rel="stylesheet">` in `index.html` is bundled by Vite into the main CSS file, which is render-blocking in `<head>`. Probed 2026-11-06 with Vite 8: the font URL is rewritten to `/assets/Archivo-600-<hash>.ttf` in both the preload and the CSS. So the first paint is the same as with inline CSS.
- **R2 — the keyframes live in `opening.js` (the Web Animations API, effect-level easing), as in the mockup.** CSS has only per-keyframe easing, which would change the three-keyframe light. The CSS holds the static look, and the word stays `visibility: hidden` until the script's first frame. A CSS-only valve fades the overlay at 14 s and reveals the wordmark, for the case where the script never runs.
- **R3 — the overlay takes clicks until the hand-over starts** (an invisible app must not be clickable), then lets them through. Keys still reach the app; the opening never takes focus.
- **R4 — the remembered time** is measured from the opening's first frame to ready, and kept in localStorage key `lorren.opening.lastStartMs` (spec §9). A start ended by the valve is not remembered.
- **R5 — the e2e gets its own WebView2 profile** (`WEBVIEW2_USER_DATA_FOLDER` = `e2e/.work/webview`, wiped every run). Each run then starts as a first start, and its starts never teach Kabir's app. The env var is honoured (probed in R68's notes, `e2e/run-demo-e2e.mjs:166`).
- **R6 — the e2e's slow host is a debug-build-only hold on `models`** (`LORREN_DEBUG_HOLD_MODELS_MS`). `models` becomes async, so the hold never blocks the window's thread (Tauri runs sync commands on the main thread).
- **R7 — the script's valve counts from the page's navigation start** (`performance.now()`), the nearest thing the page can see to "the window appeared".
- **R8 — the script waits for the font at most 1 s**, then starts with whatever face it has. The log line records the font time.
- **R9 — the setup blocks the page's load:** wry raises the custom-protocol requests on the main thread, and the setup's library scan holds that thread. The beta's log shows 228-501 ms from "starting" to "library:". Task 4's log line measures window → page → first frame, and Task 7 moves the scan off the main thread **only if** that gap is over 150 ms.

## Review Focus

The five input classes no task's main tests exercise and that are most likely to bite a person, each pinned by a test in its owning task:

1. **A click while the opening covers the app** must not reach the hidden app. → Task 5, the held-host step: a hit-test at the Models tab lands on `#opening` for every sample until the hand-over.
2. **A stalled frame** (the app's bundle or first render holding the thread, a minimized window) must not make the word jump. → Task 1: one 500 ms frame advances the timeline by at most 64 ms × the rate.
3. **A corrupt or extreme remembered value** (`"abc"`, `""`, `"-1"`, `"1e9"`, missing storage) must plan 2 s or clamp to 2-5 s. → Task 1: `parseRemembered` + `planFor` cases.
4. **A 30 Hz or 144 Hz display** must land on the same times as 60 Hz. → Task 1: "ready at 1 s" simulated at 30, 60 and 144 Hz.
5. **Reduced motion with a slow host**: no motion, the lit word stays until ready, and the valve still applies. → Task 1 (`reducedHandoverAt`, `valveDue`) and Task 5 (the reduced-motion reload step).

---

### Task 1: The pacing, as pure functions

**Files:**
- Create: `public/opening.js` (the pure part only; Task 2 adds the runner to the same file)
- Test: `src/__tests__/opening.test.ts`

**Interfaces:**
- Produces: `public/opening.js` is a classic script of the form `(function (root) { "use strict"; … root.LorrenOpening = {…}; if (root.document && root.document.getElementById("opening")) run(root); })(this);`. Task 1 leaves `run` as an empty function that Task 2 fills in. `LorrenOpening` exposes:
  - constants `T = 2000`, `PLAN_MIN = 2000`, `PLAN_MAX = 5000`, `R_MIN = 0.2`, `R_MAX = 1.6`, `VALVE_MS = 12000`, `HANDOVER_DELAY_MS = 300`, `REDUCED_MIN_MS = 900`, `STORAGE_KEY = "lorren.opening.lastStartMs"`
  - `parseRemembered(raw: string | null): number | null` — a finite number > 0, else null
  - `planFor(remembered: number | null): number` — null or not finite → `T`, else clamp to [`PLAN_MIN`, `PLAN_MAX`]
  - `begin(plan: number): PaceState` → `{ it: 0, rate: T / plan, glow: 0 }`
  - `pace(s: PaceState, f: { real: number, dt: number, readyAt: number | null, plan: number }): PaceState` (a new object; never mutates `s`)
  - `handoverAt(finishedAt: number | null, readyAt: number | null): number | null` → both present ? `max(finishedAt, readyAt) + 300` : null
  - `valveDue(sinceNavigation: number, readyAt: number | null): boolean` → `readyAt === null && sinceNavigation >= 12000`
  - `reducedHandoverAt(readyAt: number): number` → `max(900, readyAt)`
  - `glowFilter(glow: number, real: number): string` → `""` when `glow <= 0`, else the drop-shadow below
  - `glide(from: { boxLeft, boxTop, textLeft, textWidth, baseline }, to: { textLeft, textWidth, baseline }): { tx, ty, s }` (CSS px; the transform origin is the word box's top-left)
- All times are in ms on the opening's clock: `real` counts from its first frame, and `it` is the position on the default 2 s timeline.

- [ ] **Step 1: Write the failing tests** in `src/__tests__/opening.test.ts`. They load the script as text and run it against a plain object, so jsdom's `document` is never touched:

```ts
import source from "../../public/opening.js?raw";
import { describe, expect, it } from "vitest";

type State = { it: number; rate: number; glow: number };
const root: Record<string, any> = {};
new Function(source).call(root);
const O = root.LorrenOpening;

/** Frames at `hz` until the hand-over time (or `maxMs`); `readyAt` null = never ready. */
function simulate({ remembered = null, readyAt = null, hz = 60, maxMs = 15000 }: { remembered?: number | null; readyAt?: number | null; hz?: number; maxMs?: number }) {
  const plan = O.planFor(remembered);
  const dt = 1000 / hz;
  let s: State = O.begin(plan);
  let finishedAt: number | null = null;
  const frames: { real: number; it: number; rate: number; glow: number }[] = [];
  for (let real = dt; real <= maxMs; real += dt) {
    const ready = readyAt !== null && real >= readyAt ? readyAt : null;
    s = O.pace(s, { real, dt, readyAt: ready, plan });
    frames.push({ real, ...s });
    if (finishedAt === null && s.it >= O.T) finishedAt = real;
    const h = O.handoverAt(finishedAt, ready);
    if (h !== null && real >= h) return { plan, frames, finishedAt, handoverAt: h };
  }
  return { plan, frames, finishedAt, handoverAt: null };
}
```

Test cases (names and assertions):
  - `planFor`: `null→2000`, `1200→2000`, `4000→4000`, `9000→5000`, `NaN→2000`, `Infinity→2000`.
  - `parseRemembered`: `null→null`, `""→null`, `"abc"→null`, `"-1"→null`, `"0"→null`, `"3200"→3200`, `"1e9"→1e9` (then `planFor` gives 5000).
  - "ready at 1 s lands on 2.0 s": `simulate({ readyAt: 1000 })` → `finishedAt` within `[1983, 2017]`, `handoverAt === Math.max(finishedAt, 1000) + 300`.
  - "a first start ready at 4 s finishes at 4.0 s": `finishedAt` within `[3900, 4100]`.
  - "a learned 4 s start plays at 0.5x and finishes at 4.1 s": `simulate({ remembered: 4000, readyAt: 4000 })` → every frame with `real < 3400` has `|rate - 0.5| < 0.02`; `finishedAt` within `[4000, 4200]`.
  - "ready at 1 s after a learned 4 s plan speeds up to 1.6x and finishes at 2.1 s": `remembered: 4000, readyAt: 1000` → `max(rate) >= 1.5`, `max(rate) <= 1.6`, `finishedAt` within `[2000, 2150]`.
  - "ready at 7 s finishes at 7.0 s and never before ready": `readyAt: 7000` → `finishedAt` within `[6900, 7100]`, and every frame with `real < 7000` has `it <= T - 0.5`.
  - "it never finishes while loading and keeps approaching": `readyAt: null, maxMs: 12000` → every frame `it <= T - 0.5`; `it` never decreases; `it` at 3000 < `it` at 6000; `handoverAt === null`.
  - "never shorter than 2 s": `readyAt: 0` → `finishedAt >= 1983`.
  - "every speed change is eased": in each simulation above, for consecutive frames, `|rate[n] - rate[n-1]| <= 0.25`.
  - "one stalled frame does not jump": `pace(begin(2000), { real: 500, dt: 500, readyAt: null, plan: 2000 }).it <= 64 * 1.6`.
  - "30, 60 and 144 Hz land alike": `readyAt: 1000` at `hz` 30, 60, 144 → each `finishedAt` within `[2000 - 1000/hz, 2000 + 1000/hz]`.
  - "the glow breathes only while waiting with the light across": in `readyAt: null` frames, `glow > 0` only where `T - it < 120`, and `glow` reaches 1 within 520 ms of first being > 0. In `readyAt: 7000`, `glow` is 0 within 320 ms after 7000. `glowFilter(0, 1234) === ""`. `glowFilter(1, 0)` (sin 0 → b = 0.5) `=== "drop-shadow(0 0 11px oklch(0.85 0.07 75 / 0.21))"`.
  - `valveDue(11999, null) === false`, `valveDue(12000, null) === true`, `valveDue(13000, 5000) === false`.
  - `reducedHandoverAt(400) === 900`, `reducedHandoverAt(3000) === 3000`.
  - "the glide lands the text exactly": with `from = { boxLeft: 500, boxTop: 300, textLeft: 500, textWidth: 290.4, baseline: 398.6 }` and `to = { textLeft: 22, textWidth: 38.9, baseline: 32.2 }`, `{tx, ty, s} = glide(from, to)`. Then `s === 38.9 / 290.4`, and `500 + tx + s * (500 - 500)` ≈ 22, `300 + ty + s * (398.6 - 300)` ≈ 32.2 (each within 1e-9). Repeat with `textLeft: 503.5` (a text that starts inside its box).

- [ ] **Step 2: Run them to verify they fail**

Run: `npx vitest run src/__tests__/opening.test.ts`
Expected: FAIL. Vite cannot resolve `../../public/opening.js?raw` (the file does not exist yet).

- [ ] **Step 3: Implement the pure functions in `public/opening.js`**

`pace` is the mockup's frame step (`opening-v10.html:227-253`), ported as is:

```
dt = min(64, f.dt)
if f.readyAt === null:
  onPlan = f.real < f.plan - 600
  target = onPlan ? T / f.plan : min(R_MIN, (T - s.it) / 900);   ease = onPlan ? 120 : 400
else:
  target = f.real < T ? (T - s.it) / max(1, T - f.real) : R_MAX;  ease = 200
target = clamp(target, 0, R_MAX)
rate = s.rate + (target - s.rate) * (1 - exp(-dt / ease))
it = min(T, s.it + dt * rate);  if f.readyAt === null: it = min(it, T - 0.5)
waiting = f.readyAt === null && T - it < 120
glow = waiting ? min(1, s.glow + dt / 500) : max(0, s.glow - dt / 300)
```

`glowFilter`: `b = glow * (0.5 + 0.5 * Math.sin(real / 520))`, then `` `drop-shadow(0 0 ${round2(6 + 10 * b)}px oklch(0.85 0.07 75 / ${round2(0.1 + 0.22 * b)}))` `` (`round2` = two decimals, trailing zeros dropped).
`glide`: `s = to.textWidth / from.textWidth`, `tx = to.textLeft - from.boxLeft - s * (from.textLeft - from.boxLeft)`, `ty = to.baseline - from.boxTop - s * (from.baseline - from.boxTop)`.
The file starts with a comment block: what it is, why it is a classic script (the CSP, loaded before the bundle), the spec path, and that `LorrenOpening` exists for the tests.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npx vitest run src/__tests__/opening.test.ts`, then `npx vitest run` and `npx tsc --noEmit`.
Expected: all PASS. The whole suite is unchanged apart from the new file. `tsc` is clean: `*?raw` is typed by `vite/client`.

- [ ] **Step 5: Commit**

```bash
git add public/opening.js src/__tests__/opening.test.ts
git commit -m "desktop: the opening's pacing as pure functions, ported from the approved mockup (spec 2026-10-23 section 4)"
```

---

### Task 2: The opening on screen

**Files:**
- Modify: `index.html` (the `opening` class, the font preload, the stylesheet link, the markup, the classic script)
- Create: `src/opening/opening.css`
- Modify: `public/opening.js` (the runner, `run(root)`)
- Modify: `src-tauri/tauri.conf.json` and `src-tauri/tauri.e2e.conf.json` (`"backgroundColor": "#090b0e"` on the main window; the e2e file restates the window, see `checkE2eOverlay`)

**Interfaces:**
- Consumes: Task 1's `LorrenOpening` (everything in its Produces).
- Produces, for Tasks 3-5:
  - the page listens for `window` event **`lorren:ready`** (Task 3 fires it);
  - `window.__lorrenOpening = { phase: "font" | "playing" | "handover" | "done", report: Report | null }`, kept current;
  - at the end the runner calls `window.__TAURI_INTERNALS__.invoke("opening_report", { report })` when `__TAURI_INTERNALS__` exists. A rejection is caught and `console.warn`ed. Task 4 adds the command.
  - `Report` (snake_case; epoch ms = `Math.round(performance.timeOrigin + t)`; nullable fields are `null`): `navigation_ms`, `first_frame_ms|null`, `ready_ms|null`, `finished_ms|null`, `handover_start_ms`, `handover_end_ms`, `plan_ms`, `remembered_ms|null`, `longest_gap_ms` (the longest frame-to-frame gap from the first frame to the hand-over's end, raw, not clamped), `longest_task_ms|null`, `longest_task_at_ms|null` (relative to the first frame; from a buffered `PerformanceObserver` of `longtask`), `landing_error_px|null` (null when there was no glide), `font_ms|null` (navigation → font loaded; null if the 1 s cap won), `reduced_motion`, `valve`, `error|null`.
  - the markup's class names: `#opening > .op-backdrop + .op-word > .op-halo, .op-pressed, .op-lit, .op-spot, svg.op-trace > text.op-line` (in that order, which is the stacking order). `.op-pressed`, `.op-lit` and `.op-spot` each contain the text `lorren`, and so does `.op-line`.

- [ ] **Step 1: `index.html`**. `<html lang="en" class="opening">`. In `<head>`: `<link rel="preload" href="/src/fonts/Archivo-600.ttf" as="font" type="font/ttf" crossorigin />` and `<link rel="stylesheet" href="/src/opening/opening.css" />`. In `<body>`, in this order: `#root`, `<div id="opening" aria-hidden="true">` with the markup above, `<script src="/opening.js"></script>`, then the existing module script. No inline `<script>` and no `<style>` (R1). No `style` attributes in the markup.

- [ ] **Step 2: `src/opening/opening.css`** — the static look, from the mockup (`opening-v10.html:59-71` and `build()`), at t = 0:
  - `@font-face { font-family: "Lorren Opening"; src: url("../fonts/Archivo-600.ttf") format("truetype"); font-weight: 600; font-display: block; }` and `html { background: oklch(0.15 0.006 255); }`
  - `#opening`: fixed, `inset: 0`, `z-index: 2147483647`, `display: grid; place-items: center`, plus the CSS valve `animation: op-valve 520ms ease-out 14s forwards` with `@keyframes op-valve { to { opacity: 0; visibility: hidden; } }`
  - `html.opening .topbar .wordmark { visibility: hidden; animation: op-valve-mark 0s linear 14.52s forwards; }` with `@keyframes op-valve-mark { to { visibility: visible; } }`
  - `.op-backdrop`: absolute, `inset: 0`, `background: oklch(0.15 0.006 255)`
  - `.op-word`: relative; `font: 600 112px/1 "Lorren Opening", sans-serif`; `letter-spacing: -0.01em`; `white-space: nowrap`; `color: oklch(0.95 0.004 255)`; `transform-origin: 0 0`; `visibility: hidden`
  - `.op-lit`: `display: block`, the mask `linear-gradient(90deg, #000 44%, transparent 56%)` at size `300% 100%`, `no-repeat`, position `84% 0` (unprefixed and `-webkit-` forms)
  - `.op-pressed`: absolute `inset: 0`; `color: oklch(0.205 0.007 255)`; `text-shadow: 0 -1px 0 oklch(0.09 0.005 255), 0 1px 0 oklch(0.34 0.008 255 / 0.7)`; `opacity: 0`
  - `.op-spot`: absolute `inset: 0`; `color: transparent`; `background: radial-gradient(ellipse 7% 75% at 50% 50%, #fffaf0 0%, #f2d9a6 40%, transparent 100%) 84% 0 / 300% 100% no-repeat`; `background-clip: text` (both forms)
  - `.op-halo`: absolute; `left: -280px; top: 50%; width: 560px; height: 320px; margin-top: -160px`; `background: radial-gradient(closest-side, oklch(0.76 0.09 70 / 0.14), transparent)`; `opacity: 0`
  - `.op-trace`: absolute, `left: 0; top: 0`, `overflow: visible`
  - `.op-line`: `font: 600 112px "Lorren Opening"`; `letter-spacing: -0.01em`; `fill: none`; `stroke: #eef0f4`; `stroke-width: 1.2`; `stroke-dasharray: 360`; `stroke-dashoffset: 360`; `stroke-opacity: 0.9`; round joins and caps

- [ ] **Step 3: The runner `run(root)` in `public/opening.js`** (behaviour, in order):
  1. Look up the elements. `reduced = matchMedia("(prefers-reduced-motion: reduce)").matches`. Read `remembered = parseRemembered(localStorage.getItem(STORAGE_KEY))`, inside try/catch (no storage means null). `plan = planFor(remembered)`. Listen for `lorren:ready`, recording `readyPerf = performance.now()`. Start a buffered `PerformanceObserver({ type: "longtask" })` and keep the longest task. Set `phase = "font"`.
  2. `Promise.race([document.fonts.load('600 112px "Lorren Opening"'), a 1000 ms timer])`, then the next animation frame. Its timestamp is `t0`, the first frame.
  3. At t0, measure the word: `W` and `H` from `.op-word`'s box; the baseline from a zero-size `inline-block` probe appended to `.op-lit` (`vertical-align: baseline`), read, then removed. Set the svg's `width`/`height` to W/H and the text's `x = 0`, `y = baseline - box top`. Build the driven animations with `el.animate(keyframes, { duration, delay, easing, fill: "both" })`, then `pause()` them at `currentTime = 0`. The keyframes are the mockup's `build()` (lines 184-197):

| element | keyframes | duration | delay | easing |
|---|---|---|---|---|
| `.op-line` | `strokeDashoffset` 360 → 0 | 950 | 0 | `cubic-bezier(.45,0,.2,1)` |
| `.op-pressed` | `opacity` 0 → 1 | 400 | 550 | `ease-out` |
| `.op-lit` | `maskPosition` (and `webkitMaskPosition`) `84% 0` → `16% 0` | 1400 | 600 | `cubic-bezier(.3,.1,.7,.9)` |
| `.op-spot` | `{backgroundPosition: 84% 0, opacity 1}` → `{28.2% 0, opacity 1, offset .82}` → `{16% 0, opacity 0}` | 1400 | 600 | same |
| `.op-halo` | `{translateX(-0.18·W px), opacity 0}` → `{opacity 1, offset .2}` → `{opacity 1, offset .8}` → `{translateX(1.18·W px), opacity 0}` | 1400 | 600 | same |
| `.op-line` | `strokeOpacity` 0.9 → 0 | 700 | 1300 | `ease-out` |

     Then make `.op-word` visible, set `phase = "playing"`, and start the frame loop. Under reduced motion, set every driven animation to `currentTime = T` (lit), and fade `.op-word` in with opacity 0 → 1 over 400 ms, `ease-out`.
  4. Each frame: `gap = now - last` (record the longest raw gap); `real = now - t0`; `readyAt = readyPerf === null ? null : max(0, readyPerf - t0)`. If `valveDue(performance.now(), readyAt)`, set `readyAt = real` and `valve = true`. `s = pace(s, …)`. Set every driven animation's `currentTime = s.it`, and `.op-word.style.filter = glowFilter(s.glow, real)`. Record `finishedAt` the first time `s.it >= T`. Once `real >= handoverAt(finishedAt, readyAt)`, start the hand-over. Under reduced motion there is no pacing: the hand-over starts at `real >= reducedHandoverAt(readyAt)`, or at the valve.
  5. The hand-over: `phase = "handover"`, `#opening.style.pointerEvents = "none"`. If the wordmark `.topbar .wordmark` exists with a non-zero width and motion is not reduced, measure:
     - `from`: the word box; the text's left and width from a `Range` over `.op-lit`'s text node; the baseline from the probe.
     - `to`: the same for `.wordmark`, with the probe appended to it and removed in the same task.
     Then animate `.op-word` `transform: translate(0,0) scale(1)` → `translate(tx px, ty px) scale(s)` over 760 ms, `cubic-bezier(.65,0,.35,1)`, `fill: "forwards"`, and `.op-backdrop` opacity 1 → 0 over 520 ms, delay 260, `ease-out`, `fill: "forwards"`. On the glide's `finished`, measure the landing error: the max of |left|, |width| and |baseline| differences between the transformed text and the wordmark's. Otherwise (reduced, or no wordmark), fade `#opening` from opacity 1 to 0 over 520 ms, `ease-out`.
  6. The swap, in one task: remove the `opening` class from `<html>`, remove `#opening`, set `phase = "done"`, build the report, set `__lorrenOpening.report`, send it (see Produces), and if ready came without the valve, `localStorage.setItem(STORAGE_KEY, String(Math.round(readyAt)))`.
  7. Any exception in steps 3-5 sets `error` to its message, stops the loop, and takes the fade path of step 5. Nothing is thrown out of the script.

- [ ] **Step 4: Verify the build**

Gate first (the box rule), then from `desktop/`: `npm run build`.
Expected: exit 0, and `dist/index.html` has:
- `class="opening"`
- a preload `href="/assets/Archivo-600-` … `.ttf"`
- the `#opening` markup
- `<script src="/opening.js"></script>` before the end of `<body>`
- no `<style>`

Also: `cmp public/opening.js dist/opening.js` is silent; `grep -c "Lorren Opening" dist/assets/*.css` is ≥ 1; `grep -c "op-valve" dist/assets/*.css` is ≥ 1; `npx vitest run` passes; and `node -e "JSON.parse(require('fs').readFileSync('src-tauri/tauri.conf.json'))"` parses.

- [ ] **Step 5: Look at it.** `npx vite preview --port 4401 --strictPort`, run in the background with its PID recorded. In the Playwright browser, run `browser_run_code_unsafe` with `goto http://127.0.0.1:4401/`, then screenshots at 0.5, 1.3 and 2.3 s after load. Expected: the outline half drawn at 0.5 s; the light partway across at 1.3 s; the word fully lit at 2.3 s; no fallback face in any. Then kill the preview by its PID.

  With no Tauri, the app's calls fail and Task 3's event does not exist yet, so the word waits and glows, and at 12 s the valve fades it out. Check one screenshot at 13 s: the app (top bar "host unreachable") with its wordmark visible.

- [ ] **Step 6: Commit**

```bash
git add index.html src/opening/opening.css public/opening.js src-tauri/tauri.conf.json src-tauri/tauri.e2e.conf.json
git commit -m "desktop: the opening on screen -- drawn then lit, the hand-over onto the wordmark, the valves, the theme background behind the window (spec 2026-10-23)"
```

---

### Task 3: The ready signal

**Files:**
- Create: `src/opening/ready.ts`
- Modify: `src/App.tsx` (report `overview`), `src/chat/ChatTab.tsx` (report `chats`, `chat`, `models`)
- Test: `src/__tests__/ready.test.tsx`

**Interfaces:**
- Produces:
  - `export type ReadyPart = "overview" | "chats" | "chat" | "models"`
  - `export const READY_EVENT = "lorren:ready"`
  - `export function reportReady(part: ReadyPart): void`: idempotent. When the fourth distinct part arrives, it waits for `document.fonts?.ready` (if present), then two animation frames (falling back to `setTimeout(16)` where there is no `requestAnimationFrame`), then dispatches `new Event(READY_EVENT)` on `window`, once per page load.
  - `export function resetReady(): void` — tests only. It forgets the parts and cancels a pending dispatch.
- Where they are reported:
  - App, in an effect on `[overview, error]`: `overview !== null || error !== undefined` → `"overview"` (this covers "host unreachable").
  - ChatTab, in the models call's success **and** failure handlers → `"models"`.
  - ChatTab, in the first chat-list load's `.then`: `"chats"`; and `"chat"` as well when the list is null (failed) or empty.
  - ChatTab, in an effect on `[view]`: `view !== null` → `"chat"`. Also in `reload`'s catch → `"chat"`.

- [ ] **Step 1: Write the failing tests** in `src/__tests__/ready.test.tsx`. Mock the api as `App.test.tsx` does; in `beforeEach`, call `resetReady()` and add a `lorren:ready` listener spy. Cases:
  - "fires once, after the fourth part": `reportReady` overview, chats, models → after 100 ms the spy has 0 calls. Then `"chat"` → `waitFor` 1 call. Then report all four again → still 1 call after 100 ms.
  - "a fresh app with no chats is ready once the overview, the empty list and the models are in": `render(<App />)` with the default mocks → `waitFor` 1 call.
  - "waits for the chat list": `chats` returns a promise held open → 0 calls after 200 ms. Resolve with `[]` → 1 call.
  - "waits for the open chat's messages, and they are on screen when it fires": `chats → [chat]`, `chat` held open → 0 calls after 200 ms. Resolve with a view holding a message → 1 call, and in the listener itself `screen.queryByText(<that message's text>)` is not null.
  - "fires for a host that does not answer": `overview` rejects → 1 call, and the top bar shows `host unreachable`.
  - "fires when the chat list fails": `chats` rejects → 1 call.
  - "does not wait for the model to warm up": an overview whose engine `state` is `"starting"` → 1 call.

- [ ] **Step 2: Run them to verify they fail**

Run: `npx vitest run src/__tests__/ready.test.tsx`
Expected: FAIL (`../opening/ready` cannot be resolved).

- [ ] **Step 3: Implement `ready.ts`, then the five report calls** (the Interfaces list). Each call site gets a one-line comment naming spec §3.

- [ ] **Step 4: Run all tests**

Run: `npx vitest run` and `npx tsc --noEmit`.
Expected: all PASS, including the existing `App.test.tsx` and `ChatTab.test.tsx` unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/opening/ready.ts src/App.tsx src/chat/ChatTab.tsx src/__tests__/ready.test.tsx
git commit -m "desktop: the app tells the opening when everything visible is in place (spec 2026-10-23 section 3)"
```

---

### Task 4: The host: the log line, the launch times, the e2e's hold

**Files:**
- Modify: `src-tauri/src/commands.rs`:
  - add `OpeningReport`, `opening_line`, the `opening_report` command, `hold_left`, `debug_hold`;
  - make `models` async;
  - give `AppState` `launched_ms` and `shown_ms`;
  - add the tests in a `#[cfg(test)] mod opening_tests`.
- Modify: `src-tauri/src/lib.rs`:
  - take `launched_ms` at the top of `run()`, and `shown_ms` right after `window_state::restore(app)`;
  - the setup closure becomes `move`;
  - add both times to `AppState`;
  - register `commands::opening_report`.

**Interfaces:**
- Consumes: Task 2's `Report` (field for field; `serde::Deserialize`, snake_case as sent).
- Produces:
  - `#[derive(Debug, Clone, Deserialize)] pub struct OpeningReport { pub navigation_ms: f64, pub first_frame_ms: Option<f64>, pub ready_ms: Option<f64>, pub finished_ms: Option<f64>, pub handover_start_ms: f64, pub handover_end_ms: f64, pub plan_ms: f64, pub remembered_ms: Option<f64>, pub longest_gap_ms: f64, pub longest_task_ms: Option<f64>, pub longest_task_at_ms: Option<f64>, pub landing_error_px: Option<f64>, pub font_ms: Option<f64>, pub reduced_motion: bool, pub valve: bool, pub error: Option<String> }`
  - `pub fn opening_line(r: &OpeningReport, launched_ms: i64, shown_ms: i64) -> (&'static str, String)` (level, text)
  - `#[tauri::command] pub fn opening_report(state: State<'_, AppState>, report: OpeningReport)` → `state.monitor.log("app", level, text)`
  - `#[tauri::command] pub async fn models(state: State<'_, AppState>) -> Result<Vec<ModelInfo>, String>` (the frontend's `invoke<ModelInfo[]>("models")` is unchanged)
  - `pub fn hold_left(launched_ms: i64, hold_ms: Option<u64>, now_ms: i64) -> u64` → `max(0, launched_ms + hold - now_ms)`, and 0 when `hold_ms` is None
  - `debug_hold(launched_ms)`: in a `#[cfg(debug_assertions)]` build it reads `LORREN_DEBUG_HOLD_MODELS_MS` and `tokio::time::sleep`s for `hold_left(…)`; the `#[cfg(not(debug_assertions))]` twin is empty.

The line: `opening: window {shown-launched} ms, page {navigation-launched} ms, first frame {first_frame-launched} ms, ready {ready-launched} ms, in the top bar {handover_end-launched} ms after launch; planned {plan/1000:.1} s ({first start | last start {remembered/1000:.1} s}); longest frame gap {gap:.0} ms`. The parts that follow are each present only when they apply:
- `; longest task {t:.0} ms at {at:.0} ms`
- `; landed within {err:.2} px`
- `; reduced motion: faded`
- `; no wordmark to land on: faded` (no landing error and not reduced)
- `; the opening failed: {error}`

When `ready_ms` is None, the ready clause reads `ready never came (the 12 s valve)`; when `first_frame_ms` is None, `first frame never`. The level is `"warn"` when `valve` or `error` is set, else `"info"`.

- [ ] **Step 1: Write the failing tests** (`cargo test` names):
  - `opening_line_reads_a_normal_start`: `launched 1_000_000`, `shown 1_000_012`, report `navigation 1_000_380.4`, `first_frame 1_000_455.2`, `ready 1_001_430.7`, `finished 1_002_455.0`, `handover_start 1_002_756.0`, `handover_end 1_003_515.9`, `plan 2000`, `remembered None`, `gap 18.4`, `task Some(140.2)` `at Some(80.0)`, `landing Some(0.0213)`, `font Some(60.0)`, flags false, `error None`. Expected `("info", "opening: window 12 ms, page 380 ms, first frame 455 ms, ready 1431 ms, in the top bar 3516 ms after launch; planned 2.0 s (first start); longest frame gap 18 ms; longest task 140 ms at 80 ms; landed within 0.02 px")`.
  - `opening_line_reads_the_valve`: the same but `ready None`, `valve true`, `plan 5000`, `remembered Some(7200.0)`, `handover_end 1_013_080.0`, `task None`, `landing Some(0.03)`. Expected `("warn", "opening: window 12 ms, page 380 ms, first frame 455 ms, ready never came (the 12 s valve), in the top bar 13080 ms after launch; planned 5.0 s (last start 7.2 s); longest frame gap 18 ms; landed within 0.03 px")`.
  - `opening_line_reads_reduced_motion`: the first report with `landing None`, `reduced true`, `task None` → ends `"; longest frame gap 18 ms; reduced motion: faded"`, `"info"`.
  - `opening_line_reads_a_failure`: `landing None`, `error Some("TypeError: x is undefined")` → ends `"; no wordmark to land on: faded; the opening failed: TypeError: x is undefined"`, `"warn"`.
  - `hold_left_counts_from_launch`: `hold_left(1000, None, 1500) == 0`, `hold_left(1000, Some(6000), 1500) == 5500`, `hold_left(1000, Some(6000), 9000) == 0`.

- [ ] **Step 2: Run them to verify they fail**

Gate first. Run from `src-tauri/`: `cargo test --lib opening` (and `hold_left`).
Expected: compile errors naming `OpeningReport`, `opening_line`, `hold_left`.

- [ ] **Step 3: Implement the Interfaces list**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib` (the whole lib suite) and `cargo build`.
Expected: all PASS; no new warnings.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/commands.rs src-tauri/src/lib.rs
git commit -m "desktop: one log line per opening (spec 2026-10-23 section 7), the launch and window times it is measured from, and a debug-only hold on models for the e2e's slow host"
```

---

### Task 5: The e2e: a fast host, a host held back 6 s, reduced motion

**Files:**
- Modify: `e2e/run-demo-e2e.mjs`

**Interfaces:**
- Consumes:
  - Task 2's `window.__lorrenOpening` (`phase`, `report`) and the `Report` fields;
  - the class names `#opening`, `.op-word`, `html.opening`;
  - Task 4's log line (`opening: …`, fetched by `invoke("logs", { after: 0 })`) and `LORREN_DEBUG_HOLD_MODELS_MS`.

- [ ] **Step 1: The harness**
  - `env()` adds `WEBVIEW2_USER_DATA_FOLDER: path.join(WORK, "webview")` (R5), and `main()` removes that folder with the others.
  - `launch(overrides = {}, { keepOpening = false } = {})`:
    - record `spawnedAt = Date.now()` before the spawn;
    - once `page` is found, wait until `document.getElementById("root")` exists, then take `firstFrame = await page.evaluate(openingSnapshot)` and store `lastLaunch = { spawnedAt, firstFrame }`;
    - unless `keepOpening`, wait (20 s) for `__lorrenOpening.phase === "done"`, then for `.topbar .wordmark` to be visible;
    - this replaces the `getByText("lorren", { exact: true })` wait, which now matches the opening's layers too.
  - `openingSnapshot` (in-page, beside `layoutDefects`) returns `{ present, phase, word: DOMRect-like | null, vw: innerWidth, vh: innerHeight, hitsOpening }`. `hitsOpening` = whether `document.elementFromPoint` at the centre of the Models tab (`[role=tab]` named Models) lies inside `#opening`; null if the tab has no box.

- [ ] **Step 2: Three steps.** `"the opening: …"` goes FIRST in `steps`; the other two go LAST, in this order.
  - **"the opening: on screen at the first frame, lit by 2.0 s, handed over onto the wordmark, logged"**:
    - `lastLaunch.firstFrame.present` and its `phase !== "done"`. Otherwise fail with "the harness connected after the opening ended (X ms after launch)", an instrument failure, said as such.
    - The word is centred within 1 px on both axes.
    - The report:
      - `ready_ms !== null`, `!valve`, `!reduced_motion`, `error === null`;
      - `finished_ms - first_frame_ms >= 1983`;
      - `handover_start_ms - Math.max(finished_ms, ready_ms) >= 298`;
      - `landing_error_px <= 0.5`.
    - After the hand-over: no `#opening`, no `opening` class, and `.topbar .wordmark` computed `visibility === "visible"`.
    - Exactly one log line starting `opening: `, and it contains `landed within`.
    - `measurements.opening` = the report's first frame (from navigation), ready (from the first frame), top bar (from the first frame), longest gap, longest task, landing, plan — a measurement (R66), not asserted.
    - Save a screenshot of the first captured frame as `shots/opening-first.png`, taken in `launch()` right after the snapshot.
  - **"the opening with the host held back 6 s: centred and taking the clicks until ready, then lands"**:
    - `await close()`, then `launch({ LORREN_DEBUG_HOLD_MODELS_MS: "6000" }, { keepOpening: true })`.
    - Every 200 ms until the phase is `"handover"` or `"done"` (20 s cap), take a snapshot. Every snapshot with phase `"playing"` must have the word centred within 1 px and `hitsOpening === true`.
    - At least 10 such snapshots (a control that the sampling ran).
    - The first snapshot that is no longer `"playing"` is taken at least 6000 ms after `spawnedAt`. This is the harness's own clock, not the report's word for it.
    - Then wait for `"done"`. The report:
      - `ready_ms - lastLaunch.spawnedAt >= 6000`;
      - `!valve`;
      - `handover_start_ms >= ready_ms`;
      - `finished_ms >= ready_ms - 20` (it did not finish while loading);
      - `landing_error_px <= 0.5`.
    - The log has its `opening: ` line.
    - `measurements.opening_held` as above.
  - **"reduced motion: the lit word fades in, and out after ready"**:
    - `await page.emulateMedia({ reducedMotion: "reduce" })`, `await page.reload()`, then wait for `"done"`.
    - The report has `reduced_motion === true`, `landing_error_px === null`, `ready_ms !== null` and `handover_start_ms >= ready_ms`.
    - `.wordmark` is visible.
    - The newest log line contains `reduced motion: faded`.

- [ ] **Step 3: Run the whole e2e** (gate first; it builds the debug app and the fake engine).

Run: `npm run e2e`
Expected: the last line is `e2e: 16/16 OK` (13 existing steps + 3), and `measurements.json` has `opening` and `opening_held`. A failure is diagnosed (superpowers:systematic-debugging) before anything is changed.

- [ ] **Step 4: Run the held-host step red once.** Lowering `HANDOVER_DELAY_MS` proves nothing: `pace` never lets the word finish while loading, so the hand-over would still wait. Make the opposite edit instead. Temporarily make the runner treat the page as ready at its first frame (set `readyPerf = t0` there, ignoring `lorren:ready`), then run `npm run e2e`. It must rebuild, because the debug exe embeds `dist/`. Expected: the held-host step FAILS on "the first snapshot no longer playing is ≥ 6000 ms after spawn". Revert, run `npm run e2e` once more (16/16), and record the red line in the ledger.

- [ ] **Step 5: Commit**

```bash
git add e2e/run-demo-e2e.mjs
git commit -m "e2e: the opening with a fast host, a host held back 6 s and reduced motion; the e2e's own WebView2 profile"
```

---

### Task 6: On Kabir's PC — measure, then he watches

**Files:**
- Modify: `.superpowers/sdd/2026-11-06-lorren-opening/progress.md` (the readings), `docs/superpowers/specs/2026-10-23-lorren-opening-design.md` (status line: built, and the readings in one line)

- [ ] **Step 1: Gate, then build the release exe without the installer:** `npm run tauri build -- --no-bundle`. Expected: `src-tauri/target/release/lorren.exe`, built from this branch's HEAD (compare the exe's mtime with the build start).
- [ ] **Step 2: Four launches** (close Kabir's installed Lorren first; single instance). Each launch uses `LORREN_ENGINE_EXE=src-tauri/target/debug/fake-engine.exe` (no model server, per the standing rule), with his real data and library. Close each by `taskkill /PID` (graceful) after its `opening:` line appears in `%LOCALAPPDATA%/Lorren/logs/lorren.log`. Launch 1 comes right after the build; launches 2-4 are warm. Record the four lines.
- [ ] **Step 3: Decide from the numbers** and record each decision as a `Ruling:` in the ledger:
  - median `page - window` over launches 2-4 > 150 ms → Task 7;
  - `longest frame gap` > 50 ms on 2 or more of launches 2-4 → Task 8;
  - otherwise both are N/A, with the numbers.
- [ ] **Step 4: Kabir watches.** In two lines, tell him the release exe's full Windows path, to start it twice (once now, once after closing it), and what to look for. His verdict goes in the ledger. A change he asks for becomes its own task.
- [ ] **Step 5: Commit** the ledger and the spec status line: `git commit -m "docs: the opening measured on Kabir's PC"`.

---

### Task 7 (only if Task 6 says so): the library scan leaves the main thread

**Files:**
- Modify: `src-tauri/src/lib.rs`:
  - the scan, its log line and the default model's start move into a `std::thread::spawn`, after `AppState` is managed;
  - `library` starts as an empty list.
- Modify: `src-tauri/src/commands.rs`:
  - `AppState` gains `library_ready: tokio::sync::watch::Receiver<bool>`;
  - `models` waits on it before reading, using `wait_library(rx: &mut watch::Receiver<bool>)`.

**Interfaces:**
- Produces: `pub async fn wait_library(rx: &mut tokio::sync::watch::Receiver<bool>)`, which returns once the value is `true`, at once if it already is, and also if the sender has been dropped.

- [ ] **Step 1: The failing test:** `#[tokio::test] wait_library_returns_once_the_scan_is_published`. It spawns `wait_library` on a fresh channel (false), asserts the task is not finished after 50 ms, sends `true`, and asserts it finishes within 1 s. A second case: a dropped sender also returns.
- [ ] **Step 2: Run it red:** `cargo test --lib wait_library`. Expected: a compile error naming `wait_library`.
- [ ] **Step 3: Implement it, and move the scan.** The `library: …` log line keeps its text. The `ChatService` keeps the same `Arc<RwLock<…>>`.
- [ ] **Step 4: Verify.** Run `cargo test --lib`, then `npm run e2e` (16/16), then Task 6's step 2 again. Expected: median `page - window` ≤ 150 ms, and the `library:` line still before `starting the engine`.
- [ ] **Step 5: Commit** `"desktop: the library is scanned beside the window, not before it, so the opening paints at once (R9)"`.

---

### Task 8 (only if Task 6 says so): the frame budget

- [ ] **Step 1: Attribute.** The log line's `longest task … at …` names when the long task ran. If it ran before the first React commit, it is the bundle's evaluation: compare the task's start with the navigation-to-first-frame time; React commits after the bundle has run. If it ran after, it is the first render.
- [ ] **Step 2: The remedy for that cause.** Each is a separate commit, with ChatTab's tests and the e2e green:
  - **First render**: ChatTab renders its shell first. The chat list and the open chat's messages render from `useDeferredValue` copies, and the `"chat"` / `"chats"` reports move to the effects of those deferred renders, so ready still means "on screen".
  - **Bundle evaluation**: `chat/Markdown.tsx`'s importers load it with `React.lazy` (the katex / highlight / react-markdown stack leaves the first chunk). A message waiting for it shows its plain text; the ready report for `"chat"` waits for the lazy chunk.
- [ ] **Step 3: Re-measure.** Task 6's step 2. Expected: the longest frame gap ≤ 50 ms on all of launches 2-4. Ledger the before/after lines.

---

## Execution notes for the manager

- Ledger: `.superpowers/sdd/2026-11-06-lorren-opening/progress.md`, kept with the same shape as step 2's ledger. Tasks 1, 3, 4 and 5 go to Sonnet implementers with this plan as the brief; the manager reviews every diff itself. Task 2 is the manager's own, because it is the visual port and the design call. Tasks 6-8 are the manager's too.
- Step 2 of distribution (Discover, `docs/superpowers/plans/2026-11-04-lorren-distribution-step2-discover.md`) waits: Kabir, 2026-11-06, "do the app animation first".
