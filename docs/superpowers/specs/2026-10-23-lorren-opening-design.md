# Lorren's opening — design

Status: designed with Kabir on 2026-10-23 (brainstorming, ten mockup rounds); APPROVED by Kabir 2026-10-23 ("approved"). BUILT
2026-11-06 (plan `docs/superpowers/plans/2026-11-06-lorren-opening.md`, commits `4c7cb71`..`b4a7849`; e2e 17/17); Kabir on the
installed build: "looks good". Measured on his PC (release, his library, fake engine): first frame 40-88 ms after the window,
ready ~0.55 s and in the top bar ~3.5 s after launch, longest frame gap 33-40 ms, landing 0.00 px. The CSS is a linked
stylesheet, not inline (Tauri nonces every <style>; the plan's R1), and the library scan runs beside the window (R9). This
is the authority for the opening's implementation plan. Reference mockup: `stoke-lorren/.superpowers/brainstorm/1092-1792727290/content/opening-v10.html`
(git-ignored, throwaway: port its values, not its structure). Parent spec: `2026-09-25-lorren-desktop-design.md`.

## 1. Intent

When Kabir opens Lorren, the window must appear at once — not a blank or white window that fills in later — and
the first thing in it must feel premium: the letters of "lorren" are drawn and then lit, while the app finishes
loading behind them. The opening hides the load; it never adds a wait of its own beyond two seconds, and it never
lets the app appear half-built.

## 2. What the person sees

1. **The window appears immediately**, already in the app's background colour (`--bg`, `oklch(0.15 0.006 255)`),
   with no white flash at any point.
2. **Drawn (0 to 0.95 s).** The outline of each letter of the lowercase wordmark "lorren" (Archivo 600, the
   top bar's wordmark, 112 CSS px at every window size — the system's display scaling applies — centred in the
   window's content area) draws itself — all six letters together, a 1.2 px
   light line. From 0.55 s the letters settle into the surface inside their outlines: dark, slightly pressed in
   (an emboss: `oklch(0.205 0.007 255)` with a dark top edge and a faint light bottom edge).
3. **Lit (0.6 to 2.0 s).** A warm light crosses the word from left to right at an even pace, about one second across
   the letters (0.79 to 1.8 s): a soft warm halo behind, a hot spot on the letters it is passing, and behind it the
   letters are lit (the app's `--text` white). The outline fades as the light passes. The last letter is fully lit
   at 2.0 s. (Kabir slowed the light twice; it was 0.4 s across the letters.)
4. **The hand-over.** Once Lorren is ready (section 3), the word glides from the centre into the top bar's wordmark
   (0.76 s, ease in-out), shrinking to it exactly, while the app fades in around it (0.52 s, starting 0.26 s in).
   At the end the real top-bar wordmark takes over; the swap is invisible.

The mockup's measured timings for this sequence are the reference.

## 3. When Lorren is ready

Lorren is **ready** when everything the person would see behind the word is in place:

- the host's first overview has arrived (the top bar's engine status and version are known), **or** it has failed
  in a way the app shows (the "host unreachable" state);
- the chat list is loaded, the open chat's messages are rendered (or the empty state is), and the models list is
  loaded;
- the fonts are loaded, and one frame has been painted after those renders.

**Not** included: the model warming up in the engine. That can take several seconds and already shows in the top
bar's engine status (Kabir, 2026-10-23: "dont wait to warm up the model").

The word **never leaves the centre before Lorren is ready.** Safety valve: if ready has not come 12 s after the
window appeared, the hand-over happens anyway and the app shows whatever state it has (its own loading or error
states) — a stuck host must never trap the person behind the opening.

## 4. The pacing (dynamic)

- **Two seconds by default, never shorter.** Even when Lorren is ready at 0.5 s, the word finishes at 2.0 s.
- **It plans from the last start.** The app remembers how long the previous start took (from the opening's first
  frame to ready) and
  plans the whole animation for that time from the first frame, between 2 s and 5 s: on a PC that takes 4 s, every
  part plays at half speed (the drawing too). The first start ever plans for 2 s.
- **It corrects itself while it plays**, every frame, with every speed change eased (never a jerk):
  - Still loading: it follows the plan until 0.6 s before the planned end; from there it slows down, and as it
    nears the end it slows in proportion to what is left, so it keeps moving but never finishes while Lorren is
    loading. Once the light has crossed the letters, the lit word breathes a soft warm glow while it waits.
  - Ready before 2 s: it moves at exactly the speed that lands the word on 2.0 s.
  - Ready after 2 s: it finishes what is left briskly (up to 1.6×), then hands over 0.3 s later.
- After each start the time it took is remembered for the next one.

Measured in the mockup (Kabir's PC, 2026-10-23): ready at 1 s → finished 2.0 s; a first start ready at 4 s →
finished exactly 4.0 s; a learned 4 s start → 0.5× throughout, finished 4.1 s; ready at 1 s after a learned 4 s
plan → sped up to 1.6×, finished 2.1 s; ready at 7 s → slowed steadily, finished exactly 7.0 s. The word never
left the centre before ready in any case.

## 5. When it plays

- **Every launch** of Lorren, including the restart after an update installs.
- **Not** when Lorren is already running and is opened again: the existing window comes forward, as today.
- **Reduced motion** (Windows' "Animation effects" off, which the webview reports as `prefers-reduced-motion`):
  no drawing, no light, no glide — the lit word fades in, and fades out into the app once Lorren is ready (still
  never before ready, still the 12 s valve).

## 6. Constraints

- **The opening must not wait for the app's code.** It is in `index.html` itself — its markup and its CSS, inline
  — so it is on screen from the webview's first paint. The pacing is a small separate script file loaded before the
  app bundle (the app's content security policy forbids inline scripts; keep it that way).
- **No font flash.** The wordmark's font (the bundled Archivo 600) is preloaded and the opening's text waits for it
  (`font-display: block`); no frame may show a fallback face. (If a measurement shows the font arriving late, the
  glyphs become SVG outlines generated from the same font file at build time.)
- **The window's own background is the theme background**, set in the window configuration (`backgroundColor`) and
  on `html`/`body` in `index.html`, so the native window, the webview before its first paint, and the page agree.
- **Smooth while the app loads behind it.** The app's first render runs on the same thread as the opening. Budget:
  no frame of the opening more than 50 ms late on Kabir's PC. Measure it (the opening logs its longest frame gap);
  if the app's first render breaks the budget, split that render (shell first, then the lists), never shorten the
  opening.
- **Never a trap**: the 12 s valve, and a failure of the app's code (an error before it can say ready) still ends
  the opening at the valve, revealing the app's error screen.
- **The hand-over lands exactly** on the top bar's wordmark (the mockup lands within 0.03 px), measured from the
  live layout, at any window size, maximised or not.
- **Accessibility**: the opening is decoration (`aria-hidden`); the app's own content is what assistive technology
  reads, and keyboard focus is not taken by the opening.
- **No new dependencies** for the opening itself.

## 7. What it records

The app's log (the Logs tab, `lorren.log`) gets one line per start: when the window appeared, when the opening
first painted, when Lorren was ready, when the hand-over ended, the plan it used, and its longest frame gap. That is
how the timings above are checked on real PCs, and where the remembered start time comes from.

## 8. How it is tested

- The pacing as a pure function (given the plan, the clock and the ready moment → speed and position): never
  finishes while loading, never before 2.0 s, lands on 2.0 s when ready early, plans from the remembered time and
  clamps it to 2–5 s, eases every change, the 12 s valve, reduced motion. Each test is run red first.
- The ready signal: it does not fire until all of section 3's parts are in, and fires on the host-unreachable state.
- End to end, in the app's existing harness: the opening is on screen in the first captured frame, the word stays
  centred until ready, the hand-over lands on the wordmark, and the log line is written — with a fast host and with
  a host held back for 6 s.
- On Kabir's PC, by the manager: the time from launch to the opening's first paint, the longest frame gap during the
  opening, and Kabir watching it, cold start and warm start.

## 9. Decided, and left to the builder

Decided with Kabir: the sequence and its timings (sections 2 and 4), the two-second default, the dynamic pacing,
waiting for everything visible but not for the model, never leaving the centre before ready, every launch.

The manager's calls (Kabir can overrule): the 12 s valve; the 2–5 s planning range; the reduced-motion behaviour;
remembering the start time in the webview's local storage; the 50 ms frame budget; the log line.

Left to the builder: the markup and file layout of the opening, how the ready signal is wired through the app
(any small module the components report to), and how the window configuration is changed — within the
constraints above.

## 10. Sequencing

This touches `index.html`, the app's root component, the top bar, the chat tab's loading, and the window
configuration. The paused demo-fix wave (D9 phase B) has uncommitted changes in the chat tab and the host: **D9
lands first**, then this is planned on top of it. Building and measuring wait for the box to be free of the M7 soak.
