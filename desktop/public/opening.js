/*
 * Lorren's opening, "drawn, then lit": the launch animation that plays while the app loads behind it.
 * Spec: docs/superpowers/specs/2026-10-23-lorren-opening-design.md (approved d0b7dcf); the values are those of the
 * approved mockup (opening-v10.html). Plan: docs/superpowers/plans/2026-11-06-lorren-opening.md.
 *
 * A CLASSIC script (no imports or exports), loaded by a plain <script src="/opening.js"> before the app's bundle: the
 * opening must not wait for the app's code, and the app's CSP allows no inline script. index.html holds the markup and
 * src/opening/opening.css the static look; this file holds the timeline, the pacing and the hand-over.
 *
 * The first half is pure (the pacing, exposed as `LorrenOpening` only so the unit tests can reach it:
 * src/__tests__/opening.test.ts); the second half, `run`, plays it in the page.
 *
 * All times are ms on the opening's clock: `real` counts from its first frame, `it` is the position on the default
 * 2 s timeline.
 */
(function (root) {
  "use strict";

  const T = 2000;
  const PLAN_MIN = 2000;
  const PLAN_MAX = 5000;
  const R_MIN = 0.2;
  const R_MAX = 1.6;
  const VALVE_MS = 12000;
  const HANDOVER_DELAY_MS = 300;
  const REDUCED_MIN_MS = 900;
  const STORAGE_KEY = "lorren.opening.lastStartMs";

  function clamp(x, lo, hi) {
    return Math.min(hi, Math.max(lo, x));
  }

  function round2(x) {
    return Number(x.toFixed(2)).toString();
  }

  function parseRemembered(raw) {
    if (raw === null || raw === undefined || raw === "") return null;
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : null;
  }

  function planFor(remembered) {
    if (remembered === null || remembered === undefined || !Number.isFinite(remembered)) return T;
    return clamp(remembered, PLAN_MIN, PLAN_MAX);
  }

  function begin(plan) {
    return { it: 0, rate: T / plan, glow: 0 };
  }

  function pace(s, f) {
    const dt = Math.min(64, f.dt);
    let target;
    let ease;
    if (f.readyAt === null) {
      const onPlan = f.real < f.plan - 600;
      target = onPlan ? T / f.plan : Math.min(R_MIN, (T - s.it) / 900);
      ease = onPlan ? 120 : 400;
    } else {
      target = f.real < T ? (T - s.it) / Math.max(1, T - f.real) : R_MAX;
      ease = 200;
    }
    target = clamp(target, 0, R_MAX);
    const rate = s.rate + (target - s.rate) * (1 - Math.exp(-dt / ease));
    let it = Math.min(T, s.it + dt * rate);
    if (f.readyAt === null) it = Math.min(it, T - 0.5);
    const waiting = f.readyAt === null && T - it < 120;
    const glow = waiting ? Math.min(1, s.glow + dt / 500) : Math.max(0, s.glow - dt / 300);
    return { it: it, rate: rate, glow: glow };
  }

  function handoverAt(finishedAt, readyAt) {
    return finishedAt !== null && readyAt !== null
      ? Math.max(finishedAt, readyAt) + HANDOVER_DELAY_MS
      : null;
  }

  function valveDue(sinceNavigation, readyAt) {
    return readyAt === null && sinceNavigation >= VALVE_MS;
  }

  function reducedHandoverAt(readyAt) {
    return Math.max(REDUCED_MIN_MS, readyAt);
  }

  function glowFilter(glow, real) {
    if (glow <= 0) return "";
    const b = glow * (0.5 + 0.5 * Math.sin(real / 520));
    return (
      "drop-shadow(0 0 " + round2(6 + 10 * b) + "px oklch(0.85 0.07 75 / " + round2(0.1 + 0.22 * b) + "))"
    );
  }

  // CSS px; the transform origin is the word box's top-left.
  function glide(from, to) {
    const s = to.textWidth / from.textWidth;
    return {
      tx: to.textLeft - from.boxLeft - s * (from.textLeft - from.boxLeft),
      ty: to.baseline - from.boxTop - s * (from.baseline - from.boxTop),
      s: s,
    };
  }

  // ---- The runner: the page's side (everything above this line is pure, and tested) ----

  const READY_EVENT = "lorren:ready"; // fired once by the app (src/opening/ready.ts) when everything visible is in place
  const FONT = '600 112px "Lorren Opening"';
  const FONT_WAIT_MS = 1000; // the longest the first frame waits for the font (plan ruling R8)
  const GLIDE_MS = 760;
  const GLIDE_EASE = "cubic-bezier(.65,0,.35,1)";
  const FADE_MS = 520; // the app fading in (the backdrop fading out), and every fade-out of the opening
  const FADE_DELAY_MS = 260; // into the glide
  const FADE_IN_MS = 400; // reduced motion: the lit word fading in
  const LIGHT_EASE = "cubic-bezier(.3,.1,.7,.9)";

  /** Plays the opening in the page: `root` is the window. Never throws: a failure fades the opening out and is reported. */
  function run(root) {
    const doc = root.document;
    const perf = root.performance;
    const opening = doc.getElementById("opening");
    const pick = (cls) => opening.querySelector("." + cls);
    const backdrop = pick("op-backdrop");
    const word = pick("op-word");
    const lit = pick("op-lit");
    const pressed = pick("op-pressed");
    const spot = pick("op-spot");
    const halo = pick("op-halo");
    const trace = pick("op-trace");
    const line = pick("op-line");

    // What the e2e and anyone debugging can read: the phase, and the report once it is over.
    const state = { phase: "font", report: null };
    root.__lorrenOpening = state;

    let reduced = false;
    try {
      reduced = root.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch (e) {}
    let remembered = null;
    try {
      remembered = parseRemembered(root.localStorage.getItem(STORAGE_KEY));
    } catch (e) {}
    const plan = planFor(remembered);

    // The app says it is ready once; the time is kept on the page's clock and turned into the opening's when it is used.
    let readyPerf = null;
    root.addEventListener(READY_EVENT, () => {
      if (readyPerf === null) readyPerf = perf.now();
    });

    // The longest task the main thread ran (from page load on): it names what made a frame late, if one was (spec section 6).
    let longestTask = null;
    try {
      new root.PerformanceObserver((list) => {
        for (const e of list.getEntries()) if (!longestTask || e.duration > longestTask.duration) longestTask = e;
      }).observe({ type: "longtask", buffered: true });
    } catch (e) {}

    let t0 = null; // the first frame, on the page's clock
    let last = null;
    let longestGap = 0;
    let fontPerf = null;
    let s = null;
    let readyAt = null; // on the opening's clock (ms after t0)
    let finishedAt = null;
    let valve = false;
    let handoverPerf = null;
    let landing = null;
    let error = null;
    let ended = false;
    let raf = 0;
    const driven = [];

    const fontLoaded =
      doc.fonts && doc.fonts.load
        ? doc.fonts.load(FONT).then(() => {
            fontPerf = perf.now();
          })
        : Promise.resolve();
    Promise.race([fontLoaded, new Promise((resolve) => root.setTimeout(resolve, FONT_WAIT_MS))])
      .catch(() => {})
      .then(() => {
        raf = root.requestAnimationFrame(first);
      });

    function first(now) {
      try {
        t0 = now;
        last = now;
        build();
        s = begin(plan);
        if (reduced) {
          // No drawing, no light: the word is lit from its first frame, and fades in.
          for (const a of driven) a.currentTime = T;
          finishedAt = 0;
          word.animate([{ opacity: 0 }, { opacity: 1 }], { duration: FADE_IN_MS, easing: "ease-out" });
        }
        word.style.visibility = "visible";
        state.phase = "playing";
        raf = root.requestAnimationFrame(frame);
      } catch (e) {
        fail(e);
      }
    }

    /** Puts the outline on the word's baseline and makes the animations, held at the start of the opening's clock. The
     *  timeline is the approved mockup's (opening-v10.html, build()): every time below is on the default 2 s timeline. */
    function build() {
      const box = word.getBoundingClientRect();
      const W = box.width;
      trace.setAttribute("width", String(W));
      trace.setAttribute("height", String(box.height));
      line.setAttribute("x", "0");
      line.setAttribute("y", String(baselineOf(lit) - box.top));
      // Act one (0 to 0.95 s): the outlines draw; the letters settle into the surface inside them.
      drive(line, [{ strokeDashoffset: 360 }, { strokeDashoffset: 0 }], 950, 0, "cubic-bezier(.45,0,.2,1)");
      drive(pressed, [{ opacity: 0 }, { opacity: 1 }], 400, 550, "ease-out");
      // Act two (0.6 to 2.0 s): the light crosses the letters at an even pace in about 1 s, on a path from -0.18 to 1.18
      // word widths; the lit area's edge is 0.36 of a word wide, so the last letter is fully lit at 2.0 s.
      drive(lit, [{ maskPosition: "84% 0" }, { maskPosition: "16% 0" }], 1400, 600, LIGHT_EASE);
      drive(
        spot,
        [
          { backgroundPosition: "84% 0", opacity: 1 },
          { backgroundPosition: "28.2% 0", opacity: 1, offset: 0.82 },
          { backgroundPosition: "16% 0", opacity: 0 },
        ],
        1400,
        600,
        LIGHT_EASE,
      );
      drive(
        halo,
        [
          { transform: `translateX(${-0.18 * W}px)`, opacity: 0 },
          { opacity: 1, offset: 0.2 },
          { opacity: 1, offset: 0.8 },
          { transform: `translateX(${1.18 * W}px)`, opacity: 0 },
        ],
        1400,
        600,
        LIGHT_EASE,
      );
      // The outline fades as the light passes.
      drive(line, [{ strokeOpacity: 0.9 }, { strokeOpacity: 0 }], 700, 1300, "ease-out");
    }

    function drive(el, frames, duration, delay, easing) {
      const a = el.animate(frames, { duration, delay, easing, fill: "both" });
      a.pause();
      a.currentTime = 0;
      driven.push(a);
    }

    function frame(now) {
      if (ended) return;
      try {
        const gap = now - last;
        last = now;
        if (gap > longestGap) longestGap = gap;
        // Once the hand-over has begun, frames are only counted (the longest gap runs to its end).
        if (handoverPerf === null) {
          const real = now - t0;
          if (readyAt === null && readyPerf !== null) readyAt = Math.max(0, readyPerf - t0);
          if (valveDue(perf.now(), readyAt)) {
            readyAt = real;
            valve = true;
          }
          if (reduced) {
            if (readyAt !== null && real >= reducedHandoverAt(readyAt)) handover();
          } else {
            s = pace(s, { real, dt: gap, readyAt, plan });
            for (const a of driven) a.currentTime = s.it;
            word.style.filter = glowFilter(s.glow, real);
            if (finishedAt === null && s.it >= T) finishedAt = real;
            const at = handoverAt(finishedAt, readyAt);
            if (at !== null && real >= at) handover();
          }
        }
        if (!ended) raf = root.requestAnimationFrame(frame);
      } catch (e) {
        fail(e);
      }
    }

    /** The word glides into the top bar's wordmark while the app fades in around it; with reduced motion, or with no
     *  wordmark to land on (the app's code failed before it drew its top bar), the opening fades out instead. */
    function handover() {
      state.phase = "handover";
      handoverPerf = perf.now();
      opening.style.pointerEvents = "none"; // from here the app takes the clicks
      word.style.filter = "";
      const mark = doc.querySelector(".topbar .wordmark");
      if (reduced || !mark || mark.getBoundingClientRect().width <= 0) return fadeOut();
      const box = word.getBoundingClientRect();
      const from = textRect(lit);
      const to = textRect(mark);
      const g = glide(
        { boxLeft: box.left, boxTop: box.top, textLeft: from.left, textWidth: from.width, baseline: baselineOf(lit) },
        { textLeft: to.left, textWidth: to.width, baseline: baselineOf(mark) },
      );
      const move = word.animate(
        [{ transform: "translate(0px, 0px) scale(1)" }, { transform: `translate(${g.tx}px, ${g.ty}px) scale(${g.s})` }],
        { duration: GLIDE_MS, easing: GLIDE_EASE, fill: "forwards" },
      );
      backdrop.animate([{ opacity: 1 }, { opacity: 0 }], { duration: FADE_MS, delay: FADE_DELAY_MS, easing: "ease-out", fill: "forwards" });
      move.finished.then(
        () => {
          try {
            // Where the word landed against where the wordmark is, read from the live layout (spec section 6).
            const a = textRect(lit);
            const b = textRect(mark);
            landing = Math.max(Math.abs(a.left - b.left), Math.abs(a.width - b.width), Math.abs(baselineOf(lit) - baselineOf(mark)));
          } catch (e) {
            if (error === null) error = message(e);
          }
          swap();
        },
        swap,
      );
    }

    function fadeOut() {
      const a = opening.animate([{ opacity: 1 }, { opacity: 0 }], { duration: FADE_MS, easing: "ease-out", fill: "forwards" });
      a.finished.then(swap, swap);
    }

    /** The real wordmark takes over and the opening leaves the page, in one task: no frame shows both or neither. */
    function swap() {
      if (ended) return;
      ended = true;
      root.cancelAnimationFrame(raf);
      doc.documentElement.classList.remove("opening");
      if (opening.parentNode) opening.parentNode.removeChild(opening);
      const report = makeReport(perf.now());
      state.report = report;
      state.phase = "done";
      // The time this start took, for the next one to plan by (spec section 4); a start the valve ended is not one to learn.
      if (readyPerf !== null && !valve && error === null && readyAt !== null) {
        try {
          root.localStorage.setItem(STORAGE_KEY, String(Math.round(readyAt)));
        } catch (e) {}
      }
      const ipc = root.__TAURI_INTERNALS__;
      if (ipc && typeof ipc.invoke === "function") {
        Promise.resolve()
          .then(() => ipc.invoke("opening_report", { report }))
          .catch((e) => root.console.warn("opening: the report did not reach the host:", e));
      }
    }

    /** A failure anywhere in the opening ends it: it fades out (or, failing that, leaves at once) and says why. */
    function fail(e) {
      if (error === null) error = message(e);
      if (ended) return;
      if (handoverPerf === null) {
        state.phase = "handover";
        handoverPerf = perf.now();
        opening.style.pointerEvents = "none";
      }
      try {
        fadeOut();
      } catch (e2) {
        swap();
      }
    }

    function makeReport(endPerf) {
      const epoch = (p) => (p === null ? null : Math.round(perf.timeOrigin + p));
      return {
        navigation_ms: Math.round(perf.timeOrigin),
        first_frame_ms: epoch(t0),
        ready_ms: valve || readyAt === null || t0 === null ? null : epoch(t0 + readyAt),
        finished_ms: finishedAt === null || t0 === null ? null : epoch(t0 + finishedAt),
        handover_start_ms: epoch(handoverPerf === null ? endPerf : handoverPerf),
        handover_end_ms: epoch(endPerf),
        plan_ms: plan,
        remembered_ms: remembered,
        longest_gap_ms: Math.round(longestGap * 10) / 10,
        longest_task_ms: longestTask ? Math.round(longestTask.duration) : null,
        longest_task_at_ms: longestTask && t0 !== null ? Math.round(longestTask.startTime - t0) : null,
        landing_error_px: landing === null ? null : Math.round(landing * 1000) / 1000,
        font_ms: fontPerf === null ? null : Math.round(fontPerf),
        reduced_motion: reduced,
        valve,
        error,
      };
    }

    /** The y of an element's first line's baseline: a zero-size inline-block sits on it. The probe is gone before the
     *  next frame, so the page never paints it. */
    function baselineOf(el) {
      const probe = doc.createElement("span");
      probe.style.cssText = "display:inline-block;width:0;height:0;vertical-align:baseline";
      el.appendChild(probe);
      const y = probe.getBoundingClientRect().top;
      el.removeChild(probe);
      return y;
    }

    /** The box of an element's text itself (its glyph run), not of the element. */
    function textRect(el) {
      const r = doc.createRange();
      r.selectNodeContents(el);
      return r.getBoundingClientRect();
    }

    function message(e) {
      return e && e.message ? String(e.message) : String(e);
    }
  }

  root.LorrenOpening = {
    T: T,
    PLAN_MIN: PLAN_MIN,
    PLAN_MAX: PLAN_MAX,
    R_MIN: R_MIN,
    R_MAX: R_MAX,
    VALVE_MS: VALVE_MS,
    HANDOVER_DELAY_MS: HANDOVER_DELAY_MS,
    REDUCED_MIN_MS: REDUCED_MIN_MS,
    STORAGE_KEY: STORAGE_KEY,
    parseRemembered: parseRemembered,
    planFor: planFor,
    begin: begin,
    pace: pace,
    handoverAt: handoverAt,
    valveDue: valveDue,
    reducedHandoverAt: reducedHandoverAt,
    glowFilter: glowFilter,
    glide: glide,
  };

  if (root.document && root.document.getElementById("opening")) run(root);
})(this);
