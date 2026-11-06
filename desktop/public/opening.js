/*
 * Lorren's opening: the launch animation shown before the app bundle is up.
 *
 * This is a CLASSIC script (no imports/exports), loaded with a plain
 * <script src="/opening.js"> before the bundle: the app's CSP allows no inline
 * script, and the opening must run while the bundle is still loading.
 *
 * Spec: docs/superpowers/specs/2026-10-23-lorren-desktop-m1-design.md, section 4
 * (the opening's pacing). Ported from the approved mockup's frame step.
 *
 * `LorrenOpening` is exposed on the global only so the unit tests can reach
 * the pure functions; the page itself never reads it.
 *
 * All times are ms on the opening's clock: `real` counts from its first
 * frame, `it` is the position on the default 2 s timeline.
 */
(function (root) {
  "use strict";

  var T = 2000;
  var PLAN_MIN = 2000;
  var PLAN_MAX = 5000;
  var R_MIN = 0.2;
  var R_MAX = 1.6;
  var VALVE_MS = 12000;
  var HANDOVER_DELAY_MS = 300;
  var REDUCED_MIN_MS = 900;
  var STORAGE_KEY = "lorren.opening.lastStartMs";

  function clamp(x, lo, hi) {
    return Math.min(hi, Math.max(lo, x));
  }

  function round2(x) {
    return Number(x.toFixed(2)).toString();
  }

  function parseRemembered(raw) {
    if (raw === null || raw === undefined || raw === "") return null;
    var n = Number(raw);
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
    var dt = Math.min(64, f.dt);
    var target;
    var ease;
    if (f.readyAt === null) {
      var onPlan = f.real < f.plan - 600;
      target = onPlan ? T / f.plan : Math.min(R_MIN, (T - s.it) / 900);
      ease = onPlan ? 120 : 400;
    } else {
      target = f.real < T ? (T - s.it) / Math.max(1, T - f.real) : R_MAX;
      ease = 200;
    }
    target = clamp(target, 0, R_MAX);
    var rate = s.rate + (target - s.rate) * (1 - Math.exp(-dt / ease));
    var it = Math.min(T, s.it + dt * rate);
    if (f.readyAt === null) it = Math.min(it, T - 0.5);
    var waiting = f.readyAt === null && T - it < 120;
    var glow = waiting ? Math.min(1, s.glow + dt / 500) : Math.max(0, s.glow - dt / 300);
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
    var b = glow * (0.5 + 0.5 * Math.sin(real / 520));
    return (
      "drop-shadow(0 0 " + round2(6 + 10 * b) + "px oklch(0.85 0.07 75 / " + round2(0.1 + 0.22 * b) + "))"
    );
  }

  // CSS px; the transform origin is the word box's top-left.
  function glide(from, to) {
    var s = to.textWidth / from.textWidth;
    return {
      tx: to.textLeft - from.boxLeft - s * (from.textLeft - from.boxLeft),
      ty: to.baseline - from.boxTop - s * (from.baseline - from.boxTop),
      s: s,
    };
  }

  function run(root) {}

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
