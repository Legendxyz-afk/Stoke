import source from "../../public/opening.js?raw";
import { describe, expect, it } from "vitest";

type State = { it: number; rate: number; glow: number };
type Frame = { real: number; it: number; rate: number; glow: number };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const root: Record<string, any> = {};
new Function(source).call(root);
const O = root.LorrenOpening;

/** Frames at `hz` until the hand-over time (or `maxMs`); `readyAt` null = never ready. */
function simulate({ remembered = null, readyAt = null, hz = 60, maxMs = 15000 }: { remembered?: number | null; readyAt?: number | null; hz?: number; maxMs?: number }) {
  const plan = O.planFor(remembered);
  const dt = 1000 / hz;
  let s: State = O.begin(plan);
  let finishedAt: number | null = null;
  const frames: Frame[] = [];
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

function maxRateStep(frames: Frame[]) {
  let m = 0;
  for (let i = 1; i < frames.length; i++) m = Math.max(m, Math.abs(frames[i].rate - frames[i - 1].rate));
  return m;
}

describe("plan", () => {
  it("planFor clamps", () => {
    expect(O.planFor(null)).toBe(2000);
    expect(O.planFor(1200)).toBe(2000);
    expect(O.planFor(4000)).toBe(4000);
    expect(O.planFor(9000)).toBe(5000);
    expect(O.planFor(NaN)).toBe(2000);
    expect(O.planFor(Infinity)).toBe(2000);
  });
  it("parseRemembered accepts only positive finite numbers", () => {
    expect(O.parseRemembered(null)).toBeNull();
    expect(O.parseRemembered("")).toBeNull();
    expect(O.parseRemembered("abc")).toBeNull();
    expect(O.parseRemembered("-1")).toBeNull();
    expect(O.parseRemembered("0")).toBeNull();
    expect(O.parseRemembered("3200")).toBe(3200);
    expect(O.parseRemembered("1e9")).toBe(1e9);
    expect(O.planFor(O.parseRemembered("1e9"))).toBe(5000);
  });
});

describe("pace", () => {
  it("ready at 1 s lands on 2.0 s", () => {
    const r = simulate({ readyAt: 1000 });
    expect(r.finishedAt).toBeGreaterThanOrEqual(1983);
    expect(r.finishedAt).toBeLessThanOrEqual(2017);
    expect(r.handoverAt).toBe(Math.max(r.finishedAt as number, 1000) + 300);
    expect(maxRateStep(r.frames)).toBeLessThanOrEqual(0.25);
  });
  it("a first start ready at 4 s finishes at 4.0 s", () => {
    const r = simulate({ readyAt: 4000 });
    expect(r.finishedAt).toBeGreaterThanOrEqual(3900);
    expect(r.finishedAt).toBeLessThanOrEqual(4100);
    expect(maxRateStep(r.frames)).toBeLessThanOrEqual(0.25);
  });
  it("a learned 4 s start plays at 0.5x and finishes at 4.1 s", () => {
    const r = simulate({ remembered: 4000, readyAt: 4000 });
    for (const f of r.frames.filter((f) => f.real < 3400)) expect(Math.abs(f.rate - 0.5)).toBeLessThan(0.02);
    expect(r.finishedAt).toBeGreaterThanOrEqual(4000);
    expect(r.finishedAt).toBeLessThanOrEqual(4200);
    expect(maxRateStep(r.frames)).toBeLessThanOrEqual(0.25);
  });
  it("ready at 1 s after a learned 4 s plan speeds up to 1.6x and finishes at 2.1 s", () => {
    const r = simulate({ remembered: 4000, readyAt: 1000 });
    const mx = Math.max(...r.frames.map((f) => f.rate));
    expect(mx).toBeGreaterThanOrEqual(1.5);
    expect(mx).toBeLessThanOrEqual(1.6);
    expect(r.finishedAt).toBeGreaterThanOrEqual(2000);
    expect(r.finishedAt).toBeLessThanOrEqual(2150);
    expect(maxRateStep(r.frames)).toBeLessThanOrEqual(0.25);
  });
  it("ready at 7 s finishes at 7.0 s and never before ready", () => {
    const r = simulate({ readyAt: 7000 });
    expect(r.finishedAt).toBeGreaterThanOrEqual(6900);
    expect(r.finishedAt).toBeLessThanOrEqual(7100);
    for (const f of r.frames.filter((f) => f.real < 7000)) expect(f.it).toBeLessThanOrEqual(O.T - 0.5);
    expect(maxRateStep(r.frames)).toBeLessThanOrEqual(0.25);
  });
  it("it never finishes while loading and keeps approaching", () => {
    const r = simulate({ readyAt: null, maxMs: 12000 });
    for (const f of r.frames) expect(f.it).toBeLessThanOrEqual(O.T - 0.5);
    for (let i = 1; i < r.frames.length; i++) expect(r.frames[i].it).toBeGreaterThanOrEqual(r.frames[i - 1].it);
    const at = (ms: number) => r.frames.find((f) => f.real >= ms) as Frame;
    expect(at(3000).it).toBeLessThan(at(6000).it);
    expect(r.handoverAt).toBeNull();
    expect(maxRateStep(r.frames)).toBeLessThanOrEqual(0.25);
  });
  it("never shorter than 2 s", () => {
    const r = simulate({ readyAt: 0 });
    expect(r.finishedAt).toBeGreaterThanOrEqual(1983);
  });
  it("one stalled frame does not jump", () => {
    expect(O.pace(O.begin(2000), { real: 500, dt: 500, readyAt: null, plan: 2000 }).it).toBeLessThanOrEqual(64 * 1.6);
  });
  it("pace does not mutate its input", () => {
    const s = O.begin(2000);
    const copy = { ...s };
    O.pace(s, { real: 16, dt: 16, readyAt: null, plan: 2000 });
    expect(s).toEqual(copy);
  });
  it("30, 60 and 144 Hz land alike", () => {
    for (const hz of [30, 60, 144]) {
      const r = simulate({ readyAt: 1000, hz });
      expect(r.finishedAt).toBeGreaterThanOrEqual(2000 - 1000 / hz);
      expect(r.finishedAt).toBeLessThanOrEqual(2000 + 1000 / hz);
    }
  });
});

describe("glow", () => {
  it("breathes only while waiting with the light across", () => {
    const w = simulate({ readyAt: null, maxMs: 12000 });
    for (const f of w.frames) if (f.glow > 0) expect(O.T - f.it).toBeLessThan(120);
    const first = w.frames.find((f) => f.glow > 0) as Frame;
    const full = w.frames.find((f) => f.glow >= 1) as Frame;
    expect(full).toBeDefined();
    expect(full.real - first.real).toBeLessThanOrEqual(520);
    const r = simulate({ readyAt: 7000 });
    const after = r.frames[r.frames.length - 1]; // the hand-over frame, 300 ms after ready
    expect(after.real).toBeLessThanOrEqual(7320);
    expect(after.glow).toBe(0);
  });
  it("glowFilter", () => {
    expect(O.glowFilter(0, 1234)).toBe("");
    expect(O.glowFilter(1, 0)).toBe("drop-shadow(0 0 11px oklch(0.85 0.07 75 / 0.21))");
  });
});

describe("hand-over", () => {
  it("valveDue", () => {
    expect(O.valveDue(11999, null)).toBe(false);
    expect(O.valveDue(12000, null)).toBe(true);
    expect(O.valveDue(13000, 5000)).toBe(false);
  });
  it("reducedHandoverAt", () => {
    expect(O.reducedHandoverAt(400)).toBe(900);
    expect(O.reducedHandoverAt(3000)).toBe(3000);
  });
});

describe("glide", () => {
  const to = { textLeft: 22, textWidth: 38.9, baseline: 32.2 };
  for (const textLeft of [500, 503.5]) {
    it(`lands the text exactly (text starts at ${textLeft})`, () => {
      const from = { boxLeft: 500, boxTop: 300, textLeft, textWidth: 290.4, baseline: 398.6 };
      const { tx, ty, s } = O.glide(from, to);
      expect(s).toBe(38.9 / 290.4);
      expect(Math.abs(500 + tx + s * (textLeft - 500) - 22)).toBeLessThan(1e-9);
      expect(Math.abs(300 + ty + s * (398.6 - 300) - 32.2)).toBeLessThan(1e-9);
    });
  }
});
