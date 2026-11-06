/** The app's side of the opening (spec 2026-10-23-lorren-opening-design.md section 3): once everything visible is in
 *  place, `window` gets one `lorren:ready` event, which public/opening.js waits for before it hands over. */
export type ReadyPart = "overview" | "chats" | "chat" | "models";
export const READY_EVENT = "lorren:ready";

const ALL: readonly ReadyPart[] = ["overview", "chats", "chat", "models"];
const reported = new Set<ReadyPart>();
let started = false;
let fired = false;
let generation = 0;

function nextFrame(cb: () => void) {
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => cb());
  else setTimeout(cb, 16);
}

/** Idempotent. The fourth distinct part starts the hand-over: fonts, then two frames, then the event (once). */
export function reportReady(part: ReadyPart): void {
  reported.add(part);
  if (started || fired || !ALL.every((p) => reported.has(p))) return;
  started = true;
  const mine = generation;
  const go = () => {
    if (mine !== generation) return;
    nextFrame(() => nextFrame(() => {
      if (mine !== generation || fired) return;
      fired = true;
      window.dispatchEvent(new Event(READY_EVENT));
    }));
  };
  const fonts = typeof document !== "undefined" ? document.fonts : undefined;
  if (fonts?.ready) void fonts.ready.then(go, go);
  else go();
}

/** Tests only: forget the parts, and make a dispatch still pending do nothing. */
export function resetReady(): void {
  reported.clear();
  started = false;
  fired = false;
  generation++;
}
