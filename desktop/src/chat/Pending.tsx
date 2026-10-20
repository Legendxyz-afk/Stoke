import { useEffect, useState } from "react";

/** A reply that is waiting for its first token (demo D2): when Send was pressed, and the model the engine is loading for it, if
 *  it is loading one. */
export type Pending = { since: number; loading: string | null };

/** What a pending reply says it is doing, in this order, each for 2.5 s, round and round. They are not the engine's states
 *  (it reports none while it reads the chat); they are there so that a wait of a minute does not read as a hang. */
export const PHASE_WORDS = [
  "thinking",
  "reading the chat",
  "pondering",
  "mulling it over",
  "connecting the dots",
  "weighing it up",
  "piecing it together",
  "working it out",
  "gathering thoughts",
] as const;

const WORD_MS = 2500;

/** After this long with no token, a dim second line says why. */
const HINT_AFTER_MS = 15_000;

/** The first turn after a long chat or a model switch takes the longest: the engine keeps no cache between turns, so the
 *  history is read again (spec §8). */
export const HINT = "long chats take longer: the model re-reads the last ~1,500 tokens of the chat before it answers";

/** How often the status looks at the clock. The seconds change once a second and the word every 2.5 s, so each shows at most
 *  half a tick late. */
const TICK_MS = 500;

export function phaseWord(elapsedMs: number): string {
  return PHASE_WORDS[Math.floor(Math.max(0, elapsedMs) / WORD_MS) % PHASE_WORDS.length];
}

/** `<phase> · <N> s`, with N the whole seconds since Send, shown from the first one on. While the engine is loading the
 *  chat's model the phase is `loading <model>` instead of a word. */
export function statusLine(elapsedMs: number, loading: string | null): string {
  const phase = loading !== null ? `loading ${loading}` : phaseWord(elapsedMs);
  const seconds = Math.floor(Math.max(0, elapsedMs) / 1000);
  return seconds >= 1 ? `${phase} · ${seconds} s` : phase;
}

/** The body of the reply row until its first token arrives, then it is gone (the row shows the text instead). One interval
 *  for the one pending reply, cleared when the component goes away. */
export function PendingStatus({ since, loading }: Pending) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), TICK_MS);
    return () => window.clearInterval(id);
  }, []);
  const elapsed = Math.max(0, now - since);
  return (
    <div className="pending">
      <span className="pending-phase">{statusLine(elapsed, loading)}</span>
      {elapsed >= HINT_AFTER_MS && <span className="pending-hint dim">{HINT}</span>}
    </div>
  );
}
