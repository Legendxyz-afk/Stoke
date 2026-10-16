import type { Message } from "../ipc/bindings/Message";
import type { ReplyStats } from "../ipc/bindings/ReplyStats";
import type { StreamEvent } from "../ipc/bindings/StreamEvent";

/** The reply being written in the open chat, as the UI sees it. */
export type Live = {
  /** From the moment Send is pressed until the reply ends. */
  busy: boolean;
  /** The assistant message, once the engine has accepted the request. */
  messageId: string | null;
  text: string;
  /** How many of the oldest turns were left out by the history budget. */
  dropped: number;
  stats: ReplyStats | null;
  ended: "finished" | "stopped" | "interrupted" | "error" | null;
  error: string | null;
};

export const idle: Live = { busy: false, messageId: null, text: "", dropped: 0, stats: null, ended: null, error: null };

export function begin(): Live {
  return { ...idle, busy: true };
}

export function applyEvent(s: Live, e: StreamEvent): Live {
  switch (e.kind) {
    case "started":
      return { ...s, busy: true, messageId: e.message_id, text: "", dropped: e.dropped_turns };
    case "token":
      return { ...s, text: s.text + e.text };
    case "finished":
      return { ...s, busy: false, stats: e.stats, ended: "finished" };
    case "stopped":
      return { ...s, busy: false, stats: e.stats, ended: "stopped" };
    case "interrupted":
      return { ...s, busy: false, ended: "interrupted", error: e.reason };
    case "error":
      return { ...s, busy: false, ended: "error", error: e.message };
  }
}

/** A saved reply's numbers in the shape of a live one. Speed counts the tokens after the first. */
export function statsOf(m: Message): ReplyStats {
  const s = m.stats;
  const gen = s.total_ms != null && s.ttft_ms != null ? s.total_ms - s.ttft_ms : null;
  const n = s.completion_tokens;
  return {
    prompt_tokens: s.prompt_tokens,
    completion_tokens: n,
    ttft_ms: s.ttft_ms,
    total_ms: s.total_ms ?? 0,
    tokens_per_s: n != null && n > 1 && gen != null && gen > 0 ? (n - 1) / (gen / 1000) : null,
    finish_reason: null,
  };
}
