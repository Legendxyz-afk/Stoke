import { describe, expect, it } from "vitest";
import { applyEvent, begin, idle, statsOf } from "../chat/stream";
import { msg } from "./testApi";

const stats = { prompt_tokens: 50, completion_tokens: 4, ttft_ms: 212, total_ms: 600, tokens_per_s: 7.7, finish_reason: "stop" };

describe("applyEvent", () => {
  it("collects a reply from started to finished", () => {
    let s = begin();
    s = applyEvent(s, { kind: "started", message_id: "a3", user_message_id: "u3", dropped_turns: 2, request_id: "req_1" });
    s = applyEvent(s, { kind: "token", text: "Buy " });
    s = applyEvent(s, { kind: "token", text: "at ₹68,400" });
    expect(s).toMatchObject({ busy: true, messageId: "a3", text: "Buy at ₹68,400", dropped: 2 });
    s = applyEvent(s, { kind: "finished", stats });
    expect(s).toMatchObject({ busy: false, ended: "finished", stats });
  });

  // Demo D2: `queued` comes first, as soon as the host has stored the message and the reply's placeholder; `started` only
  // when the engine has taken the request. The time Send was pressed is kept through both, for the pending status.
  it("knows the reply's message from queued on, and keeps when Send was pressed", () => {
    let s = begin(1000);
    expect(s).toMatchObject({ busy: true, sentAt: 1000, messageId: null });
    s = applyEvent(s, { kind: "queued", message_id: "a3", user_message_id: "u3" });
    expect(s).toMatchObject({ busy: true, sentAt: 1000, messageId: "a3", text: "" });
    s = applyEvent(s, { kind: "started", message_id: "a3", user_message_id: "u3", dropped_turns: 2, request_id: "req_1" });
    expect(s).toMatchObject({ busy: true, sentAt: 1000, messageId: "a3", dropped: 2 });
    s = applyEvent(s, { kind: "finished", stats });
    expect(s).toMatchObject({ busy: false, ended: "finished" });
    expect(idle.sentAt).toBeNull();
  });

  it("keeps a refusal that arrives before the reply starts", () => {
    const s = applyEvent(begin(), { kind: "error", message: "the engine refused the request (400 Bad Request)" });
    expect(s).toMatchObject({ busy: false, messageId: null, ended: "error", error: "the engine refused the request (400 Bad Request)" });
  });

  it("ends on stopped and interrupted", () => {
    expect(applyEvent(begin(), { kind: "stopped", stats }).ended).toBe("stopped");
    const i = applyEvent(begin(), { kind: "interrupted", reason: "the engine stopped responding" });
    expect(i).toMatchObject({ busy: false, ended: "interrupted", error: "the engine stopped responding" });
    expect(idle.busy).toBe(false);
  });
});

describe("statsOf", () => {
  it("derives the decode speed from a saved reply", () => {
    const m = msg({ id: "a", role: "assistant", content: "x",
                    stats: { prompt_tokens: 41, completion_tokens: 21, ttft_ms: 500, total_ms: 1500, error: null } });
    expect(statsOf(m)).toMatchObject({ prompt_tokens: 41, completion_tokens: 21, ttft_ms: 500, total_ms: 1500, tokens_per_s: 20 });
  });

  it("has no total for a reply that never got one, not a total of 0", () => {
    const streaming = msg({ id: "a", role: "assistant", content: "", status: "streaming" });
    expect(statsOf(streaming)).toMatchObject({ prompt_tokens: null, ttft_ms: null, total_ms: null, tokens_per_s: null });
    const early = msg({ id: "b", role: "assistant", content: "half", status: "interrupted",
                        stats: { prompt_tokens: 41, completion_tokens: 3, ttft_ms: 300, total_ms: null, error: "the engine stopped responding" } });
    expect(statsOf(early)).toMatchObject({ prompt_tokens: 41, completion_tokens: 3, ttft_ms: 300, total_ms: null, tokens_per_s: null });
  });
});
