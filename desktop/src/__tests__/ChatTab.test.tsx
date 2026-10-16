import { createEvent, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { ChatTab } from "../chat/ChatTab";
import { api } from "../ipc/api";
import type { StreamEvent } from "../ipc/bindings/StreamEvent";
import { chat, models, msg, overview, view, withTurn } from "./testApi";

const started: StreamEvent = { kind: "started", message_id: "a3", user_message_id: "u3", dropped_turns: 0, request_id: "req_1" };
const done = { prompt_tokens: 60, completion_tokens: 4, ttft_ms: 212, total_ms: 650, tokens_per_s: 7.0, finish_reason: "stop" };

function type(text: string) {
  fireEvent.change(screen.getByLabelText("message"), { target: { value: text } });
  fireEvent.click(screen.getByText("SEND"));
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.models).mockResolvedValue(models);
  vi.mocked(api.chats).mockResolvedValue([chat]);
  vi.mocked(api.chat).mockResolvedValue(view);
});

describe("ChatTab", () => {
  it("opens the latest chat and renders its reply as Markdown, with its versions and stats", async () => {
    render(<ChatTab overview={overview} />);
    const verdict = await screen.findByText("Verdict: HOLD.");
    expect(verdict.tagName).toBe("STRONG");
    expect(screen.getByText("2/2")).toBeTruthy();
    expect(screen.getByText("480 ms")).toBeTruthy();
  });

  it("streams a reply into the window and then shows its stats", async () => {
    let release!: () => void;
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      onEvent(started);
      onEvent({ kind: "token", text: "Buy " });
      onEvent({ kind: "token", text: "at ₹68,400" });
      await new Promise<void>((r) => (release = r));
      onEvent({ kind: "finished", stats: done });
    });
    vi.mocked(api.chat)
      .mockResolvedValueOnce(view)
      .mockResolvedValueOnce(withTurn(view, "and now?", msg({ id: "a3", role: "assistant", parent_id: "u3", content: "", status: "streaming" })))
      .mockResolvedValue(withTurn(view, "and now?", msg({ id: "a3", role: "assistant", parent_id: "u3", content: "Buy at ₹68,400" })));
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    type("and now?");
    expect(await screen.findByText("Buy at ₹68,400")).toBeTruthy();
    expect(screen.getByText("STOP")).toBeTruthy();
    expect(api.send).toHaveBeenCalledWith("c1", "and now?", expect.any(Function));
    release();
    expect(await screen.findByText("212 ms")).toBeTruthy();
    expect(screen.getByText("SEND")).toBeTruthy();
  });

  it("asks the host to stop the reply", async () => {
    let end!: () => void;
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      onEvent(started);
      await new Promise<void>((r) => (end = r));
      onEvent({ kind: "stopped", stats: done });
    });
    vi.mocked(api.stop).mockImplementation(async () => end());
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    type("go");
    fireEvent.click(await screen.findByText("STOP"));
    await waitFor(() => expect(api.stop).toHaveBeenCalledWith("c1"));
    expect(await screen.findByText("SEND")).toBeTruthy();
  });

  it("shows that it is waiting for the engine before the reply starts", async () => {
    let go!: () => void;
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      await new Promise<void>((r) => (go = r));
      onEvent(started);
      onEvent({ kind: "finished", stats: done });
    });
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    type("hello");
    expect(await screen.findByText(/waiting for the engine/)).toBeTruthy();
    go();
    await waitFor(() => expect(screen.queryByText(/waiting for the engine/)).toBeNull());
  });

  it("shows a refusal that arrives before the reply starts", async () => {
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      onEvent({ kind: "error", message: "the engine refused the request (400 Bad Request)" });
    });
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    type("hello");
    expect((await screen.findByRole("alert")).textContent).toContain("the engine refused the request");
  });

  it("shows the host's refusal of a message longer than the context (Review Focus 5)", async () => {
    vi.mocked(api.send).mockRejectedValue("this message is about 6250 tokens, more than a 4096-token context can take");
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    type("word ".repeat(5000));
    expect((await screen.findByRole("alert")).textContent).toContain("more than a 4096-token context");
    expect(screen.getByText("SEND")).toBeTruthy();
  });

  it("marks where the history stops being sent, counting only the turns the host sends", async () => {
    // The host leaves an error reply out of the history (chat.rs turns()). So the 3 turns not sent are
    // the first question, the second question and the second answer, and the cut falls after that answer.
    // Counting every message, as before, would put it after the second question.
    const one = (m: ReturnType<typeof msg>) => ({ message: m, version: 1, versions: 1, sibling_ids: [m.id] });
    vi.mocked(api.chat).mockResolvedValue({
      ...view,
      messages: [
        one(msg({ id: "q1", role: "user", content: "first question" })),
        one(msg({ id: "e1", role: "assistant", parent_id: "q1", status: "error", content: "" })),
        one(msg({ id: "q2", role: "user", parent_id: "e1", content: "second question" })),
        one(msg({ id: "r2", role: "assistant", parent_id: "q2", content: "second answer" })),
        one(msg({ id: "q3", role: "user", parent_id: "r2", content: "third question" })),
        one(msg({ id: "r3", role: "assistant", parent_id: "q3", content: "third answer" })),
      ],
    });
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      onEvent({ ...started, dropped_turns: 3 } as StreamEvent);
      onEvent({ kind: "token", text: "ok" });
    });
    render(<ChatTab overview={overview} />);
    await screen.findByText("third answer");
    type("next");
    const cut = await screen.findByText(/3 earlier turns not sent/);
    expect(cut.previousElementSibling?.textContent).toContain("second answer");
  });

  it("switches reply versions and regenerates", async () => {
    render(<ChatTab overview={overview} />);
    await screen.findByText("2/2");
    fireEvent.click(screen.getByLabelText("previous version"));
    await waitFor(() => expect(api.selectVersion).toHaveBeenCalledWith("c1", "a1"));
    fireEvent.click(screen.getByText("REGENERATE"));
    await waitFor(() => expect(api.regenerate).toHaveBeenCalledWith("c1", expect.any(Function)));
  });

  it("switches the chat's model, and a model that can't run cannot be picked", async () => {
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    const option = screen.getByRole("option", { name: /qwen2\.5vl:3b/ }) as HTMLOptionElement;
    expect(option.disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("model"), { target: { value: "file:q" } });
    await waitFor(() => expect(api.setChatModel).toHaveBeenCalledWith("c1", "file:q"));
  });

  it("starts a new chat on the default model", async () => {
    vi.mocked(api.newChat).mockResolvedValue({ ...chat, id: "c2", title: "" });
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    fireEvent.click(screen.getByText("+ NEW"));
    await waitFor(() => expect(api.newChat).toHaveBeenCalledWith(null));
  });
});

// Controller rulings R36(b), (c) and (d). Each of these was run against the plan's code first and failed there.
describe("ChatTab, R36", () => {
  const messageBox = () => screen.getByLabelText("message") as HTMLTextAreaElement;
  const modelBox = () => screen.getByLabelText("model") as HTMLSelectElement;

  // (b) A message the host refuses is stored nowhere (api.send rejects), so the composer must give it back.
  it("(b) keeps a message the host refused in the composer, trimmed as it was sent", async () => {
    vi.mocked(api.send).mockRejectedValue("this message is about 6250 tokens, more than a 4096-token context can take");
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    type("  a pasted document  ");
    expect((await screen.findByRole("alert")).textContent).toContain("more than a 4096-token context");
    expect(messageBox().value).toBe("a pasted document");
  });

  it("(b) does not put a refused message over a newer draft", async () => {
    let refuse!: (why: string) => void;
    vi.mocked(api.send).mockImplementation(() => new Promise<void>((_resolve, reject) => (refuse = reject)));
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    type("the first one");
    expect(messageBox().value).toBe(""); // sending clears the composer at once
    fireEvent.change(messageBox(), { target: { value: "typed meanwhile" } });
    refuse("a reply is already being written in this chat");
    expect((await screen.findByRole("alert")).textContent).toContain("already being written");
    expect(messageBox().value).toBe("typed meanwhile");
  });

  // (c) The Enter that confirms an input-method composition must not send the half-written message.
  it("(c) does not send on Enter while an input method is composing", async () => {
    vi.mocked(api.send).mockImplementation(async () => {});
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    fireEvent.change(messageBox(), { target: { value: "こんにちは" } });
    // The fixture must really carry the state under test: an event without isComposing would pass for any build.
    const composing = createEvent.keyDown(messageBox(), { key: "Enter", isComposing: true }) as KeyboardEvent;
    expect(composing.isComposing).toBe(true);
    fireEvent(messageBox(), composing);
    expect(api.send).not.toHaveBeenCalled();
    fireEvent.keyDown(messageBox(), { key: "Enter", shiftKey: true }); // Shift+Enter is a new line
    expect(api.send).not.toHaveBeenCalled();
    fireEvent.keyDown(messageBox(), { key: "Enter" }); // the control: a plain Enter still sends
    await waitFor(() => expect(api.send).toHaveBeenCalledWith("c1", "こんにちは", expect.any(Function)));
  });

  // (d) Switching the model restarts the engine, which interrupts a reply that is still being written.
  it("(d) does not let the model be switched while a reply is streaming in the stored chat", async () => {
    vi.mocked(api.chat).mockResolvedValue(
      withTurn(view, "go", msg({ id: "a3", role: "assistant", parent_id: "u3", content: "Bu", status: "streaming" })),
    );
    render(<ChatTab overview={overview} />);
    await screen.findByText("Bu");
    expect(screen.getByText("STOP")).toBeTruthy(); // the composer already counts this reply as busy
    expect(modelBox().disabled).toBe(true);
  });

  it("(d) does not let the model be switched while this window is writing a reply", async () => {
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      onEvent(started);
      await new Promise<void>(() => {}); // the reply does not end within the test
    });
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    expect(modelBox().disabled).toBe(false); // the control: idle, the picker works
    type("go");
    await screen.findByText("STOP");
    expect(modelBox().disabled).toBe(true);
  });
});
