import { act, createEvent, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { ChatTab } from "../chat/ChatTab";
import { api } from "../ipc/api";
import type { ChatView } from "../ipc/bindings/ChatView";
import type { StreamEvent } from "../ipc/bindings/StreamEvent";
import { captureHostErrors, chat, models, msg, overview, view, withTurn } from "./testApi";

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

// ChatTab logs "<call> failed:" for a rejected host call (R57). Those lines go to this spy, not to stderr; anything else
// that reaches console.error (an act() warning, say) still reaches stderr.
let hostErrors: MockInstance<typeof console.error>;
beforeEach(() => {
  hostErrors = captureHostErrors();
});
afterEach(() => {
  hostErrors.mockRestore();
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
    // R57: a refused send is a failed host call like any other, and is logged once
    expect(hostErrors.mock.calls).toEqual([
      ["send failed:", "this message is about 6250 tokens, more than a 4096-token context can take"],
    ]);
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

// Review fix round 1: rulings R56 to R60 and the review's items 6 to 8. Each test was run against the code of commit
// d21e826 first. A test that passes there is a pin, and the mutation that proves it is named in the report.
const settle = () => new Promise((resolve) => setTimeout(resolve, 20));

/** Node's unhandledRejection events during a test. Vitest also fails the whole run on one; this makes the test say so. */
function watchUnhandledRejections() {
  type Emitter = {
    on(event: "unhandledRejection", listener: (reason: unknown) => void): void;
    off(event: "unhandledRejection", listener: (reason: unknown) => void): void;
  };
  const proc = (globalThis as unknown as { process: Emitter }).process;
  const seen: unknown[] = [];
  const listener = (reason: unknown) => {
    seen.push(reason);
  };
  proc.on("unhandledRejection", listener);
  return { seen, stop: () => proc.off("unhandledRejection", listener) };
}

describe("ChatTab, fix round 1", () => {
  // Nothing an earlier test left on the mocks (a rejection, a queue of one-off results) may reach these tests.
  beforeEach(() => {
    vi.mocked(api.models).mockReset().mockResolvedValue(models);
    vi.mocked(api.chats).mockReset().mockResolvedValue([chat]);
    vi.mocked(api.chat).mockReset().mockResolvedValue(view);
    vi.mocked(api.send).mockReset();
    vi.mocked(api.regenerate).mockReset();
    vi.mocked(api.stop).mockReset();
    vi.mocked(api.selectVersion).mockReset();
    vi.mocked(api.setChatModel).mockReset();
    vi.mocked(api.newChat).mockReset();
  });

  describe("R57: a failed host call is shown in the notice and logged once", () => {
    it("models(): shown, logged, and not wiped when the first chat opens", async () => {
      vi.mocked(api.models).mockRejectedValue("the model list could not be read");
      render(<ChatTab overview={overview} />);
      expect((await screen.findByRole("alert")).textContent).toContain("the model list could not be read");
      await screen.findByText("Verdict: HOLD.");
      expect(screen.getByRole("alert").textContent).toContain("the model list could not be read");
      expect(hostErrors.mock.calls).toEqual([["models failed:", "the model list could not be read"]]);
    });

    it("selectVersion(): shown, logged, nothing switched, and no unhandled rejection", async () => {
      const unhandled = watchUnhandledRejections();
      try {
        vi.mocked(api.selectVersion).mockRejectedValue("there is no such message");
        render(<ChatTab overview={overview} />);
        await screen.findByText("2/2");
        fireEvent.click(screen.getByLabelText("previous version"));
        expect((await screen.findByRole("alert")).textContent).toContain("there is no such message");
        expect(hostErrors.mock.calls).toEqual([["selectVersion failed:", "there is no such message"]]);
        expect(screen.getByText("2/2")).toBeTruthy();
        await settle(); // Node reports an unhandled rejection once the microtasks have drained
        expect(unhandled.seen).toEqual([]);
      } finally {
        unhandled.stop();
      }
    });

    it("stop(): shown and logged", async () => {
      vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
        onEvent(started);
        await new Promise<void>(() => {}); // the reply goes on for the whole test
      });
      vi.mocked(api.stop).mockRejectedValue("the host did not answer");
      render(<ChatTab overview={overview} />);
      await screen.findByText("Verdict: HOLD.");
      type("go");
      fireEvent.click(await screen.findByText("STOP"));
      expect((await screen.findByRole("alert")).textContent).toContain("the host did not answer");
      expect(hostErrors.mock.calls).toEqual([["stop failed:", "the host did not answer"]]);
    });

    it("the chat reload that follows an event: shown and logged", async () => {
      vi.mocked(api.chat).mockResolvedValueOnce(view).mockRejectedValue("the store is busy");
      vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
        onEvent(started);
      });
      render(<ChatTab overview={overview} />);
      await screen.findByText("Verdict: HOLD.");
      type("go");
      expect((await screen.findByRole("alert")).textContent).toContain("the store is busy");
      expect(hostErrors.mock.calls).toEqual([["chat failed:", "the store is busy"]]);
    });

    it("the chat list refresh that follows an event: shown and logged", async () => {
      vi.mocked(api.chats).mockResolvedValueOnce([chat]).mockRejectedValue("the chat list is unavailable");
      vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
        onEvent(started);
        onEvent({ kind: "finished", stats: done });
      });
      render(<ChatTab overview={overview} />);
      await screen.findByText("Verdict: HOLD.");
      type("go");
      expect((await screen.findByRole("alert")).textContent).toContain("the chat list is unavailable");
      expect(hostErrors.mock.calls).toEqual([["chats failed:", "the chat list is unavailable"]]);
    });

    it("switching the model and starting a chat were shown already; they are logged too", async () => {
      vi.mocked(api.setChatModel).mockRejectedValue("the model cannot be started");
      vi.mocked(api.newChat).mockRejectedValue("the store is read-only");
      render(<ChatTab overview={overview} />);
      await screen.findByText("Verdict: HOLD.");
      fireEvent.change(screen.getByLabelText("model"), { target: { value: "file:q" } });
      expect((await screen.findByRole("alert")).textContent).toContain("the model cannot be started");
      fireEvent.click(screen.getByText("+ NEW"));
      await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("the store is read-only"));
      expect(hostErrors.mock.calls).toEqual([
        ["setChatModel failed:", "the model cannot be started"],
        ["newChat failed:", "the store is read-only"],
      ]);
    });
  });

  describe("R56: the transcript follows a streaming reply only while the reader is at the bottom", () => {
    /** A chat with a reply that goes on for the whole test, in a transcript that is 1000 px long in a 100 px window
     *  (jsdom has no layout, so the geometry and the scroll position are given). */
    async function openStreamingChat() {
      let say!: (text: string) => void;
      vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
        onEvent(started);
        say = (text) => onEvent({ kind: "token", text });
        await new Promise<void>(() => {});
      });
      vi.mocked(api.chat)
        .mockResolvedValueOnce(view)
        .mockResolvedValue(withTurn(view, "go", msg({ id: "a3", role: "assistant", parent_id: "u3", content: "", status: "streaming" })));
      const { rerender } = render(<ChatTab overview={overview} active />);
      await screen.findByText("Verdict: HOLD.");
      const box = screen.getByLabelText("messages");
      const at = { top: 0 };
      Object.defineProperty(box, "scrollHeight", { configurable: true, value: 1000 });
      Object.defineProperty(box, "clientHeight", { configurable: true, value: 100 });
      Object.defineProperty(box, "scrollTop", { configurable: true, get: () => at.top, set: (v: number) => (at.top = v) });
      type("go");
      await screen.findByText("go"); // the turn is in the transcript
      return {
        at,
        /** The reader scrolls: the browser fires a scroll event. */
        scrollTo: (top: number) => {
          at.top = top;
          fireEvent.scroll(box);
        },
        say: (text: string) => say(text),
        show: (shown: boolean) => rerender(<ChatTab overview={overview} active={shown} />),
      };
    }

    it("leaves a reader who scrolled up where they are, and follows one at the bottom", async () => {
      const t = await openStreamingChat();
      t.scrollTo(0); // 1000 - 0 - 100 = 900 px from the end
      t.say("Hello");
      await screen.findByText("Hello");
      expect(t.at.top).toBe(0);

      t.scrollTo(900); // 1000 - 900 - 100 = 0 px from the end
      t.say(" and more");
      await screen.findByText("Hello and more");
      expect(t.at.top).toBe(1000);
    });

    it("leaves the transcript alone while the tab is hidden, and follows to the end when it is shown again", async () => {
      const t = await openStreamingChat();
      t.scrollTo(900); // the reader is at the end
      t.show(false);
      t.at.top = 123; // a hidden element has no layout; whatever it reports is not touched
      t.say("Arrived while hidden");
      await screen.findByText("Arrived while hidden");
      expect(t.at.top).toBe(123);
      t.show(true); // text came in while they were away: they come back to the end of it
      expect(t.at.top).toBe(1000);
    });

    it("does not pull a reader who left in the middle of the transcript down when the tab is shown again", async () => {
      const t = await openStreamingChat();
      t.scrollTo(300);
      t.show(false);
      t.say("Arrived while hidden");
      await screen.findByText("Arrived while hidden");
      t.show(true);
      expect(t.at.top).toBe(300);
    });
  });

  describe("R59: reloads keep their order", () => {
    it("drops the result of an older reload that lands after a newer one", async () => {
      render(<ChatTab overview={overview} />);
      await screen.findByText("2/2");
      const saying = (text: string): ChatView => ({
        ...view,
        messages: [view.messages[0], { ...view.messages[1], message: { ...view.messages[1].message, content: text } }],
      });
      let first!: (v: ChatView) => void;
      let second!: (v: ChatView) => void;
      vi.mocked(api.chat)
        .mockImplementationOnce(() => new Promise<ChatView>((resolve) => (first = resolve)))
        .mockImplementationOnce(() => new Promise<ChatView>((resolve) => (second = resolve)));
      // each version switch reloads the chat: two reloads are in flight at once
      fireEvent.click(screen.getByLabelText("previous version"));
      fireEvent.click(screen.getByLabelText("previous version"));
      await waitFor(() => expect(api.chat).toHaveBeenCalledTimes(3)); // the open's, and these two
      await act(async () => second(saying("the newer answer")));
      expect(screen.getByText("the newer answer")).toBeTruthy();
      await act(async () => first(saying("the older answer"))); // the first call answers last
      expect(screen.queryByText("the older answer")).toBeNull();
      expect(screen.getByText("the newer answer")).toBeTruthy();
    });
  });

  describe("R60: the tab refreshes what the host can change while it is hidden, each time it is shown", () => {
    const qwen = () => screen.getByRole("option", { name: /Qwen3-4B/ }) as HTMLOptionElement;

    it("fetches the models again, and the chat list only if its first load failed", async () => {
      const { rerender } = render(<ChatTab overview={overview} active />);
      await screen.findByText("Verdict: HOLD.");
      expect(api.models).toHaveBeenCalledTimes(1);
      expect(api.chats).toHaveBeenCalledTimes(1);
      expect(qwen().disabled).toBe(false);
      // the host has since found that it cannot run this one
      vi.mocked(api.models).mockResolvedValue(
        models.map((m) => (m.id === "file:q" ? { ...m, runnable: false, reason: "the model file is gone" } : m)),
      );
      rerender(<ChatTab overview={overview} active={false} />);
      expect(api.models).toHaveBeenCalledTimes(1); // hiding it asks for nothing
      rerender(<ChatTab overview={overview} active />);
      await waitFor(() => expect(qwen().disabled).toBe(true));
      expect(api.models).toHaveBeenCalledTimes(2);
      expect(api.chats).toHaveBeenCalledTimes(1);
    });

    it("loads the chat list again when its first load failed, and the failure notice goes with it", async () => {
      vi.mocked(api.chats).mockRejectedValueOnce("the store is locked");
      const { rerender } = render(<ChatTab overview={overview} active />);
      expect((await screen.findByRole("alert")).textContent).toContain("the store is locked");
      expect(screen.queryByText("Verdict: HOLD.")).toBeNull();
      rerender(<ChatTab overview={overview} active={false} />);
      rerender(<ChatTab overview={overview} active />);
      expect(await screen.findByText("Verdict: HOLD.")).toBeTruthy();
      expect(screen.queryByRole("alert")).toBeNull();
      expect(api.chats).toHaveBeenCalledTimes(2);
      expect(hostErrors.mock.calls).toEqual([["chats failed:", "the store is locked"]]);
    });

    it("clears a models() failure when the next load succeeds", async () => {
      vi.mocked(api.models).mockRejectedValueOnce("the model list could not be read").mockResolvedValue(models);
      const { rerender } = render(<ChatTab overview={overview} active />);
      expect((await screen.findByRole("alert")).textContent).toContain("the model list could not be read");
      await screen.findByText("Verdict: HOLD.");
      rerender(<ChatTab overview={overview} active={false} />);
      rerender(<ChatTab overview={overview} active />);
      await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
      expect(screen.getByRole("option", { name: /nexus/ })).toBeTruthy(); // the list is there now
    });
  });

  describe("item 8: the LAST REPLY panel has no total for a reply that never got one", () => {
    const totalShown = () => screen.getByText("total").nextElementSibling?.textContent;

    it("shows a dash, not 0 ms, while the saved reply is still streaming", async () => {
      vi.mocked(api.chat).mockResolvedValue(
        withTurn(view, "go", msg({ id: "a3", role: "assistant", parent_id: "u3", content: "Bu", status: "streaming" })),
      );
      render(<ChatTab overview={overview} />);
      await screen.findByText("Bu");
      expect(totalShown()).toBe("—");
      expect(screen.queryByText("0 ms")).toBeNull();
    });

    it("shows a dash, not 0 ms, for a reply that ended early, and still shows the figures it has", async () => {
      vi.mocked(api.chat).mockResolvedValue(
        withTurn(view, "go", msg({
          id: "a3", role: "assistant", parent_id: "u3", content: "Half an ans", status: "interrupted",
          stats: { prompt_tokens: 41, completion_tokens: 3, ttft_ms: 300, total_ms: null, error: "the engine stopped responding" },
        })),
      );
      render(<ChatTab overview={overview} />);
      await screen.findByText("Half an ans");
      expect(totalShown()).toBe("—");
      expect(screen.queryByText("0 ms")).toBeNull();
      expect(screen.getByText("300 ms")).toBeTruthy();
    });
  });
});

// R61: the transcript followed only the live text of a streaming reply, so a chat opened at its TOP, a chat clicked in the
// list stayed wherever the one before had been left, and the reload that ends a reply (the stored reply replaces the live
// text, and may be longer) did not re-follow. It now follows a change of the chat view as well, under the same rule as
// R56: only while the reader is at the end. jsdom has no layout, so the box is given a geometry and a scroll position the
// test can read and move; `height` can grow, as the transcript does when the stored reply lands.
describe("ChatTab, R61: the transcript follows a change of the chat while the reader is at its end", () => {
  beforeEach(() => {
    vi.mocked(api.models).mockReset().mockResolvedValue(models);
    vi.mocked(api.chats).mockReset().mockResolvedValue([chat]);
    // A new object on every call, as the host's answer is: the same one twice would not be a change for React.
    vi.mocked(api.chat).mockReset().mockImplementation(async () => ({ ...view }));
    vi.mocked(api.send).mockReset();
    vi.mocked(api.selectVersion).mockReset().mockResolvedValue(undefined);
  });

  /** The messages box exists from the first render, before any chat has loaded: give it its geometry then. */
  function giveGeometry(height: number) {
    const box = screen.getByLabelText("messages");
    const at = { top: 0, height };
    Object.defineProperty(box, "scrollHeight", { configurable: true, get: () => at.height });
    Object.defineProperty(box, "clientHeight", { configurable: true, value: 100 });
    Object.defineProperty(box, "scrollTop", { configurable: true, get: () => at.top, set: (v: number) => (at.top = v) });
    return {
      at,
      /** The reader scrolls: the browser moves the position and fires a scroll event. */
      scrollTo: (top: number) => {
        at.top = top;
        fireEvent.scroll(box);
      },
    };
  }

  it("opens a chat at the end of its transcript, not at its top", async () => {
    render(<ChatTab overview={overview} active />);
    const { at } = giveGeometry(1000);
    await screen.findByText("Verdict: HOLD.");
    expect(at.top).toBe(1000);
  });

  it("opens the chat that is clicked in the list at its end, whatever the reader had done in the one before", async () => {
    const second = { ...chat, id: "c2", title: "Second chat" };
    vi.mocked(api.chats).mockResolvedValue([chat, second]);
    vi.mocked(api.chat).mockImplementation(async (id) => ({ ...view, chat: id === "c2" ? second : chat }));
    render(<ChatTab overview={overview} active />);
    const t = giveGeometry(1000);
    await screen.findByText("Verdict: HOLD.");
    t.scrollTo(100); // the reader is partway up the first chat
    t.at.height = 1400; // the second one is longer
    fireEvent.click(screen.getByText("Second chat"));
    await waitFor(() => expect(t.at.top).toBe(1400));
  });

  it("re-follows when the reload that ends a reply replaces the live text with a longer stored reply", async () => {
    render(<ChatTab overview={overview} active />);
    const t = giveGeometry(1000);
    await screen.findByText("Verdict: HOLD.");
    let finish!: () => void;
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      onEvent(started);
      onEvent({ kind: "token", text: "Hello" });
      await new Promise<void>((resolve) => (finish = resolve));
      onEvent({ kind: "finished", stats: done });
    });
    vi.mocked(api.chat).mockImplementation(async () =>
      withTurn(view, "go", msg({ id: "a3", role: "assistant", parent_id: "u3", content: "Hello, and the rest of the stored reply" })),
    );
    type("go");
    await screen.findByText("Hello"); // the live text is in; the reader is at the end (nobody scrolled)
    t.at.height = 1600; // the stored reply is longer than what streamed
    finish();
    await waitFor(() => expect(t.at.top).toBe(1600));
  });

  it("does not pull a reader who scrolled up down when the chat is reloaded", async () => {
    render(<ChatTab overview={overview} active />);
    const t = giveGeometry(1000);
    await screen.findByText("2/2");
    expect(t.at.top).toBe(1000);
    t.scrollTo(200); // 1000 - 200 - 100 = 700 px from the end
    t.at.height = 1300; // the reload brings more
    fireEvent.click(screen.getByLabelText("previous version")); // a version switch reloads the chat
    await waitFor(() => expect(api.chat).toHaveBeenCalledTimes(2));
    await act(async () => {}); // the answer is applied
    expect(t.at.top).toBe(200);
  });
});
