import { act, createEvent, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { ChatTab, FLUSH_MS } from "../chat/ChatTab";
import { api } from "../ipc/api";
import type { Chat } from "../ipc/bindings/Chat";
import type { ChatView } from "../ipc/bindings/ChatView";
import type { Overview } from "../ipc/bindings/Overview";
import type { StreamEvent } from "../ipc/bindings/StreamEvent";
import { captureHostErrors, chat, model, models, msg, overview, view, withTurn } from "./testApi";

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

// ---------------------------------------------------------------------------------------------------------------------
// Demo wave (2026-10-20), group 2: D1 (type first), D2 (the pending reply), D4 (the picker marks the default).
// Each test was run RED first, against the code of commit 3f7fd2a.
// ---------------------------------------------------------------------------------------------------------------------

const queuedFor = (messageId: string, userMessageId: string): StreamEvent => ({ kind: "queued", message_id: messageId, user_message_id: userMessageId });
const one = (m: ReturnType<typeof msg>) => ({ message: m, version: 1, versions: 1, sibling_ids: [m.id] });

describe("ChatTab, demo D4: the picker marks the default model", () => {
  const optionTexts = () => within(screen.getByLabelText("model")).getAllByRole("option").map((o) => o.textContent);

  it("writes (default) after the default model's name, and (can't run) after one that cannot run", async () => {
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    expect(optionTexts()).toEqual(["Qwen3-4B-Instruct-2507-Q4_K_M (default)", "nexus", "qwen2.5vl:3b (can't run)"]);
  });

  it("says both for a default that the engine refused", async () => {
    vi.mocked(api.models).mockResolvedValue(models.map((m) => (m.is_default ? { ...m, runnable: false, reason: "the engine refused it" } : m)));
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    expect(optionTexts()[0]).toBe("Qwen3-4B-Instruct-2507-Q4_K_M (default) (can't run)");
  });
});

describe("ChatTab, demo D1: type first, the chat is made on Send", () => {
  const messageBox = () => screen.getByLabelText("message") as HTMLTextAreaElement;
  const modelBox = () => screen.getByLabelText("model") as HTMLSelectElement;
  /** The chat the host makes for the first message. */
  const fresh: Chat = { ...chat, id: "c9", title: "", model_id: "file:q", active_leaf_id: null };
  /** A library whose FIRST row is not the default: the demo's picker showed that first row. */
  const llama = model({ id: "file:l", name: "Llama-3.2-3B-Instruct-Q4_K_M" });
  const library = [llama, ...models];
  const storedFresh = (text: string): ChatView => ({
    chat: { ...fresh, title: text, active_leaf_id: "a1" },
    messages: [
      one(msg({ id: "u1", role: "user", chat_id: "c9", content: text })),
      one(msg({ id: "a1", role: "assistant", chat_id: "c9", parent_id: "u1", content: "", status: "streaming", model_id: "file:q" })),
    ],
  });
  /** The host as it answers a send: a moment after the call, when the chat is open, it says queued, started, and a token. */
  const answerWith = (reply: string) =>
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      await Promise.resolve();
      onEvent(queuedFor("a1", "u1"));
      onEvent({ ...started, message_id: "a1", user_message_id: "u1" });
      onEvent({ kind: "token", text: reply });
      await new Promise<void>(() => {}); // the reply goes on for the whole test
    });
  const openEmpty = async () => {
    render(<ChatTab overview={overview} />);
    await screen.findByText("Type a message to start a chat.");
    await waitFor(() => expect(modelBox().value).toBe("file:q")); // the models have landed
  };

  beforeEach(() => {
    vi.mocked(api.models).mockReset().mockResolvedValue(library);
    vi.mocked(api.chats).mockReset().mockResolvedValueOnce([]).mockResolvedValue([{ ...fresh, title: "hello" }]);
    vi.mocked(api.chat).mockReset().mockImplementation(async () => storedFresh("hello"));
    vi.mocked(api.newChat).mockReset().mockResolvedValue(fresh);
    vi.mocked(api.send).mockReset().mockImplementation(async () => {});
    vi.mocked(api.setChatModel).mockReset();
    vi.mocked(api.stop).mockReset();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("opens with the composer enabled, a prompt to type, and the model a new chat would get in the picker", async () => {
    await openEmpty();
    expect(screen.queryByText(/Start a chat with/)).toBeNull();
    expect(messageBox().disabled).toBe(false);
    expect(modelBox().disabled).toBe(false);
    expect(modelBox().value).toBe("file:q"); // the default, not the first row (Llama)
    expect(screen.getByText("engine").nextElementSibling?.textContent).toBe("—");
    expect((screen.getByText("SEND") as HTMLButtonElement).disabled).toBe(true); // nothing typed yet
    fireEvent.change(messageBox(), { target: { value: "hello" } });
    expect((screen.getByText("SEND") as HTMLButtonElement).disabled).toBe(false);
  });

  it("offers the first model that can run when the default cannot", async () => {
    vi.mocked(api.models).mockResolvedValue([
      model({ id: "file:x", name: "broken", runnable: false, reason: "the engine refused it" }),
      ...models.map((m) => (m.is_default ? { ...m, runnable: false, reason: "the engine refused it" } : m)),
    ]);
    render(<ChatTab overview={overview} />);
    await screen.findByText("Type a message to start a chat.");
    await waitFor(() => expect(modelBox().value).toBe("file:n")); // nexus: the first row that can run
  });

  it("makes the chat on Enter with the picker's model, sends the text in it, and the reply streams in", async () => {
    answerWith("Buy at ₹68,400");
    await openEmpty();
    fireEvent.change(messageBox(), { target: { value: "hello" } });
    fireEvent.keyDown(messageBox(), { key: "Enter" });
    expect(await screen.findByText("Buy at ₹68,400")).toBeTruthy();
    expect(api.newChat).toHaveBeenCalledTimes(1);
    expect(api.newChat).toHaveBeenCalledWith("file:q");
    expect(api.send).toHaveBeenCalledTimes(1);
    expect(api.send).toHaveBeenCalledWith("c9", "hello", expect.any(Function));
    // The chat is made, the list refreshed, and only then is the text sent.
    const at = (f: unknown) => vi.mocked(f as typeof api.send).mock.invocationCallOrder[0];
    expect(at(api.newChat)).toBeLessThan(at(api.send));
    expect(vi.mocked(api.chats).mock.invocationCallOrder[1]).toBeGreaterThan(at(api.newChat));
    expect(vi.mocked(api.chats).mock.invocationCallOrder[1]).toBeLessThan(at(api.send));
    // The open-chat effect resets the live reply; the send has to come after it, or the reply that has started is wiped.
    expect(screen.getByText("STOP")).toBeTruthy();
    expect(messageBox().value).toBe("");
    expect(screen.getByText("hello", { selector: ".msg-user" })).toBeTruthy(); // the user's message, from the stored chat
  });

  it("makes the chat on the SEND button too", async () => {
    answerWith("Hello there");
    await openEmpty();
    fireEvent.change(messageBox(), { target: { value: "hello" } });
    fireEvent.click(screen.getByText("SEND"));
    expect(await screen.findByText("Hello there")).toBeTruthy();
    expect(api.newChat).toHaveBeenCalledTimes(1);
    expect(api.send).toHaveBeenCalledWith("c9", "hello", expect.any(Function));
  });

  it("makes the chat on the model that was picked first, and the pick alone calls no host function", async () => {
    answerWith("ok");
    vi.mocked(api.newChat).mockResolvedValue({ ...fresh, model_id: "file:n" });
    await openEmpty();
    fireEvent.change(modelBox(), { target: { value: "file:n" } });
    expect(modelBox().value).toBe("file:n");
    expect(api.setChatModel).not.toHaveBeenCalled();
    expect(api.newChat).not.toHaveBeenCalled();
    expect(screen.getByText("engine").nextElementSibling?.textContent).toBe("—"); // still no chat
    fireEvent.change(messageBox(), { target: { value: "hello" } });
    fireEvent.keyDown(messageBox(), { key: "Enter" });
    await waitFor(() => expect(api.send).toHaveBeenCalled());
    expect(api.newChat).toHaveBeenCalledTimes(1);
    expect(api.newChat).toHaveBeenCalledWith("file:n");
    expect(api.setChatModel).not.toHaveBeenCalled();
  });

  it("puts the text back when the host refuses the send, and shows why, as for a chat that was already open", async () => {
    vi.mocked(api.send).mockRejectedValue("this message is about 6250 tokens, more than a 4096-token context can take");
    vi.mocked(api.chat).mockImplementation(async () => ({ chat: fresh, messages: [] })); // a refused message is stored nowhere
    await openEmpty();
    fireEvent.change(messageBox(), { target: { value: "  a pasted document  " } });
    fireEvent.keyDown(messageBox(), { key: "Enter" });
    expect((await screen.findByRole("alert")).textContent).toContain("more than a 4096-token context");
    expect(messageBox().value).toBe("a pasted document");
    expect(hostErrors.mock.calls).toEqual([["send failed:", "this message is about 6250 tokens, more than a 4096-token context can take"]]);
    expect(screen.getByText("SEND")).toBeTruthy(); // not stuck on STOP
  });

  it("puts the text back, and sends nothing, when the chat cannot be made", async () => {
    vi.mocked(api.newChat).mockRejectedValue("there is no runnable model in the library");
    await openEmpty();
    fireEvent.change(messageBox(), { target: { value: "hello" } });
    fireEvent.keyDown(messageBox(), { key: "Enter" });
    expect((await screen.findByRole("alert")).textContent).toContain("there is no runnable model in the library");
    expect(messageBox().value).toBe("hello");
    expect(api.send).not.toHaveBeenCalled();
    expect(hostErrors.mock.calls).toEqual([["newChat failed:", "there is no runnable model in the library"]]);
    expect(screen.getByText("SEND")).toBeTruthy();
  });

  it("does not make a chat on the Enter that confirms an input-method composition", async () => {
    await openEmpty();
    fireEvent.change(messageBox(), { target: { value: "こんにちは" } });
    const composing = createEvent.keyDown(messageBox(), { key: "Enter", isComposing: true }) as KeyboardEvent;
    expect(composing.isComposing).toBe(true);
    fireEvent(messageBox(), composing);
    expect(api.newChat).not.toHaveBeenCalled();
  });

  it("makes one chat for one message, however often Enter is pressed while it is being made", async () => {
    let made!: (c: Chat) => void;
    vi.mocked(api.newChat).mockImplementation(() => new Promise<Chat>((resolve) => (made = resolve)));
    answerWith("ok");
    await openEmpty();
    fireEvent.change(messageBox(), { target: { value: "hello" } });
    fireEvent.keyDown(messageBox(), { key: "Enter" });
    expect(messageBox().value).toBe("");
    fireEvent.change(messageBox(), { target: { value: "and another" } });
    fireEvent.keyDown(messageBox(), { key: "Enter" });
    expect(api.newChat).toHaveBeenCalledTimes(1);
    expect(screen.getByText("STOP")).toBeTruthy(); // the composer counts the message as on its way
    await act(async () => made(fresh));
    await waitFor(() => expect(api.send).toHaveBeenCalledTimes(1));
    expect(api.send).toHaveBeenCalledWith("c9", "hello", expect.any(Function));
  });

  it("shows the pending reply's status for a message typed with no chat open, counted from Send", async () => {
    let say!: (e: StreamEvent) => void;
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      say = onEvent;
      await new Promise<void>(() => {});
    });
    await openEmpty();
    vi.useFakeTimers();
    fireEvent.change(messageBox(), { target: { value: "hello" } });
    fireEvent.keyDown(messageBox(), { key: "Enter" });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0); // the chat is made and opened, and the send has begun
    });
    expect(api.send).toHaveBeenCalledTimes(1);
    await act(async () => say(queuedFor("a1", "u1")));
    expect(screen.getByText("thinking")).toBeTruthy();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(screen.getByText("reading the chat · 3 s")).toBeTruthy(); // the open-chat reset did not take Send's time away
  });
});

describe("ChatTab, demo D2: the pending reply shows that work is going on", () => {
  /** The engine serves the chat's model (nexus), so the status is a word and not `loading nexus`. */
  const nexusServing: Overview = { ...overview, engine: { ...overview.engine, model_id: "file:n" } };
  const queued = queuedFor("a3", "u3");
  const pendingView = () =>
    withTurn(view, "and now?", msg({ id: "a3", role: "assistant", parent_id: "u3", model_id: "file:n", content: "", status: "streaming" }));

  beforeEach(() => {
    vi.mocked(api.models).mockReset().mockResolvedValue(models);
    vi.mocked(api.chats).mockReset().mockResolvedValue([chat]);
    vi.mocked(api.chat).mockReset().mockResolvedValue(view);
    vi.mocked(api.send).mockReset();
    vi.mocked(api.regenerate).mockReset();
    vi.mocked(api.stop).mockReset();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  /** Opens the chat, presses SEND with fake timers running from that moment, and lets the host say `queued`. */
  async function queueAReply(shownOverview: Overview = nexusServing) {
    let say!: (e: StreamEvent) => void;
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      say = onEvent;
      await new Promise<void>(() => {}); // the reply goes on for the whole test
    });
    const shown = render(<ChatTab overview={shownOverview} />);
    await screen.findByText("Verdict: HOLD.");
    vi.useFakeTimers();
    vi.mocked(api.chat).mockResolvedValue(pendingView());
    type("and now?");
    expect(api.send).toHaveBeenCalledTimes(1);
    await act(async () => say(queued));
    return {
      ...shown,
      say: (e: StreamEvent) => act(async () => say(e)),
      tick: (ms: number) =>
        act(async () => {
          await vi.advanceTimersByTimeAsync(ms);
        }),
    };
  }

  it("shows the user's message and the pending reply as soon as the host says queued, before started", async () => {
    await queueAReply();
    expect(screen.getByText("and now?", { selector: ".msg-user" })).toBeTruthy(); // the stored message, before any `started`
    expect(vi.mocked(api.chat)).toHaveBeenCalledTimes(2); // the chat's opening, and queued's reload
    const status = screen.getByText("thinking"); // under a second: the word alone
    expect(status.closest(".msg")?.classList.contains("assistant")).toBe(true);
    expect(status.closest(".msg")?.querySelector(".msg-head")?.textContent).toContain("MODEL");
    expect(screen.queryByText(/waiting for the engine/)).toBeNull();
  });

  it("counts whole seconds from Send and turns the word every 2.5 s", async () => {
    const t = await queueAReply();
    await t.tick(1000);
    expect(screen.getByText("thinking · 1 s")).toBeTruthy();
    await t.tick(2000);
    expect(screen.getByText("reading the chat · 3 s")).toBeTruthy();
    await t.tick(2000);
    expect(screen.getByText("pondering · 5 s")).toBeTruthy();
  });

  it("says `loading <model>` while the engine is starting, restarting, or on another model", async () => {
    for (const engine of [
      { ...overview.engine, model_id: "file:n", state: "starting" as const },
      { ...overview.engine, model_id: "file:n", state: "restarting" as const },
      { ...overview.engine, model_id: "file:q", state: "serving" as const }, // serving the OTHER model
    ]) {
      const t = await queueAReply({ ...overview, engine });
      await t.tick(3000);
      expect(screen.getByText("loading nexus · 3 s"), engine.state + " " + engine.model_id).toBeTruthy();
      t.unmount();
      vi.useRealTimers();
      vi.mocked(api.send).mockReset();
      vi.mocked(api.chat).mockReset().mockResolvedValue(view);
    }
  });

  it("goes from `loading <model>` to the words when the engine is serving the chat's model", async () => {
    const starting = { ...overview, engine: { ...overview.engine, model_id: "file:n", state: "starting" as const } };
    const t = await queueAReply(starting);
    await t.tick(3000);
    expect(screen.getByText("loading nexus · 3 s")).toBeTruthy();
    t.rerender(<ChatTab overview={nexusServing} />);
    expect(screen.getByText("reading the chat · 3 s")).toBeTruthy();
  });

  it("adds the dim hint about long chats after 15 s with no token", async () => {
    const t = await queueAReply();
    await t.tick(14_000);
    expect(screen.queryByText(/long chats take longer/)).toBeNull();
    await t.tick(1000);
    const hint = screen.getByText("long chats take longer: the model re-reads the last ~1,500 tokens of the chat before it answers");
    expect(hint.classList.contains("dim")).toBe(true);
  });

  it("is gone with the first token, and the interval with it", async () => {
    const t = await queueAReply();
    await t.tick(2000);
    expect(vi.getTimerCount()).toBe(1); // one interval for the one pending reply
    await t.say({ ...started, message_id: "a3", user_message_id: "u3" });
    expect(screen.getByText("thinking · 2 s")).toBeTruthy(); // started is not a token
    await t.say({ kind: "token", text: "Buy" });
    await t.tick(FLUSH_MS); // the redraw that shows the tokens
    expect(screen.getByText("Buy")).toBeTruthy();
    expect(screen.queryByText(/thinking|reading the chat|pondering/)).toBeNull();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("clears its interval when the Chat tab goes away", async () => {
    const t = await queueAReply();
    expect(vi.getTimerCount()).toBe(1);
    t.unmount();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("keeps STOP working throughout, and the status goes when the reply is stopped before its first token", async () => {
    const t = await queueAReply();
    await t.tick(2000);
    vi.mocked(api.stop).mockImplementation(async () => {});
    fireEvent.click(screen.getByText("STOP"));
    expect(api.stop).toHaveBeenCalledWith("c1");
    // the host has marked the placeholder stopped by the time it says so
    vi.mocked(api.chat).mockResolvedValue(withTurn(view, "and now?", msg({ id: "a3", role: "assistant", parent_id: "u3", content: "", status: "stopped" })));
    await t.say({ kind: "stopped", stats: done });
    expect(screen.queryByText(/thinking|reading the chat/)).toBeNull();
    expect(screen.getByText("SEND")).toBeTruthy();
  });

  it("shows the same status for a regenerate, counted from the press", async () => {
    let say!: (e: StreamEvent) => void;
    vi.mocked(api.regenerate).mockImplementation(async (_chatId, onEvent) => {
      say = onEvent;
      await new Promise<void>(() => {});
    });
    render(<ChatTab overview={nexusServing} />);
    await screen.findByText("Verdict: HOLD.");
    vi.useFakeTimers();
    vi.mocked(api.chat).mockResolvedValue({
      ...view,
      messages: [
        view.messages[0],
        { message: msg({ id: "a3", role: "assistant", parent_id: "u1", model_id: "file:n", content: "", status: "streaming" }), version: 3, versions: 3, sibling_ids: ["a1", "a2", "a3"] },
      ],
    });
    fireEvent.click(screen.getByText("REGENERATE"));
    await act(async () => say(queuedFor("a3", "u1")));
    expect(screen.getByText("thinking")).toBeTruthy();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(screen.getByText("reading the chat · 3 s")).toBeTruthy();
  });
});

// Demo D5: the reply is redrawn at most once every FLUSH_MS. The redraws of a streaming reply were 19% of a core at 15 tok/s
// (the whole text parsed on each, one per token), and a redraw has costs that no parser fix takes away (React, layout, paint,
// the GPU process): at 100 ms the tokens of ~two redraws' time arrive together. A token that arrives after a quiet spell is
// shown at once, so the first word of a reply is not held back.
describe("ChatTab, demo D5: streaming redraws are throttled", () => {
  beforeEach(() => {
    vi.mocked(api.models).mockReset().mockResolvedValue(models);
    vi.mocked(api.chats).mockReset().mockResolvedValue([chat]);
    vi.mocked(api.chat).mockReset().mockResolvedValue(view);
    vi.mocked(api.send).mockReset();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows a token at once after a quiet spell, and gathers the tokens that follow into one redraw per FLUSH_MS", async () => {
    let say!: (e: StreamEvent) => void;
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      say = onEvent;
      await new Promise<void>(() => {});
    });
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    vi.useFakeTimers();
    vi.mocked(api.chat).mockResolvedValue(withTurn(view, "go", msg({ id: "a3", role: "assistant", parent_id: "u3", content: "", status: "streaming" })));
    type("go");
    const at = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });
    const tell = (e: StreamEvent) => act(async () => say(e));
    await tell({ kind: "queued", message_id: "a3", user_message_id: "u3" });
    await tell(started);
    await at(FLUSH_MS + 1); // a quiet spell
    await tell({ kind: "token", text: "A" });
    await at(1);
    expect(screen.getByText("A")).toBeTruthy(); // at once, not FLUSH_MS later
    await tell({ kind: "token", text: "B" });
    await at(20);
    await tell({ kind: "token", text: "C" });
    await at(20);
    expect(screen.queryByText("ABC")).toBeNull(); // B and C wait for the interval
    expect(screen.getByText("A")).toBeTruthy();
    await at(FLUSH_MS);
    expect(screen.getByText("ABC")).toBeTruthy(); // ... and are drawn together
  });

  it("draws at most once in FLUSH_MS however fast the tokens come", async () => {
    let say!: (e: StreamEvent) => void;
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      say = onEvent;
      await new Promise<void>(() => {});
    });
    render(<ChatTab overview={overview} />);
    await screen.findByText("Verdict: HOLD.");
    vi.useFakeTimers();
    vi.mocked(api.chat).mockResolvedValue(withTurn(view, "go", msg({ id: "a3", role: "assistant", parent_id: "u3", content: "", status: "streaming" })));
    type("go");
    const at = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });
    await act(async () => say({ kind: "queued", message_id: "a3", user_message_id: "u3" }));
    await act(async () => say(started));
    await at(FLUSH_MS + 1);
    const redraws = new Set<string>();
    for (let i = 0; i < 40; i++) {
      await act(async () => say({ kind: "token", text: "x" }));
      await at(5); // 200 tokens a second
      redraws.add(document.querySelector(".msg.assistant:last-child .md")?.textContent ?? "");
    }
    // 40 tokens over 200 ms: the first at once, then one redraw per FLUSH_MS
    expect(redraws.size).toBeLessThanOrEqual(Math.ceil(200 / FLUSH_MS) + 3); // (the empty text before the first redraw counts too)
  });
});

// ---------------------------------------------------------------------------------------------------------------------
// Demo D9: the engine loads a chat's model when the user starts typing in it, not when a chat or a model is picked. The first key
// of EVERY draft (the box was empty, now it is not) tells the host which model the draft is for, whatever the engine is doing
// (R77): only the host knows whether a reply is under way in some other chat, and it alone decides what the call means (load the
// model, keep it for after the replies, or nothing). The web remembers nothing, except a model the host refused. Each test was
// run RED first, or says that it is a pin.
// ---------------------------------------------------------------------------------------------------------------------

describe("ChatTab, demo D9: typing loads the chat's model", () => {
  const messageBox = () => screen.getByLabelText("message") as HTMLTextAreaElement;
  const modelBox = () => screen.getByLabelText("model") as HTMLSelectElement;
  /** The engine serves the default model (qwen); the chat in the fixtures is on nexus, which it does not serve. */
  const engineOn = (modelId: string, state: "serving" | "starting" | "restarting" | "failed" = "serving"): Overview => ({
    ...overview,
    engine: { ...overview.engine, model_id: modelId, state },
  });
  const key = (text: string) => fireEvent.change(messageBox(), { target: { value: text } });

  beforeEach(() => {
    vi.mocked(api.models).mockReset().mockResolvedValue(models);
    vi.mocked(api.chats).mockReset().mockResolvedValue([chat]);
    vi.mocked(api.chat).mockReset().mockResolvedValue(view);
    vi.mocked(api.send).mockReset();
    vi.mocked(api.setChatModel).mockReset().mockResolvedValue(undefined);
    vi.mocked(api.prepareModel).mockReset().mockResolvedValue(undefined);
  });

  it("asks the host to prepare the chat's model at the first key of a draft", async () => {
    render(<ChatTab overview={engineOn("file:q")} />);
    await screen.findByText("Verdict: HOLD.");
    expect(api.prepareModel).not.toHaveBeenCalled(); // opening the chat is not typing
    key("L");
    expect(api.prepareModel).toHaveBeenCalledTimes(1);
    expect(api.prepareModel).toHaveBeenCalledWith("file:n");
  });

  // R77: the web cannot tell whether a reply is under way in ANOTHER chat (the overview says nothing of it), so it cannot know
  // whether "the engine has this model" is the end of the matter. The host knows, and it keeps the latest model asked for during a
  // reply, the engine's own included. So the web asks at every draft, and the host decides.
  it("a new draft's first key asks the host even when the engine already serves that model", async () => {
    render(<ChatTab overview={engineOn("file:n")} />); // the chat in the fixtures is on nexus
    await screen.findByText("Verdict: HOLD.");
    key("hello");
    expect(api.prepareModel).toHaveBeenCalledTimes(1);
    expect(api.prepareModel).toHaveBeenCalledWith("file:n");
  });

  it("asks while the engine is starting or restarting the chat's model, too", async () => {
    for (const state of ["starting", "restarting"] as const) {
      vi.mocked(api.prepareModel).mockClear();
      const view_ = render(<ChatTab overview={engineOn("file:n", state)} />);
      await screen.findByText("Verdict: HOLD.");
      key("hello");
      expect(api.prepareModel, state).toHaveBeenCalledWith("file:n");
      view_.unmount();
    }
  });

  it("asks once for each draft: the keys that follow ask nothing, and a draft that is started again asks again", async () => {
    render(<ChatTab overview={engineOn("file:q")} />);
    await screen.findByText("Verdict: HOLD.");
    key("L");
    key("Lo");
    key("Lon");
    expect(api.prepareModel).toHaveBeenCalledTimes(1); // one draft, however many keys
    key("");
    key("L"); // the draft was emptied and is started again
    expect(api.prepareModel).toHaveBeenCalledTimes(2);
    key("Long or wait?");
    expect(api.prepareModel).toHaveBeenCalledTimes(2);
  });

  // The flow R77 is for: the reply of a send is still being written, and the user starts the next message. The engine serves this
  // chat's model, and the host is told all the same: it may be holding the model of another chat's draft for after the reply, and
  // the draft here is the later intent.
  it("asks for the draft that follows a send, typed while the reply of that send is still being written", async () => {
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      onEvent(started);
      await new Promise<void>(() => {}); // the reply goes on for the whole test
    });
    render(<ChatTab overview={engineOn("file:n")} />);
    await screen.findByText("Verdict: HOLD.");
    type("go"); // the first key of a draft, and then Send
    expect(api.prepareModel).toHaveBeenCalledTimes(1);
    await screen.findByText("STOP"); // the reply is being written
    key("and then");
    expect(api.prepareModel).toHaveBeenCalledTimes(2);
    expect(api.prepareModel).toHaveBeenLastCalledWith("file:n");
  });

  it("asks when the engine failed on the model, or serves another one", async () => {
    render(<ChatTab overview={engineOn("file:n", "failed")} />);
    await screen.findByText("Verdict: HOLD.");
    key("hello");
    expect(api.prepareModel).toHaveBeenCalledWith("file:n");
  });

  it("asks nothing for opening chats or changing the picker without typing", async () => {
    const second = { ...chat, id: "c2", title: "Second chat", model_id: "file:q" };
    vi.mocked(api.chats).mockResolvedValue([chat, second]);
    vi.mocked(api.chat).mockImplementation(async (id) => ({ ...view, chat: id === "c2" ? second : chat }));
    render(<ChatTab overview={engineOn("file:q")} />);
    await screen.findByText("Verdict: HOLD.");
    fireEvent.click(screen.getByText("Second chat"));
    await waitFor(() => expect(api.chat).toHaveBeenCalledWith("c2"));
    fireEvent.change(modelBox(), { target: { value: "file:n" } });
    await waitFor(() => expect(api.setChatModel).toHaveBeenCalledWith("c2", "file:n"));
    expect(api.prepareModel).not.toHaveBeenCalled();
  });

  it("asks for the model of the chat that is open when a draft starts", async () => {
    const second = { ...chat, id: "c2", title: "Second chat", model_id: "file:q" };
    vi.mocked(api.chats).mockResolvedValue([chat, second]);
    vi.mocked(api.chat).mockImplementation(async (id) => ({ ...view, chat: id === "c2" ? second : chat }));
    render(<ChatTab overview={engineOn("file:x")} />); // the engine serves neither
    await screen.findByText("Verdict: HOLD.");
    key("one");
    expect(api.prepareModel).toHaveBeenLastCalledWith("file:n");
    fireEvent.click(screen.getByText("Second chat"));
    await waitFor(() => expect(api.chat).toHaveBeenCalledWith("c2"));
    await waitFor(() => expect(messageBox().value).toBe("one")); // the draft stays with the composer
    key("");
    key("two");
    expect(api.prepareModel).toHaveBeenLastCalledWith("file:q");
    expect(api.prepareModel).toHaveBeenCalledTimes(2);
  });

  // R75, kept as a pin of the flow its memo used to break, now that nothing is remembered: chat A (q) -> type -> the engine moves to
  // q -> chat B (n) -> type -> the engine moves to n -> back to A -> type. Every draft asks, so the last one asks for q again.
  it("asks for each draft as the user goes from chat A to chat B and back, whatever the engine does in between (R75)", async () => {
    const a = { ...chat, id: "c1", title: "Chat A", model_id: "file:q" };
    const b = { ...chat, id: "c2", title: "Chat B", model_id: "file:n" };
    vi.mocked(api.chats).mockResolvedValue([a, b]);
    vi.mocked(api.chat).mockImplementation(async (id) => ({ ...view, chat: id === "c2" ? b : a }));
    const shown = render(<ChatTab overview={engineOn("file:x")} />); // the engine serves neither model
    await screen.findByText("Verdict: HOLD.");
    await waitFor(() => expect(modelBox().value).toBe("file:q")); // chat A is open
    key("one");
    expect(api.prepareModel).toHaveBeenLastCalledWith("file:q");
    shown.rerender(<ChatTab overview={engineOn("file:q")} />); // the host moved the engine to q

    fireEvent.click(screen.getByText("Chat B"));
    await waitFor(() => expect(modelBox().value).toBe("file:n")); // chat B is open (the draft stays with the composer)
    key("");
    key("two");
    expect(api.prepareModel).toHaveBeenLastCalledWith("file:n");
    expect(api.prepareModel).toHaveBeenCalledTimes(2);
    shown.rerender(<ChatTab overview={engineOn("file:n")} />); // ... and on to n

    fireEvent.click(screen.getByText("Chat A"));
    await waitFor(() => expect(modelBox().value).toBe("file:q")); // back in chat A, which the engine no longer serves
    key("");
    key("three");
    expect(api.prepareModel).toHaveBeenCalledTimes(3);
    expect(api.prepareModel).toHaveBeenLastCalledWith("file:q");
  });

  it("with no chat open, prepares the model the picker shows", async () => {
    vi.mocked(api.chats).mockResolvedValue([]);
    render(<ChatTab overview={engineOn("file:x")} />);
    await screen.findByText("Type a message to start a chat.");
    await waitFor(() => expect(modelBox().value).toBe("file:q")); // the default
    key("h");
    expect(api.prepareModel).toHaveBeenLastCalledWith("file:q");
    key("");
    fireEvent.change(modelBox(), { target: { value: "file:n" } }); // picking asks nothing
    expect(api.prepareModel).toHaveBeenCalledTimes(1);
    key("h"); // but it is another model now: the first key of this draft asks for it
    expect(api.prepareModel).toHaveBeenLastCalledWith("file:n");
    expect(api.prepareModel).toHaveBeenCalledTimes(2);
  });

  it("with no chat open, asks even when the engine already serves the model of the picker", async () => {
    vi.mocked(api.chats).mockResolvedValue([]);
    render(<ChatTab overview={engineOn("file:q")} />);
    await screen.findByText("Type a message to start a chat.");
    await waitFor(() => expect(modelBox().value).toBe("file:q"));
    key("h");
    expect(api.prepareModel).toHaveBeenCalledTimes(1);
    expect(api.prepareModel).toHaveBeenCalledWith("file:q");
  });

  it("shows a host failure in the notice, once, like the other host calls", async () => {
    vi.mocked(api.prepareModel).mockRejectedValue("Qwen3 cannot run: the engine refused it");
    render(<ChatTab overview={engineOn("file:q")} />);
    await screen.findByText("Verdict: HOLD.");
    key("hello");
    expect((await screen.findByRole("alert")).textContent).toContain("cannot run");
    expect(hostErrors.mock.calls).toEqual([["prepareModel failed:", "Qwen3 cannot run: the engine refused it"]]);
    key("hello again"); // no second call, so no second failure
    await act(async () => {});
    expect(hostErrors.mock.calls).toHaveLength(1);
  });

  // The one thing the web remembers (R77): a model the host refused. Every draft asks, so a model that cannot run would give a notice
  // for each draft; it is not asked for again until the list of models is another list (the host marks a refused model CAN'T RUN, and
  // the Chat tab asks for the list again each time it is shown). The overview that arrives every second is not that list.
  it("does not ask again for a model the host refused, until the list of models changes", async () => {
    vi.mocked(api.prepareModel).mockRejectedValue("nexus cannot run: the engine refused it");
    const shown = render(<ChatTab overview={engineOn("file:q")} active />);
    await screen.findByText("Verdict: HOLD.");
    key("one");
    await waitFor(() => expect(hostErrors.mock.calls).toHaveLength(1)); // refused: one notice
    expect(api.prepareModel).toHaveBeenCalledTimes(1);

    key("");
    key("two"); // a new draft: the model the host refused is not asked for again
    shown.rerender(<ChatTab overview={engineOn("file:q", "restarting")} active />); // the overview keeps arriving
    shown.rerender(<ChatTab overview={{ ...engineOn("file:n"), request_count: 9 }} active />);
    key("");
    key("three");
    expect(api.prepareModel).toHaveBeenCalledTimes(1);
    expect(hostErrors.mock.calls).toHaveLength(1);

    // The host has marked the model CAN'T RUN, and the tab is shown again: the list of models is not the list it was.
    vi.mocked(api.models).mockResolvedValue(models.map((m) => (m.id === "file:n" ? { ...m, runnable: false, reason: "the engine refused it" } : m)));
    shown.rerender(<ChatTab overview={engineOn("file:q")} active={false} />);
    shown.rerender(<ChatTab overview={engineOn("file:q")} active />);
    await waitFor(() => expect((screen.getByRole("option", { name: /nexus/ }) as HTMLOptionElement).disabled).toBe(true));
    key("");
    key("four"); // asked for again, and refused again: a second notice, now that the list says why
    expect(api.prepareModel).toHaveBeenCalledTimes(2);
    await waitFor(() => expect(hostErrors.mock.calls).toHaveLength(2));
    key("");
    key("five");
    expect(api.prepareModel).toHaveBeenCalledTimes(2); // and once more it is not asked for again
  });

  it("says in the panel that the model loads when the user starts typing", async () => {
    render(<ChatTab overview={engineOn("file:q")} />);
    await screen.findByText("Verdict: HOLD.");
    expect(screen.getByText("engine").nextElementSibling?.textContent).toBe("loads when you start typing");
  });
});
