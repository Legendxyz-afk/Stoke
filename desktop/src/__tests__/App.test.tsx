import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { App } from "../App";
import { StatusBar } from "../components/StatusBar";
import { TopBar } from "../components/TopBar";
import { api } from "../ipc/api";
import type { StreamEvent } from "../ipc/bindings/StreamEvent";
import { chat, models, msg, overview, view as storedChat, withTurn } from "./testApi";

describe("App", () => {
  it("shows the wordmark, the six tabs and the engine state", async () => {
    render(<App />);
    expect(screen.getByText("lorren")).toBeTruthy();
    expect(screen.getAllByRole("tab").map((t) => t.textContent)).toEqual(["Chat", "Swarm", "Models", "Requests", "Logs", "Settings"]);
    expect(await screen.findByText("serving")).toBeTruthy();
    expect(screen.getByText("Qwen3-4B-Instruct-2507-Q4_K_M")).toBeTruthy();
    expect(screen.getByText(/engine 2\.4 GB/)).toBeTruthy();
  });

  it("switches tabs", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Swarm" }));
    expect(screen.getByRole("tab", { name: "Swarm" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getByText(/milestone 3/)).toBeTruthy();
    await screen.findByText("serving");
  });
});

// A rejected overview() is an IPC-level fault (a misnamed or unregistered command, an ACL, serialisation). It must
// not look like first paint: no "starting" forever, and nothing swallowed.
describe("App when the host does not answer", () => {
  let logged: MockInstance<typeof console.error>;
  const line = (message: string) => ["overview failed:", message];

  beforeEach(() => {
    vi.mocked(api.overview).mockReset().mockImplementation(async () => overview);
    vi.mocked(api.onEngineStatus).mockReset().mockImplementation(async () => () => {});
    logged = vi.spyOn(console, "error").mockImplementation(() => {});
  });
  afterEach(() => {
    logged.mockRestore();
    vi.useRealTimers();
  });

  it("says host unreachable, not starting, when overview() rejects", async () => {
    vi.mocked(api.overview).mockRejectedValue("Command overview not found");
    render(<App />);
    const label = await screen.findByText("host unreachable");
    expect(label.classList.contains("flame-text")).toBe(true);
    expect(label.getAttribute("title")).toBe("Command overview not found");
    expect(screen.queryByText("starting")).toBeNull();
    expect(logged.mock.calls).toEqual([line("Command overview not found")]);
  });

  it("logs a failure once per distinct message, shows the latest, recovers, and logs a new outage again", async () => {
    vi.useFakeTimers();
    vi.mocked(api.overview)
      .mockRejectedValueOnce("boom") // Tauri rejects with the command's error string...
      .mockRejectedValueOnce("boom")
      .mockRejectedValueOnce(new Error("other")) // ...a JS fault rejects with an Error, whose message is what counts
      .mockResolvedValueOnce(overview)
      .mockRejectedValueOnce(new Error("other"));
    render(<App />);
    const poll = () => act(async () => { await vi.advanceTimersByTimeAsync(2000); });

    await act(async () => {}); // the first overview() call, made on mount, rejects with "boom"
    expect(screen.getByText("host unreachable").getAttribute("title")).toBe("boom");
    expect(logged.mock.calls).toEqual([line("boom")]);

    await poll(); // the same message again: shown, not logged again
    expect(api.overview).toHaveBeenCalledTimes(2); // so the poll did fire; it is the log that held back
    expect(logged.mock.calls).toEqual([line("boom")]);

    await poll(); // a different message: logged, and it is the one shown
    expect(logged.mock.calls).toEqual([line("boom"), line("other")]);
    expect(screen.getByText("host unreachable").getAttribute("title")).toBe("other");

    await poll(); // the host answers: the real state replaces the failure
    expect(screen.queryByText("host unreachable")).toBeNull();
    expect(screen.getByText("serving")).toBeTruthy();

    await poll(); // a failure after a recovery is a new outage: logged again, but an overview has arrived, so the bar keeps it
    expect(logged.mock.calls).toEqual([line("boom"), line("other"), line("other")]);
    expect(screen.queryByText("host unreachable")).toBeNull();
    expect(screen.getByText("serving")).toBeTruthy();
    expect(api.overview).toHaveBeenCalledTimes(5);
  });

  it("logs a failing engine-status listener, and unmounting after it does not throw", async () => {
    vi.mocked(api.onEngineStatus).mockRejectedValueOnce("event:default not allowed by ACL");
    const view = render(<App />);
    expect(await screen.findByText("serving")).toBeTruthy(); // overview() still answers
    view.unmount(); // the cleanup runs unlisten.then((f) => f()); a missing function would be an unhandled rejection
    await act(async () => {});
    expect(logged.mock.calls).toEqual([["engine-status listener failed:", "event:default not allowed by ACL"]]);
  });
});

// R36(a): the Chat tab stays mounted while another tab shows. If it were unmounted on a tab switch, a reply streaming
// into it would lose its events and its live text (and be left on STOP for good), and the composer draft would be gone.
describe("App keeps the Chat tab mounted", () => {
  const started: StreamEvent = { kind: "started", message_id: "a3", user_message_id: "u3", dropped_turns: 0, request_id: "req_1" };
  const done = { prompt_tokens: 60, completion_tokens: 1, ttft_ms: 212, total_ms: 650, tokens_per_s: null, finish_reason: "stop" };
  const messageBox = () => screen.getByLabelText("message") as HTMLTextAreaElement;
  const tab = (name: string) => fireEvent.click(screen.getByRole("tab", { name }));

  beforeEach(() => {
    vi.mocked(api.overview).mockReset().mockImplementation(async () => overview);
    vi.mocked(api.onEngineStatus).mockReset().mockImplementation(async () => () => {});
    vi.mocked(api.models).mockReset().mockResolvedValue(models);
    vi.mocked(api.chats).mockReset().mockResolvedValue([chat]);
    vi.mocked(api.chat).mockReset().mockResolvedValue(storedChat);
    vi.mocked(api.send).mockReset();
  });

  it("keeps the composer draft across a tab switch", async () => {
    render(<App />);
    await screen.findByText("Verdict: HOLD.");
    fireEvent.change(messageBox(), { target: { value: "draft text" } });
    tab("Swarm");
    expect(screen.queryByRole("textbox", { name: "message" })).toBeNull(); // not shown over the other tab
    tab("Chat");
    expect(messageBox().value).toBe("draft text");
  });

  it("keeps a reply that streams while another tab shows live, and ends it, on return", async () => {
    // The store as the host would have it: the reply is a streaming row until it finishes.
    let phase: "idle" | "streaming" | "done" = "idle";
    vi.mocked(api.chat).mockImplementation(async () =>
      phase === "idle"
        ? storedChat
        : withTurn(storedChat, "go", msg({ id: "a3", role: "assistant", parent_id: "u3",
            content: phase === "done" ? "Buy" : "", status: phase === "done" ? "done" : "streaming" })));
    let end!: () => void;
    vi.mocked(api.send).mockImplementation(async (_chatId, _text, onEvent) => {
      onEvent(started);
      onEvent({ kind: "token", text: "Buy " });
      await new Promise<void>((r) => (end = r));
      onEvent({ kind: "finished", stats: done });
    });
    render(<App />);
    await screen.findByText("Verdict: HOLD.");
    phase = "streaming";
    fireEvent.change(messageBox(), { target: { value: "go" } });
    fireEvent.click(screen.getByRole("button", { name: "SEND" }));
    tab("Swarm");
    tab("Chat");
    expect(await screen.findByText("Buy")).toBeTruthy(); // the live text: the stored row is still empty
    expect(screen.getByRole("button", { name: "STOP" })).toBeTruthy();
    phase = "done";
    end();
    expect(await screen.findByRole("button", { name: "SEND" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "STOP" })).toBeNull();
    expect(screen.getByText("Buy")).toBeTruthy();
  });

  // R60: it stays mounted, so its models() would otherwise run once per session. App tells it when it is shown.
  it("asks the host for the models again each time the Chat tab is shown, and for the chats only once", async () => {
    render(<App />);
    await screen.findByText("Verdict: HOLD.");
    expect(api.models).toHaveBeenCalledTimes(1);
    expect(api.chats).toHaveBeenCalledTimes(1);
    tab("Swarm");
    expect(api.models).toHaveBeenCalledTimes(1); // hiding it asks for nothing
    tab("Chat");
    await waitFor(() => expect(api.models).toHaveBeenCalledTimes(2));
    tab("Settings"); // not Models: that tab asks the host for the models itself, and this test counts the Chat tab's calls
    tab("Chat");
    await waitFor(() => expect(api.models).toHaveBeenCalledTimes(3));
    expect(api.chats).toHaveBeenCalledTimes(1); // its first load worked, so it is not asked again
  });
});

// The two bars cut a long text with an ellipsis (a 30 px status bar, a top bar that shares its width with six tabs): the
// whole text is in the element's title, so it can be read where it is cut.
describe("the bars keep the full text of what they cut off", () => {
  it("the top bar's model name carries its whole name in a title", () => {
    render(<TopBar tab="Chat" onTab={() => {}} overview={overview} />);
    const name = screen.getByText("Qwen3-4B-Instruct-2507-Q4_K_M");
    expect(name.classList.contains("ell")).toBe(true); // the element that gets the ellipsis
    expect(name.getAttribute("title")).toBe("Qwen3-4B-Instruct-2507-Q4_K_M");
  });

  // The memory item reads "app N MB · engine N GB · used/total GB": the unit once at the end, every number kept (the bar
  // is tight at 1100 px). `static` marks the two settings that the stylesheet leaves out below 1280 px.
  it("the top bar's memory writes the GB unit once after used/total, and marks t= and kv as the items that can go", () => {
    render(<TopBar tab="Chat" onTab={() => {}} overview={overview} />);
    expect(screen.getByText(/^app 150 MB/).textContent).toBe("app 150 MB · engine 2.4 GB · 10.0/15.6 GB");
    expect(screen.getByText("t=auto").classList.contains("static")).toBe(true);
    expect(screen.getByText("kv q8").classList.contains("static")).toBe(true);
    expect(screen.getByText("serving").classList.contains("static")).toBe(false);
  });

  it("the status bar's last engine error carries its whole text in a title", () => {
    const last_error = "the engine stopped while loading: stoke serve: model: 'M:/New Drive/ai models/x.gguf' -- general.architecture is 'qwen2vl'";
    render(<StatusBar overview={{ ...overview, engine: { ...overview.engine, last_error } }} />);
    const line = screen.getByText(last_error);
    expect(line.classList.contains("ell")).toBe(true);
    expect(line.getAttribute("title")).toBe(last_error);
  });
});
