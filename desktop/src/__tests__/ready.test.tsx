import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { App } from "../App";
import { api } from "../ipc/api";
import { READY_EVENT, reportReady, resetReady } from "../opening/ready";
import { captureHostErrors, chat, models, msg, overview, view } from "./testApi";

const wait = (ms: number) => act(async () => { await new Promise((r) => setTimeout(r, ms)); });

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

describe("the ready signal", () => {
  let spy: Mock<() => void>;
  let seen: (() => void) | undefined;

  beforeEach(() => {
    resetReady();
    spy = vi.fn<() => void>(() => seen?.());
    seen = undefined;
    window.addEventListener(READY_EVENT, spy);
    vi.mocked(api.overview).mockReset().mockImplementation(async () => overview);
    vi.mocked(api.models).mockReset().mockImplementation(async () => models);
    vi.mocked(api.chats).mockReset().mockImplementation(async () => []);
    vi.mocked(api.chat).mockReset();
    vi.mocked(api.onEngineStatus).mockReset().mockImplementation(async () => () => {});
  });
  afterEach(() => {
    window.removeEventListener(READY_EVENT, spy);
    resetReady();
  });

  it("fires once, after the fourth part", async () => {
    reportReady("overview");
    reportReady("chats");
    reportReady("models");
    await wait(100);
    expect(spy).toHaveBeenCalledTimes(0);
    reportReady("chat");
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    for (const p of ["overview", "chats", "chat", "models"] as const) reportReady(p);
    await wait(100);
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("a fresh app with no chats is ready once the overview, the empty list and the models are in", async () => {
    render(<App />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
  });

  it("waits for the chat list", async () => {
    const d = deferred<never[]>();
    vi.mocked(api.chats).mockImplementation(() => d.promise);
    render(<App />);
    await wait(200);
    expect(spy).toHaveBeenCalledTimes(0);
    await act(async () => { d.resolve([]); });
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
  });

  it("waits for the open chat's messages, and they are on screen when it fires", async () => {
    const d = deferred<typeof view>();
    vi.mocked(api.chats).mockImplementation(async () => [chat]);
    vi.mocked(api.chat).mockImplementation(() => d.promise);
    const text = "Verdict from the stored chat";
    const shown = {
      ...view,
      messages: [{ message: msg({ id: "u1", role: "user", content: text }), version: 1, versions: 1, sibling_ids: ["u1"] }],
    };
    let onScreen = false;
    render(<App />);
    seen = () => { onScreen = screen.queryByText(text) !== null; };
    await wait(200);
    expect(spy).toHaveBeenCalledTimes(0);
    await act(async () => { d.resolve(shown); });
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    expect(onScreen).toBe(true);
  });

  it("fires for a host that does not answer", async () => {
    const logged = captureHostErrors();
    vi.mocked(api.overview).mockRejectedValue("Command overview not found");
    render(<App />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    expect(screen.getByText("host unreachable")).toBeTruthy();
    logged.mockRestore();
  });

  it("fires when the chat list fails", async () => {
    const logged = captureHostErrors();
    vi.mocked(api.chats).mockRejectedValue("no list");
    render(<App />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    logged.mockRestore();
  });

  it("does not wait for the model to warm up", async () => {
    vi.mocked(api.overview).mockImplementation(async () => ({ ...overview, engine: { ...overview.engine, state: "starting" } }));
    render(<App />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
  });
});
