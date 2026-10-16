import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { App } from "../App";
import { api } from "../ipc/api";
import { overview } from "./testApi";

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
