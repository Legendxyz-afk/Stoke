import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { App } from "../App";
import { api } from "../ipc/api";
import type { LogLine } from "../ipc/bindings/LogLine";
import type { RequestRecord } from "../ipc/bindings/RequestRecord";
import { LogsTab } from "../tabs/LogsTab";
import { ModelsTab } from "../tabs/ModelsTab";
import { RequestsTab } from "../tabs/RequestsTab";
import { captureHostErrors, models, overview } from "./testApi";

beforeEach(() => vi.clearAllMocks());

describe("ModelsTab", () => {
  it("lists every model, marks the default and the loaded one, and the one that can't run", async () => {
    vi.mocked(api.models).mockResolvedValue(models);
    render(<ModelsTab overview={overview} />);
    expect(await screen.findByText("qwen2.5vl:3b")).toBeTruthy();
    expect(screen.getAllByRole("row")).toHaveLength(3);
    expect(screen.getByText("DEFAULT")).toBeTruthy();
    expect(screen.getByText("LOADED")).toBeTruthy();
    expect(screen.getAllByText("CAN'T RUN")).toHaveLength(1);
    expect(screen.getByText("3 IN LIBRARY · 2 RUNNABLE")).toBeTruthy();
  });

  it("shows a model's Modelfile prompt, and why another cannot run", async () => {
    vi.mocked(api.models).mockResolvedValue(models);
    render(<ModelsTab overview={overview} />);
    fireEvent.click(await screen.findByText("nexus"));
    expect(screen.getByText(/You are Nexus AI/)).toBeTruthy();
    expect(screen.getByText(/FROM MODELFILE/)).toBeTruthy();
    fireEvent.click(screen.getByText("qwen2.5vl:3b"));
    expect(screen.getAllByText("CAN'T RUN")).toHaveLength(2);
    expect(screen.getByText(/architecture qwen2vl/)).toBeTruthy();
  });

  it("shows the memory a model needs to run with 2 and with 8 chats at once", async () => {
    vi.mocked(api.models).mockResolvedValue(models);
    render(<ModelsTab overview={overview} />);
    await screen.findByText("qwen2.5vl:3b");
    // Qwen3-4B: run_bytes 3,407,444,704 B = 2,497,280,736 (file) + 641,728,512 (78,336 B/token x 4,096 x 2) + 268,435,456 (256 MiB)
    // 3,407,444,704 / 1,073,741,824 = 3.173 GB, shown with one decimal
    expect(screen.getByText("~3.2 GB")).toBeTruthy();
    // run_bytes_max 5,332,630,240 B = 2,497,280,736 + 2,566,914,048 (78,336 x 4,096 x 8) + 268,435,456; / 1,073,741,824 = 4.966
    expect(screen.getByText("~5.0 GB")).toBeTruthy();
  });
});

describe("RequestsTab", () => {
  it("shows each request with its timings and outcome", async () => {
    vi.mocked(api.requests).mockResolvedValue([
      { id: "req_1", started_ms: Date.now(), source: "chat", endpoint: "/v1/chat/completions", model: "nexus",
        prompt_tokens: 41, completion_tokens: 5, ttft_ms: 480, total_ms: 730, tokens_per_s: 16.0, status: "ok", dropped_turns: 0 },
      { id: "req_2", started_ms: Date.now(), source: "chat", endpoint: "/v1/chat/completions", model: "nexus",
        prompt_tokens: null, completion_tokens: null, ttft_ms: null, total_ms: 90, tokens_per_s: null, status: "stopped", dropped_turns: 2 },
    ]);
    render(<RequestsTab />);
    expect(await screen.findByText("480 ms")).toBeTruthy();
    expect(screen.getByText("730 ms")).toBeTruthy();
    expect(screen.getByText("16.0")).toBeTruthy();
    expect(screen.getByText("OK")).toBeTruthy();
    expect(screen.getByText("STOPPED · 2 CUT")).toBeTruthy();
  });

  it("says when there is nothing yet", async () => {
    vi.mocked(api.requests).mockResolvedValue([]);
    render(<RequestsTab />);
    expect(await screen.findByText("No requests yet.")).toBeTruthy();
  });
});

describe("LogsTab", () => {
  const line = (seq: number, source: string, text: string, level = "info"): LogLine =>
    ({ seq, time_ms: Date.now(), source, level, text });

  it("asks only for lines after the last one it has, and filters by source", async () => {
    vi.mocked(api.logs)
      .mockResolvedValueOnce([line(1, "app", "Lorren 0.0.1 starting"), line(2, "engine", "stoke serve: listening on 127.0.0.1:50123")])
      .mockResolvedValue([]);
    render(<LogsTab />);
    expect(await screen.findByText(/listening on 127\.0\.0\.1:50123/)).toBeTruthy();
    expect(api.logs).toHaveBeenNthCalledWith(1, 0);
    await waitFor(() => expect(api.logs).toHaveBeenLastCalledWith(2), { timeout: 2500 });
    fireEvent.click(screen.getByText("ENGINE"));
    expect(screen.queryByText(/Lorren 0\.0\.1 starting/)).toBeNull();
    expect(screen.getByText(/listening on/)).toBeTruthy();
  });

  // R37: the box follows new lines only while the reader is at the bottom; one who scrolled up to read is not pulled
  // down every second. jsdom has no layout, so the box is given a 1000 px transcript in a 100 px window and a scroll
  // position the test can read and move.
  describe("R37: follows new lines only while the reader is at the bottom", () => {
    beforeEach(() => {
      vi.useFakeTimers();
      vi.mocked(api.logs).mockReset().mockResolvedValue([]);
    });
    afterEach(() => vi.useRealTimers());

    function giveLayout() {
      const box = screen.getByLabelText("log");
      const at = { top: 0 };
      Object.defineProperty(box, "scrollHeight", { configurable: true, value: 1000 });
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
    /** The next poll, one second on, brings these lines. */
    const deliver = (...ls: LogLine[]) => {
      vi.mocked(api.logs).mockResolvedValueOnce(ls);
      return act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    };

    it("leaves a reader who scrolled up where they are, and follows one at the bottom", async () => {
      render(<LogsTab />);
      await act(async () => {}); // the poll made on mount brings nothing
      const t = giveLayout();

      t.scrollTo(0); // 1000 - 0 - 100 = 900 px from the end
      await deliver(line(1, "app", "first"));
      expect(screen.getByText("first")).toBeTruthy();
      expect(t.at.top).toBe(0);

      t.scrollTo(900); // 1000 - 900 - 100 = 0 px from the end
      await deliver(line(2, "app", "second"));
      expect(screen.getByText("second")).toBeTruthy();
      expect(screen.getByText("first")).toBeTruthy(); // the lines already shown stay
      expect(t.at.top).toBe(1000);
    });

    it("follows from the start: a tab just opened is at its end before anyone has scrolled", async () => {
      vi.mocked(api.logs).mockResolvedValueOnce([line(1, "app", "first")]);
      render(<LogsTab />);
      const t = giveLayout(); // before the answer to the poll made on mount is applied
      await act(async () => {});
      expect(screen.getByText("first")).toBeTruthy();
      expect(t.at.top).toBe(1000);
    });
  });
});

describe("App shows the three tabs", () => {
  it("opens Models, Requests and Logs from the top bar, one at a time", async () => {
    vi.mocked(api.models).mockResolvedValue(models);
    vi.mocked(api.requests).mockResolvedValue([]);
    vi.mocked(api.logs).mockResolvedValue([]);
    render(<App />);
    await screen.findByText("serving");

    fireEvent.click(screen.getByRole("tab", { name: "Models" }));
    expect(await screen.findByText("3 IN LIBRARY · 2 RUNNABLE")).toBeTruthy();
    expect(screen.getByText("LOADED")).toBeTruthy(); // App passes its overview on: the engine's model is the loaded one

    fireEvent.click(screen.getByRole("tab", { name: "Requests" }));
    expect(await screen.findByText("No requests yet.")).toBeTruthy();
    expect(screen.queryByText(/IN LIBRARY/)).toBeNull();

    fireEvent.click(screen.getByRole("tab", { name: "Logs" }));
    expect(await screen.findByText("UPDATER")).toBeTruthy(); // one of the Logs tab's source filters
    expect(screen.queryByText("No requests yet.")).toBeNull();
  });
});

// R63: a host call that fails must read as a failure, not as an empty library, an empty list or an empty log. The two
// polled tabs repeat their call every second, so they log a failure once per distinct message, not on every tick.
describe("R63: a failed host call is shown, not stated as an empty tab", () => {
  let hostErrors: MockInstance<typeof console.error>;
  beforeEach(() => {
    vi.mocked(api.models).mockReset();
    vi.mocked(api.requests).mockReset();
    vi.mocked(api.logs).mockReset();
    hostErrors = captureHostErrors(); // the "<call> failed:" lines go to this spy; anything else still reaches stderr
  });
  afterEach(() => {
    hostErrors.mockRestore();
    vi.useRealTimers();
  });
  /** One second on: a polled tab makes its next call. */
  const poll = () => act(async () => { await vi.advanceTimersByTimeAsync(1000); });

  describe("ModelsTab", () => {
    it("says why the library could not be read, not '0 IN LIBRARY · 0 RUNNABLE'", async () => {
      vi.mocked(api.models).mockRejectedValue("Command models not found"); // Tauri rejects with the command's error string
      render(<ModelsTab overview={overview} />);
      const notice = await screen.findByText("could not load the models: Command models not found");
      expect(notice.classList.contains("flame-text")).toBe(true);
      expect(screen.getByRole("alert")).toBe(notice); // announced, as ChatTab's notice is
      expect(screen.queryByText(/IN LIBRARY/)).toBeNull();
      expect(hostErrors.mock.calls).toEqual([["models failed:", "Command models not found"]]);
    });
  });

  describe("RequestsTab", () => {
    const request = (model: string): RequestRecord => ({
      id: `req_${model}`, started_ms: Date.now(), source: "chat", endpoint: "/v1/chat/completions", model,
      prompt_tokens: 41, completion_tokens: 5, ttft_ms: 480, total_ms: 730, tokens_per_s: 16.0, status: "ok", dropped_turns: 0,
    });

    it("says why the requests could not be read, not 'No requests yet.', and logs it once however long the poll fails", async () => {
      vi.useFakeTimers();
      const fault = new Error("the monitor is busy"); // a JavaScript fault rejects with an Error, whose message counts
      vi.mocked(api.requests)
        .mockRejectedValueOnce(fault).mockRejectedValueOnce(fault).mockRejectedValueOnce(fault)
        .mockResolvedValueOnce([]);
      render(<RequestsTab />);
      await act(async () => {}); // the call made on mount fails
      const notice = screen.getByText("could not load the requests: the monitor is busy");
      expect(notice.classList.contains("flame-text")).toBe(true);
      expect(screen.queryByText("No requests yet.")).toBeNull();
      expect(screen.queryByText(/THIS SESSION/)).toBeNull();

      await poll();
      await poll();
      expect(api.requests).toHaveBeenCalledTimes(3); // three failing calls...
      expect(hostErrors.mock.calls).toEqual([["requests failed:", "the monitor is busy"]]); // ...one line in the log

      await poll(); // the host answers, and says there are none: now that is what the tab says
      expect(screen.queryByText(/could not load/)).toBeNull();
      expect(screen.getByText("No requests yet.")).toBeTruthy();
      expect(screen.getByText("0 THIS SESSION · NEWEST FIRST")).toBeTruthy();
    });

    it("keeps the rows it has, shows a new message, clears the notice when the host answers, and reports a new outage again", async () => {
      vi.useFakeTimers();
      vi.mocked(api.requests)
        .mockResolvedValueOnce([request("nexus")])
        .mockRejectedValueOnce("boom")
        .mockRejectedValueOnce(new Error("other"))
        .mockResolvedValueOnce([request("llama"), request("nexus")])
        .mockRejectedValueOnce(new Error("other")); // the message it failed with before it recovered
      render(<RequestsTab />);
      await act(async () => {}); // the call made on mount works
      expect(screen.getByText("nexus")).toBeTruthy();
      expect(screen.queryByText(/could not load/)).toBeNull();

      await poll(); // fails: the rows already shown stay, with the reason above them
      expect(screen.getByText("could not load the requests: boom")).toBeTruthy();
      expect(screen.getByText("nexus")).toBeTruthy();
      expect(screen.getByText("1 THIS SESSION · NEWEST FIRST")).toBeTruthy();

      await poll(); // fails with another message: that one is shown, and logged
      expect(screen.getByText("could not load the requests: other")).toBeTruthy();
      expect(screen.queryByText("could not load the requests: boom")).toBeNull();
      expect(hostErrors.mock.calls).toEqual([["requests failed:", "boom"], ["requests failed:", "other"]]);

      await poll(); // works again: the notice goes and the new rows show
      expect(screen.queryByText(/could not load/)).toBeNull();
      expect(screen.getByText("llama")).toBeTruthy();
      expect(screen.getByText("2 THIS SESSION · NEWEST FIRST")).toBeTruthy();

      await poll(); // a failure after that is a new outage, reported again even though the message is the same as before
      expect(screen.getByText("could not load the requests: other")).toBeTruthy();
      expect(hostErrors.mock.calls).toEqual([["requests failed:", "boom"], ["requests failed:", "other"], ["requests failed:", "other"]]);
    });
  });

  describe("LogsTab", () => {
    const logLine = (seq: number, text: string): LogLine => ({ seq, time_ms: Date.now(), source: "app", level: "info", text });

    it("says why the log could not be read, and logs it once however long the poll fails", async () => {
      vi.useFakeTimers();
      vi.mocked(api.logs)
        .mockRejectedValueOnce("boom").mockRejectedValueOnce("boom").mockRejectedValueOnce("boom")
        .mockResolvedValueOnce([]);
      render(<LogsTab />);
      await act(async () => {}); // the call made on mount fails
      const notice = screen.getByText("could not load the log: boom");
      expect(notice.classList.contains("flame-text")).toBe(true);

      await poll();
      await poll();
      expect(api.logs).toHaveBeenCalledTimes(3); // three failing calls...
      expect(hostErrors.mock.calls).toEqual([["logs failed:", "boom"]]); // ...one line in the log

      await poll(); // the host answers, with nothing new
      expect(screen.queryByText(/could not load/)).toBeNull();
    });

    it("keeps the lines it has, asks again from the last one it got, shows a new message, clears the notice when the host answers, and reports a new outage again", async () => {
      vi.useFakeTimers();
      vi.mocked(api.logs)
        .mockResolvedValueOnce([logLine(1, "first")])
        .mockRejectedValueOnce("boom")
        .mockRejectedValueOnce(new Error("other"))
        .mockResolvedValueOnce([logLine(2, "second")])
        .mockRejectedValueOnce(new Error("other")); // the message it failed with before it recovered
      render(<LogsTab />);
      await act(async () => {}); // the call made on mount works
      expect(screen.getByText("first")).toBeTruthy();
      expect(screen.queryByText(/could not load/)).toBeNull();

      await poll(); // fails: the lines already shown stay, with the reason above them
      expect(screen.getByText("could not load the log: boom")).toBeTruthy();
      expect(screen.getByText("first")).toBeTruthy();

      await poll(); // fails with another message: that one is shown, and logged
      expect(screen.getByText("could not load the log: other")).toBeTruthy();
      expect(screen.queryByText("could not load the log: boom")).toBeNull();
      expect(hostErrors.mock.calls).toEqual([["logs failed:", "boom"], ["logs failed:", "other"]]);

      await poll(); // works again: the notice goes and the new line is added to the old
      expect(screen.queryByText(/could not load/)).toBeNull();
      expect(screen.getByText("first")).toBeTruthy();
      expect(screen.getByText("second")).toBeTruthy();

      await poll(); // a failure after that is a new outage, reported again even though the message is the same as before
      expect(screen.getByText("could not load the log: other")).toBeTruthy();
      expect(hostErrors.mock.calls).toEqual([["logs failed:", "boom"], ["logs failed:", "other"], ["logs failed:", "other"]]);

      // A failed call does not move the cursor: every call asked for the lines after the last one it had received.
      expect(vi.mocked(api.logs).mock.calls.map(([after]) => after)).toEqual([0, 1, 1, 1, 2]);
    });
  });
});
