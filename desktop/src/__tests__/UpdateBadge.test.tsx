import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { StatusBar } from "../components/StatusBar";
import { UpdateBadge } from "../components/UpdateBadge";
import { api } from "../ipc/api";
import type { UpdateState } from "../ipc/bindings/UpdateState";
import { captureHostErrors, overview } from "./testApi";

beforeEach(() => vi.clearAllMocks());

describe("UpdateBadge", () => {
  it("offers a verified update for install on restart", async () => {
    vi.mocked(api.updateState).mockResolvedValue({ state: "ready", version: "0.0.2" });
    render(<UpdateBadge />);
    fireEvent.click(await screen.findByText(/0\.0\.2 ready/));
    await waitFor(() => expect(api.installUpdate).toHaveBeenCalled());
  });

  it("says a refused update was refused, and can check again", async () => {
    vi.mocked(api.updateState).mockResolvedValue({
      state: "refused",
      version: "0.0.2",
      label: "installer altered or damaged",
      reason: "the downloaded installer does not match its signature, so it changed after it was signed",
    });
    render(<UpdateBadge />);
    fireEvent.click(await screen.findByText(/0\.0\.2 refused/));
    await waitFor(() => expect(api.checkUpdate).toHaveBeenCalled());
  });

  // Demo D7: the badge said "refused: signature did not verify" and its tooltip said "the signature did not verify (The
  // signature verification failed)", which is the same thing twice. The badge now gives the cause in a few words, and the
  // tooltip the same cause as one sentence, and that nothing was installed.
  it("says why in a few words in the badge, and in one sentence, with what that means, in the tooltip", async () => {
    vi.mocked(api.updateState).mockResolvedValue({
      state: "refused",
      version: "0.0.2",
      label: "untrusted signing key",
      reason: "the update is signed with a key this Lorren does not trust",
    });
    render(<UpdateBadge />);
    const badge = await screen.findByRole("button", { name: /0\.0\.2 refused/ });
    expect(badge.textContent).toBe("update 0.0.2 refused · untrusted signing key · check again");
    expect(badge.getAttribute("title")).toBe("the update is signed with a key this Lorren does not trust; nothing was installed");
    expect(badge.classList.contains("flame-text")).toBe(true);
    fireEvent.click(badge);
    await waitFor(() => expect(api.checkUpdate).toHaveBeenCalled());
  });

  it("shows nothing when updates are not configured", async () => {
    // Before any state arrives the badge also renders nothing, so first wait until the not-configured
    // state has been delivered and applied; only then does an empty badge mean anything.
    const delivered = Promise.resolve<UpdateState>({ state: "not_configured" });
    vi.mocked(api.updateState).mockReturnValue(delivered);
    const { container } = render(<UpdateBadge />);
    await act(async () => {
      await delivered;
    });
    expect(api.updateState).toHaveBeenCalled();
    expect(container.textContent).toBe("");
  });
});

const settle = () => new Promise((resolve) => setTimeout(resolve, 20));

/** Node's unhandledRejection events during a test (the same helper as in ChatTab.test.tsx). Vitest also fails the whole
 *  run on one; this makes the test say so. */
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

// R64: a host call the badge makes can fail like any other (a misnamed or unregistered command, an ACL, or, for the
// install, the plugin refusing). The reason is shown in the badge, ahead of its action, and logged once; a state that
// arrives takes it away. The badge must not swallow it, and must not look like "nothing to report".
describe("UpdateBadge when a host call fails (R64)", () => {
  let hostErrors: MockInstance<typeof console.error>;
  /** The callback the badge handed to onUpdateState: calling it is the host emitting an update-state event. */
  let emit!: (s: UpdateState) => void;
  const ready: UpdateState = { state: "ready", version: "0.0.2" };
  const upToDate: UpdateState = { state: "up_to_date", version: "0.0.1" };
  /** The badge where App puts it: Task 16's end-to-end test presses the last `.statusbar button`. */
  const inStatusBar = () =>
    render(
      <StatusBar overview={overview}>
        <UpdateBadge />
      </StatusBar>,
    );

  beforeEach(() => {
    // Nothing an earlier test left on the mocks (a rejection, a one-off result) may reach these tests.
    vi.mocked(api.updateState).mockReset().mockImplementation(async () => ({ state: "not_configured" }));
    vi.mocked(api.checkUpdate).mockReset().mockImplementation(async () => {});
    vi.mocked(api.installUpdate).mockReset().mockImplementation(async () => {});
    vi.mocked(api.onUpdateState).mockReset().mockImplementation(async (cb) => {
      emit = cb;
      return () => {};
    });
    hostErrors = captureHostErrors(); // the "<call> failed:" lines go to this spy; anything else still reaches stderr
  });
  afterEach(() => {
    hostErrors.mockRestore();
  });

  it("installUpdate(): the reason is shown before the button and logged once, the button stays the last one and can be pressed again", async () => {
    const reason = "update 0.0.2 could not be installed: Access is denied. (os error 5)";
    vi.mocked(api.updateState).mockResolvedValue(ready);
    vi.mocked(api.installUpdate).mockRejectedValue(reason);
    const { container } = inStatusBar();
    const button = await screen.findByText(/0\.0\.2 ready · RESTART TO INSTALL/);

    fireEvent.click(button);
    const notice = await screen.findByText(`could not install the update: ${reason}`);
    expect(notice.tagName).toBe("SPAN"); // a span: the badge's action stays the last button in the bar
    expect(notice.classList.contains("flame-text")).toBe(true);
    expect(notice.getAttribute("title")).toBe(`could not install the update: ${reason}`); // the line is cut to fit the bar; this has all of it
    expect(notice.compareDocumentPosition(button) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy(); // before the button
    const buttons = container.querySelectorAll(".statusbar button");
    expect(buttons[buttons.length - 1]).toBe(button);
    expect(hostErrors.mock.calls).toEqual([["installUpdate failed:", reason]]);

    fireEvent.click(button); // the state is still Ready, so the host can be asked again; the same reason is not logged again
    await waitFor(() => expect(api.installUpdate).toHaveBeenCalledTimes(2));
    await act(async () => {});
    expect(screen.getByText(`could not install the update: ${reason}`)).toBeTruthy();
    expect(hostErrors.mock.calls).toEqual([["installUpdate failed:", reason]]);
  });

  // R64 and demo D7: the e2e presses the last `.statusbar button`, which has to be the badge's action, refused or not.
  it("a refused update's badge is the last button of the status bar", async () => {
    vi.mocked(api.updateState).mockResolvedValue({ state: "refused", version: "0.0.2", label: "unsigned", reason: "the update has no signature" });
    const { container } = inStatusBar();
    const badge = await screen.findByText("update 0.0.2 refused · unsigned · check again");
    const buttons = container.querySelectorAll(".statusbar button");
    expect(buttons[buttons.length - 1]).toBe(badge);
  });

  it("checkUpdate(): the reason is shown before the button and logged once", async () => {
    vi.mocked(api.updateState).mockResolvedValue(upToDate);
    vi.mocked(api.checkUpdate).mockRejectedValue("Command check_update not found");
    const { container } = inStatusBar();
    const button = await screen.findByText(/up to date \(0\.0\.1\)/);

    fireEvent.click(button);
    const notice = await screen.findByText("could not check for updates: Command check_update not found");
    expect(notice.tagName).toBe("SPAN");
    expect(notice.classList.contains("flame-text")).toBe(true);
    expect(notice.compareDocumentPosition(button) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    const buttons = container.querySelectorAll(".statusbar button");
    expect(buttons[buttons.length - 1]).toBe(button);
    expect(hostErrors.mock.calls).toEqual([["checkUpdate failed:", "Command check_update not found"]]);
  });

  it("updateState(): the reason is shown, and it is all the badge shows, so the badge is not empty", async () => {
    vi.mocked(api.updateState).mockRejectedValue("Command update_state not found");
    const { container } = render(<UpdateBadge />);
    const notice = await screen.findByText("could not read the update state: Command update_state not found");
    expect(notice.tagName).toBe("SPAN");
    expect(notice.classList.contains("flame-text")).toBe(true);
    expect(container.textContent).toBe("could not read the update state: Command update_state not found");
    expect(screen.getByRole("alert")).toBe(notice); // announced, as the tabs' notices and ChatTab's are
    expect(container.querySelector("button")).toBeNull();
    expect(hostErrors.mock.calls).toEqual([["updateState failed:", "Command update_state not found"]]);
  });

  it("onUpdateState(): the reason is shown and logged, and unmounting after it rejects nothing", async () => {
    const unhandled = watchUnhandledRejections();
    try {
      vi.mocked(api.onUpdateState).mockRejectedValueOnce("event:default not allowed by ACL");
      const view = render(<UpdateBadge />);
      const notice = await screen.findByText("could not follow the update state: event:default not allowed by ACL");
      expect(notice.tagName).toBe("SPAN");
      expect(notice.classList.contains("flame-text")).toBe(true);
      expect(hostErrors.mock.calls).toEqual([["onUpdateState failed:", "event:default not allowed by ACL"]]);
      view.unmount(); // the cleanup runs un.then((f) => f()); with no function to call that would be an unhandled rejection
      await settle(); // Node reports an unhandled rejection once the microtasks have drained
      expect(unhandled.seen).toEqual([]);
    } finally {
      unhandled.stop();
    }
  });

  it("a state event after a failure takes the failure line away, and a failure after that is shown and logged again", async () => {
    vi.mocked(api.updateState).mockRejectedValueOnce("Command update_state not found");
    vi.mocked(api.installUpdate).mockRejectedValue("the installer could not be started");
    vi.mocked(api.checkUpdate).mockRejectedValue("Command check_update not found");
    const { container } = render(<UpdateBadge />);
    await screen.findByText("could not read the update state: Command update_state not found");

    act(() => emit(ready)); // the snapshot failed, then a state arrives: the badge has something to show, so the failure goes
    expect(screen.queryByText(/could not read/)).toBeNull();
    expect(container.textContent).toBe("update 0.0.2 ready · RESTART TO INSTALL");

    fireEvent.click(screen.getByText(/RESTART TO INSTALL/));
    await screen.findByText("could not install the update: the installer could not be started");
    act(() => emit(upToDate)); // another state event takes the install failure away as well
    expect(screen.queryByText(/could not install/)).toBeNull();
    expect(container.textContent).toBe("up to date (0.0.1)");

    fireEvent.click(screen.getByText(/up to date/));
    await screen.findByText("could not check for updates: Command check_update not found");
    act(() => emit(ready)); // and the check failure
    expect(screen.queryByText(/could not check/)).toBeNull();
    expect(container.textContent).toBe("update 0.0.2 ready · RESTART TO INSTALL");

    fireEvent.click(screen.getByText(/RESTART TO INSTALL/)); // the same failure again is a new outage, not a repeat of the one logged
    await screen.findByText("could not install the update: the installer could not be started");
    expect(hostErrors.mock.calls).toEqual([
      ["updateState failed:", "Command update_state not found"],
      ["installUpdate failed:", "the installer could not be started"],
      ["checkUpdate failed:", "Command check_update not found"],
      ["installUpdate failed:", "the installer could not be started"],
    ]);
  });

  it.each([
    { press: "install", state: ready, label: /RESTART TO INSTALL/, call: () => api.installUpdate, notice: "could not install the update: the host said no" },
    { press: "check", state: upToDate, label: /up to date/, call: () => api.checkUpdate, notice: "could not check for updates: the host said no" },
  ])("$press: a press that works the next time takes its own failure line away", async ({ state, label, call, notice }) => {
    vi.mocked(api.updateState).mockResolvedValue(state);
    vi.mocked(call()).mockRejectedValueOnce("the host said no").mockResolvedValue(undefined);
    render(<UpdateBadge />);
    const button = await screen.findByText(label);
    fireEvent.click(button);
    await screen.findByText(notice);

    fireEvent.click(button); // no state event in between: it is the answer that works which clears the line
    await waitFor(() => expect(screen.queryByText(/could not/)).toBeNull());
    expect(call()).toHaveBeenCalledTimes(2);
    expect(screen.getByText(label)).toBe(button);
  });

  it("a snapshot that arrives after the listener failed does not hide that the badge can no longer follow the state", async () => {
    let snapshot!: (s: UpdateState) => void;
    vi.mocked(api.updateState).mockReturnValue(new Promise<UpdateState>((resolve) => (snapshot = resolve)));
    vi.mocked(api.onUpdateState).mockRejectedValueOnce("event:default not allowed by ACL");
    render(<UpdateBadge />);
    await screen.findByText("could not follow the update state: event:default not allowed by ACL");

    await act(async () => snapshot(upToDate)); // only an event shows that following works; a snapshot is a one-off read
    expect(screen.getByText(/up to date \(0\.0\.1\)/)).toBeTruthy();
    expect(screen.getByText("could not follow the update state: event:default not allowed by ACL")).toBeTruthy();
  });
});

// The badge subscribes before it reads. Reading first loses an event emitted before the listener is registered, and lets a
// snapshot that is answered after an event overwrite the newer state with an older one. Each of these was run against the
// badge as it was (snapshot first) and failed there.
describe("UpdateBadge: the listener first, then the snapshot", () => {
  /** The callback the badge handed to onUpdateState: calling it is the host emitting an update-state event. */
  let emit!: (s: UpdateState) => void;
  const ready: UpdateState = { state: "ready", version: "0.0.2" };
  const upToDate: UpdateState = { state: "up_to_date", version: "0.0.1" };

  beforeEach(() => {
    vi.mocked(api.updateState).mockReset().mockImplementation(async () => ({ state: "not_configured" }));
    vi.mocked(api.onUpdateState).mockReset().mockImplementation(async (cb) => {
      emit = cb;
      return () => {};
    });
  });

  it("an event that arrives before the snapshot is answered is not overwritten by it", async () => {
    let answer!: (s: UpdateState) => void;
    vi.mocked(api.updateState).mockReturnValue(new Promise<UpdateState>((resolve) => (answer = resolve)));
    const { container } = render(<UpdateBadge />);
    await waitFor(() => expect(api.updateState).toHaveBeenCalled()); // the snapshot is on its way
    act(() => emit(ready)); // the host emits while it is
    expect(container.textContent).toBe("update 0.0.2 ready · RESTART TO INSTALL");

    await act(async () => answer(upToDate)); // the older answer lands
    expect(container.textContent).toBe("update 0.0.2 ready · RESTART TO INSTALL");
  });

  it("registers the listener before it asks for the snapshot, and asks for it once the listener is in place", async () => {
    let registered!: (unlisten: () => void) => void;
    vi.mocked(api.onUpdateState).mockImplementation(() => new Promise((resolve) => (registered = resolve)));
    render(<UpdateBadge />);
    await act(async () => {}); // whatever the badge is going to do has been done
    expect(api.onUpdateState).toHaveBeenCalledTimes(1);
    expect(api.updateState).not.toHaveBeenCalled(); // the listener is not registered yet, so nothing is read

    await act(async () => registered(() => {}));
    expect(api.updateState).toHaveBeenCalledTimes(1);
  });

  it("still reads the snapshot when the listener could not be registered", async () => {
    vi.mocked(api.onUpdateState).mockRejectedValue("event:default not allowed by ACL");
    vi.mocked(api.updateState).mockResolvedValue(upToDate);
    const hostErrors = captureHostErrors();
    try {
      render(<UpdateBadge />);
      expect(await screen.findByText(/up to date \(0\.0\.1\)/)).toBeTruthy(); // a one-off read is better than none
      expect(screen.getByText("could not follow the update state: event:default not allowed by ACL")).toBeTruthy();
    } finally {
      hostErrors.mockRestore();
    }
  });

  it("an answer that arrives with no event before it is shown", async () => {
    vi.mocked(api.updateState).mockResolvedValue(ready);
    const { container } = render(<UpdateBadge />);
    await screen.findByText(/0\.0\.2 ready/);
    expect(container.textContent).toBe("update 0.0.2 ready · RESTART TO INSTALL");
  });
});

describe("UpdateBadge, updates off and an update available", () => {
  beforeEach(() => {
    vi.mocked(api.onUpdateState).mockReset().mockImplementation(async () => () => {});
    vi.mocked(api.downloadUpdate).mockReset().mockImplementation(async () => {});
  });

  it("the_badge_says_updates_off_in_off_and_offers_nothing", async () => {
    vi.mocked(api.updateState).mockResolvedValue({ state: "off" });
    render(<UpdateBadge />);
    expect(await screen.findByText("updates off")).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("the_badge_offers_DOWNLOAD_for_available_and_calls_downloadUpdate", async () => {
    vi.mocked(api.updateState).mockResolvedValue({ state: "available", version: "0.1.1" });
    render(<UpdateBadge />);
    const b = await screen.findByRole("button", { name: "update 0.1.1 available · DOWNLOAD" });
    fireEvent.click(b);
    await waitFor(() => expect(api.downloadUpdate).toHaveBeenCalledTimes(1));
  });

  it("shows why a download could not start", async () => {
    const errors = captureHostErrors();
    vi.mocked(api.updateState).mockResolvedValue({ state: "available", version: "0.1.1" });
    vi.mocked(api.downloadUpdate).mockRejectedValue("no update is waiting to be downloaded");
    render(<UpdateBadge />);
    fireEvent.click(await screen.findByRole("button", { name: /DOWNLOAD/ }));
    expect((await screen.findByRole("alert")).textContent).toBe(
      "could not download the update: no update is waiting to be downloaded",
    );
    errors.mockRestore();
  });
});
