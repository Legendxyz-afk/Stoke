import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { UpdateBadge } from "../components/UpdateBadge";
import { api } from "../ipc/api";
import type { UpdateState } from "../ipc/bindings/UpdateState";

beforeEach(() => vi.clearAllMocks());

describe("UpdateBadge", () => {
  it("offers a verified update for install on restart", async () => {
    vi.mocked(api.updateState).mockResolvedValue({ state: "ready", version: "0.0.2" });
    render(<UpdateBadge />);
    fireEvent.click(await screen.findByText(/0\.0\.2 ready/));
    await waitFor(() => expect(api.installUpdate).toHaveBeenCalled());
  });

  it("says a refused update was refused, and can check again", async () => {
    vi.mocked(api.updateState).mockResolvedValue({ state: "refused", version: "0.0.2", reason: "the signature did not verify" });
    render(<UpdateBadge />);
    fireEvent.click(await screen.findByText(/0\.0\.2 refused/));
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
