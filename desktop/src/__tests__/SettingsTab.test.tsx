import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { api } from "../ipc/api";
import { SettingsTab } from "../tabs/SettingsTab";

beforeEach(() => vi.resetAllMocks());

// resetAllMocks clears the implementations too, so the defaults are put back here.
beforeEach(() => {
  vi.mocked(api.updateMode).mockResolvedValue("auto");
  vi.mocked(api.setUpdateMode).mockResolvedValue(undefined);
});

const radio = (name: RegExp) => screen.getByRole("radio", { name });
const checked = () => screen.getAllByRole("radio").filter((r) => r.getAttribute("aria-checked") === "true");

describe("SettingsTab", () => {
  it("settings_shows_the_updates_section_with_the_stored_mode", async () => {
    render(<SettingsTab />);
    await waitFor(() => expect(checked().length).toBe(1));
    expect(checked()[0].textContent).toContain("AUTO");
    expect(screen.getByText("WHEN A NEW VERSION COMES OUT")).toBeTruthy();
    expect(screen.getByText("DEFAULT")).toBeTruthy();
    expect(screen.getByText("Downloads updates in the background and installs them when you restart.")).toBeTruthy();
    expect(screen.getByText("Tells you when there is one, and downloads only when you say.")).toBeTruthy();
    expect(screen.getByText("Never checks.")).toBeTruthy();
    expect(screen.queryByText(/Could not save/)).toBeNull();
  });

  it("shows the stored mode when it is not the default", async () => {
    vi.mocked(api.updateMode).mockResolvedValue("notify");
    render(<SettingsTab />);
    await waitFor(() => expect(checked()[0]?.textContent).toContain("NOTIFY"));
  });

  it("selects nothing until the host has answered", () => {
    vi.mocked(api.updateMode).mockReturnValue(new Promise(() => {}));
    render(<SettingsTab />);
    expect(checked().length).toBe(0);
  });

  it("choosing_off_calls_setUpdateMode_once_and_shows_it", async () => {
    render(<SettingsTab />);
    await waitFor(() => expect(checked().length).toBe(1));
    fireEvent.click(radio(/OFF/));
    expect(checked()[0].textContent).toContain("OFF");
    await waitFor(() => expect(api.setUpdateMode).toHaveBeenCalledTimes(1));
    expect(api.setUpdateMode).toHaveBeenCalledWith("off");
    fireEvent.click(radio(/OFF/)); // the selected choice: nothing
    expect(api.setUpdateMode).toHaveBeenCalledTimes(1);
  });

  it("a_failed_set_is_shown_and_the_choice_goes_back", async () => {
    vi.mocked(api.setUpdateMode).mockRejectedValue("the store is read-only");
    render(<SettingsTab />);
    await waitFor(() => expect(checked().length).toBe(1));
    fireEvent.click(radio(/NOTIFY/));
    expect((await screen.findByRole("alert")).textContent).toBe("Could not save the choice: the store is read-only");
    expect(checked()[0].textContent).toContain("AUTO");
  });
});
