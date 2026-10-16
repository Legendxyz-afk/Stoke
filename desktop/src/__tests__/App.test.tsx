import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { App } from "../App";

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
  });
});
