import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "../App";

describe("App", () => {
  it("shows the lorren wordmark", () => {
    render(<App />);
    expect(screen.getByText("lorren")).toBeTruthy();
  });
});
