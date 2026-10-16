import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Markdown } from "../chat/Markdown";

describe("Markdown", () => {
  it("renders GFM tables, highlighted code and maths", () => {
    const text = "**HOLD** at ₹68,400\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```rust\nfn main() {}\n```\n\n$x^2$";
    const { container } = render(<Markdown text={text} />);
    expect(container.querySelector("strong")?.textContent).toBe("HOLD");
    expect(container.querySelector("table")).not.toBeNull();
    expect(container.querySelector("pre code.hljs")).not.toBeNull();
    expect(container.querySelector(".katex")).not.toBeNull();
    expect(container.textContent).toContain("₹68,400");
  });

  it("never renders raw HTML, fetches an image or makes a link that navigates", () => {
    const text = '<img src=x onerror="alert(1)"><script>alert(2)</script>\n\n![chart](https://example.com/x.png) and [site](https://example.com)';
    const { container } = render(<Markdown text={text} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("a")).toBeNull();
    expect(container.textContent).toContain("[image: chart]");
    expect(container.textContent).toContain("site");
  });
});
