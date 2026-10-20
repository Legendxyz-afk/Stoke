import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

// Every parse is one call of react-markdown's component with the text to parse as its child, so counting the calls by text
// counts the parses of each piece. The wrapper hands everything on to the real component.
const parsed = vi.hoisted(() => new Map<string, number>());
vi.mock("react-markdown", async (importOriginal) => {
  const real = await importOriginal<typeof import("react-markdown")>();
  return {
    default: (props: { children?: string }) => {
      const text = props.children ?? "";
      parsed.set(text, (parsed.get(text) ?? 0) + 1);
      return real.default(props as never);
    },
  };
});

import { Markdown, StreamingMarkdown } from "../chat/Markdown";
import { MessageRow } from "../chat/MessageView";
import { msg } from "./testApi";

const parses = () => [...parsed.values()].reduce((a, b) => a + b, 0);
const parsedCharacters = () => [...parsed].reduce((sum, [text, n]) => sum + text.length * n, 0);

beforeEach(() => parsed.clear());

// Demo D5: the whole reply was parsed on every flush (19% of one core at 15 tok/s). A block that has ended is parsed once now.
describe("StreamingMarkdown", () => {
  const block0 = "The trend is intact, so wait for a close above the range high before adding to the position today.\n\n";
  const block1 = "- first point to watch\n- second point to watch\n- third point to watch\n\n";
  const block2 = "Call `run(1)` and read `total` once, then watch the volume and the open interest for the rest of the session.";
  const reply = block0 + block1 + block2;

  /** The text of the stream after each of 200 flushes of about the same size. */
  const flushes = Array.from({ length: 200 }, (_, i) => reply.slice(0, Math.ceil(((i + 1) / 200) * reply.length)));

  it("parses a block that has ended once, however long the reply goes on", () => {
    const view = render(<StreamingMarkdown text="" />);
    for (const text of flushes) view.rerender(<StreamingMarkdown text={text} />);
    expect(view.container.textContent).toContain("for the rest of the session.");
    // The first two blocks ended long before the last flush, and were parsed once each in their final form.
    expect(parsed.get(block0), "the first block, as it ended").toBe(1);
    expect(parsed.get(block1), "the second block, as it ended").toBe(1);
    expect(parsed.get(block2), "the last block, as it ended").toBe(1);
    // Every text that was parsed was parsed once: no flush parsed a piece whose text had not changed.
    expect([...parsed.entries()].filter(([, n]) => n !== 1)).toEqual([]);
  });

  it("parses exactly one block on a flush that only grows the last one", () => {
    const view = render(<StreamingMarkdown text="" />);
    const started = flushes.findIndex((text) => text.length > block0.length + block1.length + 1); // block 2 has begun
    expect(started).toBeGreaterThan(50);
    for (const text of flushes.slice(0, started + 1)) view.rerender(<StreamingMarkdown text={text} />);
    for (const text of flushes.slice(started + 1)) {
      const [calls, characters] = [parses(), parsedCharacters()];
      view.rerender(<StreamingMarkdown text={text} />);
      expect(parses() - calls, `the flush that made the text ${text.length} characters long`).toBe(1);
      // ... and the one parse is of the growing block, not of the reply
      expect(parsedCharacters() - characters, `the flush that made the text ${text.length} characters long`).toBeLessThanOrEqual(block2.length);
    }
  });

  it("parses far less than the whole text on every flush, which is what it replaces", () => {
    const streaming = render(<StreamingMarkdown text="" />);
    for (const text of flushes) streaming.rerender(<StreamingMarkdown text={text} />);
    const streamedCharacters = parsedCharacters();
    streaming.unmount();
    parsed.clear();
    const whole = render(<Markdown text="" />);
    for (const text of flushes) whole.rerender(<Markdown text={text} />);
    const wholeCharacters = parsedCharacters();
    expect(streamedCharacters).toBeLessThan(wholeCharacters / 2);
  });

  it("shows what one Markdown shows, block by block", () => {
    const text = [
      "# Plan",
      "Wait for a close above the range high, then add:\n\n- size: **1%**\n- stop: below the range low\n\n- target: the prior high",
      "```js\nconst n = 3;\n\nfor (let i = 0; i < n; i++) {\n  total += i;\n}\n```",
      "$$\nx^2\n\n+ y^2\n$$",
      "| k | v |\n|---|---|\n| a | 1 |\n| b | 2 |",
      "> note: this is not advice",
      "Entry $68,400, target $72,000.",
    ].join("\n\n");
    // The whitespace the parser puts between two block elements is not shown; compare without it.
    const html = (node: React.ReactElement) => render(node).container.innerHTML.replace(/>\s+</g, "><");
    expect(html(<StreamingMarkdown text={text} />)).toBe(html(<Markdown text={text} />));
    expect(html(<StreamingMarkdown text={text} />)).toContain("katex"); // the maths block was one block, and was rendered as maths
  });

  it("renders nothing for no text", () => {
    expect(render(<StreamingMarkdown text="" />).container.querySelector(".md")?.childElementCount).toBe(0);
  });
});

describe("Markdown (a settled message)", () => {
  it("is ONE parse of the whole text, with the exact semantics of the whole", () => {
    const text = "Pick [one][a].\n\nSecond block.\n\n[a]: https://example.com\n";
    const view = render(<Markdown text={text} />);
    expect(parsed.get(text)).toBe(1);
    expect(parses()).toBe(1);
    expect(view.container.querySelector(".md-link")?.textContent).toBe("one"); // the reference resolves across blocks
  });
});

describe("MessageRow", () => {
  const reply = "First paragraph of the reply.\n\nSecond paragraph of the reply.";
  const row = (liveText: string | null) => (
    <MessageRow
      mv={{ message: msg({ id: "a1", role: "assistant", content: reply, status: liveText === null ? "done" : "streaming" }), version: 1, versions: 1, sibling_ids: ["a1"] }}
      liveText={liveText}
      pending={null}
      onVersion={() => {}}
    />
  );

  it("draws the reply that is being written block by block, and a settled message as one parse", () => {
    render(row(reply));
    expect([...parsed.keys()]).toEqual(["First paragraph of the reply.\n\n", "Second paragraph of the reply."]);
    parsed.clear();
    render(row(null));
    expect([...parsed.keys()]).toEqual([reply]);
  });

  it("hands the reply that has ended to ONE parse of the whole text, once", () => {
    const view = render(row(reply));
    parsed.clear();
    view.rerender(row(null));
    expect([...parsed]).toEqual([[reply, 1]]);
  });
});
