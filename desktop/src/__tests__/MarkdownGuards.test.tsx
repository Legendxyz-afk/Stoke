import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import lock from "../../package-lock.json";

// react-markdown builds a new processor on every render, so this plugin is attached once per parse of a message.
// Counting the attachments counts the parses. The wrapper hands everything on to the real plugin.
const attached = vi.hoisted(() => ({ n: 0 }));
vi.mock("rehype-highlight", async (importOriginal) => {
  const real = await importOriginal<typeof import("rehype-highlight")>();
  return {
    default: function countingHighlight(...args: Parameters<typeof real.default>) {
      attached.n += 1;
      return real.default(...args);
    },
  };
});

import { Markdown } from "../chat/Markdown";

describe("Markdown re-rendering", () => {
  // A parse costs ~6 ms for a 900-character reply. ChatTab re-renders on every keystroke (it owns the draft, R36b)
  // and on every streamed flush (~30 a second), and each of those must not re-parse every settled message.
  it("parses a message again only when its text changes", () => {
    const { rerender } = render(<Markdown text="**a**" />);
    const first = attached.n;
    expect(first).toBeGreaterThan(0);
    rerender(<Markdown text="**a**" />);
    expect(attached.n).toBe(first);
    rerender(<Markdown text="**b**" />);
    expect(attached.n).toBeGreaterThan(first);
  });
});

describe("KaTeX", () => {
  // Markdown.tsx imports the stylesheet from the top-level `katex`, but the markup is made by the copy rehype-katex
  // resolves. KaTeX 0.19 renamed ten structural classes (base, strut, sizing, root, overline, underline, accent,
  // overlay, stretchy, rule -> katex-*), so a 0.19 stylesheet leaves 0.16 markup unstyled: `.base` and `.strut` wrap
  // every formula. The maths test in Markdown.test.tsx cannot see that, so pin the pairing itself: the lockfile,
  // which decides what every other machine installs, must hold exactly one KaTeX for the stylesheet and the renderer.
  it("has one copy in the install, so the stylesheet Markdown imports is the renderer's", () => {
    const packages = (lock as { packages: Record<string, { version: string }> }).packages;
    const copies = Object.entries(packages)
      .filter(([where]) => /(^|\/)node_modules\/katex$/.test(where))
      .map(([where, p]) => `${where}@${p.version}`);
    expect(copies, `KaTeX copies in package-lock.json: ${copies.join(", ")}`).toHaveLength(1);
  });
});

describe("chat.css", () => {
  // Vitest blanks every CSS import (`?raw` too) and this project has no Node types, so the stylesheet is read as a file.
  // (Node's fs rejects the `URL` class that the jsdom environment installs, so the path is built from the string.)
  const readStylesheet = async () => {
    const [fs, url] = ["node:fs", "node:url"];
    const { readFileSync } = (await import(/* @vite-ignore */ fs)) as { readFileSync(file: string, encoding: "utf8"): string };
    const { fileURLToPath } = (await import(/* @vite-ignore */ url)) as { fileURLToPath(file: string): string };
    return readFileSync(fileURLToPath(import.meta.url).replace(/__tests__[\\/][^\\/]+$/, "chat/chat.css"), "utf8");
  };

  // github-dark.css gives `pre code.hljs` its own background and 1em of padding. Without this rule they are painted
  // over the design's code box (`.md pre`: --code background, 10px 12px padding), which then shows as a box in a box.
  it("takes highlight.js's own background and padding off a code block, so the design's code box shows", async () => {
    expect(await readStylesheet()).toMatch(/\.md pre code\.hljs\s*\{[^}]*\bbackground:\s*transparent;[^}]*\bpadding:\s*0;/);
  });
});
