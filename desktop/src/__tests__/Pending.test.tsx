import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HINT, PHASE_WORDS, PendingStatus, phaseWord, statusLine } from "../chat/Pending";

// Demo D2: a reply that has not produced its first token shows that work is going on. These are the pieces of that
// status line; ChatTab.test.tsx has it in the transcript.

const WORDS = [
  "thinking",
  "reading the chat",
  "pondering",
  "mulling it over",
  "connecting the dots",
  "weighing it up",
  "piecing it together",
  "working it out",
  "gathering thoughts",
];

describe("the words of a pending reply", () => {
  it("are nine, in this order, and change every 2.5 s, wrapping round", () => {
    expect([...PHASE_WORDS]).toEqual(WORDS); // spelled out here on purpose: the list and its order are the requirement
    WORDS.forEach((word, i) => {
      expect(phaseWord(i * 2500), `from ${i * 2500} ms`).toBe(word);
      expect(phaseWord(i * 2500 + 2499), `until ${i * 2500 + 2499} ms`).toBe(word);
    });
    expect(phaseWord(9 * 2500)).toBe("thinking");
    expect(phaseWord(10 * 2500 + 1)).toBe("reading the chat");
    expect(phaseWord(-5)).toBe("thinking"); // a clock that stepped back
  });

  it("are followed by whole seconds from the first one on, and give way to `loading <model>` while the engine is not on the chat's model", () => {
    expect(statusLine(0, null)).toBe("thinking");
    expect(statusLine(999, null)).toBe("thinking");
    expect(statusLine(1000, null)).toBe("thinking · 1 s");
    expect(statusLine(3400, null)).toBe("reading the chat · 3 s");
    expect(statusLine(20_100, null)).toBe("gathering thoughts · 20 s"); // the ninth word
    expect(statusLine(59_999, null)).toBe("weighing it up · 59 s"); // 23 steps in: 23 mod 9 is the sixth word
    expect(statusLine(400, "nexus")).toBe("loading nexus");
    expect(statusLine(3400, "nexus")).toBe("loading nexus · 3 s");
  });
});

describe("PendingStatus", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });
  const tick = (ms: number) =>
    act(async () => {
      await vi.advanceTimersByTimeAsync(ms);
    });

  it("counts the seconds from when Send was pressed, turns the word, and adds the hint at 15 s", async () => {
    render(<PendingStatus since={Date.now()} loading={null} />);
    expect(screen.getByText("thinking")).toBeTruthy();
    await tick(1000);
    expect(screen.getByText("thinking · 1 s")).toBeTruthy();
    await tick(2000);
    expect(screen.getByText("reading the chat · 3 s")).toBeTruthy();
    await tick(11_000); // 14 s
    expect(screen.getByText(/ · 14 s$/)).toBeTruthy();
    expect(screen.queryByText(HINT)).toBeNull();
    await tick(1000); // 15 s
    expect(screen.getByText(/ · 15 s$/)).toBeTruthy();
    expect(screen.getByText(HINT).classList.contains("dim")).toBe(true);
    expect(HINT).toBe("long chats take longer: the model re-reads the last ~1,500 tokens of the chat before it answers");
  });

  it("counts from `since`, not from when it appeared", () => {
    render(<PendingStatus since={Date.now() - 4000} loading={null} />);
    expect(screen.getByText("reading the chat · 4 s")).toBeTruthy();
  });

  it("says `loading <model>` for as long as it is told so", async () => {
    const { rerender } = render(<PendingStatus since={Date.now()} loading="nexus" />);
    await tick(3000);
    expect(screen.getByText("loading nexus · 3 s")).toBeTruthy();
    rerender(<PendingStatus since={Date.now() - 3000} loading={null} />);
    expect(screen.getByText("reading the chat · 3 s")).toBeTruthy();
  });

  it("runs one interval, and stops it when it goes away", () => {
    const view = render(<PendingStatus since={Date.now()} loading={null} />);
    expect(vi.getTimerCount()).toBe(1);
    view.unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe("chat.css", () => {
  // The same reading of the stylesheet as in MarkdownGuards.test.tsx: Vitest blanks CSS imports.
  const readStylesheet = async () => {
    const [fs, url] = ["node:fs", "node:url"];
    const { readFileSync } = (await import(/* @vite-ignore */ fs)) as { readFileSync(file: string, encoding: "utf8"): string };
    const { fileURLToPath } = (await import(/* @vite-ignore */ url)) as { fileURLToPath(file: string): string };
    return readFileSync(fileURLToPath(import.meta.url).replace(/__tests__[\\/][^\\/]+$/, "chat/chat.css"), "utf8");
  };

  it("pulses the status text on opacity, 1.6 s a round, and stops for a reader who asked for less motion", async () => {
    const css = await readStylesheet();
    expect(css).toMatch(/\.pending-phase\s*\{[^}]*\banimation:\s*pending-pulse\s+1\.6s\b[^}]*\binfinite\b/);
    expect(css).toMatch(/@keyframes pending-pulse\s*\{[^}]*\bopacity:/);
    expect(css).toMatch(/@media \(prefers-reduced-motion: reduce\)\s*\{[^}]*\.pending-phase\s*\{[^}]*\banimation:\s*none\b/);
  });
});
