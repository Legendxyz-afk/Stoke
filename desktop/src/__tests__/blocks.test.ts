import { describe, expect, it } from "vitest";
import { splitBlocks } from "../chat/blocks";

// Demo D5: a reply that is still streaming is cut into its top-level Markdown blocks, so that a block that has ended is
// parsed once and only the block still growing is parsed on each flush. The cut must lose nothing, and must not cut
// through a construct that is one block to the reader.

const SAMPLES: Record<string, string> = {
  paragraphs: "First paragraph.\n\nSecond paragraph.\n\nThird.",
  leadingAndTrailingBlankLines: "\n\nA\n\nB\n\n\n",
  list: "- one\n- two\n\n- three (loose)\n\nAfter the list.",
  numbered: "1. a\n\n2. b\n\n3. c\n",
  fence: "Intro\n\n```js\nconst a = 1;\n\nconst b = 2;\n```\n\nOutro",
  openFence: "Intro\n\n```js\nconst a = 1;\n\nconst b = 2;",
  math: "Before\n\n$$\nx^2\n\n+ y^2\n$$\n\nAfter",
  table: "| a | b |\n|---|---|\n| 1 | 2 |\n\nText",
  headings: "# Title\n\nbody\n\n---\n\n> quote\n> more\n\ntail",
  tabs: "text\n\n\tindented with a tab\n\n- item\n\t- nested by tab\n",
  crlf: "one\r\n\r\ntwo\r\n\r\n- a\r\n- b\r\n",
  loneCarriageReturns: "one\r\rtwo",
  references: "See [the docs][d].\n\n[d]: https://example.com\n",
  empty: "",
  oneLine: "no newline at all",
  prices: "Entry $68,400, target $72,000, stop $67,850.\n\nNext.",
  unicode: "₹68,400 🚀\n\nदूसरा अनुच्छेद\n\n",
  html: "<div>hi</div>\n\ntext",
  whitespaceOnly: "  \n \n\t\n",
};

/** A reply with every kind of block in it (and a `$$` block with a blank line, which the lexer alone would cut). */
const REPLY = [
  "# Plan",
  "Wait for a close above the range high, then add:\n\n- size: 1%\n- stop: below the range low\n\n- target: the prior high",
  "1. entry\n\n2. stop\n\n3. exit",
  "```js\nconst n = 3;\n\nfor (let i = 0; i < n; i++) {\n\tconsole.log(i);\n}\n```",
  "$$\nR = \\frac{a}{b}\n\n+ c\n$$",
  "| k | v |\n|---|---|\n| a | 1 |\n| b | 2 |",
  "> note: this is not advice\n> at all",
  "Done. [ref][1]\n\n[1]: https://example.com\n",
].join("\n\n");

describe("splitBlocks", () => {
  it("gives the text back byte for byte, whatever the text holds", () => {
    for (const [name, text] of Object.entries(SAMPLES)) {
      expect(splitBlocks(text).join(""), name).toBe(text);
    }
  });

  it("gives every prefix of a reply back, as a stream shows it: the cut loses nothing at any length", () => {
    for (let n = 0; n <= REPLY.length; n++) {
      const prefix = REPLY.slice(0, n);
      expect(splitBlocks(prefix).join(""), `a prefix of ${n} characters`).toBe(prefix);
    }
  });

  // What makes a streamed reply cheap: a block that has ended does not change as the reply goes on, so it is not parsed again.
  // The last block of the text so far grows. The one before it can still take the last one in, when what follows turns out to
  // continue it (the second item of a numbered list is a paragraph until its "." is typed); nothing before those two changes.
  it("changes nothing but the last two blocks as a reply grows, and the second to last hardly ever", () => {
    let previous = splitBlocks("");
    const secondToLastChangedAt: number[] = [];
    for (let n = 1; n <= REPLY.length; n++) {
      const blocks = splitBlocks(REPLY.slice(0, n));
      const settled = previous.slice(0, -2);
      expect(blocks.slice(0, settled.length), `the blocks that had ended before character ${n}`).toEqual(settled);
      const ended = previous.slice(0, -1);
      if (JSON.stringify(blocks.slice(0, ended.length)) !== JSON.stringify(ended)) secondToLastChangedAt.push(n);
      previous = blocks;
    }
    // In this reply: the "2." and the "3." of the numbered list.
    expect(secondToLastChangedAt.length, `the second to last block changed at characters ${secondToLastChangedAt}`).toBeLessThanOrEqual(4);
  });

  it("cuts at the top-level blocks, each keeping the blank lines after it", () => {
    expect(splitBlocks(SAMPLES.paragraphs)).toEqual(["First paragraph.\n\n", "Second paragraph.\n\n", "Third."]);
    expect(splitBlocks(SAMPLES.table)).toEqual(["| a | b |\n|---|---|\n| 1 | 2 |\n\n", "Text"]);
    expect(splitBlocks(SAMPLES.headings)).toEqual(["# Title\n\n", "body\n\n", "---\n\n", "> quote\n> more\n\n", "tail"]);
  });

  it("gives no block for no text and one for a text with no blank line", () => {
    expect(splitBlocks("")).toEqual([]);
    expect(splitBlocks("no newline at all")).toEqual(["no newline at all"]);
    expect(splitBlocks("a\nb\nc")).toEqual(["a\nb\nc"]);
  });

  it("keeps a fenced code block with blank lines in it as one block, finished or still open", () => {
    expect(splitBlocks(SAMPLES.fence)).toEqual(["Intro\n\n", "```js\nconst a = 1;\n\nconst b = 2;\n```\n\n", "Outro"]);
    expect(splitBlocks(SAMPLES.openFence)).toEqual(["Intro\n\n", "```js\nconst a = 1;\n\nconst b = 2;"]);
  });

  it("keeps a loose list as one block", () => {
    expect(splitBlocks(SAMPLES.list)).toEqual(["- one\n- two\n\n- three (loose)\n\n", "After the list."]);
    expect(splitBlocks("1. a\n\n2. b\n\n3. c\n\nAfter.")).toEqual(["1. a\n\n2. b\n\n3. c\n\n", "After."]);
  });

  describe("a $$ display-maths block", () => {
    it("with blank lines in it is one block (the lexer alone would cut it into a paragraph and a list)", () => {
      expect(splitBlocks(SAMPLES.math)).toEqual(["Before\n\n", "$$\nx^2\n\n+ y^2\n$$\n\n", "After"]);
    });

    it("is one block while it is still open, and the blocks after it are separate again once it closes", () => {
      expect(splitBlocks("Before\n\n$$\nx^2\n\n+ y")).toEqual(["Before\n\n", "$$\nx^2\n\n+ y"]);
      expect(splitBlocks("$$\na\n\nb\n\n```\nc\n\nd\n```\n\n$$\n\nAfter\n\nMore")).toEqual(["$$\na\n\nb\n\n```\nc\n\nd\n```\n\n$$\n\n", "After\n\n", "More"]);
    });

    it("closes only with a fence at least as long as the one that opened it", () => {
      expect(splitBlocks("$$$\nx\n\n$$\ny\n\n$$$\n\nAfter")).toEqual(["$$$\nx\n\n$$\ny\n\n$$$\n\n", "After"]);
    });

    it("is not opened by a line of maths with its own dollar signs, by prices, or by a $$ inside a code block", () => {
      expect(splitBlocks("$$x^2$$\n\nNext")).toEqual(["$$x^2$$\n\n", "Next"]);
      expect(splitBlocks(SAMPLES.prices)).toEqual(["Entry $68,400, target $72,000, stop $67,850.\n\n", "Next."]);
      expect(splitBlocks("```\n$$\n```\n\nAfter\n\nMore")).toEqual(["```\n$$\n```\n\n", "After\n\n", "More"]);
    });
  });

  it("cuts text with Windows line ends at the same blocks, and keeps its line ends", () => {
    expect(splitBlocks(SAMPLES.crlf)).toEqual(["one\r\n\r\n", "two\r\n\r\n", "- a\r\n- b\r\n"]);
  });

  it("reads a lone carriage return as a line end, as the lexer does, and keeps it", () => {
    expect(splitBlocks(SAMPLES.loneCarriageReturns)).toEqual(["one\r\r", "two"]);
  });

  // The lexer's tokens are not always the text's characters (an empty list item's trailing space is a "\n" in its token), and
  // measuring them by lines or by characters would cut the blocks before it differently from one flush to the next: a block
  // that had ended would change, and be parsed again. They are measured by length.
  it("cuts a list marker with a trailing space where it cuts it without one", () => {
    expect(splitBlocks("Para.\n\n-")).toEqual(["Para.\n\n", "-"]);
    expect(splitBlocks("Para.\n\n- ")).toEqual(["Para.\n\n", "- "]);
    expect(splitBlocks("Para.\n\n- first ")).toEqual(["Para.\n\n", "- first "]);
  });

  it("keeps a block of white space alone as a block, not nothing", () => {
    expect(splitBlocks(SAMPLES.whitespaceOnly).join("")).toBe(SAMPLES.whitespaceOnly);
  });
});
