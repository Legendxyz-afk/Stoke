import { marked } from "marked";

// A reply that is still streaming is cut into its top-level Markdown blocks (demo D5), so that a block that has ended is
// parsed once and only the block that is still growing is parsed again on a flush.

/** Opens a display-maths block (remark-math's `$$` fence): up to three spaces, two or more dollar signs, and anything after
 *  them that has no dollar sign of its own. `$$x^2$$` on one line is not a fence; it is maths inside a paragraph. */
const FENCE_OPEN = /^ {0,3}(\${2,})[^$\n]*$/;
/** Closes it: a line of dollar signs, at least as many as opened it. */
const FENCE_CLOSE = /^ {0,3}(\$+)[ \t\r]*$/;

/** The fence still open after `block`, given the one open before it (null for none; otherwise the length of its opening). */
function scanFences(block: string, open: number | null): number | null {
  for (const line of block.split("\n")) {
    if (open === null) {
      const m = FENCE_OPEN.exec(line);
      if (m) open = m[1].length;
    } else {
      const m = FENCE_CLOSE.exec(line);
      if (m && m[1].length >= open) open = null;
    }
  }
  return open;
}

/** The text as the lexer reads it, and where each place in that copy is in the text. `marked` turns "\r\n" and "\r" into
 *  "\n" before it reads (so does this, to know the copy it measures), which makes the copy shorter than the text for every
 *  "\r\n". `origin(k)` is where the k-th character of the copy is in the text; `origin(copy.length)` is the text's end. */
function lexerCopy(text: string): { copy: string; origin: (k: number) => number } {
  if (!text.includes("\r")) return { copy: text, origin: (k) => k };
  let copy = "";
  const at: number[] = [];
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "\r" && text[i + 1] === "\n") continue; // the "\n" of the pair stands for both
    copy += text[i] === "\r" ? "\n" : text[i];
    at.push(i);
  }
  at.push(text.length);
  return { copy, origin: (k) => at[k] };
}

/** The text of a reply cut into its top-level Markdown blocks, in order, each with the blank lines that follow it. The blocks
 *  joined give back `text` byte for byte. The cut is `marked.lexer`'s, so a fenced code block or a list is one block however
 *  many blank lines it holds, with one thing added: `marked` does not know remark-math's `$$` fence, so a block that leaves one
 *  open takes the blocks after it with it until it is closed (as remark-math reads it, an unclosed fence runs to the end).
 *
 *  The cut is made by LENGTH: the lexer advances through its copy of the text by the length of each token's `raw`, so the
 *  lengths are where the blocks end. (The characters of a `raw` are not always the text's: an empty list item's trailing space
 *  is a "\n" in its `raw`.) Anything that cannot be cut with certainty is one block, which is always right and merely slower. */
export function splitBlocks(text: string): string[] {
  if (text === "") return [];
  const { copy, origin } = lexerCopy(text);
  // The lexer's tokens, with the blank lines (`space` tokens) after a block joined to it; one that starts the text stands alone.
  const groups: { length: number; code: boolean }[] = [];
  try {
    for (const token of marked.lexer(copy)) {
      if (token.type === "space" && groups.length > 0) groups[groups.length - 1].length += token.raw.length;
      else groups.push({ length: token.raw.length, code: token.type === "code" });
    }
  } catch {
    return [text];
  }
  if (groups.reduce((sum, g) => sum + g.length, 0) !== copy.length) return [text];
  const blocks: string[] = [];
  let open: number | null = null;
  let [from, consumed] = [0, 0];
  for (const group of groups) {
    consumed += group.length;
    const to = origin(consumed);
    const block = text.slice(from, to);
    from = to;
    if (open !== null && blocks.length > 0) blocks[blocks.length - 1] += block;
    else blocks.push(block);
    // Code is not maths: a `$$` line in a code block neither opens nor closes a fence.
    if (!group.code) open = scanFences(block, open);
  }
  return blocks;
}
