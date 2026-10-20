import "highlight.js/styles/github-dark.css";
import "katex/dist/katex.min.css";
import { memo, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { splitBlocks } from "./blocks";

/** rehype-highlight makes a lowlight with all of its ~37 languages every time it is attached, and a language's grammar is
 *  compiled the first time it is used, in THAT lowlight. react-markdown attaches its plugins on every render, so every render
 *  paid for the registration and every code block for its grammar, again: 40% of what it cost to parse a block that is still
 *  growing (demo D5). The transformer is made once, here, and attaching it is free. It keeps no state between parses. */
const highlighter = rehypeHighlight();
const rehypeHighlightOnce = () => highlighter;

/** The parse of one piece of Markdown, without the `.md` wrapper: `Markdown` is one of these, and a reply that is still
 *  streaming is several (`StreamingMarkdown`).
 *  Model output. Raw HTML is dropped (`skipHtml`), a link never navigates the app window, and an image is never fetched: the
 *  app shows no remote content (spec §3.8).
 *  Maths is `$$...$$` only (R58): Lorren's chats are about prices, and "entry $68,400, target $72,000" must not be read as a
 *  formula between two single dollar signs.
 *  Memoized: react-markdown parses its whole text on every render (~6 ms for a 900-character reply), and ChatTab re-renders
 *  on every keystroke and every streamed flush, so a piece whose text has not changed must not be parsed again. */
export const MarkdownBody = memo(function MarkdownBody({ text }: { text: string }) {
  return (
    <ReactMarkdown
      skipHtml
      remarkPlugins={[remarkGfm, [remarkMath, { singleDollarTextMath: false }]]}
      rehypePlugins={[rehypeKatex, rehypeHighlightOnce]}
      components={{
        a: ({ children, href }) => (
          <span className="md-link" title={href ?? ""}>
            {children}
          </span>
        ),
        img: ({ alt }) => <span className="md-img">[image: {alt ?? ""}]</span>,
      }}
    >
      {text}
    </ReactMarkdown>
  );
});

/** A settled message: its whole text is ONE parse, with the exact semantics of Markdown (lists, footnotes, references). */
export const Markdown = memo(function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <MarkdownBody text={text} />
    </div>
  );
});

/** The reply that is being written (demo D5). Parsing all of it on every flush made the UI cost grow with the reply: the
 *  whole text was parsed ~15-30 times a second, 19% of one core at 15 tok/s. The text is cut into its top-level blocks
 *  (`splitBlocks`) and each block is parsed on its own, keyed by its position and memoized by its text: a block that has
 *  ended keeps its text, so it is parsed once, and only the block still growing is parsed again on a flush.
 *  What a block cannot see of the others (a reference-style link whose definition comes later, a footnote) is resolved when
 *  the reply ends and ChatTab hands the whole text to `Markdown`. */
export function StreamingMarkdown({ text }: { text: string }) {
  const blocks = useMemo(() => splitBlocks(text), [text]);
  return (
    <div className="md">
      {blocks.map((block, i) => (
        <MarkdownBody key={i} text={block} />
      ))}
    </div>
  );
}
