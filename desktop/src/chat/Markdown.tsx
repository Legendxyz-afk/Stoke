import "highlight.js/styles/github-dark.css";
import "katex/dist/katex.min.css";
import { memo } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

/** Model output. Raw HTML is dropped (`skipHtml`), a link never navigates the app window, and an
 *  image is never fetched: the app shows no remote content (spec §3.8).
 *  Memoized: react-markdown parses its whole text on every render (~6 ms for a 900-character reply), and ChatTab
 *  re-renders on every keystroke and every streamed flush, so a settled message must not be parsed again. */
export const Markdown = memo(function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        skipHtml
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
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
    </div>
  );
});
