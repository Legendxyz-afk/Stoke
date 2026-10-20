import type { MessageView } from "../ipc/bindings/MessageView";
import { Markdown, StreamingMarkdown } from "./Markdown";
import { PendingStatus, type Pending } from "./Pending";

export function MessageRow({ mv, liveText, pending, onVersion }: {
  mv: MessageView;
  /** The streaming text, while this is the reply being written. */
  liveText: string | null;
  /** This is the reply being written and it has no token yet: its body is the status line (demo D2) instead of an empty text. */
  pending: Pending | null;
  onVersion: (messageId: string) => void;
}) {
  const m = mv.message;
  const text = liveText ?? m.content;
  return (
    <div className={`msg ${m.role}`} data-status={m.status}>
      <div className="msg-head cap">
        <span>{m.role === "user" ? "YOU" : "MODEL"}</span>
        {mv.versions > 1 && (
          <span className="versions">
            <button className="link" aria-label="previous version" disabled={mv.version <= 1}
                    onClick={() => onVersion(mv.sibling_ids[mv.version - 2])}>‹</button>
            <span>{mv.version}/{mv.versions}</span>
            <button className="link" aria-label="next version" disabled={mv.version >= mv.versions}
                    onClick={() => onVersion(mv.sibling_ids[mv.version])}>›</button>
          </span>
        )}
        {(m.status === "stopped" || m.status === "interrupted" || m.status === "error") && (
          <span className={`pill status-${m.status}`}>{m.status.toUpperCase()}</span>
        )}
      </div>
      {m.role === "user" ? (
        <div className="msg-user">{text}</div>
      ) : pending ? (
        <PendingStatus since={pending.since} loading={pending.loading} />
      ) : liveText !== null ? (
        // The reply being written is parsed block by block, so that only the block still growing is parsed on a flush (demo D5).
        <StreamingMarkdown text={text} />
      ) : (
        // A settled message is ONE parse of the whole text, with the exact semantics of the whole (lists, footnotes, references).
        <Markdown text={text} />
      )}
      {m.status === "error" && m.stats.error && <div className="msg-error">{m.stats.error}</div>}
    </div>
  );
}
