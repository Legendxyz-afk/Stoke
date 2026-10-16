export function Composer({ draft, onDraft, busy, disabled, canRegenerate, onSend, onStop, onRegenerate }: {
  /** The text in the box. ChatTab owns it, so a message the host refuses can be put back (R36b). */
  draft: string;
  onDraft: (text: string) => void;
  busy: boolean;
  disabled: boolean;
  canRegenerate: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onRegenerate: () => void;
}) {
  const submit = () => {
    const t = draft.trim();
    if (!t || busy || disabled) return;
    onSend(t);
  };
  return (
    <div className="composer">
      <textarea
        aria-label="message"
        placeholder="Message (Enter sends, Shift+Enter is a new line)"
        value={draft}
        disabled={disabled}
        onChange={(e) => onDraft(e.target.value)}
        onKeyDown={(e) => {
          // The Enter that confirms an input-method composition is not a send (R36c).
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
      />
      <div className="composer-actions">
        {canRegenerate && !busy && (
          <button className="btn" onClick={onRegenerate} disabled={disabled}>REGENERATE</button>
        )}
        {busy ? (
          <button className="btn accent" onClick={onStop}>STOP</button>
        ) : (
          <button className="btn accent" onClick={submit} disabled={disabled || !draft.trim()}>SEND</button>
        )}
      </div>
    </div>
  );
}
