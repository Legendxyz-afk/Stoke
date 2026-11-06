/** The transcript of a chat when the library holds no model that can run (the design's demo): what is missing and the way to it. */
export function NoModel({ onDiscover }: { onDiscover: () => void }) {
  return (
    <>
      <div className="mono" style={{ fontSize: 11.5, lineHeight: 1.6, maxWidth: 520 }}>
        No model yet. Lorren needs one to chat: get one from Discover (about 2 GB, a few minutes).
      </div>
      <div><button className="btn accent" onClick={onDiscover}>OPEN DISCOVER</button></div>
    </>
  );
}

/** The composer with the reason it cannot be used: nothing can be sent until a model is in the library. */
export function NoModelComposer() {
  return (
    <div className="composer">
      <textarea aria-label="message" placeholder="Get a model from Discover to start chatting" disabled />
      <div className="composer-actions">
        <button className="btn accent" disabled>SEND</button>
      </div>
    </div>
  );
}
