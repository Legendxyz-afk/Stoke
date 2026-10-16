import type { Chat } from "../ipc/bindings/Chat";
import type { ModelInfo } from "../ipc/bindings/ModelInfo";

export function ChatList({ chats, models, activeId, onSelect, onNew }: {
  chats: Chat[];
  models: ModelInfo[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
}) {
  return (
    <nav className="col left">
      <div className="head">
        <span className="title">Chats</span>
        <button className="link" style={{ marginLeft: "auto" }} onClick={onNew}>
          + NEW
        </button>
      </div>
      <div className="scroll">
        {chats.length === 0 && <div className="dim mono" style={{ padding: 20, fontSize: 11 }}>No chats yet.</div>}
        {chats.map((c) => (
          <button
            key={c.id}
            className={`chat-item ${c.id === activeId ? "selected" : ""}`}
            aria-current={c.id === activeId}
            onClick={() => onSelect(c.id)}
          >
            <span className="ell">{c.title || "New chat"}</span>
            <span className="cap ell">{models.find((m) => m.id === c.model_id)?.name ?? "missing model"}</span>
          </button>
        ))}
      </div>
    </nav>
  );
}
