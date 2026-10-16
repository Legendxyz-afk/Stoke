import { commas, secs } from "../format";
import type { Chat } from "../ipc/bindings/Chat";
import type { ModelInfo } from "../ipc/bindings/ModelInfo";
import type { Overview } from "../ipc/bindings/Overview";
import type { ReplyStats } from "../ipc/bindings/ReplyStats";
import type { Live } from "./stream";

export function ReplyPanel({ chat, models, overview, live, busy, last, onModel }: {
  chat: Chat | null;
  models: ModelInfo[];
  overview: Overview | null;
  live: Live;
  /** A reply is being written: this window's (`live.busy`) or one the stored chat still shows as streaming (R36d). */
  busy: boolean;
  last: ReplyStats | null;
  onModel: (modelId: string) => void;
}) {
  const current = models.find((m) => m.id === chat?.model_id);
  const e = overview?.engine;
  const engine = !chat ? "—" : e?.model_id === chat.model_id ? (e?.state ?? "—") : e?.state === "starting" ? "switching" : "loads on the next send";
  const stats = live.stats ?? last;
  const ctx = current?.ctx_engine ?? 4096;
  const used = stats?.prompt_tokens != null ? stats.prompt_tokens + (stats.completion_tokens ?? 0) : null;
  return (
    <aside className="col right">
      <div className="section">
        <span className="cap">MODEL</span>
        <select aria-label="model" value={chat?.model_id ?? ""} disabled={!chat || busy}
                onChange={(ev) => onModel(ev.target.value)}>
          {models.map((m) => (
            <option key={m.id} value={m.id} disabled={!m.runnable} title={m.reason ?? ""}>
              {m.name}{m.runnable ? "" : " (can't run)"}
            </option>
          ))}
        </select>
        <div className="fact"><span className="k">engine</span><span className="v">{engine}</span></div>
        {current?.system_source && (
          <div className="fact"><span className="k">system prompt</span><span className="v">from {current.system_source}</span></div>
        )}
      </div>
      <div className="section">
        <span className="cap">LAST REPLY</span>
        <div className="fact"><span className="k">first token</span><span className="v num">{secs(stats?.ttft_ms)}</span></div>
        <div className="fact"><span className="k">speed</span>
          <span className="v num">{stats?.tokens_per_s != null ? `${stats.tokens_per_s.toFixed(1)} tok/s` : "—"}</span></div>
        <div className="fact"><span className="k">tokens in · out</span>
          <span className="v num">{stats?.prompt_tokens ?? "—"} · {stats?.completion_tokens ?? "—"}</span></div>
        <div className="fact"><span className="k">total</span><span className="v num">{stats ? secs(stats.total_ms) : "—"}</span></div>
      </div>
      <div className="section">
        <span className="cap">CONTEXT</span>
        <div className="bar" aria-label="context used">
          <i style={{ width: `${Math.min(100, ((used ?? 0) / ctx) * 100)}%` }} />
        </div>
        <div className="fact"><span className="k">last request</span><span className="v num">{used ?? "—"} / {commas(ctx)}</span></div>
        <span className="mono dim" style={{ fontSize: 10.5, lineHeight: 1.5 }}>
          Each turn sends the last ~1,500 tokens of the chat. The engine re-reads them on every turn until it keeps
          a cache between turns, so more history means a longer wait for the first word.
        </span>
      </div>
    </aside>
  );
}
