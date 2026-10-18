import { gb, gbn, mb } from "../format";
import type { Overview } from "../ipc/bindings/Overview";

export const TABS = ["Chat", "Swarm", "Models", "Requests", "Logs", "Settings"] as const;
export type Tab = (typeof TABS)[number];

export function modelName(path: string | null | undefined): string {
  if (!path) return "no model";
  const file = path.split(/[\\/]/).pop() ?? path;
  return file.replace(/\.gguf$/i, "");
}

export function TopBar({
  tab,
  onTab,
  overview,
  error,
}: {
  tab: Tab;
  onTab: (t: Tab) => void;
  overview: Overview | null;
  /** Why the host did not answer. Until one overview has arrived it replaces "starting" with "host unreachable". */
  error?: string;
}) {
  const e = overview?.engine;
  const unreachable = !overview && error !== undefined;
  return (
    <header className="topbar">
      <div className="brand">
        <span className="wordmark">lorren</span>
        <span className="version">{overview?.app_version ?? "…"}</span>
      </div>
      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t} role="tab" className="tab" aria-selected={t === tab} onClick={() => onTab(t)}>
            {t}
          </button>
        ))}
      </nav>
      <div className="engine" aria-label="engine">
        {unreachable ? (
          <span className="flame-text" title={error}>
            <i className="dot failed" />
            host unreachable
          </span>
        ) : (
          <span>
            <i className={`dot ${e?.state ?? "stopped"}`} />
            {e?.state ?? "starting"}
          </span>
        )}
        <span className="ell" style={{ maxWidth: 260 }} title={modelName(e?.model_path)}>{modelName(e?.model_path)}</span>
        {/* `static`: settings that do not change while the app runs; below 1280 px the bar leaves them out to make room (theme.css) */}
        <span className="dim static">t={e?.threads ?? "auto"}</span>
        <span className="dim static">kv {e?.kv ?? "q8"}</span>
        {overview && (
          <span className="dim">
            app {mb(overview.memory.app_bytes)} · engine {gb(overview.memory.engine_bytes)} ·{" "}
            {gbn(overview.memory.used_bytes)}/{gb(overview.memory.total_bytes)}
          </span>
        )}
      </div>
    </header>
  );
}
