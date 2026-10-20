import type { ReactNode } from "react";
import { count, secs, since } from "../format";
import type { Overview } from "../ipc/bindings/Overview";

export function StatusBar({ overview, children }: { overview: Overview | null; children?: ReactNode }) {
  const e = overview?.engine;
  return (
    <footer className="statusbar">
      <span>session {overview ? since(overview.started_ms) : "—"}</span>
      <span>{count(overview?.request_count ?? 0, "request")}</span>
      <span>{count(e?.restarts ?? 0, "engine restart")}</span>
      {e?.load_ms != null && <span>last load {secs(e.load_ms)}</span>}
      {e?.last_error && (
        <span className="ell" style={{ color: "var(--flame)", maxWidth: 520 }} title={e.last_error}>{e.last_error}</span>
      )}
      <span className="spacer" />
      {children}
      <span>engine frozen b1ee6b3 · ctx {e?.ctx ?? 4096} · {e?.seqs ?? 2} parallel</span>
    </footer>
  );
}
