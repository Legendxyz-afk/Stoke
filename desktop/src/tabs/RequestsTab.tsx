import { useEffect, useState } from "react";
import { clock, secs } from "../format";
import { api } from "../ipc/api";
import type { RequestRecord } from "../ipc/bindings/RequestRecord";
import "./tabs.css";

const COLS = "86px 64px minmax(200px, 2fr) 64px 64px 96px 86px 70px 150px";

export function RequestsTab() {
  const [rows, setRows] = useState<RequestRecord[]>([]);
  useEffect(() => {
    let alive = true;
    const tick = () => api.requests().then((r) => alive && setRows(r)).catch(() => {});
    tick();
    const t = setInterval(tick, 1000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);
  return (
    <section className="col center wide">
      <div className="head">
        <span className="title">Requests</span>
        <span className="cap">{rows.length} THIS SESSION · NEWEST FIRST</span>
      </div>
      <div className="table-head" style={{ gridTemplateColumns: COLS }}>
        <span>TIME</span><span>FROM</span><span>MODEL</span><span>IN</span><span>OUT</span><span>FIRST TOKEN</span><span>TOTAL</span><span>TOK/S</span><span>STATUS</span>
      </div>
      <div className="scroll">
        {rows.length === 0 && <div className="dim mono" style={{ padding: 22, fontSize: 11 }}>No requests yet.</div>}
        {rows.map((r) => (
          <div key={r.id} role="row" className="table-row" style={{ gridTemplateColumns: COLS, height: 36 }}>
            <span className="mono num">{clock(r.started_ms)}</span>
            <span className="mono dim">{r.source}</span>
            <span className="ell">{r.model}</span>
            <span className="mono num">{r.prompt_tokens ?? "—"}</span>
            <span className="mono num">{r.completion_tokens ?? "—"}</span>
            <span className="mono num">{secs(r.ttft_ms)}</span>
            <span className="mono num">{secs(r.total_ms)}</span>
            <span className="mono num">{r.tokens_per_s != null ? r.tokens_per_s.toFixed(1) : "—"}</span>
            <span className={`pill status-${r.status}`}>
              {r.status.toUpperCase()}{r.dropped_turns > 0 ? ` · ${r.dropped_turns} CUT` : ""}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
