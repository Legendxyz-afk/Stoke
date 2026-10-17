import { useEffect, useState } from "react";
import { clock, secs } from "../format";
import { api } from "../ipc/api";
import type { RequestRecord } from "../ipc/bindings/RequestRecord";
import { HostFailure, useHostFailure } from "./HostFailure";
import "./tabs.css";

const COLS = "86px 64px minmax(200px, 2fr) 64px 64px 96px 86px 70px 150px";

export function RequestsTab() {
  const [rows, setRows] = useState<RequestRecord[]>([]);
  const { failure, report, clear } = useHostFailure("requests");
  useEffect(() => {
    let alive = true;
    const tick = () =>
      api.requests().then(
        (r) => {
          if (!alive) return;
          clear();
          setRows(r);
        },
        (e) => {
          if (alive) report(e);
        },
      );
    tick();
    const t = setInterval(tick, 1000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [report, clear]);
  // After a failure the rows already loaded stay (with the reason above them). With none loaded, a count of 0 or
  // "No requests yet." would be a statement about the host's answer, and there is none.
  const unknown = failure !== null && rows.length === 0;
  return (
    <section className="col center wide">
      <div className="head">
        <span className="title">Requests</span>
        {!unknown && <span className="cap">{rows.length} THIS SESSION · NEWEST FIRST</span>}
      </div>
      {failure !== null && <HostFailure what="requests" reason={failure} />}
      <div className="table-head" style={{ gridTemplateColumns: COLS }}>
        <span>TIME</span><span>FROM</span><span>MODEL</span><span>IN</span><span>OUT</span><span>FIRST TOKEN</span><span>TOTAL</span><span>TOK/S</span><span>STATUS</span>
      </div>
      <div className="scroll">
        {rows.length === 0 && failure === null && <div className="dim mono" style={{ padding: 22, fontSize: 11 }}>No requests yet.</div>}
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
