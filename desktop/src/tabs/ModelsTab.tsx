import { useEffect, useState } from "react";
import { commas, gb } from "../format";
import { api } from "../ipc/api";
import type { ModelInfo } from "../ipc/bindings/ModelInfo";
import type { Overview } from "../ipc/bindings/Overview";
import { HostFailure, useHostFailure } from "./HostFailure";
import "./tabs.css";

// The fixed columns are each as wide as their longest real value: SOURCE folder/ollama, ARCH qwen2vl, QUANT Q4_K_M, SIZE
// "12.5 GB", TEMPLATE chatml/llama3 (and the header's own word), STATUS the CAN'T RUN pill. The name takes what is left and
// shortens first; with a 320 px right panel that is ~265 px at the window's minimum width, and all seven columns show.
const COLS = "minmax(140px, 1fr) 56px 64px 56px 64px 64px 80px";

export function ModelsTab({ overview }: { overview: Overview | null }) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const { failure, report } = useHostFailure("models");
  // One call each time the tab is shown (it mounts when shown), so a failure lasts until the tab is shown again.
  useEffect(() => {
    api.models().then(
      (ms) => {
        setModels(ms);
        setSelected((s) => s ?? ms[0]?.id ?? null);
      },
      report,
    );
  }, [report]);
  const m = models.find((x) => x.id === selected) ?? null;
  const loaded = overview?.engine.model_id;
  return (
    <>
      <section className="col center">
        <div className="head">
          <span className="title">Models</span>
          {failure === null && <span className="cap">{models.length} IN LIBRARY · {models.filter((x) => x.runnable).length} RUNNABLE</span>}
        </div>
        {failure !== null && <HostFailure what="models" reason={failure} />}
        <div className="table-head" style={{ gridTemplateColumns: COLS }}>
          <span>NAME</span><span>SOURCE</span><span>ARCH</span><span>QUANT</span><span>SIZE</span><span>TEMPLATE</span><span>STATUS</span>
        </div>
        <div className="scroll">
          {models.map((x) => (
            <div key={x.id} role="row" className={`table-row clickable ${x.id === selected ? "selected" : ""}`}
                 style={{ gridTemplateColumns: COLS, height: 44 }} onClick={() => setSelected(x.id)}>
              <span className="name-cell">
                <span className="ell">{x.name}</span>
                {x.is_default && <span className="pill tag accent-text">DEFAULT</span>}
                {x.id === loaded && <span className="pill tag shared-text">LOADED</span>}
              </span>
              <span className="mono dim">{x.source}</span>
              <span className="mono">{x.arch}</span>
              <span className="mono">{x.quant}</span>
              <span className="mono num">{gb(x.size_bytes)}</span>
              <span className="mono">{x.template}</span>
              <span className={`pill ${x.runnable ? "" : "flame-text"}`}>{x.runnable ? "RUNNABLE" : "CAN'T RUN"}</span>
            </div>
          ))}
        </div>
      </section>
      <aside className="col right">{m && <ModelDetails m={m} />}</aside>
    </>
  );
}

function ModelDetails({ m }: { m: ModelInfo }) {
  return (
    <>
      <div className="section">
        <span className="cap">MODEL</span>
        <span className="title">{m.name}</span>
        <span className="mono dim ell" title={m.path} style={{ fontSize: 10.5 }}>{m.path}</span>
      </div>
      {!m.runnable && (
        <div className="section">
          <span className="cap flame-text">CAN'T RUN</span>
          <span>{m.reason}</span>
        </div>
      )}
      <div className="section">
        <div className="fact"><span className="k">size</span><span className="v">{m.size_label || "—"} · {m.quant}</span></div>
        <div className="fact"><span className="k">layers · heads · KV heads</span><span className="v num">{m.n_layer} · {m.n_head} · {m.n_head_kv}</span></div>
        <div className="fact"><span className="k">context (file / engine)</span><span className="v num">{commas(m.ctx_file)} / {commas(m.ctx_engine)}</span></div>
        <div className="fact"><span className="k">KV per token (q8)</span><span className="v num">{(m.kv_bytes_per_token / 1024).toFixed(1)} KiB</span></div>
        <div className="fact"><span className="k">memory to run (model + KV cache + runtime)</span><span className="v num">{m.run_bytes > 0 ? `~${gb(m.run_bytes)}` : "—"}</span></div>
      </div>
      <div className="section">
        <span className="cap">SYSTEM PROMPT{m.system_source ? ` · FROM ${m.system_source.toUpperCase()}` : ""}</span>
        <span className="mono" style={{ fontSize: 11.5, whiteSpace: "pre-wrap" }}>{m.system_prompt ?? "none"}</span>
      </div>
    </>
  );
}
