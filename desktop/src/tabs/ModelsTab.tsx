import { useCallback, useEffect, useRef, useState } from "react";
import { commas, gb } from "../format";
import { api } from "../ipc/api";
import type { CatalogEntry } from "../ipc/bindings/CatalogEntry";
import type { CatalogView } from "../ipc/bindings/CatalogView";
import type { DiskRoom } from "../ipc/bindings/DiskRoom";
import type { DownloadView } from "../ipc/bindings/DownloadView";
import type { ModelInfo } from "../ipc/bindings/ModelInfo";
import type { Overview } from "../ipc/bindings/Overview";
import { DiscoverClosed, DiscoverSection, EntryPanel, entryState } from "./Discover";
import { HostFailure, useHostFailure } from "./HostFailure";
import "./tabs.css";

// The fixed columns are each as wide as their longest real value: SOURCE folder/ollama, ARCH qwen2vl, QUANT Q4_K_M, SIZE
// "12.5 GB", TEMPLATE chatml/llama3 (and the header's own word), STATUS the CAN'T RUN pill. The name takes what is left and
// shortens first; with a 320 px right panel that is ~265 px at the window's minimum width, and all seven columns show.
const COLS = "minmax(140px, 1fr) 56px 64px 56px 64px 64px 80px";

export type ModelsView = "library" | "discover";

/** The catalog is read from the network the first time Discover is opened in a session, and only then. React's double effect
 *  and a tab that is left and shown again must not make a second read, so the read is kept here, outside any component. */
let firstRead: Promise<CatalogView> | null = null;
let firstReadSettled = false;
/** For the tests: forget that this session has read the catalog. */
export function resetDiscoverSession() {
  firstRead = null;
  firstReadSettled = false;
}

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));
/** A folder path compared case-insensitively, with `/` and `\` alike and no trailing separator. */
const normDir = (p: string) => p.replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
const folderOf = (path: string) => normDir(path.replace(/[\\/][^\\/]*$/, ""));
const RUNNING = ["downloading", "paused", "verifying"];

export function ModelsTab({ overview, view = "library", onView = () => {} }: {
  overview: Overview | null;
  /** App holds the view, so the chat's OPEN DISCOVER can set it. */
  view?: ModelsView;
  onView?: (v: ModelsView) => void;
}) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<CatalogView | null>(null);
  const [dls, setDls] = useState<Record<string, DownloadView | undefined>>({});
  const [room, setRoom] = useState<DiskRoom | null>(null);
  const [entryId, setEntryId] = useState<string | null>(null);
  const [refusal, setRefusal] = useState<{ id: string; text: string } | null>(null);
  const [q, setQ] = useState("");
  const [applied, setApplied] = useState("");
  const { failure, report } = useHostFailure("models");
  const openedOnce = useRef(false);
  const onViewRef = useRef(onView);
  onViewRef.current = onView;

  const loadModels = useCallback(
    (first: boolean) =>
      api.models().then(
        (ms) => {
          setModels(ms);
          setModelsLoaded(true);
          setSelected((s) => s ?? ms[0]?.id ?? null);
          // The tab opens on Discover when the library has no model; once, so that going back to the library stays possible.
          if (first && ms.length === 0 && !openedOnce.current) {
            openedOnce.current = true;
            onViewRef.current("discover");
          }
        },
        report,
      ),
    [report],
  );
  // One call each time the tab is shown (it mounts when shown), so a failure lasts until the tab is shown again.
  useEffect(() => {
    void loadModels(true);
  }, [loadModels]);

  // The local calls: what is downloading, and the downloads folder with the room in it; and the two events.
  useEffect(() => {
    let alive = true;
    api.downloads().then(
      (ds) => { if (alive) setDls((cur) => ({ ...Object.fromEntries(ds.map((d) => [d.entry_id, d])), ...cur })); },
      (e) => console.error("downloads failed:", message(e)),
    );
    api.diskRoom().then((r) => { if (alive) setRoom(r); }, (e) => console.error("diskRoom failed:", message(e)));
    const guard = (what: string) => (e: unknown) => {
      console.error(`${what} listener failed:`, message(e));
      return () => {};
    };
    const offState = api.onDownloadState((d) => setDls((cur) => ({ ...cur, [d.entry_id]: d }))).catch(guard("download-state"));
    const offLib = api
      .onLibraryChanged(() => {
        void loadModels(false);
        api.diskRoom().then(setRoom, (e) => console.error("diskRoom failed:", message(e)));
      })
      .catch(guard("library-changed"));
    return () => {
      alive = false;
      void offState.then((f) => f());
      void offLib.then((f) => f());
    };
  }, [loadModels]);

  const failCatalog = useCallback((e: unknown) => {
    const text = message(e);
    console.error("refreshCatalog failed:", text);
    setCatalog((c) => ({ ...(c ?? { configured: true, entries: [], needs_app: [], fetched_ms: null }), stale: true, error: text }));
  }, []);

  // The catalog: the first opening of Discover in a session reads it over the network; every other time the kept view is shown.
  useEffect(() => {
    let alive = true;
    let p: Promise<CatalogView>;
    if (view === "discover" && (firstRead === null || !firstReadSettled)) {
      firstRead ??= api.refreshCatalog().then(
        (c) => { firstReadSettled = true; return c; },
        (e) => { firstReadSettled = true; throw e; },
      );
      p = firstRead;
    } else {
      p = api.catalog();
    }
    p.then((c) => { if (alive) setCatalog(c); }, (e) => { if (alive) failCatalog(e); });
    return () => { alive = false; };
  }, [view, failCatalog]);

  const readAgain = () => {
    api.refreshCatalog().then(setCatalog, failCatalog);
  };

  const entries = catalog?.entries ?? [];
  const entry = entries.find((x) => x.id === entryId) ?? entries[0] ?? null;
  const needle = applied.trim().toLowerCase();
  const shown = entries.filter((x) => x.name.toLowerCase().includes(needle));
  const stateOf = (x: CatalogEntry) => entryState(x, catalog!, dls, models);

  const attempt = (id: string, call: () => Promise<unknown>) => {
    setRefusal(null);
    call().then(
      (r) => { if (r && typeof r === "object") setDls((cur) => ({ ...cur, [id]: r as DownloadView })); },
      (e) => setRefusal({ id, text: message(e) }),
    );
  };
  const cancel = (id: string) => {
    setRefusal(null);
    // The host emits no event for a cancel (the job and its row are gone), so the entry's view is dropped here.
    api.cancelDownload(id).then(
      () => setDls((cur) => ({ ...cur, [id]: undefined })),
      (e) => setRefusal({ id, text: message(e) }),
    );
  };

  const m = models.find((x) => x.id === selected) ?? null;
  const loaded = overview?.engine.model_id;
  const running = Object.values(dls).filter((d): d is DownloadView => !!d && RUNNING.includes(d.state));
  const dir = room ? normDir(room.dir) : null;
  const sourceOf = (x: ModelInfo) => (dir !== null && folderOf(x.path) === dir ? "discover" : x.source);
  return (
    <>
      <section className="col center">
        <div className="head">
          <span className="title">Models</span>
          {failure === null && <span className="cap">{models.length} IN LIBRARY · {models.filter((x) => x.runnable).length} RUNNABLE</span>}
        </div>
        {failure !== null && <HostFailure what="models" reason={failure} />}
        <div className="scroll" style={{ flex: 1 }}>
          <div className="table-head" style={{ gridTemplateColumns: COLS }}>
            <span>NAME</span><span>SOURCE</span><span>ARCH</span><span>QUANT</span><span>SIZE</span><span>TEMPLATE</span><span>STATUS</span>
          </div>
          {models.map((x) => (
            <div key={x.id} role="row" className={`table-row clickable ${x.id === selected && view === "library" ? "selected" : ""}`}
                 style={{ gridTemplateColumns: COLS, height: 44 }} onClick={() => { setSelected(x.id); onView("library"); }}>
              <span className="name-cell">
                <span className="ell">{x.name}</span>
                {x.is_default && <span className="pill tag accent-text">DEFAULT</span>}
                {x.id === loaded && <span className="pill tag shared-text">LOADED</span>}
              </span>
              <span className="mono dim">{sourceOf(x)}</span>
              <span className="mono">{x.arch}</span>
              <span className="mono">{x.quant}</span>
              <span className="mono num">{gb(x.size_bytes)}</span>
              <span className="mono">{x.template}</span>
              <span className={`pill ${x.runnable ? "" : "flame-text"}`}>{x.runnable ? "RUNNABLE" : "CAN'T RUN"}</span>
            </div>
          ))}
          {running.map((d) => {
            const e = entries.find((x) => x.id === d.entry_id);
            return (
              <div key={`dl:${d.entry_id}`} role="row" className="table-row" style={{ gridTemplateColumns: COLS, height: 44 }}>
                <span className="name-cell"><span className="ell">{e?.name ?? d.name}</span></span>
                <span className="mono dim">discover</span>
                <span className="mono">{e?.architecture ?? "—"}</span>
                <span className="mono">{e?.quantization ?? "—"}</span>
                <span className="mono num">{gb(e?.size ?? d.total)}</span>
                <span className="mono">—</span>
                <span className="pill accent-text">{d.state.toUpperCase()}</span>
              </div>
            );
          })}
          {modelsLoaded && failure === null && models.length === 0 && running.length === 0 && (
            <div className="dim mono" style={{ padding: "20px 22px", fontSize: 11 }}>No models yet. Discover has some to start with.</div>
          )}
          {view === "library" ? (
            <DiscoverClosed onOpen={() => onView("discover")} />
          ) : (
            <DiscoverSection catalog={catalog} shown={shown} state={stateOf} selected={entry?.id ?? null}
                             onSelect={setEntryId} onReadAgain={readAgain} />
          )}
        </div>
        {view === "discover" && (
          <div className="search">
            <span className="lab">discover</span>
            <input aria-label="search the catalog" placeholder="qwen" value={q} autoComplete="off"
                   onChange={(ev) => setQ(ev.target.value)} onKeyDown={(ev) => { if (ev.key === "Enter") setApplied(q); }} />
            <button className="btn accent" onClick={() => setApplied(q)}>SEARCH</button>
          </div>
        )}
      </section>
      <aside className="col right">
        {view === "library" ? (
          m && <ModelDetails m={m} />
        ) : (
          entry && catalog && (
            <EntryPanel entry={entry} state={stateOf(entry)} room={room} refusal={refusal?.id === entry.id ? refusal.text : null}
                        onGet={() => attempt(entry.id, () => api.startDownload(entry.id))}
                        onPause={() => attempt(entry.id, () => api.pauseDownload(entry.id))}
                        onResume={() => attempt(entry.id, () => api.resumeDownload(entry.id))}
                        onCancel={() => cancel(entry.id)}
                        onLicence={() => { api.openLicence(entry.id).catch((e) => setRefusal({ id: entry.id, text: message(e) })); }} />
          )
        )}
      </aside>
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
        <div className="fact"><span className="k">memory to run, 2 chats at once (model + KV cache + runtime)</span><span className="v num">{m.run_bytes > 0 ? `~${gb(m.run_bytes)}` : "—"}</span></div>
        <div className="fact"><span className="k">memory to run, 8 chats at once</span><span className="v num">{m.run_bytes > 0 ? `~${gb(m.run_bytes_max)}` : "—"}</span></div>
      </div>
      <div className="section">
        <span className="cap">SYSTEM PROMPT{m.system_source ? ` · FROM ${m.system_source.toUpperCase()}` : ""}</span>
        <span className="mono" style={{ fontSize: 11.5, whiteSpace: "pre-wrap" }}>{m.system_prompt ?? "none"}</span>
      </div>
    </>
  );
}
