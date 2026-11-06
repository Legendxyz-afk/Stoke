import type { ReactNode } from "react";
import { clock, gb, gbn } from "../format";
import type { CatalogEntry } from "../ipc/bindings/CatalogEntry";
import type { CatalogView } from "../ipc/bindings/CatalogView";
import type { DiskRoom } from "../ipc/bindings/DiskRoom";
import type { DownloadView } from "../ipc/bindings/DownloadView";
import type { ModelInfo } from "../ipc/bindings/ModelInfo";

const GIB = 1073741824;
const MIB = 1048576;

// NAME, ARCH, QUANT, SIZE, LICENCE, STATUS (the design's demo).
export const DCOLS = "minmax(140px, 1fr) 64px 64px 64px 150px 96px";

/** What a catalog entry is to the user, first match wins (task 7, resolution 3). */
export type EntryState =
  | { kind: "in_library"; dl: DownloadView }
  | { kind: "update"; dl: DownloadView }
  | { kind: "on_disk" }
  | { kind: "needs_app" }
  | { kind: "running"; dl: DownloadView }
  | { kind: "failed"; dl: DownloadView }
  | { kind: "get" };

const fileName = (path: string) => path.split(/[\\/]/).pop() ?? path;
const sameName = (a: string, b: string) => a.toLowerCase() === b.toLowerCase();

export function entryState(e: CatalogEntry, view: CatalogView, dls: Record<string, DownloadView | undefined>, models: ModelInfo[]): EntryState {
  const d = dls[e.id];
  const listed = (file: string) => models.some((m) => sameName(fileName(m.path), file));
  if (d && d.state === "done" && d.version === e.version && listed(d.file)) return { kind: "in_library", dl: d };
  if (d && d.state === "done" && d.version < e.version) return { kind: "update", dl: d };
  if (!d && listed(e.file)) return { kind: "on_disk" };
  if (view.needs_app.includes(e.id)) return { kind: "needs_app" };
  if (d && (d.state === "downloading" || d.state === "paused" || d.state === "verifying")) return { kind: "running", dl: d };
  if (d && d.state === "failed") return { kind: "failed", dl: d };
  return { kind: "get" };
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** When the catalog was read: the time of day if that was today, else the day too, as `6 Nov 09:41`. */
export function readAt(ms: number, now = Date.now()): string {
  const d = new Date(ms);
  if (d.toDateString() === new Date(now).toDateString()) return clock(ms);
  const two = (n: number) => String(n).padStart(2, "0");
  return `${d.getDate()} ${MONTHS[d.getMonth()]} ${two(d.getHours())}:${two(d.getMinutes())}`;
}

export const percent = (d: DownloadView) => (d.total > 0 ? Math.floor((d.done / d.total) * 100) : 0);

/** The status pill: its word and its colour class. */
export function statusWord(s: EntryState): [string, string] {
  switch (s.kind) {
    case "in_library": return ["IN LIBRARY", "shared-text"];
    case "update": return ["UPDATE", "accent-text"];
    case "on_disk": return ["ON DISK", ""];
    case "needs_app": return ["NEWER APP", ""];
    case "running": {
      const d = s.dl;
      if (d.state === "paused") return ["PAUSED", ""];
      if (d.state === "verifying") return ["VERIFYING", "accent-text"];
      return [`${percent(d)}%`, d.retry_in_ms != null ? "flame-text" : "accent-text"];
    }
    default: return ["GET", "accent-text"];
  }
}

const mbps = (b: number) => `${(b / MIB).toFixed(1)} MB/s`;
const dur = (s: number) => {
  s = Math.max(1, Math.round(s));
  return s >= 60 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${s} s`;
};

/** The Discover section as it stands in the "discover" view: its head, the line about the catalog, the table. */
export function DiscoverSection({ catalog, shown, state, selected, onSelect, onReadAgain }: {
  catalog: CatalogView | null;
  shown: CatalogEntry[];
  state: (e: CatalogEntry) => EntryState;
  selected: string | null;
  onSelect: (id: string) => void;
  onReadAgain: () => void;
}) {
  const note = (text: string, again: boolean) => (
    <div className="mono dim" style={{ padding: "10px 22px", fontSize: 11, overflowWrap: "anywhere" }}>
      {text}
      {again && <> <button className="link" onClick={onReadAgain}>READ AGAIN</button></>}
    </div>
  );
  let line: ReactNode = null;
  if (catalog === null) line = note("Reading the catalog…", false);
  else if (!catalog.configured) line = note("This build has no model catalog.", false);
  else if (catalog.entries.length === 0 && catalog.error) line = note(`The catalog could not be read: ${catalog.error}`, true);
  else if (catalog.stale)
    line = note(`the list as read ${catalog.fetched_ms == null ? "earlier" : readAt(catalog.fetched_ms)} · reading it again failed: ${catalog.error ?? "unknown"}`, true);
  return (
    <>
      <div className="head" style={{ borderTop: "1px solid var(--l27)", marginTop: 20 }}>
        <span className="title">Discover</span>
        <span className="cap">{catalog?.entries.length ?? 0} IN THE CATALOG</span>
      </div>
      {line}
      {catalog !== null && catalog.entries.length > 0 && (
        <>
          <div className="table-head" style={{ gridTemplateColumns: DCOLS }}>
            <span>NAME</span><span>ARCH</span><span>QUANT</span><span>SIZE</span><span>LICENCE</span><span>STATUS</span>
          </div>
          {shown.length === 0 && <div className="dim mono" style={{ padding: "20px 22px", fontSize: 11 }}>No match in the catalog.</div>}
          {shown.map((e) => {
            const [w, c] = statusWord(state(e));
            return (
              <div key={e.id} role="row" className={`table-row clickable ${e.id === selected ? "selected" : ""}`}
                   style={{ gridTemplateColumns: DCOLS, height: 44 }} onClick={() => onSelect(e.id)}>
                <span className="name-cell"><span className="ell">{e.name}</span></span>
                <span className="mono">{e.architecture}</span>
                <span className="mono">{e.quantization}</span>
                <span className="mono num">{gb(e.size)}</span>
                <span className="mono dim ell" title={e.licence.name}>{e.licence.name}</span>
                <span className={`pill ${c}`}>{w}</span>
              </div>
            );
          })}
        </>
      )}
    </>
  );
}

/** The Discover section closed to its head (the "library" view): nothing is read until it is opened. */
export function DiscoverClosed({ onOpen }: { onOpen: () => void }) {
  return (
    <div className="head" style={{ borderTop: "1px solid var(--l27)", marginTop: 20 }}>
      <span className="title">Discover</span>
      <span className="mono dim" style={{ fontSize: 11 }}>
        Models you can download. Lorren reads the list from the internet only when you open it.
      </span>
      <button className="btn accent" style={{ marginLeft: "auto" }} onClick={onOpen}>OPEN</button>
    </div>
  );
}

/** The right panel for a catalog entry: what it is, then what can be done with it. */
export function EntryPanel({ entry, state, room, refusal, onGet, onPause, onResume, onCancel, onLicence }: {
  entry: CatalogEntry;
  state: EntryState;
  room: DiskRoom | null;
  /** The host's refusal of the last action on this entry, in its own words. */
  refusal: string | null;
  onGet: () => void;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
  onLicence: () => void;
}) {
  const e = entry;
  const small = (t: ReactNode, cls = "") => <span className={`mono ${cls}`} style={{ fontSize: 11 }}>{t}</span>;
  const roomOk = room === null || e.size + GIB <= room.free_bytes;
  const need = `needs ${gb(e.size)}${room ? ` · ${gbn(room.free_bytes)} GB free` : ""}`;
  const getBlock = (
    <>
      {roomOk ? small(need) : small(`${need}: free some space first`, "flame-text")}
      {refusal && small(refusal, "flame-text")}
      <div><button className="btn accent" onClick={onGet} disabled={!roomOk}>GET</button></div>
    </>
  );
  let action: ReactNode;
  switch (state.kind) {
    case "in_library":
      action = <div className="section"><span className="cap">IN LIBRARY</span>{small(`Saved as ${state.dl.file}. Chat can use it now.`, "dim")}</div>;
      break;
    case "on_disk":
      action = <div className="section"><span className="cap">ON DISK</span>{small(`${e.file} is already in your library. Lorren never replaces a file.`, "dim")}</div>;
      break;
    case "needs_app":
      action = (
        <div className="section">
          <span className="cap">NEEDS A NEWER LORREN</span>
          {small(`This model needs Lorren ${e.min_app_version} or newer.`, "dim")}
        </div>
      );
      break;
    case "update":
      action = (
        <div className="section">
          <span className="cap">UPDATE AVAILABLE</span>
          {small(`Version ${e.version} is out; you have version ${state.dl.version}. It is saved beside the old file.`, "dim")}
          {getBlock}
        </div>
      );
      break;
    case "running": {
      const d = state.dl;
      const verifying = d.state === "verifying";
      const p = verifying ? 100 : percent(d);
      const retry = d.state === "downloading" && d.retry_in_ms != null;
      const speed = d.state === "downloading" && d.bytes_per_sec > 0 ? `${mbps(d.bytes_per_sec)} · ${dur((d.total - d.done) / d.bytes_per_sec)} left` : "—";
      const note = verifying ? small("checking the file against its sha256", "dim")
        : retry ? small(`connection lost · retrying in ${Math.ceil((d.retry_in_ms ?? 0) / 1000)} s`, "flame-text")
        : d.state === "paused" ? small(`paused at ${p}%`, "dim") : null;
      action = (
        <div className="section">
          <span className="cap">{verifying ? "VERIFYING" : d.state === "paused" ? "PAUSED" : "DOWNLOADING"}</span>
          <div className="bar thick"><i style={{ width: `${verifying ? 100 : d.total > 0 ? (d.done / d.total) * 100 : 0}%` }} /></div>
          <div className="fact"><span className="k">{gb(verifying ? d.total : d.done)} / {gb(d.total)} · {p}%</span><span className="v num">{speed}</span></div>
          {note}
          {refusal && small(refusal, "flame-text")}
          {!verifying && (
            <span className="rowline">
              {d.state === "paused" && <button className="link" onClick={onResume}>RESUME</button>}
              {d.state === "downloading" && <button className="link" onClick={onPause}>PAUSE</button>}
              <button className="link" onClick={onCancel}>CANCEL</button>
            </span>
          )}
        </div>
      );
      break;
    }
    case "failed":
      action = (
        <div className="section">
          <span className="cap">DISK ROOM</span>
          {state.dl.error && small(state.dl.error, "flame-text")}
          {getBlock}
        </div>
      );
      break;
    default:
      action = <div className="section"><span className="cap">DISK ROOM</span>{getBlock}</div>;
  }
  const from = `${e.source.repo} @ ${e.source.commit.slice(0, 7)}`;
  return (
    <>
      <div className="section">
        <span className="cap">MODEL</span>
        <span className="title">{e.name}</span>
        <span style={{ lineHeight: 1.5, color: "var(--d80)" }}>{e.description}</span>
      </div>
      <div className="section">
        <div className="fact"><span className="k">architecture</span><span className="v">{e.architecture}</span></div>
        <div className="fact"><span className="k">quantization</span><span className="v">{e.quantization}</span></div>
        <div className="fact"><span className="k">size</span><span className="v num">{gb(e.size)}</span></div>
        <div className="fact">
          <span className="k">licence</span>
          <span className="v">
            {/* No element carries an href (no remote content in the window; a middle-click or "open in new window" must
                have nothing to open): the host opens the page in the default browser, only when this is clicked. */}
            <button className="link md-link" title={e.licence.url} onClick={onLicence}>{e.licence.name}</button>
          </span>
        </div>
        <div className="fact"><span className="k">comes from</span><span className="v" title={from}>{from}</span></div>
        <div className="fact"><span className="k">saved as</span><span className="v" title={e.file}>{e.file}</span></div>
      </div>
      {action}
    </>
  );
}
