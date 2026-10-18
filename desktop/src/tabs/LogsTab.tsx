import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { clock } from "../format";
import { api } from "../ipc/api";
import type { LogLine } from "../ipc/bindings/LogLine";
import { HostFailure, useHostFailure } from "./HostFailure";
import "./tabs.css";

const SOURCES = ["all", "app", "engine", "updater"] as const;
type Source = (typeof SOURCES)[number];
/** The host keeps 5,000 lines (Task 6); the tab keeps the same. */
const KEEP = 5000;
/** The reader counts as being at the end of the box within this many pixels of it (R37, as for the Chat transcript in R56). */
const AT_BOTTOM_PX = 24;

export function LogsTab() {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [filter, setFilter] = useState<Source>("all");
  const after = useRef(0);
  const box = useRef<HTMLDivElement>(null);
  /** Whether the reader was at the end of the box at the last scroll event. It has to be known from before new lines make
   *  the box longer: measured afterwards, a reader who is at the end looks like one who is not. A box just opened follows its end. */
  const atBottom = useRef(true);
  const { failure, report, clear } = useHostFailure("logs");

  useEffect(() => {
    let alive = true;
    // A failed call leaves `after` where it was, so the next call asks again for the lines it did not get.
    const tick = () =>
      api.logs(after.current).then(
        (ls) => {
          if (!alive) return;
          clear();
          if (ls.length === 0) return;
          after.current = ls[ls.length - 1].seq;
          setLines((prev) => [...prev, ...ls].slice(-KEEP));
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

  // R37: new lines are followed only while the reader is at the end. One who scrolled up to read is left where they are.
  // A layout effect, as in the Chat tab: it runs before the browser paints, so the new lines are never seen one frame at
  // the old scroll position.
  useLayoutEffect(() => {
    const b = box.current;
    if (b && atBottom.current) b.scrollTop = b.scrollHeight;
  }, [lines, filter]);

  const shown = filter === "all" ? lines : lines.filter((l) => l.source === filter);
  return (
    <section className="col center wide">
      <div className="head">
        <span className="title">Logs</span>
        <span className="filters">
          {SOURCES.map((f) => (
            <button key={f} className={`link ${f === filter ? "on" : ""}`} aria-pressed={f === filter} onClick={() => setFilter(f)}>
              {f.toUpperCase()}
            </button>
          ))}
        </span>
      </div>
      {failure !== null && <HostFailure what="log" reason={failure} />}
      <div
        className="scroll mono logbox"
        ref={box}
        aria-label="log"
        onScroll={(e) => {
          const b = e.currentTarget;
          atBottom.current = b.scrollHeight - b.scrollTop - b.clientHeight < AT_BOTTOM_PX;
        }}
      >
        {shown.map((l) => (
          <div key={l.seq} className={`log-${l.level}`}>
            <span className="dim">{clock(l.time_ms)}</span> <span className="dim">{l.source.padEnd(8, " ")}</span>
            <span>{l.text}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
