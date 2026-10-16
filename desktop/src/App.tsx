import { useEffect, useState, type ReactNode } from "react";
import "./fonts.css";
import "./theme.css";
import { Placeholder } from "./components/Placeholder";
import { StatusBar } from "./components/StatusBar";
import { TopBar, type Tab } from "./components/TopBar";
import { api } from "./ipc/api";
import type { Overview } from "./ipc/bindings/Overview";

export function App() {
  const [tab, setTab] = useState<Tab>("Chat");
  const [overview, setOverview] = useState<Overview | null>(null);
  const [error, setError] = useState<string>();

  useEffect(() => {
    let alive = true;
    let logged: string | undefined; // the failure already reported: the poll repeats every 2 s, the log must not
    const tick = () =>
      api.overview().then(
        (o) => {
          if (!alive) return;
          logged = undefined; // a failure after this one is a new outage and is reported again
          setError(undefined);
          setOverview(o);
        },
        (e) => {
          if (!alive) return;
          // A rejection is an IPC-level fault (a misnamed or unregistered command, an ACL, serialisation), not first paint.
          const message = e instanceof Error ? e.message : String(e);
          setError(message);
          if (message !== logged) {
            logged = message;
            console.error("overview failed:", message);
          }
        },
      );
    tick();
    const timer = setInterval(tick, 2000);
    const unlisten = api
      .onEngineStatus((engine) => setOverview((o) => (o ? { ...o, engine } : o)))
      .catch((e) => {
        console.error("engine-status listener failed:", e);
        return () => {}; // the cleanup below still calls whatever this resolves to
      });
    return () => {
      alive = false;
      clearInterval(timer);
      unlisten.then((f) => f());
    };
  }, []);

  return (
    <div className="app">
      <TopBar tab={tab} onTab={setTab} overview={overview} error={error} />
      <main className="content">{renderTab(tab, overview)}</main>
      <StatusBar overview={overview} />
    </div>
  );
}

function renderTab(tab: Tab, _overview: Overview | null): ReactNode {
  return <Placeholder tab={tab} />;
}
