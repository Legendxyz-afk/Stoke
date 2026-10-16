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

  useEffect(() => {
    let alive = true;
    const tick = () => api.overview().then((o) => alive && setOverview(o)).catch(() => {});
    tick();
    const timer = setInterval(tick, 2000);
    const unlisten = api.onEngineStatus((engine) => setOverview((o) => (o ? { ...o, engine } : o)));
    return () => {
      alive = false;
      clearInterval(timer);
      unlisten.then((f) => f());
    };
  }, []);

  return (
    <div className="app">
      <TopBar tab={tab} onTab={setTab} overview={overview} />
      <main className="content">{renderTab(tab, overview)}</main>
      <StatusBar overview={overview} />
    </div>
  );
}

function renderTab(tab: Tab, _overview: Overview | null): ReactNode {
  return <Placeholder tab={tab} />;
}
