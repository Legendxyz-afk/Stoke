import { useEffect, useState } from "react";
import { api } from "../ipc/api";
import type { UpdateState } from "../ipc/bindings/UpdateState";

/** The update state, in the status bar. Nothing at all when updates are not configured. */
export function UpdateBadge() {
  const [s, setS] = useState<UpdateState | null>(null);
  useEffect(() => {
    api.updateState().then(setS).catch(() => {});
    const un = api.onUpdateState(setS);
    return () => {
      un.then((f) => f());
    };
  }, []);
  const check = () => api.checkUpdate().catch(() => {});
  if (!s || s.state === "not_configured") return null;
  switch (s.state) {
    case "idle":
      return <button className="link" onClick={check}>check for updates</button>;
    case "checking":
      return <span>checking for updates…</span>;
    case "up_to_date":
      return <button className="link" onClick={check}>up to date ({s.version})</button>;
    case "downloading":
      return <span>downloading {s.version}{s.total ? ` ${Math.round((s.done / s.total) * 100)}%` : ""}</span>;
    case "ready":
      return (
        <button className="link accent-text" onClick={() => api.installUpdate().catch(() => {})}>
          update {s.version} ready · RESTART TO INSTALL
        </button>
      );
    case "refused":
      return (
        <button className="link flame-text" title={s.reason} onClick={check}>
          update {s.version} refused: signature did not verify · check again
        </button>
      );
    case "failed":
      return <button className="link" title={s.message} onClick={check}>update check failed · retry</button>;
  }
}
