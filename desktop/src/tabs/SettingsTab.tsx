import { useEffect, useRef, useState } from "react";
import { api } from "../ipc/api";
import type { UpdateMode } from "../ipc/bindings/UpdateMode";
import { HostFailure, useHostFailure } from "./HostFailure";
import "./tabs.css";

const CHOICES: [UpdateMode, string, string][] = [
  ["auto", "AUTO", "Downloads updates in the background and installs them when you restart."],
  ["notify", "NOTIFY", "Tells you when there is one, and downloads only when you say."],
  ["off", "OFF", "Never checks."],
];

/** Settings: the update choice. No choice is shown selected until the host has said which one is stored. A choice is shown
 *  selected at once; if the host cannot keep it, the selection goes back to the stored one and the reason is shown. */
export function SettingsTab() {
  const [mode, setMode] = useState<UpdateMode | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  /** The mode the host holds, as far as this tab knows. */
  const stored = useRef<UpdateMode | null>(null);
  const read = useHostFailure("updateMode");

  useEffect(() => {
    let alive = true;
    api.updateMode().then(
      (m) => {
        if (!alive) return;
        stored.current = m;
        setMode(m);
        read.clear();
      },
      (e) => {
        if (alive) read.report(e);
      },
    );
    return () => {
      alive = false;
    };
  }, [read.report, read.clear]);

  const choose = (next: UpdateMode) => {
    if (next === mode) return;
    setMode(next);
    setSaveError(null);
    api.setUpdateMode(next).then(
      () => {
        stored.current = next;
      },
      (e) => {
        setMode(stored.current);
        setSaveError(e instanceof Error ? e.message : String(e));
      },
    );
  };

  return (
    <>
      <nav className="col left">
        <div className="head"><span className="title">Settings</span></div>
        <div className="scroll">
          <button className="chat-item selected" aria-current="true"><span className="cap">UPDATES</span></button>
        </div>
      </nav>
      <section className="col center wide">
        <div className="head">
          <span className="title">Updates</span>
          <span className="cap">WHEN A NEW VERSION COMES OUT</span>
        </div>
        <div className="scroll">
          {read.failure !== null && <HostFailure what="update choice" reason={read.failure} />}
          <div role="radiogroup" aria-label="Updates">
            {CHOICES.map(([k, label, text]) => (
              <button key={k} className={`choice ${mode === k ? "selected" : ""}`} role="radio" aria-checked={mode === k}
                      onClick={() => choose(k)}>
                <i className="radio" />
                <span style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-start" }}>
                  <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span className="mono" style={{ fontSize: 11, letterSpacing: "0.08em" }}>{label}</span>
                    {k === "auto" && <span className="pill">DEFAULT</span>}
                  </span>
                  <span className="mono dim" style={{ fontSize: 11 }}>{text}</span>
                </span>
              </button>
            ))}
          </div>
          {saveError !== null && (
            <div className="tab-notice flame-text" role="alert">{`Could not save the choice: ${saveError}`}</div>
          )}
        </div>
      </section>
    </>
  );
}
