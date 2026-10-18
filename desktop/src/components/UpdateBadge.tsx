import { useEffect, useState, type ReactNode } from "react";
import { api } from "../ipc/api";
import type { UpdateState } from "../ipc/bindings/UpdateState";
import { useHostFailure } from "../tabs/HostFailure";

/** The update state, in the status bar. Nothing at all when updates are not configured.
 *
 *  A host call that fails (R64) is shown ahead of the action and logged once, as the tabs do (R63), not swallowed. The
 *  notices are spans, so the action stays the last button in the bar, and they show even when no state has arrived or
 *  the state is `not_configured`: the call that failed may be the one that would have said. A state event takes every
 *  notice away (it shows the host answering and the listener working); a snapshot takes only the snapshot's own, as it
 *  says nothing about a listener that failed; a press that works takes its own. */
export function UpdateBadge() {
  const [s, setS] = useState<UpdateState | null>(null);
  const read = useHostFailure("updateState");
  const follow = useHostFailure("onUpdateState");
  const check = useHostFailure("checkUpdate");
  const install = useHostFailure("installUpdate");
  useEffect(() => {
    let alive = true;
    api.updateState().then(
      (next) => {
        if (!alive) return;
        setS(next);
        read.clear();
      },
      (e) => {
        if (alive) read.report(e);
      },
    );
    const unlisten = api
      .onUpdateState((next) => {
        setS(next);
        read.clear();
        follow.clear();
        check.clear();
        install.clear();
      })
      .catch((e) => {
        if (alive) follow.report(e);
        return () => {}; // the cleanup below still calls whatever this resolves to (R51)
      });
    return () => {
      alive = false;
      unlisten.then((f) => f());
    };
  }, [read.report, read.clear, follow.report, follow.clear, check.clear, install.clear]);
  const checkNow = () => api.checkUpdate().then(check.clear, check.report);
  const installNow = () => api.installUpdate().then(install.clear, install.report);
  return (
    <>
      {read.failure !== null && <Notice what="could not read the update state" reason={read.failure} />}
      {follow.failure !== null && <Notice what="could not follow the update state" reason={follow.failure} />}
      {check.failure !== null && <Notice what="could not check for updates" reason={check.failure} />}
      {install.failure !== null && <Notice what="could not install the update" reason={install.failure} />}
      {s && action(s, checkNow, installNow)}
    </>
  );
}

/** One failed host call. The bar is 30 px high and the reason can be long (a plugin's error text), so the line is cut
 *  at the width the engine's last error gets and the title has all of it. */
function Notice({ what, reason }: { what: string; reason: string }) {
  const text = `${what}: ${reason}`;
  return (
    <span className="flame-text ell" role="alert" title={text} style={{ maxWidth: 520 }}>
      {text}
    </span>
  );
}

function action(s: UpdateState, check: () => void, install: () => void): ReactNode {
  switch (s.state) {
    case "not_configured":
      return null;
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
        <button className="link accent-text" onClick={install}>
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
