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
  const download = useHostFailure("downloadUpdate");
  useEffect(() => {
    let alive = true;
    /** An event has arrived. It is newer than a snapshot that is still on its way, so that snapshot is dropped. */
    let heard = false;
    // The listener goes in FIRST, and the snapshot is read once it is registered (or has failed to be): read first, an
    // event emitted before the listener was in place would be lost, and an answer that landed after an event would put
    // the older state over the newer one.
    const unlisten = api
      .onUpdateState((next) => {
        heard = true;
        setS(next);
        read.clear();
        follow.clear();
        check.clear();
        install.clear();
        download.clear();
      })
      .catch((e) => {
        if (alive) follow.report(e);
        return () => {}; // the cleanup below still calls whatever this resolves to (R51)
      });
    void unlisten.then(() =>
      api.updateState().then(
        (next) => {
          if (!alive || heard) return;
          setS(next);
          read.clear();
        },
        (e) => {
          if (alive) read.report(e);
        },
      ),
    );
    return () => {
      alive = false;
      unlisten.then((f) => f());
    };
  }, [read.report, read.clear, follow.report, follow.clear, check.clear, install.clear, download.clear]);
  const checkNow = () => api.checkUpdate().then(check.clear, check.report);
  const installNow = () => api.installUpdate().then(install.clear, install.report);
  const downloadNow = () => api.downloadUpdate().then(download.clear, download.report);
  return (
    <>
      {read.failure !== null && <Notice what="could not read the update state" reason={read.failure} />}
      {follow.failure !== null && <Notice what="could not follow the update state" reason={follow.failure} />}
      {check.failure !== null && <Notice what="could not check for updates" reason={check.failure} />}
      {install.failure !== null && <Notice what="could not install the update" reason={install.failure} />}
      {download.failure !== null && <Notice what="could not download the update" reason={download.failure} />}
      {s && action(s, checkNow, installNow, downloadNow)}
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

function action(s: UpdateState, check: () => void, install: () => void, download: () => void): ReactNode {
  switch (s.state) {
    case "off":
      return <span>updates off</span>;
    case "available":
      return <button className="link" onClick={download}>update {s.version} available · DOWNLOAD</button>;
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
        <button className="link flame-text" title={`${s.reason}; nothing was installed`} onClick={check}>
          {`update ${s.version} refused · ${s.label} · check again`}
        </button>
      );
    case "failed":
      return <button className="link" title={s.message} onClick={check}>update check failed · retry</button>;
  }
}
