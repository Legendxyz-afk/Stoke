import { useCallback, useRef, useState } from "react";

/** The failure of a host call that a tab makes (R63), kept apart from the data the tab shows so that a call that failed
 *  reads as a failure, not as an empty library, an empty list or an empty log.
 *  `report` shows the reason and logs it, once per distinct message however often a poll repeats. `clear` is for the
 *  answer that works; the next failure after it is a new outage and is reported again (R51's rule for overview()). */
export function useHostFailure(call: string) {
  const [failure, setFailure] = useState<string | null>(null);
  const reported = useRef<string | null>(null);

  const report = useCallback((e: unknown) => {
    // Tauri rejects with the command's error string, a JavaScript fault with an Error: it is the message that counts.
    const message = e instanceof Error ? e.message : String(e);
    if (message === reported.current) return;
    reported.current = message;
    setFailure(message);
    console.error(`${call} failed:`, message);
  }, [call]);

  const clear = useCallback(() => {
    if (reported.current === null) return; // nothing is showing: a good poll must not set state every second
    reported.current = null;
    setFailure(null);
  }, []);

  return { failure, report, clear };
}

/** The line a tab shows in place of what it could not load. */
export function HostFailure({ what, reason }: { what: string; reason: string }) {
  return <div className="tab-notice flame-text" role="alert">{`could not load the ${what}: ${reason}`}</div>;
}
