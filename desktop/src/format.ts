export const mb = (b: number) => `${Math.round(b / 1048576).toLocaleString("en-US")} MB`;
export const gb = (b: number) => `${(b / 1073741824).toFixed(1)} GB`;
export const secs = (ms: number | null | undefined) =>
  ms == null ? "—" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
export const clock = (ms: number) => new Date(ms).toLocaleTimeString("en-GB", { hour12: false });
export const since = (startMs: number, now = Date.now()) => {
  const m = Math.max(0, Math.floor((now - startMs) / 60000));
  return m < 60 ? `${m}m` : `${Math.floor(m / 60)}h ${m % 60}m`;
};
export const commas = (n: number) => n.toLocaleString("en-US");
