export function Placeholder({ tab }: { tab: string }) {
  const text = "The Swarm view arrives in milestone 3. Its engine panels need telemetry the engine adds after m7.";
  return (
    <div className="placeholder">
      <span style={{ fontSize: 15, fontWeight: 600 }}>{tab}</span>
      <span className="mono dim" style={{ fontSize: 11 }}>{text}</span>
    </div>
  );
}
