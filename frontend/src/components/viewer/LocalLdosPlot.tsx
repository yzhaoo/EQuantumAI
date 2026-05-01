import type { ViewerSiteLdosResponse } from "../../api";

type LocalLdosPlotProps = {
  data: ViewerSiteLdosResponse | null;
  isLoading?: boolean;
};

function buildPolylinePoints(xs: number[], ys: number[], width: number, height: number, padding: number) {
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  return xs
    .map((x, index) => {
      const px = padding + ((x - minX) / spanX) * (width - padding * 2);
      const py = height - padding - ((ys[index] - minY) / spanY) * (height - padding * 2);
      return `${px},${py}`;
    })
    .join(" ");
}

export function LocalLdosPlot({ data, isLoading = false }: LocalLdosPlotProps) {
  if (isLoading) {
    return <div className="viewer-empty">Loading site LDOS…</div>;
  }

  if (!data) {
    return <div className="viewer-empty">Click a site in the heatmap to inspect the local LDOS.</div>;
  }

  const pairs = data.energy
    .map((energy, index) => [energy, data.ldos[index]] as const)
    .filter((pair): pair is [number, number] => pair[0] !== null && pair[1] !== null);

  if (pairs.length < 2) {
    return <div className="viewer-empty">No LDOS data is available for site {data.site_id}.</div>;
  }

  const xs = pairs.map(([x]) => x);
  const ys = pairs.map(([, y]) => y);
  const width = 420;
  const height = 240;
  const padding = 28;
  const polylinePoints = buildPolylinePoints(xs, ys, width, height, padding);

  return (
    <div className="viewer-block">
      <div className="viewer-block-header">
        <h5>Local LDOS</h5>
        <span>site {data.site_id}</span>
      </div>
      <svg className="ldos-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`LDOS plot for site ${data.site_id}`}>
        <rect x={0} y={0} width={width} height={height} rx={20} fill="rgba(246, 248, 250, 0.95)" />
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="rgba(71, 85, 105, 0.65)" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="rgba(71, 85, 105, 0.65)" />
        <polyline fill="none" stroke="#0f766e" strokeWidth={2.4} points={polylinePoints} />
      </svg>
      <div className="viewer-metrics">
        <span>Ui: {data.Ui ?? "n/a"}</span>
        <span>ni: {data.ni ?? "n/a"}</span>
        <span>LDOS(0): {data.ldos_at_0 ?? "n/a"}</span>
        <span>LDOS(Ui): {data.ldos_at_Ui ?? "n/a"}</span>
      </div>
    </div>
  );
}
