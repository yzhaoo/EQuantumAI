import type { ViewerQuantumHeatmapResponse } from "../../api";

type QuantumHeatmapProps = {
  data: ViewerQuantumHeatmapResponse | null;
  selectedSiteId: number | null;
  onSelectSite: (siteId: number) => void;
};

const SVG_SIZE = 420;
const PADDING = 24;

function clamp(value: number, low: number, high: number) {
  return Math.min(high, Math.max(low, value));
}

function colorForValue(value: number | null, low: number, high: number) {
  if (value === null || Number.isNaN(value)) {
    return "rgba(148, 163, 184, 0.55)";
  }

  const span = high - low || 1;
  const t = clamp((value - low) / span, 0, 1);
  const hue = 220 - 200 * t;
  const light = 58 - 18 * t;
  return `hsl(${hue} 82% ${light}%)`;
}

function normalize(values: Array<number | null>) {
  const finite = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (finite.length === 0) {
    return { min: 0, max: 1 };
  }
  return {
    min: Math.min(...finite),
    max: Math.max(...finite),
  };
}

export function QuantumHeatmap({ data, selectedSiteId, onSelectSite }: QuantumHeatmapProps) {
  if (!data) {
    return <div className="viewer-empty">Select a snapshot to see the quantum-layer heatmap.</div>;
  }

  const xRange = normalize(data.x);
  const yRange = normalize(data.y);
  const xSpan = xRange.max - xRange.min || 1;
  const ySpan = yRange.max - yRange.min || 1;

  return (
    <div className="viewer-block">
      <div className="viewer-block-header">
        <h5>Quantum Heatmap</h5>
        <span>{data.property}</span>
      </div>
      <svg className="heatmap-svg" viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`} role="img" aria-label={`Quantum heatmap for ${data.property}`}>
        <rect x={0} y={0} width={SVG_SIZE} height={SVG_SIZE} rx={20} fill="rgba(246, 248, 250, 0.95)" />
        {data.site_ids.map((siteId, index) => {
          const x = data.x[index];
          const y = data.y[index];
          if (x === null || y === null) {
            return null;
          }

          const svgX = PADDING + ((x - xRange.min) / xSpan) * (SVG_SIZE - PADDING * 2);
          const svgY = SVG_SIZE - PADDING - ((y - yRange.min) / ySpan) * (SVG_SIZE - PADDING * 2);
          const selected = siteId === selectedSiteId;

          return (
            <circle
              key={`${siteId}-${index}`}
              cx={svgX}
              cy={svgY}
              r={selected ? 7 : 5}
              fill={colorForValue(data.values[index], data.color_min, data.color_max)}
              stroke={selected ? "#ef4444" : data.qprime_mask[index] ? "rgba(15, 23, 42, 0.48)" : "rgba(148, 163, 184, 0.38)"}
              strokeWidth={selected ? 2.5 : 1}
              onClick={() => onSelectSite(siteId)}
            >
              <title>{`site ${siteId}`}</title>
            </circle>
          );
        })}
      </svg>
      <div className="viewer-legend">
        <span>{data.color_min.toPrecision(4)}</span>
        <div className="viewer-legend-bar" />
        <span>{data.color_max.toPrecision(4)}</span>
      </div>
    </div>
  );
}
