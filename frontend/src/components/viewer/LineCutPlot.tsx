import { memo } from "react";

import type { ViewerLdosCutResponse, ViewerSurfaceCutResponse } from "../../api";

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

function buildPolyline(xs: number[], ys: number[], width: number, height: number, padding: number) {
  const xRange = normalize(xs);
  const yRange = normalize(ys);
  const xSpan = xRange.max - xRange.min || 1;
  const ySpan = yRange.max - yRange.min || 1;

  return xs
    .map((x, index) => {
      const px = padding + ((x - xRange.min) / xSpan) * (width - padding * 2);
      const py = height - padding - ((ys[index] - yRange.min) / ySpan) * (height - padding * 2);
      return `${px},${py}`;
    })
    .join(" ");
}

function colorForValue(value: number, low: number, high: number) {
  const span = high - low || 1;
  const t = Math.min(1, Math.max(0, (value - low) / span));
  const hue = 260 - 260 * t;
  const light = 28 + 40 * t;
  return `hsl(${hue} 95% ${light}%)`;
}

function finiteTriples(
  xs: Array<number | null>,
  ys: Array<number | null>,
  zs: Array<number | null>,
) {
  return xs
    .map((x, index) => [x, ys[index], zs[index]] as const)
    .filter((triple): triple is [number, number, number] => triple[0] !== null && triple[1] !== null && triple[2] !== null);
}

export const SurfaceCutPlot = memo(function SurfaceCutPlot({
  data,
  isLoading,
}: {
  data: ViewerSurfaceCutResponse | null;
  isLoading: boolean;
}) {
  if (isLoading) {
    return <div className="viewer-empty">Loading surface cut…</div>;
  }

  if (!data || data.distance_along.length < 2) {
    return <div className="viewer-empty">Surface cut data appears here once a snapshot and cut line are available.</div>;
  }

  const triples = finiteTriples(data.distance_along, data.z, data.values);

  if (triples.length < 2) {
    return <div className="viewer-empty">Not enough surface-cut samples were found for the selected line.</div>;
  }

  const width = 920;
  const height = 250;
  const padding = 24;
  const innerWidth = width - padding * 2;
  const innerHeight = height - padding * 2;
  const xValues = triples.map(([x]) => x);
  const zValues = triples.map(([, z]) => z);
  const vValues = triples.map(([, , value]) => value);
  const xRange = normalize(xValues);
  const zRange = normalize(zValues);
  const valueRange = normalize(vValues);
  const cols = Math.max(18, Math.min(80, Math.round(Math.sqrt(triples.length) * 1.4)));
  const rows = Math.max(10, Math.min(40, Math.round(cols * 0.45)));
  const sumGrid = Array.from({ length: rows }, () => Array<number>(cols).fill(0));
  const countGrid = Array.from({ length: rows }, () => Array<number>(cols).fill(0));

  for (const [x, z, value] of triples) {
    const col = Math.min(cols - 1, Math.max(0, Math.floor(((x - xRange.min) / (xRange.max - xRange.min || 1)) * cols)));
    const rowFromBottom = Math.min(rows - 1, Math.max(0, Math.floor(((z - zRange.min) / (zRange.max - zRange.min || 1)) * rows)));
    const row = rows - 1 - rowFromBottom;
    sumGrid[row][col] += value;
    countGrid[row][col] += 1;
  }

  const cellWidth = innerWidth / cols;
  const cellHeight = innerHeight / rows;

  return (
    <div className="viewer-panel-content">
      <div className="viewer-block-header">
        <h5>Surface cut heatmap: {data.property}</h5>
        <span>width {data.cut_width.toFixed(3)}</span>
      </div>
      <svg className="surface-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label={`Surface cut ${data.property}`}>
        <rect x="0" y="0" width={width} height={height} rx="24" fill="rgba(245, 247, 250, 0.97)" />
        {sumGrid.map((row, rowIndex) =>
          row.map((sum, colIndex) => {
            const count = countGrid[rowIndex][colIndex];
            if (count === 0) {
              return null;
            }
            const value = sum / count;
            return (
              <rect
                key={`${rowIndex}-${colIndex}`}
                x={padding + colIndex * cellWidth}
                y={padding + rowIndex * cellHeight}
                width={Math.max(cellWidth + 0.6, 1)}
                height={Math.max(cellHeight + 0.6, 1)}
                fill={colorForValue(value, valueRange.min, valueRange.max)}
              />
            );
          }),
        )}
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="rgba(71, 85, 105, 0.38)" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="rgba(71, 85, 105, 0.38)" />
      </svg>
    </div>
  );
});

export const LdosCutPlot = memo(function LdosCutPlot({
  data,
  isLoading,
}: {
  data: ViewerLdosCutResponse | null;
  isLoading: boolean;
}) {
  if (isLoading) {
    return <div className="viewer-empty">Loading DOS heatmap…</div>;
  }

  if (!data || data.distance_along.length < 2 || data.energy.length < 2 || data.ldos_matrix.length === 0) {
    return <div className="viewer-empty">DOS heatmap along the cut appears here once the selected snapshot exposes ILDOS data.</div>;
  }

  const finiteValues = data.ldos_matrix.flat().filter((value): value is number => value !== null && Number.isFinite(value));
  const low = finiteValues.length > 0 ? Math.min(...finiteValues) : 0;
  const high = finiteValues.length > 0 ? Math.max(...finiteValues) : 1;
  const width = 420;
  const height = 250;
  const padding = 20;
  const innerWidth = width - padding * 2;
  const innerHeight = height - padding * 2;
  const rowHeight = innerHeight / data.ldos_matrix.length;
  const columnWidth = innerWidth / (data.ldos_matrix[0]?.length || 1);

  return (
    <div className="viewer-panel-content">
      <div className="viewer-block-header">
        <h5>DOS heatmap along line cut</h5>
        <span>{data.snapshot}</span>
      </div>
      <svg className="ldos-cut-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="LDOS cut heatmap">
        <rect x="0" y="0" width={width} height={height} rx="20" fill="rgba(245, 247, 250, 0.97)" />
        {data.ldos_matrix.map((row, rowIndex) =>
          row.map((value, columnIndex) => {
            if (value === null) {
              return null;
            }
            return (
              <rect
                key={`${rowIndex}-${columnIndex}`}
                x={padding + columnIndex * columnWidth}
                y={padding + rowIndex * rowHeight}
                width={Math.max(columnWidth + 0.4, 1)}
                height={Math.max(rowHeight + 0.4, 1)}
                fill={colorForValue(value, low, high)}
              />
            );
          }),
        )}
        {data.overlays.map((overlay) => {
          const pairs = overlay.distance_along
            .map((distance, index) => [distance, overlay.Ui[index]] as const)
            .filter((pair): pair is [number, number] => pair[0] !== null && pair[1] !== null);

          if (pairs.length < 2) {
            return null;
          }

          const points = buildPolyline(
            pairs.map(([x]) => x),
            pairs.map(([, y]) => y),
            width,
            height,
            padding,
          );

          return (
            <polyline
              key={overlay.snapshot}
              fill="none"
              stroke={overlay.is_current ? "rgba(255,255,255,0.95)" : "rgba(15,23,42,0.75)"}
              strokeDasharray={overlay.is_current ? undefined : "5 4"}
              strokeWidth={overlay.is_current ? 2.4 : 1.8}
              points={points}
            />
          );
        })}
      </svg>
    </div>
  );
});
