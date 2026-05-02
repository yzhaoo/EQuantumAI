import { memo, useState } from "react";
import Plot from "react-plotly.js";
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

const TURBO_COLORSCALE: [number, string][] = [
  [0.0, '#30123b'],
  [0.1, '#4662d7'],
  [0.2, '#36aaf9'],
  [0.3, '#1fd0a5'],
  [0.4, '#48f882'],
  [0.5, '#95fb51'],
  [0.6, '#dedd32'],
  [0.7, '#ffa423'],
  [0.8, '#f65f18'],
  [0.9, '#ba2208'],
  [1.0, '#7a0403']
];

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
  const [vminStr, setVminStr] = useState("");
  const [vmaxStr, setVmaxStr] = useState("");
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

  const xValues = triples.map(([x]) => x);
  const zValues = triples.map(([, z]) => z);
  const vValues = triples.map(([, , value]) => value);
  const xRange = normalize(xValues);
  const zRange = normalize(zValues);
  const cols = Math.max(18, Math.min(80, Math.round(Math.sqrt(triples.length) * 1.4)));
  const rows = Math.max(10, Math.min(40, Math.round(cols * 0.45)));
  const sumGrid = Array.from({ length: rows }, () => Array<number>(cols).fill(0));
  const countGrid = Array.from({ length: rows }, () => Array<number>(cols).fill(0));

  for (const [x, z, value] of triples) {
    const col = Math.min(cols - 1, Math.max(0, Math.floor(((x - xRange.min) / (xRange.max - xRange.min || 1)) * cols)));
    const row = Math.min(rows - 1, Math.max(0, Math.floor(((z - zRange.min) / (zRange.max - zRange.min || 1)) * rows)));
    sumGrid[row][col] += value;
    countGrid[row][col] += 1;
  }

  const zArray = sumGrid.map((row, rowIndex) =>
    row.map((sum, colIndex) => {
      const count = countGrid[rowIndex][colIndex];
      return count === 0 ? null : sum / count;
    }),
  );

  const xVals = Array.from({ length: cols }, (_, i) => xRange.min + ((i + 0.5) * (xRange.max - xRange.min)) / cols);
  const yVals = Array.from({ length: rows }, (_, i) => zRange.min + ((i + 0.5) * (zRange.max - zRange.min)) / rows);

  const zmin = vminStr !== "" && !Number.isNaN(Number(vminStr)) ? Number(vminStr) : undefined;
  const zmax = vmaxStr !== "" && !Number.isNaN(Number(vmaxStr)) ? Number(vmaxStr) : undefined;

  return (
    <div className="viewer-panel-content">
      <div className="viewer-block-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h5>Surface cut heatmap: {data.property}</h5>
          <span>width {data.cut_width.toFixed(3)}</span>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
            <label style={{ fontSize: "0.7rem", color: "#64748b" }}>vmin</label>
            <input
              type="number"
              value={vminStr}
              onChange={(e) => setVminStr(e.target.value)}
              style={{ width: "50px", fontSize: "0.75rem", padding: "2px 4px", border: "1px solid #cbd5e1", borderRadius: "4px" }}
            />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
            <label style={{ fontSize: "0.7rem", color: "#64748b" }}>vmax</label>
            <input
              type="number"
              value={vmaxStr}
              onChange={(e) => setVmaxStr(e.target.value)}
              style={{ width: "50px", fontSize: "0.75rem", padding: "2px 4px", border: "1px solid #cbd5e1", borderRadius: "4px" }}
            />
          </div>
          <button
            onClick={() => { setVminStr(""); setVmaxStr(""); }}
            style={{ fontSize: "0.7rem", padding: "2px 6px", border: "1px solid #cbd5e1", borderRadius: "4px", background: "#f8fafc", cursor: "pointer" }}
          >
            Reset
          </button>
        </div>
      </div>
      <Plot
        data={[
          {
            type: "heatmap",
            z: zArray,
            x: xVals,
            y: yVals,
            colorscale: TURBO_COLORSCALE,
            zmin: zmin,
            zmax: zmax,
            hoverinfo: "x+y+z",
            colorbar: {
              title: { text: data.property, side: "top" },
              tickformat: ".3g",
              thickness: 15,
              len: 1,
              outlinewidth: 1,
              outlinecolor: "#111",
            },
          },
        ]}
        layout={{
          autosize: true,
          margin: { l: 40, r: 10, t: 10, b: 40 },
          paper_bgcolor: "rgba(246, 248, 250, 0)",
          plot_bgcolor: "rgba(246, 248, 250, 0.95)",
          xaxis: { title: { text: "Distance" }, zeroline: false },
          yaxis: { title: { text: "Z" }, zeroline: false },
        }}
        useResizeHandler={true}
        style={{ width: "100%", height: "250px" }}
        config={{ displayModeBar: true, displaylogo: false }}
      />
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
  const [vminStr, setVminStr] = useState("");
  const [vmaxStr, setVmaxStr] = useState("");
  if (isLoading) {
    return <div className="viewer-empty">Loading DOS heatmap…</div>;
  }

  if (!data || data.distance_along.length < 2 || data.energy.length < 2 || data.ldos_matrix.length === 0) {
    return <div className="viewer-empty">DOS heatmap along the cut appears here once the selected snapshot exposes ILDOS data.</div>;
  }

  const zmin = vminStr !== "" && !Number.isNaN(Number(vminStr)) ? Number(vminStr) : undefined;
  const zmax = vmaxStr !== "" && !Number.isNaN(Number(vmaxStr)) ? Number(vmaxStr) : undefined;

  const traces: any[] = [
    {
      type: "heatmap",
      z: data.ldos_matrix,
      x: data.distance_along,
      y: data.energy,
      colorscale: TURBO_COLORSCALE,
      zmin: zmin,
      zmax: zmax,
      hoverinfo: "x+y+z",
      colorbar: {
        title: { text: "LDOS", side: "top" },
        tickformat: ".3g",
        thickness: 15,
        len: 1,
        outlinewidth: 1,
        outlinecolor: "#111",
      },
    },
  ];

  for (const overlay of data.overlays) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x: overlay.distance_along,
      y: overlay.Ui,
      line: {
        color: overlay.is_current ? "rgba(255,255,255,0.95)" : "rgba(15,23,42,0.75)",
        width: overlay.is_current ? 2.4 : 1.8,
        dash: overlay.is_current ? "solid" : "dash",
      },
      hoverinfo: "none",
    });
  }

  return (
    <div className="viewer-panel-content">
      <div className="viewer-block-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h5>DOS heatmap along line cut</h5>
          <span>{data.snapshot}</span>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
            <label style={{ fontSize: "0.7rem", color: "#64748b" }}>vmin</label>
            <input
              type="number"
              value={vminStr}
              onChange={(e) => setVminStr(e.target.value)}
              style={{ width: "50px", fontSize: "0.75rem", padding: "2px 4px", border: "1px solid #cbd5e1", borderRadius: "4px" }}
            />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
            <label style={{ fontSize: "0.7rem", color: "#64748b" }}>vmax</label>
            <input
              type="number"
              value={vmaxStr}
              onChange={(e) => setVmaxStr(e.target.value)}
              style={{ width: "50px", fontSize: "0.75rem", padding: "2px 4px", border: "1px solid #cbd5e1", borderRadius: "4px" }}
            />
          </div>
          <button
            onClick={() => { setVminStr(""); setVmaxStr(""); }}
            style={{ fontSize: "0.7rem", padding: "2px 6px", border: "1px solid #cbd5e1", borderRadius: "4px", background: "#f8fafc", cursor: "pointer" }}
          >
            Reset
          </button>
        </div>
      </div>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          margin: { l: 50, r: 10, t: 10, b: 40 },
          paper_bgcolor: "rgba(246, 248, 250, 0)",
          plot_bgcolor: "rgba(246, 248, 250, 0.95)",
          xaxis: { title: { text: "Distance" }, zeroline: false },
          yaxis: { title: { text: "Energy" }, zeroline: false },
          showlegend: false,
        }}
        useResizeHandler={true}
        style={{ width: "100%", height: "300px" }}
        config={{ displayModeBar: true, displaylogo: false }}
      />
    </div>
  );
});
