import { memo, useState } from "react";
import Plot from "react-plotly.js";
import type { ViewerQuantumHeatmapResponse } from "../../api";

type QuantumHeatmapProps = {
  data: ViewerQuantumHeatmapResponse | null;
  selectedSiteId: number | null;
  onSelectSite: (siteId: number) => void;
  cutLine?: { p0: [number, number]; p1: [number, number] } | null;
};

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

export const QuantumHeatmap = memo(function QuantumHeatmap({ data, selectedSiteId, onSelectSite, cutLine = null }: QuantumHeatmapProps) {
  const [vminStr, setVminStr] = useState("");
  const [vmaxStr, setVmaxStr] = useState("");
  const [isLog, setIsLog] = useState(false);

  if (!data) {
    return <div className="viewer-empty">Select a snapshot to see the quantum-layer heatmap.</div>;
  }

  const markerSize = data.site_ids.map((id) => (id === selectedSiteId ? 22 : 18));
  const markerLineColor = data.site_ids.map((id, index) =>
    id === selectedSiteId ? "#ef4444" : "transparent"
  );
  const markerLineWidth = data.site_ids.map((id) => (id === selectedSiteId ? 2.5 : 0));

  const safeLog = (val: number | null) => val === null ? null : Math.log10(Math.max(val, 1e-12));
  const safeLogNum = (val: number) => Math.log10(Math.max(val, 1e-12));

  const plotValues = isLog ? data.values.map(safeLog) : data.values;
  const dataMin = isLog ? safeLogNum(data.color_min) : data.color_min;
  const dataMax = isLog ? safeLogNum(data.color_max) : data.color_max;

  const cmin = vminStr !== "" && !Number.isNaN(Number(vminStr)) ? (isLog ? safeLogNum(Number(vminStr)) : Number(vminStr)) : dataMin;
  const cmax = vmaxStr !== "" && !Number.isNaN(Number(vmaxStr)) ? (isLog ? safeLogNum(Number(vmaxStr)) : Number(vmaxStr)) : dataMax;
  const span = cmax - cmin || 1;

  const tickvals = [
    cmin,
    cmin + span * 0.25,
    cmin + span * 0.5,
    cmin + span * 0.75,
    cmax,
  ];
  const ticktext = isLog ? tickvals.map(v => Math.pow(10, v).toPrecision(3)) : undefined;

  const traces: any[] = [
    {
      type: "scatter",
      mode: "markers",
      x: data.x,
      y: data.y,
      text: data.site_ids.map((id, i) => `site ${id}<br>Value: ${data.values[i]?.toPrecision(4) ?? "N/A"}`),
      hoverinfo: "text",
      customdata: data.site_ids,
      marker: {
        color: plotValues,
        colorscale: TURBO_COLORSCALE,
        cmin: cmin,
        cmax: cmax,
        size: markerSize,
        opacity: 0.8,
        line: {
          color: markerLineColor,
          width: markerLineWidth,
        },
        showscale: true,
        colorbar: {
          title: { text: data.property, side: "top" },
          tickmode: "array",
          tickvals: tickvals,
          ticktext: ticktext,
          tickformat: isLog ? undefined : ".3g",
          thickness: 15,
          len: 0.8,
          outlinewidth: 1,
          outlinecolor: "#111",
        },
      },
    },
  ];

  if (cutLine) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x: [cutLine.p0[0], cutLine.p1[0]],
      y: [cutLine.p0[1], cutLine.p1[1]],
      line: {
        color: "rgba(239, 68, 68, 0.9)",
        width: 3,
        dash: "dash",
      },
      hoverinfo: "none",
    });
  }

  return (
    <div className="viewer-block viewer-block-fill" style={{ position: "relative" }}>
      <div style={{
        position: "absolute",
        top: "16px",
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 10,
        display: "flex",
        gap: "8px",
        background: "rgba(255, 255, 255, 0.85)",
        backdropFilter: "blur(4px)",
        padding: "6px 10px",
        borderRadius: "8px",
        boxShadow: "0 2px 8px rgba(15, 23, 42, 0.08)",
        border: "1px solid rgba(207, 212, 220, 0.6)"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <label style={{ fontSize: "0.7rem", color: "#64748b" }}>vmin</label>
          <input
            type="number"
            value={vminStr}
            onChange={(e) => setVminStr(e.target.value)}
            style={{ width: "60px", fontSize: "0.75rem", padding: "2px 4px", border: "1px solid #cbd5e1", borderRadius: "4px" }}
          />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <label style={{ fontSize: "0.7rem", color: "#64748b" }}>vmax</label>
          <input
            type="number"
            value={vmaxStr}
            onChange={(e) => setVmaxStr(e.target.value)}
            style={{ width: "60px", fontSize: "0.75rem", padding: "2px 4px", border: "1px solid #cbd5e1", borderRadius: "4px" }}
          />
        </div>
        <button
          onClick={() => { setVminStr(""); setVmaxStr(""); setIsLog(false); }}
          style={{ fontSize: "0.7rem", padding: "2px 6px", border: "1px solid #cbd5e1", borderRadius: "4px", background: "#f8fafc", cursor: "pointer", marginLeft: "4px" }}
        >
          Reset
        </button>
        <button
          onClick={() => setIsLog(!isLog)}
          style={{
            fontSize: "0.7rem",
            padding: "2px 8px",
            border: "1px solid #cbd5e1",
            borderRadius: "4px",
            background: isLog ? "#0f172a" : "#f8fafc",
            color: isLog ? "#fff" : "#0f172a",
            cursor: "pointer",
            marginLeft: "4px",
            fontWeight: 500
          }}
        >
          {isLog ? "Log" : "Linear"}
        </button>
      </div>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          margin: { l: 40, r: 10, t: 30, b: 40 },
          paper_bgcolor: "rgba(246, 248, 250, 0.95)",
          plot_bgcolor: "rgba(246, 248, 250, 0.95)",
          xaxis: { visible: false, scaleanchor: "y", scaleratio: 1 },
          yaxis: { visible: false },
          showlegend: false,
          hovermode: "closest",
        }}
        useResizeHandler={true}
        style={{ width: "100%", height: "100%", minHeight: "420px" }}
        config={{ displayModeBar: true, displaylogo: false }}
        onClick={(event) => {
          if (event.points && event.points.length > 0) {
            const siteId = event.points[0].customdata;
            if (siteId !== undefined && siteId !== null) {
              onSelectSite(siteId as number);
            }
          }
        }}
      />
    </div>
  );
});
