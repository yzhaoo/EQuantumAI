import { memo } from "react";
import Plot from "react-plotly.js";
import type { ViewerSiteLdosResponse } from "../../api";

type LocalLdosPlotProps = {
  data: ViewerSiteLdosResponse | null;
  isLoading?: boolean;
};

function Chart({
  title,
  subtitle,
  xs,
  ysA,
  ysB,
  colorA,
  colorB,
  labelA,
  labelB,
  markerX,
  markerColor,
}: {
  title: string;
  subtitle: string;
  xs: number[];
  ysA: number[];
  ysB?: number[];
  colorA: string;
  colorB?: string;
  labelA: string;
  labelB?: string;
  markerX?: number | null;
  markerColor?: string;
}) {
  const traces: any[] = [
    {
      x: xs,
      y: ysA,
      type: "scatter",
      mode: "lines",
      name: labelA,
      line: { color: colorA, width: 2.4 },
    },
  ];

  if (ysB && colorB && labelB) {
    traces.push({
      x: xs,
      y: ysB,
      type: "scatter",
      mode: "lines",
      name: labelB,
      line: { color: colorB, width: 2.2 },
    });
  }

  const shapes: any[] = [
    {
      type: "line",
      x0: 0,
      x1: 0,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: "rgba(15, 23, 42, 0.28)", width: 1, dash: "dot" },
    },
  ];

  if (markerX !== null && markerX !== undefined) {
    shapes.push({
      type: "line",
      x0: markerX,
      x1: markerX,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: markerColor ?? "rgba(239, 68, 68, 0.55)", width: 1.5, dash: "dot" },
    });
  }

  return (
    <div className="viewer-subplot" style={{ display: "flex", flexDirection: "column" }}>
      <div className="viewer-subplot-header">
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </div>
      <div style={{ flex: 1, minHeight: "180px" }}>
        <Plot
          data={traces}
          layout={{
            autosize: true,
            margin: { l: 40, r: 10, t: 10, b: 20 },
            paper_bgcolor: "rgba(246, 248, 250, 0)",
            plot_bgcolor: "rgba(246, 248, 250, 0.96)",
            xaxis: { zeroline: false },
            yaxis: { zeroline: false },
            showlegend: true,
            legend: { orientation: "h", y: -0.2, x: 0 },
            shapes,
          }}
          useResizeHandler={true}
          style={{ width: "100%", height: "100%" }}
          config={{ displayModeBar: false, displaylogo: false }}
        />
      </div>
    </div>
  );
}

export const LocalLdosPlot = memo(function LocalLdosPlot({ data, isLoading = false }: LocalLdosPlotProps) {
  if (isLoading) {
    return <div className="viewer-empty">Loading site LDOS…</div>;
  }

  if (!data) {
    return <div className="viewer-empty">Click a site in the heatmap to inspect the local LDOS and local solver consistency.</div>;
  }

  const ldosPairs = data.energy
    .map((energy, index) => [energy, data.ldos[index]] as const)
    .filter((pair): pair is [number, number] => pair[0] !== null && pair[1] !== null);

  const consistencyPairs = data.consistency_delta_u
    .map((deltaU, index) => [deltaU, data.consistency_poisson[index], data.consistency_integrated[index]] as const)
    .filter(
      (pair): pair is [number, number, number] =>
        pair[0] !== null && pair[1] !== null && pair[2] !== null,
    );

  if (ldosPairs.length < 2) {
    return <div className="viewer-empty">No LDOS data is available for site {data.site_id}.</div>;
  }

  const ldosXs = ldosPairs.map(([x]) => x);
  const ldosYs = ldosPairs.map(([, y]) => y);
  const consistencyXs = consistencyPairs.map(([x]) => x);
  const consistencyPoisson = consistencyPairs.map(([, y]) => y);
  const consistencyIntegrated = consistencyPairs.map(([, , y]) => y);

  return (
    <div className="viewer-panel-content">
      <div className="viewer-block-header">
        <h5>Onsite Plots</h5>
        <span>site {data.site_id}</span>
      </div>
      <div className="viewer-stack-plots" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        <Chart
          title="Local consistency"
          subtitle={data.Ci !== null ? `Ci ${data.Ci.toPrecision(4)}` : "Ci n/a"}
          xs={consistencyXs.length > 1 ? consistencyXs : ldosXs}
          ysA={consistencyPoisson.length > 1 ? consistencyPoisson : ldosYs}
          ysB={consistencyIntegrated.length > 1 ? consistencyIntegrated : undefined}
          colorA="#1d4ed8"
          colorB="#f97316"
          labelA="Poisson"
          labelB="Integrated LDOS"
          markerX={data.dU_solution}
          markerColor="rgba(15, 23, 42, 0.95)"
        />
        <Chart
          title="LDOS"
          subtitle={data.snapshot.replace(".npz", "")}
          xs={ldosXs}
          ysA={ldosYs}
          colorA="#0f766e"
          labelA="LDOS"
          markerX={data.Ui}
          markerColor="rgba(239, 68, 68, 0.55)"
        />
      </div>
      <div className="viewer-metrics" style={{ marginTop: "12px" }}>
        <span>Ui: {data.Ui ?? "n/a"}</span>
        <span>ni: {data.ni ?? "n/a"}</span>
        <span>Ci: {data.Ci ?? "n/a"}</span>
        <span>LDOS(0): {data.ldos_at_0 ?? "n/a"}</span>
        <span>LDOS(Ui): {data.ldos_at_Ui ?? "n/a"}</span>
      </div>
    </div>
  );
});
