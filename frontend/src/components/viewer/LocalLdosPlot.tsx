import { memo } from "react";

import type { ViewerSiteLdosResponse } from "../../api";

type LocalLdosPlotProps = {
  data: ViewerSiteLdosResponse | null;
  isLoading?: boolean;
};

const WIDTH = 420;
const HEIGHT = 180;
const PADDING_X = 36;
const PADDING_Y = 24;

function normalize(values: number[]) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) {
    return { min: 0, max: 1 };
  }
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  if (min === max) {
    return { min: min - 1, max: max + 1 };
  }
  return { min, max };
}

function scaleX(x: number, range: { min: number; max: number }) {
  return PADDING_X + ((x - range.min) / (range.max - range.min || 1)) * (WIDTH - PADDING_X * 2);
}

function scaleY(y: number, range: { min: number; max: number }) {
  return HEIGHT - PADDING_Y - ((y - range.min) / (range.max - range.min || 1)) * (HEIGHT - PADDING_Y * 2);
}

function buildPolyline(xs: number[], ys: number[]) {
  const xRange = normalize(xs);
  const yRange = normalize(ys);
  return xs
    .map((x, index) => `${scaleX(x, xRange)},${scaleY(ys[index], yRange)}`)
    .join(" ");
}

function axisX(value: number, range: { min: number; max: number }) {
  const scaled = scaleX(value, range);
  return Math.min(WIDTH - PADDING_X, Math.max(PADDING_X, scaled));
}

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
  const xRange = normalize(xs);
  const yRange = normalize([...ysA, ...(ysB ?? [])]);
  const polyA = buildPolyline(xs, ysA);
  const polyB = ysB ? buildPolyline(xs, ysB) : null;
  const zeroX = axisX(0, xRange);
  const marker = markerX === null || markerX === undefined ? null : axisX(markerX, xRange);

  return (
    <div className="viewer-subplot">
      <div className="viewer-subplot-header">
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </div>
      <svg className="ldos-svg" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none" role="img" aria-label={title}>
        <rect x={0} y={0} width={WIDTH} height={HEIGHT} rx={18} fill="rgba(246, 248, 250, 0.96)" />
        <line x1={PADDING_X} y1={HEIGHT - PADDING_Y} x2={WIDTH - PADDING_X} y2={HEIGHT - PADDING_Y} stroke="rgba(71, 85, 105, 0.5)" />
        <line x1={PADDING_X} y1={PADDING_Y} x2={PADDING_X} y2={HEIGHT - PADDING_Y} stroke="rgba(71, 85, 105, 0.5)" />
        <line x1={zeroX} y1={PADDING_Y} x2={zeroX} y2={HEIGHT - PADDING_Y} stroke="rgba(15, 23, 42, 0.28)" strokeDasharray="5 5" />
        {marker !== null ? (
          <line x1={marker} y1={PADDING_Y} x2={marker} y2={HEIGHT - PADDING_Y} stroke={markerColor ?? "rgba(239, 68, 68, 0.55)"} strokeDasharray="5 5" />
        ) : null}
        <polyline fill="none" stroke={colorA} strokeWidth={2.4} points={polyA} />
        {polyB ? <polyline fill="none" stroke={colorB} strokeWidth={2.2} points={polyB} /> : null}
      </svg>
      <div className="viewer-inline-legend">
        <span><i style={{ background: colorA }} />{labelA}</span>
        {labelB && colorB ? <span><i style={{ background: colorB }} />{labelB}</span> : null}
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
      <div className="viewer-stack-plots">
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
          markerColor="rgba(15, 23, 42, 0.45)"
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
      <div className="viewer-metrics">
        <span>Ui: {data.Ui ?? "n/a"}</span>
        <span>ni: {data.ni ?? "n/a"}</span>
        <span>Ci: {data.Ci ?? "n/a"}</span>
        <span>LDOS(0): {data.ldos_at_0 ?? "n/a"}</span>
        <span>LDOS(Ui): {data.ldos_at_Ui ?? "n/a"}</span>
      </div>
    </div>
  );
});
