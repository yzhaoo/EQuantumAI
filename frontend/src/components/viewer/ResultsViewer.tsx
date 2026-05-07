import { memo, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import Plot from "react-plotly.js";
import { SnapshotViewer } from "./SnapshotViewer";

type ResultVisualization =
  | { type: "ldos_volume_3d"; artifact_dir?: string; snapshot?: string; energy?: number[]; site_ids?: number[]; ldos_matrix?: Array<Array<number | null>> }
  | { type: "ldos_linecut_with_ui"; artifact_dir?: string; snapshot?: string; distance_along?: Array<number | null>; energy?: Array<number | null>; ldos_matrix?: Array<Array<number | null>>; overlays?: Array<{ snapshot: string; Ui: Array<number | null>; is_current: boolean }> }
  | { type: "ldos_linecut_multi"; items?: Array<Record<string, unknown>> }
  | { type: "ui_linecut_compare"; traces?: Array<{ label?: string; distance_along?: Array<number | null>; values?: Array<number | null> }> }
  | { type: "ldos_ui_heatmap"; artifact_dir?: string; snapshot?: string; x?: Array<number | null>; y?: Array<number | null>; values?: Array<number | null>; color_min?: number; color_max?: number }
  | { type: "ldos_ui_heatmap_multi"; items?: Array<Record<string, unknown>> }
  | { type: "ldos_comparison"; label_a?: string; label_b?: string; energy?: number[]; ldos_a?: number[]; ldos_b?: number[]; delta?: number[] }
  | Record<string, unknown>;

type PanelLayout = Record<string, { x: number; y: number; w: number; h: number; z: number }>;
type ResultPlotOption = {
  id: "ldos_volume_3d" | "ldos_linecut_with_ui" | "ldos_ui_heatmap" | "ldos_comparison" | "ui_linecut_compare";
  label: string;
  description: string;
};
type PointerAction =
  | {
      mode: "drag";
      id: string;
      stageWidth: number;
      stageHeight: number;
      panelWidth: number;
      panelHeight: number;
      startX: number;
      startY: number;
      clientX: number;
      clientY: number;
    }
  | {
      mode: "resize";
      id: string;
      stageWidth: number;
      stageHeight: number;
      startWidth: number;
      startHeight: number;
      startX: number;
      startY: number;
      clientX: number;
      clientY: number;
    };

const TURBO_COLORSCALE: [number, string][] = [
  [0.0, "#30123b"],
  [0.1, "#4662d7"],
  [0.2, "#36aaf9"],
  [0.3, "#1fd0a5"],
  [0.4, "#48f882"],
  [0.5, "#95fb51"],
  [0.6, "#dedd32"],
  [0.7, "#ffa423"],
  [0.8, "#f65f18"],
  [0.9, "#ba2208"],
  [1.0, "#7a0403"],
];

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function coerceNumbers(values: Array<number | null> | undefined) {
  return (values ?? []).map((value) => (typeof value === "number" ? value : NaN));
}

function defaultLayoutForVisualization(type: string, index: number) {
  const presets: Record<string, { x: number; y: number; w: number; h: number }> = {
    ldos_comparison: { x: 0.04, y: 0.05, w: 0.42, h: 0.38 },
    ui_linecut_compare: { x: 0.52, y: 0.08, w: 0.38, h: 0.32 },
    ldos_ui_heatmap: { x: 0.04, y: 0.50, w: 0.34, h: 0.34 },
    ldos_ui_heatmap_multi: { x: 0.04, y: 0.50, w: 0.62, h: 0.40 },
    ldos_volume_3d: { x: 0.42, y: 0.42, w: 0.52, h: 0.46 },
    ldos_linecut_with_ui: { x: 0.56, y: 0.52, w: 0.38, h: 0.34 },
    ldos_linecut_multi: { x: 0.34, y: 0.46, w: 0.62, h: 0.40 },
  };
  const preset = presets[type] ?? { x: 0.06 + (index % 2) * 0.42, y: 0.08 + Math.floor(index / 2) * 0.3, w: 0.38, h: 0.32 };
  return { ...preset, z: index + 2 };
}

function buildLayout(visualizations: ResultVisualization[]): PanelLayout {
  const next: PanelLayout = {
    info: { x: 0.03, y: 0.04, w: 0.28, h: 0.34, z: 8 },
  };
  visualizations.forEach((visualization, index) => {
    const id = panelIdForVisualization(visualization, index);
    next[id] = defaultLayoutForVisualization(String((visualization as { type?: unknown }).type ?? "result"), index);
  });
  return next;
}

function panelIdForVisualization(visualization: ResultVisualization, index: number) {
  const typed = visualization as { type?: unknown; artifact_dir?: unknown; snapshot?: unknown; label_a?: unknown; label_b?: unknown };
  return [
    "viz",
    index,
    String(typed.type ?? "result"),
    String(typed.artifact_dir ?? ""),
    String(typed.snapshot ?? ""),
    String(typed.label_a ?? ""),
    String(typed.label_b ?? ""),
  ].join("-");
}

const FloatingPanel = memo(function FloatingPanel({
  id,
  title,
  subtitle,
  layout,
  onDragStart,
  onResizeStart,
  onFocus,
  children,
}: {
  id: string;
  title: string;
  subtitle?: string;
  layout: { x: number; y: number; w: number; h: number; z: number };
  onDragStart: (id: string, event: ReactPointerEvent<HTMLDivElement>) => void;
  onResizeStart: (id: string, event: ReactPointerEvent<HTMLButtonElement>) => void;
  onFocus: (id: string) => void;
  children: ReactNode;
}) {
  return (
    <div
      className="floating-panel"
      style={{
        left: `${layout.x * 100}%`,
        top: `${layout.y * 100}%`,
        width: `${layout.w * 100}%`,
        height: `${layout.h * 100}%`,
        zIndex: layout.z,
      }}
      onPointerDown={() => onFocus(id)}
    >
      <button
        className="floating-panel-dragzone"
        type="button"
        aria-label={`Move ${title}`}
        onPointerDown={(event) => {
          event.stopPropagation();
          onFocus(id);
          onDragStart(id, event as unknown as ReactPointerEvent<HTMLDivElement>);
        }}
      />
      <div className="floating-panel-body results-floating-body">{children}</div>
      <button className="floating-panel-resize" type="button" aria-label={`Resize ${title}`} onPointerDown={(event) => onResizeStart(id, event)} />
    </div>
  );
});

function plotConfig() {
  return {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
  };
}

function plotLayout(title: string, extras?: Record<string, unknown>) {
  return {
    title: { text: title, font: { size: 14 } },
    paper_bgcolor: "rgba(255,255,255,0)",
    plot_bgcolor: "rgba(255,255,255,0)",
    margin: { l: 56, r: 24, t: 42, b: 46 },
    font: { family: "Avenir Next, Helvetica Neue, sans-serif", size: 11, color: "#1f2937" },
    autosize: true,
    ...extras,
  };
}

const ResultsPlotPanel = memo(function ResultsPlotPanel({ visualization }: { visualization: ResultVisualization }) {
  const type = String((visualization as { type?: unknown }).type ?? "result");
  if (type === "ldos_comparison") {
    const energy = (visualization as { energy?: number[] }).energy ?? [];
    return (
      <div className="results-panel-shell">
        <div className="floating-panel-title results-panel-title">
          <strong>LDOS comparison</strong>
        </div>
        <Plot
          className="results-plot"
          useResizeHandler
          data={[
          { type: "scattergl", mode: "lines", name: String((visualization as { label_a?: unknown }).label_a ?? "A"), x: energy, y: (visualization as { ldos_a?: number[] }).ldos_a ?? [], line: { color: "#2f6d4d", width: 2 } },
          { type: "scattergl", mode: "lines", name: String((visualization as { label_b?: unknown }).label_b ?? "B"), x: energy, y: (visualization as { ldos_b?: number[] }).ldos_b ?? [], line: { color: "#b42318", width: 2 } },
          { type: "scattergl", mode: "lines", name: "delta", x: energy, y: (visualization as { delta?: number[] }).delta ?? [], line: { color: "#1d4ed8", width: 1.5, dash: "dot" } },
          ]}
          layout={plotLayout("LDOS comparison", { xaxis: { title: "Energy" }, yaxis: { title: "Density / delta" } })}
          config={plotConfig()}
        />
      </div>
    );
  }
  if (type === "ui_linecut_compare") {
    const traces = ((visualization as { traces?: Array<{ label?: string; distance_along?: Array<number | null>; values?: Array<number | null> }> }).traces) ?? [];
    const colors = ["#2f6d4d", "#b42318", "#1d4ed8", "#9333ea"];
    return (
      <div className="results-panel-shell">
        <div className="floating-panel-title results-panel-title">
          <strong>Ui linecut comparison</strong>
        </div>
        <Plot
          className="results-plot"
          useResizeHandler
          data={traces.map((trace, index) => ({
          type: "scattergl",
          mode: "lines",
          name: trace.label ?? `run ${index + 1}`,
          x: coerceNumbers(trace.distance_along),
          y: coerceNumbers(trace.values),
          line: { color: colors[index % colors.length], width: 2 },
          }))}
          layout={plotLayout("Ui linecut comparison", { xaxis: { title: "Distance along cut" }, yaxis: { title: "Ui" } })}
          config={plotConfig()}
        />
      </div>
    );
  }
  if (type === "ldos_ui_heatmap") {
    return (
      <div className="results-panel-shell">
        <div className="floating-panel-title results-panel-title">
          <strong>LDOS @ Ui heatmap</strong>
        </div>
        <Plot
          className="results-plot"
          useResizeHandler
          data={[
          {
            type: "scattergl",
            mode: "markers",
            x: coerceNumbers((visualization as { x?: Array<number | null> }).x),
            y: coerceNumbers((visualization as { y?: Array<number | null> }).y),
            marker: {
              size: 8,
              color: coerceNumbers((visualization as { values?: Array<number | null> }).values),
              colorscale: TURBO_COLORSCALE,
              colorbar: { title: "LDOS@Ui" },
              cmin: Number((visualization as { color_min?: unknown }).color_min ?? 0),
              cmax: Number((visualization as { color_max?: unknown }).color_max ?? 1),
            },
          },
          ]}
          layout={plotLayout("LDOS @ Ui heatmap", { xaxis: { title: "x" }, yaxis: { title: "y", scaleanchor: "x", scaleratio: 1 } })}
          config={plotConfig()}
        />
      </div>
    );
  }
  if (type === "ldos_ui_heatmap_multi") {
    const items = ((visualization as { items?: Array<Record<string, unknown>> }).items) ?? [];
    return (
      <div className="results-panel-shell">
        <div className="floating-panel-title results-panel-title">
          <strong>LDOS @ Ui heatmap</strong>
          <span>{items.length} runs</span>
        </div>
        <div className="results-multi-grid">
          {items.map((item, index) => {
            const snapshotLabel = String((item as { snapshot?: unknown }).snapshot ?? `run ${index + 1}`);
            return (
              <div key={`ldos-ui-heatmap-item-${index}`} className="results-multi-cell">
                <div className="results-multi-title">{snapshotLabel}</div>
                <Plot
                  className="results-plot"
                  useResizeHandler
                  data={[
                    {
                      type: "scattergl",
                      mode: "markers",
                      x: coerceNumbers((item as { x?: Array<number | null> }).x),
                      y: coerceNumbers((item as { y?: Array<number | null> }).y),
                      marker: {
                        size: 8,
                        color: coerceNumbers((item as { values?: Array<number | null> }).values),
                        colorscale: TURBO_COLORSCALE,
                        colorbar: { title: "LDOS@Ui" },
                        cmin: Number((item as { color_min?: unknown }).color_min ?? 0),
                        cmax: Number((item as { color_max?: unknown }).color_max ?? 1),
                      },
                    },
                  ]}
                  layout={plotLayout("LDOS @ Ui heatmap", { xaxis: { title: "x" }, yaxis: { title: "y", scaleanchor: "x", scaleratio: 1 } })}
                  config={plotConfig()}
                />
              </div>
            );
          })}
        </div>
      </div>
    );
  }
  if (type === "ldos_volume_3d") {
    return (
      <div className="results-panel-shell">
        <div className="floating-panel-title results-panel-title">
          <strong>3D LDOS volume</strong>
        </div>
        <Plot
          className="results-plot"
          useResizeHandler
          data={[
          {
            type: "surface",
            x: (visualization as { energy?: number[] }).energy ?? [],
            y: (visualization as { site_ids?: number[] }).site_ids ?? [],
            z: (((visualization as { ldos_matrix?: Array<Array<number | null>> }).ldos_matrix) ?? []).map((row) => row.map((value) => (typeof value === "number" ? value : NaN))),
            colorscale: TURBO_COLORSCALE,
            showscale: true,
          },
          ]}
          layout={plotLayout("3D LDOS volume", {
          scene: {
            xaxis: { title: "Energy" },
            yaxis: { title: "Site id" },
            zaxis: { title: "Density" },
          },
          margin: { l: 0, r: 0, t: 42, b: 0 },
          })}
          config={plotConfig()}
        />
      </div>
    );
  }
  if (type === "ldos_linecut_with_ui") {
    const overlays = ((visualization as { overlays?: Array<{ snapshot: string; Ui: Array<number | null>; is_current: boolean }> }).overlays) ?? [];
    return (
      <div className="results-panel-shell">
        <div className="floating-panel-title results-panel-title">
          <strong>DOS heatmap along linecut</strong>
          {String((visualization as { snapshot?: unknown }).snapshot ?? "") ? <span>{String((visualization as { snapshot?: unknown }).snapshot ?? "")}</span> : null}
        </div>
        <Plot
          className="results-plot"
          useResizeHandler
          data={[
          {
            type: "heatmap",
            x: coerceNumbers((visualization as { distance_along?: Array<number | null> }).distance_along),
            y: coerceNumbers((visualization as { energy?: Array<number | null> }).energy),
            z: (((visualization as { ldos_matrix?: Array<Array<number | null>> }).ldos_matrix) ?? []).map((row) => row.map((value) => (typeof value === "number" ? value : NaN))),
            colorscale: TURBO_COLORSCALE,
            colorbar: { title: "LDOS" },
          },
            ...overlays.map((overlay) => ({
            type: "scattergl",
            mode: "lines",
            name: overlay.snapshot.replace(".npz", ""),
            x: coerceNumbers((visualization as { distance_along?: Array<number | null> }).distance_along),
            y: coerceNumbers(overlay.Ui),
            line: { color: overlay.is_current ? "#ffffff" : "#111827", width: overlay.is_current ? 2.2 : 1.6, dash: overlay.is_current ? "solid" : "dot" },
            })),
          ]}
          layout={plotLayout("DOS heatmap along linecut", { xaxis: { title: "Distance along cut" }, yaxis: { title: "Energy / Ui" } })}
          config={plotConfig()}
        />
      </div>
    );
  }
  if (type === "ldos_linecut_multi") {
    const items = ((visualization as { items?: Array<Record<string, unknown>> }).items) ?? [];
    return (
      <div className="results-panel-shell">
        <div className="floating-panel-title results-panel-title">
          <strong>DOS heatmap along linecut</strong>
          <span>{items.length} runs</span>
        </div>
        <div className="results-multi-grid">
          {items.map((item, index) => {
            const overlays = ((item as { overlays?: Array<{ snapshot: string; Ui: Array<number | null>; is_current: boolean }> }).overlays) ?? [];
            const snapshotLabel = String((item as { snapshot?: unknown }).snapshot ?? `run ${index + 1}`);
            return (
              <div key={`ldos-linecut-item-${index}`} className="results-multi-cell">
                <div className="results-multi-title">{snapshotLabel}</div>
                <Plot
                  className="results-plot"
                  useResizeHandler
                  data={[
                    {
                      type: "heatmap",
                      x: coerceNumbers((item as { distance_along?: Array<number | null> }).distance_along),
                      y: coerceNumbers((item as { energy?: Array<number | null> }).energy),
                      z: ((((item as { ldos_matrix?: Array<Array<number | null>> }).ldos_matrix) ?? []).map((row) => row.map((value) => (typeof value === "number" ? value : NaN)))),
                      colorscale: TURBO_COLORSCALE,
                      colorbar: { title: "LDOS" },
                    },
                    ...overlays.map((overlay) => ({
                      type: "scattergl",
                      mode: "lines",
                      name: overlay.snapshot.replace(".npz", ""),
                      x: coerceNumbers((item as { distance_along?: Array<number | null> }).distance_along),
                      y: coerceNumbers(overlay.Ui),
                      line: { color: overlay.is_current ? "#ffffff" : "#111827", width: overlay.is_current ? 2.2 : 1.6, dash: overlay.is_current ? "solid" : "dot" },
                    })),
                  ]}
                  layout={plotLayout("DOS heatmap along linecut", { xaxis: { title: "Distance along cut" }, yaxis: { title: "Energy / Ui" } })}
                  config={plotConfig()}
                />
              </div>
            );
          })}
        </div>
      </div>
    );
  }
  return <div className="viewer-empty">No renderer is defined yet for {type}.</div>;
});

export function ResultsViewer({
  visualizations,
  infoLines,
  plotOptions,
  onOpenPlot,
  onClearPlots,
  source,
}: {
  visualizations: ResultVisualization[];
  infoLines: string[];
  plotOptions: ResultPlotOption[];
  onOpenPlot: (plotId: ResultPlotOption["id"]) => void;
  onClearPlots: () => void;
  source:
    | { kind: "live"; runId: string | null; runStatus: string | null }
    | { kind: "history"; runPath: string };
}) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const pointerActionRef = useRef<PointerAction | null>(null);
  const frameRef = useRef<number | null>(null);
  const pendingLayoutRef = useRef<PanelLayout | null>(null);
  const [layout, setLayout] = useState<PanelLayout>(() => buildLayout(visualizations));
  const layoutRef = useRef<PanelLayout>(layout);

  const layoutSeed = useMemo(
    () => visualizations.map((visualization, index) => panelIdForVisualization(visualization, index)).join("|"),
    [visualizations],
  );

  useEffect(() => {
    const next = buildLayout(visualizations);
    setLayout(next);
    layoutRef.current = next;
  }, [layoutSeed]);

  useEffect(() => {
    layoutRef.current = layout;
  }, [layout]);

  function commitLayout(nextLayout: PanelLayout) {
    pendingLayoutRef.current = nextLayout;
    if (frameRef.current !== null) {
      return;
    }
    frameRef.current = window.requestAnimationFrame(() => {
      frameRef.current = null;
      if (pendingLayoutRef.current) {
        layoutRef.current = pendingLayoutRef.current;
        setLayout(pendingLayoutRef.current);
        pendingLayoutRef.current = null;
      }
    });
  }

  useEffect(() => {
    function handlePointerMove(event: PointerEvent) {
      const action = pointerActionRef.current;
      if (!action) {
        return;
      }
      const deltaX = event.clientX - action.clientX;
      const deltaY = event.clientY - action.clientY;
      if (action.mode === "drag") {
        const nextX = clamp((action.startX + deltaX) / action.stageWidth, 0, 1 - action.panelWidth / action.stageWidth);
        const nextY = clamp((action.startY + deltaY) / action.stageHeight, 0, 1 - action.panelHeight / action.stageHeight);
        const current = layoutRef.current;
        commitLayout({
          ...current,
          [action.id]: {
            ...current[action.id],
            x: nextX,
            y: nextY,
          },
        });
        return;
      }
      const minWidth = 280;
      const minHeight = 220;
      const nextWidthPx = clamp(action.startWidth + deltaX, minWidth, action.stageWidth - action.startX);
      const nextHeightPx = clamp(action.startHeight + deltaY, minHeight, action.stageHeight - action.startY);
      const current = layoutRef.current;
      commitLayout({
        ...current,
        [action.id]: {
          ...current[action.id],
          w: nextWidthPx / action.stageWidth,
          h: nextHeightPx / action.stageHeight,
        },
      });
    }
    function handlePointerUp() {
      pointerActionRef.current = null;
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
      if (pendingLayoutRef.current) {
        layoutRef.current = pendingLayoutRef.current;
        setLayout(pendingLayoutRef.current);
        pendingLayoutRef.current = null;
      }
    }
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, []);

  function focusPanel(id: string) {
    const current = layoutRef.current;
    const topZ = Math.max(...Object.values(current).map((panel) => panel.z));
    const next = {
      ...current,
      [id]: {
        ...current[id],
        z: topZ + 1,
      },
    };
    layoutRef.current = next;
    setLayout(next);
  }

  function startDrag(id: string, event: ReactPointerEvent<HTMLDivElement>) {
    const stage = stageRef.current;
    if (!stage) {
      return;
    }
    event.preventDefault();
    focusPanel(id);
    const rect = stage.getBoundingClientRect();
    const panel = layoutRef.current[id];
    pointerActionRef.current = {
      mode: "drag",
      id,
      stageWidth: rect.width,
      stageHeight: rect.height,
      panelWidth: panel.w * rect.width,
      panelHeight: panel.h * rect.height,
      startX: panel.x * rect.width,
      startY: panel.y * rect.height,
      clientX: event.clientX,
      clientY: event.clientY,
    };
  }

  function startResize(id: string, event: ReactPointerEvent<HTMLButtonElement>) {
    const stage = stageRef.current;
    if (!stage) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    focusPanel(id);
    const rect = stage.getBoundingClientRect();
    const panel = layoutRef.current[id];
    pointerActionRef.current = {
      mode: "resize",
      id,
      stageWidth: rect.width,
      stageHeight: rect.height,
      startWidth: panel.w * rect.width,
      startHeight: panel.h * rect.height,
      startX: panel.x * rect.width,
      startY: panel.y * rect.height,
      clientX: event.clientX,
      clientY: event.clientY,
    };
  }

  return (
    <div ref={stageRef} className="simulation-canvas">
      <div className="simulation-background results-background">
        <div className="result-stage-viewer">
          {source.kind === "history" || source.runId ? (
            <SnapshotViewer source={source.kind === "history" ? source : { kind: "live", runId: source.runId as string, runStatus: source.runStatus }} />
          ) : (
            <div className="viewer-empty result-stage-note">
              Results will appear here once a completed run or selected history run is available.
            </div>
          )}
        </div>
      </div>
      <FloatingPanel
        id="info"
        title="Result Controls"
        layout={layout.info ?? { x: 0.03, y: 0.04, w: 0.28, h: 0.34, z: 8 }}
        onDragStart={startDrag}
        onResizeStart={startResize}
        onFocus={focusPanel}
      >
        <div className="results-panel-shell">
          <div className="floating-panel-title results-panel-title">
            <strong>Selected Runs</strong>
          </div>
          <div className="results-info-list">
            {infoLines.map((line, index) => (
              <span key={`info-${index}`}>{line}</span>
            ))}
          </div>
          <div className="floating-panel-title results-panel-title">
            <strong>Available Plots</strong>
          </div>
          <div className="results-pill-grid">
            {plotOptions.length === 0 ? (
              <div className="viewer-empty">Select a completed run, or two history runs, to enable plot actions.</div>
            ) : (
              plotOptions.map((option) => (
                <button key={option.id} className="property-pill" type="button" onClick={() => onOpenPlot(option.id)}>
                  {option.label}
                </button>
              ))
            )}
          </div>
          {visualizations.length > 0 ? (
            <button className="secondary-button results-clear-button" type="button" onClick={onClearPlots}>
              Clear Plot Windows
            </button>
          ) : null}
        </div>
      </FloatingPanel>
      {visualizations.map((visualization, index) => {
        const id = panelIdForVisualization(visualization, index);
        const panel = layout[id];
        if (!panel) {
          return null;
        }
        return (
          <FloatingPanel
            key={id}
            id={id}
            title={String((visualization as { type?: unknown }).type ?? "result").replaceAll("_", " ")}
            subtitle={String((visualization as { snapshot?: unknown }).snapshot ?? "")}
            layout={panel}
            onDragStart={startDrag}
            onResizeStart={startResize}
            onFocus={focusPanel}
          >
            <ResultsPlotPanel visualization={visualization} />
          </FloatingPanel>
        );
      })}
    </div>
  );
}
