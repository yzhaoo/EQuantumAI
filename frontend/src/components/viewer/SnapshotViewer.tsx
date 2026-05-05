import { memo, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";

import {
  fetchHistoryLdosCut,
  fetchHistoryQuantumHeatmap,
  fetchHistorySiteLdos,
  fetchHistorySnapshots,
  fetchHistorySurfaceCut,
  fetchLdosCut,
  fetchQuantumHeatmap,
  fetchSiteLdos,
  fetchSnapshots,
  fetchSurfaceCut,
  type ViewerLdosCutResponse,
  type ViewerQuantumHeatmapResponse,
  type ViewerSiteLdosResponse,
  type ViewerSurfaceCutResponse,
} from "../../api";
import { LdosCutPlot, SurfaceCutPlot } from "./LineCutPlot";
import { LocalLdosPlot } from "./LocalLdosPlot";
import { QuantumHeatmap } from "./QuantumHeatmap";
import { SnapshotTimeline } from "./SnapshotTimeline";

type SnapshotViewerProps = {
  source:
    | {
        kind: "live";
        runId: string;
        runStatus: string | null;
      }
    | {
        kind: "history";
        runPath: string;
      };
};

type PanelId = "site" | "surface" | "ldosCut" | "displayData" | "metrics";
type PanelLayout = Record<PanelId, { x: number; y: number; w: number; h: number; z: number }>;
type PointerAction =
  | {
      mode: "drag";
      id: PanelId;
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
      id: PanelId;
      stageWidth: number;
      stageHeight: number;
      startWidth: number;
      startHeight: number;
      startX: number;
      startY: number;
      clientX: number;
      clientY: number;
    };

const QUANTUM_PROPERTIES = ["Ui", "ni", "Ci", "Qprime_mask", "ΔUi", "LDOS@0", "LDOS@Ui"] as const;
const SURFACE_PROPERTY = "Ui";
const DEFAULT_LAYOUT: PanelLayout = {
  displayData: { x: 0.02, y: 0.14, w: 0.14, h: 0.22, z: 2 },
  metrics: { x: 0.02, y: 0.37, w: 0.14, h: 0.14, z: 2 },
  surface: { x: 0.02, y: 0.70, w: 0.32, h: 0.28, z: 2 },
  site: { x: 0.72, y: 0.02, w: 0.27, h: 0.46, z: 2 },
  ldosCut: { x: 0.72, y: 0.50, w: 0.27, h: 0.48, z: 2 },
};

const PROPERTY_LABELS: Record<string, string> = {
  "Ui": "Ui",
  "ni": "ni",
  "Ci": "Ci",
  "Qprime_mask": "Active Quantum Area",
  "ΔUi": "Delta Ui",
  "LDOS@0": "LDOS at 0",
  "LDOS@Ui": "LDOS at Ui"
};


function isLiveStatus(status: string | null) {
  return status !== null && !["completed", "failed", "aborted"].includes(status);
}

function isPendingArtifactError(message: string) {
  return message.includes("Run artifact directory is not available yet") || message.includes("Missing run_static.npz");
}

function boundsFromHeatmap(data: ViewerQuantumHeatmapResponse | null) {
  if (!data) {
    return null;
  }

  const xs = data.x.filter((value): value is number => value !== null);
  const ys = data.y.filter((value): value is number => value !== null);
  if (xs.length === 0 || ys.length === 0) {
    return null;
  }

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const midY = (minY + maxY) / 2;

  return {
    p0: [minX, midY] as [number, number],
    p1: [maxX, midY] as [number, number],
  };
}

function arraysEqual(left: string[], right: string[]) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

const FloatingPanel = memo(function FloatingPanel({
  title,
  layout,
  onDragStart,
  onResizeStart,
  onFocus,
  children,
}: {
  title: string;
  subtitle?: string;
  layout: { x: number; y: number; w: number; h: number; z: number };
  onDragStart: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onResizeStart: (event: ReactPointerEvent<HTMLButtonElement>) => void;
  onFocus: () => void;
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
      onPointerDown={() => {
        onFocus();
      }}
    >
      <button
        className="floating-panel-dragzone"
        type="button"
        aria-label={`Move ${title}`}
        onPointerDown={(event) => {
          event.stopPropagation();
          onFocus();
          onDragStart(event as unknown as ReactPointerEvent<HTMLDivElement>);
        }}
      />
      <div className="floating-panel-body">{children}</div>
      <button 
        className="floating-panel-resize" 
        type="button" 
        aria-label={`Resize ${title}`} 
        onPointerDown={(e) => {
          e.stopPropagation();
          onResizeStart(e);
        }} 
      />
    </div>
  );
});

export function SnapshotViewer({ source }: SnapshotViewerProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const pointerActionRef = useRef<PointerAction | null>(null);
  const layoutRef = useRef<PanelLayout>(DEFAULT_LAYOUT);
  const frameRef = useRef<number | null>(null);
  const pendingLayoutRef = useRef<PanelLayout | null>(null);
  const [layout, setLayout] = useState<PanelLayout>(DEFAULT_LAYOUT);
  const [snapshots, setSnapshots] = useState<string[]>([]);
  const [hasStatic, setHasStatic] = useState(false);
  const [snapshot, setSnapshot] = useState<string>("");
  const [property, setProperty] = useState<(typeof QUANTUM_PROPERTIES)[number]>("Ui");
  const [heatmap, setHeatmap] = useState<ViewerQuantumHeatmapResponse | null>(null);
  const [selectedSiteId, setSelectedSiteId] = useState<number | null>(null);
  const [siteLdos, setSiteLdos] = useState<ViewerSiteLdosResponse | null>(null);
  const [surfaceCut, setSurfaceCut] = useState<ViewerSurfaceCutResponse | null>(null);
  const [ldosCut, setLdosCut] = useState<ViewerLdosCutResponse | null>(null);
  const [isLoadingSnapshots, setIsLoadingSnapshots] = useState(false);
  const [isLoadingHeatmap, setIsLoadingHeatmap] = useState(false);
  const [isLoadingSiteLdos, setIsLoadingSiteLdos] = useState(false);
  const [isLoadingSurfaceCut, setIsLoadingSurfaceCut] = useState(false);
  const [isLoadingLdosCut, setIsLoadingLdosCut] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cutWidth, setCutWidth] = useState(0.05);
  const [hasLoadedSnapshotsOnce, setHasLoadedSnapshotsOnce] = useState(false);

  useEffect(() => {
    setLayout(DEFAULT_LAYOUT);
    layoutRef.current = DEFAULT_LAYOUT;
  }, [source.kind, source.kind === "live" ? source.runId : source.runPath]);

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

      const minWidth = 220;
      const minHeight = action.id === "site" ? 320 : 150;
      const nextWidthPx = clamp(action.startWidth + deltaX, minWidth, action.stageWidth - action.startX);
      const nextHeightPx = clamp(action.startHeight + deltaY, minHeight, action.stageHeight - action.startY);
      const nextWidth = nextWidthPx / action.stageWidth;
      const nextHeight = nextHeightPx / action.stageHeight;

      const current = layoutRef.current;
      commitLayout({
        ...current,
        [action.id]: {
          ...current[action.id],
          w: nextWidth,
          h: nextHeight,
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
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
      }
    };
  }, []);

  function focusPanel(id: PanelId) {
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

  function startDrag(id: PanelId, event: ReactPointerEvent<HTMLDivElement>) {
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

  function startResize(id: PanelId, event: ReactPointerEvent<HTMLButtonElement>) {
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

  useEffect(() => {
    let cancelled = false;

    async function loadSnapshots() {
      if (!hasLoadedSnapshotsOnce) {
        setIsLoadingSnapshots(true);
      }
      try {
        const response =
          source.kind === "live"
            ? await fetchSnapshots(source.runId)
            : await fetchHistorySnapshots(source.runPath);
        if (cancelled) {
          return;
        }



        setSnapshots((current) => (arraysEqual(current, response.snapshots) ? current : response.snapshots));
        setHasStatic(response.has_static);
        setSnapshot((current) => {
          if (current && response.snapshots.includes(current)) {
            return current;
          }
          return response.snapshots[response.snapshots.length - 1] ?? "";
        });
        setHasLoadedSnapshotsOnce(true);
        setError(null);
      } catch (err) {
        if (cancelled) {
          return;
        }
        const message = err instanceof Error ? err.message : "Failed to load snapshots.";
        if (isPendingArtifactError(message)) {
          setError(null);
          return;
        }
        setError(message);
      } finally {
        if (!cancelled) {
          setIsLoadingSnapshots(false);
        }
      }
    }

    void loadSnapshots();
    if (source.kind !== "live" || !isLiveStatus(source.runStatus)) {
      return () => {
        cancelled = true;
      };
    }

    const timer = window.setInterval(() => {
      void loadSnapshots();
    }, 3000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [hasLoadedSnapshotsOnce, source.kind, source.kind === "live" ? source.runId : source.runPath, source.kind === "live" ? source.runStatus : null]);

  useEffect(() => {
    if (!snapshot) {
      setHeatmap(null);
      return;
    }

    let cancelled = false;
    setIsLoadingHeatmap(true);

    const request =
      source.kind === "live"
        ? fetchQuantumHeatmap(source.runId, snapshot, property)
        : fetchHistoryQuantumHeatmap(source.runPath, snapshot, property);

    request
      .then((response) => {
        if (cancelled) {
          return;
        }
        setHeatmap(response);
        setSelectedSiteId((current) => (current && response.site_ids.includes(current) ? current : response.site_ids[0] ?? null));
        setError(null);
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setHeatmap(null);
        setSelectedSiteId(null);
        const message = err instanceof Error ? err.message : "Failed to load heatmap.";
        if (isPendingArtifactError(message)) {
          setError(null);
          return;
        }
        setError(message);
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingHeatmap(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [snapshot, property, source.kind, source.kind === "live" ? source.runId : source.runPath]);

  useEffect(() => {
    if (!snapshot || selectedSiteId === null) {
      setSiteLdos(null);
      return;
    }

    let cancelled = false;
    setIsLoadingSiteLdos(true);

    const request =
      source.kind === "live"
        ? fetchSiteLdos(source.runId, snapshot, selectedSiteId)
        : fetchHistorySiteLdos(source.runPath, snapshot, selectedSiteId);

    request
      .then((response) => {
        if (cancelled) {
          return;
        }
        setSiteLdos(response);
        setError(null);
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setSiteLdos(null);
        const message = err instanceof Error ? err.message : "Failed to load site LDOS.";
        if (isPendingArtifactError(message)) {
          setError(null);
          return;
        }
        setError(message);
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingSiteLdos(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [snapshot, selectedSiteId, source.kind, source.kind === "live" ? source.runId : source.runPath]);

  const cutLine = useMemo(() => boundsFromHeatmap(heatmap), [heatmap]);

  useEffect(() => {
    if (!snapshot || !cutLine) {
      setSurfaceCut(null);
      setLdosCut(null);
      return;
    }

    let cancelled = false;
    setIsLoadingSurfaceCut(true);
    setIsLoadingLdosCut(true);

    const overlaySnapshots = snapshots.length > 1 && snapshots[0] !== snapshot ? [snapshots[0], snapshot] : [snapshot];

    const surfaceRequest =
      source.kind === "live"
        ? fetchSurfaceCut(source.runId, {
            snapshot,
            property: SURFACE_PROPERTY,
            p0: cutLine.p0,
            p1: cutLine.p1,
            cut_width: cutWidth,
          })
        : fetchHistorySurfaceCut(source.runPath, {
            snapshot,
            property: SURFACE_PROPERTY,
            p0: cutLine.p0,
            p1: cutLine.p1,
            cut_width: cutWidth,
          });

    surfaceRequest
      .then((response) => {
        if (cancelled) {
          return;
        }
        setSurfaceCut(response);
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setSurfaceCut(null);
        const message = err instanceof Error ? err.message : "Failed to load surface cut.";
        if (isPendingArtifactError(message)) {
          setError(null);
          return;
        }
        setError(message);
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingSurfaceCut(false);
        }
      });

    const ldosRequest =
      source.kind === "live"
        ? fetchLdosCut(source.runId, {
            snapshot,
            p0: cutLine.p0,
            p1: cutLine.p1,
            cut_width: cutWidth,
            overlay_snapshots: overlaySnapshots,
          })
        : fetchHistoryLdosCut(source.runPath, {
            snapshot,
            p0: cutLine.p0,
            p1: cutLine.p1,
            cut_width: cutWidth,
            overlay_snapshots: overlaySnapshots,
          });

    ldosRequest
      .then((response) => {
        if (cancelled) {
          return;
        }
        setLdosCut(response);
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setLdosCut(null);
        const message = err instanceof Error ? err.message : "Failed to load LDOS cut.";
        if (isPendingArtifactError(message)) {
          setError(null);
          return;
        }
        setError(message);
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingLdosCut(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [cutLine, cutWidth, snapshot, snapshots, source.kind, source.kind === "live" ? source.runId : source.runPath]);

  const statusText = useMemo(() => {
    if (isLoadingSnapshots && !hasLoadedSnapshotsOnce) {
      return "Loading snapshot index…";
    }
    if (!hasStatic) {
      return "Waiting for run_static.npz…";
    }
    if (snapshots.length === 0) {
      return "Waiting for snapshot data…";
    }
    return null;
  }, [hasLoadedSnapshotsOnce, hasStatic, isLoadingSnapshots, snapshots.length]);

  return (
    <section className="stage-card simulation-stage compact-stage">



      {error ? <div className="viewer-error">{error}</div> : null}
      {statusText ? <div className="viewer-empty">{statusText}</div> : null}

      {!statusText ? (
        <div ref={stageRef} className="simulation-canvas">
          <div className="simulation-background">
            {isLoadingHeatmap ? (
              <div className="viewer-empty">Loading heatmap…</div>
            ) : (
              <QuantumHeatmap
                data={heatmap}
                selectedSiteId={selectedSiteId}
                onSelectSite={setSelectedSiteId}
                cutLine={cutLine}
              />
            )}
            <div className="timeline-overlay">
              <SnapshotTimeline
                snapshots={snapshots}
                activeSnapshot={snapshot}
                onSelect={setSnapshot}
              />
            </div>
          </div>

          <FloatingPanel
            title="Onsite Inspector"
            subtitle={selectedSiteId !== null ? `site ${selectedSiteId}` : "select a site"}
            layout={layout.site}
            onFocus={() => focusPanel("site")}
            onDragStart={(event) => startDrag("site", event)}
            onResizeStart={(event) => startResize("site", event)}
          >
            <LocalLdosPlot data={siteLdos} isLoading={isLoadingSiteLdos} />
          </FloatingPanel>

          <FloatingPanel
            title="Surface Cut"
            subtitle={`width ${cutWidth.toFixed(2)}`}
            layout={layout.surface}
            onFocus={() => focusPanel("surface")}
            onDragStart={(event) => startDrag("surface", event)}
            onResizeStart={(event) => startResize("surface", event)}
          >
            <SurfaceCutPlot data={surfaceCut} isLoading={isLoadingSurfaceCut} />
          </FloatingPanel>

          <FloatingPanel
            title="Quantum Cut"
            subtitle={snapshot.replace(".npz", "")}
            layout={layout.ldosCut}
            onFocus={() => focusPanel("ldosCut")}
            onDragStart={(event) => startDrag("ldosCut", event)}
            onResizeStart={(event) => startResize("ldosCut", event)}
          >
            <LdosCutPlot data={ldosCut} isLoading={isLoadingLdosCut} />
          </FloatingPanel>


          <FloatingPanel
            title="Display Data"
            layout={layout.displayData}
            onFocus={() => focusPanel("displayData")}
            onDragStart={(event) => startDrag("displayData", event)}
            onResizeStart={(event) => startResize("displayData", event)}
          >
            <div className="display-data-panel">
              <h3 className="panel-title">Display Data</h3>
              <div className="property-pills">
                {QUANTUM_PROPERTIES.map((item) => (
                  <button
                    key={item}
                    className={`property-pill ${property === item ? "active" : ""}`}
                    onClick={() => setProperty(item)}
                  >
                    {PROPERTY_LABELS[item] || item}
                  </button>
                ))}
              </div>
            </div>
          </FloatingPanel>

          <FloatingPanel
            title="Metrics"
            layout={layout.metrics}
            onFocus={() => focusPanel("metrics")}
            onDragStart={(event) => startDrag("metrics", event)}
            onResizeStart={(event) => startResize("metrics", event)}
          >
            <div className="metrics-panel">
              <div className="metric-row">
                <span>Selected Site id:</span>
                <span className="metric-value">{selectedSiteId ?? "-"}</span>
              </div>
              <div className="metric-row">
                <span>Linecut Width:</span>
                <div className="number-stepper">
                  <button className="stepper-btn" onClick={() => setCutWidth(Math.max(0.01, cutWidth - 0.01))}>-</button>
                  <span className="stepper-val">{cutWidth.toFixed(2)}</span>
                  <button className="stepper-btn" onClick={() => setCutWidth(Math.min(0.20, cutWidth + 0.01))}>+</button>
                </div>
              </div>
              <div className="metric-row">
                <span>Linecut y:</span>
                <span className="metric-value">{cutLine ? cutLine.p0[1].toFixed(2) : "-"}</span>
              </div>
            </div>
          </FloatingPanel>
        </div>
      ) : null}
    </section>
  );
}
