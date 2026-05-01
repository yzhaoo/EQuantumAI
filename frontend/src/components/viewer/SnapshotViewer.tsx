import { useEffect, useMemo, useState } from "react";

import {
  fetchQuantumHeatmap,
  fetchSiteLdos,
  fetchSnapshots,
  type ViewerQuantumHeatmapResponse,
  type ViewerSiteLdosResponse,
} from "../../api";
import { LocalLdosPlot } from "./LocalLdosPlot";
import { LineCutPlot } from "./LineCutPlot";
import { QuantumHeatmap } from "./QuantumHeatmap";

type SnapshotViewerProps = {
  runId: string;
  runStatus: string | null;
};

const QUANTUM_PROPERTIES = ["Ui", "ni", "Ci", "Qprime_mask", "ΔUi", "LDOS@0", "LDOS@Ui"] as const;

function isLiveStatus(status: string | null) {
  return status !== null && !["completed", "failed", "aborted"].includes(status);
}

export function SnapshotViewer({ runId, runStatus }: SnapshotViewerProps) {
  const [snapshots, setSnapshots] = useState<string[]>([]);
  const [hasStatic, setHasStatic] = useState(false);
  const [snapshot, setSnapshot] = useState<string>("");
  const [property, setProperty] = useState<(typeof QUANTUM_PROPERTIES)[number]>("Ui");
  const [heatmap, setHeatmap] = useState<ViewerQuantumHeatmapResponse | null>(null);
  const [selectedSiteId, setSelectedSiteId] = useState<number | null>(null);
  const [siteLdos, setSiteLdos] = useState<ViewerSiteLdosResponse | null>(null);
  const [isLoadingSnapshots, setIsLoadingSnapshots] = useState(false);
  const [isLoadingHeatmap, setIsLoadingHeatmap] = useState(false);
  const [isLoadingSiteLdos, setIsLoadingSiteLdos] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadSnapshots() {
      setIsLoadingSnapshots(true);
      try {
        const response = await fetchSnapshots(runId);
        if (cancelled) {
          return;
        }
        setSnapshots(response.snapshots);
        setHasStatic(response.has_static);
        setSnapshot((current) => {
          if (current && response.snapshots.includes(current)) {
            return current;
          }
          return response.snapshots[response.snapshots.length - 1] ?? "";
        });
        setError(null);
      } catch (err) {
        if (cancelled) {
          return;
        }
        const message = err instanceof Error ? err.message : "Failed to load snapshots.";
        setError(message);
      } finally {
        if (!cancelled) {
          setIsLoadingSnapshots(false);
        }
      }
    }

    void loadSnapshots();
    if (!isLiveStatus(runStatus)) {
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
  }, [runId, runStatus]);

  useEffect(() => {
    if (!snapshot) {
      setHeatmap(null);
      return;
    }

    let cancelled = false;
    setIsLoadingHeatmap(true);

    fetchQuantumHeatmap(runId, snapshot, property)
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
        setError(err instanceof Error ? err.message : "Failed to load heatmap.");
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingHeatmap(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [runId, snapshot, property]);

  useEffect(() => {
    if (!snapshot || selectedSiteId === null) {
      setSiteLdos(null);
      return;
    }

    let cancelled = false;
    setIsLoadingSiteLdos(true);

    fetchSiteLdos(runId, snapshot, selectedSiteId)
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
        setError(err instanceof Error ? err.message : "Failed to load site LDOS.");
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingSiteLdos(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [runId, snapshot, selectedSiteId]);

  const statusText = useMemo(() => {
    if (isLoadingSnapshots) {
      return "Loading snapshot index…";
    }
    if (!hasStatic) {
      return "Waiting for run_static.npz…";
    }
    if (snapshots.length === 0) {
      return "Waiting for snapshot data…";
    }
    return null;
  }, [hasStatic, isLoadingSnapshots, snapshots.length]);

  return (
    <div className="result-card">
      <h4>Snapshot Viewer</h4>
      <p>Browse saved FSC snapshots directly in the browser. This first version focuses on the quantum heatmap and per-site LDOS.</p>

      <div className="viewer-controls">
        <label>
          Snapshot
          <select value={snapshot} onChange={(event) => setSnapshot(event.target.value)} disabled={snapshots.length === 0}>
            {snapshots.length === 0 ? <option value="">No snapshots yet</option> : null}
            {snapshots.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>

        <label>
          Property
          <select value={property} onChange={(event) => setProperty(event.target.value as (typeof QUANTUM_PROPERTIES)[number])}>
            {QUANTUM_PROPERTIES.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
      </div>

      {error ? <div className="viewer-error">{error}</div> : null}
      {statusText ? <div className="viewer-empty">{statusText}</div> : null}
      {!statusText ? (
        <div className="viewer-grid">
          {isLoadingHeatmap ? (
            <div className="viewer-empty">Loading heatmap…</div>
          ) : (
            <QuantumHeatmap data={heatmap} selectedSiteId={selectedSiteId} onSelectSite={setSelectedSiteId} />
          )}
          <LocalLdosPlot data={siteLdos} isLoading={isLoadingSiteLdos} />
          <LineCutPlot />
        </div>
      ) : null}
    </div>
  );
}
