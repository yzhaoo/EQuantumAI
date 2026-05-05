import type { RunStateResponse } from "../api";
import { SnapshotViewer } from "./viewer/SnapshotViewer";

type ResultPanelProps = {
  runState: RunStateResponse | null;
  runId: string | null;
  onApproveManualCheck: (approved: boolean) => void;
};

function renderArtifacts(runState: RunStateResponse | null) {
  const artifacts = (runState?.result?.artifacts as Record<string, string> | undefined) ?? {};
  const entries = Object.entries(artifacts);
  if (entries.length === 0) {
    return <p>No artifacts yet.</p>;
  }

  return (
    <ul>
      {entries.map(([key, value]) => (
        <li key={key}>
          <strong>{key}</strong>:{" "}
          <a className="artifact-link" href={value} target="_blank" rel="noreferrer">
            {value}
          </a>
        </li>
      ))}
    </ul>
  );
}

export function ResultPanel({
  runState,
  runId,
  onApproveManualCheck,
}: ResultPanelProps) {
  const result = runState?.result;

  return (
    <section className="panel-card">
      <header className="shell-header">
        <h3 className="shell-title">Run Console</h3>
        <p className="shell-subtitle">Inspect run progress, artifacts, plots, and manual approval checkpoints.</p>
      </header>
      <div className="panel-body">
        <div className="result-section">
          <div className="result-card">
            <h4>Manual Check</h4>
            {runState?.manual_check_pending ? (
              <>
                <p>{String(runState.manual_check_payload?.message ?? "Boundary inspection is waiting for approval.")}</p>
                {typeof runState.manual_check_payload?.plot_path === "string" ? (
                  <p>
                    Plot:{" "}
                    <a className="artifact-link" href={String(runState.manual_check_payload.plot_path)} target="_blank" rel="noreferrer">
                      {String(runState.manual_check_payload.plot_path)}
                    </a>
                  </p>
                ) : null}
                <div className="action-row">
                  <button className="primary-button" onClick={() => onApproveManualCheck(true)}>
                    Approve
                  </button>
                  <button className="danger-button" onClick={() => onApproveManualCheck(false)}>
                    Reject / Abort
                  </button>
                </div>
              </>
            ) : (
              <p>No manual check is pending.</p>
            )}
          </div>

          <div className="result-card">
            <h4>Artifacts</h4>
            {renderArtifacts(runState)}
          </div>

          {runId && ["building_system", "initializing_fsc", "manual_check_required", "solving", "exporting_artifacts", "completed"].includes(runState?.status ?? "") ? (
            <SnapshotViewer source={{ kind: "live", runId, runStatus: runState?.status ?? null }} />
          ) : null}

          <div className="result-card">
            <h4>Summary</h4>
            {result ? (
              <ul>
                <li>Artifact dir: {String(result.artifact_dir ?? "-")}</li>
                <li>Profile: {String(result.profile ?? "-")}</li>
                <li>Task: {String(result.task ?? "-")}</li>
                <li>Qsites: {String(result.qsites ?? "-")}</li>
              </ul>
            ) : (
              <p>No completed run summary yet.</p>
            )}
          </div>

          <div className="result-card">
            <h4>Logs</h4>
            <div className="log-list">
              {(runState?.logs ?? []).length > 0 ? (
                runState?.logs.map((line, index) => (
                  <div className="log-line" key={`${index}-${line.slice(0, 16)}`}>
                    {line}
                  </div>
                ))
              ) : (
                <p>No logs yet.</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
