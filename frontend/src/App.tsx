import { useEffect, useMemo, useRef, useState } from "react";
import {
  abortRun,
  approveManualCheck,
  createRun,
  fetchHistoryRuns,
  fetchRun,
  sendAgentTurn,
  type AgentTurnResponse,
  type HistoryRunItem,
  type RunStateResponse,
  type SimulationSpec,
} from "./api";
import { ChatPanel, type ChatMessage } from "./components/ChatPanel";
import { SetupGeometryViewer } from "./components/viewer/SetupGeometryViewer";
import { SnapshotViewer } from "./components/viewer/SnapshotViewer";

type WorkspaceView = "setup" | "simulation" | "history";
type InspectorTab = "simulation-info" | "prompt-log" | "terminal";
type ParserMode = "openai" | "langchain" | "regex";

function buildMessage(role: ChatMessage["role"], text: string, response?: AgentTurnResponse): ChatMessage {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role,
    text,
    response,
  };
}

function formatValue(value: unknown) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  return String(value);
}

function SetupStage({
  spec,
  source,
  currentStatus,
}: {
  spec: SimulationSpec | null;
  source:
    | { kind: "live"; runId: string | null; runStatus: string | null }
    | { kind: "history"; runPath: string };
  currentStatus: string | null;
}) {
  const metrics = [
    { label: "Profile", value: spec?.profile ?? "dotgate_center" },
    { label: "Task", value: spec?.task ?? "dos" },
    { label: "Lattice", value: spec?.lattice_type ?? "square" },
    { label: "Magnetic field", value: spec?.magnetic_field_T !== null ? `${spec?.magnetic_field_T} T` : "-" },
    { label: "Backgate", value: spec?.backgate_voltage !== null ? `${spec?.backgate_voltage} V` : "-" },
    { label: "Run state", value: currentStatus ?? "idle" },
  ];

  return (
    <section className="stage-card">
      <div className="stage-header">
        <h2>System Sites</h2>
      </div>

      <div className="setup-scene">
        <SetupGeometryViewer source={source} />
      </div>

      <div className="stage-metric-grid">
        {metrics.map((metric) => (
          <div key={metric.label} className="metric-card">
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function SimulationStage({
  source,
  currentStatus,
}: {
  source:
    | { kind: "live"; runId: string | null; runStatus: string | null }
    | { kind: "history"; runPath: string };
  currentStatus: string | null;
}) {
  const canInspect =
    source.kind === "history"
      ? true
      : source.runId && ["building_system", "initializing_fsc", "manual_check_required", "solving", "exporting_artifacts", "completed"].includes(currentStatus ?? "");

  if (!canInspect) {
    return (
      <section className="stage-card stage-empty">
        <div className="stage-header">
          <h2>Snapshot Info</h2>
        </div>
        <div className="empty-stage-copy">
          <p>The simulation view becomes active after a run starts writing `run_static.npz` and step snapshots.</p>
          <p>Use the chat panel to finish the spec, then launch the run to unlock the heatmap, local LDOS, and cut views.</p>
        </div>
      </section>
    );
  }

  return <SnapshotViewer source={source.kind === "history" ? source : { kind: "live", runId: source.runId as string, runStatus: source.runStatus }} />;
}

function HistoryStage({
  historyRuns,
  selectedRunPath,
  onSelectRun,
}: {
  historyRuns: HistoryRunItem[];
  selectedRunPath: string | null;
  onSelectRun: (runPath: string) => void;
}) {
  return (
    <section className="stage-card history-stage">
      <div className="stage-header">
        <h2>History</h2>
      </div>
      <div className="history-grid">
        <div className="history-card">
          <h3>Datas</h3>
          <div className="history-scroll">
            {historyRuns.length === 0 ? <p>No saved runs found under `Datas`.</p> : historyRuns.map((run) => (
              <button
                key={run.run_path}
                className={`history-entry history-run-card ${selectedRunPath === run.run_path ? "active" : ""}`}
                onClick={() => onSelectRun(run.run_path)}
                type="button"
              >
                <span>{run.profile ?? "profile"} / {run.run_name}</span>
                <p>{run.task ?? "task unknown"} · {run.snapshot_count} snapshots · {run.status}</p>
              </button>
            ))}
          </div>
        </div>
        <div className="history-card">
          <h3>Selected Run</h3>
          {historyRuns.find((run) => run.run_path === selectedRunPath) ? (
            <dl className="history-facts">
              {(() => {
                const run = historyRuns.find((item) => item.run_path === selectedRunPath)!;
                return (
                  <>
                    <div><dt>Path</dt><dd>{run.run_name}</dd></div>
                    <div><dt>Status</dt><dd>{run.status}</dd></div>
                    <div><dt>Profile</dt><dd>{formatValue(run.profile)}</dd></div>
                    <div><dt>Task</dt><dd>{formatValue(run.task)}</dd></div>
                    <div><dt>Snapshots</dt><dd>{run.snapshot_count}</dd></div>
                  </>
                );
              })()}
            </dl>
          ) : (
            <p>Select a saved run to enter visualization mode.</p>
          )}
        </div>
      </div>
    </section>
  );
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [sessionState, setSessionState] = useState<Record<string, unknown> | null>(null);
  const [currentSpec, setCurrentSpec] = useState<SimulationSpec | null>(null);
  const [lastAgentResponse, setLastAgentResponse] = useState<AgentTurnResponse | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [runState, setRunState] = useState<RunStateResponse | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [isLaunching, setIsLaunching] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>("setup");
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("simulation-info");
  const [parserMode, setParserMode] = useState<ParserMode>("openai");
  const [historyRuns, setHistoryRuns] = useState<HistoryRunItem[]>([]);
  const [selectedHistoryRunPath, setSelectedHistoryRunPath] = useState<string | null>(null);
  const pollTimer = useRef<number | null>(null);

  const launchable = lastAgentResponse?.status === "running" && currentSpec !== null;
  const selectedHistoryRun = historyRuns.find((run) => run.run_path === selectedHistoryRunPath) ?? null;
  const displayedSpec = (selectedHistoryRun?.spec as SimulationSpec | null) ?? currentSpec;
  const visualizationSource = useMemo(
    () =>
      selectedHistoryRun
        ? ({ kind: "history", runPath: selectedHistoryRun.run_path } as const)
        : ({ kind: "live", runId, runStatus: runState?.status ?? null } as const),
    [selectedHistoryRun, runId, runState?.status],
  );
  const visualizationStatus = selectedHistoryRun?.status ?? runState?.status ?? null;

  useEffect(() => {
    fetchHistoryRuns()
      .then(setHistoryRuns)
      .catch(() => {
        setHistoryRuns([]);
      });
  }, [runState?.status]);

  useEffect(() => {
    if (!runId) {
      return;
    }
    const activeRunId = runId;

    async function loadRun() {
      try {
        const state = await fetchRun(activeRunId);
        setRunState(state);
        if (state.status === "completed" || state.status === "failed" || state.status === "aborted") {
          if (pollTimer.current !== null) {
            window.clearInterval(pollTimer.current);
            pollTimer.current = null;
          }
        }
      } catch (error) {
        setErrorMessage(error instanceof Error ? error.message : "Failed to fetch run state.");
      }
    }

    void loadRun();
    pollTimer.current = window.setInterval(() => {
      void loadRun();
    }, 2500);

    return () => {
      if (pollTimer.current !== null) {
        window.clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
    };
  }, [runId]);

  async function handleSubmit() {
    const text = draft.trim();
    if (!text) {
      return;
    }

    setErrorMessage(null);
    setIsSending(true);
    setMessages((prev) => [...prev, buildMessage("user", text)]);
    setDraft("");

    try {
      const response = await sendAgentTurn({
        message: text,
        session_state: sessionState,
        execute: false,
        parser: parserMode,
      });
      setSessionState(response.session_state);
      setCurrentSpec(response.spec);
      setLastAgentResponse(response);
      setMessages((prev) => [...prev, buildMessage("assistant", response.message, response)]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Agent request failed.";
      setErrorMessage(message);
      setMessages((prev) => [...prev, buildMessage("assistant", `Request failed: ${message}`)]);
    } finally {
      setIsSending(false);
    }
  }

  async function handleStartRun() {
    if (!currentSpec) {
      return;
    }

    setErrorMessage(null);
    setIsLaunching(true);
    setWorkspaceView("simulation");

    try {
      const response = await createRun({
        spec: currentSpec,
        parser: parserMode,
      });
      setRunId(response.run_id);
      setRunState(null);
      setSelectedHistoryRunPath(null);
      setMessages((prev) => [
        ...prev,
        buildMessage("assistant", `Run queued with id ${response.run_id}. I’ll keep updating the console while it runs.`),
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to launch run.";
      setErrorMessage(message);
    } finally {
      setIsLaunching(false);
    }
  }

  async function handleManualCheck(approved: boolean) {
    if (!runId) {
      return;
    }

    try {
      await approveManualCheck(runId, approved);
      const refreshed = await fetchRun(runId);
      setRunState(refreshed);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to submit manual check decision.");
    }
  }

  async function handleAbortRun() {
    if (!runId) {
      return;
    }

    try {
      setErrorMessage(null);
      await abortRun(runId);
      const refreshed = await fetchRun(runId);
      setRunState(refreshed);
      setMessages((prev) => [
        ...prev,
        buildMessage(
          "assistant",
          "Abort requested. If the run is waiting or between phases it will stop quickly. If it is already inside the solver, it may finish the current solver call before the abort takes effect.",
        ),
      ]);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to abort run.");
    }
  }

  const parserTabs = [
    { id: "openai" as const, label: "AI-agent" },
    { id: "langchain" as const, label: "LLM" },
    { id: "regex" as const, label: "Regex" },
  ];

  const workspaceTabs = [
    { id: "setup" as const, label: "Setup" },
    { id: "simulation" as const, label: "Simulation" },
    { id: "history" as const, label: "History" },
  ];

  const inspectorTabs = [
    { id: "simulation-info" as const, label: "Simulation Info" },
    { id: "prompt-log" as const, label: "Prompt Log" },
    { id: "terminal" as const, label: "Terminal" },
  ];

  const artifactEntries = Object.entries((runState?.result?.artifacts as Record<string, string> | undefined) ?? {});

  return (
    <div className="app-shell">
      <div className="workbench">
        <div className="workbench-main">
          <header className="topbar">
            <h1>EQuantum</h1>
            <nav className="segmented-control" aria-label="Workspace views">
              {workspaceTabs.map((tab) => (
                <button
                  key={tab.id}
                  className={workspaceView === tab.id ? "active" : ""}
                  onClick={() => setWorkspaceView(tab.id)}
                  type="button"
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </header>

          <div className="main-stage">
            {workspaceView === "setup" ? <SetupStage spec={displayedSpec} source={visualizationSource} currentStatus={visualizationStatus} /> : null}
            {workspaceView === "simulation" ? <SimulationStage source={visualizationSource} currentStatus={visualizationStatus} /> : null}
            {workspaceView === "history" ? <HistoryStage historyRuns={historyRuns} selectedRunPath={selectedHistoryRunPath} onSelectRun={(runPath) => {
              setSelectedHistoryRunPath(runPath);
              setWorkspaceView("setup");
            }} /> : null}
          </div>
        </div>

        <aside className="workbench-rail">
          <section className="rail-card rail-chat">
            <nav className="segmented-control segmented-control-rail" aria-label="Parser modes">
              {parserTabs.map((tab) => (
                <button
                  key={tab.id}
                  className={parserMode === tab.id ? "active" : ""}
                  onClick={() => setParserMode(tab.id)}
                  type="button"
                >
                  {tab.label}
                </button>
              ))}
            </nav>

            <ChatPanel
              messages={messages}
              draft={draft}
              isBusy={isSending}
              isLaunching={isLaunching}
              launchable={launchable}
              runId={runId}
              runStatus={runState?.status ?? null}
              onDraftChange={setDraft}
              onSubmit={handleSubmit}
              onStartRun={handleStartRun}
              onAbortRun={handleAbortRun}
              hideHeader
              className="chat-shell-embedded"
            />
          </section>

          <section className="rail-card rail-inspector">
            <nav className="segmented-control segmented-control-rail" aria-label="Inspector panels">
              {inspectorTabs.map((tab) => (
                <button
                  key={tab.id}
                  className={inspectorTab === tab.id ? "active" : ""}
                  onClick={() => setInspectorTab(tab.id)}
                  type="button"
                >
                  {tab.label}
                </button>
              ))}
            </nav>

            <div className="inspector-body">
              {inspectorTab === "simulation-info" ? (
                <div className="inspector-stack">
                  <div className="inspector-card-grid">
                    <div className="metric-card">
                      <span>Agent</span>
                      <strong>{lastAgentResponse?.status ?? "idle"}</strong>
                    </div>
                    <div className="metric-card">
                      <span>Run</span>
                      <strong>{runState?.status ?? (runId ? "queued" : "not started")}</strong>
                    </div>
                    <div className="metric-card">
                      <span>Parser</span>
                      <strong>{parserMode}</strong>
                    </div>
                    <div className="metric-card">
                      <span>Run ID</span>
                      <strong>{runId ?? "-"}</strong>
                    </div>
                  </div>

                  {errorMessage ? <div className="viewer-error">{errorMessage}</div> : null}

                  {runState?.manual_check_pending ? (
                    <div className="inspector-panel">
                      <h3>Manual Check</h3>
                      <p>{String(runState.manual_check_payload?.message ?? "Boundary inspection is waiting for approval.")}</p>
                      <div className="action-row">
                        <button className="primary-button" onClick={() => handleManualCheck(true)}>Approve</button>
                        <button className="danger-button" onClick={() => handleManualCheck(false)}>Reject / Abort</button>
                      </div>
                    </div>
                  ) : null}

                  <div className="inspector-panel">
                    <h3>Spec</h3>
                    <dl className="spec-list">
                      <div><dt>Task</dt><dd>{formatValue(currentSpec?.task)}</dd></div>
                      <div><dt>Profile</dt><dd>{formatValue(currentSpec?.profile)}</dd></div>
                      <div><dt>Lattice</dt><dd>{formatValue(currentSpec?.lattice_type)}</dd></div>
                      <div><dt>Device</dt><dd>{formatValue(currentSpec?.device_shape)}</dd></div>
                      <div><dt>Backgate</dt><dd>{currentSpec?.backgate_voltage !== null ? `${currentSpec?.backgate_voltage} V` : "-"}</dd></div>
                      <div><dt>Magnetic field</dt><dd>{currentSpec?.magnetic_field_T !== null ? `${currentSpec?.magnetic_field_T} T` : "-"}</dd></div>
                    </dl>
                  </div>

                  <div className="inspector-panel">
                    <h3>Artifacts</h3>
                    {artifactEntries.length === 0 ? (
                      <p>No artifacts yet.</p>
                    ) : (
                      <ul className="artifact-list">
                        {artifactEntries.map(([key, value]) => (
                          <li key={key}>
                            <strong>{key}</strong>
                            <a className="artifact-link" href={value} target="_blank" rel="noreferrer">{value}</a>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              ) : null}

              {inspectorTab === "prompt-log" ? (
                <div className="inspector-stack">
                  {messages.length === 0 ? <div className="viewer-empty">Conversation history will appear here.</div> : messages.map((message) => (
                    <article key={message.id} className={`history-entry ${message.role}`}>
                      <span>{message.role === "user" ? "You" : "Assistant"}</span>
                      <p>{message.text}</p>
                    </article>
                  ))}
                </div>
              ) : null}

              {inspectorTab === "terminal" ? (
                <div className="terminal-panel">
                  {(runState?.logs ?? []).length > 0 ? (
                    runState?.logs.map((line, index) => (
                      <div className="terminal-line" key={`${index}-${line.slice(0, 16)}`}>
                        {line}
                      </div>
                    ))
                  ) : (
                    <div className="viewer-empty">Run output will stream here after the background job starts.</div>
                  )}
                </div>
              ) : null}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
