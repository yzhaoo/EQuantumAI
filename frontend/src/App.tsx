import { useEffect, useMemo, useRef, useState } from "react";
import {
  type AgentMode,
  type PlannerModel,
  abortRun,
  approveManualCheck,
  createPlannerRun,
  createRun,
  fetchHistoryRuns,
  fetchRun,
  sendAgentTurn,
  sendPlannerTurn,
  type AgentTurnResponse,
  type HistoryRunItem,
  type RunStateResponse,
  type SimulationSpec,
} from "./api";
import { ChatPanel, type ChatMessage } from "./components/ChatPanel";
import { SetupGeometryViewer } from "./components/viewer/SetupGeometryViewer";
import { SnapshotViewer } from "./components/viewer/SnapshotViewer";
import { ConvergenceChart } from "./components/viewer/ConvergenceChart";

type WorkspaceView = "setup" | "simulation" | "history" | "conversation";
type InspectorTab = "simulation-info" | "terminal";

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

function extractPlannerArtifactDirs(response: AgentTurnResponse | null): string[] {
  const toolTrace =
    (response?.result?.tool_trace as Array<Record<string, unknown>> | undefined) ??
    (response?.session_state?.tool_trace as Array<Record<string, unknown>> | undefined) ??
    [];
  const artifactDirs: string[] = [];
  for (const step of toolTrace) {
    const result = (step.result as Record<string, unknown> | undefined) ?? {};
    const run = (result.run as Record<string, unknown> | undefined) ?? {};
    const resultPayload = (result.result as Record<string, unknown> | undefined) ?? {};
    const candidates = [run.artifact_dir, resultPayload.artifact_dir];
    for (const candidate of candidates) {
      if (typeof candidate === "string" && candidate.length > 0) {
        artifactDirs.push(candidate);
      }
    }
  }
  return artifactDirs;
}

function matchHistoryRunByArtifactDir(historyRuns: HistoryRunItem[], artifactDir: string | null): HistoryRunItem | null {
  if (!artifactDir) {
    return null;
  }
  return (
    historyRuns.find((run) => run.artifact_dir === artifactDir) ??
    historyRuns.find((run) => {
      const result = run.result as Record<string, unknown> | null;
      return result?.artifact_dir === artifactDir;
    }) ??
    null
  );
}

function SetupStage({
  source,
}: {
  source:
    | { kind: "live"; runId: string | null; runStatus: string | null }
    | { kind: "history"; runPath: string };
}) {
  return (
    <section className="stage-card">
      <div className="setup-scene">
        <SetupGeometryViewer source={source} />
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
      : Boolean(
          source.runId &&
            currentStatus &&
            !["queued", "failed", "aborted"].includes(currentStatus),
        );

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

function ConversationStage({
  messages,
}: {
  messages: ChatMessage[];
}) {
  return (
    <section className="stage-card history-stage">
      <div className="stage-header">
        <h2>Conversation Log</h2>
      </div>
      <div className="conversation-stage">
        {messages.length === 0 ? (
          <div className="viewer-empty">The raw conversation between you and the agent will appear here.</div>
        ) : (
          <div className="conversation-scroll">
            {messages.map((message, index) => (
              <article key={message.id} className={`history-entry conversation-entry ${message.role}`}>
                <div className="conversation-entry-header">
                  <span>{message.role === "user" ? "User" : "Assistant"}</span>
                  <strong>Turn {index + 1}</strong>
                </div>
                <pre className="conversation-entry-body">{message.text}</pre>
              </article>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

export default function App() {
  const plannerModels: Array<{ id: PlannerModel; label: string }> = [
    { id: "gpt-4o-mini", label: "GPT-4o Mini" },
    { id: "gpt-4.1", label: "GPT-4.1" },
    { id: "gpt-5", label: "GPT-5" },
  ];

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [parserSessionState, setParserSessionState] = useState<Record<string, unknown> | null>(null);
  const [plannerSessionState, setPlannerSessionState] = useState<Record<string, unknown> | null>(null);
  const [currentSpec, setCurrentSpec] = useState<SimulationSpec | null>(null);
  const [lastAgentResponse, setLastAgentResponse] = useState<AgentTurnResponse | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [runState, setRunState] = useState<RunStateResponse | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [isLaunching, setIsLaunching] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>("setup");
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("simulation-info");
  const [agentMode, setAgentMode] = useState<AgentMode>("parser");
  const [plannerModel, setPlannerModel] = useState<PlannerModel>("gpt-4o-mini");
  const [historyRuns, setHistoryRuns] = useState<HistoryRunItem[]>([]);
  const [selectedHistoryRunPath, setSelectedHistoryRunPath] = useState<string | null>(null);
  const pollTimer = useRef<number | null>(null);
  const lastRunNoticeKey = useRef<string | null>(null);

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
  const plannerToolTrace =
    agentMode === "planner"
      ? (((lastAgentResponse?.result?.tool_trace as Array<Record<string, unknown>> | undefined) ??
          (lastAgentResponse?.session_state?.tool_trace as Array<Record<string, unknown>> | undefined) ??
          []) as Array<Record<string, unknown>>)
      : [];

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

  useEffect(() => {
    if (!runId || !runState?.status) {
      return;
    }
    const status = runState.status;
    if (!["failed", "aborted", "completed"].includes(status)) {
      return;
    }

    const noticeKey = `${runId}:${status}:${runState.error ?? ""}`;
    if (lastRunNoticeKey.current === noticeKey) {
      return;
    }
    lastRunNoticeKey.current = noticeKey;

    if (status === "failed") {
      const failureMessage = runState.error?.trim()
        ? `Run failed: ${runState.error}`
        : "Run failed. Check the terminal panel for details.";
      setMessages((prev) => [...prev, buildMessage("assistant", failureMessage)]);
      return;
    }

    if (status === "aborted") {
      const abortedMessage = runState.error?.trim()
        ? `Run aborted: ${runState.error}`
        : "Run aborted.";
      setMessages((prev) => [...prev, buildMessage("assistant", abortedMessage)]);
      return;
    }

    setMessages((prev) => [...prev, buildMessage("assistant", "Run completed.")]);
  }, [runId, runState?.status, runState?.error]);

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
      const response =
        agentMode === "planner"
          ? await sendPlannerTurn({
              message: text,
              session_state: plannerSessionState,
              parser: "openai",
              openai_model: plannerModel,
              max_iterations: 12,
            })
          : await sendAgentTurn({
              message: text,
              session_state: parserSessionState,
              execute: false,
              parser: "openai",
            });
      if (agentMode === "planner") {
        setPlannerSessionState(response.session_state);
      } else {
        setParserSessionState(response.session_state);
      }
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
    setErrorMessage(null);
    setIsLaunching(true);
    setWorkspaceView("simulation");

    try {
      if (agentMode === "planner") {
        const approvedPlan = (plannerSessionState?.approved_plan as Array<Record<string, unknown>> | undefined) ?? [];
        const originalRequest = (plannerSessionState?.original_request as string | undefined) ?? "";
        if (approvedPlan.length === 0) {
          throw new Error("No approved planner workflow is ready to run.");
        }
        const response = await createPlannerRun({
          approved_plan: approvedPlan,
          original_request: originalRequest,
          parser: "openai",
          openai_model: plannerModel,
        });
        setRunId(response.run_id);
        setRunState(null);
        setSelectedHistoryRunPath(null);
        setMessages((prev) => [
          ...prev,
          buildMessage("assistant", `Planner run queued with id ${response.run_id}. I’ll keep updating the console while it runs.`),
        ]);
      } else {
        if (!currentSpec) {
          return;
        }
        const response = await createRun({
          spec: currentSpec,
          parser: "openai",
        });
        setRunId(response.run_id);
        setRunState(null);
        setSelectedHistoryRunPath(null);
        setMessages((prev) => [
          ...prev,
          buildMessage("assistant", `Run queued with id ${response.run_id}. I’ll keep updating the console while it runs.`),
        ]);
      }
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

  const agentTabs = [
    { id: "parser" as const, label: "LLM Parser" },
    { id: "planner" as const, label: "Task Planner" },
  ];

  const workspaceTabs = [
    { id: "setup" as const, label: "Setup" },
    { id: "simulation" as const, label: "Simulation" },
    { id: "history" as const, label: "History" },
    { id: "conversation" as const, label: "Conversation Log" },
  ];

  const inspectorTabs = [
    { id: "simulation-info" as const, label: "Simulation Info" },
    { id: "terminal" as const, label: "Terminal" },
  ];

  const displayedResult =
    (selectedHistoryRun?.result as Record<string, unknown> | null) ??
    (runState?.result as Record<string, unknown> | null) ??
    null;
  const artifactEntries = Object.entries((displayedResult?.artifacts as Record<string, string> | undefined) ?? {});

  useEffect(() => {
    if (agentMode !== "planner") {
      return;
    }
    if (!runState || runState.status !== "completed") {
      return;
    }
    void fetchHistoryRuns()
      .then((refreshedHistory) => {
        setHistoryRuns(refreshedHistory);
        const result = runState.result as Record<string, unknown> | null;
        const latestArtifactDir = typeof result?.artifact_dir === "string" ? result.artifact_dir : null;
        const matchedRun = matchHistoryRunByArtifactDir(refreshedHistory, latestArtifactDir);
        if (matchedRun) {
          setSelectedHistoryRunPath(matchedRun.run_path);
          setWorkspaceView("simulation");
        }
      })
      .catch(() => {
        return;
      });
  }, [agentMode, runState]);

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
            {workspaceView === "setup" ? <SetupStage source={visualizationSource} /> : null}
            {workspaceView === "simulation" ? <SimulationStage source={visualizationSource} currentStatus={visualizationStatus} /> : null}
            {workspaceView === "history" ? <HistoryStage historyRuns={historyRuns} selectedRunPath={selectedHistoryRunPath} onSelectRun={(runPath) => {
              setSelectedHistoryRunPath(runPath);
              setWorkspaceView("setup");
            }} /> : null}
            {workspaceView === "conversation" ? <ConversationStage messages={messages} /> : null}
          </div>
        </div>

        <aside className="workbench-rail">
          <section className="rail-card rail-chat">
            <nav className="segmented-control segmented-control-rail" aria-label="Agent modes">
              {agentTabs.map((tab) => (
                <button
                  key={tab.id}
                  className={agentMode === tab.id ? "active" : ""}
                  onClick={() => setAgentMode(tab.id)}
                  type="button"
                >
                  {tab.label}
                </button>
              ))}
            </nav>

            {agentMode === "planner" ? (
              <nav className="segmented-control segmented-control-rail" aria-label="Planner models">
                {plannerModels.map((model) => (
                  <button
                    key={model.id}
                    className={plannerModel === model.id ? "active" : ""}
                    onClick={() => setPlannerModel(model.id)}
                    type="button"
                  >
                    {model.label}
                  </button>
                ))}
              </nav>
            ) : null}

            <ChatPanel
              messages={messages}
              draft={draft}
              isBusy={isSending}
              isLaunching={isLaunching}
              launchable={launchable}
              runId={runId}
              runStatus={runState?.status ?? null}
              agentMode={agentMode}
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
                      <strong>{selectedHistoryRun?.status ?? runState?.status ?? (runId ? "queued" : "not started")}</strong>
                    </div>
                    <div className="metric-card">
                      <span>Mode</span>
                      <strong>{agentMode === "planner" ? `planner:${plannerModel}` : "parser:openai"}</strong>
                    </div>
                    <div className="metric-card">
                      <span>Run ID</span>
                      <strong>{runId ?? "-"}</strong>
                    </div>
                    <div className="metric-card">
                      <span>Profile</span>
                      <strong>{displayedSpec?.profile ?? "dotgate_center"}</strong>
                    </div>
                    <div className="metric-card">
                      <span>Task</span>
                      <strong>{displayedSpec?.task ?? "dos"}</strong>
                    </div>
                    <div className="metric-card">
                      <span>Lattice</span>
                      <strong>{displayedSpec?.lattice_type ?? "square"}</strong>
                    </div>
                    <div className="metric-card">
                      <span>Magnetic field</span>
                      <strong>{displayedSpec?.magnetic_field_T !== null ? `${displayedSpec?.magnetic_field_T} T` : "-"}</strong>
                    </div>
                    <div className="metric-card">
                      <span>Backgate</span>
                      <strong>{displayedSpec?.backgate_voltage !== null ? `${displayedSpec?.backgate_voltage} V` : "-"}</strong>
                    </div>
                    <div className="metric-card">
                      <span>Run state</span>
                      <strong>{visualizationStatus ?? "idle"}</strong>
                    </div>
                  </div>

                  {errorMessage ? <div className="viewer-error">{errorMessage}</div> : null}
                  {runState?.error ? <div className="viewer-error">{runState.error}</div> : null}

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
                      <div><dt>Self-consistent</dt><dd>{formatValue(currentSpec?.solve_self_consistent)}</dd></div>
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

                  {agentMode === "planner" ? (
                    <div className="inspector-panel">
                      <h3>Planner Trace</h3>
                      {plannerToolTrace.length === 0 ? (
                        <p>No planner tool calls recorded yet.</p>
                      ) : (
                        <ol className="artifact-list">
                          {plannerToolTrace.map((entry, index) => (
                            <li key={`planner-trace-${index}`}>
                              <strong>{String(entry.tool_name ?? `step_${index + 1}`)}</strong>
                              <span>{JSON.stringify(entry.arguments ?? {})}</span>
                            </li>
                          ))}
                        </ol>
                      )}
                    </div>
                  ) : null}
                </div>
              ) : null}

              {inspectorTab === "terminal" ? (
                <div style={{ display: "flex", flexDirection: "row", height: "100%", gap: "1rem" }}>
                  <div className="terminal-panel" style={{ flex: 1, overflowY: "auto" }}>
                    {(runState?.logs ?? []).length > 0 ? (
                      runState?.logs.map((line, index) => (
                        <div className="terminal-line" key={`${index}-${line.slice(0, 16)}`}>
                          {line}
                        </div>
                      ))
                    ) : runState?.error ? (
                      <div className="viewer-error">{runState.error}</div>
                    ) : (
                      <div className="viewer-empty">Run output will stream here after the background job starts.</div>
                    )}
                  </div>
                  <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
                    <ConvergenceChart events={runState?.events ?? []} />
                  </div>
                </div>
              ) : null}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
