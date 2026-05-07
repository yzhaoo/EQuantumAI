import { useEffect, useMemo, useRef, useState } from "react";
import {
  type AgentMode,
  type PlannerModel,
  abortRun,
  approveManualCheck,
  callTool,
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
import { ResultsViewer } from "./components/viewer/ResultsViewer";

type WorkspaceView = "setup" | "simulation" | "results" | "history" | "conversation";
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
  selectedRunPaths,
  onToggleRun,
}: {
  historyRuns: HistoryRunItem[];
  selectedRunPaths: string[];
  onToggleRun: (runPath: string) => void;
}) {
  const selectedRuns = historyRuns.filter((run) => selectedRunPaths.includes(run.run_path));
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
                className={`history-entry history-run-card ${selectedRunPaths.includes(run.run_path) ? "active" : ""}`}
                onClick={() => onToggleRun(run.run_path)}
                type="button"
              >
                <span>{run.profile ?? "profile"} / {run.run_name}</span>
                <p>{run.task ?? "task unknown"} · {run.snapshot_count} snapshots · {run.status}</p>
              </button>
            ))}
          </div>
        </div>
        <div className="history-card">
          <h3>{selectedRuns.length === 2 ? "Selected Runs For Comparison" : "Selected Run"}</h3>
          {selectedRuns.length > 0 ? (
            <div className="history-scroll">
              {selectedRuns.map((run) => (
                <dl key={run.run_path} className="history-facts">
                  <div><dt>Path</dt><dd>{run.run_name}</dd></div>
                  <div><dt>Status</dt><dd>{run.status}</dd></div>
                  <div><dt>Profile</dt><dd>{formatValue(run.profile)}</dd></div>
                  <div><dt>Task</dt><dd>{formatValue(run.task)}</dd></div>
                  <div><dt>Snapshots</dt><dd>{run.snapshot_count}</dd></div>
                </dl>
              ))}
            </div>
          ) : (
            <p>Select one run to inspect it, or two runs to make them available for comparison on the results page and in the agent context.</p>
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
              (() => {
                const modelRequest = message.response?.result?.model_request;
                const formattedRequest =
                  modelRequest && typeof modelRequest === "object"
                    ? JSON.stringify(modelRequest, null, 2)
                    : null;
                return (
                  <article key={message.id} className={`history-entry conversation-entry ${message.role}`}>
                    <div className="conversation-entry-header">
                      <span>{message.role === "user" ? "User" : "Assistant"}</span>
                      <strong>Turn {index + 1}</strong>
                    </div>
                    <pre className="conversation-entry-body">{message.text}</pre>
                    {message.role === "assistant" && formattedRequest ? (
                      <section className="conversation-json-panel">
                        <div className="conversation-json-header">Model Request JSON</div>
                        <pre className="conversation-json-body">{formattedRequest}</pre>
                      </section>
                    ) : null}
                  </article>
                );
              })()
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

type ResultVisualization =
  | { type: "ldos_volume_3d"; artifact_dir?: string; snapshot?: string; energy?: number[]; site_ids?: number[]; ldos_matrix?: Array<Array<number | null>> }
  | { type: "ldos_linecut_with_ui"; artifact_dir?: string; snapshot?: string; distance_along?: Array<number | null>; energy?: Array<number | null>; ldos_matrix?: Array<Array<number | null>>; overlays?: Array<{ snapshot: string; Ui: Array<number | null>; is_current: boolean }> }
  | { type: "ldos_linecut_multi"; items?: Array<Record<string, unknown>> }
  | { type: "ui_linecut_compare"; traces?: Array<{ label?: string; distance_along?: Array<number | null>; values?: Array<number | null> }> }
  | { type: "ldos_ui_heatmap"; artifact_dir?: string; snapshot?: string; x?: Array<number | null>; y?: Array<number | null>; values?: Array<number | null>; color_min?: number; color_max?: number }
  | { type: "ldos_ui_heatmap_multi"; items?: Array<Record<string, unknown>> }
  | { type: "ldos_comparison"; label_a?: string; label_b?: string; energy?: number[]; ldos_a?: number[]; ldos_b?: number[]; delta?: number[] }
  | Record<string, unknown>;

type ResultPlotOption = {
  id: "ldos_volume_3d" | "ldos_linecut_with_ui" | "ldos_ui_heatmap" | "ldos_comparison" | "ui_linecut_compare";
  label: string;
  description: string;
};

function extractResultVisualizations(
  response: AgentTurnResponse | null,
  runState: RunStateResponse | null,
): ResultVisualization[] {
  const visualizations: ResultVisualization[] = [];
  const pushIfVisualization = (value: unknown) => {
    if (value && typeof value === "object" && typeof (value as { type?: unknown }).type === "string") {
      visualizations.push(value as ResultVisualization);
    }
  };

  const responseTrace =
    (response?.result?.tool_trace as Array<Record<string, unknown>> | undefined) ??
    (response?.session_state?.tool_trace as Array<Record<string, unknown>> | undefined) ??
    [];
  for (const step of responseTrace) {
    const result = step.result as Record<string, unknown> | undefined;
    pushIfVisualization(result?.visualization);
  }

  const runTrace = (runState?.result?.tool_trace as Array<Record<string, unknown>> | undefined) ?? [];
  for (const step of runTrace) {
    const result = step.result as Record<string, unknown> | undefined;
    pushIfVisualization(result?.visualization);
  }
  return visualizations;
}

function ResultStage({
  source,
  visualizations,
  infoLines,
  plotOptions,
  onOpenPlot,
  onClearPlots,
}: {
  source:
    | { kind: "live"; runId: string | null; runStatus: string | null }
    | { kind: "history"; runPath: string };
  visualizations: ResultVisualization[];
  infoLines: string[];
  plotOptions: ResultPlotOption[];
  onOpenPlot: (plotId: ResultPlotOption["id"]) => void;
  onClearPlots: () => void;
}) {
  return (
    <section className="stage-card">
      <div className="stage-header">
        <h2>Results</h2>
      </div>
      <ResultsViewer
        visualizations={visualizations}
        infoLines={infoLines}
        plotOptions={plotOptions}
        onOpenPlot={onOpenPlot}
        onClearPlots={onClearPlots}
        source={source}
      />
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
  const [selectedHistoryRunPaths, setSelectedHistoryRunPaths] = useState<string[]>([]);
  const [manualResultVisualizations, setManualResultVisualizations] = useState<ResultVisualization[]>([]);
  const pollTimer = useRef<number | null>(null);
  const lastRunNoticeKey = useRef<string | null>(null);

  const launchable = lastAgentResponse?.status === "running" && currentSpec !== null;
  const selectedHistoryRun = historyRuns.find((run) => run.run_path === selectedHistoryRunPath) ?? null;
  const selectedHistoryRuns = historyRuns.filter((run) => selectedHistoryRunPaths.includes(run.run_path));
  const selectedHistoryRunContext = useMemo(
    () =>
      selectedHistoryRuns
        .filter((run) => typeof run.artifact_dir === "string" && run.artifact_dir.length > 0)
        .map((run) => ({
          artifact_dir: run.artifact_dir,
          spec: run.spec,
          result: run.result,
        })),
    [selectedHistoryRuns],
  );
  const currentCompletedRunContext = useMemo(() => {
    if (runState?.status !== "completed" || !runState.result || typeof runState.result !== "object") {
      return [];
    }
    const artifact_dir = typeof runState.result.artifact_dir === "string" ? runState.result.artifact_dir : null;
    if (!artifact_dir) {
      return [];
    }
    return [{ artifact_dir, spec: runState.spec, result: runState.result }];
  }, [runState]);
  const resultContextRuns = selectedHistoryRunContext.length > 0 ? selectedHistoryRunContext : currentCompletedRunContext;
  const displayedSpec = (selectedHistoryRun?.spec as SimulationSpec | null) ?? currentSpec;
  const visualizationSource = useMemo(
    () =>
      selectedHistoryRun
        ? ({ kind: "history", runPath: selectedHistoryRun.run_path } as const)
        : ({ kind: "live", runId, runStatus: runState?.status ?? null } as const),
    [selectedHistoryRun, runId, runState?.status],
  );
  const visualizationStatus = selectedHistoryRun?.status ?? runState?.status ?? null;
  const resultVisualizations = useMemo(
    () => {
      const fromAgent = extractResultVisualizations(lastAgentResponse, runState);
      return manualResultVisualizations.length > 0 ? [...manualResultVisualizations, ...fromAgent] : fromAgent;
    },
    [lastAgentResponse, runState, manualResultVisualizations],
  );
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
    setManualResultVisualizations([]);
  }, [selectedHistoryRunPaths.join("|"), runState?.result ? JSON.stringify((runState.result as Record<string, unknown>).artifact_dir ?? null) : ""]);

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

    setMessages((prev) => [
      ...prev,
      buildMessage(
        "assistant",
        "Run completed. I’m standing by for your next run-planning request.",
      ),
    ]);
    const completedRunContext =
      runState.result && typeof runState.result === "object"
        ? {
            artifact_dir: typeof runState.result.artifact_dir === "string" ? runState.result.artifact_dir : null,
            spec: runState.spec,
            result: runState.result,
          }
        : null;
    if (completedRunContext?.artifact_dir) {
      setPlannerSessionState((prev) => ({
        ...(prev ?? {}),
        completed_runs: [...(((prev?.completed_runs as Array<Record<string, unknown>> | undefined) ?? []).filter((entry) => entry?.artifact_dir !== completedRunContext.artifact_dir)), completedRunContext].slice(-4),
      }));
      setParserSessionState((prev) => ({
        ...(prev ?? {}),
        completed_runs: [...(((prev?.completed_runs as Array<Record<string, unknown>> | undefined) ?? []).filter((entry) => entry?.artifact_dir !== completedRunContext.artifact_dir)), completedRunContext].slice(-4),
      }));
    }
    setWorkspaceView("results");
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
      const mergeCompletedRuns = (state: Record<string, unknown> | null | undefined) => {
        const baseRuns = ((state?.completed_runs as Array<Record<string, unknown>> | undefined) ?? []);
        const merged = [...baseRuns];
        for (const run of selectedHistoryRunContext) {
          if (!merged.some((entry) => entry?.artifact_dir === run.artifact_dir)) {
            merged.push(run);
          }
        }
        return {
          ...(state ?? {}),
          completed_runs: merged.slice(-4),
        };
      };
      const response =
        agentMode === "planner"
          ? await sendPlannerTurn({
              message: text,
              session_state: mergeCompletedRuns(plannerSessionState),
              parser: "openai",
              openai_model: plannerModel,
              max_iterations: 12,
            })
          : await sendAgentTurn({
              message: text,
              session_state: mergeCompletedRuns(parserSessionState),
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

  const resultInfoLines = useMemo(() => {
    if (selectedHistoryRuns.length === 2) {
      return ["Comparison context", selectedHistoryRuns[0].run_name, selectedHistoryRuns[1].run_name];
    }
    if (selectedHistoryRuns.length === 1) {
      return ["History run selected", selectedHistoryRuns[0].run_name];
    }
    if (currentCompletedRunContext.length === 1) {
      return [
        "Current completed run",
        String(currentCompletedRunContext[0].artifact_dir).split("/").slice(-2).join("/"),
      ];
    }
    return ["No run selected yet", "Select one history run to inspect it, or two to compare them."];
  }, [selectedHistoryRuns, currentCompletedRunContext]);

  const resultPlotOptions = useMemo<ResultPlotOption[]>(() => {
    if (resultContextRuns.length >= 2) {
      return [
        { id: "ldos_comparison", label: "LDOS Compare", description: "Compare aligned LDOS curves from two completed runs." },
        { id: "ldos_linecut_with_ui", label: "DOS Heatmap", description: "Open one LDOS heatmap-along-linecut window for each selected run." },
        { id: "ldos_ui_heatmap", label: "LDOS @ Ui", description: "Open the LDOS@Ui heatmaps for both selected runs in one window." },
        { id: "ui_linecut_compare", label: "Ui Linecut", description: "Compare Ui along the default linecut across both runs." },
      ];
    }
    if (resultContextRuns.length === 1) {
      return [
        { id: "ldos_volume_3d", label: "3D LDOS", description: "Energy vs site id vs density surface." },
        { id: "ldos_linecut_with_ui", label: "LDOS Linecut", description: "LDOS cut with Ui overlay." },
        { id: "ldos_ui_heatmap", label: "LDOS @ Ui", description: "Quantum-layer LDOS@Ui heatmap." },
      ];
    }
    return [];
  }, [resultContextRuns]);

  async function handleOpenResultPlot(plotId: ResultPlotOption["id"]) {
    if (resultContextRuns.length === 0) {
      setErrorMessage("No completed run context is available for the results page yet.");
      return;
    }
    setErrorMessage(null);
    try {
      let payload;
      if (plotId === "ldos_comparison") {
        if (resultContextRuns.length < 2) {
          throw new Error("Select two runs before opening an LDOS comparison.");
        }
        payload = await callTool({
          tool_name: "compare_ldos_runs",
          arguments: {
            artifact_dir_a: resultContextRuns[0].artifact_dir,
            artifact_dir_b: resultContextRuns[1].artifact_dir,
          },
          parser: "openai",
        });
        const visualization = payload.result?.visualization;
        if (visualization && typeof visualization === "object") {
          setManualResultVisualizations((prev) => {
            const next = [
              ...prev.filter((item) => String((item as { type?: unknown }).type ?? "") !== "ldos_comparison"),
              visualization as ResultVisualization,
            ];
            return next;
          });
          setWorkspaceView("results");
        }
        return;
      } else if (plotId === "ui_linecut_compare") {
        if (resultContextRuns.length < 2) {
          throw new Error("Select two runs before opening a Ui linecut comparison.");
        }
        payload = await callTool({
          tool_name: "compare_ui_linecuts",
          arguments: {
            artifact_dirs: resultContextRuns.map((run) => run.artifact_dir),
          },
          parser: "openai",
        });
        const visualization = payload.result?.visualization;
        if (visualization && typeof visualization === "object") {
          setManualResultVisualizations((prev) => {
            const next = [
              ...prev.filter((item) => String((item as { type?: unknown }).type ?? "") !== "ui_linecut_compare"),
              visualization as ResultVisualization,
            ];
            return next;
          });
          setWorkspaceView("results");
        }
        return;
      } else if (plotId === "ldos_linecut_with_ui" && resultContextRuns.length >= 2) {
        const results = await Promise.all(
          resultContextRuns.map((run) =>
            callTool({
              tool_name: "show_ldos_linecut_with_ui",
              arguments: {
                artifact_dir: run.artifact_dir,
              },
              parser: "openai",
            }),
          ),
        );
        const visualizations = results
          .map((response) => response.result?.visualization)
          .filter((visualization): visualization is ResultVisualization => Boolean(visualization && typeof visualization === "object"));
        if (visualizations.length > 0) {
          const combinedVisualization: ResultVisualization = {
            type: "ldos_linecut_multi",
            items: visualizations as Array<Record<string, unknown>>,
          };
          setManualResultVisualizations((prev) => {
            const preserved = prev.filter((item) => {
              const itemType = String((item as { type?: unknown }).type ?? "");
              return itemType !== "ldos_linecut_with_ui" && itemType !== "ldos_linecut_multi";
            });
            return [...preserved, combinedVisualization];
          });
          setWorkspaceView("results");
        }
        return;
      } else if (plotId === "ldos_ui_heatmap" && resultContextRuns.length >= 2) {
        const results = await Promise.all(
          resultContextRuns.map((run) =>
            callTool({
              tool_name: "show_ldos_ui_heatmap",
              arguments: {
                artifact_dir: run.artifact_dir,
              },
              parser: "openai",
            }),
          ),
        );
        const visualizations = results
          .map((response) => response.result?.visualization)
          .filter((visualization): visualization is ResultVisualization => Boolean(visualization && typeof visualization === "object"));
        if (visualizations.length > 0) {
          const combinedVisualization: ResultVisualization = {
            type: "ldos_ui_heatmap_multi",
            items: visualizations as Array<Record<string, unknown>>,
          };
          setManualResultVisualizations((prev) => {
            const preserved = prev.filter((item) => {
              const itemType = String((item as { type?: unknown }).type ?? "");
              return itemType !== "ldos_ui_heatmap" && itemType !== "ldos_ui_heatmap_multi";
            });
            return [...preserved, combinedVisualization];
          });
          setWorkspaceView("results");
        }
        return;
      } else {
        payload = await callTool({
          tool_name: plotId === "ldos_volume_3d" ? "show_ldos_volume_3d" : plotId === "ldos_linecut_with_ui" ? "show_ldos_linecut_with_ui" : "show_ldos_ui_heatmap",
          arguments: {
            artifact_dir: resultContextRuns[0].artifact_dir,
          },
          parser: "openai",
        });
      }
      const visualization = payload.result?.visualization;
      if (visualization && typeof visualization === "object") {
        setManualResultVisualizations((prev) => {
          const incoming = visualization as ResultVisualization;
          const incomingType = String((incoming as { type?: unknown }).type ?? "");
          const incomingArtifact = String((incoming as { artifact_dir?: unknown }).artifact_dir ?? "");
          const preserved = prev.filter((item) => {
            const existingType = String((item as { type?: unknown }).type ?? "");
            const existingArtifact = String((item as { artifact_dir?: unknown }).artifact_dir ?? "");
            if (incomingType === "ldos_linecut_with_ui") {
              return !(existingType === incomingType && existingArtifact === incomingArtifact);
            }
            return existingType !== incomingType;
          });
          return [...preserved, incoming];
        });
        setWorkspaceView("results");
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to open the requested results plot.");
    }
  }

  function handleClearResultPlots() {
    setManualResultVisualizations([]);
  }

  function handleToggleHistoryRun(runPath: string) {
    setSelectedHistoryRunPaths((prev) => {
      if (prev.includes(runPath)) {
        const next = prev.filter((item) => item !== runPath);
        setSelectedHistoryRunPath(next.length > 0 ? next[next.length - 1] : null);
        return next;
      }
      const next = [...prev, runPath].slice(-2);
      setSelectedHistoryRunPath(runPath);
      return next;
    });
  }

  const agentTabs = [
    { id: "parser" as const, label: "LLM Parser" },
    { id: "planner" as const, label: "Task Planner" },
  ];

  const workspaceTabs = [
    { id: "setup" as const, label: "Setup" },
    { id: "simulation" as const, label: "Simulation" },
    { id: "results" as const, label: "Results" },
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
          setSelectedHistoryRunPaths((prev) => {
            const merged = [...prev.filter((path) => path !== matchedRun.run_path), matchedRun.run_path];
            return merged.slice(-2);
          });
          setWorkspaceView("results");
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
            {workspaceView === "results" ? (
              <ResultStage
                source={visualizationSource}
                visualizations={resultVisualizations}
                infoLines={resultInfoLines}
                plotOptions={resultPlotOptions}
                onOpenPlot={handleOpenResultPlot}
                onClearPlots={handleClearResultPlots}
              />
            ) : null}
            {workspaceView === "history" ? <HistoryStage historyRuns={historyRuns} selectedRunPaths={selectedHistoryRunPaths} onToggleRun={handleToggleHistoryRun} /> : null}
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
                      <strong>{displayedSpec?.task ?? "artifact-driven"}</strong>
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
