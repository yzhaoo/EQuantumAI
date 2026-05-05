import type { AgentMode, AgentTurnResponse } from "../api";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  response?: AgentTurnResponse;
};

type ChatPanelProps = {
  messages: ChatMessage[];
  draft: string;
  isBusy: boolean;
  isLaunching: boolean;
  launchable: boolean;
  runId: string | null;
  runStatus: string | null;
  agentMode: AgentMode;
  onDraftChange: (value: string) => void;
  onSubmit: () => void;
  onStartRun: () => void;
  onAbortRun: () => void;
  hideHeader?: boolean;
  className?: string;
};

type PlannerPlanStep = {
  tool_name?: string;
  purpose?: string;
  arguments?: Record<string, unknown>;
};

type PlannerSpecReviewRun = {
  step_index?: number;
  tool_name?: string;
  purpose?: string;
  requested_spec?: Record<string, unknown>;
  resolved_spec?: Record<string, unknown>;
  defaults_used?: Record<string, unknown>;
  warnings?: string[];
  missing_fields?: string[];
  ok?: boolean;
};

function formatJson(value: unknown) {
  return JSON.stringify(value ?? {}, null, 2);
}

function renderPlannerStepCard(
  step: {
    tool_name?: string;
    purpose?: string;
    arguments?: Record<string, unknown>;
  },
  index: number,
) {
  return (
    <div
      key={`planner-step-${index}`}
      style={{
        border: "1px solid rgba(15, 23, 42, 0.08)",
        borderRadius: 16,
        padding: 14,
        background: "rgba(255, 255, 255, 0.72)",
        display: "grid",
        gap: 10,
        minWidth: 0,
        overflow: "hidden",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", minWidth: 0 }}>
        <span
          style={{
            fontSize: 12,
            fontWeight: 700,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
            opacity: 0.65,
          }}
        >
          Step {index + 1}
        </span>
        <code
          style={{
            fontSize: 14,
            fontWeight: 700,
            minWidth: 0,
            overflowWrap: "anywhere",
            wordBreak: "break-word",
          }}
        >
          {step.tool_name ?? "-"}
        </code>
      </div>
      {step.purpose ? (
        <p
          style={{
            margin: 0,
            minWidth: 0,
            overflowWrap: "anywhere",
            wordBreak: "break-word",
          }}
        >
          {step.purpose}
        </p>
      ) : null}
      <div>
        <div
          style={{
            fontSize: 12,
            fontWeight: 700,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
            opacity: 0.65,
            marginBottom: 6,
          }}
        >
          Parameters
        </div>
        <pre
          style={{
            margin: 0,
            padding: 12,
            borderRadius: 12,
            background: "rgba(15, 23, 42, 0.04)",
            maxWidth: "100%",
            maxHeight: 220,
            overflowX: "auto",
            overflowY: "auto",
            whiteSpace: "pre-wrap",
            overflowWrap: "anywhere",
            wordBreak: "break-word",
            boxSizing: "border-box",
          }}
        >
          <code>{formatJson(step.arguments)}</code>
        </pre>
      </div>
    </div>
  );
}

function renderPlannerPlan(response: AgentTurnResponse) {
  const planPreview = response.result?.plan_preview as
    | { summary?: string; human_readable_plan?: string; tool_calls?: PlannerPlanStep[] }
    | undefined;
  if (!planPreview || !Array.isArray(planPreview.tool_calls) || planPreview.tool_calls.length === 0) {
    return null;
  }

  return (
    <div className="assistant-card">
      <p className="assistant-card-title">Planned Workflow</p>
      {planPreview.summary ? <p>{planPreview.summary}</p> : null}
      {planPreview.human_readable_plan ? (
        <pre
          style={{
            margin: "10px 0 0",
            padding: 12,
            borderRadius: 12,
            background: "rgba(15, 23, 42, 0.04)",
            whiteSpace: "pre-wrap",
            overflowWrap: "anywhere",
            wordBreak: "break-word",
          }}
        >
          {planPreview.human_readable_plan}
        </pre>
      ) : null}
      <div style={{ display: "grid", gap: 12 }}>
        {planPreview.tool_calls.map((step, index) => renderPlannerStepCard(step, index))}
      </div>
    </div>
  );
}

function renderPlannerExecutionTrace(response: AgentTurnResponse) {
  const toolTrace = response.result?.tool_trace as
    | Array<{ tool_name?: string; purpose?: string; arguments?: Record<string, unknown>; result?: unknown }>
    | undefined;
  if (!toolTrace || toolTrace.length === 0) {
    return null;
  }

  return (
    <div className="assistant-card">
      <p className="assistant-card-title">Executed Workflow</p>
      <div style={{ display: "grid", gap: 12 }}>
        {toolTrace.map((step, index) => renderPlannerStepCard(step, index))}
      </div>
    </div>
  );
}

function renderSpecList(spec: Record<string, unknown> | undefined, title: string) {
  const entries = Object.entries(spec ?? {});
  if (entries.length === 0) {
    return null;
  }
  return (
    <div>
      <div
        style={{
          fontSize: 12,
          fontWeight: 700,
          letterSpacing: "0.04em",
          textTransform: "uppercase",
          opacity: 0.65,
          marginBottom: 6,
        }}
      >
        {title}
      </div>
      <div style={{ display: "grid", gap: 6 }}>
        {entries.map(([key, value]) => (
          <div
            key={`${title}-${key}`}
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(110px, 160px) minmax(0, 1fr)",
              gap: 10,
              alignItems: "start",
            }}
          >
            <strong style={{ fontSize: 12, color: "rgba(15, 23, 42, 0.7)" }}>{key}</strong>
            <span style={{ overflowWrap: "anywhere", wordBreak: "break-word" }}>
              {typeof value === "string" ? value : JSON.stringify(value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function renderPlannerSpecReview(response: AgentTurnResponse) {
  const specReview = response.result?.spec_review as
    | { summary?: string; needs_confirmation_message?: string; requires_default_confirmation?: boolean; runs?: PlannerSpecReviewRun[] }
    | undefined;
  if (!specReview || !Array.isArray(specReview.runs) || specReview.runs.length === 0) {
    return null;
  }

  return (
    <div className="assistant-card">
      <p className="assistant-card-title">{specReview.requires_default_confirmation ? "Defaults Check" : "Spec Check"}</p>
      {specReview.summary ? <p>{specReview.summary}</p> : null}
      <div style={{ display: "grid", gap: 12, marginTop: 10 }}>
        {specReview.runs.map((run, index) => (
          <div
            key={`spec-review-${index}`}
            style={{
              border: "1px solid rgba(15, 23, 42, 0.08)",
              borderRadius: 16,
              padding: 14,
              background: "rgba(255, 255, 255, 0.72)",
              display: "grid",
              gap: 10,
            }}
          >
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
              <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: "0.04em", textTransform: "uppercase", opacity: 0.65 }}>
                Run {index + 1}
              </span>
              <code style={{ fontSize: 14, fontWeight: 700 }}>{run.tool_name ?? "-"}</code>
            </div>
            {run.purpose ? <p style={{ margin: 0 }}>{run.purpose}</p> : null}
            {renderSpecList(run.resolved_spec, "Resolved Parameters")}
            {run.defaults_used && Object.keys(run.defaults_used).length > 0 ? renderSpecList(run.defaults_used, "Defaulted Values") : null}
            {run.warnings && run.warnings.length > 0 ? (
              <div>
                <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: "0.04em", textTransform: "uppercase", opacity: 0.65, marginBottom: 6 }}>
                  Warnings
                </div>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {run.warnings.map((warning, warningIndex) => <li key={`warning-${index}-${warningIndex}`}>{warning}</li>)}
                </ul>
              </div>
            ) : null}
          </div>
        ))}
      </div>
      {specReview.needs_confirmation_message ? <p style={{ marginTop: 10 }}>{specReview.needs_confirmation_message}</p> : null}
    </div>
  );
}

function renderPlannerSpecFallback(response: AgentTurnResponse) {
  const spec = response.spec as Record<string, unknown> | undefined;
  const hasSpecValues = Boolean(
    spec &&
      Object.entries(spec).some(([key, value]) => key !== "raw_query" && value !== null && value !== undefined && value !== ""),
  );
  if (!hasSpecValues) {
    return null;
  }

  return (
    <div className="assistant-card">
      <p className="assistant-card-title">Parameters To Confirm</p>
      {renderSpecList(spec, "Resolved Parameters")}
      {response.missing_fields.length > 0 ? (
        <p style={{ marginTop: 10 }}>Still missing: {response.missing_fields.join(", ")}</p>
      ) : null}
    </div>
  );
}

function renderAssistantCard(
  response: AgentTurnResponse | undefined,
  options: {
    launchable: boolean;
    isLaunching: boolean;
    runId: string | null;
    runStatus: string | null;
    agentMode: AgentMode;
    onStartRun: () => void;
  },
) {
  if (!response) {
    return null;
  }

  if (response.status === "needs_clarification") {
    const plannerSpecReview = renderPlannerSpecReview(response);
    if (plannerSpecReview) {
      return plannerSpecReview;
    }
    const plannerPlan = renderPlannerPlan(response);
    if (plannerPlan) {
      return plannerPlan;
    }
    if (options.agentMode === "planner") {
      const plannerSpecFallback = renderPlannerSpecFallback(response);
      if (plannerSpecFallback) {
        return plannerSpecFallback;
      }
    }
    return (
      <div className="assistant-card">
        <p className="assistant-card-title">Clarification Needed</p>
        <p>
          {response.missing_fields.length > 0
            ? `Still missing: ${response.missing_fields.join(", ")}`
            : "The spec is almost ready, but the agent wants confirmation before moving on."}
        </p>
      </div>
    );
  }

  if (response.status === "running") {
    return (
      <div className="assistant-card">
        <p className="assistant-card-title">Ready To Launch</p>
        <p>
          {options.agentMode === "planner"
            ? "The planner has an approved workflow ready. Start the run to execute the full tool sequence."
            : "The simulation spec is complete. Start the background run directly from this message when you’re ready."}
        </p>
        <div className="action-row assistant-card-actions">
          <button
            className="primary-button"
            disabled={!options.launchable || options.isLaunching || options.runId !== null}
            onClick={options.onStartRun}
          >
            {options.isLaunching ? "Launching..." : options.runId ? `Run ${options.runStatus ?? "active"}` : "Start Run"}
          </button>
        </div>
      </div>
    );
  }

  if (response.status === "completed") {
    return (
      <>
        <div className="assistant-card">
          <p className="assistant-card-title">Completed</p>
          <p>The simulation returned a result payload directly.</p>
        </div>
        {renderPlannerExecutionTrace(response)}
      </>
    );
  }

  return null;
}

export function ChatPanel({
  messages,
  draft,
  isBusy,
  isLaunching,
  launchable,
  runId,
  runStatus,
  agentMode,
  onDraftChange,
  onSubmit,
  onStartRun,
  onAbortRun,
  hideHeader = false,
  className = "",
}: ChatPanelProps) {
  const canAbort =
    runId !== null &&
    runStatus !== "completed" &&
    runStatus !== "failed" &&
    runStatus !== "aborted";

  return (
    <section className={`chat-shell ${className}`.trim()}>
      {!hideHeader ? (
        <header className="shell-header">
          <h2 className="shell-title">EQuantum Assistant</h2>
          <p className="shell-subtitle">
            {agentMode === "planner"
              ? "Ask for a multi-step experiment and let the planner use simulation tools directly."
              : "Describe a DOS or LDOS run in natural language, then confirm defaults and launch it from the workbench."}
          </p>
        </header>
      ) : null}

      <div className="chat-layout">
        <div className="messages">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Start a simulation conversation</h2>
              <p>
                {agentMode === "planner" ? (
                  <>
                    Try: <strong>compare the LDOS for a square lattice under B=1 T and backgate 0.2 V with and without self-consistency</strong>
                  </>
                ) : (
                  <>
                    Try: <strong>calculate the density of states for a square lattice system with backgate voltage 0.5 and magnetic field 1 T</strong>
                  </>
                )}
              </p>
            </div>
          ) : (
            messages.map((message) => (
              <article key={message.id} className={`message ${message.role}`}>
                <div className="message-meta">{message.role === "user" ? "You" : "Assistant"}</div>
                <div style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", wordBreak: "break-word" }}>{message.text}</div>
                {message.role === "assistant"
                  ? renderAssistantCard(message.response, {
                      launchable,
                      isLaunching,
                      runId,
                      runStatus,
                      agentMode,
                      onStartRun,
                    })
                  : null}
              </article>
            ))
          )}
        </div>

        <div className="composer">
          <div className="composer-box">
            <textarea
              value={draft}
              onChange={(event) => onDraftChange(event.target.value)}
              disabled={canAbort}
              placeholder={
                canAbort
                  ? "A run is active. Use Abort Run below if you want to stop it."
                  : agentMode === "planner"
                    ? "Ask for a simulation plan or comparison workflow."
                    : "Ask for a simulation, answer a clarification, or say 'use defaults' / 'start'."
              }
            />
            <div className="composer-actions">
              <span className="helper-text">
                {canAbort
                  ? "The run is active. The main action here switches to abort so you can stop it quickly."
                  : isBusy
                    ? agentMode === "planner"
                      ? "Planner mode is running a server-side tool loop. The completed tool trace will appear in the inspector."
                      : "Working through the current turn..."
                    : agentMode === "planner"
                      ? "Planner mode executes tool calls on the server and returns a final summary."
                      : "Multi-turn session state stays in the browser for now."}
              </span>
              {canAbort ? (
                <button className="danger-button" onClick={onAbortRun}>
                  Abort Run
                </button>
              ) : (
                <button className="primary-button" disabled={isBusy || draft.trim().length === 0} onClick={onSubmit}>
                  Send
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
