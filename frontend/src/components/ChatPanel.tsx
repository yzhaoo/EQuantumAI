import type { AgentTurnResponse } from "../api";

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
  onDraftChange: (value: string) => void;
  onSubmit: () => void;
  onStartRun: () => void;
  onAbortRun: () => void;
  hideHeader?: boolean;
  className?: string;
};

function renderAssistantCard(
  response: AgentTurnResponse | undefined,
  options: {
    launchable: boolean;
    isLaunching: boolean;
    runId: string | null;
    runStatus: string | null;
    onStartRun: () => void;
  },
) {
  if (!response) {
    return null;
  }

  if (response.status === "needs_clarification") {
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
        <p>The simulation spec is complete. Start the background run directly from this message when you’re ready.</p>
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
      <div className="assistant-card">
        <p className="assistant-card-title">Completed</p>
        <p>The simulation returned a result payload directly.</p>
      </div>
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
            Describe a DOS or LDOS run in natural language, then confirm defaults and launch it from the workbench.
          </p>
        </header>
      ) : null}

      <div className="chat-layout">
        <div className="messages">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Start a simulation conversation</h2>
              <p>
                Try: <strong>calculate the density of states for a square lattice system with backgate voltage 0.5 and magnetic field 1 T</strong>
              </p>
            </div>
          ) : (
            messages.map((message) => (
              <article key={message.id} className={`message ${message.role}`}>
                <div className="message-meta">{message.role === "user" ? "You" : "Assistant"}</div>
                <div>{message.text}</div>
                {message.role === "assistant"
                  ? renderAssistantCard(message.response, {
                      launchable,
                      isLaunching,
                      runId,
                      runStatus,
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
                  : "Ask for a simulation, answer a clarification, or say 'use defaults' / 'start'."
              }
            />
            <div className="composer-actions">
              <span className="helper-text">
                {canAbort
                  ? "The run is active. The main action here switches to abort so you can stop it quickly."
                  : isBusy
                    ? "Working through the current turn..."
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
