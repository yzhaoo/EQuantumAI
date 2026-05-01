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
  onDraftChange: (value: string) => void;
  onSubmit: () => void;
};

function renderAssistantCard(response?: AgentTurnResponse) {
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
        <p>The simulation spec is complete. You can start the background run from the panel on the right.</p>
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

export function ChatPanel({ messages, draft, isBusy, onDraftChange, onSubmit }: ChatPanelProps) {
  return (
    <section className="chat-shell glass-panel">
      <header className="shell-header">
        <h2 className="shell-title">EQuantum Assistant</h2>
        <p className="shell-subtitle">
          Describe a DOS or LDOS run in natural language, then confirm defaults and launch it from the workbench.
        </p>
      </header>

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
                {message.role === "assistant" ? renderAssistantCard(message.response) : null}
              </article>
            ))
          )}
        </div>

        <div className="composer">
          <div className="composer-box">
            <textarea
              value={draft}
              onChange={(event) => onDraftChange(event.target.value)}
              placeholder="Ask for a simulation, answer a clarification, or say 'use defaults' / 'start'."
            />
            <div className="composer-actions">
              <span className="helper-text">{isBusy ? "Working through the current turn..." : "Multi-turn session state stays in the browser for now."}</span>
              <button className="primary-button" disabled={isBusy || draft.trim().length === 0} onClick={onSubmit}>
                Send
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
