import { useEffect, useRef, useState } from "react";
import {
  approveManualCheck,
  createRun,
  fetchRun,
  sendAgentTurn,
  type AgentTurnResponse,
  type RunStateResponse,
  type SimulationSpec,
} from "./api";
import { ChatPanel, type ChatMessage } from "./components/ChatPanel";
import { ResultPanel } from "./components/ResultPanel";
import { SpecPanel } from "./components/SpecPanel";

function buildMessage(role: ChatMessage["role"], text: string, response?: AgentTurnResponse): ChatMessage {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role,
    text,
    response,
  };
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
  const pollTimer = useRef<number | null>(null);

  const launchable = lastAgentResponse?.status === "running" && currentSpec !== null;

  useEffect(() => {
    if (!runId) {
      return;
    }

    async function loadRun() {
      try {
        const state = await fetchRun(runId);
        setRunState(state);
        if (state.status === "completed" || state.status === "failed") {
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
        parser: "openai",
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

    try {
      const response = await createRun({
        spec: currentSpec,
        parser: "openai",
      });
      setRunId(response.run_id);
      setRunState(null);
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

  const statusLabel = runState?.status ?? lastAgentResponse?.status ?? "idle";

  return (
    <div className="app-shell">
      <div className="workbench">
        <aside className="sidebar glass-panel">
          <p className="eyebrow">Minimal Prototype</p>
          <h1>
            EQuantumAI
            <br />
            Workbench
          </h1>

          <div className="sidebar-card">
            <p>
              This prototype mirrors the desktop flow in the browser: chat with the agent, inspect the parsed spec, then launch a background run.
            </p>
          </div>

          <div className="sidebar-card">
            <p>Suggested prompts</p>
            <ul>
              <li>calculate the density of states for a square lattice system with backgate voltage 0.5 and magnetic field 1 T</li>
              <li>use defaults</li>
              <li>start</li>
            </ul>
          </div>

          <div className="sidebar-card">
            <p>
              Current mode: <strong>{statusLabel}</strong>
            </p>
            {runId ? <p style={{ marginTop: 10 }}>Run id: {runId}</p> : null}
            {errorMessage ? <p style={{ marginTop: 10, color: "var(--danger)" }}>{errorMessage}</p> : null}
          </div>
        </aside>

        <ChatPanel
          messages={messages}
          draft={draft}
          isBusy={isSending}
          onDraftChange={setDraft}
          onSubmit={handleSubmit}
        />

        <section className="right-column glass-panel">
          <div className="status-strip">
            <span className={`status-pill ${runState?.status && runState.status !== "failed" ? "live" : ""}`}>
              Agent: {lastAgentResponse?.status ?? "idle"}
            </span>
            <span className={`status-pill ${runState?.status === "failed" ? "error" : runId ? "live" : ""}`}>
              Run: {runState?.status ?? (runId ? "queued" : "not started")}
            </span>
            <span className={`status-pill ${runState?.manual_check_pending ? "live" : ""}`}>
              Manual check: {runState?.manual_check_pending ? "pending" : "clear"}
            </span>
          </div>

          <div className="panel-stack">
            <SpecPanel spec={currentSpec} sessionState={sessionState} />
            <ResultPanel
              pendingSpec={currentSpec}
              runState={runState}
              runId={runId}
              isLaunching={isLaunching}
              launchable={launchable}
              onStartRun={handleStartRun}
              onApproveManualCheck={handleManualCheck}
            />
          </div>
        </section>
      </div>
    </div>
  );
}
