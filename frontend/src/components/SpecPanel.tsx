import type { SimulationSpec } from "../api";

type SpecPanelProps = {
  spec: SimulationSpec | null;
  sessionState: Record<string, unknown> | null;
};

export function SpecPanel({ spec, sessionState }: SpecPanelProps) {
  const activeStep =
    typeof sessionState?.pending_action === "string"
      ? sessionState.pending_action
      : typeof sessionState?.status === "string"
        ? sessionState.status
        : "idle";

  const missingFields = Array.isArray(sessionState?.missing_fields)
    ? sessionState.missing_fields.map((item) => String(item))
    : [];

  const lastParserDebug =
    sessionState && typeof sessionState === "object" && sessionState.last_parser_debug && typeof sessionState.last_parser_debug === "object"
      ? (sessionState.last_parser_debug as Record<string, unknown>)
      : null;

  const parserLabel =
    lastParserDebug && typeof lastParserDebug.parser === "string" ? lastParserDebug.parser : "unknown";

  const parserIntent =
    lastParserDebug && typeof lastParserDebug.intent === "string" ? lastParserDebug.intent : "unknown";

  const rawPayload = {
    spec,
    session_state: sessionState,
  };

  return (
    <section className="panel-card">
      <header className="shell-header">
        <h3 className="shell-title">Spec Inspector</h3>
        <p className="shell-subtitle">Compact run summary with optional raw state when you need it.</p>
      </header>
      <div className="panel-body">
        <div className="spec-summary">
          <div className="spec-chip-row">
            <span className="spec-chip">parser: {parserLabel}</span>
            <span className="spec-chip">intent: {parserIntent}</span>
            <span className="spec-chip">step: {activeStep}</span>
            <span className="spec-chip">task: {spec?.task ?? "-"}</span>
            <span className="spec-chip">profile: {spec?.profile ?? "-"}</span>
            <span className="spec-chip">lattice: {spec?.lattice_type ?? "-"}</span>
          </div>

          <div className="spec-grid">
            <div className="spec-item">
              <span>Device</span>
              <strong>{spec?.device_shape ?? "-"}</strong>
            </div>
            <div className="spec-item">
              <span>Backgate</span>
              <strong>{spec?.backgate_voltage ?? "-"}</strong>
            </div>
            <div className="spec-item">
              <span>Magnetic field</span>
              <strong>{spec?.magnetic_field_T ?? "-"}</strong>
            </div>
            <div className="spec-item">
              <span>Ncore</span>
              <strong>{spec?.Ncore ?? "-"}</strong>
            </div>
          </div>

          {missingFields.length > 0 ? (
            <div className="spec-note">
              Missing: {missingFields.join(", ")}
            </div>
          ) : null}

          <details className="spec-details">
            <summary>Show raw spec and session JSON</summary>
            <pre>{JSON.stringify(rawPayload, null, 2)}</pre>
          </details>
        </div>
      </div>
    </section>
  );
}
