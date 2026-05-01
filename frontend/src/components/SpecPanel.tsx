import type { SimulationSpec } from "../api";

type SpecPanelProps = {
  spec: SimulationSpec | null;
  sessionState: Record<string, unknown> | null;
};

export function SpecPanel({ spec, sessionState }: SpecPanelProps) {
  const payload = {
    spec,
    session_state: sessionState,
  };

  return (
    <section className="panel-card">
      <header className="shell-header">
        <h3 className="shell-title">Spec Inspector</h3>
        <p className="shell-subtitle">Parsed spec plus the browser-held session state.</p>
      </header>
      <div className="panel-body">
        <pre>{JSON.stringify(payload, null, 2)}</pre>
      </div>
    </section>
  );
}
