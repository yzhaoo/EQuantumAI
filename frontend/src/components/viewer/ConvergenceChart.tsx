import React, { useMemo, useEffect, useRef } from "react";
import { type RunEvent, type FscIterationPayload } from "../../api";

interface ConvergenceChartProps {
  events: RunEvent[];
}

function formatScientific(val: number | null | undefined): string {
  if (val === null || val === undefined) return "-";
  return val.toExponential(3);
}

function formatFloat(val: number | null | undefined): string {
  if (val === null || val === undefined) return "-";
  return val.toFixed(3);
}

export function ConvergenceChart({ events }: ConvergenceChartProps) {
  const iterationEvents = useMemo(() => {
    return events.filter((e) => e.type === "fsc_iteration") as Array<RunEvent & { payload: FscIterationPayload }>;
  }, [events]);

  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [iterationEvents]);

  if (iterationEvents.length === 0) {
    return (
      <div
        className="viewer-empty"
        style={{
          flex: 1,
          backgroundColor: "#1e1e1e",
          color: "#94a3b8",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "monospace",
        }}
      >
        Iteration summary will appear here once the solver starts.
      </div>
    );
  }

  return (
    <div
      className="terminal-panel"
      ref={containerRef}
      style={{
        flex: 1,
        overflowY: "auto",
        fontFamily: "monospace",
        whiteSpace: "pre-wrap",
        display: "flex",
        flexDirection: "column",
        gap: "1rem",
        padding: "1rem",
        backgroundColor: "#1e1e1e",
      }}
    >
      {iterationEvents.map((e, idx) => {
        const p = e.payload;
        return (
          <div key={idx} style={{ color: "#e2e8f0" }}>
            <div style={{ fontWeight: "bold" }}>--- Iteration {p.iteration} ---</div>
            <div>Qprime size   : {p.qprime_size}</div>
            {p.max_dn !== null && <div>max |Δn|      : {formatScientific(p.max_dn)}</div>}
            {p.dn_per_site !== null && <div>|Δn| per site  : {formatScientific(p.dn_per_site)}</div>}
            {p.dn_per_site_pct !== null && <div>|Δn| per site % : {formatScientific(p.dn_per_site_pct)}</div>}
            {p.max_dildos !== null && <div>max |ΔILDOS|  : {formatScientific(p.max_dildos)}</div>}
            {p.dildos_per_site !== null && <div>|ΔILDOS| per site  : {formatScientific(p.dildos_per_site)}</div>}
            {p.dildos_per_site_pct !== null && <div>|ΔILDOS| per site % : {formatScientific(p.dildos_per_site_pct)}</div>}
            {p.time_poisson !== null && <div>last Poisson  : {formatFloat(p.time_poisson)} s</div>}
            {p.time_quantum !== null && <div>last Quantum  : {formatFloat(p.time_quantum)} s</div>}
          </div>
        );
      })}
    </div>
  );
}
