export type AgentStatus = "needs_clarification" | "running" | "completed";

export type SimulationSpec = {
  raw_query?: string;
  task: "dos" | "ldos" | null;
  profile: string;
  lattice_type: "square" | "honeycomb" | null;
  device_shape: string | null;
  backgate_voltage: number | null;
  magnetic_field_T: number | null;
  solve_self_consistent: boolean;
  spacing0: number | null;
  density_k: number | null;
  dielectric_constant: number | null;
  gate_potential: number | null;
  convergence_tol: number[] | null;
  Ncore: number | null;
  eta: number | null;
  ldos_method: string | null;
};

export type AgentTurnResponse = {
  status: AgentStatus;
  message: string;
  spec: SimulationSpec;
  missing_fields: string[];
  session_state: Record<string, unknown>;
  result?: Record<string, unknown>;
};

export type AgentTurnRequest = {
  message: string;
  session_state?: Record<string, unknown> | null;
  execute?: boolean;
  parser?: string;
  profile?: string;
  device_shape?: string;
};

export type CreateRunRequest = {
  spec: SimulationSpec;
  parser?: string;
  profile?: string;
  device_shape?: string;
  require_manual_check?: boolean;
};

export type CreateRunResponse = {
  run_id: string;
  status: string;
};

export type RunEvent = {
  type: "status" | "log" | "manual_check" | "result" | "error";
  message?: string | null;
  payload?: Record<string, unknown> | null;
  timestamp: string;
};

export type RunStateResponse = {
  run_id: string;
  status: string;
  spec: SimulationSpec;
  result: Record<string, unknown> | null;
  error: string | null;
  logs: string[];
  events: RunEvent[];
  manual_check_pending: boolean;
  manual_check_payload: Record<string, unknown> | null;
};

const API_PREFIX = "/api";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_PREFIX}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }

  return (await response.json()) as T;
}

export function sendAgentTurn(payload: AgentTurnRequest): Promise<AgentTurnResponse> {
  return requestJson<AgentTurnResponse>("/agent/turn", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function createRun(payload: CreateRunRequest): Promise<CreateRunResponse> {
  return requestJson<CreateRunResponse>("/runs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function fetchRun(runId: string): Promise<RunStateResponse> {
  return requestJson<RunStateResponse>(`/runs/${runId}`);
}

export function approveManualCheck(runId: string, approved: boolean): Promise<{ run_id: string; approved: boolean }> {
  return requestJson(`/runs/${runId}/manual-check`, {
    method: "POST",
    body: JSON.stringify({ approved }),
  });
}
