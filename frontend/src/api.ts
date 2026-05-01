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
  abort_requested?: boolean;
};

export type ViewerSnapshotsResponse = {
  snapshots: string[];
  has_static: boolean;
};

export type ViewerQuantumHeatmapResponse = {
  snapshot: string;
  property: string;
  site_ids: number[];
  x: Array<number | null>;
  y: Array<number | null>;
  values: Array<number | null>;
  qprime_mask: boolean[];
  color_min: number;
  color_max: number;
};

export type ViewerSiteLdosResponse = {
  site_id: number;
  snapshot: string;
  energy: Array<number | null>;
  ldos: Array<number | null>;
  Ui: number | null;
  ni: number | null;
  ldos_at_0: number | null;
  ldos_at_Ui: number | null;
};

export type ViewerSurfaceCutPayload = {
  snapshot: string;
  property: string;
  p0: [number, number];
  p1: [number, number];
  cut_width: number;
};

export type ViewerSurfaceCutResponse = {
  snapshot: string;
  property: string;
  p0: Array<number | null>;
  p1: Array<number | null>;
  cut_width: number;
  distance_along: Array<number | null>;
  z: Array<number | null>;
  values: Array<number | null>;
};

export type ViewerOverlayCurve = {
  snapshot: string;
  distance_along: Array<number | null>;
  Ui: Array<number | null>;
  is_current: boolean;
};

export type ViewerLdosCutPayload = {
  snapshot: string;
  p0: [number, number];
  p1: [number, number];
  cut_width: number;
  overlay_snapshots: string[];
};

export type ViewerLdosCutResponse = {
  snapshot: string;
  p0: Array<number | null>;
  p1: Array<number | null>;
  cut_width: number;
  distance_along: Array<number | null>;
  energy: Array<number | null>;
  ldos_matrix: Array<Array<number | null>>;
  overlays: ViewerOverlayCurve[];
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

export function abortRun(runId: string): Promise<{ run_id: string; status: string }> {
  return requestJson(`/runs/${runId}/abort`, {
    method: "POST",
  });
}

export function fetchSnapshots(runId: string): Promise<ViewerSnapshotsResponse> {
  return requestJson(`/runs/${runId}/viewer/snapshots`);
}

export function fetchQuantumHeatmap(
  runId: string,
  snapshot: string,
  property: string,
): Promise<ViewerQuantumHeatmapResponse> {
  const search = new URLSearchParams({ snapshot, property });
  return requestJson(`/runs/${runId}/viewer/quantum-heatmap?${search.toString()}`);
}

export function fetchSiteLdos(
  runId: string,
  snapshot: string,
  siteId: number,
): Promise<ViewerSiteLdosResponse> {
  const search = new URLSearchParams({ snapshot, site_id: String(siteId) });
  return requestJson(`/runs/${runId}/viewer/site-ldos?${search.toString()}`);
}

export function fetchSurfaceCut(
  runId: string,
  payload: ViewerSurfaceCutPayload,
): Promise<ViewerSurfaceCutResponse> {
  return requestJson(`/runs/${runId}/viewer/surface-cut`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function fetchLdosCut(
  runId: string,
  payload: ViewerLdosCutPayload,
): Promise<ViewerLdosCutResponse> {
  return requestJson(`/runs/${runId}/viewer/ldos-cut`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
