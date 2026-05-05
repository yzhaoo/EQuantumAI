export type AgentStatus = "needs_clarification" | "running" | "completed";
export type AgentMode = "parser" | "planner";
export type PlannerModel = "gpt-4o-mini" | "gpt-4.1" | "gpt-5";

export type SimulationSpec = {
  raw_query?: string;
  task: "dos" | "ldos" | null;
  profile: string;
  lattice_type: "square" | "honeycomb" | null;
  device_shape: string | null;
  backgate_voltage: number | null;
  magnetic_field_T: number | null;
  solve_self_consistent: boolean | null;
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
  openai_model?: string;
  profile?: string;
  device_shape?: string;
};

export type PlannerTurnRequest = {
  message: string;
  session_state?: Record<string, unknown> | null;
  parser?: string;
  openai_model?: string;
  profile?: string;
  device_shape?: string;
  max_iterations?: number;
};

export type CreateRunRequest = {
  spec: SimulationSpec;
  parser?: string;
  openai_model?: string;
  profile?: string;
  device_shape?: string;
  require_manual_check?: boolean;
};

export type CreateRunResponse = {
  run_id: string;
  status: string;
};

export type PlannerExecuteRequest = {
  approved_plan: Array<Record<string, unknown>>;
  original_request: string;
  parser?: string;
  openai_model?: string;
  profile?: string;
  device_shape?: string;
};

export type FscIterationPayload = {
  iteration: number;
  qprime_size: number;
  max_dn: number | null;
  dn_per_site: number | null;
  dn_per_site_pct: number | null;
  max_dildos: number | null;
  dildos_per_site: number | null;
  dildos_per_site_pct: number | null;
  time_poisson: number | null;
  time_quantum: number | null;
};

export type RunEvent = {
  type: "status" | "log" | "manual_check" | "result" | "error" | "fsc_iteration";
  message?: string | null;
  payload?: Record<string, unknown> | FscIterationPayload | null;
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

export type HistoryRunItem = {
  run_path: string;
  run_name: string;
  artifact_dir: string;
  profile: string | null;
  task: string | null;
  status: string;
  has_static: boolean;
  snapshot_count: number;
  created_at: string | null;
  spec: Record<string, unknown> | null;
  result: Record<string, unknown> | null;
};

export type ViewerSnapshotsResponse = {
  snapshots: string[];
  has_static: boolean;
};

export type ViewerSetupGeometryResponse = {
  site_ids: number[];
  coordinates: Array<Array<number | null>>;
  materials: string[];
  qsite_ids: number[];
  bounds_min: Array<number | null>;
  bounds_max: Array<number | null>;
  geometry_params: Record<string, unknown>;
  gate_info: Record<string, unknown>;
};

export type ViewerSetupFieldResponse = {
  snapshot: string;
  property: string;
  site_ids: number[];
  values: Array<number | null>;
  color_min: number;
  color_max: number;
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
  Ci: number | null;
  ldos_at_0: number | null;
  ldos_at_Ui: number | null;
  consistency_delta_u: Array<number | null>;
  consistency_poisson: Array<number | null>;
  consistency_integrated: Array<number | null>;
  dU_solution: number | null;
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
    if (text) {
      try {
        const parsed = JSON.parse(text) as { detail?: unknown };
        if (typeof parsed.detail === "string" && parsed.detail.trim()) {
          throw new Error(parsed.detail);
        }
      } catch {
        throw new Error(text);
      }
    }
    throw new Error(`Request failed with ${response.status}`);
  }

  return (await response.json()) as T;
}

export function sendAgentTurn(payload: AgentTurnRequest): Promise<AgentTurnResponse> {
  return requestJson<AgentTurnResponse>("/agent/turn", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function sendPlannerTurn(payload: PlannerTurnRequest): Promise<AgentTurnResponse> {
  return requestJson<AgentTurnResponse>("/agent/plan", {
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

export function createPlannerRun(payload: PlannerExecuteRequest): Promise<CreateRunResponse> {
  return requestJson<CreateRunResponse>("/planner/runs", {
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

export function fetchHistoryRuns(): Promise<HistoryRunItem[]> {
  return requestJson("/history/runs");
}

export function fetchHistoryRun(runPath: string): Promise<HistoryRunItem> {
  return requestJson(`/history/runs/${encodeURIComponent(runPath)}`);
}

export function fetchHistorySnapshots(runPath: string): Promise<ViewerSnapshotsResponse> {
  return requestJson(`/history/runs/${encodeURIComponent(runPath)}/viewer/snapshots`);
}

export function fetchSetupGeometry(runId: string): Promise<ViewerSetupGeometryResponse> {
  return requestJson(`/runs/${runId}/viewer/setup-geometry`);
}

export function fetchHistorySetupGeometry(runPath: string): Promise<ViewerSetupGeometryResponse> {
  return requestJson(`/history/runs/${encodeURIComponent(runPath)}/viewer/setup-geometry`);
}

export function fetchSetupField(
  runId: string,
  snapshot: string,
  property: string,
): Promise<ViewerSetupFieldResponse> {
  const search = new URLSearchParams({ snapshot, property });
  return requestJson(`/runs/${runId}/viewer/setup-field?${search.toString()}`);
}

export function fetchHistorySetupField(
  runPath: string,
  snapshot: string,
  property: string,
): Promise<ViewerSetupFieldResponse> {
  const search = new URLSearchParams({ snapshot, property });
  return requestJson(`/history/runs/${encodeURIComponent(runPath)}/viewer/setup-field?${search.toString()}`);
}

export function fetchQuantumHeatmap(
  runId: string,
  snapshot: string,
  property: string,
): Promise<ViewerQuantumHeatmapResponse> {
  const search = new URLSearchParams({ snapshot, property });
  return requestJson(`/runs/${runId}/viewer/quantum-heatmap?${search.toString()}`);
}

export function fetchHistoryQuantumHeatmap(
  runPath: string,
  snapshot: string,
  property: string,
): Promise<ViewerQuantumHeatmapResponse> {
  const search = new URLSearchParams({ snapshot, property });
  return requestJson(`/history/runs/${encodeURIComponent(runPath)}/viewer/quantum-heatmap?${search.toString()}`);
}

export function fetchSiteLdos(
  runId: string,
  snapshot: string,
  siteId: number,
): Promise<ViewerSiteLdosResponse> {
  const search = new URLSearchParams({ snapshot, site_id: String(siteId) });
  return requestJson(`/runs/${runId}/viewer/site-ldos?${search.toString()}`);
}

export function fetchHistorySiteLdos(
  runPath: string,
  snapshot: string,
  siteId: number,
): Promise<ViewerSiteLdosResponse> {
  const search = new URLSearchParams({ snapshot, site_id: String(siteId) });
  return requestJson(`/history/runs/${encodeURIComponent(runPath)}/viewer/site-ldos?${search.toString()}`);
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

export function fetchHistorySurfaceCut(
  runPath: string,
  payload: ViewerSurfaceCutPayload,
): Promise<ViewerSurfaceCutResponse> {
  return requestJson(`/history/runs/${encodeURIComponent(runPath)}/viewer/surface-cut`, {
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

export function fetchHistoryLdosCut(
  runPath: string,
  payload: ViewerLdosCutPayload,
): Promise<ViewerLdosCutResponse> {
  return requestJson(`/history/runs/${encodeURIComponent(runPath)}/viewer/ldos-cut`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
