import {
  SHOWCASE_API_VERSION,
  guardTraceVersion,
  type RuntimeTraceV1,
  type ShowcaseErrorResponse,
  type TraceEnvelope,
} from "../types/runtimeTrace";

export interface HealthResponse {
  schema_version?: string;
  api_version?: string;
  status?: string;
  service?: string;
  api_ready?: boolean;
  backend?: string;
  backend_reachable?: boolean;
  knowledge_base_available?: boolean;
  embedding_available?: boolean;
  reranker_available?: boolean;
  generation_available?: boolean;
  registry?: Record<string, unknown>;
  concurrency?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ScenarioRecord {
  scenario_id: "S01" | "S02" | "S03" | "S04" | string;
  name?: string;
  title?: string;
  query?: string;
  command?: "search" | "ask" | string;
  description?: string;
  [key: string]: unknown;
}

export interface ScenarioListResponse {
  schema_version?: string;
  scenarios: ScenarioRecord[];
  [key: string]: unknown;
}

export interface RuntimeAuthorityResponse {
  schema_version?: string;
  runtime_authority?: string;
  showcase_api_version?: string;
  runtime_trace_schema_version?: string;
  runtime_trace_event_schema_version?: string;
  vector_backend?: string;
  embedding_model?: string;
  reranker_model?: string;
  search_precision?: string;
  ask_precision?: string;
  initial_retrieval_policy?: string;
  graph_hop_depth?: number;
  maximum_recovery_attempt_count?: number;
  planner_enabled?: boolean;
  unbounded_agent_loop_enabled?: boolean;
  [key: string]: unknown;
}

export interface QueryCitation {
  citation_id: string;
  chunk_id?: string;
  document_id?: string;
  relative_path: string;
  heading_path?: string[];
  start_line?: number | null;
  end_line?: number | null;
  snippet?: string;
  source_status?: string | null;
}

export interface QueryAnswerResult {
  status: "answered" | "refused" | string;
  answerable: boolean;
  answer: string;
  refusal_reason_code?: string | null;
  generation_latency_ms?: number | null;
  grounding_valid?: boolean | null;
  citations: QueryCitation[];
}

export type AskTraceEnvelope = TraceEnvelope & { result?: QueryAnswerResult | null };

export interface ConversationSessionSummary {
  id: string;
  knowledge_base_id: string;
  title?: string | null;
  status: string;
  created_at?: string | null;
  updated_at?: string | null;
  last_turn_at?: string | null;
  turn_count: number;
}

export interface ConversationTurnRecord {
  id: string;
  session_id: string;
  turn_number: number;
  client_request_id?: string | null;
  user_query: string;
  standalone_query?: string | null;
  is_followup?: boolean;
  rewrite_status?: string;
  answer_decision?: string | null;
  answer_text?: string | null;
  abstention_reason?: string | null;
  turn_status: string;
  retrieval_mode?: string | null;
  reranking_enabled?: boolean | null;
  provider_id?: string | null;
  model_id?: string | null;
  latency_ms?: number | null;
  error_code?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
  trace_id?: string | null;
  citations: QueryCitation[];
}

export interface ConversationSessionListResponse {
  schema_version: string;
  sessions: ConversationSessionSummary[];
}

export interface ConversationSessionResponse {
  schema_version: string;
  session: ConversationSessionSummary;
  turns?: ConversationTurnRecord[];
}

export interface ConversationAskResponse {
  schema_version: string;
  session: ConversationSessionSummary;
  turn: ConversationTurnRecord;
  resolution: {
    standalone_query: string;
    is_followup: boolean;
    referenced_turn_numbers: number[];
    referenced_citation_ids: string[];
    model_id?: string;
    prompt_version?: string;
  };
  answer?: QueryAnswerResult | null;
  idempotent_replay: boolean;
  trace?: TraceEnvelope | null;
}

export interface ClientResult<T> {
  ok: boolean;
  data?: T;
  error?: ShowcaseErrorResponse | { code: string; message: string; retryable?: boolean };
}

export const DEFAULT_API_BASE_URL = "http://127.0.0.1:8766/api/showcase/v1";

export function getApiBaseUrl(): string {
  return (import.meta.env.VITE_SHOWCASE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/+$/, "");
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<ClientResult<T>> {
  const url = `${getApiBaseUrl()}${path}`;
  try {
    const response = await fetch(url, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers || {}),
      },
    });
    const body = (await response.json().catch(() => null)) as T | ShowcaseErrorResponse | null;
    if (!response.ok) {
      return { ok: false, error: (body as ShowcaseErrorResponse) || { code: "api_error", message: response.statusText } };
    }
    return { ok: true, data: body as T };
  } catch (error) {
    return {
      ok: false,
      error: {
        code: "api_unavailable",
        message: error instanceof Error ? error.message : "Showcase API is unavailable",
        retryable: true,
      },
    };
  }
}

function validateEnvelope<T extends TraceEnvelope>(envelope: T): ClientResult<T> {
  if (envelope.schema_version !== SHOWCASE_API_VERSION) {
    return { ok: false, error: { code: "schema_incompatible", message: "Showcase API version mismatch" } };
  }
  const guard = guardTraceVersion(envelope.trace);
  if (!guard.ok) {
    return { ok: false, error: { code: "schema_incompatible", message: guard.reason || "Runtime Trace V1 mismatch" } };
  }
  return { ok: true, data: envelope };
}

export class ShowcaseApiClient {
  async getHealth(): Promise<ClientResult<HealthResponse>> {
    return requestJson<HealthResponse>("/health");
  }

  async getRuntime(): Promise<ClientResult<RuntimeAuthorityResponse>> {
    return requestJson<RuntimeAuthorityResponse>("/runtime");
  }

  async getScenarios(): Promise<ClientResult<ScenarioListResponse>> {
    return requestJson<ScenarioListResponse>("/scenarios");
  }

  async runSearch(query: string, topK?: number | null): Promise<ClientResult<TraceEnvelope>> {
    const result = await requestJson<TraceEnvelope>("/search", {
      method: "POST",
      body: JSON.stringify({ query, top_k: topK ?? null }),
    });
    return result.ok && result.data ? validateEnvelope(result.data) : result;
  }

  async runAsk(query: string, topK?: number | null): Promise<ClientResult<AskTraceEnvelope>> {
    const result = await requestJson<AskTraceEnvelope>("/ask", {
      method: "POST",
      body: JSON.stringify({ query, top_k: topK ?? null }),
    });
    return result.ok && result.data ? validateEnvelope(result.data) : result;
  }

  async listSessions(): Promise<ClientResult<ConversationSessionListResponse>> {
    return requestJson<ConversationSessionListResponse>("/sessions");
  }

  async createSession(title?: string | null): Promise<ClientResult<ConversationSessionResponse>> {
    return requestJson<ConversationSessionResponse>("/sessions", {
      method: "POST",
      body: JSON.stringify({ title: title?.trim() || null }),
    });
  }

  async getSession(sessionId: string): Promise<ClientResult<ConversationSessionResponse>> {
    return requestJson<ConversationSessionResponse>(`/sessions/${encodeURIComponent(sessionId)}`);
  }

  async getSessionTurns(sessionId: string): Promise<ClientResult<ConversationSessionResponse>> {
    return requestJson<ConversationSessionResponse>(`/sessions/${encodeURIComponent(sessionId)}/turns`);
  }

  async askSession(sessionId: string, query: string, clientRequestId?: string | null, topK?: number | null): Promise<ClientResult<ConversationAskResponse>> {
    const result = await requestJson<ConversationAskResponse>(`/sessions/${encodeURIComponent(sessionId)}/ask`, {
      method: "POST",
      body: JSON.stringify({ query, client_request_id: clientRequestId ?? null, top_k: topK ?? null }),
    });
    if (result.ok && result.data?.trace) {
      const checked = validateEnvelope(result.data.trace);
      if (!checked.ok) return { ok: false, error: checked.error };
    }
    return result;
  }

  async runScenario(scenarioId: string): Promise<ClientResult<TraceEnvelope>> {
    const result = await requestJson<TraceEnvelope>(`/scenarios/${encodeURIComponent(scenarioId)}/run`, { method: "POST" });
    return result.ok && result.data ? validateEnvelope(result.data) : result;
  }

  async getTrace(traceId: string): Promise<ClientResult<TraceEnvelope>> {
    const result = await requestJson<TraceEnvelope>(`/traces/${encodeURIComponent(traceId)}`);
    return result.ok && result.data ? validateEnvelope(result.data) : result;
  }

  eventUrl(traceId: string, afterSequence = 0): string {
    const query = afterSequence > 0 ? `?after_sequence=${afterSequence}` : "";
    return `${getApiBaseUrl()}/traces/${encodeURIComponent(traceId)}/events${query}`;
  }
}

export const showcaseApi = new ShowcaseApiClient();

export type { RuntimeTraceV1 };
