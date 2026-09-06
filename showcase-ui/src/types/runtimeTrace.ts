export const SHOWCASE_API_VERSION = "opk-rag.showcase-api.v1" as const;
export const RUNTIME_TRACE_SCHEMA_VERSION = "opk-rag.runtime-trace.v1" as const;
export const RUNTIME_TRACE_CONTRACT_VERSION = "opk-rag.runtime-trace-contract.v1" as const;
export const RUNTIME_TRACE_EVENT_SCHEMA_VERSION = "opk-rag.runtime-trace-event.v1" as const;

export type ExecutionLifecycle =
  | "idle"
  | "connecting"
  | "executing"
  | "completed"
  | "refused"
  | "failed"
  | "partial";

export type RuntimeTraceStatus = "started" | "running" | "completed" | "failed" | "refused" | "partial";
export type StageState = "not_started" | "active" | "completed" | "skipped" | "not_applicable" | "failed" | "unavailable";
export type ExecutionScope = "search" | "ask" | "demo" | string;
export type TraceStage =
  | "trace"
  | "query"
  | "runtime"
  | "guard"
  | "retrieval"
  | "structure_recovery"
  | "graph_recovery"
  | "rerank"
  | "evidence"
  | "answerability"
  | "generation"
  | "grounding"
  | "citation"
  | "timings"
  | "outcome";

export const TRACE_STAGE_ORDER: TraceStage[] = [
  "trace",
  "query",
  "runtime",
  "guard",
  "retrieval",
  "structure_recovery",
  "graph_recovery",
  "rerank",
  "evidence",
  "answerability",
  "generation",
  "grounding",
  "citation",
  "timings",
  "outcome",
];

export interface StagePayload {
  stage_state?: StageState;
}

export interface TraceMeta {
  trace_id: string;
  trace_schema_version: typeof RUNTIME_TRACE_SCHEMA_VERSION | string;
  trace_contract_version?: typeof RUNTIME_TRACE_CONTRACT_VERSION | string;
  query_execution_identity_origin?: string;
  source_authority?: string;
  source_authority_sha256?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  status: RuntimeTraceStatus | string;
  captured_from_frozen_authority?: boolean;
  trace_complete?: boolean;
  trace_semantic_digest?: string;
  unavailable_fields?: string[];
}

export interface QueryTrace extends StagePayload {
  query_id?: string | null;
  query_text?: string | null;
  normalized_query?: string | null;
  scenario_id?: string | null;
  execution_scope?: ExecutionScope;
  requested_top_k?: number | null;
  candidate_k?: number | null;
}

export interface RuntimeTraceRuntime extends StagePayload {
  vector_backend?: string | null;
  vector_collection?: string | null;
  knowledge_base_id?: string | null;
  embedding_model?: string | null;
  embedding_revision?: string | null;
  embedding_dimension?: number | null;
  reranker_model?: string | null;
  reranker_revision?: string | null;
  reranker_device?: string | null;
  reranker_precision?: string | null;
  initial_retrieval_policy?: string | null;
  graph_runtime_enabled?: boolean | null;
  graph_hop_depth?: number | null;
  maximum_recovery_attempt_count?: number | null;
  generation_provider?: string | null;
  generation_model?: string | null;
  generation_endpoint_type?: string | null;
}

export interface GuardTrace extends StagePayload {
  agent_type?: string | null;
  selected_route?: string | null;
  initial_decision?: string | null;
  final_decision?: string | null;
  guard_triggered?: boolean | null;
  guard_reason_code?: string | null;
  recovery_required?: boolean | null;
  recovery_reason_code?: string | null;
  recovery_action?: string | null;
  recovery_attempt_count?: number | null;
  maximum_recovery_attempt_count?: number | null;
  fail_closed?: boolean | null;
  refusal_reason_code?: string | null;
  hidden_chain_of_thought_exposed?: boolean;
}

export interface CandidateProvenance {
  source_kind?: "initial" | "structure_expansion" | "graph_recovery" | string;
  source_lane?: string;
}

export interface RuntimeCandidate {
  candidate_id?: string | null;
  chunk_id?: string | null;
  document_id?: string | null;
  document_path?: string | null;
  section_path?: string[] | null;
  text_preview?: string | null;
  retrieval_sources?: string[];
  provenance?: CandidateProvenance[];
  retrieval_score?: number | null;
  initial_rank?: number | null;
  final_rank?: number | null;
  rerank_score?: number | null;
  structure_expanded?: boolean;
  graph_recovered?: boolean;
  selected_for_context?: boolean;
}

export interface RetrievalTrace extends StagePayload {
  requested_policy?: string | null;
  executed_policy?: string | null;
  retrieval_mode?: string | null;
  retrievers_used?: string[];
  initial_candidate_count?: number | null;
  vector_candidate_count?: number | null;
  lexical_candidate_count?: number | null;
  post_structure_candidate_count?: number | null;
  post_graph_unified_candidate_count?: number | null;
  rerank_input_count?: number | null;
  final_result_count?: number | null;
  candidate_pool_count_after_recovery?: number | null;
  observed_candidate_count?: number | null;
  observed_candidates_complete?: boolean;
  candidates: RuntimeCandidate[];
}

export interface StructureRecoveryTrace extends StagePayload {
  triggered?: boolean;
  reason_code?: string | null;
  seed_candidate_ids?: string[];
  expanded_candidate_ids?: string[];
  expanded_candidate_count?: number | null;
  expanded_candidate_ids_complete?: boolean;
  source_document_ids?: string[];
  candidate_pool_count_before?: number | null;
  candidate_pool_count_after?: number | null;
}

export interface GraphEdge {
  edge_id?: string | null;
  source_node_id?: string | null;
  target_node_id?: string | null;
  relation_type?: string | null;
  source_document_id?: string | null;
  target_document_id?: string | null;
  hop_depth?: number | null;
  edge_observation?: string | null;
}

export interface GraphCandidateProvenanceTrace {
  candidate_id?: string | null;
  canonical_chunk_id?: string | null;
  document_id?: string | null;
  edge_types?: string[];
  graph_added?: boolean;
  graph_edge_types?: string[];
  graph_expansion_reason?: string | null;
  graph_hop_count?: number | null;
  graph_path?: Array<{
    authority_level?: string | null;
    authority_source_node_id?: string | null;
    authority_target_node_id?: string | null;
    edge_id?: string | null;
    edge_type?: string | null;
    graph_record_class?: string | null;
    source_node_id?: string | null;
    target_node_id?: string | null;
  }>;
  section_id?: string | null;
  seed_candidate_id?: string | null;
  source_seed_candidate_id?: string | null;
}

export interface GraphRecoveryTrace extends StagePayload {
  graph_activated?: boolean;
  activation_reason_code?: string | null;
  activation_policy?: string | null;
  hop_depth?: number | null;
  seed_node_ids?: string[];
  seed_candidate_ids?: string[];
  traversed_edges?: GraphEdge[];
  recovered_node_ids?: string[];
  recovered_candidate_ids?: string[];
  recovered_candidate_count?: number | null;
  candidate_pool_count_before?: number | null;
  candidate_pool_count_after?: number | null;
  candidate_provenance?: GraphCandidateProvenanceTrace[];
  runtime_gold_metadata_usage?: boolean | null;
}

export interface RerankTrace extends StagePayload {
  reranker_model?: string | null;
  reranker_revision?: string | null;
  reranker_device?: string | null;
  execution_scope?: ExecutionScope;
  precision?: string | null;
  input_candidate_count?: number | null;
  output_candidate_count?: number | null;
  ranked_candidates?: Array<{ candidate_id?: string | null; rerank_score?: number | null; rerank_position?: number | null }>;
}

export interface EvidenceItem {
  evidence_id?: string | null;
  source_candidate_id?: string | null;
  chunk_id?: string | null;
  document_id?: string | null;
  document_path?: string | null;
  section_path?: string[] | null;
  start_line?: number | null;
  end_line?: number | null;
  evidence_position?: number | null;
  evidence_score?: number | null;
  selection_reason_code?: string | null;
  source_provenance?: CandidateProvenance[];
  retrieval_sources?: string[];
}

export interface EvidenceTrace extends StagePayload {
  evidence_count?: number | null;
  evidence_items: EvidenceItem[];
  candidate_evidence_semantic_separation?: boolean;
  provenance_available?: boolean;
}

export interface AnswerabilityTrace extends StagePayload {
  answerability_state?: string | null;
  answerable?: boolean | null;
  reason_code?: string | null;
  evidence_count?: number | null;
  required_support_state?: string | null;
  confidence?: number | null;
  considered_evidence_count?: number | null;
  evidence_chunk_ids?: string[];
  note?: string | null;
}

export interface GenerationTrace extends StagePayload {
  generation_attempted?: boolean | null;
  generation_provider?: string | null;
  generation_model?: string | null;
  generation_revision?: string | null;
  generation_endpoint_type?: string | null;
  generation_latency_ms?: number | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  finish_reason?: string | null;
  generation_completed?: boolean | null;
  generation_abstained?: boolean | null;
  evidence_count?: number | null;
}

export interface GroundingTrace extends StagePayload {
  grounding_checked?: boolean | null;
  grounding_passed?: boolean | null;
  supported_claim_count?: number | null;
  unsupported_claim_count?: number | null;
  failure_reason_code?: string | null;
  status?: string | null;
  reason_code?: string | null;
  cited_ids?: string[];
  valid_cited_ids?: string[];
  invalid_cited_ids?: string[];
  available_evidence_ids?: string[];
  citation_coverage?: number | null;
}

export interface CitationItem extends EvidenceItem {
  citation_id?: string | null;
}

export interface CitationTrace extends StagePayload {
  citation_checked?: boolean | null;
  citation_valid?: boolean | null;
  citation_count?: number | null;
  invalid_citation_count?: number | null;
  citations: CitationItem[];
  note?: string | null;
}

export interface TimingsTrace extends StagePayload {
  timing_unit?: "milliseconds" | string;
  timing_source?: string | null;
  total_ms?: number | null;
  query_preparation_ms?: number | null;
  embedding_ms?: number | null;
  vector_search_ms?: number | null;
  lexical_search_ms?: number | null;
  structure_expansion_ms?: number | null;
  graph_recovery_ms?: number | null;
  reranking_ms?: number | null;
  evidence_composition_ms?: number | null;
  generation_ms?: number | null;
  validation_ms?: number | null;
  stage_timings_available?: boolean;
  retrieval_timing_snapshot?: unknown;
  instrumentation_overhead_ms?: number | null;
  historical_benchmark_values_mixed_into_live_trace?: boolean;
}

export interface OutcomeTrace {
  status?: RuntimeTraceStatus | string;
  answer_status?: string | null;
  refused?: boolean;
  failure_stage?: string | null;
  final_candidate_count?: number | null;
  final_evidence_count?: number | null;
  grounding_passed?: boolean | null;
  citation_valid?: boolean | null;
  final_decision?: string | null;
  refusal_reason_code?: string | null;
}

export interface RuntimeTraceV1 {
  trace: TraceMeta;
  query: QueryTrace;
  runtime: RuntimeTraceRuntime;
  guard: GuardTrace;
  retrieval: RetrievalTrace;
  structure_recovery: StructureRecoveryTrace;
  graph_recovery: GraphRecoveryTrace;
  rerank: RerankTrace;
  evidence: EvidenceTrace;
  answerability: AnswerabilityTrace;
  generation: GenerationTrace;
  grounding: GroundingTrace;
  citation: CitationTrace;
  timings: TimingsTrace;
  outcome: OutcomeTrace;
}

export interface ShowcaseErrorResponse {
  schema_version: "opk-rag.showcase-api-error.v1" | string;
  code: string;
  message: string;
  trace_id?: string | null;
  stage?: string | null;
  retryable?: boolean;
}

export interface TraceEnvelope {
  schema_version: typeof SHOWCASE_API_VERSION | string;
  trace_id: string;
  status: "completed" | "failed" | "refused" | "partial" | string;
  trace: RuntimeTraceV1;
}

export interface RuntimeTraceEventV1 {
  event_schema_version: typeof RUNTIME_TRACE_EVENT_SCHEMA_VERSION | string;
  event_id: string;
  trace_id: string;
  sequence: number;
  event_type: "stage_snapshot" | "trace_completed" | "trace_refused" | "trace_failed" | "trace_partial" | string;
  stage: TraceStage | string;
  timestamp: string | null;
  payload: Record<string, unknown>;
  terminal: boolean;
}

export interface VersionGuardResult {
  ok: boolean;
  reason?: string;
}

export function guardTraceVersion(trace: unknown): VersionGuardResult {
  const candidate = trace as Partial<RuntimeTraceV1> | null;
  if (!candidate || typeof candidate !== "object") return { ok: false, reason: "trace_not_object" };
  if (!candidate.trace || typeof candidate.trace !== "object") return { ok: false, reason: "trace_meta_missing" };
  if (candidate.trace.trace_schema_version !== RUNTIME_TRACE_SCHEMA_VERSION) {
    return { ok: false, reason: "runtime_trace_schema_version_mismatch" };
  }
  for (const section of TRACE_STAGE_ORDER) {
    if (!(section in candidate)) return { ok: false, reason: `missing_section:${section}` };
  }
  return { ok: true };
}

export function guardEventVersion(event: unknown): VersionGuardResult {
  const candidate = event as Partial<RuntimeTraceEventV1> | null;
  if (!candidate || typeof candidate !== "object") return { ok: false, reason: "event_not_object" };
  if (candidate.event_schema_version !== RUNTIME_TRACE_EVENT_SCHEMA_VERSION) {
    return { ok: false, reason: "runtime_trace_event_schema_version_mismatch" };
  }
  if (typeof candidate.event_id !== "string" || typeof candidate.sequence !== "number") {
    return { ok: false, reason: "event_identity_missing" };
  }
  return { ok: true };
}

export function lifecycleFromTraceStatus(status?: string | null): ExecutionLifecycle {
  if (status === "completed") return "completed";
  if (status === "refused") return "refused";
  if (status === "failed") return "failed";
  if (status === "partial") return "partial";
  if (status === "started" || status === "running") return "executing";
  return "idle";
}

export function lifecycleFromEvent(event: RuntimeTraceEventV1): ExecutionLifecycle {
  if (event.event_type === "trace_completed") return "completed";
  if (event.event_type === "trace_refused") return "refused";
  if (event.event_type === "trace_failed") return "failed";
  if (event.event_type === "trace_partial") return "partial";
  return "executing";
}

export function stageLabel(stage: string): string {
  return stage.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}
