import type { RuntimeTraceV1 } from "../../types/runtimeTrace";
import { buildEvidenceValidationModel } from "./evidenceRuntime";

export type ExecutiveStageId = "retrieve" | "guard" | "recover" | "rerank" | "evidence" | "validate" | "outcome";
export interface ExecutiveStage { id: ExecutiveStageId; label: string; state: string; detail: string; }
export interface ExecutiveEvidence { evidenceId: string | null; document: string; section: string; lines: string; provenance: string; cited: boolean; graphRecovered: boolean; }
export interface ExecutiveGraphStory { active: boolean; hop: number | null; recoveredCandidates: number | null; source: string | null; relation: string | null; target: string | null; evidenceId: string | null; citationId: string | null; }
export interface ExecutiveStory {
  scope: string;
  status: string;
  safeRefusal: boolean;
  recoveryKind: "none" | "structure" | "graph";
  recoveryLabel: string;
  messages: string[];
  stages: ExecutiveStage[];
  evidence: ExecutiveEvidence[];
  graph: ExecutiveGraphStory;
}

const title = (value: string | null | undefined) => value ? value.split("/").pop()?.replace(/\.md$/i, "") || value : null;
const lines = (start?: number | null, end?: number | null) => start == null ? "Lines unavailable" : end == null || end === start ? `L${start}` : `L${start}–L${end}`;
const stageState = (value: string | null | undefined) => value || "unavailable";

export function recoveryKind(trace: RuntimeTraceV1): ExecutiveStory["recoveryKind"] {
  if (trace.structure_recovery.triggered) return "structure";
  if (trace.graph_recovery.graph_activated) return "graph";
  return "none";
}

function validationState(trace: RuntimeTraceV1): string {
  if (trace.query.execution_scope === "search") return "not_applicable";
  if (trace.outcome.status === "refused" && !trace.outcome.failure_stage) return "refused";
  if (trace.grounding.grounding_passed === true && trace.citation.citation_valid !== false) return "completed";
  if (trace.grounding.stage_state === "failed" || trace.citation.stage_state === "failed") return "failed";
  return trace.citation.stage_state || trace.grounding.stage_state || "unavailable";
}

function outcomeDetail(trace: RuntimeTraceV1): string {
  if (trace.outcome.status === "refused" && !trace.outcome.failure_stage) return "Safe refusal";
  if (trace.outcome.status === "failed") return trace.outcome.failure_stage ? `Failed at ${trace.outcome.failure_stage}` : "Failed safely";
  if (trace.outcome.answer_status === "answered") return "Answered";
  if (trace.query.execution_scope === "search") return "Search complete";
  return trace.outcome.status || trace.trace.status;
}

export function buildExecutiveStory(trace: RuntimeTraceV1): ExecutiveStory {
  const recovery = recoveryKind(trace);
  const safeRefusal = trace.outcome.status === "refused" && !trace.outcome.failure_stage;
  const evidenceModel = buildEvidenceValidationModel(trace);
  const messages: string[] = [];
  if (recovery === "none") messages.push("Initial retrieval was sufficient, so the bounded Guarded Agent preserved the normal path without unnecessary recovery.");
  if (recovery === "structure") messages.push(`The bounded recovery path used Structure Recovery before reranking; the observed Candidate Pool moved from ${trace.structure_recovery.candidate_pool_count_before ?? "?"} to ${trace.structure_recovery.candidate_pool_count_after ?? "?"}.`);
  if (recovery === "graph") messages.push(`The bounded recovery path used authoritative one-hop Graph Retrieval and recovered ${trace.graph_recovery.recovered_candidate_count ?? "?"} Candidate before unified reranking.`);
  if (trace.query.execution_scope === "search") messages.push("Search scope ends at Evidence; answer generation and downstream answer validation are not executed for this trace.");
  else if (safeRefusal) messages.push("Evidence was available and Answerability passed, but Generation abstained; the system returned a safe refusal instead of forcing an answer.");
  else if (trace.grounding.grounding_passed === true) messages.push("The generated answer passed Grounding and Citation validation against the selected Evidence before release.");
  else if (trace.trace.status === "failed") messages.push("The execution failed at an authoritative runtime stage; no success narrative is synthesized.");

  const graphEvidence = evidenceModel.evidence.find((row) => row.graphRecovered);
  const graphCitation = graphEvidence ? evidenceModel.citations.find((row) => row.evidenceId === graphEvidence.evidenceId) : undefined;
  const edge = trace.graph_recovery.traversed_edges?.[0];
  const graph: ExecutiveGraphStory = {
    active: trace.graph_recovery.graph_activated === true,
    hop: trace.graph_recovery.hop_depth ?? null,
    recoveredCandidates: trace.graph_recovery.recovered_candidate_count ?? null,
    source: title(edge?.source_node_id || edge?.source_document_id),
    relation: edge?.relation_type || null,
    target: title(edge?.target_node_id || edge?.target_document_id),
    evidenceId: graphEvidence?.evidenceId || null,
    citationId: graphCitation?.citationId || null,
  };

  const evidence = evidenceModel.evidence.slice(0, 4).map((row) => ({
    evidenceId: row.evidenceId,
    document: title(row.evidence.document_path) || "Document unavailable",
    section: row.evidence.section_path?.join(" / ") || "Section unavailable",
    lines: lines(row.evidence.start_line, row.evidence.end_line),
    provenance: row.graphRecovered ? "Graph recovered" : (row.evidence.source_provenance || []).some((p) => p.source_kind === "structure_expansion") ? "Structure expanded" : "Initial retrieval",
    cited: row.citationIds.length > 0,
    graphRecovered: row.graphRecovered,
  }));

  const stages: ExecutiveStage[] = [
    { id: "retrieve", label: "Retrieve", state: stageState(trace.retrieval.stage_state), detail: `${trace.retrieval.executed_policy || trace.retrieval.retrieval_mode || "retrieval"} · ${trace.retrieval.initial_candidate_count ?? "?"} initial` },
    { id: "guard", label: "Guard", state: stageState(trace.guard.stage_state), detail: trace.guard.recovery_required ? "Recovery required" : "Normal path" },
    { id: "recover", label: "Recover", state: recovery === "none" ? "skipped" : "completed", detail: recovery === "graph" ? `Graph · ${trace.graph_recovery.hop_depth ?? "?"} hop` : recovery === "structure" ? "Structure Recovery" : "No Recovery" },
    { id: "rerank", label: "Rerank", state: stageState(trace.rerank.stage_state), detail: `${trace.rerank.input_candidate_count ?? "?"} → ${trace.rerank.output_candidate_count ?? "?"}` },
    { id: "evidence", label: "Evidence", state: stageState(trace.evidence.stage_state), detail: `${trace.evidence.evidence_count ?? "?"} selected` },
    { id: "validate", label: "Validate", state: validationState(trace), detail: trace.query.execution_scope === "search" ? "Search scope · N/A" : safeRefusal ? "Generation abstained" : trace.grounding.grounding_passed ? "Grounding passed" : trace.grounding.status || "Validation" },
    { id: "outcome", label: "Answer / Refuse", state: trace.outcome.status || trace.trace.status, detail: outcomeDetail(trace) },
  ];

  return { scope: String(trace.query.execution_scope || "unknown"), status: trace.outcome.status || trace.trace.status, safeRefusal, recoveryKind: recovery, recoveryLabel: recovery === "graph" ? "Graph Recovery" : recovery === "structure" ? "Structure Recovery" : "No Recovery", messages, stages, evidence, graph };
}

export const EXECUTIVE_ARCHITECTURE = [
  "Markdown / Obsidian",
  "Structure-aware Chunking",
  "Qwen3 Embedding",
  "Qdrant + Lexical Retrieval",
  "Guarded Structure / Graph Recovery",
  "BGE Reranking",
  "Evidence Composition",
  "LLM Generation",
  "Grounding / Citation",
] as const;

export const EXECUTIVE_CAPABILITIES = [
  ["Hybrid Retrieval", "Vector + lexical + structure-aware retrieval"],
  ["Guarded Agent", "Bounded recovery · maximum one attempt"],
  ["Graph Retrieval", "Authoritative one-hop relation recovery"],
  ["Evidence Governance", "Candidate ≠ Evidence"],
  ["Safe Answering", "Answerability + Grounding + Citation validation"],
] as const;
