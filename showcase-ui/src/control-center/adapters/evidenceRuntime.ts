import type { CitationItem, EvidenceItem, RuntimeCandidate, RuntimeTraceV1 } from "../../types/runtimeTrace";

export type ValidationStageId = "evidence_stage" | "answerability" | "generation" | "grounding" | "citation" | "outcome";

export interface RuntimeEvidenceRecord {
  key: string;
  traceIndex: number;
  evidence: EvidenceItem;
  evidenceId: string | null;
  sourceCandidateId: string | null;
  candidate?: RuntimeCandidate;
  candidateFound: boolean;
  graphRecovered: boolean;
  citationIds: string[];
}

export interface RuntimeCitationRecord {
  key: string;
  traceIndex: number;
  citation: CitationItem;
  citationId: string | null;
  evidenceId: string | null;
  evidence?: RuntimeEvidenceRecord;
  evidenceFound: boolean;
  directSourceCandidateId: string | null;
  resolvedSourceCandidateId: string | null;
  candidate?: RuntimeCandidate;
  candidateFound: boolean;
  candidateIdentityMatches: boolean | null;
  graphRecovered: boolean;
}

export interface EvidenceValidationModel {
  executionScope: string | null;
  searchOnly: boolean;
  evidence: RuntimeEvidenceRecord[];
  citations: RuntimeCitationRecord[];
  evidenceCount: number | null;
  stages: Array<{ id: ValidationStageId; state: string; detail: string }>;
  orphanCitationCount: number;
  citationMissingEvidenceIdentityCount: number;
  citationCandidateMismatchCount: number;
  evidenceCandidateMissingCount: number;
  evidenceMissingCandidateIdentityCount: number;
  semanticSeparationDeclared: boolean | null;
}

function n(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function candidateByRuntimeId(trace: RuntimeTraceV1, candidateId: string | null | undefined): RuntimeCandidate | undefined {
  if (!candidateId) return undefined;
  return (trace.retrieval.candidates || []).find((candidate) => candidate.candidate_id === candidateId || candidate.chunk_id === candidateId);
}

export function evidenceSelectionKey(item: EvidenceItem, index: number): string {
  return item.evidence_id ? `evidence:${item.evidence_id}` : `evidence-index:${index}`;
}

export function citationSelectionKey(item: CitationItem, index: number): string {
  return item.citation_id ? `citation:${item.citation_id}` : `citation-index:${index}`;
}

export function evidenceProvenanceLabels(item: EvidenceItem): string[] {
  const labels = new Set<string>();
  for (const source of item.source_provenance || []) {
    if (source.source_kind === "graph_recovery") labels.add("Graph recovered");
    else if (source.source_kind === "structure_expansion") labels.add("Structure expanded");
    else if (source.source_kind === "initial") labels.add("Initial retrieval");
    else if (source.source_kind) labels.add(source.source_kind);
  }
  return [...labels];
}

export function buildEvidenceValidationModel(trace: RuntimeTraceV1): EvidenceValidationModel {
  const rawEvidence = trace.evidence.evidence_items || [];
  const rawCitations = trace.citation.citations || [];
  const citationIdsByEvidence = new Map<string, string[]>();
  for (const citation of rawCitations) {
    if (!citation.evidence_id) continue;
    const values = citationIdsByEvidence.get(citation.evidence_id) || [];
    values.push(citation.citation_id || "Citation ID unavailable");
    citationIdsByEvidence.set(citation.evidence_id, values);
  }

  const evidence = rawEvidence.map<RuntimeEvidenceRecord>((item, index) => {
    const sourceCandidateId = item.source_candidate_id || null;
    const candidate = candidateByRuntimeId(trace, sourceCandidateId);
    return {
      key: evidenceSelectionKey(item, index),
      traceIndex: index,
      evidence: item,
      evidenceId: item.evidence_id || null,
      sourceCandidateId,
      candidate,
      candidateFound: Boolean(candidate),
      graphRecovered: candidate?.graph_recovered === true || (item.source_provenance || []).some((row) => row.source_kind === "graph_recovery"),
      citationIds: item.evidence_id ? (citationIdsByEvidence.get(item.evidence_id) || []) : [],
    };
  });
  const evidenceById = new Map(evidence.filter((row) => row.evidenceId).map((row) => [row.evidenceId!, row]));

  const citations = rawCitations.map<RuntimeCitationRecord>((citation, index) => {
    const evidenceId = citation.evidence_id || null;
    const linkedEvidence = evidenceId ? evidenceById.get(evidenceId) : undefined;
    const directSourceCandidateId = citation.source_candidate_id || null;
    const resolvedSourceCandidateId = directSourceCandidateId || linkedEvidence?.sourceCandidateId || null;
    const candidate = candidateByRuntimeId(trace, resolvedSourceCandidateId);
    const candidateIdentityMatches = linkedEvidence && directSourceCandidateId && linkedEvidence.sourceCandidateId
      ? directSourceCandidateId === linkedEvidence.sourceCandidateId
      : null;
    return {
      key: citationSelectionKey(citation, index),
      traceIndex: index,
      citation,
      citationId: citation.citation_id || null,
      evidenceId,
      evidence: linkedEvidence,
      evidenceFound: Boolean(linkedEvidence),
      directSourceCandidateId,
      resolvedSourceCandidateId,
      candidate,
      candidateFound: Boolean(candidate),
      candidateIdentityMatches,
      graphRecovered: candidate?.graph_recovered === true || linkedEvidence?.graphRecovered === true,
    };
  });

  const executionScope = trace.query.execution_scope || null;
  const searchOnly = executionScope === "search" && [trace.answerability.stage_state, trace.generation.stage_state, trace.grounding.stage_state, trace.citation.stage_state].every((state) => state === "not_applicable");
  const outcomeStatus = trace.outcome.status || trace.trace.status || "unavailable";
  const evidenceCount = n(trace.evidence.evidence_count);
  const detail = (id: ValidationStageId): string => {
    if (id === "evidence_stage") return evidenceCount == null ? "Unavailable" : `${evidenceCount} items`;
    if (id === "answerability") return trace.answerability.answerability_state || trace.answerability.reason_code || "Unavailable";
    if (id === "generation") return trace.generation.generation_abstained === true ? "abstained" : trace.generation.generation_completed === true ? "completed" : "Unavailable";
    if (id === "grounding") return trace.grounding.status || trace.grounding.reason_code || "Unavailable";
    if (id === "citation") return trace.citation.citation_count == null ? "Unavailable" : `${trace.citation.citation_count} citations`;
    return outcomeStatus;
  };
  const state = (id: ValidationStageId): string => {
    if (id === "evidence_stage") return trace.evidence.stage_state || "unavailable";
    if (id === "outcome") return outcomeStatus;
    return trace[id].stage_state || "unavailable";
  };
  const stageIds: ValidationStageId[] = ["evidence_stage", "answerability", "generation", "grounding", "citation", "outcome"];

  return {
    executionScope,
    searchOnly,
    evidence,
    citations,
    evidenceCount,
    stages: stageIds.map((id) => ({ id, state: state(id), detail: detail(id) })),
    orphanCitationCount: citations.filter((row) => Boolean(row.evidenceId) && !row.evidenceFound).length,
    citationMissingEvidenceIdentityCount: citations.filter((row) => !row.evidenceId).length,
    citationCandidateMismatchCount: citations.filter((row) => row.candidateIdentityMatches === false).length,
    evidenceCandidateMissingCount: evidence.filter((row) => Boolean(row.sourceCandidateId) && !row.candidateFound).length,
    evidenceMissingCandidateIdentityCount: evidence.filter((row) => !row.sourceCandidateId).length,
    semanticSeparationDeclared: trace.evidence.candidate_evidence_semantic_separation ?? null,
  };
}

export function evidenceByKey(trace: RuntimeTraceV1, key: string | null): RuntimeEvidenceRecord | undefined {
  if (!key) return undefined;
  return buildEvidenceValidationModel(trace).evidence.find((row) => row.key === key);
}

export function citationByKey(trace: RuntimeTraceV1, key: string | null): RuntimeCitationRecord | undefined {
  if (!key) return undefined;
  return buildEvidenceValidationModel(trace).citations.find((row) => row.key === key);
}
