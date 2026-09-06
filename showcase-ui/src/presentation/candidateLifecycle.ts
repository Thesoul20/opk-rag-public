import type { Translate } from "../i18n";
import type { EvidenceItem, RuntimeCandidate, RuntimeTraceV1 } from "../types/runtimeTrace";

export type CandidateSource = "vector" | "lexical" | "structure" | "graph" | string;

export interface CandidateLineageRow {
  candidateId: string;
  chunkId?: string | null;
  documentPath?: string | null;
  sectionPath?: string[] | null;
  sources: CandidateSource[];
  initialRank: number | null;
  retrievalScore: number | null;
  rerankPosition: number | null;
  rerankScore: number | null;
  rankDelta: number | null;
  selectedForContext: boolean;
  structureExpanded: boolean;
  graphRecovered: boolean;
  evidenceId?: string | null;
  evidencePosition: number | null;
  evidenceScore: number | null;
  evidenceSelectionReason?: string | null;
  selectedAsEvidence: boolean;
}

export interface CandidateLifecycleModel {
  rows: CandidateLineageRow[];
  evidence: EvidenceItem[];
  initialCandidateCount: number | null;
  structureExpandedCount: number | null;
  graphRecoveredCount: number | null;
  unifiedCandidateCount: number | null;
  rerankInputCount: number | null;
  rerankOutputCount: number | null;
  evidenceCount: number | null;
  rerankerModel?: string | null;
  precision?: string | null;
  executionScope?: string | null;
  outcomeStatus: string;
  observedCandidateCount: number;
  candidateMetricTraceMismatchCount: number;
  orphanEvidenceCandidateCount: number;
  candidateIdentityLinkageUsesAuthoritativeId: true;
  candidateEvidenceSemanticSeparation: boolean;
  recoveredCandidateBypassRerank: false;
  uiRerankSortUsesRuntimePosition: true;
  s03GraphCandidateIdentityChainValid: boolean;
}

function n(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function candidateSources(candidate: RuntimeCandidate): CandidateSource[] {
  const values = new Set<string>();
  for (const source of candidate.retrieval_sources || []) {
    const normalized = source.toLowerCase();
    if (normalized.includes("keyword")) values.add("lexical");
    else values.add(normalized);
  }
  for (const provenance of candidate.provenance || []) {
    if (provenance.source_kind === "structure_expansion") values.add("structure");
    else if (provenance.source_kind === "graph_recovery") values.add("graph");
    else if (provenance.source_lane) values.add(provenance.source_lane.toLowerCase());
  }
  if (candidate.structure_expanded) values.add("structure");
  if (candidate.graph_recovered) values.add("graph");
  return [...values];
}

export function buildCandidateLifecycleModel(trace: RuntimeTraceV1 | undefined, _t?: Translate): CandidateLifecycleModel | null {
  if (!trace) return null;
  const retrievalCandidates = trace.retrieval.candidates || [];
  const candidateById = new Map(retrievalCandidates.filter((row) => row.candidate_id).map((row) => [row.candidate_id!, row]));
  const rerankById = new Map((trace.rerank.ranked_candidates || []).filter((row) => row.candidate_id).map((row) => [row.candidate_id!, row]));
  const evidenceByCandidate = new Map((trace.evidence.evidence_items || []).filter((row) => row.source_candidate_id).map((row) => [row.source_candidate_id!, row]));

  // Runtime rerank_position is presentation authority; never sort by score.
  const orderedIds = (trace.rerank.ranked_candidates || [])
    .filter((row) => row.candidate_id)
    .slice()
    .sort((a, b) => (a.rerank_position ?? Number.MAX_SAFE_INTEGER) - (b.rerank_position ?? Number.MAX_SAFE_INTEGER))
    .map((row) => row.candidate_id!);
  for (const candidate of retrievalCandidates) {
    if (candidate.candidate_id && !orderedIds.includes(candidate.candidate_id)) orderedIds.push(candidate.candidate_id);
  }

  const rows = orderedIds.map<CandidateLineageRow>((candidateId) => {
    const candidate = candidateById.get(candidateId);
    const rerank = rerankById.get(candidateId);
    const evidence = evidenceByCandidate.get(candidateId);
    const initialRank = n(candidate?.initial_rank);
    const rerankPosition = n(rerank?.rerank_position ?? candidate?.final_rank);
    return {
      candidateId,
      chunkId: candidate?.chunk_id,
      documentPath: candidate?.document_path,
      sectionPath: candidate?.section_path,
      sources: candidate ? candidateSources(candidate) : [],
      initialRank,
      retrievalScore: n(candidate?.retrieval_score),
      rerankPosition,
      rerankScore: n(rerank?.rerank_score ?? candidate?.rerank_score),
      rankDelta: initialRank !== null && rerankPosition !== null ? initialRank - rerankPosition : null,
      selectedForContext: candidate?.selected_for_context === true,
      structureExpanded: candidate?.structure_expanded === true,
      graphRecovered: candidate?.graph_recovered === true,
      evidenceId: evidence?.evidence_id,
      evidencePosition: n(evidence?.evidence_position),
      evidenceScore: n(evidence?.evidence_score),
      evidenceSelectionReason: evidence?.selection_reason_code,
      selectedAsEvidence: Boolean(evidence),
    };
  });

  const orphanEvidenceCandidateCount = (trace.evidence.evidence_items || []).filter((row) => row.source_candidate_id && !candidateById.has(row.source_candidate_id)).length;
  const metricChecks: Array<[number | null, number | null]> = [
    [n(trace.rerank.output_candidate_count), n(trace.retrieval.final_result_count)],
    [n(trace.evidence.evidence_count), n(trace.evidence.evidence_items?.length)],
  ];
  const candidateMetricTraceMismatchCount = metricChecks.filter(([a, b]) => a !== null && b !== null && a !== b).length;

  const graphIds = new Set(trace.graph_recovery.recovered_candidate_ids || []);
  const s03GraphCandidateIdentityChainValid = trace.graph_recovery.graph_activated !== true || [...graphIds].every((id) => rerankById.has(id) && (!evidenceByCandidate.has(id) || evidenceByCandidate.get(id)?.source_candidate_id === id));

  return {
    rows,
    evidence: trace.evidence.evidence_items || [],
    initialCandidateCount: n(trace.retrieval.initial_candidate_count),
    structureExpandedCount: n(trace.structure_recovery.expanded_candidate_count),
    graphRecoveredCount: n(trace.graph_recovery.recovered_candidate_count),
    unifiedCandidateCount: n(trace.retrieval.post_graph_unified_candidate_count ?? trace.retrieval.candidate_pool_count_after_recovery ?? trace.graph_recovery.candidate_pool_count_after ?? trace.structure_recovery.candidate_pool_count_after),
    rerankInputCount: n(trace.rerank.input_candidate_count ?? trace.retrieval.rerank_input_count),
    rerankOutputCount: n(trace.rerank.output_candidate_count ?? trace.retrieval.final_result_count),
    evidenceCount: n(trace.evidence.evidence_count),
    rerankerModel: trace.rerank.reranker_model || trace.runtime.reranker_model,
    precision: trace.rerank.precision || trace.runtime.reranker_precision,
    executionScope: trace.rerank.execution_scope || trace.query.execution_scope,
    outcomeStatus: trace.trace.status,
    observedCandidateCount: rows.length,
    candidateMetricTraceMismatchCount,
    orphanEvidenceCandidateCount,
    candidateIdentityLinkageUsesAuthoritativeId: true,
    candidateEvidenceSemanticSeparation: trace.evidence.candidate_evidence_semantic_separation === true,
    recoveredCandidateBypassRerank: false,
    uiRerankSortUsesRuntimePosition: true,
    s03GraphCandidateIdentityChainValid,
  };
}
