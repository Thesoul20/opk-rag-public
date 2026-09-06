import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import type { Translate, TranslationKey } from "../i18n";

export type AnswerStageId = "evidence" | "answerability" | "generation" | "grounding" | "citation" | "outcome";

export interface AnswerStageModel {
  id: AnswerStageId;
  state: string;
  detail: string;
}

export interface CitationLinkModel {
  citationId: string;
  evidenceId: string;
  sourceCandidateId: string | null;
  documentPath: string | null;
  sectionPath: string[] | null;
  startLine: number | null;
  endLine: number | null;
  evidenceFound: boolean;
  candidateIdentityMatches: boolean;
  graphRecovered: boolean;
}

export interface AnswerValidationModel {
  executionScope: string | null;
  searchOnly: boolean;
  evidenceCount: number | null;
  stages: AnswerStageModel[];
  answerability: {
    stageState: string;
    state: string | null;
    answerable: boolean | null;
    reasonCode: string | null;
    confidence: number | null;
    consideredEvidenceCount: number | null;
  };
  generation: {
    stageState: string;
    attempted: boolean | null;
    completed: boolean | null;
    abstained: boolean | null;
    provider: string | null;
    model: string | null;
    endpointType: string | null;
    latencyMs: number | null;
    promptTokens: number | null;
    completionTokens: number | null;
    totalTokens: number | null;
    finishReason: string | null;
  };
  grounding: {
    stageState: string;
    checked: boolean | null;
    passed: boolean | null;
    status: string | null;
    reasonCode: string | null;
    supportedClaimCount: number | null;
    unsupportedClaimCount: number | null;
    citationCoverage: number | null;
  };
  citation: {
    stageState: string;
    checked: boolean | null;
    valid: boolean | null;
    count: number | null;
    invalidCount: number | null;
    links: CitationLinkModel[];
  };
  outcome: {
    status: string;
    answerStatus: string | null;
    refused: boolean;
    failureStage: string | null;
    refusalReasonCode: string | null;
    explanation: string;
  };
  finalAnswerBodyAvailable: false;
  orphanCitationEvidenceCount: number;
  citationCandidateIdentityMismatchCount: number;
}

function n(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

const refusalReasonKeys: Record<string, TranslationKey> = {
  answerable_generation_abstained: "answerViz.reason.generationAbstained",
};

export function buildAnswerValidationModel(trace: RuntimeTraceV1 | undefined, t: Translate): AnswerValidationModel | null {
  if (!trace) return null;
  const evidenceById = new Map((trace.evidence.evidence_items || []).filter((item) => item.evidence_id).map((item) => [item.evidence_id!, item]));
  const retrievalById = new Map((trace.retrieval.candidates || []).filter((item) => item.candidate_id).map((item) => [item.candidate_id!, item]));
  const citationLinks: CitationLinkModel[] = (trace.citation.citations || []).map((citation, index) => {
    const evidenceId = citation.evidence_id || citation.citation_id || `citation-${index + 1}`;
    const evidence = evidenceById.get(evidenceId);
    const sourceCandidateId = citation.source_candidate_id || evidence?.source_candidate_id || null;
    const candidate = sourceCandidateId ? retrievalById.get(sourceCandidateId) : undefined;
    return {
      citationId: citation.citation_id || evidenceId,
      evidenceId,
      sourceCandidateId,
      documentPath: citation.document_path || evidence?.document_path || null,
      sectionPath: citation.section_path || evidence?.section_path || null,
      startLine: n(citation.start_line ?? evidence?.start_line),
      endLine: n(citation.end_line ?? evidence?.end_line),
      evidenceFound: Boolean(evidence),
      candidateIdentityMatches: Boolean(evidence && sourceCandidateId && evidence.source_candidate_id === sourceCandidateId),
      graphRecovered: candidate?.graph_recovered === true,
    };
  });
  const executionScope = trace.query.execution_scope || null;
  const searchOnly = executionScope === "search" && [trace.answerability.stage_state, trace.generation.stage_state, trace.grounding.stage_state, trace.citation.stage_state].every((state) => state === "not_applicable");
  const outcomeStatus = trace.outcome.status || trace.trace.status || "unavailable";
  const refusalCode = trace.outcome.refusal_reason_code || null;
  const outcomeExplanation = searchOnly
    ? t("answerViz.reason.searchScope")
    : outcomeStatus === "refused"
      ? refusalCode && refusalReasonKeys[refusalCode]
        ? t(refusalReasonKeys[refusalCode])
        : t("answerViz.reason.refusedFallback", { code: refusalCode || "--" })
      : outcomeStatus === "failed"
        ? t("answerViz.reason.failed")
        : t("answerViz.reason.completed");

  const evidenceCount = n(trace.evidence.evidence_count);
  const stageDetail = (id: AnswerStageId): string => {
    if (id === "evidence") return evidenceCount === null ? "--" : String(evidenceCount);
    if (id === "answerability") return trace.answerability.answerability_state || trace.answerability.reason_code || "--";
    if (id === "generation") return trace.generation.generation_abstained === true ? t("answerViz.abstained") : trace.generation.generation_completed === true ? t("answerViz.generated") : "--";
    if (id === "grounding") return trace.grounding.status || trace.grounding.reason_code || "--";
    if (id === "citation") return trace.citation.citation_count === null || trace.citation.citation_count === undefined ? "--" : String(trace.citation.citation_count);
    return outcomeStatus;
  };
  const stateFor = (id: AnswerStageId): string => {
    if (id === "evidence") return trace.evidence.stage_state || "unavailable";
    if (id === "outcome") return outcomeStatus;
    return trace[id].stage_state || "unavailable";
  };

  return {
    executionScope,
    searchOnly,
    evidenceCount,
    stages: (["evidence", "answerability", "generation", "grounding", "citation", "outcome"] as AnswerStageId[]).map((id) => ({ id, state: stateFor(id), detail: stageDetail(id) })),
    answerability: {
      stageState: trace.answerability.stage_state || "unavailable",
      state: trace.answerability.answerability_state || null,
      answerable: trace.answerability.answerable ?? null,
      reasonCode: trace.answerability.reason_code || null,
      confidence: n(trace.answerability.confidence),
      consideredEvidenceCount: n(trace.answerability.considered_evidence_count ?? trace.answerability.evidence_count),
    },
    generation: {
      stageState: trace.generation.stage_state || "unavailable",
      attempted: trace.generation.generation_attempted ?? null,
      completed: trace.generation.generation_completed ?? null,
      abstained: trace.generation.generation_abstained ?? null,
      provider: trace.generation.generation_provider || null,
      model: trace.generation.generation_model || null,
      endpointType: trace.generation.generation_endpoint_type || null,
      latencyMs: n(trace.generation.generation_latency_ms),
      promptTokens: n(trace.generation.prompt_tokens),
      completionTokens: n(trace.generation.completion_tokens),
      totalTokens: n(trace.generation.total_tokens),
      finishReason: trace.generation.finish_reason || null,
    },
    grounding: {
      stageState: trace.grounding.stage_state || "unavailable",
      checked: trace.grounding.grounding_checked ?? null,
      passed: trace.grounding.grounding_passed ?? null,
      status: trace.grounding.status || null,
      reasonCode: trace.grounding.reason_code || trace.grounding.failure_reason_code || null,
      supportedClaimCount: n(trace.grounding.supported_claim_count),
      unsupportedClaimCount: n(trace.grounding.unsupported_claim_count),
      citationCoverage: n(trace.grounding.citation_coverage),
    },
    citation: {
      stageState: trace.citation.stage_state || "unavailable",
      checked: trace.citation.citation_checked ?? null,
      valid: trace.citation.citation_valid ?? null,
      count: n(trace.citation.citation_count),
      invalidCount: n(trace.citation.invalid_citation_count),
      links: citationLinks,
    },
    outcome: {
      status: outcomeStatus,
      answerStatus: trace.outcome.answer_status || null,
      refused: trace.outcome.refused === true || outcomeStatus === "refused",
      failureStage: trace.outcome.failure_stage || null,
      refusalReasonCode: refusalCode,
      explanation: outcomeExplanation,
    },
    finalAnswerBodyAvailable: false,
    orphanCitationEvidenceCount: citationLinks.filter((link) => !link.evidenceFound).length,
    citationCandidateIdentityMismatchCount: citationLinks.filter((link) => link.evidenceFound && !link.candidateIdentityMatches).length,
  };
}
