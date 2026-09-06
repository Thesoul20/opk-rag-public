import type { RuntimeCandidate, RuntimeTraceV1 } from "../../types/runtimeTrace";
import { CONTROL_CENTER_LIMITS } from "../architecture";

export type ControlCenterRuntimeSummary = {
  status: string;
  traceId: string | null;
  executionScope: string | null;
  graphHop: number | null;
  recoveryAttempts: number | null;
  answerStatus: string | null;
  groundingPassed: boolean | null;
  uiDecisionAuthority: false;
};

export type CandidateRelationship = "candidate" | "context_candidate" | "evidence_selected";

export function adaptRuntimeTrace(trace?: RuntimeTraceV1): ControlCenterRuntimeSummary {
  return {
    status: trace?.trace?.status ?? "idle",
    traceId: trace?.trace?.trace_id ?? null,
    executionScope: trace?.query?.execution_scope ?? null,
    graphHop: trace?.graph_recovery?.hop_depth ?? null,
    recoveryAttempts: trace?.guard?.recovery_attempt_count ?? null,
    answerStatus: trace?.outcome?.status ?? null,
    groundingPassed: trace?.grounding?.grounding_passed ?? null,
    uiDecisionAuthority: false,
  };
}

export function rankMovement(initialRank?: number | null, finalRank?: number | null): number | null {
  if (initialRank == null || finalRank == null) return null;
  return initialRank - finalRank;
}

export function candidateRelationship(trace: RuntimeTraceV1, candidate: RuntimeCandidate): CandidateRelationship {
  const evidenceItems = trace.evidence?.evidence_items || [];
  const selectedAsEvidence = evidenceItems.some(item =>
    Boolean(candidate.candidate_id && item.source_candidate_id === candidate.candidate_id)
    || Boolean(candidate.chunk_id && item.chunk_id === candidate.chunk_id),
  );
  if (selectedAsEvidence) return "evidence_selected";
  if (candidate.selected_for_context) return "context_candidate";
  return "candidate";
}

export function candidateProvenanceLabels(candidate: RuntimeCandidate): string[] {
  const labels = new Set<string>();
  for (const provenance of candidate.provenance || []) {
    if (provenance.source_kind === "initial") labels.add(provenance.source_lane ? `Initial:${provenance.source_lane}` : "Initial");
    else if (provenance.source_kind === "structure_expansion") labels.add("Structure expanded");
    else if (provenance.source_kind === "graph_recovery") labels.add("Graph recovered");
    else if (provenance.source_kind) labels.add(provenance.source_kind);
  }
  if (candidate.structure_expanded) labels.add("Structure expanded");
  if (candidate.graph_recovered) labels.add("Graph recovered");
  return [...labels];
}

export function runtimeBoundaryContract() {
  return {
    runtimeTraceAuthoritative: true,
    uiDecisionAuthority: false,
    candidateEqualsEvidence: false,
    graphMaxHop: CONTROL_CENTER_LIMITS.graphMaxHop,
    recoveryMaxAttempts: CONTROL_CENTER_LIMITS.recoveryMaxAttempts,
    fakeRuntimeStateAllowed: false,
  } as const;
}
