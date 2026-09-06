import type { Translate, TranslationKey } from "../i18n";
import type { RuntimeTraceV1, StageState, TraceStage } from "../types/runtimeTrace";

export type GuardActionId = "continue" | "structure" | "graph" | "fail_closed";

export interface PipelineNodeModel {
  id: string;
  labelKey: TranslationKey;
  stage?: TraceStage;
  state: StageState | string;
  primary?: string;
  secondary?: string;
  tone?: string;
}

export interface RetrievalGuardModel {
  nodes: PipelineNodeModel[];
  selectedAction: GuardActionId | null;
  recoveryAttempts: number | null;
  recoveryMaximum: number | null;
  reasonCode: string | null;
  reasonExplanation: string;
  initialCandidates: number | null;
  structureBefore: number | null;
  structureExpanded: number | null;
  structureAfter: number | null;
  graphBefore: number | null;
  graphRecovered: number | null;
  graphAfter: number | null;
  unifiedCandidates: number | null;
  rerankInput: number | null;
  rerankOutput: number | null;
  evidenceCount: number | null;
  graphHop: number | null;
}

function n(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stageState(trace: RuntimeTraceV1, stage: TraceStage): StageState | string {
  if (stage === "outcome") return trace.outcome.status || "unavailable";
  const payload = trace[stage] as { stage_state?: StageState | string };
  return payload?.stage_state || "unavailable";
}

export function actionFromTrace(trace?: RuntimeTraceV1): GuardActionId | null {
  if (!trace) return null;
  if (trace.trace.status === "refused" || trace.outcome.refused || trace.guard.fail_closed) return "fail_closed";
  if (trace.guard.recovery_action === "structure_recovery") return "structure";
  if (trace.guard.recovery_action === "graph_recovery") return "graph";
  if (trace.guard.recovery_action === "none" || trace.guard.recovery_required === false) return "continue";
  return null;
}

export function explainReasonCode(code: string | null | undefined, t: Translate): string {
  if (!code) return t("pipeline.reasonUnavailable");
  const known: Record<string, TranslationKey> = {
    body_confidence_preserved: "pipeline.reason.bodyConfidence",
    structural_query_phrase_or_entity_section_signal: "pipeline.reason.structureSignal",
    query_signal_with_seed_graph_availability: "pipeline.reason.graphSignal",
    missing_query_or_retrieval_signal: "pipeline.reason.noRecoverySignal",
    answerable_generation_abstained: "pipeline.reason.generationAbstained",
  };
  const key = known[code];
  return key ? t(key) : t("pipeline.reasonFallback", { code });
}

export function buildRetrievalGuardModel(trace: RuntimeTraceV1 | undefined, t: Translate): RetrievalGuardModel | null {
  if (!trace) return null;

  const initialCandidates = n(trace.retrieval.initial_candidate_count);
  const structureBefore = n(trace.structure_recovery.candidate_pool_count_before);
  const structureExpanded = n(trace.structure_recovery.expanded_candidate_count);
  const structureAfter = n(trace.structure_recovery.candidate_pool_count_after ?? trace.retrieval.post_structure_candidate_count);
  const graphBefore = n(trace.graph_recovery.candidate_pool_count_before);
  const graphRecovered = n(trace.graph_recovery.recovered_candidate_count);
  const graphAfter = n(trace.graph_recovery.candidate_pool_count_after);
  const rerankInput = n(trace.rerank.input_candidate_count ?? trace.retrieval.rerank_input_count);
  const rerankOutput = n(trace.rerank.output_candidate_count ?? trace.retrieval.final_result_count);
  const evidenceCount = n(trace.evidence.evidence_count);
  const unifiedCandidates = rerankInput ?? graphAfter ?? structureAfter ?? initialCandidates;

  const nodes: PipelineNodeModel[] = [
    { id: "query", labelKey: "pipeline.query", stage: "query", state: stageState(trace, "query"), primary: trace.query.query_text || undefined },
    {
      id: "retrieval", labelKey: "pipeline.initialRetrieval", stage: "retrieval", state: stageState(trace, "retrieval"),
      primary: trace.retrieval.executed_policy || trace.retrieval.requested_policy || undefined,
      secondary: initialCandidates === null ? t("pipeline.candidatesUnavailable") : t("pipeline.candidateCount", { count: initialCandidates }),
    },
    {
      id: "guard", labelKey: "pipeline.guard", stage: "guard", state: stageState(trace, "guard"),
      primary: trace.guard.recovery_action || trace.guard.final_decision || undefined,
      secondary: t("pipeline.budgetCompact", { attempts: trace.guard.recovery_attempt_count ?? "--", max: trace.guard.maximum_recovery_attempt_count ?? "--" }),
    },
    {
      id: "structure", labelKey: "pipeline.structure", stage: "structure_recovery", state: stageState(trace, "structure_recovery"),
      primary: trace.structure_recovery.triggered ? t("pipeline.activated") : t("pipeline.notTriggered"),
      secondary: structureBefore !== null && structureAfter !== null ? t("pipeline.countFlow", { before: structureBefore, after: structureAfter }) : undefined,
    },
    {
      id: "graph", labelKey: "pipeline.graph", stage: "graph_recovery", state: stageState(trace, "graph_recovery"),
      primary: trace.graph_recovery.graph_activated ? t("pipeline.activated") : t("pipeline.notTriggered"),
      secondary: trace.graph_recovery.graph_activated && n(trace.graph_recovery.hop_depth) !== null ? t("pipeline.hop", { hop: trace.graph_recovery.hop_depth ?? "--" }) : undefined,
    },
    {
      id: "unified", labelKey: "pipeline.unified", state: "completed",
      primary: unifiedCandidates === null ? t("common.unavailable") : t("pipeline.candidateCount", { count: unifiedCandidates }),
    },
    {
      id: "rerank", labelKey: "pipeline.rerank", stage: "rerank", state: stageState(trace, "rerank"),
      primary: rerankInput !== null && rerankOutput !== null ? t("pipeline.countFlow", { before: rerankInput, after: rerankOutput }) : undefined,
      secondary: [trace.rerank.precision, trace.rerank.execution_scope].filter(Boolean).join(" · ") || undefined,
    },
    {
      id: "evidence", labelKey: "pipeline.evidence", stage: "evidence", state: stageState(trace, "evidence"),
      primary: evidenceCount === null ? t("common.unavailable") : t("pipeline.evidenceCount", { count: evidenceCount }),
    },
    {
      id: "outcome", labelKey: "pipeline.outcome", stage: "outcome", state: trace.outcome.status || trace.trace.status,
      primary: trace.outcome.status || trace.trace.status,
    },
  ];

  return {
    nodes,
    selectedAction: actionFromTrace(trace),
    recoveryAttempts: n(trace.guard.recovery_attempt_count),
    recoveryMaximum: n(trace.guard.maximum_recovery_attempt_count),
    reasonCode: trace.guard.recovery_reason_code || trace.guard.guard_reason_code || trace.guard.refusal_reason_code || null,
    reasonExplanation: explainReasonCode(trace.guard.recovery_reason_code || trace.guard.guard_reason_code || trace.guard.refusal_reason_code, t),
    initialCandidates,
    structureBefore,
    structureExpanded,
    structureAfter,
    graphBefore,
    graphRecovered,
    graphAfter,
    unifiedCandidates,
    rerankInput,
    rerankOutput,
    evidenceCount,
    graphHop: n(trace.graph_recovery.hop_depth),
  };
}
