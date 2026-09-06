import type { Translate } from "../i18n";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";

export interface ExecutiveNarrative {
  messages: string[];
  recoveryLabel: string;
  graphLabel: string;
  candidateBefore: number | null;
  candidateAfter: number | null;
}

function numberOrNull(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function buildExecutiveNarrative(trace: RuntimeTraceV1 | undefined, t: Translate): ExecutiveNarrative {
  if (!trace) {
    return { messages: [], recoveryLabel: t("exec.noRecovery"), graphLabel: t("common.no"), candidateBefore: null, candidateAfter: null };
  }

  const messages: string[] = [];
  let recoveryLabel = t("exec.noRecovery");

  if (trace.structure_recovery.triggered) {
    recoveryLabel = t("exec.structureRecovery");
    messages.push(t("exec.structure"));
  } else if (trace.graph_recovery.graph_activated) {
    recoveryLabel = t("exec.graphRecoveryLabel");
    messages.push(t("exec.graphRecovery"));
  } else {
    messages.push(t("exec.normal"));
  }

  if (trace.graph_recovery.graph_activated && numberOrNull(trace.graph_recovery.hop_depth) !== null) {
    messages.push(t("exec.oneHop", { hop: trace.graph_recovery.hop_depth ?? 0 }));
  }

  if (trace.trace.status === "refused" || trace.outcome.refused) {
    messages.push(t("exec.refused"));
  } else if (trace.trace.status === "failed") {
    messages.push(t("exec.failed"));
  } else if (trace.trace.status === "completed") {
    messages.push(t("exec.completed"));
  }

  const candidateBefore = numberOrNull(trace.rerank.input_candidate_count ?? trace.retrieval.rerank_input_count);
  const candidateAfter = numberOrNull(trace.rerank.output_candidate_count ?? trace.retrieval.final_result_count);
  const graphLabel = trace.graph_recovery.graph_activated
    ? t("exec.oneHopLabel", { hop: trace.graph_recovery.hop_depth ?? "?" })
    : t("exec.noRecovery");

  return { messages, recoveryLabel, graphLabel, candidateBefore, candidateAfter };
}
