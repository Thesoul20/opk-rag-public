import { usePresentation, localizeStatus, stageTranslationKey } from "../i18n";
import { formatMs } from "../lib/format";
import type { EventReplayState } from "../lib/sse";
import { TRACE_STAGE_ORDER, type RuntimeTraceV1, type TraceStage } from "../types/runtimeTrace";
import { Panel, Pill } from "./common";

const timingByStage: Partial<Record<TraceStage, keyof RuntimeTraceV1["timings"]>> = {
  query: "query_preparation_ms", retrieval: "vector_search_ms", structure_recovery: "structure_expansion_ms", graph_recovery: "graph_recovery_ms", rerank: "reranking_ms", evidence: "evidence_composition_ms", generation: "generation_ms", grounding: "validation_ms",
};

export function TraceTimeline({ trace, replay }: { trace?: RuntimeTraceV1; replay: EventReplayState }) {
  const { t } = usePresentation();
  const eventStages = new Set(replay.events.map((event) => event.stage));
  return (
    <Panel title={t("trace.timeline")} right={<span className="mono subtle">{t("trace.events")} {replay.events.length}</span>}>
      <ol className="timeline" aria-label={t("trace.timeline")}>
        {TRACE_STAGE_ORDER.map((stage) => {
          const payload = trace?.[stage] as { stage_state?: string } | undefined;
          const state = payload?.stage_state || (stage === "outcome" ? trace?.outcome.status : undefined);
          const timingKey = timingByStage[stage];
          const timing = timingKey ? trace?.timings?.[timingKey] : stage === "timings" ? trace?.timings.total_ms : null;
          const stageKey = stageTranslationKey(stage);
          return <li key={stage} className={eventStages.has(stage) ? "seen" : ""}>
            <div><strong>{stageKey ? t(stageKey) : stage}</strong><span className="mono">{formatMs(typeof timing === "number" ? timing : null)}</span></div>
            <Pill tone={state}>{localizeStatus(state, t)}</Pill>
          </li>;
        })}
      </ol>
    </Panel>
  );
}
