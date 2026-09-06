import { usePresentation } from "../i18n";
import type { EventReplayState } from "../lib/sse";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { AnswerValidationPanel } from "./AnswerValidationPanel";
import { KnowledgeGraphPanel } from "./KnowledgeGraphPanel";
import { RetrievalGuardPipeline } from "./RetrievalGuardPipeline";

export function ScenarioSpotlight({ trace, replay }: { trace?: RuntimeTraceV1; replay: EventReplayState }) {
  const { t } = usePresentation();
  if (!trace) return null;
  const scenario = trace.query.scenario_id;
  return <section id="scenario-spotlight" className="scenario-spotlight" aria-label={t("recording.spotlight")}>
    <div className="spotlight-heading"><span className="eyebrow">{t("recording.spotlight")}</span><strong>{scenario || t("trace.custom")}</strong></div>
    {scenario === "S03" || trace.graph_recovery.graph_activated
      ? <KnowledgeGraphPanel trace={trace} replay={replay} />
      : scenario === "S04" || trace.query.execution_scope === "ask"
        ? <AnswerValidationPanel trace={trace} replay={replay} compact />
        : <RetrievalGuardPipeline trace={trace} replay={replay} compact />}
  </section>;
}
