import { ArrowRight, Circle, GitBranch, ShieldCheck } from "lucide-react";
import { localizeStatus, usePresentation, type TranslationKey } from "../i18n";
import { buildEndToEndModel } from "../presentation/endToEnd";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { Panel, Pill } from "./common";

const stageKeys: Record<string, TranslationKey> = {
  query: "e2e.stage.query",
  retrieval: "e2e.stage.retrieval",
  guard: "e2e.stage.guard",
  recovery: "e2e.stage.recovery",
  rerank: "e2e.stage.rerank",
  evidence: "e2e.stage.evidence",
  answer: "e2e.stage.answer",
  outcome: "e2e.stage.outcome",
};

function recoveryKey(value: string): TranslationKey {
  return `e2e.recovery.${value}` as TranslationKey;
}

export function EndToEndOverview({ trace }: { trace?: RuntimeTraceV1 }) {
  const { t, recordingMode } = usePresentation();
  const model = buildEndToEndModel(trace);
  if (!trace || !model) return <Panel title={t("e2e.title")}><p className="empty">{t("e2e.noTrace")}</p></Panel>;
  return (
    <section id="runtime-overview" className={`e2e-overview ${recordingMode ? "recording" : ""}`} data-testid="end-to-end-overview">
      <div className="e2e-heading">
        <div><span className="eyebrow">{t("e2e.eyebrow")}</span><h2>{t("e2e.title")}</h2><p>{trace.query.query_text}</p></div>
        <div className="e2e-outcome"><Pill tone={model.outcomeStatus}>{localizeStatus(model.outcomeStatus, t)}</Pill><small>{model.scenarioId || t("trace.custom")} · {model.executionScope || "--"}</small></div>
      </div>
      <div className="e2e-stage-flow" aria-label={t("e2e.pipelineLabel")}>
        {model.stages.map((stage, index) => <div className="e2e-stage-wrap" key={stage.id}>
          <article className={`e2e-stage state-${stage.state}`}>
            <Circle size={10} aria-hidden="true" /><span>{t(stageKeys[stage.id])}</span>
            {stage.id === "recovery" ? <strong>{t(recoveryKey(model.recovery))}</strong> : stage.id === "evidence" ? <strong>{model.evidenceCount ?? "--"}</strong> : stage.id === "rerank" ? <strong>{model.rerankInput ?? "--"} → {model.rerankOutput ?? "--"}</strong> : <strong>{stage.value ?? localizeStatus(stage.state, t)}</strong>}
          </article>
          {index < model.stages.length - 1 && <ArrowRight size={15} className="e2e-arrow" aria-hidden="true" />}
        </div>)}
      </div>
      <div className="e2e-takeaway">
        <GitBranch size={17} aria-hidden="true" /><span>{t("e2e.recoveryLabel")}</span><strong>{t(recoveryKey(model.recovery))}{model.recovery === "graph" && model.graphHop !== null ? ` · ${model.graphHop} ${t("e2e.hop")}` : ""}</strong>
        <ShieldCheck size={17} aria-hidden="true" /><span>{t("e2e.outcomeLabel")}</span><strong className={model.refused ? "refused-text" : ""}>{localizeStatus(model.outcomeStatus, t)}</strong>
      </div>
    </section>
  );
}
