import { useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowRight, CheckCircle2, FileCheck2, Link2, ShieldCheck, ShieldX, Sparkles } from "lucide-react";
import { localizeStatus, usePresentation, type TranslationKey } from "../i18n";
import type { EventReplayState } from "../lib/sse";
import { formatMs, present } from "../lib/format";
import { buildAnswerValidationModel, type AnswerStageId, type CitationLinkModel } from "../presentation/answerValidation";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { Field, Panel, Pill } from "./common";

const stageKeys: Record<AnswerStageId, TranslationKey> = {
  evidence: "answerViz.stage.evidence",
  answerability: "answerViz.stage.answerability",
  generation: "answerViz.stage.generation",
  grounding: "answerViz.stage.grounding",
  citation: "answerViz.stage.citation",
  outcome: "answerViz.stage.outcome",
};

function boolLabel(value: boolean | null, yes: string, no: string, unavailable: string): string {
  return value === true ? yes : value === false ? no : unavailable;
}

export function AnswerValidationPanel({ trace, replay, compact = false }: { trace?: RuntimeTraceV1; replay?: EventReplayState; compact?: boolean }) {
  const { t, viewMode } = usePresentation();
  const model = useMemo(() => buildAnswerValidationModel(trace, t), [trace, t]);
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);

  useEffect(() => setSelectedEvidenceId(null), [trace?.trace.trace_id]);
  useEffect(() => {
    if (!model || typeof window === "undefined" || window.location.hash !== "#answer-validation") return;
    window.requestAnimationFrame(() => document.getElementById("answer-validation")?.scrollIntoView({ block: "start" }));
  }, [model, trace?.trace.trace_id]);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (event: Event) => {
      const evidenceId = (event as CustomEvent<{ evidenceId?: string }>).detail?.evidenceId;
      if (evidenceId) setSelectedEvidenceId(evidenceId);
    };
    window.addEventListener("opk-showcase-evidence-selected", handler);
    return () => window.removeEventListener("opk-showcase-evidence-selected", handler);
  }, []);

  if (!trace || !model) return <Panel title={t("answerViz.title")}><p className="empty">{t("answerViz.noTrace")}</p></Panel>;

  const seen = new Set(replay?.events.map((event) => event.stage) || []);
  const focusCitation = (citation: CitationLinkModel) => {
    setSelectedEvidenceId(citation.evidenceId);
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("opk-showcase-evidence-focus", { detail: { evidenceId: citation.evidenceId, candidateId: citation.sourceCandidateId } }));
    }
  };
  const outcomeTone = model.outcome.status;

  return (
    <Panel title={t("answerViz.title")} right={<span className="pipeline-authority"><ShieldCheck size={14}/>{t("pipeline.authority")}</span>}>
      <section id="answer-validation" className={`answer-validation-visual ${compact ? "compact" : ""}`} data-testid="answer-validation-pipeline">
        <div className="answer-stage-flow" aria-label={t("answerViz.pipelineLabel")}>
          {model.stages.map((stage, index) => <div className="answer-stage-wrap" key={stage.id}>
            <div className={`answer-stage-card state-${stage.state} ${seen.has(stage.id) ? "observed" : ""}`}>
              <span>{t(stageKeys[stage.id])}</span>
              <strong>{stage.id === "outcome" ? localizeStatus(stage.state, t) : stage.detail}</strong>
              <Pill tone={stage.state}>{localizeStatus(stage.state, t)}</Pill>
            </div>
            {index < model.stages.length - 1 && <ArrowRight size={16} className="answer-stage-arrow"/>}
          </div>)}
        </div>

        {model.searchOnly && <div className="answer-scope-note"><FileCheck2 size={20}/><div><strong>{t("answerViz.searchScopeTitle")}</strong><p>{model.outcome.explanation}</p></div></div>}

        {!model.searchOnly && viewMode === "executive" && <div className={`answer-executive-story ${model.outcome.refused ? "refused" : "completed"}`}>
          {model.outcome.refused ? <ShieldX size={24}/> : <ShieldCheck size={24}/>}<div><strong>{model.outcome.refused ? t("answerViz.safeRefusal") : t("answerViz.validatedOutcome")}</strong><p>{model.outcome.explanation}</p></div>
          <div className="answer-exec-funnel"><span>{model.evidenceCount ?? "--"}<small>{t("answerViz.evidenceItems")}</small></span><ArrowDown size={14}/><span>{model.answerability.state || "--"}<small>{t("answerViz.answerabilityShort")}</small></span><ArrowDown size={14}/><span>{model.citation.count ?? "--"}<small>{t("answerViz.citationsShort")}</small></span></div>
        </div>}

        {!compact && !model.searchOnly && <div className="answer-validation-grid">
          <article className="answer-detail-card">
            <header><CheckCircle2 size={17}/><strong>{t("answerViz.answerability")}</strong></header>
            <Field label={t("answerViz.state")} value={present(model.answerability.state, t("common.unavailable"))}/>
            <Field label={t("answerViz.answerable")} value={boolLabel(model.answerability.answerable, t("common.yes"), t("common.no"), t("common.unavailable"))}/>
            <Field label={t("answerViz.reasonCode")} value={present(model.answerability.reasonCode, t("common.unavailable"))} mono/>
            <Field label={t("answerViz.evidenceConsidered")} value={present(model.answerability.consideredEvidenceCount, "--")} mono/>
            <Field label={t("answerViz.runtimeConfidence")} value={model.answerability.confidence === null ? "--" : model.answerability.confidence.toFixed(4)} mono/>
          </article>

          <article className="answer-detail-card generation-card">
            <header><Sparkles size={17}/><strong>{t("answerViz.generation")}</strong></header>
            <Field label={t("answerViz.provider")} value={present(model.generation.provider, t("common.unavailable"))}/>
            <Field label={t("answerViz.model")} value={present(model.generation.model, t("common.unavailable"))}/>
            <Field label={t("answerViz.endpoint")} value={present(model.generation.endpointType, t("common.unavailable"))}/>
            <Field label={t("answerViz.attempted")} value={boolLabel(model.generation.attempted,t("common.yes"),t("common.no"),t("common.unavailable"))}/>
            <Field label={t("answerViz.abstainedLabel")} value={boolLabel(model.generation.abstained,t("common.yes"),t("common.no"),t("common.unavailable"))}/>
            <Field label={t("answerViz.latency")} value={formatMs(model.generation.latencyMs)} mono/>
            <Field label={t("answerViz.tokens")} value={model.generation.totalTokens === null ? "--" : String(model.generation.totalTokens)} mono/>
          </article>

          <article className="answer-detail-card grounding-card">
            <header><ShieldCheck size={17}/><strong>{t("answerViz.grounding")}</strong></header>
            <Field label={t("answerViz.checked")} value={boolLabel(model.grounding.checked,t("common.yes"),t("common.no"),t("common.unavailable"))}/>
            <Field label={t("answerViz.passed")} value={boolLabel(model.grounding.passed,t("common.yes"),t("common.no"),t("common.unavailable"))}/>
            <Field label={t("answerViz.status")} value={present(model.grounding.status,t("common.unavailable"))}/>
            <Field label={t("answerViz.reasonCode")} value={present(model.grounding.reasonCode,t("common.unavailable"))} mono/>
            <Field label={t("answerViz.unsupportedClaims")} value={present(model.grounding.unsupportedClaimCount,"--")} mono/>
            <Field label={t("answerViz.citationCoverage")} value={model.grounding.citationCoverage === null ? "--" : `${(model.grounding.citationCoverage*100).toFixed(0)}%`} mono/>
          </article>

          <article className={`answer-detail-card outcome-card ${model.outcome.refused ? "refused" : ""}`}>
            <header>{model.outcome.refused ? <ShieldX size={17}/> : <ShieldCheck size={17}/>}<strong>{t("answerViz.finalOutcome")}</strong></header>
            <Field label={t("answerViz.status")} value={localizeStatus(model.outcome.status,t)}/>
            <Field label={t("answerViz.answerStatus")} value={present(model.outcome.answerStatus,t("common.unavailable"))}/>
            <Field label={t("answerViz.failureStage")} value={present(model.outcome.failureStage,t("common.none"))}/>
            <Field label={t("answerViz.refusalReason")} value={present(model.outcome.refusalReasonCode,t("common.none"))} mono/>
            <p>{model.outcome.explanation}</p>
          </article>
        </div>}

        {!model.searchOnly && <div className="answer-boundary-note"><Sparkles size={16}/><div><strong>{t("answerViz.generatedNotFinal")}</strong><p>{t("answerViz.generatedNotFinalExplanation")}</p></div></div>}

        {!model.searchOnly && <div className="answer-citation-section">
          <div className="answer-citation-head"><div><span className="section-kicker">{t("answerViz.citationValidation")}</span><h3>{t("answerViz.citations")}</h3></div><div><Pill tone={model.citation.valid === true ? "evidence" : model.citation.valid === false ? "failed" : "unavailable"}>{boolLabel(model.citation.valid,t("common.yes"),t("common.no"),t("common.unavailable"))}</Pill><strong className="mono">{model.citation.count ?? "--"}</strong></div></div>
          {model.citation.links.length ? <div className="answer-citation-grid">{model.citation.links.map((citation) => <button type="button" key={`${citation.citationId}-${citation.evidenceId}`} className={`answer-citation-card ${selectedEvidenceId === citation.evidenceId ? "selected" : ""}`} onClick={() => focusCitation(citation)}>
            <header><Link2 size={15}/><strong>[{citation.citationId}]</strong>{citation.graphRecovered && <span className="source-tag source-graph">{t("candidateViz.source.graph")}</span>}</header>
            <p>{present(citation.documentPath,t("common.unavailable"))}</p>
            <small>{t("answerViz.evidenceId")}: <b>{citation.evidenceId}</b> · {t("candidateViz.lines")} {citation.startLine ?? "--"}–{citation.endLine ?? "--"}</small>
            <code>{citation.sourceCandidateId || "--"}</code>
          </button>)}</div> : <div className="answer-no-citations"><Link2 size={18}/><span>{t("answerViz.noCitations")}</span></div>}
        </div>}

        <div className={`answer-final-card ${model.outcome.refused ? "refused" : "completed"}`}>
          <div>{model.outcome.refused ? <ShieldX size={22}/> : <ShieldCheck size={22}/>}<div><span>{t("answerViz.finalAnswer")}</span><strong>{model.outcome.refused ? t("answerViz.safeRefusal") : present(model.outcome.answerStatus, localizeStatus(model.outcome.status,t))}</strong></div></div>
          <p>{model.finalAnswerBodyAvailable ? "" : t("answerViz.answerBodyUnavailable")}</p>
        </div>
      </section>
    </Panel>
  );
}
