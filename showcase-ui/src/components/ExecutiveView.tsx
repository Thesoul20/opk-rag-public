import { ArrowRight, CheckCircle2, GitBranch, ShieldCheck } from "lucide-react";
import { usePresentation } from "../i18n";
import { formatMs, formatScore, present } from "../lib/format";
import { buildExecutiveNarrative } from "../presentation/narrative";
import type { RuntimeTraceV1, TraceStage } from "../types/runtimeTrace";
import { localizeStatus, stageTranslationKey } from "../i18n";
import { Field, Panel, Pill } from "./common";

const executiveStages: TraceStage[] = ["query", "retrieval", "structure_recovery", "graph_recovery", "rerank", "evidence", "answerability", "grounding", "citation", "outcome"];

function stageState(trace: RuntimeTraceV1, stage: TraceStage): string | undefined {
  if (stage === "outcome") return trace.outcome.status;
  const section = trace[stage] as { stage_state?: string };
  return section?.stage_state;
}

export function ExecutiveView({ trace, compact = false }: { trace?: RuntimeTraceV1; compact?: boolean }) {
  const { t } = usePresentation();
  const narrative = buildExecutiveNarrative(trace, t);

  if (!trace) {
    return <section className="executive-empty"><ShieldCheck aria-hidden="true" size={28} /><h2>{t("exec.title")}</h2><p>{t("exec.noTrace")}</p></section>;
  }

  const evidence = trace.evidence.evidence_items || [];
  const citations = trace.citation.citations || [];
  const resultTone = trace.trace.status;
  const candidateFlow = narrative.candidateBefore !== null && narrative.candidateAfter !== null
    ? t("exec.candidateFlow", { before: narrative.candidateBefore, after: narrative.candidateAfter })
    : t("common.unavailable");

  return (
    <div className="executive-view">
      <section className="executive-hero">
        <div>
          <span className="eyebrow">EXECUTIVE VIEW</span>
          <h2>{t("exec.title")}</h2>
          <p>{t("exec.subtitle")}</p>
        </div>
        <Pill tone={resultTone}>{localizeStatus(resultTone, t)}</Pill>
      </section>

      <Panel title={t("exec.summary")} right={<span className="mono subtle">{present(trace.query.scenario_id, t("trace.custom"))}</span>}>
        <div className="executive-kpis">
          <Field label={t("exec.result")} value={localizeStatus(trace.trace.status, t)} />
          <Field label={t("exec.recovery")} value={narrative.recoveryLabel} />
          <Field label={t("exec.graph")} value={narrative.graphLabel} />
          <Field label={t("exec.candidates")} value={candidateFlow} mono />
          <Field label={t("exec.evidenceCount")} value={present(trace.evidence.evidence_count, "--")} mono />
          <Field label={t("exec.citations")} value={present(trace.citation.citation_count, "--")} mono />
          <Field label={t("exec.latency")} value={formatMs(trace.timings.total_ms)} mono />
        </div>
      </Panel>

      <div className="executive-two-column">
        <Panel title={t("exec.whatHappened")} right={<GitBranch size={16} aria-hidden="true" />}>
          <div className="narrative-list">
            {narrative.messages.map((message, index) => (
              <div className="narrative-item" key={`${index}-${message}`}>
                <span className="narrative-index">{index + 1}</span>
                <p>{message}</p>
              </div>
            ))}
          </div>
          {trace.grounding.grounding_passed && <p className="executive-assurance"><CheckCircle2 size={16} aria-hidden="true" />{t("exec.groundingPassed")}</p>}
          {(trace.trace.status === "refused" || trace.outcome.refused) && <p className="executive-assurance refused"><ShieldCheck size={16} aria-hidden="true" />{t("exec.safeRefusal")}</p>}
        </Panel>

        <Panel title={t("exec.pipeline")}>
          <ol className="executive-pipeline">
            {executiveStages.map((stage, index) => {
              const state = stageState(trace, stage);
              const stageKey = stageTranslationKey(stage);
              return (
                <li key={stage} className={state === "skipped" || state === "not_applicable" ? "muted-step" : ""}>
                  <div><span className="pipeline-dot" /><strong>{stageKey ? t(stageKey) : stage}</strong></div>
                  <Pill tone={state}>{localizeStatus(state, t)}</Pill>
                  {index < executiveStages.length - 1 && <ArrowRight className="pipeline-arrow" size={14} aria-hidden="true" />}
                </li>
              );
            })}
          </ol>
        </Panel>
      </div>

      {!compact && <Panel title={t("exec.evidence")} right={<span className="mono subtle">{evidence.length}</span>}>
        <div className="executive-evidence-grid">
          {evidence.slice(0, 6).map((item) => {
            const cited = citations.some((citation) => citation.evidence_id === item.evidence_id || citation.source_candidate_id === item.source_candidate_id);
            return (
              <article className="evidence-card" key={item.evidence_id || item.source_candidate_id}>
                <header><strong>{item.evidence_id || t("common.unavailable")}</strong>{cited && <Pill tone="evidence">Citation</Pill>}</header>
                <Field label={t("exec.source")} value={present(item.document_path, t("common.unavailable"))} />
                <Field label={t("exec.section")} value={present(item.section_path, t("common.unavailable"))} />
                <div className="evidence-card-footer">
                  <span>{t("exec.lines")}: <b className="mono">{item.start_line ?? "--"}–{item.end_line ?? "--"}</b></span>
                  <span>{t("exec.score")}: <b className="mono">{formatScore(item.evidence_score)}</b></span>
                </div>
              </article>
            );
          })}
        </div>
      </Panel>}
    </div>
  );
}
