import { usePresentation, localizeStatus } from "../i18n";
import { formatMs, present } from "../lib/format";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { Field, Panel, Pill } from "./common";

export function AnswerCitationPanel({ trace }: { trace?: RuntimeTraceV1 }) {
  const { t } = usePresentation();
  const answerability = trace?.answerability;
  const generation = trace?.generation;
  const grounding = trace?.grounding;
  const citation = trace?.citation;
  const bool = (value: boolean | null | undefined) => value === true ? t("common.yes") : value === false ? t("common.no") : t("common.unavailable");
  return (
    <Panel title={t("answer.title")} right={<Pill tone={trace?.outcome.status}>{localizeStatus(trace?.outcome.status, t)}</Pill>}>
      <div className="field-grid">
        <Field label={t("answer.answerability")} value={present(answerability?.answerability_state, t("common.unavailable"))} />
        <Field label={t("answer.answerable")} value={bool(answerability?.answerable)} />
        <Field label={t("answer.generationAttempted")} value={bool(generation?.generation_attempted)} />
        <Field label={t("answer.generationLatency")} value={formatMs(generation?.generation_latency_ms)} mono />
        <Field label={t("answer.abstained")} value={bool(generation?.generation_abstained)} />
        <Field label={t("answer.grounding")} value={bool(grounding?.grounding_passed)} />
        <Field label={t("answer.citationValid")} value={bool(citation?.citation_valid)} />
        <Field label={t("answer.citationCount")} value={present(citation?.citation_count, "--")} mono />
      </div>
    </Panel>
  );
}
