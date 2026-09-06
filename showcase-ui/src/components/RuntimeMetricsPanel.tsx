import { usePresentation, localizeStatus } from "../i18n";
import { formatMs, present } from "../lib/format";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { Field, Panel, Pill } from "./common";

export function RuntimeMetricsPanel({ trace }: { trace?: RuntimeTraceV1 }) {
  const { t } = usePresentation();
  const timings = trace?.timings;
  const runtime = trace?.runtime;
  return (
    <Panel title={t("metrics.title")} right={<Pill tone={timings?.stage_state}>{localizeStatus(timings?.stage_state, t)}</Pill>}>
      <div className="field-grid">
        <Field label={t("metrics.total")} value={formatMs(timings?.total_ms)} mono />
        <Field label={t("metrics.embedding")} value={formatMs(timings?.embedding_ms)} mono />
        <Field label={t("metrics.vectorSearch")} value={formatMs(timings?.vector_search_ms)} mono />
        <Field label={t("metrics.graphRecovery")} value={formatMs(timings?.graph_recovery_ms)} mono />
        <Field label={t("metrics.reranking")} value={formatMs(timings?.reranking_ms)} mono />
        <Field label={t("metrics.evidence")} value={formatMs(timings?.evidence_composition_ms)} mono />
        <Field label={t("metrics.vectorBackend")} value={present(runtime?.vector_backend, t("common.unavailable"))} />
        <Field label={t("metrics.rerankerPrecision")} value={present(runtime?.reranker_precision, t("common.unavailable"))} />
      </div>
    </Panel>
  );
}
