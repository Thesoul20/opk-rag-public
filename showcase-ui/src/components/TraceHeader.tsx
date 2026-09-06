import { usePresentation, localizeStatus } from "../i18n";
import { present } from "../lib/format";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { Field, Panel, Pill } from "./common";

export function TraceHeader({ trace }: { trace?: RuntimeTraceV1 }) {
  const { t } = usePresentation();
  return (
    <Panel title={t("trace.header")} right={trace ? <Pill tone={trace.trace.status}>{localizeStatus(trace.trace.status, t)}</Pill> : <Pill>{t("status.idle")}</Pill>}>
      {trace ? <div className="field-grid">
        <Field label={t("trace.id")} value={trace.trace.trace_id} mono />
        <Field label={t("trace.scenario")} value={present(trace.query.scenario_id, t("trace.custom"))} mono />
        <Field label={t("trace.scope")} value={present(trace.query.execution_scope, t("common.unavailable"))} />
        <Field label={t("trace.started")} value={present(trace.trace.started_at, t("common.unavailable"))} mono />
        <Field label={t("trace.completed")} value={present(trace.trace.completed_at, t("common.unavailable"))} mono />
        <Field label={t("trace.digest")} value={present(trace.trace.trace_semantic_digest, t("common.unavailable"))} mono />
      </div> : <p className="empty">{t("trace.empty")}</p>}
    </Panel>
  );
}
