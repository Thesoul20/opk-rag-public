import { usePresentation, localizeStatus } from "../i18n";
import { present } from "../lib/format";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { Field, Panel, Pill } from "./common";

export function GuardDecisionPanel({ trace }: { trace?: RuntimeTraceV1 }) {
  const { t } = usePresentation();
  const guard = trace?.guard;
  return (
    <Panel title={t("guard.title")} right={<Pill tone={guard?.stage_state}>{localizeStatus(guard?.stage_state, t)}</Pill>}>
      <div className="field-grid">
        <Field label={t("guard.route")} value={present(guard?.selected_route, t("common.unavailable"))} />
        <Field label={t("guard.initial")} value={present(guard?.initial_decision, t("common.unavailable"))} />
        <Field label={t("guard.final")} value={present(guard?.final_decision, t("common.unavailable"))} />
        <Field label={t("guard.recovery")} value={present(guard?.recovery_action, t("common.none"))} />
        <Field label={t("guard.attempts")} value={`${present(guard?.recovery_attempt_count, "--")} / ${present(guard?.maximum_recovery_attempt_count, "--")}`} mono />
        <Field label={t("guard.refusal")} value={guard?.fail_closed ? <Pill tone="refused">{t("guard.governedRefusal")}</Pill> : t("guard.notRefused")} />
        <Field label={t("guard.refusalReason")} value={present(guard?.refusal_reason_code, t("common.unavailable"))} />
        <Field label={t("guard.hiddenReasoning")} value={guard?.hidden_chain_of_thought_exposed ? "exposed" : t("guard.notExposed")} />
      </div>
    </Panel>
  );
}
