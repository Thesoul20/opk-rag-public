import { Activity, RefreshCw } from "lucide-react";
import { usePresentation, localizeStatus } from "../i18n";
import { getApiBaseUrl } from "../lib/api";
import { present } from "../lib/format";
import { useShowcase } from "../state/ShowcaseState";
import { Pill, Field } from "./common";

function HealthPill({ ready, label }: { ready: boolean; label?: string }) {
  const { t } = usePresentation();
  return <Pill tone={ready ? "completed" : "unavailable"}>{label || (ready ? t("status.ready") : t("status.unavailable"))}</Pill>;
}

export function RuntimeStatus() {
  const { state, actions } = useShowcase();
  const { t } = usePresentation();
  return (
    <section className="status-strip" aria-label={t("runtime.status")}>
      <div className="brand-block">
        <Activity aria-hidden="true" />
        <div>
          <h1>{t("app.title")}</h1>
          <p>{t("app.subtitle")}</p>
        </div>
      </div>
      <div className="status-grid">
        <Field label={t("runtime.lifecycle")} value={<Pill tone={state.lifecycle}>{localizeStatus(state.lifecycle, t)}</Pill>} />
        <Field label={t("runtime.api")} value={<HealthPill ready={!state.error && state.health?.api_ready === true} label={state.error ? t("status.unavailable") : state.health?.api_ready === true ? t("status.ready") : t("status.checking")} />} />
        <Field label="Qdrant" value={<HealthPill ready={state.health?.backend_reachable === true} />} />
        <Field label={t("runtime.knowledgeBase")} value={<HealthPill ready={state.health?.knowledge_base_available === true} />} />
        <Field label={t("runtime.embedding")} value={<HealthPill ready={state.health?.embedding_available === true} />} />
        <Field label={t("runtime.reranker")} value={<HealthPill ready={state.health?.reranker_available === true} />} />
        <Field label={t("runtime.generation")} value={<HealthPill ready={state.health?.generation_available === true} />} />
        <Field label={t("runtime.vectorBackend")} value={present(state.runtime?.vector_backend, t("common.unavailable"))} />
        <Field label={t("runtime.retrievalPolicy")} value={present(state.runtime?.initial_retrieval_policy, t("common.unavailable"))} />
        <Field label={t("runtime.graphHop")} value={present(state.runtime?.graph_hop_depth, "--")} mono />
        <Field label={t("runtime.recoveryMax")} value={present(state.runtime?.maximum_recovery_attempt_count, "--")} mono />
        <Field label={t("runtime.apiUrl")} value={getApiBaseUrl()} mono />
      </div>
      <button className="icon-button" type="button" onClick={() => void actions.refresh()} aria-label={t("runtime.refresh")}>
        <RefreshCw aria-hidden="true" size={18} />
      </button>
    </section>
  );
}
