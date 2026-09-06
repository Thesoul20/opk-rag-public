import { usePresentation, localizeStatus } from "../i18n";
import { present } from "../lib/format";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { Field, Panel, Pill } from "./common";

export function RetrievalTracePanel({ trace }: { trace?: RuntimeTraceV1 }) {
  const { t } = usePresentation();
  const retrieval = trace?.retrieval;
  const structure = trace?.structure_recovery;
  return (
    <Panel title={t("retrieval.title")} right={<Pill tone={retrieval?.stage_state}>{localizeStatus(retrieval?.stage_state, t)}</Pill>}>
      <div className="field-grid">
        <Field label={t("retrieval.policy")} value={present(retrieval?.executed_policy, t("common.unavailable"))} />
        <Field label={t("retrieval.retrievers")} value={present(retrieval?.retrievers_used, t("common.unavailable"))} />
        <Field label={t("retrieval.initial")} value={present(retrieval?.initial_candidate_count, "--")} mono />
        <Field label={t("retrieval.final")} value={present(retrieval?.final_result_count, "--")} mono />
        <Field label={t("retrieval.structure")} value={<Pill tone={structure?.stage_state}>{localizeStatus(structure?.stage_state, t)}</Pill>} />
        <Field label={t("retrieval.structureExpanded")} value={present(structure?.expanded_candidate_count, "--")} mono />
      </div>
      <div className="mini-table" role="table" aria-label={t("candidate.candidates")}>
        {(retrieval?.candidates || []).slice(0, 6).map((candidate) => <div className="mini-row" role="row" key={candidate.candidate_id || candidate.chunk_id}>
          <span className="mono">{candidate.final_rank ?? "--"}</span><span>{present(candidate.document_path, t("common.unavailable"))}</span><span>{present(candidate.retrieval_sources, t("common.unavailable"))}</span>
        </div>)}
      </div>
    </Panel>
  );
}
