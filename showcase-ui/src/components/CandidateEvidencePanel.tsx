import { usePresentation, localizeStatus } from "../i18n";
import { formatScore, present } from "../lib/format";
import type { RuntimeCandidate, RuntimeTraceV1 } from "../types/runtimeTrace";
import { Panel, Pill } from "./common";

function candidateSources(candidate: RuntimeCandidate): string[] {
  const values = new Set<string>();
  for (const source of candidate.retrieval_sources || []) values.add(source.toLowerCase());
  for (const provenance of candidate.provenance || []) {
    if (provenance.source_kind === "structure_expansion") values.add("structure");
    else if (provenance.source_kind === "graph_recovery") values.add("graph");
    else if (provenance.source_lane) values.add(provenance.source_lane.toLowerCase());
  }
  if (candidate.structure_expanded) values.add("structure");
  if (candidate.graph_recovered) values.add("graph");
  return [...values];
}

export function CandidateEvidencePanel({ trace }: { trace?: RuntimeTraceV1 }) {
  const { t } = usePresentation();
  const candidates = trace?.retrieval.candidates || [];
  const evidence = trace?.evidence.evidence_items || [];
  const evidenceCandidateIds = new Set(evidence.map((row) => row.source_candidate_id));
  const sourceLabel = (source: string) => {
    if (source.includes("vector")) return t("candidate.source.vector");
    if (source.includes("lexical") || source.includes("keyword")) return t("candidate.source.lexical");
    if (source.includes("structure")) return t("candidate.source.structure");
    if (source.includes("graph")) return t("candidate.source.graph");
    return source;
  };

  return (
    <Panel title={t("candidate.title")} right={<Pill tone={trace?.evidence.stage_state}>{localizeStatus(trace?.evidence.stage_state, t)}</Pill>}>
      <div className="split-table">
        <div><h3>{t("candidate.candidates")}</h3>{candidates.slice(0, 5).map((candidate) => {
          const sources = candidateSources(candidate);
          return <div className="candidate-record" key={candidate.candidate_id || candidate.chunk_id}>
            <div className="record-row"><span className="mono">{candidate.final_rank ?? candidate.initial_rank ?? "--"}</span><span>{present(candidate.document_path, t("common.unavailable"))}</span><Pill tone={evidenceCandidateIds.has(candidate.candidate_id) ? "evidence" : "unavailable"}>{evidenceCandidateIds.has(candidate.candidate_id) ? t("candidate.selected") : t("candidate.candidate")}</Pill></div>
            <div className="provenance-row"><span>{t("candidate.provenance")}</span>{sources.length ? sources.map((source) => <span className={`source-tag source-${source}`} key={source}>{sourceLabel(source)}</span>) : <span className="source-tag">--</span>}</div>
          </div>;
        })}</div>
        <div><h3>{t("candidate.evidence")}</h3>{evidence.slice(0, 5).map((item) => <div className="record-row" key={item.evidence_id || item.source_candidate_id}>
          <span className="mono">{item.evidence_id || t("common.unavailable")}</span><span>{present(item.document_path, t("common.unavailable"))}</span><span className="mono">{formatScore(item.evidence_score)}</span>
        </div>)}</div>
      </div>
      <p className="note">{t("candidate.note")}</p>
    </Panel>
  );
}
