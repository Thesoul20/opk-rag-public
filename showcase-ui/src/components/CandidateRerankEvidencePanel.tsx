import { useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowRight, CheckCircle2, Filter, GitBranch, Layers3, ShieldCheck } from "lucide-react";
import { usePresentation, type TranslationKey } from "../i18n";
import type { EventReplayState } from "../lib/sse";
import { formatScore, present } from "../lib/format";
import { buildCandidateLifecycleModel, type CandidateLineageRow } from "../presentation/candidateLifecycle";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { Field, Panel, Pill } from "./common";

type FilterId = "all" | "vector" | "lexical" | "structure" | "graph" | "evidence";

const filters: Array<{ id: FilterId; key: TranslationKey }> = [
  { id: "all", key: "candidateViz.filter.all" },
  { id: "vector", key: "candidateViz.source.vector" },
  { id: "lexical", key: "candidateViz.source.lexical" },
  { id: "structure", key: "candidateViz.source.structure" },
  { id: "graph", key: "candidateViz.source.graph" },
  { id: "evidence", key: "candidateViz.filter.evidence" },
];

function sourceKey(source: string): TranslationKey | null {
  if (source === "vector") return "candidateViz.source.vector";
  if (source === "lexical") return "candidateViz.source.lexical";
  if (source === "structure") return "candidateViz.source.structure";
  if (source === "graph") return "candidateViz.source.graph";
  return null;
}

function rankDeltaLabel(row: CandidateLineageRow, t: ReturnType<typeof usePresentation>["t"]): string {
  if (row.rankDelta === null) return "--";
  if (row.rankDelta > 0) return t("candidateViz.rankUp", { delta: row.rankDelta });
  if (row.rankDelta < 0) return t("candidateViz.rankDown", { delta: Math.abs(row.rankDelta) });
  return t("candidateViz.rankSame");
}

export function CandidateRerankEvidencePanel({ trace, replay, compact = false }: { trace?: RuntimeTraceV1; replay?: EventReplayState; compact?: boolean }) {
  const { t, viewMode } = usePresentation();
  const model = useMemo(() => buildCandidateLifecycleModel(trace, t), [trace, t]);
  const [filter, setFilter] = useState<FilterId>("all");
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);

  useEffect(() => { setFilter("all"); setSelectedCandidateId(null); }, [trace?.trace.trace_id]);

  useEffect(() => {
    if (!model || typeof window === "undefined" || window.location.hash !== "#candidate-rerank-evidence") return;
    window.requestAnimationFrame(() => document.getElementById("candidate-rerank-evidence")?.scrollIntoView({ block: "start" }));
  }, [model, trace?.trace.trace_id]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (event: Event) => {
      const candidateId = (event as CustomEvent<{ candidateId?: string }>).detail?.candidateId;
      if (!candidateId) return;
      setSelectedCandidateId(candidateId);
    };
    window.addEventListener("opk-showcase-candidate-focus", handler);
    return () => window.removeEventListener("opk-showcase-candidate-focus", handler);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<{ evidenceId?: string; candidateId?: string | null }>).detail;
      if (!detail?.candidateId) return;
      setSelectedCandidateId(detail.candidateId);
      window.dispatchEvent(new CustomEvent("opk-showcase-candidate-selected", { detail: { candidateId: detail.candidateId } }));
    };
    window.addEventListener("opk-showcase-evidence-focus", handler);
    return () => window.removeEventListener("opk-showcase-evidence-focus", handler);
  }, []);

  if (!trace || !model) return <Panel title={t("candidateViz.title")}><p className="empty">{t("candidateViz.noTrace")}</p></Panel>;

  const seen = new Set(replay?.events.map((event) => event.stage) || []);
  const rows = model.rows.filter((row) => filter === "all" ? true : filter === "evidence" ? row.selectedAsEvidence : row.sources.includes(filter));
  const selected = model.rows.find((row) => row.candidateId === selectedCandidateId) || null;
  const focusCandidate = (candidateId: string) => {
    setSelectedCandidateId(candidateId);
    if (typeof window !== "undefined") window.dispatchEvent(new CustomEvent("opk-showcase-candidate-selected", { detail: { candidateId } }));
  };

  return (
    <Panel title={t("candidateViz.title")} right={<span className="pipeline-authority"><ShieldCheck size={14} />{t("pipeline.authority")}</span>}>
      <section id="candidate-rerank-evidence" className={`candidate-deep-visual ${compact ? "compact" : ""}`} data-testid="candidate-rerank-evidence">
        <div className="candidate-count-flow" aria-label={t("candidateViz.countFlow")}>
          <div className={seen.has("retrieval") ? "observed" : ""}><span>{t("candidateViz.initial")}</span><strong>{model.initialCandidateCount ?? "--"}</strong></div>
          <ArrowRight size={15} />
          <div><span>{t("candidateViz.structureAdded")}</span><strong>+{model.structureExpandedCount ?? "--"}</strong></div>
          <ArrowRight size={15} />
          <div><span>{t("candidateViz.graphAdded")}</span><strong>+{model.graphRecoveredCount ?? "--"}</strong></div>
          <ArrowRight size={15} />
          <div><span>{t("candidateViz.rerankInput")}</span><strong>{model.rerankInputCount ?? "--"}</strong></div>
          <ArrowRight size={15} />
          <div className={seen.has("rerank") ? "observed" : ""}><span>{t("candidateViz.rerankOutput")}</span><strong>{model.rerankOutputCount ?? "--"}</strong></div>
          <ArrowRight size={15} />
          <div className={seen.has("evidence") ? "observed" : ""}><span>{t("candidateViz.evidence")}</span><strong>{model.evidenceCount ?? "--"}</strong></div>
        </div>

        <div className="candidate-model-strip">
          <div><span>{t("candidateViz.unifiedPool")}</span><strong>{model.unifiedCandidateCount ?? "--"}</strong></div>
          <div><span>{t("candidateViz.reranker")}</span><strong>{present(model.rerankerModel, t("common.unavailable"))}</strong></div>
          <div><span>{t("candidateViz.precision")}</span><strong className="mono">{present(model.precision, t("common.unavailable"))}</strong></div>
          <div><span>{t("candidateViz.scope")}</span><strong>{present(model.executionScope, t("common.unavailable"))}</strong></div>
          <div><span>{t("candidateViz.observedCandidates")}</span><strong>{model.observedCandidateCount}</strong></div>
          <div><span>{t("candidateViz.outcome")}</span><Pill tone={model.outcomeStatus}>{model.outcomeStatus === "refused" ? t("status.refused") : model.outcomeStatus}</Pill></div>
        </div>

        {viewMode === "executive" && <div className="candidate-executive-story">
          <Layers3 size={22} aria-hidden="true" />
          <div><strong>{t("candidateViz.execQuestion")}</strong><p>{t("candidateViz.execExplanation")}</p></div>
          <div className="candidate-exec-funnel"><span>{model.rerankInputCount ?? "--"}<small>{t("candidateViz.candidatesShort")}</small></span><ArrowDown size={15}/><span>{model.rerankOutputCount ?? "--"}<small>{t("candidateViz.rerankedShort")}</small></span><ArrowDown size={15}/><span>{model.evidenceCount ?? "--"}<small>{t("candidateViz.evidenceShort")}</small></span></div>
        </div>}

        {!compact && <div className="candidate-filter-bar" aria-label={t("candidateViz.filterLabel")}><Filter size={14}/>{filters.map((item) => <button type="button" key={item.id} className={filter === item.id ? "selected" : ""} onClick={() => setFilter(item.id)}>{t(item.key)}</button>)}</div>}

        <div className="candidate-lineage-layout">
          <div className="candidate-lineage-board" role="table" aria-label={t("candidateViz.lineage")}>
            <div className="candidate-lineage-head" role="row"><span>{t("candidateViz.candidate")}</span><span>{t("candidateViz.provenance")}</span><span>{t("candidateViz.initialRank")}</span><span>{t("candidateViz.rerankRank")}</span><span>{t("candidateViz.rerankScore")}</span><span>{t("candidateViz.evidence")}</span></div>
            {rows.map((row) => <button type="button" className={`candidate-lineage-row ${selectedCandidateId === row.candidateId ? "selected" : ""}`} role="row" key={row.candidateId} data-candidate-id={row.candidateId} onClick={() => focusCandidate(row.candidateId)}>
              <span className="candidate-name"><strong>{row.documentPath?.split("/").at(-1) || row.candidateId}</strong><small className="mono">{row.candidateId.slice(0,8)}…</small></span>
              <span className="candidate-source-stack">{row.sources.length ? row.sources.map((source) => <i key={source} className={`source-tag source-${source}`}>{sourceKey(source) ? t(sourceKey(source)!) : source}</i>) : <i className="source-tag">--</i>}</span>
              <span className="rank-cell"><strong>#{row.initialRank ?? "--"}</strong><small>{rankDeltaLabel(row,t)}</small></span>
              <span className="rank-cell"><strong>#{row.rerankPosition ?? "--"}</strong><small>{row.selectedForContext ? t("candidateViz.contextSelected") : t("candidateViz.notContext")}</small></span>
              <span className="score-cell mono">{formatScore(row.rerankScore)}</span>
              <span>{row.selectedAsEvidence ? <Pill tone="evidence">{row.evidenceId || t("candidateViz.selectedEvidence")}</Pill> : <Pill tone="skipped">{t("candidateViz.notEvidence")}</Pill>}</span>
            </button>)}
          </div>

          {!compact && <aside className="candidate-inspector" aria-live="polite">
            <span className="section-kicker">{t("candidateViz.inspector")}</span>
            {selected ? <><h3>{selected.documentPath?.split("/").at(-1) || selected.candidateId}</h3><Field label={t("candidateViz.candidateId")} value={selected.candidateId} mono /><Field label={t("candidateViz.section")} value={present(selected.sectionPath, t("common.unavailable"))} /><Field label={t("candidateViz.retrievalScore")} value={formatScore(selected.retrievalScore)} mono /><Field label={t("candidateViz.rankChange")} value={`#${selected.initialRank ?? "--"} → #${selected.rerankPosition ?? "--"}`} mono /><Field label={t("candidateViz.evidenceState")} value={selected.selectedAsEvidence ? selected.evidenceId || t("candidateViz.selectedEvidence") : t("candidateViz.notEvidence")} />{selected.graphRecovered && <a href="#graph-recovery-detail" className="view-graph-path" onClick={() => { if (typeof window !== "undefined") window.dispatchEvent(new CustomEvent("opk-showcase-candidate-focus", { detail: { candidateId: selected.candidateId } })); }}>{t("candidateViz.viewGraph")}</a>}</> : <p>{t("candidateViz.inspectHint")}</p>}
          </aside>}
        </div>

        <div className="evidence-boundary"><span>{t("candidateViz.rankedCandidates")}</span><ArrowDown size={15}/><strong>{t("candidateViz.evidenceComposition")}</strong><ArrowDown size={15}/><span>{t("candidateViz.finalEvidence")}</span></div>

        <div className="candidate-evidence-cards">
          {model.evidence.map((item) => <button type="button" key={item.evidence_id || item.source_candidate_id} className={`candidate-evidence-card ${selectedCandidateId === item.source_candidate_id ? "selected" : ""}`} onClick={() => { if (item.source_candidate_id) focusCandidate(item.source_candidate_id); if (typeof window !== "undefined" && item.evidence_id) window.dispatchEvent(new CustomEvent("opk-showcase-evidence-selected", { detail: { evidenceId: item.evidence_id, candidateId: item.source_candidate_id } })); }}>
            <header><CheckCircle2 size={15}/><strong>{item.evidence_id || "--"}</strong><span className="mono">{formatScore(item.evidence_score)}</span></header>
            <p>{present(item.document_path, t("common.unavailable"))}</p>
            <small>{t("candidateViz.lines")} {item.start_line ?? "--"}–{item.end_line ?? "--"}</small>
            <code>{item.source_candidate_id || "--"}</code>
          </button>)}
        </div>

        <div className="candidate-semantic-note"><GitBranch size={16}/><div><strong>{t("candidateViz.notEqual")}</strong><p>{t("candidateViz.notEqualExplanation")}</p></div></div>
      </section>
    </Panel>
  );
}
