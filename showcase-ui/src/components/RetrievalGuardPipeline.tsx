import { ArrowRight, Check, Circle, GitBranch, RotateCcw, ShieldCheck, X } from "lucide-react";
import { localizeStatus, usePresentation, type TranslationKey } from "../i18n";
import type { EventReplayState } from "../lib/sse";
import { buildRetrievalGuardModel, type GuardActionId } from "../presentation/retrievalGuard";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { Panel, Pill } from "./common";

const actions: Array<{ id: GuardActionId; labelKey: TranslationKey }> = [
  { id: "continue", labelKey: "pipeline.action.continue" },
  { id: "structure", labelKey: "pipeline.action.structure" },
  { id: "graph", labelKey: "pipeline.action.graph" },
  { id: "fail_closed", labelKey: "pipeline.action.failClosed" },
];

function StateIcon({ state }: { state: string }) {
  if (state === "completed") return <Check size={15} aria-hidden="true" />;
  if (state === "failed") return <X size={15} aria-hidden="true" />;
  if (state === "active" || state === "running") return <RotateCcw size={15} aria-hidden="true" />;
  return <Circle size={12} aria-hidden="true" />;
}

export function RetrievalGuardPipeline({ trace, replay, compact = false }: { trace?: RuntimeTraceV1; replay: EventReplayState; compact?: boolean }) {
  const { t, viewMode } = usePresentation();
  const model = buildRetrievalGuardModel(trace, t);
  if (!trace || !model) {
    return <Panel title={t("pipeline.title")}><p className="empty">{t("pipeline.empty")}</p></Panel>;
  }

  const seenStages = new Set(replay.events.map((event) => event.stage));
  const coreNodes = model.nodes.filter((node) => !compact || ["retrieval", "guard", "structure", "graph", "rerank", "evidence", "outcome"].includes(node.id));

  return (
    <Panel title={t("pipeline.title")} right={<span className="pipeline-authority"><ShieldCheck size={14} />{t("pipeline.authority")}</span>}>
      <div className={`retrieval-guard-visual ${compact ? "compact" : ""}`} data-testid="retrieval-guard-pipeline">
        <div className="pipeline-flow" aria-label={t("pipeline.title")}>
          {coreNodes.map((node, index) => {
            const observed = node.stage ? seenStages.has(node.stage) : false;
            const skipped = node.state === "skipped" || node.state === "not_applicable";
            return <div className="pipeline-node-wrap" key={node.id}>
              <article className={`pipeline-node state-${node.state} ${observed ? "observed" : ""} ${skipped ? "skipped" : ""}`} data-node-id={node.id}>
                <header><span className="pipeline-state-icon"><StateIcon state={node.state} /></span><strong>{t(node.labelKey)}</strong></header>
                {node.primary && <p>{node.primary}</p>}
                {node.secondary && <small>{node.secondary}</small>}
                <Pill tone={node.state}>{localizeStatus(node.state, t)}</Pill>
              </article>
              {index < coreNodes.length - 1 && <ArrowRight className="pipeline-connector" size={18} aria-hidden="true" />}
            </div>;
          })}
        </div>

        <div className="guard-visual-grid">
          <section className="guard-core-card">
            <div className="guard-core-heading"><GitBranch size={18} aria-hidden="true" /><div><span>{t("pipeline.guardDecision")}</span><strong>{trace.guard.selected_route || t("common.unavailable")}</strong></div></div>
            <p>{model.reasonExplanation}</p>
            {viewMode === "engineer" && <code>{model.reasonCode || "--"}</code>}
          </section>

          <section className="guard-actions" aria-label={t("pipeline.actionSpace")}>
            <span className="section-kicker">{t("pipeline.actionSpace")}</span>
            <div>{actions.map((action) => <span key={action.id} className={`guard-action ${model.selectedAction === action.id ? "selected" : ""}`} data-action={action.id}>{t(action.labelKey)}</span>)}</div>
          </section>

          <section className="recovery-budget">
            <span className="section-kicker">{t("pipeline.recoveryBudget")}</span>
            <strong className="mono">{model.recoveryAttempts ?? "--"} / {model.recoveryMaximum ?? "--"}</strong>
            <div className="budget-track"><span style={{ width: model.recoveryMaximum && model.recoveryAttempts !== null ? `${Math.min(100, (model.recoveryAttempts / model.recoveryMaximum) * 100)}%` : "0%" }} /></div>
            <small>{t("pipeline.boundedRecovery")}</small>
          </section>
        </div>

        <div className="pipeline-facts">
          <div><span>{t("pipeline.retrievers")}</span><strong>{(trace.retrieval.retrievers_used || []).join(" + ") || t("common.unavailable")}</strong></div>
          <div><span>{t("pipeline.requestedTopK")}</span><strong className="mono">{trace.query.requested_top_k ?? "--"}</strong></div>
          <div><span>{t("pipeline.candidateK")}</span><strong className="mono">{trace.query.candidate_k ?? "--"}</strong></div>
          <div><span>{t("pipeline.rerankerModel")}</span><strong>{trace.rerank.reranker_model || trace.runtime.reranker_model || t("common.unavailable")}</strong></div>
          <div><span>{t("pipeline.precision")}</span><strong className="mono">{trace.rerank.precision || trace.runtime.reranker_precision || t("common.unavailable")}</strong></div>
          <div><span>{t("pipeline.scope")}</span><strong>{trace.rerank.execution_scope || trace.query.execution_scope || t("common.unavailable")}</strong></div>
          <div><span>{t("pipeline.graphHop")}</span><strong className="mono">{model.graphHop ?? "--"}</strong></div>
        </div>

        {trace.graph_recovery.graph_activated && <a className="view-graph-path" href="#graph-recovery-detail">{t("graphViz.viewPath")}</a>}

        <div className="candidate-flow-strip">
          <div><span>{t("pipeline.initialCandidates")}</span><strong className="mono">{model.initialCandidates ?? "--"}</strong></div>
          <ArrowRight size={15} />
          <div><span>{t("pipeline.structureExpansion")}</span><strong className="mono">{model.structureExpanded ?? "--"}</strong></div>
          <ArrowRight size={15} />
          <div><span>{t("pipeline.graphRecovered")}</span><strong className="mono">{model.graphRecovered ?? "--"}</strong></div>
          <ArrowRight size={15} />
          <div><span>{t("pipeline.rerankInput")}</span><strong className="mono">{model.rerankInput ?? "--"}</strong></div>
          <ArrowRight size={15} />
          <div><span>{t("pipeline.rerankOutput")}</span><strong className="mono">{model.rerankOutput ?? "--"}</strong></div>
          <ArrowRight size={15} />
          <div><span>{t("pipeline.evidence")}</span><strong className="mono">{model.evidenceCount ?? "--"}</strong></div>
        </div>
      </div>
    </Panel>
  );
}
