import { useEffect, useMemo, useState } from "react";
import { ArrowRight, CheckCircle2, CircleSlash2, GitBranch, Link2, Network, ShieldCheck } from "lucide-react";
import { localizeStatus, usePresentation, type TranslationKey } from "../i18n";
import type { EventReplayState } from "../lib/sse";
import { formatScore, present } from "../lib/format";
import { buildGraphRecoveryModel, type GraphCandidateLinkModel, type GraphEdgeModel, type GraphNodeModel } from "../presentation/graphRecovery";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { Field, Panel, Pill } from "./common";

function nodeTitle(node: GraphNodeModel): string {
  return `${node.role}: ${node.id}`;
}

function roleKey(role: GraphNodeModel["role"]): TranslationKey {
  return `graphViz.role.${role}` as TranslationKey;
}

function edgeTitle(edge: GraphEdgeModel): string {
  return `${edge.relationType} · hop ${edge.hopDepth ?? "--"}\n${edge.sourceNodeId} → ${edge.targetNodeId}`;
}

export function KnowledgeGraphPanel({ trace, replay }: { trace?: RuntimeTraceV1; replay?: EventReplayState }) {
  const { t, viewMode } = usePresentation();
  const model = useMemo(() => buildGraphRecoveryModel(trace, t), [trace, t]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);

  useEffect(() => {
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    setSelectedCandidateId(null);
  }, [trace?.trace.trace_id]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (event: Event) => {
      const candidateId = (event as CustomEvent<{ candidateId?: string }>).detail?.candidateId;
      if (!candidateId || !model?.active) return;
      const candidate = model.candidates.find((row) => row.candidateId === candidateId);
      if (!candidate) return;
      setSelectedCandidateId(candidateId);
      setSelectedEdgeId(null);
      if (candidate.recoveredNodeId) setSelectedNodeId(candidate.recoveredNodeId);
    };
    window.addEventListener("opk-showcase-candidate-selected", handler);
    return () => window.removeEventListener("opk-showcase-candidate-selected", handler);
  }, [model]);

  useEffect(() => {
    if (!model?.active || typeof window === "undefined" || window.location.hash !== "#graph-recovery-detail") return;
    window.requestAnimationFrame(() => document.getElementById("graph-recovery-detail")?.scrollIntoView({ block: "start" }));
  }, [model?.active, trace?.trace.trace_id]);

  if (!trace || !model) {
    return <Panel title={t("graphViz.title")}><p className="empty">{t("graphViz.noTrace")}</p></Panel>;
  }

  const graphObserved = Boolean(replay?.events.some((event) => event.stage === "graph_recovery"));
  const selectedNode = model.nodes.find((node) => node.id === selectedNodeId);
  const selectedEdge = model.edges.find((edge) => edge.id === selectedEdgeId);
  const selectedCandidate = model.candidates.find((candidate) => candidate.candidateId === selectedCandidateId);

  const selectCandidate = (candidate: GraphCandidateLinkModel) => {
    setSelectedCandidateId(candidate.candidateId);
    setSelectedEdgeId(null);
    if (candidate.recoveredNodeId) setSelectedNodeId(candidate.recoveredNodeId);
    if (typeof window !== "undefined") window.dispatchEvent(new CustomEvent("opk-showcase-candidate-focus", { detail: { candidateId: candidate.candidateId } }));
  };

  if (!model.active) {
    return (
      <Panel title={t("graphViz.title")} right={<Pill tone={model.stageState}>{localizeStatus(model.stageState, t)}</Pill>}>
        <section id="graph-recovery-detail" className="graph-not-triggered" data-testid="graph-not-triggered">
          <CircleSlash2 aria-hidden="true" size={28} />
          <div><h3>{t("graphViz.notTriggered")}</h3><p>{model.reasonExplanation}</p></div>
        </section>
        {viewMode === "engineer" && <div className="field-grid graph-inactive-fields"><Field label={t("graph.reason")} value={present(model.reasonCode, t("common.unavailable"))} /><Field label={t("graphViz.policy")} value={present(model.activationPolicy, t("common.unavailable"))} /><Field label={t("graph.hop")} value={present(model.hopDepth, "--")} mono /><Field label={t("graphViz.resultState")} value={trace.trace.status === "refused" ? t("status.refused") : localizeStatus(trace.trace.status, t)} /></div>}
      </Panel>
    );
  }

  return (
    <Panel title={t("graphViz.title")} right={<div className="graph-title-meta"><Pill tone={model.stageState}>{localizeStatus(model.stageState, t)}</Pill><span className="one-hop-badge">{t("graphViz.oneHop", { hop: model.hopDepth ?? "--" })}</span></div>}>
      <section id="graph-recovery-detail" className={`graph-recovery-detail ${graphObserved ? "observed" : ""}`} data-testid="interactive-graph-recovery">
        <div className="graph-summary-strip">
          <div><span>{t("graphViz.seedCount")}</span><strong className="mono">{model.seedCount}</strong></div>
          <div><span>{t("graphViz.edgeCount")}</span><strong className="mono">{model.traversedEdgeCount}</strong></div>
          <div><span>{t("graphViz.recoveredNodes")}</span><strong className="mono">{model.recoveredNodeCount}</strong></div>
          <div><span>{t("graphViz.recoveredCandidates")}</span><strong className="mono">{model.recoveredCandidateCount}</strong></div>
          <div><span>{t("graphViz.evidenceLinks")}</span><strong className="mono">{model.evidenceLinkCount}</strong></div>
          <div><span>{t("graphViz.scope")}</span><strong>{t("graphViz.oneHop", { hop: model.hopDepth ?? "--" })}</strong></div>
        </div>

        <div className="graph-explanation">
          <div><Network size={20} aria-hidden="true" /><div><span>{t("graphViz.why")}</span><p>{model.reasonExplanation}</p></div></div>
          <div className="graph-safety"><ShieldCheck size={16} aria-hidden="true" /><span>{t("graphViz.noMultiHop")}</span></div>
          {viewMode === "engineer" && <code>{model.reasonCode || "--"}</code>}
        </div>

        <div className="graph-layout">
          <div className="graph-path-canvas" role="group" aria-label={t("graphViz.canvasLabel")}>
            {model.edges.map((edge) => {
              const source = model.nodes.find((node) => node.id === edge.sourceNodeId)!;
              const target = model.nodes.find((node) => node.id === edge.targetNodeId)!;
              const selected = selectedEdgeId === edge.id || selectedNodeId === source.id || selectedNodeId === target.id;
              return (
                <div className={`graph-path-row ${selected ? "selected" : ""}`} key={edge.id} data-edge-id={edge.id}>
                  <button className={`graph-entity-node role-${source.role} ${selectedNodeId === source.id ? "selected" : ""}`} type="button" title={nodeTitle(source)} onClick={() => { setSelectedNodeId(source.id); setSelectedEdgeId(null); if (source.recoveredCandidateId) setSelectedCandidateId(source.recoveredCandidateId); }}>
                    <span>{t(roleKey(source.role))}</span><strong>{source.label}</strong>
                  </button>
                  <button className={`graph-entity-edge ${selectedEdgeId === edge.id ? "selected" : ""}`} type="button" title={edgeTitle(edge)} onClick={() => { setSelectedEdgeId(edge.id); setSelectedNodeId(null); }}>
                    <span>{edge.relationType}</span><small>{t("graphViz.hop", { hop: edge.hopDepth ?? "--" })}</small><ArrowRight size={18} aria-hidden="true" />
                  </button>
                  <button className={`graph-entity-node role-${target.role} ${selectedNodeId === target.id ? "selected" : ""}`} type="button" title={nodeTitle(target)} onClick={() => { setSelectedNodeId(target.id); setSelectedEdgeId(null); if (target.recoveredCandidateId) setSelectedCandidateId(target.recoveredCandidateId); }}>
                    <span>{t(roleKey(target.role))}</span><strong>{target.label}</strong>
                  </button>
                </div>
              );
            })}
          </div>

          <aside className="graph-inspector" aria-live="polite">
            <span className="section-kicker">{t("graphViz.inspector")}</span>
            {selectedNode ? <><h3>{selectedNode.label}</h3><Field label={t("graphViz.nodeRole")} value={t(roleKey(selectedNode.role))} /><Field label={t("graphViz.documentPath")} value={selectedNode.id} mono />{selectedNode.recoveredCandidateId && <Field label={t("graphViz.candidateId")} value={selectedNode.recoveredCandidateId} mono />}</> : selectedEdge ? <><h3>{selectedEdge.relationType}</h3><Field label={t("graphViz.hopDepth")} value={present(selectedEdge.hopDepth, "--")} mono /><Field label={t("graphViz.source")} value={selectedEdge.sourceNodeId} /><Field label={t("graphViz.target")} value={selectedEdge.targetNodeId} />{viewMode === "engineer" && <Field label={t("graphViz.edgeObservation")} value={present(selectedEdge.observation, t("common.unavailable"))} />}</> : selectedCandidate ? <><h3>{t("graphViz.graphCandidate")}</h3><Field label={t("graphViz.candidateId")} value={selectedCandidate.candidateId} mono /><Field label={t("graphViz.rerankPosition")} value={present(selectedCandidate.rerankPosition, "--")} mono /><Field label={t("graphViz.rerankScore")} value={formatScore(selectedCandidate.rerankScore)} mono /><Field label={t("graphViz.evidenceState")} value={selectedCandidate.selectedAsEvidence ? t("graphViz.selectedEvidence") : t("graphViz.notEvidence")} /></> : <p>{t("graphViz.inspectHint")}</p>}
          </aside>
        </div>

        <div className="graph-candidate-chain" aria-label={t("graphViz.candidateChain")}>
          {model.candidates.map((candidate) => (
            <div className={`graph-candidate-row ${selectedCandidateId === candidate.candidateId ? "selected" : ""}`} key={candidate.candidateId} data-candidate-id={candidate.candidateId}>
              <button type="button" className="graph-chain-card recovered-candidate" onClick={() => selectCandidate(candidate)}>
                <GitBranch size={15} aria-hidden="true" /><span>{t("graphViz.graphCandidate")}</span><strong>{candidate.documentPath?.split("/").at(-1) || candidate.candidateId}</strong>
              </button>
              <ArrowRight size={16} aria-hidden="true" />
              <div className="graph-chain-card rerank-card"><span>{t("graphViz.bgeRerank")}</span><strong>#{candidate.rerankPosition ?? "--"}</strong><small>{formatScore(candidate.rerankScore)}</small></div>
              <ArrowRight size={16} aria-hidden="true" />
              <button type="button" className={`graph-chain-card evidence-link ${candidate.selectedAsEvidence ? "selected" : ""}`} onClick={() => selectCandidate(candidate)}>
                {candidate.selectedAsEvidence ? <CheckCircle2 size={15} aria-hidden="true" /> : <Link2 size={15} aria-hidden="true" />}<span>{t("graphViz.evidence")}</span><strong>{candidate.evidenceId || t("graphViz.notEvidence")}</strong>
              </button>
            </div>
          ))}
        </div>

        <div className="graph-footer-grid">
          <div className="graph-legend"><span className="section-kicker">{t("graphViz.legend")}</span><div><i className="legend-seed" />{t("graphViz.role.seed")}<i className="legend-recovered" />{t("graphViz.role.recovered")}<i className="legend-context" />{t("graphViz.role.context")}<i className="legend-evidence" />{t("graphViz.evidence")}</div></div>
          <div className="graph-truth"><span>{t("graphViz.truth")}</span><strong>{t("graphViz.traceOnly")}</strong></div>
          {viewMode === "engineer" && <div className="graph-engineer-facts"><Field label={t("graphViz.policy")} value={present(model.activationPolicy, t("common.unavailable"))} /><Field label={t("graph.gold")} value={model.runtimeGoldMetadataUsage === true ? t("common.yes") : model.runtimeGoldMetadataUsage === false ? t("common.no") : t("common.unavailable")} /><Field label={t("graphViz.poolFlow")} value={`${model.candidatePoolBefore ?? "--"} → ${model.candidatePoolAfter ?? "--"}`} mono /></div>}
        </div>
      </section>
    </Panel>
  );
}
