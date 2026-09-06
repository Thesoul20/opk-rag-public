import type { RuntimeCandidate, RuntimeTraceV1 } from "../../types/runtimeTrace";
import { Badge, EmptyState, KeyValue, Metric, Panel, StatusBadge } from "./Primitives";
import { useControlCenter } from "../state/ControlCenterState";

export type GuardInspectorEntityKind = "guard" | "structure_recovery" | "graph_recovery";

const REASON_LABELS: Record<string, string> = {
  body_confidence_preserved: "Body confidence preserved",
  structural_query_phrase_or_entity_section_signal: "Structural signal detected",
  query_signal_with_seed_graph_availability: "Graph relation available",
  missing_query_or_retrieval_signal: "No bounded recovery signal",
  answerable_generation_abstained: "Generation abstained after answerability",
};

export function guardReasonLabel(code: string | null | undefined): string {
  if (!code) return "Unavailable";
  return REASON_LABELS[code] || "Bounded runtime reason";
}

export function recoveryDelta(before: number | null | undefined, after: number | null | undefined): number | null {
  return typeof before === "number" && typeof after === "number" ? after - before : null;
}

function value(input: unknown): string | number {
  return input == null ? "Unavailable" : typeof input === "number" ? input : String(input);
}

function stageStatus(state: string | null | undefined): "completed" | "failed" | "unavailable" | "partial" {
  if (state === "failed") return "failed";
  if (!state || state === "not_started" || state === "not_applicable") return "unavailable";
  if (state === "partial") return "partial";
  return "completed";
}

function DecisionStage({ label, primary, secondary, active, state, onClick }: { label: string; primary: string; secondary?: string; active?: boolean; state?: string | null; onClick?: () => void }) {
  const content = <><span>{label}</span><strong>{primary}</strong>{secondary && <small>{secondary}</small>}</>;
  return onClick ? <button type="button" className="cc-control-stage" data-active={active === true} data-state={state || "unavailable"} onClick={onClick}>{content}</button>
    : <div className="cc-control-stage" data-active={active === true} data-state={state || "unavailable"}>{content}</div>;
}

function CandidateLinkage({ trace, ids, label }: { trace: RuntimeTraceV1; ids: string[] | undefined; label: string }) {
  const { actions } = useControlCenter();
  const candidates = (trace.retrieval?.candidates || []).filter(candidate => {
    const id = candidate.candidate_id || candidate.chunk_id;
    return Boolean(id && ids?.includes(id));
  });
  if (!ids?.length) return null;
  return <div className="cc-recovery-candidates"><span className="cc-label">{label}</span><div>{ids.map(id => {
    const candidate = candidates.find(row => (row.candidate_id || row.chunk_id) === id);
    return candidate ? <button key={id} onClick={() => actions.selectCandidate(id)} title={candidate.document_path || id}>{candidate.final_rank ? `#${candidate.final_rank}` : "Candidate"} · {(candidate.document_path || id).split("/").pop()}</button>
      : <span key={id} className="cc-mono">{id.slice(0, 10)}…</span>;
  })}</div></div>;
}

export function GuardRecoveryControlPlane({ trace, compact = false }: { trace: RuntimeTraceV1; compact?: boolean }) {
  const { state, actions } = useControlCenter();
  const guard = trace.guard;
  const structure = trace.structure_recovery;
  const graph = trace.graph_recovery;
  const action = guard.recovery_action || "none";
  const reasonCode = guard.recovery_reason_code || guard.guard_reason_code || guard.refusal_reason_code;
  const structureDelta = recoveryDelta(structure.candidate_pool_count_before, structure.candidate_pool_count_after);
  const graphDelta = recoveryDelta(graph.candidate_pool_count_before, graph.candidate_pool_count_after);
  const recoveryLabel = action === "structure_recovery" ? "Structure Recovery" : action === "graph_recovery" ? "Graph Recovery" : "No Recovery";
  const recoveryState = action === "structure_recovery" ? structure.stage_state : action === "graph_recovery" ? graph.stage_state : "skipped";
  const runtimeOutcome = trace.outcome?.status || trace.trace.status || "unavailable";
  const guardOutcome = guard.final_decision || "Unavailable";
  const budgetWarning = (guard.maximum_recovery_attempt_count ?? 0) > 1 || (guard.recovery_attempt_count ?? 0) > 1 || (graph.hop_depth ?? 0) > 1;

  return <div className="cc-guard-recovery" data-compact={compact} data-agent-authority="bounded" data-hidden-reasoning-exposed={String(guard.hidden_chain_of_thought_exposed === true)}>
    <Panel title="Guarded Agent & Recovery" action={<div className="cc-inline-actions"><Badge tone="info">Bounded control plane</Badge><StatusBadge status={guard.fail_closed ? "refused" : stageStatus(guard.stage_state)} label={guard.fail_closed ? "fail closed" : guard.stage_state || "observed"} /></div>}>
      <div className="cc-control-plane-flow" aria-label="Guarded Agent control plane">
        <DecisionStage label="Initial Retrieval" primary={trace.retrieval?.retrieval_mode || trace.retrieval?.executed_policy || "Unavailable"} secondary={trace.retrieval?.initial_candidate_count == null ? undefined : `${trace.retrieval.initial_candidate_count} candidates`} state={trace.retrieval?.stage_state} />
        <DecisionStage label="Guard Observation" primary={guard.guard_triggered ? "Triggered" : "Observed"} secondary={guardReasonLabel(guard.guard_reason_code)} active={true} state={guard.stage_state} onClick={() => actions.selectInspectorEntity("guard")} />
        <DecisionStage label="Guard Decision" primary={guard.initial_decision || "Unavailable"} secondary={`Recovery ${guard.recovery_attempt_count ?? "—"} / ${guard.maximum_recovery_attempt_count ?? "—"}`} active={true} state={guard.stage_state} onClick={() => actions.selectInspectorEntity("guard")} />
        <DecisionStage label="Bounded Action" primary={recoveryLabel} secondary={guard.recovery_required ? guardReasonLabel(reasonCode) : "Continue without recovery"} active={action !== "none"} state={recoveryState} onClick={() => actions.selectInspectorEntity(action === "structure_recovery" ? "structure_recovery" : action === "graph_recovery" ? "graph_recovery" : "guard")} />
        <DecisionStage label="Unified Candidate Pool" primary={String(trace.retrieval?.candidate_pool_count_after_recovery ?? graph.candidate_pool_count_after ?? structure.candidate_pool_count_after ?? "Unavailable")} secondary="Candidate count only · not quality" state="completed" />
        <DecisionStage label="BGE Reranking" primary={trace.rerank?.reranker_model?.split("/").pop() || "Reranker"} secondary={trace.rerank?.input_candidate_count == null || trace.rerank?.output_candidate_count == null ? undefined : `${trace.rerank.input_candidate_count} → ${trace.rerank.output_candidate_count}`} state={trace.rerank?.stage_state} />
      </div>

      {!compact && <div className="cc-guard-summary-grid">
        <Metric label="Agent type" value={value(guard.agent_type)} />
        <Metric label="Route" value={value(guard.selected_route)} />
        <Metric label="Recovery budget" value={`${guard.recovery_attempt_count ?? "—"} / ${guard.maximum_recovery_attempt_count ?? "—"}`} />
        <Metric label="Graph hop" value={graph.graph_activated ? value(graph.hop_depth) : "not used"} />
        <Metric label="Guard outcome" value={guardOutcome} />
        <Metric label="Runtime outcome" value={runtimeOutcome} />
        <Metric label="Fail closed" value={guard.fail_closed ? "true" : "false"} />
        <Metric label="Hidden reasoning" value={guard.hidden_chain_of_thought_exposed ? "policy violation" : "not exposed"} />
      </div>}

      <div className="cc-guard-reason">
        <div><span className="cc-label">Bounded reason</span><strong>{guardReasonLabel(reasonCode)}</strong></div>
        <code>{reasonCode || "unavailable"}</code>
      </div>
      {budgetWarning && <div className="cc-governance-warning">Observed trace exceeds frozen UI governance limits (Recovery ≤1, Graph hop ≤1). Runtime data is shown unchanged.</div>}
    </Panel>

    {!compact && <div className="cc-recovery-grid">
      <Panel title="Structure Recovery" action={<button type="button" className="cc-panel-link" onClick={() => actions.selectInspectorEntity("structure_recovery")}><StatusBadge status={structure.triggered ? "completed" : "unavailable"} label={structure.triggered ? "executed" : "skipped"} /></button>}>
        {structure.triggered ? <>
          <div className="cc-recovery-impact"><strong>{value(structure.candidate_pool_count_before)} → {value(structure.candidate_pool_count_after)}</strong><span>{structureDelta == null ? "delta unavailable" : `${structureDelta >= 0 ? "+" : ""}${structureDelta} candidates`}</span></div>
          <KeyValue label="Expanded candidates">{value(structure.expanded_candidate_count)}</KeyValue>
          <KeyValue label="Seed candidates">{structure.seed_candidate_ids?.length ?? "Unavailable"}</KeyValue>
          <KeyValue label="Reason"><span><span>{guardReasonLabel(structure.reason_code)}</span><code className="cc-reason-code">{structure.reason_code || "unavailable"}</code></span></KeyValue>
          <CandidateLinkage trace={trace} ids={structure.expanded_candidate_ids} label="Structure-added candidates" />
        </> : <EmptyState title="Structure Recovery skipped" detail="No authoritative Structure Recovery executed for this trace." />}
      </Panel>

      <Panel title="Graph Recovery" action={<button type="button" className="cc-panel-link" onClick={() => actions.selectInspectorEntity("graph_recovery")}><StatusBadge status={graph.graph_activated ? "completed" : "unavailable"} label={graph.graph_activated ? "one-hop executed" : "skipped"} /></button>}>
        {graph.graph_activated ? <>
          <div className="cc-recovery-impact"><strong>{value(graph.candidate_pool_count_before)} → {value(graph.candidate_pool_count_after)}</strong><span>{graphDelta == null ? "delta unavailable" : `${graphDelta >= 0 ? "+" : ""}${graphDelta} candidates`}</span></div>
          <KeyValue label="Hop depth">{value(graph.hop_depth)}</KeyValue>
          <KeyValue label="Recovered candidates">{value(graph.recovered_candidate_count)}</KeyValue>
          <KeyValue label="Traversed edges">{graph.traversed_edges?.length ?? "Unavailable"}</KeyValue>
          <KeyValue label="Activation policy">{value(graph.activation_policy)}</KeyValue>
          <KeyValue label="Reason"><span><span>{guardReasonLabel(graph.activation_reason_code)}</span><code className="cc-reason-code">{graph.activation_reason_code || "unavailable"}</code></span></KeyValue>
          <CandidateLinkage trace={trace} ids={graph.recovered_candidate_ids} label="Graph-recovered candidates" />
          <button type="button" className="cc-inspect-graph-path" onClick={() => actions.setActiveNav("graph")}>Inspect Graph Path</button>
        </> : <EmptyState title="Graph Recovery skipped" detail="No authoritative Graph Recovery executed for this trace. The Graph page will not fabricate topology." />}
      </Panel>
    </div>}

    <Panel title="Decision Boundary" className="cc-decision-boundary">
      <div className="cc-decision-compare">
        <div><span className="cc-label">Guard decision</span><strong>{guardOutcome}</strong><small>Controls whether the pipeline proceeds to bounded recovery / reranking.</small></div>
        <div><span className="cc-label">Final runtime outcome</span><strong>{runtimeOutcome}</strong><small>{guard.fail_closed ? `Fail closed · ${guard.refusal_reason_code || trace.outcome?.refusal_reason_code || "reason unavailable"}` : "Downstream Evidence / Answerability / Grounding remain authoritative."}</small></div>
      </div>
    </Panel>
  </div>;
}

export function guardCandidateById(trace: RuntimeTraceV1, id: string): RuntimeCandidate | undefined {
  return (trace.retrieval?.candidates || []).find(candidate => candidate.candidate_id === id || candidate.chunk_id === id);
}
