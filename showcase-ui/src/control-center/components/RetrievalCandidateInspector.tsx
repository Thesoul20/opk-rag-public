import type { RuntimeCandidate, RuntimeTraceV1 } from "../../types/runtimeTrace";
import { Badge, EmptyState, KeyValue, Metric, Panel, StatusBadge } from "./Primitives";
import { candidateProvenanceLabels, candidateRelationship, rankMovement } from "../adapters/runtimeTrace";
import { useControlCenter } from "../state/ControlCenterState";

function value(value: unknown): string | number {
  return value == null ? "Unavailable" : typeof value === "number" ? value : String(value);
}

function score(value: number | null | undefined): string {
  return value == null ? "Unavailable" : value.toFixed(6);
}

function relationshipLabel(trace: RuntimeTraceV1, candidate: RuntimeCandidate): string {
  const relationship = candidateRelationship(trace, candidate);
  if (relationship === "evidence_selected") return "Evidence Selected";
  if (relationship === "context_candidate") return "Context Candidate";
  return "Candidate";
}

function movementLabel(candidate: RuntimeCandidate): string {
  const movement = rankMovement(candidate.initial_rank, candidate.final_rank);
  if (movement == null) return "Unavailable";
  if (movement === 0) return "unchanged";
  return movement > 0 ? `+${movement}` : `${movement}`;
}

export function RetrievalOverview({ trace }: { trace: RuntimeTraceV1 }) {
  const retrieval = trace.retrieval;
  return <Panel title="Retrieval Overview" action={<StatusBadge status={retrieval.stage_state === "failed" ? "failed" : retrieval.stage_state === "unavailable" ? "unavailable" : "completed"} label={retrieval.stage_state || "observed"} />}>
    <div className="cc-metric-grid cc-retrieval-metrics">
      <Metric label="Mode" value={value(retrieval.retrieval_mode)} />
      <Metric label="Initial" value={value(retrieval.initial_candidate_count)} />
      <Metric label="Vector" value={value(retrieval.vector_candidate_count)} />
      <Metric label="Lexical / BM25" value={value(retrieval.lexical_candidate_count)} />
      <Metric label="Post Structure" value={value(retrieval.post_structure_candidate_count)} />
      <Metric label="Post Graph" value={value(retrieval.post_graph_unified_candidate_count)} />
      <Metric label="Rerank input" value={value(retrieval.rerank_input_count)} />
      <Metric label="Final results" value={value(retrieval.final_result_count)} />
    </div>
    <div className="cc-retrieval-policy">
      <KeyValue label="Requested policy">{value(retrieval.requested_policy)}</KeyValue>
      <KeyValue label="Executed policy">{value(retrieval.executed_policy)}</KeyValue>
      <KeyValue label="Retrievers">{retrieval.retrievers_used?.length ? retrieval.retrievers_used.join(" · ") : "Unavailable"}</KeyValue>
      <KeyValue label="Pool after recovery">{value(retrieval.candidate_pool_count_after_recovery)}</KeyValue>
    </div>
  </Panel>;
}

export function RerankingSummary({ trace }: { trace: RuntimeTraceV1 }) {
  const rerank = trace.rerank;
  const unavailable = rerank.stage_state === "unavailable" || rerank.stage_state === "not_applicable" || rerank.stage_state === "skipped";
  return <Panel title="BGE Reranking" action={<StatusBadge status={unavailable ? "unavailable" : rerank.stage_state === "failed" ? "failed" : "completed"} label={rerank.stage_state || "observed"} />}>
    {unavailable ? <EmptyState title="Reranking unavailable" detail="This trace does not contain an executed reranking stage. No ranking values are reconstructed." /> : <div className="cc-rerank-summary">
      <KeyValue label="Model">{value(rerank.reranker_model)}</KeyValue>
      <KeyValue label="Revision">{value(rerank.reranker_revision)}</KeyValue>
      <KeyValue label="Device">{value(rerank.reranker_device)}</KeyValue>
      <KeyValue label="Precision">{value(rerank.precision)}</KeyValue>
      <KeyValue label="Execution scope">{value(rerank.execution_scope)}</KeyValue>
      <KeyValue label="Input → Output">{rerank.input_candidate_count == null || rerank.output_candidate_count == null ? "Unavailable" : `${rerank.input_candidate_count} → ${rerank.output_candidate_count}`}</KeyValue>
    </div>}
  </Panel>;
}

export function CandidatePool({ trace }: { trace: RuntimeTraceV1 }) {
  const { state, actions } = useControlCenter();
  const candidates = [...(trace.retrieval?.candidates || [])].sort((a, b) => (a.final_rank ?? Number.MAX_SAFE_INTEGER) - (b.final_rank ?? Number.MAX_SAFE_INTEGER));
  if (!candidates.length) return <Panel title="Candidate Pool"><EmptyState title="No candidates" detail="The authoritative Runtime Trace contains no candidate records." /></Panel>;
  return <Panel title={`Candidate Pool · ${candidates.length}`} action={<Badge tone="info">Candidate ≠ Evidence</Badge>}>
    <div className="cc-candidate-table-wrap">
      <table className="cc-candidate-table">
        <thead><tr><th>Initial</th><th>Final</th><th>Document</th><th>Retrieval score</th><th>Rerank score</th><th>Movement</th><th>Provenance</th><th>Downstream</th></tr></thead>
        <tbody>{candidates.map((candidate, index) => {
          const id = candidate.candidate_id || candidate.chunk_id || `candidate-${index}`;
          const provenance = candidateProvenanceLabels(candidate);
          return <tr key={id} data-selected={state.selectedCandidateId === id} onClick={() => actions.selectCandidate(id)} tabIndex={0} onKeyDown={event => { if (event.key === "Enter" || event.key === " ") actions.selectCandidate(id); }} aria-label={`Inspect candidate ${candidate.final_rank ?? index + 1}`}>
            <td className="cc-mono">{candidate.initial_rank == null ? "—" : `#${candidate.initial_rank}`}</td>
            <td className="cc-mono">{candidate.final_rank == null ? "—" : `#${candidate.final_rank}`}</td>
            <td><strong>{candidate.document_path || "Unknown document"}</strong><span>{candidate.section_path?.join(" / ") || "Root"}</span></td>
            <td className="cc-mono">{score(candidate.retrieval_score)}</td>
            <td className="cc-mono">{score(candidate.rerank_score)}</td>
            <td className="cc-mono">{movementLabel(candidate)}</td>
            <td><div className="cc-provenance-tags">{provenance.length ? provenance.map(label => <Badge key={label} tone="info">{label}</Badge>) : <span className="cc-muted">Unavailable</span>}</div></td>
            <td><Badge tone={candidateRelationship(trace, candidate) === "evidence_selected" ? "success" : "info"}>{relationshipLabel(trace, candidate)}</Badge></td>
          </tr>;
        })}</tbody>
      </table>
    </div>
  </Panel>;
}

export function RetrievalCandidateInspection({ trace, compact = false }: { trace: RuntimeTraceV1; compact?: boolean }) {
  if (trace.retrieval?.stage_state === "not_started") return <Panel><EmptyState title="Retrieval not executed" detail="This trace has no executed retrieval stage." /></Panel>;
  return <div className="cc-retrieval-inspection" data-compact={compact}>
    <RetrievalOverview trace={trace} />
    {!compact && <RerankingSummary trace={trace} />}
    <CandidatePool trace={trace} />
  </div>;
}
