import { useEffect } from "react";
import { X } from "lucide-react";
import { useShowcase } from "../../state/ShowcaseState";
import type { RuntimeCandidate, RuntimeTraceV1 } from "../../types/runtimeTrace";
import { graphEdgeById, graphNodeById, graphPathByCandidateId, type RuntimeGraphCandidatePath, type RuntimeGraphEdge, type RuntimeGraphNode } from "../adapters/graphRuntime";
import { buildEvidenceValidationModel, citationByKey, evidenceByKey, evidenceProvenanceLabels, type RuntimeCitationRecord, type RuntimeEvidenceRecord } from "../adapters/evidenceRuntime";
import { adaptRuntimeTrace, candidateProvenanceLabels, candidateRelationship, rankMovement } from "../adapters/runtimeTrace";
import type { InspectorTab } from "../architecture";
import { guardReasonLabel, recoveryDelta } from "../components/GuardRecoveryControlPlane";
import { Badge, Button, EmptyState, KeyValue } from "../components/Primitives";
import { useControlCenter, type InspectorEntityKind } from "../state/ControlCenterState";

const tabs: InspectorTab[] = ["details", "trace", "evidence", "source", "metadata"];

function unavailable(value: unknown): string | number {
  return value == null ? "Unavailable" : typeof value === "number" ? value : String(value);
}

function selectedCandidate(trace: RuntimeTraceV1 | undefined, candidateId: string | null): RuntimeCandidate | undefined {
  if (!trace || !candidateId) return undefined;
  return (trace.retrieval?.candidates || []).find(candidate => (candidate.candidate_id || candidate.chunk_id) === candidateId);
}

function CandidateDetails({ trace, candidate, onViewEvidence, onViewGraphPath }: { trace: RuntimeTraceV1; candidate: RuntimeCandidate; onViewEvidence?:()=>void; onViewGraphPath?:()=>void }) {
  const provenance = candidateProvenanceLabels(candidate);
  const movement = rankMovement(candidate.initial_rank, candidate.final_rank);
  const relation = candidateRelationship(trace, candidate);
  return <>
    <div className="cc-inspector-entity"><span className="cc-label">Selected candidate</span><strong>{candidate.document_path || "Unknown document"}</strong><span>{candidate.section_path?.join(" / ") || "Root"}</span></div>
    <KeyValue label="Initial rank">{candidate.initial_rank == null ? "Unavailable" : `#${candidate.initial_rank}`}</KeyValue>
    <KeyValue label="Final rank">{candidate.final_rank == null ? "Unavailable" : `#${candidate.final_rank}`}</KeyValue>
    <KeyValue label="Rank movement">{movement == null ? "Unavailable" : movement === 0 ? "unchanged" : movement > 0 ? `+${movement}` : `${movement}`}</KeyValue>
    <KeyValue label="Retrieval score" mono>{candidate.retrieval_score == null ? "Unavailable" : candidate.retrieval_score.toFixed(6)}</KeyValue>
    <KeyValue label="Rerank score" mono>{candidate.rerank_score == null ? "Unavailable" : candidate.rerank_score.toFixed(6)}</KeyValue>
    <KeyValue label="Retrieval sources">{candidate.retrieval_sources?.length ? candidate.retrieval_sources.join(" · ") : "Unavailable"}</KeyValue>
    <KeyValue label="Provenance"><span className="cc-provenance-tags">{provenance.length ? provenance.map(label => <Badge key={label}>{label}</Badge>) : "Unavailable"}</span></KeyValue>
    <KeyValue label="Downstream">{relation === "evidence_selected" ? "Evidence Selected" : relation === "context_candidate" ? "Context Candidate" : "Candidate"}</KeyValue>
    {(onViewEvidence||onViewGraphPath)&&<div className="cc-inline-actions">{onViewGraphPath&&<Button onClick={onViewGraphPath}>View Graph Path</Button>}{onViewEvidence&&<Button variant="primary" onClick={onViewEvidence}>View Evidence</Button>}</div>}
  </>;
}

function TraceView({ trace, candidate, kind }: { trace: RuntimeTraceV1; candidate?: RuntimeCandidate; kind?: InspectorEntityKind }) {
  const summary = adaptRuntimeTrace(trace);
  const graphEntity = kind === "graph_node" || kind === "graph_edge" || kind === "graph_path" || kind === "graph_recovery";
  const downstreamEntity = kind === "evidence_stage" || kind === "evidence" || kind === "answerability" || kind === "generation" || kind === "grounding" || kind === "citation" || kind === "outcome";
  return <>
    <KeyValue label="Trace ID" mono>{summary.traceId || "Unavailable"}</KeyValue>
    <KeyValue label="Status">{summary.status}</KeyValue>
    <KeyValue label="Scope">{summary.executionScope || "Unavailable"}</KeyValue>
    <KeyValue label="Retrieval mode">{trace.retrieval?.retrieval_mode || "Unavailable"}</KeyValue>
    <KeyValue label="Guard stage">{trace.guard?.stage_state || "Unavailable"}</KeyValue>
    <KeyValue label="Recovery action">{trace.guard?.recovery_action || "Unavailable"}</KeyValue>
    {graphEntity && <><KeyValue label="Graph stage">{trace.graph_recovery.stage_state || "Unavailable"}</KeyValue><KeyValue label="Graph hop">{unavailable(trace.graph_recovery.hop_depth)}</KeyValue><KeyValue label="Graph policy">{trace.graph_recovery.activation_policy || "Unavailable"}</KeyValue></>}
    {downstreamEntity && <><KeyValue label="Evidence stage">{trace.evidence.stage_state || "Unavailable"}</KeyValue><KeyValue label="Answerability stage">{trace.answerability.stage_state || "Unavailable"}</KeyValue><KeyValue label="Generation stage">{trace.generation.stage_state || "Unavailable"}</KeyValue><KeyValue label="Grounding stage">{trace.grounding.stage_state || "Unavailable"}</KeyValue><KeyValue label="Citation stage">{trace.citation.stage_state || "Unavailable"}</KeyValue></>}
    {candidate && <>
      <KeyValue label="Candidate origin">{candidateProvenanceLabels(candidate).join(" · ") || "Unavailable"}</KeyValue>
      <KeyValue label="Final rank">{candidate.final_rank == null ? "Unavailable" : `#${candidate.final_rank}`}</KeyValue>
      <KeyValue label="Rerank stage">{trace.rerank?.stage_state || "Unavailable"}</KeyValue>
    </>}
  </>;
}

function Source({ candidate }: { candidate: RuntimeCandidate }) {
  return <>
    <KeyValue label="Document">{candidate.document_path || "Unavailable"}</KeyValue>
    <KeyValue label="Section">{candidate.section_path?.join(" / ") || "Root"}</KeyValue>
    <KeyValue label="Document ID" mono>{candidate.document_id || "Unavailable"}</KeyValue>
    <KeyValue label="Chunk ID" mono>{candidate.chunk_id || "Unavailable"}</KeyValue>
    <div className="cc-source-preview"><span className="cc-label">Trace preview</span><p>{candidate.text_preview || "Preview unavailable in this Runtime Trace."}</p></div>
  </>;
}

function CandidateMetadata({ candidate }: { candidate: RuntimeCandidate }) {
  return <>
    <KeyValue label="Candidate ID" mono>{candidate.candidate_id || "Unavailable"}</KeyValue>
    <KeyValue label="Chunk ID" mono>{candidate.chunk_id || "Unavailable"}</KeyValue>
    <KeyValue label="Document ID" mono>{candidate.document_id || "Unavailable"}</KeyValue>
    <KeyValue label="Retrieval score" mono>{unavailable(candidate.retrieval_score)}</KeyValue>
    <KeyValue label="Rerank score" mono>{unavailable(candidate.rerank_score)}</KeyValue>
    <KeyValue label="Initial rank">{unavailable(candidate.initial_rank)}</KeyValue>
    <KeyValue label="Final rank">{unavailable(candidate.final_rank)}</KeyValue>
    <KeyValue label="Structure expanded">{String(candidate.structure_expanded === true)}</KeyValue>
    <KeyValue label="Graph recovered">{String(candidate.graph_recovered === true)}</KeyValue>
    <KeyValue label="Selected for context">{String(candidate.selected_for_context === true)}</KeyValue>
  </>;
}

function EvidenceRelationship({ trace, candidate }: { trace: RuntimeTraceV1; candidate: RuntimeCandidate }) {
  const relation = candidateRelationship(trace, candidate);
  const evidence = (trace.evidence?.evidence_items || []).find(item =>
    Boolean(candidate.candidate_id && item.source_candidate_id === candidate.candidate_id)
    || Boolean(candidate.chunk_id && item.chunk_id === candidate.chunk_id),
  );
  return <>
    <KeyValue label="Semantic boundary">Candidate ≠ Evidence</KeyValue>
    <KeyValue label="Relationship">{relation === "evidence_selected" ? "Evidence Selected" : relation === "context_candidate" ? "Context Candidate" : "Candidate only"}</KeyValue>
    {evidence ? <>
      <KeyValue label="Evidence ID" mono>{evidence.evidence_id || "Unavailable"}</KeyValue>
      <KeyValue label="Evidence position">{unavailable(evidence.evidence_position)}</KeyValue>
      <KeyValue label="Selection reason">{evidence.selection_reason_code || "Unavailable"}</KeyValue>
    </> : <p className="cc-muted">This candidate is not identified as selected Evidence by the active Runtime Trace.</p>}
  </>;
}

function GuardDetails({ trace }: { trace: RuntimeTraceV1 }) {
  const guard = trace.guard;
  const reason = guard.recovery_reason_code || guard.guard_reason_code || guard.refusal_reason_code;
  return <>
    <div className="cc-inspector-entity"><span className="cc-label">Guarded Agent</span><strong>{guard.agent_type || "Unavailable"}</strong><span>Bounded controller · no unrestricted planner</span></div>
    <KeyValue label="Selected route">{guard.selected_route || "Unavailable"}</KeyValue>
    <KeyValue label="Guard triggered">{String(guard.guard_triggered === true)}</KeyValue>
    <KeyValue label="Initial decision">{guard.initial_decision || "Unavailable"}</KeyValue>
    <KeyValue label="Final decision">{guard.final_decision || "Unavailable"}</KeyValue>
    <KeyValue label="Recovery required">{String(guard.recovery_required === true)}</KeyValue>
    <KeyValue label="Recovery action">{guard.recovery_action || "Unavailable"}</KeyValue>
    <KeyValue label="Recovery budget">{`${guard.recovery_attempt_count ?? "—"} / ${guard.maximum_recovery_attempt_count ?? "—"}`}</KeyValue>
    <KeyValue label="Reason"><span>{guardReasonLabel(reason)}<code className="cc-reason-code">{reason || "unavailable"}</code></span></KeyValue>
    <KeyValue label="Fail closed">{String(guard.fail_closed === true)}</KeyValue>
    <KeyValue label="Hidden reasoning">{guard.hidden_chain_of_thought_exposed ? "policy violation" : "not exposed"}</KeyValue>
  </>;
}

function StructureDetails({ trace }: { trace: RuntimeTraceV1 }) {
  const stage = trace.structure_recovery;
  const delta = recoveryDelta(stage.candidate_pool_count_before, stage.candidate_pool_count_after);
  return <>
    <div className="cc-inspector-entity"><span className="cc-label">Structure Recovery</span><strong>{stage.triggered ? "Executed" : "Skipped"}</strong><span>Bounded structural candidate expansion</span></div>
    <KeyValue label="Stage state">{stage.stage_state || "Unavailable"}</KeyValue>
    <KeyValue label="Triggered">{String(stage.triggered === true)}</KeyValue>
    <KeyValue label="Candidate pool">{stage.candidate_pool_count_before == null || stage.candidate_pool_count_after == null ? "Unavailable" : `${stage.candidate_pool_count_before} → ${stage.candidate_pool_count_after}`}</KeyValue>
    <KeyValue label="Deterministic delta">{delta == null ? "Unavailable" : `${delta >= 0 ? "+" : ""}${delta}`}</KeyValue>
    <KeyValue label="Expanded candidates">{unavailable(stage.expanded_candidate_count)}</KeyValue>
    <KeyValue label="Seed candidates">{stage.seed_candidate_ids?.length ?? "Unavailable"}</KeyValue>
    <KeyValue label="Source documents">{stage.source_document_ids?.length ?? "Unavailable"}</KeyValue>
    <KeyValue label="Reason"><span>{guardReasonLabel(stage.reason_code)}<code className="cc-reason-code">{stage.reason_code || "unavailable"}</code></span></KeyValue>
  </>;
}

function GraphDetails({ trace }: { trace: RuntimeTraceV1 }) {
  const stage = trace.graph_recovery;
  const delta = recoveryDelta(stage.candidate_pool_count_before, stage.candidate_pool_count_after);
  return <>
    <div className="cc-inspector-entity"><span className="cc-label">Graph Recovery</span><strong>{stage.graph_activated ? "Executed" : "Skipped"}</strong><span>Bounded one-hop graph candidate recovery</span></div>
    <KeyValue label="Stage state">{stage.stage_state || "Unavailable"}</KeyValue>
    <KeyValue label="Activated">{String(stage.graph_activated === true)}</KeyValue>
    <KeyValue label="Activation policy">{stage.activation_policy || "Unavailable"}</KeyValue>
    <KeyValue label="Hop depth">{unavailable(stage.hop_depth)}</KeyValue>
    <KeyValue label="Candidate pool">{stage.candidate_pool_count_before == null || stage.candidate_pool_count_after == null ? "Unavailable" : `${stage.candidate_pool_count_before} → ${stage.candidate_pool_count_after}`}</KeyValue>
    <KeyValue label="Deterministic delta">{delta == null ? "Unavailable" : `${delta >= 0 ? "+" : ""}${delta}`}</KeyValue>
    <KeyValue label="Recovered candidates">{unavailable(stage.recovered_candidate_count)}</KeyValue>
    <KeyValue label="Traversed edges">{stage.traversed_edges?.length ?? "Unavailable"}</KeyValue>
    <KeyValue label="Reason"><span>{guardReasonLabel(stage.activation_reason_code)}<code className="cc-reason-code">{stage.activation_reason_code || "unavailable"}</code></span></KeyValue>
    <div className="cc-inspector-boundary-note">Open Graph Retrieval for exact query-scoped node, edge and Candidate path inspection. No additional traversal is performed.</div>
  </>;
}

function GraphNodeDetails({ node }: { node: RuntimeGraphNode }) {
  return <>
    <div className="cc-inspector-entity"><span className="cc-label">Graph Node</span><strong>{node.label}</strong><span>{node.role} · runtime subgraph only</span></div>
    <KeyValue label="Semantic boundary">Graph Node ≠ Candidate</KeyValue>
    <KeyValue label="Role">{node.role}</KeyValue>
    <KeyValue label="Node ID" mono>{node.id}</KeyValue>
    <KeyValue label="Observed outgoing edges">{node.observedOutgoingEdgeCount}</KeyValue>
    <KeyValue label="Observed incoming edges">{node.observedIncomingEdgeCount}</KeyValue>
    <KeyValue label="Recovered Candidate IDs" mono>{node.recoveredCandidateIds.join(", ") || "None observed"}</KeyValue>
  </>;
}

function GraphEdgeDetails({ edge }: { edge: RuntimeGraphEdge }) {
  return <>
    <div className="cc-inspector-entity"><span className="cc-label">Graph Edge</span><strong>{edge.relationType || "Relation unavailable"}</strong><span>Authoritative traversed edge</span></div>
    <KeyValue label="Edge ID" mono>{edge.edgeId || "Unavailable"}</KeyValue>
    <KeyValue label="Source node" mono>{edge.sourceNodeId || "Unavailable"}</KeyValue>
    <KeyValue label="Relation type" mono>{edge.relationType || "Unavailable"}</KeyValue>
    <KeyValue label="Target node" mono>{edge.targetNodeId || "Unavailable"}</KeyValue>
    <KeyValue label="Hop depth">{unavailable(edge.hopDepth)}</KeyValue>
    <KeyValue label="Observation" mono>{edge.observation || "Unavailable"}</KeyValue>
  </>;
}

function GraphPathDetails({ trace, path, onViewCandidate }: { trace: RuntimeTraceV1; path: RuntimeGraphCandidatePath; onViewCandidate: () => void }) {
  const provenance = path.provenance;
  const first = provenance.graph_path?.[0];
  const candidate = path.candidate || selectedCandidate(trace, path.candidateId);
  return <>
    <div className="cc-inspector-entity"><span className="cc-label">Candidate Graph Path</span><strong>{candidate?.document_path || provenance.document_id || path.candidateId}</strong><span>Graph provenance · not Evidence authority</span></div>
    <KeyValue label="Semantic boundary">Graph Node ≠ Candidate ≠ Evidence</KeyValue>
    <KeyValue label="Candidate ID" mono>{path.candidateId}</KeyValue>
    <KeyValue label="Canonical chunk ID" mono>{provenance.canonical_chunk_id || "Unavailable"}</KeyValue>
    <KeyValue label="Source seed" mono>{provenance.source_seed_candidate_id || provenance.seed_candidate_id || "Unavailable"}</KeyValue>
    <KeyValue label="Document ID" mono>{provenance.document_id || "Unavailable"}</KeyValue>
    <KeyValue label="Section ID" mono>{provenance.section_id || "Unavailable"}</KeyValue>
    <KeyValue label="Graph added">{String(provenance.graph_added === true)}</KeyValue>
    <KeyValue label="Edge types" mono>{provenance.graph_edge_types?.join(" · ") || provenance.edge_types?.join(" · ") || "Unavailable"}</KeyValue>
    <KeyValue label="Hop count">{unavailable(provenance.graph_hop_count)}</KeyValue>
    <KeyValue label="Expansion reason" mono>{provenance.graph_expansion_reason || "Unavailable"}</KeyValue>
    <KeyValue label="Authority">{first?.authority_level || "Unavailable"}</KeyValue>
    <KeyValue label="Record class" mono>{first?.graph_record_class || "Unavailable"}</KeyValue>
    <Button variant="primary" onClick={onViewCandidate}>View Candidate</Button>
  </>;
}

function GuardMetadata({ trace, kind }: { trace: RuntimeTraceV1; kind: "guard" | "structure_recovery" | "graph_recovery" }) {
  if (kind === "guard") return <>
    <KeyValue label="Agent type" mono>{trace.guard.agent_type || "Unavailable"}</KeyValue>
    <KeyValue label="Guard reason code" mono>{trace.guard.guard_reason_code || "Unavailable"}</KeyValue>
    <KeyValue label="Recovery reason code" mono>{trace.guard.recovery_reason_code || "Unavailable"}</KeyValue>
    <KeyValue label="Refusal reason code" mono>{trace.guard.refusal_reason_code || "Unavailable"}</KeyValue>
    <KeyValue label="Hidden CoT exposed">{String(trace.guard.hidden_chain_of_thought_exposed === true)}</KeyValue>
  </>;
  if (kind === "structure_recovery") return <>
    <KeyValue label="Seed candidate IDs" mono>{trace.structure_recovery.seed_candidate_ids?.join(", ") || "Unavailable"}</KeyValue>
    <KeyValue label="Expanded candidate IDs" mono>{trace.structure_recovery.expanded_candidate_ids?.join(", ") || "Unavailable"}</KeyValue>
  </>;
  return <>
    <KeyValue label="Seed candidate IDs" mono>{trace.graph_recovery.seed_candidate_ids?.join(", ") || "Unavailable"}</KeyValue>
    <KeyValue label="Recovered candidate IDs" mono>{trace.graph_recovery.recovered_candidate_ids?.join(", ") || "Unavailable"}</KeyValue>
    <KeyValue label="Recovered node IDs" mono>{trace.graph_recovery.recovered_node_ids?.join(", ") || "Unavailable"}</KeyValue>
    <KeyValue label="Runtime Gold used">{String(trace.graph_recovery.runtime_gold_metadata_usage === true)}</KeyValue>
  </>;
}

function GraphEdgeMetadata({ edge }: { edge: RuntimeGraphEdge }) {
  return <><KeyValue label="Source document ID" mono>{edge.sourceDocumentId || "Unavailable"}</KeyValue><KeyValue label="Target document ID" mono>{edge.targetDocumentId || "Unavailable"}</KeyValue><KeyValue label="Trace edge index">{edge.traceIndex}</KeyValue></>;
}

function GraphPathMetadata({ path }: { path: RuntimeGraphCandidatePath }) {
  const rows = path.provenance.graph_path || [];
  if (!rows.length) return <div className="cc-inspector-boundary-note">No bounded graph_path rows are available for this Candidate.</div>;
  return <div className="cc-graph-path-metadata">{rows.map((row, index) => <div key={`${row.edge_id || "edge"}-${index}`}><KeyValue label={`Path edge ${index + 1}`} mono>{row.edge_id || "Unavailable"}</KeyValue><KeyValue label="Relation" mono>{row.edge_type || "Unavailable"}</KeyValue><KeyValue label="Authority">{row.authority_level || "Unavailable"}</KeyValue><KeyValue label="Record class" mono>{row.graph_record_class || "Unavailable"}</KeyValue><KeyValue label="Source" mono>{row.source_node_id || row.authority_source_node_id || "Unavailable"}</KeyValue><KeyValue label="Target" mono>{row.target_node_id || row.authority_target_node_id || "Unavailable"}</KeyValue></div>)}</div>;
}

function EvidenceStageDetails({ trace }: { trace: RuntimeTraceV1 }) {
  return <>
    <div className="cc-inspector-entity"><span className="cc-label">Evidence Composition</span><strong>{trace.evidence.stage_state || "Unavailable"}</strong><span>Candidate selection boundary · Candidate ≠ Evidence</span></div>
    <KeyValue label="Evidence count">{unavailable(trace.evidence.evidence_count)}</KeyValue>
    <KeyValue label="Candidate ≠ Evidence">{trace.evidence.candidate_evidence_semantic_separation == null ? "Unavailable" : String(trace.evidence.candidate_evidence_semantic_separation)}</KeyValue>
    <KeyValue label="Provenance available">{trace.evidence.provenance_available == null ? "Unavailable" : String(trace.evidence.provenance_available)}</KeyValue>
    <div className="cc-inspector-boundary-note">Evidence Composition is authoritative runtime output. This UI cannot add, remove, reorder, or promote Candidates into Evidence.</div>
  </>;
}

function EvidenceDetails({ row, onViewCandidate, onViewGraphPath, onViewCitation }: { row: RuntimeEvidenceRecord; onViewCandidate: () => void; onViewGraphPath?:()=>void; onViewCitation?:()=>void }) {
  const e = row.evidence;
  return <>
    <div className="cc-inspector-entity"><span className="cc-label">Evidence</span><strong>{e.evidence_id || "Evidence ID unavailable"}</strong><span>{e.document_path || "Document unavailable"}</span></div>
    <KeyValue label="Semantic boundary">Candidate ≠ Evidence ≠ Citation</KeyValue>
    <KeyValue label="Evidence position">{unavailable(e.evidence_position)}</KeyValue>
    <KeyValue label="Evidence score" mono>{unavailable(e.evidence_score)}</KeyValue>
    <KeyValue label="Selection reason" mono>{e.selection_reason_code || "Unavailable"}</KeyValue>
    <KeyValue label="Document">{e.document_path || "Unavailable"}</KeyValue>
    <KeyValue label="Section">{e.section_path?.join(" / ") || "Unavailable"}</KeyValue>
    <KeyValue label="Lines">{e.start_line == null ? "Unavailable" : e.end_line == null || e.end_line === e.start_line ? `L${e.start_line}` : `L${e.start_line}–L${e.end_line}`}</KeyValue>
    <KeyValue label="Cited by">{row.citationIds.join(", ") || "Not cited"}</KeyValue>
    {row.sourceCandidateId ? <div className="cc-inline-actions"><Button disabled={!row.candidateFound} onClick={onViewCandidate}>View Source Candidate</Button>{onViewGraphPath&&<Button onClick={onViewGraphPath}>View Graph Path</Button>}{onViewCitation&&<Button variant="primary" onClick={onViewCitation}>View Citation</Button>}</div> : <div className="cc-governance-warning">Source Candidate identity is unavailable; the UI does not infer one.</div>}
  </>;
}

function EvidenceLineage({ row, onViewCandidate }: { row: RuntimeEvidenceRecord; onViewCandidate: () => void }) {
  return <>
    <KeyValue label="Lineage">Candidate → Evidence → Citation</KeyValue>
    <KeyValue label="Source Candidate ID" mono>{row.sourceCandidateId || "Unavailable"}</KeyValue>
    <KeyValue label="Evidence ID" mono>{row.evidenceId || "Unavailable"}</KeyValue>
    <KeyValue label="Citation IDs" mono>{row.citationIds.join(", ") || "None"}</KeyValue>
    <KeyValue label="Graph recovered">{String(row.graphRecovered)}</KeyValue>
    {row.sourceCandidateId && <Button disabled={!row.candidateFound} onClick={onViewCandidate}>View Source Candidate</Button>}
    {!row.candidateFound && row.sourceCandidateId && <div className="cc-governance-warning">The authoritative source_candidate_id does not resolve to an observed Candidate in this bounded trace.</div>}
  </>;
}

function EvidenceSource({ row }: { row: RuntimeEvidenceRecord }) {
  const e = row.evidence;
  return <>
    <KeyValue label="Document">{e.document_path || "Unavailable"}</KeyValue>
    <KeyValue label="Section">{e.section_path?.join(" / ") || "Unavailable"}</KeyValue>
    <KeyValue label="Start line">{unavailable(e.start_line)}</KeyValue>
    <KeyValue label="End line">{unavailable(e.end_line)}</KeyValue>
    <div className="cc-inspector-boundary-note">Only bounded Runtime Trace source metadata is shown. TASK-0276 does not read the local knowledge-base file to reconstruct missing content or line ranges.</div>
  </>;
}

function EvidenceMetadata({ row }: { row: RuntimeEvidenceRecord }) {
  const e = row.evidence;
  const provenance = evidenceProvenanceLabels(e);
  return <>
    <KeyValue label="Evidence ID" mono>{e.evidence_id || "Unavailable"}</KeyValue><KeyValue label="Source Candidate ID" mono>{e.source_candidate_id || "Unavailable"}</KeyValue><KeyValue label="Chunk ID" mono>{e.chunk_id || "Unavailable"}</KeyValue><KeyValue label="Document ID" mono>{e.document_id || "Unavailable"}</KeyValue><KeyValue label="Retrieval sources" mono>{e.retrieval_sources?.join(" · ") || "Unavailable"}</KeyValue><KeyValue label="Provenance">{provenance.join(" · ") || "Unavailable"}</KeyValue><KeyValue label="Trace item index">{row.traceIndex}</KeyValue>
  </>;
}

function AnswerabilityDetails({ trace }: { trace: RuntimeTraceV1 }) {
  const a = trace.answerability;
  return <>
    <div className="cc-inspector-entity"><span className="cc-label">Runtime Answerability Gate</span><strong>{a.answerability_state || a.stage_state || "Unavailable"}</strong><span>Evidence sufficiency gate · not a correctness probability</span></div>
    <KeyValue label="Stage state">{a.stage_state || "Unavailable"}</KeyValue><KeyValue label="Answerable">{a.answerable == null ? "Unavailable" : String(a.answerable)}</KeyValue><KeyValue label="Reason" mono>{a.reason_code || "Unavailable"}</KeyValue><KeyValue label="Required support">{a.required_support_state || "Unavailable"}</KeyValue><KeyValue label="Evidence count">{unavailable(a.evidence_count)}</KeyValue><KeyValue label="Evidence considered">{unavailable(a.considered_evidence_count)}</KeyValue><KeyValue label="Confidence telemetry" mono>{a.confidence == null ? "Unavailable" : a.confidence.toFixed(6)}</KeyValue>
    <div className="cc-inspector-boundary-note">Answerability=true does not force Generation to release an answer.</div>
  </>;
}

function GenerationDetails({ trace }: { trace: RuntimeTraceV1 }) {
  const g = trace.generation;
  return <>
    <div className="cc-inspector-entity"><span className="cc-label">Generation</span><strong>{g.generation_abstained ? "Abstained" : g.generation_completed ? "Completed" : g.stage_state || "Unavailable"}</strong><span>Safe runtime metadata only · hidden reasoning excluded</span></div>
    <KeyValue label="Attempted">{g.generation_attempted == null ? "Unavailable" : String(g.generation_attempted)}</KeyValue><KeyValue label="Completed">{g.generation_completed == null ? "Unavailable" : String(g.generation_completed)}</KeyValue><KeyValue label="Abstained">{g.generation_abstained == null ? "Unavailable" : String(g.generation_abstained)}</KeyValue><KeyValue label="Provider">{g.generation_provider || "Unavailable"}</KeyValue><KeyValue label="Model">{g.generation_model || "Unavailable"}</KeyValue><KeyValue label="Endpoint">{g.generation_endpoint_type || "Unavailable"}</KeyValue><KeyValue label="Latency">{g.generation_latency_ms == null ? "Unavailable" : `${g.generation_latency_ms} ms`}</KeyValue><KeyValue label="Tokens">{g.total_tokens == null ? "Unavailable" : `${g.prompt_tokens ?? "?"} + ${g.completion_tokens ?? "?"} = ${g.total_tokens}`}</KeyValue><KeyValue label="Finish reason">{g.finish_reason || "Unavailable"}</KeyValue>
    <div className="cc-inspector-boundary-note">Raw prompts, provider envelopes, secrets, scratchpads and Chain-of-Thought are not exposed.</div>
  </>;
}

function GroundingDetails({ trace }: { trace: RuntimeTraceV1 }) {
  const g = trace.grounding;
  return <>
    <div className="cc-inspector-entity"><span className="cc-label">Grounding Validation</span><strong>{g.status || g.stage_state || "Unavailable"}</strong><span>Generated claims checked against authoritative Evidence</span></div>
    <KeyValue label="Checked">{g.grounding_checked == null ? "Unavailable" : String(g.grounding_checked)}</KeyValue><KeyValue label="Passed">{g.grounding_passed == null ? "Unavailable" : String(g.grounding_passed)}</KeyValue><KeyValue label="Reason" mono>{g.reason_code || g.failure_reason_code || "Unavailable"}</KeyValue><KeyValue label="Supported claims">{unavailable(g.supported_claim_count)}</KeyValue><KeyValue label="Unsupported claims">{unavailable(g.unsupported_claim_count)}</KeyValue><KeyValue label="Citation coverage">{g.citation_coverage == null ? "Unavailable" : `${Math.round(g.citation_coverage * 100)}%`}</KeyValue>
    <div className="cc-inspector-boundary-note">Claim counts are aggregate runtime telemetry. Individual claim text is not reconstructed when Runtime Trace does not expose it.</div>
  </>;
}

function CitationStageDetails({ trace }: { trace: RuntimeTraceV1 }) {
  const c = trace.citation;
  return <>
    <div className="cc-inspector-entity"><span className="cc-label">Citation Validation</span><strong>{c.citation_valid == null ? c.stage_state || "Unavailable" : c.citation_valid ? "Valid" : "Invalid"}</strong><span>Evidence lineage and released reference integrity</span></div>
    <KeyValue label="Checked">{c.citation_checked == null ? "Unavailable" : String(c.citation_checked)}</KeyValue><KeyValue label="Valid">{c.citation_valid == null ? "Unavailable" : String(c.citation_valid)}</KeyValue><KeyValue label="Citation count">{unavailable(c.citation_count)}</KeyValue><KeyValue label="Invalid count">{unavailable(c.invalid_citation_count)}</KeyValue><KeyValue label="Note">{c.note || "Unavailable"}</KeyValue>
  </>;
}

function CitationDetails({ row, onViewEvidence, onViewCandidate }: { row: RuntimeCitationRecord; onViewEvidence: () => void; onViewCandidate: () => void }) {
  const c = row.citation;
  return <>
    <div className="cc-inspector-entity"><span className="cc-label">Citation</span><strong>{c.citation_id || "Citation ID unavailable"}</strong><span>{c.document_path || row.evidence?.evidence.document_path || "Document unavailable"}</span></div>
    <KeyValue label="Semantic boundary">Evidence ≠ Citation</KeyValue><KeyValue label="Evidence ID" mono>{row.evidenceId || "Unavailable"}</KeyValue><KeyValue label="Evidence resolved">{String(row.evidenceFound)}</KeyValue><KeyValue label="Source Candidate ID" mono>{row.resolvedSourceCandidateId || "Unavailable"}</KeyValue><KeyValue label="Candidate identity match">{row.candidateIdentityMatches == null ? "Unavailable" : String(row.candidateIdentityMatches)}</KeyValue><KeyValue label="Graph lineage">{String(row.graphRecovered)}</KeyValue>
    <div className="cc-inline-actions">{row.evidence && <Button onClick={onViewEvidence}>View Evidence</Button>}{row.resolvedSourceCandidateId && <Button variant="primary" disabled={!row.candidateFound} onClick={onViewCandidate}>View Candidate</Button>}</div>
    {!row.evidenceFound && <div className="cc-governance-warning">Orphan or incomplete Citation lineage: the authoritative Evidence reference is not resolvable in this bounded trace. Nothing is inferred.</div>}
  </>;
}

function CitationLineage({ row, onViewEvidence, onViewCandidate }: { row: RuntimeCitationRecord; onViewEvidence: () => void; onViewCandidate: () => void }) {
  return <>
    <KeyValue label="Lineage">Citation → Evidence → Candidate</KeyValue><KeyValue label="Citation ID" mono>{row.citationId || "Unavailable"}</KeyValue><KeyValue label="Evidence ID" mono>{row.evidenceId || "Unavailable"}</KeyValue><KeyValue label="Source Candidate ID" mono>{row.resolvedSourceCandidateId || "Unavailable"}</KeyValue><KeyValue label="Graph recovered">{String(row.graphRecovered)}</KeyValue>
    <div className="cc-inline-actions">{row.evidence && <Button onClick={onViewEvidence}>View Evidence</Button>}{row.resolvedSourceCandidateId && <Button disabled={!row.candidateFound} onClick={onViewCandidate}>View Candidate</Button>}</div>
  </>;
}

function CitationSource({ row }: { row: RuntimeCitationRecord }) {
  const c = row.citation; const e = row.evidence?.evidence;
  return <><KeyValue label="Document">{c.document_path || e?.document_path || "Unavailable"}</KeyValue><KeyValue label="Section">{c.section_path?.join(" / ") || e?.section_path?.join(" / ") || "Unavailable"}</KeyValue><KeyValue label="Start line">{unavailable(c.start_line ?? e?.start_line)}</KeyValue><KeyValue label="End line">{unavailable(c.end_line ?? e?.end_line)}</KeyValue><div className="cc-inspector-boundary-note">Source location is displayed only from Citation/Evidence Runtime Trace metadata.</div></>;
}

function CitationMetadata({ row }: { row: RuntimeCitationRecord }) {
  const c = row.citation;
  return <><KeyValue label="Citation ID" mono>{c.citation_id || "Unavailable"}</KeyValue><KeyValue label="Evidence ID" mono>{c.evidence_id || "Unavailable"}</KeyValue><KeyValue label="Direct Candidate ID" mono>{c.source_candidate_id || "Unavailable"}</KeyValue><KeyValue label="Resolved Candidate ID" mono>{row.resolvedSourceCandidateId || "Unavailable"}</KeyValue><KeyValue label="Chunk ID" mono>{c.chunk_id || "Unavailable"}</KeyValue><KeyValue label="Document ID" mono>{c.document_id || "Unavailable"}</KeyValue><KeyValue label="Trace item index">{row.traceIndex}</KeyValue></>;
}

function OutcomeDetails({ trace }: { trace: RuntimeTraceV1 }) {
  const o = trace.outcome; const safeRefusal = o.status === "refused" && !o.failure_stage;
  return <><div className="cc-inspector-entity"><span className="cc-label">Final Runtime Outcome</span><strong>{safeRefusal ? "Safe Refusal" : o.status || trace.trace.status}</strong><span>Release outcome · separate from prior validation stages</span></div><KeyValue label="Status">{o.status || trace.trace.status}</KeyValue><KeyValue label="Answer status">{o.answer_status || "Unavailable"}</KeyValue><KeyValue label="Refused">{String(o.refused === true)}</KeyValue><KeyValue label="Failure stage">{o.failure_stage || "None"}</KeyValue><KeyValue label="Final Candidate count">{unavailable(o.final_candidate_count)}</KeyValue><KeyValue label="Final Evidence count">{unavailable(o.final_evidence_count)}</KeyValue><KeyValue label="Grounding passed">{o.grounding_passed == null ? "Unavailable" : String(o.grounding_passed)}</KeyValue><KeyValue label="Citation valid">{o.citation_valid == null ? "Unavailable" : String(o.citation_valid)}</KeyValue><KeyValue label="Final decision">{o.final_decision || "Unavailable"}</KeyValue><KeyValue label="Refusal reason" mono>{o.refusal_reason_code || "None"}</KeyValue>{safeRefusal && <div className="cc-inspector-boundary-note">Refused ≠ failed. A governed refusal is an expected safe outcome, not automatically a runtime error.</div>}</>;
}

function entityTitle(kind: InspectorEntityKind, candidate?: RuntimeCandidate, node?: RuntimeGraphNode, edge?: RuntimeGraphEdge, path?: RuntimeGraphCandidatePath, evidence?: RuntimeEvidenceRecord, citation?: RuntimeCitationRecord): string | undefined {
  if (kind === "candidate") return candidate?.document_path || "Candidate";
  if (kind === "guard") return "Guarded Agent";
  if (kind === "structure_recovery") return "Structure Recovery";
  if (kind === "graph_recovery") return "Graph Recovery";
  if (kind === "graph_node") return node?.label || "Graph Node";
  if (kind === "graph_edge") return edge?.relationType || "Graph Edge";
  if (kind === "graph_path") return path?.candidate?.document_path || path?.provenance.document_id || "Candidate Graph Path";
  if (kind === "evidence_stage") return "Evidence Composition";
  if (kind === "evidence") return evidence?.evidenceId || "Evidence";
  if (kind === "answerability") return "Answerability";
  if (kind === "generation") return "Generation";
  if (kind === "grounding") return "Grounding";
  if (kind === "citation") return citation?.citationId || "Citation Validation";
  if (kind === "outcome") return "Final Outcome";
  return undefined;
}

export function Inspector() {
  const { state, actions } = useControlCenter();
  const { state: showcase } = useShowcase();
  const trace = showcase.trace;
  const candidate = selectedCandidate(trace, state.selectedCandidateId);
  const node = trace ? graphNodeById(trace, state.selectedGraphNodeId) : undefined;
  const edge = trace ? graphEdgeById(trace, state.selectedGraphEdgeId) : undefined;
  const path = trace ? graphPathByCandidateId(trace, state.selectedGraphPathCandidateId) : undefined;
  const pathCandidate = trace && path ? path.candidate || selectedCandidate(trace, path.candidateId) : undefined;
  const evidenceRecord = trace ? evidenceByKey(trace, state.selectedEvidenceKey) : undefined;
  const citationRecord = trace ? citationByKey(trace, state.selectedCitationKey) : undefined;
  const candidateEvidence = trace && candidate ? buildEvidenceValidationModel(trace).evidence.find(row => row.sourceCandidateId === (candidate.candidate_id || candidate.chunk_id)) : undefined;
  const candidateGraphPath = trace && candidate ? graphPathByCandidateId(trace, candidate.candidate_id || candidate.chunk_id || null) : undefined;
  const kind = state.selectedInspectorEntityKind;

  useEffect(() => {
    if (kind === "candidate" && state.selectedCandidateId && !candidate) actions.selectCandidate(null);
    if (kind === "graph_node" && state.selectedGraphNodeId && !node) actions.selectGraphNode(null);
    if (kind === "graph_edge" && state.selectedGraphEdgeId && !edge) actions.selectGraphEdge(null);
    if (kind === "graph_path" && state.selectedGraphPathCandidateId && !path) actions.selectGraphPath(null);
    if (kind === "evidence" && state.selectedEvidenceKey && !evidenceRecord) actions.selectEvidence(null);
    if (kind === "citation" && state.selectedCitationKey && !citationRecord) actions.selectCitation(null);
  }, [trace?.trace?.trace_id, kind, state.selectedCandidateId, state.selectedGraphNodeId, state.selectedGraphEdgeId, state.selectedGraphPathCandidateId, state.selectedEvidenceKey, state.selectedCitationKey, candidate, node, edge, path, evidenceRecord, citationRecord]);

  let body = <EmptyState title="No selected entity" detail="Select a Candidate, Evidence/Citation, validation stage, Guard/Recovery stage, or Graph entity to inspect bounded Runtime Trace telemetry." />;
  if (trace && state.inspectorTab === "trace") body = <TraceView trace={trace} candidate={candidate || pathCandidate} kind={kind} />;
  else if (trace && kind === "candidate" && candidate && state.inspectorTab === "details") body = <CandidateDetails trace={trace} candidate={candidate} onViewEvidence={candidateEvidence ? () => { actions.setActiveNav("evidence"); actions.selectEvidence(candidateEvidence.key); } : undefined} onViewGraphPath={candidateGraphPath ? () => { actions.setActiveNav("graph"); actions.selectGraphPath(candidateGraphPath.candidateId); } : undefined} />;
  else if (trace && kind === "candidate" && candidate && state.inspectorTab === "evidence") body = <EvidenceRelationship trace={trace} candidate={candidate} />;
  else if (kind === "candidate" && candidate && state.inspectorTab === "source") body = <Source candidate={candidate} />;
  else if (kind === "candidate" && candidate && state.inspectorTab === "metadata") body = <CandidateMetadata candidate={candidate} />;
  else if (trace && kind === "guard" && state.inspectorTab === "details") body = <GuardDetails trace={trace} />;
  else if (trace && kind === "structure_recovery" && state.inspectorTab === "details") body = <StructureDetails trace={trace} />;
  else if (trace && kind === "graph_recovery" && state.inspectorTab === "details") body = <GraphDetails trace={trace} />;
  else if (trace && (kind === "guard" || kind === "structure_recovery" || kind === "graph_recovery") && state.inspectorTab === "metadata") body = <GuardMetadata trace={trace} kind={kind} />;
  else if (kind === "graph_node" && node && state.inspectorTab === "details") body = <GraphNodeDetails node={node} />;
  else if (kind === "graph_edge" && edge && state.inspectorTab === "details") body = <GraphEdgeDetails edge={edge} />;
  else if (trace && kind === "graph_path" && path && state.inspectorTab === "details") body = <GraphPathDetails trace={trace} path={path} onViewCandidate={() => actions.selectCandidate(path.candidateId)} />;
  else if (kind === "graph_edge" && edge && state.inspectorTab === "metadata") body = <GraphEdgeMetadata edge={edge} />;
  else if (kind === "graph_path" && path && state.inspectorTab === "metadata") body = <GraphPathMetadata path={path} />;
  else if (kind === "graph_node" && node && state.inspectorTab === "metadata") body = <><KeyValue label="Node ID" mono>{node.id}</KeyValue><KeyValue label="Role">{node.role}</KeyValue><KeyValue label="Observed outgoing">{node.observedOutgoingEdgeCount}</KeyValue><KeyValue label="Observed incoming">{node.observedIncomingEdgeCount}</KeyValue></>;
  else if (trace && kind === "graph_path" && pathCandidate && state.inspectorTab === "evidence") body = <EvidenceRelationship trace={trace} candidate={pathCandidate} />;
  else if (kind === "graph_path" && pathCandidate && state.inspectorTab === "source") body = <Source candidate={pathCandidate} />;
  else if (trace && kind === "evidence_stage" && state.inspectorTab === "details") body = <EvidenceStageDetails trace={trace} />;
  else if (kind === "evidence" && evidenceRecord && state.inspectorTab === "details") body = <EvidenceDetails row={evidenceRecord} onViewCandidate={() => evidenceRecord.sourceCandidateId && actions.selectCandidate(evidenceRecord.sourceCandidateId)} onViewGraphPath={trace && evidenceRecord.sourceCandidateId && graphPathByCandidateId(trace,evidenceRecord.sourceCandidateId) ? () => { actions.setActiveNav("graph"); actions.selectGraphPath(evidenceRecord.sourceCandidateId!); } : undefined} onViewCitation={trace && evidenceRecord.citationIds.length ? () => { const row=buildEvidenceValidationModel(trace).citations.find(c=>c.citationId===evidenceRecord.citationIds[0]); if(row){actions.setActiveNav("evidence");actions.selectCitation(row.key);} } : undefined} />;
  else if (kind === "evidence" && evidenceRecord && state.inspectorTab === "evidence") body = <EvidenceLineage row={evidenceRecord} onViewCandidate={() => evidenceRecord.sourceCandidateId && actions.selectCandidate(evidenceRecord.sourceCandidateId)} />;
  else if (kind === "evidence" && evidenceRecord && state.inspectorTab === "source") body = <EvidenceSource row={evidenceRecord} />;
  else if (kind === "evidence" && evidenceRecord && state.inspectorTab === "metadata") body = <EvidenceMetadata row={evidenceRecord} />;
  else if (trace && kind === "answerability" && state.inspectorTab === "details") body = <AnswerabilityDetails trace={trace} />;
  else if (trace && kind === "generation" && state.inspectorTab === "details") body = <GenerationDetails trace={trace} />;
  else if (trace && kind === "grounding" && state.inspectorTab === "details") body = <GroundingDetails trace={trace} />;
  else if (trace && kind === "citation" && !citationRecord && state.inspectorTab === "details") body = <CitationStageDetails trace={trace} />;
  else if (kind === "citation" && citationRecord && state.inspectorTab === "details") body = <CitationDetails row={citationRecord} onViewEvidence={() => citationRecord.evidence && actions.selectEvidence(citationRecord.evidence.key)} onViewCandidate={() => citationRecord.resolvedSourceCandidateId && actions.selectCandidate(citationRecord.resolvedSourceCandidateId)} />;
  else if (kind === "citation" && citationRecord && state.inspectorTab === "evidence") body = <CitationLineage row={citationRecord} onViewEvidence={() => citationRecord.evidence && actions.selectEvidence(citationRecord.evidence.key)} onViewCandidate={() => citationRecord.resolvedSourceCandidateId && actions.selectCandidate(citationRecord.resolvedSourceCandidateId)} />;
  else if (kind === "citation" && citationRecord && state.inspectorTab === "source") body = <CitationSource row={citationRecord} />;
  else if (kind === "citation" && citationRecord && state.inspectorTab === "metadata") body = <CitationMetadata row={citationRecord} />;
  else if (trace && kind === "outcome" && state.inspectorTab === "details") body = <OutcomeDetails trace={trace} />;
  else if (trace && (kind === "evidence_stage" || kind === "answerability" || kind === "generation" || kind === "grounding" || kind === "outcome" || (kind === "citation" && !citationRecord)) && state.inspectorTab === "metadata") body = <><KeyValue label="Trace ID" mono>{trace.trace.trace_id}</KeyValue><KeyValue label="Execution scope">{trace.query.execution_scope || "Unavailable"}</KeyValue><div className="cc-inspector-boundary-note">Only bounded stage metadata from Runtime Trace V1 is exposed.</div></>;
  else if (trace && (kind === "guard" || kind === "structure_recovery" || kind === "graph_recovery" || kind === "graph_node" || kind === "graph_edge" || kind === "evidence_stage" || kind === "answerability" || kind === "generation" || kind === "grounding" || kind === "outcome") && (state.inspectorTab === "evidence" || state.inspectorTab === "source")) body = <div className="cc-inspector-boundary-note">This stage/control-plane entity has no direct Source/Evidence record authority. Select a Candidate, Evidence item, Citation, or Graph path for lineage inspection.</div>;

  const title = entityTitle(kind, candidate, node, edge, path, evidenceRecord, citationRecord);
  return <aside className="cc-inspector" aria-label="Context inspector" data-selected-entity={kind || "none"}>
    <div className="cc-inspector-head"><div><strong>Inspector</strong>{title && <span>{title}</span>}</div><button className="cc-button" aria-label="Close inspector" onClick={() => actions.setInspectorOpen(false)}><X size={14} /></button></div>
    <div className="cc-inspector-tabs" role="tablist">{tabs.map(tab => <button key={tab} role="tab" aria-selected={state.inspectorTab === tab} onClick={() => actions.setInspectorTab(tab)}>{tab}</button>)}</div>
    <div className="cc-inspector-body">{body}</div>
  </aside>;
}
