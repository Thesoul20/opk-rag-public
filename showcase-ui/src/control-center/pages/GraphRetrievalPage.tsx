import { ArrowRight, GitBranch, Network } from "lucide-react";
import { useShowcase } from "../../state/ShowcaseState";
import type { RuntimeCandidate, RuntimeTraceV1 } from "../../types/runtimeTrace";
import { buildRuntimeGraphSubgraph, graphNodeLabel, type RuntimeGraphEdge, type RuntimeGraphNode } from "../adapters/graphRuntime";
import { candidateRelationship } from "../adapters/runtimeTrace";
import { buildEvidenceValidationModel } from "../adapters/evidenceRuntime";
import { guardReasonLabel } from "../components/GuardRecoveryControlPlane";
import { Badge, Button, EmptyState, KeyValue, Metric, Panel, StatusBadge } from "../components/Primitives";
import { useControlCenter } from "../state/ControlCenterState";

function value(input: unknown): string | number {
  return input == null ? "Unavailable" : typeof input === "number" ? input : String(input);
}

function roleLabel(role: RuntimeGraphNode["role"]): string {
  if (role === "seed") return "Seed Node";
  if (role === "recovered") return "Recovered Node";
  return "Related Node";
}

function nodeTone(role: RuntimeGraphNode["role"]): string {
  return role === "recovered" ? "success" : role === "seed" ? "info" : "partial";
}

function candidateFor(trace: RuntimeTraceV1, candidateId: string): RuntimeCandidate | undefined {
  return trace.retrieval.candidates.find(candidate => candidate.candidate_id === candidateId || candidate.chunk_id === candidateId);
}

function relationship(trace: RuntimeTraceV1, candidate?: RuntimeCandidate): string {
  if (!candidate) return "Candidate";
  const state = candidateRelationship(trace, candidate);
  return state === "evidence_selected" ? "Evidence Selected" : state === "context_candidate" ? "Context Candidate" : "Candidate";
}

function GraphNodeButton({ node }: { node: RuntimeGraphNode }) {
  const { state, actions } = useControlCenter();
  return <button type="button" className="cc-graph-node" data-role={node.role} data-selected={state.selectedGraphNodeId === node.id} onClick={() => actions.selectGraphNode(node.id)} title={node.id}>
    <span className="cc-label">{roleLabel(node.role)}</span>
    <strong>{node.label}</strong>
    <small>{node.observedOutgoingEdgeCount} out · {node.observedIncomingEdgeCount} in observed</small>
  </button>;
}

function GraphEdgeButton({ edge }: { edge: RuntimeGraphEdge }) {
  const { state, actions } = useControlCenter();
  const selectable = Boolean(edge.edgeId);
  return <button type="button" className="cc-graph-edge" data-selected={Boolean(edge.edgeId && state.selectedGraphEdgeId === edge.edgeId)} disabled={!selectable} onClick={() => edge.edgeId && actions.selectGraphEdge(edge.edgeId)} title={edge.edgeId || "Edge identity unavailable"}>
    <span className="cc-mono">{edge.relationType || "relation unavailable"}</span>
    <ArrowRight size={17} aria-hidden="true" />
    <small>hop {edge.hopDepth ?? "?"}</small>
  </button>;
}

function RuntimeSubgraphCanvas({ trace }: { trace: RuntimeTraceV1 }) {
  const graph = buildRuntimeGraphSubgraph(trace);
  const nodeById = new Map(graph.nodes.map(node => [node.id, node]));
  if (!graph.active) return <EmptyState title="Graph Recovery skipped" detail="No authoritative traversed runtime graph path exists for this execution. The UI does not synthesize topology." />;
  if (!graph.edges.length) return <EmptyState title="No traversed edges" detail="Graph Recovery was active, but this Runtime Trace contains no traversed edge records. Nothing is reconstructed." />;
  return <div className="cc-runtime-subgraph" aria-label="Runtime Graph Recovery Subgraph" data-layout="deterministic-one-hop">
    <div className="cc-runtime-subgraph-head"><span>Seed / source node</span><span>Authoritative edge</span><span>Target / recovered node</span></div>
    {graph.edges.map((edge, index) => {
      const fallback = (id: string): RuntimeGraphNode => ({ id, label: graphNodeLabel(id), role: "related", seedCandidateIds: [], recoveredCandidateIds: [], observedOutgoingEdgeCount: 0, observedIncomingEdgeCount: 0 });
      const source = edge.sourceNodeId ? (nodeById.get(edge.sourceNodeId) || fallback(edge.sourceNodeId)) : null;
      const target = edge.targetNodeId ? (nodeById.get(edge.targetNodeId) || fallback(edge.targetNodeId)) : null;
      return <div className="cc-runtime-subgraph-row" key={edge.edgeId || `trace-edge-row-${index}`} data-edge-index={edge.traceIndex}>
        {source ? <GraphNodeButton node={source} /> : <div className="cc-graph-node-missing">Source node identity unavailable</div>}
        <GraphEdgeButton edge={edge} />
        {target ? <GraphNodeButton node={target} /> : <div className="cc-graph-node-missing">Target node identity unavailable</div>}
      </div>;
    })}
  </div>;
}

function ObservedNodes({ trace }: { trace: RuntimeTraceV1 }) {
  const { actions } = useControlCenter();
  const graph = buildRuntimeGraphSubgraph(trace);
  if (!graph.nodes.length) return null;
  return <Panel title={`Observed Nodes · ${graph.nodes.length}`} action={<Badge tone="info">Graph Node ≠ Candidate</Badge>}>
    <div className="cc-graph-node-list">{graph.nodes.map(node => <button type="button" key={node.id} className="cc-graph-node-list-item" onClick={() => actions.selectGraphNode(node.id)}>
      <Badge tone={nodeTone(node.role)}>{roleLabel(node.role)}</Badge>
      <strong>{node.label}</strong>
      <span className="cc-mono">{node.id}</span>
      <small>{node.observedOutgoingEdgeCount} outgoing · {node.observedIncomingEdgeCount} incoming observed edges</small>
    </button>)}</div>
  </Panel>;
}

function EdgeTable({ trace }: { trace: RuntimeTraceV1 }) {
  const { state, actions } = useControlCenter();
  const graph = buildRuntimeGraphSubgraph(trace);
  if (!graph.edges.length) return null;
  return <Panel title={`Traversed Edges · ${graph.edges.length}`} action={<Badge tone="info">Duplicate edge IDs preserved</Badge>}>
    <div className="cc-candidate-table-wrap"><table className="cc-candidate-table cc-graph-edge-table">
      <thead><tr><th>Edge ID</th><th>Source</th><th>Relation</th><th>Target</th><th>Hop</th><th>Observation</th></tr></thead>
      <tbody>{graph.edges.map((edge, index) => <tr key={edge.edgeId || `edge-row-${index}`} data-selected={Boolean(edge.edgeId && state.selectedGraphEdgeId === edge.edgeId)} onClick={() => edge.edgeId && actions.selectGraphEdge(edge.edgeId)}>
        <td className="cc-mono">{edge.edgeId || "Unavailable"}</td><td>{graphNodeLabel(edge.sourceNodeId)}</td><td className="cc-mono">{edge.relationType || "Unavailable"}</td><td>{graphNodeLabel(edge.targetNodeId)}</td><td>{value(edge.hopDepth)}</td><td className="cc-mono">{edge.observation || "Unavailable"}</td>
      </tr>)}</tbody>
    </table></div>
  </Panel>;
}

function RecoveredCandidates({ trace }: { trace: RuntimeTraceV1 }) {
  const { actions } = useControlCenter();
  const graph = buildRuntimeGraphSubgraph(trace);
  if (!graph.candidatePaths.length) return graph.active ? <Panel title="Recovered Candidate Paths"><EmptyState title="No candidate graph paths" detail="This trace does not provide bounded candidate graph provenance. No path is inferred." /></Panel> : null;
  return <Panel title={`Recovered Candidate Paths · ${graph.candidatePaths.length}`} action={<Badge tone="info">Candidate ≠ Evidence</Badge>}>
    <div className="cc-graph-candidate-paths">{graph.candidatePaths.map(path => {
      const candidate = path.candidate || candidateFor(trace, path.candidateId);
      const first = path.provenance.graph_path?.[0];
      return <div className="cc-graph-candidate-path" key={path.candidateId}>
        <div><span className="cc-label">Recovered Candidate</span><strong>{candidate?.document_path || path.provenance.document_id || path.candidateId}</strong><span>{candidate?.section_path?.join(" / ") || path.provenance.section_id || "Section unavailable"}</span></div>
        <div className="cc-graph-path-inline"><span>{graphNodeLabel(first?.source_node_id || path.provenance.source_seed_candidate_id || "Seed unavailable")}</span><code>{first?.edge_type || path.provenance.edge_types?.[0] || "relation unavailable"}</code><span>{graphNodeLabel(first?.target_node_id || path.provenance.document_id || "Target unavailable")}</span></div>
        <div className="cc-result-meta"><Badge tone="info">Hop {path.provenance.graph_hop_count ?? "?"}</Badge><Badge tone={candidate?.graph_recovered ? "success" : "info"}>{relationship(trace, candidate)}</Badge>{candidate?.final_rank != null && <span>Final #{candidate.final_rank}</span>}</div>
        <div className="cc-inline-actions"><Button onClick={() => actions.selectGraphPath(path.candidateId)}>Inspect path</Button>{candidate && <Button onClick={() => actions.selectCandidate(path.candidateId)}>View Candidate</Button>}{buildEvidenceValidationModel(trace).evidence.find(row => row.sourceCandidateId === path.candidateId) && <Button variant="primary" onClick={() => { const row=buildEvidenceValidationModel(trace).evidence.find(item => item.sourceCandidateId === path.candidateId)!; actions.setActiveNav("evidence"); actions.selectEvidence(row.key); }}>View Evidence</Button>}</div>
      </div>;
    })}</div>
  </Panel>;
}

export function GraphRuntimeInspector({ trace }: { trace: RuntimeTraceV1 }) {
  const { state } = useControlCenter();
  const graph = buildRuntimeGraphSubgraph(trace);
  const hopWarning = graph.hopDepth != null && graph.hopDepth > 1;
  const goldWarning = graph.runtimeGoldMetadataUsage === true;
  return <div className="cc-graph-inspector-stack" data-presentation-mode={state.mode} data-runtime-graph-authority="trace-v1">
    <Panel title="Graph Execution Summary" action={<StatusBadge status={graph.active ? "completed" : graph.stageState === "failed" ? "failed" : "unavailable"} label={graph.active ? "one-hop active" : graph.stageState} />}>
      <div className="cc-metric-grid cc-graph-summary-grid">
        <Metric label="Activation policy" value={value(graph.activationPolicy)} />
        <Metric label="Hop" value={graph.hopDepth == null ? "Unavailable" : `${graph.hopDepth} / max 1`} />
        <Metric label="Seed nodes" value={graph.seedNodeCount} />
        <Metric label="Traversed edges" value={graph.traversedEdgeCount} />
        <Metric label="Recovered nodes" value={graph.recoveredNodeCount} />
        <Metric label="Recovered candidates" value={value(graph.recoveredCandidateCount)} />
        <Metric label="Candidate pool" value={graph.candidatePoolBefore == null || graph.candidatePoolAfter == null ? "Unavailable" : `${graph.candidatePoolBefore} → ${graph.candidatePoolAfter}`} />
        <Metric label="Runtime Gold" value={graph.runtimeGoldMetadataUsage === true ? "USED" : graph.runtimeGoldMetadataUsage === false ? "false" : "Unavailable"} />
      </div>
      <div className="cc-graph-authority-note"><Network size={16} /><div><strong>Runtime Graph Recovery Subgraph</strong><span>This view contains only graph records observed by the active request. It is not the complete knowledge graph.</span></div></div>
      <KeyValue label="Activation reason"><span>{guardReasonLabel(graph.activationReasonCode)} <code className="cc-reason-code">{graph.activationReasonCode || "unavailable"}</code></span></KeyValue>
      {(hopWarning || goldWarning) && <div className="cc-governance-warning">Governance warning: {hopWarning ? `observed hop depth ${graph.hopDepth} exceeds frozen max 1. ` : ""}{goldWarning ? "runtime Gold metadata usage is true. " : ""}The authoritative trace is displayed unchanged; the UI performs no additional traversal.</div>}
    </Panel>

    <Panel title="Runtime Subgraph" action={<Badge tone="info">Deterministic one-hop layout</Badge>}>
      <RuntimeSubgraphCanvas trace={trace} />
    </Panel>
    {graph.active && <ObservedNodes trace={trace} />}
    {state.mode === "operator" && <EdgeTable trace={trace} />}
    <RecoveredCandidates trace={trace} />
    <Panel title="Semantic Boundaries"><div className="cc-graph-boundaries"><span>Graph Node ≠ Candidate</span><span>Candidate ≠ Evidence</span><span>UI selection ≠ Graph traversal</span></div></Panel>
  </div>;
}

export function GraphRetrievalPage() {
  const { state } = useShowcase();
  const trace = state.trace;
  return <div className="cc-page" data-graph-retrieval-page="active">
    <header className="cc-page-heading"><div><h1 className="cc-title">Graph Retrieval</h1><p>Inspect the authoritative one-hop graph path used by the active Runtime Trace.</p></div><Badge tone="info"><GitBranch size={12} /> Runtime Trace V1 only</Badge></header>
    {!trace ? <Panel><EmptyState title="No active Runtime Trace" detail="Run Search/Ask or reopen a persisted turn trace, then inspect its query-scoped Graph Recovery subgraph here." /></Panel> : <GraphRuntimeInspector trace={trace} />}
  </div>;
}
