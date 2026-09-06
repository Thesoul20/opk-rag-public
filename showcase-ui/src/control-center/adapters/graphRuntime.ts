import type { GraphCandidateProvenanceTrace, GraphEdge, RuntimeCandidate, RuntimeTraceV1 } from "../../types/runtimeTrace";

export type RuntimeGraphNodeRole = "seed" | "recovered" | "related";

export interface RuntimeGraphNode {
  id: string;
  label: string;
  role: RuntimeGraphNodeRole;
  seedCandidateIds: string[];
  recoveredCandidateIds: string[];
  observedOutgoingEdgeCount: number;
  observedIncomingEdgeCount: number;
}

export interface RuntimeGraphEdge {
  edgeId: string | null;
  sourceNodeId: string | null;
  targetNodeId: string | null;
  relationType: string | null;
  sourceDocumentId: string | null;
  targetDocumentId: string | null;
  hopDepth: number | null;
  observation: string | null;
  traceIndex: number;
}

export interface RuntimeGraphCandidatePath {
  candidateId: string;
  candidate?: RuntimeCandidate;
  provenance: GraphCandidateProvenanceTrace;
}

export interface RuntimeGraphSubgraph {
  active: boolean;
  stageState: string;
  activationPolicy: string | null;
  activationReasonCode: string | null;
  hopDepth: number | null;
  seedNodeCount: number;
  recoveredNodeCount: number;
  recoveredCandidateCount: number | null;
  traversedEdgeCount: number;
  candidatePoolBefore: number | null;
  candidatePoolAfter: number | null;
  runtimeGoldMetadataUsage: boolean | null;
  nodes: RuntimeGraphNode[];
  edges: RuntimeGraphEdge[];
  candidatePaths: RuntimeGraphCandidatePath[];
}

function numeric(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function graphNodeLabel(id: string | null | undefined): string {
  if (!id) return "Unavailable";
  return id.split("/").filter(Boolean).at(-1) || id;
}

function exactEdge(edge: GraphEdge, index: number): RuntimeGraphEdge {
  return {
    edgeId: edge.edge_id || null,
    sourceNodeId: edge.source_node_id || null,
    targetNodeId: edge.target_node_id || null,
    relationType: edge.relation_type || null,
    sourceDocumentId: edge.source_document_id || null,
    targetDocumentId: edge.target_document_id || null,
    hopDepth: numeric(edge.hop_depth),
    observation: edge.edge_observation || null,
    traceIndex: index,
  };
}

export function buildRuntimeGraphSubgraph(trace: RuntimeTraceV1): RuntimeGraphSubgraph {
  const graph = trace.graph_recovery;
  const active = graph.graph_activated === true;
  const seedNodeIds = [...new Set((graph.seed_node_ids || []).filter(Boolean))];
  const recoveredNodeIds = [...new Set((graph.recovered_node_ids || []).filter(Boolean))];
  const edges = active ? (graph.traversed_edges || []).map(exactEdge) : [];

  const seedCandidateIds = (graph.seed_candidate_ids || []).filter(Boolean);
  const candidateToDocument = new Map<string, string>();
  const candidatePaths: RuntimeGraphCandidatePath[] = [];
  for (const provenance of graph.candidate_provenance || []) {
    if (provenance.candidate_id && provenance.document_id) candidateToDocument.set(provenance.candidate_id, provenance.document_id);
    if (provenance.candidate_id) {
      const candidate = (trace.retrieval.candidates || []).find(row => row.candidate_id === provenance.candidate_id || row.chunk_id === provenance.candidate_id);
      candidatePaths.push({ candidateId: provenance.candidate_id, candidate, provenance });
    }
  }

  const nodeIds = active
    ? [...new Set([...seedNodeIds, ...recoveredNodeIds, ...edges.flatMap(edge => [edge.sourceNodeId, edge.targetNodeId]).filter((id): id is string => Boolean(id))])]
    : [];
  const seedSet = new Set(seedNodeIds);
  const recoveredSet = new Set(recoveredNodeIds);
  const retrievalById = new Map((trace.retrieval.candidates || []).filter(row => row.candidate_id || row.chunk_id).map(row => [row.candidate_id || row.chunk_id || "", row]));
  const nodes = nodeIds.map<RuntimeGraphNode>(id => ({
    id,
    label: graphNodeLabel(id),
    role: recoveredSet.has(id) ? "recovered" : seedSet.has(id) ? "seed" : "related",
    seedCandidateIds: seedCandidateIds.filter(candidateId => {
      const candidate = retrievalById.get(candidateId);
      return candidate?.document_path === id || candidate?.document_id === id;
    }),
    recoveredCandidateIds: [...candidateToDocument.entries()].filter(([, documentId]) => documentId === id).map(([candidateId]) => candidateId),
    observedOutgoingEdgeCount: edges.filter(edge => edge.sourceNodeId === id).length,
    observedIncomingEdgeCount: edges.filter(edge => edge.targetNodeId === id).length,
  }));

  return {
    active,
    stageState: graph.stage_state || "unavailable",
    activationPolicy: graph.activation_policy || null,
    activationReasonCode: graph.activation_reason_code || null,
    hopDepth: numeric(graph.hop_depth),
    seedNodeCount: seedNodeIds.length,
    recoveredNodeCount: recoveredNodeIds.length,
    recoveredCandidateCount: numeric(graph.recovered_candidate_count),
    traversedEdgeCount: edges.length,
    candidatePoolBefore: numeric(graph.candidate_pool_count_before),
    candidatePoolAfter: numeric(graph.candidate_pool_count_after),
    runtimeGoldMetadataUsage: graph.runtime_gold_metadata_usage ?? null,
    nodes,
    edges,
    candidatePaths,
  };
}

export function graphNodeById(trace: RuntimeTraceV1, id: string | null): RuntimeGraphNode | undefined {
  if (!id) return undefined;
  return buildRuntimeGraphSubgraph(trace).nodes.find(node => node.id === id);
}

export function graphEdgeById(trace: RuntimeTraceV1, id: string | null): RuntimeGraphEdge | undefined {
  if (!id) return undefined;
  return buildRuntimeGraphSubgraph(trace).edges.find(edge => edge.edgeId === id);
}

export function graphPathByCandidateId(trace: RuntimeTraceV1, candidateId: string | null): RuntimeGraphCandidatePath | undefined {
  if (!candidateId) return undefined;
  return buildRuntimeGraphSubgraph(trace).candidatePaths.find(path => path.candidateId === candidateId);
}
