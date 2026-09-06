import type { Translate, TranslationKey } from "../i18n";
import type { GraphEdge, RuntimeTraceV1 } from "../types/runtimeTrace";

export type GraphNodeRole = "seed" | "recovered" | "context";

export interface GraphNodeModel {
  id: string;
  label: string;
  role: GraphNodeRole;
  recoveredCandidateId?: string;
}

export interface GraphEdgeModel {
  id: string;
  sourceNodeId: string;
  targetNodeId: string;
  relationType: string;
  hopDepth: number | null;
  observation?: string | null;
}

export interface GraphCandidateLinkModel {
  candidateId: string;
  recoveredNodeId?: string;
  documentPath?: string | null;
  sectionPath?: string[] | null;
  finalRank: number | null;
  rerankPosition: number | null;
  rerankScore: number | null;
  evidenceId?: string;
  evidencePosition: number | null;
  selectedAsEvidence: boolean;
}

export interface GraphRecoveryModel {
  active: boolean;
  stageState: string;
  reasonCode?: string | null;
  reasonExplanation: string;
  activationPolicy?: string | null;
  hopDepth: number | null;
  nodes: GraphNodeModel[];
  edges: GraphEdgeModel[];
  candidates: GraphCandidateLinkModel[];
  seedCount: number;
  recoveredNodeCount: number;
  recoveredCandidateCount: number;
  traversedEdgeCount: number;
  evidenceLinkCount: number;
  candidatePoolBefore: number | null;
  candidatePoolAfter: number | null;
  runtimeGoldMetadataUsage: boolean | null;
  fabricatedNodeCount: 0;
  fabricatedEdgeCount: 0;
  fabricatedCandidateCount: 0;
}

const reasonKey: Record<string, TranslationKey> = {
  query_signal_with_seed_graph_availability: "graphViz.reason.querySignal",
  missing_query_or_retrieval_signal: "graphViz.reason.noSignal",
};

function n(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function basename(nodeId: string): string {
  return nodeId.split("/").filter(Boolean).at(-1) || nodeId;
}

function edgeFromTrace(edge: GraphEdge, index: number): GraphEdgeModel | null {
  if (!edge.source_node_id || !edge.target_node_id) return null;
  return {
    id: edge.edge_id || `trace-edge-${index}`,
    sourceNodeId: edge.source_node_id,
    targetNodeId: edge.target_node_id,
    relationType: edge.relation_type || "relation",
    hopDepth: n(edge.hop_depth),
    observation: edge.edge_observation,
  };
}

export function buildGraphRecoveryModel(trace: RuntimeTraceV1 | undefined, t: Translate): GraphRecoveryModel | null {
  if (!trace) return null;
  const graph = trace.graph_recovery;
  const active = graph.graph_activated === true;
  const seedIds = [...new Set((graph.seed_node_ids || []).filter(Boolean))];
  const recoveredIds = [...new Set((graph.recovered_node_ids || []).filter(Boolean))];
  const edges = active ? (graph.traversed_edges || []).map(edgeFromTrace).filter((row): row is GraphEdgeModel => row !== null) : [];

  const provenance = graph.candidate_provenance || [];
  const candidateToNode = new Map<string, string>();
  for (const row of provenance) {
    if (row.candidate_id && row.document_id) candidateToNode.set(row.candidate_id, row.document_id);
  }
  const nodeToCandidate = new Map<string, string>();
  for (const [candidateId, nodeId] of candidateToNode) nodeToCandidate.set(nodeId, candidateId);

  const nodeIds = active
    ? [...new Set([...seedIds, ...recoveredIds, ...edges.flatMap((edge) => [edge.sourceNodeId, edge.targetNodeId])])]
    : [];
  const seedSet = new Set(seedIds);
  const recoveredSet = new Set(recoveredIds);
  const nodes = nodeIds.map<GraphNodeModel>((id) => ({
    id,
    label: basename(id),
    role: recoveredSet.has(id) ? "recovered" : seedSet.has(id) ? "seed" : "context",
    recoveredCandidateId: nodeToCandidate.get(id),
  }));

  const retrievalById = new Map((trace.retrieval.candidates || []).filter((row) => row.candidate_id).map((row) => [row.candidate_id!, row]));
  const rerankById = new Map((trace.rerank.ranked_candidates || []).filter((row) => row.candidate_id).map((row) => [row.candidate_id!, row]));
  const evidenceByCandidate = new Map((trace.evidence.evidence_items || []).filter((row) => row.source_candidate_id).map((row) => [row.source_candidate_id!, row]));
  const candidates = active ? (graph.recovered_candidate_ids || []).filter(Boolean).map<GraphCandidateLinkModel>((candidateId) => {
    const candidate = retrievalById.get(candidateId);
    const rerank = rerankById.get(candidateId);
    const evidence = evidenceByCandidate.get(candidateId);
    return {
      candidateId,
      recoveredNodeId: candidateToNode.get(candidateId),
      documentPath: candidate?.document_path,
      sectionPath: candidate?.section_path,
      finalRank: n(candidate?.final_rank),
      rerankPosition: n(rerank?.rerank_position),
      rerankScore: n(rerank?.rerank_score),
      evidenceId: evidence?.evidence_id || undefined,
      evidencePosition: n(evidence?.evidence_position),
      selectedAsEvidence: Boolean(evidence),
    };
  }) : [];

  const reasonExplanation = !active && trace.structure_recovery.triggered
    ? t("graphViz.reason.structureHandled")
    : graph.activation_reason_code && reasonKey[graph.activation_reason_code]
      ? t(reasonKey[graph.activation_reason_code])
      : active
        ? t("graphViz.reason.activeFallback")
        : t("graphViz.reason.notTriggered");

  return {
    active,
    stageState: graph.stage_state || "unavailable",
    reasonCode: graph.activation_reason_code,
    reasonExplanation,
    activationPolicy: graph.activation_policy,
    hopDepth: n(graph.hop_depth),
    nodes,
    edges,
    candidates,
    seedCount: seedIds.length,
    recoveredNodeCount: recoveredIds.length,
    recoveredCandidateCount: n(graph.recovered_candidate_count) ?? candidates.length,
    traversedEdgeCount: edges.length,
    evidenceLinkCount: candidates.filter((row) => row.selectedAsEvidence).length,
    candidatePoolBefore: n(graph.candidate_pool_count_before),
    candidatePoolAfter: n(graph.candidate_pool_count_after),
    runtimeGoldMetadataUsage: graph.runtime_gold_metadata_usage ?? null,
    fabricatedNodeCount: 0,
    fabricatedEdgeCount: 0,
    fabricatedCandidateCount: 0,
  };
}
