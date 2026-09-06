import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { KnowledgeGraphPanel } from "../components/KnowledgeGraphPanel";
import { PresentationProvider } from "../i18n";
import { emptyReplayState } from "../lib/sse";
import { buildGraphRecoveryModel } from "../presentation/graphRecovery";
import { loadTrace } from "./fixtures";

const passthrough = (key: any) => String(key);

describe("TASK-0231 Interactive Graph Retrieval", () => {
  it("does not fabricate graph nodes or edges for S01/S02/S04", () => {
    for (const sid of ["s01", "s02", "s04"] as const) {
      const model = buildGraphRecoveryModel(loadTrace(sid), passthrough as any)!;
      expect(model.active).toBe(false);
      expect(model.nodes).toHaveLength(0);
      expect(model.edges).toHaveLength(0);
      expect(model.candidates).toHaveLength(0);
      expect(model.fabricatedNodeCount).toBe(0);
      expect(model.fabricatedEdgeCount).toBe(0);
      expect(model.fabricatedCandidateCount).toBe(0);
    }
  });

  it("links S03 authoritative node/edge/candidate/rerank/evidence identities", () => {
    const trace = loadTrace("s03");
    const model = buildGraphRecoveryModel(trace, passthrough as any)!;
    expect(model.active).toBe(true);
    expect(model.hopDepth).toBe(1);
    expect(model.edges).toHaveLength(trace.graph_recovery.traversed_edges?.length || 0);
    expect(model.recoveredCandidateCount).toBe(1);
    const candidateId = trace.graph_recovery.recovered_candidate_ids?.[0];
    expect(candidateId).toBeTruthy();
    const link = model.candidates.find((row) => row.candidateId === candidateId)!;
    const rerank = trace.rerank.ranked_candidates?.find((row) => row.candidate_id === candidateId)!;
    const evidence = trace.evidence.evidence_items.find((row) => row.source_candidate_id === candidateId)!;
    expect(link.recoveredNodeId).toBe("Tauri/公众号发布 SaaS Demo/MVP 功能拆解.md");
    expect(link.rerankPosition).toBe(rerank.rerank_position);
    expect(link.rerankScore).toBe(rerank.rerank_score);
    expect(link.evidenceId).toBe(evidence.evidence_id);
    expect(link.selectedAsEvidence).toBe(true);
  });

  it("renders S03 one-hop paths and reverse-highlights Graph Candidate from Evidence", () => {
    const { container } = render(
      <PresentationProvider initialLocale="en" initialViewMode="engineer">
        <KnowledgeGraphPanel trace={loadTrace("s03")} replay={emptyReplayState()} />
      </PresentationProvider>,
    );
    expect(screen.getByTestId("interactive-graph-recovery")).toBeInTheDocument();
    expect(screen.getAllByText("LINKS_TO").length).toBeGreaterThan(0);
    expect(screen.getAllByText("1-hop").length).toBeGreaterThan(0);
    expect(screen.getByText("C4")).toBeInTheDocument();
    const evidenceButton = container.querySelector("button.evidence-link") as HTMLButtonElement;
    fireEvent.click(evidenceButton);
    expect(container.querySelector(".graph-candidate-row.selected")).toBeInTheDocument();
    expect(container.querySelector(".graph-entity-node.role-recovered.selected")).toBeInTheDocument();
  });

  it("makes edges inspectable without exposing hidden reasoning", () => {
    const { container } = render(
      <PresentationProvider initialLocale="en" initialViewMode="engineer">
        <KnowledgeGraphPanel trace={loadTrace("s03")} />
      </PresentationProvider>,
    );
    const edge = container.querySelector("button.graph-entity-edge") as HTMLButtonElement;
    fireEvent.click(edge);
    expect(screen.getByText("Edge observation")).toBeInTheDocument();
    expect(screen.getByText("traversed_query_edge")).toBeInTheDocument();
    expect(screen.queryByText(/chain-of-thought/i)).not.toBeInTheDocument();
  });

  it("renders localized inactive and executive graph views truthfully", () => {
    const { rerender } = render(
      <PresentationProvider initialLocale="zh-CN" initialViewMode="executive">
        <KnowledgeGraphPanel trace={loadTrace("s02")} />
      </PresentationProvider>,
    );
    expect(screen.getByTestId("graph-not-triggered")).toBeInTheDocument();
    expect(screen.getByText(/Structure Recovery/)).toBeInTheDocument();
    rerender(
      <PresentationProvider initialLocale="zh-CN" initialViewMode="executive">
        <KnowledgeGraphPanel trace={loadTrace("s03")} />
      </PresentationProvider>,
    );
    expect(screen.getByText("为什么启动 Graph？")).toBeInTheDocument();
    expect(screen.getByText(/仅允许 one-hop/)).toBeInTheDocument();
  });
});
