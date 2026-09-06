import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CandidateRerankEvidencePanel } from "../components/CandidateRerankEvidencePanel";
import { KnowledgeGraphPanel } from "../components/KnowledgeGraphPanel";
import { PresentationProvider } from "../i18n";
import { emptyReplayState } from "../lib/sse";
import { buildCandidateLifecycleModel } from "../presentation/candidateLifecycle";
import { loadTrace } from "./fixtures";

const t = ((key: string) => key) as any;

function renderCandidate(sid: "s01"|"s02"|"s03"|"s04", locale: "en"|"zh-CN" = "en", mode: "engineer"|"executive" = "engineer") {
  return render(<PresentationProvider initialLocale={locale} initialViewMode={mode}><CandidateRerankEvidencePanel trace={loadTrace(sid)} replay={emptyReplayState()} compact={mode === "executive"} /></PresentationProvider>);
}

describe("TASK-0232 Candidate → Rerank → Evidence", () => {
  it("uses runtime rerank_position ordering rather than sorting by score", () => {
    const model = buildCandidateLifecycleModel(loadTrace("s01"), t)!;
    expect(model.rows.map((row) => row.rerankPosition)).toEqual([1,2,3,4,5]);
    expect(model.rows[1].rerankScore!).toBeGreaterThan(model.rows[0].rerankScore!);
    expect(model.uiRerankSortUsesRuntimePosition).toBe(true);
  });

  it("shows S02 Structure provenance and unified rerank path", () => {
    const trace = loadTrace("s02");
    const model = buildCandidateLifecycleModel(trace, t)!;
    expect(model.structureExpandedCount).toBeGreaterThan(0);
    expect(model.rows.some((row) => row.sources.includes("structure"))).toBe(true);
    expect(model.rerankInputCount).toBe(trace.rerank.input_candidate_count);
    expect(model.evidenceCount).toBe(trace.evidence.evidence_count);
  });

  it("preserves the S03 Graph candidate identity through rerank and Evidence", () => {
    const trace = loadTrace("s03");
    const graphId = trace.graph_recovery.recovered_candidate_ids![0];
    const model = buildCandidateLifecycleModel(trace, t)!;
    const row = model.rows.find((candidate) => candidate.candidateId === graphId)!;
    expect(row.graphRecovered).toBe(true);
    expect(row.sources).toContain("graph");
    expect(row.rerankPosition).toBe(5);
    expect(row.evidenceId).toBe("C4");
    expect(model.s03GraphCandidateIdentityChainValid).toBe(true);
    expect(model.orphanEvidenceCandidateCount).toBe(0);
  });

  it("keeps Candidate separate from Evidence and keeps S04 refused", () => {
    const trace = loadTrace("s04");
    const model = buildCandidateLifecycleModel(trace, t)!;
    expect(model.candidateEvidenceSemanticSeparation).toBe(true);
    expect(model.rows.length).toBeGreaterThan(model.evidence.length);
    expect(model.outcomeStatus).toBe("refused");
    expect(model.outcomeStatus).not.toBe("failed");
  });

  it("renders authoritative rank, score, Evidence boundary and filtering", () => {
    renderCandidate("s03");
    expect(screen.getByTestId("candidate-rerank-evidence")).toBeInTheDocument();
    expect(screen.getByText("Candidate → Rerank → Evidence")).toBeInTheDocument();
    expect(screen.getByText("Evidence Composition")).toBeInTheDocument();
    expect(screen.getAllByText("0.6919").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Graph" }));
    expect(screen.getAllByText(/MVP 功能拆解/).length).toBeGreaterThan(0);
  });

  it("supports candidate-id cross-panel highlighting without document-name matching", () => {
    const trace = loadTrace("s03");
    const graphId = trace.graph_recovery.recovered_candidate_ids![0];
    render(<PresentationProvider initialLocale="en" initialViewMode="engineer"><KnowledgeGraphPanel trace={trace} replay={emptyReplayState()} /><CandidateRerankEvidencePanel trace={trace} replay={emptyReplayState()} /></PresentationProvider>);
    act(() => { window.dispatchEvent(new CustomEvent("opk-showcase-candidate-selected", { detail: { candidateId: graphId } })); });
    const graphRow = document.querySelector(`.graph-candidate-row[data-candidate-id="${graphId}"]`);
    expect(graphRow).toHaveClass("selected");
  });

  it("renders Chinese executive funnel from trace counts", () => {
    renderCandidate("s03", "zh-CN", "executive");
    expect(screen.getByText("候选如何变成证据？")).toBeInTheDocument();
    expect(screen.getByText("Candidate ≠ Evidence")).toBeInTheDocument();
  });
});
