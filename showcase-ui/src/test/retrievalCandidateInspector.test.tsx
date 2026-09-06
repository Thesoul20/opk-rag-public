import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { CandidatePool, RerankingSummary, RetrievalCandidateInspection } from "../control-center/components/RetrievalCandidateInspector";
import { Inspector } from "../control-center/layout/Inspector";
import { ControlCenterStateProvider, useControlCenter } from "../control-center/state/ControlCenterState";
import { candidateProvenanceLabels, candidateRelationship, rankMovement, runtimeBoundaryContract } from "../control-center/adapters/runtimeTrace";
import { loadTrace } from "./fixtures";

const mocked = vi.hoisted(() => ({ trace: undefined as RuntimeTraceV1 | undefined }));
vi.mock("../state/ShowcaseState", () => ({ useShowcase: () => ({ state: { trace: mocked.trace }, actions: {} }) }));

function renderInspector(trace: RuntimeTraceV1) {
  mocked.trace = trace;
  return render(<ControlCenterStateProvider><RetrievalCandidateInspection trace={trace} /><Inspector /></ControlCenterStateProvider>);
}

function SelectionEcho() {
  const { state, actions } = useControlCenter();
  return <><span data-testid="selection">{state.selectedCandidateId || "none"}</span><button onClick={() => actions.selectCandidate("missing-candidate")}>Select stale</button></>;
}

describe("TASK-0273 Retrieval / Candidate / Reranking Inspector", () => {
  beforeEach(() => { mocked.trace = undefined; });

  it("renders authoritative retrieval overview with vector and lexical counts kept distinct", () => {
    const trace = loadTrace("s02");
    renderInspector(trace);
    const vector = screen.getByText("Vector").closest(".cc-metric") as HTMLElement;
    const lexical = screen.getByText("Lexical / BM25").closest(".cc-metric") as HTMLElement;
    expect(within(vector).getByText("20")).toBeInTheDocument();
    expect(within(lexical).getByText("0")).toBeInTheDocument();
    expect(screen.getByText("BGE Reranking")).toBeInTheDocument();
    expect(screen.getByText("Candidate ≠ Evidence")).toBeInTheDocument();
  });

  it("selects a candidate presentation-only and populates Details and Evidence inspector tabs", async () => {
    const user = userEvent.setup();
    const trace = loadTrace("s02");
    renderInspector(trace);
    await user.click(screen.getByRole("row", { name: "Inspect candidate 3" }));
    const inspector = screen.getByLabelText("Context inspector");
    expect(within(inspector).getByText("#4")).toBeInTheDocument();
    expect(within(inspector).getByText("#3")).toBeInTheDocument();
    expect(within(inspector).getByText("+1")).toBeInTheDocument();
    expect(within(inspector).getByText("Evidence Selected")).toBeInTheDocument();
    await user.click(within(inspector).getByRole("tab", { name: "evidence" }));
    expect(within(inspector).getByText("Candidate ≠ Evidence")).toBeInTheDocument();
    expect(runtimeBoundaryContract().uiDecisionAuthority).toBe(false);
    expect(runtimeBoundaryContract().candidateEqualsEvidence).toBe(false);
  });

  it("renders authority-backed structure and graph provenance and deterministic rank movement", () => {
    const structure = loadTrace("s02").retrieval.candidates.find(candidate => candidate.structure_expanded)!;
    const graph = loadTrace("s03").retrieval.candidates.find(candidate => candidate.graph_recovered)!;
    expect(candidateProvenanceLabels(structure)).toContain("Structure expanded");
    expect(candidateProvenanceLabels(graph)).toContain("Graph recovered");
    expect(rankMovement(5, 3)).toBe(2);
    expect(rankMovement(null, 3)).toBeNull();
    expect(candidateRelationship(loadTrace("s03"), graph)).toBe("evidence_selected");
  });

  it("keeps a non-evidence final candidate distinct instead of inferring Evidence from rank", () => {
    const trace = loadTrace("s02");
    const evidenceIds = new Set(trace.evidence.evidence_items.map(item => item.source_candidate_id));
    const candidate = trace.retrieval.candidates.find(row => !evidenceIds.has(row.candidate_id))!;
    expect(candidate).toBeTruthy();
    expect(candidateRelationship(trace, candidate)).not.toBe("evidence_selected");
    mocked.trace = trace;
    render(<ControlCenterStateProvider><CandidatePool trace={trace} /></ControlCenterStateProvider>);
    expect(screen.getAllByText(/Candidate$/).length).toBeGreaterThan(0);
  });

  it("fails soft when reranking is unavailable and never reconstructs values", () => {
    const trace = structuredClone(loadTrace("s01"));
    trace.rerank.stage_state = "not_applicable";
    mocked.trace = trace;
    render(<RerankingSummary trace={trace} />);
    expect(screen.getByText("Reranking unavailable")).toBeInTheDocument();
    expect(screen.getByText(/No ranking values are reconstructed/)).toBeInTheDocument();
  });

  it("clears stale candidate selection when active trace cannot resolve the selected candidate", async () => {
    const user = userEvent.setup();
    const trace = loadTrace("s01");
    mocked.trace = trace;
    render(<ControlCenterStateProvider><SelectionEcho /><Inspector /></ControlCenterStateProvider>);
    await user.click(screen.getByRole("button", { name: "Select stale" }));
    await waitFor(() => expect(screen.getByTestId("selection")).toHaveTextContent("none"));
  });
});
