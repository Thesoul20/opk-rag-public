import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { buildEvidenceValidationModel } from "../control-center/adapters/evidenceRuntime";
import { EvidenceValidationInspector, EvidenceValidationPage } from "../control-center/pages/EvidenceValidationPage";
import { EvidenceValidationSummary } from "../control-center/components/EvidenceValidationSummary";
import { Inspector } from "../control-center/layout/Inspector";
import { ControlCenterStateProvider, useControlCenter } from "../control-center/state/ControlCenterState";
import { loadTrace } from "./fixtures";

const mocked = vi.hoisted(() => ({ trace: undefined as RuntimeTraceV1 | undefined }));
vi.mock("../state/ShowcaseState", () => ({ useShowcase: () => ({ state: { trace: mocked.trace }, actions: {} }) }));

function loadS03Ask(): RuntimeTraceV1 {
  return JSON.parse(readFileSync(resolve(process.cwd(), "..", "evaluation-data", "showcase", "runtime_trace_v1_live_s03_ask_task0233.json"), "utf-8")) as RuntimeTraceV1;
}
function renderInspector(trace: RuntimeTraceV1) {
  mocked.trace = trace;
  return render(<ControlCenterStateProvider><EvidenceValidationInspector trace={trace} /><Inspector /></ControlCenterStateProvider>);
}
function NavEcho() { const { state } = useControlCenter(); return <span data-testid="active-nav">{state.activeNav}</span>; }
function ModeToggle() { const { state, actions } = useControlCenter(); return <><span data-testid="mode">{state.mode}</span><button onClick={() => actions.setMode(state.mode === "operator" ? "showcase" : "operator")}>toggle mode</button></>; }

describe("TASK-0276 Evidence / Answerability / Grounding / Citation Inspector", () => {
  beforeEach(() => { mocked.trace = undefined; });

  it("keeps Search scope truthfully ending at Evidence without fake downstream execution", () => {
    const trace = loadTrace("s03");
    const model = buildEvidenceValidationModel(trace);
    expect(model.searchOnly).toBe(true);
    expect(model.stages.filter((s) => ["answerability","generation","grounding","citation"].includes(s.id)).every((s) => s.state === "not_applicable")).toBe(true);
    renderInspector(trace);
    expect(screen.getByText(/Search scope ends at Evidence/)).toBeInTheDocument();
    expect(screen.queryByText("deepseek-v4-flash")).not.toBeInTheDocument();
  });

  it("preserves the S03 Ask Graph Candidate → Evidence C4 → Citation C4 identity chain", () => {
    const trace = loadS03Ask();
    const model = buildEvidenceValidationModel(trace);
    const evidence = model.evidence.find((row) => row.evidenceId === "C4");
    const citation = model.citations.find((row) => row.citationId === "C4");
    expect(evidence?.sourceCandidateId).toBe("30b273ba-f462-452b-9fd7-ad3456fde347");
    expect(evidence?.graphRecovered).toBe(true);
    expect(citation?.evidenceId).toBe("C4");
    expect(citation?.resolvedSourceCandidateId).toBe("30b273ba-f462-452b-9fd7-ad3456fde347");
    expect(citation?.evidenceFound).toBe(true);
    expect(citation?.candidateIdentityMatches).toBe(true);
    expect(model.orphanCitationCount).toBe(0);
  });

  it("renders Evidence inventory with line-level provenance and graph provenance", () => {
    renderInspector(loadS03Ask());
    expect(screen.getByText("Evidence Inventory · 4")).toBeInTheDocument();
    expect(screen.getAllByText("L10–L78").length).toBeGreaterThan(0);
    expect(screen.getByText("Graph recovered")).toBeInTheDocument();
    expect(screen.getAllByText("Candidate ≠ Evidence").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Evidence ≠ Citation").length).toBeGreaterThan(0);
  });

  it("selects Evidence presentation-only and links back to the exact source Candidate", async () => {
    const user = userEvent.setup();
    renderInspector(loadS03Ask());
    const row = document.querySelector('tr[data-evidence-key="evidence:C4"]') as HTMLElement;
    expect(row).toBeTruthy();
    await user.click(row);
    const inspector = screen.getByLabelText("Context inspector");
    expect(inspector).toHaveAttribute("data-selected-entity", "evidence");
    expect(within(inspector).getByText("Candidate ≠ Evidence ≠ Citation")).toBeInTheDocument();
    await user.click(within(inspector).getByRole("button", { name: "View Source Candidate" }));
    expect(inspector).toHaveAttribute("data-selected-entity", "candidate");
    expect(within(inspector).getByText("Graph recovered")).toBeInTheDocument();
  });

  it("selects Citation and follows exact Citation → Evidence → Candidate lineage", async () => {
    const user = userEvent.setup();
    renderInspector(loadS03Ask());
    const citation = document.querySelector('[data-citation-key="citation:C4"]') as HTMLElement;
    await user.click(citation);
    const inspector = screen.getByLabelText("Context inspector");
    expect(inspector).toHaveAttribute("data-selected-entity", "citation");
    expect(within(inspector).getByText("Evidence ≠ Citation")).toBeInTheDocument();
    expect(within(inspector).getByText("30b273ba-f462-452b-9fd7-ad3456fde347")).toBeInTheDocument();
    await user.click(within(inspector).getByRole("button", { name: "View Evidence" }));
    expect(inspector).toHaveAttribute("data-selected-entity", "evidence");
  });

  it("makes Answerability, Generation, Grounding and Outcome separately inspectable", async () => {
    const user = userEvent.setup();
    renderInspector(loadS03Ask());
    const inspector = screen.getByLabelText("Context inspector");
    await user.click(screen.getByRole("button", { name: /Answerability/ }));
    expect(inspector).toHaveAttribute("data-selected-entity", "answerability");
    expect(within(inspector).getByText("Runtime Answerability Gate")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Generation/ }));
    expect(inspector).toHaveAttribute("data-selected-entity", "generation");
    expect(within(inspector).getByText("deepseek-v4-flash")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Grounding/ }));
    expect(inspector).toHaveAttribute("data-selected-entity", "grounding");
    expect(within(inspector).getByText("100%")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Outcome/ }));
    expect(inspector).toHaveAttribute("data-selected-entity", "outcome");
  });

  it("preserves S04 Answerable → Generation abstained → Grounding N/A → zero Citation → safe Refusal", async () => {
    const user = userEvent.setup();
    const trace = loadTrace("s04");
    renderInspector(trace);
    expect(screen.getAllByText("safe refusal").length).toBeGreaterThan(0);
    expect(screen.getByText(/Refused ≠ failed/)).toBeInTheDocument();
    expect(trace.answerability.answerable).toBe(true);
    expect(trace.generation.generation_abstained).toBe(true);
    expect(trace.grounding.status).toBe("not_applicable");
    expect(trace.citation.citation_count).toBe(0);
    expect(trace.citation.citation_valid).toBe(true);
    expect(trace.outcome.failure_stage).toBeNull();
    await user.click(screen.getByRole("button", { name: /Generation/ }));
    expect(within(screen.getByLabelText("Context inspector")).getAllByText("Abstained").length).toBeGreaterThan(0);
  });

  it("treats Grounding not_applicable as authoritative state rather than runtime failure", async () => {
    const user = userEvent.setup();
    renderInspector(loadTrace("s04"));
    await user.click(screen.getByRole("button", { name: /Grounding/ }));
    const inspector = screen.getByLabelText("Context inspector");
    expect(within(inspector).getAllByText("not_applicable").length).toBeGreaterThan(0);
    expect(within(inspector).queryByText("failed")).not.toBeInTheDocument();
  });

  it("surfaces orphan Citation identity instead of repairing it", () => {
    const trace = structuredClone(loadS03Ask());
    trace.citation.citations[0].evidence_id = "C404";
    const model = buildEvidenceValidationModel(trace);
    expect(model.orphanCitationCount).toBe(1);
    renderInspector(trace);
    expect(screen.getByText("1 issue(s)")).toBeInTheDocument();
  });

  it("surfaces Citation/Evidence Candidate mismatch instead of normalizing it", () => {
    const trace = structuredClone(loadS03Ask());
    trace.citation.citations[3].source_candidate_id = trace.citation.citations[0].source_candidate_id;
    const model = buildEvidenceValidationModel(trace);
    expect(model.citationCandidateMismatchCount).toBe(1);
    renderInspector(trace);
    expect(screen.getByText("1 issue(s)")).toBeInTheDocument();
  });

  it("retains partial Evidence records while showing missing identity as unavailable", () => {
    const trace = structuredClone(loadS03Ask());
    trace.evidence.evidence_items[0].evidence_id = null;
    trace.evidence.evidence_items[0].source_candidate_id = null;
    const model = buildEvidenceValidationModel(trace);
    expect(model.evidence).toHaveLength(4);
    expect(model.evidenceMissingCandidateIdentityCount).toBe(1);
    renderInspector(trace);
    expect(screen.getAllByText("Unavailable").length).toBeGreaterThan(0);
  });

  it("uses presentation-only Query progressive disclosure to navigate into Evidence", async () => {
    const user = userEvent.setup();
    const trace = loadS03Ask();
    mocked.trace = trace;
    render(<ControlCenterStateProvider><NavEcho /><EvidenceValidationSummary trace={trace} /></ControlCenterStateProvider>);
    expect(screen.getByTestId("active-nav")).toHaveTextContent("overview");
    await user.click(screen.getByRole("button", { name: /Inspect Evidence & Validation/ }));
    expect(screen.getByTestId("active-nav")).toHaveTextContent("evidence");
    expect(trace.evidence.evidence_items).toHaveLength(4);
  });

  it("uses identical Runtime Trace authority in Operator and Showcase modes", async () => {
    const user = userEvent.setup();
    const trace = loadS03Ask();
    render(<ControlCenterStateProvider><ModeToggle /><EvidenceValidationInspector trace={trace} /></ControlCenterStateProvider>);
    expect(screen.getByTestId("mode")).toHaveTextContent("operator");
    expect(screen.getByText("Evidence Inventory · 4")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "toggle mode" }));
    expect(screen.getByTestId("mode")).toHaveTextContent("showcase");
    expect(screen.getByText("Evidence Inventory · 4")).toBeInTheDocument();
  });

  it("activates the modern Evidence route from the active trace", () => {
    mocked.trace = loadS03Ask();
    render(<ControlCenterStateProvider><EvidenceValidationPage /></ControlCenterStateProvider>);
    expect(screen.getByText("Evidence & Validation")).toBeInTheDocument();
    expect(screen.getByText("Answer Validation Pipeline")).toBeInTheDocument();
  });
});
