import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { GuardRecoveryControlPlane, guardReasonLabel, recoveryDelta } from "../control-center/components/GuardRecoveryControlPlane";
import { Inspector } from "../control-center/layout/Inspector";
import { ControlCenterStateProvider } from "../control-center/state/ControlCenterState";
import { loadTrace } from "./fixtures";

const mocked = vi.hoisted(() => ({ trace: undefined as RuntimeTraceV1 | undefined }));
vi.mock("../state/ShowcaseState", () => ({ useShowcase: () => ({ state: { trace: mocked.trace }, actions: {} }) }));

function renderControl(trace: RuntimeTraceV1) {
  mocked.trace = trace;
  return render(<ControlCenterStateProvider><GuardRecoveryControlPlane trace={trace} /><Inspector /></ControlCenterStateProvider>);
}

describe("TASK-0274 Guarded Agent & Recovery Trace Visualization", () => {
  beforeEach(() => { mocked.trace = undefined; });

  it("renders the no-recovery path with a bounded 0 / 1 budget", () => {
    const trace = loadTrace("s01");
    renderControl(trace);
    expect(screen.getByText("No Recovery")).toBeInTheDocument();
    expect(screen.getAllByText("0 / 1").length).toBeGreaterThan(0);
    expect(screen.getByText("Continue without recovery")).toBeInTheDocument();
    expect(screen.getAllByText("Body confidence preserved").length).toBeGreaterThan(0);
    expect(screen.getByText("body_confidence_preserved")).toBeInTheDocument();
    expect(trace.guard.hidden_chain_of_thought_exposed).toBe(false);
  });

  it("renders Structure Recovery from authoritative before/after counts without equating delta to expanded count", () => {
    const trace = loadTrace("s02");
    renderControl(trace);
    const panel = screen.getByText("Structure Recovery", { selector: ".cc-panel-header > strong" }).closest(".cc-panel") as HTMLElement;
    expect(within(panel).getByText("3 → 7")).toBeInTheDocument();
    expect(within(panel).getByText("+4 candidates")).toBeInTheDocument();
    expect(within(panel).getByText("2")).toBeInTheDocument();
    expect(within(panel).getByText("Structural signal detected")).toBeInTheDocument();
    expect(within(panel).getByText("structural_query_phrase_or_entity_section_signal")).toBeInTheDocument();
    expect(recoveryDelta(3, 7)).toBe(4);
  });

  it("renders one-hop Graph Recovery and recovered candidate linkage", () => {
    const trace = loadTrace("s03");
    renderControl(trace);
    const panel = screen.getByText("Graph Recovery", { selector: ".cc-panel-header > strong" }).closest(".cc-panel") as HTMLElement;
    expect(within(panel).getByText("3 → 4")).toBeInTheDocument();
    expect(within(panel).getByText("+1 candidates")).toBeInTheDocument();
    expect(within(panel).getByText("Graph relation available")).toBeInTheDocument();
    expect(within(panel).getByText("one-hop executed")).toBeInTheDocument();
    const hopRow = within(panel).getByText("Hop depth").closest(".cc-key-value") as HTMLElement;
    expect(within(hopRow).getByText("1")).toBeInTheDocument();
    expect(trace.graph_recovery.hop_depth).toBe(1);
    expect(trace.graph_recovery.recovered_candidate_ids).toHaveLength(1);
  });

  it("keeps Guard decision separate from later refused/fail-closed runtime outcome", () => {
    const trace = loadTrace("s04");
    renderControl(trace);
    const boundary = screen.getByText("Decision Boundary").closest(".cc-panel") as HTMLElement;
    expect(within(boundary).getByText("continue_to_reranking")).toBeInTheDocument();
    expect(within(boundary).getByText("refused")).toBeInTheDocument();
    expect(within(boundary).getByText(/Fail closed · answerable_generation_abstained/)).toBeInTheDocument();
    expect(trace.guard.final_decision).toBe("continue_to_reranking");
    expect(trace.outcome.status).toBe("refused");
  });

  it("selects Guard and Recovery entities into the Contextual Inspector without exposing hidden reasoning", async () => {
    const user = userEvent.setup();
    const trace = loadTrace("s03");
    renderControl(trace);
    await user.click(screen.getByText("Guard Observation").closest("button")!);
    const inspector = screen.getByLabelText("Context inspector");
    expect(inspector).toHaveAttribute("data-selected-entity", "guard");
    expect(within(inspector).getByText("guarded_agent")).toBeInTheDocument();
    expect(within(inspector).getByText("not exposed")).toBeInTheDocument();
    expect(within(inspector).queryByText(/chain of thought/i)).not.toBeInTheDocument();

    await user.click(screen.getByText("Graph Recovery", { selector: ".cc-panel-header strong" }).parentElement!.querySelector("button")!);
    expect(inspector).toHaveAttribute("data-selected-entity", "graph_recovery");
    expect(within(inspector).getByText("Bounded one-hop graph candidate recovery")).toBeInTheDocument();
    expect(within(inspector).getByText(/Open Graph Retrieval for exact query-scoped node/)).toBeInTheDocument();
  });

  it("uses deterministic reason labels and preserves unknown codes as bounded telemetry", () => {
    expect(guardReasonLabel("body_confidence_preserved")).toBe("Body confidence preserved");
    expect(guardReasonLabel("query_signal_with_seed_graph_availability")).toBe("Graph relation available");
    expect(guardReasonLabel("unknown_future_reason")).toBe("Bounded runtime reason");
    expect(recoveryDelta(null, 4)).toBeNull();
  });
});
