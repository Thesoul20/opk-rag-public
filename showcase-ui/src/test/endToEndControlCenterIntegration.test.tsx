import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { loadTrace } from "./fixtures";
import { buildIntegratedStages, recoveryState } from "../control-center/adapters/endToEndRuntime";
import { ControlCenterStateProvider, useControlCenter } from "../control-center/state/ControlCenterState";
import { TraceSelectionSynchronizer } from "../control-center/components/TraceSelectionSynchronizer";
import { ActiveExecutionBar } from "../control-center/components/ActiveExecutionBar";
import { GraphRuntimeInspector } from "../control-center/pages/GraphRetrievalPage";
import { EvidenceValidationInspector } from "../control-center/pages/EvidenceValidationPage";
import { Inspector } from "../control-center/layout/Inspector";

const mocked = vi.hoisted(() => ({
  trace: undefined as RuntimeTraceV1 | undefined,
  runQueryInput: vi.fn(),
}));
vi.mock("../state/ShowcaseState", () => ({ useShowcase: () => ({ state: { trace: mocked.trace, lifecycle: mocked.trace?.trace.status || "idle" }, actions: { runQueryInput: mocked.runQueryInput } }) }));

function StateEcho(){const {state,actions}=useControlCenter();return <div>
  <span data-testid="nav">{state.activeNav}</span><span data-testid="trace">{state.activeTraceId||"none"}</span><span data-testid="kind">{state.selectedInspectorEntityKind||"none"}</span>
  <button onClick={()=>actions.selectCandidate("candidate-stale")}>select candidate</button>
  <button onClick={()=>actions.selectGraphNode("node-stale")}>select graph</button>
  <button onClick={()=>actions.selectEvidence("evidence:C4")}>select evidence</button>
  <button onClick={()=>actions.selectCitation("citation:C4")}>select citation</button>
  <button onClick={()=>actions.selectValidationStage("generation")}>select generation</button>
  <button onClick={()=>actions.setMode(state.mode==="operator"?"showcase":"operator")}>toggle mode</button><span data-testid="mode">{state.mode}</span>
</div>}

function loadS03Ask():RuntimeTraceV1 {
  return JSON.parse(readFileSync(resolve(process.cwd(),"..","evaluation-data","showcase","runtime_trace_v1_live_s03_ask_task0233.json"),"utf8"));
}

function renderIntegrated(trace:RuntimeTraceV1){mocked.trace=trace;return render(<ControlCenterStateProvider><TraceSelectionSynchronizer/><ActiveExecutionBar/><StateEcho/><GraphRuntimeInspector trace={trace}/><EvidenceValidationInspector trace={trace}/><Inspector/></ControlCenterStateProvider>)}

describe("TASK-0277 End-to-End Control Center Integration",()=>{
  beforeEach(()=>{mocked.trace=undefined;mocked.runQueryInput.mockReset();});

  it("maps S01 no-recovery and S02 Structure Recovery truthfully",()=>{
    const s01=loadTrace("s01"), s02=loadTrace("s02");
    expect(recoveryState(s01).detail).toBe("No bounded recovery");
    expect(recoveryState(s01).state).toBe("skipped");
    expect(recoveryState(s02).detail).toContain("Structure");
    expect(recoveryState(s02).inspectorKind).toBe("structure_recovery");
  });

  it("maps the S03 Ask complete Graph→Evidence→Citation lifecycle from one Runtime Trace",()=>{
    const trace=loadS03Ask(); const stages=buildIntegratedStages(trace);
    expect(stages.find(s=>s.id==="optional_recovery")?.detail).toContain("Graph · hop 1");
    expect(stages.find(s=>s.id==="evidence_composition")?.detail).toBe("4 items");
    expect(stages.find(s=>s.id==="grounding")?.detail).toBe("grounded");
    expect(stages.find(s=>s.id==="citation")?.detail).toBe("4 citations");
    expect(stages.find(s=>s.id==="outcome")?.state).toBe("completed");
  });

  it("preserves S04 Answerable→Generation abstained→Grounding N/A→Safe Refusal",()=>{
    const trace=loadTrace("s04"); const stages=buildIntegratedStages(trace);
    expect(stages.find(s=>s.id==="answerability")?.detail).toBe("answerable");
    expect(stages.find(s=>s.id==="generation")?.detail).toBe("abstained");
    expect(stages.find(s=>s.id==="grounding")?.detail).toBe("not_applicable");
    expect(stages.find(s=>s.id==="citation")?.detail).toBe("0 citations");
    expect(stages.find(s=>s.id==="outcome")?.detail).toBe("Safe refusal");
    expect(trace.outcome.failure_stage).toBeNull();
  });

  it("keeps the same trace authority across global header, Graph and Evidence views",()=>{
    const trace=loadS03Ask(); renderIntegrated(trace);
    expect(screen.getByText(trace.trace.trace_id.slice(0,12)+"…")).toBeInTheDocument();
    expect(screen.getByLabelText("Runtime Graph Recovery Subgraph")).toBeInTheDocument();
    expect(screen.getByLabelText("Evidence answer validation pipeline")).toBeInTheDocument();
    expect(screen.getByTestId("trace")).toHaveTextContent(trace.trace.trace_id);
  });

  it("global drill-down navigation is presentation-only and never calls query execution",async()=>{
    const user=userEvent.setup(); const trace=loadS03Ask(); renderIntegrated(trace);
    await user.click(screen.getByRole("button",{name:"Inspect active runtime"})); expect(screen.getByTestId("nav")).toHaveTextContent("runtime");
    await user.click(screen.getByRole("button",{name:"Inspect active graph"})); expect(screen.getByTestId("nav")).toHaveTextContent("graph");
    await user.click(screen.getByRole("button",{name:"Inspect active evidence"})); expect(screen.getByTestId("nav")).toHaveTextContent("evidence");
    expect(mocked.runQueryInput).not.toHaveBeenCalled();
  });

  it.each([
    ["candidate","select candidate"],["graph_node","select graph"],["evidence","select evidence"],["citation","select citation"],["generation","select generation"],
  ])("clears stale %s selection when active Trace ID changes",async(kind,label)=>{
    const user=userEvent.setup(); mocked.trace=loadS03Ask(); const view=render(<ControlCenterStateProvider><TraceSelectionSynchronizer/><StateEcho/></ControlCenterStateProvider>);
    await user.click(screen.getByRole("button",{name:label})); expect(screen.getByTestId("kind")).toHaveTextContent(kind);
    mocked.trace=loadTrace("s04"); view.rerender(<ControlCenterStateProvider><TraceSelectionSynchronizer/><StateEcho/></ControlCenterStateProvider>);
    expect(await screen.findByTestId("kind")).toHaveTextContent("none"); expect(screen.getByTestId("trace")).toHaveTextContent(loadTrace("s04").trace.trace_id);
  });

  it("navigates exact S03 recovered Candidate to Evidence and back through authoritative IDs",async()=>{
    const user=userEvent.setup(); const trace=loadS03Ask(); renderIntegrated(trace);
    const graphPanel=screen.getByText("Recovered Candidate Paths · 1").closest("section")!;
    await user.click(within(graphPanel).getByRole("button",{name:"View Evidence"}));
    expect(screen.getByTestId("nav")).toHaveTextContent("evidence");
    const inspector=screen.getByLabelText("Context inspector"); expect(inspector).toHaveAttribute("data-selected-entity","evidence"); expect(within(inspector).getAllByText("C4").length).toBeGreaterThan(0);
    await user.click(within(inspector).getByRole("button",{name:"View Source Candidate"}));
    expect(inspector).toHaveAttribute("data-selected-entity","candidate"); expect(within(inspector).getByText("Graph recovered")).toBeInTheDocument();
    await user.click(within(inspector).getByRole("button",{name:"View Evidence"})); expect(inspector).toHaveAttribute("data-selected-entity","evidence");
  });

  it("uses identical active trace when switching Operator and Showcase modes",async()=>{
    const user=userEvent.setup(); const trace=loadS03Ask(); renderIntegrated(trace); const id=trace.trace.trace_id;
    expect(screen.getByTestId("mode")).toHaveTextContent("operator"); expect(screen.getByTestId("trace")).toHaveTextContent(id);
    await user.click(screen.getByRole("button",{name:"toggle mode"})); expect(screen.getByTestId("mode")).toHaveTextContent("showcase"); expect(screen.getByTestId("trace")).toHaveTextContent(id);
  });
});
