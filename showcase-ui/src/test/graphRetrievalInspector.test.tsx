import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { buildRuntimeGraphSubgraph } from "../control-center/adapters/graphRuntime";
import { GraphRetrievalPage, GraphRuntimeInspector } from "../control-center/pages/GraphRetrievalPage";
import { GuardRecoveryControlPlane } from "../control-center/components/GuardRecoveryControlPlane";
import { Inspector } from "../control-center/layout/Inspector";
import { ControlCenterStateProvider, useControlCenter } from "../control-center/state/ControlCenterState";
import { loadTrace } from "./fixtures";

const mocked = vi.hoisted(() => ({ trace: undefined as RuntimeTraceV1 | undefined }));
vi.mock("../state/ShowcaseState", () => ({ useShowcase: () => ({ state: { trace: mocked.trace }, actions: {} }) }));

function renderGraph(trace: RuntimeTraceV1) {
  mocked.trace = trace;
  return render(<ControlCenterStateProvider><GraphRuntimeInspector trace={trace} /><Inspector /></ControlCenterStateProvider>);
}

function ModeToggle() {
  const { state, actions } = useControlCenter();
  return <><span data-testid="mode">{state.mode}</span><button onClick={() => actions.setMode(state.mode === "operator" ? "showcase" : "operator")}>toggle mode</button></>;
}

function NavEcho() {
  const { state } = useControlCenter();
  return <span data-testid="active-nav">{state.activeNav}</span>;
}

describe("TASK-0275 Graph Retrieval Interactive Inspector", () => {
  beforeEach(() => { mocked.trace = undefined; });

  it("renders S03 authoritative one-hop runtime subgraph and execution summary", () => {
    const trace = loadTrace("s03");
    renderGraph(trace);
    expect(screen.getByText("Graph Execution Summary")).toBeInTheDocument();
    expect(screen.getByText("1 / max 1")).toBeInTheDocument();
    expect(screen.getByText("3 → 4")).toBeInTheDocument();
    expect(screen.getByText("Runtime Graph Recovery Subgraph")).toBeInTheDocument();
    expect(screen.getByLabelText("Runtime Graph Recovery Subgraph")).toHaveAttribute("data-layout", "deterministic-one-hop");
    expect(screen.getAllByText("LINKS_TO").length).toBeGreaterThanOrEqual(3);
  });

  it("preserves duplicate-looking authoritative edge identities instead of merging them", () => {
    const trace = loadTrace("s03");
    const model = buildRuntimeGraphSubgraph(trace);
    expect(model.edges).toHaveLength(3);
    expect(model.edges.map(edge => edge.edgeId)).toEqual([
      "link:b8d74db1bf694f3f",
      "link:cdac97bda4e1cb59",
      "link:16bb2b6937f90a48",
    ]);
    renderGraph(trace);
    expect(screen.getByText("link:b8d74db1bf694f3f")).toBeInTheDocument();
    expect(screen.getByText("link:cdac97bda4e1cb59")).toBeInTheDocument();
  });

  it("selects an authoritative graph edge and populates the Contextual Inspector", async () => {
    const user = userEvent.setup();
    const trace = loadTrace("s03");
    renderGraph(trace);
    await user.click(screen.getByTitle("link:b8d74db1bf694f3f"));
    const inspector = screen.getByLabelText("Context inspector");
    expect(inspector).toHaveAttribute("data-selected-entity", "graph_edge");
    expect(within(inspector).getByText("link:b8d74db1bf694f3f")).toBeInTheDocument();
    expect(within(inspector).getByText("Tauri/Tauri 学习路线.md")).toBeInTheDocument();
    expect(within(inspector).getByText("Tauri/公众号发布 SaaS Demo/MVP 功能拆解.md")).toBeInTheDocument();
    expect(within(inspector).getAllByText("LINKS_TO").length).toBeGreaterThanOrEqual(1);
  });

  it("selects a graph node without relabeling it as Candidate", async () => {
    const user = userEvent.setup();
    const trace = loadTrace("s03");
    renderGraph(trace);
    await user.click(screen.getAllByTitle("Tauri/Tauri 学习路线.md")[0]);
    const inspector = screen.getByLabelText("Context inspector");
    expect(inspector).toHaveAttribute("data-selected-entity", "graph_node");
    expect(within(inspector).getByText("Graph Node ≠ Candidate")).toBeInTheDocument();
    expect(within(inspector).getByText("seed")).toBeInTheDocument();
  });

  it("inspects recovered Candidate graph provenance then links back to TASK-0273 Candidate Inspector", async () => {
    const user = userEvent.setup();
    const trace = loadTrace("s03");
    renderGraph(trace);
    await user.click(screen.getByRole("button", { name: "Inspect path" }));
    const inspector = screen.getByLabelText("Context inspector");
    expect(inspector).toHaveAttribute("data-selected-entity", "graph_path");
    expect(within(inspector).getByText("Graph Node ≠ Candidate ≠ Evidence")).toBeInTheDocument();
    expect(within(inspector).getByText("G1")).toBeInTheDocument();
    expect(within(inspector).getByText("authoritative_edge")).toBeInTheDocument();
    await user.click(within(inspector).getByRole("button", { name: "View Candidate" }));
    expect(inspector).toHaveAttribute("data-selected-entity", "candidate");
    expect(within(inspector).getByText("Graph recovered")).toBeInTheDocument();
  });

  it("renders no fabricated topology when Graph Recovery is skipped", () => {
    const trace = loadTrace("s02");
    renderGraph(trace);
    expect(screen.getByText("Graph Recovery skipped")).toBeInTheDocument();
    expect(screen.queryByLabelText("Runtime Graph Recovery Subgraph")).not.toBeInTheDocument();
    expect(screen.queryByText("LINKS_TO")).not.toBeInTheDocument();
    expect(buildRuntimeGraphSubgraph(trace).nodes).toHaveLength(0);
    expect(buildRuntimeGraphSubgraph(trace).edges).toHaveLength(0);
  });

  it("retains a partial authoritative edge while refusing to fabricate a missing endpoint node", () => {
    const trace = structuredClone(loadTrace("s03"));
    trace.graph_recovery.traversed_edges![0].target_node_id = null;
    const model = buildRuntimeGraphSubgraph(trace);
    expect(model.edges).toHaveLength(3);
    expect(model.edges[0].edgeId).toBe("link:b8d74db1bf694f3f");
    expect(model.edges[0].targetNodeId).toBeNull();
    renderGraph(trace);
    expect(screen.getByText("Target node identity unavailable")).toBeInTheDocument();
  });

  it("keeps runtime Gold false and preserves one-hop governance from S03", () => {
    const model = buildRuntimeGraphSubgraph(loadTrace("s03"));
    expect(model.runtimeGoldMetadataUsage).toBe(false);
    expect(model.hopDepth).toBe(1);
    expect(model.recoveredCandidateCount).toBe(1);
    expect(model.nodes.some(node => node.role === "recovered")).toBe(true);
  });

  it("uses the same Runtime Trace authority in Operator and Showcase modes", async () => {
    const user = userEvent.setup();
    const trace = loadTrace("s03");
    mocked.trace = trace;
    render(<ControlCenterStateProvider><ModeToggle /><GraphRuntimeInspector trace={trace} /></ControlCenterStateProvider>);
    const canvas = screen.getByLabelText("Runtime Graph Recovery Subgraph");
    expect(screen.getByTestId("mode")).toHaveTextContent("operator");
    expect(within(canvas).getAllByText("LINKS_TO")).toHaveLength(3);
    await user.click(screen.getByRole("button", { name: "toggle mode" }));
    expect(screen.getByTestId("mode")).toHaveTextContent("showcase");
    expect(within(screen.getByLabelText("Runtime Graph Recovery Subgraph")).getAllByText("LINKS_TO")).toHaveLength(3);
  });

  it("uses TASK-0274 Inspect Graph Path as presentation-only progressive disclosure", async () => {
    const user = userEvent.setup();
    const trace = loadTrace("s03");
    mocked.trace = trace;
    render(<ControlCenterStateProvider><NavEcho /><GuardRecoveryControlPlane trace={trace} /></ControlCenterStateProvider>);
    expect(screen.getByTestId("active-nav")).toHaveTextContent("overview");
    await user.click(screen.getByRole("button", { name: "Inspect Graph Path" }));
    expect(screen.getByTestId("active-nav")).toHaveTextContent("graph");
    expect(trace.graph_recovery.traversed_edges).toHaveLength(3);
  });

  it("activates the modern Graph route surface from the active trace without executing new traversal", () => {
    mocked.trace = loadTrace("s03");
    render(<ControlCenterStateProvider><GraphRetrievalPage /></ControlCenterStateProvider>);
    expect(screen.getByText("Graph Retrieval")).toBeInTheDocument();
    expect(screen.getByText(/active Runtime Trace/)).toBeInTheDocument();
    expect(screen.getByText("Runtime Graph Recovery Subgraph")).toBeInTheDocument();
  });
});
