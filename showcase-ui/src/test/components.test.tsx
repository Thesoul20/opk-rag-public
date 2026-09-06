import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CandidateEvidencePanel } from "../components/CandidateEvidencePanel";
import { GuardDecisionPanel } from "../components/GuardDecisionPanel";
import { KnowledgeGraphPanel } from "../components/KnowledgeGraphPanel";
import { RuntimeMetricsPanel } from "../components/RuntimeMetricsPanel";
import { TraceHeader } from "../components/TraceHeader";
import { TraceTimeline } from "../components/TraceTimeline";
import { PresentationProvider } from "../i18n";
import { emptyReplayState } from "../lib/sse";
import { loadTrace } from "./fixtures";

function renderLocalized(node: React.ReactNode) {
  return render(<PresentationProvider initialLocale="en">{node}</PresentationProvider>);
}

describe("major component shell", () => {
  it("renders S03 graph activation and hop depth", () => {
    renderLocalized(<KnowledgeGraphPanel trace={loadTrace("s03")} />);
    expect(screen.getByText("Interactive Graph Retrieval")).toBeInTheDocument();
    expect(screen.getByTestId("interactive-graph-recovery")).toBeInTheDocument();
    expect(screen.getAllByText("1-hop").length).toBeGreaterThan(0);
  });

  it("renders S04 refusal as governed status, not failed", () => {
    renderLocalized(<GuardDecisionPanel trace={loadTrace("s04")} />);
    expect(screen.getByText("governed refusal")).toBeInTheDocument();
    expect(screen.queryByText("failed")).not.toBeInTheDocument();
  });

  it("renders Candidate and Evidence sections independently", () => {
    renderLocalized(<CandidateEvidencePanel trace={loadTrace("s03")} />);
    expect(screen.getByText("Candidates")).toBeInTheDocument();
    expect(screen.getByText("Evidence")).toBeInTheDocument();
  });

  it("renders timing, timeline, and trace header shells", () => {
    const trace = loadTrace("s01");
    renderLocalized(<><TraceHeader trace={trace} /><TraceTimeline trace={trace} replay={emptyReplayState()} /><RuntimeMetricsPanel trace={trace} /></>);
    expect(screen.getByText("Trace Header")).toBeInTheDocument();
    expect(screen.getByText("Runtime Timeline")).toBeInTheDocument();
    expect(screen.getByText("Runtime Metrics")).toBeInTheDocument();
    expect(screen.getAllByText("--").length).toBeGreaterThan(0);
  });
});
