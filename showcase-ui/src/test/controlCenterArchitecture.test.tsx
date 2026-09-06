import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ControlCenterShell } from "../control-center/ControlCenterShell";
import { CONTROL_CENTER_LIMITS, CONTROL_CENTER_PIPELINE, FROZEN_AGENT_ACTIONS } from "../control-center/architecture";

describe("TASK-0269 Control Center architecture freeze", () => {
  it("preserves bounded Agent and Graph authority", () => {
    expect(CONTROL_CENTER_LIMITS.graphMaxHop).toBe(1);
    expect(CONTROL_CENTER_LIMITS.recoveryMaxAttempts).toBe(1);
    expect(CONTROL_CENTER_LIMITS.unrestrictedProductionAgent).toBe(false);
    expect(FROZEN_AGENT_ACTIONS).toEqual(["hybrid_search","structure_search","graph_search","rewrite_query","inspect_evidence","finish","abstain"]);
  });
  it("freezes the trace-first pipeline and no UI decision authority", () => {
    render(<ControlCenterShell />);
    expect(screen.getByText("OPK-RAG Control Center")).toBeInTheDocument();
    expect(CONTROL_CENTER_PIPELINE).toContain("evidence_composition");
    expect(CONTROL_CENTER_PIPELINE.at(-1)).toBe("outcome");
    expect(document.querySelector('[data-ui-decision-authority="false"]')).toBeTruthy();
  });
});
