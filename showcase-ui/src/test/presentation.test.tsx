import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ExecutiveView } from "../components/ExecutiveView";
import { PresentationToolbar } from "../components/PresentationToolbar";
import { PresentationProvider, usePresentation } from "../i18n";
import { buildExecutiveNarrative } from "../presentation/narrative";
import { loadTrace } from "./fixtures";

function Probe() {
  const { locale, viewMode } = usePresentation();
  return <output aria-label="presentation-state">{locale}:{viewMode}</output>;
}


describe("TASK-0229 presentation layer", () => {
  it("switches language and view without touching runtime trace data", async () => {
    const user = userEvent.setup();
    const traceId = loadTrace("s03").trace.trace_id;
    render(<PresentationProvider initialLocale="en" initialViewMode="engineer"><PresentationToolbar /><Probe /><span data-testid="runtime-trace-id">{traceId}</span></PresentationProvider>);
    expect(screen.getByLabelText("presentation-state")).toHaveTextContent("en:engineer");
    await user.click(screen.getByRole("button", { name: "中文" }));
    await user.click(screen.getByRole("button", { name: "领导展示" }));
    expect(screen.getByLabelText("presentation-state")).toHaveTextContent("zh-CN:executive");
    expect(screen.getByTestId("runtime-trace-id")).toHaveTextContent(traceId);
  });

  it("renders Chinese S03 executive Graph narrative from authoritative fields", () => {
    render(<PresentationProvider initialLocale="zh-CN"><ExecutiveView trace={loadTrace("s03")} /></PresentationProvider>);
    expect(screen.getByText("本次运行摘要")).toBeInTheDocument();
    expect(screen.getByText(/系统启用 Graph Recovery/)).toBeInTheDocument();
    expect(screen.getByText(/图谱只扩展 1 跳/)).toBeInTheDocument();
    expect(screen.getAllByText("1 跳").length).toBeGreaterThan(0);
  });

  it("renders S04 as safe refusal rather than runtime failure", () => {
    render(<PresentationProvider initialLocale="zh-CN"><ExecutiveView trace={loadTrace("s04")} /></PresentationProvider>);
    expect(screen.getAllByText("安全拒答").length).toBeGreaterThan(0);
    expect(screen.getByText(/主动的安全结果，不是运行失败/)).toBeInTheDocument();
  });

  it("maps S01 normal, S02 structure and S03 graph recovery without synthetic metrics", () => {
    const t = (key: Parameters<typeof buildExecutiveNarrative>[1] extends infer T ? never : never) => key;
    void t;
    const dictionaryT = ((key: string, vars?: Record<string, string | number>) => {
      const values: Record<string, string> = {
        "exec.noRecovery": "none", "common.no": "no", "exec.normal": "normal", "exec.structureRecovery": "structure", "exec.structure": "structure narrative", "exec.graphRecoveryLabel": "graph", "exec.graphRecovery": "graph narrative", "exec.oneHop": `hop ${vars?.hop}`, "exec.refused": "refused", "exec.failed": "failed", "exec.completed": "completed", "exec.oneHopLabel": `graph ${vars?.hop}`,
      };
      return values[key] || key;
    }) as Parameters<typeof buildExecutiveNarrative>[1];
    expect(buildExecutiveNarrative(loadTrace("s01"), dictionaryT).recoveryLabel).toBe("none");
    expect(buildExecutiveNarrative(loadTrace("s02"), dictionaryT).recoveryLabel).toBe("structure");
    const s03 = buildExecutiveNarrative(loadTrace("s03"), dictionaryT);
    expect(s03.recoveryLabel).toBe("graph");
    expect(s03.candidateBefore).toBe(loadTrace("s03").rerank.input_candidate_count);
    expect(s03.candidateAfter).toBe(loadTrace("s03").rerank.output_candidate_count);
  });
});
