import { act, fireEvent, render, screen } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { AnswerValidationPanel } from "../components/AnswerValidationPanel";
import { CandidateRerankEvidencePanel } from "../components/CandidateRerankEvidencePanel";
import { PresentationProvider } from "../i18n";
import { buildAnswerValidationModel } from "../presentation/answerValidation";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { loadTrace } from "./fixtures";

function loadS03Ask(): RuntimeTraceV1 {
  return JSON.parse(readFileSync(resolve(process.cwd(), "..", "evaluation-data", "showcase", "runtime_trace_v1_live_s03_ask_task0233.json"), "utf-8")) as RuntimeTraceV1;
}
const t = (key: string, variables?: Record<string,string|number>) => Object.entries(variables||{}).reduce((v,[k,r])=>v.replaceAll(`{${k}}`,String(r)),key) as never;

describe("TASK-0233 Answer Validation Visualization", () => {
  it("keeps frozen S01-S03 Search answer stages truthfully not applicable", () => {
    for (const sid of ["s01","s02","s03"] as const) {
      const trace=loadTrace(sid);
      const model=buildAnswerValidationModel(trace,t)!;
      expect(model.searchOnly).toBe(true);
      expect(model.stages.filter((s)=>["answerability","generation","grounding","citation"].includes(s.id)).every((s)=>s.state==="not_applicable")).toBe(true);
    }
  });

  it("links the real S03 Ask Graph Evidence C4 through valid Citation C4", () => {
    const trace=loadS03Ask();
    const model=buildAnswerValidationModel(trace,t)!;
    expect(model.searchOnly).toBe(false);
    expect(model.answerability.answerable).toBe(true);
    expect(model.generation.completed).toBe(true);
    expect(model.generation.abstained).toBe(false);
    expect(model.grounding.passed).toBe(true);
    expect(model.citation.count).toBe(4);
    const c4=model.citation.links.find((c)=>c.evidenceId==="C4");
    expect(c4?.sourceCandidateId).toBe("30b273ba-f462-452b-9fd7-ad3456fde347");
    expect(c4?.graphRecovered).toBe(true);
    expect(model.orphanCitationEvidenceCount).toBe(0);
    expect(model.citationCandidateIdentityMismatchCount).toBe(0);
  });

  it("renders a Search scope boundary instead of fabricating Generation", () => {
    render(<PresentationProvider initialLocale="en" initialViewMode="engineer"><AnswerValidationPanel trace={loadTrace("s03")}/></PresentationProvider>);
    expect(screen.getByText("Search scope ends at Evidence")).toBeInTheDocument();
    expect(screen.getAllByText("not applicable").length).toBeGreaterThanOrEqual(4);
    expect(screen.queryByText("deepseek-v4-flash")).not.toBeInTheDocument();
  });

  it("renders S03 Ask provider, grounding, structured citations, and graph citation identity", () => {
    render(<PresentationProvider initialLocale="en" initialViewMode="engineer"><AnswerValidationPanel trace={loadS03Ask()}/></PresentationProvider>);
    expect(screen.getByText("deepseek-v4-flash")).toBeInTheDocument();
    expect(screen.getAllByText("100%").length).toBeGreaterThan(0);
    expect(screen.getByRole("button",{name:/\[C4\]/})).toBeInTheDocument();
    expect(screen.getByText("Graph")).toBeInTheDocument();
    expect(screen.getByText(/Runtime Trace V1 does not expose the final answer body/)).toBeInTheDocument();
  });

  it("uses authoritative Evidence IDs for Citation → Evidence → Candidate highlighting", () => {
    const trace=loadS03Ask();
    render(<PresentationProvider initialLocale="en" initialViewMode="engineer"><><CandidateRerankEvidencePanel trace={trace}/><AnswerValidationPanel trace={trace}/></></PresentationProvider>);
    const citation=screen.getByRole("button",{name:/\[C4\]/});
    fireEvent.click(citation);
    const candidate=document.querySelector('[data-candidate-id="30b273ba-f462-452b-9fd7-ad3456fde347"]');
    expect(candidate).toHaveClass("selected");
  });

  it("preserves S04 answerable → Generation abstained → safe refusal without failure", () => {
    const trace=loadTrace("s04");
    const model=buildAnswerValidationModel(trace,t)!;
    expect(model.evidenceCount).toBe(4);
    expect(model.answerability.answerable).toBe(true);
    expect(model.generation.abstained).toBe(true);
    expect(model.grounding.status).toBe("not_applicable");
    expect(model.citation.count).toBe(0);
    expect(model.outcome.status).toBe("refused");
    expect(model.outcome.failureStage).toBeNull();
    expect(model.outcome.refusalReasonCode).toBe("answerable_generation_abstained");
  });

  it("renders Chinese Executive safe refusal as refusal, not failed", () => {
    render(<PresentationProvider initialLocale="zh-CN" initialViewMode="executive"><AnswerValidationPanel trace={loadTrace("s04")} compact/></PresentationProvider>);
    expect(screen.getAllByText("安全拒答").length).toBeGreaterThan(0);
    expect(screen.getByText(/Generation 主动放弃生成/)).toBeInTheDocument();
    expect(screen.queryByText("执行失败")).not.toBeInTheDocument();
    expect(screen.getByText(/当前 Trace 没有释放任何 Citation/)).toBeInTheDocument();
  });

  it("reacts to Evidence reverse-selection using authoritative Evidence ID", async () => {
    const trace=loadS03Ask();
    render(<PresentationProvider initialLocale="en" initialViewMode="engineer"><AnswerValidationPanel trace={trace}/></PresentationProvider>);
    await act(async()=>window.dispatchEvent(new CustomEvent("opk-showcase-evidence-selected",{detail:{evidenceId:"C4",candidateId:"30b273ba-f462-452b-9fd7-ad3456fde347"}})));
    expect(screen.getByRole("button",{name:/\[C4\]/})).toHaveClass("selected");
  });
});
