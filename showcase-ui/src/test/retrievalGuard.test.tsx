import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RetrievalGuardPipeline } from "../components/RetrievalGuardPipeline";
import { PresentationProvider } from "../i18n";
import { emptyReplayState } from "../lib/sse";
import { buildRetrievalGuardModel } from "../presentation/retrievalGuard";
import { loadTrace } from "./fixtures";

const t = (key: any, vars?: Record<string,string|number>) => { let s=String(key); for (const [k,v] of Object.entries(vars || {})) s=s.replaceAll(`{${k}}`,String(v)); return s; };

describe("TASK-0230 retrieval and guard visualization", () => {
  it("maps S01 bounded continue", () => { const m=buildRetrievalGuardModel(loadTrace("s01"), t as any)!; expect(m.selectedAction).toBe("continue"); expect([m.recoveryAttempts,m.recoveryMaximum]).toEqual([0,1]); });
  it("maps S02 structure counts from trace", () => { const tr=loadTrace("s02"); const m=buildRetrievalGuardModel(tr,t as any)!; expect(m.selectedAction).toBe("structure"); expect(m.structureBefore).toBe(tr.structure_recovery.candidate_pool_count_before); expect(m.structureAfter).toBe(tr.structure_recovery.candidate_pool_count_after); });
  it("maps S03 one-hop graph recovery", () => { const m=buildRetrievalGuardModel(loadTrace("s03"),t as any)!; expect(m.selectedAction).toBe("graph"); expect(m.graphHop).toBe(1); expect(m.graphRecovered).toBeGreaterThan(0); expect([m.recoveryAttempts,m.recoveryMaximum]).toEqual([1,1]); });
  it("maps S04 fail closed not failed", () => { const tr=loadTrace("s04"); const m=buildRetrievalGuardModel(tr,t as any)!; expect(m.selectedAction).toBe("fail_closed"); expect(tr.trace.status).toBe("refused"); expect(tr.trace.status).not.toBe("failed"); });
  it("renders pipeline budget and rerank stage", () => { render(<PresentationProvider initialLocale="en" initialViewMode="engineer"><RetrievalGuardPipeline trace={loadTrace("s03")} replay={emptyReplayState()} /></PresentationProvider>); expect(screen.getByTestId("retrieval-guard-pipeline")).toBeInTheDocument(); expect(screen.getByText("Recovery budget")).toBeInTheDocument(); expect(screen.getByText("1 / 1")).toBeInTheDocument(); expect(screen.getByText("BGE Reranking")).toBeInTheDocument(); });
});
