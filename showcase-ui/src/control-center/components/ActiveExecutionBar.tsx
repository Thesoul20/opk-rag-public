import { Activity, FileCheck2, GitBranch, Workflow } from "lucide-react";
import { useShowcase } from "../../state/ShowcaseState";
import { activeExecutionSummary } from "../adapters/endToEndRuntime";
import { Badge, Button, StatusBadge } from "./Primitives";
import { useControlCenter } from "../state/ControlCenterState";

export function ActiveExecutionBar() {
  const { state: showcase } = useShowcase();
  const { actions } = useControlCenter();
  const trace=showcase.trace;
  if (!trace) return <div className="cc-active-execution" data-state="empty"><Activity size={14}/><span>No active Runtime Trace · run Search/Ask or reopen a Conversation turn.</span></div>;
  const summary=activeExecutionSummary(trace);
  return <div className="cc-active-execution" data-state={summary.status} data-trace-id={summary.traceId}>
    <div className="cc-active-execution-main"><Badge tone="info">{summary.scope}</Badge><StatusBadge status={summary.safeRefusal ? "refused" : summary.status === "failed" ? "failed" : summary.status === "partial" ? "partial" : "completed"} label={summary.safeRefusal ? "safe refusal" : summary.status}/><code>{summary.traceId.slice(0,12)}…</code><span title={summary.query}>{summary.query}</span></div>
    <div className="cc-active-execution-facts"><span>{summary.graph}</span><span>{summary.evidence}</span><span>{summary.grounding}</span><span>{summary.citations}</span></div>
    <div className="cc-inline-actions"><Button aria-label="Inspect active runtime" onClick={()=>actions.setActiveNav("runtime")}><Workflow size={13}/> Runtime</Button>{trace.graph_recovery.graph_activated && <Button aria-label="Inspect active graph" onClick={()=>actions.setActiveNav("graph")}><GitBranch size={13}/> Graph</Button>}<Button aria-label="Inspect active evidence" onClick={()=>actions.setActiveNav("evidence")}><FileCheck2 size={13}/> Evidence</Button></div>
  </div>;
}
