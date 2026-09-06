import { ShieldCheck } from "lucide-react";
import type { RuntimeTraceV1 } from "../../types/runtimeTrace";
import { buildEvidenceValidationModel } from "../adapters/evidenceRuntime";
import { Button, Metric, Panel, StatusBadge } from "./Primitives";
import { useControlCenter } from "../state/ControlCenterState";

function value(v: unknown): string | number { return v == null ? "Unavailable" : typeof v === "number" ? v : String(v); }

export function EvidenceValidationSummary({ trace, compact = true }: { trace: RuntimeTraceV1; compact?: boolean }) {
  const { actions } = useControlCenter();
  const model = buildEvidenceValidationModel(trace);
  const safeRefusal = (trace.outcome.status || trace.trace.status) === "refused" && !trace.outcome.failure_stage;
  return <Panel title="Evidence & Validation" action={<Button onClick={() => actions.setActiveNav("evidence")}><ShieldCheck size={14} /> Inspect Evidence & Validation</Button>}>
    <div className="cc-metric-grid cc-validation-summary-grid">
      <Metric label="Evidence" value={value(model.evidenceCount)} />
      <Metric label="Answerability" value={model.searchOnly ? "N/A" : value(trace.answerability.answerability_state)} />
      <Metric label="Generation" value={model.searchOnly ? "N/A" : trace.generation.generation_abstained ? "abstained" : trace.generation.generation_completed ? "completed" : value(trace.generation.stage_state)} />
      <Metric label="Grounding" value={model.searchOnly ? "N/A" : value(trace.grounding.status)} />
      <Metric label="Citations" value={model.searchOnly ? "N/A" : value(trace.citation.citation_count)} />
      <Metric label="Outcome" value={safeRefusal ? <StatusBadge status="refused" label="safe refusal" /> : value(trace.outcome.status || trace.trace.status)} />
    </div>
    {model.searchOnly && <div className="cc-inspector-boundary-note">Search scope ends at Evidence. Answerability, Generation, Grounding and Citation are not fabricated.</div>}
    {!compact && safeRefusal && <div className="cc-inspector-boundary-note">Refused ≠ failed: the governed pipeline safely declined release with no runtime failure stage.</div>}
  </Panel>;
}
