import { ArrowRight, FileCheck2 } from "lucide-react";
import { useShowcase } from "../../state/ShowcaseState";
import type { RuntimeTraceV1 } from "../../types/runtimeTrace";
import { buildEvidenceValidationModel, evidenceProvenanceLabels, type RuntimeCitationRecord, type RuntimeEvidenceRecord, type ValidationStageId } from "../adapters/evidenceRuntime";
import { Badge, EmptyState, KeyValue, Metric, Panel, StatusBadge } from "../components/Primitives";
import { useControlCenter } from "../state/ControlCenterState";

function value(input: unknown): string | number { return input == null ? "Unavailable" : typeof input === "number" ? input : String(input); }
function lines(start?: number | null, end?: number | null): string { return start == null ? "Unavailable" : end == null || end === start ? `L${start}` : `L${start}–L${end}`; }
function stageLabel(id: ValidationStageId): string { return id === "evidence_stage" ? "Evidence" : id[0].toUpperCase() + id.slice(1); }
function stageTone(state: string): "completed" | "partial" | "failed" | "refused" | "unavailable" {
  if (state === "failed") return "failed";
  if (state === "refused") return "refused";
  if (state === "partial") return "partial";
  if (["not_applicable","skipped","unavailable","not_started"].includes(state)) return "unavailable";
  return "completed";
}

function ValidationFlow({ trace }: { trace: RuntimeTraceV1 }) {
  const { actions } = useControlCenter();
  const model = buildEvidenceValidationModel(trace);
  return <Panel title="Answer Validation Pipeline" action={<Badge tone="info">Runtime Trace V1 only</Badge>}>
    <div className="cc-validation-flow" aria-label="Evidence answer validation pipeline">
      {model.stages.map((stage, index) => <div className="cc-validation-flow-fragment" key={stage.id}>
        <button type="button" className="cc-validation-stage" data-state={stage.state} onClick={() => actions.selectValidationStage(stage.id)}>
          <span>{stageLabel(stage.id)}</span><strong>{stage.detail}</strong><StatusBadge status={stageTone(stage.state)} label={stage.state} />
        </button>{index < model.stages.length - 1 && <ArrowRight size={16} aria-hidden="true" />}
      </div>)}
    </div>
    {model.searchOnly && <div className="cc-inspector-boundary-note">Search scope ends at Evidence. Downstream Answerability / Generation / Grounding / Citation states are authoritatively not applicable.</div>}
  </Panel>;
}

function EvidenceRow({ row }: { row: RuntimeEvidenceRecord }) {
  const { state, actions } = useControlCenter();
  const e = row.evidence;
  const provenance = evidenceProvenanceLabels(e);
  return <tr data-evidence-key={row.key} data-selected={state.selectedEvidenceKey === row.key} onClick={() => actions.selectEvidence(row.key)}>
    <td>{value(e.evidence_position)}</td><td className="cc-mono">{e.evidence_id || "Unavailable"}</td><td>{e.document_path || "Unavailable"}<small>{e.section_path?.join(" / ") || "Section unavailable"}</small></td><td>{lines(e.start_line,e.end_line)}</td><td className="cc-mono">{e.evidence_score == null ? "Unavailable" : e.evidence_score.toFixed(6)}</td><td>{e.selection_reason_code || "Unavailable"}</td><td><div className="cc-provenance-tags">{provenance.length ? provenance.map((p) => <Badge key={p}>{p}</Badge>) : "Unavailable"}</div></td><td>{row.citationIds.length ? row.citationIds.join(", ") : "not cited"}</td>
  </tr>;
}

function EvidenceInventory({ trace }: { trace: RuntimeTraceV1 }) {
  const model = buildEvidenceValidationModel(trace);
  if (!model.evidence.length) return <Panel title="Evidence Inventory"><EmptyState title="No Evidence records" detail="The active Runtime Trace contains no Evidence items. Nothing is reconstructed from Candidates or answer text." /></Panel>;
  return <Panel title={`Evidence Inventory · ${model.evidence.length}`} action={<div className="cc-inline-actions"><Badge tone="info">Candidate ≠ Evidence</Badge><Badge tone="info">Evidence ≠ Citation</Badge></div>}>
    <div className="cc-candidate-table-wrap"><table className="cc-candidate-table cc-evidence-table"><thead><tr><th>Pos</th><th>Evidence ID</th><th>Document / Section</th><th>Lines</th><th>Score</th><th>Selection</th><th>Provenance</th><th>Citation</th></tr></thead><tbody>{model.evidence.map((row) => <EvidenceRow key={row.key} row={row} />)}</tbody></table></div>
  </Panel>;
}

function CitationCard({ row }: { row: RuntimeCitationRecord }) {
  const { state, actions } = useControlCenter();
  const c = row.citation;
  return <button type="button" className="cc-validation-citation" data-citation-key={row.key} data-selected={state.selectedCitationKey === row.key} onClick={() => actions.selectCitation(row.key)}>
    <div><Badge tone={row.evidenceFound ? "info" : "warning"}>{c.citation_id || "Citation ID unavailable"}</Badge>{row.graphRecovered && <Badge tone="success">Graph lineage</Badge>}</div>
    <strong>{c.document_path || row.evidence?.evidence.document_path || "Document unavailable"}</strong>
    <span>{c.section_path?.join(" / ") || row.evidence?.evidence.section_path?.join(" / ") || "Section unavailable"} · {lines(c.start_line ?? row.evidence?.evidence.start_line, c.end_line ?? row.evidence?.evidence.end_line)}</span>
    <code>{c.evidence_id || "Evidence ID unavailable"}</code>
  </button>;
}

function CitationInventory({ trace }: { trace: RuntimeTraceV1 }) {
  const model = buildEvidenceValidationModel(trace);
  if (model.searchOnly) return null;
  return <Panel title={`Citations · ${model.citations.length}`} action={<StatusBadge status={trace.citation.citation_valid === false ? "failed" : trace.citation.citation_valid === true ? "completed" : "unavailable"} label={trace.citation.citation_valid === true ? "valid" : trace.citation.citation_valid === false ? "invalid" : trace.citation.stage_state || "unavailable"} />}>
    {model.citations.length ? <div className="cc-validation-citations">{model.citations.map((row) => <CitationCard key={row.key} row={row} />)}</div> : <EmptyState title="No released Citations" detail={trace.outcome.status === "refused" ? "The governed outcome is a safe refusal; zero Citations are authoritative for this trace." : "This Runtime Trace contains no Citation records."} />}
  </Panel>;
}

function StageSummary({ trace }: { trace: RuntimeTraceV1 }) {
  const model = buildEvidenceValidationModel(trace);
  if (model.searchOnly) return null;
  const safeRefusal = trace.outcome.status === "refused" && !trace.outcome.failure_stage;
  return <div className="cc-validation-stage-grid">
    <Panel title="Answerability"><div className="cc-metric-grid"><Metric label="State" value={value(trace.answerability.answerability_state)} /><Metric label="Answerable" value={trace.answerability.answerable == null ? "Unavailable" : String(trace.answerability.answerable)} /><Metric label="Evidence considered" value={value(trace.answerability.considered_evidence_count)} /><Metric label="Confidence telemetry" value={trace.answerability.confidence == null ? "Unavailable" : trace.answerability.confidence.toFixed(4)} /></div><KeyValue label="Reason" mono>{trace.answerability.reason_code || "Unavailable"}</KeyValue></Panel>
    <Panel title="Generation"><div className="cc-metric-grid"><Metric label="Attempted" value={String(trace.generation.generation_attempted === true)} /><Metric label="Completed" value={String(trace.generation.generation_completed === true)} /><Metric label="Abstained" value={String(trace.generation.generation_abstained === true)} /><Metric label="Latency" value={trace.generation.generation_latency_ms == null ? "Unavailable" : `${trace.generation.generation_latency_ms} ms`} /></div><KeyValue label="Model">{trace.generation.generation_model || "Unavailable"}</KeyValue><KeyValue label="Provider">{trace.generation.generation_provider || "Unavailable"}</KeyValue></Panel>
    <Panel title="Grounding"><div className="cc-metric-grid"><Metric label="Checked" value={String(trace.grounding.grounding_checked === true)} /><Metric label="Passed" value={trace.grounding.grounding_passed == null ? "Unavailable" : String(trace.grounding.grounding_passed)} /><Metric label="Supported claims" value={value(trace.grounding.supported_claim_count)} /><Metric label="Unsupported claims" value={value(trace.grounding.unsupported_claim_count)} /></div><KeyValue label="Status">{trace.grounding.status || "Unavailable"}</KeyValue><KeyValue label="Citation coverage">{trace.grounding.citation_coverage == null ? "Unavailable" : `${Math.round(trace.grounding.citation_coverage * 100)}%`}</KeyValue></Panel>
    <Panel title="Final Outcome" action={<StatusBadge status={safeRefusal ? "refused" : trace.outcome.status === "failed" ? "failed" : "completed"} label={safeRefusal ? "safe refusal" : trace.outcome.status || trace.trace.status} />}><KeyValue label="Answer status">{trace.outcome.answer_status || "Unavailable"}</KeyValue><KeyValue label="Failure stage">{trace.outcome.failure_stage || "None"}</KeyValue><KeyValue label="Final decision">{trace.outcome.final_decision || "Unavailable"}</KeyValue><KeyValue label="Refusal reason" mono>{trace.outcome.refusal_reason_code || "None"}</KeyValue>{safeRefusal && <div className="cc-inspector-boundary-note">Refused ≠ failed. Answerability may pass while Generation still abstains; downstream release remains governed.</div>}</Panel>
  </div>;
}

function Integrity({ trace }: { trace: RuntimeTraceV1 }) {
  const model = buildEvidenceValidationModel(trace);
  const issues = model.orphanCitationCount + model.citationMissingEvidenceIdentityCount + model.citationCandidateMismatchCount + model.evidenceCandidateMissingCount + model.evidenceMissingCandidateIdentityCount;
  return <Panel title="Identity & Semantic Integrity" action={<StatusBadge status={issues ? "partial" : "completed"} label={issues ? `${issues} issue(s)` : "integrity clean"} />}>
    <div className="cc-metric-grid"><Metric label="Candidate = Evidence" value="false" /><Metric label="Evidence = Citation" value="false" /><Metric label="Orphan Citations" value={model.orphanCitationCount} /><Metric label="Citation missing Evidence ID" value={model.citationMissingEvidenceIdentityCount} /><Metric label="Citation/Candidate mismatch" value={model.citationCandidateMismatchCount} /><Metric label="Evidence/Candidate unresolved" value={model.evidenceCandidateMissingCount + model.evidenceMissingCandidateIdentityCount} /></div>
    {issues > 0 && <div className="cc-governance-warning">Identity inconsistencies are displayed unchanged. The frontend does not infer, repair, or reassign Evidence/Citation lineage.</div>}
  </Panel>;
}

export function EvidenceValidationInspector({ trace }: { trace: RuntimeTraceV1 }) {
  const { state } = useControlCenter();
  return <div className="cc-evidence-validation-stack" data-presentation-mode={state.mode} data-runtime-authority="trace-v1">
    <ValidationFlow trace={trace} /><EvidenceInventory trace={trace} /><StageSummary trace={trace} /><CitationInventory trace={trace} /><Integrity trace={trace} />
    <Panel title="Semantic Boundaries"><div className="cc-graph-boundaries"><span>Candidate ≠ Evidence</span><span>Evidence ≠ Citation</span><span>Refused ≠ Failed</span><span>UI selection ≠ runtime execution</span></div></Panel>
  </div>;
}

export function EvidenceValidationPage() {
  const { state } = useShowcase();
  return <div className="cc-page" data-evidence-validation-page="active"><header className="cc-page-heading"><div><h1 className="cc-title">Evidence & Validation</h1><p>Inspect how reranked Candidates become Evidence and how the answer passes or fails downstream validation.</p></div><Badge tone="info"><FileCheck2 size={12} /> Runtime Trace V1 only</Badge></header>{state.trace ? <EvidenceValidationInspector trace={state.trace} /> : <Panel><EmptyState title="No active Runtime Trace" detail="Run Search/Ask or reopen a persisted Conversation turn trace, then inspect its Evidence and downstream validation here." /></Panel>}</div>;
}
