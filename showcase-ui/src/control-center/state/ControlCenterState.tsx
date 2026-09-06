import { createContext, useContext, useMemo, useState } from "react";
import { CONTROL_CENTER_NAV, type ControlCenterMode, type ControlCenterNavItem, type InspectorTab } from "../architecture";
import type { ValidationStageId } from "../adapters/evidenceRuntime";

export type InspectorEntityKind = "candidate" | "guard" | "structure_recovery" | "graph_recovery" | "graph_node" | "graph_edge" | "graph_path" | "evidence_stage" | "evidence" | "answerability" | "generation" | "grounding" | "citation" | "outcome" | null;
type State = {
  activeNav: ControlCenterNavItem;
  activeTraceId: string | null;
  sidebarCollapsed: boolean;
  inspectorOpen: boolean;
  inspectorTab: InspectorTab;
  mode: ControlCenterMode;
  selectedCandidateId: string | null;
  selectedGraphNodeId: string | null;
  selectedGraphEdgeId: string | null;
  selectedGraphPathCandidateId: string | null;
  selectedEvidenceKey: string | null;
  selectedCitationKey: string | null;
  selectedInspectorEntityKind: InspectorEntityKind;
};
type Actions = {
  setActiveNav: (x: ControlCenterNavItem) => void;
  bindActiveTrace: (traceId: string | null) => void;
  clearSelections: () => void;
  toggleSidebar: () => void;
  setInspectorOpen: (x: boolean) => void;
  setInspectorTab: (x: InspectorTab) => void;
  setMode: (x: ControlCenterMode) => void;
  selectCandidate: (candidateId: string | null) => void;
  selectInspectorEntity: (kind: "guard" | "structure_recovery" | "graph_recovery" | null) => void;
  selectGraphNode: (nodeId: string | null) => void;
  selectGraphEdge: (edgeId: string | null) => void;
  selectGraphPath: (candidateId: string | null) => void;
  selectValidationStage: (kind: ValidationStageId | null) => void;
  selectEvidence: (key: string | null) => void;
  selectCitation: (key: string | null) => void;
};
const Context = createContext<{ state: State; actions: Actions } | null>(null);

function queryValue(name:string):string|null { try { return typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get(name); } catch { return null; } }
function initialNav():ControlCenterNavItem { const value=queryValue("page"); return value && (CONTROL_CENTER_NAV as readonly string[]).includes(value) ? value as ControlCenterNavItem : "overview"; }
function initialMode():ControlCenterMode { return queryValue("view") === "executive" ? "showcase" : "operator"; }

export function ControlCenterStateProvider({ children }: { children: React.ReactNode }) {
  const [activeNav, setActiveNav] = useState<ControlCenterNavItem>(initialNav);
  const [activeTraceId, setActiveTraceId] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("details");
  const [mode, setMode] = useState<ControlCenterMode>(initialMode);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [selectedGraphNodeId, setSelectedGraphNodeId] = useState<string | null>(null);
  const [selectedGraphEdgeId, setSelectedGraphEdgeId] = useState<string | null>(null);
  const [selectedGraphPathCandidateId, setSelectedGraphPathCandidateId] = useState<string | null>(null);
  const [selectedEvidenceKey, setSelectedEvidenceKey] = useState<string | null>(null);
  const [selectedCitationKey, setSelectedCitationKey] = useState<string | null>(null);
  const [selectedInspectorEntityKind, setSelectedInspectorEntityKind] = useState<InspectorEntityKind>(null);

  const clearGraphSelection = () => { setSelectedGraphNodeId(null); setSelectedGraphEdgeId(null); setSelectedGraphPathCandidateId(null); };
  const clearValidationSelection = () => { setSelectedEvidenceKey(null); setSelectedCitationKey(null); };
  const clearSelections = () => { setSelectedCandidateId(null); clearGraphSelection(); clearValidationSelection(); setSelectedInspectorEntityKind(null); };
  const bindActiveTrace = (traceId: string | null) => { if (traceId === activeTraceId) return; clearSelections(); setActiveTraceId(traceId); };
  const openEntity = (kind: InspectorEntityKind) => { setSelectedInspectorEntityKind(kind); if (kind) { setInspectorOpen(true); setInspectorTab("details"); } };
  const selectCandidate = (candidateId: string | null) => { clearGraphSelection(); clearValidationSelection(); setSelectedCandidateId(candidateId); openEntity(candidateId ? "candidate" : null); };
  const selectInspectorEntity = (kind: "guard" | "structure_recovery" | "graph_recovery" | null) => { setSelectedCandidateId(null); clearGraphSelection(); clearValidationSelection(); openEntity(kind); };
  const selectGraphNode = (nodeId: string | null) => { setSelectedCandidateId(null); clearValidationSelection(); setSelectedGraphEdgeId(null); setSelectedGraphPathCandidateId(null); setSelectedGraphNodeId(nodeId); openEntity(nodeId ? "graph_node" : null); };
  const selectGraphEdge = (edgeId: string | null) => { setSelectedCandidateId(null); clearValidationSelection(); setSelectedGraphNodeId(null); setSelectedGraphPathCandidateId(null); setSelectedGraphEdgeId(edgeId); openEntity(edgeId ? "graph_edge" : null); };
  const selectGraphPath = (candidateId: string | null) => { setSelectedCandidateId(null); clearValidationSelection(); setSelectedGraphNodeId(null); setSelectedGraphEdgeId(null); setSelectedGraphPathCandidateId(candidateId); openEntity(candidateId ? "graph_path" : null); };
  const selectValidationStage = (kind: ValidationStageId | null) => { setSelectedCandidateId(null); clearGraphSelection(); clearValidationSelection(); openEntity(kind); };
  const selectEvidence = (key: string | null) => { setSelectedCandidateId(null); clearGraphSelection(); setSelectedCitationKey(null); setSelectedEvidenceKey(key); openEntity(key ? "evidence" : null); };
  const selectCitation = (key: string | null) => { setSelectedCandidateId(null); clearGraphSelection(); setSelectedEvidenceKey(null); setSelectedCitationKey(key); openEntity(key ? "citation" : null); };

  const value = useMemo(() => ({
    state: { activeNav, activeTraceId, sidebarCollapsed, inspectorOpen, inspectorTab, mode, selectedCandidateId, selectedGraphNodeId, selectedGraphEdgeId, selectedGraphPathCandidateId, selectedEvidenceKey, selectedCitationKey, selectedInspectorEntityKind },
    actions: { setActiveNav, bindActiveTrace, clearSelections, toggleSidebar: () => setSidebarCollapsed(value => !value), setInspectorOpen, setInspectorTab, setMode, selectCandidate, selectInspectorEntity, selectGraphNode, selectGraphEdge, selectGraphPath, selectValidationStage, selectEvidence, selectCitation },
  }), [activeNav, activeTraceId, sidebarCollapsed, inspectorOpen, inspectorTab, mode, selectedCandidateId, selectedGraphNodeId, selectedGraphEdgeId, selectedGraphPathCandidateId, selectedEvidenceKey, selectedCitationKey, selectedInspectorEntityKind]);
  return <Context.Provider value={value}>{children}</Context.Provider>;
}
export function useOptionalControlCenter() { return useContext(Context); }
export function useControlCenter() { const ctx = useOptionalControlCenter(); if (!ctx) throw new Error("useControlCenter must be used within ControlCenterStateProvider"); return ctx; }
