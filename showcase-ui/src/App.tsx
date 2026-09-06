import { AlertTriangle } from "lucide-react";
import { AnswerValidationPanel } from "./components/AnswerValidationPanel";
import { CandidateRerankEvidencePanel } from "./components/CandidateRerankEvidencePanel";
import { EndToEndOverview } from "./components/EndToEndOverview";
import { ExecutiveView } from "./components/ExecutiveView";
import { GuardDecisionPanel } from "./components/GuardDecisionPanel";
import { KnowledgeGraphPanel } from "./components/KnowledgeGraphPanel";
import { PresentationToolbar } from "./components/PresentationToolbar";
import { RetrievalTracePanel } from "./components/RetrievalTracePanel";
import { RetrievalGuardPipeline } from "./components/RetrievalGuardPipeline";
import { RuntimeMetricsPanel } from "./components/RuntimeMetricsPanel";
import { RuntimeStatus } from "./components/RuntimeStatus";
import { ScenarioSelector } from "./components/ScenarioSelector";
import { ScenarioSpotlight } from "./components/ScenarioSpotlight";
import { TraceHeader } from "./components/TraceHeader";
import { TraceTimeline } from "./components/TraceTimeline";
import { PresentationProvider, usePresentation } from "./i18n";
import { ShowcaseProvider, useShowcase } from "./state/ShowcaseState";
import "./styles.css";
import { ControlCenterApp } from "./control-center/app/ControlCenterApp";

function Dashboard() {
  const { state } = useShowcase();
  const { viewMode, recordingMode, t } = usePresentation();
  return (
    <main className={`app-shell ${recordingMode ? "recording-mode" : ""}`}>
      <RuntimeStatus />
      <PresentationToolbar />
      <ScenarioSelector />
      {(state.error || state.schemaError) && (
        <section className="alert" role="status" aria-live="polite">
          <AlertTriangle aria-hidden="true" size={18} />
          <span>{state.schemaError ? t("error.schema", { message: state.schemaError }) : state.error}</span>
        </section>
      )}
      <EndToEndOverview trace={state.trace} />
      {viewMode === "executive" ? (
        <>
          <ScenarioSpotlight trace={state.trace} replay={state.replay} />
          <ExecutiveView trace={state.trace} compact={recordingMode} />
          {!recordingMode && <details className="executive-details">
            <summary>{t("recording.technicalDepth")}</summary>
            <CandidateRerankEvidencePanel trace={state.trace} replay={state.replay} compact />
            <AnswerValidationPanel trace={state.trace} replay={state.replay} compact />
          </details>}
        </>
      ) : recordingMode ? (
        <>
          <ScenarioSpotlight trace={state.trace} replay={state.replay} />
          <CandidateRerankEvidencePanel trace={state.trace} replay={state.replay} compact />
          {state.trace?.query.execution_scope === "ask" && <AnswerValidationPanel trace={state.trace} replay={state.replay} compact />}
        </>
      ) : (
        <>
          <RetrievalGuardPipeline trace={state.trace} replay={state.replay} />
          <KnowledgeGraphPanel trace={state.trace} replay={state.replay} />
          <CandidateRerankEvidencePanel trace={state.trace} replay={state.replay} />
          <AnswerValidationPanel trace={state.trace} replay={state.replay} />
          <div className="dashboard-grid">
            <div className="left-rail">
              <TraceHeader trace={state.trace} />
              <TraceTimeline trace={state.trace} replay={state.replay} />
              <RuntimeMetricsPanel trace={state.trace} />
            </div>
            <div className="main-rail">
              <GuardDecisionPanel trace={state.trace} />
              <RetrievalTracePanel trace={state.trace} />
            </div>
          </div>
        </>
      )}
    </main>
  );
}

export default function App() {
  const legacy = typeof window !== "undefined" && new URLSearchParams(window.location.search).get("ui") === "legacy";
  return (
    <PresentationProvider>
      <ShowcaseProvider>
        {legacy ? <Dashboard /> : <ControlCenterApp />}
      </ShowcaseProvider>
    </PresentationProvider>
  );
}
