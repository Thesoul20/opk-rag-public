import { Play, Search } from "lucide-react";
import { usePresentation, type TranslationKey } from "../i18n";
import { useShowcase } from "../state/ShowcaseState";
import type { ScenarioRecord } from "../lib/api";
import { SHOWCASE_SCENARIO_ORDER } from "../presentation/endToEnd";

function scenarioLabel(id: string, t: (key: TranslationKey) => string, fallback?: ScenarioRecord["name"]): string {
  const key = `scenario.${id}` as TranslationKey;
  return SHOWCASE_SCENARIO_ORDER.includes(id as (typeof SHOWCASE_SCENARIO_ORDER)[number]) ? t(key) : fallback || id;
}

export function ScenarioSelector() {
  const { state, actions } = useShowcase();
  const { t, recordingMode } = usePresentation();
  const descriptionKey = `scenario.${state.selectedScenarioId}.description` as TranslationKey;
  const hasKnownScenario = SHOWCASE_SCENARIO_ORDER.includes(state.selectedScenarioId as (typeof SHOWCASE_SCENARIO_ORDER)[number]);
  const executing = state.lifecycle === "executing" || state.lifecycle === "connecting";
  return (
    <section className={`control-band ${recordingMode ? "recording-controls" : ""}`} aria-label={t("scenario.label")}>
      {recordingMode ? <div className="demo-control-strip" role="group" aria-label={t("recording.demoControls")}>
        {SHOWCASE_SCENARIO_ORDER.map((id) => <button type="button" key={id} className={state.selectedScenarioId === id ? "selected" : ""} onClick={() => actions.selectScenario(id)} disabled={executing || !state.scenarios.some((row) => row.scenario_id === id)}><strong>{id}</strong><span>{scenarioLabel(id,t)}</span></button>)}
      </div> : <>
        <div className="control-group">
          <label htmlFor="scenario-select">{t("scenario.label")}</label>
          <select id="scenario-select" value={state.selectedScenarioId} onChange={(event) => actions.selectScenario(event.target.value)} disabled={!state.scenarios.length || executing}>
            {state.scenarios.length ? state.scenarios.map((scenario) => <option key={scenario.scenario_id} value={scenario.scenario_id}>{scenario.scenario_id} — {scenarioLabel(scenario.scenario_id, t, scenario.name || scenario.title || scenario.description)}</option>) : <option>{t("scenario.apiUnavailable")}</option>}
          </select>
          {hasKnownScenario && <p className="scenario-description">{t(descriptionKey)}</p>}
        </div>
        <div className="segmented" role="group" aria-label={t("scenario.mode")}>
          <button type="button" className={state.mode === "search" ? "selected" : ""} onClick={() => actions.setMode("search")} disabled={executing}>{t("common.search")}</button>
          <button type="button" className={state.mode === "ask" ? "selected" : ""} onClick={() => actions.setMode("ask")} disabled={executing}>{t("common.ask")}</button>
        </div>
        <div className="query-box"><label htmlFor="query-input">{t("scenario.query")}</label><input id="query-input" value={state.query} onChange={(event) => actions.setQuery(event.target.value)} placeholder={t("scenario.placeholder")} disabled={executing} /></div>
      </>}
      <button className="primary-button" type="button" onClick={() => void actions.runSelectedScenario()} disabled={!state.scenarios.length || executing}>
        <Play aria-hidden="true" size={17} />{executing ? t("recording.running") : t("scenario.run")}
      </button>
      {!recordingMode && <button className="secondary-button" type="button" onClick={() => void actions.runQuery()} disabled={!state.query.trim() || executing}><Search aria-hidden="true" size={17} />{t("scenario.runQuery")}</button>}
    </section>
  );
}
