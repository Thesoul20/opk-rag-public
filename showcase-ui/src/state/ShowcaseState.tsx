import { createContext, useCallback, useContext, useEffect, useMemo, useReducer, useRef } from "react";
import { ShowcaseApiClient, showcaseApi, type HealthResponse, type QueryAnswerResult, type RuntimeAuthorityResponse, type ScenarioRecord } from "../lib/api";
import { RuntimeEventSourceClient, appendRuntimeEvent, emptyReplayState, type EventReplayState } from "../lib/sse";
import { lifecycleFromTraceStatus, type ExecutionLifecycle, type RuntimeTraceV1, type TraceEnvelope } from "../types/runtimeTrace";

interface ShowcaseState {
  lifecycle: ExecutionLifecycle;
  health?: HealthResponse;
  runtime?: RuntimeAuthorityResponse;
  scenarios: ScenarioRecord[];
  selectedScenarioId: string;
  query: string;
  mode: "search" | "ask";
  trace?: RuntimeTraceV1;
  traceId?: string;
  answerResult?: QueryAnswerResult;
  replay: EventReplayState;
  error?: string;
  schemaError?: string;
}

type Action =
  | { type: "bootstrap"; health?: HealthResponse; runtime?: RuntimeAuthorityResponse; scenarios: ScenarioRecord[]; requestedScenarioId?: string | null; error?: string }
  | { type: "selectScenario"; scenarioId: string }
  | { type: "query"; query: string }
  | { type: "mode"; mode: "search" | "ask" }
  | { type: "executing" }
  | { type: "trace"; trace: RuntimeTraceV1; traceId: string; answerResult?: QueryAnswerResult }
  | { type: "replay"; replay: EventReplayState }
  | { type: "apiError"; error: string }
  | { type: "schemaError"; error: string };

const initialState: ShowcaseState = {
  lifecycle: "idle",
  scenarios: [],
  selectedScenarioId: "S01",
  query: "",
  mode: "search",
  replay: emptyReplayState(),
};

function reducer(state: ShowcaseState, action: Action): ShowcaseState {
  switch (action.type) {
    case "bootstrap": {
      const preferred = action.requestedScenarioId ? action.scenarios.find((row) => row.scenario_id === action.requestedScenarioId) : undefined;
      const selected = preferred || action.scenarios[0];
      return {
        ...state,
        health: action.health,
        runtime: action.runtime,
        scenarios: action.scenarios,
        selectedScenarioId: selected?.scenario_id || state.selectedScenarioId,
        query: selected?.query || state.query,
        mode: selected?.command === "ask" ? "ask" : selected?.command === "search" ? "search" : state.mode,
        error: action.error,
      };
    }
    case "selectScenario": {
      const scenario = state.scenarios.find((row) => row.scenario_id === action.scenarioId);
      return {
        ...state,
        selectedScenarioId: action.scenarioId,
        query: scenario?.query || state.query,
        mode: scenario?.command === "ask" ? "ask" : scenario?.command === "search" ? "search" : state.mode,
      };
    }
    case "query":
      return { ...state, query: action.query };
    case "mode":
      return { ...state, mode: action.mode };
    case "executing":
      return { ...state, lifecycle: "executing", error: undefined, schemaError: undefined, replay: emptyReplayState() };
    case "trace":
      return { ...state, trace: action.trace, traceId: action.traceId, answerResult: action.answerResult, lifecycle: lifecycleFromTraceStatus(action.trace.trace.status) };
    case "replay":
      return { ...state, replay: action.replay, lifecycle: action.replay.terminal ? action.replay.lifecycle : state.lifecycle };
    case "apiError":
      return { ...state, lifecycle: "failed", error: action.error };
    case "schemaError":
      return { ...state, lifecycle: "failed", schemaError: action.error };
  }
}

interface ShowcaseActions {
  refresh: () => Promise<void>;
  runSelectedScenario: () => Promise<void>;
  runQuery: () => Promise<void>;
  runQueryInput: (query: string, mode: "search" | "ask") => Promise<void>;
  acceptTraceEnvelope: (envelope: TraceEnvelope) => void;
  selectScenario: (scenarioId: string) => void;
  setQuery: (query: string) => void;
  setMode: (mode: "search" | "ask") => void;
}

const ShowcaseContext = createContext<{ state: ShowcaseState; actions: ShowcaseActions } | null>(null);

export function ShowcaseProvider({ children, client = showcaseApi }: { children: React.ReactNode; client?: ShowcaseApiClient }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const deepLinkLoaded = useRef(false);

  const refresh = useCallback(async () => {
    const [health, runtime, scenarios] = await Promise.all([client.getHealth(), client.getRuntime(), client.getScenarios()]);
    if (!health.ok || !runtime.ok || !scenarios.ok) {
      dispatch({ type: "bootstrap", scenarios: [], error: "Showcase API unavailable" });
      return;
    }
    const requestedScenarioId = typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("scenario") : null;
    dispatch({ type: "bootstrap", health: health.data, runtime: runtime.data, scenarios: scenarios.data?.scenarios || [], requestedScenarioId });
  }, [client]);

  const attachReplay = useCallback(
    (traceId: string) => {
      const source = new RuntimeEventSourceClient((after) => client.eventUrl(traceId, after));
      return source.connect((replay) => dispatch({ type: "replay", replay }));
    },
    [client],
  );

  const acceptTraceEnvelope = useCallback((envelope: TraceEnvelope) => {
    const answerResult = (envelope as TraceEnvelope & { result?: QueryAnswerResult | null }).result ?? undefined;
    dispatch({ type: "trace", trace: envelope.trace, traceId: envelope.trace_id, answerResult });
    attachReplay(envelope.trace_id);
  }, [attachReplay]);

  const acceptResult = useCallback(
    async (runner: () => ReturnType<ShowcaseApiClient["runScenario"]>) => {
      dispatch({ type: "executing" });
      const result = await runner();
      if (!result.ok || !result.data) {
        const message = result.error?.code === "schema_incompatible" ? result.error.message : result.error?.message || "Showcase API unavailable";
        dispatch(result.error?.code === "schema_incompatible" ? { type: "schemaError", error: message } : { type: "apiError", error: message });
        return;
      }
      acceptTraceEnvelope(result.data);
    },
    [acceptTraceEnvelope],
  );

  useEffect(() => {
    if (deepLinkLoaded.current || typeof window === "undefined") return;
    const traceId = new URLSearchParams(window.location.search).get("trace");
    if (!traceId) return;
    deepLinkLoaded.current = true;
    void client.getTrace(traceId).then((result) => {
      if (!result.ok || !result.data) return;
      dispatch({ type: "trace", trace: result.data.trace, traceId: result.data.trace_id });
      attachReplay(result.data.trace_id);
    });
  }, [attachReplay, client]);

  const runSelectedScenario = useCallback(async () => {
    await acceptResult(() => client.runScenario(state.selectedScenarioId));
  }, [acceptResult, client, state.selectedScenarioId]);

  const runQueryInput = useCallback(async (queryInput: string, mode: "search" | "ask") => {
    const query = queryInput.trim();
    if (!query) return;
    dispatch({ type: "query", query });
    dispatch({ type: "mode", mode });
    await acceptResult(() => (mode === "ask" ? client.runAsk(query) : client.runSearch(query)));
  }, [acceptResult, client]);

  const runQuery = useCallback(async () => {
    await runQueryInput(state.query, state.mode);
  }, [runQueryInput, state.mode, state.query]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const actions = useMemo<ShowcaseActions>(
    () => ({
      refresh,
      runSelectedScenario,
      runQuery,
      runQueryInput,
      acceptTraceEnvelope,
      selectScenario: (scenarioId) => dispatch({ type: "selectScenario", scenarioId }),
      setQuery: (query) => dispatch({ type: "query", query }),
      setMode: (mode) => dispatch({ type: "mode", mode }),
    }),
    [acceptTraceEnvelope, refresh, runQuery, runQueryInput, runSelectedScenario],
  );

  return <ShowcaseContext.Provider value={{ state, actions }}>{children}</ShowcaseContext.Provider>;
}

export function useOptionalShowcase() { return useContext(ShowcaseContext); }

export function useShowcase() {
  const context = useOptionalShowcase();
  if (!context) throw new Error("useShowcase must be used inside ShowcaseProvider");
  return context;
}

export { appendRuntimeEvent };
