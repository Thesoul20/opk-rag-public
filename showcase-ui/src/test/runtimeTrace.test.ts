import { describe, expect, it } from "vitest";
import { formatMs } from "../lib/format";
import { appendRuntimeEvent, emptyReplayState } from "../lib/sse";
import {
  guardTraceVersion,
  lifecycleFromEvent,
  lifecycleFromTraceStatus,
  RUNTIME_TRACE_EVENT_SCHEMA_VERSION,
  stageLabel,
  type RuntimeTraceEventV1,
} from "../types/runtimeTrace";
import { loadTrace } from "./fixtures";

function event(sequence: number, overrides: Partial<RuntimeTraceEventV1> = {}): RuntimeTraceEventV1 {
  return {
    event_schema_version: RUNTIME_TRACE_EVENT_SCHEMA_VERSION,
    event_id: `trace:${sequence}`,
    trace_id: "trace",
    sequence,
    event_type: "stage_snapshot",
    stage: "guard",
    timestamp: null,
    payload: {},
    terminal: false,
    ...overrides,
  };
}

describe("Runtime Trace V1 mapping", () => {
  it("maps lifecycle states and preserves refused != failed", () => {
    expect(lifecycleFromTraceStatus("running")).toBe("executing");
    expect(lifecycleFromTraceStatus("completed")).toBe("completed");
    expect(lifecycleFromTraceStatus("refused")).toBe("refused");
    expect(lifecycleFromTraceStatus("failed")).toBe("failed");
    expect(lifecycleFromEvent(event(9, { event_type: "trace_refused", terminal: true }))).toBe("refused");
  });

  it("maps stage labels without changing stage identity", () => {
    expect(stageLabel("graph_recovery")).toBe("Graph Recovery");
    expect(stageLabel("structure_recovery")).toBe("Structure Recovery");
  });

  it("renders null timing as dashes, never zero milliseconds", () => {
    expect(formatMs(null)).toBe("--");
    expect(formatMs(undefined)).toBe("--");
    expect(formatMs(0)).toBe("0ms");
  });

  it("validates real trace version and rejects incompatible schema", () => {
    const s03 = loadTrace("s03");
    expect(guardTraceVersion(s03).ok).toBe(true);
    expect(guardTraceVersion({ ...s03, trace: { ...s03.trace, trace_schema_version: "wrong" } }).ok).toBe(false);
  });

  it("dedupes event replay by sequence and event id while keeping terminal state", () => {
    let state = emptyReplayState();
    state = appendRuntimeEvent(state, event(2));
    state = appendRuntimeEvent(state, event(1));
    state = appendRuntimeEvent(state, event(2));
    state = appendRuntimeEvent(state, event(3, { event_type: "trace_completed", terminal: true }));
    expect(state.events.map((row) => row.sequence)).toEqual([1, 2, 3]);
    expect(state.terminal).toBe(true);
    expect(state.lifecycle).toBe("completed");
  });

  it("captures S01 skipped recovery, S02 structure, S03 graph hop, and S04 refusal", () => {
    const s01 = loadTrace("s01");
    const s02 = loadTrace("s02");
    const s03 = loadTrace("s03");
    const s04 = loadTrace("s04");
    expect(s01.structure_recovery.stage_state).toBe("skipped");
    expect(s01.graph_recovery.stage_state).toBe("skipped");
    expect(s02.structure_recovery.stage_state).toBe("completed");
    expect(s02.structure_recovery.triggered).toBe(true);
    expect(s03.graph_recovery.graph_activated).toBe(true);
    expect(s03.graph_recovery.hop_depth).toBe(1);
    expect(s04.trace.status).toBe("refused");
    expect(s04.guard.fail_closed).toBe(true);
  });

  it("keeps Candidate and Evidence as separate semantic sections", () => {
    const s03 = loadTrace("s03");
    expect(s03.evidence.candidate_evidence_semantic_separation).toBe(true);
    expect(s03.retrieval.candidates[0]).toHaveProperty("candidate_id");
    expect(s03.evidence.evidence_items[0]).toHaveProperty("evidence_id");
  });
});
