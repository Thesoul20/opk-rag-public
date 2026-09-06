import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";

export function loadTrace(scenarioId: "s01" | "s02" | "s03" | "s04"): RuntimeTraceV1 {
  const path = resolve(process.cwd(), "..", "evaluation-data", "showcase", `runtime_trace_v1_live_${scenarioId}.json`);
  return JSON.parse(readFileSync(path, "utf-8")) as RuntimeTraceV1;
}

export function envelope(trace: RuntimeTraceV1) {
  return {
    schema_version: "opk-rag.showcase-api.v1",
    trace_id: trace.trace.trace_id,
    status: trace.trace.status,
    trace,
  };
}
