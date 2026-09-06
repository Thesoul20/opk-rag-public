import { afterEach, describe, expect, it, vi } from "vitest";
import { getApiBaseUrl, ShowcaseApiClient } from "../lib/api";
import { envelope, loadTrace } from "./fixtures";

describe("Showcase API client", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("uses the default API URL", () => {
    expect(getApiBaseUrl()).toBe("http://127.0.0.1:8766/api/showcase/v1");
  });

  it("parses health, runtime, and scenarios", async () => {
    const responses = [
      { status: "ok" },
      { runtime_authority: "OPK-RAG Core" },
      { scenarios: [{ scenario_id: "S01", query: "q", command: "search" }] },
    ];
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify(responses.shift()), { status: 200 }));
    const client = new ShowcaseApiClient();
    expect((await client.getHealth()).data?.status).toBe("ok");
    expect((await client.getRuntime()).data?.runtime_authority).toBe("OPK-RAG Core");
    expect((await client.getScenarios()).data?.scenarios[0].scenario_id).toBe("S01");
  });

  it("parses trace envelopes and reports API unavailable", async () => {
    const s03 = loadTrace("s03");
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(new Response(JSON.stringify(envelope(s03)), { status: 200 }));
    const client = new ShowcaseApiClient();
    const result = await client.runScenario("S03");
    expect(result.ok).toBe(true);
    expect(result.data?.trace.graph_recovery.hop_depth).toBe(1);

    vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new Error("ECONNREFUSED"));
    const unavailable = await client.getHealth();
    expect(unavailable.ok).toBe(false);
    expect(unavailable.error?.code).toBe("api_unavailable");
  });

  it("reports incompatible trace schema", async () => {
    const s01 = loadTrace("s01");
    const bad = envelope({ ...s01, trace: { ...s01.trace, trace_schema_version: "bad.version" } });
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(new Response(JSON.stringify(bad), { status: 200 }));
    const result = await new ShowcaseApiClient().runScenario("S01");
    expect(result.ok).toBe(false);
    expect(result.error?.code).toBe("schema_incompatible");
  });
});
