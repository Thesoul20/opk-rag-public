import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { buildExecutiveStory, recoveryKind } from "../control-center/adapters/executiveRuntime";
import { ShowcaseExecutivePage } from "../control-center/pages/ShowcaseExecutivePage";
import { ControlCenterStateProvider } from "../control-center/state/ControlCenterState";
import { RuntimeStatusProvider } from "../control-center/state/RuntimeStatusState";
import { ShowcaseApiClient } from "../lib/api";
import { ShowcaseProvider } from "../state/ShowcaseState";
import type { RuntimeTraceV1 } from "../types/runtimeTrace";
import { envelope, loadTrace } from "./fixtures";

function special(name:string):RuntimeTraceV1{return JSON.parse(readFileSync(resolve(process.cwd(),"..","evaluation-data","showcase",name),"utf-8")) as RuntimeTraceV1}
const s03Ask=()=>special("runtime_trace_v1_live_s03_ask_task0233.json");
const runtime:any={schema_version:"opk-rag.control-center-runtime-status.v1",observed_at:"2026-09-02T00:00:00Z",ui_decision_authority:false,graph_max_hop:1,recovery_max_attempts:1,health:{system_status:"healthy",search_ready:true,ask_ready:true,generation_health:"configured"},knowledge_base:{available:true,knowledge_base_name:"OPK",indexed_document_count:8,chunk_count:11},qdrant:{server_reachable:true,backend:"qdrant"},graph:{graph_hop_limit:1,graph_freshness:"A"},models:{embedding:{model:"Qwen/Qwen3-Embedding-0.6B"},reranker:{model:"BAAI/bge-reranker-v2-m3"},generation:{model:"deepseek-v4-flash",configured:true}},gpu:{gpu_available:true,device_name:"NVIDIA GeForce RTX 3060"}};
class FakeEventSource{onmessage:any;onerror:any;constructor(_url:string){}addEventListener(){}close(){}}
class Client extends ShowcaseApiClient{
 runScenarioCalls:string[]=[];
 async getHealth(){return {ok:true as const,data:{api_ready:true}}}
 async getRuntime(){return {ok:true as const,data:{runtime_authority:"OPK-RAG"}}}
 async getScenarios(){return {ok:true as const,data:{scenarios:[{scenario_id:"S01",title:"Normal Retrieval",query:"q1",command:"search"},{scenario_id:"S02",title:"Structure Recovery",query:"q2",command:"search"},{scenario_id:"S03",title:"Graph Recovery",query:"q3",command:"search"},{scenario_id:"S04",title:"Safe Refusal",query:"q4",command:"ask"}]}}}
 async runScenario(id:string){this.runScenarioCalls.push(id);const key=id.toLowerCase() as "s01"|"s02"|"s03"|"s04";return {ok:true as const,data:envelope(loadTrace(key))}}
}
function renderWithClient(client:Client){return render(<RuntimeStatusProvider loader={async()=>runtime}><ControlCenterStateProvider><ShowcaseProvider client={client}><ShowcaseExecutivePage/></ShowcaseProvider></ControlCenterStateProvider></RuntimeStatusProvider>)}

describe("TASK-0278 Showcase / Executive View",()=>{
 beforeEach(()=>vi.stubGlobal("EventSource",FakeEventSource)); afterEach(()=>vi.unstubAllGlobals());
 it("maps frozen S01/S02/S03 recovery semantics without synthetic action",()=>{expect(recoveryKind(loadTrace("s01"))).toBe("none");expect(recoveryKind(loadTrace("s02"))).toBe("structure");expect(recoveryKind(loadTrace("s03"))).toBe("graph")});
 it("preserves S02 pool growth without equating delta to expanded count",()=>{const t=loadTrace("s02");const story=buildExecutiveStory(t);expect(story.messages.join(" ")).toContain("3 to 7");expect(t.structure_recovery.expanded_candidate_count).toBe(2)});
 it("preserves S03 Ask one-hop Graph Candidate → Evidence C4 → Citation C4 lineage",()=>{const story=buildExecutiveStory(s03Ask());expect(story.graph.active).toBe(true);expect(story.graph.hop).toBe(1);expect(story.graph.recoveredCandidates).toBe(1);expect(story.graph.evidenceId).toBe("C4");expect(story.graph.citationId).toBe("C4");expect(story.graph.relation).toBe("LINKS_TO")});
 it("presents S04 as safe refusal, not runtime failure",()=>{const story=buildExecutiveStory(loadTrace("s04"));expect(story.safeRefusal).toBe(true);expect(story.status).toBe("refused");expect(story.messages.join(" ")).toContain("Generation abstained");expect(loadTrace("s04").outcome.failure_stage).toBeNull()});
 it("keeps Search validation N/A instead of inventing an answer chain",()=>{const story=buildExecutiveStory(loadTrace("s01"));expect(story.stages.find(s=>s.id==="validate")?.state).toBe("not_applicable");expect(story.messages.join(" ")).toContain("Search scope ends at Evidence")});
 it("renders modern empty Showcase without fabricated runtime metrics",async()=>{const client=new Client();renderWithClient(client);expect(await screen.findByText("A governed personal-knowledge RAG, built to explain its own execution.")).toBeInTheDocument();expect(screen.getByText("No active Runtime Trace")).toBeInTheDocument();expect(screen.getByText("System Architecture")).toBeInTheDocument()});
 it("scenario selection does not execute and explicit Run Demo uses existing scenario action",async()=>{const user=userEvent.setup();const client=new Client();renderWithClient(client);await screen.findByText("Normal Retrieval");await user.click(screen.getByText("Structure Recovery"));expect(client.runScenarioCalls).toEqual([]);await user.click(screen.getByRole("button",{name:/Run Demo/}));await waitFor(()=>expect(client.runScenarioCalls).toEqual(["S02"]));expect(await screen.findByText("Current Execution")).toBeInTheDocument()});
 it("renders the active S03 executive Graph story and same-trace drill-down controls",async()=>{const client=new Client();renderWithClient(client);await screen.findByText("Graph Recovery");await userEvent.setup().click(screen.getByText("Graph Recovery"));await userEvent.setup().click(screen.getByRole("button",{name:/Run Demo/}));expect(await screen.findByLabelText("Executive Runtime Storyboard")).toBeInTheDocument();expect(screen.getAllByText("LINKS_TO").length).toBeGreaterThan(0);expect(screen.getByRole("button",{name:"Inspect Runtime"})).toBeInTheDocument();expect(screen.getByRole("button",{name:"Inspect Graph"})).toBeInTheDocument();expect(screen.getByRole("button",{name:"Inspect Evidence"})).toBeInTheDocument()});
 it("shows governance and does not claim hidden reasoning or correctness confidence",async()=>{const user=userEvent.setup();const client=new Client();renderWithClient(client);await screen.findByText("Normal Retrieval");await user.click(screen.getByRole("button",{name:/Run Demo/}));expect(await screen.findByText("UI authority = false")).toBeInTheDocument();expect(screen.getByText("Graph max hop = 1")).toBeInTheDocument();expect(screen.queryByText(/chain[- ]of[- ]thought/i)).not.toBeInTheDocument();expect(screen.queryByText(/correctness score/i)).not.toBeInTheDocument()});
 it("uses only live trace timing and exposes historical-mixed=false",()=>{const t=s03Ask();const story=buildExecutiveStory(t);expect(t.timings.historical_benchmark_values_mixed_into_live_trace).toBe(false);expect(story.status).toBe("completed")});
});
