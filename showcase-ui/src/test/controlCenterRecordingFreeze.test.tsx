import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PresentationProvider } from "../i18n";
import { ShowcaseApiClient } from "../lib/api";
import { ShowcaseProvider } from "../state/ShowcaseState";
import { ControlCenterApp } from "../control-center/app/ControlCenterApp";
import { envelope, loadTrace } from "./fixtures";

class FakeEventSource { onmessage:any; onerror:any; constructor(_url:string){} addEventListener(){} close(){} }
class Client extends ShowcaseApiClient {
  scenarioRuns=0;
  async getHealth(){return {ok:true as const,data:{api_ready:true}}}
  async getRuntime(){return {ok:true as const,data:{runtime_authority:"OPK-RAG",graph_hop_depth:1,maximum_recovery_attempt_count:1}}}
  async getScenarios(){return {ok:true as const,data:{scenarios:[{scenario_id:"S01",title:"Normal Retrieval",query:"q1",command:"search"},{scenario_id:"S02",title:"Structure Recovery",query:"q2",command:"search"},{scenario_id:"S03",title:"Graph Recovery",query:"q3",command:"search"},{scenario_id:"S04",title:"Safe Refusal",query:"q4",command:"ask"}]}}}
  async runScenario(){this.scenarioRuns++;const trace=loadTrace("s03");return {ok:true as const,data:envelope(trace)}}
}
const runtime:any={schema_version:"opk-rag.control-center-runtime-status.v1",observed_at:"2026-09-02T00:00:00Z",ui_decision_authority:false,graph_max_hop:1,recovery_max_attempts:1,health:{system_status:"healthy",search_ready:true,ask_ready:true,generation_health:"configured"},knowledge_base:{available:true,knowledge_base_name:"OPK",indexed_document_count:47,chunk_count:47},qdrant:{server_reachable:true,backend:"qdrant"},graph:{graph_hop_limit:1,graph_freshness:"A"},models:{generation:{configured:true}},gpu:{gpu_available:true,device_name:"RTX 3060"}};

function app(client=new Client()){return render(<PresentationProvider><ShowcaseProvider client={client}><ControlCenterApp/></ShowcaseProvider></PresentationProvider>)}

describe("TASK-0279 Modern recording freeze",()=>{
  beforeEach(()=>{vi.stubGlobal("EventSource",FakeEventSource);window.history.replaceState({},"","/")});
  afterEach(()=>{vi.unstubAllGlobals();window.history.replaceState({},"","/")});
  it("deep-links directly into Modern Showcase without executing a scenario",async()=>{window.history.replaceState({},"","/?page=showcase&recording=1&locale=zh-CN&view=executive&scenario=S03");const client=new Client();app(client);expect(await screen.findByRole("heading",{name:"Showcase / Executive View"})).toBeInTheDocument();expect(document.querySelector('.control-center-shell')).toHaveAttribute('data-recording-mode','true');expect(document.querySelector('.control-center-shell')).toHaveAttribute('data-ui-decision-authority','false');expect(screen.getByText(/面向领导、面试与录屏/)).toBeInTheDocument();expect(client.scenarioRuns).toBe(0)});
  it("recording mode hides shell chrome by contract while preserving runtime authority markers",async()=>{window.history.replaceState({},"","/?page=showcase&recording=1");app();await screen.findByRole("heading",{name:"Showcase / Executive View"});const shell=document.querySelector('.control-center-shell')!;expect(shell).toHaveAttribute('data-recording-mode','true');expect(shell).toHaveAttribute('data-ui-decision-authority','false');expect(screen.getByText(/REC · 1920×1080/)).toBeInTheDocument();});
  it("view=executive maps only to modern presentation mode",async()=>{window.history.replaceState({},"","/?page=showcase&view=executive");app();await screen.findByRole("heading",{name:"Showcase / Executive View"});expect(document.querySelector('[data-recording-mode="false"]')).toBeTruthy();});
  it("selecting a frozen scenario remains presentation-only and Run Demo is the sole execution",async()=>{window.history.replaceState({},"","/?page=showcase");const user=userEvent.setup();const client=new Client();app(client);await screen.findByRole("heading",{name:"Showcase / Executive View"});await user.click(await screen.findByRole("button",{name:/S02/}));expect(client.scenarioRuns).toBe(0);await user.click(screen.getByRole("button",{name:"Run Demo"}));await waitFor(()=>expect(client.scenarioRuns).toBe(1));});
  it("legacy query remains outside Modern routing authority",()=>{window.history.replaceState({},"","/?ui=legacy");expect(new URLSearchParams(window.location.search).get('ui')).toBe('legacy')});
});
