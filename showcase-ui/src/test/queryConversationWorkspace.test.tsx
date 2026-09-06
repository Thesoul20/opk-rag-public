import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryWorkspacePage } from "../control-center/pages/QueryWorkspacePage";
import { ControlCenterStateProvider } from "../control-center/state/ControlCenterState";
import { RuntimeStatusProvider } from "../control-center/state/RuntimeStatusState";
import { ShowcaseApiClient, type ConversationSessionResponse } from "../lib/api";
import { ShowcaseProvider } from "../state/ShowcaseState";
import { envelope, loadTrace } from "./fixtures";

const session = { id:"11111111-1111-1111-1111-111111111111", knowledge_base_id:"kb", title:"Persisted chat", status:"active", created_at:"2026-09-02T00:00:00Z", updated_at:"2026-09-02T00:00:00Z", last_turn_at:"2026-09-02T00:00:01Z", turn_count:1 };
const firstTurn:any = { id:"t1", session_id:session.id, turn_number:1, user_query:"First question", standalone_query:"First question", is_followup:false, answer_decision:"answer", answer_text:"First grounded answer [C1].", turn_status:"completed", retrieval_mode:"hybrid", latency_ms:120, trace_id:"trace-first", citations:[{citation_id:"C1",relative_path:"notes/first.md",heading_path:["First"],start_line:1,end_line:3,source_status:"source_current"}] };

class FakeEventSource {
  onmessage:any; onerror:any;
  constructor(_url:string) {}
  addEventListener() {}
  close() {}
}

class Client extends ShowcaseApiClient {
  turns:any[] = [firstTurn];
  async getHealth(){return {ok:true as const,data:{api_ready:true}}}
  async getRuntime(){return {ok:true as const,data:{runtime_authority:"OPK-RAG"}}}
  async getScenarios(){return {ok:true as const,data:{scenarios:[]}}}
  async runSearch(){const trace=loadTrace("s01");return {ok:true as const,data:envelope(trace)}}
  async runAsk(){const trace=loadTrace("s01");return {ok:true as const,data:{...envelope(trace),result:{status:"answered",answerable:true,answer:"Direct grounded answer [C1].",refusal_reason_code:null,generation_latency_ms:45,grounding_valid:true,citations:[{citation_id:"C1",relative_path:"notes/direct.md",heading_path:["Direct"],start_line:4,end_line:8,snippet:"direct evidence"}]}}}}
  async getTrace(traceId:string){const trace=structuredClone(loadTrace("s01"));trace.trace.trace_id=traceId;return {ok:true as const,data:envelope(trace)}}
  async listSessions(){return {ok:true as const,data:{schema_version:"opk-rag.control-center-conversation.v1",sessions:[{...session,turn_count:this.turns.length}]}}}
  async createSession(){return {ok:true as const,data:{schema_version:"opk-rag.control-center-conversation.v1",session:{...session,turn_count:0}}}}
  async getSession():Promise<any>{return {ok:true,data:{schema_version:"opk-rag.control-center-conversation.v1",session:{...session,turn_count:this.turns.length},turns:[...this.turns]} satisfies ConversationSessionResponse}}
  async askSession(_id:string,query:string){const second={...firstTurn,id:"t2",turn_number:2,user_query:query,standalone_query:`First question；追问：${query}`,is_followup:true,answer_text:"Second grounded answer [C1].",trace_id:"trace-second"};this.turns.push(second);const trace=loadTrace("s01");return {ok:true as const,data:{schema_version:"opk-rag.control-center-conversation.v1",session:{...session,turn_count:2},turn:second,resolution:{standalone_query:second.standalone_query,is_followup:true,referenced_turn_numbers:[1],referenced_citation_ids:["C1"]},answer:{status:"answered",answerable:true,answer:second.answer_text,citations:second.citations},idempotent_replay:false,trace:envelope(trace)}}}
}

const runtime:any={schema_version:"opk-rag.control-center-runtime-status.v1",observed_at:"2026-09-02T00:00:00Z",ui_decision_authority:false,graph_max_hop:1,recovery_max_attempts:1,health:{system_status:"healthy",search_ready:true,ask_ready:true,generation_health:"not_probed"},knowledge_base:{available:true,knowledge_base_name:"OPK"},qdrant:{server_reachable:true},graph:{graph_hop_limit:1},models:{generation:{configured:true}},gpu:{gpu_available:true}};

function renderPage(client:Client){return render(<RuntimeStatusProvider loader={async()=>runtime}><ControlCenterStateProvider><ShowcaseProvider client={client}><QueryWorkspacePage client={client}/></ShowcaseProvider></ControlCenterStateProvider></RuntimeStatusProvider>)}

describe("TASK-0272 Query / Conversation Workspace",()=>{
 beforeEach(()=>vi.stubGlobal("EventSource",FakeEventSource));
 afterEach(()=>vi.unstubAllGlobals());
 it("executes real Search and Ask surfaces without fabricating answer text",async()=>{const user=userEvent.setup();const client=new Client();renderPage(client);const input=screen.getByRole("textbox",{name:"Query input"});await user.type(input,"direct question");await user.click(screen.getAllByRole("button",{name:"Ask"})[1]);expect(await screen.findByText("Direct grounded answer [C1].")).toBeInTheDocument();expect(screen.getByText("notes/direct.md")).toBeInTheDocument();await user.click(screen.getAllByRole("button",{name:"Search"})[0]);await user.clear(input);await user.type(input,"search query");await user.click(screen.getAllByRole("button",{name:"Search"})[1]);const path=loadTrace("s01").retrieval.candidates[0].document_path!;expect((await screen.findAllByText(path)).length).toBeGreaterThan(0)});
 it("loads persisted sessions and submits a multi-turn follow-up",async()=>{const user=userEvent.setup();const client=new Client();renderPage(client);await user.click(screen.getByRole("button",{name:"Conversation"}));expect((await screen.findAllByText("Persisted chat")).length).toBeGreaterThanOrEqual(2);expect(await screen.findByText("First grounded answer [C1].")).toBeInTheDocument();const input=screen.getByRole("textbox",{name:"Conversation question"});await user.type(input,"Continue");await user.click(screen.getByRole("button",{name:"Send"}));expect(await screen.findByText("Second grounded answer [C1].")).toBeInTheDocument();expect(screen.getByText("First question；追问：Continue")).toBeInTheDocument();expect(client.turns).toHaveLength(2);await user.click(screen.getAllByRole("button",{name:/trace trace-first/})[0]);expect(await screen.findByText("Retrieval Overview")).toBeInTheDocument()});
 it("shows runtime gating and keeps UI decision authority false",async()=>{const client=new Client();renderPage(client);expect(await screen.findByText("ask ready")).toBeInTheDocument();expect(document.querySelector('[data-query-workspace="active"]')).toHaveAttribute("data-presentation-mode","operator")});
});
