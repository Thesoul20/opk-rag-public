import { useCallback, useEffect, useMemo, useState } from "react";
import { FileCheck2, GitBranch, MessageSquarePlus, Play, RefreshCw, Search, Send, Sparkles, Workflow } from "lucide-react";
import {
  showcaseApi,
  type ShowcaseApiClient,
  type ConversationSessionResponse,
  type ConversationSessionSummary,
  type QueryCitation,
} from "../../lib/api";
import { useShowcase } from "../../state/ShowcaseState";
import { Badge, Button, EmptyState, KeyValue, Panel, StatusBadge } from "../components/Primitives";
import { RetrievalCandidateInspection } from "../components/RetrievalCandidateInspector";
import { GuardRecoveryControlPlane } from "../components/GuardRecoveryControlPlane";
import { EvidenceValidationSummary } from "../components/EvidenceValidationSummary";
import { useControlCenter } from "../state/ControlCenterState";
import { useRuntimeStatus } from "../state/RuntimeStatusState";

type WorkspaceMode = "search" | "ask" | "conversation";

function Heading() {
  return <header className="cc-page-heading"><div><h1 className="cc-title">Query</h1><p>Real Search, Ask and persistent conversation against the active knowledge base.</p></div></header>;
}

function CitationCard({ citation }: { citation: QueryCitation }) {
  const lines = citation.start_line == null ? "lines unavailable" : citation.end_line == null || citation.end_line === citation.start_line ? `L${citation.start_line}` : `L${citation.start_line}–${citation.end_line}`;
  return <div className="cc-citation-card"><div><Badge tone={citation.source_status === "source_changed" ? "warning" : citation.source_status === "source_deleted" ? "error" : "info"}>{citation.citation_id}</Badge><strong>{citation.relative_path}</strong></div><span>{citation.heading_path?.join(" / ") || "Root"} · {lines}</span>{citation.snippet && <p>{citation.snippet}</p>}</div>;
}


function ExecutionDrilldown({ trace }: { trace: import("../../types/runtimeTrace").RuntimeTraceV1 }) {
  const {actions}=useControlCenter();
  return <Panel title="Execution Drill-down" action={<Badge tone="info">same Runtime Trace</Badge>}><div className="cc-inline-actions cc-query-drilldown"><Button onClick={()=>actions.setActiveNav("runtime")}><Workflow size={14}/> Inspect Runtime</Button>{trace.graph_recovery.graph_activated&&<Button onClick={()=>actions.setActiveNav("graph")}><GitBranch size={14}/> Inspect Graph</Button>}<Button variant="primary" onClick={()=>actions.setActiveNav("evidence")}><FileCheck2 size={14}/> Inspect Evidence</Button></div></Panel>;
}

function DirectResult({ mode }: { mode: "search" | "ask" }) {
  const { state } = useShowcase();
  const trace = state.trace;
  if (state.lifecycle === "executing") return <Panel><EmptyState title="Executing" detail="Running the authoritative OPK-RAG pipeline. The UI does not synthesize intermediate state." /></Panel>;
  if (state.error || state.schemaError) return <Panel><EmptyState title="Execution failed" detail={state.schemaError || state.error || "Execution unavailable"} /></Panel>;
  if (!trace) return <Panel><EmptyState title={mode === "search" ? "Search the knowledge base" : "Ask with grounded evidence"} detail="Submit a query to start a real runtime execution." /></Panel>;
  if (mode === "search") return <div className="cc-query-result-stack"><ExecutionDrilldown trace={trace}/><EvidenceValidationSummary trace={trace} /><GuardRecoveryControlPlane trace={trace} compact/><RetrievalCandidateInspection trace={trace} compact/></div>;
  const answer = state.answerResult;
  const refused = trace.trace.status === "refused" || answer?.status === "refused";
  return <div className="cc-query-result-stack"><Panel title="Answer" action={<StatusBadge status={refused ? "refused" : state.lifecycle === "completed" ? "completed" : state.lifecycle === "partial" ? "partial" : "ready"} />}>
    {answer ? <><div className="cc-answer-text">{answer.answer || "No answer text returned."}</div>{answer.refusal_reason_code && <KeyValue label="Reason">{answer.refusal_reason_code}</KeyValue>}<KeyValue label="Grounding">{answer.grounding_valid === true ? "passed" : answer.grounding_valid === false ? "failed" : "Unavailable"}</KeyValue></> : <EmptyState title="Answer text unavailable" detail="This trace came from an older envelope without the bounded answer-result surface." />}
  </Panel><ExecutionDrilldown trace={trace}/>{answer?.citations?.length ? <Panel title={`Citations · ${answer.citations.length}`}><div className="cc-citation-list">{answer.citations.map(citation => <CitationCard key={citation.citation_id} citation={citation} />)}</div></Panel> : null}<EvidenceValidationSummary trace={trace} /><GuardRecoveryControlPlane trace={trace} compact /><RetrievalCandidateInspection trace={trace} compact /></div>;
}

function SessionRail({ sessions, selectedId, loading, error, onSelect, onCreate, onRefresh }: { sessions: ConversationSessionSummary[]; selectedId?: string; loading: boolean; error?: string; onSelect: (id: string) => void; onCreate: () => void; onRefresh: () => void }) {
  return <Panel title="Sessions" action={<div className="cc-inline-actions"><Button aria-label="Refresh sessions" onClick={onRefresh}><RefreshCw size={14} /></Button><Button aria-label="New session" variant="primary" onClick={onCreate}><MessageSquarePlus size={14} /></Button></div>} className="cc-session-panel">
    {loading && !sessions.length ? <EmptyState title="Loading sessions" detail="Reading persisted conversation metadata." /> : error && !sessions.length ? <EmptyState title="Sessions unavailable" detail={error} /> : <div className="cc-session-list">{sessions.map(session => <button className="cc-session-item" data-selected={session.id === selectedId} key={session.id} onClick={() => onSelect(session.id)}><strong>{session.title || `Conversation ${session.id.slice(0, 8)}`}</strong><span>{session.turn_count} turns · {session.status}</span></button>)}{!sessions.length && <EmptyState title="No conversations" detail="Create a session to start a persisted multi-turn conversation." />}</div>}
  </Panel>;
}

function ConversationView({ query, setQuery, client }: { query: string; setQuery: (value: string) => void; client: ShowcaseApiClient }) {
  const showcase = useShowcase();
  const control = useControlCenter();
  const [sessions, setSessions] = useState<ConversationSessionSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string>();
  const [detail, setDetail] = useState<ConversationSessionResponse>();
  const [loading, setLoading] = useState(true);
  const [executing, setExecuting] = useState(false);
  const [error, setError] = useState<string>();
  const [inspectedTraceId, setInspectedTraceId] = useState<string>();

  const refreshSessions = useCallback(async (preferred?: string) => {
    setLoading(true);
    const result = await client.listSessions();
    setLoading(false);
    if (!result.ok || !result.data) { setError(result.error?.message || "Conversation storage unavailable"); return; }
    setError(undefined);
    setSessions(result.data.sessions);
    const target = preferred || selectedId || result.data.sessions[0]?.id;
    if (target) {
      setSelectedId(target);
      const loaded = await client.getSession(target);
      if (loaded.ok && loaded.data) setDetail(loaded.data);
    } else setDetail(undefined);
  }, [selectedId]);

  useEffect(() => { void refreshSessions(); }, []); // persisted backend state is the authority; load once on entry

  const select = useCallback(async (id: string) => {
    setSelectedId(id); setLoading(true);
    const result = await client.getSession(id);
    setLoading(false);
    if (!result.ok || !result.data) { setError(result.error?.message || "Session unavailable"); return; }
    setError(undefined); setDetail(result.data);
  }, []);

  const create = useCallback(async () => {
    const result = await client.createSession();
    if (!result.ok || !result.data) { setError(result.error?.message || "Could not create session"); return; }
    setSelectedId(result.data.session.id); setDetail({ ...result.data, turns: [] });
    await refreshSessions(result.data.session.id);
  }, [refreshSessions]);

  const inspectTrace = useCallback(async (traceId: string) => {
    const result = await client.getTrace(traceId);
    if (!result.ok || !result.data) {
      setError("Trace unavailable. The runtime registry may have expired this historical trace.");
      setInspectedTraceId(undefined);
      return;
    }
    showcase.actions.acceptTraceEnvelope(result.data);
    setInspectedTraceId(traceId);
    setError(undefined);
    control.actions.setInspectorOpen(true);
    control.actions.setInspectorTab("trace");
  }, [client, showcase.actions, control.actions]);

  const submit = useCallback(async () => {
    const text = query.trim(); if (!text || executing) return;
    let sessionId = selectedId;
    if (!sessionId) {
      const created = await client.createSession();
      if (!created.ok || !created.data) { setError(created.error?.message || "Could not create session"); return; }
      sessionId = created.data.session.id; setSelectedId(sessionId);
    }
    setExecuting(true); setError(undefined);
    const clientRequestId = typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `ui-${Date.now()}`;
    const result = await client.askSession(sessionId, text, clientRequestId);
    setExecuting(false);
    if (!result.ok || !result.data) { setError(result.error?.message || "Conversation execution failed"); await refreshSessions(sessionId); return; }
    if (result.data.trace) showcase.actions.acceptTraceEnvelope(result.data.trace);
    setQuery("");
    const loaded = await client.getSession(sessionId);
    if (loaded.ok && loaded.data) setDetail(loaded.data);
    await refreshSessions(sessionId);
  }, [executing, query, refreshSessions, selectedId, setQuery, showcase.actions]);

  const turns = detail?.turns || [];
  const inspectedTrace = inspectedTraceId && showcase.state.trace?.trace?.trace_id === inspectedTraceId ? showcase.state.trace : undefined;
  return <div className="cc-conversation-stack"><div className="cc-conversation-layout"><SessionRail sessions={sessions} selectedId={selectedId} loading={loading} error={error} onSelect={select} onCreate={create} onRefresh={() => void refreshSessions()} /><Panel title={detail?.session.title || "Conversation"} action={detail?.session ? <Badge>{detail.session.turn_count} turns</Badge> : undefined} className="cc-conversation-panel"><div className="cc-conversation-thread">{turns.length ? turns.map(turn => <div className="cc-turn" key={turn.id}><div className="cc-turn-user"><span>User · #{turn.turn_number}</span><p>{turn.user_query}</p></div>{turn.is_followup && turn.standalone_query && turn.standalone_query !== turn.user_query && <div className="cc-query-rewrite"><span>Retrieval query</span><p>{turn.standalone_query}</p></div>}<div className="cc-turn-assistant" data-status={turn.turn_status}><div className="cc-turn-meta"><span>Assistant</span><StatusBadge status={turn.turn_status === "completed" ? "completed" : turn.turn_status === "abstained" ? "refused" : turn.turn_status === "failed" ? "failed" : "running"} label={turn.turn_status} /></div><p>{turn.answer_text || (turn.turn_status === "failed" ? "Execution failed safely." : "No answer text persisted.")}</p>{turn.abstention_reason && <span className="cc-muted">Reason: {turn.abstention_reason}</span>}{turn.citations?.length ? <div className="cc-citation-list">{turn.citations.map(citation => <CitationCard key={`${turn.id}-${citation.citation_id}`} citation={citation} />)}</div> : null}<div className="cc-result-meta"><span>{turn.retrieval_mode || "retrieval unavailable"}</span><span>{turn.latency_ms == null ? "latency unavailable" : `${turn.latency_ms} ms`}</span>{turn.trace_id ? <button className="cc-trace-link cc-mono" onClick={() => void inspectTrace(turn.trace_id!)}>trace {turn.trace_id.slice(0, 12)}…</button> : <span className="cc-mono">trace unavailable</span>}</div></div></div>) : <EmptyState title={loading ? "Loading conversation" : "Start a conversation"} detail="History is persisted by the backend and used only as bounded query context, never as knowledge-base evidence." />}</div>{error && <div className="cc-query-error">{error}</div>}<div className="cc-conversation-composer"><textarea aria-label="Conversation question" value={query} onChange={event => setQuery(event.target.value)} placeholder="Ask a follow-up or start a new topic…" rows={3} /><Button variant="primary" disabled={executing || !query.trim()} onClick={() => void submit()}><Send size={15} />{executing ? "Running…" : "Send"}</Button></div></Panel></div>{inspectedTrace && <><ExecutionDrilldown trace={inspectedTrace}/><EvidenceValidationSummary trace={inspectedTrace} compact={false} /><GuardRecoveryControlPlane trace={inspectedTrace} /><RetrievalCandidateInspection trace={inspectedTrace} /></>}</div>;
}

export function QueryWorkspacePage({ client = showcaseApi }: { client?: ShowcaseApiClient }) {
  const showcase = useShowcase();
  const runtime = useRuntimeStatus();
  const { state: control } = useControlCenter();
  const [mode, setMode] = useState<WorkspaceMode>("ask");
  const [query, setQuery] = useState("");
  const ready = mode === "search" ? runtime.state.status?.health.search_ready : runtime.state.status?.health.ask_ready;
  const running = showcase.state.lifecycle === "executing";
  const modeDetail = useMemo(() => mode === "search" ? "Retrieve and rank authoritative candidates without generation." : mode === "ask" ? "Generate a grounded answer with authoritative citations." : "Use persisted bounded context for multi-turn follow-ups.", [mode]);

  const runDirect = useCallback(async () => {
    if (mode === "conversation" || !query.trim()) return;
    await showcase.actions.runQueryInput(query, mode);
  }, [mode, query, showcase.actions]);

  return <div className="cc-page" data-query-workspace="active" data-presentation-mode={control.mode}><Heading /><div className="cc-query-toolbar"><div className="cc-segmented" role="group" aria-label="Query mode"><button aria-pressed={mode === "search"} onClick={() => setMode("search")}><Search size={13} />Search</button><button aria-pressed={mode === "ask"} onClick={() => setMode("ask")}><Sparkles size={13} />Ask</button><button aria-pressed={mode === "conversation"} onClick={() => setMode("conversation")}><MessageSquarePlus size={13} />Conversation</button></div><span className="cc-muted">{modeDetail}</span></div>{mode === "conversation" ? <ConversationView query={query} setQuery={setQuery} client={client} /> : <><Panel title="Query Composer" action={<StatusBadge status={ready === true ? "ready" : runtime.state.loading ? "checking" : "unavailable"} label={ready === true ? `${mode} ready` : runtime.state.loading ? "checking runtime" : `${mode} unavailable`} />}><div className="cc-query-composer"><textarea aria-label="Query input" value={query} onChange={event => setQuery(event.target.value)} placeholder={mode === "search" ? "Search your knowledge base…" : "Ask a grounded question…"} rows={4} /><Button variant="primary" disabled={running || !query.trim() || ready !== true} onClick={() => void runDirect()}><Play size={15} />{running ? "Running…" : mode === "search" ? "Search" : "Ask"}</Button></div></Panel><DirectResult mode={mode} /></>}</div>;
}
