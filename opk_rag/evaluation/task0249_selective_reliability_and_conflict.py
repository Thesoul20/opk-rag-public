from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from opk_rag.agentic_v2.base import stable_digest
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.evaluation.task0248_selective_runtime import AmbiguityRewriteDecision, AbstentionVetoDecision


class StrictEvalContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AnswerabilityConflictSignals(StrictEvalContract):
    contract_version: Literal["opk-rag.task0249.answerability-conflict-signals.v1"] = "opk-rag.task0249.answerability-conflict-signals.v1"
    exact_fact_required: bool = False
    exact_fact_support_present: bool = False
    comparison_required: bool = False
    comparison_support_present: bool = False
    linked_target_required: bool = False
    linked_target_status: Literal["not_required", "resolved", "unresolved"] = "not_required"
    explicit_graph_relation_required: bool = False
    explicit_graph_relation_supported: bool = False
    partial_answerability_conflict: bool = False
    weak_evidence_conflict: bool = False
    recovery_no_gain_conflict: bool = False
    reason_codes: tuple[str, ...] = ()


class ConflictGateV2Decision(StrictEvalContract):
    contract_version: Literal["opk-rag.task0249.conflict-gate-v2.v1"] = "opk-rag.task0249.conflict-gate-v2.v1"
    invoke_veto: bool
    reason_codes: tuple[str, ...] = ()


class ConflictSignalDetector:
    """Deterministic evaluation-only conflict detector. No evaluator Gold/sample id input."""

    _version_re = re.compile(r"(?<!\d)[vV]?\d+\.\d+(?:\.\d+)?(?:[-+._a-zA-Z0-9]*)?")
    _numeric_re = re.compile(r"(?<!\w)\d+(?:\.\d+)?\s*(?:ms|s|秒|毫秒|%|MB|GB|GiB|MiB|QPS|元|美元|USD|CNY|PHP)?", re.I)
    _exact_markers = ("版本号", "版本是多少", "最新版本", "端口", "地址", "价格", "订阅价格", "数值", "QPS", "TTL", "路径")
    _comparison_markers = ("相比", "比较", "哪个更", "谁更", "更快", "更慢", "更高", "更低", "更好", "更差")
    _comparison_support_markers = ("相比", "比较", "更快", "更慢", "高于", "低于", "分别为", "vs", "versus", "延迟", "耗时")
    _graph_markers = ("显式关系", "显式边", "是否链接", "关系图", "G0-G2", "可遍历")

    @staticmethod
    def _evidence_text(excerpts: tuple[str, ...]) -> str:
        return "\n".join(excerpts)

    @staticmethod
    def _extract_link_target(query: str) -> str | None:
        # Generic target phrases used in note-link questions; never uses benchmark IDs or Gold.
        pats = (
            r"链接(?:到)?的([^，。？?]+?)(?:笔记|中|给出|包含|展示|如何|怎样)",
            r"链接(?:到)?([^，。？?]+?)(?:笔记|中|给出|包含)",
        )
        for pat in pats:
            m = re.search(pat, query)
            if m:
                target = m.group(1).strip(" `[]")
                if target:
                    return target[:120]
        m = re.search(r"([\w.-]+\.(?:png|jpg|jpeg|webp|md|docx))", query, re.I)
        return m.group(1) if m else None

    def detect(self, *, query: str, evidence_excerpts: tuple[str, ...], answerability_status: str,
               top_rerank_score: float | None, graph_relation_available: bool,
               recovery_attempted: bool, recovery_improved_evidence: bool) -> AnswerabilityConflictSignals:
        text = self._evidence_text(evidence_excerpts)
        qlow, tlow = query.lower(), text.lower()
        reasons: list[str] = []

        exact = any(x.lower() in qlow for x in self._exact_markers)
        exact_support = False
        if exact:
            if any(x in qlow for x in ("版本", "版本号")):
                exact_support = bool(self._version_re.search(text))
            elif "路径" in query:
                exact_support = bool(re.search(r"(?:^|[\s`])(?:~?/|%[A-Z_]+%\\|[A-Za-z]:\\)", text))
            else:
                exact_support = bool(self._numeric_re.search(text))
            if not exact_support:
                reasons.append("exact_fact_support_missing")

        comparison = any(x.lower() in qlow for x in self._comparison_markers)
        comparison_support = comparison and any(x.lower() in tlow for x in self._comparison_support_markers)
        if comparison and not comparison_support:
            reasons.append("comparison_support_missing")

        target = self._extract_link_target(query)
        linked_required = target is not None or ("链接的" in query or "链接到的" in query)
        linked_status: Literal["not_required", "resolved", "unresolved"] = "not_required"
        if linked_required:
            linked_status = "unresolved"
            if target:
                t = target.lower().replace("笔记", "").strip()
                source_lines = [x.split("; evidence=", 1)[0].lower() for x in evidence_excerpts]
                # Resolved means the actual target is an evidence source, not merely mentioned by another note.
                if t and any(t in s for s in source_lines):
                    linked_status = "resolved"
            # Image target requires the actual image target as evidence source; a syntax mention is not resolution.
            if target and re.search(r"\.(?:png|jpg|jpeg|webp)$", target, re.I):
                linked_status = "resolved" if any(target.lower() in x.split("; evidence=",1)[0].lower() for x in evidence_excerpts) else "unresolved"
            if linked_status == "unresolved":
                reasons.append("required_link_target_unresolved")

        graph_required = any(x.lower() in qlow for x in self._graph_markers)
        graph_supported = graph_required and graph_relation_available
        if graph_required and not graph_supported:
            reasons.append("explicit_graph_relation_support_missing")

        partial = answerability_status == "partially_answerable" and (top_rerank_score is None or top_rerank_score < 0.35)
        if partial:
            reasons.append("partial_answerability_conflict")
        weak = (top_rerank_score is None or top_rerank_score < 0.35) and answerability_status == "answerable"
        if weak and not any(r in reasons for r in ("exact_fact_support_missing", "comparison_support_missing", "explicit_graph_relation_support_missing")):
            reasons.append("weak_evidence_conflict")
        recovery_no_gain = recovery_attempted and not recovery_improved_evidence and answerability_status != "answerable" and (top_rerank_score is None or top_rerank_score < 0.35)
        if recovery_no_gain:
            reasons.append("recovery_no_gain_conflict")

        return AnswerabilityConflictSignals(
            exact_fact_required=exact, exact_fact_support_present=exact_support,
            comparison_required=comparison, comparison_support_present=comparison_support,
            linked_target_required=linked_required, linked_target_status=linked_status,
            explicit_graph_relation_required=graph_required, explicit_graph_relation_supported=graph_supported,
            partial_answerability_conflict=partial, weak_evidence_conflict=weak,
            recovery_no_gain_conflict=recovery_no_gain, reason_codes=tuple(dict.fromkeys(reasons)),
        )


class ConflictGateV2:
    def decide(self, *, terminal: str, signals: AnswerabilityConflictSignals) -> ConflictGateV2Decision:
        if terminal != "finished":
            return ConflictGateV2Decision(invoke_veto=False, reason_codes=())
        authorized = tuple(r for r in signals.reason_codes if r in {
            "exact_fact_support_missing", "comparison_support_missing", "required_link_target_unresolved",
            "explicit_graph_relation_support_missing", "partial_answerability_conflict",
            "weak_evidence_conflict", "recovery_no_gain_conflict",
        })
        return ConflictGateV2Decision(invoke_veto=bool(authorized), reason_codes=authorized)


@dataclass(frozen=True)
class TransportResult:
    raw_output: str
    error_code: str | None
    latency_ms: int
    input_tokens: int
    output_tokens: int
    response_received: bool
    mode: str
    request_count: int = 1


class EvaluationStructuredTransport:
    """Evaluation-only OpenAI-compatible transport capability probe; production provider is untouched."""
    def __init__(self, *, mode: Literal["json_object", "json_schema", "function_call"], timeout_seconds: float = 60.0, max_tokens: int = 320):
        load_project_env()
        self.mode = mode
        self.base_url = (os.getenv("OPK_RAG_AGENT_POLICY_BASE_URL") or "").rstrip("/")
        self.api_key = os.getenv("OPK_RAG_AGENT_POLICY_API_KEY")
        self.model = os.getenv("OPK_RAG_AGENT_POLICY_MODEL") or "unknown-policy-model"
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def call(self, *, prompt: str, schema: dict[str, Any], function_name: str) -> TransportResult:
        started=time.monotonic()
        if not self.base_url or not self.api_key:
            return TransportResult("", "provider_not_configured", 0, 0, 0, False, self.mode)
        payload: dict[str, Any] = {"model":self.model,"temperature":0,"max_tokens":self.max_tokens,"messages":[{"role":"system","content":prompt}],"thinking":{"type":"disabled"}}
        if self.mode == "json_object":
            payload["response_format"]={"type":"json_object"}
        elif self.mode == "json_schema":
            payload["response_format"]={"type":"json_schema","json_schema":{"name":function_name,"strict":True,"schema":schema}}
        else:
            payload["tools"]=[{"type":"function","function":{"name":function_name,"description":"Return the bounded OPK-RAG controller decision.","parameters":schema}}]
            payload["tool_choice"]={"type":"function","function":{"name":function_name}}
        request=urllib.request.Request(f"{self.base_url}/chat/completions",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json","Authorization":f"Bearer {self.api_key}"},method="POST")
        try:
            with urllib.request.urlopen(request,timeout=self.timeout_seconds) as resp:
                body=json.loads(resp.read().decode())
        except TimeoutError:
            return TransportResult("","provider_timeout",int((time.monotonic()-started)*1000),0,0,False,self.mode)
        except urllib.error.HTTPError as exc:
            return TransportResult("",f"provider_http_{exc.code}",int((time.monotonic()-started)*1000),0,0,False,self.mode)
        except (urllib.error.URLError,json.JSONDecodeError) as exc:
            return TransportResult("",f"provider_{type(exc).__name__.lower()}",int((time.monotonic()-started)*1000),0,0,False,self.mode)
        choice=(body.get("choices") or [{}])[0]; msg=choice.get("message") or {}; usage=body.get("usage") or {}
        if self.mode == "function_call":
            calls=msg.get("tool_calls") or []
            raw=((calls[0].get("function") or {}).get("arguments") if calls else "") or ""
        else:
            raw=msg.get("content") or ""
        return TransportResult(raw,None,int((time.monotonic()-started)*1000),int(usage.get("prompt_tokens") or 0),int(usage.get("completion_tokens") or 0),True,self.mode)


T=TypeVar("T",bound=BaseModel)
@dataclass(frozen=True)
class ReliabilityDecision:
    valid: bool
    decision: BaseModel | None
    failure_code: str | None
    provider_requests: int
    transport_retry_count: int
    structural_repair_count: int
    latency_ms: int
    input_tokens: int
    output_tokens: int
    response_count: int


class BoundedReliableNarrowPolicy:
    """At most one transport retry total and one structural repair; finite max 3 provider requests."""
    def __init__(self, *, transport: EvaluationStructuredTransport, max_transport_retries: int = 0, max_structural_repairs: int = 1):
        if max_transport_retries not in (0,1) or max_structural_repairs not in (0,1):
            raise ValueError("TASK-0249 budgets are frozen at <=1 transport retry and <=1 structural repair")
        self.transport=transport; self.max_transport_retries=max_transport_retries; self.max_structural_repairs=max_structural_repairs

    def decide(self, *, prompt: str, model_type: type[T], function_name: str) -> ReliabilityDecision:
        schema=model_type.model_json_schema(); req=retry=repair=lat=it=ot=responses=0; last_failure=None; previous_digest="none"
        semantic_attempt=0
        while semantic_attempt <= self.max_structural_repairs:
            current=prompt if semantic_attempt==0 else prompt+f"\nSTRUCTURAL_REPAIR: previous_digest={previous_digest}; failure={last_failure}. Return exactly one corrected structured decision."
            while True:
                tr=self.transport.call(prompt=current,schema=schema,function_name=function_name); req+=1; lat+=tr.latency_ms; it+=tr.input_tokens; ot+=tr.output_tokens
                if tr.response_received: responses+=1
                if tr.error_code and retry < self.max_transport_retries:
                    retry+=1
                    continue
                break
            if tr.error_code:
                return ReliabilityDecision(False,None,tr.error_code,req,retry,repair,lat,it,ot,responses)
            previous_digest=stable_digest(tr.raw_output)
            try:
                payload=json.loads(tr.raw_output)
                if not isinstance(payload,dict): raise ValueError("not_object")
                decision=model_type.model_validate(payload)
            except json.JSONDecodeError:
                last_failure="invalid_json"
            except ValidationError:
                last_failure="schema_validation_failed"
            except ValueError:
                last_failure="not_json_object"
            else:
                return ReliabilityDecision(True,decision,None,req,retry,repair,lat,it,ot,responses)
            if semantic_attempt >= self.max_structural_repairs:
                return ReliabilityDecision(False,None,last_failure,req,retry,repair,lat,it,ot,responses)
            repair+=1; semantic_attempt+=1
        raise AssertionError("unreachable")


def rewrite_prompt(query:str,context:str)->str:
    schema=AmbiguityRewriteDecision.model_json_schema()
    return "\n".join(["OPK-RAG bounded rewrite policy. Do not answer. Resolve only from context. Return exactly one JSON structured output object only.",f"QUERY={query}",f"CONTEXT={context[:1000]}",f"SCHEMA={json.dumps(schema,ensure_ascii=False,sort_keys=True)}"])


def veto_prompt(*,query:str,signals:AnswerabilityConflictSignals,evidence_excerpts:tuple[str,...])->str:
    schema=AbstentionVetoDecision.model_json_schema()
    state={"query":query,"conflict_signals":signals.model_dump(mode="json"),"evidence_excerpts":list(evidence_excerpts[:3]),"authority":"keep_finish_or_abstain_only"}
    return "\n".join(["OPK-RAG bounded abstention veto. Never answer and never authorize Finish beyond keep_finish. Use only supplied evidence. Return exactly one JSON structured output object only.","decision must be exactly keep_finish or abstain.","reason_code must be exactly one of: direct_support_present, exact_fact_not_supported, evidence_conflict, recovery_no_gain, insufficient_direct_support.","public_summary must be concise and no longer than 120 characters.",f"STATE={json.dumps(state,ensure_ascii=False,sort_keys=True)}",f"SCHEMA={json.dumps(schema,ensure_ascii=False,sort_keys=True)}"])
