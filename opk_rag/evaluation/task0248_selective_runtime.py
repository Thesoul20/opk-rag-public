from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from opk_rag.agent.policy_provider import OpenAICompatiblePolicyProvider, PolicyProviderResult
from opk_rag.agentic_v2.base import stable_digest, stable_json


class EvalStrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QueryAmbiguityDecision(EvalStrictContract):
    contract_version: Literal["opk-rag.task0248.query-ambiguity.v1"] = "opk-rag.task0248.query-ambiguity.v1"
    ambiguous: bool
    reason_code: Literal["explicit_reference", "omitted_subject", "context_dependent_shorthand", "clear_standalone_query"]
    detected_reference_terms: tuple[str, ...] = ()
    rewrite_allowed: bool
    runtime_context_available: bool


class AmbiguityRewriteDecision(EvalStrictContract):
    contract_version: Literal["opk-rag.task0248.ambiguity-rewrite.v1"] = "opk-rag.task0248.ambiguity-rewrite.v1"
    rewritten_query: str = Field(min_length=1, max_length=512)
    reason_code: Literal["resolved_reference", "expanded_omitted_subject", "normalized_shorthand"]
    confidence: float = Field(ge=0.0, le=1.0)
    public_summary: str = Field(min_length=1, max_length=160)


class AbstentionVetoInvocationDecision(EvalStrictContract):
    contract_version: Literal["opk-rag.task0248.abstention-veto-invocation.v1"] = "opk-rag.task0248.abstention-veto-invocation.v1"
    invoke_veto: bool
    reason_code: Literal["partial_answerability_conflict", "weak_exact_fact_support", "recovery_no_gain", "clear_finish", "terminal_not_finish"]
    conflict_signals: tuple[str, ...] = ()
    answerability_status: str
    recovery_attempted: bool
    recovery_improved_evidence: bool


class AbstentionVetoDecision(EvalStrictContract):
    contract_version: Literal["opk-rag.task0248.abstention-veto.v1"] = "opk-rag.task0248.abstention-veto.v1"
    decision: Literal["keep_finish", "abstain"]
    reason_code: Literal["direct_support_present", "exact_fact_not_supported", "evidence_conflict", "recovery_no_gain", "insufficient_direct_support"]
    confidence: float = Field(ge=0.0, le=1.0)
    public_summary: str = Field(min_length=1, max_length=160)


class QueryAmbiguityDetector:
    """Evaluation-only deterministic ambiguity detector. Evaluator Gold is not an input."""

    _markers = (
        "之前那个", "前面那个", "上面那个", "后来那个", "那个替代", "这个方案", "这个在",
        "那里面", "它们", "它呢", "这个呢", "那个呢", "那 Linux", "那之前", "刚才那个",
        "这边具体", "后来怎么样", "这个问题", "那几个", "其中那个",
    )

    def decide(self, *, query: str, context: str | None = None) -> QueryAmbiguityDecision:
        found = tuple(marker for marker in self._markers if marker.lower() in query.lower())
        ambiguous = bool(found)
        if not ambiguous and len(query.strip()) <= 14 and context:
            omitted = any(token in query for token in ("怎么", "哪里", "哪个", "是什么", "呢", "多少", "负责"))
            if omitted:
                return QueryAmbiguityDecision(ambiguous=True, reason_code="omitted_subject", detected_reference_terms=(), rewrite_allowed=True, runtime_context_available=True)
        if ambiguous:
            reason = "context_dependent_shorthand" if any(x in query for x in ("它们", "那里面", "这个在")) else "explicit_reference"
            return QueryAmbiguityDecision(ambiguous=True, reason_code=reason, detected_reference_terms=found, rewrite_allowed=bool(context), runtime_context_available=bool(context))
        return QueryAmbiguityDecision(ambiguous=False, reason_code="clear_standalone_query", detected_reference_terms=(), rewrite_allowed=False, runtime_context_available=bool(context))


class AbstentionVetoGate:
    """One-way safety gate: only candidate Finish states can reach the veto Policy."""

    _exact_fact_markers = ("地址", "端口", "版本号", "数值", "QPS", "价格", "订阅", "API Server", "TTL", "precision")

    def decide(
        self,
        *,
        query: str,
        terminal: str,
        answerability_status: str,
        top_rerank_score: float | None,
        recovery_attempted: bool,
        recovery_improved_evidence: bool,
    ) -> AbstentionVetoInvocationDecision:
        if terminal != "finished":
            return AbstentionVetoInvocationDecision(invoke_veto=False, reason_code="terminal_not_finish", answerability_status=answerability_status, recovery_attempted=recovery_attempted, recovery_improved_evidence=recovery_improved_evidence)
        signals: list[str] = []
        if answerability_status == "partially_answerable":
            signals.append("answerability=partially_answerable")
            return AbstentionVetoInvocationDecision(invoke_veto=True, reason_code="partial_answerability_conflict", conflict_signals=tuple(signals), answerability_status=answerability_status, recovery_attempted=recovery_attempted, recovery_improved_evidence=recovery_improved_evidence)
        exact = any(marker.lower() in query.lower() for marker in self._exact_fact_markers)
        weak = top_rerank_score is None or top_rerank_score < 0.35
        if exact and weak:
            signals.extend(("exact_fact_query=true", f"top_rerank_score={top_rerank_score}"))
            return AbstentionVetoInvocationDecision(invoke_veto=True, reason_code="weak_exact_fact_support", conflict_signals=tuple(signals), answerability_status=answerability_status, recovery_attempted=recovery_attempted, recovery_improved_evidence=recovery_improved_evidence)
        if recovery_attempted and not recovery_improved_evidence and answerability_status != "answerable":
            signals.append("recovery_improved_evidence=false")
            return AbstentionVetoInvocationDecision(invoke_veto=True, reason_code="recovery_no_gain", conflict_signals=tuple(signals), answerability_status=answerability_status, recovery_attempted=True, recovery_improved_evidence=False)
        return AbstentionVetoInvocationDecision(invoke_veto=False, reason_code="clear_finish", conflict_signals=(), answerability_status=answerability_status, recovery_attempted=recovery_attempted, recovery_improved_evidence=recovery_improved_evidence)


@dataclass(frozen=True)
class NarrowPolicyExecution:
    valid: bool
    decision: BaseModel | None
    failure_code: str | None
    attempts: int
    latency_ms: int
    input_tokens: int
    output_tokens: int
    raw_output_persisted: bool = False


T = TypeVar("T", bound=BaseModel)


class NarrowStructuredPolicy:
    """Evaluation-only typed Policy wrapper over the frozen OpenAI-compatible transport."""

    def __init__(self, provider: OpenAICompatiblePolicyProvider | None = None) -> None:
        self.provider = provider or OpenAICompatiblePolicyProvider(timeout_seconds=60, temperature=0.0, thinking_mode="disabled", json_output_enabled=True, max_tokens=384)

    def decide(self, *, prompt: str, state_view: dict[str, Any], model_type: type[T]) -> NarrowPolicyExecution:
        schema = model_type.model_json_schema()
        total_latency = total_in = total_out = 0
        failure_code: str | None = None
        previous_digest = "none"
        for attempt in range(2):
            current_prompt = prompt if attempt == 0 else "\n".join((prompt, "STRUCTURAL_REPAIR:", f"previous_output_digest={previous_digest}", f"failure_code={failure_code}", "Return exactly one corrected JSON object matching the schema. No prose or markdown."))
            result = self.provider.decide(prompt=current_prompt, state_view=state_view, output_schema=schema, repair_context=None if attempt == 0 else {"failure_code": failure_code, "previous_output_digest": previous_digest})
            total_latency += int(result.latency_ms or 0); total_in += int(result.input_tokens or 0); total_out += int(result.output_tokens or 0)
            if result.error_code:
                return NarrowPolicyExecution(False, None, result.error_code, attempt + 1, total_latency, total_in, total_out)
            previous_digest = stable_digest(result.raw_output)
            try:
                payload = json.loads(result.raw_output)
                if not isinstance(payload, dict): raise ValueError("not_object")
                decision = model_type.model_validate(payload)
            except (json.JSONDecodeError, ValidationError, ValueError):
                failure_code = "narrow_policy_contract_invalid"
                continue
            return NarrowPolicyExecution(True, decision, None, attempt + 1, total_latency, total_in, total_out)
        return NarrowPolicyExecution(False, None, "narrow_policy_structural_repair_exhausted", 2, total_latency, total_in, total_out)

    def rewrite(self, *, query: str, context: str) -> NarrowPolicyExecution:
        state = {"current_query": query, "bounded_context": context[:1200], "authority": "rewrite_only"}
        prompt = "\n".join((
            "You are OPK-RAG's bounded ambiguity rewrite policy.",
            "Resolve the current query using only BOUNDED_CONTEXT. Do not answer the question. Do not select tools. Do not invent facts.",
            "Return one JSON object only matching OUTPUT_SCHEMA. rewritten_query must be standalone and <=512 characters.",
            "CURRENT_QUERY:", query,
            "BOUNDED_CONTEXT:", context[:1200],
            "OUTPUT_SCHEMA:", stable_json(AmbiguityRewriteDecision.model_json_schema()),
        ))
        return self.decide(prompt=prompt, state_view=state, model_type=AmbiguityRewriteDecision)

    def veto(self, *, query: str, answerability_status: str, top_rerank_score: float | None, evidence_excerpts: tuple[str, ...], recovery_attempted: bool, recovery_improved_evidence: bool) -> NarrowPolicyExecution:
        state = {
            "query": query,
            "answerability_status": answerability_status,
            "top_rerank_score": top_rerank_score,
            "evidence_excerpts": list(evidence_excerpts[:3]),
            "recovery_attempted": recovery_attempted,
            "recovery_improved_evidence": recovery_improved_evidence,
            "authority": "keep_finish_or_abstain_only",
        }
        prompt = "\n".join((
            "You are OPK-RAG's bounded abstention-veto policy.",
            "A deterministic runtime currently proposes Finish. You may ONLY keep that Finish or reduce authority to Abstain.",
            "Never answer the user's question. Never create or select tools. Never infer unsupported exact values.",
            "Choose abstain when the provided evidence excerpts do not directly support the requested exact fact or when support is materially conflicting/partial.",
            "Choose keep_finish only when the excerpts directly support the requested answer scope.",
            "Return one JSON object only matching OUTPUT_SCHEMA.",
            "STATE:", stable_json(state),
            "OUTPUT_SCHEMA:", stable_json(AbstentionVetoDecision.model_json_schema()),
        ))
        return self.decide(prompt=prompt, state_view=state, model_type=AbstentionVetoDecision)


def compact_evidence_excerpts(search_response: Any, *, max_items: int = 3, max_chars: int = 520) -> tuple[str, ...]:
    bundle = getattr(search_response, "evidence_bundle", None)
    items = tuple(getattr(bundle, "items", ()) or ())[:max_items]
    excerpts=[]
    for item in items:
        path=str(getattr(item,"relative_path", ""))
        content=" ".join(str(getattr(item,"content", "")).split())[:max_chars]
        excerpts.append(f"source={path}; evidence={content}")
    return tuple(excerpts)
