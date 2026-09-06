from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from opk_rag.agentic_v2.base import stable_digest
from opk_rag.evaluation.task0248_selective_runtime import AbstentionVetoDecision
from opk_rag.evaluation.task0249_selective_reliability_and_conflict import (
    EvaluationStructuredTransport,
    ReliabilityDecision,
    StrictEvalContract,
)


class AnswerabilityConflictSignalsV3(StrictEvalContract):
    contract_version: Literal["opk-rag.task0258.answerability-conflict-signals.v3"] = "opk-rag.task0258.answerability-conflict-signals.v3"
    exact_fact_required: bool = False
    exact_fact_support_present: bool = False
    exact_metric: str | None = None
    comparison_required: bool = False
    comparison_kind: Literal["none", "qualitative", "quantitative_or_directional"] = "none"
    comparison_support_present: bool = False
    explicit_link_intent: bool = False
    linked_target: str | None = None
    linked_target_status: Literal["not_required", "resolved", "unresolved"] = "not_required"
    explicit_graph_relation_required: bool = False
    explicit_graph_relation_supported: bool = False
    partial_answerability_conflict: bool = False
    severe_partial_evidence_conflict: bool = False
    weak_evidence_conflict: bool = False
    recovery_no_gain_conflict: bool = False
    recovery_improved_evidence: bool = False
    reason_codes: tuple[str, ...] = ()


class ConflictGateV3Decision(StrictEvalContract):
    contract_version: Literal["opk-rag.task0258.conflict-gate-v3.v1"] = "opk-rag.task0258.conflict-gate-v3.v1"
    invoke_veto: bool
    strong_reason_codes: tuple[str, ...] = ()
    contextual_reason_codes: tuple[str, ...] = ()

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return self.strong_reason_codes + self.contextual_reason_codes


class ConflictSignalDetectorV3:
    """Generic conflict detector repaired from TASK-0257; no sample IDs or Gold inputs."""

    _version_re = re.compile(r"(?<!\d)[vV]?\d+\.\d+(?:\.\d+)?(?:[-+._a-zA-Z0-9]*)?")
    _numeric_re = re.compile(r"(?<!\w)\d+(?:\.\d+)?\s*(?:ms|s|秒|毫秒|%|MB|GB|GiB|MiB|QPS|元|美元|USD|CNY|PHP|分钟|倍)?", re.I)
    _exact_markers = ("版本号", "版本是多少", "最新版本", "端口", "地址", "价格", "订阅价格", "数值", "qps", "ttl", "路径", "多少分钟", "多少毫秒", "多少倍")
    _metric_markers = ("precision", "recall", "f1", "accuracy", "qps", "mrr", "p50", "p95", "p99", "准确率", "召回率", "精确率")
    _comparison_markers = ("相比", "比较", "区别", "差异", "哪个更", "谁更", "更快", "更慢", "更高", "更低", "更好", "更差")
    _directional_markers = ("哪个更", "谁更", "更快", "更慢", "更高", "更低", "更好", "更差", "差多少", "高多少", "低多少", "快多少", "慢多少", "百分比", "几倍", "多少倍")
    _comparison_support_markers = ("相比", "比较", "更快", "更慢", "高于", "低于", "分别为", "vs", "versus", "延迟", "耗时", "倍")
    _graph_markers = ("显式关系", "显式边", "是否链接", "关系图", "g0-g2", "可遍历")
    _link_intent_markers = ("链接的", "链接到", "链接至", "指向的", "指向", "关联到", "引用的笔记", "链接目标", "link to")

    @staticmethod
    def _evidence_text(excerpts: tuple[str, ...]) -> str:
        return "\n".join(excerpts)

    @staticmethod
    def _extract_link_target(query: str) -> str | None:
        patterns = (
            r"链接(?:到|至)?的?([^，。？?]+?)(?:笔记|中|给出|包含|展示|如何|怎样|页面)",
            r"指向的?([^，。？?]+?)(?:笔记|中|给出|包含|页面)",
            r"关联到的?([^，。？?]+?)(?:笔记|中|给出|包含)",
        )
        for pattern in patterns:
            match = re.search(pattern, query, re.I)
            if match:
                target = match.group(1).strip(" `[]")
                if target:
                    return target[:120]
        match = re.search(r"([\w.-]+\.(?:png|jpg|jpeg|webp|md|docx))", query, re.I)
        return match.group(1) if match else None

    @classmethod
    def _metric_from_query(cls, query: str) -> str | None:
        q = query.lower()
        for metric in cls._metric_markers:
            if metric in q:
                return metric
        return None

    @staticmethod
    def _metric_supported(metric: str, text: str) -> bool:
        pattern = re.compile(rf"{re.escape(metric)}[^\n]{{0,48}}\d|\d[^\n]{{0,48}}{re.escape(metric)}", re.I)
        return bool(pattern.search(text))

    def detect(
        self,
        *,
        query: str,
        evidence_excerpts: tuple[str, ...],
        answerability_status: str,
        top_rerank_score: float | None,
        graph_relation_available: bool,
        recovery_attempted: bool,
        recovery_improved_evidence: bool,
    ) -> AnswerabilityConflictSignalsV3:
        text = self._evidence_text(evidence_excerpts)
        qlow = query.lower()
        tlow = text.lower()
        reasons: list[str] = []

        metric = self._metric_from_query(query)
        exact = metric is not None or any(marker.lower() in qlow for marker in self._exact_markers)
        exact_support = False
        if exact:
            if metric is not None:
                exact_support = self._metric_supported(metric, tlow)
            elif "版本" in qlow:
                exact_support = bool(self._version_re.search(text))
            elif "路径" in query:
                exact_support = bool(re.search(r"(?:^|[\s`])(?:~?/|%[A-Z_]+%\\|[A-Za-z]:\\)", text))
            else:
                exact_support = bool(self._numeric_re.search(text))
            if not exact_support:
                reasons.append("exact_fact_support_missing")

        comparison = any(marker.lower() in qlow for marker in self._comparison_markers)
        directional = any(marker.lower() in qlow for marker in self._directional_markers)
        comparison_kind: Literal["none", "qualitative", "quantitative_or_directional"] = "none"
        comparison_support = False
        if comparison:
            comparison_kind = "quantitative_or_directional" if directional else "qualitative"
            if comparison_kind == "qualitative":
                # Qualitative comparison can be composed from separately supported alternatives.
                comparison_support = bool(text.strip())
            else:
                comparison_support = any(marker.lower() in tlow for marker in self._comparison_support_markers)
                if not comparison_support:
                    reasons.append("comparison_support_missing")

        explicit_link_intent = any(marker.lower() in qlow for marker in self._link_intent_markers)
        target = self._extract_link_target(query) if explicit_link_intent else None
        linked_status: Literal["not_required", "resolved", "unresolved"] = "not_required"
        if explicit_link_intent:
            linked_status = "unresolved"
            if target:
                normalized = target.lower().replace("笔记", "").strip()
                sources = [part.split("; evidence=", 1)[0].lower() for part in evidence_excerpts]
                if normalized and any(normalized in source for source in sources):
                    linked_status = "resolved"
            if linked_status == "unresolved":
                reasons.append("required_link_target_unresolved")

        graph_required = any(marker.lower() in qlow for marker in self._graph_markers)
        graph_supported = graph_required and graph_relation_available
        if graph_required and not graph_supported:
            reasons.append("explicit_graph_relation_support_missing")

        score = None if top_rerank_score is None else float(top_rerank_score)
        partial = answerability_status == "partially_answerable" and (score is None or score < 0.35) and not recovery_improved_evidence
        severe_partial = partial and (score is None or score < 0.10)
        if partial:
            reasons.append("partial_answerability_conflict")
        if severe_partial:
            reasons.append("severe_partial_evidence_conflict")
        weak = answerability_status == "answerable" and (score is None or score < 0.35)
        if weak and not any(r in reasons for r in ("exact_fact_support_missing", "comparison_support_missing", "explicit_graph_relation_support_missing")):
            reasons.append("weak_evidence_conflict")
        recovery_no_gain = recovery_attempted and not recovery_improved_evidence and answerability_status != "answerable" and (score is None or score < 0.35)
        if recovery_no_gain:
            reasons.append("recovery_no_gain_conflict")

        return AnswerabilityConflictSignalsV3(
            exact_fact_required=exact,
            exact_fact_support_present=exact_support,
            exact_metric=metric,
            comparison_required=comparison,
            comparison_kind=comparison_kind,
            comparison_support_present=comparison_support,
            explicit_link_intent=explicit_link_intent,
            linked_target=target,
            linked_target_status=linked_status,
            explicit_graph_relation_required=graph_required,
            explicit_graph_relation_supported=graph_supported,
            partial_answerability_conflict=partial,
            severe_partial_evidence_conflict=severe_partial,
            weak_evidence_conflict=weak,
            recovery_no_gain_conflict=recovery_no_gain,
            recovery_improved_evidence=recovery_improved_evidence,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )


class ConflictGateV3:
    _strong = {
        "exact_fact_support_missing",
        "comparison_support_missing",
        "required_link_target_unresolved",
        "explicit_graph_relation_support_missing",
        "severe_partial_evidence_conflict",
        "weak_evidence_conflict",
        "recovery_no_gain_conflict",
    }
    _contextual = {"partial_answerability_conflict"}

    def decide(self, *, terminal: str, signals: AnswerabilityConflictSignalsV3) -> ConflictGateV3Decision:
        if terminal != "finished":
            return ConflictGateV3Decision(invoke_veto=False)
        reasons = set(signals.reason_codes)
        strong = tuple(reason for reason in signals.reason_codes if reason in self._strong)
        contextual = tuple(reason for reason in signals.reason_codes if reason in self._contextual)
        return ConflictGateV3Decision(invoke_veto=bool(strong), strong_reason_codes=strong, contextual_reason_codes=contextual)


T = TypeVar("T", bound=BaseModel)


class BoundedReliableNarrowPolicyV3:
    """V3 narrow policy: global <=1 transport retry + <=1 structural repair, <=3 requests total."""

    def __init__(self, *, transport: EvaluationStructuredTransport, max_transport_retries: int = 1, max_structural_repairs: int = 1) -> None:
        if max_transport_retries not in (0, 1) or max_structural_repairs not in (0, 1):
            raise ValueError("TASK-0258 retry/repair budgets are frozen at <=1 each")
        self.transport = transport
        self.max_transport_retries = max_transport_retries
        self.max_structural_repairs = max_structural_repairs

    def decide(self, *, prompt: str, model_type: type[T], function_name: str) -> ReliabilityDecision:
        schema = model_type.model_json_schema()
        req = retry = repair = latency = input_tokens = output_tokens = responses = 0
        last_failure: str | None = None
        previous_digest = "none"
        semantic_attempt = 0
        while semantic_attempt <= self.max_structural_repairs:
            current = prompt if semantic_attempt == 0 else (
                prompt
                + f"\nSTRUCTURAL_REPAIR: previous_digest={previous_digest}; failure={last_failure}. "
                  "Return exactly one corrected JSON object and no prose."
            )
            while True:
                result = self.transport.call(prompt=current, schema=schema, function_name=function_name)
                req += 1
                latency += result.latency_ms
                input_tokens += result.input_tokens
                output_tokens += result.output_tokens
                if result.response_received:
                    responses += 1
                error = result.error_code
                if error is None and not result.raw_output.strip():
                    error = "provider_empty_structured_content"
                if error and retry < self.max_transport_retries and _transport_retryable(error):
                    retry += 1
                    continue
                break
            if error:
                return ReliabilityDecision(False, None, error, req, retry, repair, latency, input_tokens, output_tokens, responses)
            previous_digest = stable_digest(result.raw_output)
            try:
                payload = json.loads(result.raw_output)
                if not isinstance(payload, dict):
                    raise ValueError("not_object")
                decision = model_type.model_validate(payload)
            except json.JSONDecodeError:
                last_failure = "invalid_json"
            except ValidationError:
                last_failure = "schema_validation_failed"
            except ValueError:
                last_failure = "not_json_object"
            else:
                return ReliabilityDecision(True, decision, None, req, retry, repair, latency, input_tokens, output_tokens, responses)
            if semantic_attempt >= self.max_structural_repairs:
                return ReliabilityDecision(False, None, last_failure, req, retry, repair, latency, input_tokens, output_tokens, responses)
            repair += 1
            semantic_attempt += 1
        raise AssertionError("unreachable")


def _transport_retryable(code: str) -> bool:
    return code in {
        "provider_timeout",
        "provider_empty_structured_content",
        "provider_urlerror",
        "agent_policy_timeout",
        "agent_policy_provider_failure",
        "agent_policy_empty_structured_content",
    } or code.startswith("provider_http_")


def veto_prompt_v3(*, query: str, signals: AnswerabilityConflictSignalsV3, evidence_excerpts: tuple[str, ...]) -> str:
    schema = AbstentionVetoDecision.model_json_schema()
    state = {
        "query": query,
        "conflict_signals": signals.model_dump(mode="json"),
        "evidence_excerpts": list(evidence_excerpts[:3]),
        "authority": "keep_finish_or_abstain_only",
    }
    return "\n".join(
        [
            "OPK-RAG bounded abstention veto V3. Never answer and never authorize Finish beyond keep_finish.",
            "Use only supplied evidence. Return exactly one JSON object; no markdown or prose.",
            "decision must be exactly keep_finish or abstain.",
            "Do not abstain merely because the query asks for a qualitative comparison when both alternatives are directly supported.",
            "Do not abstain merely because answerability is partial if Recovery improved the evidence and no strong unsupported-claim signal remains.",
            "reason_code must be exactly one of: direct_support_present, exact_fact_not_supported, evidence_conflict, recovery_no_gain, insufficient_direct_support.",
            "public_summary must be concise and no longer than 120 characters.",
            f"STATE={json.dumps(state, ensure_ascii=False, sort_keys=True)}",
            f"SCHEMA={json.dumps(schema, ensure_ascii=False, sort_keys=True)}",
        ]
    )
