from __future__ import annotations

from opk_rag.answerability import AnswerabilityDecision

DEFAULT_REFUSAL_TEXT = "当前知识库中没有找到足够的证据来可靠回答这个问题。你可以尝试补充相关文档、重新建立索引，或将问题描述得更具体。"
NO_EVIDENCE_REFUSAL_TEXT = "当前知识库中没有检索到与该问题相关的内容，因此无法基于知识库可靠回答。你可以尝试补充相关文档、重新建立索引，或将问题描述得更具体。"
INSUFFICIENT_EVIDENCE_REFUSAL_TEXT = "当前检索到的内容不足以支持一个可靠回答，因此本次不生成推测性结论。你可以尝试补充相关文档、重新建立索引，或将问题描述得更具体。"
PROMPT_INJECTION_REFUSAL_TEXT = "当前问题包含可能要求绕过引用或泄露系统指令的内容，因此本次不基于知识库生成回答。"
GROUNDING_REFUSAL_TEXT = "已检索到相关资料，但生成的回答未能通过引用一致性校验，因此本次不返回该回答。你可以重试，或调整问题后再次查询。"


def refusal_text(decision: AnswerabilityDecision | None, *, reason_code: str | None = None) -> str:
    code = decision.reason_code if decision is not None else reason_code
    if code == "no_evidence":
        return NO_EVIDENCE_REFUSAL_TEXT
    if code in {"insufficient_evidence", "below_score_threshold", "insufficient_distinct_sources", "invalid_evidence_bundle", "no_relevant_context", "partial_evidence", "conflicting_evidence", "unsupported_inference", "false_premise"}:
        return INSUFFICIENT_EVIDENCE_REFUSAL_TEXT
    if code == "prompt_injection_detected":
        return PROMPT_INJECTION_REFUSAL_TEXT
    if code in {
        "ungrounded_answer",
        "invalid_citations",
        "unsupported_claims",
        "answerable_generation_abstained",
        "partial_supported_scope_missing",
        "partial_unsupported_scope_fabricated",
        "partial_disclosure_missing",
        "false_premise_not_corrected",
        "false_premise_followed",
        "citation_not_supporting_claim",
        "generation_overreach",
    }:
        return GROUNDING_REFUSAL_TEXT
    return DEFAULT_REFUSAL_TEXT
