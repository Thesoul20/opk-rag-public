from __future__ import annotations

from opk_rag.answerability.decision import AnswerabilityPolicy
from opk_rag.answerability.models import AnswerabilityConfig, AnswerabilityDecision
from opk_rag.search.models import SearchResponse


def decide_answerability(search_response: SearchResponse, *, config: AnswerabilityConfig | None = None, prompt_injection_detected: bool = False) -> AnswerabilityDecision:
    return AnswerabilityPolicy(config).evaluate(
        question=search_response.query,
        evidence_bundle=search_response.evidence_bundle,
        search_response=search_response,
        prompt_injection_detected=prompt_injection_detected,
    )
