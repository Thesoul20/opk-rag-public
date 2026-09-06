from __future__ import annotations

import re
import unicodedata

from opk_rag.answerability.models import AnswerabilityConfig, AnswerabilityDecision, AnswerabilityReasonCode, AnswerabilityStatus
from opk_rag.search.models import EvidenceBundle, EvidenceItem, SearchResponse

_ASCII_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+.-]{1,}")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_STOP_CHARS = set("的是了和与及或在我你他她它们这个那个是否怎么如何什么为什么一个已经目前时候进行计划")


class AnswerabilityPolicy:
    def __init__(self, config: AnswerabilityConfig | None = None) -> None:
        self.config = config or AnswerabilityConfig()

    def evaluate(
        self,
        *,
        question: str,
        evidence_bundle: EvidenceBundle | None,
        search_response: SearchResponse | None = None,
        retrieval_diagnostics: dict[str, object] | None = None,
        prompt_injection_detected: bool = False,
    ) -> AnswerabilityDecision:
        evidence_count = len(evidence_bundle.items) if evidence_bundle is not None else 0
        diagnostics = self._diagnostics(
            question=question,
            evidence_bundle=evidence_bundle,
            search_response=search_response,
            retrieval_diagnostics=retrieval_diagnostics,
            prompt_injection_detected=prompt_injection_detected,
        )
        valid_items, invalid_reasons = _valid_items(evidence_bundle)
        diagnostics["invalid_evidence_reasons"] = invalid_reasons
        diagnostics["valid_evidence_count"] = len(valid_items)
        diagnostics["answer_llm_called"] = False

        evidence_ids = tuple(str(item.chunk_id) for item in valid_items)
        evidence_score = _evidence_score(evidence_bundle=evidence_bundle, search_response=search_response)
        diagnostics["evidence_score"] = evidence_score

        if not self.config.enabled:
            return _decision("answerable", "disabled", "Answerability policy is disabled by configuration.", 1.0, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)
        if prompt_injection_detected:
            return _decision("unanswerable", "prompt_injection_detected", "The current query matched the deterministic prompt-injection guard.", 0.99, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)
        if not question.strip():
            return _decision("unanswerable", "insufficient_evidence", "The question is empty after trimming.", 0.95, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)
        if evidence_bundle is None or evidence_count == 0:
            return _decision("unanswerable", "no_evidence", "The current retrieval returned no evidence items.", 0.98, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)
        if search_response is not None and search_response.retrieval_degraded:
            return _decision("unanswerable", "insufficient_evidence", "The current retrieval was degraded.", 0.85, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)
        if evidence_bundle.context_token_count < self.config.min_context_tokens or (search_response is not None and search_response.context_token_count < 0):
            return _decision("unanswerable", "insufficient_evidence", "The current evidence does not contain enough context tokens.", 0.85, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)
        if evidence_bundle.query != question:
            return _decision("unanswerable", "invalid_evidence_bundle", "Evidence bundle query does not match the current question.", 0.95, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)
        if search_response is not None and evidence_bundle.knowledge_base_id != search_response.knowledge_base_id:
            return _decision("unanswerable", "invalid_evidence_bundle", "Evidence bundle knowledge_base_id does not match the search response.", 0.95, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)
        if search_response is not None and search_response.context_token_count > 0 and evidence_bundle.context_token_count != search_response.context_token_count:
            return _decision("unanswerable", "invalid_evidence_bundle", "Evidence bundle context token count does not match the search response.", 0.9, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)
        if not valid_items:
            return _decision("unanswerable", "insufficient_evidence", "The current retrieval did not provide any valid evidence items.", 0.9, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)

        distinct_sources = {item.relative_path for item in valid_items}
        diagnostics["distinct_source_count"] = len(distinct_sources)
        if len(distinct_sources) < self.config.min_distinct_sources:
            return _decision("unanswerable", "insufficient_distinct_sources", "The current evidence does not include enough distinct document sources.", 0.85, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)

        coverage = _coverage(question, valid_items)
        diagnostics.update(coverage)
        exact_missing = _missing_exact_requirement(question, valid_items)
        diagnostics["missing_exact_requirements"] = exact_missing
        if exact_missing:
            return _decision("unanswerable", "no_relevant_context", "The question asks for an exact value that is absent from the retrieved evidence.", 0.82, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)

        if _looks_like_partial_necessity_inference(question, valid_items):
            diagnostics["semantic_refinement"] = "necessity_inference_only_partially_supported"
            return _decision("partially_answerable", "partial_evidence", "Evidence supports the described method or capability but does not prove the exclusive necessity asserted by the question.", 0.6, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)

        unsupported_inference = _looks_like_unsupported_inference(question, valid_items)
        if unsupported_inference:
            return _decision("unanswerable", "unsupported_inference", "The question asks for an inference or guarantee that retrieved evidence does not support.", 0.78, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)

        false_premise = _looks_like_false_premise_question(question, valid_items)
        if false_premise:
            return _decision("partially_answerable", "false_premise", "Retrieved evidence appears to contradict the premise; generation must correct only the supported premise.", 0.65, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)

        conflict = _conflicting_evidence(question, valid_items)
        diagnostics["conflicting_values"] = conflict
        if conflict:
            return _decision("partially_answerable", "conflicting_evidence", "Retrieved evidence contains conflicting checkable values; generation must describe the conflict without resolving it from external knowledge.", 0.55, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)

        score_status = self._score_status(evidence_score)
        if score_status == "reject":
            return _decision("unanswerable", "no_relevant_context", "Comparable retrieval scores are too low and other evidence signals are insufficient.", 0.8, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)
        if len(valid_items) < self.config.min_evidence:
            if coverage["coverage_ratio"] >= self.config.min_partial_coverage:
                return _decision("partially_answerable", "partial_evidence", "Only a subset of the requested minimum evidence is available; generation must limit the answer to cited evidence.", 0.55, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)
            return _decision("unanswerable", "insufficient_evidence", "The current retrieval did not provide enough valid evidence items.", 0.85, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)
        if coverage["coverage_ratio"] < self.config.min_full_coverage and coverage["query_term_count"] >= 3:
            return _decision("partially_answerable", "partial_evidence", "Retrieved evidence covers only part of the question; generation must state the missing part.", 0.6, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)
        if score_status == "partial":
            return _decision("partially_answerable", "partial_evidence", "Retrieval score is marginal; generation must avoid unsupported expansion.", 0.58, evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)
        return _decision("answerable", "answerable", "The current retrieval evidence satisfies the MVP answerability decision policy.", _confidence(coverage["coverage_ratio"], evidence_score), evidence_ids, evidence_score, evidence_count, len(valid_items), diagnostics)

    def _score_status(self, evidence_score: float | None) -> str:
        if self.config.min_score is None:
            return "ok"
        if evidence_score is None:
            return "ok"
        if evidence_score >= self.config.min_score:
            return "ok"
        if evidence_score >= max(0.0, self.config.min_score - self.config.low_score_partial_margin):
            return "partial"
        return "reject"

    def _diagnostics(
        self,
        *,
        question: str,
        evidence_bundle: EvidenceBundle | None,
        search_response: SearchResponse | None,
        retrieval_diagnostics: dict[str, object] | None,
        prompt_injection_detected: bool,
    ) -> dict[str, object]:
        diagnostics: dict[str, object] = {
            "enabled": self.config.enabled,
            "min_evidence": self.config.min_evidence,
            "min_distinct_sources": self.config.min_distinct_sources,
            "min_score": self.config.min_score,
            "min_context_tokens": self.config.min_context_tokens,
            "min_full_coverage": self.config.min_full_coverage,
            "min_partial_coverage": self.config.min_partial_coverage,
            "question_empty": not question.strip(),
            "prompt_injection_detected": prompt_injection_detected,
            "evidence_bundle_present": evidence_bundle is not None,
        }
        if evidence_bundle is not None:
            diagnostics.update(
                {
                    "bundle_context_token_count": evidence_bundle.context_token_count,
                    "bundle_context_token_budget": evidence_bundle.context_token_budget,
                    "bundle_total_token_count": evidence_bundle.total_token_count,
                }
            )
        if search_response is not None:
            diagnostics.update(
                {
                    "retrieval_mode": search_response.retrieval_mode,
                    "candidate_count": search_response.candidate_count,
                    "result_count": search_response.result_count,
                    "context_token_count": search_response.context_token_count,
                    "retrieval_degraded": search_response.retrieval_degraded,
                    "reranker_enabled": search_response.reranker_enabled,
                }
            )
            if search_response.evidence_signals is not None:
                diagnostics["selected_source_document_count"] = search_response.evidence_signals.selected_source_document_count
                diagnostics["selected_chunk_count"] = search_response.evidence_signals.selected_chunk_count
                diagnostics["max_vector_similarity"] = search_response.evidence_signals.max_vector_similarity
                diagnostics["max_bm25_score"] = search_response.evidence_signals.max_bm25_score
                diagnostics["max_rrf_score"] = search_response.evidence_signals.max_rrf_score
                diagnostics["top_reranker_score"] = search_response.evidence_signals.top_reranker_score
                diagnostics["query_term_coverage_signal"] = search_response.evidence_signals.query_term_coverage
        if retrieval_diagnostics:
            diagnostics.update(retrieval_diagnostics)
        return diagnostics


def _decision(status: AnswerabilityStatus, reason_code: AnswerabilityReasonCode, reason: str, confidence: float, evidence_chunk_ids: tuple[str, ...], evidence_score: float | None, evidence_count: int, considered_evidence_count: int, diagnostics: dict[str, object]) -> AnswerabilityDecision:
    return AnswerabilityDecision(
        status=status,
        reason_code=reason_code,
        reason=reason,
        confidence=max(0.0, min(1.0, confidence)),
        evidence_chunk_ids=evidence_chunk_ids,
        evidence_score=evidence_score,
        evidence_count=evidence_count,
        considered_evidence_count=considered_evidence_count,
        diagnostics=diagnostics,
    )


def _valid_items(evidence_bundle: EvidenceBundle | None) -> tuple[list[EvidenceItem], list[str]]:
    valid_items: list[EvidenceItem] = []
    invalid_reasons: list[str] = []
    if evidence_bundle is None:
        return valid_items, invalid_reasons
    seen_chunk_ids = set()
    for item in evidence_bundle.items:
        invalid = _invalid_item_reason(item)
        if invalid is not None:
            invalid_reasons.append(invalid)
            continue
        if item.chunk_id in seen_chunk_ids:
            invalid_reasons.append("duplicate_chunk_id")
            continue
        seen_chunk_ids.add(item.chunk_id)
        valid_items.append(item)
    return valid_items, invalid_reasons


def _invalid_item_reason(item: EvidenceItem) -> str | None:
    if not str(item.chunk_id).strip() or not str(item.document_id).strip():
        return "missing_identifier"
    if not item.relative_path or item.relative_path.startswith("/") or ".." in item.relative_path.split("/"):
        return "invalid_relative_path"
    if not item.content.strip():
        return "empty_content"
    if item.context_token_count < 1:
        return "empty_context_tokens"
    return None


def _evidence_score(*, evidence_bundle: EvidenceBundle | None, search_response: SearchResponse | None) -> float | None:
    if evidence_bundle is None or search_response is None:
        return None
    scores = _comparable_scores(evidence_bundle=evidence_bundle, search_response=search_response)
    if not scores:
        return None
    return max(scores)


def _comparable_scores(*, evidence_bundle: EvidenceBundle, search_response: SearchResponse | None) -> list[float]:
    if search_response is None:
        return []
    if search_response.reranker_enabled:
        return [item.rerank_score for item in evidence_bundle.items if item.rerank_score is not None]
    if search_response.retrieval_mode == "vector":
        by_chunk = {result.chunk_id: result.similarity for result in search_response.results}
        return [by_chunk[item.chunk_id] for item in evidence_bundle.items if item.chunk_id in by_chunk]
    return []


def _coverage(question: str, items: list[EvidenceItem]) -> dict[str, object]:
    terms = _query_terms(question)
    if not terms:
        return {"query_terms": [], "covered_query_terms": [], "query_term_count": 0, "covered_query_term_count": 0, "coverage_ratio": 1.0}
    haystack = _normalize(" ".join(" ".join((item.relative_path, " ".join(item.heading_path), item.content)) for item in items))
    covered = [term for term in terms if term in haystack]
    return {
        "query_terms": terms,
        "covered_query_terms": covered,
        "query_term_count": len(terms),
        "covered_query_term_count": len(covered),
        "coverage_ratio": len(covered) / len(terms),
    }


def _query_terms(question: str) -> list[str]:
    normalized = _normalize(question)
    terms: list[str] = []
    terms.extend(match.group(0) for match in _ASCII_WORD_RE.finditer(normalized))
    terms.extend(match.group(0) for match in _NUMBER_RE.finditer(normalized))
    terms.extend(char for char in _CJK_RE.findall(normalized) if char not in _STOP_CHARS)
    deduped: list[str] = []
    seen = set()
    for term in terms:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped


def _conflicting_evidence(question: str, items: list[EvidenceItem]) -> list[str]:
    terms = set(_query_terms(question))
    if len(items) < 2 or not terms:
        return []
    values_by_item = []
    for item in items:
        item_terms = set(_query_terms(item.content))
        if terms & item_terms:
            values_by_item.append(tuple(sorted(set(_NUMBER_RE.findall(item.content)))))
    if len(values_by_item) < 2 or any(len(values) != 1 for values in values_by_item):
        return []
    distinct_values = {values[0] for values in values_by_item}
    return sorted(distinct_values) if len(distinct_values) > 1 else []


def _looks_like_false_premise_question(question: str, items: list[EvidenceItem]) -> bool:
    normalized_question = _normalize(question)
    premise_markers = ("既然", "是不是", "已经", "都过了", "都完成", "都实现", "完整", "为什么")
    if not any(fragment in normalized_question for fragment in premise_markers):
        return False
    # Neutral causal confirmations such as “是不是跟 npm 版本有关” do not assert
    # a completed/positive premise that negated evidence can falsify.
    if re.search(r"是不是(?:跟|与).+有关", normalized_question):
        return False
    # A why-question that itself contains a negative premise ("为什么不……") is
    # asking for an explanation, not asserting the opposite positive premise.
    if "为什么" in normalized_question and _contains_negative_expression(normalized_question):
        return False
    # Likewise, negative yes/no propositions ("是不是没法……") should be handled
    # by the negative/partial semantic layer rather than classified as false premise.
    if "是不是" in normalized_question and _contains_negative_expression(normalized_question):
        return False
    for item in items:
        for line in item.content.splitlines() or [item.content]:
            normalized_line = _normalize(line)
            if not (_contains_negative_expression(normalized_line) or re.match(r"^\s*[-*]?\s*\[\s*\]", line)):
                continue
            if _semantic_overlap(question, line):
                return True
    return False


def _contains_negative_expression(text: str) -> bool:
    # "是不是" is an interrogative marker; its embedded "不是" must not be
    # mistaken for a negative proposition. Any real negation elsewhere remains.
    normalized = text.replace("是不是", "")
    return any(fragment in normalized for fragment in ("不", "未", "没有", "没法", "无法", "不是", "不属于", "尚未", "还没", "暂不"))


def _semantic_overlap(question: str, evidence_line: str) -> bool:
    q = _normalize(question)
    e = _normalize(evidence_line)
    q_ascii = {m.group(0) for m in _ASCII_WORD_RE.finditer(q)}
    e_ascii = {m.group(0) for m in _ASCII_WORD_RE.finditer(e)}
    generic_ascii = {"agent", "skill", "cli", "app", "word", "html", "xml"}
    ascii_hits = (q_ascii - generic_ascii) & (e_ascii - generic_ascii)
    if len(ascii_hits) >= 1:
        return True
    if "agent skill" in q and "agent skill" in e:
        return True
    q_cjk = {q[i:i+2] for i in range(len(q)-1) if all("\u4e00" <= ch <= "\u9fff" for ch in q[i:i+2])}
    e_cjk = {e[i:i+2] for i in range(len(e)-1) if all("\u4e00" <= ch <= "\u9fff" for ch in e[i:i+2])}
    generic_cjk = {"现在", "已经", "是不是", "为什么", "这个", "那个", "应该", "可以", "完成", "实现"}
    return len((q_cjk - generic_cjk) & (e_cjk - generic_cjk)) >= 2


def _looks_like_partial_necessity_inference(question: str, items: list[EvidenceItem]) -> bool:
    q = _normalize(question)
    necessity = bool(re.search(r"(?:如果)?不.+(?:无法|没法|不能)|(?:是不是|是否).*(?:只能|必须|唯一)", q))
    if not necessity:
        return False
    evidence = _normalize(" ".join(item.content for item in items))
    # If evidence itself explicitly states exclusivity/necessity, this is not merely
    # partial. Otherwise positive support for the method does not prove that all
    # alternatives are impossible.
    if any(marker in evidence for marker in ("只能", "唯一", "必须通过", "不修改就无法", "否则无法")):
        return False
    return _semantic_overlap(question, evidence)


def _looks_like_unsupported_inference(question: str, items: list[EvidenceItem]) -> bool:
    normalized_question = _normalize(question)
    if not any(fragment in normalized_question for fragment in ("任何", "保证", "是否说明", "是不是说明")):
        return False
    evidence = _normalize(" ".join(item.content for item in items))
    return any(fragment in evidence for fragment in ("考虑", "计划", "后续", "短期不做", "只做", "可以", "优势", "极快"))


def _missing_exact_requirement(question: str, items: list[EvidenceItem]) -> list[str]:
    normalized_question = _normalize(question)
    evidence = _normalize(" ".join(item.content for item in items))
    missing = [date for date in _ISO_DATE_RE.findall(normalized_question) if date not in evidence]
    if re.search(r"(?:哪一个|哪个|具体哪个|具体哪一个).{0,12}版本", normalized_question):
        if "npm" in normalized_question:
            version_like = re.findall(r"(?:npm\s*(?:v?\d+(?:\.\d+){0,3})|(?:v?\d+(?:\.\d+){0,3})\s*(?:版|版本)?[^\n]{0,12}npm)", evidence, re.I)
        else:
            version_like = re.findall(r"(?:v?\d+(?:\.\d+){1,3}|node\s*\d+|npm\s*\d+)", evidence, re.I)
        if not version_like:
            missing.append("exact_version")
    for phrase in ("月活", "用户数", "默认安装目录"):
        if phrase in normalized_question and phrase not in evidence:
            missing.append(phrase)
    if "默认安装目录" in normalized_question and not any(marker in evidence for marker in ("/", "\\", "c:", "program files")):
        missing.append("path")
    empirical_request = any(marker in normalized_question for marker in ("实际测试", "实测", "有测过数字", "测过数字", "成功率", "高多少", "最多会丢", "最多丢失"))
    if empirical_request:
        if not re.search(r"\d", evidence):
            missing.append("empirical_numeric_result")
        if "断电" in normalized_question and "断电" not in evidence:
            missing.append("power_loss_test")
        if "稳定性" in normalized_question and not any(marker in evidence for marker in ("稳定性测试", "稳定性对比", "实测", "测试结果")):
            missing.append("stability_comparison_test")
        if "成功率" in normalized_question and "成功率" not in evidence:
            missing.append("success_rate")
    return sorted(set(missing))


def _confidence(coverage_ratio: float, evidence_score: float | None) -> float:
    if evidence_score is None:
        return 0.7 + min(coverage_ratio, 1.0) * 0.2
    return 0.45 + min(coverage_ratio, 1.0) * 0.25 + min(max(evidence_score, 0.0), 1.0) * 0.3


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()
