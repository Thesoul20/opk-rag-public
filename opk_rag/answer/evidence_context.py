from __future__ import annotations

from opk_rag.answer.config import ANSWER_OUTPUT_SCHEMA_VERSION
from opk_rag.answerability import AnswerabilityDecision
from opk_rag.answer.negative_semantics import classify_answer_semantics
from opk_rag.search.models import EvidenceBundle

EVIDENCE_CONTEXT_SCHEMA_VERSION = "evidence-context-v1"


def build_evidence_context(
    *,
    question: str,
    bundle: EvidenceBundle,
    answerability: AnswerabilityDecision | None,
    output_schema_version: str = ANSWER_OUTPUT_SCHEMA_VERSION,
) -> dict[str, object]:
    """Build the structured context given to the answer provider."""
    contract = generation_contract(answerability, bundle, question=question)
    evidence = evidence_items(bundle, selected_citation_ids=set(contract["selected_evidence_citation_ids"]))
    return {
        "schema_version": EVIDENCE_CONTEXT_SCHEMA_VERSION,
        "question": question,
        "output_schema_version": output_schema_version,
        "answerability_status": contract["answerability_status"],
        "reason_code": contract["reason_code"],
        "allowed_behavior": contract["allowed_behavior"],
        "supported_scope": contract["supported_scope"],
        "unsupported_scope": contract["unsupported_scope"],
        "premise_correction": contract["premise_correction"],
        "selected_evidence_citation_ids": contract["selected_evidence_citation_ids"],
        "instructions": contract["instructions"],
        "answer_semantic_class": contract["answer_semantic_class"],
        "negative_answer_allowed": contract["negative_answer_allowed"],
        "negative_support_citation_ids": contract["negative_support_citation_ids"],
        "negative_support_type": contract["negative_support_type"],
        "absence_only": contract["absence_only"],
        "evidence": evidence,
        "diagnostics": evidence_context_diagnostics(bundle=bundle, context={"evidence": evidence, **contract}, provider_response_mode=None),
    }


def generation_contract(decision: AnswerabilityDecision | None, bundle: EvidenceBundle, *, question: str | None = None) -> dict[str, object]:
    citation_ids_by_chunk = {str(item.chunk_id): f"C{index}" for index, item in enumerate(bundle.items, start=1)}
    selected = [citation_ids_by_chunk[chunk_id] for chunk_id in decision.evidence_chunk_ids if chunk_id in citation_ids_by_chunk] if decision is not None else list(citation_ids_by_chunk.values())
    if decision is not None and not selected:
        selected = list(citation_ids_by_chunk.values())
    diagnostics = decision.diagnostics if decision is not None else {}
    status = decision.status if decision is not None else "answerable"
    reason_code = decision.reason_code if decision is not None else "answerable"
    supported_scope = _supported_scope(diagnostics, selected)
    unsupported_scope = _unsupported_scope(diagnostics, reason_code)
    premise_correction = _premise_correction(decision)
    semantic = classify_answer_semantics(question=question or bundle.query, bundle=bundle, answerability=decision)
    allowed_behavior = {
        "answerable": "answer_only_cited_evidence",
        "partially_answerable": "answer_supported_scope_and_disclose_missing_scope",
        "unanswerable": "abstain_without_citations",
    }[status]
    return {
        "answerability_status": status,
        "reason_code": reason_code,
        "selected_evidence_citation_ids": selected,
        "supported_scope": supported_scope,
        "unsupported_scope": unsupported_scope,
        "premise_correction": premise_correction,
        "allowed_behavior": allowed_behavior,
        "answer_semantic_class": semantic.semantic_class,
        "negative_answer_allowed": semantic.semantic_class == "supported_negative" and semantic.safe_to_finish,
        "negative_support_citation_ids": list(semantic.supporting_citation_ids),
        "negative_support_type": semantic.negative_support_type,
        "absence_only": semantic.absence_only,
        "instructions": _contract_instructions(status=status, reason_code=reason_code, unsupported_scope=unsupported_scope, premise_correction=premise_correction, semantic_class=semantic.semantic_class),
    }


def evidence_items(bundle: EvidenceBundle, *, selected_citation_ids: set[str] | None = None) -> list[dict[str, object]]:
    selected_citation_ids = selected_citation_ids or set()
    rows = []
    for index, item in enumerate(bundle.items, start=1):
        citation_id = f"C{index}"
        rows.append(
            {
                "citation_id": citation_id,
                "selected_for_answer": not selected_citation_ids or citation_id in selected_citation_ids,
                "source": {
                    "file": item.relative_path,
                    "heading_path": list(item.heading_path),
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "chunk_id": str(item.chunk_id),
                    "document_id": str(item.document_id),
                },
                "retrieval": {
                    "context_rank": item.context_rank,
                    "result_rank": item.result_rank,
                    "retrieval_sources": list(item.retrieval_sources),
                    "rerank_score": item.rerank_score,
                },
                "content": item.content,
            }
        )
    return rows


def evidence_context_diagnostics(*, bundle: EvidenceBundle, context: dict[str, object], provider_response_mode: str | None) -> dict[str, object]:
    evidence = context.get("evidence")
    supported_scope = context.get("supported_scope")
    unsupported_scope = context.get("unsupported_scope")
    selected = context.get("selected_evidence_citation_ids")
    citation_ids = [str(row.get("citation_id")) for row in evidence if isinstance(row, dict) and row.get("citation_id")] if isinstance(evidence, list) else []
    return {
        "schema_version": EVIDENCE_CONTEXT_SCHEMA_VERSION,
        "context_token_budget": bundle.context_token_budget,
        "context_token_count": bundle.context_token_count,
        "evidence_count": len(bundle.items),
        "supported_scope_count": len(supported_scope) if isinstance(supported_scope, list) else 0,
        "unsupported_scope_count": len(unsupported_scope) if isinstance(unsupported_scope, list) else 0,
        "citation_count": len(citation_ids),
        "selected_citation_count": len(selected) if isinstance(selected, list) else 0,
        "citation_ids": citation_ids,
        "provider_response_mode": provider_response_mode,
    }


def _supported_scope(diagnostics: dict[str, object], selected_citation_ids: list[str]) -> list[str]:
    semantic = diagnostics.get("semantic_supported_scope")
    if isinstance(semantic, list) and semantic:
        return [str(value).strip() for value in semantic if str(value).strip()]
    covered = diagnostics.get("covered_query_terms")
    if isinstance(covered, list):
        values = [str(value).strip() for value in covered if str(value).strip()]
    else:
        values = []
    if values:
        return values
    return [f"Use evidence {citation_id} for supported facts." for citation_id in selected_citation_ids]


def _unsupported_scope(diagnostics: dict[str, object], reason_code: str) -> list[str]:
    values: list[str] = []
    semantic = diagnostics.get("semantic_unsupported_scope")
    if isinstance(semantic, list):
        values.extend(str(value).strip() for value in semantic if str(value).strip())
    for key in ("missing_exact_requirements", "conflicting_values"):
        raw = diagnostics.get(key)
        if isinstance(raw, list):
            values.extend(str(value).strip() for value in raw if str(value).strip())
    query_terms = diagnostics.get("query_terms")
    covered_terms = diagnostics.get("covered_query_terms")
    if isinstance(query_terms, list) and isinstance(covered_terms, list):
        covered = {str(value) for value in covered_terms}
        values.extend(f"未由证据覆盖的问题要素：{value}" for value in query_terms if str(value) not in covered)
    if reason_code == "unsupported_inference":
        values.append("问题要求的推断或保证没有被当前证据支持")
    return list(dict.fromkeys(values))


def _premise_correction(decision: AnswerabilityDecision | None) -> str | None:
    if decision is None or decision.reason_code != "false_premise":
        return None
    return "The question premise conflicts with the retrieved evidence. Correct the premise first, cite the supporting evidence, and do not continue the false assumption."


def _contract_instructions(*, status: str, reason_code: str, unsupported_scope: list[str], premise_correction: str | None, semantic_class: str) -> list[str]:
    if semantic_class == "supported_negative":
        return [
            "Answer exactly one factual sentence for the proposition asked by the user; do not add a second explanatory or concluding factual sentence.",
            "State the bounded negative conclusion clearly, preserve qualifiers such as current/not-yet/platform scope, and place one or more direct negative citation markers immediately in that same sentence.",
            "Use only negative_support_citation_ids for the negative conclusion and make sure the sentence itself contains the citation marker before it ends.",
            "Do not add adjacent capabilities, completed items, broader roadmap facts, or a separate missing-evidence disclosure unless the user explicitly asked for them.",
            "Do not generalize the bounded negative into a permanent or broader claim, and never treat missing evidence as proof of a negative fact.",
        ]
    if reason_code == "false_premise":
        return [
            "State that the question premise is not supported or is contradicted by the knowledge base.",
            "Then provide only the corrected fact that is directly supported by the selected evidence.",
            "Attach inline citations to every corrected factual statement.",
        ]
    if status == "partially_answerable":
        instructions = [
            "Do not abstain if selected evidence supports at least one requested part.",
            "Start the answer with the supported facts and cite each factual sentence.",
            "Then state what the current knowledge base does not provide, without citations unless a citation directly supports the absence claim.",
        ]
        if unsupported_scope:
            instructions.append("Do not answer the unsupported_scope items with outside knowledge.")
        return instructions
    if status == "unanswerable":
        return ["Abstain without citations."]
    return [
        "Answer only the proposition or scope explicitly asked by the user, using facts directly supported by evidence and citing each factual sentence.",
        "Do not add adjacent implementation details or nearby facts merely because they appear in the same evidence chunk.",
    ]
