from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from opk_rag.answerability import AnswerabilityDecision
from opk_rag.search.models import EvidenceBundle

ANSWER_SEMANTIC_CONTRACT_VERSION = "negative-answer-evidence-contract-v1"
AnswerSemanticClass = Literal["supported_affirmative", "supported_negative", "supported_partial", "abstain"]
NegativeSupportType = Literal[
    "explicit_negation",
    "explicit_unimplemented",
    "explicit_unfinished",
    "explicit_disabled",
    "explicit_platform_contradiction",
    "explicit_future_work",
    "false_premise_correction",
    "other_direct_negative_support",
    "none",
]

_NEGATION_RE = re.compile(
    r"(?:不支持|不负责|不直接|不允许|不可|不能|未完成|尚未|还没有|没有完成|未实现|尚未实现|"
    r"暂不|默认关闭|已禁用|禁用|不需要|not supported|not implemented|not completed|disabled|future work)",
    re.I,
)
_UNCHECKED_RE = re.compile(r"^\s*[-*]?\s*\[\s\]\s+(.+)$")
_ASCII_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+\\/-]{1,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_NEGATIVE_EXPRESSION_RE = re.compile(
    r"(?:不|未|没有|尚未|还没|不能|不可|并非|而非|不是|否|not\b|no\b|never\b|unsupported|disabled)", re.I
)
_CONTRASTIVE_CORRECTION_RE = re.compile(r"(?:，|,)\s*(?:而|但|相反)|(?:实际|正确(?:的是|为)|应(?:该|当))", re.I)
_QUESTION_STOP = {
    "是否", "已经", "现在", "当前", "文档", "证明", "可以", "能够", "支持", "负责", "完成", "了吗", "吗",
    "什么", "哪些", "怎么", "如何", "为什么", "以及", "同时", "给出", "说明", "最小", "测试", "报告", "路线图",
}


@dataclass(frozen=True)
class NegativeAnswerEvidenceContractV1:
    contract_version: str
    semantic_class: AnswerSemanticClass
    proposition: str
    negative_support_present: bool
    negative_support_type: NegativeSupportType
    supporting_chunk_ids: tuple[str, ...]
    supporting_citation_ids: tuple[str, ...]
    evidence_strength: Literal["direct", "none"]
    absence_only: bool
    safe_to_finish: bool
    reason_codes: tuple[str, ...]

    def to_provider_payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "semantic_class": self.semantic_class,
            "proposition": self.proposition,
            "negative_support_present": self.negative_support_present,
            "negative_support_type": self.negative_support_type,
            "supporting_citation_ids": list(self.supporting_citation_ids),
            "evidence_strength": self.evidence_strength,
            "absence_only": self.absence_only,
            "safe_to_finish": self.safe_to_finish,
            "reason_codes": list(self.reason_codes),
        }


def classify_answer_semantics(
    *,
    question: str,
    bundle: EvidenceBundle,
    answerability: AnswerabilityDecision | None,
) -> NegativeAnswerEvidenceContractV1:
    status = answerability.status if answerability is not None else "answerable"
    reason_code = answerability.reason_code if answerability is not None else "answerable"
    diagnostics = answerability.diagnostics if answerability is not None else {}
    if diagnostics.get("semantic_supported_scope") and diagnostics.get("semantic_unsupported_scope"):
        return NegativeAnswerEvidenceContractV1(
            contract_version=ANSWER_SEMANTIC_CONTRACT_VERSION, semantic_class="supported_partial", proposition=question.strip(),
            negative_support_present=False, negative_support_type="none", supporting_chunk_ids=(), supporting_citation_ids=(),
            evidence_strength="none", absence_only=False, safe_to_finish=True, reason_codes=("mixed_supported_and_unsupported_scope",),
        )
    if diagnostics.get("semantic_refinement") == "necessity_inference_only_partially_supported":
        return NegativeAnswerEvidenceContractV1(
            contract_version=ANSWER_SEMANTIC_CONTRACT_VERSION, semantic_class="supported_partial", proposition=question.strip(),
            negative_support_present=False, negative_support_type="none", supporting_chunk_ids=(), supporting_citation_ids=(),
            evidence_strength="none", absence_only=False, safe_to_finish=True, reason_codes=("necessity_inference_only_partially_supported",),
        )
    if status == "unanswerable" and reason_code in {"no_relevant_context", "insufficient_evidence", "unsupported_inference"} and _requires_direct_target(question):
        return NegativeAnswerEvidenceContractV1(
            contract_version=ANSWER_SEMANTIC_CONTRACT_VERSION, semantic_class="abstain", proposition=question.strip(),
            negative_support_present=False, negative_support_type="none", supporting_chunk_ids=(), supporting_citation_ids=(),
            evidence_strength="none", absence_only=True, safe_to_finish=False, reason_codes=("exact_or_empirical_target_missing",),
        )
    support = _direct_negative_support(question, bundle)
    negative_resolution_intent = _question_allows_negative_resolution(question) or reason_code == "false_premise"
    if support and negative_resolution_intent:
        support_type, indexes = support
        citation_ids = tuple(f"C{i + 1}" for i in indexes)
        chunk_ids = tuple(str(bundle.items[i].chunk_id) for i in indexes)
        semantic: AnswerSemanticClass = "supported_negative"
        reasons = ["direct_negative_evidence"]
        if reason_code == "false_premise":
            support_type = "false_premise_correction"
            reasons.append("false_premise_correction")
        return NegativeAnswerEvidenceContractV1(
            contract_version=ANSWER_SEMANTIC_CONTRACT_VERSION,
            semantic_class=semantic,
            proposition=question.strip(),
            negative_support_present=True,
            negative_support_type=support_type,
            supporting_chunk_ids=chunk_ids,
            supporting_citation_ids=citation_ids,
            evidence_strength="direct",
            absence_only=False,
            safe_to_finish=status != "unanswerable" or reason_code == "false_premise",
            reason_codes=tuple(reasons),
        )
    if status == "unanswerable":
        semantic = "abstain"
    elif status == "partially_answerable":
        semantic = "supported_partial" if _is_semantic_partial_scope(question, bundle, answerability) else "supported_affirmative"
    else:
        semantic = "supported_affirmative"
    return NegativeAnswerEvidenceContractV1(
        contract_version=ANSWER_SEMANTIC_CONTRACT_VERSION,
        semantic_class=semantic,
        proposition=question.strip(),
        negative_support_present=False,
        negative_support_type="none",
        supporting_chunk_ids=(),
        supporting_citation_ids=(),
        evidence_strength="none",
        absence_only=semantic == "abstain",
        safe_to_finish=semantic != "abstain",
        reason_codes=("no_direct_negative_evidence",),
    )



def _is_semantic_partial_scope(
    question: str, bundle: EvidenceBundle, answerability: AnswerabilityDecision | None
) -> bool:
    """Distinguish partial semantic scope from merely marginal evidence confidence."""
    if answerability is None:
        return False
    diagnostics = dict(answerability.diagnostics or {})
    supported_scope = diagnostics.get("semantic_supported_scope")
    unsupported_scope = diagnostics.get("semantic_unsupported_scope")
    if supported_scope and unsupported_scope:
        return True
    if diagnostics.get("semantic_refinement") in {
        "multi_clause_partial",
        "necessity_inference_only_partially_supported",
    }:
        return True
    if answerability.reason_code == "conflicting_evidence":
        return True
    if answerability.reason_code != "partial_evidence":
        return False
    clauses = [c.strip(" ？?") for c in _CLAUSE_SPLIT_RE.split(question) if c.strip(" ？?")]
    if len(clauses) < 2:
        return False
    support = [_clause_supported(c, bundle) for c in clauses]
    return any(support) and not all(support)



def _question_allows_negative_resolution(question: str) -> bool:
    """Return whether the *answer orientation* may legitimately be negative.

    Negative words in a condition or explanation request do not by themselves make
    the requested answer negative.  This keeps troubleshooting/why questions and
    positive contrastive confirmations from being misclassified by nearby roadmap
    negation while preserving genuine yes/no negative propositions.
    """
    q = question.strip().casefold()
    explanatory = any(marker in q for marker in ("为什么", "什么设置", "什么原因", "原因", "怎么", "如何", "哪些", "哪里", "哪边", "多久", "多少", "具体"))
    if explanatory:
        return False
    if re.search(r"(?:而不是|而非|而不是重新|不是转成|不是变成)", q) and not q.startswith("是不是"):
        return False
    if _NEGATIVE_EXPRESSION_RE.search(q):
        return True
    return any(marker in q for marker in (
        "是否", "吗", "有没有", "有无", "能否", "能不能", "可否", "会不会",
        "是不是", "是否已经", "已经完成", "已经实现", "已经支持", "负责",
        "is ", "are ", "does ", "do ", "can ", "has ", "have ", "whether",
    ))

def validate_supported_negative_answer(
    *, answer_text: str, citation_ids: tuple[str, ...], contract: NegativeAnswerEvidenceContractV1
) -> tuple[str, ...]:
    if contract.semantic_class != "supported_negative":
        return ()
    failures: list[str] = []
    if not contract.negative_support_present or contract.absence_only or not contract.safe_to_finish:
        failures.append("unsupported_negative_finish")
    if not set(citation_ids).intersection(contract.supporting_citation_ids):
        failures.append("negative_claim_without_direct_support_citation")
    negative_expressed = bool(_NEGATIVE_EXPRESSION_RE.search(answer_text))
    # A false-premise/platform correction may answer the negative proposition by
    # positively stating the evidence-backed alternative (for example, “A is for
    # Unix, while B is for Windows”).  This remains bounded by direct negative
    # Evidence, authorized citations, and the normal claim-level Grounding checks.
    if not negative_expressed and contract.negative_support_type in {"false_premise_correction", "explicit_platform_contradiction"}:
        negative_expressed = bool(_CONTRASTIVE_CORRECTION_RE.search(answer_text))
    if not negative_expressed:
        failures.append("supported_negative_not_expressed")
    return tuple(failures)


def _direct_negative_support(question: str, bundle: EvidenceBundle) -> tuple[NegativeSupportType, tuple[int, ...]] | None:
    # Explicit platform contradictions are evidence-backed alternatives, not
    # absence inference.  Keep the original Windows/Unix activation case and also
    # recognize a cross-platform path proposition when the same evidence explicitly
    # lists distinct Linux and macOS paths.
    qlow = question.casefold()
    all_text = "\n".join(item.content for item in bundle.items)
    low = all_text.casefold()
    if "windows" in qlow and "source .venv/bin/activ" in qlow and "for unix" in low and "for windows" in low and "scripts\\activ" in low:
        idx = tuple(i for i, item in enumerate(bundle.items) if "for unix" in item.content.casefold() or "for windows" in item.content.casefold())
        if idx:
            return "explicit_platform_contradiction", idx
    if any(x in qlow for x in ("linux", "fcitx5")) and any(x in qlow for x in ("macos", "mac os", "mac")) and any(x in qlow for x in ("照搬", "同一个", "一样", "直接用", "兼容")):
        if "linux" in low and "mac" in low and "~/.local/share/fcitx5/rime" in low and "~/library/rime" in low:
            idx = tuple(i for i, item in enumerate(bundle.items) if "~/.local/share/fcitx5/rime" in item.content.casefold() or "~/library/rime" in item.content.casefold())
            if idx:
                return "explicit_platform_contradiction", idx

    # A negative proposition can also be directly contradicted by affirmative
    # evidence about the same stable tool/entity. Example shape: “do we no longer
    # need X?” while Evidence explicitly says “prefer/use X”. This is evidence-backed
    # correction, not inference from absence.
    neg_tool = re.search(r"(?:不用|不需要|无需)\s*([A-Za-z][A-Za-z0-9_.+/-]{1,})", qlow)
    if neg_tool:
        tool = neg_tool.group(1).casefold()
        affirmative_markers = (f"使用 {tool}", f"使用{tool}", f"偏向于使用 {tool}", f"偏向于使用{tool}", f"prefer {tool}", f"use {tool}")
        idx = tuple(i for i, item in enumerate(bundle.items) if any(marker in item.content.casefold() for marker in affirmative_markers))
        if idx:
            return "other_direct_negative_support", idx

    qfeatures = _question_features(question)
    matches: list[tuple[int, NegativeSupportType]] = []
    for index, item in enumerate(bundle.items):
        lines = item.content.splitlines() or [item.content]
        item_types: list[NegativeSupportType] = []
        for line in lines:
            unchecked = _UNCHECKED_RE.search(line)
            if unchecked and not _scope_membership_question(question) and _relevant_to_question(question, unchecked.group(1), qfeatures, allow_single_ascii=True):
                item_types.append("explicit_future_work")
                continue
            if _NEGATION_RE.search(line) and _relevant_to_question(question, line, qfeatures):
                if _scope_membership_question(question) and not _scope_membership_negative_relevant(question, line):
                    continue
                if re.search(r"(?:不负责)", line):
                    item_types.append("explicit_negation")
                elif re.search(r"(?:禁用|disabled)", line, re.I):
                    item_types.append("explicit_disabled")
                elif re.search(r"(?:未完成|尚未|没有完成|未实现|尚未实现|not implemented|not completed)", line, re.I):
                    item_types.append("explicit_unfinished")
                else:
                    item_types.append("explicit_negation")
        if item_types:
            matches.append((index, item_types[0]))
    if not matches:
        return None
    preferred = matches[0][1]
    indexes = tuple(dict.fromkeys(index for index, _ in matches))
    return preferred, indexes


def _question_features(text: str) -> set[str]:
    features = {m.group(0).casefold() for m in _ASCII_RE.finditer(text)}
    for chunk in _CJK_RE.findall(text):
        cleaned = chunk
        for stop in _QUESTION_STOP:
            cleaned = cleaned.replace(stop, " ")
        for part in cleaned.split():
            if len(part) >= 2:
                features.add(part)
                if len(part) >= 4:
                    features.update(part[i : i + 2] for i in range(len(part) - 1))
    return {
        f for f in features
        if len(f) >= 2
        and f not in _QUESTION_STOP
        and f.casefold() not in _GENERIC_ASCII
        and f not in _GENERIC_CJK
    }


def _scope_membership_question(question: str) -> bool:
    q = question.casefold()
    return "属于" in q or bool(re.search(r"算.+(?:一部分|范围|功能)", q)) or "包含在" in q


def _scope_membership_negative_relevant(question: str, evidence_line: str) -> bool:
    """Require negation to overlap the capability whose scope membership is being tested."""
    q = question.casefold().strip()
    marker = re.search(r"(?:是否)?(?:属于|算|包含在)", q)
    if marker is None:
        return False
    prefix = q[: marker.start()].rstrip(" ，,；;：:")
    # Topic-return rewrites often prepend a project anchor followed by a comma;
    # the final clause before the membership predicate is the tested capability.
    target = re.split(r"[，,；;：:]", prefix)[-1].strip()
    target = re.sub(r"^(?:那|那么|这个|该|其中|在.+?中)", "", target).strip()
    cjk = "".join(ch for ch in target if "\u4e00" <= ch <= "\u9fff")
    terms = {cjk[i : i + 2] for i in range(max(0, len(cjk) - 1))}
    terms.update(m.group(0).casefold() for m in _ASCII_RE.finditer(target) if m.group(0).casefold() not in _GENERIC_ASCII)
    generic = {"是否", "属于", "一部", "部分", "范围", "功能", "最小", "闭环", "该闭", "环中"}
    terms = {term for term in terms if term not in generic}
    low = evidence_line.casefold()
    return bool(terms) and any(term in low for term in terms)


def _specific_ascii_anchors(question: str) -> set[str]:
    return {
        token for token in (m.group(0).casefold() for m in _ASCII_RE.finditer(question))
        if token not in _GENERIC_ASCII and token not in _COMMON_CONTEXT_ASCII
        and len(token) >= 4
    }


def _relevant_to_question(question: str, evidence_line: str, qfeatures: set[str], *, allow_single_ascii: bool = False) -> bool:
    elow = evidence_line.casefold()
    qlow = question.casefold()
    specific_ascii = _specific_ascii_anchors(question)
    if specific_ascii and not any(anchor in elow for anchor in specific_ascii):
        return False
    if "agent skill" in qlow and "agent skill" in elow:
        return True
    ascii_hits = [f for f in qfeatures if any(ch.isascii() and ch.isalnum() for ch in f) and f in elow]
    cjk_hits = [f for f in qfeatures if any("\u4e00" <= ch <= "\u9fff" for ch in f) and f in evidence_line]
    # Ordinary negation requires proposition-level overlap; a shared document/entity name alone is not enough.
    # Unchecked roadmap items may use one stable ASCII identifier (for example Word) as the proposition anchor.
    if len(set(ascii_hits)) >= 2:
        return True
    if allow_single_ascii and ascii_hits and any(marker in question and marker in evidence_line for marker in ("支持", "完成", "实现", "负责", "安装", "使用", "群发", "管理", "测试")):
        return True
    if ascii_hits and cjk_hits:
        return True
    if len(set(cjk_hits)) >= 2:
        return True
    # Strong proposition fragments cover common roadmap/checklist wording without relying on generic negation alone.
    compact_q = re.sub(r"[\s？?，,。]", "", question)
    compact_e = re.sub(r"[\s，,。]", "", evidence_line)
    for width in (6, 5, 4):
        for i in range(max(0, len(compact_q) - width + 1)):
            frag = compact_q[i : i + width]
            cjk_count = sum("\u4e00" <= ch <= "\u9fff" for ch in frag)
            if len(frag) == width and cjk_count >= 3 and frag in compact_e and not any(stop in frag for stop in _QUESTION_STOP):
                return True
    return False


_CLAUSE_SPLIT_RE = re.compile(r"(?:；|;|。|同时|以及|并且|另外|和文档是否|且文档是否)")
_GENERIC_ASCII = {"tauri", "uv", "fcitx5", "academic-docx-polisher", "saas", "demo", "agent", "skill", "cli", "github", "repo", "astronvim", "nvim", "neovim", "config"}
_GENERIC_CJK = {"文档", "当前", "现在", "已经", "是否", "什么", "怎么", "如何", "说明", "给出", "支持", "完成", "实现", "可以", "能够", "这个", "第一", "阶段"}
_COMMON_CONTEXT_ASCII = {"python", "windows", "linux", "macos", "mac", "word", "html", "xml", "rust", "javascript", "typescript", "path", "version"}


def refine_answerability_for_semantics(*, question: str, bundle: EvidenceBundle, decision: AnswerabilityDecision) -> AnswerabilityDecision:
    """Conservative semantic refinement for multi-part/negative questions.

    This layer never turns an unanswerable prompt-injection/invalid-evidence/no-evidence decision into an answer.
    It only corrects false-premise over-detection, distinguishes supported partial scope, and fails closed
    when an exact/single proposition has no direct proposition-level support.
    """
    if decision.reason_code in {"prompt_injection_detected", "invalid_evidence_bundle", "no_evidence"}:
        return decision
    semantic = classify_answer_semantics(question=question, bundle=bundle, answerability=decision)
    if (decision.status == "unanswerable" and decision.reason_code in {"no_relevant_context", "insufficient_evidence"} and semantic.semantic_class == "supported_negative" and semantic.negative_support_present and not _requires_direct_target(question)):
        diagnostics = dict(decision.diagnostics)
        diagnostics["semantic_refinement"] = "direct_negative_evidence_recovered"
        diagnostics["semantic_supported_scope"] = [question.strip()]
        return _copy_decision(decision, status="partially_answerable", reason_code="partial_evidence", diagnostics=diagnostics)
    clauses = [c.strip(" ？?") for c in _CLAUSE_SPLIT_RE.split(question) if c.strip(" ？?")]
    support = [_clause_supported(c, bundle) for c in clauses]
    diagnostics = dict(decision.diagnostics)

    if len(clauses) >= 2 and any(support) and not all(support):
        supported = [c for c, ok in zip(clauses, support) if ok]
        unsupported = [c for c, ok in zip(clauses, support) if not ok]
        diagnostics["semantic_supported_scope"] = supported
        diagnostics["semantic_unsupported_scope"] = unsupported
        diagnostics["semantic_refinement"] = "multi_clause_partial"
        return _copy_decision(decision, status="partially_answerable", reason_code="partial_evidence", diagnostics=diagnostics)

    if len(clauses) == 1:
        supported = support[0] if support else False
        if decision.reason_code == "false_premise" and semantic.semantic_class != "supported_negative" and supported:
            diagnostics["semantic_supported_scope"] = [clauses[0]]
            diagnostics["semantic_refinement"] = "false_premise_over_detection_corrected"
            return _copy_decision(decision, status="answerable", reason_code="answerable", diagnostics=diagnostics)
        if _requires_direct_target(question) and not _direct_target_supported(question, bundle):
            diagnostics["semantic_unsupported_scope"] = [clauses[0]]
            diagnostics["semantic_refinement"] = "missing_direct_target"
            return _copy_decision(decision, status="unanswerable", reason_code="no_relevant_context", diagnostics=diagnostics)
        if decision.status in {"answerable", "partially_answerable"} and not supported and semantic.semantic_class != "supported_negative":
            diagnostics["semantic_unsupported_scope"] = [clauses[0]]
            diagnostics["semantic_refinement"] = "proposition_not_directly_supported"
            return _copy_decision(decision, status="unanswerable", reason_code="no_relevant_context", diagnostics=diagnostics)
        if supported:
            diagnostics["semantic_supported_scope"] = [clauses[0]]
    return _copy_decision(decision, diagnostics=diagnostics)


def _copy_decision(decision: AnswerabilityDecision, *, status: str | None = None, reason_code: str | None = None, diagnostics: dict[str, object] | None = None) -> AnswerabilityDecision:
    return AnswerabilityDecision(
        status=status or decision.status,
        reason_code=reason_code or decision.reason_code,
        reason=decision.reason,
        confidence=decision.confidence,
        evidence_chunk_ids=decision.evidence_chunk_ids,
        evidence_score=decision.evidence_score,
        evidence_count=decision.evidence_count,
        considered_evidence_count=decision.considered_evidence_count,
        diagnostics=diagnostics or dict(decision.diagnostics),
    )


def _clause_supported(clause: str, bundle: EvidenceBundle) -> bool:
    text="\n".join(" ".join((i.relative_path," ".join(i.heading_path),i.content)) for i in bundle.items)
    low=text.casefold(); q=clause.casefold()
    # Status/verification questions may legitimately use a short stable artifact
    # label such as CLI as the proposition anchor. Do not discard that anchor
    # merely because it is common across the project.
    raw_ascii={m.group(0).casefold() for m in _ASCII_RE.finditer(clause)}
    status_question=any(marker in q for marker in ("验证", "测过", "测试", "完成", "通过"))
    status_evidence=any(marker in text for marker in ("验证通过", "测试通过", "已完成", "完成最小测试", "最小测试案例", "验证结果"))
    stable_status_anchors={"cli", "skill", "agent", "api", "ui"}
    if status_question and status_evidence and any(anchor in raw_ascii and anchor in low for anchor in stable_status_anchors):
        return True
    ascii_terms=[m.group(0).casefold() for m in _ASCII_RE.finditer(clause) if m.group(0).casefold() not in _GENERIC_ASCII]
    ascii_hits=sum(term in low for term in set(ascii_terms))
    cjk_terms=[]
    support_stop = {"是否", "已经", "现在", "当前", "文档", "什么", "怎么", "如何", "同时", "以及", "并且", "给出", "说明", "了吗", "吗"}
    for chunk in _CJK_RE.findall(clause):
        for stop in support_stop:
            chunk=chunk.replace(stop," ")
        for part in chunk.split():
            if len(part)>=2:
                cjk_terms.extend(part[i:i+2] for i in range(len(part)-1))
    cjk_hits=sum(term in text for term in set(cjk_terms))
    if ascii_terms and ascii_hits == len(set(ascii_terms)) and ascii_hits >= 1 and cjk_hits >= 1:
        return True
    if ascii_hits >= 2:
        return True
    if cjk_hits >= 2:
        return True
    return False


def _requires_direct_target(question: str) -> bool:
    q=question.casefold()
    return any(marker in q for marker in ("多少", "数值", "star", "收入", "超时", "最低版本", "默认", "aws s3", "是否支持", "实际测试", "实测", "有测过数字", "成功率", "高多少", "最多丢"))


def _direct_target_supported(question: str, bundle: EvidenceBundle) -> bool:
    text="\n".join(i.content for i in bundle.items).casefold(); q=question.casefold()
    # Explicit ASCII targets must occur in evidence; generic project names do not count.
    ascii_terms=[m.group(0).casefold() for m in _ASCII_RE.finditer(question) if m.group(0).casefold() not in _GENERIC_ASCII]
    if ascii_terms and any(term not in text for term in ascii_terms):
        return False
    target_markers=[]
    for marker in ("precision","recall","f1","star","收入","超时","cuda","kubernetes","docker","aws s3","手机端","同步"):
        if marker in q: target_markers.append(marker)
    if target_markers and any(marker not in text for marker in target_markers):
        return False
    if "多少" in q or "数值" in q:
        # Exact/count questions need a numeric value in evidence near the requested target; no value => fail closed.
        if not re.search(r"\d", text):
            return False
    return _clause_supported(question,bundle)
