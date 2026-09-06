from __future__ import annotations

from dataclasses import dataclass
import re

_BRACKET_TOKEN_RE = re.compile(r"\[([^\[\]\n]{0,32})\]")
_VALID_CITATION_RE = re.compile(r"C[1-9][0-9]*")
_FENCED_CODE_RE = re.compile(r"(^|\n)```.*?(?:\n```|$)", re.DOTALL)


@dataclass(frozen=True)
class CitationParseResult:
    cited_ids: tuple[str, ...]
    invalid_tokens: tuple[str, ...]
    raw_matches: tuple[str, ...]


def parse_citation_markers(text: str) -> CitationParseResult:
    """Parse strict inline citation markers from answer text.

    Official answer citations are independent markers like [C1] or [C1][C2].
    Fenced code blocks are ignored so examples do not become answer citations.
    """

    scan_text = _strip_fenced_code_blocks(text)
    cited_ids: list[str] = []
    invalid_tokens: list[str] = []
    raw_matches: list[str] = []
    seen: set[str] = set()
    for match in _BRACKET_TOKEN_RE.finditer(scan_text):
        if match.end() < len(scan_text) and scan_text[match.end()] == "(":
            continue
        raw = match.group(0)
        token = match.group(1)
        if _VALID_CITATION_RE.fullmatch(token):
            raw_matches.append(raw)
            if token not in seen:
                cited_ids.append(token)
                seen.add(token)
            continue
        if _looks_like_citation_token(token):
            raw_matches.append(raw)
            invalid_tokens.append(raw)
    return CitationParseResult(
        cited_ids=tuple(cited_ids),
        invalid_tokens=tuple(dict.fromkeys(invalid_tokens)),
        raw_matches=tuple(raw_matches),
    )


def strip_citation_markers(text: str) -> str:
    return re.sub(r"\[C[1-9][0-9]*\]", "", _strip_fenced_code_blocks(text))


def _strip_fenced_code_blocks(text: str) -> str:
    return _FENCED_CODE_RE.sub("\n", text)


def _looks_like_citation_token(token: str) -> bool:
    stripped = token.strip()
    if stripped != token and _VALID_CITATION_RE.fullmatch(stripped):
        return True
    if not stripped:
        return False
    if stripped[0] in {"C", "c"}:
        return True
    return False
