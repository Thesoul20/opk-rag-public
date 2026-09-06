from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from importlib.metadata import version
import re
import string
import unicodedata
from typing import Protocol

import jieba

TECHNICAL_TOKEN_RE = re.compile(r"--[a-z0-9][a-z0-9-]*|[a-z0-9]+(?:[._/-][a-z0-9]+)*", re.IGNORECASE)
PUNCTUATION = set(string.punctuation) | {
    "，",
    "。",
    "、",
    "；",
    "：",
    "？",
    "！",
    "（",
    "）",
    "【",
    "】",
    "《",
    "》",
    "“",
    "”",
    "‘",
    "’",
    "·",
}
STOPWORDS = frozenset({"的", "了", "和", "与", "及", "或", "在"})


class LexicalTokenizer(Protocol):
    @property
    def tokenizer_id(self) -> str:
        ...

    @property
    def version(self) -> str:
        ...

    def tokenize_document(self, text: str) -> tuple[str, ...]:
        ...

    def tokenize_query(self, query: str) -> tuple[str, ...]:
        ...


class JiebaLexicalTokenizer:
    tokenizer_id = "jieba"

    @property
    def version(self) -> str:
        return version("jieba")

    def tokenize_document(self, text: str) -> tuple[str, ...]:
        return _tokenize(text)

    def tokenize_query(self, query: str) -> tuple[str, ...]:
        return _tokenize(query)


def normalize_lexical_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "".join(char.lower() if "A" <= char <= "Z" else char for char in normalized)


def term_frequencies(tokens: tuple[str, ...]) -> Mapping[str, int]:
    return dict(Counter(tokens))


def _tokenize(text: str) -> tuple[str, ...]:
    normalized = normalize_lexical_text(text)
    technical_tokens: list[str] = []

    def collect(match: re.Match[str]) -> str:
        token = _clean_token(match.group(0))
        if token:
            technical_tokens.append(token)
            technical_tokens.extend(_split_compound_token(token))
        return " "

    chinese_text = TECHNICAL_TOKEN_RE.sub(collect, normalized)
    segmented = (_clean_token(token) for token in jieba.cut(chinese_text, cut_all=False))
    tokens = [token for token in (*technical_tokens, *segmented) if _keep_token(token)]
    return tuple(tokens)


def _clean_token(token: str) -> str:
    return token.strip().strip("`'\"()[]{}<>")


def _keep_token(token: str) -> bool:
    if not token or token.isspace() or token in STOPWORDS:
        return False
    return any(char not in PUNCTUATION and not char.isspace() for char in token)


def _split_compound_token(token: str) -> tuple[str, ...]:
    if not any(separator in token for separator in ("_", "-", "/", ".")):
        return ()
    parts = [part for part in re.split(r"[_\-/\.]+", token) if _keep_token(part)]
    suffixes = [
        suffix
        for suffix in re.split(r"[/\.]+", token)[1:]
        if _keep_token(suffix) and any(separator in suffix for separator in ("_", "-"))
    ]
    return tuple(dict.fromkeys((*suffixes, *parts)))
