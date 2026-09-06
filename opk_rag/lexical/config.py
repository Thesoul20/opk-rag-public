from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

LEXICAL_INDEX_SCHEMA_VERSION = "lexical-index-v1"
DEFAULT_BM25_K1 = 1.2
DEFAULT_BM25_B = 0.75
DEFAULT_STOPWORD_VERSION = "opk-rag-stopwords-v1"
DEFAULT_HEADING_STRATEGY = "heading_path_plus_content_unweighted"


@dataclass(frozen=True)
class LexicalIndexConfig:
    tokenizer_id: str = "jieba"
    tokenizer_version: str = "0.42.1"
    tokenizer_mode: str = "accurate"
    unicode_normalization: str = "NFKC"
    newline_normalization: str = "CRLF_CR_TO_LF"
    ascii_case: str = "lower"
    stopword_version: str = DEFAULT_STOPWORD_VERSION
    heading_strategy: str = DEFAULT_HEADING_STRATEGY
    schema_version: str = LEXICAL_INDEX_SCHEMA_VERSION
    bm25_k1: float = DEFAULT_BM25_K1
    bm25_b: float = DEFAULT_BM25_B


def build_lexical_configuration_fingerprint(config: LexicalIndexConfig) -> str:
    payload = json.dumps(asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
