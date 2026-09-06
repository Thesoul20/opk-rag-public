from __future__ import annotations

RERANKER_INPUT_TEMPLATE_VERSION = "reranker-pair-v1"
RERANKER_SCORE_SEMANTICS = "raw_cross_encoder_logit_higher_is_more_relevant_not_probability"


def render_reranker_document(heading_path: tuple[str, ...], content: str) -> str:
    heading = " > ".join(part.strip() for part in heading_path if part.strip())
    body = content.strip()
    if heading and body:
        return f"{heading}\n\n{body}"
    return heading or body
