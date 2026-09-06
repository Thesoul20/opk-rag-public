from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
import math
from pathlib import Path
from typing import Any

from opk_rag.embedding.qwen import _select_device
from opk_rag.reranking.config import RerankerConfig, RerankerExecutionScope
from opk_rag.reranking.provider import RerankerInferenceError, RerankerModelLoadError, RerankerPairMetadata


ZERANK2_RERANKER = "zeroentropy/zerank-2-reranker"
ZERANK2_RERANKER_REVISION = "5eae30d5ee3c6b2df2ef6d723bde45172d761c4c"
ZERANK_SCORE_SEMANTICS = {
    "raw_score_type": "float",
    "raw_score_shape": "one scalar per query-document pair",
    "higher_is_better": True,
    "score_transform_applied": False,
    "ranking_order": "descending(raw_relevance_score)",
}


class Zerank2LocalRerankerProvider:
    def __init__(self, config: RerankerConfig, *, local_model_path: Path | str | None = None) -> None:
        if config.provider != "local_zerank2_cross_encoder":
            raise RerankerModelLoadError(
                f"Zerank2LocalRerankerProvider requires provider='local_zerank2_cross_encoder': {config.provider}"
            )
        if config.model_name != ZERANK2_RERANKER:
            raise RerankerModelLoadError(f"unsupported Zerank reranker model: {config.model_name}")
        self.config = config
        self.device = _select_device(config.device)
        self.local_model_path = Path(local_model_path).resolve() if local_model_path is not None else None

    @property
    def model_id(self) -> str:
        return self.config.model_name

    @property
    def model_revision(self) -> str | None:
        return self.config.model_revision

    @property
    def input_template_version(self) -> str:
        return self.config.input_template_version

    @property
    def max_pair_tokens(self) -> int:
        return self.config.max_pair_tokens

    @property
    def score_semantics(self) -> dict[str, Any]:
        return dict(ZERANK_SCORE_SEMANTICS)

    @property
    def actual_model_device(self) -> str | None:
        try:
            return str(next(self._model.model.parameters()).device)
        except Exception:
            return None

    @property
    def actual_model_dtype(self) -> str | None:
        try:
            return str(next(self._model.model.parameters()).dtype)
        except Exception:
            return None

    @property
    def model_kwargs(self) -> dict[str, Any]:
        if self.device == "cuda":
            try:
                import torch
            except ModuleNotFoundError:
                return {}
            return {"torch_dtype": torch.bfloat16}
        return {}

    @cached_property
    def _model(self):
        try:
            from sentence_transformers import CrossEncoder
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RerankerModelLoadError(
                "sentence-transformers is required for local Zerank reranking. Install project dependencies first."
            ) from exc
        try:
            model_name_or_path = str(self.local_model_path) if self.local_model_path is not None else self.config.model_name
            model = CrossEncoder(
                model_name_or_path,
                revision=None if self.local_model_path is not None else self.config.model_revision,
                max_length=self.config.max_pair_tokens,
                device=self.device,
                cache_folder=self.config.cache_dir_string,
                local_files_only=True if self.local_model_path is not None else self.config.local_files_only,
                trust_remote_code=False,
                model_kwargs=self.model_kwargs,
            )
            model.model.eval()
            return model
        except Exception as exc:  # pragma: no cover - exercised by real model benchmark
            raise RerankerModelLoadError(
                f"Failed to load Zerank model {self.config.model_name}@{self.config.model_revision or 'default'} "
                f"on device {self.device}."
            ) from exc

    def score_pairs(
        self,
        query: str,
        documents: Sequence[str],
        *,
        execution_scope: RerankerExecutionScope = "search",
    ) -> Sequence[float]:
        if not documents:
            return ()
        pairs = [(query, document) for document in documents]
        try:
            try:
                import torch
            except ModuleNotFoundError:
                torch = None
            if torch is None:
                scores = self._model.predict(pairs, batch_size=self.config.batch_size, show_progress_bar=False)
            else:
                with torch.inference_mode():
                    scores = self._model.predict(pairs, batch_size=self.config.batch_size, show_progress_bar=False)
        except RuntimeError as exc:
            raise RerankerInferenceError(_runtime_error_message(exc, self.device)) from exc
        except Exception as exc:
            raise RerankerInferenceError(f"Zerank reranker inference failed on device {self.device}.") from exc
        values = [float(value) for value in scores]
        if len(values) != len(documents):
            raise RerankerInferenceError(f"Expected {len(documents)} Zerank scores, got {len(values)}.")
        if any(not math.isfinite(value) for value in values):
            raise RerankerInferenceError("Zerank reranker returned a non-finite score.")
        return tuple(values)

    def count_pair_tokens(self, query: str, document: str) -> int:
        return self.prepare_pair_metadata(query, document).pair_token_count

    def prepare_pair_metadata(self, query: str, document: str) -> RerankerPairMetadata:
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            raise RerankerModelLoadError("Loaded Zerank CrossEncoder model does not expose a tokenizer.")
        try:
            original_ids = tokenizer.encode(query, document, add_special_tokens=True, truncation=False)
            truncated_ids = tokenizer.encode(
                query,
                document,
                add_special_tokens=True,
                truncation=True,
                max_length=self.config.max_pair_tokens,
            )
        except Exception as exc:
            raise RerankerInferenceError("Zerank tokenizer failed to encode reranker pair.") from exc
        original_count = len(original_ids)
        pair_count = len(truncated_ids)
        return RerankerPairMetadata(
            original_pair_token_count=original_count,
            pair_token_count=pair_count,
            input_truncated=original_count > pair_count,
        )


def _runtime_error_message(exc: RuntimeError, device: str) -> str:
    message = str(exc).lower()
    if "out of memory" in message or "oom" in message:
        return f"Zerank reranker inference ran out of memory on {device}; retry with batch_size=1."
    return f"Zerank reranker inference failed on device {device}."
