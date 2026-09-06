from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
import math

from opk_rag.embedding.qwen import _select_device
from opk_rag.reranking.config import RerankerConfig, RerankerExecutionScope
from opk_rag.reranking.provider import RerankerInferenceError, RerankerModelLoadError, RerankerPairMetadata


QWEN3_RERANKER_0_6B = "Qwen/Qwen3-Reranker-0.6B"
QWEN3_RERANKER_4B = "Qwen/Qwen3-Reranker-4B"
QWEN3_RERANKER_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class Qwen3LocalRerankerProvider:
    def __init__(self, config: RerankerConfig) -> None:
        if config.provider != "local_qwen3_cross_encoder":
            raise RerankerModelLoadError(
                f"Qwen3LocalRerankerProvider requires provider='local_qwen3_cross_encoder': {config.provider}"
            )
        if config.model_name not in {QWEN3_RERANKER_0_6B, QWEN3_RERANKER_4B}:
            raise RerankerModelLoadError(f"unsupported Qwen3 reranker model: {config.model_name}")
        self.config = config
        self.device = _select_device(config.device)

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

    @cached_property
    def _model(self):
        try:
            from sentence_transformers import CrossEncoder
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RerankerModelLoadError(
                "sentence-transformers is required for local Qwen3 reranking. Install project dependencies first."
            ) from exc
        try:
            return CrossEncoder(
                self.config.model_name,
                revision=self.config.model_revision,
                max_length=self.config.max_pair_tokens,
                device=self.device,
                cache_folder=self.config.cache_dir_string,
                local_files_only=self.config.local_files_only,
                trust_remote_code=False,
            )
        except Exception as exc:  # pragma: no cover - exercised by real model benchmark
            raise RerankerModelLoadError(
                f"Failed to load Qwen3 reranker model {self.config.model_name}@{self.config.model_revision or 'default'} "
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
            scores = self._model.predict(pairs, batch_size=self.config.batch_size)
        except RuntimeError as exc:
            raise RerankerInferenceError(_runtime_error_message(exc, self.device)) from exc
        except Exception as exc:
            raise RerankerInferenceError(f"Qwen3 reranker inference failed on device {self.device}.") from exc
        values = [float(value) for value in scores]
        if len(values) != len(documents):
            raise RerankerInferenceError(f"Expected {len(documents)} Qwen3 reranker scores, got {len(values)}.")
        if any(not math.isfinite(value) for value in values):
            raise RerankerInferenceError("Qwen3 reranker returned a non-finite score.")
        return tuple(values)

    def count_pair_tokens(self, query: str, document: str) -> int:
        return self.prepare_pair_metadata(query, document).pair_token_count

    def prepare_pair_metadata(self, query: str, document: str) -> RerankerPairMetadata:
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            raise RerankerModelLoadError("Loaded Qwen3 CrossEncoder model does not expose a tokenizer.")
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
            raise RerankerInferenceError("Qwen3 tokenizer failed to encode reranker pair.") from exc
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
        return f"Qwen3 reranker inference ran out of memory on {device}; reduce batch size or use the 0.6B model."
    return f"Qwen3 reranker inference failed on device {device}."
