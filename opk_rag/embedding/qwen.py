from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

from opk_rag.embedding.config import EmbeddingConfig
from opk_rag.embedding.provider import EmbeddingInferenceError, EmbeddingModelLoadError
from opk_rag.embedding.query import QUERY_INSTRUCTION, render_query_input


class QwenLocalEmbeddingProvider:
    _instance_count = 0
    _model_load_count = 0

    def __init__(self, config: EmbeddingConfig) -> None:
        if config.provider != "local_qwen":
            raise EmbeddingModelLoadError(f"QwenLocalEmbeddingProvider requires provider='local_qwen': {config.provider}")
        self.config = config
        self.device = _select_device(config.device)
        QwenLocalEmbeddingProvider._instance_count += 1

    @property
    def model_id(self) -> str:
        return self.config.model_name

    @property
    def dimension(self) -> int:
        return self.config.dimension

    @cached_property
    def _model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise EmbeddingModelLoadError(
                "sentence-transformers is required for local Qwen embeddings. "
                "Install project dependencies before running real model indexing."
            ) from exc

        try:
            QwenLocalEmbeddingProvider._model_load_count += 1
            return SentenceTransformer(
                self.config.model_name,
                revision=self.config.model_revision,
                cache_folder=self.config.cache_dir_string,
                device=self.device,
                local_files_only=self.config.local_files_only,
            )
        except Exception as exc:  # pragma: no cover - exercised by smoke tests with real model/env
            raise EmbeddingModelLoadError(
                f"Failed to load embedding model {self.config.model_name}@{self.config.model_revision or 'default'} "
                f"on device {self.device}."
            ) from exc

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return ()
        try:
            vectors = self._model.encode(
                list(texts),
                batch_size=len(texts),
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except RuntimeError as exc:
            raise EmbeddingInferenceError(_runtime_error_message(exc, self.device)) from exc
        except Exception as exc:
            raise EmbeddingInferenceError(f"Embedding inference failed on device {self.device}.") from exc
        return vectors.tolist()

    def embed_query(self, query: str, *, instruction: str | None = None) -> Sequence[float]:
        text = render_query_input(query, instruction=instruction or QUERY_INSTRUCTION)
        vectors = self.embed_documents([text])
        if len(vectors) != 1:
            raise EmbeddingInferenceError(f"Expected one query embedding vector, got {len(vectors)}.")
        return vectors[0]

    def count_tokens(self, text: str) -> int:
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            raise EmbeddingModelLoadError("Loaded SentenceTransformer model does not expose a tokenizer.")
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=True)
        except Exception as exc:
            raise EmbeddingInferenceError("Tokenizer failed to encode embedding input.") from exc
        return len(token_ids)

    @classmethod
    def resource_counters(cls) -> dict[str, int]:
        return {
            "embedding_provider_instance_count": cls._instance_count,
            "embedding_model_load_count": cls._model_load_count,
        }

    @classmethod
    def reset_resource_counters(cls) -> None:
        cls._instance_count = 0
        cls._model_load_count = 0


def _select_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ModuleNotFoundError as exc:
        if requested == "auto":
            return "cpu"
        raise EmbeddingModelLoadError(f"Device '{requested}' requires torch to be installed.") from exc

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise EmbeddingModelLoadError("CUDA device requested but torch reports CUDA is unavailable.")
        return "cuda"

    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise EmbeddingModelLoadError("MPS device requested but torch reports MPS is unavailable.")
        return "mps"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _runtime_error_message(exc: RuntimeError, device: str) -> str:
    message = str(exc).lower()
    if "out of memory" in message or "oom" in message:
        return f"Embedding inference ran out of memory on {device}; reduce OPK_RAG_EMBEDDING_BATCH_SIZE."
    return f"Embedding inference failed on device {device}."
