from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
import math
from typing import Any

from opk_rag.embedding.qwen import _select_device
from opk_rag.reranking.config import RerankerConfig, RerankerExecutionScope, RerankerPrecision
from opk_rag.reranking.provider import RerankerInferenceError, RerankerModelLoadError, RerankerPairMetadata


class BgeLocalRerankerProvider:
    def __init__(self, config: RerankerConfig) -> None:
        if config.provider != "local_bge_cross_encoder":
            raise RerankerModelLoadError(
                f"BgeLocalRerankerProvider requires provider='local_bge_cross_encoder': {config.provider}"
        )
        self.config = config
        self.device = _select_device(config.device)
        self._initialization_audit_emitted = False
        self._last_forward_audit: dict[str, Any] | None = None

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
                "sentence-transformers is required for local reranking. Install project dependencies first."
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
        except Exception as exc:  # pragma: no cover - exercised by real model smoke tests
            raise RerankerModelLoadError(
                f"Failed to load reranker model {self.config.model_name}@{self.config.model_revision or 'default'} "
                f"on device {self.device}."
            ) from exc

    @property
    def initialization_audit(self) -> dict[str, Any]:
        model = self._model
        parameter = next(model.model.parameters(), None)
        parameter_device = str(parameter.device) if parameter is not None else "unknown"
        parameter_dtype = str(parameter.dtype) if parameter is not None else "unknown"
        return {
            "reranker_model": self.config.model_name,
            "requested_device": self.config.device,
            "resolved_device": self.device,
            "configured_search_precision": self.config.search_precision,
            "configured_ask_precision": self.config.ask_precision,
            "model_parameter_device": parameter_device,
            "model_parameter_dtype": parameter_dtype,
            "autocast_enabled_for_search": self.config.search_precision == "fp16_autocast" and self.device.startswith("cuda"),
            "autocast_dtype_for_search": "torch.float16" if self.config.search_precision == "fp16_autocast" else None,
            "autocast_enabled_for_ask": self.config.ask_precision == "fp16_autocast" and self.device.startswith("cuda"),
            "fallback_policy": "fail_closed",
        }

    @property
    def last_forward_audit(self) -> dict[str, Any] | None:
        return self._last_forward_audit

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
            scores = self._predict_tokenized_pairs(pairs, execution_scope=execution_scope)
        except RerankerInferenceError:
            raise
        except RuntimeError as exc:
            raise RerankerInferenceError(_runtime_error_message(exc, self.device)) from exc
        except Exception as exc:
            raise RerankerInferenceError(f"Reranker inference failed on device {self.device}.") from exc
        values = [float(value) for value in scores]
        if len(values) != len(documents):
            raise RerankerInferenceError(f"Expected {len(documents)} reranker scores, got {len(values)}.")
        if any(not math.isfinite(value) for value in values):
            raise RerankerInferenceError("Reranker returned a non-finite score.")
        return tuple(values)

    def _predict_tokenized_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        execution_scope: RerankerExecutionScope,
    ) -> tuple[float, ...]:
        import torch

        model = self._model
        model.eval()
        precision = self._precision_for_scope(execution_scope)
        autocast_enabled = precision == "fp16_autocast"
        if autocast_enabled and not self.device.startswith("cuda"):
            raise RerankerInferenceError(
                "Reranker precision fp16_autocast requires CUDA; fail-closed prevents silent FP32 fallback."
            )
        scores: list[float] = []
        effective_forward_dtype = "torch.float32"
        for start in range(0, len(pairs), self.config.batch_size):
            batch = list(pairs[start : start + self.config.batch_size])
            features = model.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_pair_tokens,
                return_tensors="pt",
            )
            features.to(model.model.device)
            with torch.inference_mode():
                if autocast_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions = model.model(**features, return_dict=True)
                        logits = model.activation_fn(predictions.logits)
                else:
                    predictions = model.model(**features, return_dict=True)
                    logits = model.activation_fn(predictions.logits)
                effective_forward_dtype = str(logits.dtype)
            values = logits.detach().cpu().reshape(-1).tolist()
            scores.extend(float(value) for value in values)
        self._last_forward_audit = {
            "execution_scope": execution_scope,
            "configured_precision": precision,
            "autocast_enabled": autocast_enabled,
            "autocast_dtype": "torch.float16" if autocast_enabled else None,
            "effective_forward_dtype": effective_forward_dtype,
            "silent_fp32_fallback_detected": autocast_enabled and effective_forward_dtype == "torch.float32",
            "inference_mode_active": True,
            "model_eval_mode_active": not model.model.training,
        }
        if self._last_forward_audit["silent_fp32_fallback_detected"]:
            raise RerankerInferenceError("Reranker fp16_autocast silently fell back to FP32.")
        return tuple(scores)

    def _precision_for_scope(self, execution_scope: RerankerExecutionScope) -> RerankerPrecision:
        if execution_scope == "search":
            return self.config.search_precision
        if execution_scope == "ask":
            return self.config.ask_precision
        raise RerankerInferenceError(f"Unsupported reranker execution scope: {execution_scope}")

    def count_pair_tokens(self, query: str, document: str) -> int:
        return self.prepare_pair_metadata(query, document).pair_token_count

    def prepare_pair_metadata(self, query: str, document: str) -> RerankerPairMetadata:
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            raise RerankerModelLoadError("Loaded CrossEncoder model does not expose a tokenizer.")
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
            raise RerankerInferenceError("Tokenizer failed to encode reranker pair.") from exc
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
        return f"Reranker inference ran out of memory on {device}; reduce OPK_RAG_RERANKER_BATCH_SIZE."
    return f"Reranker inference failed on device {device}."
