from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opk_rag.core_tools.contracts import TOOL_SCHEMA_VERSION, CoreToolError, ToolDefinition, ToolName

TOOL_VERSION = "1.0.0"


@dataclass(frozen=True)
class ToolRegistry:
    definitions: tuple[ToolDefinition, ...]

    def get(self, name: str) -> ToolDefinition:
        for definition in self.definitions:
            if definition.name == name:
                return definition
        raise CoreToolError("invalid_tool_name", f"Unknown Core RAG tool: {name}")

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        return self.definitions

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TOOL_SCHEMA_VERSION,
            "tools": [definition.to_dict() for definition in self.definitions],
        }


def default_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        definitions=(
            _definition(
                "search_knowledge_base",
                "Run the existing governed retrieval service and return trusted evidence identities.",
                required=("knowledge_base_id", "query"),
                optional=("top_k", "candidate_k", "mode"),
                output_required=("search_response", "trusted_evidence"),
                sequence={"requires": [], "produces": ["search_response", "trusted_evidence"]},
            ),
            _definition(
                "expand_document_section",
                "Expand bounded context from evidence already returned by retrieval.",
                required=("trusted_evidence", "chunk_id"),
                optional=("max_items",),
                output_required=("evidence_bundle", "trusted_evidence"),
                sequence={"requires": ["trusted_evidence"], "produces": ["trusted_evidence"]},
            ),
            _definition(
                "assess_answerability",
                "Run the existing AnswerabilityPolicy over trusted current-turn evidence.",
                required=("search_response",),
                optional=(),
                output_required=("answerability",),
                sequence={"requires": ["search_response", "trusted_evidence"], "produces": ["answerability"]},
            ),
            _definition(
                "generate_grounded_answer",
                "Run existing grounded generation only after answerability permits generation.",
                required=("search_response", "answerability"),
                optional=(),
                output_required=("answer_response",),
                sequence={"requires": ["search_response", "trusted_evidence", "answerability"], "produces": ["answer_response"]},
            ),
            _definition(
                "verify_grounding",
                "Run existing grounding validation against trusted current-turn evidence.",
                required=("answer_text", "citations", "evidence_bundle"),
                optional=(),
                output_required=("grounding",),
                sequence={"requires": ["trusted_evidence"], "produces": ["grounding"]},
            ),
        )
    )


def tool_sequence_policy() -> dict[str, Any]:
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "approved_sequence": [
            "search_knowledge_base",
            "expand_document_section",
            "assess_answerability",
            "generate_grounded_answer",
            "verify_grounding",
        ],
        "rules": [
            "assess_answerability requires a SearchResponse produced by search_knowledge_base.",
            "generate_grounded_answer requires answerability.answerable=true.",
            "verify_grounding accepts only citation IDs in the current trusted evidence bundle.",
            "No tool accepts SQL, arbitrary filesystem paths, model IDs, provider endpoints, or system prompts from the caller.",
            "expand_document_section can only reuse evidence already present in trusted_evidence.",
        ],
    }


def _definition(
    name: ToolName,
    description: str,
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    output_required: tuple[str, ...],
    sequence: dict[str, Any],
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        version=TOOL_VERSION,
        description=description,
        input_schema=_object_schema(required=required, optional=optional),
        output_schema=_object_schema(required=output_required, optional=("trace",)),
        sequence=sequence,
    )


def _object_schema(*, required: tuple[str, ...], optional: tuple[str, ...]) -> dict[str, Any]:
    properties = {name: {"description": f"{name} contract field"} for name in (*required, *optional)}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": TOOL_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": properties,
    }
