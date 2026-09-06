from __future__ import annotations

from opk_rag.document_adapters.errors import UnsupportedRepresentationError
from opk_rag.document_adapters.html import HTMLDocumentAdapter
from opk_rag.document_adapters.markdown import MarkdownDocumentAdapter
from opk_rag.document_adapters.pdf import PdfDocumentAdapter
from opk_rag.document_adapters.tex import TeXDocumentAdapter
from opk_rag.document_ir.serialization import normalize_representation_type


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters = {
            "markdown": MarkdownDocumentAdapter(),
            "html": HTMLDocumentAdapter(),
            "pdf": PdfDocumentAdapter(),
            "tex": TeXDocumentAdapter(),
        }

    def get(self, representation_type: str):
        normalized = normalize_representation_type(representation_type)
        adapter = self._adapters.get(normalized)
        if adapter is None:
            raise UnsupportedRepresentationError(f"unsupported representation: {representation_type}")
        return adapter

    def adapt(self, source):
        return self.get(source.representation_type).adapt(source)

    def supported_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


def default_adapter_registry() -> AdapterRegistry:
    return AdapterRegistry()
