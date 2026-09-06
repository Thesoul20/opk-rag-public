from __future__ import annotations


class AdapterError(ValueError):
    code = "adapter_error"


class UnsupportedRepresentationError(AdapterError):
    code = "unsupported_representation"


class AdapterParseError(AdapterError):
    code = "parse_failure"


class EmptyAdapterOutputError(AdapterError):
    code = "empty_parse"
