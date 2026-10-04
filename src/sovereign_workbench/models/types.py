"""Domain types and exceptions for model adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class ModelAdapterError(Exception):
    """Base exception for all model adapter errors."""
    pass


class ModelServiceUnavailableError(ModelAdapterError):
    """Raised when the model backend service (e.g. Ollama) is unreachable or offline."""
    pass


class ModelTimeoutError(ModelAdapterError):
    """Raised when a request to the model service times out."""
    pass


class ModelResponseError(ModelAdapterError):
    """Raised when the model service returns an HTTP error or malformed payload."""
    pass


@dataclass(frozen=True)
class ChatMessage:
    """Represents a single chat message."""
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ModelResponse:
    """Represents the standardized output from a model adapter."""
    content: str
    model: str
    role: str = "assistant"
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)
