"""Models module exposing adapter abstractions and implementations."""

from sovereign_workbench.models.base import ModelAdapter
from sovereign_workbench.models.config import ModelMetadata, ModelsConfig, ProviderConfig
from sovereign_workbench.models.ollama import OllamaAdapter
from sovereign_workbench.models.types import (
    ChatMessage,
    ModelAdapterError,
    ModelResponse,
    ModelResponseError,
    ModelServiceUnavailableError,
    ModelTimeoutError,
)

__all__ = [
    "ModelAdapter",
    "OllamaAdapter",
    "ChatMessage",
    "ModelResponse",
    "ModelAdapterError",
    "ModelServiceUnavailableError",
    "ModelTimeoutError",
    "ModelResponseError",
    "ModelsConfig",
    "ModelMetadata",
    "ProviderConfig",
]
