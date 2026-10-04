"""Abstract Base Class for provider-agnostic model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

from sovereign_workbench.models.types import ChatMessage, ModelResponse


class ModelAdapter(ABC):
    """Abstract interface defining the model generation contract.

    Application code depends solely on this interface, remaining decoupled
    from the specific provider (e.g., Ollama, vLLM, or other local inference engines).
    """

    @abstractmethod
    def generate(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a non-streaming response for the provided chat messages.

        Args:
            messages: A sequence of ChatMessage objects (e.g. system, user, assistant).
            model: The identifier of the model to execute.
            options: Optional inference parameters (e.g., temperature, top_p, num_ctx).
            **kwargs: Additional provider-specific parameters.

        Returns:
            ModelResponse containing the generated response content and model identifier.

        Raises:
            ModelServiceUnavailableError: If the backend service cannot be contacted.
            ModelTimeoutError: If the request times out.
            ModelResponseError: If the service returns an HTTP error or malformed payload.
        """
        raise NotImplementedError
