"""Ollama model adapter implementation."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Sequence

from sovereign_workbench.models.base import ModelAdapter
from sovereign_workbench.models.types import (
    ChatMessage,
    ModelResponse,
    ModelResponseError,
    ModelServiceUnavailableError,
    ModelTimeoutError,
)


class OllamaAdapter(ModelAdapter):
    """Adapter communicating with Ollama's local HTTP API (/api/chat).

    Features:
    - Direct HTTP communication using standard library urllib (no external dependencies).
    - Enforces non-streaming generation for Milestone 1.
    - Translates network failures, HTTP errors, timeouts, and malformed schemas into clean domain exceptions.
    """

    def __init__(self, base_url: str = "http://localhost:11434", default_timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.default_timeout = default_timeout

    def generate(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Call Ollama /api/chat endpoint to generate a response.

        Args:
            messages: List or tuple of ChatMessage objects.
            model: Name of the model (e.g. 'gemma3:latest', 'qwen3:8b').
            options: Optional inference options (e.g. temperature, seed, num_predict).
            **kwargs: Extra arguments passed to request (such as request-specific timeout).
        """
        if not messages:
            raise ValueError("Messages list cannot be empty.")
        if not model or not model.strip():
            raise ValueError("Model name must be provided.")

        timeout = kwargs.get("timeout", self.default_timeout)
        endpoint = f"{self.base_url}/api/chat"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [msg.to_dict() for msg in messages],
            "stream": False,
        }

        if options:
            payload["options"] = options

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.getcode()
                raw_bytes = response.read()

        except urllib.error.HTTPError as exc:
            error_detail = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
                error_detail = f": {error_body}"
            except Exception:
                pass
            raise ModelResponseError(
                f"Ollama HTTP error {exc.code} for endpoint '{endpoint}'{error_detail}"
            ) from exc

        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise ModelTimeoutError(
                    f"Ollama request to '{endpoint}' timed out after {timeout}s."
                ) from exc
            raise ModelServiceUnavailableError(
                f"Failed to connect to Ollama at '{self.base_url}': {exc.reason}"
            ) from exc

        except (socket.timeout, TimeoutError) as exc:
            raise ModelTimeoutError(
                f"Ollama request to '{endpoint}' timed out after {timeout}s."
            ) from exc

        except Exception as exc:
            raise ModelResponseError(f"Unexpected connection error with Ollama: {exc}") from exc

        # Parse JSON response
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ModelResponseError(
                f"Malformed JSON response from Ollama at '{endpoint}': {exc}"
            ) from exc

        # Extract message content
        try:
            message_obj = parsed["message"]
            content = message_obj["content"]
            role = message_obj.get("role", "assistant")
        except (KeyError, TypeError) as exc:
            raise ModelResponseError(
                f"Unexpected response schema from Ollama: missing 'message.content'. Response was: {parsed}"
            ) from exc

        return ModelResponse(
            content=content,
            model=parsed.get("model", model),
            role=role,
            raw=parsed,
        )
