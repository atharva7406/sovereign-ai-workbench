"""Unit tests for OllamaAdapter and model configuration."""

import io
import json
import socket
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sovereign_workbench.models.base import ModelAdapter
from sovereign_workbench.models.config import ModelsConfig
from sovereign_workbench.models.ollama import OllamaAdapter
from sovereign_workbench.models.types import (
    ChatMessage,
    ModelResponse,
    ModelResponseError,
    ModelServiceUnavailableError,
    ModelTimeoutError,
)


class TestOllamaAdapter(unittest.TestCase):
    """Unit tests for OllamaAdapter with mocked HTTP calls."""

    def setUp(self):
        self.adapter = OllamaAdapter(base_url="http://localhost:11434", default_timeout=5.0)
        self.messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello, world!"),
        ]

    def test_implements_interface(self):
        """Adapter must subclass ModelAdapter."""
        self.assertIsInstance(self.adapter, ModelAdapter)

    def test_input_validation(self):
        """Adapter validates non-empty messages and valid model names."""
        with self.assertRaises(ValueError):
            self.adapter.generate(messages=[], model="gemma3:latest")

        with self.assertRaises(ValueError):
            self.adapter.generate(messages=self.messages, model="")

    @patch("urllib.request.urlopen")
    def test_successful_generation(self, mock_urlopen):
        """Test successful response parsing from Ollama."""
        ollama_response_payload = {
            "model": "gemma3:latest",
            "created_at": "2026-10-04T09:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I assist you with industrial workflows today?",
            },
            "done": True,
            "total_duration": 120000000,
        }

        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = json.dumps(ollama_response_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        result = self.adapter.generate(
            messages=self.messages,
            model="gemma3:latest",
            options={"temperature": 0.2},
        )

        self.assertIsInstance(result, ModelResponse)
        self.assertEqual(result.model, "gemma3:latest")
        self.assertEqual(result.role, "assistant")
        self.assertEqual(result.content, "Hello! How can I assist you with industrial workflows today?")
        self.assertIsNotNone(result.raw)

        # Verify request structure
        mock_urlopen.assert_called_once()
        req_arg = mock_urlopen.call_args[0][0]
        self.assertEqual(req_arg.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(req_arg.get_method(), "POST")

        sent_body = json.loads(req_arg.data.decode("utf-8"))
        self.assertEqual(sent_body["model"], "gemma3:latest")
        self.assertFalse(sent_body["stream"])
        self.assertEqual(sent_body["options"], {"temperature": 0.2})
        self.assertEqual(len(sent_body["messages"]), 2)
        self.assertEqual(sent_body["messages"][0]["role"], "system")

    @patch("urllib.request.urlopen")
    def test_http_failure(self, mock_urlopen):
        """Test handling of HTTP 404 or 500 status codes."""
        error_file = io.BytesIO(b'{"error":"model not found"}')
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://localhost:11434/api/chat",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=error_file,
        )

        with self.assertRaises(ModelResponseError) as ctx:
            self.adapter.generate(messages=self.messages, model="nonexistent:model")

        self.assertIn("404", str(ctx.exception))
        self.assertIn("model not found", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_malformed_json_response(self, mock_urlopen):
        """Test handling when response body is not valid JSON."""
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = b"<html>502 Bad Gateway</html>"
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with self.assertRaises(ModelResponseError) as ctx:
            self.adapter.generate(messages=self.messages, model="gemma3:latest")

        self.assertIn("Malformed JSON", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_unexpected_response_schema(self, mock_urlopen):
        """Test handling when JSON response lacks expected keys."""
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = json.dumps({"unexpected": "structure"}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with self.assertRaises(ModelResponseError) as ctx:
            self.adapter.generate(messages=self.messages, model="gemma3:latest")

        self.assertIn("Unexpected response schema", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_connection_refused(self, mock_urlopen):
        """Test handling when Ollama daemon is offline/unreachable."""
        mock_urlopen.side_effect = urllib.error.URLError(reason=ConnectionRefusedError("Connection refused"))

        with self.assertRaises(ModelServiceUnavailableError) as ctx:
            self.adapter.generate(messages=self.messages, model="gemma3:latest")

        self.assertIn("Failed to connect", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_request_timeout_urlerror(self, mock_urlopen):
        """Test handling when connection/read times out via URLError."""
        mock_urlopen.side_effect = urllib.error.URLError(reason=socket.timeout("timed out"))

        with self.assertRaises(ModelTimeoutError) as ctx:
            self.adapter.generate(messages=self.messages, model="gemma3:latest")

        self.assertIn("timed out", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_request_timeout_socket(self, mock_urlopen):
        """Test handling when direct socket.timeout is raised."""
        mock_urlopen.side_effect = TimeoutError("The read operation timed out")

        with self.assertRaises(ModelTimeoutError) as ctx:
            self.adapter.generate(messages=self.messages, model="gemma3:latest")

        self.assertIn("timed out", str(ctx.exception))


class TestModelsConfig(unittest.TestCase):
    """Test configuration loading from models.json."""

    def test_load_models_config(self):
        import pathlib
        config_path = pathlib.Path(__file__).resolve().parents[2] / "config" / "models.json"
        config = ModelsConfig.from_file(config_path)

        self.assertEqual(config.default_provider, "ollama")
        self.assertIn("ollama", config.providers)
        self.assertEqual(config.providers["ollama"].base_url, "http://localhost:11434")

        # Check required initial models
        expected_models = ["gemma3:latest", "qwen3:8b", "aya-expanse:8b"]
        for m in expected_models:
            model_meta = config.get_model(m)
            self.assertIsNotNone(model_meta, f"Model {m} not found in configuration")
            self.assertEqual(model_meta.provider, "ollama")
            self.assertTrue(len(model_meta.display_name) > 0)


if __name__ == "__main__":
    unittest.main()
