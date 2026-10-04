"""Integration test against the live local Ollama instance.

NOTE:
This test is intentionally separated from unit tests because it requires a
running Ollama daemon at http://localhost:11434 with models installed.

Run independently via:
    python tests/integration/test_ollama_live.py
or
    python -m unittest tests/integration/test_ollama_live.py
"""

import sys
import unittest
import urllib.request
from pathlib import Path

# Add src to python path for standalone runs
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sovereign_workbench.models import (
    ChatMessage,
    ModelAdapter,
    ModelResponse,
    ModelsConfig,
    OllamaAdapter,
)


def is_ollama_online(base_url: str = "http://localhost:11434") -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as resp:
            return resp.getcode() == 200
    except Exception:
        return False


class TestOllamaLiveIntegration(unittest.TestCase):
    """Verifies end-to-end communication with real local Ollama using ModelAdapter."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = PROJECT_ROOT / "config" / "models.json"
        cls.config = ModelsConfig.from_file(cls.config_path)
        cls.provider_cfg = cls.config.providers["ollama"]
        cls.adapter: ModelAdapter = OllamaAdapter(
            base_url=cls.provider_cfg.base_url,
            default_timeout=cls.provider_cfg.timeout_seconds,
        )

        if not is_ollama_online(cls.provider_cfg.base_url):
            raise unittest.SkipTest(
                f"Local Ollama server is not reachable at {cls.provider_cfg.base_url}."
            )

    def test_live_generation_via_model_adapter_interface(self):
        """Invoke non-streaming generation through the generic ModelAdapter interface."""
        # Use gemma3:latest as the primary model
        target_model = "gemma3:latest"
        self.assertIn(target_model, self.config.models)

        messages = [
            ChatMessage(role="system", content="You are a concise industrial AI assistant."),
            ChatMessage(role="user", content="Respond in exactly one short sentence: What is predictive maintenance?"),
        ]

        # Application calls ModelAdapter.generate without any Ollama-specific code
        response: ModelResponse = self.adapter.generate(
            messages=messages,
            model=target_model,
            options={"temperature": 0.1},
        )

        self.assertIsInstance(response, ModelResponse)
        self.assertIsInstance(response.content, str)
        self.assertTrue(len(response.content.strip()) > 0)
        self.assertEqual(response.role, "assistant")
        print(f"\n[LIVE TEST RESULT - {target_model}]: {response.content.strip()}\n")


if __name__ == "__main__":
    unittest.main()
