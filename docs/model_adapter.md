# Milestone 1: Provider-Agnostic Model Adapter

## Overview
This document outlines the architecture, rationale, and configuration model for the `ModelAdapter` interface in the **Sovereign AI Workbench**.

The workbench is an on-premise, air-gapped system designed for confidential industrial workflows. The core principle of Milestone 1 is to ensure zero vendor/runtime lock-in to Ollama while maintaining zero external dependencies.

---

## Architecture

```
Application Layer (Business Logic / Industrial Workflows)
                        │
                        ▼
      [ModelAdapter (ABC Interface)]
      - generate(messages, model, options, **kwargs) -> ModelResponse
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
  [OllamaAdapter]               [vLLMAdapter (Future)]
         │                             │
         ▼ (HTTP POST /api/chat)       ▼ (HTTP POST /v1/chat/completions)
  Local Ollama Daemon           Local vLLM Server
```

---

## Key Design Questions

### 1. Why does `ModelAdapter` exist?
Application features (such as industrial report analysis, confidential telemetry extraction, and tool execution) should never be tightly coupled to a specific inference engine.

By establishing an abstract base class (`ModelAdapter`), the workbench ensures:
- **Zero Coupling**: Calling code is unaware of transport mechanisms (Ollama `/api/chat`, vLLM OpenAI-compatible endpoints, TensorRT-LLM, llama.cpp server, etc.).
- **Swappability**: The backend inference server can be substituted without changing downstream application logic.
- **Unified Contracts**: Inputs (`ChatMessage`) and outputs (`ModelResponse`) use strict, standardized dataclasses across the entire system.

### 2. Why is Ollama placed behind an adapter?
- **Portability**: Ollama is well suited for rapid local deployment, but production enterprise deployments in air-gapped facilities may require higher-throughput servers (e.g., vLLM or Triton).
- **Domain Exception Translation**: Ollama's HTTP/socket error codes are caught and translated into system-wide domain exceptions (`ModelServiceUnavailableError`, `ModelTimeoutError`, `ModelResponseError`).
- **Standardized Payload Shaping**: Ollama's proprietary `/api/chat` payload structure is encapsulated entirely inside `OllamaAdapter`.

### 3. How a future provider (e.g., vLLM) could be added
Adding a new provider such as `vLLM` requires two simple steps:

1. **Implement `ModelAdapter`**:
   Create `src/sovereign_workbench/models/vllm.py`:
   ```python
   from sovereign_workbench.models.base import ModelAdapter
   from sovereign_workbench.models.types import ChatMessage, ModelResponse

   class VLLMAdapter(ModelAdapter):
       def __init__(self, base_url: str = "http://localhost:8000"):
           self.base_url = base_url

       def generate(self, messages, model, *, options=None, **kwargs) -> ModelResponse:
           # Communicate with vLLM's /v1/chat/completions endpoint
           ...
   ```

2. **Register provider & models in `config/models.json`**:
   ```json
   "providers": {
     "vllm": {
       "base_url": "http://localhost:8000",
       "timeout_seconds": 60.0
     }
   },
   "models": {
     "qwen-industrial-finetuned": {
       "display_name": "Qwen Industrial Finetuned",
       "provider": "vllm",
       "capabilities": ["text", "tools"]
     }
   }
   ```

No application code or existing adapters need modification.

### 4. How models are configured
Model metadata and endpoints are separated from source code in `config/models.json`:
- **Provider definitions**: Host URL, timeout settings.
- **Model specifications**: Model identifier, human-readable display name, capabilities (`text`, `multimodal`, `tools`, `thinking`), context window length, and description.

The `ModelsConfig` loader (`src/sovereign_workbench/models/config.py`) parses this configuration and exposes helper methods to query available models and configurations.
