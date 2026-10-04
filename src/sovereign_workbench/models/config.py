"""Model configuration loader and registry for Sovereign AI Workbench."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata describing a configured model."""
    name: str
    display_name: str
    provider: str
    capabilities: List[str] = field(default_factory=list)
    context_window: Optional[int] = None
    description: str = ""


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a specific model backend provider."""
    name: str
    base_url: str
    timeout_seconds: float = 60.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelsConfig:
    """Root configuration containing all providers and available models."""
    default_provider: str
    providers: Dict[str, ProviderConfig]
    models: Dict[str, ModelMetadata]

    def get_model(self, model_name: str) -> Optional[ModelMetadata]:
        return self.models.get(model_name)

    def list_models(self) -> List[ModelMetadata]:
        return list(self.models.values())

    @classmethod
    def from_file(cls, path: str | Path) -> ModelsConfig:
        filepath = Path(path)
        if not filepath.is_file():
            raise FileNotFoundError(f"Model configuration file not found at: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        providers: Dict[str, ProviderConfig] = {}
        for prov_name, prov_data in data.get("providers", {}).items():
            providers[prov_name] = ProviderConfig(
                name=prov_name,
                base_url=prov_data.get("base_url", "http://localhost:11434"),
                timeout_seconds=float(prov_data.get("timeout_seconds", 60.0)),
                extra={k: v for k, v in prov_data.items() if k not in ("base_url", "timeout_seconds")},
            )

        models: Dict[str, ModelMetadata] = {}
        for model_name, model_info in data.get("models", {}).items():
            models[model_name] = ModelMetadata(
                name=model_name,
                display_name=model_info.get("display_name", model_name),
                provider=model_info.get("provider", "ollama"),
                capabilities=model_info.get("capabilities", []),
                context_window=model_info.get("context_window"),
                description=model_info.get("description", ""),
            )

        return cls(
            default_provider=data.get("default_provider", "ollama"),
            providers=providers,
            models=models,
        )
