"""Maps the `provider` key in models.yaml to a concrete adapter class.

This is the ONE place that knows about every provider. The orchestrator only
calls ``build_adapter`` and gets back something implementing the
ProviderAdapter interface.
"""

from __future__ import annotations

from typing import Any

from .anthropic_provider import AnthropicAdapter
from .base import ProviderAdapter, ProviderError
from .cohere_provider import CohereAdapter
from .deepseek_provider import DeepSeekAdapter
from .google_provider import GoogleAdapter
from .mistral_provider import MistralAdapter
from .moonshot_provider import MoonshotAdapter
from .openai_provider import OpenAIAdapter
from .qwen_provider import QwenAdapter
from .together_provider import TogetherAdapter
from .xai_provider import XAIAdapter

REGISTRY: dict[str, type[ProviderAdapter]] = {
    "anthropic": AnthropicAdapter,
    "google": GoogleAdapter,
    "openai": OpenAIAdapter,
    "mistral": MistralAdapter,
    "xai": XAIAdapter,
    "deepseek": DeepSeekAdapter,
    "moonshot": MoonshotAdapter,
    "qwen": QwenAdapter,
    "together": TogetherAdapter,
    "cohere": CohereAdapter,
}


def build_adapter(provider: str, model_id: str, params: dict[str, Any] | None = None) -> ProviderAdapter:
    try:
        cls = REGISTRY[provider]
    except KeyError as exc:
        raise ProviderError(
            f"Unknown provider '{provider}'. Known providers: {', '.join(sorted(REGISTRY))}."
        ) from exc
    return cls(model_id=model_id, params=params)
