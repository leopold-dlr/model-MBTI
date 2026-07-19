"""xAI (Grok) adapter via the OpenAI-compatible endpoint."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleAdapter


class XAIAdapter(OpenAICompatibleAdapter):
    provider_key = "xai"
    api_key_env = "XAI_API_KEY"
    base_url = "https://api.x.ai/v1"
