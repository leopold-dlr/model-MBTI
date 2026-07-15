"""OpenAI (GPT) adapter -- the reference OpenAI-compatible endpoint."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleAdapter


class OpenAIAdapter(OpenAICompatibleAdapter):
    provider_key = "openai"
    api_key_env = "OPENAI_API_KEY"
    base_url = None  # default api.openai.com
