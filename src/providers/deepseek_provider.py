"""DeepSeek adapter via the OpenAI-compatible endpoint."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleAdapter


class DeepSeekAdapter(OpenAICompatibleAdapter):
    provider_key = "deepseek"
    api_key_env = "DEEPSEEK_API_KEY"
    base_url = "https://api.deepseek.com"
