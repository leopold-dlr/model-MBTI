"""Moonshot AI (Kimi) adapter via the OpenAI-compatible endpoint."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleAdapter


class MoonshotAdapter(OpenAICompatibleAdapter):
    provider_key = "moonshot"
    api_key_env = "MOONSHOT_API_KEY"
    base_url = "https://api.moonshot.ai/v1"
