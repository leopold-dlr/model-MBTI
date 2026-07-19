"""OpenAI (GPT) adapter -- the reference OpenAI-compatible endpoint."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleAdapter


class OpenAIAdapter(OpenAICompatibleAdapter):
    provider_key = "openai"
    api_key_env = "OPENAI_API_KEY"
    base_url = None  # default api.openai.com
    # gpt-5 / o-series reject `max_tokens` on /chat/completions; they require
    # `max_completion_tokens`. This also covers reasoning-token spend, which
    # otherwise silently eats the whole budget before any visible output.
    max_tokens_param = "max_completion_tokens"
