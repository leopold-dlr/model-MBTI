"""Cohere (Command R+) via its OpenAI-compatibility endpoint."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleAdapter


class CohereAdapter(OpenAICompatibleAdapter):
    provider_key = "cohere"
    api_key_env = "COHERE_API_KEY"
    base_url = "https://api.cohere.ai/compatibility/v1"
