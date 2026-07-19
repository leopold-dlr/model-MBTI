"""Mistral adapter via the OpenAI-compatible endpoint."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleAdapter


class MistralAdapter(OpenAICompatibleAdapter):
    provider_key = "mistral"
    api_key_env = "MISTRAL_API_KEY"
    base_url = "https://api.mistral.ai/v1"
