"""Meta Llama (and other open-weight models) via Together AI's
OpenAI-compatible endpoint. Meta ships no first-party API, so a hosting
provider is required; Together is the default here.
"""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleAdapter


class TogetherAdapter(OpenAICompatibleAdapter):
    provider_key = "together"
    api_key_env = "TOGETHER_API_KEY"
    base_url = "https://api.together.xyz/v1"
