"""OpenRouter adapter -- one key, (almost) every model.

OpenRouter (openrouter.ai) exposes models from all major labs behind a single
OpenAI-compatible endpoint and a single API key, which makes it the cheapest
way -- in setup effort -- to run the full 20-model portfolio: use
`config/models_openrouter.yaml` instead of per-provider keys.

Trade-off to keep in mind for the methodology: calls are routed through an
intermediary, and OpenRouter may itself route a slug to different underlying
deployments. Every run already records the model id the endpoint echoes back
(`returned_model`), so drift stays auditable -- check that field when
comparing OpenRouter runs against direct-API runs.
"""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleAdapter


class OpenRouterAdapter(OpenAICompatibleAdapter):
    provider_key = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai/api/v1"
