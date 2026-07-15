"""Adapter for any provider exposing an OpenAI-compatible Chat Completions API.

The `openai` Python SDK can talk to any endpoint that implements the
`/chat/completions` contract simply by pointing `base_url` at it. Mistral, xAI,
DeepSeek, Moonshot, Qwen (DashScope compat mode), Together and Cohere all do,
so they share this single implementation and only differ by base_url +
api_key_env, set in their thin subclasses (see registry.py).
"""

from __future__ import annotations

from typing import Any

from .base import GenerationResult, ProviderAdapter, ProviderError


class OpenAICompatibleAdapter(ProviderAdapter):
    """Chat-Completions adapter with a configurable base URL."""

    #: Override in subclasses to hit a non-OpenAI endpoint.
    base_url: str | None = None

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ProviderError(
                "The `openai` package is required for OpenAI-compatible providers. "
                "Install it with `pip install openai`."
            ) from exc
        return OpenAI(api_key=self._resolve_api_key(), base_url=self.base_url)

    def generate(self, system_prompt: str, user_prompt: str, **params: Any) -> GenerationResult:
        client = self._client()
        call = self.merged_params(**params)
        max_tokens = call.pop("max_tokens", None)
        temperature = call.pop("temperature", None)

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        # Anything else the config passed (top_p, etc.) is forwarded as-is.
        kwargs.update(call)

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surface provider errors uniformly
            raise ProviderError(f"{self.provider_key} request failed: {exc}") from exc

        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        return GenerationResult(text=text, usage=usage, model=getattr(resp, "model", self.model_id))
