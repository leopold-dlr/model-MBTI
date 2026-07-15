"""Anthropic (Claude) adapter using the native Messages API."""

from __future__ import annotations

from typing import Any

from .base import GenerationResult, ProviderAdapter, ProviderError


class AnthropicAdapter(ProviderAdapter):
    provider_key = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"

    def _client(self):
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ProviderError(
                "The `anthropic` package is required. Install it with `pip install anthropic`."
            ) from exc
        return Anthropic(api_key=self._resolve_api_key())

    def generate(self, system_prompt: str, user_prompt: str, **params: Any) -> GenerationResult:
        client = self._client()
        call = self.merged_params(**params)
        # Anthropic requires max_tokens; default generously for 32 JSON items.
        max_tokens = call.pop("max_tokens", None) or 2048
        temperature = call.pop("temperature", None)

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        kwargs.update(call)

        try:
            resp = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"anthropic request failed: {exc}") from exc

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        ).strip()
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "input_tokens", None),
                "completion_tokens": getattr(resp.usage, "output_tokens", None),
            }
        return GenerationResult(text=text, usage=usage, model=getattr(resp, "model", self.model_id))
