"""Google (Gemini) adapter using the google-genai SDK."""

from __future__ import annotations

from typing import Any

from .base import GenerationResult, ProviderAdapter, ProviderError


class GoogleAdapter(ProviderAdapter):
    provider_key = "google"
    api_key_env = "GOOGLE_API_KEY"

    def _client(self):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ProviderError(
                "The `google-genai` package is required. Install it with `pip install google-genai`."
            ) from exc
        return genai.Client(api_key=self._resolve_api_key())

    def generate(self, system_prompt: str, user_prompt: str, **params: Any) -> GenerationResult:
        try:
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("The `google-genai` package is required.") from exc

        client = self._client()
        call = self.merged_params(**params)
        max_tokens = call.pop("max_tokens", None)
        temperature = call.pop("temperature", None)

        config_kwargs: dict[str, Any] = {"system_instruction": system_prompt}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if temperature is not None:
            config_kwargs["temperature"] = temperature

        try:
            resp = client.models.generate_content(
                model=self.model_id,
                contents=user_prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"google request failed: {exc}") from exc

        text = (getattr(resp, "text", None) or "").strip()
        usage = {}
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            usage = {
                "prompt_tokens": getattr(meta, "prompt_token_count", None),
                "completion_tokens": getattr(meta, "candidates_token_count", None),
                "total_tokens": getattr(meta, "total_token_count", None),
            }
        return GenerationResult(text=text, usage=usage, model=self.model_id)
