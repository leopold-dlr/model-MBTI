"""Common provider adapter interface.

Every provider is an interchangeable adapter behind a single method:

    generate(system_prompt, user_prompt, **params) -> GenerationResult

Adding a model = adding an adapter class + a line in config/models.yaml. The
orchestrator never imports a concrete provider directly; it goes through
`registry.build_adapter`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ProviderError(RuntimeError):
    """Raised when a provider call fails or is misconfigured."""


class MissingAPIKey(ProviderError):
    """Raised when the required API key env var is unset."""


@dataclass
class GenerationResult:
    """Everything we learned from a single model call."""

    text: str
    # Raw usage dict if the provider reports it (prompt/completion tokens).
    usage: dict[str, Any] = field(default_factory=dict)
    # The exact model id the provider echoed back, if any.
    model: str | None = None


class ProviderAdapter(ABC):
    """Base class for all provider adapters.

    Subclasses set ``api_key_env`` (or override ``_resolve_api_key``) and
    implement ``generate``.
    """

    #: Name of the environment variable holding this provider's API key.
    api_key_env: str = ""
    #: Registry key (also used as a human-readable provider label).
    provider_key: str = ""

    def __init__(self, model_id: str, params: dict[str, Any] | None = None) -> None:
        self.model_id = model_id
        self.params = dict(params or {})

    # -- key handling -----------------------------------------------------
    def _resolve_api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise MissingAPIKey(
                f"{self.provider_key}: environment variable {self.api_key_env} is not set."
            )
        return key

    def is_available(self) -> bool:
        """True if the API key is present (used to skip unconfigured models)."""
        try:
            self._resolve_api_key()
            return True
        except MissingAPIKey:
            return False

    # -- the one method every adapter must implement ----------------------
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **params: Any) -> GenerationResult:
        """Return the model's raw text response.

        ``params`` may include ``temperature`` (usually omitted so the model's
        default is used) and ``max_tokens``.
        """
        raise NotImplementedError

    def merged_params(self, **overrides: Any) -> dict[str, Any]:
        merged = dict(self.params)
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return merged
