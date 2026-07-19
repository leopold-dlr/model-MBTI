"""Alibaba Qwen adapter via the DashScope OpenAI-compatible endpoint.

Accepts either DASHSCOPE_API_KEY (preferred) or QWEN_API_KEY.
"""

from __future__ import annotations

import os

from .base import MissingAPIKey
from .openai_compatible import OpenAICompatibleAdapter


class QwenAdapter(OpenAICompatibleAdapter):
    provider_key = "qwen"
    api_key_env = "DASHSCOPE_API_KEY"
    # International DashScope compatibility endpoint.
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def _resolve_api_key(self) -> str:
        key = os.environ.get("DASHSCOPE_API_KEY", "").strip() or os.environ.get(
            "QWEN_API_KEY", ""
        ).strip()
        if not key:
            raise MissingAPIKey(
                "qwen: set DASHSCOPE_API_KEY or QWEN_API_KEY."
            )
        return key
