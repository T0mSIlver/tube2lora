from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI

from tube2lora.config import LLMEndpointConfig


@dataclass(slots=True)
class ChatRequest:
    model: str
    system_prompt: str
    user_prompt: str
    temperature: float = 0.1
    max_tokens: int = 4096


class OpenAIChatClient:
    def __init__(self, endpoint: LLMEndpointConfig):
        api_key = os.environ.get(endpoint.api_key_env, "dummy")
        self.endpoint = endpoint
        self.logger = logging.getLogger("tube2lora.llm")
        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint.base_url,
            timeout=float(endpoint.timeout_seconds),
        )

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    if part.strip():
                        parts.append(part.strip())
                    continue
                if not isinstance(part, dict):
                    continue
                text_value = part.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value.strip())
            return "\n".join(parts).strip()
        return str(content).strip()

    def complete(self, request: ChatRequest) -> str:
        last_error: Exception | None = None
        for attempt in range(self.endpoint.max_retries):
            try:
                messages = []
                if request.system_prompt.strip():
                    messages.append({"role": "system", "content": request.system_prompt.strip()})
                messages.append({"role": "user", "content": request.user_prompt.strip()})
                response = self.client.chat.completions.create(
                    model=request.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
                if not response.choices:
                    return ""

                choice = response.choices[0]
                message = choice.message
                finish_reason = getattr(choice, "finish_reason", None)
                # Ignore any provider-specific reasoning payload.
                _ = getattr(message, "reasoning_content", None)
                content_text = self._content_to_text(getattr(message, "content", None))
                if not content_text and finish_reason == "length":
                    self.logger.warning(
                        "LLM returned empty content with finish_reason='length' "
                        "(model=%s, max_tokens=%d). "
                        "This can mean max_tokens is too low; consider increasing it.",
                        request.model,
                        request.max_tokens,
                    )
                return content_text
            except (APIStatusError, APIConnectionError, TimeoutError) as exc:
                last_error = exc
                if attempt == self.endpoint.max_retries - 1:
                    break
                sleep_seconds = 2**attempt
                time.sleep(sleep_seconds)
            except Exception as exc:  # pragma: no cover - safety net
                last_error = exc
                if attempt == self.endpoint.max_retries - 1:
                    break
                time.sleep(1)

        raise RuntimeError(f"LLM completion failed after retries: {last_error}")
