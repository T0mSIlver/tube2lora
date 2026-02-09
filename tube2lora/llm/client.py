from __future__ import annotations

import os
import time
from dataclasses import dataclass

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
        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint.base_url,
            timeout=float(endpoint.timeout_seconds),
        )

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
                content = response.choices[0].message.content
                if content is None:
                    return ""
                return str(content).strip()
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
