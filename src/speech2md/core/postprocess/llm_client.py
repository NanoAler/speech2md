from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from speech2md.core.models import LLMBackend


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        ...


class OllamaClient(LLMClient):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma4", timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # def complete(self, prompt: str) -> str:
    #     resp = httpx.post(
    #         f"{self.base_url}/api/generate",
    #         json={"model": self.model, "prompt": prompt, "stream": False},
    #         timeout=self.timeout,
    #     )
    #     resp.raise_for_status()
    #     data: dict[str, Any] = resp.json()
    #     return str(data["response"]).strip()

    def complete(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            raise RuntimeError(
                f"Ollama: модель '{self.model}' не найдена. "
                f"Выполните: ollama pull {self.model}\nОтвет сервера: {resp.text}"
            )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return str(data["response"]).strip()


class OpenAIClient(LLMClient):
    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: str = "",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return str(data["choices"][0]["message"]["content"]).strip()


def create_client(
    backend: LLMBackend | str,
    url: str = "",
    model: str = "",
    api_key: str = "",
    timeout: int = 120,
) -> LLMClient:
    backend = LLMBackend(backend) if isinstance(backend, str) else backend
    if backend == LLMBackend.ollama:
        return OllamaClient(base_url=url, model=model or "gemma4", timeout=timeout)
    return OpenAIClient(base_url=url, model=model or "gpt-4o-mini", api_key=api_key, timeout=timeout)
