"""OpenRouter API clients used by CodeSmith."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Generator, Union

import requests

from .agent import ModelResponse


OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b:free"


class OpenRouterError(Exception):
    """Raised when an OpenRouter request cannot be completed."""


def _tool_calls(message: dict) -> list[dict]:
    """Convert OpenAI/OpenRouter tool calls to the agent's provider format."""
    calls = []
    for call in message.get("tool_calls", []) or []:
        function = call.get("function", call)
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        calls.append({"id": call.get("id"), "name": function.get("name", ""), "arguments": arguments})
    return calls


class OpenRouterChatProvider:
    """Model-provider adapter for the repository-aware agent runtime."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: int = 600,
        base_url: str = OPENROUTER_URL,
        session_id: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is not set. Create an OpenRouter key and "
                "export it before starting CodeSmith."
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "CodeSmith",
        }

    async def generate(self, messages: list, tools: list) -> ModelResponse:
        return await asyncio.to_thread(self._generate, messages, tools)

    def _generate(self, messages: list, tools: list) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "reasoning": {"enabled": True},
        }
        if self.session_id:
            # OpenRouter uses this as a sticky routing and observability key;
            # conversation content remains managed locally by CodeSmith.
            payload["session_id"] = self.session_id
        if tools:
            payload["tools"] = [
                {"type": "function", "function": tool} for tool in tools
            ]
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            self._raise_for_status(response)
            data = response.json()
            if data.get("error"):
                error = data["error"]
                if isinstance(error, dict):
                    detail = error.get("message") or error.get("code") or str(error)
                else:
                    detail = str(error)
                raise OpenRouterError(f"OpenRouter rejected the request: {detail}")
            choices = data.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                raise OpenRouterError(
                    "OpenRouter returned no assistant choices. This model may not "
                    "support the requested tool-calling format."
                )
            message = choices[0].get("message") or {}
            return ModelResponse(
                content=message.get("content") or "",
                tool_calls=_tool_calls(message),
                reasoning_details=message.get("reasoning_details"),
                usage=data.get("usage") or {},
            )
        except (requests.exceptions.RequestException, ValueError, KeyError, TypeError) as exc:
            raise OpenRouterError(f"Chat request failed: {exc}") from exc

    @staticmethod
    def _raise_for_status(response) -> None:
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            detail = getattr(response, "text", "").strip()
            message = f"OpenRouter request failed ({response.status_code})"
            try:
                error = response.json().get("error", {})
                if isinstance(error, dict):
                    error_message = error.get("message")
                    remedy = error.get("remedy_hint")
                    if response.status_code == 429:
                        message = "OpenRouter rate limit reached"
                        if error_message:
                            message += f": {error_message}"
                        if remedy:
                            message += f". {remedy}"
                    elif error_message:
                        message += f": {error_message}"
                elif error:
                    message += f": {error}"
            except (ValueError, TypeError, AttributeError):
                if detail:
                    message += f": {detail}"
            raise OpenRouterError(message) from exc


class OpenRouterClient:
    """Simple text-generation client for the batch command."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: int = 600,
        base_url: str = OPENROUTER_URL,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set")

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Union[Generator[str, None, None], str]:
        if stream:
            # Keep the batch client's historical streaming interface.
            return self._generate_stream(prompt, temperature, top_p)
        return self._generate_full(prompt, temperature, top_p)

    def _request(self, prompt: str, temperature: float, top_p: float, stream: bool = False):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "reasoning": {"enabled": True},
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "CodeSmith",
            },
            json=payload,
            stream=stream,
            timeout=self.timeout,
        )
        OpenRouterChatProvider._raise_for_status(response)
        return response

    def _generate_full(self, prompt: str, temperature: float, top_p: float) -> str:
        try:
            data = self._request(prompt, temperature, top_p).json()
            return data["choices"][0]["message"].get("content") or ""
        except (requests.exceptions.RequestException, ValueError, KeyError, TypeError) as exc:
            raise OpenRouterError(f"API request failed: {exc}") from exc

    def _generate_stream(self, prompt: str, temperature: float, top_p: float):
        try:
            response = self._request(prompt, temperature, top_p, stream=True)
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    delta = json.loads(line)["choices"][0].get("delta", {})
                    if delta.get("content"):
                        yield delta["content"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        except requests.exceptions.RequestException as exc:
            raise OpenRouterError(f"API request failed: {exc}") from exc
