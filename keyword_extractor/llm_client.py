"""
LLM Keyword Generator - LLM Client

Generic client for any OpenAI-compatible chat completions API (OpenAI
itself, Azure OpenAI's OpenAI-compatible surface, or a compatible
gateway/proxy), driven by a single shared API_KEY + ENDPOINT.
"""

import json
import re
import time
from typing import Dict, Optional

from .config import get_model_config
from .prompts import DEFAULT_SYSTEM_MESSAGE


class LLMError(Exception):
    """Raised when an LLM call fails or returns an unusable response."""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict:
    """Parse a JSON object out of a model response, tolerating stray text
    or markdown code fences around the JSON body.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip common markdown code fences
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    # Fall back to grabbing the first {...} block
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise LLMError(f"Failed to parse JSON from model response: {text[:300]!r}")


def _rejects_max_tokens(exc: Exception) -> bool:
    """Detect the OpenAI API's "'max_tokens' is not supported ... use
    'max_completion_tokens'" error, returned by newer reasoning-style models
    (e.g. gpt-5.x, o1/o3) regardless of provider/endpoint.
    """
    message = str(exc)
    return "max_tokens" in message and "max_completion_tokens" in message


class LLMClient:
    """Client for chat-completion style keyword extraction calls."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the client for a given model.

        Args:
            model_name: Preset name from `config.list_models()`, or any
                arbitrary deployment/model id. Defaults to the configured
                default model.
        """
        self.config = get_model_config(model_name)
        self.model_name = self.config["name"]
        self._client = self._build_client()

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("openai package not installed. Run: pip install openai") from exc

        kwargs = {"api_key": self.config["api_key"]}
        if self.config.get("endpoint"):
            kwargs["base_url"] = self.config["endpoint"]
        return OpenAI(**kwargs)

    def complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        json_mode: bool = True,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Run a single chat completion and return the raw text response."""
        messages = []
        if self.config.get("supports_system_message", True):
            messages.append({"role": "system", "content": system_message or DEFAULT_SYSTEM_MESSAGE})
        messages.append({"role": "user", "content": prompt})

        kwargs = {"model": self.config["deployment"], "messages": messages}

        effective_max_tokens = max_tokens or self.config.get("default_max_tokens")
        token_key = "max_completion_tokens" if self.config.get("requires_completion_tokens") else "max_tokens"
        if effective_max_tokens:
            kwargs[token_key] = effective_max_tokens

        if json_mode and self.config.get("supports_json_mode", True):
            kwargs["response_format"] = {"type": "json_object"}

        max_retries = self.config.get("max_retries", 2)
        swapped_token_param = False
        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as exc:  # noqa: BLE001 - surfaced as LLMError
                # Some models (e.g. gpt-5.x, o1/o3) reject `max_tokens` and
                # require `max_completion_tokens` instead. Self-heal once so
                # this works even for models not marked in the registry, then
                # remember the correction for subsequent calls on this client.
                if not swapped_token_param and "max_tokens" in kwargs and _rejects_max_tokens(exc):
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    self.config["requires_completion_tokens"] = True
                    swapped_token_param = True
                    continue
                if attempt == max_retries:
                    raise LLMError(f"API call failed after {max_retries} retries: {exc}") from exc
                time.sleep(2 ** attempt)

    def complete_json(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict:
        """Run a completion and parse the response as JSON."""
        raw = self.complete(prompt, system_message=system_message, json_mode=True, max_tokens=max_tokens)
        return _extract_json(raw)

    def __repr__(self):
        return f"LLMClient(model={self.model_name}, deployment={self.config['deployment']})"
