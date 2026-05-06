"""LLM client abstraction.

Agents depend on this interface instead of importing a provider SDK directly.
If OPENAI_API_KEY is absent, the client returns a deterministic offline response
so lab tests and demos still work.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from multi_agent_research_lab.core.config import get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client with offline fallback."""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a completion from OpenAI if configured, otherwise a local fallback."""

        # Make CLI runs load .env automatically.
        load_dotenv()

        settings = get_settings()

        if not settings.openai_api_key:
            return self._offline_complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

        try:
            client_cls = self._resolve_openai_client_class()
            client: Any = client_cls(
                api_key=settings.openai_api_key,
                timeout=settings.timeout_seconds,
            )

            # Keep the request minimal and compatible.
            # Do not pass metadata here because some Chat Completions setups reject it.
            response: Any = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )

            content = str(response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)

            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=None,
            )

        except Exception as exc:  # pragma: no cover - external provider dependent
            fallback = self._offline_complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            # Show useful provider error without leaking API key.
            error_message = str(exc)
            api_key = str(settings.openai_api_key or "")
            if api_key:
                error_message = error_message.replace(api_key, "***")

            error_message = error_message[:500]

            return LLMResponse(
                content=(
                    f"{fallback.content}\n\n"
                    f"[Provider fallback: {exc.__class__.__name__}: {error_message}]"
                ),
                input_tokens=fallback.input_tokens,
                output_tokens=fallback.output_tokens,
                cost_usd=fallback.cost_usd,
            )

    def _resolve_openai_client_class(self) -> Any:
        """Prefer Langfuse OpenAI wrapper when installed, otherwise use OpenAI SDK.

        Langfuse wrapper is a drop-in replacement for OpenAI calls.
        If Langfuse is unavailable or incompatible, the normal OpenAI SDK is used.
        """

        try:
            langfuse_openai_module = importlib.import_module("langfuse.openai")
            return langfuse_openai_module.OpenAI
        except Exception:
            openai_module = importlib.import_module("openai")
            return openai_module.OpenAI

    def _offline_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        merged = f"{system_prompt}\n{user_prompt}".strip()
        words = merged.split()
        topic = " ".join(user_prompt.split()[:18]).strip()

        content = (
            "Offline LLM fallback: "
            f"for the request '{topic}', use a structured answer with evidence, risks, "
            "and next steps. Configure OPENAI_API_KEY to replace this deterministic response."
        )

        return LLMResponse(
            content=content,
            input_tokens=len(words),
            output_tokens=len(content.split()),
            cost_usd=0.0,
        )
