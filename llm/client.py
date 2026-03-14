"""
LLM client with two switchable providers.

Set LLM_PROVIDER in .env:
  LLM_PROVIDER=lmstudio   (default) — fully offline, routes to LM Studio
  LLM_PROVIDER=claude               — routes to Anthropic API (requires ANTHROPIC_API_KEY)
"""

from __future__ import annotations
import logging
import time
from config import settings

log = logging.getLogger(__name__)


def _lmstudio_chat(messages: list[dict], model: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(
        base_url=settings.lm_studio_url,
        api_key="lm-studio",  # required by SDK, ignored by LM Studio
    )
    last_error: Exception | None = None
    for attempt in range(settings.lm_studio_retries):
        try:
            log.debug("LM Studio request attempt %d/%d", attempt + 1, settings.lm_studio_retries)
            response = client.chat.completions.create(
                model=model or settings.lm_studio_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            if attempt < settings.lm_studio_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s, …
                log.warning(
                    "LM Studio attempt %d/%d failed: %s — retrying in %ds",
                    attempt + 1, settings.lm_studio_retries, e, wait,
                )
                time.sleep(wait)

    raise RuntimeError(
        f"LM Studio request failed after {settings.lm_studio_retries} attempts. "
        f"Is LM Studio running at {settings.lm_studio_url} with a model loaded?\n"
        f"Last error: {last_error}"
    ) from last_error


def _lmstudio_list_models() -> list[str]:
    from openai import OpenAI
    client = OpenAI(base_url=settings.lm_studio_url, api_key="lm-studio")
    try:
        return [m.id for m in client.models.list().data]
    except Exception:
        return []


# ── Claude (Anthropic API) ────────────────────────────────────────────────────

_CLAUDE_RETRIES = 3


def _claude_chat(messages: list[dict], model: str, temperature: float, max_tokens: int) -> str:
    import anthropic
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set in .env. "
            "Get your key at https://console.anthropic.com/"
        )
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    model = model or settings.claude_model
    system_text = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            user_messages.append({"role": m["role"], "content": m["content"]})
    kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature, messages=user_messages)
    if system_text:
        kwargs["system"] = system_text

    last_error: Exception | None = None
    for attempt in range(_CLAUDE_RETRIES):
        try:
            log.debug("Claude API request attempt %d/%d: model=%s", attempt + 1, _CLAUDE_RETRIES, model)
            response = client.messages.create(**kwargs)
            return response.content[0].text.strip()
        except Exception as e:
            last_error = e
            if attempt < _CLAUDE_RETRIES - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                log.warning(
                    "Claude API attempt %d/%d failed: %s — retrying in %ds",
                    attempt + 1, _CLAUDE_RETRIES, e, wait,
                )
                time.sleep(wait)

    raise RuntimeError(
        f"Claude API request failed after {_CLAUDE_RETRIES} attempts.\n"
        f"Last error: {last_error}"
    ) from last_error


def _claude_list_models() -> list[str]:
    return ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"]


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    provider = settings.llm_provider.lower()
    if provider == "claude":
        return _claude_chat(messages, model or settings.claude_model, temperature, max_tokens)
    return _lmstudio_chat(messages, model or settings.lm_studio_model, temperature, max_tokens)


def list_models() -> list[str]:
    if settings.llm_provider.lower() == "claude":
        return _claude_list_models()
    return _lmstudio_list_models()


def active_provider() -> str:
    return settings.llm_provider.lower()
