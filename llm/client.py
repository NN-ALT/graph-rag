"""
LLM client with two switchable providers.

Set LLM_PROVIDER in .env:
  LLM_PROVIDER=lmstudio   (default) — fully offline, routes to LM Studio
  LLM_PROVIDER=claude               — routes to Anthropic API (requires ANTHROPIC_API_KEY)
"""

from __future__ import annotations
from config import settings


def _lmstudio_chat(messages: list[dict], model: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=settings.lm_studio_url, api_key="lm-studio")
    try:
        response = client.chat.completions.create(
            model=model or settings.lm_studio_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(
            f"LM Studio request failed. Is LM Studio running at {settings.lm_studio_url} "
            f"with a model loaded?\nError: {e}"
        ) from e


def _lmstudio_list_models() -> list[str]:
    from openai import OpenAI
    client = OpenAI(base_url=settings.lm_studio_url, api_key="lm-studio")
    try:
        return [m.id for m in client.models.list().data]
    except Exception:
        return []


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
    try:
        response = client.messages.create(**kwargs)
        return response.content[0].text.strip()
    except Exception as e:
        raise RuntimeError(f"Claude API request failed: {e}") from e


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
