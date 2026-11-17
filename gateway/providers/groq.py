"""Groq adapter (OpenAI-compatible chat completions API)."""

from __future__ import annotations

from gateway.providers.openai_compat import OpenAICompatibleProvider, parse_response

__all__ = ["GroqProvider", "parse_response"]


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    label = "Groq"
    key_env = "GROQ_API_KEY"
    default_base_url = "https://api.groq.com/openai/v1"
