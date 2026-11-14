"""Request bodies: the OpenAI chat completions request, plus the gateway's own endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Generation parameters forwarded to the providers (each adapter keeps the ones it supports).
PARAMS = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "seed", "presence_penalty",
          "frequency_penalty", "reasoning_effort", "response_format")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(BaseModel):
    """OpenAI ``POST /v1/chat/completions`` body. Unknown fields are accepted and ignored."""

    model_config = ConfigDict(extra="allow")

    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = Field(None, ge=-2, le=2)
    frequency_penalty: float | None = Field(None, ge=-2, le=2)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    response_format: dict[str, Any] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    user: str | None = None
    n: int | None = None
    tools: list[Any] | None = None
    tool_choice: Any = None
    functions: list[Any] | None = None

    def params(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in PARAMS if getattr(self, key) is not None}


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=100)
    score: int | Literal["up", "down"]
    comment: str | None = Field(None, max_length=2000)

    @field_validator("score")
    @classmethod
    def _score(cls, value: int | str) -> int | str:
        if isinstance(value, int) and value not in (1, -1):
            raise ValueError("score must be 1, -1, 'up' or 'down'")
        return value

    @property
    def value(self) -> int:
        return {"up": 1, "down": -1}.get(self.score, self.score) if isinstance(self.score, str) else self.score


class ActivateRequest(BaseModel):
    release: str
    note: str = ""


class CanaryRequest(BaseModel):
    release: str
    share: float = Field(ge=0, le=1)
    note: str = ""


class NoteRequest(BaseModel):
    note: str = ""
