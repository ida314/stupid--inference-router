"""OpenAI-compatible wire types.

Services already speak this dialect, so `sir` speaks it too — that's the entire reason a
service can be pointed at `:8000` and stop caring which model is loaded. Requests carry
`extra="allow"` because real OpenAI clients send fields we don't implement, and rejecting
a request over `top_p` would be a poor way to introduce a router.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from sir.types import GenerationRequest


def _now() -> int:
    return int(time.time())


class ChatMessage(BaseModel):
    model_config = {"extra": "allow"}

    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: float = 1.0

    def token_budget(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens

    def to_generation_request(self, request_id: str) -> GenerationRequest:
        return GenerationRequest(
            model=self.model,
            prompt=render_prompt(self.messages),
            max_tokens=self.token_budget(),
            temperature=self.temperature,
            request_id=request_id,
        )


def render_prompt(messages: list[ChatMessage]) -> str:
    """Flatten a chat transcript into the plain prompt a backend takes.

    A real backend applies the model's chat template; the mock just needs the text.
    """
    return "\n".join(f"{m.role}: {m.content or ''}" for m in messages)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=_now)
    model: str
    choices: list[ChatChoice]
    usage: Usage = Field(default_factory=Usage)


class ChatDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatChunkChoice(BaseModel):
    index: int = 0
    delta: ChatDelta = Field(default_factory=ChatDelta)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=_now)
    model: str
    choices: list[ChatChunkChoice]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=_now)
    owned_by: str = "sir"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


def error_body(message: str, kind: str = "invalid_request_error", code: str | None = None) -> dict[str, Any]:
    """The error envelope OpenAI clients know how to parse."""
    return {"error": {"message": message, "type": kind, "param": None, "code": code}}
