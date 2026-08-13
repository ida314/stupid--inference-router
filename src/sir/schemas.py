"""OpenAI-compatible wire types.

Services already speak this dialect, so `sir` speaks it too — that's the entire reason a
service can be pointed at `:8000` and stop caring which model is loaded. Requests carry
`extra="allow"` because real OpenAI clients send fields we don't implement, and rejecting
a request over `top_p` would be a poor way to introduce a router.

The models below are validation and routing only, never a filter. `sir` needs `model` to
pick a queue and `stream` to pick a response shape; everything else is forwarded to the
backend untouched, including the extras vLLM and SGLang each add on top of the OpenAI
schema. So the fields here are typed loosely on purpose — a bound like `max_tokens > 0`
would be `sir` enforcing a rule it doesn't own, and would turn a backend's future
behaviour into a router bug.
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

    def to_generation_request(
        self, internal_name: str, backend_tag: str, request_id: str = ""
    ) -> GenerationRequest:
        """Build the backend-bound request, preserving the client's body as sent.

        `exclude_unset` is what makes this a pass-through rather than a rewrite: a field
        the client omitted stays omitted, so the backend applies its own default instead
        of one `sir` invented. Only `model` is changed, from the tag the client used to
        the one the backend answers to.
        """
        payload = self.model_dump(exclude_unset=True, exclude_none=False)
        payload["model"] = backend_tag
        return GenerationRequest(
            model=internal_name,
            served_model=self.model,
            payload=payload,
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
    # vLLM reports the underlying model an alias points at; clients that resolve aliases
    # read this. For a model with one served name, `root` is just `id`.
    root: str | None = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


def error_body(message: str, kind: str = "invalid_request_error", code: str | None = None) -> dict[str, Any]:
    """The error envelope OpenAI clients know how to parse."""
    return {"error": {"message": message, "type": kind, "param": None, "code": code}}
