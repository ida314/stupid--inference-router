"""A backend that serves from vLLM, and optionally owns the container it runs in.

Generation is a pass-through: vLLM answers to the same tag the client asked for, so the
body `sir` received is the body vLLM gets, byte for byte. Nothing is translated, which is
the contract the README argues for at length — `sir` reads `model` to route and `stream`
to pick a rendering, and interprets nothing else.

Residency is the part that varies, and it is one flag:

* `manage_lifecycle: false` — **adopt**. The container is somebody else's; `sir` waits for
  it to be healthy and never starts or stops anything. This is the correct setting while
  one model has the GPU to itself, since there is nothing to swap to and no reason to hand
  a network-facing service the power to stop the box's only inference server.
* `manage_lifecycle: true` — **own**. `start()` and `stop()` drive the container through
  the Docker Engine API, which is what makes a swap possible.

Owning does not mean *creating*. `sir` starts and stops a container that already exists,
defined in compose where a thirty-flag vLLM invocation can be read and reviewed. A router
that assembled that command line from its own YAML would be a second, worse compose file.

The measured swap cost on this hardware is around five minutes — 113s of weight loading and
146s of engine warmup, against roughly 30s of container and process startup. That ratio is
why stopping a container is a perfectly good unload: nearly all of the cost is the engine
coming up, and no in-process unload avoids paying it again.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote

import httpx

from sir.backend import BackendError
from sir.clock import Clock
from sir.config import VllmParams
from sir.types import Chunk, GenerationRequest, StreamEnd

_DONE = "[DONE]"

# The Docker Engine API is reached over its unix socket, so the host in the URL is a
# placeholder that httpx requires and the transport ignores.
_DOCKER_HOST = "http://docker"


class VllmBackend:
    """One vLLM server: generation always, lifecycle only when asked."""

    def __init__(
        self,
        model_name: str,
        params: VllmParams,
        clock: Clock,
        client: httpx.AsyncClient | None = None,
        docker: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_name = model_name
        self._params = params
        self._clock = clock

        # Both injectable so tests drive a fake vLLM and a fake Docker through
        # httpx.MockTransport — no sockets, no daemon, no weights.
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                params.stream_stall_seconds, connect=params.connect_timeout_seconds
            )
        )
        self._docker = docker
        if self._docker is None and params.manage_lifecycle:
            self._docker = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(uds=params.docker_socket),
                base_url=_DOCKER_HOST,
                timeout=httpx.Timeout(params.stop_timeout_seconds + 30),
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Become able to serve, and do not return until that is actually true.

        Adopting or owning, the postcondition is the same — `/health` answers — because
        that is the only claim the engine can safely act on.
        """
        if self._params.manage_lifecycle:
            await self._docker_post("start")

        if not await self._await_health(True, self._params.start_timeout_seconds):
            raise BackendError(
                f"vllm for {self._model_name!r} did not become healthy within "
                f"{self._params.start_timeout_seconds:.0f}s"
            )

    async def stop(self) -> None:
        """Give the GPU back. Never raises — usually called while cleaning up a crash."""
        if not self._params.manage_lifecycle:
            # Adopted: the container is not ours to stop. Reporting success is honest —
            # this backend holds nothing that needs releasing.
            return

        try:
            await self._docker_post("stop", params={"t": int(self._params.stop_timeout_seconds)})
        except BackendError:
            # A daemon we cannot reach leaves nothing we can do, and the engine is on a
            # path to loading something else. Fall through to the wait, which will time
            # out honestly rather than pretend the memory came back.
            pass

        # Docker returns once SIGTERM is delivered, not once the process is gone, and the
        # GPU memory outlives the API call. Waiting for the port to stop answering is what
        # keeps the next model's load from racing this one's teardown — the exact moment a
        # swap would otherwise turn into an out-of-memory crash.
        await self._await_health(False, self._params.stop_timeout_seconds)

    async def health(self) -> bool:
        try:
            response = await self._client.get(
                self._health_url, timeout=self._params.connect_timeout_seconds
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def _await_health(self, want: bool, timeout: float) -> bool:
        deadline = self._clock.now() + timeout
        while True:
            if await self.health() == want:
                return True
            if self._clock.now() >= deadline:
                return False
            await self._clock.sleep(self._params.ready_poll_seconds)

    async def _docker_post(self, action: str, params: dict[str, Any] | None = None) -> None:
        """Drive the container. 304 is success: it already was in the desired state."""
        name = quote(self._params.container_name or "", safe="")
        url = f"/containers/{name}/{action}"
        try:
            response = await self._docker.post(url, params=params)
        except httpx.HTTPError as exc:
            raise BackendError(
                f"docker could not {action} {self._params.container_name!r}: {exc}"
            ) from exc

        if response.status_code not in (204, 304):
            raise BackendError(
                f"docker refused to {action} {self._params.container_name!r}: "
                f"HTTP {response.status_code} {_detail(response)}"
            )

    # ---------------------------------------------------------------- generation

    async def stream(self, request: GenerationRequest) -> AsyncIterator[Chunk | StreamEnd]:
        payload = self._upstream_payload(request)
        url = f"{self._base}/v1/chat/completions"

        index = 0
        finish_reason = "stop"
        usage: dict[str, int] | None = None

        try:
            async with self._client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode(errors="replace")
                    raise BackendError(
                        f"vllm rejected generation for {self._model_name!r}: "
                        f"HTTP {response.status_code} {body[:400]}"
                    )

                async for line in response.aiter_lines():
                    event = _parse_sse(line)
                    if event is None:
                        continue
                    if event is _DONE:
                        break

                    choices = event.get("choices") or []
                    if choices:
                        choice = choices[0]
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield Chunk(text=text, index=index)
                            index += 1
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])

                    # Last chunk, no choices, real token counts.
                    if event.get("usage"):
                        usage = _usage(event["usage"])
        except httpx.HTTPError as exc:
            # Covers the stall timeout too. A backend that stopped speaking mid-generation
            # has died as far as the engine is concerned, and the engine's containment —
            # fail these requests, back off this model, keep scheduling the rest — is what
            # should handle it.
            raise BackendError(
                f"vllm stream for {self._model_name!r} failed: {exc}"
            ) from exc

        yield StreamEnd(finish_reason=finish_reason, usage=usage)

    def _upstream_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """The client's body, with the two changes `sir` is entitled to make.

        `model` is already the tag this backend serves — the API layer set it when routing
        — so unlike a backend that renames models, there is nothing to rewrite here.
        """
        payload = dict(request.payload)

        # `sir` always consumes a stream and re-renders it, whether or not the client
        # asked for one: one dispatch path, two renderings.
        payload["stream"] = True

        # Ask for token counts. Merged, so a client's own stream_options survive.
        options = dict(payload.get("stream_options") or {})
        options["include_usage"] = True
        payload["stream_options"] = options

        return payload

    # ---------------------------------------------------------------- plumbing

    @property
    def _base(self) -> str:
        return self._params.base_url.rstrip("/")

    @property
    def _health_url(self) -> str:
        return f"{self._base}{self._params.health_path}"

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._docker is not None:
            await self._docker.aclose()


def _detail(response: httpx.Response) -> str:
    """The daemon's own explanation, trimmed. Usually the only clue worth logging."""
    try:
        return response.text.strip()[:400]
    except Exception:  # noqa: BLE001 - an unreadable body must not mask the error
        return ""


def _parse_sse(line: str) -> dict[str, Any] | str | None:
    """One SSE line to an event, `_DONE`, or nothing.

    Blank lines separate events and `:` lines are keep-alives; both are noise. A malformed
    data line is skipped rather than fatal — losing one chunk is recoverable, and failing
    the request over it would not be.
    """
    line = line.strip()
    if not line or line.startswith(":") or not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if data == _DONE:
        return _DONE
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _usage(raw: Any) -> dict[str, int] | None:
    """Keep the three counts `sir` reports and drop the backend's extras."""
    if not isinstance(raw, dict):
        return None
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    usage = {key: int(raw[key]) for key in keys if isinstance(raw.get(key), int)}
    return usage or None
