"""Configuration: YAML in, validated objects out.

Every check here happens once, at startup, and fails loudly. A router that discovers its
config is wrong at 3am — mid-swap, with requests queued — is worse than one that refused
to boot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

BackendKind = Literal["mock", "vllm"]


class VllmParams(BaseModel):
    """Where a vLLM server is, and whether `sir` is allowed to start and stop it."""

    model_config = {"extra": "forbid"}

    # The OpenAI-compatible server. vLLM answers to the tag it was given as
    # `--served-model-name`, which is the same tag clients send, so nothing is renamed
    # anywhere along this path.
    base_url: str = "http://vllm:8000"
    health_path: str = "/health"

    # False: adopt a container somebody else runs. `start()` waits for health, `stop()`
    # does nothing, and `sir` never needs the Docker socket. Correct while a single model
    # owns the GPU, because there is nothing to swap to and no reason to give a
    # network-facing service the power to stop the box's only inference server.
    #
    # True: own it. `start()` and `stop()` drive the container, which is what makes a swap
    # possible — and what a second model requires.
    manage_lifecycle: bool = False

    # The container to drive. `sir` starts and stops it; it does not create it. The
    # container is defined in compose, where a thirty-flag vLLM command line can be read
    # and reviewed, rather than assembled here from a second copy of the same knobs.
    container_name: str | None = None
    docker_socket: str = "/var/run/docker.sock"

    # Measured on this hardware: ~113s loading weights, ~146s of engine profiling and
    # warmup, ~30s of container and process startup. The ceiling is set well above that,
    # since a cold page cache is the slow case and giving up early on a load that would
    # have succeeded costs far more than waiting.
    start_timeout_seconds: float = Field(default=900.0, gt=0)
    ready_poll_seconds: float = Field(default=2.0, gt=0)
    # SIGTERM, then Docker's own SIGKILL. This also bounds how long `stop()` waits for the
    # GPU memory to actually come back.
    stop_timeout_seconds: float = Field(default=120.0, gt=0)

    # A server that is down should fail fast rather than hang the engine's control loop.
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    # A read gap this long mid-stream means the backend died without closing the socket.
    # There is deliberately no ceiling on *total* generation time: a long completion at a
    # low token rate is normal, and the client's own disconnect is the real cancellation
    # signal. Only silence is evidence of death.
    stream_stall_seconds: float = Field(default=300.0, gt=0)

    # What the policy charges for making this model resident, before anything has been
    # measured. The default is the measured cold start above. Phase 4 replaces this with
    # observations.
    estimated_load_seconds: float = Field(default=310.0, ge=0)
    # Opening guess at one request's duration, seeding the engine's running average.
    estimated_request_seconds: float = Field(default=8.0, gt=0)

    @model_validator(mode="after")
    def _owning_needs_a_container(self) -> VllmParams:
        if self.manage_lifecycle and not self.container_name:
            raise ValueError(
                "manage_lifecycle requires container_name: `sir` can only start and stop "
                "a container it can address by name"
            )
        return self


class MockParams(BaseModel):
    """Knobs for the simulated backend, standing in for real GPU behavior."""

    model_config = {"extra": "forbid"}

    load_seconds: float = Field(default=8.0, ge=0)
    unload_seconds: float = Field(default=0.5, ge=0)
    tokens_per_second: float = Field(default=40.0, gt=0)
    first_token_seconds: float = Field(default=0.2, ge=0)
    default_max_tokens: int = Field(default=64, gt=0)

    # Fault injection — how the crash-recovery path gets exercised without a real GPU.
    fail_on_load_every_n: int = Field(default=0, ge=0)
    crash_after_n_requests: int = Field(default=0, ge=0)


class ModelConfig(BaseModel):
    model_config = {"extra": "forbid"}

    # Internal label. Appears in config, logs, and `/status`, and never on the wire —
    # a timeline reading `chat -> translate` is legible in a way that one reading
    # `Qwen/Qwen3-8B -> facebook/nllb-200-distilled-600M` is not.
    name: str = Field(min_length=1)

    # What clients put in `model`. This is the identity `sir` shares with the backend,
    # so it must be the tag the backend itself serves — mirroring vLLM's
    # `--served-model-name`, including its support for several aliases per model.
    # Defaults to `name` so small configs stay small.
    served_model_name: str | list[str] | None = None

    backend: BackendKind = "mock"
    # Multiplies the model's score. A translation model that must feel snappy despite low
    # volume gets a higher number than a batch summarizer.
    priority: float = Field(default=1.0, gt=0)
    mock: MockParams = Field(default_factory=MockParams)
    vllm: VllmParams = Field(default_factory=VllmParams)

    @property
    def served_names(self) -> list[str]:
        """Every tag a client may address this model by."""
        if self.served_model_name is None:
            return [self.name]
        if isinstance(self.served_model_name, str):
            return [self.served_model_name]
        return list(self.served_model_name)

    @property
    def served_as(self) -> str:
        """The canonical tag, sent to the backend and reported in `/status`."""
        return self.served_names[0]

    @model_validator(mode="after")
    def _served_names_are_usable(self) -> ModelConfig:
        names = self.served_names
        if not names or any(not name.strip() for name in names):
            raise ValueError(
                f"model {self.name!r} has an empty served_model_name; omit the field to "
                "fall back to the model name"
            )
        return self

    @property
    def estimated_load_seconds(self) -> float:
        """What the policy charges for making this model resident.

        Phase 1 takes the configured value. Phase 3 replaces this with measurements.
        """
        if self.backend == "vllm":
            return self.vllm.estimated_load_seconds
        return self.mock.load_seconds

    @property
    def estimated_request_seconds(self) -> float:
        """Opening guess at how long one request takes, before any have been observed.

        Only ever the seed for the engine's running average — the moment a real request
        completes, measurement takes over. It exists so the first client to ask for a wait
        estimate doesn't get told zero.
        """
        if self.backend == "vllm":
            return self.vllm.estimated_request_seconds
        return (
            self.mock.first_token_seconds
            + self.mock.default_max_tokens / self.mock.tokens_per_second
        )


class SchedulerConfig(BaseModel):
    model_config = {"extra": "forbid"}

    # Hysteresis: a freshly loaded model is protected for this long, so A B A B traffic
    # doesn't buy four swaps.
    min_residency_seconds: float = Field(default=30.0, ge=0)
    # The starvation ceiling. Not a tuning knob — a correctness backstop.
    max_wait_seconds: float = Field(default=120.0, gt=0)

    age_weight: float = Field(default=1.0, ge=0)
    depth_weight: float = Field(default=0.5, ge=0)
    switch_cost_weight: float = Field(default=1.0, ge=0)

    max_concurrent_requests: int = Field(default=8, gt=0)
    tick_interval_seconds: float = Field(default=0.25, gt=0)
    # After a backend crash or a failed load, wait this long before retrying that model.
    backend_retry_seconds: float = Field(default=5.0, ge=0)

    @model_validator(mode="after")
    def _ceiling_above_hysteresis(self) -> SchedulerConfig:
        if self.max_wait_seconds <= self.min_residency_seconds:
            raise ValueError(
                "max_wait_seconds must exceed min_residency_seconds, otherwise the "
                "starvation ceiling fires before hysteresis can ever hold a model "
                f"(got {self.max_wait_seconds} <= {self.min_residency_seconds})"
            )
        return self


class JobsConfig(BaseModel):
    """The async submission path: clients that would rather poll than hold a socket open.

    Off by nothing and on by default, because it changes no behaviour for a client that
    doesn't ask for it — a request without the opt-in header is served exactly as before.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = True

    # Polling is the liveness signal. Holding a socket open is what tells `sir` today that
    # a client still wants its answer; once the socket is gone, polling has to carry that
    # signal instead. A job unpolled for this long is cancelled exactly as a dropped
    # connection cancels, which bounds how long abandoned work can go on costing GPU time.
    #
    # Keep this below `scheduler.max_wait_seconds` — the default pair is 60 against 120.
    # That ordering is what stops abandoned work from reaching the starvation ceiling,
    # where a swap becomes mandatory and outranks every other consideration. It does not
    # prevent a *scored* swap, which can happen within seconds of a request arriving; no
    # lease short enough to catch that would be long enough to be safe.
    #
    # Set to 0 to disable, accepting that a crashed service leaves its queued work behind.
    lease_seconds: float = Field(default=60.0, ge=0)

    # How long a finished result is kept for a client that never came back for it...
    result_ttl_seconds: float = Field(default=300.0, gt=0)
    # ...and how long after it *has* been read. Shorter, but not zero: dropping the result
    # the instant it is first fetched makes a connection failure mid-fetch unrecoverable.
    read_ttl_seconds: float = Field(default=60.0, gt=0)

    sweep_interval_seconds: float = Field(default=1.0, gt=0)
    # A ceiling on retained jobs, so an unattended router can't grow without bound.
    max_jobs: int = Field(default=1000, gt=0)

    @model_validator(mode="after")
    def _lease_outlives_a_sweep(self) -> JobsConfig:
        if self.lease_seconds and self.lease_seconds <= self.sweep_interval_seconds:
            raise ValueError(
                "lease_seconds must exceed sweep_interval_seconds, otherwise a lease can "
                "expire before the client has had a chance to renew it "
                f"(got {self.lease_seconds} <= {self.sweep_interval_seconds})"
            )
        return self


class ServerConfig(BaseModel):
    model_config = {"extra": "forbid"}

    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, le=65535)
    log_level: str = "info"


class AppConfig(BaseModel):
    model_config = {"extra": "forbid"}

    server: ServerConfig = Field(default_factory=ServerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    jobs: JobsConfig = Field(default_factory=JobsConfig)
    models: list[ModelConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_model_names(self) -> AppConfig:
        seen: set[str] = set()
        for model in self.models:
            if model.name in seen:
                raise ValueError(f"duplicate model name: {model.name!r}")
            seen.add(model.name)

        # Two models answering to one tag would make routing depend on config order.
        served: dict[str, str] = {}
        for model in self.models:
            for tag in model.served_names:
                if tag in served:
                    raise ValueError(
                        f"duplicate served_model_name {tag!r}: claimed by both "
                        f"{served[tag]!r} and {model.name!r}"
                    )
                served[tag] = model.name
        return self

    def model_by_name(self, name: str) -> ModelConfig | None:
        """Look up by internal label. For config and logs, never for client input."""
        for model in self.models:
            if model.name == name:
                return model
        return None

    def model_by_served_name(self, tag: str) -> ModelConfig | None:
        """Resolve what a client asked for. The only lookup the API is allowed to use."""
        for model in self.models:
            if tag in model.served_names:
                return model
        return None

    @property
    def model_names(self) -> list[str]:
        return [model.name for model in self.models]

    @property
    def served_model_names(self) -> list[str]:
        """Every tag clients may address, in config order."""
        return [tag for model in self.models for tag in model.served_names]


def load_config(path: str | Path) -> AppConfig:
    """Read and validate a YAML config file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    raw: Any = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")

    return AppConfig.model_validate(raw)
