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

BackendKind = Literal["mock"]


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
        return self.mock.load_seconds


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


class ServerConfig(BaseModel):
    model_config = {"extra": "forbid"}

    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, le=65535)
    log_level: str = "info"


class AppConfig(BaseModel):
    model_config = {"extra": "forbid"}

    server: ServerConfig = Field(default_factory=ServerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
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
