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

    name: str = Field(min_length=1)
    backend: BackendKind = "mock"
    # Multiplies the model's score. A translation model that must feel snappy despite low
    # volume gets a higher number than a batch summarizer.
    priority: float = Field(default=1.0, gt=0)
    mock: MockParams = Field(default_factory=MockParams)

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
        return self

    def model_by_name(self, name: str) -> ModelConfig | None:
        for model in self.models:
            if model.name == name:
                return model
        return None

    @property
    def model_names(self) -> list[str]:
        return [model.name for model in self.models]


def load_config(path: str | Path) -> AppConfig:
    """Read and validate a YAML config file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    raw: Any = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")

    return AppConfig.model_validate(raw)
