"""Config validation. Every one of these must fail at startup, never mid-flight."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sir.config import AppConfig, load_config

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.yaml"


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return path


def test_the_shipped_example_config_loads():
    config = load_config(EXAMPLE)
    assert config.model_names == ["chat", "translate"]
    assert config.scheduler.max_wait_seconds > config.scheduler.min_residency_seconds
    assert config.model_by_name("chat").estimated_load_seconds == 8.0


# ---------------------------------------------------------------- served model names


def test_the_served_name_defaults_to_the_internal_name():
    config = AppConfig.model_validate({"models": [{"name": "chat"}]})
    assert config.served_model_names == ["chat"]
    assert config.model_by_served_name("chat").name == "chat"


def test_clients_address_the_served_tag_not_the_internal_label():
    """The internal name is a logging convenience and must not be routable."""
    config = AppConfig.model_validate(
        {"models": [{"name": "chat", "served_model_name": "Qwen/Qwen3-8B"}]}
    )
    assert config.served_model_names == ["Qwen/Qwen3-8B"]
    assert config.model_by_served_name("Qwen/Qwen3-8B").name == "chat"
    assert config.model_by_served_name("chat") is None


def test_several_aliases_are_allowed_and_the_first_is_canonical():
    """Mirrors vLLM's repeated `--served-model-name`, for migrating services."""
    config = AppConfig.model_validate(
        {
            "models": [
                {"name": "translate", "served_model_name": ["facebook/nllb-200", "translate"]}
            ]
        }
    )
    model = config.model_by_name("translate")
    assert model.served_as == "facebook/nllb-200"
    assert config.model_by_served_name("translate") is model
    assert config.model_by_served_name("facebook/nllb-200") is model


def test_two_models_cannot_claim_the_same_served_tag():
    """Otherwise routing would depend on config order."""
    with pytest.raises(ValidationError, match="duplicate served_model_name"):
        AppConfig.model_validate(
            {
                "models": [
                    {"name": "a", "served_model_name": "shared/tag"},
                    {"name": "b", "served_model_name": ["other/tag", "shared/tag"]},
                ]
            }
        )


def test_an_empty_served_name_is_rejected():
    with pytest.raises(ValidationError, match="empty served_model_name"):
        AppConfig.model_validate(
            {"models": [{"name": "chat", "served_model_name": "  "}]}
        )


def test_a_missing_file_says_so():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_duplicate_model_names_are_rejected(tmp_path):
    path = write(tmp_path, "models:\n  - name: chat\n  - name: chat\n")
    with pytest.raises(ValidationError, match="duplicate model name"):
        load_config(path)


def test_at_least_one_model_is_required(tmp_path):
    path = write(tmp_path, "models: []\n")
    with pytest.raises(ValidationError):
        load_config(path)


def test_a_ceiling_below_the_hysteresis_window_is_rejected():
    """Otherwise the backstop fires before hysteresis can ever hold anything."""
    with pytest.raises(ValidationError, match="max_wait_seconds must exceed"):
        AppConfig.model_validate(
            {
                "models": [{"name": "chat"}],
                "scheduler": {"min_residency_seconds": 60, "max_wait_seconds": 30},
            }
        )


def test_unknown_keys_are_rejected_rather_than_silently_ignored(tmp_path):
    """A typo'd knob that quietly does nothing is worse than a startup failure."""
    path = write(tmp_path, "models:\n  - name: chat\n    prioritY: 2\n")
    with pytest.raises(ValidationError):
        load_config(path)


def test_an_unknown_backend_kind_is_rejected(tmp_path):
    path = write(tmp_path, "models:\n  - name: chat\n    backend: tensorrt\n")
    with pytest.raises(ValidationError):
        load_config(path)


def test_a_non_mapping_root_is_rejected(tmp_path):
    path = write(tmp_path, "- chat\n- translate\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(path)


def test_nonsense_numbers_are_rejected():
    for scheduler in (
        {"tick_interval_seconds": 0},
        {"max_concurrent_requests": 0},
        {"max_wait_seconds": -1},
        {"age_weight": -1},
    ):
        with pytest.raises(ValidationError):
            AppConfig.model_validate(
                {"models": [{"name": "chat"}], "scheduler": scheduler}
            )
