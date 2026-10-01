"""Strict configuration contracts for one-character LoRA training."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator
import yaml

from ..config import ConfigurationError, StrictModel


class LoRATrainingConfig(StrictModel):
    """The reproducible inputs required for an initial character LoRA run."""

    model: str
    model_revision: str | None = None
    dataset: str
    output_dir: str
    training_resolution: int = Field(ge=64)
    rank: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    steps: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    gradient_accumulation: int = Field(ge=1)
    mixed_precision: Literal["no", "fp16", "bf16"]
    seed: int = Field(ge=0)
    trigger_token: str

    @field_validator("model", "dataset", "output_dir", "trigger_token")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()


def load_training_config(path: str | Path) -> LoRATrainingConfig:
    config_path = Path(path)
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError("training configuration must be a .yaml or .yml file")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot load training configuration: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError("training configuration root must be a YAML mapping")
    try:
        return LoRATrainingConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(str(exc)) from exc
