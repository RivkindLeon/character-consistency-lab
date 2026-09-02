"""Typed YAML configuration for prompt-manifest experiments."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
import yaml


class ConfigurationError(ValueError):
    """Raised when an experiment configuration cannot be loaded or validated."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ExperimentConfig(StrictModel):
    name: str
    base_prompt: str
    negative_prompt: str = ""
    base_seed: int = Field(default=1, ge=0)

    @field_validator("name", "base_prompt")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()


class CharacterConfig(StrictModel):
    name: str = ""
    identity: list[str] = Field(default_factory=list)


class ConsistencyConfig(StrictModel):
    always: list[str] = Field(default_factory=list)


class VariantsConfig(StrictModel):
    shots: list[str] = Field(default_factory=list)
    expressions: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    backgrounds: list[str] = Field(default_factory=list)
    outfits: list[str] = Field(default_factory=list)
    lighting: list[str] = Field(default_factory=list)

    @field_validator("shots", "expressions", "actions", "backgrounds", "outfits", "lighting")
    @classmethod
    def reject_empty_dimensions(cls, value: list[str]) -> list[str]:
        if value and not all(item.strip() for item in value):
            raise ValueError("values must be non-empty strings")
        return [item.strip() for item in value]


class RenderConfig(StrictModel):
    model_id: str | None = None
    lora_adapter: str | None = None
    guidance_scale: float | None = Field(default=None, ge=0)
    num_inference_steps: int | None = Field(default=None, ge=1)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    lora_scale: float | None = Field(default=None, ge=0)


class SweepsConfig(StrictModel):
    model_ids: list[str] | None = None
    lora_adapters: list[str] | None = None
    guidance_scales: list[float] | None = None
    num_inference_steps: list[int] | None = None
    widths: list[int] | None = None
    heights: list[int] | None = None
    lora_scales: list[float] | None = None

    @field_validator("model_ids", "lora_adapters", "guidance_scales", "num_inference_steps", "widths", "heights", "lora_scales")
    @classmethod
    def reject_empty_sweeps(cls, value: list[object] | None) -> list[object] | None:
        if value is not None and not value:
            raise ValueError("must contain at least one value")
        return value

    @field_validator("num_inference_steps", "widths", "heights")
    @classmethod
    def require_positive_integers(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and any(item < 1 for item in value):
            raise ValueError("values must be at least 1")
        return value

    @field_validator("guidance_scales", "lora_scales")
    @classmethod
    def require_non_negative_numbers(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and any(item < 0 for item in value):
            raise ValueError("values must be non-negative")
        return value


class ManifestConfig(StrictModel):
    experiment: ExperimentConfig
    character: CharacterConfig = Field(default_factory=CharacterConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)
    variants: VariantsConfig = Field(default_factory=VariantsConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    sweeps: SweepsConfig = Field(default_factory=SweepsConfig)

    def as_manifest_spec(self) -> dict[str, object]:
        return self.model_dump(exclude_none=True)


def load_config(path: str | Path) -> ManifestConfig:
    config_path = Path(path)
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError("experiment configuration must be a .yaml or .yml file")

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigurationError("configuration root must be a YAML mapping")
    try:
        return ManifestConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(str(exc)) from exc
