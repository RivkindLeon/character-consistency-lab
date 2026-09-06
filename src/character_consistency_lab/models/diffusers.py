"""Optional Diffusers inference backend with lazy heavyweight imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator
import yaml

from ..config import ConfigurationError, StrictModel
from .base import DryRunBackend, GenerationRequest, GenerationResult, ModelBackend


class DiffusersBackendConfig(StrictModel):
    backend: Literal["flux", "sdxl"]
    model_id: str
    revision: str | None = None
    device: str = "cuda"
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"

    @field_validator("model_id", "device")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()


def load_backend_config(path: str | Path) -> DiffusersBackendConfig:
    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot load model configuration: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError("model configuration root must be a YAML mapping")
    try:
        return DiffusersBackendConfig.model_validate(data)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc


def create_backend(config: DiffusersBackendConfig, *, dry_run: bool) -> ModelBackend:
    if dry_run:
        return DryRunBackend(
            config.model_id,
            config.revision,
            backend_name=f"{config.backend}-diffusers-dry-run",
        )
    return DiffusersBackend(config)


class DiffusersBackend(ModelBackend):
    """Real backend; imports and model loading happen only on explicit load."""

    # DiffusionPipeline resolves the concrete class from model_index.json. This
    # covers Flux2KleinPipeline as well as StableDiffusionXLPipeline without
    # coupling configuration to a rapidly evolving class name.
    _PIPELINES = {"flux": "DiffusionPipeline", "sdxl": "DiffusionPipeline"}

    def __init__(self, config: DiffusersBackendConfig) -> None:
        self.config = config
        self.pipeline: Any | None = None
        self._torch: Any | None = None

    @property
    def name(self) -> str:
        return f"{self.config.backend}-diffusers"

    def load(self) -> None:
        if self.pipeline is not None:
            return
        try:
            import diffusers
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "real inference requires optional dependencies; install with: "
                "pip install -e '.[inference]'"
            ) from exc
        pipeline_class = getattr(diffusers, self._PIPELINES[self.config.backend])
        self.pipeline = pipeline_class.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            torch_dtype=getattr(torch, self.config.dtype),
        ).to(self.config.device)
        self._torch = torch

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self.pipeline is None or self._torch is None:
            raise RuntimeError("backend must be loaded before generation")
        if request.output_path is None:
            raise ValueError("output_path is required for real inference")
        generator = self._torch.Generator(device=self.config.device).manual_seed(request.seed)
        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "generator": generator,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance,
        }
        if request.negative_prompt is not None:
            kwargs["negative_prompt"] = request.negative_prompt
        image = self.pipeline(**kwargs).images[0]
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(request.output_path)
        return GenerationResult(
            backend=self.name,
            model_id=self.config.model_id,
            model_revision=self.config.revision,
            request=request,
            image_path=request.output_path,
            metadata={"device": self.config.device, "dtype": self.config.dtype},
        )

    def unload(self) -> None:
        self.pipeline = None
        self._torch = None
