"""Run a fixed benchmark through a model backend with complete metadata."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from pydantic import Field, field_validator
import yaml

from ..benchmarks import load_benchmark
from ..config import ConfigurationError, StrictModel
from ..models import GenerationRequest, create_backend, load_backend_config


class GenerationConfig(StrictModel):
    width: int = Field(default=1024, gt=0)
    height: int = Field(default=1024, gt=0)
    steps: int = Field(default=28, gt=0)
    guidance: float = Field(default=3.5, ge=0)
    negative_prompt: str | None = None
    adapter_configuration: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(StrictModel):
    name: str
    benchmark: str
    model: str
    output_dir: str
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @field_validator("name", "benchmark", "model", "output_dir")
    @classmethod
    def require_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot load experiment configuration: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError("experiment configuration root must be a YAML mapping")
    try:
        return ExperimentConfig.model_validate(data)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def run_experiment(config_path: str | Path, *, dry_run: bool) -> Path:
    """Execute every benchmark scene and write one self-contained metadata file."""

    config_path = Path(config_path)
    config = load_experiment_config(config_path)
    benchmark = load_benchmark(config.benchmark)
    model_config = load_backend_config(config.model)
    output_dir = Path(config.output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    backend = create_backend(model_config, dry_run=dry_run)
    with backend:
        for scene in benchmark.scenes:
            request = GenerationRequest(
                prompt=scene.prompt,
                negative_prompt=config.generation.negative_prompt,
                seed=scene.seed,
                width=config.generation.width,
                height=config.generation.height,
                steps=config.generation.steps,
                guidance=config.generation.guidance,
                output_path=images_dir / f"{scene.id}.png",
                adapter_config=config.generation.adapter_configuration,
            )
            result = backend.generate(request)
            records.append(
                {
                    "scene_id": scene.id,
                    "category": scene.category,
                    "characters": list(scene.characters),
                    "model": result.model_id,
                    "model_revision": result.model_revision,
                    "lora": request.adapter_config.get("lora"),
                    "lora_weight": request.adapter_config.get("weight"),
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "seed": request.seed,
                    "width": request.width,
                    "height": request.height,
                    "steps": request.steps,
                    "guidance": request.guidance,
                    "adapter_configuration": dict(request.adapter_config),
                    "image": str(result.image_path) if result.image_path else None,
                    "planned_image": str(request.output_path),
                    "backend_metadata": dict(result.metadata),
                    "dry_run": result.dry_run,
                }
            )

    snapshot_path = output_dir / "config.yaml"
    snapshot_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    metadata = {
        "experiment": config.name,
        "timestamp": started_at,
        "git_commit_sha": _git_commit(),
        "benchmark": {"path": str(config.benchmark), "version": benchmark.version},
        "backend": backend.name,
        "dry_run": dry_run,
        "generation_count": len(records),
        "generations": records,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path
