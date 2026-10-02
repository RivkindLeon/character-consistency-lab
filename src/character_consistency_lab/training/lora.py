"""CPU-safe planning seam for remote LoRA training."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

from ..config import ConfigurationError
from ..data import load_dataset, validate_dataset
from .configs import LoRATrainingConfig


@dataclass(frozen=True)
class TrainingPlan:
    config_path: Path
    model: str
    dataset: Path
    output_dir: Path
    effective_batch_size: int
    steps: int
    seed: int
    trigger_token: str


def create_training_plan(
    config: LoRATrainingConfig, config_path: str | Path
) -> TrainingPlan:
    """Build a deterministic plan without importing ML libraries or loading weights."""

    source = Path(config_path).resolve()
    project_root = source.parent.parent.parent if source.parent.name == "training" else Path.cwd()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (project_root / path).resolve()

    return TrainingPlan(
        config_path=source,
        model=config.model,
        dataset=resolve(config.dataset),
        output_dir=resolve(config.output_dir),
        effective_batch_size=config.batch_size * config.gradient_accumulation,
        steps=config.steps,
        seed=config.seed,
        trigger_token=config.trigger_token,
    )


def check_training_runtime(plan: TrainingPlan, config: LoRATrainingConfig) -> dict[str, Any]:
    """Validate a remote training host without loading model weights."""

    manifest = load_dataset(plan.dataset)
    issues = validate_dataset(manifest)
    if issues:
        summary = "; ".join(f"[{issue.code}] {issue.message}" for issue in issues[:3])
        suffix = f"; and {len(issues) - 3} more" if len(issues) > 3 else ""
        raise ConfigurationError(f"training dataset is invalid: {summary}{suffix}")

    modules: dict[str, Any] = {}
    missing: list[str] = []
    for name in ("accelerate", "diffusers", "peft", "safetensors", "torch", "transformers"):
        try:
            modules[name] = importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise ConfigurationError(
            "real training dependencies are missing: "
            f"{', '.join(missing)}; install with: pip install -e '.[training]'"
        )

    torch = modules["torch"]
    if not torch.cuda.is_available():
        raise ConfigurationError("LoRA training requires CUDA, but CUDA is not available")
    if config.mixed_precision == "bf16":
        supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not supported:
            raise ConfigurationError("configured bfloat16 training is not supported by this CUDA device")

    return {
        "device": "cuda",
        "mixed_precision": config.mixed_precision,
        "torch": getattr(torch, "__version__", "unknown"),
        "diffusers": getattr(modules["diffusers"], "__version__", "unknown"),
        "peft": getattr(modules["peft"], "__version__", "unknown"),
        "dataset_images": len(manifest.records),
    }
