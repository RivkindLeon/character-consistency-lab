"""CPU-safe planning seam for remote LoRA training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
