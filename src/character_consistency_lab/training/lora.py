"""CPU-safe planning seam for remote LoRA training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Protocol

import yaml

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


@dataclass(frozen=True)
class TrainingStep:
    """One observed optimizer step from a real training backend."""

    step: int
    loss: float


@dataclass(frozen=True)
class TrainingResult:
    """Artifacts produced by a real backend after successful training."""

    weights_path: Path
    loss_history: tuple[TrainingStep, ...]
    sample_paths: tuple[Path, ...] = ()
    backend_metadata: dict[str, Any] | None = None


class TrainingBackend(Protocol):
    """GPU trainer boundary; tests can replace it without importing ML libraries."""

    def train(
        self,
        plan: TrainingPlan,
        config: LoRATrainingConfig,
        on_step: Callable[[TrainingStep], None],
    ) -> TrainingResult: ...


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_training(
    plan: TrainingPlan,
    config: LoRATrainingConfig,
    backend: TrainingBackend,
) -> Path:
    """Run a backend and durably persist the reproducibility artifacts it reports."""

    manifest = load_dataset(plan.dataset)
    issues = validate_dataset(manifest)
    if issues:
        summary = "; ".join(f"[{issue.code}] {issue.message}" for issue in issues[:3])
        raise ConfigurationError(f"training dataset is invalid: {summary}")

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = plan.output_dir / "config.yaml"
    config_snapshot.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    loss_path = plan.output_dir / "loss_history.json"
    metadata_path = plan.output_dir / "metadata.json"
    losses: list[dict[str, int | float]] = []
    started_at = _utc_now()
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "git_commit": _git_commit(plan.config_path.parent),
        "model": config.model,
        "model_revision": config.model_revision,
        "dataset": str(plan.dataset),
        "dataset_images": len(manifest.records),
        "trigger_token": config.trigger_token,
        "weights": None,
        "samples": [],
    }
    _atomic_json(loss_path, losses)
    _atomic_json(metadata_path, metadata)

    def record_step(item: TrainingStep) -> None:
        if item.step < 1 or not 0 <= item.loss < float("inf"):
            raise ConfigurationError("training backend reported an invalid step or loss")
        if losses and item.step <= losses[-1]["step"]:
            raise ConfigurationError("training backend reported non-increasing step numbers")
        losses.append({"step": item.step, "loss": item.loss})
        _atomic_json(loss_path, losses)

    try:
        result = backend.train(plan, config, record_step)
        weights = result.weights_path.resolve()
        if not weights.is_file():
            raise ConfigurationError(f"training backend did not create weights: {weights}")
        samples = [path.resolve() for path in result.sample_paths]
        missing_samples = [str(path) for path in samples if not path.is_file()]
        if missing_samples:
            raise ConfigurationError(
                f"training backend reported missing sample image(s): {', '.join(missing_samples)}"
            )
        if result.loss_history:
            reported = [{"step": item.step, "loss": item.loss} for item in result.loss_history]
            if reported != losses:
                raise ConfigurationError(
                    "training backend loss history does not match its checkpointed step events"
                )
        if not losses or losses[-1]["step"] != config.steps:
            raise ConfigurationError(
                f"training backend stopped at step {losses[-1]['step'] if losses else 0}; "
                f"expected {config.steps}"
            )
        metadata.update(
            status="completed",
            completed_at=_utc_now(),
            weights=str(weights),
            samples=[str(path) for path in samples],
            backend=result.backend_metadata or {},
        )
        _atomic_json(metadata_path, metadata)
        return metadata_path
    except Exception as exc:
        metadata.update(status="failed", completed_at=_utc_now(), error=str(exc))
        _atomic_json(metadata_path, metadata)
        raise
