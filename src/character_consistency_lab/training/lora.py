"""CPU-safe planning seam for remote LoRA training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import re
import shlex
import shutil
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


class DiffusersTrainingBackend:
    """Execute the official Diffusers trainer and translate its progress to artifacts."""

    _PROGRESS = re.compile(
        r"(?P<step>\d+)/(?P<total>\d+).*?loss[=:]\s*(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    )

    def __init__(self, trainer_script: str | Path) -> None:
        self.trainer_script = Path(trainer_script)

    def train(
        self,
        plan: TrainingPlan,
        config: LoRATrainingConfig,
        on_step: Callable[[TrainingStep], None],
    ) -> TrainingResult:
        prepared = plan.output_dir / "input_dataset"
        if not (prepared / "metadata.jsonl").is_file():
            raise ConfigurationError(
                f"prepared training dataset is missing: {prepared}; run --prepare-data first"
            )
        if not self.trainer_script.is_file():
            raise ConfigurationError(f"Diffusers trainer script does not exist: {self.trainer_script}")

        command = build_diffusers_training_command(plan, config, self.trainer_script)
        log_path = plan.output_dir / "trainer.log"
        observed_steps: list[TrainingStep] = []
        tail: list[str] = []
        with log_path.open("w", encoding="utf-8") as log:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                raise ConfigurationError(f"could not start Diffusers trainer: {exc}") from exc
            assert process.stdout is not None
            line = ""
            while True:
                character = process.stdout.read(1)
                if character == "":
                    if line:
                        self._record_line(line, config.steps, observed_steps, on_step)
                        log.write(line)
                    break
                log.write(character)
                log.flush()
                if character in "\r\n":
                    if line:
                        tail.append(line)
                        tail[:] = tail[-20:]
                        self._record_line(line, config.steps, observed_steps, on_step)
                    line = ""
                else:
                    line += character
            return_code = process.wait()

        if return_code != 0:
            detail = " | ".join(tail[-3:])
            suffix = f": {detail}" if detail else ""
            raise ConfigurationError(
                f"Diffusers trainer exited with status {return_code}{suffix}; see {log_path}"
            )

        weights = plan.output_dir / "pytorch_lora_weights.safetensors"
        samples = tuple(sorted(plan.output_dir.glob("image_*.png")))
        return TrainingResult(
            weights_path=weights,
            loss_history=tuple(observed_steps),
            sample_paths=samples,
            backend_metadata={
                "name": "diffusers",
                "trainer_script": str(self.trainer_script.resolve()),
                "command": list(command),
                "log": str(log_path.resolve()),
                "return_code": return_code,
            },
        )

    @classmethod
    def _record_line(
        cls,
        line: str,
        expected_steps: int,
        observed_steps: list[TrainingStep],
        on_step: Callable[[TrainingStep], None],
    ) -> None:
        match = cls._PROGRESS.search(line)
        if match is None:
            return
        step = int(match.group("step"))
        total = int(match.group("total"))
        if total != expected_steps or (observed_steps and step <= observed_steps[-1].step):
            return
        item = TrainingStep(step=step, loss=float(match.group("loss")))
        on_step(item)
        observed_steps.append(item)


def prepare_diffusers_dataset(plan: TrainingPlan) -> Path:
    """Create an ImageFolder snapshot containing only training records.

    Diffusers' DreamBooth scripts can load this directory with ``--dataset_name``
    and retain each manifest caption through ``--caption_column text``.
    """

    manifest = load_dataset(plan.dataset)
    issues = validate_dataset(manifest)
    if issues:
        summary = "; ".join(f"[{issue.code}] {issue.message}" for issue in issues[:3])
        raise ConfigurationError(f"training dataset is invalid: {summary}")

    records = [record for record in manifest.records if record.split.value == "train"]
    if not records:
        raise ConfigurationError("training dataset has no records in the train split")

    destination = plan.output_dir / "input_dataset"
    if destination.exists():
        raise ConfigurationError(
            f"prepared training dataset already exists: {destination}; "
            "move or remove it explicitly before rebuilding"
        )
    destination.mkdir(parents=True)
    metadata_lines: list[str] = []
    for index, record in enumerate(records, start=1):
        source = manifest.image_path(record)
        filename = f"{index:04d}{source.suffix.lower()}"
        shutil.copy2(source, destination / filename)
        metadata_lines.append(json.dumps({"file_name": filename, "text": record.caption}))
    (destination / "metadata.jsonl").write_text(
        "\n".join(metadata_lines) + "\n", encoding="utf-8"
    )
    return destination


def build_diffusers_training_command(
    plan: TrainingPlan,
    config: LoRATrainingConfig,
    trainer_script: str | Path,
) -> tuple[str, ...]:
    """Build the official Accelerate/Diffusers FLUX.2 Klein LoRA command."""

    command = [
        "accelerate",
        "launch",
        str(Path(trainer_script)),
        "--pretrained_model_name_or_path",
        config.model,
        "--dataset_name",
        str(plan.output_dir / "input_dataset"),
        "--image_column",
        "image",
        "--caption_column",
        "text",
        "--instance_prompt",
        config.trigger_token,
        "--output_dir",
        str(plan.output_dir),
        "--resolution",
        str(config.training_resolution),
        "--rank",
        str(config.rank),
        "--learning_rate",
        str(config.learning_rate),
        "--max_train_steps",
        str(config.steps),
        "--train_batch_size",
        str(config.batch_size),
        "--gradient_accumulation_steps",
        str(config.gradient_accumulation),
        "--mixed_precision",
        config.mixed_precision,
        "--seed",
        str(config.seed),
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "0",
    ]
    if config.model_revision is not None:
        command.extend(("--revision", config.model_revision))
    return tuple(command)


def format_training_command(command: tuple[str, ...]) -> str:
    """Render an argv tuple as a shell-safe command for a remote operator."""

    return shlex.join(command)


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
