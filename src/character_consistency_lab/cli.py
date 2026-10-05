from __future__ import annotations

import argparse
from pathlib import Path

from .config import ConfigurationError
from .data import (
    DatasetSchemaError,
    calculate_dataset_stats,
    format_dataset_stats,
    load_dataset,
    validate_dataset,
)
from .manifest import SpecValidationError, generate_manifest, load_spec, manifest_to_json, validate_spec
from .experiments import check_experiment_runtime, run_experiment
from .training import (
    DiffusersTrainingBackend,
    build_diffusers_training_command,
    check_training_runtime,
    create_training_plan,
    format_training_command,
    load_training_config,
    prepare_diffusers_dataset,
    run_training,
)


def build_manifest(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    manifest = generate_manifest(spec)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(manifest_to_json(manifest), encoding="utf-8")
    print(f"Wrote {manifest['sample_count']} prompts to {output}")
    return 0


def validate_manifest_spec(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    validate_spec(spec)
    print(f"Spec is valid: {args.spec}")
    return 0


def validate_dataset_command(args: argparse.Namespace) -> int:
    manifest = load_dataset(args.root)
    issues = validate_dataset(manifest)
    if issues:
        for issue in issues:
            print(f"[{issue.code}] {issue.message}")
        print(f"Dataset is invalid: {len(issues)} issue(s)")
        return 1
    print(f"Dataset is valid: {args.root} ({len(manifest.records)} images)")
    return 0


def dataset_stats_command(args: argparse.Namespace) -> int:
    manifest = load_dataset(args.root)
    issues = validate_dataset(manifest)
    if issues:
        for issue in issues:
            print(f"[{issue.code}] {issue.message}")
        print(f"Cannot calculate stats: dataset has {len(issues)} issue(s)")
        return 1
    print(format_dataset_stats(manifest, calculate_dataset_stats(manifest)))
    return 0


def generate_command(args: argparse.Namespace) -> int:
    if args.check_runtime:
        runtime = check_experiment_runtime(args.experiment)
        print(
            "Runtime ready: "
            f"{runtime['backend']} on {runtime['device']} ({runtime['dtype']}), "
            f"torch {runtime['torch']}, diffusers {runtime['diffusers']}"
        )
        return 0
    metadata_path = run_experiment(args.experiment, dry_run=args.dry_run, resume=args.resume)
    mode = "Dry run" if args.dry_run else "Generation"
    print(f"{mode} complete: {metadata_path}")
    return 0


def train_command(args: argparse.Namespace) -> int:
    config = load_training_config(args.config)
    plan = create_training_plan(config, args.config)
    if args.check_runtime:
        runtime = check_training_runtime(plan, config)
        print(
            "Training runtime ready: "
            f"{runtime['device']} ({runtime['mixed_precision']}), "
            f"torch {runtime['torch']}, diffusers {runtime['diffusers']}, "
            f"peft {runtime['peft']}; dataset {runtime['dataset_images']} image(s)"
        )
        return 0
    if args.prepare_data:
        path = prepare_diffusers_dataset(plan)
        print(f"Prepared {path}")
        return 0
    if args.emit_command:
        command = build_diffusers_training_command(plan, config, args.trainer_script)
        print(format_training_command(command))
        return 0
    if args.execute:
        metadata = run_training(plan, config, DiffusersTrainingBackend(args.trainer_script))
        print(f"Training complete: {metadata}")
        return 0
    print("LoRA training dry run (no model loaded)")
    print(f"Model: {plan.model}")
    print(f"Dataset: {plan.dataset}")
    print(f"Output: {plan.output_dir}")
    print(f"Steps: {plan.steps}")
    print(f"Effective batch size: {plan.effective_batch_size}")
    print(f"Seed: {plan.seed}")
    print(f"Trigger token: {plan.trigger_token}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Character Consistency Lab tools")
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-manifest",
        help="Expand a YAML experiment config into a reproducible prompt manifest.",
    )
    build.add_argument("--spec", required=True, help="Path to a YAML experiment config.")
    build.add_argument("--output", required=True, help="Path to the generated JSON manifest.")
    build.set_defaults(func=build_manifest)

    validate = subparsers.add_parser(
        "validate-spec",
        help="Validate a YAML experiment config before running downstream jobs.",
    )
    validate.add_argument("--spec", required=True, help="Path to a YAML experiment config.")
    validate.set_defaults(func=validate_manifest_spec)

    dataset = subparsers.add_parser("dataset", help="Inspect character datasets.")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_validate = dataset_commands.add_parser(
        "validate", help="Validate dataset metadata, files, images, and split isolation."
    )
    dataset_validate.add_argument("root", help="Dataset directory.")
    dataset_validate.set_defaults(func=validate_dataset_command)
    dataset_stats = dataset_commands.add_parser(
        "stats", help="Report image counts by character/split and resolution."
    )
    dataset_stats.add_argument("root", help="Dataset directory.")
    dataset_stats.set_defaults(func=dataset_stats_command)

    generate = subparsers.add_parser(
        "generate", help="Generate a fixed benchmark and save reproducibility metadata."
    )
    generate.add_argument("--experiment", required=True, help="Experiment YAML configuration.")
    generate_mode = generate.add_mutually_exclusive_group()
    generate_mode.add_argument(
        "--dry-run", action="store_true", help="Record all planned generations without loading a model."
    )
    generate_mode.add_argument(
        "--resume",
        action="store_true",
        help="Continue a real interrupted run using its checkpointed metadata and images.",
    )
    generate_mode.add_argument(
        "--check-runtime",
        action="store_true",
        help="Verify dependencies and accelerator support without downloading model weights.",
    )
    generate.set_defaults(func=generate_command)

    train = subparsers.add_parser(
        "train", help="Validate and inspect a one-character LoRA training run."
    )
    train.add_argument("--config", required=True, help="LoRA training YAML configuration.")
    train_mode = train.add_mutually_exclusive_group(required=True)
    train_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without importing ML libraries or training.",
    )
    train_mode.add_argument(
        "--prepare-data",
        action="store_true",
        help="Snapshot train-split images and captions for the Diffusers trainer.",
    )
    train_mode.add_argument(
        "--emit-command",
        action="store_true",
        help="Print a shell-safe Accelerate command for the official FLUX.2 Klein trainer.",
    )
    train.add_argument(
        "--trainer-script",
        default="train_dreambooth_lora_flux2_klein.py",
        help="Path to Diffusers' train_dreambooth_lora_flux2_klein.py example.",
    )
    train_mode.add_argument(
        "--execute",
        action="store_true",
        help="Run the official trainer and checkpoint observed losses and artifacts.",
    )
    train_mode.add_argument(
        "--check-runtime",
        action="store_true",
        help="Verify the dataset, dependencies, CUDA, and precision without loading weights.",
    )
    train.set_defaults(func=train_command)

    return parser


def main() -> int:
    parser = make_parser()
    try:
        args = parser.parse_args()
        return args.func(args)
    except DatasetSchemaError as exc:
        parser.exit(status=2, message=f"Dataset validation failed: {exc}\n")
    except (OSError, ConfigurationError, SpecValidationError) as exc:
        parser.exit(status=2, message=f"Spec validation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
