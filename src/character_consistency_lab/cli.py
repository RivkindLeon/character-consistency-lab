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
