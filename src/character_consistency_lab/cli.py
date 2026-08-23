from __future__ import annotations

import argparse
from pathlib import Path

from .manifest import generate_manifest, load_spec, manifest_to_json


def build_manifest(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    manifest = generate_manifest(spec)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(manifest_to_json(manifest), encoding="utf-8")
    print(f"Wrote {manifest['sample_count']} prompts to {output}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Character Consistency Lab tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-manifest",
        help="Expand a TOML experiment spec into a reproducible prompt manifest.",
    )
    build.add_argument("--spec", required=True, help="Path to a TOML experiment spec.")
    build.add_argument("--output", required=True, help="Path to the generated JSON manifest.")
    build.set_defaults(func=build_manifest)

    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
