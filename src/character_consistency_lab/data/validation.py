"""Filesystem and image-content validation for character datasets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
import yaml

from .dataset import DatasetManifest, DatasetSchemaError, DatasetSplit


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


def load_dataset(root: str | Path) -> DatasetManifest:
    """Load the conventional characters.yaml + manifest.jsonl dataset layout."""

    dataset_root = Path(root)
    metadata_path = dataset_root / "characters.yaml"
    manifest_path = dataset_root / "manifest.jsonl"
    try:
        metadata: Any = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetSchemaError(f"cannot read {metadata_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise DatasetSchemaError(f"invalid YAML in {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict) or not isinstance(metadata.get("characters"), list):
        raise DatasetSchemaError("characters.yaml must contain a 'characters' list")
    try:
        return DatasetManifest.from_jsonl(
            manifest_path, metadata["characters"], root=dataset_root
        )
    except OSError as exc:
        raise DatasetSchemaError(f"cannot read {manifest_path}: {exc}") from exc


def validate_dataset(manifest: DatasetManifest) -> tuple[ValidationIssue, ...]:
    """Return every discoverable dataset problem without changing source files."""

    issues: list[ValidationIssue] = []
    seen_paths: set[Path] = set()
    content_splits: dict[str, set[DatasetSplit]] = {}
    content_paths: dict[str, Path] = {}

    for record in manifest.records:
        relative_path = record.image
        if ".." in relative_path.parts:
            issues.append(
                ValidationIssue(
                    "invalid_path", f"image path escapes dataset root: {relative_path}"
                )
            )
            continue
        if relative_path in seen_paths:
            issues.append(
                ValidationIssue("duplicate_filename", f"duplicate image path: {relative_path}")
            )
            continue
        seen_paths.add(relative_path)

        image_path = manifest.image_path(record)
        if not image_path.is_file():
            issues.append(ValidationIssue("missing_image", f"missing image: {relative_path}"))
            continue

        try:
            content = image_path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("width and height must be positive")
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            issues.append(
                ValidationIssue("invalid_image", f"invalid image {relative_path}: {exc}")
            )
            continue

        content_splits.setdefault(digest, set()).add(record.split)
        content_paths.setdefault(digest, relative_path)

    for digest, splits in content_splits.items():
        if DatasetSplit.TRAIN in splits and DatasetSplit.REFERENCE in splits:
            issues.append(
                ValidationIssue(
                    "train_reference_leakage",
                    f"identical image content appears in train and reference splits: "
                    f"{content_paths[digest]}",
                )
            )

    return tuple(issues)
