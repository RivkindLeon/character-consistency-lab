"""Deterministic summary statistics for character datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from PIL import Image

from .dataset import DatasetManifest, DatasetSplit


@dataclass(frozen=True)
class DatasetStats:
    """Counts by character/split and source-image resolution."""

    counts: dict[str, dict[DatasetSplit, int]]
    resolutions: dict[tuple[int, int], int]


def calculate_dataset_stats(manifest: DatasetManifest) -> DatasetStats:
    """Calculate stats without modifying source images.

    Callers should validate the dataset first so every manifest image is known
    to exist and be readable.
    """

    counts = {
        character.id: {split: 0 for split in DatasetSplit}
        for character in manifest.characters
    }
    resolutions: Counter[tuple[int, int]] = Counter()
    for record in manifest.records:
        counts[record.character][record.split] += 1
        with Image.open(manifest.image_path(record)) as image:
            resolutions[(image.width, image.height)] += 1

    return DatasetStats(counts=counts, resolutions=dict(sorted(resolutions.items())))


def format_dataset_stats(manifest: DatasetManifest, stats: DatasetStats) -> str:
    """Render stable, human-readable CLI output."""

    lines = [f"Characters: {len(manifest.characters)}"]
    for character in manifest.characters:
        lines.extend(("", character.id))
        for split in DatasetSplit:
            lines.append(f"{split.value}: {stats.counts[character.id][split]}")

    lines.extend(("", "Resolution distribution:"))
    if stats.resolutions:
        lines.extend(
            f"{width}x{height}: {count}"
            for (width, height), count in stats.resolutions.items()
        )
    else:
        lines.append("(no images)")
    return "\n".join(lines)
