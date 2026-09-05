"""Typed, reusable benchmark scene definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class BenchmarkSchemaError(ValueError):
    """Raised when a benchmark file does not match the scene schema."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkSchemaError(f"'{field}' must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class BenchmarkScene:
    """One prompt and seed that remain fixed across experiment variants."""

    id: str
    category: str
    characters: tuple[str, ...]
    prompt: str
    seed: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BenchmarkScene:
        characters = value.get("characters")
        if not isinstance(characters, list) or not characters:
            raise BenchmarkSchemaError("'scene.characters' must be a non-empty list")
        parsed_characters = tuple(
            _required_text(character, "scene.characters[]") for character in characters
        )
        if len(set(parsed_characters)) != len(parsed_characters):
            raise BenchmarkSchemaError("'scene.characters' must not contain duplicates")

        seed = value.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise BenchmarkSchemaError("'scene.seed' must be a non-negative integer")

        allowed = {"id", "category", "characters", "prompt", "seed"}
        unknown = set(value) - allowed
        if unknown:
            raise BenchmarkSchemaError(
                "unknown scene field(s): " + ", ".join(sorted(unknown))
            )

        return cls(
            id=_required_text(value.get("id"), "scene.id"),
            category=_required_text(value.get("category"), "scene.category"),
            characters=parsed_characters,
            prompt=_required_text(value.get("prompt"), "scene.prompt"),
            seed=seed,
        )


@dataclass(frozen=True)
class Benchmark:
    """A versioned collection of scenes for controlled comparisons."""

    version: int
    scenes: tuple[BenchmarkScene, ...]

    def __post_init__(self) -> None:
        if self.version < 1:
            raise BenchmarkSchemaError("'version' must be a positive integer")
        if not self.scenes:
            raise BenchmarkSchemaError("benchmark must contain at least one scene")
        ids = [scene.id for scene in self.scenes]
        if len(set(ids)) != len(ids):
            raise BenchmarkSchemaError("scene IDs must be unique")


def load_benchmark(path: str | Path) -> Benchmark:
    """Load a benchmark without deriving or changing its prompts or seeds."""

    benchmark_path = Path(path)
    try:
        data = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise BenchmarkSchemaError(f"invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise BenchmarkSchemaError("benchmark root must be a YAML mapping")
    unknown = set(data) - {"version", "scenes"}
    if unknown:
        raise BenchmarkSchemaError(
            "unknown benchmark field(s): " + ", ".join(sorted(unknown))
        )
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise BenchmarkSchemaError("'version' must be a positive integer")
    scenes = data.get("scenes")
    if not isinstance(scenes, list):
        raise BenchmarkSchemaError("'scenes' must be a list")
    if not all(isinstance(scene, dict) for scene in scenes):
        raise BenchmarkSchemaError("every scene must be a YAML mapping")
    return Benchmark(
        version=version,
        scenes=tuple(BenchmarkScene.from_mapping(scene) for scene in scenes),
    )
