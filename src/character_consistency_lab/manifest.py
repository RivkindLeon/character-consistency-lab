from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    seed: int
    prompt: str
    negative_prompt: str
    tags: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "seed": self.seed,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "tags": self.tags,
        }


DIMENSION_KEYS = {
    "shots": "shot",
    "expressions": "expression",
    "actions": "action",
    "backgrounds": "background",
    "outfits": "outfit",
    "lighting": "lighting",
}


def load_spec(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _normalize_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _derive_seed(base_seed: int, experiment_name: str, sample_id: str) -> int:
    digest = sha256(f"{experiment_name}:{sample_id}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big")
    return (base_seed + offset) % (2**31 - 1)


def _compose_prompt(
    base_prompt: str,
    character_name: str,
    identity_traits: list[str],
    consistency_rules: list[str],
    tags: dict[str, str],
) -> str:
    parts = [base_prompt.strip()]

    if character_name:
        parts.append(f"character: {character_name}")
    if identity_traits:
        parts.append("identity traits: " + ", ".join(identity_traits))
    if consistency_rules:
        parts.append("consistency locks: " + ", ".join(consistency_rules))

    ordered_tag_values = [value for _, value in sorted(tags.items()) if value]
    if ordered_tag_values:
        parts.append("scene variations: " + ", ".join(ordered_tag_values))

    return "; ".join(parts)


def generate_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    experiment = spec.get("experiment", {})
    character = spec.get("character", {})
    consistency = spec.get("consistency", {})
    variants = spec.get("variants", {})

    experiment_name = experiment.get("name", "unnamed-experiment")
    base_prompt = experiment.get("base_prompt", "character concept art")
    negative_prompt = experiment.get("negative_prompt", "")
    base_seed = int(experiment.get("base_seed", 1))

    character_name = str(character.get("name", "")).strip()
    identity_traits = _normalize_items(character.get("identity"))
    consistency_rules = _normalize_items(consistency.get("always"))

    dimensions: list[tuple[str, list[str]]] = []
    for plural_key, singular_key in DIMENSION_KEYS.items():
        values = _normalize_items(variants.get(plural_key)) or [""]
        dimensions.append((singular_key, values))

    samples: list[ManifestSample] = []
    for combination in product(*(values for _, values in dimensions)):
        tags = {
            key: value
            for (key, _), value in zip(dimensions, combination, strict=True)
            if value
        }
        tag_suffix = "-".join(
            value.lower().replace(" ", "-").replace(",", "") for value in tags.values()
        ) or "base"
        sample_id = f"{experiment_name}-{len(samples) + 1:03d}-{tag_suffix}"
        samples.append(
            ManifestSample(
                sample_id=sample_id,
                seed=_derive_seed(base_seed, experiment_name, sample_id),
                prompt=_compose_prompt(
                    base_prompt=base_prompt,
                    character_name=character_name,
                    identity_traits=identity_traits,
                    consistency_rules=consistency_rules,
                    tags=tags,
                ),
                negative_prompt=negative_prompt,
                tags=tags,
            )
        )

    return {
        "experiment": {
            "name": experiment_name,
            "base_prompt": base_prompt,
            "negative_prompt": negative_prompt,
            "base_seed": base_seed,
        },
        "character": {
            "name": character_name,
            "identity": identity_traits,
        },
        "consistency": {
            "always": consistency_rules,
        },
        "sample_count": len(samples),
        "samples": [sample.as_dict() for sample in samples],
    }


def manifest_to_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
