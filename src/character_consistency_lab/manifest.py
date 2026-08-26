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
    render_settings: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "seed": self.seed,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "tags": self.tags,
            "render_settings": self.render_settings,
        }


DIMENSION_KEYS = {
    "shots": "shot",
    "expressions": "expression",
    "actions": "action",
    "backgrounds": "background",
    "outfits": "outfit",
    "lighting": "lighting",
}

RENDER_SWEEP_KEYS = {
    "model_ids": "model_id",
    "lora_adapters": "lora_adapter",
    "guidance_scales": "guidance_scale",
    "num_inference_steps": "num_inference_steps",
    "widths": "width",
    "heights": "height",
    "lora_scales": "lora_scale",
}

RENDER_SLUG_KEYS = {
    "model_id": "model",
    "lora_adapter": "adapter",
    "guidance_scale": "gs",
    "num_inference_steps": "steps",
    "width": "w",
    "height": "h",
    "lora_scale": "lora",
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


def _slugify(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "-")
        .replace(",", "")
        .replace(".", "p")
    )


def _coerce_render_value(key: str, value: Any) -> Any:
    if key in {"width", "height", "num_inference_steps"}:
        return int(value)
    if key in {"guidance_scale", "lora_scale"}:
        return float(value)
    if key in {"model_id", "lora_adapter"}:
        return str(value).strip()
    return value


def _build_render_dimensions(spec: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, list[Any]]]]:
    render = spec.get("render", {})
    sweeps = spec.get("sweeps", {})

    render_defaults = {
        key: _coerce_render_value(key, value)
        for key, value in render.items()
        if key in RENDER_SLUG_KEYS and value is not None
    }

    render_dimensions: list[tuple[str, list[Any]]] = []
    for sweep_key, render_key in RENDER_SWEEP_KEYS.items():
        sweep_values = sweeps.get(sweep_key)
        if sweep_values is not None:
            values = [_coerce_render_value(render_key, item) for item in sweep_values]
        elif render_key in render_defaults:
            values = [render_defaults[render_key]]
        else:
            values = [None]
        render_dimensions.append((render_key, values))

    return render_defaults, render_dimensions


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
    render_defaults, render_dimensions = _build_render_dimensions(spec)

    dimensions: list[tuple[str, list[str]]] = []
    for plural_key, singular_key in DIMENSION_KEYS.items():
        values = _normalize_items(variants.get(plural_key)) or [""]
        dimensions.append((singular_key, values))

    samples: list[ManifestSample] = []
    all_dimensions = dimensions + render_dimensions
    for combination in product(*(values for _, values in all_dimensions)):
        scene_values = combination[: len(dimensions)]
        render_values = combination[len(dimensions) :]

        tags = {
            key: value
            for (key, _), value in zip(dimensions, scene_values, strict=True)
            if value
        }
        render_settings = {
            key: value
            for (key, _), value in zip(render_dimensions, render_values, strict=True)
            if value is not None
        }

        sample_slug_parts = [_slugify(value) for value in tags.values()]
        sample_slug_parts.extend(
            f"{RENDER_SLUG_KEYS[key]}-{_slugify(value)}"
            for key, value in render_settings.items()
        )
        tag_suffix = "-".join(sample_slug_parts) or "base"
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
                render_settings=render_settings,
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
        "render": render_defaults,
        "sample_count": len(samples),
        "samples": [sample.as_dict() for sample in samples],
    }


def manifest_to_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
