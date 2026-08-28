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
    comparison_group_id: str
    seed: int
    prompt: str
    negative_prompt: str
    tags: dict[str, str]
    render_settings: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "comparison_group_id": self.comparison_group_id,
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


class SpecValidationError(ValueError):
    """Raised when an experiment spec is structurally invalid."""


def load_spec(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _ensure_table(spec: dict[str, Any], key: str) -> dict[str, Any]:
    value = spec.get(key, {})
    if not isinstance(value, dict):
        raise SpecValidationError(f"'{key}' must be a TOML table.")
    return value


def _normalize_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _require_non_empty_string(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise SpecValidationError(f"'{field_name}' must be a non-empty string.")
    return text


def _require_int(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise SpecValidationError(f"'{field_name}' must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SpecValidationError(f"'{field_name}' must be an integer.") from exc
    if minimum is not None and parsed < minimum:
        raise SpecValidationError(f"'{field_name}' must be >= {minimum}.")
    return parsed


def _require_float(value: Any, field_name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise SpecValidationError(f"'{field_name}' must be a number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SpecValidationError(f"'{field_name}' must be a number.") from exc
    if minimum is not None and parsed < minimum:
        raise SpecValidationError(f"'{field_name}' must be >= {minimum}.")
    return parsed


def _validate_string_list(value: Any, field_name: str) -> list[str]:
    items = _normalize_items(value)
    if not items:
        raise SpecValidationError(f"'{field_name}' must contain at least one non-empty value.")
    return items


def validate_spec(spec: dict[str, Any]) -> None:
    experiment = _ensure_table(spec, "experiment")
    _ensure_table(spec, "character")
    _ensure_table(spec, "consistency")
    variants = _ensure_table(spec, "variants")
    render = _ensure_table(spec, "render")
    sweeps = _ensure_table(spec, "sweeps")

    _require_non_empty_string(experiment.get("name", ""), "experiment.name")
    _require_non_empty_string(experiment.get("base_prompt", ""), "experiment.base_prompt")
    _require_int(experiment.get("base_seed", 1), "experiment.base_seed", minimum=0)

    for plural_key in DIMENSION_KEYS:
        if plural_key in variants:
            _validate_string_list(variants[plural_key], f"variants.{plural_key}")

    string_render_fields = {"model_id", "lora_adapter"}
    int_render_fields = {"width", "height", "num_inference_steps"}
    float_render_fields = {"guidance_scale", "lora_scale"}

    for key, value in render.items():
        field_name = f"render.{key}"
        if key in string_render_fields:
            _require_non_empty_string(value, field_name)
        elif key in int_render_fields:
            _require_int(value, field_name, minimum=1)
        elif key in float_render_fields:
            _require_float(value, field_name, minimum=0.0)

    for sweep_key, render_key in RENDER_SWEEP_KEYS.items():
        if sweep_key not in sweeps:
            continue
        values = sweeps[sweep_key]
        field_name = f"sweeps.{sweep_key}"
        if not isinstance(values, list):
            raise SpecValidationError(f"'{field_name}' must be a TOML array.")
        if not values:
            raise SpecValidationError(f"'{field_name}' must contain at least one value.")
        for item in values:
            if render_key in string_render_fields:
                _require_non_empty_string(item, field_name)
            elif render_key in int_render_fields:
                _require_int(item, field_name, minimum=1)
            elif render_key in float_render_fields:
                _require_float(item, field_name, minimum=0.0)


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


def _build_comparison_group_id(experiment_name: str, tags: dict[str, str]) -> str:
    tag_suffix = "-".join(_slugify(value) for value in tags.values()) or "base"
    return f"{experiment_name}-group-{tag_suffix}"


def generate_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    validate_spec(spec)

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
    comparison_groups: dict[str, dict[str, Any]] = {}
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

        comparison_group_id = _build_comparison_group_id(experiment_name, tags)
        sample_slug_parts = [_slugify(value) for value in tags.values()]
        sample_slug_parts.extend(
            f"{RENDER_SLUG_KEYS[key]}-{_slugify(value)}"
            for key, value in render_settings.items()
        )
        tag_suffix = "-".join(sample_slug_parts) or "base"
        sample_id = f"{experiment_name}-{len(samples) + 1:03d}-{tag_suffix}"
        sample = ManifestSample(
            sample_id=sample_id,
            comparison_group_id=comparison_group_id,
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
        samples.append(sample)

        group = comparison_groups.setdefault(
            comparison_group_id,
            {
                "comparison_group_id": comparison_group_id,
                "prompt": sample.prompt,
                "tags": tags,
                "sample_ids": [],
            },
        )
        group["sample_ids"].append(sample_id)

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
        "comparison_groups": list(comparison_groups.values()),
        "sample_count": len(samples),
        "samples": [sample.as_dict() for sample in samples],
    }


def manifest_to_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
