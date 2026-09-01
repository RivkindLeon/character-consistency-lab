"""Typed, model-independent dataset manifest contracts.

The manifest describes source images only. Loading it never opens or modifies an
image; filesystem and image-content checks belong to the validation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class DatasetSchemaError(ValueError):
    """Raised when dataset metadata does not conform to the manifest schema."""


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    REFERENCE = "reference"


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetSchemaError(f"'{field}' must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class Character:
    """A recurring character and the token used to identify it in prompts."""

    id: str
    trigger: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Character:
        return cls(
            id=_required_text(value.get("id"), "character.id"),
            trigger=_required_text(value.get("trigger"), "character.trigger"),
        )


@dataclass(frozen=True)
class DatasetRecord:
    """One image/caption pair assigned to a character and dataset split."""

    image: Path
    character: str
    caption: str
    split: DatasetSplit

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DatasetRecord:
        image_text = _required_text(value.get("image"), "record.image")
        image = Path(image_text)
        if image.is_absolute():
            raise DatasetSchemaError("'record.image' must be relative to the dataset root")

        try:
            split = DatasetSplit(_required_text(value.get("split"), "record.split"))
        except ValueError as exc:
            allowed = ", ".join(item.value for item in DatasetSplit)
            raise DatasetSchemaError(f"'record.split' must be one of: {allowed}") from exc

        return cls(
            image=image,
            character=_required_text(value.get("character"), "record.character"),
            caption=_required_text(value.get("caption"), "record.caption"),
            split=split,
        )


@dataclass(frozen=True)
class DatasetManifest:
    """The complete typed description of a character image dataset."""

    root: Path
    characters: tuple[Character, ...]
    records: tuple[DatasetRecord, ...]

    def __post_init__(self) -> None:
        character_ids = [character.id for character in self.characters]
        if not character_ids:
            raise DatasetSchemaError("manifest must define at least one character")
        if len(set(character_ids)) != len(character_ids):
            raise DatasetSchemaError("character IDs must be unique")

        triggers = [character.trigger for character in self.characters]
        if len(set(triggers)) != len(triggers):
            raise DatasetSchemaError("character triggers must be unique")

        known_ids = set(character_ids)
        for record in self.records:
            if record.character not in known_ids:
                raise DatasetSchemaError(
                    f"record references unknown character ID '{record.character}'"
                )

    @classmethod
    def from_data(
        cls,
        root: str | Path,
        characters: Iterable[Mapping[str, Any]],
        records: Iterable[Mapping[str, Any]],
    ) -> DatasetManifest:
        """Build a manifest from already-parsed YAML/JSON-compatible data."""

        return cls(
            root=Path(root),
            characters=tuple(Character.from_mapping(item) for item in characters),
            records=tuple(DatasetRecord.from_mapping(item) for item in records),
        )

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        characters: Iterable[Mapping[str, Any]],
        *,
        root: str | Path | None = None,
    ) -> DatasetManifest:
        """Load newline-delimited JSON records without touching image files."""

        manifest_path = Path(path)
        parsed_records: list[Mapping[str, Any]] = []
        with manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetSchemaError(
                        f"invalid JSON on manifest line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise DatasetSchemaError(
                        f"manifest line {line_number} must contain a JSON object"
                    )
                parsed_records.append(value)

        return cls.from_data(
            root=manifest_path.parent if root is None else root,
            characters=characters,
            records=parsed_records,
        )

    def image_path(self, record: DatasetRecord) -> Path:
        """Resolve a record path without asserting that the image exists."""

        return self.root / record.image
