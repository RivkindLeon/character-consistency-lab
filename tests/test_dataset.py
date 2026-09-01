import tempfile
import unittest
from pathlib import Path

from character_consistency_lab.data import (
    DatasetManifest,
    DatasetSchemaError,
    DatasetSplit,
)


CHARACTERS = [
    {"id": "dino", "trigger": "chr_dino"},
    {"id": "tira", "trigger": "chr_tira"},
]


class DatasetManifestTests(unittest.TestCase):
    def test_builds_typed_manifest_and_resolves_image_path(self) -> None:
        manifest = DatasetManifest.from_data(
            "/datasets/story",
            CHARACTERS,
            [
                {
                    "image": "images/dino_001.png",
                    "character": "dino",
                    "caption": "chr_dino standing in a sunny meadow",
                    "split": "train",
                }
            ],
        )

        self.assertEqual(manifest.records[0].split, DatasetSplit.TRAIN)
        self.assertEqual(
            manifest.image_path(manifest.records[0]),
            Path("/datasets/story/images/dino_001.png"),
        )

    def test_loads_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            path.write_text(
                '{"image":"references/tira.png","character":"tira",'
                '"caption":"chr_tira portrait","split":"reference"}\n',
                encoding="utf-8",
            )
            manifest = DatasetManifest.from_jsonl(path, CHARACTERS)

        self.assertEqual(manifest.root, path.parent)
        self.assertEqual(manifest.records[0].split, DatasetSplit.REFERENCE)

    def test_rejects_unknown_character(self) -> None:
        with self.assertRaisesRegex(DatasetSchemaError, "unknown character ID 'dipo'"):
            DatasetManifest.from_data(
                ".",
                CHARACTERS,
                [
                    {
                        "image": "images/dipo.png",
                        "character": "dipo",
                        "caption": "chr_dipo portrait",
                        "split": "validation",
                    }
                ],
            )

    def test_rejects_invalid_fields_and_duplicate_characters(self) -> None:
        invalid_records = (
            {"image": "/absolute.png", "character": "dino", "caption": "ok", "split": "train"},
            {"image": "image.png", "character": "dino", "caption": " ", "split": "train"},
            {"image": "image.png", "character": "dino", "caption": "ok", "split": "test"},
        )
        for record in invalid_records:
            with self.subTest(record=record), self.assertRaises(DatasetSchemaError):
                DatasetManifest.from_data(".", CHARACTERS, [record])

        with self.assertRaisesRegex(DatasetSchemaError, "character IDs must be unique"):
            DatasetManifest.from_data(".", [CHARACTERS[0], CHARACTERS[0]], [])

    def test_reports_invalid_jsonl_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            path.write_text('{}\n{"broken":\n', encoding="utf-8")
            with self.assertRaisesRegex(DatasetSchemaError, "line 2"):
                DatasetManifest.from_jsonl(path, CHARACTERS)


if __name__ == "__main__":
    unittest.main()
