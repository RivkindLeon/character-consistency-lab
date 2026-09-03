import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from character_consistency_lab.cli import main
from character_consistency_lab.data import load_dataset, validate_dataset


def write_image(path: Path, color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=color).save(path)


def write_dataset(root: Path, records: list[str]) -> None:
    (root / "characters.yaml").write_text(
        "characters:\n  - id: dino\n    trigger: chr_dino\n", encoding="utf-8"
    )
    (root / "manifest.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")


class DatasetValidationTests(unittest.TestCase):
    def test_valid_dataset_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_image(root / "images/train.png")
            write_image(root / "images/reference.png", "blue")
            write_dataset(root, [
                '{"image":"images/train.png","character":"dino","caption":"chr_dino","split":"train"}',
                '{"image":"images/reference.png","character":"dino","caption":"chr_dino portrait","split":"reference"}',
            ])
            self.assertEqual(validate_dataset(load_dataset(root)), ())
            output = io.StringIO()
            with redirect_stdout(output):
                result = main_with_args(["dataset", "validate", str(root)])
            self.assertEqual(result, 0)
            self.assertIn("Dataset is valid", output.getvalue())

    def test_reports_missing_duplicate_invalid_and_leaked_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_image(root / "images/train.png")
            (root / "images/broken.png").parent.mkdir(parents=True, exist_ok=True)
            (root / "images/broken.png").write_text("not an image", encoding="utf-8")
            (root / "images/copy.png").write_bytes((root / "images/train.png").read_bytes())
            write_dataset(root, [
                '{"image":"images/train.png","character":"dino","caption":"a","split":"train"}',
                '{"image":"images/train.png","character":"dino","caption":"b","split":"validation"}',
                '{"image":"images/missing.png","character":"dino","caption":"c","split":"train"}',
                '{"image":"images/broken.png","character":"dino","caption":"d","split":"reference"}',
                '{"image":"images/copy.png","character":"dino","caption":"e","split":"reference"}',
                '{"image":"../outside.png","character":"dino","caption":"f","split":"train"}',
            ])
            codes = {issue.code for issue in validate_dataset(load_dataset(root))}
            self.assertEqual(codes, {
                "duplicate_filename", "missing_image", "invalid_image", "invalid_path",
                "train_reference_leakage"
            })


def main_with_args(arguments: list[str]) -> int:
    import sys
    from unittest.mock import patch

    with patch.object(sys, "argv", ["character-lab", *arguments]):
        return main()


if __name__ == "__main__":
    unittest.main()
