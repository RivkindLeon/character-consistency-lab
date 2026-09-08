import tempfile
import unittest
from pathlib import Path

from PIL import Image

from character_consistency_lab.reports import ContactSheetItem, create_contact_sheet


class ContactSheetTests(unittest.TestCase):
    def test_creates_labeled_grid_from_small_local_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = []
            for index, size in enumerate(((40, 20), (20, 40), (30, 30))):
                image_path = root / f"scene-{index}.png"
                Image.new("RGB", size, (index * 50, 20, 30)).save(image_path)
                items.append(ContactSheetItem(image_path, f"scene-{index}"))

            output = create_contact_sheet(
                items,
                root / "report" / "grid.png",
                columns=2,
                thumbnail_size=(64, 64),
                padding=8,
            )

            self.assertTrue(output.is_file())
            with Image.open(output) as sheet:
                self.assertEqual(sheet.mode, "RGB")
                self.assertEqual(sheet.size, (152, 200))

    def test_rejects_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least one image"):
                create_contact_sheet([], Path(directory) / "grid.png")

    def test_reports_unreadable_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.png"
            with self.assertRaisesRegex(ValueError, "cannot read contact sheet image"):
                create_contact_sheet(
                    [ContactSheetItem(missing, "missing-scene")],
                    Path(directory) / "grid.png",
                )


if __name__ == "__main__":
    unittest.main()
