"""Create a compact, deterministic overview of benchmark generations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass(frozen=True)
class ContactSheetItem:
    image_path: Path
    label: str


def create_contact_sheet(
    items: Sequence[ContactSheetItem],
    output_path: str | Path,
    *,
    columns: int = 4,
    thumbnail_size: tuple[int, int] = (256, 256),
    padding: int = 16,
) -> Path:
    """Arrange labeled images in input order and save one RGB PNG."""

    if not items:
        raise ValueError("contact sheet requires at least one image")
    if columns <= 0:
        raise ValueError("columns must be positive")
    if thumbnail_size[0] <= 0 or thumbnail_size[1] <= 0:
        raise ValueError("thumbnail dimensions must be positive")
    if padding < 0:
        raise ValueError("padding must be non-negative")

    font = ImageFont.load_default()
    caption_height = 24
    cell_width = thumbnail_size[0]
    cell_height = thumbnail_size[1] + caption_height
    used_columns = min(columns, len(items))
    rows = math.ceil(len(items) / columns)
    sheet = Image.new(
        "RGB",
        (
            used_columns * cell_width + (used_columns + 1) * padding,
            rows * cell_height + (rows + 1) * padding,
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for index, item in enumerate(items):
        if not item.label.strip():
            raise ValueError("contact sheet labels must not be empty")
        column = index % columns
        row = index // columns
        x = padding + column * (cell_width + padding)
        y = padding + row * (cell_height + padding)
        try:
            with Image.open(item.image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
        except OSError as exc:
            raise ValueError(f"cannot read contact sheet image {item.image_path}: {exc}") from exc
        image_x = x + (cell_width - image.width) // 2
        image_y = y + (thumbnail_size[1] - image.height) // 2
        sheet.paste(image, (image_x, image_y))
        draw.text((x, y + thumbnail_size[1] + 6), item.label, fill="black", font=font)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG")
    return destination
