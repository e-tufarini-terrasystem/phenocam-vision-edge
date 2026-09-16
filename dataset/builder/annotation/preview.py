"""Annotation: preview responsibility extracted without changing the data contract."""

from pathlib import Path
from PIL import Image, ImageOps
from ..common import DatasetError


def _materialize_preview(row, destination):
    destination = Path(destination)
    if destination.is_file():
        with Image.open(destination) as existing:
            if existing.size == (int(row["width"]), int(row["height"])):
                return "cached"
        raise DatasetError(f"stale annotation preview: {destination}")
    with Image.open(row["local_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        rotation = int(row.get("source_rotation_ccw", "") or 0)
        if rotation:
            image = image.rotate(rotation, expand=True)
        image.load()
    expected = (int(row["width"]), int(row["height"]))
    if image.size != expected:
        raise DatasetError(f"annotation preview dimensions disagree for {row['source_id']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        image.save(temporary, format="JPEG", quality=95, subsampling=0, optimize=True)
        temporary.replace(destination)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return "created"
