"""Shared deterministic, atomic, and image-validation primitives."""

import csv
import hashlib
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class DatasetError(RuntimeError):
    """A dataset input or build stage failed validation."""


@contextmanager
def atomic_text(path, newline=None):
    """Write UTF-8 text beside ``path`` and atomically replace on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline=newline) as output:
            yield output
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def write_csv(path, fieldnames, rows):
    with atomic_text(path, newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed, *parts):
    material = "\x1f".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def normalized_pixels_sha256(image):
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}\0RGB\0".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _dct_matrix(size):
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos((math.pi / size) * (positions + 0.5) * frequencies)
    matrix[0] *= math.sqrt(1 / size)
    matrix[1:] *= math.sqrt(2 / size)
    return matrix


_DCT_32 = _dct_matrix(32)


def phash64(image):
    """Return a deterministic 64-bit DCT perceptual hash as 16 hex digits."""
    gray = ImageOps.exif_transpose(image).convert("L").resize(
        (32, 32), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(gray, dtype=np.float64)
    coefficients = _DCT_32 @ pixels @ _DCT_32.T
    low = coefficients[:8, :8]
    median = float(np.median(low))
    value = 0
    for bit in (low > median).reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming64(left, right):
    return (int(left, 16) ^ int(right, 16)).bit_count()


def inspect_image(path, eligibility, rotation_ccw=0):
    """Decode an untrusted image and return normalized properties and hashes."""
    path = Path(path)
    maximum_bytes = int(eligibility["maximum_source_bytes"])
    try:
        size_bytes = path.stat().st_size
    except OSError as error:
        raise DatasetError("image_unreadable") from error
    if size_bytes <= 0 or size_bytes > maximum_bytes:
        raise DatasetError("image_size_bytes_out_of_range")

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = int(eligibility["maximum_pixels"])
    try:
        with Image.open(path) as source:
            source.verify()
        with Image.open(path) as source:
            normalized = ImageOps.exif_transpose(source).convert("RGB")
            if rotation_ccw:
                normalized = normalized.rotate(int(rotation_ccw), expand=True)
            normalized.load()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise DatasetError("image_decode_failed") from error
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    width, height = normalized.size
    if width < int(eligibility["minimum_width"]) or height < int(
        eligibility["minimum_height"]
    ):
        raise DatasetError("image_dimensions_too_small")
    aspect = width / height
    if not (
        float(eligibility["minimum_aspect_ratio"])
        <= aspect
        <= float(eligibility["maximum_aspect_ratio"])
    ):
        raise DatasetError("image_aspect_ratio_out_of_range")
    return {
        "width": width,
        "height": height,
        "source_sha256": sha256_file(path),
        "decoded_sha256": normalized_pixels_sha256(normalized),
        "phash": phash64(normalized),
    }


def require_columns(reader, required, source_name):
    available = set(reader.fieldnames or ())
    missing = sorted(set(required) - available)
    if missing:
        raise DatasetError(f"{source_name}: missing CSV columns: {', '.join(missing)}")
