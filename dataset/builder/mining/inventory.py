"""Inventory an explicit external image root without crossing write boundaries."""

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path

from ..common import DatasetError, inspect_image, write_csv


INVENTORY_FIELDS = (
    "source_dataset", "source_subset", "source_id", "site_id", "timestamp",
    "calendar_date", "group_id", "relative_path", "local_path",
    "source_rotation_ccw", "width", "height", "source_bytes", "source_sha256",
    "decoded_sha256", "phash",
)
REJECTION_FIELDS = ("entry_sha256", "reason_code")
_ALLOWED_SUFFIXES = frozenset((".jpg", ".jpeg", ".png"))
_STAMP = re.compile(
    r"^(?P<site>[A-Za-z0-9.-]+?)[_-](?P<date>[0-9]{4}[-_][0-9]{2}[-_][0-9]{2})_(?P<time>[0-9]{6})$"
)


def _safe_root(root):
    supplied = Path(root)
    if supplied.is_symlink():
        raise DatasetError("inventory root must not be a symlink")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise DatasetError("inventory root does not exist") from error
    if not resolved.is_dir() or resolved in (Path("/"), Path.home().resolve()):
        raise DatasetError("inventory root is unsafe")
    return resolved


def _entries(root, rejections, counters):
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError:
            identity = hashlib.sha256(str(directory.relative_to(root)).encode()).hexdigest()
            rejections.append({"entry_sha256": identity, "reason_code": "directory_unreadable"})
            continue
        for entry in reversed(entries):
            if entry.name.startswith("."):
                counters["ignored_hidden"] += 1
                continue
            if entry.is_symlink():
                counters["ignored_symlink"] += 1
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    path = Path(entry.path)
                    if path.suffix.lower() in _ALLOWED_SUFFIXES:
                        yield path
                    else:
                        counters["ignored_format"] += 1
            except OSError:
                identity = hashlib.sha256(entry.name.encode()).hexdigest()
                rejections.append({"entry_sha256": identity, "reason_code": "entry_unreadable"})


def _identity(path):
    match = _STAMP.fullmatch(path.stem)
    if match is None:
        raise DatasetError("timestamp_not_recognized")
    date = match.group("date").replace("_", "-")
    stamp = datetime.strptime(f"{date}_{match.group('time')}", "%Y-%m-%d_%H%M%S")
    return match.group("site"), stamp


def inventory(root, output_path, rejection_path, config):
    root = _safe_root(root)
    rows = []
    rejections = []
    counters = {"ignored_hidden": 0, "ignored_symlink": 0, "ignored_format": 0}
    for path in _entries(root, rejections, counters):
        relative = path.relative_to(root)
        try:
            site, stamp = _identity(path)
            properties = inspect_image(path, config["image_eligibility"])
            rows.append(
                {
                    "source_dataset": "internal",
                    "source_subset": site,
                    "source_id": path.stem,
                    "site_id": site,
                    "timestamp": stamp.isoformat(),
                    "calendar_date": stamp.date().isoformat(),
                    "group_id": f"{site}:{stamp.date().isoformat()}",
                    "relative_path": relative.as_posix(),
                    "local_path": str(path),
                    "source_rotation_ccw": "0",
                    "width": properties["width"],
                    "height": properties["height"],
                    "source_bytes": path.stat().st_size,
                    "source_sha256": properties["source_sha256"],
                    "decoded_sha256": properties["decoded_sha256"],
                    "phash": properties["phash"],
                }
            )
        except (DatasetError, OSError, ValueError) as error:
            reason = str(error) if isinstance(error, DatasetError) else "image_unreadable"
            rejections.append(
                {
                    "entry_sha256": hashlib.sha256(relative.as_posix().encode()).hexdigest(),
                    "reason_code": reason,
                }
            )
    rows.sort(key=lambda row: (row["site_id"], row["timestamp"], row["source_id"]))
    if len({row["source_id"] for row in rows}) != len(rows):
        raise DatasetError("inventory source identities are not unique")
    write_csv(output_path, INVENTORY_FIELDS, rows)
    write_csv(rejection_path, REJECTION_FIELDS, rejections)
    return {"accepted": len(rows), "rejected": len(rejections), **counters}
