"""Own deterministic detection metadata replacement and atomic persistence.

Trusted final detections and validated paths become one fixed ``[detection]``
section. The existing UTF-8 document remains untrusted: unrelated bytes are
preserved, and replacement commits atomically within the target directory.
"""

import os
import re
import stat
import tempfile
from pathlib import Path

from phenocam import __version__

SOFTWARE_NAME = "phenocam-detection"
MODEL_ID = "yolo26n"
MODEL_VERSION = "0.1.0"

_LINE_ENDING = re.compile(r"\r\n|\r|\n")
_SECTION_HEADER = re.compile(r"\[([^\[\]]+)\]")


class MetadataWriteError(RuntimeError):
    """Report a metadata failure without exposing private details."""

    def __init__(self, *_ignored):
        super().__init__()


def _lines(text):
    parts = re.split(r"(\r\n|\r|\n)", text)
    return [
        parts[index] + (parts[index + 1] if index + 1 < len(parts) else "")
        for index in range(0, len(parts), 2)
        if parts[index] or index + 1 < len(parts)
    ]


def _section_name(line):
    content = _LINE_ENDING.sub("", line, count=1).strip(" \t")
    match = _SECTION_HEADER.fullmatch(content)
    return match.group(1) if match else None


def _without_detection_sections(text):
    retained = []
    removing = False
    for line in _lines(text):
        name = _section_name(line)
        if name is not None:
            removing = name == "detection"
        if not removing:
            retained.append(line)
    return "".join(retained)


def _detection_section(
    detections,
    enabled_names,
    model_names,
    annotated_output_path,
    privacy_output_path,
    line_ending,
):
    counts = {name: 0 for name in enabled_names}
    for detection in detections:
        name = model_names[detection.class_id]
        if name in counts:
            counts[name] += 1

    detected_names = tuple(name for name in enabled_names if counts[name])
    total_count = sum(counts[name] for name in detected_names)
    fields = [
        "[detection]",
        f"detected={'true' if total_count else 'false'}",
        f"software_name={SOFTWARE_NAME}",
        f"software_version={__version__}",
        f"model_id={MODEL_ID}",
        f"model_version={MODEL_VERSION}",
        f"annotated_image={annotated_output_path or ''}",
        f"privacy_image={privacy_output_path or ''}",
        f"classes={','.join(detected_names)}",
    ]
    fields.extend(
        f"{name.replace(' ', '_')}_count={counts[name]}"
        for name in detected_names
    )
    fields.append(f"total_count={total_count}")
    return line_ending.join(fields) + line_ending


def _separator(retained, line_ending):
    if not retained:
        return ""
    match = re.search(r"(?:\r\n|\r|\n)+$", retained)
    ending_count = len(_LINE_ENDING.findall(match.group())) if match else 0
    return line_ending * max(0, 2 - ending_count)


def update_detection_metadata(
    metadata_path,
    detections,
    enabled_names,
    model_names,
    annotated_output_path,
    privacy_output_path,
):
    """Replace all exact detection sections with one current summary."""
    temporary_path = None
    descriptor = None
    committed = False
    try:
        metadata_path = Path(metadata_path)
        original = metadata_path.read_bytes()
        text = original.decode("utf-8", errors="strict")
        mode = stat.S_IMODE(metadata_path.stat().st_mode)
        line_match = _LINE_ENDING.search(text)
        line_ending = line_match.group() if line_match else "\n"
        retained = _without_detection_sections(text)
        updated = (
            retained
            + _separator(retained, line_ending)
            + _detection_section(
                detections,
                enabled_names,
                model_names,
                annotated_output_path,
                privacy_output_path,
                line_ending,
            )
        ).encode("utf-8")

        descriptor, name = tempfile.mkstemp(
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            dir=metadata_path.parent,
        )
        temporary_path = Path(name)
        with os.fdopen(descriptor, "wb") as temporary:
            descriptor = None
            os.fchmod(temporary.fileno(), mode)
            temporary.write(updated)
            temporary.flush()
            os.fsync(temporary.fileno())
        # No fallible operation follows the atomic commit point.
        os.replace(temporary_path, metadata_path)
        committed = True
    except Exception:
        raise MetadataWriteError() from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path is not None and not committed:
            try:
                temporary_path.unlink()
            except OSError:
                pass
