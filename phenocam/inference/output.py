"""Own selected-class rendering and verified final-image persistence.

Geometry and detection validation are complete before this boundary. This file
creates independent annotated/privacy products and writes them deterministically.
The pipeline decides whether detections warrant writing any products.
``write_outputs`` is the single boundary mapping failures to the fixed error.
"""

from math import ceil, floor
import os
from pathlib import Path
import stat
import tempfile

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .errors import OutputWriteError

_PRIVACY_MARGIN_RATIO = 0.10
_PRIVACY_BLUR_RADIUS_RATIO = 0.10
_PRIVACY_MIN_BLUR_RADIUS = 8


def write_outputs(
    source, detections, enabled_ids, model_names, annotated_path, privacy_path
):
    try:
        selected = set(enabled_ids)
        if annotated_path is not None:
            annotated = source.copy()
            _render_annotated(annotated, detections, selected, model_names)
            _save_output(annotated, annotated_path)
        if privacy_path is not None:
            # Each product starts from the unmodified normalized source.
            privacy = source.copy()
            _render_privacy(privacy, detections, selected)
            _save_output(privacy, privacy_path)
    except OutputWriteError:
        raise
    except Exception:
        raise OutputWriteError() from None


def _render_annotated(image, detections, selected, model_names):
    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, round(min(width, height) / 120)))
    line_width = max(2, round(min(width, height) / 500))

    for detection in detections:
        if detection.class_id not in selected:
            continue
        colour = (255, 70, 40) if detection.class_id == 0 else (30, 180, 255)
        box = tuple(
            round(value)
            for value in (detection.x1, detection.y1, detection.x2, detection.y2)
        )
        draw.rectangle(box, outline=colour, width=line_width)
        label = f"{model_names[detection.class_id]} {detection.confidence:.2f}"
        text_box = draw.textbbox(
            (box[0], box[1]), label, font=font, stroke_width=1
        )
        text_height = text_box[3] - text_box[1] + 4
        label_y = max(0, box[1] - text_height)
        background = (
            box[0],
            label_y,
            box[0] + text_box[2] - text_box[0] + 4,
            box[1],
        )
        draw.rectangle(background, fill=colour)
        draw.text(
            (box[0] + 2, label_y + 1),
            label,
            fill=(0, 0, 0),
            font=font,
            stroke_width=1,
            stroke_fill=colour,
        )


def _render_privacy(image, detections, selected):
    for detection in detections:
        if detection.class_id not in selected:
            continue
        width = detection.x2 - detection.x1
        height = detection.y2 - detection.y1
        left = max(0, floor(detection.x1 - width * _PRIVACY_MARGIN_RATIO))
        top = max(0, floor(detection.y1 - height * _PRIVACY_MARGIN_RATIO))
        right = min(
            image.width, ceil(detection.x2 + width * _PRIVACY_MARGIN_RATIO)
        )
        bottom = min(
            image.height, ceil(detection.y2 + height * _PRIVACY_MARGIN_RATIO)
        )
        box = (left, top, right, bottom)
        radius = max(
            float(_PRIVACY_MIN_BLUR_RADIUS),
            min(right - left, bottom - top) * _PRIVACY_BLUR_RADIUS_RATIO,
        )
        # Crop from the current image so overlaps are blurred in detection order.
        region = image.crop(box).filter(ImageFilter.GaussianBlur(radius))
        image.paste(region, box)


def _save_output(image, output_path):
    temporary_path = None
    try:
        output_path = Path(output_path)
        # Preserve the requested format and ordinary output symlink behavior.
        # CLI validation has already rejected symbolic aliases of the input.
        image_format = Image.registered_extensions()[output_path.suffix.lower()]
        destination = output_path.resolve()
        mode = stat.S_IMODE(destination.stat().st_mode) if destination.exists() else None
        with tempfile.NamedTemporaryFile(
            prefix=".phenocam-", suffix=output_path.suffix,
            dir=destination.parent, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            image.save(temporary, format=image_format)
            temporary.flush()
            if mode is not None:
                os.fchmod(temporary.fileno(), mode)
            os.fsync(temporary.fileno())
        # Verify before replacing: a failed encode must leave the original intact.
        with Image.open(temporary_path) as saved:
            saved.verify()
        # JPEG verify() does not decode pixels; reopen to reject truncated data.
        with Image.open(temporary_path) as saved:
            saved.load()
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass
