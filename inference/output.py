"""Render final selected detections and persist one verified output image.

Geometry and row validation are complete before this boundary. ``classes.py``
selection is final here, and output failures reveal no filesystem internals.
"""

from PIL import ImageDraw, ImageFont

from .errors import OutputWriteError


def write_output(image, detections, enabled_ids, model_names, output_path):
    try:
        selected = set(enabled_ids)
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

        image.save(output_path)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise OutputWriteError()
    except OutputWriteError:
        raise
    except Exception:
        raise OutputWriteError() from None
