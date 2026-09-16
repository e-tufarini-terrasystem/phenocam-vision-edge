"""Map reviewed boxes into a balanced subset of the exact runtime crops."""

from phenocam.inference.views import _crop_rectangles


def selected_rectangles(width, height, column):
    """Return one runtime crop per row, rotating the selected column."""
    rectangles = _crop_rectangles(width, height)
    if len(rectangles) != 15 or not 0 <= column < 5:
        raise ValueError("invalid runtime crop geometry")
    return tuple((priority, rectangles[priority]) for priority in (column, 5 + column, 10 + column))


def remap_boxes(boxes, rectangle, minimum_visible_fraction=0.5):
    """Clip boxes to a crop while excluding minor edge fragments."""
    crop_x, crop_y, crop_width, crop_height = rectangle
    if not 0 < minimum_visible_fraction <= 1:
        raise ValueError("invalid visible fraction")
    output = []
    for box in boxes:
        x1, y1, x2, y2 = box["box"]
        clipped_x1, clipped_y1 = max(x1, crop_x), max(y1, crop_y)
        clipped_x2 = min(x2, crop_x + crop_width)
        clipped_y2 = min(y2, crop_y + crop_height)
        clipped_width, clipped_height = clipped_x2 - clipped_x1, clipped_y2 - clipped_y1
        if clipped_width < 1 or clipped_height < 1:
            continue
        original_area = (x2 - x1) * (y2 - y1)
        visible_fraction = clipped_width * clipped_height / original_area
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        center_inside = (
            crop_x <= center_x < crop_x + crop_width
            and crop_y <= center_y < crop_y + crop_height
        )
        if not center_inside and visible_fraction < minimum_visible_fraction:
            continue
        output.append({
            "class_id": box["class_id"],
            "box": (
                clipped_x1 - crop_x,
                clipped_y1 - crop_y,
                clipped_x2 - crop_x,
                clipped_y2 - crop_y,
            ),
        })
    return output
