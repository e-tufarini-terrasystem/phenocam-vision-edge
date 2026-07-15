"""Own the complete adaptive-gamma policy for model input images.

This module validates version-controlled constants, classifies whole-image
luminance, and applies one RGB transform. The inference transaction retains
ownership of the unmodified source image used for final rendering.
"""

import math as _math

from .errors import GammaConfigurationError as _GammaConfigurationError

ADAPTIVE_GAMMA_ENABLED = False
DARK_THRESHOLD = 0.15
DIM_THRESHOLD = 0.35
DARK_GAMMA = 0.60
DIM_GAMMA = 0.80
NORMAL_GAMMA = 1.00


def apply_adaptive_gamma(image):
    """Return the source image or a gamma-adjusted model image."""
    if type(ADAPTIVE_GAMMA_ENABLED) is not bool:
        raise _GammaConfigurationError()

    values = (
        DARK_THRESHOLD,
        DIM_THRESHOLD,
        DARK_GAMMA,
        DIM_GAMMA,
        NORMAL_GAMMA,
    )
    if any(type(value) not in (int, float) for value in values):
        raise _GammaConfigurationError()
    try:
        dark_threshold, dim_threshold, dark_gamma, dim_gamma, normal_gamma = (
            float(value) for value in values
        )
    except (OverflowError, ValueError):
        raise _GammaConfigurationError() from None

    if not all(_math.isfinite(value) for value in values):
        raise _GammaConfigurationError()
    if not 0 < dark_threshold < dim_threshold < 1:
        raise _GammaConfigurationError()
    if not 0 < dark_gamma <= dim_gamma <= normal_gamma == 1.0:
        raise _GammaConfigurationError()

    # Disabled configuration is still validated, but must perform no image work.
    if not ADAPTIVE_GAMMA_ENABLED:
        return image

    histogram = image.convert("L").histogram()
    median_target = (image.width * image.height + 1) // 2
    cumulative = 0
    for level, count in enumerate(histogram):
        cumulative += count
        if cumulative >= median_target:
            median = level / 255
            break

    if median >= dim_threshold:
        selected_gamma = normal_gamma
    elif median >= dark_threshold:
        selected_gamma = dim_gamma
    else:
        selected_gamma = dark_gamma

    # Identity returns preserve source ownership and avoid building an unused LUT.
    if selected_gamma == 1.0:
        return image

    table = [
        max(0, min(255, round(255 * (value / 255) ** selected_gamma)))
        for value in range(256)
    ]
    return image.point(table * 3)
