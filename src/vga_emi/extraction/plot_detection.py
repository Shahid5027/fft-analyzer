"""Plot area detection in oscilloscope screenshots using OpenCV."""

from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


def detect_plot_bounds(
    image_rgb: np.ndarray,
    min_area_fraction: float = 0.25,
) -> Tuple[int, int, int, int]:
    """Detect probable oscilloscope plot bounds (x_min, y_min, x_max, y_max).

    Finds the largest rectangular bounding box corresponding to the
    waveform/grid display area, falling back to a 10% margin if undetected.

    Args:
        image_rgb: Input RGB image array (H, W, 3).
        min_area_fraction: Minimum fraction of total image area to consider as plot area.

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) coordinates.
    """
    h, w = image_rgb.shape[:2]
    total_area = h * w

    # Convert to grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Blur slightly and find edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find external contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    max_area = 0.0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area > min_area_fraction * total_area and area > max_area:
            # Check aspect ratio is reasonable for an oscilloscope display (e.g. 1.0 to 3.0)
            aspect = cw / float(ch)
            if 0.8 <= aspect <= 3.5:
                max_area = area
                best_box = (x, y, x + cw, y + ch)

    if best_box is not None:
        return best_box

    # Fallback: Central 80% region
    pad_x = int(w * 0.08)
    pad_y = int(h * 0.08)
    return (pad_x, pad_y, w - pad_x, h - pad_y)
