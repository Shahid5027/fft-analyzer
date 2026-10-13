"""Trace extraction from oscilloscope screenshots via HSV color segmentation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


# Predefined HSV ranges for standard oscilloscope channel trace colors
TRACE_COLOR_RANGES: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
    "yellow": [
        (np.array([20, 100, 100]), np.array([35, 255, 255])),
    ],
    "cyan": [
        (np.array([80, 100, 100]), np.array([100, 255, 255])),
    ],
    "magenta": [
        (np.array([140, 80, 100]), np.array([170, 255, 255])),
    ],
    "green": [
        (np.array([35, 80, 100]), np.array([80, 255, 255])),
    ],
    "blue": [
        (np.array([100, 100, 100]), np.array([135, 255, 255])),
    ],
    "bright_any": [
        (np.array([0, 0, 180]), np.array([180, 255, 255])),
    ],
}


def extract_trace_pixels(
    image_rgb: np.ndarray,
    bounds: Tuple[int, int, int, int],
    target_color: str = "yellow",
    custom_hsv_lower: Optional[np.ndarray] = None,
    custom_hsv_upper: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Segment a colored trace within plot bounds and extract (x, y) pixel coordinates.

    Args:
        image_rgb: Full RGB image.
        bounds: (x_min, y_min, x_max, y_max) plot boundaries.
        target_color: Color preset ('yellow', 'cyan', 'magenta', 'green', 'blue', 'bright_any').
        custom_hsv_lower: Optional custom lower HSV bound [H, S, V].
        custom_hsv_upper: Optional custom upper HSV bound [H, S, V].

    Returns:
        Tuple of:
          - x_pixel_coords (1D int array)
          - y_pixel_coords (1D float array)
          - confidence_score (float in [0.0, 1.0])
          - binary_mask (2D uint8 mask inside ROI)
    """
    x_min, y_min, x_max, y_max = bounds
    roi_rgb = image_rgb[y_min:y_max, x_min:x_max]

    if roi_rgb.size == 0:
        return np.array([]), np.array([]), 0.0, np.zeros((1, 1), dtype=np.uint8)

    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

    # Build mask
    if custom_hsv_lower is not None and custom_hsv_upper is not None:
        mask = cv2.inRange(roi_hsv, custom_hsv_lower, custom_hsv_upper)
    else:
        color_ranges = TRACE_COLOR_RANGES.get(target_color.lower(), TRACE_COLOR_RANGES["yellow"])
        mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            sub_mask = cv2.inRange(roi_hsv, lower, upper)
            mask = cv2.bitwise_or(mask, sub_mask)

    # Clean isolated noise pixels with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    roi_h, roi_w = mask_clean.shape
    x_list = []
    y_list = []

    # For each vertical column, find candidate trace pixels
    for col_x in range(roi_w):
        y_indices = np.where(mask_clean[:, col_x] > 0)[0]
        if len(y_indices) > 0:
            # Use median y to be robust against vertical spikes / thickness
            med_y = float(np.median(y_indices))
            x_list.append(x_min + col_x)
            y_list.append(y_min + med_y)

    if not x_list:
        return np.array([]), np.array([]), 0.0, mask_clean

    x_pixels = np.array(x_list, dtype=int)
    y_pixels = np.array(y_list, dtype=float)

    # Confidence is the fraction of ROI columns successfully tracked
    confidence = float(len(x_pixels)) / float(roi_w) if roi_w > 0 else 0.0

    return x_pixels, y_pixels, confidence, mask_clean


def create_preview_overlay(
    image_rgb: np.ndarray,
    bounds: Tuple[int, int, int, int],
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
) -> np.ndarray:
    """Generate an annotated preview overlay of the detected bounds and extracted trace."""
    overlay = np.copy(image_rgb)
    x_min, y_min, x_max, y_max = bounds

    # Draw detected plot bounding rectangle in bright green
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Draw extracted trace points in bright red
    for x, y in zip(x_pixels, y_pixels.astype(int)):
        if 0 <= y < overlay.shape[0] and 0 <= x < overlay.shape[1]:
            cv2.circle(overlay, (int(x), int(y)), 2, (255, 0, 0), -1)

    return overlay
