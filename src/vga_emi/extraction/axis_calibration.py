"""Axis calibration mapping pixel coordinates to engineering units."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from vga_emi.models import MeasurementTrace


def calibrate_axis_linear(
    pixel_coords: np.ndarray,
    pixel_min: float,
    pixel_max: float,
    val_min: float,
    val_max: float,
    invert: bool = False,
) -> np.ndarray:
    """Map pixel coordinates to physical values linearly.

    Args:
        pixel_coords: Array of pixel values.
        pixel_min: Pixel coordinate corresponding to val_min (or val_max if inverted).
        pixel_max: Pixel coordinate corresponding to val_max (or val_min if inverted).
        val_min: Minimum physical value.
        val_max: Maximum physical value.
        invert: True if higher pixel coordinates correspond to lower physical values (e.g. Y-axis).

    Returns:
        Array of mapped physical values.
    """
    if pixel_max == pixel_min:
        raise ValueError("pixel_max cannot equal pixel_min.")
    if val_max == val_min:
        raise ValueError("val_max cannot equal val_min.")

    norm = (np.asarray(pixel_coords, dtype=float) - pixel_min) / (pixel_max - pixel_min)
    if invert:
        norm = 1.0 - norm
    return val_min + norm * (val_max - val_min)


def calibrate_axis_log(
    pixel_coords: np.ndarray,
    pixel_min: float,
    pixel_max: float,
    val_min: float,
    val_max: float,
    invert: bool = False,
) -> np.ndarray:
    """Map pixel coordinates to physical values logarithmically (base 10).

    Args:
        pixel_coords: Array of pixel coordinates.
        pixel_min: Reference pixel coordinate for minimum value.
        pixel_max: Reference pixel coordinate for maximum value.
        val_min: Minimum physical value (> 0).
        val_max: Maximum physical value (> 0).
        invert: True if higher pixel coordinates correspond to lower physical values.

    Returns:
        Array of mapped physical values.
    """
    if pixel_max == pixel_min:
        raise ValueError("pixel_max cannot equal pixel_min.")
    if val_min <= 0 or val_max <= 0:
        raise ValueError(f"Logarithmic axis bounds must be strictly positive: got min={val_min}, max={val_max}")
    if val_max == val_min:
        raise ValueError("val_max cannot equal val_min.")

    log_min = np.log10(val_min)
    log_max = np.log10(val_max)

    norm = (np.asarray(pixel_coords, dtype=float) - pixel_min) / (pixel_max - pixel_min)
    if invert:
        norm = 1.0 - norm

    log_val = log_min + norm * (log_max - log_min)
    return 10.0**log_val


def create_trace_from_extracted_pixels(
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
    bounds: Tuple[int, int, int, int],
    f_min: float = 10.0,
    f_max: float = 400000.0,
    f_scale: str = "log",
    m_min: float = 0.0,
    m_max: float = 80.0,
    m_scale: str = "linear",
    frequency_unit: str = "Hz",
    measurement_unit: str = "dBµA/m (provisional)",
    trace_id: str = "extracted_screenshot_1",
    name: str = "Extracted Trace (Screenshot)",
    source_filename: Optional[str] = None,
    confidence: float = 0.5,
    is_calibrated: bool = False,
) -> MeasurementTrace:
    """Convert extracted pixel arrays into a calibrated MeasurementTrace.

    Args:
        x_pixels: X pixel coordinates.
        y_pixels: Y pixel coordinates.
        bounds: (x_min, y_min, x_max, y_max) plot bounds.
        f_min, f_max: Frequency axis range.
        f_scale: Frequency axis scale ('linear' or 'log').
        m_min, m_max: Measurement axis range.
        m_scale: Measurement axis scale ('linear' or 'log').
        frequency_unit: Unit string.
        measurement_unit: Unit string.
        trace_id: Unique trace id.
        name: Trace name.
        source_filename: Source file name.
        confidence: Extraction confidence in [0, 1].
        is_calibrated: Whether manual calibration was performed.

    Returns:
        Calibrated MeasurementTrace.
    """
    x_min, y_min, x_max, y_max = bounds

    # Calibrate X (frequency)
    if f_scale == "log":
        freqs = calibrate_axis_log(x_pixels, float(x_min), float(x_max), f_min, f_max, invert=False)
    else:
        freqs = calibrate_axis_linear(x_pixels, float(x_min), float(x_max), f_min, f_max, invert=False)

    # Calibrate Y (amplitude): higher pixel y in images is lower physical value -> invert=True
    if m_scale == "log":
        meas = calibrate_axis_log(y_pixels, float(y_min), float(y_max), m_min, m_max, invert=True)
    else:
        meas = calibrate_axis_linear(y_pixels, float(y_min), float(y_max), m_min, m_max, invert=True)

    # Ensure strictly increasing frequency ordering
    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]
    meas = meas[sort_idx]

    # Deduplicate matching frequencies
    unique_f, uniq_idx = np.unique(freqs, return_index=True)
    if len(unique_f) < len(freqs):
        freqs = unique_f
        meas = meas[uniq_idx]

    calibration_info: Dict[str, Any] = {
        "x_bounds_px": [x_min, x_max],
        "y_bounds_px": [y_min, y_max],
        "f_range": [f_min, f_max],
        "f_scale": f_scale,
        "m_range": [m_min, m_max],
        "m_scale": m_scale,
        "is_calibrated": is_calibrated,
    }

    status = "calibrated" if is_calibrated else "provisional"

    return MeasurementTrace(
        trace_id=trace_id,
        name=name,
        frequencies=freqs,
        measurements=meas,
        frequency_unit=frequency_unit,
        measurement_unit=measurement_unit,
        source_type="screenshot_derived",
        source_filename=source_filename,
        axis_scales={"x": f_scale, "y": m_scale},
        calibration_settings=calibration_info,
        extraction_confidence=confidence,
        processing_history=[
            f"Extracted from screenshot '{source_filename or 'image'}'",
            f"Plot bounds: {bounds}",
            f"Extraction confidence: {confidence:.2f}",
            f"Axis calibration: X={f_scale} [{f_min}, {f_max}], Y={m_scale} [{m_min}, {m_max}]",
        ],
        review_status=status,
        metadata={
            "source": "screenshot_extraction",
            "is_calibrated": is_calibrated,
        },
    )
