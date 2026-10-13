"""Tests for axis calibration mapping and coordinate conversions."""

import numpy as np
import pytest

from vga_emi.extraction.axis_calibration import (
    calibrate_axis_linear,
    calibrate_axis_log,
    create_trace_from_extracted_pixels,
)


def test_calibrate_linear_x():
    px = np.array([100.0, 200.0, 300.0])
    res = calibrate_axis_linear(px, pixel_min=100.0, pixel_max=300.0, val_min=0.0, val_max=1000.0)
    np.testing.assert_allclose(res, [0.0, 500.0, 1000.0])


def test_calibrate_linear_y_inverted():
    # In images, px=100 is top (max val), px=300 is bottom (min val)
    py = np.array([100.0, 200.0, 300.0])
    res = calibrate_axis_linear(py, pixel_min=100.0, pixel_max=300.0, val_min=0.0, val_max=100.0, invert=True)
    np.testing.assert_allclose(res, [100.0, 50.0, 0.0])


def test_calibrate_log():
    px = np.array([0.0, 50.0, 100.0])
    res = calibrate_axis_log(px, pixel_min=0.0, pixel_max=100.0, val_min=10.0, val_max=1000.0)
    # 0% -> 10^1 = 10; 50% -> 10^2 = 100; 100% -> 10^3 = 1000
    np.testing.assert_allclose(res, [10.0, 100.0, 1000.0])


def test_calibrate_log_invalid_bounds():
    with pytest.raises(ValueError, match="strictly positive"):
        calibrate_axis_log(np.array([10.0]), pixel_min=0.0, pixel_max=100.0, val_min=-5.0, val_max=100.0)


def test_create_trace_from_extracted_pixels():
    x_px = np.array([10, 50, 100])
    y_px = np.array([80, 50, 20])
    bounds = (10, 10, 100, 100)

    trace = create_trace_from_extracted_pixels(
        x_px, y_px, bounds,
        f_min=100.0, f_max=10000.0, f_scale="log",
        m_min=0.0, m_max=50.0, m_scale="linear",
        is_calibrated=True,
    )
    assert trace.review_status == "calibrated"
    assert len(trace.frequencies) == 3
    assert np.all(np.diff(trace.frequencies) > 0)
