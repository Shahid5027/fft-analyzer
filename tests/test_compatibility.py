"""Tests for scientific compatibility analyzer and frequency overlap detection."""

import numpy as np
import pytest

from vga_emi.models import MeasurementTrace
from vga_emi.processing.compatibility import analyze_trace_compatibility


def test_compatible_traces():
    f = np.linspace(10, 1000, 100)
    t1 = MeasurementTrace("t1", "Trace 1", f, np.ones_like(f) * 10, measurement_unit="dBµA/m", review_status="verified")
    t2 = MeasurementTrace("t2", "Trace 2", f, np.ones_like(f) * 15, measurement_unit="dBµA/m", review_status="verified")

    report = analyze_trace_compatibility([t1, t2])
    assert report.is_unit_compatible is True
    assert report.has_frequency_overlap is True
    assert report.overlap_range == (10.0, 1000.0)
    assert report.can_combine_scientifically is True
    assert len(report.warnings) == 0


def test_provisional_calibration_warning():
    f = np.linspace(10, 1000, 100)
    t1 = MeasurementTrace("t1", "Trace 1", f, np.ones_like(f) * 10, measurement_unit="dBµA/m", review_status="provisional")
    report = analyze_trace_compatibility([t1])
    assert any("provisional calibration" in w for w in report.warnings)


def test_incompatible_units_detected():
    f = np.linspace(10, 1000, 100)
    t1 = MeasurementTrace("t1", "Trace 1", f, np.ones_like(f) * 10, measurement_unit="dBµA/m")
    t2 = MeasurementTrace("t2", "Trace 2", f, np.ones_like(f) * 0.5, measurement_unit="V")

    report = analyze_trace_compatibility([t1, t2])
    assert report.is_unit_compatible is False
    assert "Incompatible measurement units detected" in report.warnings[0]
    assert report.can_combine_scientifically is False


def test_zero_frequency_overlap():
    f1 = np.linspace(10, 500, 50)
    f2 = np.linspace(1000, 5000, 50)
    t1 = MeasurementTrace("t1", "Trace 1", f1, np.ones_like(f1) * 10, measurement_unit="dBµA/m")
    t2 = MeasurementTrace("t2", "Trace 2", f2, np.ones_like(f2) * 15, measurement_unit="dBµA/m")

    report = analyze_trace_compatibility([t1, t2])
    assert report.is_unit_compatible is True
    assert report.has_frequency_overlap is False
    assert report.overlap_range is None
    assert any("Zero frequency overlap" in w for w in report.warnings)
    assert report.can_combine_scientifically is False
