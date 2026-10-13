"""Tests for combination calculations and dB conversions."""

import numpy as np
import pytest

from vga_emi.models import MeasurementTrace
from vga_emi.processing.combination import (
    db_to_linear,
    linear_to_db,
    combine_traces_rss,
    compute_demo_sqrt,
)


def test_db_linear_roundtrip():
    original_db = np.array([0.0, 20.0, 40.0, 60.0])
    lin = db_to_linear(original_db, ref=1.0, is_power=False)
    # 20 dB amplitude is factor of 10, 40 dB is 100, 60 dB is 1000
    np.testing.assert_allclose(lin, [1.0, 10.0, 100.0, 1000.0])

    recon_db = linear_to_db(lin, ref=1.0, is_power=False)
    np.testing.assert_allclose(recon_db, original_db)


def test_combine_traces_linear_rss():
    # Test controlled linear inputs: trace1 = 3.0, trace2 = 4.0 -> RSS = sqrt(3^2 + 4^2) = 5.0
    freqs = np.array([10.0, 100.0, 1000.0])
    t1 = MeasurementTrace(
        trace_id="t1",
        name="T1",
        frequencies=freqs,
        measurements=np.array([3.0, 3.0, 3.0]),
        measurement_unit="V",
    )
    t2 = MeasurementTrace(
        trace_id="t2",
        name="T2",
        frequencies=freqs,
        measurements=np.array([4.0, 4.0, 4.0]),
        measurement_unit="V",
    )

    combined = combine_traces_rss([t1, t2], treat_as_db=False)
    np.testing.assert_allclose(combined.measurements, [5.0, 5.0, 5.0])
    assert combined.metadata["is_demo_calculation"] is True
    assert "DEMO CALCULATION" in combined.processing_history[0]


def test_combine_incompatible_units_error():
    freqs = np.array([10.0, 100.0])
    t1 = MeasurementTrace("t1", "T1", freqs, np.array([1.0, 2.0]), measurement_unit="dBµA/m")
    t2 = MeasurementTrace("t2", "T2", freqs, np.array([1.0, 2.0]), measurement_unit="V")

    with pytest.raises(ValueError, match="Incompatible measurement units"):
        combine_traces_rss([t1, t2])


def test_compute_demo_sqrt():
    freqs = np.array([10.0, 100.0])
    # For dB inputs, linear sqrt corresponds to halving dB: sqrt(10^(40/20)) = 10^(20/20) -> 20 dB
    t_db = MeasurementTrace("t_db", "T_dB", freqs, np.array([40.0, 60.0]), measurement_unit="dBµA/m")
    sqrt_res = compute_demo_sqrt(t_db)
    np.testing.assert_allclose(sqrt_res.measurements, [20.0, 30.0])

    # For linear inputs
    t_lin = MeasurementTrace("t_lin", "T_lin", freqs, np.array([9.0, 16.0]), measurement_unit="V")
    sqrt_lin = compute_demo_sqrt(t_lin)
    np.testing.assert_allclose(sqrt_lin.measurements, [3.0, 4.0])
