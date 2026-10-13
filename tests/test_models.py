"""Tests for MeasurementTrace and TimeDomainTrace models and validation."""

import numpy as np
import pytest

from vga_emi.models import MeasurementTrace, TimeDomainTrace, PeakResult


def test_measurement_trace_valid():
    freqs = np.array([10.0, 100.0, 1000.0, 10000.0])
    meas = np.array([20.0, 35.0, 42.0, 15.0])
    trace = MeasurementTrace(
        trace_id="test_1",
        name="Valid Trace",
        frequencies=freqs,
        measurements=meas,
    )
    assert trace.trace_id == "test_1"
    assert len(trace.frequencies) == 4
    assert trace.to_dict()["num_points"] == 4


def test_measurement_trace_empty_error():
    with pytest.raises(ValueError, match="frequencies array cannot be empty"):
        MeasurementTrace(
            trace_id="empty_1",
            name="Empty Trace",
            frequencies=np.array([]),
            measurements=np.array([]),
        )


def test_measurement_trace_length_mismatch():
    with pytest.raises(ValueError, match="Array length mismatch"):
        MeasurementTrace(
            trace_id="mismatch_1",
            name="Mismatch Trace",
            frequencies=np.array([10.0, 20.0]),
            measurements=np.array([5.0]),
        )


def test_measurement_trace_nan_error():
    with pytest.raises(ValueError, match="NaN or infinite"):
        MeasurementTrace(
            trace_id="nan_1",
            name="NaN Trace",
            frequencies=np.array([10.0, 20.0]),
            measurements=np.array([5.0, np.nan]),
        )


def test_measurement_trace_non_monotonic_frequency():
    with pytest.raises(ValueError, match="strictly increasing"):
        MeasurementTrace(
            trace_id="mono_1",
            name="Non-monotonic Trace",
            frequencies=np.array([100.0, 50.0, 200.0]),
            measurements=np.array([10.0, 20.0, 30.0]),
        )


def test_measurement_trace_log_scale_negative_freq():
    with pytest.raises(ValueError, match="contains non-positive frequencies"):
        MeasurementTrace(
            trace_id="log_neg_1",
            name="Log Neg Trace",
            frequencies=np.array([-10.0, 20.0, 30.0]),
            measurements=np.array([10.0, 20.0, 30.0]),
            axis_scales={"x": "log", "y": "linear"},
        )


def test_time_domain_trace_valid():
    t = np.linspace(0, 100, 50)
    v = np.sin(t)
    trace = TimeDomainTrace(
        channel_id="ch1",
        name="Channel 1",
        time=t,
        voltage=v,
    )
    assert trace.channel_id == "ch1"
    assert len(trace.time) == 50


def test_peak_result_to_records():
    res = PeakResult(
        frequencies=np.array([1000.0, 250000.0]),
        amplitudes=np.array([45.2, 38.1]),
        indices=np.array([10, 55]),
    )
    records = res.to_records()
    assert len(records) == 2
    assert records[0]["Rank"] == 1
    assert "1.00 kHz" in records[0]["Frequency Display"]
