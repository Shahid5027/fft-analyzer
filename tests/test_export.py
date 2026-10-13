"""Tests for export utilities: PNG figures, numerical CSVs, and peak tables."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vga_emi.models import MeasurementTrace, PeakResult
from vga_emi.processing.export import (
    figure_to_png_bytes,
    trace_to_csv_bytes,
    peaks_to_csv_bytes,
    comparison_to_csv_bytes,
)


def test_figure_to_png_bytes():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([1, 2, 3], [4, 5, 6])
    png_bytes = figure_to_png_bytes(fig, dpi=100)
    plt.close(fig)

    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0
    # Standard PNG magic signature: \x89PNG\r\n\x1a\n
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_trace_to_csv_bytes():
    freqs = np.array([10.0, 50.0, 100.0])
    meas = np.array([25.4, 38.1, 42.0])
    trace = MeasurementTrace(
        trace_id="test_exp_1",
        name="Ch1 Export Test",
        frequencies=freqs,
        measurements=meas,
        frequency_unit="Hz",
        measurement_unit="dBµA/m",
        source_filename="ch1_raw.csv",
    )

    csv_bytes = trace_to_csv_bytes(trace)
    text = csv_bytes.decode("utf-8")

    assert "# VGA EMI Analyzer Export" in text
    assert "# Trace Name: Ch1 Export Test" in text
    assert "Frequency_Hz" in text
    assert "Measurement_dBµA_m" in text
    assert "25.4" in text


def test_peaks_to_csv_bytes():
    peaks = PeakResult(
        frequencies=np.array([31468.0, 62936.0]),
        amplitudes=np.array([48.2, 38.1]),
        indices=np.array([10, 20]),
    )
    csv_bytes = peaks_to_csv_bytes(peaks, freq_unit="Hz", meas_unit="dBµA/m")
    text = csv_bytes.decode("utf-8")

    assert "# Total Peaks: 2" in text
    assert "Rank" in text
    assert "Frequency (Hz)" in text
    assert "31468" in text


def test_comparison_to_csv_bytes():
    t1 = MeasurementTrace("t1", "T1", np.array([10.0, 100.0]), np.array([5.0, 10.0]))
    t2 = MeasurementTrace("t2", "T2", np.array([20.0, 200.0]), np.array([8.0, 12.0]))
    csv_bytes = comparison_to_csv_bytes([t1, t2])
    text = csv_bytes.decode("utf-8")

    assert "Trace ID" in text
    assert "T1" in text
    assert "T2" in text
