"""Tests for CSV loading and validation."""

import io
import pytest

from vga_emi.ingestion.csv_loader import load_csv_trace, inspect_csv


def test_load_valid_csv():
    csv_text = """freq,amp
10.0,22.5
50.0,30.1
100.0,35.0
500.0,28.2
"""
    stream = io.StringIO(csv_text)
    trace, report = load_csv_trace(stream, freq_col="freq", meas_col="amp", trace_name="Test CSV")

    assert report["valid_rows"] == 4
    assert report["dropped_rows"] == 0
    assert len(trace.frequencies) == 4
    assert trace.frequencies[0] == 10.0
    assert trace.frequencies[-1] == 500.0


def test_load_csv_with_invalid_rows():
    csv_text = """Frequency,Measurement
10.0,20.0
INVALID,30.0
50.0,NaN
-5.0,40.0
100.0,25.0
"""
    stream = io.StringIO(csv_text)
    trace, report = load_csv_trace(
        stream,
        freq_col="Frequency",
        meas_col="Measurement",
        axis_scales={"x": "log", "y": "linear"},
    )
    # 10.0 and 100.0 are valid; INVALID, NaN, and -5.0 (for log x) are dropped
    assert report["valid_rows"] == 2
    assert report["dropped_rows"] == 3
    assert len(trace.frequencies) == 2


def test_load_csv_missing_column():
    csv_text = """colA,colB
1,2
"""
    stream = io.StringIO(csv_text)
    with pytest.raises(ValueError, match="Frequency column 'freq' not found"):
        load_csv_trace(stream, freq_col="freq", meas_col="colB")
