"""Tests for sample lab CSV fixtures and end-to-end ingestion."""

from pathlib import Path
import pytest

from vga_emi.ingestion.csv_loader import guess_csv_columns, inspect_csv, load_csv_trace
from vga_emi.processing.compatibility import analyze_trace_compatibility


SAMPLE_DIR = Path("sample_data")


def test_load_sample_ch1_fixture():
    fpath = SAMPLE_DIR / "vga_ch1_red_spectrum.csv"
    assert fpath.exists()

    df = inspect_csv(fpath)
    guessed_f, guessed_m = guess_csv_columns(list(df.columns))
    assert guessed_f == "Frequency_Hz"
    assert guessed_m == "Amplitude_dBuA_m"

    trace, report = load_csv_trace(
        fpath,
        freq_col=guessed_f,
        meas_col=guessed_m,
        source_filename=fpath.name,
    )
    assert report["valid_rows"] == 27
    assert report["dropped_rows"] == 0
    assert report["f_min"] == 10.0
    assert report["f_max"] == 400000.0
    assert trace.measurement_unit == "dBµA/m"


def test_sample_fixtures_compatibility():
    traces = []
    for fname in ["vga_ch1_red_spectrum.csv", "vga_ch2_green_spectrum.csv", "vga_ch3_blue_spectrum.csv"]:
        fpath = SAMPLE_DIR / fname
        trace, _ = load_csv_trace(fpath, freq_col="Frequency_Hz", meas_col="Amplitude_dBuA_m", source_filename=fname)
        traces.append(trace)

    report = analyze_trace_compatibility(traces)
    assert report.is_unit_compatible is True
    assert report.has_frequency_overlap is True
    assert report.can_combine_scientifically is True
    assert len(report.warnings) == 0


def test_incompatible_sample_fixture():
    f1 = SAMPLE_DIR / "vga_ch1_red_spectrum.csv"
    f2 = SAMPLE_DIR / "vga_incompatible_volts.csv"

    t1, _ = load_csv_trace(f1, freq_col="Frequency_Hz", meas_col="Amplitude_dBuA_m")
    t2, _ = load_csv_trace(f2, freq_col="Frequency_Hz", meas_col="Voltage_V", measurement_unit="V")

    report = analyze_trace_compatibility([t1, t2])
    assert report.is_unit_compatible is False
    assert report.can_combine_scientifically is False
