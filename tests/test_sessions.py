"""Tests for safe session serialization and persistence."""

import numpy as np
import pytest

from vga_emi.models import MeasurementTrace, TimeDomainTrace
from vga_emi.sessions.session_store import (
    save_session_to_bytes,
    load_session_from_bytes,
    save_session_to_file,
    load_session_from_file,
)


def test_session_roundtrip(tmp_path):
    settings = {
        "num_traces": 3,
        "f_min": 20.0,
        "f_max": 200000.0,
        "ref_line_db": 45.0,
    }
    freqs = np.array([20.0, 50.0, 100.0])
    meas = np.array([12.3, 34.5, 45.6])
    trace = MeasurementTrace(
        trace_id="t1",
        name="Trace 1",
        frequencies=freqs,
        measurements=meas,
    )

    t = np.linspace(0, 10, 20)
    v = np.sin(t)
    w_trace = TimeDomainTrace(
        channel_id="ch1",
        name="Channel 1",
        time=t,
        voltage=v,
    )

    bundle_bytes = save_session_to_bytes(settings, [trace], [w_trace])
    loaded_settings, loaded_traces, loaded_waves = load_session_from_bytes(bundle_bytes)

    assert loaded_settings["num_traces"] == 3
    assert loaded_settings["f_min"] == 20.0
    assert len(loaded_traces) == 1
    assert loaded_traces[0].trace_id == "t1"
    np.testing.assert_array_equal(loaded_traces[0].frequencies, freqs)
    np.testing.assert_array_equal(loaded_traces[0].measurements, meas)

    assert len(loaded_waves) == 1
    assert loaded_waves[0].channel_id == "ch1"
    np.testing.assert_array_equal(loaded_waves[0].time, t)
    np.testing.assert_array_equal(loaded_waves[0].voltage, v)

    # Test file path save & load
    session_file = tmp_path / "test_session.vga_session"
    save_session_to_file(session_file, settings, [trace], [w_trace])
    f_settings, f_traces, f_waves = load_session_from_file(session_file)
    assert len(f_traces) == 1
    np.testing.assert_array_equal(f_traces[0].measurements, meas)


def test_corrupted_session_bytes():
    with pytest.raises(Exception):
        load_session_from_bytes(b"not a valid zip file")
