"""Tests for deterministic demo data generators."""

import numpy as np
import pytest

from vga_emi.demo_data import generate_synthetic_fft_traces, generate_vga_time_domain


def test_fft_traces_determinism():
    traces1 = generate_synthetic_fft_traces(seed=42, f_min=10.0, f_max=400000.0, num_points=500, num_traces=3)
    traces2 = generate_synthetic_fft_traces(seed=42, f_min=10.0, f_max=400000.0, num_points=500, num_traces=3)

    assert len(traces1) == 3
    assert len(traces2) == 3
    for t1, t2 in zip(traces1, traces2):
        np.testing.assert_array_equal(t1.frequencies, t2.frequencies)
        np.testing.assert_array_equal(t1.measurements, t2.measurements)


def test_fft_traces_distinct():
    traces = generate_synthetic_fft_traces(seed=42, f_min=10.0, f_max=400000.0, num_points=500, num_traces=3)
    # Ensure trace 1, trace 2, and trace 3 have different measurements
    assert not np.allclose(traces[0].measurements, traces[1].measurements)
    assert not np.allclose(traces[1].measurements, traces[2].measurements)
    assert not np.allclose(traces[0].measurements, traces[2].measurements)


def test_fft_traces_invalid_range():
    with pytest.raises(ValueError, match="f_min .* must be less than f_max"):
        generate_synthetic_fft_traces(f_min=500.0, f_max=100.0)

    with pytest.raises(ValueError, match="f_min must be positive"):
        generate_synthetic_fft_traces(f_min=0.0, f_max=100.0, scale="log")


def test_time_domain_waveforms():
    time_traces = generate_vga_time_domain(seed=42, time_window_us=100.0, num_samples=1000, num_channels=3)
    assert len(time_traces) == 3
    assert time_traces[0].channel_id == "ch1"
    assert time_traces[1].channel_id == "ch2"
    assert time_traces[2].channel_id == "ch3"

    # Ensure traces have distinct characteristics
    assert not np.allclose(time_traces[0].voltage, time_traces[1].voltage)
    assert not np.allclose(time_traces[1].voltage, time_traces[2].voltage)

    # Determinism
    time_traces_repeat = generate_vga_time_domain(seed=42, time_window_us=100.0, num_samples=1000, num_channels=3)
    np.testing.assert_array_equal(time_traces[0].voltage, time_traces_repeat[0].voltage)
