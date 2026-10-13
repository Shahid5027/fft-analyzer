"""Tests for peak detection algorithms."""

import numpy as np
import pytest

from vga_emi.processing.peak_detection import detect_peaks


def test_detect_known_peaks():
    freqs = np.linspace(1000, 10000, 1000)
    amps = np.ones_like(freqs) * 10.0  # 10 dB baseline

    # Add 2 known peaks at indices 200 and 600
    amps[200] = 50.0
    amps[600] = 70.0

    peaks = detect_peaks(freqs, amps, prominence=15.0, distance=50)
    assert len(peaks.frequencies) == 2
    assert np.isclose(peaks.frequencies[0], freqs[200])
    assert np.isclose(peaks.frequencies[1], freqs[600])
    assert np.isclose(peaks.amplitudes[0], 50.0)
    assert np.isclose(peaks.amplitudes[1], 70.0)


def test_detect_peaks_flat_signal():
    freqs = np.linspace(100, 1000, 100)
    amps = np.ones_like(freqs) * 25.0
    peaks = detect_peaks(freqs, amps, prominence=5.0)
    assert len(peaks.frequencies) == 0


def test_detect_peaks_top_k():
    freqs = np.linspace(1, 100, 100)
    amps = np.zeros_like(freqs)
    # Add 10 spikes of varying heights
    for i in range(10):
        amps[i * 10] = 10.0 + i

    peaks = detect_peaks(freqs, amps, prominence=2.0, distance=5, top_k=3)
    assert len(peaks.frequencies) == 3
    # Top 3 should be 17, 18, 19
    assert np.all(peaks.amplitudes >= 17.0)
