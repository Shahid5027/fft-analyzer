"""Spectral peak detection utilities for VGA EMI Analyzer."""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.signal import find_peaks

from vga_emi.models import MeasurementTrace, PeakResult


def detect_peaks(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    prominence: float = 4.0,
    distance: int = 15,
    height: Optional[float] = None,
    top_k: int = 8,
) -> PeakResult:
    """Detect prominent spectral peaks in frequency-domain measurements.

    Args:
        frequencies: 1D array of frequency points.
        amplitudes: 1D array of amplitudes (dB or linear).
        prominence: Minimum prominence required for a peak.
        distance: Minimum distance (in samples) between adjacent peaks.
        height: Optional minimum absolute height for peaks.
        top_k: Maximum number of highest peaks to return (ordered by amplitude descending).

    Returns:
        PeakResult containing peak frequencies, amplitudes, and indices.
    """
    if len(frequencies) != len(amplitudes):
        raise ValueError("Frequencies and amplitudes must have equal length.")
    if len(frequencies) < 3:
        return PeakResult(
            frequencies=np.array([]),
            amplitudes=np.array([]),
            indices=np.array([], dtype=int),
        )

    peak_indices, _ = find_peaks(
        amplitudes,
        prominence=prominence,
        distance=distance,
        height=height,
    )

    if len(peak_indices) == 0:
        return PeakResult(
            frequencies=np.array([]),
            amplitudes=np.array([]),
            indices=np.array([], dtype=int),
        )

    # Sort peaks by amplitude descending
    sorted_order = np.argsort(amplitudes[peak_indices])[::-1]
    top_indices = peak_indices[sorted_order][:top_k]

    # Re-order by frequency for logical display
    freq_order = np.argsort(frequencies[top_indices])
    final_indices = top_indices[freq_order]

    return PeakResult(
        frequencies=frequencies[final_indices],
        amplitudes=amplitudes[final_indices],
        indices=final_indices,
    )


def detect_trace_peaks(
    trace: MeasurementTrace,
    prominence: float = 4.0,
    distance: int = 15,
    top_k: int = 8,
) -> PeakResult:
    """Convenience helper to detect peaks directly on a MeasurementTrace."""
    return detect_peaks(
        frequencies=trace.frequencies,
        amplitudes=trace.measurements,
        prominence=prominence,
        distance=distance,
        top_k=top_k,
    )
