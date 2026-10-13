"""Data and figure export utilities for VGA EMI Analyzer."""

from __future__ import annotations

import io
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vga_emi.models import MeasurementTrace, PeakResult


def figure_to_png_bytes(fig: plt.Figure, dpi: int = 300) -> bytes:
    """Render a Matplotlib figure to PNG bytes in memory.

    Args:
        fig: Matplotlib Figure.
        dpi: Dots per inch resolution.

    Returns:
        PNG image bytes.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def trace_to_csv_bytes(trace: MeasurementTrace) -> bytes:
    """Export a MeasurementTrace to CSV bytes with metadata header comments."""
    df = pd.DataFrame({
        f"Frequency_{trace.frequency_unit}": trace.frequencies,
        f"Measurement_{trace.measurement_unit.replace('/', '_')}": trace.measurements,
    })
    header_lines = [
        f"# VGA EMI Analyzer Export",
        f"# Trace Name: {trace.name}",
        f"# Source Type: {trace.source_type}",
        f"# Source File: {trace.source_filename or 'None'}",
        f"# Review Status: {trace.review_status}",
        f"# Points: {len(trace.frequencies)}",
        f"# Frequency Unit: {trace.frequency_unit}",
        f"# Measurement Unit: {trace.measurement_unit}",
        f"#",
    ]
    header_str = "\n".join(header_lines) + "\n"
    csv_str = df.to_csv(index=False)
    return (header_str + csv_str).encode("utf-8")


def peaks_to_csv_bytes(peaks: PeakResult, freq_unit: str = "Hz", meas_unit: str = "dBµA/m") -> bytes:
    """Export detected peaks to CSV bytes."""
    records = peaks.to_records(f_unit=freq_unit, m_unit=meas_unit)
    df = pd.DataFrame(records)
    header_lines = [
        f"# VGA EMI Analyzer - Detected Spectral Peaks",
        f"# Total Peaks: {len(records)}",
        f"#",
    ]
    header_str = "\n".join(header_lines) + "\n"
    csv_str = df.to_csv(index=False)
    return (header_str + csv_str).encode("utf-8")


def comparison_to_csv_bytes(traces: List[MeasurementTrace]) -> bytes:
    """Export a multi-trace comparison summary to CSV bytes."""
    rows = []
    for tr in traces:
        rows.append({
            "Trace ID": tr.trace_id,
            "Trace Name": tr.name,
            "Source Type": tr.source_type,
            "Source File": tr.source_filename or "N/A",
            "Review Status": tr.review_status,
            "Frequency Min (Hz)": float(np.min(tr.frequencies)),
            "Frequency Max (Hz)": float(np.max(tr.frequencies)),
            "Frequency Points": len(tr.frequencies),
            "Measurement Min": float(np.min(tr.measurements)),
            "Measurement Max": float(np.max(tr.measurements)),
            "Measurement Unit": tr.measurement_unit,
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")
