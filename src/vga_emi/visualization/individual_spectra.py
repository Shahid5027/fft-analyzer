"""Individual trace spectra and multi-trace comparison visualization."""

from __future__ import annotations

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from vga_emi.models import MeasurementTrace, PeakResult
from vga_emi.visualization.oscilloscope_dashboard import (
    DARK_BG, PANEL_BG, GRID_COLOR, SPINE_COLOR, TEXT_COLOR, TRACE_COLORS
)


def render_spectra_overlay(
    traces: List[MeasurementTrace],
    selected_indices: Optional[List[int]] = None,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    scale_type: str = "log",
    peak_results: Optional[Dict[str, PeakResult]] = None,
    show_peaks: bool = True,
    dark_mode: bool = True,
) -> plt.Figure:
    """Render overlaid comparison of multiple frequency-domain spectra.

    Args:
        traces: List of MeasurementTrace objects.
        selected_indices: Indices of traces to display.
        f_min, f_max: Frequency bounds.
        scale_type: 'log' or 'linear'.
        peak_results: Dictionary mapping trace_id to PeakResult for annotation.
        show_peaks: Whether to highlight detected peaks.
        dark_mode: True for dark oscilloscope theme, False for publication white.

    Returns:
        Matplotlib Figure object.
    """
    bg_color = DARK_BG if dark_mode else "#ffffff"
    panel_color = PANEL_BG if dark_mode else "#ffffff"
    grid_color = GRID_COLOR if dark_mode else "#e5e7eb"
    text_color = TEXT_COLOR if dark_mode else "#1f2937"
    spine_color = SPINE_COLOR if dark_mode else "#6b7280"

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=bg_color)
    ax.set_facecolor(panel_color)
    ax.grid(True, which="both", color=grid_color, linestyle=":", linewidth=0.6)

    if scale_type == "log":
        ax.set_xscale("log")

    if selected_indices is None:
        selected_indices = list(range(len(traces)))

    for idx in selected_indices:
        if idx >= len(traces):
            continue
        t = traces[idx]
        color = TRACE_COLORS[idx % len(TRACE_COLORS)] if dark_mode else None
        line = ax.plot(
            t.frequencies,
            t.measurements,
            label=f"{t.name}",
            linewidth=1.1,
            color=color,
        )
        line_color = line[0].get_color()

        if show_peaks and peak_results and t.trace_id in peak_results:
            p_res = peak_results[t.trace_id]
            if len(p_res.frequencies) > 0:
                ax.scatter(
                    p_res.frequencies,
                    p_res.amplitudes,
                    color=line_color,
                    s=25,
                    edgecolors="#ffffff" if dark_mode else "#000000",
                    linewidths=0.5,
                    zorder=5,
                    marker="^",
                )

    if f_min is not None and f_max is not None:
        ax.set_xlim(f_min, f_max)

    ax.set_title("Frequency Spectra Comparison & Overlay", color=text_color, fontsize=10.5, fontweight="semibold", pad=8, loc="left")
    freq_unit = traces[0].frequency_unit if traces else "Hz"
    meas_unit = traces[0].measurement_unit if traces else "dBµA/m"
    ax.set_xlabel(f"Frequency ({freq_unit})", color=text_color, fontsize=8.5)
    ax.set_ylabel(f"Amplitude ({meas_unit})", color=text_color, fontsize=8.5)
    ax.tick_params(colors=text_color, labelsize=8, which="both")
    for spine in ax.spines.values():
        spine.set_color(spine_color)

    ax.legend(
        facecolor=panel_color,
        edgecolor=spine_color,
        labelcolor=text_color,
        fontsize=8,
        loc="upper right",
    )
    plt.tight_layout()
    return fig
