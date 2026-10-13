"""Oscilloscope-style 6-panel technical dashboard visualization."""

from __future__ import annotations

from typing import List, Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from vga_emi.models import MeasurementTrace, TimeDomainTrace


DARK_BG = "#0f1117"
PANEL_BG = "#161922"
GRID_COLOR = "#252936"
SPINE_COLOR = "#343946"
TEXT_COLOR = "#c8ccd4"

TRACE_COLORS = [
    "#ffea00",  # Math 1 / CH1 - Yellow
    "#00e5ff",  # Math 2 / CH2 - Cyan
    "#e040fb",  # Math 3 / CH3 - Magenta
    "#00e676",  # Math 4 / CH4 - Green
]

COMBINED_COLOR = "#ff7043"  # Math 6 - Coral
SQRT_COLOR = "#76ff03"      # Math 4 - Lime


def _style_oscilloscope_axis(ax: plt.Axes, title: str, x_label: str, y_label: str, is_log_x: bool = True) -> None:
    """Apply laboratory oscilloscope styling to an axis."""
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, which="both", color=GRID_COLOR, linestyle=":", linewidth=0.6)
    if is_log_x:
        ax.set_xscale("log")
    ax.set_title(title, color=TEXT_COLOR, fontsize=9.5, fontweight="semibold", pad=6, loc="left")
    ax.set_xlabel(x_label, color=TEXT_COLOR, fontsize=8)
    ax.set_ylabel(y_label, color=TEXT_COLOR, fontsize=8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7.5, which="both")
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)


def render_oscilloscope_dashboard(
    fft_traces: List[MeasurementTrace],
    combined_trace: Optional[MeasurementTrace],
    sqrt_trace: Optional[MeasurementTrace],
    time_traces: List[TimeDomainTrace],
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
) -> plt.Figure:
    """Render the 6-panel oscilloscope-style instrument dashboard.

    Layout:
      - Row 1: Math 1 FFT, Math 2 FFT, Math 3 FFT (3 columns)
      - Row 2: Math 6 Combined FFT (Demo), Math 4 Square-Root Result (Demo) (2 columns)
      - Row 3: Waveform View showing multi-channel time-domain waveforms (full width)

    Args:
        fft_traces: Individual frequency domain traces.
        combined_trace: Combined FFT trace (Math 6).
        sqrt_trace: Sqrt trace (Math 4).
        time_traces: Time-domain waveform channels.
        f_min: Optional frequency minimum limit in Hz.
        f_max: Optional frequency maximum limit in Hz.

    Returns:
        Matplotlib Figure object.
    """
    fig = plt.figure(figsize=(15, 10.5), facecolor=DARK_BG)
    gs = GridSpec(3, 6, figure=fig, hspace=0.34, wspace=0.28)

    # ----------------- Row 1: Individual FFT Traces -----------------
    ax_fft1 = fig.add_subplot(gs[0, 0:2])
    ax_fft2 = fig.add_subplot(gs[0, 2:4])
    ax_fft3 = fig.add_subplot(gs[0, 4:6])
    row1_axes = [ax_fft1, ax_fft2, ax_fft3]

    for idx, ax in enumerate(row1_axes):
        if idx < len(fft_traces):
            t = fft_traces[idx]
            color = TRACE_COLORS[idx % len(TRACE_COLORS)]
            is_log = t.axis_scales.get("x") == "log"
            _style_oscilloscope_axis(
                ax,
                title=f"{t.name}",
                x_label=f"Frequency ({t.frequency_unit})",
                y_label=f"Amplitude ({t.measurement_unit})",
                is_log_x=is_log,
            )
            ax.plot(t.frequencies, t.measurements, color=color, linewidth=1.1, label=f"Trace {idx+1}")
            if f_min and f_max:
                ax.set_xlim(f_min, f_max)
            ax.legend(facecolor=PANEL_BG, edgecolor=SPINE_COLOR, labelcolor=TEXT_COLOR, fontsize=7, loc="upper right")
        else:
            _style_oscilloscope_axis(ax, f"CH{idx+1} FFT [Unassigned]", "Frequency (Hz)", "Amplitude", True)
            ax.text(0.5, 0.5, "No Data Loaded", transform=ax.transAxes, ha="center", va="center", color="#4b5563", fontsize=8)

    # ----------------- Row 2: Combined FFT & Sqrt Result -----------------
    ax_comb = fig.add_subplot(gs[1, 0:3])
    ax_sqrt = fig.add_subplot(gs[1, 3:6])

    # Combined FFT (Math 6)
    if combined_trace is not None:
        is_log = combined_trace.axis_scales.get("x") == "log"
        _style_oscilloscope_axis(
            ax_comb,
            title=f"{combined_trace.name}",
            x_label=f"Frequency ({combined_trace.frequency_unit})",
            y_label=f"Amplitude ({combined_trace.measurement_unit})",
            is_log_x=is_log,
        )
        ax_comb.plot(
            combined_trace.frequencies,
            combined_trace.measurements,
            color=COMBINED_COLOR,
            linewidth=1.15,
            label="Math 6 (Linear RSS)",
        )
        if f_min and f_max:
            ax_comb.set_xlim(f_min, f_max)
        ax_comb.text(
            0.02, 0.90, "[DEMO CALCULATION — LINEAR RSS]",
            transform=ax_comb.transAxes,
            color="#9ca3af", fontsize=7.5,
        )
        ax_comb.legend(facecolor=PANEL_BG, edgecolor=SPINE_COLOR, labelcolor=TEXT_COLOR, fontsize=7.5, loc="upper right")
    else:
        _style_oscilloscope_axis(ax_comb, "Math 6: Combined FFT [Unassigned]", "Frequency (Hz)", "Amplitude", True)
        ax_comb.text(0.5, 0.5, "Awaiting Validated Combination Model", transform=ax_comb.transAxes, ha="center", va="center", color="#4b5563", fontsize=8)

    # Sqrt Result (Math 4)
    if sqrt_trace is not None:
        is_log = sqrt_trace.axis_scales.get("x") == "log"
        _style_oscilloscope_axis(
            ax_sqrt,
            title=f"{sqrt_trace.name}",
            x_label=f"Frequency ({sqrt_trace.frequency_unit})",
            y_label=f"Magnitude ({sqrt_trace.measurement_unit})",
            is_log_x=is_log,
        )
        ax_sqrt.plot(
            sqrt_trace.frequencies,
            sqrt_trace.measurements,
            color=SQRT_COLOR,
            linewidth=1.15,
            label="Math 4 (Sqrt)",
        )
        if f_min and f_max:
            ax_sqrt.set_xlim(f_min, f_max)
        ax_sqrt.text(
            0.02, 0.90, "[DEMO CALCULATION — SQUARE ROOT]",
            transform=ax_sqrt.transAxes,
            color="#9ca3af", fontsize=7.5,
        )
        ax_sqrt.legend(facecolor=PANEL_BG, edgecolor=SPINE_COLOR, labelcolor=TEXT_COLOR, fontsize=7.5, loc="upper right")
    else:
        _style_oscilloscope_axis(ax_sqrt, "Math 4: Square Root Result [Unassigned]", "Frequency (Hz)", "Amplitude", True)
        ax_sqrt.text(0.5, 0.5, "Awaiting Source Trace", transform=ax_sqrt.transAxes, ha="center", va="center", color="#4b5563", fontsize=8)

    # ----------------- Row 3: Time-Domain Waveform View -----------------
    ax_wave = fig.add_subplot(gs[2, 0:6])
    _style_oscilloscope_axis(
        ax_wave,
        title="Time-Domain Waveforms (VGA Video & Synchronization Channels)",
        x_label=f"Time ({time_traces[0].time_unit if time_traces else 'µs'})",
        y_label=f"Voltage ({time_traces[0].voltage_unit if time_traces else 'V'})",
        is_log_x=False,
    )

    for ch_idx, w_trace in enumerate(time_traces):
        color = TRACE_COLORS[ch_idx % len(TRACE_COLORS)]
        ax_wave.plot(
            w_trace.time,
            w_trace.voltage,
            color=color,
            linewidth=1.1,
            label=f"{w_trace.name}",
        )

    ax_wave.legend(
        facecolor=PANEL_BG,
        edgecolor=SPINE_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=7.5,
        loc="upper right",
        ncol=min(len(time_traces), 4),
    )

    # Header labels
    fig.text(
        0.02, 0.985, "Multi-Channel Oscilloscope & FFT Monitor",
        color="#d1d5db", fontsize=10, fontweight="bold", va="top"
    )
    is_synth = any(t.source_type == "synthetic" for t in fft_traces) if fft_traces else True
    fig.text(
        0.98, 0.985, "Synthetic Reference Dataset" if is_synth else "Imported Laboratory Traces",
        color="#9ca3af", fontsize=8, ha="right", va="top"
    )

    return fig
