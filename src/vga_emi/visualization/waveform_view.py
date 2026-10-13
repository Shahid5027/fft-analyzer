"""Dedicated time-domain waveform visualization."""

from __future__ import annotations

from typing import List, Optional
import matplotlib.pyplot as plt

from vga_emi.models import TimeDomainTrace
from vga_emi.visualization.oscilloscope_dashboard import (
    DARK_BG, PANEL_BG, GRID_COLOR, SPINE_COLOR, TEXT_COLOR, TRACE_COLORS
)


def render_waveform_view(
    traces: List[TimeDomainTrace],
    selected_channels: Optional[List[str]] = None,
    time_window_us: Optional[float] = None,
    dark_mode: bool = True,
) -> plt.Figure:
    """Render high-resolution time-domain waveform panel.

    Args:
        traces: List of TimeDomainTrace objects.
        selected_channels: List of channel_ids to display (default: all).
        time_window_us: Optional upper limit for time axis.
        dark_mode: Whether to use oscilloscope dark styling.

    Returns:
        Matplotlib Figure object.
    """
    bg_color = DARK_BG if dark_mode else "#ffffff"
    panel_color = PANEL_BG if dark_mode else "#ffffff"
    grid_color = GRID_COLOR if dark_mode else "#e5e7eb"
    text_color = TEXT_COLOR if dark_mode else "#1f2937"
    spine_color = SPINE_COLOR if dark_mode else "#6b7280"

    fig, ax = plt.subplots(figsize=(11, 5.0), facecolor=bg_color)
    ax.set_facecolor(panel_color)
    ax.grid(True, which="both", color=grid_color, linestyle=":", linewidth=0.7)

    for idx, trace in enumerate(traces):
        if selected_channels is not None and trace.channel_id not in selected_channels:
            continue
        color = TRACE_COLORS[idx % len(TRACE_COLORS)] if dark_mode else None
        ax.plot(trace.time, trace.voltage, label=trace.name, linewidth=1.1, color=color)

    time_unit = traces[0].time_unit if traces else "µs"
    v_unit = traces[0].voltage_unit if traces else "V"

    if time_window_us is not None:
        ax.set_xlim(0, time_window_us)

    ax.set_title(
        "Time-Domain Waveforms (VGA Video & Synchronization Channels)",
        color=text_color, fontsize=10.5, fontweight="semibold", pad=8, loc="left"
    )
    ax.set_xlabel(f"Time ({time_unit})", color=text_color, fontsize=8.5)
    ax.set_ylabel(f"Signal Voltage ({v_unit})", color=text_color, fontsize=8.5)
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
