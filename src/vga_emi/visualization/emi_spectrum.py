"""Publication-style final EMI spectrum visualization resembling the reference Python output."""

from __future__ import annotations

from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np

from vga_emi.models import MeasurementTrace, PeakResult


SPECTRUM_COLORS = [
    "#0b57d0",  # Publication royal blue
    "#00796b",  # Muted teal
    "#6a1b9a",  # Muted purple
    "#c2185b",  # Muted rose
    "#e65100",  # Muted amber
]


def render_final_emi_spectrum(
    trace_or_traces: Union[MeasurementTrace, List[MeasurementTrace]],
    peaks: Optional[PeakResult] = None,
    f_min: float = 10.0,
    f_max: float = 400000.0,
    show_reference_line: bool = False,
    reference_line_db: float = 50.0,
    reference_line_label: str = "Reference Level (50 dBµA/m)",
    show_peak_annotations: bool = True,
    title: Optional[str] = None,
    is_simulated: Optional[bool] = None,
) -> plt.Figure:
    """Render publication-style final EMI spectrum plot.

    Features:
      - Crisp white background
      - Dark axes with subtle gray grid
      - Publication royal blue trace
      - Logarithmic frequency axis (10 Hz to 400 kHz)
      - Collision-aware compact peak callouts
      - Clean engineering typography

    Args:
        trace_or_traces: Single MeasurementTrace or list of traces to plot.
        peaks: Optional detected peaks on the primary trace.
        f_min: Minimum display frequency in Hz.
        f_max: Maximum display frequency in Hz.
        show_reference_line: Whether to draw the user reference line.
        reference_line_db: Value for reference line in dBµA/m.
        reference_line_label: Label for the reference line.
        show_peak_annotations: Whether to annotate peak frequencies.
        title: Optional custom plot title.
        is_simulated: Override for simulation indicator.

    Returns:
        Matplotlib Figure object.
    """
    if isinstance(trace_or_traces, list):
        traces = trace_or_traces
    else:
        traces = [trace_or_traces] if trace_or_traces is not None else []

    fig, ax = plt.subplots(figsize=(10.5, 6.0), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    # Logarithmic frequency axis
    ax.set_xscale("log")
    ax.grid(True, which="major", color="#e2e4e8", linestyle="-", linewidth=0.7)
    ax.grid(True, which="minor", color="#f1f3f5", linestyle=":", linewidth=0.5)

    if not traces:
        ax.text(0.5, 0.5, "No Spectrum Data Available", ha="center", va="center", color="#6b7280", fontsize=9)
        return fig

    primary_trace = traces[0]
    simulated_flag = is_simulated if is_simulated is not None else (primary_trace.source_type == "synthetic")

    # Plot traces
    for idx, t in enumerate(traces):
        color = SPECTRUM_COLORS[idx % len(SPECTRUM_COLORS)]
        ax.plot(
            t.frequencies,
            t.measurements,
            color=color,
            linewidth=1.2 if idx == 0 else 1.0,
            label=f"{t.name}",
            zorder=3,
        )

    # Draw reference line if requested
    if show_reference_line:
        ax.axhline(
            y=reference_line_db,
            color="#b91c1c",
            linestyle="--",
            linewidth=1.1,
            label=f"{reference_line_label} [User Defined]",
            zorder=4,
        )

    # Collision-aware peak annotations on primary trace
    if show_peak_annotations and peaks is not None and len(peaks.frequencies) > 0:
        valid_mask = (peaks.frequencies >= f_min) & (peaks.frequencies <= f_max)
        p_freqs = peaks.frequencies[valid_mask]
        p_amps = peaks.amplitudes[valid_mask]

        if len(p_freqs) > 0:
            # Sort candidate peaks by amplitude descending
            sorted_order = np.argsort(p_amps)[::-1]
            chosen_indices: List[int] = []

            # Minimum log-frequency spacing threshold to prevent collision
            min_log_dist = 0.22

            for candidate in sorted_order:
                cand_log_f = np.log10(p_freqs[candidate])
                # Check collision with already chosen peaks
                collides = False
                for chosen in chosen_indices:
                    chosen_log_f = np.log10(p_freqs[chosen])
                    if abs(cand_log_f - chosen_log_f) < min_log_dist:
                        collides = True
                        break
                if not collides:
                    chosen_indices.append(candidate)
                if len(chosen_indices) >= 4:
                    break

            # Plot marker points
            ax.scatter(
                p_freqs[chosen_indices],
                p_amps[chosen_indices],
                color="#b91c1c",
                s=28,
                edgecolors="#7f1d1d",
                marker="o",
                zorder=5,
            )

            # Draw compact callout boxes
            for p_idx in chosen_indices:
                fx = p_freqs[p_idx]
                fy = p_amps[p_idx]
                f_display = f"{fx:.1f} Hz" if fx < 1000 else (f"{fx/1e3:.2f} kHz" if fx < 1e6 else f"{fx/1e6:.2f} MHz")
                label_text = f"{f_display} | {fy:.1f} {primary_trace.measurement_unit}"
                ax.annotate(
                    label_text,
                    xy=(fx, fy),
                    xytext=(0, 12),
                    textcoords="offset points",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    bbox=dict(boxstyle="square,pad=0.2", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.6),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.6),
                )

    # Compute bounding limits
    all_meas = np.concatenate([t.measurements for t in traces])
    y_min = max(0.0, float(np.min(all_meas)) - 5.0)
    y_max = float(np.max(all_meas)) + 12.0
    if show_reference_line:
        y_max = max(y_max, reference_line_db + 8.0)

    ax.set_xlim(f_min, f_max)
    ax.set_ylim(y_min, y_max)

    # Typography & Axis Spines
    plot_title = title or ("Radiated Emission Spectrum (10 Hz – 400 kHz)" if simulated_flag else f"Radiated Emission Spectrum — {primary_trace.name}")
    ax.set_title(plot_title, fontsize=11, fontweight="semibold", pad=10, color="#1f2937", loc="left")
    ax.set_xlabel(f"Frequency ({primary_trace.frequency_unit}) [Logarithmic]", fontsize=9, color="#374151")
    ax.set_ylabel(f"Magnitude ({primary_trace.measurement_unit})", fontsize=9, color="#374151")

    ax.tick_params(colors="#374151", labelsize=8, which="both")
    for spine in ax.spines.values():
        spine.set_color("#4b5563")
        spine.set_linewidth(0.8)

    ax.legend(
        facecolor="#ffffff",
        edgecolor="#e5e7eb",
        fontsize=8,
        loc="upper right",
        framealpha=0.9,
    )

    # Subtle footer metadata
    footer_text = (
        "Dataset: Deterministic Synthetic Reference | Configurable Reference Level is User-Defined"
        if simulated_flag
        else f"Dataset: Calibrated Laboratory Measurement | Source: {primary_trace.source_filename or 'Direct Import'}"
    )
    fig.text(0.02, 0.015, footer_text, fontsize=7, color="#6b7280", style="normal")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig
