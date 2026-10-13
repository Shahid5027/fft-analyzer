"""VGA Cable EMI Analyzer - Scientific Visualization & Measurement Software.

A laboratory-grade engineering application for analyzing VGA cable radiated
electromagnetic emissions.

Sections:
- Overview (Oscilloscope dashboard, signal metrics, system status)
- Import Measurements (Multi-CSV ingestion, validation reports, screenshot calibration)
- Review and Compare (Multi-trace overlay, compatibility analysis, peak tables)
- EMI Spectrum (Publication-quality logarithmic spectrum)
- Export Results (High-resolution figures, tabular CSV spectra, peak records)
- Sessions (Safe JSON metadata + NumPy NPZ array persistence)
"""

from __future__ import annotations

import io
from pathlib import Path
import sys

# Ensure src directory is on pythonpath for seamless local execution
SRC_DIR = str(Path(__file__).resolve().parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from vga_emi import __version__
from vga_emi.demo_data import generate_synthetic_fft_traces, generate_vga_time_domain
from vga_emi.extraction.axis_calibration import create_trace_from_extracted_pixels
from vga_emi.extraction.plot_detection import detect_plot_bounds
from vga_emi.extraction.trace_detection import create_preview_overlay, extract_trace_pixels
from vga_emi.ingestion.csv_loader import guess_csv_columns, inspect_csv, load_csv_trace
from vga_emi.ingestion.image_loader import load_and_validate_image
from vga_emi.models import MeasurementTrace, PeakResult, TimeDomainTrace
from vga_emi.processing.combination import combine_traces_rss, compute_demo_sqrt
from vga_emi.processing.compatibility import analyze_trace_compatibility
from vga_emi.processing.export import (
    comparison_to_csv_bytes,
    figure_to_png_bytes,
    peaks_to_csv_bytes,
    trace_to_csv_bytes,
)
from vga_emi.processing.peak_detection import detect_trace_peaks
from vga_emi.sessions.session_store import load_session_from_bytes, save_session_to_bytes
from vga_emi.visualization.emi_spectrum import render_final_emi_spectrum
from vga_emi.visualization.individual_spectra import render_spectra_overlay
from vga_emi.visualization.oscilloscope_dashboard import render_oscilloscope_dashboard
from vga_emi.visualization.waveform_view import render_waveform_view


# ---------------------------------------------------------
# Page Configuration & Professional Engineering Stylesheet
# ---------------------------------------------------------
st.set_page_config(
    page_title="VGA Cable EMI Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal CSS for restrained engineering instrument interface
st.markdown(
    """
    <style>
    /* Typography & Layout Tightening */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .block-container {
        padding-top: 4.25rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    /* Compact Application Header */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1rem;
    }
    .app-title {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.8rem;
        color: #6b7280;
        margin-left: 0.75rem;
    }
    .app-version {
        font-size: 0.75rem;
        color: #9ca3af;
        font-family: monospace;
    }
    /* Status Badges */
    .status-badge-demo {
        display: inline-block;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
        border-radius: 3px;
        background: rgba(217, 119, 6, 0.12);
        color: #d97706;
        border: 1px solid rgba(217, 119, 6, 0.3);
        margin-bottom: 0.75rem;
    }
    .status-badge-meas {
        display: inline-block;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
        border-radius: 3px;
        background: rgba(16, 185, 129, 0.12);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.3);
        margin-bottom: 0.75rem;
    }
    /* Clean sidebar headers */
    .sidebar-section-title {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280;
        margin-top: 1rem;
        margin-bottom: 0.35rem;
    }
    /* Streamlit tabs font size and spacing */
    button[data-baseweb="tab"] {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.4rem 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------
def load_default_config() -> dict:
    config_path = Path(__file__).parent / "config" / "defaults.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


DEFAULTS = load_default_config()


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "demo"  # 'demo' or 'measurement'

if "seed" not in st.session_state:
    st.session_state.seed = 42

if "num_demo_traces" not in st.session_state:
    st.session_state.num_demo_traces = DEFAULTS.get("traces", {}).get("num_active", 3)

if "f_min" not in st.session_state:
    st.session_state.f_min = float(DEFAULTS.get("frequency", {}).get("min_hz", 10.0))

if "f_max" not in st.session_state:
    st.session_state.f_max = float(DEFAULTS.get("frequency", {}).get("max_hz", 400000.0))

if "f_scale" not in st.session_state:
    st.session_state.f_scale = DEFAULTS.get("frequency", {}).get("scale", "log")

if "time_window_us" not in st.session_state:
    st.session_state.time_window_us = float(DEFAULTS.get("waveform", {}).get("time_window_us", 100.0))

if "num_samples" not in st.session_state:
    st.session_state.num_samples = int(DEFAULTS.get("waveform", {}).get("num_samples", 2048))

if "peak_prominence" not in st.session_state:
    st.session_state.peak_prominence = float(DEFAULTS.get("peak_detection", {}).get("prominence", 4.0))

if "ref_line_enabled" not in st.session_state:
    st.session_state.ref_line_enabled = bool(DEFAULTS.get("reference_line", {}).get("enabled", False))

if "ref_line_db" not in st.session_state:
    st.session_state.ref_line_db = float(DEFAULTS.get("reference_line", {}).get("value_db", 50.0))

if "ref_line_label" not in st.session_state:
    st.session_state.ref_line_label = str(DEFAULTS.get("reference_line", {}).get("label", "Reference Level (50 dBµA/m)"))

if "imported_traces" not in st.session_state:
    st.session_state.imported_traces = {}  # Dict[filename, MeasurementTrace]

if "allow_demo_rss_on_measurement" not in st.session_state:
    st.session_state.allow_demo_rss_on_measurement = False


# ---------------------------------------------------------
# Synthetic Demo Data Generation
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_demo_data(seed: int, f_min: float, f_max: float, num_traces: int, scale: str, time_window_us: float, num_samples: int):
    fft_traces = generate_synthetic_fft_traces(
        seed=seed,
        f_min=f_min,
        f_max=f_max,
        num_points=2048,
        num_traces=num_traces,
        scale=scale,
    )
    time_traces = generate_vga_time_domain(
        seed=seed,
        time_window_us=time_window_us,
        num_samples=num_samples,
        num_channels=num_traces,
    )
    return fft_traces, time_traces


demo_fft_traces, demo_time_traces = get_cached_demo_data(
    seed=st.session_state.seed,
    f_min=st.session_state.f_min,
    f_max=st.session_state.f_max,
    num_traces=st.session_state.num_demo_traces,
    scale=st.session_state.f_scale,
    time_window_us=st.session_state.time_window_us,
    num_samples=st.session_state.num_samples,
)

# Demo Math Calculations
demo_combined_trace = combine_traces_rss(demo_fft_traces[:3], name="Math 6: Linear RSS [Simulated]", treat_as_db=True)
demo_sqrt_trace = compute_demo_sqrt(demo_combined_trace, name="Math 4: Square Root [Simulated]")


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-section-title">Operation Mode</div>', unsafe_allow_html=True)
    mode_selection = st.radio(
        "Mode Selection",
        options=["Demo Mode (Synthetic Data)", "Measurement Mode (Imported Data)"],
        index=0 if st.session_state.app_mode == "demo" else 1,
        label_visibility="collapsed",
    )
    new_mode = "demo" if "Demo" in mode_selection else "measurement"
    if new_mode != st.session_state.app_mode:
        st.session_state.app_mode = new_mode
        st.rerun()

    # Presentation Demo Action
    if st.button("Load Presentation Demo Dataset", use_container_width=True):
        st.session_state.app_mode = "demo"
        st.session_state.seed = 42
        st.session_state.num_demo_traces = 3
        st.session_state.f_min = 10.0
        st.session_state.f_max = 400000.0
        st.session_state.f_scale = "log"
        st.session_state.ref_line_enabled = True
        st.session_state.ref_line_db = 50.0
        st.session_state.ref_line_label = "Reference Level (50 dBµA/m)"
        st.rerun()

    st.markdown('<div class="sidebar-section-title">Input and Channel Selection</div>', unsafe_allow_html=True)
    if st.session_state.app_mode == "demo":
        st.session_state.num_demo_traces = st.slider(
            "Simulated Channels",
            min_value=2,
            max_value=8,
            value=st.session_state.num_demo_traces,
        )
    else:
        num_loaded = len(st.session_state.imported_traces)
        if num_loaded > 0:
            st.caption(f"{num_loaded} laboratory trace(s) active in workspace.")
            if st.session_state.imported_traces:
                selected_preview_key = st.selectbox(
                    "Inspect Active Trace",
                    options=list(st.session_state.imported_traces.keys()),
                    index=0,
                )
        else:
            st.caption("No imported traces loaded. Use 'Import Measurements' to load CSV or image data.")

    st.markdown('<div class="sidebar-section-title">Frequency Range</div>', unsafe_allow_html=True)
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.session_state.f_min = st.number_input(
            "F Min (Hz)",
            min_value=1.0,
            max_value=10000.0,
            value=st.session_state.f_min,
            step=10.0,
        )
    with f_col2:
        st.session_state.f_max = st.number_input(
            "F Max (Hz)",
            min_value=1000.0,
            max_value=2000000.0,
            value=st.session_state.f_max,
            step=10000.0,
        )

    st.markdown('<div class="sidebar-section-title">Axis Scaling</div>', unsafe_allow_html=True)
    st.session_state.f_scale = st.radio(
        "Frequency Axis Scale",
        options=["log", "linear"],
        index=0 if st.session_state.f_scale == "log" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section-title">Peak Detection</div>', unsafe_allow_html=True)
    st.session_state.peak_prominence = st.slider(
        "Prominence Threshold",
        min_value=1.0,
        max_value=20.0,
        value=st.session_state.peak_prominence,
        step=0.5,
    )

    st.markdown('<div class="sidebar-section-title">Reference Level</div>', unsafe_allow_html=True)
    st.session_state.ref_line_enabled = st.checkbox(
        "Display Reference Line",
        value=st.session_state.ref_line_enabled,
    )
    if st.session_state.ref_line_enabled:
        st.session_state.ref_line_db = st.number_input(
            "Amplitude Value",
            value=st.session_state.ref_line_db,
            step=1.0,
        )
        st.session_state.ref_line_label = st.text_input(
            "Annotation Label",
            value=st.session_state.ref_line_label,
        )

    st.markdown('<div class="sidebar-section-title">Waveform Settings</div>', unsafe_allow_html=True)
    st.session_state.time_window_us = st.slider(
        "Window Duration (µs)",
        min_value=10.0,
        max_value=500.0,
        value=st.session_state.time_window_us,
        step=10.0,
    )

    st.markdown("---")
    if st.button("Reset Parameters", use_container_width=True):
        st.session_state.seed = 42
        st.session_state.f_min = 10.0
        st.session_state.f_max = 400000.0
        st.session_state.f_scale = "log"
        st.rerun()

    st.caption(f"VGA EMI Analyzer {__version__} | Python 3.12")


# ---------------------------------------------------------
# Application Header & Status Chip
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <div>
            <span class="app-title">VGA Cable EMI Analyzer</span>
            <span class="app-subtitle">Electromagnetic Interference Analysis & Measurement Platform</span>
        </div>
        <div class="app-version">v{__version__}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.app_mode == "demo":
    st.markdown('<div class="status-badge-demo">DEMO — SYNTHETIC DATA</div>', unsafe_allow_html=True)
else:
    num_imported = len(st.session_state.imported_traces)
    st.markdown(f'<div class="status-badge-meas">MEASUREMENT — IMPORTED DATA ({num_imported} Traces Active)</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Navigation Sections
# ---------------------------------------------------------
tab_overview, tab_import, tab_compare, tab_emi, tab_export, tab_sessions = st.tabs([
    "Overview",
    "Import Measurements",
    "Review and Compare",
    "EMI Spectrum",
    "Export Results",
    "Sessions",
])


# =========================================================
# SECTION 1: Overview
# =========================================================
with tab_overview:
    # Determine traces to render in overview dashboard
    if st.session_state.app_mode == "demo" or len(st.session_state.imported_traces) == 0:
        dash_fft_traces = demo_fft_traces[:3]
        dash_comb = demo_combined_trace
        dash_sqrt = demo_sqrt_trace
        dash_waves = demo_time_traces
        if len(st.session_state.imported_traces) == 0 and st.session_state.app_mode == "measurement":
            st.caption("No measurement CSVs currently loaded. Showing reference baseline until measurements are imported in section 'Import Measurements'.")
    else:
        imported_list = list(st.session_state.imported_traces.values())
        dash_fft_traces = imported_list[:3]
        if len(dash_fft_traces) > 1 and st.session_state.allow_demo_rss_on_measurement:
            try:
                dash_comb = combine_traces_rss(dash_fft_traces, name="Math 6: Linear RSS [Simulated]")
                dash_sqrt = compute_demo_sqrt(dash_comb)
            except Exception:
                dash_comb = None
                dash_sqrt = None
        else:
            dash_comb = None
            dash_sqrt = None
        dash_waves = demo_time_traces

    fig_dash = render_oscilloscope_dashboard(
        fft_traces=dash_fft_traces,
        combined_trace=dash_comb,
        sqrt_trace=dash_sqrt,
        time_traces=dash_waves,
        f_min=st.session_state.f_min,
        f_max=st.session_state.f_max,
    )
    st.pyplot(fig_dash)

    # Metrics Bar
    m_cols = st.columns(5)
    for i in range(min(3, len(dash_fft_traces))):
        with m_cols[i]:
            st.metric(f"Trace {i+1} Peak", f"{float(np.max(dash_fft_traces[i].measurements)):.1f} {dash_fft_traces[i].measurement_unit}")
    with m_cols[3]:
        if dash_comb:
            st.metric("Math 6 Peak", f"{float(np.max(dash_comb.measurements)):.1f} {dash_comb.measurement_unit}")
        else:
            st.metric("Math 6", "Uncombined")
    with m_cols[4]:
        if dash_sqrt:
            st.metric("Math 4 Peak", f"{float(np.max(dash_sqrt.measurements)):.1f} {dash_sqrt.measurement_unit}")
        else:
            st.metric("Math 4", "Unassigned")

    with st.expander("Mathematical & Physical Principles", expanded=False):
        st.markdown(
            """
            - **Decibel Mathematics**: Decibel units ($dB$) represent logarithmic ratios. Directly summing decibel readings ($dB_1 + dB_2$) is mathematically invalid.
            - **Linear RSS Operation**: In the synthetic demonstration, amplitudes are converted to linear scale, combined via Root-Sum-Square ($A_{rss} = \\sqrt{\\sum A_i^2}$), and converted back to decibels.
            - **Physical Integrity**: For actual laboratory inputs, multiple channels are not combined automatically without verified physical calibration and sensor coupling models.
            """
        )


# =========================================================
# SECTION 2: Import Measurements
# =========================================================
with tab_import:
    ingest_tabs = st.tabs(["CSV Numerical Data", "Screenshot Extraction"])

    # --- CSV Ingestion ---
    with ingest_tabs[0]:
        top_col1, top_col2 = st.columns([3, 1])
        with top_col2:
            if st.button("Load Sample Measurement Fixtures", use_container_width=True, help="Load 3 sample laboratory CSV spectra"):
                fixtures = [
                    (Path("sample_data/vga_ch1_red_spectrum.csv"), "Ch1 Red Video (Lab Sample)", "Hz", "dBµA/m"),
                    (Path("sample_data/vga_ch2_green_spectrum.csv"), "Ch2 Green Video (Lab Sample)", "Hz", "dBµA/m"),
                    (Path("sample_data/vga_ch3_blue_spectrum.csv"), "Ch3 Blue Sync (Lab Sample)", "Hz", "dBµA/m"),
                ]
                loaded_count = 0
                for fpath, name, f_unit, m_unit in fixtures:
                    if fpath.exists():
                        trace, rep = load_csv_trace(
                            fpath,
                            freq_col="Frequency_Hz",
                            meas_col="Amplitude_dBuA_m",
                            trace_id=fpath.stem,
                            trace_name=name,
                            source_filename=fpath.name,
                            frequency_unit=f_unit,
                            measurement_unit=m_unit,
                        )
                        st.session_state.imported_traces[fpath.name] = trace
                        loaded_count += 1
                if loaded_count > 0:
                    st.session_state.app_mode = "measurement"
                    st.rerun()

        with top_col1:
            uploaded_csvs = st.file_uploader(
                "Upload Measurement CSV Files (Up to 8 files)",
                type=["csv"],
                accept_multiple_files=True,
                help="Supports standard comma-separated spectra with frequency and amplitude columns.",
            )

        if uploaded_csvs:
            if len(uploaded_csvs) > 8:
                st.warning("Maximum of 8 CSV files supported. Displaying first 8 files.")
                uploaded_csvs = uploaded_csvs[:8]

            for idx, uploaded_file in enumerate(uploaded_csvs):
                with st.expander(f"File {idx+1}: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)", expanded=(idx < 3)):
                    try:
                        file_bytes = uploaded_file.getvalue()
                        df_preview = inspect_csv(file_bytes)
                        col_names = list(df_preview.columns)

                        guessed_f, guessed_m = guess_csv_columns(col_names)

                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            f_col_idx = col_names.index(guessed_f) if guessed_f in col_names else 0
                            sel_f_col = st.selectbox(f"Frequency Column [{uploaded_file.name}]", col_names, index=f_col_idx, key=f"f_col_{idx}")
                        with c2:
                            m_col_idx = col_names.index(guessed_m) if guessed_m in col_names else (1 if len(col_names) > 1 else 0)
                            sel_m_col = st.selectbox(f"Measurement Column [{uploaded_file.name}]", col_names, index=m_col_idx, key=f"m_col_{idx}")
                        with c3:
                            sel_f_unit = st.selectbox("Frequency Unit", ["Hz", "kHz", "MHz"], index=0, key=f"f_unit_{idx}")
                        with c4:
                            sel_m_unit = st.selectbox("Measurement Unit", ["dBµA/m", "dBm", "dBµV", "V", "mV", "a.u."], index=0, key=f"m_unit_{idx}")

                        trace, report = load_csv_trace(
                            file_bytes,
                            freq_col=sel_f_col,
                            meas_col=sel_m_col,
                            trace_id=f"csv_{idx+1}_{uploaded_file.name}",
                            trace_name=f"Trace {idx+1} ({uploaded_file.name})",
                            source_filename=uploaded_file.name,
                            frequency_unit=sel_f_unit,
                            measurement_unit=sel_m_unit,
                        )

                        # Validation Metrics Row
                        r1, r2, r3, r4, r5 = st.columns(5)
                        with r1:
                            st.metric("Total Rows", report["total_rows"])
                        with r2:
                            st.metric("Valid Points", report["valid_rows"])
                        with r3:
                            st.metric("Dropped Rows", report["dropped_rows"])
                        with r4:
                            st.metric("Frequency Span", f"{report['f_min']:.1f} - {report['f_max']:.1f} {sel_f_unit}")
                        with r5:
                            st.metric("Duplicates Resolved", report["duplicates_resolved"])

                        if report["dropped_rows"] > 0:
                            st.caption(f"Row Drop Breakdown: {report['nan_freq_rows']} invalid freq, {report['nan_meas_rows']} invalid meas, {report['non_positive_freq_rows']} non-positive freq.")

                        st.session_state.imported_traces[uploaded_file.name] = trace

                        # Mini plot preview
                        fig_mini, ax_mini = plt.subplots(figsize=(8, 1.8), facecolor="#161922")
                        ax_mini.set_facecolor("#161922")
                        ax_mini.set_xscale("log")
                        ax_mini.plot(trace.frequencies, trace.measurements, color="#00e5ff", linewidth=1.0)
                        ax_mini.grid(True, linestyle=":", color="#252936", linewidth=0.5)
                        ax_mini.tick_params(colors="#c8ccd4", labelsize=7)
                        for sp in ax_mini.spines.values():
                            sp.set_color("#343946")
                        st.pyplot(fig_mini)

                    except Exception as exc:
                        st.error(f"Error parsing '{uploaded_file.name}': {str(exc)}")

        if st.session_state.imported_traces:
            st.markdown(f"**Loaded Traces ({len(st.session_state.imported_traces)})**")
            inv_rows = []
            for name, tr in st.session_state.imported_traces.items():
                inv_rows.append({
                    "Source File": tr.source_filename,
                    "Identifier": tr.name,
                    "Points": len(tr.frequencies),
                    "Frequency Span": f"{tr.frequencies[0]:.1f} – {tr.frequencies[-1]:.1f} {tr.frequency_unit}",
                    "Unit": tr.measurement_unit,
                    "Status": tr.review_status,
                })
            st.dataframe(pd.DataFrame(inv_rows), hide_index=True)
            if st.button("Clear Loaded Measurements"):
                st.session_state.imported_traces = {}
                st.session_state.app_mode = "demo"
                st.rerun()

    # --- Screenshot Ingestion ---
    with ingest_tabs[1]:
        st.caption("Extract provisional traces from oscilloscope screenshots. Data remains marked provisional until verified with manual axis calibration.")
        uploaded_img = st.file_uploader("Upload Oscilloscope Screenshot", type=["png", "jpg", "jpeg"], key="screenshot_uploader")

        if uploaded_img is not None:
            try:
                img_array, pil_img = load_and_validate_image(uploaded_img.getvalue(), filename=uploaded_img.name)

                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(pil_img, caption=f"Original: {uploaded_img.name}", width="stretch")

                default_bounds = detect_plot_bounds(img_array)
                b_c1, b_c2, b_c3, b_c4 = st.columns(4)
                with b_c1:
                    x_min = st.number_input("X Min (px)", value=default_bounds[0], min_value=0, max_value=img_array.shape[1])
                with b_c2:
                    y_min = st.number_input("Y Min (px)", value=default_bounds[1], min_value=0, max_value=img_array.shape[0])
                with b_c3:
                    x_max = st.number_input("X Max (px)", value=default_bounds[2], min_value=0, max_value=img_array.shape[1])
                with b_c4:
                    y_max = st.number_input("Y Max (px)", value=default_bounds[3], min_value=0, max_value=img_array.shape[0])

                chosen_bounds = (int(x_min), int(y_min), int(x_max), int(y_max))
                target_color = st.selectbox("Trace Color Filter", ["yellow", "cyan", "magenta", "green", "blue", "bright_any"])

                x_px, y_px, conf, mask = extract_trace_pixels(img_array, chosen_bounds, target_color=target_color)

                with col_img2:
                    if len(x_px) > 0:
                        overlay = create_preview_overlay(img_array, chosen_bounds, x_px, y_px)
                        st.image(overlay, caption=f"Extraction Overlay (Confidence: {conf*100:.1f}%)", width="stretch")
                    else:
                        st.warning("No candidate pixels detected in selected bounding area.")

                cal_c1, cal_c2, cal_c3, cal_c4 = st.columns(4)
                with cal_c1:
                    cal_f_min = st.number_input("F Min (Hz)", value=10.0, min_value=0.1)
                with cal_c2:
                    cal_f_max = st.number_input("F Max (Hz)", value=400000.0, min_value=1.0)
                with cal_c3:
                    cal_m_min = st.number_input("Amp Min (dBµA/m)", value=0.0)
                with cal_c4:
                    cal_m_max = st.number_input("Amp Max (dBµA/m)", value=80.0)

                cal_scale_x = st.selectbox("Frequency Axis Scale", ["log", "linear"], index=0, key="cal_scale_x")

                if st.button("Calibrate and Add Trace", disabled=(len(x_px) == 0)):
                    extracted_trace = create_trace_from_extracted_pixels(
                        x_pixels=x_px,
                        y_pixels=y_px,
                        bounds=chosen_bounds,
                        f_min=cal_f_min,
                        f_max=cal_f_max,
                        f_scale=cal_scale_x,
                        m_min=cal_m_min,
                        m_max=cal_m_max,
                        m_scale="linear",
                        name=f"Extracted {target_color.capitalize()} ({uploaded_img.name})",
                        source_filename=uploaded_img.name,
                        confidence=conf,
                        is_calibrated=True,
                    )
                    st.session_state.imported_traces[f"screenshot_{uploaded_img.name}"] = extracted_trace
                    st.session_state.app_mode = "measurement"
                    st.rerun()

            except Exception as exc:
                st.error(f"Image processing error: {str(exc)}")


# =========================================================
# SECTION 3: Review and Compare
# =========================================================
with tab_compare:
    if st.session_state.app_mode == "measurement" and st.session_state.imported_traces:
        available_traces = list(st.session_state.imported_traces.values())
        default_selected = [t.name for t in available_traces[: min(3, len(available_traces))]]
    else:
        available_traces = list(demo_fft_traces)
        default_selected = [t.name for t in available_traces[: min(3, len(available_traces))]]

    trace_map = {t.name: t for t in available_traces}

    sel_col1, sel_col2 = st.columns([3, 1])
    with sel_col1:
        chosen_names = st.multiselect(
            "Select Traces for Comparison",
            options=list(trace_map.keys()),
            default=default_selected,
        )
    with sel_col2:
        theme_choice = st.radio("Plot Theme", ["Oscilloscope Dark", "Publication Light"], horizontal=True)

    selected_traces = [trace_map[name] for name in chosen_names if name in trace_map]

    if selected_traces:
        # Scientific Compatibility Check
        compat_report = analyze_trace_compatibility(selected_traces)

        if not compat_report.is_unit_compatible:
            st.error(f"Incompatible Units Detected: {', '.join(compat_report.units_found)}. Traces have disparate units and cannot be compared on a single vertical scale.")
        elif not compat_report.has_frequency_overlap:
            st.warning("Zero Frequency Overlap: Selected traces span mutually exclusive frequency intervals.")
        elif compat_report.warnings:
            for w in compat_report.warnings:
                st.caption(f"Notice: {w}")
        else:
            ov = compat_report.overlap_range
            st.caption(f"Compatible units ({compat_report.units_found[0]}) with frequency overlap [{ov[0]:.1f} Hz – {ov[1]:.1f} Hz].")

        # Overlaid plot
        fig_overlay = render_spectra_overlay(
            traces=selected_traces,
            selected_indices=list(range(len(selected_traces))),
            f_min=st.session_state.f_min,
            f_max=st.session_state.f_max,
            scale_type=st.session_state.f_scale,
            peak_results={t.trace_id: detect_trace_peaks(t, prominence=st.session_state.peak_prominence) for t in selected_traces},
            show_peaks=True,
            dark_mode=(theme_choice == "Oscilloscope Dark"),
        )
        st.pyplot(fig_overlay)

        # Spectral Peak Tables
        st.markdown("**Spectral Peak Analysis**")
        p_tabs = st.tabs([t.name for t in selected_traces])
        for idx, t in enumerate(selected_traces):
            with p_tabs[idx]:
                p_res = detect_trace_peaks(t, prominence=st.session_state.peak_prominence)
                if len(p_res.frequencies) > 0:
                    st.dataframe(pd.DataFrame(p_res.to_records(f_unit=t.frequency_unit, m_unit=t.measurement_unit)), hide_index=True)
                else:
                    st.caption("No peaks detected above sensitivity threshold.")
    else:
        st.caption("Select at least one trace to display.")


# =========================================================
# SECTION 4: EMI Spectrum
# =========================================================
with tab_emi:
    if st.session_state.app_mode == "measurement" and st.session_state.imported_traces:
        meas_traces = list(st.session_state.imported_traces.values())
        trace_selector = st.selectbox(
            "Primary Trace for Publication Plot",
            options=[t.name for t in meas_traces] + (["Multi-Trace Overlay"] if len(meas_traces) > 1 else []),
            index=0,
        )

        if trace_selector == "Multi-Trace Overlay":
            st.caption("Displaying multiple channels without automatic combination. Combination formula is withheld pending verified sensor coupling models.")
            emi_target = meas_traces
            primary_tr = meas_traces[0]
            sim_flag = False
        else:
            primary_tr = next(t for t in meas_traces if t.name == trace_selector)
            emi_target = primary_tr
            sim_flag = (primary_tr.source_type == "synthetic")
    else:
        emi_target = demo_combined_trace
        primary_tr = demo_combined_trace
        sim_flag = True

    emi_peaks = detect_trace_peaks(primary_tr, prominence=st.session_state.peak_prominence)
    fig_emi = render_final_emi_spectrum(
        trace_or_traces=emi_target,
        peaks=emi_peaks,
        f_min=st.session_state.f_min,
        f_max=st.session_state.f_max,
        show_reference_line=st.session_state.ref_line_enabled,
        reference_line_db=st.session_state.ref_line_db,
        reference_line_label=st.session_state.ref_line_label,
        show_peak_annotations=True,
        is_simulated=sim_flag,
    )
    st.pyplot(fig_emi)

    # In-screen quick export
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        png_data = figure_to_png_bytes(fig_emi, dpi=300)
        st.download_button(
            "Download Plot (PNG, 300 DPI)",
            data=png_data,
            file_name=f"vga_emi_spectrum_{primary_tr.name.replace(' ', '_').lower()}.png",
            mime="image/png",
            use_container_width=True,
        )
    with q_col2:
        csv_data = trace_to_csv_bytes(primary_tr)
        st.download_button(
            "Download Spectrum Data (CSV)",
            data=csv_data,
            file_name=f"vga_spectrum_{primary_tr.name.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with q_col3:
        if len(emi_peaks.frequencies) > 0:
            peak_csv = peaks_to_csv_bytes(emi_peaks, freq_unit=primary_tr.frequency_unit, meas_unit=primary_tr.measurement_unit)
            st.download_button(
                "Download Peak Table (CSV)",
                data=peak_csv,
                file_name=f"vga_peaks_{primary_tr.name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if len(emi_peaks.frequencies) > 0:
        st.markdown("**Dominant Spectral Peaks**")
        st.dataframe(pd.DataFrame(emi_peaks.to_records(f_unit=primary_tr.frequency_unit, m_unit=primary_tr.measurement_unit)), hide_index=True)


# =========================================================
# SECTION 5: Export Results
# =========================================================
with tab_export:
    ex_col1, ex_col2 = st.columns(2)

    with ex_col1:
        st.markdown("**Figure Export**")
        st.caption("Generate high-resolution PNG plots suitable for technical reports.")

        export_fig_choice = st.selectbox("Select Figure", ["Publication EMI Spectrum", "Oscilloscope 6-Panel Dashboard"])
        if export_fig_choice == "Publication EMI Spectrum":
            target_fig = fig_emi
            exp_name = "publication_emi_spectrum.png"
        else:
            target_fig = fig_dash
            exp_name = "oscilloscope_dashboard.png"

        fig_bytes = figure_to_png_bytes(target_fig, dpi=300)
        st.download_button(
            label=f"Download {export_fig_choice} (300 DPI PNG)",
            data=fig_bytes,
            file_name=exp_name,
            mime="image/png",
            use_container_width=True,
        )

    with ex_col2:
        st.markdown("**Numerical Data Export**")
        st.caption("Export tabular spectrum data with physical units and metadata.")

        active_export_traces = list(st.session_state.imported_traces.values()) if st.session_state.app_mode == "measurement" and st.session_state.imported_traces else demo_fft_traces
        selected_export_trace_name = st.selectbox("Select Spectrum", [t.name for t in active_export_traces])
        selected_export_trace = next(t for t in active_export_traces if t.name == selected_export_trace_name)

        csv_bytes = trace_to_csv_bytes(selected_export_trace)
        st.download_button(
            label=f"Download CSV for '{selected_export_trace.name}'",
            data=csv_bytes,
            file_name=f"{selected_export_trace.name.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("**Multi-Trace Comparison Export**")
    if active_export_traces:
        comp_csv = comparison_to_csv_bytes(active_export_traces)
        st.download_button(
            label="Download Multi-Trace Comparison Summary (CSV)",
            data=comp_csv,
            file_name="vga_traces_comparison_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# SECTION 6: Sessions
# =========================================================
with tab_sessions:
    sess_c1, sess_c2 = st.columns(2)

    with sess_c1:
        st.markdown("**Save Session**")
        st.caption("Package analysis settings, loaded measurements, and processing metadata into a local bundle.")

        current_traces = list(st.session_state.imported_traces.values()) if st.session_state.imported_traces else demo_fft_traces

        session_settings = {
            "app_mode": st.session_state.app_mode,
            "f_min": st.session_state.f_min,
            "f_max": st.session_state.f_max,
            "f_scale": st.session_state.f_scale,
            "time_window_us": st.session_state.time_window_us,
            "num_samples": st.session_state.num_samples,
            "peak_prominence": st.session_state.peak_prominence,
            "ref_line_enabled": st.session_state.ref_line_enabled,
            "ref_line_db": st.session_state.ref_line_db,
            "ref_line_label": st.session_state.ref_line_label,
            "seed": st.session_state.seed,
        }

        bundle_bytes = save_session_to_bytes(
            settings=session_settings,
            traces=current_traces,
            time_traces=demo_time_traces,
        )

        st.download_button(
            label="Download Session Bundle (.vga_session)",
            data=bundle_bytes,
            file_name=f"vga_session_{st.session_state.app_mode}.vga_session",
            mime="application/zip",
            use_container_width=True,
        )

    with sess_c2:
        st.markdown("**Restore Session**")
        st.caption("Upload a saved .vga_session bundle to restore exact analysis state.")

        uploaded_session = st.file_uploader("Select .vga_session File", type=["vga_session", "zip"], key="sess_uploader")
        if uploaded_session is not None:
            if st.button("Restore Session", use_container_width=True):
                try:
                    s_settings, s_traces, s_waves = load_session_from_bytes(uploaded_session.getvalue())
                    st.session_state.app_mode = s_settings.get("app_mode", "measurement")
                    st.session_state.f_min = s_settings.get("f_min", 10.0)
                    st.session_state.f_max = s_settings.get("f_max", 400000.0)
                    st.session_state.f_scale = s_settings.get("f_scale", "log")
                    st.session_state.time_window_us = s_settings.get("time_window_us", 100.0)
                    st.session_state.peak_prominence = s_settings.get("peak_prominence", 4.0)
                    st.session_state.ref_line_enabled = s_settings.get("ref_line_enabled", False)
                    st.session_state.ref_line_db = s_settings.get("ref_line_db", 50.0)
                    st.session_state.ref_line_label = s_settings.get("ref_line_label", "Reference Level")

                    st.session_state.imported_traces = {t.name: t for t in s_traces if t.source_type != "synthetic"}
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to restore session: {str(exc)}")

    st.markdown("---")
    st.markdown("**Trace Provenance & Processing History**")
    all_active = list(st.session_state.imported_traces.values()) if st.session_state.imported_traces else demo_fft_traces
    for tr in all_active:
        with st.expander(f"Trace: {tr.name} ({tr.source_type})", expanded=False):
            st.write(f"- **Identifier**: `{tr.trace_id}`")
            st.write(f"- **Source File**: `{tr.source_filename or 'Synthetic Generator'}`")
            st.write(f"- **Review Status**: `{tr.review_status}`")
            st.write(f"- **Point Count**: {len(tr.frequencies)}")
            st.write(f"- **Physical Units**: `{tr.frequency_unit}` / `{tr.measurement_unit}`")
            st.write("- **Processing Log**:")
            for h in tr.processing_history:
                st.write(f"  - {h}")
