"""CSV ingestion and validation module for spectrum data."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from vga_emi.models import MeasurementTrace


def inspect_csv(
    source: Union[str, Path, io.StringIO, io.BytesIO, bytes],
) -> pd.DataFrame:
    """Read CSV into a DataFrame for column inspection.

    Args:
        source: File path, StringIO, BytesIO, or raw bytes.

    Returns:
        pd.DataFrame.
    """
    try:
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        df = pd.read_csv(source, comment="#")
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file: {str(exc)}") from exc

    if df.empty:
        raise ValueError("Uploaded CSV file is empty or contains no readable data.")

    return df


def guess_csv_columns(columns: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Guess probable frequency and measurement column names from a list of column names."""
    freq_candidates = [
        "frequency_hz", "frequency (hz)", "frequency", "freq", "freq_hz", "f_hz", "f", "hz",
        "freq(hz)", "frequency(mhz)", "frequency_mhz", "freq_khz", "frequency_khz",
    ]
    meas_candidates = [
        "amplitude_dbua_m", "amplitude (dbµa/m)", "amplitude (dbua/m)", "amplitude",
        "magnitude", "dbt", "dbua_m", "dbµa/m", "dbm", "dbuv", "voltage", "voltage_v", "meas", "measurement", "value",
    ]

    freq_col = None
    meas_col = None

    col_lower_map = {c.strip().lower(): c for c in columns}

    # Match frequency
    for candidate in freq_candidates:
        if candidate in col_lower_map:
            freq_col = col_lower_map[candidate]
            break
    if not freq_col:
        for c, orig in col_lower_map.items():
            if "freq" in c or "hz" in c:
                freq_col = orig
                break

    # Match measurement
    for candidate in meas_candidates:
        if candidate in col_lower_map:
            meas_col = col_lower_map[candidate]
            break
    if not meas_col:
        for c, orig in col_lower_map.items():
            if ("amp" in c or "mag" in c or "db" in c or "volt" in c or "val" in c) and orig != freq_col:
                meas_col = orig
                break

    # Fallbacks
    if not freq_col and len(columns) > 0:
        freq_col = columns[0]
    if not meas_col and len(columns) > 1:
        meas_col = columns[1] if columns[1] != freq_col else (columns[0] if len(columns) == 1 else None)

    return freq_col, meas_col


def load_csv_trace(
    source: Union[str, Path, io.StringIO, io.BytesIO, bytes],
    freq_col: str,
    meas_col: str,
    trace_id: str = "imported_csv_1",
    trace_name: str = "Imported Measurement",
    source_filename: Optional[str] = None,
    frequency_unit: str = "Hz",
    measurement_unit: str = "dBµA/m",
    axis_scales: Optional[Dict[str, str]] = None,
) -> Tuple[MeasurementTrace, Dict[str, Any]]:
    """Parse, clean, and validate a MeasurementTrace from a CSV source.

    Performs detailed data hygiene:
    - Drops non-numeric, NaN, and infinite rows.
    - Drops non-positive frequency entries if scale is logarithmic.
    - Sorts frequencies strictly monotonically.
    - Resolves duplicate frequency points by averaging.
    - Produces an audit report with row breakdown and span details.

    Args:
        source: CSV file path, StringIO, BytesIO, or raw bytes.
        freq_col: Name of column containing frequency values.
        meas_col: Name of column containing amplitude/measurement values.
        trace_id: Unique trace identifier.
        trace_name: Display name.
        source_filename: Name of original file.
        frequency_unit: Unit for frequency ("Hz", "kHz", "MHz").
        measurement_unit: Unit for measurement ("dBµA/m", "dBm", "V", etc.).
        axis_scales: Optional dictionary e.g. {"x": "log", "y": "linear"}.

    Returns:
        Tuple of (MeasurementTrace, report_dict).

    Raises:
        ValueError: If specified columns are missing or no valid numeric rows exist.
    """
    df = inspect_csv(source)

    if freq_col not in df.columns:
        raise ValueError(f"Frequency column '{freq_col}' not found in CSV. Available columns: {list(df.columns)}")
    if meas_col not in df.columns:
        raise ValueError(f"Measurement column '{meas_col}' not found in CSV. Available columns: {list(df.columns)}")

    total_rows = len(df)

    # Coerce to numeric, converting unparseable entries to NaN
    freq_series = pd.to_numeric(df[freq_col], errors="coerce")
    meas_series = pd.to_numeric(df[meas_col], errors="coerce")

    # Audit breakdown
    nan_freq_count = int(freq_series.isna().sum())
    nan_meas_count = int(meas_series.isna().sum())
    non_pos_freq_count = int((freq_series <= 0).sum())

    # Build valid boolean mask
    valid_mask = freq_series.notna() & meas_series.notna() & np.isfinite(freq_series) & np.isfinite(meas_series)

    # If log x axis, frequencies must be strictly positive
    scales = axis_scales or {"x": "log", "y": "linear"}
    if scales.get("x") == "log":
        valid_mask = valid_mask & (freq_series > 0)

    clean_freqs = freq_series[valid_mask].to_numpy(dtype=float)
    clean_meas = meas_series[valid_mask].to_numpy(dtype=float)

    dropped_rows = total_rows - len(clean_freqs)

    if len(clean_freqs) == 0:
        raise ValueError(
            f"No valid numeric data found in columns '{freq_col}' and '{meas_col}'. "
            f"All {total_rows} rows were dropped due to non-numeric or non-positive values."
        )

    # Sort by frequency strictly increasing
    sort_idx = np.argsort(clean_freqs)
    clean_freqs = clean_freqs[sort_idx]
    clean_meas = clean_meas[sort_idx]

    # Deduplicate duplicate frequency entries by averaging measurements
    unique_freqs, unique_indices, counts = np.unique(clean_freqs, return_index=True, return_counts=True)
    duplicates_resolved = int(np.sum(counts > 1))
    if len(unique_freqs) < len(clean_freqs):
        df_clean = pd.DataFrame({"f": clean_freqs, "m": clean_meas})
        grouped = df_clean.groupby("f", as_index=False).mean()
        clean_freqs = grouped["f"].to_numpy()
        clean_meas = grouped["m"].to_numpy()

    # Resolution step
    delta_f = np.diff(clean_freqs) if len(clean_freqs) > 1 else np.array([0.0])
    min_delta = float(np.min(delta_f)) if len(delta_f) > 0 else 0.0
    median_delta = float(np.median(delta_f)) if len(delta_f) > 0 else 0.0

    report = {
        "total_rows": total_rows,
        "valid_rows": len(clean_freqs),
        "dropped_rows": dropped_rows,
        "nan_freq_rows": nan_freq_count,
        "nan_meas_rows": nan_meas_count,
        "non_positive_freq_rows": non_pos_freq_count,
        "duplicates_resolved": duplicates_resolved,
        "f_min": float(clean_freqs[0]),
        "f_max": float(clean_freqs[-1]),
        "f_span": float(clean_freqs[-1] - clean_freqs[0]),
        "m_min": float(np.min(clean_meas)),
        "m_max": float(np.max(clean_meas)),
        "m_span": float(np.max(clean_meas) - np.min(clean_meas)),
        "min_delta_f": min_delta,
        "median_delta_f": median_delta,
        "frequency_unit": frequency_unit,
        "measurement_unit": measurement_unit,
        "source_filename": source_filename,
    }

    trace = MeasurementTrace(
        trace_id=trace_id,
        name=trace_name,
        frequencies=clean_freqs,
        measurements=clean_meas,
        frequency_unit=frequency_unit,
        measurement_unit=measurement_unit,
        source_type="imported_csv",
        source_filename=source_filename,
        axis_scales=scales,
        calibration_settings={"calibrated": True, "source": "csv_direct", "unit": measurement_unit},
        extraction_confidence=1.0,
        processing_history=[
            f"Imported from CSV '{source_filename or 'uploaded'}'",
            f"Mapped columns: Freq='{freq_col}' ({frequency_unit}), Amp='{meas_col}' ({measurement_unit})",
            f"Validation: {len(clean_freqs)} valid points ({dropped_rows} dropped, {duplicates_resolved} duplicates averaged)",
        ],
        review_status="verified",
        metadata={
            "source_type": "csv",
            "report": report,
            "raw_filename": source_filename,
        },
    )

    return trace, report
