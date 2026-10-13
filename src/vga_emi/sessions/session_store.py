"""Safe session storage using JSON metadata and NumPy NPZ arrays (Zero Pickle)."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import zipfile
import numpy as np

from vga_emi import __version__
from vga_emi.models import MeasurementTrace, TimeDomainTrace


def save_session_to_bytes(
    settings: Dict[str, Any],
    traces: List[MeasurementTrace],
    time_traces: Optional[List[TimeDomainTrace]] = None,
) -> bytes:
    """Pack analysis session settings, metadata, and numerical arrays into a zip archive.

    Uses JSON for settings/metadata and NumPy NPZ for numerical arrays.
    Guarantees no unsafe pickle serialization.

    Args:
        settings: UI controls, frequency ranges, calibration, reference line dict.
        traces: List of MeasurementTrace objects.
        time_traces: Optional list of TimeDomainTrace objects.

    Returns:
        Bytes containing the zipped session bundle.
    """
    time_traces = time_traces or []
    zip_buffer = io.BytesIO()

    # Prepare JSON metadata
    metadata = {
        "version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "traces_meta": [t.to_dict() for t in traces],
        "time_traces_meta": [
            {
                "channel_id": wt.channel_id,
                "name": wt.name,
                "time_unit": wt.time_unit,
                "voltage_unit": wt.voltage_unit,
                "source_type": wt.source_type,
                "is_simulated": wt.is_simulated,
                "metadata": wt.metadata,
                "num_samples": len(wt.time),
            }
            for wt in time_traces
        ],
    }

    # Prepare NPZ arrays dictionary
    arrays_dict: Dict[str, np.ndarray] = {}
    for idx, t in enumerate(traces):
        arrays_dict[f"trace_{idx}_freqs"] = t.frequencies
        arrays_dict[f"trace_{idx}_meas"] = t.measurements

    for idx, wt in enumerate(time_traces):
        arrays_dict[f"time_{idx}_t"] = wt.time
        arrays_dict[f"time_{idx}_v"] = wt.voltage

    # Save NPZ into buffer
    npz_buffer = io.BytesIO()
    np.savez(npz_buffer, **arrays_dict)
    npz_bytes = npz_buffer.getvalue()

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("session.json", json.dumps(metadata, indent=2))
        zf.writestr("arrays.npz", npz_bytes)

    return zip_buffer.getvalue()


def load_session_from_bytes(
    data: bytes,
) -> Tuple[Dict[str, Any], List[MeasurementTrace], List[TimeDomainTrace]]:
    """Unpack analysis session bundle from bytes.

    Uses json.loads and np.load(allow_pickle=False) for total safety.

    Args:
        data: Zip archive bytes.

    Returns:
        Tuple of (settings_dict, list of MeasurementTrace, list of TimeDomainTrace).
    """
    zip_buffer = io.BytesIO(data)

    with zipfile.ZipFile(zip_buffer, mode="r") as zf:
        if "session.json" not in zf.namelist() or "arrays.npz" not in zf.namelist():
            raise ValueError("Invalid session bundle: Missing session.json or arrays.npz.")

        meta_str = zf.read("session.json").decode("utf-8")
        metadata = json.loads(meta_str)

        npz_bytes = zf.read("arrays.npz")
        npz_buffer = io.BytesIO(npz_bytes)
        npz_data = np.load(npz_buffer, allow_pickle=False)

        traces: List[MeasurementTrace] = []
        for idx, t_meta in enumerate(metadata.get("traces_meta", [])):
            freq_key = f"trace_{idx}_freqs"
            meas_key = f"trace_{idx}_meas"
            if freq_key not in npz_data or meas_key not in npz_data:
                raise ValueError(f"Corrupted session: missing array keys for trace index {idx}")

            freqs = npz_data[freq_key]
            meas = npz_data[meas_key]

            trace = MeasurementTrace(
                trace_id=t_meta["trace_id"],
                name=t_meta["name"],
                frequencies=freqs,
                measurements=meas,
                frequency_unit=t_meta.get("frequency_unit", "Hz"),
                measurement_unit=t_meta.get("measurement_unit", "dBµA/m"),
                source_type=t_meta.get("source_type", "synthetic"),
                source_filename=t_meta.get("source_filename"),
                axis_scales=t_meta.get("axis_scales", {"x": "log", "y": "linear"}),
                calibration_settings=t_meta.get("calibration_settings", {}),
                extraction_confidence=t_meta.get("extraction_confidence"),
                processing_history=t_meta.get("processing_history", []),
                review_status=t_meta.get("review_status", "provisional"),
                metadata=t_meta.get("metadata", {}),
            )
            traces.append(trace)

        time_traces: List[TimeDomainTrace] = []
        for idx, wt_meta in enumerate(metadata.get("time_traces_meta", [])):
            t_key = f"time_{idx}_t"
            v_key = f"time_{idx}_v"
            if t_key in npz_data and v_key in npz_data:
                t_arr = npz_data[t_key]
                v_arr = npz_data[v_key]
                time_trace = TimeDomainTrace(
                    channel_id=wt_meta["channel_id"],
                    name=wt_meta["name"],
                    time=t_arr,
                    voltage=v_arr,
                    time_unit=wt_meta.get("time_unit", "µs"),
                    voltage_unit=wt_meta.get("voltage_unit", "V"),
                    source_type=wt_meta.get("source_type", "synthetic"),
                    is_simulated=wt_meta.get("is_simulated", True),
                    metadata=wt_meta.get("metadata", {}),
                )
                time_traces.append(time_trace)

        return metadata.get("settings", {}), traces, time_traces


def save_session_to_file(
    file_path: Union[str, Path],
    settings: Dict[str, Any],
    traces: List[MeasurementTrace],
    time_traces: Optional[List[TimeDomainTrace]] = None,
) -> Path:
    """Save session to a local file path."""
    p = Path(file_path)
    bundle_bytes = save_session_to_bytes(settings, traces, time_traces)
    p.write_bytes(bundle_bytes)
    return p


def load_session_from_file(
    file_path: Union[str, Path],
) -> Tuple[Dict[str, Any], List[MeasurementTrace], List[TimeDomainTrace]]:
    """Load session from a local file path."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Session file not found: {p}")
    return load_session_from_bytes(p.read_bytes())
