"""Data models for VGA EMI Analyzer.

Encapsulates measurement traces, time-domain waveforms, and peak detection
results with strict scientific validation and metadata tracking.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional
import numpy as np


@dataclasses.dataclass
class MeasurementTrace:
    """Represents a frequency-domain measurement or simulation trace.

    Adheres to strict validation for finite values, length matching,
    monotonic frequencies, and positive values for logarithmic scales.
    """

    trace_id: str
    name: str
    frequencies: np.ndarray
    measurements: np.ndarray
    frequency_unit: str = "Hz"
    measurement_unit: str = "dBµA/m"
    source_type: str = "synthetic"  # 'synthetic' | 'screenshot_derived' | 'imported_csv' | 'demo_combination'
    source_filename: Optional[str] = None
    axis_scales: Dict[str, str] = dataclasses.field(
        default_factory=lambda: {"x": "log", "y": "linear"}
    )
    calibration_settings: Dict[str, Any] = dataclasses.field(default_factory=dict)
    extraction_confidence: Optional[float] = None
    processing_history: List[str] = dataclasses.field(default_factory=list)
    review_status: str = "provisional"  # 'unreviewed' | 'provisional' | 'calibrated' | 'verified'
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce lists/sequences into NumPy float arrays and run validation."""
        if not isinstance(self.frequencies, np.ndarray):
            self.frequencies = np.asarray(self.frequencies, dtype=float)
        else:
            self.frequencies = self.frequencies.astype(float, copy=False)

        if not isinstance(self.measurements, np.ndarray):
            self.measurements = np.asarray(self.measurements, dtype=float)
        else:
            self.measurements = self.measurements.astype(float, copy=False)

        self.validate()

    def validate(self) -> None:
        """Validate trace data integrity.

        Raises:
            ValueError: If lengths do not match, values are non-finite,
                        frequencies are not non-decreasing, or logarithmic frequencies
                        contain non-positive values.
        """
        if len(self.frequencies) == 0:
            raise ValueError(f"Trace '{self.trace_id}' frequencies array cannot be empty.")

        if len(self.frequencies) != len(self.measurements):
            raise ValueError(
                f"Array length mismatch for trace '{self.trace_id}': "
                f"frequencies has {len(self.frequencies)} items, "
                f"measurements has {len(self.measurements)} items."
            )

        if not np.all(np.isfinite(self.frequencies)):
            raise ValueError(f"Trace '{self.trace_id}' frequencies contain NaN or infinite values.")

        if not np.all(np.isfinite(self.measurements)):
            raise ValueError(f"Trace '{self.trace_id}' measurements contain NaN or infinite values.")

        # Check frequency monotonicity (strictly increasing preferred, non-decreasing required)
        if len(self.frequencies) > 1:
            diffs = np.diff(self.frequencies)
            if np.any(diffs <= 0):
                raise ValueError(
                    f"Trace '{self.trace_id}' frequencies must be strictly increasing. "
                    f"Found non-increasing step at indices: {np.where(diffs <= 0)[0]}."
                )

        # Check logarithmic axis frequency positivity
        if self.axis_scales.get("x") == "log":
            if np.any(self.frequencies <= 0):
                min_f = float(np.min(self.frequencies))
                raise ValueError(
                    f"Trace '{self.trace_id}' has logarithmic frequency scale, "
                    f"but contains non-positive frequencies (minimum: {min_f})."
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to a JSON-serializable dictionary (excluding raw arrays)."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "frequency_unit": self.frequency_unit,
            "measurement_unit": self.measurement_unit,
            "source_type": self.source_type,
            "source_filename": self.source_filename,
            "axis_scales": dict(self.axis_scales),
            "calibration_settings": dict(self.calibration_settings),
            "extraction_confidence": self.extraction_confidence,
            "processing_history": list(self.processing_history),
            "review_status": self.review_status,
            "metadata": dict(self.metadata),
            "num_points": len(self.frequencies),
            "f_min": float(self.frequencies[0]) if len(self.frequencies) > 0 else None,
            "f_max": float(self.frequencies[-1]) if len(self.frequencies) > 0 else None,
        }

    def copy(self) -> MeasurementTrace:
        """Create an independent copy of this trace."""
        return MeasurementTrace(
            trace_id=self.trace_id,
            name=self.name,
            frequencies=np.copy(self.frequencies),
            measurements=np.copy(self.measurements),
            frequency_unit=self.frequency_unit,
            measurement_unit=self.measurement_unit,
            source_type=self.source_type,
            source_filename=self.source_filename,
            axis_scales=dict(self.axis_scales),
            calibration_settings=dict(self.calibration_settings),
            extraction_confidence=self.extraction_confidence,
            processing_history=list(self.processing_history),
            review_status=self.review_status,
            metadata=dict(self.metadata),
        )


@dataclasses.dataclass
class TimeDomainTrace:
    """Represents a time-domain voltage channel waveform."""

    channel_id: str
    name: str
    time: np.ndarray
    voltage: np.ndarray
    time_unit: str = "µs"
    voltage_unit: str = "V"
    source_type: str = "synthetic"
    is_simulated: bool = True
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.time, np.ndarray):
            self.time = np.asarray(self.time, dtype=float)
        if not isinstance(self.voltage, np.ndarray):
            self.voltage = np.asarray(self.voltage, dtype=float)

        if len(self.time) != len(self.voltage):
            raise ValueError(
                f"Channel '{self.channel_id}' time and voltage arrays must have identical length: "
                f"time={len(self.time)}, voltage={len(self.voltage)}"
            )

        if not np.all(np.isfinite(self.time)) or not np.all(np.isfinite(self.voltage)):
            raise ValueError(f"Channel '{self.channel_id}' contains NaN or infinite values.")


@dataclasses.dataclass
class PeakResult:
    """Represents detected peaks from a frequency-domain spectrum."""

    frequencies: np.ndarray
    amplitudes: np.ndarray
    indices: np.ndarray

    def to_records(self, f_unit: str = "Hz", m_unit: str = "dBµA/m") -> List[Dict[str, Any]]:
        """Convert detected peaks into tabular records."""
        records = []
        for i in range(len(self.frequencies)):
            f = float(self.frequencies[i])
            f_display = f"{f:.1f}" if f < 1000 else (f"{f/1000:.2f} k" if f < 1e6 else f"{f/1e6:.3f} M")
            records.append({
                "Rank": i + 1,
                f"Frequency ({f_unit})": f,
                "Frequency Display": f"{f_display}{f_unit}",
                f"Amplitude ({m_unit})": round(float(self.amplitudes[i]), 2),
            })
        return records
