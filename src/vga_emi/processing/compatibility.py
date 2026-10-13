"""Trace compatibility and scientific validation analyzer."""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple
import numpy as np

from vga_emi.models import MeasurementTrace


@dataclasses.dataclass
class TraceCompatibilityReport:
    """Detailed scientific assessment of multi-trace comparison and combination validity."""

    is_unit_compatible: bool
    units_found: List[str]
    has_frequency_overlap: bool
    overlap_range: Optional[Tuple[float, float]]
    trace_ranges: Dict[str, Tuple[float, float]]
    warnings: List[str]
    can_combine_scientifically: bool
    combination_blockers: List[str]


def analyze_trace_compatibility(traces: List[MeasurementTrace]) -> TraceCompatibilityReport:
    """Analyze a collection of measurement traces for unit compatibility and frequency overlap.

    Args:
        traces: List of MeasurementTrace objects.

    Returns:
        TraceCompatibilityReport with detailed findings and warnings.
    """
    if not traces:
        return TraceCompatibilityReport(
            is_unit_compatible=True,
            units_found=[],
            has_frequency_overlap=True,
            overlap_range=None,
            trace_ranges={},
            warnings=["No traces selected for compatibility analysis."],
            can_combine_scientifically=False,
            combination_blockers=["No input traces available."],
        )

    warnings: List[str] = []
    blockers: List[str] = []

    # 1. Analyze Measurement Units
    units = list(dict.fromkeys(t.measurement_unit for t in traces))
    is_unit_compatible = len(units) <= 1
    if not is_unit_compatible:
        warnings.append(
            f"Incompatible measurement units detected: {', '.join(units)}. "
            f"Traces with disparate units cannot be overlaid on a single physical scale or combined."
        )
        blockers.append(f"Disparate physical units: {', '.join(units)}")

    # 2. Analyze Frequency Ranges and Overlap
    trace_ranges: Dict[str, Tuple[float, float]] = {}
    mins = []
    maxs = []

    for t in traces:
        t_min = float(np.min(t.frequencies))
        t_max = float(np.max(t.frequencies))
        trace_ranges[t.name] = (t_min, t_max)
        mins.append(t_min)
        maxs.append(t_max)

    overlap_min = max(mins)
    overlap_max = min(maxs)

    if overlap_min < overlap_max:
        has_overlap = True
        overlap_range = (overlap_min, overlap_max)
        overlap_ratio = (overlap_max - overlap_min) / (max(maxs) - min(mins)) if max(maxs) > min(mins) else 1.0
        if overlap_ratio < 0.2:
            warnings.append(
                f"Narrow frequency overlap [{overlap_min:.1f} Hz - {overlap_max:.1f} Hz] "
                f"(covers only {overlap_ratio*100:.1f}% of total span)."
            )
    else:
        has_overlap = False
        overlap_range = None
        warnings.append(
            f"Zero frequency overlap among selected traces. "
            f"Trace ranges: {', '.join(f'{k}: [{v[0]:.1f}-{v[1]:.1f} Hz]' for k, v in trace_ranges.items())}."
        )
        blockers.append("No common frequency overlap interval.")

    # 3. Scientific Calibration & Combination Assessment
    uncalibrated = [t.name for t in traces if t.review_status == "provisional"]
    if uncalibrated:
        warnings.append(
            f"Traces with provisional calibration detected ({', '.join(uncalibrated)}). "
            f"Ensure axes are calibrated to engineering units before drawing EMI compliance conclusions."
        )

    # Scientific combination rule
    can_combine = is_unit_compatible and has_overlap and len(blockers) == 0

    return TraceCompatibilityReport(
        is_unit_compatible=is_unit_compatible,
        units_found=units,
        has_frequency_overlap=has_overlap,
        overlap_range=overlap_range,
        trace_ranges=trace_ranges,
        warnings=warnings,
        can_combine_scientifically=can_combine,
        combination_blockers=blockers,
    )
