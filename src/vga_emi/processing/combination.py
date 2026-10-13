"""Combination and mathematical operations for spectrum traces.

SCIENTIFIC PRINCIPLES & REQUIREMENTS:
1. The exact formula from the original reference experiment is unknown.
2. Direct linear addition of decibel values (dB + dB) is physically invalid.
3. Quantities with incompatible units cannot be combined.
4. Linear Root-Sum-Square (RSS) and square-root operations here are DEMO CALCULATIONS
   strictly marked as illustrative simulations, not verified physical combinations.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from scipy.interpolate import interp1d

from vga_emi.models import MeasurementTrace


def db_to_linear(values_db: np.ndarray, ref: float = 1.0, is_power: bool = False) -> np.ndarray:
    """Convert decibel values to linear scale.

    For field/amplitude quantities (e.g. dBµA/m, dBV): linear = 10^(dB / 20) * ref
    For power quantities (e.g. dBm, dBW): linear = 10^(dB / 10) * ref

    Args:
        values_db: Array of decibel values.
        ref: Reference level in linear units.
        is_power: True if power quantity (10*log10), False for amplitude/field (20*log10).

    Returns:
        Array in linear units.
    """
    divisor = 10.0 if is_power else 20.0
    return ref * (10.0 ** (np.asarray(values_db, dtype=float) / divisor))


def linear_to_db(values_linear: np.ndarray, ref: float = 1.0, is_power: bool = False, eps: float = 1e-12) -> np.ndarray:
    """Convert linear values to decibels with numerical safety clamp.

    Args:
        values_linear: Array of positive linear values.
        ref: Reference level.
        is_power: True for power (10*log10), False for amplitude (20*log10).
        eps: Minimum positive floor to prevent log(0).

    Returns:
        Array in decibels.
    """
    multiplier = 10.0 if is_power else 20.0
    safe_linear = np.maximum(np.asarray(values_linear, dtype=float) / ref, eps)
    return multiplier * np.log10(safe_linear)


def resample_trace_to_grid(
    trace: MeasurementTrace,
    target_frequencies: np.ndarray,
) -> np.ndarray:
    """Interpolate trace measurements onto a common frequency grid."""
    if np.array_equal(trace.frequencies, target_frequencies):
        return np.copy(trace.measurements)

    interpolator = interp1d(
        trace.frequencies,
        trace.measurements,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return interpolator(target_frequencies)


def combine_traces_rss(
    traces: List[MeasurementTrace],
    name: str = "Math 6 — Combined FFT (Demo RSS)",
    treat_as_db: bool = True,
) -> MeasurementTrace:
    """Combine multiple traces using Root-Sum-Square (RSS) illustrative calculation.

    Demonstration Calculation:
    - If input units are decibels (e.g., dBµA/m), values are first converted to
      linear amplitude equivalents: A_lin = 10^(dB / 20).
    - Linear amplitudes are combined via RSS: A_combined = sqrt(sum(A_lin_i^2)).
    - Combined linear amplitude is converted back to decibels: dB_combined = 20 * log10(A_combined).

    DISCLAIMER: This is an illustrative simulation formula. It does NOT claim to reproduce
    the exact formula from the original reference experiment.

    Args:
        traces: List of MeasurementTrace objects (must have compatible units).
        name: Name for the resulting trace.
        treat_as_db: Whether input measurements represent decibels.

    Returns:
        MeasurementTrace representing the combined demo spectrum.
    """
    if not traces:
        raise ValueError("Cannot combine an empty list of traces.")

    if len(traces) == 1:
        res = traces[0].copy()
        res.trace_id = "math_combined_demo"
        res.name = name
        return res

    # Validate unit compatibility
    base_unit = traces[0].measurement_unit
    for t in traces[1:]:
        if t.measurement_unit != base_unit:
            raise ValueError(
                f"Incompatible measurement units cannot be combined: "
                f"'{traces[0].name}' has '{base_unit}', but '{t.name}' has '{t.measurement_unit}'."
            )

    # Use first trace's frequency array as common grid
    target_f = traces[0].frequencies
    aligned_measurements = []

    for t in traces:
        m = resample_trace_to_grid(t, target_f)
        aligned_measurements.append(m)

    aligned_stack = np.stack(aligned_measurements, axis=0)

    if treat_as_db:
        # Step 1: Convert dB to linear amplitude
        linear_stack = db_to_linear(aligned_stack, ref=1.0, is_power=False)
        # Step 2: Root-Sum-Square in linear domain
        combined_linear = np.sqrt(np.sum(linear_stack**2, axis=0))
        # Step 3: Convert back to dB
        combined_result = linear_to_db(combined_linear, ref=1.0, is_power=False)
    else:
        # Direct linear RSS
        combined_result = np.sqrt(np.sum(aligned_stack**2, axis=0))

    history = [
        "DEMO CALCULATION: Illustrative linear Root-Sum-Square (RSS)",
        f"Input traces: {[t.name for t in traces]}",
        f"Conversion: {'dB -> linear -> RSS -> dB' if treat_as_db else 'Linear RSS'}",
        "DISCLAIMER: Synthetic illustrative formula. Not verified physical combination.",
    ]

    return MeasurementTrace(
        trace_id="math_combined_demo",
        name=name,
        frequencies=np.copy(target_f),
        measurements=combined_result,
        frequency_unit=traces[0].frequency_unit,
        measurement_unit=base_unit,
        source_type="demo_combination",
        source_filename="combined_demo.dat",
        axis_scales=dict(traces[0].axis_scales),
        calibration_settings={"mode": "demo_rss", "calibrated": False},
        extraction_confidence=1.0,
        processing_history=history,
        review_status="provisional",
        metadata={
            "is_simulated": True,
            "is_demo_calculation": True,
            "method": "linear_root_sum_square",
            "num_combined": len(traces),
        },
    )


def compute_demo_sqrt(
    trace: MeasurementTrace,
    name: str = "Math 4 — Square-Root Result (Demo Sqrt)",
) -> MeasurementTrace:
    """Compute illustrative square root of trace amplitudes.

    In the reference oscilloscope display, Math 4 represents an illustrative
    square-root mathematical operation.
    For decibels, we demonstrate sqrt on the linear equivalent:
      A_sqrt_db = 20 * log10(sqrt(10^(dB/20))) = dB / 2.
    For linear values: sqrt(A).

    Args:
        trace: Input MeasurementTrace.
        name: Name for the resulting trace.

    Returns:
        MeasurementTrace representing the square-root demo result.
    """
    is_db = "db" in trace.measurement_unit.lower()

    if is_db:
        # In dB domain, taking the square root of the linear amplitude equals dividing dB by 2:
        # 20 * log10(sqrt(A)) = 20 * log10(A^0.5) = 10 * log10(A) = (20 * log10(A)) / 2
        sqrt_meas = trace.measurements / 2.0
        unit = trace.measurement_unit
    else:
        non_neg = np.maximum(trace.measurements, 0.0)
        sqrt_meas = np.sqrt(non_neg)
        unit = f"sqrt({trace.measurement_unit})"

    history = list(trace.processing_history)
    history.append("DEMO CALCULATION: Illustrative square root of amplitude")
    history.append("DISCLAIMER: Synthetic illustrative formula.")

    return MeasurementTrace(
        trace_id="math_sqrt_demo",
        name=name,
        frequencies=np.copy(trace.frequencies),
        measurements=sqrt_meas,
        frequency_unit=trace.frequency_unit,
        measurement_unit=unit,
        source_type="demo_combination",
        source_filename=trace.source_filename,
        axis_scales=dict(trace.axis_scales),
        calibration_settings={"mode": "demo_sqrt", "calibrated": False},
        extraction_confidence=1.0,
        processing_history=history,
        review_status="provisional",
        metadata={
            "is_simulated": True,
            "is_demo_calculation": True,
            "parent_trace_id": trace.trace_id,
        },
    )
