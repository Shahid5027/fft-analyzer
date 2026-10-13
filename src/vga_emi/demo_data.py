"""Deterministic synthetic data generator for VGA EMI Analyzer.

Generates realistic frequency-domain FFT spectra and time-domain waveforms
resembling VGA cable emissions (e.g. 31.47 kHz horizontal scan frequency harmonics,
pixel-clock intermodulation, and RGB video waveforms).
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from vga_emi.models import MeasurementTrace, TimeDomainTrace


def generate_synthetic_fft_traces(
    seed: int = 42,
    f_min: float = 10.0,
    f_max: float = 400000.0,
    num_points: int = 2048,
    num_traces: int = 3,
    scale: str = "log",
    measurement_unit: str = "dBµA/m",
) -> List[MeasurementTrace]:
    """Generate deterministic, distinct synthetic FFT traces.

    Simulates realistic cable emissions including 1/f noise floor,
    harmonics of the VGA line rate (31.468 kHz), and line/pixel coupling.

    Args:
        seed: Random seed for exact determinism.
        f_min: Minimum frequency in Hz (must be > 0 for log scale).
        f_max: Maximum frequency in Hz.
        num_points: Number of frequency points.
        num_traces: Number of distinct traces to generate (2 to 8).
        scale: Frequency scale ('log' or 'linear').
        measurement_unit: Unit label (explicitly marked simulated).

    Returns:
        List of MeasurementTrace objects.
    """
    if f_min <= 0 and scale == "log":
        raise ValueError(f"f_min must be positive for logarithmic scale, got {f_min}")
    if f_min >= f_max:
        raise ValueError(f"f_min ({f_min}) must be less than f_max ({f_max})")
    if num_traces < 1:
        raise ValueError("num_traces must be at least 1")

    # Generate strictly increasing frequency grid
    if scale == "log":
        freqs = np.geomspace(f_min, f_max, num=num_points)
    else:
        freqs = np.linspace(f_min, f_max, num=num_points)

    # Base VGA fundamental frequency (standard 640x480 @ 60Hz horizontal line frequency ~ 31.468 kHz)
    f_hsync = 31468.0

    # Trace characteristic profiles
    trace_profiles = [
        {
            "name": "Math 1 — Ch1 FFT (VGA Red)",
            "base_floor": 22.0,
            "slope": -3.5,  # 1/f drop per decade
            "peaks": [
                (15734.0, 18.0, 0.04),  # Subharmonic
                (f_hsync, 38.0, 0.03),   # Fundamental HSync
                (2 * f_hsync, 32.0, 0.035),
                (3 * f_hsync, 26.0, 0.04),
                (4 * f_hsync, 21.0, 0.045),
                (6 * f_hsync, 17.0, 0.05),
                (250000.0, 24.0, 0.06), # Pixel toggle harmonic
            ],
            "noise_std": 1.4,
        },
        {
            "name": "Math 2 — Ch2 FFT (VGA Green)",
            "base_floor": 25.0,
            "slope": -4.0,
            "peaks": [
                (f_hsync, 42.0, 0.025),  # Stronger fundamental
                (2 * f_hsync, 28.0, 0.03),
                (4 * f_hsync, 29.0, 0.035),
                (5 * f_hsync, 19.0, 0.04),
                (120000.0, 22.0, 0.05),
                (8 * f_hsync, 23.0, 0.045),
                (320000.0, 19.0, 0.06),
            ],
            "noise_std": 1.6,
        },
        {
            "name": "Math 3 — Ch3 FFT (VGA Blue / Sync)",
            "base_floor": 20.0,
            "slope": -3.0,
            "peaks": [
                (f_hsync, 35.0, 0.03),
                (3 * f_hsync, 34.0, 0.035),  # Pronounced odd harmonic from square sync
                (5 * f_hsync, 27.0, 0.04),
                (7 * f_hsync, 23.0, 0.045),
                (9 * f_hsync, 19.0, 0.05),
                (175000.0, 25.0, 0.055),
                (360000.0, 18.0, 0.06),
            ],
            "noise_std": 1.5,
        },
    ]

    traces: List[MeasurementTrace] = []

    for i in range(num_traces):
        rng = np.random.RandomState(seed + i * 1000)
        
        if i < len(trace_profiles):
            profile = trace_profiles[i]
            trace_name = profile["name"]
            base_floor = profile["base_floor"]
            slope = profile["slope"]
            peaks = profile["peaks"]
            noise_std = profile["noise_std"]
        else:
            trace_name = f"Math {i+1} — Ch{i+1} FFT"
            base_floor = 20.0 + (i % 3) * 3.0
            slope = -3.2 - (i % 2) * 0.5
            peaks = [
                (f_hsync * (k + 1), 30.0 / (1 + 0.3 * k), 0.03 + 0.01 * k)
                for k in range(5)
            ]
            noise_std = 1.5

        # 1/f noise floor in log frequency domain
        log_f = np.log10(freqs)
        floor = base_floor + slope * (log_f - np.log10(max(f_min, 1.0)))

        # Add spectral peaks with realistic Lorentzian / Gaussian line shape
        peak_spectrum = np.zeros_like(freqs)
        for center_f, height_db, rel_width in peaks:
            if center_f < f_min or center_f > f_max:
                continue
            width_hz = center_f * rel_width
            # Skewed peak profile resembling analog filter / spectrum analyzer resolution bandwidth
            dx = (freqs - center_f) / (width_hz / 2.0)
            lorentzian = 1.0 / (1.0 + dx**2)
            peak_spectrum = np.maximum(peak_spectrum, height_db * lorentzian)

        # Realistic high-frequency noise with slight coloring
        raw_noise = rng.normal(0, noise_std, size=num_points)
        # Apply gentle low-pass smoothing to noise to prevent single-pixel spikes
        kernel = np.array([0.25, 0.5, 0.25])
        smoothed_noise = np.convolve(raw_noise, kernel, mode="same")

        spectrum_db = floor + peak_spectrum + smoothed_noise
        # Floor baseline at 0 dBµA/m for realistic instrumentation display
        spectrum_db = np.clip(spectrum_db, 0.0, 120.0)

        trace = MeasurementTrace(
            trace_id=f"trace_{i+1}",
            name=trace_name,
            frequencies=freqs,
            measurements=spectrum_db,
            frequency_unit="Hz",
            measurement_unit=measurement_unit,
            source_type="synthetic",
            source_filename=f"synthetic_ch{i+1}.dat",
            axis_scales={"x": scale, "y": "linear"},
            calibration_settings={"calibrated": False, "mode": "synthetic"},
            extraction_confidence=1.0,
            processing_history=[
                f"Generated synthetic deterministic trace with seed={seed + i*1000}",
                f"Synthesized VGA line-rate harmonics around {f_hsync:.1f} Hz",
            ],
            review_status="provisional",
            metadata={
                "is_simulated": True,
                "channel_index": i + 1,
                "seed": seed + i * 1000,
            },
        )
        traces.append(trace)

    return traces


def generate_vga_time_domain(
    seed: int = 42,
    time_window_us: float = 100.0,
    num_samples: int = 2048,
    num_channels: int = 3,
) -> List[TimeDomainTrace]:
    """Generate deterministic synthetic time-domain waveforms for VGA signals.

    Mimics standard analog VGA signals:
    - Channel 1 (Red): Active video steps with horizontal blanking intervals.
    - Channel 2 (Green): Ramp color test pattern with sync level.
    - Channel 3 (Blue / HSync): Horizontal synchronization pulses (active low pulse ~3.8µs).

    Args:
        seed: Random seed for determinism.
        time_window_us: Total time duration in microseconds.
        num_samples: Number of sample points.
        num_channels: Number of channels (up to 4).

    Returns:
        List of TimeDomainTrace objects.
    """
    if time_window_us <= 0:
        raise ValueError(f"time_window_us must be positive, got {time_window_us}")
    if num_samples < 16:
        raise ValueError(f"num_samples must be at least 16, got {num_samples}")

    t = np.linspace(0, time_window_us, num_samples)
    rng = np.random.RandomState(seed + 99)

    # Standard VGA 640x480 timing: line period = 31.777 µs
    # HSync pulse width = 3.77 µs, Back porch = 1.89 µs, Front porch = 0.94 µs
    t_line = 31.777
    hsync_width = 3.77

    traces: List[TimeDomainTrace] = []
    channel_configs = [
        {
            "id": "ch1",
            "name": "CH1 — Red Video (Active Video + Blanking)",
            "base_voltage": 0.0,
            "peak_v": 0.7,  # Standard VGA 0.7V p-p into 75 ohm
        },
        {
            "id": "ch2",
            "name": "CH2 — Green Video (Test Pattern Ramp)",
            "base_voltage": 0.0,
            "peak_v": 0.7,
        },
        {
            "id": "ch3",
            "name": "CH3 — HSync (Horizontal Sync TTL)",
            "base_voltage": 5.0,  # TTL 5V logic (active low)
            "peak_v": 0.0,
        },
        {
            "id": "ch4",
            "name": "CH4 — Blue Video / Clock Coupling",
            "base_voltage": 0.0,
            "peak_v": 0.7,
        },
    ]

    for ch_idx in range(num_channels):
        cfg = channel_configs[ch_idx % len(channel_configs)]
        # Phase within line period
        phase = t % t_line

        if ch_idx == 0:
            # Channel 1: Video line pulses (stepped color bars during active video, 0 during blanking)
            is_active = (phase > (hsync_width + 1.89)) & (phase < (t_line - 0.94))
            v = np.zeros_like(t)
            if np.any(is_active):
                active_t = phase[is_active] - (hsync_width + 1.89)
                v[is_active] = 0.7 * np.floor(active_t / 5.0) / 5.0
                v[is_active] += 0.02 * np.sin(2 * np.pi * 5.0 * t[is_active])
            v += rng.normal(0, 0.008, size=num_samples)

        elif ch_idx == 1:
            # Channel 2: Ramp active video test pattern
            is_active = (phase > (hsync_width + 1.89)) & (phase < (t_line - 0.94))
            v = np.zeros_like(t)
            if np.any(is_active):
                active_len = t_line - (hsync_width + 1.89 + 0.94)
                norm_ramp = np.clip((phase[is_active] - (hsync_width + 1.89)) / active_len, 0.0, 1.0)
                v[is_active] = 0.7 * norm_ramp
            v += rng.normal(0, 0.007, size=num_samples)

        elif ch_idx == 2:
            # Channel 3: HSync Pulse (standard VGA 0V pulse during sync, 5.0V TTL logic high)
            is_sync_pulse = phase < hsync_width
            v = np.where(is_sync_pulse, 0.2, 4.8)
            # Slight rise/fall exponential edge
            edge_noise = rng.normal(0, 0.03, size=num_samples)
            v += edge_noise

        else:
            # Additional channels: sinusoidal tone / clock feedthrough
            v = 0.35 + 0.35 * np.sin(2 * np.pi * (0.05 * (ch_idx + 1)) * t)
            v += rng.normal(0, 0.01, size=num_samples)

        trace = TimeDomainTrace(
            channel_id=cfg["id"],
            name=cfg["name"],
            time=t,
            voltage=v,
            time_unit="µs",
            voltage_unit="V",
            source_type="synthetic",
            is_simulated=True,
            metadata={
                "time_window_us": time_window_us,
                "num_samples": num_samples,
                "seed": seed,
            },
        )
        traces.append(trace)

    return traces
