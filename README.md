# VGA Cable EMI Analyzer — Live Demonstration & Measurement Platform

A modular, scientifically rigorous Python application for analyzing VGA cable radiated electromagnetic interference (EMI). Built with Python 3.12, Streamlit, Matplotlib, NumPy, SciPy, OpenCV, and Pillow.

---

## ⚡ Demonstration & Operational Modes

The application features an unmistakable, top-level distinction between simulation and measurement:

1. **⚡ DEMO MODE — SIMULATED DATA**:
   - Generates deterministic synthetic VGA cable signals with reproducible seeds (`seed=42`).
   - Recreates the 6-panel oscilloscope instrument dashboard (Math 1, Math 2, Math 3, Math 6 Combined FFT, Math 4 Sqrt, and Time-Domain Waveforms).
   - Math 6 and Math 4 operations are documented strictly as **illustrative demo calculations** (linear Root-Sum-Square). Decibels are never added directly.
   - Distinct badges (`[SIMULATED DATA — NOT PHYSICAL MEASUREMENT]`) ensure simulated data is never mistaken for real hardware measurements.
   - Includes a one-click **"Load Presentation Demo Dataset"** button in the sidebar.

2. **🔬 MEASUREMENT MODE — IMPORTED DATA**:
   - Ingests and processes up to 8 real numerical measurement CSV files.
   - Plots actual imported values without synthetic replacement or invented combination formulas.
   - Withholds automated multi-channel combination until physical sensor coupling models are verified.
   - Analyzes unit compatibility and frequency overlap with automated warnings.

---

## 📋 6-Section Workflow Architecture

1. **1. Overview**:
   - Interactive 6-panel dark oscilloscope dashboard (`#0c0e14` background, `#222b3d` fine grid).
   - Real-time channel peak amplitude metrics bar.
   - Comprehensive scientific assumptions expander.
2. **2. Import Measurements**:
   - Batch upload of up to 8 measurement CSV files (or one-click **"Load Sample Lab Measurement Fixtures"**).
   - Automatic column guessing (`Frequency_Hz`, `Amplitude_dBuA_m`, etc.).
   - Unit configuration (`Hz`, `kHz`, `MHz`, `dBµA/m`, `dBm`, `dBµV`, `V`, `mV`, `a.u.`).
   - Detailed audit validation card: total rows, valid points, dropped rows, non-positive frequency check, and duplicates resolved.
   - Interactive per-file mini-plot previews.
   - Preserved oscilloscope screenshot trace segmentation and manual axis calibration interface.
3. **3. Review and Compare**:
   - Multi-trace selector supporting any combination of channels.
   - Automated compatibility report: checks for disparate units and calculates shared frequency overlap intervals.
   - Multi-channel overlay comparison plot with dark/light theme toggle.
   - Spectral peak detection tables with ranking, frequency callouts, and amplitudes.
4. **4. EMI Spectrum**:
   - Publication-grade final EMI spectrum (crisp white background, black axes, light grey grid, deep blue curve, logarithmic frequency axis from 10 Hz to 400 kHz).
   - For single measurement: displays actual calibrated data directly.
   - For multiple measurements: displays individual channels and clearly explains missing combination semantics.
   - Dominant emission peak markers with callouts.
   - In-screen quick export for PNG graphs, CSV spectrum data, and CSV peak tables.
5. **5. Export Results**:
   - High-resolution publication graph export (300 DPI PNG).
   - Tabular numerical spectrum CSV export with metadata headers.
   - Detected spectral peaks table CSV export.
   - Multi-trace comparison summary CSV export.
6. **6. Sessions**:
   - Full session save and restore via secure `.vga_session` zip archives (JSON metadata + NumPy `.npz` arrays).
   - **Zero pickle deserialization** for complete security.
   - Trace provenance and processing history audit log.

---

## 📁 Sample Lab Fixtures (`sample_data/`)

Pre-packaged CSV files are included in the repository for immediate offline testing and live demonstration:
- `vga_ch1_red_spectrum.csv`: 27-point VGA Red video line spectrum (10 Hz to 400 kHz, fundamental at 31.47 kHz).
- `vga_ch2_green_spectrum.csv`: 22-point VGA Green video line spectrum (10 Hz to 400 kHz, harmonics).
- `vga_ch3_blue_spectrum.csv`: 20-point VGA Blue sync line spectrum (10 Hz to 400 kHz).
- `vga_incompatible_volts.csv`: Voltage-domain probe measurement (triggers unit mismatch warning).
- `vga_nonoverlapping_rf.csv`: High-frequency RF emission (10 MHz to 100 MHz, triggers frequency overlap warning).

---

## 🚀 Windows Launch Instructions

### 1. Open Windows PowerShell in Project Directory
```powershell
cd c:\Users\shahi\OneDrive\Documents\GitHub\pro2
```

### 2. Activate Virtual Environment (if not active)
```powershell
.venv\Scripts\Activate.ps1
```

### 3. Launch the Application
```powershell
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

### 4. Run Automated Test Suite
```powershell
python -m pytest -v
```

---

## 🧪 Automated Test Verification

All 44 automated tests pass with 100% success rate:
```
============================= test session starts =============================
platform win32 -- Python 3.12.7, pytest-9.1.1
rootdir: C:\Users\shahi\OneDrive\Documents\GitHub\pro2
44 passed in 2.20s
```

Test coverage includes:
- Model validation, array shapes, non-finite handling, and monotonic frequency checks (`test_models.py`).
- Deterministic synthetic FFT spectra and time-domain waveforms (`test_demo_data.py`).
- SciPy peak detection, ranking, and sensitivity (`test_peak_detection.py`).
- Linear Root-Sum-Square (RSS), decibel roundtrips, and unit checking (`test_combination.py`).
- CSV parsing, column auto-guessing, NaN/inf filtering, duplicate averaging (`test_csv_loader.py`).
- Multi-trace compatibility, unit mismatches, and overlap interval detection (`test_compatibility.py`).
- Export routines: 300 DPI PNG buffers, CSV spectrum, peak tables (`test_export.py`).
- Screenshot image validation, HSV trace segmentation, and axis calibration (`test_image_loader.py`, `test_axis_calibration.py`).
- Sample lab fixtures end-to-end integration (`test_sample_fixtures.py`).
- Session bundle save and restore roundtrips (`test_sessions.py`).
