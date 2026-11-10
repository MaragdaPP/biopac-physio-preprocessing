"""
RSA (Respiratory Sinus Arrhythmia) Analysis Script - Cardiorespiratory Coupling Metrics.

This script combines:
1. Cleaned RSP data from rsp_preprocess_bh.py (RSP_clean_data/)
2. Cleaned HRV data from ecg_preprocess_bh.py (HRV_clean_data/)

Output: Final Excel file with cardiorespiratory coupling metrics (RSA):
- HF-HRV power (primary RSA measure)
- Peak-valley method (RR interval range per breath)
- Porges-Bohrer method (band-pass filtered HR variance)
- Cardiorespiratory ratio (HR/breathing rate)
- Respiratory frequency analysis

Based on:
- Grossman & Taylor (2007) - RSA assessment methods
- Yasuma & Hayano (2004) - RSA mechanisms
- Porges (2007) - Polyvagal theory and RSA
- Task Force (1996) - HRV standards

Note: Run rsp_analysis_final.py for respiratory-only metrics
Note: Run hrv_analysis_final.py for HRV-only metrics
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy < 2.0 has only trapz
import pandas as pd

try:
    from scipy import signal as scipy_signal
    from scipy.interpolate import interp1d
    from scipy.signal import find_peaks
    _HAVE_SCIPY = True
    _HAVE_SCIPY_PEAKS = True
except ImportError:
    scipy_signal = None
    interp1d = None
    find_peaks = None
    _HAVE_SCIPY = False
    _HAVE_SCIPY_PEAKS = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input folders
SCRIPT_DIR = Path(__file__).resolve().parent
RSP_INPUT_FOLDER = str(SCRIPT_DIR / "rsp_bh" / "RSP_clean_data")
HRV_INPUT_FOLDER = str(SCRIPT_DIR / "ecg_bh" / "HRV_clean_data")

# Output file
OUTPUT_FILE = str(SCRIPT_DIR / "RSA_parametros_finales.xlsx")

# Segments to process (ordered: LB, Intero, BH, CPT, Recovery)
SEGMENTS = [
    "LB",
    "Intero 1",
    "Intero 2",
    "Intero 3",
    "Intero 4",
    "Intero 5",
    "Intero 6",
    "BH 1",
    "BH 2",
    "BH 3",
    "CPT",
    "Recovery",
]

# BH segments that require RSP-based trimming (load _trimmed RSP sheet when available)
BH_SEGMENTS = {"BH 1", "BH 2", "BH 3"}

# Ordered list of RSA parameters to keep column order consistent
RSA_PARAM_ORDER = [
    "rsa_hf_power_ms2",
    "rsa_lf_power_ms2",
    "rsa_lf_hf_ratio",
    "rsa_peak_valley_mean_ms",
    "rsa_peak_valley_std_ms",
    "rsa_peak_valley_median_ms",
    "rsa_porges_bohrer_bpm2",
    "cardio_resp_ratio",
]

# RSA frequency band (respiratory frequency range)
RSA_BAND_HZ = (0.15, 0.4)  # 9-24 breaths/min typical range

# ============================================================================
# RSA CALCULATION FUNCTIONS
# ============================================================================

def calculate_hf_hrv_power(hrv_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate HF-HRV power (0.15-0.4 Hz) as primary RSA measure.
    
    This is the most widely accepted RSA metric - the high-frequency component
    of heart rate variability that corresponds to respiratory frequency.
    
    Args:
        hrv_df: DataFrame with RR intervals
    
    Returns:
        Dictionary with HF-HRV power metrics
    """
    params = {
        "rsa_hf_power_ms2": np.nan,
        "rsa_lf_power_ms2": np.nan,
        "rsa_lf_hf_ratio": np.nan,
    }
    
    if not _HAVE_SCIPY:
        return params
    
    # Filter non-interpolated beats
    valid_beats = hrv_df[hrv_df['is_interpolated'] == False].copy()
    
    if len(valid_beats) < 20:  # Need minimum data for spectral analysis
        return params
    
    # Resample RR intervals to evenly spaced time series (4 Hz standard)
    times = valid_beats['time_s'].values
    rr_ms_vals = valid_beats['rr_ms'].values

    # Sort by timestamp and drop duplicate timestamps so interp1d doesn't raise
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    rr_ms_vals = rr_ms_vals[sort_idx]
    unique_mask = np.concatenate(([True], np.diff(times) > 0))
    times = times[unique_mask]
    rr_ms_vals = rr_ms_vals[unique_mask]
    if len(times) < 20:
        return params

    # Interpolate to 4 Hz (0.25 s spacing)
    fs_resample = 4.0
    t_uniform = np.arange(times[0], times[-1], 1.0 / fs_resample)

    if len(t_uniform) < 16:
        return params

    try:
        interp_func = interp1d(times, rr_ms_vals, kind='cubic', fill_value='extrapolate')
        rr_uniform = interp_func(t_uniform)
    except Exception as e:
        warnings.warn(f"[calculate_hf_hrv_power] interpolation failed: {e}")
        return params
    
    # Detrend
    rr_detrended = rr_uniform - np.mean(rr_uniform)
    
    # Welch's method for PSD
    try:
        nperseg = min(256, len(rr_detrended))
        freqs, psd = scipy_signal.welch(rr_detrended, fs=fs_resample, nperseg=nperseg, 
                                        nfft=4096, scaling='density')
        
        # Frequency bands
        lf_band = (0.04, 0.15)  # Low frequency
        hf_band = (0.15, 0.4)   # High frequency (RSA band)
        
        # Calculate power in each band
        lf_power = _trapz(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])],
                               freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
        hf_power = _trapz(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])],
                               freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
        
        params["rsa_hf_power_ms2"] = hf_power
        params["rsa_lf_power_ms2"] = lf_power
        params["rsa_lf_hf_ratio"] = lf_power / hf_power if hf_power > 0 else np.nan
        
    except Exception as e:
        warnings.warn(f"HF-HRV power calculation failed: {e}")
    
    return params


def calculate_peak_valley_rsa(hrv_df: pd.DataFrame, cycles_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate peak-valley RSA (Grossman & Taylor 2007).
    
    For each breathing cycle, find the maximum and minimum RR intervals.
    RSA = mean of (max RR - min RR) across all cycles.
    
    Args:
        hrv_df: DataFrame with RR intervals
        cycles_df: DataFrame with breathing cycles
    
    Returns:
        Dictionary with peak-valley RSA metrics
    """
    params = {
        "rsa_peak_valley_mean_ms": np.nan,
        "rsa_peak_valley_std_ms": np.nan,
        "rsa_peak_valley_median_ms": np.nan,
    }
    
    # Filter valid data
    valid_cycles = cycles_df[cycles_df['valid_for_coupling'] == True].copy()
    valid_beats = hrv_df[hrv_df['is_interpolated'] == False].copy()
    
    if len(valid_cycles) < 2 or len(valid_beats) < 2:
        return params
    
    # For each breathing cycle, find HR peaks and valleys
    peak_valley_diffs = []
    
    for _, cycle in valid_cycles.iterrows():
        cycle_start = cycle['trough_time_s']
        cycle_end = cycle['next_trough_time_s']
        
        # Find beats within this cycle
        cycle_beats = valid_beats[(valid_beats['time_s'] >= cycle_start) & 
                                 (valid_beats['time_s'] < cycle_end)]
        
        if len(cycle_beats) >= 2:
            # Find max and min RR intervals
            max_rr = cycle_beats['rr_ms'].max()
            min_rr = cycle_beats['rr_ms'].min()
            peak_valley_diffs.append(max_rr - min_rr)
    
    if len(peak_valley_diffs) > 0:
        params["rsa_peak_valley_mean_ms"] = np.mean(peak_valley_diffs)
        params["rsa_peak_valley_std_ms"] = np.std(peak_valley_diffs, ddof=1) if len(peak_valley_diffs) > 1 else 0
        params["rsa_peak_valley_median_ms"] = np.median(peak_valley_diffs)
    
    return params


def calculate_porges_bohrer_rsa(hrv_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Porges-Bohrer RSA (band-pass filtered HR variance).
    
    This method filters HR in the respiratory frequency range (0.15-0.4 Hz)
    and calculates the variance of the filtered signal.
    
    Args:
        hrv_df: DataFrame with RR intervals
    
    Returns:
        Dictionary with Porges-Bohrer RSA metric
    """
    params = {
        "rsa_porges_bohrer_bpm2": np.nan,
    }
    
    if not _HAVE_SCIPY:
        return params
    
    # Filter valid beats
    valid_beats = hrv_df[hrv_df['is_interpolated'] == False].copy()
    
    if len(valid_beats) < 30:
        return params
    
    try:
        # Resample HR to uniform time series
        times = valid_beats['time_s'].values
        hr_bpm = valid_beats['hr_bpm'].values
        
        fs_resample = 4.0
        t_uniform = np.arange(times[0], times[-1], 1.0 / fs_resample)
        
        if len(t_uniform) < 20:
            return params
        
        interp_func = interp1d(times, hr_bpm, kind='cubic', fill_value='extrapolate')
        hr_uniform = interp_func(t_uniform)
        
        # Bandpass filter in respiratory frequency range
        sos = scipy_signal.butter(5, RSA_BAND_HZ, btype='band', fs=fs_resample, output='sos')
        hr_filtered = scipy_signal.sosfiltfilt(sos, hr_uniform)
        
        # RSA = variance of filtered HR
        params["rsa_porges_bohrer_bpm2"] = np.var(hr_filtered, ddof=1)
        
    except Exception as e:
        warnings.warn(f"Porges-Bohrer calculation failed: {e}")
    
    return params


def calculate_cardiorespiratory_ratio(hrv_df: pd.DataFrame, cycles_df: pd.DataFrame, 
                                    duration_s: float) -> Dict[str, float]:
    """
    Calculate cardiorespiratory ratio and related coupling metrics.
    
    Args:
        hrv_df: DataFrame with RR intervals
        cycles_df: DataFrame with breathing cycles
        duration_s: Segment duration in seconds
    
    Returns:
        Dictionary with cardiorespiratory coupling metrics
    """
    params = {
        "cardio_resp_ratio": np.nan,
    }
    
    # Calculate mean HR
    valid_beats = hrv_df[hrv_df['is_interpolated'] == False].copy()
    mean_hr = valid_beats['hr_bpm'].mean() if len(valid_beats) > 0 else np.nan
    
    # Calculate mean breathing rate
    valid_cycles = cycles_df[cycles_df['valid_for_metrics'] == True].copy()
    mean_breathing_rate = valid_cycles['resp_rate_bpm'].mean() if len(valid_cycles) > 0 else np.nan
    
    # Calculate ratio
    if pd.notna(mean_hr) and pd.notna(mean_breathing_rate) and mean_breathing_rate > 0:
        params["cardio_resp_ratio"] = mean_hr / mean_breathing_rate
    
    return params


# ============================================================================
# CYCLE DETECTION FROM SIGNAL (fallback when _cycles sheet missing)
# ============================================================================

def detect_cycles_from_signal(time_s: np.ndarray, rsp_clean: np.ndarray) -> pd.DataFrame:
    """
    Detect breathing cycles directly from the cleaned signal.
    Used only when per-cycle sheets are missing.
    """
    if not _HAVE_SCIPY_PEAKS:
        return pd.DataFrame()

    if len(time_s) < 3 or len(rsp_clean) < 3:
        return pd.DataFrame()

    # Estimate sampling rate from time vector
    dt = np.diff(time_s)
    fs = 1.0 / np.median(dt) if len(dt) > 0 and np.median(dt) > 0 else 63.0

    # Light smoothing to reduce spurious peaks
    win = max(3, int(0.3 * fs))
    kernel = np.ones(win) / win
    rsp_smooth = np.convolve(rsp_clean, kernel, mode="same")

    # Breathing rate bounds
    min_bpm = 4.0
    max_bpm = 60.0
    min_dist = int(fs * (60.0 / max_bpm))

    peaks, _ = find_peaks(rsp_smooth, distance=min_dist)
    troughs, _ = find_peaks(-rsp_smooth, distance=min_dist)

    if len(troughs) < 2:
        return pd.DataFrame()

    rows = []
    troughs = np.sort(troughs)
    peaks = np.sort(peaks)

    for i in range(len(troughs) - 1):
        t_idx = troughs[i]
        t_next_idx = troughs[i + 1]
        if t_idx >= len(time_s) or t_next_idx >= len(time_s):
            continue

        peaks_between = peaks[(peaks > t_idx) & (peaks < t_next_idx)]
        if len(peaks_between) == 0:
            continue
        p_idx = peaks_between[0]

        trough_time = time_s[t_idx]
        next_trough_time = time_s[t_next_idx]
        peak_time = time_s[p_idx]

        interval_s = next_trough_time - trough_time
        if interval_s <= 0:
            continue

        resp_rate_bpm = 60.0 / interval_s
        if resp_rate_bpm < min_bpm or resp_rate_bpm > max_bpm:
            continue

        amplitude = abs(rsp_smooth[p_idx] - rsp_smooth[t_idx])

        rows.append({
            "cycle_idx": i,
            "trough_time_s": trough_time,
            "peak_time_s": peak_time,
            "next_trough_time_s": next_trough_time,
            "interval_s": interval_s,
            "amplitude": amplitude,
            "resp_rate_bpm": resp_rate_bpm,
            "overlaps_bad": False,
            "overlaps_interpolated": False,
            "valid_for_metrics": True,
            "valid_for_coupling": True,
        })

    return pd.DataFrame(rows)

# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def load_subject_data(subject_id: int) -> Tuple[Optional[pd.ExcelFile], Optional[pd.ExcelFile]]:
    """
    Load RSP and HRV data for a subject.
    
    Args:
        subject_id: Subject ID number
    
    Returns:
        Tuple of (rsp_excel, hrv_excel) or (None, None) if files not found
    """
    rsp_file = Path(RSP_INPUT_FOLDER) / f"RSP_data_sujeto_{subject_id}.xlsx"
    hrv_file = Path(HRV_INPUT_FOLDER) / f"HRV_data_sujeto_{subject_id}.xlsx"
    
    rsp_excel = None
    hrv_excel = None
    
    if rsp_file.exists():
        try:
            rsp_excel = pd.ExcelFile(rsp_file)
        except Exception as e:
            warnings.warn(f"Error loading RSP file for subject {subject_id}: {e}")
    
    if hrv_file.exists():
        try:
            hrv_excel = pd.ExcelFile(hrv_file)
        except Exception as e:
            warnings.warn(f"Error loading HRV file for subject {subject_id}: {e}")
    
    return rsp_excel, hrv_excel


def process_segment(segment_name: str, rsp_excel: pd.ExcelFile,
                    hrv_excel: pd.ExcelFile) -> Dict[str, float]:
    """
    Process a single segment and calculate RSA metrics.

    For BH segments the RSP _trimmed sheet is used (if present) so that the
    analysis window matches the true apnea period.  HF-HRV power and
    Porges-Bohrer RSA are computed whenever enough HRV beats are available,
    independent of whether breathing cycles were detected.  Cycle-based metrics
    (peak-valley RSA, cardiorespiratory ratio) are computed only when cycles
    exist — they will naturally be NaN for BH segments.

    Args:
        segment_name: Name of the segment
        rsp_excel: RSP Excel file object
        hrv_excel: HRV Excel file object

    Returns:
        Dictionary with all calculated RSA metrics (NaN where not applicable)
    """
    # Initialise all params with NaN so every column is always present
    all_params = {
        "rsa_hf_power_ms2": np.nan,
        "rsa_lf_power_ms2": np.nan,
        "rsa_lf_hf_ratio": np.nan,
        "rsa_peak_valley_mean_ms": np.nan,
        "rsa_peak_valley_std_ms": np.nan,
        "rsa_peak_valley_median_ms": np.nan,
        "rsa_porges_bohrer_bpm2": np.nan,
        "cardio_resp_ratio": np.nan,
    }

    is_bh = segment_name in BH_SEGMENTS

    # HRV data is always required
    has_hrv = segment_name in hrv_excel.sheet_names if hrv_excel else False
    if not has_hrv:
        return all_params

    # ── RSP: for BH try the _trimmed sheet first, fall back to regular sheet ──
    rsp_sheet_name = segment_name
    if is_bh and rsp_excel is not None:
        trimmed_sheet = f"{segment_name}_trimmed"
        if trimmed_sheet in rsp_excel.sheet_names:
            rsp_sheet_name = trimmed_sheet
        # else: fall back to regular segment sheet (or missing RSP handled below)

    rsp_df = None
    if rsp_excel is not None and rsp_sheet_name in rsp_excel.sheet_names:
        rsp_df = pd.read_excel(rsp_excel, sheet_name=rsp_sheet_name)

    # ── Cap LB / Recovery to 300s (older exports may be longer) ──────────────
    if (not is_bh) and segment_name in ("LB", "Recovery"):
        if rsp_df is not None and "time_s" in rsp_df.columns and len(rsp_df) > 0:
            t0_rsp = float(rsp_df["time_s"].iloc[0])
            rsp_df = rsp_df[rsp_df["time_s"] <= (t0_rsp + 300.0)].copy()

    # Derive breathing cycles from the RSP signal (empty for BH — no breathing)
    cycles_df = None
    if rsp_df is not None and "time_s" in rsp_df.columns and "rsp_clean" in rsp_df.columns:
        cycles_df = detect_cycles_from_signal(
            rsp_df["time_s"].to_numpy(),
            rsp_df["rsp_clean"].to_numpy(),
        )

    # Load full HRV data for this segment
    hrv_df_full = pd.read_excel(hrv_excel, sheet_name=segment_name)

    # Cap LB / Recovery to 300s using HRV time vector (if present)
    if (not is_bh) and segment_name in ("LB", "Recovery"):
        if "time_s" in hrv_df_full.columns and len(hrv_df_full) > 0:
            t0_hrv = float(hrv_df_full["time_s"].iloc[0])
            hrv_df_full = hrv_df_full[hrv_df_full["time_s"] <= (t0_hrv + 300.0)].copy()

    # ── Trim HRV to match RSP trimmed duration for BH segments ───────────────
    if (
        is_bh
        and rsp_df is not None
        and "time_s" in hrv_df_full.columns
        and "time_s" in rsp_df.columns
        and len(rsp_df) > 0
        and len(hrv_df_full) > 0
    ):
        trimmed_duration = rsp_df["time_s"].iloc[-1] - rsp_df["time_s"].iloc[0]
        t0 = hrv_df_full["time_s"].iloc[0]
        hrv_df = hrv_df_full[hrv_df_full["time_s"] <= t0 + trimmed_duration].copy()
    else:
        hrv_df = hrv_df_full

    # Duration from RSP (or HRV as fallback)
    if rsp_df is not None and "time_s" in rsp_df.columns and len(rsp_df) > 0:
        duration_s = rsp_df["time_s"].iloc[-1] - rsp_df["time_s"].iloc[0]
    elif "time_s" in hrv_df.columns and len(hrv_df) > 0:
        duration_s = hrv_df["time_s"].iloc[-1] - hrv_df["time_s"].iloc[0]
    else:
        duration_s = np.nan

    # ── Spectral methods: run whenever enough HRV data (no cycles required) ──
    if len(hrv_df) >= 20:
        hf_params = calculate_hf_hrv_power(hrv_df)
        all_params.update(hf_params)

        pb_params = calculate_porges_bohrer_rsa(hrv_df)
        all_params.update(pb_params)

    # ── Cycle-dependent metrics: only when breathing cycles are available ─────
    if cycles_df is not None and len(cycles_df) > 0:
        pv_params = calculate_peak_valley_rsa(hrv_df, cycles_df)
        all_params.update(pv_params)

        cr_params = calculate_cardiorespiratory_ratio(hrv_df, cycles_df, duration_s)
        all_params.update(cr_params)

    return all_params


def process_subject(subject_id: int) -> Dict[str, Dict[str, float]]:
    """
    Process all segments for a single subject.
    
    Args:
        subject_id: Subject ID number
    
    Returns:
        Dictionary mapping segment names to parameter dictionaries
    """
    print(f"\nProcessing subject {subject_id}...")
    
    # Load data files
    rsp_excel, hrv_excel = load_subject_data(subject_id)
    
    if rsp_excel is None or hrv_excel is None:
        warnings.warn(f"Missing RSP or HRV data for subject {subject_id}, skipping.")
        return {}
    
    results = {}
    
    # Process segments
    for segment_name in SEGMENTS:
        print(f"  Processing segment: {segment_name}")
        params = process_segment(segment_name, rsp_excel, hrv_excel)
        results[segment_name] = params
    
    return results


def build_output_dataframe(all_results: Dict[int, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """
    Build final output DataFrame from all subject results.
    
    Args:
        all_results: Nested dictionary {subject_id: {segment_name: {param: value}}}
    
    Returns:
        DataFrame with one row per subject, columns for each segment's parameters
    """
    rows = []
    
    for subject_id, subject_results in sorted(all_results.items()):
        row = {"subject_id": subject_id}

        for segment_name in SEGMENTS:
            params = subject_results.get(segment_name, {})
            segment_prefix = segment_name.replace(" ", "_")
            for param_name in RSA_PARAM_ORDER:
                col_name = f"{segment_prefix}_{param_name}"
                row[col_name] = params.get(param_name, np.nan)

        rows.append(row)

    # Build DataFrame with consistent column order
    columns = ["subject_id"]
    for segment_name in SEGMENTS:
        segment_prefix = segment_name.replace(" ", "_")
        for param_name in RSA_PARAM_ORDER:
            columns.append(f"{segment_prefix}_{param_name}")

    df = pd.DataFrame(rows, columns=columns)
    
    return df


def main():
    """
    Main function to process all subjects and save results.
    """
    print("="*80)
    print("RSA Analysis - Cardiorespiratory Coupling Metrics")
    print("="*80)
    print(f"\nRSP Input folder: {RSP_INPUT_FOLDER}")
    print(f"HRV Input folder: {HRV_INPUT_FOLDER}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Check dependencies
    print(f"  SciPy available: {_HAVE_SCIPY}")
    if not _HAVE_SCIPY:
        warnings.warn("SciPy not available. RSA frequency-domain metrics will not be calculated.")
    
    # Find all subject IDs from RSP files
    rsp_folder = Path(RSP_INPUT_FOLDER)
    rsp_files = list(rsp_folder.glob("RSP_data_sujeto_*.xlsx"))
    
    subject_ids = []
    for f in rsp_files:
        try:
            subject_id = int(f.stem.split("_")[-1])
            subject_ids.append(subject_id)
        except:
            pass
    
    subject_ids = sorted(subject_ids)
    print(f"\nFound {len(subject_ids)} subjects: {subject_ids}")
    
    # Process all subjects
    all_results = {}
    
    for subject_id in subject_ids:
        try:
            subject_results = process_subject(subject_id)
            if subject_results:
                all_results[subject_id] = subject_results
        except Exception as e:
            warnings.warn(f"Error processing subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Build output DataFrame
    print("\nBuilding output DataFrame...")
    output_df = build_output_dataframe(all_results)
    
    # Save to Excel
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        output_df.to_excel(writer, sheet_name="RSA_Parameters", index=False)

        # Parameter definitions
        param_definitions = pd.DataFrame({
            "parameter": RSA_PARAM_ORDER,
            "description": [
                "HF power of HRV in RSA band (0.15-0.40 Hz) (ms²)",
                "LF power of HRV (0.04-0.15 Hz) (ms²)",
                "LF/HF ratio",
                "Peak-valley RSA mean (ms)",
                "Peak-valley RSA standard deviation (ms)",
                "Peak-valley RSA median (ms)",
                "Porges-Bohrer RSA (variance of bandpass filtered HR) (bpm²)",
                "Cardiorespiratory ratio (mean HR / mean breathing rate)",
            ],
        })
        param_definitions.to_excel(writer, sheet_name="Parameter_Definitions", index=False)
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Total subjects processed: {len(all_results)}")
    print(f"Output shape: {output_df.shape[0]} rows × {output_df.shape[1]} columns")
    print("="*80)
    
    return output_df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Run main analysis
    results_df = main()
    
    # Display summary
    print("\nFirst few rows of output:")
    print(results_df.head())
    
    print("\nColumn names (first 20):")
    for i, col in enumerate(results_df.columns[:20]):
        print(f"  {i+1}. {col}")
    
    if len(results_df.columns) > 20:
        print(f"  ... and {len(results_df.columns) - 20} more columns")
