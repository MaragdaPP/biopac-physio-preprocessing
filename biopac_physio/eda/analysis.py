"""
EDA Analysis Script - Calculate EDA parameters from cleaned EDA data.

This script reads the cleaned EDA data exported by eda_preprocess_bh.py
and calculates comprehensive EDA parameters (tonic and phasic).

Output: Final Excel file with one row per subject, columns for each segment's EDA parameters.

Based on:
- Boucsein et al. (2012) SPR recommendations
- Braithwaite et al. guide
- cvxEDA decomposition method
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import neurokit2 as nk
    _HAVE_NK = True
except ImportError as e:
    print(f"⚠️ WARNING: NeuroKit2 import failed: {e}")
    nk = None
    _HAVE_NK = False
except Exception as e:
    print(f"⚠️ WARNING: NeuroKit2 import error: {e}")
    nk = None
    _HAVE_NK = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# SCR detection parameters (should match eda_preprocess_bh.py)
SCR_MIN_AMPLITUDE_US = 0.05  # µS - minimum amplitude for SCR detection (aligned with eda_preprocess_bh)
SAMPLING_RATE = 32.0  # Hz - default sampling rate for EDA

# Input folder containing cleaned EDA Excel files (one per subject)
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = str(SCRIPT_DIR / "EDA_clean_data")
RSP_FOLDER = str(SCRIPT_DIR.parent / "rsp_bh" / "RSP_clean_data")  # For BH_trimmed timing

# Output file for final EDA parameters
OUTPUT_FILE = str(SCRIPT_DIR / "EDA_parametros_finales.xlsx")

# Segments to process (in order) — BH segments are handled separately with RSP-based trimming
SEGMENTS = [
    "LB",
    "Intero 1",
    "Intero 2",
    "Intero 3",
    "Intero 4",
    "Intero 5",
    "Intero 6",
    "CPT",
    "Recovery",
]

# BH segments processed separately (trimmed to true apnea period using RSP_clean_data)
BH_SEGMENTS = ["BH 1", "BH 2", "BH 3", "BH 4"]

# EDA parameter names
EDA_PARAM_NAMES = [
    # Tonic parameters (SCL - Skin Conductance Level)
    "scl_mean", "scl_std", "scl_min", "scl_max", "scl_range",
    "scl_slope",  # Drift/slope of tonic component
    
    # Phasic parameters (SCR - Skin Conductance Response)
    "scr_count", "scr_frequency_rate",  # Number and frequency (SCRs per minute)
    "scr_amplitude_mean_raw", "scr_amplitude_std_raw",  # Mean amplitude (only detected SCRs)
    "scr_rise_time_mean", "scr_rise_time_std",  # Mean rise time
    
    # Transformed parameters
    "scr_amplitude_sqrt", "scr_amplitude_log",  # Square root and log(x+1) transformed
    
    # Standardized parameters (Z-scores) - calculated per subject
    "scr_amplitude_z",
    
    # Range correction (normalized by max response) - calculated per subject
    "scr_amplitude_range_corrected",
    
    # Quality metrics
    "duration_s", "bad_pct", "n_samples", "qc_valid",
]


# ============================================================================
# EDA CALCULATION FUNCTIONS
# ============================================================================

def calculate_tonic_parameters(tonic: np.ndarray, times: np.ndarray) -> Dict[str, float]:
    """
    Calculate tonic (SCL) parameters.
    
    Args:
        tonic: Tonic component (SCL) in microsiemens (µS)
        times: Time vector in seconds
    
    Returns:
        Dictionary with tonic parameters
    """
    params = {}
    
    # Remove NaN values for statistics
    tonic_valid = tonic[~np.isnan(tonic)]
    times_valid = times[~np.isnan(tonic)]
    
    if len(tonic_valid) == 0:
        params["scl_mean"] = np.nan
        params["scl_std"] = np.nan
        params["scl_min"] = np.nan
        params["scl_max"] = np.nan
        params["scl_range"] = np.nan
        params["scl_slope"] = np.nan
        return params
    
    # Basic statistics
    params["scl_mean"] = np.mean(tonic_valid)
    params["scl_std"] = np.std(tonic_valid, ddof=1) if len(tonic_valid) > 1 else 0
    params["scl_min"] = np.min(tonic_valid)
    params["scl_max"] = np.max(tonic_valid)
    params["scl_range"] = params["scl_max"] - params["scl_min"]
    
    # Drift/slope: linear regression of tonic component over time
    if len(times_valid) > 1:
        try:
            # Simple linear regression: y = a + b*x
            coeffs = np.polyfit(times_valid, tonic_valid, 1)
            params["scl_slope"] = coeffs[0]  # Slope in µS/s
        except:
            params["scl_slope"] = np.nan
    else:
        params["scl_slope"] = np.nan
    
    return params


def detect_scr_peaks(
    phasic: np.ndarray,
    times: np.ndarray,
    fs: float = SAMPLING_RATE,
    min_amplitude: float = SCR_MIN_AMPLITUDE_US,
) -> Dict[str, np.ndarray]:
    """
    Detect SCR peaks from phasic component.
    
    Args:
        phasic: Phasic component (SCR) in microsiemens (µS)
        times: Time vector in seconds
        fs: Sampling frequency in Hz
        min_amplitude: Minimum amplitude threshold in µS
    
    Returns:
        Dictionary with 'peaks', 'onsets', 'amplitudes', 'rise_times'
    """
    scr_info = {
        "peaks": np.array([]),
        "onsets": np.array([]),
        "amplitudes": np.array([]),
        "rise_times": np.array([]),
    }
    
    if not _HAVE_NK:
        # Silently skip if neurokit2 not available
        return scr_info
    
    # Remove NaN values and check if we have enough valid data
    valid_mask = ~np.isnan(phasic)
    if np.sum(valid_mask) < 10:
        return scr_info
    
    # Check if phasic has any meaningful variation
    phasic_valid = phasic[valid_mask]
    if np.std(phasic_valid) < 0.001:  # Too flat
        return scr_info
    
    try:
        # Create temporary DataFrame as nk.eda_findpeaks expects
        # Replace NaN with 0 for NeuroKit (it doesn't handle NaN well)
        phasic_clean = np.where(np.isnan(phasic), 0, phasic)
        signals = pd.DataFrame({"EDA_Phasic": phasic_clean})
        info = nk.eda_findpeaks(signals, sampling_rate=int(fs), method="neurokit")
        
        scr_onsets_raw = info.get("SCR_Onsets", [])
        scr_peaks_raw = info.get("SCR_Peaks", [])
        
        if len(scr_onsets_raw) == 0 or len(scr_peaks_raw) == 0:
            return scr_info
        
        # Filter out invalid indices
        valid_mask_onsets = [i >= 0 and i < len(phasic) and np.isfinite(i) for i in scr_onsets_raw]
        valid_mask_peaks = [i >= 0 and i < len(phasic) and np.isfinite(i) for i in scr_peaks_raw]
        
        scr_onsets = np.array([int(scr_onsets_raw[i]) for i in range(len(scr_onsets_raw)) if valid_mask_onsets[i]], dtype=int)
        scr_peaks = np.array([int(scr_peaks_raw[i]) for i in range(len(scr_peaks_raw)) if valid_mask_peaks[i]], dtype=int)
        
        # Filter by amplitude
        if len(scr_peaks) > 0 and len(scr_onsets) > 0:
            amplitudes = []
            valid_onsets = []
            valid_peaks = []
            valid_rise_times = []
            
            for i, peak_idx in enumerate(scr_peaks):
                if i < len(scr_onsets):
                    onset_idx = scr_onsets[i]
                    if 0 <= onset_idx < peak_idx < len(phasic) and not np.isnan(phasic[peak_idx]) and not np.isnan(phasic[onset_idx]):
                        amp = phasic[peak_idx] - phasic[onset_idx]
                        rise_time = (peak_idx - onset_idx) / fs
                        
                        if amp >= min_amplitude and np.isfinite(amp):
                            amplitudes.append(amp)
                            valid_onsets.append(onset_idx)
                            valid_peaks.append(peak_idx)
                            valid_rise_times.append(rise_time)
            
            scr_info["peaks"] = np.array(valid_peaks, dtype=int)
            scr_info["onsets"] = np.array(valid_onsets, dtype=int)
            scr_info["amplitudes"] = np.array(amplitudes)
            scr_info["rise_times"] = np.array(valid_rise_times)
    
    except Exception as e:
        # Print error for debugging
        import traceback
        print(f"    ⚠️ SCR detection failed: {e}")
        # Uncomment for full traceback:
        # traceback.print_exc()
        pass
    
    return scr_info


def calculate_phasic_parameters(
    phasic: np.ndarray,
    times: np.ndarray,
    scr_info: Dict[str, np.ndarray] = None,
    duration_s: float = None,
) -> Dict[str, float]:
    """
    Calculate phasic (SCR) parameters.
    
    Args:
        phasic: Phasic component (SCR) in microsiemens (µS)
        times: Time vector in seconds
        scr_info: Dictionary with 'peaks', 'onsets', 'amplitudes', 'rise_times'
        duration_s: Duration of segment in seconds (for frequency calculation)
    
    Returns:
        Dictionary with phasic parameters
    """
    params = {}
    
    # Initialize all SCR parameters with appropriate defaults
    params["scr_count"] = 0
    params["scr_frequency_rate"] = 0.0
    params["scr_amplitude_mean_raw"] = np.nan
    params["scr_amplitude_std_raw"] = np.nan
    params["scr_rise_time_mean"] = np.nan
    params["scr_rise_time_std"] = np.nan
    params["scr_amplitude_sqrt"] = np.nan
    params["scr_amplitude_log"] = np.nan
    # Z-scores and range correction are set separately
    params["scr_amplitude_z"] = np.nan
    params["scr_amplitude_range_corrected"] = np.nan
    
    # Check if we have valid SCR info
    if scr_info is None or len(scr_info.get("amplitudes", [])) == 0:
        return params
    
    amplitudes = scr_info.get("amplitudes", np.array([]))
    rise_times = scr_info.get("rise_times", np.array([]))
    
    # Filter valid amplitudes (non-NaN, positive, finite)
    valid_mask = np.isfinite(amplitudes) & (amplitudes > 0)
    amplitudes_valid = amplitudes[valid_mask]
    
    # Count and frequency
    n_scrs = len(amplitudes_valid)
    params["scr_count"] = n_scrs
    
    # Frequency: SCRs per minute
    if duration_s is not None and duration_s > 0:
        params["scr_frequency_rate"] = (n_scrs / duration_s) * 60.0  # SCRs per minute
    else:
        # Fallback: estimate from times if duration not provided
        if len(times) > 1:
            dur = times[-1] - times[0]
            params["scr_frequency_rate"] = (n_scrs / dur) * 60.0 if dur > 0 else 0.0
        else:
            params["scr_frequency_rate"] = 0.0
    
    if n_scrs == 0:
        return params
    
    # --- RAW AMPLITUDE (mean of only detected SCRs) ---
    params["scr_amplitude_mean_raw"] = np.mean(amplitudes_valid)
    params["scr_amplitude_std_raw"] = np.std(amplitudes_valid, ddof=1) if n_scrs > 1 else 0.0
    
    # --- RISE TIME ---
    if len(rise_times) > 0:
        if len(rise_times) == len(amplitudes):
            rise_times_valid = rise_times[valid_mask]
        else:
            rise_times_valid = rise_times
        
        rise_times_valid = rise_times_valid[np.isfinite(rise_times_valid) & (rise_times_valid > 0)]
        
        if len(rise_times_valid) > 0:
            params["scr_rise_time_mean"] = np.mean(rise_times_valid)
            params["scr_rise_time_std"] = np.std(rise_times_valid, ddof=1) if len(rise_times_valid) > 1 else 0.0
    
    # --- TRANSFORMED PARAMETERS ---
    # Square root transformation
    try:
        params["scr_amplitude_sqrt"] = np.mean(np.sqrt(amplitudes_valid))
    except Exception as e:
        params["scr_amplitude_sqrt"] = np.nan
    
    # Logarithmic transformation: log(SCR + 1)
    try:
        params["scr_amplitude_log"] = np.mean(np.log(amplitudes_valid + 1))
    except Exception as e:
        params["scr_amplitude_log"] = np.nan
    
    return params


def calculate_eda_parameters(
    eda: np.ndarray,
    tonic: np.ndarray,
    phasic: np.ndarray,
    times: np.ndarray,
    is_bad: np.ndarray = None,
    scr_info: Dict[str, np.ndarray] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Calculate all EDA parameters from cleaned data.
    
    Args:
        eda: Cleaned EDA signal in microsiemens (µS)
        tonic: Tonic component (SCL) in microsiemens (µS)
        phasic: Phasic component (SCR) in microsiemens (µS)
        times: Time vector in seconds
        is_bad: Boolean mask indicating bad samples (optional)
        scr_info: Dictionary with SCR peak information (optional)
        verbose: Print calculation progress
    
    Returns:
        Dictionary with all EDA parameters
    """
    # Initialize result dictionary with NaN
    params = {name: np.nan for name in EDA_PARAM_NAMES}
    
    # Validate input
    if eda is None or len(eda) == 0 or times is None or len(times) == 0:
        if verbose:
            print(f"  ⚠️ Insufficient data: {len(eda) if eda is not None else 0} samples")
        return params
    
    # Quality metrics
    params["n_samples"] = len(eda)
    params["duration_s"] = times[-1] - times[0] if len(times) > 1 else 0
    
    if is_bad is not None and len(is_bad) == len(eda):
        bad_count = np.sum(is_bad)
        params["bad_pct"] = (bad_count / len(eda)) * 100
    else:
        params["bad_pct"] = 0.0
    
    # QC validity (segment valid if < 20% bad data)
    params["qc_valid"] = 1 if params["bad_pct"] < 20.0 else 0
    
    # Tonic parameters
    tonic_params = calculate_tonic_parameters(tonic, times)
    params.update(tonic_params)
    
    if verbose:
        scl_mean = params['scl_mean']
        scl_slope = params['scl_slope']
        print(f"    ✓ Tonic: SCL_mean={scl_mean:.3f}µS, slope={scl_slope:.4f}µS/s")
    
    # Detect SCR peaks if not provided
    if scr_info is None:
        # Estimate sampling rate from time vector
        if len(times) > 1:
            time_diffs = np.diff(times)
            time_diffs = time_diffs[np.isfinite(time_diffs) & (time_diffs > 0)]
            if len(time_diffs) > 0:
                fs = 1.0 / np.median(time_diffs)
            else:
                fs = SAMPLING_RATE
        else:
            fs = SAMPLING_RATE
        
        if verbose:
            print(f"    → Detecting SCR peaks (fs={fs:.1f}Hz, neurokit={_HAVE_NK})...")
        
        scr_info = detect_scr_peaks(phasic, times, fs=fs)
        
        if verbose and len(scr_info.get("peaks", [])) == 0:
            print(f"    ⚠️ No SCR peaks detected (phasic_std={np.std(phasic[~np.isnan(phasic)]):.6f})")
    
    # Phasic parameters
    phasic_params = calculate_phasic_parameters(phasic, times, scr_info, duration_s=params["duration_s"])
    params.update(phasic_params)
    
    if verbose:
        n_scrs = params.get('scr_count', 0)
        amp_mean = params.get('scr_amplitude_mean_raw', np.nan)
        if n_scrs > 0:
            print(f"    ✓ Phasic: SCR_count={n_scrs}, amplitude_mean={amp_mean:.4f}µS")
        else:
            print(f"    ✓ Phasic: No SCRs detected (or phasic too flat)")
    
    return params


def get_bh_trimmed_duration(subject_id: str, bh_segment: str) -> Optional[float]:
    """
    Get the trimmed duration for a BH segment from RSP clean data.

    Reads the BH X_trimmed sheet written by rsp_preprocess_bh.py and returns
    the time span of that sheet (i.e. the true apnea period excluding the first
    recovery breath).

    Args:
        subject_id: Subject ID string (e.g. "42")
        bh_segment: BH segment name (e.g. "BH 1")

    Returns:
        Duration in seconds, or None if the trimmed sheet cannot be found/read.
    """
    try:
        rsp_file = Path(RSP_FOLDER) / f"RSP_data_sujeto_{subject_id}.xlsx"
        if not rsp_file.exists():
            return None
        rsp_xl = pd.ExcelFile(rsp_file)
        trimmed_sheet = f"{bh_segment}_trimmed"
        if trimmed_sheet not in rsp_xl.sheet_names:
            return None
        df = pd.read_excel(rsp_xl, sheet_name=trimmed_sheet)
        if "time_s" in df.columns and len(df) > 0:
            return float(df["time_s"].iloc[-1] - df["time_s"].iloc[0])
    except Exception as e:
        print(f"    [WARNING] Could not get BH_trimmed duration for {bh_segment}: {e}")
    return None


def load_segment_data(df: pd.DataFrame, scr_info_global: Dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load segment data from DataFrame.
    
    Args:
        df: DataFrame with columns: time_s, eda_us, tonic_us, phasic_us, is_bad, scr_peak
        scr_info_global: Global dictionary with SCR information (optional)
    
    Returns:
        Tuple of (times, eda, tonic, phasic, is_bad, scr_info)
    """
    times = df["time_s"].values if "time_s" in df.columns else np.array([])
    eda = df["eda_us"].values if "eda_us" in df.columns else np.array([])
    tonic = df["tonic_us"].values if "tonic_us" in df.columns else np.array([])
    phasic = df["phasic_us"].values if "phasic_us" in df.columns else np.array([])
    is_bad = df["is_bad"].values.astype(bool) if "is_bad" in df.columns else np.zeros(len(times), dtype=bool)
    
    # Extract SCR peak information from the scr_peak column if available
    scr_info = None
    if "scr_peak" in df.columns:
        scr_peak_marker = df["scr_peak"].values
        peak_indices = np.where(scr_peak_marker > 0)[0]
        
        if len(peak_indices) > 0 and len(phasic) > 0:
            # Calculate amplitudes and rise times from peak indices
            amplitudes = []
            onsets = []
            valid_peaks = []
            rise_times = []
            
            # Estimate sampling rate
            if len(times) > 1:
                time_diffs = np.diff(times)
                time_diffs = time_diffs[np.isfinite(time_diffs) & (time_diffs > 0)]
                fs = 1.0 / np.median(time_diffs) if len(time_diffs) > 0 else 32.0
            else:
                fs = 32.0
            
            for peak_idx in peak_indices:
                if peak_idx >= len(phasic) or np.isnan(phasic[peak_idx]):
                    continue
                
                # Look backward for onset (where phasic crosses near zero or reaches local minimum)
                onset_idx = peak_idx
                search_window = int(3.0 * fs)  # Look back up to 3 seconds
                for i in range(peak_idx - 1, max(0, peak_idx - search_window), -1):
                    if np.isnan(phasic[i]):
                        onset_idx = i + 1
                        break
                    if phasic[i] <= phasic[onset_idx]:
                        onset_idx = i
                    elif phasic[i] < phasic[onset_idx] * 0.5:  # Significant decrease
                        break
                
                # Calculate amplitude (peak - onset)
                if onset_idx < peak_idx and not np.isnan(phasic[onset_idx]):
                    amp = phasic[peak_idx] - phasic[onset_idx]
                    
                    # Only include if amplitude exceeds threshold
                    if amp >= SCR_MIN_AMPLITUDE_US:
                        amplitudes.append(amp)
                        onsets.append(onset_idx)
                        valid_peaks.append(peak_idx)
                        rise_times.append((peak_idx - onset_idx) / fs)
            
            if len(valid_peaks) > 0:
                scr_info = {
                    "peaks": np.array(valid_peaks, dtype=int),
                    "onsets": np.array(onsets, dtype=int),
                    "amplitudes": np.array(amplitudes),
                    "rise_times": np.array(rise_times),
                }
    
    # Fallback to provided scr_info_global if available
    if scr_info is None and scr_info_global is not None:
        scr_info = scr_info_global
    
    return times, eda, tonic, phasic, is_bad, scr_info


def load_subject_data(excel_path: str, verbose: bool = False) -> Dict[str, Dict]:
    """
    Load all segment data from a subject's Excel file.
    
    Returns a dictionary with segment names as keys and data dictionaries as values.
    """
    segments_data = {}
    
    try:
        xl = pd.ExcelFile(excel_path)
        
        for sheet in xl.sheet_names:
            if sheet.upper() == "RESUMEN":
                continue
            
            df = pd.read_excel(xl, sheet_name=sheet)
            
            # Check required columns
            required_cols = ["time_s", "eda_us", "tonic_us", "phasic_us"]
            if not all(col in df.columns for col in required_cols):
                if verbose:
                    print(f"    ⚠️ Sheet '{sheet}': missing required columns")
                continue
            
            times, eda, tonic, phasic, is_bad, scr_info = load_segment_data(df)
            
            if len(times) > 0:
                # Cap LB / Recovery at 300s to ensure consistent downstream analysis even if
                # older exports contain longer segments.
                sheet_norm = sheet.strip().lower().replace(" ", "")
                if sheet_norm in ("lb", "recovery"):
                    t0 = float(times[0])
                    t_cap = t0 + 300.0
                    keep_mask = times <= t_cap
                    if np.any(keep_mask) and not np.all(keep_mask):
                        n_keep = int(np.sum(keep_mask))
                        times = times[:n_keep]
                        eda = eda[:n_keep]
                        tonic = tonic[:n_keep]
                        phasic = phasic[:n_keep]
                        is_bad = is_bad[:n_keep]

                        # scr_info uses sample indices; drop peaks/onsets that fall outside trimmed range
                        if scr_info is not None and "peaks" in scr_info:
                            peaks = np.asarray(scr_info.get("peaks", []), dtype=int)
                            onsets = np.asarray(scr_info.get("onsets", []), dtype=int)
                            mask_peaks = peaks < n_keep
                            scr_info = {
                                **scr_info,
                                "peaks": peaks[mask_peaks],
                                "onsets": onsets[mask_peaks] if len(onsets) == len(peaks) else onsets,
                                "amplitudes": np.asarray(scr_info.get("amplitudes", []))[mask_peaks] if len(np.asarray(scr_info.get("amplitudes", []))) == len(peaks) else scr_info.get("amplitudes", []),
                                "rise_times": np.asarray(scr_info.get("rise_times", []))[mask_peaks] if len(np.asarray(scr_info.get("rise_times", []))) == len(peaks) else scr_info.get("rise_times", []),
                            }

                segments_data[sheet] = {
                    "times": times,
                    "eda": eda,
                    "tonic": tonic,
                    "phasic": phasic,
                    "is_bad": is_bad,
                    "scr_info": scr_info,
                }
                if verbose and scr_info is not None and len(scr_info.get("peaks", [])) > 0:
                    print(f"    ✓ '{sheet}': Loaded {len(scr_info['peaks'])} pre-detected SCR peaks")
            else:
                if verbose:
                    print(f"    ⚠️ Sheet '{sheet}': no valid data")
        
    except Exception as e:
        print(f"  ✗ Error loading {excel_path}: {e}")
    
    return segments_data


def find_subject_files(input_folder: str) -> List[Tuple[str, str]]:
    """Find all subject Excel files in the input folder."""
    folder = Path(input_folder)
    subjects = []
    
    if not folder.exists():
        print(f"⚠️ Input folder not found: {input_folder}")
        return subjects
    
    for f in folder.glob("EDA_data_sujeto_*.xlsx"):
        name = f.stem
        parts = name.split("_")
        if len(parts) >= 4:
            subject_id = parts[-1]
            subjects.append((subject_id, str(f)))
    
    def sort_key(x):
        try:
            return int(x[0])
        except ValueError:
            return x[0]
    
    subjects.sort(key=sort_key)
    return subjects


def standardize_parameters_per_subject(
    df_results: pd.DataFrame,
    segments: List[str],
) -> pd.DataFrame:
    """
    Calculate Z-scores and range-corrected parameters per subject.
    
    Z-scores: (Value - Subject_Mean) / Subject_SD
    Range correction: Value / Subject_Max
    
    NOTE: Mean and SD are calculated using ALL data from the subject, not just baseline.
    """
    # Initialize Z-score and range correction columns
    for seg_name in segments:
        df_results[f"{seg_name}_scr_amplitude_z"] = np.nan
        df_results[f"{seg_name}_scr_amplitude_range_corrected"] = np.nan
    
    for subject_id in df_results["subject"].unique():
        subject_mask = df_results["subject"] == subject_id
        
        # Collect all amplitude values for this subject across all segments
        amplitude_values = []
        
        for seg_name in segments:
            amp_col = f"{seg_name}_scr_amplitude_mean_raw"
            
            if amp_col in df_results.columns:
                amp_val = df_results.loc[subject_mask, amp_col].values[0]
                if np.isfinite(amp_val) and amp_val > 0:  # Only positive, finite values
                    amplitude_values.append(amp_val)
        
        # Calculate subject-level statistics only if we have enough data
        if len(amplitude_values) >= 2:  # Need at least 2 values for meaningful standardization
            amp_mean = np.mean(amplitude_values)
            amp_std = np.std(amplitude_values, ddof=1)
            amp_max = np.max(amplitude_values)
            
            # Only calculate if std > 0 (i.e., there's variation)
            if amp_std > 0:
                # Calculate Z-scores and range correction for each segment
                for seg_name in segments:
                    amp_col = f"{seg_name}_scr_amplitude_mean_raw"
                    
                    if amp_col in df_results.columns:
                        amp_val = df_results.loc[subject_mask, amp_col].values[0]
                        
                        # Z-score
                        z_col = f"{seg_name}_scr_amplitude_z"
                        if np.isfinite(amp_val):
                            df_results.loc[subject_mask, z_col] = (amp_val - amp_mean) / amp_std
                        
                        # Range correction
                        range_col = f"{seg_name}_scr_amplitude_range_corrected"
                        if np.isfinite(amp_val) and amp_max > 0:
                            df_results.loc[subject_mask, range_col] = amp_val / amp_max
    
    return df_results


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_all_subjects(
    input_folder: str = INPUT_FOLDER,
    output_file: str = OUTPUT_FILE,
    segments: List[str] = SEGMENTS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Process all subjects and calculate EDA parameters."""
    
    print("=" * 70)
    print("EDA ANALYSIS - CALCULATING PARAMETERS FROM CLEANED DATA")
    print("=" * 70)
    print(f"\nInput folder: {input_folder}")
    print(f"Output file: {output_file}")
    print(f"Segments: {len(segments)}")
    print(f"Parameters per segment: {len(EDA_PARAM_NAMES)}")
    
    subject_files = find_subject_files(input_folder)
    
    if not subject_files:
        print("\n⚠️ No subject files found!")
        return pd.DataFrame()
    
    print(f"\nFound {len(subject_files)} subjects: {[s[0] for s in subject_files]}")
    
    all_results = []
    
    for subject_id, file_path in subject_files:
        print(f"\n{'─'*70}")
        print(f"Processing Subject {subject_id}")
        print(f"{'─'*70}")
        
        segments_data = load_subject_data(file_path, verbose=verbose)
        
        if not segments_data:
            print(f"  ⚠️ No valid segment data found")
            continue
        
        print(f"  Loaded segments: {list(segments_data.keys())}")
        
        result_row = {"subject": subject_id}
        
        for seg_name in segments:
            # Find matching segment
            seg_key = None
            for key in segments_data.keys():
                if key.lower().replace(" ", "") == seg_name.lower().replace(" ", ""):
                    seg_key = key
                    break
            
            if seg_key is None:
                if verbose:
                    print(f"  [{seg_name}]: Not available")
                for param in EDA_PARAM_NAMES:
                    result_row[f"{seg_name}_{param}"] = np.nan
                continue
            
            seg_data = segments_data[seg_key]
            
            if verbose:
                print(f"  [{seg_name}]: {len(seg_data['times'])} samples, {seg_data['times'][-1]-seg_data['times'][0]:.1f}s")
            
            eda_params = calculate_eda_parameters(
                eda=seg_data["eda"],
                tonic=seg_data["tonic"],
                phasic=seg_data["phasic"],
                times=seg_data["times"],
                is_bad=seg_data["is_bad"],
                scr_info=seg_data["scr_info"],
                verbose=verbose,
            )
            
            for param, value in eda_params.items():
                result_row[f"{seg_name}_{param}"] = value

        # ── BH segments: load EDA and trim to RSP-based apnea duration ──────
        xl_eda = pd.ExcelFile(file_path)
        for bh_seg_name in BH_SEGMENTS:
            # Find matching BH sheet in EDA data
            seg_key = next(
                (k for k in segments_data
                 if k.lower().replace(" ", "") == bh_seg_name.lower().replace(" ", "")),
                None,
            )
            if seg_key is None:
                if verbose:
                    print(f"  [{bh_seg_name}]: Not available in EDA data")
                for param in EDA_PARAM_NAMES:
                    result_row[f"{bh_seg_name}_{param}"] = np.nan
                continue

            # Get RSP-based trimmed duration (true apnea period)
            trimmed_duration = get_bh_trimmed_duration(subject_id, bh_seg_name)

            # Load raw DataFrame so we can trim by time
            df_bh = pd.read_excel(xl_eda, sheet_name=seg_key)

            if trimmed_duration is not None and "time_s" in df_bh.columns:
                t0 = df_bh["time_s"].iloc[0]
                df_bh = df_bh[df_bh["time_s"] <= t0 + trimmed_duration].copy()
                if verbose:
                    print(f"  [{bh_seg_name}]: trimmed to {trimmed_duration:.1f}s (RSP-based)")
            else:
                if verbose:
                    if trimmed_duration is None:
                        print(f"  [{bh_seg_name}]: no RSP trimming info, using full BH duration")
                    else:
                        print(f"  [{bh_seg_name}]: no time_s column, using full BH duration")

            # Re-derive scr_info from trimmed DataFrame (indices are local to trimmed array)
            times, eda, tonic, phasic, is_bad, scr_info = load_segment_data(df_bh)

            if verbose and len(times) > 0:
                print(f"  [{bh_seg_name}]: {len(times)} samples, {times[-1] - times[0]:.1f}s")

            eda_params = calculate_eda_parameters(
                eda=eda,
                tonic=tonic,
                phasic=phasic,
                times=times,
                is_bad=is_bad,
                scr_info=scr_info,
                verbose=verbose,
            )
            for param, value in eda_params.items():
                result_row[f"{bh_seg_name}_{param}"] = value

        all_results.append(result_row)
    
    if not all_results:
        print("\n⚠️ No results to save!")
        return pd.DataFrame()
    
    df_results = pd.DataFrame(all_results)
    
    # Calculate subject-level standardization (Z-scores and range correction)
    # Include BH segments so Z-scores are computed across the full session.
    print(f"\n{'='*70}")
    print("CALCULATING SUBJECT-LEVEL STANDARDIZATION")
    print(f"{'='*70}")
    all_segments_for_std = list(segments) + BH_SEGMENTS
    df_results = standardize_parameters_per_subject(df_results, all_segments_for_std)
    print("✓ Z-scores and range correction calculated")
    
    # Reorder columns: standard segments → BH segments → any remainder
    columns_ordered = ["subject"]
    for seg_name in segments:
        for param in EDA_PARAM_NAMES:
            col_name = f"{seg_name}_{param}"
            if col_name in df_results.columns:
                columns_ordered.append(col_name)
    for bh_seg_name in BH_SEGMENTS:
        for param in EDA_PARAM_NAMES:
            col_name = f"{bh_seg_name}_{param}"
            if col_name in df_results.columns:
                columns_ordered.append(col_name)
    for col in df_results.columns:
        if col not in columns_ordered:
            columns_ordered.append(col)

    df_results = df_results[columns_ordered]
    
    # Save to Excel
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="EDA_Parameters", index=False)
        
        # Summary sheet (key parameters in long format — all segments including BH)
        summary_params = [
            "scl_mean", "scl_slope", "scr_count", "scr_frequency_rate",
            "scr_amplitude_mean_raw", "scr_rise_time_mean", "bad_pct", "qc_valid"
        ]
        summary_data = []
        all_segments_for_summary = list(segments) + BH_SEGMENTS

        for _, row in df_results.iterrows():
            subject = row["subject"]
            for seg_name in all_segments_for_summary:
                seg_row = {"subject": subject, "segment": seg_name}
                for param in summary_params:
                    col_name = f"{seg_name}_{param}"
                    seg_row[param] = row.get(col_name, np.nan)
                summary_data.append(seg_row)

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

        # Parameter definitions
        param_definitions = pd.DataFrame({
            "parameter": EDA_PARAM_NAMES,
            "description": [
                # Tonic (SCL)
                "Mean skin conductance level (µS)",
                "Standard deviation of skin conductance level (µS)",
                "Minimum skin conductance level (µS)",
                "Maximum skin conductance level (µS)",
                "Range of skin conductance level (max - min) (µS)",
                "Drift/slope of tonic component (µS/s)",
                
                # Phasic (SCR) - raw
                "Number of valid skin conductance responses (SCRs)",
                "SCR frequency rate (SCRs per minute)",
                "Mean SCR amplitude - only detected SCRs (µS)",
                "Standard deviation of SCR amplitude (µS)",
                "Mean SCR rise time (s)",
                "Standard deviation of SCR rise time (s)",
                
                # Transformed
                "Square root transformed mean amplitude (√µS)",
                "Log transformed mean amplitude (log[µS+1])",
                
                # Standardized
                "Z-score of amplitude (normalized by subject mean & SD)",
                
                # Range corrected
                "Amplitude normalized by subject's maximum response",
                
                # Quality
                "Duration of segment (s)",
                "Percentage of bad/rejected data (%)",
                "Number of samples in segment",
                "Quality control valid flag (1=valid, 0=invalid)",
            ],
        })
        param_definitions.to_excel(writer, sheet_name="Parameter_Definitions", index=False)

        # Notes sheet
        notes = pd.DataFrame({
            "Topic": [
                "BH EDA trimming",
            ],
            "Explanation": [
                "BH 1-4 EDA parameters are computed on the true apnea period only. "
                "The duration is determined by the BH X_trimmed sheet in RSP_clean_data "
                "(written by rsp_preprocess_bh.py), which ends at the first recovery breath. "
                "If the trimmed RSP sheet is unavailable, the full BH EDA segment is used as fallback. "
                "This matches the trimming applied in hrv_analysis_final.py so that HRV and EDA "
                "BH parameters cover the same time window.",
            ],
        })
        notes.to_excel(writer, sheet_name="Notes", index=False)

    print(f"\n✔ Results saved to: {output_file}")
    print(f"  • Sheet 'EDA_Parameters': {len(df_results)} subjects × {len(df_results.columns)-1} parameters")
    print(f"  • Sheet 'Summary': Key parameters in long format")
    print(f"  • Sheet 'Parameter_Definitions': Parameter descriptions")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"\nSubjects processed: {len(df_results)}")
    print(f"Segments: {len(segments)}")
    
    all_segs_for_stats = list(segments) + BH_SEGMENTS
    print(f"\nData availability by segment:")
    for seg_name in all_segs_for_stats:
        col_check = f"{seg_name}_duration_s"
        if col_check in df_results.columns:
            n_available = df_results[col_check].notna().sum()
            print(f"  {seg_name}: {n_available}/{len(df_results)} subjects")

    print(f"\nQuality check (segments with < 20% bad data):")
    for seg_name in all_segs_for_stats:
        col_check = f"{seg_name}_qc_valid"
        if col_check in df_results.columns:
            n_valid = df_results[col_check].sum()
            n_total = df_results[col_check].notna().sum()
            if n_total > 0:
                print(f"  {seg_name}: {n_valid}/{n_total} valid ({n_valid/n_total*100:.1f}%)")
    
    return df_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if NeuroKit2 is available
    if not _HAVE_NK:
        print("\n⚠️ WARNING: NeuroKit2 not found! SCR detection will not work.")
        print("   Install with: pip install neurokit2")
        print("   SCR-related parameters will be empty.\n")
    else:
        print(f"✓ NeuroKit2 version {nk.__version__} available\n")
    
    df_final = process_all_subjects(
        input_folder=INPUT_FOLDER,
        output_file=OUTPUT_FILE,
        segments=SEGMENTS,
        verbose=True,
    )
