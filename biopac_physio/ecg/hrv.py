"""
HRV Analysis Script - Calculate HRV parameters from cleaned ECG data.

This script reads the cleaned RR interval data exported by ecg_preprocess_bh.py
and calculates comprehensive HRV parameters.

Output: Final Excel file with one row per subject, columns for each segment's HRV parameters.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import signal

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input folder containing cleaned HRV Excel files (one per subject)
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = str(SCRIPT_DIR / "HRV_clean_data")
RSP_FOLDER = str(SCRIPT_DIR.parent / "rsp_bh" / "RSP_clean_data")  # For BH_trimmed timing

# Output file for final HRV parameters
OUTPUT_FILE = str(SCRIPT_DIR / "HRV_parametros_finales.xlsx")

# Segments to process (in order)
# Trim LB/Recovery RR series to their first 300 s before computing HRV (BH: True)
TRIM_BASELINE_TO_300S = True

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

# BH segments to process with RSP-based trimming (until first breath after breath-hold)
BH_SEGMENTS = [
    "BH 1",
    "BH 2",
    "BH 3",
    "BH 4",
]

# Fixed-duration windows used for cross-subject comparisons (60 seconds each)
WINDOW_BASE_METRICS = [
    "mean_hr",
    "hr_slope_bpm_per_min",
    "rmssd",
    "ln_rmssd",
    "sd1",
    "sdnn",
]
WINDOW_FREQ_METRICS = [
    "vlf",
    "lf",
    "hf",
    "total_power",
    "lf_hf_ratio",
    "lf_nu",
    "hf_nu",
]
FIXED_WINDOWS = [
    {"name": "LB_60s", "segment": "LB", "start_s": 180.0, "end_s": 240.0, "include_freq": True},
    {"name": "Recovery_60s", "segment": "Recovery", "start_s": 180.0, "end_s": 240.0, "include_freq": True},
    {"name": "CPT_60s", "segment": "CPT", "start_s": 10.0, "end_s": 70.0, "include_freq": False},
]
# BH 60-second window configuration (applied within BH_trimmed segments)
BH_60S_START = 10.0
BH_60S_END = 70.0

# HRV parameter names
HRV_PARAM_NAMES = [
    # Time domain
    "mean_nn", "sdnn", "sdsd", "rmssd",
    "nn50", "pnn50", "nn20", "pnn20",
    "median_nn", "range_nn",
    "cvnn", "cvsd",
    "mean_hr", "max_hr", "min_hr", "std_hr",
    # Geometrical
    "triangular_index",
    # Frequency domain
    "vlf", "lf", "hf", "total_power",
    "lf_hf_ratio", "lf_nu", "hf_nu",
    # Poincaré
    "sd1", "sd2", "sd1_sd2_ratio",
    # CSI/CVI
    "csi", "cvi", "csi_modified",
    # Entropy
    "sample_entropy",
    # Count
    "n_rr",
]


# ============================================================================
# HRV CALCULATION FUNCTIONS
# ============================================================================

def calculate_time_domain(rr_ms: np.ndarray) -> Dict[str, float]:
    """Calculate time domain HRV parameters."""
    params = {}
    
    # Basic statistics
    params["mean_nn"] = np.mean(rr_ms)
    params["sdnn"] = np.std(rr_ms, ddof=1)
    params["median_nn"] = np.median(rr_ms)
    params["range_nn"] = np.max(rr_ms) - np.min(rr_ms)
    
    # Successive differences
    diff_rr = np.diff(rr_ms)
    params["sdsd"] = np.std(diff_rr, ddof=1)
    params["rmssd"] = np.sqrt(np.mean(diff_rr ** 2))
    
    # NN50 and pNN50
    nn50 = np.sum(np.abs(diff_rr) > 50)
    params["nn50"] = nn50
    params["pnn50"] = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
    
    # NN20 and pNN20
    nn20 = np.sum(np.abs(diff_rr) > 20)
    params["nn20"] = nn20
    params["pnn20"] = (nn20 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
    
    # Coefficients of variation
    params["cvnn"] = (params["sdnn"] / params["mean_nn"]) * 100 if params["mean_nn"] > 0 else 0
    params["cvsd"] = (params["rmssd"] / params["mean_nn"]) * 100 if params["mean_nn"] > 0 else 0
    
    # Heart rate statistics (convert RR in ms to HR in bpm)
    hr = 60000.0 / rr_ms
    params["mean_hr"] = np.mean(hr)
    params["max_hr"] = np.max(hr)
    params["min_hr"] = np.min(hr)
    params["std_hr"] = np.std(hr, ddof=1)
    
    return params


def calculate_geometrical(rr_ms: np.ndarray) -> Dict[str, float]:
    """Calculate geometrical HRV parameters."""
    params = {}
    
    # Triangular index: total number of NNs / max of histogram
    # Use 1/128 second bins (approximately 7.8125 ms)
    bin_width = 1000 / 128  # ~7.8125 ms
    bins = np.arange(np.min(rr_ms) - bin_width, np.max(rr_ms) + bin_width * 2, bin_width)
    
    if len(bins) > 1:
        hist, _ = np.histogram(rr_ms, bins=bins)
        max_hist = np.max(hist)
        params["triangular_index"] = len(rr_ms) / max_hist if max_hist > 0 else 0
    else:
        params["triangular_index"] = 0
    
    return params


def calculate_frequency_domain(rr_ms: np.ndarray, rr_times: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate frequency domain HRV parameters using Welch's method.
    
    Args:
        rr_ms: RR intervals in milliseconds
        rr_times: Cumulative times of RR intervals in seconds (optional)
    """
    params = {
        "vlf": np.nan, "lf": np.nan, "hf": np.nan, "total_power": np.nan,
        "lf_hf_ratio": np.nan, "lf_nu": np.nan, "hf_nu": np.nan
    }
    
    if len(rr_ms) < 20:
        return params
    
    try:
        # Convert RR to seconds
        rr_s = rr_ms / 1000.0
        
        # Create time vector if not provided
        if rr_times is None:
            rr_times = np.cumsum(rr_s)
            rr_times = rr_times - rr_times[0]  # Start from 0
        
        # Interpolate to regular sampling (4 Hz is standard for HRV)
        fs = 4.0  # Hz
        t_interp = np.arange(0, rr_times[-1], 1/fs)
        
        if len(t_interp) < 16:
            return params

        # Ensure strictly increasing times (drop duplicate cumsum timestamps
        # caused by zero/near-zero RR intervals)
        valid_mask = np.diff(rr_times, prepend=-np.inf) > 0
        rr_times = rr_times[valid_mask]
        rr_s = rr_s[valid_mask]
        if len(rr_times) < 4:
            return params

        # Cubic spline interpolation
        f_interp = interpolate.interp1d(rr_times, rr_s, kind='cubic', fill_value='extrapolate')
        rr_interp = f_interp(t_interp)
        
        # Remove mean (detrend)
        rr_interp = rr_interp - np.mean(rr_interp)
        
        # Welch's method for PSD estimation
        # nperseg should be at least 256 samples (64 seconds at 4 Hz)
        nperseg = min(256, len(rr_interp))
        
        freqs, psd = signal.welch(rr_interp, fs=fs, nperseg=nperseg, 
                                   noverlap=nperseg//2, window='hann')
        
        # Convert PSD from s² to ms²
        psd = psd * 1e6
        
        # Define frequency bands
        vlf_band = (0.003, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        
        # Calculate band powers (integrate PSD)
        df = freqs[1] - freqs[0]
        
        vlf_mask = (freqs >= vlf_band[0]) & (freqs < vlf_band[1])
        lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
        hf_mask = (freqs >= hf_band[0]) & (freqs < hf_band[1])
        
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        vlf = _trapz(psd[vlf_mask], freqs[vlf_mask]) if np.any(vlf_mask) else 0
        lf = _trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
        hf = _trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
        
        params["vlf"] = vlf
        params["lf"] = lf
        params["hf"] = hf
        params["total_power"] = vlf + lf + hf
        
        # LF/HF ratio
        params["lf_hf_ratio"] = lf / hf if hf > 0 else np.nan
        
        # Normalized units (exclude VLF)
        lf_hf_sum = lf + hf
        params["lf_nu"] = (lf / lf_hf_sum) * 100 if lf_hf_sum > 0 else np.nan
        params["hf_nu"] = (hf / lf_hf_sum) * 100 if lf_hf_sum > 0 else np.nan
        
    except Exception as e:
        warnings.warn(f"[calculate_frequency_domain] failed: {e}")

    return params


def calculate_poincare(rr_ms: np.ndarray) -> Dict[str, float]:
    """Calculate Poincaré plot parameters (SD1, SD2)."""
    params = {}
    
    if len(rr_ms) < 2:
        params["sd1"] = np.nan
        params["sd2"] = np.nan
        params["sd1_sd2_ratio"] = np.nan
        return params
    
    # Poincaré plot: RR(n) vs RR(n+1)
    rr_n = rr_ms[:-1]
    rr_n1 = rr_ms[1:]
    
    # SD1: std of points perpendicular to line of identity
    # SD1 = sqrt(0.5) * SDSD
    diff_rr = rr_n1 - rr_n
    sd1 = np.sqrt(0.5) * np.std(diff_rr, ddof=1)
    
    # SD2: std of points along line of identity  
    # SD2 = sqrt(2*SDNN² - 0.5*SDSD²)
    sdnn = np.std(rr_ms, ddof=1)
    sdsd = np.std(diff_rr, ddof=1)
    sd2_squared = 2 * sdnn**2 - 0.5 * sdsd**2
    sd2 = np.sqrt(sd2_squared) if sd2_squared > 0 else 0
    
    params["sd1"] = sd1
    params["sd2"] = sd2
    params["sd1_sd2_ratio"] = sd1 / sd2 if sd2 > 0 else np.nan
    
    return params


def calculate_csi_cvi(sd1: float, sd2: float) -> Dict[str, float]:
    """Calculate CSI (Cardiac Sympathetic Index) and CVI (Cardiac Vagal Index)."""
    params = {}
    
    if np.isfinite(sd1) and np.isfinite(sd2) and sd1 > 0 and sd2 > 0:
        # CSI = SD2 / SD1 (reflects sympathetic activity)
        params["csi"] = sd2 / sd1
        # CVI = log10(SD1 * SD2) (reflects vagal activity)
        params["cvi"] = np.log10(sd1 * sd2)
        # Modified CSI = SD2² / SD1
        params["csi_modified"] = (sd2 ** 2) / sd1
    else:
        params["csi"] = np.nan
        params["cvi"] = np.nan
        params["csi_modified"] = np.nan
    
    return params


def calculate_sample_entropy(rr_ms: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Calculate Sample Entropy of RR intervals.
    
    Args:
        rr_ms: RR intervals in milliseconds
        m: Embedding dimension (default 2)
        r_factor: Tolerance factor (default 0.2 * std)
    """
    if len(rr_ms) < 20:
        return np.nan
    
    try:
        N = len(rr_ms)
        r = r_factor * np.std(rr_ms, ddof=1)
        
        def count_matches(template_length):
            count = 0
            for i in range(N - template_length):
                for j in range(i + 1, N - template_length):
                    # Check if templates match within tolerance r
                    diff = np.abs(rr_ms[i:i+template_length] - rr_ms[j:j+template_length])
                    if np.max(diff) < r:
                        count += 1
            return count
        
        # Count matches for m and m+1
        A = count_matches(m + 1)
        B = count_matches(m)
        
        if B == 0 or A == 0:
            return np.nan
        
        return -np.log(A / B)
    
    except Exception:
        return np.nan


def calculate_hrv_parameters(rr_ms: np.ndarray, verbose: bool = False) -> Dict[str, float]:
    """
    Calculate all HRV parameters from RR intervals.
    
    Args:
        rr_ms: Array of RR intervals in milliseconds
        verbose: Print calculation progress
    
    Returns:
        Dictionary with all HRV parameters.
    """
    # Initialize result dictionary with NaN
    params = {name: np.nan for name in HRV_PARAM_NAMES}
    
    # Validate input
    if rr_ms is None or len(rr_ms) < 10:
        if verbose:
            print(f"  [WARNING] Insufficient data: {len(rr_ms) if rr_ms is not None else 0} RR intervals")
        return params
    
    # Store RR count
    params["n_rr"] = len(rr_ms)
    
    # Time domain
    time_params = calculate_time_domain(rr_ms)
    params.update(time_params)
    
    if verbose:
        print(f"    [OK] Time domain: mean_nn={params['mean_nn']:.1f}ms, sdnn={params['sdnn']:.1f}ms, rmssd={params['rmssd']:.1f}ms")
    
    # Geometrical
    geom_params = calculate_geometrical(rr_ms)
    params.update(geom_params)
    
    if verbose:
        print(f"    [OK] Geometrical: triangular_index={params['triangular_index']:.2f}")
    
    # Frequency domain
    freq_params = calculate_frequency_domain(rr_ms)
    params.update(freq_params)
    
    if verbose:
        lf = params.get('lf', np.nan)
        hf = params.get('hf', np.nan)
        lfhf = params.get('lf_hf_ratio', np.nan)
        print(f"    [OK] Frequency: LF={lf:.1f}ms2, HF={hf:.1f}ms2, LF/HF={lfhf:.2f}")
    
    # Poincaré
    poincare_params = calculate_poincare(rr_ms)
    params.update(poincare_params)
    
    if verbose:
        print(f"    [OK] Poincare: SD1={params['sd1']:.1f}ms, SD2={params['sd2']:.1f}ms")
    
    # CSI/CVI
    csi_params = calculate_csi_cvi(params["sd1"], params["sd2"])
    params.update(csi_params)
    
    # Sample Entropy (can be slow for long segments)
    if len(rr_ms) <= 500:  # Only for reasonable length segments
        params["sample_entropy"] = calculate_sample_entropy(rr_ms)
    else:
        # Use subset for very long segments
        params["sample_entropy"] = calculate_sample_entropy(rr_ms[:500])
    
    if verbose and np.isfinite(params["sample_entropy"]):
        print(f"    [OK] Entropy: SampEn={params['sample_entropy']:.3f}")
    
    return params


def find_segment_key(segments_data: Dict[str, Dict[str, np.ndarray]], target_segment: str) -> Optional[str]:
    """Return the matching segment key ignoring case and spaces."""
    normalized_target = target_segment.lower().replace(" ", "")
    for key in segments_data.keys():
        if key.lower().replace(" ", "") == normalized_target:
            return key
    return None


def extract_rr_window(rr_ms: np.ndarray, start_s: float, end_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return RR intervals and relative times inside a fixed window.
    """
    if rr_ms is None or len(rr_ms) == 0 or start_s >= end_s:
        return np.array([], dtype=float), np.array([], dtype=float)

    cumsum_s = np.cumsum(rr_ms) / 1000.0
    if cumsum_s[-1] <= start_s:
        return np.array([], dtype=float), np.array([], dtype=float)

    start_idx = np.searchsorted(cumsum_s, start_s, side="right")
    end_idx = np.searchsorted(cumsum_s, end_s, side="right")

    rr_window = rr_ms[start_idx:end_idx]
    window_times = cumsum_s[start_idx:end_idx] - start_s

    return rr_window, window_times


def calculate_hr_slope(hr_values: np.ndarray, window_times: np.ndarray) -> float:
    """Fit HR(t) = a + b*t and return slope in bpm/min."""
    if len(hr_values) < 2 or len(window_times) != len(hr_values):
        return np.nan

    try:
        slope = np.polyfit(window_times, hr_values, 1)[0]
        return slope * 60.0  # convert bpm/s to bpm/min
    except Exception:
        return np.nan


def calculate_window_metrics(
    rr_ms: np.ndarray,
    window_times: np.ndarray,
    include_freq: bool = False,
) -> Dict[str, float]:
    """
    Calculate selected HRV metrics for short fixed-duration windows.
    """
    metrics = {name: np.nan for name in WINDOW_BASE_METRICS}
    freq_metrics = {name: np.nan for name in WINDOW_FREQ_METRICS}

    if rr_ms is None or len(rr_ms) < 10:
        requested = WINDOW_BASE_METRICS + (WINDOW_FREQ_METRICS if include_freq else [])
        combined = {**metrics, **freq_metrics}
        return {key: combined[key] for key in requested}

    time_params = calculate_time_domain(rr_ms)
    metrics["mean_hr"] = time_params["mean_hr"]
    metrics["sdnn"] = time_params["sdnn"]
    metrics["rmssd"] = time_params["rmssd"]
    metrics["ln_rmssd"] = np.log(metrics["rmssd"]) if metrics["rmssd"] > 0 else np.nan

    poincare_params = calculate_poincare(rr_ms)
    metrics["sd1"] = poincare_params.get("sd1", np.nan)

    hr_values = 60000.0 / rr_ms
    metrics["hr_slope_bpm_per_min"] = calculate_hr_slope(hr_values, window_times)

    if include_freq:
        freq_params = calculate_frequency_domain(rr_ms)
        for key in WINDOW_FREQ_METRICS:
            freq_metrics[key] = freq_params.get(key, np.nan)

    combined = {**metrics, **freq_metrics}
    requested = WINDOW_BASE_METRICS + (WINDOW_FREQ_METRICS if include_freq else [])
    return {key: combined[key] for key in requested}


def load_subject_data(excel_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load all segment RR data from a subject's Excel file.
    
    Returns dictionary with segment names as keys, each containing:
    - 'full': full segment RR intervals
    - 'duration_s': total duration of segment
    """
    segments_data = {}
    
    try:
        xl = pd.ExcelFile(excel_path)
        
        for sheet in xl.sheet_names:
            if sheet.upper() == "RESUMEN":
                continue
            
            df = pd.read_excel(xl, sheet_name=sheet)
            
            if "rr_ms" not in df.columns:
                print(f"    [WARNING] Sheet '{sheet}': no 'rr_ms' column")
                continue
            
            # Get RR intervals and times
            rr_ms = df["rr_ms"].values
            rr_ms = rr_ms[~np.isnan(rr_ms)]
            
            if len(rr_ms) == 0:
                print(f"    [WARNING] Sheet '{sheet}': no valid RR data")
                continue
            
            # Calculate total duration
            duration_s = np.sum(rr_ms) / 1000.0  # Convert ms to seconds

            # Cap LB / Recovery at 300s (trim from end) for consistency across exports
            # This protects analysis if older preprocess runs saved LB longer than 300s.
            sheet_norm = sheet.strip().lower().replace(" ", "")
            if TRIM_BASELINE_TO_300S and sheet_norm in ("lb", "recovery") and duration_s > 300.0:
                cumsum_s = np.cumsum(rr_ms) / 1000.0
                idx_300 = np.searchsorted(cumsum_s, 300.0, side="right")
                rr_ms = rr_ms[:idx_300]
                duration_s = float(np.sum(rr_ms) / 1000.0)
            
            # Store full segment
            segments_data[sheet] = {
                'full': rr_ms,
                'duration_s': duration_s,
            }
        
    except Exception as e:
        print(f"  [ERROR] Error loading {excel_path}: {e}")
    
    return segments_data


def get_bh_trimmed_duration(subject_id: str, bh_segment: str) -> float:
    """
    Get the trimmed duration for a BH segment from RSP data.
    
    Args:
        subject_id: Subject ID
        bh_segment: BH segment name (e.g., "BH 1")
    
    Returns:
        Duration in seconds until first breath after breath-hold, or None if not found
    """
    try:
        rsp_file = Path(RSP_FOLDER) / f"RSP_data_sujeto_{subject_id}.xlsx"
        if not rsp_file.exists():
            return None
        
        rsp_xl = pd.ExcelFile(rsp_file)
        trimmed_sheet = f"{bh_segment}_trimmed"
        
        if trimmed_sheet not in rsp_xl.sheet_names:
            return None
        
        # Read the trimmed segment to get duration
        df = pd.read_excel(rsp_xl, sheet_name=trimmed_sheet)
        if 'time_s' in df.columns and len(df) > 0:
            duration = df['time_s'].iloc[-1] - df['time_s'].iloc[0]
            return duration
        
    except Exception as e:
            print(f"    [WARNING] Could not get BH_trimmed duration: {e}")
    
    return None


def find_subject_files(input_folder: str) -> List[Tuple[str, str]]:
    """Find all subject Excel files in the input folder."""
    folder = Path(input_folder)
    subjects = []
    
    if not folder.exists():
        print(f"[WARNING] Input folder not found: {input_folder}")
        return subjects
    
    for f in folder.glob("HRV_data_sujeto_*.xlsx"):
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


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_all_subjects(
    input_folder: str = INPUT_FOLDER,
    output_file: str = OUTPUT_FILE,
    segments: List[str] = SEGMENTS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Process all subjects and calculate HRV parameters."""
    
    print("=" * 70)
    print("HRV ANALYSIS - CALCULATING PARAMETERS FROM CLEANED DATA")
    print("=" * 70)
    print(f"\nInput folder: {input_folder}")
    print(f"Output file: {output_file}")
    print(f"Segments: {len(segments)}")
    print(f"Parameters per segment: {len(HRV_PARAM_NAMES)}")
    
    subject_files = find_subject_files(input_folder)
    
    if not subject_files:
        print("\n[WARNING] No subject files found!")
        return pd.DataFrame()
    
    print(f"\nFound {len(subject_files)} subjects: {[s[0] for s in subject_files]}")
    
    all_results = []
    
    for subject_id, file_path in subject_files:
        print(f"\n{'-'*70}")
        print(f"Processing Subject {subject_id}")
        print(f"{'-'*70}")
        
        segments_data = load_subject_data(file_path)
        
        if not segments_data:
            print(f"  [WARNING] No valid segment data found")
            continue
        
        print(f"  Loaded segments: {list(segments_data.keys())}")
        
        result_row = {"subject": subject_id}
        
        # Process standard segments
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
                for param in HRV_PARAM_NAMES:
                    result_row[f"{seg_name}_{param}"] = np.nan
                continue
            
            seg_data = segments_data[seg_key]
            rr_ms = seg_data['full']
            
            if verbose:
                duration = seg_data['duration_s']
                print(f"  [{seg_name}]: {len(rr_ms)} RR intervals, {duration:.1f}s duration")
            
            # Calculate HRV for full segment
            hrv_params = calculate_hrv_parameters(rr_ms, verbose=verbose)
            
            for param, value in hrv_params.items():
                result_row[f"{seg_name}_{param}"] = value
            
        
        # Process BH segments with RSP-based trimming
        for bh_seg_name in BH_SEGMENTS:
            # Find matching BH segment in HRV data
            seg_key = None
            for key in segments_data.keys():
                if key.lower().replace(" ", "") == bh_seg_name.lower().replace(" ", ""):
                    seg_key = key
                    break
            
            if seg_key is None:
                if verbose:
                    print(f"  [{bh_seg_name}]: Not available in HRV data")
                for param in HRV_PARAM_NAMES:
                    result_row[f"{bh_seg_name}_trimmed_{param}"] = np.nan
                continue
            
            # Get trimmed duration from RSP data
            trimmed_duration = get_bh_trimmed_duration(subject_id, bh_seg_name)
            
            if trimmed_duration is None:
                if verbose:
                    print(f"  [{bh_seg_name}]: No RSP trimming info available")
                for param in HRV_PARAM_NAMES:
                    result_row[f"{bh_seg_name}_trimmed_{param}"] = np.nan
                continue
            
            # Trim HRV data to match RSP trimming
            seg_data = segments_data[seg_key]
            rr_ms_full = seg_data['full']
            
            # Calculate cumulative time to find cutoff
            cumsum_s = np.cumsum(rr_ms_full) / 1000.0
            idx_trim = np.searchsorted(cumsum_s, trimmed_duration, side='right')
            rr_ms_trimmed = rr_ms_full[:idx_trim]
            
            if len(rr_ms_trimmed) < 10:
                if verbose:
                    print(f"  [{bh_seg_name}_trimmed]: Too few beats after trimming ({len(rr_ms_trimmed)})")
                for param in HRV_PARAM_NAMES:
                    result_row[f"{bh_seg_name}_trimmed_{param}"] = np.nan
                continue
            
            if verbose:
                print(f"  [{bh_seg_name}_trimmed]: {len(rr_ms_trimmed)} RR intervals, {trimmed_duration:.1f}s duration (RSP-based)")
            
            # Calculate HRV for trimmed segment
            hrv_params_trimmed = calculate_hrv_parameters(rr_ms_trimmed, verbose=False)
            
            for param, value in hrv_params_trimmed.items():
                result_row[f"{bh_seg_name}_trimmed_{param}"] = value
            
            # Fixed 60-second window within trimmed BH (if long enough)
            window_prefix = f"{bh_seg_name}_trimmed_60s"
            for metric in WINDOW_BASE_METRICS:
                result_row[f"{window_prefix}_{metric}"] = np.nan

            if trimmed_duration >= BH_60S_END:
                rr_window, window_times = extract_rr_window(rr_ms_trimmed, BH_60S_START, BH_60S_END)
                if len(rr_window) >= 10:
                    window_metrics = calculate_window_metrics(rr_window, window_times, include_freq=False)
                    for metric, value in window_metrics.items():
                        result_row[f"{window_prefix}_{metric}"] = value
                elif verbose:
                    print(f"    -> {window_prefix}: insufficient beats ({len(rr_window)}) for 60s window")
            elif verbose:
                print(f"    -> {window_prefix}: trimmed duration < {BH_60S_END}s, skipping fixed window")

        # Fixed-duration LB and Recovery windows (min 3–4)
        for window in FIXED_WINDOWS:
            window_prefix = window["name"]
            metrics_to_init = WINDOW_BASE_METRICS + (
                WINDOW_FREQ_METRICS if window["include_freq"] else []
            )

            for metric in metrics_to_init:
                result_row[f"{window_prefix}_{metric}"] = np.nan

            seg_key = find_segment_key(segments_data, window["segment"])
            if seg_key is None:
                if verbose:
                    print(f"  [{window_prefix}]: segment '{window['segment']}' not found")
                continue

            rr_ms = segments_data[seg_key]["full"]
            segment_duration = segments_data[seg_key]["duration_s"]
            if segment_duration < window["end_s"]:
                if verbose:
                    print(
                        f"  [{window_prefix}]: segment '{window['segment']}' duration "
                        f"{segment_duration:.1f}s < {int(window['end_s'])}s, skipping window"
                    )
                continue
            rr_window, window_times = extract_rr_window(
                rr_ms, window["start_s"], window["end_s"]
            )

            if len(rr_window) < 10:
                if verbose:
                    print(
                        f"  [{window_prefix}]: insufficient data ({len(rr_window)} RR) "
                        f"within {window['segment']} window"
                    )
                continue

            window_metrics = calculate_window_metrics(
                rr_window, window_times, include_freq=window["include_freq"]
            )

            for metric, value in window_metrics.items():
                result_row[f"{window_prefix}_{metric}"] = value
        
        all_results.append(result_row)
    
    if not all_results:
        print("\n[WARNING] No results to save!")
        return pd.DataFrame()
    
    df_results = pd.DataFrame(all_results)
    
    # Reorder columns
    columns_ordered = ["subject"]

    # 1) Normal segments in order
    for seg_name in segments:
        for param in HRV_PARAM_NAMES:
            col_name = f"{seg_name}_{param}"
            if col_name in df_results.columns:
                columns_ordered.append(col_name)

    # 2) BH trimmed (BH 1-4)
    for bh_name in BH_SEGMENTS:
        for param in HRV_PARAM_NAMES:
            col_name = f"{bh_name}_trimmed_{param}"
            if col_name in df_results.columns:
                columns_ordered.append(col_name)

    # 3) Fixed 60-second windows (LB, Recovery, BH_trimmed)
    for window in FIXED_WINDOWS:
        for metric in WINDOW_BASE_METRICS:
            col_name = f"{window['name']}_{metric}"
            if col_name in df_results.columns:
                columns_ordered.append(col_name)
        if window["include_freq"]:
            for metric in WINDOW_FREQ_METRICS:
                col_name = f"{window['name']}_{metric}"
                if col_name in df_results.columns:
                    columns_ordered.append(col_name)

    for bh_name in BH_SEGMENTS:
        window_prefix = f"{bh_name}_trimmed_60s"
        for metric in WINDOW_BASE_METRICS:
            col_name = f"{window_prefix}_{metric}"
            if col_name in df_results.columns:
                columns_ordered.append(col_name)

    # 4) Any remaining columns (e.g., other 2-min BH segments) at the end
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
        df_results.to_excel(writer, sheet_name="HRV_Parameters", index=False)
        
        # Summary sheet
        summary_params = ["mean_nn", "sdnn", "rmssd", "lf", "hf", "lf_hf_ratio", "sd1", "sd2", "n_rr"]
        summary_data = []
        
        for _, row in df_results.iterrows():
            subject = row["subject"]
            for seg_name in segments:
                seg_row = {"subject": subject, "segment": seg_name}
                for param in summary_params:
                    col_name = f"{seg_name}_{param}"
                    seg_row[param] = row.get(col_name, np.nan)
                summary_data.append(seg_row)
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        
        # Parameter definitions
        param_definitions = pd.DataFrame({
            "parameter": HRV_PARAM_NAMES,
            "description": [
                "Mean of NN intervals (ms)",
                "Standard deviation of NN intervals (ms)",
                "Standard deviation of successive NN differences (ms)",
                "Root mean square of successive NN differences (ms)",
                "Number of successive NN differences > 50ms",
                "Percentage of successive NN differences > 50ms (%)",
                "Number of successive NN differences > 20ms",
                "Percentage of successive NN differences > 20ms (%)",
                "Median of NN intervals (ms)",
                "Range of NN intervals (max - min) (ms)",
                "Coefficient of variation of NN intervals (SDNN/MeanNN × 100)",
                "Coefficient of variation of successive differences (RMSSD/MeanNN × 100)",
                "Mean heart rate (bpm)",
                "Maximum heart rate (bpm)",
                "Minimum heart rate (bpm)",
                "Standard deviation of heart rate (bpm)",
                "HRV triangular index",
                "Very low frequency power (0.003-0.04 Hz) (ms²)",
                "Low frequency power (0.04-0.15 Hz) (ms²)",
                "High frequency power (0.15-0.40 Hz) (ms²)",
                "Total spectral power (VLF+LF+HF) (ms²)",
                "LF/HF ratio",
                "Normalized LF power (%)",
                "Normalized HF power (%)",
                "Poincaré SD1 - short-term variability (ms)",
                "Poincaré SD2 - long-term variability (ms)",
                "SD1/SD2 ratio",
                "Cardiac Sympathetic Index (SD2/SD1)",
                "Cardiac Vagal Index (log10(SD1×SD2))",
                "Modified CSI (SD2²/SD1)",
                "Sample Entropy",
                "Number of RR intervals used in calculation",
            ],
        })
        extra_definitions = pd.DataFrame({
            "parameter": ["ln_rmssd", "hr_slope_bpm_per_min"],
            "description": [
                "Natural logarithm of RMSSD (ms)",
                "Slope of HR (bpm/min) from linear regression across the window",
            ],
        })
        param_definitions = pd.concat([param_definitions, extra_definitions], ignore_index=True)
        param_definitions.to_excel(writer, sheet_name="Parameter_Definitions", index=False)

        # Notes sheet
        notes = pd.DataFrame({
            "Topic": [
                "BH trimmed",
                "Fixed 60s windows",
            ],
            "Explanation": [
                "BH_trimmed HRV uses the duration of the BH_trimmed segment from RSP_clean_data. RR intervals are truncated to that duration before calculating HRV metrics.",
                "Fixed 60-second windows capture minute 3–4 inside LB and Recovery (LB_60s, Recovery_60s), seconds 10–70 of each BH_trimmed (BH_X_trimmed_60s), and the first 60 seconds of CPT if ≥70s. These windows report mean HR, HR slope, RMSSD, lnRMSSD, SD1, SDNN, and HF/LF only for LB_60s and Recovery_60s.",
            ],
        })
        notes.to_excel(writer, sheet_name="Notes", index=False)
    
    print(f"\n[SUCCESS] Results saved to: {output_file}")
    print(f"  • Sheet 'HRV_Parameters': {len(df_results)} subjects × {len(df_results.columns)-1} parameters")
    print(f"  • Sheet 'Summary': Key parameters in long format")
    print(f"  • Sheet 'Parameter_Definitions': Parameter descriptions")
    print(f"  • Sheet 'Notes': BH trimmed and 2-min normalization notes")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"\nSubjects processed: {len(df_results)}")
    print(f"Segments: {len(segments)}")
    
    print(f"\nData availability by segment:")
    for seg_name in segments:
        col_check = f"{seg_name}_n_rr"
        if col_check in df_results.columns:
            n_available = df_results[col_check].notna().sum()
            print(f"  {seg_name}: {n_available}/{len(df_results)} subjects")
    
    return df_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    df_final = process_all_subjects(
        input_folder=INPUT_FOLDER,
        output_file=OUTPUT_FILE,
        segments=SEGMENTS,
        verbose=True,
    )
