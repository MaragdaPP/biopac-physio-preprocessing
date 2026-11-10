"""
RSP Analysis Script - Calculate respiratory parameters from cleaned RSP data.

This script reads cleaned RSP data exported by rsp_preprocess_bh.py (from RSP_clean_data/)
and calculates comprehensive respiratory parameters.

Output: Final Excel file with one row per subject, columns for each segment's respiratory parameters:
- Respiratory rate and variability (RRV)
- Breathing amplitude and depth
- Inspiration/expiration timing
- Quality metrics

Based on:
- Khodadad et al. (2018) - Respiratory signal analysis
- Recent RRV (Respiratory Rate Variability) literature (2020-2026)
- AASM respiratory signal standards

Note: For cardiorespiratory coupling (RSA), use rsa_analysis_final.py (combines RSP + HRV)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
try:
    from scipy.signal import find_peaks
    _HAVE_SCIPY = True
except ImportError:
    find_peaks = None
    _HAVE_SCIPY = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# Sampling rate (should match preprocessing script)
RSP_SAMPLING_RATE = 63.0  # Hz - from rsp_preprocess_bh.py

# Input folder
SCRIPT_DIR = Path(__file__).resolve().parent
RSP_INPUT_FOLDER = str(SCRIPT_DIR / "RSP_clean_data")

# Output file for final parameters
OUTPUT_FILE = str(SCRIPT_DIR / "RSP_parametros_finales.xlsx")

# Segments to process (in order) - EXCLUDING BH segments
# BH segments excluded as per analysis plan: only keeping BH_trimmed timing for HRV
SEGMENTS = [
    "LB",
    "Intero 1",
    "Intero 2",
    "Intero 3",
    "Intero 4",
    "Intero 5",
    "Intero 6",
    "CPT",  # Added CPT
    "Recovery",
]

# Ordered list of respiratory parameters to keep column order consistent
RSP_PARAM_ORDER = [
    "segment_available",
    "duration_s",
    "rr_mean_bpm",
    "rr_std_bpm",
    "rr_cv",
    "rr_min_bpm",
    "rr_max_bpm",
    "rr_range_bpm",
    "breath_intervals_mean_s",
    "breath_intervals_std_s",
    "breath_intervals_cv",
    "rrv_rmssd_s",
    "rrv_sdnn_s",
    "n_breaths",
    "breath_rate_per_min",
    "amplitude_mean",
    "amplitude_std",
    "amplitude_cv",
    "amplitude_min",
    "amplitude_max",
    "amplitude_range",
    "inspiration_time_mean_s",
    "expiration_time_mean_s",
    "ie_ratio_mean",
    "ie_ratio_std",
]

# ============================================================================
# RESPIRATORY PARAMETER FUNCTIONS
# ============================================================================

def calculate_respiratory_rate_metrics(cycles_df: pd.DataFrame, duration_s: float) -> Dict[str, float]:
    """
    Calculate respiratory rate and variability metrics from breathing cycles.
    
    Args:
        cycles_df: DataFrame with breathing cycle information
        duration_s: Total duration of the segment in seconds
    
    Returns:
        Dictionary with respiratory rate metrics
    """
    params = {}
    
    # Filter valid cycles for metrics
    valid_cycles = cycles_df[cycles_df['valid_for_metrics'] == True].copy()
    
    if len(valid_cycles) == 0:
        # No valid cycles - return NaN
        params["rr_mean_bpm"] = np.nan
        params["rr_std_bpm"] = np.nan
        params["rr_cv"] = np.nan
        params["rr_min_bpm"] = np.nan
        params["rr_max_bpm"] = np.nan
        params["rr_range_bpm"] = np.nan
        params["breath_intervals_mean_s"] = np.nan
        params["breath_intervals_std_s"] = np.nan
        params["breath_intervals_cv"] = np.nan
        params["rrv_rmssd_s"] = np.nan
        params["rrv_sdnn_s"] = np.nan
        params["n_breaths"] = 0
        params["breath_rate_per_min"] = np.nan
        return params
    
    # Respiratory rate (BPM - breaths per minute)
    rr_bpm = valid_cycles['resp_rate_bpm'].values
    params["rr_mean_bpm"] = np.mean(rr_bpm)
    params["rr_std_bpm"] = np.std(rr_bpm, ddof=1) if len(rr_bpm) > 1 else 0
    params["rr_cv"] = params["rr_std_bpm"] / params["rr_mean_bpm"] if params["rr_mean_bpm"] > 0 else np.nan
    params["rr_min_bpm"] = np.min(rr_bpm)
    params["rr_max_bpm"] = np.max(rr_bpm)
    params["rr_range_bpm"] = params["rr_max_bpm"] - params["rr_min_bpm"]
    
    # Breath-to-breath intervals (seconds)
    intervals_s = valid_cycles['interval_s'].values
    params["breath_intervals_mean_s"] = np.mean(intervals_s)
    params["breath_intervals_std_s"] = np.std(intervals_s, ddof=1) if len(intervals_s) > 1 else 0
    params["breath_intervals_cv"] = params["breath_intervals_std_s"] / params["breath_intervals_mean_s"] if params["breath_intervals_mean_s"] > 0 else np.nan
    
    # RRV (Respiratory Rate Variability) - analogous to HRV metrics
    # RMSSD: root mean square of successive differences
    if len(intervals_s) > 1:
        successive_diffs = np.diff(intervals_s)
        params["rrv_rmssd_s"] = np.sqrt(np.mean(successive_diffs**2))
    else:
        params["rrv_rmssd_s"] = np.nan
    
    # SDNN: standard deviation of breath intervals
    params["rrv_sdnn_s"] = params["breath_intervals_std_s"]
    
    # Total number of breaths
    params["n_breaths"] = len(valid_cycles)
    
    # Average breathing rate (total breaths / duration)
    params["breath_rate_per_min"] = (len(valid_cycles) / duration_s) * 60 if duration_s > 0 else np.nan
    
    return params


def calculate_respiratory_amplitude_metrics(cycles_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate respiratory amplitude and depth metrics.
    
    Args:
        cycles_df: DataFrame with breathing cycle information
    
    Returns:
        Dictionary with amplitude metrics
    """
    params = {}
    
    # Filter valid cycles
    valid_cycles = cycles_df[cycles_df['valid_for_metrics'] == True].copy()
    
    if len(valid_cycles) == 0:
        params["amplitude_mean"] = np.nan
        params["amplitude_std"] = np.nan
        params["amplitude_cv"] = np.nan
        params["amplitude_min"] = np.nan
        params["amplitude_max"] = np.nan
        params["amplitude_range"] = np.nan
        return params
    
    amplitudes = valid_cycles['amplitude'].values
    
    params["amplitude_mean"] = np.mean(amplitudes)
    params["amplitude_std"] = np.std(amplitudes, ddof=1) if len(amplitudes) > 1 else 0
    params["amplitude_cv"] = params["amplitude_std"] / params["amplitude_mean"] if params["amplitude_mean"] > 0 else np.nan
    params["amplitude_min"] = np.min(amplitudes)
    params["amplitude_max"] = np.max(amplitudes)
    params["amplitude_range"] = params["amplitude_max"] - params["amplitude_min"]
    
    return params


def calculate_respiratory_timing_metrics(cycles_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate respiratory timing metrics (inspiration/expiration durations and ratios).
    
    Args:
        cycles_df: DataFrame with breathing cycle information
    
    Returns:
        Dictionary with timing metrics
    """
    params = {}
    
    # Filter valid cycles
    valid_cycles = cycles_df[cycles_df['valid_for_metrics'] == True].copy()
    
    if len(valid_cycles) == 0:
        params["inspiration_time_mean_s"] = np.nan
        params["expiration_time_mean_s"] = np.nan
        params["ie_ratio_mean"] = np.nan
        params["ie_ratio_std"] = np.nan
        return params
    
    # Calculate inspiration and expiration times
    # Inspiration: trough to peak
    # Expiration: peak to next trough
    inspiration_times = valid_cycles['peak_time_s'].values - valid_cycles['trough_time_s'].values
    expiration_times = valid_cycles['next_trough_time_s'].values - valid_cycles['peak_time_s'].values
    
    params["inspiration_time_mean_s"] = np.mean(inspiration_times)
    params["expiration_time_mean_s"] = np.mean(expiration_times)
    
    # I/E ratio (inspiration/expiration ratio)
    ie_ratios = inspiration_times / expiration_times
    params["ie_ratio_mean"] = np.mean(ie_ratios)
    params["ie_ratio_std"] = np.std(ie_ratios, ddof=1) if len(ie_ratios) > 1 else 0
    
    return params


# ============================================================================
# CYCLE DETECTION FROM SIGNAL (fallback when _cycles sheet missing)
# ============================================================================

def detect_cycles_from_signal(time_s: np.ndarray, rsp_clean: np.ndarray) -> pd.DataFrame:
    """
    Detect breathing cycles directly from the cleaned signal.
    Used only when per-cycle sheets are missing.
    """
    if not _HAVE_SCIPY:
        return pd.DataFrame()

    if len(time_s) < 3 or len(rsp_clean) < 3:
        return pd.DataFrame()

    # Estimate sampling rate from time vector
    dt = np.diff(time_s)
    fs = 1.0 / np.median(dt) if len(dt) > 0 and np.median(dt) > 0 else RSP_SAMPLING_RATE

    # Light smoothing to reduce spurious peaks
    win = max(3, int(0.3 * fs))
    kernel = np.ones(win) / win
    rsp_smooth = np.convolve(rsp_clean, kernel, mode="same")

    # Breathing rate bounds
    min_bpm = 4.0
    max_bpm = 60.0
    min_dist = int(fs * (60.0 / max_bpm))

    # Detect peaks and troughs
    peaks, _ = find_peaks(rsp_smooth, distance=min_dist)
    troughs, _ = find_peaks(-rsp_smooth, distance=min_dist)

    if len(troughs) < 2:
        return pd.DataFrame()

    # Build cycles from consecutive troughs
    rows = []
    troughs = np.sort(troughs)
    peaks = np.sort(peaks)

    for i in range(len(troughs) - 1):
        t_idx = troughs[i]
        t_next_idx = troughs[i + 1]
        if t_idx >= len(time_s) or t_next_idx >= len(time_s):
            continue

        # Find a peak between the troughs
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

def load_subject_data(subject_id: int) -> Optional[pd.ExcelFile]:
    """
    Load RSP data for a subject.
    
    Args:
        subject_id: Subject ID number
    
    Returns:
        RSP Excel file object or None if not found
    """
    rsp_file = Path(RSP_INPUT_FOLDER) / f"RSP_data_sujeto_{subject_id}.xlsx"
    
    if rsp_file.exists():
        try:
            return pd.ExcelFile(rsp_file)
        except Exception as e:
            warnings.warn(f"Error loading RSP file for subject {subject_id}: {e}")
            return None
    else:
        warnings.warn(f"RSP file not found for subject {subject_id}: {rsp_file}")
        return None


def process_segment(segment_name: str, rsp_excel: pd.ExcelFile) -> Dict[str, float]:
    """
    Process a single segment and calculate respiratory metrics.

    Data source: individual RSP_data_sujeto_X.xlsx only.
    - Segment sheet (time_s, rsp_clean): used for duration_s and segment_available.
    - RESUMEN sheet: primary source for rr_mean_bpm, rr_std_bpm, n_breaths,
      mean_amplitude, mean_ie_ratio, etc. (all segments that have a RESUMEN row get filled).
    - _cycles sheets: not required; when present they add/override detailed metrics
      (min/max, intervals, RRV, etc.). Preprocessing is not modified.
    """
    all_params = {}

    # Check if segment exists (segment sheet with time_s, rsp_clean)
    has_rsp = segment_name in rsp_excel.sheet_names

    if not has_rsp:
        return {"segment_available": False}

    # Load segment sheet for duration
    rsp_df = pd.read_excel(rsp_excel, sheet_name=segment_name)
    all_params["segment_available"] = True

    if "time_s" in rsp_df.columns and len(rsp_df) > 0:
        duration_s = float(rsp_df["time_s"].iloc[-1] - rsp_df["time_s"].iloc[0])
        all_params["duration_s"] = duration_s
    else:
        duration_s = len(rsp_df) / RSP_SAMPLING_RATE
        all_params["duration_s"] = duration_s

    # Primary source: RESUMEN sheet (same structure in all subject Excels)
    resumen_stats = extract_resumen_stats(segment_name, rsp_excel)
    all_params.update(resumen_stats)

    # Optional: when _cycles sheet exists, add/override with cycle-based metrics
    cycles_sheet_name = f"{segment_name}_cycles"
    cycles_df = None
    if cycles_sheet_name in rsp_excel.sheet_names:
        cycles_df = pd.read_excel(rsp_excel, sheet_name=cycles_sheet_name)

    if cycles_df is not None and len(cycles_df) > 0:
        rr_params = calculate_respiratory_rate_metrics(cycles_df, duration_s)
        all_params.update(rr_params)
        amplitude_params = calculate_respiratory_amplitude_metrics(cycles_df)
        all_params.update(amplitude_params)
        timing_params = calculate_respiratory_timing_metrics(cycles_df)
        all_params.update(timing_params)
    else:
        # Fallback: detect cycles directly from signal if available
        if "time_s" in rsp_df.columns and "rsp_clean" in rsp_df.columns:
            fallback_cycles = detect_cycles_from_signal(
                rsp_df["time_s"].to_numpy(), rsp_df["rsp_clean"].to_numpy()
            )
            if len(fallback_cycles) > 0:
                rr_params = calculate_respiratory_rate_metrics(fallback_cycles, duration_s)
                all_params.update(rr_params)
                amplitude_params = calculate_respiratory_amplitude_metrics(fallback_cycles)
                all_params.update(amplitude_params)
                timing_params = calculate_respiratory_timing_metrics(fallback_cycles)
                all_params.update(timing_params)

    return all_params


def extract_resumen_stats(segment_name: str, rsp_excel: pd.ExcelFile) -> Dict[str, float]:
    """
    Attempt to read aggregate metrics from the RESUMEN sheet when detailed cycles are missing.
    """
    stats: Dict[str, float] = {}

    sheet_name = None
    for sheet in rsp_excel.sheet_names:
        if sheet.lower() == "resumen":
            sheet_name = sheet
            break

    if sheet_name is None:
        return stats

    try:
        resumen_df = pd.read_excel(rsp_excel, sheet_name=sheet_name)
    except Exception:
        return stats

    normalized_segment = segment_name.replace(" ", "").lower()
    resumen_match = resumen_df[
        resumen_df["segment"].astype(str).str.replace(" ", "").str.lower() == normalized_segment
    ]

    if resumen_match.empty:
        return stats

    row = resumen_match.iloc[0]
    
    # Extract basic stats
    mean_rate = row.get("mean_resp_rate_bpm", np.nan)
    std_rate = row.get("std_resp_rate_bpm", np.nan)
    
    # Calculate CV from mean and std if both available
    cv_rate = (std_rate / mean_rate) if (pd.notna(mean_rate) and mean_rate > 0 and pd.notna(std_rate)) else np.nan
    
    stats.update({
        "rr_mean_bpm": mean_rate,
        "rr_std_bpm": std_rate,
        "rr_cv": cv_rate,
        "rr_min_bpm": np.nan,  # Requires individual cycles
        "rr_max_bpm": np.nan,  # Requires individual cycles
        "rr_range_bpm": np.nan,  # Requires individual cycles
        "breath_intervals_mean_s": np.nan,  # Requires individual cycles
        "breath_intervals_std_s": np.nan,  # Requires individual cycles
        "breath_intervals_cv": np.nan,  # Requires individual cycles
        "rrv_rmssd_s": np.nan,  # Requires cycle-to-cycle differences
        "rrv_sdnn_s": std_rate,  # SDNN ≈ std of rates (approximate from RESUMEN)
        "n_breaths": row.get("n_breaths", np.nan),
        "breath_rate_per_min": mean_rate,  # Same as rr_mean_bpm
        "amplitude_mean": row.get("mean_amplitude", np.nan),
        "amplitude_std": np.nan,  # Requires individual cycles
        "amplitude_cv": np.nan,  # Requires individual cycles
        "amplitude_min": np.nan,  # Requires individual cycles
        "amplitude_max": np.nan,  # Requires individual cycles
        "amplitude_range": np.nan,  # Requires individual cycles
        "inspiration_time_mean_s": np.nan,  # Requires individual cycles
        "expiration_time_mean_s": np.nan,  # Requires individual cycles
        "ie_ratio_mean": row.get("mean_ie_ratio", np.nan),
        "ie_ratio_std": np.nan,  # Requires individual cycles
    })

    return stats


def process_subject(subject_id: int) -> Dict[str, Dict[str, float]]:
    """
    Process all segments for a single subject.
    
    Args:
        subject_id: Subject ID number
    
    Returns:
        Dictionary mapping segment names to parameter dictionaries
    """
    print(f"\nProcessing subject {subject_id}...")
    
    # Load data file
    rsp_excel = load_subject_data(subject_id)
    
    if rsp_excel is None:
        warnings.warn(f"No RSP data found for subject {subject_id}, skipping.")
        return {}
    
    results = {}
    
    # Process main segments only (BH segments excluded from respiratory analysis)
    for segment_name in SEGMENTS:
        print(f"  Processing segment: {segment_name}")
        params = process_segment(segment_name, rsp_excel)
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
            for param_name in RSP_PARAM_ORDER:
                col_name = f"{segment_prefix}_{param_name}"
                row[col_name] = params.get(param_name, np.nan)

        rows.append(row)

    # Build DataFrame with consistent column order
    columns = ["subject_id"]
    for segment_name in SEGMENTS:
        segment_prefix = segment_name.replace(" ", "_")
        for param_name in RSP_PARAM_ORDER:
            columns.append(f"{segment_prefix}_{param_name}")

    df = pd.DataFrame(rows, columns=columns)
    
    return df


def main():
    """
    Main function to process all subjects and save results.
    """
    print("="*80)
    print("RSP Analysis - Respiratory Parameters Only")
    print("="*80)
    print(f"\nRSP Input folder: {RSP_INPUT_FOLDER}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Note: For RSA metrics (cardiorespiratory coupling), run rsa_analysis_final.py")
    
    # Find all subject IDs from RSP files
    rsp_folder = Path(RSP_INPUT_FOLDER)
    rsp_files = list(rsp_folder.glob("RSP_data_sujeto_*.xlsx"))
    
    subject_ids = []
    for f in rsp_files:
        # Extract subject ID from filename
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
    
    # Save to Excel with documentation sheets
    print(f"\nSaving results to {OUTPUT_FILE}...")
    output_path = Path(OUTPUT_FILE)
    if output_path.exists():
        try:
            output_path.unlink()
        except PermissionError:
            warnings.warn(f"Unable to delete existing output file: {OUTPUT_FILE}")

    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
            # Main data sheet
            output_df.to_excel(writer, sheet_name='RSP_Parameters', index=False)
            
            # Summary sheet (key metrics in long format)
            summary_params = ["rr_mean_bpm", "rr_std_bpm", "rr_cv", "n_breaths", 
                            "amplitude_mean", "ie_ratio_mean"]
            summary_data = []
            for _, row in output_df.iterrows():
                subject = row["subject_id"]
                for seg_name in SEGMENTS:
                    seg_prefix = seg_name.replace(" ", "_")
                    seg_row = {"subject_id": subject, "segment": seg_name}
                    for param in summary_params:
                        col_name = f"{seg_prefix}_{param}"
                        seg_row[param] = row.get(col_name, np.nan)
                    summary_data.append(seg_row)
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Parameter definitions sheet
            param_defs = pd.DataFrame({
                "parameter": RSP_PARAM_ORDER,
                "description": [
                    "Segment available in data (True/False)",
                    "Total duration of segment (seconds)",
                    "Mean respiratory rate (breaths per minute)",
                    "Standard deviation of respiratory rate (bpm)",
                    "Coefficient of variation of respiratory rate (std/mean)",
                    "Minimum respiratory rate (bpm) - requires cycles",
                    "Maximum respiratory rate (bpm) - requires cycles",
                    "Range of respiratory rates (max-min) (bpm) - requires cycles",
                    "Mean breath-to-breath interval (seconds) - requires cycles",
                    "Std of breath-to-breath intervals (s) - requires cycles",
                    "CV of breath-to-breath intervals - requires cycles",
                    "RRV RMSSD: Root mean square of successive breath interval differences (s) - requires cycles",
                    "RRV SDNN: Standard deviation of breath intervals (s) ≈ rr_std_bpm",
                    "Total number of breaths in segment",
                    "Breathing rate calculated from total breaths/duration (bpm)",
                    "Mean breath amplitude (trough-to-peak) - requires cycles",
                    "Standard deviation of breath amplitude - requires cycles",
                    "Coefficient of variation of amplitude - requires cycles",
                    "Minimum breath amplitude - requires cycles",
                    "Maximum breath amplitude - requires cycles",
                    "Range of amplitudes (max-min) - requires cycles",
                    "Mean inspiration time (trough to peak) (seconds) - requires cycles",
                    "Mean expiration time (peak to next trough) (seconds) - requires cycles",
                    "Mean I/E ratio (inspiration/expiration time) - requires cycles",
                    "Standard deviation of I/E ratio - requires cycles",
                ],
            })
            param_defs.to_excel(writer, sheet_name='Parameter_Definitions', index=False)
            
            # Add notes sheet
            notes_data = pd.DataFrame({
                "Topic": [
                    "RRV Metrics",
                    "Missing Cycles",
                    "Cycle Detection Fallback",
                    "rr_mean_bpm vs breath_rate_per_min",
                    "Segments Analyzed",
                    "BH Segments",
                ],
                "Explanation": [
                    "RRV (Respiratory Rate Variability) measures breath-to-breath variability. RMSSD = root mean square of successive differences between breath intervals. SDNN = standard deviation of breath intervals.",
                    "Metrics marked 'requires cycles' need individual breath cycle data (_cycles sheets). When cycles are missing, basic stats (mean, std, n_breaths) are extracted from RESUMEN.",
                    "When _cycles sheets are missing, the analysis script detects cycles directly from time_s + rsp_clean to compute min/max, interval stats, and RRV metrics. This is a signal-based fallback and may differ from preprocessing results.",
                    "rr_mean_bpm and breath_rate_per_min should be identical or very similar. rr_mean_bpm comes from individual cycle rates, breath_rate_per_min = total_breaths/duration. Small differences may occur due to edge effects.",
                    "LB, Intero 1-6, CPT, Recovery. BH segments excluded (no breathing during breath-hold).",
                    "Breath-hold segments excluded from respiratory analysis as there are no breathing cycles during apnea.",
                ],
            })
            notes_data.to_excel(writer, sheet_name='Notes', index=False)
            
    except PermissionError:
        alternative_path = output_path.with_name(output_path.stem + "_updated.xlsx")
        with pd.ExcelWriter(alternative_path, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='RSP_Parameters', index=False)
        warnings.warn(
            f"Primary output file was locked. Saved updated data to: {alternative_path}"
        )
    
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
    # Suppress pandas warnings
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
