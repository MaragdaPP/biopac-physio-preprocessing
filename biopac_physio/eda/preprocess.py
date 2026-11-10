# Part of biopac_physio — https://github.com/MaragdaPP/biopac-physio-preprocessing
# Library only: the per-subject workflow lives in studies/<study>/eda_clean.py

# %% Imports & configuration
"""
Standalone EDA (Electrodermal Activity) preprocessing helper for BIOPAC .acq files.
Run cell-by-cell in VSCode/Cursor.
Required: neurokit2, bioread, matplotlib, pandas, numpy, scipy

Based on:
- SPR recommendations (Boucsein et al., 2012)
- Braithwaite et al. guide
- cvxEDA model-based decomposition

Key differences from ECG preprocessing:
- Tonic (SCL) vs Phasic (SCR) decomposition instead of R-peak detection
- Minimal filtering (low-pass ~5 Hz only)
- Artifact detection: flat lines, jumps, saturation
- SCR detection with amplitude thresholds
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import neurokit2 as nk

try:  # Optional plotting
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:  # BIOPAC reader
    import bioread
except ImportError:
    bioread = None

try:
    from scipy import signal as scipy_signal
    from scipy.interpolate import interp1d
    from scipy.ndimage import uniform_filter1d
    _HAVE_SCIPY = True
except ImportError:
    scipy_signal = None
    interp1d = None
    _HAVE_SCIPY = False

# ---- Configuration defaults (edit if needed) ----
# Downsampling
TARGET_FS_HZ = 32.0  # Recommended for EDA: 10-32 Hz

# Filtering (minimal, low-pass only)
LOWPASS_CUTOFF_HZ = 3.0  # Low-pass filter cutoff (Boucsein 2012 recommends 3 Hz as conservative standard)

# Artifact detection thresholds
FLAT_LINE_THRESHOLD_S = 5.0  # Minimum duration to consider as flat line (very conservative)
FLAT_LINE_STD_THRESHOLD = 0.001  # µS - absolute threshold (very strict - only truly flat)
FLAT_LINE_ENABLED = False  # Disable by default - too many false positives for slow EDA
JUMP_THRESHOLD_US = 0.5  # µS - sudden change in consecutive samples
SATURATION_HIGH_US = 20.0  # µS - values above this are likely saturated (typical for commercial equipment)
SATURATION_LOW_US = 0.0  # µS - values at or below this indicate detachment

# SCR detection
SCR_MIN_AMPLITUDE_US = 0.05  # µS - minimum amplitude (guías ~0.01–0.05; 0.05 evita ruido)
SCR_MIN_RISE_TIME_S = 1.0   # Minimum rise time (literatura: <1 s suele ser artefacto)
SCR_MAX_RISE_TIME_S = 4.0   # Maximum rise time (guías típico 1–4 s; equilibrio dobletes vs deriva)

# Quality control
BAD_PROP_THRESHOLD = 0.20  # Exclude segment if BAD > 20%

# Enlarged window inspection thresholds
ENLARGED_DERIVATIVE_THRESHOLD_US_S = 0.1  # µS/s - |dEDA/dt| threshold for enlarged inspection
ENLARGED_PHASIC_AMPLITUDE_MIN_US = 0.05  # µS - minimum phasic amplitude for enlarged inspection
ENLARGED_PHASIC_AMPLITUDE_MAX_US = 0.1  # µS - maximum phasic amplitude threshold (use range 0.05-0.1)
ENLARGED_SCR_PROXIMITY_S = 2.0  # s - minimum time between SCRs to consider them "close"
ENLARGED_WINDOW_MARGIN_S = 5.0  # s - margin around detected regions for enlarged view

# Plot colors
PLOT_COLORS = {
    "raw": "#1f77b4",
    "clean": "#2ca02c",
    "tonic": "#9467bd",
    "phasic": "#d62728",
    "scr": "#ff7f0e",
    "bad": "#ffbb78",
}


@dataclass
class EDASegmentResult:
    """Container for processed EDA segment data."""
    name: str
    start_idx: int
    end_idx: int
    fs: float  # Original sampling frequency
    fs_downsampled: float  # Downsampled frequency
    
    # Raw and processed signals (downsampled)
    eda_raw: np.ndarray = field(default_factory=lambda: np.array([]))
    eda_clean: np.ndarray = field(default_factory=lambda: np.array([]))
    eda_tonic: np.ndarray = field(default_factory=lambda: np.array([]))  # SCL
    eda_phasic: np.ndarray = field(default_factory=lambda: np.array([]))  # SCR driver
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # SCR peaks
    scr_onsets: np.ndarray = field(default_factory=lambda: np.array([]))
    scr_peaks: np.ndarray = field(default_factory=lambda: np.array([]))
    scr_amplitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Bad intervals: (start_s, end_s, reason)
    bad_intervals: List[Tuple[float, float, str]] = field(default_factory=list)
    
    @property
    def start_s(self) -> float:
        return self.start_idx / self.fs
    
    @property
    def end_s(self) -> float:
        return self.end_idx / self.fs
    
    @property
    def duration_s(self) -> float:
        return (self.end_idx - self.start_idx) / self.fs


def _ensure_optional(dep, name: str):
    if dep is None:
        raise ImportError(
            f"{name} is required for this action but is not installed. "
            f"Install with `pip install {name}`."
        )


def _to_path(path_like) -> Path:
    return Path(path_like).expanduser()


# =============================================================================
# SIGNAL LOADING (reused from ECG script with adaptations)
# =============================================================================

def cargar_senal_y_eventos(acq_file: str, eda_col: str):
    """
    Load EDA signal and event markers from BIOPAC .acq file.
    
    Returns:
        - eda_raw: np.ndarray (full EDA signal)
        - fs_eda: float (Hz)
        - eventos: dict[str, int] (text -> first sample_index)
        - eventos_full: dict[str, list[int]] (text -> all sample_indices)
        - data_events: bioread object
    """
    # Read EDA data using neurokit2 (handles scaling properly)
    physio_file, fs = nk.read_acqknowledge(acq_file)

    # Allow eda_col to be optional or incorrect: try to auto-detect EDA channel
    if eda_col is None or eda_col not in physio_file.columns:
        candidate_cols = [c for c in physio_file.columns if "EDA" in str(c).upper() or "EDA" in str(c)]
        if not candidate_cols:
            # Fallback: choose first numeric column
            numeric_cols = [c for c in physio_file.columns if np.issubdtype(physio_file[c].dtype, np.number)]
            if numeric_cols:
                chosen = numeric_cols[0]
                print(f"  ⚠️  EDA channel '{eda_col}' not found - using numeric column '{chosen}' as fallback")
                eda_col = chosen
            else:
                raise ValueError(
                    f"EDA channel '{eda_col}' not found and no numeric column available. Available columns: {list(physio_file.columns)}"
                )
        else:
            eda_col = candidate_cols[0]
            print(f"  ⚠️  EDA channel not specified or not found - auto-detected '{eda_col}'")

    eda_raw = physio_file[eda_col].values
    fs_eda = float(fs)
    
    # Read event markers using bioread
    _ensure_optional(bioread, "bioread")
    data_events = bioread.read_file(acq_file)
    
    # Parse event markers
    eventos_full: Dict[str, List[int]] = {}
    for m in data_events.event_markers:
        k = (m.text or "").strip()
        if k == "":
            continue
        eventos_full.setdefault(k, []).append(int(m.sample_index))
    
    eventos: Dict[str, int] = {k: v[0] for k, v in eventos_full.items()}
    
    print("=== EDA SIGNAL INFO ===")
    print(f"Sampling rate: {fs_eda} Hz")
    print(f"Total samples: {len(eda_raw)}")
    print(f"Duration: {len(eda_raw)/fs_eda:.1f} s")
    print(f"EDA range: {np.nanmin(eda_raw):.4f} to {np.nanmax(eda_raw):.4f} µS")
    
    print("\n=== EVENTS FOUND ===")
    for k, v in eventos_full.items():
        if len(v) == 1:
            print(repr(k), "->", v[0])
        else:
            print(repr(k), "->", v, " (DUPLICATE)")
    
    return eda_raw, fs_eda, eventos, eventos_full, data_events


def cargar_datos_y_segmentos(
    acq_file: str,
    eda_col: str,
    segmentos_interes: Sequence[str],
    duraciones_objetivo: Optional[Dict[str, float]] = None,
):
    """
    Carga la señal EDA y eventos, y construye el diccionario de segmentos.

    Returns:
        eda_raw, fs_eda, eventos, eventos_full, data_events, segmentos_dict
    """
    eda_raw, fs_eda, eventos, eventos_full, data_events = cargar_senal_y_eventos(acq_file, eda_col)

    segmentos_dict = construir_diccionario_segmentos(
        eventos=eventos,
        segmentos_interes=segmentos_interes,
        duraciones_objetivo=duraciones_objetivo,
        fs_eda=fs_eda,
        n_total=len(eda_raw),
    )

    return eda_raw, fs_eda, eventos, eventos_full, data_events, segmentos_dict


def inspeccionar_duraciones_segmentos(
    segmentos_dict: Dict[str, Tuple[int, int]],
    segmentos_interes: Sequence[str],
    fs_eda: float,
    sujeto_id: str = "N/A"
) -> None:
    """
    Muestra una tabla con las duraciones, tiempos de inicio/fin e índices de muestras
    para todos los segmentos de interés.
    
    Args:
        segmentos_dict: Diccionario con {segmento: (start_idx, end_idx)}
        segmentos_interes: Lista de nombres de segmentos a inspeccionar
        fs_eda: Frecuencia de muestreo del EDA (Hz)
        sujeto_id: ID del sujeto para mostrar en el encabezado
    """
    print(f"\n=== INSPECCIÓN DE DURACIONES DE SEGMENTOS (SUJETO {sujeto_id}) ===\n")
    if segmentos_dict is None:
        print("⚠️  segmentos_dict es None - no se puede inspeccionar duraciones")
        return
    for seg in segmentos_interes:
        info = segmentos_dict.get(seg)
        if info is None:
            print(f"{seg:15s}: NO ENCONTRADO en segmentos_dict")
            continue
        start_idx, end_idx = info
        dur_s = (end_idx - start_idx) / fs_eda if fs_eda else float("nan")
        t_start_s = start_idx / fs_eda if fs_eda else float("nan")
        t_end_s = end_idx / fs_eda if fs_eda else float("nan")
        print(f"{seg:15s}: {dur_s:7.2f} s  |  Inicio: {t_start_s:8.2f} s  |  Fin: {t_end_s:8.2f} s  |  Samples: {start_idx:8d} - {end_idx:8d}")
    print()


def _keys_segmento(seg: str) -> Tuple[str, str]:
    """Get (key_ini, key_fin) for segment markers."""
    seg_lower = seg.lower().strip()
    
    if seg_lower.startswith("intero"):
        key_ini = seg.strip()
        key_fin = "Fin " + seg_lower
        return key_ini, key_fin
    
    key_ini = f"Inicio {seg.strip()}"
    key_fin = f"Fin {seg.strip()}"
    return key_ini, key_fin


def comprobar_marcadores_segmentos(
    eventos: Dict[str, int],
    segmentos_interes: Sequence[str],
    permitir_post10: bool = True,
):
    """Check that required markers exist for all segments of interest."""
    print("\n=== MARKER CHECK BY SEGMENT ===")
    
    for seg in segmentos_interes:
        seg_lower = seg.lower().strip()
        
        if seg_lower.startswith("post"):
            if not permitir_post10:
                raise ValueError(f"Post-type segment '{seg}' not allowed.")
            if "Post 10min" not in eventos:
                raise ValueError("Missing 'Post 10min' marker.")
            print(f"[OK] Segment {seg}: start found at 'Post 10min'")
            continue
        
        key_ini, key_fin = _keys_segmento(seg)
        
        if key_ini not in eventos:
            raise ValueError(f"Missing marker '{key_ini}' for segment {seg}")
        if key_fin not in eventos:
            raise ValueError(f"Missing marker '{key_fin}' for segment {seg}")
        
        print(f"[OK] Segment {seg}: '{key_ini}' and '{key_fin}' found.")
    
    print("Marker check completed.\n")


def get_segment_indices(
    eventos: Dict[str, int], seg: str, n_total: int
) -> Tuple[int, int]:
    """Get (start_idx, end_idx) sample indices for a segment."""
    seg_lower = seg.lower().strip()
    
    if seg_lower.startswith("post"):
        if "Post 10min" not in eventos:
            raise ValueError("Missing 'Post 10min' marker")
        start = int(eventos["Post 10min"])
        end = int(n_total)
        return start, end
    
    key_ini, key_fin = _keys_segmento(seg)
    
    start = int(eventos[key_ini])
    end = int(eventos[key_fin])
    
    if start < 0 or end > n_total or end <= start:
        raise ValueError(
            f"Segment '{seg}' out of range (start={start}, end={end}, n_total={n_total})"
        )
    
    return start, end


def construir_diccionario_segmentos(
    eventos: Dict[str, int],
    segmentos_interes: Sequence[str],
    duraciones_objetivo: Optional[Dict[str, float]] = None,
    fs_eda: Optional[float] = None,
    n_total: Optional[int] = None,
) -> Dict[str, Tuple[int, int]]:
    """Build {segment: (start_idx, end_idx)} dict using marker conventions."""
    duraciones_objetivo = duraciones_objetivo or {}
    if not eventos:
        return {}
    
    segmentos_dict: Dict[str, Tuple[int, int]] = {}
    for seg in segmentos_interes:
        try:
            start_idx, end_idx = get_segment_indices(eventos, seg, n_total or 0)
            segmentos_dict[seg] = (start_idx, end_idx)
        except (KeyError, ValueError) as e:
            key_ini, _ = _keys_segmento(seg)
            if key_ini in eventos and seg in duraciones_objetivo and fs_eda is not None:
                start_idx = eventos[key_ini]
                dur_samples = int(duraciones_objetivo[seg] * fs_eda)
                end_idx = start_idx + dur_samples
                if n_total is not None:
                    end_idx = min(end_idx, n_total)
                segmentos_dict[seg] = (start_idx, end_idx)
                print(f"⚠️  Using target duration for {seg}: {duraciones_objetivo[seg]}s")
            else:
                print(f"⚠️  Could not build segment {seg}: {e}")
    return segmentos_dict


# =============================================================================
# DOWNSAMPLING
# =============================================================================

def downsample_signal(signal: np.ndarray, fs_original: float, fs_target: float) -> Tuple[np.ndarray, float]:
    """
    Downsample EDA signal to target frequency.
    EDA is slow; high sampling rates add noise without physiological benefit.
    
    Args:
        signal: Original signal
        fs_original: Original sampling frequency (Hz)
        fs_target: Target sampling frequency (Hz)
    
    Returns:
        (downsampled_signal, actual_fs)
    """
    if fs_original <= fs_target:
        return signal.copy(), fs_original
    
    factor = int(fs_original / fs_target)
    if factor <= 1:
        return signal.copy(), fs_original
    
    # Minimum samples required for Butterworth filter (order 4) + padding
    # filtfilt requires at least max(len(a), len(b)) samples, plus padding
    MIN_SAMPLES_FOR_FILTER = 20  # Conservative minimum for order-4 filter
    
    # Anti-aliasing filter before downsampling
    if _HAVE_SCIPY and len(signal) >= MIN_SAMPLES_FOR_FILTER:
        # Design lowpass filter at Nyquist of target frequency
        nyq = fs_target / 2.0
        b, a = scipy_signal.butter(4, nyq / (fs_original / 2.0), btype='low')
        filtered = scipy_signal.filtfilt(b, a, signal)
        downsampled = filtered[::factor]
    else:
        # Simple averaging for very short segments or if scipy not available
        # This is acceptable for very short segments where filtering is less critical
        n_new = len(signal) // factor
        if n_new > 0:
            downsampled = signal[:n_new * factor].reshape(-1, factor).mean(axis=1)
        else:
            # If segment is shorter than factor, just return as-is
            downsampled = signal.copy()
    
    actual_fs = fs_original / factor
    return downsampled, actual_fs


# =============================================================================
# ARTIFACT DETECTION (EDA-specific)
# =============================================================================

def detect_flat_lines(
    eda: np.ndarray, 
    fs: float,
    min_duration_s: float = FLAT_LINE_THRESHOLD_S,
    std_threshold: float = FLAT_LINE_STD_THRESHOLD,
) -> List[Tuple[float, float]]:
    """
    Detect flat line segments (electrode detachment).
    
    A flat line is defined as a segment where the standard deviation
    is below threshold for a minimum duration.
    
    Uses relative threshold: std < max(absolute_threshold, 1% of signal range)
    This prevents flagging normal slow EDA changes as "flat".
    
    Returns: List of (start_s, end_s) tuples
    """
    win_samples = int(min_duration_s * fs)
    if win_samples < 2:
        win_samples = 2
    
    # Use absolute threshold only (relative thresholds cause false positives for slow EDA)
    # Flat lines should be truly flat (std < 0.001 µS) for at least 5 seconds
    relative_threshold = std_threshold
    
    flat_intervals = []
    
    # Calculate rolling std
    if _HAVE_SCIPY:
        # Use uniform filter for efficiency
        rolling_std = np.zeros_like(eda)
        for i in range(win_samples, len(eda)):
            rolling_std[i] = np.std(eda[i-win_samples:i])
    else:
        rolling_std = np.array([
            np.std(eda[max(0, i-win_samples):i]) if i > 0 else 0
            for i in range(len(eda))
        ])
    
    # Find flat regions (using relative threshold)
    is_flat = rolling_std < relative_threshold
    
    # Convert to intervals
    flat_intervals = _bool_to_intervals(is_flat, fs)
    
    # Filter by minimum duration
    flat_intervals = [(s, e) for s, e in flat_intervals if (e - s) >= min_duration_s]
    
    return flat_intervals


def detect_jumps(
    eda: np.ndarray,
    fs: float,
    threshold_us: float = JUMP_THRESHOLD_US,
) -> List[Tuple[float, float]]:
    """
    Detect sudden jumps or drops in EDA signal.
    
    Returns: List of (start_s, end_s) tuples marking jump locations
    """
    diff = np.abs(np.diff(eda))
    jump_idx = np.where(diff > threshold_us)[0]
    
    if len(jump_idx) == 0:
        return []
    
    # Mark ±0.5s around each jump
    margin_samples = int(0.5 * fs)
    intervals = []
    for idx in jump_idx:
        start_s = max(0, idx - margin_samples) / fs
        end_s = min(len(eda) - 1, idx + margin_samples) / fs
        intervals.append((start_s, end_s))
    
    # Merge overlapping
    return _merge_intervals([(s, e) for s, e in intervals])


def detect_saturation(
    eda: np.ndarray,
    fs: float,
    high_threshold: float = SATURATION_HIGH_US,
    low_threshold: float = SATURATION_LOW_US,
) -> List[Tuple[float, float]]:
    """
    Detect saturation (clipping) or detachment (zero/negative values).
    
    Returns: List of (start_s, end_s) tuples
    """
    is_saturated = (eda >= high_threshold) | (eda <= low_threshold)
    return _bool_to_intervals(is_saturated, fs)


def detect_all_artifacts(
    eda: np.ndarray,
    fs: float,
    flat_min_dur_s: float = FLAT_LINE_THRESHOLD_S,
    flat_std_threshold: float = FLAT_LINE_STD_THRESHOLD,
    jump_threshold: float = JUMP_THRESHOLD_US,
    saturation_high: float = SATURATION_HIGH_US,
    saturation_low: float = SATURATION_LOW_US,
    detect_flat_lines_flag: bool = FLAT_LINE_ENABLED,
    verbose: bool = True,
) -> List[Tuple[float, float, str]]:
    """
    Detect all artifact types and return consolidated list.
    
    Returns: List of (start_s, end_s, reason) tuples
    """
    all_artifacts = []
    
    # Flat lines (disabled by default - too many false positives for slow EDA)
    if detect_flat_lines_flag:
        flat = detect_flat_lines(eda, fs, flat_min_dur_s, flat_std_threshold)
        for s, e in flat:
            all_artifacts.append((s, e, "flat_line"))
    else:
        flat = []
    
    # Jumps
    jumps = detect_jumps(eda, fs, jump_threshold)
    for s, e in jumps:
        all_artifacts.append((s, e, "jump"))
    
    # Saturation
    sat = detect_saturation(eda, fs, saturation_high, saturation_low)
    for s, e in sat:
        all_artifacts.append((s, e, "saturation"))
    
    if verbose:
        total = len(flat) + len(jumps) + len(sat)
        if total > 0:
            print(f"  ⚠️  {total} artifact zone(s) detected (marked in yellow on plots)")
            if len(flat) > 0:
                print(f"     - {len(flat)} zone(s) with very low variability (possible electrode issue)")
            if len(jumps) > 0:
                print(f"     - {len(jumps)} zone(s) with sudden changes")
            if len(sat) > 0:
                print(f"     - {len(sat)} zone(s) with saturation/clipping")
        else:
            print(f"  ✓ No artifacts detected")
        if not detect_flat_lines_flag:
            print(f"     (Flat line detection disabled - too many false positives for slow EDA)")
    
    return all_artifacts


def _bool_to_intervals(mask: np.ndarray, fs: float) -> List[Tuple[float, float]]:
    """Convert boolean mask to list of (start_s, end_s) intervals."""
    if not np.any(mask):
        return []
    
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    # Handle edge cases
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    
    intervals = [(s / fs, e / fs) for s, e in zip(starts, ends)]
    return intervals


def _merge_intervals(intervals: List[Tuple[float, float]], margin: float = 0.0) -> List[Tuple[float, float]]:
    """Merge overlapping intervals."""
    if not intervals:
        return []
    
    intervals = sorted(intervals)
    merged = [intervals[0]]
    
    for a, b in intervals[1:]:
        last_a, last_b = merged[-1]
        if a <= last_b + margin:
            merged[-1] = (last_a, max(last_b, b))
        else:
            merged.append((a, b))
    
    return merged


# =============================================================================
# FILTERING (Minimal, as per guidelines)
# =============================================================================

def filter_eda(
    eda: np.ndarray,
    fs: float,
    lowpass_cutoff: float = LOWPASS_CUTOFF_HZ,
    method: str = "neurokit",
) -> np.ndarray:
    """
    Apply minimal low-pass filtering to EDA signal using NeuroKit2's eda_clean().
    
    Per NeuroKit2 documentation:
    - NeuroKit method: Low-pass filter with 3 Hz cutoff and 4th order Butterworth filter
    - BioSPPy method: More aggressive filtering (5 Hz cutoff)
    - If sampling rate < 7 Hz (e.g., wearables), filtering is skipped
    
    Note: When using cvxEDA decomposition method, filtering is technically optional
    because the cvxEDA model incorporates noise directly into the mathematical model
    as Gaussian white noise. However, applying a gentle low-pass filter does no harm
    and may help with visualization. For other decomposition methods, filtering is recommended.
    
    Args:
        eda: Raw EDA signal
        fs: Sampling frequency
        lowpass_cutoff: Low-pass filter cutoff (Hz) - only used if method != "neurokit"
        method: 'neurokit' (recommended, uses nk.eda_clean), 'biosppy', 'butterworth', or 'savgol'
    
    Returns:
        Filtered EDA signal
    """
    if method == "neurokit":
        # Use NeuroKit2's official cleaning function (recommended)
        # This handles edge cases like low sampling rates automatically
        try:
            eda_clean = nk.eda_clean(eda, sampling_rate=int(fs), method="neurokit")
            return eda_clean
        except Exception as e:
            print(f"  ⚠️  NeuroKit cleaning failed: {e}, falling back to custom filter")
            method = "butterworth"  # Fallback
    
    if method == "biosppy":
        # More aggressive filtering (5 Hz cutoff)
        try:
            eda_clean = nk.eda_clean(eda, sampling_rate=int(fs), method="biosppy")
            return eda_clean
        except Exception as e:
            print(f"  ⚠️  BioSPPy cleaning failed: {e}, falling back to custom filter")
            method = "butterworth"  # Fallback
    
    # Custom filtering methods (fallback or explicit choice)
    if not _HAVE_SCIPY:
        # Simple moving average fallback
        win = int(fs / lowpass_cutoff)
        if win < 2:
            win = 2
        return uniform_filter1d(eda.astype(float), size=win)
    
    if method == "butterworth":
        # filtfilt needs len(x) > padlen (for order-4, padlen is typically 15)
        MIN_LEN_FILTFILT = 20
        if len(eda) <= MIN_LEN_FILTFILT:
            print(f"  [filter_eda] Signal too short ({len(eda)} samples), skipping low-pass filter")
            return eda.astype(float)
        # 4th order Butterworth low-pass (matches NeuroKit default)
        nyq = fs / 2.0
        normalized_cutoff = min(lowpass_cutoff / nyq, 0.99)
        b, a = scipy_signal.butter(4, normalized_cutoff, btype='low')
        return scipy_signal.filtfilt(b, a, eda)
    
    elif method == "savgol":
        # Savitzky-Golay filter (preserves peaks better)
        win = int(fs / lowpass_cutoff)
        if win % 2 == 0:
            win += 1
        if win < 5:
            win = 5
        return scipy_signal.savgol_filter(eda, win, 3)
    
    else:
        raise ValueError(f"Unknown filter method: {method}. Use 'neurokit', 'biosppy', 'butterworth', or 'savgol'")


# =============================================================================
# TONIC/PHASIC DECOMPOSITION
# =============================================================================

def decompose_eda(
    eda: np.ndarray,
    fs: float,
    method: str = "cvxeda",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose EDA into tonic (SCL) and phasic (SCR driver) components.
    
    This is a CORE step in EDA preprocessing.
    
    Methods:
        - 'cvxeda': Convex optimization (recommended, model-based)
            * Note: cvxEDA authors state that traditional preprocessing (e.g., bandpass
              filters) is not required because noise is incorporated into the model as
              Gaussian white noise. A gentle low-pass filter can still be applied but is
              not strictly necessary.
        - 'highpass': Simple high-pass filter (less recommended)
        - 'median': Median filter for tonic estimation
        - 'smoothmedian': NeuroKit2's smoothed median approach
    
    Args:
        eda: Cleaned EDA signal (or raw if using cvxEDA)
        fs: Sampling frequency
        method: Decomposition method
    
    Returns:
        (tonic, phasic) arrays
    """
    if method == "cvxeda":
        # Use neurokit2's cvxEDA implementation
        # cvxEDA uses these internal parameters (Greco et al., 2016):
        #   - alpha = 8e-4 (sparsity penalty for phasic component)
        #   - gamma = 1e-2 (smoothness penalty for tonic component)
        #   - tau1 = 0.7s (fast time constant)
        #   - tau0 = 2.0s (slow time constant)
        try:
            decomposed = nk.eda_phasic(eda, sampling_rate=int(fs), method="cvxeda")
            tonic = decomposed["EDA_Tonic"].values
            phasic = decomposed["EDA_Phasic"].values
            print("Using cvxEDA")
        except (ImportError, ValueError) as e:
            # cvxEDA requires cvxopt package - if not available, smoothmedian is also acceptable
            print(f"  ⚠️  cvxEDA error: {type(e).__name__}")
            print(f"  → Falling back to smoothmedian (also acceptable per SPR guidelines)")
            try:
                decomposed = nk.eda_phasic(eda, sampling_rate=int(fs), method="smoothmedian")
                tonic = decomposed["EDA_Tonic"].values
                phasic = decomposed["EDA_Phasic"].values
            except Exception as e2:
                print(f"  ✗ Decomposition failed: {e2}")
                raise
        except Exception as e:
            print(f"  ✗ Unexpected decomposition error: {e}")
            raise
    
    elif method == "highpass":
        # Simple high-pass for phasic (NOT recommended per guidelines)
        if _HAVE_SCIPY:
            # Very low cutoff (0.05 Hz) to preserve SCR shape
            nyq = fs / 2.0
            b, a = scipy_signal.butter(2, 0.05 / nyq, btype='high')
            phasic = scipy_signal.filtfilt(b, a, eda)
            tonic = eda - phasic
        else:
            # Fallback: subtract moving average
            win = int(20 * fs)  # 20 second window
            tonic = uniform_filter1d(eda.astype(float), size=win)
            phasic = eda - tonic
    
    elif method == "median":
        # Median filter for tonic (simple, conservative)
        win = int(4 * fs)  # 4 second window
        if win % 2 == 0:
            win += 1
        if _HAVE_SCIPY:
            tonic = scipy_signal.medfilt(eda, kernel_size=win)
        else:
            tonic = np.array([np.median(eda[max(0, i-win//2):i+win//2+1]) for i in range(len(eda))])
        phasic = eda - tonic
    
    elif method == "smoothmedian":
        # NeuroKit2's default method
        decomposed = nk.eda_phasic(eda, sampling_rate=int(fs), method="smoothmedian")
        tonic = decomposed["EDA_Tonic"].values
        phasic = decomposed["EDA_Phasic"].values
    
    else:
        raise ValueError(f"Unknown decomposition method: {method}")
    
    return tonic, phasic


# =============================================================================
# SCR DETECTION
# =============================================================================

def detect_scrs(
    phasic: np.ndarray,
    fs: float,
    min_amplitude: float = SCR_MIN_AMPLITUDE_US,
    min_rise_time: float = SCR_MIN_RISE_TIME_S,
    max_rise_time: float = SCR_MAX_RISE_TIME_S,
    method: str = "neurokit",
) -> Dict[str, np.ndarray]:
    """
    Detect Skin Conductance Responses (SCRs) from phasic component using NeuroKit2.
    
    Per NeuroKit2 documentation, eda_findpeaks() expects a phasic signal array directly,
    not a DataFrame. This function uses the correct API and extracts all SCR features.
    
    Args:
        phasic: Phasic EDA component (array)
        fs: Sampling frequency
        min_amplitude: Minimum SCR amplitude (µS) - relative threshold for 'neurokit' method
        min_rise_time: Minimum rise time (s)
        max_rise_time: Maximum rise time (s)
        method: Detection method ('neurokit', 'gamboa2008', 'kim2004', 'vanhalem2020', 'nabian2018')
    
    Returns:
        Dict with 'onsets', 'peaks', 'amplitudes', 'rise_times', 'recovery' arrays
    """
    # Handle NaN values (replace with 0 for detection, but track original positions)
    phasic_clean = phasic.copy()
    nan_mask = np.isnan(phasic_clean)
    if np.any(nan_mask):
        phasic_clean[nan_mask] = 0.0
    
    # Use neurokit2 for SCR detection - CORRECT API: pass array directly
    try:
        # eda_findpeaks expects phasic signal array, not DataFrame
        info = nk.eda_findpeaks(
            phasic_clean, 
            sampling_rate=int(fs), 
            method=method,
            amplitude_min=min_amplitude
        )
        
        scr_peaks_raw = np.asarray(info.get("SCR_Peaks", []), dtype=float)
        
        # Sanitize peaks: remove NaN, out-of-bounds, and artifact regions
        def _sanitize_peaks(raw: np.ndarray) -> np.ndarray:
            if raw.size == 0 or len(phasic) == 0:
                return np.array([], dtype=int)
            valid_mask = (
                (raw >= 0) &
                (raw < len(phasic)) &
                np.isfinite(raw)
            )
            if not np.any(valid_mask):
                return np.array([], dtype=int)
            clipped = np.clip(raw[valid_mask].astype(int), 0, len(phasic) - 1)
            artifact_mask = ~nan_mask[clipped]
            return clipped[artifact_mask]

        scr_peaks = _sanitize_peaks(scr_peaks_raw)
        
        # For each peak, find onset by looking backward for local minimum
        def _find_onset(peak_idx: int, search_window_s: float = 4.0) -> int:
            """Find SCR onset by looking backward from peak for local minimum."""
            search_samples = int(search_window_s * fs)
            start = max(0, peak_idx - search_samples)
            window = phasic[start:peak_idx]
            if len(window) == 0:
                return start
            # Find minimum in window
            min_idx = np.argmin(window)
            return start + min_idx
        
        # Process each peak
        amplitudes = []
        valid_onsets = []
        valid_peaks = []
        valid_rise_times = []
        recovery_indices = []
        
        for peak_idx in scr_peaks:
            if np.isnan(phasic[peak_idx]):
                continue
            
            # Find onset
            onset_idx = _find_onset(peak_idx)
            
            if onset_idx >= peak_idx or np.isnan(phasic[onset_idx]):
                continue
            
            amp = phasic[peak_idx] - phasic[onset_idx]
            rise_time = (peak_idx - onset_idx) / fs
            
            # Filter by amplitude and rise time
            if amp >= min_amplitude and min_rise_time <= rise_time <= max_rise_time:
                amplitudes.append(amp)
                valid_onsets.append(onset_idx)
                valid_peaks.append(peak_idx)
                valid_rise_times.append(rise_time)
                
                # Find recovery point (half-amplitude after peak)
                recovery_idx = _find_recovery_point(
                    phasic, peak_idx, onset_idx, fs
                )
                recovery_indices.append(recovery_idx)
            
        return {
            "onsets": np.array(valid_onsets, dtype=int),
            "peaks": np.array(valid_peaks, dtype=int),
            "amplitudes": np.array(amplitudes),
            "rise_times": np.array(valid_rise_times),
            "recovery": np.array(recovery_indices, dtype=int),
        }
    
    except Exception as e:
        print(f"  ⚠️  SCR detection error: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        "onsets": np.array([], dtype=int),
        "peaks": np.array([], dtype=int),
        "amplitudes": np.array([]),
        "rise_times": np.array([]),
        "recovery": np.array([], dtype=int),
    }


def _find_recovery_point(
    phasic: np.ndarray,
    peak_idx: int,
    onset_idx: int,
    fs: float,
) -> int:
    """
    Find recovery point (half-amplitude) after SCR peak.
    
    Returns the index where phasic component drops to half amplitude,
    or len(phasic) if not found within reasonable time.
    """
    if peak_idx >= len(phasic) - 1:
        return len(phasic) - 1
    
    peak_val = phasic[peak_idx]
    onset_val = phasic[onset_idx]
    half_amplitude = onset_val + (peak_val - onset_val) / 2.0
    
    # Look forward from peak (up to 10 seconds)
    max_lookahead = min(len(phasic), peak_idx + int(10 * fs))
    
    for i in range(peak_idx + 1, max_lookahead):
        if np.isnan(phasic[i]):
            continue
        if phasic[i] <= half_amplitude:
            return i
    
    # Not found within reasonable time
    return max_lookahead - 1


# =============================================================================
# ENLARGED WINDOW DETECTION
# =============================================================================

def _choose_best_scr_window(
    times: np.ndarray,
    scr_peaks: np.ndarray,
    fs: float,
    duration_s: float = 300.0,
) -> Tuple[int, int]:
    """
    Choose the 300s window that covers the most SCR peaks.
    """
    n = len(times)
    trim_samples = int(duration_s * fs)
    if n <= trim_samples or trim_samples == 0:
        return 0, n

    scr_times = times[scr_peaks] if len(scr_peaks) > 0 else np.array([])
    if len(scr_times) == 0:
        start_idx = n - trim_samples
    else:
        max_start_s = max(times[-1] - duration_s, 0.0)
        step = 1.0
        starts = np.arange(0.0, max_start_s + step, step)
        if len(starts) == 0:
            start_idx = n - trim_samples
        else:
            ends = starts + duration_s
            left = np.searchsorted(scr_times, starts, side="left")
            right = np.searchsorted(scr_times, ends, side="right")
            counts = right - left
            max_count = np.max(counts)
            winners = np.where(counts == max_count)[0]
            best_start_s = starts[winners[-1]]
            start_idx = int(np.searchsorted(times, best_start_s, side="left"))

    end_idx = min(start_idx + trim_samples, n)
    if end_idx - start_idx < trim_samples:
        start_idx = max(0, n - trim_samples)
        end_idx = n

    return start_idx, end_idx

def detect_enlarged_windows(
    eda_clean: np.ndarray,
    phasic: np.ndarray,
    times: np.ndarray,
    scr_peaks: np.ndarray,
    fs: float,
    derivative_threshold: float = ENLARGED_DERIVATIVE_THRESHOLD_US_S,
    phasic_amplitude_min: float = ENLARGED_PHASIC_AMPLITUDE_MIN_US,
    phasic_amplitude_max: float = ENLARGED_PHASIC_AMPLITUDE_MAX_US,
    scr_proximity_s: float = ENLARGED_SCR_PROXIMITY_S,
    margin_s: float = ENLARGED_WINDOW_MARGIN_S,
) -> List[Tuple[float, float, str]]:
    """
    Detect regions that need enlarged inspection.
    
    Conditions:
    1. |dEDA/dt| exceeds threshold
    2. Phasic amplitude exceeds threshold (0.05-0.1 µS range)
    3. SCRs very close to each other
    
    Returns:
        List of (start_s, end_s, reason) tuples for regions needing enlarged view
    """
    regions = []
    
    # 1. Detect high derivative regions
    if len(eda_clean) > 1:
        derivative = np.abs(np.diff(eda_clean)) * fs  # Convert to µS/s
        high_deriv_mask = derivative > derivative_threshold
        
        if np.any(high_deriv_mask):
            # Convert to time intervals
            high_deriv_intervals = _bool_to_intervals(high_deriv_mask, fs)
            for start_s, end_s in high_deriv_intervals:
                # Add margin
                start_s = max(0.0, start_s - margin_s)
                end_s = min(times[-1] if len(times) > 0 else 0.0, end_s + margin_s)
                if end_s > start_s:
                    regions.append((start_s, end_s, "high_derivative"))
    
    # 2. Detect high phasic amplitude regions
    if len(phasic) > 0:
        # Use absolute value of phasic component
        phasic_abs = np.abs(phasic)
        high_phasic_mask = phasic_abs > phasic_amplitude_min
        
        if np.any(high_phasic_mask):
            high_phasic_intervals = _bool_to_intervals(high_phasic_mask, fs)
            for start_s, end_s in high_phasic_intervals:
                # Add margin
                start_s = max(0.0, start_s - margin_s)
                end_s = min(times[-1] if len(times) > 0 else 0.0, end_s + margin_s)
                if end_s > start_s:
                    regions.append((start_s, end_s, "high_phasic_amplitude"))
    
    # 3. Detect closely spaced SCRs
    if len(scr_peaks) > 1:
        scr_times = times[scr_peaks]
        scr_times_sorted = np.sort(scr_times)
        time_diffs = np.diff(scr_times_sorted)
        close_pairs = np.where(time_diffs < scr_proximity_s)[0]
        
        if len(close_pairs) > 0:
            for idx in close_pairs:
                t1 = scr_times_sorted[idx]
                t2 = scr_times_sorted[idx + 1]
                start_s = max(0.0, t1 - margin_s)
                end_s = min(times[-1] if len(times) > 0 else 0.0, t2 + margin_s)
                if end_s > start_s:
                    regions.append((start_s, end_s, "close_scrs"))
    
    # Merge overlapping regions
    if len(regions) > 0:
        # Sort by start time
        regions = sorted(regions, key=lambda x: x[0])
        merged = []
        current_start, current_end, current_reasons = regions[0]
        current_reasons_set = {current_reasons}
        
        for start_s, end_s, reason in regions[1:]:
            if start_s <= current_end:
                # Overlapping - merge
                current_end = max(current_end, end_s)
                current_reasons_set.add(reason)
            else:
                # Non-overlapping - save current and start new
                merged.append((current_start, current_end, "+".join(sorted(current_reasons_set))))
                current_start, current_end = start_s, end_s
                current_reasons_set = {reason}
        
        # Add last region
        merged.append((current_start, current_end, "+".join(sorted(current_reasons_set))))
        return merged
    
    return []


# =============================================================================
# SEGMENT INSPECTION (Visual QC)
# =============================================================================

def inspeccionar_segmento_eda(
    eda_raw: np.ndarray,
    fs_eda: float,
    eventos: Dict[str, int],
    seg_name: str = "LB",
    win_len: Optional[float] = None,  # None = auto (full segment if <6min, else 180s windows)
    target_fs: float = TARGET_FS_HZ,
    decomposition_method: str = "cvxeda",
    segmentos_dict: Optional[Dict[str, Tuple[int, int]]] = None,
) -> dict:
    """
    Inspect and process an EDA segment with visualization.
    
    Per SPR guidelines:
    - Visual inspection should focus on RAW signal first (artifacts, stability)
    - Tonic (SCL) should be checked for baseline stability
    - Phasic component is primarily for SCR detection (less critical for visual QC)
    
    Steps:
    1. Extract segment using markers
    2. Downsample to target frequency
    3. Detect artifacts
    4. Apply minimal filtering (optional for cvxEDA, but applied by default)
    5. Decompose into tonic/phasic
    6. Detect SCRs
    7. Visualize in windows
    
    Note: When using cvxEDA, the low-pass filter is technically optional because
    the cvxEDA model handles noise internally. However, it's still applied by default
    for consistency and visualization purposes.
    
    Args:
        win_len: Window length in seconds. If None, auto-adjusts:
                - Full segment if duration < 6 minutes
                - 180s windows if duration >= 6 minutes
    
    Returns: seg_dict with all processed data
    """
    if plt is None:
        print("matplotlib not installed; no plots will be generated.")
        return None
    
    print("\n" + "=" * 60)
    print(f"Inspecting EDA segment: {seg_name}")
    print("=" * 60)
    
    n_total = len(eda_raw)
    
    # Get segment boundaries
    if segmentos_dict is not None and seg_name in segmentos_dict:
        start_idx, end_idx = segmentos_dict[seg_name]
    else:
        start_idx, end_idx = get_segment_indices(eventos, seg_name, n_total)
    
    dur_s = (end_idx - start_idx) / fs_eda
    print(f"Segment {seg_name}: duration = {dur_s:.1f} s")
    
    # Extract segment
    eda_seg_raw = eda_raw[start_idx:end_idx].copy()
    
    # Downsample segment
    eda_downsampled, fs_down = downsample_signal(eda_seg_raw, fs_eda, target_fs)
    print(f"Downsampled: {fs_eda:.0f} Hz → {fs_down:.0f} Hz ({len(eda_downsampled)} samples)")
    
    # Create time vector
    times = np.arange(len(eda_downsampled)) / fs_down
    dur_seg = float(times[-1]) if len(times) else 0.0
    
    # Artifact detection disabled - bad zones should only be marked manually using marcar_bad_interval()
    artifacts = []
    
    # Filter using NeuroKit2's eda_clean() (recommended)
    # Note: For cvxEDA, filtering is technically optional (noise is handled by the model),
    # but a gentle low-pass filter can still be applied for visualization purposes.
    # NeuroKit2's eda_clean() handles edge cases (e.g., low sampling rates) automatically.
    eda_clean = filter_eda(eda_downsampled, fs_down, method="neurokit")
    
    # Decompose
    try:
        tonic, phasic = decompose_eda(eda_clean, fs_down, method=decomposition_method)
    except Exception as e:
        print(f"  ✗ Decomposition error: {e}")
        tonic = eda_clean.copy()
        phasic = np.zeros_like(eda_clean)
    
    # Auto-trim if duration > 300s (keep first 300s)
    if dur_seg > 300.0:
        excess = dur_seg - 300.0
        keep_samples = int(300.0 * fs_down)
        eda_downsampled = eda_downsampled[:keep_samples]
        times = times[:keep_samples]
        eda_clean = eda_clean[:keep_samples]
        tonic = tonic[:keep_samples]
        phasic = phasic[:keep_samples]
        dur_seg = float(times[-1]) if len(times) else 0.0
        dur_s = dur_seg  # so export and summary show 300s, not original segment length
        print(f"  ⚠️  Auto-trimmed: kept first 300.0s, removed last {excess:.1f}s")

    # Detect SCRs using improved NeuroKit2-based detection (after trimming)
    scr_info = detect_scrs(phasic, fs_down, method="neurokit")
    
    # Artifact detection disabled - bad zones should only be marked manually using marcar_bad_interval()
    # artifacts list remains empty (already set above)
    
    # Calculate statistics (using all data - bad zones not yet marked as NaN)
    # After marking bad zones as NaN, SCL will be recalculated using only clean data
    scl_mean = np.nanmean(tonic)
    scl_std = np.nanstd(tonic)
    
    print("\n--- SCR ---")
    n_scrs = len(scr_info["peaks"])
    if n_scrs > 0:
        print(f" SCRs detected: {n_scrs}")
        print(f" Mean amplitude: {np.mean(scr_info['amplitudes']):.3f} µS")
        print(f" Mean rise time: {np.mean(scr_info['rise_times']):.2f} s")
        if len(scr_info["recovery"]) > 0:
            recovery_times = (scr_info["recovery"] - scr_info["peaks"]) / fs_down
            print(f" Mean recovery time: {np.mean(recovery_times):.2f} s")
    else:
        print(f" SCRs: 0 (normal for baseline/rest segments)")
    
    print("\n--- SCL ---")
    print(f" SCL mean: {scl_mean:.3f} µS")
    print(f" SCL std: {scl_std:.3f} µS")
    
    # Auto-adjust window length per guidelines (EDA is slow, longer windows help)
    if win_len is None:
        if dur_seg < 360:  # < 6 minutes: show full segment
            win_len = dur_seg
        else:  # >= 6 minutes: use 180s windows
            win_len = 180.0

    # Visualize
    n_windows = int(math.ceil(dur_seg / win_len)) if dur_seg > 0 else 0
    
    print()  # Empty line before window info
    
    for k in range(n_windows):
        t0 = k * win_len
        win = min(win_len, dur_seg - t0)
        if win <= 0:
            break
        t_end = t0 + win
        print(f"Window {k+1}/{n_windows}: [{t0:.1f}s - {t_end:.1f}s]")
        plot_eda_window(
            eda_raw=eda_downsampled,
            eda_clean=eda_clean,
            tonic=tonic,
            phasic=phasic,
            times=times,
            scr_peaks=scr_info['peaks'],
            scr_onsets=scr_info.get('onsets', np.array([], dtype=int)),
            scr_recovery=scr_info.get('recovery', np.array([], dtype=int)),
            scr_amplitudes=scr_info.get('amplitudes', np.array([])),
            artifacts=artifacts,
            t0=t0,
            win_len=win,
            seg_name=seg_name,
            fs=fs_down,
            show_onsets_recovery=False,  # Main inspection: only peaks
        )
    
    # Build result dict
    # Note: enlarged_windows detection removed - use inspeccionar_ventana_local() manually if needed
    enlarged_regions = []  # Empty list since we removed automatic detection
    
    seg_dict = {
        "seg": seg_name,
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "dur_s": float(dur_s),
        "fs_original": float(fs_eda),
        "fs_eda": float(fs_down),
        "times": times,
        "eda_raw": eda_downsampled,
        "eda_clean": eda_clean,
        "eda_tonic": tonic,
        "eda_phasic": phasic,
        "scr_onsets": scr_info.get("onsets", np.array([], dtype=int)),
        "scr_peaks": scr_info.get("peaks", np.array([], dtype=int)),
        "scr_amplitudes": scr_info.get("amplitudes", np.array([])),
        "scr_rise_times": scr_info.get("rise_times", np.array([])),
        "scr_recovery": scr_info.get("recovery", np.array([], dtype=int)),
        "bad_intervals": pd.DataFrame(columns=["t_ini_s", "t_fin_s", "motivo", "sev"]),
        "artifacts": artifacts,
        "enlarged_windows": enlarged_regions,  # Empty - use inspeccionar_ventana_local() for manual inspection
        "scl_mean": scl_mean,
        "scl_std": scl_std,
    }
    
    # Artifacts are detected and stored but NOT automatically marked as bad
    # Bad zones should only be marked manually using marcar_bad_interval() after inspection
    
    # Calculate BAD percentage (will be 0% if no bad zones manually marked)
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    
    return seg_dict


def plot_eda_window_with_presegment(
    eda_pre_raw: np.ndarray,
    eda_pre_clean: np.ndarray,
    tonic_pre: np.ndarray,
    phasic_pre: np.ndarray,
    times_pre: np.ndarray,
    eda_seg_raw: np.ndarray,
    eda_seg_clean: np.ndarray,
    tonic_seg: np.ndarray,
    phasic_seg: np.ndarray,
    times_seg: np.ndarray,
    scr_peaks: np.ndarray,
    artifacts: List[Tuple[float, float, str]],
    seg_name: str,
    fs: float,
):
    """Plot pre-segment data (5s before) and segment start for normalization reference."""
    if plt is None:
        return
    
    if len(times_pre) == 0 or len(times_seg) == 0:
        return
    
    # Show all 5 seconds of pre-segment and first 30 seconds of segment
    seg_window_s = 30.0   # First 30 seconds of segment
    
    # Pre-segment: all 5 seconds
    pre_i0, pre_i1 = 0, len(times_pre)
    t_pre_win = times_pre[pre_i0:pre_i1]
    
    # Segment: first 30 seconds
    seg_mask = times_seg <= seg_window_s
    seg_idx = np.where(seg_mask)[0]
    if len(seg_idx) == 0:
        seg_idx = np.arange(min(len(times_seg), int(seg_window_s * fs)))
    seg_i0, seg_i1 = 0, min(seg_idx[-1] + 1, len(times_seg))
    t_seg_win = times_seg[seg_i0:seg_i1]
    
    # Combine times (pre-segment negative, segment positive)
    t_combined = np.concatenate([t_pre_win, t_seg_win])
    
    fig, axs = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(f"{seg_name} – Pre-segment (5s before) + Segment Start (for normalization reference)", 
                 fontsize=12, fontweight='bold')
    
    lw = 0.8
    
    # Plot 1: Raw EDA
    ax = axs[0]
    ax.plot(t_pre_win, eda_pre_raw[pre_i0:pre_i1], lw=lw, color=PLOT_COLORS["raw"], 
           alpha=0.7, label="Pre-segment (raw)")
    ax.plot(t_seg_win, eda_seg_raw[seg_i0:seg_i1], lw=lw, color=PLOT_COLORS["raw"], 
           label="Segment (raw)")
    # Vertical line at segment start
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label="Segment start")
    ax.set_ylabel("Raw EDA (µS)", fontsize=10, fontweight='bold')
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, "Pre-segment (5s before)", transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Plot 2: Tonic (SCL)
    ax = axs[1]
    ax.plot(t_pre_win, tonic_pre[pre_i0:pre_i1], lw=lw, color=PLOT_COLORS["tonic"], 
           alpha=0.7, label="Pre-segment (tonic)")
    ax.plot(t_seg_win, tonic_seg[seg_i0:seg_i1], lw=lw, color=PLOT_COLORS["tonic"], 
           label="Segment (tonic)")
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel("Tonic / SCL (µS)", fontsize=10, fontweight='bold')
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Phasic
    ax = axs[2]
    ax.plot(t_pre_win, phasic_pre[pre_i0:pre_i1], lw=lw, color=PLOT_COLORS["phasic"], 
           alpha=0.7, label="Pre-segment (phasic)")
    ax.plot(t_seg_win, phasic_seg[seg_i0:seg_i1], lw=lw, color=PLOT_COLORS["phasic"], 
           label="Segment (phasic)")
    
    # Mark SCR peaks in segment
    peaks_in_window = scr_peaks[(scr_peaks >= seg_i0) & (scr_peaks < seg_i1)]
    if len(peaks_in_window) > 0:
        peak_times = times_seg[peaks_in_window]
        peak_vals = phasic_seg[peaks_in_window]
        ax.scatter(peak_times, peak_vals, s=60, c=PLOT_COLORS["scr"], 
                  marker='o', edgecolors='black', linewidths=1.5, zorder=5, label="SCR Peaks")
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel("Phasic (µS)", fontsize=10, fontweight='bold')
    ax.set_xlabel("Time (s) - 0 = Segment start", fontsize=10, fontweight='bold')
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def plot_eda_window(
    eda_raw: np.ndarray,
    eda_clean: np.ndarray,
    tonic: np.ndarray,
    phasic: np.ndarray,
    times: np.ndarray,
    scr_peaks: np.ndarray,
    scr_onsets: np.ndarray = None,
    scr_recovery: np.ndarray = None,
    scr_amplitudes: np.ndarray = None,
    artifacts: List[Tuple[float, float, str]] = None,
    t0: float = 0.0,
    win_len: float = None,
    seg_name: str = "",
    fs: float = 32.0,
    show_onsets_recovery: bool = False,
):
    """Plot raw EDA, tonic, and tonic annotated with SCR markers."""
    if plt is None:
        return

    if artifacts is None:
        artifacts = []
    if scr_onsets is None:
        scr_onsets = np.array([], dtype=int)
    if scr_recovery is None:
        scr_recovery = np.array([], dtype=int)
    if scr_amplitudes is None:
        scr_amplitudes = np.array([])

    if win_len is None:
        win_len = times[-1] - t0 if len(times) > 0 else 60.0

    mask = (times >= t0) & (times < t0 + win_len)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print(f"Window [{t0:.1f},{t0+win_len:.1f}] s empty")
        return

    i0, i1 = idx[0], idx[-1] + 1
    t_win = times[i0:i1]
    t_end = t0 + win_len

    fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{seg_name} – [{t0:.1f}s - {t_end:.1f}s]", fontsize=12, fontweight='bold')
    lw = 0.8

    def shade_artifacts(ax):
        for start_s, end_s, _ in artifacts:
            if end_s < t0 or start_s > t_end:
                continue
            a_start = max(start_s, t0)
            a_end = min(end_s, t_end)
            ax.axvspan(a_start, a_end, color=PLOT_COLORS["bad"], alpha=0.3)

    peaks_in_window = scr_peaks[(scr_peaks >= i0) & (scr_peaks < i1)]
    onsets_in_window = scr_onsets[(scr_onsets >= i0) & (scr_onsets < i1)]
    recovery_in_window = scr_recovery[(scr_recovery >= i0) & (scr_recovery < i1)]

    peak_times = times[peaks_in_window] if len(peaks_in_window) > 0 else np.array([])
    onset_times = times[onsets_in_window] if len(onsets_in_window) > 0 else np.array([])
    recovery_times = times[recovery_in_window] if len(recovery_in_window) > 0 else np.array([])

    # Raw EDA
    ax = axs[0]
    ax.plot(t_win, eda_raw[i0:i1], lw=lw, color=PLOT_COLORS["raw"], label="Raw EDA")
    shade_artifacts(ax)
    ax.set_ylabel("Raw EDA (µS)", fontsize=10, fontweight='bold')
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Tonic / SCL (clean)
    ax = axs[1]
    ax.plot(t_win, tonic[i0:i1], lw=lw, color=PLOT_COLORS["tonic"], label="Tonic / SCL")
    shade_artifacts(ax)
    ax.set_ylabel("Tonic (µS)", fontsize=10, fontweight='bold')
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Raw EDA with SCR peaks (main inspection: peaks only, no annotations)
    ax = axs[2]
    ax.plot(t_win, eda_raw[i0:i1], lw=lw, color=PLOT_COLORS["raw"], label="Raw EDA")

    if len(peaks_in_window) > 0:
        # Mark SCR peaks on raw EDA (no amplitude annotations for main inspection)
        raw_peaks = eda_raw[peaks_in_window]
        ax.scatter(peak_times, raw_peaks, s=70, c=PLOT_COLORS["scr"],
                   edgecolors='black', linewidths=1.2, label="SCR Peaks", zorder=5)

    # Show onsets and recovery only if explicitly requested (for local inspection)
    if show_onsets_recovery:
        if len(onsets_in_window) > 0:
            raw_onsets = eda_raw[onsets_in_window]
            ax.scatter(onset_times, raw_onsets, s=50, c='green',
                       marker='v', edgecolors='black', linewidths=1, label="SCR Onsets", alpha=0.8, zorder=5)

        if len(recovery_in_window) > 0:
            raw_recovery = eda_raw[recovery_in_window]
            ax.scatter(recovery_times, raw_recovery, s=60, c='red',
                       marker='D', edgecolors='black', linewidths=1, label="Recovery (50%)", alpha=0.8, zorder=5)

    shade_artifacts(ax)
    ax.set_ylabel("Raw EDA + SCR peaks (µS)", fontsize=10, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=10, fontweight='bold')
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.show()


def plot_eda_enlarged_window(
    eda_raw: np.ndarray,
    eda_clean: np.ndarray,
    tonic: np.ndarray,
    phasic: np.ndarray,
    times: np.ndarray,
    scr_peaks: np.ndarray,
    scr_onsets: np.ndarray = None,
    scr_recovery: np.ndarray = None,
    scr_amplitudes: np.ndarray = None,
    artifacts: List[Tuple[float, float, str]] = None,
    t0: float = 0.0,
    win_len: float = 30.0,
    seg_name: str = "",
    fs: float = 32.0,
    reason: str = "",
):
    """
    Plot an enlarged window with detailed view including derivative and annotations.
    
    This provides a zoomed-in view for regions with:
    - High |dEDA/dt|
    - High phasic amplitude
    - Closely spaced SCRs
    """
    if plt is None:
        return
    
    mask = (times >= t0) & (times < t0 + win_len)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print(f"Enlarged window [{t0:.1f},{t0+win_len:.1f}] s empty")
        return
    
    i0, i1 = idx[0], idx[-1] + 1
    t_win = times[i0:i1]
    t_end = t0 + win_len
    
    # Calculate derivative for this window
    eda_win = eda_clean[i0:i1]
    derivative = np.abs(np.diff(eda_win)) * fs if len(eda_win) > 1 else np.array([])
    derivative_times = (t_win[:-1] + t_win[1:]) / 2 if len(t_win) > 1 else np.array([])
    
    fig, axs = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"{seg_name} – ENLARGED VIEW [{t0:.1f}s - {t_end:.1f}s] - {reason}", fontsize=12, fontweight='bold')
    
    lw = 1.2
    
    # Initialize defaults
    if artifacts is None:
        artifacts = []
    if scr_onsets is None:
        scr_onsets = np.array([], dtype=int)
    if scr_recovery is None:
        scr_recovery = np.array([], dtype=int)
    if scr_amplitudes is None:
        scr_amplitudes = np.array([])
    
    # Helper to shade artifacts
    def shade_artifacts(ax):
        for start_s, end_s, art_reason in artifacts:
            if end_s < t0 or start_s > t_end:
                continue
            a_start = max(start_s, t0)
            a_end = min(end_s, t_end)
            ax.axvspan(a_start, a_end, color=PLOT_COLORS["bad"], alpha=0.3)
    
    # Plot 1: Raw EDA and Filtered EDA (for comparison - detection is based on filtered)
    ax = axs[0]
    ax.plot(t_win, eda_raw[i0:i1], lw=lw*0.8, color=PLOT_COLORS["raw"], alpha=0.6, label="Raw EDA")
    ax.plot(t_win, eda_clean[i0:i1], lw=lw, color=PLOT_COLORS["clean"], label="Filtered EDA (used for detection)")
    shade_artifacts(ax)
    ax.set_ylabel("EDA (µS)", fontsize=10, fontweight='bold')
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Derivative |dEDA/dt|
    ax = axs[1]
    if len(derivative) > 0:
        ax.plot(derivative_times, derivative, lw=lw, color='purple', label="|dEDA/dt|")
        # Mark threshold
        threshold = ENLARGED_DERIVATIVE_THRESHOLD_US_S
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f"Threshold ({threshold:.2f} µS/s)")
    shade_artifacts(ax)
    ax.set_ylabel("|dEDA/dt| (µS/s)", fontsize=10, fontweight='bold')
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Phasic component with peaks, onsets, and recovery points
    ax = axs[2]
    ax.plot(t_win, phasic[i0:i1], lw=lw, color=PLOT_COLORS["phasic"], label="Phasic")
    
    # Mark SCR features
    peaks_in_window = scr_peaks[(scr_peaks >= i0) & (scr_peaks < i1)]
    onsets_in_window = scr_onsets[(scr_onsets >= i0) & (scr_onsets < i1)] if scr_onsets is not None and len(scr_onsets) > 0 else np.array([], dtype=int)
    recovery_in_window = scr_recovery[(scr_recovery >= i0) & (scr_recovery < i1)] if scr_recovery is not None and len(scr_recovery) > 0 else np.array([], dtype=int)
    
    if len(peaks_in_window) > 0:
        peak_times = times[peaks_in_window]
        peak_vals = phasic[peaks_in_window]
        ax.scatter(peak_times, peak_vals, s=80, c=PLOT_COLORS["scr"], zorder=5, 
                  marker='o', edgecolors='black', linewidths=1.5, label="SCR Peaks")
        
        # Annotate peak amplitudes (use provided amplitudes if available)
        for i, (pt, pv) in enumerate(zip(peak_times, peak_vals)):
            if scr_amplitudes is not None and i < len(scr_amplitudes):
                amp_str = f'{scr_amplitudes[i]:.3f} µS'
            else:
                amp_str = f'{pv:.3f} µS'
            ax.annotate(amp_str, xy=(pt, pv), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    if len(onsets_in_window) > 0:
        onset_times = times[onsets_in_window]
        onset_vals = phasic[onsets_in_window]
        ax.scatter(onset_times, onset_vals, s=50, c='green', zorder=5, 
                  marker='v', edgecolors='black', linewidths=1, label="SCR Onsets", alpha=0.7)
    
    if len(recovery_in_window) > 0:
        recovery_times = times[recovery_in_window]
        recovery_vals = phasic[recovery_in_window]
        ax.scatter(recovery_times, recovery_vals, s=60, c='red', zorder=5, 
                  marker='D', edgecolors='black', linewidths=1, label="Recovery (50%)", alpha=0.8)
    
    # Mark phasic amplitude threshold
    threshold_min = ENLARGED_PHASIC_AMPLITUDE_MIN_US
    threshold_max = ENLARGED_PHASIC_AMPLITUDE_MAX_US
    ax.axhline(y=threshold_min, color='orange', linestyle='--', alpha=0.7, 
              label=f"Threshold ({threshold_min:.3f} µS)")
    ax.axhline(y=threshold_max, color='red', linestyle='--', alpha=0.7, 
              label=f"Threshold ({threshold_max:.3f} µS)")
    ax.axhline(y=-threshold_min, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(y=-threshold_max, color='red', linestyle='--', alpha=0.5)
    
    shade_artifacts(ax)
    ax.set_ylabel("Phasic (µS)", fontsize=10, fontweight='bold')
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Tonic (SCL) for reference
    ax = axs[3]
    ax.plot(t_win, tonic[i0:i1], lw=lw, color=PLOT_COLORS["tonic"], label="Tonic / SCL")
    shade_artifacts(ax)
    ax.set_ylabel("Tonic / SCL (µS)", fontsize=10, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=10, fontweight='bold')
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()


def inspeccionar_ventana_local(
    seg_dict: dict,
    start_s: float,
    end_s: float,
    show_onsets_recovery: bool = True,
):
    """
    Optional local inspection helper.

    Call this manually after `inspeccionar_segmento_eda` when you want to
    zoom into a specific interval. It reuses the three-panel layout with
    all SCR features (onsets, peaks, recovery) shown.

    Example:
        # Inspect a specific 30-second window starting at 50s
        inspeccionar_ventana_local(resultado_LB, start_s=50.0, end_s=80.0)
    """
    times = seg_dict.get("times", np.array([]))
    if len(times) == 0:
        raise ValueError("Segment dictionary has no time vector.")

    mask = (times >= start_s) & (times <= end_s)
    if not np.any(mask):
        raise ValueError("Requested interval has no data.")

    idx = np.where(mask)[0]
    i0, i1 = idx[0], idx[-1] + 1

    # Get SCR features and adjust indices to window
    scr_peaks_full = seg_dict.get("scr_peaks", np.array([]))
    scr_onsets_full = seg_dict.get("scr_onsets", np.array([], dtype=int))
    scr_recovery_full = seg_dict.get("scr_recovery", np.array([], dtype=int))
    scr_amplitudes_full = seg_dict.get("scr_amplitudes", np.array([]))

    # Filter SCR features to those within the window and adjust indices
    peaks_in_window = scr_peaks_full[(scr_peaks_full >= i0) & (scr_peaks_full < i1)] - i0
    onsets_in_window = scr_onsets_full[(scr_onsets_full >= i0) & (scr_onsets_full < i1)] - i0 if len(scr_onsets_full) > 0 else np.array([], dtype=int)
    recovery_in_window = scr_recovery_full[(scr_recovery_full >= i0) & (scr_recovery_full < i1)] - i0 if len(scr_recovery_full) > 0 else np.array([], dtype=int)
    
    # Get amplitudes for peaks in window
    peak_indices_in_full = scr_peaks_full[(scr_peaks_full >= i0) & (scr_peaks_full < i1)]
    amplitudes_in_window = scr_amplitudes_full[np.isin(scr_peaks_full, peak_indices_in_full)] if len(scr_amplitudes_full) > 0 else np.array([])

    # Adjust artifact times to window
    artifacts_full = seg_dict.get("artifacts", [])
    artifacts_window = []
    for art_start, art_end, reason in artifacts_full:
        if art_end >= start_s and art_start <= end_s:
            artifacts_window.append((max(art_start, start_s) - start_s, min(art_end, end_s) - start_s, reason))

    plot_eda_window(
        eda_raw=seg_dict.get("eda_raw", np.array([]))[i0:i1],
        eda_clean=seg_dict.get("eda_clean", np.array([]))[i0:i1],
        tonic=seg_dict.get("eda_tonic", np.array([]))[i0:i1],
        phasic=seg_dict.get("eda_phasic", np.array([]))[i0:i1],
        times=times[i0:i1] - times[i0],  # Reset time to start at 0 for the window
        scr_peaks=peaks_in_window,
        scr_onsets=onsets_in_window,
        scr_recovery=recovery_in_window,
        scr_amplitudes=amplitudes_in_window,
        artifacts=artifacts_window,
        t0=0.0,  # Window starts at 0
        win_len=end_s - start_s,
        seg_name=f"{seg_dict.get('seg', 'SEG')} (local)",
        fs=seg_dict.get("fs_eda", TARGET_FS_HZ),
        show_onsets_recovery=show_onsets_recovery,
    )


def plot_eda_window_with_bad(
    eda_raw: np.ndarray,
    eda_clean: np.ndarray,
    tonic: np.ndarray,
    phasic: np.ndarray,
    times: np.ndarray,
    scr_peaks: np.ndarray,
    bad_intervals_df: pd.DataFrame,
    is_interpolated: np.ndarray,  # Now represents is_bad (zones marked as NaN, not interpolated)
    t0: float,
    win_len: float,
    seg_name: str,
    fs: float,
):
    """Plot EDA with BAD zones marked (as NaN, per literature guidelines - no interpolation)."""
    if plt is None:
        return
    
    mask = (times >= t0) & (times < t0 + win_len)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print(f"Window [{t0:.1f},{t0+win_len:.1f}] s empty")
        return
    
    i0, i1 = idx[0], idx[-1] + 1
    t_win = times[i0:i1]
    t_end = t0 + win_len
    is_bad_win = is_interpolated[i0:i1] if len(is_interpolated) == len(times) else np.zeros(len(t_win), dtype=bool)
    
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{seg_name} – [{t0:.1f}s - {t_end:.1f}s] (after cleaning)")
    
    lw = 0.8
    
    # Helper to shade BAD zones
    def shade_bad_zones(ax):
        if bad_intervals_df is not None and len(bad_intervals_df) > 0:
            for _, row in bad_intervals_df.iterrows():
                b_ini = float(row["t_ini_s"])
                b_fin = float(row["t_fin_s"])
                if b_fin < t0 or b_ini > t_end:
                    continue
                b_ini_clipped = max(t0, b_ini)
                b_fin_clipped = min(t_end, b_fin)
                # Yellow for BAD zones
                ax.axvspan(b_ini_clipped, b_fin_clipped, color=PLOT_COLORS["bad"], alpha=0.3)
                # Label
                center = (b_ini_clipped + b_fin_clipped) / 2
                y_pos = ax.get_ylim()[1] * 0.95
                ax.text(center, y_pos, f"[{int(row['id'])}]", 
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       color='black', bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    # Helper to mark bad zones (marked as NaN, not interpolated)
    def mark_bad_zones(ax):
        if np.any(is_bad_win):
            # Find contiguous bad regions (marked as NaN)
            diff = np.diff(np.concatenate(([False], is_bad_win, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                if s < len(t_win) and e <= len(t_win):
                    t_s = t_win[s] if s < len(t_win) else t_win[-1]
                    t_e = t_win[e-1] if e > 0 else t_win[0]
                    # Light red overlay for rejected zones (NaN)
                    ax.axvspan(t_s, t_e, color='lightcoral', alpha=0.2, label='Rejected (NaN)' if s == starts[0] else '')
    
    # Plot 1: Raw EDA
    ax = axs[0]
    ax.plot(t_win, eda_raw[i0:i1], lw=lw, color=PLOT_COLORS["raw"], label="Raw")
    shade_bad_zones(ax)
    ax.set_ylabel("Raw EDA (µS)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Clean EDA (with bad zones marked as NaN)
    ax = axs[1]
    ax.plot(t_win, eda_clean[i0:i1], lw=lw, color=PLOT_COLORS["clean"], label="Clean")
    mark_bad_zones(ax)  # Mark zones rejected as NaN (per literature: no interpolation)
    shade_bad_zones(ax)
    ax.set_ylabel("Clean EDA (µS)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Phasic (SCR) with peaks
    ax = axs[2]
    ax.plot(t_win, phasic[i0:i1], lw=lw, color=PLOT_COLORS["phasic"])
    
    # Mark SCR peaks
    peaks_in_window = scr_peaks[(scr_peaks >= i0) & (scr_peaks < i1)]
    if len(peaks_in_window) > 0:
        peak_times = times[peaks_in_window]
        peak_vals = phasic[peaks_in_window]
        ax.scatter(peak_times, peak_vals, s=30, c=PLOT_COLORS["scr"], zorder=5)
    
    shade_bad_zones(ax)
    ax.set_ylabel("Phasic (µS)", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


# =============================================================================
# BAD INTERVAL MARKING
# =============================================================================

def marcar_bad_interval(seg_dict: dict, t_ini: float, t_fin: float,
                        motivo: str = "bad", sev: float = 1.0) -> dict:
    """Add a BAD interval to seg_dict."""
    bi = seg_dict.get("bad_intervals", None)
    new_row = {
        "t_ini_s": float(min(t_ini, t_fin)),
        "t_fin_s": float(max(t_ini, t_fin)),
        "motivo": str(motivo),
        "sev": float(sev)
    }
    if bi is None or not isinstance(bi, pd.DataFrame) or len(bi) == 0:
        bi = pd.DataFrame([new_row])
    else:
        bi = pd.concat([bi, pd.DataFrame([new_row])], ignore_index=True)
    seg_dict["bad_intervals"] = bi
    return seg_dict


def reset_bad_intervals(seg_dict: dict) -> dict:
    """Clear all BAD intervals."""
    seg_dict["bad_intervals"] = pd.DataFrame(columns=["t_ini_s", "t_fin_s", "motivo", "sev"])
    return seg_dict


def marcar_enlarged_windows_como_bad(
    seg_dict: dict,
    indices: Optional[List[int]] = None,
    verbose: bool = True,
) -> dict:
    """
    Mark enlarged windows (detected during inspection) as bad zones.
    
    Enlarged windows are regions detected with:
    - High |dEDA/dt| (exceeding threshold)
    - High phasic amplitude (0.05-0.1 µS range)
    - Closely spaced SCRs (< 2s apart)
    
    These are shown during inspection for visual review. If you decide they are
    problematic, use this function to mark them as bad zones.
    
    Args:
        seg_dict: Segment dictionary with 'enlarged_windows' list
        indices: Optional list of indices to mark (0-based). If None, marks all enlarged windows.
                Example: indices=[0, 2] marks the 1st and 3rd enlarged windows.
        verbose: Print information about marked windows
    
    Returns:
        Updated seg_dict with enlarged windows added to bad_intervals
    
    Example:
        # Mark all enlarged windows as bad
        resultado = marcar_enlarged_windows_como_bad(resultado)
        
        # Mark only specific enlarged windows (e.g., first and third)
        resultado = marcar_enlarged_windows_como_bad(resultado, indices=[0, 2])
    """
    seg_name = seg_dict.get("seg", "SEG")
    enlarged_windows = seg_dict.get("enlarged_windows", [])
    
    if len(enlarged_windows) == 0:
        if verbose:
            print(f"[Mark Enlarged Windows] {seg_name}: No enlarged windows detected")
        return seg_dict
    
    # Determine which windows to mark
    if indices is None:
        # Mark all
        windows_to_mark = enlarged_windows
        if verbose:
            print(f"[Mark Enlarged Windows] {seg_name}: Marking all {len(windows_to_mark)} enlarged window(s) as bad")
    else:
        # Mark only specified indices
        windows_to_mark = []
        for idx in indices:
            if 0 <= idx < len(enlarged_windows):
                windows_to_mark.append(enlarged_windows[idx])
            else:
                if verbose:
                    print(f"[Mark Enlarged Windows] {seg_name}: Warning: index {idx} out of range (0-{len(enlarged_windows)-1})")
        if verbose and len(windows_to_mark) > 0:
            print(f"[Mark Enlarged Windows] {seg_name}: Marking {len(windows_to_mark)} enlarged window(s) as bad (indices: {indices})")
    
    # Mark each window as bad
    for start_s, end_s, reason in windows_to_mark:
        # Create descriptive motivo that includes the original reason
        motivo = f"enlarged_window_{reason}"
        seg_dict = marcar_bad_interval(seg_dict, start_s, end_s, motivo=motivo)
        if verbose:
            print(f"  → Marked [{start_s:.1f}s - {end_s:.1f}s] as bad (reason: {reason})")
    
    # Recalculate bad_pct after marking
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    
    if verbose and len(windows_to_mark) > 0:
        print(f"  Updated bad_pct: {bad_pct*100:.1f}%")
    
    return seg_dict


def calcular_bad_pct(seg_dict: dict) -> Tuple[float, float, float]:
    """Calculate percentage of segment marked as BAD."""
    fs = float(seg_dict.get("fs_eda", np.nan))
    n = len(seg_dict.get("eda_raw", []))
    if n == 0 or not np.isfinite(fs) or fs <= 0:
        return np.nan, np.nan, np.nan
    dur_s = n / fs
    
    bi = seg_dict.get("bad_intervals", None)
    if bi is None or len(bi) == 0:
        return 0.0, 0.0, dur_s
    
    intervals = []
    for _, row in bi.iterrows():
        s = float(row["t_ini_s"])
        e = float(row["t_fin_s"])
        s, e = min(s, e), max(s, e)
        s = max(0.0, min(s, dur_s))
        e = max(0.0, min(e, dur_s))
        if e > s:
            intervals.append((s, e))
    
    if not intervals:
        return 0.0, 0.0, dur_s
    
    # Merge overlapping intervals
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    
    bad_s = sum(e - s for s, e in merged)
    bad_pct = bad_s / dur_s if dur_s > 0 else np.nan
    return bad_pct, bad_s, dur_s


# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================

def validar_calidad_eda_extendida(
    seg_dict: dict,
    scl_min_plausible: float = 0.5,
    scl_max_plausible: float = 30.0,
    min_clean_duration_s: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, bool]:
    """
    Extended quality validation for EDA segments.
    
    Per literature guidelines (Boucsein 2012, Braithwaite 2013):
    - SCL should be in physiologically plausible range (0.5-20 µS typical)
    - Minimum clean duration may be required for stable baseline estimates
    - Multiple QC flags for transparency
    
    Args:
        seg_dict: Segment dictionary
        scl_min_plausible: Minimum plausible SCL (µS)
        scl_max_plausible: Maximum plausible SCL (µS)
        min_clean_duration_s: Minimum required clean duration (s)
        verbose: Print detailed QC info
    
    Returns:
        Dictionary of QC flags: {
            'bad_pct_ok': bool,
            'scl_plausible': bool,
            'sufficient_clean': bool,
            'overall_valid': bool
        }
    """
    seg_name = seg_dict.get("seg", "SEG")
    bad_pct = seg_dict.get("bad_pct", 0.0)
    scl_mean = seg_dict.get("scl_mean", np.nan)
    bad_s = seg_dict.get("bad_s", 0.0)
    dur_s = seg_dict.get("dur_s", 0.0)
    clean_s = dur_s - bad_s if np.isfinite(bad_s) else dur_s
    
    # QC checks
    qc_flags = {}
    qc_flags['bad_pct_ok'] = bool(np.isfinite(bad_pct) and bad_pct <= BAD_PROP_THRESHOLD)
    qc_flags['scl_plausible'] = bool(
        np.isfinite(scl_mean) and scl_min_plausible <= scl_mean <= scl_max_plausible
    )
    
    if min_clean_duration_s is not None:
        qc_flags['sufficient_clean'] = bool(clean_s >= min_clean_duration_s)
    else:
        qc_flags['sufficient_clean'] = True
    
    # Overall validity requires all checks to pass
    qc_flags['overall_valid'] = all([
        qc_flags['bad_pct_ok'],
        qc_flags['scl_plausible'],
        qc_flags['sufficient_clean']
    ])
    
    if verbose:
        print(f"\n[Extended QC] {seg_name}:")
        print(f"  BAD percentage: {bad_pct*100:.1f}% {'✓' if qc_flags['bad_pct_ok'] else '✗'}")
        print(f"  SCL plausibility: {scl_mean:.3f} µS ({'✓' if qc_flags['scl_plausible'] else '✗ OUT OF RANGE'})")
        if min_clean_duration_s:
            print(f"  Clean duration: {clean_s:.1f}s {'✓' if qc_flags['sufficient_clean'] else '✗'}")
        print(f"  Overall: {'✓ VALID' if qc_flags['overall_valid'] else '✗ EXCLUDED'}")
    
    return qc_flags


def limpiar_segmento_etapa2(
    seg_dict: dict,
    bad_pct_max: float = BAD_PROP_THRESHOLD,
    min_clean_duration_s: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[dict, bool]:
    """
    Stage 2: Validate segment quality.
    
    Args:
        bad_pct_max: Maximum allowed BAD percentage
        min_clean_duration_s: Minimum required CLEAN duration
    
    Returns: (seg_dict, qc_valid)
    """
    seg_name = seg_dict.get("seg", "SEG")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EDA QC VALIDATION: {seg_name}")
        print(f"{'='*60}")
    
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    clean_s = dur_s - bad_s if np.isfinite(bad_s) else dur_s
    
    # Check QC criteria
    qc_pct_ok = bool(np.isfinite(bad_pct) and (bad_pct <= bad_pct_max))
    qc_dur_ok = True
    
    if min_clean_duration_s is not None:
        qc_dur_ok = clean_s >= min_clean_duration_s
    
    qc_valid = qc_pct_ok and qc_dur_ok
    
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    seg_dict["clean_s"] = clean_s
    seg_dict["qc_valid"] = qc_valid
    
    # Run extended QC validation and store flags
    qc_flags_extended = validar_calidad_eda_extendida(
        seg_dict,
        min_clean_duration_s=min_clean_duration_s,
        verbose=verbose
    )
    seg_dict["qc_flags"] = qc_flags_extended
    
    # Use extended validation result
    qc_valid = qc_flags_extended['overall_valid']
    seg_dict["qc_valid"] = qc_valid
    
    if verbose:
        print(f"  Total duration: {dur_s:.1f}s")
        print(f"  BAD: {bad_s:.1f}s ({bad_pct*100:.1f}%) | threshold: {bad_pct_max*100:.0f}%")
        print(f"  CLEAN: {clean_s:.1f}s", end="")
        if min_clean_duration_s is not None:
            print(f" | minimum required: {min_clean_duration_s:.0f}s", end="")
        print()
        
        if not qc_pct_ok:
            print(f"  ✘ EXCLUDED: BAD > {bad_pct_max*100:.0f}%")
        elif not qc_dur_ok:
            print(f"  ✘ EXCLUDED: CLEAN ({clean_s:.1f}s) < minimum ({min_clean_duration_s:.0f}s)")
        else:
            print(f"  ✔ QC VALID")
    
    return seg_dict, qc_valid


# =============================================================================
# OUTLIER DETECTION HELPERS (EDA)
# =============================================================================

def detectar_outliers_eda(
    seg_dict: dict,
    scl_min_plausible: float = 0.5,
    scl_max_plausible: float = 30.0,
    scl_std_limit: float = 2.5,
    bad_pct_max: float = BAD_PROP_THRESHOLD,
    max_scr_rate_per_min: float = 15.0,
) -> Tuple[dict, List[Tuple[str, float]]]:
    qc_flags = validar_calidad_eda_extendida(
        seg_dict,
        scl_min_plausible=scl_min_plausible,
        scl_max_plausible=scl_max_plausible,
        verbose=False,
    )
    seg_dict["qc_flags_extended"] = qc_flags
    outliers: List[Tuple[str, float]] = []
    scl_mean = seg_dict.get("scl_mean", np.nan)
    if not qc_flags.get("scl_plausible", False):
        outliers.append(("scl_mean", float(scl_mean) if np.isfinite(scl_mean) else np.nan))
    bad_pct = seg_dict.get("bad_pct", 0.0)
    if not qc_flags.get("bad_pct_ok", False) or bad_pct > bad_pct_max:
        outliers.append(("bad_pct", float(bad_pct)))
    scl_std = seg_dict.get("scl_std", np.nan)
    if np.isfinite(scl_std) and scl_std > scl_std_limit:
        outliers.append(("scl_std", float(scl_std)))
    dur_s = seg_dict.get("dur_s", 0.0)
    n_scrs = len(seg_dict.get("scr_peaks", []))
    if dur_s > 0:
        scr_rate = n_scrs / (dur_s / 60.0)
        if scr_rate > max_scr_rate_per_min:
            outliers.append(("scr_rate_per_min", float(scr_rate)))

    seg_dict["outliers_eda"] = outliers
    return seg_dict, outliers


def listar_outliers_eda(outliers: List[Tuple[str, float]]) -> pd.DataFrame:
    rows = []
    for category, value in outliers:
        rows.append({
            "category": category,
            "value": value,
        })
    return pd.DataFrame(rows)


def verificar_outliers_todos(
    resultados: dict,
    scl_min: float = 0.5,
    scl_max: float = 30.0,
    scl_std_limit: float = 2.5,
    max_scr_rate_per_min: float = 15.0,
    bad_pct_max: float = BAD_PROP_THRESHOLD,
    verbose: bool = True,
) -> Tuple[dict, bool]:
    if verbose:
        print("\n" + "=" * 70)
        print("VERIFICACIÓN DE OUTLIERS EDA")
        print("=" * 70)

    hay_problemas = False
    problemas = []

    for seg_name, seg_dict in resultados.items():
        seg_dict, outliers = detectar_outliers_eda(
            seg_dict,
            scl_min_plausible=scl_min,
            scl_max_plausible=scl_max,
            scl_std_limit=scl_std_limit,
            bad_pct_max=bad_pct_max,
            max_scr_rate_per_min=max_scr_rate_per_min,
        )
        resultados[seg_name] = seg_dict

        if len(outliers) == 0:
            if verbose:
                print(f"{seg_name}: ✔ OK")
        else:
            hay_problemas = True
            problemas.append(seg_name)
            if verbose:
                print(f"\n{seg_name}: ⚠️  {len(outliers)} outlier issue(s)")
                df_out = listar_outliers_eda(outliers)
                print(df_out.to_string(index=False))
                print("  → Consider re-inspecting and marking bad zones for this segment.")

    if verbose:
        print("\n" + "-" * 70)
        if hay_problemas:
            print(f"⚠️  Segments requiring review: {', '.join(problemas)}")
            print("Review the flagged segments before exporting.")
        else:
            print("✔ All segments OK - ready for export")
        print("-" * 70)

    return resultados, hay_problemas


# =============================================================================
# RE-INSPECTION
# =============================================================================

def consolidar_bad_intervals(seg_dict: dict) -> pd.DataFrame:
    """Consolidate BAD intervals."""
    bi = seg_dict.get("bad_intervals", pd.DataFrame())
    if bi is None or len(bi) == 0:
        return pd.DataFrame(columns=["id", "t_ini_s", "t_fin_s", "severity", "motivos", "dur_s"])
    
    dur_s = len(seg_dict.get("eda_raw", [])) / float(seg_dict.get("fs_eda", 32))
    
    intervals = []
    for _, row in bi.iterrows():
        s = float(row["t_ini_s"])
        e = float(row["t_fin_s"])
        s, e = min(s, e), max(s, e)
        s = max(0.0, min(s, dur_s))
        e = max(0.0, min(e, dur_s))
        if e > s:
            intervals.append((s, e, str(row.get("motivo", "bad"))))
    
    if not intervals:
        return pd.DataFrame(columns=["id", "t_ini_s", "t_fin_s", "severity", "motivos", "dur_s"])
    
    intervals = sorted(intervals, key=lambda x: x[0])
    
    # Merge overlapping
    merged = []
    cur_start, cur_end, cur_motivos = intervals[0][0], intervals[0][1], [intervals[0][2]]
    cur_severity = 1
    
    for s, e, m in intervals[1:]:
        if s < cur_end:
            cur_end = max(cur_end, e)
            cur_severity += 1
            if m not in cur_motivos:
                cur_motivos.append(m)
        else:
            merged.append((cur_start, cur_end, cur_severity, cur_motivos))
            cur_start, cur_end, cur_motivos = s, e, [m]
            cur_severity = 1
    
    merged.append((cur_start, cur_end, cur_severity, cur_motivos))
    
    rows = []
    for i, (s, e, sev, motivos) in enumerate(merged):
        rows.append({
            "id": i + 1,
            "t_ini_s": s,
            "t_fin_s": e,
            "severity": sev,
            "motivos": "+".join(motivos),
            "dur_s": e - s
        })
    
    return pd.DataFrame(rows)


def reinspeccionar_segmento(
    seg_dict: dict,
    win_len: Optional[float] = None,  # None = auto (full segment if <6min, else 180s)
    min_clean_duration_s: Optional[float] = None,
) -> pd.DataFrame:
    """Re-inspect segment after cleaning."""
    if plt is None:
        print("matplotlib not available.")
        return pd.DataFrame()
    
    seg_name = seg_dict.get("seg", "SEG")
    eda_raw = seg_dict.get("eda_raw", np.array([]))
    eda_clean = seg_dict.get("eda_clean", eda_raw)
    # Use data with NaN markers if available (per literature: mark bad zones as NaN, don't interpolate)
    eda_with_nan = seg_dict.get("eda_with_nan", eda_clean)
    tonic = seg_dict.get("tonic_with_nan", seg_dict.get("eda_tonic", np.zeros_like(eda_raw)))
    phasic = seg_dict.get("eda_phasic", np.zeros_like(eda_raw))
    times = seg_dict.get("times", np.array([]))
    scr_peaks = seg_dict.get("scr_peaks", np.array([]))
    artifacts = seg_dict.get("artifacts", [])
    fs = float(seg_dict.get("fs_eda", 32.0))
    bad_zones_info = seg_dict.get("bad_zones_info", {})
    is_bad = bad_zones_info.get("is_bad", np.zeros(len(times), dtype=bool))
    
    if len(times) == 0:
        print("No data to inspect.")
        return pd.DataFrame()
    
    bad_consolidated = consolidar_bad_intervals(seg_dict)
    
    bad_pct, bad_s, total_s = calcular_bad_pct(seg_dict)
    clean_s = total_s - bad_s if np.isfinite(bad_s) else total_s
    
    # Determine status
    if min_clean_duration_s is not None:
        segment_ok = clean_s >= min_clean_duration_s
        if segment_ok:
            status_str = f"  →  ✔ OK (clean >= {min_clean_duration_s:.0f}s)"
        else:
            status_str = f"  →  ❌ EXCLUDED (need {min_clean_duration_s:.0f}s clean)"
    else:
        segment_ok = bad_pct <= 0.20
        status_str = "  →  ✔ OK" if segment_ok else "  →  ❌ EXCLUDED (BAD > 20%)"
    
    print(f"\n{'='*70}")
    print(f"EDA RE-INSPECTION: {seg_name}")
    print(f"{'='*70}")
    print(f"Duration: {total_s:.1f}s | BAD: {bad_s:.1f}s ({bad_pct*100:.1f}%) | CLEAN: {clean_s:.1f}s{status_str}")
    print(f"SCRs: {len(scr_peaks)} | SCL mean: {seg_dict.get('scl_mean', 0):.3f} µS")
    
    if len(bad_consolidated) > 0:
        print(f"\n--- BAD ZONES ---")
        for _, row in bad_consolidated.iterrows():
            print(f"  [{row['id']}] {row['t_ini_s']:.1f}s - {row['t_fin_s']:.1f}s ({row['dur_s']:.1f}s) | {row['motivos']}")
    else:
        print("\n✔ No BAD zones")
    
    # Auto-adjust window length
    dur_seg = float(times[-1]) if len(times) else 0.0
    if win_len is None:
        if dur_seg < 360:  # < 6 minutes: show full segment
            win_len = dur_seg
        else:  # >= 6 minutes: use 180s windows
            win_len = 180.0
    
    n_windows = int(math.ceil(dur_seg / win_len)) if dur_seg > 0 else 0
    
    for k in range(n_windows):
        t0 = k * win_len
        win = min(win_len, dur_seg - t0)
        if win <= 0:
            break
        print(f"\nWindow {k+1}/{n_windows}: [{t0:.1f}s - {t0+win:.1f}s]")
        plot_eda_window_with_bad(
            eda_raw=eda_raw,
            eda_clean=eda_with_nan,  # Use data with NaN markers (bad zones marked as NaN, not interpolated)
            tonic=tonic,
            phasic=phasic,
            times=times,
            scr_peaks=scr_peaks,
            bad_intervals_df=bad_consolidated,
            is_interpolated=is_bad,  # Now represents bad zones marked as NaN
            t0=t0,
            win_len=win,
            seg_name=seg_name,
            fs=fs,
        )
    
    return bad_consolidated


# =============================================================================
# INTERPOLATION (for BAD zones)
# =============================================================================

def marcar_bad_zones_como_nan(
    seg_dict: dict,
    verbose: bool = True,
) -> dict:
    """
    Mark BAD zones as NaN (following Boucsein 2012 and Li et al. 2022 guidelines).
    
    Per literature recommendations:
    - Do NOT interpolate artifacts (breaks temporal continuity for cvxEDA)
    - Mark bad zones as NaN (Not a Number) for exclusion
    - SCL mean will be calculated only from clean data (using np.nanmean)
    - Report % of data rejected (bad_pct) for transparency
    
    This approach maintains temporal continuity for cvxEDA while properly
    excluding artifact zones from analysis.
    
    Args:
        verbose: Print information about marked zones
    
    Returns: seg_dict with 'eda_with_nan' and 'tonic_with_nan' (NaN in bad zones)
    """
    seg_name = seg_dict.get("seg", "SEG")
    eda_clean = seg_dict.get("eda_clean", np.array([]))
    tonic = seg_dict.get("eda_tonic", np.array([]))
    times = seg_dict.get("times", np.array([]))
    bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
    
    if len(eda_clean) == 0:
        return seg_dict
    
    # Create mask for BAD zones
    is_bad = np.zeros(len(times), dtype=bool)
    
    if bad_intervals is not None and len(bad_intervals) > 0:
        for _, row in bad_intervals.iterrows():
            t_ini = float(row["t_ini_s"])
            t_fin = float(row["t_fin_s"])
            mask = (times >= t_ini) & (times <= t_fin)
            is_bad |= mask
    
    n_bad = int(np.sum(is_bad))
    n_good = len(times) - n_bad
    
    if n_bad == 0:
        if verbose:
            print(f"[Mark BAD as NaN] {seg_name}: No bad zones to mark")
        # Still create the arrays for consistency
        seg_dict["eda_with_nan"] = eda_clean.copy()
        seg_dict["tonic_with_nan"] = tonic.copy()
        seg_dict["bad_zones_info"] = {
            "n_bad": 0,
            "n_total": len(times),
            "is_bad": is_bad,
        }
        # Ensure bad_pct is set to 0.0
        seg_dict["bad_pct"] = 0.0
        seg_dict["bad_s"] = 0.0
        return seg_dict
    
    # Mark bad zones as NaN (do NOT interpolate)
    eda_with_nan = eda_clean.copy().astype(float)
    tonic_with_nan = tonic.copy().astype(float)
    
    eda_with_nan[is_bad] = np.nan
    tonic_with_nan[is_bad] = np.nan
    
    seg_dict["eda_with_nan"] = eda_with_nan
    seg_dict["tonic_with_nan"] = tonic_with_nan
    seg_dict["bad_zones_info"] = {
        "n_bad": n_bad,
        "n_total": len(times),
        "is_bad": is_bad,
    }
    
    # Recalculate SCL mean using only clean data (NaN will be ignored)
    scl_mean_clean = float(np.nanmean(tonic_with_nan))
    scl_std_clean = float(np.nanstd(tonic_with_nan))
    seg_dict["scl_mean"] = scl_mean_clean
    seg_dict["scl_std"] = scl_std_clean
    
    # Update bad_pct and bad_s in dictionary for summary reporting
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct if np.isfinite(bad_pct) else 0.0
    seg_dict["bad_s"] = bad_s if np.isfinite(bad_s) else 0.0
    
    if verbose:
        pct = n_bad / len(times) * 100 if len(times) > 0 else 0
        print(f"\n[Mark BAD as NaN] {seg_name}: {n_bad}/{len(times)} values marked as NaN ({pct:.1f}% rejected)")
        print(f"  SCL mean (clean data only): {scl_mean_clean:.3f} µS")
        print(f"  Note: These zones are excluded from analysis, not interpolated")
    
    return seg_dict


def trim_segment_eda(
    seg_dict: dict,
    target_duration_s: float = 300.0,
    trim_from: str = "start",  # "start" or "end"
    verbose: bool = True,
) -> dict:
    """
    Trim segment to target duration by removing from start or end.
    
    Args:
        seg_dict: Segment dictionary
        target_duration_s: Target duration in seconds
        trim_from: "start" to remove from beginning, "end" to remove from end
        verbose: Print info
    
    Returns: Updated seg_dict with trimmed data
    """
    seg_name = seg_dict.get("seg", "SEG")
    times = seg_dict.get("times", np.array([]))
    fs = float(seg_dict.get("fs_eda", 32.0))
    
    if len(times) == 0:
        return seg_dict
    
    current_dur = float(times[-1]) if len(times) > 0 else 0.0
    trim_dur = current_dur - target_duration_s
    
    if trim_dur <= 0:
        if verbose:
            print(f"[Trim] {seg_name}: Already ≤ {target_duration_s:.1f}s ({current_dur:.1f}s), no trimming needed")
        return seg_dict
    
    if trim_from == "start":
        t_start_new = trim_dur
        t_end_new = current_dur
        idx_start = np.searchsorted(times, t_start_new)
        idx_end = len(times)
        trim_info = f"first {trim_dur:.1f}s"
    else:  # trim_from == "end"
        t_start_new = 0.0
        t_end_new = target_duration_s
        idx_start = 0
        idx_end = np.searchsorted(times, t_end_new)
        trim_info = f"last {trim_dur:.1f}s"
    
    # Trim all arrays
    seg_dict["times"] = times[idx_start:idx_end] - times[idx_start]  # Reset time to start at 0
    seg_dict["eda_raw"] = seg_dict.get("eda_raw", np.array([]))[idx_start:idx_end]
    seg_dict["eda_clean"] = seg_dict.get("eda_clean", np.array([]))[idx_start:idx_end]
    seg_dict["eda_tonic"] = seg_dict.get("eda_tonic", np.array([]))[idx_start:idx_end]
    seg_dict["eda_phasic"] = seg_dict.get("eda_phasic", np.array([]))[idx_start:idx_end]
    
    # Trim data with NaN markers if present
    if "eda_with_nan" in seg_dict:
        seg_dict["eda_with_nan"] = seg_dict["eda_with_nan"][idx_start:idx_end]
    if "tonic_with_nan" in seg_dict:
        seg_dict["tonic_with_nan"] = seg_dict["tonic_with_nan"][idx_start:idx_end]
    # Also update bad_zones_info mask
    if "bad_zones_info" in seg_dict:
        bad_info = seg_dict["bad_zones_info"]
        if "is_bad" in bad_info:
            bad_info["is_bad"] = bad_info["is_bad"][idx_start:idx_end]
            bad_info["n_bad"] = int(np.sum(bad_info["is_bad"]))
            bad_info["n_total"] = len(bad_info["is_bad"])
    
    # Update SCR peaks (adjust indices and filter)
    scr_peaks = seg_dict.get("scr_peaks", np.array([]))
    scr_onsets = seg_dict.get("scr_onsets", np.array([]))
    if len(scr_peaks) > 0:
        valid_peaks = (scr_peaks >= idx_start) & (scr_peaks < idx_end)
        seg_dict["scr_peaks"] = scr_peaks[valid_peaks] - idx_start
        if len(scr_onsets) > 0:
            valid_onsets = (scr_onsets >= idx_start) & (scr_onsets < idx_end)
            seg_dict["scr_onsets"] = scr_onsets[valid_onsets] - idx_start
            # Match amplitudes and rise_times to valid peaks
            scr_amplitudes = seg_dict.get("scr_amplitudes", np.array([]))
            scr_rise_times = seg_dict.get("scr_rise_times", np.array([]))
            if len(scr_amplitudes) == len(scr_peaks):
                seg_dict["scr_amplitudes"] = scr_amplitudes[valid_peaks]
            if len(scr_rise_times) == len(scr_peaks):
                seg_dict["scr_rise_times"] = scr_rise_times[valid_peaks]
    
    # Update bad intervals (shift times and filter)
    bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
    if bad_intervals is not None and len(bad_intervals) > 0:
        bad_intervals = bad_intervals.copy()
        bad_intervals["t_ini_s"] = bad_intervals["t_ini_s"] - t_start_new
        bad_intervals["t_fin_s"] = bad_intervals["t_fin_s"] - t_start_new
        # Filter intervals that are completely outside the new range
        bad_intervals = bad_intervals[
            (bad_intervals["t_fin_s"] > 0) & (bad_intervals["t_ini_s"] < target_duration_s)
        ].copy()
        # Clip to new range
        bad_intervals["t_ini_s"] = bad_intervals["t_ini_s"].clip(lower=0.0, upper=target_duration_s)
        bad_intervals["t_fin_s"] = bad_intervals["t_fin_s"].clip(lower=0.0, upper=target_duration_s)
        # Remove zero-duration intervals
        bad_intervals = bad_intervals[bad_intervals["t_fin_s"] > bad_intervals["t_ini_s"]].copy()
        seg_dict["bad_intervals"] = bad_intervals
    
    # Update duration and indices
    seg_dict["dur_s"] = float(target_duration_s)
    seg_dict["start_idx"] = int(seg_dict.get("start_idx", 0) + idx_start)
    seg_dict["end_idx"] = int(seg_dict.get("start_idx", 0) + idx_end)
    
    # Recalculate statistics
    tonic_new = seg_dict["eda_tonic"]
    seg_dict["scl_mean"] = float(np.nanmean(tonic_new))
    seg_dict["scl_std"] = float(np.nanstd(tonic_new))
    
    if verbose:
        print(f"[Trim] {seg_name}: Removed {trim_info} → {current_dur:.1f}s → {target_duration_s:.1f}s")
    
    return seg_dict


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def exportar_segmento_eda_excel(seg_dict: dict, output_path: str, verbose: bool = True) -> bool:
    """Export EDA segment data to Excel."""
    seg_name = seg_dict.get("seg", "SEG")
    times = seg_dict.get("times", np.array([]))
    
    if len(times) == 0:
        if verbose:
            print(f"[Export] {seg_name}: No data")
        return False
    
    # Use data with NaN markers if available (per literature: mark bad zones as NaN, don't interpolate)
    # Fallback to clean data if bad zones haven't been marked yet
    eda = seg_dict.get("eda_with_nan", seg_dict.get("eda_clean", np.array([])))
    tonic = seg_dict.get("tonic_with_nan", seg_dict.get("eda_tonic", np.array([])))
    phasic = seg_dict.get("eda_phasic", np.array([]))
    
    bad_zones_info = seg_dict.get("bad_zones_info", {})
    is_bad = bad_zones_info.get("is_bad", np.zeros(len(times), dtype=bool))
    
    # Create SCR peak indicator (binary: 1 at peak locations, 0 elsewhere)
    scr_peaks = seg_dict.get("scr_peaks", np.array([]))
    scr_peak_indicator = np.zeros(len(times), dtype=int)
    if len(scr_peaks) > 0:
        valid_peaks = scr_peaks[scr_peaks < len(times)]
        scr_peak_indicator[valid_peaks] = 1
    
    df_export = pd.DataFrame({
        "time_s": times,
        "eda_us": eda if len(eda) == len(times) else np.full(len(times), np.nan),
        "tonic_us": tonic if len(tonic) == len(times) else np.full(len(times), np.nan),
        "phasic_us": phasic if len(phasic) == len(times) else np.full(len(times), np.nan),
        "scr_peak": scr_peak_indicator,  # Binary indicator: 1 = SCR peak, 0 = no peak
        "is_bad": is_bad.astype(int) if len(is_bad) == len(times) else np.zeros(len(times), dtype=int),
    })
    
    # Add segment info
    df_export["segment"] = seg_name
    df_export["fs_eda"] = seg_dict.get("fs_eda", 32)
    df_export["duration_s"] = seg_dict.get("dur_s", 0)
    df_export["bad_pct"] = seg_dict.get("bad_pct", 0)
    
    df_export.to_excel(output_path, index=False, sheet_name=seg_name[:31])
    
    if verbose:
        print(f"[Export] {seg_name}: Saved to {output_path}")
    
    return True


def exportar_sujeto_completo_eda(
    resultados: dict,
    sujeto_id: str,
    output_folder: str,
    verbose: bool = True,
) -> str:
    """Export all EDA segments for one subject to Excel."""
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"EDA_data_sujeto_{sujeto_id}.xlsx")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_rows = []
        
        for seg_name, seg_dict in resultados.items():
            times = seg_dict.get("times", np.array([]))
            
            if len(times) == 0:
                if verbose:
                    print(f"[Export] {seg_name}: Skipped (no data)")
                continue
            
            # Use data with NaN markers if available (per literature: mark bad zones as NaN, don't interpolate)
            # Fallback to clean data if bad zones haven't been marked yet
            eda = seg_dict.get("eda_with_nan", seg_dict.get("eda_clean", np.array([])))
            tonic = seg_dict.get("tonic_with_nan", seg_dict.get("eda_tonic", np.array([])))
            phasic = seg_dict.get("eda_phasic", np.array([]))
            
            bad_zones_info = seg_dict.get("bad_zones_info", {})
            is_bad = bad_zones_info.get("is_bad", np.zeros(len(times), dtype=bool))
            
            # Create SCR peak indicator (binary: 1 at peak locations, 0 elsewhere)
            scr_peaks = seg_dict.get("scr_peaks", np.array([]))
            scr_peak_indicator = np.zeros(len(times), dtype=int)
            if len(scr_peaks) > 0:
                valid_peaks = scr_peaks[scr_peaks < len(times)]
                scr_peak_indicator[valid_peaks] = 1
            
            df_seg = pd.DataFrame({
                "time_s": times,
                "eda_us": eda if len(eda) == len(times) else np.full(len(times), np.nan),
                "tonic_us": tonic if len(tonic) == len(times) else np.full(len(times), np.nan),
                "phasic_us": phasic if len(phasic) == len(times) else np.full(len(times), np.nan),
                "scr_peak": scr_peak_indicator,  # Binary indicator: 1 = SCR peak, 0 = no peak
                "is_bad": is_bad.astype(int) if len(is_bad) == len(times) else np.zeros(len(times), dtype=int),
            })
            
            sheet_name = seg_name[:31].replace("/", "_").replace("\\", "_")
            df_seg.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Summary row
            enlarged_windows = seg_dict.get("enlarged_windows", [])
            qc_flags = seg_dict.get("qc_flags", {})
            summary_rows.append({
                "segment": seg_name,
                "duration_s": seg_dict.get("dur_s", 0),
                "bad_pct": seg_dict.get("bad_pct", 0) * 100,
                "n_rejected": bad_zones_info.get("n_bad", 0),
                "n_enlarged_windows": len(enlarged_windows),  # Number of regions detected for detailed inspection
                "qc_valid": seg_dict.get("qc_valid", True),
                "qc_bad_pct_ok": qc_flags.get("bad_pct_ok", True),
                "qc_scl_plausible": qc_flags.get("scl_plausible", True),
                "qc_sufficient_clean": qc_flags.get("sufficient_clean", True),
                "scl_mean": seg_dict.get("scl_mean", 0),
                "scl_std": seg_dict.get("scl_std", 0),
                "n_scrs": len(seg_dict.get("scr_peaks", [])),
            })
        
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, sheet_name="RESUMEN", index=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EXPORTED: {output_path}")
        print(f"Segments: {len(summary_rows)}")
        print(f"{'='*60}")
    
    return output_path


def desguardar_segmento(seg_name: str, resultados: dict) -> None:
    """
    Quita un segmento de resultados para que no se exporte (p. ej. señal mala).
    Ejecutar después de haber guardado el segmento si decides no incluirlo.
    """
    if seg_name in resultados:
        del resultados[seg_name]
        print(f"  → Segmento '{seg_name}' quitado (no se exportará).")
    else:
        print(f"  → '{seg_name}' no estaba en resultados.")


def exportar_y_resumir_eda(
    resultados: dict,
    sujeto_id: str,
    output_folder: str,
    archivo_resumen: str,
) -> str:
    """Export to Excel, show summary, and save cumulative summary."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Export
    output_file = exportar_sujeto_completo_eda(
        resultados=resultados,
        sujeto_id=sujeto_id,
        output_folder=output_folder,
        verbose=True
    )
    
    # Print summary
    print("\n" + "="*70)
    print(f"FINAL SUMMARY - SUBJECT {sujeto_id}")
    print("="*70)
    print(f"\n{'Segment':<15} {'Duration':>10} {'BAD%':>8} {'SCL mean':>10} {'SCRs':>8}")
    print("-"*70)
    
    resumen_filas = []
    for seg_name, seg_dict in resultados.items():
        dur = seg_dict.get("dur_s", 0)
        bad_pct = seg_dict.get("bad_pct", 0) * 100
        scl_mean = seg_dict.get("scl_mean", 0)
        n_scrs = len(seg_dict.get("scr_peaks", []))
        
        print(f"{seg_name:<15} {dur:>10.1f}s {bad_pct:>7.1f}% {scl_mean:>9.3f} {n_scrs:>8}")
        
        resumen_filas.append({
            "sujeto": sujeto_id, "segmento": seg_name,
            "duracion_s": round(dur, 1),
            "bad_pct": round(bad_pct, 2),
            "scl_mean": round(scl_mean, 4),
            "scl_std": round(seg_dict.get("scl_std", 0), 4),
            "n_scrs": n_scrs,
        })
    
    print("-"*70)
    print(f"Total segments: {len(resultados)}")
    
    # Save cumulative summary
    fila_sujeto = {"sujeto": sujeto_id}
    for seg_info in resumen_filas:
        seg = seg_info["segmento"].replace(" ", "")
        fila_sujeto[f"{seg}_duracion_s"] = seg_info["duracion_s"]
        fila_sujeto[f"{seg}_bad_pct"] = seg_info["bad_pct"]
        fila_sujeto[f"{seg}_scl_mean"] = seg_info["scl_mean"]
        fila_sujeto[f"{seg}_scl_std"] = seg_info["scl_std"]
        fila_sujeto[f"{seg}_n_scrs"] = seg_info["n_scrs"]
    
    df_nuevo = pd.DataFrame([fila_sujeto])
    if os.path.exists(archivo_resumen):
        df_existente = pd.read_csv(archivo_resumen)
        df_existente = df_existente[df_existente["sujeto"].astype(str) != str(sujeto_id)]
        df_final = pd.concat([df_existente, df_nuevo], ignore_index=True).fillna("")
    else:
        df_final = df_nuevo
    
    try:
        df_final = df_final.sort_values("sujeto", key=lambda x: pd.to_numeric(x, errors='coerce'))
    except:
        df_final = df_final.sort_values("sujeto")
    
    df_final.to_csv(archivo_resumen, index=False)
    print(f"\n✔ Excel: {output_file}")
    print(f"✔ Cumulative summary: {archivo_resumen}")
    
    return output_file


# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
