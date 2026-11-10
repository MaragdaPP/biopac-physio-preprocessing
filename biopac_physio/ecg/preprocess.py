# Part of biopac_physio — https://github.com/MaragdaPP/biopac-physio-preprocessing
# Library only: the per-subject workflow lives in studies/<study>/ecg_clean.py
# %% Imports & configuration
"""
Standalone ECG preprocessing helper for BIOPAC .acq files.
Run cell-by-cell in VSCode/Cursor.
Required: neurokit2, bioread, physio, matplotlib, pandas, numpy
Note: physio is from physiopy package (pip install physio), NOT systole.
"""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

import numpy as np
import pandas as pd
import neurokit2 as nk

try:  # Optional plotting
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional
    plt = None

try:  # BIOPAC reader
    import bioread
except ImportError:  # pragma: no cover - handled in loader
    bioread = None

# physio package for ECG processing (from physiopy, NOT systole)
try:
    import physio
    _HAVE_PHYSIO = True
except ImportError:
    physio = None
    _HAVE_PHYSIO = False

# scipy for interpolation
try:
    from scipy.interpolate import interp1d
    _HAVE_SCIPY = True
except ImportError:
    interp1d = None
    _HAVE_SCIPY = False


# ---- configuration defaults (edit if needed) ----
BAD_PROP_THRESHOLD = 0.20  # Stage 2 exclusion if BAD > 20% of segment
RR_SHORT_MS = 300.0
RR_LONG_MS = 1800.0
CV_WINDOW_BEATS = 12
CV_THRESHOLD = 0.12  # coefficient of variation threshold for instability
NOISE_STD_RATIO = 3.0  # relative to median moving-std
SEVERE_NOISE_RATIO = 6.0  # peaks removed if above this
MOVING_STD_SEC = 1.0  # window for noise screening

PLOT_COLORS = {
    "raw": "#1f77b4",
    "clean": "#2ca02c",
    "peaks": "#d62728",
    "hr": "#9467bd",
    "bad": "#ffbb78",
}

@dataclass
class SegmentResult:
    name: str
    start_idx: int
    end_idx: int
    fs: float
    clean_ecg: np.ndarray
    rpeaks_idx: np.ndarray
    rr_ms: np.ndarray
    rr_times_s: np.ndarray
    bad_intervals: List[Tuple[float, float, str]]

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


def cargar_senal_y_eventos(acq_file: str, ecg_col: str):
    """
    Lee ECG escalado con neurokit2.read_acqknowledge y marcadores con bioread.

    Devuelve:
      - ecg_raw: np.ndarray (ECG completo)
      - fs_ecg: float (Hz)
      - eventos: dict[str,int] (texto -> sample_index; si hay duplicados, usa el primero)
      - eventos_full: dict[str,list[int]] (texto -> todos los sample_index)
      - data_events: objeto bioread
    """
    # Read ECG data using neurokit2 (handles scaling properly)
    physio_file, fs = nk.read_acqknowledge(acq_file)

    # Allow ecg_col to be optional or incorrect: try to auto-detect ECG channel
    if ecg_col is None or ecg_col not in physio_file.columns:
        candidate_cols = [c for c in physio_file.columns if "ECG" in str(c).upper()]
        if not candidate_cols:
            numeric_cols = [c for c in physio_file.columns if np.issubdtype(physio_file[c].dtype, np.number)]
            if numeric_cols:
                chosen = numeric_cols[0]
                print(f"  ⚠️  ECG channel '{ecg_col}' not found - using numeric column '{chosen}' as fallback")
                ecg_col = chosen
            else:
                raise ValueError(
                    f"El canal ECG '{ecg_col}' no existe y no hay columnas numéricas. Columnas: {list(physio_file.columns)}"
                )
        else:
            ecg_col = candidate_cols[0]
            print(f"  ⚠️  ECG channel not specified or not found - auto-detected '{ecg_col}'")

    ecg_raw = physio_file[ecg_col].values
    fs_ecg = float(fs)

    # Read event markers using bioread
    _ensure_optional(bioread, "bioread")
    data_events = bioread.read_file(acq_file)

    # Parse event markers: text -> list of sample indices
    eventos_full: Dict[str, List[int]] = {}
    for m in data_events.event_markers:
        k = (m.text or "").strip()
        if k == "":
            continue
        eventos_full.setdefault(k, []).append(int(m.sample_index))

    # eventos "simple": primer marcador de cada texto
    eventos: Dict[str, int] = {k: v[0] for k, v in eventos_full.items()}

    print("=== INFO ECG ===")
    print(f"fs_ecg: {fs_ecg} Hz")
    print(f"n_total samples: {len(ecg_raw)}")
    print(f"ECG range: {np.nanmin(ecg_raw):.4f} to {np.nanmax(ecg_raw):.4f}")

    print("\n=== EVENTS FOUND ===")
    for k, v in eventos_full.items():
        if len(v) == 1:
            print(repr(k), "->", v[0])
        else:
            print(repr(k), "->", v, " (DUPLICATE)")

    return ecg_raw, fs_ecg, eventos, eventos_full, data_events


def cargar_datos_y_segmentos(
    acq_file: str,
    ecg_col: str,
    segmentos_interes: Sequence[str],
    duraciones_objetivo: Optional[Dict[str, float]] = None,
    segmentos_manuales: Sequence[str] = (),
):
    """
    Carga la señal ECG y eventos, y construye el diccionario de segmentos.
    
    Returns:
        ecg_raw, fs_ecg, eventos, eventos_full, data_events, segmentos_dict
    """
    ecg_raw, fs_ecg, eventos, eventos_full, data_events = cargar_senal_y_eventos(acq_file, ecg_col)
    
    segmentos_dict = construir_diccionario_segmentos(
        eventos=eventos,
        segmentos_interes=segmentos_interes,
        duraciones_objetivo=duraciones_objetivo,
        fs_ecg=fs_ecg,
        n_total=len(ecg_raw),
        segmentos_manuales=segmentos_manuales,
    )
    
    return ecg_raw, fs_ecg, eventos, eventos_full, data_events, segmentos_dict


def _keys_segmento(seg: str) -> Tuple[str, str]:
    """
    Devuelve (key_ini, key_fin) con soporte:
      - Intero N:  "Intero 1" / "Fin intero 1"
      - Post...:   (si lo usas) "Post 10min" / fin = n_total
      - General:   "Inicio X" / "Fin X"
    """
    seg_lower = seg.lower().strip()

    if seg_lower.startswith("intero"):
        key_ini = seg.strip()
        key_fin = "Fin " + seg_lower
        return key_ini, key_fin

    # BH, CPT, Recovery, LB, SB...
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
                raise ValueError(f"Segmento tipo Post '{seg}' no permitido.")
            if "Post 10min" not in eventos:
                raise ValueError("Falta marcador 'Post 10min'.")
            print(f"[OK] Segmento {seg}: inicio encontrado en 'Post 10min'")
            continue

        key_ini, key_fin = _keys_segmento(seg)

        if key_ini not in eventos:
            raise ValueError(f"Falta marcador '{key_ini}' para el segmento {seg}")
        if key_fin not in eventos:
            raise ValueError(f"Falta marcador '{key_fin}' para el segmento {seg}")

        print(f"[OK] Segment {seg}: '{key_ini}' and '{key_fin}' found.")

    print("Marker check completed.\n")


def get_segment_indices(
    eventos: Dict[str, int], seg: str, n_total: int
) -> Tuple[int, int]:
    """Get (start_idx, end_idx) sample indices for a segment."""
    seg_lower = seg.lower().strip()

    if seg_lower.startswith("post"):
        if "Post 10min" not in eventos:
            raise ValueError("Falta marcador 'Post 10min'")
        start = int(eventos["Post 10min"])
        end = int(n_total)
        return start, end

    key_ini, key_fin = _keys_segmento(seg)

    start = int(eventos[key_ini])
    end = int(eventos[key_fin])

    if start < 0 or end > n_total or end <= start:
        raise ValueError(
            f"Segmento '{seg}' fuera de rango (start={start}, end={end}, n_total={n_total})"
        )

    return start, end


def construir_diccionario_segmentos(
    eventos: Dict[str, int],
    segmentos_interes: Sequence[str],
    duraciones_objetivo: Optional[Dict[str, float]] = None,
    fs_ecg: Optional[float] = None,
    n_total: Optional[int] = None,
    segmentos_manuales: Sequence[str] = (),
) -> Dict[str, Tuple[int, int]]:
    """
    Build a {segmento: (start_idx, end_idx)} dict using marker conventions.
    Returns sample indices (not times).
    """
    duraciones_objetivo = duraciones_objetivo or {}
    if not eventos:
        return {}

    segmentos_dict: Dict[str, Tuple[int, int]] = {}
    for seg in segmentos_interes:
        if seg in segmentos_manuales:
            continue  # built by the study workflow, not from start/end markers
        try:
            start_idx, end_idx = get_segment_indices(eventos, seg, n_total or 0)
            segmentos_dict[seg] = (start_idx, end_idx)
        except (KeyError, ValueError) as e:
            # If duration is provided, try to compute end from start + duration
            seg_lower = seg.lower().strip()
            key_ini, _ = _keys_segmento(seg)
            if key_ini in eventos and seg in duraciones_objetivo and fs_ecg is not None:
                start_idx = eventos[key_ini]
                dur_samples = int(duraciones_objetivo[seg] * fs_ecg)
                end_idx = start_idx + dur_samples
                if n_total is not None:
                    end_idx = min(end_idx, n_total)
                segmentos_dict[seg] = (start_idx, end_idx)
                print(f"  Using target duration for {seg}: {duraciones_objetivo[seg]}s")
            else:
                print(f"  Could not build segment {seg}: {e}")
    return segmentos_dict


def inspeccionar_duraciones_segmentos(
    segmentos_dict: Dict[str, Tuple[int, int]],
    segmentos_interes: Sequence[str],
    fs_ecg: float,
    sujeto_id: str = "N/A"
) -> None:
    """
    Muestra una tabla con las duraciones, tiempos de inicio/fin e índices de muestras
    para todos los segmentos de interés.
    
    Args:
        segmentos_dict: Diccionario con {segmento: (start_idx, end_idx)}
        segmentos_interes: Lista de nombres de segmentos a inspeccionar
        fs_ecg: Frecuencia de muestreo del ECG (Hz)
        sujeto_id: ID del sujeto para mostrar en el encabezado
    """
    print(f"\n=== SEGMENT DURATION INSPECTION (SUBJECT {sujeto_id}) ===\n")
    for seg in segmentos_interes:
        info = segmentos_dict.get(seg)
        if info is None:
            print(f"{seg:15s}: NOT FOUND in segmentos_dict")
            continue
        start_idx, end_idx = info
        dur_s = (end_idx - start_idx) / fs_ecg if fs_ecg else float("nan")
        t_start_s = start_idx / fs_ecg if fs_ecg else float("nan")
        t_end_s = end_idx / fs_ecg if fs_ecg else float("nan")
        print(f"{seg:15s}: {dur_s:7.2f} s  |  Start: {t_start_s:8.2f} s  |  End: {t_end_s:8.2f} s  |  Samples: {start_idx:8d} - {end_idx:8d}")
    print()


def extraer_segmento(ecg_raw: np.ndarray, fs: float, start_s: float, end_s: float) -> np.ndarray:
    """Slice ECG between start and end seconds."""
    i0 = max(int(math.floor(start_s * fs)), 0)
    i1 = min(int(math.ceil(end_s * fs)), len(ecg_raw))
    return ecg_raw[i0:i1]


def detectar_peaks_rr(ecg_segment: np.ndarray, fs: float, parameter_preset: str = "human_ecg"):
    """
    Clean ECG and detect R-peaks using physio (preferred) or neurokit2 fallback.
    """
    if _HAVE_PHYSIO:
        # Use physio - from physiopy package
        clean_ecg, ecg_peaks_df = physio.compute_ecg(
            ecg_segment,
            srate=fs,
            parameter_preset=parameter_preset
        )
        if "peak_index" in ecg_peaks_df.columns:
            rpeaks_idx = ecg_peaks_df["peak_index"].astype(int).values
        else:
            rpeaks_idx = np.array([], dtype=int)
    else:
        # Fallback to neurokit2
        clean_ecg = nk.ecg_clean(ecg_segment, sampling_rate=int(fs), method="neurokit")
        _, info = nk.ecg_peaks(clean_ecg, sampling_rate=int(fs), method="neurokit")
        rpeaks_idx = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    
    rr_ms, rr_times = _rr_from_peaks(rpeaks_idx, fs)
    return clean_ecg, rpeaks_idx, rr_ms, rr_times


def _rr_from_peaks(rpeaks_idx: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(rpeaks_idx) < 2:
        return np.array([]), np.array([])
    rr = np.diff(rpeaks_idx) / fs * 1000.0
    times = rpeaks_idx[:-1] / fs
    return rr, times


def _calc_hr_from_rr(rr_ms: np.ndarray) -> np.ndarray:
    if len(rr_ms) == 0:
        return np.array([])
    return 60000.0 / rr_ms


def _moving_std(signal: np.ndarray, win_samples: int) -> np.ndarray:
    if win_samples <= 1:
        return np.zeros_like(signal)
    kernel = np.ones(win_samples) / win_samples
    mean = np.convolve(signal, kernel, mode="same")
    mean_sq = np.convolve(signal ** 2, kernel, mode="same")
    var = np.maximum(mean_sq - mean ** 2, 0)
    return np.sqrt(var)


def _merge_intervals(intervals: Iterable[Tuple[float, float]], margin: float = 0.0):
    intervals = [(a - margin, b + margin) for a, b in intervals if b > a]
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for a, b in intervals[1:]:
        last_a, last_b = merged[-1]
        if a <= last_b:
            merged[-1] = (last_a, max(last_b, b))
        else:
            merged.append((a, b))
    return merged


def _intervals_from_rr_indices(
    rpeaks_idx: np.ndarray, fs: float, idx_start: int, idx_end: int
) -> Tuple[float, float]:
    """RR index to time interval (sec) spanning involved peaks."""
    idx_start = max(idx_start, 0)
    idx_end = min(idx_end, len(rpeaks_idx) - 1)
    return rpeaks_idx[idx_start] / fs, rpeaks_idx[idx_end] / fs


def aplicar_etapa1_structural(
    seg_name: str,
    ecg_raw: np.ndarray,
    fs: float,
    segmentos_dict: Dict[str, Tuple[int, int]],
    rr_short_ms: float = RR_SHORT_MS,
    rr_long_ms: float = RR_LONG_MS,
    cv_window_beats: int = CV_WINDOW_BEATS,
    cv_threshold: float = CV_THRESHOLD,
    noise_std_ratio: float = NOISE_STD_RATIO,
    severe_noise_ratio: float = SEVERE_NOISE_RATIO,
    moving_std_sec: float = MOVING_STD_SEC,
) -> SegmentResult:
    """Stage 1: structural corrections + BAD marking."""
    if seg_name not in segmentos_dict:
        raise KeyError(f"{seg_name} no está en segmentos_dict.")
    start_idx, end_idx = segmentos_dict[seg_name]
    ecg_segment = ecg_raw[start_idx:end_idx]
    clean_ecg, rpeaks_idx, rr_ms, rr_times = detectar_peaks_rr(ecg_segment, fs)
    bad_intervals: List[Tuple[float, float, str]] = []

    # 1) clusters of short RR -> delete internal peaks
    rr_short = rr_ms < rr_short_ms
    clusters = []
    i = 0
    while i < len(rr_short):
        if rr_short[i]:
            start = i
            while i + 1 < len(rr_short) and rr_short[i + 1]:
                i += 1
            end = i
            if end - start + 1 >= 2:
                clusters.append((start, end))
        i += 1
    if clusters:
        mask = np.ones(len(rpeaks_idx), dtype=bool)
        for start, end in clusters:
            # remove internal peaks of the cluster
            mask[start + 1 : end + 1] = False
        rpeaks_idx = rpeaks_idx[mask]
        rr_ms, rr_times = _rr_from_peaks(rpeaks_idx, fs)

    # 2) isolated short RR -> flag only
    rr_short = rr_ms < rr_short_ms
    for k, is_short in enumerate(rr_short):
        if not is_short:
            continue
        in_cluster = any(start <= k <= end for start, end in clusters)
        if not in_cluster:
            bad_intervals.append((*_intervals_from_rr_indices(rpeaks_idx, fs, k, k + 1), "RR corto aislado"))

    # 3) short-long / long-short -> mark BAD
    for k in range(len(rr_ms) - 1):
        cond1 = rr_ms[k] < rr_short_ms and rr_ms[k + 1] > rr_long_ms
        cond2 = rr_ms[k] > rr_long_ms and rr_ms[k + 1] < rr_short_ms
        if cond1 or cond2:
            bad_intervals.append((*_intervals_from_rr_indices(rpeaks_idx, fs, k, k + 2), "patrón corto-largo"))

    # 4) sustained RR instability via CV
    if len(rr_ms) >= cv_window_beats:
        cv_vals = []
        for k in range(len(rr_ms) - cv_window_beats + 1):
            window = rr_ms[k : k + cv_window_beats]
            cv = np.std(window) / np.mean(window)
            cv_vals.append(cv)
            if cv > cv_threshold:
                bad_intervals.append(
                    (*_intervals_from_rr_indices(rpeaks_idx, fs, k, k + cv_window_beats), f"CV>{cv_threshold:.2f}")
                )
    else:
        cv_vals = []

    # 5) noise detection on cleaned ECG
    win = max(int(moving_std_sec * fs), 1)
    mov_std = _moving_std(clean_ecg, win)
    median_std = np.median(mov_std) if mov_std.size else 0
    if median_std > 0:
        ratio = mov_std / median_std
        noisy = ratio > noise_std_ratio
        severe = ratio > severe_noise_ratio
        if noisy.any():
            noisy_idxs = np.where(noisy)[0]
            noise_intervals = _samples_to_intervals(noisy_idxs, fs)
            bad_intervals.extend([(a, b, "ruido alto") for a, b in noise_intervals])
        if severe.any():
            severe_idxs = np.where(severe)[0]
            severe_intervals = _samples_to_intervals(severe_idxs, fs)
            # remove peaks inside severe noise
            if len(rpeaks_idx):
                keep_mask = np.ones(len(rpeaks_idx), dtype=bool)
                for a, b in severe_intervals:
                    keep_mask &= ~((rpeaks_idx / fs >= a) & (rpeaks_idx / fs <= b))
                rpeaks_idx = rpeaks_idx[keep_mask]
                rr_ms, rr_times = _rr_from_peaks(rpeaks_idx, fs) if len(rpeaks_idx) >= 2 else (np.array([]), np.array([]))

    return SegmentResult(
        name=seg_name,
        start_idx=start_idx,
        end_idx=end_idx,
        fs=fs,
        clean_ecg=clean_ecg,
        rpeaks_idx=rpeaks_idx,
        rr_ms=rr_ms,
        rr_times_s=rr_times,
        bad_intervals=bad_intervals,
    )


def _samples_to_intervals(idxs: np.ndarray, fs: float) -> List[Tuple[float, float]]:
    """Convert sample indices to contiguous time intervals."""
    if len(idxs) == 0:
        return []
    idxs = np.sort(idxs)
    starts = [idxs[0]]
    ends = []
    for i in range(1, len(idxs)):
        if idxs[i] != idxs[i - 1] + 1:
            ends.append(idxs[i - 1])
            starts.append(idxs[i])
    ends.append(idxs[-1])
    return [(s / fs, (e + 1) / fs) for s, e in zip(starts, ends)]


def plot_ecg_window(ecg_raw_seg, ecg_proc_seg, times_seg,
                    r_idx_seg, hr_inst_seg, t0, win_len=60.0,
                    seg_actual="SEG"):
    """Plot a single window of ECG data with raw, processed, and HR."""
    if plt is None:
        print("matplotlib not installed; no plots generated.")
        return
    
    mask = (times_seg >= t0) & (times_seg < t0 + win_len)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print(f"Window [{t0:.1f},{t0+win_len:.1f}] s empty")
        return

    i0, i1 = idx[0], idx[-1] + 1

    # Use REAL segment times (not starting from 0)
    t_win = times_seg[i0:i1]
    raw_win = ecg_raw_seg[i0:i1]
    proc_win = ecg_proc_seg[i0:i1]
    hr_win = hr_inst_seg[i0:i1]

    # R-peaks positions in real time
    r_mask = (r_idx_seg >= i0) & (r_idx_seg < i1)
    r_idx_win = r_idx_seg[r_mask]
    r_times_win = times_seg[r_idx_win] if len(r_idx_win) > 0 else np.array([])

    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    t_end = t0 + win_len
    fig.suptitle(f"{seg_actual} - [{t0:.1f}s - {t_end:.1f}s]")

    lw = 0.6

    ax = axs[0]
    ax.plot(t_win, raw_win, lw=lw)
    if len(r_times_win) > 0:
        ax.scatter(r_times_win, raw_win[r_idx_win - i0], s=10, label="R peaks")
    ax.set_ylabel("ECG raw")
    ax.legend(loc="upper right")

    ax = axs[1]
    ax.plot(t_win, proc_win, lw=lw)
    if len(r_times_win) > 0:
        ax.scatter(r_times_win, proc_win[r_idx_win - i0], s=10, label="R peaks")
    ax.set_ylabel("ECG proc")
    ax.legend(loc="upper right")

    ax = axs[2]
    ax.plot(t_win, hr_win, lw=lw)
    ax.set_ylabel("HR (bpm)")
    ax.set_xlabel("Tiempo en segmento (s)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()


def inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name="LB",
                             win_len=60.0, limites_hr=(30, 200),
                             parameter_preset="human_ecg",
                             segmentos_dict=None,
                             mostrar_info_segmentos_dict=True,
                             mostrar_tabla_peaks=True,
                             mostrar_outliers_rr=True,
                             mostrar_ventanas=True):
    """
    Recorta el segmento con marcadores, detecta picos con physio.compute_ecg,
    calcula HR inst y plotea por ventanas.
    Devuelve seg_dict completo.
    
    If segmentos_dict is provided, it will be used to get segment boundaries
    instead of eventos (useful when markers have wrong names, like subject 11 BH4).

    Args:
        mostrar_info_segmentos_dict: mostrar mensaje extra al usar segmentos_dict
        mostrar_tabla_peaks: imprimir la tabla inicial de peaks detectados
        mostrar_outliers_rr: imprimir RR medio y advertencias ±3*SD
        mostrar_ventanas: mostrar y graficar las ventanas
    """
    if plt is None:
        print("matplotlib not installed; no plots generated.")
        return None
    if not _HAVE_PHYSIO:
        raise ImportError("physio no está disponible. Instalar con: pip install physio")
    
    print("\n" + "=" * 60)
    print(f"Inspeccionando segmento: {seg_name}")

    n_total = len(ecg_raw)
    
    # Use segmentos_dict if provided, otherwise use eventos
    if segmentos_dict is not None and seg_name in segmentos_dict:
        start_idx, end_idx = segmentos_dict[seg_name]
        if mostrar_info_segmentos_dict:
            print(f"  (usando segmentos_dict para límites)")
    else:
        start_idx, end_idx = get_segment_indices(eventos, seg_name, n_total)
    dur_s = (end_idx - start_idx) / fs_ecg
    print(f"Segmento {seg_name}: duración total = {dur_s:.1f} s")

    ecg_seg_raw = ecg_raw[start_idx:end_idx].copy()  # Make a copy to preserve original raw signal
    times_seg = np.arange(len(ecg_seg_raw)) / fs_ecg

    # Use physio.compute_ecg (from physiopy package)
    ecg_proc, ecg_peaks = physio.compute_ecg(
        ecg_seg_raw,
        srate=fs_ecg,
        parameter_preset=parameter_preset
    )

    # Asegurar columnas mínimas
    ecg_peaks = ecg_peaks.copy().reset_index(drop=True)
    if "peak_index" not in ecg_peaks.columns:
        raise ValueError("physio.compute_ecg no devolvió columna 'peak_index'.")

    if "peak_time" not in ecg_peaks.columns:
        ecg_peaks["peak_time"] = ecg_peaks["peak_index"].astype(int) / fs_ecg

    if mostrar_tabla_peaks:
        print("\nPrimeros picos detectados:")
        print(ecg_peaks.head())
    r_idx = ecg_peaks["peak_index"].astype(int).values
    print(f"Nº de picos detectados: {len(r_idx)}")

    hr_inst = physio.compute_instantaneous_rate(
        ecg_peaks,
        new_times=times_seg,
        limits=list(limites_hr),
        units="bpm",
        interpolation_kind="linear",
    )
    print(f"HR media (inst) en {seg_name}: {np.nanmean(hr_inst):.1f} bpm")

    # Detectar outliers RR (para mostrar después de las ventanas)
    outliers_rr = np.array([])
    if len(r_idx) > 2:
        rr = np.diff(r_idx) / fs_ecg
        rr_mean = np.mean(rr)
        rr_std = np.std(rr)
        print(f"RR medio: {rr_mean:.3f} s, RR SD: {rr_std:.3f} s")
        outliers_rr = np.where((rr < rr_mean - 3 * rr_std) | (rr > rr_mean + 3 * rr_std))[0]
        if len(outliers_rr) > 0:
            print(f"WARNING: {len(outliers_rr)} RR intervals outside ±3*SD (possible artifact/erroneous detection).")
    else:
        print("Muy pocos picos para estimar RR.")

    dur_seg = float(times_seg[-1]) if len(times_seg) else 0.0
    print(f"Duración real del segmento (times_seg): {dur_seg:.2f} s")

    n_windows = int(math.ceil(dur_seg / win_len)) if dur_seg > 0 else 0

    # 1. Primero: mostrar ventanas normales de inspección
    for k in range(n_windows):
        t0 = k * win_len
        win = min(win_len, dur_seg - t0)
        if win <= 0:
            break
        t_end = t0 + win
        print(f"\nWindow {k+1}/{n_windows}: [{t0:.1f}s - {t_end:.1f}s]")
        plot_ecg_window(
            ecg_seg_raw, ecg_proc, times_seg,
            r_idx, hr_inst,
            t0=t0, win_len=win,
            seg_actual=seg_name
        )
    
    # 2. Después: si hay outliers RR, mostrar zoom de esos tramos
    if len(outliers_rr) > 0:
        print(f"\n{'='*60}")
        print(f"ZOOM EN INTERVALOS RR ANOMALOS ({len(outliers_rr)} detectados)")
        print(f"{'='*60}")
        seg_preview = {
            "seg": seg_name,
            "ecg_raw": ecg_seg_raw,
            "ecg_proc": ecg_proc,
            "times": times_seg,
            "r_idx": r_idx,
            "hr_inst": hr_inst,
        }
        _mostrar_outlier_zoom(seg_preview, outliers_rr, fs_ecg, max_zoom=3)

    return {
        "seg": seg_name,
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "dur_s": float(dur_s),
        "fs_ecg": float(fs_ecg),
        "times": times_seg,
        "ecg_raw": ecg_seg_raw.copy(),  # Store as copy to preserve original raw signal
        "ecg_proc": ecg_proc,
        "ecg_peaks": ecg_peaks,
        "r_idx": r_idx,
        "hr_inst": hr_inst,
        "tabla_beats": None,
        "bad_intervals": pd.DataFrame(columns=["t_ini_s", "t_fin_s", "motivo", "sev"]),
        "peak_changes": {
            "peaks_added": [],
            "peaks_removed": [],
            "peaks_moved": []
        }
    }


def aplicar_etapa2_interpolar(
    seg_result: SegmentResult,
    prop_bad_threshold: float = BAD_PROP_THRESHOLD,
):
    """Stage 2: merge BAD, check threshold, interpolate NN series."""
    merged_bad = _merge_intervals([(a, b) for a, b, _ in seg_result.bad_intervals])
    dur_bad = sum(b - a for a, b in merged_bad)
    prop_bad = dur_bad / max(seg_result.duration_s, 1e-9)
    accepted = prop_bad <= prop_bad_threshold

    rr_ms = seg_result.rr_ms
    rr_times = seg_result.rr_times_s
    if len(rr_ms) == 0:
        accepted = False

    bad_mask = np.zeros_like(rr_ms, dtype=bool)
    for a, b in merged_bad:
        bad_mask |= ((rr_times >= a) & (rr_times <= b))

    rr_series = pd.Series(rr_ms, index=rr_times)
    rr_series_masked = rr_series.mask(bad_mask, np.nan)

    if accepted:
        rr_interp = rr_series_masked.interpolate(
            method="time", limit_direction="both"
        ).to_numpy()
    else:
        rr_interp = np.array([])

    return {
        "accepted": bool(accepted),
        "prop_bad": float(prop_bad),
        "bad_intervals_merged": merged_bad,
        "rr_times_s": rr_times,
        "rr_ms_raw": rr_ms,
        "rr_ms_interp": rr_interp,
        "bad_mask": bad_mask,
    }


# =============================================================================
# CLEANING HELPER FUNCTIONS (from ecg_preprocess_bh.py)
# =============================================================================

def actualizar_tabla_beats(seg_dict: dict) -> dict:
    """Compute RR intervals table from peaks."""
    ecg_peaks = seg_dict.get("ecg_peaks", None)
    if ecg_peaks is None or not isinstance(ecg_peaks, pd.DataFrame) or len(ecg_peaks) < 2:
        seg_dict["tabla_beats"] = pd.DataFrame()
        return seg_dict
    
    fs = seg_dict.get("fs_ecg", None)
    if fs is None:
        seg_dict["tabla_beats"] = pd.DataFrame()
        return seg_dict
    fs = float(fs)
    dfp = ecg_peaks.copy().sort_values("peak_index").reset_index(drop=True)
    
    pi = dfp["peak_index"].astype(int).values
    pt = dfp["peak_time"].astype(float).values
    
    rr_ms = np.diff(pi) / fs * 1000.0
    
    df_rr = pd.DataFrame({
        "rr_idx": np.arange(len(rr_ms)),
        "peak_row": np.arange(1, len(pi)),
        "peak_index": pi[1:],
        "peak_time_s": pt[1:],
        "prev_peak_index": pi[:-1],
        "prev_peak_time_s": pt[:-1],
        "rr_ms": rr_ms
    })
    df_rr["rr_s"] = df_rr["rr_ms"] / 1000.0
    df_rr["hr_from_rr_bpm"] = 60.0 / df_rr["rr_s"]
    
    seg_dict["ecg_peaks"] = dfp
    seg_dict["tabla_beats"] = df_rr
    return seg_dict


def recomputar_hr_desde_peaks(seg_dict: dict, limites_hr: Tuple[float, float] = (30, 200)) -> dict:
    """Recompute instantaneous HR using physio."""
    if not _HAVE_PHYSIO:
        seg_dict["hr_inst"] = np.full_like(seg_dict["times"], np.nan, dtype=float)
        return seg_dict
    
    ecg_peaks = seg_dict.get("ecg_peaks", None)
    if ecg_peaks is None or len(ecg_peaks) < 2:
        seg_dict["hr_inst"] = np.full_like(seg_dict["times"], np.nan, dtype=float)
        return seg_dict
    
    hr_inst = physio.compute_instantaneous_rate(
        ecg_peaks,
        new_times=seg_dict["times"],
        limits=list(limites_hr),
        units="bpm",
        interpolation_kind="linear",
    )
    seg_dict["hr_inst"] = hr_inst
    seg_dict["r_idx"] = ecg_peaks["peak_index"].astype(int).values
    return seg_dict


def actualizar_todo(seg_dict: dict, limites_hr: Tuple[float, float] = (30, 200)) -> dict:
    """Update all derived fields after peak modifications."""
    seg_dict = actualizar_tabla_beats(seg_dict)
    seg_dict = recomputar_hr_desde_peaks(seg_dict, limites_hr=limites_hr)
    return seg_dict


def marcar_bad_interval(seg_dict: dict, t_ini: float, t_fin: float, 
                        motivo: str = "bad", sev: float = 1.0) -> dict:
    """
    Add a BAD interval to seg_dict.
    Evita marcar intervalos que contengan peaks movidos/insertados manualmente.
    """
    # Verificar si hay peaks modificados manualmente en este intervalo
    peak_changes = seg_dict.get("peak_changes", {
        "peaks_added": [],
        "peaks_removed": [],
        "peaks_moved": []
    })
    
    t_min = float(min(t_ini, t_fin))
    t_max = float(max(t_ini, t_fin))
    
    # Verificar peaks agregados en este intervalo
    for t_added in peak_changes.get("peaks_added", []):
        if t_min <= t_added <= t_max:
            # Hay un peak agregado manualmente en este intervalo, no marcar como bad
            return seg_dict
    
    # Verificar peaks movidos en este intervalo
    for t_orig, t_new in peak_changes.get("peaks_moved", []):
        # Verificar si el intervalo contiene el peak nuevo (donde se movió)
        if t_min <= t_new <= t_max:
            # Hay un peak movido manualmente en este intervalo, no marcar como bad
            return seg_dict
    
    # Si no hay peaks modificados, proceder con el marcado normal
    bi = seg_dict.get("bad_intervals", None)
    new_row = {
        "t_ini_s": t_min,
        "t_fin_s": t_max,
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


def borrar_peaks_por_fila(seg_dict: dict, peak_rows: List[int], 
                          limites_hr: Tuple[float, float] = (30, 200)) -> dict:
    """Delete peaks by row index."""
    peak_rows = sorted(set(int(r) for r in peak_rows))
    before = len(seg_dict["ecg_peaks"])
    seg_dict["ecg_peaks"] = seg_dict["ecg_peaks"].drop(index=peak_rows, errors="ignore").reset_index(drop=True)
    after = len(seg_dict["ecg_peaks"])
    print(f"[borrar_peaks] {seg_dict.get('seg','SEG')}: {before}->{after} (borrados={before-after})")
    seg_dict = actualizar_todo(seg_dict, limites_hr=limites_hr)
    return seg_dict


def borrar_peaks_por_intervalo(seg_dict: dict, t_ini: float, t_fin: float,
                               limites_hr: Tuple[float, float] = (30, 200)) -> dict:
    """Delete peaks within a time interval."""
    a, b = float(min(t_ini, t_fin)), float(max(t_ini, t_fin))
    dfp = seg_dict["ecg_peaks"].copy().reset_index(drop=True)
    mask = (dfp["peak_time"].astype(float) >= a) & (dfp["peak_time"].astype(float) <= b)
    rows = dfp.index[mask].tolist()
    if len(rows) == 0:
        return seg_dict
    return borrar_peaks_por_fila(seg_dict, rows, limites_hr=limites_hr)


def ver_peaks_en_tiempo(seg_dict: dict, t_centro: float, ventana_s: float = 5.0) -> pd.DataFrame:
    """
    Ver los peaks alrededor de un tiempo especifico.
    Util para identificar que peaks borrar manualmente.
    
    Args:
        t_centro: Tiempo central en segundos
        ventana_s: Ventana de tiempo alrededor del centro (±ventana_s/2)
    
    Returns: DataFrame con los peaks en esa ventana
    
    Example:
        ver_peaks_en_tiempo(resultado_cpt, t_centro=147.0, ventana_s=6.0)
    """
    seg_name = seg_dict.get("seg", "SEG")
    ecg_peaks = seg_dict.get("ecg_peaks", pd.DataFrame())
    
    if len(ecg_peaks) == 0:
        print("No peaks detected.")
        return pd.DataFrame()
    
    t_ini = t_centro - ventana_s / 2
    t_fin = t_centro + ventana_s / 2
    
    mask = (ecg_peaks["peak_time"].astype(float) >= t_ini) & (ecg_peaks["peak_time"].astype(float) <= t_fin)
    peaks_ventana = ecg_peaks[mask].copy()
    peaks_ventana = peaks_ventana.reset_index()
    peaks_ventana = peaks_ventana.rename(columns={"index": "peak_row"})
    
    # Calculate RR and HR for context
    if len(peaks_ventana) > 1:
        times = peaks_ventana["peak_time"].astype(float).values
        rr_ms = np.diff(times) * 1000
        hr_bpm = 60000.0 / rr_ms
        peaks_ventana["rr_ms"] = [np.nan] + list(rr_ms)
        peaks_ventana["hr_bpm"] = [np.nan] + list(hr_bpm)
    
    print(f"\n{'='*60}")
    print(f"PEAKS en {seg_name} [{t_ini:.1f}s - {t_fin:.1f}s]")
    print(f"{'='*60}")
    
    for _, row in peaks_ventana.iterrows():
        t = float(row["peak_time"])
        peak_row = int(row["peak_row"])
        rr = row.get("rr_ms", np.nan)
        hr = row.get("hr_bpm", np.nan)
        rr_str = f"RR={rr:.0f}ms" if pd.notna(rr) else ""
        hr_str = f"HR={hr:.0f}bpm" if pd.notna(hr) else ""
        print(f"  Peak {peak_row}: t={t:.2f}s  {rr_str}  {hr_str}")
    
    return peaks_ventana


def ampliar_senal_en_tiempo(seg_dict: dict, t_centro: float, ventana_s: float = 10.0):
    """
    Ampliar y visualizar la señal ECG en un tiempo específico.
    Muestra ECG raw, ECG proc, peaks detectados y HR en una ventana de tiempo.
    
    Args:
        seg_dict: Diccionario del segmento
        t_centro: Tiempo central en segundos donde ampliar
        ventana_s: Ancho de la ventana de tiempo en segundos (por defecto 10s)
    
    Example:
        ampliar_senal_en_tiempo(resultado, t_centro=205.0, ventana_s=15.0)
    """
    if plt is None:
        print("matplotlib no está disponible.")
        return
    
    seg_name = seg_dict.get("seg", "SEG")
    ecg_raw = seg_dict.get("ecg_raw", np.array([]))
    ecg_proc = seg_dict.get("ecg_proc", np.array([]))
    times = seg_dict.get("times", np.array([]))
    r_idx = seg_dict.get("r_idx", np.array([]))
    hr_inst = seg_dict.get("hr_inst", np.array([]))
    
    if len(times) == 0:
        print("No hay datos de señal para ampliar.")
        return
    
    # Calcular límites de la ventana
    t_ini = max(0, t_centro - ventana_s / 2)
    t_fin = min(times[-1], t_centro + ventana_s / 2)
    
    # Máscara para la ventana de tiempo
    mask = (times >= t_ini) & (times <= t_fin)
    idx = np.where(mask)[0]
    
    if len(idx) == 0:
        print(f"Ventana [{t_ini:.1f}s - {t_fin:.1f}s] está fuera del rango de datos.")
        return
    
    i0, i1 = idx[0], idx[-1] + 1
    
    # Datos de la ventana
    t_win = times[i0:i1]
    raw_win = ecg_raw[i0:i1]
    proc_win = ecg_proc[i0:i1]
    hr_win = hr_inst[i0:i1] if len(hr_inst) == len(times) else np.full_like(t_win, np.nan)
    
    # R-peaks en la ventana
    r_mask = (r_idx >= i0) & (r_idx < i1)
    r_idx_win = r_idx[r_mask]
    r_times_win = times[r_idx_win] if len(r_idx_win) > 0 else np.array([])
    
    # Crear figura con 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{seg_name} - Ampliación en t={t_centro:.1f}s [{t_ini:.1f}s - {t_fin:.1f}s]", fontsize=12)
    
    lw = 0.8
    
    # Plot 1: ECG raw
    ax = axs[0]
    ax.plot(t_win, raw_win, lw=lw, color="#1f77b4", label="ECG raw")
    if len(r_times_win) > 0:
        ax.scatter(r_times_win, raw_win[r_idx_win - i0], s=20, c="red", 
                  zorder=5, label="R peaks", marker="o")
    ax.axvline(t_centro, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, label="Tiempo central")
    ax.set_ylabel("ECG raw", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: ECG processed
    ax = axs[1]
    ax.plot(t_win, proc_win, lw=lw, color="#2ca02c", label="ECG proc")
    if len(r_times_win) > 0:
        ax.scatter(r_times_win, proc_win[r_idx_win - i0], s=20, c="red", 
                  zorder=5, label="R peaks", marker="o")
    ax.axvline(t_centro, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, label="Tiempo central")
    ax.set_ylabel("ECG proc", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: HR
    ax = axs[2]
    if not np.all(np.isnan(hr_win)):
        ax.plot(t_win, hr_win, lw=lw, color="#9467bd", label="HR")
    if len(r_times_win) > 0:
        ax.axvline(t_centro, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, label="Tiempo central")
    ax.set_ylabel("HR (bpm)", fontsize=10)
    ax.set_xlabel("Tiempo en segmento (s)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    # Mostrar información de peaks en la ventana
    if len(r_idx_win) > 0:
        print(f"\n{'='*60}")
        print(f"Zoom {seg_name} at t={t_centro:.1f}s [{t_ini:.1f}s - {t_fin:.1f}s]")
        print(f"{'='*60}")
        print(f"Peaks detectados en ventana: {len(r_idx_win)}")
        if len(r_idx_win) > 1:
            rr_times = r_times_win
            rr_intervals = np.diff(rr_times) * 1000  # en ms
            hr_values = 60000.0 / rr_intervals  # en bpm
            print(f"RR medio: {np.mean(rr_intervals):.0f} ms")
            print(f"HR media: {np.mean(hr_values):.1f} bpm")
    else:
        print(f"\n[!] No peaks detected in window [{t_ini:.1f}s - {t_fin:.1f}s]")


def _mostrar_outlier_zoom(seg_dict: dict, outliers: np.ndarray, fs_ecg: float, max_zoom: int = 2):
    if plt is None or len(outliers) == 0:
        return

    r_idx = np.array(seg_dict.get("r_idx", []), dtype=float)
    if len(r_idx) == 0:
        return

    r_times = r_idx / fs_ecg
    for idx in outliers[:max_zoom]:
        if idx >= len(r_times):
            continue
        t_ini = r_times[idx]
        t_fin = r_times[idx + 1] if idx + 1 < len(r_times) else t_ini + 1.0
        t_centro_zoom = (t_ini + t_fin) / 2
        ventana_zoom = max(8.0, (t_fin - t_ini) + 4.0)
        print(f"\n  Auto zoom: interval [{t_ini:.2f}s - {t_fin:.2f}s]")
        ampliar_senal_en_tiempo(seg_dict, t_centro=t_centro_zoom, ventana_s=ventana_zoom)


def borrar_peaks_manual(seg_dict: dict, t_ini: float, t_fin: float,
                        limites_hr: Tuple[float, float] = (30, 200),
                        verbose: bool = True) -> dict:
    """
    Borrar manualmente peaks en un rango de tiempo específico.
    
    Usar después de ver_peaks_en_tiempo() para identificar el rango.
    
    Args:
        t_ini: Tiempo inicial en segundos
        t_fin: Tiempo final en segundos
    
    Example:
        # Primero ver los peaks problemáticos:
        ver_peaks_en_tiempo(resultado_cpt, t_centro=147.0)
        
        # Luego borrar los que están mal:
        resultado_cpt = borrar_peaks_manual(resultado_cpt, t_ini=146.5, t_fin=148.0)
        
        # Re-inspeccionar:
        reinspeccionar_segmento(resultado_cpt)
    """
    seg_name = seg_dict.get("seg", "SEG")
    
    # Count peaks before
    n_before = len(seg_dict.get("ecg_peaks", []))
    
    # Delete peaks
    seg_dict = borrar_peaks_por_intervalo(seg_dict, t_ini, t_fin, limites_hr=limites_hr)
    
    # Count after
    n_after = len(seg_dict.get("ecg_peaks", []))
    n_deleted = n_before - n_after
    
    if verbose:
        print(f"\n[borrar_peaks_manual] {seg_name}: {n_deleted} peaks borrados en [{t_ini:.1f}s - {t_fin:.1f}s]")
        if n_deleted > 0:
            print(f"  Peaks: {n_before} -> {n_after}")
    
    # Update bad_pct
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    
    return seg_dict


def agregar_peak_manual(seg_dict: dict, t_nuevo: float,
                        limites_hr: Tuple[float, float] = (30, 200),
                        verbose: bool = True) -> dict:
    """
    Agregar manualmente un peak en un tiempo específico.
    Encuentra el pico R más cercano en la señal ECG alrededor de ese tiempo.
    Registra el cambio en peak_changes para evitar que se marque como bad.
    
    Args:
        t_nuevo: Tiempo aproximado donde agregar el peak (en segundos)
        
    Example:
        resultado_LB = agregar_peak_manual(resultado_LB, t_nuevo=123.5)
    """
    seg_name = seg_dict.get("seg", "SEG")
    fs = float(seg_dict.get("fs_ecg", 2000))
    ecg_proc = seg_dict.get("ecg_proc", np.array([]))
    times = seg_dict.get("times", np.array([]))
    ecg_peaks = seg_dict.get("ecg_peaks", pd.DataFrame()).copy()
    
    if len(ecg_proc) == 0:
        print(f"[agregar_peak] {seg_name}: No hay señal procesada")
        return seg_dict
    
    # Buscar el máximo local en ±100ms del tiempo especificado
    search_window_s = 0.1  # ±100ms
    t_ini = t_nuevo - search_window_s
    t_fin = t_nuevo + search_window_s
    
    idx_ini = max(0, int(t_ini * fs))
    idx_fin = min(len(ecg_proc), int(t_fin * fs))
    
    # Encontrar el máximo en esa ventana
    window_ecg = ecg_proc[idx_ini:idx_fin]
    if len(window_ecg) == 0:
        print(f"[add_peak] {seg_name}: Empty window")
        return seg_dict
    
    local_max_idx = np.argmax(window_ecg)
    new_peak_idx = idx_ini + local_max_idx
    new_peak_time = times[new_peak_idx] if new_peak_idx < len(times) else t_nuevo
    
    # Verificar que no exista ya un peak muy cerca
    existing_times = ecg_peaks["peak_time"].values if len(ecg_peaks) > 0 else np.array([])
    min_distance = 0.2  # 200ms mínimo entre peaks
    if len(existing_times) > 0:
        distances = np.abs(existing_times - new_peak_time)
        if np.min(distances) < min_distance:
            closest_idx = np.argmin(distances)
            print(f"[agregar_peak] {seg_name}: Ya existe un peak cercano en {existing_times[closest_idx]:.3f}s")
            return seg_dict
    
    # Agregar el nuevo peak
    new_row = pd.DataFrame({
        "peak_index": [new_peak_idx],
        "peak_time": [new_peak_time],
        "peak_amp": [ecg_proc[new_peak_idx] if new_peak_idx < len(ecg_proc) else 0]
    })
    ecg_peaks = pd.concat([ecg_peaks, new_row], ignore_index=True)
    ecg_peaks = ecg_peaks.sort_values("peak_time").reset_index(drop=True)
    
    seg_dict["ecg_peaks"] = ecg_peaks
    seg_dict["r_idx"] = ecg_peaks["peak_index"].astype(int).values
    
    # Registrar cambio en peak_changes
    peak_changes = seg_dict.get("peak_changes", {
        "peaks_added": [],
        "peaks_removed": [],
        "peaks_moved": []
    })
    peak_changes["peaks_added"].append(new_peak_time)
    seg_dict["peak_changes"] = peak_changes
    
    # Limpiar bad_intervals relacionadas con este peak agregado
    bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
    if len(bad_intervals) > 0:
        mask = ~((bad_intervals["t_ini_s"] <= new_peak_time + 0.5) & (bad_intervals["t_fin_s"] >= new_peak_time - 0.5))
        seg_dict["bad_intervals"] = bad_intervals[mask].reset_index(drop=True)
    
    # Recalcular RR y HR
    seg_dict = actualizar_todo(seg_dict, limites_hr=limites_hr)
    
    if verbose:
        print(f"[agregar_peak] {seg_name}: Peak agregado en {new_peak_time:.3f}s (idx={new_peak_idx})")
    
    return seg_dict


def mover_peak(seg_dict: dict, t_original: float, t_nuevo: float,
               limites_hr: Tuple[float, float] = (30, 200),
               verbose: bool = True) -> dict:
    """
    Mover un peak de una posicion a otra.
    Util cuando un peak esta ligeramente desplazado.
    Registra el cambio en peak_changes para evitar que se marque como bad.
    
    Args:
        t_original: Tiempo aproximado del peak actual (en segundos)
        t_nuevo: Tiempo aproximado donde debería estar el peak (en segundos)
        
    Example:
        # Ver los peaks para identificar cuál mover:
        ver_peaks_en_tiempo(resultado_LB, t_centro=123.0, ventana_s=5.0)
        
        # Mover el peak de 123.2s a 123.5s:
        resultado_LB = mover_peak(resultado_LB, t_original=123.2, t_nuevo=123.5)
    """
    seg_name = seg_dict.get("seg", "SEG")
    ecg_peaks = seg_dict.get("ecg_peaks", pd.DataFrame())
    
    if len(ecg_peaks) == 0:
        print(f"[mover_peak] {seg_name}: No hay peaks")
        return seg_dict
    
    # Encontrar el peak más cercano a t_original
    peak_times = ecg_peaks["peak_time"].values
    idx_closest = np.argmin(np.abs(peak_times - t_original))
    t_actual = peak_times[idx_closest]
    
    if np.abs(t_actual - t_original) > 0.5:
        print(f"[mover_peak] {seg_name}: No hay peak cerca de {t_original:.2f}s")
        print(f"  Peak más cercano en {t_actual:.3f}s (diferencia: {np.abs(t_actual - t_original):.3f}s)")
        return seg_dict
    
    if verbose:
        print(f"[mover_peak] {seg_name}: Moviendo peak de {t_actual:.3f}s -> {t_nuevo:.2f}s")
    
    # Borrar el peak original
    seg_dict = borrar_peaks_por_intervalo(seg_dict, t_actual - 0.05, t_actual + 0.05, limites_hr=limites_hr)
    
    # Agregar el nuevo peak (sin registrar cambio aquí, lo haremos después)
    seg_dict = agregar_peak_manual(seg_dict, t_nuevo, limites_hr=limites_hr, verbose=False)
    
    # Registrar el movimiento en peak_changes (remover el add que agregó agregar_peak_manual y agregar el move)
    peak_changes = seg_dict.get("peak_changes", {
        "peaks_added": [],
        "peaks_removed": [],
        "peaks_moved": []
    })
    # Remover el último add (que fue agregado por agregar_peak_manual)
    if peak_changes["peaks_added"] and len(peak_changes["peaks_added"]) > 0:
        peak_changes["peaks_added"].pop()
    # Agregar el movimiento
    peak_changes["peaks_moved"].append((t_actual, t_nuevo))
    seg_dict["peak_changes"] = peak_changes
    
    # Limpiar las bad_intervals relacionadas con este movimiento
    bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
    if len(bad_intervals) > 0:
        # Eliminar bad zones que cubran el área del peak movido
        mask = ~((bad_intervals["t_ini_s"] <= t_nuevo + 0.5) & (bad_intervals["t_fin_s"] >= t_nuevo - 0.5))
        seg_dict["bad_intervals"] = bad_intervals[mask].reset_index(drop=True)
    
    # Recalcular bad_pct
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    
    if verbose:
        print(f"  Peak movido correctamente. BAD: {bad_pct:.1%}")
    
    return seg_dict


def mover_peak_visual(seg_dict: dict, t_original: float, t_nuevo: float,
                      limites_hr: Tuple[float, float] = (30, 200)) -> dict:
    """
    Mover un peak con visualización comparativa.
    Muestra: ECG raw con peaks originales vs ECG proc con peaks corregidos.
    
    Args:
        seg_dict: Diccionario del segmento
        t_original: Tiempo del peak mal ubicado (segundos)
        t_nuevo: Tiempo correcto donde mover el peak (segundos)
        
    Returns: seg_dict actualizado
    
    Example:
        resultado_LB = mover_peak_visual(resultado_LB, t_original=108.21, t_nuevo=108.42)
    """
    seg_name = seg_dict.get("seg", "SEG")
    
    # Guardar peaks originales para comparar
    r_idx_original = seg_dict.get("r_idx", np.array([])).copy()
    
    # Mover el peak
    seg_dict = mover_peak(seg_dict, t_original, t_nuevo, limites_hr=limites_hr, verbose=True)
    
    # Obtener datos para visualización
    times = seg_dict.get("times", np.array([]))
    ecg_raw = seg_dict.get("ecg_raw", np.array([]))
    ecg_proc = seg_dict.get("ecg_proc", np.array([]))
    r_idx_nuevo = seg_dict.get("r_idx", np.array([]))
    hr_inst = seg_dict.get("hr_inst", np.array([]))
    
    # Ventana de visualización: 3s antes y 3s después
    t_plot = max(0, t_original - 3)
    win_plot = 6.0
    
    if len(times) == 0 or plt is None:
        return seg_dict
    
    # Crear figura con 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    mask = (times >= t_plot) & (times < t_plot + win_plot)
    t_win = times[mask]
    
    # ECG Raw con peaks ORIGINALES
    ax1 = axes[0]
    ax1.plot(t_win, ecg_raw[mask], 'b-', lw=0.8, label='ECG raw')
    for idx in r_idx_original:
        if idx < len(times) and t_plot <= times[idx] < t_plot + win_plot:
            ax1.axvline(times[idx], color='red', alpha=0.7, lw=1.5)
            ax1.plot(times[idx], ecg_raw[idx], 'ro', ms=6)
    ax1.set_ylabel('ECG Raw\n(peaks originales)', fontsize=10)
    ax1.set_title(f'{seg_name}: Peak movido de {t_original:.2f}s -> {t_nuevo:.2f}s', fontsize=12)
    ax1.legend(loc='upper right')
    
    # ECG Proc con peaks CORREGIDOS
    ax2 = axes[1]
    ax2.plot(t_win, ecg_proc[mask], 'g-', lw=0.8, label='ECG proc')
    for idx in r_idx_nuevo:
        if idx < len(times) and t_plot <= times[idx] < t_plot + win_plot:
            ax2.axvline(times[idx], color='green', alpha=0.7, lw=1.5)
            ax2.plot(times[idx], ecg_proc[idx], 'go', ms=6)
    ax2.set_ylabel('ECG Proc\n(peaks corregidos)', fontsize=10)
    ax2.legend(loc='upper right')
    
    # HR instantáneo (basado en peaks corregidos)
    ax3 = axes[2]
    if len(hr_inst) == len(times):
        ax3.plot(t_win, hr_inst[mask], 'm-', lw=1, label='HR (corregido)')
    ax3.set_ylabel('HR (bpm)', fontsize=10)
    ax3.set_xlabel('Tiempo (s)', fontsize=10)
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar peaks después del movimiento
    print("\nPeaks after move:")
    ver_peaks_en_tiempo(seg_dict, t_centro=t_original, ventana_s=6.0)
    
    return seg_dict






def editar_peaks_interactivos_externo(seg_dict, enable=True, t0: float = None, win_len: float = None):
    """
    Lanza el editor Tk externo para editar peaks de ECG.
    Similar a editar_ciclos_interactivos_externo de RSP pero para ECG.
    
    Args:
        seg_dict: Diccionario del segmento
        enable: Si False, retorna sin hacer nada
        t0: Tiempo inicial para mostrar en el editor (opcional)
        win_len: Longitud de ventana para mostrar (opcional)
        
    Returns:
        seg_dict actualizado con los cambios guardados
        
    Example:
        resultado_recovery = editar_peaks_interactivos_externo(resultado_recovery, enable=True)
    """
    if not enable:
        return seg_dict

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    
    # Preparar datos para el editor
    peaks = seg_dict.get("r_idx", np.array([]))
    times = seg_dict.get("times", np.array([]))
    ecg_proc = seg_dict.get("ecg_proc", np.array([]))
    fs = seg_dict.get("fs_ecg", 2000.0)
    seg_name = seg_dict.get("seg", "SEG")
    
    np.savez(
        tmp.name,
        peaks=peaks,
        times=times,
        ecg_proc=ecg_proc,
        fs=fs,
        seg=seg_name,
    )

    editor = SCRIPT_DIR / "peak_editor.py"  # biopac_physio/ecg/peak_editor.py
    if not editor.is_file():
        print(f"  [!] No se encuentra el editor: {editor}")
    cmd = [sys.executable, str(editor), tmp.name]
    if t0 is not None:
        cmd += ["--t0", str(t0)]
    if win_len is not None:
        cmd += ["--win_len", str(win_len)]
    
    # Run the editor (don't check=True to avoid exceptions if window is closed)
    # TEMPORAL: capturar salida para ver errores del editor
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print("  [!] Editor: timeout")
        result = None
    except Exception as e:
        print(f"  [!] Error ejecutando editor: {e}")
        result = None

    # Check if changes were saved
    status_file = tmp.name + ".saved"
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status = f.read().strip()
            os.remove(status_file)  # Clean up status file
        except Exception as e:
            print(f"  [!] Error leyendo archivo de estado: {e}")
            status = "cancelled"
        
        if status == "saved":
            print("  [OK] Changes saved, reloading data...")
            data = np.load(tmp.name)
            new_peaks = np.asarray(data["peaks"], dtype=int)
            
            ecg_peaks = seg_dict.get("ecg_peaks", pd.DataFrame())
            if len(ecg_peaks) > 0:
                new_peak_times = new_peaks / fs
                new_peak_indices = new_peaks
                new_peak_amps = ecg_proc[new_peaks] if len(ecg_proc) > 0 else np.zeros(len(new_peaks))
                
                seg_dict["ecg_peaks"] = pd.DataFrame({
                    "peak_index": new_peak_indices,
                    "peak_time": new_peak_times,
                    "peak_amp": new_peak_amps
                }).sort_values("peak_time").reset_index(drop=True)
            
            seg_dict["r_idx"] = new_peaks
            
            changes_file = tmp.name + ".changes.json"
            if os.path.exists(changes_file):
                try:
                    with open(changes_file, 'r') as f:
                        new_changes = json.load(f)
                    existing_changes = seg_dict.get("peak_changes", {
                        "peaks_added": [],
                        "peaks_removed": [],
                        "peaks_moved": []
                    })
                    merged_changes = {
                        "peaks_added": existing_changes.get("peaks_added", []) + new_changes.get("peaks_added", []),
                        "peaks_removed": existing_changes.get("peaks_removed", []) + new_changes.get("peaks_removed", []),
                        "peaks_moved": existing_changes.get("peaks_moved", []) + new_changes.get("peaks_moved", []),
                    }
                    seg_dict["peak_changes"] = merged_changes
                    
                    bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
                    if len(bad_intervals) > 0:
                        for t_added in merged_changes.get("peaks_added", []):
                            mask = ~((bad_intervals["t_ini_s"] <= t_added + 0.5) & (bad_intervals["t_fin_s"] >= t_added - 0.5))
                            bad_intervals = bad_intervals[mask].reset_index(drop=True)
                        for t_orig, t_new in merged_changes.get("peaks_moved", []):
                            mask = ~((bad_intervals["t_ini_s"] <= t_new + 0.5) & (bad_intervals["t_fin_s"] >= t_new - 0.5))
                            bad_intervals = bad_intervals[mask].reset_index(drop=True)
                        seg_dict["bad_intervals"] = bad_intervals
                except Exception as e:
                    print(f"  [!] Error leyendo cambios: {e}")
                finally:
                    try:
                        os.remove(changes_file)
                    except:
                        pass
            
            limites_hr = (30, 200)
            seg_dict = actualizar_todo(seg_dict, limites_hr=limites_hr)
        else:
            print("  [-] Changes cancelled, data not reloaded")
    else:
        print("  [-] Ventana cerrada sin guardar, no se recargaron datos")
        # TEMPORAL: mostrar salida del editor para diagnosticar
        if result is not None and (result.returncode != 0 or result.stderr or result.stdout):
            print("  [DEBUG] Editor returncode:", result.returncode)
            if result.stdout:
                print("  [DEBUG] stdout:", result.stdout[:2000])
            if result.stderr:
                print("  [DEBUG] stderr:", result.stderr[:2000])
    
    try:
        os.remove(tmp.name)
    except:
        pass
    
    return seg_dict



def borrar_peaks_en_zonas_bad(seg_dict: dict, zone_ids: List[int],
                               limites_hr: Tuple[float, float] = (30, 200)) -> dict:
    """
    Delete peaks in specified BAD zone IDs.
    
    Usage: After running reinspeccionar_segmento(), look at the zone numbers
           and call this function to delete peaks in specific zones.
           
    Example:
        bad_df = reinspeccionar_segmento(resultado_recovery)
        resultado_recovery = borrar_peaks_en_zonas_bad(resultado_recovery, [1, 2, 5])
    """
    bad_consolidated = consolidar_bad_intervals(seg_dict)
    
    if len(bad_consolidated) == 0:
        print("No hay zonas BAD definidas.")
        return seg_dict
    
    for zone_id in zone_ids:
        zone = bad_consolidated[bad_consolidated["id"] == zone_id]
        if len(zone) == 0:
            print(f"Zona [{zone_id}] no encontrada.")
            continue
        
        t_ini = float(zone.iloc[0]["t_ini_s"])
        t_fin = float(zone.iloc[0]["t_fin_s"])
        sev = int(zone.iloc[0]["severity"])
        sev_label = "SEVERE" if sev >= 2 else "BAD"
        
        print(f"Borrando peaks en zona [{zone_id}] ({t_ini:.1f}s - {t_fin:.1f}s) [{sev_label}]...")
        seg_dict = borrar_peaks_por_intervalo(seg_dict, t_ini, t_fin, limites_hr=limites_hr)
    
    return seg_dict


def agregar_bad_zone(seg_dict: dict, t_ini: float, t_fin: float, motivo: str = "manual") -> dict:
    """
    Manually add a BAD zone by time interval.
    
    Use when automatic detection misses a problematic region.
    
    Args:
        seg_dict: Segment dictionary
        t_ini: Start time in seconds
        t_fin: End time in seconds
        motivo: Reason for marking (default: "manual")
        
    Example:
        # Add BAD zone between 147.5s and 149.7s:
        resultado_cpt = agregar_bad_zone(resultado_cpt, 147.5, 149.7)
        reinspeccionar_segmento(resultado_cpt)
    """
    seg_name = seg_dict.get("seg", "SEG")
    seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo=motivo, sev=1.0)
    
    # Recalculate bad_pct
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    
    print(f"[agregar_bad_zone] {seg_name}: [OK] Zona agregada [{t_ini:.1f}s - {t_fin:.1f}s] ({motivo})")
    print(f"  BAD total: {bad_s:.1f}s ({bad_pct*100:.1f}%)")
    
    return seg_dict


def eliminar_bad_zones(seg_dict: dict, zone_ids: List[int]) -> dict:
    """
    Remove specific BAD zones by their ID numbers (does NOT delete peaks, just removes the BAD marking).
    
    Use after reinspeccionar_segmento() shows zones that are actually OK.
    
    Args:
        seg_dict: Segment dictionary
        zone_ids: List of zone IDs to remove (e.g., [7] or [3, 5, 7])
        
    Example:
        # After seeing zone 7 is actually OK:
        resultado_cpt = eliminar_bad_zones(resultado_cpt, [7])
        reinspeccionar_segmento(resultado_cpt)
    """
    seg_name = seg_dict.get("seg", "SEG")
    bad_consolidated = consolidar_bad_intervals(seg_dict)
    
    if len(bad_consolidated) == 0:
        print(f"[remove_bad_zones] {seg_name}: No BAD zones defined.")
        return seg_dict
    
    bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
    if len(bad_intervals) == 0:
        print(f"[eliminar_bad_zones] {seg_name}: No hay bad_intervals.")
        return seg_dict
    
    zones_removed = []
    for zone_id in zone_ids:
        zone = bad_consolidated[bad_consolidated["id"] == zone_id]
        if len(zone) == 0:
            print(f"[eliminar_bad_zones] {seg_name}: Zona [{zone_id}] no encontrada.")
            continue
        
        t_ini = float(zone.iloc[0]["t_ini_s"])
        t_fin = float(zone.iloc[0]["t_fin_s"])
        
        # Remove all bad_intervals that overlap with this zone
        # An interval overlaps if: interval.t_fin > zone.t_ini AND interval.t_ini < zone.t_fin
        mask_keep = ~(
            (bad_intervals["t_fin_s"].astype(float) > t_ini - 0.01) & 
            (bad_intervals["t_ini_s"].astype(float) < t_fin + 0.01)
        )
        n_before = len(bad_intervals)
        bad_intervals = bad_intervals[mask_keep].reset_index(drop=True)
        n_removed = n_before - len(bad_intervals)
        
        if n_removed > 0:
            zones_removed.append(zone_id)
            print(f"[eliminar_bad_zones] {seg_name}: [OK] Zona [{zone_id}] eliminada ({t_ini:.1f}s - {t_fin:.1f}s)")
    
    seg_dict["bad_intervals"] = bad_intervals
    
    # Recalculate bad_pct
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    
    if zones_removed:
        print(f"[eliminar_bad_zones] {seg_name}: {len(zones_removed)} zonas eliminadas. BAD: {bad_s:.1f}s ({bad_pct*100:.1f}%)")
    
    return seg_dict


def modificar_zona_bad(seg_dict: dict, zona_id: int, nuevo_t_ini: float = None, nuevo_t_fin: float = None) -> dict:
    """
    Modificar los límites de una zona BAD existente (extender o reducir).
    
    Args:
        seg_dict: Diccionario del segmento
        zona_id: ID de la zona a modificar (ver reinspeccionar_segmento)
        nuevo_t_ini: Nuevo tiempo inicial (None = mantener actual)
        nuevo_t_fin: Nuevo tiempo final (None = mantener actual)
        
    Example:
        # Extender zona [2] hacia la izquierda:
        resultado = modificar_zona_bad(resultado, zona_id=2, nuevo_t_ini=45.0)
        
        # Reducir zona [3] por la derecha:
        resultado = modificar_zona_bad(resultado, zona_id=3, nuevo_t_fin=52.0)
        
        # Cambiar ambos límites:
        resultado = modificar_zona_bad(resultado, zona_id=1, nuevo_t_ini=10.0, nuevo_t_fin=15.0)
    """
    seg_name = seg_dict.get("seg", "SEG")
    bad_consolidated = consolidar_bad_intervals(seg_dict)
    
    if len(bad_consolidated) == 0:
        print(f"[modify_bad_zone] {seg_name}: No BAD zones.")
        return seg_dict
    
    # Encontrar la zona por ID
    zona = bad_consolidated[bad_consolidated["id"] == zona_id]
    if len(zona) == 0:
        print(f"[modificar_zona_bad] {seg_name}: Zona [{zona_id}] no encontrada. IDs disponibles: {bad_consolidated['id'].tolist()}")
        return seg_dict
    
    t_ini_actual = float(zona.iloc[0]["t_ini_s"])
    t_fin_actual = float(zona.iloc[0]["t_fin_s"])
    motivos = zona.iloc[0]["motivos"]
    
    # Determinar nuevos límites
    t_ini_nuevo = nuevo_t_ini if nuevo_t_ini is not None else t_ini_actual
    t_fin_nuevo = nuevo_t_fin if nuevo_t_fin is not None else t_fin_actual
    
    if t_ini_nuevo >= t_fin_nuevo:
        print(f"[modificar_zona_bad] Error: t_ini ({t_ini_nuevo:.1f}s) debe ser menor que t_fin ({t_fin_nuevo:.1f}s)")
        return seg_dict
    
    # Eliminar la zona actual
    seg_dict = eliminar_bad_zones(seg_dict, [zona_id])
    
    # Añadir la zona con nuevos límites directamente a bad_intervals (sin pasar por
    # marcar_bad_interval) para que no se rechace por peaks manuales en el intervalo.
    bi = seg_dict.get("bad_intervals", pd.DataFrame(columns=["t_ini_s", "t_fin_s", "motivo", "sev"]))
    if not isinstance(bi, pd.DataFrame):
        bi = pd.DataFrame(columns=["t_ini_s", "t_fin_s", "motivo", "sev"])
    new_row = {
        "t_ini_s": float(t_ini_nuevo),
        "t_fin_s": float(t_fin_nuevo),
        "motivo": f"modificado:{motivos}",
        "sev": 1.0
    }
    bi = pd.concat([bi, pd.DataFrame([new_row])], ignore_index=True)
    seg_dict["bad_intervals"] = bi
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    
    print(f"[agregar_bad_zone] {seg_name}: [OK] Zona agregada [{t_ini_nuevo:.1f}s - {t_fin_nuevo:.1f}s] (modificado:{motivos})")
    print(f"  BAD total: {bad_s:.1f}s ({bad_pct*100:.1f}%)")
    print(f"[modificar_zona_bad] {seg_name}: Zona [{zona_id}] modificada:")
    print(f"  Antes: {t_ini_actual:.1f}s - {t_fin_actual:.1f}s")
    print(f"  Ahora: {t_ini_nuevo:.1f}s - {t_fin_nuevo:.1f}s")
    
    return seg_dict


def insertar_peak_visual(seg_dict: dict, t_nuevo: float,
                         limites_hr: Tuple[float, float] = (30, 200)) -> dict:
    """
    Insertar un peak no detectado con visualización comparativa.
    Muestra: ECG con peaks antes vs después de insertar.
    
    Args:
        seg_dict: Diccionario del segmento
        t_nuevo: Tiempo aproximado donde insertar el peak (segundos)
        
    Returns: seg_dict actualizado
    
    Example:
        # Ver zona problemática:
        ver_peaks_en_tiempo(resultado, t_centro=258.0, ventana_s=6.0)
        
        # Insertar peak faltante:
        resultado = insertar_peak_visual(resultado, t_nuevo=258.5)
    """
    seg_name = seg_dict.get("seg", "SEG")
    
    # Guardar peaks originales para comparar
    r_idx_original = seg_dict.get("r_idx", np.array([])).copy()
    
    # Insertar el peak
    seg_dict = agregar_peak_manual(seg_dict, t_nuevo, limites_hr=limites_hr, verbose=True)
    
    # Limpiar bad_intervals relacionadas con esta zona
    bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
    if len(bad_intervals) > 0:
        mask = ~((bad_intervals["t_ini_s"] <= t_nuevo + 0.5) & (bad_intervals["t_fin_s"] >= t_nuevo - 0.5))
        seg_dict["bad_intervals"] = bad_intervals[mask].reset_index(drop=True)
    
    # Obtener datos para visualización
    times = seg_dict.get("times", np.array([]))
    ecg_proc = seg_dict.get("ecg_proc", np.array([]))
    r_idx_nuevo = seg_dict.get("r_idx", np.array([]))
    hr_inst = seg_dict.get("hr_inst", np.array([]))
    
    # Ventana de visualización: 3s antes y 3s después
    t_plot = max(0, t_nuevo - 3)
    win_plot = 6.0
    
    if len(times) == 0 or plt is None:
        return seg_dict
    
    # Crear figura con 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    mask = (times >= t_plot) & (times < t_plot + win_plot)
    t_win = times[mask]
    
    # ECG con peaks ANTES de insertar
    ax1 = axes[0]
    ax1.plot(t_win, ecg_proc[mask], 'b-', lw=0.8, label='ECG')
    for idx in r_idx_original:
        if idx < len(times) and t_plot <= times[idx] < t_plot + win_plot:
            ax1.axvline(times[idx], color='red', alpha=0.7, lw=1.5)
            ax1.plot(times[idx], ecg_proc[idx], 'ro', ms=6)
    ax1.axvline(t_nuevo, color='orange', alpha=0.5, lw=2, linestyle='--', label='Nuevo peak')
    ax1.set_ylabel('ANTES\n(peak faltante)', fontsize=10)
    ax1.set_title(f'{seg_name}: Peak insertado en {t_nuevo:.2f}s', fontsize=12)
    ax1.legend(loc='upper right')
    
    # ECG con peaks DESPUES de insertar
    ax2 = axes[1]
    ax2.plot(t_win, ecg_proc[mask], 'g-', lw=0.8, label='ECG')
    for idx in r_idx_nuevo:
        if idx < len(times) and t_plot <= times[idx] < t_plot + win_plot:
            ax2.axvline(times[idx], color='green', alpha=0.7, lw=1.5)
            ax2.plot(times[idx], ecg_proc[idx], 'go', ms=6)
    ax2.set_ylabel('DESPUES\n(peak insertado)', fontsize=10)
    ax2.legend(loc='upper right')
    
    # HR instantáneo (basado en peaks corregidos)
    ax3 = axes[2]
    if len(hr_inst) == len(times):
        ax3.plot(t_win, hr_inst[mask], 'm-', lw=1, label='HR (corregido)')
    ax3.set_ylabel('HR (bpm)', fontsize=10)
    ax3.set_xlabel('Tiempo (s)', fontsize=10)
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar peaks después de insertar
    print("\nPeaks after insert:")
    ver_peaks_en_tiempo(seg_dict, t_centro=t_nuevo, ventana_s=6.0)
    
    return seg_dict


def revertir_a_checkpoint(checkpoint: dict, seg_dict: dict) -> dict:
    """
    Revertir el segmento al estado guardado en checkpoint.
    
    Uso: Cuando quieres deshacer cambios y volver al estado inicial.
    
    Args:
        checkpoint: Diccionario guardado con copy.deepcopy() antes de modificaciones
        seg_dict: Diccionario actual (se ignora, solo para mostrar qué se revierte)
        
    Returns: Copia del checkpoint
    
    Example:
        # Guardar checkpoint después de inspección:
        _checkpoint_LB = copy.deepcopy(resultado_LB)
        
        # Hacer modificaciones...
        resultado_LB = limpiar(resultado_LB, ...)
        
        # Si algo sale mal, revertir:
        resultado_LB = revertir_a_checkpoint(_checkpoint_LB, resultado_LB)
    """
    import copy
    seg_name = checkpoint.get("seg", "SEG")
    n_peaks_checkpoint = len(checkpoint.get("r_idx", []))
    n_peaks_actual = len(seg_dict.get("r_idx", []))
    
    print(f"[revertir_a_checkpoint] {seg_name}: Revirtiendo al checkpoint")
    print(f"  Peaks: {n_peaks_actual} -> {n_peaks_checkpoint}")
    
    return copy.deepcopy(checkpoint)


def revertir_borrado_clusters(seg_dict: dict, limites_hr: Tuple[float, float] = (30, 200)) -> dict:
    """
    Revertir SOLO el borrado de peaks en clusters, manteniendo otras modificaciones.
    
    Esto restaura los peaks que fueron borrados por la detección automática de clusters
    durante la llamada a limpiar(borrar_clusters=True).
    
    Args:
        seg_dict: Diccionario del segmento (debe haber pasado por limpiar())
        limites_hr: Límites de HR para recalcular
        
    Returns: seg_dict con los peaks de clusters restaurados
    
    Example:
        resultado = limpiar(resultado, borrar_clusters=True)
        # Ver que se borraron peaks que eran válidos...
        resultado = revertir_borrado_clusters(resultado)
    """
    seg_name = seg_dict.get("seg", "SEG")
    
    # Verificar si hay peaks guardados pre-clusters
    ecg_peaks_pre = seg_dict.get("_ecg_peaks_pre_clusters", pd.DataFrame())
    
    if len(ecg_peaks_pre) == 0:
        print(f"[revertir_borrado_clusters] {seg_name}: No hay peaks guardados para restaurar.")
        print("  Nota: Solo funciona si antes ejecutaste limpiar()")
        return seg_dict
    
    n_peaks_antes = len(seg_dict.get("ecg_peaks", pd.DataFrame()))
    n_peaks_pre = len(ecg_peaks_pre)
    
    if n_peaks_pre <= n_peaks_antes:
        print(f"[revert_cluster_removal] {seg_name}: No removed peaks to restore.")
        return seg_dict
    
    # Restaurar peaks
    seg_dict["ecg_peaks"] = ecg_peaks_pre.copy()
    
    # Recalcular todo
    seg_dict = actualizar_todo(seg_dict, limites_hr=limites_hr)
    
    n_restaurados = n_peaks_pre - n_peaks_antes
    print(f"[revertir_borrado_clusters] {seg_name}: [OK] {n_restaurados} peaks restaurados")
    print(f"  Peaks: {n_peaks_antes} -> {n_peaks_pre}")
    
    return seg_dict


def calcular_bad_pct(seg_dict: dict) -> Tuple[float, float, float]:
    """Calculate percentage of segment marked as BAD."""
    fs = float(seg_dict.get("fs_ecg", np.nan))
    n = len(seg_dict.get("ecg_raw", []))
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
# STAGE 1: STRUCTURAL CLEANING (MODULAR & SELECTIVE)
# =============================================================================

def detectar_problemas_etapa1(
    seg_dict: dict,
    limites_hr: Tuple[float, float] = (30, 200),
    short_abs_ms: float = 600.0,
    short_rel_factor: float = 0.70,
    long_abs_ms: float = 2000.0,
    long_rel_factor: float = 1.80,
    # CV detection - TIGHTER windows
    cv_ventana_beats: int = 4,    # Reduced from 6 (smaller window = tighter detection)
    cv_umbral: float = 0.25,
    cv_min_racha: int = 2,        # Reduced from 4 (triggers on shorter sequences)
    # Amplitude detection
    amp_win_s: float = 0.5,       # Reduced from 1.0 (tighter window)
    amp_step_s: float = 0.25,
    amp_z_umbral: float = 4.0,
    amp_min_dur_s: float = 0.3,   # Reduced from 0.5
    verbose: bool = True,
) -> Tuple[dict, dict]:
    """
    DETECT problems in ECG but DON'T delete anything automatically.
    Returns seg_dict with 'detections' dict containing all findings by type.
    
    User can then selectively apply fixes with aplicar_limpieza_selectiva().
    
    Detection types:
        - 'clusters': Short RR clusters (extra peaks to delete)
        - 'short_isolated': Isolated short RR (just mark)
        - 'long_rr': Long RR intervals (just mark)
        - 'short_long_pattern': Short-long adjacent pattern (just mark)
        - 'cv_instability': CV > threshold zones (just mark)
        - 'amplitude': High amplitude turbulence (just mark)
        - 'ectopic': Ectopic beats (just mark)
    
    Returns: (seg_dict, detections)
    """
    seg_name = seg_dict.get("seg", "SEG")
    print(f"\n{'='*60}")
    print(f"DETECCION DE PROBLEMAS: {seg_name}")
    print(f"{'='*60}")
    
    seg_dict = actualizar_todo(seg_dict, limites_hr=limites_hr)
    
    fs = float(seg_dict["fs_ecg"])
    dfr = seg_dict.get("tabla_beats", pd.DataFrame())
    
    # Initialize detections dict
    detections = {
        'clusters': [],         # [(t_ini, t_fin, peak_rows_to_delete), ...]
        'short_isolated': [],   # [(t_ini, t_fin), ...]
        'long_rr': [],          # [(t_ini, t_fin), ...]
        'short_long_pattern': [],  # [(t_ini, t_fin), ...]
        'cv_instability': [],   # [(t_ini, t_fin, cv_value), ...]
        'amplitude': [],        # [(t_ini, t_fin, z_score), ...]
        'ectopic': [],          # [(t_ini, t_fin, peak_row), ...]
    }
    
    if dfr is None or dfr.empty:
        print("[WARN] No hay tabla_beats para analizar.")
        seg_dict['detections'] = detections
        return seg_dict, detections
    
    rr = dfr["rr_ms"].astype(float).values
    rr_med = float(np.nanmedian(rr))
    thr_short_rel = rr_med * short_rel_factor
    thr_long_rel = rr_med * long_rel_factor
    
    # --- 1) Short RR intervals ---
    is_short = (rr < short_abs_ms) | (rr < thr_short_rel)
    
    runs_short = []
    i = 0
    while i < len(is_short):
        if is_short[i]:
            j = i
            while j + 1 < len(is_short) and is_short[j + 1]:
                j += 1
            runs_short.append((i, j))
            i = j + 1
        else:
            i += 1
    
    for a, b in runs_short:
        run_len = b - a + 1
        t_ini = float(dfr.loc[a, "prev_peak_time_s"])
        t_fin = float(dfr.loc[b, "peak_time_s"])
        
        if run_len >= 2:
            # Cluster: collect peaks to delete (the INTERNAL peaks)
            # For 2 consecutive short RRs [A->extra->B], we need to delete 'extra'
            pr_start = int(dfr.loc[a, "peak_row"])  # First extra peak
            pr_end = int(dfr.loc[b, "peak_row"])    # Peak after last extra
            peaks_to_delete = list(range(pr_start, pr_end))
            detections['clusters'].append((t_ini, t_fin, peaks_to_delete))
        else:
            detections['short_isolated'].append((t_ini, t_fin))
    
    # --- 2) Long RR intervals ---
    is_long = (rr > long_abs_ms) | (rr > thr_long_rel)
    for i, is_l in enumerate(is_long):
        if is_l:
            t_ini = float(dfr.loc[i, "prev_peak_time_s"])
            t_fin = float(dfr.loc[i, "peak_time_s"])
            detections['long_rr'].append((t_ini, t_fin))
    
    # --- 3) Short-long adjacent pattern ---
    for i in range(len(rr) - 1):
        if is_short[i] and is_long[i + 1]:
            t_ini = float(dfr.loc[i, "prev_peak_time_s"])
            t_fin = float(dfr.loc[i + 1, "peak_time_s"])
            detections['short_long_pattern'].append((t_ini, t_fin))
        elif is_long[i] and is_short[i + 1]:
            t_ini = float(dfr.loc[i, "prev_peak_time_s"])
            t_fin = float(dfr.loc[i + 1, "peak_time_s"])
            detections['short_long_pattern'].append((t_ini, t_fin))
    
    # --- 4) CV instability (TIGHTER marking) ---
    if len(rr) >= cv_ventana_beats:
        cvs = []
        for i in range(len(rr) - cv_ventana_beats + 1):
            w = rr[i:i + cv_ventana_beats]
            cv_val = float(np.nanstd(w) / np.nanmean(w)) if np.nanmean(w) > 0 else 0
            cvs.append(cv_val)
        
        cvs = np.asarray(cvs, dtype=float)
        high_cv = cvs > cv_umbral
        
        i = 0
        while i < len(high_cv):
            if not high_cv[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(high_cv) and high_cv[j + 1]:
                j += 1
            if (j - i + 1) >= cv_min_racha:
                # TIGHTER marking: only mark the affected beats, not full window extension
                # Start at first beat of first high-CV window
                t_ini = float(dfr.loc[i, "prev_peak_time_s"])
                # End at last beat of last high-CV window (just +1, not +window_size)
                end_idx = min(j + 1, len(dfr) - 1)  # Changed from j + cv_ventana_beats - 1
                t_fin = float(dfr.loc[end_idx, "peak_time_s"])
                max_cv = float(cvs[i:j + 1].max())
                detections['cv_instability'].append((t_ini, t_fin, max_cv))
            i = j + 1
    
    # --- 5) Amplitude turbulence ---
    ecg_proc = seg_dict.get("ecg_proc", seg_dict.get("ecg_raw", np.array([])))
    if len(ecg_proc) > 0:
        win_samp = int(amp_win_s * fs)
        step_samp = int(amp_step_s * fs)
        
        if win_samp > 0 and step_samp > 0:
            stds = []
            t0s = []
            for i in range(0, len(ecg_proc) - win_samp, step_samp):
                w = ecg_proc[i:i + win_samp]
                stds.append(float(np.nanstd(w)))
                t0s.append(float(i / fs))
            
            stds = np.asarray(stds, dtype=float)
            t0s = np.asarray(t0s, dtype=float)
            
            mu = float(np.nanmean(stds))
            sd = float(np.nanstd(stds)) + 1e-12
            z = (stds - mu) / sd
            
            mask = z > amp_z_umbral
            
            i = 0
            while i < len(mask):
                if not mask[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(mask) and mask[j + 1]:
                    j += 1
                t_ini = t0s[i]
                t_fin = t0s[j] + amp_win_s
                if (t_fin - t_ini) >= amp_min_dur_s:
                    max_z = float(z[i:j + 1].max())
                    detections['amplitude'].append((t_ini, t_fin, max_z))
                i = j + 1
    
    # --- 6) Ectopic detection ---
    tabla = seg_dict.get("tabla_beats", pd.DataFrame())
    if tabla is not None and len(tabla) >= 3:
        rr = tabla["rr_ms"].astype(float).values
        ectopic_threshold_pct = 20.0
        
        for i in range(1, len(rr) - 1):
            rr_prev = rr[i - 1]
            rr_curr = rr[i]
            rr_next = rr[i + 1]
            
            if rr_prev > 0 and rr_next > 0:
                diff_prev_pct = abs(rr_curr - rr_prev) / rr_prev * 100
                diff_next_pct = abs(rr_curr - rr_next) / rr_next * 100
                
                if diff_prev_pct > ectopic_threshold_pct and diff_next_pct > ectopic_threshold_pct:
                    peak_row = int(tabla.loc[i, "peak_row"])
                    t_prev = float(tabla.loc[i, "prev_peak_time_s"])
                    t_ectopic = float(tabla.loc[i, "peak_time_s"])
                    detections['ectopic'].append((t_prev, t_ectopic, peak_row))
    
    seg_dict['detections'] = detections
    
    # --- Print summary ---
    if verbose:
        print(f"\n[RESUMEN] RESUMEN DE DETECCIONES:")
        print(f"  [1] Clusters RR cortos:     {len(detections['clusters'])} -> peaks a borrar")
        if detections['clusters']:
            for i, (t0, t1, peaks) in enumerate(detections['clusters']):
                print(f"      -> Cluster {i+1}: {t0:.1f}s-{t1:.1f}s ({len(peaks)} peaks extra)")
        
        print(f"  [2] Short isolated RR:       {len(detections['short_isolated'])} -> mark BAD")
        print(f"  [3] Long RR:                {len(detections['long_rr'])} -> mark BAD")
        print(f"  [4] Short-long pattern:     {len(detections['short_long_pattern'])} -> mark BAD")
        print(f"  [5] CV instability:         {len(detections['cv_instability'])} -> marcar BAD")
        if detections['cv_instability']:
            for i, (t0, t1, cv) in enumerate(detections['cv_instability']):
                print(f"      -> Zona {i+1}: {t0:.1f}s-{t1:.1f}s (CV={cv:.2f})")
        
        print(f"  [6] Anomalous amplitude:    {len(detections['amplitude'])} -> mark BAD")
        if detections['amplitude']:
            for i, (t0, t1, z) in enumerate(detections['amplitude']):
                print(f"      -> Zona {i+1}: {t0:.1f}s-{t1:.1f}s (z={z:.1f})")
        
        print(f"  [7] Ectópicos:              {len(detections['ectopic'])} -> marcar BAD")
        
        print(f"\n[TIP] Usa aplicar_limpieza_selectiva() o limpiar() para corregir.")
    
    return seg_dict, detections


# =============================================================================
# SIMPLIFIED CLEANING (Colleague's approach)
# =============================================================================

def limpiar(
    seg_dict: dict,
    # Step 1: Remove extra peaks (clusters)
    borrar_clusters: bool = True,
    # Step 2: Ectopic detection threshold
    ectopic_threshold_pct: float = 20.0,
    # Step 3: Long RR detection
    long_rr_abs_ms: float = 2000.0,      # Absolute threshold (2 seconds)
    long_rr_rel_factor: float = 1.80,    # Relative threshold (1.8x median)
    # Step 4: Short-Long pattern uses 0.70x and 1.30x of median (hardcoded)
    # Step 5: HR transitorio threshold (set to 1.0 to disable)
    hr_change_threshold: float = 0.20,   # 20% = default, 1.0 = disabled
    # Step 5b: HR abrupto threshold as percentage (set to 1.0 to disable)
    # Changed from absolute 20 bpm to percentage-based (25%) to be less strict and consistent with ectopic/HR_transitorio
    hr_abrupto_threshold: float = 0.25,  # 25% = default (less strict than before), 1.0 = disabled
    # Step 6: HR physiological limits
    hr_min: float = 30.0,
    hr_max: float = 200.0,
    # Step 7: IQR outlier factor
    iqr_factor: float = 3.5,
    # Options
    limites_hr: Tuple[float, float] = (30, 200),
    min_clean_duration_s: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    """
    Limpieza automatica de segmento ECG:
    
    1. CLUSTERS    -> BORRAR peaks extra (errores de deteccion)
    2. ECTOPICOS   -> MARCAR para interpolacion (>20% diferencia)
    3. RR LARGOS   -> MARCAR intervalos muy largos (>1.8x mediana)
    4. PATRON      -> MARCAR si RR corto-largo o largo-corto juntos
    5. HR RANGOS   -> MARCAR si HR fuera de limites fisiologicos
    6. IQR OUTLIERS-> MARCAR valores extremos (Q1-3.5*IQR, Q3+3.5*IQR)
    
    All marked values become "BAD" (no SEVERE distinction).
    After this, call interpolar_rr_en_bad_zones() to fill the gaps.
    
    Args:
        min_clean_duration_s: Si se especifica, muestra si la duración clean cumple el mínimo
    """
    seg_name = seg_dict.get("seg", "SEG")
    
    print(f"\n{'='*70}")
    print(f"LIMPIEZA: {seg_name}")
    print(f"{'='*70}")
    
    # Ensure data is updated
    seg_dict = actualizar_todo(seg_dict, limites_hr=limites_hr)
    
    # Reset bad intervals (start fresh)
    seg_dict = reset_bad_intervals(seg_dict)
    
    n_peaks_inicial = len(seg_dict.get('r_idx', []))
    n_marcados = 0
    
    # -----------------------------------------------------------------
    # STEP 1: Remove extra peaks (clusters of short RR)
    # -----------------------------------------------------------------
    # Guardar estado antes de borrar clusters (para poder revertir)
    seg_dict["_ecg_peaks_pre_clusters"] = seg_dict.get("ecg_peaks", pd.DataFrame()).copy()
    
    if borrar_clusters:
        dfr = seg_dict.get("tabla_beats", pd.DataFrame())
        if len(dfr) > 0:
            rr = dfr["rr_ms"].astype(float).values
            rr_med = float(np.nanmedian(rr))
            thr_short = min(600.0, rr_med * 0.70)  # Short RR threshold
            
            is_short = rr < thr_short
            
            # Find runs of 2+ consecutive short RRs (clusters)
            peaks_to_delete = []
            i = 0
            while i < len(is_short):
                if is_short[i]:
                    j = i
                    while j + 1 < len(is_short) and is_short[j + 1]:
                        j += 1
                    if (j - i + 1) >= 2:  # Cluster of 2+ short RRs
                        pr_start = int(dfr.loc[i, "peak_row"])
                        pr_end = int(dfr.loc[j, "peak_row"])
                        peaks_to_delete.extend(range(pr_start, pr_end))
                    i = j + 1
                else:
                    i += 1
            
            if peaks_to_delete:
                seg_dict["_peaks_borrados_clusters"] = peaks_to_delete.copy()
                seg_dict = borrar_peaks_por_fila(seg_dict, peaks_to_delete, limites_hr=limites_hr)
                n_clusters_borrados = len(peaks_to_delete)
            else:
                seg_dict["_peaks_borrados_clusters"] = []
                n_clusters_borrados = 0
    else:
        seg_dict["_peaks_borrados_clusters"] = []
        n_clusters_borrados = 0
    
    if borrar_clusters:
        if n_clusters_borrados > 0:
            print(f"Automatic cluster removal: {n_clusters_borrados} peaks removed")
        else:
            print(f"Eliminación automática de Clusters: 0 (ningún cluster detectado)")
    
    # Refresh tabla_beats after potential deletion
    seg_dict = actualizar_tabla_beats(seg_dict)
    dfr = seg_dict.get("tabla_beats", pd.DataFrame())
    
    if len(dfr) == 0:
        print(f"  [!] No beats to process")
        return seg_dict
    
    rr = dfr["rr_ms"].astype(float).values
    hr = 60000.0 / rr
    
    # -----------------------------------------------------------------
    # STEP 2: Detect ectopic beats (>20% difference from neighbors)
    # -----------------------------------------------------------------
    # Lista para almacenar información de detecciones (para output posterior)
    detecciones_info = []
    
    ectopic_count = 0
    for i in range(1, len(rr) - 1):
        rr_prev, rr_curr, rr_next = rr[i-1], rr[i], rr[i+1]
        if rr_prev > 0 and rr_next > 0:
            diff_prev = abs(rr_curr - rr_prev) / rr_prev * 100
            diff_next = abs(rr_curr - rr_next) / rr_next * 100
            if diff_prev > ectopic_threshold_pct and diff_next > ectopic_threshold_pct:
                t_ini = float(dfr.loc[i-1, "peak_time_s"])
                t_fin = float(dfr.loc[i+1, "peak_time_s"]) if i+1 < len(dfr) else float(dfr.loc[i, "peak_time_s"])
                seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo="ectópico", sev=1.0)
                detecciones_info.append({
                    't_ini': t_ini, 't_fin': t_fin, 
                    'motivo': f"Ectópico: >{ectopic_threshold_pct:.0f}%"
                })
                ectopic_count += 1
                n_marcados += 1
    
    # -----------------------------------------------------------------
    # STEP 3: Long RR detection (catches "moved" peaks)
    # -----------------------------------------------------------------
    rr_med = float(np.nanmedian(rr))
    thr_long_rel = rr_med * long_rr_rel_factor
    thr_long = max(long_rr_abs_ms, thr_long_rel)  # Use the more conservative
    
    long_rr_count = 0
    for i in range(len(rr)):
        if rr[i] > thr_long:
            t_ini = float(dfr.loc[i, "prev_peak_time_s"]) if "prev_peak_time_s" in dfr.columns else float(dfr.loc[max(0, i-1), "peak_time_s"])
            t_fin = float(dfr.loc[i, "peak_time_s"])
            seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo="RR_largo", sev=1.0)
            detecciones_info.append({
                't_ini': t_ini, 't_fin': t_fin,
                'motivo': f"RR_largo: >{thr_long:.0f}ms"
            })
            long_rr_count += 1
            n_marcados += 1
    
    # -----------------------------------------------------------------
    # STEP 4: SHORT-LONG or LONG-SHORT pattern (moved peak detection)
    # A "moved" peak creates one short RR and one long RR adjacent
    # -----------------------------------------------------------------
    thr_short = rr_med * 0.70  # Short threshold
    thr_long_pattern = rr_med * 1.30  # Long threshold for pattern (less strict)
    
    pattern_count = 0
    for i in range(len(rr) - 1):
        rr_curr = rr[i]
        rr_next = rr[i + 1]
        
        # Check for SHORT-LONG pattern
        is_short_long = (rr_curr < thr_short) and (rr_next > thr_long_pattern)
        # Check for LONG-SHORT pattern
        is_long_short = (rr_curr > thr_long_pattern) and (rr_next < thr_short)
        
        if is_short_long or is_long_short:
            t_ini = float(dfr.loc[i, "prev_peak_time_s"]) if "prev_peak_time_s" in dfr.columns else float(dfr.loc[max(0, i-1), "peak_time_s"])
            t_fin = float(dfr.loc[i + 1, "peak_time_s"])
            pattern_type = "corto->largo" if is_short_long else "largo->corto"
            seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo=f"patrón_{pattern_type}", sev=1.0)
            detecciones_info.append({
                't_ini': t_ini, 't_fin': t_fin,
                'motivo': f"Patrón_{pattern_type}: RR: {rr_curr:.0f}ms -> {rr_next:.0f}ms"
            })
            pattern_count += 1
            n_marcados += 1
    
    # -----------------------------------------------------------------
    # STEP 5: Transient HR anomaly (drop + recovery = artifact)
    # If HR drops >20% AND recovers >20% within 4 beats, it's not real
    # Extended window to catch "moved" peaks where HR stays low for a few beats
    # -----------------------------------------------------------------
    transient_count = 0
    # hr_change_threshold comes from function parameter
    
    for i in range(1, len(hr) - 1):
        hr_prev = hr[i-1]
        hr_curr = hr[i]
        
        if hr_prev > 0 and hr_curr > 0:
            # Check for significant DROP from previous beat
            drop = (hr_prev - hr_curr) / hr_prev  # Positive if HR dropped
            
            if drop > hr_change_threshold:
                # Look for recovery within the next 4 beats
                recovery_found = False
                recovery_idx = i + 1
                recovery_pct = 0
                
                for j in range(1, 5):  # Check next 4 beats
                    if i + j < len(hr):
                        hr_future = hr[i + j]
                        if hr_curr > 0:
                            recovery = (hr_future - hr_curr) / hr_curr
                            if recovery > hr_change_threshold:
                                recovery_found = True
                                recovery_idx = i + j
                                recovery_pct = recovery
                                break
                
                if recovery_found:
                    t_ini = float(dfr.loc[max(0, i-1), "peak_time_s"])
                    t_fin = float(dfr.loc[min(recovery_idx, len(dfr)-1), "peak_time_s"])
                    seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo="HR_transitorio", sev=1.0)
                    detecciones_info.append({
                        't_ini': t_ini, 't_fin': t_fin,
                        'motivo': f"HR_transitorio: drop={drop*100:.0f}%, recovery={recovery_pct*100:.0f}%"
                    })
                    transient_count += 1
                    n_marcados += 1
            
            # Also check for SPIKE then DROP (opposite pattern)
            spike = (hr_curr - hr_prev) / hr_prev  # Positive if HR spiked
            
            if spike > hr_change_threshold:
                # Look for fall within the next 4 beats
                fall_found = False
                fall_idx = i + 1
                fall_pct = 0
                
                for j in range(1, 5):  # Check next 4 beats
                    if i + j < len(hr):
                        hr_future = hr[i + j]
                        if hr_curr > 0:
                            fall = (hr_curr - hr_future) / hr_curr
                            if fall > hr_change_threshold:
                                fall_found = True
                                fall_idx = i + j
                                fall_pct = fall
                                break
                
                if fall_found:
                    t_ini = float(dfr.loc[max(0, i-1), "peak_time_s"])
                    t_fin = float(dfr.loc[min(fall_idx, len(dfr)-1), "peak_time_s"])
                    seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo="HR_transitorio", sev=1.0)
                    detecciones_info.append({
                        't_ini': t_ini, 't_fin': t_fin,
                        'motivo': f"HR_transitorio: spike={spike*100:.0f}%, fall={fall_pct*100:.0f}%"
                    })
                    transient_count += 1
                    n_marcados += 1
    
    # -----------------------------------------------------------------
    # STEP 5b: Abrupt HR change (percentage-based, similar to ectopic/HR_transitorio)
    # Changed from absolute 20 bpm to percentage-based (25%) to be less strict
    # Only marks the immediate abrupt change interval
    # -----------------------------------------------------------------
    abrupt_count = 0
    hr_abrupt_threshold_pct = hr_abrupto_threshold  # from function parameter (now percentage-based)
    
    for i in range(1, len(hr)):
        hr_prev = hr[i-1]
        hr_curr = hr[i]
        
        if hr_prev > 0:
            hr_change_pct = abs(hr_curr - hr_prev) / hr_prev
            
            if hr_change_pct > hr_abrupt_threshold_pct:
                t_ini = float(dfr.loc[max(0, i-1), "peak_time_s"])
                t_fin = float(dfr.loc[i, "peak_time_s"])
                seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo="HR_abrupto", sev=1.0)
                detecciones_info.append({
                    't_ini': t_ini, 't_fin': t_fin,
                    'motivo': f"HR_abrupto: cambio={hr_change_pct*100:.0f}%"
                })
                abrupt_count += 1
                n_marcados += 1
    
    # -----------------------------------------------------------------
    # STEP 6: HR physiological range filter
    # -----------------------------------------------------------------
    hr_out_count = 0
    for i in range(len(hr)):
        if hr[i] < hr_min or hr[i] > hr_max:
            t_ini = float(dfr.loc[max(0, i-1), "peak_time_s"]) if i > 0 else 0
            t_fin = float(dfr.loc[i, "peak_time_s"])
            seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo="HR_fuera_rango", sev=1.0)
            detecciones_info.append({
                't_ini': t_ini, 't_fin': t_fin,
                'motivo': f"HR_fuera_rango: {hr[i]:.0f}bpm (límites: {hr_min}-{hr_max})"
            })
            hr_out_count += 1
            n_marcados += 1
    
    # -----------------------------------------------------------------
    # STEP 7: IQR outlier detection
    # -----------------------------------------------------------------
    q1 = float(np.nanpercentile(rr, 25))
    q3 = float(np.nanpercentile(rr, 75))
    iqr = q3 - q1
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr
    
    outlier_count = 0
    for i in range(len(rr)):
        if rr[i] < lower or rr[i] > upper:
            t_ini = float(dfr.loc[max(0, i-1), "peak_time_s"]) if i > 0 else 0
            t_fin = float(dfr.loc[i, "peak_time_s"])
            seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo="outlier_IQR", sev=1.0)
            detecciones_info.append({
                't_ini': t_ini, 't_fin': t_fin,
                'motivo': f"outlier_IQR: {rr[i]:.0f}ms (límites: {lower:.0f}-{upper:.0f})"
            })
            outlier_count += 1
            n_marcados += 1
    
    # -----------------------------------------------------------------
    # Summary with new format
    # -----------------------------------------------------------------
    seg_dict = actualizar_todo(seg_dict, limites_hr=limites_hr)
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    
    n_peaks_final = len(seg_dict.get('r_idx', []))
    clean_s = dur_s - bad_s
    
    # Mostrar zonas BAD detectadas (consolidadas)
    bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
    if len(bad_intervals) > 0:
        print(f"\nBAD zone detection:")
        # Agrupar intervalos que se solapan o están muy cerca
        if len(bad_intervals) > 0:
            bad_sorted = bad_intervals.sort_values("t_ini_s").reset_index(drop=True)
            zonas_consolidadas = []
            motivos_por_zona = []
            
            for idx, row in bad_sorted.iterrows():
                t_ini, t_fin = float(row["t_ini_s"]), float(row["t_fin_s"])
                motivo = row.get("motivo", "")
                
                # Buscar si se solapa con alguna zona existente
                merged = False
                for i, (z_ini, z_fin, z_motivos) in enumerate(zip(
                    [z[0] for z in zonas_consolidadas],
                    [z[1] for z in zonas_consolidadas],
                    motivos_por_zona
                )):
                    # Si se solapan o están muy cerca (dentro de 0.5s)
                    if t_ini <= z_fin + 0.5 and t_fin >= z_ini - 0.5:
                        zonas_consolidadas[i] = (min(z_ini, t_ini), max(z_fin, t_fin))
                        if motivo and motivo not in z_motivos:
                            z_motivos.append(motivo)
                        merged = True
                        break
                
                if not merged:
                    zonas_consolidadas.append((t_ini, t_fin))
                    motivos_por_zona.append([motivo] if motivo else [])
            
            # Mostrar zonas consolidadas
            for i, ((t_ini, t_fin), motivos) in enumerate(zip(zonas_consolidadas, motivos_por_zona), 1):
                dur_zona = t_fin - t_ini
                motivos_unicos = list(set(m for m in motivos if m))
                print(f"[{i}] {t_ini:.1f}s - {t_fin:.1f}s ({dur_zona:.1f}s):")
                for m in motivos_unicos:
                    print(f"    {m}")
    else:
        print(f"\nDetección zonas BAD: ninguna")
    
    # Resumen final
    print(f"\n{'-'*70}")
    print(f"RESUMEN:")
    
    # Duración con check de mínimo si se especificó
    dur_str = f"  -> Total duration: {dur_s:.1f}s"
    if min_clean_duration_s is not None:
        if clean_s >= min_clean_duration_s:
            dur_str += f" | minimum required: {min_clean_duration_s:.0f}s -> [OK] OK"
        else:
            dur_str += f" | minimum required: {min_clean_duration_s:.0f}s -> [!] INSUFFICIENT"
    print(dur_str)
    
    print(f"  -> Peaks: {n_peaks_inicial} -> {n_peaks_final} (borrados: {n_peaks_inicial - n_peaks_final})")
    print(f"  -> MARKED intervals: {n_marcados} (will be interpolated)")
    
    # BAD con status
    bad_status = "[OK] OK" if bad_pct <= 0.20 else "[!] REVISAR"
    print(f"  -> BAD total: {bad_s:.1f}s ({bad_pct*100:.1f}%) -> CLEAN: {clean_s:.1f}s -> {bad_status}")
    print(f"{'-'*70}")
    
    return seg_dict


# Alias para compatibilidad con código anterior
limpiar_simple = limpiar


def aplicar_limpieza_selectiva(
    seg_dict: dict,
    # ===== BORRAR PEAKS (elimina datos) =====
    borrar_clusters: bool = True,           # BORRAR peaks extra en clusters RR cortos
    borrar_peaks_en_zonas_marcadas: bool = False,  # BORRAR peaks dentro de zonas BAD
    
    # ===== MARCAR ZONAS (solo marca, peaks quedan, se interpola después) =====
    marcar_rr_cortos_aislados: bool = False,
    marcar_rr_largos: bool = False,
    marcar_patron_corto_largo: bool = False,
    marcar_cv_alto: bool = False,
    marcar_amplitud: bool = False,
    marcar_ectopicos: bool = False,
    
    limites_hr: Tuple[float, float] = (30, 200),
    verbose: bool = True,
) -> dict:
    """
    Limpieza selectiva con acciones CLARAS:
    
    +-----------------------------------------------------------------+
    | BORRAR (elimina peaks del dato):                                |
    |   -> borrar_clusters=True     -> Borra peaks extra en clusters    |
    |   -> borrar_peaks_en_zonas_marcadas=True -> Borra peaks en BAD    |
    |                                                                 |
    | MARCAR (zona amarilla, peaks quedan, RR se interpola después):  |
    |   -> marcar_cv_alto=True      -> Marca zonas con CV > umbral      |
    |   -> marcar_amplitud=True     -> Marca zonas con amplitud rara    |
    |   -> marcar_ectopicos=True    -> Marca beats ectópicos            |
    |   -> (etc.)                                                      |
    +-----------------------------------------------------------------+
    
    Returns: seg_dict con limpieza aplicada
    """
    seg_name = seg_dict.get("seg", "SEG")
    detections = seg_dict.get('detections', {})
    
    if not detections:
        print(f"[WARN] No hay detecciones. Primero ejecutar detectar_problemas_etapa1()")
        return seg_dict
    
    print(f"\n{'='*70}")
    print(f"LIMPIEZA SELECTIVA: {seg_name}")
    print(f"{'='*70}")
    
    # Reset bad intervals
    seg_dict = reset_bad_intervals(seg_dict)
    
    peaks_borrados = 0
    zonas_marcadas = 0
    
    # =====================================================================
    # PASO 1: BORRAR clusters (elimina peaks extra)
    # =====================================================================
    if borrar_clusters and detections.get('clusters'):
        all_peaks = []
        for t_ini, t_fin, peaks in detections['clusters']:
            all_peaks.extend(peaks)
        
        if all_peaks:
            n_before = len(seg_dict.get('r_idx', []))
            seg_dict = borrar_peaks_por_fila(seg_dict, all_peaks, limites_hr=limites_hr)
            n_after = len(seg_dict.get('r_idx', []))
            peaks_borrados += (n_before - n_after)
            print(f"  [DEL]  BORRADO clusters: {n_before - n_after} peaks eliminados")
    
    # =====================================================================
    # PASO 2: MARCAR zonas (solo marca, no borra - se interpola después)
    # =====================================================================
    if marcar_rr_cortos_aislados and detections.get('short_isolated'):
        for t_ini, t_fin in detections['short_isolated']:
            seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo="RR_corto", sev=1.0)
            zonas_marcadas += 1
    
    if marcar_rr_largos and detections.get('long_rr'):
        for t_ini, t_fin in detections['long_rr']:
            seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo="RR_largo", sev=1.0)
            zonas_marcadas += 1
    
    if marcar_patron_corto_largo and detections.get('short_long_pattern'):
        for t_ini, t_fin in detections['short_long_pattern']:
            seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo="patrón_corto_largo", sev=1.0)
            zonas_marcadas += 1
    
    if marcar_cv_alto and detections.get('cv_instability'):
        for t_ini, t_fin, cv in detections['cv_instability']:
            seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo="CV_alto", sev=cv)
            zonas_marcadas += 1
    
    if marcar_amplitud and detections.get('amplitude'):
        for t_ini, t_fin, z in detections['amplitude']:
            seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo="amp_turb", sev=z)
            zonas_marcadas += 1
    
    if marcar_ectopicos and detections.get('ectopic'):
        for t_ini, t_fin, peak_row in detections['ectopic']:
            seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo="ectópico", sev=1.0)
            zonas_marcadas += 1
        seg_dict['ectopic_indices'] = [p for _, _, p in detections['ectopic']]
    
    if zonas_marcadas > 0:
        print(f"  [MARK] MARKED: {zonas_marcadas} zones as BAD (peaks kept, RR will be interpolated)")
    
    # =====================================================================
    # PASO 3: BORRAR peaks en zonas marcadas (OPCIONAL - más agresivo)
    # =====================================================================
    if borrar_peaks_en_zonas_marcadas:
        # Consolidar BAD intervals para obtener DataFrame con zonas
        bad_df = consolidar_bad_intervals(seg_dict)
        if len(bad_df) > 0:
            peaks_antes = len(seg_dict.get('r_idx', []))
            for _, row in bad_df.iterrows():
                t_ini = float(row["t_ini_s"])
                t_fin = float(row["t_fin_s"])
                seg_dict = borrar_peaks_por_intervalo(seg_dict, t_ini, t_fin, limites_hr=limites_hr)
            peaks_despues = len(seg_dict.get('r_idx', []))
            borrados_extra = peaks_antes - peaks_despues
            if borrados_extra > 0:
                peaks_borrados += borrados_extra
                print(f"  [DEL]  BORRADO en zonas BAD: {borrados_extra} peaks adicionales eliminados")
    
    # Update all derived fields
    seg_dict = actualizar_todo(seg_dict, limites_hr=limites_hr)
    
    # Calculate and STORE bad_pct for later use in RESUMEN FINAL
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    
    # =====================================================================
    # RESUMEN FINAL
    # =====================================================================
    if verbose:
        print(f"\n{'-'*70}")
        print(f"RESUMEN:")
        n_peaks = len(seg_dict.get('r_idx', []))
        
        if peaks_borrados > 0:
            print(f"  -> Peaks BORRADOS: {peaks_borrados}")
        if zonas_marcadas > 0:
            print(f"  -> Zonas MARCADAS: {zonas_marcadas} (amarillo = solo marca, no borra)")
        if peaks_borrados == 0 and zonas_marcadas == 0:
            print(f"  -> Sin cambios aplicados")
        
        print(f"  -> Peaks finales: {n_peaks}")
        print(f"  -> BAD total: {bad_s:.1f}s ({bad_pct*100:.1f}%)")
        print(f"{'-'*70}")
    
    return seg_dict


def limpiar_segmento_etapa1(
    seg_dict: dict,
    limites_hr: Tuple[float, float] = (30, 200),
    short_abs_ms: float = 600.0,
    short_rel_factor: float = 0.70,
    long_abs_ms: float = 2000.0,
    long_rel_factor: float = 1.80,
    # CV instability detection (LESS AGGRESSIVE)
    cv_ventana_beats: int = 6,
    cv_umbral: float = 0.25,      # Higher threshold (was 0.20)
    cv_min_racha: int = 4,        # Require more consecutive windows (was 3)
    # Amplitude detection (LESS AGGRESSIVE)
    amp_win_s: float = 1.0,       # Smaller window (was 2.0)
    amp_step_s: float = 0.25,     # Finer step (was 0.5)
    amp_z_umbral: float = 4.0,    # Higher z-score required (was 3.0)
    amp_min_dur_s: float = 0.5,   # Shorter minimum duration (was 1.0)
    # Peak deletion thresholds (VERY CONSERVATIVE - rarely triggered)
    amp_drop_z: float = 10.0,     # Only delete peaks if extremely severe noise
    rr_drop_cv: float = 0.60,     # Only delete peaks if extremely unstable
    delete_short_rr_clusters: bool = False,  # Don't delete peaks in short RR clusters
    # Padding (NO PADDING by default)
    pad_mark_s: float = 0.0,      # No padding when marking BAD (was 0.03)
    pad_drop_s: float = 0.0,      # No padding when deleting
    verbose: bool = True,
) -> Tuple[dict, dict]:
    """
    Stage 1: Structural correction and BAD marking.
    
    Returns: (seg_dict, cleaning_info)
    """
    seg_name = seg_dict.get("seg", "SEG")
    print(f"\n{'='*60}")
    print(f"LIMPIEZA ETAPA 1: {seg_name}")
    print(f"{'='*60}")
    
    seg_dict = actualizar_todo(seg_dict, limites_hr=limites_hr)
    seg_dict = reset_bad_intervals(seg_dict)
    
    fs = float(seg_dict["fs_ecg"])
    dfr = seg_dict.get("tabla_beats", pd.DataFrame())
    
    if dfr is None or dfr.empty:
        print("[WARN] No hay tabla_beats para limpiar.")
        return seg_dict, {"rr_short_isolated": [], "rr_long_isolated": []}
    
    rr = dfr["rr_ms"].astype(float).values
    rr_med = float(np.nanmedian(rr))
    thr_short_rel = rr_med * short_rel_factor
    thr_long_rel = rr_med * long_rel_factor
    
    # --- 1) Short RR intervals ---
    is_short = (rr < short_abs_ms) | (rr < thr_short_rel)
    
    runs_short = []
    i = 0
    while i < len(is_short):
        if is_short[i]:
            j = i
            while j + 1 < len(is_short) and is_short[j + 1]:
                j += 1
            runs_short.append((i, j))
            i = j + 1
        else:
            i += 1
    
    peak_rows_short_cluster = []
    peak_rows_short_isolated = []
    
    for a, b in runs_short:
        run_len = b - a + 1
        t_ini = float(dfr.loc[a, "prev_peak_time_s"])
        t_fin = float(dfr.loc[b, "peak_time_s"])
        seg_dict = marcar_bad_interval(seg_dict, t_ini - pad_mark_s, t_fin + pad_mark_s,
                                       motivo="RR_corto_cluster" if run_len >= 2 else "RR_corto", 
                                       sev=float(run_len))
        if run_len >= 2:
            # Delete INTERNAL peaks of cluster (the extra peaks causing short RRs)
            pr_start = int(dfr.loc[a, "peak_row"])  # First extra peak
            pr_end = int(dfr.loc[b, "peak_row"])    # Peak after last extra
            for pr in range(pr_start, pr_end):  # Fixed: was pr_start + 1
                peak_rows_short_cluster.append(pr)
        else:
            peak_rows_short_isolated.append(int(dfr.loc[a, "peak_row"]))
    
    if delete_short_rr_clusters and len(peak_rows_short_cluster) > 0:
        seg_dict = borrar_peaks_por_fila(seg_dict, peak_rows_short_cluster, limites_hr=limites_hr)
        # HIERARCHICAL: Refresh tabla_beats after deleting peaks so step 2 uses updated data
        seg_dict = actualizar_tabla_beats(seg_dict)
        dfr = seg_dict.get("tabla_beats", pd.DataFrame())
        if len(dfr) > 0:
            rr = dfr["rr_ms"].astype(float).values
            rr_med = float(np.nanmedian(rr))
            thr_short_rel = rr_med * short_rel_factor
            thr_long_rel = rr_med * long_rel_factor
    
    # --- 2) Long RR intervals (HIERARCHICAL: uses updated data if peaks were deleted) ---
    # Refresh thresholds based on current RR data
    dfr = seg_dict.get("tabla_beats", pd.DataFrame())
    if dfr is not None and len(dfr) > 0:
        rr = dfr["rr_ms"].astype(float).values
        rr_med = float(np.nanmedian(rr))
        thr_long_rel = rr_med * long_rel_factor
    
    is_long = (rr > long_abs_ms) | (rr > thr_long_rel)
    
    runs_long = []
    i = 0
    while i < len(is_long):
        if is_long[i]:
            j = i
            while j + 1 < len(is_long) and is_long[j + 1]:
                j += 1
            runs_long.append((i, j))
            i = j + 1
        else:
            i += 1
    
    peak_rows_long = []
    for a, b in runs_long:
        t_ini = float(dfr.loc[a, "prev_peak_time_s"])
        t_fin = float(dfr.loc[b, "peak_time_s"])
        seg_dict = marcar_bad_interval(seg_dict, t_ini - pad_mark_s, t_fin + pad_mark_s,
                                       motivo="RR_largo", sev=1.0)
        peak_rows_long.append(int(dfr.loc[a, "peak_row"]))
    
    if verbose:
        print(f"\nRR cortos:")
        print(f"  mediana={rr_med:.1f}ms | umbral_abs={short_abs_ms} | umbral_rel={thr_short_rel:.1f}")
        print(f"  clusters: {len([r for r in runs_short if r[1]-r[0]>=1])} | aislados: {len(peak_rows_short_isolated)}")
        
        print(f"\nRR largos:")
        print(f"  umbral_abs={long_abs_ms} | umbral_rel={thr_long_rel:.1f}")
        print(f"  detectados: {len(runs_long)}")
    
    # --- 3) Short-long / long-short adjacent pattern -> mark BAD ---
    seg_dict = actualizar_tabla_beats(seg_dict)
    dfr = seg_dict.get("tabla_beats", pd.DataFrame())
    
    n_patron_corto_largo = 0
    if not dfr.empty:
        rr = dfr["rr_ms"].astype(float).values
        rr_med = float(np.nanmedian(rr))
        thr_short = min(short_abs_ms, rr_med * short_rel_factor)
        thr_long = max(long_abs_ms, rr_med * long_rel_factor)
        
        is_short = rr < thr_short
        is_long = rr > thr_long
        
        for i in range(len(rr) - 1):
            if (is_short[i] and is_long[i + 1]) or (is_long[i] and is_short[i + 1]):
                t_ini = float(dfr.loc[i, "prev_peak_time_s"])
                t_fin = float(dfr.loc[i + 1, "peak_time_s"])
                seg_dict = marcar_bad_interval(seg_dict, t_ini - pad_mark_s, t_fin + pad_mark_s,
                                               motivo="patrón_corto_largo", sev=1.0)
                n_patron_corto_largo += 1
    
    if verbose:
        print(f"\nShort-long pattern: {n_patron_corto_largo} -> BAD")
    
    # --- 4) Sustained RR instability (CV in sliding window) -> mark BAD ---
    seg_dict = actualizar_tabla_beats(seg_dict)
    dfr = seg_dict.get("tabla_beats", pd.DataFrame())
    
    n_cv_zones = 0
    if len(dfr) >= cv_ventana_beats:
        rr = dfr["rr_ms"].astype(float).values
        
        cv_high_idx = []
        for end in range(cv_ventana_beats - 1, len(rr)):
            start = end - (cv_ventana_beats - 1)
            w = rr[start:end + 1]
            m = float(np.nanmean(w))
            s = float(np.nanstd(w))
            cv = (s / m) if m > 0 else np.nan
            if cv > cv_umbral:
                cv_high_idx.append((start, end, cv))
        
        if cv_high_idx:
            i = 0
            while i < len(cv_high_idx):
                j = i
                while j + 1 < len(cv_high_idx) and cv_high_idx[j + 1][0] <= cv_high_idx[j][1] + 1:
                    j += 1
                if (j - i + 1) >= cv_min_racha:
                    t_ini = float(dfr.loc[cv_high_idx[i][0], "prev_peak_time_s"])
                    t_fin = float(dfr.loc[cv_high_idx[j][1], "peak_time_s"])
                    max_cv = max(x[2] for x in cv_high_idx[i:j + 1])
                    seg_dict = marcar_bad_interval(seg_dict, t_ini - pad_mark_s, t_fin + pad_mark_s,
                                                   motivo="CV_alto", sev=max_cv)
                    n_cv_zones += 1
                i = j + 1
    
    if verbose:
        print(f"\nCV instability (>{cv_umbral}): {n_cv_zones} zones -> BAD")
    
    # --- 5) Amplitude turbulence -> mark BAD ---
    n_amp_zones = 0
    ecg_proc = seg_dict.get("ecg_proc", seg_dict.get("ecg_raw", np.array([])))
    if len(ecg_proc) > 0:
        win = int(round(amp_win_s * fs))
        step = int(round(amp_step_s * fs))
        
        if win >= 5 and step >= 1 and len(ecg_proc) >= win:
            stds = []
            t0s = []
            for i in range(0, len(ecg_proc) - win + 1, step):
                w = ecg_proc[i:i + win]
                stds.append(float(np.nanstd(w)))
                t0s.append(float(i / fs))
            
            stds = np.asarray(stds, dtype=float)
            t0s = np.asarray(t0s, dtype=float)
            
            mu = float(np.nanmean(stds))
            sd = float(np.nanstd(stds)) + 1e-12
            z = (stds - mu) / sd
            
            mask = z > amp_z_umbral
            
            i = 0
            while i < len(mask):
                if not mask[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(mask) and mask[j + 1]:
                    j += 1
                t_ini = t0s[i]
                t_fin = t0s[j] + amp_win_s
                if (t_fin - t_ini) >= amp_min_dur_s:
                    max_z = float(z[i:j + 1].max())
                    seg_dict = marcar_bad_interval(seg_dict, t_ini - pad_mark_s, t_fin + pad_mark_s,
                                                   motivo="amp_turb", sev=max_z)
                    n_amp_zones += 1
                i = j + 1
    
    if verbose:
        print(f"\nAmplitude (z>{amp_z_umbral}): {n_amp_zones} zones -> BAD")
    
    # --- 6) Ectopic detection (>20% rule) -> mark as BAD ---
    seg_dict = actualizar_todo(seg_dict, limites_hr=limites_hr)
    tabla = seg_dict.get("tabla_beats", pd.DataFrame())
    n_ectopicos = 0
    ectopic_indices = []
    
    if tabla is not None and len(tabla) >= 3:
        rr = tabla["rr_ms"].astype(float).values
        ectopic_threshold_pct = 20.0
        
        for i in range(1, len(rr) - 1):
            rr_prev = rr[i - 1]
            rr_curr = rr[i]
            rr_next = rr[i + 1]
            
            if rr_prev > 0 and rr_next > 0:
                diff_prev_pct = abs(rr_curr - rr_prev) / rr_prev * 100
                diff_next_pct = abs(rr_curr - rr_next) / rr_next * 100
                
                if diff_prev_pct > ectopic_threshold_pct and diff_next_pct > ectopic_threshold_pct:
                    peak_row = int(tabla.loc[i, "peak_row"])
                    ectopic_indices.append(peak_row)
                    # Mark as BAD interval
                    t_ectopic = float(tabla.loc[i, "peak_time_s"])
                    t_prev = float(tabla.loc[i, "prev_peak_time_s"])
                    seg_dict = marcar_bad_interval(seg_dict, t_prev, t_ectopic,
                                                   motivo="ectópico", sev=1.0)
                    n_ectopicos += 1
    
    seg_dict["ectopic_indices"] = ectopic_indices
    
    if verbose:
        print(f"\nEctopic (>20%): {n_ectopicos} -> BAD")
    
    # Final update
    seg_dict = actualizar_todo(seg_dict, limites_hr=limites_hr)
    
    # --- AUTO-DELETE peaks in SEVERE zones ---
    bad_consolidated = consolidar_bad_intervals(seg_dict)
    severe_zones = bad_consolidated[bad_consolidated["severity"] >= 2]
    peaks_deleted = 0
    
    if len(severe_zones) > 0:
        for _, row in severe_zones.iterrows():
            t_ini = float(row["t_ini_s"])
            t_fin = float(row["t_fin_s"])
            n_before = len(seg_dict.get("ecg_peaks", pd.DataFrame()))
            seg_dict = borrar_peaks_por_intervalo(seg_dict, t_ini, t_fin, limites_hr=limites_hr)
            n_after = len(seg_dict.get("ecg_peaks", pd.DataFrame()))
            peaks_deleted += (n_before - n_after)
        
        seg_dict = actualizar_todo(seg_dict, limites_hr=limites_hr)
    
    if verbose and len(severe_zones) > 0:
        print(f"\n[SEVERE] SEVERE zones ({len(severe_zones)}): peaks auto-deleted = {peaks_deleted}")
    
    # Calculate final BAD percentage
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    
    cleaning_info = {
        "rr_short_isolated": peak_rows_short_isolated,
        "rr_long": peak_rows_long,
        "n_ectopicos": n_ectopicos,
        "ectopic_indices": ectopic_indices,
        "n_severe_zones": len(severe_zones),
        "peaks_deleted": peaks_deleted,
        "bad_pct": bad_pct,
        "bad_s": bad_s,
        "dur_s": dur_s,
    }
    
    return seg_dict, cleaning_info


# =============================================================================
# STAGE 2: CONSOLIDATION & QC
# =============================================================================

def limpiar_segmento_etapa2(
    seg_dict: dict,
    bad_pct_max: float = 0.20,
    min_clean_duration_s: Optional[float] = None,  # For LB/Recovery: require 290s clean
    limites_hr: Tuple[float, float] = (30, 200),
    verbose: bool = True,
) -> Tuple[dict, bool]:
    """
    Stage 2: Consolidate BAD, check threshold, validate duration.
    
    Args:
        bad_pct_max: Maximum allowed BAD percentage (default 20%)
        min_clean_duration_s: Minimum required CLEAN duration in seconds.
                              For LB/Recovery, set to 290.0 to require 290s of clean data.
                              If clean_s < min_clean_duration_s, segment is excluded.
    
    Returns: (seg_dict, qc_valid)
    """
    seg_name = seg_dict.get("seg", "SEG")
    print(f"\n{'='*60}")
    print(f"LIMPIEZA ETAPA 2: {seg_name}")
    print(f"{'='*60}")
    
    seg_dict = actualizar_todo(seg_dict, limites_hr=limites_hr)
    
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    clean_s = dur_s - bad_s if np.isfinite(bad_s) else dur_s
    
    # Check QC criteria
    qc_pct_ok = bool(np.isfinite(bad_pct) and (bad_pct <= bad_pct_max))
    qc_dur_ok = True
    
    if min_clean_duration_s is not None:
        # For subject 4 only: allow 0.1s tolerance
        # Check if SUJETO is defined (global variable)
        try:
            sujeto_actual = globals().get("SUJETO", None)
            if sujeto_actual == "4":
                tolerance = 0.1
                qc_dur_ok = clean_s >= (min_clean_duration_s - tolerance)
            else:
                qc_dur_ok = clean_s >= min_clean_duration_s
        except:
            # If SUJETO not available, use strict check
            qc_dur_ok = clean_s >= min_clean_duration_s
    
    qc_valid = qc_pct_ok and qc_dur_ok
    
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    seg_dict["clean_s"] = clean_s
    seg_dict["qc_valid"] = qc_valid
    
    if verbose:
        print(f"  Total duration: {dur_s:.1f}s")
        print(f"  BAD: {bad_s:.1f}s ({bad_pct*100:.1f}%) | threshold: {bad_pct_max*100:.0f}%")
        print(f"  CLEAN: {clean_s:.1f}s", end="")
        if min_clean_duration_s is not None:
            print(f" | minimum required: {min_clean_duration_s:.0f}s", end="")
        print()
        
        if not qc_pct_ok:
            print(f"  [!] EXCLUIDO: BAD > {bad_pct_max*100:.0f}%")
        elif not qc_dur_ok:
            # Check if it's within tolerance (0.1s) - only for subject 4
            try:
                sujeto_actual = globals().get("SUJETO", None)
                if sujeto_actual == "4":
                    tolerance = 0.1
                    if clean_s >= (min_clean_duration_s - tolerance):
                        print(f"  [!]  ACCEPTED (0.1s tolerance - Subject 4): CLEAN ({clean_s:.1f}s) is {min_clean_duration_s - clean_s:.1f}s from minimum ({min_clean_duration_s:.0f}s)")
                        qc_dur_ok = True  # Override to accept
                        qc_valid = qc_pct_ok and qc_dur_ok  # Recalculate
                        seg_dict["qc_valid"] = qc_valid  # Update in dict
                    else:
                        print(f"  [!] EXCLUDED: CLEAN ({clean_s:.1f}s) < minimum required ({min_clean_duration_s:.0f}s)")
                else:
                    print(f"  [!] EXCLUDED: CLEAN ({clean_s:.1f}s) < minimum required ({min_clean_duration_s:.0f}s)")
            except:
                print(f"  [!] EXCLUDED: CLEAN ({clean_s:.1f}s) < minimum required ({min_clean_duration_s:.0f}s)")
        else:
            print(f"  [OK] QC VÁLIDO")
    
    return seg_dict, qc_valid


def recortar_segmento(seg_dict: dict, target_duration_s: float = 300.0,
                      limites_hr: Tuple[float, float] = (30, 200),
                      verbose: bool = True) -> dict:
    """
    Trim segment to maximum target_duration_s by removing data from the END.
    Only trims if segment is LONGER than target_duration_s.
    
    For LB and Recovery:
        - If segment > 300s -> trim to 300s
        - If segment 290-300s -> keep as is
        - If segment < 290s -> excluded by min_clean_duration_s validation
    
    Args:
        target_duration_s: Maximum duration in seconds (default 300s)
    
    Returns: seg_dict (trimmed if > target, unchanged if -> target)
    """
    seg_name = seg_dict.get("seg", "SEG")
    fs = float(seg_dict.get("fs_ecg", 2000))
    
    # Current duration
    times = seg_dict.get("times", np.array([]))
    if len(times) == 0:
        if verbose:
            print(f"[Trim] {seg_name}: No data")
        return seg_dict
    
    current_dur = float(times[-1])
    
    if current_dur <= target_duration_s:
        if verbose:
            print(f"[Recortar] {seg_name}: Ya tiene {current_dur:.1f}s (-> {target_duration_s}s)")
        return seg_dict
    
    # Find index where time exceeds target
    target_idx = np.searchsorted(times, target_duration_s)
    
    if verbose:
        print(f"[Trim] {seg_name}: {current_dur:.1f}s -> {target_duration_s:.0f}s (removing last {current_dur - target_duration_s:.1f}s)")
    
    # Trim arrays
    seg_dict["times"] = times[:target_idx]
    seg_dict["ecg_raw"] = seg_dict.get("ecg_raw", np.array([]))[:target_idx]
    seg_dict["ecg_proc"] = seg_dict.get("ecg_proc", np.array([]))[:target_idx]
    seg_dict["hr_inst"] = seg_dict.get("hr_inst", np.array([]))[:target_idx]
    
    # Trim peaks
    ecg_peaks = seg_dict.get("ecg_peaks", pd.DataFrame())
    if len(ecg_peaks) > 0 and "peak_time" in ecg_peaks.columns:
        mask = ecg_peaks["peak_time"].astype(float) <= target_duration_s
        seg_dict["ecg_peaks"] = ecg_peaks[mask].reset_index(drop=True)
    
    # Update r_idx
    r_idx = seg_dict.get("r_idx", np.array([]))
    peak_times = seg_dict.get("ecg_peaks", pd.DataFrame())
    if len(peak_times) > 0:
        seg_dict["r_idx"] = peak_times["peak_index"].astype(int).values
    
    # Trim bad_intervals
    bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
    if len(bad_intervals) > 0:
        # Keep only intervals that start before target
        mask = bad_intervals["t_ini_s"].astype(float) < target_duration_s
        trimmed = bad_intervals[mask].copy()
        # Clip end times to target
        trimmed.loc[trimmed["t_fin_s"] > target_duration_s, "t_fin_s"] = target_duration_s
        seg_dict["bad_intervals"] = trimmed.reset_index(drop=True)
    
    # Update duration
    seg_dict["dur_s"] = target_duration_s
    
    # Recompute tabla_beats and HR
    seg_dict = actualizar_todo(seg_dict, limites_hr=limites_hr)
    
    # Recalculate BAD percentage
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    seg_dict["clean_s"] = dur_s - bad_s if np.isfinite(bad_s) else dur_s
    
    if verbose:
        n_peaks = len(seg_dict.get("ecg_peaks", []))
        print(f"  Peaks: {n_peaks} | BAD: {bad_pct*100:.1f}%")
    
    return seg_dict


def guardar_segmento(
    resultado: dict,
    resultados_dict: dict,
    seg_name: str,
    min_clean_duration_s: Optional[float] = None,
    target_duration_s: Optional[float] = None,
    bad_pct_max: float = 0.20,
    limites_hr: Tuple[float, float] = (30, 200),
) -> Tuple[dict, bool]:
    """
    Guardar segmento procesado: interpola BAD zones, valida QC, recorta si es necesario.
    
    Args:
        resultado: Diccionario del segmento procesado
        resultados_dict: Diccionario donde guardar (resultados_segmentos_limpios)
        seg_name: Nombre del segmento (ej: "LB", "BH 1")
        min_clean_duration_s: Mínimo de datos limpios requeridos (ej: 290 para LB/Recovery)
        target_duration_s: Duración objetivo para recortar (ej: 300 para LB/Recovery)
        bad_pct_max: Máximo porcentaje BAD permitido (default 20%)
        limites_hr: Límites de HR
        
    Returns: (resultado, qc_valid)
    
    Example:
        resultado_LB, qc_valid = guardar_segmento(
            resultado_LB, resultados_segmentos_limpios, "LB",
            min_clean_duration_s=290.0, target_duration_s=300.0
        )
    """
    # Print simplified header for the segment (will appear for all segments)
    print("\n" + "=" * 60)
    print(f"{seg_name}")
    print("=" * 60)

    # 1. Interpolar zonas BAD (function prints interpolation summary)
    resultado = interpolar_rr_en_bad_zones(resultado, verbose=True)

    # 2. Validar QC (do not print stage 2 verbose block here; we'll print a compact summary)
    resultado, qc_valid = limpiar_segmento_etapa2(
        resultado,
        bad_pct_max=bad_pct_max,
        min_clean_duration_s=min_clean_duration_s,
        limites_hr=limites_hr,
        verbose=False
    )

    if qc_valid:
        # 3. Recortar si es necesario (silencioso)
        if target_duration_s is not None:
            resultado = recortar_segmento(resultado, target_duration_s=target_duration_s, verbose=False)

        # 4. Guardar y mostrar resumen compacto
        resultados_dict[seg_name] = resultado
        dur_final = resultado["times"][-1] if len(resultado.get("times", [])) > 0 else 0
        n_peaks = len(resultado.get("r_idx", []))
        print("\n" + "-" * 45)
        print("GUARDADO")
        dur_total = resultado.get("dur_s", 0.0)
        bad_s = resultado.get("bad_s", 0.0)
        bad_pct = resultado.get("bad_pct", 0.0)
        clean_s = resultado.get("clean_s", dur_total)
        print(f"  Total duration: {dur_total:.1f}s")
        print(f"  BAD: {bad_s:.1f}s ({bad_pct*100:.1f}%) | threshold: {bad_pct_max*100:.0f}%")
        print(f"  CLEAN: {clean_s:.1f}s")
        print("  [OK] QC VALID")
        print(f"[OK] {seg_name} saved: {n_peaks} R-peaks, duration: {dur_final:.1f}s")
    else:
        print("\n" + "-" * 45)
        print("GUARDADO")
        print("  [!] QC FALLIDO")
        print(f"[!] {seg_name} excluido por QC")
    
    return resultado, qc_valid


# =============================================================================
# STAGE 3: ECTOPIC DETECTION, OUTLIER REMOVAL, INTERPOLATION
# =============================================================================

def detectar_ectopicos(seg_dict: dict, threshold_pct: float = 20.0, 
                       verbose: bool = True) -> Tuple[dict, List[int]]:
    """
    Detect ectopic beats using the >20% rule.
    An RR interval is considered ectopic if it differs >threshold_pct% from 
    both the previous AND next RR interval.
    
    Returns: (seg_dict with ectopic_indices marked, list of ectopic peak indices)
    """
    seg_name = seg_dict.get("seg", "SEG")
    tabla = seg_dict.get("tabla_beats", pd.DataFrame())
    
    if tabla is None or len(tabla) < 3:
        if verbose:
            print(f"[Ectopic] {seg_name}: Not enough beats to detect ectopic")
        return seg_dict, []
    
    rr = tabla["rr_ms"].astype(float).values
    ectopic_indices = []
    
    for i in range(1, len(rr) - 1):
        rr_prev = rr[i - 1]
        rr_curr = rr[i]
        rr_next = rr[i + 1]
        
        if rr_prev > 0 and rr_next > 0:
            diff_prev_pct = abs(rr_curr - rr_prev) / rr_prev * 100
            diff_next_pct = abs(rr_curr - rr_next) / rr_next * 100
            
            # Ectopic if differs >threshold% from BOTH neighbors
            if diff_prev_pct > threshold_pct and diff_next_pct > threshold_pct:
                peak_row = int(tabla.loc[i, "peak_row"])
                ectopic_indices.append(peak_row)
    
    seg_dict["ectopic_indices"] = ectopic_indices
    
    if verbose:
        print(f"\n[Ectópicos] {seg_name}: {len(ectopic_indices)} latidos ectópicos detectados (>{threshold_pct}%)")
        if len(ectopic_indices) > 0:
            # Show ectopic times
            ecg_peaks = seg_dict.get("ecg_peaks", pd.DataFrame())
            if len(ecg_peaks) > 0:
                for idx in ectopic_indices[:10]:  # Show first 10
                    if idx < len(ecg_peaks):
                        t = float(ecg_peaks.iloc[idx]["peak_time"])
                        print(f"    Peak {idx} at t={t:.2f}s")
                if len(ectopic_indices) > 10:
                    print(f"    ... y {len(ectopic_indices) - 10} más")
    
    return seg_dict, ectopic_indices


def detectar_outliers_rr_hr(seg_dict: dict, iqr_factor: float = 3.5,
                             hr_min: float = 30.0, hr_max: float = 200.0,
                             verbose: bool = True) -> Tuple[dict, dict]:
    """
    Detect outliers using IQR-based method and physiological HR limits.
    
    Uses nn_series (interpolated data) if available, otherwise tabla_beats.
    
    Returns: (seg_dict with outlier info, dict with outlier details)
    """
    seg_name = seg_dict.get("seg", "SEG")
    
    outliers = {
        "rr_iqr_low": [],
        "rr_iqr_high": [],
        "hr_too_low": [],
        "hr_too_high": [],
    }
    
    # Use nn_series (interpolated) if available, otherwise tabla_beats
    nn = seg_dict.get("nn_series", None)
    if nn is not None and "rr_ms" in nn:
        rr = np.array(nn["rr_ms"])
        rr_times = np.asarray(nn.get("times_s", np.arange(len(rr))), dtype=float)
        source = "nn_series"
    else:
        # Fallback to tabla_beats
        tabla = seg_dict.get("tabla_beats", None)
        if tabla is None or len(tabla) == 0:
            seg_dict = actualizar_tabla_beats(seg_dict)
            tabla = seg_dict.get("tabla_beats", pd.DataFrame())
        
        if tabla is None or len(tabla) < 4:
            if verbose:
                print(f"[Outliers] {seg_name}: Not enough data")
            return seg_dict, outliers
        
        rr = tabla["rr_ms"].astype(float).values
        rr_times = tabla["peak_time_s"].astype(float).values
        source = "tabla_beats (original)"
    
    if len(rr) < 4:
        if verbose:
                print(f"[Outliers] {seg_name}: Not enough data")
        return seg_dict, outliers
    
    hr = 60000.0 / rr  # Convert to BPM
    
    # IQR-based outliers for RR
    q1_rr = float(np.nanpercentile(rr, 25))
    q3_rr = float(np.nanpercentile(rr, 75))
    iqr_rr = q3_rr - q1_rr
    rr_low = q1_rr - iqr_factor * iqr_rr
    rr_high = q3_rr + iqr_factor * iqr_rr
    
    for i, val in enumerate(rr):
        if val < rr_low:
            t = float(rr_times[i]) if i < len(rr_times) else np.nan
            outliers["rr_iqr_low"].append((t, val))
        elif val > rr_high:
            t = float(rr_times[i]) if i < len(rr_times) else np.nan
            outliers["rr_iqr_high"].append((t, val))
    
    # Physiological HR limits
    for i, val in enumerate(hr):
        if val < hr_min:
            t = float(rr_times[i]) if i < len(rr_times) else np.nan
            outliers["hr_too_low"].append((t, val))
        elif val > hr_max:
            t = float(rr_times[i]) if i < len(rr_times) else np.nan
            outliers["hr_too_high"].append((t, val))
    
    seg_dict["outliers"] = outliers
    
    if verbose:
        total = sum(len(v) for v in outliers.values())
        print(f"\n[Outliers] {seg_name}: {total} outliers en {source}")
        print(f"  RR IQR: valid range = [{rr_low:.0f}, {rr_high:.0f}] ms")
        print(f"  RR < limit: {len(outliers['rr_iqr_low'])}")
        print(f"  RR > limit: {len(outliers['rr_iqr_high'])}")
        print(f"  HR < {hr_min} bpm: {len(outliers['hr_too_low'])}")
        print(f"  HR > {hr_max} bpm: {len(outliers['hr_too_high'])}")
    
    return seg_dict, outliers


def interpolar_rr_en_bad_zones(seg_dict: dict, method: str = "linear",
                                limites_hr: Tuple[float, float] = (30, 200),
                                verbose: bool = True) -> dict:
    """
    Interpolate RR intervals within BAD zones using linear interpolation.
    Creates a clean NN series suitable for HRV analysis.
    
    For segments WITHOUT BAD zones (clean segments), just copies the RR series.
    
    Args:
        method: 'linear' (conservative) or 'cubic' (smoother)
    
    Returns: seg_dict with 'nn_series' (interpolated RR times and values)
    """
    seg_name = seg_dict.get("seg", "SEG")
    
    # Ensure tabla_beats exists
    tabla = seg_dict.get("tabla_beats", None)
    if tabla is None or len(tabla) == 0:
        # Try to create it from ecg_peaks
        seg_dict = actualizar_tabla_beats(seg_dict)
        tabla = seg_dict.get("tabla_beats", pd.DataFrame())
    
    bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
    
    if tabla is None or len(tabla) == 0:
        if verbose:
            print(f"[Interpolation] {seg_name}: No data to interpolate")
        return seg_dict
    
    # Get RR times and values
    rr_times = tabla["peak_time_s"].astype(float).values
    rr_values = tabla["rr_ms"].astype(float).values
    
    # Create mask for BAD zones
    is_bad = np.zeros(len(rr_times), dtype=bool)
    
    if bad_intervals is not None and len(bad_intervals) > 0:
        for _, row in bad_intervals.iterrows():
            t_ini = float(row["t_ini_s"])
            t_fin = float(row["t_fin_s"])
            mask = (rr_times >= t_ini) & (rr_times <= t_fin)
            is_bad |= mask
    
    n_bad = int(np.sum(is_bad))
    n_good = len(rr_times) - n_bad
    
    if n_good < 2:
        if verbose:
            print(f"[Interpolación] {seg_name}: No hay suficientes datos buenos para interpolar")
        return seg_dict
    
    # Interpolate BAD values using good values
    rr_interp = rr_values.copy()
    
    if n_bad > 0:
        good_times = rr_times[~is_bad]
        good_values = rr_values[~is_bad]
        
        if method == "cubic" and len(good_times) >= 4 and _HAVE_SCIPY:
            f = interp1d(good_times, good_values, kind='cubic', 
                        bounds_error=False, fill_value='extrapolate')
            rr_interp[is_bad] = f(rr_times[is_bad])
        else:
            # Linear interpolation
            rr_interp[is_bad] = np.interp(rr_times[is_bad], good_times, good_values)
    
    # Store interpolated series
    seg_dict["nn_series"] = {
        "times_s": rr_times,
        "rr_ms": rr_interp,
        "is_interpolated": is_bad,
        "n_interpolated": n_bad,
        "n_total": len(rr_times),
    }
    
    if verbose:
        pct = n_bad / len(rr_times) * 100 if len(rr_times) > 0 else 0
        print("\nINTERPOLACION")
        print(f"{n_bad}/{len(rr_times)} valores interpolados ({pct:.1f}%)")
    
    return seg_dict


def verificar_limpieza(seg_dict: dict, 
                       hr_min: float = 30.0, 
                       hr_max: float = 200.0,
                       verbose: bool = True) -> dict:
    """
    Verify cleaning results and show any remaining problematic values.
    
    Returns a dict with:
      - 'ok': True if no problems remain
      - 'problemas': list of remaining issues
      - 'nn_series': the interpolated series if available
    """
    seg_name = seg_dict.get("seg", "SEG")
    result = {'ok': True, 'problemas': [], 'seg': seg_name}
    
    print(f"\n{'='*60}")
    print(f"VERIFICACION: {seg_name}")
    print(f"{'='*60}")
    
    # Check if nn_series exists (interpolated data)
    nn = seg_dict.get("nn_series", None)
    if nn is None:
        print("  [!]  No se ha ejecutado interpolar_rr_en_bad_zones()")
        print("     Verificando tabla_beats original...")
        dfr = seg_dict.get("tabla_beats", pd.DataFrame())
        if len(dfr) == 0:
            print("  [X] No hay datos")
            result['ok'] = False
            return result
        rr_values = dfr["rr_ms"].astype(float).values
        rr_times = dfr["peak_time_s"].astype(float).values
    else:
        rr_values = nn["rr_ms"]
        rr_times = nn["times_s"]
        print(f"  [OK] Serie interpolada: {nn['n_interpolated']}/{nn['n_total']} valores interpolados")
    
    # Calculate HR from RR
    hr_values = 60000.0 / rr_values
    
    # Check for problems
    problemas = []
    
    # 1. HR too low
    mask_low = hr_values < hr_min
    if np.any(mask_low):
        for i in np.where(mask_low)[0]:
            problemas.append({
                'tipo': 'HR_bajo',
                'tiempo_s': rr_times[i],
                'valor': f"HR={hr_values[i]:.1f}bpm (RR={rr_values[i]:.0f}ms)",
                'limite': f"< {hr_min} bpm"
            })
    
    # 2. HR too high  
    mask_high = hr_values > hr_max
    if np.any(mask_high):
        for i in np.where(mask_high)[0]:
            problemas.append({
                'tipo': 'HR_alto',
                'tiempo_s': rr_times[i],
                'valor': f"HR={hr_values[i]:.1f}bpm (RR={rr_values[i]:.0f}ms)",
                'limite': f"> {hr_max} bpm"
            })
    
    # 3. RR too short (< 300ms = > 200 bpm)
    mask_short = rr_values < 300
    if np.any(mask_short):
        for i in np.where(mask_short)[0]:
            if i not in np.where(mask_high)[0]:  # Don't duplicate
                problemas.append({
                    'tipo': 'RR_corto',
                    'tiempo_s': rr_times[i],
                    'valor': f"RR={rr_values[i]:.0f}ms",
                    'limite': "< 300ms"
                })
    
    # 4. RR too long (> 2500ms = < 24 bpm)
    mask_long = rr_values > 2500
    if np.any(mask_long):
        for i in np.where(mask_long)[0]:
            if i not in np.where(mask_low)[0]:  # Don't duplicate
                problemas.append({
                    'tipo': 'RR_largo',
                    'tiempo_s': rr_times[i],
                    'valor': f"RR={rr_values[i]:.0f}ms",
                    'limite': "> 2500ms"
                })
    
    result['problemas'] = problemas
    result['nn_series'] = nn
    
    if len(problemas) == 0:
        print(f"\n  [OK] ALL OK - No problematic values")
        print(f"     HR rango: {hr_values.min():.1f} - {hr_values.max():.1f} bpm")
        print(f"     RR rango: {rr_values.min():.0f} - {rr_values.max():.0f} ms")
        result['ok'] = True
    else:
        print(f"\n  [!]  {len(problemas)} PROBLEMATIC VALUES:")
        for p in problemas[:10]:  # Show max 10
            print(f"     -> t={p['tiempo_s']:.1f}s: {p['valor']} ({p['tipo']})")
        if len(problemas) > 10:
            print(f"     ... and {len(problemas)-10} more")
        
        print(f"\n  [TIP] OPCIONES:")
        print(f"     1. Revisar visualmente esas zonas con reinspeccionar_segmento()")
        print(f"     2. Marcar manualmente con: marcar_bad_interval(seg_dict, t_ini, t_fin, 'manual')")
        print(f"     3. Volver a interpolar: interpolar_rr_en_bad_zones(seg_dict)")
        result['ok'] = False
    
    return result


def validar_rangos_hr(seg_dict: dict, 
                      mean_hr_range: Tuple[float, float] = (35, 180),
                      max_hr_limit: float = 200,
                      min_hr_limit: float = 30,
                      std_hr_limit: float = 40,
                      verbose: bool = True) -> Tuple[bool, dict]:
    """
    Validate HR is within physiological ranges (like colleague's approach).
    
    Uses nn_series (interpolated data) if available, otherwise tabla_beats.
    
    Returns: (is_valid, validation_details)
    """
    seg_name = seg_dict.get("seg", "SEG")
    
    # Use nn_series (interpolated) if available, otherwise tabla_beats
    nn = seg_dict.get("nn_series", None)
    if nn is not None and "rr_ms" in nn:
        rr = np.array(nn["rr_ms"])
    else:
        # Fallback to tabla_beats
        tabla = seg_dict.get("tabla_beats", None)
        if tabla is None or len(tabla) == 0:
            seg_dict = actualizar_tabla_beats(seg_dict)
            tabla = seg_dict.get("tabla_beats", pd.DataFrame())
        
        if tabla is None or len(tabla) == 0:
            if verbose:
                print(f"[Validación HR] {seg_name}: No hay datos")
            return False, {"error": "No data"}
        
        rr = tabla["rr_ms"].astype(float).values
    
    if len(rr) == 0:
        if verbose:
            print(f"[Validación HR] {seg_name}: No hay datos")
        return False, {"error": "No data"}
    
    hr = 60000.0 / rr
    
    hr_mean = float(np.nanmean(hr))
    hr_max = float(np.nanmax(hr))
    hr_min = float(np.nanmin(hr))
    hr_std = float(np.nanstd(hr))
    
    validation = {
        "hr_mean": hr_mean,
        "hr_max": hr_max,
        "hr_min": hr_min,
        "hr_std": hr_std,
        "mean_ok": mean_hr_range[0] <= hr_mean <= mean_hr_range[1],
        "max_ok": hr_max <= max_hr_limit,
        "min_ok": hr_min >= min_hr_limit,
        "std_ok": hr_std <= std_hr_limit,
    }
    
    is_valid = all([validation["mean_ok"], validation["max_ok"], 
                    validation["min_ok"], validation["std_ok"]])
    
    if verbose:
        status = "[OK] OK" if is_valid else "[!] FUERA DE RANGO"
        print(f"\n[HR Validation] {seg_name}: {status}")
        print(f"  HR mean: {hr_mean:.1f} bpm (valid: {mean_hr_range[0]}-{mean_hr_range[1]})")
        print(f"  HR max:  {hr_max:.1f} bpm (limit: {max_hr_limit})")
        print(f"  HR min:  {hr_min:.1f} bpm (limit: {min_hr_limit})")
        print(f"  HR std:  {hr_std:.1f} bpm (limit: {std_hr_limit})")
    
    return is_valid, validation


def preparar_segmento_para_hrv(seg_dict: dict, 
                                resample_hz: float = 4.0,
                                verbose: bool = True) -> dict:
    """
    Prepare segment for HRV analysis:
    1. Use interpolated NN series
    2. Resample to regular grid (for frequency analysis) using cubic spline
    
    Returns: seg_dict with 'hrv_ready' data
    """
    seg_name = seg_dict.get("seg", "SEG")
    nn = seg_dict.get("nn_series", None)
    
    if nn is None:
        if verbose:
            print(f"[HRV Prep] {seg_name}: Primero ejecutar interpolar_rr_en_bad_zones()")
        return seg_dict
    
    times = nn["times_s"]
    rr_ms = nn["rr_ms"]
    
    if len(times) < 4:
        if verbose:
            print(f"[HRV Prep] {seg_name}: No hay suficientes datos")
        return seg_dict
    
    # Create regular time grid for frequency analysis
    t_start = times[0]
    t_end = times[-1]
    dt = 1.0 / resample_hz
    t_regular = np.arange(t_start, t_end, dt)
    
    # Cubic spline resampling for frequency analysis
    if not _HAVE_SCIPY:
        if verbose:
            print(f"[HRV Prep] scipy no disponible, usando interpolación lineal")
        rr_resampled = np.interp(t_regular, times, rr_ms)
    else:
        f_cubic = interp1d(times, rr_ms, kind='cubic', bounds_error=False, fill_value='extrapolate')
        rr_resampled = f_cubic(t_regular)
    
    seg_dict["hrv_ready"] = {
        # Original (linear interpolated) for time-domain
        "times_s": times,
        "rr_ms": rr_ms,
        "n_beats": len(times),
        # Resampled (cubic) for frequency-domain
        "times_regular_s": t_regular,
        "rr_resampled_ms": rr_resampled,
        "fs_resample": resample_hz,
    }
    
    if verbose:
        dur = t_end - t_start
        print(f"\n[HRV Prep] {seg_name}:")
        print(f"  Duración: {dur:.1f}s | Beats: {len(times)}")
        print(f"  Resampled: {len(t_regular)} puntos @ {resample_hz} Hz")
    
    return seg_dict


def exportar_segmento_excel(seg_dict: dict, output_path: str, verbose: bool = True) -> bool:
    """
    Export segment data to Excel for HRV analysis.
    Creates columns needed for all HRV parameters.
    """
    seg_name = seg_dict.get("seg", "SEG")
    nn = seg_dict.get("nn_series", None)
    hrv_ready = seg_dict.get("hrv_ready", None)
    
    if nn is None:
        if verbose:
            print(f"[Export] {seg_name}: No hay datos interpolados")
        return False
    
    # Create DataFrame for export
    df_export = pd.DataFrame({
        "beat_idx": np.arange(len(nn["times_s"])),
        "time_s": nn["times_s"],
        "rr_ms": nn["rr_ms"],
        "is_interpolated": nn["is_interpolated"].astype(int),
    })
    
    # Add HR
    df_export["hr_bpm"] = 60000.0 / df_export["rr_ms"]
    
    # Add segment info as metadata columns
    df_export["segment"] = seg_name
    df_export["fs_ecg"] = seg_dict.get("fs_ecg", 2000)
    df_export["duration_s"] = seg_dict.get("dur_s", 0)
    df_export["bad_pct"] = seg_dict.get("bad_pct", 0)
    df_export["qc_valid"] = int(seg_dict.get("qc_valid", True))
    
    # Save
    df_export.to_excel(output_path, index=False, sheet_name=seg_name[:31])
    
    if verbose:
        print(f"[Export] {seg_name}: Guardado en {output_path}")
        print(f"  {len(df_export)} filas, columnas: {list(df_export.columns)}")
    
    return True


def exportar_sujeto_completo(resultados: dict, sujeto_id: str, 
                              output_folder: str, verbose: bool = True) -> str:
    """
    Export all segments for one subject to a single Excel file.
    Each segment is a separate sheet.
    
    Args:
        resultados: dict with segment names as keys, seg_dict as values
        sujeto_id: Subject ID
        output_folder: Folder to save the Excel file
    
    Returns: Path to saved file
    """
    import os
    
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"HRV_data_sujeto_{sujeto_id}.xlsx")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_rows = []
        
        for seg_name, seg_dict in resultados.items():
            nn = seg_dict.get("nn_series", None)
            
            if nn is None:
                if verbose:
                    print(f"[Export] {seg_name}: Saltado (sin datos interpolados)")
                continue
            
            # Create DataFrame for this segment
            df_seg = pd.DataFrame({
                "beat_idx": np.arange(len(nn["times_s"])),
                "time_s": nn["times_s"],
                "rr_ms": nn["rr_ms"],
                "is_interpolated": nn["is_interpolated"].astype(int),
                "hr_bpm": 60000.0 / nn["rr_ms"],
            })
            
            # Sheet name (max 31 chars)
            sheet_name = seg_name[:31].replace("/", "_").replace("\\", "_")
            df_seg.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Add to summary
            summary_rows.append({
                "segment": seg_name,
                "n_beats": len(nn["times_s"]),
                "duration_s": seg_dict.get("dur_s", 0),
                "bad_pct": seg_dict.get("bad_pct", 0) * 100,
                "n_interpolated": nn.get("n_interpolated", 0),
                "qc_valid": seg_dict.get("qc_valid", True),
                "hr_mean": float(np.nanmean(60000.0 / nn["rr_ms"])),
                "rr_mean_ms": float(np.nanmean(nn["rr_ms"])),
            })
        
        # Write summary sheet
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, sheet_name="RESUMEN", index=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EXPORTED: {output_path}")
        print(f"Segments: {len(summary_rows)}")
        print(f"{'='*60}")
    
    return output_path


# =============================================================================
# FUNCIONES PARA PROCESAMIENTO FINAL (INTERPOLACION + OUTLIERS + EXPORT)
# =============================================================================

def interpolar_todos_segmentos(resultados: dict, method: str = "linear", 
                                resample_hz: float = 4.0, verbose: bool = True) -> dict:
    """
    Interpola zonas BAD de todos los segmentos y prepara para HRV.
    
    Args:
        resultados: dict con segmentos limpios
        method: método de interpolación ("linear", "cubic")
        resample_hz: frecuencia de resampleo para HRV
        verbose: mostrar progreso
    
    Returns:
        dict actualizado con nn_series
    """
    if verbose:
        print("\n" + "="*60)
        print("INTERPOLACION")
        print("="*60)
    
    for seg_name, seg_dict in resultados.items():
        seg_dict = actualizar_tabla_beats(seg_dict)
        seg_dict = interpolar_rr_en_bad_zones(seg_dict, method=method, verbose=False)
        seg_dict = preparar_segmento_para_hrv(seg_dict, resample_hz=resample_hz, verbose=False)
        resultados[seg_name] = seg_dict
        
        if verbose:
            nn = seg_dict.get("nn_series", {})
            n_interp = nn.get("n_interpolated", 0)
            n_total = nn.get("n_total", 0)
            pct = (n_interp / n_total * 100) if n_total > 0 else 0
            print(f"{seg_name}: {n_interp}/{n_total} valores interpolados ({pct:.1f}%)")
    
    return resultados


def verificar_outliers_todos(resultados: dict, iqr_factor: float = 3.5,
                              hr_min: float = 30.0, hr_max: float = 200.0,
                              mean_hr_range: Tuple[float, float] = (35, 180),
                              std_hr_limit: float = 40.0,
                              verbose: bool = True) -> Tuple[dict, bool]:
    """
    Verifica outliers en todos los segmentos (sobre serie interpolada).
    
    Args:
        resultados: dict con segmentos interpolados
        iqr_factor: factor IQR para detectar outliers
        hr_min, hr_max: límites de HR instantáneo
        mean_hr_range: rango válido para HR media
        std_hr_limit: límite para desviación estándar de HR
        verbose: mostrar progreso
    
    Returns:
        (resultados actualizados, hay_problemas: bool)
    """
    if verbose:
        print("\n" + "="*60)
        print("VERIFICACION DE OUTLIERS")
        print("="*60)
    
    hay_problemas = False
    segmentos_con_problemas = []
    
    for seg_name, seg_dict in resultados.items():
        seg_dict, outliers = detectar_outliers_rr_hr(seg_dict, iqr_factor=iqr_factor, 
                                                      hr_min=hr_min, hr_max=hr_max, verbose=False)
        hr_valid, hr_info = validar_rangos_hr(seg_dict, mean_hr_range=mean_hr_range,
                                               max_hr_limit=hr_max, min_hr_limit=hr_min,
                                               std_hr_limit=std_hr_limit, verbose=False)
        resultados[seg_name] = seg_dict
        
        # Contar outliers
        total_outliers = sum(len(v) for v in outliers.values())
        
        if verbose:
            if hr_valid and total_outliers == 0:
                print(f"{seg_name}: [OK] OK")
            else:
                print(f"\n{seg_name}: outliers a comprobar:")
                
                # nn_series (RR)
                nn = seg_dict.get("nn_series", {})
                if "rr_ms" in nn:
                    rr = np.array(nn["rr_ms"])
                    q1_rr = float(np.nanpercentile(rr, 25))
                    q3_rr = float(np.nanpercentile(rr, 75))
                    iqr_rr = q3_rr - q1_rr
                    rr_low = q1_rr - iqr_factor * iqr_rr
                    rr_high = q3_rr + iqr_factor * iqr_rr
                    
                    print(f"  nn_series:")
                    print(f"  RR IQR: rango válido = [{rr_low:.0f}, {rr_high:.0f}] ms")
                    print(f"  RR < límite: {len(outliers.get('rr_iqr_low', []))}")
                    print(f"  RR > límite: {len(outliers.get('rr_iqr_high', []))}")
                
                # HR stats
                print(f"\n  HR:")
                print(f"  HR < {hr_min:.0f} bpm: {len(outliers.get('hr_too_low', []))}")
                print(f"  HR > {hr_max:.0f} bpm: {len(outliers.get('hr_too_high', []))}")
                print(f"  HR mean: {hr_info.get('hr_mean', 0):.1f} bpm (válido: {mean_hr_range[0]}-{mean_hr_range[1]})")
                print(f"  HR max:  {hr_info.get('hr_max', 0):.1f} bpm (límite: {hr_max})")
                print(f"  HR min:  {hr_info.get('hr_min', 0):.1f} bpm (límite: {hr_min})")
                print(f"  HR std:  {hr_info.get('hr_std', 0):.1f} bpm (límite: {std_hr_limit})")

                def describe_outliers(detail_label, entries, value_label):
                    if not entries:
                        return
                    preview = entries[:5]
                    summary = ", ".join(f"{t:.1f}s ({val:.0f}{value_label})" for t, val in preview)
                    if len(entries) > 5:
                        summary += f", ... (+{len(entries)-5} más)"
                    print(f"    {detail_label}: {summary}")

                describe_outliers("rr_iqr_low", outliers.get("rr_iqr_low", []), " ms")
                describe_outliers("rr_iqr_high", outliers.get("rr_iqr_high", []), " ms")
                describe_outliers("hr_too_low", outliers.get("hr_too_low", []), " bpm")
                describe_outliers("hr_too_high", outliers.get("hr_too_high", []), " bpm")
                
                segmentos_con_problemas.append(seg_name)
                hay_problemas = True
    
    # Resumen final
    if verbose:
        print(f"\n{'-'*60}")
        if hay_problemas:
            print(f"[!]  {len(segmentos_con_problemas)} SEGMENTO(S) A REVISAR: {', '.join(segmentos_con_problemas)}")
        else:
            print("[OK] TODOS LOS SEGMENTOS OK")
        print(f"{'-'*60}")
    
    return resultados, hay_problemas


def exportar_y_resumir(resultados: dict, sujeto_id: str, 
                        output_folder: str, archivo_resumen: str) -> str:
    """
    Exporta a Excel, muestra resumen y guarda resumen acumulativo.
    
    Args:
        resultados: dict con segmentos procesados
        sujeto_id: ID del sujeto
        output_folder: carpeta para Excel
        archivo_resumen: ruta al CSV acumulativo
    
    Returns:
        Path al archivo Excel generado
    """
    import os
    
    # --- 1. Exportar a Excel ---
    os.makedirs(output_folder, exist_ok=True)
    output_file = exportar_sujeto_completo(
        resultados=resultados,
        sujeto_id=sujeto_id,
        output_folder=output_folder,
        verbose=True
    )
    
    # --- 2. Resumen del sujeto ---
    print("\n" + "="*70)
    print(f"FINAL SUMMARY - SUBJECT {sujeto_id}")
    print("="*70)
    print(f"\n{'Segment':<15} {'Beats':>8} {'Duration':>10} {'BAD%':>8} {'Interp.':>8} {'HR mean':>10}")
    print("-"*70)
    
    resumen_filas = []
    for seg_name, seg_dict in resultados.items():
        nn = seg_dict.get("nn_series", {})
        n_beats = nn.get("n_total", 0)
        n_interp = nn.get("n_interpolated", 0)
        dur = seg_dict.get("dur_s", 0)
        bad_pct = seg_dict.get("bad_pct", 0) * 100
        clean_s = dur * (1 - seg_dict.get("bad_pct", 0))
        rr_ms = nn.get("rr_ms", [])
        hr_mean = float(np.nanmean(60000.0 / np.array(rr_ms))) if len(rr_ms) > 0 else 0
        
        print(f"{seg_name:<15} {n_beats:>8} {dur:>10.1f}s {bad_pct:>7.1f}% {n_interp:>8} {hr_mean:>9.1f}")
        resumen_filas.append({
            "sujeto": sujeto_id, "segmento": seg_name, "beats_total": n_beats,
            "beats_interpolados": n_interp, "duracion_s": round(dur, 1),
            "clean_s": round(clean_s, 1), "bad_pct": round(bad_pct, 2), 
            "hr_mean": round(hr_mean, 1)
        })
    
    print("-"*70)
    print(f"Total segmentos: {len(resultados)}")
    
    # --- 3. Guardar resumen acumulativo ---
    fila_sujeto = {"sujeto": sujeto_id}
    for seg_info in resumen_filas:
        seg = seg_info["segmento"].replace(" ", "")
        fila_sujeto[f"{seg}_beats"] = seg_info["beats_total"]
        fila_sujeto[f"{seg}_interp"] = seg_info["beats_interpolados"]
        fila_sujeto[f"{seg}_duracion_s"] = seg_info["duracion_s"]
        fila_sujeto[f"{seg}_clean_s"] = seg_info["clean_s"]
        fila_sujeto[f"{seg}_bad_pct"] = seg_info["bad_pct"]
        fila_sujeto[f"{seg}_hr_mean"] = seg_info["hr_mean"]
    
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
    print(f"\n[OK] Excel: {output_file}")
    print(f"[OK] Resumen acumulativo: {archivo_resumen}")
    print(f"  Sujetos: {df_final['sujeto'].tolist()}")
    
    return output_file


def consolidar_bad_intervals(seg_dict: dict, merge_gap_s: float = 0.0) -> pd.DataFrame:
    """
    Consolidate BAD intervals: merge only truly overlapping, compute severity, assign IDs.
    
    Args:
        merge_gap_s: Only merge if gap between intervals is <= this value.
                     Default 0.0 = only merge if truly overlapping (not just adjacent).
    
    Severity: 1 = single BAD, 2+ = overlapping BAD zones (SEVERE).
    Returns DataFrame with columns: id, t_ini_s, t_fin_s, severity, motivos
    """
    bi = seg_dict.get("bad_intervals", pd.DataFrame())
    if bi is None or len(bi) == 0:
        return pd.DataFrame(columns=["id", "t_ini_s", "t_fin_s", "severity", "motivos", "dur_s"])
    
    dur_s = len(seg_dict.get("ecg_raw", [])) / float(seg_dict.get("fs_ecg", 1000))
    
    # Convert to list of (start, end, motivo)
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
    
    # Sort by start time
    intervals = sorted(intervals, key=lambda x: x[0])
    
    # Merge only truly overlapping intervals (not adjacent)
    merged = []
    cur_start, cur_end, cur_motivos = intervals[0][0], intervals[0][1], [intervals[0][2]]
    cur_severity = 1
    
    for s, e, m in intervals[1:]:
        # Only merge if truly overlapping (s < cur_end, not s <= cur_end)
        if s < cur_end + merge_gap_s:  # Truly overlapping
            cur_end = max(cur_end, e)
            cur_severity += 1  # More overlaps = higher severity
            if m not in cur_motivos:
                cur_motivos.append(m)
        else:
            # Not overlapping - save current and start new
            merged.append((cur_start, cur_end, cur_severity, cur_motivos))
            cur_start, cur_end, cur_motivos = s, e, [m]
            cur_severity = 1
    
    merged.append((cur_start, cur_end, cur_severity, cur_motivos))
    
    # Create DataFrame with IDs
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


def reinspeccionar_segmento(seg_dict: dict, win_len: float = 60.0,
                            min_clean_duration_s: Optional[float] = None,
                            mostrar_todas_detecciones: bool = True) -> pd.DataFrame:
    """
    Re-inspect segment after cleaning with numbered BAD zones.
    Shows segment status based on duration requirements.
    
    Args:
        mostrar_todas_detecciones: If True, also shows ALL detections from 
            detectar_problemas_etapa1 (even if not applied), for comparison.
    
    Returns consolidated BAD intervals DataFrame for reference.
    """
    if plt is None:
        print("matplotlib no disponible.")
        return pd.DataFrame()
    
    seg_name = seg_dict.get("seg", "SEG")
    # Get ecg_raw and ecg_proc - ensure we use the original raw signal
    # ecg_raw should be the original unprocessed signal from initial inspection
    ecg_raw = seg_dict.get("ecg_raw", np.array([]))
    if len(ecg_raw) == 0:
        raise ValueError(f"[reinspeccionar_segmento] {seg_name}: ecg_raw no está disponible en seg_dict")
    # Make a copy to ensure we're using the original raw signal (not a reference)
    ecg_raw = ecg_raw.copy()
    
    # ecg_proc is the processed signal (may be the same as raw if no processing was done)
    ecg_proc = seg_dict.get("ecg_proc", ecg_raw.copy())
    if len(ecg_proc) == 0:
        ecg_proc = ecg_raw.copy()
    
    times = seg_dict.get("times", np.array([]))
    r_idx = seg_dict.get("r_idx", np.array([]))
    hr_inst = seg_dict.get("hr_inst", np.zeros_like(times))
    
    if len(times) == 0:
        print("No data to re-inspect.")
        return pd.DataFrame()
    
    # Consolidate BAD intervals with IDs and severity
    bad_consolidated = consolidar_bad_intervals(seg_dict)
    
    dur_seg = float(times[-1]) if len(times) else 0.0
    n_windows = int(math.ceil(dur_seg / win_len)) if dur_seg > 0 else 0
    
    # Calculate BAD stats and STORE for later RESUMEN FINAL
    bad_pct, bad_s, total_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    clean_s = total_s - bad_s if np.isfinite(bad_s) else total_s
    
    # Determine segment status
    if min_clean_duration_s is not None:
        segment_ok = clean_s >= min_clean_duration_s
        if segment_ok:
            status_str = f"  ->  [OK] OK (clean >= {min_clean_duration_s:.0f}s)"
        else:
            status_str = f"  ->  [X] SEGMENT EXCLUDED (need {min_clean_duration_s:.0f}s clean)"
    else:
        segment_ok = bad_pct <= 0.20
        status_str = "  ->  [OK] OK" if segment_ok else "  ->  [X] SEGMENT EXCLUDED (BAD > 20%)"
    
    print(f"\n{'='*70}")
    print(f"RE-INSPECTION: {seg_name}")
    print(f"{'='*70}")
    print(f"Duración total: {total_s:.1f}s | BAD: {bad_s:.1f}s ({bad_pct*100:.1f}%) | CLEAN: {clean_s:.1f}s{status_str}")
    print(f"Peaks: {len(r_idx)}")
    
    # Show ALL detections (even if not applied)
    detections = seg_dict.get('detections', {})
    if mostrar_todas_detecciones and detections:
        print(f"\n--- TODAS LAS DETECCIONES (referencia) ---")
        if detections.get('clusters'):
            print(f"  Clusters RR cortos: {len(detections['clusters'])}")
            for i, (t0d, t1d, peaks) in enumerate(detections['clusters']):
                print(f"    -> [{t0d:.1f}s - {t1d:.1f}s] ({len(peaks)} peaks)")
        if detections.get('short_isolated'):
            print(f"  RR cortos aislados: {len(detections['short_isolated'])}")
        if detections.get('long_rr'):
            print(f"  RR largos: {len(detections['long_rr'])}")
        if detections.get('short_long_pattern'):
            print(f"  Patrón corto-largo: {len(detections['short_long_pattern'])}")
        if detections.get('cv_instability'):
            print(f"  CV instability (NO APLICADO): {len(detections['cv_instability'])}")
            for i, (t0d, t1d, cv) in enumerate(detections['cv_instability']):
                print(f"    -> [{t0d:.1f}s - {t1d:.1f}s] CV={cv:.2f}")
        if detections.get('amplitude'):
            print(f"  Anomalous amplitude (NOT APPLIED): {len(detections['amplitude'])}")
            for i, (t0d, t1d, z) in enumerate(detections['amplitude']):
                print(f"    -> [{t0d:.1f}s - {t1d:.1f}s] z={z:.1f}")
        if detections.get('ectopic'):
            print(f"  Ectópicos: {len(detections['ectopic'])}")
    
    if len(bad_consolidated) > 0:
        print(f"\n--- ZONAS BAD APLICADAS ---")
        for _, row in bad_consolidated.iterrows():
            print(f"  [{row['id']}] {row['t_ini_s']:.1f}s - {row['t_fin_s']:.1f}s ({row['dur_s']:.1f}s) | {row['motivos']}")
    else:
        print("\n[OK] No hay zonas BAD aplicadas")
    
    for k in range(n_windows):
        t0 = k * win_len
        win = min(win_len, dur_seg - t0)
        if win <= 0:
            break
        t_end = t0 + win
        print(f"\nWindow {k+1}/{n_windows}: [{t0:.1f}s - {t_end:.1f}s]")
        plot_ecg_window_with_bad(ecg_raw, ecg_proc, times, r_idx, hr_inst, 
                                  bad_consolidated, t0=t0, win_len=win, seg_actual=seg_name)
    
    return bad_consolidated


def plot_ecg_window_with_bad(ecg_raw_seg, ecg_proc_seg, times_seg,
                              r_idx_seg, hr_inst_seg, bad_consolidated,
                              t0, win_len=60.0, seg_actual="SEG"):
    """Plot ECG window with numbered BAD zones (all red)."""
    if plt is None:
        return
    
    mask = (times_seg >= t0) & (times_seg < t0 + win_len)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print(f"Window [{t0:.1f},{t0+win_len:.1f}] s empty")
        return

    i0, i1 = idx[0], idx[-1] + 1
    
    # Use REAL segment times (not starting from 0)
    t_win = times_seg[i0:i1]
    t_end = t0 + win_len
    raw_win = ecg_raw_seg[i0:i1]
    proc_win = ecg_proc_seg[i0:i1]
    hr_win = hr_inst_seg[i0:i1] if hr_inst_seg is not None else np.full_like(t_win, np.nan)

    # R-peaks positions in real time
    r_mask = (r_idx_seg >= i0) & (r_idx_seg < i1)
    r_idx_win = r_idx_seg[r_mask]
    r_times_win = times_seg[r_idx_win] if len(r_idx_win) > 0 else np.array([])

    fig, axs = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"{seg_actual} - [{t0:.1f}s - {t_end:.1f}s]")

    lw = 0.6
    bad_color = "#ff3333"      # Red for all BAD zones
    bad_alpha = 0.35

    # Helper to shade BAD zones with numbers (uses REAL segment times)
    def shade_bad_numbered(ax, bad_df, t0, t_end, show_labels=True):
        if bad_df is None or len(bad_df) == 0:
            return
        for _, row in bad_df.iterrows():
            b_ini = float(row["t_ini_s"])
            b_fin = float(row["t_fin_s"])
            # Clip to window (in real time)
            if b_fin < t0 or b_ini > t_end:
                continue
            b_ini_clipped = max(t0, b_ini)
            b_fin_clipped = min(t_end, b_fin)
            
            ax.axvspan(b_ini_clipped, b_fin_clipped, color=bad_color, alpha=bad_alpha)
            
            # Add ID label at center of zone
            if show_labels:
                center = (b_ini_clipped + b_fin_clipped) / 2
                y_pos = ax.get_ylim()[1] * 0.95
                ax.text(center, y_pos, f"[{row['id']}]", 
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       color='black', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # Plot 1: ECG raw
    ax = axs[0]
    ax.plot(t_win, raw_win, lw=lw, color="#1f77b4")
    if len(r_times_win) > 0:
        ax.scatter(r_times_win, raw_win[r_idx_win - i0], s=15, c="red", zorder=5, label="R peaks")
    shade_bad_numbered(ax, bad_consolidated, t0, t_end, show_labels=True)
    ax.set_ylabel("ECG raw")
    ax.legend(loc="upper right")

    # Plot 2: ECG processed
    ax = axs[1]
    ax.plot(t_win, proc_win, lw=lw, color="#2ca02c")
    if len(r_times_win) > 0:
        ax.scatter(r_times_win, proc_win[r_idx_win - i0], s=15, c="red", zorder=5, label="R peaks")
    shade_bad_numbered(ax, bad_consolidated, t0, t_end, show_labels=False)
    ax.set_ylabel("ECG proc")
    ax.legend(loc="upper right")

    # Plot 3: HR
    ax = axs[2]
    ax.plot(t_win, hr_win, lw=lw, color="#9467bd")
    shade_bad_numbered(ax, bad_consolidated, t0, t_end, show_labels=False)
    ax.set_ylabel("HR (bpm)")
    ax.set_xlabel("Tiempo en segmento (s)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


# =============================================================================
# UNIFIED SAVE FUNCTION
# =============================================================================

def guardar_salida_segmento(
    subject_folder: str,
    sujeto: str,
    seg_name: str,
    resultado: dict,
):
    """Save NN/RR series from cleaned result; keep minimal outputs for HRV."""
    folder = _to_path(subject_folder)
    folder.mkdir(parents=True, exist_ok=True)

    # Check if segment was accepted
    if not resultado.get("qc_valid", True):
        print(f"Segmento {seg_name}: EXCLUIDO (no se guarda).")
        # Still save QC entry
        qc_entry = {
            "sujeto": sujeto,
            "segmento": seg_name,
            "qc_valid": False,
            "bad_pct": float(resultado.get("bad_pct", np.nan)),
        }
        qc_file = folder / "qc_summary.json"
        existing: List[dict] = []
        if qc_file.exists():
            try:
                existing = json.loads(qc_file.read_text())
            except json.JSONDecodeError:
                existing = []
        existing = [x for x in existing if not (x.get("sujeto") == sujeto and x.get("segmento") == seg_name)]
        existing.append(qc_entry)
        qc_file.write_text(json.dumps(existing, indent=2))
        return qc_entry

    # Extract RR intervals from peak indices
    r_idx = resultado.get("r_idx", np.array([]))
    fs_ecg = resultado.get("fs_ecg", 1000.0)
    
    if len(r_idx) < 2:
        print(f"Segment {seg_name}: not enough R-peaks to save.")
        return None
    
    # Compute RR intervals (in ms)
    rr_ms = np.diff(r_idx) / fs_ecg * 1000.0
    rr_times_s = r_idx[:-1] / fs_ecg
    
    df = pd.DataFrame({
        "time_s": rr_times_s,
        "rr_ms": rr_ms,
    })
    out_csv = folder / f"{sujeto}_{seg_name}_rr.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Guardado RR: {out_csv}")

    # QC summary
    bad_pct = resultado.get("bad_pct", 0.0)
    qc_entry = {
        "sujeto": sujeto,
        "segmento": seg_name,
        "qc_valid": True,
        "bad_pct": float(bad_pct) if np.isfinite(bad_pct) else 0.0,
        "n_peaks": int(len(r_idx)),
        "dur_s": float(resultado.get("dur_s", 0)),
        "rr_mean_ms": float(np.mean(rr_ms)),
        "rr_std_ms": float(np.std(rr_ms)),
        "hr_mean_bpm": float(60000.0 / np.mean(rr_ms)) if np.mean(rr_ms) > 0 else 0.0,
    }
    qc_file = folder / "qc_summary.json"
    existing: List[dict] = []
    if qc_file.exists():
        try:
            existing = json.loads(qc_file.read_text())
        except json.JSONDecodeError:
            existing = []
    existing = [x for x in existing if not (x.get("sujeto") == sujeto and x.get("segmento") == seg_name)]
    existing.append(qc_entry)
    qc_file.write_text(json.dumps(existing, indent=2))
    return qc_entry
