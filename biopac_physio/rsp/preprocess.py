# Part of biopac_physio — https://github.com/MaragdaPP/biopac-physio-preprocessing
# Library only: the per-subject workflow lives in studies/<study>/rsp_clean.py

# %% Imports & configuration
"""
Standalone RSP (Respiration) preprocessing helper for BIOPAC .acq files.
Run cell-by-cell in VSCode/Cursor.
Required: neurokit2, bioread, matplotlib, pandas, numpy, scipy

Based on:
- AASM respiratory signal standards
- Khodadad et al. (2018) - Breathing detection algorithm
- Recent RRV literature (2024-2026)
- BIOPAC respiration belt processing guidelines

Key features:
- Bandpass filtering (0.05-1 Hz default, adaptive option)
- 50 Hz sampling for RSA synchronization capability
- Inspiration/expiration peak detection
- Comprehensive artifact detection (flat, clipping, steps, baseline wander, tremor)
- Interpolation for short artifacts (1-2 breathing cycles)
- Manual bad zone management (add, remove, modify)
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
import subprocess
import sys
import tempfile

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
    from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator
    from scipy.ndimage import uniform_filter1d
    from scipy.stats import zscore
    _HAVE_SCIPY = True
except ImportError:
    scipy_signal = None
    interp1d = None
    CubicSpline = None
    PchipInterpolator = None
    _HAVE_SCIPY = False

# ---- Configuration defaults ----
# Sampling
TARGET_FS_HZ = 63.0  # 63 Hz for RSA synchronization (50-100 Hz recommended)

# Filtering (bandpass)
# Conservative: 0.05-1 Hz captures 3-60 breaths/min
# For typical breathing (12-20 bpm = 0.2-0.33 Hz), this is safe
LOWPASS_CUTOFF_HZ = 1.0    # Upper cutoff
HIGHPASS_CUTOFF_HZ = 0.05  # Lower cutoff (removes slow drift)
FIR_NUMTAPS = 4            # FIR numtaps (for firwin)

# Breathing rate expectations
MIN_BREATHS_PER_MIN = 4.0   # Very slow breathing (0.067 Hz)
MAX_BREATHS_PER_MIN = 60.0  # Very fast breathing (1.0 Hz)
TYPICAL_BREATH_DURATION_S = 4.0  # ~15 breaths/min

# Artifact detection thresholds (PERMISSIVE by default - manual review recommended)
# Using MAD (Median Absolute Deviation) for robustness against outliers
FLAT_LINE_THRESHOLD_S = 5.0      # Min duration for flat line (5s is very long)
FLAT_LINE_STD_THRESHOLD = 0.001  # Very strict - only truly stuck signals
CLIPPING_PERCENTILE = 0.1        # Only extreme 0.1% considered clipping

# MAD-based detection thresholds (robust to outliers)
# Literature standard: 3×MAD is typical threshold; we use higher values (7-8×MAD) to be more permissive
# and only flag clear, significant artifacts (not normal variations)
# MAD_TO_STD = 1.4826 converts MAD to equivalent std for normal distributions
MAD_TO_STD = 1.4826
STEP_MAD_THRESHOLD = 10.0        # MAD units (literature: 3×MAD typical; 10×MAD = extremely permissive, only very extreme spikes)
BASELINE_SHIFT_MAD = 9.0         # MAD units (literature: 3×MAD typical; 9×MAD = very permissive, only major shifts)
BASELINE_SHIFT_SHORT_WIN_S = 2.0 # Short window for rolling mean (2s)
BASELINE_SHIFT_LONG_WIN_S = 10.0 # Long window for rolling mean (10s)
STEP_ARTIFACT_MARGIN_S = 0.05    # Margin around detected step (very narrow - only mark the specific peak, not surrounding area)
MIN_ARTIFACT_DURATION_S = 0.2    # Minimum artifact duration to flag (very short - allow marking single peaks)
BASELINE_SHIFT_MIN_DURATION_S = 2.0  # Minimum baseline shift duration to flag (only flag long, sustained shifts)
BASELINE_SHIFT_MARGIN_S = 0.05   # Margin around baseline shift (very narrow - only mark the specific shift point)
ARTIFACT_MERGE_MARGIN_S = 0.0    # Margin for merging intervals (0.0 = don't merge, keep each peak separate)

# Absolute thresholds (alternative to MAD).
# Used by `detect_step_artifacts` and `detect_baseline_shifts` to guard against clearly unrealistic derivatives.
STEP_ABS_THRESHOLD = 0.5         # Absolute derivative threshold
# Baseline shift guard for sudden drifts that exceed physiologic bounds.
BASELINE_SHIFT_ABS = 0.4         # Absolute baseline shift threshold

# Legacy (kept for compatibility)
STEP_THRESHOLD_STD = 6.0         # 6 STDs - very permissive
BASELINE_WANDER_WINDOW_S = 60.0  # Longer window for drift (60s)
BASELINE_WANDER_THRESHOLD = 0.5  # 50% of range - very permissive
TREMOR_FREQ_MIN_HZ = 3.0         # Only very high freq noise (>3Hz)
TREMOR_POWER_THRESHOLD = 0.5     # 50% power in tremor band - very permissive

# Per-cycle quality detection (more permissive to avoid flagging normal variability)
AMPLITUDE_IQR_FACTOR = 3.5       # IQR multiplier for amplitude outliers (detect_amplitude_outliers)
TIMING_MAD_THRESHOLD = 4.0       # MAD units for breath interval outliers (detect_physiological_outliers)

# Interpolation settings
MAX_INTERP_CYCLES = 2            # Max breathing cycles to interpolate
MAX_INTERP_DURATION_S = 6.0      # Max seconds to interpolate (~2 cycles at 12 bpm)
INTERP_METHOD = "monotone_cubic" # "linear", "cubic", or "monotone_cubic" (NeuroKit2 default - prevents overshoots)

# Quality control
BAD_PROP_THRESHOLD_METRICS = 0.05   # 5% max bad for valid metrics (strict)
BAD_PROP_THRESHOLD_COUPLING = 0.05  # 5% max bad for coupling analyses
BAD_PROP_THRESHOLD = 0.20           # 20% max bad before excluding segment entirely
MAX_CONSECUTIVE_BAD_CYCLES = 2      # Max consecutive bad cycles for coupling validity

# Breath-Hold (BH) segment settings
BH_MIN_AMPLITUDE_THRESHOLD_FRACTION = 0.5  # 50% of signal range - filters out small noise during breath-hold
BH_RECOVERY_BREATH_AMPLITUDE_FRACTION = 0.3  # 30% of signal range - threshold for detecting recovery breath
BH_FLAT_PERIOD_MIN_DURATION_S = 5.0  # Minimum flat period duration to consider as breath-hold
MIN_IQR_OUTLIER_CYCLES = 8  # Minimum cycles before considering IQR-based RR outliers

# Plot colors
PLOT_COLORS = {
    "raw": "#1f77b4",
    "clean": "#2ca02c",
    "peaks": "#d62728",
    "troughs": "#ff7f0e",
    "rate": "#9467bd",
    "bad": "#ffbb78",
    "interpolated": "#17becf",
    "inspiration": "#87CEEB",
    "expiration": "#FFB6C1",
}


def _ensure_optional(dep, name: str):
    if dep is None:
        raise ImportError(f"{name} is required. Install with `pip install {name}`.")


def _to_path(path_like) -> Path:
    return Path(path_like).expanduser()


# =============================================================================
# GRÁFICOS INTERACTIVOS
# =============================================================================

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def editar_ciclos_interactivos(seg_dict, t0: float = None, win_len: float = None):
    times = seg_dict["times"]
    signal = seg_dict["rsp_clean"]
    fs = seg_dict["fs_rsp"]
    peaks = seg_dict["peaks"].tolist()
    troughs = seg_dict["troughs"].tolist()
    
    # Initialize change tracking
    initial_peaks = set(peaks)
    initial_troughs = set(troughs)
    changes = {
        "peaks_added": [],
        "peaks_removed": [],
        "troughs_added": [],
        "troughs_removed": []
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, signal, lw=1, color="#2ca02c", label="Señal filtrada")
    scat_peaks = ax.scatter(np.array(peaks)/fs, signal[peaks], c="red", marker="^", label="Peaks")
    scat_troughs = ax.scatter(np.array(troughs)/fs, signal[troughs], c="orange", marker="v", label="Troughs")
    ax.set_title(f"Editar ciclos: {seg_dict.get('seg', 'SEG')}\nSelecciona un modo y usa Ctrl+clic en el gráfico", fontsize=10)
    ax.legend(loc="upper right")
    ax.set_xlabel("Tiempo (s)")
    ax.grid(True)

    def refrescar():
        scat_peaks.set_offsets(np.column_stack((np.array(peaks)/fs, signal[peaks])))
        scat_troughs.set_offsets(np.column_stack((np.array(troughs)/fs, signal[troughs])))
        fig.canvas.draw_idle()

    # Mode: 'add_peak', 'add_trough', 'remove_peak', 'remove_trough'
    current_mode = 'remove_trough'
    
    def set_mode_add_peak(event):
        nonlocal current_mode
        current_mode = 'add_peak'
        btn_add_peak.color = 'lightgreen'
        btn_add_trough.color = 'lightgray'
        btn_remove_peak.color = 'lightgray'
        btn_remove_trough.color = 'lightgray'
        fig.canvas.draw_idle()
        print("  Mode: Add Peak (Ctrl+click)")
    
    def set_mode_add_trough(event):
        nonlocal current_mode
        current_mode = 'add_trough'
        btn_add_peak.color = 'lightgray'
        btn_add_trough.color = 'lightgreen'
        btn_remove_peak.color = 'lightgray'
        btn_remove_trough.color = 'lightgray'
        fig.canvas.draw_idle()
        print("  Mode: Add Trough (Ctrl+click)")
    
    def set_mode_remove_peak(event):
        nonlocal current_mode
        current_mode = 'remove_peak'
        btn_add_peak.color = 'lightgray'
        btn_add_trough.color = 'lightgray'
        btn_remove_peak.color = 'lightcoral'
        btn_remove_trough.color = 'lightgray'
        fig.canvas.draw_idle()
        print("  Mode: Remove Peak (Ctrl+click)")
    
    def set_mode_remove_trough(event):
        nonlocal current_mode
        current_mode = 'remove_trough'
        btn_add_peak.color = 'lightgray'
        btn_add_trough.color = 'lightgray'
        btn_remove_peak.color = 'lightgray'
        btn_remove_trough.color = 'lightcoral'
        fig.canvas.draw_idle()
        print("  Mode: Remove Trough (Ctrl+click)")

    def on_click(event):
        # Ignore clicks on buttons (they have their own handlers)
        if hasattr(event, 'inaxes') and event.inaxes != ax:
            return
        if event.inaxes != ax:
            return
        if event.xdata is None:
            return
        
        # Check for Ctrl key
        key_str = (event.key or "").lower() if event.key else ""
        has_ctrl = "control" in key_str or "ctrl" in key_str
        if hasattr(event, 'modifiers') and event.modifiers:
            has_ctrl = has_ctrl or bool(event.modifiers & 4)  # Control mask = 4
        
        if not has_ctrl:
            return
        
        idx = int(event.xdata * fs)
        idx = np.clip(idx, 0, len(signal)-1)
        
        if current_mode == 'add_peak':
            peaks.append(idx)
            t_added = idx / fs
            changes["peaks_added"].append(t_added)
            print(f"  ✓ Added peak at {t_added:.2f}s")
            refrescar()
        elif current_mode == 'add_trough':
            troughs.append(idx)
            t_added = idx / fs
            changes["troughs_added"].append(t_added)
            print(f"  ✓ Added trough at {t_added:.2f}s")
            refrescar()
        elif current_mode == 'remove_peak':
            if not peaks:
                print("  ⚠ No peaks to remove")
                return
            d_peaks = np.abs(np.array(peaks) - idx)
            remove_idx = int(np.argmin(d_peaks))
            removed_idx = peaks[remove_idx]
            removed_time = removed_idx / fs
            peaks.pop(remove_idx)
            changes["peaks_removed"].append(removed_time)
            print(f"  ✓ Removed peak at {removed_time:.2f}s")
            refrescar()
        elif current_mode == 'remove_trough':
            if not troughs:
                print("  ⚠ No troughs to remove")
                return
            d_troughs = np.abs(np.array(troughs) - idx)
            remove_idx = int(np.argmin(d_troughs))
            removed_idx = troughs[remove_idx]
            removed_time = removed_idx / fs
            troughs.pop(remove_idx)
            changes["troughs_removed"].append(removed_time)
            print(f"  ✓ Removed trough at {removed_time:.2f}s")
            refrescar()

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)

    def guardar(event):
        print(f"\n  ✓ Saving changes: {len(peaks)} peaks, {len(troughs)} troughs")
        seg_dict["peaks"] = np.array(sorted(set(peaks)))
        seg_dict["troughs"] = np.array(sorted(set(troughs)))
        
        # Save change log
        total_changes = (len(changes["peaks_added"]) + len(changes["peaks_removed"]) + 
                        len(changes["troughs_added"]) + len(changes["troughs_removed"]))
        if total_changes > 0:
            seg_dict["peak_trough_changes"] = changes
            print(f"  📝 Change log:")
            if changes["peaks_added"]:
                print(f"     + {len(changes['peaks_added'])} peaks added: {[f'{t:.2f}s' for t in sorted(changes['peaks_added'])]}")
            if changes["peaks_removed"]:
                print(f"     - {len(changes['peaks_removed'])} peaks removed: {[f'{t:.2f}s' for t in sorted(changes['peaks_removed'])]}")
            if changes["troughs_added"]:
                print(f"     + {len(changes['troughs_added'])} troughs added: {[f'{t:.2f}s' for t in sorted(changes['troughs_added'])]}")
            if changes["troughs_removed"]:
                print(f"     - {len(changes['troughs_removed'])} troughs removed: {[f'{t:.2f}s' for t in sorted(changes['troughs_removed'])]}")
        else:
            # Keep existing changes if any, or initialize empty
            if "peak_trough_changes" not in seg_dict:
                seg_dict["peak_trough_changes"] = {
                    "peaks_added": [],
                    "peaks_removed": [],
                    "troughs_added": [],
                    "troughs_removed": []
                }
        
        if len(seg_dict["troughs"]) > 1:
            metrics = calculate_breathing_metrics(seg_dict["rsp_clean"], seg_dict["peaks"], seg_dict["troughs"], fs)
            seg_dict.update(metrics)
        fig.canvas.mpl_disconnect(cid_click)
        plt.close(fig)

    def cancelar(event):
        print(f"\n  ✗ Cancelled - changes not saved")
        fig.canvas.mpl_disconnect(cid_click)
        plt.close(fig)

    # Create mode buttons
    ax_add_peak = plt.axes([0.02, 0.02, 0.12, 0.04])
    ax_add_trough = plt.axes([0.15, 0.02, 0.12, 0.04])
    ax_remove_peak = plt.axes([0.28, 0.02, 0.12, 0.04])
    ax_remove_trough = plt.axes([0.41, 0.02, 0.12, 0.04])
    ax_guardar = plt.axes([0.75, 0.02, 0.1, 0.04])
    ax_cancelar = plt.axes([0.86, 0.02, 0.1, 0.04])
    
    btn_add_peak = Button(ax_add_peak, "Añadir Peak")
    btn_add_trough = Button(ax_add_trough, "Añadir Trough")
    btn_remove_peak = Button(ax_remove_peak, "Eliminar Peak")
    btn_remove_trough = Button(ax_remove_trough, "Eliminar Trough")
    
    btn_add_peak.on_clicked(set_mode_add_peak)
    btn_add_trough.on_clicked(set_mode_add_trough)
    btn_remove_peak.on_clicked(set_mode_remove_peak)
    btn_remove_trough.on_clicked(set_mode_remove_trough)
    
    # Set initial mode
    btn_remove_trough.color = 'lightcoral'
    
    Button(ax_guardar, "Guardar").on_clicked(guardar)
    Button(ax_cancelar, "Cancelar").on_clicked(cancelar)
    fig.subplots_adjust(bottom=0.15, top=0.95)
    if t0 is not None and win_len is not None:
        ax.set_xlim(t0, t0 + win_len)
    plt.show()


def editar_ciclos_interactivos_externo(seg_dict, enable=True, t0: float = None, win_len: float = None):
    """
    Lanza el editor Tk externo para tocar picos solo cuando quieras.
    """
    if not enable:
        return seg_dict

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    np.savez(
        tmp.name,
        peaks=seg_dict["peaks"],
        troughs=seg_dict["troughs"],
        times=seg_dict["times"],
        rsp_clean=seg_dict["rsp_clean"],
        fs=seg_dict["fs_rsp"],
        seg=seg_dict.get("seg", "SEG"),
    )

    editor = Path(__file__).with_name("peak_editor.py")
    if not editor.exists():
        print(f"  ⚠ No se encuentra el editor: {editor}")
        return seg_dict
    cmd = [sys.executable, str(editor), tmp.name]
    if t0 is not None:
        cmd += ["--t0", str(t0)]
    if win_len is not None:
        cmd += ["--win_len", str(win_len)]

    print(f"  ▶ Lanzando editor: {' '.join(cmd)}")
    # Run the editor (don't check=True to avoid exceptions if window is closed).
    # Capture output so the subprocess console (and any error) is visible even
    # when running inside VS Code / Jupyter, which hide child-process output.
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.returncode != 0:
            print(f"  ⚠ El editor terminó con código {proc.returncode}")
            if proc.stderr:
                print("  ----- error del editor -----")
                print(proc.stderr, end="")
                print("  ----------------------------")
    except Exception as e:
        print(f"  ⚠ Error running editor: {e}")

    # Check if changes were saved
    status_file = tmp.name + ".saved"
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status = f.read().strip()
            os.remove(status_file)  # Clean up status file
        except Exception as e:
            print(f"  ⚠ Error reading state file: {e}")
            status = "cancelled"
        
        if status == "saved":
            print("  ✓ Changes saved, reloading data...")
            data = np.load(tmp.name)
            # Ensure peaks and troughs are numpy arrays
            seg_dict["peaks"] = np.asarray(data["peaks"], dtype=int)
            seg_dict["troughs"] = np.asarray(data["troughs"], dtype=int)
            
            # Load changes from JSON file if it exists
            changes_file = tmp.name + ".changes.json"
            if os.path.exists(changes_file):
                import json
                try:
                    with open(changes_file, 'r') as f:
                        new_changes = json.load(f)
                    # Merge with existing changes
                    existing_changes = seg_dict.get("peak_trough_changes", {
                        "peaks_added": [],
                        "peaks_removed": [],
                        "troughs_added": [],
                        "troughs_removed": []
                    })
                    merged_changes = {
                        "peaks_added": existing_changes.get("peaks_added", []) + new_changes.get("peaks_added", []),
                        "peaks_removed": existing_changes.get("peaks_removed", []) + new_changes.get("peaks_removed", []),
                        "troughs_added": existing_changes.get("troughs_added", []) + new_changes.get("troughs_added", []),
                        "troughs_removed": existing_changes.get("troughs_removed", []) + new_changes.get("troughs_removed", []),
                    }
                    seg_dict["peak_trough_changes"] = merged_changes
                except Exception as e:
                    print(f"  ⚠ Error reading changes: {e}")
                finally:
                    # Clean up changes file
                    try:
                        os.remove(changes_file)
                    except:
                        pass
            
            if len(seg_dict["troughs"]) > 1:
                metrics = calculate_breathing_metrics(seg_dict["rsp_clean"], seg_dict["peaks"], seg_dict["troughs"], seg_dict["fs_rsp"])
                seg_dict.update(metrics)
        else:
            print("  ✗ Changes cancelled, data not reloaded")
    else:
        # If no status file, assume cancelled (window closed without clicking buttons)
        print("  ✗ Window closed without saving, data not reloaded")
    
    # Clean up temp file
    try:
        os.remove(tmp.name)
    except:
        pass
    
    return seg_dict


def convertir_cambios_a_bad_zones(seg_dict: dict, 
                                   merge_gap_cycles: float = 1.5,
                                   edge_margin_cycles: float = 0.5) -> dict:
    """
    Convierte picos/troughs eliminados manualmente en zonas BAD.
    
    NUEVA ESTRATEGIA (basada en NeuroKit2 y literatura):
    - Agrupa picos/troughs eliminados cercanos en UNA zona continua
    - NO fragmenta zonas grandes (evita crear múltiples intervalos solapados)
    - Conservador: marca zona completa como BAD sin interpolar
    
    Basado en mejores prácticas:
    - NeuroKit2: NO interpola artifacts, solo corrige picos individuales
    - Literatura RSA/coupling: usar NaN/mask para gaps >2 ciclos
    - Representación dual: rsp_masked (NaN) para coupling, rsp_interp opcional
    
    Args:
        seg_dict: Segment dictionary con peak_trough_changes y peaks/troughs
        merge_gap_cycles: Distancia máxima (en ciclos) para agrupar picos eliminados
                         (default: 1.5 ciclos = picos a <1.5 ciclos se agrupan en UNA zona)
        edge_margin_cycles: Margen en ciclos a añadir en los bordes de cada zona
                           (default: 0.5 ciclos antes y después del grupo)
    
    Returns:
        seg_dict con zonas BAD añadidas (una zona continua por grupo de picos eliminados)
    
    Example:
        # Después de editar picos manualmente:
        resultado_LB = editar_ciclos_interactivos_externo(resultado_LB, enable=True)
        # Convertir eliminaciones en zonas BAD (agrupa cercanos):
        resultado_LB = convertir_cambios_a_bad_zones(resultado_LB)
        # Verificar zonas creadas:
        ver_señal_completa(resultado_LB, show_bad=True)
        # Crear representaciones (sin interpolar, usar NaN):
        resultado_LB = limpiar(resultado_LB, aplicar_interpolacion=False)
        # Usar rsp_masked para RSA/coupling
    """
    seg_name = seg_dict.get("seg", "SEG")
    changes = seg_dict.get("peak_trough_changes", {})
    
    if not changes:
        return seg_dict
    
    peaks_removed = changes.get("peaks_removed", [])
    troughs_removed = changes.get("troughs_removed", [])
    
    if not peaks_removed and not troughs_removed:
        return seg_dict
    
    times = seg_dict.get("times", np.array([]))
    
    if len(times) == 0:
        return seg_dict
    
    # Estimar duración típica del ciclo respiratorio
    breath_intervals = seg_dict.get("breath_intervals_s", np.array([]))
    if len(breath_intervals) > 0:
        typical_cycle_dur = float(np.median(breath_intervals))
    else:
        typical_cycle_dur = 4.0  # Default: 4 segundos por ciclo
    
    # 1. Combinar todos los picos/troughs eliminados y ordenar
    all_removed_times = sorted(list(peaks_removed) + list(troughs_removed))
    
    if len(all_removed_times) == 0:
        return seg_dict
    
    # 2. Agrupar picos/troughs que están cercanos (dentro de merge_gap_cycles)
    gap_max_s = merge_gap_cycles * typical_cycle_dur
    grupos = []
    current_group = [all_removed_times[0]]
    
    for i in range(1, len(all_removed_times)):
        prev_time = all_removed_times[i-1]
        curr_time = all_removed_times[i]
        gap = curr_time - prev_time
        
        if gap <= gap_max_s:
            # Está cerca del anterior, añadir al grupo actual
            current_group.append(curr_time)
        else:
            # Gap grande, guardar grupo actual y empezar uno nuevo
            grupos.append(current_group)
            current_group = [curr_time]
    
    # Añadir último grupo
    grupos.append(current_group)
    
    # 3. Para cada grupo, crear UNA zona BAD continua
    zones_created = 0
    margin_s = edge_margin_cycles * typical_cycle_dur
    
    for grupo in grupos:
        # Tiempo del primer y último pico/trough del grupo
        t_first = min(grupo)
        t_last = max(grupo)
        
        # Crear zona desde el primer al último punto, con márgenes en los bordes
        t_ini = t_first - margin_s
        t_fin = t_last + margin_s
        
        # Asegurar que no salga de los límites de la señal
        t_ini = max(t_ini, times[0])
        t_fin = min(t_fin, times[-1])
        
        # Determinar motivo basado en el número de puntos en el grupo
        if len(grupo) == 1:
            motivo = "peak_removed"
        else:
            motivo = f"multiple_peaks_removed_{len(grupo)}"
        
        # Crear zona BAD
        seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo=motivo, sev=1.0)
        zones_created += 1
    
    if zones_created > 0:
        # Recalcular bad_pct
        bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
        seg_dict["bad_pct"] = bad_pct
        seg_dict["bad_s"] = bad_s
        print(f"✓ {zones_created} BAD zone(s) created | Total: {bad_s:.1f}s ({bad_pct*100:.1f}%)")
    
    return seg_dict


# =============================================================================
# RECOMMENDED CLEANING WORKFLOW (NeuroKit2 + Literatura)
# =============================================================================
"""
FLUJO RECOMENDADO DE LIMPIEZA DE SEÑALES RESPIRATORIAS
======================================================

Basado en: NeuroKit2, Khodadad et al. (2018), literatura RSA/coupling 2024-2026

1. INSPECCIÓN INICIAL
   └─ ver_señal_completa() - evaluación visual completa de la señal

2. EDICIÓN MANUAL DE PICOS/TROUGHS
   └─ editar_ciclos_interactivos_externo() - eliminar picos/troughs anómalos
   └─ convertir_cambios_a_bad_zones() - agrupa picos eliminados cercanos en zonas únicas

3. MARCACIÓN MANUAL ADICIONAL (si necesario)
   └─ agregar_bad_zone() - marcar zonas problemáticas adicionales

4. CREAR REPRESENTACIONES DUALES (sin interpolar por defecto)
   └─ limpiar() - crea rsp_masked (NaN) y opcionalmente rsp_interp
      IMPORTANTE: aplicar_interpolacion=False por defecto (NeuroKit2 approach)

5. VALIDACIÓN VISUAL
   └─ ver_señal_limpia() - verificar resultado final
   └─ reinspeccionar_segmento() - métricas finales

PRINCIPIOS CLAVE (NeuroKit2 + Literatura):
- NO interpolar por defecto: usa NaN/mask (recomendado para RSA/coupling)
- Interpolar solo si necesitas métricas segmentales Y zonas son <2 ciclos
- Usar monotone_cubic si interpolación es necesaria (previene overshoots)
- Agrupar picos eliminados cercanos (no fragmentar zonas grandes)
- Trazabilidad completa de todas las intervenciones

¿CUÁNDO USAR CADA SEÑAL?
------------------------
rsp_masked (NaN en zonas BAD):
  - RSA (Respiratory Sinus Arrhythmia)
  - ECG-RSP coupling
  - Heart-brain coupling
  - Cualquier análisis tiempo-sincronizado con otra señal
  → Uso principal, enfoque conservador

rsp_interp (zonas cortas interpoladas):
  - Métricas segmentales (frecuencia media, amplitud media)
  - Solo si las zonas BAD son muy cortas (<2 ciclos, <6s)
  → Uso secundario, solo cuando es estrictamente necesario

rsp_clean (original filtrada):
  - Calcular métricas excluyendo ciclos BAD manualmente
  - Visualización de señal antes de limpieza

EJEMPLO DE USO (ENFOQUE CONSERVADOR):
    # 1. Inspección inicial
    resultado_LB = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name="LB")
    ver_señal_completa(resultado_LB, show_bad=False)
    
    # 2. Limpieza completa (NO interpolar por defecto)
    resultado_LB = limpiar_segmento_lb(resultado_LB, t0_edicion=180.0, 
                                       aplicar_interpolacion=False)
    
    # 3. Ver resultado (rsp_masked con NaN)
    ver_señal_limpia(resultado_LB)
    
    # 4. Usar rsp_masked para RSA/coupling
    rsp_for_rsa = resultado_LB["rsp_masked"]  # NaN en zonas BAD

EJEMPLO ALTERNATIVO (SI NECESITAS INTERPOLAR):
    # Solo si necesitas métricas segmentales Y zonas son cortas
    resultado_LB = limpiar_segmento_lb(resultado_LB, t0_edicion=180.0,
                                       aplicar_interpolacion=True)
    
    # Usar rsp_masked para RSA, rsp_interp para métricas
    rsp_for_rsa = resultado_LB["rsp_masked"]      # NaN (no interpolado)
    rsp_for_metrics = resultado_LB["rsp_interp"]  # Interpolado (solo zonas cortas)
"""


# =============================================================================
# SIGNAL LOADING
# =============================================================================

def cargar_senal_y_eventos(acq_file: str, rsp_col: str):
    """
    Load RSP signal and event markers from BIOPAC .acq file.
    
    Returns:
        - rsp_raw: np.ndarray (full RSP signal)
        - fs_rsp: float (Hz)
        - eventos: dict[str, int] (text -> first sample_index)
        - eventos_full: dict[str, list[int]] (text -> all sample_indices)
        - data_events: bioread object
    """
    physio_file, fs = nk.read_acqknowledge(acq_file)
    
    if rsp_col not in physio_file.columns:
        raise ValueError(f"RSP channel '{rsp_col}' not found. Available: {list(physio_file.columns)}")
    
    rsp_raw = physio_file[rsp_col].values
    fs_rsp = float(fs)
    
    _ensure_optional(bioread, "bioread")
    data_events = bioread.read_file(acq_file)
    
    eventos_full: Dict[str, List[int]] = {}
    for m in data_events.event_markers:
        k = (m.text or "").strip()
        if k == "":
            continue
        eventos_full.setdefault(k, []).append(int(m.sample_index))
    
    eventos: Dict[str, int] = {k: v[0] for k, v in eventos_full.items()}
    
    print("=== RSP SIGNAL INFO ===")
    print(f"Sampling rate: {fs_rsp} Hz")
    print(f"Total samples: {len(rsp_raw)}")
    print(f"Duration: {len(rsp_raw)/fs_rsp:.1f} s")
    print(f"RSP range: {np.nanmin(rsp_raw):.4f} to {np.nanmax(rsp_raw):.4f}")
    
    print("\n=== EVENTS FOUND ===")
    for k, v in eventos_full.items():
        if len(v) == 1:
            print(repr(k), "->", v[0])
        else:
            print(repr(k), "->", v, " (DUPLICATE)")
    
    return rsp_raw, fs_rsp, eventos, eventos_full, data_events


def cargar_datos_y_segmentos(
    acq_file: str,
    rsp_col: str,
    segmentos_interes: Sequence[str],
    duraciones_objetivo: Optional[Dict[str, float]] = None,
    segmentos_manuales: Sequence[str] = (),
):
    """
    Load RSP signal and build segment dictionary.
    
    Returns:
        rsp_raw, fs_rsp, eventos, eventos_full, data_events, segmentos_dict
    """
    rsp_raw, fs_rsp, eventos, eventos_full, data_events = cargar_senal_y_eventos(acq_file, rsp_col)
    
    segmentos_dict = construir_diccionario_segmentos(
        eventos=eventos,
        segmentos_interes=segmentos_interes,
        duraciones_objetivo=duraciones_objetivo,
        fs=fs_rsp,
        n_total=len(rsp_raw),
        segmentos_manuales=segmentos_manuales,
    )
    
    return rsp_raw, fs_rsp, eventos, eventos_full, data_events, segmentos_dict


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


def get_segment_indices(eventos: Dict[str, int], seg: str, n_total: int) -> Tuple[int, int]:
    """Get (start_idx, end_idx) sample indices for a segment."""
    seg_lower = seg.lower().strip()
    
    if seg_lower.startswith("post"):
        if "Post 10min" not in eventos:
            raise ValueError("Missing 'Post 10min' marker")
        return int(eventos["Post 10min"]), int(n_total)
    
    key_ini, key_fin = _keys_segmento(seg)
    start = int(eventos[key_ini])
    end = int(eventos[key_fin])
    
    if start < 0 or end > n_total or end <= start:
        raise ValueError(f"Segment '{seg}' out of range (start={start}, end={end}, n_total={n_total})")
    
    return start, end


def construir_diccionario_segmentos(
    eventos: Dict[str, int],
    segmentos_interes: Sequence[str],
    duraciones_objetivo: Optional[Dict[str, float]] = None,
    fs: Optional[float] = None,
    n_total: Optional[int] = None,
    segmentos_manuales: Sequence[str] = (),
) -> Dict[str, Tuple[int, int]]:
    """Build {segment: (start_idx, end_idx)} dict using marker conventions."""
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
            key_ini, _ = _keys_segmento(seg)
            if key_ini in eventos and seg in duraciones_objetivo and fs is not None:
                start_idx = eventos[key_ini]
                dur_samples = int(duraciones_objetivo[seg] * fs)
                end_idx = min(start_idx + dur_samples, n_total or start_idx + dur_samples)
                segmentos_dict[seg] = (start_idx, end_idx)
                print(f"⚠️  Using target duration for {seg}: {duraciones_objetivo[seg]}s")
            else:
                print(f"⚠️  Could not build segment {seg}: {e}")
    return segmentos_dict


def inspeccionar_duraciones_segmentos(
    segmentos_dict: Dict[str, Tuple[int, int]],
    segmentos_interes: Sequence[str],
    fs_rsp: float,
    sujeto_id: str = "N/A"
) -> None:
    """
    Muestra una tabla con las duraciones, tiempos de inicio/fin e índices de muestras
    para todos los segmentos de interés.
    
    Args:
        segmentos_dict: Diccionario con {segmento: (start_idx, end_idx)}
        segmentos_interes: Lista de nombres de segmentos a inspeccionar
        fs_rsp: Frecuencia de muestreo del RSP (Hz)
        sujeto_id: ID del sujeto para mostrar en el encabezado
    """
    print(f"\n=== SEGMENT DURATION INSPECTION (SUBJECT {sujeto_id}) ===\n")
    for seg in segmentos_interes:
        info = segmentos_dict.get(seg)
        if info is None:
            print(f"{seg:15s}: NOT FOUND in segmentos_dict")
            continue
        start_idx, end_idx = info
        dur_s = (end_idx - start_idx) / fs_rsp if fs_rsp else float("nan")
        t_start_s = start_idx / fs_rsp if fs_rsp else float("nan")
        t_end_s = end_idx / fs_rsp if fs_rsp else float("nan")
        print(f"{seg:15s}: {dur_s:7.2f} s  |  Start: {t_start_s:8.2f} s  |  End: {t_end_s:8.2f} s  |  Samples: {start_idx:8d} - {end_idx:8d}")
    print()


# =============================================================================
# DOWNSAMPLING
# =============================================================================

def downsample_signal(signal: np.ndarray, fs_original: float, fs_target: float) -> Tuple[np.ndarray, float]:
    """
    Downsample RSP signal to target frequency with anti-aliasing.
    
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
    
    if _HAVE_SCIPY and len(signal) >= 20:
        # Anti-aliasing filter
        nyq = fs_target / 2.0
        b, a = scipy_signal.butter(4, nyq / (fs_original / 2.0), btype='low')
        filtered = scipy_signal.filtfilt(b, a, signal)
        downsampled = filtered[::factor]
    else:
        n_new = len(signal) // factor
        if n_new > 0:
            downsampled = signal[:n_new * factor].reshape(-1, factor).mean(axis=1)
        else:
            downsampled = signal.copy()
    
    return downsampled, fs_original / factor


# =============================================================================
# FILTERING
# =============================================================================

def filter_rsp(
    rsp: np.ndarray,
    fs: float,
    lowpass_cutoff: float = LOWPASS_CUTOFF_HZ,
    highpass_cutoff: float = HIGHPASS_CUTOFF_HZ,
    numtaps: int = FIR_NUMTAPS,
) -> np.ndarray:
    """
    Apply bandpass FIR filter to RSP signal.
    
    Default: 0.05-1 Hz (captures 3-60 breaths/min)
    
    Args:
        rsp: Raw RSP signal
        fs: Sampling frequency
        lowpass_cutoff: Upper cutoff (Hz)
        highpass_cutoff: Lower cutoff (Hz)
        numtaps: Number of taps (filter length)
    
    Returns:
        Filtered RSP signal
    """
    if not _HAVE_SCIPY:
        print("  ⚠️  scipy not available, returning unfiltered signal")
        return rsp.copy()
    
    nyq = fs / 2.0
    low_norm = min(highpass_cutoff / nyq, 0.99)
    high_norm = min(lowpass_cutoff / nyq, 0.99)
    
    if low_norm >= high_norm:
        print(f"  ⚠️  Invalid filter cutoffs: {highpass_cutoff}-{lowpass_cutoff} Hz")
        return rsp.copy()
    
    # Use FIR filter (firwin)
    # Note: for bandpass, pass_zero must be False
    try:
        b = scipy_signal.firwin(numtaps, [low_norm, high_norm], pass_zero=False)
        return scipy_signal.filtfilt(b, [1.0], rsp)
    except ValueError as e:
        # If numtaps is even and we have a bandpass, firwin might fail depending on window
        # Try with numtaps + 1 if it fails
        b = scipy_signal.firwin(numtaps + 1, [low_norm, high_norm], pass_zero=False)
        return scipy_signal.filtfilt(b, [1.0], rsp)


# =============================================================================
# ROBUST STATISTICS HELPERS (MAD-based)
# =============================================================================

def compute_mad(values: np.ndarray) -> float:
    """
    Compute Median Absolute Deviation (MAD).
    MAD is robust to outliers, unlike standard deviation.
    """
    median = np.median(values)
    return np.median(np.abs(values - median))


def compute_robust_zscore(values: np.ndarray) -> np.ndarray:
    """
    Compute robust z-scores using MAD instead of std.
    z_robust = (x - median) / (MAD * 1.4826)
    The factor 1.4826 makes MAD comparable to std for normal distributions.
    """
    median = np.median(values)
    mad = compute_mad(values)
    if mad < 1e-10:
        return np.zeros_like(values)
    return (values - median) / (mad * MAD_TO_STD)


def compute_robust_threshold(values: np.ndarray, k: float = 5.0) -> float:
    """
    Compute robust threshold as median + k * MAD * 1.4826.
    Default k=5.0 is permissive (catches only extreme outliers).
    """
    median = np.median(values)
    mad = compute_mad(values)
    return median + k * mad * MAD_TO_STD


def compute_iqr_bounds(values: np.ndarray, factor: float = 2.5) -> Tuple[float, float]:
    """
    Compute IQR-based outlier bounds.
    Returns (lower_bound, upper_bound).
    Default factor=2.5 is permissive.
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return (q1 - factor * iqr, q3 + factor * iqr)


# =============================================================================
# ARTIFACT DETECTION (Mathematical Rules - MAD-based for robustness)
# =============================================================================

def detect_flat_segments(
    rsp: np.ndarray,
    fs: float,
    min_duration_s: float = FLAT_LINE_THRESHOLD_S,
    std_threshold: float = FLAT_LINE_STD_THRESHOLD,
) -> List[Tuple[float, float, str]]:
    """
    Detect flat/stuck signal segments.
    Rule: Rolling std < threshold for > min_duration
    """
    win_samples = max(int(min_duration_s * fs), 2)
    
    # Normalize signal for threshold comparison
    rsp_norm = (rsp - np.mean(rsp)) / (np.std(rsp) + 1e-10)
    
    # Rolling std
    rolling_std = np.zeros_like(rsp_norm)
    for i in range(win_samples, len(rsp_norm)):
        rolling_std[i] = np.std(rsp_norm[i-win_samples:i])
    
    is_flat = rolling_std < std_threshold
    is_flat[:win_samples] = False  # Ignore initial window
    
    # Convert to intervals
    intervals = _bool_to_intervals(is_flat, fs)
    return [(s, e, "flat_segment") for s, e in intervals if (e - s) >= min_duration_s]


def detect_clipping(
    rsp: np.ndarray,
    fs: float,
    percentile: float = CLIPPING_PERCENTILE,
    min_duration_s: float = 0.5,
) -> List[Tuple[float, float, str]]:
    """
    Detect signal clipping at sensor limits.
    Rule: Signal at extreme percentiles for extended periods
    """
    high_thresh = np.percentile(rsp, 100 - percentile)
    low_thresh = np.percentile(rsp, percentile)
    
    # Check for truly flat at extremes
    is_clipped_high = rsp >= high_thresh
    is_clipped_low = rsp <= low_thresh
    
    intervals = []
    for is_clipped in [is_clipped_high, is_clipped_low]:
        for s, e in _bool_to_intervals(is_clipped, fs):
            if (e - s) >= min_duration_s:
                intervals.append((s, e, "clipping"))
    
    return _merge_artifact_intervals(intervals, margin=ARTIFACT_MERGE_MARGIN_S)


def detect_low_amplitude_zones(
    rsp: np.ndarray,
    fs: float,
    window_s: float = 5.0,
    step_s: float = 1.0,
    amp_frac: float = 0.15,
    min_duration_s: float = 2.0,
) -> List[Tuple[float, float, str]]:
    """
    Detect zones where respiration amplitude (peak-to-trough) is very low,
    which often indicates a loose belt or poor contact (flat-but-not-stuck).

    Args:
        rsp: Signal array
        fs: Sampling frequency
        window_s: Window size to compute local ptp (seconds)
        step_s: Step between windows (seconds)
        amp_frac: Fraction of global signal range considered 'low amplitude'
        min_duration_s: Minimum contiguous duration to report
    Returns:
        List of (start_s, end_s, "low_amplitude")
    """
    n = len(rsp)
    if n == 0:
        return []

    win = max(1, int(window_s * fs))
    step = max(1, int(step_s * fs))
    signal_range = np.ptp(rsp)
    if signal_range <= 0:
        return []

    thr = max(amp_frac * signal_range, 1e-6)
    low_mask = np.zeros(n, dtype=bool)

    for start in range(0, n - win + 1, step):
        seg = rsp[start:start + win]
        local_ptp = np.ptp(seg)
        if local_ptp <= thr:
            low_mask[start:start + win] = True

    # Convert boolean mask to intervals
    intervals = _bool_to_intervals(low_mask, fs)
    return [(s, e, "low_amplitude") for s, e in intervals if (e - s) >= min_duration_s]


def detect_step_artifacts(
    rsp: np.ndarray,
    fs: float,
    threshold_mad: float = STEP_MAD_THRESHOLD,
    margin_s: float = STEP_ARTIFACT_MARGIN_S,
    use_robust: bool = True,
) -> List[Tuple[float, float, str]]:
    """
    Detect sudden step changes (jumps/drops) using MAD-based robust threshold.
    
    Rule: |derivative| > median + threshold_mad * MAD * 1.4826
    
    MAD (Median Absolute Deviation) is robust to outliers, so normal breathing
    variations won't inflate the threshold like std would.
    
    Args:
        rsp: Respiration signal
        fs: Sampling frequency
        threshold_mad: Threshold in MAD units (default 5.0 = permissive)
        margin_s: Margin around detected step (seconds)
        use_robust: If True, use MAD; if False, fall back to std
    """
    diff = np.abs(np.diff(rsp))
    
    if use_robust:
        # MAD-based threshold (robust to outliers)
        threshold = compute_robust_threshold(diff, k=threshold_mad)
    else:
        # Legacy std-based threshold
        threshold = np.median(diff) + threshold_mad * np.std(diff)

    # Use the absolute derivative guard only for clearly unrealistic spikes;
    # this prevents moderate breaths from being flagged while still catching
    # extreme jitters (see STEP_ABS_THRESHOLD above).
    threshold = max(threshold, STEP_ABS_THRESHOLD)
    
    jump_idx = np.where(diff > threshold)[0]
    if len(jump_idx) == 0:
        return []
    
    # IMPROVEMENT: Find real peak (maximum local derivative)
    # This ensures we mark the actual spike even if detection was slightly off
    refined_indices = []
    search_win = int(fs * 0.05) # 50ms search window
    for idx in jump_idx:
        start_search = max(0, idx - search_win)
        end_search = min(len(diff), idx + search_win)
        local_max_idx = start_search + np.argmax(diff[start_search:end_search])
        refined_indices.append(local_max_idx)
    
    jump_idx = sorted(list(set(refined_indices)))
    
    # Create very narrow intervals centered on each detected spike (only mark the specific peak)
    intervals = []
    for idx in jump_idx:
        # Create a very small interval around the spike (e.g., 0.1s total width)
        peak_time_s = idx / fs
        interval_width_s = max(0.1, 2 * margin_s)  # At least 0.1s, or 2*margin if larger
        start_s = max(0, peak_time_s - interval_width_s / 2)
        end_s = min(len(rsp) / fs, peak_time_s + interval_width_s / 2)
        intervals.append((start_s, end_s, "step_artifact"))
    
    # Don't merge intervals - keep each peak separate (user wants only specific peaks, not merged zones)
    merged = _merge_artifact_intervals(intervals, margin=ARTIFACT_MERGE_MARGIN_S)
    
    # Filter out artifacts shorter than minimum duration (very short threshold to allow single peaks)
    result = [(s, e, reason) for s, e, reason in merged if (e - s) >= MIN_ARTIFACT_DURATION_S]
    return result


def detect_baseline_wander(
    rsp: np.ndarray,
    fs: float,
    window_s: float = BASELINE_WANDER_WINDOW_S,
    threshold_frac: float = BASELINE_WANDER_THRESHOLD,
) -> List[Tuple[float, float, str]]:
    """
    Detect excessive baseline wander/drift.
    Rule: Local trend exceeds threshold fraction of signal range
    """
    if not _HAVE_SCIPY:
        return []
    
    win_samples = int(window_s * fs)
    if win_samples >= len(rsp):
        return []
    
    # Estimate baseline with moving median
    from scipy.ndimage import median_filter
    baseline = median_filter(rsp, size=win_samples)
    
    # Calculate local trend magnitude
    signal_range = np.ptp(rsp)
    if signal_range < 1e-10:
        return []
    
    # Check for excessive drift
    drift_magnitude = np.abs(baseline - np.mean(baseline))
    is_wandering = drift_magnitude > (threshold_frac * signal_range)
    
    intervals = _bool_to_intervals(is_wandering, fs)
    return [(s, e, "baseline_wander") for s, e in intervals if (e - s) >= window_s/2]


def detect_tremor_noise(
    rsp: np.ndarray,
    fs: float,
    freq_min: float = TREMOR_FREQ_MIN_HZ,
    power_threshold: float = TREMOR_POWER_THRESHOLD,
    window_s: float = 5.0,
) -> List[Tuple[float, float, str]]:
    """
    Detect high-frequency tremor/noise ("tembleque").
    Rule: Power in high-freq band > threshold of total power
    """
    if not _HAVE_SCIPY:
        return []
    
    win_samples = int(window_s * fs)
    if win_samples >= len(rsp):
        return []
    
    intervals = []
    step = win_samples // 2  # 50% overlap
    
    for i in range(0, len(rsp) - win_samples, step):
        segment = rsp[i:i + win_samples]
        
        # Calculate power spectrum
        freqs = np.fft.rfftfreq(len(segment), 1/fs)
        psd = np.abs(np.fft.rfft(segment))**2
        
        # Calculate power in tremor band vs total
        tremor_mask = freqs >= freq_min
        total_power = np.sum(psd)
        tremor_power = np.sum(psd[tremor_mask])
        
        if total_power > 0 and (tremor_power / total_power) > power_threshold:
            start_s = i / fs
            end_s = (i + win_samples) / fs
            intervals.append((start_s, end_s, "tremor_noise"))
    
    return _merge_artifact_intervals(intervals, margin=ARTIFACT_MERGE_MARGIN_S)


def detect_baseline_shifts(
    rsp: np.ndarray,
    fs: float,
    short_win_s: float = BASELINE_SHIFT_SHORT_WIN_S,
    long_win_s: float = BASELINE_SHIFT_LONG_WIN_S,
    threshold_mad: float = BASELINE_SHIFT_MAD,
    margin_s: float = BASELINE_SHIFT_MARGIN_S,
    min_duration_s: float = BASELINE_SHIFT_MIN_DURATION_S,
) -> List[Tuple[float, float, str]]:
    """
    Detect sudden baseline shifts using rolling mean comparison.
    
    Rule: |short_mean - long_mean| > threshold_mad * MAD
    
    This catches abrupt changes in signal level (e.g., strap adjustment)
    that may not show up as sharp spikes but shift the baseline.
    
    Args:
        rsp: Respiration signal
        fs: Sampling frequency
        short_win_s: Short window for rolling mean (default 2s)
        long_win_s: Long window for rolling mean (default 10s)
        threshold_mad: Threshold in MAD units
        margin_s: Margin around detected shift
    """
    if not _HAVE_SCIPY:
        return []
    
    short_win = int(short_win_s * fs)
    long_win = int(long_win_s * fs)
    
    if long_win >= len(rsp):
        return []
    
    # Compute rolling means
    short_mean = uniform_filter1d(rsp, size=short_win, mode='nearest')
    long_mean = uniform_filter1d(rsp, size=long_win, mode='nearest')
    
    # Difference between short and long-term means
    diff = np.abs(short_mean - long_mean)
    
    # MAD-based threshold
    threshold = compute_robust_threshold(diff, k=threshold_mad)
    
    is_shifted = diff > threshold
    
    # Ignore edges (where rolling mean is unreliable)
    edge_samples = long_win // 2
    is_shifted[:edge_samples] = False
    is_shifted[-edge_samples:] = False
    
    intervals = _bool_to_intervals(is_shifted, fs)
    
    # Add very narrow margin and filter short detections
    # Only mark the specific shift point, not a wide zone around it
    result = []
    for s, e in intervals:
        duration = e - s
        if duration >= min_duration_s:  # Only flag long, sustained shifts (default 2.0s)
            # Use very narrow margin - only mark the shift point itself
            center_s = (s + e) / 2
            interval_width_s = max(0.1, 2 * margin_s)  # At least 0.1s, or 2*margin if larger
            s_margin = max(0, center_s - interval_width_s / 2)
            e_margin = min(len(rsp)/fs, center_s + interval_width_s / 2)
            result.append((s_margin, e_margin, "baseline_shift"))
    
    # Don't merge - keep each shift separate
    return _merge_artifact_intervals(result, margin=ARTIFACT_MERGE_MARGIN_S)


def detect_physiological_outliers(
    breath_intervals_s: np.ndarray,
    breath_times_s: np.ndarray,
    min_breath_dur: float = 60.0 / MAX_BREATHS_PER_MIN,
    max_breath_dur: float = 60.0 / MIN_BREATHS_PER_MIN,
    timing_mad_threshold: float = TIMING_MAD_THRESHOLD,
    margin_s: float = 2.0,
    use_adaptive: bool = True,
) -> List[Tuple[float, float, str]]:
    """
    Detect physiologically implausible breathing patterns using adaptive bounds.
    
    Rule: Breath duration outside adaptive IQR/MAD-based bounds,
          constrained by physiological limits.
    
    Args:
        breath_intervals_s: Array of breath-to-breath intervals (seconds)
        breath_times_s: Corresponding timestamps
        min_breath_dur: Absolute minimum (60/MAX_BPM)
        max_breath_dur: Absolute maximum (60/MIN_BPM)
        timing_mad_threshold: MAD threshold for adaptive bounds
        margin_s: Margin around detected outlier
        use_adaptive: If True, use individual's pattern to set bounds
    """
    if len(breath_intervals_s) < 4:
        return []
    
    if use_adaptive and len(breath_intervals_s) >= 10:
        # Adaptive bounds based on individual's breathing pattern
        median_interval = np.median(breath_intervals_s)
        mad_interval = compute_mad(breath_intervals_s)
        
        # Adaptive bounds constrained by physiological limits
        adaptive_low = median_interval - timing_mad_threshold * mad_interval * MAD_TO_STD
        adaptive_high = median_interval + timing_mad_threshold * mad_interval * MAD_TO_STD
        
        # Constrain to physiological limits
        outlier_low = max(min_breath_dur, adaptive_low)
        outlier_high = min(max_breath_dur, adaptive_high)
    else:
        # Fall back to fixed physiological limits
        outlier_low = min_breath_dur
        outlier_high = max_breath_dur
    
    intervals = []
    for i, (interval, t) in enumerate(zip(breath_intervals_s, breath_times_s)):
        if interval < outlier_low or interval > outlier_high:
            # Mark only the specific problematic cycle, not a wide zone
            # Use the cycle duration itself, with minimal margin
            start_s = max(0, t - 0.05)  # Very small margin before
            end_s = t + interval + 0.05  # Very small margin after
            intervals.append((start_s, end_s, "physiological_outlier"))
    
    # Don't merge - keep each outlier cycle separate
    merged = _merge_artifact_intervals(intervals, margin=ARTIFACT_MERGE_MARGIN_S)
    
    # Filter out artifacts shorter than minimum duration (very short threshold to allow single cycles)
    result = [(s, e, reason) for s, e, reason in merged if (e - s) >= MIN_ARTIFACT_DURATION_S]
    return result


def detect_amplitude_outliers(
    breath_amplitudes: np.ndarray,
    breath_times_s: np.ndarray,
    breath_intervals_s: np.ndarray,
    iqr_factor: float = AMPLITUDE_IQR_FACTOR,
    margin_s: float = 0.05,  # Very small margin - only mark the specific problematic cycle
) -> List[Tuple[float, float, str]]:
    """
    Detect cycles with outlier peak-to-trough amplitude.
    
    Rule: Amplitude outside IQR-based bounds
    
    Args:
        breath_amplitudes: Peak-to-trough amplitudes per cycle
        breath_times_s: Timestamps for each cycle
        breath_intervals_s: Duration of each cycle
        iqr_factor: IQR multiplier (default 2.5 = permissive)
        margin_s: Margin around detected outlier
    """
    if len(breath_amplitudes) < 4:
        return []
    
    low_bound, high_bound = compute_iqr_bounds(breath_amplitudes, factor=iqr_factor)
    
    intervals = []
    for i, (amp, t, dur) in enumerate(zip(breath_amplitudes, breath_times_s, breath_intervals_s)):
        if amp < low_bound or amp > high_bound:
            # Mark only the specific problematic cycle, not a wide zone
            start_s = max(0, t - margin_s)
            end_s = t + dur + margin_s
            intervals.append((start_s, end_s, "amplitude_outlier"))
    
    # Don't merge - keep each outlier cycle separate
    merged = _merge_artifact_intervals(intervals, margin=ARTIFACT_MERGE_MARGIN_S)
    
    # Filter out artifacts shorter than minimum duration (very short threshold to allow single cycles)
    result = [(s, e, reason) for s, e, reason in merged if (e - s) >= MIN_ARTIFACT_DURATION_S]
    return result


def detect_all_artifacts(
    rsp: np.ndarray,
    fs: float,
    breath_intervals_s: Optional[np.ndarray] = None,
    breath_times_s: Optional[np.ndarray] = None,
    breath_amplitudes: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[List[Tuple[float, float, str]], Dict[str, List]]:
    """
    Run all artifact detection algorithms (MAD-based for robustness).
    
    Detection criteria:
    1. Flat segments: Rolling std below threshold
    2. Clipping: Signal at extreme percentiles
    3. Step artifacts: High derivative (MAD-based threshold)
    4. Baseline shifts: Sudden change in rolling mean
    5. Baseline wander: Excessive low-frequency drift
    6. Tremor noise: High power in high-frequency band
    7. Physiological outliers: Implausible breath timing (adaptive bounds)
    8. Amplitude outliers: Extreme peak-to-trough amplitude (IQR-based)
    
    Returns:
        (all_artifacts, detections_by_type)
    """
    detections = {}
    all_artifacts = []
    
    # 1. Flat segments
    flat = detect_flat_segments(rsp, fs)
    detections['flat_segment'] = flat
    all_artifacts.extend(flat)
    
    # 2. Clipping
    clip = detect_clipping(rsp, fs)
    detections['clipping'] = clip
    all_artifacts.extend(clip)
    
    # 2b. Low-amplitude zones (possible loose belt / poor contact)
    low_amp = detect_low_amplitude_zones(rsp, fs)
    detections['low_amplitude'] = low_amp
    all_artifacts.extend(low_amp)
    
    # 3. Step artifacts (MAD-based)
    steps = detect_step_artifacts(rsp, fs, use_robust=True)
    detections['step_artifact'] = steps
    all_artifacts.extend(steps)
    
    # 4. Baseline shifts (NEW - rolling mean comparison)
    shifts = detect_baseline_shifts(rsp, fs)
    detections['baseline_shift'] = shifts
    all_artifacts.extend(shifts)
    
    # 5. Baseline wander (slow drift)
    wander = detect_baseline_wander(rsp, fs)
    detections['baseline_wander'] = wander
    all_artifacts.extend(wander)
    
    # 6. Tremor noise
    tremor = detect_tremor_noise(rsp, fs)
    detections['tremor_noise'] = tremor
    all_artifacts.extend(tremor)
    
    # 7. Physiological outliers - breath timing (adaptive bounds)
    if breath_intervals_s is not None and breath_times_s is not None and len(breath_intervals_s) > 0:
        physio = detect_physiological_outliers(breath_intervals_s, breath_times_s, use_adaptive=True)
        detections['physiological_outlier'] = physio
        all_artifacts.extend(physio)
    
    # 8. Amplitude outliers (NEW - IQR-based)
    if (breath_amplitudes is not None and breath_times_s is not None and 
        breath_intervals_s is not None and len(breath_amplitudes) > 0):
        amp_outliers = detect_amplitude_outliers(breath_amplitudes, breath_times_s, breath_intervals_s)
        detections['amplitude_outlier'] = amp_outliers
        all_artifacts.extend(amp_outliers)
    
    if verbose:
        total = len(all_artifacts)
        if total > 0:
            print(f"  ⚠️  {total} artifact zone(s) detected:")
            for atype, alist in detections.items():
                if len(alist) > 0:
                    print(f"     - {len(alist)} {atype}")
        else:
            print(f"  ✓ No artifacts detected")
    
    return all_artifacts, detections


def _bool_to_intervals(mask: np.ndarray, fs: float) -> List[Tuple[float, float]]:
    """Convert boolean mask to list of (start_s, end_s) intervals."""
    if not np.any(mask):
        return []
    
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    
    return [(s / fs, e / fs) for s, e in zip(starts, ends)]


def _merge_artifact_intervals(intervals: List[Tuple[float, float, str]], margin: float = ARTIFACT_MERGE_MARGIN_S) -> List[Tuple[float, float, str]]:
    """
    Merge overlapping artifact intervals.
    
    By default, margin=0.0 means intervals are NOT merged - each peak is kept separate.
    This allows marking only specific problematic peaks, not broad zones around them.
    """
    if not intervals:
        return []
    
    # Sort by start time
    intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = []
    cur_start, cur_end, cur_reasons = intervals[0]
    cur_reasons = {cur_reasons}
    
    for s, e, reason in intervals[1:]:
        if s <= cur_end + margin:
            cur_end = max(cur_end, e)
            cur_reasons.add(reason)
        else:
            merged.append((cur_start, cur_end, "+".join(sorted(cur_reasons))))
            cur_start, cur_end = s, e
            cur_reasons = {reason}
    
    merged.append((cur_start, cur_end, "+".join(sorted(cur_reasons))))
    return merged


# =============================================================================
# BREATHING CYCLE DETECTION
# =============================================================================

def detect_breathing_cycles(
    rsp_clean: np.ndarray,
    fs: float,
    method: str = "khodadad2018",
    use_fallback: bool = True,
    verbose: bool = False,
    min_amplitude_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect breathing cycles (peaks and troughs) with improved edge detection.
    
    Peak = exhalation onset (local maximum - chest at max expansion)
    Trough = inhalation onset (local minimum - chest at min)
    
    Uses primary method (khodadad2018) and optionally supplements with
    scipy find_peaks for edge regions where primary method may miss peaks.
    
    For Breath-Hold (BH) segments, use min_amplitude_threshold to filter out
    small noise fluctuations during the flat breath-hold period.
    
    Args:
        rsp_clean: Filtered respiration signal
        fs: Sampling frequency
        method: Primary detection method (default: "khodadad2018")
        use_fallback: If True, use scipy find_peaks to catch missed edge peaks
        verbose: If True, print diagnostic information
        min_amplitude_threshold: Optional minimum amplitude threshold. Cycles with
            amplitude below this threshold will be filtered out. Useful for BH segments
            to ignore small noise during breath-hold. If None, no filtering applied.
    
    Returns:
        (peaks, troughs) as arrays of sample indices
    """
    if len(rsp_clean) < 10:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    # Primary detection using NeuroKit2
    peaks_primary = np.array([], dtype=int)
    troughs_primary = np.array([], dtype=int)
    
    try:
        signals, info = nk.rsp_peaks(rsp_clean, sampling_rate=int(fs), method=method)
        peaks_primary = np.array(info.get("RSP_Peaks", []), dtype=int)
        troughs_primary = np.array(info.get("RSP_Troughs", []), dtype=int)
        
        # Filter invalid indices
        peaks_primary = peaks_primary[(peaks_primary >= 0) & (peaks_primary < len(rsp_clean))]
        troughs_primary = troughs_primary[(troughs_primary >= 0) & (troughs_primary < len(rsp_clean))]
        
        if verbose:
            print(f"    Primary ({method}): {len(peaks_primary)} peaks, {len(troughs_primary)} troughs")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Primary peak detection error: {e}")
        if not use_fallback:
            return np.array([], dtype=int), np.array([], dtype=int)
    
    # Fallback: Use scipy find_peaks for edge regions and to catch missed peaks
    if use_fallback and _HAVE_SCIPY:
        # Estimate minimum distance between peaks (breathing rate 4-60 bpm)
        min_breath_dur_samples = int(fs * (60.0 / MAX_BREATHS_PER_MIN))  # ~0.5s at 50Hz
        max_breath_dur_samples = int(fs * (60.0 / MIN_BREATHS_PER_MIN))   # ~15s at 50Hz
        
        # Detect peaks (exhalation - local maxima) with adaptive prominence
        std_est = np.std(rsp_clean) if np.std(rsp_clean) > 0 else 1e-6
        base_prom = std_est * 0.3
        peaks_fallback, props_peaks = scipy_signal.find_peaks(
            rsp_clean,
            distance=min_breath_dur_samples,
            prominence=base_prom,
        )

        # Detect troughs (inhalation - local minima)
        troughs_fallback, props_troughs = scipy_signal.find_peaks(
            -rsp_clean,  # Invert to find minima
            distance=min_breath_dur_samples,
            prominence=base_prom,
        )

        # Check peak widths and selectively accept fallback peaks that are reasonably wide
        try:
            widths_peaks = scipy_signal.peak_widths(rsp_clean, peaks_fallback, rel_height=0.5)[0]
        except Exception:
            widths_peaks = np.zeros_like(peaks_fallback, dtype=float)

        min_width_samples = max(1, int(0.15 * fs))  # require ~150 ms width for robust peaks
        selected_peaks = []
        for p_idx, width in zip(peaks_fallback, widths_peaks):
            prom = props_peaks.get("prominences", np.zeros_like(peaks_fallback))[np.where(peaks_fallback == p_idx)[0][0]] if "prominences" in props_peaks else std_est * 0.0
            if width >= min_width_samples or prom >= std_est * 0.45:
                selected_peaks.append(p_idx)
        peaks_fallback = np.array(selected_peaks, dtype=int)

        # For troughs, do the same using inverted signal
        try:
            widths_troughs = scipy_signal.peak_widths(-rsp_clean, troughs_fallback, rel_height=0.5)[0]
        except Exception:
            widths_troughs = np.zeros_like(troughs_fallback, dtype=float)

        selected_troughs = []
        for t_idx, width in zip(troughs_fallback, widths_troughs):
            prom = props_troughs.get("prominences", np.zeros_like(troughs_fallback))[np.where(troughs_fallback == t_idx)[0][0]] if "prominences" in props_troughs else std_est * 0.0
            if width >= min_width_samples or prom >= std_est * 0.45:
                selected_troughs.append(t_idx)
        troughs_fallback = np.array(selected_troughs, dtype=int)
        
        # Merge primary and fallback results
        # Priority: Keep primary detections, add fallback only if they're:
        # 1. Near signal edges (first/last 20% of signal)
        # 2. Not too close to existing primary detections
        
        edge_margin = int(len(rsp_clean) * 0.2)  # 20% of signal
        
        # Merge peaks: accept fallback if near edge, primary empty, or if not too close and selected by width/prominence
        all_peaks = set(peaks_primary.tolist())
        for p in peaks_fallback:
            is_near_edge = (p < edge_margin) or (p > len(rsp_clean) - edge_margin)
            too_close = any(abs(p - existing) < min_breath_dur_samples // 2 for existing in all_peaks)
            if (is_near_edge or len(peaks_primary) == 0 or not too_close) and not too_close:
                all_peaks.add(p)
        
        # Merge troughs
        all_troughs = set(troughs_primary.tolist())
        for t in troughs_fallback:
            is_near_edge = (t < edge_margin) or (t > len(rsp_clean) - edge_margin)
            too_close = any(abs(t - existing) < min_breath_dur_samples // 2 for existing in all_troughs)
            if (is_near_edge or len(troughs_primary) == 0 or not too_close) and not too_close:
                all_troughs.add(t)
        
        peaks = np.array(sorted(all_peaks), dtype=int)
        troughs = np.array(sorted(all_troughs), dtype=int)
        
        # Ensure alternating pattern (trough, peak, trough, peak...)
        # Sort all extrema and assign based on value
        if len(peaks) > 0 and len(troughs) > 0:
            peaks, troughs = _ensure_alternating_pattern(rsp_clean, peaks, troughs)
        
        if verbose:
            added_peaks = len(peaks) - len(peaks_primary)
            added_troughs = len(troughs) - len(troughs_primary)
            if added_peaks > 0 or added_troughs > 0:
                print(f"    Fallback (scipy): +{added_peaks} peaks, +{added_troughs} troughs (edge detection)")
    else:
        peaks = peaks_primary
        troughs = troughs_primary
    
    # Final validation: remove any that are too close together
    if len(peaks) > 1:
        min_dist = int(fs * 0.5)  # Minimum 0.5s between peaks
        keep_peaks = [peaks[0]]
        for p in peaks[1:]:
            if p - keep_peaks[-1] >= min_dist:
                keep_peaks.append(p)
        peaks = np.array(keep_peaks, dtype=int)
    
    if len(troughs) > 1:
        min_dist = int(fs * 0.5)
        keep_troughs = [troughs[0]]
        for t in troughs[1:]:
            if t - keep_troughs[-1] >= min_dist:
                keep_troughs.append(t)
        troughs = np.array(keep_troughs, dtype=int)
    
    # Filter by amplitude threshold (for BH segments)
    if min_amplitude_threshold is not None and len(peaks) > 0 and len(troughs) > 0:
        # Calculate amplitude for each breathing cycle (trough to peak)
        # and filter out cycles with amplitude below threshold
        valid_peaks = set()
        valid_troughs = set()
        
        # For each trough, find the nearest peak after it
        for t_idx in troughs:
            t_val = rsp_clean[t_idx]
            
            # Find next peak after this trough
            peaks_after = peaks[peaks > t_idx]
            if len(peaks_after) > 0:
                next_peak_idx = peaks_after[0]
                next_peak_val = rsp_clean[next_peak_idx]
                
                # Calculate amplitude for this cycle
                amplitude = abs(next_peak_val - t_val)
                
                # Keep if amplitude >= threshold
                if amplitude >= min_amplitude_threshold:
                    valid_troughs.add(t_idx)
                    valid_peaks.add(next_peak_idx)
        
        # Also check peaks that don't have a preceding trough (edge cases)
        for p_idx in peaks:
            if p_idx not in valid_peaks:
                # Find nearest trough before this peak
                troughs_before = troughs[troughs < p_idx]
                if len(troughs_before) > 0:
                    prev_trough_idx = troughs_before[-1]
                    prev_trough_val = rsp_clean[prev_trough_idx]
                    p_val = rsp_clean[p_idx]
                    amplitude = abs(p_val - prev_trough_val)
                    
                    if amplitude >= min_amplitude_threshold:
                        valid_peaks.add(p_idx)
                        valid_troughs.add(prev_trough_idx)
        
        peaks = np.array(sorted(valid_peaks), dtype=int)
        troughs = np.array(sorted(valid_troughs), dtype=int)
        
        if verbose:
            original_count = len(peaks_primary) + len(troughs_primary)
            filtered_count = original_count - (len(peaks) + len(troughs))
            if filtered_count > 0:
                print(f"    Amplitude filter: Removed {filtered_count} small cycles (threshold: {min_amplitude_threshold:.3f})")
    
    return peaks, troughs


def _ensure_alternating_pattern(
    signal: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure peaks and troughs alternate properly.
    Removes any that break the pattern.
    """
    if len(peaks) == 0 or len(troughs) == 0:
        return peaks, troughs
    
    # Combine and sort by position
    all_extrema = []
    for p in peaks:
        all_extrema.append((p, 'peak', signal[p]))
    for t in troughs:
        all_extrema.append((t, 'trough', signal[t]))
    
    all_extrema.sort(key=lambda x: x[0])
    
    # Rebuild alternating pattern
    new_peaks = []
    new_troughs = []
    
    # Start with the first extrema
    if all_extrema[0][1] == 'trough':
        new_troughs.append(all_extrema[0][0])
        expect_peak = True
    else:
        new_peaks.append(all_extrema[0][0])
        expect_peak = False
    
    for idx, ext_type, value in all_extrema[1:]:
        if expect_peak and ext_type == 'peak':
            new_peaks.append(idx)
            expect_peak = False
        elif not expect_peak and ext_type == 'trough':
            new_troughs.append(idx)
            expect_peak = True
        # Skip if pattern is broken (e.g., two peaks in a row)
    
    return np.array(new_peaks, dtype=int), np.array(new_troughs, dtype=int)


def calculate_breathing_metrics(
    rsp_clean: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    fs: float,
) -> Dict[str, np.ndarray]:
    """Calculate breathing pattern metrics."""
    metrics = {}
    
    # Breath intervals (trough to trough)
    if len(troughs) > 1:
        breath_intervals_s = np.diff(troughs) / fs
        breath_times_s = troughs[:-1] / fs
        metrics["breath_intervals_s"] = breath_intervals_s
        metrics["breath_times_s"] = breath_times_s
        metrics["resp_rate_bpm"] = 60.0 / breath_intervals_s
    else:
        metrics["breath_intervals_s"] = np.array([])
        metrics["breath_times_s"] = np.array([])
        metrics["resp_rate_bpm"] = np.array([])
    
    # Inspiration/expiration durations and I:E ratio
    inspiration_durations = []
    expiration_durations = []
    ie_ratios = []
    breath_amplitudes = []
    
    for i in range(len(troughs) - 1):
        trough_start = troughs[i]
        trough_end = troughs[i + 1]
        
        peaks_between = peaks[(peaks > trough_start) & (peaks < trough_end)]
        
        if len(peaks_between) > 0:
            peak = peaks_between[0]
            insp_dur = (peak - trough_start) / fs
            exp_dur = (trough_end - peak) / fs
            
            inspiration_durations.append(insp_dur)
            expiration_durations.append(exp_dur)
            
            if exp_dur > 0:
                ie_ratios.append(insp_dur / exp_dur)
            
            amplitude = rsp_clean[peak] - rsp_clean[trough_start]
            breath_amplitudes.append(amplitude)
    
    metrics["inspiration_durations"] = np.array(inspiration_durations)
    metrics["expiration_durations"] = np.array(expiration_durations)
    metrics["ie_ratios"] = np.array(ie_ratios)
    metrics["breath_amplitudes"] = np.array(breath_amplitudes)
    
    return metrics


def calculate_phase_signal(peaks: np.ndarray, troughs: np.ndarray, desired_length: int) -> np.ndarray:
    """Calculate respiratory phase (1=inspiration, 0=expiration)."""
    try:
        phase_df = nk.rsp_phase(peaks=peaks, troughs=troughs, desired_length=desired_length)
        return phase_df["RSP_Phase"].values
    except:
        return np.zeros(desired_length)


# =============================================================================
# INTERPOLATION FOR SHORT ARTIFACTS
# =============================================================================

def interpolate_artifact(
    signal: np.ndarray,
    fs: float,
    start_s: float,
    end_s: float,
    method: str = INTERP_METHOD,
    max_cycles: int = MAX_INTERP_CYCLES,
    typical_breath_dur: float = TYPICAL_BREATH_DURATION_S,
    verbose: bool = False,
) -> Tuple[np.ndarray, bool]:
    """
    Interpolate short artifact if within acceptable duration.
    
    Args:
        signal: Full signal array
        fs: Sampling frequency
        start_s, end_s: Artifact boundaries in seconds
        method: "linear" or "cubic"
        max_cycles: Maximum breathing cycles to interpolate
        typical_breath_dur: Expected breath duration for cycle counting
        verbose: If True, print reasons when interpolation is skipped or falls back
    
    Returns:
        (interpolated_signal, was_interpolated)
    """
    duration_s = end_s - start_s
    estimated_cycles = duration_s / typical_breath_dur
    
    if estimated_cycles > max_cycles:
        if verbose:
            print(
                f"  ⚠️ Interpolation skipped [{start_s:.1f}s - {end_s:.1f}s]: "
                f"~{estimated_cycles:.1f} cycles exceeds max ({max_cycles})"
            )
        return signal, False  # Too long, don't interpolate
    
    start_idx = int(start_s * fs)
    end_idx = int(end_s * fs)
    
    # Ensure valid indices
    start_idx = max(0, start_idx)
    end_idx = min(len(signal), end_idx)
    
    if end_idx - start_idx < 2:
        if verbose:
            print(
                f"  ⚠️ Artifact window too short for interpolation [{start_s:.1f}s - {end_s:.1f}s]"
            )
        return signal, False
    
    signal_interp = signal.copy()
    
    # Get boundary values for interpolation
    # Use at least one full breathing cycle before and after for better interpolation
    margin_s = max(typical_breath_dur, 2.0)  # At least 1 cycle or 2s, whichever is larger
    margin = int(margin_s * fs)
    left_idx = max(0, start_idx - margin)
    right_idx = min(len(signal), end_idx + margin)
    
    # Create interpolation points (exclude artifact zone)
    # Use indices before and after the bad zone
    idx_before = np.arange(left_idx, start_idx)
    idx_after = np.arange(end_idx, right_idx)
    
    # Need points on both sides for proper interpolation
    if len(idx_before) == 0 or len(idx_after) == 0:
        if verbose:
            print(
                f"  ⚠️ Not enough interpolation anchors for [{start_s:.1f}s - {end_s:.1f}s]: "
                f"Missing points on one side (before: {len(idx_before)}, after: {len(idx_after)})"
            )
        return signal, False
    
    # Combine known points
    x_known_idx = np.concatenate([idx_before, idx_after])
    y_known = signal[x_known_idx]
    
    # Create timestamps for known points (for time-based interpolation)
    times_known = x_known_idx / fs
    times_interp = np.arange(start_idx, end_idx) / fs
    
    if len(x_known_idx) < 4:
        if verbose:
            print(
                f"  ⚠️ Not enough interpolation anchors for [{start_s:.1f}s - {end_s:.1f}s]: "
                f"{len(x_known_idx)} known points → relying on masked signal"
            )
        return signal, False
    
    # Interpolate using time-based interpolation (more robust for irregular sampling)
    def _apply_np_interp():
        # Use linear interpolation based on timestamps
        signal_interp[start_idx:end_idx] = np.interp(times_interp, times_known, y_known)

    try:
        if method == "monotone_cubic" and _HAVE_SCIPY and PchipInterpolator is not None and len(x_known_idx) >= 4:
            # Use PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) - monotone cubic
            # Prevents overshoots/undershoots (NeuroKit2 recommendation)
            pchip = PchipInterpolator(times_known, y_known)
            signal_interp[start_idx:end_idx] = pchip(times_interp)
        elif method == "cubic" and _HAVE_SCIPY and len(x_known_idx) >= 4:
            # Use cubic spline interpolation based on timestamps
            cs = CubicSpline(times_known, y_known, bc_type='natural')
            signal_interp[start_idx:end_idx] = cs(times_interp)
        elif _HAVE_SCIPY:
            # Use linear interpolation based on timestamps
            f = interp1d(times_known, y_known, kind='linear', 
                        fill_value='extrapolate', bounds_error=False)
            signal_interp[start_idx:end_idx] = f(times_interp)
        else:
            _apply_np_interp()
    except Exception as err:
        if verbose:
            print(
                f"  ⚠️ Interpolation error for [{start_s:.1f}s - {end_s:.1f}s]: {err} → np.interp fallback applied"
            )
        _apply_np_interp()
    
    # Verify interpolation quality: check if interpolated values are reasonable
    # (not NaN, not infinite, within reasonable range)
    interp_values = signal_interp[start_idx:end_idx]
    if np.any(np.isnan(interp_values)) or np.any(np.isinf(interp_values)):
        if verbose:
            print(
                f"  ⚠️ Interpolation produced invalid values for [{start_s:.1f}s - {end_s:.1f}s] → using fallback"
            )
        # Fallback: simple linear interpolation between boundary values
        if len(idx_before) > 0 and len(idx_after) > 0:
            val_before = signal[start_idx - 1] if start_idx > 0 else signal[idx_before[-1]]
            val_after = signal[end_idx] if end_idx < len(signal) else signal[idx_after[0]]
            n_points = end_idx - start_idx
            signal_interp[start_idx:end_idx] = np.linspace(val_before, val_after, n_points)
        else:
            return signal, False
    
    return signal_interp, True


def apply_interpolation_to_artifacts(
    signal: np.ndarray,
    fs: float,
    artifacts: List[Tuple[float, float, str]],
    max_cycles: int = MAX_INTERP_CYCLES,
    typical_breath_dur: float = TYPICAL_BREATH_DURATION_S,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[Tuple[float, float]], List[Tuple[float, float, str]]]:
    """
    Apply interpolation to short artifacts, leave long ones as invalid.
    
    Returns:
        (interpolated_signal, interpolated_intervals, remaining_invalid)
    """
    signal_out = signal.copy()
    interpolated = []
    remaining = []
    
    for start_s, end_s, reason in artifacts:
        duration_s = end_s - start_s
        estimated_cycles = duration_s / typical_breath_dur
        
        if estimated_cycles <= max_cycles:
            signal_out, was_interp = interpolate_artifact(
                signal_out, fs, start_s, end_s,
                max_cycles=max_cycles,
                typical_breath_dur=typical_breath_dur,
                verbose=verbose
            )
            if was_interp:
                interpolated.append((start_s, end_s))
                if verbose:
                    print(f"  ✓ Interpolated [{start_s:.1f}s - {end_s:.1f}s] ({duration_s:.1f}s, ~{estimated_cycles:.1f} cycles)")
            else:
                remaining.append((start_s, end_s, reason))
        else:
            remaining.append((start_s, end_s, reason))
            if verbose:
                print(f"  ✗ Too long to interpolate [{start_s:.1f}s - {end_s:.1f}s] ({duration_s:.1f}s, ~{estimated_cycles:.1f} cycles > {max_cycles})")
    
    return signal_out, interpolated, remaining


# =============================================================================
# BAD INTERVAL MANAGEMENT
# =============================================================================

def marcar_bad_interval(seg_dict: dict, t_ini: float, t_fin: float,
                        motivo: str = "manual", sev: float = 1.0) -> dict:
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


def agregar_bad_zone(seg_dict: dict, t_ini: float, t_fin: float, motivo: str = "manual") -> dict:
    """Add a new BAD zone (alias for marcar_bad_interval)."""
    seg_dict = marcar_bad_interval(seg_dict, t_ini, t_fin, motivo=motivo)
    print(f"  ✓ Added BAD zone [{t_ini:.1f}s - {t_fin:.1f}s] ({motivo})")
    return seg_dict


def eliminar_bad_zones(seg_dict: dict, zone_ids: List[int]) -> dict:
    """
    Remove BAD zones by their ID numbers.
    
    Args:
        seg_dict: Segment dictionary
        zone_ids: List of zone IDs to remove (1-based, as shown in plots)
    """
    bad_consolidated = consolidar_bad_intervals(seg_dict)
    if len(bad_consolidated) == 0:
        print("  No BAD zones to remove")
        return seg_dict
    
    # Get intervals to remove
    to_remove = bad_consolidated[bad_consolidated['id'].isin(zone_ids)]
    
    if len(to_remove) == 0:
        print(f"  No zones found with IDs: {zone_ids}")
        return seg_dict
    
    # Remove from bad_intervals DataFrame
    bi = seg_dict.get("bad_intervals", pd.DataFrame())
    if len(bi) == 0:
        return seg_dict
    
    for _, row in to_remove.iterrows():
        t_ini = row['t_ini_s']
        t_fin = row['t_fin_s']
        # Remove overlapping intervals
        mask = ~((bi['t_ini_s'] >= t_ini - 0.1) & (bi['t_fin_s'] <= t_fin + 0.1))
        bi = bi[mask]
        print(f"  ✓ Removed BAD zone [{row['id']}]: [{t_ini:.1f}s - {t_fin:.1f}s]")
    
    seg_dict["bad_intervals"] = bi.reset_index(drop=True)
    return seg_dict


def modificar_zona_bad(seg_dict: dict, zona_id: int, nuevo_t_ini: Optional[float] = None,
                       nuevo_t_fin: Optional[float] = None) -> dict:
    """
    Modify boundaries of an existing BAD zone.
    
    Args:
        seg_dict: Segment dictionary
        zona_id: Zone ID to modify (1-based)
        nuevo_t_ini: New start time (None to keep current)
        nuevo_t_fin: New end time (None to keep current)
    """
    bad_consolidated = consolidar_bad_intervals(seg_dict)
    if len(bad_consolidated) == 0:
        print("  No BAD zones to modify")
        return seg_dict
    
    zone_row = bad_consolidated[bad_consolidated['id'] == zona_id]
    if len(zone_row) == 0:
        print(f"  Zone ID {zona_id} not found")
        return seg_dict
    
    old_t_ini = zone_row.iloc[0]['t_ini_s']
    old_t_fin = zone_row.iloc[0]['t_fin_s']
    
    # Remove old zone
    seg_dict = eliminar_bad_zones(seg_dict, [zona_id])
    
    # Add modified zone
    new_t_ini = nuevo_t_ini if nuevo_t_ini is not None else old_t_ini
    new_t_fin = nuevo_t_fin if nuevo_t_fin is not None else old_t_fin
    seg_dict = marcar_bad_interval(seg_dict, new_t_ini, new_t_fin, motivo="modified")
    
    print(f"  ✓ Modified zone {zona_id}: [{old_t_ini:.1f}-{old_t_fin:.1f}] → [{new_t_ini:.1f}-{new_t_fin:.1f}]")
    return seg_dict


def reset_bad_intervals(seg_dict: dict) -> dict:
    """Clear all BAD intervals."""
    seg_dict["bad_intervals"] = pd.DataFrame(columns=["t_ini_s", "t_fin_s", "motivo", "sev"])
    print("  ✓ All BAD intervals cleared")
    return seg_dict


def consolidar_bad_intervals(seg_dict: dict) -> pd.DataFrame:
    """Consolidate and number BAD intervals."""
    bi = seg_dict.get("bad_intervals", pd.DataFrame())
    if bi is None or len(bi) == 0:
        return pd.DataFrame(columns=["id", "t_ini_s", "t_fin_s", "severity", "motivos", "dur_s"])
    
    fs = float(seg_dict.get("fs_rsp", 50))
    dur_s = len(seg_dict.get("rsp_clean", [])) / fs if len(seg_dict.get("rsp_clean", [])) > 0 else 300
    
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
        if s <= cur_end:
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


def calcular_bad_pct(seg_dict: dict) -> Tuple[float, float, float]:
    """Calculate percentage of segment marked as BAD."""
    fs = float(seg_dict.get("fs_rsp", 50))
    n = len(seg_dict.get("rsp_clean", []))
    if n == 0 or fs <= 0:
        return np.nan, np.nan, np.nan
    dur_s = n / fs
    
    bi = seg_dict.get("bad_intervals", pd.DataFrame())
    if bi is None or len(bi) == 0:
        return 0.0, 0.0, dur_s
    
    # Merge and calculate total bad time
    bad_consolidated = consolidar_bad_intervals(seg_dict)
    if len(bad_consolidated) == 0:
        return 0.0, 0.0, dur_s
    
    bad_s = bad_consolidated['dur_s'].sum()
    bad_pct = bad_s / dur_s if dur_s > 0 else 0
    return bad_pct, bad_s, dur_s


# =============================================================================
# BREATH-HOLD END DETECTION & MARKING
# =============================================================================

def detectar_fin_breath_hold(
    seg_dict: dict,
    min_flat_duration_s: float = BH_FLAT_PERIOD_MIN_DURATION_S,
    recovery_amplitude_fraction: float = BH_RECOVERY_BREATH_AMPLITUDE_FRACTION,
    verbose: bool = True,
) -> Optional[float]:
    """
    Automatically detect when breath-hold ends (first recovery breath).
    
    Algorithm (tries multiple methods):
    1. Use detected peaks/troughs: First significant peak/trough after flat period
    2. Fallback: Find flat period, then detect first large amplitude change
    3. Fallback: Use first large peak/trough in entire signal
    
    Args:
        seg_dict: Segment dictionary
        min_flat_duration_s: Minimum duration of flat period to consider as breath-hold
        recovery_amplitude_fraction: Minimum amplitude (as fraction of signal range) for recovery breath
        verbose: Print diagnostic information
    
    Returns:
        Time in seconds of first recovery breath, or None if not detected
    """
    rsp_clean = seg_dict.get("rsp_clean", np.array([]))
    times = seg_dict.get("times", np.array([]))
    peaks = seg_dict.get("peaks", np.array([]))
    troughs = seg_dict.get("troughs", np.array([]))
    fs = seg_dict.get("fs_rsp", 50)
    
    if len(rsp_clean) < 10:
        return None
    
    signal_range = np.ptp(rsp_clean)
    recovery_threshold = signal_range * recovery_amplitude_fraction
    
    # Method 1: Use detected peaks/troughs (most reliable if available)
    if len(peaks) > 0 or len(troughs) > 0:
        # Find flat period first
        window_samples = int(fs * 2.0)
        if window_samples < len(rsp_clean):
            rolling_var = np.zeros(len(rsp_clean))
            for i in range(window_samples, len(rsp_clean)):
                rolling_var[i] = np.var(rsp_clean[i-window_samples:i])
            
            flat_threshold = np.percentile(rolling_var, 30)  # Bottom 30% variance (more permissive)
            is_flat = rolling_var < flat_threshold
            
            # Find end of flat period
            flat_end_idx = None
            for i in range(len(is_flat) - 1):
                if is_flat[i] and not is_flat[i+1]:
                    flat_start = i
                    while flat_start > 0 and is_flat[flat_start-1]:
                        flat_start -= 1
                    flat_duration = (i - flat_start) / fs
                    if flat_duration >= min_flat_duration_s:
                        flat_end_idx = i
                        break
            
            # Find first peak or trough after flat period
            all_extrema = []
            for p in peaks:
                all_extrema.append((p, 'peak', rsp_clean[p], times[p]))
            for t in troughs:
                all_extrema.append((t, 'trough', rsp_clean[t], times[t]))
            
            if len(all_extrema) > 0:
                all_extrema.sort(key=lambda x: x[0])  # Sort by index
                
                # Find first extrema after flat period
                if flat_end_idx is not None:
                    for idx, ext_type, val, t in all_extrema:
                        if idx > flat_end_idx:
                            # Check if amplitude is significant
                            # For trough: check distance to nearby peak
                            # For peak: check distance to nearby trough
                            if ext_type == 'trough':
                                # Find next peak
                                next_peaks = [p for p in peaks if p > idx]
                                if len(next_peaks) > 0:
                                    next_peak_idx = next_peaks[0]
                                    amplitude = abs(rsp_clean[next_peak_idx] - val)
                                    if amplitude >= recovery_threshold:
                                        t_adjusted = _ajustar_a_extremo_mas_cercano(seg_dict, t)
                                        if verbose:
                                            if abs(t_adjusted - t) > 0.1:
                                                print(f"  ✓ Detected breath-hold end at {t_adjusted:.1f}s (adjusted from {t:.1f}s to nearest {ext_type})")
                                            else:
                                                print(f"  ✓ Detected breath-hold end at {t_adjusted:.1f}s (first recovery {ext_type} after flat period)")
                                        return t_adjusted
                            else:  # peak
                                # Find previous trough
                                prev_troughs = [t for t in troughs if t < idx]
                                if len(prev_troughs) > 0:
                                    prev_trough_idx = prev_troughs[-1]
                                    amplitude = abs(val - rsp_clean[prev_trough_idx])
                                    if amplitude >= recovery_threshold:
                                        t_adjusted = _ajustar_a_extremo_mas_cercano(seg_dict, t)
                                        if verbose:
                                            if abs(t_adjusted - t) > 0.1:
                                                print(f"  ✓ Detected breath-hold end at {t_adjusted:.1f}s (adjusted from {t:.1f}s to nearest {ext_type})")
                                            else:
                                                print(f"  ✓ Detected breath-hold end at {t_adjusted:.1f}s (first recovery {ext_type} after flat period)")
                                        return t_adjusted
                
                # Fallback: Use first large extrema in signal (if no flat period detected)
                if flat_end_idx is None:
                    for idx, ext_type, val, t in all_extrema:
                        # Check amplitude relative to signal range
                        if ext_type == 'trough':
                            next_peaks = [p for p in peaks if p > idx]
                            if len(next_peaks) > 0:
                                amplitude = abs(rsp_clean[next_peaks[0]] - val)
                            else:
                                continue
                        else:
                            prev_troughs = [t for t in troughs if t < idx]
                            if len(prev_troughs) > 0:
                                amplitude = abs(val - rsp_clean[prev_troughs[-1]])
                            else:
                                continue
                        
                        if amplitude >= recovery_threshold * 0.5:  # More permissive (50% of threshold)
                            t_adjusted = _ajustar_a_extremo_mas_cercano(seg_dict, t)
                            if verbose:
                                if abs(t_adjusted - t) > 0.1:
                                    print(f"  ✓ Detected breath-hold end at {t_adjusted:.1f}s (adjusted from {t:.1f}s to nearest {ext_type}, amplitude={amplitude:.3f})")
                                else:
                                    print(f"  ✓ Detected breath-hold end at {t_adjusted:.1f}s (first large {ext_type}, amplitude={amplitude:.3f})")
                            return t_adjusted
    
    # Method 2: Fallback - detect by signal variance and amplitude
    window_samples = int(fs * 2.0)
    if window_samples >= len(rsp_clean):
        if verbose:
            print("  ⚠️  Signal too short for breath-hold detection")
        return None
    
    rolling_var = np.zeros(len(rsp_clean))
    for i in range(window_samples, len(rsp_clean)):
        rolling_var[i] = np.var(rsp_clean[i-window_samples:i])
    
    # More permissive flat threshold
    flat_threshold = np.percentile(rolling_var, 30)  # Bottom 30%
    is_flat = rolling_var < flat_threshold
    
    # Find flat period end
    flat_end_idx = None
    for i in range(len(is_flat) - 1):
        if is_flat[i] and not is_flat[i+1]:
            flat_start = i
            while flat_start > 0 and is_flat[flat_start-1]:
                flat_start -= 1
            flat_duration = (i - flat_start) / fs
            if flat_duration >= min_flat_duration_s:
                flat_end_idx = i
                break
    
    if flat_end_idx is None:
        if verbose:
            print("  ⚠️  Could not detect breath-hold end automatically (no clear flat period found)")
        return None
    
    # Find first significant breath after flat period (more permissive)
    search_start = flat_end_idx
    search_end = min(flat_end_idx + int(fs * 30), len(rsp_clean))
    
    # Use a sliding window to find first large amplitude
    # First try derivative-based detection: a sudden increase in |diff| after flat period
    diff = np.abs(np.diff(rsp_clean))
    if len(diff) > 0:
        deriv_thr = max(STEP_ABS_THRESHOLD, compute_robust_threshold(diff, k=4.0))
        for j in range(search_start, min(search_end, len(diff))):
            if diff[j] >= deriv_thr:
                recovery_time = times[min(j + 1, len(times)-1)]
                recovery_time_adjusted = _ajustar_a_extremo_mas_cercano(seg_dict, recovery_time)
                if verbose:
                    if abs(recovery_time_adjusted - recovery_time) > 0.1:
                        print(f"  ✓ Detected breath-hold end at {recovery_time_adjusted:.1f}s (adjusted from {recovery_time:.1f}s, derivative spike, val={diff[j]:.3f} >= thr={deriv_thr:.3f})")
                    else:
                        print(f"  ✓ Detected breath-hold end at {recovery_time_adjusted:.1f}s (derivative spike, val={diff[j]:.3f} >= thr={deriv_thr:.3f})")
                return recovery_time_adjusted

    # Fallback: sliding-window amplitude check (3s window)
    for i in range(search_start, search_end - int(fs * 2)):
        window = rsp_clean[i:min(i+int(fs*3), len(rsp_clean))]  # 3-second window
        if len(window) > 0:
            window_amplitude = np.ptp(window)
            # More permissive: 50% of original threshold
            if window_amplitude >= recovery_threshold * 0.5:
                recovery_time = times[i]
                recovery_time_adjusted = _ajustar_a_extremo_mas_cercano(seg_dict, recovery_time)
                if verbose:
                    if abs(recovery_time_adjusted - recovery_time) > 0.1:
                        print(f"  ✓ Detected breath-hold end at {recovery_time_adjusted:.1f}s (adjusted from {recovery_time:.1f}s, first recovery breath, amplitude={window_amplitude:.3f})")
                    else:
                        print(f"  ✓ Detected breath-hold end at {recovery_time_adjusted:.1f}s (first recovery breath, amplitude={window_amplitude:.3f})")
                return recovery_time_adjusted
    
    if verbose:
        print("  ⚠️  Found flat period but no significant recovery breath detected")
    return None


def _ajustar_a_extremo_mas_cercano(seg_dict: dict, t_target: float) -> float:
    """
    Helper function to adjust a time to the nearest peak or trough.
    
    Args:
        seg_dict: Segment dictionary
        t_target: Target time in seconds
    
    Returns:
        Adjusted time (nearest peak or trough)
    """
    peaks = seg_dict.get("peaks", np.array([]))
    troughs = seg_dict.get("troughs", np.array([]))
    fs = seg_dict.get("fs_rsp", 50)
    
    if len(peaks) == 0 and len(troughs) == 0:
        return t_target
    
    # Find nearest peak
    if len(peaks) > 0:
        peak_times = peaks / fs
        peak_dists = np.abs(peak_times - t_target)
        nearest_peak_time = peak_times[np.argmin(peak_dists)]
        nearest_peak_dist = peak_dists[np.argmin(peak_dists)]
    else:
        nearest_peak_time = None
        nearest_peak_dist = np.inf
    
    # Find nearest trough
    if len(troughs) > 0:
        trough_times = troughs / fs
        trough_dists = np.abs(trough_times - t_target)
        nearest_trough_time = trough_times[np.argmin(trough_dists)]
        nearest_trough_dist = trough_dists[np.argmin(trough_dists)]
    else:
        nearest_trough_time = None
        nearest_trough_dist = np.inf
    
    # Choose the closest one
    if nearest_peak_dist < nearest_trough_dist and nearest_peak_time is not None:
        return nearest_peak_time
    elif nearest_trough_time is not None:
        return nearest_trough_time
    
    return t_target


def marcar_fin_breath_hold(seg_dict: dict, t_fin_bh: float) -> dict:
    """
    Manually mark the end of breath-hold (first recovery breath time).
    
    This marker will be used to trim the segment for HRV analysis,
    preventing recovery breathing from affecting metrics.
    
    Uses the EXACT time specified by the user (no automatic adjustment).
    For automatic detection that adjusts to nearest peak/trough, use detectar_fin_breath_hold().
    
    Args:
        seg_dict: Segment dictionary
        t_fin_bh: Time in seconds when breath-hold ends (exact time, no adjustment)
    
    Returns:
        Updated seg_dict with bh_end_time_s marker
    """
    dur = seg_dict.get("dur_s", 0)
    t_fin_bh = max(0.0, min(float(t_fin_bh), dur))
    
    seg_dict["bh_end_time_s"] = t_fin_bh
    seg_name = seg_dict.get("seg", "SEG")
    print(f"  ✓ Marked breath-hold end for {seg_name} at {t_fin_bh:.1f}s (manual, exact time)")
    
    return seg_dict


def crear_version_recortada_bh(seg_dict: dict, verbose: bool = True) -> Optional[dict]:
    """
    Create a trimmed copy of BH segment at the breath-hold end time.
    
    Does NOT modify the original seg_dict. Returns a new trimmed copy.
    This version excludes recovery breathing for HRV analysis.
    
    Args:
        seg_dict: Segment dictionary (must have bh_end_time_s)
        verbose: Print trimming information
    
    Returns:
        New trimmed seg_dict copy, or None if bh_end_time_s not set
    """
    bh_end_time = seg_dict.get("bh_end_time_s", None)
    if bh_end_time is None:
        if verbose:
            print("  ⚠️  No breath-hold end marker found. Skipping trimmed version.")
        return None
    
    # Create a deep copy to avoid modifying original
    seg_trimmed = copy.deepcopy(seg_dict)
    
    times = seg_trimmed.get("times", np.array([]))
    fs = seg_trimmed.get("fs_rsp", 50)
    
    if len(times) == 0:
        return None
    
    # Find index corresponding to bh_end_time
    target_idx = int(bh_end_time * fs)
    target_idx = min(target_idx, len(times))
    
    original_dur = seg_trimmed.get("dur_s", len(times) / fs)
    trimmed_dur = times[target_idx] if target_idx < len(times) else original_dur
    
    if trimmed_dur >= original_dur:
        if verbose:
            print(f"  ℹ️  Segment already ends before breath-hold end marker ({bh_end_time:.1f}s)")
        return None
    
    # Trim all arrays in the copy
    seg_trimmed["times"] = times[:target_idx].copy()
    seg_trimmed["rsp_raw"] = seg_trimmed.get("rsp_raw", np.array([]))[:target_idx].copy()
    seg_trimmed["rsp_clean"] = seg_trimmed.get("rsp_clean", np.array([]))[:target_idx].copy()
    if "rsp_interp" in seg_trimmed:
        seg_trimmed["rsp_interp"] = seg_trimmed["rsp_interp"][:target_idx].copy()
    if "rsp_masked" in seg_trimmed:
        seg_trimmed["rsp_masked"] = seg_trimmed["rsp_masked"][:target_idx].copy()
    seg_trimmed["phase_signal"] = seg_trimmed.get("phase_signal", np.array([]))[:target_idx].copy()
    
    # Update peaks/troughs
    peaks = seg_trimmed.get("peaks", np.array([]))
    troughs = seg_trimmed.get("troughs", np.array([]))
    seg_trimmed["peaks"] = peaks[peaks < target_idx].copy()
    seg_trimmed["troughs"] = troughs[troughs < target_idx].copy()
    
    # Recalculate metrics for trimmed version
    if len(seg_trimmed["troughs"]) > 1:
        metrics = calculate_breathing_metrics(
            seg_trimmed["rsp_clean"], seg_trimmed["peaks"], seg_trimmed["troughs"], fs
        )
        seg_trimmed.update(metrics)
    
    seg_trimmed["dur_s"] = trimmed_dur
    seg_trimmed["dur_s_full"] = original_dur  # Store original duration for reference
    seg_trimmed["is_trimmed"] = True  # Flag to identify trimmed version
    
    if verbose:
        print(f"  ✓ Created trimmed version: {original_dur:.1f}s → {trimmed_dur:.1f}s (removed {original_dur - trimmed_dur:.1f}s of recovery)")
    
    return seg_trimmed


def recortar_en_fin_breath_hold(seg_dict: dict, verbose: bool = True) -> dict:
    """
    Trim segment at the marked breath-hold end time (MODIFIES IN-PLACE).
    
    DEPRECATED: Use crear_version_recortada_bh() instead to preserve original.
    This function is kept for backward compatibility but modifies the original.
    
    Args:
        seg_dict: Segment dictionary (must have bh_end_time_s)
        verbose: Print trimming information
    
    Returns:
        Updated seg_dict trimmed at breath-hold end
    """
    bh_end_time = seg_dict.get("bh_end_time_s", None)
    if bh_end_time is None:
        if verbose:
            print("  ⚠️  No breath-hold end marker found. Use marcar_fin_breath_hold() first.")
        return seg_dict
    
    times = seg_dict.get("times", np.array([]))
    fs = seg_dict.get("fs_rsp", 50)
    
    if len(times) == 0:
        return seg_dict
    
    # Find index corresponding to bh_end_time
    target_idx = int(bh_end_time * fs)
    target_idx = min(target_idx, len(times))
    
    original_dur = seg_dict.get("dur_s", len(times) / fs)
    trimmed_dur = times[target_idx] if target_idx < len(times) else original_dur
    
    if trimmed_dur >= original_dur:
        if verbose:
            print(f"  ℹ️  Segment already ends before breath-hold end marker ({bh_end_time:.1f}s)")
        return seg_dict
    
    # Trim all arrays
    seg_dict["times"] = times[:target_idx]
    seg_dict["rsp_raw"] = seg_dict.get("rsp_raw", np.array([]))[:target_idx]
    seg_dict["rsp_clean"] = seg_dict.get("rsp_clean", np.array([]))[:target_idx]
    if "rsp_interp" in seg_dict:
        seg_dict["rsp_interp"] = seg_dict["rsp_interp"][:target_idx]
    if "rsp_masked" in seg_dict:
        seg_dict["rsp_masked"] = seg_dict["rsp_masked"][:target_idx]
    seg_dict["phase_signal"] = seg_dict.get("phase_signal", np.array([]))[:target_idx]
    
    # Update peaks/troughs
    peaks = seg_dict.get("peaks", np.array([]))
    troughs = seg_dict.get("troughs", np.array([]))
    seg_dict["peaks"] = peaks[peaks < target_idx]
    seg_dict["troughs"] = troughs[troughs < target_idx]
    
    # Recalculate metrics
    if len(seg_dict["troughs"]) > 1:
        metrics = calculate_breathing_metrics(
            seg_dict["rsp_clean"], seg_dict["peaks"], seg_dict["troughs"], fs
        )
        seg_dict.update(metrics)
    
    seg_dict["dur_s"] = trimmed_dur
    
    if verbose:
        print(f"  ✓ Trimmed segment: {original_dur:.1f}s → {trimmed_dur:.1f}s (removed {original_dur - trimmed_dur:.1f}s of recovery)")
    
    return seg_dict


# =============================================================================
# SEGMENT INSPECTION
# =============================================================================

def inspeccionar_segmento_rsp(
    rsp_raw: np.ndarray,
    fs_rsp: float,
    eventos: Dict[str, int],
    seg_name: str = "LB",
    win_len: Optional[float] = None,
    target_fs: float = TARGET_FS_HZ,
    segmentos_dict: Optional[Dict[str, Tuple[int, int]]] = None,
    mostrar_info_segmentos_dict: bool = True,
) -> dict:
    """
    Inspect and process an RSP segment with visualization.
    
    Steps:
    1. Extract segment using markers
    2. Downsample to target frequency (50 Hz default for RSA sync)
    3. Apply bandpass filter (0.05-1 Hz)
    4. Detect breathing cycles (peaks/troughs)
    5. Calculate breathing metrics
    6. Detect artifacts automatically
    7. Visualize in windows
    """
    if plt is None:
        print("matplotlib not installed; no plots will be generated.")
        return {}
    
    print("\n" + "=" * 60)
    print(f"Inspecting RSP segment: {seg_name}")
    print("=" * 60)
    
    n_total = len(rsp_raw)
    
    # Get segment boundaries
    if segmentos_dict is not None and seg_name in segmentos_dict:
        start_idx, end_idx = segmentos_dict[seg_name]
        if mostrar_info_segmentos_dict:
            print(f"  (using segmentos_dict for boundaries)")
    else:
        start_idx, end_idx = get_segment_indices(eventos, seg_name, n_total)
    
    dur_s = (end_idx - start_idx) / fs_rsp
    print(f"Segment {seg_name}: samples [{start_idx} - {end_idx}], duration = {dur_s:.1f} s")
    
    # Extract segment
    rsp_seg_raw = rsp_raw[start_idx:end_idx].copy()
    
    # Downsample
    rsp_downsampled, fs_down = downsample_signal(rsp_seg_raw, fs_rsp, target_fs)
    print(f"Downsampled: {fs_rsp:.0f} Hz → {fs_down:.0f} Hz ({len(rsp_downsampled)} samples)")
    
    # Create time vector
    times = np.arange(len(rsp_downsampled)) / fs_down
    dur_seg = float(times[-1]) if len(times) else 0.0
    
    # Filter
    print(f"\n--- Filtering (bandpass: {HIGHPASS_CUTOFF_HZ}-{LOWPASS_CUTOFF_HZ} Hz, numtaps {FIR_NUMTAPS}) ---")
    rsp_clean = filter_rsp(rsp_downsampled, fs_down)
    
    # Detect breathing cycles
    print(f"\n--- Breathing cycle detection ---")
    
    # For BH segments, use high amplitude threshold to filter out small noise during breath-hold
    min_amplitude_threshold = None
    is_bh_segment = "BH" in seg_name
    if is_bh_segment:
        signal_range = np.ptp(rsp_clean)  # Peak-to-peak range
        min_amplitude_threshold = signal_range * BH_MIN_AMPLITUDE_THRESHOLD_FRACTION
        print(f"  BH segment detected: Using amplitude threshold = {min_amplitude_threshold:.3f} ({BH_MIN_AMPLITUDE_THRESHOLD_FRACTION*100:.0f}% of signal range)")
    
    # Suppress verbose output for BH segments (expected to filter out most detections)
    peaks, troughs = detect_breathing_cycles(
        rsp_clean, fs_down, 
        verbose=not is_bh_segment,  # No verbose for BH segments
        min_amplitude_threshold=min_amplitude_threshold
    )
    print(f"  Final: {len(troughs)} breaths (troughs), {len(peaks)} peaks")
    
    # Calculate metrics
    if len(troughs) > 1:
        metrics = calculate_breathing_metrics(rsp_clean, peaks, troughs, fs_down)
        mean_rate = np.mean(metrics["resp_rate_bpm"]) if len(metrics["resp_rate_bpm"]) > 0 else 0
        mean_ie = np.mean(metrics["ie_ratios"]) if len(metrics["ie_ratios"]) > 0 else 0
        print(f"  Mean respiratory rate: {mean_rate:.1f} breaths/min")
        print(f"  Mean I:E ratio: {mean_ie:.2f}")
    else:
        metrics = {
            "breath_intervals_s": np.array([]),
            "breath_times_s": np.array([]),
            "resp_rate_bpm": np.array([]),
            "inspiration_durations": np.array([]),
            "expiration_durations": np.array([]),
            "ie_ratios": np.array([]),
            "breath_amplitudes": np.array([]),
        }
        # Skip warning for BH segments (expected to have few/no breaths)
        if "BH" not in seg_name:
            print(f"  ⚠️  Insufficient breaths for metrics")
    
    # Phase signal
    phase_signal = calculate_phase_signal(peaks, troughs, len(rsp_clean))
    
    # Build result dict (NO auto-marking of artifacts - user decides)
    seg_dict = {
        "seg": seg_name,
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "dur_s": float(dur_seg),
        "fs_original": float(fs_rsp),
        "fs_rsp": float(fs_down),
        "times": times,
        "rsp_raw": rsp_downsampled,
        "rsp_clean": rsp_clean,
        "peaks": peaks,
        "troughs": troughs,
        "phase_signal": phase_signal,
        "breath_intervals_s": metrics.get("breath_intervals_s", np.array([])),
        "breath_times_s": metrics.get("breath_times_s", np.array([])),
        "resp_rate_bpm": metrics.get("resp_rate_bpm", np.array([])),
        "inspiration_durations": metrics.get("inspiration_durations", np.array([])),
        "expiration_durations": metrics.get("expiration_durations", np.array([])),
        "ie_ratios": metrics.get("ie_ratios", np.array([])),
        "breath_amplitudes": metrics.get("breath_amplitudes", np.array([])),
        "bad_intervals": pd.DataFrame(columns=["t_ini_s", "t_fin_s", "motivo", "sev"]),
        "interpolated_intervals": [],
        "mean_resp_rate_bpm": float(np.mean(metrics["resp_rate_bpm"])) if len(metrics["resp_rate_bpm"]) > 0 else np.nan,
        "std_resp_rate_bpm": float(np.std(metrics["resp_rate_bpm"])) if len(metrics["resp_rate_bpm"]) > 0 else np.nan,
        "mean_ie_ratio": float(np.mean(metrics["ie_ratios"])) if len(metrics["ie_ratios"]) > 0 else np.nan,
    }
    
    # Initialize bad_pct to 0 (no auto-marking)
    seg_dict["bad_pct"] = 0.0
    seg_dict["bad_s"] = 0.0
    
    return seg_dict


def plot_rsp_window(seg_dict: dict, t0: float = 0, win_len: float = 60.0, show_bad: bool = True):
    """
    Plot a single window of RSP data (NeuroKit2-style with 5 panels).
    
    Panels:
    1. Raw and Cleaned Signal with peaks/troughs
    2. Breathing Rate
    3. Breathing Amplitude
    4. Respiratory Volume per Time (RVT) - approximated
    5. Cycle Symmetry (I:E ratio visualization)
    """
    if plt is None:
        return
    
    times = seg_dict.get("times", np.array([]))
    rsp_raw = seg_dict.get("rsp_raw", np.array([]))
    rsp_clean = seg_dict.get("rsp_clean", np.array([]))
    peaks = seg_dict.get("peaks", np.array([]))
    troughs = seg_dict.get("troughs", np.array([]))
    resp_rate_bpm = seg_dict.get("resp_rate_bpm", np.array([]))
    breath_times_s = seg_dict.get("breath_times_s", np.array([]))
    breath_amplitudes = seg_dict.get("breath_amplitudes", np.array([]))
    ie_ratios = seg_dict.get("ie_ratios", np.array([]))
    seg_name = seg_dict.get("seg", "SEG")
    
    mask = (times >= t0) & (times < t0 + win_len)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return
    
    i0, i1 = idx[0], idx[-1] + 1
    t_win = times[i0:i1]
    t_end = t0 + win_len
    
    fig, axs = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Respiration (RSP) - {seg_name} [{t0:.1f}s - {t_end:.1f}s]", fontsize=12, fontweight='bold')
    
    bad_consolidated = consolidar_bad_intervals(seg_dict) if show_bad else pd.DataFrame()
    
    def shade_bad(ax):
        if not show_bad or len(bad_consolidated) == 0:
            return
        for _, row in bad_consolidated.iterrows():
            b_ini, b_fin = row['t_ini_s'], row['t_fin_s']
            if b_fin < t0 or b_ini > t_end:
                continue
            ax.axvspan(max(b_ini, t0), min(b_fin, t_end), color=PLOT_COLORS["bad"], alpha=0.3)
    
    # Panel 1: Raw and Cleaned Signal
    ax = axs[0]
    ax.set_title("Raw and Cleaned Signal", fontsize=10, loc='left')
    ax.plot(t_win, rsp_raw[i0:i1], lw=0.6, color='#1f77b4', alpha=0.4, label="Raw")
    ax.plot(t_win, rsp_clean[i0:i1], lw=1.0, color='#00CED1', label="Cleaned")
    
    # Peaks and troughs
    peaks_win = peaks[(peaks >= i0) & (peaks < i1)]
    troughs_win = troughs[(troughs >= i0) & (troughs < i1)]
    
    if len(peaks_win) > 0:
        ax.scatter(times[peaks_win], rsp_clean[peaks_win], s=40, c='red',
                  marker='o', edgecolors='black', linewidths=0.5, zorder=5, label="Exhalation Onsets")
    if len(troughs_win) > 0:
        ax.scatter(times[troughs_win], rsp_clean[troughs_win], s=40, c='orange',
                  marker='o', edgecolors='black', linewidths=0.5, zorder=5, label="Inhalation Onsets")
    
    shade_bad(ax)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel("Signal")
    
    # Panel 2: Breathing Rate
    ax = axs[1]
    ax.set_title("Breathing Rate", fontsize=10, loc='left')
    if len(resp_rate_bpm) > 0 and len(breath_times_s) > 0:
        rate_mask = (breath_times_s >= t0) & (breath_times_s < t_end)
        if np.any(rate_mask):
            ax.plot(breath_times_s[rate_mask], resp_rate_bpm[rate_mask], 
                   lw=1.5, color='#2ca02c', label="Rate")
            mean_rate = np.mean(resp_rate_bpm[rate_mask])
            ax.axhline(y=mean_rate, color='cyan', linestyle='--', lw=1, alpha=0.8, label=f"Mean")
            ax.legend(loc="upper right", fontsize=7)
    shade_bad(ax)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel("Rate (bpm)")
    
    # Panel 3: Breathing Amplitude
    ax = axs[2]
    ax.set_title("Breathing Amplitude", fontsize=10, loc='left')
    if len(breath_amplitudes) > 0 and len(breath_times_s) > 0:
        # Match amplitudes with times (amplitudes are per cycle)
        n_amp = min(len(breath_amplitudes), len(breath_times_s))
        amp_mask = (breath_times_s[:n_amp] >= t0) & (breath_times_s[:n_amp] < t_end)
        if np.any(amp_mask):
            ax.plot(breath_times_s[:n_amp][amp_mask], breath_amplitudes[:n_amp][amp_mask],
                   lw=1.5, color='#9467bd', label="Amplitude")
            mean_amp = np.mean(breath_amplitudes[:n_amp][amp_mask])
            ax.axhline(y=mean_amp, color='cyan', linestyle='--', lw=1, alpha=0.8, label="Mean")
            ax.legend(loc="upper right", fontsize=7)
    shade_bad(ax)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel("Amplitude")
    
    # Panel 4: RVT approximation (amplitude * rate)
    ax = axs[3]
    ax.set_title("Respiratory Volume per Time (RVT approx.)", fontsize=10, loc='left')
    if len(breath_amplitudes) > 0 and len(resp_rate_bpm) > 0 and len(breath_times_s) > 0:
        n_min = min(len(breath_amplitudes), len(resp_rate_bpm), len(breath_times_s))
        rvt_approx = breath_amplitudes[:n_min] * resp_rate_bpm[:n_min] / 60.0  # Normalize
        rvt_mask = (breath_times_s[:n_min] >= t0) & (breath_times_s[:n_min] < t_end)
        if np.any(rvt_mask):
            ax.plot(breath_times_s[:n_min][rvt_mask], rvt_approx[rvt_mask],
                   lw=1.5, color='#17becf', label="RVT")
            mean_rvt = np.mean(rvt_approx[rvt_mask])
            ax.axhline(y=mean_rvt, color='cyan', linestyle='--', lw=1, alpha=0.8, label="Mean")
            ax.legend(loc="upper right", fontsize=7)
    shade_bad(ax)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel("RVT")
    
    # Panel 5: Cycle Symmetry (I:E ratio)
    ax = axs[4]
    ax.set_title("Cycle Symmetry (I:E Ratio)", fontsize=10, loc='left')
    if len(ie_ratios) > 0 and len(breath_times_s) > 0:
        n_ie = min(len(ie_ratios), len(breath_times_s))
        ie_mask = (breath_times_s[:n_ie] >= t0) & (breath_times_s[:n_ie] < t_end)
        if np.any(ie_mask):
            ax.plot(breath_times_s[:n_ie][ie_mask], ie_ratios[:n_ie][ie_mask],
                   lw=1.5, color='#8B008B', label="I:E Ratio")
            mean_ie = np.mean(ie_ratios[:n_ie][ie_mask])
            ax.axhline(y=mean_ie, color='cyan', linestyle='--', lw=1, alpha=0.8, label=f"Mean: {mean_ie:.2f}")
            ax.axhline(y=0.5, color='gray', linestyle=':', lw=1, alpha=0.5, label="Normal ~0.5")
            ax.legend(loc="upper right", fontsize=7)
    shade_bad(ax)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel("I:E Ratio")
    ax.set_xlabel("Time (seconds)")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.25)
    plt.show()


def plot_rsp_simple(seg_dict: dict, t0: float = 0, win_len: float = 60.0):
    """
    Simple 2-panel plot: Raw vs Cleaned signal (for quick inspection).
    """
    if plt is None:
        return
    
    times = seg_dict.get("times", np.array([]))
    rsp_raw = seg_dict.get("rsp_raw", np.array([]))
    rsp_clean = seg_dict.get("rsp_clean", np.array([]))
    peaks = seg_dict.get("peaks", np.array([]))
    troughs = seg_dict.get("troughs", np.array([]))
    seg_name = seg_dict.get("seg", "SEG")
    
    mask = (times >= t0) & (times < t0 + win_len)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return
    
    i0, i1 = idx[0], idx[-1] + 1
    t_win = times[i0:i1]
    t_end = t0 + win_len
    
    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(f"{seg_name} - Signal Comparison [{t0:.1f}s - {t_end:.1f}s]", fontsize=11)
    
    # Panel 1: Raw signal
    ax = axs[0]
    ax.plot(t_win, rsp_raw[i0:i1], lw=0.8, color='#1f77b4', label="Raw RSP")
    ax.set_ylabel("Raw Signal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Cleaned with peaks/troughs
    ax = axs[1]
    ax.plot(t_win, rsp_clean[i0:i1], lw=1.0, color='#2ca02c', label="Filtered RSP")
    
    peaks_win = peaks[(peaks >= i0) & (peaks < i1)]
    troughs_win = troughs[(troughs >= i0) & (troughs < i1)]
    
    if len(peaks_win) > 0:
        ax.scatter(times[peaks_win], rsp_clean[peaks_win], s=50, c='red',
                  marker='v', edgecolors='black', linewidths=1, zorder=5, label="Peaks")
    if len(troughs_win) > 0:
        ax.scatter(times[troughs_win], rsp_clean[troughs_win], s=50, c='orange',
                  marker='^', edgecolors='black', linewidths=1, zorder=5, label="Troughs")
    
    ax.set_ylabel("Filtered Signal")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def reinspeccionar_segmento(seg_dict: dict, win_len: Optional[float] = None) -> pd.DataFrame:
    """Re-inspect segment after modifications - same 4-panel view as initial inspection."""
    if plt is None:
        return pd.DataFrame()
    
    seg_name = seg_dict.get("seg", "SEG")
    times = seg_dict.get("times", np.array([]))
    
    if len(times) == 0:
        print("No data to inspect")
        return pd.DataFrame()
    
    bad_consolidated = consolidar_bad_intervals(seg_dict)
    bad_pct, bad_s, total_s = calcular_bad_pct(seg_dict)
    clean_s = total_s - bad_s if np.isfinite(bad_s) else total_s
    
    status = "✔ OK" if bad_pct <= BAD_PROP_THRESHOLD else f"❌ EXCLUDED (BAD > {BAD_PROP_THRESHOLD*100:.0f}%)"
    
    print(f"\n{'='*70}")
    print(f"RSP RE-INSPECTION: {seg_name}")
    print(f"{'='*70}")
    print(f"Duration: {total_s:.1f}s | BAD: {bad_s:.1f}s ({bad_pct*100:.1f}%) | CLEAN: {clean_s:.1f}s → {status}")
    print(f"Breaths: {len(seg_dict.get('troughs', []))} | Mean rate: {seg_dict.get('mean_resp_rate_bpm', 0):.1f} bpm")
    
    if len(bad_consolidated) > 0:
        print(f"\n--- BAD ZONES ---")
        for _, row in bad_consolidated.iterrows():
            print(f"  [{row['id']}] {row['t_ini_s']:.1f}s - {row['t_fin_s']:.1f}s ({row['dur_s']:.1f}s) | {row['motivos']}")
    else:
        print("\n✔ No BAD zones")
    
    # Show same 4-panel view as initial inspection
    ver_señal_completa(seg_dict, show_bad=True)
    
    return bad_consolidated


def ampliar_senal_en_tiempo(seg_dict: dict, t_centro: float, ventana_s: float = 10.0):
    """
    Zoom into a specific time region.
    
    For BH segments: 2 panels (Raw, Filtered)
    For other segments: 4 panels (Raw, Filtered, Rate, Amplitude)
    """
    if plt is None:
        return
    
    seg_name = seg_dict.get("seg", "SEG")
    rsp_raw = seg_dict.get("rsp_raw", np.array([]))
    # Use interpolated signal if available (after cleaning), otherwise use clean
    rsp_to_plot = seg_dict.get("rsp_interp", seg_dict.get("rsp_clean", np.array([])))
    rsp_clean = seg_dict.get("rsp_clean", np.array([]))  # Keep original for reference
    times = seg_dict.get("times", np.array([]))
    peaks = seg_dict.get("peaks", np.array([]))
    troughs = seg_dict.get("troughs", np.array([]))
    resp_rate = seg_dict.get("resp_rate_bpm", np.array([]))
    breath_amplitudes = seg_dict.get("breath_amplitudes", np.array([]))
    breath_times = seg_dict.get("breath_times_s", np.array([]))
    bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
    
    t0 = max(0, t_centro - ventana_s/2)
    t_end = t0 + ventana_s
    
    # Get indices for window
    mask = (times >= t0) & (times < t_end)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print(f"No data in window [{t0:.1f}s - {t_end:.1f}s]")
        return
    
    i0, i1 = idx[0], idx[-1] + 1
    t_win = times[i0:i1]
    
    print(f"\n--- ZOOMED VIEW: [{t0:.1f}s - {t_end:.1f}s] ---")
    
    # For BH segments, only show 2 panels
    is_bh = "BH" in seg_name
    n_panels = 2 if is_bh else 4
    
    fig, axs = plt.subplots(n_panels, 1, figsize=(16, 6*n_panels if n_panels == 4 else 8), sharex=True)
    if n_panels == 2:
        axs = [axs[0], axs[1]]  # Ensure axs is a list for consistency
    fig.suptitle(f"{seg_name} - ZOOMED [{t0:.1f}s - {t_end:.1f}s]", fontsize=13, fontweight='bold')
    
    # Panel 1: Raw signal
    ax = axs[0]
    ax.plot(t_win, rsp_raw[i0:i1], lw=0.5, color='#1f77b4', alpha=0.9, label="Raw RSP")
    ax.set_ylabel("Raw Signal", fontsize=10)
    if ax.get_legend_handles_labels()[0]:  # Only show legend if there are handles
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Panel 1: Raw Signal (before filtering)", fontsize=10, loc='left')
    
    # Panel 2: Filtered signal with peaks/troughs and BAD zones
    ax = axs[1]
    # Plot interpolated signal if available (after cleaning), otherwise plot clean
    signal_label = "Filtered RSP (interpolated)" if "rsp_interp" in seg_dict else "Filtered RSP"
    ax.plot(t_win, rsp_to_plot[i0:i1], lw=0.8, color='#2ca02c', label=signal_label)
    
    peaks_win = peaks[(peaks >= i0) & (peaks < i1)]
    troughs_win = troughs[(troughs >= i0) & (troughs < i1)]
    
    if len(peaks_win) > 0:
        ax.scatter(times[peaks_win], rsp_to_plot[peaks_win], s=30, c='red',
                  marker='v', edgecolors='black', linewidths=0.5, zorder=5, label="Peaks (exh)")
    if len(troughs_win) > 0:
        ax.scatter(times[troughs_win], rsp_to_plot[troughs_win], s=30, c='orange',
                  marker='^', edgecolors='black', linewidths=0.5, zorder=5, label="Troughs (inh)")
    
    # Show BAD zones
    if len(bad_intervals) > 0:
        bad_cons = consolidar_bad_intervals(seg_dict)
        for _, row in bad_cons.iterrows():
            if row['t_fin_s'] < t0 or row['t_ini_s'] > t_end:
                continue
            ax.axvspan(max(row['t_ini_s'], t0), min(row['t_fin_s'], t_end), 
                      color='red', alpha=0.3, zorder=1)
            ax.annotate(f"[{row['id']}]", xy=(max(row['t_ini_s'], t0), ax.get_ylim()[1]*0.95),
                       fontsize=7, color='red', fontweight='bold')
    
    # Show breath-hold end marker (blue vertical line)
    bh_end_time = seg_dict.get("bh_end_time_s", None)
    if bh_end_time is not None and t0 <= bh_end_time <= t_end:
        ax.axvline(x=bh_end_time, color='blue', linestyle='--', linewidth=2, 
                  alpha=0.7, zorder=10, label="BH End")
        ax.annotate("BH End", xy=(bh_end_time, ax.get_ylim()[1]*0.98),
                   fontsize=9, color='blue', fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_ylabel("Filtered Signal", fontsize=10)
    if ax.get_legend_handles_labels()[0]:  # Only show legend if there are handles
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Panel 2: Filtered Signal + Peaks/Troughs + BAD zones", fontsize=10, loc='left')
    
    # Only show Rate and Amplitude panels for non-BH segments
    if not is_bh:
        # Panel 3: Breathing Rate
        ax = axs[2]
        if len(resp_rate) > 0 and len(breath_times) > 0:
            rate_mask = (breath_times >= t0) & (breath_times < t_end)
            if np.any(rate_mask):
                ax.plot(breath_times[rate_mask], resp_rate[rate_mask], 'o-', color='#9467bd',
                       markersize=4, lw=1, label=f"Rate (mean: {np.mean(resp_rate[rate_mask]):.1f} bpm)")
                ax.axhline(y=12, color='gray', linestyle='--', alpha=0.5, label="Normal range")
                ax.axhline(y=20, color='gray', linestyle='--', alpha=0.5)
                ax.fill_between(breath_times[rate_mask], 12, 20, alpha=0.1, color='green')
        ax.set_ylabel("Rate (bpm)", fontsize=10)
        ax.set_ylim([0, max(40, np.max(resp_rate[rate_mask]) + 5) if len(resp_rate) > 0 and np.any(rate_mask) else 40])
        if ax.get_legend_handles_labels()[0]:  # Only show legend if there are handles
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Panel 3: Breathing Rate (normal: 12-20 bpm at rest)", fontsize=10, loc='left')
        
        # Panel 4: Breath Amplitude
        ax = axs[3]
        if len(breath_amplitudes) > 0 and len(breath_times) > 0:
            amp_mask = (breath_times >= t0) & (breath_times < t_end)
            if np.any(amp_mask):
                ax.bar(breath_times[amp_mask], breath_amplitudes[amp_mask], width=2.0,
                      color='#17becf', alpha=0.7, label=f"Amplitude (mean: {np.mean(breath_amplitudes[amp_mask]):.2f})")
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)
        if ax.get_legend_handles_labels()[0]:  # Only show legend if there are handles
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Panel 4: Breath Amplitude (look for sudden changes)", fontsize=10, loc='left')
    else:
        # For BH segments, add xlabel to the last panel
        ax.set_xlabel("Time (s)", fontsize=10)
    
    plt.tight_layout()
    plt.show()


def ver_minuto(seg_dict: dict, minuto: int = 0):
    """
    View a specific minute.
    
    For BH segments: 2 panels (Raw, Filtered)
    For other segments: 4 panels (Raw, Filtered, Rate, Amplitude)
    """
    t0 = minuto * 60.0
    dur = seg_dict.get("dur_s", 0)
    
    if t0 >= dur:
        print(f"Minute {minuto} is beyond segment duration ({dur:.0f}s)")
        return
    
    win = min(60.0, dur - t0)
    
    # Use ampliar_senal_en_tiempo with the minute's center
    t_centro = t0 + win / 2
    print(f"\n--- MINUTE {minuto}: [{t0:.0f}s - {t0+win:.0f}s] ---")
    ampliar_senal_en_tiempo(seg_dict, t_centro=t_centro, ventana_s=win)


def ver_señal_completa(seg_dict: dict, show_bad: bool = True):
    """
    View the ENTIRE segment.
    
    For BH segments: 2 panels (Raw, Filtered) - breathing metrics don't apply
    For other segments: 4 panels (Raw, Filtered, Breathing Rate, Amplitude)
    Best for inspecting full segments before cleaning.
    """
    if plt is None:
        return
    
    seg_name = seg_dict.get("seg", "SEG")
    rsp_raw = seg_dict.get("rsp_raw", np.array([]))
    # Use interpolated signal if available (after cleaning), otherwise use clean
    rsp_to_plot = seg_dict.get("rsp_interp", seg_dict.get("rsp_clean", np.array([])))
    rsp_clean = seg_dict.get("rsp_clean", np.array([]))  # Keep original for reference
    times = seg_dict.get("times", np.array([]))
    peaks = seg_dict.get("peaks", np.array([]))
    troughs = seg_dict.get("troughs", np.array([]))
    resp_rate = seg_dict.get("resp_rate_bpm", np.array([]))
    breath_amplitudes = seg_dict.get("breath_amplitudes", np.array([]))
    breath_times = seg_dict.get("breath_times_s", np.array([]))
    bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
    dur = seg_dict.get("dur_s", len(times) / seg_dict.get("fs_rsp", 50) if len(times) > 0 else 0)
    
    if len(rsp_to_plot) == 0:
        print("No data to plot")
        return
    
    # For BH segments, only show 2 panels (Raw and Filtered)
    is_bh = "BH" in seg_name
    n_panels = 2 if is_bh else 4
    
    fig, axs = plt.subplots(n_panels, 1, figsize=(16, 6*n_panels if n_panels == 4 else 8), sharex=True)
    if n_panels == 2:
        axs = [axs[0], axs[1]]  # Ensure axs is a list for consistency
    fig.suptitle(f"{seg_name} - FULL SIGNAL ({dur:.0f}s)", fontsize=13, fontweight='bold')
    
    ax = axs[0]
    ax.plot(times, rsp_raw, lw=0.5, color='#1f77b4', alpha=0.9, label="Raw RSP")
    ax.set_ylabel("Raw Signal", fontsize=10)
    if ax.get_legend_handles_labels()[0]:  # Only show legend if there are handles
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Panel 1: Raw Signal (before filtering)", fontsize=10, loc='left')
    
    ax = axs[1]
    # Plot interpolated signal if available (after cleaning), otherwise plot clean
    signal_label = "Filtered RSP (interpolated)" if "rsp_interp" in seg_dict else "Filtered RSP"
    ax.plot(times, rsp_to_plot, lw=0.8, color='#2ca02c', label=signal_label)
    if len(peaks) > 0:
        ax.scatter(times[peaks], rsp_to_plot[peaks], s=20, c='red',
                  marker='v', edgecolors='black', linewidths=0.5, zorder=5, label="Peaks (exh)")
    if len(troughs) > 0:
        ax.scatter(times[troughs], rsp_to_plot[troughs], s=20, c='orange',
                  marker='^', edgecolors='black', linewidths=0.5, zorder=5, label="Troughs (inh)")
    if show_bad and len(bad_intervals) > 0:
        bad_cons = consolidar_bad_intervals(seg_dict)
        for _, row in bad_cons.iterrows():
            ax.axvspan(row['t_ini_s'], row['t_fin_s'], color='red', alpha=0.3, zorder=1)
            ax.annotate(f"[{row['id']}]", xy=(row['t_ini_s'], ax.get_ylim()[1]*0.95),
                       fontsize=7, color='red', fontweight='bold')
    
    # Show breath-hold end marker (blue vertical line)
    bh_end_time = seg_dict.get("bh_end_time_s", None)
    if bh_end_time is not None:
        # Ensure it's a float and within the time range
        bh_end_time = float(bh_end_time)
        times_min = times[0] if len(times) > 0 else 0
        times_max = times[-1] if len(times) > 0 else dur
        if times_min <= bh_end_time <= times_max:
            for ax_panel in axs:
                ax_panel.axvline(x=bh_end_time, color='blue', linestyle='--', linewidth=2.5, 
                               alpha=0.8, zorder=10, label="BH End")
                ylim = ax_panel.get_ylim()
                ax_panel.annotate("BH End", xy=(bh_end_time, ylim[1]*0.98),
                                fontsize=9, color='blue', fontweight='bold', ha='center',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            if is_bh:
                print(f"  [DEBUG] BH End marker drawn at {bh_end_time:.2f}s (time range: {times_min:.1f}s - {times_max:.1f}s)")
        else:
            if is_bh:
                print(f"  [DEBUG] BH End marker at {bh_end_time:.2f}s is outside visible range ({times_min:.1f}s - {times_max:.1f}s)")
    else:
        if is_bh:
            print(f"  [DEBUG] No BH End marker found in seg_dict")
    
    ax.set_ylabel("Filtered Signal", fontsize=10)
    if ax.get_legend_handles_labels()[0]:  # Only show legend if there are handles
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Panel 2: Filtered Signal + Peaks/Troughs + BAD zones", fontsize=10, loc='left')
    
    # Only show Rate and Amplitude panels for non-BH segments
    if not is_bh:
        ax = axs[2]
        if len(resp_rate) > 0 and len(breath_times) > 0:
            ax.plot(breath_times[:len(resp_rate)], resp_rate, 'o-', color='#9467bd',
                   markersize=3, lw=1, label=f"Rate (mean: {np.mean(resp_rate):.1f} bpm)")
            ax.axhline(y=12, color='gray', linestyle='--', alpha=0.5, label="Normal range")
            ax.axhline(y=20, color='gray', linestyle='--', alpha=0.5)
            ax.fill_between(breath_times[:len(resp_rate)], 12, 20, alpha=0.1, color='green')
        ax.set_ylabel("Rate (bpm)", fontsize=10)
        ax.set_ylim([0, max(40, np.max(resp_rate)+5) if len(resp_rate) > 0 else 40])
        if ax.get_legend_handles_labels()[0]:  # Only show legend if there are handles
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Panel 3: Breathing Rate (normal: 12-20 bpm at rest)", fontsize=10, loc='left')
        
        ax = axs[3]
        if len(breath_amplitudes) > 0 and len(breath_times) > 0:
            ax.bar(breath_times[:len(breath_amplitudes)], breath_amplitudes, width=2.0,
                  color='#17becf', alpha=0.7, label=f"Amplitude (mean: {np.mean(breath_amplitudes):.2f})")
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)
        if ax.get_legend_handles_labels()[0]:  # Only show legend if there are handles
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Panel 4: Breath Amplitude (look for sudden changes)", fontsize=10, loc='left')
    else:
        # For BH segments, add xlabel to the last panel
        ax.set_xlabel("Time (s)", fontsize=10)
    
    plt.tight_layout()
    plt.show()


def ver_señal_limpia(seg_dict: dict):
    """
    View the interpolated signal (final cleaned signal).
    Simple view: just the interpolated signal with peaks/troughs.
    """
    if plt is None:
        return
    
    seg_name = seg_dict.get("seg", "SEG")
    rsp_interp = seg_dict.get("rsp_interp", seg_dict.get("rsp_clean", np.array([])))
    times = seg_dict.get("times", np.array([]))
    peaks = seg_dict.get("peaks", np.array([]))
    troughs = seg_dict.get("troughs", np.array([]))
    dur = seg_dict.get("dur_s", len(times) / seg_dict.get("fs_rsp", 50) if len(times) > 0 else 0)
    
    # Debug: check what signals are available
    print(f"\n[DEBUG ver_clean_signal]")
    print(f"  rsp_interp available: {'rsp_interp' in seg_dict}")
    print(f"  rsp_clean available: {'rsp_clean' in seg_dict}")
    print(f"  rsp_interp length: {len(rsp_interp)}")
    print(f"  Peaks: {len(peaks)}, Troughs: {len(troughs)}")
    if "interpolated_intervals" in seg_dict:
        interp_intervals = seg_dict.get("interpolated_intervals", [])
        print(f"  Interpolated intervals: {len(interp_intervals)}")
        if len(interp_intervals) > 0:
            print(f"    Intervals: {interp_intervals}")
    
    if len(rsp_interp) == 0:
        print("No interpolated signal to plot")
        return
    
    is_bh = "BH" in seg_name
    n_panels = 2 if is_bh else 4
    
    fig, axs = plt.subplots(n_panels, 1, figsize=(16, 6*n_panels if n_panels == 4 else 8), sharex=True)
    if n_panels == 2:
        axs = [axs[0], axs[1]]
    
    fig.suptitle(f"{seg_name} - INTERPOLATED SIGNAL ({dur:.0f}s)", 
                 fontsize=13, fontweight='bold', color='green')
    
    # Panel 1: Raw signal (for reference)
    rsp_raw = seg_dict.get("rsp_raw", np.array([]))
    ax = axs[0]
    ax.plot(times, rsp_raw, lw=0.5, color='#1f77b4', alpha=0.7, label="Raw RSP")
    ax.set_ylabel("Raw Signal", fontsize=10)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Panel 1: Raw Signal", fontsize=10, loc='left')
    
    # Panel 2: Interpolated signal only
    ax = axs[1]
    ax.plot(times, rsp_interp, lw=1.2, color='#2ca02c', label="Interpolated RSP", zorder=2)
    
    # Plot peaks and troughs
    if len(peaks) > 0:
        valid_peaks = peaks[(peaks >= 0) & (peaks < len(times))]
        if len(valid_peaks) > 0:
            ax.scatter(times[valid_peaks], rsp_interp[valid_peaks], s=25, c='red',
                      marker='v', edgecolors='black', linewidths=0.5, zorder=5, 
                      label=f"Peaks ({len(valid_peaks)})")
    
    if len(troughs) > 0:
        valid_troughs = troughs[(troughs >= 0) & (troughs < len(times))]
        if len(valid_troughs) > 0:
            ax.scatter(times[valid_troughs], rsp_interp[valid_troughs], s=25, c='orange',
                      marker='^', edgecolors='black', linewidths=0.5, zorder=5,
                      label=f"Troughs ({len(valid_troughs)})")
    
    ax.set_ylabel("Interpolated Signal", fontsize=10)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Panel 2: Interpolated Signal + Peaks/Troughs", fontsize=10, loc='left')
    
    # Only show Rate and Amplitude panels for non-BH segments
    if not is_bh:
        resp_rate = seg_dict.get("resp_rate_bpm", np.array([]))
        breath_amplitudes = seg_dict.get("breath_amplitudes", np.array([]))
        breath_times = seg_dict.get("breath_times_s", np.array([]))
        
        ax = axs[2]
        if len(resp_rate) > 0 and len(breath_times) > 0:
            ax.plot(breath_times[:len(resp_rate)], resp_rate, 'o-', color='#9467bd',
                   markersize=3, lw=1, label=f"Rate (mean: {np.mean(resp_rate):.1f} bpm)")
            ax.axhline(y=12, color='gray', linestyle='--', alpha=0.5, label="Normal range")
            ax.axhline(y=20, color='gray', linestyle='--', alpha=0.5)
            ax.fill_between(breath_times[:len(resp_rate)], 12, 20, alpha=0.1, color='green')
        ax.set_ylabel("Rate (bpm)", fontsize=10)
        ax.set_ylim([0, max(40, np.max(resp_rate)+5) if len(resp_rate) > 0 else 40])
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Panel 3: Breathing Rate", fontsize=10, loc='left')
        
        ax = axs[3]
        if len(breath_amplitudes) > 0 and len(breath_times) > 0:
            ax.bar(breath_times[:len(breath_amplitudes)], breath_amplitudes, width=2.0,
                  color='#17becf', alpha=0.7, label=f"Amplitude (mean: {np.mean(breath_amplitudes):.2f})")
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Panel 4: Breath Amplitude", fontsize=10, loc='left')
    else:
        ax.set_xlabel("Time (s)", fontsize=10)
    
    plt.tight_layout()
    plt.show()


def detectar_artifacts(
    seg_dict: dict,
    verbose: bool = True,
    active: bool = True,
    mask_only: bool = False,
) -> dict:
    """
    Run artifact detection and AUTO-MARK detected artifacts as BAD zones.
    
    Uses MAD-based robust detection for:
    - Step artifacts (derivative spikes)
    - Baseline shifts (rolling mean comparison)
    - Physiological outliers (adaptive timing bounds)
    - Amplitude outliers (IQR-based)
    
    Args:
        seg_dict: Dictionary containing segment data
        verbose: If True, print detailed output
        active: If False, skip detection and return as is
        mask_only: If True, mark artifacts but skip interpolation downstream (mask-only mode)
    
    Returns seg_dict with artifacts marked as bad (if active).
    """
    if not active:
        if verbose:
            print(f"\n  Artifact detection skipped for {seg_dict.get('seg', 'segment')} (active=False)")
        return seg_dict

    seg_dict["mask_only"] = bool(mask_only)

    seg_name = seg_dict.get("seg", "SEG")
    rsp_clean = seg_dict.get("rsp_clean", np.array([]))
    fs = seg_dict.get("fs_rsp", 50)
    breath_intervals = seg_dict.get("breath_intervals_s", np.array([]))
    breath_times = seg_dict.get("breath_times_s", np.array([]))
    breath_amplitudes = seg_dict.get("breath_amplitudes", np.array([]))
    
    # Calculate typical breath duration for cycle counting
    if len(breath_intervals) > 0:
        typical_breath_dur = float(np.median(breath_intervals))
    else:
        typical_breath_dur = TYPICAL_BREATH_DURATION_S
    
    # Run detection (now includes amplitude outliers)
    artifacts, detections = detect_all_artifacts(
        rsp_clean, fs,
        breath_intervals_s=breath_intervals,
        breath_times_s=breath_times,
        breath_amplitudes=breath_amplitudes,
        verbose=False  # We'll do our own output
    )
    
    # Calculate total bad duration and cycles
    total_bad_s = sum(end - start for start, end, _ in artifacts)
    total_cycles = total_bad_s / typical_breath_dur if typical_breath_dur > 0 else 0
    
    # Print formatted output
    print("\n" + "="*60)
    print(f"ARTIFACT DETECTION: {seg_name}")
    print("="*60)
    
    if len(artifacts) == 0:
        print("  ✔ No artifacts detected")
        seg_dict["detected_artifacts"] = []
        seg_dict["total_artifact_cycles"] = 0
        return seg_dict
    
    # Count by type
    artifact_counts = {}
    for start, end, reason in artifacts:
        if reason not in artifact_counts:
            artifact_counts[reason] = 0
        artifact_counts[reason] += 1
    
    print(f"  ⚠️  Artifact zone(s) detected -")
    for reason, count in artifact_counts.items():
        print(f"     - {count} {reason}")
    
    print(f"\n  Total duration: {total_bad_s:.1f}s")
    print(f"  Total: {total_cycles:.1f} breathing cycles")
    
    # Show details of each detected zone
    if verbose and len(artifacts) > 0:
        print(f"\n  Detected zones:")
        for i, (start_s, end_s, reason) in enumerate(artifacts, 1):
            dur = end_s - start_s
            cycles = dur / typical_breath_dur if typical_breath_dur > 0 else 0
            print(f"    [{i}] {start_s:.1f}s - {end_s:.1f}s ({dur:.1f}s, ~{cycles:.1f} cycles) - {reason}")
    
    # AUTO-MARK as BAD
    for start_s, end_s, reason in artifacts:
        seg_dict = marcar_bad_interval(seg_dict, start_s, end_s, motivo=reason)
    
    # Update bad percentage
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    
    # Store for reference
    seg_dict["detected_artifacts"] = artifacts
    seg_dict["total_artifact_cycles"] = total_cycles
    
    # Status message
    if total_cycles > 2:
        print(f"\n  BAD: {bad_s:.1f}s ({bad_pct*100:.1f}%) ❌ → >2 cycles affected → segment may NOT be valid")
    else:
        print(f"\n  BAD: {bad_s:.1f}s ({bad_pct*100:.1f}%) ✔ → ≤2 cycles affected → can be interpolated")
    if mask_only:
        print("  ⚠️ Mask-only mode enabled: BAD zones will remain masked (no interpolation applied)")
    
    # Show full signal with bad zones
    print(f"\n--- Showing full signal with BAD zones marked ---")
    ver_señal_completa(seg_dict, show_bad=True)
    
    return seg_dict


def marcar_artifacts_como_bad(seg_dict: dict) -> dict:
    """Mark all detected artifacts as bad intervals."""
    artifacts = seg_dict.get("detected_artifacts", [])
    
    if len(artifacts) == 0:
        print("No artifacts detected. Run detectar_artifacts() first.")
        return seg_dict
    
    for start_s, end_s, reason in artifacts:
        seg_dict = marcar_bad_interval(seg_dict, start_s, end_s, motivo=reason)
    
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    
    print(f"✓ Marked {len(artifacts)} artifact(s) as bad")
    print(f"  BAD: {bad_s:.1f}s ({bad_pct*100:.1f}%)")
    
    return seg_dict


# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================

def create_dual_representations(
    rsp_clean: np.ndarray,
    fs: float,
    bad_intervals: pd.DataFrame,
    max_interp_cycles: int = MAX_INTERP_CYCLES,
    max_interp_duration_s: float = MAX_INTERP_DURATION_S,
    typical_breath_dur: float = TYPICAL_BREATH_DURATION_S,
    verbose: bool = True,
    mask_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]], List[Tuple[float, float, str]]]:
    """
    Create dual signal representations for different analysis purposes.
    
    Returns:
        rsp_interp: Signal with short BAD intervals interpolated (for metrics)
        rsp_masked: Signal with ALL BAD intervals as NaN (for coupling/RSA)
        interpolated_intervals: List of (start_s, end_s) that were interpolated
        remaining_bad: List of (start_s, end_s, reason) too long to interpolate
    
    Notes:
        `rsp_masked` (NaNs) should be the primary input for coupling or RSA analyses.
        Interpolation is an optional fallback applied only to very short, isolated BAD zones.
        Set `mask_only=True` to skip interpolation entirely and rely on the masked signal for conservative workflows.
    """
    # Convert bad_intervals DataFrame to list
    bad_zones = []
    if len(bad_intervals) > 0:
        for _, row in bad_intervals.iterrows():
            bad_zones.append((row["t_ini_s"], row["t_fin_s"], row.get("motivo", "manual")))
    # Remove duplicate intervals that might appear after manual edits
    unique_zones = []
    seen_intervals = set()
    for start_s, end_s, motivo in bad_zones:
        key = (round(start_s, 6), round(end_s, 6))
        if key not in seen_intervals:
            seen_intervals.add(key)
            unique_zones.append((start_s, end_s, motivo))
    bad_zones = unique_zones
    
    if len(bad_zones) == 0:
        return rsp_clean.copy(), rsp_clean.copy(), [], []
    
    # Create masked version: ALL bad intervals become NaN
    rsp_masked = rsp_clean.copy().astype(float)
    for start_s, end_s, _ in bad_zones:
        i0 = int(start_s * fs)
        i1 = min(int(end_s * fs), len(rsp_masked))
        rsp_masked[i0:i1] = np.nan
    
    # Create interpolated version: only SHORT intervals are interpolated
    rsp_interp = rsp_clean.copy()
    interpolated = []
    remaining = []

    if mask_only:
        if verbose:
            print(f"  ⚠️ Mask-only mode: skipping interpolation for {len(bad_zones)} zone(s)")
        return rsp_interp, rsp_masked, interpolated, list(bad_zones)
    
    for start_s, end_s, reason in bad_zones:
        duration_s = end_s - start_s
        estimated_cycles = duration_s / typical_breath_dur
        
        # Interpolate only if: ≤ max cycles AND ≤ max duration
        can_interpolate = (estimated_cycles <= max_interp_cycles and 
                          duration_s <= max_interp_duration_s)
        
        if can_interpolate:
            rsp_interp, was_interp = interpolate_artifact(
                rsp_interp, fs, start_s, end_s,
                max_cycles=max_interp_cycles,
                typical_breath_dur=typical_breath_dur,
                verbose=verbose
            )
            if was_interp:
                interpolated.append((start_s, end_s))
                if verbose:
                    # Verify interpolation quality
                    start_idx = int(start_s * fs)
                    end_idx = int(end_s * fs)
                    interp_segment = rsp_interp[start_idx:end_idx]
                    if len(interp_segment) > 0:
                        has_nan = np.any(np.isnan(interp_segment))
                        has_inf = np.any(np.isinf(interp_segment))
                        if has_nan or has_inf:
                            print(f"  ⚠️ WARNING: Interpolated segment contains invalid values (NaN: {has_nan}, Inf: {has_inf})")
                        else:
                            print(f"  ✓ Interpolated [{start_s:.1f}s - {end_s:.1f}s] ({duration_s:.1f}s, ~{estimated_cycles:.1f} cycles) - OK")
                    else:
                        print(f"  ✓ Interpolated [{start_s:.1f}s - {end_s:.1f}s] ({duration_s:.1f}s, ~{estimated_cycles:.1f} cycles)")
            else:
                remaining.append((start_s, end_s, reason))
                if verbose:
                    print(
                        f"  ⚠️ Interpolation skipped for [{start_s:.1f}s - {end_s:.1f}s]; "
                        "`rsp_masked` (NaNs) will retain the artifact for coupling/RSA."
                    )
        else:
            remaining.append((start_s, end_s, reason))
            if verbose:
                print(
                    f"  ✗ Too long to interpolate [{start_s:.1f}s - {end_s:.1f}s] "
                    f"({duration_s:.1f}s, ~{estimated_cycles:.1f} cycles > {max_interp_cycles}) → "
                    "`rsp_masked` (NaNs) remains the conservative representation."
                )
    
    return rsp_interp, rsp_masked, interpolated, remaining


def build_cycle_quality_table(
    seg_dict: dict,
    bad_intervals: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build per-cycle quality tracking table.
    
    Each cycle's validity for metrics vs coupling is determined by whether
    it overlaps any BAD interval.
    
    Returns DataFrame with columns:
        - cycle_idx: Index of breathing cycle
        - trough_time_s: Start of cycle (inspiration onset)
        - peak_time_s: Peak of cycle (expiration onset)  
        - next_trough_time_s: End of cycle
        - interval_s: Cycle duration
        - amplitude: Peak-to-trough amplitude
        - resp_rate_bpm: Instantaneous breathing rate for this cycle
        - overlaps_bad: True if cycle overlaps ANY bad interval
        - overlaps_interpolated: True if cycle overlaps an interpolated interval
        - valid_for_metrics: True if can use for segment-level stats (from rsp_interp)
        - valid_for_coupling: True if can use for RSA/coherence (from rsp_masked)
    """
    troughs = seg_dict.get("troughs", np.array([]))
    peaks = seg_dict.get("peaks", np.array([]))
    times = seg_dict.get("times", np.array([]))
    rsp_clean = seg_dict.get("rsp_clean", np.array([]))
    
    if bad_intervals is None:
        bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
    
    interpolated = seg_dict.get("interpolated_intervals", [])
    
    if len(troughs) < 2:
        return pd.DataFrame()
    
    # Build list of bad time ranges
    bad_ranges = []
    if len(bad_intervals) > 0:
        for _, row in bad_intervals.iterrows():
            bad_ranges.append((row["t_ini_s"], row["t_fin_s"]))
    
    interp_ranges = interpolated if interpolated else []
    
    rows = []
    for i in range(len(troughs) - 1):
        trough_idx = troughs[i]
        next_trough_idx = troughs[i + 1]
        
        # Find peak between these troughs
        peaks_between = peaks[(peaks > trough_idx) & (peaks < next_trough_idx)]
        peak_idx = peaks_between[0] if len(peaks_between) > 0 else (trough_idx + next_trough_idx) // 2
        
        trough_time = times[trough_idx] if trough_idx < len(times) else np.nan
        peak_time = times[peak_idx] if peak_idx < len(times) else np.nan
        next_trough_time = times[next_trough_idx] if next_trough_idx < len(times) else np.nan
        
        interval_s = next_trough_time - trough_time
        resp_rate_bpm = 60.0 / interval_s if interval_s > 0 else np.nan
        
        # Calculate amplitude
        if trough_idx < len(rsp_clean) and peak_idx < len(rsp_clean):
            amplitude = abs(rsp_clean[peak_idx] - rsp_clean[trough_idx])
        else:
            amplitude = np.nan
        
        # Check overlap with bad intervals
        cycle_start = trough_time
        cycle_end = next_trough_time
        
        overlaps_bad = any(
            not (cycle_end <= bad_start or cycle_start >= bad_end)
            for bad_start, bad_end in bad_ranges
        )
        
        overlaps_interpolated = any(
            not (cycle_end <= interp_start or cycle_start >= interp_end)
            for interp_start, interp_end in interp_ranges
        )
        
        # Validity rules:
        # - valid_for_metrics: Can use if NOT overlapping un-interpolated bad zones
        #   (interpolated zones are OK for metrics)
        # - valid_for_coupling: Only valid if NO overlap with ANY bad zone
        #   (even interpolated zones distort phase)
        valid_for_metrics = not overlaps_bad  # Only remaining (non-interpolated) bad zones
        valid_for_coupling = not overlaps_bad and not overlaps_interpolated
        
        rows.append({
            "cycle_idx": i,
            "trough_time_s": trough_time,
            "peak_time_s": peak_time,
            "next_trough_time_s": next_trough_time,
            "interval_s": interval_s,
            "amplitude": amplitude,
            "resp_rate_bpm": resp_rate_bpm,
            "overlaps_bad": overlaps_bad,
            "overlaps_interpolated": overlaps_interpolated,
            "valid_for_metrics": valid_for_metrics,
            "valid_for_coupling": valid_for_coupling,
        })
    
    return pd.DataFrame(rows)


def calculate_metrics_excluding_bad_cycles(
    cycle_quality: pd.DataFrame,
    verbose: bool = True,
) -> Dict:
    """
    Calculate breathing metrics using only valid cycles.
    
    Returns dict with:
        - mean_resp_rate_bpm: Mean using only valid_for_metrics cycles
        - std_resp_rate_bpm: Std using only valid_for_metrics cycles  
        - median_resp_rate_bpm: Median using only valid_for_metrics cycles
        - mean_amplitude: Mean using only valid_for_metrics cycles
        - n_valid_cycles_metrics: Count of cycles valid for metrics
        - n_valid_cycles_coupling: Count of cycles valid for coupling
        - pct_valid_metrics: Percentage valid for metrics
        - pct_valid_coupling: Percentage valid for coupling
    """
    if len(cycle_quality) == 0:
        return {
            "mean_resp_rate_bpm": np.nan,
            "std_resp_rate_bpm": np.nan,
            "median_resp_rate_bpm": np.nan,
            "mean_amplitude": np.nan,
            "n_valid_cycles_metrics": 0,
            "n_valid_cycles_coupling": 0,
            "pct_valid_metrics": 0.0,
            "pct_valid_coupling": 0.0,
        }
    
    valid_for_metrics = cycle_quality[cycle_quality["valid_for_metrics"]]
    valid_for_coupling = cycle_quality[cycle_quality["valid_for_coupling"]]
    
    n_total = len(cycle_quality)
    n_metrics = len(valid_for_metrics)
    n_coupling = len(valid_for_coupling)
    
    result = {
        "mean_resp_rate_bpm": float(valid_for_metrics["resp_rate_bpm"].mean()) if n_metrics > 0 else np.nan,
        "std_resp_rate_bpm": float(valid_for_metrics["resp_rate_bpm"].std()) if n_metrics > 1 else np.nan,
        "median_resp_rate_bpm": float(valid_for_metrics["resp_rate_bpm"].median()) if n_metrics > 0 else np.nan,
        "mean_amplitude": float(valid_for_metrics["amplitude"].mean()) if n_metrics > 0 else np.nan,
        "n_valid_cycles_metrics": n_metrics,
        "n_valid_cycles_coupling": n_coupling,
        "pct_valid_metrics": n_metrics / n_total * 100 if n_total > 0 else 0.0,
        "pct_valid_coupling": n_coupling / n_total * 100 if n_total > 0 else 0.0,
    }
    
    if verbose:
        print(f"\n  Metrics (using {n_metrics}/{n_total} valid cycles, {result['pct_valid_metrics']:.1f}%):")
        print(f"    Mean RR: {result['mean_resp_rate_bpm']:.1f} bpm")
        print(f"    Median RR: {result['median_resp_rate_bpm']:.1f} bpm")
        print(f"    Std RR: {result['std_resp_rate_bpm']:.2f} bpm")
        print(f"  Coupling validity: {n_coupling}/{n_total} cycles ({result['pct_valid_coupling']:.1f}%)")
    
    return result


def limpiar(
    seg_dict: dict,
    aplicar_interpolacion: bool = False,
    max_cycles_interp: int = MAX_INTERP_CYCLES,
    max_duration_interp_s: float = MAX_INTERP_DURATION_S,
    mask_only: Optional[bool] = None,
) -> dict:
    """
    Apply cleaning based on marked bad intervals.
    Creates DUAL representations for different analysis purposes:
    
    1. rsp_masked: ALL BAD zones as NaN (for coupling/RSA analysis) - PRIMARY USE
    2. rsp_interp: Short BAD zones interpolated (for segment-level metrics) - OPTIONAL
    
    IMPORTANTE: Siguiendo NeuroKit2 y literatura, NO se interpola por defecto.
    La interpolación introduce bias, especialmente para gaps >2 ciclos.
    Usar rsp_masked (NaN) para RSA/coupling es el enfoque recomendado.
    
    Also builds per-cycle quality table to track which cycles can be used
    for metrics vs coupling analyses.
    
    Args:
        seg_dict: Segment dictionary
        aplicar_interpolacion: Whether to interpolate short bad zones (default: False, following literature)
        max_cycles_interp: Maximum breathing cycles to interpolate (default 2)
        max_duration_interp_s: Maximum seconds to interpolate (default 6s)
        mask_only: If True, skip interpolation and rely on masked signal for conservative workflows
    """
    seg_name = seg_dict.get("seg", "SEG")
    
    bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
    rsp_clean = seg_dict.get("rsp_clean", np.array([]))
    fs = seg_dict.get("fs_rsp", 50)
    
    # Estimate typical breath duration
    breath_intervals = seg_dict.get("breath_intervals_s", np.array([]))
    if len(breath_intervals) > 0:
        typical_dur = float(np.median(breath_intervals))
    else:
        typical_dur = TYPICAL_BREATH_DURATION_S
    
    # Store all bad intervals BEFORE any interpolation (for coupling mask)
    all_bad_zones = []
    if len(bad_intervals) > 0:
        for _, row in bad_intervals.iterrows():
            all_bad_zones.append((row["t_ini_s"], row["t_fin_s"], row.get("motivo", "manual")))
    seg_dict["all_bad_intervals"] = all_bad_zones.copy()

    mask_only = seg_dict.get("mask_only", False) if mask_only is None else mask_only
    mask_only = bool(mask_only)
    should_build_dual = len(all_bad_zones) > 0 and (aplicar_interpolacion or mask_only)

    if should_build_dual:
        if mask_only:
            print("  Mask-only: no interpolation")
        
        # Create both representations
        rsp_interp, rsp_masked, interpolated, remaining = create_dual_representations(
            rsp_clean, fs, bad_intervals,
            max_interp_cycles=max_cycles_interp,
            max_interp_duration_s=max_duration_interp_s,
            typical_breath_dur=typical_dur,
            verbose=True,
            mask_only=mask_only
        )
        
        # Store original clean signal before overwriting (for visualization)
        seg_dict["rsp_clean_original"] = rsp_clean.copy()
        
        # Store both representations
        seg_dict["rsp_interp"] = rsp_interp      # For metrics
        seg_dict["rsp_masked"] = rsp_masked      # For coupling (NaN where bad)
        seg_dict["rsp_clean"] = rsp_interp       # Default to interp for backward compat
        seg_dict["interpolated_intervals"] = interpolated
        
        # Update bad_intervals to only include non-interpolated (remaining)
        seg_dict["bad_intervals"] = pd.DataFrame(columns=["t_ini_s", "t_fin_s", "motivo", "sev"])
        for start_s, end_s, reason in remaining:
            seg_dict = marcar_bad_interval(seg_dict, start_s, end_s, motivo=reason)
        
        # Re-detect peaks on interpolated signal
        # Apply BH amplitude threshold if this is a BH segment
        seg_name = seg_dict.get("seg", "SEG")
        min_amplitude_threshold = None
        if "BH" in seg_name:
            signal_range = np.ptp(rsp_interp)
            min_amplitude_threshold = signal_range * BH_MIN_AMPLITUDE_THRESHOLD_FRACTION
        
        peaks, troughs = detect_breathing_cycles(
            rsp_interp, fs,
            min_amplitude_threshold=min_amplitude_threshold
        )
        seg_dict["peaks"] = peaks
        seg_dict["troughs"] = troughs
        seg_dict["phase_signal"] = calculate_phase_signal(peaks, troughs, len(rsp_interp))
        
        # Update basic metrics
        if len(troughs) > 1:
            metrics = calculate_breathing_metrics(rsp_interp, peaks, troughs, fs)
            seg_dict.update(metrics)
        
        if len(interpolated) > 0:
            print(f"  Interpoladas: {len(interpolated)} zona(s)")
        if len(remaining) > 0:
            print(f"  Sin interpolar: {len(remaining)} zona(s) (muy largas)")
    else:
        # No bad zones or interpolation disabled - create identical dual representations
        seg_dict["rsp_interp"] = rsp_clean.copy()
        seg_dict["rsp_masked"] = rsp_clean.copy()
        seg_dict["interpolated_intervals"] = []
    
    # Build cycle quality table
    cycle_quality = build_cycle_quality_table(seg_dict, bad_intervals=seg_dict.get("bad_intervals", pd.DataFrame()))
    seg_dict["cycle_quality"] = cycle_quality
    
    # Calculate metrics using only valid cycles
    if len(cycle_quality) > 0:
        valid_metrics = calculate_metrics_excluding_bad_cycles(cycle_quality, verbose=True)
        seg_dict["mean_resp_rate_bpm"] = valid_metrics["mean_resp_rate_bpm"]
        seg_dict["std_resp_rate_bpm"] = valid_metrics["std_resp_rate_bpm"]
        seg_dict["median_resp_rate_bpm"] = valid_metrics["median_resp_rate_bpm"]
        seg_dict["mean_amplitude"] = valid_metrics["mean_amplitude"]
        seg_dict["n_valid_cycles_metrics"] = valid_metrics["n_valid_cycles_metrics"]
        seg_dict["n_valid_cycles_coupling"] = valid_metrics["n_valid_cycles_coupling"]
        seg_dict["pct_valid_metrics"] = valid_metrics["pct_valid_metrics"]
        seg_dict["pct_valid_coupling"] = valid_metrics["pct_valid_coupling"]
    
    # Calculate QC flags
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    seg_dict["clean_s"] = dur_s - bad_s if np.isfinite(bad_s) else dur_s
    
    # Quality flags for different analysis types
    seg_dict["qc_valid"] = bool(bad_pct <= BAD_PROP_THRESHOLD)
    seg_dict["qc_valid_metrics"] = bool(bad_pct <= BAD_PROP_THRESHOLD_METRICS)
    seg_dict["qc_valid_coupling"] = bool(
        bad_pct <= BAD_PROP_THRESHOLD_COUPLING and
        seg_dict.get("pct_valid_coupling", 0) >= 90  # At least 90% cycles valid
    )
    
    print(f"\n--- Quality Summary ---")
    print(f"  BAD: {bad_s:.1f}s ({bad_pct*100:.1f}%)")
    print(f"  Status for metrics:  {'✔ VALID' if seg_dict['qc_valid_metrics'] else '✘ EXCLUDED'} (threshold: {BAD_PROP_THRESHOLD_METRICS*100:.0f}%)")
    print(f"  Status for coupling: {'✔ VALID' if seg_dict['qc_valid_coupling'] else '✘ EXCLUDED'} (threshold: {BAD_PROP_THRESHOLD_COUPLING*100:.0f}% + 90% valid cycles)")
    
    return seg_dict


def limpiar_segmento_lb(
    seg_dict: dict,
    t0_edicion: float = 180.0,
    win_len_edicion: float = 60.0,
    aplicar_interpolacion: bool = False,
    mostrar_intermedios: bool = False,
) -> dict:
    """
    Flujo completo de limpieza para segmento LB basado en NeuroKit2 y literatura.
    
    Flujo recomendado (basado en NeuroKit2, Khodadad et al. 2018, literatura RSA):
    1. Inspección inicial (ya realizada antes de llamar esta función)
    2. Edición manual de picos/troughs anómalos
    3. Conversión automática de eliminaciones a zonas BAD (agrupa picos cercanos)
    4. Crear representaciones duales:
       - rsp_masked (NaN) → Para RSA/coupling (uso principal, NO interpolar)
       - rsp_interp → Solo si aplicar_interpolacion=True (opcional)
    5. Validación visual del resultado
    
    ENFOQUE CONSERVADOR (NeuroKit2):
    - NO interpola por defecto (aplicar_interpolacion=False)
    - Usa NaN/mask para zonas BAD (recomendado para RSA/coupling)
    - Interpolar solo si necesitas métricas segmentales específicas
    
    Args:
        seg_dict: Segment dictionary (debe tener resultado de inspeccionar_segmento_rsp)
        t0_edicion: Tiempo inicial para ventana de edición (default: 180s)
        win_len_edicion: Duración de ventana de edición (default: 60s)
        aplicar_interpolacion: Si True, interpola zonas cortas (default: False, siguiendo literatura)
        mostrar_intermedios: Si True, muestra señales en cada paso (default: False)
    
    Returns:
        seg_dict con señal limpia (rsp_masked con NaN en zonas BAD)
    
    Example:
        # 1. Inspección inicial
        resultado_LB = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name="LB")
        ver_señal_completa(resultado_LB, show_bad=False)
        
        # 2. Limpieza completa (enfoque conservador, NO interpolar)
        resultado_LB = limpiar_segmento_lb(resultado_LB, t0_edicion=180.0, win_len_edicion=60.0)
        
        # 3. Ver resultado final (con NaN en zonas BAD)
        ver_señal_limpia(resultado_LB)
        
        # 4. Usar rsp_masked para RSA/coupling (contiene NaN en zonas BAD)
    """
    seg_name = seg_dict.get("seg", "SEG")
    
    # Paso 2: Edición manual de picos/troughs
    seg_dict = editar_ciclos_interactivos_externo(
        seg_dict, enable=True, t0=t0_edicion, win_len=win_len_edicion
    )
    if mostrar_intermedios:
        ver_señal_completa(seg_dict, show_bad=False)
    
    # Paso 3: Convertir eliminaciones a zonas BAD (agrupa picos cercanos)
    seg_dict = convertir_cambios_a_bad_zones(seg_dict)
    if mostrar_intermedios:
        ver_señal_completa(seg_dict, show_bad=True)
    
    # Paso 4: Crear representaciones (NaN por defecto, sin interpolar)
    seg_dict = limpiar(
        seg_dict,
        aplicar_interpolacion=aplicar_interpolacion,
        max_cycles_interp=MAX_INTERP_CYCLES,
        max_duration_interp_s=MAX_INTERP_DURATION_S,
    )
    
    return seg_dict



def recortar_segmento_rsp(seg_dict: dict, target_duration_s: float = 300.0,
                          verbose: bool = True) -> dict:
    """
    Trim segment to maximum target_duration_s by removing data from the END.
    Only trims if segment is LONGER than target_duration_s.
    
    For LB and Recovery:
        - If segment > 300s → trim to 300s from end
        - If segment ≤ 300s → keep as is
    
    Args:
        target_duration_s: Maximum duration in seconds (default 300s)
    
    Returns: seg_dict (trimmed if > target, unchanged if ≤ target)
    """
    seg_name = seg_dict.get("seg", "SEG")
    times = seg_dict.get("times", np.array([]))
    
    if len(times) == 0:
        if verbose:
            print(f"[Trim] {seg_name}: No data")
        return seg_dict
    
    current_dur = float(times[-1])
    
    if current_dur <= target_duration_s:
        if verbose:
            print(f"[Trim] {seg_name}: Already {current_dur:.1f}s (≤ {target_duration_s}s)")
        return seg_dict
    
    # Find index where time exceeds target
    target_idx = np.searchsorted(times, target_duration_s)
    
    if verbose:
        print(f"[Trim] {seg_name}: {current_dur:.1f}s → {target_duration_s:.0f}s (removing last {current_dur - target_duration_s:.1f}s)")
    
    # Trim arrays
    seg_dict["times"] = times[:target_idx]
    seg_dict["rsp_raw"] = seg_dict.get("rsp_raw", np.array([]))[:target_idx]
    seg_dict["rsp_clean"] = seg_dict.get("rsp_clean", np.array([]))[:target_idx]
    seg_dict["phase_signal"] = seg_dict.get("phase_signal", np.array([]))[:target_idx]
    if "rsp_interp" in seg_dict:
        seg_dict["rsp_interp"] = seg_dict["rsp_interp"][:target_idx]
    if "rsp_masked" in seg_dict:
        seg_dict["rsp_masked"] = seg_dict["rsp_masked"][:target_idx]
    
    # Trim peaks/troughs that are beyond target duration
    fs = seg_dict.get("fs_rsp", 50)
    max_idx = target_idx
    
    peaks = seg_dict.get("peaks", np.array([]))
    troughs = seg_dict.get("troughs", np.array([]))
    
    if len(peaks) > 0:
        seg_dict["peaks"] = peaks[peaks < max_idx]
    if len(troughs) > 0:
        seg_dict["troughs"] = troughs[troughs < max_idx]
    
    # Recalculate metrics with trimmed data
    peaks_trimmed = seg_dict["peaks"]
    troughs_trimmed = seg_dict["troughs"]
    
    if len(troughs_trimmed) > 1:
        rsp_clean_trimmed = seg_dict["rsp_clean"]
        metrics = calculate_breathing_metrics(rsp_clean_trimmed, peaks_trimmed, troughs_trimmed, fs)
        seg_dict.update(metrics)
        seg_dict["mean_resp_rate_bpm"] = float(np.mean(metrics["resp_rate_bpm"])) if len(metrics["resp_rate_bpm"]) > 0 else np.nan
        seg_dict["std_resp_rate_bpm"] = float(np.std(metrics["resp_rate_bpm"])) if len(metrics["resp_rate_bpm"]) > 0 else np.nan
        seg_dict["mean_ie_ratio"] = float(np.mean(metrics["ie_ratios"])) if len(metrics["ie_ratios"]) > 0 else np.nan
    
    # Update duration
    seg_dict["dur_s"] = target_duration_s
    
    # Trim bad_intervals
    bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
    if len(bad_intervals) > 0:
        mask = bad_intervals["t_fin_s"] <= target_duration_s
        seg_dict["bad_intervals"] = bad_intervals[mask].reset_index(drop=True)
    
    # Recalculate bad percentage
    bad_pct, bad_s, dur_s = calcular_bad_pct(seg_dict)
    seg_dict["bad_pct"] = bad_pct
    seg_dict["bad_s"] = bad_s
    
    return seg_dict


def guardar_segmento(seg_dict: dict, resultados: dict, seg_name: str,
                     target_duration_s: Optional[float] = None) -> Tuple[dict, dict]:
    """
    Save segment to results dictionary. Optionally trims to target duration.
    
    For BH segments with bh_end_time_s marker, automatically creates both:
    - Full version: "BH X" (original signal with recovery breathing)
    - Trimmed version: "BH X_trimmed" (only breath-hold period, for HRV analysis)
    
    For ALL segments (non-BH), automatically creates dual signal representations:
    - rsp_interp: Short bad zones interpolated (for robust segment-level metrics)
    - rsp_masked: All bad zones as NaN (for time-coupled analyses like RSA)
    
    Args:
        seg_dict: Segment dictionary
        resultados: Results dictionary
        seg_name: Segment name
        target_duration_s: Target duration (e.g., 300.0 for LB/Recovery)
    """
    # Trim to target duration if specified (for LB/Recovery segments)
    if target_duration_s is not None:
        seg_dict = recortar_segmento_rsp(seg_dict, target_duration_s=target_duration_s, verbose=True)
    
    # For non-BH segments: automatically create dual representations if they don't exist
    if "BH" not in seg_name:
        if "rsp_interp" not in seg_dict or "rsp_masked" not in seg_dict:
            rsp_clean = seg_dict.get("rsp_clean", np.array([]))
            fs = seg_dict.get("fs_rsp", 50)
            bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame())
            
            if len(rsp_clean) > 0 and len(bad_intervals) > 0:
                rsp_interp, rsp_masked, interpolated_intervals, remaining_bad = create_dual_representations(
                    rsp_clean, fs, bad_intervals,
                    max_interp_cycles=MAX_INTERP_CYCLES,
                    max_interp_duration_s=MAX_INTERP_DURATION_S,
                    verbose=False  # Don't print during auto-creation
                )
                seg_dict["rsp_interp"] = rsp_interp
                seg_dict["rsp_masked"] = rsp_masked
                seg_dict["interpolated_intervals"] = interpolated_intervals
                
                # IMPORTANT: Recalculate peaks, troughs, and metrics using rsp_interp
                # This ensures outliers detection uses the cleaned signal, not the original
                peaks, troughs = detect_breathing_cycles(rsp_interp, fs)
                seg_dict["peaks"] = peaks
                seg_dict["troughs"] = troughs
                seg_dict["phase_signal"] = calculate_phase_signal(peaks, troughs, len(rsp_interp))
                
                # Recalculate metrics using rsp_interp
                if len(troughs) > 1:
                    metrics = calculate_breathing_metrics(rsp_interp, peaks, troughs, fs)
                    seg_dict.update(metrics)
            else:
                # If no bad intervals, both representations are the same as clean
                seg_dict["rsp_interp"] = rsp_clean.copy()
                seg_dict["rsp_masked"] = rsp_clean.copy()
                seg_dict["interpolated_intervals"] = []
    
    # Save full version
    resultados[seg_name] = seg_dict
    
    # For BH segments: automatically create trimmed version if bh_end_time_s is set
    if "BH" in seg_name and seg_dict.get("bh_end_time_s") is not None:
        seg_trimmed = crear_version_recortada_bh(seg_dict, verbose=True)
        if seg_trimmed is not None:
            trimmed_name = f"{seg_name}_trimmed"
            resultados[trimmed_name] = seg_trimmed
            # Don't print additional messages for BH segments - only the "Created trimmed version" message above
    
    # Only print detailed save message for non-BH segments
    if "BH" not in seg_name:
        bad_pct = seg_dict.get("bad_pct", 0) * 100
        mean_rr = seg_dict.get("mean_resp_rate_bpm", 0)
        n_breaths = len(seg_dict.get("troughs", []))
        dur_final = seg_dict.get("dur_s", 0)
        print(f"✔ {seg_name} saved: {n_breaths} breaths, {mean_rr:.1f} bpm, {dur_final:.1f}s, BAD={bad_pct:.1f}%")
    
    return seg_dict, resultados


# =============================================================================
# OUTLIER DETECTION & VERIFICATION
# =============================================================================

def detectar_outliers_rsp(seg_dict: dict, 
                          rr_min_bpm: float = 4.0,
                          rr_max_bpm: float = 60.0,
                          iqr_factor: float = 3.0,
                          verbose: bool = True) -> Tuple[dict, dict]:
    """
    Detect outliers in respiratory rate using IQR-based method and physiological limits.
    
    IMPORTANT: Uses resp_rate_bpm and breath_times_s from seg_dict, which should be
    calculated from the cleaned signal (rsp_interp) after limpiar() has been called.
    These values are updated in limpiar() using the cleaned/interpolated signal.
    
    Args:
        seg_dict: Segment dictionary (should contain cleaned signal metrics)
        rr_min_bpm: Minimum physiological respiratory rate (4 bpm = very slow breathing)
        rr_max_bpm: Maximum physiological respiratory rate (60 bpm = extreme exercise)
        iqr_factor: Factor for IQR-based outlier detection
        verbose: Print results
    
    Returns:
        seg_dict: Updated with outliers field
        outliers: Dict with outlier indices and values
    """
    seg_name = seg_dict.get("seg", "SEG")
    # These values come from the cleaned signal (rsp_interp) after limpiar()
    resp_rate = seg_dict.get("resp_rate_bpm", np.array([]))
    breath_times = seg_dict.get("breath_times_s", np.array([]))
    
    outliers = {
        "rr_too_low": [],
        "rr_too_high": [],
        "rr_iqr_low": [],
        "rr_iqr_high": [],
    }
    
    if len(resp_rate) < 4:
        if verbose:
            print(f"[Outliers] {seg_name}: Not enough data")
        return seg_dict, outliers
    
    # IQR-based outliers
    q1 = float(np.nanpercentile(resp_rate, 25))
    q3 = float(np.nanpercentile(resp_rate, 75))
    iqr = q3 - q1
    rr_low = q1 - iqr_factor * iqr
    rr_high = q3 + iqr_factor * iqr
    
    iqr_enabled = len(resp_rate) >= MIN_IQR_OUTLIER_CYCLES
    if not iqr_enabled and verbose:
        print(f"  ℹ️  IQR outlier detection skipped ({len(resp_rate)} breaths < {MIN_IQR_OUTLIER_CYCLES} threshold)")

    for i, val in enumerate(resp_rate):
        # Physiological limits
        if val < rr_min_bpm:
            outliers["rr_too_low"].append((i, val, breath_times[i] if i < len(breath_times) else np.nan))
        elif val > rr_max_bpm:
            outliers["rr_too_high"].append((i, val, breath_times[i] if i < len(breath_times) else np.nan))
        # IQR-based
        elif iqr_enabled:
            if val < rr_low:
                outliers["rr_iqr_low"].append((i, val, breath_times[i] if i < len(breath_times) else np.nan))
            elif val > rr_high:
                outliers["rr_iqr_high"].append((i, val, breath_times[i] if i < len(breath_times) else np.nan))
    
    seg_dict["outliers"] = outliers
    
    seg_dict["iqr_outliers_skipped"] = not iqr_enabled
    if verbose:
        total = sum(len(v) for v in outliers.values())
        print(f"\n[Outliers] {seg_name}: {total} outliers")
        print(f"  RR IQR: valid range = [{rr_low:.1f}, {rr_high:.1f}] bpm")
        print(f"  RR < {rr_min_bpm} bpm (too slow): {len(outliers['rr_too_low'])}")
        print(f"  RR > {rr_max_bpm} bpm (too fast): {len(outliers['rr_too_high'])}")
        print(f"  RR < IQR limit: {len(outliers['rr_iqr_low'])}")
        print(f"  RR > IQR limit: {len(outliers['rr_iqr_high'])}")
    
    return seg_dict, outliers


def listar_outliers(seg_dict: dict) -> pd.DataFrame:
    """
    Return a DataFrame summarizing detected RR outliers (idx, value, time).
    """
    outliers = seg_dict.get("outliers", {})
    rows = []
    for category, items in outliers.items():
        for idx, value, time_s in items:
            rows.append({
                "category": category,
                "idx": int(idx),
                "value": float(value),
                "time_s": float(time_s) if not np.isnan(time_s) else np.nan,
            })
    return pd.DataFrame(rows)


def sugerir_accion_para_outliers(outliers_df: pd.DataFrame, max_keep: int = 2) -> str:
    """
    Suggest whether to keep or remove outliers based on counts/severity.
    """
    if len(outliers_df) == 0:
        return "keep"
    n_high_rr = len(outliers_df[outliers_df["category"] == "rr_too_high"])
    n_low_rr = len(outliers_df[outliers_df["category"] == "rr_too_low"])
    n_iqr = len(outliers_df[outliers_df["category"].isin(["rr_iqr_low", "rr_iqr_high"])])
    if n_high_rr + n_low_rr > max_keep or n_iqr > max_keep:
        return "review_remove"
    return "review_keep"


def verificar_outliers_todos(resultados: dict,
                             rr_min_bpm: float = 4.0,
                             rr_max_bpm: float = 60.0,
                             mean_rr_range: Tuple[float, float] = (8, 30),
                             verbose: bool = True) -> Tuple[dict, bool]:
    """
    Verify outliers in all segments before export.
    
    IMPORTANT: Uses mean_resp_rate_bpm and bad_pct from seg_dict, which are calculated
    from the cleaned signal (rsp_interp) after limpiar() has been called. These values
    reflect the final cleaned state of the signal stored in resultados.
    
    Args:
        resultados: Dict with all processed segments (should contain cleaned signals)
        rr_min_bpm: Minimum physiological RR
        rr_max_bpm: Maximum physiological RR
        mean_rr_range: Valid range for mean respiratory rate
        verbose: Print results
    
    Returns:
        resultados: Updated with outliers
        hay_problemas: True if any segment has issues
    """
    print("\n" + "="*70)
    print("OUTLIER VERIFICATION FOR ALL SEGMENTS")
    print("="*70)
    
    hay_problemas = False
    segmentos_con_problemas = []
    
    for seg_name, seg_dict in resultados.items():
        # Skip BH segments (breath-hold has intentionally abnormal rates)
        if "BH" in seg_name:
            continue
        
        seg_dict, outliers = detectar_outliers_rsp(
            seg_dict, rr_min_bpm=rr_min_bpm, rr_max_bpm=rr_max_bpm, verbose=False
        )
        resultados[seg_name] = seg_dict
        
        total_outliers = sum(len(v) for v in outliers.values())
        mean_rr = seg_dict.get("mean_resp_rate_bpm", np.nan)
        bad_pct = seg_dict.get("bad_pct", 0) * 100
        
        # Check validity
        mean_valid = mean_rr_range[0] <= mean_rr <= mean_rr_range[1] if not np.isnan(mean_rr) else False
        
        if total_outliers == 0 and mean_valid and bad_pct <= BAD_PROP_THRESHOLD * 100:
            if verbose:
                print(f"{seg_name}: ✔ OK (RR={mean_rr:.1f} bpm, BAD={bad_pct:.1f}%)")
        else:
            hay_problemas = True
            segmentos_con_problemas.append(seg_name)
            if verbose:
                print(f"\n{seg_name}: ⚠️ CHECK REQUIRED")
                print(f"  Mean RR: {mean_rr:.1f} bpm (valid: {mean_rr_range[0]}-{mean_rr_range[1]})")
                print(f"  BAD: {bad_pct:.1f}% (threshold: {BAD_PROP_THRESHOLD*100:.0f}%)")
                print(f"  Outliers: {total_outliers}")
                if total_outliers > 0:
                    outliers_df = listar_outliers(seg_dict)
                    non_iqr = outliers_df[~outliers_df["category"].isin({"rr_iqr_low", "rr_iqr_high"})]
                    iqr_only = outliers_df[outliers_df["category"].isin({"rr_iqr_low", "rr_iqr_high"})]
                    
                    if not non_iqr.empty:
                        print("    Details:")
                        print(non_iqr[["category", "time_s", "value"]].to_string(index=False))
                        rec = sugerir_accion_para_outliers(non_iqr)
                        print(f"    Recommendation: {rec} (use `reinspeccionar_segmento(resultados['{seg_name}'])` to review and mark)")
                    else:
                        if seg_dict.get("iqr_outliers_skipped", False):
                            print("    Only physiological outliers flagged, but IQR was skipped (few cycles).")
                        else:
                            # Show IQR outliers time range even if they're "soft" outliers
                            if not iqr_only.empty:
                                times = iqr_only["time_s"].dropna()
                                if len(times) > 0:
                                    t_min = times.min()
                                    t_max = times.max()
                                    if t_min == t_max:
                                        print(f"    Only IQR-based outliers detected at {t_min:.1f}s (few-cycle variations). Consider keeping.")
                                    else:
                                        print(f"    Only IQR-based outliers detected in range [{t_min:.1f}s - {t_max:.1f}s] (few-cycle variations). Consider keeping.")
                                else:
                                    print("    Only IQR-based outliers detected (few-cycle variations). Consider keeping.")
                            else:
                                print("    Only IQR-based outliers detected (few-cycle variations). Consider keeping.")
    
    print("\n" + "-"*70)
    if hay_problemas:
        print(f"⚠️  Segments requiring review: {segmentos_con_problemas}")
        print("Consider going back and cleaning these segments before export.")
    else:
        print("✔ All segments OK - ready for export")
    
    return resultados, hay_problemas


def resumen_antes_exportar(resultados: dict) -> pd.DataFrame:
    """
    Print summary table before export.
    """
    print("\n" + "="*80)
    print("RESUMEN ANTES DE EXPORTAR")
    print("="*80)
    print(f"\n{'Segment':<15} {'Breaths':>8} {'Duration':>10} {'BAD%':>8} {'Mean RR':>10} {'I:E':>8} {'Status':>10}")
    print("-"*80)
    
    rows = []
    for seg_name, seg_dict in resultados.items():
        n_breaths = len(seg_dict.get("troughs", []))
        dur = seg_dict.get("dur_s", 0)
        bad_pct = seg_dict.get("bad_pct", 0) * 100
        mean_rr = seg_dict.get("mean_resp_rate_bpm", np.nan)
        mean_ie = seg_dict.get("mean_ie_ratio", np.nan)
        qc_valid = seg_dict.get("qc_valid", True)
        
        if seg_name.startswith("BH"):
            continue
        status = "✔ VALID" if qc_valid else "❌ EXCLUDED"
        
        print(f"{seg_name:<15} {n_breaths:>8} {dur:>9.1f}s {bad_pct:>7.1f}% {mean_rr:>9.1f} {mean_ie:>7.2f} {status:>10}")
        
        rows.append({
            "segment": seg_name,
            "breaths": n_breaths,
            "duration_s": dur,
            "bad_pct": bad_pct,
            "mean_rr_bpm": mean_rr,
            "mean_ie_ratio": mean_ie,
            "qc_valid": qc_valid,
        })
    
    print("-"*80)
    print(f"Total segments: {len(resultados)}")
    
    valid_count = sum(1 for r in rows if r["qc_valid"])
    print(f"Valid: {valid_count} | Excluded: {len(rows) - valid_count}")
    
    return pd.DataFrame(rows)


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def exportar_sujeto_completo_rsp(
    resultados: dict,
    sujeto_id: str,
    output_folder: str,
    verbose: bool = True,
) -> str:
    """
    Export all RSP segments for one subject to Excel.
    
    Creates sheets:
    - One sheet per segment with time series (rsp_interp, rsp_masked, phase, peaks, troughs)
    - One sheet per segment with cycle quality table (for RSA analysis decisions)
    - RESUMEN sheet with segment-level summary including quality flags
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"RSP_data_sujeto_{sujeto_id}.xlsx")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_rows = []
        bad_intervals_all = []  # Collect all bad intervals
        manual_changes_all = []  # Collect all manual changes
        
        for seg_name, seg_dict in resultados.items():
            times = seg_dict.get("times", np.array([]))
            if len(times) == 0:
                continue
            
            # Get clean signal
            rsp_clean = seg_dict.get("rsp_clean", seg_dict.get("rsp_interp", np.array([])))
            
            # Time series sheet - only time and clean signal
            df_seg = pd.DataFrame({
                "time_s": times,
                "rsp_clean": rsp_clean[:len(times)] if len(rsp_clean) >= len(times) else np.full(len(times), np.nan),
            })
            
            sheet_name = seg_name[:31].replace("/", "_").replace("\\", "_")
            df_seg.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Collect bad intervals for aggregated sheet
            bad_intervals = seg_dict.get("bad_intervals", pd.DataFrame(columns=["t_ini_s", "t_fin_s", "motivo", "sev"]))
            if isinstance(bad_intervals, pd.DataFrame) and len(bad_intervals) > 0:
                for _, row in bad_intervals.iterrows():
                    bad_intervals_all.append({
                        "segment": seg_name,
                        "start_time_s": round(row["t_ini_s"], 3),
                        "end_time_s": round(row["t_fin_s"], 3),
                        "duration_s": round(row["t_fin_s"] - row["t_ini_s"], 3),
                        "reason": row.get("motivo", "artifact"),
                        "severity": row.get("sev", ""),
                    })
            
            # Collect interpolated intervals for aggregated sheet
            interpolated_intervals = seg_dict.get("interpolated_intervals", [])
            if isinstance(interpolated_intervals, list) and len(interpolated_intervals) > 0:
                for interval in interpolated_intervals:
                    if isinstance(interval, (list, tuple)) and len(interval) >= 2:
                        bad_intervals_all.append({
                            "segment": seg_name,
                            "start_time_s": round(interval[0], 3),
                            "end_time_s": round(interval[1], 3),
                            "duration_s": round(interval[1] - interval[0], 3),
                            "reason": "interpolated",
                            "severity": "",
                        })
            
            # Collect manual peak/trough changes for aggregated sheet
            changes = seg_dict.get("peak_trough_changes", {})
            if changes:
                for time_s in changes.get("peaks_added", []):
                    manual_changes_all.append({
                        "segment": seg_name,
                        "type": "peak",
                        "action": "added",
                        "time_s": round(time_s, 3),
                    })
                for time_s in changes.get("peaks_removed", []):
                    manual_changes_all.append({
                        "segment": seg_name,
                        "type": "peak",
                        "action": "removed",
                        "time_s": round(time_s, 3),
                    })
                for time_s in changes.get("troughs_added", []):
                    manual_changes_all.append({
                        "segment": seg_name,
                        "type": "trough",
                        "action": "added",
                        "time_s": round(time_s, 3),
                    })
                for time_s in changes.get("troughs_removed", []):
                    manual_changes_all.append({
                        "segment": seg_name,
                        "type": "trough",
                        "action": "removed",
                        "time_s": round(time_s, 3),
                    })
            
            # Cycle quality sheet (for RSA analysis decisions)
            cycle_quality = seg_dict.get("cycle_quality", pd.DataFrame())
            if len(cycle_quality) > 0:
                cycle_sheet_name = f"{seg_name[:25]}_cycles"[:31].replace("/", "_").replace("\\", "_")
                cycle_quality.to_excel(writer, sheet_name=cycle_sheet_name, index=False)
            
            # Summary row with quality flags
            is_trimmed = seg_dict.get("is_trimmed", False)
            dur_full = seg_dict.get("dur_s_full", seg_dict.get("dur_s", 0)) if is_trimmed else seg_dict.get("dur_s", 0)
            
            if seg_name.startswith("BH"):
                # For BH segments: only include original and trimmed times, all other metrics as nan
                summary_rows.append({
                    "segment": seg_name,
                    "duration_s": seg_dict.get("dur_s", 0),
                    "duration_full_s": dur_full if is_trimmed else np.nan,  # Original duration for trimmed versions
                    "bh_end_time_s": seg_dict.get("bh_end_time_s", np.nan),  # Breath-hold end marker (for HRV trimming)
                    "n_breaths": np.nan,
                    "n_valid_cycles_metrics": np.nan,
                    "n_valid_cycles_coupling": np.nan,
                    "pct_valid_metrics": np.nan,
                    "pct_valid_coupling": np.nan,
                    "bad_pct": np.nan,
                    "mean_resp_rate_bpm": np.nan,
                    "median_resp_rate_bpm": np.nan,
                    "std_resp_rate_bpm": np.nan,
                    "mean_amplitude": np.nan,
                    "mean_ie_ratio": np.nan,
                    "qc_valid_metrics": np.nan,
                    "qc_valid_coupling": np.nan,
                    "qc_valid_overall": np.nan,
                })
            else:
                summary_rows.append({
                    "segment": seg_name,
                    "duration_s": seg_dict.get("dur_s", 0),
                    "duration_full_s": dur_full if is_trimmed else np.nan,  # Original duration for trimmed versions
                    "bh_end_time_s": seg_dict.get("bh_end_time_s", np.nan),  # Breath-hold end marker (for HRV trimming)
                    "n_breaths": len(seg_dict.get("troughs", [])),
                    "n_valid_cycles_metrics": seg_dict.get("n_valid_cycles_metrics", len(seg_dict.get("troughs", [])) - 1 if len(seg_dict.get("troughs", [])) > 1 else 0),
                    "n_valid_cycles_coupling": seg_dict.get("n_valid_cycles_coupling", 0),
                    "pct_valid_metrics": seg_dict.get("pct_valid_metrics", 100.0),
                    "pct_valid_coupling": seg_dict.get("pct_valid_coupling", 100.0),
                    "bad_pct": seg_dict.get("bad_pct", 0) * 100,
                    "mean_resp_rate_bpm": seg_dict.get("mean_resp_rate_bpm", np.nan),
                    "median_resp_rate_bpm": seg_dict.get("median_resp_rate_bpm", np.nan),
                    "std_resp_rate_bpm": seg_dict.get("std_resp_rate_bpm", np.nan),
                    "mean_amplitude": seg_dict.get("mean_amplitude", np.nan),
                    "mean_ie_ratio": seg_dict.get("mean_ie_ratio", np.nan),
                    "qc_valid_metrics": seg_dict.get("qc_valid_metrics", True),
                    "qc_valid_coupling": seg_dict.get("qc_valid_coupling", True),
                    "qc_valid_overall": seg_dict.get("qc_valid", True),
                })
        
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="RESUMEN", index=False)
        
        # Add BAD_intervals sheet (aggregated across all segments)
        if bad_intervals_all:
            df_bad = pd.DataFrame(bad_intervals_all)
            df_bad = df_bad.sort_values(["segment", "start_time_s"])
            df_bad.to_excel(writer, sheet_name="BAD_intervals", index=False)
            if verbose:
                print(f"  ✓ BAD_intervals sheet: {len(df_bad)} intervals")
        
        # Add Manual_changes sheet (aggregated across all segments)
        if manual_changes_all:
            df_changes = pd.DataFrame(manual_changes_all)
            df_changes = df_changes.sort_values(["segment", "time_s"])
            df_changes.to_excel(writer, sheet_name="Manual_changes", index=False)
            if verbose:
                print(f"  ✓ Manual_changes sheet: {len(df_changes)} changes")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EXPORTED: {output_path}")
        print(f"Segments: {len(summary_rows)}")
        print(f"{'='*60}")
        print("\nSheets created:")
        print("  - Per segment: time series (time_s, rsp_clean)")
        print("  - Per segment: _cycles sheet with per-cycle quality flags")
        print("  - RESUMEN: segment-level summary with quality flags")
        print("  - BAD_intervals: all bad/interpolated intervals across segments")
        print("  - Manual_changes: all manual peak/trough edits across segments")
    
    return output_path


def exportar_y_resumir_rsp(
    resultados: dict,
    sujeto_id: str,
    output_folder: str,
    archivo_resumen: str,
) -> str:
    """Export to Excel and save cumulative summary with quality flags."""
    os.makedirs(output_folder, exist_ok=True)
    
    output_file = exportar_sujeto_completo_rsp(resultados, sujeto_id, output_folder)
    
    # Print summary with quality flags
    print("\n" + "="*90)
    print(f"FINAL SUMMARY - SUBJECT {sujeto_id}")
    print("="*90)
    print(f"\n{'Segment':<12} {'Dur':>6} {'BAD%':>6} {'MeanRR':>7} {'Cycles':>7} {'Metr%':>6} {'Coup%':>6} {'QC_M':>5} {'QC_C':>5}")
    print("-"*90)
    
    resumen_filas = []
    for seg_name, seg_dict in resultados.items():
        dur = seg_dict.get("dur_s", 0)
        bad_pct = seg_dict.get("bad_pct", 0) * 100
        mean_rr = seg_dict.get("mean_resp_rate_bpm", np.nan)
        n_breaths = len(seg_dict.get("troughs", []))
        pct_metrics = seg_dict.get("pct_valid_metrics", 100.0)
        pct_coupling = seg_dict.get("pct_valid_coupling", 100.0)
        qc_metrics = "✔" if seg_dict.get("qc_valid_metrics", True) else "✘"
        qc_coupling = "✔" if seg_dict.get("qc_valid_coupling", True) else "✘"

        if seg_name.startswith("BH"):
            bad_pct = np.nan
            mean_rr = np.nan
            pct_metrics = np.nan
            pct_coupling = np.nan
            qc_metrics = "nan"
            qc_coupling = "nan"

        print(f"{seg_name:<12} {dur:>5.0f}s {bad_pct:>5.1f}% {mean_rr:>6.1f} {n_breaths:>7} {pct_metrics:>5.1f}% {pct_coupling:>5.1f}% {qc_metrics:>5} {qc_coupling:>5}")
        
        bh_end_time = seg_dict.get("bh_end_time_s", None)
        is_trimmed = seg_dict.get("is_trimmed", False)
        dur_full = seg_dict.get("dur_s_full", dur) if is_trimmed else dur
        
        resumen_filas.append({
            "sujeto": sujeto_id, "segmento": seg_name,
            "duracion_s": round(dur, 1),
            "duracion_full_s": round(dur_full, 1) if is_trimmed else "",
            "bh_end_time_s": round(bh_end_time, 1) if bh_end_time is not None and not np.isnan(bh_end_time) else "",
            "bad_pct": round(bad_pct, 2),
            "mean_resp_rate_bpm": round(mean_rr, 2) if not np.isnan(mean_rr) else "",
            "median_resp_rate_bpm": round(seg_dict.get("median_resp_rate_bpm", 0), 2) if not np.isnan(seg_dict.get("median_resp_rate_bpm", np.nan)) else "",
            "std_resp_rate_bpm": round(seg_dict.get("std_resp_rate_bpm", 0), 2),
            "mean_ie_ratio": round(seg_dict.get("mean_ie_ratio", 0), 3),
            "n_breaths": n_breaths,
            "n_valid_metrics": seg_dict.get("n_valid_cycles_metrics", n_breaths - 1 if n_breaths > 1 else 0),
            "n_valid_coupling": seg_dict.get("n_valid_cycles_coupling", 0),
            "pct_valid_metrics": round(pct_metrics, 1),
            "pct_valid_coupling": round(pct_coupling, 1),
            "qc_valid_metrics": seg_dict.get("qc_valid_metrics", True),
            "qc_valid_coupling": seg_dict.get("qc_valid_coupling", True),
        })
    
    print("-"*70)
    
    # Show peak/trough changes log
    print("\n" + "="*90)
    print("REGISTRO DE CAMBIOS MANUALES (Picos/Troughs)")
    print("="*90)
    hay_cambios = False
    for seg_name, seg_dict in resultados.items():
        changes = seg_dict.get("peak_trough_changes", {})
        if not changes:
            continue
        
        total_changes = (len(changes.get("peaks_added", [])) + len(changes.get("peaks_removed", [])) + 
                        len(changes.get("troughs_added", [])) + len(changes.get("troughs_removed", [])))
        
        if total_changes > 0:
            hay_cambios = True
            print(f"\n{seg_name}:")
            if changes.get("peaks_added"):
                times = sorted(changes["peaks_added"])
                print(f"  + {len(times)} peaks added: {[f'{t:.2f}s' for t in times]}")
            if changes.get("peaks_removed"):
                times = sorted(changes["peaks_removed"])
                print(f"  - {len(times)} peaks removed: {[f'{t:.2f}s' for t in times]}")
            if changes.get("troughs_added"):
                times = sorted(changes["troughs_added"])
                print(f"  + {len(times)} troughs added: {[f'{t:.2f}s' for t in times]}")
            if changes.get("troughs_removed"):
                times = sorted(changes["troughs_removed"])
                print(f"  - {len(times)} troughs removed: {[f'{t:.2f}s' for t in times]}")
    
    if not hay_cambios:
        print("\n  (No manual peak/trough changes recorded)")
    print("="*90)
    
    # Save cumulative
    fila_sujeto = {"sujeto": sujeto_id}
    for seg_info in resumen_filas:
        seg = seg_info["segmento"].replace(" ", "")
        fila_sujeto[f"{seg}_duracion_s"] = seg_info["duracion_s"]
        if seg_info.get("duracion_full_s", "") != "":  # Add full duration for trimmed versions
            fila_sujeto[f"{seg}_duracion_full_s"] = seg_info["duracion_full_s"]
        if seg_info.get("bh_end_time_s", "") != "":  # Only add if BH end time exists
            fila_sujeto[f"{seg}_bh_end_time_s"] = seg_info["bh_end_time_s"]
        fila_sujeto[f"{seg}_bad_pct"] = seg_info["bad_pct"]
        fila_sujeto[f"{seg}_mean_resp_rate_bpm"] = seg_info["mean_resp_rate_bpm"]
        fila_sujeto[f"{seg}_n_breaths"] = seg_info["n_breaths"]
        
        # Add peak/trough changes summary
        seg_name = seg_info["segmento"]
        if seg_name in resultados:
            changes = resultados[seg_name].get("peak_trough_changes", {})
            if changes:
                fila_sujeto[f"{seg}_peaks_added"] = len(changes.get("peaks_added", []))
                fila_sujeto[f"{seg}_peaks_removed"] = len(changes.get("peaks_removed", []))
                fila_sujeto[f"{seg}_troughs_added"] = len(changes.get("troughs_added", []))
                fila_sujeto[f"{seg}_troughs_removed"] = len(changes.get("troughs_removed", []))
                total_changes = (fila_sujeto[f"{seg}_peaks_added"] + fila_sujeto[f"{seg}_peaks_removed"] + 
                               fila_sujeto[f"{seg}_troughs_added"] + fila_sujeto[f"{seg}_troughs_removed"])
                fila_sujeto[f"{seg}_total_changes"] = total_changes
            else:
                fila_sujeto[f"{seg}_peaks_added"] = 0
                fila_sujeto[f"{seg}_peaks_removed"] = 0
                fila_sujeto[f"{seg}_troughs_added"] = 0
                fila_sujeto[f"{seg}_troughs_removed"] = 0
                fila_sujeto[f"{seg}_total_changes"] = 0
    
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
    print(f"✔ Summary: {archivo_resumen}")
    
    return output_file

# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
