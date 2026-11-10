# %% Alcohol / stress study — per-subject EDA cleaning (alcohol)
"""
Run cell by cell (VS Code / Cursor "# %%" cells). Workflow per segment:
  1. inspect (60-s windows),  2. OPTIONAL manual edits (commented examples:
  uncomment and set the times for THIS subject),  3. save the segment.
Finally interpolate, check outliers and export. Paths and segments: config.py.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import config as cfg
from biopac_physio.eda.preprocess import *  # noqa: F401,F403

# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────


# %% ================== SUBJECT CONFIGURATION ==================

SUJETO = "1"    # Subject number (integer as string)
GRUPO  = "E"    # "E" = Experimental, "C" = Control

ACQ_FILE = str(cfg.acq_path(SUJETO, GRUPO))
EDA_COL = "EDA2306000400"  # Keep same channel ID

SEGMENTOS_INTERES = cfg.SEGMENTS

DURACIONES_OBJETIVO = cfg.TARGET_DURATIONS
CARPETA_SALIDA_EDA = cfg.OUT['eda']
ARCHIVO_RESUMEN_EDA = cfg.OUT['eda'] / 'RESUMEN_TODOS_SUJETOS_EDA.csv'

resultados_segmentos_eda: Dict[str, dict] = {}
eda_raw, fs_eda, eventos, eventos_full, data_events, segmentos_dict = cargar_datos_y_segmentos(
    ACQ_FILE, EDA_COL, SEGMENTOS_INTERES, DURACIONES_OBJETIVO,
)

# Recovery: 5-minute window ending at the "Post 10min" marker
if "Post 10min" in eventos:
    fin_rec_idx  = eventos["Post 10min"]
    ini_rec_idx  = max(0, fin_rec_idx - int(300 * fs_eda))
    segmentos_dict["Recovery"] = (ini_rec_idx, fin_rec_idx)
    print("  Recovery: 5 min window ending at 'Post 10min' marker.")
else:
    print("  WARNING: 'Post 10min' marker not found — Recovery segment skipped.")

inspeccionar_duraciones_segmentos(segmentos_dict, SEGMENTOS_INTERES, fs_eda, SUJETO)

# =============================================================================
# LB (BASELINE) - Inspection and cleaning
# =============================================================================
# %% ========== LB - Inspection and cleaning ==========
seg_name = "LB"
resultado_LB = inspeccionar_segmento_eda(
    eda_raw, fs_eda, eventos, seg_name=seg_name, win_len=None,
    decomposition_method="cvxeda", segmentos_dict=segmentos_dict
)

# %% Optional: LB - Inspect specific local windows (with detailed SCR features)
# inspeccionar_ventana_local(resultado_LB, start_s=0.0, end_s=100.0)

# %% Optional: LB - Mark bad zones as NaN
# resultado_LB = marcar_bad_interval(resultado_LB, t_ini=225.0, t_fin=230.0, motivo="artifact")
# resultado_LB = marcar_bad_zones_como_nan(resultado_LB)
# reinspeccionar_segmento(resultado_LB, win_len=None)

# %% ========== LB - Save ==========
resultados_segmentos_eda["LB"] = resultado_LB
print(f"✔ LB saved")


# =============================================================================
# INTERO 1-6
# =============================================================================

# %% ========== INTERO 1 ==========
seg_name = "Intero 1"
resultado_intero1 = inspeccionar_segmento_eda(
    eda_raw, fs_eda, eventos, seg_name=seg_name, win_len=None,
    decomposition_method="cvxeda", segmentos_dict=segmentos_dict
)

# ========== Intero 1 - Save ==========
resultados_segmentos_eda["Intero 1"] = resultado_intero1
print(f"✔ Intero 1 saved")
# %% Opcional: no guardar este segmento (señal mala)
# desguardar_segmento("Intero 1", resultados_segmentos_eda)


# %% ========== INTERO 2 ==========
seg_name = "Intero 2"
resultado_intero2 = inspeccionar_segmento_eda(
    eda_raw, fs_eda, eventos, seg_name=seg_name, win_len=None,
    decomposition_method="cvxeda", segmentos_dict=segmentos_dict
)
#  ========== Intero 2 - Save ==========
resultados_segmentos_eda["Intero 2"] = resultado_intero2
print(f"✔ Intero 2 saved")
# %% Opcional: no guardar este segmento (señal mala)
# desguardar_segmento("Intero 2", resultados_segmentos_eda)


# %% ========== INTERO 3 ==========
seg_name = "Intero 3"
resultado_intero3 = inspeccionar_segmento_eda(
    eda_raw, fs_eda, eventos, seg_name=seg_name, win_len=None,
    decomposition_method="cvxeda", segmentos_dict=segmentos_dict
)
#  ========== Intero 3 - Save ==========
resultados_segmentos_eda["Intero 3"] = resultado_intero3
print(f"✔ Intero 3 saved")
# %% Opcional: no guardar este segmento (señal mala)
# desguardar_segmento("Intero 3", resultados_segmentos_eda)


# %% ========== INTERO 4 ==========
seg_name = "Intero 4"
resultado_intero4 = inspeccionar_segmento_eda(
    eda_raw, fs_eda, eventos, seg_name=seg_name, win_len=None,
    decomposition_method="cvxeda", segmentos_dict=segmentos_dict
) 
#  ========== Intero 4 - Save ==========
resultados_segmentos_eda["Intero 4"] = resultado_intero4
print(f"✔ Intero 4 saved")
# %% Opcional: no guardar este segmento (señal mala)
# desguardar_segmento("Intero 4", resultados_segmentos_eda)


# %% ========== INTERO 5 ==========
seg_name = "Intero 5"
resultado_intero5 = inspeccionar_segmento_eda(
    eda_raw, fs_eda, eventos, seg_name=seg_name, win_len=None,
    decomposition_method="cvxeda", segmentos_dict=segmentos_dict
)
# ========== Intero 5 - Save ==========
resultados_segmentos_eda["Intero 5"] = resultado_intero5
print(f"✔ Intero 5 saved")
# %% Opcional: no guardar este segmento (señal mala)
# desguardar_segmento("Intero 5", resultados_segmentos_eda)


# %% ========== INTERO 6 ==========
seg_name = "Intero 6"
resultado_intero6 = inspeccionar_segmento_eda(
    eda_raw, fs_eda, eventos, seg_name=seg_name, win_len=None,
    decomposition_method="cvxeda", segmentos_dict=segmentos_dict
)
# ========== Intero 6 - Save ==========
resultados_segmentos_eda["Intero 6"] = resultado_intero6
print(f"✔ Intero 6 saved")
# %% Opcional: no guardar este segmento (señal mala)
# desguardar_segmento("Intero 6", resultados_segmentos_eda)


# =============================================================================
# IGT1 (Iowa Gambling Task 1 - variable duration)
# =============================================================================
# %% ========== IGT1 ==========
seg_name = "IGT1"
resultado_igt1 = inspeccionar_segmento_eda(
    eda_raw, fs_eda, eventos, seg_name=seg_name, win_len=None,
    decomposition_method="cvxeda", segmentos_dict=segmentos_dict
)
# %% Optional: Inspect specific local windows (with detailed SCR features)
# inspeccionar_ventana_local(resultado_igt1, start_s=0.0, end_s=60.0)
# %% Optional: Mark bad zones as NaN
# resultado_igt1 = marcar_bad_interval(resultado_igt1, t_ini=0.0, t_fin=5.0, motivo="artifact")
# resultado_igt1 = marcar_bad_zones_como_nan(resultado_igt1)
# reinspeccionar_segmento(resultado_igt1, win_len=None)
# %% ========== IGT1 - Save ==========
resultados_segmentos_eda["IGT1"] = resultado_igt1
print(f"✔ IGT1 saved")
# %% Optional: do not save this segment (bad signal)
# desguardar_segmento("IGT1", resultados_segmentos_eda)


# =============================================================================
# PD (variable duration)
# =============================================================================
# %% ========== PD ==========
seg_name = "PD"
resultado_pd = inspeccionar_segmento_eda(
    eda_raw, fs_eda, eventos, seg_name=seg_name, win_len=None,
    decomposition_method="cvxeda", segmentos_dict=segmentos_dict
)
# %% Optional: Inspect specific local windows (with detailed SCR features)
# inspeccionar_ventana_local(resultado_pd, start_s=0.0, end_s=60.0)
# %% Optional: Mark bad zones as NaN
# resultado_pd = marcar_bad_interval(resultado_pd, t_ini=0.0, t_fin=5.0, motivo="artifact")
# resultado_pd = marcar_bad_zones_como_nan(resultado_pd)
# reinspeccionar_segmento(resultado_pd, win_len=None)
# %% ========== PD - Save ==========
resultados_segmentos_eda["PD"] = resultado_pd
print(f"✔ PD saved")
# %% Optional: do not save this segment (bad signal)
# desguardar_segmento("PD", resultados_segmentos_eda)


# =============================================================================
# Speech (variable duration)
# =============================================================================
# %% ========== Speech ==========
seg_name = "Speech"
resultado_speech = inspeccionar_segmento_eda(
    eda_raw, fs_eda, eventos, seg_name=seg_name, win_len=None,
    decomposition_method="cvxeda", segmentos_dict=segmentos_dict
)
# %% Optional: Inspect specific local windows (with detailed SCR features)
# inspeccionar_ventana_local(resultado_speech, start_s=0.0, end_s=60.0)
# %% Optional: Mark bad zones as NaN
# resultado_speech = marcar_bad_interval(resultado_speech, t_ini=0.0, t_fin=5.0, motivo="artifact")
# resultado_speech = marcar_bad_zones_como_nan(resultado_speech)
# reinspeccionar_segmento(resultado_speech, win_len=None)
# %% ========== Speech - Save ==========
resultados_segmentos_eda["Speech"] = resultado_speech
print(f"✔ Speech saved")
# %% Optional: do not save this segment (bad signal)
# desguardar_segmento("Speech", resultados_segmentos_eda)


# =============================================================================
# Arit (Arithmetic task - variable duration)
# =============================================================================
# %% ========== Arit ==========
seg_name = "Arit"
resultado_arit = inspeccionar_segmento_eda(
    eda_raw, fs_eda, eventos, seg_name=seg_name, win_len=None,
    decomposition_method="cvxeda", segmentos_dict=segmentos_dict
)
# %% Optional: Inspect specific local windows (with detailed SCR features)
# inspeccionar_ventana_local(resultado_arit, start_s=0.0, end_s=60.0)
# %% Optional: Mark bad zones as NaN
# resultado_arit = marcar_bad_interval(resultado_arit, t_ini=0.0, t_fin=5.0, motivo="artifact")
# resultado_arit = marcar_bad_zones_como_nan(resultado_arit)
# reinspeccionar_segmento(resultado_arit, win_len=None)
# %% ========== Arit - Save ==========
resultados_segmentos_eda["Arit"] = resultado_arit
print(f"✔ Arit saved")
# %% Optional: do not save this segment (bad signal)
# desguardar_segmento("Arit", resultados_segmentos_eda)


# =============================================================================
# IGT2 (Iowa Gambling Task 2 - variable duration)
# =============================================================================
# %% ========== IGT2 ==========
seg_name = "IGT2"
resultado_igt2 = inspeccionar_segmento_eda(
    eda_raw, fs_eda, eventos, seg_name=seg_name, win_len=None,
    decomposition_method="cvxeda", segmentos_dict=segmentos_dict
)
# %% Optional: Inspect specific local windows (with detailed SCR features)
# inspeccionar_ventana_local(resultado_igt2, start_s=0.0, end_s=60.0)
# %% Optional: Mark bad zones as NaN
# resultado_igt2 = marcar_bad_interval(resultado_igt2, t_ini=0.0, t_fin=5.0, motivo="artifact")
# resultado_igt2 = marcar_bad_zones_como_nan(resultado_igt2)
# reinspeccionar_segmento(resultado_igt2, win_len=None)
# %% ========== IGT2 - Save ==========
resultados_segmentos_eda["IGT2"] = resultado_igt2
print(f"✔ IGT2 saved")
# %% Optional: do not save this segment (bad signal)
# desguardar_segmento("IGT2", resultados_segmentos_eda)


# =============================================================================
# RECOVERY - 5-minute window before "Post 10min" marker
# =============================================================================
# %% ========== RECOVERY - Inspection and cleaning ==========
seg_name = "Recovery"
resultado_recovery = inspeccionar_segmento_eda(
    eda_raw, fs_eda, eventos, seg_name=seg_name, win_len=None,
    decomposition_method="cvxeda", segmentos_dict=segmentos_dict
)
# %% Optional: RECOVERY - Inspect specific local windows (with detailed SCR features)
# inspeccionar_ventana_local(resultado_recovery, start_s=150.0, end_s=175.0)
# %% Optional: RECOVERY - Mark bad zones as NaN
# resultado_recovery = marcar_bad_interval(resultado_recovery, t_ini=0.0, t_fin=33.0, motivo="artifact")
# resultado_recovery = marcar_bad_zones_como_nan(resultado_recovery)
# reinspeccionar_segmento(resultado_recovery, win_len=None)

# %% ========== RECOVERY - Save ==========
resultados_segmentos_eda["Recovery"] = resultado_recovery
print(f"✔ Recovery saved")


# =============================================================================
# FINAL: PRE-EXPORT SUMMARY
# =============================================================================

# %% ========== STEP 1: Summary ==========
print("\n" + "=" * 60)
print("SEGMENTS READY FOR EXPORT")
print("=" * 60)
for seg, res in resultados_segmentos_eda.items():
    bad = res.get('bad_pct', 0) * 100
    scl = res.get('scl_mean', 0)
    scrs = len(res.get('scr_peaks', []))
    print(f"  {seg}: BAD={bad:.1f}%, SCL={scl:.3f} µS, SCRs={scrs}")

# %% ========== STEP 2: Check outliers ==========
# If issues found, go back and clean the affected segment
resultados_segmentos_eda, hay_outliers_eda = verificar_outliers_todos(resultados_segmentos_eda)
if hay_outliers_eda:
    print("⚠️  Outliers detected – review flagged segments before exporting.")

# %% ========== STEP 3: Export ==========
# Only run if STEP 2 is OK (no outliers)
output_file = exportar_y_resumir_eda(
    resultados=resultados_segmentos_eda,
    sujeto_id=SUJETO,
    output_folder=CARPETA_SALIDA_EDA,
    archivo_resumen=ARCHIVO_RESUMEN_EDA
)


# %%

