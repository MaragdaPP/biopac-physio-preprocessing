# %% Alcohol / stress study — per-subject RSP cleaning (alcohol)
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
from biopac_physio.rsp.preprocess import *  # noqa: F401,F403

# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────

# %% ================== SUBJECT CONFIGURATION ==================

SUJETO = "1"    # Subject number (integer as string)
GRUPO  = "E"    # "E" = Experimental, "C" = Control

ACQ_FILE = str(cfg.acq_path(SUJETO, GRUPO))
RSP_COL = "RSP1906000108"  # Respiration channel

SEGMENTOS_INTERES = cfg.SEGMENTS

DURACIONES_OBJETIVO = cfg.TARGET_DURATIONS
CARPETA_SALIDA_RSP = cfg.OUT['rsp']
ARCHIVO_RESUMEN_RSP = cfg.OUT['rsp'] / 'RESUMEN_TODOS_SUJETOS_RSP.csv'

resultados_segmentos_rsp: Dict[str, dict] = {}
rsp_raw, fs_rsp, eventos, eventos_full, data_events, segmentos_dict = cargar_datos_y_segmentos(
    ACQ_FILE, RSP_COL, SEGMENTOS_INTERES, DURACIONES_OBJETIVO,
    segmentos_manuales=cfg.MANUAL_SEGMENTS,
)

# Recovery: 5-minute window ending at the "Post 10min" marker
if "Post 10min" in eventos:
    fin_rec_idx  = eventos["Post 10min"]
    ini_rec_idx  = max(0, fin_rec_idx - int(300 * fs_rsp))
    segmentos_dict["Recovery"] = (ini_rec_idx, fin_rec_idx)
    print("  Recovery: 5 min window ending at 'Post 10min' marker.")
else:
    print("  WARNING: 'Post 10min' marker not found — Recovery segment skipped.")

inspeccionar_duraciones_segmentos(segmentos_dict, SEGMENTOS_INTERES, fs_rsp, SUJETO)

# =============================================================================
# LB (BASELINE) - ~300s
# =============================================================================
# %% ========== LB - STEP 1: Visual inspection (FULL 300s) ==========
seg_name = "LB"
resultado_LB = inspeccionar_segmento_rsp(
    rsp_raw, fs_rsp, eventos, seg_name=seg_name,
    win_len=None, segmentos_dict=segmentos_dict,
    mostrar_info_segmentos_dict=False
)
_checkpoint_LB = copy.deepcopy(resultado_LB)
ver_señal_completa(resultado_LB, show_bad=False)

# %% ========== LB - STEP 2: Manual peak editing ==========
# resultado_LB = editar_ciclos_interactivos_externo(resultado_LB, enable=True, 
# t0=100.0, win_len=60.0)
# %% Volver a inspeccionar la señal
ver_señal_completa(resultado_LB, show_bad=False)

# %% ========== LB - STEP 3: Convert to NaN/interpolate ==========
# resultado_LB = convertir_cambios_a_bad_zones(resultado_LB)
# resultado_LB = limpiar(resultado_LB, aplicar_interpolacion=True, max_cycles_interp=2, max_duration_interp_s=6.0)

# %% ========== LB - STEP 4: Save (trims to 300s) ==========
resultado_LB, resultados_segmentos_rsp = guardar_segmento(
    resultado_LB, resultados_segmentos_rsp, "LB", target_duration_s=300.0)

# Note: use resultado_LB["rsp_masked"] for RSA/coupling (NaN in BAD zones) and
#       resultado_LB["rsp_interp"] for metrics (short zones interpolated)


# =============================================================================
# INTERO 1-6
# =============================================================================
# %% ========== INTERO 1 - Inspection ==========
seg_name = "Intero 1"
resultado_intero1 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                              win_len=None, segmentos_dict=segmentos_dict,
                                              mostrar_info_segmentos_dict=False)
_checkpoint_intero1 = copy.deepcopy(resultado_intero1)
ver_señal_completa(resultado_intero1, show_bad=False)

# %% ========== Manual peak editing ==========
# resultado_intero1 = editar_ciclos_interactivos_externo(resultado_intero1, enable=True, t0=245.0, win_len=30.0)
# ver_señal_completa(resultado_intero1, show_bad=False)

# %% ========== Convert to NaN/interpolate ==========
# resultado_intero1 = convertir_cambios_a_bad_zones(resultado_intero1)
# resultado_intero1 = limpiar(resultado_intero1, aplicar_interpolacion=True, max_cycles_interp=2, max_duration_interp_s=6.0)
# %% Save
resultado_intero1, resultados_segmentos_rsp = guardar_segmento(resultado_intero1, resultados_segmentos_rsp, "Intero 1")


# %% ========== INTERO 2 ==========
seg_name = "Intero 2"
resultado_intero2 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                              win_len=None, segmentos_dict=segmentos_dict,
                                              mostrar_info_segmentos_dict=False)
_checkpoint_intero2 = copy.deepcopy(resultado_intero2)
ver_señal_completa(resultado_intero2, show_bad=False)
# %% ========== Manual peak editing ==========
# resultado_intero2 = editar_ciclos_interactivos_externo(resultado_intero2, enable=True, t0=5.0, win_len=30.0)
# ver_señal_completa(resultado_intero2, show_bad=False)

# %% ========== Convert to NaN/interpolate ==========
# resultado_intero2 = convertir_cambios_a_bad_zones(resultado_intero2)
# resultado_intero2 = limpiar(resultado_intero2, aplicar_interpolacion=True, max_cycles_interp=2, max_duration_interp_s=6.0)
# %% Save
resultado_intero2, resultados_segmentos_rsp = guardar_segmento(resultado_intero2, resultados_segmentos_rsp, "Intero 2")


# %% ========== INTERO 3 ==========
seg_name = "Intero 3"
resultado_intero3 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                              win_len=None, segmentos_dict=segmentos_dict,
                                              mostrar_info_segmentos_dict=False)
_checkpoint_intero3 = copy.deepcopy(resultado_intero3)
ver_señal_completa(resultado_intero3, show_bad=False)
# %% ========== Manual peak editing ==========
# resultado_intero3 = editar_ciclos_interactivos_externo(resultado_intero3, enable=True, t0=20.0, win_len=60.0)
# ver_señal_completa(resultado_intero3, show_bad=False)

# %% ========== Convert to NaN/interpolate ==========
# resultado_intero3 = convertir_cambios_a_bad_zones(resultado_intero3)
# resultado_intero3 = limpiar(resultado_intero3, aplicar_interpolacion=True, max_cycles_interp=2, max_duration_interp_s=6.0)
# %% Save
resultado_intero3, resultados_segmentos_rsp = guardar_segmento(resultado_intero3, resultados_segmentos_rsp, "Intero 3")


# %% ========== INTERO 4 ==========
seg_name = "Intero 4"
resultado_intero4 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                              win_len=None, segmentos_dict=segmentos_dict,
                                              mostrar_info_segmentos_dict=False)
_checkpoint_intero4 = copy.deepcopy(resultado_intero4)
ver_señal_completa(resultado_intero4, show_bad=False)
# %% ========== Manual peak editing ==========
# resultado_intero4 = editar_ciclos_interactivos_externo(resultado_intero4, enable=True, t0=0.0, win_len=60.0)
# ver_señal_completa(resultado_intero4, show_bad=False)

# %% ========== Convert to NaN/interpolate ==========
# resultado_intero4 = convertir_cambios_a_bad_zones(resultado_intero4)
# resultado_intero4 = limpiar(resultado_intero4, aplicar_interpolacion=True, max_cycles_interp=2, max_duration_interp_s=6.0)
# %% Save
resultado_intero4, resultados_segmentos_rsp = guardar_segmento(resultado_intero4, resultados_segmentos_rsp, "Intero 4")


# %% ========== INTERO 5 ==========
seg_name = "Intero 5"
resultado_intero5 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                              win_len=None, segmentos_dict=segmentos_dict,
                                              mostrar_info_segmentos_dict=False)
_checkpoint_intero5 = copy.deepcopy(resultado_intero5)
ver_señal_completa(resultado_intero5, show_bad=False)
# %% ========== Manual peak editing ==========
# resultado_intero5 = editar_ciclos_interactivos_externo(resultado_intero5, enable=True, t0=15.0, win_len=60.0)
# ver_señal_completa(resultado_intero5, show_bad=False)

# %% ========== Convert to NaN/interpolate ==========
# resultado_intero5 = convertir_cambios_a_bad_zones(resultado_intero5)
# resultado_intero5 = limpiar(resultado_intero5, aplicar_interpolacion=True, max_cycles_interp=2, max_duration_interp_s=6.0)
# %% Save
resultado_intero5, resultados_segmentos_rsp = guardar_segmento(resultado_intero5, resultados_segmentos_rsp, "Intero 5")


# %% ========== INTERO 6 ==========
seg_name = "Intero 6"
resultado_intero6 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                              win_len=None, segmentos_dict=segmentos_dict,
                                              mostrar_info_segmentos_dict=False)
_checkpoint_intero6 = copy.deepcopy(resultado_intero6)
ver_señal_completa(resultado_intero6, show_bad=False)
# %% ========== Manual peak editing ==========
# resultado_intero6 = editar_ciclos_interactivos_externo(resultado_intero6, enable=True, t0=20.0, win_len=60.0)
# ver_señal_completa(resultado_intero6, show_bad=False)
# %% Save
resultado_intero6, resultados_segmentos_rsp = guardar_segmento(resultado_intero6, resultados_segmentos_rsp, "Intero 6")


# =============================================================================
# IGT1 (Iowa Gambling Task 1 - variable duration)
# =============================================================================
# %% ========== IGT1 - Inspection ==========
seg_name = "IGT1"
resultado_igt1 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                           win_len=None, segmentos_dict=segmentos_dict,
                                           mostrar_info_segmentos_dict=False)
_checkpoint_igt1 = copy.deepcopy(resultado_igt1)
ver_señal_completa(resultado_igt1, show_bad=False)

# %% ========== Manual peak editing ==========
# resultado_igt1 = editar_ciclos_interactivos_externo(resultado_igt1, enable=True, t0=0.0, win_len=60.0)
# ver_señal_completa(resultado_igt1, show_bad=False)

# %% ========== Convert to NaN/interpolate ==========
# resultado_igt1 = convertir_cambios_a_bad_zones(resultado_igt1)
# resultado_igt1 = limpiar(resultado_igt1, aplicar_interpolacion=True, max_cycles_interp=2, max_duration_interp_s=6.0)
# %% Save
resultado_igt1, resultados_segmentos_rsp = guardar_segmento(resultado_igt1, resultados_segmentos_rsp, "IGT1")


# =============================================================================
# PD (variable duration)
# =============================================================================
# %% ========== PD - Inspection ==========
seg_name = "PD"
resultado_pd = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                         win_len=None, segmentos_dict=segmentos_dict,
                                         mostrar_info_segmentos_dict=False)
_checkpoint_pd = copy.deepcopy(resultado_pd)
ver_señal_completa(resultado_pd, show_bad=False)

# %% ========== Manual peak editing ==========
# resultado_pd = editar_ciclos_interactivos_externo(resultado_pd, enable=True, t0=0.0, win_len=60.0)
# ver_señal_completa(resultado_pd, show_bad=False)

# %% ========== Convert to NaN/interpolate ==========
# resultado_pd = convertir_cambios_a_bad_zones(resultado_pd)
# resultado_pd = limpiar(resultado_pd, aplicar_interpolacion=True, max_cycles_interp=2, max_duration_interp_s=6.0)
# %% Save
resultado_pd, resultados_segmentos_rsp = guardar_segmento(resultado_pd, resultados_segmentos_rsp, "PD")


# =============================================================================
# Speech (variable duration)
# =============================================================================
# %% ========== Speech - Inspection ==========
seg_name = "Speech"
resultado_speech = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                             win_len=None, segmentos_dict=segmentos_dict,
                                             mostrar_info_segmentos_dict=False)
_checkpoint_speech = copy.deepcopy(resultado_speech)
ver_señal_completa(resultado_speech, show_bad=False)

# %% ========== Manual peak editing ==========
# resultado_speech = editar_ciclos_interactivos_externo(resultado_speech, enable=True, t0=0.0, win_len=60.0)
# ver_señal_completa(resultado_speech, show_bad=False)

# %% ========== Convert to NaN/interpolate ==========
# resultado_speech = convertir_cambios_a_bad_zones(resultado_speech)
# resultado_speech = limpiar(resultado_speech, aplicar_interpolacion=True, max_cycles_interp=2, max_duration_interp_s=6.0)
# %% Save
resultado_speech, resultados_segmentos_rsp = guardar_segmento(resultado_speech, resultados_segmentos_rsp, "Speech")


# =============================================================================
# Arit (Arithmetic task - variable duration)
# =============================================================================
# %% ========== Arit - Inspection ==========
seg_name = "Arit"
resultado_arit = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                           win_len=None, segmentos_dict=segmentos_dict,
                                           mostrar_info_segmentos_dict=False)
_checkpoint_arit = copy.deepcopy(resultado_arit)
ver_señal_completa(resultado_arit, show_bad=False)

# %% ========== Manual peak editing ==========
# resultado_arit = editar_ciclos_interactivos_externo(resultado_arit, enable=True, t0=0.0, win_len=60.0)
# ver_señal_completa(resultado_arit, show_bad=False)

# %% ========== Convert to NaN/interpolate ==========
# resultado_arit = convertir_cambios_a_bad_zones(resultado_arit)
# resultado_arit = limpiar(resultado_arit, aplicar_interpolacion=True, max_cycles_interp=2, max_duration_interp_s=6.0)
# %% Save
resultado_arit, resultados_segmentos_rsp = guardar_segmento(resultado_arit, resultados_segmentos_rsp, "Arit")


# =============================================================================
# IGT2 (Iowa Gambling Task 2 - variable duration)
# =============================================================================
# %% ========== IGT2 - Inspection ==========
seg_name = "IGT2"
resultado_igt2 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                           win_len=None, segmentos_dict=segmentos_dict,
                                           mostrar_info_segmentos_dict=False)
_checkpoint_igt2 = copy.deepcopy(resultado_igt2)
ver_señal_completa(resultado_igt2, show_bad=False)

# %% ========== Manual peak editing ==========
# resultado_igt2 = editar_ciclos_interactivos_externo(resultado_igt2, enable=True, t0=0.0, win_len=60.0)
# ver_señal_completa(resultado_igt2, show_bad=False)

# %% ========== Convert to NaN/interpolate ==========
# resultado_igt2 = convertir_cambios_a_bad_zones(resultado_igt2)
# resultado_igt2 = limpiar(resultado_igt2, aplicar_interpolacion=True, max_cycles_interp=2, max_duration_interp_s=6.0)
# %% Save
resultado_igt2, resultados_segmentos_rsp = guardar_segmento(resultado_igt2, resultados_segmentos_rsp, "IGT2")


# =============================================================================
# RECOVERY - 5-minute window before "Post 10min" marker (~300s)
# =============================================================================
# %% ========== RECOVERY - STEP 1: Visual inspection (FULL 300s) ==========
seg_name = "Recovery"
resultado_recovery = inspeccionar_segmento_rsp(
    rsp_raw, fs_rsp, eventos, seg_name=seg_name,
    win_len=None, segmentos_dict=segmentos_dict,
    mostrar_info_segmentos_dict=False
)
_checkpoint_recovery = copy.deepcopy(resultado_recovery)
ver_señal_completa(resultado_recovery, show_bad=False)

# %% ========== RECOVERY - STEP 2: Manual peak editing ==========
# resultado_recovery = editar_ciclos_interactivos_externo(resultado_recovery, enable=True, t0=160.0, win_len=60.0)
# ver_señal_completa(resultado_recovery, show_bad=False)

# %% ========== RECOVERY - STEP 3: Convert to NaN/interpolate ==========
# resultado_recovery = convertir_cambios_a_bad_zones(resultado_recovery)
# resultado_recovery = limpiar(resultado_recovery, aplicar_interpolacion=True, max_cycles_interp=2, max_duration_interp_s=6.0)

# %% ========== RECOVERY - STEP 4: Save (trims to 300s) ==========
resultado_recovery, resultados_segmentos_rsp = guardar_segmento(
    resultado_recovery, resultados_segmentos_rsp, "Recovery", target_duration_s=300.0)

# Note: use resultado_recovery["rsp_masked"] for RSA/coupling (NaN in BAD zones) and
#       resultado_recovery["rsp_interp"] for metrics (short zones interpolated)


# =============================================================================
# FINAL: VERIFICATION + EXPORT
# =============================================================================
# %% ========== STEP 1: Check outliers in all segments ==========
# If issues found, go back and clean the affected segment
resultados_segmentos_rsp, hay_problemas = verificar_outliers_todos(resultados_segmentos_rsp)

# %% ========== STEP 2: Summary before export ==========
# Only run if STEP 1 is OK (no outliers)
df_resumen = resumen_antes_exportar(resultados_segmentos_rsp)


# %% ========== STEP 3: Export + Save cumulative ==========
output_file = exportar_y_resumir_rsp(
    resultados=resultados_segmentos_rsp,
    sujeto_id=SUJETO,
    output_folder=CARPETA_SALIDA_RSP,
    archivo_resumen=ARCHIVO_RESUMEN_RSP
)


# %%
