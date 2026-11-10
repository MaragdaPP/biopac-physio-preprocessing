# %% Alcohol / stress study — per-subject ECG cleaning (alcohol)
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
from biopac_physio.ecg.preprocess import *  # noqa: F401,F403

# %% ================== SUBJECT CONFIGURATION ==================

SUJETO = "1"    # Subject number (integer as string)
GRUPO  = "E"    # "E" = Experimental, "C" = Control

ACQ_FILE = str(cfg.acq_path(SUJETO, GRUPO))
ECG_COL  = "ECG2307000448"  # Keep same channel ID

SEGMENTOS_INTERES = cfg.SEGMENTS

DURACIONES_OBJETIVO = cfg.TARGET_DURATIONS
CARPETA_SALIDA_SUJETO = cfg.OUT['ecg']
ARCHIVO_RESUMEN = cfg.OUT['ecg'] / 'RESUMEN_TODOS_SUJETOS.csv'

resultados_segmentos_limpios: Dict[str, dict] = {}
ecg_raw, fs_ecg, eventos, eventos_full, data_events, segmentos_dict = cargar_datos_y_segmentos(
    ACQ_FILE, ECG_COL, SEGMENTOS_INTERES, DURACIONES_OBJETIVO,
    segmentos_manuales=cfg.MANUAL_SEGMENTS,
)

# Recovery: 5-minute window ending at the "Post 10min" marker
if "Post 10min" in eventos:
    fin_rec_idx  = eventos["Post 10min"]
    ini_rec_idx  = max(0, fin_rec_idx - int(300 * fs_ecg))
    segmentos_dict["Recovery"] = (ini_rec_idx, fin_rec_idx)
    print("  Recovery: 5 min window ending at 'Post 10min' marker.")
else:
    print("  WARNING: 'Post 10min' marker not found — Recovery segment skipped.")

inspeccionar_duraciones_segmentos(segmentos_dict, SEGMENTOS_INTERES, fs_ecg, SUJETO)

# =============================================================================
# LB (BASELINE) - Requires 290s of clean data
# =============================================================================
# %% ========== LB - STEP 1: Initial inspection ==========
seg_name = "LB"
resultado_LB = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_LB = copy.deepcopy(resultado_LB)  # Save checkpoint to allow reverting

# %% Optional: Zoom into specific time
# ampliar_senal_en_tiempo(resultado_LB, t_centro=19.0, ventana_s=10.0)

# ========== LB - STEP 2: Cleaning (only if needed) ==========
# %% MANUAL PEAK CORRECTION (interactive editor)
# resultado_LB = editar_peaks_interactivos_externo(resultado_LB, enable=True, t0=240, win_len=10)

# %% AUTOMATIC CLEANING
# resultado_LB = limpiar(resultado_LB, borrar_clusters=True, min_clean_duration_s=290.0)
# reinspeccionar_segmento(resultado_LB, win_len=60.0, min_clean_duration_s=290.0)

# BAD ZONE ADJUSTMENTS
# %% Remove a BAD zone that is not problematic:
# resultado_LB = eliminar_bad_zones(resultado_LB, [1])
# %% Modify BAD zone boundaries:
# resultado_LB = modificar_zona_bad(resultado_LB, zona_id=1, nuevo_t_ini=282.5, nuevo_t_fin=285.0)
# %% Add a new BAD zone:
# resultado_LB = agregar_bad_zone(resultado_LB, t_ini=100.0, t_fin=105.0)
# %% Verify result:
# reinspeccionar_segmento(resultado_LB, win_len=60.0, min_clean_duration_s=290.0)

# %% REVERT CLUSTER DELETIONS
# resultado_LB = revertir_borrado_clusters(resultado_LB)

# %% ========== LB - STEP 3: Save ==========
resultado_LB, _ = guardar_segmento(resultado_LB, resultados_segmentos_limpios, "LB")


# =============================================================================
# INTERO 1-6 (Segmentos cortos ~60s, no requieren mínimo)
# =============================================================================

# %% ========== INTERO 1 - PASO 1: Inspección ==========
seg_name = "Intero 1"
resultado_intero1 = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_intero1 = copy.deepcopy(resultado_intero1)

# %% Opcional: Ampliar señal
# ampliar_senal_en_tiempo(resultado_intero1, t_centro=30, ventana_s=10.0)

# ========== INTERO 1 - PASO 2: Limpieza ==========
# %% CORRECCION MANUAL DE PEAKS (Editor interactivo)
# resultado_intero1 = editar_peaks_interactivos_externo(resultado_intero1, enable=True,
# t0=10, win_len=10)

# %% LIMPIEZA AUTOMÁTICA
# resultado_intero1 = limpiar(resultado_intero1, borrar_clusters=True)
# reinspeccionar_segmento(resultado_intero1, win_len=60.0)

# MODIFICAR ZONAS BAD
# %% Eliminar zona BAD que NO es problemática:
# resultado_intero1 = eliminar_bad_zones(resultado_intero1, [1])
# %% Modificar límites de zona BAD:
# resultado_intero1 = modificar_zona_bad(resultado_intero1, zona_id=2, nuevo_t_ini=10.0, nuevo_t_fin=15.0)
# %% Agregar zona BAD nueva:
# resultado_intero1 = agregar_bad_zone(resultado_intero1, t_ini=20.0, t_fin=22.0)
# %% Verificar resultado:
# reinspeccionar_segmento(resultado_intero1)

# %% REVERTIR BORRADOS PEAKS EN CLUSTERS
# resultado_intero1 = revertir_borrado_clusters(resultado_intero1)

# %% ========== INTERO 1 - PASO 3: Guardar ==========
resultado_intero1, _ = guardar_segmento(resultado_intero1, resultados_segmentos_limpios, "Intero 1")


# %% ========== INTERO 2 - PASO 1: Inspección ==========
seg_name = "Intero 2"
resultado_intero2 = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_intero2 = copy.deepcopy(resultado_intero2)

# %% Opcional: Ampliar señal
# ampliar_senal_en_tiempo(resultado_intero2, t_centro=30, ventana_s=10.0)

# ========== INTERO 2 - PASO 2: Limpieza ==========
# %% CORRECCION MANUAL DE PEAKS (Editor interactivo)
# resultado_intero2 = editar_peaks_interactivos_externo(resultado_intero2, enable=True, t0=30, win_len=10)

# %% LIMPIEZA AUTOMÁTICA
# resultado_intero2 = limpiar(resultado_intero2, borrar_clusters=True)
# reinspeccionar_segmento(resultado_intero2, win_len=60.0)

# MODIFICAR ZONAS BAD
# %% Eliminar zona BAD que NO es problemática:
# resultado_intero2 = eliminar_bad_zones(resultado_intero2, [1])
# %% Modificar límites de zona BAD:
# resultado_intero2 = modificar_zona_bad(resultado_intero2, zona_id=2, nuevo_t_ini=10.0, nuevo_t_fin=15.0)
# %% Agregar zona BAD nueva:
# resultado_intero2 = agregar_bad_zone(resultado_intero2, t_ini=20.0, t_fin=22.0)
# %% Verificar resultado:
# reinspeccionar_segmento(resultado_intero2)

# %% REVERTIR BORRADOS PEAKS EN CLUSTERS
# resultado_intero2 = revertir_borrado_clusters(resultado_intero2)

# %% ========== INTERO 2 - PASO 3: Guardar ==========
resultado_intero2, _ = guardar_segmento(resultado_intero2, resultados_segmentos_limpios, "Intero 2")


# %% ========== INTERO 3 - PASO 1: Inspección ==========
seg_name = "Intero 3"
resultado_intero3 = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_intero3 = copy.deepcopy(resultado_intero3)

# %% Opcional: Ampliar señal
# ampliar_senal_en_tiempo(resultado_intero3, t_centro=30, ventana_s=10.0)

# ========== INTERO 3 - PASO 2: Limpieza ==========
# %% CORRECCION MANUAL DE PEAKS (Editor interactivo)
# resultado_intero3 = editar_peaks_interactivos_externo(resultado_intero3, enable=True, t0=30, win_len=10)

# %% LIMPIEZA AUTOMÁTICA
# resultado_intero3 = limpiar(resultado_intero3, borrar_clusters=True)
# reinspeccionar_segmento(resultado_intero3, win_len=60.0)

# MODIFICAR ZONAS BAD
# %% Eliminar zona BAD que NO es problemática:
# resultado_intero3 = eliminar_bad_zones(resultado_intero3, [1])
# %% Modificar límites de zona BAD:
# resultado_intero3 = modificar_zona_bad(resultado_intero3, zona_id=2, nuevo_t_ini=10.0, nuevo_t_fin=15.0)
# %% Agregar zona BAD nueva:
# resultado_intero3 = agregar_bad_zone(resultado_intero3, t_ini=20.0, t_fin=22.0)
# %% Verificar resultado:
# reinspeccionar_segmento(resultado_intero3)

# %% REVERTIR BORRADOS PEAKS EN CLUSTERS
# resultado_intero3 = revertir_borrado_clusters(resultado_intero3)

# %% ========== INTERO 3 - PASO 3: Guardar ==========
resultado_intero3, _ = guardar_segmento(resultado_intero3, resultados_segmentos_limpios, "Intero 3")


# %% ========== INTERO 4 - PASO 1: Inspección ==========
seg_name = "Intero 4"
resultado_intero4 = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_intero4 = copy.deepcopy(resultado_intero4)

# %% Opcional: Ampliar señal
# ampliar_senal_en_tiempo(resultado_intero4, t_centro=30, ventana_s=10.0)

# ========== INTERO 4 - PASO 2: Limpieza ==========
# %% CORRECCION MANUAL DE PEAKS (Editor interactivo)
# resultado_intero4 = editar_peaks_interactivos_externo(resultado_intero4, enable=True, t0=30, win_len=10)

# %% LIMPIEZA AUTOMÁTICA
# resultado_intero4 = limpiar(resultado_intero4, borrar_clusters=True)
# reinspeccionar_segmento(resultado_intero4, win_len=60.0)

# MODIFICAR ZONAS BAD
# %% Eliminar zona BAD que NO es problemática:
# resultado_intero4 = eliminar_bad_zones(resultado_intero4, [1])
# %% Modificar límites de zona BAD:
# resultado_intero4 = modificar_zona_bad(resultado_intero4, zona_id=2, nuevo_t_ini=10.0, nuevo_t_fin=15.0)
# %% Agregar zona BAD nueva:
# resultado_intero4 = agregar_bad_zone(resultado_intero4, t_ini=20.0, t_fin=22.0)
# %% Verificar resultado:
# reinspeccionar_segmento(resultado_intero4)

# %% REVERTIR BORRADOS PEAKS EN CLUSTERS
# resultado_intero4 = revertir_borrado_clusters(resultado_intero4)

# %% ========== INTERO 4 - PASO 3: Guardar ==========
resultado_intero4, _ = guardar_segmento(resultado_intero4, resultados_segmentos_limpios, "Intero 4")


# %% ========== INTERO 5 - PASO 1: Inspección ==========
seg_name = "Intero 5"
resultado_intero5 = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_intero5 = copy.deepcopy(resultado_intero5)

# %% Opcional: Ampliar señal
# ampliar_senal_en_tiempo(resultado_intero5, t_centro=30, ventana_s=10.0)

# ========== INTERO 5 - PASO 2: Limpieza ==========
# %% CORRECCION MANUAL DE PEAKS (Editor interactivo)
# resultado_intero5 = editar_peaks_interactivos_externo(resultado_intero5, enable=True, t0=30, win_len=10)

# %% LIMPIEZA AUTOMÁTICA
# resultado_intero5 = limpiar(resultado_intero5, borrar_clusters=True)
# reinspeccionar_segmento(resultado_intero5, win_len=60.0)

# MODIFICAR ZONAS BAD
# %% Eliminar zona BAD que NO es problemática:
# resultado_intero5 = eliminar_bad_zones(resultado_intero5, [1])
# %% Modificar límites de zona BAD:
# resultado_intero5 = modificar_zona_bad(resultado_intero5, zona_id=2, nuevo_t_ini=10.0, nuevo_t_fin=15.0)
# %% Agregar zona BAD nueva:
# resultado_intero5 = agregar_bad_zone(resultado_intero5, t_ini=20.0, t_fin=22.0)
# %% Verificar resultado:
# reinspeccionar_segmento(resultado_intero5)

# %% REVERTIR BORRADOS PEAKS EN CLUSTERS
# resultado_intero5 = revertir_borrado_clusters(resultado_intero5)

# %% ========== INTERO 5 - PASO 3: Guardar ==========
resultado_intero5, _ = guardar_segmento(resultado_intero5, resultados_segmentos_limpios, "Intero 5")


# %% ========== INTERO 6 - PASO 1: Inspección ==========
seg_name = "Intero 6"
resultado_intero6 = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_intero6 = copy.deepcopy(resultado_intero6)

# %% Opcional: Ampliar señal
# ampliar_senal_en_tiempo(resultado_intero6, t_centro=30, ventana_s=10.0)

# ========== INTERO 6 - PASO 2: Limpieza ==========
# %% CORRECCION MANUAL DE PEAKS (Editor interactivo)
# resultado_intero6 = editar_peaks_interactivos_externo(resultado_intero6, enable=True, t0=30, win_len=10)

# %% LIMPIEZA AUTOMÁTICA
# resultado_intero6 = limpiar(resultado_intero6, borrar_clusters=True)
# reinspeccionar_segmento(resultado_intero6, win_len=60.0)

# MODIFICAR ZONAS BAD
# %% Eliminar zona BAD que NO es problemática:
# resultado_intero6 = eliminar_bad_zones(resultado_intero6, [1])
# %% Modificar límites de zona BAD:
# resultado_intero6 = modificar_zona_bad(resultado_intero6, zona_id=2, nuevo_t_ini=10.0, nuevo_t_fin=15.0)
# %% Agregar zona BAD nueva:
# resultado_intero6 = agregar_bad_zone(resultado_intero6, t_ini=20.0, t_fin=22.0)
# %% Verificar resultado:
# reinspeccionar_segmento(resultado_intero6)

# %% REVERTIR BORRADOS PEAKS EN CLUSTERS
# resultado_intero6 = revertir_borrado_clusters(resultado_intero6)

# %% ========== INTERO 6 - PASO 3: Guardar ==========
resultado_intero6, _ = guardar_segmento(resultado_intero6, resultados_segmentos_limpios, "Intero 6")


# =============================================================================
# IGT1 (Iowa Gambling Task 1 - variable duration)
# =============================================================================
# %% ========== IGT1 - STEP 1: Inspection ==========
seg_name = "IGT1"
resultado_igt1 = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_igt1 = copy.deepcopy(resultado_igt1)

# %% Optional: Zoom into specific time
# ampliar_senal_en_tiempo(resultado_igt1, t_centro=30, ventana_s=10.0)

# ========== IGT1 - STEP 2: Cleaning (only if needed) ==========
# %% MANUAL PEAK CORRECTION (interactive editor)
# resultado_igt1 = editar_peaks_interactivos_externo(resultado_igt1, enable=True, t0=30, win_len=10)

# %% AUTOMATIC CLEANING
# resultado_igt1 = limpiar(resultado_igt1, borrar_clusters=True)
# reinspeccionar_segmento(resultado_igt1, win_len=60.0)

# BAD ZONE ADJUSTMENTS
# %% Remove a BAD zone that is not problematic:
# resultado_igt1 = eliminar_bad_zones(resultado_igt1, [1])
# %% Modify BAD zone boundaries:
# resultado_igt1 = modificar_zona_bad(resultado_igt1, zona_id=2, nuevo_t_ini=10.0, nuevo_t_fin=15.0)
# %% Add a new BAD zone:
# resultado_igt1 = agregar_bad_zone(resultado_igt1, t_ini=20.0, t_fin=22.0)
# %% Verify result:
# reinspeccionar_segmento(resultado_igt1)

# %% REVERT CLUSTER DELETIONS
# resultado_igt1 = revertir_borrado_clusters(resultado_igt1)

# %% ========== IGT1 - STEP 3: Save ==========
resultado_igt1, _ = guardar_segmento(resultado_igt1, resultados_segmentos_limpios, "IGT1")


# =============================================================================
# PD (variable duration)
# =============================================================================
# %% ========== PD - STEP 1: Inspection ==========
seg_name = "PD"
resultado_pd = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_pd = copy.deepcopy(resultado_pd)

# %% Optional: Zoom into specific time
# ampliar_senal_en_tiempo(resultado_pd, t_centro=30, ventana_s=10.0)

# ========== PD - STEP 2: Cleaning (only if needed) ==========
# %% MANUAL PEAK CORRECTION (interactive editor)
# resultado_pd = editar_peaks_interactivos_externo(resultado_pd, enable=True, t0=30, win_len=10)

# %% AUTOMATIC CLEANING
# resultado_pd = limpiar(resultado_pd, borrar_clusters=True)
# reinspeccionar_segmento(resultado_pd, win_len=60.0)

# BAD ZONE ADJUSTMENTS
# %% Remove a BAD zone that is not problematic:
# resultado_pd = eliminar_bad_zones(resultado_pd, [1])
# %% Modify BAD zone boundaries:
# resultado_pd = modificar_zona_bad(resultado_pd, zona_id=2, nuevo_t_ini=10.0, nuevo_t_fin=15.0)
# %% Add a new BAD zone:
# resultado_pd = agregar_bad_zone(resultado_pd, t_ini=20.0, t_fin=22.0)
# %% Verify result:
reinspeccionar_segmento(resultado_pd)

# %% REVERT CLUSTER DELETIONS
# resultado_pd = revertir_borrado_clusters(resultado_pd)

# %% ========== PD - STEP 3: Save ==========
resultado_pd, _ = guardar_segmento(resultado_pd, resultados_segmentos_limpios, "PD")


# =============================================================================
# Speech (variable duration)
# =============================================================================
# %% ========== Speech - STEP 1: Inspection ==========
seg_name = "Speech"
resultado_speech = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_speech = copy.deepcopy(resultado_speech)

# %% Optional: Zoom into specific time
# ampliar_senal_en_tiempo(resultado_speech, t_centro=30, ventana_s=10.0)

# ========== Speech - STEP 2: Cleaning (only if needed) ==========
# %% MANUAL PEAK CORRECTION (interactive editor)
# resultado_speech = editar_peaks_interactivos_externo(resultado_speech, enable=True, t0=30, win_len=10)

# %% AUTOMATIC CLEANING
# resultado_speech = limpiar(resultado_speech, borrar_clusters=True)
# reinspeccionar_segmento(resultado_speech, win_len=60.0)

# BAD ZONE ADJUSTMENTS
# %% Remove a BAD zone that is not problematic:
# resultado_speech = eliminar_bad_zones(resultado_speech, [1])
# %% Modify BAD zone boundaries:
# resultado_speech = modificar_zona_bad(resultado_speech, zona_id=2, nuevo_t_ini=10.0, nuevo_t_fin=15.0)
# %% Add a new BAD zone:
# resultado_speech = agregar_bad_zone(resultado_speech, t_ini=20.0, t_fin=22.0)
# %% Verify result:
# reinspeccionar_segmento(resultado_speech)

# %% REVERT CLUSTER DELETIONS
# resultado_speech = revertir_borrado_clusters(resultado_speech)

# %% ========== Speech - STEP 3: Save ==========
resultado_speech, _ = guardar_segmento(resultado_speech, resultados_segmentos_limpios, "Speech")


# =============================================================================
# Arit (Arithmetic task - variable duration)
# =============================================================================
# %% ========== Arit - STEP 1: Inspection ==========
seg_name = "Arit"
resultado_arit = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_arit = copy.deepcopy(resultado_arit)

# %% Optional: Zoom into specific time
# ampliar_senal_en_tiempo(resultado_arit, t_centro=30, ventana_s=10.0)

# ========== Arit - STEP 2: Cleaning (only if needed) ==========
# %% MANUAL PEAK CORRECTION (interactive editor)
# resultado_arit = editar_peaks_interactivos_externo(resultado_arit, enable=True, t0=30, win_len=10)

# %% AUTOMATIC CLEANING
# resultado_arit = limpiar(resultado_arit, borrar_clusters=True)
# reinspeccionar_segmento(resultado_arit, win_len=60.0)

# BAD ZONE ADJUSTMENTS
# %% Remove a BAD zone that is not problematic:
# resultado_arit = eliminar_bad_zones(resultado_arit, [1])
# %% Modify BAD zone boundaries:
# resultado_arit = modificar_zona_bad(resultado_arit, zona_id=2, nuevo_t_ini=10.0, nuevo_t_fin=15.0)
# %% Add a new BAD zone:
# resultado_arit = agregar_bad_zone(resultado_arit, t_ini=20.0, t_fin=22.0)
# %% Verify result:
# reinspeccionar_segmento(resultado_arit)

# %% REVERT CLUSTER DELETIONS
# resultado_arit = revertir_borrado_clusters(resultado_arit)

# %% ========== Arit - STEP 3: Save ==========
resultado_arit, _ = guardar_segmento(resultado_arit, resultados_segmentos_limpios, "Arit")


# =============================================================================
# IGT2 (Iowa Gambling Task 2 - variable duration)
# =============================================================================
# %% ========== IGT2 - STEP 1: Inspection ==========
seg_name = "IGT2"
resultado_igt2 = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_igt2 = copy.deepcopy(resultado_igt2)

# %% Optional: Zoom into specific time
# ampliar_senal_en_tiempo(resultado_igt2, t_centro=30, ventana_s=10.0)

# ========== IGT2 - STEP 2: Cleaning (only if needed) ==========
# %% MANUAL PEAK CORRECTION (interactive editor)
# resultado_igt2 = editar_peaks_interactivos_externo(resultado_igt2, enable=True, t0=30, win_len=10)

# %% AUTOMATIC CLEANING
# resultado_igt2 = limpiar(resultado_igt2, borrar_clusters=True)
# reinspeccionar_segmento(resultado_igt2, win_len=60.0)

# BAD ZONE ADJUSTMENTS
# %% Remove a BAD zone that is not problematic:
# resultado_igt2 = eliminar_bad_zones(resultado_igt2, [1])
# %% Modify BAD zone boundaries:
# resultado_igt2 = modificar_zona_bad(resultado_igt2, zona_id=2, nuevo_t_ini=10.0, nuevo_t_fin=15.0)
# %% Add a new BAD zone:
# resultado_igt2 = agregar_bad_zone(resultado_igt2, t_ini=20.0, t_fin=22.0)
# %% Verify result:
# reinspeccionar_segmento(resultado_igt2)

# %% REVERT CLUSTER DELETIONS
# resultado_igt2 = revertir_borrado_clusters(resultado_igt2)

# %% ========== IGT2 - STEP 3: Save ==========
resultado_igt2, _ = guardar_segmento(resultado_igt2, resultados_segmentos_limpios, "IGT2")


# =============================================================================
# RECOVERY - 5-minute window before "Post 10min" marker (requires 290s clean)
# =============================================================================
# %% ========== RECOVERY - STEP 1: Inspection ==========
seg_name = "Recovery"
resultado_recovery = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_recovery = copy.deepcopy(resultado_recovery)

# %% Optional: Zoom into specific time
# ampliar_senal_en_tiempo(resultado_recovery, t_centro=150, ventana_s=10.0)

# ========== RECOVERY - STEP 2: Cleaning (only if needed) ==========
# %% MANUAL PEAK CORRECTION (interactive editor)
# resultado_recovery = editar_peaks_interactivos_externo(resultado_recovery, enable=True, t0=30, win_len=10)

# %% AUTOMATIC CLEANING
# resultado_recovery = limpiar(resultado_recovery, borrar_clusters=True)
# reinspeccionar_segmento(resultado_recovery, win_len=60.0, min_clean_duration_s=290.0)

# BAD ZONE ADJUSTMENTS
# %% Remove a BAD zone that is not problematic:
# resultado_recovery = eliminar_bad_zones(resultado_recovery, [1])
# %% Modify BAD zone boundaries:
# resultado_recovery = modificar_zona_bad(resultado_recovery, zona_id=1, nuevo_t_ini=10.0, nuevo_t_fin=15.0)
# %% Add a new BAD zone:
# resultado_recovery = agregar_bad_zone(resultado_recovery, t_ini=100.0, t_fin=105.0)
# %% Verify result:
# reinspeccionar_segmento(resultado_recovery, win_len=60.0)
# %% REVERT CLUSTER DELETIONS
# resultado_recovery = revertir_borrado_clusters(resultado_recovery)

# %% ========== RECOVERY - STEP 3: Save ==========
resultado_recovery, _ = guardar_segmento(resultado_recovery, resultados_segmentos_limpios, "Recovery", target_duration_s=300.0)








# =============================================================================
# FINAL STAGE: INTERPOLATION, OUTLIERS AND EXPORT
# =============================================================================

# %% ========== STEP 1: Interpolate all segments ==========
resultados_segmentos_limpios = interpolar_todos_segmentos(resultados_segmentos_limpios)


# %% ========== STEP 2: Check outliers ==========
# If issues found, go back and clean the affected segment
resultados_segmentos_limpios, hay_problemas = verificar_outliers_todos(resultados_segmentos_limpios)


# %% ========== STEP 3: Export + Summary + Save cumulative ==========
# Only run if STEP 2 is OK (no outliers)
CARPETA_HRV_OUTPUT = cfg.OUT['ecg']
ARCHIVO_RESUMEN = cfg.OUT['ecg'] / 'RESUMEN_TODOS_SUJETOS.csv'

output_file = exportar_y_resumir(
    resultados=resultados_segmentos_limpios,
    sujeto_id=SUJETO,
    output_folder=CARPETA_HRV_OUTPUT,
    archivo_resumen=ARCHIVO_RESUMEN
)



# %%
