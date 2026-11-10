# %% Breath-hold study — per-subject ECG cleaning (breath_hold)
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

# %% Editable subject cell
SUJETO = "1"  # cambiar por el ID real del sujeto
ACQ_FILE = str(cfg.acq_path(SUJETO))
ECG_COL = "ECG2307000448"

SEGMENTOS_INTERES = cfg.SEGMENTS

DURACIONES_OBJETIVO = cfg.TARGET_DURATIONS
CARPETA_SALIDA_SUJETO = cfg.OUT['ecg']
ARCHIVO_RESUMEN = cfg.OUT['ecg'] / 'RESUMEN_TODOS_SUJETOS.csv'

resultados_segmentos_limpios: Dict[str, dict] = {}
ecg_raw, fs_ecg, eventos, eventos_full, data_events, segmentos_dict = cargar_datos_y_segmentos(
    ACQ_FILE, ECG_COL, SEGMENTOS_INTERES, DURACIONES_OBJETIVO,
    segmentos_manuales=cfg.MANUAL_SEGMENTS,
)

inspeccionar_duraciones_segmentos(segmentos_dict, SEGMENTOS_INTERES, fs_ecg, SUJETO)


# =============================================================================
# LB (BASELINE) - Requiere 290s de datos limpios
# =============================================================================
# %% ========== LB - PASO 1: Inspección inicial ==========
seg_name = "LB"
resultado_LB = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_LB = copy.deepcopy(resultado_LB)  # Guardar checkpoint para poder revertir

# %% Opcional: Ampliar señal en tiempo específico
# TIEMPO_AMPLIAR = 19.0
# VENTANA_AMPLIAR = 10.0
# ampliar_senal_en_tiempo(resultado_LB, t_centro=TIEMPO_AMPLIAR, ventana_s=VENTANA_AMPLIAR)

# ========== LB - PASO 2: Limpieza (SOLO si hay problemas) ==========
# %%  CORRECCION MANUAL DE PEAKS (Editor interactivo)
# resultado_LB = editar_peaks_interactivos_externo(resultado_LB, 
# enable=True, t0=60, win_len=10)

# %% LIMPIEZA AUTOMÁTICA 
# resultado_LB = limpiar(resultado_LB, borrar_clusters=True, min_clean_duration_s=290.0)
reinspeccionar_segmento(resultado_LB, win_len=60.0, min_clean_duration_s=290.0)

# MODIFICAR ZONAS BAD 
# %%  Eliminar zona BAD que NO es problemática:
# resultado_LB = eliminar_bad_zones(resultado_LB, [1])
# %%  Modificar límites de zona BAD (extender/reducir):
# resultado_LB = modificar_zona_bad(resultado_LB, 
# zona_id=1, nuevo_t_ini=282.5, nuevo_t_fin=285.0)
# %%  Agregar zona BAD nueva:
# resultado_LB = agregar_bad_zone(resultado_LB, t_ini=100.0, t_fin=105.0)
# %%  Verificar resultado:
reinspeccionar_segmento(resultado_LB, win_len=60.0, min_clean_duration_s=290.0)

# %%  REVERTIR BORRADOS PEAKS EN CLUSTERS
# resultado_LB = revertir_borrado_clusters(resultado_LB)

# %% ========== LB - PASO 3: Guardar ==========
resultado_LB, _ = guardar_segmento(
    resultado_LB, resultados_segmentos_limpios, "LB",
    min_clean_duration_s=290.0, target_duration_s=300.0
)


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
# resultado_intero1 = editar_peaks_interactivos_externo(resultado_intero1, enable=True, t0=30, win_len=10)

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
# resultado_intero4 = editar_peaks_interactivos_externo(resultado_intero4, 
# enable=True, t0=20, win_len=10)

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
reinspeccionar_segmento(resultado_intero4)

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
# BH 1-4 (Breath Hold - Segmentos cortos ~60-90s)
# =============================================================================

# %% ========== BH 1 - PASO 1: Inspección ==========
seg_name = "BH 1"
resultado_bh1 = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_bh1 = copy.deepcopy(resultado_bh1)

# %% Opcional: Ampliar señal
# ampliar_senal_en_tiempo(resultado_bh1, 
# t_centro=80, ventana_s=10.0)

# ========== BH 1 - PASO 2: Limpieza ==========
# %% CORRECCION MANUAL DE PEAKS (Editor interactivo)
# resultado_bh1 = editar_peaks_interactivos_externo(resultado_bh1, enable=True, t0=30, win_len=10)

# %% LIMPIEZA AUTOMÁTICA
# resultado_bh1 = limpiar(resultado_bh1, borrar_clusters=True)
reinspeccionar_segmento(resultado_bh1, win_len=60.0)

# MODIFICAR ZONAS BAD
# %% Eliminar zona BAD que NO es problemática:
# resultado_bh1 = eliminar_bad_zones(resultado_bh1, [2,3])
# %% Modificar límites de zona BAD:
# resultado_bh1 = modificar_zona_bad(resultado_bh1, 
# zona_id=1, nuevo_t_ini=125, nuevo_t_fin=128)
# %% Agregar zona BAD nueva:
# resultado_bh1 = agregar_bad_zone(resultado_bh1, t_ini=20.0, t_fin=22.0)
# %% Verificar resultado:
reinspeccionar_segmento(resultado_bh1)

# %% REVERTIR BORRADOS PEAKS EN CLUSTERS
# resultado_bh1 = revertir_borrado_clusters(resultado_bh1)

# %% ========== BH 1 - PASO 3: Guardar ==========
resultado_bh1, _ = guardar_segmento(resultado_bh1, resultados_segmentos_limpios, "BH 1")


# %% ========== BH 2 - PASO 1: Inspección ==========
seg_name = "BH 2"
resultado_bh2 = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_bh2 = copy.deepcopy(resultado_bh2)

# %% Opcional: Ampliar señal
# ampliar_senal_en_tiempo(resultado_bh2, t_centro=30, ventana_s=10.0)

# ========== BH 2 - PASO 2: Limpieza ==========
# %% CORRECCION MANUAL DE PEAKS (Editor interactivo)
# resultado_bh2 = editar_peaks_interactivos_externo(resultado_bh2, 
# enable=True, t0=50, win_len=10)
# %% Verificar resultado:
reinspeccionar_segmento(resultado_bh2)

# %% LIMPIEZA AUTOMÁTICA
# resultado_bh2 = limpiar(resultado_bh2, borrar_clusters=True)
reinspeccionar_segmento(resultado_bh2, win_len=60.0)

# MODIFICAR ZONAS BAD
# %% Eliminar zona BAD que NO es problemática:
# resultado_bh2 = eliminar_bad_zones(resultado_bh2, [1])
# %% Modificar límites de zona BAD:
# resultado_bh2 = modificar_zona_bad(resultado_bh2, zona_id=2, nuevo_t_ini=10.0, nuevo_t_fin=15.0)
# %% Agregar zona BAD nueva:
# resultado_bh2 = agregar_bad_zone(resultado_bh2, t_ini=20.0, t_fin=22.0)
# %% Verificar resultado:
reinspeccionar_segmento(resultado_bh2)

# %% REVERTIR BORRADOS PEAKS EN CLUSTERS
# resultado_bh2 = revertir_borrado_clusters(resultado_bh2)

# %% ========== BH 2 - PASO 3: Guardar ==========
resultado_bh2, _ = guardar_segmento(resultado_bh2, resultados_segmentos_limpios, "BH 2")


# %% ========== BH 3 - PASO 1: Inspección ==========
seg_name = "BH 3"
resultado_bh3 = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_bh3 = copy.deepcopy(resultado_bh3)

# %% Opcional: Ampliar señal
# ampliar_senal_en_tiempo(resultado_bh3, t_centro=30, ventana_s=10.0)

# ========== BH 3 - PASO 2: Limpieza ==========
# %% CORRECCION MANUAL DE PEAKS (Editor interactivo)
# resultado_bh3 = editar_peaks_interactivos_externo(resultado_bh3, 
# enable=True, t0=20, win_len=10)

# %% LIMPIEZA AUTOMÁTICA
# resultado_bh3 = limpiar(resultado_bh3, borrar_clusters=True)
reinspeccionar_segmento(resultado_bh3, win_len=60.0)

# %% Modificar límites de zona BAD:
# resultado_bh3 = modificar_zona_bad(resultado_bh3, 
# zona_id=1, nuevo_t_ini=20.0, nuevo_t_fin=81.0)
reinspeccionar_segmento(resultado_bh3)

# MODIFICAR ZONAS BAD
# %% Eliminar zona BAD que NO es problemática
# resultado_bh3 = eliminar_bad_zones(resultado_bh3, 
# [2])
# %% Modificar límites de zona BAD:
# resultado_bh3 = modificar_zona_bad(resultado_bh3, 
# zona_id=1, nuevo_t_ini=79.0, nuevo_t_fin=81.0)
# %% Agregar zona BAD nueva:
# resultado_bh3 = agregar_bad_zone(resultado_bh3, t_ini=20.0, t_fin=22.0)
# %% Verificar resultado:
reinspeccionar_segmento(resultado_bh3)

# %% REVERTIR BORRADOS PEAKS EN CLUSTERS
# resultado_bh3 = revertir_borrado_clusters(resultado_bh3)

# %% ========== BH 3 - PASO 3: Guardar ==========
resultado_bh3, _ = guardar_segmento(resultado_bh3, resultados_segmentos_limpios, "BH 3")


# %% ========== BH 4 - PASO 1: Inspección ==========
seg_name = "BH 4"
resultado_bh4 = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_bh4 = copy.deepcopy(resultado_bh4)

# %% Opcional: Ampliar señal
# ampliar_senal_en_tiempo(resultado_bh4, t_centro=30, ventana_s=10.0)

# ========== BH 4 - PASO 2: Limpieza ==========
# %% CORRECCION MANUAL DE PEAKS (Editor interactivo)
# resultado_bh4 = editar_peaks_interactivos_externo(resultado_bh4, 
# enable=True, t0=200, win_len=10)

# %% LIMPIEZA AUTOMÁTICA
# resultado_bh4 = limpiar(resultado_bh4, borrar_clusters=True)
reinspeccionar_segmento(resultado_bh4, win_len=60.0)

# MODIFICAR ZONAS BAD
# %% Eliminar zona BAD que NO es problemática:
# resultado_bh4 = eliminar_bad_zones(resultado_bh4, [1])
# %% Modificar límites de zona BAD:
# resultado_bh4 = modificar_zona_bad(resultado_bh4, 
# zona_id=1, nuevo_t_ini=53.0, nuevo_t_fin=55.8)
# %% Agregar zona BAD nueva:
# resultado_bh4 = agregar_bad_zone(resultado_bh4, 
# t_ini=53.0, t_fin=57.5)
# %% Verificar resultado:
reinspeccionar_segmento(resultado_bh4)

# %% REVERTIR BORRADOS PEAKS EN CLUSTERS
# resultado_bh4 = revertir_borrado_clusters(resultado_bh4)

# %% ========== BH 4 - PASO 3: Guardar ==========
resultado_bh4, _ = guardar_segmento(resultado_bh4, resultados_segmentos_limpios, "BH 4")


# =============================================================================
# CPT (Cold Pressor Test - Segmento medio ~180s)
# =============================================================================
# %% ========== CPT - PASO 1: Inspección ==========
seg_name = "CPT"
resultado_cpt = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_cpt = copy.deepcopy(resultado_cpt)

# %% Opcional: Ampliar señal
# ampliar_senal_en_tiempo(resultado_cpt, t_centro=90, ventana_s=10.0)

# ========== CPT - PASO 2: Limpieza ==========
# %% CORRECCION MANUAL DE PEAKS (Editor interactivo)
# resultado_cpt = editar_peaks_interactivos_externo(resultado_cpt, 
# enable=True, t0=70, win_len=10)

# %% LIMPIEZA AUTOMÁTICA (hr_change_threshold=1.0 para CPT ya que HR cambia naturalmente)
# resultado_cpt = limpiar(resultado_cpt, borrar_clusters=True, hr_change_threshold=1.0)
reinspeccionar_segmento(resultado_cpt, win_len=60.0)

# MODIFICAR ZONAS BAD
# %% Eliminar zona BAD que NO es problemática:
# resultado_cpt = eliminar_bad_zones(resultado_cpt, [1,2,3,4,])
# %% Modificar límites de zona BAD:
# resultado_cpt = modificar_zona_bad(resultado_cpt, zona_id=1, 
# nuevo_t_ini=74.5, nuevo_t_fin=78)
# %% Agregar zona BAD nueva:
# resultado_cpt = agregar_bad_zone(resultado_cpt, t_ini=147.5, t_fin=149.7)
# %% Verificar resultado:
reinspeccionar_segmento(resultado_cpt)

# %% REVERTIR BORRADOS PEAKS EN CLUSTERS
# resultado_cpt = revertir_borrado_clusters(resultado_cpt)

# %% ========== CPT - PASO 3: Guardar ==========
resultado_cpt, _ = guardar_segmento(resultado_cpt, resultados_segmentos_limpios, "CPT")


# =============================================================================
# RECOVERY - Requiere 290s de datos limpios
# =============================================================================
# %% ========== RECOVERY - PASO 1: Inspección ==========
seg_name = "Recovery"
resultado_recovery = inspeccionar_segmento_60s(ecg_raw, fs_ecg, eventos, seg_name=seg_name, win_len=60.0, segmentos_dict=segmentos_dict, mostrar_info_segmentos_dict=False)
_checkpoint_recovery = copy.deepcopy(resultado_recovery)

# %% Opcional: Ampliar señal
# ampliar_senal_en_tiempo(resultado_recovery, t_centro=150, ventana_s=10.0)

# ========== RECOVERY - PASO 2: Limpieza ==========
# %%  CORRECCION MANUAL DE PEAKS (Editor interactivo)
# resultado_recovery = editar_peaks_interactivos_externo(resultado_recovery, 
# enable=True, t0=220, win_len=10)

# %% LIMPIEZA 
# resultado_recovery = limpiar(resultado_recovery, borrar_clusters=True)
reinspeccionar_segmento(resultado_recovery, win_len=60.0)
# MODIFICAR ZONAS BAD
# %% Modificar límites de zona BAD:
# resultado_recovery = modificar_zona_bad(resultado_recovery, 
# zona_id=1, nuevo_t_ini=47, nuevo_t_fin=49.0)
# %% Agregar zona BAD nueva:
# resultado_recovery = agregar_bad_zone(resultado_recovery, t_ini=100.0, t_fin=105.0)
# %% Verificar resultado:
reinspeccionar_segmento(resultado_recovery, win_len=60.0)
# %% REVERTIR BORRADOS PEAKS EN CLUSTERS
# resultado_recovery = revertir_borrado_clusters(resultado_recovery)


# %% ========== RECOVERY - PASO 3: Guardar ==========
resultado_recovery, qc_valid_recovery = guardar_segmento(
    resultado_recovery, resultados_segmentos_limpios, "Recovery", 
    target_duration_s=300.0)








# =============================================================================
# ETAPA FINAL: INTERPOLACION, OUTLIERS Y EXPORTACION
# =============================================================================

# %% ========== PASO 1: Interpolar todos los segmentos ==========
resultados_segmentos_limpios = interpolar_todos_segmentos(resultados_segmentos_limpios)


# %% ========== PASO 2: Verificar outliers ==========
# Si hay problemas, volver atrás y limpiar mejor el segmento afectado
resultados_segmentos_limpios, hay_problemas = verificar_outliers_todos(resultados_segmentos_limpios)


# %% ========== PASO 3: Exportar + Resumen + Guardar acumulativo ==========
# Solo ejecutar si PASO 2 está OK (sin problemas de outliers)
CARPETA_HRV_OUTPUT = cfg.OUT['ecg']
ARCHIVO_RESUMEN = cfg.OUT['ecg'] / 'RESUMEN_TODOS_SUJETOS.csv'

output_file = exportar_y_resumir(
    resultados=resultados_segmentos_limpios,
    sujeto_id=SUJETO,
    output_folder=CARPETA_HRV_OUTPUT,
    archivo_resumen=ARCHIVO_RESUMEN
)



# %%
