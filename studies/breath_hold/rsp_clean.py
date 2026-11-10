# %% Breath-hold study — per-subject RSP cleaning (breath_hold)
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
SUJETO = "1"  # Change to actual subject ID
ACQ_FILE = str(cfg.acq_path(SUJETO))
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

if "Recovery" in segmentos_dict and str(SUJETO) in ("27", "30"):
    end_idx = segmentos_dict["Recovery"][1]
    start_idx = max(0, end_idx - int(300 * fs_rsp))
    segmentos_dict["Recovery"] = (start_idx, end_idx)
    print(f"  [Sujeto {SUJETO}] Recovery: inicio = Fin Recovery - 300s (duración 300s)")


# =============================================================================
# LB (BASELINE) - ~300s
# =============================================================================
# %% ========== LB - PASO 1: Inspección visual (FULL 300s) ==========
seg_name = "LB"
resultado_LB = inspeccionar_segmento_rsp(
    rsp_raw, fs_rsp, eventos, seg_name=seg_name,
    win_len=None, segmentos_dict=segmentos_dict,
    mostrar_info_segmentos_dict=False
)
_checkpoint_LB = copy.deepcopy(resultado_LB)
ver_señal_completa(resultado_LB, show_bad=False)

# %% ========== LB - PASO 2: Edición manual de picos ==========
# resultado_LB = editar_ciclos_interactivos_externo(
#     resultado_LB, enable=True, 
#     t0=150.0, win_len=60.0)
ver_señal_completa(resultado_LB, show_bad=False)

# %% ========== LB - PASO 3: Convertir a NaN/interpolar ==========
resultado_LB = convertir_cambios_a_bad_zones(resultado_LB)
# resultado_LB = limpiar(
#     resultado_LB,
#     aplicar_interpolacion=True,
#     max_cycles_interp=2,
#     max_duration_interp_s=6.0,
# )

# %% ========== LB - PASO 4: Guardar (recorta a 300s) ==========
resultado_LB, resultados_segmentos_rsp = guardar_segmento(
    resultado_LB, resultados_segmentos_rsp, "LB", target_duration_s=300.0)

# Nota: usar resultado_LB["rsp_masked"] para RSA/coupling (NaN en BAD) y
#       resultado_LB["rsp_interp"] para métricas (zonas cortas interpoladas)


# =============================================================================
# INTERO 1-6 
# =============================================================================
# %% ========== INTERO 1 - Inspección ==========
seg_name = "Intero 1"
resultado_intero1 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                              win_len=None, segmentos_dict=segmentos_dict,
                                              mostrar_info_segmentos_dict=False)
_checkpoint_intero1 = copy.deepcopy(resultado_intero1)
ver_señal_completa(resultado_intero1, show_bad=False)

# %% ========== Edición manual de picos ==========
# resultado_intero1 = editar_ciclos_interactivos_externo(
#     resultado_intero1, enable=True, 
#     t0=20.0, win_len=30.0)
ver_señal_completa(resultado_intero1, show_bad=False)

# %% ========== Convertir a NaN/interpolar ==========
resultado_intero1 = convertir_cambios_a_bad_zones(resultado_intero1)
# resultado_intero1 = limpiar(
#     resultado_intero1,
#     aplicar_interpolacion=True,
#     max_cycles_interp=2,
#     max_duration_interp_s=6.0,
# )
# %% Guardar
resultado_intero1, resultados_segmentos_rsp = guardar_segmento(resultado_intero1, resultados_segmentos_rsp, "Intero 1")


# %% ========== INTERO 2 ==========
seg_name = "Intero 2"
resultado_intero2 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                              win_len=None, segmentos_dict=segmentos_dict,
                                              mostrar_info_segmentos_dict=False)
_checkpoint_intero2 = copy.deepcopy(resultado_intero2)
ver_señal_completa(resultado_intero2, show_bad=False)
# %% ========== Edición manual de picos ==========
# resultado_intero2 = editar_ciclos_interactivos_externo(
#     resultado_intero2, enable=True, 
#     t0=5.0, win_len=30.0)
ver_señal_completa(resultado_intero2, show_bad=False)

# %% ========== Convertir a NaN/interpolar ==========
resultado_intero2 = convertir_cambios_a_bad_zones(resultado_intero2)
# resultado_intero2 = limpiar(
#     resultado_intero2,
#     aplicar_interpolacion=True,
#     max_cycles_interp=2,
#     max_duration_interp_s=6.0,
# )
# %% Guardar
resultado_intero2, resultados_segmentos_rsp = guardar_segmento(resultado_intero2, resultados_segmentos_rsp, "Intero 2")


# %% ========== INTERO 3 ==========
seg_name = "Intero 3"
resultado_intero3 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                              win_len=None, segmentos_dict=segmentos_dict,
                                              mostrar_info_segmentos_dict=False)
_checkpoint_intero3 = copy.deepcopy(resultado_intero3)
ver_señal_completa(resultado_intero3, show_bad=False)
# %% ========== Edición manual de picos ==========
# resultado_intero3 = editar_ciclos_interactivos_externo(
#     resultado_intero3, enable=True, 
#     t0=20.0, win_len=60.0)
ver_señal_completa(resultado_intero3, show_bad=False)

# %% ========== Convertir a NaN/interpolar ==========
resultado_intero3 = convertir_cambios_a_bad_zones(resultado_intero3)
# resultado_intero3 = limpiar(
#     resultado_intero3,
#     aplicar_interpolacion=True,
#     max_cycles_interp=2,
#     max_duration_interp_s=6.0,
# )
# %% Guardar
resultado_intero3, resultados_segmentos_rsp = guardar_segmento(resultado_intero3, resultados_segmentos_rsp, "Intero 3")


# %% ========== INTERO 4 ==========
seg_name = "Intero 4"
resultado_intero4 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                              win_len=None, segmentos_dict=segmentos_dict,
                                              mostrar_info_segmentos_dict=False)
_checkpoint_intero4 = copy.deepcopy(resultado_intero4)
ver_señal_completa(resultado_intero4, show_bad=False)
# %% ========== Edición manual de picos ==========
# resultado_intero4 = editar_ciclos_interactivos_externo(
#     resultado_intero4, enable=True, 
#     t0=0.0, win_len=60.0)
ver_señal_completa(resultado_intero4, show_bad=False)

# %% ========== Convertir a NaN/interpolar ==========
resultado_intero4 = convertir_cambios_a_bad_zones(resultado_intero4)
# resultado_intero4 = limpiar(
#     resultado_intero4,
#     aplicar_interpolacion=True,
#     max_cycles_interp=2,
#     max_duration_interp_s=6.0,
# )
# %% Guardar
resultado_intero4, resultados_segmentos_rsp = guardar_segmento(resultado_intero4, resultados_segmentos_rsp, "Intero 4")


# %% ========== INTERO 5 ==========
seg_name = "Intero 5"
resultado_intero5 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                              win_len=None, segmentos_dict=segmentos_dict,
                                              mostrar_info_segmentos_dict=False)
_checkpoint_intero5 = copy.deepcopy(resultado_intero5)
ver_señal_completa(resultado_intero5, show_bad=False)
# %% ========== Edición manual de picos ==========
# resultado_intero5 = editar_ciclos_interactivos_externo(
#     resultado_intero5, enable=True, 
#     t0=15.0, win_len=60.0)
ver_señal_completa(resultado_intero5, show_bad=False)

# %% ========== Convertir a NaN/interpolar ==========
resultado_intero5 = convertir_cambios_a_bad_zones(resultado_intero5)
# resultado_intero5 = limpiar(
#     resultado_intero5,
#     aplicar_interpolacion=True,
#     max_cycles_interp=2,
#     max_duration_interp_s=6.0,
# )
# %% Guardar
resultado_intero5, resultados_segmentos_rsp = guardar_segmento(resultado_intero5, resultados_segmentos_rsp, "Intero 5")


# %% ========== INTERO 6 ==========
seg_name = "Intero 6"
resultado_intero6 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                              win_len=None, segmentos_dict=segmentos_dict,
                                              mostrar_info_segmentos_dict=False)
_checkpoint_intero6 = copy.deepcopy(resultado_intero6)
ver_señal_completa(resultado_intero6, show_bad=False)
# %% ========== Edición manual de picos ==========
# resultado_intero6 = editar_ciclos_interactivos_externo(
#     resultado_intero6, enable=True, 
#     t0=20.0, win_len=60.0)
ver_señal_completa(resultado_intero6, show_bad=False)
# %% Guardar
resultado_intero6, resultados_segmentos_rsp = guardar_segmento(resultado_intero6, resultados_segmentos_rsp, "Intero 6")


# =============================================================================
# BH 1-4 (Breath Hold - ~60-90s each)
# ⚠️ SPECIAL SEGMENT: Breath-holding creates intentional "flat" periods (apnea)
# ⚠️ DO NOT use automatic artifact detection - it will flag apnea as artifacts!
# ⚠️ DO NOT interpolate - the flat signal IS the valid data (breath hold)
# Only mark TRUE technical artifacts: electrode issues, movement noise, clipping
# =============================================================================
# %% ========== BH 1 - Inspección ==========
seg_name = "BH 1"
resultado_bh1 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                          win_len=None, segmentos_dict=segmentos_dict,
                                          mostrar_info_segmentos_dict=False)
_checkpoint_bh1 = copy.deepcopy(resultado_bh1)
resultado_bh1["bh_end_time_s"] = detectar_fin_breath_hold(resultado_bh1)
ver_señal_completa(resultado_bh1, show_bad=False)
# %%  Marcar manualmente (si la detección automática no funciona)
# resultado_bh1 = marcar_fin_breath_hold(resultado_bh1, 
# t_fin_bh=19)  # Ejemplo: 25s
# Verificar marcador (aparecerá en azul en la visualización)
ver_señal_completa(resultado_bh1, show_bad=False)

# %% Guardar (automáticamente crea versión recortada si bh_end_time_s está marcado)
# Se guardan DOS versiones:
#   - "BH 1": señal completa (con respiración de recuperación)
#   - "BH 1_trimmed": señal recortada en bh_end_time_s (solo breath-hold, para HRV)
# Ambas se exportan a Excel en hojas separadas
resultado_bh1, resultados_segmentos_rsp = guardar_segmento(resultado_bh1, resultados_segmentos_rsp, "BH 1")


# %% ========== BH 2 ==========
seg_name = "BH 2"
resultado_bh2 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                          win_len=60.0, segmentos_dict=segmentos_dict,
                                          mostrar_info_segmentos_dict=False)
_checkpoint_bh2 = copy.deepcopy(resultado_bh2)
resultado_bh2["bh_end_time_s"] = detectar_fin_breath_hold(resultado_bh2)
ver_señal_completa(resultado_bh2, show_bad=False)
# %%
# resultado_bh2 = marcar_fin_breath_hold(resultado_bh2, 
# t_fin_bh=29.0)
ver_señal_completa(resultado_bh2, show_bad=False)
# %% Guardar (automáticamente crea versión recortada si bh_end_time_s está marcado)
resultado_bh2, resultados_segmentos_rsp = guardar_segmento(resultado_bh2, resultados_segmentos_rsp, "BH 2")


# %% ========== BH 3 ==========
seg_name = "BH 3"
resultado_bh3 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                          win_len=60.0, segmentos_dict=segmentos_dict,
                                          mostrar_info_segmentos_dict=False)
_checkpoint_bh3 = copy.deepcopy(resultado_bh3)
resultado_bh3["bh_end_time_s"] = detectar_fin_breath_hold(resultado_bh3)
ver_señal_completa(resultado_bh3, show_bad=False)
# %% 
# resultado_bh3 = marcar_fin_breath_hold(resultado_bh3, 
# t_fin_bh=49.0)
ver_señal_completa(resultado_bh3, show_bad=False)
# %% Guardar (automáticamente crea versión recortada si bh_end_time_s está marcado)
resultado_bh3, resultados_segmentos_rsp = guardar_segmento(resultado_bh3, resultados_segmentos_rsp, "BH 3")


# %% ========== BH 4 ==========
seg_name = "BH 4"
resultado_bh4 = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                          win_len=60.0, segmentos_dict=segmentos_dict,
                                          mostrar_info_segmentos_dict=False)
_checkpoint_bh4 = copy.deepcopy(resultado_bh4)
resultado_bh4["bh_end_time_s"] = detectar_fin_breath_hold(resultado_bh4)
ver_señal_completa(resultado_bh4, show_bad=False)
#%%
# resultado_bh4 = marcar_fin_breath_hold(resultado_bh4, 
# t_fin_bh=49.0)
ver_señal_completa(resultado_bh4, show_bad=False)
# %% Guardar (automáticamente crea versión recortada si bh_end_time_s está marcado)
resultado_bh4, resultados_segmentos_rsp = guardar_segmento(resultado_bh4, resultados_segmentos_rsp, "BH 4")


# =============================================================================
# CPT (Cold Pressor Test - ~180s)
# =============================================================================
# %% ========== CPT ==========
seg_name = "CPT"
resultado_cpt = inspeccionar_segmento_rsp(rsp_raw, fs_rsp, eventos, seg_name=seg_name,
                                          win_len=60.0, segmentos_dict=segmentos_dict,
                                          mostrar_info_segmentos_dict=False)
_checkpoint_cpt = copy.deepcopy(resultado_cpt)
ver_señal_completa(resultado_cpt, show_bad=False)
# %% ==========  Edición manual de picos ==========
# resultado_cpt = editar_ciclos_interactivos_externo(
#     resultado_cpt, enable=True, 
#     t0=100.0, win_len=60.0)
ver_señal_completa(resultado_cpt, show_bad=False)
# %% ========== Convertir a NaN/interpolar ==========
resultado_cpt = convertir_cambios_a_bad_zones(resultado_cpt)
# resultado_cpt = limpiar(
#     resultado_cpt,
#     aplicar_interpolacion=True,
#     max_cycles_interp=2,
#     max_duration_interp_s=6.0,
# )

# %% Guardar
resultado_cpt, resultados_segmentos_rsp = guardar_segmento(resultado_cpt, resultados_segmentos_rsp, "CPT")


# =============================================================================
# =============================================================================
# RECOVERY - ~300s
# =============================================================================
# %% ========== RECOVERY - PASO 1: Inspección visual (FULL 300s) ==========
seg_name = "Recovery"
resultado_recovery = inspeccionar_segmento_rsp(
    rsp_raw, fs_rsp, eventos, seg_name=seg_name,
    win_len=None, segmentos_dict=segmentos_dict,
    mostrar_info_segmentos_dict=False
)
_checkpoint_recovery = copy.deepcopy(resultado_recovery)
ver_señal_completa(resultado_recovery, show_bad=False)

# %% ========== RECOVERY - PASO 2: Edición manual de picos ==========
# resultado_recovery = editar_ciclos_interactivos_externo(
#     resultado_recovery, enable=True, 
#     t0=220.0, win_len=60.0)
ver_señal_completa(resultado_recovery, show_bad=False)

# %% ========== RECOVERY - PASO 3: Convertir a NaN/interpolar ==========
resultado_recovery = convertir_cambios_a_bad_zones(resultado_recovery)
# resultado_recovery = limpiar(
#     resultado_recovery,
#     aplicar_interpolacion=True,
#     max_cycles_interp=2,
#     max_duration_interp_s=6.0,
# )

# %% ========== RECOVERY - PASO 4: Guardar (recorta a 300s) ==========
resultado_recovery, resultados_segmentos_rsp = guardar_segmento(
    resultado_recovery, resultados_segmentos_rsp, "Recovery", target_duration_s=300.0)

# Nota: usar resultado_recovery["rsp_masked"] para RSA/coupling (NaN en BAD) y
#       resultado_recovery["rsp_interp"] para métricas (zonas cortas interpoladas)


# =============================================================================
# FINAL: VERIFICACIÓN + EXPORT
# =============================================================================
# %% ========== PASO 1: Verificar outliers en todos los segmentos ==========
# Si hay problemas, volver atrás y limpiar mejor el segmento afectado
resultados_segmentos_rsp, hay_problemas = verificar_outliers_todos(resultados_segmentos_rsp)

# %% ========== PASO 2: Resumen antes de exportar ==========
# Solo ejecutar si PASO 1 está OK (sin problemas de outliers)
df_resumen = resumen_antes_exportar(resultados_segmentos_rsp)


# %% ========== PASO 3: Exportar + Guardar acumulativo ==========
output_file = exportar_y_resumir_rsp(
    resultados=resultados_segmentos_rsp,
    sujeto_id=SUJETO,
    output_folder=CARPETA_SALIDA_RSP,
    archivo_resumen=ARCHIVO_RESUMEN_RSP
)


# %%
