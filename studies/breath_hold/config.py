"""Breath-hold study: file names, segments and analysis settings.

Real data paths go in config_local.py next to this file (git-ignored), e.g.
    from pathlib import Path
    ACQ_DIR = Path("/path/to/participants_biopac_bh")
    OUT_DIR = Path("/path/to/clean_outputs")
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACQ_DIR = HERE / "data" / "acq"          # biopac{ID}.acq
OUT_DIR = HERE / "data" / "clean"
RESULTS_DIR = None                       # *_parametros_finales.xlsx; None = OUT_DIR

# Segments marked in AcqKnowledge ("<name>" start / "Fin <name>" end markers)
SEGMENTS = [
    "LB",
    "Intero 1", "Intero 2", "Intero 3", "Intero 4", "Intero 5", "Intero 6",
    "BH 1", "BH 2", "BH 3", "BH 4",
    "CPT",
    "Recovery",
]
TARGET_DURATIONS = {"LB": 300.0, "Recovery": 300.0}
MANUAL_SEGMENTS = ()                      # every segment comes from its markers


def acq_path(sujeto):
    return Path(ACQ_DIR) / f"biopac{sujeto}.acq"


try:
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass

RESULTS_DIR = Path(RESULTS_DIR or OUT_DIR)
if "OUT" not in globals():                # config_local.py may give its own folder per signal
    OUT = {"ecg": Path(OUT_DIR) / "HRV_clean_data",
           "eda": Path(OUT_DIR) / "EDA_clean_data",
           "rsp": Path(OUT_DIR) / "RSP_clean_data"}

# ---------------------------------------------------------------- analysis (run_analysis.py)
# Breath-hold segments are analysed separately, trimmed to the apnoea (end detected on RSP).
ANALYSIS = {
    "hrv": dict(SEGMENTS=["LB", "Intero 1", "Intero 2", "Intero 3", "Intero 4", "Intero 5", "Intero 6",
                          "CPT", "Recovery"],
                BH_SEGMENTS=["BH 1", "BH 2", "BH 3", "BH 4"],
                FIXED_WINDOWS=[
                    {"name": "LB_60s", "segment": "LB", "start_s": 180.0, "end_s": 240.0, "include_freq": True},
                    {"name": "Recovery_60s", "segment": "Recovery", "start_s": 180.0, "end_s": 240.0, "include_freq": True},
                    {"name": "CPT_60s", "segment": "CPT", "start_s": 10.0, "end_s": 70.0, "include_freq": False},
                ],
                TRIM_BASELINE_TO_300S=True),
    "eda": dict(SEGMENTS=["LB", "Intero 1", "Intero 2", "Intero 3", "Intero 4", "Intero 5", "Intero 6",
                          "CPT", "Recovery"],
                BH_SEGMENTS=["BH 1", "BH 2", "BH 3", "BH 4"]),
    "rsp": dict(SEGMENTS=["LB", "Intero 1", "Intero 2", "Intero 3", "Intero 4", "Intero 5", "Intero 6",
                          "CPT", "Recovery"]),
    "rsa": dict(SEGMENTS=["LB", "Intero 1", "Intero 2", "Intero 3", "Intero 4", "Intero 5", "Intero 6",
                          "BH 1", "BH 2", "BH 3", "CPT", "Recovery"],
                BH_SEGMENTS={"BH 1", "BH 2", "BH 3"}),
}
