"""Alcohol / acute-stress study: file names, segments and analysis settings.

Real data paths go in config_local.py next to this file (git-ignored), e.g.
    from pathlib import Path
    ACQ_DIR = Path("/path/to/raw_biopac_proyectoalcohol")
    OUT_DIR = Path("/path/to/clean_outputs")
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACQ_DIR = HERE / "data" / "acq"          # "{ID}{group} Biopac.acq", group E = experimental, C = control
OUT_DIR = HERE / "data" / "clean"
RESULTS_DIR = None                       # *_parametros_finales.xlsx; None = OUT_DIR

SEGMENTS = [
    "LB",
    "Intero 1", "Intero 2", "Intero 3", "Intero 4", "Intero 5", "Intero 6",
    "IGT1",
    "PD",
    "Speech",
    "Arit",
    "IGT2",
    "Recovery",
]
TARGET_DURATIONS = {"LB": 300.0, "Recovery": 300.0}
# Recovery has no start/end markers here: the workflow builds it as the 5 min ending at "Post 10min".
MANUAL_SEGMENTS = ("Recovery",)


def acq_path(sujeto, grupo):
    return Path(ACQ_DIR) / f"{sujeto}{grupo} Biopac.acq"


try:
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass

RESULTS_DIR = Path(RESULTS_DIR or OUT_DIR)
if "OUT" not in globals():                # config_local.py may give its own folder per signal
    OUT = {"ecg": Path(OUT_DIR) / "Limpios_ecg",
           "eda": Path(OUT_DIR) / "Limpios_eda",
           "rsp": Path(OUT_DIR) / "Limpios_rsp"}

# ---------------------------------------------------------------- analysis (run_analysis.py)
ANALYSIS = {
    "hrv": dict(SEGMENTS=SEGMENTS, BH_SEGMENTS=[],
                FIXED_WINDOWS=[
                    {"name": "LB_60s", "segment": "LB", "start_s": 180.0, "end_s": 240.0, "include_freq": True},
                    {"name": "Recovery_60s", "segment": "Recovery", "start_s": 180.0, "end_s": 240.0, "include_freq": True},
                ],
                TRIM_BASELINE_TO_300S=False),     # as in the original alcohol run
    "eda": dict(SEGMENTS=SEGMENTS, BH_SEGMENTS=[]),
    "rsp": dict(SEGMENTS=SEGMENTS),
}
