"""Compute the final parameters from the cleaned files of one study.

    python studies/run_analysis.py breath_hold            # hrv, eda, rsp, rsa
    python studies/run_analysis.py alcohol hrv eda        # only some signals

Reads studies/<study>/config.py (cleaned-file folders and segment lists) and writes
<RESULTS_DIR>/{HRV,EDA,RSP,RSA}_parametros_finales.xlsx.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config(study):
    path = ROOT / "studies" / study / "config.py"
    sys.path.insert(0, str(path.parent))          # lets config.py find its config_local.py
    spec = importlib.util.spec_from_file_location(f"config_{study}", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def configure(module, settings):
    for name, value in settings.items():
        if not hasattr(module, name):
            raise AttributeError(f"{module.__name__} has no setting {name}")
        setattr(module, name, value)


def main(study, signals):
    cfg = load_config(study)
    out = Path(cfg.RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    if "hrv" in signals and "hrv" in cfg.ANALYSIS:
        from biopac_physio.ecg import hrv
        configure(hrv, cfg.ANALYSIS["hrv"])
        hrv.RSP_FOLDER = str(cfg.OUT["rsp"])       # breath-hold end times come from the RSP files
        hrv.process_all_subjects(input_folder=str(cfg.OUT["ecg"]), output_file=str(out / "HRV_parametros_finales.xlsx"),
                                 segments=hrv.SEGMENTS, verbose=True)
    if "eda" in signals and "eda" in cfg.ANALYSIS:
        from biopac_physio.eda import analysis as eda
        configure(eda, cfg.ANALYSIS["eda"])
        eda.RSP_FOLDER = str(cfg.OUT["rsp"])
        eda.process_all_subjects(input_folder=str(cfg.OUT["eda"]), output_file=str(out / "EDA_parametros_finales.xlsx"),
                                 segments=eda.SEGMENTS, verbose=True)
    if "rsp" in signals and "rsp" in cfg.ANALYSIS:
        from biopac_physio.rsp import analysis as rsp
        configure(rsp, cfg.ANALYSIS["rsp"])
        rsp.RSP_INPUT_FOLDER = str(cfg.OUT["rsp"])
        rsp.OUTPUT_FILE = str(out / "RSP_parametros_finales.xlsx")
        rsp.main()
    if "rsa" in signals and "rsa" in cfg.ANALYSIS:
        from biopac_physio.rsa import analysis as rsa
        configure(rsa, cfg.ANALYSIS["rsa"])
        rsa.RSP_INPUT_FOLDER, rsa.HRV_INPUT_FOLDER = str(cfg.OUT["rsp"]), str(cfg.OUT["ecg"])
        rsa.OUTPUT_FILE = str(out / "RSA_parametros_finales.xlsx")
        rsa.main()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], set(sys.argv[2:]) or {"hrv", "eda", "rsp", "rsa"})
