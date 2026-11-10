# Changelog

## 1.0.0 — 2026-11-10

First generic release, merged from the study copies used in the breath-hold study
(`analysis_electrophysiology_bh`) and the alcohol / acute-stress study.

- One library per signal (`biopac_physio/`), one `config.py` + cleaning workflows per study (`studies/`).
- ECG and RSP preprocessing: alcohol copy (English messages, macOS-safe peak-editor window, more robust
  editor launch). Its hard-coded "skip Recovery" became the `segmentos_manuales` argument
  (default: none, as in the breath-hold copy).
- EDA preprocessing: breath-hold copy (exports the trimmed 300-s duration).
- Analysis modules: breath-hold copies (superset: breath-hold trimming, CPT window), plus
  - `np.trapezoid` → works on NumPy < 2.0 too (HRV, RSA). On NumPy 1.x the original RSA script
    silently returned empty LF/HF powers;
  - `TRIM_BASELINE_TO_300S` setting (breath-hold True, alcohol False, as originally run).
- Verified against the original outputs: breath-hold HRV, EDA, RSP and RSA and alcohol HRV, EDA and RSP
  parameters identical (0 of ~180,000 numeric cells differ); segment extraction and R-peak detection
  identical on real recordings.
