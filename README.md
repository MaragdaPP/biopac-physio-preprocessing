# biopac-physio-preprocessing

Semi-automatic preprocessing of peripheral physiology recorded with BIOPAC (MP160 / AcqKnowledge `.acq`):
**ECG → HRV**, **EDA**, **respiration (RSP)** and **respiratory sinus arrhythmia (RSA)**.

The cleaning engine is the same for every study. What changes between studies (file names, the event
markers that delimit each segment, which segments are analysed) lives in a small `studies/<study>/config.py`.

## Layout

```
biopac_physio/            the library (study-independent)
  ecg/preprocess.py         R-peak detection, artefact screening, BAD zones, interpolation, export
  ecg/peak_editor.py        interactive window to add / delete / move R peaks
  ecg/hrv.py                time, frequency and non-linear HRV from the cleaned RR series
  eda/preprocess.py         cleaning, tonic/phasic decomposition, SCR detection, BAD zones
  eda/analysis.py           SCL / SCR parameters
  rsp/preprocess.py         cleaning, breath detection, breath-hold end detection, BAD zones
  rsp/peak_editor.py        interactive window for inspiration peaks / troughs
  rsp/analysis.py           rate, amplitude and pattern parameters
  rsa/analysis.py           RSA from cleaned RR + respiration
studies/
  breath_hold/              config.py + ecg_clean.py, eda_clean.py, rsp_clean.py
  alcohol/                  idem (acute-stress protocol: IGT, PD, speech, arithmetic)
  run_analysis.py           cleaned files → *_parametros_finales.xlsx
```

## How it works

1. **Clean, one participant at a time.** Open `studies/<study>/ecg_clean.py` (or `eda_`/`rsp_`) in
   VS Code / Cursor, set `SUJETO`, and run it cell by cell (`# %%`). For each segment:
   1. the signal is shown in 60-s windows with the automatically detected peaks and the automatic
      artefact flags;
   2. *if needed*, the manual-edit cells are uncommented and filled in for **that** participant
      (open the peak editor, delete / add / move peaks, add or adjust BAD zones). The workflow files
      keep these cells commented as examples;
   3. the segment is saved.

   At the end, BAD zones are interpolated, outliers are checked across segments and one Excel file
   per participant is exported, plus a running summary CSV.
2. **Compute parameters for the whole sample:** `python studies/run_analysis.py breath_hold`.

## Method in brief

- **Input:** BIOPAC `.acq` read with `bioread` / NeuroKit2; segments are cut from the event markers
  placed in AcqKnowledge (`<segment>` … `Fin <segment>`), or from a start marker + target duration.
- **ECG:** R peaks detected with `physio` (`compute_ecg`, preset `human_ecg`; S. Garcia,
  https://github.com/samuelgarcia/physio), with NeuroKit2 (`ecg_clean` + `ecg_peaks`) as fallback.
  Automatic screening flags RR < 300 ms or > 1,800 ms, short–long patterns, local RR instability
  (CV > .12 over 12 beats) and noisy or abnormal-amplitude stretches; every 60-s window is inspected
  visually and missed, spurious or displaced peaks are corrected by hand. Unrecoverable stretches are
  marked BAD and interpolated; a segment with > 20 % BAD is excluded. Baseline and recovery need
  ≥ 290 s of clean signal and are trimmed to 300 s.
- **HRV:** time domain, frequency domain (RR series interpolated at 4 Hz, Welch PSD; VLF/LF/HF),
  Poincaré; fixed 60-s windows for cross-segment comparisons.
- **EDA:** resampled to 32 Hz, low-pass 3 Hz, decomposed into tonic and phasic components
  (NeuroKit2 / cvxEDA); SCRs ≥ 0.05 µS with 1–4 s rise time; BAD zones marked by hand.
- **RSP:** resampled to 63 Hz, band-pass 0.05–1 Hz, breaths detected with NeuroKit2 and corrected by
  hand; in breath-hold segments the end of the apnoea (first recovery breath) is detected
  automatically and used to trim ECG/EDA breath-hold segments to the true apnoea.
- **RSA:** from the cleaned RR series and respiration: HF/LF power (HF 0.15–0.40 Hz), breath-by-breath
  peak–valley RSA, Porges–Bohrer RSA and the cardio-respiratory ratio.

Full parameter lists are in the `*_PARAM_NAMES` / `*_PARAM_ORDER` lists of each analysis module, and
each output workbook carries a `Parameter_Definitions` sheet.

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Your data paths

Data are never committed. Create `studies/<study>/config_local.py` (git-ignored):

```python
from pathlib import Path
ACQ_DIR = Path("/path/to/acq_files")
OUT_DIR = Path("/path/to/cleaned_outputs")
```

## Adding a new study

Copy `studies/breath_hold/` to `studies/<new_study>/`, edit `config.py` (file-name pattern in
`acq_path`, `SEGMENTS`, `TARGET_DURATIONS`, `ANALYSIS`) and the segment cells of the `*_clean.py`
workflows. The library does not change.

## How to cite

Please cite the archived version used (Zenodo DOI, see `CITATION.cff`), together with the toolboxes it
builds on: NeuroKit2 (Makowski et al., 2021, *Behavior Research Methods*, 53, 1689–1696) and `physio`.

## License

MIT — see `LICENSE`.
