# AGENTS.md

Guidance for AI coding agents working in this repository. For the full background, decisions and status, read `Project-Context.md`.

## Project
Special Problem (CMSC 199, UP Manila): early detection of diabetes risk (FBS >= 100 mg/dL) from the 2018-2021 Philippine ENNS (DOST-FNRI), with calibration and explainability (SHAP, LIME, Calibrated Explanations). Primary metric: F2 (beta = 2). Random seed: 42.

## Pipeline order (run from repo root)
1. `python main_preprocessing.py` -> `MAIN_PREPROCESS_OUTPUT/` (uses `preprocessing_pipeline_v7.py`)
2. `python main_eda_extra.py` -> `MAIN_PREPROCESS_OUTPUT/EDA/eda2_*`
3. `python main_train.py` -> `MAIN_TRAINED_MODELS/`, `MAIN_MODEL_RESULTS/`
4. `python main_xai.py` -> `MAIN_XAI/`
5. `python main_calibration.py` -> `MAIN_CALIBRATION/`, `MAIN_TRAINED_MODELS_CALIB/`
6. `python main_ce.py` (uses `ce_utils.py`, `ce_factual.py`, `ce_cce.py`, `ce_global.py`, `ce_mondrian.py`) -> `MAIN_CE_OUTPUT/`

`V2_Pipeline.ipynb` is a documentation notebook that collects the code of all scripts above. It has `RUN_*` flags; with flags off it only displays saved outputs.

## Conventions
- Paths are absolute and hard-coded in each `*Config` class (e.g. `d:\School Files\CMSC\CMSC 199\SP\V2_SP\...`). Do not "fix" them silently; if you change one, change it in the scripts and the notebook.
- Do not add or remove comments in existing code unless asked.
- Notebook markdown cells: keep the existing structure (section header, "Original script header" quote, one `####` cell per function).
- Never edit thesis text under `Thesis Manuscript/` unless explicitly asked.
- Windows machine; shell commands run under bash (Git Bash). Console encoding is not UTF-8: use `python -X utf8` when printing non-ASCII.

## Git rules
- Never push unless explicitly asked.
- Never commit trained models (`*.joblib`, `*.pkl`, `*.pt`, `*.zip`), `MAIN_TRAINED_MODELS/`, `MAIN_TRAINED_MODELS_CALIB/` (about 4.2 GB each) or `ARCHIVE/` (about 33 GB). They are git-ignored.
- LaTeX build artifacts (`.aux`, `.bbl`, `.toc`, `.out`, `.synctex.gz`, `__MACOSX/`) are git-ignored. Keep PDFs.
- `Thesis Manuscript/Reference Papers/` is git-ignored (copyrighted papers); the files stay local.
- Check any file is under 100 MB before staging (GitHub limit). The largest dataset file is about 94 MB.

## Known issues (do not "fix" without asking; see Project-Context.md)
- V2 applies SMOTE to the full dataset before the 70/15/15 split (data leakage). Reported V2 numbers are therefore optimistic.
- `main_calibration.py` ranks calibrators on test data.
- The IDE markdown notebook renderer error ("Failed to fetch dynamically imported module ... markdown-language-features/notebook-out/index.js") is an IDE problem, not a notebook file problem.
