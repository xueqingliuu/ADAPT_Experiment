# ADAPT RCT simulation testbed

Simulation environment and RL algorithms for the ADAPT walking-suggestion MRT.

This repository ships the **fitted generative testbed** (per-user environment parameters, priors, and STE-tuned copies) plus the online experiment code. It does **not** include raw participant-level MRT panels.

## What is public

- Fitted environment folders: `env_para_vanilla/`, `env_para_ste0.2/`, `env_para_ste0.5/`, `env_para_ste0.8/`
- Simulator: `vani_env.py`
- Online experiment and agents: `experiment.py`, `algorithm_helpers.py`, `agents/`
- Prior estimation and STE tuning: `est_prior.py`, `tune_ste.py`, `ste_vanilla.py`
- Aggregation and paper figures: `aggregate.py`, `scripts/fig_forest_vs_never.py`
- Fitting pipeline (scripts `0_`–`8_`) used to build the testbed from private extracts

## What is not public

`df_fit_11week.csv` and other participant-level extracts stay local. They are gitignored. To refit the testbed you need the ADAPT MRT extracts and:

```bash
export ADAPR_COMBINED_DIR=/path/to/extracted/tables
# optional, only for script 0 (raw daily JSON exports)
export ADAPR_RAW_DIR=/path/to/ADAPT_MRT
```

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run a simulation experiment

```bash
# one seed, vanilla environment
python experiment.py --seed 0

# cluster array (see run_array.sh for RESULTS_ROOT / STE folders)
bash run_array.sh
python aggregate.py
```

STE copies are selected with `ADAPR_PARAMS_DIR`, for example `env_para_ste0.8`.

## Refit notes

Scripts `0_combine_json.py` through `6_est_Ew_weights.py` rebuild `env_para_vanilla/` from private data. `5b_patch_generator.py` writes generator calibrations. `est_prior.py` writes RLSVI priors. These steps are not required to replay experiments from the committed parameter files.
