#!/bin/bash
#SBATCH -J refit47
#SBATCH -p murphy
#SBATCH -c 4
#SBATCH -t 2-00:00:00
#SBATCH --mem=32GB
#SBATCH -o logs/refit47_%j.out
#SBATCH -e logs/refit47_%j.err
#
# Cluster part of the 47-user refit (REFIT_CHECKLIST_47users.md, sections 4–5).
#
#   mkdir -p logs
#   sbatch --export=ALL,ADAPR_COMBINED_DIR=/n/home09/xliu08/ADAPT_data/combined_20261014 run_refit.sh
#   sbatch --export=ALL,ADAPR_COMBINED_DIR=...,STAGE=fit    run_refit.sh   # 06, 07, 08, check fit
#   sbatch --export=ALL,ADAPR_COMBINED_DIR=...,STAGE=priors run_refit.sh   # 09, 09 --loo, 10, check priors
#
# STAGE=all (default) runs both, stopping at the first failing step or check.
# Parameters are written to the folder chosen by vani_env.resolve_params_dir()
# (env_para_vanilla unless ADAPR_PARAMS_DIR is set).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# The fits need pymc / arviz / pytensor, which the RL environment (budgeted) does not have.
# REFIT_ENV names the conda env for these steps (default adapt_fit); REFIT_PYTHON, if set,
# is used as-is and skips the module stack.
REFIT_ENV="${REFIT_ENV:-adapt_fit}"
if [[ -z "${REFIT_PYTHON:-}" ]]; then
  set +u                      # module / conda scripts read unset variables
  module load python
  eval "$(conda shell.bash hook)"
  mamba activate "$REFIT_ENV" 2>/dev/null || conda activate "$REFIT_ENV"
  set -u
fi
PY="${REFIT_PYTHON:-python}"

: "${ADAPR_COMBINED_DIR:?set ADAPR_COMBINED_DIR to the folder with the transferred extract files}"
export ADAPR_COMBINED_DIR
STAGE="${STAGE:-all}"

echo "commit: $(git rev-parse --short HEAD)   stage: ${STAGE}   combined dir: ${ADAPR_COMBINED_DIR}"
echo "start:  $(date)"
"$PY" - <<'PYEOF'
import importlib, sys
print("python", sys.version.split()[0])
missing = []
for m in ("numpy", "pandas", "scipy", "sklearn", "statsmodels", "pymc", "arviz", "pytensor"):
    try:
        print(f"  {m} {importlib.import_module(m).__version__}")
    except ImportError:
        missing.append(m)
if missing:
    sys.exit(f"missing packages in this environment: {', '.join(missing)}")
PYEOF

step() {
  echo
  echo "=== $* ($(date +%H:%M:%S))"
  "$@"
}

if [[ "$STAGE" == "all" || "$STAGE" == "fit" ]]; then
  step "$PY" 06_fit_pu.py --chains 4 --cores "${SLURM_CPUS_PER_TASK:-4}"
  step "$PY" 07_fit_testbed.py
  step "$PY" 08_est_ew_weights.py
  step "$PY" 05_check_refit.py --stage fit
fi

if [[ "$STAGE" == "all" || "$STAGE" == "priors" ]]; then
  step "$PY" 09_est_prior.py
  step "$PY" 09_est_prior.py --loo
  step "$PY" 10_stability.py --cap 0.95
  step "$PY" 05_check_refit.py --stage priors
fi

echo
echo "done:   $(date)"
