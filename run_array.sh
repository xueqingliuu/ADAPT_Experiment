#!/bin/bash
#SBATCH -J exp_seed_array
#SBATCH -p murphy
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 2-10:00:00
#SBATCH --mem=3G
#SBATCH --array=1-300%50
# Override for a top-up, e.g. 700 more seeds after 1-300:
#   sbatch --array=301-1000%50 --export=ALL,PRIOR_MODE=loo,ENV_VARIANT=env_para_vanilla,RESULTS_ROOT=results_vanilla_loo run_array.sh
#SBATCH -o logs/exp_%A_%a.out
#SBATCH -e logs/exp_%A_%a.err

# Fitted environment folder under the repo root. Switch later, e.g.:
#   ENV_VARIANT=env_para_ste0.5 sbatch run_array.sh
ENV_VARIANT="${ENV_VARIANT:-env_para_vanilla}"
# Simulated weeks (experiment.py --nweek). Default 36.
#   sbatch --export=ALL,PRIOR_MODE=loo,NWEEK=300,RESULTS_ROOT=results_vanilla_w300_loo run_array.sh
NWEEK="${NWEEK:-36}"
# saved = shared ${ENV_VARIANT}/rl_priors.json, or zeros if USE_ESTIMATED_PRIORS=0.
# loo   = ${ENV_VARIANT}/loo_priors/held_out_<uid>.json (python est_prior.py --loo first).
# Keep saved as the cluster default so USE_ESTIMATED_PRIORS=0 still means zeros.
PRIOR_MODE="${PRIOR_MODE:-saved}"
# 0 = zero/identity PF+RLSVI priors. Ignored when PRIOR_MODE=loo (those files
# are estimated). 1 = load ${ENV_VARIANT}/rl_priors.json (PRIOR_MODE=saved).
USE_ESTIMATED_PRIORS="${USE_ESTIMATED_PRIORS:-0}"
# 1 = Q action block includes C (default). 0 = A*[1, E, b̂, b̃] only.
ACTION_BLOCK_C="${ACTION_BLOCK_C:-1}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load python
mamba activate budgeted

export ADAPR_PROJECT_ROOT="$SLURM_SUBMIT_DIR"
export ADAPR_PARAMS_DIR="$ENV_VARIANT"
export ADAPR_EXPERIMENT_PARAMS_DIR="${SLURM_SUBMIT_DIR}/${ENV_VARIANT}"
export ADAPR_PRIOR_MODE="${PRIOR_MODE}"
export USE_ESTIMATED_PRIORS
export ACTION_BLOCK_C
export SAVE_MODE=compact

if [ "${PRIOR_MODE}" != "saved" ] && [ "${PRIOR_MODE}" != "loo" ]; then
  echo "PRIOR_MODE must be saved or loo (got ${PRIOR_MODE})" >&2
  exit 1
fi
if [ "${PRIOR_MODE}" = "loo" ]; then
  LOO_DIR="${ADAPR_EXPERIMENT_PARAMS_DIR}/loo_priors"
  if [ ! -d "${LOO_DIR}" ] || ! ls "${LOO_DIR}"/held_out_*.json >/dev/null 2>&1; then
    echo "PRIOR_MODE=loo requires leave-one-out prior files in ${LOO_DIR}." >&2
    echo "Generate them first:" >&2
    echo "  ADAPR_EST_PRIOR_PARAMS_DIR=${ADAPR_EXPERIMENT_PARAMS_DIR} python est_prior.py --loo" >&2
    exit 1
  fi
  if [ "${USE_ESTIMATED_PRIORS}" = "0" ]; then
    echo "WARNING: PRIOR_MODE=loo loads estimated per-user priors; USE_ESTIMATED_PRIORS=0 is ignored."
  fi
fi

# Keep result folders per environment (vanilla, ste0.5, ...).
# Override with RESULTS_ROOT=... ; if unset and ACTION_BLOCK_C=0, suffix
# _no_action_c so this batch does not mix with the default runs.
# PRIOR_MODE=loo gets _loo so it does not mix with saved/zero-prior batches.
if [ -z "${RESULTS_ROOT:-}" ]; then
  RESULTS_ROOT="${SLURM_SUBMIT_DIR}/results_${ENV_VARIANT#env_para_}"
  if [ "${ACTION_BLOCK_C}" = "0" ]; then
    RESULTS_ROOT="${RESULTS_ROOT}_no_action_c"
  fi
  if [ "${PRIOR_MODE}" = "loo" ]; then
    RESULTS_ROOT="${RESULTS_ROOT}_loo"
  fi
fi
export RESULTS_ROOT
# Job requests 1 CPU; stop NumPy/OpenBLAS from spawning extra threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "ENV_VARIANT=${ENV_VARIANT}"
echo "PARAMS_DIR=${ADAPR_EXPERIMENT_PARAMS_DIR}"
echo "PRIOR_MODE=${PRIOR_MODE}"
echo "USE_ESTIMATED_PRIORS=${USE_ESTIMATED_PRIORS}"
echo "ACTION_BLOCK_C=${ACTION_BLOCK_C}"
echo "NWEEK=${NWEEK}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "SAVE_MODE=${SAVE_MODE}"
echo "seed-idx=${SLURM_ARRAY_TASK_ID}"

EXTRA_ARGS=()
if [ "${ACTION_BLOCK_C}" = "0" ]; then
  EXTRA_ARGS+=(--no-action-c)
fi

python experiment.py \
  --seed-idx "$SLURM_ARRAY_TASK_ID" \
  --params-dir "$ADAPR_EXPERIMENT_PARAMS_DIR" \
  --prior-mode "$PRIOR_MODE" \
  --nweek "$NWEEK" \
  --save-mode compact \
  --results-root "$RESULTS_ROOT" \
  "${EXTRA_ARGS[@]}"
