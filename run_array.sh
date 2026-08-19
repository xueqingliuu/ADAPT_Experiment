#!/bin/bash
#SBATCH -J exp_seed_array
#SBATCH -p murphy
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 2-10:00:00
#SBATCH --mem=3G
#SBATCH --array=1-100%20
#SBATCH -o logs/exp_%A_%a.out
#SBATCH -e logs/exp_%A_%a.err

# Fitted environment folder under the repo root. Switch later, e.g.:
#   ENV_VARIANT=env_para_ste0.5 sbatch run_array.sh
ENV_VARIANT="${ENV_VARIANT:-env_para_vanilla}"
# 0 = zero/identity PF+RLSVI priors ("default_prior").
# 1 = load ${ENV_VARIANT}/rl_priors.json.
USE_ESTIMATED_PRIORS="${USE_ESTIMATED_PRIORS:-0}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load python
mamba activate budgeted

export ADAPR_PROJECT_ROOT="$SLURM_SUBMIT_DIR"
export ADAPR_PARAMS_DIR="$ENV_VARIANT"
export ADAPR_EXPERIMENT_PARAMS_DIR="${SLURM_SUBMIT_DIR}/${ENV_VARIANT}"
export USE_ESTIMATED_PRIORS
export SAVE_MODE=compact
# Keep result folders per environment (vanilla, ste0.5, ...).
export RESULTS_ROOT="${SLURM_SUBMIT_DIR}/results_${ENV_VARIANT#env_para_}"
# Job requests 1 CPU; stop NumPy/OpenBLAS from spawning extra threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "ENV_VARIANT=${ENV_VARIANT}"
echo "PARAMS_DIR=${ADAPR_EXPERIMENT_PARAMS_DIR}"
echo "USE_ESTIMATED_PRIORS=${USE_ESTIMATED_PRIORS}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "SAVE_MODE=${SAVE_MODE}"
echo "seed-idx=${SLURM_ARRAY_TASK_ID}"

python experiment.py \
  --seed-idx "$SLURM_ARRAY_TASK_ID" \
  --params-dir "$ADAPR_EXPERIMENT_PARAMS_DIR" \
  --save-mode compact
