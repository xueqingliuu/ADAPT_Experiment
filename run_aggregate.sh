#!/bin/bash
#SBATCH -J aggregate_results
#SBATCH -p murphy
#SBATCH -c 1
#SBATCH -t 0-02:00:00
#SBATCH --mem=16GB
#SBATCH -o aggregate_%j.out
#SBATCH -e aggregate_%j.err

# Must match the ENV_VARIANT / ACTION_BLOCK_C used by run_array.sh.
ENV_VARIANT="${ENV_VARIANT:-env_para_vanilla}"
ACTION_BLOCK_C="${ACTION_BLOCK_C:-1}"

cd "$SLURM_SUBMIT_DIR"

module load python
mamba activate budgeted

export ADAPR_PROJECT_ROOT="$SLURM_SUBMIT_DIR"
export ADAPR_PARAMS_DIR="$ENV_VARIANT"
if [ -z "${RESULTS_ROOT:-}" ]; then
  RESULTS_ROOT="${SLURM_SUBMIT_DIR}/results_${ENV_VARIANT#env_para_}"
  if [ "${ACTION_BLOCK_C}" = "0" ]; then
    RESULTS_ROOT="${RESULTS_ROOT}_no_action_c"
  fi
fi
export RESULTS_ROOT
# Optional: restrict to one SLURM array batch (recommended).
# export SLURM_ARRAY_JOB_ID=<array_job_id>
# Or pass explicitly: python aggregate.py --array-job-id <id>
# To combine every run folder ever written under RESULTS_ROOT:
# python aggregate.py --all-runs

echo "Running aggregate.py on $(hostname)"
echo "ENV_VARIANT=${ENV_VARIANT}"
echo "ACTION_BLOCK_C=${ACTION_BLOCK_C}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-<unset; latest batch>}"

python aggregate.py
