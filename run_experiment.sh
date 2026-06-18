#!/bin/bash
#SBATCH -J experiment_run
#SBATCH -p murphy
#SBATCH -c 4
#SBATCH -t 1-00:00:00
#SBATCH --mem=16GB
#SBATCH --array=1-100          # one seed per task (matches N_EXPERIMENTS=100)
#SBATCH -o experiment_%A_%a.out
#SBATCH -e experiment_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# conda/mamba in non-interactive batch shells
module load python
eval "$(conda shell.bash hook)"
mamba activate budgeted 2>/dev/null || conda activate budgeted

# thread limits (match -c 4)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# headless plotting + writable matplotlib cache
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.matplotlib"

# results parent dir shared by all array tasks (aggregate.py reads the same)
export RESULTS_ROOT="${SLURM_SUBMIT_DIR}/results_mtd_joint"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}  Task ID: ${SLURM_ARRAY_TASK_ID:-none}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"

# sanity check before a long run
test -f env_para_vanilla/user_ids.txt

python experiment.py
# SLURM_ARRAY_TASK_ID is picked up automatically as --seed-idx
