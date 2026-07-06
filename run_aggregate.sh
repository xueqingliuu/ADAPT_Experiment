#!/bin/bash
#SBATCH -J aggregate_results
#SBATCH -p murphy
#SBATCH -c 1
#SBATCH -t 0-02:00:00
#SBATCH --mem=16GB
#SBATCH -o aggregate_%j.out
#SBATCH -e aggregate_%j.err

cd "$SLURM_SUBMIT_DIR"

module load python
mamba activate budgeted

export RESULTS_ROOT="${SLURM_SUBMIT_DIR}/results_vanilla"
# Optional: restrict to one SLURM array batch (recommended).
# export SLURM_ARRAY_JOB_ID=<array_job_id>
# Or pass explicitly: python aggregate.py --array-job-id <id>
# To combine every run folder ever written under RESULTS_ROOT:
# python aggregate.py --all-runs

echo "Running aggregate.py on $(hostname)"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-<unset; latest batch>}"

python aggregate.py
