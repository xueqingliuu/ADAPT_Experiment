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

echo "Running aggregate_results.py on $(hostname)"

python aggregate.py
