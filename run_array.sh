#!/bin/bash
#SBATCH -J exp_seed_array
#SBATCH -p murphy
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 2-10:00:00
#SBATCH --mem=1G
#SBATCH --array=1-100%6
#SBATCH -o logs/exp_%A_%a.out
#SBATCH -e logs/exp_%A_%a.err

cd "$SLURM_SUBMIT_DIR"
module load python
mamba activate budgeted
export ADAPR_PROJECT_ROOT="$SLURM_SUBMIT_DIR"

# Pass array index to Python as seed index
python experiment.py
