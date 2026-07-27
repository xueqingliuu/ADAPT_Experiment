#!/bin/bash
#SBATCH -J ste_vanilla
#SBATCH -p murphy
#SBATCH -c 4
#SBATCH -t 2-00:00:00
#SBATCH --mem=32GB
#SBATCH --array=0-30
#SBATCH -o logs/ste_%A_%a.out
#SBATCH -e logs/ste_%A_%a.err

# Average user STE for the vanilla testbed (ste_vanilla.py).
#
# Single submission (train + eval for all 31 participants, then average user STE):
#   mkdir -p logs
#   sbatch run_ste.sh
#
# Array task 0 auto-submits a follow-on aggregate job that waits for all
# array tasks to finish (afterok on the array job id).
#
# Optional overrides (sbatch --export=...):
#   STE_PHASE=train|eval|all|aggregate   default all
#   STE_EXP=1  STE_NOISE=random  STE_N_TRAIN=5000  STE_N_EVAL=500  STE_N_STEPS=100000

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

module load python
eval "$(conda shell.bash hook)"
mamba activate budgeted 2>/dev/null || conda activate budgeted

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR:-$PWD}/.matplotlib"

STE_EXP="${STE_EXP:-1}"
STE_PHASE="${STE_PHASE:-all}"
STE_NOISE="${STE_NOISE:-random}"
STE_N_TRAIN="${STE_N_TRAIN:-5000}"
STE_N_EVAL="${STE_N_EVAL:-500}"
STE_N_STEPS="${STE_N_STEPS:-100000}"
USER_IDS="${STE_USER_IDS:-env_para_vanilla/user_ids.txt}"

mkdir -p logs

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}  Task ID: ${SLURM_ARRAY_TASK_ID:-none}"
echo "STE_PHASE=${STE_PHASE}  STE_EXP=${STE_EXP}  STE_NOISE=${STE_NOISE}"
echo "STE_N_TRAIN=${STE_N_TRAIN}  STE_N_EVAL=${STE_N_EVAL}  STE_N_STEPS=${STE_N_STEPS}"

test -f "${USER_IDS}"
test -f ste_vanilla.py

NUM_USERS=$(wc -l < "${USER_IDS}")
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" && "${SLURM_ARRAY_TASK_COUNT}" != "${NUM_USERS}" ]]; then
  echo "ERROR: SLURM array has ${SLURM_ARRAY_TASK_COUNT} tasks but ${USER_IDS} has ${NUM_USERS} rows." >&2
  echo "Update #SBATCH --array=0-$((NUM_USERS - 1)) in this script to match." >&2
  exit 1
fi

COMMON_ARGS=(
  --exp "${STE_EXP}"
  --user-ids "${USER_IDS}"
  --noise "${STE_NOISE}"
)

run_train() {
  local jobid="$1"
  python ste_vanilla.py train "${jobid}" \
    "${COMMON_ARGS[@]}" \
    --n-train-episodes "${STE_N_TRAIN}" \
    --n-steps "${STE_N_STEPS}"
}

run_eval() {
  local jobid="$1"
  python ste_vanilla.py eval "${jobid}" \
    "${COMMON_ARGS[@]}" \
    --n-test "${STE_N_EVAL}"
}

run_aggregate() {
  python ste_vanilla.py aggregate \
    --exp "${STE_EXP}" \
    --user-ids "${USER_IDS}"
}

submit_aggregate_job() {
  local agg_id
  agg_id=$(sbatch --parsable \
    --dependency="afterok:${SLURM_ARRAY_JOB_ID}" \
    --job-name="ste_aggregate_exp${STE_EXP}" \
    -p "${SLURM_JOB_PARTITION:-murphy}" \
    -c 1 \
    -t 0-01:00:00 \
    --mem=8G \
    --array=0 \
    -o "logs/ste_aggregate_%j.out" \
    -e "logs/ste_aggregate_%j.err" \
    --export=ALL,STE_PHASE=aggregate \
    "${SLURM_SUBMIT_DIR}/run_ste.sh")
  echo "Submitted aggregate job ${agg_id} (runs after array ${SLURM_ARRAY_JOB_ID})"
}

case "${STE_PHASE}" in
  aggregate)
    run_aggregate
    ;;
  train|eval|all)
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      echo "ERROR: STE_PHASE=${STE_PHASE} requires a SLURM array task id (0-$((NUM_USERS - 1)))." >&2
      echo "Submit with: sbatch run_ste.sh" >&2
      exit 1
    fi
    if [[ "${STE_PHASE}" == "all" && "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
      submit_aggregate_job
    fi
    JOBID="${SLURM_ARRAY_TASK_ID}"
    echo "Participant jobid=${JOBID} (user_ids.txt row $((JOBID + 1)))"
    if [[ "${STE_PHASE}" == "train" || "${STE_PHASE}" == "all" ]]; then
      run_train "${JOBID}"
    fi
    if [[ "${STE_PHASE}" == "eval" || "${STE_PHASE}" == "all" ]]; then
      run_eval "${JOBID}"
    fi
    ;;
  *)
    echo "ERROR: unknown STE_PHASE=${STE_PHASE} (use train, eval, all, or aggregate)" >&2
    exit 1
    ;;
esac
