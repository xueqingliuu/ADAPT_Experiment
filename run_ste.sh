#!/bin/bash
#SBATCH -J ste_vanilla
#SBATCH -p murphy
#SBATCH -c 4
#SBATCH -t 2-00:00:00
#SBATCH --mem=32GB
#SBATCH --array=0-46
#SBATCH -o logs/ste_%A_%a.out
#SBATCH -e logs/ste_%A_%a.err

# Average user STE for the vanilla testbed (ste_vanilla.py).
#
# Single submission (train + eval for all 47 participants, then average user STE):
#   mkdir -p logs
#   sbatch --array=0-46 run_ste.sh
#
# #SBATCH cannot see user_ids.txt at submit time. A hand submit that omits
# --array uses 0-46 above. The script then errors if that range is not
# 0..(N-1) for ${USER_IDS}, unless STE_ALLOW_PARTIAL=1.
#
# Array task 0 auto-submits a follow-on aggregate job that waits for all
# array tasks to finish (afterok on the array job id).
#
# Optional overrides (sbatch --export=...):
#   STE_PHASE=train|eval|all|aggregate   default all
#   STE_EXP=1  STE_NOISE=ar1  STE_N_TRAIN=5000  STE_N_EVAL=500  STE_N_STEPS=100000
#   STE_ALLOW_PARTIAL=1   allow an array other than 0..(N-1)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# Resolve the participant list before the cluster modules so a bad array
# exits before any training.
STE_PHASE="${STE_PHASE:-all}"
ENV_VARIANT="${ENV_VARIANT:-${ADAPR_PARAMS_DIR:-env_para_vanilla}}"
export ADAPR_PARAMS_DIR="${ADAPR_PARAMS_DIR:-$ENV_VARIANT}"
USER_IDS="${STE_USER_IDS:-${ADAPR_PARAMS_DIR}/user_ids.txt}"
if [[ ! -f "${USER_IDS}" ]]; then
  echo "ERROR: missing user id list ${USER_IDS}" >&2
  exit 1
fi
NUM_USERS=$(( $(wc -l < "${USER_IDS}") ))
EXPECTED_MAX=$((NUM_USERS - 1))
if [[ "${STE_PHASE}" != "aggregate" ]]; then
  array_min="${SLURM_ARRAY_TASK_MIN:-}"
  array_max="${SLURM_ARRAY_TASK_MAX:-}"
  array_count="${SLURM_ARRAY_TASK_COUNT:-}"
  if [[ -n "${array_min}" || -n "${array_max}" || -n "${array_count}" ]]; then
    mismatch=0
    if [[ "${array_min}" != "0" || "${array_max}" != "${EXPECTED_MAX}" ]]; then
      mismatch=1
    fi
    if [[ -n "${array_count}" && "${array_count}" != "${NUM_USERS}" ]]; then
      mismatch=1
    fi
    if [[ "${mismatch}" -eq 1 ]]; then
      if [[ "${STE_ALLOW_PARTIAL:-0}" == "1" ]]; then
        echo "WARNING: partial SLURM array ${array_min:-?}-${array_max:-?} (count ${array_count:-?}) for ${USER_IDS} (${NUM_USERS} lines; need --array=0-${EXPECTED_MAX}). STE_ALLOW_PARTIAL=1." >&2
      else
        echo "ERROR: SLURM array ${array_min:-?}-${array_max:-?} (count ${array_count:-?}) does not match ${USER_IDS} (${NUM_USERS} lines). Need --array=0-${EXPECTED_MAX} (min index 0, max index ${EXPECTED_MAX})." >&2
        echo "Set STE_ALLOW_PARTIAL=1 for a deliberate partial run." >&2
        exit 1
      fi
    else
      echo "STE array 0-${EXPECTED_MAX} matches ${NUM_USERS} lines in ${USER_IDS}"
    fi
  fi
fi
if [[ "${STE_ARRAY_CHECK_ONLY:-0}" == "1" ]]; then
  exit 0
fi

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
STE_NOISE="${STE_NOISE:-ar1}"
STE_N_TRAIN="${STE_N_TRAIN:-5000}"
STE_N_EVAL="${STE_N_EVAL:-500}"
STE_N_STEPS="${STE_N_STEPS:-100000}"
# STE uses the testbed's own activity and ψ. Override with STE_ACTIVITY=off
# for the diagnostic no-gap STE.
ADAPR_ACTIVITY="${ADAPR_ACTIVITY:-on}"
ADAPR_STAYER_FRACTION="${ADAPR_STAYER_FRACTION:-0.5}"
ADAPR_MISSPEC="${ADAPR_MISSPEC:-none}"
STE_ACTIVITY="${STE_ACTIVITY:-}"
export ADAPR_ACTIVITY ADAPR_STAYER_FRACTION ADAPR_MISSPEC
STE_EFFECTIVE="${STE_EFFECTIVE:-0}"

mkdir -p logs

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}  Task ID: ${SLURM_ARRAY_TASK_ID:-none}"
echo "STE_PHASE=${STE_PHASE}  STE_EXP=${STE_EXP}  STE_NOISE=${STE_NOISE}"
echo "STE_N_TRAIN=${STE_N_TRAIN}  STE_N_EVAL=${STE_N_EVAL}  STE_N_STEPS=${STE_N_STEPS}"
echo "ADAPR_PARAMS_DIR=${ADAPR_PARAMS_DIR}  ADAPR_ACTIVITY=${ADAPR_ACTIVITY}  ADAPR_MISSPEC=${ADAPR_MISSPEC}"
echo "ADAPR_STAYER_FRACTION=${ADAPR_STAYER_FRACTION}  STE_ACTIVITY=${STE_ACTIVITY:-<testbed>}  STE_EFFECTIVE=${STE_EFFECTIVE}"

test -f ste_vanilla.py

COMMON_ARGS=(
  --exp "${STE_EXP}"
  --user-ids "${USER_IDS}"
  --noise "${STE_NOISE}"
  --params-dir "${ADAPR_PARAMS_DIR}"
  --activity "${ADAPR_ACTIVITY}"
  --stayer-fraction "${ADAPR_STAYER_FRACTION}"
  --misspec "${ADAPR_MISSPEC}"
)
if [[ -n "${STE_ACTIVITY}" ]]; then
  COMMON_ARGS+=(--ste-activity "${STE_ACTIVITY}")
fi
EFFECTIVE_ARGS=()
if [[ "${STE_EFFECTIVE}" == "1" ]]; then
  EFFECTIVE_ARGS=(--effective-ste)
fi

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
    --n-test "${STE_N_EVAL}" \
    "${EFFECTIVE_ARGS[@]+"${EFFECTIVE_ARGS[@]}"}"
}

run_aggregate() {
  python ste_vanilla.py aggregate \
    --exp "${STE_EXP}" \
    --user-ids "${USER_IDS}" \
    --params-dir "${ADAPR_PARAMS_DIR}" \
    --activity "${ADAPR_ACTIVITY}" \
    --stayer-fraction "${ADAPR_STAYER_FRACTION}" \
    --misspec "${ADAPR_MISSPEC}" \
    ${STE_ACTIVITY:+--ste-activity "${STE_ACTIVITY}"} \
    "${EFFECTIVE_ARGS[@]+"${EFFECTIVE_ARGS[@]}"}"
}

submit_aggregate_job() {
  local agg_id
  # One task, not one per user. STE_PHASE=aggregate skips the 0..(N-1) check.
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
