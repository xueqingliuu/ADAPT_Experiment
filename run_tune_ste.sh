#!/bin/bash
#SBATCH -J tune_ste
#SBATCH -p murphy
#SBATCH -c 16
#SBATCH -t 1-00:00:00
#SBATCH --mem=32GB
#SBATCH -o logs/tune_ste_%j.out
#SBATCH -e logs/tune_ste_%j.err
#
# Tune the testbed's effect size to target STE values (tune_ste.py).
#
# This is a single job, not an array: the root search over the knob is
# inherently sequential, and tune_ste.py parallelizes across participants
# inside each evaluation (-c controls that width).
#
#   mkdir -p logs
#   sbatch run_tune_ste.sh                                   # calibrate 0.2/0.5/0.8, then
#                                                            # auto-submit true-STE DQN jobs
#   sbatch --export=ALL,TUNE_PHASE=diagnose run_tune_ste.sh  # E_w + CAE loop gains, ~seconds
#   sbatch --export=ALL,TUNE_PHASE=eval run_tune_ste.sh      # STE of the untouched fit
#   sbatch --export=ALL,TUNE_PHASE=scan run_tune_ste.sh      # STE vs knob value
#   TUNE_PHASE=validate bash run_tune_ste.sh                 # login node: re-submit the
#                                                            # confirmation runs by hand
#
# Overrides (sbatch --export=ALL,VAR=value,...):
#   TUNE_PHASE        diagnose|eval|scan|calibrate|validate    default calibrate
#   TUNE_PARAMS_DIR   source fit to rescale                    default env_para_vanilla
#   TUNE_KNOB         action|benefit|burden|my_to_y|foursc_to_y|...  default action
#   TUNE_TARGETS      target mean STE values                   default "0.2 0.5 0.8"
#   TUNE_KAPPAS       scan grid                                default "0.25 0.5 1 2 4"
#   TUNE_EPISODES     paired episodes per arm                  default 100
#   TUNE_POLICY_GRID  Bernoulli suggestion rates               default "0.5 1.0"
#   TUNE_DQN_EXP      ste_vanilla --exp whose DQNs join as an extra arm, "" to skip
#                     (auto-skipped if the checkpoints are absent)  default 1
#   TUNE_NOISE        ar1|random|sequential                    default ar1
#   TUNE_OUT_PREFIX   tuned dirs are <prefix><target>          default env_para_ste
#   TUNE_VALIDATE     1 to submit true-STE jobs after calibrate (default 1)
#   TUNE_REQUIRE_STABLE  1 to refuse unstable folders (default 1)
#   TUNE_TOL  TUNE_MAX_ITER  TUNE_SEED  TUNE_PROXY_TO_TRUE  TUNE_JOBS
#
# Calibration writes env_para_ste0.2/, env_para_ste0.5/, env_para_ste0.8/ only
# when every participant's E_w and CAE loops are stable and the proxy STE is
# within TUNE_TOL of the target. Each folder is a drop-in parameter set plus
# ste_tuning.json.
# On success the script then submits one run_ste.sh array per folder
# (STE_EXP=ste0.2 etc.) so the true DQN STE is measured automatically.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# An explicit interpreter (TUNE_PYTHON=path/to/python) skips the cluster module
# stack, so the same script is runnable off-cluster.
if [[ -z "${TUNE_PYTHON:-}" ]]; then
  module load python
  eval "$(conda shell.bash hook)"
  mamba activate budgeted 2>/dev/null || conda activate budgeted
fi
PY="${TUNE_PYTHON:-python}"

# One BLAS thread per process: the parallelism is across participants, and
# oversubscribing threads inside each worker makes the whole sweep slower.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR:-$PWD}/.matplotlib"

TUNE_PHASE="${TUNE_PHASE:-calibrate}"
TUNE_PARAMS_DIR="${TUNE_PARAMS_DIR:-env_para_vanilla}"
TUNE_KNOB="${TUNE_KNOB:-action}"
TUNE_TARGETS="${TUNE_TARGETS:-0.2 0.5 0.8}"
TUNE_KAPPAS="${TUNE_KAPPAS:-0.25 0.5 1 2 4}"
TUNE_EPISODES="${TUNE_EPISODES:-100}"
TUNE_POLICY_GRID="${TUNE_POLICY_GRID:-0.5 1.0}"
TUNE_DQN_EXP="${TUNE_DQN_EXP:-1}"
TUNE_NOISE="${TUNE_NOISE:-ar1}"
TUNE_OUT_PREFIX="${TUNE_OUT_PREFIX:-env_para_ste}"
TUNE_TOL="${TUNE_TOL:-0.02}"
TUNE_MAX_ITER="${TUNE_MAX_ITER:-10}"
TUNE_SEED="${TUNE_SEED:-20260814}"
TUNE_PROXY_TO_TRUE="${TUNE_PROXY_TO_TRUE:-1.0}"
TUNE_JOBS="${TUNE_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
TUNE_VALIDATE="${TUNE_VALIDATE:-1}"
TUNE_REQUIRE_STABLE="${TUNE_REQUIRE_STABLE:-1}"

mkdir -p logs

test -f tune_ste.py
test -d "${TUNE_PARAMS_DIR}"
USER_IDS="${TUNE_PARAMS_DIR}/user_ids.txt"
test -f "${USER_IDS}"
NUM_USERS=$(( $(wc -l < "${USER_IDS}") ))

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "TUNE_PHASE=${TUNE_PHASE}  TUNE_PARAMS_DIR=${TUNE_PARAMS_DIR}  (${NUM_USERS} participants)"
echo "TUNE_KNOB=${TUNE_KNOB}  TUNE_EPISODES=${TUNE_EPISODES}  TUNE_JOBS=${TUNE_JOBS}"

# The DQN arm is optional. Requiring every checkpoint keeps the comparison
# across participants apples-to-apples; a partial set silently mixes arms.
DQN_ARGS=()
if [[ -n "${TUNE_DQN_EXP}" ]]; then
  DQN_DIR="d3rlpy_logs/ste_exp_${TUNE_DQN_EXP}"
  shopt -s nullglob
  CKPTS=( "${DQN_DIR}"/user*_model.d3 )
  shopt -u nullglob
  N_CKPT="${#CKPTS[@]}"
  if [[ "${N_CKPT}" -ge "${NUM_USERS}" ]]; then
    DQN_ARGS=(--dqn-exp "${TUNE_DQN_EXP}")
    echo "DQN arm: ${N_CKPT} checkpoints in ${DQN_DIR}"
  else
    echo "DQN arm: OFF (${N_CKPT}/${NUM_USERS} checkpoints in ${DQN_DIR});" \
         "run run_ste.sh with STE_EXP=${TUNE_DQN_EXP} first, or set TUNE_DQN_EXP=''" >&2
  fi
fi

COMMON_ARGS=(
  --params-dir "${TUNE_PARAMS_DIR}"
  --episodes "${TUNE_EPISODES}"
  --seed "${TUNE_SEED}"
  --noise "${TUNE_NOISE}"
  --policy-grid ${TUNE_POLICY_GRID}
  --jobs "${TUNE_JOBS}"
  --proxy-to-true "${TUNE_PROXY_TO_TRUE}"
  "${DQN_ARGS[@]+"${DQN_ARGS[@]}"}"
)

# Each tuned parameter set gets its own STE_EXP so its DQN checkpoints and
# results_ste/ outputs never overwrite the baseline run's. Only folders whose
# ste_tuning.json records a successful stable calibration are submitted.
folder_is_stable() {
  local dir="$1"
  [[ -f "${dir}/user_ids.txt" && -f "${dir}/ste_tuning.json" ]] || return 1
  "${PY}" - "${dir}/ste_tuning.json" <<'PY'
import json, sys
rep = json.load(open(sys.argv[1], encoding="utf-8"))
sys.exit(0 if rep.get("stable") is True else 1)
PY
}

submit_validation_jobs() {
  local submitted=0
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is not available; cannot submit true-STE jobs." >&2
    echo "On a login node: TUNE_PHASE=validate bash run_tune_ste.sh" >&2
    return 1
  fi
  for target in ${TUNE_TARGETS}; do
    local dir="${TUNE_OUT_PREFIX}${target}"
    if ! folder_is_stable "${dir}"; then
      echo "SKIP ${dir}: missing, or ste_tuning.json does not have stable=true." >&2
      continue
    fi
    local jid
    jid=$(sbatch --parsable \
      --job-name="ste_${target}" \
      --array="0-$(( $(wc -l < "${dir}/user_ids.txt") - 1 ))" \
      --export="ALL,ADAPR_PARAMS_DIR=${dir},STE_EXP=ste${target},STE_USER_IDS=${dir}/user_ids.txt,STE_PHASE=all" \
      "${SLURM_SUBMIT_DIR:-$PWD}/run_ste.sh")
    echo "Submitted ${jid}: full DQN train+eval in ${dir} (results_ste/expste${target})"
    submitted=$((submitted + 1))
  done
  if [[ "${submitted}" -eq 0 ]]; then
    echo "ERROR: nothing to validate." >&2
    return 1
  fi
  return 0
}

case "${TUNE_PHASE}" in
  diagnose)
    "${PY}" tune_ste.py diagnose --params-dir "${TUNE_PARAMS_DIR}"
    ;;
  eval)
    "${PY}" tune_ste.py eval "${COMMON_ARGS[@]}" \
      --report "logs/tune_ste_eval_${SLURM_JOB_ID:-local}.json"
    ;;
  scan)
    "${PY}" tune_ste.py scan "${COMMON_ARGS[@]}" \
      --knob "${TUNE_KNOB}" \
      --kappas ${TUNE_KAPPAS} \
      --report "logs/tune_ste_scan_${TUNE_KNOB}_${SLURM_JOB_ID:-local}.json"
    ;;
  calibrate)
    STABLE_FLAG=(--require-stable)
    if [[ "${TUNE_REQUIRE_STABLE}" == "0" ]]; then
      STABLE_FLAG=(--no-require-stable)
    fi
    set +e
    "${PY}" tune_ste.py calibrate "${COMMON_ARGS[@]}" \
      --knob "${TUNE_KNOB}" \
      --targets ${TUNE_TARGETS} \
      --out-prefix "${TUNE_OUT_PREFIX}" \
      --tol "${TUNE_TOL}" \
      --max-iter "${TUNE_MAX_ITER}" \
      "${STABLE_FLAG[@]}"
    cal_status=$?
    set -e
    if [[ "${TUNE_VALIDATE}" != "1" ]]; then
      echo "TUNE_VALIDATE=0: skipping true-STE jobs. Re-run with TUNE_PHASE=validate."
      exit "${cal_status}"
    fi
    if ! command -v sbatch >/dev/null 2>&1; then
      echo "WARNING: sbatch is not available here; tuned folders (if any) are written."
      echo "From a login node: TUNE_PHASE=validate bash run_tune_ste.sh"
      exit "${cal_status}"
    fi
    if [[ "${cal_status}" -eq 0 || "${cal_status}" -eq 2 ]]; then
      echo "Submitting true-STE DQN jobs for stable folders."
      submit_validation_jobs || {
        # Nothing to submit is fatal only when calibration claimed full success.
        [[ "${cal_status}" -eq 0 ]] && exit 1
      }
    else
      echo "Calibration failed (exit ${cal_status}); not submitting true-STE jobs." >&2
    fi
    exit "${cal_status}"
    ;;
  validate)
    submit_validation_jobs
    ;;
  *)
    echo "ERROR: unknown TUNE_PHASE=${TUNE_PHASE} (use diagnose, eval, scan, calibrate, or validate)" >&2
    exit 1
    ;;
esac
