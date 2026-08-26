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
#   sbatch run_tune_ste.sh                                   # action knob → 0.2/0.5/0.8, then
#                                                            # auto-submit confirmation DiscreteCQL jobs
#   sbatch --export=ALL,TUNE_KNOB=burden_shift,TUNE_TARGETS="0.2 0.5",TUNE_OUT_SUFFIX=_burden_shift \
#          run_tune_ste.sh                                   # A→ME down, E→MY up; writes
#                                                            # env_para_ste0.2_burden_shift etc.
#   sbatch --export=ALL,TUNE_PHASE=apply,TUNE_KNOB=burden_shift,TUNE_KAPPA=0.4 \
#          run_tune_ste.sh                                   # env_para_burden_shift_large/
#   sbatch --export=ALL,TUNE_PARAMS_DIR=env_para_burden_shift_large,\
#TUNE_KNOB=benefit_foursc,TUNE_TARGETS="0.5 0.8",TUNE_OUT_SUFFIX=_bs_large_bf \
#          run_tune_ste.sh                                   # A→MY + fourSC→Y on κ_b=0.4
#   # foursc_to_y_shift alone cannot hit 0.5/0.8 (job 41048118).
#   # Do NOT TUNE_KNOB=burden_shift TUNE_TARGETS=0.2: floor STE≈0.27 (40926808).
#   sbatch --export=ALL,TUNE_PHASE=diagnose run_tune_ste.sh  # E_w + CAE loop gains, ~seconds
#   sbatch --export=ALL,TUNE_PHASE=eval run_tune_ste.sh      # proxy STE of the untouched fit
#   sbatch --export=ALL,TUNE_PHASE=scan run_tune_ste.sh      # STE vs knob value
#   TUNE_PHASE=validate bash run_tune_ste.sh                 # login node: re-submit the
#                                                            # confirmation runs by hand
#
# Three-variant protocol in one job (vanilla / fatigue→0.2 / benefit→0.5,0.8):
#   sbatch --export=ALL,TUNE_PHASE=protocol run_tune_ste.sh
#
# Overrides (sbatch --export=ALL,VAR=value,...):
#   TUNE_PHASE        diagnose|eval|scan|apply|calibrate|protocol|validate  default calibrate
#   TUNE_PARAMS_DIR   source fit to rescale                    default env_para_vanilla
#   TUNE_KNOB         fatigue|action|benefit|benefit_foursc|... (burden_shift = fatigue alias)
#   TUNE_FATIGUE_TARGET   protocol stage-1 target              default 0.2
#   TUNE_BENEFIT_TARGETS  protocol stage-2 targets             default "0.5 0.8"
#   TUNE_OVERWRITE    1 to pass --overwrite (needed when replacing folders
#                     written under an older knob name/semantics)  default 0
#   TUNE_TARGETS      target mean STE values                   default "0.2 0.5 0.8"
#   TUNE_KAPPA        apply-phase knob value                   default 0.4
#   TUNE_OUT_DIR      apply-phase destination                  default
#                     env_para_burden_shift_large (burden_shift) or
#                     env_para_<knob>_<kappa>
#   TUNE_APPLY_EVAL   1 to run proxy STE after apply           default 0
#   TUNE_KAPPA0       calibrate search start                   default 1, or 0.2 /
#                     0.05 / 2 for burden_shift / foursc_to_y_shift /
#                     benefit_foursc
#   TUNE_KAPPAS       scan grid                                default "0.25 0.5 1 2 4"
#   TUNE_EPISODES     paired episodes per arm                  default 100
#   TUNE_POLICY_GRID  Bernoulli suggestion rates               default "0.5 1.0"
#   TUNE_DQN_EXP      ste_vanilla --exp whose DiscreteCQL policies join as an
#                     extra arm, "" to skip. Vanilla CQL is exp 5 (21-d frozen
#                     STE map; exp 1/2 are old DQN).  default 5
#   TUNE_NOISE        ar1|random|sequential                    default ar1
#   TUNE_OUT_PREFIX   tuned dirs are <prefix><target><suffix>  default env_para_ste
#   TUNE_OUT_SUFFIX   e.g. _burden → env_para_ste0.2_burden    default ""
#   TUNE_VALIDATE     1 to submit confirmation DiscreteCQL jobs after calibrate (default 1)
#   TUNE_REQUIRE_STABLE  1 to refuse unstable folders (default 1)
#   TUNE_TOL  TUNE_MAX_ITER  TUNE_SEED  TUNE_JOBS
#   TUNE_PROXY_TO_TRUE  multiply proxy STE after max/truncation (default 1.0;
#                     not a DiscreteCQL calibration factor unless you set one)
#   TUNE_SCRATCH      working dir for candidate params     default
#                     .ste_tune_scratch_<knob>_<jobid>
#
# Calibration writes env_para_ste0.2/, env_para_ste0.5/, env_para_ste0.8/ only
# when every participant's E_w and CAE loops are stable and the *proxy* STE
# (Bernoulli grid ± transferred CQL, same-sample max, truncated at 0) is
# within TUNE_TOL of the target. That is not ste_vanilla.aggregate_ste.
# Each folder is a drop-in parameter set plus ste_tuning.json.
# On success the script then submits one run_ste.sh array per folder
# (STE_EXP=ste0.2 etc.) so DiscreteCQL can measure the confirmation STE.

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
TUNE_FATIGUE_TARGET="${TUNE_FATIGUE_TARGET:-0.2}"
TUNE_BENEFIT_TARGETS="${TUNE_BENEFIT_TARGETS:-0.5 0.8}"
TUNE_OVERWRITE="${TUNE_OVERWRITE:-0}"
if [[ "${TUNE_KNOB}" == "burden_shift" || "${TUNE_KNOB}" == "fatigue" ]]; then
  TUNE_KAPPA0="${TUNE_KAPPA0:-0.2}"
elif [[ "${TUNE_KNOB}" == "foursc_to_y_shift" ]]; then
  TUNE_KAPPA0="${TUNE_KAPPA0:-0.05}"
elif [[ "${TUNE_KNOB}" == "benefit_foursc" ]]; then
  TUNE_KAPPA0="${TUNE_KAPPA0:-2.0}"
else
  TUNE_KAPPA0="${TUNE_KAPPA0:-1.0}"
fi
TUNE_KAPPAS="${TUNE_KAPPAS:-0.25 0.5 1 2 4}"
TUNE_KAPPA="${TUNE_KAPPA:-0.4}"
TUNE_APPLY_EVAL="${TUNE_APPLY_EVAL:-0}"
if [[ "${TUNE_KNOB}" == "burden_shift" || "${TUNE_KNOB}" == "fatigue" ]]; then
  TUNE_OUT_DIR="${TUNE_OUT_DIR:-env_para_burden_shift_large}"
else
  TUNE_OUT_DIR="${TUNE_OUT_DIR:-env_para_${TUNE_KNOB}_${TUNE_KAPPA}}"
fi
TUNE_EPISODES="${TUNE_EPISODES:-100}"
TUNE_POLICY_GRID="${TUNE_POLICY_GRID:-0.5 1.0}"
TUNE_DQN_EXP="${TUNE_DQN_EXP:-5}"
TUNE_NOISE="${TUNE_NOISE:-ar1}"
TUNE_OUT_PREFIX="${TUNE_OUT_PREFIX:-env_para_ste}"
TUNE_OUT_SUFFIX="${TUNE_OUT_SUFFIX:-}"
TUNE_TOL="${TUNE_TOL:-0.02}"
TUNE_MAX_ITER="${TUNE_MAX_ITER:-10}"
TUNE_SEED="${TUNE_SEED:-20260814}"
TUNE_PROXY_TO_TRUE="${TUNE_PROXY_TO_TRUE:-1.0}"
TUNE_JOBS="${TUNE_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
TUNE_VALIDATE="${TUNE_VALIDATE:-1}"
TUNE_REQUIRE_STABLE="${TUNE_REQUIRE_STABLE:-1}"
TUNE_SCRATCH="${TUNE_SCRATCH:-.ste_tune_scratch_${TUNE_KNOB}_${SLURM_JOB_ID:-$$}}"

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
echo "TUNE_SCRATCH=${TUNE_SCRATCH}  TUNE_OUT_PREFIX=${TUNE_OUT_PREFIX}  TUNE_OUT_SUFFIX=${TUNE_OUT_SUFFIX:-<none>}"
if [[ "${TUNE_PHASE}" == "apply" ]]; then
  echo "TUNE_KAPPA=${TUNE_KAPPA}  TUNE_OUT_DIR=${TUNE_OUT_DIR}  TUNE_APPLY_EVAL=${TUNE_APPLY_EVAL}"
fi

# The DiscreteCQL arm is optional. Requiring every checkpoint keeps the
# comparison across participants apples-to-apples; a partial set silently
# mixes arms. Presence of *.d3 files is not enough: older DQN checkpoints
# share that suffix but have algo=null, and tune_ste.py would then abort.
dqn_ckpt_status() {
  local dir="$1"
  local uids="$2"
  "${PY}" - "${dir}" "${uids}" <<'PY'
import json, sys
from pathlib import Path
from ste_vanilla import STE_ALGO, STE_OBS_DIM
model_dir = Path(sys.argv[1])
uids = [int(v) for v in Path(sys.argv[2]).read_text().split()]
for uid in uids:
    model = model_dir / f"user{uid}_model.d3"
    meta_path = model.with_name(f"{model.name}.meta.json")
    if not model.is_file() or not meta_path.is_file():
        print("MISSING")
        sys.exit(0)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("algo") != STE_ALGO:
        print("ALGO")
        sys.exit(0)
    if meta.get("state_dim") != STE_OBS_DIM:
        print(f"DIM {meta.get('state_dim')} {STE_OBS_DIM}")
        sys.exit(0)
print("OK")
PY
}

try_dqn_exp() {
  local exp="$1"
  local dir="d3rlpy_logs/ste_exp_${exp}"
  local n status
  shopt -s nullglob
  local ckpts=( "${dir}"/user*_model.d3 )
  shopt -u nullglob
  n="${#ckpts[@]}"
  if [[ "${n}" -lt "${NUM_USERS}" ]]; then
    return 1
  fi
  status="$(dqn_ckpt_status "${dir}" "${USER_IDS}")"
  if [[ "${status}" == "OK" ]]; then
    DQN_ARGS=(--dqn-exp "${exp}")
    echo "DiscreteCQL arm: ${n} checkpoints in ${dir} (state_dim matches STE_OBS_DIM)"
    return 0
  fi
  if [[ "${status}" == DIM* ]]; then
    echo "DiscreteCQL in ${dir} cannot be rolled: ${status}." \
         "Need state_dim matching the frozen 21-d STE map (exp 5)." >&2
  elif [[ "${status}" == "ALGO" ]]; then
    echo "DiscreteCQL arm: ${dir} is not discrete_cql (old DQN)." >&2
  fi
  return 1
}

DQN_ARGS=()
if [[ -n "${TUNE_DQN_EXP}" ]]; then
  if try_dqn_exp "${TUNE_DQN_EXP}"; then
    :
  else
    echo "Searching vanilla CQL folders 5, 4, 3." >&2
    used=""
    for fallback in 5 4 3; do
      if [[ "${fallback}" == "${TUNE_DQN_EXP}" ]]; then
        continue
      fi
      if try_dqn_exp "${fallback}"; then
        echo "Using DiscreteCQL from d3rlpy_logs/ste_exp_${fallback}." >&2
        used="${fallback}"
        break
      fi
    done
    if [[ -z "${used}" ]]; then
      echo "DiscreteCQL arm: OFF. Proxy STE uses the Bernoulli grid only." \
           "Point TUNE_DQN_EXP at a discrete_cql folder (vanilla CQL is 5)." >&2
    fi
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

submit_one_validation() {
  local dir="$1"
  if ! folder_is_stable "${dir}"; then
    echo "SKIP ${dir}: missing, or ste_tuning.json does not have stable=true." >&2
    return 1
  fi
  local jid exp_name
  # STE_EXP is the folder name with a leading ``env_para_`` stripped, so
  # env_para_ste0.8 → ste0.8 and env_para_burden_shift_large → burden_shift_large.
  exp_name="$(basename "${dir}")"
  exp_name="${exp_name#env_para_}"
  jid=$(sbatch --parsable \
    --job-name="ste_${exp_name}" \
    --array="0-$(( $(wc -l < "${dir}/user_ids.txt") - 1 ))" \
    --export="ALL,ADAPR_PARAMS_DIR=${dir},STE_EXP=${exp_name},STE_USER_IDS=${dir}/user_ids.txt,STE_PHASE=all" \
    "${SLURM_SUBMIT_DIR:-$PWD}/run_ste.sh")
  echo "Submitted ${jid}: full DQN train+eval in ${dir} (results_ste/exp${exp_name})"
}

# With no args, validate each calibrate output prefix+target+suffix.
# Pass folder paths (e.g. after apply) to validate those instead.
submit_validation_jobs() {
  local submitted=0 dir
  local -a dirs=()
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is not available; cannot submit confirmation DiscreteCQL STE jobs." >&2
    echo "On a login node: TUNE_PHASE=validate bash run_tune_ste.sh" >&2
    return 1
  fi
  if [[ "$#" -gt 0 ]]; then
    dirs=("$@")
  else
    for target in ${TUNE_TARGETS}; do
      dirs+=("${TUNE_OUT_PREFIX}${target}${TUNE_OUT_SUFFIX}")
    done
  fi
  for dir in "${dirs[@]}"; do
    if submit_one_validation "${dir}"; then
      submitted=$((submitted + 1))
    fi
  done
  if [[ "${submitted}" -eq 0 ]]; then
    echo "ERROR: nothing to validate." >&2
    return 1
  fi
  return 0
}

OVERWRITE_FLAG=()
if [[ "${TUNE_OVERWRITE}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

case "${TUNE_PHASE}" in
  protocol)
    STABLE_FLAG=(--require-stable)
    if [[ "${TUNE_REQUIRE_STABLE}" == "0" ]]; then
      STABLE_FLAG=(--no-require-stable)
    fi
    SUFFIX_ARGS=()
    if [[ -n "${TUNE_OUT_SUFFIX}" ]]; then
      SUFFIX_ARGS=(--out-suffix "${TUNE_OUT_SUFFIX}")
    fi
    set +e
    "${PY}" tune_ste.py protocol "${COMMON_ARGS[@]}" \
      --fatigue-target "${TUNE_FATIGUE_TARGET}" \
      --benefit-targets ${TUNE_BENEFIT_TARGETS} \
      --out-prefix "${TUNE_OUT_PREFIX}" \
      "${SUFFIX_ARGS[@]+"${SUFFIX_ARGS[@]}"}" \
      --scratch "${TUNE_SCRATCH}" \
      --tol "${TUNE_TOL}" \
      --max-iter "${TUNE_MAX_ITER}" \
      "${STABLE_FLAG[@]}" \
      "${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"}"
    proto_status=$?
    set -e
    if [[ "${TUNE_VALIDATE}" != "1" ]]; then
      echo "TUNE_VALIDATE=0: skipping confirmation DiscreteCQL STE jobs."
      exit "${proto_status}"
    fi
    if ! command -v sbatch >/dev/null 2>&1; then
      echo "WARNING: sbatch is not available here; written folders (if any) are on disk."
      exit "${proto_status}"
    fi
    if [[ "${proto_status}" -eq 0 || "${proto_status}" -eq 2 ]]; then
      PROTO_DIRS=("${TUNE_OUT_PREFIX}${TUNE_FATIGUE_TARGET}${TUNE_OUT_SUFFIX}")
      for target in ${TUNE_BENEFIT_TARGETS}; do
        PROTO_DIRS+=("${TUNE_OUT_PREFIX}${target}${TUNE_OUT_SUFFIX}")
      done
      echo "Submitting confirmation DiscreteCQL jobs for protocol folders."
      submit_validation_jobs "${PROTO_DIRS[@]}" || {
        [[ "${proto_status}" -eq 0 ]] && exit 1
      }
    else
      echo "Protocol failed (exit ${proto_status}); not submitting confirmation jobs." >&2
    fi
    exit "${proto_status}"
    ;;
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
      --scratch "${TUNE_SCRATCH}" \
      --report "logs/tune_ste_scan_${TUNE_KNOB}_${SLURM_JOB_ID:-local}.json"
    ;;
  apply)
    STABLE_FLAG=(--require-stable)
    if [[ "${TUNE_REQUIRE_STABLE}" == "0" ]]; then
      STABLE_FLAG=(--no-require-stable)
    fi
    EVAL_FLAG=(--no-eval)
    if [[ "${TUNE_APPLY_EVAL}" == "1" ]]; then
      EVAL_FLAG=(--eval)
    fi
    set +e
    "${PY}" tune_ste.py apply "${COMMON_ARGS[@]}" \
      --knob "${TUNE_KNOB}" \
      --kappa "${TUNE_KAPPA}" \
      --out-dir "${TUNE_OUT_DIR}" \
      --scratch "${TUNE_SCRATCH}" \
      "${EVAL_FLAG[@]}" \
      "${STABLE_FLAG[@]}" \
      "${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"}"
    apply_status=$?
    set -e
    if [[ "${TUNE_VALIDATE}" != "1" ]]; then
      echo "TUNE_VALIDATE=0: skipping confirmation DiscreteCQL STE jobs. Re-run with TUNE_PHASE=validate TUNE_VALIDATE_DIR=${TUNE_OUT_DIR}."
      exit "${apply_status}"
    fi
    if [[ "${apply_status}" -ne 0 ]]; then
      echo "Apply failed (exit ${apply_status}); not submitting confirmation DiscreteCQL STE jobs." >&2
      exit "${apply_status}"
    fi
    if ! command -v sbatch >/dev/null 2>&1; then
      echo "WARNING: sbatch is not available here; wrote ${TUNE_OUT_DIR}."
      echo "From a login node: TUNE_PHASE=validate TUNE_VALIDATE_DIR=${TUNE_OUT_DIR} bash run_tune_ste.sh"
      exit 0
    fi
    echo "Submitting confirmation DiscreteCQL jobs for ${TUNE_OUT_DIR}."
    submit_validation_jobs "${TUNE_OUT_DIR}"
    ;;
  calibrate)
    STABLE_FLAG=(--require-stable)
    if [[ "${TUNE_REQUIRE_STABLE}" == "0" ]]; then
      STABLE_FLAG=(--no-require-stable)
    fi
    SUFFIX_ARGS=()
    if [[ -n "${TUNE_OUT_SUFFIX}" ]]; then
      SUFFIX_ARGS=(--out-suffix "${TUNE_OUT_SUFFIX}")
    fi
    set +e
    "${PY}" tune_ste.py calibrate "${COMMON_ARGS[@]}" \
      --knob "${TUNE_KNOB}" \
      --targets ${TUNE_TARGETS} \
      --out-prefix "${TUNE_OUT_PREFIX}" \
      "${SUFFIX_ARGS[@]+"${SUFFIX_ARGS[@]}"}" \
      --scratch "${TUNE_SCRATCH}" \
      --kappa0 "${TUNE_KAPPA0}" \
      --tol "${TUNE_TOL}" \
      --max-iter "${TUNE_MAX_ITER}" \
      "${STABLE_FLAG[@]}" \
      "${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"}"
    cal_status=$?
    set -e
    if [[ "${TUNE_VALIDATE}" != "1" ]]; then
      echo "TUNE_VALIDATE=0: skipping confirmation DiscreteCQL STE jobs. Re-run with TUNE_PHASE=validate."
      exit "${cal_status}"
    fi
    if ! command -v sbatch >/dev/null 2>&1; then
      echo "WARNING: sbatch is not available here; tuned folders (if any) are written."
      echo "From a login node: TUNE_PHASE=validate bash run_tune_ste.sh"
      exit "${cal_status}"
    fi
    if [[ "${cal_status}" -eq 0 || "${cal_status}" -eq 2 ]]; then
      echo "Submitting confirmation DiscreteCQL jobs for stable folders."
      submit_validation_jobs || {
        # Nothing to submit is fatal only when calibration claimed full success.
        [[ "${cal_status}" -eq 0 ]] && exit 1
      }
    else
      echo "Calibration failed (exit ${cal_status}); not submitting confirmation DiscreteCQL STE jobs." >&2
    fi
    exit "${cal_status}"
    ;;
  validate)
    if [[ -n "${TUNE_VALIDATE_DIR:-}" ]]; then
      submit_validation_jobs "${TUNE_VALIDATE_DIR}"
    else
      submit_validation_jobs
    fi
    ;;
  *)
    echo "ERROR: unknown TUNE_PHASE=${TUNE_PHASE} (use diagnose, eval, scan, apply, calibrate, or validate)" >&2
    exit 1
    ;;
esac
