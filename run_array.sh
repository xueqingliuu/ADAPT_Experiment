#!/bin/bash
#SBATCH -J exp_seed_array
#SBATCH -p murphy
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 2-10:00:00
#SBATCH --mem=3G
#SBATCH --array=1-300%50
# Override for a top-up, e.g. 700 more seeds after 1-300:
#   sbatch --array=301-1000%50 --export=ALL,PRIOR_MODE=loo,ENV_VARIANT=env_para_vanilla,RESULTS_ROOT=results_vanilla_loo run_array.sh
#SBATCH -o logs/exp_%A_%a.out
#SBATCH -e logs/exp_%A_%a.err

# Fitted environment folder under the repo root. Switch later, e.g.:
#   ENV_VARIANT=env_para_ste0.5 sbatch run_array.sh
ENV_VARIANT="${ENV_VARIANT:-env_para_vanilla}"
# Testbed spec (sbatch --export=ALL,ENV_VARIANT=env_para_ste0.5,ADAPR_ACTIVITY=on,ADAPR_STAYER_FRACTION=0.25,ADAPR_MISSPEC=none).
# Or select catalog()[i] with TESTBED_INDEX=0..29 (overrides ENV_VARIANT / activity / ψ / misspec).
if [[ -n "${TESTBED_INDEX:-}" ]]; then
  eval "$(python3 -c "
from testbed import catalog
s = catalog()[int('${TESTBED_INDEX}')]
print(f'ENV_VARIANT={s.params_dir.name}')
print(f'ADAPR_ACTIVITY={s.activity}')
print(f'ADAPR_STAYER_FRACTION={s.stayer_fraction}')
print(f'ADAPR_MISSPEC={s.misspec}')
")"
fi
ADAPR_ACTIVITY="${ADAPR_ACTIVITY:-on}"
ADAPR_STAYER_FRACTION="${ADAPR_STAYER_FRACTION:-0}"
ADAPR_MISSPEC="${ADAPR_MISSPEC:-none}"
# Simulated weeks (experiment.py --nweek). Default 36.
#   sbatch --export=ALL,PRIOR_MODE=loo,NWEEK=300,RESULTS_ROOT=results_vanilla_w300_loo run_array.sh
NWEEK="${NWEEK:-36}"
# saved = shared ${ENV_VARIANT}/rl_priors.json, or zeros if USE_ESTIMATED_PRIORS=0.
# loo   = ${ENV_VARIANT}/loo_priors/held_out_<uid>.json (python est_prior.py --loo first).
# Keep saved as the cluster default so USE_ESTIMATED_PRIORS=0 still means zeros.
PRIOR_MODE="${PRIOR_MODE:-saved}"
# 0 = zero/identity PF+RLSVI priors. Ignored when PRIOR_MODE=loo (those files
# are estimated). 1 = load ${ENV_VARIANT}/rl_priors.json (PRIOR_MODE=saved).
USE_ESTIMATED_PRIORS="${USE_ESTIMATED_PRIORS:-0}"
# 1 = keep fitted PF / reward / Stage-2; zero-centre Q + Stage-1/1b means
# and inflate those covariances (POLICY_PRIOR_VAR_SCALE, default 10;
# diagonal floor POLICY_PRIOR_VAR_FLOOR, default 0.1).
ZERO_CENTRE_POLICY_PRIORS="${ZERO_CENTRE_POLICY_PRIORS:-0}"
POLICY_PRIOR_VAR_SCALE="${POLICY_PRIOR_VAR_SCALE:-10}"
POLICY_PRIOR_VAR_FLOOR="${POLICY_PRIOR_VAR_FLOOR:-0.1}"
# 1 = Q action block includes C (default). 0 = A*[1, E, b̂, b̃] only.
ACTION_BLOCK_C="${ACTION_BLOCK_C:-1}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load python
mamba activate budgeted

export ADAPR_PROJECT_ROOT="$SLURM_SUBMIT_DIR"
export ADAPR_PARAMS_DIR="$ENV_VARIANT"
export ADAPR_ACTIVITY
export ADAPR_STAYER_FRACTION
export ADAPR_MISSPEC
# Absolute folder for this shell only. Not exported: Python reads ADAPR_PARAMS_DIR.
PARAMS_ABS="${SLURM_SUBMIT_DIR}/${ENV_VARIANT}"
export ADAPR_PRIOR_MODE="${PRIOR_MODE}"
export USE_ESTIMATED_PRIORS
export ZERO_CENTRE_POLICY_PRIORS
export POLICY_PRIOR_VAR_SCALE
export POLICY_PRIOR_VAR_FLOOR
export ACTION_BLOCK_C
export SAVE_MODE=compact

if [ "${PRIOR_MODE}" != "saved" ] && [ "${PRIOR_MODE}" != "loo" ]; then
  echo "PRIOR_MODE must be saved or loo (got ${PRIOR_MODE})" >&2
  exit 1
fi
if [ "${PRIOR_MODE}" = "loo" ]; then
  LOO_DIR="${PARAMS_ABS}/loo_priors"
  if [ ! -d "${LOO_DIR}" ] || ! ls "${LOO_DIR}"/held_out_*.json >/dev/null 2>&1; then
    echo "PRIOR_MODE=loo requires leave-one-out prior files in ${LOO_DIR}." >&2
    echo "Generate them first:" >&2
    echo "  ADAPR_PARAMS_DIR=${PARAMS_ABS} python est_prior.py --loo" >&2
    exit 1
  fi
  if [ "${USE_ESTIMATED_PRIORS}" = "0" ]; then
    echo "WARNING: PRIOR_MODE=loo loads estimated per-user priors; USE_ESTIMATED_PRIORS=0 is ignored."
  fi
fi

# Keep result folders per environment (vanilla, ste0.5, ...).
# Override with RESULTS_ROOT=... ; if unset and ACTION_BLOCK_C=0, suffix
# _no_action_c so this batch does not mix with the default runs.
# PRIOR_MODE=loo gets _loo so it does not mix with saved/zero-prior batches.
if [ -z "${RESULTS_ROOT:-}" ]; then
  RESULTS_ROOT="$(python3 -c "from testbed import resolve_testbed_spec; print(resolve_testbed_spec('${ENV_VARIANT}').results_root)")"
  RESULTS_ROOT="${SLURM_SUBMIT_DIR}/${RESULTS_ROOT}"
  if [ "${ACTION_BLOCK_C}" = "0" ]; then
    RESULTS_ROOT="${RESULTS_ROOT}_no_action_c"
  fi
  if [ "${PRIOR_MODE}" = "loo" ]; then
    RESULTS_ROOT="${RESULTS_ROOT}_loo"
  fi
  if [ "${ZERO_CENTRE_POLICY_PRIORS}" = "1" ]; then
    RESULTS_ROOT="${RESULTS_ROOT}_zc_qs1"
  fi
fi
export RESULTS_ROOT
# Job requests 1 CPU; stop NumPy/OpenBLAS from spawning extra threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "ENV_VARIANT=${ENV_VARIANT}"
echo "TESTBED_INDEX=${TESTBED_INDEX:-<unset>}"
echo "ADAPR_ACTIVITY=${ADAPR_ACTIVITY}"
echo "ADAPR_STAYER_FRACTION=${ADAPR_STAYER_FRACTION}"
echo "ADAPR_MISSPEC=${ADAPR_MISSPEC}"
echo "PARAMS_DIR=${PARAMS_ABS}"
echo "PRIOR_MODE=${PRIOR_MODE}"
echo "USE_ESTIMATED_PRIORS=${USE_ESTIMATED_PRIORS}"
echo "ZERO_CENTRE_POLICY_PRIORS=${ZERO_CENTRE_POLICY_PRIORS}"
echo "POLICY_PRIOR_VAR_SCALE=${POLICY_PRIOR_VAR_SCALE}"
echo "POLICY_PRIOR_VAR_FLOOR=${POLICY_PRIOR_VAR_FLOOR}"
echo "ACTION_BLOCK_C=${ACTION_BLOCK_C}"
echo "NWEEK=${NWEEK}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "SAVE_MODE=${SAVE_MODE}"
echo "seed-idx=${SLURM_ARRAY_TASK_ID}"
echo "ADAPR_GENERATOR_CALIBRATION=${ADAPR_GENERATOR_CALIBRATION-<unset>}"
echo "ADAPR_PARAMS_DIR=${ADAPR_PARAMS_DIR}"
echo "host=$(hostname)"

python3 - "${PARAMS_ABS}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
p = Path(sys.argv[1]) / "params_env_13.json"
raw = p.read_bytes()
d = json.loads(raw)
m = dict(zip(d["theta_penalized_PV_names"], d["theta_penalized_PV"]))
vani = Path("vani_env.py").resolve()
print("DEBUG path", p.resolve())
print("DEBUG mtime", p.stat().st_mtime)
print("DEBUG params_md5", hashlib.md5(raw).hexdigest())
print("DEBUG alpha3", m.get("alpha3_action"))
print("DEBUG gamma3", m.get("gamma3_action"))
print("DEBUG vani_env", vani, hashlib.md5(vani.read_bytes()).hexdigest())
print("DEBUG calib", os.environ.get("ADAPR_GENERATOR_CALIBRATION"))
print("DEBUG PYTHONPATH", os.environ.get("PYTHONPATH", ""))
PY

EXTRA_ARGS=()
if [ "${ACTION_BLOCK_C}" = "0" ]; then
  EXTRA_ARGS+=(--no-action-c)
fi
if [ "${ZERO_CENTRE_POLICY_PRIORS}" = "1" ]; then
  EXTRA_ARGS+=(--zero-centre-policy-priors)
  EXTRA_ARGS+=(--policy-prior-var-scale "${POLICY_PRIOR_VAR_SCALE}")
fi

python experiment.py \
  --seed-idx "$SLURM_ARRAY_TASK_ID" \
  --params-dir "$PARAMS_ABS" \
  --prior-mode "$PRIOR_MODE" \
  --nweek "$NWEEK" \
  --activity "$ADAPR_ACTIVITY" \
  --stayer-fraction "$ADAPR_STAYER_FRACTION" \
  --misspec "$ADAPR_MISSPEC" \
  --save-mode compact \
  --results-root "$RESULTS_ROOT" \
  "${EXTRA_ARGS[@]}"
