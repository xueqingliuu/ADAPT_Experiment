"""Estimate the RLSVI prior and the leave-one-user-out priors.

``experiment.py`` imports ``load_estimated_priors`` from ``est_prior.py``,
so the fitting module keeps that name. Run this after ``08_est_ew_weights.py``.
Missing predictors stay the column mean in ``est_prior._fill_nan``; online episodes fill from past observations only, which is acceptable because this prior is a batch regression.
"""

from vani_env import resolve_params_dir

print(f"params dir: {resolve_params_dir()}", flush=True)

from est_prior import cli

if __name__ == "__main__":
    cli()
