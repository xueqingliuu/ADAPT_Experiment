"""Fit the remaining generative models and write the parameter directory.

Reads ``df_fit.csv`` (after ``06_fit_pu.py`` has attached E_w). Fits, per participant
or as a hierarchical Bayesian model:

    4-hour step counts, anticipated affect, weekly CAE / short CAE,
    prior-2h steps, active status, walking-suggestion interaction.

The fitting panel (MRT weeks 2-12, renumbered 1-11) is built in ``03_standardize.py``.

Script 1 writes the modelled cohort. This script checks that ``df_fit.csv``
is that set and raises if it is not. Low wear, missing surveys, and zero CAE
weeks are retained.
Writes ``params_env_<uid>.json`` (mediator/outcome blocks and residuals),
``pred_<uid>.json`` and ``user_ids.txt``. All models are hierarchical
Bayesian (one NUTS schedule, NUTS_TUNE / NUTS_DRAWS / NUTS_TARGET_ACCEPT).
Does not overwrite the E_w / PV / FW / PJ blocks from ``06_fit_pu.py`` (including
the ``J_w`` ``query_Jw_*`` suffix on PV/FW/PJ).
Next: ``08_est_ew_weights.py``.
"""
import json
import os
import shutil
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pathlib import Path

from vani_env import (
    assert_complete_week_slots,
    c1_from_week1_ewma,
    cae_mediator_ewma_rows,
    default_params_dir,
    impute_baseline_cae,
    load_week1_cae_inputs,
    mask_dropout_fit_outcomes,
    resolve_params_dir,
)

PROJECT_ROOT = Path(
    os.environ.get("ADAPR_PROJECT_ROOT", Path(__file__).resolve().parent)
).expanduser().resolve()
WORK_DIR = resolve_params_dir()
print(f"params dir: {WORK_DIR}", flush=True)
COMBINED_DIR = Path(
    os.environ.get("ADAPR_COMBINED_DIR", "")
).expanduser()
if not os.environ.get("ADAPR_COMBINED_DIR"):
    raise SystemExit(
        "Set ADAPR_COMBINED_DIR to the folder with extracted MRT tables."
    )
WORK_DIR.mkdir(parents=True, exist_ok=True)


# Aliases for later cells that use `folder` / `work_folder` (Path so `/` joins work)
folder = COMBINED_DIR
work_folder = Path(WORK_DIR)

std_params_source = default_params_dir() / "std_params.json"
std_params_dest = work_folder / "std_params.json"
if (
    std_params_source.is_file()
    and not std_params_dest.is_file()
    and std_params_source.resolve() != std_params_dest.resolve()
):
    shutil.copy2(std_params_source, std_params_dest)

df_fit = mask_dropout_fit_outcomes(pd.read_csv(folder / 'df_fit.csv'))

with open(std_params_dest, encoding="utf-8") as f:
    std_params = json.load(f)

file_params_env_prefix = str(work_folder / 'params_env_')
file_pred_prefix = str(work_folder / 'pred_')
file_user_ids = str(work_folder / 'user_ids.txt')

# The fitting panel (MRT weeks 2-12 renumbered 1-11; week 1 only supplies
# lags) is built in 03_standardize.py and shared with ``06_fit_pu.py``.
df_fit["Date"] = pd.to_datetime(df_fit["Date"], errors="coerce")
assert_complete_week_slots(df_fit)


def json_float(x, digits=3):
    """Round finite values; convert NaN/inf/None to JSON null."""
    if x is None:
        return None
    xf = float(x)
    if not np.isfinite(xf):
        return None
    return float(np.round(xf, digits))


def json_float_list(x, digits=3):
    """Serialize vector-like object as a JSON-safe flat list."""
    return [json_float(v, digits) for v in np.asarray(x, dtype=float).ravel()]


def merge_json_file(path, new_values):
    """Merge new values into an existing JSON file, preserving old keys."""
    path = Path(path)
    existing = {}
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)

    existing.update(new_values)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, allow_nan=False)


def fill_nan_with_mean(x, default=0.0):
    """Fill NaNs with the user's mean; use default if the whole vector is missing."""
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return np.full_like(x, default, dtype=float)
    return np.where(np.isnan(x), np.nanmean(x), x)


def _store_group_effects_as_series(effects_by_group):
    """Wrap a ``{group_key: 1d array}`` mapping as ``{group_key: pd.Series}``.

    Matches the ``.random_effects`` interface statsmodels' ``MixedLMResults``
    exposes, so :func:`get_user_random_effect` works unchanged.
    """
    return {key: pd.Series(np.asarray(vec, dtype=float)) for key, vec in effects_by_group.items()}


_FIT_DIAGNOSTICS = []


class _BayesianRandomCoefResult:
    """Lightweight stand-in for a ``MixedLMResults``, backed by an MCMC posterior.

    Exposes the same handful of attributes the rest of this script reads off a
    ``MixedLMResults`` (``fe_params``, ``random_effects``, ``scale``,
    ``converged``, ``summary()``), but every number here is a posterior mean
    from a single joint fit — there is no separate "fit then re-penalize"
    step, and no ad hoc variance floor.
    """

    def __init__(self, idata, predictor_names, group_keys):
        self.idata = idata
        self.predictor_names = list(predictor_names)
        self.group_keys = list(group_keys)

        post = idata.posterior
        self.fe_params = pd.Series(
            post["beta"].mean(("chain", "draw")).values, index=self.predictor_names
        )
        sigma_re = post["sigma_re"].mean(("chain", "draw")).values
        self.cov_re = np.diag(sigma_re ** 2)
        if "sigma_eps" in post:
            sigma_eps = float(post["sigma_eps"].mean(("chain", "draw")))
            self.scale = sigma_eps ** 2
        else:  # Bernoulli likelihood
            self.scale = float("nan")

        b_mean = post["b"].mean(("chain", "draw")).values  # (n_groups, p)
        self.random_effects = _store_group_effects_as_series(
            {key: b_mean[i] for i, key in enumerate(self.group_keys)}
        )

        n_div = int(idata.sample_stats["diverging"].sum())
        var_names = [v for v in ("beta", "sigma_re", "sigma_eps") if v in post]
        rhat = az.rhat(idata, var_names=var_names)
        max_rhat = max(float(rhat[v].max()) for v in rhat.data_vars)
        ess = az.ess(idata, var_names=var_names, method="bulk")
        min_ess = min(float(np.min(np.asarray(ess[v].values))) for v in ess.data_vars)
        self.n_divergences = n_div
        self.max_rhat = max_rhat
        self.converged = bool(n_div == 0 and max_rhat < 1.01)
        n_draws = int(post.sizes["draw"])
        n_chains = int(post.sizes["chain"])
        _FIT_DIAGNOSTICS.append(
            {
                "divergences": n_div,
                "max_rhat": max_rhat,
                "min_ess_bulk": min_ess,
                "draws": n_draws,
                "chains": n_chains,
            }
        )

    def summary(self):
        lines = [
            "Bayesian hierarchical random-coefficient model (NUTS via PyMC)",
            f"  divergences={self.n_divergences}  max_rhat={self.max_rhat:.4f}"
            f"  converged={self.converged}",
            "",
            "Posterior mean fixed effects (beta):",
            str(self.fe_params),
            "",
            "Posterior mean random-effect SD (diag of D):",
            str(pd.Series(np.sqrt(np.diag(self.cov_re)), index=self.predictor_names)),
        ]
        return "\n".join(lines)


# One shared scale for every weak/noninformative Normal slope prior:
# Normal(0, sqrt(10)), i.e. variance 10. On z-scored outcomes/predictors,
# plausible slopes are within roughly +/-1, so this is operationally flat —
# the point of a single constant is consistency, not information.
WEAK_BETA_PRIOR_SD = float(np.sqrt(10.0))

# One NUTS schedule for every hierarchical model in this script.
# ``06_fit_pu.py`` has its own sampler settings.
NUTS_TUNE = 2000
NUTS_DRAWS = 3000
NUTS_TARGET_ACCEPT = 0.95

# Random-effect SD prior mode, Gamma(2, mode): one value per outcome scale.
# Standardized continuous outcomes use 0.15; logit-scale (Bernoulli) models
# use 0.5, the value used by the perceived-utility model.
RE_SD_MODE_CONTINUOUS = 0.15
RE_SD_MODE_LOGIT = 0.5

# Sign-constrained population slopes (beta_prior_sign != 0).
#   "direction" (default): beta_j = sign_j * |N(0, 1)|. Only the direction comes
#       from the working causal model; the magnitude is left to the data
#       (outcomes are z-scored, so SD 1 is weakly informative). Same rule as
#       the perceived-utility model (06_fit_pu.py).
#   "magnitude" (legacy, up to 2026-10): HalfNormal scaled so its prior mean
#       equals beta_prior_mu_j (prior SD ~= 0.60 |mu_j|), i.e. the magnitude
#       is constrained as well.
# Select with ADAPR_SIGN_PRIOR=direction|magnitude.
SIGN_PRIOR_MODE = os.environ.get("ADAPR_SIGN_PRIOR", "direction").strip().lower()
if SIGN_PRIOR_MODE not in ("direction", "magnitude"):
    raise SystemExit(f"ADAPR_SIGN_PRIOR must be 'direction' or 'magnitude', got {SIGN_PRIOR_MODE!r}")
DIRECTION_ONLY_SCALE = 1.0
print(f"Sign-constrained slope priors: {SIGN_PRIOR_MODE}")


def fit_bayesian_random_coef_model(
    y,
    X,
    groups,
    predictor_names,
    beta_prior_mu=0.0,
    beta_prior_sd=WEAK_BETA_PRIOR_SD,
    beta_prior_sign=0.0,
    re_sd_prior_mode=RE_SD_MODE_CONTINUOUS,
    signed_prior_scale=None,
    sigma_eps_prior_alpha=2.0,
    sigma_eps_prior_beta=None,
    likelihood="gaussian",
    draws=NUTS_DRAWS,
    tune=NUTS_TUNE,
    chains=4,
    target_accept=NUTS_TARGET_ACCEPT,
    max_treedepth=12,
    seed=2026,
):
    """Fit ``y_ig = X_ig @ (beta + b_i) + eps_ig`` as one joint Bayesian model.

    This replaces the old two-stage design (fit an unregularized MixedLM,
    then separately ridge-shrink the fixed effects and clamp/float any
    near-zero random-effect variances). Here both regularizers are folded
    into a single posterior:

    - ``beta_prior_sd`` (per-column, or a scalar broadcast to all columns) is
      a Gaussian prior on the population-level slope. This *is* ridge
      regression, expressed as a prior instead of a penalty: MAP estimation
      under a ``Normal(mu, sd)`` prior is exactly ridge shrinkage toward
      ``mu`` with strength ``1/sd**2``. Pass a small ``sd`` (and/or a
      substantively motivated ``mu`` != 0) for collinear/implausible slopes,
      and a wide ``sd`` for slopes you want the data to determine freely.
    - ``beta_prior_sign`` (per-column, or scalar; entries in {-1, 0, +1})
      switches a column's population-slope prior from Normal to a
      sign-constrained HalfNormal: ``beta_j = sign_j * HalfNormal(sigma_j)``
      with ``sigma_j = |beta_prior_mu_j| / sqrt(2/pi)`` so the *prior mean*
      equals ``beta_prior_mu_j`` (a HalfNormal's mean is ``0.798 * sigma``).
      For these columns ``beta_prior_sd`` is ignored — a HalfNormal's spread
      is tied to its scale (prior SD ~= 0.60 * |mu|) — and the population
      slope can never cross zero, though individual users' ``beta + b_i``
      still can. Columns with sign 0 keep the Normal prior.
      ``signed_prior_scale`` overrides that scale with one constant for all
      signed columns (direction-only prior). ``None`` follows
      ``SIGN_PRIOR_MODE``: ``DIRECTION_ONLY_SCALE`` in "direction" mode, the
      mean-matched scale in "magnitude" mode.
    - ``re_sd_prior_mode`` sets a ``Gamma(shape=2, mode=re_sd_prior_mode)``
      prior on each random-effect SD. Unlike a Half-Normal/Half-Cauchy (the
      usual textbook default), a shape-2 Gamma has *zero density at exactly
      0*, so the posterior is pushed off the degenerate "same slope for every
      user" boundary without needing a hard-coded variance floor. (An
      InverseGamma is deliberately *not* used here: for group-level variances
      estimated from ~30 groups it is far from noninformative, see Gelman
      2006.)
    - the residual variance ``sigma_eps**2`` gets an
      ``InverseGamma(sigma_eps_prior_alpha, sigma_eps_prior_beta)`` prior —
      the conjugate family for a Gaussian likelihood, with zero density at 0
      so the noise scale can't collapse. The default scale is
      ``0.5 * var(y)`` (one rule for every Gaussian model in the testbed):
      the prior mean of the residual variance is half the outcome's sample
      variance, i.e. IG(2, 0.5) for a z-scored outcome; the prior variance
      is infinite, so the likelihood dominates. ``sigma_eps`` is exposed as a Deterministic so summaries
      and convergence checks are unchanged.

    Both priors act inside the same likelihood that estimates ``b_i``, so
    there's no need for a separate post-hoc ridge step or "protected"
    column list — every column is regularized according to its own prior,
    all at once.

    Returns a :class:`_BayesianRandomCoefResult` exposing ``fe_params``,
    ``random_effects``, ``scale``, ``converged``, and ``summary()`` so
    existing call sites (``get_user_random_effect`` etc.) work unchanged.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    predictor_names = list(predictor_names)
    p = X.shape[1]
    if len(predictor_names) != p:
        raise ValueError("predictor_names length must match X.shape[1]")

    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X, groups = y[keep], X[keep], groups[keep]
    if sigma_eps_prior_beta is None:
        sigma_eps_prior_beta = 0.5 * float(np.var(y, ddof=1))

    group_keys, group_idx = np.unique(groups, return_inverse=True)
    n_groups = len(group_keys)

    beta_prior_mu = np.broadcast_to(np.asarray(beta_prior_mu, dtype=float), (p,)).copy()
    beta_prior_sd = np.broadcast_to(np.asarray(beta_prior_sd, dtype=float), (p,)).copy()
    beta_prior_sign = np.broadcast_to(
        np.asarray(beta_prior_sign, dtype=float), (p,)
    ).copy()
    if not np.all(np.isin(beta_prior_sign, (-1.0, 0.0, 1.0))):
        raise ValueError("beta_prior_sign entries must be -1, 0, or +1")
    signed = beta_prior_sign != 0.0
    if signed_prior_scale is None and SIGN_PRIOR_MODE == "direction":
        signed_prior_scale = DIRECTION_ONLY_SCALE
    if signed_prior_scale is None and np.any(signed) and np.any(np.abs(beta_prior_mu[signed]) <= 0.0):
        raise ValueError(
            "sign-constrained columns need a nonzero beta_prior_mu "
            "(it sets the HalfNormal's prior mean)"
        )
    re_sd_prior_mode = np.broadcast_to(np.asarray(re_sd_prior_mode, dtype=float), (p,)).copy()
    re_rate = 1.0 / np.maximum(re_sd_prior_mode, 1e-6)

    halfnormal_mean_factor = float(np.sqrt(2.0 / np.pi))  # E[HalfNormal(s)] = s * this

    with pm.Model():
        if np.any(signed):
            free_idx = np.flatnonzero(~signed)
            sign_idx = np.flatnonzero(signed)
            beta_free = pm.Normal(
                "beta_free",
                mu=beta_prior_mu[free_idx],
                sigma=beta_prior_sd[free_idx],
                shape=len(free_idx),
            )
            if signed_prior_scale is None:
                mag_scale = np.abs(beta_prior_mu[sign_idx]) / halfnormal_mean_factor
            else:
                mag_scale = np.full(len(sign_idx), float(signed_prior_scale))
            beta_mag = pm.HalfNormal(
                "beta_mag",
                sigma=mag_scale,
                shape=len(sign_idx),
            )
            beta_full = pt.zeros(p)
            beta_full = pt.set_subtensor(beta_full[free_idx], beta_free)
            beta_full = pt.set_subtensor(
                beta_full[sign_idx], beta_prior_sign[sign_idx] * beta_mag
            )
            beta = pm.Deterministic("beta", beta_full)
        else:
            beta = pm.Normal("beta", mu=beta_prior_mu, sigma=beta_prior_sd, shape=p)
        sigma_re = pm.Gamma("sigma_re", alpha=2.0, beta=re_rate, shape=p)
        b_raw = pm.Normal("b_raw", mu=0.0, sigma=1.0, shape=(n_groups, p))
        b = pm.Deterministic("b", b_raw * sigma_re[None, :])
        theta = beta[None, :] + b
        mu = pm.math.sum(X * theta[group_idx], axis=1)
        if likelihood == "bernoulli":
            pm.Bernoulli("y_obs", logit_p=mu, observed=y)
        elif likelihood == "gaussian":
            sigma_eps_sq = pm.InverseGamma(
                "sigma_eps_sq", alpha=sigma_eps_prior_alpha, beta=sigma_eps_prior_beta
            )
            sigma_eps = pm.Deterministic("sigma_eps", pm.math.sqrt(sigma_eps_sq))
            pm.Normal("y_obs", mu=mu, sigma=sigma_eps, observed=y)
        else:
            raise ValueError(f"unknown likelihood {likelihood!r}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                max_treedepth=max_treedepth,
                random_seed=seed,
                progressbar=False,
            )

    result = _BayesianRandomCoefResult(idata, predictor_names, group_keys)
    if not result.converged:
        print(
            f"WARNING: Bayesian random-coefficient fit did not cleanly converge "
            f"(divergences={result.n_divergences}, max_rhat={result.max_rhat:.4f}); "
            "consider raising target_accept/tune or reparameterizing."
        )
    return result


def get_user_random_effect(result, userid, p):
    """
    Extract b_i from a fitted random-coefficient model result (MixedLM- or
    Bayesian-backed; both expose a ``.random_effects`` dict).
    If a user is missing from random_effects, return zeros.
    """
    keys_to_try = [userid, int(userid), str(userid)]

    b = None
    for key in keys_to_try:
        try:
            b = result.random_effects[key]
            break
        except KeyError:
            continue

    if b is None:
        return np.zeros(p, dtype=float)

    b = np.asarray(b, dtype=float).ravel()

    out = np.zeros(p, dtype=float)
    out[: min(p, len(b))] = b[: min(p, len(b))]
    return out
def build_cae_mixedlm_data(df, userid_all, K=14):
    """
    Build one row per participant-week for the CAE MixedLM.

    Current CAE model:
        CAE_avg ~ intercept
                  + CAE_avg_lastweek
                  + EWMA of 14 FourSC decision-slot summaries
                  + EWMA of 7 anticipated-affect daily summaries

    The two EWMA terms use :func:`vani_env.cae_mediator_ewmas` (week-local
    NaN fill, then the same normalized discount as
    ``01_extract_data._ewm_prior_rows``; most recent observation last).

    Output columns:
        ParticipantIdentifier
        week_index
        CAE_avg
        columns in THETA_CAE_NAMES
    """
    rows = []

    for userid in userid_all:
        dat_user = (
            df[df["ParticipantIdentifier"] == userid]
            .copy()
            .sort_values(["Date", "DecisionTime"], na_position="last")
            .reset_index(drop=True)
        )

        if len(dat_user) == 0:
            continue
        assert_complete_week_slots(dat_user)

        # Decision-level arrays.
        CAE_avg = dat_user["CAE_avg_norm"].to_numpy(dtype=float)
        CAE_avg_lastweek = dat_user["CAE_avg_lastweek_norm"].to_numpy(dtype=float)

        fourSC = dat_user["4hour_step_norm"].to_numpy(dtype=float)
        anticipated_affect = dat_user["anticipated_affect_norm"].to_numpy(dtype=float)

        # Match your current preprocessing.
        CAE_avg_lastweek = fill_nan_with_mean(CAE_avg_lastweek)

        # Convert decision-level rows to weekly rows.
        CAE_avg_sw = CAE_avg.reshape(-1, K)[:, 0]
        CAE_avg_lastweek_sw = CAE_avg_lastweek.reshape(-1, K)[:, 0]
        Intercept_sw = np.ones(len(CAE_avg_sw))

        # FourSC: 14 decision slots per week. Anticipated affect is stored on
        # both AM/PM rows; pass the 14-slot week so AM/PM averaging matches
        # the simulator (``_antic_daily_from_week``).
        foursc_wk = fourSC.reshape(-1, K)
        af = anticipated_affect.reshape(-1, K)
        foursc_e, antic_e = cae_mediator_ewma_rows(foursc_wk, af)

        X = np.column_stack(
            [
                Intercept_sw,
                CAE_avg_lastweek_sw,
                foursc_e,
                antic_e,
            ]
        )

        if X.shape[1] != len(THETA_CAE_NAMES):
            raise RuntimeError(
                f"User {userid}: CAE design has {X.shape[1]} columns, "
                f"but THETA_CAE_NAMES has {len(THETA_CAE_NAMES)} names."
            )

        for w in range(X.shape[0]):
            row = {
                "ParticipantIdentifier": int(userid),
                "week_index": int(w),
                "CAE_avg": CAE_avg_sw[w],
            }

            for j, name in enumerate(THETA_CAE_NAMES):
                row[name] = X[w, j]

            rows.append(row)

    return pd.DataFrame(rows)

def build_cae_short_mixedlm_data(df, userid_all, K=14):
    """
    Build one row per participant-week for the short-CAE MixedLM.

    Current short-CAE model:
        CAE_short_avg ~ intercept + CAE_avg

    Output columns:
        ParticipantIdentifier
        week_index
        CAE_short_avg
        intercept
        CAE_avg
    """
    rows = []

    for userid in userid_all:
        dat_user = (
            df[df["ParticipantIdentifier"] == userid]
            .copy()
            .sort_values(["Date", "DecisionTime"], na_position="last")
            .reset_index(drop=True)
        )

        if len(dat_user) == 0:
            continue
        assert_complete_week_slots(dat_user)

        # Decision-level arrays.
        CAE_avg = dat_user["CAE_avg_norm"].to_numpy(dtype=float)
        CAE_short_avg = dat_user["CAE_short_avg_norm"].to_numpy(dtype=float)

        # Convert from decision-level rows to weekly rows.
        # Take the first row of each week, same as your current code.
        CAE_avg_sw = CAE_avg.reshape(-1, K)[:, 0]
        CAE_short_avg_sw = CAE_short_avg.reshape(-1, K)[:, 0]

        # Predictor cannot contain NaN for MixedLM.
        CAE_avg_sw_filled = fill_nan_with_mean(CAE_avg_sw)

        for w in range(len(CAE_short_avg_sw)):
            rows.append(
                {
                    "ParticipantIdentifier": int(userid),
                    "week_index": int(w),
                    "CAE_short_avg": CAE_short_avg_sw[w],
                    "intercept": 1.0,
                    "CAE_avg": CAE_avg_sw_filled[w],
                }
            )

    return pd.DataFrame(rows)


def build_fourSC_bayes_data(df, userid_all):
    """One row per decision slot with an observed FourSC, for the pooled
    Bayesian FourSC random-coefficient model.

    Mirrors the per-user ``fourSC_cond`` construction in the main fitting
    loop exactly (same column reads, same NaN-filling, same column order
    matching ``THETA_FOURSC_NAMES``), just pooled across all users with a
    ``ParticipantIdentifier`` column for grouping instead of looping+fitting
    per user.
    """
    rows = []
    for userid in userid_all:
        dat_user = (
            df[df["ParticipantIdentifier"] == userid]
            .copy()
            .sort_values(["Date", "DecisionTime"], na_position="last")
            .reset_index(drop=True)
        )
        if len(dat_user) == 0:
            continue

        fourSC = dat_user["4hour_step_norm"].to_numpy()
        fourSC_lag1 = fill_nan_with_mean(dat_user["FourSC_lag1"].to_numpy())
        yesterday_step_count = fill_nan_with_mean(dat_user["YesterdayStepCount_norm"].to_numpy())
        prior2hour_step_count_filled = fill_nan_with_mean(dat_user["prior2hour_step_norm"].to_numpy())
        recent_burden = fill_nan_with_mean(dat_user["recent_burden_norm"].to_numpy())
        seven_day_pageview_count = fill_nan_with_mean(dat_user["Past7DaysPageviewEMA_norm"].to_numpy())
        past7days_morning_wearing = fill_nan_with_mean(dat_user["past7days_morning_wearing"].to_numpy())
        Interacted_7d_walk = fill_nan_with_mean(dat_user["Interacted_7d_walk"].to_numpy())
        anticipated_affect_yesterday = fill_nan_with_mean(
            dat_user["anticipated_affect_yesterday_norm"].to_numpy()
        )
        active_status_fraction_7days = fill_nan_with_mean(
            dat_user["active_status_fraction_7days"].to_numpy()
        )
        is_weekend = dat_user["is_weekend"].to_numpy(dtype=float)
        decision_time = dat_user["DecisionTime"].to_numpy()
        perceived_utility_lastweek = dat_user["perceived_utility_lastweek"].to_numpy(dtype=float)
        perceived_utility_lastweek = np.where(
            np.isnan(perceived_utility_lastweek),
            np.nanmean(perceived_utility_lastweek),
            perceived_utility_lastweek,
        )
        CAE_avg_lastweek = fill_nan_with_mean(dat_user["CAE_avg_lastweek_norm"].to_numpy())
        WalkingSuggestion = dat_user["WalkingSuggestion"].to_numpy()
        Intercept = np.ones(len(fourSC))

        cond = np.stack(
            [
                Intercept,
                fourSC_lag1,
                yesterday_step_count,
                prior2hour_step_count_filled,
                recent_burden,
                seven_day_pageview_count,
                past7days_morning_wearing,
                Interacted_7d_walk,
                anticipated_affect_yesterday,
                active_status_fraction_7days,
                is_weekend,
                decision_time,
                perceived_utility_lastweek,
                CAE_avg_lastweek,
                WalkingSuggestion,
                WalkingSuggestion * yesterday_step_count,
                WalkingSuggestion * prior2hour_step_count_filled,
                WalkingSuggestion * recent_burden,
                WalkingSuggestion * seven_day_pageview_count,
                WalkingSuggestion * past7days_morning_wearing,
                WalkingSuggestion * Interacted_7d_walk,
                WalkingSuggestion * anticipated_affect_yesterday,
                WalkingSuggestion * decision_time,
                WalkingSuggestion * perceived_utility_lastweek,
                WalkingSuggestion * CAE_avg_lastweek,
            ],
            axis=1,
        )

        idx_obs = ~np.isnan(fourSC)
        for j in np.flatnonzero(idx_obs):
            row = {"ParticipantIdentifier": int(userid), "FourSC": float(fourSC[j])}
            for k, name in enumerate(THETA_FOURSC_NAMES):
                row[name] = cond[j, k]
            rows.append(row)

    return pd.DataFrame(rows)


def build_antic_bayes_data(df, userid_all):
    """One row per calendar day with an observed anticipated-affect value,
    for the pooled Bayesian anticipated-affect random-coefficient model.

    Mirrors the per-user ``anticipated_affect_cond_day`` construction in the
    main fitting loop exactly, pooled across all users.
    """
    rows = []
    for userid in userid_all:
        dat_user = (
            df[df["ParticipantIdentifier"] == userid]
            .copy()
            .sort_values(["Date", "DecisionTime"], na_position="last")
            .reset_index(drop=True)
        )
        if len(dat_user) == 0:
            continue
        assert_complete_week_slots(dat_user)

        anticipated_affect = dat_user["anticipated_affect_norm"].to_numpy()
        anticipated_affect_yesterday = fill_nan_with_mean(
            dat_user["anticipated_affect_yesterday_norm"].to_numpy()
        )
        active_status_fraction_7days = fill_nan_with_mean(
            dat_user["active_status_fraction_7days"].to_numpy()
        )
        is_weekend = dat_user["is_weekend"].to_numpy(dtype=float)
        perceived_utility_lastweek = dat_user["perceived_utility_lastweek"].to_numpy(dtype=float)
        perceived_utility_lastweek = np.where(
            np.isnan(perceived_utility_lastweek),
            np.nanmean(perceived_utility_lastweek),
            perceived_utility_lastweek,
        )
        CAE_avg_lastweek = fill_nan_with_mean(dat_user["CAE_avg_lastweek_norm"].to_numpy())
        recent_burden = fill_nan_with_mean(dat_user["recent_burden_norm"].to_numpy())
        decision_time = dat_user["DecisionTime"].to_numpy()
        WalkingSuggestion = dat_user["WalkingSuggestion"].to_numpy()
        Intercept = np.ones(len(anticipated_affect))

        idx_morning = decision_time == 0
        _am = idx_morning
        Walking_pair = WalkingSuggestion.reshape(-1, 2)
        ws_morning_day = Walking_pair[:, 0]
        ws_afternoon_day = Walking_pair[:, 1]

        cond = np.stack(
            [
                Intercept[_am],
                anticipated_affect_yesterday[_am],
                active_status_fraction_7days[_am],
                is_weekend[_am],
                perceived_utility_lastweek[_am],
                CAE_avg_lastweek[_am],
                recent_burden[_am],
                ws_morning_day,
                ws_afternoon_day,
                ws_morning_day * perceived_utility_lastweek[_am],
                ws_afternoon_day * perceived_utility_lastweek[_am],
                ws_morning_day * CAE_avg_lastweek[_am],
                ws_afternoon_day * CAE_avg_lastweek[_am],
                ws_morning_day * recent_burden[_am],
                ws_afternoon_day * recent_burden[_am],
                ws_morning_day * active_status_fraction_7days[_am],
                ws_afternoon_day * active_status_fraction_7days[_am],
            ],
            axis=1,
        )

        y_antic_day = anticipated_affect[_am]
        idx_obs = ~np.isnan(y_antic_day)
        for j in np.flatnonzero(idx_obs):
            row = {"ParticipantIdentifier": int(userid), "AnticipatedAffect": float(y_antic_day[j])}
            for k, name in enumerate(THETA_ANTIC_NAMES):
                row[name] = cond[j, k]
            rows.append(row)

    return pd.DataFrame(rows)


THETA_PRIOR2HOUR_STEP_COUNT_NAMES = [
    "intercept",
    "EMA_Prior2HourStepCount",
    "is_weekend",
    "decision_time",
    # next-4h steps of the previous decision slot, separate AM / PM slopes
    # (PM: same-day morning, whose window overlaps the PM prior-2h window)
    "fourSC_lag1_morning",
    "fourSC_lag1_afternoon",
]

THETA_ACTIVE_STATUS_NAMES = [
    "intercept",
    "active_status_fraction_7days",
    "is_weekend",
]

THETA_WS_INTERACTION_NAMES = [
    "intercept",
    "Interacted_7d_walk",
    "is_weekend",
    "decision_time",
]


THETA_FOURSC_NAMES = [
    "intercept",
    "fourSC_lag1",
    "yesterday_step_count",
    "prior2hour_step_count",
    "recent_burden",
    "seven_day_pageview_count",
    "past7days_morning_wearing",
    "Interacted_7d_walk",
    "anticipated_affect_yesterday",
    "fractionofactivedayspast7days",
    "is_weekend",
    "decision_time",
    "perceived_utility_lastweek",
    "CAE_avg_lastweek",
    "WalkingSuggestion",
    "WalkingSuggestion_by_yesterday_step_count",
    "WalkingSuggestion_by_prior2hour_step_count",
    "WalkingSuggestion_by_recent_burden",
    "WalkingSuggestion_by_seven_day_pageview_count",
    "WalkingSuggestion_by_past7days_morning_wearing",
    "WalkingSuggestion_by_Interacted_7d_walk",
    "WalkingSuggestion_by_anticipated_affect_yesterday",
    "WalkingSuggestion_by_decision_time",
    "WalkingSuggestion_by_perceived_utility_lastweek",
    "WalkingSuggestion_by_CAE_avg_lastweek",
]

THETA_ANTIC_NAMES = [
    "intercept",
    "anticipated_affect_yesterday",
    "active_status_fraction_7days",
    "is_weekend",
    "perceived_utility_lastweek",
    "CAE_avg_lastweek",
    "recent_burden",
    "A0_morning",
    "A1_afternoon",
    "A0_morning_by_perceived_utility_lastweek",
    "A1_afternoon_by_perceived_utility_lastweek",
    "A0_morning_by_CAE_avg_lastweek",
    "A1_afternoon_by_CAE_avg_lastweek",
    "A0_morning_by_recent_burden",
    "A1_afternoon_by_recent_burden",
    "A0_morning_by_active_status_fraction_7days",
    "A1_afternoon_by_active_status_fraction_7days",
]

THETA_CAE_NAMES = [
    "intercept",
    "CAE_avg_lastweek",
    "fourSC_ewma",
    "anticipated_affect_ewma",
]

THETA_CAE_SHORT_AVG_NAMES = [
    "intercept",
    "CAE_avg",
]

def other_variable_designs(dat_user):
    """Designs, outcomes and observed-row masks for the additional variables.

    Missing predictors are replaced by the participant mean (same rule as the
    CAE-related models). Returns ``{name: (X, y, observed_mask)}``.
    """
    dt = dat_user["DecisionTime"].to_numpy(dtype=float)
    we = dat_user["is_weekend"].to_numpy(dtype=float)
    one = np.ones(len(dt))
    lag = fill_nan_with_mean(dat_user["FourSC_lag1"].to_numpy(dtype=float))
    ema_p2h = fill_nan_with_mean(dat_user["EMA_Prior2HourStepCount_norm"].to_numpy(dtype=float))
    y_p2h = dat_user["prior2hour_step_norm"].to_numpy(dtype=float)
    X_p2h = np.column_stack([one, ema_p2h, we, dt, (1.0 - dt) * lag, dt * lag])

    frac_active = fill_nan_with_mean(dat_user["active_status_fraction_7days"].to_numpy(dtype=float))
    y_ad = dat_user["active_status"].to_numpy(dtype=float)
    X_ad = np.column_stack([one, frac_active, we])

    frac_inter = fill_nan_with_mean(dat_user["Interacted_7d_walk"].to_numpy(dtype=float))
    y_as = dat_user["Interacted_walk"].to_numpy(dtype=float)
    sent = dat_user["WalkingSuggestion"].to_numpy(dtype=float)
    X_as = np.column_stack([one, frac_inter, we, dt])
    return {
        "prior2hour": (X_p2h, y_p2h, np.isfinite(y_p2h)),
        # active status is the end-of-day answer; one row per day (morning row)
        "active_status": (X_ad, y_ad, (dt == 0) & np.isfinite(y_ad)),
        # interaction is defined only when a suggestion was delivered
        "ws_interaction": (X_as, y_as, np.isfinite(y_as) & (sent == 1.0)),
    }


def pooled_other_variable_data(df_fit, userids):
    """Stack :func:`other_variable_designs` over participants (observed rows only)."""
    out = {k: ([], [], []) for k in ("prior2hour", "active_status", "ws_interaction")}
    for uid in userids:
        dat_user = df_fit[df_fit["ParticipantIdentifier"] == uid].sort_values(
            ["Date", "DecisionTime"], na_position="last").reset_index(drop=True)
        for k, (X, y, m) in other_variable_designs(dat_user).items():
            out[k][0].append(X[m]); out[k][1].append(y[m]); out[k][2].append(np.full(int(m.sum()), uid))
    return {k: (np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)) for k, (Xs, ys, gs) in out.items()}


def participant_theta(result, userid):
    """Posterior-mean participant coefficients beta + b_i."""
    beta = result.fe_params.to_numpy(dtype=float)
    return beta + get_user_random_effect(result, userid, beta.size)


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

# Users with no observed CAE (all-NaN ``CAE_avg_norm``) are retained since
# 2026-10-17. The Bayesian random-coefficient CAE fit simply has no rows for
# them, so their CAE coefficients are the population slopes (b_i = 0), and
# their baseline CAE is imputed from the fitted model (``cae_baseline_norm``
# below). They still inform every other model.
_all_userids = df_fit['ParticipantIdentifier'].unique()
_has_cae_obs = (
    df_fit.groupby('ParticipantIdentifier', sort=False)['CAE_avg_norm']
    .apply(lambda s: np.any(~np.isnan(s.to_numpy(dtype=float))))
)
userid_all = np.array([int(uid) for uid in _all_userids], dtype=int)
_cohort_path = Path(folder) / "modelled_cohort.csv"
if not _cohort_path.is_file():
    raise SystemExit(
        f"missing {_cohort_path}; script 1 writes the modelled cohort"
    )
_cohort = pd.read_csv(_cohort_path)
_modelled_ids = {
    int(uid)
    for uid in _cohort.loc[_cohort["in_modelled"].astype(bool), "uid"]
}
_fit_ids = {int(uid) for uid in userid_all}
if _fit_ids != _modelled_ids:
    raise SystemExit(
        "df_fit users do not match modelled_cohort.csv in_modelled: "
        f"only in df_fit {sorted(_fit_ids - _modelled_ids)}; "
        f"only modelled {sorted(_modelled_ids - _fit_ids)}"
    )
no_cae_userids = np.array(
    [uid for uid in _all_userids if not bool(_has_cae_obs.get(uid, False))],
    dtype=int,
)
if no_cae_userids.size > 0:
    print(
        "Users with no CAE observations (kept; CAE coefficients = population, "
        "baseline CAE model-imputed): "
        + ", ".join(str(int(u)) for u in no_cae_userids)
    )
if userid_all.size == 0:
    raise ValueError("No users with observed CAE values remain after filtering.")

# ============================================================
# Fit CAE random-coefficient model as one joint Bayesian hierarchical fit
# ============================================================
# Every column gets a Normal(mu, sd) prior on its population slope (beta) and
# a Gamma(shape=2, mode) prior on its between-user SD (sigma_re); see
# fit_bayesian_random_coef_model's docstring. Two of these are substantively
# informative, not just "weak regularization defaults":
#   - CAE_avg_lastweek (a1): a week-to-week autoregression coefficient for a
#     bounded, mean-reverting activity measure should plausibly sit well
#     below 1 (a loop gain >= 1 makes the simulated trajectory unstable).
#     Originally centered at 0.4; recentered at 0.2 after the
#     residual-dependence diagnostics (residual_dependence.py, 2026-09-15).
#     The fitted CAE residuals' lag-1 autocorrelation averaged -0.44 across
#     users, versus a null of -0.18 +/- 0.06 for this exact hierarchical
#     fitting pipeline on white-noise emissions (parametric bootstrap,
#     z ~ -4). The excess is the classic lagged-observed-outcome +
#     measurement-error signature: with survey noise nu_w in weekly CAE, the
#     residual contains nu_w - a1*nu_{w-1}, an MA(1) with lag-1 ACF about
#     -a1/(1+a1^2). At the refitted a1 ~ 0.27 that predicts -0.24, and
#     -0.18 (null) - 0.24 ~ -0.42, matching the observed value almost
#     exactly — i.e. the conditional-mean model is fine, but a large a1
#     over-attributes persistence to the AR term. Recentering at 0.2 shrinks
#     that over-attribution (fitted a1 drops from ~0.38, which merely echoed
#     the old prior center, to ~0.27); the leftover negative residual
#     dependence is deliberately NOT modeled here because vani_env's default
#     noise="ar1" sampler replays it per user from the stored resid_CAE
#     series at simulation time.
#   - anticipated_affect_ewma: corr(anticipated_affect_ewma, CAE_avg_lastweek)
#     = 0.90 in the training data, so an unregularized fit can't tell these
#     two apart and dumps an implausible amount of signal (~1.5-1.6) onto
#     whichever one is least constrained. Centering it at a similarly modest
#     0.3 keeps the *joint* fit from re-inflating it once a1 is pinned down.
#   - fourSC_ewma: this and anticipated_affect_ewma are the two "M^Y ->
#     Y_{w+1} nonnegative" terms that used to be sign-flipped
#     post-hoc. A flat/uninformative prior here let the population mean land
#     essentially at zero (slightly negative) with a between-user SD (~0.17)
#     much larger than that mean, so a large fraction of users came back
#     negative pre-flip — and post-hoc flipping *reflects* whatever noisy
#     magnitude those users happened to get instead of shrinking it, which
#     was inflating STE. A mild positive prior (weaker than
#     anticipated_affect_ewma's, since the raw data doesn't support as large
#     an effect) asserts "this should help on average" at the *population*
#     level while leaving each user's random effect free to land negative if
#     their own data supports it — real heterogeneity, not reflection.
# The intercept has no such prior belief, so it gets a wide prior and is
# essentially estimated by the data alone.
#
# 2026-09-15: the three informative slopes (CAE_avg_lastweek, fourSC_ewma,
# anticipated_affect_ewma) now use sign-constrained HalfNormal priors
# (beta_prior_sign = +1) instead of Normal. The HalfNormal scale is set so
# its prior *mean* equals the mu listed below (sigma = mu / 0.798), i.e.
# the same central belief with positivity enforced. A side-by-side refit on
# the training panel gave near-identical posteriors (a1 0.266 vs 0.272,
# antic 0.302 vs 0.324, identical residual lag-1 ACF, 0 divergences either
# way). Note beta_prior_sd is ignored for these columns: a HalfNormal's
# spread is tied to its mean (prior SD ~= 0.60 * mu).
CAE_BETA_PRIOR_MU = np.array([0.0, 0.2, 0.15, 0.3])
# sd applies to the unsigned column (intercept) only; entries for the
# three signed columns are ignored (their spread comes from the HalfNormal).
CAE_BETA_PRIOR_SD = np.array(
    [WEAK_BETA_PRIOR_SD, 0.15, 0.15, 0.15]
)
CAE_BETA_PRIOR_SIGN = np.array([0.0, 1.0, 1.0, 1.0])
CAE_RE_SD_PRIOR_MODE = np.full(len(THETA_CAE_NAMES), RE_SD_MODE_CONTINUOUS)

cae_ml_df = build_cae_mixedlm_data(df_fit, userid_all, K=14)

cae_ml_obs = cae_ml_df.dropna(subset=["CAE_avg"]).copy()

X_cae_obs = cae_ml_obs[THETA_CAE_NAMES].to_numpy(dtype=float)
y_cae_obs = cae_ml_obs["CAE_avg"].to_numpy(dtype=float)
groups_cae_obs = cae_ml_obs["ParticipantIdentifier"].to_numpy()

mixedlm_CAE = fit_bayesian_random_coef_model(
    y=y_cae_obs,
    X=X_cae_obs,
    groups=groups_cae_obs,
    predictor_names=list(THETA_CAE_NAMES),
    beta_prior_mu=CAE_BETA_PRIOR_MU,
    beta_prior_sd=CAE_BETA_PRIOR_SD,
    beta_prior_sign=CAE_BETA_PRIOR_SIGN,
    re_sd_prior_mode=CAE_RE_SD_PRIOR_MODE,
)

print(mixedlm_CAE.summary())
print(f"CAE Bayesian fit converged: {mixedlm_CAE.converged}")

# ============================================================
# Fit short-CAE random-coefficient model the same way, with weak/uninformative
# priors throughout (no collinearity or stability concerns for this model).
# ============================================================

cae_short_ml_df = build_cae_short_mixedlm_data(df_fit, userid_all, K=14)

cae_short_ml_obs = cae_short_ml_df.dropna(subset=["CAE_short_avg"]).copy()

X_cae_short_obs = cae_short_ml_obs[THETA_CAE_SHORT_AVG_NAMES].to_numpy(dtype=float)
y_cae_short_obs = cae_short_ml_obs["CAE_short_avg"].to_numpy(dtype=float)
groups_cae_short_obs = cae_short_ml_obs["ParticipantIdentifier"].to_numpy()

mixedlm_CAE_short = fit_bayesian_random_coef_model(
    y=y_cae_short_obs,
    X=X_cae_short_obs,
    groups=groups_cae_short_obs,
    predictor_names=list(THETA_CAE_SHORT_AVG_NAMES),
)

print(mixedlm_CAE_short.summary())
print(f"CAE_short Bayesian fit converged: {mixedlm_CAE_short.converged}")

# ============================================================
# Fit the anticipated-affect and FourSC random-coefficient models the same
# way. These were previously 31 independent single-user Ridge fits with zero
# cross-user information sharing — unlike CAE/CAE_short, nothing regularized
# their action-effect or ``recent_burden`` (habituation) coefficients. With
# only a few dozen observations against a 17-27 column design, per-user
# noise routinely flipped ``recent_burden``'s sign: 18/31 users came back
# with recent_burden >= 0 (no habituation, sometimes literally backwards),
# which let a constant-suggestion policy compound CAE gains unchecked for
# those users and was the single largest driver of outlier STE values.
#
# ``recent_burden`` (and FourSC's WalkingSuggestion-by-recent_burden
# interaction) get a mildly-informative negative-centered prior; every other
# column keeps a wide, weakly-informative prior and is essentially estimated
# by the data. The main fix isn't forcing the population mean negative (the
# data can and does push back on that, see FourSC's interaction term below)
# — it's that partial pooling shrinks each user's *own* noisy estimate
# toward a shared, precisely-estimated population value, so a handful of
# training rows can no longer single-handedly flip the sign for one person.
#
# The coefficients below are also the ones that used to be
# sign-flipped post-hoc (E_w/Y_w/A -> M^Y "should help" assumptions). A mild
# positive prior on the *population* mean asserts that assumption where the
# post-hoc flip used to, but — unlike the flip — it doesn't touch individual
# users' random effects, so a person whose own data genuinely shows no
# effect (or a negative one) stays that way instead of being reflected into
# an inflated positive value.
# Habituation columns (recent_burden and its action interactions) also get a
# *tighter* between-user random-effect SD (default is 0.15 everywhere else).
# With the wide default, individual posteriors could still land solidly on
# the "wrong" (reinforcing, i.e. no habituation) side of a negative
# population mean -- e.g. one user's WalkingSuggestion_by_recent_burden came
# back +0.057 against a population mean of -0.05 -- which let a
# constant-suggestion policy compound unchecked for that person over 36
# weeks and was the dominant remaining driver of outlier STE values. Tighter
# beta_prior_sd (more confident about the population mean) plus tighter
# re_sd_prior_mode (less between-user spread allowed) both push toward this
# fix; individuals can still deviate, just less drastically.
# 2026-09-15: informative columns switched from Normal(mu, sd) to
# sign-constrained HalfNormal priors (beta_prior_sign = sign(mu)), with the
# HalfNormal scale set so its prior mean equals the mu below (sigma =
# |mu| / 0.798). beta_prior_sd is ignored for signed columns; the effective
# prior SD becomes ~0.60 * |mu| (~0.03 for the -0.05 burden terms — close
# to the old 0.03-0.04 — but ~0.018 for the +0.03 "should help" terms,
# tighter than the old 0.08, so those population means are now confined to
# roughly (0, 0.09)).
ANTIC_RE_SD_PRIOR_MODE = np.full(len(THETA_ANTIC_NAMES), RE_SD_MODE_CONTINUOUS)
ANTIC_BETA_PRIOR_SD = np.full(len(THETA_ANTIC_NAMES), WEAK_BETA_PRIOR_SD)
ANTIC_BETA_PRIOR_MU = np.zeros(len(THETA_ANTIC_NAMES))
ANTIC_BETA_PRIOR_SIGN = np.zeros(len(THETA_ANTIC_NAMES))
for _name in ("A0_morning_by_recent_burden", "A1_afternoon_by_recent_burden"):
    ANTIC_BETA_PRIOR_MU[THETA_ANTIC_NAMES.index(_name)] = -0.05
    ANTIC_BETA_PRIOR_SIGN[THETA_ANTIC_NAMES.index(_name)] = -1.0
ANTIC_BETA_PRIOR_MU[THETA_ANTIC_NAMES.index("recent_burden")] = -0.05
ANTIC_BETA_PRIOR_SIGN[THETA_ANTIC_NAMES.index("recent_burden")] = -1.0
for _name in (
    "perceived_utility_lastweek",
    "CAE_avg_lastweek",
    "A0_morning",
    "A1_afternoon",
    "A0_morning_by_perceived_utility_lastweek",
    "A1_afternoon_by_perceived_utility_lastweek",
    "A0_morning_by_CAE_avg_lastweek",
    "A1_afternoon_by_CAE_avg_lastweek",
    "A0_morning_by_active_status_fraction_7days",
    "A1_afternoon_by_active_status_fraction_7days",
):
    ANTIC_BETA_PRIOR_MU[THETA_ANTIC_NAMES.index(_name)] = 0.03
    ANTIC_BETA_PRIOR_SIGN[THETA_ANTIC_NAMES.index(_name)] = 1.0

antic_bayes_df = build_antic_bayes_data(df_fit, userid_all)
X_antic_obs = antic_bayes_df[THETA_ANTIC_NAMES].to_numpy(dtype=float)
y_antic_obs = antic_bayes_df["AnticipatedAffect"].to_numpy(dtype=float)
groups_antic_obs = antic_bayes_df["ParticipantIdentifier"].to_numpy()

mixedlm_antic = fit_bayesian_random_coef_model(
    y=y_antic_obs,
    X=X_antic_obs,
    groups=groups_antic_obs,
    predictor_names=list(THETA_ANTIC_NAMES),
    beta_prior_mu=ANTIC_BETA_PRIOR_MU,
    beta_prior_sd=ANTIC_BETA_PRIOR_SD,
    beta_prior_sign=ANTIC_BETA_PRIOR_SIGN,
    re_sd_prior_mode=ANTIC_RE_SD_PRIOR_MODE,
)

print(mixedlm_antic.summary())
print(f"Anticipated-affect Bayesian fit converged: {mixedlm_antic.converged}")

# Same 2026-09-15 HalfNormal switch as theta_antic above: informative
# columns use beta_prior_sign = sign(mu) with prior mean matched to mu.
FOURSC_RE_SD_PRIOR_MODE = np.full(len(THETA_FOURSC_NAMES), RE_SD_MODE_CONTINUOUS)
FOURSC_BETA_PRIOR_SD = np.full(len(THETA_FOURSC_NAMES), WEAK_BETA_PRIOR_SD)
FOURSC_BETA_PRIOR_MU = np.zeros(len(THETA_FOURSC_NAMES))
FOURSC_BETA_PRIOR_SIGN = np.zeros(len(THETA_FOURSC_NAMES))
FOURSC_BETA_PRIOR_MU[THETA_FOURSC_NAMES.index("recent_burden")] = -0.05
FOURSC_BETA_PRIOR_SIGN[THETA_FOURSC_NAMES.index("recent_burden")] = -1.0
# This interaction is the one that most directly let a constant-suggestion
# policy compound unchecked for a couple of outlier users: one came back at
# +0.057 against this -0.05 population mean under the old wide re_sd prior
# (mode=0.15), i.e. that person's fitted dynamics said each additional
# suggestion got *more* effective as burden accumulated. Tighter beta_sd and
# re_sd_prior_mode here don't rule that out, but make it harder for a
# handful of noisy per-user rows to produce it.
FOURSC_BETA_PRIOR_MU[THETA_FOURSC_NAMES.index("WalkingSuggestion_by_recent_burden")] = -0.05
FOURSC_BETA_PRIOR_SIGN[THETA_FOURSC_NAMES.index("WalkingSuggestion_by_recent_burden")] = -1.0
# Same "assert the population mean, leave individuals free" treatment as
# theta_antic above, for the E_w/Y_w/A -> M^Y terms that used to be
# sign-flipped post-hoc.
for _name in (
    "perceived_utility_lastweek",
    "CAE_avg_lastweek",
    "WalkingSuggestion",
    "WalkingSuggestion_by_perceived_utility_lastweek",
    "WalkingSuggestion_by_CAE_avg_lastweek",
):
    FOURSC_BETA_PRIOR_MU[THETA_FOURSC_NAMES.index(_name)] = 0.03
    FOURSC_BETA_PRIOR_SIGN[THETA_FOURSC_NAMES.index(_name)] = 1.0

fourSC_bayes_df = build_fourSC_bayes_data(df_fit, userid_all)
X_fourSC_obs = fourSC_bayes_df[THETA_FOURSC_NAMES].to_numpy(dtype=float)
y_fourSC_obs = fourSC_bayes_df["FourSC"].to_numpy(dtype=float)
groups_fourSC_obs = fourSC_bayes_df["ParticipantIdentifier"].to_numpy()

# 24-column RE model × 28 users; 2k/1k at target_accept=0.97 hit
# max_rhat=1.018 (0 divergences). Match CAE's longer NUTS schedule.
mixedlm_fourSC = fit_bayesian_random_coef_model(
    y=y_fourSC_obs,
    X=X_fourSC_obs,
    groups=groups_fourSC_obs,
    predictor_names=list(THETA_FOURSC_NAMES),
    beta_prior_mu=FOURSC_BETA_PRIOR_MU,
    beta_prior_sd=FOURSC_BETA_PRIOR_SD,
    beta_prior_sign=FOURSC_BETA_PRIOR_SIGN,
    re_sd_prior_mode=FOURSC_RE_SD_PRIOR_MODE,
)

print(mixedlm_fourSC.summary())
print(f"FourSC Bayesian fit converged: {mixedlm_fourSC.converged}")

# ============================================================
# Additional time-varying variables: joint hierarchical fits (same model
# class as above; Bernoulli likelihood for the binary outcomes).
# ============================================================
_other = pooled_other_variable_data(df_fit, userid_all)
mixedlm_prior2hour = fit_bayesian_random_coef_model(
    y=_other["prior2hour"][1], X=_other["prior2hour"][0], groups=_other["prior2hour"][2],
    predictor_names=list(THETA_PRIOR2HOUR_STEP_COUNT_NAMES),
)
print(mixedlm_prior2hour.summary())
mixedlm_active_status = fit_bayesian_random_coef_model(
    y=_other["active_status"][1], X=_other["active_status"][0], groups=_other["active_status"][2],
    predictor_names=list(THETA_ACTIVE_STATUS_NAMES),
    likelihood="bernoulli", re_sd_prior_mode=RE_SD_MODE_LOGIT,
)
print(mixedlm_active_status.summary())
mixedlm_ws_interaction = fit_bayesian_random_coef_model(
    y=_other["ws_interaction"][1], X=_other["ws_interaction"][0], groups=_other["ws_interaction"][2],
    predictor_names=list(THETA_WS_INTERACTION_NAMES),
    likelihood="bernoulli", re_sd_prior_mode=RE_SD_MODE_LOGIT,
)
print(mixedlm_ws_interaction.summary())

cae_baseline_rows = []

# Week-1 mediator EWMAs are written by 03_standardize.py with the same
# cae_mediator_ewmas fill as c_2..c_12. A missing file is an error. A user
# with no week-1 rows falls back to c_2 and is logged.
_week1_path = folder / "week1_cae_inputs.csv"
try:
    _week1_inputs = load_week1_cae_inputs(_week1_path)
except FileNotFoundError as exc:
    raise SystemExit(str(exc)) from exc
print(f"Loaded week-1 CAE inputs for {len(_week1_inputs)} users from {_week1_path}")
_week1_c2_users = []

for i, userid in enumerate(userid_all):
    dat_user = df_fit[df_fit['ParticipantIdentifier'] == userid].copy()
    dat_user = dat_user.sort_values(['Date', 'DecisionTime'], na_position='last').reset_index(drop=True)
    assert_complete_week_slots(dat_user)

    # extract the response
    # M^Y_{w,d,t}
    fourSC = dat_user['4hour_step_norm'].to_numpy()
    fourSC_lag1 = dat_user['FourSC_lag1'].to_numpy()

    # M^Y_{w,d}
    anticipated_affect = dat_user['anticipated_affect_norm'].to_numpy()
    anticipated_affect_yesterday = dat_user['anticipated_affect_yesterday_norm'].to_numpy()

    # Y_w: CAE
    CAE_avg = dat_user['CAE_avg_norm'].to_numpy()
    CAE_avg_lastweek = dat_user['CAE_avg_lastweek_norm'].to_numpy()
    # tilde Y_w
    CAE_short_avg = dat_user['CAE_short_avg_norm'].to_numpy()

    if 'perceived_utility_lastweek' not in dat_user.columns:
        raise KeyError(
            "df_fit is missing 'perceived_utility_lastweek'; "
            "run perceived_utility.py first to populate it."
        )
    perceived_utility_lastweek = dat_user['perceived_utility_lastweek'].to_numpy(dtype=float)
    if np.all(np.isnan(perceived_utility_lastweek)):
        raise ValueError(
            f"All 'perceived_utility_lastweek' values are NaN for user {userid}; "
            "re-run perceived_utility.py."
        )
    perceived_utility_lastweek = np.where(
        np.isnan(perceived_utility_lastweek),
        np.nanmean(perceived_utility_lastweek),
        perceived_utility_lastweek,
    )

    # extract the predictors
    Intercept = np.ones(len(fourSC))

    yesterday_step_count = dat_user['YesterdayStepCount_norm'].to_numpy()

    prior2hour_step_count = dat_user['prior2hour_step_norm'].to_numpy()
    ema_prior2hour_step_count = dat_user['EMA_Prior2HourStepCount_norm'].to_numpy()

    
    recent_burden = dat_user['recent_burden_norm'].to_numpy()

    active_status = dat_user['active_status'].to_numpy()
    active_status_fraction_7days = dat_user['active_status_fraction_7days'].to_numpy()
    seven_day_pageview_count = dat_user['Past7DaysPageviewEMA_norm'].to_numpy()
    past7days_hourly_pageview_count = dat_user['Past7DaysHourlyPageviewEMA_norm'].to_numpy()
    
    past7days_morning_wearing = dat_user['past7days_morning_wearing'].to_numpy()

    ws_interaction = dat_user['Interacted_walk'].to_numpy()
    Interacted_7d_walk = dat_user['Interacted_7d_walk'].to_numpy()

    WalkingSuggestion = dat_user['WalkingSuggestion'].to_numpy()

    is_weekend = dat_user['is_weekend'].to_numpy(dtype=float)
    decision_time = dat_user['DecisionTime'].to_numpy()
    
    
    # fill in missing values (NAN) with mean for predictors except for FourSC and Intercept
    fourSC_lag1 = fill_nan_with_mean(fourSC_lag1)
    yesterday_step_count = fill_nan_with_mean(yesterday_step_count)
    prior2hour_step_count_filled = fill_nan_with_mean(prior2hour_step_count)
    ema_prior2hour_step_count = fill_nan_with_mean(ema_prior2hour_step_count)
    active_status_fraction_7days = fill_nan_with_mean(active_status_fraction_7days)

    recent_burden = fill_nan_with_mean(recent_burden)
    
    seven_day_pageview_count = fill_nan_with_mean(seven_day_pageview_count)
    past7days_hourly_pageview_count = fill_nan_with_mean(past7days_hourly_pageview_count)
    past7days_morning_wearing = fill_nan_with_mean(past7days_morning_wearing)

    # Delivered-only lag; NaN when the prior 14 slots had no sends.
    Interacted_7d_walk = fill_nan_with_mean(Interacted_7d_walk)
    anticipated_affect_yesterday = fill_nan_with_mean(anticipated_affect_yesterday)
    
    CAE_avg_lastweek = fill_nan_with_mean(CAE_avg_lastweek)

    # Per calendar day: AM/PM walking suggestions (rows ordered AM then PM for each Date).
    Walking_pair = WalkingSuggestion.reshape(-1, 2)
    ws_morning_day = Walking_pair[:, 0]
    ws_afternoon_day = Walking_pair[:, 1]
    idx_morning = decision_time == 0

    ###### fit the models ######

    #### Models 1-3: prior-2h steps, active status, suggestion interaction ####
    # Participant coefficients = posterior mean of beta + b_i from the joint
    # hierarchical fits above (a participant without observations of an
    # outcome gets the population coefficients).
    designs = other_variable_designs(dat_user)

    X_p2h, prior2hour_step_count, idx_obs_prior2hour_step_count = designs["prior2hour"]
    theta_prior2hour_step_count_mean = participant_theta(mixedlm_prior2hour, userid)
    pred_prior2hour_step_count = X_p2h @ theta_prior2hour_step_count_mean
    resid_prior2hour_step_count = np.full(len(prior2hour_step_count), np.nan)
    resid_prior2hour_step_count[idx_obs_prior2hour_step_count] = (
        prior2hour_step_count[idx_obs_prior2hour_step_count]
        - pred_prior2hour_step_count[idx_obs_prior2hour_step_count]
    )

    X_ad, active_status, idx_obs_active_status = designs["active_status"]
    theta_active_status_mean = participant_theta(mixedlm_active_status, userid)
    pred_active_status = _sigmoid_np(X_ad @ theta_active_status_mean)
    resid_active_status = np.full(len(active_status), np.nan)
    resid_active_status[idx_obs_active_status] = (
        active_status[idx_obs_active_status] - pred_active_status[idx_obs_active_status]
    )

    X_as, ws_interaction, idx_obs_ws_interaction = designs["ws_interaction"]
    theta_ws_interaction_mean = participant_theta(mixedlm_ws_interaction, userid)
    pred_ws_interaction = _sigmoid_np(X_as @ theta_ws_interaction_mean)
    resid_ws_interaction = np.full(len(ws_interaction), np.nan)
    resid_ws_interaction[idx_obs_ws_interaction] = (
        ws_interaction[idx_obs_ws_interaction] - pred_ws_interaction[idx_obs_ws_interaction]
    )

    #### Model 4: FourSC model ####
    fourSC_cond = np.stack([
        Intercept,
        fourSC_lag1,
        yesterday_step_count,
        prior2hour_step_count_filled,
        recent_burden, seven_day_pageview_count, past7days_morning_wearing, 
        Interacted_7d_walk,
        anticipated_affect_yesterday,
        active_status_fraction_7days,
        is_weekend, decision_time,
        perceived_utility_lastweek,
        CAE_avg_lastweek, 
        WalkingSuggestion, WalkingSuggestion * yesterday_step_count,
        WalkingSuggestion * prior2hour_step_count_filled,
        WalkingSuggestion * recent_burden,
        WalkingSuggestion * seven_day_pageview_count,
        WalkingSuggestion * past7days_morning_wearing,
        WalkingSuggestion * Interacted_7d_walk,
        WalkingSuggestion * anticipated_affect_yesterday,
        WalkingSuggestion * decision_time,
        WalkingSuggestion * perceived_utility_lastweek,
        WalkingSuggestion * CAE_avg_lastweek
    ], axis=1)

    # check if there are any NaN values in the condition matrix
    if np.isnan(fourSC_cond).any():
        print(f"NaN values in condition matrix for user {userid}")
        print(np.isnan(fourSC_cond))

    # filter out rows where FourSC is NaN
    idx_obs_fourSC = ~np.isnan(fourSC)
    FourSC_obs = fourSC[idx_obs_fourSC]

    # theta_fourSC_mean = population beta + this user's posterior random
    # effect from the pooled Bayesian fit above (mixedlm_fourSC), instead of
    # an unregularized per-user RidgeCV. No ridge alpha to report anymore.
    alpha_fourSC_l2 = None
    beta_fourSC = np.asarray(mixedlm_fourSC.fe_params, dtype=float)
    b_fourSC = get_user_random_effect(
        result=mixedlm_fourSC,
        userid=int(userid),
        p=len(THETA_FOURSC_NAMES),
    )
    theta_fourSC_mean = beta_fourSC + b_fourSC
    pred_fourSC = fourSC_cond @ theta_fourSC_mean

    resid_obs_fourSC = FourSC_obs - pred_fourSC[idx_obs_fourSC]
    # fill in the full residual array with NaN for unobserved
    resid_fourSC = np.full_like(fourSC, np.nan, dtype=float)
    resid_fourSC[idx_obs_fourSC] = resid_obs_fourSC

    print(f"The fit of fourSC is good for user {userid}")


    #### Model 5: Anticipated affect model ####
    # Daily outcome: one row per calendar day (morning row). Predictors from that row except treatment,
    # which enters as both ws_morning_day and ws_afternoon_day for that day.
    _am = idx_morning
    anticipated_affect_cond_day = np.stack([
        Intercept[_am],
        anticipated_affect_yesterday[_am],
        active_status_fraction_7days[_am],
        is_weekend[_am],
        perceived_utility_lastweek[_am],
        CAE_avg_lastweek[_am],
        recent_burden[_am],
        ws_morning_day,
        ws_afternoon_day,
        ws_morning_day * perceived_utility_lastweek[_am],
        ws_afternoon_day * perceived_utility_lastweek[_am],
        ws_morning_day * CAE_avg_lastweek[_am],
        ws_afternoon_day * CAE_avg_lastweek[_am],
        ws_morning_day * recent_burden[_am],
        ws_afternoon_day * recent_burden[_am],
        ws_morning_day * active_status_fraction_7days[_am],
        ws_afternoon_day * active_status_fraction_7days[_am],
    ], axis=1)

    y_antic_day = anticipated_affect[_am]
    idx_daily_antic = ~np.isnan(y_antic_day)

    # theta_anticipated_affect_mean = population beta + this user's posterior
    # random effect from the pooled Bayesian fit above (mixedlm_antic),
    # instead of an unregularized per-user Ridge/RidgeCV. No ridge alpha to
    # report anymore, and no separate n_obs branching needed: the population
    # prior already gives a sane fallback for users with very few days.
    alpha_anticipated_affect_l2 = None
    beta_antic = np.asarray(mixedlm_antic.fe_params, dtype=float)
    b_antic = get_user_random_effect(
        result=mixedlm_antic,
        userid=int(userid),
        p=len(THETA_ANTIC_NAMES),
    )
    theta_anticipated_affect_mean = beta_antic + b_antic
    pred_anticipated_affect = anticipated_affect_cond_day @ theta_anticipated_affect_mean

    resid_obs_anticipated_affect = (
        y_antic_day[idx_daily_antic] - pred_anticipated_affect[idx_daily_antic]
    )
    resid_anticipated_affect = np.full_like(y_antic_day, np.nan, dtype=float)
    resid_anticipated_affect[idx_daily_antic] = resid_obs_anticipated_affect

    _obs_antic = resid_anticipated_affect[~np.isnan(resid_anticipated_affect)]

    print(f"The fit of anticipated_affect is good for user {userid}")


    #### Model 10: CAE model #### 
#### Model 10: CAE model, Bayesian random coefficient b_i for every variable ####

    K = 14

    CAE_avg_sw = CAE_avg.reshape(-1, K)[:, 0]
    CAE_avg_lastweek_sw = CAE_avg_lastweek.reshape(-1, K)[:, 0]
    Intercept_sw = np.ones(len(CAE_avg_sw))

    foursc_wk = fourSC.reshape(-1, K)
    af = anticipated_affect.reshape(-1, K)
    foursc_e, antic_e = cae_mediator_ewma_rows(foursc_wk, af)

    CAE_cond = np.column_stack([
        Intercept_sw,
        CAE_avg_lastweek_sw,
        foursc_e,
        antic_e,
    ])
    if CAE_cond.shape[1] != len(THETA_CAE_NAMES):
        raise RuntimeError(
            f"User {userid}: CAE_cond has {CAE_cond.shape[1]} columns, "
            f"but THETA_CAE_NAMES has {len(THETA_CAE_NAMES)} names."
        )

    beta_CAE = np.asarray(mixedlm_CAE.fe_params, dtype=float)

    b_CAE = get_user_random_effect(
        result=mixedlm_CAE,
        userid=int(userid),
        p=len(THETA_CAE_NAMES),
    )

    theta_CAE_mean = beta_CAE + b_CAE

    pred_CAE = CAE_cond @ theta_CAE_mean

    idx_obs_CAE = ~np.isnan(CAE_avg_sw)

    _cae_lw_raw_first = float(dat_user['CAE_avg_lastweek_norm'].to_numpy(dtype=float)[0])
    resid_CAE = np.full_like(CAE_avg_sw, np.nan, dtype=float)

    if np.any(idx_obs_CAE):
        resid_CAE[idx_obs_CAE] = CAE_avg_sw[idx_obs_CAE] - pred_CAE[idx_obs_CAE]
        sigma2_CAE_mean = float(np.nanvar(resid_CAE[idx_obs_CAE]))
    else:
        sigma2_CAE_mean = float(mixedlm_CAE.scale)

    if not np.isfinite(sigma2_CAE_mean) or sigma2_CAE_mean <= 0:
        sigma2_CAE_mean = float(mixedlm_CAE.scale)

    # No separate ridge alpha anymore: shrinkage is baked into the Bayesian
    # priors used for the joint fit (see CAE_BETA_PRIOR_SD above).
    alpha_CAE_l2 = None


    print(f"The full random-coefficient Bayesian CAE fit is ready for user {userid}")

    #### Model 13: CAE short average model #### 

    CAE_short_avg_sw = CAE_short_avg.reshape(-1, K)[:, 0]
    CAE_avg_sw_filled = fill_nan_with_mean(CAE_avg_sw)

    CAE_short_avg_cond = np.stack([
        np.ones(len(CAE_avg_sw_filled)),
        CAE_avg_sw_filled,
    ], axis=1)

    beta_CAE_short = np.asarray(mixedlm_CAE_short.fe_params, dtype=float)

    b_CAE_short = get_user_random_effect(
        result=mixedlm_CAE_short,
        userid=int(userid),
        p=len(THETA_CAE_SHORT_AVG_NAMES),
    )

    theta_CAE_short_avg_mean = beta_CAE_short + b_CAE_short

    pred_CAE_short_avg = CAE_short_avg_cond @ theta_CAE_short_avg_mean

    idx_obs_CAE_short_avg = ~np.isnan(CAE_short_avg_sw)
    resid_CAE_short_avg = np.full_like(CAE_short_avg_sw, np.nan, dtype=float)

    if np.any(idx_obs_CAE_short_avg):
        resid_CAE_short_avg[idx_obs_CAE_short_avg] = (
            CAE_short_avg_sw[idx_obs_CAE_short_avg]
            - pred_CAE_short_avg[idx_obs_CAE_short_avg]
        )
        sigma2_CAE_short_avg_mean = float(
            np.nanvar(resid_CAE_short_avg[idx_obs_CAE_short_avg])
        )
    else:
        sigma2_CAE_short_avg_mean = float(mixedlm_CAE_short.scale)

    if not np.isfinite(sigma2_CAE_short_avg_mean) or sigma2_CAE_short_avg_mean <= 0:
        sigma2_CAE_short_avg_mean = float(mixedlm_CAE_short.scale)

    alpha_CAE_short_avg_l2 = None

    # Exact baseline backcast from the first observed full CAE. c_2..c_12
    # are the panel intercepts. c_1 uses the week-1 EWMAs in
    # week1_cae_inputs.csv, from the same mediator function as c_2..c_12.
    # No week-1 row falls back to c_2.
    _c_2_12 = (
        theta_CAE_mean[0]
        + theta_CAE_mean[2] * foursc_e
        + theta_CAE_mean[3] * antic_e
    )
    _w1 = _week1_inputs.get(int(userid))
    if _w1 is None:
        _c1 = np.nan
        _week1_reason = "no week1_cae_inputs row"
    else:
        _c1 = c1_from_week1_ewma(
            theta_CAE_mean, _w1["foursc_ewma"], _w1["antic_ewma"]
        )
        _week1_reason = _w1["reason"] or ("irregular" if _w1["irregular"] else "")
    if not np.isfinite(_c1):
        _week1_c2_users.append(int(userid))
        print(
            f"User {userid}: week-1 CAE drift unavailable ({_week1_reason or 'non-finite EWMA'}); "
            "baseline uses c_2"
        )
    _c_panel = np.concatenate(([_c1], np.asarray(_c_2_12, dtype=float).ravel()))
    _y_all = np.concatenate(([_cae_lw_raw_first], CAE_avg_sw))
    cae_baseline_old, _ = impute_baseline_cae(
        theta_CAE=theta_CAE_mean,
        c_panel=np.concatenate(([np.nan], np.asarray(_c_2_12, dtype=float).ravel())),
        y_obs=_y_all,
    )
    cae_baseline_norm, cae_baseline_source = impute_baseline_cae(
        theta_CAE=theta_CAE_mean,
        c_panel=_c_panel,
        y_obs=_y_all,
    )
    _full_obs_weeks = np.flatnonzero(np.isfinite(_y_all)) + 1
    _first_full_week = (
        int(_full_obs_weeks[0]) if _full_obs_weeks.size else np.nan
    )
    cae_baseline_rows.append(
        {
            "uid": int(userid),
            "source": cae_baseline_source,
            "k": _first_full_week,
            "old_value_norm": float(cae_baseline_old),
            "new_value_norm": float(cae_baseline_norm),
            "new_value_raw": (
                float(std_params["CAE_avg_shift"])
                + float(std_params["CAE_avg_scale"]) * float(cae_baseline_norm)
            ),
        }
    )
    if cae_baseline_source != "observed":
        print(
            f"User {userid}: no week-1 CAE; baseline imputed = "
            f"{cae_baseline_norm:.3f} (norm scale, {cae_baseline_source})"
        )

    print(f"The full random-coefficient Bayesian CAE_short_avg fit is ready for user {userid}")

    #### Store the parameters and residuals #### 
 
    digits = 3

    env_para = {
        "theta_prior2hour_step_count": json_float_list(theta_prior2hour_step_count_mean, digits),
        "theta_prior2hour_step_count_names": THETA_PRIOR2HOUR_STEP_COUNT_NAMES,

        "theta_active_status": json_float_list(theta_active_status_mean, digits),
        "theta_active_status_names": THETA_ACTIVE_STATUS_NAMES,

        "theta_ws_interaction": json_float_list(theta_ws_interaction_mean, digits),
        "theta_ws_interaction_names": THETA_WS_INTERACTION_NAMES,

        "theta_fourSC": json_float_list(theta_fourSC_mean, digits),
        "theta_fourSC_names": THETA_FOURSC_NAMES,

        "theta_antic": json_float_list(theta_anticipated_affect_mean, digits),
        "theta_antic_names": THETA_ANTIC_NAMES,

        "cae_sign_prior_mode": SIGN_PRIOR_MODE,
        "theta_CAE": json_float_list(theta_CAE_mean, digits),
        "theta_CAE_names": THETA_CAE_NAMES,
        "cae_baseline_norm": round(float(cae_baseline_norm), digits),
        "cae_baseline_source": cae_baseline_source,

        "theta_CAE_short_avg": json_float_list(theta_CAE_short_avg_mean, digits),
        "theta_CAE_short_avg_names": THETA_CAE_SHORT_AVG_NAMES,

        "resid_prior2hour_step_count": json_float_list(resid_prior2hour_step_count, digits),
        "resid_active_status": json_float_list(resid_active_status, digits),
        "resid_ws_interaction": json_float_list(resid_ws_interaction, digits),
        "resid_fourSC": json_float_list(resid_fourSC, digits),
        "resid_antic": json_float_list(resid_anticipated_affect, digits),
        "resid_CAE": json_float_list(resid_CAE, digits),
        "resid_CAE_short_avg": json_float_list(resid_CAE_short_avg, digits),

        "alpha_fourSC_l2": json_float(alpha_fourSC_l2, digits),
        "alpha_anticipated_affect_l2": json_float(alpha_anticipated_affect_l2, digits),
        "alpha_CAE_l2": json_float(alpha_CAE_l2, digits),
        "alpha_CAE_short_avg_l2": json_float(alpha_CAE_short_avg_l2, digits),
    }

    predicted = {
        "pred_prior2hour_step_count": json_float_list(pred_prior2hour_step_count, digits),
        "pred_active_status": json_float_list(pred_active_status, digits),
        "pred_ws_interaction": json_float_list(pred_ws_interaction, digits),
        "pred_fourSC": json_float_list(pred_fourSC, digits),
        "pred_antic": json_float_list(pred_anticipated_affect, digits),
        "pred_CAE": json_float_list(pred_CAE, digits),
        "pred_CAE_short_avg": json_float_list(pred_CAE_short_avg, digits),
    }

    # Sanity checks: theta length must match theta-name length.
    for key in [
        "theta_prior2hour_step_count",
        "theta_active_status",
        "theta_ws_interaction",
        "theta_fourSC",
        "theta_antic",
        "theta_CAE",
        "theta_CAE_short_avg",
    ]:
        n_theta = len(env_para[key])
        n_names = len(env_para[f"{key}_names"])
        if n_theta != n_names:
            raise RuntimeError(
                f"Participant {userid}: {key} length {n_theta} != "
                f"{key}_names length {n_names}"
            )

    p_env_path = Path(file_params_env_prefix + str(userid) + ".json")
    p_pred_path = Path(file_pred_prefix + str(userid) + ".json")

    merge_json_file(p_env_path, env_para)
    merge_json_file(p_pred_path, predicted)

if _week1_c2_users:
    print(
        "Week-1 CAE drift fell back to c_2 for "
        + ", ".join(str(u) for u in _week1_c2_users)
    )
else:
    print("Week-1 CAE drift used c_1 for every fitted user")

np.savetxt(file_user_ids, userid_all, fmt='%d')

# save df_fit.csv with the predicted columns
df_fit.loc[df_fit["ParticipantIdentifier"].isin(userid_all)].to_csv(
    work_folder / "df_fit_11week.csv",
    index=False,
)

cohort_check_dir = PROJECT_ROOT / "docs" / "cohort_checks_20261017"
cohort_check_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(cae_baseline_rows).sort_values("uid").to_csv(
    cohort_check_dir / "cae_baseline_imputation.csv",
    index=False,
    float_format="%.6f",
)

# Residual pools for the simulator's partial pooling are built by vani_env
# from all participants' params_env files (no separate population file).
_diag_path = work_folder / "testbed_diagnostics.json"
_diag_path.write_text(
    json.dumps({"models": _FIT_DIAGNOSTICS}, indent=2) + "\n",
    encoding="utf-8",
)
print(f"Wrote {_diag_path} ({len(_FIT_DIAGNOSTICS)} models)")

