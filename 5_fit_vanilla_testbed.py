"""Fit the remaining generative models and write ``env_para_vanilla/``.

Reads ``df_fit.csv`` (after script 4 has attached E_w). Fits, per participant
or as a hierarchical Bayesian model:

    4-hour step counts, anticipated affect, weekly CAE / short CAE,
    prior-2h steps, active status, walking-suggestion interaction.

Writes ``params_env_<uid>.json`` (mediator/outcome blocks and residuals),
``pred_<uid>.json``, ``user_ids.txt``, and ``population_residuals.json``.
Does not overwrite the E_w / PV / FW / PJ blocks from script 4.
Next: ``6_est_Ew_weights.py``.
"""
import json
import os
import shutil
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from patsy import dmatrix
from pathlib import Path
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeCV,
)

from vani_env import within_week_ewma


# %%
# read data
PROJECT_ROOT = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPR-MRT-Testbed")
COMBINED_DIR = Path(
    os.environ.get(
        "ADAPR_COMBINED_DIR",
        "/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/Xueqing",
    )
)
WORK_DIR = Path(os.environ.get("ADAPR_VANILLA_ENV_DIR", PROJECT_ROOT / "env_para_vanilla"))
WORK_DIR.mkdir(parents=True, exist_ok=True)


# Aliases for later cells that use `folder` / `work_folder` (Path so `/` joins work)
folder = COMBINED_DIR
work_folder = Path(WORK_DIR)

std_params_source = PROJECT_ROOT / "env_para_vanilla" / "std_params.json"
std_params_dest = work_folder / "std_params.json"
if (
    std_params_source.is_file()
    and not std_params_dest.is_file()
    and std_params_source.resolve() != std_params_dest.resolve()
):
    shutil.copy2(std_params_source, std_params_dest)

df_fit = pd.read_csv(folder / 'df_fit.csv')

file_params_env_prefix = str(work_folder / 'params_env_')
file_pred_prefix = str(work_folder / 'pred_')
file_user_ids = str(work_folder / 'user_ids.txt')

# Combiner already drops weeks 0 and 13. Vanilla fit keeps study weeks 2–12
# and renumbers them to 1–11.
df_fit = df_fit[df_fit["week"] > 1]
df_fit["week"] = df_fit["week"] - 1

# Re-index calendar covariates on the retained rows (after week drop/relabel).
VANILLA_DAY_RANGE = 84
VANILLA_WEEK_RANGE = 11
df_fit["Date"] = pd.to_datetime(df_fit["Date"], errors="coerce")
df_fit["day"] = (
    df_fit.groupby("ParticipantIdentifier", sort=False)["Date"]
    .transform(lambda d: (d - d.min()).dt.days)
)
df_fit["day_norm"] = (
    df_fit["day"] - (1 + VANILLA_DAY_RANGE) / 2
) / ((VANILLA_DAY_RANGE - 1) / 2)
df_fit["week_norm"] = (
    df_fit["week"] - (1 + VANILLA_WEEK_RANGE) / 2
) / ((VANILLA_WEEK_RANGE - 1) / 2)
# %%
# import warnings
# from sklearn.exceptions import UndefinedMetricWarning
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


def safe_nanmean(x, default=0.0):
    """Mean that stays finite when all values are missing."""
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return default
    return np.nanmean(x)

def _store_group_effects_as_series(effects_by_group):
    """Wrap a ``{group_key: 1d array}`` mapping as ``{group_key: pd.Series}``.

    Matches the ``.random_effects`` interface statsmodels' ``MixedLMResults``
    exposes, so :func:`get_user_random_effect` works unchanged.
    """
    return {key: pd.Series(np.asarray(vec, dtype=float)) for key, vec in effects_by_group.items()}


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
        sigma_eps = float(post["sigma_eps"].mean(("chain", "draw")))
        self.scale = sigma_eps ** 2

        b_mean = post["b"].mean(("chain", "draw")).values  # (n_groups, p)
        self.random_effects = _store_group_effects_as_series(
            {key: b_mean[i] for i, key in enumerate(self.group_keys)}
        )

        n_div = int(idata.sample_stats["diverging"].sum())
        rhat = az.rhat(idata, var_names=["beta", "sigma_re", "sigma_eps"])
        max_rhat = max(float(rhat[v].max()) for v in rhat.data_vars)
        self.n_divergences = n_div
        self.max_rhat = max_rhat
        self.converged = bool(n_div == 0 and max_rhat < 1.01)

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


def fit_bayesian_random_coef_model(
    y,
    X,
    groups,
    predictor_names,
    beta_prior_mu=0.0,
    beta_prior_sd=5.0,
    re_sd_prior_mode=0.15,
    sigma_eps_prior_sd=1.0,
    draws=2000,
    tune=3000,
    chains=4,
    target_accept=0.99,
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
    - ``re_sd_prior_mode`` sets a ``Gamma(shape=2, mode=re_sd_prior_mode)``
      prior on each random-effect SD. Unlike a Half-Normal/Half-Cauchy (the
      usual textbook default), a shape-2 Gamma has *zero density at exactly
      0*, so the posterior is pushed off the degenerate "same slope for every
      user" boundary without needing a hard-coded variance floor.

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

    group_keys, group_idx = np.unique(groups, return_inverse=True)
    n_groups = len(group_keys)

    beta_prior_mu = np.broadcast_to(np.asarray(beta_prior_mu, dtype=float), (p,)).copy()
    beta_prior_sd = np.broadcast_to(np.asarray(beta_prior_sd, dtype=float), (p,)).copy()
    re_sd_prior_mode = np.broadcast_to(np.asarray(re_sd_prior_mode, dtype=float), (p,)).copy()
    re_rate = 1.0 / np.maximum(re_sd_prior_mode, 1e-6)

    with pm.Model():
        beta = pm.Normal("beta", mu=beta_prior_mu, sigma=beta_prior_sd, shape=p)
        sigma_re = pm.Gamma("sigma_re", alpha=2.0, beta=re_rate, shape=p)
        b_raw = pm.Normal("b_raw", mu=0.0, sigma=1.0, shape=(n_groups, p))
        b = pm.Deterministic("b", b_raw * sigma_re[None, :])
        theta = beta[None, :] + b
        mu = pm.math.sum(X * theta[group_idx], axis=1)
        sigma_eps = pm.HalfNormal("sigma_eps", sigma=sigma_eps_prior_sd)
        pm.Normal("y_obs", mu=mu, sigma=sigma_eps, observed=y)

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
                  + week
                  + EWMA of 14 FourSC decision-slot summaries
                  + EWMA of 7 anticipated-affect daily summaries

    The two EWMA terms use the same ``gamma=6/7`` normalized discount as
    ``1_data_extraction._ewm_prior_rows`` (most recent observation last).

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

        # Keep only complete weeks of K decision rows.
        # Your current code uses reshape(-1, K), so this is needed.
        n_full_weeks = len(dat_user) // K
        if n_full_weeks == 0:
            continue

        dat_user = dat_user.iloc[: n_full_weeks * K].copy()

        # Decision-level arrays.
        CAE_avg = dat_user["CAE_avg_norm"].to_numpy(dtype=float)
        CAE_avg_lastweek = dat_user["CAE_avg_lastweek_norm"].to_numpy(dtype=float)
        week = dat_user["week_norm"].to_numpy(dtype=float)

        fourSC = dat_user["4hour_step_norm"].to_numpy(dtype=float)
        anticipated_affect = dat_user["anticipated_affect_norm"].to_numpy(dtype=float)

        # Match your current preprocessing.
        CAE_avg_lastweek = fill_nan_with_mean(CAE_avg_lastweek)

        # Convert decision-level rows to weekly rows.
        CAE_avg_sw = CAE_avg.reshape(-1, K)[:, 0]
        CAE_avg_lastweek_sw = CAE_avg_lastweek.reshape(-1, K)[:, 0]
        week_sw = week.reshape(-1, K)[:, 0]
        Intercept_sw = np.ones(len(week_sw))

        # FourSC: 14 decision slots per week.
        foursc_wk = fourSC.reshape(-1, K)
        mu_foursc = safe_nanmean(fourSC)
        foursc_wk = np.where(np.isnan(foursc_wk), mu_foursc, foursc_wk)

        # Anticipated affect: daily outcome repeated across AM/PM rows.
        # Convert 14 decision rows into 7 daily averages.
        af = anticipated_affect.reshape(-1, K)
        mu_antic = safe_nanmean(anticipated_affect)
        af = np.where(np.isnan(af), mu_antic, af)

        antic_wk = af.reshape(-1, 7, 2).mean(axis=2)
        antic_wk = np.where(np.isnan(antic_wk), mu_antic, antic_wk)

        foursc_e = np.array(
            [within_week_ewma(row) for row in foursc_wk],
            dtype=float,
        )
        antic_e = np.array(
            [within_week_ewma(row) for row in antic_wk],
            dtype=float,
        )

        X = np.column_stack(
            [
                Intercept_sw,
                CAE_avg_lastweek_sw,
                week_sw,
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

        # Keep only complete weeks of K decision rows.
        n_full_weeks = len(dat_user) // K
        if n_full_weeks == 0:
            continue

        dat_user = dat_user.iloc[: n_full_weeks * K].copy()

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
        seven_day_step_count_avg = fill_nan_with_mean(dat_user["EMA_StepCount_norm"].to_numpy())
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
                seven_day_step_count_avg,
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
    "seven_day_step_count_avg",
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
    "week",
    "fourSC_ewma",
    "anticipated_affect_ewma",
]

THETA_CAE_SHORT_AVG_NAMES = [
    "intercept",
    "CAE_avg",
]

alpha_l2_list = [1, 2, 5]  # floor at 1; drop weak penalties that inflate sparse cells
ncv = 5
seed = 2026

dat_user_all = []

# Keep only users with at least one observed CAE value.
# Users with no CAE observations lead to fallback-zero CAE models, which can
# create unrealistic downstream trajectories.
_all_userids = df_fit['ParticipantIdentifier'].unique()
_has_cae_obs = (
    df_fit.groupby('ParticipantIdentifier', sort=False)['CAE_avg_norm']
    .apply(lambda s: np.any(~np.isnan(s.to_numpy(dtype=float))))
)
userid_all = np.array(
    [uid for uid in _all_userids if bool(_has_cae_obs.get(uid, False))],
    dtype=int,
)
excluded_userids = np.array(
    [uid for uid in _all_userids if not bool(_has_cae_obs.get(uid, False))],
    dtype=int,
)
if excluded_userids.size > 0:
    print(
        "Excluding users with no CAE observations: "
        + ", ".join(str(int(u)) for u in excluded_userids)
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
#     below 1 (a loop gain >= 1 makes the simulated trajectory unstable), so
#     it's centered at 0.4 instead of 0.
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
# week and the intercept have no such prior belief, so they get
# wide/uninformative priors and are essentially estimated by the data alone.
CAE_BETA_PRIOR_MU = np.array([0.0, 0.4, 0.0, 0.15, 0.3])
CAE_BETA_PRIOR_SD = np.array([5.0, 0.15, 1.0, 0.15, 0.15])
CAE_RE_SD_PRIOR_MODE = np.array([0.1, 0.15, 0.1, 0.15, 0.15])

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
    re_sd_prior_mode=CAE_RE_SD_PRIOR_MODE,
)

print(mixedlm_CAE.summary())
print(f"CAE Bayesian fit converged: {mixedlm_CAE.converged}")

cae_random_effect_cols = list(range(len(THETA_CAE_NAMES)))

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

cae_short_random_effect_cols = list(range(len(THETA_CAE_SHORT_AVG_NAMES)))

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
ANTIC_RE_SD_PRIOR_MODE = np.full(len(THETA_ANTIC_NAMES), 0.15)
ANTIC_BETA_PRIOR_SD = np.full(len(THETA_ANTIC_NAMES), 3.0)
ANTIC_BETA_PRIOR_MU = np.zeros(len(THETA_ANTIC_NAMES))
for _name in ("A0_morning_by_recent_burden", "A1_afternoon_by_recent_burden"):
    ANTIC_BETA_PRIOR_MU[THETA_ANTIC_NAMES.index(_name)] = -0.05
    ANTIC_BETA_PRIOR_SD[THETA_ANTIC_NAMES.index(_name)] = 0.04
    ANTIC_RE_SD_PRIOR_MODE[THETA_ANTIC_NAMES.index(_name)] = 0.05
ANTIC_BETA_PRIOR_MU[THETA_ANTIC_NAMES.index("recent_burden")] = -0.05
ANTIC_BETA_PRIOR_SD[THETA_ANTIC_NAMES.index("recent_burden")] = 0.03
ANTIC_RE_SD_PRIOR_MODE[THETA_ANTIC_NAMES.index("recent_burden")] = 0.05
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
    ANTIC_BETA_PRIOR_SD[THETA_ANTIC_NAMES.index(_name)] = 0.08

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
    re_sd_prior_mode=ANTIC_RE_SD_PRIOR_MODE,
    draws=1000,
    tune=1500,
    target_accept=0.95,
)

print(mixedlm_antic.summary())
print(f"Anticipated-affect Bayesian fit converged: {mixedlm_antic.converged}")

FOURSC_RE_SD_PRIOR_MODE = np.full(len(THETA_FOURSC_NAMES), 0.15)
FOURSC_BETA_PRIOR_SD = np.full(len(THETA_FOURSC_NAMES), 3.0)
FOURSC_BETA_PRIOR_MU = np.zeros(len(THETA_FOURSC_NAMES))
FOURSC_BETA_PRIOR_MU[THETA_FOURSC_NAMES.index("recent_burden")] = -0.05
FOURSC_BETA_PRIOR_SD[THETA_FOURSC_NAMES.index("recent_burden")] = 0.03
FOURSC_RE_SD_PRIOR_MODE[THETA_FOURSC_NAMES.index("recent_burden")] = 0.05
# This interaction is the one that most directly let a constant-suggestion
# policy compound unchecked for a couple of outlier users: one came back at
# +0.057 against this -0.05 population mean under the old wide re_sd prior
# (mode=0.15), i.e. that person's fitted dynamics said each additional
# suggestion got *more* effective as burden accumulated. Tighter beta_sd and
# re_sd_prior_mode here don't rule that out, but make it harder for a
# handful of noisy per-user rows to produce it.
FOURSC_BETA_PRIOR_MU[THETA_FOURSC_NAMES.index("WalkingSuggestion_by_recent_burden")] = -0.05
FOURSC_BETA_PRIOR_SD[THETA_FOURSC_NAMES.index("WalkingSuggestion_by_recent_burden")] = 0.04
FOURSC_RE_SD_PRIOR_MODE[THETA_FOURSC_NAMES.index("WalkingSuggestion_by_recent_burden")] = 0.05
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
    FOURSC_BETA_PRIOR_SD[THETA_FOURSC_NAMES.index(_name)] = 0.08

fourSC_bayes_df = build_fourSC_bayes_data(df_fit, userid_all)
X_fourSC_obs = fourSC_bayes_df[THETA_FOURSC_NAMES].to_numpy(dtype=float)
y_fourSC_obs = fourSC_bayes_df["FourSC"].to_numpy(dtype=float)
groups_fourSC_obs = fourSC_bayes_df["ParticipantIdentifier"].to_numpy()

mixedlm_fourSC = fit_bayesian_random_coef_model(
    y=y_fourSC_obs,
    X=X_fourSC_obs,
    groups=groups_fourSC_obs,
    predictor_names=list(THETA_FOURSC_NAMES),
    beta_prior_mu=FOURSC_BETA_PRIOR_MU,
    beta_prior_sd=FOURSC_BETA_PRIOR_SD,
    re_sd_prior_mode=FOURSC_RE_SD_PRIOR_MODE,
    draws=1000,
    tune=2000,
    target_accept=0.97,
)

print(mixedlm_fourSC.summary())
print(f"FourSC Bayesian fit converged: {mixedlm_fourSC.converged}")

theta_pageview_list = []
theta_fitbitwearing_list = []
theta_eodcomplete_list = []

# Population-level residual pools (observed values only, across all users) for
# the streams that feed the STE outcome most directly: CAE, CAE_short, antic,
# fourSC. Users with very few of their own observations (e.g. n_obs_CAE == 1)
# have a degenerate own residual pool — there's nothing to resample from, so
# every noise mode (sequential/random/ar1) collapses to "replay that one
# value" for them. ``vani_env.Env`` blends in these population pools (see
# ``POPULATION_FALLBACK_MIN_OBS``) so sparse-data users still get plausible
# simulated variability instead of an artificially smooth trajectory.
population_resid_pools = {
    "resid_CAE_population": [],
    "resid_CAE_short_population": [],
    "resid_antic_population": [],
    "resid_fourSC_population": [],
}

for i, userid in enumerate(userid_all):
    dat_user = df_fit[df_fit['ParticipantIdentifier'] == userid].copy()
    dat_user = dat_user.sort_values(['Date', 'DecisionTime'], na_position='last').reset_index(drop=True)

    # fill in initial values (last week's affective association, and perceived utility)
    # set the first 0-13 days to 0
    # if len(dat_user) > 0:
    #     dat_user.loc[dat_user.index[:14], 'week_present_lastweek'] = 0
    #     dat_user.loc[dat_user.index[:14], 'CAE_avg_lastweek_norm'] = 0
    #     dat_user.loc[dat_user.index[:14], 'perceived_utility_lastweek_norm'] = 0
    #     dat_user.loc[dat_user.index[:2], 'view_status_lastdecision'] = 0

    # extract the response
    # M^Y_{w,d,t}
    fourSC = dat_user['4hour_step_norm'].to_numpy()
    fourSC_lag1 = dat_user['FourSC_lag1'].to_numpy()


    # M^Y_{w,d}
    anticipated_affect = dat_user['anticipated_affect_norm'].to_numpy()
    anticipated_affect_yesterday = dat_user['anticipated_affect_yesterday_norm'].to_numpy()
    # anticipated_affect_lag1 = dat_user['anticipated_affect_lag1'].to_numpy()

    # Y_w: CAE
    CAE_avg = dat_user['CAE_avg_norm'].to_numpy()
    # print(userid, CAE_avg_norm.shape)
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
    seven_day_step_count_avg = dat_user['EMA_StepCount_norm'].to_numpy()

    prior2hour_step_count = dat_user['prior2hour_step_norm'].to_numpy()
    ema_prior2hour_step_count = dat_user['EMA_Prior2HourStepCount_norm'].to_numpy()

    
    recent_burden = dat_user['recent_burden_norm'].to_numpy()

    active_status = dat_user['active_status'].to_numpy()
    active_status_fraction_7days = dat_user['active_status_fraction_7days'].to_numpy()
    DailyPageviewCount = dat_user['DailyPageviewCount_norm'].to_numpy()
    seven_day_pageview_count = dat_user['Past7DaysPageviewEMA_norm'].to_numpy()
    past7days_hourly_pageview_count = dat_user['Past7DaysHourlyPageviewEMA_norm'].to_numpy()
    
    past7days_morning_wearing = dat_user['past7days_morning_wearing'].to_numpy()

    ws_interaction = dat_user['Interacted_walk'].to_numpy()
    Interacted_7d_walk = dat_user['Interacted_7d_walk'].to_numpy()

    week_present = dat_user['week_present'].to_numpy()
    week_present_lastweek = dat_user['week_present_lastweek'].to_numpy()
    
    WalkingSuggestion = dat_user['WalkingSuggestion'].to_numpy()
    # WalkingSuggestion_lag1 = dat_user['WalkingSuggestion_lag1'].to_numpy()
    # planning_prompt = dat_user['planning_prompt'].to_numpy()
    # yesterday_planning_prompt = dat_user['yesterday_planning_prompt'].to_numpy()

    is_weekend = dat_user['is_weekend'].to_numpy(dtype=float)
    day = dat_user['day_norm'].to_numpy()
    week = dat_user['week_norm'].to_numpy() #TODO: check this
    decision_time = dat_user['DecisionTime'].to_numpy()
    
    

    # fill in missing values (NAN) with mean for predictors except for FourSC and Intercept
    fourSC_lag1 = fill_nan_with_mean(fourSC_lag1)
    yesterday_step_count = fill_nan_with_mean(yesterday_step_count)
    seven_day_step_count_avg = fill_nan_with_mean(seven_day_step_count_avg)
    prior2hour_step_count_filled = fill_nan_with_mean(prior2hour_step_count)
    ema_prior2hour_step_count = fill_nan_with_mean(ema_prior2hour_step_count)
    active_status_fraction_7days = fill_nan_with_mean(active_status_fraction_7days)

    recent_burden = fill_nan_with_mean(recent_burden)
    
    seven_day_pageview_count = fill_nan_with_mean(seven_day_pageview_count)
    past7days_hourly_pageview_count = fill_nan_with_mean(past7days_hourly_pageview_count)
    past7days_morning_wearing = fill_nan_with_mean(past7days_morning_wearing)

    Interacted_7d_walk = fill_nan_with_mean(Interacted_7d_walk)
    # anticipated_affect = np.where(np.isnan(anticipated_affect), np.nanmean(anticipated_affect), anticipated_affect)
    anticipated_affect_yesterday = fill_nan_with_mean(anticipated_affect_yesterday)
    
    CAE_avg_lastweek = fill_nan_with_mean(CAE_avg_lastweek)

    ws_morning = WalkingSuggestion * (1.0 - decision_time)
    ws_afternoon = WalkingSuggestion * decision_time

    # Per calendar day: AM/PM walking suggestions (rows ordered AM then PM for each Date).
    Walking_pair = WalkingSuggestion.reshape(-1, 2)
    ws_morning_day = Walking_pair[:, 0]
    ws_afternoon_day = Walking_pair[:, 1]
    idx_morning = decision_time == 0

    ###### fit the models ######

    #### Model 1: Prior 2-hour step count model #### 

    prior2hour_step_count_cond = np.stack([
        Intercept,
        ema_prior2hour_step_count,
        is_weekend, decision_time
    ], axis=1)
    #filter out rows where prior2hour_step_count is NaN
    idx_obs_prior2hour_step_count = ~np.isnan(prior2hour_step_count)
    prior2hour_step_count_cond_obs = prior2hour_step_count_cond[idx_obs_prior2hour_step_count, :]
    prior2hour_step_count_obs = prior2hour_step_count[idx_obs_prior2hour_step_count]
    n_obs_prior2hour_step_count = int(idx_obs_prior2hour_step_count.sum())

    if n_obs_prior2hour_step_count >= 2:
        cv_prior2hour_step_count = min(5, n_obs_prior2hour_step_count)
        model_prior2hour_step_count = RidgeCV(
            alphas=alpha_l2_list,
            fit_intercept=False,
            cv=cv_prior2hour_step_count,
            scoring="neg_mean_squared_error",
        )
        model_prior2hour_step_count.fit(
            prior2hour_step_count_cond_obs,
            prior2hour_step_count_obs,
        )
        alpha_prior2hour_step_count_l2 = model_prior2hour_step_count.alpha_
        theta_prior2hour_step_count_mean = model_prior2hour_step_count.coef_
        pred_prior2hour_step_count = model_prior2hour_step_count.predict(
            prior2hour_step_count_cond
        )
    elif n_obs_prior2hour_step_count == 1:
        print(f"fallback to fixed alpha for prior2hour_step_count model for user {userid}")
        alpha_prior2hour_step_count_l2 = 1.0
        model_prior2hour_step_count = Ridge(
            alpha=alpha_prior2hour_step_count_l2,
            fit_intercept=False,
        )
        model_prior2hour_step_count.fit(
            prior2hour_step_count_cond_obs,
            prior2hour_step_count_obs,
        )
        theta_prior2hour_step_count_mean = model_prior2hour_step_count.coef_
        pred_prior2hour_step_count = model_prior2hour_step_count.predict(
            prior2hour_step_count_cond
        )
    else:
        print(f"no observed prior2hour_step_count values for user {userid}; using zero fallback")
        alpha_prior2hour_step_count_l2 = float(alpha_l2_list[0])
        theta_prior2hour_step_count_mean = np.zeros(
            prior2hour_step_count_cond.shape[1],
            dtype=float,
        )
        pred_prior2hour_step_count = np.zeros(len(prior2hour_step_count), dtype=float)

    resid_obs_prior2hour_step_count = (
        prior2hour_step_count_obs
        - pred_prior2hour_step_count[idx_obs_prior2hour_step_count]
    )
    
    resid_prior2hour_step_count = np.full_like(prior2hour_step_count, np.nan, dtype=float)
    resid_prior2hour_step_count[idx_obs_prior2hour_step_count] = resid_obs_prior2hour_step_count
    sigma2_prior2hour_step_count_mean = (
        np.var(resid_obs_prior2hour_step_count)
        if resid_obs_prior2hour_step_count.size > 0
        else 0.0
    )

    # print that this fit is good
    print(f"The fit of prior2hour_step_count is good for user {userid}")

    #### Model 2: Active status model ####
    # P(active_status = 1 | prior 7-day active fraction, is_weekend); morning rows with answered survey
    active_status_cond = np.stack([
        Intercept,
        active_status_fraction_7days,
        is_weekend,
    ], axis=1)
    Cs_active_status = np.sort(1.0 / np.asarray(alpha_l2_list, dtype=float))

    idx_obs_active_status = (decision_time == 0) & ~np.isnan(active_status)
    cv_active_status = min(5, int(idx_obs_active_status.sum()))
    active_status_cond_obs = active_status_cond[idx_obs_active_status, :]
    active_status_obs = active_status[idx_obs_active_status]

    has_active_status_obs = active_status_obs.size > 0
    if has_active_status_obs and np.var(active_status_obs.astype(float)) > 0:
        n0, n1 = int(np.sum(active_status_obs == 0)), int(np.sum(active_status_obs == 1))
        min_class = min(n0, n1)
        cv_as = min(cv_active_status, min_class)
        if cv_as >= 2:
            model_active_status = LogisticRegressionCV(
                Cs=Cs_active_status,
                cv=cv_as,
                penalty="l2",
                solver="lbfgs",
                fit_intercept=False,
                scoring="neg_log_loss",
                max_iter=5000,
                random_state=seed,
            )
            model_active_status.fit(active_status_cond_obs, active_status_obs)
            C_sel = float(model_active_status.C_[0])
        else:
            C_sel = float(Cs_active_status[len(Cs_active_status) // 2])
            model_active_status = LogisticRegression(
                penalty="l2",
                C=C_sel,
                solver="lbfgs",
                fit_intercept=False,
                max_iter=5000,
                random_state=seed,
            )
            model_active_status.fit(active_status_cond_obs, active_status_obs)
        alpha_active_status_l2 = 1.0 / C_sel
        theta_active_status_mean = model_active_status.coef_.ravel()
        pred_active_status = model_active_status.predict_proba(active_status_cond)[:, 1]
        resid_obs_active_status = (
            active_status_obs.astype(float)
            - pred_active_status[idx_obs_active_status]
        )
        resid_active_status = np.full_like(active_status, np.nan, dtype=float)
        resid_active_status[idx_obs_active_status] = resid_obs_active_status
        sigma2_active_status_mean = np.var(resid_obs_active_status)
    else:
        print(f"the variance of active_status is 0 for user {userid}")
        const = float(safe_nanmean(active_status_obs.astype(float), default=np.nan))
        alpha_active_status_l2 = float(alpha_l2_list[0])
        theta_active_status_mean = np.zeros(active_status_cond.shape[1])
        if not np.isfinite(const):
            # No observed active_status rows for this user: fall back to a
            # neutral intercept logit(0.5) = 0 (P(active) = 0.5) instead of NaN.
            const = 0.5
            theta_active_status_mean[0] = 0.0
        elif const <= 0.0:
            theta_active_status_mean[0] = -8.0
        elif const >= 1.0:
            theta_active_status_mean[0] = 8.0
        else:
            theta_active_status_mean[0] = float(np.log(const / (1.0 - const)))
        pred_active_status = np.full(len(active_status), const, dtype=float)
        resid_active_status = np.full_like(active_status, np.nan, dtype=float)
        resid_active_status[idx_obs_active_status] = 0.0
        sigma2_active_status_mean = 0.0

    print(f"The fit of active_status is good for user {userid}")

    #### Model 3: Walking suggestion interaction model ####

    # P(WalkingSuggestion = 1 | context): L2 logistic CV (rows with observed WS and predictors)
    ws_interaction_cond = np.stack([
        Intercept,
        Interacted_7d_walk,
        is_weekend, decision_time
    ], axis=1)
    Cs_ws_interaction = np.sort(1.0 / np.asarray(alpha_l2_list, dtype=float))
    idx_obs_ws_interaction = ~np.isnan(ws_interaction)
    cv_ws_interaction = min(5, int(idx_obs_ws_interaction.sum()))
    ws_interaction_cond_obs = ws_interaction_cond[idx_obs_ws_interaction, :]
    ws_interaction_obs =  ws_interaction[idx_obs_ws_interaction]


    has_ws_interaction_obs = ws_interaction_obs.size > 0
    if has_ws_interaction_obs and np.var(ws_interaction_obs.astype(float)) > 0:
        n0, n1 = int(np.sum(ws_interaction_obs == 0)), int(np.sum(ws_interaction_obs == 1))
        min_class = min(n0, n1)
        cv_ws = min(cv_ws_interaction, min_class)
        if cv_ws >= 2:
            model_ws_interaction = LogisticRegressionCV(
                Cs=Cs_ws_interaction,
                cv=cv_ws,
                penalty="l2",
                solver="lbfgs",
                fit_intercept=False,
                scoring="neg_log_loss",
                max_iter=5000,
                random_state=seed,
            )
            model_ws_interaction.fit(ws_interaction_cond_obs, ws_interaction_obs)
            C_sel = float(model_ws_interaction.C_[0])
        else:
            C_sel = float(Cs_ws_interaction[len(Cs_ws_interaction) // 2])
            model_ws_interaction = LogisticRegression(
                penalty="l2",
                C=C_sel,
                solver="lbfgs",
                fit_intercept=False,
                max_iter=5000,
                random_state=seed,
            )
            model_ws_interaction.fit(ws_interaction_cond_obs, ws_interaction_obs)
        alpha_ws_interaction_l2 = 1.0 / C_sel
        theta_ws_interaction_mean = model_ws_interaction.coef_.ravel()
        pred_ws_interaction = model_ws_interaction.predict_proba(ws_interaction_cond)[:, 1]
        resid_obs_ws_interaction = ws_interaction_obs.astype(float) - pred_ws_interaction[idx_obs_ws_interaction]
        resid_ws_interaction = np.full_like(ws_interaction, np.nan, dtype=float)
        resid_ws_interaction[idx_obs_ws_interaction] = resid_obs_ws_interaction
        sigma2_ws_interaction_mean = np.var(resid_obs_ws_interaction)
    else:
        print(f"the variance of Interacted_walk is 0 for user {userid}")
        const = float(safe_nanmean(ws_interaction_obs.astype(float), default=np.nan))
        alpha_ws_interaction_l2 = float(alpha_l2_list[0])
        theta_ws_interaction_mean = np.zeros(ws_interaction_cond.shape[1])
        if not np.isfinite(const):
            const = 0.5
            theta_ws_interaction_mean[0] = 0.0
        elif const <= 0.0:
            theta_ws_interaction_mean[0] = -8.0
        elif const >= 1.0:
            theta_ws_interaction_mean[0] = 8.0
        else:
            theta_ws_interaction_mean[0] = float(np.log(const / (1.0 - const)))
        pred_ws_interaction = np.full(len(ws_interaction), const, dtype=float)
        resid_ws_interaction = np.full_like(ws_interaction, np.nan, dtype=float)
        resid_ws_interaction[idx_obs_ws_interaction] = 0.0
        sigma2_ws_interaction_mean = 0.0

    print(f"The fit of ws_interaction is good for user {userid}")


    #### Model 4: FourSC model ####
    fourSC_cond = np.stack([
        Intercept,
        fourSC_lag1,
        yesterday_step_count, seven_day_step_count_avg, 
        prior2hour_step_count_filled,
        recent_burden, seven_day_pageview_count, past7days_morning_wearing, 
        Interacted_7d_walk,
        anticipated_affect_yesterday,
        active_status_fraction_7days,
        # is_weekend, 
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
    sigma2_fourSC_mean = np.var(resid_obs_fourSC) if resid_obs_fourSC.size > 0 else 0.0
    population_resid_pools["resid_fourSC_population"].extend(resid_obs_fourSC.tolist())

    print(f"The fit of fourSC is good for user {userid}")


    #### Model 5: Anticipated affect model ####
    # Daily outcome: one row per calendar day (morning row). Predictors from that row except treatment,
    # which enters as both ws_morning_day and ws_afternoon_day for that day.
    # planning_prompt_filled = np.where(np.isnan(planning_prompt), 0, planning_prompt)
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
    sigma2_anticipated_affect_mean = (
        np.var(resid_obs_anticipated_affect) if resid_obs_anticipated_affect.size > 0 else 0.0
    )

    _obs_antic = resid_anticipated_affect[~np.isnan(resid_anticipated_affect)]
    population_resid_pools["resid_antic_population"].extend(_obs_antic.tolist())

    print(f"The fit of anticipated_affect is good for user {userid}")


    #### Model 10: CAE model #### 
#### Model 10: CAE model, Bayesian random coefficient b_i for every variable ####

    K = 14

    CAE_avg_sw = CAE_avg.reshape(-1, K)[:, 0]
    CAE_avg_lastweek_sw = CAE_avg_lastweek.reshape(-1, K)[:, 0]
    week_sw = week.reshape(-1, K)[:, 0]
    Intercept_sw = np.ones(len(week_sw))

    foursc_wk = fourSC.reshape(-1, K)
    _mu_foursc = safe_nanmean(fourSC)
    _mu_antic = safe_nanmean(anticipated_affect)

    foursc_wk = np.where(np.isnan(foursc_wk), _mu_foursc, foursc_wk)

    _af = anticipated_affect.reshape(-1, K)
    _af = np.where(np.isnan(_af), _mu_antic, _af)
    antic_wk = _af.reshape(-1, 7, 2).mean(axis=2)
    antic_wk = np.where(np.isnan(antic_wk), _mu_antic, antic_wk)

    foursc_e = np.array(
        [within_week_ewma(row) for row in foursc_wk],
        dtype=float,
    )
    antic_e = np.array(
        [within_week_ewma(row) for row in antic_wk],
        dtype=float,
    )

    CAE_cond = np.column_stack([
        Intercept_sw,
        CAE_avg_lastweek_sw,
        week_sw,
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

    population_resid_pools["resid_CAE_population"].extend(
        resid_CAE[~np.isnan(resid_CAE)].tolist()
    )

    print(f"The full random-coefficient Bayesian CAE fit is ready for user {userid}")



    # change the length of the condition matrix
    # K = 14
    # CAE_avg_sw = CAE_avg.reshape(-1, K)[:, 0]
    # # perceived_utility_norm_sw = perceived_utility_norm.reshape(-1, K)[:, 0]
    # CAE_avg_lastweek_sw = CAE_avg_lastweek.reshape(-1, K)[:, 0]
    # week_sw = week.reshape(-1, K)[:, 0]
    # Intercept_sw = np.ones(len(week_sw))
    
    # # FourSC: 14 decision slots/week. Anticipated affect is daily (repeated across 2 decisions/day) → 7 columns/week
    # foursc_wk = fourSC.reshape(-1, K)
    # _mu_foursc = safe_nanmean(fourSC)
    # _mu_antic = safe_nanmean(anticipated_affect)
    # foursc_wk = np.where(np.isnan(foursc_wk), _mu_foursc, foursc_wk)
    # _af = anticipated_affect.reshape(-1, K)
    # _af = np.where(np.isnan(_af), _mu_antic, _af)
    # antic_wk = _af.reshape(-1, 7, 2).mean(axis=2)
    # antic_wk = np.where(np.isnan(antic_wk), _mu_antic, antic_wk)

    # CAE_cond = np.hstack([
    #     Intercept_sw[:, None],
    #     CAE_avg_lastweek_sw[:, None],
    #     week_sw[:, None],
    #     foursc_wk,
    #     antic_wk
    # ])

    # ncv_w = 2
    
    # idx_obs_CAE = ~np.isnan(CAE_avg_sw)
    # CAE_cond_obs = CAE_cond[idx_obs_CAE, :]
    # CAE_avg_sw_obs = CAE_avg_sw[idx_obs_CAE]

    # # print(len(AA_avg_norm_sw_obs))

    # has_cae_obs = CAE_avg_sw_obs.size > 0
    # if has_cae_obs and CAE_avg_sw_obs.size >= 2 and np.var(CAE_avg_sw_obs) > 0:
    #     model_CAE = RidgeCV(
    #         alphas=alpha_l2_list,
    #         fit_intercept=False,
    #         cv=None,
    #     )
    #     model_CAE.fit(CAE_cond_obs, CAE_avg_sw_obs)

    #     alpha_CAE_l2 = model_CAE.alpha_
    #     theta_CAE_mean = model_CAE.coef_
    #     pred_CAE = model_CAE.predict(CAE_cond)

    #     resid_obs_CAE = CAE_avg_sw_obs - pred_CAE[idx_obs_CAE]
    #     resid_CAE = np.full_like(CAE_avg_sw, np.nan, dtype=float)
    #     resid_CAE[idx_obs_CAE] = resid_obs_CAE
    #     sigma2_CAE_mean = np.var(resid_obs_CAE)

    # elif has_cae_obs:
    #     print(
    #         f"CAE (Y_w) variance is 0 or fewer than 2 observations for user {userid}; "
    #         "using constant fallback"
    #     )
    #     const = float(safe_nanmean(CAE_avg_sw_obs, default=0.0))
    #     alpha_CAE_l2 = float(alpha_l2_list[0])
    #     theta_CAE_mean = np.zeros(CAE_cond.shape[1], dtype=float)
    #     theta_CAE_mean[0] = const
    #     pred_CAE = np.full(len(CAE_avg_sw), const, dtype=float)
    #     resid_CAE = np.full_like(CAE_avg_sw, np.nan, dtype=float)
    #     resid_CAE[idx_obs_CAE] = 0.0
    #     sigma2_CAE_mean = 0.0

    # else:
    #     print(f"no observed CAE (Y_w) values for user {userid}; using zero fallback")

    #     alpha_CAE_l2 = 1.0
    #     theta_CAE_mean = np.zeros(CAE_cond.shape[1], dtype=float)
    #     pred_CAE = np.zeros(len(CAE_avg_sw), dtype=float)

    #     resid_CAE = np.full_like(CAE_avg_sw, np.nan, dtype=float)
    #     sigma2_CAE_mean = 0.0
    
    
    
    # print(f"The fit of CAE is good for user {userid}")


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

    population_resid_pools["resid_CAE_short_population"].extend(
        resid_CAE_short_avg[~np.isnan(resid_CAE_short_avg)].tolist()
    )

    print(f"The full random-coefficient Bayesian CAE_short_avg fit is ready for user {userid}")

    # if j = 1, then the emission is 3 questions from CAE
    # CAE_short_avg_sw = CAE_short_avg.reshape(-1, K)[:, 0]
    # CAE_avg_sw_filled = fill_nan_with_mean(CAE_avg_sw)
    # idx_obs_CAE_short_avg = ~np.isnan(CAE_short_avg_sw)
    # CAE_short_avg_sw_obs = CAE_short_avg_sw[idx_obs_CAE_short_avg]
    
    # CAE_short_avg_cond = np.stack([
    #     Intercept_sw,
    #     CAE_avg_sw_filled,
    # ], axis=1)
    
    # idx_obs_CAE_short_avg = ~np.isnan(CAE_short_avg_sw)
    # CAE_short_avg_cond_obs = CAE_short_avg_cond[idx_obs_CAE_short_avg, :]
    # CAE_short_avg_sw_obs = CAE_short_avg_sw[idx_obs_CAE_short_avg]

    # if CAE_short_avg_sw_obs.size >= 2:
    #     # Use Generalized Cross-Validation; no explicit folds.
    #     model_CAE_short_avg = RidgeCV(
    #         alphas=alpha_l2_list,
    #         fit_intercept=False,
    #         cv=None,
    #     )
    #     model_CAE_short_avg.fit(CAE_short_avg_cond_obs, CAE_short_avg_sw_obs)

    #     alpha_CAE_short_avg_l2 = model_CAE_short_avg.alpha_
    #     theta_CAE_short_avg_mean = model_CAE_short_avg.coef_
    #     pred_CAE_short_avg = model_CAE_short_avg.predict(CAE_short_avg_cond)

    #     resid_obs_CAE_short_avg = (
    #         CAE_short_avg_sw_obs
    #         - pred_CAE_short_avg[idx_obs_CAE_short_avg]
    #     )

    #     resid_CAE_short_avg = np.full_like(CAE_short_avg_sw, np.nan, dtype=float)
    #     resid_CAE_short_avg[idx_obs_CAE_short_avg] = resid_obs_CAE_short_avg

    #     sigma2_CAE_short_avg_mean = np.var(resid_obs_CAE_short_avg)

    # elif CAE_short_avg_sw_obs.size == 1:
    #     print(f"fallback to fixed alpha for CAE_short_avg model for user {userid}")

    #     alpha_CAE_short_avg_l2 = 1.0

    #     model_CAE_short_avg = Ridge(
    #         alpha=alpha_CAE_short_avg_l2,
    #         fit_intercept=False,
    #     )
    #     model_CAE_short_avg.fit(CAE_short_avg_cond_obs, CAE_short_avg_sw_obs)

    #     theta_CAE_short_avg_mean = model_CAE_short_avg.coef_
    #     pred_CAE_short_avg = model_CAE_short_avg.predict(CAE_short_avg_cond)

    #     resid_obs_CAE_short_avg = (
    #         CAE_short_avg_sw_obs
    #         - pred_CAE_short_avg[idx_obs_CAE_short_avg]
    #     )

    #     resid_CAE_short_avg = np.full_like(CAE_short_avg_sw, np.nan, dtype=float)
    #     resid_CAE_short_avg[idx_obs_CAE_short_avg] = resid_obs_CAE_short_avg

    #     sigma2_CAE_short_avg_mean = np.var(resid_obs_CAE_short_avg)

    # else:
    #     print(f"no observed CAE_short_avg values for user {userid}; using zero fallback")

    #     alpha_CAE_short_avg_l2 = float(alpha_l2_list[0])

    #     theta_CAE_short_avg_mean = np.zeros(CAE_short_avg_cond.shape[1], dtype=float)

    #     pred_CAE_short_avg = np.zeros(len(CAE_short_avg_sw), dtype=float)

    #     resid_CAE_short_avg = np.full_like(CAE_short_avg_sw, np.nan, dtype=float)

    #     sigma2_CAE_short_avg_mean = 0.0
    
    # print(f"The fit of CAE_short_avg is good for user {userid}")

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

        "theta_CAE": json_float_list(theta_CAE_mean, digits),
        "theta_CAE_names": THETA_CAE_NAMES,

        "theta_CAE_short_avg": json_float_list(theta_CAE_short_avg_mean, digits),
        "theta_CAE_short_avg_names": THETA_CAE_SHORT_AVG_NAMES,

        "resid_prior2hour_step_count": json_float_list(resid_prior2hour_step_count, digits),
        "resid_active_status": json_float_list(resid_active_status, digits),
        "resid_ws_interaction": json_float_list(resid_ws_interaction, digits),
        "resid_fourSC": json_float_list(resid_fourSC, digits),
        "resid_antic": json_float_list(resid_anticipated_affect, digits),
        "resid_CAE": json_float_list(resid_CAE, digits),
        "resid_CAE_short_avg": json_float_list(resid_CAE_short_avg, digits),

        "alpha_prior2hour_step_count_l2": json_float(alpha_prior2hour_step_count_l2, digits),
        "alpha_active_status_l2": json_float(alpha_active_status_l2, digits),
        "alpha_ws_interaction_l2": json_float(alpha_ws_interaction_l2, digits),
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


    
np.savetxt(file_user_ids, userid_all, fmt='%d')

# save df_fit.csv with the predicted columns
df_fit.loc[df_fit["ParticipantIdentifier"].isin(userid_all)].to_csv(
    work_folder / "df_fit_11week.csv",
    index=False,
)

# Population-level residual pools, shared across users (see comment above the
# accumulator init). ``vani_env.EnvConfig`` loads this once per user to give
# sparse-data users a non-degenerate fallback noise distribution.
digits = 3
population_residuals_out = {
    key: json_float_list(np.asarray(vals, dtype=float), digits)
    for key, vals in population_resid_pools.items()
}
with open(work_folder / "population_residuals.json", "w", encoding="utf-8") as f:
    json.dump(population_residuals_out, f, allow_nan=False)
for key, vals in population_resid_pools.items():
    print(f"{key}: n={len(vals)}")

# %%
