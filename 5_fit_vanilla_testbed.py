# %%
# 0. import libraries
#
# Run order:
#   1) ``perceived_utility.py``  → fits the joint state-space model, writes
#      ``params_env_<id>.json`` / ``pred_<id>.json`` with the ``theta_ml_*`` /
#      ``resid_ml_*`` keys, and adds ``perceived_utility`` /
#      ``perceived_utility_lastweek`` columns to ``df_fit.csv``.
#   2) ``5_fit_vanilla_testbed.py`` (this script) → fits the vanilla mediator
#      / outcome models that consume ``perceived_utility_lastweek`` as a
#      predictor and **merges** their ``theta_*`` / ``resid_*`` keys into the
#      same JSON files (existing ML keys are preserved).
import json
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from pathlib import Path
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeCV,
)
from statsmodels.regression.mixed_linear_model import MixedLMParams


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

for userid in df_fit["ParticipantIdentifier"].unique():
    vc = df_fit.loc[df_fit["ParticipantIdentifier"] == userid, "week"].value_counts()
    if vc.get(0, 0) > 2:
        m = df_fit["ParticipantIdentifier"] == userid
        df_fit.loc[m, "week"] = df_fit.loc[m, "week"] + 1
df_fit = df_fit[df_fit["week"] < 13]

# remove week 0 and week 1 data
df_fit = df_fit[df_fit['week'] > 1]

# turn week 2 into week 1
df_fit['week'] = df_fit['week'] - 1

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

def _embed_random_effects(b_active, active_re, p):
    """Map active random coefficients back to the full p-dimensional vector."""
    b_full = np.zeros(p, dtype=float)
    for k, col_idx in enumerate(active_re):
        b_full[col_idx] = b_active[k]
    return b_full


class _MixedLMFitResult:
    """MixedLM result with random effects embedded on the full design axis."""

    def __init__(self, inner_result, active_re, p):
        self._inner = inner_result
        self.active_re = list(active_re)
        self.p = int(p)
        self.converged = bool(inner_result.converged)
        self.fe_params = pd.Series(
            np.asarray(inner_result.fe_params, dtype=float).ravel(),
            index=getattr(inner_result.fe_params, "index", None),
        )
        self.scale_var = float(inner_result.scale)
        self.cov_re = inner_result.cov_re
        self.random_effects = {}
        for key, values in inner_result.random_effects.items():
            b_active = np.asarray(values, dtype=float).ravel()
            self.random_effects[key] = pd.Series(
                _embed_random_effects(b_active, self.active_re, self.p)
            )

    def summary(self):
        return self._inner.summary()

    @property
    def scale(self):
        return self.scale_var


def _try_fit_mixedlm(model, free, start, reml, maxiter):
    """Try ML/REML with lbfgs and bfgs; return the first converged fit."""
    reml_options = [False]
    if reml:
        reml_options.append(True)

    best = None
    for use_reml in reml_options:
        for method in ("lbfgs", "bfgs"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cand = model.fit(
                        reml=use_reml,
                        method=method,
                        maxiter=maxiter,
                        disp=False,
                        free=free,
                        start_params=start,
                    )
            except Exception as err:
                print(f"MixedLM fit failed (reml={use_reml}, method={method}): {err}")
                continue
            if cand.converged:
                return cand
            if best is None:
                best = cand
    return best


def fit_diag_random_coef_mixedlm(
    y,
    X,
    groups,
    reml=True,
    maxiter=8000,
    var_floor=1e-5,
    predictor_names=None,
):
    """
    Fit MixedLM with one random coefficient per fixed-effect variable.

    Model:
        y_ij = X_ij beta + X_ij b_i + error_ij

    Random effects:
        b_i ~ N(0, D)

    Here D is constrained to be diagonal, so every variable has a random effect,
    but random-effect correlations are fixed at zero.

    If the full random-coefficient model fails to converge, near-zero random-effect
    columns are pruned and the model is refit until convergence or only one random
    effect remains.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)

    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y_fit = y[keep]
    X_fit = X[keep]
    groups_fit = groups[keep]

    p = X_fit.shape[1]
    beta_ols, _, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    start = MixedLMParams.from_components(
        fe_params=beta_ols,
        cov_re=0.01 * np.eye(p),
    )

    active_re = list(range(p))
    best_result = None

    while active_re:
        X_re = X_fit[:, active_re]
        model = sm.MixedLM(
            endog=y_fit,
            exog=X_fit,
            groups=groups_fit,
            exog_re=X_re,
            missing="none",
        )
        free = MixedLMParams.from_components(
            fe_params=np.ones(p),
            cov_re=np.eye(len(active_re)),
        )
        result = _try_fit_mixedlm(model, free, start, reml=reml, maxiter=maxiter)
        if result is None:
            raise RuntimeError("MixedLM fit failed for all optimizers.")

        best_result = _MixedLMFitResult(result, active_re, p)
        if result.converged:
            if len(active_re) < p:
                dropped = [j for j in range(p) if j not in active_re]
                dropped_names = [
                    predictor_names[j] if predictor_names is not None else j
                    for j in dropped
                ]
                print(
                    "MixedLM converged after pruning random effects for: "
                    + ", ".join(str(x) for x in dropped_names)
                )
            return best_result

        if len(active_re) == 1:
            break

        re_var = np.diag(np.asarray(result.cov_re, dtype=float))
        min_pos = int(np.argmin(re_var))
        min_var = float(re_var[min_pos])
        dropped_idx = active_re[min_pos]
        dropped_name = (
            predictor_names[dropped_idx]
            if predictor_names is not None
            else dropped_idx
        )
        print(
            f"MixedLM not converged; pruning random effect for {dropped_name} "
            f"(estimated variance={min_var:.2e}) and refitting."
        )
        active_re.pop(min_pos)
        re_var = np.delete(re_var, min_pos)
        start = MixedLMParams.from_components(
            fe_params=np.asarray(result.fe_params, dtype=float),
            cov_re=np.diag(np.maximum(re_var, var_floor)),
        )

    raise RuntimeError(
        "MixedLM did not converge. "
        f"Remaining random-effect columns: {active_re}. "
        "The random-coefficient CAE specification may be too flexible for this sample."
    )

def get_user_random_effect(result, userid, p):
    """
    Extract b_i from a fitted MixedLM result.
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
                  + 14 FourSC decision-slot summaries
                  + 7 anticipated-affect daily summaries

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

        X = np.hstack(
            [
                Intercept_sw[:, None],
                CAE_avg_lastweek_sw[:, None],
                week_sw[:, None],
                foursc_wk,
                antic_wk,
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

THETA_PRIOR2HOUR_STEP_COUNT_NAMES = [
    "intercept",
    "EMA_Prior2HourStepCount",
    "dow",
    "decision_time",
]

THETA_RECORDED_PHYSICAL_ACTIVITY_NAMES = [
    "intercept",
    "Previous7DaysRPA",
    "dow",
]

THETA_ACTIVE_STATUS_NAMES = [
    "intercept",
    "active_status_fraction_7days",
    "dow",
]

THETA_WS_INTERACTION_NAMES = [
    "intercept",
    "Interacted_7d_walk",
    "dow",
    "decision_time",
]



THETA_FOURSC_NAMES = [
    "intercept",
    "fourSC_lag1",
    "yesterday_step_count",
    "seven_day_step_count_avg",
    "prior2hour_step_count",
    "Previous7DaysRPA",
    "recent_burden",
    "seven_day_pageview_count",
    "past7days_morning_wearing",
    "yesterday_salience_message",
    "Interacted_7d_walk",
    "anticipated_affect_yesterday",
    "fractionofactivedayspast7days",
    "dow",
    "decision_time",
    "perceived_utility_lastweek",
    "CAE_avg_lastweek",
    "WalkingSuggestion",
    "WalkingSuggestion_by_yesterday_step_count",
    "WalkingSuggestion_by_prior2hour_step_count",
    "WalkingSuggestion_by_recent_burden",
    "WalkingSuggestion_by_seven_day_pageview_count",
    "WalkingSuggestion_by_past7days_morning_wearing",
    "WalkingSuggestion_by_yesterday_salience_message",
    "WalkingSuggestion_by_Interacted_7d_walk",
    "WalkingSuggestion_by_anticipated_affect_yesterday",
    "WalkingSuggestion_by_dow",
    "WalkingSuggestion_by_decision_time",
    "WalkingSuggestion_by_perceived_utility_lastweek",
    "WalkingSuggestion_by_CAE_avg_lastweek",
]

THETA_ANTIC_NAMES = [
    "intercept",
    "anticipated_affect_yesterday",
    "today_step_count",
    "recorded_physical_activity",
    "active_status",
    "salience_message",
    "dow",
    "perceived_utility_lastweek",
    "CAE_avg_lastweek",
    "A0_morning",
    "A1_afternoon",
    "A0_morning_by_salience_message",
    "A1_afternoon_by_salience_message",
    "A0_morning_by_dow",
    "A1_afternoon_by_dow",
    "A0_morning_by_perceived_utility_lastweek",
    "A1_afternoon_by_perceived_utility_lastweek",
    "A0_morning_by_CAE_avg_lastweek",
    "A1_afternoon_by_CAE_avg_lastweek",
]

THETA_CAE_NAMES = (
    ["intercept", "CAE_avg_lastweek", "week"]
    + [f"fourSC_slot_{j}" for j in range(14)]
    + [f"anticipated_affect_day_{j}" for j in range(7)]
)

THETA_CAE_SHORT_AVG_NAMES = [
    "intercept",
    "CAE_avg",
]

alpha_l2_list = [0.2, 0.5, 1, 2, 5]
alpha_lap_list = [0.5, 1, 2, 5]
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
# Fit CAE MixedLM with random coefficient for every CAE variable
# ============================================================

cae_ml_df = build_cae_mixedlm_data(df_fit, userid_all, K=14)

cae_ml_obs = cae_ml_df.dropna(subset=["CAE_avg"]).copy()

X_cae_obs = cae_ml_obs[THETA_CAE_NAMES].to_numpy(dtype=float)
y_cae_obs = cae_ml_obs["CAE_avg"].to_numpy(dtype=float)
groups_cae_obs = cae_ml_obs["ParticipantIdentifier"].to_numpy()

mixedlm_CAE = fit_diag_random_coef_mixedlm(
    y=y_cae_obs,
    X=X_cae_obs,
    groups=groups_cae_obs,
    reml=True,
    predictor_names=list(THETA_CAE_NAMES),
)

print(mixedlm_CAE.summary())
print(f"CAE MixedLM converged: {mixedlm_CAE.converged}")

cae_random_effect_cols = list(range(len(THETA_CAE_NAMES)))

# ============================================================
# Fit short-CAE MixedLM with random coefficient for every variable
# ============================================================

cae_short_ml_df = build_cae_short_mixedlm_data(df_fit, userid_all, K=14)

cae_short_ml_obs = cae_short_ml_df.dropna(subset=["CAE_short_avg"]).copy()

X_cae_short_obs = cae_short_ml_obs[THETA_CAE_SHORT_AVG_NAMES].to_numpy(dtype=float)
y_cae_short_obs = cae_short_ml_obs["CAE_short_avg"].to_numpy(dtype=float)
groups_cae_short_obs = cae_short_ml_obs["ParticipantIdentifier"].to_numpy()

mixedlm_CAE_short = fit_diag_random_coef_mixedlm(
    y=y_cae_short_obs,
    X=X_cae_short_obs,
    groups=groups_cae_short_obs,
    reml=True,
    predictor_names=list(THETA_CAE_SHORT_AVG_NAMES),
)

print(mixedlm_CAE_short.summary())
print(f"CAE_short MixedLM converged: {mixedlm_CAE_short.converged}")

cae_short_random_effect_cols = list(range(len(THETA_CAE_SHORT_AVG_NAMES)))

theta_pageview_list = []
theta_fitbitwearing_list = []
theta_eodcomplete_list = []
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

    today_step_count = dat_user['TodayStepCount_norm'].to_numpy()
    yesterday_step_count = dat_user['YesterdayStepCount_norm'].to_numpy()
    seven_day_step_count_avg = dat_user['EMA_StepCount_norm'].to_numpy()

    prior2hour_step_count = dat_user['prior2hour_step_norm'].to_numpy()
    ema_prior2hour_step_count = dat_user['EMA_Prior2HourStepCount_norm'].to_numpy()

    
    recent_burden = dat_user['recent_burden_norm'].to_numpy()

    recorded_physical_activity = dat_user['RecordedPhysicalActivity'].to_numpy()
    active_status = dat_user['active_status'].to_numpy()
    active_status_fraction_7days = dat_user['active_status_fraction_7days'].to_numpy()
    Previous7DaysRPA = dat_user['Previous7DaysRPA'].to_numpy()
    
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
    salience_message = dat_user['SalienceMessage'].to_numpy()
    # planning_prompt = dat_user['planning_prompt'].to_numpy()
    # yesterday_planning_prompt = dat_user['yesterday_planning_prompt'].to_numpy()
    yesterday_salience_message = dat_user['yesterday_SalienceMessage'].to_numpy()

    is_weekend = dat_user['is_weekend'].to_numpy()
    day = dat_user['day_norm'].to_numpy()
    week = dat_user['week_norm'].to_numpy() #TODO: check this
    decision_time = dat_user['DecisionTime'].to_numpy()
    dow = dat_user['dow_norm'].to_numpy()
    
    

    # fill in missing values (NAN) with mean for predictors except for FourSC and Intercept
    fourSC_lag1 = fill_nan_with_mean(fourSC_lag1)
    today_step_count = fill_nan_with_mean(today_step_count)
    yesterday_step_count = fill_nan_with_mean(yesterday_step_count)
    seven_day_step_count_avg = fill_nan_with_mean(seven_day_step_count_avg)
    prior2hour_step_count_filled = fill_nan_with_mean(prior2hour_step_count)
    ema_prior2hour_step_count = fill_nan_with_mean(ema_prior2hour_step_count)
    Previous7DaysRPA = fill_nan_with_mean(Previous7DaysRPA)
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
        dow, decision_time
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

    #### Model 2: Recorded physical activity model #### 
    # P(RecordedPhysicalActivity = 1 | prior 7-day RPA fraction, dow); morning rows
    recorded_physical_activity_cond = np.stack([
        Intercept,
        Previous7DaysRPA,
        dow
    ], axis=1)
    Cs_recorded_physical_activity = np.sort(1.0 / np.asarray(alpha_l2_list, dtype=float))

    idx_obs_recorded_physical_activity = (decision_time == 0) & ~np.isnan(recorded_physical_activity)
    cv_recorded_physical_activity = min(5, int(idx_obs_recorded_physical_activity.sum()))
    recorded_physical_activity_cond_obs = recorded_physical_activity_cond[idx_obs_recorded_physical_activity, :]
    recorded_physical_activity_obs = recorded_physical_activity[idx_obs_recorded_physical_activity]

    has_recorded_physical_activity_obs = recorded_physical_activity_obs.size > 0
    if has_recorded_physical_activity_obs and np.var(recorded_physical_activity_obs.astype(float)) > 0:
        n0, n1 = int(np.sum(recorded_physical_activity_obs == 0)), int(np.sum(recorded_physical_activity_obs == 1))
        min_class = min(n0, n1)
        cv_rpa = min(cv_recorded_physical_activity, min_class)
        if cv_rpa >= 2:
            model_recorded_physical_activity = LogisticRegressionCV(
                Cs=Cs_recorded_physical_activity,
                cv=cv_rpa,
                penalty="l2",
                solver="lbfgs",
                fit_intercept=False,
                scoring="neg_log_loss",
                max_iter=5000,
                random_state=seed,
            )
            model_recorded_physical_activity.fit(
                recorded_physical_activity_cond_obs, recorded_physical_activity_obs
            )
            C_sel = float(model_recorded_physical_activity.C_[0])
        else:
            C_sel = float(Cs_recorded_physical_activity[len(Cs_recorded_physical_activity) // 2])
            model_recorded_physical_activity = LogisticRegression(
                penalty="l2",
                C=C_sel,
                solver="lbfgs",
                fit_intercept=False,
                max_iter=5000,
                random_state=seed,
            )
            model_recorded_physical_activity.fit(
                recorded_physical_activity_cond_obs, recorded_physical_activity_obs
            )
        alpha_recorded_physical_activity_l2 = 1.0 / C_sel
        theta_recorded_physical_activity_mean = model_recorded_physical_activity.coef_.ravel()
        pred_recorded_physical_activity = model_recorded_physical_activity.predict_proba(
            recorded_physical_activity_cond
        )[:, 1]
        resid_obs_recorded_physical_activity = (
            recorded_physical_activity_obs.astype(float)
            - pred_recorded_physical_activity[idx_obs_recorded_physical_activity]
        )
        resid_recorded_physical_activity = np.full_like(recorded_physical_activity, np.nan, dtype=float)
        resid_recorded_physical_activity[idx_obs_recorded_physical_activity] = resid_obs_recorded_physical_activity
        sigma2_recorded_physical_activity_mean = np.var(resid_obs_recorded_physical_activity)
    else:
        print(f"the variance of recorded_physical_activity is 0 for user {userid}")
        const = float(
            safe_nanmean(recorded_physical_activity_obs.astype(float), default=np.nan)
        )
        alpha_recorded_physical_activity_l2 = float(alpha_l2_list[0])
        theta_recorded_physical_activity_mean = np.zeros(recorded_physical_activity_cond.shape[1])
        if not np.isfinite(const):
            const = 0.5
            theta_recorded_physical_activity_mean[0] = 0.0
        elif const <= 0.0:
            theta_recorded_physical_activity_mean[0] = -25.0
        elif const >= 1.0:
            theta_recorded_physical_activity_mean[0] = 25.0
        else:
            theta_recorded_physical_activity_mean[0] = float(np.log(const / (1.0 - const)))
        pred_recorded_physical_activity = np.full(len(recorded_physical_activity), const, dtype=float)
        resid_recorded_physical_activity = np.full_like(recorded_physical_activity, np.nan, dtype=float)
        resid_recorded_physical_activity[idx_obs_recorded_physical_activity] = 0.0
        sigma2_recorded_physical_activity_mean = 0.0

    print(f"The fit of recorded_physical_activity is good for user {userid}")

    #### Model 3: Active status model ####
    # P(active_status = 1 | prior 7-day active fraction, dow); morning rows with answered survey
    active_status_cond = np.stack([
        Intercept,
        active_status_fraction_7days,
        dow,
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
            theta_active_status_mean[0] = -25.0
        elif const >= 1.0:
            theta_active_status_mean[0] = 25.0
        else:
            theta_active_status_mean[0] = float(np.log(const / (1.0 - const)))
        pred_active_status = np.full(len(active_status), const, dtype=float)
        resid_active_status = np.full_like(active_status, np.nan, dtype=float)
        resid_active_status[idx_obs_active_status] = 0.0
        sigma2_active_status_mean = 0.0

    print(f"The fit of active_status is good for user {userid}")

    #### Model 4: Walking suggestion interaction model #### 

    # P(WalkingSuggestion = 1 | context): L2 logistic CV (rows with observed WS and predictors)
    ws_interaction_cond = np.stack([
        Intercept,
        Interacted_7d_walk,
        dow, decision_time
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
            theta_ws_interaction_mean[0] = -25.0
        elif const >= 1.0:
            theta_ws_interaction_mean[0] = 25.0
        else:
            theta_ws_interaction_mean[0] = float(np.log(const / (1.0 - const)))
        pred_ws_interaction = np.full(len(ws_interaction), const, dtype=float)
        resid_ws_interaction = np.full_like(ws_interaction, np.nan, dtype=float)
        resid_ws_interaction[idx_obs_ws_interaction] = 0.0
        sigma2_ws_interaction_mean = 0.0

    print(f"The fit of ws_interaction is good for user {userid}")


    #### Model 5: FourSC model #### 
    fourSC_cond = np.stack([
        Intercept,
        fourSC_lag1,
        yesterday_step_count, seven_day_step_count_avg, 
        prior2hour_step_count_filled, Previous7DaysRPA,
        recent_burden, seven_day_pageview_count, past7days_morning_wearing, 
        yesterday_salience_message,
        Interacted_7d_walk,
        anticipated_affect_yesterday,
        active_status_fraction_7days,
        # is_weekend, 
        dow, decision_time,
        perceived_utility_lastweek,
        CAE_avg_lastweek, 
        WalkingSuggestion, WalkingSuggestion * yesterday_step_count,
        WalkingSuggestion * prior2hour_step_count_filled,
        WalkingSuggestion * recent_burden,
        WalkingSuggestion * seven_day_pageview_count,
        WalkingSuggestion * past7days_morning_wearing,
        WalkingSuggestion * yesterday_salience_message,
        WalkingSuggestion * Interacted_7d_walk,
        WalkingSuggestion * anticipated_affect_yesterday,
        # WalkingSuggestion * is_weekend,
        WalkingSuggestion * dow,
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
    fourSC_cond_obs = fourSC_cond[idx_obs_fourSC, :]
    FourSC_obs = fourSC[idx_obs_fourSC]
    n_obs_fourSC = int(idx_obs_fourSC.sum())
    
    if n_obs_fourSC >= 2:
        cv_fourSC = min(5, n_obs_fourSC)
        model_fourSC = RidgeCV(
            alphas=alpha_l2_list,
            fit_intercept=False,
            cv=cv_fourSC,
            scoring="neg_mean_squared_error",
        )
        model_fourSC.fit(fourSC_cond_obs, FourSC_obs)
        alpha_fourSC_l2 = model_fourSC.alpha_
        theta_fourSC_mean = model_fourSC.coef_
        pred_fourSC = model_fourSC.predict(fourSC_cond)
    elif n_obs_fourSC == 1:
        print(f"fallback to fixed alpha for fourSC model for user {userid}")
        alpha_fourSC_l2 = 1.0
        model_fourSC = Ridge(alpha=alpha_fourSC_l2, fit_intercept=False)
        model_fourSC.fit(fourSC_cond_obs, FourSC_obs)
        theta_fourSC_mean = model_fourSC.coef_
        pred_fourSC = model_fourSC.predict(fourSC_cond)
    else:
        print(f"no observed fourSC values for user {userid}; using zero fallback")
        alpha_fourSC_l2 = float(alpha_l2_list[0])
        theta_fourSC_mean = np.zeros(fourSC_cond.shape[1], dtype=float)
        pred_fourSC = np.zeros(len(fourSC), dtype=float)

    resid_obs_fourSC = FourSC_obs - pred_fourSC[idx_obs_fourSC]
    # fill in the full residual array with NaN for unobserved
    resid_fourSC = np.full_like(fourSC, np.nan, dtype=float)
    resid_fourSC[idx_obs_fourSC] = resid_obs_fourSC
    sigma2_fourSC_mean = np.var(resid_obs_fourSC) if resid_obs_fourSC.size > 0 else 0.0

    print(f"The fit of fourSC is good for user {userid}")


    #### Model 6: Anticipated affect model #### 
    # Daily outcome: one row per calendar day (morning row). Predictors from that row except treatment,
    # which enters as both ws_morning_day and ws_afternoon_day for that day.
    recorded_physical_activity_filled = fill_nan_with_mean(recorded_physical_activity)
    active_status_filled = fill_nan_with_mean(active_status)
    # planning_prompt_filled = np.where(np.isnan(planning_prompt), 0, planning_prompt)
    _am = idx_morning
    anticipated_affect_cond_day = np.stack([
        Intercept[_am],
        anticipated_affect_yesterday[_am],
        today_step_count[_am],
        recorded_physical_activity_filled[_am],
        active_status_filled[_am],
        salience_message[_am],
        dow[_am],
        perceived_utility_lastweek[_am],
        CAE_avg_lastweek[_am],
        ws_morning_day,
        ws_afternoon_day,
        ws_morning_day * salience_message[_am],
        ws_afternoon_day * salience_message[_am],
        ws_morning_day * dow[_am],
        ws_afternoon_day * dow[_am],
        ws_morning_day * perceived_utility_lastweek[_am],
        ws_afternoon_day * perceived_utility_lastweek[_am],
        ws_morning_day * CAE_avg_lastweek[_am],
        ws_afternoon_day * CAE_avg_lastweek[_am],
    ], axis=1)

    y_antic_day = anticipated_affect[_am]
    idx_daily_antic = ~np.isnan(y_antic_day)
    n_obs_anticipated_affect = int(idx_daily_antic.sum())

    if n_obs_anticipated_affect >= 2:
        cv_anticipated_affect = min(5, n_obs_anticipated_affect)
        model_anticipated_affect = RidgeCV(
            alphas=alpha_l2_list,
            fit_intercept=False,
            cv=cv_anticipated_affect,
            scoring="neg_mean_squared_error",
        )
        model_anticipated_affect.fit(
            anticipated_affect_cond_day[idx_daily_antic],
            y_antic_day[idx_daily_antic],
        )
        alpha_anticipated_affect_l2 = model_anticipated_affect.alpha_
        theta_anticipated_affect_mean = model_anticipated_affect.coef_
        pred_anticipated_affect = model_anticipated_affect.predict(
            anticipated_affect_cond_day
        )
        resid_obs_anticipated_affect = (
            y_antic_day[idx_daily_antic]
            - pred_anticipated_affect[idx_daily_antic]
        )
        resid_anticipated_affect = np.full_like(y_antic_day, np.nan, dtype=float)
        resid_anticipated_affect[idx_daily_antic] = resid_obs_anticipated_affect
        sigma2_anticipated_affect_mean = np.var(resid_obs_anticipated_affect)
    elif n_obs_anticipated_affect == 1:
        print(f"fallback to fixed alpha for anticipated_affect model for user {userid}")

        alpha_anticipated_affect_l2 = 1.0
        model_anticipated_affect = Ridge(alpha=alpha_anticipated_affect_l2, fit_intercept=False)
        model_anticipated_affect.fit(
            anticipated_affect_cond_day[idx_daily_antic],
            y_antic_day[idx_daily_antic],
        )
        theta_anticipated_affect_mean = model_anticipated_affect.coef_
        pred_anticipated_affect = model_anticipated_affect.predict(
            anticipated_affect_cond_day
        )
        resid_obs_anticipated_affect = (
            y_antic_day[idx_daily_antic]
            - pred_anticipated_affect[idx_daily_antic]
        )
        resid_anticipated_affect = np.full_like(y_antic_day, np.nan, dtype=float)
        resid_anticipated_affect[idx_daily_antic] = resid_obs_anticipated_affect
        sigma2_anticipated_affect_mean = np.var(resid_obs_anticipated_affect)
    else:
        print(f"no observed anticipated_affect values for user {userid}; using zero fallback")

        alpha_anticipated_affect_l2 = 1.0
        theta_anticipated_affect_mean = np.zeros(anticipated_affect_cond_day.shape[1], dtype=float)
        pred_anticipated_affect = np.zeros(len(y_antic_day), dtype=float)
        resid_anticipated_affect = np.full_like(y_antic_day, np.nan, dtype=float)
        sigma2_anticipated_affect_mean = 0.0

    print(f"The fit of anticipated_affect is good for user {userid}")


    #### Model 10: CAE model #### 
#### Model 10: CAE model, MixedLM with b_i for every variable ####

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

    CAE_cond = np.hstack([
        Intercept_sw[:, None],
        CAE_avg_lastweek_sw[:, None],
        week_sw[:, None],
        foursc_wk,
        antic_wk,
    ])

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

    alpha_CAE_l2 = None

    print(f"The full random-coefficient MixedLM CAE fit is ready for user {userid}")



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

    print(f"The full random-coefficient MixedLM CAE_short_avg fit is ready for user {userid}")

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

        "theta_recorded_physical_activity": json_float_list(theta_recorded_physical_activity_mean, digits),
        "theta_recorded_physical_activity_names": THETA_RECORDED_PHYSICAL_ACTIVITY_NAMES,

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
        "resid_recorded_physical_activity": json_float_list(resid_recorded_physical_activity, digits),
        "resid_active_status": json_float_list(resid_active_status, digits),
        "resid_ws_interaction": json_float_list(resid_ws_interaction, digits),
        "resid_fourSC": json_float_list(resid_fourSC, digits),
        "resid_antic": json_float_list(resid_anticipated_affect, digits),
        "resid_CAE": json_float_list(resid_CAE, digits),
        "resid_CAE_short_avg": json_float_list(resid_CAE_short_avg, digits),

        "alpha_prior2hour_step_count_l2": json_float(alpha_prior2hour_step_count_l2, digits),
        "alpha_recorded_physical_activity_l2": json_float(alpha_recorded_physical_activity_l2, digits),
        "alpha_active_status_l2": json_float(alpha_active_status_l2, digits),
        "alpha_ws_interaction_l2": json_float(alpha_ws_interaction_l2, digits),
        "alpha_fourSC_l2": json_float(alpha_fourSC_l2, digits),
        "alpha_anticipated_affect_l2": json_float(alpha_anticipated_affect_l2, digits),
        "alpha_CAE_l2": json_float(alpha_CAE_l2, digits),
        "alpha_CAE_short_avg_l2": json_float(alpha_CAE_short_avg_l2, digits),
    }

    predicted = {
        "pred_prior2hour_step_count": json_float_list(pred_prior2hour_step_count, digits),
        "pred_recorded_physical_activity": json_float_list(pred_recorded_physical_activity, digits),
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
        "theta_recorded_physical_activity",
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

# %%
