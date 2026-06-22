"""Estimate RL and PF priors from ``df_fit`` via per-participant regressions.

For every model below we use a two-stage empirical-Bayes scheme:

  1. **Pooled fit.** Stack all users' rows of ``df_fit`` into one design
     and fit a single ridge regression. This gives ``theta_pool``.
  2. **Per-user fits.** Fit a separate ridge regression on each user's
     rows to obtain ``theta_u`` and a per-user residual variance
     ``sigma2_u``. The across-user spread of ``theta_u`` characterises
     population heterogeneity.

Pool into:

         prior mean    = theta_pool                   (from the pooled fit)
         prior cov     = diag( var_u(theta_u) )       (across-user variance
                                                       of per-user theta_u,
                                                       one number per coefficient)
         sigma2        = mean_u( sigma2_u )         (average per-user
                                                       residual variance)

For the joint ``(eta, beta)`` Q-prior, the same scheme applies coordinate-wise
but the covariance is the **full** sample covariance of per-user
``(eta_u, beta_u)`` (not diagonal), so cross-block correlations are kept.

Models estimated
----------------
PF mediator m=0 (fourSC)        -> nu_0_MY[0], Gamma_0_MY[0], sigma2_MY[0]
PF mediator m=1 (antic)         -> nu_0_MY[1], Gamma_0_MY[1], sigma2_MY[1]
PF outcome  Y  (CAE)            -> nu_0_Y, Gamma_0_Y, sigma2_Y
PF outcome  tY (CAE_short)      -> nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y

PF features differ from the env generative model in three respects (see
``experiment.py`` ``get_pf_data``):
  1. fourSC / antic regressions are restricted to RL-controlled rows
     (Mon–Sat only — 12 slots / 6 days per week; Sunday is dropped).
  2. The fourSC PF model drops the ``stepCountNext4HourLag1`` AR-1 term
     entirely, and ``dailyAnticipatedAffectYesterday`` is cleared at the first
     day (Monday), matching the PF's week-boundary lag clearing.
  3. ``caeAverageLastWeek`` (in fourSC/antic/CAE) and ``perceivedUtilityLastWeek``
     (in fourSC/antic) are substituted at runtime with per-particle
     belief / agent E_w. This is a feature-substitution mechanism and does
     not change the population parameter being estimated, so the
     regressions here use the actual observed values from df_fit.
RL reward shaping               -> mu_0_reward, Sigma_0_reward, sigma2_reward
   (build_phi_action_rewardshaping;
    sum_{d,t} Delta_{d,t} psi on
    Delta_{6,2} * Y_w)
RL Q (no TD modify, beta)       -> mu_0_micro, Sigma_0_micro, sigma2_rl_micro
   (build_phi_action, fitted Q
    bootstrapping next-week Q;
    used by micro_g0 / micro_g05 / rs_g0 / rs_g05)

RL Q joint (eta, beta) for      -> mu_0_micro_mtd_joint,
   modified-TD-loss RLSVI          Sigma_0_micro_mtd_joint (FULL cov),
   (direct joint fit of the         p_eta_micro_mtd_joint,
    stacked bottleneck-state        sigma2_Q_mtd_joint
    TD loss; Sigma = full cov
    of per-user theta=(eta,beta))
   Used by mtd_g0 / mtd_g05 / rs_mtd_g0 / rs_mtd_g05 in experiment.py.

The no-TD-modify Q prior is fit by fitted-Q iteration (FQI):

    target_{k,d,t} = gamma_{d,t} * max_a Q(s_{next}, a)            (non-terminal)
    target_{k,6,2} = R_{k+1} + gamma_{6,2} * max_a Q(s_{1,1}^{k+1}, a)

The modified-TD prior is fit by direct joint FQI on the stacked
bottleneck-state TD loss for ``theta=(eta,beta)``.

with ``R_{k+1} = caeAverage_norm`` of week k and
``s_{1,1}^{k+1}`` the start-of-week state in df_fit week k+1.  The last
training week has no successor and is dropped.

Output is written to ``env_para_vanilla/rl_priors.json``; see
``load_estimated_priors`` for reading it back into experiment.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Feature builders for the RL Q / reward / bottleneck regressions.
# We import directly from algorithm.py to guarantee the prior dimensions stay
# synchronised with the agent's runtime phi.
from algorithm_helpers import (
    _cumulative_discount,
    build_phi_action,
    build_phi_action_rewardshaping,
    build_phi_bottleneck,
)


# ──────────────────────────────────────────────────────────────────
# Paths and constants
# ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(
    os.getenv("ADAPR_PROJECT_ROOT", str(Path(__file__).resolve().parent))
).expanduser().resolve()
COMBINED_DIR = Path(
    os.getenv(
        "ADAPR_COMBINED_DIR",
        "/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/rawdata/_combined",
    )
).expanduser().resolve()
WORK_DIR = PROJECT_ROOT / "env_para_vanilla"
OUTPUT_PATH = WORK_DIR / "rl_priors.json"

# Slot / day structure (mirrors experiment.py constants)
K_SLOTS_WEEK = 14                # 7 days × 2 slots in df_fit
SLOTS_PER_DAY = 2
DAYS_PER_WEEK_RL = 6             # RL handles Mon–Sat only
N_RL_SLOTS_WEEK = SLOTS_PER_DAY * DAYS_PER_WEEK_RL  # 12

# RL hyperparameters (must match experiment.py)
GAMMA_BAR = 0.5
GAMMA_DT_SCALAR = GAMMA_BAR ** (1.0 / 12.0)
GAMMA_DT = GAMMA_DT_SCALAR * np.ones((6, 2))   # _gamma_dt_micro(gamma_bar)
DELTA = _cumulative_discount(GAMMA_DT)         # Delta_{d,t} from slot (1,1)
DELTA_TERMINAL = float(DELTA[5, 1])             # Delta_{6,2}

# Mediator state shapes used by build_phi_action.
RL_MY_SHAPE = (6, 3)
RL_ME_SHAPE = (6, 4)

# Numerical hyperparameters
RIDGE_ALPHA_PF = 1.0     # ridge prior precision for PF mediator / Y / tY fits
RIDGE_ALPHA_RL = 1.0     # ridge prior precision for RL Q / reward / joint fits
MIN_SIGMA2 = 1e-6
N_FQI_ITERS = 25         # fitted-Q iterations per user


# ──────────────────────────────────────────────────────────────────
# Data loading / numeric helpers
# ──────────────────────────────────────────────────────────────────
def _week_fix_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Match ``4_perceivedUtility`` week relabel and ``week < 13`` filter."""
    df = df.copy()
    for userid in df["ParticipantIdentifier"].unique():
        vc = df.loc[df["ParticipantIdentifier"] == userid, "week"].value_counts()
        if vc.get(0, 0) > 2:
            m = df["ParticipantIdentifier"] == userid
            df.loc[m, "week"] = df.loc[m, "week"] + 1
    return df.loc[df["week"] < 13].copy()


def _filter_users_with_cae_obs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop users with no observed weekly CAE values after the week filter."""
    has_cae_obs = (
        df.groupby("ParticipantIdentifier", sort=False)["CAE_avg_norm"]
        .apply(lambda s: np.isfinite(s.to_numpy(dtype=float)).any())
    )
    keep_ids = has_cae_obs[has_cae_obs].index
    drop_ids = has_cae_obs[~has_cae_obs].index
    if len(drop_ids) > 0:
        print(
            "Excluding users with no CAE observations: "
            + ", ".join(str(int(uid)) for uid in drop_ids)
        )
    if len(keep_ids) == 0:
        raise ValueError("No users with observed CAE values remain after filtering.")
    return df.loc[df["ParticipantIdentifier"].isin(keep_ids)].copy()


def load_df_fit(path: Optional[Path] = None) -> pd.DataFrame:
    p = path or (COMBINED_DIR / "df_fit.csv")
    return _filter_users_with_cae_obs(_week_fix_and_filter(pd.read_csv(p)))



def _fill_nan(x: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Replace NaNs with the column mean; if all NaN, fall back to ``default``."""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return np.full_like(x, default, dtype=float)
    return np.where(np.isnan(x), float(np.nanmean(x)), x)


def _ridge_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = RIDGE_ALPHA_RL,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Per-user closed-form ridge: theta = (X'X + alpha I)^-1 X' y.

    Returns (theta, sigma2). Both ``None`` when no usable observations.
    Rows with non-finite features or response are dropped.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xo, yo = X[ok], y[ok]
    if Xo.shape[0] == 0:
        return None, None
    p = Xo.shape[1]
    # Some BLAS backends emit a spurious "divide by zero in matmul"
    # RuntimeWarning here when X contains many exact zeros; the math itself
    # is just a Gram + ridge solve, which is well-defined.
    with np.errstate(divide="ignore", invalid="ignore"):
        A = Xo.T @ Xo + alpha * np.eye(p)
        theta = np.linalg.solve(A, Xo.T @ yo)
        resid = yo - Xo @ theta
    if resid.size > 1:
        sigma2 = max(float(np.var(resid, ddof=1)), MIN_SIGMA2)
    else:
        sigma2 = 1.0
    return theta, sigma2


def _mean_user_sigma2(user_sigma2s: List[Optional[float]]) -> float:
    """Average finite per-user residual variances."""
    vals = [
        float(s)
        for s in user_sigma2s
        if s is not None and np.isfinite(s)
    ]
    if not vals:
        return 1.0
    return max(float(np.mean(vals)), MIN_SIGMA2)


def _pool_user_fits(
    pooled_theta: Optional[np.ndarray],
    user_thetas: List[Optional[np.ndarray]],
    user_sigma2s: List[Optional[float]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """Build a (prior mean, diag cov, sigma2) tuple under empirical-Bayes pooling.

    Inputs
    ------
    pooled_theta   coefficient vector from a single ridge regression on all
                   users' rows stacked together; becomes the prior mean.
    user_thetas    list of per-user theta_u (None for users that could not
                   be fit). Their across-user variance is the diagonal of
                   the prior covariance.
    user_sigma2s   list of per-user residual variances aligned with
                   ``user_thetas``; their average becomes sigma2.

    The prior mean comes from the pooled fit; sigma2 is the average of
    per-user residual variances; the per-user thetas characterise
    across-user heterogeneity.
    """
    if pooled_theta is None:
        return None, None, None
    pooled_theta = np.asarray(pooled_theta, dtype=float)
    p = pooled_theta.size

    thetas = [t for t in user_thetas if t is not None]
    if len(thetas) >= 2:
        arr = np.stack(thetas, axis=0)
        var = arr.var(axis=0, ddof=1)
    else:
        # Not enough per-user fits to estimate across-user variance.
        var = np.ones(p)
    var = np.maximum(var, MIN_SIGMA2)
    cov = np.diag(var)
    sigma2 = _mean_user_sigma2(user_sigma2s)
    return pooled_theta, cov, sigma2


# ──────────────────────────────────────────────────────────────────
# 1. PF mediator / Y / tY priors
# ──────────────────────────────────────────────────────────────────
THETA_FOURSC_NAMES = [
    "intercept", "yesterdayStepCount", "stepCountLast7DaysEma",
    "prior2HourStepCount", "activityCompletedLast7Days",     "activitySuggestionsSentLast7Days", "morningFitbitWearLast7Days",
    "salienceMessageSentYesterday", "activitySuggestionInteractLast7Days",
    "activeDaysLast7Days",
    "dayOfWeekNorm", "decisionTimeSlot", "perceivedUtilityLastWeek", "caeAverageLastWeek",
    "Ah", "Ah*yesterdayStepCount", "Ah*prior2HourStepCount", "Ah*activitySuggestionsSentLast7Days",
    "Ah*morningFitbitWearLast7Days",
    "Ah*salienceMessageSentYesterday", "Ah*activitySuggestionInteractLast7Days",
    "Ah*dayOfWeekNorm", "Ah*decisionTimeSlot",
    "Ah*perceivedUtilityLastWeek", "Ah*caeAverageLastWeek",
]

THETA_ANTIC_NAMES = [
    "intercept", "dailyAnticipatedAffectYesterday", "todayStepCount",
    "recordedPhysicalActivityToday", "activityStatusToday", "salienceMessageSentToday", "dayOfWeekNorm",
    "perceivedUtilityLastWeek", "caeAverageLastWeek",
    "ws_morning", "ws_afternoon",
    "ws_morning*salienceMessageSentToday", "ws_afternoon*salienceMessageSentToday",
    "ws_morning*dayOfWeekNorm", "ws_afternoon*dayOfWeekNorm",
    "ws_morning*perceivedUtilityLastWeek",
    "ws_afternoon*perceivedUtilityLastWeek",
    "ws_morning*caeAverageLastWeek", "ws_afternoon*caeAverageLastWeek",
]

THETA_CAE_NAMES = (
    ["intercept", "caeAverageLastWeek", "week_norm"]
    + [f"fourSC_slot_{j}" for j in range(14)]
    + [f"antic_day_{j}" for j in range(7)]
)

THETA_CAE_SHORT_NAMES = ["intercept", "caeAverage"]


# PF-specific column indices (must match experiment.py):
#   * ``antic_lag1_col  = 1``: dailyAnticipatedAffectYesterday in antic, cleared
#     at Monday (day 0).
# (The fourSC PF model omits the AR-1 lag term entirely, so there is no
#  fourSC lag column to clear.)
_PF_ANTIC_LAG1_COL  = 1

# PF only updates the mediator models on RL-controlled slots / days
# (Mon–Sat).  Sunday slots / day are excluded.
_PF_FOURSC_SLOTS_PER_WEEK = N_RL_SLOTS_WEEK    # 12 (= 6 days × 2 slots)
_PF_ANTIC_DAYS_PER_WEEK   = DAYS_PER_WEEK_RL   # 6


def _build_fourSC_design(dat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """fourSC PF design matrix and target.

    Differs from ``gen_fourSC_mean`` in two PF-specific ways:
      * Only the 12 RL-controlled slots per week (Mon–Sat × AM/PM) are kept;
        Sunday rows are dropped.
      * The ``stepCountNext4HourLag1`` AR-1 term used by ``gen_fourSC_mean``
        is dropped from the PF mediator model entirely.

    Per-particle CAE substitution (``cae=0`` base + ``cae_j`` delta) is a
    runtime substitution mechanism and does **not** change the population
    parameter being estimated, so the regression here uses the actual
    ``caeAverageLastWeek_norm`` column.
    """
    int_ = np.ones(len(dat))
    yest_step = _fill_nan(dat["YesterdayStepCount_norm"].to_numpy())
    seven_step = _fill_nan(dat["EMA_StepCount_norm"].to_numpy())
    prior2 = _fill_nan(dat["prior2hour_step_norm"].to_numpy())
    prev7rpa = _fill_nan(dat["Previous7DaysRPA"].to_numpy())
    rb = _fill_nan(
        dat["recent_burden_norm" if "recent_burden_norm" in dat.columns else "recentBurdenEma_norm"].to_numpy()
    )
    past7_wear = _fill_nan(dat["past7days_morning_wearing"].to_numpy())
    y_sal = _fill_nan(dat["yesterday_SalienceMessage"].to_numpy())
    i7w = _fill_nan(dat["Interacted_7d_walk"].to_numpy())
    act_frac7 = _fill_nan(dat["active_status_fraction_7days"].to_numpy())
    dayOfWeekNorm = dat["dow_norm"].to_numpy()
    dt_ = dat["DecisionTime"].to_numpy().astype(float)
    pu = _fill_nan(dat["perceived_utility_lastweek"].to_numpy())
    cae = _fill_nan(dat["CAE_avg_lastweek_norm"].to_numpy())
    Ah = dat["WalkingSuggestion"].to_numpy().astype(float)

    X = np.column_stack([
        int_, yest_step, seven_step, prior2, prev7rpa, rb,
        past7_wear, y_sal, i7w, act_frac7, dayOfWeekNorm, dt_, pu, cae,
        Ah, Ah * yest_step, Ah * prior2, Ah * rb,
        Ah * past7_wear, Ah * y_sal, Ah * i7w,
        Ah * dayOfWeekNorm, Ah * dt_, Ah * pu, Ah * cae,
    ])
    y = dat["4hour_step_norm"].to_numpy(dtype=float)
    assert X.shape[1] == len(THETA_FOURSC_NAMES)

    # ── PF-specific shaping ───────────────────────────────────────────
    n_w = len(dat) // K_SLOTS_WEEK
    X = X[: n_w * K_SLOTS_WEEK].reshape(n_w, K_SLOTS_WEEK, X.shape[1])
    y = y[: n_w * K_SLOTS_WEEK].reshape(n_w, K_SLOTS_WEEK)
    # Drop Sunday slots (indices 12, 13) → keep 12 RL-controlled slots per week.
    X = X[:, : _PF_FOURSC_SLOTS_PER_WEEK, :]
    y = y[:, : _PF_FOURSC_SLOTS_PER_WEEK]
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(-1)
    return X, y


def _build_antic_design(dat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Antic PF design matrix and target (one row per day, morning row).

    Differs from ``gen_antic_mean`` in two PF-specific ways:
      * Only 6 RL-controlled days per week (Mon–Sat) are kept; Sunday is
        dropped.
      * ``dailyAnticipatedAffectYesterday`` (the antic AR-1 column) is set to
        0 on the first day of each week (Monday), matching the PF's
        week-boundary lag clearing.

    Like for fourSC, the per-particle CAE substitution does not change
    the regression population parameter, so actual ``caeAverageLastWeek_norm``
    values are used here.
    """
    am = dat["DecisionTime"].to_numpy() == 0
    dat_am = dat.loc[am].reset_index(drop=True)
    n = len(dat_am)
    int_ = np.ones(n)
    antic_yest = _fill_nan(dat_am["anticipated_affect_yesterday_norm"].to_numpy())
    todayStepCount = _fill_nan(dat_am["TodayStepCount_norm"].to_numpy())
    rpa = _fill_nan(dat_am["RecordedPhysicalActivity"].to_numpy())
    act = _fill_nan(dat_am["active_status"].to_numpy())
    sal_msg = _fill_nan(dat_am["SalienceMessage"].to_numpy())
    dayOfWeekNorm = dat_am["dow_norm"].to_numpy()
    pu = _fill_nan(dat_am["perceived_utility_lastweek"].to_numpy())
    cae = _fill_nan(dat_am["CAE_avg_lastweek_norm"].to_numpy())

    # Per-day WS pair (rows of dat alternate AM, PM for each Date).
    WS_all = dat["WalkingSuggestion"].to_numpy().astype(float)
    pair = WS_all[: (len(dat) // 2) * 2].reshape(-1, 2)
    ws_m = pair[:n, 0]
    ws_a = pair[:n, 1]

    X = np.column_stack([
        int_, antic_yest, todayStepCount, rpa, act, sal_msg, dayOfWeekNorm, pu, cae,
        ws_m, ws_a,
        ws_m * sal_msg, ws_a * sal_msg,
        ws_m * dayOfWeekNorm, ws_a * dayOfWeekNorm,
        ws_m * pu, ws_a * pu,
        ws_m * cae, ws_a * cae,
    ])
    y = dat_am["anticipated_affect_norm"].to_numpy(dtype=float)
    assert X.shape[1] == len(THETA_ANTIC_NAMES)

    # ── PF-specific shaping ───────────────────────────────────────────
    days_per_week = 7
    n_w = n // days_per_week
    X = X[: n_w * days_per_week].reshape(n_w, days_per_week, X.shape[1])
    y = y[: n_w * days_per_week].reshape(n_w, days_per_week)
    # Drop Sunday (day index 6) → keep 6 RL-controlled days per week.
    X = X[:, : _PF_ANTIC_DAYS_PER_WEEK, :]
    y = y[:, : _PF_ANTIC_DAYS_PER_WEEK]
    # Clear lag at the first day of every week (Monday).
    X[:, 0, _PF_ANTIC_LAG1_COL] = 0.0
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(-1)
    return X, y


def _build_CAE_design(dat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Weekly CAE design (24 features = [1, CAE_lw, week, 14 fourSC slots, 7 antic days])."""
    n_w = len(dat) // K_SLOTS_WEEK
    dat = dat.iloc[: n_w * K_SLOTS_WEEK]
    cae_y = dat["CAE_avg_norm"].to_numpy().reshape(-1, K_SLOTS_WEEK)[:, 0]
    cae_lw = _fill_nan(dat["CAE_avg_lastweek_norm"].to_numpy()).reshape(-1, K_SLOTS_WEEK)[:, 0]
    week = dat["week_norm"].to_numpy().reshape(-1, K_SLOTS_WEEK)[:, 0]

    mu_fSC = float(np.nanmean(dat["4hour_step_norm"])) if dat["4hour_step_norm"].notna().any() else 0.0
    mu_antic = float(np.nanmean(dat["anticipated_affect_norm"])) if dat["anticipated_affect_norm"].notna().any() else 0.0

    foursc = dat["4hour_step_norm"].to_numpy().reshape(-1, K_SLOTS_WEEK)
    foursc = np.where(np.isnan(foursc), mu_fSC, foursc)
    antic = dat["anticipated_affect_norm"].to_numpy().reshape(-1, K_SLOTS_WEEK)
    antic = np.where(np.isnan(antic), mu_antic, antic).reshape(-1, 7, 2).mean(axis=2)

    X = np.hstack([
        np.ones((n_w, 1)), cae_lw[:, None], week[:, None], foursc, antic,
    ])
    assert X.shape[1] == len(THETA_CAE_NAMES)
    return X, cae_y


def _build_CAE_short_design(dat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    n_w = len(dat) // K_SLOTS_WEEK
    dat = dat.iloc[: n_w * K_SLOTS_WEEK]
    cae_short = dat["CAE_short_avg_norm"].to_numpy().reshape(-1, K_SLOTS_WEEK)[:, 0]
    cae_avg = _fill_nan(dat["CAE_avg_norm"].to_numpy().reshape(-1, K_SLOTS_WEEK)[:, 0])
    X = np.column_stack([np.ones(len(cae_short)), cae_avg])
    return X, cae_short


def fit_pf_priors(df_fit: pd.DataFrame) -> Dict[str, Any]:
    """Pooled-mean / per-user-variance PF / Y / tY priors.

    For each of fourSC, antic, CAE, CAE_short:
      * stack all users' design rows and fit one ridge regression
        -> ``theta_pool`` (prior mean);
      * fit a ridge regression on each user separately
        -> ``theta_u`` collection for the diagonal across-user prior
           covariance, and ``sigma2_u`` for the average residual variance.
    """
    by_user = df_fit.groupby("ParticipantIdentifier", sort=False)

    # All-user stacked designs for the pooled fit.
    X_fSC_all, y_fSC_all = [], []
    X_ant_all, y_ant_all = [], []
    X_Y_all,   y_Y_all   = [], []
    X_tY_all,  y_tY_all  = [], []
    # Per-user theta / sigma2 collections for across-user pooling.
    fSC_thetas: List[Optional[np.ndarray]] = []
    fSC_sigma2s: List[Optional[float]] = []
    ant_thetas: List[Optional[np.ndarray]] = []
    ant_sigma2s: List[Optional[float]] = []
    Y_thetas:   List[Optional[np.ndarray]] = []
    Y_sigma2s:  List[Optional[float]] = []
    tY_thetas:  List[Optional[np.ndarray]] = []
    tY_sigma2s: List[Optional[float]] = []

    for _uid, dat in by_user:
        dat = dat.sort_values(["Date", "DecisionTime"]).reset_index(drop=True)

        Xf, yf = _build_fourSC_design(dat)
        X_fSC_all.append(Xf); y_fSC_all.append(yf)
        th, s2 = _ridge_fit(Xf, yf, alpha=RIDGE_ALPHA_PF)
        fSC_thetas.append(th)
        fSC_sigma2s.append(s2)

        Xa, ya = _build_antic_design(dat)
        X_ant_all.append(Xa); y_ant_all.append(ya)
        th, s2 = _ridge_fit(Xa, ya, alpha=RIDGE_ALPHA_PF)
        ant_thetas.append(th)
        ant_sigma2s.append(s2)

        XC, yC = _build_CAE_design(dat)
        X_Y_all.append(XC); y_Y_all.append(yC)
        th, s2 = _ridge_fit(XC, yC, alpha=RIDGE_ALPHA_PF)
        Y_thetas.append(th)
        Y_sigma2s.append(s2)

        Xs, ys = _build_CAE_short_design(dat)
        X_tY_all.append(Xs); y_tY_all.append(ys)
        th, s2 = _ridge_fit(Xs, ys, alpha=RIDGE_ALPHA_PF)
        tY_thetas.append(th)
        tY_sigma2s.append(s2)

    th_fSC_pool, _ = _ridge_fit(
        np.vstack(X_fSC_all), np.concatenate(y_fSC_all), alpha=RIDGE_ALPHA_PF)
    th_ant_pool, _ = _ridge_fit(
        np.vstack(X_ant_all), np.concatenate(y_ant_all), alpha=RIDGE_ALPHA_PF)
    th_Y_pool, _ = _ridge_fit(
        np.vstack(X_Y_all),   np.concatenate(y_Y_all),   alpha=RIDGE_ALPHA_PF)
    th_tY_pool, _ = _ridge_fit(
        np.vstack(X_tY_all),  np.concatenate(y_tY_all),  alpha=RIDGE_ALPHA_PF)

    nu_fSC, G_fSC, s2_fSC = _pool_user_fits(th_fSC_pool, fSC_thetas, fSC_sigma2s)
    nu_ant, G_ant, s2_ant = _pool_user_fits(th_ant_pool, ant_thetas, ant_sigma2s)
    nu_Y,   G_Y,   s2_Y   = _pool_user_fits(th_Y_pool,   Y_thetas,   Y_sigma2s)
    nu_tY,  G_tY,  s2_tY  = _pool_user_fits(th_tY_pool,  tY_thetas,  tY_sigma2s)

    return {
        "fourSC":    {"nu_0": nu_fSC, "Gamma_0": G_fSC, "sigma2": s2_fSC,
                       "names": THETA_FOURSC_NAMES},
        "antic":     {"nu_0": nu_ant, "Gamma_0": G_ant, "sigma2": s2_ant,
                       "names": THETA_ANTIC_NAMES},
        "CAE":       {"nu_0": nu_Y,   "Gamma_0": G_Y,   "sigma2": s2_Y,
                       "names": THETA_CAE_NAMES},
        "CAE_short": {"nu_0": nu_tY,  "Gamma_0": G_tY,  "sigma2": s2_tY,
                       "names": THETA_CAE_SHORT_NAMES},
    }


# ──────────────────────────────────────────────────────────────────
# 2. Per-user phi_action / phi_rs / phi_bottleneck row builders
# ──────────────────────────────────────────────────────────────────
def _build_rl_context_vector_from_row(row) -> np.ndarray:
    """Match ``experiment.build_rl_context_vector``."""
    return np.array([
        float(row["YesterdayStepCount_norm"]),
        float(row["EMA_StepCount_norm"]),
        float(row["prior2hour_step_norm"]),
        float(row["Previous7DaysRPA"]),
        float(row["active_status_fraction_7days"]),
        float(row["recent_burden_norm" if "recent_burden_norm" in row.index else "recentBurdenEma_norm"]),
        float(row["yesterday_SalienceMessage"]),
        float(row["Interacted_7d_walk"]),
        0.0,
    ], dtype=float)


def _user_weekly_tensors(dat: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Pre-build all per-week / per-slot quantities needed for RL phi builders.

    ``dat`` must be one participant, sorted by (Date, DecisionTime), with
    a length that is a multiple of K_SLOTS_WEEK.  Day order within each week
    is Monday .. Sunday (dayOfWeekNorm = 1 .. 7).  Rows are NaN-filled with the
    participant's mean so the resulting tensors are finite.
    """
    n_w = len(dat) // K_SLOTS_WEEK
    dat = dat.iloc[: n_w * K_SLOTS_WEEK].reset_index(drop=True)

    cols = [
        "YesterdayStepCount_norm", "EMA_StepCount_norm", "prior2hour_step_norm",
        "Previous7DaysRPA", "recent_burden_norm",
        "active_status_fraction_7days",
        "yesterday_SalienceMessage", "Interacted_7d_walk",
        "anticipated_affect_yesterday_norm",
        "4hour_step_norm", "HourlyPageviewCount_norm",
        "morning_wearing", "daily_present",
        "anticipated_affect_norm",
        "perceived_utility_lastweek", "CAE_avg_lastweek_norm",
        "CAE_avg_norm", "WalkingSuggestion",
    ]
    for c in cols:
        dat[c] = _fill_nan(dat[c].to_numpy())

    n_rl = N_RL_SLOTS_WEEK   # 12 slots Mon-Sat
    p_C = 9                  # build_rl_context_vector dim

    # Per-slot quantities for d in 1..6 (Mon-Sat).
    C_slot = np.zeros((n_w, n_rl, p_C))
    E_w_slot = np.zeros((n_w, n_rl))
    b_hat_slot = np.zeros((n_w, n_rl))
    A_slot = np.zeros((n_w, n_rl), dtype=int)

    # Per-week M_Y / M_E (mediator memory the agent would see).
    M_Y_week = np.zeros((n_w, 6, 3))
    M_E_week = np.zeros((n_w, 6, 4))

    # Per-week start-of-week proxies (for bottleneck phi at S_{k,0}).
    E_w_start = np.zeros(n_w)
    b_hat_start = np.zeros(n_w)

    # Per-week realized weekly CAE (reward target).
    R_week = np.zeros(n_w)

    for k in range(n_w):
        w0 = k * K_SLOTS_WEEK
        # First row of week: gives perceivedUtilityLastWeek + caeAverageLastWeek_norm
        E_w_start[k] = float(dat["perceived_utility_lastweek"].iloc[w0])
        b_hat_start[k] = float(dat["CAE_avg_lastweek_norm"].iloc[w0])
        R_week[k] = float(dat["CAE_avg_norm"].iloc[w0])

        for d in range(1, 7):               # RL days Mon..Sat
            for t in range(1, 3):           # AM, PM
                slot = w0 + (d - 1) * 2 + (t - 1)
                rl_idx = (d - 1) * 2 + (t - 1)
                row = dat.iloc[slot]
                C_slot[k, rl_idx] = _build_rl_context_vector_from_row(row)
                E_w_slot[k, rl_idx] = float(row["perceived_utility_lastweek"])
                b_hat_slot[k, rl_idx] = float(row["CAE_avg_lastweek_norm"])
                A_slot[k, rl_idx] = int(row["WalkingSuggestion"])

        # Per-week mediator memory: fourSC (d, AM/PM), antic (d), pageview, fitbit, dailySurveyComplete.
        for d in range(1, 7):
            morning_slot = w0 + (d - 1) * 2
            afternoon_slot = morning_slot + 1
            M_Y_week[k, d - 1, 0] = float(dat["4hour_step_norm"].iloc[morning_slot])
            M_Y_week[k, d - 1, 1] = float(dat["4hour_step_norm"].iloc[afternoon_slot])
            M_Y_week[k, d - 1, 2] = float(dat["anticipated_affect_norm"].iloc[morning_slot])

            M_E_week[k, d - 1, 0] = float(dat["HourlyPageviewCount_norm"].iloc[morning_slot])
            M_E_week[k, d - 1, 1] = float(dat["HourlyPageviewCount_norm"].iloc[afternoon_slot])
            M_E_week[k, d - 1, 2] = float(dat["morning_wearing"].iloc[morning_slot])
            M_E_week[k, d - 1, 3] = float(dat["daily_present"].iloc[morning_slot])

    return {
        "n_w":         n_w,
        "C_slot":      C_slot,
        "E_w_slot":    E_w_slot,
        "b_hat_slot":  b_hat_slot,
        "A_slot":      A_slot,
        "M_Y_week":    M_Y_week,
        "M_E_week":    M_E_week,
        "E_w_start":   E_w_start,
        "b_hat_start": b_hat_start,
        "R_week":      R_week,
    }


def _slot_state(t_dict: Dict[str, np.ndarray], k: int, rl_idx: int) -> Dict[str, Any]:
    """Build the ``state`` dict expected by ``build_phi_action``."""
    return {
        "E_w": float(t_dict["E_w_slot"][k, rl_idx]),
        "M_Y": t_dict["M_Y_week"][k],
        "M_E": t_dict["M_E_week"][k],
        "C":   t_dict["C_slot"][k, rl_idx],
    }


def _phi_action(t_dict, k, rl_idx, action) -> np.ndarray:
    d = rl_idx // 2 + 1
    t = rl_idx %  2 + 1
    return build_phi_action(
        b_hat=float(t_dict["b_hat_slot"][k, rl_idx]),
        b_tilde=0.0,
        state=_slot_state(t_dict, k, rl_idx),
        d=d, t=t,
        action=int(action),
    )


def _phi_rs(t_dict, k, rl_idx) -> np.ndarray:
    d = rl_idx // 2 + 1
    t = rl_idx %  2 + 1
    return build_phi_action_rewardshaping(
        b_hat=float(t_dict["b_hat_slot"][k, rl_idx]),
        b_tilde=0.0,
        state=_slot_state(t_dict, k, rl_idx),
        d=d, t=t,
    )


def _phi_rs_week_discounted(t_dict, k) -> np.ndarray:
    """Discount-weighted weekly reward-shaping feature (matches ``build_reward_shaping_training_data``).

        Phi_week(k) = sum_{d,t} Delta_{d,t} * psi(S_{k,d,t})
    """
    phi_week = np.zeros_like(_phi_rs(t_dict, k, 0))
    for rl_idx in range(N_RL_SLOTS_WEEK):
        d = rl_idx // 2 + 1
        t = rl_idx %  2 + 1
        phi_week += DELTA[d - 1, t - 1] * _phi_rs(t_dict, k, rl_idx)
    return phi_week


def _phi_bottleneck_start(t_dict, k) -> np.ndarray:
    state = {
        "E_w": float(t_dict["E_w_start"][k]),
        "M_Y": np.zeros(RL_MY_SHAPE),
        "M_E": np.zeros(RL_ME_SHAPE),
        "C":   np.zeros(t_dict["C_slot"].shape[2]),
    }
    return build_phi_bottleneck(
        b_hat=float(t_dict["b_hat_start"][k]),
        b_tilde=0.0,
        state=state,
    )


# ──────────────────────────────────────────────────────────────────
# 3. Reward-shaping eta prior
# ──────────────────────────────────────────────────────────────────
def fit_reward_prior(df_fit: pd.DataFrame) -> Dict[str, Any]:
    """Reward-shaping prior under the discount-corrected regression.

    Replicates ``build_reward_shaping_training_data`` / ``compute_reward_shaping_eta``:

        feature  = sum_{d,t} Delta_{d,t} * psi(S_{k,d,t})
        target   = Delta_{6,2} * Y_w

    where ``Y_w`` is the weekly CAE (``caeAverage_norm``). Shaped intermediate
    rewards and the terminal compensation ``R_{w,add}`` are *not* included
    here — those enter only the Q-function FQI targets at runtime.

    Prior mean is the pooled fit across all users' weeks; diagonal prior
    covariance comes from per-user fits.
    """
    thetas: List[Optional[np.ndarray]] = []
    sigma2s: List[Optional[float]] = []
    Phi_all, y_all = [], []
    for _uid, dat in df_fit.groupby("ParticipantIdentifier", sort=False):
        dat = dat.sort_values(["Date", "DecisionTime"]).reset_index(drop=True)
        td = _user_weekly_tensors(dat)
        n_w = td["n_w"]
        if n_w < 2:
            thetas.append(None)
            sigma2s.append(None)
            continue
        Phi = np.stack([_phi_rs_week_discounted(td, k) for k in range(n_w)])
        y = DELTA_TERMINAL * td["R_week"]
        Phi_all.append(Phi); y_all.append(y)
        theta, s2 = _ridge_fit(Phi, y, alpha=RIDGE_ALPHA_RL)
        thetas.append(theta)
        sigma2s.append(s2)

    if Phi_all:
        pooled_theta, _ = _ridge_fit(
            np.vstack(Phi_all), np.concatenate(y_all), alpha=RIDGE_ALPHA_RL)
    else:
        pooled_theta = None

    mu, Sigma, sigma2 = _pool_user_fits(pooled_theta, thetas, sigma2s)
    return {"mu_0": mu, "Sigma_0": Sigma, "sigma2": sigma2}


# ──────────────────────────────────────────────────────────────────
# 5. Fitted-Q priors (with and without TD-modify)
# ──────────────────────────────────────────────────────────────────
def _precompute_q_features(
    td: Dict[str, np.ndarray],
    use_td_modify: bool,
) -> Optional[Dict[str, Any]]:
    """Per-user phi tensors for FQI; ``None`` if the user has < 2 weeks.

    The same precompute is used for per-user FQI (one entry per user) and
    for pooled FQI (one entry per user, then stacked). The direct joint
    modified-TD fit builds bottleneck features without a staged alpha estimate.
    """
    n_w = td["n_w"]
    if n_w < 2:
        return None
    n_train = n_w - 1                       # drop last week (no successor)
    p_phi = _phi_action(td, 0, 0, 0).size

    # phi_obs[k, idx] = phi_action(s_{k,d,t}, a_{k,d,t}^obs) – training row.
    # phi_a0/a1[k, idx] = phi_action with the two possible actions – for max_a Q
    # at the successor slot.  Storing both pays off vs rebuilding 25× in FQI.
    phi_obs = np.zeros((n_train, N_RL_SLOTS_WEEK, p_phi))
    phi_a0  = np.zeros((n_train, N_RL_SLOTS_WEEK, p_phi))
    phi_a1  = np.zeros((n_train, N_RL_SLOTS_WEEK, p_phi))
    for k in range(n_train):
        for idx in range(N_RL_SLOTS_WEEK):
            phi_obs[k, idx] = _phi_action(td, k, idx, int(td["A_slot"][k, idx]))
            phi_a0[k, idx]  = _phi_action(td, k, idx, 0)
            phi_a1[k, idx]  = _phi_action(td, k, idx, 1)

    # Successor features for the terminal slot (6, 2): row 0 of week k+1.
    phi_next_a0 = np.zeros((n_train, p_phi))
    phi_next_a1 = np.zeros((n_train, p_phi))
    for k in range(n_train):
        phi_next_a0[k] = _phi_action(td, k + 1, 0, 0)
        phi_next_a1[k] = _phi_action(td, k + 1, 0, 1)

    phi_next_bot = None
    phi_bot = None
    p_eta = None
    if use_td_modify:
        p_eta = _phi_bottleneck_start(td, 0).size
        phi_bot = np.zeros((n_train, p_eta))
        phi_next_bot = np.zeros((n_train, p_eta))
        for k in range(n_train):
            phi_bot[k] = _phi_bottleneck_start(td, k)
            phi_next_bot[k] = _phi_bottleneck_start(td, k + 1)

    return {
        "phi_obs":     phi_obs,
        "phi_a0":      phi_a0,
        "phi_a1":      phi_a1,
        "phi_next_a0": phi_next_a0,
        "phi_next_a1": phi_next_a1,
        "phi_bot":      phi_bot,
        "phi_next_bot": phi_next_bot,
        "R":           td["R_week"][:n_train],
        "n_train":     n_train,
        "p_phi":       p_phi,
        "p_eta":       p_eta,
    }


def _fqi_iterate(
    feats_list: List[Dict[str, Any]],
    alpha_for_boot: Optional[np.ndarray],
    use_td_modify: bool,
    n_iters: int = N_FQI_ITERS,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    """Run FQI on one or more users' precomputed features.

    With one entry in ``feats_list`` this is per-user FQI; with many it is a
    single FQI on all users' rows concatenated together. Each FQI iteration
    rebuilds targets per (user, week, slot) using the current ``theta`` and
    refits a ridge regression on the stacked rows.

    Returns ``(theta, sigma2, None)``, where ``sigma2`` is the residual
    variance over the fitted-Q rows.
    """
    if not feats_list:
        return None, None, None
    p_phi = feats_list[0]["p_phi"]

    X_all = np.concatenate([f["phi_obs"].reshape(-1, p_phi) for f in feats_list])

    theta = np.zeros(p_phi)
    y_all = np.zeros(X_all.shape[0])
    for _ in range(n_iters):
        chunks = []
        for f in feats_list:
            n_train = f["n_train"]
            targets = np.zeros((n_train, N_RL_SLOTS_WEEK))
            for k in range(n_train):
                for idx in range(N_RL_SLOTS_WEEK):
                    if idx < N_RL_SLOTS_WEEK - 1:
                        # non-terminal: successor is same-week next slot
                        q0 = f["phi_a0"][k, idx + 1] @ theta
                        q1 = f["phi_a1"][k, idx + 1] @ theta
                        targets[k, idx] = GAMMA_DT_SCALAR * max(q0, q1)
                    else:
                        # terminal (6, 2): reward + γ_terminal · bootstrap
                        if use_td_modify and alpha_for_boot is not None:
                            boot = float(f["phi_next_bot"][k] @ alpha_for_boot)
                        else:
                            q0 = f["phi_next_a0"][k] @ theta
                            q1 = f["phi_next_a1"][k] @ theta
                            boot = float(max(q0, q1))
                        targets[k, idx] = f["R"][k] + GAMMA_DT_SCALAR * boot
            chunks.append(targets.reshape(-1))
        y_all = np.concatenate(chunks)
        theta_new, _ = _ridge_fit(X_all, y_all, alpha=RIDGE_ALPHA_RL)
        if theta_new is None:
            return None, None, None
        if np.linalg.norm(theta_new - theta) < 1e-6 * (np.linalg.norm(theta) + 1e-12):
            theta = theta_new
            break
        theta = theta_new

    resid = y_all - X_all @ theta
    sigma2 = (max(float(np.var(resid, ddof=1)), MIN_SIGMA2)
              if resid.size > 1 else 1.0)
    return theta, sigma2, None


def _mtd_joint_design(
    feats_list: List[Dict[str, Any]],
    beta_for_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Stack the three bottleneck-state TD loss blocks for theta=(alpha,beta)."""
    if not feats_list:
        return np.empty((0, 0)), np.empty((0,)), 0

    p_eta = int(feats_list[0]["p_eta"])
    p_beta = int(feats_list[0]["p_phi"])
    p = p_eta + p_beta
    beta_for_targets = np.asarray(beta_for_targets, dtype=float).ravel()

    X_chunks, y_chunks = [], []
    for feats in feats_list:
        n_train = feats["n_train"]

        # Block A: Q(S_{w,1,1}, a*) - V(S_{w,0}) = 0.
        q0_first = feats["phi_a0"][:, 0, :] @ beta_for_targets
        q1_first = feats["phi_a1"][:, 0, :] @ beta_for_targets
        phi_first = np.where(
            (q1_first >= q0_first)[:, None],
            feats["phi_a1"][:, 0, :],
            feats["phi_a0"][:, 0, :],
        )
        X_A = np.zeros((n_train, p))
        X_A[:, :p_eta] = -feats["phi_bot"]
        X_A[:, p_eta:] = phi_first
        y_A = np.zeros(n_train)

        # Block B: non-terminal TD rows.
        n_nt = n_train * (N_RL_SLOTS_WEEK - 1)
        X_B = np.zeros((n_nt, p))
        y_B = np.zeros(n_nt)
        row = 0
        for k in range(n_train):
            for idx in range(N_RL_SLOTS_WEEK - 1):
                X_B[row, p_eta:] = feats["phi_obs"][k, idx]
                q0_next = feats["phi_a0"][k, idx + 1] @ beta_for_targets
                q1_next = feats["phi_a1"][k, idx + 1] @ beta_for_targets
                y_B[row] = GAMMA_DT_SCALAR * max(q0_next, q1_next)
                row += 1

        # Block C: terminal row with next-week bottleneck bootstrap in design.
        X_C = np.zeros((n_train, p))
        X_C[:, :p_eta] = -GAMMA_DT_SCALAR * feats["phi_next_bot"]
        X_C[:, p_eta:] = feats["phi_obs"][:, N_RL_SLOTS_WEEK - 1, :]
        y_C = feats["R"]

        X_chunks.extend([X_A, X_B, X_C])
        y_chunks.extend([y_A, y_B, y_C])

    return np.vstack(X_chunks), np.concatenate(y_chunks), p_eta


def _joint_fqi_iterate(
    feats_list: List[Dict[str, Any]],
    n_iters: int = N_FQI_ITERS,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int]]:
    """Directly fit theta=(alpha,beta) under the bottleneck-state TD loss."""
    if not feats_list:
        return None, None, None

    p_eta = int(feats_list[0]["p_eta"])
    p_beta = int(feats_list[0]["p_phi"])
    theta = np.zeros(p_eta + p_beta)
    sigma2 = None

    for _ in range(n_iters):
        X, y, p_eta = _mtd_joint_design(feats_list, theta[p_eta:])
        theta_new, sigma2 = _ridge_fit(X, y, alpha=RIDGE_ALPHA_RL)
        if theta_new is None:
            return None, None, None
        if np.linalg.norm(theta_new - theta) < 1e-6 * (
            np.linalg.norm(theta) + 1e-12
        ):
            theta = theta_new
            break
        theta = theta_new

    X, y, _ = _mtd_joint_design(feats_list, theta[p_eta:])
    resid = y - X @ theta
    sigma2 = (max(float(np.var(resid, ddof=1)), MIN_SIGMA2)
              if resid.size > 1 else 1.0)
    return theta, sigma2, p_eta


def fit_q_prior(df_fit: pd.DataFrame) -> Dict[str, Any]:
    """Q-function prior (no TD-modify) under pooled-mean + per-user-variance pooling.

    Prior mean comes from a single FQI run on every user's rows stacked
    together; the diagonal prior covariance comes from the across-user
    variance of per-user FQI thetas.
    """
    th_nomod_users: List[Optional[np.ndarray]] = []
    s2_nomod_users: List[Optional[float]] = []
    feats_nomod_pool: List[Dict[str, Any]] = []

    for _uid, dat in df_fit.groupby("ParticipantIdentifier", sort=False):
        dat = dat.sort_values(["Date", "DecisionTime"]).reset_index(drop=True)
        td = _user_weekly_tensors(dat)

        feats_u_no = _precompute_q_features(td, use_td_modify=False)
        if feats_u_no is None:
            th_nomod_users.append(None)
            s2_nomod_users.append(None)
        else:
            th, s2_u, _ = _fqi_iterate([feats_u_no], None, use_td_modify=False)
            th_nomod_users.append(th)
            s2_nomod_users.append(s2_u)
            feats_nomod_pool.append(feats_u_no)

    th_nomod_pool, _, _ = _fqi_iterate(
        feats_nomod_pool, None, use_td_modify=False)
    mu, Sigma, sigma2 = _pool_user_fits(
        th_nomod_pool, th_nomod_users, s2_nomod_users)
    return {"mu_0": mu, "Sigma_0": Sigma, "sigma2": sigma2}


# ──────────────────────────────────────────────────────────────────
# 5b. Joint (eta, beta) prior for the modified-TD-loss RLSVI
# ──────────────────────────────────────────────────────────────────
def _pool_joint_user_fits(
    pooled_theta: Optional[np.ndarray],
    user_thetas: List[Optional[np.ndarray]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    """Joint-prior analogue of ``_pool_user_fits``.

    The mean ``mu`` is taken from ``pooled_theta`` (the joint coefficient
    vector obtained from pooled fits on the full multi-user dataset).
    Unlike ``_pool_user_fits``, ``Sigma`` is the **full** sample covariance
    across users of the per-user joint theta vectors, not a diagonal of
    per-coordinate variances. Output is regularised to be PSD via a tiny
    eigenvalue floor.
    """
    if pooled_theta is None:
        return None, None, 0
    pooled_theta = np.asarray(pooled_theta, dtype=float)

    thetas = [t for t in user_thetas if t is not None]
    n_used = len(thetas)
    p = pooled_theta.size
    if n_used >= 2:
        arr = np.stack(thetas, axis=0)              # (n_users, p)
        Sigma = np.cov(arr, rowvar=False, ddof=1)
        # When p == 1 numpy returns a 0-d array; force 2-d for consistency.
        Sigma = np.atleast_2d(Sigma)
    else:
        Sigma = np.eye(p)
    Sigma = 0.5 * (Sigma + Sigma.T)
    if Sigma.shape[0] > 0:
        min_eig = float(np.linalg.eigvalsh(Sigma).min())
        if min_eig < MIN_SIGMA2:
            Sigma = Sigma + (MIN_SIGMA2 - min_eig) * np.eye(Sigma.shape[0])
    return pooled_theta, Sigma, n_used


def fit_q_td_modify_joint_prior(df_fit: pd.DataFrame) -> Dict[str, Any]:
    """Joint prior on theta=(eta,beta) from the bottleneck-state TD loss.

    The pooled prior mean is fit by running the direct joint FQI objective on
    all users' rows stacked together. The prior covariance is the full sample
    covariance across per-user joint fits, preserving eta-beta cross terms.
    ``sigma2_Q`` is the average of per-user residual variances from the same
    stacked loss.

    Returns
    -------
    dict with keys
        ``mu_0``                joint mean,    shape ``(p_eta + p_beta,)``
        ``Sigma_0``             joint cov,     shape ``(p, p)`` (FULL)
        ``p_eta``               int            number of eta coordinates
        ``sigma2_Q``            float          shared joint-loss noise variance
        ``n_users``             int            number of users contributing
    """
    user_thetas: List[Optional[np.ndarray]] = []
    s2_Q_users: List[Optional[float]] = []
    feats_pool: List[Dict[str, Any]] = []
    p_eta: Optional[int] = None

    for _uid, dat in df_fit.groupby("ParticipantIdentifier", sort=False):
        dat = dat.sort_values(["Date", "DecisionTime"]).reset_index(drop=True)
        td = _user_weekly_tensors(dat)

        feats_u = _precompute_q_features(td, use_td_modify=True)
        if feats_u is None:
            user_thetas.append(None)
            s2_Q_users.append(None)
            continue

        theta_u, s2_u, p_eta_u = _joint_fqi_iterate([feats_u])
        if theta_u is None:
            user_thetas.append(None)
            s2_Q_users.append(None)
            continue
        if p_eta is None:
            p_eta = int(p_eta_u)
        user_thetas.append(theta_u)
        s2_Q_users.append(s2_u)
        feats_pool.append(feats_u)

    theta_pool, _, p_eta_pool = _joint_fqi_iterate(feats_pool)
    if p_eta is None and p_eta_pool is not None:
        p_eta = int(p_eta_pool)

    mu_0, Sigma_0, n_used = _pool_joint_user_fits(theta_pool, user_thetas)

    return {
        "mu_0":              mu_0,
        "Sigma_0":           Sigma_0,
        "p_eta":             int(p_eta) if p_eta is not None else None,
        "sigma2_Q":          _mean_user_sigma2(s2_Q_users),
        "n_users":           int(n_used),
    }


# ──────────────────────────────────────────────────────────────────
# 6. JSON I/O
# ──────────────────────────────────────────────────────────────────
def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_priors(priors: Dict[str, Any], path: Path = OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(priors), f, indent=2, allow_nan=False)
    return path


def load_estimated_priors(path: Path = OUTPUT_PATH) -> Dict[str, Any]:
    """Read ``rl_priors.json`` and convert lists back to numpy arrays.

    Returned keys match the structure produced by ``main`` below. The keys
    ``nu_0_MY`` / ``Gamma_0_MY`` / ``sigma2_MY`` are lists of length 2
    (m=0 fourSC, m=1 antic) so they can plug directly into experiment.py.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    def arr(x):
        return None if x is None else np.asarray(x, dtype=float)

    pf = raw["pf"]
    rw = raw["reward"]
    qn = raw["q_no_td_modify"]
    qj = raw.get("q_td_modify_joint")

    out = {
        "nu_0_MY":     [arr(pf["fourSC"]["nu_0"]), arr(pf["antic"]["nu_0"])],
        "Gamma_0_MY":  [arr(pf["fourSC"]["Gamma_0"]), arr(pf["antic"]["Gamma_0"])],
        "sigma2_MY":   [float(pf["fourSC"]["sigma2"]), float(pf["antic"]["sigma2"])],
        "nu_0_Y":      arr(pf["CAE"]["nu_0"]),
        "Gamma_0_Y":   arr(pf["CAE"]["Gamma_0"]),
        "sigma2_Y":    float(pf["CAE"]["sigma2"]),
        "nu_0_tilde_Y":     arr(pf["CAE_short"]["nu_0"]),
        "Gamma_0_tilde_Y":  arr(pf["CAE_short"]["Gamma_0"]),
        "sigma2_tilde_Y":   float(pf["CAE_short"]["sigma2"]),
        "mu_0_reward":     arr(rw["mu_0"]),
        "Sigma_0_reward":  arr(rw["Sigma_0"]),
        "sigma2_reward":   float(rw["sigma2"]),
        "mu_0_micro":         arr(qn["mu_0"]),
        "Sigma_0_micro":      arr(qn["Sigma_0"]),
        "sigma2_rl_micro":    float(qn["sigma2"]),
    }

    if qj is not None:
        # Joint (eta, beta) prior for the modified-TD-loss RLSVI.
        # Sigma_0 is the FULL joint covariance across users (not block-diag).
        out.update({
            "mu_0_micro_mtd_joint":       arr(qj["mu_0"]),
            "Sigma_0_micro_mtd_joint":    arr(qj["Sigma_0"]),
            "p_eta_micro_mtd_joint":      (None if qj.get("p_eta") is None
                                           else int(qj["p_eta"])),
            "sigma2_Q_mtd_joint":         float(qj.get(
                "sigma2_Q",
                qj.get("sigma2_TD", 1.0),
            )),
        })

    return out


def main() -> Dict[str, Any]:
    df_fit = load_df_fit()
    print(f"df_fit: {len(df_fit)} rows, "
          f"{df_fit['ParticipantIdentifier'].nunique()} participants")

    print("Fitting PF mediator / Y / tY priors ...")
    pf = fit_pf_priors(df_fit)
    for name, d in pf.items():
        n = None if d["nu_0"] is None else len(d["nu_0"])
        print(f"  {name:9s}: p={n}, sigma2={d['sigma2']:.4f}")

    print("Fitting reward-shaping eta prior ...")
    reward = fit_reward_prior(df_fit)
    print(f"  reward: p={len(reward['mu_0'])}, sigma2={reward['sigma2']:.4f}")

    print("Fitting Q-function prior (no TD-modify) ...")
    q_no_mod = fit_q_prior(df_fit)
    print(f"  Q (no TD-modify): p={len(q_no_mod['mu_0'])}, "
          f"sigma2={q_no_mod['sigma2']:.4f}")

    print("Fitting joint (alpha, beta) prior for modified-TD-loss RLSVI ...")
    q_mod_joint = fit_q_td_modify_joint_prior(df_fit)
    if q_mod_joint["mu_0"] is not None:
        print(f"  Q joint: p={len(q_mod_joint['mu_0'])}, "
              f"p_eta={q_mod_joint['p_eta']}, "
              f"n_users={q_mod_joint['n_users']}, "
              f"sigma2_Q={q_mod_joint['sigma2_Q']:.4f}")
    else:
        print("  Q joint: no users contributed (skipped)")

    priors = {
        "pf":           pf,
        "reward":       reward,
        "q_no_td_modify":      q_no_mod,
        "q_td_modify_joint":   q_mod_joint,
    }
    path = save_priors(priors)
    print(f"Saved priors -> {path}")
    return priors


if __name__ == "__main__":
    main()
