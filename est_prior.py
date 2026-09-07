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

Models estimated (current ``EXPERIMENT_ALGORITHMS`` roster)
----------------------------------------------------------
PF mediator m=0 (fourSC)        -> nu_0_MY[0], Gamma_0_MY[0], sigma2_MY[0]
PF mediator m=1 (antic)         -> nu_0_MY[1], Gamma_0_MY[1], sigma2_MY[1]
PF outcome  Y  (CAE)            -> nu_0_Y, Gamma_0_Y, sigma2_Y
PF outcome  tY (CAE_short)      -> nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y

RL reward shaping (legacy JSON) -> mu_0_reward, Sigma_0_reward, sigma2_reward
   Still written so old loaders do not break; no current arm uses it.

RL redistribution, Stage 1      -> reward_redistribution.daily_mediators
   AA/FW/PJ daily; slot SC/PV on ``build_foursc_stage1_phi`` (A×[1, t])
RL redistribution, Stage 2 V4   -> reward_redistribution.redistribution.v4
   Hats-only 20-d ψ (no next_my / next_me).
   y = Y + γ̄ E_{w+1} − E_w. Used by V6 and V6+leftover.

RL Q (no TD modify, weekly CAE)
   q_no_td_modify                 γ̄=0.5   -> rl_v7_base_g05
   q_no_td_modify_g09             γ̄=0.9   -> rl_v1_base_g09
   q_no_td_modify_g099            γ̄=0.99  -> rl_v8_base_g099

RL Q weekly potential (V5)      -> q_redistribution.v3
   Terminal reward = Y + F, F = γ̄ E_{w+1} − E_w
   (matches ``rl_v5_invariant_weekly``)

RL Q redistributed V4           -> q_redistribution.v4
   Slot rewards φᵀη, no leftover
   (matches ``rl_v6_invariant_redistributed``)
                                -> q_redistribution.v4_resid
   φᵀη plus Saturday leftover so the week sums to Y + F
   (matches ``rl_v6_invariant_redistributed_resid``)

RL Q joint (eta, beta)          -> mu_0_micro_mtd_joint, ...
   Modified-TD-loss RLSVI at γ̄=0.9. Used by rl_v2_mtd_g09.

The no-TD-modify Q prior is fit by fitted-Q iteration (FQI):

    target_{k,d,t} = gamma_{d,t} * max_a Q(s_{next}, a)            (non-terminal)
    target_{k,6,2} = R_{k+1} + gamma_{6,2} * max_a Q(s_{1,1}^{k+1}, a)

The modified-TD prior is fit by direct joint FQI on the stacked
bottleneck-state TD loss for ``theta=(eta,beta)``.

Output is written to ``env_para_vanilla/rl_priors.json``; see
``load_estimated_priors`` for reading it back into experiment.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

# Feature builders for the RL Q / reward / bottleneck regressions.
# We import directly from algorithm.py to guarantee the prior dimensions stay
# synchronised with the agent's runtime phi.
from algorithm_helpers import (
    N_RL_CONTEXT,
    PF_THETA_ANTIC_NAMES,
    PF_THETA_FOURSC_NAMES,
    _cumulative_discount,
    build_phi_action,
    build_phi_action_rewardshaping,
    build_daily_mediator_phi,
    build_foursc_stage1_phi,
    build_redistribution_phi,
    build_phi_bottleneck,
    build_rl_context_vector,
)
from vani_env import (
    PF_THETA_CAE_NAMES,
    assert_complete_week_slots,
    cae_mediator_ewma_rows,
    trim_pf_cae_prior,
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
        "/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/Xueqing",
    )
).expanduser().resolve()
# Priors are written into WORK_DIR. The design still comes from that
# folder's ``df_fit_11week.csv`` (RCT panel). Tuned STE folders only copy
# that CSV — they do not contain κ-scaled trajectories — so STE-folder
# priors stay RCT/vanilla-sized unless you replace the CSV.
WORK_DIR = Path(os.getenv(
    "ADAPR_EST_PRIOR_PARAMS_DIR", str(PROJECT_ROOT / "env_para_vanilla")
)).expanduser().resolve()
OUTPUT_PATH = WORK_DIR / "rl_priors.json"
REWARD_PRIOR_PATH = WORK_DIR / "reward_shaping_prior.json"
RL_Q_PRIOR_PATH = WORK_DIR / "rl_q_priors.json"
SUMMARY_TABLE_PATH = WORK_DIR / "rl_prior_tables.md"
PF_SUMMARY_PATH = WORK_DIR / "prior_summary_pf.csv"
RL_SUMMARY_PATH = WORK_DIR / "prior_summary_rl.csv"
REWARD_SUMMARY_PATH = WORK_DIR / "prior_summary_reward_shaping.csv"
RL_Q_SUMMARY_PATH = WORK_DIR / "prior_summary_rl_q.csv"
JOINT_SUMMARY_PATH = WORK_DIR / "prior_summary_joint.csv"
POOLED_PF_REGRESSION_PATH = WORK_DIR / "prior_summary_pooled_pf_regression.csv"
POOLED_PF_GEE_PATH = WORK_DIR / "prior_summary_pooled_pf_gee.csv"
POOLED_RL_Q_PATH = WORK_DIR / "prior_summary_pooled_rl_q.csv"
LOO_PRIOR_DIR = WORK_DIR / "loo_priors"
GEE_WORKING_CORR = "exchangeable"
MIN_FEATURE_STD = 1e-12

# Slot / day structure (mirrors experiment.py constants)
K_SLOTS_WEEK = 14                # 7 days × 2 slots in df_fit
SLOTS_PER_DAY = 2
DAYS_PER_WEEK_RL = 6             # RL handles Mon–Sat only
N_RL_SLOTS_WEEK = SLOTS_PER_DAY * DAYS_PER_WEEK_RL  # 12

# RL hyperparameters (must match experiment._gamma_dt_micro): discount 1 on
# non-terminal slots; weekly ``GAMMA_BAR`` only on the terminal slot.
# Default 0.9 matches V1--V6; V7's Q prior is fit with gamma_terminal=0.5.
GAMMA_BAR = 0.9
GAMMA_DT = np.ones((DAYS_PER_WEEK_RL, SLOTS_PER_DAY), dtype=float)
GAMMA_DT[DAYS_PER_WEEK_RL - 1, SLOTS_PER_DAY - 1] = GAMMA_BAR
GAMMA_TERMINAL = float(GAMMA_DT[DAYS_PER_WEEK_RL - 1, SLOTS_PER_DAY - 1])
DELTA = _cumulative_discount(GAMMA_DT)         # Delta_{d,t} from slot (0,0)
DELTA_TERMINAL = float(DELTA[DAYS_PER_WEEK_RL - 1, SLOTS_PER_DAY - 1])

# Mediator state shapes used by build_phi_action.
RL_MY_SHAPE = (6, 3)
RL_ME_SHAPE = (6, 4)

# Numerical hyperparameters
RIDGE_ALPHA_PF = 1.0     # ridge prior precision for PF mediator / Y / tY fits
RIDGE_ALPHA_RL = 1.0     # ridge prior precision for RL Q / reward / joint fits
MIN_SIGMA2 = 1e-6
# Prior variance for design columns that are identically zero offline
# (e.g. ``b_tilde``). Flooring their across-user θ-variance at MIN_SIGMA2
# yields N(0, 1e-6) — too tight for RLSVI to learn online. Intercept is
# constant but *not* unused (max |x|=1); only all-zero columns get this.
UNIDENTIFIED_PRIOR_VAR = 1.0
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


def _default_df_fit_path() -> Path:
    """Prefer WORK_DIR's 11-week RCT panel (also copied into STE folders).

    ``COMBINED_DIR/df_fit.csv`` is only a fallback (larger RCT extract).
    """
    candidates = [
        WORK_DIR / "df_fit_11week.csv",
        COMBINED_DIR / "df_fit.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find df_fit input. Set ADAPR_EST_PRIOR_PARAMS_DIR to a "
        "folder with df_fit_11week.csv, or set ADAPR_COMBINED_DIR. "
        f"Searched:\n  {searched}"
    )


def load_df_fit(path: Optional[Path] = None) -> pd.DataFrame:
    p = Path(path).expanduser().resolve() if path is not None else _default_df_fit_path()
    if not p.is_file():
        raise FileNotFoundError(f"df_fit not found: {p}")
    print(f"Loading df_fit from {p}", flush=True)
    df = _filter_users_with_cae_obs(_week_fix_and_filter(pd.read_csv(p)))
    assert_complete_week_slots(df)
    return df



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
    sigma2 = _ridge_residual_sigma2(resid, p)
    return theta, sigma2


def _ridge_residual_sigma2(resid: np.ndarray, n_features: int) -> float:
    """Residual variance ``RSS / (n - p)``, floored at ``MIN_SIGMA2``.

    Ridge residuals are not orthogonal to the columns and need not have
    mean zero, so the centered sample variance (``np.var(..., ddof=1)``)
    is the wrong estimator. When ``n <= p`` the residual df is not
    positive and this returns 1.0.
    """
    resid = np.asarray(resid, dtype=float).ravel()
    df = int(resid.size) - int(n_features)
    if df <= 0:
        return 1.0
    rss = float(np.dot(resid, resid))
    return max(rss / df, MIN_SIGMA2)


def _feature_column_std(X: np.ndarray, index: int) -> float:
    """Sample std of one design column (0 => constant, including intercept)."""
    return float(np.nanstd(np.asarray(X, dtype=float)[:, index]))


def _all_zero_design_columns(X: np.ndarray, atol: float = MIN_FEATURE_STD) -> np.ndarray:
    """True for columns that are structurally unused (identically ~0).

    Distinct from zero *standard deviation*: the intercept is constant but
    identified. ``b_tilde`` is all zeros because offline φ always passes 0.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.size == 0:
        return np.zeros(0, dtype=bool)
    peak = np.nanmax(np.abs(X), axis=0)
    return np.where(np.isfinite(peak), peak < float(atol), True)


def _inflate_unidentified_variance(
    var: np.ndarray, X_pooled: Optional[np.ndarray],
) -> np.ndarray:
    """Raise all-zero design-column variances to ``UNIDENTIFIED_PRIOR_VAR``."""
    var = np.asarray(var, dtype=float).ravel()
    if X_pooled is None:
        return var
    unused = _all_zero_design_columns(X_pooled)
    if unused.size != var.size:
        return var
    return np.where(unused, np.maximum(var, UNIDENTIFIED_PRIOR_VAR), var)


def _inflate_unidentified_cov(
    Sigma: np.ndarray, X_pooled: Optional[np.ndarray],
) -> np.ndarray:
    """Raise all-zero design-column *diagonals* to ``UNIDENTIFIED_PRIOR_VAR``."""
    Sigma = np.array(Sigma, dtype=float, copy=True)
    if X_pooled is None or Sigma.size == 0:
        return Sigma
    unused = _all_zero_design_columns(X_pooled)
    if unused.size != Sigma.shape[0]:
        return Sigma
    diag = np.diag(Sigma).copy()
    diag[unused] = np.maximum(diag[unused], UNIDENTIFIED_PRIOR_VAR)
    np.fill_diagonal(Sigma, diag)
    return Sigma


def _assert_named_coords_diffuse(
    Sigma_0: Optional[np.ndarray],
    names: List[str],
    required: Tuple[str, ...],
    *,
    min_var: float = UNIDENTIFIED_PRIOR_VAR,
) -> None:
    """Fail if named unidentified coordinates still have a dogmatic prior."""
    if Sigma_0 is None:
        return
    diag = np.diag(np.asarray(Sigma_0, dtype=float))
    lookup = {name: i for i, name in enumerate(names)}
    for name in required:
        if name not in lookup:
            continue
        i = lookup[name]
        if float(diag[i]) < float(min_var) - 1e-12:
            raise AssertionError(
                f"{name} prior variance {float(diag[i]):.3g} < {min_var}"
            )


def _stack_phi_obs(feats_list: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Stack FQI ``phi_obs`` rows used as the Q-prior design."""
    if not feats_list:
        return None
    p = int(feats_list[0]["p_phi"])
    return np.concatenate(
        [np.asarray(f["phi_obs"], dtype=float).reshape(-1, p) for f in feats_list],
        axis=0,
    )


def _unidentified_inference() -> Dict[str, Any]:
    """Placeholder significance flags for zero-variance features."""
    return {
        "significant_0.05": False,
        "significant_0.01": False,
    }


def _ridge_fit_summary(
    X: np.ndarray,
    y: np.ndarray,
    names: List[str],
    *,
    model: str,
    outcome: str,
    alpha: float = RIDGE_ALPHA_PF,
) -> pd.DataFrame:
    """Pooled ridge coefficients with approximate two-sided p-values.

    Standard errors use the ridge sandwich
    ``sigma2 * (X'X + alpha I)^{-1} X'X (X'X + alpha I)^{-1}``; p-values are
    Student-t with ``df = n_obs - n_features``. These are approximate because
    ridge shrinks coefficients.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xo, yo = X[ok], y[ok]
    n_obs, n_features = Xo.shape
    if n_obs == 0:
        return pd.DataFrame()

    with np.errstate(divide="ignore", invalid="ignore"):
        xtx = Xo.T @ Xo
        a_mat = xtx + alpha * np.eye(n_features)
        a_inv = np.linalg.inv(a_mat)
        theta = a_inv @ (Xo.T @ yo)
        resid = yo - Xo @ theta

    sigma2 = _ridge_residual_sigma2(resid, n_features)

    cov = sigma2 * (a_inv @ xtx @ a_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(se > 0, theta / se, np.nan)
    df = max(n_obs - n_features, 1)
    p_value = 2.0 * student_t.sf(np.abs(t_stat), df)

    rows = []
    for i, feature in enumerate(names):
        col_std = _feature_column_std(Xo, i)
        identified = col_std >= MIN_FEATURE_STD
        if identified:
            p_i = float(p_value[i])
            infer = {
                "std_error": float(se[i]),
                "t_stat": float(t_stat[i]),
                "p_value": p_i,
                "significant_0.05": bool(np.isfinite(p_i) and p_i < 0.05),
                "significant_0.01": bool(np.isfinite(p_i) and p_i < 0.01),
            }
        else:
            infer = {
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                **_unidentified_inference(),
            }
        rows.append({
            "model": model,
            "outcome": outcome,
            "index": i,
            "feature": feature,
            "coefficient": float(theta[i]),
            "identified": identified,
            "feature_std": col_std,
            **infer,
            "n_obs": int(n_obs),
            "n_features": int(n_features),
            "ridge_alpha": float(alpha),
            "residual_sigma2": float(sigma2),
        })
    return pd.DataFrame(rows)


def _pooled_pf_model_specs() -> Tuple[
    Tuple[str, Any, Tuple[str, ...], str], ...
]:
    return (
        ("fourSC", _build_fourSC_design, PF_THETA_FOURSC_NAMES, "4hour_step_norm"),
        ("antic", _build_antic_design, PF_THETA_ANTIC_NAMES, "anticipated_affect_norm"),
        ("CAE", _build_CAE_design, PF_THETA_CAE_NAMES, "CAE_avg_norm"),
    )


def _stack_pooled_pf_designs(df_fit: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Stack per-user PF designs for pooled fourSC / antic / CAE fits."""
    stacked: Dict[str, Dict[str, Any]] = {}
    for model, builder, names, outcome in _pooled_pf_model_specs():
        x_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        group_parts: List[np.ndarray] = []
        for uid, dat in df_fit.groupby("ParticipantIdentifier", sort=False):
            dat = dat.sort_values(["Date", "DecisionTime"]).reset_index(drop=True)
            x_part, y_part = builder(dat)
            x_parts.append(x_part)
            y_parts.append(y_part)
            group_parts.append(np.full(len(y_part), int(uid), dtype=int))
        stacked[model] = {
            "X": np.vstack(x_parts),
            "y": np.concatenate(y_parts),
            "groups": np.concatenate(group_parts),
            "names": list(names),
            "outcome": outcome,
        }
    return stacked


def build_pooled_pf_regression_summary(df_fit: pd.DataFrame) -> pd.DataFrame:
    """Summarize pooled PF ridge regressions for fourSC, antic, and weekly CAE."""
    frames: List[pd.DataFrame] = []
    for model, payload in _stack_pooled_pf_designs(df_fit).items():
        frames.append(
            _ridge_fit_summary(
                payload["X"],
                payload["y"],
                payload["names"],
                model=model,
                outcome=payload["outcome"],
                alpha=RIDGE_ALPHA_PF,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _gee_fit_summary(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    names: List[str],
    *,
    model: str,
    outcome: str,
    working_corr: str = GEE_WORKING_CORR,
    skip_indices: Optional[frozenset[int]] = None,
    block: Optional[str] = None,
    row_meta: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Pooled GEE coefficients with cluster-robust p-values by participant."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    xo, yo, go = X[ok], y[ok], groups[ok]
    n_obs, n_features = xo.shape
    if n_obs == 0:
        return pd.DataFrame()

    cov_struct = Exchangeable()
    gee = GEE(
        yo,
        xo,
        groups=go,
        family=Gaussian(),
        cov_struct=cov_struct,
    )
    result = gee.fit()

    meta = dict(row_meta or {})
    rows = []
    for i, feature in enumerate(names):
        if skip_indices and i in skip_indices:
            continue
        col_std = _feature_column_std(xo, i)
        identified = col_std >= MIN_FEATURE_STD
        if identified:
            p_i = float(result.pvalues[i])
            infer = {
                "robust_se": float(result.bse[i]),
                "z_stat": float(result.tvalues[i]),
                "p_value": p_i,
                "significant_0.05": bool(np.isfinite(p_i) and p_i < 0.05),
                "significant_0.01": bool(np.isfinite(p_i) and p_i < 0.01),
            }
        else:
            infer = {
                "robust_se": np.nan,
                "z_stat": np.nan,
                "p_value": np.nan,
                **_unidentified_inference(),
            }
        row = {
            "coefficient": float(result.params[i]),
            **infer,
            "model": model,
            "outcome": outcome,
            "index": i,
            "feature": feature,
            "identified": identified,
            "feature_std": col_std,
            "n_obs": int(n_obs),
            "n_clusters": int(len(np.unique(go))),
            "n_features": int(n_features),
            "working_correlation": working_corr,
            **meta,
        }
        if block is not None:
            row["block"] = block
        rows.append(row)
    return pd.DataFrame(rows)


def build_pooled_pf_gee_summary(df_fit: pd.DataFrame) -> pd.DataFrame:
    """Summarize pooled PF GEE fits for fourSC, antic, and weekly CAE."""
    frames: List[pd.DataFrame] = []
    for model, payload in _stack_pooled_pf_designs(df_fit).items():
        frames.append(
            _gee_fit_summary(
                payload["X"],
                payload["y"],
                payload["groups"],
                payload["names"],
                model=model,
                outcome=payload["outcome"],
                working_corr=GEE_WORKING_CORR,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
    X_pooled: Optional[np.ndarray] = None,
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
    X_pooled       optional stacked design. All-zero columns (structurally
                   unidentified) get prior variance ``UNIDENTIFIED_PRIOR_VAR``
                   instead of the ``MIN_SIGMA2`` floor.

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
    var = _inflate_unidentified_variance(var, X_pooled)
    cov = np.diag(var)
    sigma2 = _mean_user_sigma2(user_sigma2s)
    return pooled_theta, cov, sigma2


# ──────────────────────────────────────────────────────────────────
# 1. PF mediator / Y / tY priors
# ──────────────────────────────────────────────────────────────────
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

    Differs from ``gen_fourSC_mean`` in PF-specific ways:
      * Only the 12 RL-controlled slots per week (Mon–Sat × AM/PM) are kept;
        Sunday rows are dropped.
      * 7-day pageview EMA, yesterday anticipated affect, and 7-day Fitbit
        wear (and their action interactions) are omitted.
      * AR-1 lag of 4-hour step count is a main effect only.
      * Weekend has no action interaction; AM/PM enters as a main effect and
        as ``Ah*decisionTimeSlot`` (trailing column).

    Per-particle CAE substitution (``cae=0`` base + ``cae_j`` delta) is a
    runtime substitution mechanism and does **not** change the population
    parameter being estimated, so the regression here uses the actual
    ``caeAverageLastWeek_norm`` column.
    """
    assert_complete_week_slots(dat)
    int_ = np.ones(len(dat))
    lag1 = _fill_nan(dat["FourSC_lag1"].to_numpy())
    yest_step = _fill_nan(dat["YesterdayStepCount_norm"].to_numpy())
    seven_step = _fill_nan(dat["EMA_StepCount_norm"].to_numpy())
    prior2 = _fill_nan(dat["prior2hour_step_norm"].to_numpy())
    rb = _fill_nan(
        dat["recent_burden_norm" if "recent_burden_norm" in dat.columns else "recentBurdenEma_norm"].to_numpy()
    )
    i7w = _fill_nan(dat["Interacted_7d_walk"].to_numpy())
    act_frac7 = _fill_nan(dat["active_status_fraction_7days"].to_numpy())
    is_weekend = dat["is_weekend"].to_numpy()
    dt_ = dat["DecisionTime"].to_numpy().astype(float)
    pu = _fill_nan(dat["perceived_utility_lastweek"].to_numpy())
    cae = _fill_nan(dat["CAE_avg_lastweek_norm"].to_numpy())
    Ah = dat["WalkingSuggestion"].to_numpy().astype(float)

    X = np.column_stack([
        int_, lag1, yest_step, seven_step, prior2, rb,
        i7w, act_frac7, is_weekend, dt_, pu, cae,
        Ah, Ah * yest_step, Ah * prior2, Ah * rb,
        Ah * i7w, Ah * pu, Ah * cae,
        Ah * dt_,
    ])
    y = dat["4hour_step_norm"].to_numpy(dtype=float)
    assert X.shape[1] == len(PF_THETA_FOURSC_NAMES)

    # ── PF-specific shaping ───────────────────────────────────────────
    n_w = len(dat) // K_SLOTS_WEEK
    X = X.reshape(n_w, K_SLOTS_WEEK, X.shape[1])
    y = y.reshape(n_w, K_SLOTS_WEEK)
    # Drop Sunday slots (indices 12, 13) → keep 12 RL-controlled slots per week.
    X = X[:, : _PF_FOURSC_SLOTS_PER_WEEK, :]
    y = y[:, : _PF_FOURSC_SLOTS_PER_WEEK]
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(-1)
    return X, y


def _build_antic_design(dat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Antic PF design matrix and target (one row per day, morning row).

    Matches ``algorithm_helpers.build_antic_features``, with two PF-specific
    shapings:
      * Only 6 RL-controlled days per week (Mon–Sat) are kept; Sunday is
        dropped.
      * ``anticipated_affect_yesterday`` (the antic AR-1 column) is set to
        0 on the first day of each week (Monday), matching the PF's
        week-boundary lag clearing.

    Walking-suggestion interactions are with \(E_w\), last-week CAE, and
    recent burden only (no weekend or 7-day active-fraction action terms).

    Like for fourSC, the per-particle CAE substitution does not change
    the regression population parameter, so actual ``caeAverageLastWeek_norm``
    values are used here.
    """
    assert_complete_week_slots(dat)
    am = dat["DecisionTime"].to_numpy() == 0
    dat_am = dat.loc[am].reset_index(drop=True)
    n = len(dat_am)
    int_ = np.ones(n)
    antic_yest = _fill_nan(dat_am["anticipated_affect_yesterday_norm"].to_numpy())
    act = _fill_nan(dat_am["active_status_fraction_7days"].to_numpy())
    is_weekend = dat_am["is_weekend"].to_numpy()
    pu = _fill_nan(dat_am["perceived_utility_lastweek"].to_numpy())
    cae = _fill_nan(dat_am["CAE_avg_lastweek_norm"].to_numpy())
    rb = _fill_nan(
        dat_am[
            "recent_burden_norm"
            if "recent_burden_norm" in dat_am.columns
            else "recentBurdenEma_norm"
        ].to_numpy()
    )

    # Per-day WS pair (rows of dat alternate AM, PM for each Date).
    WS_all = dat["WalkingSuggestion"].to_numpy().astype(float)
    pair = WS_all.reshape(-1, 2)
    ws_m = pair[:n, 0]
    ws_a = pair[:n, 1]

    X = np.column_stack([
        int_, antic_yest, act, is_weekend, pu, cae, rb,
        ws_m, ws_a,
        ws_m * pu, ws_a * pu,
        ws_m * cae, ws_a * cae,
        ws_m * rb, ws_a * rb,
    ])
    y = dat_am["anticipated_affect_norm"].to_numpy(dtype=float)
    assert X.shape[1] == len(PF_THETA_ANTIC_NAMES)

    # ── PF-specific shaping ───────────────────────────────────────────
    days_per_week = 7
    n_w = n // days_per_week
    X = X.reshape(n_w, days_per_week, X.shape[1])
    y = y.reshape(n_w, days_per_week)
    # Drop Sunday (day index 6) → keep 6 RL-controlled days per week.
    X = X[:, : _PF_ANTIC_DAYS_PER_WEEK, :]
    y = y[:, : _PF_ANTIC_DAYS_PER_WEEK]
    # Clear lag at the first day of every week (Monday).
    X[:, 0, _PF_ANTIC_LAG1_COL] = 0.0
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(-1)
    return X, y


def _build_CAE_design(dat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Weekly CAE design: [1, CAE_lw, fourSC_ewma, antic_ewma].

    EWMA summaries use RL-controlled Mon–Sat only (12 slots / 6 days).
    Missing slots are filled with that week's finite mean
    (:func:`vani_env.cae_mediator_ewmas`), matching the runtime PF helper.
    """
    assert_complete_week_slots(dat)
    n_w = len(dat) // K_SLOTS_WEEK
    cae_y = dat["CAE_avg_norm"].to_numpy().reshape(-1, K_SLOTS_WEEK)[:, 0]
    cae_lw = _fill_nan(dat["CAE_avg_lastweek_norm"].to_numpy()).reshape(-1, K_SLOTS_WEEK)[:, 0]

    foursc = dat["4hour_step_norm"].to_numpy().reshape(-1, K_SLOTS_WEEK)[
        :, :N_RL_SLOTS_WEEK
    ]
    antic = dat["anticipated_affect_norm"].to_numpy().reshape(-1, K_SLOTS_WEEK)[
        :, :N_RL_SLOTS_WEEK
    ]
    foursc_e, antic_e = cae_mediator_ewma_rows(foursc, antic)

    X = np.column_stack([
        np.ones(n_w), cae_lw, foursc_e, antic_e,
    ])
    assert X.shape[1] == len(PF_THETA_CAE_NAMES)
    return X, cae_y


def _build_CAE_short_design(dat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    assert_complete_week_slots(dat)
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

    nu_fSC, G_fSC, s2_fSC = _pool_user_fits(
        th_fSC_pool, fSC_thetas, fSC_sigma2s, X_pooled=np.vstack(X_fSC_all))
    nu_ant, G_ant, s2_ant = _pool_user_fits(
        th_ant_pool, ant_thetas, ant_sigma2s, X_pooled=np.vstack(X_ant_all))
    nu_Y,   G_Y,   s2_Y   = _pool_user_fits(
        th_Y_pool,   Y_thetas,   Y_sigma2s,   X_pooled=np.vstack(X_Y_all))
    nu_tY,  G_tY,  s2_tY  = _pool_user_fits(
        th_tY_pool,  tY_thetas,  tY_sigma2s,  X_pooled=np.vstack(X_tY_all))

    return {
        "fourSC":    {"nu_0": nu_fSC, "Gamma_0": G_fSC, "sigma2": s2_fSC,
                       "names": PF_THETA_FOURSC_NAMES},
        "antic":     {"nu_0": nu_ant, "Gamma_0": G_ant, "sigma2": s2_ant,
                       "names": PF_THETA_ANTIC_NAMES},
        "CAE":       {"nu_0": nu_Y,   "Gamma_0": G_Y,   "sigma2": s2_Y,
                       "names": PF_THETA_CAE_NAMES},
        "CAE_short": {"nu_0": nu_tY,  "Gamma_0": G_tY,  "sigma2": s2_tY,
                       "names": THETA_CAE_SHORT_NAMES},
    }


# ──────────────────────────────────────────────────────────────────
# 2. Per-user phi_action / phi_rs / phi_bottleneck row builders
# ──────────────────────────────────────────────────────────────────
def _build_rl_context_vector_from_row(row) -> np.ndarray:
    """Match ``experiment.build_rl_context_vector`` (length ``N_RL_CONTEXT``).
    """
    return build_rl_context_vector(
        yesterdayStepCount=float(row["YesterdayStepCount_norm"]),
        prior2HourStepCount=float(row["prior2hour_step_norm"]),
        activeDaysLast7Days=float(row["active_status_fraction_7days"]),
        activitySuggestionsSentLast7Days=float(
            row["recent_burden_norm" if "recent_burden_norm" in row.index else "recentBurdenEma_norm"]
        ),
        activitySuggestionInteractLast7Days=float(row["Interacted_7d_walk"]),
    )


def _user_weekly_tensors(dat: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Pre-build all per-week / per-slot quantities needed for RL phi builders.

    ``dat`` must be one participant, sorted by (Date, DecisionTime), with
    a complete 14-slot panel every week (Monday .. Sunday).  Rows are
    NaN-filled with the participant's mean so the resulting tensors are
    finite.
    """
    assert_complete_week_slots(dat)
    n_w = len(dat) // K_SLOTS_WEEK
    dat = dat.reset_index(drop=True)

    cols = [
        "YesterdayStepCount_norm", "EMA_StepCount_norm", "prior2hour_step_norm",
        "recent_burden_norm", "active_status_fraction_7days",
        "Interacted_7d_walk",
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
    p_C = N_RL_CONTEXT

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
    d = rl_idx // 2
    t = rl_idx %  2
    return build_phi_action(
        b_hat=float(t_dict["b_hat_slot"][k, rl_idx]),
        b_tilde=0.0,
        state=_slot_state(t_dict, k, rl_idx),
        d=d, t=t,
        action=int(action),
    )


def _phi_rs(t_dict, k, rl_idx) -> np.ndarray:
    d = rl_idx // 2
    t = rl_idx %  2
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
        target   = Delta_{6,2} * Y_w + GAMMA_BAR * E_{w+1}

    where ``Y_w`` is the weekly CAE (``caeAverage_norm``) and ``E_{w+1}`` is
    next week's start-of-week engagement. The last week is dropped (same as
    V4 and FQI in this file) because it has no successor ``E_{w+1}``.
    Shaped intermediate rewards and the terminal compensation ``R_{w,add}``
    are *not* included here — those enter only the Q-function FQI targets
    at runtime.

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
        n = n_w - 1  # last week has no E_{w+1}
        Phi = np.stack([_phi_rs_week_discounted(td, k) for k in range(n)])
        y = (
            DELTA_TERMINAL * td["R_week"][:n]
            + GAMMA_BAR * td["E_w_start"][1 : n + 1]
        )
        Phi_all.append(Phi); y_all.append(y)
        theta, s2 = _ridge_fit(Phi, y, alpha=RIDGE_ALPHA_RL)
        thetas.append(theta)
        sigma2s.append(s2)

    if Phi_all:
        pooled_theta, _ = _ridge_fit(
            np.vstack(Phi_all), np.concatenate(y_all), alpha=RIDGE_ALPHA_RL)
    else:
        pooled_theta = None

    X_pooled = np.vstack(Phi_all) if Phi_all else None
    mu, Sigma, sigma2 = _pool_user_fits(
        pooled_theta, thetas, sigma2s, X_pooled=X_pooled)
    _assert_named_coords_diffuse(Sigma, _phi_state_names(), ("b_tilde",))
    return {"mu_0": mu, "Sigma_0": Sigma, "sigma2": sigma2}


# ──────────────────────────────────────────────────────────────────
# 3b. Priors for V4 two-stage reward redistribution
# ──────────────────────────────────────────────────────────────────
_DAILY_MEDIATORS = {"AA": ("M_Y_week", 2), "FW": ("M_E_week", 2), "PJ": ("M_E_week", 3)}


def _phi_daily_mediator(td: Dict[str, np.ndarray], k: int, rl_idx: int,
                        mediator: str) -> np.ndarray:
    d, t = divmod(rl_idx, SLOTS_PER_DAY)
    return build_daily_mediator_phi(
        float(td["b_hat_slot"][k, rl_idx]), 0.0, _slot_state(td, k, rl_idx),
        d, t, int(td["A_slot"][k, rl_idx]), mediator=mediator,
    )


def _daily_mediator_design(td: Dict[str, np.ndarray], mediator: str):
    """Daily sum-of-two-slots design used by the online Stage-1 fit."""
    matrix_name, col = _DAILY_MEDIATORS[mediator]
    rows, targets = [], []
    for k in range(td["n_w"]):
        for d in range(DAYS_PER_WEEK_RL):
            i0 = d * SLOTS_PER_DAY
            rows.append(_phi_daily_mediator(td, k, i0, mediator) +
                        _phi_daily_mediator(td, k, i0 + 1, mediator))
            targets.append(float(td[matrix_name][k, d, col]))
    p = _phi_daily_mediator(td, 0, 0, mediator).size
    return (np.asarray(rows, dtype=float) if rows else np.empty((0, p)),
            np.asarray(targets, dtype=float))


def _daily_eta_for_tensor(td: Dict[str, np.ndarray], mediator: str) -> np.ndarray:
    X, y = _daily_mediator_design(td, mediator)
    return _ridge_fit(X, y, alpha=RIDGE_ALPHA_RL)[0]


def _phi_foursc_stage1(td: Dict[str, np.ndarray], k: int, rl_idx: int) -> np.ndarray:
    d, t = divmod(rl_idx, SLOTS_PER_DAY)
    return build_foursc_stage1_phi(
        float(td["b_hat_slot"][k, rl_idx]), 0.0, _slot_state(td, k, rl_idx),
        d, t, int(td["A_slot"][k, rl_idx]),
    )


# Slot-level Stage-1b outcomes: fourSC from M_Y_week[k, d, t], normalised
# 4-hour page views (HourlyPageviewCount_norm) from M_E_week[k, d, t]. Both
# share ``build_foursc_stage1_phi`` (controls + A×[1, slot_pm]).
_SLOT_MEDIATORS = {"SC": "M_Y_week", "PV": "M_E_week"}


def _foursc_stage1_design(td: Dict[str, np.ndarray], mediator: str = "SC"):
    """Slot-level design for ``mediator`` in ``_SLOT_MEDIATORS``: 12 rows/week."""
    matrix = td[_SLOT_MEDIATORS[mediator]]
    rows, targets = [], []
    for k in range(td["n_w"]):
        for d in range(DAYS_PER_WEEK_RL):
            for t in range(SLOTS_PER_DAY):
                rl_idx = d * SLOTS_PER_DAY + t
                yt = float(matrix[k, d, t])
                if not np.isfinite(yt):
                    continue
                rows.append(_phi_foursc_stage1(td, k, rl_idx))
                targets.append(yt)
    p = _phi_foursc_stage1(td, 0, 0).size
    return (np.asarray(rows, dtype=float) if rows else np.empty((0, p)),
            np.asarray(targets, dtype=float))


def _sc_eta_for_tensor(td: Dict[str, np.ndarray], mediator: str = "SC") -> np.ndarray:
    X, y = _foursc_stage1_design(td, mediator)
    return _ridge_fit(X, y, alpha=RIDGE_ALPHA_RL)[0]


def _stage1_shares(td: Dict[str, np.ndarray], k: int, rl_idx: int,
                   daily_etas: Dict[str, np.ndarray]) -> np.ndarray:
    """[AA, FW, PJ, SC, PV] shares — same order as runtime ``daily_mediator_shares``."""
    shares = [
        _phi_daily_mediator(td, k, rl_idx, name) @ daily_etas[name]
        for name in ("AA", "FW", "PJ")
    ]
    for name in ("SC", "PV"):
        if name in daily_etas:
            shares.append(_phi_foursc_stage1(td, k, rl_idx) @ daily_etas[name])
    return np.array(shares, dtype=float)


def _redistribution_week_phi(td: Dict[str, np.ndarray], k: int,
                             daily_etas: Dict[str, np.ndarray]) -> np.ndarray:
    """Un-discounted weekly Stage-2 row, matching V4 runtime code."""
    full = (td["M_Y_week"][k], td["M_E_week"][k])
    out = None
    for rl_idx in range(N_RL_SLOTS_WEEK):
        d, t = divmod(rl_idx, SLOTS_PER_DAY)
        state = _slot_state(td, k, rl_idx)
        action = int(td["A_slot"][k, rl_idx])
        phi = build_redistribution_phi(
            float(td["b_hat_slot"][k, rl_idx]), 0.0, state, d, t, action,
            _stage1_shares(td, k, rl_idx, daily_etas), full_mediators=full,
        )
        out = phi.copy() if out is None else out + phi
    return out


def _v4_week_targets(td: Dict[str, np.ndarray], *, gamma_bar: float) -> np.ndarray:
    """V4 / V5 weekly scalar ``Y_w + γ̄ E_{w+1} − E_w`` (needs a successor week)."""
    n = td["n_w"] - 1
    return (
        td["R_week"][:n]
        + float(gamma_bar) * td["E_w_start"][1:n + 1]
        - td["E_w_start"][:n]
    )


def fit_reward_redistribution_priors(
    df_fit: pd.DataFrame, *, gamma_bar: float = GAMMA_BAR,
) -> Dict[str, Any]:
    """Estimate empirical-Bayes priors used by redistributed V4.

    Stage 1 fits separate AA/FW/PJ daily regressions with the *daily sum* of
    the two action-time features, including ``A×slot_pm`` so AM and PM
    sends can have different intercepts. Stage 1b fits slot-level fourSC
    and PV on controls + ``A·(β0 + β_t t)`` (12 rows/week). Stage 2 then
    uses the predicted shares (AA/FW/PJ/SC/PV) in its weekly summed
    hats-only ψ. Pooled Stage-2 rows use pooled Stage-1 coefficients;
    per-user Stage-2 rows use that user's Stage-1 coefficients.
    """
    tensors = [
        _user_weekly_tensors(dat.sort_values(["Date", "DecisionTime"]).reset_index(drop=True))
        for _uid, dat in df_fit.groupby("ParticipantIdentifier", sort=False)
    ]
    tensors = [td for td in tensors if td["n_w"] >= 2]
    if not tensors:
        raise ValueError(
            "Need at least one participant with two complete weeks to fit "
            "V4 reward-redistribution priors."
        )
    daily = {}
    pooled_daily_eta = {}
    user_daily_eta: list[Dict[str, np.ndarray]] = []
    for mediator in _DAILY_MEDIATORS:
        user_fits, user_s2, X_all, y_all = [], [], [], []
        for td in tensors:
            X, y = _daily_mediator_design(td, mediator)
            theta, s2 = _ridge_fit(X, y, alpha=RIDGE_ALPHA_RL)
            user_fits.append(theta); user_s2.append(s2); X_all.append(X); y_all.append(y)
        pooled, _ = _ridge_fit(np.vstack(X_all), np.concatenate(y_all), alpha=RIDGE_ALPHA_RL)
        mu, Sigma, sigma2 = _pool_user_fits(
            pooled, user_fits, user_s2, X_pooled=np.vstack(X_all))
        daily[mediator] = {"mu_0": mu, "Sigma_0": Sigma, "sigma2": sigma2}
        pooled_daily_eta[mediator] = pooled
    for slot_med in _SLOT_MEDIATORS:
        user_fits, user_s2, X_all, y_all = [], [], [], []
        for td in tensors:
            X, y = _foursc_stage1_design(td, slot_med)
            theta, s2 = _ridge_fit(X, y, alpha=RIDGE_ALPHA_RL)
            user_fits.append(theta); user_s2.append(s2); X_all.append(X); y_all.append(y)
        pooled, _ = _ridge_fit(np.vstack(X_all), np.concatenate(y_all), alpha=RIDGE_ALPHA_RL)
        mu, Sigma, sigma2 = _pool_user_fits(
            pooled, user_fits, user_s2, X_pooled=np.vstack(X_all))
        daily[slot_med] = {"mu_0": mu, "Sigma_0": Sigma, "sigma2": sigma2}
        pooled_daily_eta[slot_med] = pooled
    for td in tensors:
        eta = {name: _daily_eta_for_tensor(td, name) for name in _DAILY_MEDIATORS}
        for slot_med in _SLOT_MEDIATORS:
            eta[slot_med] = _sc_eta_for_tensor(td, slot_med)
        user_daily_eta.append(eta)

    user_fits, user_s2, X_all, y_all = [], [], [], []
    for td, eta_user in zip(tensors, user_daily_eta):
        n = td["n_w"] - 1
        X_user = np.stack([_redistribution_week_phi(td, k, eta_user) for k in range(n)])
        y_user = _v4_week_targets(td, gamma_bar=gamma_bar)
        theta, s2 = _ridge_fit(X_user, y_user, alpha=RIDGE_ALPHA_RL)
        user_fits.append(theta); user_s2.append(s2)
        X_all.append(np.stack([_redistribution_week_phi(td, k, pooled_daily_eta)
                                for k in range(n)]))
        y_all.append(y_user)
    pooled, _ = _ridge_fit(np.vstack(X_all), np.concatenate(y_all), alpha=RIDGE_ALPHA_RL)
    mu, Sigma, sigma2 = _pool_user_fits(
        pooled, user_fits, user_s2, X_pooled=np.vstack(X_all))
    stage2 = {
        "v4": {
            "mu_0": mu, "Sigma_0": Sigma, "sigma2": sigma2,
            "psi_has_next_my": False,
            "psi_has_next_me": False,
            "psi_has_pv_hat": True,
        }
    }
    return {"daily_mediators": daily, "redistribution": stage2}


def _fit_redistribution_coefficients(
    td: Dict[str, np.ndarray], *, gamma_bar: float = GAMMA_BAR,
):
    """Plug-in Stage-1/2 fits used to construct the V4 FQI target."""
    daily = {name: _daily_eta_for_tensor(td, name) for name in _DAILY_MEDIATORS}
    for slot_med in _SLOT_MEDIATORS:
        daily[slot_med] = _sc_eta_for_tensor(td, slot_med)
    n = td["n_w"] - 1
    X = np.stack([_redistribution_week_phi(td, k, daily) for k in range(n)])
    y = _v4_week_targets(td, gamma_bar=gamma_bar)
    eta, _ = _ridge_fit(X, y, alpha=RIDGE_ALPHA_RL)
    return daily, eta, y


def _attach_redistributed_rewards(feats: Dict[str, Any], td: Dict[str, np.ndarray],
                                  daily: Dict[str, np.ndarray], eta: np.ndarray,
                                  week_targets: np.ndarray,
                                  *, add_terminal_residual: bool = False) -> Dict[str, Any]:
    """Attach per-slot V4 rewards ``φ^⊤ η``.

    With ``add_terminal_residual`` the leftover
    ``(Y + F) − Σ φᵀη`` is added on Saturday afternoon, matching
    ``rl_v6_invariant_redistributed_resid``.
    """
    out = dict(feats)
    rewards = np.empty((feats["n_train"], N_RL_SLOTS_WEEK), dtype=float)
    y = np.asarray(week_targets, dtype=float)
    for k in range(feats["n_train"]):
        phi_slots = np.stack([_redistribution_week_phi_slot(td, k, i, daily)
                              for i in range(N_RL_SLOTS_WEEK)])
        rewards[k] = phi_slots @ eta
        if add_terminal_residual:
            rewards[k, -1] += float(y[k]) - float(np.sum(rewards[k]))
    out["rewards"] = rewards
    return out


def _attach_week_terminal_rewards(feats: Dict[str, Any],
                                  week_targets: np.ndarray) -> Dict[str, Any]:
    """V5: zero within-week rewards; last slot gets ``Y + F``."""
    out = dict(feats)
    n = int(feats["n_train"])
    rewards = np.zeros((n, N_RL_SLOTS_WEEK), dtype=float)
    rewards[:, -1] = np.asarray(week_targets, dtype=float)[:n]
    out["rewards"] = rewards
    return out


def _redistribution_week_phi_slot(td: Dict[str, np.ndarray], k: int, rl_idx: int,
                                  daily_etas: Dict[str, np.ndarray]) -> np.ndarray:
    d, t = divmod(rl_idx, SLOTS_PER_DAY)
    state = _slot_state(td, k, rl_idx)
    return build_redistribution_phi(
        float(td["b_hat_slot"][k, rl_idx]), 0.0, state, d, t,
        int(td["A_slot"][k, rl_idx]),
        _stage1_shares(td, k, rl_idx, daily_etas),
        full_mediators=(td["M_Y_week"][k], td["M_E_week"][k]),
    )


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
    *,
    user_ids: Optional[List[int]] = None,
    return_design: bool = False,
    gamma_terminal: float = GAMMA_TERMINAL,
) -> Tuple[
    Optional[np.ndarray],
    Optional[float],
    Optional[float],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Run FQI on one or more users' precomputed features.

    With one entry in ``feats_list`` this is per-user FQI; with many it is a
    single FQI on all users' rows concatenated together. Each FQI iteration
    rebuilds targets per (user, week, slot) using the current ``theta`` and
    refits a ridge regression on the stacked rows.

    Returns ``(theta, sigma2, None)`` by default. When ``return_design=True``,
    also returns the final fitted-Q design ``(X, y, groups)``.
    """
    empty_extra = (None, None, None) if return_design else ()
    if not feats_list:
        return (None, None, None, *empty_extra)

    p_phi = feats_list[0]["p_phi"]
    x_parts = [f["phi_obs"].reshape(-1, p_phi) for f in feats_list]
    X_all = np.concatenate(x_parts)
    groups = None
    if return_design:
        if user_ids is None or len(user_ids) != len(feats_list):
            raise ValueError(
                "return_design=True requires user_ids aligned with feats_list"
            )
        group_parts = [
            np.full(x_part.shape[0], int(uid), dtype=int)
            for x_part, uid in zip(x_parts, user_ids)
        ]
        groups = np.concatenate(group_parts)

    theta = np.zeros(p_phi)
    y_all = np.zeros(X_all.shape[0])
    for _ in range(n_iters):
        chunks = []
        for f in feats_list:
            n_train = f["n_train"]
            targets = np.zeros((n_train, N_RL_SLOTS_WEEK))
            for k in range(n_train):
                for idx in range(N_RL_SLOTS_WEEK):
                    reward = (float(f["rewards"][k, idx])
                              if "rewards" in f else 0.0)
                    if idx < N_RL_SLOTS_WEEK - 1:
                        q0 = f["phi_a0"][k, idx + 1] @ theta
                        q1 = f["phi_a1"][k, idx + 1] @ theta
                        # Non-terminal within-week discount is 1.
                        targets[k, idx] = reward + float(max(q0, q1))
                    else:
                        if use_td_modify and alpha_for_boot is not None:
                            boot = float(f["phi_next_bot"][k] @ alpha_for_boot)
                        else:
                            q0 = f["phi_next_a0"][k] @ theta
                            q1 = f["phi_next_a1"][k] @ theta
                            boot = float(max(q0, q1))
                        if "rewards" not in f:
                            reward = float(f["R"][k])
                        targets[k, idx] = reward + gamma_terminal * boot
            chunks.append(targets.reshape(-1))
        y_all = np.concatenate(chunks)
        theta_new, _ = _ridge_fit(X_all, y_all, alpha=RIDGE_ALPHA_RL)
        if theta_new is None:
            return (None, None, None, *empty_extra)
        if np.linalg.norm(theta_new - theta) < 1e-6 * (np.linalg.norm(theta) + 1e-12):
            theta = theta_new
            break
        theta = theta_new

    resid = y_all - X_all @ theta
    sigma2 = _ridge_residual_sigma2(resid, X_all.shape[1])
    if return_design:
        return theta, sigma2, None, X_all, y_all, groups
    return theta, sigma2, None


def _mtd_joint_design(
    feats_list: List[Dict[str, Any]],
    beta_for_targets: np.ndarray,
    *,
    gamma_terminal: float = GAMMA_TERMINAL,
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
                # Non-terminal within-week discount is 1.
                y_B[row] = float(max(q0_next, q1_next))
                row += 1

        # Block C: terminal row with next-week bottleneck bootstrap in design.
        X_C = np.zeros((n_train, p))
        X_C[:, :p_eta] = -float(gamma_terminal) * feats["phi_next_bot"]
        X_C[:, p_eta:] = feats["phi_obs"][:, N_RL_SLOTS_WEEK - 1, :]
        y_C = feats["R"]

        X_chunks.extend([X_A, X_B, X_C])
        y_chunks.extend([y_A, y_B, y_C])

    return np.vstack(X_chunks), np.concatenate(y_chunks), p_eta


def _joint_fqi_iterate(
    feats_list: List[Dict[str, Any]],
    n_iters: int = N_FQI_ITERS,
    *,
    gamma_terminal: float = GAMMA_TERMINAL,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int]]:
    """Directly fit theta=(alpha,beta) under the bottleneck-state TD loss."""
    if not feats_list:
        return None, None, None

    p_eta = int(feats_list[0]["p_eta"])
    p_beta = int(feats_list[0]["p_phi"])
    theta = np.zeros(p_eta + p_beta)
    sigma2 = None

    for _ in range(n_iters):
        X, y, p_eta = _mtd_joint_design(
            feats_list, theta[p_eta:], gamma_terminal=gamma_terminal)
        theta_new, sigma2 = _ridge_fit(X, y, alpha=RIDGE_ALPHA_RL)
        if theta_new is None:
            return None, None, None
        if np.linalg.norm(theta_new - theta) < 1e-6 * (
            np.linalg.norm(theta) + 1e-12
        ):
            theta = theta_new
            break
        theta = theta_new

    X, y, _ = _mtd_joint_design(
        feats_list, theta[p_eta:], gamma_terminal=gamma_terminal)
    resid = y - X @ theta
    sigma2 = _ridge_residual_sigma2(resid, X.shape[1])
    return theta, sigma2, p_eta


def fit_q_prior(df_fit: pd.DataFrame, *, gamma_terminal: float = GAMMA_TERMINAL) -> Dict[str, Any]:
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
            th, s2_u, _ = _fqi_iterate(
                [feats_u_no], None, use_td_modify=False, gamma_terminal=gamma_terminal)
            th_nomod_users.append(th)
            s2_nomod_users.append(s2_u)
            feats_nomod_pool.append(feats_u_no)

    th_nomod_pool, _, _ = _fqi_iterate(
        feats_nomod_pool, None, use_td_modify=False, gamma_terminal=gamma_terminal)
    mu, Sigma, sigma2 = _pool_user_fits(
        th_nomod_pool, th_nomod_users, s2_nomod_users,
        X_pooled=_stack_phi_obs(feats_nomod_pool))
    _assert_named_coords_diffuse(
        Sigma, _phi_action_names(), ("b_tilde", "A*b_tilde"))
    return {"mu_0": mu, "Sigma_0": Sigma, "sigma2": sigma2}


def _fqi_from_shaped_features(
    df_fit: pd.DataFrame,
    shape_fn,
    *,
    gamma_terminal: float,
) -> Dict[str, Any]:
    """Pooled-mean / per-user-spread FQI on caller-shaped slot rewards."""
    user_theta, user_sigma2, pooled_feats = [], [], []
    for _uid, dat in df_fit.groupby("ParticipantIdentifier", sort=False):
        td = _user_weekly_tensors(
            dat.sort_values(["Date", "DecisionTime"]).reset_index(drop=True))
        feats = _precompute_q_features(td, use_td_modify=False)
        if feats is None:
            user_theta.append(None)
            user_sigma2.append(None)
            continue
        shaped = shape_fn(feats, td)
        theta, sigma2, _ = _fqi_iterate(
            [shaped], None, use_td_modify=False, gamma_terminal=gamma_terminal)
        user_theta.append(theta)
        user_sigma2.append(sigma2)
        pooled_feats.append(shaped)
    pooled, _, _ = _fqi_iterate(
        pooled_feats, None, use_td_modify=False, gamma_terminal=gamma_terminal)
    mu, Sigma, sigma2 = _pool_user_fits(
        pooled, user_theta, user_sigma2, X_pooled=_stack_phi_obs(pooled_feats))
    _assert_named_coords_diffuse(
        Sigma, _phi_action_names(), ("b_tilde", "A*b_tilde"))
    return {"mu_0": mu, "Sigma_0": Sigma, "sigma2": sigma2}


def fit_q_potential_prior(
    df_fit: pd.DataFrame, *, gamma_terminal: float = 0.9,
) -> Dict[str, Any]:
    """FQI prior for V5: terminal reward ``Y + F``, other slots 0."""

    def _shape(feats, td):
        return _attach_week_terminal_rewards(
            feats, _v4_week_targets(td, gamma_bar=gamma_terminal))

    return _fqi_from_shaped_features(df_fit, _shape, gamma_terminal=gamma_terminal)


def fit_q_redistribution_prior(
    df_fit: pd.DataFrame, *, gamma_terminal: float = GAMMA_TERMINAL,
    add_terminal_residual: bool = False,
) -> Dict[str, Any]:
    """FQI prior for V4 using its redistributed TD rewards.

    ``add_terminal_residual=False`` matches V6 (φᵀη only).
    ``True`` matches V6+leftover (week sums to ``Y + F``).
    """

    def _shape(feats, td):
        daily, eta, targets = _fit_redistribution_coefficients(
            td, gamma_bar=gamma_terminal)
        return _attach_redistributed_rewards(
            feats, td, daily, eta, targets,
            add_terminal_residual=add_terminal_residual,
        )

    return _fqi_from_shaped_features(df_fit, _shape, gamma_terminal=gamma_terminal)


def _collect_q_features_pool(
    df_fit: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Precompute no-TD-modify Q features for all users with >= 2 weeks."""
    feats_list: List[Dict[str, Any]] = []
    user_ids: List[int] = []
    for uid, dat in df_fit.groupby("ParticipantIdentifier", sort=False):
        dat = dat.sort_values(["Date", "DecisionTime"]).reset_index(drop=True)
        td = _user_weekly_tensors(dat)
        feats = _precompute_q_features(td, use_td_modify=False)
        if feats is not None:
            feats_list.append(feats)
            user_ids.append(int(uid))
    return feats_list, user_ids


def build_pooled_rl_q_summary(df_fit: pd.DataFrame) -> pd.DataFrame:
    """Summarize pooled no-TD-modify RL Q via GEE on the final FQI design.

    Fits a Gaussian GEE on the final fitted-Q regression rows
    (``phi_obs`` vs bootstrap targets), clustered by participant.
    Features with zero variance in the design (e.g. unused ``b_tilde`` or
    masked mediators) are flagged with ``identified=False`` and omitted
    SE / p-values.
    """
    feats_list, user_ids = _collect_q_features_pool(df_fit)
    if not feats_list:
        return pd.DataFrame()

    _theta, sigma2, _, x_all, y_all, groups = _fqi_iterate(
        feats_list,
        None,
        use_td_modify=False,
        user_ids=user_ids,
        return_design=True,
        gamma_terminal=0.5,
    )
    if x_all is None or y_all is None or groups is None:
        return pd.DataFrame()

    return _gee_fit_summary(
        x_all,
        y_all,
        groups,
        _phi_action_names(),
        model="q_no_td_modify",
        outcome="fqi_target",
        working_corr=GEE_WORKING_CORR,
        block="beta",
        row_meta={
            "fqi_iters": N_FQI_ITERS,
            "ridge_alpha": RIDGE_ALPHA_RL,
            "residual_sigma2": float(sigma2) if sigma2 is not None else np.nan,
        },
    )


# ──────────────────────────────────────────────────────────────────
# 5b. Joint (eta, beta) prior for the modified-TD-loss RLSVI
# ──────────────────────────────────────────────────────────────────
def _pool_joint_user_fits(
    pooled_theta: Optional[np.ndarray],
    user_thetas: List[Optional[np.ndarray]],
    X_pooled: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    """Joint-prior analogue of ``_pool_user_fits``.

    The mean ``mu`` is taken from ``pooled_theta`` (the joint coefficient
    vector obtained from pooled fits on the full multi-user dataset).
    Unlike ``_pool_user_fits``, ``Sigma`` is the **full** sample covariance
    across users of the per-user joint theta vectors, not a diagonal of
    per-coordinate variances. Output is regularised to be PSD via a tiny
    eigenvalue floor. All-zero design columns (structurally unidentified)
    get prior variance ``UNIDENTIFIED_PRIOR_VAR`` on the diagonal.
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
    Sigma = _inflate_unidentified_cov(Sigma, X_pooled)
    Sigma = 0.5 * (Sigma + Sigma.T)
    if Sigma.shape[0] > 0:
        min_eig = float(np.linalg.eigvalsh(Sigma).min())
        if min_eig < MIN_SIGMA2:
            Sigma = Sigma + (MIN_SIGMA2 - min_eig) * np.eye(Sigma.shape[0])
    return pooled_theta, Sigma, n_used


def fit_q_td_modify_joint_prior(
    df_fit: pd.DataFrame, *, gamma_terminal: float = GAMMA_TERMINAL,
) -> Dict[str, Any]:
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

        theta_u, s2_u, p_eta_u = _joint_fqi_iterate(
            [feats_u], gamma_terminal=gamma_terminal)
        if theta_u is None:
            user_thetas.append(None)
            s2_Q_users.append(None)
            continue
        if p_eta is None:
            p_eta = int(p_eta_u)
        user_thetas.append(theta_u)
        s2_Q_users.append(s2_u)
        feats_pool.append(feats_u)

    theta_pool, _, p_eta_pool = _joint_fqi_iterate(
        feats_pool, gamma_terminal=gamma_terminal)
    if p_eta is None and p_eta_pool is not None:
        p_eta = int(p_eta_pool)

    X_joint = None
    if theta_pool is not None and feats_pool and p_eta is not None:
        X_joint, _, _ = _mtd_joint_design(
            feats_pool, theta_pool[int(p_eta):], gamma_terminal=gamma_terminal)
    mu_0, Sigma_0, n_used = _pool_joint_user_fits(
        theta_pool, user_thetas, X_pooled=X_joint)
    eta_names, beta_names = _joint_feature_names(
        {"mu_0": mu_0, "p_eta": p_eta})
    _assert_named_coords_diffuse(
        Sigma_0, eta_names + beta_names,
        ("eta_b_tilde", "beta_b_tilde", "beta_A*b_tilde"))

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


# ──────────────────────────────────────────────────────────────────
# 7. Human-readable prior summaries
# ──────────────────────────────────────────────────────────────────
def _rl_context_names() -> list[str]:
    return [
        "yesterday_step_count",
        "prior2hour_step_count",
        "active_status_fraction_7days",
        "recent_burden",
        "walk_interaction_7d",
    ]


def _rl_my_names() -> list[str]:
    """Within-week EWMA summaries for M^Y streams (AM/PM pooled for fourSC)."""
    return [
        "M_Y_anticipated_affect_ewma",
        "M_Y_fourSC_ewma",
    ]


def _rl_me_names() -> list[str]:
    """Within-week EWMA summaries for M^E streams (AM/PM pooled for pageview)."""
    return [
        "M_E_pageview_ewma",
        "M_E_fitbit_wear_ewma",
        "M_E_survey_complete_ewma",
    ]


def _phi_state_names() -> list[str]:
    return [
        "intercept",
        "weekday_vs_weekend",
        "slot_pm",
        "E_w",
        "b_hat",
        "b_tilde",
    ] + _rl_my_names() + _rl_me_names() + _rl_context_names()


def _phi_action_names() -> list[str]:
    action_context = [
        "A",
        "A*E_w",
        "A*b_hat",
        "A*b_tilde",
    ] + [f"A*{name}" for name in _rl_context_names()] + [
        "A*weekday_vs_weekend",
        "A*slot_pm",
    ]
    from algorithm_helpers import ACTION_BLOCK_INCLUDE_M
    if ACTION_BLOCK_INCLUDE_M:
        action_context += [f"A*{name}" for name in _rl_my_names() + _rl_me_names()]
    return _phi_state_names() + action_context


def _phi_daily_mediator_names() -> list[str]:
    """Names for ``build_daily_mediator_phi`` (no state-only slot/weekday)."""
    return (
        ["intercept", "E_w", "b_hat", "b_tilde"]
        + _rl_my_names()
        + _rl_me_names()
        + _rl_context_names()
        + ["A", "A*E_w", "A*b_hat", "A*b_tilde"]
        + [f"A*{name}" for name in _rl_context_names()]
        + ["A*slot_pm"]
    )


def _phi_foursc_stage1_names() -> list[str]:
    """Names for slot-level ``build_foursc_stage1_phi``."""
    return (
        ["intercept", "weekday_vs_weekend", "slot_pm", "E_w", "b_hat", "b_tilde"]
        + _rl_context_names()
        + ["A", "A*slot_pm"]
    )


def _stage1_feature_names(name: str) -> list[str]:
    if name in _SLOT_MEDIATORS:
        return _phi_foursc_stage1_names()
    return _phi_daily_mediator_names()


def _joint_feature_names(q_joint: Dict[str, Any]) -> tuple[list[str], list[str]]:
    mu_joint = q_joint.get("mu_0")
    if mu_joint is None:
        return [], []
    n_joint = np.asarray(mu_joint, dtype=float).ravel().size
    p_eta = q_joint.get("p_eta")
    p_eta = 0 if p_eta is None else int(p_eta)
    eta_names = _names_with_fallback(
        ["eta_intercept", "eta_E_w", "eta_b_hat", "eta_b_tilde"],
        min(p_eta, n_joint),
        "eta_coef",
    )
    beta_n = max(n_joint - p_eta, 0)
    beta_names = [f"beta_{name}" for name in _names_with_fallback(
        _phi_action_names(), beta_n, "beta_coef"
    )]
    return eta_names, beta_names


def save_split_rl_prior_files(
    priors: Dict[str, Any],
    *,
    reward_path: Path = REWARD_PRIOR_PATH,
    rl_q_path: Path = RL_Q_PRIOR_PATH,
) -> dict[str, Path]:
    """Save reward-shaping eta and RL Q priors as separate JSON files.

    The combined ``rl_priors.json`` remains the canonical backward-compatible
    file used by ``experiment.py``. These split files are for easier inspection
    or independent loading.
    """
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    q_joint = priors.get("q_td_modify_joint") or {}
    eta_names, beta_names = _joint_feature_names(q_joint)

    reward_payload = {
        "model": "reward_shaping",
        "block": "eta",
        "feature_names": _phi_state_names(),
        **priors["reward"],
    }
    if "reward_redistribution" in priors:
        reward_payload["reward_redistribution"] = priors["reward_redistribution"]
    rl_q_payload = {
        "q_no_td_modify": {
            "model": "q_no_td_modify",
            "block": "beta",
            "feature_names": _phi_action_names(),
            **priors["q_no_td_modify"],
        },
        "q_td_modify_joint": {
            "model": "q_td_modify_joint",
            "block": "eta_beta",
            "eta_feature_names": eta_names,
            "beta_feature_names": beta_names,
            **q_joint,
        },
    }
    if "q_no_td_modify_g09" in priors:
        rl_q_payload["q_no_td_modify_g09"] = {
            "model": "q_no_td_modify_g09", "block": "beta",
            "feature_names": _phi_action_names(), **priors["q_no_td_modify_g09"],
        }
    if "q_no_td_modify_g099" in priors:
        rl_q_payload["q_no_td_modify_g099"] = {
            "model": "q_no_td_modify_g099", "block": "beta",
            "feature_names": _phi_action_names(), **priors["q_no_td_modify_g099"],
        }
    if "q_redistribution" in priors:
        rl_q_payload["q_redistribution"] = {
            name: {"model": f"q_redistribution_{name}", "block": "beta",
                   "feature_names": _phi_action_names(), **prior}
            for name, prior in priors["q_redistribution"].items()
        }

    with open(reward_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(reward_payload), f, indent=2, allow_nan=False)
    with open(rl_q_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(rl_q_payload), f, indent=2, allow_nan=False)
    return {"reward": reward_path, "rl_q": rl_q_path}


def _names_with_fallback(base_names: list[str], n: int, prefix: str) -> list[str]:
    if len(base_names) >= n:
        return base_names[:n]
    return base_names + [f"{prefix}_{i}" for i in range(len(base_names), n)]


def _diag_variance(cov: Any, n: int) -> np.ndarray:
    if cov is None:
        return np.full(n, np.nan, dtype=float)
    arr = np.asarray(cov, dtype=float)
    if arr.ndim == 1:
        out = arr.ravel()
    elif arr.ndim == 2:
        out = np.diag(arr)
    else:
        out = np.full(n, np.nan, dtype=float)
    if out.size < n:
        out = np.pad(out, (0, n - out.size), constant_values=np.nan)
    return out[:n]


def _summary_rows(
    *,
    prior_family: str,
    model: str,
    block: str,
    mean: Any,
    cov: Any,
    names: list[str],
) -> list[dict[str, Any]]:
    if mean is None:
        return []
    mu = np.asarray(mean, dtype=float).ravel()
    var = _diag_variance(cov, mu.size)
    feature_names = _names_with_fallback(names, mu.size, f"{model}_coef")
    return [
        {
            "prior_family": prior_family,
            "model": model,
            "block": block,
            "index": i,
            "feature": feature_names[i],
            "prior_mean": float(mu[i]),
            "prior_variance": float(var[i]),
        }
        for i in range(mu.size)
    ]


def build_prior_summary_tables(priors: Dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Return the three summary tables requested for audit.

    ``prior_variance`` is the diagonal entry of the prior covariance matrix.
    For the modified-TD joint prior the full covariance is still stored in
    ``rl_priors.json``; the table shows its marginal variances.
    """
    pf = priors["pf"]
    reward = priors["reward"]
    redistribution = priors.get("reward_redistribution") or {}
    q_no_mod = priors["q_no_td_modify"]
    q_g09 = priors.get("q_no_td_modify_g09")
    q_g099 = priors.get("q_no_td_modify_g099")
    q_redistribution = priors.get("q_redistribution") or {}
    q_joint = priors.get("q_td_modify_joint") or {}

    pf_rows: list[dict[str, Any]] = []
    for model_name, d in pf.items():
        pf_rows.extend(_summary_rows(
            prior_family="PF",
            model=model_name,
            block="theta",
            mean=d.get("nu_0"),
            cov=d.get("Gamma_0"),
            names=list(d.get("names") or []),
        ))

    rl_rows: list[dict[str, Any]] = []
    rl_rows.extend(_summary_rows(
        prior_family="RL",
        model="reward_shaping",
        block="eta",
        mean=reward.get("mu_0"),
        cov=reward.get("Sigma_0"),
        names=_phi_state_names(),
    ))
    for name, prior in (redistribution.get("daily_mediators") or {}).items():
        rl_rows.extend(_summary_rows(
            prior_family="RL", model=f"redistribution_stage1_{name}", block="eta",
            mean=prior.get("mu_0"), cov=prior.get("Sigma_0"),
            names=_stage1_feature_names(name),
        ))
    for name, prior in (redistribution.get("redistribution") or {}).items():
        rl_rows.extend(_summary_rows(
            prior_family="RL", model=f"redistribution_stage2_{name}", block="eta",
            mean=prior.get("mu_0"), cov=prior.get("Sigma_0"), names=[],
        ))
    rl_rows.extend(_summary_rows(
        prior_family="RL",
        model="q_no_td_modify",
        block="beta",
        mean=q_no_mod.get("mu_0"),
        cov=q_no_mod.get("Sigma_0"),
        names=_phi_action_names(),
    ))
    if q_g09 is not None:
        rl_rows.extend(_summary_rows(
            prior_family="RL", model="q_no_td_modify_g09", block="beta",
            mean=q_g09.get("mu_0"), cov=q_g09.get("Sigma_0"), names=_phi_action_names(),
        ))
    if q_g099 is not None:
        rl_rows.extend(_summary_rows(
            prior_family="RL", model="q_no_td_modify_g099", block="beta",
            mean=q_g099.get("mu_0"), cov=q_g099.get("Sigma_0"), names=_phi_action_names(),
        ))
    for name, prior in q_redistribution.items():
        rl_rows.extend(_summary_rows(
            prior_family="RL", model=f"q_redistribution_{name}", block="beta",
            mean=prior.get("mu_0"), cov=prior.get("Sigma_0"), names=_phi_action_names(),
        ))

    joint_rows: list[dict[str, Any]] = []
    mu_joint = q_joint.get("mu_0")
    if mu_joint is not None:
        mu_joint_arr = np.asarray(mu_joint, dtype=float).ravel()
        p_eta = q_joint.get("p_eta")
        p_eta = 0 if p_eta is None else int(p_eta)
        var_joint = _diag_variance(q_joint.get("Sigma_0"), mu_joint_arr.size)
        eta_names = _names_with_fallback(
            ["eta_intercept", "eta_E_w", "eta_b_hat", "eta_b_tilde"],
            min(p_eta, mu_joint_arr.size),
            "eta_coef",
        )
        beta_n = max(mu_joint_arr.size - p_eta, 0)
        beta_names = [f"beta_{name}" for name in _names_with_fallback(
            _phi_action_names(), beta_n, "beta_coef"
        )]
        names = eta_names + beta_names
        for i, feature in enumerate(names):
            joint_rows.append({
                "prior_family": "Joint Modified TD",
                "model": "q_td_modify_joint",
                "block": "eta" if i < p_eta else "beta",
                "index": i,
                "feature": feature,
                "prior_mean": float(mu_joint_arr[i]),
                "prior_variance": float(var_joint[i]),
            })

    columns = [
        "prior_family",
        "model",
        "block",
        "index",
        "feature",
        "prior_mean",
        "prior_variance",
    ]
    return {
        "pf": pd.DataFrame(pf_rows, columns=columns),
        "rl": pd.DataFrame(rl_rows, columns=columns),
        "joint": pd.DataFrame(joint_rows, columns=columns),
    }


def _format_table_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value == 0.0:
            return "0.000000"
        if abs(value) < 1e-4 or abs(value) >= 1e6:
            return f"{value:.6e}"
        return f"{value:.6f}"
    return str(value)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(_format_table_value(row[col]) for col in columns)
            + " |"
        )
    return "\n".join(lines)


def save_prior_summary_tables(
    priors: Dict[str, Any],
    *,
    pf_path: Path = PF_SUMMARY_PATH,
    rl_path: Path = RL_SUMMARY_PATH,
    reward_path: Path = REWARD_SUMMARY_PATH,
    rl_q_path: Path = RL_Q_SUMMARY_PATH,
    joint_path: Path = JOINT_SUMMARY_PATH,
    markdown_path: Path = SUMMARY_TABLE_PATH,
    pooled_pf_regression: Optional[pd.DataFrame] = None,
    pooled_pf_regression_path: Path = POOLED_PF_REGRESSION_PATH,
    pooled_pf_gee: Optional[pd.DataFrame] = None,
    pooled_pf_gee_path: Path = POOLED_PF_GEE_PATH,
    pooled_rl_q: Optional[pd.DataFrame] = None,
    pooled_rl_q_path: Path = POOLED_RL_Q_PATH,
) -> dict[str, Path]:
    tables = build_prior_summary_tables(priors)
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    tables["pf"].to_csv(pf_path, index=False)
    tables["rl"].to_csv(rl_path, index=False)
    tables["rl"].loc[
        tables["rl"]["model"] == "reward_shaping"
    ].to_csv(reward_path, index=False)
    tables["rl"].loc[
        tables["rl"]["model"] == "q_no_td_modify"
    ].to_csv(rl_q_path, index=False)
    tables["joint"].to_csv(joint_path, index=False)

    md = [
        "# Prior Mean and Variance Summary",
        "",
        "`prior_variance` is the diagonal entry of the prior covariance. "
        "For `q_td_modify_joint`, the full covariance remains in "
        "`rl_priors.json`; this table reports marginal variances.",
        "",
        "## PF Priors",
        "",
        _dataframe_to_markdown(tables["pf"]),
        "",
        "## RL Priors",
        "",
        _dataframe_to_markdown(tables["rl"]),
        "",
        "## Joint Modified-TD Priors",
        "",
        _dataframe_to_markdown(tables["joint"]),
        "",
    ]
    out_paths = {
        "pf": pf_path,
        "rl": rl_path,
        "reward": reward_path,
        "rl_q": rl_q_path,
        "joint": joint_path,
        "markdown": markdown_path,
    }
    if pooled_pf_regression is not None and not pooled_pf_regression.empty:
        pooled_pf_regression.to_csv(pooled_pf_regression_path, index=False)
        md.extend([
            "## Pooled PF Regression Coefficients",
            "",
            "Coefficients are from the all-user stacked ridge fits used as PF "
            "prior means. `p_value` uses ridge sandwich standard errors and is "
            "approximate because ridge shrinks coefficients. "
            "`identified=False` marks zero-variance design columns; SE / "
            "p-values are omitted for those rows.",
            "",
            _dataframe_to_markdown(pooled_pf_regression),
            "",
        ])
        out_paths["pooled_pf_regression"] = pooled_pf_regression_path

    if pooled_pf_gee is not None and not pooled_pf_gee.empty:
        pooled_pf_gee.to_csv(pooled_pf_gee_path, index=False)
        md.extend([
            "## Pooled PF GEE Coefficients",
            "",
            "Population-averaged GEE fits on the same stacked PF designs, "
            "clustered by `ParticipantIdentifier`. `p_value` uses GEE "
            "sandwich standard errors with exchangeable working correlation. "
            "`identified=False` marks zero-variance design columns; SE / "
            "p-values are omitted for those rows. "
            "These are for inference/audit only; PF priors still use ridge.",
            "",
            _dataframe_to_markdown(pooled_pf_gee),
            "",
        ])
        out_paths["pooled_pf_gee"] = pooled_pf_gee_path

    if pooled_rl_q is not None and not pooled_rl_q.empty:
        pooled_rl_q.to_csv(pooled_rl_q_path, index=False)
        md.extend([
            "## Pooled RL Q GEE Coefficients",
            "",
            "Gaussian GEE on the final fitted-Q regression from pooled FQI "
            "(``phi_obs`` vs bootstrap targets), clustered by participant. "
            "``identified=False`` marks structurally unused features with "
            "zero design variance (e.g. ``b_tilde``, masked day-6 mediators); "
            "SE / p-values are omitted for those rows. RL priors still use "
            "ridge-FQI for ``mu_0_micro``; all-zero design columns get "
            "``UNIDENTIFIED_PRIOR_VAR`` instead of the ``MIN_SIGMA2`` floor.",
            "",
            _dataframe_to_markdown(pooled_rl_q),
            "",
        ])
        out_paths["pooled_rl_q"] = pooled_rl_q_path

    markdown_path.write_text("\n".join(md), encoding="utf-8")
    return out_paths


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
    qn_g09 = raw.get("q_no_td_modify_g09")
    qn_g099 = raw.get("q_no_td_modify_g099")
    q_redistribution = raw.get("q_redistribution")
    qj = raw.get("q_td_modify_joint")
    redistribution = raw.get("reward_redistribution")

    nu_Y, G_Y = trim_pf_cae_prior(arr(pf["CAE"]["nu_0"]), arr(pf["CAE"]["Gamma_0"]))

    out = {
        "nu_0_MY":     [arr(pf["fourSC"]["nu_0"]), arr(pf["antic"]["nu_0"])],
        "Gamma_0_MY":  [arr(pf["fourSC"]["Gamma_0"]), arr(pf["antic"]["Gamma_0"])],
        "sigma2_MY":   [float(pf["fourSC"]["sigma2"]), float(pf["antic"]["sigma2"])],
        "nu_0_Y":      nu_Y,
        "Gamma_0_Y":   G_Y,
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
    if qn_g09 is not None:
        out["q_no_td_modify_g09"] = {
            "mu_0": arr(qn_g09["mu_0"]), "Sigma_0": arr(qn_g09["Sigma_0"]),
            "sigma2": float(qn_g09["sigma2"]),
        }
    if qn_g099 is not None:
        out["q_no_td_modify_g099"] = {
            "mu_0": arr(qn_g099["mu_0"]), "Sigma_0": arr(qn_g099["Sigma_0"]),
            "sigma2": float(qn_g099["sigma2"]),
        }
    if q_redistribution is not None:
        out["q_redistribution"] = {
            name: {"mu_0": arr(prior["mu_0"]),
                   "Sigma_0": arr(prior["Sigma_0"]), "sigma2": float(prior["sigma2"])}
            for name, prior in q_redistribution.items()
            if name in {"v3", "v4", "v4_resid"}
        }
    if redistribution is not None:
        out["daily_mediator_priors"] = {
            name: {"mu_0": arr(prior["mu_0"]),
                   "Sigma_0": arr(prior["Sigma_0"]),
                   "sigma2": float(prior["sigma2"])}
            for name, prior in redistribution["daily_mediators"].items()
        }
        out["redistribution_priors"] = {
            name: {"mu_0": arr(prior["mu_0"]),
                   "Sigma_0": arr(prior["Sigma_0"]),
                   "sigma2": float(prior["sigma2"]),
                   "psi_has_next_my": bool(prior.get("psi_has_next_my", False)),
                   "psi_has_next_me": bool(prior.get("psi_has_next_me", False)),
                   "psi_has_pv_hat": bool(prior.get("psi_has_pv_hat", True))}
            for name, prior in redistribution["redistribution"].items()
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


def fit_all_priors(df_fit: pd.DataFrame) -> Dict[str, Any]:
    """Fit every runtime prior from an already-filtered analysis dataset.

    This pure helper is used by the experiment driver's leave-one-out warm
    start.  It deliberately performs no file I/O, so a held-out participant's
    data cannot leak through a previously saved ``rl_priors.json``.
    """
    pf = fit_pf_priors(df_fit)
    reward = fit_reward_prior(df_fit)  # retained for legacy compatibility
    reward_redistribution = fit_reward_redistribution_priors(df_fit, gamma_bar=0.9)
    q_no_mod = fit_q_prior(df_fit, gamma_terminal=0.5)
    q_no_mod_g09 = fit_q_prior(df_fit, gamma_terminal=0.9)
    q_no_mod_g099 = fit_q_prior(df_fit, gamma_terminal=0.99)
    q_redistribution = {
        "v3": fit_q_potential_prior(df_fit, gamma_terminal=0.9),
        "v4": fit_q_redistribution_prior(df_fit, gamma_terminal=0.9),
        "v4_resid": fit_q_redistribution_prior(
            df_fit, gamma_terminal=0.9, add_terminal_residual=True),
    }
    q_mod_joint = fit_q_td_modify_joint_prior(df_fit, gamma_terminal=0.9)
    return {
        "pf": pf,
        "reward": reward,
        "reward_redistribution": reward_redistribution,
        "q_no_td_modify": q_no_mod,
        "q_no_td_modify_g09": q_no_mod_g09,
        "q_no_td_modify_g099": q_no_mod_g099,
        "q_redistribution": q_redistribution,
        "q_td_modify_joint": q_mod_joint,
    }


def precompute_leave_one_out_priors(
    df_fit: Optional[pd.DataFrame] = None,
    output_dir: Path = LOO_PRIOR_DIR,
    held_out_ids: Optional[List[int]] = None,
) -> Dict[int, Path]:
    """Write one complete, leakage-free runtime-prior file per participant.

    ``held_out_<uid>.json`` is fitted from every other participant and is the
    exact file selected by ``experiment.py --prior-mode loo``.  This function
    is intentionally offline: experiment execution only reads these files.
    """
    df_path = WORK_DIR / "df_fit_11week.csv"
    df = load_df_fit(df_path) if df_fit is None else df_fit.copy()
    available_ids = sorted(int(x) for x in df["ParticipantIdentifier"].unique())
    ids = available_ids if held_out_ids is None else sorted(set(map(int, held_out_ids)))
    missing = sorted(set(ids) - set(available_ids))
    if missing:
        raise ValueError(f"requested held-out IDs absent from df_fit: {missing}")
    if len(available_ids) < 3:
        raise ValueError("leave-one-out fitting requires at least three participants")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[int, Path] = {}
    for uid in ids:
        train = df.loc[df["ParticipantIdentifier"].astype(int) != uid].copy()
        if train["ParticipantIdentifier"].nunique() < 2:
            raise ValueError(f"not enough training participants after holding out {uid}")
        print(f"Fitting leave-one-out priors: hold out {uid} ({len(train)} rows) ...", flush=True)
        path = output_dir / f"held_out_{uid}.json"
        save_priors(fit_all_priors(train), path)
        paths[uid] = path
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "source_df": str(df_path.resolve()) if df_fit is None else "caller-supplied DataFrame",
            "held_out_ids": ids,
            "files": {str(uid): path.name for uid, path in paths.items()},
        }, f, indent=2)
    return paths


def main() -> Dict[str, Any]:
    df_fit = load_df_fit()
    print(f"df_fit: {len(df_fit)} rows, "
          f"{df_fit['ParticipantIdentifier'].nunique()} participants")

    print("Fitting PF mediator / Y / tY priors ...")
    pf = fit_pf_priors(df_fit)
    for name, d in pf.items():
        n = None if d["nu_0"] is None else len(d["nu_0"])
        print(f"  {name:9s}: p={n}, sigma2={d['sigma2']:.4f}")

    print("Summarizing pooled PF regression coefficients ...")
    pooled_pf_regression = build_pooled_pf_regression_summary(df_fit)
    for model in ("fourSC", "antic", "CAE"):
        sub = pooled_pf_regression.loc[pooled_pf_regression["model"] == model]
        if sub.empty:
            continue
        identified = sub.get("identified", pd.Series(True, index=sub.index))
        n_sig = int(sub.loc[identified, "significant_0.05"].sum())
        print(
            f"  ridge {model:7s}: {int(identified.sum())} identified / "
            f"{len(sub)} coefficients, {n_sig} significant at 0.05"
        )

    print("Summarizing pooled PF GEE coefficients ...")
    pooled_pf_gee = build_pooled_pf_gee_summary(df_fit)
    for model in ("fourSC", "antic", "CAE"):
        sub = pooled_pf_gee.loc[pooled_pf_gee["model"] == model]
        if sub.empty:
            continue
        identified = sub.get("identified", pd.Series(True, index=sub.index))
        n_sig = int(sub.loc[identified, "significant_0.05"].sum())
        print(
            f"  GEE   {model:7s}: {int(identified.sum())} identified / "
            f"{len(sub)} coefficients, {n_sig} significant at 0.05"
        )

    print("Fitting reward-shaping eta prior ...")
    reward = fit_reward_prior(df_fit)
    print(f"  reward: p={len(reward['mu_0'])}, sigma2={reward['sigma2']:.4f}")

    print("Fitting V4 two-stage redistribution priors (γ̄=0.9) ...")
    reward_redistribution = fit_reward_redistribution_priors(df_fit, gamma_bar=0.9)
    for name, prior in reward_redistribution["daily_mediators"].items():
        print(f"  Stage 1 {name}: p={len(prior['mu_0'])}, sigma2={prior['sigma2']:.4f}")
    for name, prior in reward_redistribution["redistribution"].items():
        print(f"  Stage 2 {name}: p={len(prior['mu_0'])}, sigma2={prior['sigma2']:.4f}")

    print("Fitting Q-function prior (no TD-modify, γ̄=0.5 for V7) ...")
    q_no_mod = fit_q_prior(df_fit, gamma_terminal=0.5)
    print(f"  Q (no TD-modify, γ̄=0.5): p={len(q_no_mod['mu_0'])}, "
        f"sigma2={q_no_mod['sigma2']:.4f}")

    print("Fitting Q prior for the γ̄=0.9 base (V1) ...")
    q_no_mod_g09 = fit_q_prior(df_fit, gamma_terminal=0.9)
    print(f"  Q (gamma=0.9): p={len(q_no_mod_g09['mu_0'])}, "
          f"sigma2={q_no_mod_g09['sigma2']:.4f}")

    print("Fitting Q prior for the γ̄=0.99 base (V8) ...")
    q_no_mod_g099 = fit_q_prior(df_fit, gamma_terminal=0.99)
    print(f"  Q (gamma=0.99): p={len(q_no_mod_g099['mu_0'])}, "
          f"sigma2={q_no_mod_g099['sigma2']:.4f}")

    print("Fitting V5 / V6 / V6-leftover Q priors (γ̄=0.9) ...")
    q_redistribution = {
        "v3": fit_q_potential_prior(df_fit, gamma_terminal=0.9),
        "v4": fit_q_redistribution_prior(df_fit, gamma_terminal=0.9),
        "v4_resid": fit_q_redistribution_prior(
            df_fit, gamma_terminal=0.9, add_terminal_residual=True),
    }
    for variant, prior in q_redistribution.items():
        print(f"  Q ({variant}): p={len(prior['mu_0'])}, sigma2={prior['sigma2']:.4f}")

    print("Summarizing pooled RL Q GEE coefficients ...")
    pooled_rl_q = build_pooled_rl_q_summary(df_fit)
    if pooled_rl_q.empty:
        print("  pooled RL Q GEE: no users contributed")
    else:
        identified = pooled_rl_q["identified"]
        n_sig = int(pooled_rl_q.loc[identified, "significant_0.05"].sum())
        print(
            f"  pooled RL Q GEE: {int(identified.sum())} identified / "
            f"{len(pooled_rl_q)} coefficients, {n_sig} significant at 0.05"
        )

    print("Fitting joint (alpha, beta) prior for modified-TD-loss RLSVI (γ̄=0.9) ...")
    q_mod_joint = fit_q_td_modify_joint_prior(df_fit, gamma_terminal=0.9)
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
        "reward_redistribution": reward_redistribution,
        "q_no_td_modify":      q_no_mod,
        "q_no_td_modify_g09":  q_no_mod_g09,
        "q_no_td_modify_g099": q_no_mod_g099,
        "q_redistribution":    q_redistribution,
        "q_td_modify_joint":   q_mod_joint,
    }
    path = save_priors(priors)
    print(f"Saved priors -> {path}")
    split_paths = save_split_rl_prior_files(priors)
    print("Saved split RL prior JSON files:")
    print(f"  Reward shaping -> {split_paths['reward']}")
    print(f"  RL Q           -> {split_paths['rl_q']}")
    table_paths = save_prior_summary_tables(
        priors,
        pooled_pf_regression=pooled_pf_regression,
        pooled_pf_gee=pooled_pf_gee,
        pooled_rl_q=pooled_rl_q,
    )
    print("Saved prior summary tables:")
    print(f"  PF    -> {table_paths['pf']}")
    print(f"  RL combined     -> {table_paths['rl']}")
    print(f"  Reward shaping  -> {table_paths['reward']}")
    print(f"  RL Q            -> {table_paths['rl_q']}")
    print(f"  Joint           -> {table_paths['joint']}")
    if "pooled_pf_regression" in table_paths:
        print(f"  Pooled PF ridge -> {table_paths['pooled_pf_regression']}")
    if "pooled_pf_gee" in table_paths:
        print(f"  Pooled PF GEE   -> {table_paths['pooled_pf_gee']}")
    if "pooled_rl_q" in table_paths:
        print(f"  Pooled RL Q GEE -> {table_paths['pooled_rl_q']}")
    print(f"  MD              -> {table_paths['markdown']}")
    return priors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate shared or leave-one-out RL priors.")
    parser.add_argument(
        "--loo", action="store_true",
        help="precompute one held-out prior JSON per participant instead of one shared prior",
    )
    parser.add_argument(
        "--loo-output-dir", type=Path, default=LOO_PRIOR_DIR,
        help="directory for --loo files (default: <params-dir>/loo_priors)",
    )
    parser.add_argument(
        "--loo-users", nargs="+", type=int, default=None,
        help="optional subset of held-out participant IDs to precompute",
    )
    cli_args = parser.parse_args()
    if cli_args.loo:
        paths = precompute_leave_one_out_priors(
            output_dir=cli_args.loo_output_dir, held_out_ids=cli_args.loo_users)
        print(f"Saved {len(paths)} leave-one-out prior bundles to {cli_args.loo_output_dir}")
    else:
        main()
