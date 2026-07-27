"""Pooled, finite-horizon fitted-Q model for the ADAPR MRT testbed.

--------------------------------------------------------------------------
The decision process
--------------------------------------------------------------------------
The RL agent acts at 12 within-week decision slots: 6 days (Mon-Sat) x 2 slots
(AM/PM). We index a slot by ``h = 1..12``.
The only reward is the weekly ``CAE_avg_norm`` (weekly context-aware
engagement), realised at the terminal slot ``(d=6, t=2)`` and then
bootstrapping into the *start of the next week* ``s_{1,1}^{k+1}``.

The FQI target is:

    non-terminal (d,t):  target = gamma_{d,t} * max_a Q(s_{next slot}, a)
    terminal   (6,2):    target = R_k + gamma_{6,2} * max_a Q(s_{1,1}^{k+1}, a)

with ``R_k`` = the weekly CAE realised during week ``k``.

--------------------------------------------------------------------------
Step 1 - Finite-horizon, per-slot Q functions (backward induction)
--------------------------------------------------------------------------
Each week is treated as a finite-horizon episode of
``H = 12`` steps and we fit **12 distinct per-slot coefficient vectors**
``beta_1, ..., beta_12`` by backward induction:

    beta_12 : regress   R_k + gamma_{6,2} * V_start(s^{k+1})   on phi(s_{k,12}, a_obs)
                         (for the final observed week, use R_k only)
    beta_h  : regress   gamma_{d,t} * max_a phi(s_{k,h+1}, a) . beta_{h+1}
                                                              on phi(s_{k,h}, a_obs)
              for h = 11, 10, ..., 1

where ``V_start(s^{k+1}) = max_a phi(s_{1,1}^{k+1}, a) . beta_1`` is the value of
the next week's start state. The coefficients are *shared across weeks* at the
same within-week slot position (non-stationary across slots, stationary across
weeks), so ``beta_h`` is estimated by pooling every user's week-``k`` slot-``h``
transition into one ridge regression -- this is the "pooled" fitted-Q.

Because the terminal slot bootstraps into ``beta_1`` (next week's start), the
per-slot equations couple across the week boundary. We resolve this with a
*bounded* number of backward sweeps equal to the maximum number of training
weeks any user contributes: value information propagates exactly one week per
sweep, so this is exact finite-horizon dynamic programming, not an
infinite-horizon fixed-point iteration. See ``N_BACKWARD_SWEEPS``.

--------------------------------------------------------------------------
Step 2 - Choice of discount factor
--------------------------------------------------------------------------
The per-slot discount matrix ``gamma_dt`` (shape ``(6, 2)``) mirrors
``experiment._gamma_dt_micro`` and is selected by ``gamma_bar``:

  * ``gamma_bar = 0.5`` (default): every slot uses ``gamma = 0.5^(1/12) ~ 0.9439``.
    The exponent ``1/12`` spreads a per-week discount of ``0.5`` evenly across
    the 12 in-week transitions, so the cumulative discount from the start of a
    week to its terminal slot equals ``0.5`` .
  * ``gamma_bar = 0.0`` (myopic): non-terminal slots use ``gamma = 1`` and only
    the terminal transition ``(6,2)`` uses ``gamma = 0``. Value flows *within* a
    week but stops at the week boundary, so each week's Q reflects only that
    week's own reward.
  * ``gamma_bar = 1.0``: undiscounted finite horizon (all ``gamma = 1``).

--------------------------------------------------------------------------
Step 3 - Missing-data (CAE) handling
--------------------------------------------------------------------------
1. Users: participants with no observed weekly CAE at all are dropped upstream
   in ``est_prior.load_df_fit`` (``_filter_users_with_cae_obs``).
2. CAE imputation: assemble one canonical raw CAE trajectory per user over
   study weeks 1--W only (week 1 is the baseline week; there is no separate
   pre-study CAE). Recover missing end-of-week values from the next week's
   observed lag when available, then kernel-impute remaining gaps once using
   within-week ``M^Y`` summaries (mean Mon-Sat four-hour step count and mean
   Mon-Sat anticipated affect) plus ``week_frac``. Derive reward and lagged
   state from that completed trajectory: week 1's lag equals its own CAE
   (baseline week), and week ``w>=2`` uses ``C_{w-1}``. This prevents circular
   or mutually inconsistent state/reward imputations.
3. Normalization: completed raw CAE is transformed separately with the saved
   ``CAE_avg`` and ``CAE_avg_lastweek`` shift/scale parameters. The lagged
   state feature in the fitted-Q design is the imputed
   ``CAE_avg_lastweek_norm`` (not online particle-filter belief summaries
   ``b_hat`` / ``b_tilde``). The always-zero ``b_tilde`` column is dropped.
4. Other non-CAE feature gaps still use the per-participant mean imputation
   inside ``_user_weekly_tensors``.
5. Horizon / first and last week: this is an infinite-horizon problem.
   * Week 1 is the **baseline week** only: its CAE defines
     ``CAE_avg_lastweek`` for week 2, but week 1 itself is not a training
     week (it has no valid previous-week CAE state feature).
   * The last observed week is also **not** a training week; only its
     Monday-morning ``s_{1,1}`` is kept as the terminal bootstrap successor
     for the previous week.
   With weeks 1--12 in ``df_fit``, that means **10 training weeks** per user
   (weeks 2--11), plus week-1 CAE as the lag source and week-12 Monday AM
   as bootstrap only.

--------------------------------------------------------------------------
Outputs (written under ``fittedQ/``)
--------------------------------------------------------------------------
  * ``fitted_q_pooled.json``      - the fitted model (per-horizon beta, sigma2,
                                     counts, discount metadata, feature names).
  * ``prior_summary_fitted_q.csv``- tidy (horizon, d, t, feature, coefficient,
                                     bootstrap CI) table.
  * ``fitted_q_tables.md``        - human-readable coefficient-by-horizon table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the prior-estimation machinery so the fitted-Q design stays aligned
# with the RL agent's ``phi`` construction. Offline, the shared builder's
# ``b_hat`` argument carries imputed ``CAE_avg_lastweek_norm`` (not a
# particle-filter belief), and the always-zero ``b_tilde`` column is dropped.
# ``est_prior`` has no import-time side effects (its ``main()`` is
# ``__main__``-guarded).
from est_prior import (
    K_SLOTS_WEEK,
    MIN_SIGMA2,
    N_RL_SLOTS_WEEK,
    RIDGE_ALPHA_RL,
    GAMMA_BAR,
    _phi_action,
    _phi_action_names,
    _ridge_fit,
    _user_weekly_tensors,
    load_df_fit,
)

RAW_DATA_DIR = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPR-MRT-Testbed/env_para_vanilla")
WORK_DIR = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPR-MRT-Testbed/fittedQ")


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────
GAMMA_BAR_DEFAULT = GAMMA_BAR          # 0.5 (matches experiment.py micro agent)

# Number of backward sweeps. ``None`` => auto: the maximum number of training
# weeks across contributing users (exact finite-horizon propagation). An
# explicit integer overrides the auto value (still bounded, never a fixed-point
# convergence loop).
N_BACKWARD_SWEEPS: Optional[int] = None

# Stopping / diagnostic controls.
# ``N_BACKWARD_SWEEPS=None`` still means auto-select from the data horizon.
# The truncation bound is computed and reported by default. When
# ``USE_TRUNCATION_BOUND_FOR_MAX_SWEEPS`` is True, the auto sweep cap is
# ``max(data_horizon, truncation_bound)`` so the fit uses at least enough sweeps
# for the infinite-horizon tail to fall below ``FITTED_Q_TRUNCATION_TOL``.
FITTED_Q_TRUNCATION_TOL: Optional[float] = 1e-3
FITTED_Q_CONVERGENCE_TOL: Optional[float] = 1e-4
FITTED_Q_MIN_CONVERGENCE_SWEEPS = 2
FITTED_Q_EARLY_STOP_ON_CONVERGENCE = False
FITTED_Q_USE_TRUNCATION_BOUND_FOR_MAX_SWEEPS = True

FITTED_Q_JSON_PATH = WORK_DIR / "fitted_q_pooled.json"
FITTED_Q_SUMMARY_PATH = WORK_DIR / "prior_summary_fitted_q.csv"
FITTED_Q_TABLE_PATH = WORK_DIR / "fitted_q_tables.md"
FITTED_Q_PLOT_PATHS = {
    "baseline": WORK_DIR / "fitted_q_coefficients_baseline.png",
    "M_Y_fourSC": WORK_DIR / "fitted_q_coefficients_M_Y_fourSC.png",
    "M_Y_anticipated_affect": (
        WORK_DIR / "fitted_q_coefficients_M_Y_anticipated_affect.png"
    ),
    "M_E_pageview": WORK_DIR / "fitted_q_coefficients_M_E_pageview.png",
    "M_E_fitbit_wear": WORK_DIR / "fitted_q_coefficients_M_E_fitbit_wear.png",
    "M_E_survey_complete": (
        WORK_DIR / "fitted_q_coefficients_M_E_survey_complete.png"
    ),
    "interactions": WORK_DIR / "fitted_q_coefficients_interactions.png",
}

# Participant-level (cluster) bootstrap for coefficient CIs. Imputation is held
# fixed; users are resampled with replacement and FQI is re-fit.
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CI_LEVEL = 0.95

# Gaussian-kernel bandwidth for Nadaraya-Watson CAE imputation. ``None`` selects
# the median pairwise distance among rows with observed CAE (median heuristic).
KERNEL_CAE_BANDWIDTH: Optional[float] = None

KERNEL_IMPUTE_FEATURE_NAMES = [
    "M_Y_fourSC_week_mean",
    "M_Y_anticipated_affect_week_mean",
    "week_frac",
]
STD_PARAMS_PATH = RAW_DATA_DIR / "std_params.json"

# ──────────────────────────────────────────────────────────────────
# Discount factor
# ──────────────────────────────────────────────────────────────────
def gamma_dt_matrix(gamma_bar: float) -> np.ndarray:
    """Per-slot discount matrix ``gamma_{d,t}`` (mirrors ``experiment._gamma_dt_micro``).

    See the module docstring "Choice of discount factor" for the semantics of
    each ``gamma_bar`` regime.
    """
    if gamma_bar == 0.0:
        gamma_dt = np.ones((6, 2))
        gamma_dt[5, 1] = 0.0
        return gamma_dt
    return (gamma_bar ** (1.0 / 12.0)) * np.ones((6, 2))


def _empirical_reward_bound(rewards: np.ndarray) -> float:
    """Empirical bound ``B_r`` used only for truncation diagnostics.

    The theoretical FQI truncation bound assumes a known uniform reward bound.
    Here we report the empirical absolute maximum of the terminal weekly reward
    vector used by this fit. This is a diagnostic unless rewards are known to be
    clipped to this range by design.
    """
    arr = np.asarray(rewards, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.max(np.abs(finite)))


def _discounted_return_bound_sweeps(
    *,
    gamma: float,
    reward_bound: float,
    tolerance: Optional[float],
) -> Optional[int]:
    """Compute N such that 2 * gamma^N * B_r / (1 - gamma)^2 <= tolerance.

    This is the classical discounted-return truncation bound. In this code one
    backward sweep propagates value across one week boundary, so callers should
    pass the weekly discount ``gamma_bar`` rather than the per-slot discount.

    Returns ``None`` when the bound is not applicable, e.g. gamma >= 1 or no
    tolerance requested.
    """
    if tolerance is None:
        return None
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("truncation_tol must be positive or None.")
    if reward_bound <= 0.0:
        return 1
    if gamma == 0.0:
        return 1
    if not (0.0 < gamma < 1.0):
        return None

    ratio = tolerance * (1.0 - gamma) ** 2 / (2.0 * reward_bound)
    if ratio >= 1.0:
        return 1
    return max(1, int(np.ceil(np.log(ratio) / np.log(gamma))))


def _q_prediction_change_diagnostics(
    beta_old: List[np.ndarray],
    beta_new: List[np.ndarray],
    designs: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """Measure fitted-Q prediction changes across two backward sweeps.

    Prediction-space diagnostics are preferred to raw coefficient changes because
    ridge coefficients can move substantially when features are correlated even
    if fitted Q-values barely change.
    """
    abs_changes: List[float] = []
    rel_changes: List[float] = []
    coef_abs_changes: List[float] = []
    coef_rel_changes: List[float] = []
    eps = 1e-8

    for h, x_h in enumerate(designs["x_obs"]):
        q_old = x_h @ beta_old[h]
        q_new = x_h @ beta_new[h]
        ok = np.isfinite(q_old) & np.isfinite(q_new)
        if ok.any():
            diff = q_new[ok] - q_old[ok]
            abs_changes.append(float(np.max(np.abs(diff))))
            denom = max(float(np.linalg.norm(q_old[ok])),
                        float(np.linalg.norm(q_new[ok])), eps)
            rel_changes.append(float(np.linalg.norm(diff) / denom))

        coef_diff = beta_new[h] - beta_old[h]
        coef_abs_changes.append(float(np.max(np.abs(coef_diff))))
        coef_denom = max(float(np.linalg.norm(beta_old[h])),
                         float(np.linalg.norm(beta_new[h])), eps)
        coef_rel_changes.append(float(np.linalg.norm(coef_diff) / coef_denom))

    return {
        "max_abs_q_change": max(abs_changes) if abs_changes else None,
        "max_rel_q_change": max(rel_changes) if rel_changes else None,
        "max_abs_beta_change": max(coef_abs_changes) if coef_abs_changes else None,
        "max_rel_beta_change": max(coef_rel_changes) if coef_rel_changes else None,
    }


def _strip_unused_phi_columns(phi: np.ndarray) -> np.ndarray:
    """Drop structurally unused columns from the shared RL ``phi_action`` map.

    Removed columns:
      * the hardcoded-zero context placeholder (state block and action block)
      * ``b_tilde`` (always 0 offline: we use imputed ``CAE_avg_lastweek``, not
        particle-filter belief uncertainty)

    The lagged CAE value is still supplied through the shared builder's
    ``b_hat`` argument (that is how ``est_prior._phi_action`` wires
    ``CAE_avg_lastweek_norm``), but the fitted-Q feature is named
    ``CAE_avg_lastweek`` after stripping.
    """
    arr = np.asarray(phi, dtype=float)
    raw_p = arr.shape[-1]
    action_block_size = 18  # 9 action terms + 9 action-by-context terms
    b_tilde_idx = 9
    state_placeholder = raw_p - action_block_size - 1
    action_placeholder = raw_p - 1
    drop_idx = [b_tilde_idx, state_placeholder, action_placeholder]
    unused_values = arr[..., drop_idx]
    if not np.allclose(unused_values, 0.0):
        raise ValueError(
            "Expected unused fitted-Q columns (b_tilde, context placeholders) "
            "to be identically zero."
        )
    return np.delete(arr, drop_idx, axis=-1)


def _feature_names_for(p: int) -> List[str]:
    """Return names for the offline fitted-Q design.

    Starts from the shared RL ``phi_action`` names, renames the belief slots to
    the imputed lagged CAE feature they actually carry offline, and drops the
    unused ``b_tilde`` column.
    """
    names = []
    for name in _phi_action_names():
        if name == "b_tilde":
            continue
        names.append(name.replace("b_hat", "CAE_avg_lastweek"))
    if len(names) != p:
        raise ValueError(f"Expected {len(names)} fitted-Q features, got {p}.")
    return names


# ──────────────────────────────────────────────────────────────────
# Kernel imputation for the CAE state feature (primary feature handling)
# ──────────────────────────────────────────────────────────────────
def _nanmean_or_nan(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.nan
    return float(np.mean(x[finite]))


def _week_my_impute_features(week_dat: pd.DataFrame) -> np.ndarray:
    """Within-week ``M^Y`` summaries used to impute end-of-week CAE.

    * ``fourSC``: mean ``4hour_step_norm`` over the 12 RL slots (Mon-Sat × AM/PM)
    * anticipated affect: mean ``anticipated_affect_norm`` over Mon-Sat mornings
    """
    rl = week_dat.iloc[:N_RL_SLOTS_WEEK]
    foursc = rl["4hour_step_norm"].to_numpy(dtype=float)
    antic_am = week_dat.iloc[:N_RL_SLOTS_WEEK:2]["anticipated_affect_norm"].to_numpy(
        dtype=float
    )
    return np.array([
        _nanmean_or_nan(foursc),
        _nanmean_or_nan(antic_am),
    ], dtype=float)


def kernel_impute_cae_trajectory(
    df_fit: pd.DataFrame,
    *,
    bandwidth: Optional[float] = KERNEL_CAE_BANDWIDTH,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Complete one canonical raw CAE trajectory and derive reward plus lag.

    For user ``i`` with ``W`` observed study weeks, construct

        C_i = (C_{i,1}, ..., C_{i,W}),

    where ``C_{i,w}`` is week ``w``'s end-of-week ``CAE_avg``. Week 1 is the
    baseline week (no separate pre-study CAE node). A missing current-week
    value is first recovered from the next week's observed
    ``CAE_avg_lastweek`` when available. Remaining gaps are imputed by pooled
    Gaussian-kernel regression on within-week ``M^Y`` summaries

        [mean Mon-Sat fourSC, mean Mon-Sat anticipated affect, week_frac].

    The completed raw trajectory is then written back consistently:

        reward in week w                 = C_{i,w}
        CAE_avg_lastweek in week 1       = C_{i,1}   (baseline week)
        CAE_avg_lastweek in week w >= 2  = C_{i,w-1}.

    The two columns are normalized separately using their original saved
    ``df_fit`` shift/scale parameters. Predictors retain their existing scales.
    The lagged column is the state feature used by fitted-Q (via
    ``est_prior._user_weekly_tensors`` / ``_phi_action``).
    """
    required = {
        "CAE_avg",
        "CAE_avg_lastweek",
        "CAE_avg_norm",
        "CAE_avg_lastweek_norm",
        "4hour_step_norm",
        "anticipated_affect_norm",
    }
    missing_cols = sorted(required.difference(df_fit.columns))
    if missing_cols:
        raise ValueError(
            "Unified CAE imputation requires columns: " + ", ".join(missing_cols)
        )
    if not STD_PARAMS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing standardization parameters required for CAE: {STD_PARAMS_PATH}"
        )
    with open(STD_PARAMS_PATH, encoding="utf-8") as f:
        std = json.load(f)
    reward_shift = float(std["CAE_avg_shift"])
    reward_scale = float(std["CAE_avg_scale"])
    lag_shift = float(std["CAE_avg_lastweek_shift"])
    lag_scale = float(std["CAE_avg_lastweek_scale"])

    df_out = df_fit.copy()
    trajectories: List[np.ndarray] = []
    week_index_blocks: List[np.ndarray] = []
    study_nodes: List[Dict[str, Any]] = []
    n_recovered_from_lag = 0
    n_observed_disagreements = 0

    for _uid, dat in df_out.groupby("ParticipantIdentifier", sort=False):
        dat = dat.sort_values(["Date", "DecisionTime"])
        n_w = len(dat) // K_SLOTS_WEEK
        if n_w == 0:
            continue
        idx = dat.index.to_numpy()[: n_w * K_SLOTS_WEEK].reshape(n_w, K_SLOTS_WEEK)
        first_idx = idx[:, 0]
        current = df_out.loc[first_idx, "CAE_avg"].to_numpy(dtype=float)
        lagged = df_out.loc[first_idx, "CAE_avg_lastweek"].to_numpy(dtype=float)

        # Study weeks only; week 1 is the baseline week (no pre-study C_0).
        trajectory = np.full(n_w, np.nan, dtype=float)
        for k in range(n_w):
            current_value = current[k]
            next_lag = lagged[k + 1] if k + 1 < n_w else np.nan
            if np.isfinite(current_value):
                trajectory[k] = current_value
                if np.isfinite(next_lag) and not np.isclose(
                    current_value, next_lag, atol=1e-10, rtol=0.0
                ):
                    n_observed_disagreements += 1
            elif np.isfinite(next_lag):
                trajectory[k] = next_lag
                n_recovered_from_lag += 1

        user_i = len(trajectories)
        trajectories.append(trajectory)
        week_index_blocks.append(idx)

        for k in range(n_w):
            week_dat = dat.loc[idx[k]].reset_index(drop=True)
            my = _week_my_impute_features(week_dat)
            week_frac = float(k + 1) / float(max(n_w, 1))
            study_nodes.append({
                "user_i": user_i,
                "trajectory_pos": k,
                "y": float(trajectory[k]) if np.isfinite(trajectory[k]) else np.nan,
                "x": np.array([my[0], my[1], week_frac], dtype=float),
            })

    empty_audit = {
        "method": "kernel_nadaraya_watson",
        "target": "canonical_raw_CAE_trajectory",
        "feature_names": list(KERNEL_IMPUTE_FEATURE_NAMES),
        "n_observed": 0,
        "n_missing": 0,
        "n_imputed": 0,
    }
    if not trajectories:
        return df_out, empty_audit

    if study_nodes:
        y_all = np.array([node["y"] for node in study_nodes], dtype=float)
        x_all = np.vstack([node["x"] for node in study_nodes])
        # Missing M^Y summaries are predictor gaps, not the target. Use the
        # observed pooled mean so every study-week CAE node remains imputable.
        for j in range(x_all.shape[1]):
            finite = np.isfinite(x_all[:, j])
            fill = float(np.mean(x_all[finite, j])) if finite.any() else 0.0
            x_all[~finite, j] = fill

        y_imputed, audit = impute_missing_cae_kernel(
            y_all,
            x_all,
            bandwidth=bandwidth,
            feature_names=KERNEL_IMPUTE_FEATURE_NAMES,
        )
        for node, y_hat in zip(study_nodes, y_imputed):
            trajectories[node["user_i"]][node["trajectory_pos"]] = float(y_hat)
    else:
        audit = dict(empty_audit)

    audit.update({
        "target": "canonical_raw_CAE_trajectory_study_weeks",
        "applied_to": ["reward_CAE_avg", "state_CAE_avg_lastweek"],
        "baseline_week": 1,
        "predictor_note": (
            "week-1 CAE is the baseline week; lags for later weeks are the "
            "previous study-week CAE; predictors are within-week M^Y summaries"
        ),
        "n_recovered_from_observed_next_lag": n_recovered_from_lag,
        "n_observed_lag_disagreements": n_observed_disagreements,
        "reward_shift": reward_shift,
        "reward_scale": reward_scale,
        "lag_shift": lag_shift,
        "lag_scale": lag_scale,
    })

    for traj, idx in zip(trajectories, week_index_blocks):
        n_w = idx.shape[0]
        for k in range(n_w):
            reward_raw = float(traj[k])
            # Week 1 (baseline): lag equals that week's own CAE.
            lag_raw = float(traj[k] if k == 0 else traj[k - 1])
            df_out.loc[idx[k], "CAE_avg_lastweek"] = lag_raw
            df_out.loc[idx[k], "CAE_avg_lastweek_norm"] = (
                lag_raw - lag_shift
            ) / lag_scale
            df_out.loc[idx[k], "CAE_avg"] = reward_raw
            df_out.loc[idx[k], "CAE_avg_norm"] = (
                reward_raw - reward_shift
            ) / reward_scale

    return df_out, audit


# ──────────────────────────────────────────────────────────────────
# Step 1 (data) - per-user, per-slot fitted-Q tensors
# ──────────────────────────────────────────────────────────────────
def precompute_user_qh_tensors(
    df_fit: pd.DataFrame,
    *,
    bandwidth: Optional[float] = KERNEL_CAE_BANDWIDTH,
) -> Tuple[List[Dict[str, Any]], List[int], Dict[str, Any]]:
    """Precompute per-user fitted-Q feature tensors for the infinite-horizon fit.

    For every participant with at least three complete weeks we build the
    following, with ``n_train = n_w - 2``:

      * Week 1 (index 0): baseline week only. Its CAE is the lag feature for
        week 2; week 1 is **not** a training week (no valid previous-week CAE).
      * Weeks 2 .. ``n_w-1``: training weeks.
      * Week ``n_w``: **not** a training week; only Monday-morning ``s_{1,1}``
        is retained as the terminal bootstrap successor for week ``n_w-1``.

      * ``phi_obs[j, h]``      phi(s, a_obs) for training week index
                               ``k = j + 1`` (calendar week ``j + 2``).
      * ``phi_a0[j, h]``,
        ``phi_a1[j, h]``       phi(s, a) for a in {0, 1}.
      * ``phi_next_a0[j]``,
        ``phi_next_a1[j]``     phi(s_{1,1}^{k+1}, a) - next-week Monday AM,
                               including the final week's Monday AM for the
                               last training week. Every training week has a
                               successor (``terminal_has_successor`` is all True).
      * ``R_raw[j]``           weekly ``CAE_avg_norm`` for that training week.

    Returns ``(feats_list, user_ids, audit)``. ``audit`` records how many users
    were used vs. skipped for having fewer than three complete weeks.
    """
    df_fit, feature_audit = kernel_impute_cae_trajectory(
        df_fit,
        bandwidth=bandwidth,
    )

    feats_list: List[Dict[str, Any]] = []
    user_ids: List[int] = []
    n_skipped = 0
    skipped_ids: List[int] = []

    for uid, dat in df_fit.groupby("ParticipantIdentifier", sort=False):
        dat = dat.sort_values(["Date", "DecisionTime"]).reset_index(drop=True)
        td = _user_weekly_tensors(dat)
        n_w = int(td["n_w"])
        # Need >=3 weeks: baseline (week 1) + >=1 train week + bootstrap week.
        if n_w < 3:
            n_skipped += 1
            skipped_ids.append(int(uid))
            continue

        p_phi = _strip_unused_phi_columns(_phi_action(td, 0, 0, 0)).size
        # Training uses weeks 2..n_w-1 (0-based indices 1..n_w-2).
        train_start = 1
        n_train = n_w - 2
        phi_obs = np.zeros((n_train, N_RL_SLOTS_WEEK, p_phi))
        phi_a0 = np.zeros((n_train, N_RL_SLOTS_WEEK, p_phi))
        phi_a1 = np.zeros((n_train, N_RL_SLOTS_WEEK, p_phi))
        for j in range(n_train):
            k = train_start + j
            for idx in range(N_RL_SLOTS_WEEK):
                phi_obs[j, idx] = _strip_unused_phi_columns(
                    _phi_action(td, k, idx, int(td["A_slot"][k, idx]))
                )
                phi_a0[j, idx] = _strip_unused_phi_columns(
                    _phi_action(td, k, idx, 0)
                )
                phi_a1[j, idx] = _strip_unused_phi_columns(
                    _phi_action(td, k, idx, 1)
                )

        phi_next_a0 = np.zeros((n_train, p_phi))
        phi_next_a1 = np.zeros((n_train, p_phi))
        terminal_has_successor = np.ones(n_train, dtype=bool)
        for j in range(n_train):
            k = train_start + j
            # Successor is next study week's Monday AM (week n_w for last train).
            phi_next_a0[j] = _strip_unused_phi_columns(
                _phi_action(td, k + 1, 0, 0)
            )
            phi_next_a1[j] = _strip_unused_phi_columns(
                _phi_action(td, k + 1, 0, 1)
            )

        raw = dat["CAE_avg_norm"].to_numpy(dtype=float)[: n_w * K_SLOTS_WEEK]
        raw_week = raw.reshape(n_w, K_SLOTS_WEEK)[:, 0]
        feats = {
            "phi_obs": phi_obs,
            "phi_a0": phi_a0,
            "phi_a1": phi_a1,
            "phi_next_a0": phi_next_a0,
            "phi_next_a1": phi_next_a1,
            "terminal_has_successor": terminal_has_successor,
            "n_train": n_train,
            "n_weeks_available": n_w,
            "train_week_start": train_start + 1,  # 1-based calendar week
            "train_week_end": n_w - 1,
            "p_phi": p_phi,
            "R_raw": raw_week[train_start : train_start + n_train],
        }

        feats_list.append(feats)
        user_ids.append(int(uid))

    audit = {
        "n_users_used": len(user_ids),
        "n_users_skipped_lt3weeks": n_skipped,
        "n_users_skipped_lt2weeks": n_skipped,
        "n_users_skipped_no_complete_weeks": n_skipped,
        "skipped_user_ids": skipped_ids,
        "max_n_train": max((int(f["n_train"]) for f in feats_list), default=0),
        "max_n_weeks_available": max(
            (int(f["n_weeks_available"]) for f in feats_list), default=0
        ),
        "feature_audit": feature_audit,
    }
    return feats_list, user_ids, audit


# ──────────────────────────────────────────────────────────────────
# Kernel imputation helper (used by the unified CAE trajectory)
# ──────────────────────────────────────────────────────────────────
def _median_pairwise_bandwidth(x: np.ndarray) -> float:
    """Median heuristic bandwidth on covariates in their supplied scales."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n < 2:
        return 1.0
    diffs = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=2))
    upper = dist[np.triu_indices(n, k=1)]
    pos = upper[upper > 0]
    if pos.size == 0:
        return 1.0
    return float(np.median(pos))


def impute_missing_cae_kernel(
    r_raw: np.ndarray,
    impute_x: np.ndarray,
    *,
    bandwidth: Optional[float] = KERNEL_CAE_BANDWIDTH,
    feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Nadaraya-Watson kernel imputation for missing weekly CAE values.

    Donor rows are all rows with an observed CAE target. Each missing row is
    imputed as a Gaussian-kernel weighted average of donor CAE values, with
    weights based on supplied covariates in their existing ``df_fit`` scales.
    No additional predictor standardization is performed here.

    Observed values are left unchanged.
    """
    r_raw = np.asarray(r_raw, dtype=float).ravel()
    x = np.asarray(impute_x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = r_raw.size
    if x.shape[0] != n:
        raise ValueError("r_raw and impute_x must have the same number of rows.")

    observed = np.isfinite(r_raw) & np.all(np.isfinite(x), axis=1)
    missing = ~observed & np.all(np.isfinite(x), axis=1)
    r_out = r_raw.copy()

    audit: Dict[str, Any] = {
        "method": "kernel_nadaraya_watson",
        "feature_names": list(feature_names or KERNEL_IMPUTE_FEATURE_NAMES),
        "n_observed": int(observed.sum()),
        "n_missing": int(missing.sum()),
        "n_imputed": 0,
        "bandwidth": None,
        "fallback_global_mean": 0,
        "predictor_scaling": "existing_df_fit_scales",
    }
    if missing.sum() == 0 or observed.sum() == 0:
        return r_out, audit

    x_obs = x[observed]
    y_obs = r_raw[observed]

    h = float(bandwidth) if bandwidth is not None else _median_pairwise_bandwidth(x_obs)
    h = max(h, 1e-6)
    audit["bandwidth"] = h

    global_mean = float(np.mean(y_obs))
    missing_idx = np.flatnonzero(missing)
    imputed_vals: List[float] = []

    for i in missing_idx:
        diff = x[i] - x_obs
        dist2 = np.sum(diff * diff, axis=1)
        w = np.exp(-0.5 * dist2 / (h * h))
        w_sum = float(w.sum())
        if w_sum <= 1e-12:
            r_out[i] = global_mean
            audit["fallback_global_mean"] += 1
        else:
            r_out[i] = float(np.dot(w, y_obs) / w_sum)
        imputed_vals.append(float(r_out[i]))

    audit["n_imputed"] = len(imputed_vals)
    if imputed_vals:
        audit["imputed_mean"] = float(np.mean(imputed_vals))
        audit["imputed_std"] = float(np.std(imputed_vals, ddof=0))
        audit["observed_mean"] = float(np.mean(y_obs))
    return r_out, audit


# ──────────────────────────────────────────────────────────────────
# Step 1 (fit) - pooled finite-horizon backward-induction fitted-Q
# ──────────────────────────────────────────────────────────────────
def _stack_designs(
    feats_list: List[Dict[str, Any]],
    *,
    r_terminal: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Stack per-user tensors into pooled per-slot design matrices.

    All slot designs share the same row ordering (one row per (user, train-week)
    in ``feats_list`` order), so targets built from one slot align with the
    observed design of another.
    """
    H = N_RL_SLOTS_WEEK
    # Observed design per slot h: X_obs[h] has one row per (user, train-week).
    x_obs = [np.vstack([f["phi_obs"][:, h, :] for f in feats_list]) for h in range(H)]
    # Successor within-week (both actions) at slot h+1, for h = 0..H-2.
    x_next0 = [np.vstack([f["phi_a0"][:, h + 1, :] for f in feats_list]) for h in range(H - 1)]
    x_next1 = [np.vstack([f["phi_a1"][:, h + 1, :] for f in feats_list]) for h in range(H - 1)]
    # Terminal successor = next-week start (both actions), plus raw reward.
    x_term0 = np.vstack([f["phi_next_a0"] for f in feats_list])
    x_term1 = np.vstack([f["phi_next_a1"] for f in feats_list])
    terminal_has_successor = np.concatenate([
        np.asarray(f["terminal_has_successor"], dtype=bool) for f in feats_list
    ])
    if r_terminal is None:
        r_terminal = np.concatenate([np.asarray(f["R_raw"], dtype=float) for f in feats_list])
    return {
        "x_obs": x_obs,
        "x_next0": x_next0,
        "x_next1": x_next1,
        "x_term0": x_term0,
        "x_term1": x_term1,
        "terminal_has_successor": terminal_has_successor,
        "r_raw": np.asarray(r_terminal, dtype=float),
    }


def _horizon_target(
    h: int,
    beta: List[np.ndarray],
    designs: Dict[str, Any],
    gamma_dt: np.ndarray,
) -> np.ndarray:
    """Bootstrap target for slot ``h`` (zero-based ``rl_idx``) given current beta."""
    H = N_RL_SLOTS_WEEK
    if h == H - 1:                              # terminal slot (d=6, t=2)
        v_start_next = np.maximum(
            designs["x_term0"] @ beta[0],
            designs["x_term1"] @ beta[0],
        )
        has_successor = np.asarray(designs["terminal_has_successor"], dtype=float)
        return designs["r_raw"] + gamma_dt[5, 1] * has_successor * v_start_next
    d, t = h // 2, h % 2
    q0 = designs["x_next0"][h] @ beta[h + 1]
    q1 = designs["x_next1"][h] @ beta[h + 1]
    return gamma_dt[d, t] * np.maximum(q0, q1)


def _run_backward_induction(
    designs: Dict[str, Any],
    *,
    gamma_dt: np.ndarray,
    n_sweeps: int,
    p: int,
    convergence_tol: Optional[float] = None,
    min_convergence_sweeps: int = FITTED_Q_MIN_CONVERGENCE_SWEEPS,
    early_stop_on_convergence: bool = False,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], bool, int]:
    """Run pooled backward-induction FQI; return betas and sweep diagnostics."""
    H = N_RL_SLOTS_WEEK
    beta: List[np.ndarray] = [np.zeros(p) for _ in range(H)]
    convergence_trace: List[Dict[str, Any]] = []
    stopped_by_convergence = False
    n_sweeps_run = 0

    for sweep in range(n_sweeps):
        beta_old = [b.copy() for b in beta]
        for h in range(H - 1, -1, -1):
            y = _horizon_target(h, beta, designs, gamma_dt)
            theta, _ = _ridge_fit(designs["x_obs"][h], y, alpha=RIDGE_ALPHA_RL)
            if theta is not None:
                beta[h] = theta

        n_sweeps_run = sweep + 1
        change_diag = _q_prediction_change_diagnostics(beta_old, beta, designs)
        change_diag["sweep"] = int(n_sweeps_run)
        convergence_trace.append(change_diag)

        max_rel_q_change = change_diag.get("max_rel_q_change")
        if (
            early_stop_on_convergence
            and convergence_tol is not None
            and n_sweeps_run >= min_convergence_sweeps
            and max_rel_q_change is not None
            and max_rel_q_change < convergence_tol
        ):
            stopped_by_convergence = True
            break

    return beta, convergence_trace, stopped_by_convergence, n_sweeps_run


def bootstrap_coefficient_intervals(
    feats_list: List[Dict[str, Any]],
    *,
    gamma_bar: float,
    n_sweeps: int,
    n_bootstrap: int = N_BOOTSTRAP,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Participant-level cluster bootstrap percentile CIs for ``beta_h``.

    Users (with their full within-user training tensors) are resampled with
    replacement. CAE imputation is held fixed at the values already encoded in
    ``feats_list``.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must lie in (0, 1).")
    n_users = len(feats_list)
    if n_users < 1:
        raise ValueError("feats_list is empty.")

    H = N_RL_SLOTS_WEEK
    p = int(feats_list[0]["p_phi"])
    gamma_dt = gamma_dt_matrix(gamma_bar)
    rng = np.random.default_rng(seed)
    beta_boot = np.full((n_bootstrap, H, p), np.nan, dtype=float)
    n_failed = 0

    for b in range(n_bootstrap):
        draw = rng.integers(0, n_users, size=n_users)
        boot_feats = [feats_list[i] for i in draw]
        designs = _stack_designs(boot_feats)
        try:
            beta, _, _, _ = _run_backward_induction(
                designs,
                gamma_dt=gamma_dt,
                n_sweeps=n_sweeps,
                p=p,
                early_stop_on_convergence=False,
            )
            beta_boot[b] = np.stack(beta, axis=0)
        except Exception:
            n_failed += 1

    alpha = 1.0 - float(ci_level)
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    ci_low = np.nanpercentile(beta_boot, lower_q, axis=0)
    ci_high = np.nanpercentile(beta_boot, upper_q, axis=0)
    boot_se = np.nanstd(beta_boot, axis=0, ddof=1)

    return {
        "n_bootstrap": int(n_bootstrap),
        "n_bootstrap_failed": int(n_failed),
        "bootstrap_seed": int(seed),
        "bootstrap_ci_level": float(ci_level),
        "bootstrap_unit": "participant",
        "beta_ci_low_by_horizon": ci_low,
        "beta_ci_high_by_horizon": ci_high,
        "beta_boot_se_by_horizon": boot_se,
    }


def fit_pooled_finite_horizon_q(
    df_fit: pd.DataFrame,
    *,
    gamma_bar: float = GAMMA_BAR_DEFAULT,
    n_sweeps: Optional[int] = N_BACKWARD_SWEEPS,
    bandwidth: Optional[float] = KERNEL_CAE_BANDWIDTH,
    truncation_tol: Optional[float] = FITTED_Q_TRUNCATION_TOL,
    convergence_tol: Optional[float] = FITTED_Q_CONVERGENCE_TOL,
    min_convergence_sweeps: int = FITTED_Q_MIN_CONVERGENCE_SWEEPS,
    early_stop_on_convergence: bool = FITTED_Q_EARLY_STOP_ON_CONVERGENCE,
    use_truncation_bound_for_max_sweeps: bool = FITTED_Q_USE_TRUNCATION_BOUND_FOR_MAX_SWEEPS,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_ci_level: float = BOOTSTRAP_CI_LEVEL,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    feats_list: Optional[List[Dict[str, Any]]] = None,
    user_ids: Optional[List[int]] = None,
    audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit the pooled infinite-horizon fitted-Q value function.

    Completes one canonical raw CAE trajectory per user (kernel imputation on
    within-week ``M^Y`` summaries plus ``week_frac``) and derives both
    ``CAE_avg_norm`` rewards and lagged ``CAE_avg_lastweek_norm`` state features
    from it. Participant-level bootstrap percentile CIs are attached for each
    per-horizon coefficient when ``n_bootstrap > 0``.

    Returns a dict with per-horizon coefficients and diagnostics; see the module
    docstring for the algorithm.
    """
    if feats_list is None:
        feats_list, user_ids, audit = precompute_user_qh_tensors(
            df_fit,
            bandwidth=bandwidth,
        )
    else:
        user_ids = user_ids or []
        audit = audit or {}
    if not feats_list:
        raise ValueError("No users with >= 3 complete weeks available for fitted-Q.")

    H = N_RL_SLOTS_WEEK
    p = int(feats_list[0]["p_phi"])
    gamma_dt = gamma_dt_matrix(gamma_bar)
    designs = _stack_designs(feats_list)
    n_missing_reward = int(np.sum(~np.isfinite(designs["r_raw"])))

    # Data-supported maximum propagation horizon: one backward sweep can add
    # one additional week of bootstrapped future reward because the only
    # cross-week link is terminal week-k -> start of week-(k+1).
    max_n_train = int(audit["max_n_train"])
    data_horizon_sweeps = max(max_n_train, 1)

    # The discounted-return bound is a diagnostic by default. It is based on
    # the weekly discount because one sweep corresponds to one extra week of
    # future value, not one micro-decision slot.
    reward_bound_empirical = _empirical_reward_bound(designs["r_raw"])
    truncation_bound_sweeps = _discounted_return_bound_sweeps(
        gamma=float(gamma_bar),
        reward_bound=reward_bound_empirical,
        tolerance=truncation_tol,
    )

    if n_sweeps is None:
        n_sweeps_planned = data_horizon_sweeps
        sweep_selection_rule = "data_horizon"
        if (
            use_truncation_bound_for_max_sweeps
            and truncation_bound_sweeps is not None
        ):
            n_sweeps_planned = max(n_sweeps_planned, truncation_bound_sweeps)
            sweep_selection_rule = "max_data_horizon_truncation_bound"
    else:
        n_sweeps_planned = int(n_sweeps)
        sweep_selection_rule = "user_supplied"

    if n_sweeps_planned < 1:
        raise ValueError("n_sweeps must be at least 1.")
    if min_convergence_sweeps < 1:
        raise ValueError("min_convergence_sweeps must be at least 1.")
    if convergence_tol is not None and convergence_tol <= 0.0:
        raise ValueError("convergence_tol must be positive or None.")

    beta, convergence_trace, stopped_by_convergence, n_sweeps_run = (
        _run_backward_induction(
            designs,
            gamma_dt=gamma_dt,
            n_sweeps=n_sweeps_planned,
            p=p,
            convergence_tol=convergence_tol,
            min_convergence_sweeps=min_convergence_sweeps,
            early_stop_on_convergence=early_stop_on_convergence,
        )
    )

    final_change_diag = convergence_trace[-1] if convergence_trace else {}

    # Final diagnostics: per-horizon residual sigma2 and usable-row counts.
    sigma2_by_h = np.zeros(H)
    n_obs_by_h = np.zeros(H, dtype=int)
    n_dropped_by_h = np.zeros(H, dtype=int)
    for h in range(H):
        x_h = designs["x_obs"][h]
        y = _horizon_target(h, beta, designs, gamma_dt)
        ok = np.isfinite(y) & np.all(np.isfinite(x_h), axis=1)
        n_obs_by_h[h] = int(ok.sum())
        n_dropped_by_h[h] = int((~ok).sum())
        resid = y[ok] - x_h[ok] @ beta[h]
        sigma2_by_h[h] = (
            max(float(np.var(resid, ddof=1)), MIN_SIGMA2) if resid.size > 1 else 1.0
        )

    n_missing_reward = int(np.sum(~np.isfinite(designs["r_raw"])))

    out: Dict[str, Any] = {
        "gamma_bar": float(gamma_bar),
        "gamma_dt": gamma_dt,
        "gamma_dt_scalar": float(gamma_dt[0, 0]),
        "gamma_terminal": float(gamma_dt[5, 1]),
        "n_backward_sweeps": int(n_sweeps_run),
        "n_backward_sweeps_planned": int(n_sweeps_planned),
        "sweep_selection_rule": sweep_selection_rule,
        "n_sweeps_data_horizon": int(data_horizon_sweeps),
        "n_sweeps_truncation_bound": (
            int(truncation_bound_sweeps)
            if truncation_bound_sweeps is not None
            else None
        ),
        "truncation_tol": truncation_tol,
        "reward_bound_empirical": float(reward_bound_empirical),
        "use_truncation_bound_for_max_sweeps": bool(
            use_truncation_bound_for_max_sweeps
        ),
        "convergence_tol": convergence_tol,
        "min_convergence_sweeps": int(min_convergence_sweeps),
        "early_stop_on_convergence": bool(early_stop_on_convergence),
        "stopped_by_convergence": bool(stopped_by_convergence),
        "final_max_abs_q_change": final_change_diag.get("max_abs_q_change"),
        "final_max_rel_q_change": final_change_diag.get("max_rel_q_change"),
        "final_max_abs_beta_change": final_change_diag.get("max_abs_beta_change"),
        "final_max_rel_beta_change": final_change_diag.get("max_rel_beta_change"),
        "convergence_trace": convergence_trace,
        "ridge_alpha": float(RIDGE_ALPHA_RL),
        "horizon": int(H),
        "n_features": int(p),
        "feature_names": _feature_names_for(p),
        "beta_by_horizon": np.stack(beta, axis=0),           # (H, p)
        "sigma2_by_horizon": sigma2_by_h,                    # (H,)
        "n_obs_by_horizon": n_obs_by_h,                      # (H,)
        "n_dropped_by_horizon": n_dropped_by_h,              # (H,)
        "n_users": int(audit.get("n_users_used", len(user_ids))),
        "n_users_skipped_lt2weeks": int(audit.get("n_users_skipped_lt2weeks", 0)),
        "n_users_skipped_no_complete_weeks": int(
            audit.get("n_users_skipped_no_complete_weeks", 0)
        ),
        "n_terminal_no_successor": int(
            np.sum(~np.asarray(designs["terminal_has_successor"], dtype=bool))
        ),
        "n_missing_weekly_reward": n_missing_reward,
        "cae_imputation_strategy": "unified_raw_trajectory_kernel",
        "feature_audit": audit.get("feature_audit", {}),
        "user_ids": [int(u) for u in user_ids],
    }

    if n_bootstrap > 0:
        boot = bootstrap_coefficient_intervals(
            feats_list,
            gamma_bar=gamma_bar,
            n_sweeps=n_sweeps_planned,
            n_bootstrap=n_bootstrap,
            ci_level=bootstrap_ci_level,
            seed=bootstrap_seed,
        )
        out.update(boot)

    return out


# ──────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────
def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_fitted_q(model: Dict[str, Any], path: Path = FITTED_Q_JSON_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(model), f, indent=2, allow_nan=False)
    return path


def build_fitted_q_table(model: Dict[str, Any]) -> pd.DataFrame:
    """Tidy per-(horizon, feature) coefficient table, with bootstrap CIs if present."""
    beta = np.asarray(model["beta_by_horizon"], dtype=float)   # (H, p)
    names = list(model["feature_names"])
    sigma2 = np.asarray(model["sigma2_by_horizon"], dtype=float)
    n_obs = np.asarray(model["n_obs_by_horizon"], dtype=int)
    H, p = beta.shape
    ci_low = model.get("beta_ci_low_by_horizon")
    ci_high = model.get("beta_ci_high_by_horizon")
    boot_se = model.get("beta_boot_se_by_horizon")
    if ci_low is not None:
        ci_low = np.asarray(ci_low, dtype=float)
        ci_high = np.asarray(ci_high, dtype=float)
    if boot_se is not None:
        boot_se = np.asarray(boot_se, dtype=float)

    rows: List[Dict[str, Any]] = []
    for h in range(H):
        d = h // 2 + 1                     # 1-based day (Mon..Sat)
        t = h % 2 + 1                      # 1-based slot (AM=1, PM=2)
        for i in range(p):
            row: Dict[str, Any] = {
                "horizon": h + 1,
                "day": d,
                "slot": t,
                "index": i,
                "feature": names[i] if i < len(names) else f"beta_coef_{i}",
                "coefficient": float(beta[h, i]),
                "residual_sigma2": float(sigma2[h]),
                "n_obs": int(n_obs[h]),
            }
            if ci_low is not None:
                row["ci_low"] = float(ci_low[h, i])
                row["ci_high"] = float(ci_high[h, i])
                row["ci_level"] = float(model.get("bootstrap_ci_level", BOOTSTRAP_CI_LEVEL))
            if boot_se is not None:
                row["boot_se"] = float(boot_se[h, i])
            rows.append(row)
    return pd.DataFrame(rows)


def _coefficient_feature_groups(feature_names: List[str]) -> Dict[str, List[int]]:
    """Partition coefficients by substantive variable family."""
    groups: Dict[str, List[int]] = {
        "baseline": [],
        "M_Y_fourSC": [],
        "M_Y_anticipated_affect": [],
        "M_E_pageview": [],
        "M_E_fitbit_wear": [],
        "M_E_survey_complete": [],
        "interactions": [],
    }
    for i, name in enumerate(feature_names):
        if name.startswith("M_Y_"):
            if "fourSC" in name:
                groups["M_Y_fourSC"].append(i)
            elif "anticipated_affect" in name:
                groups["M_Y_anticipated_affect"].append(i)
            else:
                raise ValueError(f"Unrecognized M^Y feature: {name}")
        elif name.startswith("M_E_"):
            if "pageview" in name:
                groups["M_E_pageview"].append(i)
            elif "fitbit_wear" in name:
                groups["M_E_fitbit_wear"].append(i)
            elif "survey_complete" in name:
                groups["M_E_survey_complete"].append(i)
            else:
                raise ValueError(f"Unrecognized M^E feature: {name}")
        elif name == "A" or name.startswith("A*"):
            groups["interactions"].append(i)
        else:
            groups["baseline"].append(i)

    assigned = sorted(i for indices in groups.values() for i in indices)
    if assigned != list(range(len(feature_names))):
        raise ValueError("Coefficient feature groups are not exhaustive and disjoint.")
    return groups


def save_coefficient_plots(
    model: Dict[str, Any],
    *,
    paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, Path]:
    """Save faceted dot-and-CI plots of fitted-Q coefficients by feature family.

    Each feature gets its own horizontal panel: x-axis is within-week horizon,
    y-axis is the ridge coefficient, and each horizon is a dot with vertical
    bootstrap CI whiskers when available.
    """
    if paths is None:
        paths = FITTED_Q_PLOT_PATHS

    beta = np.asarray(model["beta_by_horizon"], dtype=float)
    names = list(model["feature_names"])
    H, p = beta.shape
    if p != len(names):
        raise ValueError("beta_by_horizon width does not match feature_names.")

    groups = _coefficient_feature_groups(names)
    titles = {
        "baseline": "Fitted-Q coefficients: baseline features",
        "M_Y_fourSC": r"Fitted-Q coefficients: $M^Y$ fourSC",
        "M_Y_anticipated_affect": (
            r"Fitted-Q coefficients: $M^Y$ anticipated affect"
        ),
        "M_E_pageview": r"Fitted-Q coefficients: $M^E$ pageview",
        "M_E_fitbit_wear": r"Fitted-Q coefficients: $M^E$ Fitbit wear",
        "M_E_survey_complete": (
            r"Fitted-Q coefficients: $M^E$ survey completion"
        ),
        "interactions": "Fitted-Q coefficients: action interactions",
    }
    footer = (
        "Source: pooled fitted-Q; "
        f"gamma_bar={model['gamma_bar']}; training weeks 2..(W-1)."
    )
    if model.get("n_bootstrap"):
        footer += (
            f" Error bars: {100 * float(model.get('bootstrap_ci_level', 0.95)):.0f}% "
            f"participant bootstrap CI (B={model['n_bootstrap']})."
        )
    out: Dict[str, Path] = {}

    ci_low = model.get("beta_ci_low_by_horizon")
    ci_high = model.get("beta_ci_high_by_horizon")
    if ci_low is not None:
        ci_low = np.asarray(ci_low, dtype=float)
        ci_high = np.asarray(ci_high, dtype=float)

    for group, indices in groups.items():
        if not indices:
            continue
        output_path = paths[group]
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_variables = len(indices)
        fig_h = max(4.0, 0.55 * n_variables)
        fig, axes = plt.subplots(
            n_variables,
            1,
            figsize=(14.0, fig_h),
            sharex=True,
            squeeze=False,
        )
        x = np.arange(H, dtype=float)
        x_labels = [f"h={h + 1}" for h in range(H)]

        for j, feature_idx in enumerate(indices):
            ax = axes[j, 0]
            y = beta[:, feature_idx]
            if ci_low is not None:
                yerr = np.vstack([
                    y - ci_low[:, feature_idx],
                    ci_high[:, feature_idx] - y,
                ])
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt="o",
                    color="#1f77b4",
                    markersize=3.2,
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    elinewidth=1.2,
                    capsize=3.5,
                    capthick=1.2,
                    zorder=3,
                )
            else:
                ax.plot(
                    x,
                    y,
                    "o",
                    color="#1f77b4",
                    markersize=3.2,
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    zorder=3,
                )

            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.65, zorder=1)
            ax.set_ylabel(
                names[feature_idx],
                rotation=0,
                ha="right",
                va="center",
                fontsize=8,
            )
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="y", labelsize=8)
            if j < n_variables - 1:
                ax.tick_params(labelbottom=False)

        axes[-1, 0].set_xticks(x)
        axes[-1, 0].set_xticklabels(x_labels)
        axes[-1, 0].set_xlabel("Within-week horizon")
        fig.suptitle(titles[group], fontsize=13, y=0.995)
        fig.text(0.34, 0.01, footer, fontsize=9, ha="left")
        fig.subplots_adjust(left=0.34, right=0.98, top=0.96, bottom=0.08, hspace=0.45)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        out[group] = output_path

    return out


def _format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
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
        "| " + " | ".join(str(c) for c in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| " + " | ".join(_format_value(row[col]) for col in columns) + " |"
        )
    return "\n".join(lines)


def save_fitted_q_tables(
    model: Dict[str, Any],
    *,
    summary_path: Path = FITTED_Q_SUMMARY_PATH,
    markdown_path: Path = FITTED_Q_TABLE_PATH,
) -> Dict[str, Path]:
    """Write the tidy CSV and a wide coefficient-by-horizon markdown table."""
    table = build_fitted_q_table(model)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(summary_path, index=False)

    # Wide view: one row per feature, one column per horizon (coefficient).
    wide = table.pivot(index="index", columns="horizon", values="coefficient")
    names = list(model["feature_names"])
    wide.insert(0, "feature", [names[i] if i < len(names) else f"beta_coef_{i}"
                               for i in wide.index])
    wide = wide.rename(columns={h: f"h{h}" for h in wide.columns if isinstance(h, (int, np.integer))})
    wide = wide.reset_index(drop=True)

    sigma2 = np.asarray(model["sigma2_by_horizon"], dtype=float)
    n_obs = np.asarray(model["n_obs_by_horizon"], dtype=int)
    n_drop = np.asarray(model["n_dropped_by_horizon"], dtype=int)
    diag = pd.DataFrame({
        "horizon": list(range(1, model["horizon"] + 1)),
        "day": [h // 2 + 1 for h in range(model["horizon"])],
        "slot": [h % 2 + 1 for h in range(model["horizon"])],
        "residual_sigma2": sigma2,
        "n_obs": n_obs,
        "n_dropped": n_drop,
    })

    fa = model.get("feature_audit") or {}
    md = [
        "# Pooled Finite-Horizon Fitted-Q",
        "",
        f"- Discount: `gamma_bar = {model['gamma_bar']}` "
        f"(per-slot `gamma = {model['gamma_dt_scalar']:.6f}`, "
        f"terminal `gamma_{{6,2}} = {model['gamma_terminal']:.6f}`)",
        f"- Backward sweeps run: `{model['n_backward_sweeps']}` "
        f"(planned `{model.get('n_backward_sweeps_planned', model['n_backward_sweeps'])}`, "
        f"rule `{model.get('sweep_selection_rule', 'n/a')}`)",
        f"- Data-horizon sweep cap: `{model.get('n_sweeps_data_horizon', 'n/a')}`; "
        f"truncation-bound sweeps: `{model.get('n_sweeps_truncation_bound', 'n/a')}`; "
        f"empirical reward bound: `{_format_value(model.get('reward_bound_empirical'))}`; "
        f"truncation tolerance: `{model.get('truncation_tol', 'n/a')}`",
        f"- Final fitted-Q change: max relative "
        f"`{_format_value(model.get('final_max_rel_q_change'))}`, max absolute "
        f"`{_format_value(model.get('final_max_abs_q_change'))}`; "
        f"stopped by convergence: `{model.get('stopped_by_convergence', False)}`",
        f"- Ridge alpha: `{model['ridge_alpha']}`",
        f"- Users: `{model['n_users']}` used, "
        f"`{model.get('n_users_skipped_no_complete_weeks', 0)}` skipped "
        "(no complete weeks)",
        f"- Week roles: week 1 = baseline lag only; last week = Monday-AM "
        f"bootstrap only; training weeks exclude both. "
        f"Terminal rows without successor: "
        f"`{model.get('n_terminal_no_successor', 0)}`",
        f"- CAE imputation strategy: "
        f"`{model.get('cae_imputation_strategy', 'unified_raw_trajectory_kernel')}`",
        f"- Missing weekly rewards excluded from terminal fit: "
        f"`{model['n_missing_weekly_reward']}`",
        f"- Unified raw-CAE kernel details: bandwidth={fa.get('bandwidth', 'n/a')}, "
        f"imputed canonical nodes={fa.get('n_imputed', 0)}",
    ]
    if model.get("n_bootstrap"):
        md.append(
            f"- Coefficient CIs: participant-level percentile bootstrap "
            f"(B=`{model['n_bootstrap']}`, "
            f"level=`{model.get('bootstrap_ci_level', BOOTSTRAP_CI_LEVEL)}`, "
            f"seed=`{model.get('bootstrap_seed', BOOTSTRAP_SEED)}`, "
            f"failed=`{model.get('n_bootstrap_failed', 0)}`). "
            "Point estimates hold CAE imputation fixed; see CSV columns "
            "`ci_low` / `ci_high` / `boot_se`."
        )
    if model.get("final_max_rel_q_change") is not None:
        print(f"    sweeps: run={model.get('n_backward_sweeps')}, "
              f"planned={model.get('n_backward_sweeps_planned')}, "
              f"data_cap={model.get('n_sweeps_data_horizon')}, "
              f"bound={model.get('n_sweeps_truncation_bound')}, "
              f"final_rel_Q_change={model.get('final_max_rel_q_change'):.6g}")
    md.extend([
        "",
        "Each horizon `h = 1..12` is a within-week decision slot "
        "(`day = h//2 rounded up`, `slot = AM/PM`). Coefficients below are the "
        "pooled fitted-Q `beta_h` for the action-augmented feature map "
        "`phi(s, a)` shared across weeks.",
        "",
        "## Per-horizon diagnostics",
        "",
        _dataframe_to_markdown(diag),
        "",
        "## Coefficients by horizon",
        "",
        _dataframe_to_markdown(wide),
        "",
    ])
    markdown_path.write_text("\n".join(md), encoding="utf-8")
    return {"summary": summary_path, "markdown": markdown_path}


def _print_model_summary(model: Dict[str, Any]) -> None:
    print(f"  strategy={model.get('cae_imputation_strategy')}, "
          f"missing/dropped terminal={model['n_missing_weekly_reward']}")
    if model.get("final_max_rel_q_change") is not None:
        print(f"    sweeps: run={model.get('n_backward_sweeps')}, "
              f"planned={model.get('n_backward_sweeps_planned')}, "
              f"data_cap={model.get('n_sweeps_data_horizon')}, "
              f"bound={model.get('n_sweeps_truncation_bound')}, "
              f"final_rel_Q_change={model.get('final_max_rel_q_change'):.6g}")
    fa = model.get("feature_audit") or {}
    print(f"    unified raw-CAE kernel: bandwidth={fa.get('bandwidth')}, "
          f"imputed_nodes={fa.get('n_imputed')}, "
          f"observed_mean={fa.get('observed_mean', float('nan')):.4f}, "
          f"imputed_mean={fa.get('imputed_mean', float('nan')):.4f}")
    for h in range(model["horizon"]):
        print(f"    h={h + 1:2d}: n_obs={int(model['n_obs_by_horizon'][h]):5d}, "
              f"sigma2={float(model['sigma2_by_horizon'][h]):.4f}")


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────
def main(
    gamma_bar: float = GAMMA_BAR_DEFAULT,
    *,
    bandwidth: Optional[float] = KERNEL_CAE_BANDWIDTH,
    truncation_tol: Optional[float] = FITTED_Q_TRUNCATION_TOL,
    convergence_tol: Optional[float] = FITTED_Q_CONVERGENCE_TOL,
    min_convergence_sweeps: int = FITTED_Q_MIN_CONVERGENCE_SWEEPS,
    early_stop_on_convergence: bool = FITTED_Q_EARLY_STOP_ON_CONVERGENCE,
    use_truncation_bound_for_max_sweeps: bool = FITTED_Q_USE_TRUNCATION_BOUND_FOR_MAX_SWEEPS,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_ci_level: float = BOOTSTRAP_CI_LEVEL,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    df_fit = load_df_fit()
    print(f"df_fit: {len(df_fit)} rows, "
          f"{df_fit['ParticipantIdentifier'].nunique()} participants")

    print(f"Fitting pooled fitted-Q (gamma_bar={gamma_bar}, "
          f"n_bootstrap={n_bootstrap}) ...")
    model = fit_pooled_finite_horizon_q(
        df_fit,
        gamma_bar=gamma_bar,
        bandwidth=bandwidth,
        truncation_tol=truncation_tol,
        convergence_tol=convergence_tol,
        min_convergence_sweeps=min_convergence_sweeps,
        early_stop_on_convergence=early_stop_on_convergence,
        use_truncation_bound_for_max_sweeps=use_truncation_bound_for_max_sweeps,
        n_bootstrap=n_bootstrap,
        bootstrap_ci_level=bootstrap_ci_level,
        bootstrap_seed=bootstrap_seed,
    )
    print(f"  users used: {model['n_users']} "
          f"(skipped {model.get('n_users_skipped_no_complete_weeks', 0)} "
          "with no complete weeks)")
    print(f"  horizon: {model['horizon']} slots, "
          f"{model['n_features']} features, "
          f"{model['n_backward_sweeps']} backward sweeps")
    print(f"  terminal rows with no successor bootstrap: "
          f"{model.get('n_terminal_no_successor', 0)}")
    print(f"  training weeks: 2..(W-1); week 1 = baseline lag only; "
          f"week W = Monday-AM bootstrap only")
    if model.get("n_bootstrap"):
        print(f"  bootstrap CIs: B={model['n_bootstrap']}, "
              f"level={model.get('bootstrap_ci_level')}, "
              f"failed={model.get('n_bootstrap_failed', 0)}")
    _print_model_summary(model)

    json_path = save_fitted_q(model, FITTED_Q_JSON_PATH)
    table_paths = save_fitted_q_tables(model)
    plot_paths = save_coefficient_plots(model)
    print(f"Saved fitted-Q model  -> {json_path}")
    print(f"Saved fitted-Q CSV    -> {table_paths['summary']}")
    print(f"Saved fitted-Q tables -> {table_paths['markdown']}")
    for group, path in plot_paths.items():
        print(f"Saved {group:12s} plot -> {path}")
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pooled infinite-horizon fitted-Q on df_fit.",
    )
    parser.add_argument(
        "--gamma-bar", type=float, default=GAMMA_BAR_DEFAULT,
        help="Weekly discount factor (default: 0.5).",
    )
    parser.add_argument(
        "--kernel-bandwidth", type=float, default=None,
        help="Gaussian kernel bandwidth for CAE imputation (default: median heuristic).",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=N_BOOTSTRAP,
        help=(
            "Participant-level bootstrap replicates for coefficient CIs "
            f"(default: {N_BOOTSTRAP}; 0 disables)."
        ),
    )
    parser.add_argument(
        "--bootstrap-ci-level", type=float, default=BOOTSTRAP_CI_LEVEL,
        help=f"Bootstrap percentile CI level (default: {BOOTSTRAP_CI_LEVEL}).",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=BOOTSTRAP_SEED,
        help=f"RNG seed for participant bootstrap (default: {BOOTSTRAP_SEED}).",
    )
    parser.add_argument(
        "--truncation-tol", type=float, default=FITTED_Q_TRUNCATION_TOL,
        help=(
            "Tolerance used to compute the discounted-return truncation-bound "
            "sweep diagnostic (default: 1e-3)."
        ),
    )
    parser.add_argument(
        "--convergence-tol", type=float, default=FITTED_Q_CONVERGENCE_TOL,
        help=(
            "Relative fitted-Q prediction-change tolerance for diagnostics and "
            "optional early stopping (default: 1e-4)."
        ),
    )
    parser.add_argument(
        "--min-convergence-sweeps",
        type=int,
        default=FITTED_Q_MIN_CONVERGENCE_SWEEPS,
        help="Minimum sweeps before optional convergence early stopping.",
    )
    parser.add_argument(
        "--early-stop-on-convergence",
        action="store_true",
        help=(
            "Stop before the data-horizon cap if the relative fitted-Q change "
            "drops below --convergence-tol."
        ),
    )
    parser.add_argument(
        "--use-truncation-bound-for-max-sweeps",
        action="store_true",
        help=(
            "Use max(data-horizon sweeps, discounted-return bound sweeps) as "
            "the automatic sweep cap. By default the bound is reported only."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        gamma_bar=args.gamma_bar,
        bandwidth=args.kernel_bandwidth,
        truncation_tol=args.truncation_tol,
        convergence_tol=args.convergence_tol,
        min_convergence_sweeps=args.min_convergence_sweeps,
        early_stop_on_convergence=args.early_stop_on_convergence,
        use_truncation_bound_for_max_sweeps=args.use_truncation_bound_for_max_sweeps,
        n_bootstrap=args.n_bootstrap,
        bootstrap_ci_level=args.bootstrap_ci_level,
        bootstrap_seed=args.bootstrap_seed,
    )
