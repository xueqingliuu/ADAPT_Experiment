# %%
"""
Joint state-space model for perceived utility E_w with weekly AR dynamics and
within-week outcomes (J_week, U1, U2, hourly PV, daily FW, daily PJ).

Likelihood for latent E_2,...,E_W is approximated by 1D quadrature (trapezoid
over the support of E) with prediction / filtering recursion as on the slides.
With ``e1_known``, E_1 is the fixed baseline (= week-1 ``perceived_utility_lastweek``);
E_2 is the state at end of week 1 / start of week 2. Otherwise a Gaussian prior
on E_1 is estimated.

Study weeks w = 1,...,12 index observation blocks; latent E_w is the state at
the start of week w (week-w outcomes are conditional on E_w). The AR transition
after week w yields E_{w+1}; with 12 weeks this produces E_1,...,E_12 in the
likelihood and a terminal E_13 at end of week 12 (predictive only, no week-13
data). On each ``df_fit`` row at study week w (1..12), the exported columns are
``perceived_utility`` = \\hat E_{w+1} (spans E_2,...,E_13; week 12 uses the
predictive terminal E_13) and ``perceived_utility_lastweek`` = \\hat E_w (spans
E_1,...,E_12); by construction ``perceived_utility_lastweek`` is the literal
weekly lag of ``perceived_utility``. ``pred_penalized_filtered_Ew`` in the JSON
export stores the same E_2,...,E_13 series (one entry per study week 1..12).

Observation models (conditional on latent E_w for that study week). Weekly survey outcomes:

  - J_week (``week_present``): Bernoulli, \\mathrm{logit}(p) = b_0 + b_1 E_w.
  - U1, U2 (``Exp-tool-1_norm``, ``Exp-tool-2_norm``; only if J_week = 1):
        U1 \\sim \\mathcal{N}(c_0 + c_1 E_w, \\sigma_{U1}^2),
        U2 \\sim \\mathcal{N}(d_0 + d_1 E_w, \\sigma_{U2}^2).

Within-week intensive measures:

  - PV (hourly ``HourlyPageviewCount_norm``):  Gaussian,
        \\mu = \\alpha_0 + \\alpha_1 E_w
             + \\alpha_{2,\\mathrm{we}}\\,\\mathrm{is\\_weekend}
             + \\alpha_{2,\\mathrm{dt}}\\,\\mathrm{dt}
             + \\alpha_{2,\\mathrm{rb}}\\,\\mathrm{recent\\_burden}
             + \\alpha_{\\mathrm{AR}}\\, z^{\\mathrm{PV,lag1}}
             + A(\\alpha_3 + \\alpha_4 E_w),
        where z^{\\mathrm{PV,lag1}} is ``hourly_pageview_count_lag1`` (same row);
        (is_weekend, dt, recent_burden) from ``is_weekend``, ``DecisionTime``,
        ``recent_burden_norm``; A = hourly ``WalkingSuggestion``; noise
        \\sigma_{\\mathrm{PV}}.

  - FW (daily outcome ``nextday_wearing``):  Bernoulli,
        \\eta = \\beta_0 + \\beta_1 E_w
             + \\beta_{2,\\mathrm{we}}\\,\\mathrm{is\\_weekend}
             + \\beta_{2,\\mathrm{rb}}\\,\\mathrm{recent\\_burden}
             + \\beta_{\\mathrm{AR}}\\, z^{\\mathrm{FW,lag}}
             + A_0(\\beta_3 + \\beta_4 E_w) + A_1(\\beta_5 + \\beta_6 E_w),
        where z^{\\mathrm{FW,lag}} is ``morning_wearing`` on the morning row;
        (is_weekend, recent_burden) from that row; A_0, A_1 from morning/afternoon
        ``WalkingSuggestion`` .

  - PJ (daily ``daily_present``):  Bernoulli,
        \\eta = \\theta_0 + \\theta_1 E_w
             + \\theta_{2,\\mathrm{we}}\\,\\mathrm{is\\_weekend}
             + \\theta_{2,\\mathrm{rb}}\\,\\mathrm{recent\\_burden}
             + \\theta_{\\mathrm{AR}}\\, z^{\\mathrm{PJ,lag}}
             + A_0(\\theta_3 + \\theta_4 E_w) + A_1(\\theta_5 + \\theta_6 E_w),
        where z^{\\mathrm{PJ,lag}} is ``daily_present_yesterday`` on the morning row;
        same (is_weekend, burden, A_0, A_1) construction as FW.

Lag covariates are read from ``df_fit`` (aligned per hour or per day); missing
values are treated as 0 in the linear predictor.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil
import logging
import os
import time
from pathlib import Path
from typing import Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
import json

logger = logging.getLogger(__name__)

# %%
# read data
PROJECT_ROOT = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPR-MRT-Testbed")
COMBINED_DIR = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/Xueqing")
WORK_DIR = PROJECT_ROOT / "env_para_vanilla"
WORK_DIR.mkdir(parents=True, exist_ok=True)

folder = COMBINED_DIR

df_fit = pd.read_csv(folder / "df_fit.csv")
# df_fit is already study weeks 1–12 (burn-in / week 0 / week 13 dropped upstream).


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bernoulli_logpmf(y, eta):
    return y * (-np.logaddexp(0.0, -eta)) + (1.0 - y) * (-np.logaddexp(0.0, eta))


def normal_pdf(x, mu, sigma):
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * sigma)


def normal_logpdf(y, mu, sigma):
    z = (y - mu) / sigma
    return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * z * z


def trapezoid_weights(grid):
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("Quadrature grid must be a one-dimensional array with at least 2 points.")
    if not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
        raise ValueError("Quadrature grid must be finite and strictly increasing.")
    w = np.empty_like(grid)
    w[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    w[0] = 0.5 * (grid[1] - grid[0])
    w[-1] = 0.5 * (grid[-1] - grid[-2])
    return w


def validate_quadrature_support(
    grid: np.ndarray,
    weights: np.ndarray,
    *,
    e1_known: Optional[float] = None,
) -> None:
    """Validate quadrature inputs and warn when fixed E1 is at the support edge."""
    grid = np.asarray(grid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if grid.ndim != 1 or weights.shape != grid.shape:
        raise ValueError("Quadrature grid and weights must be one-dimensional with equal length.")
    if grid.size < 2 or not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
        raise ValueError("Quadrature grid must contain at least 2 finite, increasing points.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Quadrature weights must be finite and strictly positive.")
    if e1_known is None:
        return

    e1 = float(e1_known)
    if not np.isfinite(e1) or e1 < grid[0] or e1 > grid[-1]:
        raise ValueError(
            f"Fixed E1={e1_known!r} must lie within [{grid[0]}, {grid[-1]}]."
        )
    edge_distance = min(e1 - grid[0], grid[-1] - e1)
    if edge_distance <= np.max(np.diff(grid)):
        warnings.warn(
            f"Fixed E1={e1:g} is at/within one grid cell of quadrature boundary "
            f"[{grid[0]:g}, {grid[-1]:g}]; transition mass may be truncated.",
            RuntimeWarning,
            stacklevel=2,
        )


def _normalize_log_density(
    log_density: np.ndarray,
    weights: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, float]:
    """Normalize a grid density in log space; return density and log retained mass."""
    log_density = np.asarray(log_density, dtype=float)
    log_norm = float(logsumexp(log_density + np.log(weights)))
    if not np.isfinite(log_norm):
        raise FloatingPointError(f"invalid {label} density normalization")
    density = np.exp(log_density - log_norm)
    if not np.all(np.isfinite(density)):
        raise FloatingPointError(f"non-finite normalized {label} density")
    return density, log_norm


def _predict_density_log(
    grid: np.ndarray,
    weights: np.ndarray,
    previous_density: np.ndarray,
    prev_block: dict,
    par: dict,
) -> tuple[np.ndarray, float]:
    """Propagate a grid density through the Gaussian transition in log space."""
    mu_prev = (
        par["a0"]
        + par["a1"] * grid
        + par["a2"] * prev_block["PV_sum_trans"]
        + par["a3"] * prev_block["FW_sum_trans"]
        + par["a4"] * prev_block["PJ_sum_trans"]
    )
    previous_mass = np.asarray(previous_density, dtype=float) * weights
    log_previous_mass = np.full_like(previous_mass, -np.inf)
    positive = previous_mass > 0
    log_previous_mass[positive] = np.log(previous_mass[positive])
    log_transition = normal_logpdf(
        grid[:, None],
        mu_prev[None, :],
        par["sigma_E"],
    )
    log_predictive = logsumexp(
        log_transition + log_previous_mass[None, :],
        axis=1,
    )
    return _normalize_log_density(
        log_predictive,
        weights,
        label="predictive",
    )


def unpack_theta(theta: np.ndarray, *, e1_known: bool) -> dict:
    """
    If e1_known: no (m0, sigma0) at the end — baseline E_1 is fixed outside theta.
    Otherwise last two entries are m0, log(sigma0) for a Gaussian prior on E_1.
    """
    i = 0
    theta = np.asarray(theta, dtype=float)

    a0 = theta[i]
    i += 1
    # a1 = np.tanh(theta[i])
    a1 = theta[i]
    i += 1
    a2 = theta[i]
    i += 1
    a3 = theta[i]
    i += 1
    a4 = theta[i]
    i += 1
    sigma_E = np.exp(theta[i])
    i += 1

    b0 = theta[i]
    i += 1
    b1 = theta[i]
    i += 1

    c0 = theta[i]
    i += 1
    c1 = theta[i]
    i += 1
    sigma_U1 = np.exp(theta[i])
    i += 1

    d0 = theta[i]
    i += 1
    d1 = theta[i]
    i += 1
    sigma_U2 = np.exp(theta[i])
    i += 1

    alpha0 = theta[i]
    i += 1
    alpha1 = theta[i]
    i += 1
    alpha2_is_weekend = theta[i]
    i += 1
    alpha2_dt = theta[i]
    i += 1
    alpha2_rb = theta[i]
    i += 1
    alpha3 = theta[i]
    i += 1
    alpha4 = theta[i]
    i += 1
    sigma_PV = np.exp(theta[i])
    i += 1
    alpha_ar1 = theta[i]
    i += 1

    beta0 = theta[i]
    i += 1
    beta1 = theta[i]
    i += 1
    beta3 = theta[i]
    i += 1
    beta4 = theta[i]
    i += 1
    beta5 = theta[i]
    i += 1
    beta6 = theta[i]
    i += 1
    beta2_is_weekend = theta[i]
    i += 1
    beta2_rb = theta[i]
    i += 1
    beta_ar1 = theta[i]
    i += 1

    theta0 = theta[i]
    i += 1
    theta1 = theta[i]
    i += 1
    theta3 = theta[i]
    i += 1
    theta4 = theta[i]
    i += 1
    theta5 = theta[i]
    i += 1
    theta6 = theta[i]
    i += 1
    theta2_is_weekend = theta[i]
    i += 1
    theta2_rb = theta[i]
    i += 1
    theta_ar1 = theta[i]
    i += 1

    out = {
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a4": a4,
        "sigma_E": sigma_E,
        "b0": b0,
        "b1": b1,
        "c0": c0,
        "c1": c1,
        "sigma_U1": sigma_U1,
        "d0": d0,
        "d1": d1,
        "sigma_U2": sigma_U2,
        "alpha0": alpha0,
        "alpha1": alpha1,
        "alpha2_is_weekend": alpha2_is_weekend,
        "alpha2_dt": alpha2_dt,
        "alpha2_rb": alpha2_rb,
        "alpha3": alpha3,
        "alpha4": alpha4,
        "sigma_PV": sigma_PV,
        "alpha_ar1": alpha_ar1,
        "beta0": beta0,
        "beta1": beta1,
        "beta3": beta3,
        "beta4": beta4,
        "beta5": beta5,
        "beta6": beta6,
        "beta2_is_weekend": beta2_is_weekend,
        "beta2_rb": beta2_rb,
        "beta_ar1": beta_ar1,
        "theta0": theta0,
        "theta1": theta1,
        "theta3": theta3,
        "theta4": theta4,
        "theta5": theta5,
        "theta6": theta6,
        "theta2_is_weekend": theta2_is_weekend,
        "theta2_rb": theta2_rb,
        "theta_ar1": theta_ar1,
    }
    if not e1_known:
        out["m0"] = theta[i]
        i += 1
        out["sigma0"] = np.exp(theta[i])
        i += 1
    return out


def log_pooled_theta(
    pooled_theta: np.ndarray,
    *,
    e1_known: bool,
    nll: Optional[float] = None,
    success: Optional[bool] = None,
    nit: Optional[int] = None,
) -> dict:
    """Log the pooled MLE vector and its unpacked parameter dictionary."""
    pooled_theta = np.asarray(pooled_theta, dtype=float)
    par = unpack_theta(pooled_theta, e1_known=e1_known)

    header_parts = ["Pooled theta"]
    if success is not None:
        header_parts.append(f"success={success}")
    if nll is not None:
        header_parts.append(f"nll={nll:.6f}")
    if nit is not None:
        header_parts.append(f"nit={nit}")
    header_parts.append(f"dim={pooled_theta.size}")
    logger.info(" | ".join(header_parts))

    groups = [
        ("E_w AR", ["a0", "a1", "a2", "a3", "a4", "sigma_E"]),
        ("J_week", ["b0", "b1"]),
        ("U1", ["c0", "c1", "sigma_U1"]),
        ("U2", ["d0", "d1", "sigma_U2"]),
        ("PV", [
            "alpha0", "alpha1", "alpha2_is_weekend", "alpha2_dt", "alpha2_rb",
            "alpha3", "alpha4", "sigma_PV", "alpha_ar1",
        ]),
        ("FW", [
            "beta0", "beta1", "beta3", "beta4", "beta5", "beta6",
            "beta2_is_weekend", "beta2_rb", "beta_ar1",
        ]),
        ("PJ", [
            "theta0", "theta1", "theta3", "theta4", "theta5", "theta6",
            "theta2_is_weekend", "theta2_rb", "theta_ar1",
        ]),
    ]
    if not e1_known:
        groups.append(("E1 prior", ["m0", "sigma0"]))

    for group_name, keys in groups:
        logger.info("[%s]", group_name)
        for key in keys:
            logger.info("  %s = %.6g", key, par[key])

    logger.debug("pooled_theta raw: %s", np.array2string(pooled_theta, precision=6, separator=", "))
    return par


def theta_dim(*, e1_known: bool) -> int:
    return 41 if e1_known else 43


def initial_theta_from_blocks(blocks, *, e1_known: bool) -> np.ndarray:
    all_J = np.array([b["J_week"] for b in blocks if not np.isnan(b["J_week"])], dtype=float)
    all_U1 = np.array([b["U1"] for b in blocks if not np.isnan(b["U1"])], dtype=float)
    all_U2 = np.array([b["U2"] for b in blocks if not np.isnan(b["U2"])], dtype=float)
    all_PV = (
        np.concatenate([b["pv_y"][~np.isnan(b["pv_y"])] for b in blocks])
        if blocks
        else np.array([0.0])
    )

    pJ = np.clip(np.mean(all_J) if len(all_J) else 0.5, 1e-4, 1 - 1e-4)
    muU1 = np.mean(all_U1) if len(all_U1) else 0.0
    muU2 = np.mean(all_U2) if len(all_U2) else 0.0
    sdU1 = np.std(all_U1) if len(all_U1) > 1 else 1.0
    sdU2 = np.std(all_U2) if len(all_U2) > 1 else 1.0
    muPV = np.mean(all_PV) if len(all_PV) else 0.0
    sdPV = np.std(all_PV) if len(all_PV) > 1 else 1.0

    parts = [
        0.0,
        # np.arctanh(0.5),
        0.5,
        0.0,
        0.0,
        0.0,
        np.log(0.5),
        np.log(pJ / (1.0 - pJ)),
        0.1,
        muU1,
        0.1,
        np.log(max(sdU1, 0.1)),
        muU2,
        0.1,
        np.log(max(sdU2, 0.1)),
        muPV,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        np.log(max(sdPV, 0.1)),
        0.0,
        0.0,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    if not e1_known:
        parts.extend([0.0, np.log(1.0)])
    return np.asarray(parts, dtype=float)


def build_user_blocks(
    dat_user,
                      week_col="week",
                      date_col="Date",
                      decision_col="DecisionTime",
                      J_col="week_present",
                      U1_col="Exp-tool-1_norm",
                      U2_col="Exp-tool-2_norm",
    PV_col="HourlyPageviewCount_norm",
    FW_col="nextday_wearing",
                      PJ_col="daily_present",
    a_col="WalkingSuggestion",
    weekend_col="is_weekend",
    burden_col="recent_burden_norm",
    PV_lag1_col="hourly_pageview_count_lag1",
    FW_lag_col="morning_wearing",
    PJ_lag_col="daily_present_yesterday",
    *,
    hourly_pv=True,
    full_weeks=None,   # e.g. range(1, 13)
    missing_week_mode="mean_impute_transition",  # or "error"
):
    """
    Build one block per study week ``w`` (default ``full_weeks=range(1, 13)`` →
    weeks 1..12). This is the discrete-time index of the AR state E_w, not the
    quadrature grid over E.

    Key design:
      - The within-week likelihood uses only actually observed rows:
            pv_y, FW_daily, PJ_daily
        plus ``day_is_weekend`` and ``day_burden`` (aligned with each calendar day, from the
        afternoon row) for FW/PJ logit covariates.
        AR coefficients multiply precomputed lag columns: ``PV_lag1_col`` per hour,
        ``FW_lag_col`` / ``PJ_lag_col`` per day (NaNs treated as 0 in the likelihood).
        Missing weeks keep these as empty arrays.

      - Weekly intensity summaries for the transition use fixed denominators
        (``nansum(pv)/14``, ``nansum(FW|PJ)/7``): each slot contributes 0 if NaN,
        not a mean over observed-only slots. With a complete 14-slot / 7-day
        panel this matches zero-imputed averages.

      - The state transition uses separate weekly summaries:
            PV_sum_trans, FW_sum_trans, PJ_sum_trans
        which equal observed sums when available and participant-specific mean
        imputation otherwise.
    """
    dat = dat_user.copy()
    dat[date_col] = pd.to_datetime(dat[date_col])
    dat = dat.sort_values([week_col, date_col, decision_col], na_position="last").reset_index(drop=True)

    if PV_col not in dat.columns:
        raise KeyError(
            f"{PV_col!r} not in dataframe (use HourlyPageviewCount_norm like 2_fit_vanilla_testbed.py)"
        )
    for _label, _col in (
        ("PV_lag1_col", PV_lag1_col),
        ("FW_lag_col", FW_lag_col),
        ("PJ_lag_col", PJ_lag_col),
    ):
        if _col is not None and _col not in dat.columns:
            raise KeyError(f"{_col!r} not in dataframe (expected for {_label})")

    observed_by_week = {int(w): g.copy() for w, g in dat.groupby(week_col, sort=True)}

    if full_weeks is None:
        wmin = int(dat[week_col].min())
        wmax = int(dat[week_col].max())
        full_weeks = range(wmin, wmax + 1)

    def _first_nonmissing(series):
            s = series.dropna()
            return float(s.iloc[0]) if len(s) else np.nan

    def _build_observed_block(week, g_week):
        g_week = g_week.sort_values([date_col, decision_col], na_position="last").reset_index(drop=True)

        J_week = _first_nonmissing(g_week[J_col])
        U1 = _first_nonmissing(g_week[U1_col])
        U2 = _first_nonmissing(g_week[U2_col])

        pv_y_list, pv_a_list = [], []
        pv_lag1_list = []
        pv_ctx_rows = []
        a0_list, a1_list = [], []
        fw_list, pj_list = [], []
        day_is_weekend_list, day_burden_list = [], []
        day_fw_lag_list, day_pj_lag_list = [], []

        for _d, g_day in g_week.groupby(date_col, sort=True):
            g_day = g_day.sort_values(decision_col, na_position="last")

            aa = g_day[a_col].to_numpy(dtype=float)
            pv = g_day[PV_col].to_numpy(dtype=float)
            weekendv = g_day[weekend_col].to_numpy(dtype=float)
            burden = g_day[burden_col].to_numpy(dtype=float)
            dtv = g_day[decision_col].to_numpy(dtype=float)
            lag1v = (
                g_day[PV_lag1_col].to_numpy(dtype=float)
                if PV_lag1_col is not None
                else np.full(len(g_day), np.nan, dtype=float)
            )

            for k in range(len(g_day)):
                pv_y_list.append(float(pv[k]) if not np.isnan(pv[k]) else np.nan)
                pv_a_list.append(float(aa[k]) if not np.isnan(aa[k]) else np.nan)
                pv_lag1_list.append(
                    float(lag1v[k]) if not np.isnan(lag1v[k]) else np.nan
                )
                weekend_k = float(weekendv[k]) if not np.isnan(weekendv[k]) else np.nan
                rb_k = float(burden[k]) if not np.isnan(burden[k]) else np.nan
                dt_k = float(dtv[k]) if (hourly_pv and not np.isnan(dtv[k])) else 0.0
                pv_ctx_rows.append([weekend_k, dt_k, rb_k])

            row_morning = g_day.loc[g_day[decision_col] == 0].iloc[0]
            row_afternoon = g_day.loc[g_day[decision_col] == 1].iloc[0]

            A0_day = float(row_morning[a_col]) if not pd.isna(row_morning[a_col]) else 0.0
            A1_day = float(row_afternoon[a_col]) if not pd.isna(row_afternoon[a_col]) else 0.0

            a0_list.append(A0_day)
            a1_list.append(A1_day)

            fw_list.append(float(row_morning[FW_col]) if not pd.isna(row_morning[FW_col]) else np.nan)
            pj_list.append(float(row_morning[PJ_col]) if not pd.isna(row_morning[PJ_col]) else np.nan)


            day_is_weekend_list.append(
                float(row_afternoon[weekend_col]) if not pd.isna(row_afternoon[weekend_col]) else np.nan
            )
            day_burden_list.append(
                float(row_afternoon[burden_col]) if not pd.isna(row_afternoon[burden_col]) else np.nan
            )

            if FW_lag_col is not None:
                day_fw_lag_list.append(
                    float(row_morning[FW_lag_col]) if not pd.isna(row_morning[FW_lag_col]) else np.nan
                )
            else:
                day_fw_lag_list.append(np.nan)

            if PJ_lag_col is not None:
                day_pj_lag_list.append(
                    float(row_morning[PJ_lag_col]) if not pd.isna(row_morning[PJ_lag_col]) else np.nan
                )
            else:
                day_pj_lag_list.append(np.nan)

        pv_y = np.asarray(pv_y_list, dtype=float)
        pv_lag1 = np.asarray(pv_lag1_list, dtype=float)
        pv_c_ctx = np.asarray(pv_ctx_rows, dtype=float) if len(pv_ctx_rows) else np.zeros((0, 3), dtype=float)
        day_A0 = np.asarray(a0_list, dtype=float)
        day_A1 = np.asarray(a1_list, dtype=float)
        FW_daily = np.asarray(fw_list, dtype=float)
        PJ_daily = np.asarray(pj_list, dtype=float)
        day_is_weekend = np.asarray(day_is_weekend_list, dtype=float)
        day_burden = np.asarray(day_burden_list, dtype=float)
        day_FW_lag = np.asarray(day_fw_lag_list, dtype=float)
        day_PJ_lag = np.asarray(day_pj_lag_list, dtype=float)

        PV_sum = float(np.nansum(pv_y) / 14.0) if np.any(~np.isnan(pv_y)) else np.nan
        FW_sum = float(np.nansum(FW_daily) / 7.0) if np.any(~np.isnan(FW_daily)) else np.nan
        PJ_sum = float(np.nansum(PJ_daily) / 7.0) if np.any(~np.isnan(PJ_daily)) else np.nan
        return {
            "week": int(week),
            "is_missing_week": False,
            "J_week": J_week,
            "U1": U1,
            "U2": U2,
            "pv_y": pv_y,
            "pv_lag1": pv_lag1,
            "pv_c_ctx": pv_c_ctx,
            "pv_a": np.asarray(pv_a_list, dtype=float),
            "day_A0": day_A0,
            "day_A1": day_A1,
            "day_is_weekend": day_is_weekend,
            "day_burden": day_burden,
            "day_FW_lag": day_FW_lag,
            "day_PJ_lag": day_PJ_lag,
            "FW_daily": FW_daily,
            "PJ_daily": PJ_daily,
            "PV_sum": PV_sum,
            "FW_sum": FW_sum,
            "PJ_sum": PJ_sum,
            # filled below
            "PV_sum_trans": np.nan,
            "FW_sum_trans": np.nan,
            "PJ_sum_trans": np.nan,
        }

    # First build observed-week blocks only
    observed_blocks = {
        week: _build_observed_block(week, g_week)
        for week, g_week in observed_by_week.items()
    }

    # Participant-specific means for transition-only imputation
    pv_obs = np.array(
        [b["PV_sum"] for b in observed_blocks.values() if not np.isnan(b["PV_sum"])],
        dtype=float,
    )
    fw_obs = np.array(
        [b["FW_sum"] for b in observed_blocks.values() if not np.isnan(b["FW_sum"])],
        dtype=float,
    )
    pj_obs = np.array(
        [b["PJ_sum"] for b in observed_blocks.values() if not np.isnan(b["PJ_sum"])],
        dtype=float,
    )

    pv_mean = float(np.mean(pv_obs)) if len(pv_obs) else 0.0
    fw_mean = float(np.mean(fw_obs)) if len(fw_obs) else 0.0
    pj_mean = float(np.mean(pj_obs)) if len(pj_obs) else 0.0

    blocks = []
    for week in full_weeks:
        if week not in observed_blocks:
            if missing_week_mode == "error":
                raise ValueError(f"Missing calendar week {week} for participant.")

            blocks.append({
                "week": int(week),
                "is_missing_week": True,
                "J_week": np.nan,
                "U1": np.nan,
                "U2": np.nan,
                "pv_y": np.asarray([], dtype=float),
                "pv_lag1": np.asarray([], dtype=float),
                "pv_c_ctx": np.zeros((0, 3), dtype=float),
                "pv_a": np.asarray([], dtype=float),
                "day_A0": np.asarray([], dtype=float),
                "day_A1": np.asarray([], dtype=float),
                "day_is_weekend": np.asarray([], dtype=float),
                "day_burden": np.asarray([], dtype=float),
                "day_FW_lag": np.asarray([], dtype=float),
                "day_PJ_lag": np.asarray([], dtype=float),
                "FW_daily": np.asarray([], dtype=float),
                "PJ_daily": np.asarray([], dtype=float),
                # actual observed sums are missing
                "PV_sum": np.nan,
                "FW_sum": np.nan,
                "PJ_sum": np.nan,
                # transition-only imputation
                "PV_sum_trans": pv_mean,
                "FW_sum_trans": fw_mean,
                "PJ_sum_trans": pj_mean,
            })
        else:
            b = observed_blocks[week]
            b["PV_sum_trans"] = b["PV_sum"] if not np.isnan(b["PV_sum"]) else pv_mean
            b["FW_sum_trans"] = b["FW_sum"] if not np.isnan(b["FW_sum"]) else fw_mean
            b["PJ_sum_trans"] = b["PJ_sum"] if not np.isnan(b["PJ_sum"]) else pj_mean
            blocks.append(b)

    return blocks


def _week_loglik_components_on_grid(grid: np.ndarray, block: dict, par: dict) -> np.ndarray:
    """log p(Y_w | E_w = grid, par) for each grid point; vector of shape grid.shape."""
    grid = np.asarray(grid, dtype=float)
    ll = np.zeros_like(grid)

    # J_week is always modeled when observed
    if not np.isnan(block["J_week"]):
        eta = par["b0"] + par["b1"] * grid
        ll += bernoulli_logpmf(block["J_week"], eta)

    # U1/U2 are structurally present only when J_week == 1
    if (not np.isnan(block["J_week"])) and (int(round(block["J_week"])) == 1):
        if not np.isnan(block["U1"]):
            mu = par["c0"] + par["c1"] * grid
            ll += normal_logpdf(block["U1"], mu, par["sigma_U1"])

        if not np.isnan(block["U2"]):
            mu = par["d0"] + par["d1"] * grid
            ll += normal_logpdf(block["U2"], mu, par["sigma_U2"])

    pv_y = block["pv_y"]
    pv_ctx = block["pv_c_ctx"]
    pv_a = block["pv_a"]
    coef2 = np.array(
        [par["alpha2_is_weekend"], par["alpha2_dt"], par["alpha2_rb"]],
        dtype=float,
    )

    for j in range(len(pv_y)):
        y = pv_y[j]
        if np.isnan(y):
            continue
        row = np.asarray(pv_ctx[j], dtype=float)
        row = np.where(np.isnan(row), 0.0, row)
        dotc = float(row @ coef2)
        a = pv_a[j]
        a = 0.0 if np.isnan(a) else a
        pl1 = block.get("pv_lag1")
        y_lag = (
            float(pl1[j])
            if pl1 is not None and len(pl1) > j and not np.isnan(pl1[j])
            else 0.0
        )
        mu = (
            par["alpha0"]
            + par["alpha1"] * grid
            + dotc
            + par["alpha_ar1"] * y_lag
            + a * (par["alpha3"] + par["alpha4"] * grid)
        )
        ll += normal_logpdf(y, mu, par["sigma_PV"])

    for d in range(len(block["FW_daily"])):
        y = block["FW_daily"][d]
        if np.isnan(y):
            continue
        a0 = block["day_A0"][d]
        a1 = block["day_A1"][d]
        a0 = 0.0 if np.isnan(a0) else a0
        a1 = 0.0 if np.isnan(a1) else a1
        dd = block.get("day_is_weekend")
        bd = block.get("day_burden")
        weekend_d = float(dd[d]) if dd is not None and len(dd) > d else np.nan
        bd_d = float(bd[d]) if bd is not None and len(bd) > d else np.nan
        weekend_d = 0.0 if np.isnan(weekend_d) else weekend_d
        bd_d = 0.0 if np.isnan(bd_d) else bd_d
        fl = block.get("day_FW_lag")
        y_lag = (
            float(fl[d])
            if fl is not None and len(fl) > d and not np.isnan(fl[d])
            else 0.0
        )
        eta = (
            par["beta0"]
            + par["beta1"] * grid
            + par["beta2_is_weekend"] * weekend_d
            + par["beta2_rb"] * bd_d
            + par["beta_ar1"] * y_lag
            + a0 * (par["beta3"] + par["beta4"] * grid)
            + a1 * (par["beta5"] + par["beta6"] * grid)
        )
        ll += bernoulli_logpmf(y, eta)

    for d in range(len(block["PJ_daily"])):
        y = block["PJ_daily"][d]
        if np.isnan(y):
            continue
        a0 = block["day_A0"][d]
        a1 = block["day_A1"][d]
        a0 = 0.0 if np.isnan(a0) else a0
        a1 = 0.0 if np.isnan(a1) else a1
        dd = block.get("day_is_weekend")
        bd = block.get("day_burden")
        weekend_d = float(dd[d]) if dd is not None and len(dd) > d else np.nan
        bd_d = float(bd[d]) if bd is not None and len(bd) > d else np.nan
        weekend_d = 0.0 if np.isnan(weekend_d) else weekend_d
        bd_d = 0.0 if np.isnan(bd_d) else bd_d
        pl = block.get("day_PJ_lag")
        y_lag = (
            float(pl[d])
            if pl is not None and len(pl) > d and not np.isnan(pl[d])
            else 0.0
        )
        eta = (
            par["theta0"]
            + par["theta1"] * grid
            + par["theta2_is_weekend"] * weekend_d
            + par["theta2_rb"] * bd_d
            + par["theta_ar1"] * y_lag
            + a0 * (par["theta3"] + par["theta4"] * grid)
            + a1 * (par["theta5"] + par["theta6"] * grid)
        )
        ll += bernoulli_logpmf(y, eta)

    return ll


def week_loglik_on_grid(grid, block, par):
    return _week_loglik_components_on_grid(grid, block, par)


def week_loglik_at_point(e: float, block: dict, par: dict) -> float:
    g = np.asarray([e], dtype=float)
    return float(_week_loglik_components_on_grid(g, block, par)[0])


def penalized_json_export(
    res,
    filt: dict,
    blocks: list,
    *,
    digits: int = 3,
) -> tuple[dict, dict]:
    """
    Build two dicts for merging into params_env_<id>.json and pred_<id>.json.

    Parameter convention:
      - one theta_penalized_* key per model
      - companion theta_penalized_*_names key documents parameter order

    Prediction/residual convention:
      - prediction and residual arrays have the same length
      - missing residuals are stored as None, which becomes JSON null
      - J, U1, U2: one entry per modeled week
      - PV: one entry per included PV row
      - FW/PJ: one entry per included day
    """
    par = filt["params"]
    e1_fixed = filt.get("e1_known") is not None

    def r3(x):
        if x is None:
            return None
        xf = float(x)
        if not np.isfinite(xf):
            return None
        return float(np.round(xf, digits))

    def _sigm(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))

    env_penalized: dict = {
        "estimation_method": "ridge_penalized_likelihood",
        "ridge_lambda": r3(filt.get("ridge_lambda")),
        "theta_penalized_Ew": [
            r3(par["a0"]),
            r3(par["a1"]),
            r3(par["a2"]),
            r3(par["a3"]),
            r3(par["a4"]),
            r3(par["sigma_E"]),
        ],
        "theta_penalized_Ew_names": [
            "a0",
            "a1",
            "a2_PV_lag_week",
            "a3_FW_lag_week",
            "a4_PJ_lag_week",
            "sigma_E",
        ],

        "theta_penalized_J": [
            r3(par["b0"]),
            r3(par["b1"]),
        ],
        "theta_penalized_J_names": [
            "b0",
            "b1_Ew",
        ],

        "theta_penalized_U1": [
            r3(par["c0"]),
            r3(par["c1"]),
            r3(par["sigma_U1"]),
        ],
        "theta_penalized_U1_names": [
            "c0",
            "c1_Ew",
            "sigma_U1",
        ],

        "theta_penalized_U2": [
            r3(par["d0"]),
            r3(par["d1"]),
            r3(par["sigma_U2"]),
        ],
        "theta_penalized_U2_names": [
            "d0",
            "d1_Ew",
            "sigma_U2",
        ],

        "theta_penalized_PV": [
            r3(par["alpha0"]),
            r3(par["alpha1"]),
            r3(par["alpha2_is_weekend"]),
            r3(par["alpha2_dt"]),
            r3(par["alpha2_rb"]),
            r3(par["alpha_ar1"]),
            r3(par["alpha3"]),
            r3(par["alpha4"]),
            r3(par["sigma_PV"]),
        ],
        "theta_penalized_PV_names": [
            "alpha0",
            "alpha1_Ew",
            "alpha2_is_weekend",
            "alpha2_decision_time",
            "alpha2_recent_burden",
            "alpha_ar1_hourly_pageview_lag1",
            "alpha3_action",
            "alpha4_action_by_Ew",
            "sigma_PV",
        ],

        "theta_penalized_FW": [
            r3(par["beta0"]),
            r3(par["beta1"]),
            r3(par["beta2_is_weekend"]),
            r3(par["beta2_rb"]),
            r3(par["beta_ar1"]),
            r3(par["beta3"]),
            r3(par["beta4"]),
            r3(par["beta5"]),
            r3(par["beta6"]),
        ],
        "theta_penalized_FW_names": [
            "beta0",
            "beta1_Ew",
            "beta2_is_weekend",
            "beta2_recent_burden",
            "beta_ar1_morning_wearing",
            "beta3_A0_morning",
            "beta4_A0_morning_by_Ew",
            "beta5_A1_afternoon",
            "beta6_A1_afternoon_by_Ew",
        ],

        "theta_penalized_PJ": [
            r3(par["theta0"]),
            r3(par["theta1"]),
            r3(par["theta2_is_weekend"]),
            r3(par["theta2_rb"]),
            r3(par["theta_ar1"]),
            r3(par["theta3"]),
            r3(par["theta4"]),
            r3(par["theta5"]),
            r3(par["theta6"]),
        ],
        "theta_penalized_PJ_names": [
            "theta0",
            "theta1_Ew",
            "theta2_is_weekend",
            "theta2_recent_burden",
            "theta_ar1_daily_present_yesterday",
            "theta3_A0_morning",
            "theta4_A0_morning_by_Ew",
            "theta5_A1_afternoon",
            "theta6_A1_afternoon_by_Ew",
        ],

        "penalized_loglik": r3(filt["loglik"]),
        "penalized_objective": r3(res.fun),
        "penalized_opt_success": bool(res.success),
        "penalized_opt_nit": int(res.nit) if hasattr(res, "nit") else None,
    }

    if not e1_fixed:
        env_penalized["theta_penalized_E1_prior"] = [
            r3(par.get("m0")),
            r3(par.get("sigma0")),
        ]
        env_penalized["theta_penalized_E1_prior_names"] = [
            "m0",
            "sigma0",
        ]

    E = np.asarray(filt["filtered_means"], dtype=float)

    pJ, rJ = [], []
    pU1, rU1 = [], []
    pU2, rU2 = [], []

    pPV, rPV = [], []
    pFW, rFW = [], []
    pPJ, rPJ = [], []

    for t, block in enumerate(blocks):
        e = float(E[t]) if t < len(E) and np.isfinite(E[t]) else 0.0

        # ------------------------------------------------------------
        # Weekly J: one entry per week
        # ------------------------------------------------------------
        eta_J = par["b0"] + par["b1"] * e
        ph_J = float(_sigm(eta_J))

        if not np.isnan(block.get("J_week", np.nan)):
            pJ.append(r3(ph_J))
            rJ.append(r3(float(block["J_week"]) - ph_J))
        else:
            pJ.append(r3(ph_J))
            rJ.append(None)

        # ------------------------------------------------------------
        # Weekly U1/U2: one entry per week
        # Residual is None if U is missing or J_week != 1.
        # ------------------------------------------------------------
        j1 = (
            not np.isnan(block.get("J_week", np.nan))
            and int(round(block["J_week"])) == 1
        )

        mu_U1 = float(par["c0"] + par["c1"] * e)
        if j1 and not np.isnan(block.get("U1", np.nan)):
            pU1.append(r3(mu_U1))
            rU1.append(r3(float(block["U1"]) - mu_U1))
        else:
            pU1.append(r3(mu_U1))
            rU1.append(None)

        mu_U2 = float(par["d0"] + par["d1"] * e)
        if j1 and not np.isnan(block.get("U2", np.nan)):
            pU2.append(r3(mu_U2))
            rU2.append(r3(float(block["U2"]) - mu_U2))
        else:
            pU2.append(r3(mu_U2))
            rU2.append(None)

        # ------------------------------------------------------------
        # Hourly PV: one entry per included PV row
        # Prediction is computed for every PV row.
        # Residual is None if observed PV is missing.
        # ------------------------------------------------------------
        coef2 = np.array(
            [par["alpha2_is_weekend"], par["alpha2_dt"], par["alpha2_rb"]],
            dtype=float,
        )

        pv_y = block["pv_y"]
        pv_ctx = block["pv_c_ctx"]
        pv_a = block["pv_a"]
        pv_lag1 = block.get("pv_lag1")

        for j in range(len(pv_y)):
            y = pv_y[j]

            row = np.asarray(pv_ctx[j], dtype=float)
            row = np.where(np.isnan(row), 0.0, row)
            dotc = float(row @ coef2)

            A = pv_a[j]
            A = 0.0 if np.isnan(A) else float(A)

            y_lag = (
                float(pv_lag1[j])
                if pv_lag1 is not None
                and len(pv_lag1) > j
                and not np.isnan(pv_lag1[j])
                else 0.0
            )

            mu = float(
                par["alpha0"]
                + par["alpha1"] * e
                + dotc
                + par["alpha_ar1"] * y_lag
                + A * (par["alpha3"] + par["alpha4"] * e)
            )

            pPV.append(r3(mu))

            if np.isnan(y):
                rPV.append(None)
            else:
                rPV.append(r3(float(y) - mu))

        # ------------------------------------------------------------
        # Daily FW: one entry per included day
        # Prediction is computed for every daily row.
        # Residual is None if observed FW is missing.
        # ------------------------------------------------------------
        fw_y = block["FW_daily"]

        for d in range(len(fw_y)):
            y = fw_y[d]

            A0 = (
                float(block["day_A0"][d])
                if len(block["day_A0"]) > d and not np.isnan(block["day_A0"][d])
                else 0.0
            )
            A1 = (
                float(block["day_A1"][d])
                if len(block["day_A1"]) > d and not np.isnan(block["day_A1"][d])
                else 0.0
            )

            day_is_weekend = block.get("day_is_weekend")
            day_burden = block.get("day_burden")
            day_FW_lag = block.get("day_FW_lag")

            weekend_d = (
                float(day_is_weekend[d])
                if day_is_weekend is not None
                and len(day_is_weekend) > d
                and not np.isnan(day_is_weekend[d])
                else 0.0
            )
            burden_d = (
                float(day_burden[d])
                if day_burden is not None
                and len(day_burden) > d
                and not np.isnan(day_burden[d])
                else 0.0
            )
            lag_d = (
                float(day_FW_lag[d])
                if day_FW_lag is not None
                and len(day_FW_lag) > d
                and not np.isnan(day_FW_lag[d])
                else 0.0
            )

            eta = (
                par["beta0"]
                + par["beta1"] * e
                + par["beta2_is_weekend"] * weekend_d
                + par["beta2_rb"] * burden_d
                + par["beta_ar1"] * lag_d
                + A0 * (par["beta3"] + par["beta4"] * e)
                + A1 * (par["beta5"] + par["beta6"] * e)
            )

            ph = float(_sigm(eta))
            pFW.append(r3(ph))

            if np.isnan(y):
                rFW.append(None)
            else:
                rFW.append(r3(float(y) - ph))

        # ------------------------------------------------------------
        # Daily PJ: one entry per included day
        # Prediction is computed for every daily row.
        # Residual is None if observed PJ is missing.
        # ------------------------------------------------------------
        pj_y = block["PJ_daily"]

        for d in range(len(pj_y)):
            y = pj_y[d]

            A0 = (
                float(block["day_A0"][d])
                if len(block["day_A0"]) > d and not np.isnan(block["day_A0"][d])
                else 0.0
            )
            A1 = (
                float(block["day_A1"][d])
                if len(block["day_A1"]) > d and not np.isnan(block["day_A1"][d])
                else 0.0
            )

            day_is_weekend = block.get("day_is_weekend")
            day_burden = block.get("day_burden")
            day_PJ_lag = block.get("day_PJ_lag")

            weekend_d = (
                float(day_is_weekend[d])
                if day_is_weekend is not None
                and len(day_is_weekend) > d
                and not np.isnan(day_is_weekend[d])
                else 0.0
            )
            burden_d = (
                float(day_burden[d])
                if day_burden is not None
                and len(day_burden) > d
                and not np.isnan(day_burden[d])
                else 0.0
            )
            lag_d = (
                float(day_PJ_lag[d])
                if day_PJ_lag is not None
                and len(day_PJ_lag) > d
                and not np.isnan(day_PJ_lag[d])
                else 0.0
            )

            eta = (
                par["theta0"]
                + par["theta1"] * e
                + par["theta2_is_weekend"] * weekend_d
                + par["theta2_rb"] * burden_d
                + par["theta_ar1"] * lag_d
                + A0 * (par["theta3"] + par["theta4"] * e)
                + A1 * (par["theta5"] + par["theta6"] * e)
            )

            ph = float(_sigm(eta))
            pPJ.append(r3(ph))

            if np.isnan(y):
                rPJ.append(None)
            else:
                rPJ.append(r3(float(y) - ph))

    env_penalized["resid_penalized_J_week"] = rJ
    env_penalized["resid_penalized_U1"] = rU1
    env_penalized["resid_penalized_U2"] = rU2
    env_penalized["resid_penalized_hourly_pageview"] = rPV
    env_penalized["resid_penalized_nextday_wearing"] = rFW
    env_penalized["resid_penalized_daily_present"] = rPJ

    E_export = build_exported_Ew_series(filt)
    pred_penalized = {
        "pred_penalized_filtered_Ew": [r3(x) for x in E_export.tolist()],
        "pred_penalized_J_week": pJ,
        "pred_penalized_U1": pU1,
        "pred_penalized_U2": pU2,
        "pred_penalized_hourly_pageview": pPV,
        "pred_penalized_nextday_wearing": pFW,
        "pred_penalized_daily_present": pPJ,
    }

    return env_penalized, pred_penalized


def attach_filtered_Ew_to_df_fit(
    df_fit: pd.DataFrame,
    filtered_states: dict,
    *,
    e1_known: Optional[float] = None,
    pu_col: str = "perceived_utility",
    pu_last_col: str = "perceived_utility_lastweek",
    week_col: str = "week",
) -> pd.DataFrame:
    """Add exported \\hat E_{w+1} and its literal one-week lag \\hat E_w to ``df_fit``.

    ``filtered_states[uid]['filtered_means']`` is indexed ``0..T-1`` for study
    weeks ``1..T`` (``full_weeks=range(1, 13)`` → T=12), with ``fm[w-1]`` =
    filtered \\hat E_w for week w (``fm[0]`` = the fixed baseline E_1 = v).

    Each existing row at study week w (1..T) gets, with ``e1_known=v``:
      - ``perceived_utility_lastweek`` = \\hat E_w = ``fm[w-1]`` (the state
        entering week w, known before week w's own data — this is exactly the
        state that governs week w's likelihood in the model). Spans
        E_1,...,E_T over weeks 1..T.
      - ``perceived_utility`` = \\hat E_{w+1} = ``fm[w]`` for w<T, or the
        predictive ``terminal_predicted_mean`` (E_{T+1}) for w=T (end of week
        T, one AR step past the last filtered state). Spans E_2,...,E_{T+1}
        over weeks 1..T.

    By construction ``perceived_utility_lastweek`` on week w equals
    ``perceived_utility`` on week w-1 (literal weekly lag), with week 1's
    lastweek anchored to the E_1 baseline v (no week-0 row exists).

    Returns a new dataframe; the input is not modified.
    """
    out = df_fit.copy()
    out[pu_col] = np.nan
    out[pu_last_col] = np.nan

    for uid, filt in filtered_states.items():
        fm = np.asarray(filt.get("filtered_means", []), dtype=float)
        T = fm.size
        if T == 0:
            continue
        user_mask = out["ParticipantIdentifier"] == uid
        if not user_mask.any():
            continue
        terminal = filt.get("terminal_predicted_mean")
        terminal = float(terminal) if terminal is not None else np.nan
        for w in range(1, T + 1):
            row_mask = user_mask & (out[week_col] == w)
            if not row_mask.any():
                continue
            v_lastweek = fm[w - 1]
            out.loc[row_mask, pu_last_col] = (
                float(v_lastweek) if np.isfinite(v_lastweek) else np.nan
            )
            v_now = fm[w] if w < T else terminal
            out.loc[row_mask, pu_col] = (
                float(v_now) if np.isfinite(v_now) else np.nan
            )
    return out


def save_df_fit_with_perceived_utility(
    df_fit: pd.DataFrame,
    filtered_states: dict,
    out_path: Path,
    *,
    e1_known: Optional[float] = None,
    pu_col: str = "perceived_utility",
    pu_last_col: str = "perceived_utility_lastweek",
    week_col: str = "week",
) -> pd.DataFrame:
    """Attach exported E_w columns to ``df_fit`` and write the result to ``out_path``.

    Expects study weeks 1–12 in ``df_fit``. On row week=w (1..12):
    ``perceived_utility`` = \\hat E_{w+1} (E_2..E_13 over weeks 1..12, terminal
    E_13 on week 12 is predictive); ``perceived_utility_lastweek`` = \\hat E_w
    (E_1..E_12 over weeks 1..12).
    """
    out = attach_filtered_Ew_to_df_fit(
        df_fit,
        filtered_states,
        e1_known=e1_known,
        pu_col=pu_col,
        pu_last_col=pu_last_col,
        week_col=week_col,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(
        f"Saved df_fit with '{pu_col}' / '{pu_last_col}' columns to {out_path} "
        f"(rows={len(out)})"
    )
    return out


def write_joint_penalized_into_vanilla_json_files(
    userid,
    res,
    filt: dict,
    blocks: list,
    *,
    work_dir: Optional[Path] = None,
    digits: int = 3,
) -> None:
    """
    Write joint ridge-penalized parameters, residuals, and predictions to:

      params_env_<id>.json
      pred_<id>.json

    The files are **overwritten** with the penalized-fit keys (any stale vanilla keys
    or query-imputation suffix from a prior run are wiped). The downstream
    ``2_fit_vanilla_testbed.py`` then merges its vanilla keys into the same
    JSONs while preserving the penalized-fit keys written here.
    """
    wd = WORK_DIR if work_dir is None else Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)

    uid = int(userid) if isinstance(userid, (int, np.integer)) else userid

    p_env = wd / f"params_env_{uid}.json"
    p_pred = wd / f"pred_{uid}.json"

    env_penalized, pred_penalized = penalized_json_export(
        res, filt, blocks, digits=digits
    )

    with open(p_env, "w", encoding="utf-8") as f:
        json.dump(env_penalized, f, allow_nan=False)

    with open(p_pred, "w", encoding="utf-8") as f:
        json.dump(pred_penalized, f, allow_nan=False)


# Indices into the perceived-utility fitted parameter vectors used for empirical-Bayes imputation.
# These are based on theta_penalized_PV/FW/PJ.

PV_QUERY_IMPUTE_IDX = np.array([0, 1, 2, 3, 4], dtype=int)
FB_QUERY_IMPUTE_IDX = np.array([0, 1, 2, 3], dtype=int)
PJ_QUERY_IMPUTE_IDX = np.array([0, 1, 2, 3], dtype=int)

PV_IMPUTE_K = int(PV_QUERY_IMPUTE_IDX.size)
FB_IMPUTE_K = int(FB_QUERY_IMPUTE_IDX.size)
PJ_IMPUTE_K = int(PJ_QUERY_IMPUTE_IDX.size)

PV_STAT_LEN = int(PV_QUERY_IMPUTE_IDX.max() + 1)
FB_STAT_LEN = int(FB_QUERY_IMPUTE_IDX.max() + 1)
PJ_STAT_LEN = int(PJ_QUERY_IMPUTE_IDX.max() + 1)

_META_KEY = "_query_interaction_impute"


def impute_query_action_interaction_effects(
    user_ids,
    *,
    work_dir: Optional[Path] = None,
    xi: float = 1.0 / 8.0,
    digits: int = 3,
    rng: Optional[np.random.Generator] = None,
    min_users_for_empirical: int = 2,
) -> None:
    """
    Impute coefficients for an unavailable query/action type using the fitted
    perceived-utility model parameters.

    This appends imputed coefficients to:

      theta_penalized_PV
      theta_penalized_FW
      theta_penalized_PJ

    The function is idempotent: if it was previously run, it removes the previously
    appended suffix before appending a fresh imputed suffix.
    """
    wd = WORK_DIR if work_dir is None else Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)

    gen = rng if rng is not None else np.random.default_rng()

    uid_list = [
        int(u) if isinstance(u, (int, np.integer)) else u
        for u in np.asarray(user_ids).ravel()
    ]

    def _strip_imputed_suffix(values: np.ndarray, names: list) -> tuple[np.ndarray, list]:
        """Strip any trailing ``query_imputed_*`` columns based on the names array.

        This is the source of truth for "is the imputed suffix already present?" —
        ``_META_KEY`` is treated as advisory only, since merge/overwrite operations
        elsewhere can desync it from the actual array length.
        """
        n_strip = 0
        for nm in reversed(names):
            if isinstance(nm, str) and nm.startswith("query_imputed_"):
                n_strip += 1
            else:
                break
        if n_strip == 0:
            return values, names
        if values.size >= n_strip:
            values = values[:-n_strip]
        return values, names[:-n_strip]

    def collect_rows(key: str, stat_len: int) -> np.ndarray:
        rows = []

        for uid in uid_list:
            p = wd / f"params_env_{uid}.json"
            if not p.is_file():
                continue

            with open(p, encoding="utf-8") as f:
                env = json.load(f)

            t = env.get(key)
            if t is None:
                continue

            v = np.asarray(t, dtype=float).ravel()
            names = list(env.get(f"{key}_names", []))
            v, _ = _strip_imputed_suffix(v, names)

            if v.size < stat_len:
                continue

            rows.append(v[:stat_len].copy())

        if not rows:
            return np.zeros((0, stat_len), dtype=float)

        return np.stack(rows, axis=0)

    def population_sample(
        rows: np.ndarray,
        idx: np.ndarray,
        stat_len: int,
        n_user: int,
    ) -> np.ndarray:
        if rows.shape[0] < min_users_for_empirical:
            mean_full = np.zeros(stat_len, dtype=float)
            var_full = np.full(stat_len, (0.05 * xi) ** 2, dtype=float)
        else:
            x = np.abs(rows)

            mean_full = np.mean(x, axis=0) * xi
            if stat_len > 1:
                mean_full[0] = 2.0 * np.mean(mean_full[1:])

            x = x * xi
            var_full = np.var(x, axis=0)
            if stat_len > 1:
                var_full[0] = 4.0 * np.mean(var_full[1:])

        var_full = np.maximum(var_full, 1e-12)

        draws = gen.normal(
            mean_full,
            np.sqrt(var_full),
            size=(n_user, stat_len),
        )

        sub = draws[:, idx]
        return np.round(sub, digits)

    # Use perceived-utility model keys, not old vanilla keys.
    pv_rows = collect_rows("theta_penalized_PV", PV_STAT_LEN)
    fb_rows = collect_rows("theta_penalized_FW", FB_STAT_LEN)
    pj_rows = collect_rows("theta_penalized_PJ", PJ_STAT_LEN)

    pv_s = population_sample(
        pv_rows,
        PV_QUERY_IMPUTE_IDX,
        PV_STAT_LEN,
        len(uid_list),
    )
    fb_s = population_sample(
        fb_rows,
        FB_QUERY_IMPUTE_IDX,
        FB_STAT_LEN,
        len(uid_list),
    )
    pj_s = population_sample(
        pj_rows,
        PJ_QUERY_IMPUTE_IDX,
        PJ_STAT_LEN,
        len(uid_list),
    )

    for i, uid in enumerate(uid_list):
        p_env = wd / f"params_env_{uid}.json"
        if not p_env.is_file():
            continue

        with open(p_env, encoding="utf-8") as f:
            env_para = json.load(f)

        meta_root = env_para.get(_META_KEY)
        if not isinstance(meta_root, dict):
            meta_root = {}

        def load_base(key: str) -> tuple[np.ndarray, list]:
            t = np.asarray(env_para.get(key, []), dtype=float).ravel()
            nm = list(env_para.get(f"{key}_names", []))
            t_stripped, nm_stripped = _strip_imputed_suffix(t, nm)
            return t_stripped, nm_stripped

        old_pv, old_pv_names = load_base("theta_penalized_PV")
        old_fb, old_fb_names = load_base("theta_penalized_FW")
        old_pj, old_pj_names = load_base("theta_penalized_PJ")

        new_pv = pv_s[i].astype(float, copy=True)
        new_fb = fb_s[i].astype(float, copy=True)
        new_pj = pj_s[i].astype(float, copy=True)

        # Sign rules for the imputed query/action coefficients.
        # Current lengths:
        #   new_pv: 5 entries, valid indices 0..4
        #   new_fb: 4 entries, valid indices 0..3
        #   new_pj: 4 entries, valid indices 0..3

        # PV imputed suffix:
        # [query_intercept_like, query_Ew_like, query_weekend_like,
        #  query_decision_time_like, query_recent_burden_like]
        new_pv[0] = -np.abs(new_pv[0])
        new_pv[1] = np.abs(new_pv[1])
        new_pv[2] = -np.abs(new_pv[2])
        new_pv[3] = -np.abs(new_pv[3])
        new_pv[4] = -np.abs(new_pv[4])

        # FW imputed suffix:
        # [query_intercept_like, query_Ew_like, query_weekend_like,
        #  query_recent_burden_like]
        new_fb[0] = -np.abs(new_fb[0])
        new_fb[1] = np.abs(new_fb[1])
        new_fb[2] = -np.abs(new_fb[2])
        new_fb[3] = -np.abs(new_fb[3])

        # PJ imputed suffix:
        # [query_intercept_like, query_Ew_like, query_weekend_like,
        #  query_recent_burden_like]
        new_pj[0] = -np.abs(new_pj[0])
        new_pj[1] = np.abs(new_pj[1])
        new_pj[2] = -np.abs(new_pj[2])
        new_pj[3] = -np.abs(new_pj[3])

        env_para["theta_penalized_PV"] = np.round(
            np.concatenate([old_pv, new_pv]),
            digits,
        ).tolist()

        env_para["theta_penalized_FW"] = np.round(
            np.concatenate([old_fb, new_fb]),
            digits,
        ).tolist()

        env_para["theta_penalized_PJ"] = np.round(
            np.concatenate([old_pj, new_pj]),
            digits,
        ).tolist()

        env_para["theta_penalized_PV_names"] = old_pv_names + [
            "query_imputed_intercept_like",
            "query_imputed_Ew_like",
            "query_imputed_weekend_like",
            "query_imputed_decision_time_like",
            "query_imputed_recent_burden_like",
        ]

        env_para["theta_penalized_FW_names"] = old_fb_names + [
            "query_imputed_intercept_like",
            "query_imputed_Ew_like",
            "query_imputed_weekend_like",
            "query_imputed_recent_burden_like",
        ]

        env_para["theta_penalized_PJ_names"] = old_pj_names + [
            "query_imputed_intercept_like",
            "query_imputed_Ew_like",
            "query_imputed_weekend_like",
            "query_imputed_recent_burden_like",
        ]

        # Sanity-check that array lengths line up with the names arrays.
        for key, expected_k in (
            ("theta_penalized_PV", PV_IMPUTE_K),
            ("theta_penalized_FW", FB_IMPUTE_K),
            ("theta_penalized_PJ", PJ_IMPUTE_K),
        ):
            arr_len = len(env_para[key])
            nm_len = len(env_para[f"{key}_names"])
            if arr_len != nm_len:
                raise RuntimeError(
                    f"Participant {uid}: {key} length {arr_len} != "
                    f"{key}_names length {nm_len} after imputation "
                    f"(expected suffix k={expected_k})."
                )

        meta_root["theta_penalized_PV"] = {"k": PV_IMPUTE_K}
        meta_root["theta_penalized_FW"] = {"k": FB_IMPUTE_K}
        meta_root["theta_penalized_PJ"] = {"k": PJ_IMPUTE_K}
        env_para[_META_KEY] = meta_root

        with open(p_env, "w", encoding="utf-8") as f:
            json.dump(env_para, f, allow_nan=False)


def _terminal_predicted_mean(
    grid: np.ndarray,
    weights: np.ndarray,
    blocks: list,
    p_list: list[np.ndarray],
    par: dict,
) -> float:
    """One-step-ahead predictive mean E_{T+1} after the final filtered state."""
    if not blocks or not p_list:
        return float("nan")
    q, _ = _predict_density_log(
        grid,
        weights,
        p_list[-1],
        blocks[-1],
        par,
    )
    return float(np.sum(grid * q * weights))


def build_exported_Ew_series(filt: dict) -> np.ndarray:
    """Build E_2,...,E_{T+1} for export (filtered through E_T, terminal predictive)."""
    fm = np.asarray(filt.get("filtered_means", []), dtype=float)
    terminal = float(filt.get("terminal_predicted_mean", np.nan))
    if fm.size < 2:
        if np.isfinite(terminal):
            return np.array([terminal], dtype=float)
        return np.empty(0, dtype=float)
    out = np.empty(fm.size, dtype=float)
    out[:-1] = fm[1:]
    out[-1] = terminal
    return out


def transition_matrix(grid, prev_block, par):
    mu_prev = (
        par["a0"]
        + par["a1"] * grid
        + par["a2"] * prev_block["PV_sum_trans"]
        + par["a3"] * prev_block["FW_sum_trans"]
        + par["a4"] * prev_block["PJ_sum_trans"]
    )
    return normal_pdf(grid[:, None], mu_prev[None, :], par["sigma_E"])


def quadrature_loglik(
    blocks,
    theta: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    *,
    e1_known: Optional[float] = None,
) -> dict:
    """
    Approximate log L(eta) via quadrature (slides): for w>=2,
    hat c_w = sum_k ell_w(e_k) hat q_w(e_k) omega_k; log L ~= sum_w log hat c_w
    (with w=1 either ell_1(E_1) at a point or integrated against a prior on E_1).

    If e1_known is float: E_1 is the fixed baseline (week-1 start / lastweek);
    first term is log p(Y_1|E_1=e1) and hat q_2 transitions from that baseline
    to E_2 (end of week 1 / start of week 2). If None: week 1 uses prior
    N(m0,sigma0^2) on E_1.
    """
    e1_fixed = e1_known is not None
    par = unpack_theta(theta, e1_known=e1_fixed)
    grid = np.asarray(grid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    validate_quadrature_support(grid, weights, e1_known=e1_known)
    T = len(blocks)
    if T == 0:
        raise ValueError("At least one weekly block is required.")

    q_list = []
    p_list = []
    log_c_list = []
    pred_mean = np.empty(T)
    filt_mean = np.empty(T)
    predictive_grid_mass = np.full(T, np.nan)
    filtered_boundary_mass = np.full(T, np.nan)
    loglik = 0.0

    if e1_fixed:
        e1 = float(e1_known)
        log_ell1 = week_loglik_at_point(e1, blocks[0], par)
        if not np.isfinite(log_ell1):
            raise FloatingPointError("invalid week-1 log-likelihood at fixed E_1")
        loglik += log_ell1
        log_c_list.append(float(log_ell1))
        pred_mean[0] = np.nan
        filt_mean[0] = e1

        if T == 1:
            log_increments = np.asarray(log_c_list, dtype=float)
            terminal_predicted_mean = _terminal_predicted_mean(
                grid, weights, blocks, p_list, par
            )
            return {
                "loglik": float(loglik),
                "predicted_densities": np.zeros((0, len(grid))),
                "filtered_densities": np.zeros((0, len(grid))),
                "increments": np.exp(np.clip(log_increments, -745.0, 709.0)),
                "log_increments": log_increments,
                "predicted_means": pred_mean,
                "filtered_means": filt_mean,
                "terminal_predicted_mean": terminal_predicted_mean,
                "predictive_grid_mass": predictive_grid_mass,
                "filtered_boundary_mass": filtered_boundary_mass,
                "params": par,
                "e1_known": e1_known,
            }

        # Predictive density for week 2, obtained by propagating the fixed week-1
        # latent through the Gaussian transition.
        mu2 = (
            par["a0"]
            + par["a1"] * e1
            + par["a2"] * blocks[0]["PV_sum_trans"]
            + par["a3"] * blocks[0]["FW_sum_trans"]
            + par["a4"] * blocks[0]["PJ_sum_trans"]
        )
        q, log_grid_mass = _normalize_log_density(
            normal_logpdf(grid, mu2, par["sigma_E"]),
            weights,
            label="predictive E_2",
        )
        predictive_grid_mass[1] = np.exp(log_grid_mass)

        for t in range(1, T):
            q_list.append(q.copy())
            pred_mean[t] = np.sum(grid * q * weights)

            log_ell = week_loglik_on_grid(grid, blocks[t], par)
            m = np.max(log_ell)
            num = np.exp(log_ell - m) * q
            den = np.sum(num * weights)
            if (not np.isfinite(den)) or (den <= 0):
                raise FloatingPointError(f"invalid c_{t+1}")
            log_c = float(m + np.log(den))
            if not np.isfinite(log_c):
                raise FloatingPointError(f"invalid log c_{t+1}")

            p = num / den
            p_list.append(p.copy())
            log_c_list.append(log_c)
            filt_mean[t] = np.sum(grid * p * weights)
            filtered_boundary_mass[t] = np.sum(
                p[[0, -1]] * weights[[0, -1]]
            )
            loglik += log_c

            if t < T - 1:
                q, log_grid_mass = _predict_density_log(
                    grid,
                    weights,
                    p,
                    blocks[t],
                    par,
                )
                predictive_grid_mass[t + 1] = np.exp(log_grid_mass)
    else:
        q, log_grid_mass = _normalize_log_density(
            normal_logpdf(grid, par["m0"], par["sigma0"]),
            weights,
            label="E_1 prior",
        )
        predictive_grid_mass[0] = np.exp(log_grid_mass)
        for t in range(T):
            q_list.append(q.copy())
            pred_mean[t] = np.sum(grid * q * weights)

            log_ell = week_loglik_on_grid(grid, blocks[t], par)
            m = np.max(log_ell)
            num = np.exp(log_ell - m) * q
            den = np.sum(num * weights)
            if (not np.isfinite(den)) or (den <= 0):
                raise FloatingPointError(f"invalid c_{t+1}")
            log_c = float(m + np.log(den))
            if not np.isfinite(log_c):
                raise FloatingPointError(f"invalid log c_{t+1}")

            p = num / den
            p_list.append(p.copy())
            log_c_list.append(log_c)
            filt_mean[t] = np.sum(grid * p * weights)
            filtered_boundary_mass[t] = np.sum(
                p[[0, -1]] * weights[[0, -1]]
            )
            loglik += log_c

            if t < T - 1:
                q, log_grid_mass = _predict_density_log(
                    grid,
                    weights,
                    p,
                    blocks[t],
                    par,
                )
                predictive_grid_mass[t + 1] = np.exp(log_grid_mass)

    log_increments = np.asarray(log_c_list, dtype=float)
    terminal_predicted_mean = _terminal_predicted_mean(
        grid, weights, blocks, p_list, par
    )
    return {
        "loglik": float(loglik),
        "predicted_densities": np.vstack(q_list),
        "filtered_densities": np.vstack(p_list),
        # ``increments`` is retained for compatibility; use log_increments for
        # calculations because the probability-scale values may underflow.
        "increments": np.exp(np.clip(log_increments, -745.0, 709.0)),
        "log_increments": log_increments,
        "predicted_means": pred_mean,
        "filtered_means": filt_mean,
        "terminal_predicted_mean": terminal_predicted_mean,
        "predictive_grid_mass": predictive_grid_mass,
        "filtered_boundary_mass": filtered_boundary_mass,
        "params": par,
        "e1_known": e1_known,
    }



def neg_loglik_blocks(theta, blocks, grid, weights, e1_known, lam=0.5):
    try:
        out = quadrature_loglik(blocks, theta, grid, weights, e1_known=e1_known)
        if not np.isfinite(out["loglik"]):
            return 1e100
        penalty = lam * np.sum(theta ** 2)
        return -out["loglik"] + penalty
    except FloatingPointError:
        return 1e100

def neg_loglik_unpenalized_blocks(theta, blocks, grid, weights, e1_known):
    try:
        out = quadrature_loglik(blocks, theta, grid, weights, e1_known=e1_known)
        if not np.isfinite(out["loglik"]):
            return 1e100
        return -out["loglik"]
    except FloatingPointError:
        return 1e100


def neg_loglik_all_users(theta, user_blocks, grid, weights, e1_known, lam=0.5):
    total_nll = 0.0

    for uid, blocks in user_blocks.items():
        val = neg_loglik_unpenalized_blocks(
            theta, blocks, grid, weights, e1_known
        )
        if not np.isfinite(val):
            return 1e100
        total_nll += val

    # Apply ridge penalty once, not once per participant
    penalty = lam * np.sum(theta ** 2)
    return total_nll + penalty


class _PooledFitProgress:
    """Track and print pooled L-BFGS-B progress (objective is very expensive)."""

    def __init__(
        self,
        user_blocks,
        grid,
        weights,
        e1_known,
        lam,
        *,
        e1_fixed: bool,
        log_every_evals: int = 5,
        min_log_interval_s: float = 15.0,
    ):
        self._args = (user_blocks, grid, weights, e1_known, lam)
        self._e1_fixed = e1_fixed
        self._log_every_evals = max(1, int(log_every_evals))
        self._min_log_interval_s = float(min_log_interval_s)
        self.n_eval = 0
        self.n_iter = 0
        self.t0 = time.perf_counter()
        self._last_log_t = self.t0
        self.last_nll = float("inf")
        self.best_nll = float("inf")
        self.best_theta: Optional[np.ndarray] = None

    def objective(self, theta: np.ndarray) -> float:
        self.n_eval += 1
        nll = neg_loglik_all_users(theta, *self._args)
        self.last_nll = float(nll)
        if nll < self.best_nll:
            self.best_nll = nll
            self.best_theta = np.asarray(theta, dtype=float).copy()

        now = time.perf_counter()
        should_log = (
            self.n_eval == 1
            or self.n_eval % self._log_every_evals == 0
            or (now - self._last_log_t) >= self._min_log_interval_s
        )
        if should_log:
            self._log(theta, nll, now, kind="eval")
            self._last_log_t = now
        return nll

    def callback(self, theta: np.ndarray) -> bool:
        self.n_iter += 1
        now = time.perf_counter()
        self._log(theta, self.last_nll, now, kind="iter")
        self._last_log_t = now
        return False

    def _log(self, theta: np.ndarray, nll: float, now: float, *, kind: str) -> None:
        elapsed = now - self.t0
        par = unpack_theta(theta, e1_known=self._e1_fixed)
        if kind == "eval":
            prefix = f"pooled eval {self.n_eval:4d}"
        else:
            prefix = f"pooled iter {self.n_iter:3d} (eval {self.n_eval:4d})"
        logger.info(
            "%s | elapsed %5.1fs | nll=%.4f | best=%.4f | "
            "a0=%.3g a1=%.3g sigE=%.3g b1=%.3g alpha1=%.3g beta1=%.3g",
            prefix,
            elapsed,
            nll,
            self.best_nll,
            par["a0"],
            par["a1"],
            par["sigma_E"],
            par["b1"],
            par["alpha1"],
            par["beta1"],
        )

    def finish(self) -> None:
        elapsed = time.perf_counter() - self.t0
        logger.info(
            "Pooled optimization finished in %.1fs (%d func evals, %d iterations, best nll=%.4f)",
            elapsed,
            self.n_eval,
            self.n_iter,
            self.best_nll,
        )


def _summarize_pooled_workload(user_blocks, grid) -> dict:
    n_users = len(user_blocks)
    n_weeks = sum(len(blocks) for blocks in user_blocks.values())
    n_pv_rows = sum(
        len(b["pv_y"])
        for blocks in user_blocks.values()
        for b in blocks
    )
    return {
        "n_users": n_users,
        "n_weeks": n_weeks,
        "n_pv_rows": n_pv_rows,
        "grid_size": len(grid),
    }


def make_bounds(*, e1_known: bool):
    bounds = [(None, None)] * theta_dim(e1_known=e1_known)

    # State AR coefficient a1
    bounds[1] = (-0.98, 0.98)

    # log sigma parameters:
    # 5: sigma_E, 10: sigma_U1, 13: sigma_U2, 21: sigma_PV
    for idx in [5, 10, 13, 21]:
        bounds[idx] = (np.log(0.03), np.log(10.0))

    # If E1 prior is estimated, bound log sigma0 too
    if not e1_known:
        bounds[42] = (np.log(0.03), np.log(10.0))

    return bounds

def _hit_iteration_limit(res, maxiter: int) -> bool:
    """True if L-BFGS-B stopped because it ran out of iterations (not because
    it actually satisfied ftol/gtol). ``res.success`` is False in that case,
    and either ``nit`` reached the cap or SciPy's message says so explicitly.
    """
    if res.success:
        return False
    msg = str(getattr(res, "message", ""))
    return int(getattr(res, "nit", 0)) >= int(maxiter) or "ITERATIONS REACHED LIMIT" in msg.upper()


def _minimize_with_restarts(
    objective,
    x0,
    *,
    args,
    bounds,
    maxiter: int,
    maxfun: int,
    maxls: int = 50,
    ftol: float,
    gtol: float,
    callback=None,
    max_restarts: int = 4,
    restart_multiplier: float = 2.0,
    log_prefix: str = "",
):
    """Run L-BFGS-B, and if it stops purely because it hit ``maxiter`` (i.e.
    it was still improving, not actually converged), warm-restart from the
    last iterate with a larger iteration budget. This fixes participants
    that get flagged as "not converged" just because a fixed maxiter (e.g.
    500) was too small for them, without wasting extra iterations on
    participants who already converge quickly.
    """
    current_maxiter = int(maxiter)
    res = minimize(
        objective,
        x0,
        args=args,
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": current_maxiter,
            "maxfun": maxfun,
            "maxls": maxls,
            "ftol": ftol,
            "gtol": gtol,
        },
    )

    restarts = 0
    while _hit_iteration_limit(res, current_maxiter) and restarts < max_restarts:
        restarts += 1
        current_maxiter = int(round(current_maxiter * restart_multiplier))
        logger.info(
            "%sL-BFGS-B hit maxiter without converging (nit=%d, |grad|_inf=%s); "
            "warm-restarting from last iterate with maxiter=%d (restart %d/%d).",
            log_prefix,
            int(res.nit),
            f"{np.linalg.norm(res.jac, ord=np.inf):.3g}" if getattr(res, "jac", None) is not None else "n/a",
            current_maxiter,
            restarts,
            max_restarts,
        )
        res = minimize(
            objective,
            res.x,
            args=args,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options={
                "maxiter": current_maxiter,
                "maxfun": maxfun,
                "maxls": maxls,
                "ftol": ftol,
                "gtol": gtol,
            },
        )

    if restarts:
        logger.info(
            "%sfinished after %d warm restart(s): success=%s, nit=%d, final maxiter=%d.",
            log_prefix,
            restarts,
            res.success,
            int(res.nit),
            current_maxiter,
        )
    return res, restarts


def fit_pooled_model(
    df_fit,
    grid=None,
    *,
    e1_known: Optional[float] = None,
    maxiter=500,
    max_restarts: int = 4,
    restart_multiplier: float = 2.0,
    week_col="week",
    date_col="Date",
    decision_col="DecisionTime",
    J_col="week_present",
    U1_col="Exp-tool-1_norm",
    U2_col="Exp-tool-2_norm",
    FW_col="nextday_wearing",
    PJ_col="daily_present",
    a_col="WalkingSuggestion",
    weekend_col="is_weekend",
    burden_col="recent_burden_norm",
    PV_col="HourlyPageviewCount_norm",
    PV_lag1_col="hourly_pageview_count_lag1",
    FW_lag_col="morning_wearing",
    PJ_lag_col="daily_present_yesterday",
    hourly_pv=True,
    lam: float = 0.5,
    progress: bool = True,
    progress_every_evals: int = 5,
    progress_min_interval_s: float = 15.0,
):
    if grid is None:
        grid = np.linspace(-3.0, 3.0, 121)

    e1_fixed = e1_known is not None
    weights = trapezoid_weights(grid)
    validate_quadrature_support(grid, weights, e1_known=e1_known)
    user_blocks = prepare_all_user_blocks(
        df_fit,
        week_col=week_col,
        date_col=date_col,
        decision_col=decision_col,
        J_col=J_col,
        U1_col=U1_col,
        U2_col=U2_col,
        FW_col=FW_col,
        PJ_col=PJ_col,
        a_col=a_col,
        weekend_col=weekend_col,
        burden_col=burden_col,
        PV_col=PV_col,
        PV_lag1_col=PV_lag1_col,
        FW_lag_col=FW_lag_col,
        PJ_lag_col=PJ_lag_col,
        hourly_pv=hourly_pv,
    )

    # initialize from all blocks pooled together
    all_blocks = []
    for blocks in user_blocks.values():
        all_blocks.extend(blocks)

    x0 = initial_theta_from_blocks(all_blocks, e1_known=e1_fixed)

    workload = _summarize_pooled_workload(user_blocks, grid)
    theta_d = theta_dim(e1_known=e1_fixed)
    logger.info(
        "Starting pooled fit: users=%d, participant-weeks=%d, hourly-PV rows=%d, "
        "grid=%d points, theta dim=%d, maxiter=%d",
        workload["n_users"],
        workload["n_weeks"],
        workload["n_pv_rows"],
        workload["grid_size"],
        theta_d,
        maxiter,
    )
    logger.info(
        "Why this stage is slow: each optimizer evaluation runs a 12-week "
        "quadrature filter for all %d users (~%d grid-point updates per user-week, "
        "plus hourly PV / daily FW-PJ likelihoods). L-BFGS-B has no analytic "
        "gradient here, so SciPy uses finite differences (~%d+ extra evals per "
        "iteration). Expect many minutes before the first iteration line appears "
        "if logging is sparse.",
        workload["n_users"],
        workload["grid_size"],
        theta_d,
    )

    if progress:
        tracker = _PooledFitProgress(
            user_blocks,
            grid,
            weights,
            e1_known,
            lam,
            e1_fixed=e1_fixed,
            log_every_evals=progress_every_evals,
            min_log_interval_s=progress_min_interval_s,
        )
        objective = tracker.objective
        callback = tracker.callback
    else:
        tracker = None
        objective = neg_loglik_all_users
        callback = None

    logger.info("Pooled fit: beginning L-BFGS-B (first eval starting now)...")
    res, restarts = _minimize_with_restarts(
        objective,
        x0,
        args=(user_blocks, grid, weights, e1_known, lam) if not progress else (),
        bounds=make_bounds(e1_known=e1_fixed),
        callback=callback,
        maxiter=maxiter,
        maxfun=1_500_000,
        maxls=50,
        ftol=1e-7,
        gtol=1e-4,
        max_restarts=max_restarts,
        restart_multiplier=restart_multiplier,
        log_prefix="Pooled fit: ",
    )
    if tracker is not None:
        tracker.finish()

    print(
        f"Pooled fit: success={res.success}, "
        f"nll={res.fun:.4f}, nit={res.nit}, message={res.message}, "
        f"warm_restarts={restarts}"
    )

    pooled_theta = np.asarray(res.x, dtype=float).copy()
    log_pooled_theta(
        pooled_theta,
        e1_known=e1_fixed,
        nll=float(res.fun),
        success=bool(res.success),
        nit=int(res.nit),
    )

    return res, user_blocks

def fit_one_user(
    dat_user,
    grid,
    *,
    e1_known: Optional[float] = None,
    maxiter=1000,
    max_restarts: int = 4,
    restart_multiplier: float = 2.0,
    week_col="week",
    date_col="Date",
    decision_col="DecisionTime",
    J_col="week_present",
    U1_col="Exp-tool-1_norm",
    U2_col="Exp-tool-2_norm",
    FW_col="nextday_wearing",
    PJ_col="daily_present",
    a_col="WalkingSuggestion",
    weekend_col="is_weekend",
    burden_col="recent_burden_norm",
    PV_col="HourlyPageviewCount_norm",
    PV_lag1_col="hourly_pageview_count_lag1",
    FW_lag_col="morning_wearing",
    PJ_lag_col="daily_present_yesterday",
    hourly_pv=True,
    x0: Optional[np.ndarray] = None,
    lam: float = 0.5,
    uid: Optional[str] = None,
):
    blocks = build_user_blocks(
        dat_user,
        week_col=week_col,
        date_col=date_col,
        decision_col=decision_col,
        J_col=J_col,
        U1_col=U1_col,
        U2_col=U2_col,
        FW_col=FW_col,
        PJ_col=PJ_col,
        a_col=a_col,
        weekend_col=weekend_col,
        burden_col=burden_col,
        PV_col=PV_col,
        PV_lag1_col=PV_lag1_col,
        FW_lag_col=FW_lag_col,
        PJ_lag_col=PJ_lag_col,
        hourly_pv=hourly_pv,
        full_weeks=range(1, 13),
        missing_week_mode="mean_impute_transition",
    )
    weights = trapezoid_weights(grid)
    validate_quadrature_support(grid, weights, e1_known=e1_known)
    e1_fixed = e1_known is not None

    if x0 is None:
        x0 = initial_theta_from_blocks(blocks, e1_known=e1_fixed)
    else:
        x0 = np.asarray(x0, dtype=float).copy()

    res, restarts = _minimize_with_restarts(
        neg_loglik_blocks,
        x0,
        args=(blocks, grid, weights, e1_known, lam),
        bounds=make_bounds(e1_known=e1_fixed),
        maxiter=maxiter,
        maxfun=300000,
        maxls=50,
        ftol=1e-6,
        gtol=1e-4,
        max_restarts=max_restarts,
        restart_multiplier=restart_multiplier,
        log_prefix=f"Participant {uid}: " if uid is not None else "",
    )
    res.warm_restarts = restarts
    filt = quadrature_loglik(blocks, res.x, grid, weights, e1_known=e1_known)
    filt["ridge_lambda"] = float(lam)

    retained = np.asarray(filt.get("predictive_grid_mass", []), dtype=float)
    retained = retained[np.isfinite(retained)]
    boundary = np.asarray(filt.get("filtered_boundary_mass", []), dtype=float)
    boundary = boundary[np.isfinite(boundary)]
    if retained.size and np.min(retained) < 0.99:
        warnings.warn(
            f"Participant grid retained as little as {np.min(retained):.3%} "
            "of predictive transition mass; widen the E grid.",
            RuntimeWarning,
            stacklevel=2,
        )
    if boundary.size and np.max(boundary) > 0.05:
        warnings.warn(
            f"Participant filtered boundary mass reached {np.max(boundary):.3%}; "
            "the E grid may be too narrow.",
            RuntimeWarning,
            stacklevel=2,
        )
    return res, filt, blocks


def _fit_one_user_task(uid, dat_user, grid, fit_kwargs):
    """Pickle-friendly process-pool wrapper for one participant fit."""
    res, filt, blocks = fit_one_user(dat_user, grid=grid, uid=uid, **fit_kwargs)
    return uid, res, filt, blocks


def fit_all_users(
    df_fit,
    grid=None,
    *,
    e1_known: Optional[float] = None,
    maxiter=500,
    max_restarts: int = 4,
    restart_multiplier: float = 2.0,
    week_col="week",
    date_col="Date",
    decision_col="DecisionTime",
    J_col="week_present",
    U1_col="Exp-tool-1_norm",
    U2_col="Exp-tool-2_norm",
    FW_col="nextday_wearing",
    PJ_col="daily_present",
    a_col="WalkingSuggestion",
    weekend_col="is_weekend",
    burden_col="recent_burden_norm",
    PV_col="HourlyPageviewCount_norm",
    PV_lag1_col="hourly_pageview_count_lag1",
    FW_lag_col="morning_wearing",
    PJ_lag_col="daily_present_yesterday",
    hourly_pv=True,
    pooled_x0: Optional[np.ndarray] = None,
    lam: float = 0.5,
    merge_into_vanilla_json: bool = False,
    vanilla_work_dir: Optional[Path] = None,
    json_digits: int = 3,
    n_jobs: Optional[int] = None,
):
    if grid is None:
        grid = np.linspace(-3.0, 3.0, 241)

    user_data = [
        (
            uid,
            g.sort_values(
                ["week", "Date", "DecisionTime"], na_position="last"
            ).reset_index(drop=True),
        )
        for uid, g in df_fit.groupby("ParticipantIdentifier")
    ]
    if n_jobs is None:
        available_cpus = max(1, (os.cpu_count() or 2) - 1)
        n_jobs = min(8, available_cpus, len(user_data))
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1.")

    fit_kwargs = {
        "e1_known": e1_known,
        "maxiter": maxiter,
        "max_restarts": max_restarts,
        "restart_multiplier": restart_multiplier,
        "week_col": week_col,
        "date_col": date_col,
        "decision_col": decision_col,
        "J_col": J_col,
        "U1_col": U1_col,
        "U2_col": U2_col,
        "FW_col": FW_col,
        "PJ_col": PJ_col,
        "a_col": a_col,
        "weekend_col": weekend_col,
        "burden_col": burden_col,
        "PV_col": PV_col,
        "PV_lag1_col": PV_lag1_col,
        "FW_lag_col": FW_lag_col,
        "PJ_lag_col": PJ_lag_col,
        "hourly_pv": hourly_pv,
        "x0": pooled_x0,
        "lam": lam,
    }
    results = {}
    filtered_states = {}
    blocks_by_user = {}

    def record_fit(uid, res, filt, blocks):
        results[uid] = res
        filtered_states[uid] = filt
        blocks_by_user[uid] = blocks
        print(
            f"Participant {uid}: success={res.success}, "
            f"nll={res.fun:.4f}, nit={res.nit}, message={res.message}, "
            f"warm_restarts={getattr(res, 'warm_restarts', 0)}"
        )

        if merge_into_vanilla_json:
            try:
                write_joint_penalized_into_vanilla_json_files(
                    uid,
                    res,
                    filt,
                    blocks,
                    work_dir=vanilla_work_dir,
                    digits=json_digits,
                )
            except Exception as exc:
                print(
                    f"Participant {uid}: could not merge penalized fit into "
                    f"vanilla JSON: {exc}"
                )

    print(f"Fitting {len(user_data)} participants with {n_jobs} worker process(es).")
    if n_jobs == 1:
        for uid, g in user_data:
            record_fit(*_fit_one_user_task(uid, g, grid, fit_kwargs))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    _fit_one_user_task, uid, g, grid, fit_kwargs
                ): uid
                for uid, g in user_data
            }
            for future in as_completed(futures):
                record_fit(*future.result())

    return results, filtered_states, blocks_by_user

def prepare_all_user_blocks(
    df_fit,
    *,
    week_col="week",
    date_col="Date",
    decision_col="DecisionTime",
    J_col="week_present",
    U1_col="Exp-tool-1_norm",
    U2_col="Exp-tool-2_norm",
    FW_col="nextday_wearing",
    PJ_col="daily_present",
    a_col="WalkingSuggestion",
    weekend_col="is_weekend",
    burden_col="recent_burden_norm",
    PV_col="HourlyPageviewCount_norm",
    PV_lag1_col="hourly_pageview_count_lag1",
    FW_lag_col="morning_wearing",
    PJ_lag_col="daily_present_yesterday",
    hourly_pv=True,
):
    user_blocks = {}
    for uid, g in df_fit.groupby("ParticipantIdentifier"):
        g = g.sort_values(["week", "Date", "DecisionTime"], na_position="last").reset_index(drop=True)
        blocks = build_user_blocks(
            g,
            week_col=week_col,
            date_col=date_col,
            decision_col=decision_col,
            J_col=J_col,
            U1_col=U1_col,
            U2_col=U2_col,
            FW_col=FW_col,
            PJ_col=PJ_col,
            a_col=a_col,
            weekend_col=weekend_col,
            burden_col=burden_col,
            PV_col=PV_col,
            PV_lag1_col=PV_lag1_col,
            FW_lag_col=FW_lag_col,
            PJ_lag_col=PJ_lag_col,
            hourly_pv=hourly_pv,
            full_weeks=range(1, 13),
            missing_week_mode="mean_impute_transition",
        )
        user_blocks[uid] = blocks
    return user_blocks


def plot_quadrature_diagnostics(
    results,
    filtered_states,
    *,
    plot_dir: Optional[Path] = None,
    show: bool = True,
):
    """
    Bar and trajectory figures analogous to 1.6_derive_perceived_utility.py, but
    using quadrature-filtered state means and MLE parameter dicts from this module.
    """
    _users = sorted(filtered_states.keys(), key=lambda x: (str(type(x)), str(x)))
    if len(_users) == 0:
        return

    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

    _n = len(_users)
    _n_cols = min(4, max(1, _n))
    _n_rows_ew = max(1, ceil(_n / _n_cols))
    _x = np.arange(len(_users))

    def _bar_users_one_series(ax, ylabel, title, vals):
        ax.bar(_x, vals, width=0.75)
        ax.set_xticks(_x)
        ax.set_xticklabels([str(u) for u in _users], rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)

    def _maybe_save_close(fig, name):
        if plot_dir is not None:
            fig.savefig(plot_dir / name, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # --- Filtered mean E_w trajectories ---
    fig_ew, axes_ew = plt.subplots(
        _n_rows_ew, _n_cols, figsize=(3.6 * _n_cols, 2.3 * _n_rows_ew), squeeze=False
    )
    for ax in np.ravel(axes_ew):
        ax.set_visible(False)
    for idx, uid in enumerate(_users):
        r, c = divmod(idx, _n_cols)
        ax = axes_ew[r][c]
        ax.set_visible(True)
        pu = build_exported_Ew_series(filtered_states[uid])
        w = np.arange(1, 1 + len(pu))
        ax.plot(w, pu, marker="o", ms=2, lw=1)
        ax.set_title(f"Participant {uid}", fontsize=9)
        ax.set_xlabel(r"$w$", fontsize=8)
        ax.set_ylabel(r"$E_w$", fontsize=8)
    fig_ew.suptitle(
        r"Exported $\hat E_{w+1}$ by study week $w$=1..12 (filtered; last point predictive $E_{13}$)",
        fontsize=11,
    )
    fig_ew.tight_layout()
    _maybe_save_close(fig_ew, "Ew_filtered_mean_by_user.png")

    # --- State dynamics a_0 .. a_4 ---
    fig_a, axes_a = plt.subplots(1, 5, figsize=(16, 3.2), squeeze=False)
    _a_specs = [
        (r"$a_0$", lambda p: float(p["a0"])),
        (r"$a_1$", lambda p: float(p["a1"])),
        (r"$a_2\,(\mathrm{PV})$", lambda p: float(p["a2"])),
        (r"$a_3\,(\mathrm{FW})$", lambda p: float(p["a3"])),
        (r"$a_4\,(\mathrm{PJ})$", lambda p: float(p["a4"])),
    ]
    for k, (tit, fn) in enumerate(_a_specs):
        vals = [fn(filtered_states[u]["params"]) for u in _users]
        _bar_users_one_series(axes_a[0][k], "Estimate", tit, vals)
    fig_a.suptitle(
        r"State: $E_w = a_0 + a_1 E_{w-1} + a_2 \mathrm{PV} + a_3 \mathrm{FW} + a_4 \mathrm{PJ} + \epsilon_w$",
        fontsize=11,
        y=1.08,
    )
    fig_a.tight_layout()
    _maybe_save_close(fig_a, "params_group_a.png")

    # --- sigma_E ---
    fig_s, ax_s = plt.subplots(1, 1, figsize=(max(7.0, len(_users) * 0.35), 3.2))
    _bar_users_one_series(
        ax_s,
        "Estimate",
        r"$\sigma_E$ — $\epsilon_w \sim \mathcal{N}(0,\sigma_E^2)$",
        [float(filtered_states[u]["params"]["sigma_E"]) for u in _users],
    )
    fig_s.tight_layout()
    _maybe_save_close(fig_s, "params_group_sigma.png")

    # --- b_0, b_1 (J_week) ---
    fig_b, axes_b = plt.subplots(1, 2, figsize=(10, 3.2), squeeze=False)
    _bar_users_one_series(
        axes_b[0][0],
        "Estimate",
        r"$b_0$ — $\mathrm{logit}(p_w)= b_0 + b_1 E_w$",
        [float(filtered_states[u]["params"]["b0"]) for u in _users],
    )
    _bar_users_one_series(
        axes_b[0][1],
        "Estimate",
        r"$b_1$ — coefficient on $E_w$ for $J_w^{\mathrm{week}}$",
        [float(filtered_states[u]["params"]["b1"]) for u in _users],
    )
    fig_b.suptitle(r"Bernoulli emission ($J_w^{\mathrm{week}}$)", fontsize=11, y=1.05)
    fig_b.tight_layout()
    _maybe_save_close(fig_b, "params_group_b.png")

    # --- c_0, c_1 (U_1) ---
    fig_c, axes_c = plt.subplots(1, 2, figsize=(10, 3.2), squeeze=False)
    _bar_users_one_series(
        axes_c[0][0],
        "Estimate",
        r"$c_0$ — $U_{w,1}$ intercept",
        [float(filtered_states[u]["params"]["c0"]) for u in _users],
    )
    _bar_users_one_series(
        axes_c[0][1],
        "Estimate",
        r"$c_1$ — coefficient on $E_w$ for $U_{w,1}$",
        [float(filtered_states[u]["params"]["c1"]) for u in _users],
    )
    fig_c.suptitle(r"Gaussian $U_{w,1}$ (helpfulness)", fontsize=11, y=1.05)
    fig_c.tight_layout()
    _maybe_save_close(fig_c, "params_group_c.png")

    # --- d_0, d_1 (U_2) ---
    fig_d, axes_d = plt.subplots(1, 2, figsize=(10, 3.2), squeeze=False)
    _bar_users_one_series(
        axes_d[0][0],
        "Estimate",
        r"$d_0$ — $U_{w,2}$ intercept",
        [float(filtered_states[u]["params"]["d0"]) for u in _users],
    )
    _bar_users_one_series(
        axes_d[0][1],
        "Estimate",
        r"$d_1$ — coefficient on $E_w$ for $U_{w,2}$",
        [float(filtered_states[u]["params"]["d1"]) for u in _users],
    )
    fig_d.suptitle(r"Gaussian $U_{w,2}$ (pleasantness)", fontsize=11, y=1.05)
    fig_d.tight_layout()
    _maybe_save_close(fig_d, "params_group_d.png")

    # --- log(sigma_U^2) ---
    fig_n, axes_n = plt.subplots(1, 2, figsize=(10, 3.2), squeeze=False)
    _bar_users_one_series(
        axes_n[0][0],
        "Estimate",
        r"$\log(\sigma_{U_1}^2)$",
        [
            float(np.log(filtered_states[u]["params"]["sigma_U1"] ** 2))
            for u in _users
        ],
    )
    _bar_users_one_series(
        axes_n[0][1],
        "Estimate",
        r"$\log(\sigma_{U_2}^2)$",
        [
            float(np.log(filtered_states[u]["params"]["sigma_U2"] ** 2))
            for u in _users
        ],
    )
    fig_n.suptitle(r"Gaussian emission noise (weekly surveys)", fontsize=11, y=1.05)
    fig_n.tight_layout()
    _maybe_save_close(fig_n, "params_group_log_sigma2.png")

    # --- PV model: alpha, AR(1), sigma_PV ---
    fig_pv, axes_pv = plt.subplots(3, 3, figsize=(12, 8.0), squeeze=False)
    flat = np.ravel(axes_pv)
    _pv_specs = [
        (r"$\alpha_0$", lambda p: p["alpha0"]),
        (r"$\alpha_1$ ($E_w$)", lambda p: p["alpha1"]),
        (r"$\alpha_{2,\mathrm{we}}$", lambda p: p["alpha2_is_weekend"]),
        (r"$\alpha_{2,\mathrm{dt}}$", lambda p: p["alpha2_dt"]),
        (r"$\alpha_{2,\mathrm{rb}}$", lambda p: p["alpha2_rb"]),
        (
            r"$\alpha_{\mathrm{AR}}$ (hourly\_pageview\_count\_lag1)",
            lambda p: p["alpha_ar1"],
        ),
        (r"$\alpha_3$ ($A$)", lambda p: p["alpha3"]),
        (r"$\alpha_4$ ($A\cdot E_w$)", lambda p: p["alpha4"]),
        (r"$\log(\sigma_{\mathrm{PV}}^2)$", lambda p: np.log(p["sigma_PV"] ** 2)),
    ]
    for k, (tit, fn) in enumerate(_pv_specs):
        vals = [float(fn(filtered_states[u]["params"])) for u in _users]
        _bar_users_one_series(flat[k], "Estimate", tit, vals)
    fig_pv.suptitle(
        r"Hourly pageview (Gaussian) — context, lag1 column, action", fontsize=11, y=1.02
    )
    fig_pv.tight_layout()
    _maybe_save_close(fig_pv, "params_group_pv.png")

    # --- FW / PJ: intercept, E_w, is_weekend, burden, AR(1) within week ---
    fig_ft, axes_ft = plt.subplots(2, 5, figsize=(16, 6.0), squeeze=False)
    _bar_users_one_series(
        axes_ft[0][0],
        "Estimate",
        r"$\beta_0$ (FW logit intercept)",
        [float(filtered_states[u]["params"]["beta0"]) for u in _users],
    )
    _bar_users_one_series(
        axes_ft[0][1],
        "Estimate",
        r"$\beta_1$ (FW $E_w$)",
        [float(filtered_states[u]["params"]["beta1"]) for u in _users],
    )
    _bar_users_one_series(
        axes_ft[0][2],
        "Estimate",
        r"$\beta_{2,\mathrm{we}}$ (FW)",
        [float(filtered_states[u]["params"]["beta2_is_weekend"]) for u in _users],
    )
    _bar_users_one_series(
        axes_ft[0][3],
        "Estimate",
        r"$\beta_{2,\mathrm{rb}}$ (FW)",
        [float(filtered_states[u]["params"]["beta2_rb"]) for u in _users],
    )
    _bar_users_one_series(
        axes_ft[0][4],
        "Estimate",
        r"$\beta_{\mathrm{AR}}$ (morning\_wearing)",
        [float(filtered_states[u]["params"]["beta_ar1"]) for u in _users],
    )
    _bar_users_one_series(
        axes_ft[1][0],
        "Estimate",
        r"$\theta_0$ (PJ logit intercept)",
        [float(filtered_states[u]["params"]["theta0"]) for u in _users],
    )
    _bar_users_one_series(
        axes_ft[1][1],
        "Estimate",
        r"$\theta_1$ (PJ $E_w$)",
        [float(filtered_states[u]["params"]["theta1"]) for u in _users],
    )
    _bar_users_one_series(
        axes_ft[1][2],
        "Estimate",
        r"$\theta_{2,\mathrm{we}}$ (PJ)",
        [float(filtered_states[u]["params"]["theta2_is_weekend"]) for u in _users],
    )
    _bar_users_one_series(
        axes_ft[1][3],
        "Estimate",
        r"$\theta_{2,\mathrm{rb}}$ (PJ)",
        [float(filtered_states[u]["params"]["theta2_rb"]) for u in _users],
    )
    _bar_users_one_series(
        axes_ft[1][4],
        "Estimate",
        r"$\theta_{\mathrm{AR}}$ (daily\_present\_yesterday)",
        [float(filtered_states[u]["params"]["theta_ar1"]) for u in _users],
    )
    fig_ft.suptitle(
        r"Daily FW / PJ logits — intercept, $E_w$, is_weekend, burden, lag covariates",
        fontsize=11,
        y=1.02,
    )
    fig_ft.tight_layout()
    _maybe_save_close(fig_ft, "params_group_fw_pj.png")

    # --- Optimization status ---
    fig_z, ax_z = plt.subplots(1, 1, figsize=(max(7.0, len(_users) * 0.35), 3.0))
    conv = [1.0 if results[u].success else 0.0 for u in _users]
    _bar_users_one_series(ax_z, "1 = success", "L-BFGS-B convergence", conv)
    fig_z.tight_layout()
    _maybe_save_close(fig_z, "optimization_success.png")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Stage 1: pooled fit on a coarser grid
    pooled_res, pooled_blocks = fit_pooled_model(
        df_fit,
        grid=np.linspace(-3.0, 3.0, 121),
        e1_known=2.0,
        maxiter=500,
        lam=0.5,
    )

    pooled_theta = pooled_res.x.copy()
    pooled_x0 = pooled_theta.copy()

    print("pooled success:", pooled_res.success)
    print("pooled message:", pooled_res.message)
    print("pooled nfev:", pooled_res.nfev)
    print("pooled njev:", getattr(pooled_res, "njev", None))
    print("pooled nit:", pooled_res.nit)
    print("pooled grad inf-norm:", np.linalg.norm(pooled_res.jac, ord=np.inf))
    print("pooled nll:", pooled_res.fun)

    if pooled_res.success:
        print("Pooled fit converged.")
    elif np.linalg.norm(pooled_res.jac, ord=np.inf) < 1e-3:
        print("Probably close enough as a warm start, but not as a final pooled MLE.")
    else:
        print("Pooled fit did not converge; consider adjusting bounds / maxfun / ftol.")

    # Stage 2: user-specific fits initialized at pooled estimate
    _e1_known = 2.0
    results, filtered_states, blocks_by_user = fit_all_users(
        df_fit,
        grid=np.linspace(-3.0, 3.0, 241),
        e1_known=_e1_known,
        maxiter=500,
        pooled_x0=pooled_x0,
        lam=0.5,
        merge_into_vanilla_json=True,
        vanilla_work_dir=WORK_DIR,
        json_digits=3,
        n_jobs=None,
    )

    # Write filtered E_w (perceived utility) back to df_fit.csv so that
    # 2_fit_vanilla_testbed.py can read perceived_utility / perceived_utility_lastweek.
    save_df_fit_with_perceived_utility(
        df_fit,
        filtered_states,
        out_path=folder / "df_fit.csv",
        e1_known=_e1_known,
    )

    _diag_dir = WORK_DIR / "plots_penalized_perceived_utility"
    plot_quadrature_diagnostics(
        results,
        filtered_states,
        plot_dir=_diag_dir,
        show=True,
    )
    print(f"Diagnostic figures saved under {_diag_dir}")

    impute_query_action_interaction_effects(
        df_fit["ParticipantIdentifier"].unique(),
        work_dir=WORK_DIR,
        xi=1.0 / 8.0,
        digits=3,
        rng=np.random.default_rng(2026),
    )
    print("Imputed query-action interaction blocks into params_env_<id>.json (see _query_interaction_impute).")
