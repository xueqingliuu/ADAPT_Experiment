# %%
"""
Joint state-space model for perceived utility E_w with weekly AR dynamics and
within-week outcomes (J_week, U1, U2, hourly PV, daily FW, daily PJ).

Likelihood for latent E_2,...,E_W is approximated by 1D quadrature (trapezoid
grid) with prediction / filtering recursion as on the slides. Optionally E_1 is
fixed (known); otherwise a Gaussian prior on E_1 at week 1 is estimated.

Observation models (conditional on latent E_w for that calendar week; ``grid`` in
code is quadrature over E_w). Weekly survey outcomes:

  - J_week (``week_present``): Bernoulli, \\mathrm{logit}(p) = b_0 + b_1 E_w.
  - U1, U2 (``Exp-tool-1_norm``, ``Exp-tool-2_norm``; only if J_week = 1):
        U1 \\sim \\mathcal{N}(c_0 + c_1 E_w, \\sigma_{U1}^2),
        U2 \\sim \\mathcal{N}(d_0 + d_1 E_w, \\sigma_{U2}^2).

Within-week intensive measures:

  - PV (hourly ``HourlyPageviewCount_norm``):  Gaussian,
        \\mu = \\alpha_0 + \\alpha_1 E_w
             + \\alpha_{2,\\mathrm{dow}}\\,\\mathrm{dow}
             + \\alpha_{2,\\mathrm{dt}}\\,\\mathrm{dt}
             + \\alpha_{2,\\mathrm{rb}}\\,\\mathrm{recent\\_burden}
             + \\alpha_{\\mathrm{AR}}\\, z^{\\mathrm{PV,lag1}}
             + A(\\alpha_3 + \\alpha_4 E_w),
        where z^{\\mathrm{PV,lag1}} is ``hourly_pageview_count_lag1`` (same row);
        (dow, dt, recent_burden) from ``dow_norm``, ``DecisionTime``,
        ``recent_burden_norm``; A = hourly ``WalkingSuggestion``; noise
        \\sigma_{\\mathrm{PV}}.

  - FW (daily outcome ``nextday_wearing``):  Bernoulli,
        \\eta = \\beta_0 + \\beta_1 E_w
             + \\beta_{2,\\mathrm{dow}}\\,\\mathrm{dow}
             + \\beta_{2,\\mathrm{rb}}\\,\\mathrm{recent\\_burden}
             + \\beta_{\\mathrm{AR}}\\, z^{\\mathrm{FW,lag}}
             + A_0(\\beta_3 + \\beta_4 E_w) + A_1(\\beta_5 + \\beta_6 E_w),
        where z^{\\mathrm{FW,lag}} is ``morning_wearing`` on the morning row;
        (dow, recent_burden) from that row; A_0, A_1 from morning/afternoon
        ``WalkingSuggestion`` (see ``build_user_blocks``).

  - PJ (daily ``daily_present``):  Bernoulli,
        \\eta = \\theta_0 + \\theta_1 E_w
             + \\theta_{2,\\mathrm{dow}}\\,\\mathrm{dow}
             + \\theta_{2,\\mathrm{rb}}\\,\\mathrm{recent\\_burden}
             + \\theta_{\\mathrm{AR}}\\, z^{\\mathrm{PJ,lag}}
             + A_0(\\theta_3 + \\theta_4 E_w) + A_1(\\theta_5 + \\theta_6 E_w),
        where z^{\\mathrm{PJ,lag}} is ``daily_present_yesterday`` on the morning row;
        same (dow, burden, A_0, A_1) construction as FW.

Lag covariates are read from ``df_fit`` (aligned per hour or per day); missing
values are treated as 0 in the linear predictor.
"""
from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json

# %%
# read data
PROJECT_ROOT = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPR-MRT-Testbed")
COMBINED_DIR = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/rawdata/_combined")
WORK_DIR = PROJECT_ROOT / "env_para_vanilla"
WORK_DIR.mkdir(parents=True, exist_ok=True)

folder = COMBINED_DIR

df_fit = pd.read_csv(folder / "df_fit.csv")

# %%
for userid in df_fit["ParticipantIdentifier"].unique():
    vc = df_fit.loc[df_fit["ParticipantIdentifier"] == userid, "week"].value_counts()
    if vc.get(0, 0) > 2:
        m = df_fit["ParticipantIdentifier"] == userid
        df_fit.loc[m, "week"] = df_fit.loc[m, "week"] + 1
df_fit = df_fit[df_fit["week"] < 13]


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
    w = np.empty_like(grid)
    w[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    w[0] = 0.5 * (grid[1] - grid[0])
    w[-1] = 0.5 * (grid[-1] - grid[-2])
    return w


def unpack_theta(theta: np.ndarray, *, e1_known: bool) -> dict:
    """
    If e1_known: no (m0, sigma0) at the end — E_1 is fixed outside the vector.
    Otherwise last two entries are m0, log(sigma0) for the week-1 prior on E_1.
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
    alpha2_dow = theta[i]
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
    beta2_dow = theta[i]
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
    theta2_dow = theta[i]
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
        "alpha2_dow": alpha2_dow,
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
        "beta2_dow": beta2_dow,
        "beta2_rb": beta2_rb,
        "beta_ar1": beta_ar1,
        "theta0": theta0,
        "theta1": theta1,
        "theta3": theta3,
        "theta4": theta4,
        "theta5": theta5,
        "theta6": theta6,
        "theta2_dow": theta2_dow,
        "theta2_rb": theta2_rb,
        "theta_ar1": theta_ar1,
    }
    if not e1_known:
        out["m0"] = theta[i]
        i += 1
        out["sigma0"] = np.exp(theta[i])
        i += 1
    return out


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
    dow_col="dow_norm",
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
    Build one block per calendar week.

    Key design:
      - The within-week likelihood uses only actually observed rows:
            pv_y, FW_daily, PJ_daily
        plus ``day_dow`` and ``day_burden`` (aligned with each calendar day, from the
        afternoon row) for FW/PJ logit covariates.
        AR coefficients multiply precomputed lag columns: ``PV_lag1_col`` per hour,
        ``FW_lag_col`` / ``PJ_lag_col`` per day (NaNs treated as 0 in the likelihood).
        Missing weeks keep these as empty arrays.

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
        day_dow_list, day_burden_list = [], []
        day_fw_lag_list, day_pj_lag_list = [], []

        for _d, g_day in g_week.groupby(date_col, sort=True):
            g_day = g_day.sort_values(decision_col, na_position="last")

            aa = g_day[a_col].to_numpy(dtype=float)
            pv = g_day[PV_col].to_numpy(dtype=float)
            dowv = g_day[dow_col].to_numpy(dtype=float)
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
                dow_k = float(dowv[k]) if not np.isnan(dowv[k]) else np.nan
                rb_k = float(burden[k]) if not np.isnan(burden[k]) else np.nan
                dt_k = float(dtv[k]) if (hourly_pv and not np.isnan(dtv[k])) else 0.0
                pv_ctx_rows.append([dow_k, dt_k, rb_k])

            row_morning = g_day.loc[g_day[decision_col] == 0].iloc[0]
            row_afternoon = g_day.loc[g_day[decision_col] == 1].iloc[0]

            A0_day = float(row_morning[a_col]) if not pd.isna(row_morning[a_col]) else 0.0
            A1_day = float(row_afternoon[a_col]) if not pd.isna(row_afternoon[a_col]) else 0.0

            a0_list.append(A0_day)
            a1_list.append(A1_day)

            fw_list.append(float(row_morning[FW_col]) if not pd.isna(row_morning[FW_col]) else np.nan)
            pj_list.append(float(row_morning[PJ_col]) if not pd.isna(row_morning[PJ_col]) else np.nan)


            day_dow_list.append(
                float(row_afternoon[dow_col]) if not pd.isna(row_afternoon[dow_col]) else np.nan
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
        day_dow = np.asarray(day_dow_list, dtype=float)
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
            "day_dow": day_dow,
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
                "day_dow": np.asarray([], dtype=float),
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
        [par["alpha2_dow"], par["alpha2_dt"], par["alpha2_rb"]],
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
        dd = block.get("day_dow")
        bd = block.get("day_burden")
        dow_d = float(dd[d]) if dd is not None and len(dd) > d else np.nan
        bd_d = float(bd[d]) if bd is not None and len(bd) > d else np.nan
        dow_d = 0.0 if np.isnan(dow_d) else dow_d
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
            + par["beta2_dow"] * dow_d
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
        dd = block.get("day_dow")
        bd = block.get("day_burden")
        dow_d = float(dd[d]) if dd is not None and len(dd) > d else np.nan
        bd_d = float(bd[d]) if bd is not None and len(bd) > d else np.nan
        dow_d = 0.0 if np.isnan(dow_d) else dow_d
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
            + par["theta2_dow"] * dow_d
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


def ml_json_export(
    res,
    filt: dict,
    blocks: list,
    *,
    digits: int = 3,
) -> tuple[dict, dict]:
    """
    Build two dicts for merging into params_env_<id>.json and pred_<id>.json.

    Parameter convention:
      - one theta_ml_* key per model
      - companion theta_ml_*_names key documents parameter order

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

    env_ml: dict = {
        "theta_ml_Ew": [
            r3(par["a0"]),
            r3(par["a1"]),
            r3(par["a2"]),
            r3(par["a3"]),
            r3(par["a4"]),
            r3(par["sigma_E"]),
        ],
        "theta_ml_Ew_names": [
            "a0",
            "a1",
            "a2_PV_lag_week",
            "a3_FW_lag_week",
            "a4_PJ_lag_week",
            "sigma_E",
        ],

        "theta_ml_J": [
            r3(par["b0"]),
            r3(par["b1"]),
        ],
        "theta_ml_J_names": [
            "b0",
            "b1_Ew",
        ],

        "theta_ml_U1": [
            r3(par["c0"]),
            r3(par["c1"]),
            r3(par["sigma_U1"]),
        ],
        "theta_ml_U1_names": [
            "c0",
            "c1_Ew",
            "sigma_U1",
        ],

        "theta_ml_U2": [
            r3(par["d0"]),
            r3(par["d1"]),
            r3(par["sigma_U2"]),
        ],
        "theta_ml_U2_names": [
            "d0",
            "d1_Ew",
            "sigma_U2",
        ],

        "theta_ml_PV": [
            r3(par["alpha0"]),
            r3(par["alpha1"]),
            r3(par["alpha2_dow"]),
            r3(par["alpha2_dt"]),
            r3(par["alpha2_rb"]),
            r3(par["alpha_ar1"]),
            r3(par["alpha3"]),
            r3(par["alpha4"]),
            r3(par["sigma_PV"]),
        ],
        "theta_ml_PV_names": [
            "alpha0",
            "alpha1_Ew",
            "alpha2_dow",
            "alpha2_decision_time",
            "alpha2_recent_burden",
            "alpha_ar1_hourly_pageview_lag1",
            "alpha3_action",
            "alpha4_action_by_Ew",
            "sigma_PV",
        ],

        "theta_ml_FW": [
            r3(par["beta0"]),
            r3(par["beta1"]),
            r3(par["beta2_dow"]),
            r3(par["beta2_rb"]),
            r3(par["beta_ar1"]),
            r3(par["beta3"]),
            r3(par["beta4"]),
            r3(par["beta5"]),
            r3(par["beta6"]),
        ],
        "theta_ml_FW_names": [
            "beta0",
            "beta1_Ew",
            "beta2_dow",
            "beta2_recent_burden",
            "beta_ar1_morning_wearing",
            "beta3_A0_morning",
            "beta4_A0_morning_by_Ew",
            "beta5_A1_afternoon",
            "beta6_A1_afternoon_by_Ew",
        ],

        "theta_ml_PJ": [
            r3(par["theta0"]),
            r3(par["theta1"]),
            r3(par["theta2_dow"]),
            r3(par["theta2_rb"]),
            r3(par["theta_ar1"]),
            r3(par["theta3"]),
            r3(par["theta4"]),
            r3(par["theta5"]),
            r3(par["theta6"]),
        ],
        "theta_ml_PJ_names": [
            "theta0",
            "theta1_Ew",
            "theta2_dow",
            "theta2_recent_burden",
            "theta_ar1_daily_present_yesterday",
            "theta3_A0_morning",
            "theta4_A0_morning_by_Ew",
            "theta5_A1_afternoon",
            "theta6_A1_afternoon_by_Ew",
        ],

        # "theta_ml_full_vector": np.round(np.asarray(res.x, dtype=float), digits).tolist(),
        # "ml_loglik": r3(filt["loglik"]),
        # "ml_opt_success": bool(res.success),
        # "ml_opt_nit": int(res.nit) if hasattr(res, "nit") else None,
    }

    if not e1_fixed:
        env_ml["theta_ml_E1_prior"] = [
            r3(par.get("m0")),
            r3(par.get("sigma0")),
        ]
        env_ml["theta_ml_E1_prior_names"] = [
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
            [par["alpha2_dow"], par["alpha2_dt"], par["alpha2_rb"]],
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

            day_dow = block.get("day_dow")
            day_burden = block.get("day_burden")
            day_FW_lag = block.get("day_FW_lag")

            dow_d = (
                float(day_dow[d])
                if day_dow is not None
                and len(day_dow) > d
                and not np.isnan(day_dow[d])
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
                + par["beta2_dow"] * dow_d
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

            day_dow = block.get("day_dow")
            day_burden = block.get("day_burden")
            day_PJ_lag = block.get("day_PJ_lag")

            dow_d = (
                float(day_dow[d])
                if day_dow is not None
                and len(day_dow) > d
                and not np.isnan(day_dow[d])
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
                + par["theta2_dow"] * dow_d
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

    env_ml["resid_ml_J_week"] = rJ
    env_ml["resid_ml_U1"] = rU1
    env_ml["resid_ml_U2"] = rU2
    env_ml["resid_ml_hourly_pageview"] = rPV
    env_ml["resid_ml_nextday_wearing"] = rFW
    env_ml["resid_ml_daily_present"] = rPJ

    pred_ml = {
        "pred_ml_filtered_Ew": [r3(x) for x in E.tolist()],
        "pred_ml_J_week": pJ,
        "pred_ml_U1": pU1,
        "pred_ml_U2": pU2,
        "pred_ml_hourly_pageview": pPV,
        "pred_ml_nextday_wearing": pFW,
        "pred_ml_daily_present": pPJ,
    }

    return env_ml, pred_ml


def attach_filtered_Ew_to_df_fit(
    df_fit: pd.DataFrame,
    filtered_states: dict,
    *,
    e1_known: Optional[float] = None,
    pu_col: str = "perceived_utility",
    pu_last_col: str = "perceived_utility_lastweek",
    week_col: str = "week",
) -> pd.DataFrame:
    """Add filtered \\hat E_w (perceived utility) and its weekly lag to ``df_fit``.

    ``filtered_states[uid]['filtered_means']`` is indexed ``0..T-1`` and
    corresponds to relabeled weeks ``1..T`` (matches ``full_weeks=range(1, 13)``
    in ``build_user_blocks``). With ``e1_known=v``, ``filtered_means[0]`` equals
    ``v`` (the fixed week-1 latent), and the regression-friendly column
    ``perceived_utility_lastweek`` stores ``NaN`` for week 1 unless ``e1_known``
    is supplied (in which case the same fixed value is used).

    Returns a new dataframe; the input is not modified.
    """
    out = df_fit.copy()
    out[pu_col] = np.nan
    out[pu_last_col] = np.nan

    for uid, filt in filtered_states.items():
        fm = np.asarray(filt.get("filtered_means", []), dtype=float)
        if fm.size == 0:
            continue
        user_mask = out["ParticipantIdentifier"] == uid
        if not user_mask.any():
            continue
        for w in range(1, fm.size + 1):
            row_mask = user_mask & (out[week_col] == w)
            if not row_mask.any():
                continue
            v_now = fm[w - 1]
            out.loc[row_mask, pu_col] = (
                float(v_now) if np.isfinite(v_now) else np.nan
            )
            if w == 1:
                out.loc[row_mask, pu_last_col] = (
                    float(e1_known) if e1_known is not None else np.nan
                )
            else:
                v_prev = fm[w - 2]
                out.loc[row_mask, pu_last_col] = (
                    float(v_prev) if np.isfinite(v_prev) else np.nan
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
    """Attach filtered E_w columns to ``df_fit`` and write the result to ``out_path``.

    The week column is whatever ``df_fit`` carries when this is called (typically
    after the ``vc.get(0, 0) > 2`` relabel and the ``week < 13`` filter applied at
    the top of this module). Re-running ``perceived_utility.py`` or
    ``2_fit_vanilla_testbed.py`` reapplies the same relabel idempotently, so the
    saved CSV is safe as the new ``df_fit.csv`` source for the vanilla fit step.
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


def write_joint_ml_into_vanilla_json_files(
    userid,
    res,
    filt: dict,
    blocks: list,
    *,
    work_dir: Optional[Path] = None,
    digits: int = 3,
) -> None:
    """
    Write joint ML parameters, residuals, and predictions to:

      params_env_<id>.json
      pred_<id>.json

    The files are **overwritten** with the ML-only keys (any stale vanilla keys
    or query-imputation suffix from a prior run are wiped). The downstream
    ``2_fit_vanilla_testbed.py`` then merges its vanilla keys into the same
    JSONs while preserving the ML keys written here.
    """
    wd = WORK_DIR if work_dir is None else Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)

    uid = int(userid) if isinstance(userid, (int, np.integer)) else userid

    p_env = wd / f"params_env_{uid}.json"
    p_pred = wd / f"pred_{uid}.json"

    env_ml, pred_ml = ml_json_export(res, filt, blocks, digits=digits)

    with open(p_env, "w", encoding="utf-8") as f:
        json.dump(env_ml, f, allow_nan=False)

    with open(p_pred, "w", encoding="utf-8") as f:
        json.dump(pred_ml, f, allow_nan=False)


# Indices into the perceived-utility fitted parameter vectors used for empirical-Bayes imputation.
# These are based on theta_ml_PV, theta_ml_FW, and theta_ml_PJ.

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

      theta_ml_PV
      theta_ml_FW
      theta_ml_PJ

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
    pv_rows = collect_rows("theta_ml_PV", PV_STAT_LEN)
    fb_rows = collect_rows("theta_ml_FW", FB_STAT_LEN)
    pj_rows = collect_rows("theta_ml_PJ", PJ_STAT_LEN)

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

        old_pv, old_pv_names = load_base("theta_ml_PV")
        old_fb, old_fb_names = load_base("theta_ml_FW")
        old_pj, old_pj_names = load_base("theta_ml_PJ")

        new_pv = pv_s[i].astype(float, copy=True)
        new_fb = fb_s[i].astype(float, copy=True)
        new_pj = pj_s[i].astype(float, copy=True)

        # Sign rules for the imputed query/action coefficients.
        # Current lengths:
        #   new_pv: 5 entries, valid indices 0..4
        #   new_fb: 4 entries, valid indices 0..3
        #   new_pj: 4 entries, valid indices 0..3

        # PV imputed suffix:
        # [query_intercept_like, query_Ew_like, query_dow_like,
        #  query_decision_time_like, query_recent_burden_like]
        new_pv[0] = -np.abs(new_pv[0])
        new_pv[1] = np.abs(new_pv[1])
        new_pv[2] = -np.abs(new_pv[2])
        new_pv[3] = -np.abs(new_pv[3])
        new_pv[4] = -np.abs(new_pv[4])

        # FW imputed suffix:
        # [query_intercept_like, query_Ew_like, query_dow_like,
        #  query_recent_burden_like]
        new_fb[0] = -np.abs(new_fb[0])
        new_fb[1] = np.abs(new_fb[1])
        new_fb[2] = -np.abs(new_fb[2])
        new_fb[3] = -np.abs(new_fb[3])

        # PJ imputed suffix:
        # [query_intercept_like, query_Ew_like, query_dow_like,
        #  query_recent_burden_like]
        new_pj[0] = -np.abs(new_pj[0])
        new_pj[1] = np.abs(new_pj[1])
        new_pj[2] = -np.abs(new_pj[2])
        new_pj[3] = -np.abs(new_pj[3])

        env_para["theta_ml_PV"] = np.round(
            np.concatenate([old_pv, new_pv]),
            digits,
        ).tolist()

        env_para["theta_ml_FW"] = np.round(
            np.concatenate([old_fb, new_fb]),
            digits,
        ).tolist()

        env_para["theta_ml_PJ"] = np.round(
            np.concatenate([old_pj, new_pj]),
            digits,
        ).tolist()

        env_para["theta_ml_PV_names"] = old_pv_names + [
            "query_imputed_intercept_like",
            "query_imputed_Ew_like",
            "query_imputed_dow_like",
            "query_imputed_decision_time_like",
            "query_imputed_recent_burden_like",
        ]

        env_para["theta_ml_FW_names"] = old_fb_names + [
            "query_imputed_intercept_like",
            "query_imputed_Ew_like",
            "query_imputed_dow_like",
            "query_imputed_recent_burden_like",
        ]

        env_para["theta_ml_PJ_names"] = old_pj_names + [
            "query_imputed_intercept_like",
            "query_imputed_Ew_like",
            "query_imputed_dow_like",
            "query_imputed_recent_burden_like",
        ]

        # Sanity-check that array lengths line up with the names arrays.
        for key, expected_k in (
            ("theta_ml_PV", PV_IMPUTE_K),
            ("theta_ml_FW", FB_IMPUTE_K),
            ("theta_ml_PJ", PJ_IMPUTE_K),
        ):
            arr_len = len(env_para[key])
            nm_len = len(env_para[f"{key}_names"])
            if arr_len != nm_len:
                raise RuntimeError(
                    f"Participant {uid}: {key} length {arr_len} != "
                    f"{key}_names length {nm_len} after imputation "
                    f"(expected suffix k={expected_k})."
                )

        meta_root["theta_ml_PV"] = {"k": PV_IMPUTE_K}
        meta_root["theta_ml_FW"] = {"k": FB_IMPUTE_K}
        meta_root["theta_ml_PJ"] = {"k": PJ_IMPUTE_K}
        env_para[_META_KEY] = meta_root

        with open(p_env, "w", encoding="utf-8") as f:
            json.dump(env_para, f, allow_nan=False)


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

    If e1_known is float: first term is log p(Y_1|E_1=e1); hat q_2 is the Gaussian
    transition from E_1. If None: week 1 uses prior N(m0,sigma0^2) on E_1.
    """
    e1_fixed = e1_known is not None
    par = unpack_theta(theta, e1_known=e1_fixed)
    grid = np.asarray(grid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    T = len(blocks)

    q_list = []
    p_list = []
    c_list = []
    pred_mean = np.empty(T)
    filt_mean = np.empty(T)
    loglik = 0.0

    if e1_fixed:
        e1 = float(e1_known)
        log_ell1 = week_loglik_at_point(e1, blocks[0], par)
        if not np.isfinite(log_ell1):
            raise FloatingPointError("invalid week-1 log-likelihood at fixed E_1")
        loglik += log_ell1
        c_list.append(float(np.exp(log_ell1)))
        pred_mean[0] = np.nan
        filt_mean[0] = e1

        if T == 1:
            return {
                "loglik": float(loglik),
                "predicted_densities": np.zeros((0, len(grid))),
                "filtered_densities": np.zeros((0, len(grid))),
                "increments": np.asarray(c_list),
                "predicted_means": pred_mean,
                "filtered_means": filt_mean,
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
        q = normal_pdf(grid, mu2, par["sigma_E"])
        denq = np.sum(q * weights)
        if denq <= 0 or not np.isfinite(denq):
            raise FloatingPointError("invalid predictive density for E_2")
        q /= denq

        for t in range(1, T):
            q_list.append(q.copy())
            pred_mean[t] = np.sum(grid * q * weights)

            log_ell = week_loglik_on_grid(grid, blocks[t], par)
            m = np.max(log_ell)
            num = np.exp(log_ell - m) * q
            den = np.sum(num * weights)
            c = np.exp(m) * den
            if (not np.isfinite(c)) or (c <= 0):
                raise FloatingPointError(f"invalid c_{t+1}")

            p = num / den
            p_list.append(p.copy())
            c_list.append(c)
            filt_mean[t] = np.sum(grid * p * weights)
            loglik += np.log(c)

            if t < T - 1:
                F = transition_matrix(grid, blocks[t], par)
                q = F @ (p * weights)
                denq = np.sum(q * weights)
                if denq <= 0 or not np.isfinite(denq):
                    raise FloatingPointError(f"invalid predictive density week {t+2}")
                q /= denq
    else:
        q = normal_pdf(grid, par["m0"], par["sigma0"])
        q /= np.sum(q * weights)
        for t in range(T):
            q_list.append(q.copy())
            pred_mean[t] = np.sum(grid * q * weights)

            log_ell = week_loglik_on_grid(grid, blocks[t], par)
            m = np.max(log_ell)
            num = np.exp(log_ell - m) * q
            den = np.sum(num * weights)
            c = np.exp(m) * den
            if (not np.isfinite(c)) or (c <= 0):
                raise FloatingPointError(f"invalid c_{t+1}")

            p = num / den
            p_list.append(p.copy())
            c_list.append(c)
            filt_mean[t] = np.sum(grid * p * weights)
            loglik += np.log(c)

            if t < T - 1:
                F = transition_matrix(grid, blocks[t], par)
                q = F @ (p * weights)
                denq = np.sum(q * weights)
                if denq <= 0 or not np.isfinite(denq):
                    raise FloatingPointError(f"invalid predictive density week {t+2}")
                q /= denq

    return {
        "loglik": float(loglik),
        "predicted_densities": np.vstack(q_list),
        "filtered_densities": np.vstack(p_list),
        "increments": np.asarray(c_list),
        "predicted_means": pred_mean,
        "filtered_means": filt_mean,
        "params": par,
        "e1_known": e1_known,
    }



def neg_loglik_blocks(theta, blocks, grid, weights, e1_known, lam=1e-2):
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


def neg_loglik_all_users(theta, user_blocks, grid, weights, e1_known, lam=1e-2):
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

def fit_pooled_model(
    df_fit,
    grid=None,
    *,
    e1_known: Optional[float] = None,
    maxiter=1000,
    week_col="week",
    date_col="Date",
    decision_col="DecisionTime",
    J_col="week_present",
    U1_col="Exp-tool-1_norm",
    U2_col="Exp-tool-2_norm",
    FW_col="nextday_wearing",
    PJ_col="daily_present",
    a_col="WalkingSuggestion",
    dow_col="dow_norm",
    burden_col="recent_burden_norm",
    PV_col="HourlyPageviewCount_norm",
    PV_lag1_col="hourly_pageview_count_lag1",
    FW_lag_col="morning_wearing",
    PJ_lag_col="daily_present_yesterday",
    hourly_pv=True,
    lam: float = 1e-2,
):
    if grid is None:
        grid = np.linspace(-2.0, 2.0, 121)

    weights = trapezoid_weights(grid)
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
        dow_col=dow_col,
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

    x0 = initial_theta_from_blocks(all_blocks, e1_known=(e1_known is not None))

    res = minimize(
        neg_loglik_all_users,
        x0,
        args=(user_blocks, grid, weights, e1_known, lam),
        method="L-BFGS-B",
        bounds=make_bounds(e1_known=(e1_known is not None)),
        options={
            "maxiter": maxiter,
            "maxfun": 1_500_000,
            "maxls": 50,
            "ftol": 1e-7,
            "gtol": 1e-4,
        },
    )

    print(
        f"Pooled fit: success={res.success}, "
        f"nll={res.fun:.4f}, nit={res.nit}, message={res.message}"
    )

    return res, user_blocks

def fit_one_user(
    dat_user,
    grid,
    *,
    e1_known: Optional[float] = None,
    maxiter=500,
    week_col="week",
    date_col="Date",
    decision_col="DecisionTime",
    J_col="week_present",
    U1_col="Exp-tool-1_norm",
    U2_col="Exp-tool-2_norm",
    FW_col="nextday_wearing",
    PJ_col="daily_present",
    a_col="WalkingSuggestion",
    dow_col="dow_norm",
    burden_col="recent_burden_norm",
    PV_col="HourlyPageviewCount_norm",
    PV_lag1_col="hourly_pageview_count_lag1",
    FW_lag_col="morning_wearing",
    PJ_lag_col="daily_present_yesterday",
    hourly_pv=True,
    x0: Optional[np.ndarray] = None,
    lam: float = 1e-2,
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
        dow_col=dow_col,
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
    e1_fixed = e1_known is not None

    if x0 is None:
        x0 = initial_theta_from_blocks(blocks, e1_known=e1_fixed)
    else:
        x0 = np.asarray(x0, dtype=float).copy()

    res = minimize(
        neg_loglik_blocks,
        x0,
        args=(blocks, grid, weights, e1_known, lam),
        method="L-BFGS-B",
        bounds=make_bounds(e1_known=e1_fixed),
        options={
            "maxiter": maxiter,
            "maxfun": 300000,
            "maxls": 50,
            "ftol": 1e-6,
            "gtol": 1e-4,
        }
    )
    filt = quadrature_loglik(blocks, res.x, grid, weights, e1_known=e1_known)
    return res, filt, blocks


def fit_all_users(
    df_fit,
    grid=None,
    *,
    e1_known: Optional[float] = None,
    maxiter=500,
    week_col="week",
    date_col="Date",
    decision_col="DecisionTime",
    J_col="week_present",
    U1_col="Exp-tool-1_norm",
    U2_col="Exp-tool-2_norm",
    FW_col="nextday_wearing",
    PJ_col="daily_present",
    a_col="WalkingSuggestion",
    dow_col="dow_norm",
    burden_col="recent_burden_norm",
    PV_col="HourlyPageviewCount_norm",
    PV_lag1_col="hourly_pageview_count_lag1",
    FW_lag_col="morning_wearing",
    PJ_lag_col="daily_present_yesterday",
    hourly_pv=True,
    pooled_x0: Optional[np.ndarray] = None,
    lam: float = 1e-2,
    merge_into_vanilla_json: bool = False,
    vanilla_work_dir: Optional[Path] = None,
    json_digits: int = 3,
):
    if grid is None:
        grid = np.linspace(-2.0, 2.0, 241)

    results = {}
    filtered_states = {}
    blocks_by_user = {}

    for uid, g in df_fit.groupby("ParticipantIdentifier"):
        g = g.sort_values(["week", "Date", "DecisionTime"], na_position="last").reset_index(drop=True)
        res, filt, blocks = fit_one_user(
            g,
            grid=grid,
            e1_known=e1_known,
            maxiter=maxiter,
            week_col=week_col,
            date_col=date_col,
            decision_col=decision_col,
            J_col=J_col,
            U1_col=U1_col,
            U2_col=U2_col,
            FW_col=FW_col,
            PJ_col=PJ_col,
            a_col=a_col,
            dow_col=dow_col,
            burden_col=burden_col,
            PV_col=PV_col,
            PV_lag1_col=PV_lag1_col,
            FW_lag_col=FW_lag_col,
            PJ_lag_col=PJ_lag_col,
            hourly_pv=hourly_pv,
            x0=pooled_x0,
            lam=lam,
        )
        results[uid] = res
        filtered_states[uid] = filt
        blocks_by_user[uid] = blocks
        print(
            f"Participant {uid}: success={res.success}, "
            f"nll={res.fun:.4f}, nit={res.nit}, message={res.message}"
        )

        if merge_into_vanilla_json:
            try:
                write_joint_ml_into_vanilla_json_files(
                    uid,
                    res,
                    filt,
                    blocks,
                    work_dir=vanilla_work_dir,
                    digits=json_digits,
                )
            except Exception as exc:
                print(f"Participant {uid}: could not merge ML into vanilla JSON: {exc}")

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
    dow_col="dow_norm",
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
            dow_col=dow_col,
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
        pu = filtered_states[uid]["filtered_means"]
        w = np.arange(len(pu))
        ax.plot(w, pu, marker="o", ms=2, lw=1)
        ax.set_title(f"Participant {uid}", fontsize=9)
        ax.set_xlabel(r"$w$", fontsize=8)
        ax.set_ylabel(r"$E_w$", fontsize=8)
    fig_ew.suptitle(r"Filtered mean $E_w$ (quadrature posterior mean on grid)", fontsize=11)
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
        (r"$\alpha_{2,\mathrm{dow}}$", lambda p: p["alpha2_dow"]),
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

    # --- FW / PJ: intercept, E_w, dow, burden, AR(1) within week ---
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
        r"$\beta_{2,\mathrm{dow}}$ (FW)",
        [float(filtered_states[u]["params"]["beta2_dow"]) for u in _users],
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
        r"$\theta_{2,\mathrm{dow}}$ (PJ)",
        [float(filtered_states[u]["params"]["theta2_dow"]) for u in _users],
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
        r"Daily FW / PJ logits — intercept, $E_w$, dow, burden, lag covariates",
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
    # Stage 1: pooled fit on a coarser grid
    pooled_res, pooled_blocks = fit_pooled_model(
        df_fit,
        grid=np.linspace(-4.5, 4.5, 121),
        e1_known=2.0,
        maxiter=2000,
        lam=1e-2,
    )

    pooled_x0 = pooled_res.x.copy()

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
        grid=np.linspace(-4.5, 4.5, 241),
        e1_known=_e1_known,
        maxiter=2000,
        pooled_x0=pooled_x0,
        lam=1e-2,
        merge_into_vanilla_json=True,
        vanilla_work_dir=WORK_DIR,
        json_digits=3,
    )

    # Write filtered E_w (perceived utility) back to df_fit.csv so that
    # 2_fit_vanilla_testbed.py can read perceived_utility / perceived_utility_lastweek.
    save_df_fit_with_perceived_utility(
        df_fit,
        filtered_states,
        out_path=folder / "df_fit.csv",
        e1_known=_e1_known,
    )

    _diag_dir = WORK_DIR / "plots_ml_perceived_utility"
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
