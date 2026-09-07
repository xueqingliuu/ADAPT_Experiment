"""Fit latent weekly perceived utility E_w for each participant.

E_w is not observed. This script treats it as a 1-D state, approximates the
likelihood by quadrature, and jointly models the weekly check-in (J, U1, U2)
and the within-week mediators (page views, Fitbit wear, daily check-in).
Each participant is fit separately after a pooled initialization.
Sparse-engagement users listed in ``ADAPR_STRONG_POOL_UIDS`` (default 248)
get a larger ridge, *centered at the pooled MLE* (not the 0.05 sign nudge),
and ``a0`` / ``log σ_E`` are included so the E_w AR cannot sit on the clip
wall with process noise at the 0.03 floor. Emission intercepts stay free so
rare wear/check-in is not forced to the population prevalence.

Weekly PV/FW/PJ summaries that enter the E_w transition use a **fixed
denominator** (14 slots / 7 days) with NaNs contributing 0. That is the
intended "missing = not engaged" convention for PV and PJ. **FW inherits
the same rule knowingly**: a missing wear flag is treated as not wearing,
even though sensor missingness is not the same as non-wear. Simulation
(``vani_env._week_means_from_arrays``) and Ê_w (script 6 / ``agents/ew_hat``)
use the same averages so fit and rollout stay aligned. Sparse-wear users
therefore have deflated FW inputs at fit time.

Writes, under ``env_para_vanilla/``:
    params_env_<uid>.json     E_w / PV / FW / PJ coefficients
    pred_<uid>.json           filtered E_w trajectory
    df_fit.csv columns        perceived_utility, perceived_utility_lastweek

PV is a two-part hurdle: Bernoulli occurrence ``P(raw count > 0)`` and,
given a positive count, a Gaussian on ``log(x)`` then z-scored among
positives (``HourlyPageviewCount_norm``). Count 0 is placed at
``log(0.5)`` on that same axis so the 14-slot PV summary still sits
below count 1. Both parts share the same covariate
design (``E_w``, weekend, slot, burden, ``week_norm``, lag1, ``A``,
``A×E_w``, ``J_w`` query). FW / PJ stay Bernoulli on wear / daily
check-in and include the same ``week_norm``. ``J_w`` is
``b0 + b1 E_w + b2 week_norm``. ``week_norm`` uses the 11-week fit
scale ``(w-6)/5``; the simulator stretches it onto ``[-1, 1]`` over
the run, the same way CAE does.

Index: ``J_w``, ``E_w``, ``U_{w,1}``, ``U_{w,2}`` sit at the end of week
``w-1`` / start of week ``w``. In the MRT table that is
``week_present_lastweek`` (not this week's Sunday ``week_present``, which
is ``J_{w+1}``). Then ``J_w | E_w``, ``M_w | E_w, J_w``, and at the end
of week ``w`` the transition writes ``E_{w+1}``, ``J_{w+1}``. The last
Sunday of week ``W`` is scored as ``J_{W+1} | E_{W+1}``.

PV / FW / PJ emissions include the opening-check-in query
``J_w · [1, E_w, recent_burden]``, written as the ``query_Jw_*`` suffix
on ``theta_penalized_{PV,FW,PJ}``. Ridge centers for those three
coefficients are 0. ``I_w`` is not in the model. Next:
``5_fit_vanilla_testbed.py``.
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

from vani_env import E_QUAD_HI, E_QUAD_LO, pv_hurdle_occurrence_z_gap, study_week_norm

logger = logging.getLogger(__name__)

# Suffix names on theta_penalized_{PV,FW,PJ}. Same layout vani_env splits.
QUERY_JW_NAMES = (
    "query_Jw_intercept",
    "query_Jw_Ew",
    "query_Jw_recent_burden",
)
QUERY_JW_INTENSITY_NAMES = (
    "intensity_query_Jw_intercept",
    "intensity_query_Jw_Ew",
    "intensity_query_Jw_recent_burden",
)
# Packed theta: 54 emission/AR coeffs + 12 J_w query coeffs
# (PV occurrence, PV intensity, FW, PJ × intercept, E_w, recent_burden).
# +5 vs the pre-week_norm packing: J, PV occ, PV intensity, FW, PJ.
_THETA_DIM_BASE = 66
# Packed log-σ indices (unpack_theta exponentiates these). σ_E, σ_U1, σ_U2, σ_PV.
_LOG_SIGMA_INDICES = (5, 11, 14, 35)
# Baseline levels / prevalences. Ridge does not shrink these (or log-σ)
# except for strong-pool users, who also penalize a0 and log σ_E toward
# the pooled MLE (see ``strong_pool_uids``).
_RIDGE_FREE_INTERCEPTS = (
    "a0", "b0", "c0", "d0", "alpha0", "gamma0", "beta0", "theta0",
)
# Default λ=0.5 → 5.0. Override with ADAPR_STRONG_POOL_LAM_MULT.
_STRONG_POOL_LAM_MULT_DEFAULT = 10.0
_STRONG_POOL_UIDS_DEFAULT = "248"
# Quadrature support for E_w. Interval is ``vani_env.E_QUAD_LO`` / ``E_QUAD_HI``
# (same as ``EnvConfig.limits_perceivedUtility``). Spacing matches the old
# [-3, 3] grids (0.05 pooled / 0.025 per-user).
E_QUAD_POOLED_N = 161
E_QUAD_USER_N = 321

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


def e_quadrature_grid(*, pooled: bool) -> np.ndarray:
    n = E_QUAD_POOLED_N if pooled else E_QUAD_USER_N
    return np.linspace(E_QUAD_LO, E_QUAD_HI, n)


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
    n_expected = _THETA_DIM_BASE if e1_known else _THETA_DIM_BASE + 2
    if theta.size != n_expected:
        raise ValueError(
            f"theta length {theta.size} != {n_expected} "
            f"(e1_known={e1_known})"
        )

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
    b2_week_norm = theta[i]
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
    alpha2_week_norm = theta[i]
    i += 1
    alpha3 = theta[i]
    i += 1
    alpha4 = theta[i]
    i += 1
    alpha_ar1 = theta[i]
    i += 1
    alpha_q0 = theta[i]
    i += 1
    alpha_qE = theta[i]
    i += 1
    alpha_qrb = theta[i]
    i += 1

    gamma0 = theta[i]
    i += 1
    gamma1 = theta[i]
    i += 1
    gamma2_is_weekend = theta[i]
    i += 1
    gamma2_dt = theta[i]
    i += 1
    gamma2_rb = theta[i]
    i += 1
    gamma2_week_norm = theta[i]
    i += 1
    gamma3 = theta[i]
    i += 1
    gamma4 = theta[i]
    i += 1
    sigma_PV = np.exp(theta[i])
    i += 1
    gamma_ar1 = theta[i]
    i += 1
    gamma_q0 = theta[i]
    i += 1
    gamma_qE = theta[i]
    i += 1
    gamma_qrb = theta[i]
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
    beta2_week_norm = theta[i]
    i += 1
    beta_ar1 = theta[i]
    i += 1
    beta_q0 = theta[i]
    i += 1
    beta_qE = theta[i]
    i += 1
    beta_qrb = theta[i]
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
    theta2_week_norm = theta[i]
    i += 1
    theta_ar1 = theta[i]
    i += 1
    theta_q0 = theta[i]
    i += 1
    theta_qE = theta[i]
    i += 1
    theta_qrb = theta[i]
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
        "b2_week_norm": b2_week_norm,
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
        "alpha2_week_norm": alpha2_week_norm,
        "alpha3": alpha3,
        "alpha4": alpha4,
        "alpha_ar1": alpha_ar1,
        "alpha_q0": alpha_q0,
        "alpha_qE": alpha_qE,
        "alpha_qrb": alpha_qrb,
        "gamma0": gamma0,
        "gamma1": gamma1,
        "gamma2_is_weekend": gamma2_is_weekend,
        "gamma2_dt": gamma2_dt,
        "gamma2_rb": gamma2_rb,
        "gamma2_week_norm": gamma2_week_norm,
        "gamma3": gamma3,
        "gamma4": gamma4,
        "sigma_PV": sigma_PV,
        "gamma_ar1": gamma_ar1,
        "gamma_q0": gamma_q0,
        "gamma_qE": gamma_qE,
        "gamma_qrb": gamma_qrb,
        "beta0": beta0,
        "beta1": beta1,
        "beta3": beta3,
        "beta4": beta4,
        "beta5": beta5,
        "beta6": beta6,
        "beta2_is_weekend": beta2_is_weekend,
        "beta2_rb": beta2_rb,
        "beta2_week_norm": beta2_week_norm,
        "beta_ar1": beta_ar1,
        "beta_q0": beta_q0,
        "beta_qE": beta_qE,
        "beta_qrb": beta_qrb,
        "theta0": theta0,
        "theta1": theta1,
        "theta3": theta3,
        "theta4": theta4,
        "theta5": theta5,
        "theta6": theta6,
        "theta2_is_weekend": theta2_is_weekend,
        "theta2_rb": theta2_rb,
        "theta2_week_norm": theta2_week_norm,
        "theta_ar1": theta_ar1,
        "theta_q0": theta_q0,
        "theta_qE": theta_qE,
        "theta_qrb": theta_qrb,
    }
    if not e1_known:
        out["m0"] = theta[i]
        i += 1
        out["sigma0"] = np.exp(theta[i])
        i += 1
    return out


def build_penalized_prior_center(*, e1_known: bool) -> np.ndarray:
    """Non-zero centers for the per-participant ridge penalty in
    ``neg_loglik_blocks`` / ``fit_one_user``.

    This replaces an earlier post-hoc sign-flip for
    theta_penalized_Ew/PV/FW/PJ (M^E -> E_{w+1}; E_w -> M^E; A -> M^E;
    A*E_w -> M^E). Instead of fitting each participant freely and then
    reflecting a negative coefficient into a positive one afterwards (which
    preserves whatever noisy *magnitude* a small per-participant sample
    happened to produce), the same "should help on average" assumption is
    asserted directly as the ridge penalty's shrinkage target: each
    participant's own L-BFGS-B fit is still free to land anywhere the
    likelihood supports, but sampling noise now gets pulled toward a small
    plausible value in the expected direction instead of toward 0 (and, on
    the old approach, potentially reflected to a large value of the wrong
    sign). Every coefficient not listed here keeps the original zero-centered
    penalty (nudge = 0), including the J_w query block
    ``(alpha|gamma|beta|theta)_q*``.

    Indices are recovered by unpacking a probe vector of its own positions
    through ``unpack_theta``, so this stays correct if that function's field
    order ever changes. Intercepts and log-σ are not in this vector and are
    excluded from the ridge entirely (see ``ridge_penalty_weights``).
    """
    n = theta_dim(e1_known=e1_known)
    idx = unpack_theta(np.arange(n, dtype=float), e1_known=e1_known)

    center = np.zeros(n, dtype=float)
    nudge = 0.05
    # M^E -> E_{w+1}: more PV/FW/PJ engagement last week shouldn't lower
    # next week's perceived utility.
    for name in ("a2", "a3", "a4"):
        center[int(round(idx[name]))] = nudge
    # E_w -> M^E: higher perceived utility -> more engagement.
    for name in ("alpha1", "gamma1", "beta1", "theta1"):
        center[int(round(idx[name]))] = nudge
    # A -> M^E (intercept-only): sending a suggestion is a direct
    # engagement/attention cost, absent any offsetting utility.
    for name in ("alpha3", "gamma3", "beta3", "beta5", "theta3", "theta5"):
        center[int(round(idx[name]))] = -nudge
    # A * E_w -> M^E: that cost should be offset for participants who find
    # the app more useful.
    for name in ("alpha4", "gamma4", "beta4", "beta6", "theta4", "theta6"):
        center[int(round(idx[name]))] = nudge
    return center


def strong_pool_uids() -> frozenset[int]:
    """Users who shrink toward the pooled MLE with a larger λ.

    Default is 248 (sparse PV/PJ, E_w AR on the σ_E floor and a0 wall).
    Set ``ADAPR_STRONG_POOL_UIDS`` to a comma list, or empty to disable.
    """
    raw = os.getenv("ADAPR_STRONG_POOL_UIDS", _STRONG_POOL_UIDS_DEFAULT)
    if not str(raw).strip():
        return frozenset()
    return frozenset(int(x) for x in str(raw).split(",") if x.strip())


def strong_pool_lam_mult() -> float:
    raw = os.getenv("ADAPR_STRONG_POOL_LAM_MULT", "").strip()
    return float(raw) if raw else float(_STRONG_POOL_LAM_MULT_DEFAULT)


def ridge_penalty_weights(
    *,
    e1_known: bool,
    penalize_a0_sigma_e: bool = False,
) -> np.ndarray:
    """Per-coordinate ridge weights: 1 = penalized, 0 = free.

    Intercepts (baseline prevalence / level) and packed log-σ stay at the
    likelihood MLE. Query-block intercepts (``*_q0``) are slopes on opening
    ``J_w`` and remain penalized.

    ``penalize_a0_sigma_e`` turns on a0 and log σ_E for strong-pool users so
    those two coordinates shrink toward the pooled MLE instead of sitting
    on the clip / bound. Other emission intercepts stay free.
    """
    n = theta_dim(e1_known=e1_known)
    w = np.ones(n, dtype=float)
    idx = unpack_theta(np.arange(n, dtype=float), e1_known=e1_known)
    for name in _RIDGE_FREE_INTERCEPTS:
        w[int(round(idx[name]))] = 0.0
    for i in _LOG_SIGMA_INDICES:
        w[i] = 0.0
    if not e1_known:
        w[int(round(idx["m0"]))] = 0.0
        w[-1] = 0.0  # log σ0
    if penalize_a0_sigma_e:
        w[int(round(idx["a0"]))] = 1.0
        w[_LOG_SIGMA_INDICES[0]] = 1.0
    return w


def _ridge_penalty(
    theta: np.ndarray,
    lam: float,
    e1_known: bool,
    prior_center=None,
    ridge_weights=None,
) -> float:
    theta = np.asarray(theta, dtype=float)
    if ridge_weights is None:
        w = ridge_penalty_weights(e1_known=e1_known)
    else:
        w = np.asarray(ridge_weights, dtype=float)
    if prior_center is None:
        center = np.zeros_like(theta)
    else:
        center = np.asarray(prior_center, dtype=float)
    return float(lam * np.sum(w * (theta - center) ** 2))


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
        ("J_week", ["b0", "b1", "b2_week_norm"]),
        ("U1", ["c0", "c1", "sigma_U1"]),
        ("U2", ["d0", "d1", "sigma_U2"]),
        ("PV hurdle occurrence", [
            "alpha0", "alpha1", "alpha2_is_weekend", "alpha2_dt", "alpha2_rb",
            "alpha2_week_norm", "alpha3", "alpha4", "alpha_ar1",
            "alpha_q0", "alpha_qE", "alpha_qrb",
        ]),
        ("PV hurdle intensity", [
            "gamma0", "gamma1", "gamma2_is_weekend", "gamma2_dt", "gamma2_rb",
            "gamma2_week_norm", "gamma3", "gamma4", "sigma_PV", "gamma_ar1",
            "gamma_q0", "gamma_qE", "gamma_qrb",
        ]),
        ("FW", [
            "beta0", "beta1", "beta3", "beta4", "beta5", "beta6",
            "beta2_is_weekend", "beta2_rb", "beta2_week_norm", "beta_ar1",
            "beta_q0", "beta_qE", "beta_qrb",
        ]),
        ("PJ", [
            "theta0", "theta1", "theta3", "theta4", "theta5", "theta6",
            "theta2_is_weekend", "theta2_rb", "theta2_week_norm", "theta_ar1",
            "theta_q0", "theta_qE", "theta_qrb",
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
    return _THETA_DIM_BASE if e1_known else _THETA_DIM_BASE + 2


def initial_theta_from_blocks(blocks, *, e1_known: bool) -> np.ndarray:
    j_open = [b["J_lag"] for b in blocks if not np.isnan(b.get("J_lag", np.nan))]
    u1_open = [b.get("U1_open", np.nan) for b in blocks if not np.isnan(b.get("U1_open", np.nan))]
    u2_open = [b.get("U2_open", np.nan) for b in blocks if not np.isnan(b.get("U2_open", np.nan))]
    if blocks:
        last = blocks[-1]
        j_close = last.get("J_week", np.nan)
        if np.isfinite(j_close):
            j_open.append(j_close)
            if int(round(float(j_close))) == 1:
                u1_c = last.get("U1", np.nan)
                u2_c = last.get("U2", np.nan)
                if np.isfinite(u1_c):
                    u1_open.append(u1_c)
                if np.isfinite(u2_c):
                    u2_open.append(u2_c)
    all_J = np.array(j_open, dtype=float)
    all_U1 = np.array(u1_open, dtype=float)
    all_U2 = np.array(u2_open, dtype=float)
    all_PV = (
        np.concatenate([b["pv_y"][~np.isnan(b["pv_y"])] for b in blocks])
        if blocks
        else np.array([0.0])
    )
    z_pos_parts = []
    for b in blocks:
        y = np.asarray(b.get("pv_y", []), dtype=float)
        z = np.asarray(b.get("pv_z", []), dtype=float)
        if y.size == 0 or z.size == 0:
            continue
        n = min(y.size, z.size)
        mask = (y[:n] == 1.0) & np.isfinite(z[:n])
        if np.any(mask):
            z_pos_parts.append(z[:n][mask])
    z_pos = np.concatenate(z_pos_parts) if z_pos_parts else np.array([], dtype=float)

    pJ = np.clip(np.mean(all_J) if len(all_J) else 0.5, 1e-4, 1 - 1e-4)
    muU1 = np.mean(all_U1) if len(all_U1) else 0.0
    muU2 = np.mean(all_U2) if len(all_U2) else 0.0
    sdU1 = np.std(all_U1) if len(all_U1) > 1 else 1.0
    sdU2 = np.std(all_U2) if len(all_U2) > 1 else 1.0
    pPV = np.clip(np.mean(all_PV) if len(all_PV) else 0.5, 1e-4, 1 - 1e-4)
    muZ = float(np.mean(z_pos)) if z_pos.size else 0.0
    sdZ = float(np.std(z_pos)) if z_pos.size > 1 else 0.5

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
        0.0,  # b2_week_norm
        muU1,
        0.1,
        np.log(max(sdU1, 0.1)),
        muU2,
        0.1,
        np.log(max(sdU2, 0.1)),
        np.log(pPV / (1.0 - pPV)),
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,  # alpha2_week_norm
        0.0,
        0.0,
        0.0,  # alpha_ar1
        0.0, 0.0, 0.0,  # alpha_q0, alpha_qE, alpha_qrb (J_w)
        muZ,  # gamma0 intensity intercept on log-then-z
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,  # gamma2_week_norm
        0.0,
        0.0,
        np.log(max(sdZ, 0.1)),
        0.0,  # gamma_ar1
        0.0, 0.0, 0.0,  # gamma_q0, gamma_qE, gamma_qrb
        0.0,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # beta2_week_norm
        0.0,  # beta_ar1
        0.0, 0.0, 0.0,  # beta_q0, beta_qE, beta_qrb
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # theta2_week_norm
        0.0,  # theta_ar1
        0.0, 0.0, 0.0,  # theta_q0, theta_qE, theta_qrb
    ]
    if not e1_known:
        parts.extend([0.0, np.log(1.0)])
    out = np.asarray(parts, dtype=float)
    n_expected = theta_dim(e1_known=e1_known)
    if out.size != n_expected:
        raise RuntimeError(
            f"initial_theta length {out.size} != theta_dim {n_expected}"
        )
    return out


def build_user_blocks(
    dat_user,
                      week_col="week",
                      date_col="Date",
                      decision_col="DecisionTime",
                      J_col="week_present",
                      U1_col="Exp-tool-1_norm",
                      U2_col="Exp-tool-2_norm",
    PV_col="HourlyPageviewCount_norm",
    PV_log_col="HourlyPageviewCount",
    FW_col="nextday_wearing",
                      PJ_col="daily_present",
    a_col="WalkingSuggestion",
    weekend_col="is_weekend",
    burden_col="recent_burden_norm",
    PV_lag1_col="hourly_pageview_count_lag1",
    FW_lag_col="morning_wearing",
    PJ_lag_col="daily_present_yesterday",
    J_lag_col="week_present_lastweek",
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
            pv_y (hurdle occurrence 0/1 from raw ``PV_log_col`` count ``> 0``),
            FW_daily, PJ_daily
        plus ``day_is_weekend`` and ``day_burden`` (aligned with each calendar day, from the
        afternoon row) for FW/PJ logit covariates.
        AR coefficients multiply precomputed lag columns: ``PV_lag1_col`` per hour
        (z-scored log pageview, same scale as the simulator state),
        ``FW_lag_col`` / ``PJ_lag_col`` per day (NaNs treated as 0 in the likelihood).
        ``J_w`` is ``week_present_lastweek`` (``J_lag_col``): the Sunday
        check-in at the end of week ``w-1`` / start of week ``w``. It is
        modeled as ``J_w | E_w`` and enters PV/FW/PJ as
        ``J_w · [1, E_w, rb]``. This week's Sunday ``week_present`` is
        ``J_{w+1}`` (copied forward as next week's ``J_lag``). The last
        block's Sunday ``week_present`` is scored as ``J_{W+1} | E_{W+1}``
        on the terminal predictive after the final transition (no extra
        mediator week). Missing ``J_lag`` is filled from the previous
        block's ``J_week`` when that exists; otherwise it stays NaN and the
        J/U Bernoulli terms are skipped (week 1 has no prior Sunday in the
        panel — do not score a fake ``J_1=0`` Bernoulli). On the PV/FW/PJ
        query, missing ``J_lag`` is 0, including week 1. Opening ``U_w`` is
        last week's Sunday tools. Missing weeks keep the mediator arrays
        empty.

      - Weekly intensity summaries for the transition use fixed denominators
        (``nansum(pv)/14``, ``nansum(FW|PJ)/7``): each slot contributes 0 if
        NaN, not a mean over observed-only slots. With a complete 14-slot /
        7-day panel this matches zero-imputed averages. For FW this codes
        sensor missingness as not wearing; that is intentional so the
        simulator and Ê_w use the same summary.

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
            f"{PV_col!r} not in dataframe (use HourlyPageviewCount_norm)"
        )
    if PV_log_col not in dat.columns:
        raise KeyError(
            f"{PV_log_col!r} not in dataframe (raw hourly pageview count; "
            "used to code hurdle occurrence)"
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
        if J_lag_col is not None and J_lag_col in g_week.columns:
            J_lag = _first_nonmissing(g_week[J_lag_col])
        else:
            J_lag = np.nan

        pv_y_list, pv_z_list, pv_a_list = [], [], []
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
            pv_log = g_day[PV_log_col].to_numpy(dtype=float)
            weekendv = g_day[weekend_col].to_numpy(dtype=float)
            burden = g_day[burden_col].to_numpy(dtype=float)
            dtv = g_day[decision_col].to_numpy(dtype=float)
            lag1v = (
                g_day[PV_lag1_col].to_numpy(dtype=float)
                if PV_lag1_col is not None
                else np.full(len(g_day), np.nan, dtype=float)
            )

            for k in range(len(g_day)):
                z_k = float(pv[k]) if not np.isnan(pv[k]) else np.nan
                pv_z_list.append(z_k)
                lv = float(pv_log[k]) if k < len(pv_log) and not np.isnan(pv_log[k]) else np.nan
                if np.isnan(lv):
                    pv_y_list.append(np.nan)
                else:
                    # Raw count > 0. Also true of a leftover log column.
                    pv_y_list.append(1.0 if lv > 0.0 else 0.0)
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
        pv_z = np.asarray(pv_z_list, dtype=float)
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

        # Fixed 14/7 denominators: NaN slot → 0. FW missingness is treated as
        # not wearing (same convention as vani_env / script 6 / ew_hat).
        PV_sum = float(np.nansum(pv_z) / 14.0) if np.any(~np.isnan(pv_z)) else np.nan
        FW_sum = float(np.nansum(FW_daily) / 7.0) if np.any(~np.isnan(FW_daily)) else np.nan
        PJ_sum = float(np.nansum(PJ_daily) / 7.0) if np.any(~np.isnan(PJ_daily)) else np.nan
        return {
            "week": int(week),
            "is_missing_week": False,
            "J_week": J_week,
            "J_lag": J_lag,
            "U1": U1,
            "U2": U2,
            "pv_y": pv_y,
            "pv_z": pv_z,
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
                "J_lag": np.nan,
                "U1": np.nan,
                "U2": np.nan,
                "U1_open": np.nan,
                "U2_open": np.nan,
                "pv_y": np.asarray([], dtype=float),
                "pv_z": np.asarray([], dtype=float),
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

    j_week_by_w = {int(b["week"]): b["J_week"] for b in blocks}
    for b in blocks:
        if np.isfinite(b.get("J_lag", np.nan)):
            continue
        prev = j_week_by_w.get(int(b["week"]) - 1, np.nan)
        if np.isfinite(prev):
            b["J_lag"] = float(prev)
        # else leave NaN: no fabricated J=0 Bernoulli at study entry.

    prev_u1, prev_u2 = np.nan, np.nan
    for b in blocks:
        # Opening U_w is last Sunday's tools (same survey as J_w).
        b["U1_open"] = prev_u1
        b["U2_open"] = prev_u2
        prev_u1 = b.get("U1", np.nan)
        prev_u2 = b.get("U2", np.nan)
        b["week_norm"] = float(study_week_norm(b["week"]))

    return blocks


def _block_jw(block) -> float:
    """Opening ``J_w`` on PV/FW/PJ. Missing ``J_lag`` (including week 1) → 0."""
    j = block.get("J_lag", np.nan)
    if j is None or not np.isfinite(j):
        return 0.0
    return float(j)


def _block_week_norm(block, *, extra=0) -> float:
    """Fit-scale ``week_norm`` for this block; ``extra=1`` for closing J_{W+1}."""
    wn = block.get("week_norm")
    w = block.get("week")
    if extra:
        if w is None or not np.isfinite(w):
            return 0.0
        return float(study_week_norm(float(w) + extra))
    if wn is not None and np.isfinite(wn):
        return float(wn)
    if w is not None and np.isfinite(w):
        return float(study_week_norm(w))
    return 0.0


def _jw_query_shift(q0, qE, qrb, j_w, e, rb):
    """``J_w * (q0 + qE * E + qrb * rb)``. ``e`` may be a quadrature grid."""
    j = 0.0 if j_w is None or not np.isfinite(j_w) else float(j_w)
    if j == 0.0:
        return 0.0 * e
    rb_f = 0.0 if rb is None or not np.isfinite(rb) else float(rb)
    return j * (q0 + qE * e + qrb * rb_f)


def _pv_linpred(par, pfx, grid, row, a, y_lag, j_w, week_norm=0.0):
    """Linear predictor for PV occurrence (``alpha``) or intensity (``gamma``)."""
    coef2 = np.array(
        [par[f"{pfx}2_is_weekend"], par[f"{pfx}2_dt"], par[f"{pfx}2_rb"]],
        dtype=float,
    )
    return (
        par[f"{pfx}0"]
        + par[f"{pfx}1"] * grid
        + float(row @ coef2)
        + par[f"{pfx}2_week_norm"] * float(week_norm)
        + par[f"{pfx}_ar1"] * y_lag
        + a * (par[f"{pfx}3"] + par[f"{pfx}4"] * grid)
        + _jw_query_shift(
            par[f"{pfx}_q0"], par[f"{pfx}_qE"], par[f"{pfx}_qrb"],
            j_w, grid, row[2],
        )
    )


def _fw_linpred(par, grid, weekend_d, bd_d, y_lag, a0, a1, j_w, week_norm=0.0):
    return (
        par["beta0"]
        + par["beta1"] * grid
        + par["beta2_is_weekend"] * weekend_d
        + par["beta2_rb"] * bd_d
        + par["beta2_week_norm"] * float(week_norm)
        + par["beta_ar1"] * y_lag
        + a0 * (par["beta3"] + par["beta4"] * grid)
        + a1 * (par["beta5"] + par["beta6"] * grid)
        + _jw_query_shift(
            par["beta_q0"], par["beta_qE"], par["beta_qrb"],
            j_w, grid, bd_d,
        )
    )


def _pj_linpred(par, grid, weekend_d, bd_d, y_lag, a0, a1, j_w, week_norm=0.0):
    return (
        par["theta0"]
        + par["theta1"] * grid
        + par["theta2_is_weekend"] * weekend_d
        + par["theta2_rb"] * bd_d
        + par["theta2_week_norm"] * float(week_norm)
        + par["theta_ar1"] * y_lag
        + a0 * (par["theta3"] + par["theta4"] * grid)
        + a1 * (par["theta5"] + par["theta6"] * grid)
        + _jw_query_shift(
            par["theta_q0"], par["theta_qE"], par["theta_qrb"],
            j_w, grid, bd_d,
        )
    )


def _j_eta(par, grid, week_norm=0.0):
    return par["b0"] + par["b1"] * grid + par["b2_week_norm"] * float(week_norm)


def _ju_loglik_on_grid(grid, j, u1, u2, par, week_norm=0.0) -> np.ndarray:
    """log p(J, U | E = grid). Missing ``j`` contributes 0."""
    ll = np.zeros_like(grid)
    if j is None or not np.isfinite(j):
        return ll
    eta = _j_eta(par, grid, week_norm)
    ll += bernoulli_logpmf(j, eta)
    if int(round(float(j))) != 1:
        return ll
    if u1 is not None and np.isfinite(u1):
        mu = par["c0"] + par["c1"] * grid
        ll += normal_logpdf(u1, mu, par["sigma_U1"])
    if u2 is not None and np.isfinite(u2):
        mu = par["d0"] + par["d1"] * grid
        ll += normal_logpdf(u2, mu, par["sigma_U2"])
    return ll


def _week_loglik_components_on_grid(grid: np.ndarray, block: dict, par: dict) -> np.ndarray:
    """log p(Y_w | E_w = grid, par) for each grid point; vector of shape grid.shape."""
    grid = np.asarray(grid, dtype=float)
    ll = np.zeros_like(grid)

    # Opening J_w (week_present_lastweek) | E_w. This week's Sunday
    # week_present is J_{w+1}: scored on the next block, or as the
    # terminal J_{W+1} | E_{W+1} increment after the last transition.
    ll += _ju_loglik_on_grid(
        grid,
        block.get("J_lag", np.nan),
        block.get("U1_open", np.nan),
        block.get("U2_open", np.nan),
        par,
        week_norm=_block_week_norm(block),
    )

    pv_y = block["pv_y"]
    pv_z = block.get("pv_z")
    pv_ctx = block["pv_c_ctx"]
    pv_a = block["pv_a"]

    for j in range(len(pv_y)):
        y = pv_y[j]
        if np.isnan(y):
            continue
        row = np.asarray(pv_ctx[j], dtype=float)
        row = np.where(np.isnan(row), 0.0, row)
        a = pv_a[j]
        a = 0.0 if np.isnan(a) else a
        pl1 = block.get("pv_lag1")
        y_lag = (
            float(pl1[j])
            if pl1 is not None and len(pl1) > j and not np.isnan(pl1[j])
            else 0.0
        )
        eta = _pv_linpred(
            par, "alpha", grid, row, a, y_lag, _block_jw(block),
            week_norm=_block_week_norm(block),
        )
        ll += bernoulli_logpmf(y, eta)
        if int(round(float(y))) == 1 and pv_z is not None and j < len(pv_z):
            z = float(pv_z[j])
            if np.isfinite(z):
                mu_z = _pv_linpred(
                    par, "gamma", grid, row, a, y_lag, _block_jw(block),
                    week_norm=_block_week_norm(block),
                )
                ll += normal_logpdf(z, mu_z, par["sigma_PV"])

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
        eta = _fw_linpred(
            par, grid, weekend_d, bd_d, y_lag, a0, a1, _block_jw(block),
            week_norm=_block_week_norm(block),
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
        eta = _pj_linpred(
            par, grid, weekend_d, bd_d, y_lag, a0, a1, _block_jw(block),
            week_norm=_block_week_norm(block),
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
        "ridge_center": filt.get("ridge_center"),
        "ridge_penalize_a0_sigma_e": bool(filt.get("ridge_penalize_a0_sigma_e")),
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
            r3(par["b2_week_norm"]),
        ],
        "theta_penalized_J_names": [
            "b0",
            "b1_Ew",
            "b2_week_norm",
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
            r3(par["alpha2_week_norm"]),
            r3(par["alpha_ar1"]),
            r3(par["alpha3"]),
            r3(par["alpha4"]),
            r3(par["alpha_q0"]),
            r3(par["alpha_qE"]),
            r3(par["alpha_qrb"]),
            r3(par["gamma0"]),
            r3(par["gamma1"]),
            r3(par["gamma2_is_weekend"]),
            r3(par["gamma2_dt"]),
            r3(par["gamma2_rb"]),
            r3(par["gamma2_week_norm"]),
            r3(par["gamma_ar1"]),
            r3(par["gamma3"]),
            r3(par["gamma4"]),
            r3(par["gamma_q0"]),
            r3(par["gamma_qE"]),
            r3(par["gamma_qrb"]),
            r3(par["sigma_PV"]),
        ],
        "theta_penalized_PV_names": [
            "alpha0",
            "alpha1_Ew",
            "alpha2_is_weekend",
            "alpha2_decision_time",
            "alpha2_recent_burden",
            "alpha2_week_norm",
            "alpha_ar1_hourly_pageview_lag1",
            "alpha3_action",
            "alpha4_action_by_Ew",
            *QUERY_JW_NAMES,
            "gamma0",
            "gamma1_Ew",
            "gamma2_is_weekend",
            "gamma2_decision_time",
            "gamma2_recent_burden",
            "gamma2_week_norm",
            "gamma_ar1_hourly_pageview_lag1",
            "gamma3_action",
            "gamma4_action_by_Ew",
            *QUERY_JW_INTENSITY_NAMES,
            "sigma_PV",
        ],

        "theta_penalized_FW": [
            r3(par["beta0"]),
            r3(par["beta1"]),
            r3(par["beta2_is_weekend"]),
            r3(par["beta2_rb"]),
            r3(par["beta2_week_norm"]),
            r3(par["beta_ar1"]),
            r3(par["beta3"]),
            r3(par["beta4"]),
            r3(par["beta5"]),
            r3(par["beta6"]),
            r3(par["beta_q0"]),
            r3(par["beta_qE"]),
            r3(par["beta_qrb"]),
        ],
        "theta_penalized_FW_names": [
            "beta0",
            "beta1_Ew",
            "beta2_is_weekend",
            "beta2_recent_burden",
            "beta2_week_norm",
            "beta_ar1_morning_wearing",
            "beta3_A0_morning",
            "beta4_A0_morning_by_Ew",
            "beta5_A1_afternoon",
            "beta6_A1_afternoon_by_Ew",
            *QUERY_JW_NAMES,
        ],

        "theta_penalized_PJ": [
            r3(par["theta0"]),
            r3(par["theta1"]),
            r3(par["theta2_is_weekend"]),
            r3(par["theta2_rb"]),
            r3(par["theta2_week_norm"]),
            r3(par["theta_ar1"]),
            r3(par["theta3"]),
            r3(par["theta4"]),
            r3(par["theta5"]),
            r3(par["theta6"]),
            r3(par["theta_q0"]),
            r3(par["theta_qE"]),
            r3(par["theta_qrb"]),
        ],
        "theta_penalized_PJ_names": [
            "theta0",
            "theta1_Ew",
            "theta2_is_weekend",
            "theta2_recent_burden",
            "theta2_week_norm",
            "theta_ar1_daily_present_yesterday",
            "theta3_A0_morning",
            "theta4_A0_morning_by_Ew",
            "theta5_A1_afternoon",
            "theta6_A1_afternoon_by_Ew",
            *QUERY_JW_NAMES,
        ],

        "penalized_loglik": r3(filt["loglik"]),
        "penalized_objective": r3(res.fun),
        "penalized_opt_success": bool(res.success),
        "penalized_opt_nit": int(res.nit) if hasattr(res, "nit") else None,
        "_query_Jw_effect": {
            "source": (
                "J_w = week_present_lastweek (end of week w-1 / start of "
                "week w) on script-4 PV/FW/PJ emissions"
            ),
            "apply": "add J_w * (q @ z) on the PV/FW/PJ linear predictor; I_w not in model",
        },
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

        # Weekly J_w: week_present_lastweek, one entry per week
        eta_J = _j_eta(par, e, _block_week_norm(block))
        ph_J = float(_sigm(eta_J))

        j_w = block.get("J_lag", np.nan)
        if not np.isnan(j_w):
            pJ.append(r3(ph_J))
            rJ.append(r3(float(j_w) - ph_J))
        else:
            pJ.append(r3(ph_J))
            rJ.append(None)

        # Weekly U_w: last Sunday's tools. Residual is None if U is
        # missing or J_w != 1.
        j1 = (not np.isnan(j_w)) and int(round(float(j_w))) == 1

        mu_U1 = float(par["c0"] + par["c1"] * e)
        u1 = block.get("U1_open", np.nan)
        if j1 and not np.isnan(u1):
            pU1.append(r3(mu_U1))
            rU1.append(r3(float(u1) - mu_U1))
        else:
            pU1.append(r3(mu_U1))
            rU1.append(None)

        mu_U2 = float(par["d0"] + par["d1"] * e)
        u2 = block.get("U2_open", np.nan)
        if j1 and not np.isnan(u2):
            pU2.append(r3(mu_U2))
            rU2.append(r3(float(u2) - mu_U2))
        else:
            pU2.append(r3(mu_U2))
            rU2.append(None)

        # ------------------------------------------------------------
        # Hourly PV hurdle: one entry per included PV row.
        # Prediction is P(count>0). Residual is the intensity leftover
        # (z − μ | count>0) for the simulator bootstrap; None if the slot
        # is zero or missing.
        pv_y = block["pv_y"]
        pv_z = block.get("pv_z")
        pv_ctx = block["pv_c_ctx"]
        pv_a = block["pv_a"]
        pv_lag1 = block.get("pv_lag1")

        for j in range(len(pv_y)):
            y = pv_y[j]

            row = np.asarray(pv_ctx[j], dtype=float)
            row = np.where(np.isnan(row), 0.0, row)
            A = pv_a[j]
            A = 0.0 if np.isnan(A) else float(A)
            y_lag = (
                float(pv_lag1[j])
                if pv_lag1 is not None
                and len(pv_lag1) > j
                and not np.isnan(pv_lag1[j])
                else 0.0
            )
            eta = float(
                _pv_linpred(
                    par, "alpha", e, row, A, y_lag, _block_jw(block),
                    week_norm=_block_week_norm(block),
                )
            )
            pPV.append(r3(float(_sigm(eta))))
            occ = (not np.isnan(y)) and int(round(float(y))) == 1
            z = (
                float(pv_z[j])
                if occ and pv_z is not None and j < len(pv_z)
                else np.nan
            )
            if occ and np.isfinite(z):
                mu_z = float(
                    _pv_linpred(
                        par, "gamma", e, row, A, y_lag, _block_jw(block),
                        week_norm=_block_week_norm(block),
                    )
                )
                rPV.append(r3(z - mu_z))
            else:
                rPV.append(None)

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

            eta = _fw_linpred(
                par, e, weekend_d, burden_d, lag_d, A0, A1, _block_jw(block),
                week_norm=_block_week_norm(block),
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

            eta = _pj_linpred(
                par, e, weekend_d, burden_d, lag_d, A0, A1, _block_jw(block),
                week_norm=_block_week_norm(block),
            )

            ph = float(_sigm(eta))
            pPJ.append(r3(ph))

            if np.isnan(y):
                rPJ.append(None)
            else:
                rPJ.append(r3(float(y) - ph))

    # Closing Sunday J_{W+1}, U_{W+1} | E_{W+1} (no extra mediator week).
    if blocks:
        e_term = filt.get("terminal_predicted_mean", np.nan)
        e_term = float(e_term) if e_term is not None and np.isfinite(e_term) else 0.0
        last = blocks[-1]
        j_close = last.get("J_week", np.nan)
        eta_J = _j_eta(par, e_term, _block_week_norm(last, extra=1))
        ph_J = float(_sigm(eta_J))
        pJ.append(r3(ph_J))
        rJ.append(
            r3(float(j_close) - ph_J) if np.isfinite(j_close) else None
        )
        j1_close = np.isfinite(j_close) and int(round(float(j_close))) == 1
        mu_U1 = float(par["c0"] + par["c1"] * e_term)
        u1_close = last.get("U1", np.nan)
        pU1.append(r3(mu_U1))
        rU1.append(
            r3(float(u1_close) - mu_U1)
            if j1_close and np.isfinite(u1_close)
            else None
        )
        mu_U2 = float(par["d0"] + par["d1"] * e_term)
        u2_close = last.get("U2", np.nan)
        pU2.append(r3(mu_U2))
        rU2.append(
            r3(float(u2_close) - mu_U2)
            if j1_close and np.isfinite(u2_close)
            else None
        )

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
    """Write the quadrature-filtered E_w plug-in into ``df_fit``.

    ``filtered_states[uid]['filtered_means']`` has length T (study weeks 1..T).
    With the production pin ``e1_known=v``:
      - ``fm[0]`` = v. E_1 is not updated with week-1 emissions.
      - ``fm[t]`` for t >= 1 is the filtered mean of E_{t+1} given Y_{1:t+1},
        so it conditions on week t+1's own emissions (J, U1/U2, PV, FW, PJ).

    On a row with study week w (1..T):
      - ``perceived_utility_lastweek`` = ``fm[w-1]``. This is the plug-in for
        the latent E_w that governs week w's SSM likelihood. For w = 1 it is
        the pin v (truly pre-week). For w >= 2 it is Ê_{w|w}, which uses week
        w's own emissions — not information known before week w.
      - ``perceived_utility`` = ``fm[w]`` for w < T (Ê_{w+1|w+1}), or the
        filtered ``terminal_predicted_mean`` (E_{T+1} | Y_{1:T}, J_{T+1},
        U_{T+1}) on week T. If the last Sunday is missing, this falls back
        to the one-step predictive E_{T+1} | Y_{1:T}.

    The two columns are still a literal weekly lag: lastweek on week w equals
    ``perceived_utility`` on week w-1 (week 1 lastweek is the pin; no week-0
    row). Script 5 uses ``perceived_utility_lastweek`` as a generated
    regressor for fourSC / anticipated affect — a contemporaneous plug-in
    for E_w, not a pre-week covariate. The agent never sees this column;
    runtime ``E_known`` comes from script 6.

    One-step predictives Ê_{w|w-1} are already in ``filt['predicted_means']``
    (NaN at t = 0 when E_1 is pinned). A smoother Ê_{w|T} is not computed.

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
    """Attach filtered E_w columns to ``df_fit`` and write ``out_path``.

    Expects study weeks 1–12. On row week=w: ``perceived_utility_lastweek``
    is the contemporaneous plug-in Ê_{w|w} (E_1 pin on week 1);
    ``perceived_utility`` is Ê_{w+1|w+1} (E_{13} given last Sunday on
    week 12). See :func:`attach_filtered_Ew_to_df_fit`.
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

    The files are **overwritten** with the penalized-fit keys (any stale vanilla
    keys from a prior run are wiped). The ``J_w`` ``query_Jw_*`` suffix is
    part of this write. Downstream ``5_fit_vanilla_testbed.py`` then merges
    its vanilla keys into the same JSONs while preserving the penalized-fit
    keys written here.
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


def _predict_after_last_block(
    grid: np.ndarray,
    weights: np.ndarray,
    blocks: list,
    p_list: list[np.ndarray],
    par: dict,
    *,
    e1_known: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """Predictive density of E_{T+1} given Y_{1:T} (transition after week T)."""
    if p_list:
        return _predict_density_log(
            grid, weights, p_list[-1], blocks[-1], par,
        )
    if e1_known is None or not blocks:
        raise FloatingPointError("cannot form terminal predictive E_{T+1}")
    e1 = float(e1_known)
    b0 = blocks[0]
    mu = (
        par["a0"]
        + par["a1"] * e1
        + par["a2"] * b0["PV_sum_trans"]
        + par["a3"] * b0["FW_sum_trans"]
        + par["a4"] * b0["PJ_sum_trans"]
    )
    return _normalize_log_density(
        normal_logpdf(grid, mu, par["sigma_E"]),
        weights,
        label="predictive E_2",
    )


def _score_closing_sunday(
    grid: np.ndarray,
    weights: np.ndarray,
    blocks: list,
    p_list: list[np.ndarray],
    par: dict,
    loglik: float,
    *,
    e1_known: Optional[float] = None,
) -> tuple[float, float]:
    """Add log p(J_{W+1}, U_{W+1} | E_{W+1}) and return the posterior mean of E_{W+1}."""
    last = blocks[-1]
    j = last.get("J_week", np.nan)
    q, log_grid_mass = _predict_after_last_block(
        grid, weights, blocks, p_list, par, e1_known=e1_known,
    )
    terminal_mean = float(np.sum(grid * q * weights))
    if not np.isfinite(j):
        return loglik, terminal_mean
    log_ell = _ju_loglik_on_grid(
        grid, j, last.get("U1", np.nan), last.get("U2", np.nan), par,
        week_norm=_block_week_norm(last, extra=1),
    )
    m = np.max(log_ell)
    num = np.exp(log_ell - m) * q
    den = np.sum(num * weights)
    if (not np.isfinite(den)) or (den <= 0):
        raise FloatingPointError("invalid closing-Sunday c_{W+1}")
    log_c = float(m + np.log(den) + log_grid_mass)
    if not np.isfinite(log_c):
        raise FloatingPointError("invalid closing-Sunday log c_{W+1}")
    p = num / den
    terminal_mean = float(np.sum(grid * p * weights))
    return loglik + log_c, terminal_mean


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
    """Build E_2,...,E_{T+1} for export (filtered through E_T, then last Sunday)."""
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

    Each increment also includes ``log_grid_mass``, the log integral of the
    unnormalized Gaussian predictive over the grid (``<= 0``). That charges
    transition mass that would have landed off-grid; filtered densities are
    still renormalized on the grid.

    If e1_known is float: E_1 is the fixed baseline (week-1 start / lastweek);
    first term is log p(Y_1|E_1=e1) and hat q_2 transitions from that baseline
    to E_2 (end of week 1 / start of week 2). If None: week 1 uses prior
    N(m0,sigma0^2) on E_1.

    After week T, the last Sunday ``week_present`` (``J_{T+1}``) and its
    tools are scored on the predictive of ``E_{T+1}``; the returned
    ``terminal_predicted_mean`` is the posterior mean given that Sunday
    when it is observed.
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
            loglik, terminal_predicted_mean = _score_closing_sunday(
                grid, weights, blocks, p_list, par, loglik, e1_known=e1,
            )
            log_increments = np.asarray(log_c_list, dtype=float)
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
            # Defective Gaussian: charge off-grid predictive mass (log m <= 0).
            log_c = float(m + np.log(den) + log_grid_mass)
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
            # Defective Gaussian: charge off-grid predictive mass (log m <= 0).
            log_c = float(m + np.log(den) + log_grid_mass)
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

    loglik, terminal_predicted_mean = _score_closing_sunday(
        grid, weights, blocks, p_list, par, loglik,
        e1_known=float(e1_known) if e1_fixed else None,
    )
    log_increments = np.asarray(log_c_list, dtype=float)
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



# ``a1`` (theta[1]) is E_w's own week-to-week persistence, box-constrained to
# (-0.98, 0.98) in ``make_bounds``. That bound alone is not sufficient for
# stability, though: E_w also feeds back into E_{w+1} indirectly through
# PV/FW/PJ: E_w -> M^E (this week's mediators, via alpha1/beta1/theta1 and
# their action interactions) -> PV_w/FW_w/PJ_w -> E_{w+1} (via a2/a3/a4).
# Linearizing that whole path (including a1 itself) gives a *compound* loop
# gain
#   g = a1 + a2 * dPV_w/dE_w + a3 * dFW_w/dE_w + a4 * dPJ_w/dE_w.
# A gain >= 1 produces a saturating-random-walk failure mode: once E_w drifts
# to its clip floor/ceiling it stays there under any policy for the rest of
# the horizon, silencing the M^E -> E_w -> M^E/CAE feedback pathway that
# should otherwise counterbalance a persistently-applied action.
#
# The weekly summaries feeding a2/a3/a4 are fixed-denominator *averages*, not
# sums: ``build_user_blocks`` forms ``nansum(pv)/14`` and ``nansum(FW|PJ)/7``,
# and ``vani_env._week_means_from_arrays`` uses the identical convention at
# simulation time. A unit shift in E_w shifts every slot (day) of the week by
# the same per-slot slope, so the weekly average shifts by exactly that slope
# -- there is no 14x or 7x accumulation. FW and PJ are Bernoulli, so
# dFW_w/dE_w and dPJ_w/dE_w carry the logistic slope pi*(1-pi), fixed at its
# largest-magnitude value 0.25 at pi=0.5. PV is a hurdle on the z-scale:
# dE[z]/dE = p(1-p) α_E (z̄₊ - z_0) + p γ_E. The proxy uses 0.25 and p ≤ 1,
# and multiplies the occurrence term by (z̄₊ - z_0) from std_params (z̄₊ = 0
# among positives; z_0 is log(0.5) on that axis). That gap is ~2.85 in the
# current folder; omitting it understates the occurrence path. The penalty
# errs toward extra stability rather than needing a nested per-user
# prevalence estimate solved inside the optimizer. Both a
# "never suggest" (A=0) and "always suggest" (A=1) week are penalized, since g
# is close to affine in the fraction of slots suggested and these two bracket
# the range seen in simulation.
#
# The conservative pi=0.5 proxy can still be fooled: for a participant whose
# real FW/PJ prevalence is far from 50/50 (e.g. uid 151, who empirically wears
# the Fitbit ~87% of days but completes the daily check-in on only ~4%), the
# true pi*(1-pi) is much smaller than 0.25, so the real FW/PJ->E_w feedback is
# weaker than what the conservative g sees. The fit can then satisfy the
# barrier by letting a2/a3/a4 (poorly identified from only 12 weekly
# observations, especially when FW/PJ are themselves near-constant) cancel a1
# in the *conservative* g, while a1 itself still sits pinned at the +-0.98
# boundary -- unsafe in its own right, just masked by an over-generous proxy
# for the mediator path. The secondary barrier below re-adds a direct, much
# weaker penalty on a1 alone (on top of, not instead of, the compound-g
# barrier) so a1 cannot coast to the boundary by exploiting that gap.
LOGISTIC_SLOPE_PROXY = 0.25  # pi*(1-pi) at pi=0.5 (conservative upper bound)
LOOP_GAIN_BARRIER_WEIGHT = 3.0
LOOP_GAIN_DENOM_FLOOR = 1e-3  # keeps the barrier finite (no NaNs) even if |g| overshoots 1 mid-optimization
A1_INDEX = 1
A1_LIGHT_BARRIER_WEIGHT = 1.0  # weaker than the compound barrier's effective weight; a secondary guard, not the primary defense
_PV_OCCURRENCE_Z_GAP: Optional[float] = None


def _pv_occurrence_z_gap() -> float:
    """``z̄₊ - z_0`` from this folder's ``std_params.json`` (cached)."""
    global _PV_OCCURRENCE_Z_GAP
    if _PV_OCCURRENCE_Z_GAP is None:
        path = WORK_DIR / "std_params.json"
        with open(path, encoding="utf-8") as f:
            std = json.load(f)
        _PV_OCCURRENCE_Z_GAP = float(pv_hurdle_occurrence_z_gap(std))
    return _PV_OCCURRENCE_Z_GAP


def _loop_gain_barrier_term(g: float) -> float:
    """Smooth barrier that is ~0 for small |g|, grows sharply as |g| -> 1,
    and stays finite (still large, still pushing back) for |g| >= 1 -- unlike
    arctanh(g)^2, this never hits a domain error, which matters here because
    g (unlike a1) is not box-constrained and can transiently leave (-1, 1)
    during optimization.
    """
    g2 = g * g
    denom = max(1.0 - g2, LOOP_GAIN_DENOM_FLOOR)
    return LOOP_GAIN_BARRIER_WEIGHT * g2 / denom


def _a1_light_barrier_penalty(theta: np.ndarray) -> float:
    """Light-touch Fisher-z barrier on a1 alone, ~3x weaker than the barrier
    that used to be the sole guard on a1 before the compound-g barrier was
    added. See the comment above for why a1-only safety is still needed even
    with the compound barrier in place.
    """
    a1 = float(np.asarray(theta, dtype=float)[A1_INDEX])
    return A1_LIGHT_BARRIER_WEIGHT * float(np.arctanh(a1) ** 2)


def _loop_gain_penalty(theta: np.ndarray, e1_known: bool) -> float:
    p = unpack_theta(theta, e1_known=e1_known)
    a1, a2, a3, a4 = p["a1"], p["a2"], p["a3"], p["a4"]
    alpha1, alpha4 = p["alpha1"], p["alpha4"]
    beta1, beta4, beta6 = p["beta1"], p["beta4"], p["beta6"]
    theta1, theta4, theta6 = p["theta1"], p["theta4"], p["theta6"]

    # Weekly summaries are fixed-denominator averages, so each derivative is the
    # per-slot (per-day) slope itself; see the note above the constants.
    # PV: 0.25 α_E (z̄₊ - z_0) + γ_E. FW/PJ: 0.25 times the logit E-slope.
    # Worst |g| over J∈{0,1} matches tune_ste.loop_gain_Ew.
    occ_gap = _pv_occurrence_z_gap()

    def g(action: float, j: float) -> float:
        dPV = LOGISTIC_SLOPE_PROXY * occ_gap * (
            alpha1 + action * alpha4 + j * p["alpha_qE"]
        ) + (p["gamma1"] + action * p["gamma4"] + j * p["gamma_qE"])
        dFW = LOGISTIC_SLOPE_PROXY * (
            beta1 + action * (beta4 + beta6) + j * p["beta_qE"]
        )
        dPJ = LOGISTIC_SLOPE_PROXY * (
            theta1 + action * (theta4 + theta6) + j * p["theta_qE"]
        )
        return a1 + a2 * dPV + a3 * dFW + a4 * dPJ

    def worst_over_j(action: float) -> float:
        g0, g1 = g(action, 0.0), g(action, 1.0)
        return g0 if abs(g0) >= abs(g1) else g1

    return (
        _loop_gain_barrier_term(worst_over_j(0.0))
        + _loop_gain_barrier_term(worst_over_j(1.0))
        + _a1_light_barrier_penalty(theta)
    )


def neg_loglik_blocks(
    theta, blocks, grid, weights, e1_known, lam=0.5, prior_center=None,
    ridge_weights=None,
):
    """Per-participant penalized negative log-likelihood.

    ``prior_center`` (default all-zero, i.e. the original behavior) lets the
    L2 penalty shrink toward a non-zero target for specific coefficients
    instead of toward 0. Ordinary users use ``build_penalized_prior_center``
    (0.05 sign nudges). Strong-pool users (``strong_pool_uids``) pass the
    pooled MLE as ``prior_center`` and a larger ``lam``. Intercepts and
    log-σ are omitted from the ridge unless ``ridge_weights`` turns them
    back on (a0 and log σ_E for strong-pool users).

    In addition to that per-coefficient ridge, the *compound* loop gain
    through E_w's own persistence (a1) and its indirect PV/FW/PJ feedback
    path gets a barrier penalty, plus a lighter secondary barrier directly
    on a1 alone (as a backstop for participants whose real FW/PJ
    prevalence is far enough from 50/50 that the compound barrier's
    conservative proxy under-penalizes it); see ``_loop_gain_penalty``.
    """
    try:
        out = quadrature_loglik(blocks, theta, grid, weights, e1_known=e1_known)
        if not np.isfinite(out["loglik"]):
            return 1e100
        penalty = (
            _ridge_penalty(
                theta, lam, bool(e1_known), prior_center,
                ridge_weights=ridge_weights,
            )
            + _loop_gain_penalty(theta, e1_known)
        )
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

    # Apply ridge penalty once, not once per participant. Intercepts and
    # log-σ are unpenalized (same mask as the per-user fits).
    penalty = _ridge_penalty(theta, lam, bool(e1_known), prior_center=None)
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

    # log sigma parameters: σ_E, σ_U1, σ_U2, σ_PV (intensity)
    for idx in _LOG_SIGMA_INDICES:
        bounds[idx] = (np.log(0.03), np.log(10.0))

    # If E1 prior is estimated, bound log sigma0 too
    if not e1_known:
        bounds[-1] = (np.log(0.03), np.log(10.0))

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
        grid = e_quadrature_grid(pooled=True)

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
    prior_center: Optional[np.ndarray] = None,
    ridge_weights: Optional[np.ndarray] = None,
    ridge_center_label: str = "sign_nudge",
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
        args=(blocks, grid, weights, e1_known, lam, prior_center, ridge_weights),
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
    filt["ridge_center"] = str(ridge_center_label)
    filt["ridge_penalize_a0_sigma_e"] = bool(
        ridge_weights is not None
        and np.asarray(ridge_weights, dtype=float)[0] > 0
    )

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
    prior_center: Optional[np.ndarray] = None,
    merge_into_vanilla_json: bool = False,
    vanilla_work_dir: Optional[Path] = None,
    json_digits: int = 3,
    n_jobs: Optional[int] = None,
):
    if grid is None:
        grid = e_quadrature_grid(pooled=False)

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
        "prior_center": prior_center,
        "ridge_center_label": "sign_nudge",
    }
    e1_fixed = e1_known is not None
    strong_uids = strong_pool_uids()
    strong_mult = strong_pool_lam_mult()
    present_strong = sorted(
        int(uid) for uid, _ in user_data if int(uid) in strong_uids
    )
    if present_strong:
        if pooled_x0 is None:
            print(
                "WARNING: strong-pool uids "
                f"{present_strong} requested but pooled_x0 is missing; "
                "falling back to the sign-nudge ridge."
            )
            present_strong = []
        else:
            print(
                f"Strong pooling toward pooled MLE for {present_strong} "
                f"(lam={float(lam) * strong_mult:g} = {lam:g}×{strong_mult:g}; "
                "also penalize a0 and log σ_E). Other users keep the "
                f"sign-nudge center at lam={lam:g}."
            )

    def kwargs_for(uid) -> dict:
        kw = dict(fit_kwargs)
        if int(uid) not in present_strong:
            return kw
        kw["lam"] = float(lam) * strong_mult
        kw["prior_center"] = np.asarray(pooled_x0, dtype=float)
        kw["ridge_weights"] = ridge_penalty_weights(
            e1_known=e1_fixed, penalize_a0_sigma_e=True,
        )
        kw["ridge_center_label"] = "pooled_mle"
        return kw
    results = {}
    filtered_states = {}
    blocks_by_user = {}

    def record_fit(uid, res, filt, blocks):
        results[uid] = res
        filtered_states[uid] = filt
        blocks_by_user[uid] = blocks
        print(
            f"Participant {uid}: success={res.success}, "
            f"nll={res.fun:.4f}, nit={res.nit}, "
            f"ridge={filt.get('ridge_center')} lam={filt.get('ridge_lambda')}, "
            f"message={res.message}, "
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
            record_fit(*_fit_one_user_task(uid, g, grid, kwargs_for(uid)))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    _fit_one_user_task, uid, g, grid, kwargs_for(uid)
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

    # --- b_0, b_1, b_2 (J_w = week_present_lastweek) ---
    fig_b, axes_b = plt.subplots(1, 3, figsize=(14, 3.2), squeeze=False)
    _bar_users_one_series(
        axes_b[0][0],
        "Estimate",
        r"$b_0$ — $\mathrm{logit}(p_w)= b_0 + b_1 E_w + b_2 \mathrm{week\_norm}$",
        [float(filtered_states[u]["params"]["b0"]) for u in _users],
    )
    _bar_users_one_series(
        axes_b[0][1],
        "Estimate",
        r"$b_1$ — coefficient on $E_w$ for $J_w^{\mathrm{week}}$",
        [float(filtered_states[u]["params"]["b1"]) for u in _users],
    )
    _bar_users_one_series(
        axes_b[0][2],
        "Estimate",
        r"$b_2$ — $\mathrm{week\_norm}$",
        [float(filtered_states[u]["params"]["b2_week_norm"]) for u in _users],
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
    fig_n, axes_n = plt.subplots(1, 3, figsize=(14, 3.2), squeeze=False)
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
    _bar_users_one_series(
        axes_n[0][2],
        "Estimate",
        r"$\log(\sigma_{\mathrm{PV}}^2)$ (intensity)",
        [
            float(np.log(filtered_states[u]["params"]["sigma_PV"] ** 2))
            for u in _users
        ],
    )
    fig_n.suptitle(r"Gaussian emission noise", fontsize=11, y=1.05)
    fig_n.tight_layout()
    _maybe_save_close(fig_n, "params_group_log_sigma2.png")

    # --- PV hurdle: alpha, AR(1) ---
    fig_pv, axes_pv = plt.subplots(2, 4, figsize=(14, 6.0), squeeze=False)
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
    ]
    for k, (tit, fn) in enumerate(_pv_specs):
        vals = [float(fn(filtered_states[u]["params"])) for u in _users]
        _bar_users_one_series(flat[k], "Estimate", tit, vals)
    fig_pv.suptitle(
        r"Hourly pageview hurdle logit $P(\mathrm{count}>0)$ — context, lag1, action",
        fontsize=11,
        y=1.02,
    )
    fig_pv.tight_layout()
    _maybe_save_close(fig_pv, "params_group_pv.png")

    fig_pv_z, axes_pv_z = plt.subplots(2, 4, figsize=(14, 6.0), squeeze=False)
    flat_z = np.ravel(axes_pv_z)
    _pv_z_specs = [
        (r"$\gamma_0$", lambda p: p["gamma0"]),
        (r"$\gamma_1$ ($E_w$)", lambda p: p["gamma1"]),
        (r"$\gamma_{2,\mathrm{we}}$", lambda p: p["gamma2_is_weekend"]),
        (r"$\gamma_{2,\mathrm{dt}}$", lambda p: p["gamma2_dt"]),
        (r"$\gamma_{2,\mathrm{rb}}$", lambda p: p["gamma2_rb"]),
        (r"$\gamma_{\mathrm{AR}}$", lambda p: p["gamma_ar1"]),
        (r"$\gamma_3$ ($A$)", lambda p: p["gamma3"]),
        (r"$\gamma_4$ ($A\cdot E_w$)", lambda p: p["gamma4"]),
    ]
    for k, (tit, fn) in enumerate(_pv_z_specs):
        vals = [float(fn(filtered_states[u]["params"])) for u in _users]
        _bar_users_one_series(flat_z[k], "Estimate", tit, vals)
    fig_pv_z.suptitle(
        r"Hourly pageview hurdle intensity (Gaussian on log-then-z $\mid$ count$>0$)",
        fontsize=11,
        y=1.02,
    )
    fig_pv_z.tight_layout()
    _maybe_save_close(fig_pv_z, "params_group_pv_intensity.png")

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
        grid=e_quadrature_grid(pooled=True),
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

    # Stage 2: user-specific fits initialized at pooled estimate. Ordinary
    # users keep the 0.05 sign-nudge ridge. Strong-pool users (default 248)
    # shrink toward this pooled MLE with a larger λ, including a0 and log σ_E.
    _e1_known = 2.0
    results, filtered_states, blocks_by_user = fit_all_users(
        df_fit,
        grid=e_quadrature_grid(pooled=False),
        e1_known=_e1_known,
        maxiter=500,
        pooled_x0=pooled_x0,
        lam=0.5,
        prior_center=build_penalized_prior_center(e1_known=(_e1_known is not None)),
        merge_into_vanilla_json=True,
        vanilla_work_dir=WORK_DIR,
        json_digits=3,
        n_jobs=None,
    )

    # Write filtered E_w plug-in back to df_fit.csv for script 5
    # (contemporaneous Ê_{w|w} in perceived_utility_lastweek; not pre-week).
    df_fit_out = save_df_fit_with_perceived_utility(
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
        show=False,
    )
    print(f"Diagnostic figures saved under {_diag_dir}")
