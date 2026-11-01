"""Hierarchical Bayesian perceived-utility state-space model (self-contained).

Fits the perceived-utility (PU) process of the testbed jointly across MRT
participants by NUTS and writes the same output files as the superseded
penalized-likelihood script in ``archive/superseded/``, so ``07_fit_testbed.py``,
``08_est_ew_weights.py``, and ``vani_env`` are unchanged. This script does not
import that archived fit.

Model (Section "Perceived-Utility Process" of the manuscript)
-------------------------------------------------------------
Latent state, ``w = 2, ..., W + 1``::

    E_w = a0 + a1 E_{w-1} + a2 PV_{w-1} + a3 FW_{w-1} + a4 PJ_{w-1} + sigma_E eta_w,

with ``E_1 = 2`` and ``sigma_E = 0.3`` fixed for every participant. These two
normalizations fix the location and scale of the latent variable: shifting or
rescaling ``E`` while adjusting intercepts and slopes leaves the likelihood
unchanged. ``PV/FW/PJ_w`` are fixed-denominator weekly means (/14, /7).

Emissions of ``E_w``:
* opening weekly check-in ``J_w ~ Bernoulli(logit^-1(b0 + b1 E_w))`` and, if
  ``J_w = 1``, ``U_{w,1} ~ N(c0 + c1 E_w, s_U1^2)``, ``U_{w,2} ~ N(d0 + d1 E_w, s_U2^2)``;
  week 1 has no opening check-in; the check-in closing the last week is an
  emission of ``E_{W+1}``;
* page-view hurdle per decision slot: occurrence (logistic, ``alpha*``; also
  ``alpha_inter`` x interaction with a delivered suggestion) and positive
  intensity on the standardized scale (Gaussian, ``gamma*``, ``s_PV``);
* Fitbit wear and daily check-in per day (logistic, ``beta*`` and ``theta*``).
Each engagement linear predictor contains an intercept, ``E_w``, weekend,
(slot), burden, its own lag, suggestion(s) ``x (1, E_w)``, and the carry-over
``J_w (q0 + qE E_w + qrb burden)``.

Priors (same principles as the CAE models of ``07_fit_testbed.py``)
* participant coefficient = population mean + random effect,
  ``psi_ik = mu_k + tau_k z_ik``, ``z_ik ~ N(0, 1)``,
  ``tau_k ~ Gamma(shape=2, mode=RE_SD_MODE)`` for every coefficient
  except the two rating loadings ``c1``, ``d1`` (``COMMON_COEFS``), which are
  common to all participants;
* population means: coefficients with a direction in the working causal model
  ``mu_k = s_k |v_k|, v_k ~ N(0, 1)`` (direction only; individuals may differ
  in sign); all others ``N(0, 10)``; no coefficient is bounded;
* residual SDs of ``U1``, ``U2`` and page-view intensity shared by all
  participants, ``s^2 ~ InverseGamma(2, 0.5 Var(y))`` (for a standardized
  outcome this is the CAE prior ``InverseGamma(2, 0.5)``).

Panel: the fitting panel of ``03_standardize.py`` (MRT weeks 2-12, renumbered 1-11; week 1
supplies only lags). ``E_1 = 2`` is the state of the first fitted week.

Missing data (handled when the weekly blocks are built): a missing page-view
slot is a zero count (standardized zero-count value), a missing wear or
check-in day is 0, missing burden is the participant mean; a week without
engagement data uses the participant's mean weekly summary in the transition
and contributes no emissions.

Activity filter: page-view, wear, and daily check-in emissions are kept only
on days with ``active_any = 1`` in the daily activity table. The weekly
check-in (and the tool ratings on that same Sunday) is kept only when that
survey day is active. A date absent from the 84-day table is not treated as
active. Weekly transition means are unchanged. The table is
``participant_dropout_daily.csv`` written by ``01_extract_data.py`` next to
``df_fit.csv``. ``ADAPR_ACTIVITY_DAILY`` overrides that path. A missing
table is an error.

Outputs (in the parameter directory; default ``env_para_vanilla/``)
* ``params_env_<uid>.json``: ``theta_penalized_{Ew,J,U1,U2,PV,FW,PJ}`` (+
  ``_names``; key names kept for the downstream readers) at the posterior mean,
  and emission residuals ``resid_penalized_*`` for the simulator's bootstrap;
* ``pred_<uid>.json``: fitted values and ``pred_penalized_filtered_Ew`` =
  posterior mean of ``E_2, ..., E_{W+1}``;
* ``df_fit.csv`` (in ``ADAPR_COMBINED_DIR``): ``perceived_utility_lastweek``
  (``Ê_w`` on week w) and ``perceived_utility`` (``Ê_{w+1}``), ``Ê`` = posterior
  mean of ``E`` given all data;
* ``pu_hier_posterior.nc`` and ``pu_hier_diagnostics.json`` (divergences,
  R-hat, ESS). Stability of the fitted models is checked by ``stability.py``.

Usage::

    ADAPR_COMBINED_DIR=... python 06_fit_pu.py \
        [--chains 4 --cores 4 --tune 2000 --draws 1000 --target-accept 0.95 --max-treedepth 12] \
        [--users 13,18,22] [--no-write] [--diag-dir DIR]

Next: ``07_fit_testbed.py``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dropout_utils import in_study_mask, weekly_checkin_completion_lag
from vani_env import mask_dropout_fit_outcomes, resolve_params_dir

logger = logging.getLogger("pu_hier")

WORK_DIR = resolve_params_dir()
ACTIVITY_DAILY_NAME = "participant_dropout_daily.csv"


def resolve_activity_daily_path(combined_dir: Path) -> Path:
    """Daily activity table used to drop inactive-day emissions."""
    env = os.environ.get("ADAPR_ACTIVITY_DAILY", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"ADAPR_ACTIVITY_DAILY does not exist: {path}")
        return path
    path = combined_dir / ACTIVITY_DAILY_NAME
    if not path.is_file():
        raise SystemExit(
            f"Missing {path}. Script 1 writes {ACTIVITY_DAILY_NAME} next to "
            "participant_dropout.csv. Set ADAPR_ACTIVITY_DAILY to override."
        )
    return path


def load_activity_lookup(path: Path) -> dict:
    """Map (participant id, YYYY-MM-DD) to active_any in {0, 1}."""
    daily = pd.read_csv(path)
    need = {"ParticipantIdentifier", "Date", "active_any"}
    missing = need.difference(daily.columns)
    if missing:
        raise SystemExit(f"{path} is missing {sorted(missing)}")
    # Unobserved days are censoring, not inactivity, and have no active_any.
    if "observed" in daily.columns:
        observed = pd.to_numeric(daily["observed"], errors="coerce").fillna(0).eq(1)
        daily = daily.loc[observed].copy()
    uid = daily["ParticipantIdentifier"].astype(str)
    date = pd.to_datetime(daily["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    flag = pd.to_numeric(daily["active_any"], errors="coerce")
    if date.isna().any() or flag.isna().any():
        raise SystemExit(f"{path} has observed rows without a date or active_any")
    keys = list(zip(uid.tolist(), date.tolist()))
    if len(keys) != len(set(keys)):
        raise SystemExit(f"{path} has duplicate participant-dates")
    return {key: float(value) for key, value in zip(keys, flag.tolist())}


def _activity_bucket(flag):
    """``None`` keeps the emission. Otherwise ``inactive`` or ``outside``."""
    if flag is None:
        return None
    try:
        value = float(flag)
    except (TypeError, ValueError):
        return "outside"
    if not np.isfinite(value):
        return "outside"
    if value == 1.0:
        return None
    return "inactive"


def assert_activity_covers_fit(df_fit: pd.DataFrame, activity_lookup: dict) -> None:
    """Every df_fit calendar day must be in the activity table."""
    days = df_fit.loc[:, ["ParticipantIdentifier", "Date"]].drop_duplicates()
    uid = days["ParticipantIdentifier"].astype(str)
    date = pd.to_datetime(days["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    missing = [key for key in zip(uid.tolist(), date.tolist()) if key not in activity_lookup]
    if missing:
        sample = ", ".join(f"{u} {d}" for u, d in missing[:8])
        raise SystemExit(
            f"{len(missing)} df_fit days are absent from the activity table "
            f"(for example {sample})."
        )


def _lookup_active(activity_lookup, uid, day) -> float:
    if activity_lookup is None:
        return 1.0
    key = (str(uid), pd.Timestamp(day).strftime("%Y-%m-%d"))
    return activity_lookup.get(key, np.nan)


def fit_rows_active_mask(df: pd.DataFrame, activity_lookup) -> np.ndarray:
    """True iff the row's calendar day has ``active_any = 1``.

    ``activity_lookup is None`` keeps every row, matching
    :func:`_lookup_active`. A date absent from the lookup is not active
    (the same "outside" bucket the PU emission filter uses).
    """
    if activity_lookup is None:
        return np.ones(len(df), dtype=bool)
    uid = df["ParticipantIdentifier"].to_numpy()
    date = df["Date"].to_numpy()
    return np.array(
        [
            _activity_bucket(_lookup_active(activity_lookup, u, d)) is None
            for u, d in zip(uid, date)
        ],
        dtype=bool,
    )


def _window_active(activity_lookup, uid, anchor, completion_days: int) -> float:
    """1 if any day in the check-in window is active, else 0 or missing.

    The window is the Sunday slot through ``completion_days`` later. A
    forced zero requires every one of those days to be inactive. One active
    day keeps the check-in. Incomplete coverage with no active day is NaN.
    """
    if activity_lookup is None:
        return 1.0
    flags = [
        _lookup_active(
            activity_lookup, uid, pd.Timestamp(anchor) + pd.Timedelta(days=k)
        )
        for k in range(int(completion_days) + 1)
    ]
    if any(np.isfinite(flag) and float(flag) == 1.0 for flag in flags):
        return 1.0
    if flags and all(np.isfinite(flag) and float(flag) == 0.0 for flag in flags):
        return 0.0
    return np.nan

E1_FIXED = 2.0
# Scale normalization of the latent state (location: E_1 = E1_FIXED).
SIGMA_E_FIXED = 0.3
# One random-effect SD prior for every coefficient, Gamma(2, mode = RE_SD_MODE).
# Coefficients are mostly on the logit scale, so the mode is larger than the
# CAE models' 0.15 (standardized continuous outcomes).
RE_SD_MODE = 0.5
# Rating loadings are common to all participants (no random effect). Each
# participant has only ~8 weekly ratings, and with participant-specific rating
# loadings an individual's loading could take either sign (about -1 or +1),
# letting that participant's latent state follow either the ratings or the
# engagement mediators; the posterior was bimodal with chains disagreeing.
# All other coefficients, including the rating intercepts, keep random effects.
COMMON_COEFS = ("c1", "d1")
# Population means: N(0, 10) (SD sqrt(10)) unless directional.
POP_SD = float(np.sqrt(10.0))
# Directional population means: sign * |N(0, DIRECTIONAL_SCALE^2)|.
DIRECTIONAL_SCALE = 1.0
# Residual SDs shared across participants (as in the CAE models).
SHARED_NOISE = ("sigma_U1", "sigma_U2", "sigma_PV")

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

_Q = ("q0", "qE", "qrb")
PARAM_NAMES = (
    ["a0", "a1", "a2", "a3", "a4", "b0", "b1", "c0", "c1", "d0", "d1"]
    + [f"alpha{s}" for s in ("0", "1", "2_is_weekend", "2_dt", "2_rb", "_ar1", "3", "4")]
    + [f"alpha_{q}" for q in _Q]
    + ["alpha_inter"]
    + [f"gamma{s}" for s in ("0", "1", "2_is_weekend", "2_dt", "2_rb", "_ar1", "3", "4")]
    + [f"gamma_{q}" for q in _Q]
    + [f"beta{s}" for s in ("0", "1", "2_is_weekend", "2_rb", "_ar1", "3", "4", "5", "6")]
    + [f"beta_{q}" for q in _Q]
    + [f"theta{s}" for s in ("0", "1", "2_is_weekend", "2_rb", "_ar1", "3", "4", "5", "6")]
    + [f"theta_{q}" for q in _Q]
)

# Directions from the working causal model (population level only).
DIRECTION = {
    # engagement last week -> perceived utility next week
    "a2": +1, "a3": +1, "a4": +1,
    # perceived utility -> engagement
    "alpha1": +1, "gamma1": +1, "beta1": +1, "theta1": +1,
    # measurement of perceived utility by the weekly check-in: completion and
    # the helpfulness / pleasantness ratings increase with E (positive
    # loadings). Without these, the sign of the rating loadings is not
    # identified (weekly ratings are nearly uncorrelated with engagement
    # within person) and chains settle in different modes.
    "b1": +1, "c1": +1, "d1": +1,
    # suggestion -> engagement (attention cost)
    "alpha3": -1, "gamma3": -1, "beta3": -1, "beta5": -1, "theta3": -1, "theta5": -1,
    # suggestion x perceived utility -> engagement (cost offset for high utility)
    "alpha4": +1, "gamma4": +1, "beta4": +1, "beta6": +1, "theta4": +1, "theta6": +1,
}

_QUERY_NAMES = ["query_Jw_intercept", "query_Jw_Ew", "query_Jw_recent_burden"]
_QUERY_INT_NAMES = ["intensity_query_Jw_intercept", "intensity_query_Jw_Ew",
                    "intensity_query_Jw_recent_burden"]
# JSON layout read by vani_env / scripts 5-6: (key, [(parameter, exported name), ...]).
EXPORT_LAYOUT = [
    ("theta_penalized_Ew", [("a0", "a0"), ("a1", "a1"), ("a2", "a2_PV_lag_week"),
                            ("a3", "a3_FW_lag_week"), ("a4", "a4_PJ_lag_week"),
                            ("sigma_E", "sigma_E")]),
    ("theta_penalized_J", [("b0", "b0"), ("b1", "b1_Ew")]),
    ("theta_penalized_U1", [("c0", "c0"), ("c1", "c1_Ew"), ("sigma_U1", "sigma_U1")]),
    ("theta_penalized_U2", [("d0", "d0"), ("d1", "d1_Ew"), ("sigma_U2", "sigma_U2")]),
    ("theta_penalized_PV",
     [("alpha0", "alpha0"), ("alpha1", "alpha1_Ew"), ("alpha2_is_weekend", "alpha2_is_weekend"),
      ("alpha2_dt", "alpha2_decision_time"), ("alpha2_rb", "alpha2_recent_burden"),
      ("alpha_ar1", "alpha_ar1_hourly_pageview_lag1"), ("alpha3", "alpha3_action"),
      ("alpha4", "alpha4_action_by_Ew")]
     + list(zip([f"alpha_{q}" for q in _Q], _QUERY_NAMES))
     + [("gamma0", "gamma0"), ("gamma1", "gamma1_Ew"), ("gamma2_is_weekend", "gamma2_is_weekend"),
        ("gamma2_dt", "gamma2_decision_time"), ("gamma2_rb", "gamma2_recent_burden"),
        ("gamma_ar1", "gamma_ar1_hourly_pageview_lag1"), ("gamma3", "gamma3_action"),
        ("gamma4", "gamma4_action_by_Ew")]
     + list(zip([f"gamma_{q}" for q in _Q], _QUERY_INT_NAMES))
     + [("sigma_PV", "sigma_PV")]),
    # Interaction with a delivered suggestion -> page-view occurrence (logit).
    ("theta_penalized_PV_interaction", [("alpha_inter", "alpha_interaction_delivered")]),
    ("theta_penalized_FW",
     [("beta0", "beta0"), ("beta1", "beta1_Ew"), ("beta2_is_weekend", "beta2_is_weekend"),
      ("beta2_rb", "beta2_recent_burden"), ("beta_ar1", "beta_ar1_morning_wearing"),
      ("beta3", "beta3_A0_morning"), ("beta4", "beta4_A0_morning_by_Ew"),
      ("beta5", "beta5_A1_afternoon"), ("beta6", "beta6_A1_afternoon_by_Ew")]
     + list(zip([f"beta_{q}" for q in _Q], _QUERY_NAMES))),
    ("theta_penalized_PJ",
     [("theta0", "theta0"), ("theta1", "theta1_Ew"), ("theta2_is_weekend", "theta2_is_weekend"),
      ("theta2_rb", "theta2_recent_burden"), ("theta_ar1", "theta_ar1_daily_present_yesterday"),
      ("theta3", "theta3_A0_morning"), ("theta4", "theta4_A0_morning_by_Ew"),
      ("theta5", "theta5_A1_afternoon"), ("theta6", "theta6_A1_afternoon_by_Ew")]
     + list(zip([f"theta_{q}" for q in _Q], _QUERY_NAMES))),
]


# --------------------------------------------------------------------------
# Weekly blocks (same construction as the penalized script)
# --------------------------------------------------------------------------

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
    inter_col="Interacted_walk",
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
    activity_lookup=None,
    completion_days=2,
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

      - Weekly transition summaries use fixed 14/7 denominators while the
        participant is enrolled: a missing PV slot counts as standardized
        zero and missing FW/PJ as 0. Post-dropout slots are excluded from both
        numerator and denominator. Missing PV lags are ``z0``; missing recent
        burden is the participant mean. These summaries are not restricted
        to active days.

      - When ``activity_lookup`` is set, each emission is gated on the date
        it refers to. PV and PJ use the row's calendar day. FW is
        ``nextday_wearing``: the row for day d is the morning of day d+1, so
        it uses ``active(d+1)``. ``J_w`` (and U1/U2 on that check-in) is
        dropped only when every day from the Sunday slot through
        ``completion_days`` later is inactive. A date missing from the
        lookup is outside the activity window.

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

    # Missing-covariate rules (Section "Missing data in model fitting"):
    #   * page views: a missing slot is a zero count, imputed on the raw scale
    #     before standardization, i.e. at the standardized zero-count value z0;
    #   * FW / PJ: missing = 0 (not worn / not completed);
    #   * recent burden: participant mean (as in the CAE-related models).
    _zero_rows = pd.to_numeric(dat[PV_log_col], errors="coerce") == 0
    _z0_vals = pd.to_numeric(dat.loc[_zero_rows, PV_col], errors="coerce").dropna()
    if len(_z0_vals):
        pv_z0 = float(_z0_vals.iloc[0])
    else:
        _pv_all = pd.to_numeric(dat[PV_col], errors="coerce").dropna()
        pv_z0 = float(_pv_all.min()) if len(_pv_all) else 0.0
    _burden_all = pd.to_numeric(dat[burden_col], errors="coerce")
    burden_mean = float(_burden_all.mean()) if _burden_all.notna().any() else 0.0

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

        pv_y_list, pv_z_list, pv_a_list, pv_i_list = [], [], [], []
        pv_lag1_list = []
        pv_ctx_rows = []
        a0_list, a1_list = [], []
        fw_list, pj_list = [], []
        slot_eligible_list, day_eligible_list = [], []
        day_is_weekend_list, day_burden_list = [], []
        day_fw_lag_list, day_pj_lag_list = [], []
        slot_active_list, day_active_list = [], []
        slot_date_list, day_date_list = [], []
        fw_active_list, fw_date_list = [], []
        uid = str(g_week["ParticipantIdentifier"].iloc[0])

        for _d, g_day in g_week.groupby(date_col, sort=True):
            g_day = g_day.sort_values(decision_col, na_position="last")
            eligible_slots = in_study_mask(g_day)

            aa = g_day[a_col].to_numpy(dtype=float)
            ii = (g_day[inter_col].to_numpy(dtype=float) if inter_col in g_day.columns
                  else np.full(len(g_day), np.nan))
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
                slot_eligible_list.append(bool(eligible_slots[k]))
                z_k = float(pv[k]) if not np.isnan(pv[k]) else np.nan
                pv_z_list.append(z_k)
                lv = float(pv_log[k]) if k < len(pv_log) and not np.isnan(pv_log[k]) else np.nan
                if np.isnan(lv):
                    pv_y_list.append(np.nan)
                else:
                    # Raw count > 0. Also true of a leftover log column.
                    pv_y_list.append(1.0 if lv > 0.0 else 0.0)
                pv_a_list.append(float(aa[k]) if not np.isnan(aa[k]) else np.nan)
                # interaction with a delivered suggestion; missing = no interaction
                pv_i_list.append(float(ii[k]) if np.isfinite(ii[k]) else 0.0)
                pv_lag1_list.append(
                    float(lag1v[k]) if not np.isnan(lag1v[k]) else pv_z0
                )
                weekend_k = float(weekendv[k]) if not np.isnan(weekendv[k]) else np.nan
                rb_k = float(burden[k]) if not np.isnan(burden[k]) else burden_mean
                dt_k = float(dtv[k]) if (hourly_pv and not np.isnan(dtv[k])) else 0.0
                pv_ctx_rows.append([weekend_k, dt_k, rb_k])

            day_key = pd.Timestamp(_d).strftime("%Y-%m-%d")
            day_flag = _lookup_active(activity_lookup, uid, _d)
            next_day = pd.Timestamp(_d).normalize() + pd.Timedelta(days=1)
            for _k in range(len(g_day)):
                slot_active_list.append(day_flag)
                slot_date_list.append(day_key)
            day_active_list.append(day_flag)
            day_date_list.append(day_key)
            # nextday_wearing on this row is the morning of the next calendar day.
            fw_active_list.append(_lookup_active(activity_lookup, uid, next_day))
            fw_date_list.append(next_day.strftime("%Y-%m-%d"))

            row_morning = g_day.loc[g_day[decision_col] == 0].iloc[0]
            row_afternoon = g_day.loc[g_day[decision_col] == 1].iloc[0]
            day_eligible_list.append(bool(eligible_slots.any()))

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
                float(row_afternoon[burden_col]) if not pd.isna(row_afternoon[burden_col]) else burden_mean
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
        slot_eligible = np.asarray(slot_eligible_list, dtype=bool)
        day_eligible = np.asarray(day_eligible_list, dtype=bool)

        n_slot_eligible = int(slot_eligible.sum())
        n_pv_obs = int(np.sum(np.isfinite(pv_z) & slot_eligible))
        if n_slot_eligible and n_pv_obs:
            PV_sum = float(
                (
                    np.nansum(pv_z[slot_eligible])
                    + (n_slot_eligible - n_pv_obs) * pv_z0
                )
                / n_slot_eligible
            )
        else:
            PV_sum = np.nan
        n_day_eligible = int(day_eligible.sum())
        FW_sum = (
            float(np.nansum(FW_daily[day_eligible]) / n_day_eligible)
            if n_day_eligible and np.any(np.isfinite(FW_daily[day_eligible]))
            else np.nan
        )
        PJ_sum = (
            float(np.nansum(PJ_daily[day_eligible]) / n_day_eligible)
            if n_day_eligible and np.any(np.isfinite(PJ_daily[day_eligible]))
            else np.nan
        )
        week_dates = pd.to_datetime(g_week[date_col])
        # Opening J is the Sunday before this Monday. Closing J is this Sunday.
        # Kept unless every day in the completion window is inactive.
        open_sunday = week_dates.min() - pd.Timedelta(days=1)
        close_sunday = week_dates.max()
        j_lag_active = _window_active(
            activity_lookup, uid, open_sunday, completion_days
        )
        j_week_active = _window_active(
            activity_lookup, uid, close_sunday, completion_days
        )
        return {
            "week": int(week),
            "is_missing_week": False,
            "J_week": J_week,
            "J_lag": J_lag,
            "J_week_active": j_week_active,
            "J_lag_active": j_lag_active,
            "J_week_date": pd.Timestamp(close_sunday).strftime("%Y-%m-%d"),
            "J_lag_date": pd.Timestamp(open_sunday).strftime("%Y-%m-%d"),
            "U1": U1,
            "U2": U2,
            "pv_y": pv_y,
            "pv_z": pv_z,
            "pv_lag1": pv_lag1,
            "pv_c_ctx": pv_c_ctx,
            "pv_a": np.asarray(pv_a_list, dtype=float),
            "pv_i": np.asarray(pv_i_list, dtype=float),
            "day_A0": day_A0,
            "day_A1": day_A1,
            "day_is_weekend": day_is_weekend,
            "day_burden": day_burden,
            "day_FW_lag": day_FW_lag,
            "day_PJ_lag": day_PJ_lag,
            "FW_daily": FW_daily,
            "PJ_daily": PJ_daily,
            "slot_active": np.asarray(slot_active_list, dtype=float),
            "slot_date": slot_date_list,
            "day_active": np.asarray(day_active_list, dtype=float),
            "day_date": day_date_list,
            "fw_active": np.asarray(fw_active_list, dtype=float),
            "fw_date": fw_date_list,
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
                "J_week_active": np.nan,
                "J_lag_active": np.nan,
                "U1": np.nan,
                "U2": np.nan,
                "U1_open": np.nan,
                "U2_open": np.nan,
                "pv_y": np.asarray([], dtype=float),
                "pv_z": np.asarray([], dtype=float),
                "pv_lag1": np.asarray([], dtype=float),
                "pv_c_ctx": np.zeros((0, 3), dtype=float),
                "pv_a": np.asarray([], dtype=float),
                "pv_i": np.asarray([], dtype=float),
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

    by_week = {int(b["week"]): b for b in blocks}
    for b in blocks:
        if np.isfinite(b.get("J_lag", np.nan)):
            continue
        prev = by_week.get(int(b["week"]) - 1)
        if prev is not None and np.isfinite(prev.get("J_week", np.nan)):
            b["J_lag"] = float(prev["J_week"])
            b["J_lag_active"] = prev.get("J_week_active", np.nan)
            b["J_lag_date"] = prev.get("J_week_date")
        # else leave NaN: no fabricated J=0 Bernoulli at study entry.

    prev_u1, prev_u2 = np.nan, np.nan
    for b in blocks:
        # Opening U_w is last Sunday's tools (same survey as J_w).
        b["U1_open"] = prev_u1
        b["U2_open"] = prev_u2
        prev_u1 = b.get("U1", np.nan)
        prev_u2 = b.get("U2", np.nan)

    return blocks



def _block_jw(block) -> float:
    """Opening ``J_w`` on PV/FW/PJ. Missing ``J_lag`` (including week 1) → 0."""
    j = block.get("J_lag", np.nan)
    if j is None or not np.isfinite(j):
        return 0.0
    return float(j)



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
    inter_col="Interacted_walk",
    weekend_col="is_weekend",
    burden_col="recent_burden_norm",
    PV_col="HourlyPageviewCount_norm",
    PV_lag1_col="hourly_pageview_count_lag1",
    FW_lag_col="morning_wearing",
    PJ_lag_col="daily_present_yesterday",
    hourly_pv=True,
    activity_lookup=None,
    completion_days=2,
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
            full_weeks=range(1, int(df_fit["week"].max()) + 1),
            missing_week_mode="mean_impute_transition",
            activity_lookup=activity_lookup,
            completion_days=completion_days,
        )
        user_blocks[uid] = blocks
    return user_blocks



# --------------------------------------------------------------------------
# Long-format data
# --------------------------------------------------------------------------

def flatten_blocks(user_blocks: dict):
    """Long-format arrays indexed by (participant u, state index t).

    State index t = 0..T: t = 0 is the fixed E_1, t = k is block k's E
    (block 0 is week 1), and t = T is E_{W+1} (closing Sunday only).
    """
    uids = sorted(user_blocks)
    T = max(len(b) for b in user_blocks.values())
    if any(len(b) != T for b in user_blocks.values()):
        raise ValueError("all participants must have the same number of weekly blocks")

    trans = np.zeros((len(uids), T, 3))  # PV/FW/PJ summaries of block t-1 -> E_t
    ju_rows, pv_rows, day_rows = [], [], {"FW": [], "PJ": []}
    excl_days = {"inactive": set(), "outside": set()}
    excl_weeks = {"inactive": 0, "outside": 0}
    nonzero = {"pv": [], "fw": [], "pj": [], "j": []}

    def _dropped(flag, kind, key=None) -> bool:
        bucket = _activity_bucket(flag)
        if bucket is None:
            return False
        if kind == "day":
            excl_days[bucket].add(key)
        else:
            excl_weeks[bucket] += 1
        return True

    def _note(kind, uid, date, y) -> None:
        if np.isfinite(y) and float(y) > 0:
            nonzero[kind].append((str(uid), None if date is None else str(date), float(y)))

    for u, uid in enumerate(uids):
        blocks = user_blocks[uid]
        for t, b in enumerate(blocks):
            trans[u, t] = [b["PV_sum_trans"], b["FW_sum_trans"], b["PJ_sum_trans"]]
            jw = _block_jw(b)
            j_open = b.get("J_lag", np.nan)
            if np.isfinite(j_open) and not _dropped(b.get("J_lag_active", None), "week"):
                ju_rows.append((u, t, j_open, b.get("U1_open", np.nan), b.get("U2_open", np.nan)))
            elif np.isfinite(j_open):
                _note("j", uid, b.get("J_lag_date"), j_open)
            if b.get("is_missing_week"):
                continue
            pv_y, pv_z = b["pv_y"], b.get("pv_z")
            slot_active = b.get("slot_active")
            slot_date = b.get("slot_date")
            for j in range(len(pv_y)):
                y = pv_y[j]
                if not np.isfinite(y):
                    continue
                flag = None if slot_active is None else slot_active[j]
                date = None if slot_date is None else slot_date[j]
                if _dropped(flag, "day", (uid, date)):
                    _note("pv", uid, date, y)
                    continue
                row = np.where(np.isnan(b["pv_c_ctx"][j]), 0.0, b["pv_c_ctx"][j])
                a = b["pv_a"][j]
                lag = b["pv_lag1"][j] if len(b["pv_lag1"]) > j else np.nan
                z = pv_z[j] if (pv_z is not None and j < len(pv_z)) else np.nan
                a = 0.0 if not np.isfinite(a) else a
                pv_rows.append((
                    u, t, y, z if (y == 1 and np.isfinite(z)) else np.nan,
                    row[0], row[1], row[2], a,
                    0.0 if not np.isfinite(lag) else lag, jw,
                    a * float(b["pv_i"][j]),  # interaction with a delivered suggestion
                ))
            for key, ykey, lagkey, active_key, date_key in (
                ("FW", "FW_daily", "day_FW_lag", "fw_active", "fw_date"),
                ("PJ", "PJ_daily", "day_PJ_lag", "day_active", "day_date"),
            ):
                active = b.get(active_key)
                dates = b.get(date_key)
                for d in range(len(b[ykey])):
                    y = b[ykey][d]
                    if not np.isfinite(y):
                        continue
                    flag = None if active is None else active[d]
                    date = None if dates is None else dates[d]
                    if _dropped(flag, "day", (uid, date)):
                        _note(key.lower(), uid, date, y)
                        continue
                    a0, a1 = b["day_A0"][d], b["day_A1"][d]
                    we = b["day_is_weekend"][d] if len(b["day_is_weekend"]) > d else np.nan
                    rb = b["day_burden"][d] if len(b["day_burden"]) > d else np.nan
                    lag = b[lagkey][d] if len(b[lagkey]) > d else np.nan
                    day_rows[key].append((
                        u, t, y,
                        0.0 if not np.isfinite(a0) else a0,
                        0.0 if not np.isfinite(a1) else a1,
                        0.0 if not np.isfinite(we) else we,
                        0.0 if not np.isfinite(rb) else rb,
                        0.0 if not np.isfinite(lag) else lag, jw,
                    ))
        last = blocks[-1]
        j_close = last.get("J_week", np.nan)
        if np.isfinite(j_close) and not _dropped(last.get("J_week_active", None), "week"):
            ju_rows.append((u, T, j_close, last.get("U1", np.nan), last.get("U2", np.nan)))
        elif np.isfinite(j_close):
            _note("j", uid, last.get("J_week_date"), j_close)

    def arr(rows, ncol):
        return np.asarray(rows, dtype=float).reshape(-1, ncol)

    return {
        "uids": uids, "T": T, "trans": trans,
        "ju": arr(ju_rows, 5), "pv": arr(pv_rows, 11),
        "fw": arr(day_rows["FW"], 9), "pj": arr(day_rows["PJ"], 9),
        "activity_filter": {
            "person_days_inactive": len(excl_days["inactive"]),
            "person_days_outside_window": len(excl_days["outside"]),
            "person_weeks_inactive": excl_weeks["inactive"],
            "person_weeks_outside_window": excl_weeks["outside"],
            "nonzero_excluded": {kind: len(rows) for kind, rows in nonzero.items()},
            "nonzero_excluded_rows": {kind: rows[:12] for kind, rows in nonzero.items()},
        },
    }


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def build_model(data):
    import pymc as pm
    import pytensor.tensor as pt

    U, T = len(data["uids"]), data["T"]
    hier = list(PARAM_NAMES)  # participant-specific coefficients
    K = len(hier)
    col = {nm: k for k, nm in enumerate(hier)}
    sign = np.array([float(DIRECTION.get(nm, 0)) for nm in hier])
    dir_idx = np.flatnonzero(sign != 0)
    free_idx = np.flatnonzero(sign == 0)
    free_sd = np.full(len(free_idx), POP_SD)

    # Outcome variances for the shared-noise priors.
    ju, pv = data["ju"], data["pv"]
    var_y = {
        "sigma_U1": np.nanvar(ju[(ju[:, 2] == 1), 3], ddof=1) if ju.shape[0] else 1.0,
        "sigma_U2": np.nanvar(ju[(ju[:, 2] == 1), 4], ddof=1) if ju.shape[0] else 1.0,
        "sigma_PV": np.nanvar(pv[:, 3], ddof=1) if pv.shape[0] else 1.0,
    }
    var_y = {k: float(v) if np.isfinite(v) and v > 0 else 1.0 for k, v in var_y.items()}

    coords = {"participant": data["uids"], "param": hier,
              "param_dir": [hier[k] for k in dir_idx], "param_free": [hier[k] for k in free_idx]}
    with pm.Model(coords=coords) as model:
        mu_free = pm.Normal("mu_free", 0.0, free_sd, dims="param_free")
        mu_dir_abs = pm.HalfNormal("mu_dir_abs", sigma=DIRECTIONAL_SCALE, dims="param_dir")
        mu = pt.zeros(K)
        mu = pt.set_subtensor(mu[free_idx], mu_free)
        mu = pt.set_subtensor(mu[dir_idx], sign[dir_idx] * mu_dir_abs)
        mu = pm.Deterministic("mu", mu, dims="param")
        re_idx = np.array([k for k, nm in enumerate(hier) if nm not in COMMON_COEFS])
        tau_free = pm.Gamma("tau_free", alpha=2.0, beta=1.0 / RE_SD_MODE, shape=len(re_idx))
        tau = pm.Deterministic("tau", pt.set_subtensor(pt.zeros(K)[re_idx], tau_free), dims="param")
        z = pm.Normal("z", 0.0, 1.0, dims=("participant", "param"))
        raw = mu[None, :] + tau[None, :] * z  # (U, K)
        Pmat = raw
        pm.Deterministic("a1_participant", Pmat[:, col["a1"]], dims="participant")

        noise = {}
        for nm in SHARED_NOISE:
            s2 = pm.InverseGamma(f"{nm}_sq", alpha=2.0, beta=0.5 * var_y[nm])
            noise[nm] = pm.Deterministic(nm, pt.sqrt(s2))

        def c(R, nm):
            return R[:, col[nm]]

        # Latent states, non-centered innovations, sigma_E fixed (scale).
        eta = pm.Normal("eta", 0.0, 1.0, shape=(U, T))
        trans = pt.as_tensor_variable(data["trans"])
        E_prev = pt.ones(U) * E1_FIXED
        E_cols = [E_prev]
        for t in range(1, T + 1):
            mean_t = (c(Pmat, "a0") + c(Pmat, "a1") * E_prev
                      + c(Pmat, "a2") * trans[:, t - 1, 0]
                      + c(Pmat, "a3") * trans[:, t - 1, 1]
                      + c(Pmat, "a4") * trans[:, t - 1, 2])
            E_prev = mean_t + SIGMA_E_FIXED * eta[:, t - 1]
            E_cols.append(E_prev)
        E = pm.Deterministic("E", pt.stack(E_cols, axis=1))  # (U, T+1)
        E_flat = E.reshape((-1,))

        def gather(rows):
            """Participant coefficient rows and latent E for each data row."""
            u_ = rows[:, 0].astype(int)
            t_ = rows[:, 1].astype(int)
            return Pmat[u_], E_flat[u_ * (T + 1) + t_]

        # Weekly J / U emissions.
        if ju.shape[0]:
            R, e = gather(ju)
            pm.Bernoulli("J_obs", logit_p=c(R, "b0") + c(R, "b1") * e, observed=ju[:, 2])
            for k, (i0, i1, sg) in ((3, ("c0", "c1", "sigma_U1")), (4, ("d0", "d1", "sigma_U2"))):
                m = (ju[:, 2] == 1) & np.isfinite(ju[:, k])
                if m.any():
                    idx = np.flatnonzero(m)
                    pm.Normal(f"U{k - 2}_obs", mu=c(R, i0)[idx] + c(R, i1)[idx] * e[idx],
                              sigma=noise[sg], observed=ju[m, k])

        # Page-view hurdle.
        if pv.shape[0]:
            R, e = gather(pv)
            we, dt, rb, a, lag, jw, inter = (pv[:, k] for k in range(4, 11))

            def linpred(pfx):
                return (c(R, f"{pfx}0") + c(R, f"{pfx}1") * e
                        + c(R, f"{pfx}2_is_weekend") * we + c(R, f"{pfx}2_dt") * dt
                        + c(R, f"{pfx}2_rb") * rb + c(R, f"{pfx}_ar1") * lag
                        + a * (c(R, f"{pfx}3") + c(R, f"{pfx}4") * e)
                        + jw * (c(R, f"{pfx}_q0") + c(R, f"{pfx}_qE") * e
                                + c(R, f"{pfx}_qrb") * rb))

            pm.Bernoulli("PV_occ_obs", logit_p=linpred("alpha") + c(R, "alpha_inter") * inter,
                         observed=pv[:, 2])
            m = np.isfinite(pv[:, 3])
            if m.any():
                idx = np.flatnonzero(m)
                pm.Normal("PV_int_obs", mu=linpred("gamma")[idx],
                          sigma=noise["sigma_PV"], observed=pv[m, 3])

        # Daily Fitbit wear and check-in.
        for key, pfx in (("fw", "beta"), ("pj", "theta")):
            d = data[key]
            if not d.shape[0]:
                continue
            R, e = gather(d)
            a0, a1, we, rb, lag, jw = (d[:, k] for k in range(3, 9))
            eta_d = (c(R, f"{pfx}0") + c(R, f"{pfx}1") * e
                     + c(R, f"{pfx}2_is_weekend") * we + c(R, f"{pfx}2_rb") * rb
                     + c(R, f"{pfx}_ar1") * lag
                     + a0 * (c(R, f"{pfx}3") + c(R, f"{pfx}4") * e)
                     + a1 * (c(R, f"{pfx}5") + c(R, f"{pfx}6") * e)
                     + jw * (c(R, f"{pfx}_q0") + c(R, f"{pfx}_qE") * e
                             + c(R, f"{pfx}_qrb") * rb))
            pm.Bernoulli(f"{key.upper()}_obs", logit_p=eta_d, observed=d[:, 2])
    return model



# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def _sigm(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def posterior_summaries(idata):
    """Per participant: posterior-mean parameters and posterior-mean E (t = 0..T)."""
    post = idata.posterior
    raw = post["mu"].values[:, :, None, :] + post["tau"].values[:, :, None, :] * post["z"].values
    means = raw.mean(axis=(0, 1))  # (U, K)
    names = list(post["param"].values)
    shared = {nm: float(post[nm].values.mean()) for nm in SHARED_NOISE}
    E_post = post["E"].values.mean(axis=(0, 1))  # (U, T+1)
    out = {}
    for u, uid in enumerate(post["participant"].values):
        par = {nm: float(means[u, k]) for k, nm in enumerate(names)}
        par["sigma_E"] = SIGMA_E_FIXED
        par.update(shared)
        out[uid] = {"par": par, "E": E_post[u]}
    return out


def emission_fits(par, E, blocks, digits=3):
    """Fitted values and residuals at the posterior means (simulator bootstrap pools)."""
    def r(x):
        if x is None or not np.isfinite(x):
            return None
        return float(np.round(float(x), digits))

    def fin(x, default=0.0):
        return float(x) if x is not None and np.isfinite(x) else default

    out = {k: [] for k in ("pJ", "rJ", "pU1", "rU1", "pU2", "rU2", "pPV", "rPV", "rPVocc",
                           "pFW", "rFW", "pPJ", "rPJ")}

    def ju(e, j, u1, u2):
        pj = float(_sigm(par["b0"] + par["b1"] * e))
        out["pJ"].append(r(pj))
        out["rJ"].append(r(j - pj) if np.isfinite(j) else None)
        j1 = np.isfinite(j) and int(round(j)) == 1
        for key, (i0, i1), u in (("U1", ("c0", "c1"), u1), ("U2", ("d0", "d1"), u2)):
            m = par[i0] + par[i1] * e
            out[f"p{key}"].append(r(m))
            out[f"r{key}"].append(r(u - m) if j1 and np.isfinite(u) else None)

    for t, b in enumerate(blocks):
        e = fin(E[t])
        jw = _block_jw(b)
        ju(e, b.get("J_lag", np.nan), b.get("U1_open", np.nan), b.get("U2_open", np.nan))
        for j in range(len(b["pv_y"])):
            y = b["pv_y"][j]
            we, dt, rb = (fin(v) for v in b["pv_c_ctx"][j])
            a = fin(b["pv_a"][j])
            inter = a * fin(b["pv_i"][j]) if len(b["pv_i"]) > j else 0.0
            lag = fin(b["pv_lag1"][j]) if len(b["pv_lag1"]) > j else 0.0

            def lp(p):
                return (par[f"{p}0"] + par[f"{p}1"] * e + par[f"{p}2_is_weekend"] * we
                        + par[f"{p}2_dt"] * dt + par[f"{p}2_rb"] * rb + par[f"{p}_ar1"] * lag
                        + a * (par[f"{p}3"] + par[f"{p}4"] * e)
                        + jw * (par[f"{p}_q0"] + par[f"{p}_qE"] * e + par[f"{p}_qrb"] * rb))

            p_occ = float(_sigm(lp("alpha") + par["alpha_inter"] * inter))
            out["pPV"].append(r(p_occ))
            out["rPVocc"].append(r(y - p_occ) if np.isfinite(y) else None)
            z = b["pv_z"][j] if j < len(b["pv_z"]) else np.nan
            occ = np.isfinite(y) and int(round(y)) == 1
            out["rPV"].append(r(z - lp("gamma")) if occ and np.isfinite(z) else None)
        for key, ykey, lagkey, p in (("FW", "FW_daily", "day_FW_lag", "beta"),
                                     ("PJ", "PJ_daily", "day_PJ_lag", "theta")):
            for d in range(len(b[ykey])):
                y = b[ykey][d]
                a0, a1 = fin(b["day_A0"][d]), fin(b["day_A1"][d])
                we = fin(b["day_is_weekend"][d]) if len(b["day_is_weekend"]) > d else 0.0
                rb = fin(b["day_burden"][d]) if len(b["day_burden"]) > d else 0.0
                lag = fin(b[lagkey][d]) if len(b[lagkey]) > d else 0.0
                eta = (par[f"{p}0"] + par[f"{p}1"] * e + par[f"{p}2_is_weekend"] * we
                       + par[f"{p}2_rb"] * rb + par[f"{p}_ar1"] * lag
                       + a0 * (par[f"{p}3"] + par[f"{p}4"] * e)
                       + a1 * (par[f"{p}5"] + par[f"{p}6"] * e)
                       + jw * (par[f"{p}_q0"] + par[f"{p}_qE"] * e + par[f"{p}_qrb"] * rb))
                ph = float(_sigm(eta))
                out[f"p{key}"].append(r(ph))
                out[f"r{key}"].append(r(y - ph) if np.isfinite(y) else None)
    # Closing Sunday of the last week: emission of E_{W+1}.
    last = blocks[-1]
    ju(fin(E[len(blocks)]), last.get("J_week", np.nan), last.get("U1", np.nan), last.get("U2", np.nan))
    return out


def participant_json(par, E, blocks, digits=3):
    """``params_env`` and ``pred`` dictionaries in the layout read downstream."""
    def r(x):
        return None if x is None or not np.isfinite(x) else float(np.round(float(x), digits))

    env = {
        "estimation_method": "hierarchical_bayes_nuts",
        "_query_Jw_effect": {
            "source": "J_w = week_present_lastweek (end of week w-1 / start of week w) "
                      "on PV/FW/PJ emissions",
            "apply": "add J_w * (q @ z) on the PV/FW/PJ linear predictor; I_w not in model",
        },
    }
    for key, items in EXPORT_LAYOUT:
        env[key] = [r(par[p]) for p, _ in items]
        env[f"{key}_names"] = [n for _, n in items]
    fits = emission_fits(par, E, blocks, digits)
    env.update({
        "resid_penalized_J_week": fits["rJ"], "resid_penalized_U1": fits["rU1"],
        "resid_penalized_U2": fits["rU2"], "resid_penalized_hourly_pageview": fits["rPV"],
        "resid_penalized_pageview_occurrence": fits["rPVocc"],
        "resid_penalized_nextday_wearing": fits["rFW"], "resid_penalized_daily_present": fits["rPJ"],
    })
    pred = {
        "pred_penalized_filtered_Ew": [r(x) for x in np.asarray(E[1:len(blocks) + 1], dtype=float)],
        "pred_penalized_J_week": fits["pJ"], "pred_penalized_U1": fits["pU1"],
        "pred_penalized_U2": fits["pU2"], "pred_penalized_hourly_pageview": fits["pPV"],
        "pred_penalized_nextday_wearing": fits["pFW"], "pred_penalized_daily_present": fits["pPJ"],
    }
    return env, pred


def attach_Ew_to_df_fit(df_fit, E_by_uid, *, week_col="week"):
    """``perceived_utility_lastweek`` = Ê_w and ``perceived_utility`` = Ê_{w+1} on week w."""
    out = df_fit.copy()
    out["perceived_utility"] = np.nan
    out["perceived_utility_lastweek"] = np.nan
    for uid, E in E_by_uid.items():
        E = np.asarray(E, dtype=float)
        T = E.size - 1
        user = out["ParticipantIdentifier"] == uid
        for w in range(1, T + 1):
            rows = user & (out[week_col] == w)
            if rows.any():
                out.loc[rows, "perceived_utility_lastweek"] = float(E[w - 1])
                out.loc[rows, "perceived_utility"] = float(E[w])
    return out


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--max-treedepth", type=int, default=12)
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--users", type=str, default="", help="comma list of participant ids (smoke tests)")
    ap.add_argument("--no-write", action="store_true", help="do not overwrite params/pred JSON or df_fit.csv")
    ap.add_argument(
        "--diag-dir",
        type=str,
        default="",
        help="where to save the posterior / diagnostics (default: the parameter directory)",
    )
    args = ap.parse_args(argv)
    print(f"params dir: {WORK_DIR}", flush=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    combined = os.environ.get("ADAPR_COMBINED_DIR", "").strip()
    if not combined:
        raise SystemExit("Set ADAPR_COMBINED_DIR to the folder with df_fit.csv.")
    combined_dir = Path(combined).expanduser().resolve()
    import arviz as az
    import pymc as pm

    df_all = mask_dropout_fit_outcomes(
        pd.read_csv(combined_dir / "df_fit.csv")
    )
    df_fit = df_all
    if args.users:
        keep = {int(x) for x in args.users.split(",") if x.strip()}
        df_fit = df_all[df_all["ParticipantIdentifier"].astype(int).isin(keep)]
    activity_path = resolve_activity_daily_path(combined_dir)
    activity_lookup = load_activity_lookup(activity_path)
    assert_activity_covers_fit(df_fit, activity_lookup)
    weekly_path = combined_dir / "df_weekly_filled.csv"
    if not weekly_path.is_file():
        raise SystemExit(
            f"Missing {weekly_path}. The J_w completion window is the max of "
            "observed_date minus the Sunday slot in this file."
        )
    completion_lag = weekly_checkin_completion_lag(
        pd.read_csv(weekly_path, usecols=["date", "observed_date", "week_present"])
    )
    completion_days = int(completion_lag.max())
    logger.info(
        "weekly check-in lag (observed_date - Sunday slot) max=%d counts=%s",
        completion_days,
        completion_lag.value_counts().sort_index().to_dict(),
    )
    user_blocks = prepare_all_user_blocks(
        df_fit, activity_lookup=activity_lookup, completion_days=completion_days
    )
    data = flatten_blocks(user_blocks)
    filt = data["activity_filter"]
    logger.info(
        "activity table %s; excluded person-days inactive=%d outside_window=%d; "
        "excluded person-weeks inactive=%d outside_window=%d; "
        "nonzero excluded PV/FW/PJ/J=%s",
        activity_path,
        filt["person_days_inactive"],
        filt["person_days_outside_window"],
        filt["person_weeks_inactive"],
        filt["person_weeks_outside_window"],
        filt["nonzero_excluded"],
    )
    if any(filt["nonzero_excluded"].values()):
        raise SystemExit(
            "Excluded emissions are not forced zeros: "
            f"{filt['nonzero_excluded']} examples {filt['nonzero_excluded_rows']}"
        )
    logger.info("participants=%d weeks=%d PV rows=%d FW rows=%d PJ rows=%d J/U rows=%d",
                len(data["uids"]), data["T"], len(data["pv"]), len(data["fw"]),
                len(data["pj"]), len(data["ju"]))

    model = build_model(data)
    with model:
        step = pm.NUTS(target_accept=args.target_accept, max_treedepth=args.max_treedepth)
        idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains, cores=args.cores,
                          step=step, random_seed=args.seed, progressbar=False,
                          idata_kwargs={"log_likelihood": False})

    n_div = int(idata.sample_stats["diverging"].values.sum())
    summ = az.summary(idata, var_names=["mu_free", "mu_dir_abs", "tau_free"]
                      + [f"{nm}_sq" for nm in SHARED_NOISE], round_to=4)
    rhat_max = float(summ["r_hat"].max())
    ess_min = float(summ["ess_bulk"].min())
    logger.info("divergences=%d  max R-hat=%.4f  min bulk ESS=%.0f", n_div, rhat_max, ess_min)
    if n_div or not (rhat_max < 1.01):
        logger.warning("convergence check failed: divergences=%d, max R-hat=%.4f", n_div, rhat_max)

    post = posterior_summaries(idata)
    # Simulator clipping range for E: pooled min / max over all posterior
    # draws of the latent states (the analogue, for a latent variable, of the
    # observed-range support limits used for the other continuous variables).
    _E_draws = idata.posterior["E"].values
    E_limits = [float(np.round(np.nanmin(_E_draws), 3)), float(np.round(np.nanmax(_E_draws), 3))]
    for uid, rec in post.items():
        if not args.no_write:
            env, pred = participant_json(rec["par"], rec["E"], user_blocks[uid])
            env["E_limits"] = E_limits
            uid_key = int(uid)
            with open(WORK_DIR / f"params_env_{uid_key}.json", "w", encoding="utf-8") as f:
                json.dump(env, f, allow_nan=False)
            with open(WORK_DIR / f"pred_{uid_key}.json", "w", encoding="utf-8") as f:
                json.dump(pred, f, allow_nan=False)

    suffix = "" if not args.users else "_subset"
    diag_dir = Path(args.diag_dir) if args.diag_dir else WORK_DIR
    diag_dir.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(diag_dir / f"pu_hier_posterior{suffix}.nc")
    with open(diag_dir / f"pu_hier_diagnostics{suffix}.json", "w", encoding="utf-8") as f:
        json.dump({
            "participants": [int(u) for u in data["uids"]],
            "chains": args.chains, "tune": args.tune, "draws": args.draws,
            "target_accept": args.target_accept, "max_treedepth": args.max_treedepth,
            "divergences": n_div,
            "max_rhat": rhat_max if np.isfinite(rhat_max) else None,
            "min_ess_bulk": ess_min if np.isfinite(ess_min) else None,
            "activity_filter": {
                "person_days_inactive": int(filt["person_days_inactive"]),
                "person_days_outside_window": int(filt["person_days_outside_window"]),
                "person_weeks_inactive": int(filt["person_weeks_inactive"]),
                "person_weeks_outside_window": int(filt["person_weeks_outside_window"]),
                "nonzero_excluded": {
                    str(k): int(v) for k, v in filt["nonzero_excluded"].items()
                },
                "zero_check": int(sum(int(v) for v in filt["nonzero_excluded"].values())),
            },
            "sigma_E_fixed": SIGMA_E_FIXED, "re_sd_mode": RE_SD_MODE, "common_coefficients": list(COMMON_COEFS), "E_limits": E_limits,
            "shared_noise_posterior_mean": {nm: float(idata.posterior[nm].values.mean())
                                            for nm in SHARED_NOISE},
        }, f, indent=2)

    if not args.no_write and not args.users:
        out = attach_Ew_to_df_fit(df_all, {uid: rec["E"] for uid, rec in post.items()})
        out.to_csv(combined_dir / "df_fit.csv", index=False)
        logger.info("wrote perceived_utility columns to %s", combined_dir / "df_fit.csv")
    return idata


if __name__ == "__main__":
    main()
