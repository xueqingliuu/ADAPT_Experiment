"""Agent-visible approximations of perceived utility (E_w) from weekly observables.

RCT ê (this module) uses only Monday–Saturday mediators, matching the
RCT action schedule:

* ``PV_sum`` = nansum of 12 Mon–Sat slots / 12
* ``FW_sum`` = nansum of 6 Mon–Sat ``nextday_wearing`` draws / 6
  (those draws are Tuesday–Sunday morning wear)
* ``PJ_sum`` = nansum of 6 Mon–Sat ``daily_present`` draws / 6
  (same-day EOD surveys)

Script 6 / script 4 / ``vani_env`` keep calendar-week /14 and /7 because
they are fit to MRT data that includes Sunday. The pooled coefficients
are still those script-6 numbers; only the online features drop Sunday.

Ê_{w+1} uses last week's Ê (AR), this week's Mon–Sat averages
(transition), and this Sunday's ``J`` / ``J(U1+U2)/14`` (update).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from algorithm_helpers import (
    FOURSC_SLOTS_PER_WEEK,
    N_RL_DAYS,
    N_RL_SLOTS,
    apply_pooled_coefs,
)
from vani_env import PARAMS_DIR

BASELINE_OFFSET = 1
DEFAULT_EW_HAT = 2.0
N_RL_PV_SLOTS = N_RL_DAYS * N_RL_SLOTS  # 12
_SUNDAY_DOW = 7


def load_pooled_coefs(work_dir=None):
    p = Path(work_dir or PARAMS_DIR) / "Ew_pooled_linear_coefs.json"
    with open(p, encoding="utf-8") as f:
        return {k: float(v) for k, v in json.load(f).items()}


def rct_pv_sum(slot_pv):
    """Mon–Sat pageview mean (12 slots / 12). Extra Sunday slots are ignored."""
    v = np.asarray(slot_pv, dtype=float).ravel()
    if v.size >= N_RL_PV_SLOTS:
        v = v[:N_RL_PV_SLOTS]
    if v.size == 0 or not np.any(np.isfinite(v)):
        return 0.0
    return float(np.nansum(v) / N_RL_PV_SLOTS)


def rct_daily_sum(daily):
    """Mon–Sat FW or PJ mean (6 days / 6). A trailing Sunday draw is ignored."""
    v = np.asarray(daily, dtype=float).ravel()
    if v.size >= N_RL_DAYS:
        v = v[:N_RL_DAYS]
    if v.size == 0:
        return 0.0
    return float(np.nansum(v) / N_RL_DAYS)


def weekly_Ew_predictor_table(
    df,
    *,
    date_col="Date",
    decision_col="DecisionTime",
    u1_impute_mean=0.0,
    u2_impute_mean=0.0,
):
    rows = []
    for (uid, wk), g in df.groupby(["ParticipantIdentifier", "week"], sort=True):
        g = g.sort_values([date_col, decision_col], na_position="last")
        sun = g.loc[g["dow"] == _SUNDAY_DOW]
        j_w = float(sun["week_present"].iloc[0]) if len(sun) else np.nan
        u1 = float(sun["Exp-tool-1"].iloc[0]) if len(sun) else np.nan
        u2 = float(sun["Exp-tool-2"].iloc[0]) if len(sun) else np.nan
        if np.isnan(u1):
            u1 = float(u1_impute_mean)
        if np.isnan(u2):
            u2 = float(u2_impute_mean)
        if np.isnan(j_w):
            j_w = 0.0
        j_w = float(j_w)
        half_j_close = j_w * (u1 + u2) / 14.0

        rl = g.loc[g["dow"] != _SUNDAY_DOW]
        pv_arr = rl["HourlyPageviewCount_norm"].to_numpy(dtype=float)
        pv_sum = rct_pv_sum(pv_arr)

        fw_daily, pj_daily = [], []
        for _, g_day in rl.groupby(date_col, sort=True):
            g_day = g_day.sort_values(decision_col, na_position="last")
            row0 = g_day.iloc[0]
            fw_daily.append(float(row0["nextday_wearing"]) if pd.notna(row0["nextday_wearing"]) else 0.0)
            pj_daily.append(float(row0["daily_present"]) if pd.notna(row0["daily_present"]) else 0.0)
        fw_sum = rct_daily_sum(fw_daily)
        pj_sum = rct_daily_sum(pj_daily)

        rows.append({
            "ParticipantIdentifier": int(uid) if isinstance(uid, (int, np.integer)) else uid,
            "week": int(wk),
            "J_close": j_w,
            "half_J_close": half_j_close,
            "PV_sum": pv_sum,
            "FW_sum": fw_sum,
            "PJ_sum": pj_sum,
        })
    return pd.DataFrame(rows)


def roll_ew_hat_series(tbl, coefs, e0=None):
    """Roll Ê through ``tbl`` (sorted by week). Returns the final Ê."""
    if tbl is None or tbl.empty:
        return float(coefs.get("E_lag0", DEFAULT_EW_HAT) if e0 is None else e0)
    e = float(coefs.get("E_lag0", DEFAULT_EW_HAT) if e0 is None else e0)
    for _, row in tbl.sort_values("week").iterrows():
        e = apply_pooled_coefs(
            coefs,
            e,
            row["PV_sum"],
            row["FW_sum"],
            row["PJ_sum"],
            row["J_close"],
            row["half_J_close"],
        )
    return float(e)


def initial_Ew_hat_for_user(user_id, df_fit=None, coefs=None):
    """Week-0 Ê_w: last pre-RL df_fit week, or ``DEFAULT_EW_HAT`` (2.0).

    ``df_fit is None`` (the experiment default) skips the table and returns
    2.0. Pass a panel to roll the pooled linear filter through that user's
    weeks, starting from the E_1 pin.
    """
    if coefs is None:
        coefs = load_pooled_coefs()
    if df_fit is None:
        return DEFAULT_EW_HAT

    sub = df_fit[df_fit["ParticipantIdentifier"] == user_id]
    if sub.empty:
        return DEFAULT_EW_HAT
    tbl = weekly_Ew_predictor_table(
        sub,
        u1_impute_mean=coefs.get("U1_impute_mean", 0.0),
        u2_impute_mean=coefs.get("U2_impute_mean", 0.0),
    )
    if tbl.empty:
        return DEFAULT_EW_HAT
    return roll_ew_hat_series(tbl, coefs)


def _finite_or(arr, idx, default):
    idx = int(idx)
    if idx < 0 or idx >= len(arr):
        return float(default)
    val = float(arr[idx])
    return val if np.isfinite(val) else float(default)


def compute_Ew_hat_from_week(
    sim_w,
    *,
    coefs,
    e_lag,
    wp_all,
    U1_all,
    U2_all,
    pageViewNext4HourAll,
    dw_wk,
    dp_wk,
    baseline_offset=BASELINE_OFFSET,
):
    """Approximate E_{w+1} from Ê_w, week ``sim_w`` Mon–Sat mediators, and this Sunday.

    ``sim_w`` is RL week ``k`` (0-based). At the end of week ``w``:

    * ``e_lag`` is last week's Ê (``E_known_all[sim_w]``), the AR term.
    * PV/FW/PJ are this week's Monday–Saturday series (12 slots / 6 days).
    * ``J_close`` / ``half_J_close`` are this Sunday's ``J_{w+1}`` and
      ``J_{w+1}(U1+U2)/14`` at ``wp_all[sim_w + baseline_offset]``.
    """
    e_lag = float(e_lag)
    if not np.isfinite(e_lag):
        e_lag = float(coefs.get("E_lag0", DEFAULT_EW_HAT))
    survey_idx = int(sim_w) + int(baseline_offset)
    J_close = _finite_or(wp_all, survey_idx, 0.0)
    u1 = _finite_or(U1_all, survey_idx, coefs.get("U1_impute_mean", 0.0))
    u2 = _finite_or(U2_all, survey_idx, coefs.get("U2_impute_mean", 0.0))
    half_j_close = J_close * (u1 + u2) / 14.0
    slot_start = int(sim_w) * FOURSC_SLOTS_PER_WEEK
    slot_stop = int(sim_w + 1) * FOURSC_SLOTS_PER_WEEK
    pv_sum = rct_pv_sum(pageViewNext4HourAll[slot_start:slot_stop])
    fw_sum = rct_daily_sum(dw_wk)
    pj_sum = rct_daily_sum(dp_wk)
    return apply_pooled_coefs(
        coefs, e_lag, pv_sum, fw_sum, pj_sum, J_close, half_j_close,
    )
