"""Agent-visible approximations of perceived utility (E_w) from weekly observables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from algorithm_helpers import (
    FOURSC_SLOTS_PER_WEEK,
    apply_pooled_coefs,
    weekly_pv_sum_for_ew,
)
from vani_env import PARAMS_DIR

BASELINE_OFFSET = 1
DEFAULT_EW_HAT = 2.0


def load_pooled_coefs(work_dir=None):
    p = Path(work_dir or PARAMS_DIR) / "Ew_pooled_linear_coefs.json"
    with open(p, encoding="utf-8") as f:
        return {k: float(v) for k, v in json.load(f).items()}


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
        sun = g.loc[g["dow"] == 7]
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
        half_J_tool8 = j_w * (u1 + u2) / 14.0
        pv_sum = weekly_pv_sum_for_ew(g["HourlyPageviewCount_norm"].to_numpy(dtype=float))

        fw_daily, pj_daily = [], []
        for _, g_day in g.groupby(date_col, sort=True):
            g_day = g_day.sort_values(decision_col, na_position="last")
            row0 = g_day.iloc[0]
            fw_daily.append(float(row0["nextday_wearing"]) if pd.notna(row0["nextday_wearing"]) else 0.0)
            pj_daily.append(float(row0["daily_present"]) if pd.notna(row0["daily_present"]) else 0.0)
        fw_sum = float(np.sum(np.asarray(fw_daily, dtype=float)) / 7.0) if fw_daily else 0.0
        pj_sum = float(np.sum(np.asarray(pj_daily, dtype=float)) / 7.0) if pj_daily else 0.0

        rows.append({
            "ParticipantIdentifier": int(uid) if isinstance(uid, (int, np.integer)) else uid,
            "week": int(wk),
            "J_w": j_w,
            "half_J_tool8": half_J_tool8,
            "PV_sum": pv_sum,
            "FW_sum": fw_sum,
            "PJ_sum": pj_sum,
        })
    return pd.DataFrame(rows)


def initial_Ew_hat_for_user(user_id, df_fit=None, coefs=None):
    """Bootstrap ``E_known_all[0]`` from the participant's last pre-RL df_fit week."""
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
    last = tbl.sort_values("week").iloc[-1]
    return apply_pooled_coefs(
        coefs,
        last["J_w"],
        last["half_J_tool8"],
        last["PV_sum"],
        last["FW_sum"],
        last["PJ_sum"],
    )


def compute_Ew_hat_from_week(
    sim_w,
    *,
    coefs,
    wp_all,
    U1_all,
    U2_all,
    pageViewNext4HourAll,
    dw_wk,
    dp_wk,
    baseline_offset=BASELINE_OFFSET,
):
    """Apply pooled linear coefficients to observable aggregates of week ``sim_w``."""
    J_w = float(wp_all[sim_w]) if not np.isnan(wp_all[sim_w]) else 0.0
    weekly_idx = int(sim_w) + int(baseline_offset)
    u1 = (
        float(U1_all[weekly_idx])
        if not np.isnan(U1_all[weekly_idx])
        else float(coefs.get("U1_impute_mean", 0.0))
    )
    u2 = (
        float(U2_all[weekly_idx])
        if not np.isnan(U2_all[weekly_idx])
        else float(coefs.get("U2_impute_mean", 0.0))
    )
    half_J_tool8 = J_w * (u1 + u2) / 14.0
    slot_start = int(sim_w) * FOURSC_SLOTS_PER_WEEK
    slot_stop = int(sim_w + 1) * FOURSC_SLOTS_PER_WEEK
    pv_sum = weekly_pv_sum_for_ew(pageViewNext4HourAll[slot_start:slot_stop])
    fw_sum = float(np.nansum(dw_wk) / 7.0)
    pj_sum = float(np.nansum(dp_wk) / 7.0)
    return apply_pooled_coefs(
        coefs, J_w, half_J_tool8, pv_sum, fw_sum, pj_sum,
    )
