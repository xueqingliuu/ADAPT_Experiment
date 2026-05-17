from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(
    os.getenv("ADAPR_PROJECT_ROOT", str(Path(__file__).resolve().parent))
).expanduser().resolve()
WORK_DIR = PROJECT_ROOT / "env_para_vanilla"
EW_POOLED_COEF_JSON = WORK_DIR / "Ew_pooled_linear_coefs.json"

FEATURE_COLS = ("J_w", "half_J_tool8", "PV_sum", "FW_sum", "PJ_sum")


def load_pooled_coefs(work_dir: Path = WORK_DIR) -> dict[str, float]:
    p = (work_dir or WORK_DIR) / "Ew_pooled_linear_coefs.json"
    with open(p, encoding="utf-8") as f:
        return {k: float(v) for k, v in json.load(f).items()}


def apply_pooled_coefs(
    coefs: dict[str, float],
    J_w: float,
    half_J_tool8: float,
    PV_sum: float,
    FW_sum: float,
    PJ_sum: float,
) -> float:
    return float(
        coefs["J_w"] * float(J_w)
        + coefs["half_J_tool8"] * float(half_J_tool8)
        + coefs["PV_sum"] * float(PV_sum)
        + coefs["FW_sum"] * float(FW_sum)
        + coefs["PJ_sum"] * float(PJ_sum)
    )


def _first_nonmissing(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[0]) if len(s) else np.nan


def weekly_predictor_table(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    decision_col: str = "DecisionTime",
) -> pd.DataFrame:
    rows = []
    for (uid, wk), g in df.groupby(["ParticipantIdentifier", "week"], sort=True):
        g = g.sort_values([date_col, decision_col], na_position="last")
        j_w = _first_nonmissing(g["week_present"])
        u1 = _first_nonmissing(g["Exp-tool-1"])
        u2 = _first_nonmissing(g["Exp-tool-2"])
        if np.isnan(u1):
            u1 = 0.0
        if np.isnan(u2):
            u2 = 0.0
        if np.isnan(j_w):
            j_w = 0.0
        j_w = float(j_w)
        tool_combo = 0.5 * j_w * ((u1 + 1.0) + (u2 + 1.0)) / 8.0
        pv_arr = g["HourlyPageviewCount_norm"].to_numpy(dtype=float)
        pv_sum = float(np.nansum(pv_arr) / 14.0) if pv_arr.size else np.nan

        fw_daily, pj_daily = [], []
        for _, g_day in g.groupby(date_col, sort=True):
            g_day = g_day.sort_values(decision_col, na_position="last")
            row0 = g_day.iloc[0]
            vfw = row0["nextday_wearing"]
            vpj = row0["daily_present"]
            fw_daily.append(float(vfw) if pd.notna(vfw) else np.nan)
            pj_daily.append(float(vpj) if pd.notna(vpj) else np.nan)
        fw_arr = np.asarray(fw_daily, dtype=float)
        pj_arr = np.asarray(pj_daily, dtype=float)
        fw_sum = float(np.nansum(fw_arr) / 7.0) if fw_arr.size else np.nan
        pj_sum = float(np.nansum(pj_arr) / 7.0) if pj_arr.size else np.nan

        rows.append(
            {
                "ParticipantIdentifier": int(uid)
                if isinstance(uid, (int, np.integer))
                else uid,
                "week": int(wk),
                "J_w": j_w,
                "half_J_tool8": tool_combo,
                "PV_sum": pv_sum,
                "FW_sum": fw_sum,
                "PJ_sum": pj_sum,
            }
        )
    return pd.DataFrame(rows)


def initial_Ew_hat_for_user(
    user_id,
    df_fit: pd.DataFrame | None = None,
    coefs: dict[str, float] | None = None,
) -> float:
    if coefs is None:
        coefs = load_pooled_coefs()
    if df_fit is None:
        return 0.0

    sub = df_fit[df_fit["ParticipantIdentifier"] == user_id]
    if sub.empty:
        return 0.0
    tbl = weekly_predictor_table(sub).dropna(subset=list(FEATURE_COLS))
    if tbl.empty:
        return 0.0
    last = tbl.sort_values("week").iloc[-1]
    return apply_pooled_coefs(
        coefs,
        last["J_w"],
        last["half_J_tool8"],
        last["PV_sum"],
        last["FW_sum"],
        last["PJ_sum"],
    )
