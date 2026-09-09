"""Temporal and cross-variable residual-dependence diagnostics.

All calculations are participant-specific and calendar aligned.  Diagnostic
outputs contain anonymous panel numbers only; private source identifiers remain
in memory solely to locate the fitted model files.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .io import SourceData, load_source, saved_vector, weekly_unique


WEEKLY_MODELS = ("CAE", "short_CAE", "helpfulness", "pleasantness", "weekly_checkin")
DAILY_MODELS = ("anticipated_affect", "fitbit_wear", "daily_checkin", "active_status")
SLOT_MODELS = (
    "next_four_hour_steps",
    "prior_two_hour_steps",
    "page_view_occurrence",
    "positive_page_view_intensity",
    "suggestion_interaction",
)


def _correlation(x: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    if n < 3:
        return n, np.nan
    xx = x[valid]
    yy = y[valid]
    if np.ptp(xx) <= 1e-12 or np.ptp(yy) <= 1e-12:
        return n, np.nan
    xx = xx - xx.mean()
    yy = yy - yy.mean()
    denom = np.sqrt((xx @ xx) * (yy @ yy))
    if denom <= 0:
        return n, np.nan
    return n, float(np.clip((xx @ yy) / denom, -1.0, 1.0))


def _participant_streams(source: SourceData, source_id, d: pd.DataFrame):
    params = source.params(source_id)
    pred = source.predictions(source_id)

    weekly = {
        "CAE": saved_vector(params, "resid_CAE", 11),
        "short_CAE": saved_vector(params, "resid_CAE_short_avg", 11),
        "helpfulness": saved_vector(params, "resid_penalized_U1", 13)[2:13],
        "pleasantness": saved_vector(params, "resid_penalized_U2", 13)[2:13],
    }
    y_j = weekly_unique(d, "week_present")
    p_j = saved_vector(pred, "pred_penalized_J_week", 13)[2:13]
    weekly["weekly_checkin"] = np.round(y_j - p_j, 3)

    am = d["DecisionTime"].to_numpy(int) == 0
    y_fw_temporal = d.loc[am, "nextday_wearing"].to_numpy(float)
    p_fw_temporal = saved_vector(pred, "pred_penalized_nextday_wearing", 84)[7:84]
    y_pj = d.loc[am, "daily_present"].to_numpy(float)
    p_pj = saved_vector(pred, "pred_penalized_daily_present", 84)[7:84]
    y_ad = d.loc[am, "active_status"].to_numpy(float)
    p_ad = saved_vector(pred, "pred_active_status", 154)[::2]
    daily_temporal = {
        "anticipated_affect": saved_vector(params, "resid_antic", 77),
        "fitbit_wear": np.round(y_fw_temporal - p_fw_temporal, 3),
        "daily_checkin": np.round(y_pj - p_pj, 3),
        "active_status": np.round(y_ad - p_ad, 3),
    }

    # For same-day joint diagnostics, Fitbit wear is aligned to the actual
    # morning on which it was observed, not to the preceding context day.
    y_fw_joint = d.loc[am, "morning_wearing"].to_numpy(float)
    p_fw_joint = saved_vector(pred, "pred_penalized_nextday_wearing", 84)[6:83]
    daily_joint = dict(daily_temporal)
    daily_joint["fitbit_wear"] = np.round(y_fw_joint - p_fw_joint, 3)

    raw_pv = d["HourlyPageviewCount"].to_numpy(float)
    p_o = saved_vector(pred, "pred_penalized_hourly_pageview", 168)[14:]
    interaction = d["Interacted_walk"].to_numpy(float)
    p_as = saved_vector(pred, "pred_ws_interaction", 154)
    delivered = d["WalkingSuggestion"].to_numpy(float) == 1
    as_resid = np.where(delivered, np.round(interaction - p_as, 3), np.nan)
    slot = {
        "next_four_hour_steps": saved_vector(params, "resid_fourSC", 154),
        "prior_two_hour_steps": saved_vector(params, "resid_prior2hour_step_count", 154),
        "page_view_occurrence": np.round(np.where(np.isfinite(raw_pv), (raw_pv > 0).astype(float), np.nan) - p_o, 3),
        "positive_page_view_intensity": saved_vector(params, "resid_penalized_hourly_pageview", 168)[14:],
        "suggestion_interaction": as_resid,
    }
    return weekly, daily_temporal, daily_joint, slot


def _weekly_temporal(participant: int, streams: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for model in WEEKLY_MODELS:
        x = streams[model]
        n, rho = _correlation(x[:-1], x[1:])
        rows.append(
            {"participant": participant, "model": model, "lag_weeks": 1, "n_pairs": n, "correlation": rho}
        )
    return rows


def _daily_temporal(participant: int, streams: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for model in DAILY_MODELS:
        x = streams[model]
        for lag in (1, 7):
            n, rho = _correlation(x[:-lag], x[lag:])
            rows.append(
                {"participant": participant, "model": model, "lag_days": lag, "n_pairs": n, "correlation": rho}
            )
    return rows


def _ampm_temporal(participant: int, streams: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for model in SLOT_MODELS:
        x = np.asarray(streams[model], float).reshape(77, 2)
        n, rho = _correlation(x[:, 0], x[:, 1])
        rows.append(
            {
                "participant": participant,
                "model": model,
                "transition": "AM_to_same_day_PM",
                "n_pairs": n,
                "correlation": rho,
            }
        )
        n, rho = _correlation(x[:-1, 1], x[1:, 0])
        rows.append(
            {
                "participant": participant,
                "model": model,
                "transition": "PM_to_next_day_AM",
                "n_pairs": n,
                "correlation": rho,
            }
        )
    return rows


def _same_time_pairs(participant: int, streams: dict[str, np.ndarray], models, time_label: str) -> list[dict]:
    rows = []
    for left, right in combinations(models, 2):
        n, rho = _correlation(streams[left], streams[right])
        rows.append(
            {
                "participant": participant,
                "time_alignment": time_label,
                "model_1": left,
                "model_2": right,
                "n_pairs": n,
                "correlation": rho,
            }
        )
    return rows


def _slot_joint(participant: int, streams: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    split = {model: np.asarray(streams[model], float).reshape(77, 2) for model in SLOT_MODELS}
    for slot_index, label in ((0, "AM"), (1, "PM")):
        at_slot = {model: split[model][:, slot_index] for model in SLOT_MODELS}
        rows.extend(_same_time_pairs(participant, at_slot, SLOT_MODELS, label))
    n, rho = _correlation(
        split["next_four_hour_steps"][:, 0], split["prior_two_hour_steps"][:, 1]
    )
    rows.append(
        {
            "participant": participant,
            "time_alignment": "AM_next_four_hour_to_same_day_PM_prior_two_hour",
            "model_1": "next_four_hour_steps",
            "model_2": "prior_two_hour_steps",
            "n_pairs": n,
            "correlation": rho,
        }
    )
    return rows


def run(source_root: str | Path, output_root: str | Path) -> dict[str, Path]:
    """Run residual-dependence diagnostics and write anonymous CSV tables."""
    source = load_source(source_root)
    output = Path(output_root).expanduser().resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    weekly_temporal: list[dict] = []
    daily_temporal: list[dict] = []
    ampm_temporal: list[dict] = []
    weekly_joint: list[dict] = []
    daily_joint: list[dict] = []
    slot_joint: list[dict] = []

    for participant, source_id, d in source.iter_participants():
        weekly, daily_t, daily_j, slot = _participant_streams(source, source_id, d)
        weekly_temporal.extend(_weekly_temporal(participant, weekly))
        daily_temporal.extend(_daily_temporal(participant, daily_t))
        ampm_temporal.extend(_ampm_temporal(participant, slot))
        weekly_joint.extend(_same_time_pairs(participant, weekly, WEEKLY_MODELS, "same_assessment_week"))
        daily_joint.extend(_same_time_pairs(participant, daily_j, DAILY_MODELS, "same_calendar_day"))
        slot_joint.extend(_slot_joint(participant, slot))

    outputs = {
        "weekly_temporal": ("weekly_residual_correlation.csv", pd.DataFrame(weekly_temporal)),
        "daily_temporal": ("daily_residual_correlation.csv", pd.DataFrame(daily_temporal)),
        "decision_time_temporal": ("decision_time_residual_correlation.csv", pd.DataFrame(ampm_temporal)),
        "weekly_cross_variable": ("weekly_cross_variable_residual_correlation.csv", pd.DataFrame(weekly_joint)),
        "daily_cross_variable": ("daily_cross_variable_residual_correlation.csv", pd.DataFrame(daily_joint)),
        "decision_time_cross_variable": ("decision_time_cross_variable_residual_correlation.csv", pd.DataFrame(slot_joint)),
    }
    created: dict[str, Path] = {}
    for key, (filename, frame) in outputs.items():
        path = tables / filename
        frame.to_csv(path, index=False)
        created[key] = path
    return created
