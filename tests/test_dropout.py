import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from dropout_utils import (
    build_participant_dropout_table,
    longest_true_run,
    mask_post_dropout_outcomes,
    max_raw_observation_timestamp,
    simulated_daily_inactivity,
    summarize_activity,
)
_dropout_check_path = (
    Path(__file__).resolve().parents[1] / "archive" / "analyses" / "dropout_check.py"
)
_dropout_check = importlib.util.spec_from_file_location(
    "dropout_check", _dropout_check_path
)
assert _dropout_check is not None and _dropout_check.loader is not None
_dropout_check_mod = importlib.util.module_from_spec(_dropout_check)
_dropout_check.loader.exec_module(_dropout_check_mod)
_local_dates = _dropout_check_mod._local_dates
from vani_env import mask_dropout_fit_outcomes


def test_terminal_gap_detection_and_longest_internal_gap():
    active = np.ones(84, dtype=bool)
    active[10:30] = False
    active[70:] = False
    got = summarize_activity(active, gap_days=14)
    assert got["last_active_day"] == 70
    assert got["longest_inactive_gap"] == 20
    assert got["terminal_inactive_gap"] == 14
    assert got["dropout_day"] == 71


def test_terminal_gap_below_threshold_is_not_dropout():
    active = np.ones(84, dtype=bool)
    active[-13:] = False
    got = summarize_activity(active, gap_days=14)
    assert np.isnan(got["dropout_day"])
    assert got["terminal_inactive_gap"] == 13


def test_all_inactive_drops_on_day_one():
    got = summarize_activity(np.zeros(84, dtype=bool))
    assert got["last_active_day"] == 0
    assert got["dropout_day"] == 1
    assert longest_true_run(np.ones(84, dtype=bool)) == 84


def test_incomplete_window_is_censored_not_dropout():
    anchors = pd.DataFrame(
        {
            "ParticipantIdentifier": ["recent"],
            "date_min": [pd.Timestamp("2026-09-15")],
        }
    )
    empty = pd.DataFrame(columns=["ParticipantIdentifier", "Date", "Value"])
    table, daily = build_participant_dropout_table(
        anchors,
        empty.drop(columns="Value"),
        empty,
        empty,
        observation_end="2026-10-14",
    )
    row = table.iloc[0]
    assert row["dropout_status"] == "still_enrolled"
    assert not bool(row["window_complete"])
    assert row["followup_days"] == 30
    assert np.isnan(row["dropout_day"])
    assert daily["observed"].sum() == 30
    assert daily.loc[daily["observed"].eq(0), "active_any"].isna().all()


def test_pageview_utc_timestamp_uses_participant_local_calendar_date():
    timestamp = pd.Series(["2026-10-15T01:30:00Z"])
    participant = pd.Series(["east"])
    got = _local_dates(
        timestamp,
        participant,
        {"east": "America/New_York"},
    )
    assert got.iloc[0] == pd.Timestamp("2026-10-14")


def test_extract_timestamp_is_maximum_across_raw_sources(tmp_path):
    pd.DataFrame({"when": ["2026-10-13T23:00:00Z"]}).to_csv(
        tmp_path / "a.csv", index=False
    )
    pd.DataFrame(
        {"stamp": ["2026-10-14T02:00:00Z", "not-a-date"]}
    ).to_csv(tmp_path / "b.csv", index=False)
    got = max_raw_observation_timestamp(
        tmp_path,
        specs=(("a.csv", "when"), ("b.csv", "stamp")),
        chunksize=1,
    )
    assert got == pd.Timestamp("2026-10-14T02:00:00Z")


def test_row_and_weekly_outcomes_are_nan_not_zero():
    dat = pd.DataFrame(
        {
            "ParticipantIdentifier": [1] * 6,
            "week": [1, 1, 1, 1, 2, 2],
            "study_day": [1, 2, 3, 4, 8, 9],
            "dropout_day": [4] * 6,
            "4hour_step_norm": np.arange(6.0),
            "daily_present": np.zeros(6),
            "week_present": np.zeros(6),
            "CAE_avg_norm": np.arange(6.0),
        }
    )
    masked = mask_dropout_fit_outcomes(dat)
    assert masked.loc[:2, "4hour_step_norm"].notna().all()
    assert masked.loc[3:, "4hour_step_norm"].isna().all()
    # Dropout occurs during week 1, so its replicated weekly emission is all NaN.
    assert masked["week_present"].isna().all()
    assert masked["CAE_avg_norm"].isna().all()
    assert dat["daily_present"].eq(0).all()


def test_generic_mask_requires_stable_study_day():
    dat = pd.DataFrame({"dropout_day": [3.0], "outcome": [0.0]})
    try:
        mask_post_dropout_outcomes(dat, ["outcome"])
    except KeyError as exc:
        assert "study_day" in str(exc)
    else:
        raise AssertionError("missing study_day should fail")


def test_simulator_inactivity_uses_standardized_pageview_zero():
    wear = [0, 0, 1]
    survey = [0, 0, 0]
    pv_zero = -2.5
    pageview = [pv_zero, pv_zero, pv_zero, 0.7, pv_zero, pv_zero]
    got = simulated_daily_inactivity(
        wear, survey, pageview, pageview_zero=pv_zero
    )
    np.testing.assert_array_equal(got, [True, False, False])


def test_pu_transition_denominator_excludes_post_dropout_days():
    path = Path(__file__).resolve().parents[1] / "06_fit_pu.py"
    spec = importlib.util.spec_from_file_location("pu_bayes_dropout_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    dates = np.repeat(pd.date_range("2025-01-06", periods=7), 2)
    decision = np.tile([0, 1], 7)
    dat = pd.DataFrame(
        {
            "ParticipantIdentifier": 1,
            "week": 1,
            "Date": dates,
            "DecisionTime": decision,
            "study_day": np.repeat(np.arange(1, 8), 2),
            "dropout_day": 5.0,
            "week_present": 1.0,
            "week_present_lastweek": 1.0,
            "Exp-tool-1_norm": 0.0,
            "Exp-tool-2_norm": 0.0,
            "HourlyPageviewCount_norm": 1.0,
            "HourlyPageviewCount": 1.0,
            "hourly_pageview_count_lag1": 0.0,
            "nextday_wearing": np.repeat([1, 1, 0, 1, 0, 0, 0], 2),
            "morning_wearing": 0.0,
            "daily_present": np.repeat([1, 0, 1, 1, 0, 0, 0], 2),
            "daily_present_yesterday": 0.0,
            "WalkingSuggestion": 0.0,
            "Interacted_walk": 0.0,
            "is_weekend": 0.0,
            "recent_burden_norm": 0.0,
        }
    )
    dat.loc[0, ["HourlyPageviewCount", "HourlyPageviewCount_norm"]] = [0.0, -1.0]
    dat.loc[1, ["HourlyPageviewCount", "HourlyPageviewCount_norm"]] = np.nan
    blocks = module.build_user_blocks(
        mask_dropout_fit_outcomes(dat), full_weeks=[1]
    )
    block = blocks[0]
    # Eight enrolled slots: six +1, one observed zero at -1, one missing
    # imputed to z0=-1. Post-dropout six slots do not enter the denominator.
    np.testing.assert_allclose(block["PV_sum"], 0.5)
    np.testing.assert_allclose(block["FW_sum"], 0.75)
    np.testing.assert_allclose(block["PJ_sum"], 0.75)


def test_activity_filter_keeps_weekly_means_and_drops_inactive_emissions():
    path = Path(__file__).resolve().parents[1] / "06_fit_pu.py"
    spec = importlib.util.spec_from_file_location("pu_bayes_activity_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    dates = np.repeat(pd.date_range("2025-01-06", periods=7), 2)
    dat = pd.DataFrame(
        {
            "ParticipantIdentifier": 1,
            "week": 1,
            "Date": dates,
            "DecisionTime": np.tile([0, 1], 7),
            "study_day": np.repeat(np.arange(1, 8), 2),
            "dropout_day": np.nan,
            "week_present": 1.0,
            "week_present_lastweek": 1.0,
            "Exp-tool-1_norm": 0.2,
            "Exp-tool-2_norm": 0.3,
            "HourlyPageviewCount_norm": 1.0,
            "HourlyPageviewCount": 1.0,
            "hourly_pageview_count_lag1": 0.0,
            "nextday_wearing": 1.0,
            "morning_wearing": 0.0,
            "daily_present": 1.0,
            "daily_present_yesterday": 0.0,
            "WalkingSuggestion": 0.0,
            "Interacted_walk": 0.0,
            "is_weekend": 0.0,
            "recent_burden_norm": 0.0,
        }
    )
    plain = module.build_user_blocks(dat, full_weeks=[1])
    lookup = {
        ("1", day.strftime("%Y-%m-%d")): 1.0
        for day in pd.date_range("2025-01-05", "2025-01-14")
    }
    lookup[("1", "2025-01-05")] = 0.0  # opening Sunday inactive; Mon and Tue are active
    lookup[("1", "2025-01-08")] = 0.0  # Wednesday inactive
    del lookup[("1", "2025-01-09")]  # Thursday outside the activity window
    filtered = module.build_user_blocks(
        dat, full_weeks=[1], activity_lookup=lookup, completion_days=2
    )
    for key in ("PV_sum", "FW_sum", "PJ_sum", "PV_sum_trans", "FW_sum_trans", "PJ_sum_trans"):
        np.testing.assert_allclose(filtered[0][key], plain[0][key])

    data = module.flatten_blocks({1: filtered})
    # PV/PJ drop Wednesday (inactive) and Thursday (outside): 14-4 and 7-2.
    # FW on day d uses active(d+1): Tuesday points at Wednesday, Wednesday at Thursday.
    assert len(data["pv"]) == 10
    assert len(data["fw"]) == 5
    assert len(data["pj"]) == 5
    # Opening window Sun-Tue is not entirely inactive, so both check-ins stay.
    assert len(data["ju"]) == 2
    filt = data["activity_filter"]
    assert filt["person_days_inactive"] == 1
    assert filt["person_days_outside_window"] == 1
    assert filt["person_weeks_inactive"] == 0
    assert filt["person_weeks_outside_window"] == 0

    closed = dict(lookup)
    closed[("1", "2025-01-06")] = 0.0
    closed[("1", "2025-01-07")] = 0.0
    window_off = module.build_user_blocks(
        dat, full_weeks=[1], activity_lookup=closed, completion_days=2
    )
    closed_data = module.flatten_blocks({1: window_off})
    assert len(closed_data["ju"]) == 1
    assert closed_data["activity_filter"]["person_weeks_inactive"] == 1
    plain_data = module.flatten_blocks({1: plain})
    assert len(plain_data["pv"]) == 14
    assert len(plain_data["ju"]) == 2
    assert plain_data["activity_filter"]["person_days_inactive"] == 0


def _load_fit_pu():
    path = Path(__file__).resolve().parents[1] / "06_fit_pu.py"
    spec = importlib.util.spec_from_file_location("pu_bayes_activity_mask", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fit_rows_active_mask_matches_lookup_definition():
    pu = _load_fit_pu()
    df = pd.DataFrame(
        {
            "ParticipantIdentifier": [1, 1, 2],
            "Date": ["2025-01-06", "2025-01-07", "2025-01-06"],
        }
    )
    lookup = {("1", "2025-01-06"): 1.0, ("1", "2025-01-07"): 0.0}
    mask = pu.fit_rows_active_mask(df, lookup)
    assert mask.tolist() == [True, False, False]
    assert pu.fit_rows_active_mask(df, None).all()


def test_ws_interaction_mask_drops_inactive_forced_zeros():
    """07 keeps C^AS only on active_any=1; inactive delivered slots are 0."""
    pu = _load_fit_pu()
    df = pd.DataFrame(
        {
            "ParticipantIdentifier": [13, 13, 13, 13],
            "Date": ["2025-01-06", "2025-01-06", "2025-01-07", "2025-01-07"],
            "Interacted_walk": [1.0, 0.0, 0.0, np.nan],
            "WalkingSuggestion": [1.0, 1.0, 1.0, 1.0],
        }
    )
    lookup = {("13", "2025-01-06"): 1.0, ("13", "2025-01-07"): 0.0}
    day_ok = pu.fit_rows_active_mask(df, lookup)
    y = df["Interacted_walk"].to_numpy(dtype=float)
    sent = df["WalkingSuggestion"].to_numpy(dtype=float)
    raw = np.isfinite(y) & (sent == 1.0)
    filt = raw & day_ok
    assert raw.tolist() == [True, True, True, False]
    assert filt.tolist() == [True, True, False, False]
    assert int((raw & ~filt).sum()) == 1
    assert float(y[raw & ~filt][0]) == 0.0
