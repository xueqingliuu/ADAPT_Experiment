from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dropout_utils import calibrate_constant_daily_hazard
from tests.param_fingerprint import require_matching_fingerprint
from vani_env import EnvConfig, PARAMS_DIR
import experiment


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "env_para_vanilla"
GOLDEN = ROOT / "tests" / "fixtures" / "dropout_hazard_zero_random_send.npz"


class _FreezeAgent:
    """Records agent calls; skips PF/RLSVI internals."""

    needs_belief = False

    def __init__(self):
        self.begin_weeks: list[int] = []
        self.rlsvi_weeks: list[int] = []
        self.acts: list[tuple[int, int, int]] = []
        self.rng = np.random.default_rng(0)

    def reset(self, dataset, week0_actions=None):
        self.dataset = dataset
        if week0_actions is not None:
            dataset.bootstrap_week0(week0_actions, self.rng)

    def begin_week(self, k, packet):
        self.begin_weeks.append(int(k))
        return 1

    def update_rlsvi(self, k):
        self.rlsvi_weeks.append(int(k))

    def act(self, k, d, t, state):
        self.acts.append((int(k), int(d), int(t)))
        return 1, 1.0

    def results(self, dataset):
        return {
            "I": dataset.I_hist,
            "A": dataset.A_hist,
            "b_hat": dataset.b_hat_hist,
            "b_tilde": dataset.b_tilde_hist,
            "pi_A": dataset.pi_A_hist,
            "y_hat": None,
            "v_hat": None,
            "pf": {},
            "betas": np.empty((dataset.W, 0, 0), dtype=float),
        }


def test_calibration_formula_matches_hand_count():
    # 4 events at the observed days and 51 completed non-events:
    # (8+57+61+62) + 51*71 = 3809. Incomplete windows are ignored.
    # The participant extract is not in git.
    dropout_days = [9, 58, 62, 63] + [np.nan] * 51
    table = pd.DataFrame(
        {
            "window_complete": [True] * 55 + [False] * 9,
            "dropout_day": dropout_days + [3] * 9,
        }
    )
    complete = table.loc[table["window_complete"].astype(bool)]
    dropout_day = pd.to_numeric(complete["dropout_day"], errors="coerce")
    event = dropout_day.notna() & dropout_day.le(71)
    exposure = np.where(
        event,
        dropout_day.to_numpy(dtype=float) - 1.0,
        71.0,
    )
    assert int(event.sum()) == 4
    assert int(np.sum(exposure)) == 3809
    got = calibrate_constant_daily_hazard(table)
    assert got["dropout_events"] == 4
    assert got["dropout_person_days"] == 3809
    assert got["dropout_risk_window_days"] == 71
    np.testing.assert_allclose(got["dropout_hazard_day"], 4 / 3809)


def test_activity_on_requires_activity_process(tmp_path):
    std = PARAMS / "std_params.json"
    params = PARAMS / "params_env_13.json"
    dest = tmp_path / "params"
    dest.mkdir()
    dest.joinpath("std_params.json").write_bytes(std.read_bytes())
    dest.joinpath("params_env_13.json").write_bytes(params.read_bytes())
    with pytest.raises(FileNotFoundError, match="generator_calibration.json"):
        EnvConfig(13, params_dir=dest, nweek=2, activity="on")
    dest.joinpath("generator_calibration.json").write_text(
        json.dumps({"dropout_hazard_day": 0.01}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="activity_process"):
        EnvConfig(13, params_dir=dest, nweek=2, activity="on")
    cfg = EnvConfig(13, params_dir=dest, nweek=2, activity="off")
    assert cfg.activity == "off"
    assert cfg.activity_process is None
    assert cfg.dropout_mechanism == "none"


def test_activity_process_and_constant_hazard_do_not_stack(tmp_path):
    dest = tmp_path / "params"
    dest.mkdir()
    for name in (
        "std_params.json",
        "params_env_13.json",
        "Ew_pooled_linear_coefs.json",
    ):
        (dest / name).write_bytes((PARAMS / name).read_bytes())
    df_fit = PARAMS / "df_fit_11week.csv"
    if df_fit.is_file():
        (dest / df_fit.name).write_bytes(df_fit.read_bytes())
    calibration = json.loads((PARAMS / "generator_calibration.json").read_text())
    calibration["dropout_hazard_day"] = 1.0
    (dest / "generator_calibration.json").write_text(json.dumps(calibration))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg, env, oenv = experiment._make_online_env(13, seed=1, params_dir=dest)
    assert cfg.dropout_mechanism == "activity"
    assert cfg.activity_process is not None
    assert cfg.dropout_hazard_day == 0.0
    assert not any(
        "neither activity_process nor dropout_hazard_day" in str(w.message)
        for w in caught
    )

    def _forbid_constant_hazard(state):
        raise AssertionError("constant hazard was drawn on top of the activity path")

    env.dropout_hazard = _forbid_constant_hazard
    for d_global in range(oenv.D):
        oenv._maybe_dropout(d_global)
    assert oenv._activity_status is not None
    assert oenv.dropout_day == oenv._dropout_day_planned


def test_hazard_zero_matches_golden_random_send():
    gold = np.load(GOLDEN)
    require_matching_fingerprint(gold, PARAMS, uids=[int(u) for u in gold["uids"]])
    seed = int(gold["seed"])
    for uid in gold["uids"]:
        uid = int(uid)
        _res, oenv = experiment.run_random_send(
            uid, seed=seed, params_dir=PARAMS, dropout_hazard_day=0.0
        )
        prefix = f"uid{uid}_"
        assert int(oenv.dropout_day) == int(gold[prefix + "dropout_day"])
        assert np.all(np.asarray(oenv._activity_status)[: oenv.D] == 0)
        assert oenv.n_gaps == 0
        for name in (
            "CAE_all",
            "pu_all",
            "action_all",
            "stepCountNext4HourAll",
            "pageViewNext4HourAll",
            "morningFitbitWearAll",
            "dailySurveyCompleteAll",
        ):
            np.testing.assert_array_equal(
                np.asarray(getattr(oenv, name), dtype=float),
                gold[prefix + name],
            )


def test_lambda_one_pi_one_drops_on_day_2_and_zeros_presence():
    seed = 11
    uid = 13
    _base, base_env = experiment.run_random_send(
        uid, seed=seed, params_dir=PARAMS, dropout_hazard_day=0.0
    )
    _drop, drop_env = experiment.run_random_send(
        uid, seed=seed, params_dir=PARAMS, dropout_hazard_day=1.0
    )
    assert drop_env.dropout_day == 2
    assert drop_env.n_gaps == 1
    assert int(np.asarray(drop_env._activity_status)[0]) == 0
    assert np.all(np.asarray(drop_env._activity_status)[1:drop_env.D] == 2)
    # Day 1 is active, so its page views and steps match the all-active episode.
    # Its stored wear is tomorrow morning and is forced to 0.
    np.testing.assert_array_equal(
        drop_env.pageViewNext4HourTruthAll[:2],
        base_env.pageViewNext4HourTruthAll[:2],
    )
    np.testing.assert_array_equal(
        drop_env.stepCountNext4HourTruthAll[:2],
        base_env.stepCountNext4HourTruthAll[:2],
    )
    assert drop_env.morningFitbitWearTruthAll[0] == 0.0
    assert drop_env.dailySurveyCompleteTruthAll[0] == base_env.dailySurveyCompleteTruthAll[0]
    assert np.isnan(drop_env.stepCountNext4HourAll[2:]).all()
    assert np.isnan(drop_env.pageViewNext4HourAll[2:]).all()
    assert np.all(drop_env.pageViewNext4HourTruthAll[2:] == drop_env._pv_absent)
    assert np.all(drop_env.dailySurveyCompleteTruthAll[1:] == 0.0)
    assert np.isnan(drop_env.dailySurveyCompleteAll[1:]).all()
    assert np.isnan(drop_env.morningFitbitWearAll[1:]).all()
    assert np.all(drop_env.morningFitbitWearTruthAll[1:] == 0.0)
    assert np.isnan(drop_env.dailyAnticipatedAffectObsAll[1:]).all()
    assert np.all(np.isfinite(drop_env.dailyAnticipatedAffectAll))
    assert np.isnan(drop_env.U1_all[1:]).all()
    assert np.isnan(drop_env.U2_all[1:]).all()
    assert np.all(np.isfinite(drop_env.U1_truth_all[1:]))
    assert np.isnan(drop_env.CAE_short_all[1:]).all()
    assert np.all(drop_env.wp_all[1:] == 0.0)
    assert np.all(np.isfinite(drop_env.CAE_all[1:]))
    assert np.all(np.isfinite(drop_env.stepCountNext4HourTruthAll))


def test_activity_path_matches_the_dropout_rng_stream():
    from activity_process import prepare_activity, simulate_activity_paths

    cfg, _env, oenv = experiment._make_online_env(13, seed=7, params_dir=PARAMS)
    assert cfg.activity_process_mode == "pooled"
    rng = np.random.default_rng(
        experiment.OnlineEnv._dropout_seed_sequence(7, 13)
    )
    coef, fit, u, gamma = prepare_activity(
        cfg.activity_process, cfg.activity_process_mode, 13, rng
    )
    sim = simulate_activity_paths(
        1, oenv.D, coef, fit, oenv.nweek, rng,
        u=None if u is None else np.asarray([u]),
        gamma=gamma,
    )
    assert oenv._dropout_day_planned == int(sim["dropout_day"][0])
    assert oenv.n_gaps == int(sim["n_gaps"][0])
    assert oenv.inactive_days == int(sim["inactive_days"][0])
    np.testing.assert_array_equal(oenv._activity_status, sim["status"][0])


def test_same_seed_and_uid_share_the_activity_path_across_policies():
    for mode in ("frailty", "pooled"):
        _never, never_env = experiment.run_never_send(
            13, seed=1, params_dir=PARAMS, activity_process_mode=mode
        )
        _always, always_env = experiment.run_always_send(
            13, seed=1, params_dir=PARAMS, activity_process_mode=mode
        )
        assert not np.array_equal(never_env.action_all, always_env.action_all)
        np.testing.assert_array_equal(never_env._activity_status, always_env._activity_status)
        assert never_env._dropout_day_planned == always_env._dropout_day_planned
        assert never_env.n_gaps == always_env.n_gaps


def _within_mc(got: dict, expected: dict) -> None:
    n = int(expected["n_paths"])
    for key in (
        "mean_gaps", "p0", "p1", "p2", "p3plus",
        "inactive_fraction", "terminal14", "permanent",
    ):
        diff = abs(float(got[key]) - float(expected[key]))
        level = float(expected[key])
        if key == "mean_gaps":
            tol = 0.05
        else:
            se = (max(level, 0.0) * (1.0 - min(level, 1.0)) / n) ** 0.5
            tol = max(3.0 * se, 0.01)
        assert diff <= tol, f"{key}: {got[key]} vs {expected[key]} (tol {tol})"
    for key in ("ended_p50", "ended_p75", "ended_p90", "km_p90"):
        if expected[key] is None or got[key] is None:
            assert expected[key] is None and got[key] is None
        else:
            assert abs(float(got[key]) - float(expected[key])) <= 2.0


def _recomputed_table(mode: str) -> tuple[dict, dict]:
    from activity_process import simulate_cohort_paths, summarize_paths

    record = json.loads((PARAMS / "generator_calibration.json").read_text())["activity_process"]
    table = record["validation_2000"][mode]
    uids = table and record["validation_2000"]["uids"]
    got84 = summarize_paths(
        simulate_cohort_paths(
            record, mode, int(table["n_paths"]), int(table["n_days_84"]), 12,
            uids, int(table["base_seed_84"]),
        )
    )
    got252 = summarize_paths(
        simulate_cohort_paths(
            record, mode, int(table["n_paths"]), int(table["n_days_252"]), 36,
            uids, int(table["base_seed_252"]),
        )
    )
    return got84, got252


def test_frailty_mode_reproduces_its_2000_path_table():
    record = json.loads((PARAMS / "generator_calibration.json").read_text())["activity_process"]
    expected = record["validation_2000"]["frailty"]
    got84, got252 = _recomputed_table("frailty")
    _within_mc(got84, expected)
    se = (expected["permanent_252"] * (1.0 - expected["permanent_252"]) / expected["n_paths"]) ** 0.5
    assert abs(got252["permanent"] - expected["permanent_252"]) <= max(3.0 * se, 0.01)


def test_pooled_mode_reproduces_its_2000_path_table():
    record = json.loads((PARAMS / "generator_calibration.json").read_text())["activity_process"]
    expected = record["validation_2000"]["pooled"]
    got84, got252 = _recomputed_table("pooled")
    _within_mc(got84, expected)
    se = (expected["permanent_252"] * (1.0 - expected["permanent_252"]) / expected["n_paths"]) ** 0.5
    assert abs(got252["permanent"] - expected["permanent_252"]) <= max(3.0 * se, 0.01)


def test_stayer_fraction_zero_matches_the_pooled_simulator_and_draws_nothing():
    from activity_process import (
        activity_draw,
        dropout_seed_sequence,
        simulate_activity_paths,
    )

    record = json.loads((PARAMS / "generator_calibration.json").read_text())["activity_process"]
    _cfg, _env, pooled = experiment._make_online_env(13, seed=7, params_dir=PARAMS)
    _cfg0, _env0, explicit = experiment._make_online_env(
        13, seed=7, params_dir=PARAMS, activity_stayer_fraction=0.0
    )
    assert pooled.env.cfg.activity_stayer_fraction == 0.0
    np.testing.assert_array_equal(pooled._activity_status, explicit._activity_status)
    direct = np.random.default_rng(dropout_seed_sequence(7, 13))
    wrapped = np.random.default_rng(dropout_seed_sequence(7, 13))
    coef, fit, _u, gamma = activity_draw(record, "pooled", 13, direct)[:4]
    sim = simulate_activity_paths(1, pooled.D, coef, fit, pooled.nweek, direct, gamma=gamma)
    coef_w, fit_w, _uw, gamma_w, is_stayer = activity_draw(
        record, "pooled", 13, wrapped, stayer_fraction=0.0
    )
    assert is_stayer is False
    sim_w = simulate_activity_paths(1, pooled.D, coef_w, fit_w, pooled.nweek, wrapped, gamma=gamma_w)
    np.testing.assert_array_equal(sim["status"], sim_w["status"])
    np.testing.assert_array_equal(pooled._activity_status, sim["status"][0])
    assert direct.random() == wrapped.random()


def test_stayer_path_is_active_every_day():
    _cfg, _env, oenv = experiment._make_online_env(
        13, seed=1, params_dir=PARAMS, activity_stayer_fraction=1.0
    )
    assert np.all(oenv._activity_status == 0)
    assert oenv.n_gaps == 0
    assert oenv._dropout_day_planned == -1
    assert oenv.inactive_days == 0


def test_same_seed_and_uid_share_the_stayer_draw_across_policies():
    fraction = 0.4
    _never, never_env = experiment.run_never_send(
        13, seed=1, params_dir=PARAMS, activity_stayer_fraction=fraction
    )
    _always, always_env = experiment.run_always_send(
        13, seed=1, params_dir=PARAMS, activity_stayer_fraction=fraction
    )
    assert not np.array_equal(never_env.action_all, always_env.action_all)
    np.testing.assert_array_equal(never_env._activity_status, always_env._activity_status)
    assert never_env.n_gaps == always_env.n_gaps
    from activity_process import dropout_seed_sequence

    rng = np.random.default_rng(dropout_seed_sequence(1, 13))
    if float(rng.random()) < fraction:
        assert np.all(never_env._activity_status == 0)
        assert never_env.n_gaps == 0


def test_calibrated_stayer_variants_match_the_stored_table_and_inactive_share():
    """Inactive share matches the data. No-gap share matches the stored 10,000-path table.

    The mixture fit does not pin the simulated no-gap share to the observed
    0.617 once the stayer fraction is 0.4 or higher. The stored table is the
    record of that. p_stay 0 still matches the observed no-gap share.
    """
    from activity_process import simulate_cohort_paths, summarize_paths

    record = json.loads((PARAMS / "generator_calibration.json").read_text())["activity_process"]
    block = record["stayer_variants"]
    uids = record["validation_2000"]["uids"]
    n = 2000
    observed_p0 = float(block["observed_no_gap_share"])
    observed_inactive = float(block["observed_inactive_share"])
    for item in block["variants"]:
        got = summarize_paths(
            simulate_cohort_paths(
                record, "pooled", n, 84, 12, uids, 20261032,
                stayer_fraction=float(item["p_stay"]), stayer_recalibrate=True,
            )
        )
        stored = item["validation_10000"]
        se_p0 = (float(stored["p0"]) * (1.0 - float(stored["p0"])) / n) ** 0.5
        assert abs(float(got["p0"]) - float(stored["p0"])) <= max(3.0 * se_p0, 0.01)
        se_in = (observed_inactive * (1.0 - observed_inactive) / n) ** 0.5
        assert abs(float(got["inactive_fraction"]) - observed_inactive) <= max(3.0 * se_in, 0.01)
        if float(item["p_stay"]) == 0.0:
            se_obs = (observed_p0 * (1.0 - observed_p0) / n) ** 0.5
            assert abs(float(got["p0"]) - observed_p0) <= max(3.0 * se_obs, 0.01)


def test_inactive_day_zeros_presence_and_checkin_window():
    _cfg, _env, oenv = experiment._make_online_env(
        13, seed=3, params_dir=PARAMS, dropout_hazard_day=0.0
    )
    # Saturday inactive; Sunday–Tuesday inactive so that check-in is forced off.
    oenv._activity_status[:] = 0
    oenv._activity_status[5:9] = 1
    oenv._dropout_day_planned = -1
    oenv._reset_episode_state()
    oenv.start_week(0, 1)
    for d in range(experiment.N_RL_DAYS):
        for t in range(oenv.K):
            oenv.step_action(0, d, t, 1, 1)
    oenv._finalize_week(0)

    absent = oenv._pv_absent
    sat = slice(5 * oenv.K, 6 * oenv.K)
    assert np.all(oenv.pageViewNext4HourTruthAll[sat] == absent)
    assert np.all(oenv.pageViewNext4HourAll[sat] == absent)
    assert np.all(oenv.ws_interaction_all[sat] == 0.0)
    assert np.all(np.isnan(oenv.stepCountNext4HourAll[sat]))
    assert np.all(np.isfinite(oenv.stepCountNext4HourTruthAll[sat]))
    assert oenv.dailySurveyCompleteTruthAll[5] == 0.0
    assert oenv.dailySurveyCompleteAll[5] == 0.0
    assert np.isnan(oenv.dailyAnticipatedAffectObsAll[5])
    assert np.isfinite(oenv.dailyAnticipatedAffectAll[5])
    # Friday's row holds Saturday morning's wear.
    assert oenv.morningFitbitWearTruthAll[4] == 0.0
    assert oenv.morningFitbitWearAll[4] == 0.0
    assert oenv.wp_all[1] == 0.0
    assert oenv.wp_truth_all[1] == 0.0
    assert np.isnan(oenv.U1_all[1])
    assert np.isnan(oenv.U2_all[1])
    assert np.isfinite(oenv.U1_truth_all[1])
    assert np.isfinite(oenv.U2_truth_all[1])


def test_suggestion_on_inactive_day_acts_as_no_suggestion():
    def run(saturday_action):
        _cfg, _env, oenv = experiment._make_online_env(
            13, seed=3, params_dir=PARAMS, dropout_hazard_day=0.0
        )
        oenv._activity_status[:] = 0
        oenv._activity_status[5] = 1
        oenv._dropout_day_planned = -1
        oenv._reset_episode_state()
        for k in range(2):
            oenv.start_week(k, 1)
            for d in range(experiment.N_RL_DAYS):
                for t in range(oenv.K):
                    a = saturday_action if (k == 0 and d == 5) else 1
                    oenv.step_action(k, d, t, a, 1)
            oenv._finalize_week(k)
        return oenv

    sent, skipped = run(1), run(0)
    sat = slice(5 * sent.K, 6 * sent.K)
    assert np.all(sent.action_all[sat] == 0.0)
    for name in (
        "stepCountNext4HourTruthAll",
        "dailyAnticipatedAffectAll",
        "activitySuggestionsSentLast7DaysAll",
        "CAE_all",
        "pu_all",
    ):
        np.testing.assert_array_equal(getattr(sent, name), getattr(skipped, name))


def test_agent_freezes_after_dropout_week_and_omits_td():
    cfg, env, oenv = experiment._make_online_env(
        13, seed=5, params_dir=PARAMS, dropout_hazard_day=1.0
    )
    agent = _FreezeAgent()
    dataset = experiment.EpisodeDataset(cfg.nweek)
    week0, i_hist = experiment.shared_episode_exogenous(5, cfg.nweek)
    result = oenv.run_episode(agent, dataset, week0_actions=week0, I_hist=i_hist)
    assert oenv.dropout_day == 2
    assert agent.begin_weeks == [0]
    assert agent.rlsvi_weeks == []
    assert agent.acts == [(0, 0, 0), (0, 0, 1)]
    assert np.all(result["A"][0, 0] == 1)
    assert np.all(result["A"][0, 1:] == 0)
    assert np.all(result["A"][1:] == 0)
    assert np.all(result["I"][1:] == 0)
    assert 0 not in dataset.med_full_hist
    packet = oenv.get_week_packet(1)
    assert packet is not None
    assert packet.J_w == 0
    assert np.isfinite(packet.Y_prev)
