"""Agent-visible context is built from observed quantities only.

Covers the prior-2h leak (the context used to fall back to the simulator's
true value on disengaged days), the agent-side burden / interaction / active
summaries, the daily-survey gate on the self-reported active day, and
finite STE buffers under never-send.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

import experiment
from experiment import OnlineEnv
from ste_vanilla import _ste_env_config, collect_mdp_episode
from vani_env import Env

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "env_para_vanilla"


def _env_with_gap(uid=13, seed=3):
    """Two weeks; days 2-4 of week 0 disengaged, everything else engaged."""
    _cfg, _env, oenv = experiment._make_online_env(
        uid, seed=seed, params_dir=PARAMS, dropout_hazard_day=0.0
    )
    # _activity_status: 0 = engaged, nonzero = disengaged.
    oenv._activity_status[:] = 0
    oenv._activity_status[2:5] = 1
    oenv._dropout_day_planned = -1
    oenv._reset_episode_state()
    return oenv


def _run_until(oenv, k_stop, d_stop):
    """Step every slot before (k_stop, d_stop); return without stepping it.

    Reads the decision context at every slot, as ``run_episode`` does.
    """
    for k in range(k_stop + 1):
        oenv.start_week(k, 1)
        for d in range(experiment.N_RL_DAYS):
            if (k, d) == (k_stop, d_stop):
                return
            d_global = oenv._day_idx(k, d)
            oenv._ensure_day_started(k, d, d_global)
            for t in range(oenv.K):
                step_idx = oenv._step_idx(k, d, t)
                oenv._generate_prior2hour_for_slot(k, d, t, d_global, step_idx)
                oenv.get_context(k, d, t)
                oenv.step_action(k, d, t, 1, 1)
        oenv._finalize_week(k)


def test_context_ignores_true_state_on_a_disengaged_day():
    oenv = _env_with_gap()
    _run_until(oenv, 0, 3)                     # day 3 is disengaged
    d_global = oenv._day_idx(0, 3)
    oenv._ensure_day_started(0, 3, d_global)
    step_idx = oenv._step_idx(0, 3, 0)
    oenv._generate_prior2hour_for_slot(0, 3, 0, d_global, step_idx)
    assert np.isnan(oenv.prior2HourStepCountAll[step_idx])

    a, b = copy.deepcopy(oenv), copy.deepcopy(oenv)
    # Perturb everything the simulator knows but the agent cannot see.
    b.s["prior2HourStepCount"] += 5.0
    b.prior2HourStepCountTruthAll[step_idx] += 5.0
    b.s["activeDaysLast7Days"] += 0.7
    b.s["activitySuggestionsSentLast7Days"] += 3.0
    b.s["activitySuggestionInteractLast7Days"] += 0.5
    b.s["yesterdayStepCount"] += 4.0
    b.s["activityStatusToday"] = 1.0 - b.s["activityStatusToday"]

    np.testing.assert_array_equal(
        a._rl_context_vector(0, 3, 0), b._rl_context_vector(0, 3, 0)
    )


def test_agent_active_fraction_skips_unobserved_reports():
    oenv = _env_with_gap()
    _run_until(oenv, 0, 5)
    hist = np.asarray(oenv._hist_active_obs)
    assert np.isnan(hist[2:5]).all()           # disengaged days are unobserved
    assert np.isfinite(oenv._agent_active_days7)
    # The simulator covariate also counts the unobserved days' true reports.
    assert len(oenv._hist_active_days) == len(hist)


def test_agent_burden_counts_sent_suggestions():
    oenv = _env_with_gap()
    _run_until(oenv, 0, 5)
    # Every slot sent 1, including the disengaged days.
    assert oenv._hist_daily_sent == [2.0] * 5
    # The simulator's burden counts only delivered suggestions.
    assert oenv._hist_daily_suggestions[2:5] == [0.0, 0.0, 0.0]


def test_zero_denominator_interaction_split():
    oenv = _env_with_gap()
    for _ in range(14):
        oenv._record_ws_interaction(0.0, 0.0)
        oenv._record_agent_ws_interaction(0.0, 0.0)
    # Simulator side: participant's fitting-panel mean.
    assert oenv.s["activitySuggestionInteractLast7Days"] == oenv._interact7_fill
    # Agent side: undefined, so the online fill (running mean) decides.
    assert np.isnan(oenv._agent_interact7)
    fill = oenv._context_fill
    fill.reset()
    fill.apply("interact", 0.4, 0)
    fill.apply("interact", 0.6, 1)
    assert np.isclose(fill.apply("interact", oenv._agent_interact7, 2), 0.5)


def test_sent_on_disengaged_day_counts_as_not_interacted():
    oenv = _env_with_gap()
    oenv._hist_agent_ws_sent, oenv._hist_agent_ws_inter = [], []
    for _ in range(14):
        oenv._record_agent_ws_interaction(1.0, 0.0)
    assert oenv._agent_interact7 == 0.0


def test_self_report_observed_only_with_a_completed_daily_survey():
    oenv = _env_with_gap()
    _run_until(oenv, 1, 0)
    days = np.arange(7)
    obs = oenv.activityStatusTodayAll[days]
    survey = oenv.dailySurveyCompleteAll[days]
    assert np.all(np.isnan(obs[survey == 0]))          # includes disengaged days
    assert np.all(np.isfinite(obs[survey == 1]))
    assert np.all(survey[2:5] == 0)                     # disengaged -> no survey
    np.testing.assert_array_equal(
        np.asarray(oenv._hist_active_obs[:7]), obs
    )


def test_every_context_column_is_finite_at_the_first_read():
    for uid in (13, 18):
        oenv = _env_with_gap(uid=uid)
        oenv.start_week(0, 1)
        d_global = oenv._day_idx(0, 0)
        oenv._ensure_day_started(0, 0, d_global)
        oenv._generate_prior2hour_for_slot(0, 0, 0, d_global, oenv._step_idx(0, 0, 0))
        assert np.all(np.isfinite(oenv._rl_context_vector(0, 0, 0)))


def test_ste_buffer_finite_under_never_send_with_activity_on():
    cfg = _ste_env_config(
        13, params_dir=PARAMS, nweek=3, activity="on", stayer_fraction=0.0,
    )
    env = Env(cfg, noise="ar1")
    oenv = OnlineEnv(env, nweek=3, seed=7, ste_mode=True)
    episode = collect_mdp_episode(
        oenv, 0.0, np.random.default_rng(7), n_transition_weeks=2,
    )
    for key in ("states", "actions", "rewards", "terminals"):
        assert np.all(np.isfinite(episode[key])), key
