"""STE buffer stays finite when the engagement process forces a dropout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from algorithm_helpers import N_RL_DAYS, N_RL_SLOTS, TERMINAL_D, TERMINAL_T
from experiment import OnlineEnv
from ste_vanilla import (
    STE_OBS_DIM,
    STE_STATE_NAMES,
    _ste_env_config,
    assert_ste_finite,
    collect_mdp_episode,
)
from vani_env import Env

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "env_para_vanilla"


def _forced_dropout_episode(n_transition_weeks: int = 2):
    """Activity on, but lambda = pi = 1 so the path drops out on day 2."""
    cfg = _ste_env_config(
        13,
        params_dir=PARAMS,
        nweek=n_transition_weeks + 1,
        activity="on",
        stayer_fraction=0.5,
    )
    cfg.activity_lambda = 1.0
    cfg.activity_pi = 1.0
    env = Env(cfg, noise="ar1")
    oenv = OnlineEnv(env, nweek=n_transition_weeks + 1, seed=11, ste_mode=True)
    episode = collect_mdp_episode(
        oenv,
        0.5,
        np.random.default_rng(11),
        n_transition_weeks=n_transition_weeks,
    )
    return oenv, episode


def test_buffer_is_finite_with_activity_on_and_forced_dropout():
    n_weeks = 2
    oenv, episode = _forced_dropout_episode(n_weeks)
    assert oenv.dropout_day == 2
    assert np.isnan(oenv.logYesterdayStepCount).any()

    states = episode["states"]
    assert states.shape[1] == STE_OBS_DIM == len(STE_STATE_NAMES)
    # Post-dropout slots are in the buffer: day 2 is the second calendar day.
    assert states.shape[0] == n_weeks * N_RL_DAYS * N_RL_SLOTS + 1
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(episode["actions"]))
    assert np.all(np.isfinite(episode["rewards"]))
    assert np.all(np.isfinite(episode["terminals"]))

    slots_per_week = N_RL_DAYS * N_RL_SLOTS
    terminal = TERMINAL_D * N_RL_SLOTS + TERMINAL_T
    for k in range(n_weeks):
        got = float(episode["rewards"][k * slots_per_week + terminal, 0])
        assert got == float(oenv.CAE_all[k + 1])
        assert np.isfinite(got)

    # The chosen action is stored. It is not delivered after dropout.
    assert np.any(episode["actions"][2:] == 1)
    assert np.all(oenv.action_all[2 : n_weeks * 14] == 0.0)


def test_finite_check_names_the_column_and_row():
    bad = np.zeros((4, STE_OBS_DIM))
    bad[2, STE_STATE_NAMES.index("yesterday_steps")] = np.nan
    with pytest.raises(SystemExit, match=r"yesterday_steps: 1 non-finite rows, indices 2"):
        assert_ste_finite(bad, what="training states", columns=STE_STATE_NAMES)
