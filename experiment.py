# %%
import argparse
import os
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
from datetime import datetime
from pathlib import Path
from functools import partial


# parameters for EWM
EWM_GAMMA = 6 / 7
EWM_WINDOW = 7
EWM_MIN_VALUES = 4


def _setup_log(*args, **kwargs):
    if __name__ == "__main__" or os.getenv("EXPERIMENT_VERBOSE_IMPORT") == "1":
        print(*args, **kwargs)


def _ewm_prior_gamma_last(
    values, gamma=EWM_GAMMA, window=EWM_WINDOW, min_values=EWM_MIN_VALUES
):
    """EWM over prior ≤window values (excludes current).

    Non-finite entries are imputed to 0. Returns 0 if the window has fewer than
    ``min_values`` prior observations.
    """
    w = np.asarray(values[-window:], dtype=float)
    if w.size < min_values:
        return 0.0
    w = np.where(np.isfinite(w), w, 0.0)
    alpha = 1.0 - gamma
    return float(pd.Series(w, dtype=float).ewm(alpha=alpha, adjust=True).mean().iloc[-1])


# parameters for rolling mean
INTERACTION_ROLLING_WINDOW = 14
RPA_ROLLING_WINDOW = 7
BASELINE_OFFSET = 1


def _rolling_mean_last(values, window=RPA_ROLLING_WINDOW, min_values=EWM_MIN_VALUES):
    if len(values) == 0:
        return 0.0
    v = np.asarray(values[-window:], dtype=float)
    v = v[np.isfinite(v)]
    if v.size < min_values:
        return 0.0
    return float(np.mean(v))
# %%
from vani_env import Env, EnvConfig, make_initial_state, PARAMS_DIR, P_FOURSC, P_ANTIC
from agents import (
    MicroQueryAgent,
    MicroQueryAgent_rewardshaping,
    MicroQueryAgent_ModifiedTDLoss,
    MicroQueryAgent_rewardshaping_modifiedTD,
    NeverSendAgent,
    AlwaysSendAgent,
    RandomSendAgent,
)
from algorithm_helpers import (  # WeekPacket.k = RL week (0-based)
    ParticleFilterRuntime,
    WeekPacket,
    build_phi_action,
    build_phi_action_rewardshaping,
    build_phi_bottleneck,
    build_fourSC_features,
    build_antic_features,
    fourSC_cae_delta,
    antic_cae_delta,
    build_CAE_features,
    build_CAE_short_features,
    build_pf_data,
    build_rl_context_vector,
    make_state,
    RL_MY_SHAPE,
    RL_ME_SHAPE,
    N_RL_CONTEXT,
    FOURSC_SLOTS_PER_WEEK,
    N_RL_DAYS,
    N_RL_SLOTS,
    QUERY_D,
    QUERY_T,
    TERMINAL_D,
)
from agents.ew_hat import (
    compute_Ew_hat_from_week,
    initial_Ew_hat_for_user,
    load_pooled_coefs,
)

SUNDAY_D_W = TERMINAL_D + 1


# %%
# ──────────────────────────────────────────────────────────────────
# Episode dataset (RL history lives outside MicroQueryAgent)
# ──────────────────────────────────────────────────────────────────

class EpisodeDataset:
    """Per-episode walking/query history and state snapshots for RLSVI updates."""

    def __init__(self, W):
        self.W = W
        self.I_hist = np.zeros(W, dtype=int)
        self.A_hist = np.zeros((W, N_RL_DAYS, N_RL_SLOTS), dtype=int)
        self.pi_A_hist = np.full((W, N_RL_DAYS, N_RL_SLOTS), np.nan)
        self.b_hat_hist = np.full(W, np.nan)
        self.b_tilde_hist = np.full(W, np.nan)
        self.pf_result = {}
        self.state_hist = {}
        # Full realized per-week mediator matrices (M_Y, M_E), recorded once a
        # week is finalized. Reward shaping reads these so its feature map can
        # access post-action decision-time / daily mediators (the per-slot
        # ``state_hist`` snapshots are pre-action and zero those out).
        self.med_full_hist = {}

    def bootstrap_week0(self, A_grid, rng):
        self.A_hist[0] = np.asarray(A_grid, dtype=int)
        self.I_hist[0] = 1
        self.pi_A_hist[0] = 0.5

    def record_query(self, k, I_w, pi_I=np.nan):
        self.I_hist[k] = int(I_w)
        if np.isfinite(pi_I):
            self.pi_I_hist = getattr(self, "pi_I_hist", np.full(self.W, np.nan))
            self.pi_I_hist[k] = float(pi_I)

    def record_week_start(self, k, state):
        """Week-start bottleneck/query state ``(k, -1, -1)``."""
        self.state_hist[f"{k}:{QUERY_D}:{QUERY_T}"] = {
            "E_w": float(state["E_w"]),
            "M_Y": np.asarray(state["M_Y"], dtype=float),
            "M_E": np.asarray(state["M_E"], dtype=float),
            "C": np.asarray(state["C"], dtype=float).ravel(),
        }

    def record_walking(self, k, d, t, state, action, pi_A):
        self.A_hist[k, d, t] = int(action)
        self.pi_A_hist[k, d, t] = float(pi_A)
        self.state_hist[f"{k}:{d}:{t}"] = {
            "E_w": float(state["E_w"]),
            "M_Y": np.asarray(state["M_Y"], dtype=float),
            "M_E": np.asarray(state["M_E"], dtype=float),
            "C": np.asarray(state["C"], dtype=float).ravel(),
        }

    def get_state(self, k, d, t):
        key = f"{k}:{d}:{t}"
        if key not in self.state_hist:
            raise KeyError(f"missing state snapshot for RL week/day/slot ({k}, {d}, {t})")
        snap = self.state_hist[key]
        return {
            "E_w": snap["E_w"],
            "M_Y": snap["M_Y"],
            "M_E": snap["M_E"],
            "C": snap["C"],
        }

    def record_full_week_mediators(self, k, M_Y, M_E):
        """Store the full realized mediator matrices for completed RL week ``k``."""
        self.med_full_hist[k] = (
            np.asarray(M_Y, dtype=float),
            np.asarray(M_E, dtype=float),
        )

    def get_full_week_mediators(self, k):
        """Full realized ``(M_Y, M_E)`` for week ``k`` or ``None`` if unrecorded."""
        return self.med_full_hist.get(k)


# %%
# ──────────────────────────────────────────────────────────────────
# OnlineEnv: adapter that wraps Env for slot-by-slot interaction
# with the RL algorithms.
#
# Indexing convention
# -------------------
# Public RL indices:     k = 0..nweek-1, d = 0..5, t = 0..K-1.
# Internal sim indices:  sim_w = k, d_w = 0..6, t_sim = 0..K-1.
# Sunday has d_w = 6; it has no RL action, but its two slots are generated
# with Ah = 0 so the weekly CAE/perceived-utility models see a full 7-day week.
# Week-start/query state uses the explicit sentinel (d,t)=(-1,-1).
# Slot/day/weekly array offsets should go through OnlineEnv._step_idx,
# _day_idx, _decode_step_idx, and _weekly_idx.
#
# get_week_packet(k) reads finalized sim week k-1 and builds PF data for week k (k >= 1).
#
# State dict ``self.s`` (beyond ``Env.gen_*`` outputs)
# ----------------------------------------------------
# ``self.s`` is the latent simulator state used by vani_env.Env.  Algorithm
# inputs should come from logged arrays: observed arrays keep NaN missingness
# for likelihood/learning, while ``*AgentAll`` arrays contain agent-visible
# observed-or-imputed values for feature construction.
# ──────────────────────────────────────────────────────────────────

class OnlineEnv:
    def __init__(
        self, env, nweek=None, seed=None, start_dow=1,
        df_fit_11week_csv=None, df_fit_full=None,
    ):
        # ``start_dow``: civil weekday index in ``{1,…,7}`` with **1 = Monday** (``df_fit`` / study).
        # ``df_fit_full``: optional pre-loaded df_fit DataFrame (multi-week aggregates)
        # for week-0 E_w_hat bootstrap. If omitted, runtime falls back to 0.0.
        if seed is not None:
            rd.seed(seed)

        self.env = env
        self.K = env.K
        self.W_days = env.W
        if self.W_days != N_RL_DAYS + 1:
            raise ValueError(
                f"OnlineEnv expects {N_RL_DAYS} controlled days plus Sunday; "
                f"env.W={self.W_days}"
            )
        if FOURSC_SLOTS_PER_WEEK != self.W_days * self.K:
            raise ValueError(
                f"FOURSC_SLOTS_PER_WEEK={FOURSC_SLOTS_PER_WEEK} does not match "
                f"env.W * env.K={self.W_days * self.K}"
            )
        self.nweek = nweek if nweek is not None else env.cfg.nweek
        self.D = self.nweek * self.W_days
        self.T = self.D * self.K
        self.start_dow = start_dow

        self.stepCountNext4HourAll = np.zeros(self.T) # latent draw (used by gen_CAE)
        self.stepCountNext4HourObsAll = np.full(self.T, np.nan) # observed by the agent
        self.stepCountNext4HourAgentAll = np.zeros(self.T) # agent-visible (model-imputed if fitbit not wear)
        self.pageViewNext4HourAll = np.zeros(self.T)
        self.action_all = np.zeros(self.T)
        self.activitySuggestionsSentLast7DaysAll = np.full(self.T, np.nan)
        self.dailyAnticipatedAffectAll = np.zeros(self.D)            # latent draw (used by gen_CAE)
        self.dailyAnticipatedAffectObsAll = np.full(self.D, np.nan) # observed by the agent
        self.dailyAnticipatedAffectAgentAll = np.zeros(self.D)      # agent-visible (model-imputed if survey missed)
        self.morningFitbitWearAll = np.zeros(self.D)
        self.dailySurveyCompleteAll = np.zeros(self.D)
        # PF-side observed copy of antic: NaN on days where the daily survey was
        # not completed (``dailySurveyComplete == 0``). The PF handles NaN at the
        # mediator-likelihood level; ``get_pf_data`` also strips NaN rows from
        # the cumulative posterior-update design / response.
        self.CAE_all = np.full(self.nweek+1, np.nan)
        self.pu_all = np.full(self.nweek+1, np.nan)        # latent E_w (env truth)
        # ``J_w`` (week-survey present) is predetermined for week ``k`` at the
        # end of week ``k-1`` (in ``_finalize_week(k-1)`` via ``gen_week_present``
        # conditioned on the just-realized ``pu[k-1]``). Week 0 has no prior
        # week to condition on, so we bootstrap ``J_w[0] = 1`` (treat the first
        # study week as a guaranteed-present week).
        self.wp_all = np.full(self.nweek+1, np.nan)
        self.wp_all[0] = 1.0
        self.CAE_short_all = np.full(self.nweek+1, np.nan)
        # Per-week observed tool surveys on the *raw* integer 0..7 Likert
        # scale, drawn from gen_tool_U1/U2 (see ``vani_env`` docstring for
        # the norm <-> raw transform).
        self.U1_all = np.full(self.nweek+1, np.nan)
        self.U2_all = np.full(self.nweek+1, np.nan)

        # Per-day covariate logs for PF / RL (not read from self.s at use time).
        self.logDayOfWeekNorm = np.zeros(self.D)
        self.logSalienceMessageSentToday = np.zeros(self.D)
        self.logSalienceMessageSentYesterday = np.zeros(self.D)
        self.logYesterdayStepCount = np.zeros(self.D)
        self.logStepCountLast7DaysEma = np.zeros(self.T)
        self.logPageViewLast7DaysEma = np.zeros(self.D)
        self.logMorningFitbitWearLast7Days = np.zeros(self.D)
        self.logDailyAnticipatedAffectYesterday = np.zeros(self.D)
        self.logTodayStepCount = np.zeros(self.D)
        self.logActiveDaysLast7Days = np.zeros(self.D)
        self.logActivitySuggestionInteractLast7Days = np.zeros(self.T)

        self._foursc_wk = np.zeros(FOURSC_SLOTS_PER_WEEK)
        self._pw_wk = np.zeros(FOURSC_SLOTS_PER_WEEK)
        self._dw_wk = np.zeros(7)
        self._dp_wk = np.zeros(7)
        self._antic_wk = np.zeros(7)

        self._hist_daily_pv = []

        self.s = make_initial_state(
            df_fit_11week_csv, participant_id=self.env.cfg.userid
        )
        self._stepCountLast7DaysEma_initial = float(self.s.get("stepCountLast7DaysEma", 0.0))
        self._hist_foursc_by_slot: dict[int, list[float]] = {0: [], 1: []}
        # Slot-specific (AM/PM) same-slot EMA, mirroring ``_hist_foursc_by_slot``.
        self._stepCountLast7DaysEma_by_slot: dict[int, float] = {
            0: self._stepCountLast7DaysEma_initial,
            1: self._stepCountLast7DaysEma_initial,
        }
        self._prev_day_salience = float(self.s.get("salienceMessageSentYesterday", 0.0))
        self._initial_prior2hour = float(self.s.get("prior2HourStepCount", 0.0))
        self.s["prior2HourStepCountAgent"] = float(self.s.get("prior2HourStepCount", 0.0))
        self.s["dailyAnticipatedAffectYesterdayAgent"] = float(
            self.s.get("dailyAnticipatedAffectYesterday", 0.0)
        )

        self.prior2HourStepCountAll = np.full(self.T, np.nan)
        self.prior2HourStepCountObsAll = np.full(self.T, np.nan)
        self.prior2HourStepCountAgentAll = np.full(self.T, np.nan)
        self._hist_prior2hour_observed: list[float] = []
        # Agent-visible prior-2-hour values (observed or imputed), per decision slot.
        self._hist_prior2hour_by_slot: dict[int, list[float]] = {0: [], 1: []}
        self.ws_interaction_all = np.zeros(self.T)

        # Seed the rolling 7-day walking-suggestion interaction fraction.
        self._ws_interaction_initial = float(self.s.get("activitySuggestionInteractLast7Days", 0.0))
        self._hist_ws_interaction = (
            [self._ws_interaction_initial] * INTERACTION_ROLLING_WINDOW
        )

        self.recordedPhysicalActivityTodayAll = np.zeros(self.D)
        self.activityStatusTodayAll = np.zeros(self.D)
        self.activityCompletedLast7DaysAll = np.zeros(self.D)

        # Seed previous-7-day RPA from df_fit_11week baseline.
        # Repeating preserves the baseline average at simulation start.
        self._rpa7_initial = float(self.s.get("activityCompletedLast7Days", 0.0))
        self._hist_recordedPhysicalActivityToday = [self._rpa7_initial] * RPA_ROLLING_WINDOW

        # Seed previous-7-day morning-wearing fraction from df_fit_11week baseline.
        # Repeating 7 times preserves the baseline average at simulation start.
        self._wear7_initial = float(self.s.get("morningFitbitWearLast7Days", 0.0))
        self._hist_morning_wear = [self._wear7_initial] * 7

        self._active_days7_initial = float(self.s.get("activeDaysLast7Days", 0.0))
        self._hist_active_days = [self._active_days7_initial] * RPA_ROLLING_WINDOW

        # ── Baseline slot [0]: pre-study values ──────────────────────────────
        # ``caeAverageLastWeek`` and ``perceivedUtilityLastWeek`` are loaded
        # from df_fit_11week_csv by ``make_initial_state``; use them to seed
        # the baseline arrays.  ``pu_all[0]`` is pinned to 2.0 by design
        # (population-mean prior for perceived utility entering the study).
        self.CAE_all[0] = float(self.s["caeAverageLastWeek"])
        self._cae_baseline = float(self.CAE_all[0])   # AR-1 seed for per-particle PF
        # Latent running state keys used as AR-1 inputs in gen_CAE / gen_perceivedUtility.
        # Not in make_initial_state; seeded here from baseline values.
        self.s["caeAverage"]           = float(self.CAE_all[0])
        self.s["perceivedUtility"] = float(self.s["perceivedUtilityLastWeek"])
        self.pu_all[0] = 2.0
        self.s["perceivedUtilityLastWeek"] = 2.0   # keep env state consistent
        # CAE_short is not used in the baseline slot
        self.CAE_short_all[0] = 0
        self.U1_all[0] = float(self.s["expTool1"])
        self.U2_all[0] = float(self.s["expTool2"])

        # First RL weekday is d_w = 0 (Monday when ``start_dow`` == 1); lag = prior civil day.
        dayOfWeekNorm_n, dayOfWeekNorm_i = self._day_of_week_norm(0)
        self.s["dayOfWeekNorm"] = dayOfWeekNorm_n
        self.s["isWeekend"] = 1.0 if dayOfWeekNorm_i >= 6 else 0.0
        lag_n, _ = self._day_of_week_norm(-1)
        self.s["dayOfWeekNormLag1"] = lag_n
        self.s["salienceMessageSentToday"] = float(rd.binomial(1, 0.5))

        self._yesterday_morning_WS = 0.0
        self._yesterday_afternoon_WS = 0.0

        # s["dailyAnticipatedAffectYesterday"] is the carry-forward of the last
        # *observed* anticipated affect (dailySurveyComplete==1 days only).  It is
        # seeded here from the df_fit baseline row (dailyAnticipatedAffectYesterday_norm)
        # and updated in _end_day on survey-present days.  It is also used
        self._today_fourSC = np.zeros(self.K)
        self._today_pageview = np.zeros(self.K)
        self._today_action = np.zeros(self.K)

        self._week_finalized = np.zeros(self.nweek, dtype=bool)
        self._day_started = {}
        self._Iw_per_week = np.zeros(self.nweek, dtype=int)

        # Pooled linear coefficients for the agent-visible E_w_hat approximation
        # (fitted offline by ``6_est_Ew_weights.py``).
        try:
            self._ew_coefs = load_pooled_coefs()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Ew_pooled_linear_coefs.json missing; run 6_est_Ew_weights.py first."
            ) from exc

        # Approximated E_w available to the agent at the start of each week.
        # ``E_known_all[k]`` is computed from observable aggregates of the previous
        # simulated week (``_finalize_week``); ``E_known_all[0]`` is bootstrapped from
        # the participant's last pre-RL week in df_fit.csv via the same linear formula.
        self.E_known_all = np.full(self.nweek + 1, np.nan)
        self.E_known_all[0] = float(
            initial_Ew_hat_for_user(
                self.env.cfg.userid,
                df_fit=df_fit_full,
                coefs=self._ew_coefs,
            )
        )

        # ``activitySuggestionsSentLast7Days``: baseline from df_fit; then daily sum +
        # ``_ewm_prior_gamma_last`` (matches ``1_data_extraction.recent_burden``).
        self._activitySuggestionsSentLast7Days_initial = float(
            self.s["activitySuggestionsSentLast7Days"]
        )
        self._hist_daily_suggestions: list[float] = []
        self._active_agent = None
        self._active_pf_runtime = None
        self._active_packet = None
        self._active_rl_week = 0

    def _weekly_idx(self, sim_w: int) -> int:
        """Weekly arrays reserve index 0 for the pre-RL baseline."""
        return int(sim_w) + BASELINE_OFFSET

    def _day_idx(self, sim_w: int, d_w: int) -> int:
        """Flatten a 0-based simulated week/day into the daily arrays."""
        return int(sim_w) * self.W_days + int(d_w)

    def _step_idx(self, sim_w: int, d_w: int, t_sim: int) -> int:
        """Flatten a 0-based simulated week/day/slot into slot arrays."""
        return int(sim_w) * FOURSC_SLOTS_PER_WEEK + int(d_w) * self.K + int(t_sim)

    def _decode_step_idx(self, step_idx: int):
        """Inverse of ``_step_idx``; returns ``(sim_w, d_w, t_sim, d_global)``."""
        step_idx = int(step_idx)
        sim_w = step_idx // FOURSC_SLOTS_PER_WEEK
        slot_in_week = step_idx % FOURSC_SLOTS_PER_WEEK
        d_w = slot_in_week // self.K
        t_sim = slot_in_week % self.K
        return sim_w, d_w, t_sim, self._day_idx(sim_w, d_w)

    def _day_of_week_norm(self, d_w: int):
        """Return ``(dayOfWeekNorm, dayOfWeek)`` for 0-based day within a week."""
        day_of_week = int(((self.start_dow - 1 + int(d_w)) % self.W_days) + 1)
        denom = (self.W_days - 1.0) / 2.0
        norm = 0.0 if denom == 0 else (day_of_week - (1.0 + self.W_days) / 2.0) / denom
        return float(norm), day_of_week

    def start_week(self, k, I_w):
        self._Iw_per_week[k] = I_w
        # ``E_known_all[k]`` is set in ``__init__`` (k=0) or by
        # ``_finalize_week(k-1)`` (k >= 1); see ``agents.ew_hat.compute_Ew_hat_from_week``.
        # ``wp_all[k]`` is also already populated: bootstrapped to 1.0 for
        # k == 0, and drawn at the end of week k-1 for k >= 1 (see
        # ``_finalize_week``). Daily mediators read ``I_w * wp_all[sim_w]``
        # with ``wp_all[sim_w]`` already a realized 0/1.

        self._foursc_wk[:] = 0.0 
        self._antic_wk[:] = 0.0
        self._pw_wk[:] = 0.0
        self._dw_wk[:] = 0.0
        self._dp_wk[:] = 0.0

    def get_week_packet(self, k):
        """Packet for the belief update at the start of RL week k (0-based).

        Simulated week ``k-1`` must already be finalized (``run_episode`` does
        ``_finalize_week(k-1)`` before this). ``get_pf_data(k)`` supplies PF inputs.
        ``Y_prev`` / ``tY_prev`` always carry the latent draws;
        ``J_w = int(wp)`` (0/1) gates whether the PF counts them as observed.
        """
        sim_w_prev = k - 1
        if sim_w_prev < 0:
            return None

        pf_data = self.get_pf_data(k)

        # Outcomes of simulated week ``sim_w_prev`` are stored at index
        # ``sim_w_prev + 1`` (index 0 = baseline).  J_w for week ``sim_w_prev``
        # is predetermined and lives at ``wp_all[sim_w_prev]``.
        y_sim = self.CAE_all[self._weekly_idx(sim_w_prev)]
        ty_sim = self.CAE_short_all[self._weekly_idx(sim_w_prev)]
        wp = self.wp_all[sim_w_prev]
        jw = int(float(wp))
        Y_prev = float(y_sim)
        tY_prev = float(ty_sim)

        return WeekPacket(
            k=k,
            pf_data=pf_data,
            Y_prev=Y_prev,
            tY_prev=tY_prev,
            J_w=jw,
        )

    def _morning_fitbit_worn(self, d_global: int) -> bool:
        """Whether Fitbit was worn on the morning that starts calendar day ``d_global``.

        ``morningFitbitWearAll[d-1]`` is realised at end of day ``d-1`` and matches df_fit:
        that row's morning-wearing flag governs observability of day ``d``'s
        prior-2-hour and 4-hour step counts.
        """
        if d_global <= 0:
            return float(self.s.get("morningFitbitWearYesterday", 0.0)) >= 0.5
        return float(self.morningFitbitWearAll[d_global - 1]) >= 0.5 # means that the fitbit was worn on the morning of the day before

    def _impute_prior2hour_7day_mean(self) -> float:
        """Impute missing prior-2-hour step count with the mean of the last 7 observed values."""
        if not self._hist_prior2hour_observed:
            return float(self.s.get("prior2HourStepCountEma7d", 0.0))
        return float(np.mean(self._hist_prior2hour_observed[-7:]))

    def _agent_prior2hour(self, d_global: int, prior2hour_latent: float):
        """Return (observed, agent-visible) prior-2-hour values for one slot."""
        if self._morning_fitbit_worn(d_global):
            val = float(prior2hour_latent)
            self._hist_prior2hour_observed.append(val)
            return val, val
        imputed = self._impute_prior2hour_7day_mean()
        return np.nan, imputed

    def _active_particle_state(self):
        """Return the current particle CAE paths and weights for mediator imputation."""
        pf_runtime = self._active_pf_runtime
        if pf_runtime is None:
            return None, None
        return (
            np.asarray(pf_runtime.y_hat, dtype=float),
            np.asarray(pf_runtime.v_hat, dtype=float),
        )

    def _particle_cae_for_feature(self, y_hat_j, sim_w):
        """CAE lag used in mediator rows for simulated week ``sim_w``."""
        sim_w = int(sim_w)
        if sim_w <= 0:
            return float(self._cae_baseline)
        y_hat_j = np.asarray(y_hat_j, dtype=float).ravel()
        if sim_w < y_hat_j.size and np.isfinite(y_hat_j[sim_w]):
            return float(y_hat_j[sim_w])
        finite = y_hat_j[np.isfinite(y_hat_j)]
        return float(finite[-1]) if finite.size else float(self._cae_baseline)

    def _posterior_predictive_my_mean(self, mediator_idx, x_base, cae_delta):
        """Particle-weighted posterior predictive mean for one missing MY value.

        Reuses the per-particle MY posterior means computed during the week's
        PF update (``ParticleFilterRuntime.theta_MY_mean[m][k, j]``) instead of
        re-fitting a regression here. Those means are conditioned on each
        particle's own CAE trajectory through the cumulative complete-case rows,
        so they are exactly the posterior the PF used to propagate beliefs.
        Particles with no recorded posterior (e.g. week 0) fall back to the prior.
        """
        agent = self._active_agent
        pf_runtime = self._active_pf_runtime
        y_hat, v_hat = self._active_particle_state()
        if agent is None or pf_runtime is None or y_hat is None or v_hat is None:
            return 0.0

        m = int(mediator_idx)
        k = int(self._active_rl_week)
        nu0 = np.asarray(agent.nu_0_MY[m], dtype=float).ravel()
        x_base = np.asarray(x_base, dtype=float).ravel()
        cae_delta = np.asarray(cae_delta, dtype=float).ravel()

        nu_post_all = np.asarray(pf_runtime.theta_MY_mean[m][k], dtype=float)  # (J, p)

        weights = np.asarray(v_hat, dtype=float).ravel()
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        if weights.size != y_hat.shape[0] or weights.sum() <= 0:
            weights = np.full(y_hat.shape[0], 1.0 / y_hat.shape[0])
        else:
            weights = weights / weights.sum()

        preds = np.empty(y_hat.shape[0], dtype=float)
        for j in range(y_hat.shape[0]):
            nu_post = nu_post_all[j]
            if nu_post.shape != nu0.shape or not np.all(np.isfinite(nu_post)):
                nu_post = nu0
            cae_pred = self._particle_cae_for_feature(y_hat[j], k)
            x_pred = x_base + cae_pred * cae_delta
            preds[j] = float(x_pred @ nu_post)

        pred = float(np.average(preds, weights=weights))
        return pred if np.isfinite(pred) else 0.0

    def _agent_fourSC(self, step_idx: int, d_global: int, Ah: float, fourSC_latent: float):
        """Return (observed, agent-visible) four-hour step counts for one slot."""
        if self._morning_fitbit_worn(d_global):
            val = float(fourSC_latent)
            return val, val
        x_base = self._pf_foursc_row(int(step_idx))
        imputed = self._posterior_predictive_my_mean(
            0, x_base, fourSC_cae_delta(x_base)
        )
        return np.nan, imputed

    def _agent_antic(self, d_global: int, ws_m: float, ws_a: float,
                     antic_latent: float, survey_present: bool):
        """Return (observed, agent-visible) anticipated affect for one day."""
        if survey_present:
            val = float(antic_latent)
            return val, val
        x_base = self._pf_antic_row(int(d_global))
        imputed = self._posterior_predictive_my_mean(
            1, x_base, antic_cae_delta(x_base)
        )
        return np.nan, imputed

    def run_episode(self, agent, dataset):
        agent.reset(dataset)
        # Fixed-policy baselines (never/always/random send) set
        # ``needs_belief = False``: their ``act`` ignores the belief state and
        # the RLSVI betas, and the evaluated outcome (env latent CAE) never
        # reads them, so the particle filter and the RLSVI refit are pure
        # overhead and are skipped. Mediator imputation and ``results`` already
        # tolerate a ``None`` PF runtime / unfilled belief & beta stores.
        needs_belief = getattr(agent, "needs_belief", True)
        pf_runtime = (
            ParticleFilterRuntime(agent, dataset, agent.rng)
            if needs_belief else None
        )
        self._active_agent = agent
        self._active_pf_runtime = pf_runtime
        self._hist_daily_suggestions.clear()
        self.s["activitySuggestionsSentLast7Days"] = (
            self._activitySuggestionsSentLast7Days_initial
        )
        self._hist_ws_interaction = (
            [self._ws_interaction_initial] * INTERACTION_ROLLING_WINDOW
        )

        self.s["activitySuggestionInteractLast7Days"] = self._ws_interaction_initial

        self._hist_recordedPhysicalActivityToday = [self._rpa7_initial] * RPA_ROLLING_WINDOW
        self.s["activityCompletedLast7Days"] = self._rpa7_initial

        
        self.s["morningFitbitWearLast7Days"] = self._wear7_initial
        self._hist_morning_wear = [self._wear7_initial] * 7
        self._hist_active_days = [self._active_days7_initial] * RPA_ROLLING_WINDOW

        self._hist_prior2hour_observed = []
        self._hist_prior2hour_by_slot = {0: [], 1: []}
        self._hist_foursc_by_slot = {0: [], 1: []}
        self._stepCountLast7DaysEma_by_slot = {
            0: self._stepCountLast7DaysEma_initial,
            1: self._stepCountLast7DaysEma_initial,
        }
        self.s["stepCountLast7DaysEma"] = self._stepCountLast7DaysEma_initial
        p2h0 = float(self.s.get("prior2HourStepCount", 0.0))
        if np.isfinite(p2h0):
            self._hist_prior2hour_observed.append(p2h0)
        self.prior2HourStepCountAgentAll[0] = float(self.s.get("prior2HourStepCountAgent", p2h0))

        self.logYesterdayStepCount[0] = float(self.s["yesterdayStepCount"])
        self.logPageViewLast7DaysEma[0] = float(self.s["pageViewLast7DaysEma"])
        self.logSalienceMessageSentYesterday[0] = float(self.s.get("salienceMessageSentYesterday", 0.0))
        self.logMorningFitbitWearLast7Days[0] = float(self.s.get("morningFitbitWearLast7Days", 0.0))
        self.logDailyAnticipatedAffectYesterday[0] = float(
            self.s.get("dailyAnticipatedAffectYesterdayAgent", 0.0)
        )
        self.logDayOfWeekNorm[0] = float(self.s["dayOfWeekNorm"])
        self.activityCompletedLast7DaysAll[0] = float(self.s.get("activityCompletedLast7Days", 0.0))
        self.logActiveDaysLast7Days[0] = float(self.s.get("activeDaysLast7Days", 0.0))

        for k in range(self.nweek):
            packet = self.get_week_packet(k)
            dataset.record_week_start(
                k, make_state(self.get_context(k, QUERY_D, QUERY_T))
            )
            I_w = agent.begin_week(k, packet)
            if needs_belief:
                pf_runtime.update_standard(k, packet, I_w)
            self._active_packet = packet
            self._active_rl_week = k

            self.start_week(k, I_w)
            if needs_belief and hasattr(agent, "prepare_week"):
                agent.prepare_week(k)

            for d in range(N_RL_DAYS):
                # Day 0 walks under last week's beta; refit beta_k before day 1.
                if needs_belief and d == 1 and hasattr(agent, "update_rlsvi"):
                    agent.update_rlsvi(k)
                for t in range(self.K):
                    context = self.get_context(k, d, t)
                    state = make_state(context)
                    A_wdt, pi_A = agent.act(k, d, t, state)
                    dataset.record_walking(k, d, t, state, A_wdt, pi_A)
                    self.step_action(k, d, t, A_wdt, I_w)

            self._finalize_week(k)

            # Record the full realized weekly mediators for reward shaping
            # (post-action decision-time/daily mediators). Skipped for
            # fixed-policy baselines, which do no learning.
            if needs_belief:
                full_ctx = self.get_context(k, N_RL_DAYS, 0)
                dataset.record_full_week_mediators(
                    k, full_ctx["M_Y"], full_ctx["M_E"])

        return agent.results(dataset)

    def step_action(self, k, d, t, action, I_w):
        if not (0 <= d < N_RL_DAYS):
            raise ValueError(f"RL day d must be in 0..{N_RL_DAYS - 1}, got {d}")
        if not (0 <= t < self.K):
            raise ValueError(f"decision slot t must be in 0..{self.K - 1}, got {t}")
        sim_w, d_w, t_sim = int(k), int(d), int(t)
        d_global = self._day_idx(sim_w, d_w)
        step_idx = self._step_idx(sim_w, d_w, t_sim)

        self._Iw_per_week[sim_w] = I_w

        if t_sim == 0 and (sim_w, d_w) not in self._day_started:
            self._day_started[(sim_w, d_w)] = True
            self._start_day(sim_w, d_w, d_global)

        self.s["decisionTimeSlot"] = float(t_sim)
        Ah = float(action)

        self.activitySuggestionsSentLast7DaysAll[step_idx] = float(
            self.s["activitySuggestionsSentLast7Days"]
        )
        self.logActivitySuggestionInteractLast7Days[step_idx] = float(
            self.s.get("activitySuggestionInteractLast7Days", 0.0)
        )
        self.action_all[step_idx] = Ah

        slot = int(t_sim)
        self.s["stepCountLast7DaysEma"] = self._stepCountLast7DaysEma_by_slot[slot]
        self.logStepCountLast7DaysEma[step_idx] = float(self.s["stepCountLast7DaysEma"])

        fourSC = self.env.gen_fourSC(self.s, Ah, step_idx)
        pv = self.env.gen_pageview(self.s, Ah, I_w*self.wp_all[sim_w], step_idx)

        # Generate current-slot auxiliary mediators using the OLD state summaries.
        # Important: interaction history should not include the current slot
        # until after ws_interaction has been generated.
        ema_hist = self._hist_prior2hour_by_slot[slot]
        if ema_hist:
            self.s["prior2HourStepCountEma7d"] = _ewm_prior_gamma_last(ema_hist)
        else:
            self.s["prior2HourStepCountEma7d"] = float(
                self.s.get("prior2HourStepCountEma7d", 0.0)
            )
        prior2HourStepCount = self.env.gen_prior2hour_step_count(self.s, step_idx)
        ws_interaction = self.env.gen_ws_interaction(self.s, step_idx)

        p2h_obs, p2h_agent = self._agent_prior2hour(d_global, prior2HourStepCount)
        self.prior2HourStepCountAll[step_idx] = prior2HourStepCount
        self.prior2HourStepCountObsAll[step_idx] = p2h_obs
        self.prior2HourStepCountAgentAll[step_idx] = p2h_agent

        sc_obs, sc_agent = self._agent_fourSC(step_idx, d_global, Ah, fourSC)

        self.stepCountNext4HourAll[step_idx] = fourSC
        self.stepCountNext4HourObsAll[step_idx] = sc_obs
        self.stepCountNext4HourAgentAll[step_idx] = sc_agent
        self.pageViewNext4HourAll[step_idx] = pv
        self.ws_interaction_all[step_idx] = ws_interaction

        self._today_fourSC[t_sim] = fourSC
        self._today_pageview[t_sim] = pv
        self._today_action[t_sim] = Ah

        week_slot = d_w * self.K + t_sim
        self._foursc_wk[week_slot] = fourSC
        self._pw_wk[week_slot] = pv

        # Latent lag/state for the env generative chain (unchanged when unobserved).
        self.s["stepCountNext4HourLag1"] = fourSC
        self.s["pageViewNext4HourLag1"] = pv
        self.s["prior2HourStepCount"] = prior2HourStepCount
        self.s["prior2HourStepCountAgent"] = p2h_agent
        self._hist_prior2hour_by_slot[slot].append(float(p2h_agent))
        self._hist_foursc_by_slot[slot].append(float(sc_agent))
        self._stepCountLast7DaysEma_by_slot[slot] = self._stepCountLast7DaysEma_for_slot(slot)

        # Update rolling 7-day interaction fractions after observing current slot.
        self._hist_ws_interaction.append(float(ws_interaction))

        self.s["activitySuggestionInteractLast7Days"] = _rolling_mean_last(
            self._hist_ws_interaction,
            INTERACTION_ROLLING_WINDOW,
        )

        if t_sim == self.K - 1:
            self._end_day(sim_w, d_w, d_global)

    def _start_day(self, sim_w, d_w, d_global):
        # Periodic calendar within each RL week: d_w ∈ {0,…,6} (Mon–Sun block), not d_global.
        # Match ``df_fit``: (dayOfWeekNorm - (1+7)/2) / ((7-1)/2)  →  dayOfWeekNorm ∈ {1,…,7}  to  [-1, 1]; 1 = Monday.
        dayOfWeekNorm_n, dayOfWeekNorm_i = self._day_of_week_norm(d_w)
        self.s["dayOfWeekNorm"] = dayOfWeekNorm_n
        self.s["isWeekend"] = 1.0 if dayOfWeekNorm_i >= 6 else 0.0
        sal = float(rd.binomial(1, 0.5))
        self.s["salienceMessageSentToday"] = sal

        # Daily RPA is modeled on the morning row.
        self.s["decisionTimeSlot"] = 0.0

        morning_step_idx = self._step_idx(sim_w, d_w, 0)

        rpa = self.env.gen_recorded_physical_activity(
            self.s,
            morning_step_idx,
        )

        self.s["recordedPhysicalActivityToday"] = rpa
        self.recordedPhysicalActivityTodayAll[d_global] = rpa

        active_status = self.env.gen_active_status(self.s, d_global)
        self.s["activityStatusToday"] = active_status
        self.activityStatusTodayAll[d_global] = active_status

        self.logDayOfWeekNorm[d_global] = dayOfWeekNorm_n
        self.logSalienceMessageSentToday[d_global] = sal
        self.logSalienceMessageSentYesterday[d_global] = float(self._prev_day_salience)
        self.logActiveDaysLast7Days[d_global] = float(
            self.s.get("activeDaysLast7Days", 0.0)
        )
        if self._hist_daily_pv:
            self.logPageViewLast7DaysEma[d_global] = _ewm_prior_gamma_last(
                self._hist_daily_pv
            )
        else:
            self.logPageViewLast7DaysEma[d_global] = float(self.s["pageViewLast7DaysEma"])
        self.logMorningFitbitWearLast7Days[d_global] = _rolling_mean_last(self._hist_morning_wear, 7)

        # Store the previous-7-day predictor used during this day.
        self.activityCompletedLast7DaysAll[d_global] = _rolling_mean_last(
            self._hist_recordedPhysicalActivityToday,
            RPA_ROLLING_WINDOW,
        )

        if self._hist_daily_suggestions:
            self.s["activitySuggestionsSentLast7Days"] = _ewm_prior_gamma_last(
                self._hist_daily_suggestions,
            )
        else:
            self.s["activitySuggestionsSentLast7Days"] = (
                self._activitySuggestionsSentLast7Days_initial
            )

        # Antic / fitbit / daily are drawn in _end_day (today’s WS). gen_pageview uses
        # yesterday_* in vani_env; fourSC uses dailyAnticipatedAffectYesterday (prior-day affect) only.

        self._today_fourSC[:] = 0.0
        self._today_pageview[:] = 0.0
        self._today_action[:] = 0.0

    def _end_day(self, sim_w, d_w, d_global):
        # Yesterday's step count = sum of the two 4-hour slots (AM + PM).
        daily_sum_step = float(np.sum(self._today_fourSC))
        daily_mean_pv = np.mean(self._today_pageview)

        ws_m = float(self._today_action[0])
        ws_a = float(self._today_action[1])
        Iw = self._Iw_per_week[sim_w]

        # Same-day predictors for daily mediators (before fitbit / dailysurvey / antic).
        self.s["todayStepCount"] = daily_sum_step
        self.s["pageViewMorningToday"] = float(self._today_pageview[0])
        self.s["pageViewAfternoonToday"] = float(self._today_pageview[1])

        fitbit = self.env.gen_fitbitwearing(self.s, ws_m, ws_a, Iw*self.wp_all[sim_w], d_global)
        self.s["morningFitbitWear"] = fitbit
        daily_pres = self.env.gen_dailysurvey(self.s, ws_m, ws_a, Iw*self.wp_all[sim_w], d_global)
        self.s["dailySurveyComplete"] = daily_pres
        # Latent anticipated affect — always drawn from the generative process.
        # Stored as s["dailyAnticipatedAffect"] so that tomorrow's decision-slot calls
        # to gen_fourSC_mean / gen_antic_mean in vani_env.py see the true latent
        # from yesterday (not the NaN-masked carry-forward).
        antic = self.env.gen_antic(self.s, ws_m, ws_a, d_global)
        self.s["dailyAnticipatedAffect"] = antic

        self.morningFitbitWearAll[d_global] = fitbit
        self.dailySurveyCompleteAll[d_global] = daily_pres
        self.dailyAnticipatedAffectAll[d_global] = antic                                   # latent, always finite

        day_base = self._step_idx(sim_w, d_w, 0)
        daily_sum_step_agent = float(np.sum(
            self.stepCountNext4HourAgentAll[day_base:day_base + self.K]
        ))
        self.logTodayStepCount[d_global] = daily_sum_step_agent

        survey_present = float(daily_pres) == 1.0
        antic_obs, antic_agent = self._agent_antic(
            d_global, ws_m, ws_a, antic, survey_present
        )
        self.dailyAnticipatedAffectObsAll[d_global] = antic_obs
        self.dailyAnticipatedAffectAgentAll[d_global] = antic_agent

        # Latent carry-forward for the env generative chain (observed days only).
        if survey_present:
            self.s["dailyAnticipatedAffectYesterday"] = antic
        self.s["dailyAnticipatedAffectYesterdayAgent"] = antic_agent

        self.s["yesterdayStepCount"] = daily_sum_step
        self.s["morningFitbitWearYesterday"] = fitbit
        self.s["dailySurveyCompleteYesterday"] = daily_pres
        self.s["pageViewMorningYesterday"] = self._today_pageview[0]
        self.s["pageViewAfternoonYesterday"] = self._today_pageview[1]
        self.s["dayOfWeekNormLag1"] = self.s["dayOfWeekNorm"]

        self._hist_daily_pv.append(float(daily_mean_pv))
        self._hist_morning_wear.append(float(fitbit))
        self.s["pageViewLast7DaysEma"] = _ewm_prior_gamma_last(self._hist_daily_pv)

        past7_wear = _rolling_mean_last(self._hist_morning_wear, 7)
        self.s["morningFitbitWearLast7Days"] = past7_wear

        self._prev_day_salience = float(self.s["salienceMessageSentToday"])
        if d_global + 1 < self.D:
            self.logYesterdayStepCount[d_global + 1] = daily_sum_step_agent
            self.logSalienceMessageSentYesterday[d_global + 1] = float(self._prev_day_salience)
            self.logDailyAnticipatedAffectYesterday[d_global + 1] = float(antic_agent)

        self._yesterday_morning_WS = self._today_action[0]
        self._yesterday_afternoon_WS = self._today_action[1]

        daily_suggestions = float(self._today_action[0] + self._today_action[1])
        self._hist_daily_suggestions.append(daily_suggestions)

        self._antic_wk[d_w] = antic
        self._dw_wk[d_w] = fitbit
        self._dp_wk[d_w] = daily_pres

        # After today's daily RPA is known, update histories for tomorrow.
        today_rpa = float(self.s["recordedPhysicalActivityToday"])

        self._hist_recordedPhysicalActivityToday.append(today_rpa)

        self.s["recordedPhysicalActivityLag1"] = today_rpa
        self.s["activityCompletedLast7Days"] = _rolling_mean_last(
            self._hist_recordedPhysicalActivityToday,
            RPA_ROLLING_WINDOW,
        )

        self._hist_active_days.append(float(self.s.get("activityStatusToday", 0.0)))
        self.s["activeDaysLast7Days"] = _rolling_mean_last(
            self._hist_active_days,
            RPA_ROLLING_WINDOW,
        )

        # fourSC / pageview use calendar-yesterday message flags (lagged one day).
        self.s["salienceMessageSentYesterday"] = self.s["salienceMessageSentToday"]

    def _finalize_week(self, sim_w):
        if sim_w < 0 or sim_w >= self.nweek or self._week_finalized[sim_w]:
            return

        Iw = self._Iw_per_week[sim_w]
        d_w_sun = SUNDAY_D_W
        d_global = self._day_idx(sim_w, d_w_sun)
        # Match df_fit / 1.1_standardization: (week - (1+n)/2) / ((n-1)/2) → week ∈ {1,…,n} maps to [-1, 1]
        wk = sim_w + 1
        n = self.nweek
        week_norm = 0.0 if n <= 1 else (wk - (1.0 + n) / 2.0) / ((n - 1.0) / 2.0)

        if (sim_w, d_w_sun) not in self._day_started:
            self._day_started[(sim_w, d_w_sun)] = True
            self._start_day(sim_w, d_w_sun, d_global)

        for t_sim in range(self.K):
            step_idx = self._step_idx(sim_w, d_w_sun, t_sim)
            Ah = 0.0
            self.s["decisionTimeSlot"] = float(t_sim)
            self.activitySuggestionsSentLast7DaysAll[step_idx] = float(
                self.s["activitySuggestionsSentLast7Days"]
            )
            self.logActivitySuggestionInteractLast7Days[step_idx] = float(
                self.s.get("activitySuggestionInteractLast7Days", 0.0)
            )
            self.s["stepCountLast7DaysEma"] = self._stepCountLast7DaysEma_by_slot[int(t_sim)]
            self.logStepCountLast7DaysEma[step_idx] = float(
                self.s["stepCountLast7DaysEma"]
            )
            self.action_all[step_idx] = Ah

            fourSC = self.env.gen_fourSC(self.s, Ah, step_idx)
            pv = self.env.gen_pageview(self.s, Ah, Iw*self.wp_all[sim_w], step_idx)

            sc_obs, sc_agent = self._agent_fourSC(step_idx, d_global, Ah, fourSC)
            self.stepCountNext4HourAll[step_idx] = fourSC
            self.stepCountNext4HourObsAll[step_idx] = sc_obs
            self.stepCountNext4HourAgentAll[step_idx] = sc_agent
            self.pageViewNext4HourAll[step_idx] = pv
            self._today_fourSC[t_sim] = fourSC
            self._today_pageview[t_sim] = pv
            self._today_action[t_sim] = Ah
            week_slot = d_w_sun * self.K + t_sim
            self._foursc_wk[week_slot] = fourSC
            self._pw_wk[week_slot] = pv

            self.s["stepCountNext4HourLag1"] = fourSC
            self.s["pageViewNext4HourLag1"] = pv
            self._hist_foursc_by_slot[int(t_sim)].append(float(sc_agent))
            self._stepCountLast7DaysEma_by_slot[int(t_sim)] = (
                self._stepCountLast7DaysEma_for_slot(int(t_sim))
            )

        self._end_day(sim_w, d_w_sun, d_global)

        cae = self.env.gen_CAE(
            self.s["caeAverageLastWeek"], week_norm,
            self._foursc_wk, self._antic_wk, sim_w,
        )

        pu = self.env.gen_perceivedUtility(
            self.s["perceivedUtilityLastWeek"], week_norm,
            self._pw_wk, self._dw_wk, self._dp_wk, sim_w,
        )

        cs = self.env.gen_CAE_short(cae, sim_w)

        # Tool-survey draws (only meaningful when ``J_w[sim_w] == 1``; the
        # agent's E_w approximation multiplies by ``J_w[sim_w]`` so the term
        # is zeroed out otherwise).
        u1 = self.env.gen_tool_U1(pu, sim_w)
        u2 = self.env.gen_tool_U2(pu, sim_w)


        # ``wp_all[sim_w]`` was set at end of week ``sim_w - 1`` (or in
        # ``__init__`` for ``sim_w == 0``); we do NOT overwrite it here.
        # Instead, draw ``J_w[sim_w + 1]`` conditioned on this week's ``pu``.
        weekly_idx = self._weekly_idx(sim_w)
        self.CAE_all[weekly_idx] = cae
        self.pu_all[weekly_idx] = pu
        self.wp_all[weekly_idx] = self.env.gen_week_present(pu, weekly_idx)
        self.CAE_short_all[weekly_idx] = cs
        self.U1_all[weekly_idx] = u1
        self.U2_all[weekly_idx] = u2

        self.s["caeAverageLastWeek"] = cae
        # Latent E_w driving env generative process (PV / FW / PJ / J / fourSC / antic).
        # The agent never sees this; it gets ``E_known_all`` instead.
        self.s["perceivedUtilityLastWeek"] = pu

        # Compute E_w_hat for the *next* RL week using observable aggregates of
        # the just-finalized week (J, U1, U2, PV_sum, FW_sum, PJ_sum).

        self.E_known_all[weekly_idx] = compute_Ew_hat_from_week(
            sim_w,
            coefs=self._ew_coefs,
            wp_all=self.wp_all,
            U1_all=self.U1_all,
            U2_all=self.U2_all,
            pageViewNext4HourAll=self.pageViewNext4HourAll,
            dw_wk=self._dw_wk,
            dp_wk=self._dp_wk,
            baseline_offset=BASELINE_OFFSET,
        )

        self._week_finalized[sim_w] = True

    def _week_norm(self, sim_w):
        wk = sim_w + 1
        n = self.nweek
        return 0.0 if n <= 1 else (wk - (1.0 + n) / 2.0) / ((n - 1.0) / 2.0)

    def _stepCountLast7DaysEma_for_slot(self, slot):
        """EWM of prior ≤7 same-slot 4-hour step counts (excludes current slot).

        Matches ``1_data_extraction`` ``EMA_StepCount`` (groupby DecisionTime).
        """
        hist = self._hist_foursc_by_slot[int(slot)]
        if hist:
            return _ewm_prior_gamma_last(hist)
        return self._stepCountLast7DaysEma_initial

    def _pf_foursc_row(self, step_idx):
        sim_w, _d_w, t_sim, d_global = self._decode_step_idx(step_idx)
        Ah = float(self.action_all[step_idx])
        return build_fourSC_features(
            yesterdayStepCount=self.logYesterdayStepCount[d_global],
            stepCountLast7DaysEma=self.logStepCountLast7DaysEma[step_idx],
            prior2HourStepCount=float(self.prior2HourStepCountAgentAll[step_idx]),
            activityCompletedLast7Days=self.activityCompletedLast7DaysAll[d_global],
            activitySuggestionsSentLast7Days=float(
                self.activitySuggestionsSentLast7DaysAll[step_idx]
            ),
            morningFitbitWearLast7Days=self.logMorningFitbitWearLast7Days[d_global],
            salienceMessageSentYesterday=self.logSalienceMessageSentYesterday[d_global],
            activitySuggestionInteractLast7Days=self.logActivitySuggestionInteractLast7Days[step_idx],
            activeDaysLast7Days=self.logActiveDaysLast7Days[d_global],
            dayOfWeekNorm=self.logDayOfWeekNorm[d_global],
            decisionTimeSlot=float(t_sim),
            perceivedUtility=self.E_known_all[sim_w],
            caeAverageLastWeek=0.0,
            Ah=Ah,
            cae=0.0,
        )

    def _pf_antic_row(self, d_global):
        sim_w = d_global // self.W_days
        d_w = d_global % self.W_days
        base = self._step_idx(sim_w, d_w, 0)
        ws_m = float(self.action_all[base])
        ws_a = float(self.action_all[base + 1])
        return build_antic_features(
            dailyAnticipatedAffectYesterday=self.logDailyAnticipatedAffectYesterday[d_global],
            todayStepCount=self.logTodayStepCount[d_global],
            recordedPhysicalActivityToday=self.recordedPhysicalActivityTodayAll[d_global],
            activityStatusToday=self.activityStatusTodayAll[d_global],
            salienceMessageSentToday=self.logSalienceMessageSentToday[d_global],
            dayOfWeekNorm=self.logDayOfWeekNorm[d_global],
            perceivedUtility=self.E_known_all[sim_w],
            caeAverageLastWeek=0.0,
            ws_morning=ws_m,
            ws_afternoon=ws_a,
            pu=self.E_known_all[sim_w],
            cae=0.0,
        )

    def _pf_cae_row(self, sim_w):
        slot_start = self._step_idx(sim_w, 0, 0)
        slot_stop = self._step_idx(sim_w + 1, 0, 0)
        day_start = self._day_idx(sim_w, 0)
        day_stop = self._day_idx(sim_w + 1, 0)
        foursc_wk = self.stepCountNext4HourAgentAll[slot_start:slot_stop]
        antic_wk = self.dailyAnticipatedAffectAgentAll[day_start:day_stop]
        return build_CAE_features(0.0, self._week_norm(sim_w), foursc_wk, antic_wk)

    def _rl_context_vector(self, k, d, t):
        d_global = self._day_idx(k, d)
        step_idx = self._step_idx(k, d, t)
        if step_idx > 0 and np.isfinite(self.prior2HourStepCountAgentAll[step_idx - 1]):
            p2h = float(self.prior2HourStepCountAgentAll[step_idx - 1])
        elif np.isfinite(self.prior2HourStepCountAgentAll[0]):
            p2h = float(self.prior2HourStepCountAgentAll[0])
        else:
            p2h = float(self.s.get("prior2HourStepCountAgent", self._initial_prior2hour))
        return build_rl_context_vector(
            yesterdayStepCount=self.logYesterdayStepCount[d_global],
            stepCountLast7DaysEma=self._stepCountLast7DaysEma_by_slot[int(t)],
            prior2HourStepCountAgent=p2h,
            activityCompletedLast7Days=self.activityCompletedLast7DaysAll[d_global],
            activeDaysLast7Days=float(self.s["activeDaysLast7Days"]),
            activitySuggestionsSentLast7Days=float(self.s["activitySuggestionsSentLast7Days"]),
            salienceMessageSentYesterday=self.logSalienceMessageSentYesterday[d_global],
            activitySuggestionInteractLast7Days=float(
                self.s["activitySuggestionInteractLast7Days"]
            ),
        )

    def get_pf_data(self, k):
        """PF inputs for week ``k``; rows built from logged outcomes + covariates."""
        return build_pf_data(
            k,
            sim_w_prev=k - 1,
            nweek=self.nweek,
            W_days=self.W_days,
            stepCountNext4HourObsAll=self.stepCountNext4HourObsAll,
            dailyAnticipatedAffectObsAll=self.dailyAnticipatedAffectObsAll,
            CAE_all=self.CAE_all,
            CAE_short_all=self.CAE_short_all,
            wp_all=self.wp_all,
            week_finalized=self._week_finalized,
            cae_baseline=self._cae_baseline,
            fourSC_row_fn=self._pf_foursc_row,
            antic_row_fn=self._pf_antic_row,
            cae_row_fn=self._pf_cae_row,
        )

    def get_context(self, k, d, t):
        """Decision-point context: mediators from trajectories, ``C`` from logs."""
        E_w = self.E_known_all[k] if (
            0 <= k < self.nweek and not np.isnan(self.E_known_all[k])
        ) else 0.0

        if d == QUERY_D and t == QUERY_T:
            return {
                "E_w": E_w,
                "M_Y": np.zeros(RL_MY_SHAPE),
                "M_E": np.zeros(RL_ME_SHAPE),
                "C": np.zeros(N_RL_CONTEXT),
            }

        M_Y = np.zeros(RL_MY_SHAPE)
        M_E = np.zeros(RL_ME_SHAPE)
        for dd in range(N_RL_DAYS):
            d_global = self._day_idx(k, dd)
            if dd < d and 0 <= d_global < self.D:
                M_Y[dd, 2] = self.dailyAnticipatedAffectAgentAll[d_global]
                M_E[dd, 2] = self.morningFitbitWearAll[d_global]
                M_E[dd, 3] = self.dailySurveyCompleteAll[d_global]
            for tt in range(self.K):
                if (dd, tt) < (d, t):
                    r = self._step_idx(k, dd, tt)
                    if r < self.T:
                        M_Y[dd, tt] = self.stepCountNext4HourAgentAll[r]
                        M_E[dd, tt] = self.pageViewNext4HourAll[r]

        return {
            "E_w": E_w,
            "M_Y": M_Y,
            "M_E": M_E,
            "C": self._rl_context_vector(k, d, t),
        }

_setup_log("OnlineEnv defined.")

# %%
# ──────────────────────────────────────────────────────────────────
# Placeholder priors and hyperparameters
# ──────────────────────────────────────────────────────────────────

# Pooled mediator-model feature dimensions for the particle filter:
# m=0 is fourSC (all slots share one model), m=1 is antic (all days share one model).
# The PF fourSC model is a reduced version of the env's generative model: it drops
# the AR-1 lag, the pageview-EMA, and the anticipated-affect predictors (and their
# action interactions). Its dimension is therefore taken directly from the PF
# feature builder rather than from the env's P_FOURSC.
P_MY_FOURSC = int(
    build_fourSC_features(
        yesterdayStepCount=0.0,
        stepCountLast7DaysEma=0.0,
        prior2HourStepCount=0.0,
        activityCompletedLast7Days=0.0,
        activitySuggestionsSentLast7Days=0.0,
        morningFitbitWearLast7Days=0.0,
        salienceMessageSentYesterday=0.0,
        activitySuggestionInteractLast7Days=0.0,
        activeDaysLast7Days=0.0,
        dayOfWeekNorm=0.0,
        decisionTimeSlot=0.0,
        perceivedUtility=0.0,
        caeAverageLastWeek=0.0,
        Ah=0.0,
    ).shape[0]
)
P_MY_ANTIC  = P_ANTIC
P_CAE = int(build_CAE_features(0.0, 0.0, np.zeros(FOURSC_SLOTS_PER_WEEK), np.zeros(7)).shape[0])
P_TY  = int(build_CAE_short_features(0.0).shape[0])

_n_my_flat = RL_MY_SHAPE[0] * RL_MY_SHAPE[1]
_n_me_flat = RL_ME_SHAPE[0] * RL_ME_SHAPE[1]

# Compute phi dimensions directly from the algorithm's feature builders so
# the priors stay in sync with whatever is in the base / mediator / context /
# action-interaction blocks. Avoids drift when build_phi_action changes
# (e.g. adding b_tilde to the base).
_DUMMY_RL_STATE = {
    "E_w": 0.0,
    "M_Y": np.zeros(RL_MY_SHAPE),
    "M_E": np.zeros(RL_ME_SHAPE),
    "C":   np.zeros(N_RL_CONTEXT),
}
P_RL_MICRO = int(
    build_phi_action(0.0, 0.0, _DUMMY_RL_STATE, 1, 1, 0).shape[0]
)
P_RL_REWARDSHAPING = int(
    build_phi_action_rewardshaping(0.0, 0.0, _DUMMY_RL_STATE, 1, 1).shape[0]
)
P_RL_BOTTLENECK = int(
    build_phi_bottleneck(0.0, 0.0, _DUMMY_RL_STATE).shape[0]
)

# ── RL hyperparameters (shared) ──
# Note: gamma_bar is no longer a single module-level constant; each algorithm
# in ALGORITHMS below is registered at gamma_bar=0.0 and gamma_bar=0.5 via
# functools.partial, and the per-slot discount is computed by _gamma_dt_micro.
TARGET_C     = 1
EPSILON_0    = 0.1   # this is the clipping parameter
J_PARTICLES  = 50
B_ENSEMBLES  = 50
NWEEK        = 36


# ──────────────────────────────────────────────────────────────────
# Load priors estimated from df_fit (env_para_vanilla/rl_priors.json).
# Run ``python est_prior.py`` to (re)generate that file. If it is missing
# (or USE_ESTIMATED_PRIORS=0 is set in the environment) we fall back to
# the original zero / identity placeholder priors so a fresh checkout
# still runs.
# ──────────────────────────────────────────────────────────────────
def _default_pf_priors():
    return {
        "nu_0_MY":    [np.zeros(P_MY_FOURSC), np.zeros(P_MY_ANTIC)],
        "Gamma_0_MY": [np.eye(P_MY_FOURSC),   np.eye(P_MY_ANTIC)],
        "sigma2_MY":  [1.0, 1.0],
        "nu_0_Y":    np.zeros(P_CAE),
        "Gamma_0_Y": np.eye(P_CAE),
        "sigma2_Y":  1.0,
        "nu_0_tilde_Y":    np.zeros(P_TY),
        "Gamma_0_tilde_Y": np.eye(P_TY),
        "sigma2_tilde_Y":  1.0,
    }


def _default_rl_priors(p_rl):
    return (np.zeros(p_rl),  np.eye(p_rl), 1.0)


def _default_rl_joint_priors(p_eta, p_beta):
    """Fallback joint prior for the modified-TD-loss RLSVI agents.

    Returns ``(mu_0, Sigma_0, p_eta, sigma2_bottleneck, sigma2_TD, sigma2_T)``
    with mean 0, covariance ``10 * I_{p_eta+p_beta}`` (block-diagonal only
    because we have no informative prior; the agents will then learn the
    cross-terms from data), and unit noise variances.
    """
    p = p_eta + p_beta
    return (np.zeros(p), np.eye(p), int(p_eta), 1.0, 1.0, 1.0)


def _load_priors():
    """Try ``est_prior.load_estimated_priors``; fall back to defaults on miss."""
    if os.getenv("USE_ESTIMATED_PRIORS", "0") == "0":
        return None, "USE_ESTIMATED_PRIORS=0 -> defaults"
    try:
        from est_prior import load_estimated_priors, OUTPUT_PATH
        if not OUTPUT_PATH.exists():
            return None, f"{OUTPUT_PATH} not found"
        return load_estimated_priors(), str(OUTPUT_PATH)
    except Exception as exc:  # noqa: BLE001 - any failure -> safe fallback
        return None, f"load_estimated_priors failed: {exc!r}"


_priors, _priors_src = _load_priors()
if _priors is None:
    _setup_log(f"[priors] using default zero/identity priors ({_priors_src})")
    _pf = _default_pf_priors()
    nu_0_MY        = _pf["nu_0_MY"]
    Gamma_0_MY     = _pf["Gamma_0_MY"]
    sigma2_MY      = _pf["sigma2_MY"]
    nu_0_Y         = _pf["nu_0_Y"]
    Gamma_0_Y      = _pf["Gamma_0_Y"]
    sigma2_Y       = _pf["sigma2_Y"]
    nu_0_tilde_Y   = _pf["nu_0_tilde_Y"]
    Gamma_0_tilde_Y= _pf["Gamma_0_tilde_Y"]
    sigma2_tilde_Y = _pf["sigma2_tilde_Y"]
    mu_0_micro,     Sigma_0_micro,     sigma2_rl_micro     = _default_rl_priors(P_RL_MICRO)
    mu_0_micro_mtd, Sigma_0_micro_mtd, sigma2_rl_micro_mtd = _default_rl_priors(P_RL_MICRO)
    mu_0_reward,    Sigma_0_reward,    sigma2_reward      = _default_rl_priors(P_RL_REWARDSHAPING)
    mu_0_bottleneck,Sigma_0_bottleneck,sigma2_bottleneck  = _default_rl_priors(P_RL_BOTTLENECK)
    # Joint (alpha, beta) prior for the modified-TD-loss RLSVI agents.
    (mu_0_mtd_joint, Sigma_0_mtd_joint, p_eta_mtd_joint,
     sigma2_bottleneck_mtd_joint, sigma2_TD_mtd_joint,
     sigma2_T_mtd_joint) = _default_rl_joint_priors(
        P_RL_BOTTLENECK, P_RL_MICRO)
else:
    _setup_log(f"[priors] loaded estimated priors from {_priors_src}")
    nu_0_MY        = _priors["nu_0_MY"]
    Gamma_0_MY     = _priors["Gamma_0_MY"]
    sigma2_MY      = _priors["sigma2_MY"]
    nu_0_Y         = _priors["nu_0_Y"]
    Gamma_0_Y      = _priors["Gamma_0_Y"]
    sigma2_Y       = _priors["sigma2_Y"]
    nu_0_tilde_Y   = _priors["nu_0_tilde_Y"]
    Gamma_0_tilde_Y= _priors["Gamma_0_tilde_Y"]
    sigma2_tilde_Y = _priors["sigma2_tilde_Y"]
    # Q prior used by MicroQueryAgent / MicroQueryAgent_rewardshaping
    # (no TD-modify variant).
    mu_0_micro      = _priors["mu_0_micro"]
    Sigma_0_micro   = _priors["Sigma_0_micro"]
    sigma2_rl_micro = _priors["sigma2_rl_micro"]
    # Separate Q prior used by the two TD-modify variants
    # (MicroQueryAgent_ModifiedTDLoss and MicroQueryAgent_rewardshaping_modifiedTD)
    # whose FQI target bootstraps from V_alpha at the terminal slot.
    mu_0_micro_mtd      = _priors["mu_0_micro_mtd"]
    Sigma_0_micro_mtd   = _priors["Sigma_0_micro_mtd"]
    sigma2_rl_micro_mtd = _priors["sigma2_rl_micro_mtd"]
    mu_0_reward     = _priors["mu_0_reward"]
    Sigma_0_reward  = _priors["Sigma_0_reward"]
    sigma2_reward   = _priors["sigma2_reward"]
    mu_0_bottleneck = _priors["mu_0_bottleneck"]
    Sigma_0_bottleneck = _priors["Sigma_0_bottleneck"]
    sigma2_bottleneck  = _priors["sigma2_bottleneck"]
    # Joint (alpha, beta) prior for the modified-TD-loss RLSVI agents.
    # Sigma_0 is the FULL joint covariance across users (not block-diagonal).
    # Falls back to the block-diagonal default if the older rl_priors.json
    # was produced before est_prior.py started emitting q_td_modify_joint.
    if "mu_0_micro_mtd_joint" in _priors and _priors["mu_0_micro_mtd_joint"] is not None:
        mu_0_mtd_joint              = _priors["mu_0_micro_mtd_joint"]
        Sigma_0_mtd_joint           = _priors["Sigma_0_micro_mtd_joint"]
        p_eta_mtd_joint             = int(_priors["p_eta_micro_mtd_joint"])
        sigma2_bottleneck_mtd_joint = float(_priors["sigma2_bottleneck_mtd_joint"])
        sigma2_TD_mtd_joint         = float(_priors["sigma2_TD_mtd_joint"])
        sigma2_T_mtd_joint          = float(_priors["sigma2_T_mtd_joint"])
    else:
        _setup_log("[priors] q_td_modify_joint missing from rl_priors.json -> "
                   "falling back to default joint prior (rerun est_prior.py "
                   "to regenerate).")
        (mu_0_mtd_joint, Sigma_0_mtd_joint, p_eta_mtd_joint,
         sigma2_bottleneck_mtd_joint, sigma2_TD_mtd_joint,
         sigma2_T_mtd_joint) = _default_rl_joint_priors(
            P_RL_BOTTLENECK, P_RL_MICRO)

# Sanity check: every loaded prior must agree with the phi-builder dims.
assert nu_0_MY[0].shape == (P_MY_FOURSC,), f"fourSC dim {nu_0_MY[0].shape} != {P_MY_FOURSC}"
assert nu_0_MY[1].shape == (P_MY_ANTIC,),  f"antic dim {nu_0_MY[1].shape} != {P_MY_ANTIC}"
assert nu_0_Y.shape       == (P_CAE,),     f"CAE dim {nu_0_Y.shape} != {P_CAE}"
assert nu_0_tilde_Y.shape == (P_TY,),      f"CAE_short dim {nu_0_tilde_Y.shape} != {P_TY}"
assert mu_0_micro.shape       == (P_RL_MICRO,)
assert mu_0_micro_mtd.shape   == (P_RL_MICRO,)
assert mu_0_reward.shape      == (P_RL_REWARDSHAPING,)
assert mu_0_bottleneck.shape  == (P_RL_BOTTLENECK,)
_P_MTD_JOINT = P_RL_BOTTLENECK + P_RL_MICRO
assert mu_0_mtd_joint.shape    == (_P_MTD_JOINT,), \
    f"joint mu_0 dim {mu_0_mtd_joint.shape} != ({_P_MTD_JOINT},)"
assert Sigma_0_mtd_joint.shape == (_P_MTD_JOINT, _P_MTD_JOINT), \
    f"joint Sigma_0 shape {Sigma_0_mtd_joint.shape} != " \
    f"({_P_MTD_JOINT}, {_P_MTD_JOINT})"
assert p_eta_mtd_joint == P_RL_BOTTLENECK, \
    f"joint p_eta {p_eta_mtd_joint} != P_RL_BOTTLENECK={P_RL_BOTTLENECK}"

_setup_log(
    f"Priors ready:  p_rl(micro)={P_RL_MICRO}, "
    f"p_rs={P_RL_REWARDSHAPING}, p_b={P_RL_BOTTLENECK}, "
    f"p_mtd_joint={_P_MTD_JOINT}"
)

# %%
# ──────────────────────────────────────────────────────────────────
# Run one participant with a given algorithm
# ──────────────────────────────────────────────────────────────────

def _gamma_dt_micro(gamma_bar):
    """Per-slot discount γ_{d,t} = γ_bar^{1/12}.

    When γ_bar=0 the first 11 slots keep discount 1.0 and only the terminal
    slot (6,2) is 0, so bootstrapping flows within the week but stops at the
    terminal transition.
    """
    if gamma_bar == 0.0:
        gamma_dt = np.ones((6, 2))
        gamma_dt[5, 1] = 0.0
        return gamma_dt
    return (gamma_bar ** (1.0 / 12)) * np.ones((6, 2))


def _episode_seed(exp_seed: int, draw_idx: int) -> int:
    """RNG seed for one participant draw within an experiment.

    The experiment-level ``exp_seed`` fixes the uid resample and makes runs
    replicable; ``draw_idx`` gives each cohort slot its own env/agent noise
    path even when the same uid is drawn more than once.
    """
    return int(np.random.SeedSequence([exp_seed, draw_idx]).generate_state(1)[0])


def run_micro_query(uid, seed=42, gamma_bar=0.5):
    cfg = EnvConfig(uid)
    nweek = cfg.nweek
    env = Env(cfg, noise="sequential")
    oenv = OnlineEnv(env, nweek=nweek, seed=seed)

    dataset = EpisodeDataset(nweek)
    agent = MicroQueryAgent(
        W=nweek, J=J_PARTICLES, B=B_ENSEMBLES, epsilon_0=EPSILON_0,
        mu_0_rl=mu_0_micro, Sigma_0_rl=Sigma_0_micro, sigma2_rl=sigma2_rl_micro,
        gamma_dt=_gamma_dt_micro(gamma_bar), gamma_bar=gamma_bar,
        target_update_C=TARGET_C,
        nu_0_MY=nu_0_MY, Gamma_0_MY=Gamma_0_MY, sigma2_MY=sigma2_MY,
        nu_0_Y=nu_0_Y, Gamma_0_Y=Gamma_0_Y, sigma2_Y=sigma2_Y,
        nu_0_tilde_Y=nu_0_tilde_Y, Gamma_0_tilde_Y=Gamma_0_tilde_Y,
        sigma2_tilde_Y=sigma2_tilde_Y,
        Y_1=float(oenv.CAE_all[0]),
        rng=np.random.default_rng(seed),
    )
    # agent.update_sigma2_online = False

    result = oenv.run_episode(agent, dataset)
    return result, oenv


def run_micro_query_rs(uid, seed=42, gamma_bar=0.5):
    """Micro-query agent with reward shaping (per-slot R_dt = phi_rs · eta)."""
    cfg = EnvConfig(uid)
    nweek = cfg.nweek
    env = Env(cfg, noise="sequential")
    oenv = OnlineEnv(env, nweek=nweek, seed=seed)

    agent = MicroQueryAgent_rewardshaping(
        W=nweek, J=J_PARTICLES, B=B_ENSEMBLES, epsilon_0=EPSILON_0,
        mu_0_rl=mu_0_micro, Sigma_0_rl=Sigma_0_micro, sigma2_rl=sigma2_rl_micro,
        gamma_dt=_gamma_dt_micro(gamma_bar), gamma_bar=gamma_bar,
        target_update_C=TARGET_C,
        nu_0_MY=nu_0_MY, Gamma_0_MY=Gamma_0_MY, sigma2_MY=sigma2_MY,
        nu_0_Y=nu_0_Y, Gamma_0_Y=Gamma_0_Y, sigma2_Y=sigma2_Y,
        nu_0_tilde_Y=nu_0_tilde_Y, Gamma_0_tilde_Y=Gamma_0_tilde_Y,
        sigma2_tilde_Y=sigma2_tilde_Y,
        mu_0_reward=mu_0_reward, Sigma_0_reward=Sigma_0_reward,
        sigma2_reward=sigma2_reward,
        Y_1=float(oenv.CAE_all[0]),
        rng=np.random.default_rng(seed),
    )       
    # agent.update_sigma2_online = False
    dataset = EpisodeDataset(nweek)
    result = oenv.run_episode(agent, dataset)
    return result, oenv


def run_micro_query_mtd(uid, seed=42, gamma_bar=0.5):
    """Micro-query agent with modified TD loss (week-start bottleneck V_alpha)."""
    cfg = EnvConfig(uid)
    nweek = cfg.nweek
    env = Env(cfg, noise="sequential")
    oenv = OnlineEnv(env, nweek=nweek, seed=seed)

    agent = MicroQueryAgent_ModifiedTDLoss(
        W=nweek, J=J_PARTICLES, B=B_ENSEMBLES, epsilon_0=EPSILON_0,
        mu_0_joint=mu_0_mtd_joint, Sigma_0_joint=Sigma_0_mtd_joint,
        p_eta=p_eta_mtd_joint,
        sigma2_bottleneck=sigma2_bottleneck_mtd_joint,
        sigma2_TD=sigma2_TD_mtd_joint,
        sigma2_T=sigma2_T_mtd_joint,
        gamma_dt=_gamma_dt_micro(gamma_bar), gamma_bar=gamma_bar,
        target_update_C=TARGET_C,
        nu_0_MY=nu_0_MY, Gamma_0_MY=Gamma_0_MY, sigma2_MY=sigma2_MY,
        nu_0_Y=nu_0_Y, Gamma_0_Y=Gamma_0_Y, sigma2_Y=sigma2_Y,
        nu_0_tilde_Y=nu_0_tilde_Y, Gamma_0_tilde_Y=Gamma_0_tilde_Y,
        sigma2_tilde_Y=sigma2_tilde_Y,
        Y_1=float(oenv.CAE_all[0]),
        rng=np.random.default_rng(seed),
    )
    # agent.update_sigma2_online = False
    dataset = EpisodeDataset(nweek)
    result = oenv.run_episode(agent, dataset)
    return result, oenv


def run_micro_query_rs_mtd(uid, seed=42, gamma_bar=0.5):
    """Micro-query agent with both reward shaping and modified TD loss."""
    cfg = EnvConfig(uid)
    nweek = cfg.nweek
    env = Env(cfg, noise="sequential")
    oenv = OnlineEnv(env, nweek=nweek, seed=seed)

    agent = MicroQueryAgent_rewardshaping_modifiedTD(
        W=nweek, J=J_PARTICLES, B=B_ENSEMBLES, epsilon_0=EPSILON_0,
        mu_0_joint=mu_0_mtd_joint, Sigma_0_joint=Sigma_0_mtd_joint,
        p_eta=p_eta_mtd_joint,
        sigma2_bottleneck=sigma2_bottleneck_mtd_joint,
        sigma2_TD=sigma2_TD_mtd_joint,
        sigma2_T=sigma2_T_mtd_joint,
        gamma_dt=_gamma_dt_micro(gamma_bar), gamma_bar=gamma_bar,
        target_update_C=TARGET_C,
        nu_0_MY=nu_0_MY, Gamma_0_MY=Gamma_0_MY, sigma2_MY=sigma2_MY,
        nu_0_Y=nu_0_Y, Gamma_0_Y=Gamma_0_Y, sigma2_Y=sigma2_Y,
        nu_0_tilde_Y=nu_0_tilde_Y, Gamma_0_tilde_Y=Gamma_0_tilde_Y,
        sigma2_tilde_Y=sigma2_tilde_Y,
        mu_0_reward=mu_0_reward, Sigma_0_reward=Sigma_0_reward,
        sigma2_reward=sigma2_reward,
        Y_1=float(oenv.CAE_all[0]),
        rng=np.random.default_rng(seed),
    )
    # agent.update_sigma2_online = False
    dataset = EpisodeDataset(nweek)
    result = oenv.run_episode(agent, dataset)
    return result, oenv


def _run_fixed_policy(agent_cls, uid, seed=42):
    """Run a fixed walking-suggestion policy (never/always/random send).

    These agents share MicroQueryAgent's query + belief-state machinery; only
    the action is fixed, so the RLSVI discount (gamma_bar) is irrelevant and a
    single configuration suffices.
    """
    cfg = EnvConfig(uid)
    nweek = cfg.nweek
    env = Env(cfg, noise="sequential")
    oenv = OnlineEnv(env, nweek=nweek, seed=seed)

    agent = agent_cls(
        W=nweek, J=J_PARTICLES, B=B_ENSEMBLES, epsilon_0=EPSILON_0,
        mu_0_rl=mu_0_micro, Sigma_0_rl=Sigma_0_micro, sigma2_rl=sigma2_rl_micro,
        gamma_dt=_gamma_dt_micro(0.0), gamma_bar=0.0,
        target_update_C=TARGET_C,
        nu_0_MY=nu_0_MY, Gamma_0_MY=Gamma_0_MY, sigma2_MY=sigma2_MY,
        nu_0_Y=nu_0_Y, Gamma_0_Y=Gamma_0_Y, sigma2_Y=sigma2_Y,
        nu_0_tilde_Y=nu_0_tilde_Y, Gamma_0_tilde_Y=Gamma_0_tilde_Y,
        sigma2_tilde_Y=sigma2_tilde_Y,
        Y_1=float(oenv.CAE_all[0]),
        rng=np.random.default_rng(seed),
    )
    dataset = EpisodeDataset(nweek)
    result = oenv.run_episode(agent, dataset)
    return result, oenv


def run_never_send(uid, seed=42):
    """Baseline: never send a walking suggestion (A = 0 in every slot)."""
    return _run_fixed_policy(NeverSendAgent, uid, seed=seed)


def run_always_send(uid, seed=42):
    """Baseline: always send a walking suggestion (A = 1 in every slot)."""
    return _run_fixed_policy(AlwaysSendAgent, uid, seed=seed)


def run_random_send(uid, seed=42):
    """Baseline: send at random with fixed propensity pi_A = 0.5."""
    return _run_fixed_policy(RandomSendAgent, uid, seed=seed)


# Algorithm registry: name -> (runner, display label).
# Every algorithm is run at both gamma_bar = 0 and gamma_bar = 0.5
# (γ̄=0 ⇒ γ_{d,t}=0 ⇒ no within-week / bottleneck bootstrapping; the agent
# is myopic w.r.t. future slots and relies only on the immediate reward
# signal — for plain micro/mtd this is the strict-myopic baseline; for the
# reward-shaping variants the per-slot shaping does all the work).
ALGORITHMS = {
    "micro_g0":     (partial(run_micro_query,        gamma_bar=0.0), "Micro-query (γ̄=0)"),
    "micro_g05":    (partial(run_micro_query,        gamma_bar=0.5), "Micro-query (γ̄=0.5)"),
    "mtd_g0":       (partial(run_micro_query_mtd,    gamma_bar=0.0), "Micro-query + MTD (γ̄=0)"),
    "mtd_g05":      (partial(run_micro_query_mtd,    gamma_bar=0.5), "Micro-query + MTD (γ̄=0.5)"),
    "rs_g0":        (partial(run_micro_query_rs,     gamma_bar=0.0), "Micro-query + RS (γ̄=0)"),
    "rs_g05":       (partial(run_micro_query_rs,     gamma_bar=0.5), "Micro-query + RS (γ̄=0.5)"),
    "rs_mtd_g0":    (partial(run_micro_query_rs_mtd, gamma_bar=0.0), "Micro-query + RS + MTD (γ̄=0)"),
    "rs_mtd_g05":   (partial(run_micro_query_rs_mtd, gamma_bar=0.5), "Micro-query + RS + MTD (γ̄=0.5)"),
    "never_send":   (run_never_send,  "Never send (A=0)"),
    "always_send":  (run_always_send, "Always send (A=1)"),
    "random_send":  (run_random_send, "Random send (π_A=0.5)"),
}


_setup_log("Runner functions defined.")

# %%
# ──────────────────────────────────────────────────────────────────
# Main experiment: run both algorithms for all users
# (guarded so ``import experiment`` does not start long jobs — use ``%run`` or
# ``python experiment.py`` for the full driver.)
# ──────────────────────────────────────────────────────────────────

def _snapshot_oenv(oenv):
    """Capture all per-episode trajectory arrays from an OnlineEnv as a dict.

    Slot-level arrays are length T = nweek * 7 * 2 (= 504 for nweek=36).
    Daily arrays are length D = nweek * 7 (= 252).
    Weekly arrays are length nweek + 1 (index 0 = pre-RL baseline).

    PF design rows are built on demand via ``get_pf_data`` from logged
    trajectories and per-day covariate logs (not from ``self.s``).
    ``self.s`` is only maintained for the generative simulation chain.
    Precomputed ``fourSC_cond`` / ``antic_cond`` / ``CAE_cond`` arrays are not
    stored — they are re-derivable from
    the realised trajectories + state.
    """
    return {
        # ── weekly ────────────────────────────────────────────────
        "CAE_all":       oenv.CAE_all.copy(),
        "pu_all":        oenv.pu_all.copy(),         # latent E_w (env truth)
        "wp_all":        oenv.wp_all.copy(),         # J_w (week-survey present)
        "CAE_short_all": oenv.CAE_short_all.copy(),
        "U1_all":        oenv.U1_all.copy(),
        "U2_all":        oenv.U2_all.copy(),
        "E_known_all":   oenv.E_known_all.copy(),    # agent-visible E_w_hat
        # ── daily ─────────────────────────────────────────────────
        "dailyAnticipatedAffectAll":              oenv.dailyAnticipatedAffectAll.copy(),       # latent
        "dailyAnticipatedAffectObsAll":          oenv.dailyAnticipatedAffectObsAll.copy(),   # NaN if survey missed
        "dailyAnticipatedAffectAgentAll":        oenv.dailyAnticipatedAffectAgentAll.copy(), # model-imputed if survey missed
        "morningFitbitWearAll":             oenv.morningFitbitWearAll.copy(),
        "dailySurveyCompleteAll":              oenv.dailySurveyCompleteAll.copy(),       # daily-survey present
        "recordedPhysicalActivityTodayAll": oenv.recordedPhysicalActivityTodayAll.copy(),
        "activityCompletedLast7DaysAll": oenv.activityCompletedLast7DaysAll.copy(),
        # ── slot-level ────────────────────────────────────────────
        "stepCountNext4HourAll":             oenv.stepCountNext4HourAll.copy(),      # latent
        "stepCountNext4HourObsAll":         oenv.stepCountNext4HourObsAll.copy(),  # NaN if Fitbit not worn
        "stepCountNext4HourAgentAll":       oenv.stepCountNext4HourAgentAll.copy(),
        "pageViewNext4HourAll":           oenv.pageViewNext4HourAll.copy(),
        "action_all":             oenv.action_all.copy(),
        "prior2HourStepCountAll":    oenv.prior2HourStepCountAll.copy(),  # latent
        "prior2HourStepCountObsAll":     oenv.prior2HourStepCountObsAll.copy(),
        "prior2HourStepCountAgentAll": oenv.prior2HourStepCountAgentAll.copy(),
        "ws_interaction_all":     oenv.ws_interaction_all.copy(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RL experiments; optionally select a single seed by index."
    )
    parser.add_argument(
        "--seed-idx",
        type=int,
        default=None,
        help=(
            "1-based seed index in [1, N_EXPERIMENTS]. If omitted and "
            "SLURM_ARRAY_TASK_ID is set, that value is used."
        ),
    )
    args = parser.parse_args()

    user_ids = np.loadtxt(PARAMS_DIR / "user_ids.txt", dtype=int)
    N_EXPERIMENTS = 100
    all_seeds = list(range(N_EXPERIMENTS))
    seed_idx = args.seed_idx
    if seed_idx is None:
        slurm_seed_idx = os.getenv("SLURM_ARRAY_TASK_ID")
        if slurm_seed_idx:
            seed_idx = int(slurm_seed_idx)
    if seed_idx is not None:
        if not (1 <= seed_idx <= N_EXPERIMENTS):
            raise ValueError(
                f"--seed-idx must be in [1, {N_EXPERIMENTS}], got {seed_idx}"
            )
        SEEDS = [all_seeds[seed_idx - 1]]
        print(
            f"Running a single seed from index {seed_idx}: "
            f"seed={SEEDS[0]} (N_EXPERIMENTS={N_EXPERIMENTS})"
        )
    else:
        SEEDS = all_seeds
        print(f"Running all seeds: {len(SEEDS)} experiments")
    n_users = 100
    # Pre-draw the participant sample for ALL experiments so that array-mode
    # (one seed per task) and full-run mode see identical populations:
    # experiment with seed s always uses row s, regardless of which seeds
    # this process actually runs.
    uid_draw_rng = np.random.default_rng(0)
    all_sampled_uids = uid_draw_rng.choice(
        user_ids, size=(N_EXPERIMENTS, n_users), replace=True
    )

    # Per-algorithm stores nested by experiment:
    #   run_uids[exp_idx]                 → array of n_users uids for that exp
    #   cae_runs[name][exp_idx][draw_idx] → CAE_all for one participant
    #   oenv_runs[name][exp_idx][draw_idx]→ full trajectory snapshot dict
    # i.e. axis 0 = experiment (one seed = one "run" of the policy on a
    # population of n_users participants), axis 1 = participant draw within
    # that experiment.
    run_uids   = []                                  # list of (n_users,) arrays
    cae_runs   = {name: [] for name in ALGORITHMS}
    piA_runs   = {name: [] for name in ALGORITHMS}
    pf_runs    = {name: [] for name in ALGORITHMS}
    oenv_runs  = {name: [] for name in ALGORITHMS}
    cae_by_uid = {name: {} for name in ALGORITHMS}   # keyed by uid (flat across exps)

    for exp_idx, seed in enumerate(SEEDS):
        sampled_uids = all_sampled_uids[seed]
        run_uids.append(np.asarray(sampled_uids, dtype=int))
        # Per-experiment sub-list, one entry per participant draw.
        for name in ALGORITHMS:
            cae_runs[name].append([])
            piA_runs[name].append([])
            pf_runs[name].append([])
            oenv_runs[name].append([])

        for draw_idx, uid in enumerate(sampled_uids):
            uid = int(uid)
            draw_seed = _episode_seed(seed, draw_idx)
            print(
                f"  Experiment {exp_idx}, draw {draw_idx + 1}/{n_users}, "
                f"user {uid}, exp_seed {seed}, draw_seed {draw_seed} ... ",
                end="", flush=True
            )

            summary_parts = []
            for name, (runner, _label) in ALGORITHMS.items():
                res, oenv = runner(uid, seed=draw_seed)
                snap = _snapshot_oenv(oenv)
                oenv_runs[name][exp_idx].append(snap)
                cae_full = snap["CAE_all"]
                cae_runs[name][exp_idx].append(cae_full)
                piA_runs[name][exp_idx].append(res["pi_A"].copy())
                pf_runs[name][exp_idx].append(res["pf"])
                cae_by_uid[name].setdefault(uid, []).append(cae_full)
                summary_parts.append(
                    f"{name}={np.nanmean(cae_full[1:]):.3f}"
                )
            print("CAE means: " + ", ".join(summary_parts))

    print("\nExperiment complete.")

    # ``run_uids_arr[exp_idx, draw_idx]`` is the participant id of that draw.
    run_uids_arr = np.stack(run_uids)            # shape (N_EXPERIMENTS, n_users)

    # Stack every snapshot field across (experiment, participant) into a
    # (N_EXPERIMENTS, n_users, …) array so the full trajectory of each
    # variable is easy to inspect / plot, e.g.
    #   trajectories["rs_g05"]["stepCountNext4HourAll"]      → (N_EXPERIMENTS, n_users, T)
    #   trajectories["rs_g05"]["dailyAnticipatedAffectObsAll"]   → (N_EXPERIMENTS, n_users, D)
    #   trajectories["rs_g05"]["pu_all"]          → (N_EXPERIMENTS, n_users, nweek + 1)
    trajectories = {
        name: {
            field: np.stack([
                np.stack([snap[field] for snap in per_exp])
                for per_exp in oenv_runs[name]
            ])
            for field in oenv_runs[name][0][0].keys()
        }
        for name in ALGORITHMS
        if oenv_runs[name] and oenv_runs[name][0]
    }

    # ──────────────────────────────────────────────────────────────────
    # Persist results to disk (timestamped folder under the parent dir)
    #
    # Default parent is ./results_vanilla/. Override with RESULTS_ROOT=<path>
    # if you want a different folder, e.g.
    #     RESULTS_ROOT=results_mtd_joint python experiment.py
    # ──────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_tag = f"seed{SEEDS[0]}" if len(SEEDS) == 1 else f"all{len(SEEDS)}"
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    slurm_suffix = ""
    if slurm_job_id:
        slurm_suffix = f"_job{slurm_job_id}"
        if slurm_task_id:
            slurm_suffix += f"_task{slurm_task_id}"
    RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "results_vanilla"))
    OUTPUT_DIR = RESULTS_ROOT / f"{ts}_{run_tag}{slurm_suffix}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {OUTPUT_DIR.resolve()}")

    # Run-level metadata (one file shared across all algorithms).
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump({
            "nweek":          NWEEK,
            "n_users":        n_users,
            "n_experiments":  len(SEEDS),
            "n_experiments_configured": N_EXPERIMENTS,
            "seeds":          list(SEEDS),
            "algorithms":     list(ALGORITHMS.keys()),
            "labels":         {n: ALGORITHMS[n][1] for n in ALGORITHMS},
            "gamma_bars":     [0.0, 0.5],
            "epsilon_0":      EPSILON_0,
            "J_particles":    J_PARTICLES,
            "B_ensembles":    B_ENSEMBLES,
            "target_update_C": TARGET_C,
        }, f, indent=2)

    np.save(OUTPUT_DIR / "run_uids.npy", run_uids_arr)

    for name in ALGORITHMS:
        # cae_runs / piA_runs as dense (N_EXP, n_users, …) arrays.
        cae_arr = np.stack([np.stack(per_exp) for per_exp in cae_runs[name]])
        piA_arr = np.stack([np.stack(per_exp) for per_exp in piA_runs[name]])

        # Single compressed npz per algorithm: cae, piA, run_uids, every snapshot field.
        np.savez_compressed(
            OUTPUT_DIR / f"{name}.npz",
            cae_runs=cae_arr,
            piA_runs=piA_arr,
            run_uids=run_uids_arr,
            **trajectories[name],
        )

        # cae_by_uid is heterogeneous (per-uid list lengths differ when a uid
        # is sampled different numbers of times across experiments). Pickle it
        # as {uid: (n_occurrences, NWEEK + 1) array}.
        cae_by_uid_stacked = {
            int(uid): np.stack(arrs)
            for uid, arrs in cae_by_uid[name].items()
        }
        with open(OUTPUT_DIR / f"{name}_cae_by_uid.pkl", "wb") as f:
            pickle.dump(cae_by_uid_stacked, f)
        with open(OUTPUT_DIR / f"{name}_pf.pkl", "wb") as f:
            pickle.dump(pf_runs[name], f)

    # %%
    # ──────────────────────────────────────────────────────────────────
    # Visualise results
    # ──────────────────────────────────────────────────────────────────

    weeks = np.arange(1, NWEEK + 1)
    rl_weeks = np.arange(2, NWEEK + 1)

    # Stack per-algorithm CAE arrays:
    #   all_cae_full[name] → (N_EXPERIMENTS, n_users, NWEEK + 1)   includes baseline
    #   all_cae[name]      → (N_EXPERIMENTS, n_users, NWEEK)       drops baseline
    all_cae_full = {
        name: np.stack([np.stack(per_exp) for per_exp in cae_runs[name]])
        for name in ALGORITHMS
    }
    all_cae = {name: arr[..., 1:] for name, arr in all_cae_full.items()}

    # Aggregate per-week stats over (experiments × participants).
    mean_cae = {name: np.nanmean(arr, axis=(0, 1)) for name, arr in all_cae.items()}
    se_cae   = {
        name: np.nanstd(arr, axis=(0, 1)) / np.sqrt(arr.shape[0] * arr.shape[1])
        for name, arr in all_cae.items()
    }
    median_cae = {name: np.nanmedian(arr, axis=(0, 1)) for name, arr in all_cae.items()}
    p25_cae    = {
        name: np.nanpercentile(arr, 25, axis=(0, 1)) for name, arr in all_cae.items()
    }
    p75_cae    = {
        name: np.nanpercentile(arr, 75, axis=(0, 1)) for name, arr in all_cae.items()
    }

    markers = {
        "micro_g0":   "o:",
        "micro_g05":  "o-",
        "mtd_g0":     "^:",
        "mtd_g05":    "^-",
        "rs_g0":      "s:",
        "rs_g05":     "s-",
        "rs_mtd_g0":  "d:",
        "rs_mtd_g05": "d-",
        "never_send":  "x-",
        "always_send": "*-",
        "random_send": "+-",
    }

    # ── 2×2 layout: mean / median / cumulative / action prob ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # (0,0) Mean weekly CAE ± SE
    ax = axes[0, 0]
    for name, (_runner, label) in ALGORITHMS.items():
        m = mean_cae[name]
        s = se_cae[name]
        ax.plot(weeks, m, markers.get(name, "o-"), label=label)
        ax.fill_between(weeks, m - s, m + s, alpha=0.15)
    ax.set_xlabel("Week")
    ax.set_ylabel("CAE")
    ax.set_title("Mean weekly CAE (± SE)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Median weekly CAE with 25th-percentile lower band (band spans p25..p75).
    ax = axes[0, 1]
    for name, (_runner, label) in ALGORITHMS.items():
        med = median_cae[name]
        p25 = p25_cae[name]
        p75 = p75_cae[name]
        line, = ax.plot(weeks, med, markers.get(name, "o-"), label=label)
        ax.plot(
            weeks, p25, markers.get(name, "o-")[0] + "--",
            color=line.get_color(), alpha=0.6, linewidth=1.0,
        )
        ax.fill_between(weeks, p25, p75, color=line.get_color(), alpha=0.12)
    ax.set_xlabel("Week")
    ax.set_ylabel("CAE")
    ax.set_title("Median weekly CAE  (solid = median, dashed = 25th pct, band = IQR)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) Cumulative mean CAE
    ax = axes[1, 0]
    cum_cae = {name: np.cumsum(mean_cae[name]) for name in ALGORITHMS}
    for name, (_runner, label) in ALGORITHMS.items():
        ax.plot(weeks, cum_cae[name], markers.get(name, "o-"), label=label)
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative CAE")
    ax.set_title("Cumulative CAE over time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (bottom-right) Action probability
    ax = axes[1, 1]
    for name, (_runner, label) in ALGORITHMS.items():
        # piA_all shape: (N_EXPERIMENTS, n_users, W, 6, 2)
        piA_all = np.stack([np.stack(per_exp) for per_exp in piA_runs[name]])
        piA_mean = np.nanmean(piA_all[:, :, 1:, :, :], axis=(0, 1, 3, 4))
        ax.plot(rl_weeks, piA_mean, markers.get(name, "o-"), label=label)
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean P(walking suggestion = 1)")
    ax.set_title("Action probability over time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "overview.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "overview.pdf", bbox_inches="tight")
    plt.show()

    # ── Summary table ──
    headers = ["Metric"] + [name for name in ALGORITHMS]
    col_w = 18
    sep_w = col_w * (len(headers))
    summary_lines = []
    summary_lines.append("=" * sep_w)
    summary_lines.append("".join(f"{h:>{col_w}}" for h in headers))
    summary_lines.append("-" * sep_w)
    summary_lines.append(
        "".join([f"{'Mean CAE (all weeks)':>{col_w}}"]
        + [f"{np.nanmean(all_cae[n]):>{col_w}.4f}" for n in ALGORITHMS])
    )
    # SE of the grand mean, clustered by user (treats each participant as
    # the unit of randomness — collapse experiments × weeks to a per-user
    # mean, then take std/sqrt(n_users) across users).
    def _se_clustered_by_user(arr):
        per_user = np.nanmean(arr, axis=(0, 2))       # shape (n_users,)
        n_eff = int(np.sum(~np.isnan(per_user)))
        if n_eff <= 1:
            return float("nan")
        return float(np.nanstd(per_user, ddof=1) / np.sqrt(n_eff))
    summary_lines.append(
        "".join([f"{'SE CAE (clustered)':>{col_w}}"]
        + [f"{_se_clustered_by_user(all_cae[n]):>{col_w}.4f}" for n in ALGORITHMS])
    )
    summary_lines.append(
        "".join([f"{'Mean CAE (week 3+)':>{col_w}}"]
        + [f"{np.nanmean(all_cae[n][..., 2:]):>{col_w}.4f}" for n in ALGORITHMS])
    )
    summary_lines.append(
        "".join([f"{'Median CAE (all wks)':>{col_w}}"]
        + [f"{np.nanmedian(all_cae[n]):>{col_w}.4f}" for n in ALGORITHMS])
    )
    summary_lines.append(
        "".join([f"{'25th pct CAE':>{col_w}}"]
        + [f"{np.nanpercentile(all_cae[n], 25):>{col_w}.4f}" for n in ALGORITHMS])
    )
    summary_lines.append(
        "".join([f"{'75th pct CAE':>{col_w}}"]
        + [f"{np.nanpercentile(all_cae[n], 75):>{col_w}.4f}" for n in ALGORITHMS])
    )
    summary_lines.append(
        "".join([f"{'Cumulative CAE (total)':>{col_w}}"]
        + [f"{cum_cae[n][-1]:>{col_w}.4f}" for n in ALGORITHMS])
    )
    summary_lines.append("=" * sep_w)

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    (OUTPUT_DIR / "summary.txt").write_text(summary_text + "\n")
    print(f"\nResults saved to {OUTPUT_DIR.resolve()}")


# import numpy as np, json, pickle
# from pathlib import Path
# RES = Path("results/20260516-...")
# cfg  = json.loads((RES/"config.json").read_text())
# uids = np.load(RES/"run_uids.npy")
# data = np.load(RES/"rs_g05.npz")            # has cae_runs, piA_runs, stepCountNext4HourAll, …
# cae_by_uid = pickle.load(open(RES/"rs_g05_cae_by_uid.pkl","rb"))
