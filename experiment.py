"""Online wrapper around ``vani_env.Env`` for RL training and evaluation.

``OnlineEnv`` steps one decision slot at a time, logs observed vs agent-visible
arrays, and builds week packets for the particle-filter agents. Sunday is
generated with no walking suggestion so weekly CAE / E_w see a full 7-day week.

This file also contains the RCT experiment CLI (train / eval loops over
agents). 
"""
import argparse
import os
import numpy as np
import numpy.random as rd
import pandas as pd
import json
import pickle
from datetime import datetime
from pathlib import Path
from functools import partial


from ewm_utils import ewma_gamma

# parameters for EWM
EWM_WINDOW = 7
EWM_MIN_VALUES = 4


def _setup_log(*args, **kwargs):
    if __name__ == "__main__" or os.getenv("EXPERIMENT_VERBOSE_IMPORT") == "1":
        print(*args, **kwargs)


def _ewm_prior_gamma_last(
    values, gamma=None, window=EWM_WINDOW, min_values=EWM_MIN_VALUES
):
    """EWM over prior ≤window values (excludes current).

    Same normalized discount as ``1_data_extraction._ewm_prior_rows``.
    ``gamma=None`` (default) derives the decay from the window length
    (:func:`ewm_utils.gamma_from_n`), so an early partial window (e.g. 4 of
    7 days) decays differently than a full one. Non-finite entries are
    imputed to 0 *before* the average (missing counts as zero, not as a
    calendar gap). Returns 0 if the window has fewer than ``min_values``
    prior observations.
    """
    w = np.asarray(values[-window:], dtype=float)
    if w.size < min_values:
        return 0.0
    w = np.where(np.isfinite(w), w, 0.0)
    return ewma_gamma(w, gamma, empty=0.0)


def _seeded_ewm(generated, base, window=EWM_WINDOW):
    """EWM over a full ``window``, left-padded with the study-entry ``base``.

    ``base`` is the first-row / first-finite EMA from ``df_fit_11week``
    (vanilla panel start), not the MRT-end value. Fitting never sees a
    cold start (scripts 1--2 already dropped burn-in). Without this pad,
    a simulated 7-day summary either returns 0 until ``EWM_MIN_VALUES``
    generated days exist, or jumps to a 1-day EWM of the first simulated
    observation. Copies of ``base`` occupy the missing prior days and are
    displaced as simulated values arrive.
    """
    base = float(base) if np.isfinite(base) else 0.0
    gen = [
        float(v)
        for v in list(generated)[-int(window):]
        if np.isfinite(v)
    ]
    nbase = max(0, int(window) - len(gen))
    return ewma_gamma([base] * nbase + gen, None, empty=base)


# parameters for rolling mean
INTERACTION_ROLLING_WINDOW = 14
ACTIVE_STATUS_ROLLING_WINDOW = 7
BASELINE_OFFSET = 1


def _rolling_mean_last(values, window=ACTIVE_STATUS_ROLLING_WINDOW, min_values=EWM_MIN_VALUES):
    if len(values) == 0:
        return 0.0
    v = np.asarray(values[-window:], dtype=float)
    v = v[np.isfinite(v)]
    if v.size < min_values:
        return 0.0
    return float(np.mean(v))


def _delivered_interaction_fraction(actions, interactions, window=14, fallback=0.0):
    """Mean interaction among delivered slots in the last ``window`` pairs.

    Non-deliveries are omitted from the denominator. If the window has no
    deliveries, return ``fallback`` (last defined fraction), not 0.
    """
    a = np.asarray(actions[-window:], dtype=float)
    i = np.asarray(interactions[-window:], dtype=float)
    if a.size == 0:
        return float(fallback)
    delivered = (a == 1.0) & np.isfinite(i)
    if not np.any(delivered):
        return float(fallback)
    return float(np.mean(i[delivered]))
# %%
from vani_env import (
    Env,
    EnvConfig,
    make_initial_state,
    slot_ema_initials_from_df_fit,
    invert_log_zscore,
    invert_zscore,
    zscore_value,
    PARAMS_DIR as DEFAULT_PARAMS_DIR,
    trim_pf_cae_prior,
)
from agents import (
    MicroQueryAgent,
    MicroQueryAgent_ModifiedTDLoss,
    MicroQueryRewardDesignAgent,
    NeverSendAgent,
    AlwaysSendAgent,
    RandomSendAgent,
)
from algorithm_helpers import (  # WeekPacket.k = RL week (0-based)
    ParticleFilterRuntime,
    WeekPacket,
    build_phi_action,
    build_phi_action_rewardshaping,
    build_daily_mediator_phi,
    build_redistribution_phi,
    build_phi_bottleneck,
    build_fourSC_features,
    build_antic_features,
    build_pf_CAE_features,
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
    TERMINAL_T,
    ACTION_BLOCK_INCLUDE_C,
    EPSILON_0,
    SOFTMAX_TAU,
    ENSEMBLE_ACTION_MODE,
    set_action_block_include_c,
)
from agents.ew_hat import (
    compute_Ew_hat_from_week,
    initial_Ew_hat_for_user,
    load_pooled_coefs,
)

SUNDAY_D_W = TERMINAL_D + 1


def resolve_params_dir(params_dir=None):
    """Resolve the parameter folder selected for this experiment run."""
    raw = (
        params_dir
        or os.getenv("ADAPR_EXPERIMENT_PARAMS_DIR")
        or DEFAULT_PARAMS_DIR
    )
    return Path(raw).expanduser().resolve()


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
# ``self.s`` is the latent simulator state used by vani_env.Env.
# 4-hour and prior-2-hour steps are not Fitbit-gated (one array each).
# Anticipated affect is still gated on daily-survey completion.
# ──────────────────────────────────────────────────────────────────

class OnlineEnv:
    def __init__(
        self, env, nweek=None, seed=None, start_dow=1,
        df_fit_11week_csv=None, df_fit_full=None, params_dir=None,
    ):
        # ``start_dow``: civil weekday index in ``{1,…,7}`` with **1 = Monday** (``df_fit`` / study).
        # ``df_fit_full``: optional pre-loaded df_fit for week-0 Ê_w from the
        # last pre-RL week. No current runner passes it (``_make_online_env``,
        # ``ste_vanilla``), so ``E_known_all[0]`` is always
        # ``initial_Ew_hat_for_user``'s default 2.0, not 0.0.
        if seed is not None:
            rd.seed(seed)

        self.env = env
        self.params_dir = resolve_params_dir(
            params_dir or getattr(self.env.cfg, "params_dir", None)
        )
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

        self.stepCountNext4HourAll = np.zeros(self.T)
        self.pageViewNext4HourAll = np.zeros(self.T)
        self.action_all = np.zeros(self.T)
        self.activitySuggestionsSentLast7DaysAll = np.full(self.T, np.nan)
        self.dailyAnticipatedAffectAll = np.zeros(self.D)            # latent draw (used by gen_CAE)
        self.dailyAnticipatedAffectObsAll = np.full(self.D, np.nan) # observed by the agent
        self.dailyAnticipatedAffectAgentAll = np.zeros(self.D)      # agent-visible (last observed if survey missed)
        self.morningFitbitWearAll = np.zeros(self.D)
        self.dailySurveyCompleteAll = np.zeros(self.D)
        # PF-side observed copy of antic: NaN on days where the daily survey was
        # not completed (``dailySurveyComplete == 0``). The PF handles NaN at the
        # mediator-likelihood level; ``get_pf_data`` also strips NaN rows from
        # the cumulative posterior-update design / response.
        self.CAE_all = np.full(self.nweek+1, np.nan)
        # Latent (noise-free) weekly CAE: E[CAE_w | realized history], i.e. the
        # gen_CAE_mean output BEFORE adding residual noise / clipping. Used for
        # cross-algorithm comparison (removes per-draw measurement noise).
        self.CAE_mean_all = np.full(self.nweek+1, np.nan)
        self.pu_all = np.full(self.nweek+1, np.nan)        # latent E_w (env truth)
        # ``wp_all[k] = J_k`` is drawn at the end of week ``k-1`` from the
        # newly created ``E_k`` (``J_k | E_k``). That is the opening
        # check-in for week k (MRT ``week_present_lastweek``). The same
        # value gates whether CAE of week ``k-1`` (at ``CAE_all[k]``) is
        # observed. Week-k mediators read ``wp_all[k]``. ``wp_all[0] = 1``
        # bootstraps week 0 (no prior Sunday).
        self.wp_all = np.full(self.nweek+1, np.nan)
        self.wp_all[0] = 1.0
        self.CAE_short_all = np.full(self.nweek+1, np.nan)
        # Per-week observed tool surveys on the *raw* integer 0..7 Likert
        # scale, drawn from gen_tool_U1/U2 (see ``vani_env`` docstring for
        # the norm <-> raw transform).
        self.U1_all = np.full(self.nweek+1, np.nan)
        self.U2_all = np.full(self.nweek+1, np.nan)

        # Per-day covariate logs for PF / RL (not read from self.s at use time).
        self.logIsWeekend = np.zeros(self.D)
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

        if df_fit_11week_csv is None:
            df_fit_11week_csv = self.params_dir / "df_fit_11week.csv"
        self.s = make_initial_state(
            df_fit_11week_csv, participant_id=self.env.cfg.userid
        )
        self._pageViewLast7DaysEma_initial = float(
            self.s.get("pageViewLast7DaysEma", 0.0)
        )
        if not np.isfinite(self._pageViewLast7DaysEma_initial):
            self._pageViewLast7DaysEma_initial = 0.0
        slot_ema = slot_ema_initials_from_df_fit(
            df_fit_11week_csv, participant_id=self.env.cfg.userid, n_slots=self.K
        )
        self._stepCountLast7DaysEma_initial_by_slot = {
            s: float(slot_ema["stepCountLast7DaysEma"][s]) for s in range(self.K)
        }
        self._prior2HourStepCountEma7d_initial_by_slot = {
            s: float(slot_ema["prior2HourStepCountEma7d"][s]) for s in range(self.K)
        }
        self._stepCountLast7DaysEma_initial = float(
            self._stepCountLast7DaysEma_initial_by_slot[0]
        )
        self._step_ema_raw_seed_by_slot = {
            s: max(
                0.0,
                invert_zscore(
                    v, self.env.cfg.ema_step_shift, self.env.cfg.ema_step_scale
                ),
            )
            for s, v in self._stepCountLast7DaysEma_initial_by_slot.items()
        }
        self._p2h_ema_raw_seed_by_slot = {
            s: max(
                0.0,
                invert_zscore(
                    v,
                    self.env.cfg.ema_prior2hour_shift,
                    self.env.cfg.ema_prior2hour_scale,
                ),
            )
            for s, v in self._prior2HourStepCountEma7d_initial_by_slot.items()
        }
        self._hist_foursc_by_slot: dict[int, list[float]] = {s: [] for s in range(self.K)}
        self._stepCountLast7DaysEma_by_slot: dict[int, float] = dict(
            self._stepCountLast7DaysEma_initial_by_slot
        )
        self.s["stepCountLast7DaysEma"] = self._stepCountLast7DaysEma_initial
        self.s["prior2HourStepCountEma7d"] = float(
            self._prior2HourStepCountEma7d_initial_by_slot[0]
        )
        self._initial_prior2hour = float(self.s.get("prior2HourStepCount", 0.0))
        lag0 = float(self.s.get("stepCountNext4HourLag1", 0.0))
        self._initial_foursc_lag1 = lag0 if np.isfinite(lag0) else 0.0
        self.s["dailyAnticipatedAffectYesterdayAgent"] = float(
            self.s.get("dailyAnticipatedAffectYesterday", 0.0)
        )

        self.prior2HourStepCountAll = np.full(self.T, np.nan)
        # Agent-visible prior-2-hour values (observed or imputed), per decision slot.
        self._hist_prior2hour_by_slot: dict[int, list[float]] = {s: [] for s in range(self.K)}
        self.ws_interaction_all = np.zeros(self.T)

        # Seed the rolling 7-day walking-suggestion interaction fraction
        # (delivered-only over the last 14 slots; last-defined if none sent).
        self._ws_interaction_initial = float(self.s.get("activitySuggestionInteractLast7Days", 0.0))
        if not np.isfinite(self._ws_interaction_initial):
            self._ws_interaction_initial = 0.0
        self._ws_interaction_last = self._ws_interaction_initial
        self._hist_ws_action: list[float] = []
        self._hist_ws_interaction: list[float] = []

        self.activityStatusTodayAll = np.zeros(self.D)

        # Seed previous-7-day morning-wearing fraction from df_fit_11week baseline.
        # Repeating 7 times preserves the baseline average at simulation start.
        self._wear7_initial = float(self.s.get("morningFitbitWearLast7Days", 0.0))
        self._hist_morning_wear = [self._wear7_initial] * 7

        self._active_days7_initial = float(self.s.get("activeDaysLast7Days", 0.0))
        self._hist_active_days = [self._active_days7_initial] * ACTIVE_STATUS_ROLLING_WINDOW

        # ── Baseline slot [0]: pre-study values ──────────────────────────────
        # ``caeAverageLastWeek`` and ``perceivedUtilityLastWeek`` are loaded
        # from df_fit_11week_csv by ``make_initial_state``; use them to seed
        # the baseline arrays.  ``pu_all[0]`` is pinned to 2.0 by design
        # (population-mean prior for perceived utility entering the study).
        self.CAE_all[0] = float(self.s["caeAverageLastWeek"])
        self.CAE_mean_all[0] = self.CAE_all[0]   # baseline has no noise term
        self._cae_baseline = float(self.CAE_all[0])   # AR-1 seed for per-particle PF
        # Latent running state keys used as AR-1 inputs in gen_CAE / gen_perceivedUtility.
        # Not in make_initial_state; seeded here from baseline values.
        self.s["caeAverage"]           = float(self.CAE_all[0])
        self.s["perceivedUtility"] = float(self.s["perceivedUtilityLastWeek"])
        self.pu_all[0] = 2.0
        self.s["perceivedUtilityLastWeek"] = 2.0   # keep env state consistent
        # CAE_short is not used in the baseline slot
        self.CAE_short_all[0] = 0
        # Week-1 Exp-tool-1/2 from the first ``df_fit_11week`` row (Monday).
        # NaN if that week's Sunday survey was unanswered — do not fill from a
        # later week. ``U1_all[0]`` / ``U2_all[0]`` are snapshot-only; week-0
        # Ê_w is 2.0 unless ``df_fit_full`` is passed (no runner does).
        self.U1_all[0] = float(self.s["expTool1"])
        self.U2_all[0] = float(self.s["expTool2"])

        # First RL weekday is d_w = 0 (Monday when ``start_dow`` == 1); lag = prior civil day.
        dayOfWeekNorm_n, dayOfWeekNorm_i = self._day_of_week_norm(0)
        self.s["dayOfWeekNorm"] = dayOfWeekNorm_n
        self.s["isWeekend"] = 1.0 if dayOfWeekNorm_i >= 6 else 0.0
        lag_n, _ = self._day_of_week_norm(-1)
        self.s["dayOfWeekNormLag1"] = lag_n

        self._yesterday_morning_WS = 0.0
        self._yesterday_afternoon_WS = 0.0

        # Latent lag s["dailyAnticipatedAffectYesterday"] advances every day
        # (seeded from df_fit dailyAnticipatedAffectYesterday_norm). Agent lag
        # s["dailyAnticipatedAffectYesterdayAgent"] is last-observation-carried-
        # forward: it stays at the last completed daily survey when the survey
        # is missed.
        self._today_fourSC = np.zeros(self.K)
        self._today_pageview = np.zeros(self.K)
        self._today_action = np.zeros(self.K)

        self._week_finalized = np.zeros(self.nweek, dtype=bool)
        self._day_started = {}
        self._Iw_per_week = np.zeros(self.nweek, dtype=int)

        # Pooled linear coefficients for the agent-visible E_w_hat approximation
        # (fitted offline by ``6_est_Ew_weights.py``).
        try:
            self._ew_coefs = load_pooled_coefs(work_dir=self.params_dir)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Ew_pooled_linear_coefs.json missing; run 6_est_Ew_weights.py first."
            ) from exc

        # Approximated E_w available to the agent at the start of each week.
        # ``E_known_all[k]`` is computed from observable aggregates of the previous
        # simulated week (``_finalize_week``). ``E_known_all[0]`` is 2.0 unless
        # ``df_fit_full`` is passed into ``OnlineEnv`` (no runner currently does).
        self.E_known_all = np.full(self.nweek + 1, np.nan)
        self.E_known_all[0] = float(
            initial_Ew_hat_for_user(
                self.env.cfg.userid,
                df_fit=df_fit_full,
                coefs=self._ew_coefs,
            )
        )

        # ``activitySuggestionsSentLast7Days``: baseline from df_fit
        # (``recent_burden_norm``); then daily sum + EWM of raw {0,1,2} counts
        # + z-score with ``recent_burden_shift`` / ``scale``.
        self._activitySuggestionsSentLast7Days_initial = float(
            self.s["activitySuggestionsSentLast7Days"]
        )
        if not np.isfinite(self._activitySuggestionsSentLast7Days_initial):
            self._activitySuggestionsSentLast7Days_initial = 0.0
        self._burden_raw_seed = invert_zscore(
            self._activitySuggestionsSentLast7Days_initial,
            self.env.cfg.recent_burden_shift,
            self.env.cfg.recent_burden_scale,
        )
        self._hist_daily_suggestions: list[float] = []

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

    def _jw_week(self, sim_w) -> int:
        """Opening ``J_w`` (``wp_all[sim_w]``), 0/1. Not gated on ``I_w``."""
        if not (0 <= sim_w < self.wp_all.size):
            return 0
        v = float(self.wp_all[sim_w])
        return int(v > 0.5) if np.isfinite(v) else 0

    def start_week(self, k, I_w):
        self._Iw_per_week[k] = I_w
        # ``E_known_all[k]`` is set in ``__init__`` (k=0) or by
        # ``_finalize_week(k-1)`` (k >= 1); see ``agents.ew_hat.compute_Ew_hat_from_week``.
        # ``wp_all[k]`` is already populated: bootstrapped to 1.0 for k == 0,
        # and drawn at the end of week k-1 from the new ``E_k`` (``J_k | E_k``)
        # for k >= 1. Daily mediators read that opening ``J_k`` (MRT
        # ``week_present_lastweek``), not ``I_w``. The same value still
        # gates observation of CAE of week ``k-1`` (stored at ``CAE_all[k]``).

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

        # End of week ``sim_w_prev`` writes CAE of that week and ``J_{sim_w_prev+1}``
        # at the same storage index (0 = baseline). ``J_k`` therefore gates
        # ``CAE_all[k]``.
        weekly_idx = self._weekly_idx(sim_w_prev)
        y_sim = self.CAE_all[weekly_idx]
        ty_sim = self.CAE_short_all[weekly_idx]
        wp = self.wp_all[weekly_idx]
        jw = 0 if (not np.isfinite(wp)) else int(float(wp))
        Y_prev = float(y_sim)
        tY_prev = float(ty_sim)

        return WeekPacket(
            k=k,
            pf_data=pf_data,
            Y_prev=Y_prev,
            tY_prev=tY_prev,
            J_w=jw,
        )

    def _ensure_day_started(self, sim_w, d_w, d_global):
        """Run morning-day setup once per simulated calendar day."""
        if (sim_w, d_w) not in self._day_started:
            self._day_started[(sim_w, d_w)] = True
            self._start_day(sim_w, d_w, d_global)

    def _generate_prior2hour_for_slot(self, sim_w, d_w, t_sim, d_global, step_idx):
        """Draw and log prior-2-hour steps for the current decision slot.

        Called before ``get_context`` / ``act`` so ``C`` carries this slot's
        own 2-hour-window covariate (not the previous slot's). Idempotent per
        ``step_idx`` so ``step_action`` can reuse the same draw.
        """
        step_idx = int(step_idx)
        if np.isfinite(self.prior2HourStepCountAll[step_idx]):
            self.s["prior2HourStepCount"] = float(self.prior2HourStepCountAll[step_idx])
            return

        slot = int(t_sim)
        self.s["decisionTimeSlot"] = float(t_sim)
        self.s["prior2HourStepCountEma7d"] = self._prior2HourStepCountEma7d_for_slot(slot)

        prior2HourStepCount = self.env.gen_prior2hour_step_count(self.s, step_idx)
        self.prior2HourStepCountAll[step_idx] = prior2HourStepCount
        self.s["prior2HourStepCount"] = prior2HourStepCount

    def _agent_antic(self, antic_latent: float, survey_present: bool):
        """Return (observed, agent-visible) anticipated affect for one day.

        Missed surveys keep the observation as NaN and show the agent the last
        completed-survey value (LOCF from ``dailyAnticipatedAffectYesterdayAgent``).
        The latent lag is a separate key and is not used here.
        """
        if survey_present:
            val = float(antic_latent)
            return val, val
        last = float(self.s.get("dailyAnticipatedAffectYesterdayAgent", 0.0))
        if not np.isfinite(last):
            last = 0.0
        return np.nan, last

    def _record_ws_interaction(self, Ah, ws_interaction):
        """Append this slot to the 14-slot delivered-only interaction window."""
        self._hist_ws_action.append(float(Ah))
        self._hist_ws_interaction.append(float(ws_interaction))
        frac = _delivered_interaction_fraction(
            self._hist_ws_action,
            self._hist_ws_interaction,
            INTERACTION_ROLLING_WINDOW,
            fallback=self._ws_interaction_last,
        )
        self.s["activitySuggestionInteractLast7Days"] = frac
        window_a = np.asarray(
            self._hist_ws_action[-INTERACTION_ROLLING_WINDOW:], dtype=float
        )
        if np.any(window_a == 1.0):
            self._ws_interaction_last = frac

    def _reset_episode_state(self):
        """Reset mutable episode histories before simulating a policy."""
        self._hist_daily_pv = []
        self.s["pageViewLast7DaysEma"] = self._pageViewLast7DaysEma_initial
        self._hist_daily_suggestions = []
        self.s["activitySuggestionsSentLast7Days"] = (
            self._activitySuggestionsSentLast7Days_initial
        )
        self._ws_interaction_last = self._ws_interaction_initial
        self._hist_ws_action = []
        self._hist_ws_interaction = []

        self.s["activitySuggestionInteractLast7Days"] = self._ws_interaction_initial

        self.s["morningFitbitWearLast7Days"] = self._wear7_initial
        self._hist_morning_wear = [self._wear7_initial] * 7
        self._hist_active_days = [self._active_days7_initial] * ACTIVE_STATUS_ROLLING_WINDOW

        self._hist_prior2hour_by_slot = {s: [] for s in range(self.K)}
        self._hist_foursc_by_slot = {s: [] for s in range(self.K)}
        self._stepCountLast7DaysEma_by_slot = dict(
            self._stepCountLast7DaysEma_initial_by_slot
        )
        self.s["stepCountLast7DaysEma"] = float(
            self._stepCountLast7DaysEma_initial_by_slot[0]
        )
        self.s["prior2HourStepCountEma7d"] = float(
            self._prior2HourStepCountEma7d_initial_by_slot[0]
        )
        p2h0 = float(self._initial_prior2hour)
        if not np.isfinite(p2h0):
            p2h0 = 0.0
        self.s["prior2HourStepCount"] = p2h0
        self.prior2HourStepCountAll[0] = p2h0
        self.s["stepCountNext4HourLag1"] = float(self._initial_foursc_lag1)

        self.logYesterdayStepCount[0] = float(self.s["yesterdayStepCount"])
        self.logPageViewLast7DaysEma[0] = float(self.s["pageViewLast7DaysEma"])
        self.logMorningFitbitWearLast7Days[0] = float(self.s.get("morningFitbitWearLast7Days", 0.0))
        self.logDailyAnticipatedAffectYesterday[0] = float(
            self.s.get("dailyAnticipatedAffectYesterdayAgent", 0.0)
        )
        self.logIsWeekend[0] = float(self.s["isWeekend"])
        self.logActiveDaysLast7Days[0] = float(self.s.get("activeDaysLast7Days", 0.0))

    def run_episode(self, agent, dataset, week0_actions=None, I_hist=None):
        if I_hist is not None:
            dataset.I_hist[:] = np.asarray(I_hist, dtype=int).reshape(-1)
        agent.reset(dataset, week0_actions=week0_actions)
        # Adaptive agents need the PF/RLSVI machinery. Agents can opt out with
        # ``needs_belief = False``; the fixed baselines use a faster dedicated
        # path in ``_run_fixed_policy_fast`` and normally do not enter here.
        needs_belief = getattr(agent, "needs_belief", True)
        pf_runtime = (
            ParticleFilterRuntime(agent, dataset, agent.rng)
            if needs_belief else None
        )
        self._reset_episode_state()

        for k in range(self.nweek):
            packet = self.get_week_packet(k)
            dataset.record_week_start(
                k, make_state(self.get_context(k, QUERY_D, QUERY_T))
            )
            I_w = agent.begin_week(k, packet)
            if needs_belief:
                pf_runtime.update_standard(k, packet, I_w)

            self.start_week(k, I_w)
            if needs_belief and hasattr(agent, "prepare_week"):
                agent.prepare_week(k)

            for d in range(N_RL_DAYS):
                # Day 0 walks under last week's beta; refit beta_k before day 1.
                if needs_belief and d == 1 and hasattr(agent, "update_rlsvi"):
                    agent.update_rlsvi(k)
                for t in range(self.K):
                    d_global = self._day_idx(k, d)
                    step_idx = self._step_idx(k, d, t)
                    self._ensure_day_started(k, d, d_global)
                    self._generate_prior2hour_for_slot(k, d, t, d_global, step_idx)
                    context = self.get_context(k, d, t)
                    state = make_state(context)
                    A_wdt, pi_A = agent.act(k, d, t, state)
                    dataset.record_walking(k, d, t, state, A_wdt, pi_A)
                    self.step_action(k, d, t, A_wdt, I_w)

            self._finalize_week(k)

            # Record Mon–Sat realized mediators for reward shaping
            # (post-action, not used as decision-time state). Sunday is
            # generated for weekly CAE / E_w but is not an RL feature.
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

        self._ensure_day_started(sim_w, d_w, d_global)

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

        # Pre-decision covariate: same slot's 2-hour window (also set before act()).
        self._generate_prior2hour_for_slot(sim_w, d_w, t_sim, d_global, step_idx)
        prior2HourStepCount = float(self.prior2HourStepCountAll[step_idx])

        fourSC = self.env.gen_fourSC(self.s, Ah, step_idx)
        pv = self.env.gen_pageview(self.s, Ah, self._jw_week(sim_w), step_idx)

        # Interaction history should not include the current slot until after
        # ``ws_interaction`` has been generated.
        ws_interaction = self.env.gen_ws_interaction(self.s, step_idx, Ah)

        self.stepCountNext4HourAll[step_idx] = fourSC
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
        self._hist_prior2hour_by_slot[slot].append(
            invert_log_zscore(
                prior2HourStepCount,
                self.env.cfg.prior2hour_log_shift,
                self.env.cfg.prior2hour_log_scale,
            )
        )
        self._hist_foursc_by_slot[slot].append(
            invert_log_zscore(
                fourSC,
                self.env.cfg.fourSC_log_shift,
                self.env.cfg.fourSC_log_scale,
            )
        )
        self._stepCountLast7DaysEma_by_slot[slot] = self._stepCountLast7DaysEma_for_slot(slot)

        # Update rolling delivered-only interaction fraction after this slot.
        self._record_ws_interaction(Ah, ws_interaction)

        if t_sim == self.K - 1:
            self._end_day(sim_w, d_w, d_global)

    def _start_day(self, sim_w, d_w, d_global):
        # Periodic calendar within each RL week: d_w ∈ {0,…,6} (Mon–Sun block), not d_global.
        # Match ``df_fit``: (dayOfWeekNorm - (1+7)/2) / ((7-1)/2)  →  dayOfWeekNorm ∈ {1,…,7}  to  [-1, 1]; 1 = Monday.
        dayOfWeekNorm_n, dayOfWeekNorm_i = self._day_of_week_norm(d_w)
        self.s["dayOfWeekNorm"] = dayOfWeekNorm_n
        self.s["isWeekend"] = 1.0 if dayOfWeekNorm_i >= 6 else 0.0
        self.s["decisionTimeSlot"] = 0.0

        active_status = self.env.gen_active_status(self.s, d_global)
        self.s["activityStatusToday"] = active_status
        self.activityStatusTodayAll[d_global] = active_status

        self.logIsWeekend[d_global] = float(self.s["isWeekend"])
        self.logActiveDaysLast7Days[d_global] = float(
            self.s.get("activeDaysLast7Days", 0.0)
        )
        self.logPageViewLast7DaysEma[d_global] = _seeded_ewm(
            self._hist_daily_pv,
            self._pageViewLast7DaysEma_initial,
        )
        self.logMorningFitbitWearLast7Days[d_global] = _rolling_mean_last(self._hist_morning_wear, 7)

        raw_burden = _seeded_ewm(
            self._hist_daily_suggestions,
            self._burden_raw_seed,
        )
        self.s["activitySuggestionsSentLast7Days"] = zscore_value(
            raw_burden,
            self.env.cfg.recent_burden_shift,
            self.env.cfg.recent_burden_scale,
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
        Jw = self._jw_week(sim_w)

        # Same-day predictors for daily mediators (before fitbit / dailysurvey / antic).
        self.s["todayStepCount"] = daily_sum_step
        self.s["pageViewMorningToday"] = float(self._today_pageview[0])
        self.s["pageViewAfternoonToday"] = float(self._today_pageview[1])

        fitbit = self.env.gen_fitbitwearing(self.s, ws_m, ws_a, Jw, d_global)
        self.s["morningFitbitWear"] = fitbit
        daily_pres = self.env.gen_dailysurvey(self.s, ws_m, ws_a, Jw, d_global)
        self.s["dailySurveyComplete"] = daily_pres
        # Latent anticipated affect — always drawn. Tomorrow's gen_fourSC_mean /
        # gen_antic_mean read s["dailyAnticipatedAffectYesterday"], which is
        # updated to this draw even if the daily survey is missing.
        antic = self.env.gen_antic(self.s, ws_m, ws_a, d_global)
        self.s["dailyAnticipatedAffect"] = antic

        self.morningFitbitWearAll[d_global] = fitbit
        self.dailySurveyCompleteAll[d_global] = daily_pres
        self.dailyAnticipatedAffectAll[d_global] = antic                                   # latent, always finite

        day_base = self._step_idx(sim_w, d_w, 0)
        daily_sum_step_agent = float(np.sum(
            self.stepCountNext4HourAll[day_base:day_base + self.K]
        ))
        self.logTodayStepCount[d_global] = daily_sum_step_agent

        survey_present = float(daily_pres) == 1.0
        antic_obs, antic_agent = self._agent_antic(antic, survey_present)
        self.dailyAnticipatedAffectObsAll[d_global] = antic_obs
        self.dailyAnticipatedAffectAgentAll[d_global] = antic_agent

        # Latent lag always advances. Agent lag is LOCF of last completed survey.
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
        self.s["pageViewLast7DaysEma"] = _seeded_ewm(
            self._hist_daily_pv,
            self._pageViewLast7DaysEma_initial,
        )

        past7_wear = _rolling_mean_last(self._hist_morning_wear, 7)
        self.s["morningFitbitWearLast7Days"] = past7_wear

        if d_global + 1 < self.D:
            self.logYesterdayStepCount[d_global + 1] = daily_sum_step_agent
            self.logDailyAnticipatedAffectYesterday[d_global + 1] = float(antic_agent)

        self._yesterday_morning_WS = self._today_action[0]
        self._yesterday_afternoon_WS = self._today_action[1]

        daily_suggestions = float(self._today_action[0] + self._today_action[1])
        self._hist_daily_suggestions.append(daily_suggestions)

        self._antic_wk[d_w] = antic
        self._dw_wk[d_w] = fitbit
        self._dp_wk[d_w] = daily_pres

        self._hist_active_days.append(float(self.s.get("activityStatusToday", 0.0)))
        self.s["activeDaysLast7Days"] = _rolling_mean_last(
            self._hist_active_days,
            ACTIVE_STATUS_ROLLING_WINDOW,
        )

    def _finalize_week(self, sim_w):
        if sim_w < 0 or sim_w >= self.nweek or self._week_finalized[sim_w]:
            return

        Jw = self._jw_week(sim_w)
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
            pv = self.env.gen_pageview(self.s, Ah, Jw, step_idx)

            self.stepCountNext4HourAll[step_idx] = fourSC
            self.pageViewNext4HourAll[step_idx] = pv
            self._today_fourSC[t_sim] = fourSC
            self._today_pageview[t_sim] = pv
            self._today_action[t_sim] = Ah
            week_slot = d_w_sun * self.K + t_sim
            self._foursc_wk[week_slot] = fourSC
            self._pw_wk[week_slot] = pv

            self.s["stepCountNext4HourLag1"] = fourSC
            self.s["pageViewNext4HourLag1"] = pv
            self._hist_foursc_by_slot[int(t_sim)].append(
                invert_log_zscore(
                    fourSC,
                    self.env.cfg.fourSC_log_shift,
                    self.env.cfg.fourSC_log_scale,
                )
            )
            self._stepCountLast7DaysEma_by_slot[int(t_sim)] = (
                self._stepCountLast7DaysEma_for_slot(int(t_sim))
            )
            # Sunday has no walking suggestion; still occupies a slot in the
            # 14-slot delivered-only interaction window.
            self._record_ws_interaction(0.0, 0.0)

        self._end_day(sim_w, d_w_sun, d_global)

        e_w = float(self.s["perceivedUtilityLastWeek"])

        # Latent (noise-free) conditional mean E[CAE | realized history]; the
        # realized ``cae`` below adds residual noise (and clipping) on top.
        cae_mean = self.env.gen_CAE_mean(
            self.s["caeAverageLastWeek"], week_norm,
            self._foursc_wk, self._antic_wk,
        )
        cae = self.env.gen_CAE(
            self.s["caeAverageLastWeek"], week_norm,
            self._foursc_wk, self._antic_wk, sim_w,
        )
        cs = self.env.gen_CAE_short(cae, sim_w)

        weekly_idx = self._weekly_idx(sim_w)
        # End-of-week boundary: first E_{w+1} from this week's mediators,
        # then J_{w+1}, U_{w+1} | E_{w+1}. Stored at weekly_idx so next
        # week's mediators read them as opening J_w. ``wp_all[sim_w]`` was
        # set at the previous Sunday and is not overwritten.
        pu = self.env.gen_perceivedUtility(
            e_w, week_norm,
            self._pw_wk, self._dw_wk, self._dp_wk, sim_w,
        )
        u1 = self.env.gen_tool_U1(pu, sim_w)
        u2 = self.env.gen_tool_U2(pu, sim_w)
        j_w = self.env.gen_week_present(pu, weekly_idx)

        self.CAE_all[weekly_idx] = cae
        self.CAE_mean_all[weekly_idx] = cae_mean
        self.pu_all[weekly_idx] = pu
        self.wp_all[weekly_idx] = j_w
        self.CAE_short_all[weekly_idx] = cs
        self.U1_all[weekly_idx] = u1
        self.U2_all[weekly_idx] = u2

        self.s["caeAverageLastWeek"] = cae
        # Latent E_w driving env generative process (PV / FW / PJ / J / fourSC / antic).
        # The agent never sees this; it gets ``E_known_all`` instead.
        self.s["perceivedUtilityLastWeek"] = pu

        # E-hat for next week: this week's PV/FW/PJ with this Sunday's J/U1/U2
        # (index ``weekly_idx = sim_w + 1``). Stored there so ``start_week(k)``
        # reads ``E_known_all[k]``.

        self.E_known_all[weekly_idx] = compute_Ew_hat_from_week(
            sim_w,
            coefs=self._ew_coefs,
            wp_all=self.wp_all,
            U1_all=self.U1_all,
            U2_all=self.U2_all,
            pageViewNext4HourAll=self.pageViewNext4HourAll,
            dw_wk=self._dw_wk,
            dp_wk=self._dp_wk,
        )

        self._week_finalized[sim_w] = True

    def _week_norm(self, sim_w):
        wk = sim_w + 1
        n = self.nweek
        return 0.0 if n <= 1 else (wk - (1.0 + n) / 2.0) / ((n - 1.0) / 2.0)

    def _stepCountLast7DaysEma_for_slot(self, slot):
        """EWM of prior ≤7 same-slot *raw* 4-hour counts, then EMA z-score.

        Matches ``1_data_extraction`` ``EMA_StepCount`` (groupby DecisionTime)
        followed by ``3_standardization`` ``EMA_StepCount_norm``. The 7-day
        window is left-padded with the study-entry EMA (first finite
        per-slot value from ``df_fit_11week``, inverted to raw counts)
        until enough simulated slots exist.
        """
        slot = int(slot)
        raw_ema = _seeded_ewm(
            self._hist_foursc_by_slot[slot],
            self._step_ema_raw_seed_by_slot[slot],
        )
        return zscore_value(
            raw_ema, self.env.cfg.ema_step_shift, self.env.cfg.ema_step_scale
        )

    def _prior2HourStepCountEma7d_for_slot(self, slot):
        """EWM of prior ≤7 same-slot *raw* prior-2-hour counts, then EMA z-score.

        Matches ``1_data_extraction`` ``EMA_Prior2HourStepCount`` (groupby
        DecisionTime) followed by ``3_standardization`` ``EMA_Prior2HourStepCount_norm``.
        The 7-day window is left-padded with the study-entry EMA (first
        finite per-slot value from ``df_fit_11week``, inverted to raw
        counts) until enough simulated slots exist.
        """
        slot = int(slot)
        raw_ema = _seeded_ewm(
            self._hist_prior2hour_by_slot[slot],
            self._p2h_ema_raw_seed_by_slot[slot],
        )
        return zscore_value(
            raw_ema,
            self.env.cfg.ema_prior2hour_shift,
            self.env.cfg.ema_prior2hour_scale,
        )

    def _pf_foursc_row(self, step_idx):
        sim_w, _d_w, t_sim, d_global = self._decode_step_idx(step_idx)
        Ah = float(self.action_all[step_idx])
        if step_idx > 0:
            prev = self.stepCountNext4HourAll[step_idx - 1]
            lag1 = float(prev) if np.isfinite(prev) else 0.0
        else:
            # Frozen MRT-start lag. Live s["stepCountNext4HourLag1"] is the
            # most recently generated fourSC, so using it would inject a
            # future outcome into week-0 Monday AM when historical PF rows
            # are rebuilt at k ≥ 2.
            lag1 = float(self._initial_foursc_lag1)
            if not np.isfinite(lag1):
                lag1 = 0.0
        return build_fourSC_features(
            yesterdayStepCount=self.logYesterdayStepCount[d_global],
            stepCountLast7DaysEma=self.logStepCountLast7DaysEma[step_idx],
            prior2HourStepCount=float(self.prior2HourStepCountAll[step_idx]),
            activitySuggestionsSentLast7Days=float(
                self.activitySuggestionsSentLast7DaysAll[step_idx]
            ),
            activitySuggestionInteractLast7Days=self.logActivitySuggestionInteractLast7Days[step_idx],
            activeDaysLast7Days=self.logActiveDaysLast7Days[d_global],
            isWeekend=self.logIsWeekend[d_global],
            decisionTimeSlot=float(t_sim),
            perceivedUtility=self.E_known_all[sim_w],
            caeAverageLastWeek=0.0,
            Ah=Ah,
            stepCountNext4HourLag1=lag1,
            cae=0.0,
        )

    def _pf_antic_row(self, d_global):
        sim_w = d_global // self.W_days
        d_w = d_global % self.W_days
        base = self._step_idx(sim_w, d_w, 0)
        ws_m = float(self.action_all[base])
        ws_a = float(self.action_all[base + 1])
        rb = float(self.activitySuggestionsSentLast7DaysAll[base])
        if not np.isfinite(rb):
            rb = float(self.s.get("activitySuggestionsSentLast7Days", 0.0))
        return build_antic_features(
            dailyAnticipatedAffectYesterday=self.logDailyAnticipatedAffectYesterday[d_global],
            activeDaysLast7Days=self.logActiveDaysLast7Days[d_global],
            isWeekend=self.logIsWeekend[d_global],
            perceivedUtility=self.E_known_all[sim_w],
            caeAverageLastWeek=0.0,
            activitySuggestionsSentLast7Days=rb,
            ws_morning=ws_m,
            ws_afternoon=ws_a,
            pu=self.E_known_all[sim_w],
            cae=0.0,
        )

    def _pf_cae_row(self, sim_w):
        """PF CAE transition row: EWMA of Mon–Sat mediators only (12 slots / 6 days).

        Anticipated affect uses the agent-visible series (LOCF on missed
        daily surveys). Current-week and historical CAE rows share this.
        """
        slot_start = self._step_idx(sim_w, 0, 0)
        slot_stop = slot_start + N_RL_DAYS * self.K
        day_start = self._day_idx(sim_w, 0)
        day_stop = day_start + N_RL_DAYS
        foursc_wk = self.stepCountNext4HourAll[slot_start:slot_stop]
        antic_wk = self.dailyAnticipatedAffectAgentAll[day_start:day_stop]
        return build_pf_CAE_features(0.0, foursc_wk, antic_wk)

    def _rl_context_vector(self, k, d, t):
        d_global = self._day_idx(k, d)
        step_idx = self._step_idx(k, d, t)
        if np.isfinite(self.prior2HourStepCountAll[step_idx]):
            p2h = float(self.prior2HourStepCountAll[step_idx])
        else:
            p2h = float(self.s.get("prior2HourStepCount", self._initial_prior2hour))
        return build_rl_context_vector(
            yesterdayStepCount=self.logYesterdayStepCount[d_global],
            prior2HourStepCount=p2h,
            activeDaysLast7Days=float(self.s["activeDaysLast7Days"]),
            activitySuggestionsSentLast7Days=float(self.s["activitySuggestionsSentLast7Days"]),
            activitySuggestionInteractLast7Days=float(
                self.s["activitySuggestionInteractLast7Days"]
            ),
        )

    def get_pf_data(self, k):
        """PF inputs for week ``k``; rows built from logs and frozen initials."""
        return build_pf_data(
            k,
            sim_w_prev=k - 1,
            nweek=self.nweek,
            W_days=self.W_days,
            stepCountNext4HourAll=self.stepCountNext4HourAll,
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
                        M_Y[dd, tt] = self.stepCountNext4HourAll[r]
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
# The PF fourSC model is a reduced version of the env's generative model: it
# keeps AR-1 lag as a main effect only, and drops pageview-EMA, anticipated
# affect, and 7-day Fitbit wear (and their action interactions). Weekend and
# AM/PM are main effects only. Dimension is taken from the PF feature builder
# rather than from the env's P_FOURSC.
P_MY_FOURSC = int(
    build_fourSC_features(
        yesterdayStepCount=0.0,
        stepCountLast7DaysEma=0.0,
        prior2HourStepCount=0.0,
        activitySuggestionsSentLast7Days=0.0,
        activitySuggestionInteractLast7Days=0.0,
        activeDaysLast7Days=0.0,
        isWeekend=0.0,
        decisionTimeSlot=0.0,
        perceivedUtility=0.0,
        caeAverageLastWeek=0.0,
        Ah=0.0,
        stepCountNext4HourLag1=0.0,
    ).shape[0]
)
# Antic PF design matches env main effects but drops A0/A1 interactions
# with weekend and 7-day active fraction. Dimension is taken from the PF
# feature builder (may differ from env P_ANTIC).
P_MY_ANTIC = int(
    build_antic_features(
        dailyAnticipatedAffectYesterday=0.0,
        activeDaysLast7Days=0.0,
        isWeekend=0.0,
        perceivedUtility=0.0,
        caeAverageLastWeek=0.0,
        activitySuggestionsSentLast7Days=0.0,
        ws_morning=0.0,
        ws_afternoon=0.0,
    ).shape[0]
)
P_CAE = int(build_pf_CAE_features(
    0.0, np.zeros(N_RL_DAYS * N_RL_SLOTS), np.zeros(N_RL_DAYS)
).shape[0])
P_TY  = int(build_CAE_short_features(0.0).shape[0])

# Compute phi dimensions directly from the algorithm's feature builders so
# the priors stay in sync with whatever is in the base / mediator / context /
# action-interaction blocks. Avoids drift when build_phi_action changes
# (e.g. mediator EWMA summaries vs flattened week grids).
_DUMMY_RL_STATE = {
    "E_w": 0.0,
    "M_Y": np.zeros(RL_MY_SHAPE),
    "M_E": np.zeros(RL_ME_SHAPE),
    "C":   np.zeros(N_RL_CONTEXT),
}
P_RL_MICRO = int(
    build_phi_action(0.0, 0.0, _DUMMY_RL_STATE, 0, 0, 0).shape[0]
)
P_RL_REWARDSHAPING = int(
    build_phi_action_rewardshaping(0.0, 0.0, _DUMMY_RL_STATE, 0, 0).shape[0]
)
P_RL_BOTTLENECK = int(
    build_phi_bottleneck(0.0, 0.0, _DUMMY_RL_STATE).shape[0]
)
P_DAILY_MEDIATOR = {
    name: int(build_daily_mediator_phi(
        0.0, 0.0, _DUMMY_RL_STATE, 0, 0, 0, name).size)
    for name in ("AA", "FW", "PJ")
}
P_REDISRIBUTION = int(build_redistribution_phi(
    0.0, 0.0, _DUMMY_RL_STATE, 0, 0, 0, np.zeros(3)).size)


def _refresh_phi_dims():
    """Recompute Q dimensions after :func:`set_action_block_include_c`."""
    global P_RL_MICRO, P_RL_REWARDSHAPING, P_RL_BOTTLENECK
    P_RL_MICRO = int(
        build_phi_action(0.0, 0.0, _DUMMY_RL_STATE, 0, 0, 0).shape[0]
    )
    P_RL_REWARDSHAPING = int(
        build_phi_action_rewardshaping(0.0, 0.0, _DUMMY_RL_STATE, 0, 0).shape[0]
    )
    P_RL_BOTTLENECK = int(
        build_phi_bottleneck(0.0, 0.0, _DUMMY_RL_STATE).shape[0]
    )

# ── RL hyperparameters (shared) ──
# Weekly discount used by micro-query agents. V1--V6 are registered at
# gamma_bar=0.9; V7 / V8 are the gamma_bar=0.5 and 0.99 sensitivities of
# the base. The per-slot matrix is built by _gamma_dt_micro (1 within
# week, gamma_bar on the terminal slot).
GAMMA_BAR = 0.9
TARGET_C     = 1
# EPSILON_0 is defined in algorithm_helpers: RLSVI clips π to [ε, 1-ε].
# The always/never baselines are hard 1 / 0 (not the clip bounds), so they
# are outside the ε-greedy policy class the learning agents live in.
# Default action probability is the ensemble fraction
# (ADAPR_ENSEMBLE_ACTION=fraction). Softmax is ADAPR_ENSEMBLE_ACTION=softmax.
J_PARTICLES  = 50
B_ENSEMBLES  = 50
DEFAULT_NWEEK = 36


def _resolve_nweek(cli_value=None) -> int:
    """CLI ``--nweek`` overrides ``NWEEK`` env, else 36."""
    if cli_value is not None:
        n = int(cli_value)
    else:
        n = int(os.getenv("NWEEK", str(DEFAULT_NWEEK)))
    if n < 1:
        raise ValueError(f"nweek must be >= 1, got {n}")
    return n


NWEEK = _resolve_nweek()
# V1/V2 bonus weight: λ = ρ · sd(b̂) / sd(ê) from agent-visible histories
# (default ρ=0.5). Override ρ with ENGAGEMENT_RHO / --engagement-rho, or
# skip scale-matching and set λ directly with ENGAGEMENT_BONUS /
# --engagement-bonus.
ENGAGEMENT_RHO = float(os.getenv("ENGAGEMENT_RHO", "0.5"))


def _optional_float_env(name):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return None
    return float(raw)


ENGAGEMENT_BONUS = _optional_float_env("ENGAGEMENT_BONUS")

# Result persistence: ``compact`` (default) skips pf.pkl and writes trajectory
# arrays in .npz only for TRAJECTORY_REFERENCE_ALGO; ``full`` writes pf.pkl
# and trajectories for every algorithm.
SAVE_MODE_FULL = "full"
SAVE_MODE_COMPACT = "compact"
TRAJECTORY_REFERENCE_ALGO = "random_send"


def _parse_save_mode(raw):
    mode = (raw or SAVE_MODE_COMPACT).strip().lower()
    if mode in (SAVE_MODE_FULL, SAVE_MODE_COMPACT):
        return mode
    raise ValueError(
        f"SAVE_MODE must be {SAVE_MODE_FULL!r} or {SAVE_MODE_COMPACT!r}, got {raw!r}"
    )


def _save_pf_pkl(save_mode):
    return save_mode == SAVE_MODE_FULL


def _save_trajectories_for_algo(save_mode, algo_name):
    return (
        save_mode == SAVE_MODE_FULL
        or algo_name == TRAJECTORY_REFERENCE_ALGO
    )


# ──────────────────────────────────────────────────────────────────
# Load priors estimated from df_fit (``<params_dir>/rl_priors.json``).
# Run ``python est_prior.py`` to (re)generate vanilla priors, or copy/regenerate
# the priors into the selected parameter folder. If the file is missing (or
# USE_ESTIMATED_PRIORS=0 is set) we fall back to zero / identity placeholders.
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


def _default_reward_redistribution_priors():
    daily = {
        name: {"mu_0": np.zeros(p), "Sigma_0": np.eye(p), "sigma2": 1.0}
        for name, p in P_DAILY_MEDIATOR.items()
    }
    stage2 = {
        name: {"mu_0": np.zeros(P_REDISRIBUTION),
               "Sigma_0": np.eye(P_REDISRIBUTION), "sigma2": 1.0}
        for name in ("v2", "v4")
    }
    return daily, stage2


def _default_variant_q_priors():
    return {
        name: {"mu_0": np.zeros(P_RL_MICRO), "Sigma_0": np.eye(P_RL_MICRO), "sigma2": 1.0}
        for name in ("g09", "g099", "v2", "v4")
    }


def _default_rl_joint_priors(p_eta, p_beta):
    """Fallback joint prior for the modified-TD-loss RLSVI agents.

    Returns ``(mu_0, Sigma_0, p_eta, sigma2_Q)``
    with mean 0, covariance ``10 * I_{p_eta+p_beta}`` (block-diagonal only
    because we have no informative prior; the agents will then learn the
    cross-terms from data), and unit joint-loss noise variance.
    """
    p = p_eta + p_beta
    return (np.zeros(p), np.eye(p), int(p_eta), 1.0)


_PRIORS_CONFIGURED_DIR = None
_priors_src = None
_LOO_PRIOR_CACHE = {}


def _load_priors(params_dir=None):
    """Try selected-dir ``rl_priors.json``; fall back to defaults on miss."""
    if os.getenv("USE_ESTIMATED_PRIORS", "0") == "0":
        return None, "USE_ESTIMATED_PRIORS=0 -> defaults"
    try:
        from est_prior import load_estimated_priors
        prior_path = resolve_params_dir(params_dir) / "rl_priors.json"
        if not prior_path.exists():
            return None, f"{prior_path} not found"
        return load_estimated_priors(prior_path), str(prior_path)
    except Exception as exc:  # noqa: BLE001 - any failure -> safe fallback
        return None, f"load_estimated_priors failed: {exc!r}"


def _configure_priors(params_dir=None, *, force=False):
    """Configure module-level PF/RL priors for the selected parameter folder."""
    global _PRIORS_CONFIGURED_DIR, _priors_src
    global nu_0_MY, Gamma_0_MY, sigma2_MY
    global nu_0_Y, Gamma_0_Y, sigma2_Y
    global nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y
    global mu_0_micro, Sigma_0_micro, sigma2_rl_micro
    global variant_q_priors
    global mu_0_reward, Sigma_0_reward, sigma2_reward
    global daily_mediator_priors, redistribution_priors
    global mu_0_mtd_joint, Sigma_0_mtd_joint, p_eta_mtd_joint
    global sigma2_Q_mtd_joint, _P_MTD_JOINT

    params_dir = resolve_params_dir(params_dir)
    if not force and _PRIORS_CONFIGURED_DIR == str(params_dir):
        return

    _priors, _priors_src = _load_priors(params_dir)
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
        mu_0_micro, Sigma_0_micro, sigma2_rl_micro = _default_rl_priors(P_RL_MICRO)
        variant_q_priors = _default_variant_q_priors()
        mu_0_reward, Sigma_0_reward, sigma2_reward = _default_rl_priors(
            P_RL_REWARDSHAPING
        )
        daily_mediator_priors, redistribution_priors = _default_reward_redistribution_priors()
        # Joint (alpha, beta) prior for the modified-TD-loss RLSVI agents.
        (mu_0_mtd_joint, Sigma_0_mtd_joint, p_eta_mtd_joint,
         sigma2_Q_mtd_joint) = _default_rl_joint_priors(
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
        # Q prior used by the base and reward-design RL variants
        # (no TD-modify variant).
        mu_0_micro      = _priors["mu_0_micro"]
        Sigma_0_micro   = _priors["Sigma_0_micro"]
        sigma2_rl_micro = _priors["sigma2_rl_micro"]
        variant_q_priors = _default_variant_q_priors()
        if "q_no_td_modify_g09" in _priors:
            variant_q_priors["g09"] = _priors["q_no_td_modify_g09"]
        if "q_no_td_modify_g099" in _priors:
            variant_q_priors["g099"] = _priors["q_no_td_modify_g099"]
        if "q_redistribution" in _priors:
            variant_q_priors.update(_priors["q_redistribution"])
        else:
            _setup_log("[priors] variant-specific Q priors missing; V2/V4 and gamma=0.9 use zero/identity fallback (rerun est_prior.py).")
        mu_0_reward     = _priors["mu_0_reward"]
        Sigma_0_reward  = _priors["Sigma_0_reward"]
        sigma2_reward   = _priors["sigma2_reward"]
        daily_mediator_priors, redistribution_priors = _default_reward_redistribution_priors()
        if "daily_mediator_priors" in _priors and "redistribution_priors" in _priors:
            daily_mediator_priors = _priors["daily_mediator_priors"]
            redistribution_priors = _priors["redistribution_priors"]
        else:
            _setup_log("[priors] V2/V4 redistribution priors missing; using zero/identity fallback (rerun est_prior.py).")
        # Joint (alpha, beta) prior for the modified-TD-loss RLSVI agents.
        # Sigma_0 is the FULL joint covariance across users (not block-diagonal).
        # Falls back to the block-diagonal default if the older rl_priors.json
        # was produced before est_prior.py started emitting q_td_modify_joint.
        if (
            "mu_0_micro_mtd_joint" in _priors
            and _priors["mu_0_micro_mtd_joint"] is not None
        ):
            mu_0_mtd_joint              = _priors["mu_0_micro_mtd_joint"]
            Sigma_0_mtd_joint           = _priors["Sigma_0_micro_mtd_joint"]
            p_eta_mtd_joint             = int(_priors["p_eta_micro_mtd_joint"])
            sigma2_Q_mtd_joint          = float(_priors.get(
                "sigma2_Q_mtd_joint",
                _priors.get("sigma2_TD_mtd_joint", 1.0),
            ))
        else:
            _setup_log("[priors] q_td_modify_joint missing from rl_priors.json -> "
                       "falling back to default joint prior (rerun est_prior.py "
                       "to regenerate).")
            (mu_0_mtd_joint, Sigma_0_mtd_joint, p_eta_mtd_joint,
             sigma2_Q_mtd_joint) = _default_rl_joint_priors(
                P_RL_BOTTLENECK, P_RL_MICRO)

    # Sanity check: every loaded prior must agree with the phi-builder dims.
    assert nu_0_MY[0].shape == (P_MY_FOURSC,), f"fourSC dim {nu_0_MY[0].shape} != {P_MY_FOURSC}"
    assert nu_0_MY[1].shape == (P_MY_ANTIC,),  f"antic dim {nu_0_MY[1].shape} != {P_MY_ANTIC}"
    assert nu_0_Y.shape       == (P_CAE,),     f"CAE dim {nu_0_Y.shape} != {P_CAE}"
    assert nu_0_tilde_Y.shape == (P_TY,),      f"CAE_short dim {nu_0_tilde_Y.shape} != {P_TY}"
    if mu_0_micro.shape != (P_RL_MICRO,):
        raise ValueError(
            f"Q prior dim {mu_0_micro.shape[0]} != phi dim {P_RL_MICRO}. "
            "If C was dropped from the action block (--no-action-c / "
            "ACTION_BLOCK_C=0), use USE_ESTIMATED_PRIORS=0 or regenerate "
            "rl_priors.json for this feature map."
        )
    assert mu_0_reward.shape      == (P_RL_REWARDSHAPING,)
    for name, prior in variant_q_priors.items():
        assert prior["mu_0"].shape == (P_RL_MICRO,), f"{name} Q mean has wrong dimension"
        assert prior["Sigma_0"].shape == (P_RL_MICRO, P_RL_MICRO), f"{name} Q covariance has wrong dimension"
    for name, p in P_DAILY_MEDIATOR.items():
        assert daily_mediator_priors[name]["mu_0"].shape == (p,)
        assert daily_mediator_priors[name]["Sigma_0"].shape == (p, p)
    for name in ("v2", "v4"):
        assert redistribution_priors[name]["mu_0"].shape == (P_REDISRIBUTION,)
        assert redistribution_priors[name]["Sigma_0"].shape == (P_REDISRIBUTION, P_REDISRIBUTION)
    _P_MTD_JOINT = P_RL_BOTTLENECK + P_RL_MICRO
    assert mu_0_mtd_joint.shape    == (_P_MTD_JOINT,), \
        f"joint mu_0 dim {mu_0_mtd_joint.shape} != ({_P_MTD_JOINT},)"
    assert Sigma_0_mtd_joint.shape == (_P_MTD_JOINT, _P_MTD_JOINT), \
        f"joint Sigma_0 shape {Sigma_0_mtd_joint.shape} != " \
        f"({_P_MTD_JOINT}, {_P_MTD_JOINT})"
    assert p_eta_mtd_joint == P_RL_BOTTLENECK, \
        f"joint p_eta {p_eta_mtd_joint} != P_RL_BOTTLENECK={P_RL_BOTTLENECK}"

    _PRIORS_CONFIGURED_DIR = str(params_dir)
    _setup_log(
        f"Priors ready:  p_rl(micro)={P_RL_MICRO}, "
        f"p_rs={P_RL_REWARDSHAPING}, p_b={P_RL_BOTTLENECK}, "
        f"p_mtd_joint={_P_MTD_JOINT}"
    )


def _ensure_priors_configured(params_dir=None):
    _configure_priors(params_dir=params_dir, force=False)


def _apply_fitted_loo_priors(fitted, held_out_uid):
    """Install an in-memory prior bundle fitted without ``held_out_uid``."""
    global _priors_src
    global nu_0_MY, Gamma_0_MY, sigma2_MY, nu_0_Y, Gamma_0_Y, sigma2_Y
    global nu_0_tilde_Y, Gamma_0_tilde_Y, sigma2_tilde_Y
    global mu_0_micro, Sigma_0_micro, sigma2_rl_micro, variant_q_priors
    global mu_0_reward, Sigma_0_reward, sigma2_reward
    global daily_mediator_priors, redistribution_priors
    global mu_0_mtd_joint, Sigma_0_mtd_joint, p_eta_mtd_joint, sigma2_Q_mtd_joint

    pf, rw, qn = fitted["pf"], fitted["reward"], fitted["q_no_td_modify"]
    nu_0_MY = [np.asarray(pf["fourSC"]["nu_0"], dtype=float),
               np.asarray(pf["antic"]["nu_0"], dtype=float)]
    Gamma_0_MY = [np.asarray(pf["fourSC"]["Gamma_0"], dtype=float),
                  np.asarray(pf["antic"]["Gamma_0"], dtype=float)]
    sigma2_MY = [float(pf["fourSC"]["sigma2"]), float(pf["antic"]["sigma2"])]
    nu_0_Y, Gamma_0_Y = trim_pf_cae_prior(
        np.asarray(pf["CAE"]["nu_0"], dtype=float),
        np.asarray(pf["CAE"]["Gamma_0"], dtype=float))
    sigma2_Y = float(pf["CAE"]["sigma2"])
    nu_0_tilde_Y = np.asarray(pf["CAE_short"]["nu_0"], dtype=float)
    Gamma_0_tilde_Y = np.asarray(pf["CAE_short"]["Gamma_0"], dtype=float)
    sigma2_tilde_Y = float(pf["CAE_short"]["sigma2"])
    mu_0_reward = np.asarray(rw["mu_0"], dtype=float)
    Sigma_0_reward = np.asarray(rw["Sigma_0"], dtype=float)
    sigma2_reward = float(rw["sigma2"])
    mu_0_micro = np.asarray(qn["mu_0"], dtype=float)
    Sigma_0_micro = np.asarray(qn["Sigma_0"], dtype=float)
    sigma2_rl_micro = float(qn["sigma2"])
    variant_q_priors = {
        "g09": fitted["q_no_td_modify_g09"],
        **fitted["q_redistribution"],
    }
    if fitted.get("q_no_td_modify_g099"):
        variant_q_priors["g099"] = fitted["q_no_td_modify_g099"]
    for prior in variant_q_priors.values():
        prior["mu_0"] = np.asarray(prior["mu_0"], dtype=float)
        prior["Sigma_0"] = np.asarray(prior["Sigma_0"], dtype=float)
        prior["sigma2"] = float(prior["sigma2"])
    daily_mediator_priors = fitted["reward_redistribution"]["daily_mediators"]
    redistribution_priors = fitted["reward_redistribution"]["redistribution"]
    for prior in list(daily_mediator_priors.values()) + list(redistribution_priors.values()):
        prior["mu_0"] = np.asarray(prior["mu_0"], dtype=float)
        prior["Sigma_0"] = np.asarray(prior["Sigma_0"], dtype=float)
        prior["sigma2"] = float(prior["sigma2"])
    joint = fitted.get("q_td_modify_joint") or {}
    mu_joint = joint.get("mu_0")
    joint_ok = (
        mu_joint is not None
        and np.asarray(mu_joint, dtype=float).size > 0
        and joint.get("Sigma_0") is not None
        and joint.get("p_eta") is not None
    )
    if joint_ok:
        mu_0_mtd_joint = np.asarray(mu_joint, dtype=float)
        Sigma_0_mtd_joint = np.asarray(joint["Sigma_0"], dtype=float)
        p_eta_mtd_joint = int(joint["p_eta"])
        sigma2_Q_mtd_joint = float(joint.get("sigma2_Q", 1.0))
    else:
        _setup_log(
            f"[priors] WARNING: LOO joint MTD prior missing or null for "
            f"held-out user {int(held_out_uid)}; using default zero/identity "
            "joint prior (not the previous user's)."
        )
        (mu_0_mtd_joint, Sigma_0_mtd_joint, p_eta_mtd_joint,
         sigma2_Q_mtd_joint) = _default_rl_joint_priors(
            P_RL_BOTTLENECK, P_RL_MICRO)
    _priors_src = f"leave-one-out fit; held out participant {int(held_out_uid)}"


def configure_leave_one_out_priors(held_out_uid, params_dir=None):
    """Load a precomputed prior bundle fitted without one participant.

    Fitting happens only through ``python est_prior.py --loo``.  The cache is
    therefore read-only and avoids repeated FQI/PF work during experiments.
    """
    params_dir = resolve_params_dir(params_dir)
    key = (str(params_dir), int(held_out_uid))
    if key not in _LOO_PRIOR_CACHE:
        path = params_dir / "loo_priors" / f"held_out_{int(held_out_uid)}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing leave-one-out prior bundle for user {held_out_uid}: {path}. "
                f"Generate it offline with ADAPR_EST_PRIOR_PARAMS_DIR={params_dir} "
                "python est_prior.py --loo"
            )
        with open(path, encoding="utf-8") as f:
            _LOO_PRIOR_CACHE[key] = json.load(f)
    _apply_fitted_loo_priors(_LOO_PRIOR_CACHE[key], held_out_uid)


_configure_priors()

# %%
# ──────────────────────────────────────────────────────────────────
# Run one participant with a given algorithm
# ──────────────────────────────────────────────────────────────────

def _gamma_dt_micro(gamma_bar):
    """Per-slot discounts: 1 within week; ``gamma_bar`` only on the terminal slot.

    Non-terminal weekday slots (Mon–Sat morning/afternoon except the last)
    use discount 1 so value flows undiscounted within the week. Only the
    terminal controlled slot ``(TERMINAL_D, TERMINAL_T)`` (Sat afternoon)
    applies the weekly discount ``gamma_bar`` into the next week. In
    particular ``gamma_bar=0.5`` is the more myopic weekly continuation
    and ``gamma_bar=0.9`` is the longer-horizon weekly continuation.
    """
    gamma_dt = np.ones((N_RL_DAYS, N_RL_SLOTS), dtype=float)
    gamma_dt[TERMINAL_D, TERMINAL_T] = float(gamma_bar)
    return gamma_dt


# Independent of agent-RNG consumption so every variant sees the same
# week-0 walking grid and the same weekly query path.
_SHARED_EXOGENOUS_SALT = 904201


def shared_episode_exogenous(seed, nweek):
    """Week-0 actions and query indicators shared across algorithm variants.

    Drawn from ``SeedSequence([seed, salt])``, not from the agent RNG, so
    Micro / RS / MTD comparisons are not confounded by different prior
    draws consuming different amounts of randomness before ``begin_week``.
    ``I_0 = I_1 = 1``; later weeks are i.i.d. Bernoulli(0.5).
    """
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), _SHARED_EXOGENOUS_SALT])
    )
    week0_actions = rng.integers(0, 2, size=(N_RL_DAYS, N_RL_SLOTS))
    I_hist = np.ones(int(nweek), dtype=int)
    if nweek > 2:
        I_hist[2:] = rng.binomial(1, 0.5, size=int(nweek) - 2)
    return week0_actions, I_hist


def _episode_seed(exp_seed: int, draw_idx: int) -> int:
    """RNG seed for one participant draw within an experiment.

    The experiment-level ``exp_seed`` fixes the uid resample and makes runs
    replicable; ``draw_idx`` gives each cohort slot its own env/agent noise
    path even when the same uid is drawn more than once.
    """
    return int(np.random.SeedSequence([exp_seed, draw_idx]).generate_state(1)[0])


def _make_online_env(uid, seed=42, params_dir=None):
    params_dir = resolve_params_dir(params_dir)
    cfg = EnvConfig(uid, params_dir=params_dir, nweek=NWEEK)
    nweek = cfg.nweek
    env = Env(cfg, noise="ar1")
    oenv = OnlineEnv(env, nweek=nweek, seed=seed, params_dir=params_dir)
    return cfg, env, oenv


def _q_prior_for_gamma(gamma_bar):
    """Q prior whose FQI discount matches ``gamma_bar`` when available.

    0.99 uses ``q_no_td_modify_g099`` after ``est_prior.py`` is regenerated;
    older LOO / shared files fall back to the 0.9 prior.
    """
    g = float(gamma_bar)
    if abs(g - 0.99) < 1e-9 and "g099" in variant_q_priors:
        return variant_q_priors["g099"]
    if g >= 0.9:
        return variant_q_priors["g09"]
    return {
        "mu_0": mu_0_micro, "Sigma_0": Sigma_0_micro, "sigma2": sigma2_rl_micro,
    }


def run_micro_query(uid, seed=42, gamma_bar=0.5, params_dir=None):
    _ensure_priors_configured(params_dir)
    cfg, env, oenv = _make_online_env(uid, seed=seed, params_dir=params_dir)
    nweek = cfg.nweek

    week0_actions, I_hist = shared_episode_exogenous(seed, nweek)
    dataset = EpisodeDataset(nweek)
    q_prior = _q_prior_for_gamma(gamma_bar)
    agent = MicroQueryAgent(
        W=nweek, J=J_PARTICLES, B=B_ENSEMBLES, epsilon_0=EPSILON_0,
        mu_0_rl=q_prior["mu_0"], Sigma_0_rl=q_prior["Sigma_0"], sigma2_rl=q_prior["sigma2"],
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

    result = oenv.run_episode(
        agent, dataset, week0_actions=week0_actions, I_hist=I_hist
    )
    return result, oenv


def run_micro_query_mtd(uid, seed=42, gamma_bar=0.5, params_dir=None):
    """Micro-query agent with modified TD loss (week-start bottleneck V_alpha)."""
    _ensure_priors_configured(params_dir)
    cfg, env, oenv = _make_online_env(uid, seed=seed, params_dir=params_dir)
    nweek = cfg.nweek

    agent = MicroQueryAgent_ModifiedTDLoss(
        W=nweek, J=J_PARTICLES, B=B_ENSEMBLES, epsilon_0=EPSILON_0,
        mu_0_joint=mu_0_mtd_joint, Sigma_0_joint=Sigma_0_mtd_joint,
        p_eta=p_eta_mtd_joint,
        sigma2_Q=sigma2_Q_mtd_joint,
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
    week0_actions, I_hist = shared_episode_exogenous(seed, nweek)
    dataset = EpisodeDataset(nweek)
    result = oenv.run_episode(
        agent, dataset, week0_actions=week0_actions, I_hist=I_hist
    )
    return result, oenv


def run_micro_query_reward_design(uid, seed=42, reward_design="v4", gamma_bar=0.5,
                                   params_dir=None, engagement_bonus=None,
                                   engagement_rho=None):
    """Run one of the protocol reward designs V1--V4.

    V1/V2 use the engagement-biased weekly target ``b̂_{w+1} + λ ê_{w+1}``,
    with ``λ = ρ · sd(b̂) / sd(ê)`` unless ``engagement_bonus`` is set.
    V3/V4 keep discounted CAE and add the potential ``F = γ̄ ê_{w+1} - ê_w``.
    V2/V4 additionally redistribute with the two-stage daily-mediator
    decomposition, then add a terminal leftover so the week sums to the
    weekly target. ``ê_{w+1}`` is used only in that target, not in Stage-2
    slot features.
    """
    _ensure_priors_configured(params_dir)
    cfg, env, oenv = _make_online_env(uid, seed=seed, params_dir=params_dir)
    nweek = cfg.nweek
    week0_actions, I_hist = shared_episode_exogenous(seed, nweek)
    dataset = EpisodeDataset(nweek)
    if engagement_rho is None:
        engagement_rho = ENGAGEMENT_RHO
    if engagement_bonus is None:
        engagement_bonus = ENGAGEMENT_BONUS
    if reward_design in variant_q_priors:
        q_prior = variant_q_priors[reward_design]
    elif float(gamma_bar) == 0.9:
        q_prior = variant_q_priors["g09"]
    else:
        q_prior = {
            "mu_0": mu_0_micro, "Sigma_0": Sigma_0_micro, "sigma2": sigma2_rl_micro}
    agent = MicroQueryRewardDesignAgent(
        W=nweek, J=J_PARTICLES, B=B_ENSEMBLES, epsilon_0=EPSILON_0,
        mu_0_rl=q_prior["mu_0"], Sigma_0_rl=q_prior["Sigma_0"], sigma2_rl=q_prior["sigma2"],
        gamma_dt=_gamma_dt_micro(gamma_bar), gamma_bar=gamma_bar,
        target_update_C=TARGET_C,
        nu_0_MY=nu_0_MY, Gamma_0_MY=Gamma_0_MY, sigma2_MY=sigma2_MY,
        nu_0_Y=nu_0_Y, Gamma_0_Y=Gamma_0_Y, sigma2_Y=sigma2_Y,
        nu_0_tilde_Y=nu_0_tilde_Y, Gamma_0_tilde_Y=Gamma_0_tilde_Y,
        sigma2_tilde_Y=sigma2_tilde_Y, Y_1=float(oenv.CAE_all[0]),
        rng=np.random.default_rng(seed), reward_design=reward_design,
        engagement_bonus=engagement_bonus, engagement_rho=engagement_rho,
        daily_mediator_priors=daily_mediator_priors,
        redistribution_prior=redistribution_priors.get(reward_design),
    )
    return oenv.run_episode(agent, dataset, week0_actions=week0_actions, I_hist=I_hist), oenv


def _fixed_policy_rng_after_legacy_reset(seed: int):
    """RNG positioned where the old fixed-policy ``MicroQueryAgent.reset`` left it.

    Fixed policies do not use RLSVI betas, but older output consumed the week-0
    bootstrap actions and initial beta draws before weekly query/action draws.
    Skipping those draws would change the simulated query path and random-send
    actions. ``Generator.multivariate_normal(..., size=B)`` consumes the same
    standard-normal stream as the call below; this keeps reproducibility without
    paying for the unused covariance transform.
    """
    rng = np.random.default_rng(seed)
    rng.integers(0, 2, size=(N_RL_DAYS, N_RL_SLOTS))
    rng.standard_normal(size=(B_ENSEMBLES, mu_0_micro.shape[0]))
    return rng


def _run_fixed_policy_fast(policy: str, uid, seed=42, params_dir=None):
    """Run fixed baselines without PF/RLSVI/state-feature bookkeeping."""
    _ensure_priors_configured(params_dir)
    cfg, env, oenv = _make_online_env(uid, seed=seed, params_dir=params_dir)
    nweek = cfg.nweek

    if policy not in {"never_send", "always_send", "random_send"}:
        raise ValueError(f"unknown fixed policy {policy!r}")

    rng = _fixed_policy_rng_after_legacy_reset(seed)
    _week0_actions, I_hist = shared_episode_exogenous(seed, nweek)
    A_hist = np.zeros((nweek, N_RL_DAYS, N_RL_SLOTS), dtype=int)
    pi_A_hist = np.full((nweek, N_RL_DAYS, N_RL_SLOTS), np.nan)
    b_hat_hist = np.full(nweek, np.nan)
    b_tilde_hist = np.full(nweek, np.nan)
    b_hat_hist[0] = float(oenv.CAE_all[0])
    b_tilde_hist[0] = 0.0

    oenv._reset_episode_state()

    for k in range(nweek):
        I_w = int(I_hist[k])
        oenv.start_week(k, I_w)

        for d in range(N_RL_DAYS):
            for t in range(N_RL_SLOTS):
                if policy == "never_send":
                    pi_A = 0.0
                    action = 0
                elif policy == "always_send":
                    pi_A = 1.0
                    action = 1
                else:
                    pi_A = 0.5
                    action = int(rng.binomial(1, pi_A))

                A_hist[k, d, t] = action
                pi_A_hist[k, d, t] = pi_A
                oenv.step_action(k, d, t, action, I_w)

        oenv._finalize_week(k)

    result = {
        "I": I_hist,
        "A": A_hist,
        "b_hat": b_hat_hist,
        "b_tilde": b_tilde_hist,
        "pi_A": pi_A_hist,
        "y_hat": None,
        "v_hat": None,
        "pf": {},
        "betas": np.empty((nweek, 0, 0), dtype=float),
    }
    return result, oenv


def _run_fixed_policy(agent_cls, uid, seed=42, params_dir=None):
    """Run a fixed walking-suggestion policy (never/always/random send)."""
    if agent_cls is NeverSendAgent:
        return _run_fixed_policy_fast("never_send", uid, seed=seed, params_dir=params_dir)
    if agent_cls is AlwaysSendAgent:
        return _run_fixed_policy_fast("always_send", uid, seed=seed, params_dir=params_dir)
    if agent_cls is RandomSendAgent:
        return _run_fixed_policy_fast("random_send", uid, seed=seed, params_dir=params_dir)
    raise TypeError(f"unsupported fixed-policy class {agent_cls!r}")


def run_never_send(uid, seed=42, params_dir=None):
    """Baseline: never send (A=0, π_A=0). Not the RLSVI clip π_A=ε."""
    return _run_fixed_policy(NeverSendAgent, uid, seed=seed, params_dir=params_dir)


def run_always_send(uid, seed=42, params_dir=None):
    """Baseline: always send (A=1, π_A=1). Not the RLSVI clip π_A=1-ε."""
    return _run_fixed_policy(AlwaysSendAgent, uid, seed=seed, params_dir=params_dir)


def run_random_send(uid, seed=42, params_dir=None):
    """Baseline: send at random with fixed propensity pi_A = 0.5."""
    return _run_fixed_policy(RandomSendAgent, uid, seed=seed, params_dir=params_dir)


# Algorithm registry: V1--V6 at γ̄=0.9, V7/V8 base-discount sensitivities,
# then the three fixed-policy baselines.
ALGORITHMS = {
    "rl_v1_base_g09": (partial(run_micro_query, gamma_bar=0.9), "RL base (γ̄=0.9)"),
    "rl_v2_mtd_g09": (partial(run_micro_query_mtd, gamma_bar=0.9), "RL + bottleneck TD (γ̄=0.9)"),
    "rl_v3_biased_weekly": (partial(run_micro_query_reward_design, reward_design="v1", gamma_bar=0.9), "RL V1: biased weekly reward (γ̄=0.9)"),
    "rl_v4_biased_redistributed": (partial(run_micro_query_reward_design, reward_design="v2", gamma_bar=0.9), "RL V2: biased redistributed reward (γ̄=0.9)"),
    "rl_v5_invariant_weekly": (partial(run_micro_query_reward_design, reward_design="v3", gamma_bar=0.9), "RL V3: return-invariant weekly reward (γ̄=0.9)"),
    "rl_v6_invariant_redistributed": (partial(run_micro_query_reward_design, reward_design="v4", gamma_bar=0.9), "RL V4: return-invariant redistributed reward (γ̄=0.9)"),
    "rl_v7_base_g05": (partial(run_micro_query, gamma_bar=0.5), "RL base (γ̄=0.5 sensitivity)"),
    "rl_v8_base_g099": (partial(run_micro_query, gamma_bar=0.99), "RL base (γ̄=0.99 sensitivity)"),
    "never_send":   (run_never_send,  "Never send (π_A=0)"),
    "always_send":  (run_always_send, "Always send (π_A=1)"),
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
        "CAE_mean_all":  oenv.CAE_mean_all.copy(),   # latent (noise-free) E[CAE | history]
        "pu_all":        oenv.pu_all.copy(),         # latent E_w (env truth)
        "wp_all":        oenv.wp_all.copy(),         # J_w (week-survey present)
        "CAE_short_all": oenv.CAE_short_all.copy(),
        "U1_all":        oenv.U1_all.copy(),
        "U2_all":        oenv.U2_all.copy(),
        "E_known_all":   oenv.E_known_all.copy(),    # agent-visible E_w_hat
        # ── daily ─────────────────────────────────────────────────
        "dailyAnticipatedAffectAll":              oenv.dailyAnticipatedAffectAll.copy(),       # latent
        "dailyAnticipatedAffectObsAll":          oenv.dailyAnticipatedAffectObsAll.copy(),   # NaN if survey missed
        "dailyAnticipatedAffectAgentAll":        oenv.dailyAnticipatedAffectAgentAll.copy(), # last observed if survey missed
        "morningFitbitWearAll":             oenv.morningFitbitWearAll.copy(),
        "dailySurveyCompleteAll":              oenv.dailySurveyCompleteAll.copy(),       # daily-survey present
        "activityStatusTodayAll":           oenv.activityStatusTodayAll.copy(),
        # ── slot-level ────────────────────────────────────────────
        "stepCountNext4HourAll":             oenv.stepCountNext4HourAll.copy(),
        "pageViewNext4HourAll":           oenv.pageViewNext4HourAll.copy(),
        "action_all":             oenv.action_all.copy(),
        "prior2HourStepCountAll":    oenv.prior2HourStepCountAll.copy(),
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
    parser.add_argument(
        "--save-mode",
        choices=[SAVE_MODE_FULL, SAVE_MODE_COMPACT],
        default=None,
        help=(
            f"{SAVE_MODE_COMPACT} (default): skip pf.pkl and save trajectory "
            f"arrays only for {TRAJECTORY_REFERENCE_ALGO}; "
            f"{SAVE_MODE_FULL}: save pf.pkl and full trajectories for all "
            "algorithms. Override with SAVE_MODE=<mode>."
        ),
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=None,
        help=(
            "Parameter folder containing user_ids.txt, params_env_<id>.json, "
            "std_params.json, df_fit_11week.csv, and Ew_pooled_linear_coefs.json. "
            "Defaults to ADAPR_EXPERIMENT_PARAMS_DIR, then env_para_vanilla."
        ),
    )
    parser.add_argument(
        "--engagement-rho",
        type=float,
        default=None,
        help=(
            "ρ in λ = ρ · sd(b̂) / sd(ê) for V1/V2 (default: ENGAGEMENT_RHO "
            "env or 0.5). Ignored if --engagement-bonus is set."
        ),
    )
    parser.add_argument(
        "--engagement-bonus",
        type=float,
        default=None,
        help=(
            "Fixed λ on ê_{w+1} in V1/V2, skipping scale-matching. "
            "Default: ENGAGEMENT_BONUS env, else scale-matched λ."
        ),
    )
    parser.add_argument(
        "--no-action-c",
        action="store_true",
        help=(
            "Drop all C features from the Q action block "
            "(A*[1, E, b̂, b̃] only). C remains in the state features. "
            "Also set by ACTION_BLOCK_C=0. Requires matching prior dim "
            "(default zero priors, or regenerated rl_priors.json)."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Parent folder for this run's timestamped output. "
            "Default: RESULTS_ROOT env, else results_vanilla."
        ),
    )
    parser.add_argument(
        "--prior-mode",
        choices=["loo", "saved"],
        default=os.getenv("ADAPR_PRIOR_MODE", "loo"),
        help=(
            "loo (default): refit a warm-start prior bundle excluding each drawn "
            "participant; saved: use the folder's shared rl_priors.json."
        ),
    )
    parser.add_argument(
        "--nweek",
        type=int,
        default=None,
        help=(
            "Simulated RL weeks (default: NWEEK env or 36). Also passed to "
            "EnvConfig so the env horizon matches config.json."
        ),
    )
    args = parser.parse_args()
    NWEEK = _resolve_nweek(args.nweek)

    include_action_c = ACTION_BLOCK_INCLUDE_C and (not args.no_action_c)
    set_action_block_include_c(include_action_c)
    _refresh_phi_dims()
    print(
        f"Q action block C={'on' if include_action_c else 'off'} "
        f"(p_rl_micro={P_RL_MICRO})"
    )

    params_dir = resolve_params_dir(args.params_dir)
    _configure_priors(params_dir=params_dir, force=True)
    print(f"Using parameter directory: {params_dir}")
    print(f"Prior mode: {args.prior_mode}")
    print(f"nweek={NWEEK}")

    if args.engagement_rho is not None:
        ENGAGEMENT_RHO = float(args.engagement_rho)
    if args.engagement_bonus is not None:
        ENGAGEMENT_BONUS = float(args.engagement_bonus)
    if ENGAGEMENT_BONUS is not None:
        print(f"V1/V2 λ is fixed: ENGAGEMENT_BONUS={ENGAGEMENT_BONUS}")
    else:
        print(f"V1/V2 λ = ρ · sd(b̂)/sd(ê) with ENGAGEMENT_RHO={ENGAGEMENT_RHO}")

    save_mode = _parse_save_mode(args.save_mode or os.getenv("SAVE_MODE"))
    save_pf = _save_pf_pkl(save_mode)
    traj_msg = (
        "trajectories=all algos"
        if save_mode == SAVE_MODE_FULL
        else f"trajectories={TRAJECTORY_REFERENCE_ALGO} only"
    )
    print(
        f"SAVE_MODE={save_mode} "
        f"(pf.pkl={'yes' if save_pf else 'no'}, {traj_msg})"
    )

    user_ids = np.loadtxt(params_dir / "user_ids.txt", dtype=int)
    N_EXPERIMENTS = 500
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
    n_users = 75
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
    cae_mean_runs = {name: [] for name in ALGORITHMS}  # latent (noise-free) CAE
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
            cae_mean_runs[name].append([])
            piA_runs[name].append([])
            pf_runs[name].append([])
            oenv_runs[name].append([])

        for draw_idx, uid in enumerate(sampled_uids):
            uid = int(uid)
            if args.prior_mode == "loo":
                configure_leave_one_out_priors(uid, params_dir=params_dir)
            draw_seed = _episode_seed(seed, draw_idx)
            print(
                f"  Experiment {exp_idx}, draw {draw_idx + 1}/{n_users}, "
                f"user {uid}, exp_seed {seed}, draw_seed {draw_seed} ... ",
                end="", flush=True
            )

            summary_parts = []
            for name, (runner, _label) in ALGORITHMS.items():
                res, oenv = runner(uid, seed=draw_seed, params_dir=params_dir)
                snap = _snapshot_oenv(oenv)
                if _save_trajectories_for_algo(save_mode, name):
                    oenv_runs[name][exp_idx].append(snap)
                cae_full = snap["CAE_all"]
                cae_runs[name][exp_idx].append(cae_full)
                cae_mean_runs[name][exp_idx].append(snap["CAE_mean_all"])
                piA_runs[name][exp_idx].append(res["pi_A"].copy())
                if save_pf:
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
        if _save_trajectories_for_algo(save_mode, name)
        and oenv_runs[name] and oenv_runs[name][0]
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
    RESULTS_ROOT = Path(
        args.results_root
        if args.results_root is not None
        else os.getenv("RESULTS_ROOT", "results_vanilla")
    )
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
            "gamma_bars":     [0.5, 0.9],
            "engagement_rho": ENGAGEMENT_RHO,
            "engagement_bonus": ENGAGEMENT_BONUS,
            "epsilon_0":      EPSILON_0,
            "ensemble_action": ENSEMBLE_ACTION_MODE,
            "softmax_tau":    SOFTMAX_TAU,
            "J_particles":    J_PARTICLES,
            "B_ensembles":    B_ENSEMBLES,
            "target_update_C": TARGET_C,
            "save_mode":      save_mode,
            "trajectory_reference_algo": TRAJECTORY_REFERENCE_ALGO,
            "params_dir":      str(params_dir),
            "priors_source":   _priors_src,
            "prior_mode":      args.prior_mode,
            "loo_prior_cache_size": len(_LOO_PRIOR_CACHE) if args.prior_mode == "loo" else 0,
            "action_block_include_c": include_action_c,
            "p_rl_micro":      P_RL_MICRO,
            "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
            "slurm_job_id":       os.getenv("SLURM_JOB_ID"),
        }, f, indent=2)

    np.save(OUTPUT_DIR / "run_uids.npy", run_uids_arr)

    for name in ALGORITHMS:
        # cae_runs / piA_runs as dense (N_EXP, n_users, …) arrays.
        cae_arr = np.stack([np.stack(per_exp) for per_exp in cae_runs[name]])
        cae_mean_arr = np.stack([np.stack(per_exp) for per_exp in cae_mean_runs[name]])
        piA_arr = np.stack([np.stack(per_exp) for per_exp in piA_runs[name]])

        # Single compressed npz per algorithm: cae (realized), cae_mean (latent),
        # piA, run_uids; trajectory snapshot fields only when save_mode is full
        # or name is the reference algo.
        npz_payload = {
            "cae_runs": cae_arr,
            "cae_mean_runs": cae_mean_arr,
            "piA_runs": piA_arr,
            "run_uids": run_uids_arr,
        }
        if name in trajectories:
            npz_payload.update(trajectories[name])
        np.savez_compressed(OUTPUT_DIR / f"{name}.npz", **npz_payload)

        # cae_by_uid is heterogeneous (per-uid list lengths differ when a uid
        # is sampled different numbers of times across experiments). Pickle it
        # as {uid: (n_occurrences, NWEEK + 1) array}.
        cae_by_uid_stacked = {
            int(uid): np.stack(arrs)
            for uid, arrs in cae_by_uid[name].items()
        }
        with open(OUTPUT_DIR / f"{name}_cae_by_uid.pkl", "wb") as f:
            pickle.dump(cae_by_uid_stacked, f)
        if save_pf:
            with open(OUTPUT_DIR / f"{name}_pf.pkl", "wb") as f:
                pickle.dump(pf_runs[name], f)

    print(f"\nResults saved to {OUTPUT_DIR.resolve()}")
    print("Plots and pooled summaries: python aggregate.py --results-root "
          f"{RESULTS_ROOT}")
