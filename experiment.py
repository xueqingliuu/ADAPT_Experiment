# %%
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
import importlib.util
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from functools import partial


def _ewm_last(values, span=7):
    """Last point of pandas ewm(span=..., adjust=True).mean() — matches df_fit rolling."""
    return pd.Series(values, dtype=float).ewm(span=span, adjust=True).mean().iloc[-1]


# ``1_data_combine.ipynb``: mean(WalkingSuggestion, yesterday_salience_message)
# then ewm(span=14, adjust=False) over decision rows.
RECENT_BURDEN_EWM_SPAN = 14


def _ewm_recent_burden_last(values):
    """Last point of ewm(span=14, adjust=False) on instant-burden series (same as data pipeline)."""
    if not values:
        return float("nan")
    return float(
        pd.Series(values, dtype=float).ewm(
            span=RECENT_BURDEN_EWM_SPAN, adjust=False, min_periods=1
        ).mean().iloc[-1]
    )


# %%
# Converted from ``4.1_gen_vanilla_testbed.ipynb`` / ``RlAlgorithm.ipynb``
from vani_env import Env, EnvConfig, make_initial_state, PARAMS_DIR, P_FOURSC
from algorithm import (  # WeekPacket.k = RL week (0-based)
    MicroQueryAgent,
    MicroQueryAgent_rewardshaping,
    MicroQueryAgent_ModifiedTDLoss,
    MicroQueryAgent_rewardshaping_modifiedTD,
    WeekPacket,
    build_phi_action,
    build_phi_action_rewardshaping,
    build_phi_bottleneck,
)
_EW_WEIGHTS_MODULE_PATH = Path(__file__).with_name("6_est_Ew_weights.py")
_ew_weights_spec = importlib.util.spec_from_file_location(
    "est_Ew_weights", _EW_WEIGHTS_MODULE_PATH
)
if _ew_weights_spec is None or _ew_weights_spec.loader is None:
    raise ImportError(f"Could not load module from {_EW_WEIGHTS_MODULE_PATH}")
_ew_weights_module = importlib.util.module_from_spec(_ew_weights_spec)
_ew_weights_spec.loader.exec_module(_ew_weights_module)

apply_pooled_coefs = _ew_weights_module.apply_pooled_coefs
initial_Ew_hat_for_user = _ew_weights_module.initial_Ew_hat_for_user
load_pooled_coefs = _ew_weights_module.load_pooled_coefs

# %%
# ──────────────────────────────────────────────────────────────────
# Feature extraction helpers – replicate the feature vectors that
# Env.gen_*_mean methods construct internally, so we can build the
# design matrices the particle filter needs.
# ──────────────────────────────────────────────────────────────────

def build_fourSC_features(s, Ah, pu=None, cae=None):
    """Return the feature vector used by ``Env.gen_fourSC_mean`` (length ``P_FOURSC``).

    ``pu`` and ``cae`` override ``s["perceived_utility_lastweek"]`` and
    ``s["CAE_avg_lastweek"]``.  Pass ``cae=0.0`` when storing base features
    for the particle filter so that the per-particle CAE can be substituted
    later via the delta vector.
    """
    dow_n = float(s["dow"])
    wear7 = float(s.get("past7days_morning_wearing",))
    y_sal = float(s["yesterday_salience_message"])
    _pu  = float(s["perceived_utility_lastweek"]) if pu  is None else float(pu)
    _cae = float(s["CAE_avg_lastweek"])            if cae is None else float(cae)
    return np.array([
        1.0, s["fourSC_lag1"], s["yesterday_step"], s["seven_day_step_count_avg"],
        s["prior2hour_step"], s["Previous7DaysRPA"], s["recent_burden"],
        s["seven_day_pageview"], wear7,
        y_sal,
        s["Interacted_7d_walk"], s["Interacted_7d_salience"],
        s["anticipated_affect_yesterday"],
        dow_n, s["decision_time"], _pu, _cae,
        Ah,
        Ah * s["yesterday_step"],       Ah * s["prior2hour_step"],
        Ah * s["recent_burden"],        Ah * s["seven_day_pageview"],
        Ah * wear7, Ah * y_sal,
        Ah * s["Interacted_7d_walk"],
        Ah * s["Interacted_7d_salience"], Ah * s["anticipated_affect_yesterday"],
        Ah * dow_n, Ah * s["decision_time"], Ah * _pu, Ah * _cae,
    ], dtype=float)


def build_rl_context_vector(s):
    """
    Context C for RLSVI: main-effect predictors from the fourSC design,
    excluding intercept, action A, their interactions, and any quantity
    already carried separately in the state dict:

    * ``E_w`` (perceived utility) is passed as ``state["E_w"]`` and enters
      ``build_phi_action`` / ``build_phi_action_query`` directly in the base
      block and every action-interaction block — it must NOT also be in C.
    * ``CAE_avg_lastweek`` is the env's latent CAE; the agent's belief about
      CAE is ``b_hat`` / ``b_tilde`` from the particle filter, also passed
      separately in ``build_phi_action`` — it must NOT also be in C.

    Resulting length: 10.
    """
    dow_n = float(s["dow"])
    # wear7 = float(s.get("past7days_morning_wearing", s["past7days_daywearing"]))
    y_sal = float(s["yesterday_salience_message"])
    return np.array([
        s["yesterday_step"], s["seven_day_step_count_avg"],
        s["prior2hour_step"], s["Previous7DaysRPA"], 
        s["recent_burden"],
        s["seven_day_pageview"], 
        y_sal,
        s["Interacted_7d_walk"], 
        s["Interacted_7d_salience"],
        s["anticipated_affect_yesterday"],
    ], dtype=float)


# RL mediator layout: 2 slot columns + day-level tail (must match algorithm masking)
RL_MY_SHAPE = (6, 3)   # fourSC_m, fourSC_a, anticip (daily)
RL_ME_SHAPE = (6, 4)   # pageview_m, pageview_a, fitbit wearing, daily_present
N_RL_CONTEXT = int(build_rl_context_vector(make_initial_state(participant_id=118)).size)


def build_antic_features(s, ws_morning, ws_afternoon, pu=None, cae=None):
    """Return the feature vector for ``gen_antic_mean`` / ``gen_antic`` (length ``P_ANTIC``).

    ``pu`` and ``cae`` override ``s["perceived_utility_lastweek"]`` and
    ``s["CAE_avg_lastweek"]``.  Pass ``cae=0.0`` when storing base features
    for the particle filter.
    """
    ys = float(s["today_step"])
    rpa = float(s["recorded_physical_activity"])
    sal = float(s["salience_message"])
    dow = float(s["dow"])
    _pu  = float(s["perceived_utility_lastweek"]) if pu  is None else float(pu)
    _cae = float(s["CAE_avg_lastweek"])            if cae is None else float(cae)
    return np.array([
        1.0, float(s["anticipated_affect_yesterday"]), ys, rpa, sal, dow, _pu, _cae,
        ws_morning, ws_afternoon,
        ws_morning * ys, ws_afternoon * ys,
        ws_morning * rpa, ws_afternoon * rpa,
        ws_morning * sal, ws_afternoon * sal,
        ws_morning * dow, ws_afternoon * dow,
        ws_morning * _pu, ws_afternoon * _pu,
        ws_morning * _cae, ws_afternoon * _cae,
    ], dtype=float)


P_ANTIC = int(build_antic_features(make_initial_state(participant_id=118), 0.0, 0.0).shape[0])

# Column indices for CAE-dependent terms in each feature vector.
# Used by get_pf_data to build per-particle delta vectors and by algorithm.py
# to substitute particle-specific CAE values at runtime.
_FOURSC_CAE_COL     = 16   # cae main effect in fourSC  (P_FOURSC=31)
_FOURSC_AH_COL      = 17   # action Ah                  (used to compute delta for Ah*cae)
_FOURSC_AH_CAE_COL  = 30   # Ah * cae interaction
_ANTIC_CAE_COL      = 7    # cae main effect in antic   (P_ANTIC=22)
_ANTIC_WS_M_COL     = 8    # ws_morning                 (used to compute ws_m*cae delta)
_ANTIC_WS_A_COL     = 9    # ws_afternoon
_ANTIC_CAE_WS_M_COL = 20   # ws_morning  * cae
_ANTIC_CAE_WS_A_COL = 21   # ws_afternoon * cae
_CAE_AR1_COL        = 1    # CAE_lastweek (AR-1) in the 24-dim CAE feature vector

# PF mediator indices: 12 walking slots (fourSC) + 6 weekdays (antic, Mon–Sat)
N_MED_SLOT = 12
N_MED_ANTIC_DAY = 6
N_MED = N_MED_SLOT + N_MED_ANTIC_DAY

# fourSC / pageview flat storage per sim week: 7×2 = 14 (includes Sunday for CAE).
# RL state and PF fourSC rows use only Mon–Sat → N_MED_SLOT == 12 positions.
FOURSC_SLOTS_PER_WEEK = 7 * 2


def build_CAE_features(CAE_lastweek, week_norm, foursc_wk, antic_wk):
    """Return the (24,) feature vector used by gen_CAE_mean."""
    return np.concatenate([
        [1.0, CAE_lastweek, week_norm],
        foursc_wk.ravel(),   # 14
        antic_wk.ravel(),    #  7
    ])


def build_CAE_short_features(CAE_avg):
    """Return the (2,) feature vector used by gen_CAE_short_mean."""
    return np.array([1.0, CAE_avg])


print(f"fourSC features dim: {build_fourSC_features(make_initial_state(participant_id=118), 0).shape[0]} (expect {P_FOURSC})")
print(f"CAE features dim:    {build_CAE_features(0., 0., np.zeros(FOURSC_SLOTS_PER_WEEK), np.zeros(7)).shape[0]}")
print(f"CAE_short feat dim:  {build_CAE_short_features(0.).shape[0]}")


INTERACTION_ROLLING_WINDOW = 14
RPA_ROLLING_WINDOW = 7

def _rolling_mean_last(values, window=RPA_ROLLING_WINDOW):
    if len(values) == 0:
        return 0.0
    v = np.asarray(values[-window:], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    return float(np.mean(v))
# %%
# ──────────────────────────────────────────────────────────────────
# OnlineEnv: adapter that wraps Env for slot-by-slot interaction
# with the RL algorithms.
#
# Indexing convention
# -------------------
# RL weeks are 0-based:  k = 0 .. nweek-1  (same index as simulation week sim_w)
# Days within a week:    d = 1..6 (RL-controlled), day 7 = Sunday
# Slots per day:         t = 1 or 2
# Step index:            step_idx = k * FOURSC_SLOTS_PER_WEEK + (d-1)*2 + (t-1)
# RL walking uses d=1..6 only (12 slots); Sunday slots feed CAE via _finalize_week.
#
# get_week_packet(k) finalises sim week k-1 and builds PF data for week k (k >= 1).
# reward_fn(k) returns CAE_all[k] (surrogate reward tied to completed sim week k).
#
# State dict ``self.s`` (beyond ``Env.gen_*`` outputs)
# ----------------------------------------------------
# Updated each slot in ``step_action``: ``fourSC_lag1``, ``hourly_pageview_lag1``,
# ``decision_time``, ``recent_burden`` (EWM of instant burden; see below).
# Each ``_start_day`` (morning of each RL day): ``dow``, ``is_weekend``,
# ``salience_message``.
# Each ``_end_day``: ``today_step``, ``today_pageview_*``, ``fitbitwearing_morning``,
# ``daily_present``, ``anticipated_affect``, ``anticipated_affect_yesterday``,
# ``yesterday_step``, ``yesterday_fitbitwearing_morning``, ``yesterday_present``,
# ``yesterday_pageview_*``, ``dow_lag1``, ``seven_day_*``, ``past7days_daywearing``,
# ``yesterday_salience_message`` (lagged copy of today’s).
# Each ``_finalize_week`` (Sunday): ``CAE_avg_lastweek``, ``perceived_utility_lastweek``.
# ``recent_burden``: initial from ``make_initial_state`` (first CSV row for that ``userid``); then
# EWM(span=14, adjust=False) over per-slot instant burden each ``step_action``; reset at ``run_episode`` start.
#
# **Still initial / CSV-only unless you add logic:** ``prior2hour_step`` (commented stub),
# ``recorded_physical_activity``, ``Previous7DaysRPA``, ``Interacted_7d_walk``,
# ``Interacted_7d_salience`` — these stay at ``make_initial_state`` values for the whole run.
# ──────────────────────────────────────────────────────────────────

class OnlineEnv:
    def __init__(
        self, env, nweek=None, seed=None, start_dow=1,
        df_fit_11week_csv=None, df_fit_full=None,
    ):
        # ``start_dow``: civil weekday index in ``{1,…,7}`` with **1 = Monday** (``df_fit`` / study).
        # ``df_fit_full``: optional pre-loaded df_fit DataFrame (multi-week aggregates).
        # If ``None`` we lazily load via ``est_Ew_weights.load_df_fit()`` only for the
        # week-0 E_w_hat bootstrap.
        if seed is not None:
            rd.seed(seed)

        self.env = env
        self.K = env.K
        self.W_days = env.W
        self.nweek = nweek if nweek is not None else env.cfg.nweek
        self.D = self.nweek * self.W_days
        self.T = self.D * self.K
        self.start_dow = start_dow

        self.fourSC_all = np.zeros(self.T)
        self.pageview_all = np.zeros(self.T)
        self.action_all = np.zeros(self.T)
        self.antic_all = np.zeros(self.D)            # latent draw (used by gen_CAE)
        self.fitbit_all = np.zeros(self.D)
        self.daily_all = np.zeros(self.D)
        # PF-side observed copy of antic: NaN on days where the daily survey was
        # not completed (``daily_present == 0``). The PF handles NaN at the
        # mediator-likelihood level; ``get_pf_data`` also strips NaN rows from
        # the cumulative posterior-update design / response.
        self.antic_obs_all = np.full(self.D, np.nan)
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

        self.fourSC_cond = np.zeros((self.T, P_FOURSC))
        self.antic_cond = np.zeros((self.D, P_ANTIC))
        self.CAE_cond = np.zeros((self.nweek, 24))
        self.CAE_short_cond = np.zeros((self.nweek, 2))

        self._foursc_wk = np.zeros(FOURSC_SLOTS_PER_WEEK)
        self._pw_wk = np.zeros(FOURSC_SLOTS_PER_WEEK)
        self._dw_wk = np.zeros(7)
        self._dp_wk = np.zeros(7)
        self._antic_wk = np.zeros(7)

        self._ewm_span = 7
        self._hist_daily_step = []
        self._hist_daily_pv = []

        self.s = make_initial_state(
            df_fit_11week_csv, participant_id=self.env.cfg.userid
        )

        self.prior2hour_step_all = np.zeros(self.T)
        self.ws_interaction_all = np.zeros(self.T)
        self.salience_interaction_all = np.zeros(self.T)

        # Seed the rolling histories with the pre-RL 7-day fractions.
        # Repeating the fraction 14 times makes the initial rolling mean equal to the CSV baseline.
        self._ws_interaction_initial = float(self.s.get("Interacted_7d_walk", 0.0))
        self._salience_interaction_initial = float(self.s.get("Interacted_7d_salience", 0.0))

        self._hist_ws_interaction = (
            [self._ws_interaction_initial] * INTERACTION_ROLLING_WINDOW
        )
        self._hist_salience_interaction = (
            [self._salience_interaction_initial] * INTERACTION_ROLLING_WINDOW
        )

        self.recorded_physical_activity_all = np.zeros(self.D)
        self.Previous7DaysRPA_all = np.zeros(self.D)

        # Seed previous-7-day RPA from df_fit_11week baseline.
        # Repeating preserves the baseline average at simulation start.
        self._rpa7_initial = float(self.s.get("Previous7DaysRPA", 0.0))
        self._hist_recorded_physical_activity = [self._rpa7_initial] * RPA_ROLLING_WINDOW

        # Seed previous-7-day morning-wearing fraction from df_fit_11week baseline.
        # Repeating 7 times preserves the baseline average at simulation start.
        self._wear7_initial = float(self.s.get("past7days_morning_wearing", 0.0))
        self._hist_morning_wear = [self._wear7_initial] * 7

        # ── Baseline slot [0]: pre-study values ──────────────────────────────
        # ``CAE_avg_lastweek`` and ``perceived_utility_lastweek`` are loaded
        # from df_fit_11week_csv by ``make_initial_state``; use them to seed
        # the baseline arrays.  ``pu_all[0]`` is pinned to 2.0 by design
        # (population-mean prior for perceived utility entering the study).
        self.CAE_all[0] = float(self.s["CAE_avg_lastweek"])
        self._cae_baseline = float(self.CAE_all[0])   # AR-1 seed for per-particle PF
        # Latent running state keys used as AR-1 inputs in gen_CAE / gen_perceived_utility.
        # Not in make_initial_state; seeded here from baseline values.
        self.s["CAE_avg"]           = float(self.CAE_all[0])
        self.s["perceived_utility"] = float(self.s["perceived_utility_lastweek"])
        self.pu_all[0] = 2.0
        self.s["perceived_utility_lastweek"] = 2.0   # keep env state consistent
        # CAE_short is not used in the baseline slot
        self.CAE_short_all[0] = 0
        # U1_all[0] / U2_all[0]: pre-study tool surveys not in df_fit_11week_csv;
        # left as NaN (baseline slot is never read by _compute_Ew_hat_from_week,
        # which indexes at sim_w+1 for simulated weeks).
        # they are not used in the baseline slot
        self.U1_all[0] = np.nan
        self.U2_all[0] = np.nan

        # First RL weekday is d_w = 0 (Monday when ``start_dow`` == 1); lag = prior civil day.
        sd, d_w = self.start_dow, 0
        dow_i = int(((sd - 1 + d_w) % 7) + 1)
        dow_n = (dow_i - (1.0 + 7.0) / 2.0) / ((7.0 - 1.0) / 2.0)
        self.s["dow"] = dow_n
        self.s["is_weekend"] = 1.0 if dow_i >= 6 else 0.0
        dow_lag_i = int(((sd - 1 + d_w - 1) % 7) + 1)
        lag_n = (dow_lag_i - (1.0 + 7.0) / 2.0) / ((7.0 - 1.0) / 2.0)
        self.s["dow_lag1"] = lag_n
        self.s["salience_message"] = float(rd.binomial(1, 0.5))

        self._yesterday_morning_WS = 0.0
        self._yesterday_afternoon_WS = 0.0

        # s["anticipated_affect_yesterday"] is the carry-forward of the last
        # *observed* anticipated affect (daily_present==1 days only).  It is
        # seeded here from the df_fit baseline row (anticipated_affect_yesterday_norm)
        # and updated in _end_day on survey-present days.  It is also used
        # directly in get_state for the M_Y carry-forward.

        self._today_fourSC = np.zeros(self.K)
        self._today_pageview = np.zeros(self.K)
        self._today_action = np.zeros(self.K)

        self._week_finalized = np.zeros(self.nweek, dtype=bool)
        self._day_started = {}
        self._Iw_per_week = np.zeros(self.nweek, dtype=int)

        # Pooled linear coefficients for the agent-visible E_w_hat approximation
        # (``est_Ew_weights.fit_pooled_linear_Ew``).  Saved by running
        # ``python est_Ew_weights.py`` after ``perceived_utility.py``.
        try:
            self._ew_coefs = load_pooled_coefs()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Ew_pooled_linear_coefs.json missing; run est_Ew_weights.py first."
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

        # ``recent_burden``: initial value from ``make_initial_state`` / CSV; then EWM on the fly.
        self._recent_burden_initial = float(self.s["recent_burden"])
        self._hist_instant_burden = []

    def start_week(self, k, I_w):
        self._Iw_per_week[k] = I_w
        # ``E_known_all[k]`` is set in ``__init__`` (k=0) or by
        # ``_finalize_week(k-1)`` (k >= 1); see ``_compute_Ew_hat_from_week``.
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
        y_sim = self.CAE_all[sim_w_prev + 1]
        ty_sim = self.CAE_short_all[sim_w_prev + 1]
        wp = self.wp_all[sim_w_prev + 1]
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

    def run_episode(self, agent):
        agent.reset()
        self._hist_instant_burden.clear()
        self.s["recent_burden"] = self._recent_burden_initial
        self._hist_ws_interaction = (
            [self._ws_interaction_initial] * INTERACTION_ROLLING_WINDOW
        )
        self._hist_salience_interaction = (
            [self._salience_interaction_initial] * INTERACTION_ROLLING_WINDOW
        )

        self.s["Interacted_7d_walk"] = self._ws_interaction_initial
        self.s["Interacted_7d_salience"] = self._salience_interaction_initial

        self._hist_recorded_physical_activity = [self._rpa7_initial] * RPA_ROLLING_WINDOW
        self.s["Previous7DaysRPA"] = self._rpa7_initial

        
        self.s["past7days_morning_wearing"] = self._wear7_initial
        self._hist_morning_wear = [self._wear7_initial] * 7

        for k in range(self.nweek):
            packet = self.get_week_packet(k)
            I_w = agent.begin_week(k, packet)

            self.start_week(k, I_w)

            for d in range(1, 7):
                for t in range(1, 3):
                    state = self.get_state(k, d, t)
                    A_wdt = agent.act(k, d, t, state)
                    self.step_action(k, d, t, A_wdt, I_w)

            self._finalize_week(k)

        return agent.results()

    def step_action(self, k, d, t, action, I_w):
        sim_w = k
        d_w = d - 1
        t_sim = t - 1
        d_global = sim_w * self.W_days + d_w
        step_idx = sim_w * FOURSC_SLOTS_PER_WEEK + d_w * 2 + t_sim

        self._Iw_per_week[sim_w] = I_w

        if t_sim == 0 and (sim_w, d_w) not in self._day_started:
            self._day_started[(sim_w, d_w)] = True
            self._start_day(sim_w, d_w, d_global, I_w)

        self.s["decision_time"] = float(t_sim)
        Ah = float(action)

        inst_burden = (
            Ah
            + float(self.s["yesterday_salience_message"])
        ) / 2.0
        self._hist_instant_burden.append(inst_burden)
        self.s["recent_burden"] = _ewm_recent_burden_last(self._hist_instant_burden)

        # Store base feature with cae=0; per-particle substitution happens in PF.
        x_fourSC = build_fourSC_features(self.s, Ah, pu=self.E_known_all[sim_w], cae=0.0)
        fourSC = self.env.gen_fourSC(self.s, Ah, step_idx)
        pv = self.env.gen_pageview(self.s, Ah, I_w*self.wp_all[sim_w], step_idx)

        # Generate current-slot auxiliary mediators using the OLD state summaries.
        # Important: Interacted_7d_* should not include the current slot until after
        # ws_interaction / salience_interaction have been generated.
        prior2hour_step = self.env.gen_prior2hour_step_count(self.s, step_idx)
        ws_interaction = self.env.gen_ws_interaction(self.s, step_idx)
        salience_interaction = self.env.gen_salience_interaction(self.s, step_idx)

        self.fourSC_all[step_idx] = fourSC
        self.pageview_all[step_idx] = pv
        self.prior2hour_step_all[step_idx] = prior2hour_step
        self.ws_interaction_all[step_idx] = ws_interaction
        self.salience_interaction_all[step_idx] = salience_interaction
        self.action_all[step_idx] = Ah

        self.fourSC_cond[step_idx] = x_fourSC

        self._today_fourSC[t_sim] = fourSC
        self._today_pageview[t_sim] = pv
        self._today_action[t_sim] = Ah

        self._foursc_wk[d_w * self.K + t_sim] = fourSC
        self._pw_wk[d_w * self.K + t_sim] = pv

        # Update lag/state variables for the NEXT decision slot.
        self.s["fourSC_lag1"] = fourSC
        self.s["hourly_pageview_lag1"] = pv

        # Current prior-2-hour value becomes both the current state value and the lag
        # used by the next decision-slot prior2hour model.
        self.s["prior2hour_step"] = prior2hour_step
        self.s["prior2hour_step_count_lag1"] = prior2hour_step

        # Update rolling 7-day interaction fractions after observing current slot.
        self._hist_ws_interaction.append(float(ws_interaction))
        self._hist_salience_interaction.append(float(salience_interaction))

        self.s["Interacted_7d_walk"] = _rolling_mean_last(
            self._hist_ws_interaction,
            INTERACTION_ROLLING_WINDOW,
        )
        self.s["Interacted_7d_salience"] = _rolling_mean_last(
            self._hist_salience_interaction,
            INTERACTION_ROLLING_WINDOW,
        )

        if t_sim == self.K - 1:
            self._end_day(sim_w, d_w, d_global)

    def _start_day(self, sim_w, d_w, d_global, Iw):
        # Periodic calendar within each RL week: d_w ∈ {0,…,6} (Mon–Sun block), not d_global.
        # Match ``df_fit``: (dow - (1+7)/2) / ((7-1)/2)  →  dow ∈ {1,…,7}  to  [-1, 1]; 1 = Monday.
        dow_i = int(((self.start_dow - 1 + d_w) % 7) + 1)
        dow_n = (dow_i - (1.0 + 7.0) / 2.0) / ((7.0 - 1.0) / 2.0)
        self.s["dow"] = dow_n
        self.s["is_weekend"] = 1.0 if dow_i >= 6 else 0.0
        self.s["salience_message"] = float(rd.binomial(1, 0.5))

        # Daily RPA is modeled on the morning row.
        self.s["decision_time"] = 0.0

        morning_step_idx = sim_w * FOURSC_SLOTS_PER_WEEK + d_w * self.K

        rpa = self.env.gen_recorded_physical_activity(
            self.s,
            morning_step_idx,
        )

        self.s["recorded_physical_activity"] = rpa
        self.recorded_physical_activity_all[d_global] = rpa

        # Store the previous-7-day predictor used during this day.
        self.Previous7DaysRPA_all[d_global] = self.s["Previous7DaysRPA"]

        # Antic / fitbit / daily are drawn in _end_day (today’s WS). gen_pageview uses
        # yesterday_* in vani_env; fourSC uses anticipated_affect_yesterday (prior-day affect) only.

        self._today_fourSC[:] = 0.0
        self._today_pageview[:] = 0.0
        self._today_action[:] = 0.0

    def _end_day(self, sim_w, d_w, d_global):
        daily_mean_step = np.mean(self._today_fourSC)
        daily_mean_pv = np.mean(self._today_pageview)

        ws_m = float(self._today_action[0])
        ws_a = float(self._today_action[1])
        Iw = self._Iw_per_week[sim_w]

        # Same-day predictors for daily mediators (before fitbit / dailysurvey / antic).
        self.s["today_step"] = float(daily_mean_step)
        self.s["today_pageview_morning"] = float(self._today_pageview[0])
        self.s["today_pageview_afternoon"] = float(self._today_pageview[1])

        fitbit = self.env.gen_fitbitwearing(self.s, ws_m, ws_a, Iw*self.wp_all[sim_w], d_global)
        self.s["fitbitwearing_morning"] = fitbit
        daily_pres = self.env.gen_dailysurvey(self.s, ws_m, ws_a, Iw*self.wp_all[sim_w], d_global)
        self.s["daily_present"] = daily_pres
        # Latent anticipated affect — always drawn from the generative process.
        # Stored as s["anticipated_affect"] so that tomorrow's decision-slot calls
        # to gen_fourSC_mean / gen_antic_mean in vani_env.py see the true latent
        # from yesterday (not the NaN-masked carry-forward).
        antic = self.env.gen_antic(self.s, ws_m, ws_a, d_global)
        self.s["anticipated_affect"] = antic

        # Store base feature with cae=0; per-particle substitution happens in PF.
        self.antic_cond[d_global] = build_antic_features(
            self.s, ws_m, ws_a, pu=self.E_known_all[sim_w], cae=0.0
        )
        self.fitbit_all[d_global] = fitbit
        self.daily_all[d_global] = daily_pres
        self.antic_all[d_global] = antic                                   # latent, always finite
        # PF / RL see NaN when the daily survey was skipped.
        survey_present = float(daily_pres) == 1.0
        self.antic_obs_all[d_global] = antic if survey_present else np.nan

        # Carry-forward: only advance on observed days so the agent never sees
        # information it couldn't have observed (latent stays in s["anticipated_affect"]).
        if survey_present:
            self.s["anticipated_affect_yesterday"] = antic

        self.s["yesterday_step"] = daily_mean_step
        self.s["yesterday_fitbitwearing_morning"] = fitbit
        self.s["yesterday_present"] = daily_pres
        self.s["yesterday_pageview_morning"] = self._today_pageview[0]
        self.s["yesterday_pageview_afternoon"] = self._today_pageview[1]
        self.s["dow_lag1"] = self.s["dow"]

        sp = self._ewm_span
        self._hist_daily_step.append(float(daily_mean_step))
        self._hist_daily_pv.append(float(daily_mean_pv))
        self._hist_morning_wear.append(float(fitbit))
        self.s["seven_day_step_count_avg"] = _ewm_last(self._hist_daily_step, sp)
        self.s["seven_day_pageview"] = _ewm_last(self._hist_daily_pv, sp)

        past7_wear = _rolling_mean_last(self._hist_morning_wear, 7)
        self.s["past7days_morning_wearing"] = past7_wear

        self._yesterday_morning_WS = self._today_action[0]
        self._yesterday_afternoon_WS = self._today_action[1]

        self._antic_wk[d_w] = antic
        self._dw_wk[d_w] = fitbit
        self._dp_wk[d_w] = daily_pres

        # After today's daily RPA is known, update histories for tomorrow.
        today_rpa = float(self.s["recorded_physical_activity"])

        self._hist_recorded_physical_activity.append(today_rpa)

        self.s["recorded_physical_activity_lag1"] = today_rpa
        self.s["Previous7DaysRPA"] = _rolling_mean_last(
            self._hist_recorded_physical_activity,
            RPA_ROLLING_WINDOW,
        )

        # fourSC / pageview use calendar-yesterday message flags (lagged one day).
        self.s["yesterday_salience_message"] = self.s["salience_message"]

    def _finalize_week(self, sim_w):
        if sim_w < 0 or sim_w >= self.nweek or self._week_finalized[sim_w]:
            return

        Iw = self._Iw_per_week[sim_w]
        d_w_sun = 6
        d_global = sim_w * self.W_days + d_w_sun
        # Match df_fit / 1.1_standardization: (week - (1+n)/2) / ((n-1)/2) → week ∈ {1,…,n} maps to [-1, 1]
        wk = sim_w + 1
        n = self.nweek
        week_norm = 0.0 if n <= 1 else (wk - (1.0 + n) / 2.0) / ((n - 1.0) / 2.0)

        if (sim_w, d_w_sun) not in self._day_started:
            self._day_started[(sim_w, d_w_sun)] = True
            self._start_day(sim_w, d_w_sun, d_global, Iw)

        for t_sim in range(self.K):
            step_idx = sim_w * FOURSC_SLOTS_PER_WEEK + d_w_sun * 2 + t_sim
            Ah = 0.0
            self.s["decision_time"] = float(t_sim)

            # Store base feature with cae=0; per-particle substitution in PF.
            x_fourSC = build_fourSC_features(self.s, Ah, pu=self.E_known_all[sim_w], cae=0.0)
            fourSC = self.env.gen_fourSC(self.s, Ah, step_idx)
            pv = self.env.gen_pageview(self.s, Ah, Iw*self.wp_all[sim_w], step_idx)

            self.fourSC_all[step_idx] = fourSC
            self.pageview_all[step_idx] = pv
            self.action_all[step_idx] = Ah
            self.fourSC_cond[step_idx] = x_fourSC
            self._today_fourSC[t_sim] = fourSC
            self._today_pageview[t_sim] = pv
            self._today_action[t_sim] = Ah
            self._foursc_wk[d_w_sun * self.K + t_sim] = fourSC
            self._pw_wk[d_w_sun * self.K + t_sim] = pv

            self.s["fourSC_lag1"] = fourSC
            self.s["hourly_pageview_lag1"] = pv
            # self.s["prior2hour_step"] = fourSC

        self._end_day(sim_w, d_w_sun, d_global)

        cae = self.env.gen_CAE(
            self.s["CAE_avg_lastweek"], week_norm,
            self._foursc_wk, self._antic_wk, sim_w,
        )

        pu = self.env.gen_perceived_utility(
            self.s["perceived_utility_lastweek"], week_norm,
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
        self.CAE_all[sim_w + 1] = cae
        self.pu_all[sim_w + 1] = pu
        self.wp_all[sim_w + 1] = self.env.gen_week_present(pu, sim_w + 1)
        self.CAE_short_all[sim_w + 1] = cs
        self.U1_all[sim_w + 1] = u1
        self.U2_all[sim_w + 1] = u2

        # Store AR-1 = 0; the per-particle CAE value is substituted via
        # cae_delta_cumul_Y[sw, _CAE_AR1_COL] = 1 inside estimate_belief_state.
        self.CAE_cond[sim_w] = build_CAE_features(
            0.0, week_norm,
            self._foursc_wk, self._antic_wk,
        )
        # Store base feature with cae=0; per-particle substitution happens in PF.
        # tY uses the CURRENT week's CAE (contemporaneous), so the delta column
        # is filled with y_w^(j) — the particle's own draw — inside estimate_belief_state.
        self.CAE_short_cond[sim_w] = build_CAE_short_features(0.0)

        self.s["CAE_avg_lastweek"] = cae
        # Latent E_w driving env generative process (PV / FW / PJ / J / fourSC / antic).
        # The agent never sees this; it gets ``E_known_all`` instead.
        self.s["perceived_utility_lastweek"] = pu

        # Compute E_w_hat for the *next* RL week using observable aggregates of
        # the just-finalized week (J, U1, U2, PV_sum, FW_sum, PJ_sum).

        self.E_known_all[sim_w + 1] = self._compute_Ew_hat_from_week(sim_w)

        self._week_finalized[sim_w] = True

    def _compute_Ew_hat_from_week(self, sim_w):
        """Apply pooled linear coefficients to observable aggregates of week ``sim_w``."""
        # J_w / U1 / U2 for simulated week sim_w are stored at sim_w + 1
        # because index 0 is the pre-RL baseline.
        J_w = float(self.wp_all[sim_w + 1]) if not np.isnan(self.wp_all[sim_w + 1]) else 0.0
        u1 = float(self.U1_all[sim_w + 1]) if not np.isnan(self.U1_all[sim_w + 1]) else 0.0
        u2 = float(self.U2_all[sim_w + 1]) if not np.isnan(self.U2_all[sim_w + 1]) else 0.0
        half_J_tool8 = 0.5 * J_w * ((u1 + 1.0) + (u2 + 1.0)) / 8.0
        pv_sum = float(np.nansum(self._pw_wk) / 14.0)
        fw_sum = float(np.nansum(self._dw_wk) / 7.0)
        pj_sum = float(np.nansum(self._dp_wk) / 7.0)
        return apply_pooled_coefs(
            self._ew_coefs, J_w, half_J_tool8, pv_sum, fw_sum, pj_sum,
        )

    def get_pf_data(self, k):
        """Return decomposed PF data for particle-learning belief-state update.

        Feature vectors are stored as base + cae_delta pairs so that each
        particle j can substitute its own CAE history:

            X_j = X_base + cae_j * cae_delta

        where ``cae_j`` comes from ``_cae_by_sw(j, sw)`` in algorithm.py.
        """
        sim_w_prev = k - 1
        K14 = FOURSC_SLOTS_PER_WEEK
        foursc_lag1_col = 1   # AR-1 lag inside fourSC (cleared for first slot)
        antic_lag1_col = 1    # anticipated-affect lag in antic (cleared for first day)

        if sim_w_prev >= 0:
            assert self._week_finalized[sim_w_prev], (
                "call _finalize_week on simulated week k-1 before get_pf_data(k)."
            )

        # ── helpers to extract per-row CAE delta vectors ─────────────────────
        def _foursc_delta(x):
            d = np.zeros_like(x)
            d[_FOURSC_CAE_COL]    = 1.0
            d[_FOURSC_AH_CAE_COL] = x[_FOURSC_AH_COL]   # = Ah stored in base
            return d

        def _antic_delta(x):
            d = np.zeros_like(x)
            d[_ANTIC_CAE_COL]     = 1.0
            d[_ANTIC_CAE_WS_M_COL] = x[_ANTIC_WS_M_COL]  # ws_morning in base
            d[_ANTIC_CAE_WS_A_COL] = x[_ANTIC_WS_A_COL]  # ws_afternoon in base
            return d

        # ── current-week prediction rows (previous simulation week) ──────────
        # Pooled mediator models: m=0 is fourSC (all slots), m=1 is antic (all days).
        start_row = sim_w_prev * K14
        X_MY_base, cae_delta_MY, M_Y_obs = [], [], []

        four_rows = np.array([start_row + m for m in range(N_MED_SLOT)], dtype=int)
        X_four = self.fourSC_cond[four_rows].copy()
        if N_MED_SLOT > 0:
            X_four[0, foursc_lag1_col] = 0.0  # clear AR-1 carry-in for week boundary
        delta_four = np.array([_foursc_delta(X_four[i]) for i in range(X_four.shape[0])])
        y_four = self.fourSC_all[four_rows].copy()

        antic_rows = np.array([sim_w_prev * self.W_days + d for d in range(N_MED_ANTIC_DAY)], dtype=int)
        X_antic = self.antic_cond[antic_rows].copy()
        if N_MED_ANTIC_DAY > 0:
            X_antic[0, antic_lag1_col] = 0.0  # clear lag carry-in at week boundary (Monday)
        delta_antic = np.array([_antic_delta(X_antic[i]) for i in range(X_antic.shape[0])])
        y_antic = self.antic_obs_all[antic_rows].copy()

        X_MY_base.extend([X_four, X_antic])
        cae_delta_MY.extend([delta_four, delta_antic])
        M_Y_obs.extend([y_four, y_antic])

        # CAE model (AR-1 stored as 0; delta = e_{_CAE_AR1_COL})
        X_Y_base    = self.CAE_cond[sim_w_prev].copy()
        cae_delta_Y = np.zeros(X_Y_base.shape[0])
        cae_delta_Y[_CAE_AR1_COL] = 1.0

        # CAE-short model: [1, CAE_avg] – same current-week CAE as y_w^(j)
        X_tY_base    = np.array([1.0, 0.0])     # CAE filled per-particle as y_w[j]
        cae_delta_tY = np.array([0.0, 1.0])

        # ── cumulative data for per-particle Bayesian posterior update ────────
        if k >= 2:
            X_cumul_MY_base, cae_delta_cumul_MY = [], []
            week_idx_cumul_MY, y_cumul_MY        = [], []

            # m = 0 : pooled fourSC rows across all slots and all past weeks.
            four_pairs = [(sw, m) for sw in range(sim_w_prev) for m in range(N_MED_SLOT)]
            four_sw_idx = np.array([sw for sw, _ in four_pairs], dtype=int)
            four_slot_idx = np.array([m for _, m in four_pairs], dtype=int)
            four_rows = np.array([sw * K14 + m for sw, m in four_pairs], dtype=int)
            X_four_c = self.fourSC_cond[four_rows].copy()
            X_four_c[four_slot_idx == 0, foursc_lag1_col] = 0.0
            delta_four_c = np.array([_foursc_delta(X_four_c[i]) for i in range(X_four_c.shape[0])])
            y_four_c = self.fourSC_all[four_rows].copy()
            X_cumul_MY_base.append(X_four_c)
            cae_delta_cumul_MY.append(delta_four_c)
            week_idx_cumul_MY.append(four_sw_idx)
            y_cumul_MY.append(y_four_c)

            # m = 1 : pooled antic rows across all days and all past weeks.
            antic_pairs = [(sw, day_d) for sw in range(sim_w_prev) for day_d in range(N_MED_ANTIC_DAY)]
            antic_sw_idx = np.array([sw for sw, _ in antic_pairs], dtype=int)
            antic_day_idx = np.array([day_d for _, day_d in antic_pairs], dtype=int)
            antic_rows = np.array([sw * self.W_days + day_d for sw, day_d in antic_pairs], dtype=int)
            X_antic_c = self.antic_cond[antic_rows].copy()
            X_antic_c[antic_day_idx == 0, antic_lag1_col] = 0.0
            y_antic_c = self.antic_obs_all[antic_rows].copy()
            obs_mask = np.isfinite(y_antic_c)
            delta_antic_c = np.array([_antic_delta(X_antic_c[i]) for i in range(X_antic_c.shape[0])])
            X_cumul_MY_base.append(X_antic_c[obs_mask])
            cae_delta_cumul_MY.append(delta_antic_c[obs_mask])
            week_idx_cumul_MY.append(antic_sw_idx[obs_mask])
            y_cumul_MY.append(y_antic_c[obs_mask])

            # CAE model: rows sw = 0..sim_w_prev-1, outcome at CAE_all[sw+1]
            X_cumul_Y_base  = self.CAE_cond[:sim_w_prev].copy()
            cae_delta_cumul_Y = np.zeros((sim_w_prev, X_cumul_Y_base.shape[1]))
            cae_delta_cumul_Y[:, _CAE_AR1_COL] = 1.0
            y_cumul_Y = self.CAE_all[1:sim_w_prev + 1].copy()

            # CAE-short: base stored as [1,0]; per-particle substitution uses
            # particle j's CAE at week sw (contemporaneous, not lagged).
            #
            # tilde_Y is observed only when the weekly survey was completed,
            # i.e. J_w == 1. For J_w == 0 weeks, the env still wrote a value
            # into ``CAE_short_all`` (the env's true latent CAE_short), but the
            # agent never saw it; including those rows would leak unobserved
            # truth into the tY posterior. Drop them and emit ``week_idx_cumul_tY``
            # so algorithm.py can map each kept row back to its sim_w when
            # plugging in particle j's CAE.
            X_tY_full      = self.CAE_short_cond[:sim_w_prev].copy()    # all [1,0] rows
            cae_delta_full = np.zeros((sim_w_prev, 2))
            cae_delta_full[:, 1] = 1.0
            y_tY_full      = self.CAE_short_all[1:sim_w_prev + 1].copy()
            jw_tY_mask     = self.wp_all[:sim_w_prev] == 1.0           # J_w[sw] for sw=0..sim_w_prev-1
            X_cumul_tY         = X_tY_full[jw_tY_mask]
            cae_delta_cumul_tY = cae_delta_full[jw_tY_mask]
            y_cumul_tY         = y_tY_full[jw_tY_mask]
            week_idx_cumul_tY  = np.arange(sim_w_prev)[jw_tY_mask]
        else:
            X_cumul_MY_base = cae_delta_cumul_MY = week_idx_cumul_MY = y_cumul_MY = None
            X_cumul_Y_base  = cae_delta_cumul_Y  = y_cumul_Y = None
            X_cumul_tY = cae_delta_cumul_tY = y_cumul_tY = None
            week_idx_cumul_tY = None

        return {
            # current-week prediction rows
            "X_MY_base":    X_MY_base,
            "cae_delta_MY": cae_delta_MY,
            "M_Y_obs":      M_Y_obs,
            "X_Y_base":     X_Y_base,
            "cae_delta_Y":  cae_delta_Y,
            "X_tY_base":    X_tY_base,
            "cae_delta_tY": cae_delta_tY,
            # cumulative rows for per-particle posterior update
            "X_cumul_MY_base":    X_cumul_MY_base,
            "cae_delta_cumul_MY": cae_delta_cumul_MY,
            "week_idx_cumul_MY":  week_idx_cumul_MY,
            "y_cumul_MY":         y_cumul_MY,
            "X_cumul_Y_base":     X_cumul_Y_base,
            "cae_delta_cumul_Y":  cae_delta_cumul_Y,
            "y_cumul_Y":          y_cumul_Y,
            "X_cumul_tY":          X_cumul_tY,
            "cae_delta_cumul_tY":  cae_delta_cumul_tY,
            "week_idx_cumul_tY":   week_idx_cumul_tY,
            "y_cumul_tY":          y_cumul_tY,
            # baseline CAE for AR-1 seed at sim_w=0
            "cae_all_0":    self._cae_baseline,
            "sim_w_prev":   sim_w_prev,
        }

    def get_state(self, k, d, t):
        E_w = self.E_known_all[k] if (
            0 <= k < self.nweek and not np.isnan(self.E_known_all[k])
        ) else 0.0

        # E_w is passed as state["E_w"] and used directly by build_phi_action;
        # CAE belief is b_hat/b_tilde from the PF. Neither belongs in C_vec.
        C_vec = build_rl_context_vector(self.s)

        if d == 0 and t == 0:
            return {
                "E_w": E_w,
                "M_Y": np.zeros(RL_MY_SHAPE),
                "M_E": np.zeros(RL_ME_SHAPE),
                "C": C_vec,
            }

        M_Y = np.zeros(RL_MY_SHAPE)
        M_E = np.zeros(RL_ME_SHAPE)
        base = k * self.W_days

        # RL M_Y / M_E: six days × two slots = 12 walking mediators (stride in flat arrays is still 14).
        for dd in range(1, 7):
            d_global = base + (dd - 1)
            # Day-level mediators: written in _start_day to weekday dd’s index
            if dd < d and 0 <= d_global < self.D:
                # M_Y antic slot: use observed value; if survey was missed that
                # day (NaN in antic_obs_all), carry forward the last observed
                # value — the agent can't see latent affect through a missed survey.
                obs_a = self.antic_obs_all[d_global]
                M_Y[dd - 1, 2] = obs_a if np.isfinite(obs_a) else self.s["anticipated_affect_yesterday"]
                M_E[dd - 1, 2] = self.fitbit_all[d_global]
                M_E[dd - 1, 3] = self.daily_all[d_global]
            for tt in range(1, 3):
                if (dd, tt) < (d, t):
                    r = k * FOURSC_SLOTS_PER_WEEK + (dd - 1) * self.K + (tt - 1)
                    if r < self.T:
                        M_Y[dd - 1, tt - 1] = self.fourSC_all[r]
                        M_E[dd - 1, tt - 1] = self.pageview_all[r]

        return {"E_w": E_w, "M_Y": M_Y, "M_E": M_E, "C": C_vec}

    def reward_fn(self, k):
        """Surrogate reward after sim week k (0-based): weekly CAE for that week."""
        idx = k + 1
        if 0 <= k < self.nweek and not np.isnan(self.CAE_all[idx]):
            return self.CAE_all[idx]
        return 0.0


print("OnlineEnv defined.")

# %%
# ──────────────────────────────────────────────────────────────────
# Placeholder priors and hyperparameters
# ──────────────────────────────────────────────────────────────────

# Pooled mediator-model feature dimensions for the particle filter:
# m=0 is fourSC (all slots share one model), m=1 is antic (all days share one model).
P_MY_FOURSC = P_FOURSC
P_MY_ANTIC  = P_ANTIC
P_CAE = 24   # CAE model feature dimension
P_TY  = 2    # CAE_short model feature dimension
print(f"PF mediator dims: P_MY_FOURSC={P_MY_FOURSC}, P_MY_ANTIC={P_MY_ANTIC}, "
      f"P_CAE={P_CAE}, P_TY={P_TY}")

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

# ── pooled mediator model priors (m=0 fourSC, m=1 antic) ──
nu_0_MY = [
    np.zeros(P_MY_FOURSC),
    np.zeros(P_MY_ANTIC),
]
Gamma_0_MY = [
    np.eye(P_MY_FOURSC),
    np.eye(P_MY_ANTIC),
]
sigma2_MY = [1.0, 1.0]

# ── Y model (CAE) ──
nu_0_Y    = np.zeros(P_CAE)
Gamma_0_Y = np.eye(P_CAE)
sigma2_Y  = 1.0

# ── tilde_Y model (CAE_short) ──
nu_0_tilde_Y    = np.zeros(P_TY)
Gamma_0_tilde_Y = np.eye(P_TY)
sigma2_tilde_Y  = 1.0

# ── RL hyperparameters (shared) ──

GAMMA_BAR    = 0.5
GAMMA_DT_micro     = (GAMMA_BAR ** (1.0 / 12)) * np.ones((6, 2))
TARGET_C     = 2
EPSILON_0    = 0.05   # this is the clipping parameter
J_PARTICLES  = 50
B_ENSEMBLES  = 50
Y1_BASELINE_DEFAULT = 0.0
NWEEK        = 36


def _initial_cae_from_env(uid):
    """Participant-specific CAE baseline; match vani_env missing handling (fallback 0)."""
    y1 = float(make_initial_state(participant_id=uid).get("CAE_avg_lastweek", Y1_BASELINE_DEFAULT))
    return y1 if np.isfinite(y1) else Y1_BASELINE_DEFAULT

# RL priors (per-algorithm dimensions)
def _rl_priors(p_rl):
    return (np.zeros(p_rl), 10.0 * np.eye(p_rl), 1.0)

mu_0_micro, Sigma_0_micro, sigma2_rl_micro = _rl_priors(P_RL_MICRO)
mu_0_reward, Sigma_0_reward, sigma2_reward = _rl_priors(P_RL_REWARDSHAPING)
mu_0_bottleneck, Sigma_0_bottleneck, sigma2_bottleneck = _rl_priors(P_RL_BOTTLENECK)

print(
    f"Priors ready:  p_rl(micro)={P_RL_MICRO}, "
    f"p_rs={P_RL_REWARDSHAPING}, p_b={P_RL_BOTTLENECK}"
)

# %%
# ──────────────────────────────────────────────────────────────────
# Run one participant with a given algorithm
# ──────────────────────────────────────────────────────────────────

def _gamma_dt_micro(gamma_bar):
    """Per-slot discount γ_{d,t} = γ_bar^{1/12}; vanishes to zeros when γ_bar=0."""
    return (gamma_bar ** (1.0 / 12)) * np.ones((6, 2))


def run_micro_query(uid, seed=42, gamma_bar=GAMMA_BAR):
    cfg = EnvConfig(uid)
    nweek = cfg.nweek
    env = Env(cfg, noise="sequential")
    oenv = OnlineEnv(env, nweek=nweek, seed=seed)

    agent = MicroQueryAgent(
        W=nweek, J=J_PARTICLES, B=B_ENSEMBLES, epsilon_0=EPSILON_0,
        mu_0_rl=mu_0_micro, Sigma_0_rl=Sigma_0_micro, sigma2_rl=sigma2_rl_micro,
        gamma_dt=_gamma_dt_micro(gamma_bar), gamma_bar=gamma_bar,
        target_update_C=TARGET_C,
        nu_0_MY=nu_0_MY, Gamma_0_MY=Gamma_0_MY, sigma2_MY=sigma2_MY,
        nu_0_Y=nu_0_Y, Gamma_0_Y=Gamma_0_Y, sigma2_Y=sigma2_Y,
        nu_0_tilde_Y=nu_0_tilde_Y, Gamma_0_tilde_Y=Gamma_0_tilde_Y,
        sigma2_tilde_Y=sigma2_tilde_Y,
        Y_1=_initial_cae_from_env(uid),
        get_state=oenv.get_state,
        reward_fn=oenv.reward_fn,
        rng=np.random.default_rng(seed),
    )

    result = oenv.run_episode(agent)
    return result, oenv


def run_micro_query_rs(uid, seed=42, gamma_bar=GAMMA_BAR):
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
        Y_1=_initial_cae_from_env(uid),
        get_state=oenv.get_state,
        reward_fn=oenv.reward_fn,
        rng=np.random.default_rng(seed),
    )

    result = oenv.run_episode(agent)
    return result, oenv


def run_micro_query_mtd(uid, seed=42, gamma_bar=GAMMA_BAR):
    """Micro-query agent with modified TD loss (week-start bottleneck V_alpha)."""
    cfg = EnvConfig(uid)
    nweek = cfg.nweek
    env = Env(cfg, noise="sequential")
    oenv = OnlineEnv(env, nweek=nweek, seed=seed)

    agent = MicroQueryAgent_ModifiedTDLoss(
        W=nweek, J=J_PARTICLES, B=B_ENSEMBLES, epsilon_0=EPSILON_0,
        mu_0_rl=mu_0_micro, Sigma_0_rl=Sigma_0_micro, sigma2_rl=sigma2_rl_micro,
        gamma_dt=_gamma_dt_micro(gamma_bar), gamma_bar=gamma_bar,
        target_update_C=TARGET_C,
        nu_0_MY=nu_0_MY, Gamma_0_MY=Gamma_0_MY, sigma2_MY=sigma2_MY,
        nu_0_Y=nu_0_Y, Gamma_0_Y=Gamma_0_Y, sigma2_Y=sigma2_Y,
        nu_0_tilde_Y=nu_0_tilde_Y, Gamma_0_tilde_Y=Gamma_0_tilde_Y,
        sigma2_tilde_Y=sigma2_tilde_Y,
        mu_0_bottleneck=mu_0_bottleneck, Sigma_0_bottleneck=Sigma_0_bottleneck,
        sigma2_bottleneck=sigma2_bottleneck,
        Y_1=_initial_cae_from_env(uid),
        get_state=oenv.get_state,
        reward_fn=oenv.reward_fn,
        rng=np.random.default_rng(seed),
    )

    result = oenv.run_episode(agent)
    return result, oenv


def run_micro_query_rs_mtd(uid, seed=42, gamma_bar=GAMMA_BAR):
    """Micro-query agent with both reward shaping and modified TD loss."""
    cfg = EnvConfig(uid)
    nweek = cfg.nweek
    env = Env(cfg, noise="sequential")
    oenv = OnlineEnv(env, nweek=nweek, seed=seed)

    agent = MicroQueryAgent_rewardshaping_modifiedTD(
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
        mu_0_bottleneck=mu_0_bottleneck, Sigma_0_bottleneck=Sigma_0_bottleneck,
        sigma2_bottleneck=sigma2_bottleneck,
        Y_1=_initial_cae_from_env(uid),
        get_state=oenv.get_state,
        reward_fn=oenv.reward_fn,
        rng=np.random.default_rng(seed),
    )

    result = oenv.run_episode(agent)
    return result, oenv


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
}


print("Runner functions defined.")

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

    The conditioning feature matrices (fourSC_cond, antic_cond, CAE_cond,
    CAE_short_cond) are not stored — they are large and re-derivable from
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
        "antic_all":              oenv.antic_all.copy(),       # latent
        "antic_obs_all":          oenv.antic_obs_all.copy(),   # NaN if survey missed
        "fitbit_all":             oenv.fitbit_all.copy(),
        "daily_all":              oenv.daily_all.copy(),       # daily-survey present
        "recorded_physical_activity_all": oenv.recorded_physical_activity_all.copy(),
        "Previous7DaysRPA_all":   oenv.Previous7DaysRPA_all.copy(),
        # ── slot-level ────────────────────────────────────────────
        "fourSC_all":             oenv.fourSC_all.copy(),
        "pageview_all":           oenv.pageview_all.copy(),
        "action_all":             oenv.action_all.copy(),
        "prior2hour_step_all":    oenv.prior2hour_step_all.copy(),
        "ws_interaction_all":     oenv.ws_interaction_all.copy(),
        "salience_interaction_all": oenv.salience_interaction_all.copy(),
    }


if __name__ == "__main__":
    user_ids = np.loadtxt(PARAMS_DIR / "user_ids.txt", dtype=int)
    N_EXPERIMENTS = 100
    SEEDS = list(range(N_EXPERIMENTS))
    uid_draw_rng = np.random.default_rng(0)
    n_users = 100

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
    oenv_runs  = {name: [] for name in ALGORITHMS}
    cae_by_uid = {name: {} for name in ALGORITHMS}   # keyed by uid (flat across exps)

    for exp_idx, seed in enumerate(SEEDS):
        sampled_uids = uid_draw_rng.choice(user_ids, size=n_users, replace=True)
        run_uids.append(np.asarray(sampled_uids, dtype=int))
        # Per-experiment sub-list, one entry per participant draw.
        for name in ALGORITHMS:
            cae_runs[name].append([])
            piA_runs[name].append([])
            oenv_runs[name].append([])

        for draw_idx, uid in enumerate(sampled_uids):
            uid = int(uid)
            print(
                f"  Experiment {exp_idx}, draw {draw_idx + 1}/{n_users}, "
                f"user {uid}, seed {seed} ... ",
                end="", flush=True
            )

            summary_parts = []
            for name, (runner, _label) in ALGORITHMS.items():
                res, oenv = runner(uid, seed=seed)
                snap = _snapshot_oenv(oenv)
                oenv_runs[name][exp_idx].append(snap)
                cae_full = snap["CAE_all"]
                cae_runs[name][exp_idx].append(cae_full)
                piA_runs[name][exp_idx].append(res["pi_A"].copy())
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
    #   trajectories["rs_g05"]["fourSC_all"]      → (N_EXPERIMENTS, n_users, T)
    #   trajectories["rs_g05"]["antic_obs_all"]   → (N_EXPERIMENTS, n_users, D)
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
    # Persist results to disk (timestamped folder under ./results/)
    # ──────────────────────────────────────────────────────────────────
    OUTPUT_DIR = Path("results") / datetime.now().strftime("%Y%m%d-%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {OUTPUT_DIR.resolve()}")

    # Run-level metadata (one file shared across all algorithms).
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump({
            "nweek":          NWEEK,
            "n_users":        n_users,
            "n_experiments":  N_EXPERIMENTS,
            "seeds":          list(SEEDS),
            "algorithms":     list(ALGORITHMS.keys()),
            "labels":         {n: ALGORITHMS[n][1] for n in ALGORITHMS},
            "gamma_bar":      GAMMA_BAR,
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

    # (1,1) Action probability
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
# data = np.load(RES/"rs_g05.npz")            # has cae_runs, piA_runs, fourSC_all, …
# cae_by_uid = pickle.load(open(RES/"rs_g05_cae_by_uid.pkl","rb"))