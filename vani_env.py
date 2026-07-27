# %%
"""
Generative environment for the vanilla testbed.

**Vanilla mediators** (``5_fit_vanilla_testbed.py``): Ridge / L2 logistic with
``Intercept`` in the design matrix and ``fit_intercept=False`` — the leading
column of ones is part of ``theta_*``, not sklearn's intercept.

**Perceived-utility stack** (``4_perceived_utility.py``): parameters are merged
into the same ``params_env_<userid>.json`` as ``theta_penalized_*`` /
``resid_penalized_*`` (legacy ``*_ml_*`` keys remain readable).
Within-week outcomes (hourly PV, daily FW/PJ) and weekly ``week_present`` use
those coefficients conditional on latent ``E_w`` carried in
``state[\"perceivedUtilityLastWeek\"]``.

Weekly AR transition for ``E_w``::

    E_{w+1} = a0 + a1 E_w + a2 \\bar{PV}_w + a3 \\bar{FW}_w + a4 \\bar{PJ}_w + \\varepsilon,

with :math:`\\varepsilon \\sim N(0, \\sigma_E^2)` (``sigma_E`` from
``theta_penalized_Ew``).
Means match ``perceivedUtility.transition_matrix`` / ``est_Ew_weights`` conventions:
``\\bar{PV}_w = (1/14)\\sum`` hourly pageviews, ``\\bar{FW}_w`` and ``\\bar{PJ}_w``
are :math:`(1/7)\\sum` over calendar days (one value per day).
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import numpy.random as rd

PROJECT_ROOT = Path(
    os.environ.get("ADAPR_PROJECT_ROOT", Path(__file__).resolve().parent)
).expanduser().resolve()
PARAMS_DIR = Path(
    PROJECT_ROOT / "env_para_vanilla"
).expanduser().resolve()


def load_CAE_norm_params(params_dir=PARAMS_DIR):
    """Return ``(shift, scale)`` used to z-score raw weekly CAE into ``CAE_avg_norm``.

    Standardization (``3_standardization.py``):
        ``CAE_avg_norm = (CAE_avg - shift) / scale``.
    Inverse (raw level): ``CAE_avg = shift + scale * CAE_avg_norm``.
    """
    with open(Path(params_dir) / "std_params.json", encoding="utf-8") as f:
        std = json.load(f)
    return float(std["CAE_avg_shift"]), float(std["CAE_avg_scale"])


def denormalize_CAE(cae_norm, params_dir=PARAMS_DIR):
    """Map normalized weekly CAE back to the raw (pre-normalization) scale."""
    shift, scale = load_CAE_norm_params(params_dir)
    return shift + scale * np.asarray(cae_norm, dtype=float)

THETA_PRIOR2HOUR_STEP_COUNT_NAMES = [
    "intercept",
    "EMA_Prior2HourStepCount",
    "is_weekend",
    "decision_time",
]

THETA_ACTIVE_STATUS_NAMES = [
    "intercept",
    "active_status_fraction_7days",
    "is_weekend",
]

THETA_WS_INTERACTION_NAMES = [
    "intercept",
    "Interacted_7d_walk",
    "is_weekend",
    "decision_time",
]

THETA_FOURSC_NAMES = [
    "intercept",
    "fourSC_lag1",
    "yesterday_step_count",
    "seven_day_step_count_avg",
    "prior2hour_step_count",
    "recent_burden",
    "seven_day_pageview_count",
    "past7days_morning_wearing",
    "Interacted_7d_walk",
    "anticipated_affect_yesterday",
    "fractionofactivedayspast7days",
    "is_weekend",
    "decision_time",
    "perceived_utility_lastweek",
    "CAE_avg_lastweek",
    "WalkingSuggestion",
    "WalkingSuggestion_by_yesterday_step_count",
    "WalkingSuggestion_by_prior2hour_step_count",
    "WalkingSuggestion_by_recent_burden",
    "WalkingSuggestion_by_seven_day_pageview_count",
    "WalkingSuggestion_by_past7days_morning_wearing",
    "WalkingSuggestion_by_Interacted_7d_walk",
    "WalkingSuggestion_by_anticipated_affect_yesterday",
    "WalkingSuggestion_by_is_weekend",
    "WalkingSuggestion_by_decision_time",
    "WalkingSuggestion_by_perceived_utility_lastweek",
    "WalkingSuggestion_by_CAE_avg_lastweek",
]

THETA_ANTIC_NAMES = [
    "intercept",
    "anticipated_affect_yesterday",
    "today_step_count",
    "active_status",
    "is_weekend",
    "perceived_utility_lastweek",
    "CAE_avg_lastweek",
    "A0_morning",
    "A1_afternoon",
    "A0_morning_by_is_weekend",
    "A1_afternoon_by_is_weekend",
    "A0_morning_by_perceived_utility_lastweek",
    "A1_afternoon_by_perceived_utility_lastweek",
    "A0_morning_by_CAE_avg_lastweek",
    "A1_afternoon_by_CAE_avg_lastweek",
]

THETA_CAE_NAMES = (
    ["intercept", "CAE_avg_lastweek", "week"]
    + [f"fourSC_slot_{j}" for j in range(14)]
    + [f"anticipated_affect_day_{j}" for j in range(7)]
)

THETA_CAE_SHORT_AVG_NAMES = [
    "intercept",
    "CAE_avg",
]

# Design sizes (``5_fit_vanilla_testbed.py``); no 7-day salience-interaction covariate.
# The environment uses the full fitted fourSC model, including the
# ``seven_day_pageview_count`` and ``anticipated_affect_yesterday`` predictors
# (and their WalkingSuggestion interactions) — these are NOT trimmed away.
P_FOURSC = len(THETA_FOURSC_NAMES)
_LEGACY_INTERACT_DROP = (2,)  # removed legacy salience-history covariate
_LEGACY_FOURSC_SALIENCE_DROP = (8, 21)
_LEGACY_ANTIC_SALIENCE_DROP = (4, 10, 11)
P_ANTIC = len(THETA_ANTIC_NAMES)
P_ACTIVE_STATUS = len(THETA_ACTIVE_STATUS_NAMES)
P_PRIOR2HOUR = len(THETA_PRIOR2HOUR_STEP_COUNT_NAMES)
P_CAE = len(THETA_CAE_NAMES)
P_CAE_SHORT = len(THETA_CAE_SHORT_AVG_NAMES)
PV_ML_BASE = 9
PV_ML_QUERY = 5
FW_ML_BASE = 9
FW_ML_QUERY = 4
PJ_ML_BASE = 9
PJ_ML_QUERY = 4


def _json_float_list(key: str, d: dict, n: int | None = None) -> np.ndarray:
    raw = d.get(key)
    if raw is None:
        if n is None:
            return np.array([], dtype=float)
        raise KeyError(f"Missing required key {key!r} in params JSON")
    out = np.asarray([float(x) for x in raw], dtype=float)
    if n is not None and out.shape[0] != n:
        raise ValueError(f"{key}: expected length {n}, got {out.shape[0]}")
    return out


def _validate_json_names(
    key: str,
    d: dict,
    expected: list[str],
    *,
    legacy_lengths: tuple[int, ...] = (),
) -> None:
    raw = d.get(key)
    if raw is None:
        return
    got = [str(x) for x in raw]
    if got == list(expected):
        return
    if len(got) in legacy_lengths:
        return
    raise ValueError(
        f"{key}: fitted coefficient names do not match 5_fit_vanilla_testbed.py. "
        f"Expected {list(expected)!r}, got {got!r}."
    )


def _finite_mean_or_default(x: np.ndarray, default: float = 0.0) -> float:
    vals = np.asarray(x, dtype=float).ravel()
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float(default)
    return float(np.mean(finite))


def _fill_nan_with_finite_mean(x: np.ndarray, default: float = 0.0) -> np.ndarray:
    vals = np.asarray(x, dtype=float).copy()
    if np.all(np.isfinite(vals)):
        return vals
    return np.where(np.isfinite(vals), vals, _finite_mean_or_default(vals, default))


def build_CAE_features(CAE_lastweek, week_norm, foursc_wk, antic_wk) -> np.ndarray:
    """Feature vector matching ``THETA_CAE_NAMES`` in ``5_fit_vanilla_testbed.py``."""
    foursc = np.asarray(foursc_wk, dtype=float).ravel()
    if foursc.size != 14:
        raise ValueError(f"foursc_wk must have 14 weekly decision slots, got {foursc.size}")
    foursc = _fill_nan_with_finite_mean(foursc)

    antic = np.asarray(antic_wk, dtype=float).ravel()
    if antic.size == 14:
        # Fitter converts 14 AM/PM decision rows into 7 daily means.
        antic = _fill_nan_with_finite_mean(antic).reshape(7, 2).mean(axis=1)
    elif antic.size == 7:
        antic = _fill_nan_with_finite_mean(antic)
    else:
        raise ValueError(
            f"antic_wk must have 7 daily values or 14 AM/PM slots, got {antic.size}"
        )

    x = np.concatenate(
        [
            np.array([1.0, CAE_lastweek, week_norm], dtype=float),
            foursc,
            antic,
        ]
    )
    if x.size != P_CAE:
        raise RuntimeError(f"CAE feature length {x.size} != {P_CAE}")
    return x


def build_CAE_short_features(caeAverage) -> np.ndarray:
    """Feature vector matching ``THETA_CAE_SHORT_AVG_NAMES``."""
    return np.array([1.0, caeAverage], dtype=float)


def trim_theta_interaction(theta, *, name: str = "theta") -> np.ndarray:
    """Accept 4-dim interaction fits or legacy 5-dim (with salience 7d row)."""
    a = np.asarray(theta, dtype=float).ravel()
    if a.size == 4:
        return a
    if a.size == 5:
        return np.delete(a, _LEGACY_INTERACT_DROP)
    raise ValueError(f"{name} length {a.size}; expected 4 or legacy 5")


def trim_theta_foursc(theta) -> np.ndarray:
    """Accept current fits or trim legacy salience predictors."""
    a = np.asarray(theta, dtype=float).ravel()
    if a.size == P_FOURSC:
        return a
    if a.size == P_FOURSC + 2:
        return np.delete(a, _LEGACY_FOURSC_SALIENCE_DROP)
    raise ValueError(
        f"theta_fourSC length {a.size}; expected {P_FOURSC} or legacy {P_FOURSC + 2}"
    )


def trim_theta_antic(theta) -> np.ndarray:
    """Accept current fits or trim legacy salience predictors."""
    a = np.asarray(theta, dtype=float).ravel()
    if a.size == P_ANTIC:
        return a
    if a.size == P_ANTIC + 3:
        return np.delete(a, _LEGACY_ANTIC_SALIENCE_DROP)
    raise ValueError(
        f"theta_antic length {a.size}; expected {P_ANTIC} or legacy {P_ANTIC + 3}"
    )


def _json_resid_list(key: str, d: dict) -> np.ndarray:
    raw = d.get(key, [])
    if not raw:
        return np.array([], dtype=float)
    nums = []
    for x in raw:
        if x is None:
            nums.append(np.nan)
        else:
            nums.append(float(x))
    return np.asarray(nums, dtype=float)


def _split_ml_pv_full(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(theta, dtype=float).ravel()
    base = t[:PV_ML_BASE].copy()
    if t.size >= PV_ML_BASE + PV_ML_QUERY:
        q = t[PV_ML_BASE : PV_ML_BASE + PV_ML_QUERY].copy()
    else:
        q = np.zeros(PV_ML_QUERY, dtype=float)
    return base, q


def _split_ml_fw(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(theta, dtype=float).ravel()
    base = t[:FW_ML_BASE].copy()
    if t.size >= FW_ML_BASE + FW_ML_QUERY:
        q = t[FW_ML_BASE : FW_ML_BASE + FW_ML_QUERY].copy()
    else:
        q = np.zeros(FW_ML_QUERY, dtype=float)
    return base, q


def _split_ml_pj(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(theta, dtype=float).ravel()
    base = t[:PJ_ML_BASE].copy()
    if t.size >= PJ_ML_BASE + PJ_ML_QUERY:
        q = t[PJ_ML_BASE : PJ_ML_BASE + PJ_ML_QUERY].copy()
    else:
        q = np.zeros(PJ_ML_QUERY, dtype=float)
    return base, q


# %%
class EnvConfig:
    """Load environment parameters from ``params_env_{userid}.json``."""

    def __init__(self, userid, params_dir=PARAMS_DIR, nweek=36):
        self.userid = userid
        self.K = 2
        self.W = 7
        self.nweek = int(nweek)
        self.D = self.nweek * self.W

        params_path = Path(params_dir).expanduser().resolve()
        self.params_dir = params_path
        with open(params_path / "std_params.json", encoding="utf-8") as f:
            std = json.load(f)
        self.limits_fourSC = std["4hour_step_count_limit"]
        self.limits_pageview = std["HourlyPageviewCount_limit"]
        self.limits_CAE = std["CAE_avg_limit"]
        self.limits_CAE_short = std["CAE_short_avg_limit"]
        # Affine map back to the raw (pre-normalization) CAE level:
        #   raw_CAE = CAE_shift + CAE_scale * CAE_avg_norm
        self.CAE_shift = float(std["CAE_avg_shift"])
        self.CAE_scale = float(std["CAE_avg_scale"])
        self.limits_antic = std["anticipated_affect_yesterday_limit"]
        self.limits_fitbitwearing = [0.0, 1.0]
        self.limits_dailysurvey = [0.0, 1.0]
        self.limits_perceivedUtility = [-2.0, 2.0]
        self.limits_week_present = [0.0, 1.0]
        self.limits_prior2hour_step_count = std["prior2hour_step_count_limit"]
        self.limits_active_status = [0.0, 1.0]
        self.limits_ws_interaction = [0.0, 1.0]
        # Tool surveys live on the [(0+1)/8, (7+1)/8] = [0.375, 1.0] normalized scale
        # (from ``1.5_standardization``: Exp-tool-i_norm = (Exp-tool-i + 1) / 8).
        # self.limits_exp1 = std["exp1_limit"]
        self.limits_exp1 = [0.125, 1.0]
        # self.limits_exp2 = std["exp2_limit"]
        self.limits_exp2 = [0.125, 1.0]
        with open(params_path / f"params_env_{userid}.json", encoding="utf-8") as f:
            p = json.load(f)

        _validate_json_names(
            "theta_prior2hour_step_count_names",
            p,
            THETA_PRIOR2HOUR_STEP_COUNT_NAMES,
        )
        _validate_json_names("theta_active_status_names", p, THETA_ACTIVE_STATUS_NAMES)
        _validate_json_names(
            "theta_ws_interaction_names",
            p,
            THETA_WS_INTERACTION_NAMES,
            legacy_lengths=(5,),
        )
        _validate_json_names(
            "theta_fourSC_names",
            p,
            THETA_FOURSC_NAMES,
            legacy_lengths=(P_FOURSC + 2,),
        )
        _validate_json_names(
            "theta_antic_names",
            p,
            THETA_ANTIC_NAMES,
            legacy_lengths=(P_ANTIC + 3,),
        )
        _validate_json_names("theta_CAE_names", p, THETA_CAE_NAMES)
        _validate_json_names("theta_CAE_short_avg_names", p, THETA_CAE_SHORT_AVG_NAMES)

        self.theta_prior2hour_step_count = _json_float_list("theta_prior2hour_step_count", p)
        if p.get("theta_active_status"):
            self.theta_active_status = _json_float_list("theta_active_status", p)
        else:
            self.theta_active_status = np.zeros(P_ACTIVE_STATUS, dtype=float)
        self.theta_ws_interaction = trim_theta_interaction(
            _json_float_list("theta_ws_interaction", p), name="theta_ws_interaction"
        )
        self.theta_fourSC = trim_theta_foursc(_json_float_list("theta_fourSC", p))
        self.theta_antic = trim_theta_antic(_json_float_list("theta_antic", p))
        self.theta_CAE = _json_float_list("theta_CAE", p)
        self.theta_CAE_short = _json_float_list("theta_CAE_short_avg", p)

        self.resid_prior2hour_step_count = _json_resid_list("resid_prior2hour_step_count", p)
        self.resid_active_status = _json_resid_list("resid_active_status", p)
        self.resid_ws_interaction = _json_resid_list("resid_ws_interaction", p)
        self.resid_fourSC = _json_resid_list("resid_fourSC", p)
        self.resid_antic = _json_resid_list("resid_antic", p)
        self.resid_CAE = _json_resid_list("resid_CAE", p)
        self.resid_week_present = _json_resid_list("resid_week_present", p)
        self.resid_CAE_short = _json_resid_list("resid_CAE_short_avg", p)

        def _penalized_or_legacy(new_key, old_key):
            return np.asarray(
                p.get(new_key) or p.get(old_key) or [],
                dtype=float,
            ).ravel()

        self.theta_ml_Ew = _penalized_or_legacy("theta_penalized_Ew", "theta_ml_Ew")
        self.theta_ml_J = _penalized_or_legacy("theta_penalized_J", "theta_ml_J")
        self.theta_ml_U1 = _penalized_or_legacy("theta_penalized_U1", "theta_ml_U1")
        self.theta_ml_U2 = _penalized_or_legacy("theta_penalized_U2", "theta_ml_U2")
        self.theta_ml_PV = _penalized_or_legacy("theta_penalized_PV", "theta_ml_PV")
        self.theta_ml_FW = _penalized_or_legacy("theta_penalized_FW", "theta_ml_FW")
        self.theta_ml_PJ = _penalized_or_legacy("theta_penalized_PJ", "theta_ml_PJ")

        def _penalized_resid_or_legacy(new_key, old_key):
            return _json_resid_list(new_key if p.get(new_key) is not None else old_key, p)

        self.resid_ml_J_week = _penalized_resid_or_legacy(
            "resid_penalized_J_week", "resid_ml_J_week"
        )
        self.resid_ml_U1 = _penalized_resid_or_legacy(
            "resid_penalized_U1", "resid_ml_U1"
        )
        self.resid_ml_U2 = _penalized_resid_or_legacy(
            "resid_penalized_U2", "resid_ml_U2"
        )
        self.resid_ml_hourly_pageview = _penalized_resid_or_legacy(
            "resid_penalized_hourly_pageview", "resid_ml_hourly_pageview"
        )
        self.resid_ml_nextday_wearing = _penalized_resid_or_legacy(
            "resid_penalized_nextday_wearing", "resid_ml_nextday_wearing"
        )
        self.resid_ml_daily_present = _penalized_resid_or_legacy(
            "resid_penalized_daily_present", "resid_ml_daily_present"
        )

        self.has_ml_stack = bool(
            self.theta_ml_Ew.size >= 6
            and self.theta_ml_J.size >= 2
            and self.theta_ml_U1.size >= 3
            and self.theta_ml_U2.size >= 3
            and self.theta_ml_PV.size >= PV_ML_BASE
            and self.theta_ml_FW.size >= FW_ML_BASE
            and self.theta_ml_PJ.size >= PJ_ML_BASE
        )
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        expected = {
            "theta_prior2hour_step_count": P_PRIOR2HOUR,
            "theta_active_status": P_ACTIVE_STATUS,
            "theta_ws_interaction": 4,
            "theta_fourSC": P_FOURSC,
            "theta_antic": P_ANTIC,
            "theta_CAE": P_CAE,
            "theta_CAE_short": P_CAE_SHORT,
        }

        for name, n in expected.items():
            arr = getattr(self, name)
            if arr.shape[0] == 0:
                continue
            if arr.shape[0] != n:
                raise ValueError(
                    f"{name} length {arr.shape[0]} != {n}; "
                    "re-run the fitting scripts and regenerate params JSON."
                )

        if self.theta_ml_Ew.size and self.theta_ml_Ew.shape[0] < 6:
            raise ValueError("theta_ml_Ew must have length >= 6")

        if self.theta_ml_PV.size and self.theta_ml_PV.shape[0] < PV_ML_BASE:
            raise ValueError(f"theta_ml_PV must have length >= {PV_ML_BASE}")

        if self.theta_ml_FW.size and self.theta_ml_FW.shape[0] < FW_ML_BASE:
            raise ValueError(f"theta_ml_FW must have length >= {FW_ML_BASE}")

        if self.theta_ml_PJ.size and self.theta_ml_PJ.shape[0] < PJ_ML_BASE:
            raise ValueError(f"theta_ml_PJ must have length >= {PJ_ML_BASE}")


# %%
class Env:
    """Generative environment: vanilla Ridge/logistic + ML perceived-utility stack."""

    def __init__(self, env_config: EnvConfig, noise="sequential"):
        assert noise in ("sequential", "random")
        self.noise = noise
        self.cfg = env_config
        self.K = env_config.K
        self.W = env_config.W

    def _sample_noise(self, resid, idx, obs_resid=None):
        if obs_resid is None:
            obs_resid = resid[~np.isnan(resid)]
        if len(obs_resid) == 0:
            return 0.0
        if self.noise == "sequential":
            val = resid[idx % len(resid)]
            return float(val) if not np.isnan(val) else float(rd.choice(obs_resid))
        return float(rd.choice(obs_resid))

    @staticmethod
    def _sigmoid(eta):
        z = float(np.clip(eta, -60.0, 60.0))
        return float(1.0 / (1.0 + np.exp(-z)))

    # ----- decision-level (K=2) -----

    def gen_prior2hour_step_count_mean(self, s):
        """[1, EMA_Prior2HourStepCount, isWeekend, decisionTimeSlot]."""
        X = np.array(
            [
                1.0,
                s["prior2HourStepCountEma7d"],
                s["isWeekend"],
                s["decisionTimeSlot"],
            ],
            dtype=float,
        )
        return float(self.cfg.theta_prior2hour_step_count @ X)

    def gen_prior2hour_step_count(self, s, step_idx):
        mean = self.gen_prior2hour_step_count_mean(s)
        noise = self._sample_noise(self.cfg.resid_prior2hour_step_count, step_idx)
        return float(np.clip(mean + noise, *self.cfg.limits_prior2hour_step_count))

    def gen_active_status_mean(self, s, return_logit=False):
        """Daily morning model: [1, activeDaysLast7Days, isWeekend]."""
        X = np.array(
            [
                1.0,
                s["activeDaysLast7Days"],
                s["isWeekend"],
            ],
            dtype=float,
        )
        eta = float(self.cfg.theta_active_status @ X)
        return eta if return_logit else self._sigmoid(eta)

    def gen_active_status(self, s, day_idx):
        eta = self.gen_active_status_mean(s, return_logit=True)
        base_p = self._sigmoid(eta)
        noise = self._sample_noise(self.cfg.resid_active_status, day_idx)
        p = float(np.clip(base_p + noise, *self.cfg.limits_active_status))
        return float(rd.binomial(1, p))

    def gen_ws_interaction_mean(self, s, return_logit=False):
        """[1, activitySuggestionInteractLast7Days, isWeekend, decisionTimeSlot]."""
        X = np.array(
            [
                1.0,
                s["activitySuggestionInteractLast7Days"],
                s["isWeekend"],
                s["decisionTimeSlot"],
            ],
            dtype=float,
        )
        eta = float(self.cfg.theta_ws_interaction @ X)
        return eta if return_logit else self._sigmoid(eta)

    def gen_ws_interaction(self, s, step_idx):
        eta = self.gen_ws_interaction_mean(s, return_logit=True)
        base_p = self._sigmoid(eta)
        noise = self._sample_noise(self.cfg.resid_ws_interaction, step_idx)
        p = float(np.clip(base_p + noise, *self.cfg.limits_ws_interaction))
        return float(rd.binomial(1, p))

    def gen_fourSC_mean(self, s, Ah):
        """
        ``P_FOURSC`` predictors — matches ``5_fit_vanilla_testbed`` fourSC_cond.
        ``yesterdayStepCount`` is the sum of the prior day's two 4-hour slots.
        """
        wear7 = float(s.get("morningFitbitWearLast7Days", s["morningFitbitWearLast7DaysAlt"]))
        pv7 = float(s["pageViewLast7DaysEma"])
        antic_y = float(s["dailyAnticipatedAffectYesterday"])
        pu = float(s["perceivedUtilityLastWeek"])
        cae = float(s["caeAverageLastWeek"])
        X = np.array(
            [
                1.0,
                s["stepCountNext4HourLag1"],
                s["yesterdayStepCount"],
                s["stepCountLast7DaysEma"],
                s["prior2HourStepCount"],
                s["activitySuggestionsSentLast7Days"],
                pv7,
                wear7,
                s["activitySuggestionInteractLast7Days"],
                antic_y,
                s["activeDaysLast7Days"],
                s["isWeekend"],
                s["decisionTimeSlot"],
                pu,
                cae,
                Ah,
                Ah * s["yesterdayStepCount"],
                Ah * s["prior2HourStepCount"],
                Ah * s["activitySuggestionsSentLast7Days"],
                Ah * pv7,
                Ah * wear7,
                Ah * s["activitySuggestionInteractLast7Days"],
                Ah * antic_y,
                Ah * s["isWeekend"],
                Ah * s["decisionTimeSlot"],
                Ah * pu,
                Ah * cae,
            ],
            dtype=float,
        )
        return float(self.cfg.theta_fourSC @ X)

    def gen_fourSC(self, s, Ah, step_idx):
        mean = self.gen_fourSC_mean(s, Ah)
        noise = self._sample_noise(self.cfg.resid_fourSC, step_idx)
        return float(np.clip(mean + noise, *self.cfg.limits_fourSC))

    def _ml_pv_mean(self, s, Ah: float, Iw: int) -> float:
        if not self.cfg.has_ml_stack:
            raise RuntimeError("ML parameters missing from params JSON")
        Ew = float(s["perceivedUtilityLastWeek"])
        base, q = _split_ml_pv_full(self.cfg.theta_ml_PV)
        (
            alpha0,
            alpha1,
            a2_isWeekend,
            a2_dt,
            a2_rb,
            alpha_ar1,
            alpha3,
            alpha4,
            _sigma_pv,
        ) = base
        lag1 = float(s["pageViewNext4HourLag1"])
        mu = (
            alpha0
            + alpha1 * Ew
            + a2_isWeekend * float(s["isWeekend"])
            + a2_dt * float(s["decisionTimeSlot"])
            + a2_rb * float(s["activitySuggestionsSentLast7Days"])
            + alpha_ar1 * lag1
            + Ah * (alpha3 + alpha4 * Ew)
        )
        if Iw != 0:
            xq = np.array([1.0, Ew, float(s["isWeekend"]), float(s["decisionTimeSlot"]), float(s["activitySuggestionsSentLast7Days"])])
            mu += float(Iw * (q @ xq))
        return float(mu)

    def gen_pageview_mean(self, s, Ah, Iw=0):
        return self._ml_pv_mean(s, float(Ah), int(Iw))

    def gen_pageview(self, s, Ah, Iw, step_idx):
        mu = self._ml_pv_mean(s, float(Ah), int(Iw))
        base, _ = _split_ml_pv_full(self.cfg.theta_ml_PV)
        sigma = float(base[8]) if base.size > 8 else 0.1
        if self.cfg.resid_ml_hourly_pageview.size > 0 and np.any(np.isfinite(self.cfg.resid_ml_hourly_pageview)):
            noise = self._sample_noise(self.cfg.resid_ml_hourly_pageview, step_idx)
        else:
            noise = float(rd.normal(0.0, sigma))
        return float(np.clip(mu + noise, *self.cfg.limits_pageview))

    # ----- daily mediators -----

    def gen_antic_mean(self, s, ws_morning, ws_afternoon):
        """
        Daily ridge design (morning row): 15 columns — matches
        ``5_fit_vanilla_testbed`` ``anticipated_affect_cond_day``
        (no ``planning_prompt``; includes ``perceivedUtilityLastWeek`` main and
        AM/PM interactions).
        """
        ys = float(s["todayStepCount"])
        act = float(s["activityStatusToday"])
        is_weekend = float(s["isWeekend"])
        pu = float(s["perceivedUtilityLastWeek"])
        cae = float(s["caeAverageLastWeek"])
        X = np.array(
            [
                1.0,
                float(s["dailyAnticipatedAffectYesterday"]),
                ys,
                act,
                is_weekend,
                pu,
                cae,
                ws_morning,
                ws_afternoon,
                ws_morning * is_weekend,
                ws_afternoon * is_weekend,
                ws_morning * pu,
                ws_afternoon * pu,
                ws_morning * cae,
                ws_afternoon * cae,
            ],
            dtype=float,
        )
        return float(self.cfg.theta_antic @ X)

    def gen_antic(self, s, ws_morning, ws_afternoon, day_idx):
        mean = self.gen_antic_mean(s, ws_morning, ws_afternoon)
        noise = self._sample_noise(self.cfg.resid_antic, day_idx)
        return float(np.clip(mean + noise, *self.cfg.limits_antic))

    def _ml_fw_eta(self, s, ws_morning, ws_afternoon, Iw: int, return_logit: bool):
        if not self.cfg.has_ml_stack:
            raise RuntimeError("ML parameters missing from params JSON")
        Ew = float(s["perceivedUtilityLastWeek"])
        base, q = _split_ml_fw(self.cfg.theta_ml_FW)
        beta0, beta1, b2_isWeekend, b2_rb, beta_ar1, beta3, beta4, beta5, beta6 = base
        is_weekend = float(s["isWeekend"])
        rb = float(s["activitySuggestionsSentLast7Days"])
        y_lag = float(s["morningFitbitWearYesterday"])
        eta = (
            beta0
            + beta1 * Ew
            + b2_isWeekend * is_weekend
            + b2_rb * rb
            + beta_ar1 * y_lag
            + ws_morning * (beta3 + beta4 * Ew)
            + ws_afternoon * (beta5 + beta6 * Ew)
        )
        if Iw != 0:
            xq = np.array([1.0, Ew, is_weekend, rb])
            eta += float(Iw * (q @ xq))
        return eta if return_logit else self._sigmoid(eta)

    def gen_fitbitwearing_mean(self, s, ws_morning, ws_afternoon, Iw=0, return_logit=False):
        return self._ml_fw_eta(s, ws_morning, ws_afternoon, int(Iw), return_logit)

    def gen_fitbitwearing(self, s, ws_morning, ws_afternoon, Iw, day_idx):
        eta = self._ml_fw_eta(s, ws_morning, ws_afternoon, int(Iw), return_logit=True)
        base_p = self._sigmoid(eta)
        noise = self._sample_noise(self.cfg.resid_ml_nextday_wearing, day_idx)
        if not np.isfinite(noise):
            noise = 0.0
        p = float(np.clip(base_p + noise, *self.cfg.limits_fitbitwearing))
        return float(rd.binomial(1, p))

    def _ml_pj_eta(self, s, ws_morning, ws_afternoon, Iw: int, return_logit: bool):
        if not self.cfg.has_ml_stack:
            raise RuntimeError("ML parameters missing from params JSON")
        Ew = float(s["perceivedUtilityLastWeek"])
        base, q = _split_ml_pj(self.cfg.theta_ml_PJ)
        t0, t1, t2_isWeekend, t2_rb, t_ar1, t3, t4, t5, t6 = base
        is_weekend = float(s["isWeekend"])
        rb = float(s["activitySuggestionsSentLast7Days"])
        y_lag = float(s["dailySurveyCompleteYesterday"])
        eta = (
            t0
            + t1 * Ew
            + t2_isWeekend * is_weekend
            + t2_rb * rb
            + t_ar1 * y_lag
            + ws_morning * (t3 + t4 * Ew)
            + ws_afternoon * (t5 + t6 * Ew)
        )
        if Iw != 0:
            xq = np.array([1.0, Ew, is_weekend, rb])
            eta += float(Iw * (q @ xq))
        return eta if return_logit else self._sigmoid(eta)

    def gen_dailysurvey_mean(self, s, ws_morning, ws_afternoon, Iw=0, return_logit=False):
        return self._ml_pj_eta(s, ws_morning, ws_afternoon, int(Iw), return_logit)

    def gen_dailysurvey(self, s, ws_morning, ws_afternoon, Iw, day_idx):
        eta = self._ml_pj_eta(s, ws_morning, ws_afternoon, int(Iw), return_logit=True)
        base_p = self._sigmoid(eta)
        noise = self._sample_noise(self.cfg.resid_ml_daily_present, day_idx)
        if not np.isfinite(noise):
            noise = 0.0
        p = float(np.clip(base_p + noise, *self.cfg.limits_dailysurvey))
        return float(rd.binomial(1, p))

    # ----- weekly -----

    def gen_CAE_mean(self, CAE_lastweek, week_norm, foursc_wk, antic_wk):
        X = build_CAE_features(CAE_lastweek, week_norm, foursc_wk, antic_wk)
        return float(self.cfg.theta_CAE @ X)

    def gen_CAE(self, CAE_lastweek, week_norm, foursc_wk, antic_wk, week_idx):
        mean = self.gen_CAE_mean(CAE_lastweek, week_norm, foursc_wk, antic_wk)
        noise = self._sample_noise(self.cfg.resid_CAE, week_idx)
        return float(np.clip(mean + noise, *self.cfg.limits_CAE))

    @staticmethod
    def _week_means_from_arrays(pw_wk, dw_wk, dp_wk):
        pw = np.asarray(pw_wk, dtype=float).ravel()
        if pw.size == 14:
            pv_m = float(np.nansum(pw) / 14.0)
        else:
            pv_m = float(np.nanmean(pw)) if pw.size else 0.0
        dw = np.asarray(dw_wk, dtype=float).ravel()
        pj = np.asarray(dp_wk, dtype=float).ravel()
        fw_m = float(np.nansum(dw) / 7.0) if dw.size else 0.0
        pj_m = float(np.nansum(pj) / 7.0) if pj.size else 0.0
        return pv_m, fw_m, pj_m

    def gen_perceivedUtility_mean(self, Ew_carry, week_norm, pw_wk, dw_wk, dp_wk):
        """Gaussian AR on ``E_w``; ``week_norm`` kept for API compatibility (unused)."""
        del week_norm
        if self.cfg.theta_ml_Ew.size < 6:
            raise RuntimeError("theta_ml_Ew required for latent E_w transition")
        a0, a1, a2, a3, a4, _sigma_e = [float(x) for x in self.cfg.theta_ml_Ew[:6]]
        pv_m, fw_m, pj_m = self._week_means_from_arrays(pw_wk, dw_wk, dp_wk)
        return float(a0 + a1 * float(Ew_carry) + a2 * pv_m + a3 * fw_m + a4 * pj_m)

    def gen_perceivedUtility(self, Ew_carry, week_norm, pw_wk, dw_wk, dp_wk, week_idx):
        del week_idx
        mean = self.gen_perceivedUtility_mean(Ew_carry, week_norm, pw_wk, dw_wk, dp_wk)
        sigma_e = float(self.cfg.theta_ml_Ew[5]) if self.cfg.theta_ml_Ew.size > 5 else 0.1
        noise = float(rd.normal(0.0, sigma_e))
        return float(np.clip(mean + noise, *self.cfg.limits_perceivedUtility))

    def gen_week_present_mean(self, perceivedUtility, return_logit=False):
        if self.cfg.theta_ml_J.size < 2:
            raise RuntimeError("theta_ml_J required for week_present (J_week)")
        b0, b1 = float(self.cfg.theta_ml_J[0]), float(self.cfg.theta_ml_J[1])
        eta = b0 + b1 * float(perceivedUtility)
        return eta if return_logit else self._sigmoid(eta)

    def gen_week_present(self, perceivedUtility, week_idx):
        eta = self.gen_week_present_mean(perceivedUtility, return_logit=True)
        base_p = self._sigmoid(eta)
        noise = self._sample_noise(self.cfg.resid_ml_J_week, week_idx)
        if not np.isfinite(noise):
            noise = self._sample_noise(self.cfg.resid_week_present, week_idx)
        lim = self.cfg.limits_week_present
        p = float(np.clip(base_p + noise, *lim))
        return float(rd.binomial(1, p))

    def gen_CAE_short_mean(self, caeAverage):
        return float(self.cfg.theta_CAE_short @ build_CAE_short_features(caeAverage))

    def gen_CAE_short(self, caeAverage, week_idx):
        mean = self.gen_CAE_short_mean(caeAverage)
        noise = self._sample_noise(self.cfg.resid_CAE_short, week_idx)
        return float(np.clip(mean + noise, *self.cfg.limits_CAE_short))

    # ----- weekly tool surveys (Exp-tool-1 / Exp-tool-2) -----
    # Used both as observed weekly outcomes when ``J_w == 1`` and as inputs to the
    # agent's pooled-linear E_w approximation in ``OnlineEnv`` (``est_Ew_weights``).
    #
    # ``theta_ml_U1`` / ``theta_ml_U2`` were fit on the *normalized* scale
    #   Exp-tool-i_norm = (Exp-tool-i + 1) / 8   (see ``1.5_standardization.py``)
    # so we sample + clip in normalized space (``limits_exp{1,2}``) and then
    # transform back to the original integer 0..7 Likert scale via
    #   raw = round(8 * norm - 1), clipped to [0, 7].
    # In practice the observed [0.375, 1.0] norm support yields raw in {2..7}.

    @staticmethod
    def _norm_to_raw_int(norm):
        """Inverse of ``Exp-tool-i_norm = (Exp-tool-i + 1) / 8`` rounded to int.

        ``Exp-tool-i`` is an integer Likert response on the 0..7 scale, so we
        round the de-standardized value and then clip to ``[0, 7]`` for safety.
        """
        return int(np.clip(np.rint(8.0 * float(norm) - 1.0), 0, 7))

    def gen_tool_U1_mean(self, Ew):
        """Mean of ``Exp-tool-1`` on the *raw* 0..7 Likert scale (integer)."""
        if self.cfg.theta_ml_U1.size < 3:
            raise RuntimeError("theta_ml_U1 missing")
        c0, c1, _sig = map(float, self.cfg.theta_ml_U1[:3])
        return self._norm_to_raw_int(c0 + c1 * Ew)

    def gen_tool_U1(self, Ew, week_idx):
        """Sample ``Exp-tool-1`` on the *raw* 0..7 Likert scale (integer)."""
        if self.cfg.theta_ml_U1.size < 3:
            raise RuntimeError("theta_ml_U1 missing")
        c0, c1, _sig = map(float, self.cfg.theta_ml_U1[:3])
        sig = float(self.cfg.theta_ml_U1[2])
        if self.cfg.resid_ml_U1.size > 0 and np.any(np.isfinite(self.cfg.resid_ml_U1)):
            noise = self._sample_noise(self.cfg.resid_ml_U1, week_idx)
        else:
            noise = float(rd.normal(0.0, sig))
        norm = float(np.clip(c0 + c1 * Ew + noise, *self.cfg.limits_exp1))
        return self._norm_to_raw_int(norm)

    def gen_tool_U2_mean(self, Ew):
        """Mean of ``Exp-tool-2`` on the *raw* 0..7 Likert scale (integer)."""
        if self.cfg.theta_ml_U2.size < 3:
            raise RuntimeError("theta_ml_U2 missing")
        d0, d1, _sig = map(float, self.cfg.theta_ml_U2[:3])
        return self._norm_to_raw_int(d0 + d1 * Ew)

    def gen_tool_U2(self, Ew, week_idx):
        """Sample ``Exp-tool-2`` on the *raw* 0..7 Likert scale (integer)."""
        if self.cfg.theta_ml_U2.size < 3:
            raise RuntimeError("theta_ml_U2 missing")
        d0, d1, _sig = map(float, self.cfg.theta_ml_U2[:3])
        sig = float(self.cfg.theta_ml_U2[2])
        if self.cfg.resid_ml_U2.size > 0 and np.any(np.isfinite(self.cfg.resid_ml_U2)):
            noise = self._sample_noise(self.cfg.resid_ml_U2, week_idx)
        else:
            noise = float(rd.normal(0.0, sig))
        norm = float(np.clip(d0 + d1 * Ew + noise, *self.cfg.limits_exp2))
        return self._norm_to_raw_int(norm)


def _float_csv_cell(row, *candidates, default=0.0):
    for key in candidates:
        if key not in row:
            continue
        raw = row[key]
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        return v
    return default


def _read_first_csv_row(path):
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            return row
    return None


def _participant_id_column(fieldnames):
    if not fieldnames:
        return None
    lower_map = {fn.lower(): fn for fn in fieldnames if fn}
    for cand in ("participantidentifier", "participant_id", "userid", "user_id"):
        if cand in lower_map:
            return lower_map[cand]
    return None


def _read_first_csv_row_for_participant(path, participant_id):
    if participant_id is None:
        return _read_first_csv_row(path)
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        fieldnames = list(rdr.fieldnames or [])
        rows = list(rdr)
    if not rows:
        return None
    id_key = _participant_id_column(fieldnames)
    if not id_key:
        return None
    try:
        want = int(participant_id)
    except (TypeError, ValueError):
        want = str(participant_id).strip()
    for row in rows:
        raw = row.get(id_key)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            continue
        try:
            if int(float(str(raw).strip())) == int(want):
                return row
        except (TypeError, ValueError):
            if str(raw).strip() == str(want):
                return row
    return None


def _read_last_csv_row_for_participant(path, participant_id):
    if participant_id is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        fieldnames = list(rdr.fieldnames or [])
        rows = list(rdr)
    if not rows:
        return None
    id_key = _participant_id_column(fieldnames)
    if not id_key:
        return None
    try:
        want = int(participant_id)
    except (TypeError, ValueError):
        want = str(participant_id).strip()
    last_match = None
    for row in rows:
        raw = row.get(id_key)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            continue
        try:
            if int(float(str(raw).strip())) == int(want):
                last_match = row
        except (TypeError, ValueError):
            if str(raw).strip() == str(want):
                last_match = row
    return last_match


_STATE_FROM_DF_FIT_ROW = (
    ("stepCountNext4HourLag1", ("FourSC_lag1",)),
    ("pageViewNext4HourLag1", ("hourly_pageview_count_lag1",)),
    ("prior2HourStepCountEma7d", ("EMA_Prior2HourStepCount_norm", "EMA_Prior2HourStepCount")),
    ("prior2HourStepCountLag1", ("prior2HourStepCountLag1",)),
    ("prior2HourStepCount", ("prior2HourStepCount_norm", "prior2HourStepCount")),
    ("yesterdayStepCount", ("YesterdayStepCount_norm", "YesterdayStepCount")),
    ("stepCountLast7DaysEma", ("EMA_StepCount_norm", "EMA_StepCount")),
    ("pageViewLast7DaysEma", ("Past7DaysPageviewEMA_norm", "Past7DaysPageviewEMA")),
    ("morningFitbitWearLast7DaysAlt", ("morningFitbitWearLast7Days", )),
    ("morningFitbitWearLast7Days", ("morningFitbitWearLast7Days",)),
    ("dailySurveyComplete", ("dailySurveyComplete",)),
    ("activeDaysLast7Days", ("active_status_fraction_7days", "activeDaysLast7Days")),
    ("activitySuggestionsSentLast7Days", (
        "recent_burden_norm",
        "recentBurdenEma_norm",
        "recentBurdenEma",
        "activitySuggestionsSentLast7Days",
    )),
    ("activitySuggestionInteractLast7Days", ("activitySuggestionInteractLast7Days",)),
    ("dailyAnticipatedAffectYesterday", ("dailyAnticipatedAffectYesterday_norm", "dailyAnticipatedAffectYesterday")),
    ("dailyAnticipatedAffect", ("dailyAnticipatedAffect_norm", "dailyAnticipatedAffect")),
    ("caeAverageLastWeek", ("caeAverageLastWeek_norm", "caeAverageLastWeek")),
    ("morningFitbitWearYesterday", ("morning_wearing",)),
    ("dailySurveyCompleteYesterday", ("dailySurveyComplete_yesterday",)),
    ("salienceMessageSentYesterday", ("yesterday_SalienceMessage",)),
    ("activityStatusToday", ("active_status",)),
)


def _state_updates_from_df_fit_row(row):
    if row is None:
        return {}
    out = {}
    for skey, csv_keys in _STATE_FROM_DF_FIT_ROW:
        out[skey] = _float_csv_cell(row, *csv_keys, default=0.0)
    if "morningFitbitWearLast7Days" not in out or out.get("morningFitbitWearLast7Days", 0) == 0:
        out["morningFitbitWearLast7Days"] = out.get("morningFitbitWearLast7DaysAlt", 0.0)
    yh = _float_csv_cell(
        row,
        "yesterday_hourly_pageview_count_norm",
        "yesterday_hourly_pageview_count",
        default=float("nan"),
    )
    if math.isfinite(yh):
        out["pageViewMorningYesterday"] = yh
        out["pageViewAfternoonYesterday"] = yh
    else:
        hm = _float_csv_cell(row, "pageViewMorningYesterday", default=float("nan"))
        ha = _float_csv_cell(row, "pageViewAfternoonYesterday", default=float("nan"))
        if math.isfinite(hm):
            out["pageViewMorningYesterday"] = hm
        if math.isfinite(ha):
            out["pageViewAfternoonYesterday"] = ha
    return out


def _resolve_df_fit_11week_path(explicit_path=None):
    import os

    env_p = os.environ.get("ADAPR_DF_FIT_11WEEK", "").strip()
    if env_p:
        return Path(env_p)
    if explicit_path:
        return Path(explicit_path)
    return PARAMS_DIR / "df_fit_11week.csv"


def make_initial_state(df_fit_11week_csv=None, participant_id=None):
    s = {
        "stepCountNext4HourLag1": 0.0,
        "pageViewNext4HourLag1": 0.0,
        "prior2HourStepCount": 0.0,
        "prior2HourStepCountEma7d": 0.0,
        "prior2HourStepCountLag1": 0.0,
        "yesterdayStepCount": 0.0,
        "todayStepCount": 0.0,
        "morningFitbitWearYesterday": 0.0,
        "dailySurveyCompleteYesterday": 0.0,
        "pageViewMorningYesterday": 0.0,
        "pageViewAfternoonYesterday": 0.0,
        "pageViewMorningToday": 0.0,
        "pageViewAfternoonToday": 0.0,
        "stepCountLast7DaysEma": 0.0,
        "pageViewLast7DaysEma": 0.0,
        "morningFitbitWearLast7DaysAlt": 0.0,
        "morningFitbitWearLast7Days": 0.0,
        "morningFitbitWear": 0.0,
        "dailySurveyComplete": 0.0,
        "caeAverageLastWeek": 0.0,
        "perceivedUtilityLastWeek": 2.0,
        "dailyAnticipatedAffect": 0.0,
        "dailyAnticipatedAffectYesterday": 0.0,
        "activeDaysLast7Days": 0.0,
        "activitySuggestionsSentLast7Days": 0.0,
        "activityStatusToday": 0.0,
        "salienceMessageSentToday": 0.0,
        "salienceMessageSentYesterday": 0.0,
        "activitySuggestionInteractLast7Days": 0.0,
        "expTool1": float("nan"),
        "expTool2": float("nan"),
        "isWeekend": 0.0,
        "dayOfWeekNorm": 0.0,
        "dayOfWeekNormLag1": 0.0,
        "decisionTimeSlot": 0.0,
    }
    df_path = _resolve_df_fit_11week_path(df_fit_11week_csv)
    row = _read_first_csv_row_for_participant(df_path, participant_id)
    s.update(_state_updates_from_df_fit_row(row))
    last_row = _read_last_csv_row_for_participant(df_path, participant_id)
    if last_row is not None:
        s["expTool1"] = _float_csv_cell(last_row, "Exp-tool-1", default=float("nan"))
        s["expTool2"] = _float_csv_cell(last_row, "Exp-tool-2", default=float("nan"))
    return s
