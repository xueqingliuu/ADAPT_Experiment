# %%
"""
Generative environment for the vanilla testbed.

**Vanilla mediators** (``2_fit_vanilla_testbed.py``): Ridge / L2 logistic with
``Intercept`` in the design matrix and ``fit_intercept=False`` — the leading
column of ones is part of ``theta_*``, not sklearn's intercept.

**Perceived-utility stack** (``perceived_utility.py``): parameters are merged into
the same ``params_env_<userid>.json`` as ``theta_ml_*`` / ``resid_ml_*``.
Within-week outcomes (hourly PV, daily FW/PJ) and weekly ``week_present`` use
those coefficients conditional on latent ``E_w`` carried in
``state[\"perceived_utility_lastweek\"]``.

Weekly AR transition for ``E_w``::

    E_{w+1} = a0 + a1 E_w + a2 \\bar{PV}_w + a3 \\bar{FW}_w + a4 \\bar{PJ}_w + \\varepsilon,

with :math:`\\varepsilon \\sim N(0, \\sigma_E^2)` (``sigma_E`` from ``theta_ml_Ew``).
Means match ``perceived_utility.transition_matrix`` / ``est_Ew_weights`` conventions:
``\\bar{PV}_w = (1/14)\\sum`` hourly pageviews, ``\\bar{FW}_w`` and ``\\bar{PJ}_w``
are :math:`(1/7)\\sum` over calendar days (one value per day).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import numpy.random as rd

PROJECT_ROOT = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPR-MRT-Testbed")
PARAMS_DIR = PROJECT_ROOT / "env_para_vanilla"

# Design sizes (``2_fit_vanilla_testbed.py``)
P_FOURSC = 31
P_ANTIC = 22
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

        params_path = Path(params_dir)
        with open(params_path / "std_params.json", encoding="utf-8") as f:
            std = json.load(f)
        self.limits_fourSC = std["4hour_step_count_limit"]
        self.limits_pageview = std["HourlyPageviewCount_limit"]
        self.limits_CAE = std["CAE_avg_limit"]
        self.limits_CAE_short = std["CAE_short_avg_limit"]
        self.limits_antic = std["anticipated_affect_yesterday_limit"]
        self.limits_fitbitwearing = [0.0, 1.0]
        self.limits_dailysurvey = [0.0, 1.0]
        self.limits_perceived_utility = [-2.0, 2.0]
        self.limits_week_present = [0.0, 1.0]
        self.limits_prior2hour_step_count = std["prior2hour_step_count_limit"]
        self.limits_recorded_physical_activity = [0.0, 1.0]
        self.limits_ws_interaction = [0.0, 1.0]
        self.limits_salience_interaction = [0.0, 1.0]
        # Tool surveys live on the [(0+1)/8, (7+1)/8] = [0.375, 1.0] normalized scale
        # (from ``1.5_standardization``: Exp-tool-i_norm = (Exp-tool-i + 1) / 8).
        # self.limits_exp1 = std["exp1_limit"]
        self.limits_exp1 = [0.125, 1.0]
        # self.limits_exp2 = std["exp2_limit"]
        self.limits_exp2 = [0.125, 1.0]
        with open(params_path / f"params_env_{userid}.json", encoding="utf-8") as f:
            p = json.load(f)

        self.theta_prior2hour_step_count = _json_float_list("theta_prior2hour_step_count", p)
        self.theta_recorded_physical_activity = _json_float_list("theta_recorded_physical_activity", p)
        self.theta_ws_interaction = _json_float_list("theta_ws_interaction", p)
        self.theta_salience_interaction = _json_float_list("theta_salience_interaction", p)
        self.theta_fourSC = _json_float_list("theta_fourSC", p)
        self.theta_antic = _json_float_list("theta_antic", p)
        self.theta_CAE = _json_float_list("theta_CAE", p)
        self.theta_CAE_short = _json_float_list("theta_CAE_short_avg", p)

        self.resid_prior2hour_step_count = _json_resid_list("resid_prior2hour_step_count", p)
        self.resid_recorded_physical_activity = _json_resid_list("resid_recorded_physical_activity", p)
        self.resid_ws_interaction = _json_resid_list("resid_ws_interaction", p)
        self.resid_salience_interaction = _json_resid_list("resid_salience_interaction", p)
        self.resid_fourSC = _json_resid_list("resid_fourSC", p)
        self.resid_antic = _json_resid_list("resid_antic", p)
        self.resid_CAE = _json_resid_list("resid_CAE", p)
        self.resid_week_present = _json_resid_list("resid_week_present", p)
        self.resid_CAE_short = _json_resid_list("resid_CAE_short_avg", p)

        self.theta_ml_Ew = np.asarray(p.get("theta_ml_Ew") or [], dtype=float).ravel()
        self.theta_ml_J = np.asarray(p.get("theta_ml_J") or [], dtype=float).ravel()
        self.theta_ml_U1 = np.asarray(p.get("theta_ml_U1") or [], dtype=float).ravel()
        self.theta_ml_U2 = np.asarray(p.get("theta_ml_U2") or [], dtype=float).ravel()
        self.theta_ml_PV = np.asarray(p.get("theta_ml_PV") or [], dtype=float).ravel()
        self.theta_ml_FW = np.asarray(p.get("theta_ml_FW") or [], dtype=float).ravel()
        self.theta_ml_PJ = np.asarray(p.get("theta_ml_PJ") or [], dtype=float).ravel()

        self.resid_ml_J_week = _json_resid_list("resid_ml_J_week", p)
        self.resid_ml_U1 = _json_resid_list("resid_ml_U1", p)
        self.resid_ml_U2 = _json_resid_list("resid_ml_U2", p)
        self.resid_ml_hourly_pageview = _json_resid_list("resid_ml_hourly_pageview", p)
        self.resid_ml_nextday_wearing = _json_resid_list("resid_ml_nextday_wearing", p)
        self.resid_ml_daily_present = _json_resid_list("resid_ml_daily_present", p)

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
            "theta_prior2hour_step_count": 4,
            "theta_recorded_physical_activity": 4,
            "theta_ws_interaction": 5,
            "theta_salience_interaction": 5,
            "theta_fourSC": P_FOURSC,
            "theta_antic": P_ANTIC,
            "theta_CAE": 24,
            "theta_CAE_short": 2,
        }

        for name, n in expected.items():
            arr = getattr(self, name)
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
        """[1, lag1, dow, decision_time]."""
        X = np.array(
            [
                1.0,
                s["prior2hour_step_count_lag1"],
                s["dow"],
                s["decision_time"],
            ],
            dtype=float,
        )
        return float(self.cfg.theta_prior2hour_step_count @ X)

    def gen_prior2hour_step_count(self, s, step_idx):
        mean = self.gen_prior2hour_step_count_mean(s)
        noise = self._sample_noise(self.cfg.resid_prior2hour_step_count, step_idx)
        return float(np.clip(mean + noise, *self.cfg.limits_prior2hour_step_count))

    def gen_recorded_physical_activity_mean(self, s, return_logit=False):
        X = np.array(
            [
                1.0,
                s["recorded_physical_activity_lag1"],
                s["dow"],
                s["decision_time"],
            ],
            dtype=float,
        )
        eta = float(self.cfg.theta_recorded_physical_activity @ X)
        return eta if return_logit else self._sigmoid(eta)

    def gen_recorded_physical_activity(self, s, step_idx):
        eta = self.gen_recorded_physical_activity_mean(s, return_logit=True)
        base_p = self._sigmoid(eta)
        noise = self._sample_noise(self.cfg.resid_recorded_physical_activity, step_idx)
        p = float(
            np.clip(
                base_p + noise,
                *self.cfg.limits_recorded_physical_activity,
            )
        )
        return float(rd.binomial(1, p))

    def gen_ws_interaction_mean(self, s, return_logit=False):
        """[1, Interacted_7d_walk, Interacted_7d_salience, dow, decision_time]."""
        X = np.array(
            [
                1.0,
                s["Interacted_7d_walk"],
                s["Interacted_7d_salience"],
                s["dow"],
                s["decision_time"],
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

    def gen_salience_interaction_mean(self, s, return_logit=False):
        X = np.array(
            [
                1.0,
                s["Interacted_7d_walk"],
                s["Interacted_7d_salience"],
                s["dow"],
                s["decision_time"],
            ],
            dtype=float,
        )
        eta = float(self.cfg.theta_salience_interaction @ X)
        return eta if return_logit else self._sigmoid(eta)

    def gen_salience_interaction(self, s, step_idx):
        eta = self.gen_salience_interaction_mean(s, return_logit=True)
        base_p = self._sigmoid(eta)
        noise = self._sample_noise(self.cfg.resid_salience_interaction, step_idx)
        p = float(
            np.clip(base_p + noise, *self.cfg.limits_salience_interaction)
        )
        return float(rd.binomial(1, p))

    def gen_fourSC_mean(self, s, Ah):
        """
        31 predictors — matches ``2_fit_vanilla_testbed`` fourSC_cond (no ``is_weekend``;
        ``past7days_morning_wearing``, ``yesterday_SalienceMessage``;
        includes ``perceived_utility_lastweek`` and its action interaction).
        """
        wear7 = float(s.get("past7days_morning_wearing", s["past7days_daywearing"]))
        y_sal = float(s["yesterday_salience_message"])
        pu = float(s["perceived_utility_lastweek"])
        cae = float(s["CAE_avg_lastweek"])
        X = np.array(
            [
                1.0,
                s["fourSC_lag1"],
                s["yesterday_step"],
                s["seven_day_step_count_avg"],
                s["prior2hour_step"],
                s["Previous7DaysRPA"],
                s["recent_burden"],
                s["seven_day_pageview"],
                wear7,
                y_sal,
                s["Interacted_7d_walk"],
                s["Interacted_7d_salience"],
                s["anticipated_affect_yesterday"],
                s["dow"],
                s["decision_time"],
                pu,
                cae,
                Ah,
                Ah * s["yesterday_step"],
                Ah * s["prior2hour_step"],
                Ah * s["recent_burden"],
                Ah * s["seven_day_pageview"],
                Ah * wear7,
                Ah * y_sal,
                Ah * s["Interacted_7d_walk"],
                Ah * s["Interacted_7d_salience"],
                Ah * s["anticipated_affect_yesterday"],
                Ah * s["dow"],
                Ah * s["decision_time"],
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
        Ew = float(s["perceived_utility_lastweek"])
        base, q = _split_ml_pv_full(self.cfg.theta_ml_PV)
        (
            alpha0,
            alpha1,
            a2_dow,
            a2_dt,
            a2_rb,
            alpha_ar1,
            alpha3,
            alpha4,
            _sigma_pv,
        ) = base
        lag1 = float(s["hourly_pageview_lag1"])
        mu = (
            alpha0
            + alpha1 * Ew
            + a2_dow * float(s["dow"])
            + a2_dt * float(s["decision_time"])
            + a2_rb * float(s["recent_burden"])
            + alpha_ar1 * lag1
            + Ah * (alpha3 + alpha4 * Ew)
        )
        if Iw != 0:
            xq = np.array([1.0, Ew, float(s["dow"]), float(s["decision_time"]), float(s["recent_burden"])])
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
        Daily ridge design (morning row): 22 columns — matches
        ``2_fit_vanilla_testbed`` ``anticipated_affect_cond_day``
        (no ``planning_prompt``; includes ``perceived_utility_lastweek`` main and
        AM/PM interactions).
        """
        ys = float(s["today_step"])
        rpa = float(s["recorded_physical_activity"])
        sal = float(s["salience_message"])
        dow = float(s["dow"])
        pu = float(s["perceived_utility_lastweek"])
        cae = float(s["CAE_avg_lastweek"])
        X = np.array(
            [
                1.0,
                float(s["anticipated_affect"]),
                ys,
                rpa,
                sal,
                dow,
                pu,
                cae,
                ws_morning,
                ws_afternoon,
                ws_morning * ys,
                ws_afternoon * ys,
                ws_morning * rpa,
                ws_afternoon * rpa,
                ws_morning * sal,
                ws_afternoon * sal,
                ws_morning * dow,
                ws_afternoon * dow,
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
        Ew = float(s["perceived_utility_lastweek"])
        base, q = _split_ml_fw(self.cfg.theta_ml_FW)
        beta0, beta1, b2_dow, b2_rb, beta_ar1, beta3, beta4, beta5, beta6 = base
        dow = float(s["dow"])
        rb = float(s["recent_burden"])
        y_lag = float(s["yesterday_fitbitwearing_morning"])
        eta = (
            beta0
            + beta1 * Ew
            + b2_dow * dow
            + b2_rb * rb
            + beta_ar1 * y_lag
            + ws_morning * (beta3 + beta4 * Ew)
            + ws_afternoon * (beta5 + beta6 * Ew)
        )
        if Iw != 0:
            xq = np.array([1.0, Ew, dow, rb])
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
        Ew = float(s["perceived_utility_lastweek"])
        base, q = _split_ml_pj(self.cfg.theta_ml_PJ)
        t0, t1, t2_dow, t2_rb, t_ar1, t3, t4, t5, t6 = base
        dow = float(s["dow"])
        rb = float(s["recent_burden"])
        y_lag = float(s["yesterday_present"])
        eta = (
            t0
            + t1 * Ew
            + t2_dow * dow
            + t2_rb * rb
            + t_ar1 * y_lag
            + ws_morning * (t3 + t4 * Ew)
            + ws_afternoon * (t5 + t6 * Ew)
        )
        if Iw != 0:
            xq = np.array([1.0, Ew, dow, rb])
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
        X = np.concatenate(
            [
                np.array([1.0, CAE_lastweek, week_norm], dtype=float),
                np.asarray(foursc_wk, dtype=float).ravel(),
                np.asarray(antic_wk, dtype=float).ravel(),
            ]
        )
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

    def gen_perceived_utility_mean(self, Ew_carry, week_norm, pw_wk, dw_wk, dp_wk):
        """Gaussian AR on ``E_w``; ``week_norm`` kept for API compatibility (unused)."""
        del week_norm
        if self.cfg.theta_ml_Ew.size < 6:
            raise RuntimeError("theta_ml_Ew required for latent E_w transition")
        a0, a1, a2, a3, a4, _sigma_e = [float(x) for x in self.cfg.theta_ml_Ew[:6]]
        pv_m, fw_m, pj_m = self._week_means_from_arrays(pw_wk, dw_wk, dp_wk)
        return float(a0 + a1 * float(Ew_carry) + a2 * pv_m + a3 * fw_m + a4 * pj_m)

    def gen_perceived_utility(self, Ew_carry, week_norm, pw_wk, dw_wk, dp_wk, week_idx):
        del week_idx
        mean = self.gen_perceived_utility_mean(Ew_carry, week_norm, pw_wk, dw_wk, dp_wk)
        sigma_e = float(self.cfg.theta_ml_Ew[5]) if self.cfg.theta_ml_Ew.size > 5 else 0.1
        noise = float(rd.normal(0.0, sigma_e))
        return float(np.clip(mean + noise, *self.cfg.limits_perceived_utility))

    def gen_week_present_mean(self, perceived_utility, return_logit=False):
        if self.cfg.theta_ml_J.size < 2:
            raise RuntimeError("theta_ml_J required for week_present (J_week)")
        b0, b1 = float(self.cfg.theta_ml_J[0]), float(self.cfg.theta_ml_J[1])
        eta = b0 + b1 * float(perceived_utility)
        return eta if return_logit else self._sigmoid(eta)

    def gen_week_present(self, perceived_utility, week_idx):
        eta = self.gen_week_present_mean(perceived_utility, return_logit=True)
        base_p = self._sigmoid(eta)
        noise = self._sample_noise(self.cfg.resid_ml_J_week, week_idx)
        if not np.isfinite(noise):
            noise = self._sample_noise(self.cfg.resid_week_present, week_idx)
        lim = self.cfg.limits_week_present
        p = float(np.clip(base_p + noise, *lim))
        return float(rd.binomial(1, p))

    def gen_CAE_short_mean(self, CAE_avg):
        return float(self.cfg.theta_CAE_short @ np.array([1.0, CAE_avg], dtype=float))

    def gen_CAE_short(self, CAE_avg, week_idx):
        mean = self.gen_CAE_short_mean(CAE_avg)
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


_STATE_FROM_DF_FIT_ROW = (
    ("fourSC_lag1", ("FourSC_lag1",)),
    ("hourly_pageview_lag1", ("hourly_pageview_count_lag1",)),
    ("prior2hour_step_count_lag1", ("prior2hour_step_count_lag1",)),
    ("prior2hour_step", ("prior2hour_step_norm", "prior2hour_step")),
    ("yesterday_step", ("YesterdayStepCount_norm", "YesterdayStepCount")),
    ("seven_day_step_count_avg", ("EMA_StepCount_norm", "EMA_StepCount")),
    ("seven_day_pageview", ("Past7DaysPageviewEMA_norm", "Past7DaysPageviewEMA")),
    ("past7days_daywearing", ("past7days_morning_wearing", )),
    ("past7days_morning_wearing", ("past7days_morning_wearing",)),
    ("daily_present", ("daily_present",)),
    ("Previous7DaysRPA", ("Previous7DaysRPA",)),
    ("recent_burden", ("recent_burden_norm", "recent_burden")),
    ("Interacted_7d_walk", ("Interacted_7d_walk",)),
    ("Interacted_7d_salience", ("Interacted_7d_salience",)),
    ("anticipated_affect_yesterday", ("anticipated_affect_yesterday_norm", "anticipated_affect_yesterday")),
    ("anticipated_affect", ("anticipated_affect_norm", "anticipated_affect")),
    ("CAE_avg_lastweek", ("CAE_avg_lastweek_norm", "CAE_avg_lastweek")),
    ("yesterday_fitbitwearing_morning", ("morning_wearing",)),
    ("yesterday_present", ("daily_present_yesterday",)),
    ("yesterday_salience_message", ("yesterday_SalienceMessage",)),
    ("recorded_physical_activity", ("RecordedPhysicalActivity",)),
    ("recorded_physical_activity_lag1", ("recorded_physical_activity_lag1",)),
)


def _state_updates_from_df_fit_row(row):
    if row is None:
        return {}
    out = {}
    for skey, csv_keys in _STATE_FROM_DF_FIT_ROW:
        out[skey] = _float_csv_cell(row, *csv_keys, default=0.0)
    if "past7days_morning_wearing" not in out or out.get("past7days_morning_wearing", 0) == 0:
        out["past7days_morning_wearing"] = out.get("past7days_daywearing", 0.0)
    yh = _float_csv_cell(
        row,
        "yesterday_hourly_pageview_count_norm",
        "yesterday_hourly_pageview_count",
        default=float("nan"),
    )
    if math.isfinite(yh):
        out["yesterday_pageview_morning"] = yh
        out["yesterday_pageview_afternoon"] = yh
    else:
        hm = _float_csv_cell(row, "yesterday_pageview_morning", default=float("nan"))
        ha = _float_csv_cell(row, "yesterday_pageview_afternoon", default=float("nan"))
        if math.isfinite(hm):
            out["yesterday_pageview_morning"] = hm
        if math.isfinite(ha):
            out["yesterday_pageview_afternoon"] = ha
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
        "fourSC_lag1": 0.0,
        "hourly_pageview_lag1": 0.0,
        "prior2hour_step": 0.0,
        "prior2hour_step_count_lag1": 0.0,
        "yesterday_step": 0.0,
        "today_step": 0.0,
        "yesterday_fitbitwearing_morning": 0.0,
        "yesterday_present": 0.0,
        "yesterday_pageview_morning": 0.0,
        "yesterday_pageview_afternoon": 0.0,
        "today_pageview_morning": 0.0,
        "today_pageview_afternoon": 0.0,
        "seven_day_step_count_avg": 0.0,
        "seven_day_pageview": 0.0,
        "past7days_daywearing": 0.0,
        "past7days_morning_wearing": 0.0,
        "fitbitwearing_morning": 0.0,
        "daily_present": 0.0,
        "CAE_avg_lastweek": 0.0,
        "perceived_utility_lastweek": 2.0,
        "anticipated_affect": 0.0,
        "anticipated_affect_yesterday": 0.0,
        "Previous7DaysRPA": 0.0,
        "recent_burden": 0.0,
        "recorded_physical_activity": 0.0,
        "recorded_physical_activity_lag1": 0.0,
        "salience_message": 0.0,
        "yesterday_salience_message": 0.0,
        "Interacted_7d_walk": 0.0,
        "Interacted_7d_salience": 0.0,
        "is_weekend": 0.0,
        "dow": 0.0,
        "dow_lag1": 0.0,
        "decision_time": 0.0,
    }
    row = _read_first_csv_row_for_participant(
        _resolve_df_fit_11week_path(df_fit_11week_csv), participant_id
    )
    s.update(_state_updates_from_df_fit_row(row))
    return s
