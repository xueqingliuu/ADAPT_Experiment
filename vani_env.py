"""Simulate one participant from the fitted vanilla testbed.

``EnvConfig`` loads ``params_env_<uid>.json`` (and supporting files) from
``PARAMS_DIR``. ``Env`` draws the next 4-hour steps, anticipated affect,
engagement mediators, weekly CAE, and latent E_w given the current state and
actions. Residuals default to an AR(1) bootstrap of the fitted leftovers.

``resolve_params_dir`` selects the parameter folder: ``ADAPR_PARAMS_DIR``
if set, otherwise ``<repo>/env_para_vanilla``. ``ADAPR_VANILLA_ENV_DIR``,
``ADAPR_EXPERIMENT_PARAMS_DIR``, and ``ADAPR_EST_PRIOR_PARAMS_DIR`` are
deprecated aliases of that folder. This module does not run an RL agent;
``experiment.OnlineEnv`` wraps it for slot-by-slot control.

History summaries (7-day EWMAs, lags, "yesterday" step count) are defined in
``derived_covariates.py`` and computed identically from the MRT data and from
simulated data, so fitted coefficients are used as stored. Residuals use a
credibility-weighted AR(1) bootstrap (``residual_pool_stats``).

Page views are a two-part hurdle whose coefficients come from script 4
(``theta_penalized_PV``): logistic ``P(count>0)`` and, given a positive
count, a Gaussian on ``log(x+1)`` z-scored with the positive-count
mean / SD. Count 0 sits at ``log(1) = 0`` before z-scoring (legacy folders:
``log(x)`` for positives and ``log(0.5)`` for zero). Both parts include ``E_w``,
weekend, slot, burden, lag1, ``A``, ``A×E_w``, and the opening ``J_w``
query (``week_present_lastweek``) on the linear predictor. ``I_w`` is not
in this model. The coefficient layout is the one ``06_fit_pu.py`` writes,
with no study-week slope. A stored week slot is dropped only when that
coefficient is exactly 0. STE-tuned folders carry edited action slopes in
JSON, so they enter
both hurdle parts directly.

Page-view occurrence also includes the interaction with a delivered
suggestion (``theta_penalized_PV_interaction``, 4b); interaction is therefore
generated before page views. The constant daily app-dropout hazard is loaded
from ``generator_calibration.json``; all participant dynamics remain fitted.
"""
from __future__ import annotations

import csv
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import numpy.random as rd
import pandas as pd

from dropout_utils import mask_post_dropout_outcomes


DROPOUT_OUTCOME_COLUMNS = (
    "4hour_step",
    "4hour_step_norm",
    "prior2hour_step",
    "prior2hour_step_norm",
    "active_status",
    "Interacted_walk",
    "anticipated_affect",
    "anticipated_affect_norm",
    "HourlyPageviewCount",
    "HourlyPageviewCount_norm",
    "morning_wearing",
    "nextday_wearing",
    "daily_present",
    "week_present",
    "Exp-tool-1",
    "Exp-tool-1_norm",
    "Exp-tool-2",
    "Exp-tool-2_norm",
    "CAE_avg",
    "CAE_avg_norm",
    "CAE_short_avg",
    "CAE_short_avg_norm",
)

DROPOUT_WEEKLY_OUTCOME_COLUMNS = (
    "week_present",
    "Exp-tool-1",
    "Exp-tool-1_norm",
    "Exp-tool-2",
    "Exp-tool-2_norm",
    "CAE_avg",
    "CAE_avg_norm",
    "CAE_short_avg",
    "CAE_short_avg_norm",
)


def mask_dropout_fit_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Mask modeled outcomes on/after dropout and invalid weekly emissions.

    Weekly survey values are replicated across their calendar week upstream.
    If dropout occurs during that week, mask the weekly outcome on every row
    so a pre-dropout Monday copy cannot leak an end-of-week post-dropout zero.
    """
    out = mask_post_dropout_outcomes(df, DROPOUT_OUTCOME_COLUMNS)
    required = {"ParticipantIdentifier", "week", "study_day", "dropout_day"}
    if required.issubset(out.columns):
        dropout = pd.to_numeric(out["dropout_day"], errors="coerce")
        study_day = pd.to_numeric(out["study_day"], errors="coerce")
        week_has_dropout = (
            np.isfinite(dropout)
            & out.assign(_post=study_day >= dropout)
            .groupby(["ParticipantIdentifier", "week"], sort=False)["_post"]
            .transform("any")
            .to_numpy(dtype=bool)
        )
        weekly = [
            col for col in DROPOUT_WEEKLY_OUTCOME_COLUMNS if col in out.columns
        ]
        if weekly and week_has_dropout.any():
            out.loc[week_has_dropout, weekly] = np.nan
    return out

from ewm_utils import ewma_gamma

PROJECT_ROOT = Path(
    os.environ.get("ADAPR_PROJECT_ROOT", Path(__file__).resolve().parent)
).expanduser().resolve()
_DEFAULT_PARAMS_NAME = "env_para_vanilla"
# Old names for the parameter folder. Each must resolve to the same directory
# as ``ADAPR_PARAMS_DIR`` (or the default); a different directory is an error.
_PARAMS_DIR_ALIASES = (
    "ADAPR_VANILLA_ENV_DIR",
    "ADAPR_EXPERIMENT_PARAMS_DIR",
    "ADAPR_EST_PRIOR_PARAMS_DIR",
)


def default_params_dir() -> Path:
    """``<repo>/env_para_vanilla``, ignoring ``ADAPR_PARAMS_DIR``."""
    return (PROJECT_ROOT / _DEFAULT_PARAMS_NAME).resolve()


def _params_path(raw) -> Path:
    """Absolute path. Relative values are under ``PROJECT_ROOT``."""
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_params_dir(params_dir=None) -> Path:
    """Parameter folder for the pipeline and the simulator.

    An explicit ``params_dir`` wins. Otherwise ``ADAPR_PARAMS_DIR`` if it is
    set (absolute, or relative to the repository), otherwise
    ``<repo>/env_para_vanilla``.

    ``ADAPR_VANILLA_ENV_DIR``, ``ADAPR_EXPERIMENT_PARAMS_DIR``, and
    ``ADAPR_EST_PRIOR_PARAMS_DIR`` are deprecated aliases of that folder.
    If one is set and resolves to a different directory than
    ``ADAPR_PARAMS_DIR`` (or the default), raise ``ValueError`` naming both.
    If it resolves to the same directory, warn and continue.
    """
    primary_raw = os.environ.get("ADAPR_PARAMS_DIR", "").strip()
    if primary_raw:
        chosen = _params_path(primary_raw)
        chosen_label = "ADAPR_PARAMS_DIR"
    else:
        chosen = default_params_dir()
        chosen_label = "the default parameter directory"
    for name in _PARAMS_DIR_ALIASES:
        alias_raw = os.environ.get(name, "").strip()
        if not alias_raw:
            continue
        alias = _params_path(alias_raw)
        if alias != chosen:
            raise ValueError(
                f"{name} resolves to {alias}, but "
                f"{chosen_label} resolves to {chosen}. "
                "Set only ADAPR_PARAMS_DIR."
            )
        warnings.warn(
            f"{name} is deprecated; "
            f"it matches {chosen_label} ({chosen}). "
            "Set only ADAPR_PARAMS_DIR if you need a non-default folder.",
            DeprecationWarning,
            stacklevel=2,
        )
    if params_dir is not None and str(params_dir).strip():
        return _params_path(params_dir)
    return chosen


# Import-time snapshot. Launchers export ``ADAPR_PARAMS_DIR`` before Python
# starts. Call ``resolve_params_dir`` when the variable may change later.
PARAMS_DIR = resolve_params_dir()

# Latent E_w support: script-4 quadrature and the simulator clip share this
# interval. Change it here only (``06_fit_pu`` imports these names).
E_QUAD_LO = -4.0
E_QUAD_HI = 4.0


def _load_json(path: Path):
    """Load JSON and name the file if it is empty or truncated."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"empty JSON file: {path} ({path.stat().st_size} bytes)")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {path} ({path.stat().st_size} bytes): {exc}"
        ) from exc


def load_CAE_norm_params(params_dir=PARAMS_DIR):
    """Return ``(shift, scale)`` used to z-score raw weekly CAE into ``CAE_avg_norm``.

    Standardization (``03_standardize.py``):
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


def invert_log_zscore(norm, shift, scale):
    """Undo ``(log(x+1) - shift) / scale`` and return raw count ``x >= 0``.

    Script 3 stores step counts as ``log(x+1)`` then z-scores. Gaussian draws
    that reconstruct a negative ``log(x+1)`` are mapped to 0.
    """
    s = float(scale)
    if s == 0.0 or not math.isfinite(s):
        s = 1.0
    try:
        z = float(norm)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(z):
        return 0.0
    log_val = float(shift) + s * z
    if not math.isfinite(log_val):
        return 0.0
    count = math.expm1(log_val)
    if not math.isfinite(count) or count <= 0.0:
        return 0.0
    return float(count)


def zscore_value(raw, shift, scale):
    """Apply ``(raw - shift) / scale`` (the EMA_Prior2Hour map)."""
    s = float(scale)
    if s == 0.0 or not math.isfinite(s):
        s = 1.0
    try:
        x = float(raw)
    except (TypeError, ValueError):
        x = 0.0
    if not math.isfinite(x):
        x = 0.0
    return (x - float(shift)) / s


def invert_zscore(norm, shift, scale):
    """Undo ``(raw - shift) / scale`` (the EMA / recent-burden map)."""
    s = float(scale)
    if s == 0.0 or not math.isfinite(s):
        s = 1.0
    try:
        z = float(norm)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(z):
        return 0.0
    return float(shift) + s * z

THETA_PRIOR2HOUR_STEP_COUNT_NAMES = [
    "intercept",
    "EMA_Prior2HourStepCount",
    "is_weekend",
    "decision_time",
    "fourSC_lag1_morning",
    "fourSC_lag1_afternoon",
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
    "WalkingSuggestion_by_decision_time",
    "WalkingSuggestion_by_perceived_utility_lastweek",
    "WalkingSuggestion_by_CAE_avg_lastweek",
]

THETA_ANTIC_NAMES = [
    "intercept",
    "anticipated_affect_yesterday",
    "active_status_fraction_7days",
    "is_weekend",
    "perceived_utility_lastweek",
    "CAE_avg_lastweek",
    "recent_burden",
    "A0_morning",
    "A1_afternoon",
    "A0_morning_by_perceived_utility_lastweek",
    "A1_afternoon_by_perceived_utility_lastweek",
    "A0_morning_by_CAE_avg_lastweek",
    "A1_afternoon_by_CAE_avg_lastweek",
    "A0_morning_by_recent_burden",
    "A1_afternoon_by_recent_burden",
    "A0_morning_by_active_status_fraction_7days",
    "A1_afternoon_by_active_status_fraction_7days",
]

THETA_CAE_NAMES = [
    "intercept",
    "CAE_avg_lastweek",
    "fourSC_ewma",
    "anticipated_affect_ewma",
]
PF_THETA_CAE_NAMES = [
    "intercept",
    "CAE_avg_lastweek",
    "fourSC_ewma",
    "anticipated_affect_ewma",
]

THETA_CAE_SHORT_AVG_NAMES = [
    "intercept",
    "CAE_avg",
]

# Design sizes (``07_fit_testbed.py``); no 7-day salience-interaction covariate.
# The environment uses the full fitted fourSC model, including the
# ``seven_day_pageview_count`` and ``anticipated_affect_yesterday`` predictors
# (and their WalkingSuggestion interactions) — these are NOT trimmed away.
P_FOURSC = len(THETA_FOURSC_NAMES)
_LEGACY_INTERACT_DROP = (2,)  # removed legacy salience-history covariate
_LEGACY_FOURSC_SALIENCE_DROP = (8, 21)
# Layout immediately before dropping WalkingSuggestion × is_weekend.
_P_FOURSC_WITH_WEEKEND = 27
_FOURSC_WEEKEND_INTERACT_IDX = 23
_P_FOURSC_WITH_WEEKEND_AND_SALIENCE = 29
# Legacy antic vectors (relative to the pre-recent-burden 14-column layout):
# +1 = includes today_step_count; +4 = that plus salience terms.
_LEGACY_ANTIC_TODAY_STEP_DROP = (2,)
_LEGACY_ANTIC_SALIENCE_AND_TODAY_STEP_DROP = (2, 4, 10, 11)
P_ANTIC = len(THETA_ANTIC_NAMES)
# Layout immediately before adding A0/A1 × 7-day active fraction.
_P_ANTIC_NO_ACT7 = 15
# Layout immediately before dropping A0/A1 × is_weekend (still 17 columns,
# but those two were weekend interactions, not the active-fraction terms).
_P_ANTIC_WITH_WEEKEND = 17
_ANTIC_WEEKEND_INTERACT_IDX = (9, 10)
# Pre-recent-burden fits lack the recent_burden main effect (index 6) and
# the two trailing A0/A1 x recent_burden interactions; they are zero-padded.
# Length is hardcoded so it does not drift when later columns are dropped.
_P_ANTIC_NO_BURDEN = 14
_ANTIC_BURDEN_MAIN_IDX = THETA_ANTIC_NAMES.index("recent_burden")
P_ACTIVE_STATUS = len(THETA_ACTIVE_STATUS_NAMES)
P_PRIOR2HOUR = len(THETA_PRIOR2HOUR_STEP_COUNT_NAMES)
P_CAE = len(THETA_CAE_NAMES)
P_PF_CAE = len(PF_THETA_CAE_NAMES)
P_CAE_SHORT = len(THETA_CAE_SHORT_AVG_NAMES)
CAE_FOURSC_SLOTS = 14
CAE_FOURSC_SLOTS_RL = 12
CAE_ANTIC_DAYS = 7
CAE_ANTIC_DAYS_RL = 6
# Legacy env CAE vector: intercept, AR, week, fourSC, antic.
_LEGACY_CAE_WITH_WEEK = 5
_LEGACY_CAE_WEEK_IDX = 2
# 06_fit_pu.py layout. PV base is intercept, E_w, weekend, slot, burden,
# lag1, A, A×E_w. A stored week slot sits after burden and lengthens the
# base from 8 to 9 (hurdle 23 to 25).
PV_ML_BASE = 8
PV_ML_QUERY = 3
PV_ML_HURDLE = PV_ML_BASE + PV_ML_QUERY + PV_ML_BASE + PV_ML_QUERY + 1
_PV_WEEK_SLOT = 5
_PV_BASE_WITH_WEEK = PV_ML_BASE + 1
_PV_HURDLE_WITH_WEEK = (
    _PV_BASE_WITH_WEEK + PV_ML_QUERY + _PV_BASE_WITH_WEEK + PV_ML_QUERY + 1
)
# FW/PJ base is intercept, E_w, weekend, burden, lag1, then the two action
# pairs. A stored week slot sits after burden and lengthens the base from 9 to 10.
FW_ML_BASE = 9
PJ_ML_BASE = 9
_FW_PJ_WEEK_SLOT = 4
_FW_PJ_BASE_WITH_WEEK = FW_ML_BASE + 1
# Query suffix: J_w × (intercept, E_w, recent_burden), jointly fit in script 06.
# Added to the linear predictor when J_w=1; I_w is not in the PV/FW/PJ model.
FW_ML_QUERY = 3
PJ_ML_QUERY = 3
# Legacy suffixes (before weekend/decision-time were removed from the query block).
_PV_QUERY_LEGACY_KEEP = np.array([0, 1, 4], dtype=int)  # skip weekend, decision time
_FW_PJ_QUERY_LEGACY_KEEP = np.array([0, 1, 3], dtype=int)  # skip weekend


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
        f"{key}: fitted coefficient names do not match 07_fit_testbed.py. "
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


def assert_complete_week_slots(
    df,
    *,
    uid_col: str = "ParticipantIdentifier",
    week_col: str = "week",
    date_col: str = "Date",
    decision_col: str = "DecisionTime",
    slots_per_week: int = 14,
    slots_per_day: int = 2,
) -> None:
    """Raise unless every (user, week) is a full 7-day × 2-slot panel.

    Fit loaders chunk by positional ``reshape(-1, 14)`` / ``len // 14``.
    A mid-panel hole would shift every later week's AM/PM and Sunday
    assignment without this guard.
    """
    if df is None or len(df) == 0:
        return
    missing = [
        c for c in (uid_col, week_col, date_col, decision_col) if c not in df.columns
    ]
    if missing:
        raise KeyError(f"assert_complete_week_slots missing columns {missing}")

    week_sizes = df.groupby([uid_col, week_col], sort=False).size()
    bad_weeks = week_sizes[week_sizes != slots_per_week]
    if len(bad_weeks) > 0:
        preview = ", ".join(
            f"user={uid} week={w} n={int(n)}"
            for (uid, w), n in bad_weeks.head(8).items()
        )
        raise ValueError(
            f"expected {slots_per_week} rows per (user, week); "
            f"{len(bad_weeks)} groups differ ({preview})"
        )

    day_sizes = df.groupby([uid_col, date_col], sort=False).size()
    bad_days = day_sizes[day_sizes != slots_per_day]
    if len(bad_days) > 0:
        preview = ", ".join(
            f"user={uid} date={d} n={int(n)}"
            for (uid, d), n in bad_days.head(8).items()
        )
        raise ValueError(
            f"expected {slots_per_day} rows per (user, Date); "
            f"{len(bad_days)} groups differ ({preview})"
        )

    slot_pairs = (
        df.groupby([uid_col, date_col], sort=False)[decision_col]
        .agg(lambda s: tuple(sorted(int(v) for v in s if pd.notna(v))))
    )
    expected_slots = tuple(range(slots_per_day))
    bad_slots = slot_pairs[slot_pairs != expected_slots]
    if len(bad_slots) > 0:
        preview = ", ".join(
            f"user={uid} date={d} slots={slots}"
            for (uid, d), slots in bad_slots.head(8).items()
        )
        raise ValueError(
            f"expected DecisionTime {expected_slots} each (user, Date); "
            f"{len(bad_slots)} groups differ ({preview})"
        )


def within_week_ewma(values, gamma=None) -> float:
    """Normalized discounted average over a chronological sequence.

    Same formula as ``01_extract_data._ewm_prior_rows`` / ``ewm_utils.ewma_gamma``.
    ``gamma=None`` (default) derives the decay from the calendar length of
    ``values`` (:func:`ewm_utils.gamma_from_n`), so a 14-slot fourSC week and a
    7-day antic week get different-but-comparable decay envelopes. NaNs keep
    their slot in the decay. Empty / all-NaN → 0.
    """
    return ewma_gamma(values, gamma, empty=0.0)


def _antic_daily_from_week(antic_wk) -> np.ndarray:
    antic = np.asarray(antic_wk, dtype=float).ravel()
    if antic.size == 14:
        return antic.reshape(7, 2).mean(axis=1)
    if antic.size == 12:
        return antic.reshape(6, 2).mean(axis=1)
    if antic.size in (7, 6):
        return antic
    raise ValueError(
        f"antic_wk must have 6/7 daily values or 12/14 AM/PM slots, got {antic.size}"
    )


def cae_mediator_ewmas(foursc_wk, antic_wk) -> tuple[float, float]:
    """Compress one week's fourSC slots and antic days to EWMA scalars.

    Accepts a full calendar week (14 slots / 7 days) or RL Mon–Sat
    (12 slots / 6 days). Decay follows ``gamma_from_n`` of the window length.
    Missing entries are filled with **this week's** finite mean (not the
    user's trajectory mean) so fit-time and simulation EWMAs match.
    """
    foursc = _fill_nan_with_finite_mean(np.asarray(foursc_wk, dtype=float).ravel())
    if foursc.size not in (CAE_FOURSC_SLOTS, CAE_FOURSC_SLOTS_RL):
        raise ValueError(
            f"foursc_wk must have {CAE_FOURSC_SLOTS_RL} or {CAE_FOURSC_SLOTS} "
            f"decision slots, got {foursc.size}"
        )
    antic = _fill_nan_with_finite_mean(_antic_daily_from_week(antic_wk))
    if antic.size not in (CAE_ANTIC_DAYS, CAE_ANTIC_DAYS_RL):
        raise ValueError(
            f"antic_wk must have {CAE_ANTIC_DAYS_RL} or {CAE_ANTIC_DAYS} "
            f"daily values, got {antic.size}"
        )
    return (
        within_week_ewma(foursc),
        within_week_ewma(antic),
    )


def _antic_daily_from_week_nan(antic_wk) -> np.ndarray:
    """Daily anticipated affect; a day is NaN only if all its entries are NaN."""
    antic = np.asarray(antic_wk, dtype=float).ravel()
    if antic.size in (7, 6):
        return antic
    if antic.size not in (14, 12):
        raise ValueError(
            f"antic_wk must have 6/7 daily values or 12/14 AM/PM slots, got {antic.size}"
        )
    pairs = antic.reshape(-1, 2)
    out = np.full(pairs.shape[0], np.nan)
    ok = np.isfinite(pairs).any(axis=1)
    out[ok] = np.nanmean(pairs[ok], axis=1)
    return out


def pf_cae_mediator_ewmas(foursc_wk, antic_wk) -> tuple[float, float]:
    """PF CAE-transition mediator summaries (Mon-Sat), consistent with the RL
    within-week mediator summaries: calendar-weighted EWMA in which a missing
    entry keeps its calendar slot but is skipped (no imputation); an all-missing
    week gives 0, the standardized centre.

    Unlike :func:`cae_mediator_ewmas` (environment model, week-mean fill), this
    is used only by the particle-filter working model and its prior fit.
    """
    foursc = np.asarray(foursc_wk, dtype=float).ravel()
    if foursc.size not in (CAE_FOURSC_SLOTS, CAE_FOURSC_SLOTS_RL):
        raise ValueError(
            f"foursc_wk must have {CAE_FOURSC_SLOTS_RL} or {CAE_FOURSC_SLOTS} "
            f"decision slots, got {foursc.size}"
        )
    antic = _antic_daily_from_week_nan(antic_wk)
    return within_week_ewma(foursc), within_week_ewma(antic)


def pf_cae_mediator_ewma_rows(foursc_weeks, antic_weeks) -> tuple[np.ndarray, np.ndarray]:
    """Apply :func:`pf_cae_mediator_ewmas` independently to each week."""
    foursc_weeks = np.atleast_2d(np.asarray(foursc_weeks, dtype=float))
    antic_weeks = np.atleast_2d(np.asarray(antic_weeks, dtype=float))
    if antic_weeks.shape[0] != foursc_weeks.shape[0]:
        raise ValueError("foursc_weeks and antic_weeks have different numbers of weeks")
    out = [pf_cae_mediator_ewmas(f, a) for f, a in zip(foursc_weeks, antic_weeks)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def cae_mediator_ewma_rows(foursc_weeks, antic_weeks) -> tuple[np.ndarray, np.ndarray]:
    """Apply :func:`cae_mediator_ewmas` independently to each week."""
    foursc_weeks = np.asarray(foursc_weeks, dtype=float)
    antic_weeks = np.asarray(antic_weeks, dtype=float)
    if foursc_weeks.ndim == 1:
        foursc_weeks = foursc_weeks[None, :]
    if antic_weeks.ndim == 1:
        antic_weeks = antic_weeks[None, :]
    n = int(foursc_weeks.shape[0])
    if antic_weeks.shape[0] != n:
        raise ValueError(
            f"foursc_weeks has {n} weeks but antic_weeks has {antic_weeks.shape[0]}"
        )
    foursc_e = np.empty(n, dtype=float)
    antic_e = np.empty(n, dtype=float)
    for i in range(n):
        foursc_e[i], antic_e[i] = cae_mediator_ewmas(foursc_weeks[i], antic_weeks[i])
    return foursc_e, antic_e


def c1_from_week1_ewma(theta_CAE, foursc_ewma, antic_ewma) -> float:
    """Week-1 CAE intercept from the two mediator EWMAs.

    ``c_1 = intercept + b_SC*fourSC_ewma + b_AA*antic_ewma``. A non-finite
    EWMA returns NaN so :func:`impute_baseline_cae` falls back to ``c_2``.
    That happens only when week 1 has no rows, or a length
    :func:`cae_mediator_ewmas` rejects. An all-missing mediator is already
    filled inside that function.
    """
    theta_CAE = trim_theta_cae(theta_CAE)
    if not (np.isfinite(foursc_ewma) and np.isfinite(antic_ewma)):
        return float("nan")
    return float(
        theta_CAE[0]
        + theta_CAE[2] * float(foursc_ewma)
        + theta_CAE[3] * float(antic_ewma)
    )


def week1_mediator_slots(dat_week1, std_params) -> tuple[np.ndarray, np.ndarray]:
    """Study-week-1 fourSC and anticipated affect on the fit scale.

    Uses the script-3 log-then-z and Likert maps. ``std_params`` must be the
    scales fit on weeks 2-12, not a scale that includes week 1.
    """
    dat = (
        dat_week1.sort_values(["Date", "DecisionTime"], na_position="last")
        .reset_index(drop=True)
    )
    if "4hour_step_norm" in dat.columns:
        foursc = dat["4hour_step_norm"].to_numpy(dtype=float)
    else:
        raw = pd.to_numeric(dat["4hour_step"], errors="coerce").to_numpy(dtype=float)
        logged = np.full_like(raw, np.nan, dtype=float)
        ok = np.isfinite(raw) & (raw >= 0.0)
        logged[ok] = np.log(raw[ok] + 1.0)
        shift = float(std_params["4hour_step_count_shift"])
        scale = float(std_params["4hour_step_count_scale"])
        foursc = (logged - shift) / scale
    if "anticipated_affect_norm" in dat.columns:
        antic = dat["anticipated_affect_norm"].to_numpy(dtype=float)
    else:
        raw = pd.to_numeric(dat["anticipated_affect"], errors="coerce").to_numpy(dtype=float)
        invalid = np.isfinite(raw) & ((raw < 1.0) | (raw > 7.0))
        raw = np.where(invalid, np.nan, raw)
        antic = 2.0 * (raw - 1.0) / 6.0 - 1.0
    return foursc, antic


def week1_cae_input_record(foursc, antic) -> dict:
    """One user's week-1 EWMA row.

    Calls :func:`cae_mediator_ewmas`, the same week-local mean fill used for
    ``c_2``..``c_12``. An all-missing mediator is filled with 0, the
    standardized centre, and both EWMAs stay finite. No week-1 rows leaves
    both EWMAs NaN (``reason="no_rows"``). A length that function rejects is
    ``irregular=1`` and also NaN. Neither case raises.
    """
    foursc = np.asarray(foursc, dtype=float).ravel()
    antic = np.asarray(antic, dtype=float).ravel()
    record = {
        "foursc_ewma": float("nan"),
        "antic_ewma": float("nan"),
        "irregular": 0,
        "n_foursc": int(foursc.size),
        "n_antic": int(antic.size),
        "reason": "",
    }
    if foursc.size == 0 or antic.size == 0:
        record["reason"] = "no_rows"
        return record
    try:
        foursc_e, antic_e = cae_mediator_ewmas(foursc, antic)
    except ValueError as exc:
        record["irregular"] = 1
        record["reason"] = str(exc)
        return record
    record["foursc_ewma"] = float(foursc_e)
    record["antic_ewma"] = float(antic_e)
    return record


def week1_cae_input_table(week1, std_params) -> pd.DataFrame:
    """Per-user week-1 mediator EWMAs. ``week1`` is already dropout-masked."""
    rows = []
    grouped = week1.groupby("ParticipantIdentifier", sort=False)
    for uid, dat in grouped:
        foursc, antic = week1_mediator_slots(dat, std_params)
        rec = week1_cae_input_record(foursc, antic)
        rec["ParticipantIdentifier"] = int(uid)
        rows.append(rec)
    columns = [
        "ParticipantIdentifier",
        "foursc_ewma",
        "antic_ewma",
        "irregular",
        "n_foursc",
        "n_antic",
        "reason",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def load_week1_cae_inputs(path) -> dict:
    """Read ``week1_cae_inputs.csv``. A missing file is an error."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. 03_standardize.py writes week1_cae_inputs.csv "
            "next to df_fit.csv."
        )
    frame = pd.read_csv(path)
    need = {"ParticipantIdentifier", "foursc_ewma", "antic_ewma", "irregular"}
    missing = sorted(need - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    out = {}
    for row in frame.itertuples(index=False):
        uid = int(row.ParticipantIdentifier)
        out[uid] = {
            "foursc_ewma": float(row.foursc_ewma) if pd.notna(row.foursc_ewma) else float("nan"),
            "antic_ewma": float(row.antic_ewma) if pd.notna(row.antic_ewma) else float("nan"),
            "irregular": int(row.irregular),
            "reason": "" if not hasattr(row, "reason") or pd.isna(row.reason) else str(row.reason),
        }
    return out


def week1_cae_drift(theta_CAE, foursc_wk, antic_wk) -> float:
    """Week-1 CAE intercept from study-week-1 mediator rows.

    Same value as :func:`c1_from_week1_ewma` on the EWMAs of those rows.
    Returns NaN only when there are no rows, or the slot count is not one
    :func:`cae_mediator_ewmas` accepts, so the caller can fall back to ``c_2``.
    """
    rec = week1_cae_input_record(foursc_wk, antic_wk)
    return c1_from_week1_ewma(theta_CAE, rec["foursc_ewma"], rec["antic_ewma"])


def impute_baseline_cae(
    theta_CAE,
    c_panel,
    y_obs,
) -> tuple[float, str]:
    """Backcast study-week-1 CAE from the first observed full CAE.

    For study weeks ``w=2..12``, the fitted transition is
    ``y_w = c_w + a1*y_{w-1} + eps_w``. ``c_panel`` is
    ``[c_1, c_2, ..., c_12]`` and ``a1 = clip(theta_CAE[1], -0.95, 0.95)``.
    Initialize at the week-1 stationary mean ``m_1 = c_1/(1-a1)``. If
    ``c_1`` is non-finite (no week-1 rows), fall back to
    ``m_1 = c_2/(1-a1)`` and suffix the source with ``_c2``.

    If the first observed full CAE is ``y_k``, propagate the mean to week k
    starting from ``m_1`` and using ``c_2..c_k``
    (``M_k = a1**(k-1)*m_1 + sum_{j=2..k} a1**(k-j)*c_j``), then return
    ``m_1 + a1**(k-1)*(y_k-M_k)``. Later observations add no information
    given ``y_k`` under the Markov model. An observed ``y_1`` is returned
    verbatim; no observations returns ``m_1``.
    """
    theta_CAE = np.asarray(theta_CAE, dtype=float).ravel()
    c_panel = np.asarray(c_panel, dtype=float).ravel()
    y_obs = np.asarray(y_obs, dtype=float).ravel()

    if theta_CAE.size < 2:
        raise ValueError("theta_CAE must contain at least intercept and AR(1) slope")
    if c_panel.size != 12 or y_obs.size != 12:
        raise ValueError("c_panel must contain weeks 1..12 and y_obs weeks 1..12")
    if not np.all(np.isfinite(c_panel[1:])):
        raise ValueError("all transition intercepts c_2..c_12 must be finite")

    a1 = float(np.clip(theta_CAE[1], -0.95, 0.95))
    used_c2 = not np.isfinite(c_panel[0])
    m1 = float((c_panel[1] if used_c2 else c_panel[0]) / (1.0 - a1))
    src_suffix = "_c2" if used_c2 else ""

    if np.isfinite(y_obs[0]):
        return float(y_obs[0]), "observed"

    future_obs = np.flatnonzero(np.isfinite(y_obs[1:]))
    if future_obs.size == 0:
        return m1, "model_mean" + src_suffix

    k = int(future_obs[0]) + 2
    mean_k = m1
    for study_week in range(2, k + 1):
        mean_k = float(c_panel[study_week - 1] + a1 * mean_k)

    y1_hat = m1 + a1 ** (k - 1) * (float(y_obs[k - 1]) - mean_k)
    return float(y1_hat), f"backcast_{k - 1}w{src_suffix}"


def build_CAE_features(CAE_lastweek, foursc_wk, antic_wk) -> np.ndarray:
    """Env generative CAE features matching ``THETA_CAE_NAMES``.

    fourSC and anticipated affect enter as EWMA summaries of the full calendar
    week (14 slots / 7 days). There is no study-week slope. The PF transition
    uses :func:`build_pf_CAE_features`, which also has no week term.
    """
    foursc_e, antic_e = cae_mediator_ewmas(foursc_wk, antic_wk)
    x = np.array(
        [1.0, float(CAE_lastweek), foursc_e, antic_e],
        dtype=float,
    )
    if x.size != P_CAE:
        raise RuntimeError(f"CAE feature length {x.size} != {P_CAE}")
    return x


def build_pf_CAE_features(CAE_lastweek, foursc_wk, antic_wk) -> np.ndarray:
    """PF CAE transition features matching ``PF_THETA_CAE_NAMES``.

    Mediator summaries follow :func:`pf_cae_mediator_ewmas` (missing entries
    skipped, as in the RL state), and there is no ``week`` term.
    Callers should pass Mon–Sat mediators (12 slots / 6 days).
    """
    foursc_e, antic_e = pf_cae_mediator_ewmas(foursc_wk, antic_wk)
    x = np.array(
        [1.0, float(CAE_lastweek), foursc_e, antic_e],
        dtype=float,
    )
    if x.size != P_PF_CAE:
        raise RuntimeError(f"PF CAE feature length {x.size} != {P_PF_CAE}")
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
    """Accept current fits or trim legacy salience / weekend-interaction columns."""
    a = np.asarray(theta, dtype=float).ravel()
    if a.size == P_FOURSC:
        return a
    if a.size == _P_FOURSC_WITH_WEEKEND:
        return np.delete(a, _FOURSC_WEEKEND_INTERACT_IDX)
    if a.size == _P_FOURSC_WITH_WEEKEND_AND_SALIENCE:
        a = np.delete(a, _LEGACY_FOURSC_SALIENCE_DROP)
        return np.delete(a, _FOURSC_WEEKEND_INTERACT_IDX)
    raise ValueError(
        f"theta_fourSC length {a.size}; expected {P_FOURSC}, "
        f"legacy {_P_FOURSC_WITH_WEEKEND}, or legacy "
        f"{_P_FOURSC_WITH_WEEKEND_AND_SALIENCE}"
    )


_DROPPED_WEEK_WARNED: set[str] = set()


def _warn_dropped_week(label: str, detail: str) -> None:
    """Warn once per coefficient block when a legacy week slot is removed."""
    if label in _DROPPED_WEEK_WARNED:
        return
    _DROPPED_WEEK_WARNED.add(label)
    warnings.warn(detail, UserWarning, stacklevel=3)


def trim_theta_cae(theta) -> np.ndarray:
    """Current 4-vector, or a legacy 5-vector with the week coefficient removed."""
    a = np.asarray(theta, dtype=float).ravel()
    if a.size == 0 or a.size == P_CAE:
        return a
    if a.size == _LEGACY_CAE_WITH_WEEK:
        _warn_dropped_week(
            "theta_CAE",
            "theta_CAE has 5 coefficients including week; dropping the week coefficient",
        )
        return np.delete(a, _LEGACY_CAE_WEEK_IDX)
    raise ValueError(
        f"theta_CAE length {a.size}; expected {P_CAE} or legacy {_LEGACY_CAE_WITH_WEEK}"
    )


def trim_pf_cae_prior(nu, Gamma=None):
    """PF CAE prior has no week column. A legacy 5-vector drops that column."""
    nu = np.asarray(nu, dtype=float).ravel()
    if nu.size == _LEGACY_CAE_WITH_WEEK:
        _warn_dropped_week(
            "theta_CAE",
            "theta_CAE has 5 coefficients including week; dropping the week coefficient",
        )
        nu = np.delete(nu, _LEGACY_CAE_WEEK_IDX)
        if Gamma is None:
            return nu
        G = np.asarray(Gamma, dtype=float)
        if G.shape != (_LEGACY_CAE_WITH_WEEK, _LEGACY_CAE_WITH_WEEK):
            raise ValueError(
                "CAE prior covariance shape "
                f"{G.shape}; expected {(_LEGACY_CAE_WITH_WEEK, _LEGACY_CAE_WITH_WEEK)}"
            )
        keep = [i for i in range(_LEGACY_CAE_WITH_WEEK) if i != _LEGACY_CAE_WEEK_IDX]
        return nu, G[np.ix_(keep, keep)]
    if nu.size == P_PF_CAE:
        if Gamma is None:
            return nu
        G = np.asarray(Gamma, dtype=float)
        if G.shape == (P_PF_CAE, P_PF_CAE):
            return nu, G
        raise ValueError(
            f"PF CAE prior covariance shape {G.shape}; expected {(P_PF_CAE, P_PF_CAE)}"
        )
    raise ValueError(
        f"CAE prior mean length {nu.size}; expected {P_PF_CAE} or legacy {_LEGACY_CAE_WITH_WEEK}"
    )


def _pad_theta_antic_burden(a: np.ndarray) -> np.ndarray:
    """Map a pre-recent-burden coefficient vector into the current layout."""
    a = np.insert(a, _ANTIC_BURDEN_MAIN_IDX, 0.0)
    return np.concatenate([a, np.zeros(2)])


def _drop_antic_weekend_interact(a: np.ndarray) -> np.ndarray:
    """Drop A0/A1 × is_weekend from a 17-column (with-weekend) vector."""
    return np.delete(a, _ANTIC_WEEKEND_INTERACT_IDX)


def _pad_antic_act7(a: np.ndarray) -> np.ndarray:
    """Append zero A0/A1 × 7-day active-fraction interactions."""
    return np.concatenate([np.asarray(a, dtype=float).ravel(), np.zeros(2)])


def trim_theta_antic(theta, names=None) -> np.ndarray:
    """Accept current fits; trim legacy today_step / salience / weekend
    interactions and zero-pad pre-recent-burden / pre-act7 fits."""
    a = np.asarray(theta, dtype=float).ravel()
    name_list = [str(x) for x in names] if names is not None else None
    if name_list == list(THETA_ANTIC_NAMES) and a.size == P_ANTIC:
        return a
    if name_list is not None and "A0_morning_by_is_weekend" in name_list:
        if a.size >= _P_ANTIC_WITH_WEEKEND:
            a = _drop_antic_weekend_interact(a)
        if a.size == _P_ANTIC_NO_ACT7:
            a = _pad_antic_act7(a)
        if a.size == P_ANTIC:
            return a
    if a.size == P_ANTIC:
        return a
    if a.size == _P_ANTIC_NO_ACT7:
        return _pad_antic_act7(a)
    if a.size == _P_ANTIC_NO_BURDEN:
        return _pad_antic_act7(_drop_antic_weekend_interact(_pad_theta_antic_burden(a)))
    if a.size == _P_ANTIC_NO_BURDEN + 1:
        return _pad_antic_act7(
            _drop_antic_weekend_interact(
                _pad_theta_antic_burden(np.delete(a, _LEGACY_ANTIC_TODAY_STEP_DROP))
            )
        )
    if a.size == _P_ANTIC_NO_BURDEN + 4:
        return _pad_antic_act7(
            _drop_antic_weekend_interact(
                _pad_theta_antic_burden(
                    np.delete(a, _LEGACY_ANTIC_SALIENCE_AND_TODAY_STEP_DROP)
                )
            )
        )
    raise ValueError(
        f"theta_antic length {a.size}; expected {P_ANTIC}, "
        f"legacy {_P_ANTIC_NO_ACT7}, legacy {_P_ANTIC_WITH_WEEKEND}, "
        f"pre-recent-burden {_P_ANTIC_NO_BURDEN}, "
        f"legacy {_P_ANTIC_NO_BURDEN + 1}, or legacy {_P_ANTIC_NO_BURDEN + 4}"
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


def _split_ml_query(
    theta: np.ndarray,
    base_len: int,
    query_len: int,
    legacy_keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(theta, dtype=float).ravel()
    base = t[:base_len].copy() if t.size >= base_len else np.zeros(base_len, dtype=float)
    rest = t[base_len:]
    if rest.size == query_len:
        q = rest.copy()
    elif rest.size >= int(legacy_keep.max()) + 1:
        q = rest[legacy_keep].copy()
    else:
        q = np.zeros(query_len, dtype=float)
    return base, q


def _drop_zero_week_slot(base, slot: int, plain_len: int, label: str) -> np.ndarray:
    """Drop a legacy week coefficient when it is exactly 0."""
    b = np.asarray(base, dtype=float).ravel()
    if b.size == plain_len:
        return b.copy()
    if b.size != plain_len + 1:
        raise ValueError(
            f"{label} base length {b.size}; expected {plain_len} or {plain_len + 1}"
        )
    coef = float(b[slot])
    if coef != 0.0:
        raise ValueError(
            f"{label} study-week coefficient is {coef}; only an exact 0 can be dropped"
        )
    _warn_dropped_week(
        label,
        f"{label} includes a zero study-week coefficient; dropping it",
    )
    return np.concatenate([b[:slot], b[slot + 1 :]])


def _trim_theta_j(theta) -> np.ndarray:
    """J_w is intercept and E_w. A third slot is the legacy week slope."""
    t = np.asarray(theta, dtype=float).ravel()
    if t.size < 3:
        return t
    coef = float(t[2])
    if coef != 0.0:
        raise ValueError(
            f"theta_ml_J study-week coefficient is {coef}; only an exact 0 can be dropped"
        )
    _warn_dropped_week(
        "theta_ml_J",
        "theta_ml_J includes a zero study-week coefficient; dropping it",
    )
    return np.concatenate([t[:2], t[3:]])


def _split_ml_pv_hurdle(theta: np.ndarray):
    """Split script-06 PV hurdle theta.

    Written layout (23): occurrence 8+3, intensity 8+3, ``sigma_PV``.
    A 25-vector has a week slot in each base; that slot is dropped only
    when the coefficient is exactly 0. Legacy occurrence-only vectors
    return ``int_base is None``.
    """
    t = np.asarray(theta, dtype=float).ravel()
    if t.size >= _PV_HURDLE_WITH_WEEK:
        base_w = _PV_BASE_WITH_WEEK
        occ_len = base_w + PV_ML_QUERY
        occ, rest = t[:occ_len], t[occ_len:]
        occ_base = _drop_zero_week_slot(
            occ[:base_w], _PV_WEEK_SLOT, PV_ML_BASE, "theta_ml_PV"
        )
        occ_q = occ[base_w:]
        int_base = _drop_zero_week_slot(
            rest[:base_w], _PV_WEEK_SLOT, PV_ML_BASE, "theta_ml_PV"
        )
        int_q = rest[base_w : base_w + PV_ML_QUERY]
        sigma = (
            float(rest[base_w + PV_ML_QUERY])
            if rest.size > base_w + PV_ML_QUERY
            else 0.1
        )
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = 0.1
        return occ_base, occ_q, int_base, int_q, sigma
    if t.size >= PV_ML_HURDLE:
        occ_len = PV_ML_BASE + PV_ML_QUERY
        occ, rest = t[:occ_len], t[occ_len:]
        occ_base, occ_q = occ[:PV_ML_BASE], occ[PV_ML_BASE:]
        int_base = rest[:PV_ML_BASE]
        int_q = rest[PV_ML_BASE : PV_ML_BASE + PV_ML_QUERY]
        sigma = (
            float(rest[PV_ML_BASE + PV_ML_QUERY])
            if rest.size > PV_ML_BASE + PV_ML_QUERY
            else 0.1
        )
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = 0.1
        return occ_base, occ_q, int_base, int_q, sigma
    if t.size in (9, 12) and t.size > PV_ML_BASE:
        t = np.concatenate([t[:PV_ML_BASE], t[PV_ML_BASE + 1 :]])
    occ_base, occ_q = _split_ml_query(
        t, PV_ML_BASE, PV_ML_QUERY, _PV_QUERY_LEGACY_KEEP
    )
    return occ_base, occ_q, None, None, None


def _split_ml_pv_full(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    occ_base, occ_q, _int_base, _int_q, _sigma = _split_ml_pv_hurdle(theta)
    return occ_base, occ_q


def _split_ml_action_block(
    theta: np.ndarray,
    *,
    label: str,
    legacy_keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """FW or PJ block: 9+3 as written, or 10+3 with a zero week slot."""
    t = np.asarray(theta, dtype=float).ravel()
    query_len = FW_ML_QUERY
    if t.size >= _FW_PJ_BASE_WITH_WEEK + query_len:
        base = _drop_zero_week_slot(
            t[:_FW_PJ_BASE_WITH_WEEK],
            _FW_PJ_WEEK_SLOT,
            FW_ML_BASE,
            label,
        )
        q = t[_FW_PJ_BASE_WITH_WEEK : _FW_PJ_BASE_WITH_WEEK + query_len].copy()
        return base, q
    if t.size >= FW_ML_BASE + query_len:
        return (
            t[:FW_ML_BASE].copy(),
            t[FW_ML_BASE : FW_ML_BASE + query_len].copy(),
        )
    return _split_ml_query(t, FW_ML_BASE, query_len, legacy_keep)


def _split_ml_fw(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _split_ml_action_block(
        theta, label="theta_ml_FW", legacy_keep=_FW_PJ_QUERY_LEGACY_KEEP
    )


def _split_ml_pj(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _split_ml_action_block(
        theta, label="theta_ml_PJ", legacy_keep=_FW_PJ_QUERY_LEGACY_KEEP
    )


def _ml_query_features(s) -> np.ndarray:
    """Query design: intercept, current E_w, recent burden."""
    return np.array(
        [
            1.0,
            float(s["perceivedUtilityLastWeek"]),
            float(s["activitySuggestionsSentLast7Days"]),
        ],
        dtype=float,
    )


def _ml_query_applied_shift(q, s, Jw: int) -> float:
    """Add ``J_w * (q @ z)`` to the linear predictor. ``I_w`` is not in this model."""
    q = np.asarray(q, dtype=float).ravel()
    if q.size == 0 or int(Jw) == 0:
        return 0.0
    return float(np.dot(q, _ml_query_features(s)))


def _uid_key(uid):
    try:
        return int(float(uid))
    except (TypeError, ValueError):
        return str(uid).strip()


def _fill_nan_with_mean(x, default=0.0):
    x = np.asarray(x, dtype=float)
    if np.all(~np.isfinite(x)):
        return np.full_like(x, float(default), dtype=float)
    return np.where(np.isfinite(x), x, float(np.nanmean(x)))


def _df_numeric_col(df, *names, default=np.nan):
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), default, dtype=float)


def _raw_pageview_counts(g: pd.DataFrame) -> np.ndarray:
    """Integer 4-hour pageview counts from ``HourlyPageviewCount``.

    Script 3 keeps this column raw. Older ``df_fit`` files overwrote it with
    ``log(x+1)``; those are recovered with ``expm1`` when the values are not
    integer-like.
    """
    vals = _df_numeric_col(g, "HourlyPageviewCount")
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.zeros(vals.size, dtype=int)
    frac = np.abs(finite - np.rint(finite))
    looks_raw = float(np.mean(frac < 1e-6)) >= 0.9
    raw = vals if looks_raw else np.expm1(vals)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    return np.rint(np.clip(raw, 0.0, None)).astype(int)


def _pv_positive_pool_record(g: pd.DataFrame) -> dict:
    """Empirical positive integer counts, optionally split by walking suggestion."""
    raw = _raw_pageview_counts(g)
    ah = np.rint(
        np.clip(
            _fill_nan_with_mean(
                _df_numeric_col(g, "WalkingSuggestion"), default=0.0
            ),
            0.0,
            1.0,
        )
    ).astype(int)
    n = min(raw.size, ah.size)
    raw, ah = raw[:n], ah[:n]
    positive = raw[raw > 0].astype(int)
    pools = {a: raw[(ah == a) & (raw > 0)].astype(int) for a in (0, 1)}
    return {
        "positive_pool": positive,
        "action_positive_pools": pools,
        "n": int(raw.size),
        "n_pos": int(positive.size),
        "p_hat": float(np.mean(raw > 0)) if raw.size else 0.0,
    }


_PV_HURDLE_BUNDLE: dict = {}


def _pv_hurdle_bundle_for_params_dir(params_dir) -> dict:
    """Per-user positive page-view count pools (cached). Occurrence logit is script 4."""
    params_dir = Path(params_dir).expanduser().resolve()
    cached = _PV_HURDLE_BUNDLE.get(params_dir)
    if cached is not None:
        return cached

    bundle: dict = {"_population": np.array([], dtype=int)}
    df_path = params_dir / "df_fit_11week.csv"
    if not df_path.is_file():
        _PV_HURDLE_BUNDLE[params_dir] = bundle
        return bundle

    df = pd.read_csv(df_path)
    need = {"ParticipantIdentifier", "HourlyPageviewCount", "WalkingSuggestion"}
    if not need.issubset(df.columns):
        _PV_HURDLE_BUNDLE[params_dir] = bundle
        return bundle

    df["ParticipantIdentifier"] = df["ParticipantIdentifier"].map(_uid_key)
    population = []
    for uid, g in df.groupby("ParticipantIdentifier", sort=False):
        uid = _uid_key(uid)
        rec = _pv_positive_pool_record(g)
        bundle[uid] = rec
        population.extend(rec["positive_pool"].tolist())
    bundle["_population"] = np.asarray(population, dtype=int)
    _PV_HURDLE_BUNDLE[params_dir] = bundle
    if os.environ.get("ADAPR_PV_HURDLE_LOG", "").strip() not in {"", "0", "false"}:
        phats = [v["p_hat"] for v in bundle.values() if isinstance(v, dict) and "p_hat" in v]
        npos = [v["n_pos"] for v in bundle.values() if isinstance(v, dict) and "n_pos" in v]
        if phats:
            print(
                f"Page-view hurdle pools: n={len(phats)} "
                f"median P(PV>0)={float(np.median(phats)):.3f} "
                f"median n_pos={float(np.median(npos)):.0f}"
            )
    return bundle


def _pv_hurdle_positive_pool(rec: dict, Ah: float, population: np.ndarray) -> np.ndarray:
    a = 0 if float(Ah) < 0.5 else 1
    pool = np.asarray(rec["action_positive_pools"].get(a, []), dtype=int)
    if pool.size < 5:
        pool = np.asarray(rec["positive_pool"], dtype=int)
    if pool.size == 0:
        pool = np.asarray(population, dtype=int)
    if pool.size == 0:
        pool = np.array([1], dtype=int)
    return pool


def _standardize_pageview_count(raw, shift: float, scale: float, zero_count=None) -> float:
    """Map a raw hourly pageview count onto script-3 ``HourlyPageviewCount_norm``.

    Current convention (``zero_count is None``): ``log(x+1)`` for every count,
    so count 0 sits at ``(0 - shift) / scale``. Legacy folders written with
    ``HourlyPageviewCount_zero_count`` use ``log(x)`` for positives and
    ``log(zero_count)`` for zero.
    """
    s = float(scale) if float(scale) != 0.0 and math.isfinite(float(scale)) else 1.0
    try:
        x = float(raw)
    except (TypeError, ValueError):
        x = 0.0
    if not math.isfinite(x) or x < 0.0:
        x = 0.0
    if zero_count is None:
        return (math.log1p(x) - float(shift)) / s
    if x <= 0.0:
        x = float(zero_count) if float(zero_count) > 0.0 else 0.5
    return (math.log(x) - float(shift)) / s


def _pageview_zero_count(std: dict):
    """``None`` for the log1p convention, else the legacy zero placement."""
    zc = std.get("HourlyPageviewCount_zero_count")
    return None if zc is None else float(zc)


def pv_hurdle_occurrence_z_gap(std: dict) -> float:
    """``z̄₊ - z_0`` on ``HourlyPageviewCount_norm``.

    Positives are z-scored among themselves, so ``z̄₊ = 0``. Count 0 sits at
    ``(log1p(0) - shift) / scale`` (legacy: ``(log(zero_count) - shift) / scale``). The E_w loop-gain proxy multiplies
    the occurrence slope by this gap (script 4 and ``tune_ste.loop_gain_Ew``).
    """
    shift = float(std["HourlyPageviewCount_shift"])
    scale = float(std["HourlyPageviewCount_scale"])
    zc = _pageview_zero_count(std)
    return float(-_standardize_pageview_count(0.0, shift, scale, zero_count=zc))


class EnvConfig:
    """Load environment parameters from ``params_env_{userid}.json``."""

    def __init__(
        self,
        userid,
        params_dir=PARAMS_DIR,
        nweek=36,
        activity_process_mode=None,
        activity_stayer_fraction=None,
        activity_stayer_recalibrate=None,
        activity=None,
        misspec=None,
        misspec_strengths=None,
    ):
        self.userid = userid
        self.K = 2
        self.W = 7
        self.nweek = int(nweek)
        self.D = self.nweek * self.W

        params_path = Path(params_dir).expanduser().resolve()
        self.params_dir = params_path
        calibration_enabled = (
            os.environ.get("ADAPR_GENERATOR_CALIBRATION", "1")
            .strip()
            .lower()
            not in {"0", "false", "no", "off"}
        )
        self.dropout_hazard_day = 0.0
        self.activity_process = None
        self.activity_lambda = None
        self.activity_pi = None
        mode = activity_process_mode
        if mode is None:
            mode = os.environ.get("ADAPR_ACTIVITY_PROCESS_MODE", "pooled")
        mode = str(mode).strip().lower()
        if mode not in {"frailty", "pooled"}:
            raise ValueError(
                "activity_process_mode must be 'frailty' or 'pooled', "
                f"got {activity_process_mode!r}"
            )
        self.activity_process_mode = mode
        if activity_stayer_fraction is None:
            from testbed import parse_stayer_fraction

            fraction = parse_stayer_fraction()
        else:
            fraction = float(activity_stayer_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"activity_stayer_fraction must be in [0, 1], got {activity_stayer_fraction!r}"
            )
        self.activity_stayer_fraction = fraction
        if activity_stayer_recalibrate is None:
            raw = os.environ.get("ADAPR_ACTIVITY_STAYER_RECALIBRATE")
            recalibrate = True if raw is None or raw.strip() == "" else raw.strip().lower() not in {
                "0", "false", "no", "off",
            }
        else:
            recalibrate = bool(activity_stayer_recalibrate)
        self.activity_stayer_recalibrate = recalibrate
        if activity is None:
            from testbed import parse_activity

            activity = parse_activity()
        else:
            from testbed import parse_activity

            activity = parse_activity(activity)
        self.activity = activity
        if misspec is None:
            from testbed import parse_misspec

            misspec = parse_misspec()
        else:
            from testbed import parse_misspec

            misspec = parse_misspec(misspec)
        self.misspec = misspec
        self._misspec_strengths = dict(misspec_strengths or {})
        if self.activity == "off" and self.activity_stayer_fraction != 0.0:
            warnings.warn(
                f"stayer fraction {self.activity_stayer_fraction:g} is ignored "
                "when activity is off",
                RuntimeWarning,
                stacklevel=2,
            )
            self.activity_stayer_fraction = 0.0
        if self.misspec == "D" and self.activity != "on":
            raise ValueError("misspec D requires activity on")
        # ``activity`` uses the stored process. ``dropout_*`` in the same
        # file are comparison records, not a second daily draw. The legacy
        # constant-hazard path is used only when a caller sets
        # ``dropout_hazard_day`` (or ``activity_lambda``) explicitly.
        self.dropout_mechanism = "none"
        calibration_path = params_path / "generator_calibration.json"
        if calibration_enabled:
            calibration = (
                _load_json(calibration_path) if calibration_path.is_file() else {}
            )
            has_activity = isinstance(calibration.get("activity_process"), dict)
            if has_activity:
                self.activity_process = calibration["activity_process"]
                self.dropout_mechanism = "activity"
        # Activity off is the existing lambda = 0 hook. Set after the
        # calibration load so it wins over a stored activity_process.
        # Activity off needs no activity_process block.
        if self.activity == "off":
            self.activity_lambda = 0.0
        elif self.activity_process is None:
            raise FileNotFoundError(
                f"{calibration_path} has no activity_process; "
                "activity='on' requires that block (no silent zero-dropout "
                "fallback). Use activity='off' when no activity_process is stored."
            )
        std = _load_json(params_path / "std_params.json")
        self.limits_fourSC = std["4hour_step_count_limit"]
        self.fourSC_log_shift = float(std["4hour_step_count_shift"])
        self.fourSC_log_scale = float(std["4hour_step_count_scale"])
        self.limits_pageview = std["HourlyPageviewCount_limit"]
        self.pageview_log_shift = float(std["HourlyPageviewCount_shift"])
        self.pageview_log_scale = float(std["HourlyPageviewCount_scale"])
        self.pageview_zero_count = _pageview_zero_count(std)
        self.limits_CAE = std["CAE_avg_limit"]
        self.limits_CAE_short = std["CAE_short_avg_limit"]
        # Affine map back to the raw (pre-normalization) CAE level:
        #   raw_CAE = CAE_shift + CAE_scale * CAE_avg_norm
        self.CAE_shift = float(std["CAE_avg_shift"])
        self.CAE_scale = float(std["CAE_avg_scale"])
        self.limits_antic = std["anticipated_affect_yesterday_limit"]
        self.limits_fitbitwearing = [0.0, 1.0]
        self.limits_dailysurvey = [0.0, 1.0]
        # Default E range; replaced by ``E_limits`` from params_env (4b) below.
        self.limits_perceivedUtility = [E_QUAD_LO, E_QUAD_HI]
        self.limits_week_present = [0.0, 1.0]
        self.limits_prior2hour_step_count = std["prior2hour_step_count_limit"]
        self.prior2hour_log_shift = float(std["prior2hour_step_count_shift"])
        self.prior2hour_log_scale = float(std["prior2hour_step_count_scale"])
        self.ema_prior2hour_shift = float(std["EMA_prior2hour_step_count_shift"])
        self.ema_prior2hour_scale = float(std["EMA_prior2hour_step_count_scale"])
        self.recent_burden_shift = float(std["recent_burden_shift"])
        self.recent_burden_scale = float(std["recent_burden_scale"])
        # Derived history covariates (derived_covariates.py), log scale -> z.
        self.yesterday_step_shift = float(std["yesterday_step_count_shift"])
        self.yesterday_step_scale = float(std["yesterday_step_count_scale"])
        self.pv7_shift = float(std["Past7DaysPageviewEMA_shift"])
        self.pv7_scale = float(std["Past7DaysPageviewEMA_scale"])
        self.limits_active_status = [0.0, 1.0]
        self.limits_ws_interaction = [0.0, 1.0]
        # Tool surveys are normalized from the raw 1..7 scale to [-1, 1].
        self.limits_exp1 = [-1.0, 1.0]
        self.limits_exp2 = [-1.0, 1.0]
        p = _load_json(params_path / f"params_env_{userid}.json")

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
            legacy_lengths=(
                _P_FOURSC_WITH_WEEKEND,
                _P_FOURSC_WITH_WEEKEND_AND_SALIENCE,
            ),
        )
        _validate_json_names(
            "theta_antic_names",
            p,
            THETA_ANTIC_NAMES,
            legacy_lengths=(
                P_ANTIC,
                _P_ANTIC_NO_ACT7,
                _P_ANTIC_WITH_WEEKEND,
                _P_ANTIC_NO_BURDEN,
                _P_ANTIC_NO_BURDEN + 1,
                _P_ANTIC_NO_BURDEN + 4,
            ),
        )
        _validate_json_names(
            "theta_CAE_names",
            p,
            THETA_CAE_NAMES,
            legacy_lengths=(_LEGACY_CAE_WITH_WEEK,),
        )
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
        self.theta_antic = trim_theta_antic(
            _json_float_list("theta_antic", p),
            names=p.get("theta_antic_names"),
        )
        self.theta_CAE = trim_theta_cae(_json_float_list("theta_CAE", p))
        self.theta_CAE_short = _json_float_list("theta_CAE_short_avg", p)

        self.resid_prior2hour_step_count = _json_resid_list("resid_prior2hour_step_count", p)
        self.resid_active_status = _json_resid_list("resid_active_status", p)
        self.resid_ws_interaction = _json_resid_list("resid_ws_interaction", p)
        self.resid_fourSC = _json_resid_list("resid_fourSC", p)
        self.resid_antic = _json_resid_list("resid_antic", p)
        self.resid_CAE = _json_resid_list("resid_CAE", p)
        self.resid_week_present = _json_resid_list("resid_week_present", p)
        self.resid_CAE_short = _json_resid_list("resid_CAE_short_avg", p)
        self.resid_pageview_occurrence = _json_resid_list("resid_penalized_pageview_occurrence", p)
        # Interaction with a delivered suggestion -> page-view occurrence (4b).
        self.theta_pv_interaction = np.asarray(
            p.get("theta_penalized_PV_interaction") or [], dtype=float
        ).ravel()
        # E clipping range from the perceived-utility fit (pooled posterior range).
        if p.get("E_limits") is not None:
            self.limits_perceivedUtility = [float(x) for x in p["E_limits"]]

        def _penalized_or_legacy(new_key, old_key):
            return np.asarray(
                p.get(new_key) or p.get(old_key) or [],
                dtype=float,
            ).ravel()

        self.theta_ml_Ew = _penalized_or_legacy("theta_penalized_Ew", "theta_ml_Ew")
        self.theta_ml_J = _trim_theta_j(
            _penalized_or_legacy("theta_penalized_J", "theta_ml_J")
        )
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
        self._install_misspec_strengths()
        self._validate_shapes()
        self._apply_pv_hurdle()

    def _install_misspec_strengths(self) -> None:
        """Residual SDs and stored A/B strengths. Unused when misspec is none."""
        from misspec import require_A, require_B, residual_sigma

        self.sigma_SC = residual_sigma(self.resid_fourSC)
        self.sigma_AA = residual_sigma(self.resid_antic)
        self.sigma_Y = residual_sigma(self.resid_CAE)
        self.misspec_lam_Y = 0.5
        self.misspec_lam_SC = None
        self.misspec_lam_AA = None
        self.misspec_beta_D = None
        self.misspec_rho = 0.5
        overrides = self._misspec_strengths
        if self.misspec == "A":
            got = require_A(self.params_dir, overrides or None)
            self.misspec_lam_Y = float(got["lam_Y"])
            self.misspec_rho = float(got["rho"])
            self.misspec_lam_SC = float(got["lam_SC"])
            self.misspec_lam_AA = float(got["lam_AA"])
        elif self.misspec == "B":
            self.misspec_beta_D = require_B(
                self.params_dir, overrides.get("beta_D") if overrides else None
            )

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

    def _apply_pv_hurdle(self) -> None:
        """Load this participant's empirical positive page-view count pool."""
        self.pv_hurdle = None
        self.pv_hurdle_population = np.array([], dtype=int)
        bundle = _pv_hurdle_bundle_for_params_dir(self.params_dir)
        rec = bundle.get(_uid_key(self.userid))
        pop = np.asarray(bundle.get("_population", []), dtype=int)
        if rec is None:
            rec = {
                "positive_pool": pop.copy(),
                "action_positive_pools": {0: pop.copy(), 1: pop.copy()},
                "n": 0,
                "n_pos": int(pop.size),
                "p_hat": float("nan"),
            }
        self.pv_hurdle = {
            "positive_pool": np.asarray(rec["positive_pool"], dtype=int).copy(),
            "action_positive_pools": {
                int(a): np.asarray(p, dtype=int).copy()
                for a, p in rec["action_positive_pools"].items()
            },
            "n": rec["n"],
            "n_pos": rec["n_pos"],
            "p_hat": rec["p_hat"],
        }
        self.pv_hurdle_population = pop.copy()


# --------------------------------------------------------------------------
# Residual bootstrap with credibility pooling (one rule for every stream)
# --------------------------------------------------------------------------
# Each stream's simulated residuals follow
#     e_t = mu_i + rho_i (e_{t-1} - mu_i) + sqrt(1 - rho_i^2) z_t,
# where mu_i is the participant's residual mean, and z_t is drawn from the
# participant's own mean-centred residual pool with probability w_i and from
# the population's pooled (participant-centred) residual pool otherwise.
# w_i and rho_i are empirical-Bayes (credibility) weights estimated from the
# between- and within-participant variation of all participants' residuals:
#   * w_i  = tau_v^2 / (tau_v^2 + 2/(n_i - 1)), where 2/(n_i - 1) is the
#            sampling variance of the log residual variance and tau_v^2 its
#            between-participant variance;
#   * rho_i = w_rho rho_hat_i + (1 - w_rho) rho_bar, with
#            w_rho = tau_rho^2 / (tau_rho^2 + (1 - rho_bar^2)/m_i), m_i the
#            number of adjacent residual pairs.
# Participants with many residuals keep their own distribution and serial
# dependence; sparse ones borrow from the population.

RESIDUAL_KEYS = {
    "prior2hour_step_count": ("resid_prior2hour_step_count",),
    "active_status": ("resid_active_status",),
    "ws_interaction": ("resid_ws_interaction",),
    "fourSC": ("resid_fourSC",),
    "antic": ("resid_antic",),
    "CAE": ("resid_CAE",),
    "CAE_short": ("resid_CAE_short_avg",),
    "hourly_pageview": ("resid_penalized_hourly_pageview", "resid_ml_hourly_pageview"),
    "pageview_occurrence": ("resid_penalized_pageview_occurrence",),
    "nextday_wearing": ("resid_penalized_nextday_wearing", "resid_ml_nextday_wearing"),
    "daily_present": ("resid_penalized_daily_present", "resid_ml_daily_present"),
    "J_week": ("resid_penalized_J_week", "resid_ml_J_week", "resid_week_present"),
    "U1": ("resid_penalized_U1", "resid_ml_U1"),
    "U2": ("resid_penalized_U2", "resid_ml_U2"),
}
RHO_GUARD = 0.99  # numerical guard only (stationarity); shrinkage keeps |rho| below it
_POOL_STATS_CACHE: dict = {}


def _lag1_pairs(resid):
    r = np.asarray(resid, dtype=float).ravel()
    if r.size < 2:
        return np.empty(0), np.empty(0)
    x, y = r[:-1], r[1:]
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _lag1_autocorr(resid) -> tuple[float, int]:
    """Lag-1 correlation of adjacent observed residuals and the number of pairs."""
    x, y = _lag1_pairs(resid)
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0, int(x.size)
    rho = float(np.corrcoef(x, y)[0, 1])
    return (rho if np.isfinite(rho) else 0.0), int(x.size)


def _stream_residuals(params: dict, name: str) -> np.ndarray:
    for key in RESIDUAL_KEYS.get(name, ()):
        if params.get(key) is not None:
            return _json_resid_list(key, params)
    return np.empty(0)


def residual_pool_stats(params_dir, name: str) -> dict:
    """Population quantities for stream ``name`` from all participants in ``params_dir``."""
    params_dir = Path(params_dir).expanduser().resolve()
    key = (str(params_dir), name)
    if key in _POOL_STATS_CACHE:
        return _POOL_STATS_CACHE[key]
    log_v, samp_v, rhos, samp_n, pool = [], [], [], [], []
    for f in sorted(params_dir.glob("params_env_*.json")):
        r = _stream_residuals(_load_json(f), name)
        a = r[np.isfinite(r)]
        if a.size == 0:
            continue
        pool.append(a - a.mean())
        if a.size >= 3 and np.var(a, ddof=1) > 0:
            log_v.append(np.log(np.var(a, ddof=1)))
            samp_v.append(2.0 / (a.size - 1))
        rho, m = _lag1_autocorr(r)
        if m >= 3:
            rhos.append(rho)
            samp_n.append(m)
    pool = np.concatenate(pool) if pool else np.zeros(1)
    tau2_v = max(float(np.var(log_v) - np.mean(samp_v)), 1e-6) if len(log_v) > 1 else 1e-6
    if rhos:
        w = np.asarray(samp_n, dtype=float)
        rho_bar = float(np.sum(w * np.asarray(rhos)) / np.sum(w))
        samp_r = (1.0 - rho_bar ** 2) / w
        tau2_rho = max(float(np.var(rhos) - np.mean(samp_r)), 1e-6) if len(rhos) > 1 else 1e-6
    else:
        rho_bar, tau2_rho = 0.0, 1e-6
    out = {"pool": pool, "tau2_v": tau2_v, "rho_bar": rho_bar, "tau2_rho": tau2_rho}
    _POOL_STATS_CACHE[key] = out
    return out


def participant_residual_model(resid, stats: dict) -> dict:
    """Credibility-weighted residual model (mean, own-pool weight, rho, pools)."""
    r = np.asarray(resid, dtype=float).ravel()
    own = r[np.isfinite(r)]
    n = own.size
    mu = float(own.mean()) if n else 0.0
    w_own = stats["tau2_v"] / (stats["tau2_v"] + 2.0 / (n - 1)) if n >= 2 else 0.0
    rho_hat, m = _lag1_autocorr(r)
    rho_bar = stats["rho_bar"]
    if m >= 3:
        w_rho = stats["tau2_rho"] / (stats["tau2_rho"] + (1.0 - rho_bar ** 2) / m)
        rho = w_rho * rho_hat + (1.0 - w_rho) * rho_bar
    else:
        rho = rho_bar
    return {
        "mu": mu,
        "w_own": float(w_own),
        "rho": float(np.clip(rho, -RHO_GUARD, RHO_GUARD)),
        "own": own - mu if n else np.zeros(0),
        "pop": stats["pool"],
    }


class Env:
    """Generative environment: vanilla Ridge/logistic + ML perceived-utility stack."""

    def __init__(self, env_config: EnvConfig, noise="ar1"):
        # "none": deterministic skeleton (residuals 0, Bernoulli variables at
        # their probabilities), used by the stability check.
        assert noise in ("sequential", "random", "ar1", "none")
        self.noise = noise
        self.deterministic = noise == "none"
        self.cfg = env_config
        self.K = env_config.K
        self.W = env_config.W
        # Per-stream state for the "ar1" noise model: last simulated residual
        # value and cached lag-1 autocorrelation, keyed by stream name so a
        # fresh Env (created per simulated episode) starts with fresh state.
        self._ar_prev: dict = {}
        self._resid_models: dict = {}
        self.protocol_week = 1
        self._confounder_C = None
        self._week_action_frac = None

    def install_confounder(self, seed, uid, nweek=None) -> None:
        """Draw ``C_w`` from its own ``(seed, uid)`` child. No-op unless misspec A."""
        if getattr(self.cfg, "misspec", "none") != "A":
            self._confounder_C = None
            return
        from misspec import draw_confounder

        n = int(self.cfg.nweek if nweek is None else nweek)
        self._confounder_C = draw_confounder(seed, uid, n, rho=float(self.cfg.misspec_rho))

    def _confounder_at_week(self) -> float:
        c = self._confounder_C
        if c is None:
            return 0.0
        idx = max(int(self.protocol_week) - 1, 0)
        if idx >= c.size:
            idx = int(c.size) - 1
        return float(c[idx])

    def _action_scale(self) -> float:
        from misspec import action_scale

        return action_scale(getattr(self.cfg, "misspec", "none"), self.protocol_week)

    def dropout_hazard(self, state) -> float:
        """Legacy constant daily hazard.

        Used only when ``cfg.dropout_mechanism`` is ``constant``. An
        ``activity_process`` record does not consult this value. ``state``
        is accepted so the hook stays in place.
        """
        del state
        return float(self.cfg.dropout_hazard_day)

    def pageview_absent_z(self) -> float:
        """Standardized page-view value of a zero count."""
        z = _standardize_pageview_count(
            0.0,
            self.cfg.pageview_log_shift,
            self.cfg.pageview_log_scale,
            zero_count=self.cfg.pageview_zero_count,
        )
        return float(np.clip(z, *self.cfg.limits_pageview))

    def _resid_model(self, resid, name):
        key = name if name is not None else id(resid)
        model = self._resid_models.get(key)
        if model is None:
            stats = residual_pool_stats(self.cfg.params_dir, name) if name in RESIDUAL_KEYS else {
                "pool": np.zeros(1), "tau2_v": 1e-6, "rho_bar": 0.0, "tau2_rho": 1e-6}
            model = participant_residual_model(resid, stats)
            self._resid_models[key] = model
        return key, model

    def _innovation(self, model) -> float:
        """Mean-zero innovation: own pool w.p. ``w_own``, else the population pool."""
        own, pop = model["own"], model["pop"]
        if own.size and rd.random() < model["w_own"]:
            return float(rd.choice(own))
        return float(rd.choice(pop)) if pop.size else 0.0

    def _sample_noise(self, resid, idx, obs_resid=None, name=None):
        if self.deterministic:
            return 0.0
        key, model = self._resid_model(resid, name)
        if self.noise == "sequential":
            r = np.asarray(resid, dtype=float)
            val = r[idx % len(r)] if r.size else np.nan
            return float(val) if np.isfinite(val) else model["mu"] + self._innovation(model)
        if self.noise == "random":
            return model["mu"] + self._innovation(model)
        # AR(1) residual bootstrap
        rho, mu = model["rho"], model["mu"]
        prev = self._ar_prev.get(key)
        if prev is None:
            prev = mu + self._innovation(model)
        val = mu + rho * (prev - mu) + math.sqrt(max(1.0 - rho ** 2, 0.0)) * self._innovation(model)
        self._ar_prev[key] = val
        return float(val)

    def _bernoulli(self, p) -> float:
        """Bernoulli draw, or its probability in the deterministic skeleton."""
        p = float(np.clip(p, 0.0, 1.0))
        return p if self.deterministic else float(rd.binomial(1, p))

    @staticmethod
    def _sigmoid(eta):
        z = float(np.clip(eta, -60.0, 60.0))
        return float(1.0 / (1.0 + np.exp(-z)))

    # ----- decision-level (K=2) -----

    def gen_prior2hour_step_count_mean(self, s):
        """[1, EMA_Prior2HourStepCount, isWeekend, decisionTimeSlot,
        (1 - slot) * lag, slot * lag], lag = next-4h steps of the previous slot."""
        dt = float(s["decisionTimeSlot"])
        lag = float(s["stepCountNext4HourLag1"])
        X = np.array(
            [1.0, s["prior2HourStepCountEma7d"], s["isWeekend"], dt, (1.0 - dt) * lag, dt * lag],
            dtype=float,
        )
        return float(self.cfg.theta_prior2hour_step_count @ X)

    def gen_prior2hour_step_count(self, s, step_idx):
        mean = self.gen_prior2hour_step_count_mean(s)
        noise = self._sample_noise(
            self.cfg.resid_prior2hour_step_count, step_idx, name="prior2hour_step_count"
        )
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
        noise = self._sample_noise(self.cfg.resid_active_status, day_idx, name="active_status")
        p = float(np.clip(base_p + noise, *self.cfg.limits_active_status))
        return self._bernoulli(p)

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

    def gen_ws_interaction(self, s, step_idx, Ah=1.0):
        """Draw interaction with a delivered activity suggestion.

        ``A = 0`` means no suggestion was sent, so the interaction event is 0.
        Residual and Bernoulli draws still run so two policies that first
        differ on A stay on the same global ``numpy.random`` stream
        (same idea as ``gen_pageview`` intensity noise).
        """
        eta = self.gen_ws_interaction_mean(s, return_logit=True)
        base_p = self._sigmoid(eta)
        noise = self._sample_noise(self.cfg.resid_ws_interaction, step_idx, name="ws_interaction")
        p = float(np.clip(base_p + noise, *self.cfg.limits_ws_interaction))
        latent_draw = self._bernoulli(p)
        return float(latent_draw) if float(Ah) != 0.0 else 0.0

    def gen_fourSC_mean(self, s, Ah):
        """
        ``P_FOURSC`` predictors — same column order as script-5 ``fourSC_cond``.

        History summaries (``yesterdayStepCount``,
        ``pageViewLast7DaysEma``, ...) are defined in ``derived_covariates.py``
        and computed identically from the MRT data and from simulated data.
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
                Ah * s["decisionTimeSlot"],
                Ah * pu,
                Ah * cae,
            ],
            dtype=float,
        )
        miss = getattr(self.cfg, "misspec", "none")
        if miss == "none":
            return float(self.cfg.theta_fourSC @ X)
        from misspec import FOURSC_ACTION_IDX, scale_action_theta

        theta = scale_action_theta(
            self.cfg.theta_fourSC, FOURSC_ACTION_IDX, self._action_scale()
        )
        mu = float(theta @ X)
        if miss == "A":
            mu = mu + float(self.cfg.misspec_lam_SC) * float(self.cfg.sigma_SC) * self._confounder_at_week()
        return mu

    def gen_fourSC(self, s, Ah, step_idx):
        mean = self.gen_fourSC_mean(s, Ah)
        noise = self._sample_noise(self.cfg.resid_fourSC, step_idx, name="fourSC")
        if getattr(self.cfg, "misspec", "none") == "A":
            from misspec import residual_scale

            noise = float(noise) * residual_scale(self.cfg.misspec_lam_SC)
        return float(np.clip(mean + noise, *self.cfg.limits_fourSC))

    def _ml_pv_query_shift(self, s, Jw: int, q=None) -> float:
        if q is None:
            _, q = _split_ml_pv_full(self.cfg.theta_ml_PV)
        return _ml_query_applied_shift(q, s, Jw)

    def _ml_pv_linpred(self, s, Ah: float, Jw: int, base, q) -> float:
        Ew = float(s["perceivedUtilityLastWeek"])
        (
            b0,
            b1,
            b2_isWeekend,
            b2_dt,
            b2_rb,
            b_ar1,
            b3,
            b4,
        ) = np.asarray(base, dtype=float).ravel()
        lag1 = float(s["pageViewNext4HourLag1"])
        eta = (
            b0
            + b1 * Ew
            + b2_isWeekend * float(s["isWeekend"])
            + b2_dt * float(s["decisionTimeSlot"])
            + b2_rb * float(s["activitySuggestionsSentLast7Days"])
            + b_ar1 * lag1
            + Ah * (b3 + b4 * Ew)
        )
        return float(eta + self._ml_pv_query_shift(s, Jw, q=q))

    def _ml_pv_logit(self, s, Ah: float, Jw: int) -> float:
        """Script-06 hurdle logit for ``P(pageview count > 0)``."""
        if not self.cfg.has_ml_stack:
            raise RuntimeError("ML parameters missing from params JSON")
        occ_base, occ_q, _int_base, _int_q, _sigma = _split_ml_pv_hurdle(
            self.cfg.theta_ml_PV
        )
        return self._ml_pv_linpred(s, float(Ah), int(Jw), occ_base, occ_q)

    def _ml_pv_intensity_mean(self, s, Ah: float, Jw: int):
        """Mean log-then-z pageview given count>0, or None if not in JSON."""
        if not self.cfg.has_ml_stack:
            raise RuntimeError("ML parameters missing from params JSON")
        _occ_base, _occ_q, int_base, int_q, sigma = _split_ml_pv_hurdle(
            self.cfg.theta_ml_PV
        )
        if int_base is None:
            return None, None
        mu = self._ml_pv_linpred(s, float(Ah), int(Jw), int_base, int_q)
        return float(mu), float(sigma)

    def _pv_hurdle_occurrence(self, s, Ah: float, Jw: int):
        rec = self.cfg.pv_hurdle
        p = float(self._sigmoid(self._ml_pv_logit(s, float(Ah), int(Jw))))
        pool = _pv_hurdle_positive_pool(rec, Ah, self.cfg.pv_hurdle_population)
        shift = float(self.cfg.pageview_log_shift)
        scale = float(self.cfg.pageview_log_scale)
        zc = self.cfg.pageview_zero_count
        z_zero = _standardize_pageview_count(0.0, shift, scale, zero_count=zc)
        z_one = _standardize_pageview_count(1.0, shift, scale, zero_count=zc)
        mu_int, sigma_int = self._ml_pv_intensity_mean(s, float(Ah), int(Jw))
        if mu_int is None:
            z_pos = np.array(
                [
                    _standardize_pageview_count(v, shift, scale, zero_count=zc)
                    for v in pool
                ],
                dtype=float,
            )
            mu_int = float(np.mean(z_pos)) if z_pos.size else z_one
            sigma_int = 0.1
        return p, pool, z_zero, z_one, float(mu_int), float(sigma_int)

    def _pv_interaction_shift(self, Ah, ws_interaction) -> float:
        """Occurrence-logit term for interaction with a delivered suggestion."""
        if ws_interaction is None or self.cfg.theta_pv_interaction.size == 0:
            return 0.0
        return float(self.cfg.theta_pv_interaction[0]) * float(Ah) * float(ws_interaction)

    def gen_pageview_mean(self, s, Ah, Jw=0, ws_interaction=None):
        p_hat, _pool, z_zero, _z_one, mu_int, _sigma = self._pv_hurdle_occurrence(
            s, float(Ah), int(Jw)
        )
        if ws_interaction is not None:
            eta = math.log(max(p_hat, 1e-12) / max(1.0 - p_hat, 1e-12))
            p_hat = self._sigmoid(eta + self._pv_interaction_shift(Ah, ws_interaction))
        mu = (1.0 - p_hat) * z_zero + p_hat * mu_int
        return float(np.clip(mu, *self.cfg.limits_pageview))

    def gen_pageview(self, s, Ah, Jw, step_idx, ws_interaction=None):
        p_base, pool, z_zero, z_one, mu_int, sigma_int = self._pv_hurdle_occurrence(
            s, float(Ah), int(Jw)
        )
        eta = math.log(max(p_base, 1e-12) / max(1.0 - p_base, 1e-12))
        p_hat = self._sigmoid(eta + self._pv_interaction_shift(Ah, ws_interaction))
        # Occurrence: same rule as the other binary variables (probability +
        # bootstrapped response residual, then Bernoulli).
        occ_noise = self._sample_noise(
            self.cfg.resid_pageview_occurrence, step_idx, name="pageview_occurrence"
        )
        p_hat = float(np.clip(p_hat + occ_noise, 0.0, 1.0))
        u = rd.random()
        # Always consume intensity noise so policies that differ on P(count=0)
        # stay aligned on the global RNG stream.
        if (
            self.cfg.resid_ml_hourly_pageview.size > 0
            and np.any(np.isfinite(self.cfg.resid_ml_hourly_pageview))
        ):
            noise = self._sample_noise(
                self.cfg.resid_ml_hourly_pageview, step_idx, name="hourly_pageview"
            )
        else:
            noise = float(rd.normal(0.0, sigma_int))
        if self.deterministic:
            z = p_hat * max(mu_int, z_one) + (1.0 - p_hat) * z_zero
            return float(np.clip(z, *self.cfg.limits_pageview))
        if u >= p_hat:
            return float(np.clip(z_zero, *self.cfg.limits_pageview))
        _occ_base, _occ_q, int_base, _int_q, _sig = _split_ml_pv_hurdle(
            self.cfg.theta_ml_PV
        )
        if int_base is None:
            pick = int(rd.choice(pool)) if pool.size else 1
            z = _standardize_pageview_count(
                pick,
                self.cfg.pageview_log_shift,
                self.cfg.pageview_log_scale,
                zero_count=self.cfg.pageview_zero_count,
            )
            return float(np.clip(z, *self.cfg.limits_pageview))
        z = mu_int + (0.0 if not np.isfinite(noise) else float(noise))
        z = max(z, z_one)
        return float(np.clip(z, *self.cfg.limits_pageview))

    # ----- daily mediators -----

    def gen_antic_mean(self, s, ws_morning, ws_afternoon):
        """
        Daily ridge design (morning row): ``P_ANTIC`` columns — matches
        ``07_fit_testbed`` ``anticipated_affect_cond_day``
        (no ``today_step_count`` / ``planning_prompt``; includes
        ``perceivedUtilityLastWeek`` and ``recent_burden`` mains and
        AM/PM interactions). Uses the prior 7-day active fraction, not
        today's binary active status.
        """
        act = float(s["activeDaysLast7Days"])
        is_weekend = float(s["isWeekend"])
        pu = float(s["perceivedUtilityLastWeek"])
        cae = float(s["caeAverageLastWeek"])
        rb = float(s["activitySuggestionsSentLast7Days"])
        X = np.array(
            [
                1.0,
                float(s["dailyAnticipatedAffectYesterday"]),
                act,
                is_weekend,
                pu,
                cae,
                rb,
                ws_morning,
                ws_afternoon,
                ws_morning * pu,
                ws_afternoon * pu,
                ws_morning * cae,
                ws_afternoon * cae,
                ws_morning * rb,
                ws_afternoon * rb,
                ws_morning * act,
                ws_afternoon * act,
            ],
            dtype=float,
        )
        miss = getattr(self.cfg, "misspec", "none")
        if miss == "none":
            return float(self.cfg.theta_antic @ X)
        from misspec import ANTIC_ACTION_IDX, scale_action_theta

        theta = scale_action_theta(
            self.cfg.theta_antic, ANTIC_ACTION_IDX, self._action_scale()
        )
        mu = float(theta @ X)
        if miss == "A":
            mu = mu + float(self.cfg.misspec_lam_AA) * float(self.cfg.sigma_AA) * self._confounder_at_week()
        return mu

    def gen_antic(self, s, ws_morning, ws_afternoon, day_idx):
        mean = self.gen_antic_mean(s, ws_morning, ws_afternoon)
        noise = self._sample_noise(self.cfg.resid_antic, day_idx, name="antic")
        if getattr(self.cfg, "misspec", "none") == "A":
            from misspec import residual_scale

            noise = float(noise) * residual_scale(self.cfg.misspec_lam_AA)
        return float(np.clip(mean + noise, *self.cfg.limits_antic))

    def _ml_fw_eta(self, s, ws_morning, ws_afternoon, Jw: int, return_logit: bool):
        if not self.cfg.has_ml_stack:
            raise RuntimeError("ML parameters missing from params JSON")
        Ew = float(s["perceivedUtilityLastWeek"])
        base, q = _split_ml_fw(self.cfg.theta_ml_FW)
        (
            beta0, beta1, b2_isWeekend, b2_rb, beta_ar1,
            beta3, beta4, beta5, beta6,
        ) = base
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
        eta += _ml_query_applied_shift(q, s, Jw)
        return eta if return_logit else self._sigmoid(eta)

    def gen_fitbitwearing_mean(self, s, ws_morning, ws_afternoon, Jw=0, return_logit=False):
        return self._ml_fw_eta(
            s, ws_morning, ws_afternoon, int(Jw), return_logit
        )

    def gen_fitbitwearing(self, s, ws_morning, ws_afternoon, Jw, day_idx):
        eta = self._ml_fw_eta(
            s, ws_morning, ws_afternoon, int(Jw), return_logit=True
        )
        base_p = self._sigmoid(eta)
        noise = self._sample_noise(
            self.cfg.resid_ml_nextday_wearing, day_idx, name="nextday_wearing"
        )
        if not np.isfinite(noise):
            noise = 0.0
        p = float(np.clip(base_p + noise, *self.cfg.limits_fitbitwearing))
        return self._bernoulli(p)

    def _ml_pj_eta(self, s, ws_morning, ws_afternoon, Jw: int, return_logit: bool):
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
        eta += _ml_query_applied_shift(q, s, Jw)
        return eta if return_logit else self._sigmoid(eta)

    def gen_dailysurvey_mean(self, s, ws_morning, ws_afternoon, Jw=0, return_logit=False):
        return self._ml_pj_eta(
            s, ws_morning, ws_afternoon, int(Jw), return_logit
        )

    def gen_dailysurvey(self, s, ws_morning, ws_afternoon, Jw, day_idx):
        eta = self._ml_pj_eta(
            s, ws_morning, ws_afternoon, int(Jw), return_logit=True
        )
        base_p = self._sigmoid(eta)
        noise = self._sample_noise(
            self.cfg.resid_ml_daily_present, day_idx, name="daily_present"
        )
        if not np.isfinite(noise):
            noise = 0.0
        p = float(np.clip(base_p + noise, *self.cfg.limits_dailysurvey))
        return self._bernoulli(p)

    # ----- weekly -----

    def gen_CAE_mean(self, CAE_lastweek, foursc_wk, antic_wk):
        X = build_CAE_features(CAE_lastweek, foursc_wk, antic_wk)
        miss = getattr(self.cfg, "misspec", "none")
        if miss == "none":
            return float(self.cfg.theta_CAE @ X)
        from misspec import CAUSAL_ALPHA_SCALE, scale_cae_alphas

        theta = self.cfg.theta_CAE
        if miss == "A":
            theta = scale_cae_alphas(theta, CAUSAL_ALPHA_SCALE)
        mu = float(np.asarray(theta, dtype=float) @ X)
        if miss == "A":
            mu = mu + float(self.cfg.misspec_lam_Y) * float(self.cfg.sigma_Y) * self._confounder_at_week()
        if miss == "B" and self._week_action_frac is not None:
            mu = mu + float(self.cfg.misspec_beta_D) * float(self._week_action_frac)
        return mu

    def gen_CAE(self, CAE_lastweek, foursc_wk, antic_wk, week_idx):
        mean = self.gen_CAE_mean(CAE_lastweek, foursc_wk, antic_wk)
        noise = self._sample_noise(self.cfg.resid_CAE, week_idx, name="CAE")
        if getattr(self.cfg, "misspec", "none") == "A":
            from misspec import residual_scale

            noise = float(noise) * residual_scale(self.cfg.misspec_lam_Y)
        return float(np.clip(mean + noise, *self.cfg.limits_CAE))

    @staticmethod
    def _week_means_from_arrays(pw_wk, dw_wk, dp_wk):
        """Weekly PV/FW/PJ averages for the E_w transition.

        Fixed denominators: ``nansum(pv)/14``, ``nansum(FW|PJ)/7``. NaN
        slots contribute 0, matching ``06_fit_pu.build_user_blocks``.
        For FW that means a missing wear flag is treated as not wearing.
        """
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

    def gen_perceivedUtility_mean(self, Ew_carry, pw_wk, dw_wk, dp_wk):
        """Gaussian AR on ``E_w`` from last week's state and this week's mediators."""
        if self.cfg.theta_ml_Ew.size < 6:
            raise RuntimeError("theta_ml_Ew required for latent E_w transition")
        a0, a1, a2, a3, a4, _sigma_e = [float(x) for x in self.cfg.theta_ml_Ew[:6]]
        pv_m, fw_m, pj_m = self._week_means_from_arrays(pw_wk, dw_wk, dp_wk)
        return float(a0 + a1 * float(Ew_carry) + a2 * pv_m + a3 * fw_m + a4 * pj_m)

    def gen_perceivedUtility(self, Ew_carry, pw_wk, dw_wk, dp_wk, week_idx):
        del week_idx
        mean = self.gen_perceivedUtility_mean(Ew_carry, pw_wk, dw_wk, dp_wk)
        sigma_e = float(self.cfg.theta_ml_Ew[5]) if self.cfg.theta_ml_Ew.size > 5 else 0.1
        noise = 0.0 if self.deterministic else float(rd.normal(0.0, sigma_e))
        return float(np.clip(mean + noise, *self.cfg.limits_perceivedUtility))

    def gen_week_present_mean(self, perceivedUtility, return_logit=False):
        if self.cfg.theta_ml_J.size < 2:
            raise RuntimeError("theta_ml_J required for week_present (J_week)")
        b0, b1 = float(self.cfg.theta_ml_J[0]), float(self.cfg.theta_ml_J[1])
        eta = b0 + b1 * float(perceivedUtility)
        return eta if return_logit else self._sigmoid(eta)

    def gen_week_present(self, perceivedUtility, week_idx):
        # ``week_idx`` is the weekly-array index of this J (``sim_w + 1``).
        eta = self.gen_week_present_mean(perceivedUtility, return_logit=True)
        base_p = self._sigmoid(eta)
        noise = self._sample_noise(self.cfg.resid_ml_J_week, week_idx, name="J_week")
        if not np.isfinite(noise):
            noise = self._sample_noise(
                self.cfg.resid_week_present, week_idx, name="week_present"
            )
        lim = self.cfg.limits_week_present
        p = float(np.clip(base_p + noise, *lim))
        return self._bernoulli(p)

    def gen_CAE_short_mean(self, caeAverage):
        return float(self.cfg.theta_CAE_short @ build_CAE_short_features(caeAverage))

    def gen_CAE_short(self, caeAverage, week_idx):
        mean = self.gen_CAE_short_mean(caeAverage)
        noise = self._sample_noise(self.cfg.resid_CAE_short, week_idx, name="CAE_short")
        return float(np.clip(mean + noise, *self.cfg.limits_CAE_short))

    # ----- weekly tool surveys (Exp-tool-1 / Exp-tool-2) -----
    # Used both as observed weekly outcomes when ``J_w == 1`` and as inputs to the
    # agent's pooled-linear E_w approximation in ``OnlineEnv`` (``est_Ew_weights``).
    #
    # ``theta_ml_U1`` / ``theta_ml_U2`` were fit on the *normalized* scale
    #   Exp-tool-i_norm = 2 * (Exp-tool-i - 1) / 6 - 1
    # (see ``03_standardize.py``)
    # so we sample + clip in normalized space (``limits_exp{1,2}``) and then
    # transform back to the original integer 1..7 Likert scale via
    #   raw = round(3 * norm + 4), clipped to [1, 7].

    @staticmethod
    def _norm_to_raw_int(norm):
        """Inverse of the 1..7 to [-1, 1] normalization, rounded to integer.

        ``Exp-tool-i`` is an integer Likert response on the 1..7 scale, so we
        round the de-standardized value and then clip to ``[1, 7]`` for safety.
        """
        return int(np.clip(np.rint(3.0 * float(norm) + 4.0), 1, 7))

    def gen_tool_U1_mean(self, Ew):
        """Mean of ``Exp-tool-1`` on the *raw* 1..7 Likert scale (integer)."""
        if self.cfg.theta_ml_U1.size < 3:
            raise RuntimeError("theta_ml_U1 missing")
        c0, c1, _sig = map(float, self.cfg.theta_ml_U1[:3])
        return self._norm_to_raw_int(c0 + c1 * Ew)

    def gen_tool_U1(self, Ew, week_idx):
        """Sample ``Exp-tool-1`` on the *raw* 1..7 Likert scale (integer)."""
        if self.cfg.theta_ml_U1.size < 3:
            raise RuntimeError("theta_ml_U1 missing")
        c0, c1, _sig = map(float, self.cfg.theta_ml_U1[:3])
        sig = float(self.cfg.theta_ml_U1[2])
        if self.cfg.resid_ml_U1.size > 0 and np.any(np.isfinite(self.cfg.resid_ml_U1)):
            noise = self._sample_noise(self.cfg.resid_ml_U1, week_idx, name="U1")
        else:
            noise = float(rd.normal(0.0, sig))
        norm = float(np.clip(c0 + c1 * Ew + noise, *self.cfg.limits_exp1))
        return self._norm_to_raw_int(norm)

    def gen_tool_U2_mean(self, Ew):
        """Mean of ``Exp-tool-2`` on the *raw* 1..7 Likert scale (integer)."""
        if self.cfg.theta_ml_U2.size < 3:
            raise RuntimeError("theta_ml_U2 missing")
        d0, d1, _sig = map(float, self.cfg.theta_ml_U2[:3])
        return self._norm_to_raw_int(d0 + d1 * Ew)

    def gen_tool_U2(self, Ew, week_idx):
        """Sample ``Exp-tool-2`` on the *raw* 1..7 Likert scale (integer)."""
        if self.cfg.theta_ml_U2.size < 3:
            raise RuntimeError("theta_ml_U2 missing")
        d0, d1, _sig = map(float, self.cfg.theta_ml_U2[:3])
        sig = float(self.cfg.theta_ml_U2[2])
        if self.cfg.resid_ml_U2.size > 0 and np.any(np.isfinite(self.cfg.resid_ml_U2)):
            noise = self._sample_noise(self.cfg.resid_ml_U2, week_idx, name="U2")
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


# Current ``df_fit_11week.csv`` names first; older camelCase aliases kept for
# backward compatibility. ``_float_csv_cell`` uses the first finite match.
_STATE_FROM_DF_FIT_ROW = (
    ("stepCountNext4HourLag1", ("FourSC_lag1", "fourSC_lag1")),
    ("pageViewNext4HourLag1", ("hourly_pageview_count_lag1",)),
    ("prior2HourStepCountEma7d", ("EMA_Prior2HourStepCount_norm", "EMA_Prior2HourStepCount")),
    ("prior2HourStepCountLag1", ("prior2hour_step_count_lag1", "prior2HourStepCountLag1")),
    ("prior2HourStepCount", (
        "prior2hour_step_norm",
        "prior2hour_step",
        "prior2HourStepCount_norm",
        "prior2HourStepCount",
    )),
    ("yesterdayStepCount", ("YesterdayStepCount_norm", "YesterdayStepCount")),
    ("pageViewLast7DaysEma", ("Past7DaysPageviewEMA_norm", "Past7DaysPageviewEMA")),
    ("morningFitbitWearLast7DaysAlt", ("past7days_daywearing", "morningFitbitWearLast7Days")),
    ("morningFitbitWearLast7Days", ("past7days_morning_wearing", "morningFitbitWearLast7Days")),
    ("dailySurveyComplete", ("daily_present", "dailySurveyComplete")),
    ("activeDaysLast7Days", ("active_status_fraction_7days", "activeDaysLast7Days")),
    ("activitySuggestionsSentLast7Days", (
        "recent_burden_norm",
        "recentBurdenEma_norm",
        "recentBurdenEma",
        "activitySuggestionsSentLast7Days",
    )),
    ("activitySuggestionInteractLast7Days", ("Interacted_7d_walk", "activitySuggestionInteractLast7Days")),
    ("dailyAnticipatedAffectYesterday", (
        "anticipated_affect_yesterday_norm",
        "anticipated_affect_yesterday",
        "dailyAnticipatedAffectYesterday_norm",
        "dailyAnticipatedAffectYesterday",
    )),
    ("dailyAnticipatedAffect", (
        "anticipated_affect_norm",
        "anticipated_affect",
        "dailyAnticipatedAffect_norm",
        "dailyAnticipatedAffect",
    )),
    ("caeAverageLastWeek", (
        "CAE_avg_lastweek_norm",
        "CAE_avg_lastweek",
        "caeAverageLastWeek_norm",
        "caeAverageLastWeek",
    )),
    ("morningFitbitWearYesterday", ("morning_wearing",)),
    ("dailySurveyCompleteYesterday", (
        "daily_present_yesterday",
        "yesterday_present",
        "dailySurveyComplete_yesterday",
    )),
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


def _csv_id_equal(raw, want):
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return False
    try:
        return int(float(str(raw).strip())) == int(want)
    except (TypeError, ValueError):
        return str(raw).strip() == str(want)


def _decision_time_slot(row):
    for key in ("DecisionTime", "decisionTime", "decision_time"):
        if key not in row:
            continue
        raw = row[key]
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            continue
        try:
            return int(float(str(raw).strip()))
        except (TypeError, ValueError):
            return None
    return None


def _load_csv_rows_for_participant(path, participant_id):
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        fieldnames = list(rdr.fieldnames or [])
        rows = list(rdr)
    if not rows:
        return []
    id_key = _participant_id_column(fieldnames)
    if participant_id is None:
        if not id_key:
            return rows[:1]
        return [r for r in rows if _csv_id_equal(r.get(id_key), rows[0].get(id_key))]
    try:
        want = int(participant_id)
    except (TypeError, ValueError):
        want = str(participant_id).strip()
    if not id_key:
        return []
    return [r for r in rows if _csv_id_equal(r.get(id_key), want)]


_SLOT_EMA_CSV_KEYS = (
    ("prior2HourStepCountEma7d", ("EMA_Prior2HourStepCount_norm", "EMA_Prior2HourStepCount")),
)


def slot_ema_initials_from_df_fit(df_fit_11week_csv=None, participant_id=None, n_slots=2):
    """First finite per-slot EMA from ``df_fit_11week.csv``.

    ``EMA_Prior2HourStepCount`` is fit within each
    ``DecisionTime`` stream. Missing slots fall back to the participant's
    first finite value, then 0.
    """
    n_slots = int(n_slots)
    path = _resolve_df_fit_11week_path(df_fit_11week_csv)
    per_slot = {name: {} for name, _ in _SLOT_EMA_CSV_KEYS}
    participant_first = {name: None for name, _ in _SLOT_EMA_CSV_KEYS}
    for row in _load_csv_rows_for_participant(path, participant_id):
        slot = _decision_time_slot(row)
        for name, csv_keys in _SLOT_EMA_CSV_KEYS:
            v = _float_csv_cell(row, *csv_keys, default=float("nan"))
            if not math.isfinite(v):
                continue
            if participant_first[name] is None:
                participant_first[name] = v
            if slot is None or slot < 0 or slot >= n_slots:
                continue
            if slot not in per_slot[name]:
                per_slot[name][slot] = v
        if all(len(d) >= n_slots for d in per_slot.values()):
            break
    out = {}
    for name, _ in _SLOT_EMA_CSV_KEYS:
        fallback = participant_first[name]
        if fallback is None or not math.isfinite(fallback):
            fallback = 0.0
        out[name] = {
            s: float(per_slot[name][s]) if s in per_slot[name] else float(fallback)
            for s in range(n_slots)
        }
    return out


def make_initial_state(df_fit_11week_csv=None, participant_id=None):
    """Study-entry state from the first ``df_fit_11week`` row for this user.

    EMA pads, lags, CAE last week, and Exp-tool-1/2 all come from that first
    Monday row (vanilla-panel start), not the MRT-end row. Tools are NaN when
    week 1's Sunday survey was unanswered — do not borrow a later week.
    """
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
    if row is not None:
        s["expTool1"] = _float_csv_cell(row, "Exp-tool-1", default=float("nan"))
        s["expTool2"] = _float_csv_cell(row, "Exp-tool-2", default=float("nan"))
    # Baseline CAE (study week 1). Observed -> the panel value. Missing (15 of
    # 42 users in the 2026-10-14 extract, plus the no-CAE users) -> the value
    # imputed from the fitted CAE model by 07_fit_testbed.py and stored
    # as ``cae_baseline_norm`` in params_env_{uid}.json next to the panel.
    # Before 2026-10-17 a missing baseline silently became 0.0.
    _obs_cae = _float_csv_cell(
        row if row is not None else {},
        "CAE_avg_lastweek_norm", "CAE_avg_lastweek", default=float("nan"),
    )
    s["caeBaselineSource"] = "observed"
    if not math.isfinite(_obs_cae):
        imputed = _imputed_cae_baseline(df_path, participant_id)
        if imputed is None:
            warnings.warn(
                f"Participant {participant_id}: no week-1 CAE in {df_path} and no "
                "'cae_baseline_norm' in its params_env json (rerun "
                "07_fit_testbed.py); using 0.0 (population mean) as before.",
                RuntimeWarning, stacklevel=2,
            )
            s["caeAverageLastWeek"], s["caeBaselineSource"] = 0.0, "fallback_zero"
        else:
            s["caeAverageLastWeek"], s["caeBaselineSource"] = imputed
    return s


def _imputed_cae_baseline(df_path, participant_id):
    """``(cae_baseline_norm, source)`` from params_env_{uid}.json beside ``df_path``."""
    if df_path is None or participant_id is None:
        return None
    p = Path(df_path).expanduser().resolve().parent / f"params_env_{int(participant_id)}.json"
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    v = d.get("cae_baseline_norm")
    if v is None or not math.isfinite(float(v)):
        return None
    return float(v), str(d.get("cae_baseline_source", "imputed"))
