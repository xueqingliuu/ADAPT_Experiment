"""Fixed-strength misspecifications A–D (no re-tuning).

When misspec is ``none`` these helpers are not called, so the generator is
unchanged. Calibrated strengths for A and B live in
``<params_dir>/misspec_calibration.json`` and are written by
``calibrate_misspec.py`` after a refit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from vani_env import THETA_ANTIC_NAMES, THETA_CAE_NAMES, THETA_FOURSC_NAMES


CONFOUNDER_SALT = 191337
CONFOUNDER_RHO = 0.5
CONFOUNDER_LAM_Y = 0.5
CAUSAL_ALPHA_SCALE = 0.5
PATHWAY_ACTION_SCALE = 0.5
DECAY_HALF_LIFE_WEEKS = 12.0
BURDEN_DROPOUT_OR = 1.5
CALIBRATION_NAME = "misspec_calibration.json"

FOURSC_ACTION_IDX = np.array(
    [
        i
        for i, name in enumerate(THETA_FOURSC_NAMES)
        if name == "WalkingSuggestion" or name.startswith("WalkingSuggestion_by_")
    ],
    dtype=int,
)
ANTIC_ACTION_IDX = np.array(
    [
        i
        for i, name in enumerate(THETA_ANTIC_NAMES)
        if name.startswith("A0_") or name.startswith("A1_")
    ],
    dtype=int,
)
CAE_SC_IDX = THETA_CAE_NAMES.index("fourSC_ewma")
CAE_AA_IDX = THETA_CAE_NAMES.index("anticipated_affect_ewma")


DECAY_C = 1.0

MISSPEC_C_BLURB = (
    "Action effects on SC/AA decay from week 1 as m(w)=2^{-(w-1)/12} "
    "(c=1: m(1)=1, m(12)≈0.53, m(36)≈0.132)."
)


def decay_m(week, *, c: float = DECAY_C, half_life: float = DECAY_HALF_LIFE_WEEKS) -> float:
    """``m(w) = c * 2^(-(w-1)/12)``. ``week`` is the 1-based protocol week.

    Default ``c = 1`` (no mean-1 normalization): m(1)=1, m(12)≈0.53,
    m(36)≈0.132.
    """
    w = int(week)
    return float(c) * (2.0 ** (-(w - 1) / float(half_life)))


def action_scale(misspec: str, week: int) -> float:
    """Multiplier on SC/AA action mains and action interactions."""
    if misspec == "B":
        return PATHWAY_ACTION_SCALE
    if misspec == "C":
        return decay_m(week)
    return 1.0


def residual_scale(lam: float) -> float:
    lam = float(lam)
    return math.sqrt(max(1.0 - lam * lam, 0.0))


def confounder_seed_sequence(seed, uid) -> np.random.SeedSequence:
    """Own stream, a child of ``(seed, uid)``. Independent of the activity salt."""
    parent = np.random.SeedSequence([0 if seed is None else int(seed), int(uid)])
    return parent.spawn(1)[0]


def draw_confounder(seed, uid, nweek: int, rho: float = CONFOUNDER_RHO) -> np.ndarray:
    """Weekly ``C_w``: ``C_1 ~ N(0,1)``, then AR(1) with stationary variance 1."""
    rng = np.random.default_rng(confounder_seed_sequence(seed, uid))
    n = int(nweek)
    c = np.empty(n, dtype=float)
    innov = math.sqrt(max(1.0 - float(rho) ** 2, 0.0))
    c[0] = float(rng.normal())
    for w in range(1, n):
        c[w] = float(rho) * c[w - 1] + innov * float(rng.normal())
    return c


def residual_sigma(resid) -> float:
    a = np.asarray(resid, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 1.0
    sd = float(np.std(a, ddof=1))
    return sd if sd > 0.0 else 1.0


def calibration_path(params_dir) -> Path:
    return Path(params_dir).expanduser().resolve() / CALIBRATION_NAME


def load_calibration(params_dir) -> dict:
    path = calibration_path(params_dir)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_calibration(params_dir, payload: dict) -> Path:
    path = calibration_path(params_dir)
    current = load_calibration(params_dir)
    current.update(payload)
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return path


def strengths_A(params_dir, overrides: dict | None = None) -> dict:
    """``lam_Y``, ``lam_SC``, ``lam_AA``. Overrides win; else the stored solution."""
    stored = load_calibration(params_dir).get("A") or {}
    out = {
        "lam_Y": CONFOUNDER_LAM_Y,
        "lam_SC": stored.get("lam_SC"),
        "lam_AA": stored.get("lam_AA"),
        "rho": stored.get("rho", CONFOUNDER_RHO),
    }
    if overrides:
        out.update({k: overrides[k] for k in out if k in overrides})
    return out


def strength_B(params_dir, override=None) -> float | None:
    if override is not None:
        return float(override)
    stored = load_calibration(params_dir).get("B") or {}
    if "beta_D" not in stored:
        return None
    return float(stored["beta_D"])


def require_A(params_dir, overrides: dict | None = None) -> dict:
    got = strengths_A(params_dir, overrides)
    if got["lam_SC"] is None or got["lam_AA"] is None:
        raise FileNotFoundError(
            f"misspec A needs lam_SC and lam_AA in {calibration_path(params_dir)} "
            "(run python calibrate_misspec.py --params-dir <dir>)"
        )
    return {k: float(got[k]) for k in ("lam_Y", "lam_SC", "lam_AA", "rho")}


def require_B(params_dir, override=None) -> float:
    beta = strength_B(params_dir, override)
    if beta is None:
        raise FileNotFoundError(
            f"misspec B needs beta_D in {calibration_path(params_dir)} "
            "(run python calibrate_misspec.py --params-dir <dir>)"
        )
    return float(beta)


def scale_action_theta(theta, indices: np.ndarray, scale: float) -> np.ndarray:
    """Copy ``theta`` and multiply the action block. Identity when ``scale`` is 1."""
    if abs(float(scale) - 1.0) < 1e-15:
        return theta
    out = np.asarray(theta, dtype=float).copy()
    out[indices] *= float(scale)
    return out


def scale_cae_alphas(theta, scale: float = CAUSAL_ALPHA_SCALE) -> np.ndarray:
    if abs(float(scale) - 1.0) < 1e-15:
        return theta
    out = np.asarray(theta, dtype=float).copy()
    out[CAE_SC_IDX] *= float(scale)
    out[CAE_AA_IDX] *= float(scale)
    return out


def burden_dropout_shift(burden_z: float) -> float:
    """Model-1 logit addend: ``log(1.5) * (B - B_bar) / sd_B``.

    The state already stores recent burden on the MRT z-score, so this is
    ``log(1.5) * B_z``.
    """
    return math.log(BURDEN_DROPOUT_OR) * float(burden_z)
