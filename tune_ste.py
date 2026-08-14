"""Rescale ``env_para_vanilla`` so mean STE is near a chosen target (0.2 / 0.5 / 0.8).

STE is the same quantity as ``ste_vanilla.py``: for each participant, never-suggest
vs a treatment policy, then ``(mean_G_treat - mean_G_control) / sd(G_control)``,
averaged over users. This script does **not** train DQNs. It multiplies a small
set of coefficients by a scalar ``kappa`` and simulates cheap policies until the
proxy STE hits the target.

Commands
    diagnose    E_w and CAE loop gains (no simulation)
    eval        measure proxy STE of one parameter folder
    scan        proxy STE vs a grid of ``kappa``
    calibrate   find ``kappa`` for each target and write ``env_para_ste0.2/`` etc.

How STE is measured here
    Treatment arms are constant suggestion rates (``--policy-grid``, default
    0.5 and 1.0). Optionally add a already-trained DQN with ``--dqn-exp``;
    that network is only evaluated, not retrained. The reported STE is a
    lower bound on a DQN trained inside the tuned environment.

Which coefficients are scaled (``--knob``, default ``action``)
    action    walking-suggestion effects on both benefit (step-count /
              anticipated affect) and burden (engagement) mediators
    benefit   benefit path only
    burden    burden path only
    my_to_y / me_to_e / e_to_my   structural paths; these also change the
              control arm, so they are not the recommended dial

Stability
    ``calibrate`` writes a folder only if every participant has |loop gain| < 1
    for both E_w and CAE, under never-suggest and always-suggest. Search will
    not go past the largest such ``kappa``. Missed targets are skipped.

Cluster
    ``sbatch run_tune_ste.sh`` runs calibrate, then submits ``run_ste.sh`` in
    each written folder so a fresh DQN measures the true STE.

Use a tuned folder instead of vanilla::

    ADAPR_PARAMS_DIR=env_para_ste0.5 python ste_vanilla.py train 0 --exp ste0.5
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ste_vanilla import _model_metadata_path, rollout_total_cae
from vani_env import (
    PARAMS_DIR,
    THETA_ANTIC_NAMES,
    THETA_CAE_NAMES,
    THETA_FOURSC_NAMES,
    EnvConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parent

# Copied verbatim into every tuned parameter directory so it can be handed to
# ``ste_vanilla.py`` / ``experiment.py`` as a drop-in replacement.
SUPPORTING_FILES = (
    "std_params.json",
    "user_ids.txt",
    "df_fit_11week.csv",
    "Ew_pooled_linear_coefs.json",
    "population_residuals.json",
    "rl_priors.json",
)

# ---------------------------------------------------------------------------
# Pathway definitions
# ---------------------------------------------------------------------------
# Each pathway maps a theta block to the coefficient names it owns. The sets are
# disjoint: every WalkingSuggestion / A0_morning / A1_afternoon interaction is
# owned by the action pathway that switches it on, so scaling two pathways at
# once never double-scales a coefficient. In particular
# ``WalkingSuggestion_by_perceived_utility_lastweek`` belongs to ``A_to_MY``,
# not ``E_to_MY``.
#
# The ``query_imputed_*`` block of PV/FW/PJ is deliberately excluded: STE
# rollouts hold ``I_w = 1`` in both arms, so it contributes identically to
# treatment and control and cancels out of ``Delta_i``.

_ACTION_PREFIXES_ANTIC = ("A0_morning", "A1_afternoon")

PATHWAYS: dict[str, dict[str, tuple[str, ...]]] = {
    "A_to_MY": {
        "theta_fourSC": tuple(
            n for n in THETA_FOURSC_NAMES if n.startswith("WalkingSuggestion")
        ),
        "theta_antic": tuple(
            n for n in THETA_ANTIC_NAMES if n.startswith(_ACTION_PREFIXES_ANTIC)
        ),
    },
    "MY_to_Y": {
        "theta_CAE": ("fourSC_ewma", "anticipated_affect_ewma"),
    },
    "A_to_ME": {
        "theta_penalized_PV": ("alpha3_action", "alpha4_action_by_Ew"),
        "theta_penalized_FW": (
            "beta3_A0_morning",
            "beta4_A0_morning_by_Ew",
            "beta5_A1_afternoon",
            "beta6_A1_afternoon_by_Ew",
        ),
        "theta_penalized_PJ": (
            "theta3_A0_morning",
            "theta4_A0_morning_by_Ew",
            "theta5_A1_afternoon",
            "theta6_A1_afternoon_by_Ew",
        ),
    },
    "ME_to_E": {
        "theta_penalized_Ew": ("a2_PV_lag_week", "a3_FW_lag_week", "a4_PJ_lag_week"),
    },
    "E_to_MY": {
        "theta_fourSC": ("perceived_utility_lastweek",),
        "theta_antic": ("perceived_utility_lastweek",),
    },
}

# Pathways that are gated by an action indicator, hence invisible to pi_0.
ACTION_GATED = frozenset({"A_to_MY", "A_to_ME"})

# Fallback coefficient names for blocks that carry no ``*_names`` key in JSON.
_NAME_CONSTANTS = {
    "theta_fourSC": THETA_FOURSC_NAMES,
    "theta_antic": THETA_ANTIC_NAMES,
    "theta_CAE": THETA_CAE_NAMES,
}

# A knob maps a scalar kappa to a per-pathway multiplier dict. Knobs are named so
# that STE is monotone in kappa; ``increasing`` records the direction.
Knob = Callable[[float], dict[str, float]]


@dataclass(frozen=True)
class KnobSpec:
    name: str
    build: Knob
    zero_is_null: bool  # STE(kappa = 0) == 0 holds exactly, for free
    sigma_invariant: bool  # control arm untouched, so sigma_i can be cached
    doc: str


KNOBS: dict[str, KnobSpec] = {
    "action": KnobSpec(
        "action",
        lambda k: {"A_to_MY": k, "A_to_ME": k},
        zero_is_null=True,
        sigma_invariant=True,
        doc=(
            "Scale the whole action block, benefit and burden together. Keeps the "
            "fitted benefit/burden balance, leaves sigma_i and control-arm "
            "stability exactly unchanged, and is exactly null at kappa=0. "
            "Recommended, and raises STE as long as the fitted net action effect "
            "is positive."
        ),
    ),
    "benefit": KnobSpec(
        "benefit",
        lambda k: {"A_to_MY": k},
        zero_is_null=False,
        sigma_invariant=True,
        doc=(
            "Scale action -> 4h step counts / anticipated affect only, holding the "
            "engagement burden fixed. Use when the fitted net action effect is "
            "negative or near zero and 'action' therefore cannot reach the target."
        ),
    ),
    "burden": KnobSpec(
        "burden",
        lambda k: {"A_to_ME": k},
        zero_is_null=False,
        sigma_invariant=True,
        doc="Scale action -> engagement mediators only; larger kappa = more burden = lower STE.",
    ),
    "my_to_y": KnobSpec(
        "my_to_y",
        lambda k: {"MY_to_Y": k},
        zero_is_null=False,
        sigma_invariant=False,
        doc="Structural: mediator -> CAE loading. Moves sigma_i as well as Delta_i.",
    ),
    "me_to_e": KnobSpec(
        "me_to_e",
        lambda k: {"ME_to_E": k},
        zero_is_null=False,
        sigma_invariant=False,
        doc="Structural: mediator -> E_w feedback. Moves sigma_i and the E_w loop gain.",
    ),
    "e_to_my": KnobSpec(
        "e_to_my",
        lambda k: {"E_to_MY": k},
        zero_is_null=False,
        sigma_invariant=False,
        doc="Structural: E_w -> step-count mediators. Moves sigma_i as well as Delta_i.",
    ),
}


# ---------------------------------------------------------------------------
# Parameter rescaling
# ---------------------------------------------------------------------------
def _names_for(key: str, params: dict) -> list[str]:
    names = params.get(f"{key}_names")
    if names:
        return [str(n) for n in names]
    if key in _NAME_CONSTANTS:
        return list(_NAME_CONSTANTS[key])
    raise KeyError(f"No coefficient names available for {key!r}")


def scale_params(
    params: dict, multipliers: dict[str, float]
) -> tuple[dict, list[dict]]:
    """Return a copy of ``params`` with the requested pathways rescaled.

    ``multipliers`` maps pathway name -> scale factor; missing pathways are left
    at 1.0. The audit trail lists every coefficient actually touched.
    """
    unknown = set(multipliers) - set(PATHWAYS)
    if unknown:
        raise KeyError(f"Unknown pathway(s): {sorted(unknown)}")

    out = dict(params)
    audit: list[dict] = []
    for pathway, factor in multipliers.items():
        if float(factor) == 1.0:
            continue
        for key, coef_names in PATHWAYS[pathway].items():
            if key not in out:
                raise KeyError(
                    f"{key!r} missing from participant params -- run "
                    f"5_fit_vanilla_testbed.py before tuning."
                )
            values = np.asarray(out[key], dtype=float).ravel().copy()
            names = _names_for(key, out)
            for coef in coef_names:
                try:
                    idx = names.index(coef)
                except ValueError as exc:
                    raise KeyError(f"{coef!r} not in {key}_names") from exc
                if idx >= values.size:
                    raise IndexError(
                        f"{key} has {values.size} values but {coef!r} is at index {idx}"
                    )
                old = float(values[idx])
                values[idx] = old * float(factor)
                audit.append(
                    {
                        "pathway": pathway,
                        "block": key,
                        "coef": coef,
                        "factor": float(factor),
                        "old": old,
                        "new": float(values[idx]),
                    }
                )
            out[key] = values.tolist()
    return out, audit


def write_scaled_params(
    src_dir: Path,
    dst_dir: Path,
    user_ids: Sequence[int],
    multipliers: dict[str, float],
) -> list[dict]:
    """Materialise a tuned copy of ``src_dir`` at ``dst_dir``."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in SUPPORTING_FILES:
        src = src_dir / fname
        if src.is_file():
            shutil.copy2(src, dst_dir / fname)

    audit: list[dict] = []
    for uid in user_ids:
        src = src_dir / f"params_env_{uid}.json"
        with open(src, encoding="utf-8") as f:
            params = json.load(f)
        scaled, rows = scale_params(params, multipliers)
        for r in rows:
            r["userid"] = int(uid)
        audit.extend(rows)
        with open(dst_dir / f"params_env_{uid}.json", "w", encoding="utf-8") as f:
            json.dump(scaled, f, allow_nan=False)
    return audit


# ---------------------------------------------------------------------------
# Monte-Carlo STE proxy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProxySpec:
    """Monte-Carlo settings for the STE proxy.

    ``episodes`` controls precision. With common random numbers the paired
    difference has far less variance than either arm, and averaging over ~30
    participants shrinks it further, so 100 episodes already puts the standard
    error of the reported mean STE around 0.01 -- below the default tolerance.
    A short ``policy_grid`` is deliberate: ``Delta_i`` takes a max over the
    treatment arms, and a max over noisy estimates is biased upward, so extra
    arms buy a tighter lower bound at the cost of more bias and more compute.

    ``dqn_model_dir`` points at a directory of ``ste_vanilla.py`` checkpoints
    (``user<uid>_model.d3``); when set, each participant's trained DQN is added
    as one more treatment arm.
    """

    episodes: int = 100
    seed0: int = 20260814
    noise: str = "ar1"
    policy_grid: tuple[float, ...] = (0.5, 1.0)
    i_w_fixed: int = 1
    dqn_model_dir: str | None = None


_DQN_CACHE: dict[str, object] = {}


def load_dqn(model_dir: Path, uid: int, *, nweek: int, noise: str):
    """Load a ``ste_vanilla.py`` DQN checkpoint, validating what must match.

    ``userid``/``nweek``/``noise`` have to agree or the policy is being applied
    to a different problem than it was trained on. The fitted parameters are
    deliberately *not* checked: evaluating a baseline-trained policy in a
    rescaled environment is the entire point.
    """
    cache_key = f"{model_dir}|{uid}"
    if cache_key in _DQN_CACHE:
        return _DQN_CACHE[cache_key]

    path = Path(model_dir) / f"user{uid}_model.d3"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; train it first with "
            f"`python ste_vanilla.py train <jobid> --exp <exp>`."
        )
    meta_path = _model_metadata_path(path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} missing; retrain the STE model.")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    for name, expected in (("userid", uid), ("nweek", nweek), ("noise", noise)):
        if meta.get(name) != expected:
            raise ValueError(
                f"DQN checkpoint {path.name} mismatch on {name}: "
                f"trained={meta.get(name)!r}, evaluating={expected!r}"
            )

    try:
        import d3rlpy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "d3rlpy is required for --dqn-exp / --dqn-model-dir; install it or "
            "drop back to the Bernoulli-only arms."
        ) from exc

    model = d3rlpy.load_learnable(str(path))
    _DQN_CACHE[cache_key] = model
    return model


def _rollouts(
    uid: int,
    params_dir: str,
    spec: ProxySpec,
    policy: str,
    nweek: int,
    *,
    walk_prob: float = 0.0,
    dqn=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Totals and per-week CAE across ``spec.episodes`` paired-seed episodes."""
    totals = np.empty(spec.episodes, dtype=float)
    weekly = np.empty((spec.episodes, nweek), dtype=float)
    for n in range(spec.episodes):
        total, wk = rollout_total_cae(
            uid,
            nweek=nweek,
            seed=spec.seed0 + n,
            policy=policy,
            dqn=dqn,
            noise=spec.noise,
            i_w_fixed=spec.i_w_fixed,
            walk_prob=walk_prob,
            params_dir=Path(params_dir),
            return_weekly=True,
        )
        totals[n] = total
        weekly[n] = wk
    return totals, weekly


def _eval_user(task: tuple) -> dict:
    """Worker: one participant's control and treatment arms under common seeds."""
    uid, params_dir, spec_kwargs, zero_totals = task
    spec = ProxySpec(**spec_kwargs)
    cfg = EnvConfig(uid, params_dir=Path(params_dir))
    nweek = int(cfg.nweek)
    lo_cae, hi_cae = (float(x) for x in cfg.limits_CAE)

    if zero_totals is None:
        zero_totals, zero_weekly = _rollouts(uid, params_dir, spec, "zero", nweek)
        zero_sat = _clip_rate(zero_weekly, lo_cae, hi_cae)
    else:
        zero_totals = np.asarray(zero_totals, dtype=float)
        zero_sat = float("nan")

    arms: dict[str, dict] = {}

    def add_arm(label: str, policy: str, **kw) -> None:
        totals, weekly = _rollouts(uid, params_dir, spec, policy, nweek, **kw)
        arms[label] = {
            "mean_total": float(np.mean(totals)),
            "delta": float(np.mean(totals) - np.mean(zero_totals)),
            "clip_rate": _clip_rate(weekly, lo_cae, hi_cae),
        }

    for p in spec.policy_grid:
        add_arm(f"p={p:g}", "bernoulli", walk_prob=float(p))
    if spec.dqn_model_dir:
        add_arm(
            "dqn",
            "dqn_greedy",
            dqn=load_dqn(
                Path(spec.dqn_model_dir), uid, nweek=nweek, noise=spec.noise
            ),
        )

    best = max(arms, key=lambda k: arms[k]["delta"])
    sigma = float(np.std(zero_totals, ddof=1))
    # The control arm is "never suggest", which belongs to every policy class
    # considered here, so the optimum can never do worse than it; a negative
    # best-arm delta means never-suggest wins and Delta_i is 0.
    delta = max(0.0, arms[best]["delta"])
    return {
        "userid": int(uid),
        "nweek": nweek,
        "sigma": sigma,
        "delta": delta,
        "delta_raw": float(arms[best]["delta"]),
        "ste": delta / sigma if sigma > 0 else float("nan"),
        "best_policy": best,
        "mean_total_zero": float(np.mean(zero_totals)),
        "clip_rate_zero": zero_sat,
        "clip_rate_best": arms[best]["clip_rate"],
        "arms": arms,
        "zero_totals": zero_totals.tolist(),
    }


def _clip_rate(weekly: np.ndarray, lo: float, hi: float, tol: float = 1e-6) -> float:
    """Fraction of simulated weeks sitting on a CAE clip boundary."""
    w = np.asarray(weekly, dtype=float)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return float("nan")
    return float(np.mean((w <= lo + tol) | (w >= hi - tol)))


def evaluate(
    params_dir: Path,
    user_ids: Sequence[int],
    spec: ProxySpec,
    *,
    zero_cache: dict[int, list[float]] | None = None,
    n_jobs: int | None = None,
    proxy_to_true: float = 1.0,
) -> dict:
    """Monte-Carlo the proxy STE for every participant in ``params_dir``."""
    spec_kwargs = asdict(spec)
    spec_kwargs["policy_grid"] = tuple(spec.policy_grid)
    tasks = [
        (
            int(uid),
            str(params_dir),
            spec_kwargs,
            None if zero_cache is None else zero_cache.get(int(uid)),
        )
        for uid in user_ids
    ]

    if n_jobs is not None and n_jobs <= 1:
        rows = [_eval_user(t) for t in tasks]
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                rows = list(pool.map(_eval_user, tasks))
        except (OSError, PermissionError, NotImplementedError, ImportError) as exc:
            print(f"  [warn] process pool unavailable ({exc}); running serially")
            rows = [_eval_user(t) for t in tasks]

    for r in rows:
        r["ste"] = float(r["ste"]) * float(proxy_to_true)
    ste = np.array([r["ste"] for r in rows], dtype=float)
    finite = ste[np.isfinite(ste)]
    return {
        "params_dir": str(params_dir),
        "proxy_to_true": float(proxy_to_true),
        "mean_ste": float(np.mean(finite)) if finite.size else float("nan"),
        "median_ste": float(np.median(finite)) if finite.size else float("nan"),
        "min_ste": float(np.min(finite)) if finite.size else float("nan"),
        "max_ste": float(np.max(finite)) if finite.size else float("nan"),
        "n_users": len(rows),
        "users": rows,
    }


def zero_cache_from(result: dict) -> dict[int, list[float]]:
    return {int(r["userid"]): r["zero_totals"] for r in result["users"]}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def loop_gain_Ew(params: dict) -> dict[str, float]:
    """Compound ``E_w`` loop gain under never-suggest and always-suggest.

    ``E_w`` is linear in the weekly mediator summaries, and those summaries are
    fixed-denominator *averages* (``nansum(pv)/14``, ``nansum(FW|PJ)/7``) in both
    the estimator and ``vani_env._week_means_from_arrays``. A unit shift in
    ``E_{w-1}`` shifts every slot of the week, so the average shifts by the
    per-slot slope -- no 14x / 7x factor. (The training-time barrier in
    ``4_perceived_utility.py`` multiplies by 14 and 7 and is therefore a strictly
    more conservative bound than this one.) Logistic mediators use the
    worst-case slope ``pi(1-pi) = 0.25``.
    """
    def pick(key: str, coef: str) -> float:
        vals = np.asarray(params[key], dtype=float).ravel()
        return float(vals[_names_for(key, params).index(coef)])

    a1 = pick("theta_penalized_Ew", "a1")
    a2 = pick("theta_penalized_Ew", "a2_PV_lag_week")
    a3 = pick("theta_penalized_Ew", "a3_FW_lag_week")
    a4 = pick("theta_penalized_Ew", "a4_PJ_lag_week")

    alpha1 = pick("theta_penalized_PV", "alpha1_Ew")
    alpha4 = pick("theta_penalized_PV", "alpha4_action_by_Ew")
    beta1 = pick("theta_penalized_FW", "beta1_Ew")
    beta4 = pick("theta_penalized_FW", "beta4_A0_morning_by_Ew")
    beta6 = pick("theta_penalized_FW", "beta6_A1_afternoon_by_Ew")
    theta1 = pick("theta_penalized_PJ", "theta1_Ew")
    theta4 = pick("theta_penalized_PJ", "theta4_A0_morning_by_Ew")
    theta6 = pick("theta_penalized_PJ", "theta6_A1_afternoon_by_Ew")

    slope = 0.25
    g_zero = a1 + a2 * alpha1 + a3 * slope * beta1 + a4 * slope * theta1
    g_always = (
        a1
        + a2 * (alpha1 + alpha4)
        + a3 * slope * (beta1 + beta4 + beta6)
        + a4 * slope * (theta1 + theta4 + theta6)
    )
    return {"g_zero": float(g_zero), "g_always": float(g_always)}


def loop_gain_CAE(params: dict) -> dict[str, float]:
    """First-order weekly CAE loop gain under zero and always suggestion.

    Last week's CAE enters this week's CAE directly and indirectly through the
    fourSC and anticipated-affect EWMAs.  Because an EWMA's weights sum to one,
    a week-constant shift has the same slope as each mediator observation.
    The always-suggest gain includes the fitted action-by-lagged-CAE terms.
    """

    def pick(key: str, coef: str) -> float:
        vals = np.asarray(params[key], dtype=float).ravel()
        return float(vals[_names_for(key, params).index(coef)])

    direct = pick("theta_CAE", "CAE_avg_lastweek")
    load_four = pick("theta_CAE", "fourSC_ewma")
    load_antic = pick("theta_CAE", "anticipated_affect_ewma")

    four_zero = pick("theta_fourSC", "CAE_avg_lastweek")
    four_always = four_zero + pick(
        "theta_fourSC", "WalkingSuggestion_by_CAE_avg_lastweek"
    )

    antic_zero = pick("theta_antic", "CAE_avg_lastweek")
    antic_always = (
        antic_zero
        + pick("theta_antic", "A0_morning_by_CAE_avg_lastweek")
        + pick("theta_antic", "A1_afternoon_by_CAE_avg_lastweek")
    )

    g_zero = direct + load_four * four_zero + load_antic * antic_zero
    g_always = direct + load_four * four_always + load_antic * antic_always
    return {"cae_g_zero": float(g_zero), "cae_g_always": float(g_always)}


def diagnose(params_dir: Path, user_ids: Sequence[int]) -> list[dict]:
    rows = []
    for uid in user_ids:
        with open(params_dir / f"params_env_{uid}.json", encoding="utf-8") as f:
            params = json.load(f)
        rows.append(
            {
                "userid": int(uid),
                **loop_gain_Ew(params),
                **loop_gain_CAE(params),
            }
        )
    return rows


GAIN_LIMIT = 1.0  # |g| must be strictly below this


def max_abs_gain(row: dict) -> float:
    return max(
        abs(float(row["g_zero"])),
        abs(float(row["g_always"])),
        abs(float(row["cae_g_zero"])),
        abs(float(row["cae_g_always"])),
    )


def unstable_users(
    gains: Sequence[dict], *, limit: float = GAIN_LIMIT
) -> list[dict]:
    return [r for r in gains if max_abs_gain(r) >= limit]


def loop_gains_for(
    src_dir: Path,
    user_ids: Sequence[int],
    multipliers: dict[str, float],
) -> list[dict]:
    """E_w and CAE loop gains after applying ``multipliers``, without writing."""
    rows = []
    for uid in user_ids:
        with open(src_dir / f"params_env_{uid}.json", encoding="utf-8") as f:
            params = json.load(f)
        scaled, _ = scale_params(params, multipliers) if multipliers else (params, [])
        rows.append(
            {
                "userid": int(uid),
                **loop_gain_Ew(scaled),
                **loop_gain_CAE(scaled),
            }
        )
    return rows


def largest_stable_kappa(
    src_dir: Path,
    user_ids: Sequence[int],
    knob: KnobSpec,
    *,
    x_hi: float = 256.0,
    limit: float = GAIN_LIMIT,
) -> float:
    """Largest ``kappa`` in ``[0, x_hi]`` with every participant's ``|g| < limit``."""

    def ok(k: float) -> bool:
        return not unstable_users(
            loop_gains_for(src_dir, user_ids, knob.build(float(k))), limit=limit
        )

    if not ok(0.0):
        return 0.0
    if ok(x_hi):
        return float(x_hi)
    lo, hi = 0.0, float(x_hi)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


# ---------------------------------------------------------------------------
# Root finding
# ---------------------------------------------------------------------------
def solve_scalar(
    f: Callable[[float], float],
    target: float,
    *,
    x0: float = 1.0,
    f_at_zero: float | None = None,
    seed_points: dict[float, float] | None = None,
    tol: float = 0.02,
    max_iter: int = 10,
    x_lo: float = 0.0,
    x_hi: float = 256.0,
    log=print,
) -> tuple[float, float, list[tuple[float, float]]]:
    """Solve ``f(x) = target`` for an expensive, near-linear, monotone ``f``.

    The direction of monotonicity is inferred from the evaluations rather than
    declared up front: whether a knob raises or lowers STE depends on which
    pathway dominates in the fitted coefficients, which is an empirical
    question. Once two points straddle the target the search switches to the
    the Illinois variant of false position, which keeps a valid bracket (so
    Monte-Carlo wobble cannot make the search diverge) while converging
    superlinearly. The Illinois damping matters here: the STE response is
    noticeably concave in ``kappa`` over long horizons, and undamped false
    position on a concave function retains the same endpoint every iteration and
    creeps toward the root. Before a bracket exists the search takes secant
    steps, which walk toward the target from either side.

    ``f_at_zero`` supplies a free evaluation at ``x = 0``; for the ``action``
    knob every action-gated coefficient vanishes there, so both arms follow the
    identical data-generating process and ``f(0) = 0`` holds exactly.
    ``seed_points`` carries evaluations already paid for -- the response curve
    does not depend on the target, so solving for several targets in one run
    should share it. ``max_iter`` counts only genuinely new evaluations.
    """
    pts: dict[float, float] = {}
    hist: list[tuple[float, float]] = []

    def ev(x: float) -> float:
        x = float(np.clip(x, x_lo, x_hi))
        if x in pts:
            return pts[x]
        y = float(f(x))
        pts[x] = y
        hist.append((x, y))
        log(f"    kappa={x:9.4f}  ->  mean STE={y:8.4f}   (target {target:.3f})")
        return y

    def best() -> tuple[float, float]:
        return min(pts.items(), key=lambda kv: abs(kv[1] - target))

    def bracket() -> tuple[float, float] | None:
        """Adjacent pair of evaluations straddling the target, if any."""
        xs = sorted(pts)
        for a, b in zip(xs, xs[1:]):
            if (pts[a] - target) * (pts[b] - target) <= 0.0:
                return a, b
        return None

    if seed_points:
        pts.update({float(k): float(v) for k, v in seed_points.items()})
    if f_at_zero is not None:
        pts[0.0] = float(f_at_zero)
    ev(x0)

    # Phase 1: secant search until two evaluations straddle the target.
    while len(hist) < max_iter and bracket() is None and abs(best()[1] - target) > tol:
        xs = sorted(pts, key=lambda x: abs(pts[x] - target))[:2]
        if len(xs) < 2 or pts[xs[0]] == pts[xs[1]]:
            x_new = max(x0, best()[0], 1e-3) * 2.0
        else:
            x_a, x_b = xs
            slope = (pts[x_b] - pts[x_a]) / (x_b - x_a)
            x_new = x_a + (target - pts[x_a]) / slope
            if not np.isfinite(x_new) or x_new <= x_lo:
                x_new = max(x0, best()[0], 1e-3) * 2.0
            # Never extrapolate more than 4x past the explored range.
            x_new = min(x_new, 4.0 * max(pts))
        if float(np.clip(x_new, x_lo, x_hi)) in pts:
            log("    search stalled without bracketing the target")
            break
        ev(x_new)

    # Phase 2: Illinois false position inside the bracket.
    br = bracket()
    if br is not None:
        a, b = br
        fa, fb = pts[a] - target, pts[b] - target
        retained = 0
        while len(hist) < max_iter and abs(best()[1] - target) > tol:
            if fb == fa:
                break
            x_new = (a * fb - b * fa) / (fb - fa)
            span = b - a
            x_new = float(np.clip(x_new, a + 1e-3 * span, b - 1e-3 * span))
            if x_new in pts:
                x_new = 0.5 * (a + b)
                if x_new in pts:
                    break
            fx = ev(x_new) - target
            if fx * fb > 0.0:
                b, fb = x_new, fx
                if retained < 0:
                    fa *= 0.5  # Illinois: damp the endpoint we keep reusing
                retained = -1
            elif fx * fa > 0.0:
                a, fa = x_new, fx
                if retained > 0:
                    fb *= 0.5
                retained = 1
            else:
                break

    if abs(best()[1] - target) > tol:
        log(f"    warning: closest achievable was {best()[1]:.4f} at kappa={best()[0]:.4f}")
    return (*best(), hist)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------
def load_user_ids(params_dir: Path) -> list[int]:
    path = params_dir / "user_ids.txt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    return [int(v) for v in np.loadtxt(path, dtype=int).ravel()]


def require_fitted_blocks(params_dir: Path, user_ids: Sequence[int]) -> None:
    """Fail early (and legibly) if 5_fit_vanilla_testbed.py has not been re-run."""
    needed = ("theta_CAE", "theta_fourSC", "theta_antic", "theta_ws_interaction")
    uid = user_ids[0]
    with open(params_dir / f"params_env_{uid}.json", encoding="utf-8") as f:
        params = json.load(f)
    missing = [k for k in needed if params.get(k) is None]
    if missing:
        raise SystemExit(
            f"{params_dir}/params_env_{uid}.json is missing {missing}.\n"
            "These blocks come from 5_fit_vanilla_testbed.py; run scripts 5 and 6 "
            "after 4_perceived_utility.py before tuning."
        )


def require_dqn_checkpoints(spec: ProxySpec, user_ids: Sequence[int]) -> None:
    """Fail before any rollouts if the DQN arm cannot be served for every user."""
    if not spec.dqn_model_dir:
        return
    try:
        import d3rlpy  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "d3rlpy is required for --dqn-exp / --dqn-model-dir; install it or "
            "drop the flag to use the Bernoulli-only arms."
        ) from exc
    model_dir = Path(spec.dqn_model_dir)
    missing = [
        uid for uid in user_ids if not (model_dir / f"user{uid}_model.d3").exists()
    ]
    if missing:
        raise SystemExit(
            f"{model_dir} has no checkpoint for {len(missing)} of {len(user_ids)} "
            f"participants: {missing}\nTrain them with `python ste_vanilla.py train "
            "<jobid> --exp <exp>` (one job per user_ids.txt index)."
        )
    print(f"DQN arm: {len(user_ids)} checkpoints from {model_dir}")


def summarise(result: dict, label: str = "") -> None:
    users = sorted(result["users"], key=lambda r: r["ste"])
    head = f"  {label}" if label else ""
    print(f"{head}  mean STE={result['mean_ste']:.4f}  median={result['median_ste']:.4f}"
          f"  range=[{result['min_ste']:.3f}, {result['max_ste']:.3f}]  n={result['n_users']}")
    print(f"{'uid':>6} {'STE':>8} {'Delta':>10} {'sigma':>9} {'best arm':>9} {'clip%':>7}")
    for r in users:
        print(
            f"{r['userid']:>6} {r['ste']:>8.3f} {r['delta']:>10.3f} {r['sigma']:>9.3f}"
            f" {r['best_policy']:>9} {100 * r['clip_rate_best']:>6.1f}%"
        )
    won = [r["best_policy"] for r in users]
    if "dqn" in won:
        print(f"  DQN arm won for {won.count('dqn')} of {len(won)} participants")


def _json_safe(obj):
    """Drop bulky raw draws and turn NaN/inf into null so the report is valid JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if k != "zero_totals"}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, allow_nan=False)
    print(f"  wrote {path}")


def resolve_dqn_dir(args) -> str | None:
    if getattr(args, "dqn_model_dir", None):
        return str(Path(args.dqn_model_dir).expanduser().resolve())
    if getattr(args, "dqn_exp", None):
        return str((PROJECT_ROOT / "d3rlpy_logs" / f"ste_exp_{args.dqn_exp}").resolve())
    return None


def make_spec(args) -> ProxySpec:
    return ProxySpec(
        episodes=args.episodes,
        seed0=args.seed,
        noise=args.noise,
        policy_grid=tuple(float(p) for p in args.policy_grid),
        dqn_model_dir=resolve_dqn_dir(args),
    )


def cmd_eval(args) -> None:
    params_dir = Path(args.params_dir).expanduser().resolve()
    user_ids = load_user_ids(params_dir)
    require_fitted_blocks(params_dir, user_ids)
    spec = make_spec(args)
    require_dqn_checkpoints(spec, user_ids)
    t0 = time.perf_counter()
    result = evaluate(
        params_dir, user_ids, spec, n_jobs=args.jobs, proxy_to_true=args.proxy_to_true
    )
    print(f"\n{params_dir.name}  ({time.perf_counter() - t0:.1f}s)")
    summarise(result)
    if args.report:
        write_report(Path(args.report), {"spec": asdict(spec), **result})


def cmd_scan(args) -> None:
    base_dir = Path(args.params_dir).expanduser().resolve()
    user_ids = load_user_ids(base_dir)
    require_fitted_blocks(base_dir, user_ids)
    spec = make_spec(args)
    require_dqn_checkpoints(spec, user_ids)
    knob = KNOBS[args.knob]
    scratch = Path(args.scratch).expanduser().resolve()

    print(f"knob '{knob.name}': {knob.doc}")
    zero_cache = None
    rows = []
    for kappa in args.kappas:
        mult = knob.build(float(kappa))
        write_scaled_params(base_dir, scratch, user_ids, mult)
        result = evaluate(
            scratch,
            user_ids,
            spec,
            zero_cache=zero_cache,
            n_jobs=args.jobs,
            proxy_to_true=args.proxy_to_true,
        )
        if zero_cache is None and knob.sigma_invariant:
            # Action-gated scaling leaves the control arm untouched: measure once.
            zero_cache = zero_cache_from(result)
        rows.append((float(kappa), result))
        print(
            f"  kappa={kappa:8.4f}  mean STE={result['mean_ste']:7.4f}"
            f"  median={result['median_ste']:7.4f}"
            f"  range=[{result['min_ste']:.3f}, {result['max_ste']:.3f}]"
        )
    if args.report:
        write_report(
            Path(args.report),
            {
                "knob": knob.name,
                "spec": asdict(spec),
                "scan": [
                    {"kappa": k, **{m: r[m] for m in ("mean_ste", "median_ste", "min_ste", "max_ste")}}
                    for k, r in rows
                ],
            },
        )


def cmd_calibrate(args) -> None:
    base_dir = Path(args.params_dir).expanduser().resolve()
    user_ids = load_user_ids(base_dir)
    require_fitted_blocks(base_dir, user_ids)
    spec = make_spec(args)
    require_dqn_checkpoints(spec, user_ids)
    knob = KNOBS[args.knob]
    scratch = Path(args.scratch).expanduser().resolve()
    require_stable = bool(args.require_stable)

    print(f"knob '{knob.name}': {knob.doc}")
    arms = [f"Bernoulli p in {spec.policy_grid}"]
    if spec.dqn_model_dir:
        arms.append(f"DQN from {Path(spec.dqn_model_dir).name}")
    print(f"proxy: max over [{', '.join(arms)}] vs never-suggest, "
          f"{spec.episodes} paired episodes/arm, {len(user_ids)} participants")
    if args.proxy_to_true != 1.0:
        print(f"proxy-to-true calibration factor: {args.proxy_to_true:g}")

    source_gains = diagnose(base_dir, user_ids)
    source_bad = unstable_users(source_gains)
    if source_bad:
        uids = [int(r["userid"]) for r in source_bad]
        msg = (
            f"{base_dir.name} already has an unstable E_w or CAE loop for "
            f"{len(uids)} participants {uids} (|g| >= {GAIN_LIMIT:g})."
        )
        if require_stable:
            raise SystemExit(msg)
        print(f"WARNING: {msg}", flush=True)

    kappa_hi = 256.0
    if require_stable:
        kappa_hi = largest_stable_kappa(base_dir, user_ids, knob, x_hi=256.0)
        print(
            f"joint E_w/CAE stability cap: kappa <= {kappa_hi:.5f}  "
            f"(|g_zero|, |g_always| < {GAIN_LIMIT:g})"
        )
        if kappa_hi <= 0.0:
            raise SystemExit("no non-negative kappa is loop-stable; aborting.")

    zero_cache: dict[int, list[float]] | None = None
    last: dict[float, dict] = {}

    def f(kappa: float) -> float:
        nonlocal zero_cache
        if float(kappa) in last:  # the response curve is shared across targets
            return last[float(kappa)]["mean_ste"]
        write_scaled_params(base_dir, scratch, user_ids, knob.build(float(kappa)))
        result = evaluate(
            scratch,
            user_ids,
            spec,
            zero_cache=zero_cache,
            n_jobs=args.jobs,
            proxy_to_true=args.proxy_to_true,
        )
        if zero_cache is None and knob.sigma_invariant:
            zero_cache = zero_cache_from(result)
        last[float(kappa)] = result
        return result["mean_ste"]

    x0 = float(args.kappa0)
    if require_stable:
        x0 = min(max(x0, 0.0), kappa_hi)
        if x0 == 0.0 and kappa_hi > 0.0:
            x0 = min(1.0, kappa_hi)
        # One evaluation at the cap so the solver knows the feasible STE range.
        if kappa_hi < 256.0:
            print(f"  max-stable kappa={kappa_hi:.5f}  ->  mean STE={f(kappa_hi):.4f}")

    solutions: list[tuple[float, float, float, Path]] = []
    failures: list[str] = []
    for target in args.targets:
        print(f"\n=== target mean STE = {target:g} ===")
        t0 = time.perf_counter()
        kappa, achieved, hist = solve_scalar(
            f,
            float(target),
            x0=x0,
            f_at_zero=0.0 if knob.zero_is_null else None,
            seed_points={k: r["mean_ste"] for k, r in last.items()},
            tol=args.tol,
            max_iter=args.max_iter,
            x_hi=kappa_hi,
        )
        if kappa not in last:  # e.g. the free f(0) endpoint won the search
            f(kappa)
        result = last[kappa]
        gains = loop_gains_for(base_dir, user_ids, knob.build(kappa))
        bad = unstable_users(gains)
        worst = max(gains, key=max_abs_gain)
        on_target = abs(achieved - float(target)) <= args.tol
        stable = not bad

        print(f"  kappa* = {kappa:.5f}  ->  mean STE {achieved:.4f} "
              f"({time.perf_counter() - t0:.1f}s, {len(hist)} evaluations)")
        summarise(result)
        print(
            f"  worst joint loop gain: uid {worst['userid']}  "
            f"E_w=({worst['g_zero']:+.3f}, {worst['g_always']:+.3f})  "
            f"CAE=({worst['cae_g_zero']:+.3f}, {worst['cae_g_always']:+.3f})"
            f"{'  UNSTABLE' if bad else ''}"
        )

        reasons = []
        if require_stable and not stable:
            reasons.append(
                "unstable users "
                + str([int(r["userid"]) for r in bad])
            )
        if not on_target:
            reasons.append(
                f"proxy STE {achieved:.4f} missed target {target:g} "
                f"(tol {args.tol:g})"
            )
        if reasons:
            msg = f"  skip env_para folder: {'; '.join(reasons)}"
            print(msg)
            failures.append(f"target {target:g}: {'; '.join(reasons)}")
            continue

        out_dir = PROJECT_ROOT / f"{args.out_prefix}{target:g}"
        audit = write_scaled_params(base_dir, out_dir, user_ids, knob.build(kappa))
        written = diagnose(out_dir, user_ids)
        if require_stable and unstable_users(written):
            raise SystemExit(
                f"wrote {out_dir} but diagnose reports it unstable -- this is a bug."
            )
        print(f"  wrote {out_dir}")

        report = {
            "target_mean_ste": float(target),
            "achieved_mean_ste": float(achieved),
            "knob": knob.name,
            "kappa": float(kappa),
            "kappa_hi": float(kappa_hi),
            "stable": True,
            "multipliers": knob.build(kappa),
            "search_path": [{"kappa": x, "mean_ste": y} for x, y in hist],
            "spec": asdict(spec),
            "source_params_dir": str(base_dir),
            "loop_gains": written,
            "n_coefficients_scaled": len(audit),
            **result,
        }
        report["params_dir"] = str(out_dir)
        write_report(out_dir / "ste_tuning.json", report)
        solutions.append((float(target), float(kappa), float(achieved), out_dir))

    print("\nsummary")
    if solutions:
        for target, kappa, achieved, out_dir in solutions:
            print(f"  target {target:4.2f}  kappa {kappa:9.5f}  proxy STE {achieved:6.3f}  {out_dir.name}")
    else:
        print("  (no folders written)")
    if failures:
        print("failed targets:")
        for line in failures:
            print(f"  {line}")
        code = 2 if solutions else 1
        raise SystemExit(code)
    final = solutions[-1][3].name
    print(
        "\nNext: run_tune_ste.sh submits a full DQN train+eval in each folder "
        "(TUNE_VALIDATE=1, the default). To do it by hand:\n"
        f"    ADAPR_PARAMS_DIR={final} python ste_vanilla.py train ... / eval ... / aggregate ..."
    )


def cmd_diagnose(args) -> None:
    params_dir = Path(args.params_dir).expanduser().resolve()
    user_ids = load_user_ids(params_dir)
    rows = diagnose(params_dir, user_ids)
    print(
        f"{params_dir.name}: compound E_w and CAE loop gains "
        f"(|g| < {GAIN_LIMIT:g} required)"
    )
    print(
        f"{'uid':>6} {'Ew_zero':>9} {'Ew_always':>10} "
        f"{'CAE_zero':>9} {'CAE_always':>10}"
    )
    bad = {int(r["userid"]) for r in unstable_users(rows)}
    for r in sorted(rows, key=lambda x: -max_abs_gain(x)):
        flag = "  UNSTABLE" if int(r["userid"]) in bad else ""
        print(
            f"{r['userid']:>6} {r['g_zero']:>+9.3f} {r['g_always']:>+10.3f} "
            f"{r['cae_g_zero']:>+9.3f} {r['cae_g_always']:>+10.3f}{flag}"
        )
    print(f"{len(bad)} of {len(rows)} participants unstable")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, *, needs_knob: bool) -> None:
        sp.add_argument("--params-dir", default=str(PARAMS_DIR))
        sp.add_argument("--episodes", type=int, default=ProxySpec.episodes)
        sp.add_argument("--seed", type=int, default=ProxySpec.seed0)
        sp.add_argument("--noise", default="ar1", choices=["ar1", "random", "sequential"])
        sp.add_argument(
            "--policy-grid", type=float, nargs="+", default=list(ProxySpec.policy_grid)
        )
        sp.add_argument("--jobs", type=int, default=None)
        sp.add_argument("--proxy-to-true", type=float, default=1.0)
        sp.add_argument("--report", default=None)
        sp.add_argument(
            "--dqn-exp",
            default=None,
            help=(
                "Add each participant's trained DQN as a treatment arm, reading "
                "d3rlpy_logs/ste_exp_<EXP>/user<uid>_model.d3 (the --exp value "
                "used with ste_vanilla.py train)"
            ),
        )
        sp.add_argument(
            "--dqn-model-dir",
            default=None,
            help="Explicit checkpoint directory; overrides --dqn-exp",
        )
        if needs_knob:
            sp.add_argument("--knob", default="action", choices=sorted(KNOBS))
            sp.add_argument(
                "--scratch",
                default=str(PROJECT_ROOT / ".ste_tune_scratch"),
                help="Working directory for candidate parameter sets",
            )

    sp = sub.add_parser("eval", help="Measure proxy STE for a parameter directory")
    common(sp, needs_knob=False)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("scan", help="Trace mean STE across knob values")
    common(sp, needs_knob=True)
    sp.add_argument("--kappas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("calibrate", help="Solve for the knob value hitting a target STE")
    common(sp, needs_knob=True)
    sp.add_argument("--targets", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    sp.add_argument("--out-prefix", default="env_para_ste")
    sp.add_argument("--kappa0", type=float, default=1.0)
    sp.add_argument("--tol", type=float, default=0.02)
    sp.add_argument("--max-iter", type=int, default=10)
    sp.add_argument(
        "--require-stable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refuse to write a folder unless every participant has both "
            "|E_w loop gain| < 1 and |CAE loop gain| < 1 under never-suggest "
            "and always-suggest (default: on)"
        ),
    )
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser(
        "diagnose", help="E_w and CAE loop gains for a parameter directory"
    )
    sp.add_argument("--params-dir", default=str(PARAMS_DIR))
    sp.set_defaults(func=cmd_diagnose)

    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
