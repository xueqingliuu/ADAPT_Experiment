"""Build STE-tuned environment variants from ``env_para_vanilla``.

The scientific design (three-variant protocol)
==============================================
The testbed's causal chain, per week ``w`` (all coefficients per-participant,
fitted from the ADAPT MRT)::

    A ──(+)──> MY (4h steps, antic) ──(+)──> Y_{w+1}   (CAE, the outcome)
    A ──(−)──> ME (PV, FW, PJ)      ──(+)──> E_{w+1} ──(+)──> MY_{w+1} ──> Y_{w+2}
                                     (ME→E)             (E→MY)

Variants (``protocol`` writes three folders plus untouched vanilla; all
searches start from vanilla, not from a stacked fatigue folder):

1. **vanilla** — the fitted environment, untouched.
2. **STE 0.2** — first try knob ``burden_path``: fix the transmissions
   (E→MY ``+κ_e``, default 0.3; ME→E floored at 0.05) and *search only*
   A→ME mains ``−κ``. One raw ``κ`` must not be added to A→ME (≈0.3),
   ME→E (≈1), and E→MY (≈0.003) together — that reintroduces the
   never-send σ-inflation that ``fatigue`` decoupled. If the search
   cannot hit 0.2, fall back to ``benefit_foursc`` on vanilla.
3. **STE 0.5 / 0.8** — ``benefit_foursc`` on vanilla: *scale* A→MY by
   ``κ`` (keeps each user's A×wear / A×interact / A×steps CATE ratios)
   and *shift* fourSC→Y by ``c(κ−1)`` (capped). ``κ > 1`` raises STE.

Shift vs scale — the rule used throughout
-----------------------------------------
*Shift* (θ ← θ − κ) when the goal is a **sign-definite effect** and the fitted
coefficients are near zero with mixed signs — a scale cannot move a ≈0
coefficient and amplifies wrong signs. In the current vanilla fit: A→ME mains
are mixed-sign (PV action effect positive for 23/31 users), E→fourSC is
positive for only 10/31, fourSC→Y ≈ 0.08 with 2/31 negative. *Scale*
(θ ← κθ) when the goal is to **amplify structure you want to keep** —
A→MY's state-dependent CATEs. Never shift interaction terms; shifts belong on
main effects only.

The proxy STE (what "target 0.2/0.5/0.8" means here)
----------------------------------------------------
For each participant the proxy is ``max(0, max_arm Δ̂_i) / σ̂_i``, averaged
over users: Δ̂_i is the paired mean total-CAE gap vs never-suggest on the
**same** Monte-Carlo episodes used to pick the arm (Bernoulli rates from
``--policy-grid`` plus, optionally, a source-env DiscreteCQL via
``--dqn-exp``). The same-sample max is upward-biased for the oracle best-arm
Δ; clipping at 0 encodes that never-suggest is in the class. This is **not**
``ste_vanilla.aggregate_ste`` (DiscreteCQL trained in the tuned folder, gated
on held-out seeds, test Δ can be negative); hitting proxy 0.5 does not
guarantee confirmation STE 0.5. Report the confirmation number.

Commands
--------
    protocol    STE 0.2 via burden_path (then benefit_foursc fallback),
                then benefit_foursc → 0.5/0.8; all from vanilla
    diagnose    E_w and CAE loop gains (no simulation)
    eval        measure proxy STE of one parameter folder
    scan        proxy STE vs a grid of ``kappa``
    apply       write a folder at a fixed ``kappa`` (no STE search)
    calibrate   root-find ``kappa`` per target and write the folder(s)

Cluster::

    sbatch --export=ALL,TUNE_PHASE=protocol run_tune_ste.sh

or stage by stage::

    sbatch --export=ALL,TUNE_KNOB=burden_path,TUNE_TARGETS=0.2 run_tune_ste.sh
    sbatch --export=ALL,TUNE_KNOB=benefit_foursc,TUNE_TARGETS="0.5 0.8" run_tune_ste.sh

Knob reference (``--knob``)
---------------------------
    burden_path
              Whole burden chain vs vanilla, but only A→ME is searched:
              A→ME mains −κ (all mediators, after ME→E is floored);
              ME→E ← max(ME→E, 0.05); E→MY +κ_e (default 0.3, fixed).
              Never-send σ_i does not move with the search κ.
    fatigue (alias burden_shift)
              A→ME mains −κ (shift), per mediator only if that user's
              fitted ME→E > ADAPR_FATIGUE_ME_TO_E_MIN (default 0.05);
              E_w→fourSC / E_w→antic +κ_e (shift, κ_e =
              ADAPR_FATIGUE_E_SHIFT, default 0.3, not tied to κ). Search
              κ ≤ ADAPR_FATIGUE_KAPPA_MAX (default 2). Kept for scans /
              apply; protocol no longer writes a κ=2 fatigue folder.
    benefit_foursc
              A→MY ×κ (scale) and fourSC_ewma→Y += c(κ−1), c =
              ADAPR_BENEFIT_FOURSC_SHIFT (0.05) capped at _CAP (0.20).
              κ=1 is a no-op; κ<1 lowers STE, κ>1 raises it.
    action / benefit / burden
              pure multiplicative dials on A→{MY,ME} — cannot create a
              sign-definite effect from mixed-sign fits; kept for scans.
    foursc_to_y_shift
              +κ on fourSC_ewma→Y alone. Caps near proxy 0.36 (CAE clips;
              job 41048118) — not a route to 0.5/0.8.
    my_to_y / foursc_to_y / me_to_e / e_to_my
              structural single-pathway scales, for diagnostics only.

Recorded dead ends: burden_shift cannot reach STE 0.2 on the *old* vanilla
(floor ≈0.27, job 40926808) — re-check after a refit; foursc_to_y_shift
cannot reach 0.5/0.8 (job 41048118).

Guarantees and guards
---------------------
* Stability: a folder is written only if every participant keeps
  |loop gain| < 1 for E_w and CAE under never- and always-suggest
  (``--require-stable``, default on); the search never leaves that region.
* Sign report: after writing, each variant's intended sign story is checked
  per participant (fatigue: A→ME ≤ 0; transmission: ME→E, E→MY ≥ 0;
  benefit: fourSC→Y, antic→Y ≥ 0) and violations are printed — fitted
  heterogeneity, so warn-only.
* PV is a hurdle: ``alpha3``/``alpha4`` are the occurrence logit
  ``P(count>0)``; ``gamma3``/``gamma4`` are the Gaussian mean given
  count>0. Current A→ME knobs (``burden_path``, ``fatigue``) subtract
  ``κ`` from the mains ``alpha3`` and ``gamma3`` only; the A×E_w slopes
  ``alpha4``/``gamma4`` stay at the fitted values. Those JSON edits
  enter the simulator with no extra translation.
* ``rl_priors.json`` is copied verbatim (RCT/vanilla priors by design);
  ``loo_priors`` is a relative symlink to vanilla.

Use a tuned folder instead of vanilla::

    ADAPR_PARAMS_DIR=env_para_ste0.5 python ste_vanilla.py train 0 --exp ste0.5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ste_vanilla import (
    ADVANTAGE_MARGIN,
    STE_ALGO,
    STE_OBS_DIM,
    STE_OBSERVATION_SCALER,
    _model_metadata_path,
    rollout_total_cae,
)
from vani_env import (
    PARAMS_DIR,
    THETA_ANTIC_NAMES,
    THETA_CAE_NAMES,
    THETA_FOURSC_NAMES,
    EnvConfig,
    pv_hurdle_occurrence_z_gap,
)

PROJECT_ROOT = Path(__file__).resolve().parent

# Default "large fatigue" shift: enough that always-send is a weaker constant
# policy than the current STE-0.5 burden_shift folder (κ≈0.11), still below
# the joint loop-gain cap (κ_hi≈0.65 in vanilla). Proxy STE at this value
# was ~0.29, so a foursc_to_y_shift ladder on top should target 0.5 / 0.8,
# not 0.2.
BURDEN_SHIFT_LARGE_KAPPA = 0.4

# Fixed E_w→MY transmission addend for the fatigue knob. Fitted E→fourSC /
# E→antic are ≈0.003 (compound E→Y ≈0.001), so the conduit to Y is created
# here, once. Do not tie this to the A→ME search variable: that would make
# never-send σ_i and Y's E-driven serial correlation grow with κ, and part
# of the STE drop would come through the denominator. Override with
# ADAPR_FATIGUE_E_SHIFT (a float, or "kappa" to restore the old coupling).
FATIGUE_E_SHIFT_DEFAULT = 0.3

# Subtract κ from A→ME(m) only if that user's ME→E loading for m exceeds
# this threshold. ``> 0`` is too tight (near-zero loadings still invert
# under a large κ); 0.05 is a small but definite positive conduit.
# Override with ADAPR_FATIGUE_ME_TO_E_MIN.
FATIGUE_ME_TO_E_MIN = 0.05

# Hard search cap for the fatigue κ. A→ME mains do not enter the E_w /
# CAE loop-gain formulas, so ``largest_stable_kappa`` returns 256 and is
# not a real bound. 2 is large vs typical |A→ME| (mean 0.29, max 1.48)
# without inventing an order-of-magnitude engagement cost. Override with
# ADAPR_FATIGUE_KAPPA_MAX.
FATIGUE_KAPPA_MAX = 2.0

# Copied verbatim into every tuned parameter directory so it can be handed to
# ``ste_vanilla.py`` / ``experiment.py`` as a drop-in replacement.
# ``rl_priors.json`` is the RCT/vanilla prior (not re-fit on the κ-scaled
# DGP). ``loo_priors`` is installed as a relative symlink, not copied.
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
# The ``query_Jw_*`` suffix of PV/FW/PJ is deliberately excluded: it is a
# ``J_w`` term, not an action coefficient. STE still does not scale it.

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
    "FOURSC_to_Y": {
        "theta_CAE": ("fourSC_ewma",),
    },
    "ANTIC_to_Y": {
        "theta_CAE": ("anticipated_affect_ewma",),
    },
    "A_to_ME": {
        "theta_penalized_PV": (
            "alpha3_action",
            "alpha4_action_by_Ew",
            "gamma3_action",
            "gamma4_action_by_Ew",
        ),
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
    # Main action effects only: a constant (in E_w) shift of the A→ME
    # intercept. Subtracting κ from the A×E_w terms as well would change
    # the action effect by −κ(1+E_w) and flip sign for E_w < −1.
    # Split per mediator so fatigue can sign-gate PV / FW / PJ separately.
    "A_to_ME_main": {
        "theta_penalized_PV": ("alpha3_action", "gamma3_action"),
        "theta_penalized_FW": ("beta3_A0_morning", "beta5_A1_afternoon"),
        "theta_penalized_PJ": ("theta3_A0_morning", "theta5_A1_afternoon"),
    },
    "A_to_ME_PV_main": {
        "theta_penalized_PV": ("alpha3_action", "gamma3_action"),
    },
    "A_to_ME_FW_main": {
        "theta_penalized_FW": ("beta3_A0_morning", "beta5_A1_afternoon"),
    },
    "A_to_ME_PJ_main": {
        "theta_penalized_PJ": ("theta3_A0_morning", "theta5_A1_afternoon"),
    },
    "ME_to_E": {
        "theta_penalized_Ew": ("a2_PV_lag_week", "a3_FW_lag_week", "a4_PJ_lag_week"),
    },
    "ME_to_E_PV": {"theta_penalized_Ew": ("a2_PV_lag_week",)},
    "ME_to_E_FW": {"theta_penalized_Ew": ("a3_FW_lag_week",)},
    "ME_to_E_PJ": {"theta_penalized_Ew": ("a4_PJ_lag_week",)},
    "E_to_MY": {
        "theta_fourSC": ("perceived_utility_lastweek",),
        "theta_antic": ("perceived_utility_lastweek",),
    },
}

# Pathways that are gated by an action indicator, hence invisible to pi_0.
ACTION_GATED = frozenset({
    "A_to_MY",
    "A_to_ME",
    "A_to_ME_main",
    "A_to_ME_PV_main",
    "A_to_ME_FW_main",
    "A_to_ME_PJ_main",
})

_ME_TO_E = {
    "PV": ("theta_penalized_Ew", "a2_PV_lag_week"),
    "FW": ("theta_penalized_Ew", "a3_FW_lag_week"),
    "PJ": ("theta_penalized_Ew", "a4_PJ_lag_week"),
}

_A_ME_SIGN_TO_MEDIATOR = {
    "A->PV main <= 0": "PV",
    "A->PV intensity main <= 0": "PV",
    "A->FW main <= 0": "FW",
    "A->PJ main <= 0": "PJ",
}

# Fallback coefficient names for blocks that carry no ``*_names`` key in JSON.
_NAME_CONSTANTS = {
    "theta_fourSC": THETA_FOURSC_NAMES,
    "theta_antic": THETA_ANTIC_NAMES,
    "theta_CAE": THETA_CAE_NAMES,
}

# A knob maps a scalar kappa to a per-pathway multiplier dict. Knobs are named so
# that STE is monotone in kappa; ``increasing`` records the direction.
Knob = Callable[[float], dict[str, float]]


def _fatigue_e_tied_to_kappa() -> bool:
    """True only if ``ADAPR_FATIGUE_E_SHIFT=kappa`` restores the old coupling."""
    return os.getenv("ADAPR_FATIGUE_E_SHIFT", "").strip().lower() in {"kappa", "k"}


def _fatigue_e_shift(k: float) -> float:
    """E_w→MY transmission shift paired with the A→ME fatigue shift ``k``.

    Default: ``FATIGUE_E_SHIFT_DEFAULT`` (0.3), independent of ``k``, so the
    never-send arm is the same at every point on the A→ME ladder. Applied as
    a *shift* because fitted E→fourSC is near zero (mean ≈0.003): a scale
    cannot create the conduit. Override with ``ADAPR_FATIGUE_E_SHIFT`` (a
    float, or ``kappa`` to set ``κ_e = k`` as in the original coupling).
    """
    raw = os.getenv("ADAPR_FATIGUE_E_SHIFT", "").strip()
    if raw.lower() in {"kappa", "k"}:
        return float(k)
    if raw:
        return float(raw)
    return float(FATIGUE_E_SHIFT_DEFAULT)


def _fatigue_me_to_e_min() -> float:
    raw = os.getenv("ADAPR_FATIGUE_ME_TO_E_MIN", "").strip()
    return float(raw) if raw else float(FATIGUE_ME_TO_E_MIN)


def _fatigue_kappa_max() -> float:
    raw = os.getenv("ADAPR_FATIGUE_KAPPA_MAX", "").strip()
    return float(raw) if raw else float(FATIGUE_KAPPA_MAX)


def _fatigue_multipliers(k: float) -> dict[str, float]:
    """Population-level fatigue map (all three A→ME mains). Writes use the
    per-user gated version in ``_fatigue_multipliers_for_user``."""
    return {
        "A_to_ME_PV_main": float(k),
        "A_to_ME_FW_main": float(k),
        "A_to_ME_PJ_main": float(k),
        "E_to_MY": -_fatigue_e_shift(k),
    }


def _burden_path_multipliers(k: float) -> dict[str, float]:
    """Population-level map. ME→E floors are per-user (see multipliers_for)."""
    return {
        "A_to_ME_PV_main": float(k),
        "A_to_ME_FW_main": float(k),
        "A_to_ME_PJ_main": float(k),
        "E_to_MY": -_fatigue_e_shift(0.0),
    }


_BURDEN_PATH_APPLY = {
    "A_to_ME_PV_main": "subtract",
    "A_to_ME_FW_main": "subtract",
    "A_to_ME_PJ_main": "subtract",
    "ME_to_E_PV": "subtract",
    "ME_to_E_FW": "subtract",
    "ME_to_E_PJ": "subtract",
    "E_to_MY": "subtract",
}


def _benefit_foursc_shift(k: float) -> float:
    """fourSC_ewma → CAE addend paired with an A→MY multiplier ``k``.

    Defaults keep the shift well below the foursc_to_y_shift cap that
    saturated CAE (job 41048118, κ≈1.53). Override with
    ``ADAPR_BENEFIT_FOURSC_SHIFT`` (per unit of κ−1) and
    ``ADAPR_BENEFIT_FOURSC_SHIFT_CAP``.
    """
    per = float(os.getenv("ADAPR_BENEFIT_FOURSC_SHIFT", "0.05"))
    cap = float(os.getenv("ADAPR_BENEFIT_FOURSC_SHIFT_CAP", "0.20"))
    return float(np.clip(per * (float(k) - 1.0), -cap, cap))


def _benefit_foursc_multipliers(k: float) -> dict[str, float]:
    return {"A_to_MY": float(k), "FOURSC_to_Y": -_benefit_foursc_shift(k)}


_BENEFIT_FOURSC_APPLY = {"A_to_MY": "multiply", "FOURSC_to_Y": "subtract"}


@dataclass(frozen=True)
class KnobSpec:
    name: str
    build: Knob
    zero_is_null: bool  # STE(kappa = 0) == 0 holds exactly, for free
    sigma_invariant: bool  # control arm untouched, so sigma_i can be cached
    doc: str
    apply: str | dict[str, str] = "multiply"  # or per-pathway map
    kappa_cap: float | None = None  # search hi; fatigue uses FATIGUE_KAPPA_MAX


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
        doc=(
            "Multiply action -> engagement coefficients by kappa. Does not "
            "flip sign: fitted A→ME is often positive, so kappa>1 makes "
            "sending raise PV/FW/PJ more, not less."
        ),
    ),
    "burden_path": KnobSpec(
        "burden_path",
        _burden_path_multipliers,
        zero_is_null=False,
        sigma_invariant=True,
        apply=_BURDEN_PATH_APPLY,
        doc=(
            "Burden chain vs vanilla with a single search on A→ME: subtract "
            "κ from all A→ME mains (not A×E_w), floor each ME→E loading at "
            "ADAPR_FATIGUE_ME_TO_E_MIN (default 0.05), and add a fixed κ_e "
            "to E→MY (ADAPR_FATIGUE_E_SHIFT, default 0.3). Transmissions "
            "do not grow with the search variable, so never-send σ_i is "
            "constant on the κ ladder. First attempt for STE 0.2."
        ),
    ),
    "fatigue": KnobSpec(
        "fatigue",
        _fatigue_multipliers,
        zero_is_null=False,
        # Control arm vs vanilla moves (fixed E→MY addend) but not with κ,
        # so σ_i can be cached across the A→ME search. False only if
        # ADAPR_FATIGUE_E_SHIFT=kappa (see knob_sigma_invariant).
        sigma_invariant=True,
        apply="subtract",
        kappa_cap=FATIGUE_KAPPA_MAX,
        doc=(
            "Coherent fatigue (variant 2): subtract kappa from the main A→ME "
            "action coefficients (not A×E_w), per mediator only when that "
            "user's ME→E loading exceeds ADAPR_FATIGUE_ME_TO_E_MIN (default "
            "0.05), and add a fixed κ_e to E_w → fourSC and E_w → "
            "anticipated affect (κ_e = ADAPR_FATIGUE_E_SHIFT, default 0.3; "
            "raw units; typical |A→ME| ≈ 0.3, fitted E→MY ≈ 0.003). The "
            "fatigue addend is −κ A, independent of E_w. Sending lowers "
            "engagement, a lower E_w then lowers steps and affect, so the "
            "path reaches CAE. Search κ is capped at "
            "ADAPR_FATIGUE_KAPPA_MAX (default 2). The E→MY addend is "
            "identical across the κ ladder (set ADAPR_FATIGUE_E_SHIFT=kappa "
            "to restore the old coupling). Alias: burden_shift."
        ),
    ),
    "my_to_y": KnobSpec(
        "my_to_y",
        lambda k: {"FOURSC_to_Y": k, "ANTIC_to_Y": k},
        zero_is_null=False,
        sigma_invariant=False,
        doc="Structural: both mediator -> CAE loadings. Moves sigma_i as well as Delta_i.",
    ),
    "foursc_to_y": KnobSpec(
        "foursc_to_y",
        lambda k: {"FOURSC_to_Y": k},
        zero_is_null=False,
        sigma_invariant=False,
        doc=(
            "Structural: multiply fourSC_ewma -> CAE, holding the antic "
            "loading fixed. Fitted fourSC_ewma is small (~0.01 on average) "
            "while the fourSC CATE is the state-dependent one (A x interact / "
            "wear / steps), so kappa >> 1 is typical. Also moves sigma_i and "
            "the CAE loop gain. Users with a negative fitted loading get a "
            "more negative one — prefer foursc_to_y_shift unless you want "
            "that sign-preserving scale."
        ),
    ),
    "foursc_to_y_shift": KnobSpec(
        "foursc_to_y_shift",
        lambda k: {"FOURSC_to_Y": -k},
        zero_is_null=False,
        sigma_invariant=False,
        apply="subtract",
        doc=(
            "Add kappa to every user's fourSC_ewma → CAE loading (raw units; "
            "typical |loading| ≈ 0.01–0.05). Negative fitted loadings become "
            "less negative / positive instead of more negative. Moves the "
            "control arm and the CAE loop gain. Start a scan at 0.02–0.1."
        ),
    ),
    "benefit_foursc": KnobSpec(
        "benefit_foursc",
        _benefit_foursc_multipliers,
        zero_is_null=False,
        sigma_invariant=False,
        apply=_BENEFIT_FOURSC_APPLY,
        doc=(
            "Joint personalization + STE dial: multiply A→MY (steps/affect "
            "CATE, including A×wear/interact) by kappa and add "
            "c(kappa-1) to fourSC_ewma → CAE (c=0.05, cap 0.20). The "
            "burden chain is left alone. On vanilla this raises STE to "
            "0.5/0.8 (κ>1) and is the fallback if burden_path misses "
            "0.2. kappa=1 is a no-op. Start a raise search at 1.5–2."
        ),
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

# Backward-compatible alias: ``burden_shift`` was the original name of the
# fatigue knob. Reports and ste_tuning.json now record ``knob: fatigue``;
# overwriting a folder whose report says ``burden_shift`` therefore requires
# --overwrite, which is intentional — pre-2026-08-25 burden_shift folders
# also shifted the A×E_w interactions and must be regenerated, and the
# PV-hurdle action translation changed from ratio to additive on the same
# date. Default E→MY addend is FATIGUE_E_SHIFT_DEFAULT, not κ.
KNOBS["burden_shift"] = KNOBS["fatigue"]


def knob_sigma_invariant(knob: KnobSpec) -> bool:
    """True when the never-send arm (hence σ_i) does not depend on kappa."""
    if knob.name == "fatigue" and _fatigue_e_tied_to_kappa():
        return False
    return bool(knob.sigma_invariant)


def knob_kappa_cap(knob: KnobSpec) -> float | None:
    """Search-hi cap for this knob, honouring env overrides."""
    if knob.name == "fatigue":
        return _fatigue_kappa_max()
    if knob.kappa_cap is None:
        return None
    return float(knob.kappa_cap)


def _log_burden_path(knob: KnobSpec, *, log=print) -> None:
    if knob.name != "burden_path":
        return
    log(
        f"  burden_path search: A→ME −κ only  "
        f"(all mediators; ME→E floored at {_fatigue_me_to_e_min():g})"
    )
    log(
        f"  E→MY shift κ_e = {_fatigue_e_shift(0.0):g}  "
        "(fixed; not tied to the A→ME search)"
    )


def _log_fatigue_e_shift(knob: KnobSpec, *, log=print) -> None:
    if knob.name != "fatigue":
        return
    if _fatigue_e_tied_to_kappa():
        log("  E→MY shift κ_e = κ  (tied; control arm moves with the search)")
    else:
        log(
            f"  E→MY shift κ_e = {_fatigue_e_shift(0.0):g}  "
            "(fixed across the κ ladder)"
        )
    log(
        f"  A→ME sign-gate: ME→E > {_fatigue_me_to_e_min():g}  "
        "(ADAPR_FATIGUE_ME_TO_E_MIN; ungated mediators keep fitted A→ME)"
    )
    log(
        f"  fatigue κ cap: {_fatigue_kappa_max():g}  "
        "(ADAPR_FATIGUE_KAPPA_MAX)"
    )


def _fatigue_report_fields(knob: KnobSpec, kappa: float | None = None) -> dict:
    if knob.name == "burden_path":
        return {
            "burden_path_a_me_kappa": None if kappa is None else float(kappa),
            "fatigue_e_shift": _fatigue_e_shift(0.0),
            "fatigue_me_to_e_min": _fatigue_me_to_e_min(),
            "pathways": [
                "A_to_ME_main (search)",
                "ME_to_E (floor at fatigue_me_to_e_min)",
                "E_to_MY (fixed κ_e)",
            ],
        }
    if knob.name != "fatigue":
        return {}
    tied = _fatigue_e_tied_to_kappa()
    shift = float(kappa) if tied and kappa is not None else (
        None if tied else _fatigue_e_shift(0.0)
    )
    return {
        "fatigue_e_shift": shift,
        "fatigue_e_shift_tied_to_kappa": tied,
        "fatigue_me_to_e_min": _fatigue_me_to_e_min(),
        "fatigue_kappa_max": _fatigue_kappa_max(),
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


def _coef(params: dict, key: str, coef: str) -> float:
    names = _names_for(key, params)
    return float(np.asarray(params[key], dtype=float).ravel()[names.index(coef)])


def _fatigue_gated_mediators(params: dict, extra: float = 0.0) -> list[str]:
    """Mediators whose ME→E loading is a definite positive conduit to E."""
    thresh = _fatigue_me_to_e_min()
    return [
        m for m, (key, coef) in _ME_TO_E.items()
        if _coef(params, key, coef) + float(extra) > thresh
    ]


def _me_to_e_floor_subtract(params: dict, mediator: str) -> float:
    """Amount to subtract so ME→E(m) becomes ``max(fitted, floor)``."""
    key, coef = _ME_TO_E[mediator]
    a = _coef(params, key, coef)
    return float(a - max(a, _fatigue_me_to_e_min()))


def _fatigue_multipliers_for_user(k: float, params: dict) -> dict[str, float]:
    """Fatigue map for one user: E→MY always; A→ME(m) only if gated."""
    mult: dict[str, float] = {"E_to_MY": -_fatigue_e_shift(k)}
    for m in _fatigue_gated_mediators(params):
        mult[f"A_to_ME_{m}_main"] = float(k)
    return mult


def _burden_path_multipliers_for_user(k: float, params: dict) -> dict[str, float]:
    """A→ME −κ for every mediator; ME→E floored; E→MY +κ_e (fixed)."""
    kk = float(k)
    mult: dict[str, float] = {
        "A_to_ME_PV_main": kk,
        "A_to_ME_FW_main": kk,
        "A_to_ME_PJ_main": kk,
        "E_to_MY": -_fatigue_e_shift(0.0),
    }
    for m in ("PV", "FW", "PJ"):
        floor_sub = _me_to_e_floor_subtract(params, m)
        if floor_sub != 0.0:
            mult[f"ME_to_E_{m}"] = floor_sub
    return mult


def _knob_multipliers_fn(
    knob: KnobSpec, kappa: float
) -> Callable[[dict], dict[str, float]]:
    k = float(kappa)
    if knob.name == "fatigue":
        return lambda params: _fatigue_multipliers_for_user(k, params)
    if knob.name == "burden_path":
        return lambda params: _burden_path_multipliers_for_user(k, params)
    return lambda params: knob.build(k)


def _log_fatigue_gates(
    src_dir: Path, user_ids: Sequence[int], *, log=print
) -> dict[str, list[str]]:
    """Print and return per-user gated A→ME mediators."""
    gates: dict[int, list[str]] = {}
    for uid in user_ids:
        with open(Path(src_dir) / f"params_env_{uid}.json", encoding="utf-8") as f:
            params = json.load(f)
        gates[int(uid)] = _fatigue_gated_mediators(params, extra=0.0)
    n = len(gates)
    for m in ("PV", "FW", "PJ"):
        n_m = sum(1 for g in gates.values() if m in g)
        log(f"  A→ME {m} gated for {n_m}/{n} users")
    ungated = [u for u, g in gates.items() if not g]
    if ungated:
        log(f"  no A→ME shift: {ungated}")
    return {str(u): g for u, g in gates.items()}


def _pathway_apply(apply: str | dict[str, str], pathway: str) -> str:
    mode = apply.get(pathway, "multiply") if isinstance(apply, dict) else apply
    if mode not in {"multiply", "subtract"}:
        raise ValueError(f"unknown apply={mode!r} for pathway {pathway!r}")
    return mode


def scale_params(
    params: dict,
    multipliers: dict[str, float],
    *,
    apply: str | dict[str, str] = "multiply",
) -> tuple[dict, list[dict]]:
    """Return a copy of ``params`` with the requested pathways rescaled.

    ``multipliers`` maps pathway name -> scale factor; missing pathways are left
    unchanged. ``apply="multiply"`` does ``θ ← κθ`` (skip κ=1).
    ``apply="subtract"`` does ``θ ← θ − κ`` (skip κ=0), so a positive κ
    pushes every listed coefficient more negative. ``apply`` may also be a
    per-pathway map (used by ``benefit_foursc``).
    """
    unknown = set(multipliers) - set(PATHWAYS)
    if unknown:
        raise KeyError(f"Unknown pathway(s): {sorted(unknown)}")

    out = dict(params)
    audit: list[dict] = []
    for pathway, factor in multipliers.items():
        factor = float(factor)
        mode = _pathway_apply(apply, pathway)
        if mode == "multiply" and factor == 1.0:
            continue
        if mode == "subtract" and factor == 0.0:
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
                values[idx] = old * factor if mode == "multiply" else old - factor
                audit.append(
                    {
                        "pathway": pathway,
                        "block": key,
                        "coef": coef,
                        "apply": mode,
                        "factor": factor,
                        "old": old,
                        "new": float(values[idx]),
                    }
                )
            out[key] = values.tolist()
    return out, audit


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    with open(path, encoding="utf-8") as f:
        json.load(f)


def _install_loo_priors_link(dst_dir: Path) -> None:
    """Point ``dst_dir/loo_priors`` at vanilla with a relative symlink.

    RCT/vanilla LOO bundles are the intended STE-folder priors. Absolute
    cluster paths break locally; a relative link travels with the repo.
    """
    vanilla_loo = (PROJECT_ROOT / "env_para_vanilla" / "loo_priors").resolve()
    if not vanilla_loo.is_dir():
        return
    dst_dir = dst_dir.resolve()
    if dst_dir == vanilla_loo.parent:
        return
    dst_loo = dst_dir / "loo_priors"
    rel = os.path.relpath(vanilla_loo, start=dst_dir)
    if dst_loo.is_symlink() or dst_loo.is_file():
        dst_loo.unlink()
    elif dst_loo.exists():
        return
    dst_loo.symlink_to(rel)


def write_scaled_params(
    src_dir: Path,
    dst_dir: Path,
    user_ids: Sequence[int],
    multipliers: dict[str, float] | None = None,
    *,
    apply: str | dict[str, str] = "multiply",
    multipliers_for: Callable[[dict], dict[str, float]] | None = None,
) -> list[dict]:
    """Materialise a tuned copy of ``src_dir`` at ``dst_dir``.

    ``multipliers_for(params)`` supplies a per-user map (fatigue sign-gate).
    When set it takes precedence over the shared ``multipliers`` dict.
    """
    if multipliers is None and multipliers_for is None:
        raise ValueError("write_scaled_params needs multipliers or multipliers_for")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in SUPPORTING_FILES:
        src = src_dir / fname
        if src.is_file():
            shutil.copy2(src, dst_dir / fname)
    _install_loo_priors_link(dst_dir)

    audit: list[dict] = []
    for uid in user_ids:
        src = src_dir / f"params_env_{uid}.json"
        with open(src, encoding="utf-8") as f:
            params = json.load(f)
        mult = multipliers_for(params) if multipliers_for is not None else multipliers
        scaled, rows = scale_params(params, mult, apply=apply)
        for r in rows:
            r["userid"] = int(uid)
        audit.extend(rows)
        _write_json_atomic(dst_dir / f"params_env_{uid}.json", scaled)
    return audit


def write_knob_params(
    src_dir: Path,
    dst_dir: Path,
    user_ids: Sequence[int],
    knob: KnobSpec,
    kappa: float,
) -> list[dict]:
    """Write ``src_dir`` scaled by ``knob`` at ``kappa`` (per-user if gated)."""
    return write_scaled_params(
        src_dir,
        dst_dir,
        user_ids,
        knob.build(float(kappa)),
        apply=knob.apply,
        multipliers_for=_knob_multipliers_fn(knob, kappa),
    )


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
    A short ``policy_grid`` is deliberate: ``Delta_i`` is a max over treatment
    arms on the same episodes used to estimate it (winner's curse). Extra arms
    raise the reported proxy and the selection bias; they are not a tighter
    bound on ``ste_vanilla.aggregate_ste``.

    ``dqn_model_dir`` points at a directory of ``ste_vanilla.py`` checkpoints
    (``user<uid>_model.d3``); when set, each participant's source-env
    DiscreteCQL policy is added as one more treatment arm (not retrained).
    """

    episodes: int = 100
    seed0: int = 20260814
    noise: str = "ar1"
    policy_grid: tuple[float, ...] = (0.5, 1.0)
    i_w_fixed: int = 1
    dqn_model_dir: str | None = None


_DQN_CACHE: dict[str, object] = {}


def load_dqn(model_dir: Path, uid: int, *, nweek: int, noise: str):
    """Load a ``ste_vanilla.py`` DiscreteCQL checkpoint, validating what must match.

    ``userid``/``nweek``/``noise``/``algo``/``state_dim`` have to agree or the
    policy is being applied to a different observation map than it was trained
    on. The fitted environment parameters are deliberately *not* checked:
    evaluating a baseline-trained policy in a rescaled environment is the
    entire point.
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
    expected_meta = {
        "userid": uid,
        "nweek": nweek,
        "noise": noise,
        "algo": STE_ALGO,
        "advantage_margin": ADVANTAGE_MARGIN,
        "observation_scaler": STE_OBSERVATION_SCALER,
        "state_dim": STE_OBS_DIM,
    }
    for name, expected in expected_meta.items():
        if meta.get(name) != expected:
            raise ValueError(
                f"DiscreteCQL checkpoint {path.name} mismatch on {name}: "
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
    # Population best-arm Δ cannot be negative (never-suggest is in the class).
    # Clip the same-sample estimate at 0; the unclipped winner is ``delta_raw``.
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
    n_nan = int(ste.size - finite.size)
    if n_nan:
        print(
            f"  [warn] {n_nan}/{ste.size} users had non-finite proxy STE; "
            "excluded from the mean and from n_users"
        )
    return {
        "params_dir": str(params_dir),
        "proxy_to_true": float(proxy_to_true),
        "mean_ste": float(np.mean(finite)) if finite.size else float("nan"),
        "median_ste": float(np.median(finite)) if finite.size else float("nan"),
        "min_ste": float(np.min(finite)) if finite.size else float("nan"),
        "max_ste": float(np.max(finite)) if finite.size else float("nan"),
        "n_users": int(finite.size),
        "n_users_total": int(ste.size),
        "n_users_nan": n_nan,
        "users": rows,
    }


def zero_cache_from(result: dict) -> dict[int, list[float]]:
    return {int(r["userid"]): r["zero_totals"] for r in result["users"]}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def loop_gain_Ew(
    params: dict,
    *,
    std: dict | None = None,
    params_dir: Path | None = None,
) -> dict[str, float]:
    """Compound ``E_w`` loop gain under never-suggest and always-suggest.

    ``E_w`` is linear in the weekly mediator summaries, and those summaries are
    fixed-denominator *averages* (``nansum(pv)/14``, ``nansum(FW|PJ)/7``) in both
    the estimator and ``vani_env._week_means_from_arrays``. A unit shift in
    ``E_{w-1}`` shifts every slot of the week, so the average shifts by the
    per-slot slope -- no 14x / 7x factor. FW and PJ are logistic (worst-case
    slope ``pi(1-pi) = 0.25``). PV is a hurdle: ``0.25 α_E (z̄₊ - z_0) + γ_E``,
    with ``(z̄₊ - z_0)`` from ``std_params.json`` (positives mean 0; zeros at
    ``log(0.5)`` on that axis) and ``p ≤ 1`` on the intensity path.
    """
    def pick(key: str, coef: str) -> float:
        vals = np.asarray(params[key], dtype=float).ravel()
        return float(vals[_names_for(key, params).index(coef)])

    def pick_opt(key: str, coef: str) -> float:
        names = _names_for(key, params)
        if coef not in names:
            return 0.0
        return float(np.asarray(params[key], dtype=float).ravel()[names.index(coef)])

    a1 = pick("theta_penalized_Ew", "a1")
    a2 = pick("theta_penalized_Ew", "a2_PV_lag_week")
    a3 = pick("theta_penalized_Ew", "a3_FW_lag_week")
    a4 = pick("theta_penalized_Ew", "a4_PJ_lag_week")

    alpha1 = pick("theta_penalized_PV", "alpha1_Ew")
    alpha4 = pick("theta_penalized_PV", "alpha4_action_by_Ew")
    gamma1 = pick_opt("theta_penalized_PV", "gamma1_Ew")
    gamma4 = pick_opt("theta_penalized_PV", "gamma4_action_by_Ew")
    beta1 = pick("theta_penalized_FW", "beta1_Ew")
    beta4 = pick("theta_penalized_FW", "beta4_A0_morning_by_Ew")
    beta6 = pick("theta_penalized_FW", "beta6_A1_afternoon_by_Ew")
    theta1 = pick("theta_penalized_PJ", "theta1_Ew")
    theta4 = pick("theta_penalized_PJ", "theta4_A0_morning_by_Ew")
    theta6 = pick("theta_penalized_PJ", "theta6_A1_afternoon_by_Ew")
    # J_w query block adds qE·J to each mediator's E-slope in J=1 weeks.
    alpha_qE = pick_opt("theta_penalized_PV", "query_Jw_Ew")
    gamma_qE = pick_opt("theta_penalized_PV", "intensity_query_Jw_Ew")
    beta_qE = pick_opt("theta_penalized_FW", "query_Jw_Ew")
    theta_qE = pick_opt("theta_penalized_PJ", "query_Jw_Ew")

    slope = 0.25
    if std is None:
        std_path = Path(params_dir) if params_dir is not None else PARAMS_DIR
        with open(std_path / "std_params.json", encoding="utf-8") as f:
            std = json.load(f)
    occ_gap = float(pv_hurdle_occurrence_z_gap(std))

    def g(action: float, j: float) -> float:
        return (
            a1
            + a2 * (
                slope * occ_gap * (alpha1 + action * alpha4 + j * alpha_qE)
                + (gamma1 + action * gamma4 + j * gamma_qE)
            )
            + a3 * slope * (beta1 + action * (beta4 + beta6) + j * beta_qE)
            + a4 * slope * (theta1 + action * (theta4 + theta6) + j * theta_qE)
        )

    def worst_over_j(action: float) -> float:
        g0, g1 = g(action, 0.0), g(action, 1.0)
        return g0 if abs(g0) >= abs(g1) else g1

    return {
        "g_zero": float(worst_over_j(0.0)),
        "g_always": float(worst_over_j(1.0)),
    }


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
    with open(params_dir / "std_params.json", encoding="utf-8") as f:
        std = json.load(f)
    rows = []
    for uid in user_ids:
        with open(params_dir / f"params_env_{uid}.json", encoding="utf-8") as f:
            params = json.load(f)
        rows.append(
            {
                "userid": int(uid),
                **loop_gain_Ew(params, std=std),
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


# Pathway sign stories the tuned variants are supposed to satisfy, checked
# per participant after writing a folder. Fitted heterogeneity means some
# violations are expected (warn-only): the report tells you how far the
# written environment is from the clean causal story in the module docstring.
_SIGN_CHECKS: tuple[tuple[str, str, str, int], ...] = (
    # (label, theta block, coefficient, required sign: -1 => <= 0, +1 => >= 0)
    ("A->PV main <= 0", "theta_penalized_PV", "alpha3_action", -1),
    ("A->PV intensity main <= 0", "theta_penalized_PV", "gamma3_action", -1),
    ("A->FW main <= 0", "theta_penalized_FW", "beta3_A0_morning", -1),
    ("A->FW main <= 0", "theta_penalized_FW", "beta5_A1_afternoon", -1),
    ("A->PJ main <= 0", "theta_penalized_PJ", "theta3_A0_morning", -1),
    ("A->PJ main <= 0", "theta_penalized_PJ", "theta5_A1_afternoon", -1),
    ("PV->E >= 0", "theta_penalized_Ew", "a2_PV_lag_week", +1),
    ("FW->E >= 0", "theta_penalized_Ew", "a3_FW_lag_week", +1),
    ("PJ->E >= 0", "theta_penalized_Ew", "a4_PJ_lag_week", +1),
    ("E->fourSC >= 0", "theta_fourSC", "perceived_utility_lastweek", +1),
    ("E->antic >= 0", "theta_antic", "perceived_utility_lastweek", +1),
    ("fourSC->Y >= 0", "theta_CAE", "fourSC_ewma", +1),
    ("antic->Y >= 0", "theta_CAE", "anticipated_affect_ewma", +1),
)

# Which sign stories each knob is responsible for. Checks outside the knob's
# story are still reported (as "inherited") so stacked folders show the full
# picture.
_KNOB_SIGN_STORY: dict[str, tuple[str, ...]] = {
    "fatigue": (
        "A->PV main <= 0", "A->PV intensity main <= 0",
        "A->FW main <= 0", "A->PJ main <= 0",
        "PV->E >= 0", "FW->E >= 0", "PJ->E >= 0",
        "E->fourSC >= 0", "E->antic >= 0",
    ),
    "burden_path": (
        "A->PV main <= 0", "A->PV intensity main <= 0",
        "A->FW main <= 0", "A->PJ main <= 0",
        "PV->E >= 0", "FW->E >= 0", "PJ->E >= 0",
        "E->fourSC >= 0", "E->antic >= 0",
    ),
    "benefit_foursc": ("fourSC->Y >= 0", "antic->Y >= 0"),
}


def sign_story_report(
    params_dir: Path,
    user_ids: Sequence[int],
    *,
    knob_name: str | None = None,
    log=print,
) -> dict[str, list[int]]:
    """Per-participant check of the intended pathway signs; returns violators.

    Warn-only: violations are fitted heterogeneity, not errors, but a large
    count means the written folder does not implement the causal story the
    variant claims (e.g. fatigue with κ smaller than most users' positive
    A→ME effects).
    """
    story = set(_KNOB_SIGN_STORY.get(knob_name or "", ()))
    violators: dict[str, list[int]] = {}
    for uid in user_ids:
        with open(params_dir / f"params_env_{uid}.json", encoding="utf-8") as f:
            params = json.load(f)
        for label, key, coef, sign in _SIGN_CHECKS:
            names = _names_for(key, params)
            if coef not in names:
                continue
            if knob_name == "fatigue" and label in _A_ME_SIGN_TO_MEDIATOR:
                med = _A_ME_SIGN_TO_MEDIATOR[label]
                if _coef(params, *_ME_TO_E[med]) <= _fatigue_me_to_e_min():
                    continue
            val = float(np.asarray(params[key], dtype=float).ravel()[names.index(coef)])
            if (sign < 0 and val > 0.0) or (sign > 0 and val < 0.0):
                violators.setdefault(label, []).append(int(uid))
    n = len(list(user_ids))
    log(f"  sign story ({params_dir.name}, n={n} participants):")
    seen = set()
    for label, _key, _cname, _sign in _SIGN_CHECKS:
        if label in seen:
            continue
        seen.add(label)
        bad = sorted(set(violators.get(label, [])))
        tag = "" if not story or label in story else "  (inherited)"
        status = "OK all" if not bad else f"violated by {len(bad)}: {bad}"
        log(f"    {label:>18}: {status}{tag}")
    return violators


def loop_gains_for(
    src_dir: Path,
    user_ids: Sequence[int],
    multipliers: dict[str, float] | None,
    *,
    apply: str | dict[str, str] = "multiply",
    multipliers_for: Callable[[dict], dict[str, float]] | None = None,
) -> list[dict]:
    """E_w and CAE loop gains after applying ``multipliers``, without writing."""
    with open(src_dir / "std_params.json", encoding="utf-8") as f:
        std = json.load(f)
    rows = []
    for uid in user_ids:
        with open(src_dir / f"params_env_{uid}.json", encoding="utf-8") as f:
            params = json.load(f)
        mult = multipliers_for(params) if multipliers_for is not None else multipliers
        scaled, _ = (
            scale_params(params, mult, apply=apply) if mult else (params, [])
        )
        rows.append(
            {
                "userid": int(uid),
                **loop_gain_Ew(scaled, std=std),
                **loop_gain_CAE(scaled),
            }
        )
    return rows


def loop_gains_for_knob(
    src_dir: Path,
    user_ids: Sequence[int],
    knob: KnobSpec,
    kappa: float,
) -> list[dict]:
    return loop_gains_for(
        src_dir,
        user_ids,
        knob.build(float(kappa)),
        apply=knob.apply,
        multipliers_for=_knob_multipliers_fn(knob, kappa),
    )


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
            loop_gains_for_knob(src_dir, user_ids, knob, float(k)),
            limit=limit,
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
    """Fail before any rollouts if the DiscreteCQL arm cannot be served."""
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
    wrong_algo = []
    wrong_dim = []
    for uid in user_ids:
        meta_path = _model_metadata_path(model_dir / f"user{uid}_model.d3")
        if not meta_path.is_file():
            raise SystemExit(f"{meta_path} missing; retrain the STE model.")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("algo") != STE_ALGO:
            wrong_algo.append((int(uid), meta.get("algo")))
        if meta.get("state_dim") != STE_OBS_DIM:
            wrong_dim.append((int(uid), meta.get("state_dim")))
    if wrong_algo:
        raise SystemExit(
            f"{model_dir} checkpoints are not {STE_ALGO} "
            f"(found {wrong_algo[:5]}{'...' if len(wrong_algo) > 5 else ''}). "
            "Retrain with the current ste_vanilla.py DiscreteCQL trainer, or "
            "drop the DiscreteCQL arm (TUNE_DQN_EXP='' / omit --dqn-exp) and "
            "calibrate on the Bernoulli policy grid only."
        )
    if wrong_dim:
        raise SystemExit(
            f"{model_dir} observation dim is not the frozen STE map "
            f"(trained state_dim={wrong_dim[0][1]}, STE_OBS_DIM={STE_OBS_DIM}). "
            "Point TUNE_DQN_EXP at DiscreteCQL trained on this 20-d vector "
            "(vanilla CQL is exp 5), or set TUNE_DQN_EXP=''."
        )
    print(f"DiscreteCQL arm: {len(user_ids)} checkpoints from {model_dir}")


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
        print(f"  DiscreteCQL arm won for {won.count('dqn')} of {len(won)} participants")


def _sigma_by_uid(result: dict) -> dict[int, float]:
    return {int(r["userid"]): float(r["sigma"]) for r in result.get("users", [])}


def _sigma_ratio_report(num: dict, den: dict, *, log=print, label: str) -> dict:
    """Per-user and mean σ_i(num) / σ_i(den). Denominator-driven STE shows up here."""
    nmap = _sigma_by_uid(num)
    dmap = _sigma_by_uid(den)
    users = []
    for uid in sorted(set(nmap) & set(dmap)):
        d = dmap[uid]
        ratio = float(nmap[uid] / d) if d > 0 else float("nan")
        users.append({
            "userid": uid,
            "sigma": nmap[uid],
            "sigma_ref": d,
            "ratio": ratio,
        })
    ratios = [u["ratio"] for u in users if np.isfinite(u["ratio"])]
    mean_r = float(np.mean(ratios)) if ratios else float("nan")
    log(f"  {label}: mean σ_i ratio = {mean_r:.3f}  (n={len(ratios)})")
    return {"label": label, "mean": mean_r, "n_users": len(ratios), "users": users}


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
    _log_fatigue_e_shift(knob)
    fatigue_gates = (
        _log_fatigue_gates(base_dir, user_ids) if knob.name == "fatigue" else None
    )
    zero_cache = None
    rows = []
    for kappa in args.kappas:
        write_knob_params(base_dir, scratch, user_ids, knob, float(kappa))
        result = evaluate(
            scratch,
            user_ids,
            spec,
            zero_cache=zero_cache,
            n_jobs=args.jobs,
            proxy_to_true=args.proxy_to_true,
        )
        if zero_cache is None and knob_sigma_invariant(knob):
            # Action-gated A→ME (and a κ-invariant E→MY addend) leave the
            # control arm untouched across the ladder: measure once.
            zero_cache = zero_cache_from(result)
        rows.append((float(kappa), result))
        print(
            f"  kappa={kappa:8.4f}  mean STE={result['mean_ste']:7.4f}"
            f"  median={result['median_ste']:7.4f}"
            f"  range=[{result['min_ste']:.3f}, {result['max_ste']:.3f}]"
        )
    _cleanup_scratch(scratch, keep=getattr(args, "keep_scratch", False))
    if args.report:
        write_report(
            Path(args.report),
            {
                "knob": knob.name,
                "spec": asdict(spec),
                **_fatigue_report_fields(knob),
                **({"fatigue_gates": fatigue_gates} if fatigue_gates is not None else {}),
                "scan": [
                    {"kappa": k, **{m: r[m] for m in ("mean_ste", "median_ste", "min_ste", "max_ste")}}
                    for k, r in rows
                ],
            },
        )


def calibrated_out_dir(prefix: str, target: float, suffix: str = "") -> Path:
    """``env_para_ste0.2`` or ``env_para_ste0.2_burden`` depending on suffix."""
    return PROJECT_ROOT / f"{prefix}{float(target):g}{suffix}"


def _cleanup_scratch(scratch: Path, *, keep: bool) -> None:
    """Remove per-kappa candidate copies unless ``--keep-scratch``.

    Every candidate is a full 31-user parameter copy plus the supporting
    files; a long calibration leaves hundreds of MB behind. Only paths whose
    basename carries the scratch prefix are ever deleted.
    """
    scratch = Path(scratch)
    if keep:
        print(f"  keeping scratch {scratch}")
        return
    if not scratch.exists():
        return
    if not scratch.name.startswith(".ste_tune_scratch"):
        print(f"  not deleting non-scratch path {scratch} (rename it .ste_tune_scratch_* to auto-clean)")
        return
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"  removed scratch {scratch}")


def _knob_stack(src_dir: Path, knob: KnobSpec, kappa: float) -> list[dict]:
    """Record this apply/calibrate step on top of any earlier knobs in ``src_dir``."""
    stack: list[dict] = []
    prev_path = Path(src_dir) / "ste_tuning.json"
    if prev_path.is_file():
        try:
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prev = {}
        if isinstance(prev.get("stack"), list) and prev["stack"]:
            stack.extend(prev["stack"])
        elif prev.get("knob") is not None:
            stack.append(
                {
                    "knob": prev.get("knob"),
                    "kappa": prev.get("kappa"),
                    "apply": prev.get("apply"),
                }
            )
    stack.append(
        {"knob": knob.name, "kappa": float(kappa), "apply": knob.apply}
    )
    return stack


def refuse_knob_overwrite(out_dir: Path, knob_name: str, *, overwrite: bool) -> None:
    """Block clobbering a folder that was calibrated with a different knob."""
    report_path = out_dir / "ste_tuning.json"
    if not report_path.is_file() or overwrite:
        return
    prev = json.loads(report_path.read_text(encoding="utf-8"))
    prev_knob = prev.get("knob")
    if prev_knob and prev_knob != knob_name:
        raise SystemExit(
            f"Refuse to overwrite {out_dir} (existing knob={prev_knob!r}, "
            f"this run knob={knob_name!r}). Pass --out-suffix _burden "
            f"(or another name) so the original action-knob folders stay put, "
            f"or --overwrite if you really mean it."
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
    print(f"scratch: {scratch}", flush=True)

    print(f"knob '{knob.name}': {knob.doc}")
    _log_fatigue_e_shift(knob)
    _log_burden_path(knob)
    fatigue_gates = (
        _log_fatigue_gates(base_dir, user_ids) if knob.name == "fatigue" else None
    )
    arms = [f"Bernoulli p in {spec.policy_grid}"]
    if spec.dqn_model_dir:
        arms.append(f"DiscreteCQL from {Path(spec.dqn_model_dir).name}")
    print(f"proxy: same-sample max over [{', '.join(arms)}] vs never-suggest, "
          f"truncated at 0; {spec.episodes} paired episodes/arm, "
          f"{len(user_ids)} participants")
    if args.proxy_to_true != 1.0:
        print(f"proxy-to-true shrinkage: {args.proxy_to_true:g}")

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
        stab = largest_stable_kappa(base_dir, user_ids, knob, x_hi=256.0)
        print(
            f"joint E_w/CAE stability cap: kappa <= {stab:.5f}  "
            f"(|g_zero|, |g_always| < {GAIN_LIMIT:g})"
        )
        if stab <= 0.0:
            raise SystemExit("no non-negative kappa is loop-stable; aborting.")
        kappa_hi = stab
    cap = knob_kappa_cap(knob)
    if cap is not None:
        print(f"knob kappa cap ({knob.name}): kappa <= {cap:g}")
        kappa_hi = min(kappa_hi, float(cap))
        if kappa_hi <= 0.0:
            raise SystemExit("no non-negative kappa is feasible; aborting.")

    zero_cache: dict[int, list[float]] | None = None
    last: dict[float, dict] = {}
    source_eval: dict | None = None

    def f(kappa: float) -> float:
        nonlocal zero_cache
        if float(kappa) in last:  # the response curve is shared across targets
            return last[float(kappa)]["mean_ste"]
        kappa_dir = scratch / f"k{float(kappa):.8g}"
        write_knob_params(base_dir, kappa_dir, user_ids, knob, float(kappa))
        result = evaluate(
            kappa_dir,
            user_ids,
            spec,
            zero_cache=zero_cache,
            n_jobs=args.jobs,
            proxy_to_true=args.proxy_to_true,
        )
        if zero_cache is None and knob_sigma_invariant(knob):
            zero_cache = zero_cache_from(result)
        last[float(kappa)] = result
        return result["mean_ste"]

    x0 = float(args.kappa0)
    x0 = min(max(x0, 0.0), kappa_hi)
    if x0 == 0.0 and kappa_hi > 0.0:
        x0 = min(1.0, kappa_hi)
    # One evaluation at the search hi so the solver knows the feasible STE range.
    if kappa_hi < 256.0:
        print(f"  search hi kappa={kappa_hi:.5f}  ->  mean STE={f(kappa_hi):.4f}")

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
        gains = loop_gains_for_knob(base_dir, user_ids, knob, kappa)
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

        out_dir = calibrated_out_dir(args.out_prefix, target, args.out_suffix)
        refuse_knob_overwrite(out_dir, knob.name, overwrite=args.overwrite)
        audit = write_knob_params(base_dir, out_dir, user_ids, knob, kappa)
        written = diagnose(out_dir, user_ids)
        if require_stable and unstable_users(written):
            raise SystemExit(
                f"wrote {out_dir} but diagnose reports it unstable -- this is a bug."
            )
        print(f"  wrote {out_dir}")
        sign_violations = sign_story_report(out_dir, user_ids, knob_name=knob.name)

        if source_eval is None:
            print("  evaluating source-folder never-send σ_i (for σ ratios)")
            source_eval = evaluate(
                base_dir, user_ids, spec,
                n_jobs=args.jobs, proxy_to_true=args.proxy_to_true,
            )
        sigma_ratios = {
            "vs_source": _sigma_ratio_report(
                result, source_eval,
                label=f"σ_i(κ={kappa:g}) / σ_i(source {base_dir.name})",
            )
        }
        if 0.0 not in last:
            f(0.0)
        if 0.0 in last:
            sigma_ratios["vs_knob0"] = _sigma_ratio_report(
                result, last[0.0],
                label=f"σ_i(κ={kappa:g}) / σ_i(κ=0)",
            )

        report = {
            "target_mean_ste": float(target),
            "achieved_mean_ste": float(achieved),
            "knob": knob.name,
            "kappa": float(kappa),
            "kappa_hi": float(kappa_hi),
            "stable": True,
            "apply": knob.apply,
            "multipliers": knob.build(kappa),
            "search_path": [{"kappa": x, "mean_ste": y} for x, y in hist],
            "spec": asdict(spec),
            "source_params_dir": str(base_dir),
            "loop_gains": written,
            "n_coefficients_scaled": len(audit),
            "stack": _knob_stack(base_dir, knob, kappa),
            "sign_violations": {k: sorted(v) for k, v in sign_violations.items()},
            "sigma_ratios": sigma_ratios,
            **_fatigue_report_fields(knob, kappa),
            **({"fatigue_gates": fatigue_gates} if fatigue_gates is not None else {}),
            **result,
        }
        report["params_dir"] = str(out_dir)
        write_report(out_dir / "ste_tuning.json", report)
        solutions.append((float(target), float(kappa), float(achieved), out_dir))

    _cleanup_scratch(scratch, keep=getattr(args, "keep_scratch", False))

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
        "\nNext: run_tune_ste.sh submits a full DiscreteCQL train+eval in each folder "
        "(TUNE_VALIDATE=1, the default). To do it by hand:\n"
        f"    ADAPR_PARAMS_DIR={final} python ste_vanilla.py train ... / eval ... / aggregate ..."
    )


def cmd_apply(args) -> None:
    """Write a parameter folder at a fixed kappa, with no STE root search."""
    base_dir = Path(args.params_dir).expanduser().resolve()
    user_ids = load_user_ids(base_dir)
    require_fitted_blocks(base_dir, user_ids)
    knob = KNOBS[args.knob]
    kappa = float(args.kappa)
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    require_stable = bool(args.require_stable)

    print(f"knob '{knob.name}': {knob.doc}")
    _log_fatigue_e_shift(knob)
    _log_burden_path(knob)
    fatigue_gates = (
        _log_fatigue_gates(base_dir, user_ids) if knob.name == "fatigue" else None
    )
    cap = knob_kappa_cap(knob)
    if cap is not None and kappa > cap:
        print(f"WARNING: kappa={kappa:g} exceeds {knob.name} cap {cap:g}")
    print(f"apply kappa={kappa:g}  {base_dir.name} -> {out_dir.name}")

    gains = loop_gains_for_knob(base_dir, user_ids, knob, kappa)
    bad = unstable_users(gains)
    if require_stable and bad:
        uids = [int(r["userid"]) for r in bad]
        raise SystemExit(
            f"kappa={kappa:g} is loop-unstable for {len(uids)} participants "
            f"{uids} (|g| >= {GAIN_LIMIT:g})."
        )

    refuse_knob_overwrite(out_dir, knob.name, overwrite=args.overwrite)
    audit = write_knob_params(base_dir, out_dir, user_ids, knob, kappa)
    written = diagnose(out_dir, user_ids)
    if require_stable and unstable_users(written):
        raise SystemExit(
            f"wrote {out_dir} but diagnose reports it unstable -- this is a bug."
        )
    print(f"  wrote {out_dir}  ({len(audit)} coefficients)")
    sign_violations = sign_story_report(out_dir, user_ids, knob_name=knob.name)

    report = {
        "target_mean_ste": None,
        "achieved_mean_ste": None,
        "knob": knob.name,
        "kappa": float(kappa),
        "kappa_hi": None,
        "stable": not bool(unstable_users(written)),
        "apply": knob.apply,
        "multipliers": knob.build(kappa),
        "source_params_dir": str(base_dir),
        "loop_gains": written,
        "n_coefficients_scaled": len(audit),
        "stack": _knob_stack(base_dir, knob, kappa),
        "sign_violations": {k: sorted(v) for k, v in sign_violations.items()},
        "params_dir": str(out_dir),
        **_fatigue_report_fields(knob, kappa),
        **({"fatigue_gates": fatigue_gates} if fatigue_gates is not None else {}),
    }

    if args.eval:
        spec = make_spec(args)
        require_dqn_checkpoints(spec, user_ids)
        t0 = time.perf_counter()
        result = evaluate(
            out_dir, user_ids, spec, n_jobs=args.jobs, proxy_to_true=args.proxy_to_true
        )
        print(f"  proxy STE ({time.perf_counter() - t0:.1f}s)")
        summarise(result)
        report["spec"] = asdict(spec)
        report["achieved_mean_ste"] = float(result["mean_ste"])
        for key in (
            "mean_ste", "median_ste", "min_ste", "max_ste",
            "n_users", "n_users_total", "n_users_nan", "users", "proxy_to_true",
        ):
            if key in result:
                report[key] = result[key]

    write_report(out_dir / "ste_tuning.json", report)


def _stage_args(args, **overrides) -> argparse.Namespace:
    """Clone the parsed protocol args into a calibrate/apply-shaped namespace."""
    base = dict(
        params_dir=args.params_dir,
        episodes=args.episodes,
        seed=args.seed,
        noise=args.noise,
        policy_grid=list(args.policy_grid),
        jobs=args.jobs,
        proxy_to_true=args.proxy_to_true,
        report=None,
        dqn_exp=getattr(args, "dqn_exp", None),
        dqn_model_dir=getattr(args, "dqn_model_dir", None),
        scratch=args.scratch,
        out_prefix=args.out_prefix,
        out_suffix=args.out_suffix,
        overwrite=args.overwrite,
        tol=args.tol,
        max_iter=args.max_iter,
        require_stable=args.require_stable,
        keep_scratch=getattr(args, "keep_scratch", False),
        eval=False,
        kappa=None,
        out_dir=None,
        knob=None,
        targets=None,
        kappa0=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _tuned_folder_stable(out_dir: Path) -> bool:
    report = out_dir / "ste_tuning.json"
    if not report.is_file():
        return False
    try:
        return json.loads(report.read_text(encoding="utf-8")).get("stable") is True
    except (OSError, json.JSONDecodeError):
        return False


def _report_stamp(out_dir: Path) -> tuple[int, int] | None:
    """Identity of ``ste_tuning.json`` before a calibrate stage (mtime, size)."""
    report = out_dir / "ste_tuning.json"
    if not report.is_file():
        return None
    st = report.stat()
    return (int(st.st_mtime_ns), int(st.st_size))


def _written_this_run(out_dir: Path, before: tuple[int, int] | None) -> bool:
    """True only if this run wrote a stable ``ste_tuning.json`` at ``out_dir``.

    A leftover folder from an older protocol / knob / cohort can have
    ``stable: true``. Using that to skip the 0.2 fallback or to submit
    confirmation CQL would keep the stale environment.
    """
    if not _tuned_folder_stable(out_dir):
        return False
    after = _report_stamp(out_dir)
    return after is not None and after != before


def _calibrate_status(stage_args) -> int:
    """Run ``cmd_calibrate`` and return its exit code (0 / 1 / 2)."""
    try:
        cmd_calibrate(stage_args)
        return 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return 0 if code is None else int(code)


def cmd_protocol(args) -> None:
    """Root-find STE 0.2 / 0.5 / 0.8 from vanilla (no stacked fatigue folder).

    STE 0.2: ``burden_path`` first (A→ME, ME→E, E→MY together). If that
    search misses or is unstable, ``benefit_foursc`` on the same vanilla
    source. STE 0.5 / 0.8: ``benefit_foursc`` on vanilla. Variant 1 is
    the untouched vanilla folder.
    """
    scratch = Path(args.scratch)
    ste02 = float(args.ste02_target)
    raise_targets = [float(t) for t in args.benefit_targets]
    dest_02 = calibrated_out_dir(args.out_prefix, ste02, args.out_suffix)
    dest_hi = [
        calibrated_out_dir(args.out_prefix, t, args.out_suffix)
        for t in raise_targets
    ]
    stamp_02 = _report_stamp(dest_02)
    stamps_hi = [_report_stamp(d) for d in dest_hi]

    print("=" * 70)
    print(
        f"PROTOCOL STE {ste02:g}: burden_path on {Path(args.params_dir).name} "
        f"-> {dest_02.name}"
    )
    print("=" * 70)
    _calibrate_status(_stage_args(
        args,
        knob="burden_path",
        targets=[ste02],
        kappa0=float(args.burden_kappa0),
        scratch=str(scratch) + "_burden_path",
    ))

    if _written_this_run(dest_02, stamp_02):
        print(f"\nprotocol: {dest_02.name} written by burden_path; skip benefit fallback.")
    else:
        if _tuned_folder_stable(dest_02) and not _written_this_run(dest_02, stamp_02):
            print(
                f"\nprotocol: {dest_02.name} is leftover (stable=true but not "
                "rewritten this run); trying benefit_foursc fallback."
            )
        print("\n" + "=" * 70)
        print(
            f"PROTOCOL STE {ste02:g}: burden_path missed; "
            f"fallback benefit_foursc on {Path(args.params_dir).name} "
            f"-> {dest_02.name}"
        )
        print("=" * 70)
        _calibrate_status(_stage_args(
            args,
            knob="benefit_foursc",
            targets=[ste02],
            kappa0=float(args.ste02_benefit_kappa0),
            scratch=str(scratch) + "_ste02_benefit",
        ))

    print("\n" + "=" * 70)
    print(
        f"PROTOCOL STE {' '.join(f'{t:g}' for t in raise_targets)}: "
        f"benefit_foursc on {Path(args.params_dir).name}"
    )
    print("=" * 70)
    _calibrate_status(_stage_args(
        args,
        knob="benefit_foursc",
        targets=raise_targets,
        kappa0=float(args.benefit_kappa0),
        scratch=str(scratch) + "_benefit",
    ))

    written_dirs = []
    if _written_this_run(dest_02, stamp_02):
        written_dirs.append(dest_02)
    for d, before in zip(dest_hi, stamps_hi):
        if _written_this_run(d, before):
            written_dirs.append(d)
    written = [d.name for d in written_dirs]
    wanted = [dest_02.name] + [d.name for d in dest_hi]
    written_list = Path(str(scratch) + "_written")
    written_list.write_text("".join(f"{d}\n" for d in written_dirs), encoding="utf-8")

    print("\nprotocol written: " + (", ".join(written) if written else "(none)"))
    if written == wanted:
        print("protocol complete: vanilla + " + ", ".join(written))
        return
    print(
        "protocol incomplete; confirmation CQL will pick up folders "
        "rewritten this run (not leftover ste_tuning.json)."
    )
    raise SystemExit(2 if written else 1)


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
        sp.add_argument(
            "--proxy-to-true",
            type=float,
            default=1.0,
            help=(
                "Multiply each user's proxy STE after max/truncation "
                "(default 1.0: report the proxy as-is). Not estimated from "
                "DiscreteCQL; set only with an external shrinkage factor."
            ),
        )
        sp.add_argument("--report", default=None)
        sp.add_argument(
            "--dqn-exp",
            default=None,
            help=(
                "Add each participant's trained DiscreteCQL policy as a "
                "treatment arm, reading d3rlpy_logs/ste_exp_<EXP>/"
                "user<uid>_model.d3 (the --exp value used with "
                "ste_vanilla.py train)"
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
                default=str(
                    PROJECT_ROOT / f".ste_tune_scratch_{os.getpid()}_{time.time_ns()}"
                ),
                help="Working directory for candidate parameter sets",
            )
            sp.add_argument(
                "--keep-scratch",
                action="store_true",
                help="Keep per-kappa candidate folders (default: delete when done)",
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
    sp.add_argument(
        "--out-suffix",
        default="",
        help=(
            "Appended after the target, e.g. --out-suffix _burden writes "
            "env_para_ste0.2_burden instead of overwriting env_para_ste0.2."
        ),
    )
    sp.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacing a folder whose ste_tuning.json was written with "
            "a different knob. Default: refuse."
        ),
    )
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
        "protocol",
        help=(
            "STE 0.2 via burden_path (benefit_foursc fallback), then "
            "benefit_foursc to 0.5/0.8; all from vanilla"
        ),
    )
    common(sp, needs_knob=True)  # --knob is accepted but ignored (fixed per stage)
    sp.add_argument(
        "--ste02-target",
        type=float,
        default=0.2,
        help="Low STE target tried first with burden_path (default 0.2)",
    )
    sp.add_argument(
        "--burden-kappa0",
        type=float,
        default=0.3,
        help="Search start for burden_path on the 0.2 target (default 0.3)",
    )
    sp.add_argument(
        "--ste02-benefit-kappa0",
        type=float,
        default=1.0,
        help="Search start for benefit_foursc if burden_path misses 0.2",
    )
    sp.add_argument(
        "--benefit-targets",
        type=float,
        nargs="+",
        default=[0.5, 0.8],
        help="High STE targets via benefit_foursc on vanilla (default 0.5 0.8)",
    )
    sp.add_argument("--benefit-kappa0", type=float, default=2.0)
    sp.add_argument("--out-prefix", default="env_para_ste")
    sp.add_argument("--out-suffix", default="")
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--tol", type=float, default=0.02)
    sp.add_argument("--max-iter", type=int, default=10)
    sp.add_argument(
        "--require-stable",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    sp.set_defaults(func=cmd_protocol)

    sp = sub.add_parser(
        "apply",
        help="Write a parameter folder at a fixed kappa (no STE search)",
    )
    common(sp, needs_knob=True)
    sp.add_argument(
        "--kappa",
        type=float,
        default=BURDEN_SHIFT_LARGE_KAPPA,
        help=(
            f"Knob value to apply (default {BURDEN_SHIFT_LARGE_KAPPA:g}, the "
            "large burden_shift shift)"
        ),
    )
    sp.add_argument(
        "--out-dir",
        default="env_para_burden_shift_large",
        help="Destination parameter folder (repo-relative unless absolute)",
    )
    sp.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing a folder whose ste_tuning.json used a different knob",
    )
    sp.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run the proxy STE Monte-Carlo after writing (default: off)",
    )
    sp.add_argument(
        "--require-stable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refuse to write if any participant's |loop gain| >= 1 (default: on)",
    )
    sp.set_defaults(func=cmd_apply)

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
