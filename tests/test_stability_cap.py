from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from stability import analytic_gains_from, apply_gain_cap, gain_decomposition_from
from vani_env import THETA_CAE_NAMES, _load_json, trim_theta_cae


ROOT = Path(__file__).resolve().parents[1]
VANILLA = ROOT / "env_para_vanilla"


def _load(uid: int):
    P = _load_json(VANILLA / f"params_env_{uid}.json")
    pred = _load_json(VANILLA / f"pred_{uid}.json")
    std = _load_json(VANILLA / "std_params.json")
    return P, pred, std


def _set_cae(P: dict, alpha1=None, sc_scale=1.0, aa_scale=1.0) -> dict:
    out = copy.deepcopy(P)
    cae = trim_theta_cae(np.asarray(out["theta_CAE"], dtype=float)).copy()
    if alpha1 is not None:
        cae[THETA_CAE_NAMES.index("CAE_avg_lastweek")] = float(alpha1)
    cae[THETA_CAE_NAMES.index("fourSC_ewma")] *= float(sc_scale)
    cae[THETA_CAE_NAMES.index("anticipated_affect_ewma")] *= float(aa_scale)
    out["theta_CAE"] = cae.tolist()
    return out


def test_gain_cap_brings_synthetic_gy_to_0_95():
    P, pred, std = _load(13)
    decomp = gain_decomposition_from(P, pred, std)
    alpha1 = decomp["gY"]["alpha1"]
    loop = decomp["gY"]["signed_max"] - alpha1
    bumped = _set_cae(P, alpha1=1.003 - loop)
    assert max(analytic_gains_from(bumped, pred, std)["gY"].values()) == pytest.approx(1.003, abs=1e-9)

    capped, rec = apply_gain_cap(bumped, pred, std, cap=0.95)
    assert rec is not None
    g_c = analytic_gains_from(capped, pred, std)
    assert max(g_c["gY"].values()) == pytest.approx(0.95, abs=1e-9)
    assert rec["gY"]["method"] == "reduce_ar"
    assert bumped["theta_CAE"][THETA_CAE_NAMES.index("CAE_avg_lastweek")] == pytest.approx(1.003 - loop)


def test_gain_cap_leaves_stable_participant_untouched():
    P, pred, std = _load(13)
    g = analytic_gains_from(P, pred, std)
    assert max(g["gY"].values()) < 0.95
    assert max(abs(v) for v in g["gE"].values()) < 0.95
    out, rec = apply_gain_cap(P, pred, std, cap=0.95)
    assert rec is None
    assert out is P


def test_gain_cap_zeros_ar_and_scales_loop_when_needed():
    P, pred, std = _load(13)
    decomp = gain_decomposition_from(P, pred, std)
    loop = decomp["gY"]["signed_max"] - decomp["gY"]["alpha1"]
    tiny = _set_cae(P, alpha1=0.02, sc_scale=0.99 / loop, aa_scale=0.99 / loop)
    assert max(analytic_gains_from(tiny, pred, std)["gY"].values()) == pytest.approx(1.01, abs=1e-6)

    capped, rec = apply_gain_cap(tiny, pred, std, cap=0.95)
    assert rec is not None
    assert rec["gY"]["method"] == "zero_ar_scale_loop"
    assert rec["gY"]["new_alpha1"] == 0.0
    assert max(analytic_gains_from(capped, pred, std)["gY"].values()) == pytest.approx(0.95, abs=1e-8)
