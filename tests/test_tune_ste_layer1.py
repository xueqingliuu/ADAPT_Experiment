from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from testbed import STE_LAYER1_DIRS


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "env_para_vanilla"


def test_two_dial_kappa1_is_noop():
    from tune_ste import PATHWAYS, two_dial_multipliers, scale_params

    params = json.loads((PARAMS / "params_env_13.json").read_text(encoding="utf-8"))
    scaled, audit = scale_params(params, two_dial_multipliers(1.0, 1.0))
    assert audit == []
    for pathway in ("A_to_MY", "A_to_ME"):
        for key in PATHWAYS[pathway]:
            np.testing.assert_array_equal(
                np.asarray(params[key], dtype=float),
                np.asarray(scaled[key], dtype=float),
            )
    np.testing.assert_array_equal(
        np.asarray(params["theta_CAE"], dtype=float),
        np.asarray(scaled["theta_CAE"], dtype=float),
    )


def test_benefit_only_kappa_b1_zeros_burden_actions():
    from tune_ste import PATHWAYS, _names_for, two_dial_multipliers, scale_params

    params = json.loads((PARAMS / "params_env_13.json").read_text(encoding="utf-8"))
    scaled, audit = scale_params(params, two_dial_multipliers(1.0, 0.0))
    assert audit
    for key, coefs in PATHWAYS["A_to_MY"].items():
        names = _names_for(key, params)
        src = np.asarray(params[key], dtype=float).ravel()
        got = np.asarray(scaled[key], dtype=float).ravel()
        for coef in coefs:
            idx = names.index(coef)
            assert got[idx] == pytest.approx(src[idx])
    for key, coefs in PATHWAYS["A_to_ME"].items():
        names = _names_for(key, params)
        got = np.asarray(scaled[key], dtype=float).ravel()
        for coef in coefs:
            idx = names.index(coef)
            assert got[idx] == pytest.approx(0.0)
    np.testing.assert_array_equal(
        np.asarray(params["theta_CAE"], dtype=float),
        np.asarray(scaled["theta_CAE"], dtype=float),
    )


def test_heavy_burden_centering_keeps_never_send_cae(tmp_path, monkeypatch):
    """Never-send mean weekly CAE (linear predictor at x̄) stays within 1e-6."""
    from tune_ste import write_two_dial_params
    from vani_env import THETA_ANTIC_NAMES, THETA_CAE_NAMES, THETA_FOURSC_NAMES, EnvConfig

    centers = {
        "E": 0.4,
        "fourSC_ewma": 0.15,
        "antic_ewma": -0.25,
        "PV_week": 0.0,
        "FW_week": 0.0,
        "PJ_week": 0.0,
    }

    def fake_centers(src_dir, user_ids, **_kw):
        return {int(u): dict(centers) for u in user_ids}

    monkeypatch.setattr("tune_ste.covariate_centers", fake_centers)
    dst = tmp_path / "env_para_ste0.2_heavyburden"
    write_two_dial_params(PARAMS, dst, [13], 1.0, 2.0, heavy=True)

    van = EnvConfig(13, params_dir=PARAMS, activity="off")
    got = EnvConfig(13, params_dir=dst, activity="off")

    i_int_sc = THETA_FOURSC_NAMES.index("intercept")
    i_e_sc = THETA_FOURSC_NAMES.index("perceived_utility_lastweek")
    i_int_aa = THETA_ANTIC_NAMES.index("intercept")
    i_e_aa = THETA_ANTIC_NAMES.index("perceived_utility_lastweek")
    e_bar = centers["E"]
    four_van = float(van.theta_fourSC[i_int_sc] + van.theta_fourSC[i_e_sc] * e_bar)
    four_got = float(got.theta_fourSC[i_int_sc] + got.theta_fourSC[i_e_sc] * e_bar)
    antic_van = float(van.theta_antic[i_int_aa] + van.theta_antic[i_e_aa] * e_bar)
    antic_got = float(got.theta_antic[i_int_aa] + got.theta_antic[i_e_aa] * e_bar)
    assert four_got == pytest.approx(four_van, abs=1e-6)
    assert antic_got == pytest.approx(antic_van, abs=1e-6)
    assert got.theta_fourSC[i_e_sc] != van.theta_fourSC[i_e_sc]

    i_int = THETA_CAE_NAMES.index("intercept")
    i_sc = THETA_CAE_NAMES.index("fourSC_ewma")
    i_aa = THETA_CAE_NAMES.index("anticipated_affect_ewma")
    x = np.zeros(len(van.theta_CAE), dtype=float)
    x[i_int] = 1.0
    x[i_sc] = centers["fourSC_ewma"]
    x[i_aa] = centers["antic_ewma"]
    assert float(got.theta_CAE @ x) == pytest.approx(float(van.theta_CAE @ x), abs=1e-6)
    np.testing.assert_array_equal(got.theta_CAE, van.theta_CAE)

    src_block = json.loads(
        (PARAMS / "generator_calibration.json").read_text(encoding="utf-8")
    )["activity_process"]
    dst_block = json.loads(
        (dst / "generator_calibration.json").read_text(encoding="utf-8")
    )["activity_process"]
    assert dst_block == src_block
    assert "stayer_variants" in dst_block


def test_tune_dry_run_lists_two_action_targets(capsys):
    from tune_ste import build_parser

    args = build_parser().parse_args(["tune", "--dry-run"])
    args.func(args)
    out = capsys.readouterr().out
    for name in STE_LAYER1_DIRS:
        assert name in out
    assert "env_para_ste0.2" in out
    assert "env_para_ste0.5" in out
    assert "env_para_ste0.8" not in out
    assert "kappa_m=kappa_b" in out
    assert "knob=action" in out
    assert "knob=benefit_only" not in out
    assert "knob=heavy_burden" not in out
    assert "benefitonly" not in out
    assert "heavyburden" not in out
    assert "activity on" in out


def test_layer1_is_tune_alias(capsys):
    from tune_ste import build_parser

    args = build_parser().parse_args(["layer1", "--dry-run"])
    args.func(args)
    out = capsys.readouterr().out
    assert "env_para_ste0.2" in out
    assert "env_para_ste0.5" in out
    assert "knob=action" in out
    assert "knob=benefit_only" not in out
