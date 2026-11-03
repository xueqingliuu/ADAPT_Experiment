from __future__ import annotations

import json
import os
import subprocess
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


def test_tune_refuses_an_existing_folder(tmp_path, monkeypatch, capsys):
    from tune_ste import build_parser, cmd_tune

    dest = tmp_path / "env_para_ste0.2"
    dest.mkdir()
    (dest / "ste_tuning.json").write_text(
        json.dumps({
            "knob": "burden_path",
            "kappa": 0.8,
            "activity": "off",
            "stayer_fraction": 0.0,
            "date": "2026-09-01T00:00:00",
            "commit": "abc123",
        }),
        encoding="utf-8",
    )

    def fake_out(prefix, target, suffix=""):
        if float(target) == 0.2:
            return dest
        return tmp_path / f"absent_{target}"

    monkeypatch.setattr("tune_ste.calibrated_out_dir", fake_out)
    args = build_parser().parse_args(["tune"])
    with pytest.raises(SystemExit):
        cmd_tune(args)
    out = capsys.readouterr().out
    assert "TUNE_OVERWRITE=1" in out
    assert "knob='burden_path'" in out
    assert "kappa=0.8" in out
    assert "activity='off'" in out
    assert "stayer_fraction=0.0" in out
    assert "2026-09-01T00:00:00" in out
    assert "abc123" in out


def _write_ids(path: Path, n: int) -> None:
    path.write_text("".join(f"{i}\n" for i in range(n)), encoding="utf-8")


def _run_ste_array_check(ids: Path, **extra: str) -> subprocess.CompletedProcess:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "SLURM_ARRAY_TASK_MIN": "0",
        "SLURM_ARRAY_TASK_MAX": "46",
        "SLURM_ARRAY_TASK_COUNT": "47",
        "SLURM_ARRAY_TASK_ID": "0",
        "STE_PHASE": "train",
        "STE_ARRAY_CHECK_ONLY": "1",
        "STE_USER_IDS": str(ids),
    }
    env.update(extra)
    return subprocess.run(
        ["bash", str(ROOT / "run_ste.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_ste_array_must_cover_user_ids(tmp_path):
    ids37 = tmp_path / "ids37.txt"
    ids47 = tmp_path / "ids47.txt"
    _write_ids(ids37, 37)
    _write_ids(ids47, 47)

    short = _run_ste_array_check(ids37)
    assert short.returncode != 0
    assert "does not match" in short.stderr
    assert "STE_ALLOW_PARTIAL=1" in short.stderr

    partial = _run_ste_array_check(ids37, STE_ALLOW_PARTIAL="1")
    assert partial.returncode == 0
    assert "STE_ALLOW_PARTIAL=1" in partial.stderr
    assert "does not match" not in partial.stderr

    full = _run_ste_array_check(ids47)
    assert full.returncode == 0, full.stderr
    assert "STE array 0-46 matches 47 lines" in full.stdout
