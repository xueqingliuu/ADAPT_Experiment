from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.param_fingerprint import (
    fingerprint_arrays,
    params_fingerprint,
    require_matching_fingerprint,
)
from tests.make_golden_dropout import TRAJECTORY_NAMES, build_golden, write_golden


def _tiny_params(tmp_path: Path, uids=(13,)) -> Path:
    dest = tmp_path / "params"
    dest.mkdir()
    (dest / "std_params.json").write_text("{}", encoding="utf-8")
    (dest / "generator_calibration.json").write_text("{}", encoding="utf-8")
    (dest / "Ew_pooled_linear_coefs.json").write_text("{}", encoding="utf-8")
    for uid in uids:
        (dest / f"params_env_{uid}.json").write_text(
            json.dumps({"uid": uid}), encoding="utf-8"
        )
    return dest


def test_fingerprint_changes_when_a_hashed_file_changes(tmp_path):
    dest = _tiny_params(tmp_path)
    first, names, _ = params_fingerprint(dest, [13])
    assert "params_env_13.json" in names
    assert "std_params.json" in names
    (dest / "std_params.json").write_text('{"x": 1}', encoding="utf-8")
    second, _, _ = params_fingerprint(dest, [13])
    assert first != second


def test_require_matching_fingerprint_skips_legacy_npz_without_hash(tmp_path):
    dest = _tiny_params(tmp_path)
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, seed=1, uids=np.array([13]))
    gold = np.load(path)
    with pytest.raises(pytest.skip.Exception, match="regenerate with tests/make_golden_dropout.py"):
        require_matching_fingerprint(gold, dest, uids=[13])


def test_require_matching_fingerprint_skips_on_mismatch(tmp_path):
    dest = _tiny_params(tmp_path)
    arrays = fingerprint_arrays(dest, [13])
    path = tmp_path / "gold.npz"
    np.savez_compressed(path, seed=1, uids=np.array([13]), **arrays)
    (dest / "params_env_13.json").write_text('{"uid": 13, "moved": true}', encoding="utf-8")
    gold = np.load(path)
    with pytest.raises(pytest.skip.Exception, match="regenerate with tests/make_golden_dropout.py"):
        require_matching_fingerprint(gold, dest, uids=[13])


def test_require_matching_fingerprint_passes_when_unchanged(tmp_path):
    dest = _tiny_params(tmp_path)
    arrays = fingerprint_arrays(dest, [13])
    path = tmp_path / "gold.npz"
    np.savez_compressed(path, seed=1, uids=np.array([13]), **arrays)
    gold = np.load(path)
    require_matching_fingerprint(gold, dest, uids=[13])


def test_make_golden_writes_fingerprint_and_roundtrips(tmp_path, monkeypatch):
    """Regenerator stores the fingerprint; the test helper accepts it."""
    dest = _tiny_params(tmp_path, uids=(13, 22))

    class _Oenv:
        dropout_day = -1

        def __init__(self):
            for name in TRAJECTORY_NAMES:
                setattr(self, name, np.arange(4, dtype=float))

    def fake_run(uid, seed=0, params_dir=None, dropout_hazard_day=None):
        return {}, _Oenv()

    monkeypatch.setattr("experiment.run_random_send", fake_run)
    payload = build_golden(dest, uids=(13, 22), seed=7)
    out = tmp_path / "gold.npz"
    write_golden(out, payload)
    gold = np.load(out)
    assert "param_fingerprint" in gold.files
    require_matching_fingerprint(gold, dest, uids=[13, 22])
    assert int(gold["seed"]) == 7
    np.testing.assert_array_equal(gold["uids"], [13, 22])
    np.testing.assert_array_equal(gold["uid13_CAE_all"], np.arange(4, dtype=float))
