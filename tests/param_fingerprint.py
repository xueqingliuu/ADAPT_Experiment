"""Fingerprint env_para files so golden fixtures skip instead of failing a refit.

A golden run stores ``param_fingerprint`` (sha256) and ``param_fingerprint_files``
(newline-separated relative names). Tests recompute that hash from the live
parameter directory and ``pytest.skip`` when it differs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest


FINGERPRINT_KEY = "param_fingerprint"
FINGERPRINT_FILES_KEY = "param_fingerprint_files"
FINGERPRINT_MANIFEST_KEY = "param_fingerprint_manifest"

SHARED_FILES = (
    "std_params.json",
    "generator_calibration.json",
    "Ew_pooled_linear_coefs.json",
)
# Read for initial state when present (gitignored participant panel).
OPTIONAL_FILES = ("df_fit_11week.csv",)

GOLDEN_REGENERATOR = "tests/make_golden_dropout.py"
SKIP_MSG = (
    "golden generated for different parameters; "
    f"regenerate with {GOLDEN_REGENERATOR}"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file_names(params_dir, uids) -> list[str]:
    """Relative names hashed for a golden run over ``uids``."""
    params_dir = Path(params_dir)
    names = [f"params_env_{int(uid)}.json" for uid in uids]
    names.extend(SHARED_FILES)
    for name in OPTIONAL_FILES:
        if (params_dir / name).is_file():
            names.append(name)
    return names


def fingerprint_manifest(params_dir, names) -> str:
    """Stable ``relpath=sha256`` listing, one file per line."""
    params_dir = Path(params_dir)
    lines = []
    for name in names:
        path = params_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        lines.append(f"{name}={_sha256_file(path)}")
    return "\n".join(lines) + "\n"


def params_fingerprint(params_dir, uids=None, *, names=None) -> tuple[str, list[str], str]:
    """Return ``(overall_sha256, file_names, manifest)``."""
    params_dir = Path(params_dir)
    if names is None:
        if uids is None:
            raise ValueError("params_fingerprint needs uids or names")
        names = fingerprint_file_names(params_dir, uids)
    else:
        names = [str(n) for n in names]
    manifest = fingerprint_manifest(params_dir, names)
    overall = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    return overall, names, manifest


def _as_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr)


def fingerprint_arrays(params_dir, uids) -> dict:
    """Keyword arrays to pass into ``np.savez_compressed``."""
    overall, names, manifest = params_fingerprint(params_dir, uids)
    return {
        FINGERPRINT_KEY: np.asarray(overall),
        FINGERPRINT_FILES_KEY: np.asarray("\n".join(names)),
        FINGERPRINT_MANIFEST_KEY: np.asarray(manifest),
    }


def require_matching_fingerprint(
    gold,
    params_dir,
    *,
    uids=None,
    regenerator: str = GOLDEN_REGENERATOR,
) -> None:
    """Skip unless the golden npz was generated from these parameter files."""
    files = getattr(gold, "files", None)
    if files is None or FINGERPRINT_KEY not in files or FINGERPRINT_FILES_KEY not in files:
        pytest.skip(
            "golden generated for different parameters; "
            f"regenerate with {regenerator}"
        )
    stored_names = [n for n in _as_str(gold[FINGERPRINT_FILES_KEY]).splitlines() if n]
    try:
        current, _, _ = params_fingerprint(params_dir, uids=uids, names=stored_names)
    except FileNotFoundError:
        pytest.skip(
            "golden generated for different parameters; "
            f"regenerate with {regenerator}"
        )
    stored = _as_str(gold[FINGERPRINT_KEY])
    if stored != current:
        pytest.skip(
            "golden generated for different parameters; "
            f"regenerate with {regenerator}"
        )
