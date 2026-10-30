"""Regenerate ``tests/fixtures/dropout_hazard_zero_random_send.npz``.

Run after ``env_para_vanilla`` is the parameter set the golden should track::

    python tests/make_golden_dropout.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.param_fingerprint import fingerprint_arrays  # noqa: E402

import experiment  # noqa: E402

DEFAULT_OUT = ROOT / "tests" / "fixtures" / "dropout_hazard_zero_random_send.npz"
DEFAULT_UIDS = (13, 22)
DEFAULT_SEED = 20261017
TRAJECTORY_NAMES = (
    "CAE_all",
    "pu_all",
    "action_all",
    "stepCountNext4HourAll",
    "pageViewNext4HourAll",
    "morningFitbitWearAll",
    "dailySurveyCompleteAll",
)


def build_golden(
    params_dir,
    *,
    uids=DEFAULT_UIDS,
    seed=DEFAULT_SEED,
) -> dict:
    params_dir = Path(params_dir)
    uids = [int(u) for u in uids]
    payload = {
        "seed": np.int64(seed),
        "uids": np.asarray(uids, dtype=np.int64),
        **fingerprint_arrays(params_dir, uids),
    }
    for uid in uids:
        _res, oenv = experiment.run_random_send(
            uid, seed=int(seed), params_dir=params_dir, dropout_hazard_day=0.0
        )
        prefix = f"uid{uid}_"
        payload[prefix + "dropout_day"] = np.int64(oenv.dropout_day)
        for name in TRAJECTORY_NAMES:
            payload[prefix + name] = np.asarray(getattr(oenv, name), dtype=float)
    return payload


def write_golden(path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params-dir", default=str(ROOT / "env_para_vanilla"))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--uids", default=",".join(str(u) for u in DEFAULT_UIDS))
    args = p.parse_args(argv)
    uids = [int(u) for u in args.uids.split(",") if u.strip()]
    payload = build_golden(args.params_dir, uids=uids, seed=args.seed)
    write_golden(args.out, payload)
    print(f"wrote {args.out}  uids={uids} seed={args.seed}")
    print(f"param_fingerprint={payload['param_fingerprint']}")


if __name__ == "__main__":
    main()
